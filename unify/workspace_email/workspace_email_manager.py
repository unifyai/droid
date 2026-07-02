"""WorkspaceEmailManager: send and read the connected workspace mailbox.

Exposed to the actor as ``primitives.workspace_email.*``. This is the
impersonation surface for a *connected* Google Workspace / Microsoft 365
account (the user's own mailbox that was linked via OAuth), as opposed to the
assistant's platform-managed mailbox that ``primitives.comms.send_email``
targets.

The two identities are deliberately distinct:

* ``primitives.comms.send_email`` sends **as the assistant** from its managed
  ``@unify.ai`` mailbox and is wired into the contact graph, threading, and
  outbound history.
* ``primitives.workspace_email`` acts **on the connected account** — it sends
  from, and reads, the user's own linked mailbox. Use it only when the task is
  explicitly about the user's workspace inbox.

Credentials come from the connected account's OAuth token, resolved in the
trusted runtime via :func:`unify.common.runtime_oauth.get_provider_access_token`.
The token secrets are kept fresh by the deployed refresh cron; this manager
never handles refresh tokens itself. The ``From`` address (Gmail) is resolved
from the provider profile of the token itself, so no additional secret needs to
be mirrored into the runtime.
"""

from __future__ import annotations

import base64
import logging
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Optional

import httpx

from unify.common.runtime_oauth import get_provider_access_token
from unify.common.plain_text import normalize_outbound_plain_text

logger = logging.getLogger(__name__)

_GMAIL_BASE = "https://gmail.googleapis.com/gmail/v1/users/me"
_GRAPH_BASE = "https://graph.microsoft.com/v1.0/me"


class WorkspaceEmailError(Exception):
    """Raised when the connected mailbox cannot be reached or a call fails."""


class WorkspaceEmailManager:
    """Send from and read the assistant's connected Google/Microsoft mailbox."""

    # Discovered by ToolSurfaceRegistry via this constant (no Base* class).
    _PRIMITIVE_METHODS = (
        "send",
        "list_messages",
        "search",
        "get_message",
    )

    def __init__(self) -> None:
        self._account_email_cache: dict[str, str] = {}

    # ── Provider / token resolution ──────────────────────────────────────

    @staticmethod
    def _secret(name: str) -> Optional[str]:
        from unify.manager_registry import ManagerRegistry

        sm = ManagerRegistry.get_secret_manager()
        getter = getattr(sm, "_get_secret_value", None)
        if callable(getter):
            value = getter(name)
            if isinstance(value, str) and value:
                return value
        return None

    def _provider(self) -> str:
        """Detect the connected provider from stored OAuth grants."""
        if self._secret("GOOGLE_GRANTED_SCOPES") or self._secret("GOOGLE_ACCESS_TOKEN"):
            return "google"
        if self._secret("MICROSOFT_GRANTED_SCOPES") or self._secret(
            "MICROSOFT_ACCESS_TOKEN",
        ):
            return "microsoft"
        raise WorkspaceEmailError(
            "No connected Google or Microsoft account is available.",
        )

    def _headers(self, provider: str) -> dict[str, str]:
        token = get_provider_access_token(provider)
        return {"Authorization": f"Bearer {token}"}

    async def _account_email(self, provider: str) -> str:
        """Resolve the connected account's own email address from the token.

        Asks the provider whom the token belongs to (Gmail ``getProfile`` /
        Graph ``/me``), so the ``From`` header reflects the connected account
        rather than the assistant's managed mailbox.
        """
        if provider in self._account_email_cache:
            return self._account_email_cache[provider]
        async with httpx.AsyncClient(timeout=30) as http:
            if provider == "google":
                resp = await http.get(
                    f"{_GMAIL_BASE}/profile",
                    headers=self._headers("google"),
                )
                resp.raise_for_status()
                email = resp.json().get("emailAddress") or ""
            else:
                resp = await http.get(
                    f"{_GRAPH_BASE}",
                    params={"$select": "mail,userPrincipalName"},
                    headers=self._headers("microsoft"),
                )
                resp.raise_for_status()
                data = resp.json()
                email = data.get("mail") or data.get("userPrincipalName") or ""
        if email:
            self._account_email_cache[provider] = email
        return email

    # ── Send ─────────────────────────────────────────────────────────────

    async def send(
        self,
        *,
        to: list[str] | str,
        subject: str,
        body: str,
        cc: list[str] | str | None = None,
        bcc: list[str] | str | None = None,
        in_reply_to: str | None = None,
        thread_id: str | None = None,
    ) -> dict[str, Any]:
        """Send an email **from the connected workspace account**.

        Unlike ``primitives.comms.send_email`` (which sends as the assistant's
        own managed mailbox), this sends as the user's connected Google/
        Microsoft account. Recipients are plain email-address strings, not
        contact ids.

        Parameters
        ----------
        to : list[str] | str
            Recipient email address(es).
        subject : str
            Subject line.
        body : str
            Plain-text body.
        cc, bcc : list[str] | str | None, optional
            Additional recipients.
        in_reply_to : str | None, optional
            RFC 5322 ``Message-ID`` of the message being replied to (threads the
            reply for the recipient). Gmail only.
        thread_id : str | None, optional
            Provider thread id to attach the reply to (Gmail ``threadId``).

        Returns
        -------
        dict[str, Any]
            ``{"success": True, "id": ...}`` on success.
        """
        provider = self._provider()
        to_list = [to] if isinstance(to, str) else list(to)
        cc_list = [cc] if isinstance(cc, str) and cc else (list(cc) if cc else [])
        bcc_list = [bcc] if isinstance(bcc, str) and bcc else (list(bcc) if bcc else [])
        body = normalize_outbound_plain_text(body)
        if provider == "google":
            return await self._google_send(
                to_list,
                subject,
                body,
                cc_list,
                bcc_list,
                in_reply_to,
                thread_id,
            )
        return await self._ms_send(to_list, subject, body, cc_list, bcc_list)

    async def _google_send(
        self,
        to: list[str],
        subject: str,
        body: str,
        cc: list[str],
        bcc: list[str],
        in_reply_to: str | None,
        thread_id: str | None,
    ) -> dict[str, Any]:
        from_email = await self._account_email("google")
        msg = MIMEMultipart()
        msg["from"] = from_email
        msg["to"] = ",".join(to)
        if cc:
            msg["cc"] = ",".join(cc)
        if bcc:
            msg["bcc"] = ",".join(bcc)
        msg["subject"] = subject
        msg.attach(MIMEText(body, "plain"))
        if in_reply_to:
            reply_id = in_reply_to.strip()
            if not reply_id.startswith("<"):
                reply_id = f"<{reply_id}"
            if not reply_id.endswith(">"):
                reply_id = f"{reply_id}>"
            msg["In-Reply-To"] = reply_id
            msg["References"] = reply_id
        raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
        send_body: dict[str, Any] = {"raw": raw}
        if thread_id:
            send_body["threadId"] = thread_id
        async with httpx.AsyncClient(timeout=60) as http:
            resp = await http.post(
                f"{_GMAIL_BASE}/messages/send",
                json=send_body,
                headers=self._headers("google"),
            )
        if resp.status_code >= 400:
            raise WorkspaceEmailError(
                f"Gmail send failed ({resp.status_code}): {resp.text[:300]}",
            )
        return {"success": True, "id": resp.json().get("id")}

    async def _ms_send(
        self,
        to: list[str],
        subject: str,
        body: str,
        cc: list[str],
        bcc: list[str],
    ) -> dict[str, Any]:
        def _recips(addrs: list[str]) -> list[dict[str, Any]]:
            return [{"emailAddress": {"address": addr}} for addr in addrs]

        message: dict[str, Any] = {
            "subject": subject,
            "body": {"contentType": "Text", "content": body},
            "toRecipients": _recips(to),
        }
        if cc:
            message["ccRecipients"] = _recips(cc)
        if bcc:
            message["bccRecipients"] = _recips(bcc)
        async with httpx.AsyncClient(timeout=60) as http:
            resp = await http.post(
                f"{_GRAPH_BASE}/sendMail",
                json={"message": message, "saveToSentItems": True},
                headers=self._headers("microsoft"),
            )
        if resp.status_code >= 400:
            raise WorkspaceEmailError(
                f"Graph sendMail failed ({resp.status_code}): {resp.text[:300]}",
            )
        return {"success": True}

    # ── Read ─────────────────────────────────────────────────────────────

    async def list_messages(
        self,
        query: str | None = None,
        max_results: int = 20,
    ) -> list[dict[str, Any]]:
        """List recent messages in the connected mailbox (newest first).

        Parameters
        ----------
        query : str | None, optional
            Provider search query (Gmail search syntax, e.g. ``"from:alice
            is:unread"``; Microsoft free-text ``$search``). Omit for the most
            recent inbox messages.
        max_results : int, optional
            Maximum number of messages to return.

        Returns
        -------
        list[dict[str, Any]]
            Message summaries with ``id``, ``thread_id``, ``from``, ``to``,
            ``subject``, ``date``, and ``snippet``.
        """
        provider = self._provider()
        if provider == "google":
            return await self._google_list(query, max_results)
        return await self._ms_list(query, max_results)

    async def search(
        self,
        query: str,
        max_results: int = 20,
    ) -> list[dict[str, Any]]:
        """Search the connected mailbox by provider query. See ``list_messages``."""
        return await self.list_messages(query=query, max_results=max_results)

    async def get_message(self, message_id: str) -> dict[str, Any]:
        """Return a single message including its plain-text body.

        Parameters
        ----------
        message_id : str
            Provider message id (from ``list_messages`` / ``search``).
        """
        provider = self._provider()
        if provider == "google":
            return await self._google_get(message_id)
        return await self._ms_get(message_id)

    # ── Google read helpers ──────────────────────────────────────────────

    async def _google_list(
        self,
        query: str | None,
        max_results: int,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"maxResults": max_results}
        if query:
            params["q"] = query
        async with httpx.AsyncClient(timeout=30) as http:
            resp = await http.get(
                f"{_GMAIL_BASE}/messages",
                params=params,
                headers=self._headers("google"),
            )
            resp.raise_for_status()
            ids = [m["id"] for m in resp.json().get("messages", [])]
            out: list[dict[str, Any]] = []
            for mid in ids:
                meta = await http.get(
                    f"{_GMAIL_BASE}/messages/{mid}",
                    params={
                        "format": "metadata",
                        "metadataHeaders": ["From", "To", "Subject", "Date"],
                    },
                    headers=self._headers("google"),
                )
                meta.raise_for_status()
                out.append(self._google_summary(meta.json()))
        return out

    async def _google_get(self, message_id: str) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=30) as http:
            resp = await http.get(
                f"{_GMAIL_BASE}/messages/{message_id}",
                params={"format": "full"},
                headers=self._headers("google"),
            )
        if resp.status_code == 404:
            raise WorkspaceEmailError(f"Message not found: {message_id}")
        resp.raise_for_status()
        raw = resp.json()
        summary = self._google_summary(raw)
        summary["body"] = self._google_body(raw.get("payload") or {})
        return summary

    @staticmethod
    def _google_headers(raw: dict[str, Any]) -> dict[str, str]:
        headers = (raw.get("payload") or {}).get("headers") or []
        return {h.get("name", "").lower(): h.get("value", "") for h in headers}

    def _google_summary(self, raw: dict[str, Any]) -> dict[str, Any]:
        headers = self._google_headers(raw)
        return {
            "id": raw.get("id"),
            "thread_id": raw.get("threadId"),
            "from": headers.get("from", ""),
            "to": headers.get("to", ""),
            "subject": headers.get("subject", ""),
            "date": headers.get("date", ""),
            "snippet": raw.get("snippet", ""),
        }

    def _google_body(self, payload: dict[str, Any]) -> str:
        mime = payload.get("mimeType", "")
        body = payload.get("body") or {}
        data = body.get("data")
        if mime == "text/plain" and data:
            return base64.urlsafe_b64decode(data.encode()).decode(
                "utf-8",
                errors="replace",
            )
        for part in payload.get("parts") or []:
            text = self._google_body(part)
            if text:
                return text
        return ""

    # ── Microsoft read helpers ───────────────────────────────────────────

    async def _ms_list(
        self,
        query: str | None,
        max_results: int,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {
            "$top": max_results,
            "$select": "id,conversationId,from,toRecipients,subject,receivedDateTime,bodyPreview",
            "$orderby": "receivedDateTime desc",
        }
        headers = self._headers("microsoft")
        if query:
            # $search is incompatible with $orderby on messages.
            params.pop("$orderby", None)
            params["$search"] = f'"{query}"'
        async with httpx.AsyncClient(timeout=30) as http:
            resp = await http.get(
                f"{_GRAPH_BASE}/messages",
                params=params,
                headers=headers,
            )
            resp.raise_for_status()
        return [self._ms_summary(m) for m in resp.json().get("value", [])]

    async def _ms_get(self, message_id: str) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=30) as http:
            resp = await http.get(
                f"{_GRAPH_BASE}/messages/{message_id}",
                params={
                    "$select": (
                        "id,conversationId,from,toRecipients,subject,"
                        "receivedDateTime,bodyPreview,body"
                    ),
                },
                headers=self._headers("microsoft"),
            )
        if resp.status_code == 404:
            raise WorkspaceEmailError(f"Message not found: {message_id}")
        resp.raise_for_status()
        raw = resp.json()
        summary = self._ms_summary(raw)
        summary["body"] = (raw.get("body") or {}).get("content", "") or raw.get(
            "bodyPreview",
            "",
        )
        return summary

    @staticmethod
    def _ms_address(node: dict[str, Any] | None) -> str:
        addr = ((node or {}).get("emailAddress") or {}).get("address")
        return addr or ""

    def _ms_summary(self, raw: dict[str, Any]) -> dict[str, Any]:
        to = ", ".join(self._ms_address(r) for r in (raw.get("toRecipients") or []))
        return {
            "id": raw.get("id"),
            "thread_id": raw.get("conversationId"),
            "from": self._ms_address(raw.get("from")),
            "to": to,
            "subject": raw.get("subject", ""),
            "date": raw.get("receivedDateTime", ""),
            "snippet": raw.get("bodyPreview", ""),
        }
