"""FastAPI routes for the Gmail channel.

Ports ``communication/gmail/views.py`` into ``unify.gateway``,
applying the translation rules from
``unify/gateway/channels/README.md``:

* Service-account credentials JSON read through ``EnvCredentialStore``
  (env var ``GCP_SA_KEY``) rather than ``os.getenv`` directly.
* Workspace admin subject (the user we impersonate for Admin
  Directory calls) read through ``EnvCredentialStore`` with the
  legacy default preserved.
* Gmail Pub/Sub topic name derived from ``SETTINGS.GCP_PROJECT_ID``
  + ``SETTINGS.ENV_SUFFIX`` -- matches the
  ``f"gmail-notifications{env_suffix}"`` convention in
  ``communication/common/settings.py``.
* Orchestra lookup helper imported from
  ``unify.gateway.common.orchestra`` (promoted when outlook became
  the second channel needing the same surface in Phase B.4.prep).
* ``print()`` debug calls replaced with structured logger entries.

Wire behaviour (route paths, request/response shapes, status codes,
threading headers, attachment encoding) is preserved bit-for-bit.
"""

from __future__ import annotations

import base64
import json
import logging
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formataddr, parseaddr
from typing import Any

from fastapi import APIRouter, HTTPException, Request, Response
from google.oauth2.credentials import Credentials as OAuthCredentials
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from unify.gateway.common.orchestra import lookup_assistant
from unify.gateway.credentials import (
    CredentialNotFoundError,
    CredentialStore,
    EnvCredentialStore,
)
from unify.settings import SETTINGS

logger = logging.getLogger("unify.gateway.channels.gmail")

router = APIRouter()


_GMAIL_SCOPES: list[str] = [
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.modify",
]
_DIRECTORY_SCOPES: list[str] = [
    "https://www.googleapis.com/auth/admin.directory.user",
]


# ---------------------------------------------------------------------------
# Helpers (channel-local; promote to unify/gateway/common/ on second-channel use)
# ---------------------------------------------------------------------------


def _is_google_not_found_error(exc: Exception) -> bool:
    """Return True when a Google API error represents an already-missing user."""
    return isinstance(exc, HttpError) and getattr(exc.resp, "status", None) == 404


def _as_message_id(mid: str) -> str:
    """Return *mid* as a canonical RFC 5322 Message-ID wrapped in ``<...>``.

    The caller may pass the reply target's Message-ID with or without the
    surrounding angle brackets. A bare id makes the ``In-Reply-To`` /
    ``References`` headers malformed, so the recipient's mail client cannot
    match the reply to the original and shows it as a new thread (Gmail's own
    ``threadId`` only fixes the sender's mailbox view, not the recipient's).
    """
    mid = (mid or "").strip()
    if not mid:
        return mid
    if not mid.startswith("<"):
        mid = f"<{mid}"
    if not mid.endswith(">"):
        mid = f"{mid}>"
    return mid


def _recipient_count(value: Any) -> int:
    if not value:
        return 0
    if isinstance(value, str):
        return 1
    return len(value)


def _gmail_send_log_context(
    *,
    sender: str,
    to: Any,
    cc: Any,
    bcc: Any,
    subject: str,
    in_reply_to: str | None,
    thread_id: str | None,
    attachment: Any,
) -> dict[str, Any]:
    return {
        "sender": sender,
        "to_count": _recipient_count(to),
        "cc_count": _recipient_count(cc),
        "bcc_count": _recipient_count(bcc),
        "subject": subject,
        "has_in_reply_to": bool(in_reply_to),
        "has_thread_id": bool(thread_id),
        "has_attachment": bool(attachment),
    }


def _resolve_directory_display_name(email: str) -> str | None:
    """Return the Workspace Directory full name for *email*, if available."""
    try:
        service = get_admin_service()
        user = service.users().get(userKey=email, projection="full").execute()
    except Exception:
        logger.warning(
            "Could not resolve Workspace display name for %s",
            email,
            exc_info=True,
        )
        return None
    name = user.get("name") or {}
    full_name = (name.get("fullName") or "").strip()
    if full_name:
        return full_name
    given_name = (name.get("givenName") or "").strip()
    family_name = (name.get("familyName") or "").strip()
    combined = " ".join(part for part in (given_name, family_name) if part).strip()
    return combined or None


def _format_from_header(sender: str, from_name: str | None = None) -> str:
    """Build a RFC 5322 From header with a human-readable display name."""
    existing_name, addr = parseaddr(sender)
    if not addr:
        addr = sender.strip()
        existing_name = ""
    display_name = (from_name or existing_name or "").strip()
    if not display_name and addr.lower().endswith("@unify.ai"):
        display_name = (_resolve_directory_display_name(addr) or "").strip()
    if display_name:
        return formataddr((display_name, addr))
    return addr


def _service_account_credentials(
    *,
    scopes: list[str],
    subject: str,
    credentials: CredentialStore,
) -> Credentials:
    """Build delegated service-account credentials for Workspace operations."""
    try:
        creds_json = credentials.get("GCP_SA_KEY")
    except CredentialNotFoundError as exc:
        raise RuntimeError(
            "GCP_SA_KEY must be set for Gmail operations",
        ) from exc
    return Credentials.from_service_account_info(
        json.loads(creds_json),
        scopes=scopes,
        subject=subject,
    )


def _workspace_admin_subject(credentials: CredentialStore) -> str:
    """Workspace user impersonated for Admin SDK Directory calls.

    Reads ``WORKSPACE_ADMIN_SUBJECT`` with the legacy default
    (``dan@unify.ai``) preserved bit-for-bit from
    ``communication/common/settings.py:165-168``.
    """
    return credentials.get_optional("WORKSPACE_ADMIN_SUBJECT", "dan@unify.ai")


def _gmail_topic_path(topic_name: str | None = None) -> str:
    """Fully qualified Pub/Sub topic path for Gmail watches.

    Default topic name matches the convention in
    ``communication/common/settings.py:207``:
    ``f"gmail-notifications{env_suffix}"``.
    """
    name = topic_name or f"gmail-notifications{SETTINGS.ENV_SUFFIX}"
    return f"projects/{SETTINGS.GCP_PROJECT_ID}/topics/{name}"


def get_admin_service(credentials: CredentialStore | None = None) -> Any:
    """Directory API client with domain-wide delegation."""
    credentials = credentials or EnvCredentialStore()
    creds = _service_account_credentials(
        scopes=_DIRECTORY_SCOPES,
        subject=_workspace_admin_subject(credentials),
        credentials=credentials,
    )
    return build("admin", "directory_v1", credentials=creds)


async def get_gmail_service_async(
    sender_email: str,
    credentials: CredentialStore | None = None,
) -> Any:
    """Build a Gmail API client, preferring BYOD OAuth tokens.

    BYOD-managed assistants have a ``GOOGLE_ACCESS_TOKEN`` secret in
    their Orchestra record; we use that when available. Platform-
    managed Workspace mailboxes fall back to service-account
    delegation against ``sender_email``.

    Assistant lookup failures with 4xx status are non-fatal -- we
    fall back to SA delegation -- but 5xx propagates so transient
    Orchestra outages don't silently switch to the wrong credential.
    """
    credentials = credentials or EnvCredentialStore()
    try:
        assistant = await lookup_assistant(sender_email, credentials)
    except HTTPException as exc:
        if exc.status_code >= 500:
            raise
        logger.warning(
            "Assistant lookup failed for %s (status %s), falling back to SA",
            sender_email,
            exc.status_code,
        )
        creds = _service_account_credentials(
            scopes=_GMAIL_SCOPES,
            subject=sender_email,
            credentials=credentials,
        )
        return build("gmail", "v1", credentials=creds)
    except Exception:
        logger.warning(
            "Failed to look up assistant for %s, falling back to SA",
            sender_email,
        )
        creds = _service_account_credentials(
            scopes=_GMAIL_SCOPES,
            subject=sender_email,
            credentials=credentials,
        )
        return build("gmail", "v1", credentials=creds)

    access_token = assistant.get("secrets", {}).get("GOOGLE_ACCESS_TOKEN")
    if access_token:
        return build("gmail", "v1", credentials=OAuthCredentials(token=access_token))

    creds = _service_account_credentials(
        scopes=_GMAIL_SCOPES,
        subject=sender_email,
        credentials=credentials,
    )
    return build("gmail", "v1", credentials=creds)


def get_gmail_service(
    sender_email: str,
    credentials: CredentialStore | None = None,
) -> Any:
    """Build a Gmail API client via SA delegation (synchronous variant).

    Used by callers that cannot await (e.g. the adapters' inbound
    Gmail processor). For the async path that also supports BYOD
    tokens, use ``get_gmail_service_async``.
    """
    credentials = credentials or EnvCredentialStore()
    creds = _service_account_credentials(
        scopes=_GMAIL_SCOPES,
        subject=sender_email,
        credentials=credentials,
    )
    return build("gmail", "v1", credentials=creds)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.delete("/delete")
async def delete_email_user(request: Request):
    """Delete a Workspace user, treating an already-missing user as success."""
    data = await request.json()
    primary_email = data.get("primary_email")
    if not primary_email:
        raise HTTPException(status_code=400, detail="Missing primary_email")
    try:
        service = get_admin_service()
        service.users().delete(userKey=primary_email).execute()
        return {
            "success": True,
            "deleted": True,
            "already_absent": False,
            "message": f"User {primary_email} deleted.",
        }
    except Exception as exc:
        if _is_google_not_found_error(exc):
            logger.info(
                "Workspace user %s already absent during delete",
                primary_email,
            )
            return {
                "success": True,
                "deleted": False,
                "already_absent": True,
                "message": f"User {primary_email} already absent.",
            }
        logger.error("Failed to delete user: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/send")
async def send_email(request: Request):
    """Send an email via Gmail.

    Required body: ``from``, ``to`` (str or list), ``body``.
    Optional: ``subject``, ``cc``, ``bcc``, ``in_reply_to``, ``thread_id``,
    ``attachment={filename, content_base64}``.
    """
    data = await request.json()
    sender = data.get("from")
    from_name = data.get("from_name")
    to = data.get("to")
    cc = data.get("cc")
    bcc = data.get("bcc")
    subject = data.get("subject", "")
    body = data.get("body")
    in_reply_to = data.get("in_reply_to")
    thread_id = data.get("thread_id")
    attachment = data.get("attachment")

    if not sender or not to or body is None:
        raise HTTPException(
            status_code=400,
            detail="Missing required fields: 'from', 'to', 'body'",
        )

    msg = MIMEMultipart()
    msg["from"] = _format_from_header(sender, from_name)
    msg["to"] = to if isinstance(to, str) else ",".join(to)
    if cc:
        msg["cc"] = cc if isinstance(cc, str) else ",".join(cc)
    if bcc:
        msg["bcc"] = bcc if isinstance(bcc, str) else ",".join(bcc)
    msg["subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    if attachment:
        try:
            filename = attachment.get("filename", "attachment")
            content_base64 = attachment.get("content_base64", "")
            file_data = base64.b64decode(content_base64, validate=True)

            part = MIMEBase("application", "octet-stream")
            part.set_payload(file_data)
            encoders.encode_base64(part)
            part.add_header(
                "Content-Disposition",
                f'attachment; filename="{filename}"',
            )
            msg.attach(part)
            logger.debug("attached file %s (%d bytes)", filename, len(file_data))
        except Exception as exc:
            logger.error("failed to attach file: %s", exc)
            raise HTTPException(
                status_code=400,
                detail=f"Failed to attach file: {exc}",
            )

    if in_reply_to:
        reply_id = _as_message_id(in_reply_to)
        msg["In-Reply-To"] = reply_id
        msg["References"] = reply_id

    raw_msg = base64.urlsafe_b64encode(msg.as_bytes()).decode()
    service = await get_gmail_service_async(sender)
    send_body = {"raw": raw_msg}
    if thread_id:
        send_body["threadId"] = thread_id
    log_context = _gmail_send_log_context(
        sender=sender,
        to=to,
        cc=cc,
        bcc=bcc,
        subject=subject,
        in_reply_to=in_reply_to,
        thread_id=thread_id,
        attachment=attachment,
    )
    try:
        sent = service.users().messages().send(userId="me", body=send_body).execute()
    except HttpError as exc:
        logger.exception(
            "gmail send failed with Google API error: status=%s reason=%s context=%s",
            getattr(exc.resp, "status", None),
            getattr(exc.resp, "reason", None),
            log_context,
        )
        raise HTTPException(
            status_code=500,
            detail="Gmail send failed; see gateway logs for traceback.",
        ) from exc
    except Exception as exc:
        logger.exception("gmail send failed: context=%s", log_context)
        raise HTTPException(
            status_code=500,
            detail="Gmail send failed; see gateway logs for traceback.",
        ) from exc
    return {"success": True, "id": sent.get("id")}


@router.post("/watch")
async def watch_email(request: Request):
    """Start Gmail push notifications for ``primary_email``."""
    data = await request.json()
    user_email = data.get("primary_email")
    if not user_email:
        raise HTTPException(status_code=400, detail="Missing primary_email")

    gmail_service = await get_gmail_service_async(user_email)
    watch_request = {
        "labelIds": ["INBOX"],
        "topicName": _gmail_topic_path(data.get("topic_name")),
    }
    watch_resp = gmail_service.users().watch(userId="me", body=watch_request).execute()
    return {"success": True, "historyId": watch_resp.get("historyId")}


@router.delete("/watch")
async def delete_gmail_watch(request: Request):
    """Stop Gmail push notifications for ``primary_email``.

    Must be called *before* the BYOD access token is revoked; once
    the token is gone ``get_gmail_service_async`` either sees a
    revoked token or falls through to SA delegation, which isn't
    authorized for BYOD mailboxes.
    """
    data = await request.json()
    user_email = data.get("primary_email")
    if not user_email:
        raise HTTPException(status_code=400, detail="Missing primary_email")

    gmail_service = await get_gmail_service_async(user_email)
    try:
        gmail_service.users().stop(userId="me").execute()
    except HttpError as exc:
        if _is_google_not_found_error(exc):
            logger.info("Gmail watch already absent for %s during delete", user_email)
            return {
                "success": True,
                "primary_email": user_email,
                "already_absent": True,
            }
        logger.error("Failed to stop Gmail watch for %s: %s", user_email, exc)
        raise HTTPException(status_code=500, detail=str(exc))

    return {"success": True, "primary_email": user_email}


@router.get("/attachment")
async def get_attachment(
    receiver_email: str,
    gmail_message_id: str,
    attachment_id: str,
    filename: str | None = None,
):
    """Download an inbound email attachment as binary bytes."""
    try:
        service = await get_gmail_service_async(receiver_email)
        attachment = (
            service.users()
            .messages()
            .attachments()
            .get(userId="me", messageId=gmail_message_id, id=attachment_id)
            .execute()
        )
        data = attachment.get("data")
        if not data:
            raise HTTPException(status_code=404, detail="Attachment not found")
        file_bytes = base64.urlsafe_b64decode(data.encode("utf-8"))
        return Response(
            content=file_bytes,
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": (
                    f"attachment; filename={filename or 'attachment'}"
                ),
            },
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


__all__ = [
    "get_admin_service",
    "get_gmail_service",
    "get_gmail_service_async",
    "router",
]
