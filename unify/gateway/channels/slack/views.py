"""Slack admin endpoints (tenant-facing).

Three admin endpoints, all bearer-authed via the Orchestra admin
key. Mounted under ``/slack`` by ``unify.gateway.app``:

* ``POST /install`` -- record a workspace OAuth install in Orchestra
  (bot token, ``bot_user_id``, ``team_id``, scopes). Called by Console
  after the Slack OAuth callback completes.
* ``POST /send``    -- outbound DM, channel post, or threaded reply
  via ``chat.postMessage``. Resolves the workspace bot token from
  Orchestra by ``team_id``.
* ``POST /user-info`` -- look up a Slack user's profile (email + real /
  display name) via ``users.info``, so the inbound pipeline can resolve
  an unknown sender to an existing contact on first message.
* ``GET  /status``  -- list installs known to Orchestra (debug /
  health check).

The inbound ``/slack/events`` webhook does NOT live here -- it is in
``communication/adapters/main.py:slack_events_webhook`` alongside the
other third-party webhooks (Twilio, Microsoft, Google). Splitting
that way matches the existing channel topology: high-volume external
webhooks land in adapters; admin-authed tenant-side endpoints land
in the gateway.
"""

from __future__ import annotations

import logging

import httpx
from fastapi import APIRouter, HTTPException, Request

from unify.settings import SETTINGS

logger = logging.getLogger("unify.gateway.channels.slack.views")


def _log_field(value: object) -> str:
    return str(value).replace("\r", "").replace("\n", "")


auth_router = APIRouter()

SLACK_API_BASE = "https://slack.com/api"


def _admin_headers() -> dict:
    """Bearer headers for Orchestra admin API calls."""
    return {
        "Authorization": f"Bearer {SETTINGS.ORCHESTRA_ADMIN_KEY.get_secret_value()}",
    }


async def _resolve_bot_token(team_id: str) -> str:
    """Return the workspace bot token for ``team_id`` (admin auth).

    Orchestra's install read endpoint keys on ``slack_team_id`` and only
    returns the bot token when ``include_token=true`` is requested.
    """
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{SETTINGS.ORCHESTRA_URL}/admin/slack/install",
            params={"slack_team_id": team_id, "include_token": True},
            headers=_admin_headers(),
            timeout=10.0,
        )
    if resp.status_code == 404:
        raise HTTPException(
            status_code=404,
            detail=f"No Slack install for team {team_id}",
        )
    if resp.status_code >= 400:
        raise HTTPException(
            status_code=resp.status_code,
            detail=resp.text,
        )
    install = resp.json()
    token = install.get("bot_access_token") or ""
    if not token:
        raise HTTPException(
            status_code=503,
            detail=f"Slack install for team {team_id} has no bot token",
        )
    return token


# ---------------------------------------------------------------------------
# POST /install
# ---------------------------------------------------------------------------


@auth_router.post("/install")
async def upsert_install(request: Request):
    """Register or refresh a workspace OAuth install in Orchestra.

    Body::

        {
            "organization_id": int,
            "team_id": str,
            "team_name": str,
            "bot_user_id": str,
            "bot_access_token": str,
            "scopes": [str, ...],
            "installer_slack_user_id": str | None,
        }

    Idempotent: re-installing the same workspace updates the stored
    token and scopes (Slack rotates bot tokens on reinstall).
    """
    data = await request.json()
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{SETTINGS.ORCHESTRA_URL}/admin/slack/install",
            json=data,
            headers=_admin_headers(),
            timeout=10.0,
        )
    if resp.status_code >= 400:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)
    return resp.json()


# ---------------------------------------------------------------------------
# POST /send
# ---------------------------------------------------------------------------


@auth_router.post("/send")
async def send_slack_message(request: Request):
    """Send a Slack message via ``chat.postMessage``.

    Modes (mutually exclusive):

    * **DM**:                ``{"user_id": "...", "team_id": "..."}``.
      Communication opens (or reuses) a DM channel via
      ``conversations.open`` and posts there.
    * **Channel post / reply**: ``{"channel_id": "...", "team_id": "..."}``
      (optional ``thread_ts``).
    """
    data = await request.json()
    team_id = data["team_id"]
    body = data["body"]
    thread_ts = data.get("thread_ts")
    user_id = data.get("user_id")
    channel_id = data.get("channel_id")

    if not channel_id and not user_id:
        raise HTTPException(
            status_code=400,
            detail="Either 'channel_id' or 'user_id' is required",
        )

    bot_token = await _resolve_bot_token(team_id)
    headers = {
        "Authorization": f"Bearer {bot_token}",
        "Content-Type": "application/json; charset=utf-8",
    }

    if not channel_id:
        async with httpx.AsyncClient() as client:
            open_resp = await client.post(
                f"{SLACK_API_BASE}/conversations.open",
                json={"users": user_id},
                headers=headers,
                timeout=10.0,
            )
        open_payload = open_resp.json()
        if not open_payload.get("ok"):
            raise HTTPException(
                status_code=502,
                detail=f"conversations.open failed: {open_payload.get('error')}",
            )
        channel_id = open_payload["channel"]["id"]

    msg_payload: dict = {"channel": channel_id, "text": body}
    if thread_ts:
        msg_payload["thread_ts"] = thread_ts

    async with httpx.AsyncClient() as client:
        msg_resp = await client.post(
            f"{SLACK_API_BASE}/chat.postMessage",
            json=msg_payload,
            headers=headers,
            timeout=10.0,
        )
    msg_data = msg_resp.json()
    if not msg_data.get("ok"):
        raise HTTPException(
            status_code=502,
            detail=f"chat.postMessage failed: {msg_data.get('error')}",
        )
    logger.info(
        "sent Slack message to %s on team %s (ts=%s)",
        _log_field(channel_id),
        _log_field(team_id),
        _log_field(msg_data.get("ts")),
    )
    return {
        "success": True,
        "message_ts": msg_data.get("ts"),
        "channel_id": channel_id,
    }


# ---------------------------------------------------------------------------
# POST /user-info
# ---------------------------------------------------------------------------


async def fetch_slack_user_profile(team_id: str, slack_user_id: str) -> dict:
    """Resolve a Slack user's profile via ``users.info``.

    Returns ``{slack_user_id, email, real_name, display_name, tz}``.
    ``email`` is only populated when the workspace bot has the
    ``users:read.email`` scope; ``real_name`` / ``display_name`` need only
    ``users:read``. Raises :class:`HTTPException` on a hard failure (no
    install, missing bot token, or a Slack API error).
    """
    bot_token = await _resolve_bot_token(team_id)
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{SLACK_API_BASE}/users.info",
            params={"user": slack_user_id},
            headers={"Authorization": f"Bearer {bot_token}"},
            timeout=10.0,
        )
    payload = resp.json()
    if not payload.get("ok"):
        raise HTTPException(
            status_code=502,
            detail=f"users.info failed: {payload.get('error')}",
        )

    user = payload.get("user") or {}
    profile = user.get("profile") or {}
    return {
        "slack_user_id": slack_user_id,
        "email": profile.get("email") or None,
        "real_name": user.get("real_name") or profile.get("real_name") or None,
        "display_name": profile.get("display_name") or None,
        "tz": user.get("tz") or None,
    }


@auth_router.post("/user-info")
async def slack_user_info(request: Request):
    """Resolve a Slack user's profile via ``users.info``.

    Body::

        {"team_id": str, "slack_user_id": str}

    The inbound pipeline uses the returned profile to match an unknown
    sender to an existing contact (by email, then by name).
    """
    data = await request.json()
    return await fetch_slack_user_profile(data["team_id"], data["slack_user_id"])


# ---------------------------------------------------------------------------
# POST /user-by-email
# ---------------------------------------------------------------------------


async def lookup_slack_user_id_by_email(team_id: str, email: str) -> str | None:
    """Resolve a Slack user ID from an email via ``users.lookupByEmail``.

    This is the reverse of ``users.info`` and the only way to reach a
    workspace member the bot has never heard from: given a contact's email
    it returns their Slack user ID so an assistant can open a DM. Returns
    ``None`` when the workspace has no member with that email
    (``users_not_found``), when the bot lacks the ``users:read.email``
    scope, or when the email is empty — the caller treats any ``None`` as
    "unresolved" and falls back to its existing behaviour.
    """
    if not email:
        return None
    bot_token = await _resolve_bot_token(team_id)
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{SLACK_API_BASE}/users.lookupByEmail",
            params={"email": email},
            headers={"Authorization": f"Bearer {bot_token}"},
            timeout=10.0,
        )
    payload = resp.json()
    if not payload.get("ok"):
        logger.warning(
            f"slack users.lookupByEmail failed: {_log_field(payload.get('error'))}",
        )
        return None
    return (payload.get("user") or {}).get("id") or None


@auth_router.post("/user-by-email")
async def slack_user_by_email(request: Request):
    """Resolve a Slack user ID from an email via ``users.lookupByEmail``.

    Body::

        {"team_id": str, "email": str}

    Returns ``{"slack_user_id": str | None}``. Lets the outbound pipeline
    reach a contact that has an email on file but no Slack user ID yet.
    """
    data = await request.json()
    slack_user_id = await lookup_slack_user_id_by_email(
        data["team_id"],
        data.get("email") or "",
    )
    return {"slack_user_id": slack_user_id}


# ---------------------------------------------------------------------------
# GET /status
# ---------------------------------------------------------------------------


@auth_router.get("/status")
async def status():
    """List workspace installs known to Orchestra."""
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{SETTINGS.ORCHESTRA_URL}/admin/slack/installs",
            headers=_admin_headers(),
            timeout=10.0,
        )
    if resp.status_code >= 400:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)
    return resp.json()


__all__ = ["auth_router"]
