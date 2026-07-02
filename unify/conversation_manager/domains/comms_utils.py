from dotenv import load_dotenv
import asyncio
import base64
import aiohttp
import json
import os
from pathlib import Path

from unify.logger import LOGGER
from unify.common.hierarchical_logger import ICONS
from unify.common.plain_text import normalize_outbound_plain_text
from unify.session_details import SESSION_DETAILS
from unify.settings import SETTINGS

load_dotenv()
headers = {"Authorization": f"Bearer {SETTINGS.ORCHESTRA_ADMIN_KEY.get_secret_value()}"}

# Lazily initialized publisher (avoids import-time GCP auth failures in tests)
_publisher = None


def _get_publisher():
    """Get or create the GCP Pub/Sub publisher client."""
    global _publisher
    if _publisher is None:
        from google.cloud import pubsub_v1

        _publisher = pubsub_v1.PublisherClient()
    return _publisher


# Optional injected outbound transport (unify.gateway.OutboundTransport).
# When set, replaces the inline `_get_publisher().publish().result()` path
# used by the three publish helpers below (send_unify_message,
# publish_system_error, publish_assistant_desktop_ready). When None (the
# default for every call site at the time of this change), the existing
# inline pubsub_v1 publisher remains active so behaviour for legacy
# callers is unchanged. Wired by unify.conversation_manager.main based
# on the UNITY_CONVERSATION_OUTBOUND_TRANSPORT setting; see Phase A.bis.7
# in unify/gateway/PHASES.md.
_outbound_transport = None  # type: ignore[var-annotated]


def set_outbound_transport(transport) -> None:
    """Configure the outbound transport used by the publish helpers.

    Called once at process startup by main.py. Passing ``None`` keeps
    the legacy inline Pub/Sub path active.
    """
    global _outbound_transport
    _outbound_transport = transport


def get_outbound_transport():
    """Return the currently configured outbound transport (or ``None``)."""
    return _outbound_transport


def _publish_to_assistant_topic(
    *,
    agent_id,
    thread: str,
    event: dict,
    timeout: float | None = None,
) -> str:
    """Publish a ``{thread, event}`` envelope to the per-assistant topic.

    Uses the injected OutboundTransport when one is configured via
    ``set_outbound_transport``; otherwise falls back to the legacy
    inline pubsub_v1 publisher path. Returns the broker-assigned
    message id on success. Raises on transport-level failure -- the
    caller decides whether to log-and-swallow or propagate, matching
    today's per-site error handling.

    The topic name (``unity-{agent_id}{env_suffix}``) and the
    ``thread`` attribute are preserved bit-for-bit across both paths,
    so consumers see identical message shapes regardless of which
    transport delivered them.
    """
    env_suffix = SETTINGS.ENV_SUFFIX if agent_id is not None else ""
    topic_name = f"unity-{agent_id}{env_suffix}"
    message_bytes = json.dumps({"thread": thread, "event": event}).encode("utf-8")

    transport = _outbound_transport
    if transport is not None:
        return transport.publish(
            topic_name,
            message_bytes,
            thread=thread,
            timeout=timeout,
        )

    publisher = _get_publisher()
    topic_path = publisher.topic_path(SETTINGS.GCP_PROJECT_ID, topic_name)
    future = publisher.publish(topic_path, message_bytes, thread=thread)
    if timeout is not None:
        return future.result(timeout=timeout)
    return future.result()


def _use_local_comms() -> bool:
    enabled = getattr(SETTINGS.conversation, "LOCAL_COMMS_ENABLED", False)
    mode = getattr(SETTINGS.conversation, "LOCAL_COMMS_MODE", "hosted")
    return enabled is True or (isinstance(mode, str) and mode == "local")


def _local_comms_base_url() -> str:
    public_url = SETTINGS.conversation.LOCAL_COMMS_PUBLIC_URL.strip()
    if public_url:
        return public_url.rstrip("/")
    return (
        f"http://{SETTINGS.conversation.LOCAL_COMMS_HOST}:"
        f"{SETTINGS.conversation.LOCAL_COMMS_PORT}"
    )


def _gateway_comms_base_url() -> str:
    base_url = SETTINGS.conversation.COMMS_URL.strip()
    if base_url:
        return base_url.rstrip("/")
    return _local_comms_base_url()


def _gateway_adapters_base_url() -> str:
    base_url = SETTINGS.conversation.ADAPTERS_URL.strip()
    if base_url:
        return base_url.rstrip("/")
    return _gateway_comms_base_url()


async def _publish_local_outbox_async(payload: dict) -> bool:
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{_local_comms_base_url()}/local/comms/outbox",
            json=payload,
        ) as response:
            return response.status < 400


def _publish_local_outbox_sync(payload: dict) -> bool:
    import requests

    response = requests.post(
        f"{_local_comms_base_url()}/local/comms/outbox",
        json=payload,
        timeout=10,
    )
    return response.status_code < 400


def _inline_attachment_bytes(attachment: dict) -> bytes | None:
    encoded = attachment.get("content_base64")
    if not encoded:
        return None
    if attachment.get("content_encoding") == "hex":
        return bytes.fromhex(encoded)
    return base64.b64decode(encoded.encode("ascii"))


async def send_sms_message_via_number(to_number: str, content: str) -> str:
    """
    Send an SMS message using the SMS provider API.

    Args:
        to_number: The recipient's phone number
        content: The message content to send

    Returns:
        str: The response from the SMS API
    """
    from_number = SESSION_DETAILS.assistant.number
    if not from_number:
        return {"success": False}

    content = normalize_outbound_plain_text(content)

    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{_gateway_comms_base_url()}/phone/send-text",
            headers=headers,
            json={
                "From": from_number,
                "To": to_number,
                "Body": content,
            },
        ) as response:
            try:
                response.raise_for_status()
            except Exception as e:
                LOGGER.error(f"{ICONS['comms_outbound']} {e}")
                return {"success": False}
            return await response.json()


async def send_whatsapp_message(
    to_number: str,
    content: str,
    user_name: str = "",
    agent_name: str = "",
    media_url: str | None = None,
) -> dict:
    """
    Send a WhatsApp message via the Communication service.

    Communication automatically handles the WhatsApp 24h session window:
    if the window is open, ``content`` is sent as free-form text; if closed,
    it falls back to an approved greeting template and returns the delivered
    template body separately from the intended ``content``.

    Args:
        to_number: The recipient's WhatsApp number (E.164)
        content: The message content to send
        user_name: Recipient's first name (used in template fallback)
        agent_name: Assistant's first name (used in template fallback)
        media_url: Publicly accessible URL of a media attachment (one per
            message — WhatsApp constraint).  Supported types: images, audio,
            video, PDF, DOC/XLSX when inside the 24h window.

    Returns:
        dict with 'success' key indicating delivery status.
    """
    agent_id = SESSION_DETAILS.assistant.agent_id
    if agent_id is None:
        return {"success": False}

    content = normalize_outbound_plain_text(content)

    payload = {
        "to": to_number,
        "body": content,
        "assistant_id": agent_id,
        "user_name": user_name,
        "agent_name": agent_name,
    }
    if media_url:
        payload["media_url"] = media_url

    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{_gateway_comms_base_url()}/whatsapp/send",
            headers=headers,
            json=payload,
        ) as response:
            try:
                response.raise_for_status()
            except Exception as e:
                LOGGER.error(f"{ICONS['comms_outbound']} WhatsApp send failed: {e}")
                return {"success": False}
            return await response.json()


async def get_whatsapp_window(to_number: str) -> bool | None:
    """Best-effort read of a contact's WhatsApp free-form window state.

    Returns ``True``/``False`` when the gateway answers, or ``None`` on any
    failure (the caller then treats the window as unknown and lets the
    window-agnostic guidance stand).
    """
    agent_id = SESSION_DETAILS.assistant.agent_id
    if agent_id is None or not to_number:
        return None
    try:
        timeout = aiohttp.ClientTimeout(total=5)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(
                f"{_gateway_comms_base_url()}/whatsapp/window",
                headers=headers,
                params={"to": to_number, "assistant_id": agent_id},
            ) as response:
                response.raise_for_status()
                data = await response.json()
                value = data.get("window_open")
                return bool(value) if value is not None else None
    except Exception as e:
        LOGGER.debug(f"WhatsApp window check failed: {e}")
        return None


async def send_unify_message(
    content: str,
    contact_id: int = 1,
    attachment: dict | None = None,
) -> dict:
    """
    Send a unify message to a contact, optionally with an attachment.

    Args:
        content: The message content to send.
        contact_id: The target contact's ID. Defaults to 1 (boss).
        attachment: Optional attachment dict with keys:
            - id: Unique identifier for the attachment
            - filename: The name of the file
            - url: Signed URL to download the file

    Returns:
        dict with "success" key indicating delivery status.
    """
    content = normalize_outbound_plain_text(content)

    agent_id = SESSION_DETAILS.assistant.agent_id
    event_data = {"content": content, "role": "assistant", "contact_id": contact_id}
    if attachment:
        event_data["attachments"] = [attachment]

    message_data = {
        "thread": "unify_message_outbound",
        "event": event_data,
    }

    # The Unify chat surface (Console) always reads outbound messages
    # from the assistant's Pub/Sub topic, so we must publish there even
    # when running in ``LOCAL_COMMS_MODE=local``.  The in-memory local
    # outbox is kept as a best-effort side channel for the local Twilio
    # / email simulators (and the existing test that pokes at it),
    # which never consumed Pub/Sub.
    if _use_local_comms():
        try:
            await _publish_local_outbox_async(message_data)
        except Exception as e:
            LOGGER.debug(
                f"{ICONS['comms_outbound']} Local outbox mirror failed (non-fatal): {e}",
            )

    try:
        message_id = _publish_to_assistant_topic(
            agent_id=agent_id,
            thread="unify_message_outbound",
            event=event_data,
        )
        LOGGER.debug(
            f"{ICONS['comms_outbound']} Unify message published with ID: {message_id}",
        )
        return {"success": bool(message_id)}
    except Exception as e:
        LOGGER.error(f"{ICONS['comms_outbound']} Error sending unify message: {e}")
        return {"success": False, "error": str(e)}


async def send_unify_meet_ring(
    call_session_id: str,
    reason: str = "",
    contact_id: int = 1,
) -> dict:
    """Ring the owner on Unify Meet (the in-app live call).

    The assistant cannot place the call itself - the owner's browser mints the
    LiveKit token and joins the room. This publishes a ``unify_meet_incoming``
    signal on the assistant's Pub/Sub topic so the Console shows a pinned
    incoming-call window; when the owner clicks Answer, Console runs its normal
    connect flow (token + ``/unify/meet`` dispatch) which lands as
    ``UnifyMeetReceived`` here. ``reason`` is the briefing for how I open once
    answered; Console turns it into a briefed opening config.
    """
    agent_id = SESSION_DETAILS.assistant.agent_id
    event_data = {
        "call_session_id": call_session_id,
        "reason": reason,
        "contact_id": contact_id,
    }

    message_data = {"thread": "unify_meet_incoming", "event": event_data}

    # Console always reads from the assistant's Pub/Sub topic, even in
    # LOCAL_COMMS_MODE=local; the local outbox is a best-effort mirror.
    if _use_local_comms():
        try:
            await _publish_local_outbox_async(message_data)
        except Exception as e:
            LOGGER.debug(
                f"{ICONS['comms_outbound']} Local outbox mirror failed (non-fatal): {e}",
            )

    try:
        message_id = _publish_to_assistant_topic(
            agent_id=agent_id,
            thread="unify_meet_incoming",
            event=event_data,
        )
        LOGGER.debug(
            f"{ICONS['comms_outbound']} Unify Meet ring published with ID: {message_id}",
        )
        return {"success": bool(message_id)}
    except Exception as e:
        LOGGER.error(f"{ICONS['comms_outbound']} Error ringing Unify Meet: {e}")
        return {"success": False, "error": str(e)}


def publish_system_error(error_message: str, error_type: str = "unknown") -> None:
    """Publish a system error to the assistant's Pub/Sub topic.

    This is a best-effort, fire-and-forget publish used to notify the console
    that the container hit an unrecoverable error (OOM, unhandled exception, etc.)
    so the UI can show a user-friendly warning instead of going silent.

    Args:
        error_message: Human-readable description of the error.
        error_type: Structured error type for console classification. One of:
            ``oom``, ``startup_failed``, ``init_failed``, ``message_failed``,
            ``recovering``, ``unknown``.

    Uses a synchronous publish (no await) so it can be called from both sync
    and async contexts, including signal handlers and thread-pool callbacks.
    """
    agent_id = SESSION_DETAILS.assistant.agent_id
    if agent_id is None:
        return

    if _use_local_comms():
        try:
            _publish_local_outbox_sync(
                {
                    "thread": "system_error",
                    "event": {
                        "content": error_message,
                        "error_type": error_type,
                    },
                },
            )
        except Exception as e:
            LOGGER.error(
                f"{ICONS['comms_outbound']} Failed to publish system error: {e}",
            )
        return

    try:
        _publish_to_assistant_topic(
            agent_id=agent_id,
            thread="system_error",
            event={
                "content": error_message,
                "error_type": error_type,
            },
            timeout=5,
        )
        LOGGER.debug(
            f"{ICONS['comms_outbound']} Published system error [{error_type}]: {error_message}",
        )
    except Exception as e:
        LOGGER.error(f"{ICONS['comms_outbound']} Failed to publish system error: {e}")


async def complete_api_message(
    api_message_id: str,
    response: str | None = None,
    attachments: list[dict] | None = None,
    tags: list[str] | None = None,
) -> dict:
    """Mark an API message as completed in Orchestra, optionally with a response."""
    orchestra_url = SETTINGS.ORCHESTRA_URL
    body: dict = {"response": response}
    if attachments:
        body["attachments"] = [
            {
                "id": att.get("id", ""),
                "filename": att.get("filename", ""),
                "gs_url": att.get("gs_url", ""),
                "content_type": att.get("content_type"),
                "size_bytes": att.get("size_bytes"),
            }
            for att in attachments
        ]
    if tags:
        body["tags"] = tags
    async with aiohttp.ClientSession() as session:
        async with session.put(
            f"{orchestra_url}/admin/messages/{api_message_id}/complete",
            headers=headers,
            json=body,
        ) as resp:
            try:
                resp.raise_for_status()
            except Exception as e:
                LOGGER.error(
                    f"{ICONS['comms_outbound']} Failed to complete API message: {e}",
                )
                return {"success": False}
            return {"success": True}


async def publish_assistant_desktop_ready(
    binding_id: str,
    desktop_url: str,
    liveview_url: str,
    vm_type: str,
) -> None:
    """Publish desktop-ready notification to the assistant's Pub/Sub topic.

    The Console subscribes to this thread to update the liveview iframe.
    """
    agent_id = SESSION_DETAILS.assistant.agent_id
    if _use_local_comms():
        try:
            await _publish_local_outbox_async(
                {
                    "thread": "assistant_desktop_ready",
                    "event": {
                        "binding_id": binding_id,
                        "desktop_url": desktop_url,
                        "liveview_url": liveview_url,
                        "vm_type": vm_type,
                    },
                },
            )
        except Exception as e:
            LOGGER.error(
                f"{ICONS['comms_outbound']} Error publishing assistant_desktop_ready: {e}",
            )
        return

    try:
        _publish_to_assistant_topic(
            agent_id=agent_id,
            thread="assistant_desktop_ready",
            event={
                "binding_id": binding_id,
                "desktop_url": desktop_url,
                "liveview_url": liveview_url,
                "vm_type": vm_type,
            },
        )
        env_suffix = SETTINGS.ENV_SUFFIX if agent_id is not None else ""
        LOGGER.debug(
            f"{ICONS['comms_outbound']} Published assistant_desktop_ready to "
            f"unity-{agent_id}{env_suffix}",
        )
    except Exception as e:
        LOGGER.error(
            f"{ICONS['comms_outbound']} Error publishing assistant_desktop_ready: {e}",
        )


async def upload_unify_attachment(
    file_content: bytes,
    filename: str,
    assistant_id: int | None = None,
) -> dict:
    """
    Upload a file attachment for use in outbound Unify messages.

    Args:
        file_content: The raw bytes of the file to upload.
        filename: The name of the file.
        assistant_id: Optional assistant ID for organizing storage.

    Returns:
        dict with attachment details: {"id": str, "filename": str, "url": str}
        or {"success": False, "error": str} on failure.
    """
    if assistant_id is None:
        assistant_id = SESSION_DETAILS.assistant.agent_id

    from io import BytesIO

    adapters_url = _gateway_adapters_base_url()

    LOGGER.debug(
        f"{ICONS['comms_outbound']} Uploading unify attachment: {filename} ({len(file_content)} bytes)",
    )

    # Create form data for multipart upload
    form_data = aiohttp.FormData()
    form_data.add_field(
        "file",
        BytesIO(file_content),
        filename=filename,
        content_type="application/octet-stream",
    )
    form_data.add_field("assistant_id", str(assistant_id))

    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{adapters_url}/unify/attachment",
            headers=headers,
            data=form_data,
        ) as response:
            try:
                body = await response.text()
                if response.status >= 400:
                    try:
                        detail = json.loads(body).get("error", body)
                    except (json.JSONDecodeError, AttributeError):
                        detail = body
                    error_msg = f"Upload rejected ({response.status}): {detail}"
                    LOGGER.debug(
                        f"{ICONS['comms_outbound']} Failed to upload unify attachment: {error_msg}",
                    )
                    return {"success": False, "error": error_msg}
                result = json.loads(body)
                LOGGER.debug(f"{ICONS['comms_outbound']} Uploaded attachment: {result}")
                return result
            except Exception as e:
                LOGGER.debug(
                    f"{ICONS['comms_outbound']} Failed to upload unify attachment: {e}",
                )
                return {"success": False, "error": str(e)}


async def send_discord_message(
    to: str | None = None,
    channel_id: str | None = None,
    body: str = "",
    bot_id: str | None = None,
    media_url: str | None = None,
) -> dict:
    """Send a Discord message via the Communication service.

    Supports two modes:
    - DM: pass ``to`` (Discord user snowflake). The route is resolved
      server-side and a DM channel is opened automatically.
    - Channel reply: pass ``channel_id`` and ``bot_id``. The message is
      posted directly to the channel.

    Args:
        to: Recipient's Discord user ID (for DMs).
        channel_id: Discord channel ID (for channel replies).
        body: The text content to send.
        bot_id: Pool bot ID (required for channel messages, optional for DMs).
        media_url: Optional URL of a media attachment to embed.

    Returns:
        dict with 'success' key and optionally 'message_id', 'channel_id'.
    """
    agent_id = SESSION_DETAILS.assistant.agent_id
    if agent_id is None:
        return {"success": False}

    body = normalize_outbound_plain_text(body)

    payload: dict = {
        "body": body,
        "assistant_id": agent_id,
    }
    if to:
        payload["to"] = to
    if channel_id:
        payload["channel_id"] = channel_id
    if bot_id:
        payload["bot_id"] = bot_id
    if media_url:
        payload["media_url"] = media_url

    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{SETTINGS.conversation.COMMS_URL}/discord/send",
            headers=headers,
            json=payload,
        ) as response:
            try:
                response.raise_for_status()
            except Exception as e:
                LOGGER.error(f"{ICONS['comms_outbound']} Discord send failed: {e}")
                return {"success": False}
            result = await response.json()
            result["success"] = True
            return result


async def send_slack_message(
    *,
    team_id: str,
    channel_id: str | None = None,
    user_id: str | None = None,
    body: str = "",
    thread_ts: str | None = None,
) -> dict:
    """Send a Slack message via the Communication service.

    Supports two modes:

    - **DM**: pass ``user_id`` (Slack user ID). Communication opens a
      DM conversation with the user on behalf of the workspace's bot
      token and posts the message.
    - **Channel post / threaded reply**: pass ``channel_id``
      (and optionally ``thread_ts`` to reply inside an existing thread).

    The workspace bot token is resolved server-side from ``team_id``;
    the assistant never sees it.

    Args:
        team_id: Slack workspace ID (used to resolve the bot token).
        channel_id: Slack channel ID (for channel posts / threaded replies).
        user_id: Slack user ID (for DMs).
        body: The text content to send.
        thread_ts: Slack thread timestamp (omit for top-level posts).

    Returns:
        dict with ``success`` and optionally ``message_ts`` / ``channel_id``.
    """
    agent_id = SESSION_DETAILS.assistant.agent_id
    if agent_id is None:
        return {"success": False}

    body = normalize_outbound_plain_text(body)

    payload: dict = {
        "team_id": team_id,
        "body": body,
        "assistant_id": agent_id,
    }
    if user_id:
        payload["user_id"] = user_id
    if channel_id:
        payload["channel_id"] = channel_id
    if thread_ts:
        payload["thread_ts"] = thread_ts

    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{SETTINGS.conversation.COMMS_URL}/slack/send",
            headers=headers,
            json=payload,
        ) as response:
            try:
                response.raise_for_status()
            except Exception as e:
                LOGGER.error(f"{ICONS['comms_outbound']} Slack send failed: {e}")
                return {"success": False}
            result = await response.json()
            result["success"] = True
            return result


async def resolve_slack_user_profile(
    *,
    team_id: str,
    slack_user_id: str,
) -> dict:
    """Look up a Slack user's profile via the Communication gateway.

    Returns ``{slack_user_id, email, real_name, display_name, tz}`` (any
    value may be ``None``), or an empty dict on failure. Callers treat a
    missing/empty result as "unresolved" and fall back to other
    resolution strategies. ``email`` is only present when the workspace
    bot has the ``users:read.email`` scope; names need only ``users:read``.
    """
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{SETTINGS.conversation.COMMS_URL}/slack/user-info",
            headers=headers,
            json={"team_id": team_id, "slack_user_id": slack_user_id},
        ) as response:
            try:
                response.raise_for_status()
            except Exception as e:
                LOGGER.error(f"{ICONS['comms_outbound']} Slack user-info failed: {e}")
                return {}
            return await response.json()


async def resolve_slack_user_id_by_email(
    *,
    team_id: str,
    email: str,
) -> str | None:
    """Resolve a Slack user ID from an email via the Communication gateway.

    The reverse of ``resolve_slack_user_profile``: given a contact's email
    it returns their Slack user ID (via ``users.lookupByEmail``), so an
    assistant can DM a workspace member it has never received a message
    from. Returns ``None`` when the workspace has no member with that email,
    when the bot lacks the ``users:read.email`` scope, or on any failure —
    callers treat ``None`` as "unresolved" and fall back to their existing
    behaviour.
    """
    if not email:
        return None
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{SETTINGS.conversation.COMMS_URL}/slack/user-by-email",
            headers=headers,
            json={"team_id": team_id, "email": email},
        ) as response:
            try:
                response.raise_for_status()
            except Exception as e:
                LOGGER.error(
                    f"{ICONS['comms_outbound']} Slack user-by-email failed: {e}",
                )
                return None
            result = await response.json()
            return result.get("slack_user_id") or None


async def send_teams_message(
    chat_id: str | None = None,
    team_id: str | None = None,
    channel_id: str | None = None,
    body: str = "",
    content_type: str = "text",
    attachments: list[dict] | None = None,
) -> dict:
    """Send a Microsoft Teams message via the Communication service.

    Routes to the appropriate endpoint based on the parameters:
    - Chat (1:1, group, meeting): pass ``chat_id`` → POST /teams/send
    - Channel: pass ``team_id`` and ``channel_id`` → POST /teams/channel/{team_id}/{channel_id}/send

    Args:
        chat_id: Teams chat ID (for chat messages).
        team_id: Teams team ID (for channel messages).
        channel_id: Teams channel ID (for channel messages).
        body: The text content to send.
        content_type: "text" (default) or "html".
        attachments: Optional list of attachment dicts, each with
            ``filename`` and ``content_base64`` keys. Communication
            handles the OneDrive upload.

    Returns:
        dict with 'success' key and optionally 'message_id'.
    """
    from_email = SESSION_DETAILS.assistant.email
    if not from_email:
        return {"success": False, "error": "No sender email configured"}

    if content_type == "text":
        body = normalize_outbound_plain_text(body)

    payload: dict = {
        "from": from_email,
        "body": body,
        "content_type": content_type,
    }
    if attachments:
        payload["attachments"] = attachments

    is_channel = bool(team_id and channel_id)
    if is_channel:
        url = f"{SETTINGS.conversation.COMMS_URL}/teams/channel/{team_id}/{channel_id}/send"
    else:
        payload["chat_id"] = chat_id
        url = f"{SETTINGS.conversation.COMMS_URL}/teams/send"

    target = "channel" if is_channel else "chat"
    LOGGER.info(
        f"{ICONS['comms_outbound']} Teams {target} send → POST {url} "
        f"from={from_email} chat_id={chat_id or ''} "
        f"team_id={team_id or ''} channel_id={channel_id or ''} "
        f"body_len={len(body or '')} attachments={len(attachments or [])}",
    )
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as response:
                try:
                    response.raise_for_status()
                except Exception as e:
                    body_text = ""
                    try:
                        body_text = (await response.text())[:500]
                    except Exception:
                        pass
                    error_msg = (
                        f"HTTP {response.status}: {body_text}"
                        if body_text
                        else f"HTTP {response.status}: {e}"
                    )
                    LOGGER.error(
                        f"{ICONS['comms_outbound']} Teams {target} send failed: {error_msg}",
                    )
                    return {"success": False, "error": error_msg}
                result = await response.json()
                result["success"] = True
                return result
    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        LOGGER.error(
            f"{ICONS['comms_outbound']} Teams {target} request failed before response: {error_msg}",
        )
        return {"success": False, "error": error_msg}


async def create_teams_chat(
    chat_type: str,
    member_emails: list[str],
    topic: str | None = None,
) -> dict:
    """Create (or return the existing dedup'd) Microsoft Teams chat.

    POSTs to the Communication ``/teams/chats`` endpoint which wraps Graph
    ``POST /chats``. Graph dedupes ``oneOnOne`` chats with the same member
    pair, so repeat calls return the same ``chat_id``.

    Args:
        chat_type: ``"oneOnOne"`` for a 1:1 DM or ``"group"`` for a group chat.
        member_emails: Participant UPNs (emails) excluding the assistant
            sender — the server adds the sender implicitly.
        topic: Optional topic for ``"group"`` chats; rejected server-side
            for ``"oneOnOne"``.

    Returns:
        dict with ``success`` bool and, on success, ``chat_id`` / ``chat_type``.
    """
    from_email = SESSION_DETAILS.assistant.email
    if not from_email:
        return {"success": False, "error": "No sender email configured"}

    payload: dict = {
        "from": from_email,
        "chat_type": chat_type,
        "members": list(member_emails),
    }
    if topic:
        payload["topic"] = topic

    url = f"{SETTINGS.conversation.COMMS_URL}/teams/chats"
    LOGGER.info(
        f"{ICONS['comms_outbound']} Teams chat create → POST {url} "
        f"from={from_email} chat_type={chat_type} members={len(member_emails)} "
        f"topic={'yes' if topic else 'no'}",
    )
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as response:
                try:
                    response.raise_for_status()
                except Exception as e:
                    body_text = ""
                    try:
                        body_text = (await response.text())[:500]
                    except Exception:
                        pass
                    error_msg = (
                        f"HTTP {response.status}: {body_text}"
                        if body_text
                        else f"HTTP {response.status}: {e}"
                    )
                    LOGGER.error(
                        f"{ICONS['comms_outbound']} Teams chat create failed: {error_msg}",
                    )
                    return {"success": False, "error": error_msg}
                result = await response.json()
                result["success"] = True
                return result
    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        LOGGER.error(
            f"{ICONS['comms_outbound']} Teams chat create request failed before response: {error_msg}",
        )
        return {"success": False, "error": error_msg}


async def create_teams_channel(
    team_id: str,
    display_name: str,
    description: str | None = None,
    membership_type: str = "standard",
    owner_emails: list[str] | None = None,
) -> dict:
    """Create a new channel inside an existing Microsoft Teams team.

    POSTs to the Communication ``/teams/channels`` endpoint which wraps Graph
    ``POST /teams/{team-id}/channels``. The communication service rebuilds
    the assistant's Teams watch subscriptions on success so new-channel
    notifications flow immediately.

    Args:
        team_id: ID of the existing team to create the channel within.
        display_name: Channel display name.
        description: Optional channel description.
        membership_type: ``"standard"``, ``"private"``, or ``"shared"``.
            ``"private"`` and ``"shared"`` require at least one owner.
        owner_emails: Required iff ``membership_type != "standard"``. Owner
            UPNs (emails) for the channel.

    Returns:
        dict with ``success`` bool and, on success, ``channel_id`` /
        ``team_id`` / ``membership_type``.
    """
    from_email = SESSION_DETAILS.assistant.email
    if not from_email:
        return {"success": False, "error": "No sender email configured"}

    payload: dict = {
        "from": from_email,
        "team_id": team_id,
        "display_name": display_name,
        "membership_type": membership_type,
    }
    if description:
        payload["description"] = description
    if owner_emails:
        payload["owners"] = list(owner_emails)

    url = f"{SETTINGS.conversation.COMMS_URL}/teams/channels"
    LOGGER.info(
        f"{ICONS['comms_outbound']} Teams channel create → POST {url} "
        f"from={from_email} team_id={team_id} display_name={display_name!r} "
        f"membership_type={membership_type} owners={len(owner_emails or [])}",
    )
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as response:
                try:
                    response.raise_for_status()
                except Exception as e:
                    body_text = ""
                    try:
                        body_text = (await response.text())[:500]
                    except Exception:
                        pass
                    error_msg = (
                        f"HTTP {response.status}: {body_text}"
                        if body_text
                        else f"HTTP {response.status}: {e}"
                    )
                    LOGGER.error(
                        f"{ICONS['comms_outbound']} Teams channel create failed: {error_msg}",
                    )
                    return {"success": False, "error": error_msg}
                result = await response.json()
                result["success"] = True
                return result
    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        LOGGER.error(
            f"{ICONS['comms_outbound']} Teams channel create request failed before response: {error_msg}",
        )
        return {"success": False, "error": error_msg}


async def create_teams_meet(
    *,
    mode: str,
    subject: str | None = None,
    start: str | None = None,
    end: str | None = None,
    timezone: str = "UTC",
    attendees: list[str] | None = None,
    body_html: str | None = None,
    location: str | None = None,
) -> dict:
    """Create a Microsoft Teams meeting via the communication service.

    Mirrors the shipped ``POST /teams/create_meeting`` contract. ``mode`` is
    ``"instant"`` for a reusable ad-hoc meeting (no calendar entry) or
    ``"scheduled"`` for a calendar event with an attached Teams meeting
    (``subject``, ``start``, ``end`` required downstream).

    ``body_html`` is forwarded verbatim and the comms side sends it with
    ``contentType=HTML`` to Graph. ``attendees`` is a list of UPNs — the
    caller resolves contact ids to emails before calling this helper.

    Returns ``{success, join_web_url, meeting_id, event_id, subject, start,
    end, web_link}`` on success (fields are "" when not applicable to
    ``mode``), or ``{success: False, error: ...}`` on failure.
    """
    from_email = SESSION_DETAILS.assistant.email
    if not from_email:
        return {"success": False, "error": "No sender email configured"}

    payload: dict = {
        "assistant_email": from_email,
        "mode": mode,
        "timezone": timezone,
    }
    if subject is not None:
        payload["subject"] = subject
    if start is not None:
        payload["start"] = start
    if end is not None:
        payload["end"] = end
    if attendees:
        payload["attendees"] = list(attendees)
    if body_html is not None:
        payload["body"] = body_html
    if location is not None:
        payload["location"] = location

    url = f"{SETTINGS.conversation.COMMS_URL}/teams/create_meeting"
    LOGGER.info(
        f"{ICONS['comms_outbound']} Teams meeting create → POST {url} "
        f"from={from_email} mode={mode} subject={(subject or '')[:40]!r} "
        f"attendees={len(attendees or [])} "
        f"has_body={'yes' if body_html else 'no'}",
    )
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as response:
                try:
                    response.raise_for_status()
                except Exception as e:
                    body_text = ""
                    try:
                        body_text = (await response.text())[:500]
                    except Exception:
                        pass
                    error_msg = (
                        f"HTTP {response.status}: {body_text}"
                        if body_text
                        else f"HTTP {response.status}: {e}"
                    )
                    LOGGER.error(
                        f"{ICONS['comms_outbound']} Teams meeting create failed: {error_msg}",
                    )
                    return {"success": False, "error": error_msg}
                result = await response.json()
                result["success"] = True
                return result
    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        LOGGER.error(
            f"{ICONS['comms_outbound']} Teams meeting create request failed before response: {error_msg}",
        )
        return {"success": False, "error": error_msg}


async def send_email_via_address(
    to: list[str],
    subject: str,
    body: str,
    cc: list[str] | None = None,
    bcc: list[str] | None = None,
    email_id: str | None = None,
    thread_id: str | None = None,
    attachment: dict | None = None,
) -> dict:
    """
    Send an email using the email provider API.

    Args:
        to: List of recipient email addresses.
        subject: The subject of the email.
        body: The message body to send.
        cc: Optional list of CC email addresses.
        bcc: Optional list of BCC email addresses.
        email_id: The email identifier of the message to reply to (threading id).
        thread_id: Provider thread identifier to target.
        attachment: Optional attachment dict with keys:
            - filename: The name of the file
            - content_base64: Base64-encoded file contents

    Returns:
        dict: Response with 'success' bool and optionally 'error' message
    """
    from_email = SESSION_DETAILS.assistant.email
    if not from_email:
        return {"success": False, "error": "No sender email configured"}

    body = normalize_outbound_plain_text(body)
    from_name = (SESSION_DETAILS.assistant.name or "").strip()

    payload = {
        "from": from_email,
        "to": to,
        "subject": subject,
        "body": body,
        "in_reply_to": email_id,
    }
    if from_name:
        payload["from_name"] = from_name
    if thread_id:
        payload["thread_id"] = thread_id
    if cc:
        payload["cc"] = cc
    if bcc:
        payload["bcc"] = bcc
    if attachment:
        payload["attachment"] = attachment

    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{_gateway_comms_base_url()}/email/send",
            headers=headers,
            json=payload,
        ) as response:
            try:
                response.raise_for_status()
            except Exception as e:
                return {"success": False, "error": str(e)}
            return await response.json()


async def start_call(to_number: str) -> str:
    """
    Send a call using the call provider API.

    Args:
        to_number: The recipient's phone number

    Returns:
        str: The response
    """
    from_number = SESSION_DETAILS.assistant.number
    if not from_number:
        return {"success": False}

    from unify.conversation_manager.domains.call_manager import make_room_name

    assistant_id = str(SESSION_DETAILS.assistant.agent_id)
    room_name = make_room_name(assistant_id, "phone")

    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{_gateway_comms_base_url()}/phone/send-call",
            headers=headers,
            json={
                "From": from_number,
                "To": to_number,
                "room_name": room_name,
                "assistant_id": assistant_id,
            },
        ) as response:
            try:
                response.raise_for_status()
            except Exception:
                return {
                    "success": False,
                    "error": f"Failed to initiate call to {to_number}",
                }
            return await response.json()


async def start_whatsapp_call(
    to_number: str,
    agent_name: str,
    room_name: str,
    allow_permission_probe: bool = False,
    pending_call_context: str = "",
) -> dict:
    """
    Initiate a WhatsApp voice call via the Communication service.

    Communication checks call permission with Orchestra and decides the method:
    - Permission granted → places outbound call directly (returns method: "direct")
    - Permission not granted → sends invite template (returns method: "invite")

    Args:
        to_number: The recipient's WhatsApp number (E.164)
        agent_name: Assistant's first name (used in invite template)
        room_name: Pre-built LiveKit room name

    Returns:
        dict with 'success', 'method' ("direct"|"invite"), and other fields.
    """
    agent_id = SESSION_DETAILS.assistant.agent_id
    if agent_id is None:
        return {"success": False}

    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{_gateway_comms_base_url()}/whatsapp/send-call",
            headers=headers,
            json={
                "to": to_number,
                "assistant_id": agent_id,
                "agent_name": agent_name,
                "room_name": room_name,
                "allow_permission_probe": allow_permission_probe,
                "pending_call_context": pending_call_context,
            },
        ) as response:
            try:
                response.raise_for_status()
            except Exception:
                return {
                    "success": False,
                    "error": f"Failed to initiate WhatsApp call to {to_number}",
                }
            return await response.json()


async def end_phone_conference(conference_name: str) -> dict:
    """
    End an active Twilio conference (clean carrier hangup).

    Used to terminate the carrier leg of a phone or WhatsApp call (both use the
    same Twilio conference model). Best-effort: the LiveKit room teardown remains
    the universal session-end mechanism, this just drops the PSTN/WhatsApp leg
    cleanly when the conference name is known.

    Args:
        conference_name: The Twilio conference friendly name.

    Returns:
        dict with 'success' and any provider response fields.
    """
    if not conference_name:
        return {"success": False, "error": "no conference_name"}

    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{_gateway_comms_base_url()}/phone/end-conference",
            headers=headers,
            json={"ConferenceName": conference_name},
        ) as response:
            try:
                response.raise_for_status()
            except Exception:
                return {
                    "success": False,
                    "error": f"Failed to end conference {conference_name}",
                }
            return await response.json()


async def hang_up_call(call_sid: str) -> dict:
    """
    End a single Twilio call by SID (clean carrier hangup, no conference).

    Used for outbound calls, which are bridged via a direct ``<Dial>`` rather
    than a Twilio conference: completing the parent SIP call leg collapses the
    dial and disconnects the remote party deterministically (instead of relying
    on the LiveKit room teardown propagating a SIP BYE).

    Args:
        call_sid: The Twilio call SID to terminate.

    Returns:
        dict with 'success' and any provider response fields.
    """
    if not call_sid:
        return {"success": False, "error": "no call_sid"}

    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{_gateway_comms_base_url()}/phone/hang-up-call",
            headers=headers,
            json={"CallSid": call_sid},
        ) as response:
            try:
                response.raise_for_status()
            except Exception:
                return {
                    "success": False,
                    "error": f"Failed to hang up call {call_sid}",
                }
            return await response.json()


async def store_pending_whatsapp_call_intent(
    *,
    pool_number: str,
    contact_number: str,
    context: str,
) -> None:
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{SETTINGS.ORCHESTRA_URL}/admin/whatsapp/pending-call-intent",
            headers=headers,
            json={
                "pool_number": pool_number,
                "contact_number": contact_number,
                "context": context,
            },
        ) as response:
            response.raise_for_status()


async def get_pending_whatsapp_call_intent(
    *,
    pool_number: str,
    contact_number: str,
) -> dict | None:
    async with aiohttp.ClientSession() as session:
        async with session.get(
            f"{SETTINGS.ORCHESTRA_URL}/admin/whatsapp/pending-call-intent",
            headers=headers,
            params={
                "pool_number": pool_number,
                "contact_number": contact_number,
            },
        ) as response:
            if response.status == 404:
                return None
            response.raise_for_status()
            return await response.json()


async def clear_pending_whatsapp_call_intent(
    *,
    pool_number: str,
    contact_number: str,
) -> None:
    async with aiohttp.ClientSession() as session:
        async with session.delete(
            f"{SETTINGS.ORCHESTRA_URL}/admin/whatsapp/pending-call-intent",
            headers=headers,
            params={
                "pool_number": pool_number,
                "contact_number": contact_number,
            },
        ) as response:
            if response.status != 404:
                response.raise_for_status()


async def add_email_attachments(
    attachments: list[dict[str, str]],
    receiver_email: str,
    message_id: str,
) -> None:
    """
    Download email attachments and save to Attachments folder.

    Each attachment item should be of the form: {"id": str, "filename": str}
    """
    if not attachments:
        return

    LOGGER.debug(f"{ICONS['comms_outbound']} Saving email attachments...")
    from unify.manager_registry import ManagerRegistry

    file_manager = ManagerRegistry.get_file_manager()
    saved_display_names: list[str] = []
    async with aiohttp.ClientSession() as session:
        for att in attachments:
            try:
                att_id = att.get("id", "")
                raw_filename = att.get("filename") or f"attachment_{att_id}"
                safe_filename = os.path.basename(raw_filename)
                inline_bytes = _inline_attachment_bytes(att)
                if inline_bytes is not None:
                    data = inline_bytes
                else:
                    url = f"{SETTINGS.conversation.COMMS_URL}/email/attachment"
                    params = {
                        "receiver_email": receiver_email,
                        "message_id": message_id,
                        "attachment_id": att_id,
                    }
                    async with session.get(url, headers=headers, params=params) as resp:
                        data = await resp.read()

                display_name = await asyncio.to_thread(
                    file_manager.save_attachment,
                    att_id,
                    safe_filename,
                    data,
                    auto_ingest=False,
                )
                saved_display_names.append(display_name)

                LOGGER.debug(
                    f"{ICONS['comms_outbound']} Downloaded email attachment {safe_filename} (size={len(data)} bytes)",
                )
            except Exception as e:
                LOGGER.error(
                    f"{ICONS['comms_outbound']} Failed to fetch/write attachment '{att}': {e}",
                )

    if saved_display_names and SETTINGS.file.IMPLICIT_INGESTION:
        try:
            from unify.file_manager.managers.utils.attachment_ingestion import (
                enqueue_attachment_ingestion,
            )

            await asyncio.to_thread(
                enqueue_attachment_ingestion,
                file_manager,
                saved_display_names,
            )
        except Exception as e:
            LOGGER.error(
                f"{ICONS['comms_outbound']} Failed to queue email attachments for ingestion: {e}",
            )


async def _get_signed_url_from_gs_url(
    session: aiohttp.ClientSession,
    gs_url: str,
) -> str:
    """
    Request a signed URL from Orchestra for a gs:// path.

    Args:
        session: aiohttp session for making requests
        gs_url: The gs:// URL to get a signed URL for

    Returns:
        The signed HTTPS URL for downloading the file
    """
    orchestra_url = SETTINGS.ORCHESTRA_URL
    # Use the user's API key (not admin key) for Orchestra API calls
    user_api_key = SESSION_DETAILS.unify_key
    user_headers = {"Authorization": f"Bearer {user_api_key}"}
    async with session.post(
        f"{orchestra_url}/storage/signed-url",
        headers=user_headers,
        json={"gcs_uri": gs_url},
    ) as resp:
        resp.raise_for_status()
        result = await resp.json()
        return result.get("signed_url", "")


async def _download_single_attachment(
    session: aiohttp.ClientSession,
    att: dict[str, str],
    file_manager,
) -> str | None:
    """Download one attachment and write it to disk. Returns the display name, or None on failure."""
    att_id = att.get("id", "")
    raw_filename = att.get("filename") or f"attachment_{att_id}"
    safe_filename = os.path.basename(raw_filename)

    display_name = f"Attachments/{att_id}_{safe_filename}"
    target_path = None
    try:
        adapter = getattr(file_manager, "_adapter", None)
        root = getattr(adapter, "_root", None)
        if root:
            target_path = Path(root) / "Attachments" / f"{att_id}_{safe_filename}"
    except Exception:
        target_path = None
    if (
        target_path is not None
        and target_path.exists()
        and target_path.stat().st_size > 0
    ):
        LOGGER.debug(
            f"{ICONS['comms_outbound']} Attachment {safe_filename} already on disk, skipping download",
        )
        return display_name

    url = att.get("url")
    gs_url = att.get("gs_url")

    if not url and gs_url:
        try:
            url = await _get_signed_url_from_gs_url(session, gs_url)
        except Exception as e:
            LOGGER.error(
                f"{ICONS['comms_outbound']} Failed to get signed URL for {gs_url}: {e}",
            )
            url = None

    inline_bytes = _inline_attachment_bytes(att)

    if inline_bytes is not None:
        data = inline_bytes
    elif url:
        async with session.get(url) as resp:
            resp.raise_for_status()
            data = await resp.read()
    else:
        data = b""

    display_name = await asyncio.to_thread(
        file_manager.save_attachment,
        att_id,
        safe_filename,
        data,
        auto_ingest=False,
    )

    LOGGER.debug(
        f"{ICONS['comms_outbound']} Downloaded unify attachment {safe_filename} "
        f"(size={len(data)} bytes)",
    )
    return display_name


async def add_unify_message_attachments(
    attachments: list[dict[str, str]],
) -> None:
    """
    Download attachments from Unify console messages and save to Attachments folder.

    Each attachment item should be of the form:
        {"id": str, "filename": str, "url": str}
    or with gs_url for on-demand signed URL generation:
        {"id": str, "filename": str, "gs_url": str}

    If gs_url is present but url is not, a signed URL will be generated
    from Orchestra before downloading.

    All downloads run in parallel, then ingestion (parse/index/embed) runs
    afterward so files are immediately available to the assistant.
    """
    if not attachments:
        return

    from unify.manager_registry import ManagerRegistry

    LOGGER.debug(f"{ICONS['comms_outbound']} Saving unify message attachments...")

    file_manager = ManagerRegistry.get_file_manager()

    # Phase 1: Download all files to disk in parallel.
    async with aiohttp.ClientSession() as session:
        results = await asyncio.gather(
            *(
                _download_single_attachment(session, att, file_manager)
                for att in attachments
            ),
            return_exceptions=True,
        )

    saved_display_names: list[str] = []
    for att, result in zip(attachments, results):
        if isinstance(result, BaseException):
            LOGGER.error(
                f"{ICONS['comms_outbound']} Failed to download unify attachment '{att}': {result}",
            )
        elif result is not None:
            saved_display_names.append(result)

    # Phase 2: Queue background ingestion for any saved files.
    if saved_display_names and SETTINGS.file.IMPLICIT_INGESTION:
        try:
            from unify.file_manager.managers.utils.attachment_ingestion import (
                enqueue_attachment_ingestion,
            )

            await asyncio.to_thread(
                enqueue_attachment_ingestion,
                file_manager,
                saved_display_names,
            )
        except Exception as e:
            LOGGER.error(
                f"{ICONS['comms_outbound']} Failed to queue downloaded attachments for ingestion: {e}",
            )
