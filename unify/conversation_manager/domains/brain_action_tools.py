"""
Brain action tools for ConversationManager.

All contact information is fetched from ContactManager (source of truth).
No local caching of contact data.

Context Propagation:
- When `act` is called, the current state snapshot is passed to Actor via _parent_chat_context
- For `interject` operations, only the incremental diff from the initial snapshot is sent
  via _parent_chat_context_cont, avoiding duplication of unchanged state
"""

from __future__ import annotations

import asyncio
from functools import wraps
import inspect
import re
from typing import TYPE_CHECKING, Any, Optional

from pydantic import BaseModel as _BaseModel
from pydantic import create_model as _create_model

from unify.common.prompt_helpers import now as prompt_now
from unify.logger import LOGGER
from unify.common.hierarchical_logger import ICONS
from unify.comms import CommsPrimitives
from unify.session_details import SESSION_DETAILS

from unify.conversation_manager.domains import managers_utils
from unify.conversation_manager.domains.onboarding_tool_gating import (
    masked_reference_quiz_tools,
)
from unify.conversation_manager.event_broker import get_event_broker
from unify.conversation_manager.events import (
    ActorHandleStarted,
    ActorHandleResponse,
    FastBrainNotification,
)
from unify.common._async_tool.dynamic_tools_factory import DynamicToolFactory
from unify.common._async_tool.utils import get_handle_paused_state
from unify.conversation_manager.task_actions import (
    STEERING_OPERATIONS,
    OPERATION_MAP,
    derive_short_name,
    build_action_name,
    safe_call_id_suffix,
)
from unify.conversation_manager.domains.renderer import (
    SnapshotState,
    compute_snapshot_diff,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from unify.conversation_manager.conversation_manager import ConversationManager


# ─────────────────────────────────────────────────────────────────────────────
# Schema dict → Pydantic model conversion
# ─────────────────────────────────────────────────────────────────────────────

_SCHEMA_TYPE_MAP: dict[str, type] = {
    "string": str,
    "str": str,
    "integer": int,
    "int": int,
    "number": float,
    "float": float,
    "boolean": bool,
    "bool": bool,
}


def _resolve_schema_type(schema: Any, name_hint: str) -> type:
    """Resolve a single schema value to a Python type.

    Handles:
    - String type names (``"string"``, ``"integer"``, …)
    - Nested dicts (recursively creates a child Pydantic model)
    - Lists where the first element defines the item schema
    """
    if isinstance(schema, str):
        return _SCHEMA_TYPE_MAP.get(schema.lower(), str)
    if isinstance(schema, dict):
        return schema_dict_to_pydantic(schema, name_hint)
    if isinstance(schema, list) and len(schema) > 0:
        item_type = _resolve_schema_type(schema[0], f"{name_hint}Item")
        return list[item_type]  # type: ignore[valid-type]
    return str  # fallback for unrecognised shapes


def schema_dict_to_pydantic(
    schema: dict,
    model_name: str = "ResponseFormat",
) -> type[_BaseModel]:
    """Convert a simplified schema dict to a dynamic Pydantic model.

    The schema uses a concise, LLM-friendly format:

    - **String values** are type names: ``"string"``, ``"integer"``,
      ``"number"``, ``"boolean"`` (shorthand ``"str"``, ``"int"``, etc.
      also accepted).
    - **Dict values** define nested object schemas (recursively converted).
    - **List values** define array types; the first element is the item
      schema.

    Examples::

        # Flat
        {"email": "string", "age": "integer"}

        # Nested with array
        {"contacts": [{"name": "string", "phone": "string"}], "total": "integer"}
    """
    fields: dict[str, tuple[type, ...]] = {}
    for field_name, field_schema in schema.items():
        field_type = _resolve_schema_type(
            field_schema,
            f"{model_name}{field_name.title()}",
        )
        fields[field_name] = (field_type, ...)
    return _create_model(model_name, **fields)


_next_handle_id = 0


# Pattern matching <in_flight_actions>...</in_flight_actions> sections.
# These contain CM-level steering tools that should not be exposed to the Actor.
_IN_FLIGHT_ACTIONS_PATTERN = re.compile(
    r"<in_flight_actions>.*?</in_flight_actions>\s*",
    re.DOTALL,
)


def _filter_cm_state_for_actor(state_snapshot: dict) -> dict:
    """Filter CM state snapshot before passing to Actor as parent context.

    The CM state snapshot contains <in_flight_actions> with <steering_tools>
    listing CM-level tools (stop_, pause_, interject_, ask_) for each action.
    These are CM brain tools that exist only in the CM's tool surface.

    If passed verbatim to the Actor, the Actor LLM may interpret these tool
    names as callable functions and generate code like:
        await stop_search_the_web_for__1()
    This causes NameError since these tools don't exist in the Actor's scope.

    This function strips the <in_flight_actions> section while preserving
    other useful context (notifications, active_conversations).

    Args:
        state_snapshot: The CM state snapshot dict with "content" key.

    Returns:
        A filtered copy of the snapshot with in_flight_actions removed.
    """
    if not state_snapshot:
        return state_snapshot

    content = state_snapshot.get("content", "")
    if not content:
        return state_snapshot

    # When screenshots are attached, content is a list of multimodal parts
    # rather than a plain string. Apply the regex to each text part.
    if isinstance(content, list):
        filtered_parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                filtered_text = _IN_FLIGHT_ACTIONS_PATTERN.sub("", part["text"])
                filtered_parts.append({**part, "text": filtered_text})
            else:
                filtered_parts.append(part)
        return {**state_snapshot, "content": filtered_parts}

    filtered_content = _IN_FLIGHT_ACTIONS_PATTERN.sub("", content)
    return {**state_snapshot, "content": filtered_content}


class _DesktopActionHandle:
    """Lightweight handle wrapping a single desktop primitive call.

    Provides the minimal interface consumed by the CM watcher infrastructure
    (``actor_watch_result``) so desktop fast-path actions participate in the
    same in-flight lifecycle as ``act`` and the contact/transcript fast paths.

    Steering operations (pause/resume/interject/ask) are no-ops because these
    are atomic single-step actions with no inner loop to steer.
    """

    def __init__(self, task: asyncio.Task):
        self._task = task
        self._notification_q: asyncio.Queue = asyncio.Queue()

    def done(self) -> bool:
        return self._task.done()

    async def result(self) -> str:
        return await self._task

    async def next_notification(self) -> dict:
        while not self._task.done():
            try:
                return await asyncio.wait_for(self._notification_q.get(), timeout=30)
            except asyncio.TimeoutError:
                continue
        raise asyncio.CancelledError

    async def next_clarification(self) -> dict:
        while not self._task.done():
            await asyncio.sleep(30)
        raise asyncio.CancelledError

    async def stop(self, reason=None, **kwargs):
        self._task.cancel()

    async def interject(self, message, **kwargs):
        pass

    async def ask(self, question, **kwargs):
        pass

    async def pause(self):
        pass

    async def resume(self):
        pass


class ConversationManagerBrainActionTools:
    """
    Side-effecting tools for the Main CM Brain.

    All contact data is fetched from ContactManager - no local caching.
    """

    def __init__(self, cm: "ConversationManager"):
        self._cm = cm
        self._event_broker = get_event_broker()
        self._comms = CommsPrimitives(
            conversation_manager=cm,
            event_broker=self._event_broker,
        )

    def _boss_contact_id(self) -> int:
        return int(SESSION_DETAILS.boss_contact_id)

    @wraps(CommsPrimitives.send_sms)
    async def send_sms(
        self,
        *,
        contact_id: int | str,
        content: str,
        phone_number: str | None = None,
    ) -> dict[str, Any]:
        return await self._comms.send_sms(
            contact_id=contact_id,
            content=content,
            phone_number=phone_number,
        )

    async def send_sms_to_boss(
        self,
        *,
        content: str,
    ) -> dict[str, Any]:
        """Send an SMS message directly to my boss only.

        This Coordinator direct communication tool is restricted to the boss
        contact (``contact_id==1`` in the normal runtime). It cannot be used
        to text anyone else. If my boss asks me to draft or send an SMS on
        their behalf, route that work through ``act`` instead.

        Parameters
        ----------
        content : str
            Message body to send to my boss.
        """
        return await self._comms.send_sms(
            contact_id=self._boss_contact_id(),
            content=content,
        )

    @wraps(CommsPrimitives.send_whatsapp)
    async def send_whatsapp(
        self,
        *,
        contact_id: int | str,
        content: str,
        whatsapp_number: str | None = None,
        attachment_filepath: str | None = None,
    ) -> dict[str, Any]:
        return await self._comms.send_whatsapp(
            contact_id=contact_id,
            content=content,
            whatsapp_number=whatsapp_number,
            attachment_filepath=attachment_filepath,
        )

    async def send_whatsapp_to_boss(
        self,
        *,
        content: str,
        attachment_filepath: str | None = None,
    ) -> dict[str, Any]:
        """Send a WhatsApp message directly to my boss only.

        This Coordinator direct communication tool is restricted to the boss
        contact (``contact_id==1`` in the normal runtime). It cannot be used
        to WhatsApp anyone else. If my boss asks me to draft or send a
        WhatsApp message on their behalf, route that work through ``act``
        instead.

        Delivery is NOT confirmed in the turn I call this tool. If the
        24-hour WhatsApp window is closed, my boss receives only a generic
        template placeholder (not my verbatim text), and the real body is
        queued to resend after they reply. I will not know which happened until
        it surfaces as a ``[You WhatsApped <name>]`` row (delivered) or a
        ``[You WhatsApped <name> (not delivered directly)]`` row (placeholder
        only). So in the same turn I send, I say only that I am *sending* it —
        never that it arrived, is waiting, or is in their inbox — and I confirm
        receipt only once that proof row appears. If it is
        ``(not delivered directly)``, I tell them a placeholder went out and
        they must reply to it before I can deliver the real content.

        Parameters
        ----------
        content : str
            Message body to send to my boss.
        attachment_filepath : str | None, optional
            Workspace-relative path for one attachment.
        """
        return await self._comms.send_whatsapp(
            contact_id=self._boss_contact_id(),
            content=content,
            attachment_filepath=attachment_filepath,
        )

    @wraps(CommsPrimitives.send_discord_message)
    async def send_discord_message(
        self,
        *,
        contact_id: int | str,
        content: str,
        discord_id: str | None = None,
    ) -> dict[str, Any]:
        return await self._comms.send_discord_message(
            contact_id=contact_id,
            content=content,
            discord_id=discord_id,
        )

    async def send_discord_message_to_boss(
        self,
        *,
        content: str,
    ) -> dict[str, Any]:
        """Send a Discord direct message to my boss only.

        This Coordinator direct communication tool is restricted to the boss
        contact (``contact_id==1`` in the normal runtime). It cannot be used
        to DM anyone else. If my boss asks me to draft or send a Discord
        message on their behalf, route that work through ``act`` instead.

        Parameters
        ----------
        content : str
            Message body to send to my boss.
        """
        return await self._comms.send_discord_message(
            contact_id=self._boss_contact_id(),
            content=content,
        )

    @wraps(CommsPrimitives.send_discord_channel_message)
    async def send_discord_channel_message(
        self,
        *,
        channel_id: str,
        content: str,
        guild_id: str = "",
        contact_id: int | str | None = None,
    ) -> dict[str, Any]:
        return await self._comms.send_discord_channel_message(
            channel_id=channel_id,
            content=content,
            guild_id=guild_id,
            contact_id=contact_id,
        )

    @wraps(CommsPrimitives.send_slack_message)
    async def send_slack_message(
        self,
        *,
        contact_id: int | str,
        content: str,
        team_id: str,
        slack_user_id: str | None = None,
        thread_ts: str | None = None,
    ) -> dict[str, Any]:
        return await self._comms.send_slack_message(
            contact_id=contact_id,
            content=content,
            team_id=team_id,
            slack_user_id=slack_user_id,
            thread_ts=thread_ts,
        )

    async def send_slack_message_to_boss(
        self,
        *,
        content: str,
        team_id: str,
        thread_ts: str | None = None,
    ) -> dict[str, Any]:
        """Send a Slack direct message to my boss only.

        This Coordinator direct communication tool is restricted to the boss
        contact (``contact_id==1`` in the normal runtime). It cannot be used
        to DM anyone else or post in channels. If my boss asks me to draft or
        send Slack messages on their behalf, route that work through ``act``
        instead.

        Parameters
        ----------
        content : str
            Message body to send to my boss.
        team_id : str
            Slack workspace/team ID from the inbound boss DM annotation.
        thread_ts : str | None, optional
            Existing boss DM thread timestamp to reply inside.
        """
        return await self._comms.send_slack_message(
            contact_id=self._boss_contact_id(),
            content=content,
            team_id=team_id,
            thread_ts=thread_ts,
        )

    @wraps(CommsPrimitives.send_slack_channel_message)
    async def send_slack_channel_message(
        self,
        *,
        channel_id: str,
        content: str,
        team_id: str,
        thread_ts: str | None = None,
        contact_id: int | str | None = None,
    ) -> dict[str, Any]:
        return await self._comms.send_slack_channel_message(
            channel_id=channel_id,
            content=content,
            team_id=team_id,
            thread_ts=thread_ts,
            contact_id=contact_id,
        )

    @wraps(CommsPrimitives.send_teams_message)
    async def send_teams_message(
        self,
        *,
        contact_id: int | str | list[int | str | dict],
        content: str,
        chat_id: str | None = None,
        channel_id: str | None = None,
        team_id: str | None = None,
        chat_topic: str | None = None,
        attachment_filepath: str | None = None,
    ) -> dict[str, Any]:
        return await self._comms.send_teams_message(
            contact_id=contact_id,
            content=content,
            chat_id=chat_id,
            channel_id=channel_id,
            team_id=team_id,
            chat_topic=chat_topic,
            attachment_filepath=attachment_filepath,
        )

    async def send_teams_message_to_boss(
        self,
        *,
        content: str,
        chat_id: str | None = None,
        attachment_filepath: str | None = None,
    ) -> dict[str, Any]:
        """Send a Microsoft Teams message directly to my boss only.

        This Coordinator direct communication tool is restricted to the boss
        contact (``contact_id==1`` in the normal runtime). It cannot be used
        to start group chats or post in Teams channels. If my boss asks me to
        draft or send Teams messages on their behalf, route that work through
        ``act`` instead.

        Parameters
        ----------
        content : str
            Message body to send to my boss.
        chat_id : str | None, optional
            Existing boss Teams chat ID to reply inside. Omit to create or
            reuse a 1:1 chat with my boss.
        attachment_filepath : str | None, optional
            Workspace-relative path for one attachment.
        """
        return await self._comms.send_teams_message(
            contact_id=self._boss_contact_id(),
            content=content,
            chat_id=chat_id,
            attachment_filepath=attachment_filepath,
        )

    @wraps(CommsPrimitives.create_teams_channel)
    async def create_teams_channel(
        self,
        *,
        team_id: str,
        display_name: str,
        description: str | None = None,
        membership_type: str = "standard",
        owner_contact_ids: list[int | str | dict] | None = None,
    ) -> dict[str, Any]:
        return await self._comms.create_teams_channel(
            team_id=team_id,
            display_name=display_name,
            description=description,
            membership_type=membership_type,
            owner_contact_ids=owner_contact_ids,
        )

    @wraps(CommsPrimitives.create_teams_meet)
    async def create_teams_meet(
        self,
        *,
        mode: str = "scheduled",
        subject: str | None = None,
        start: str | None = None,
        duration_minutes: int = 30,
        timezone: str = "UTC",
        attendee_contact_ids: list[int | str | dict] | None = None,
        body_html: str | None = None,
        location: str | None = None,
    ) -> dict[str, Any]:
        return await self._comms.create_teams_meet(
            mode=mode,
            subject=subject,
            start=start,
            duration_minutes=duration_minutes,
            timezone=timezone,
            attendee_contact_ids=attendee_contact_ids,
            body_html=body_html,
            location=location,
        )

    async def create_teams_meet_with_boss(
        self,
        *,
        mode: str = "scheduled",
        subject: str | None = None,
        start: str | None = None,
        duration_minutes: int = 30,
        timezone: str = "UTC",
        body_html: str | None = None,
        location: str | None = None,
    ) -> dict[str, Any]:
        """Create a Microsoft Teams meeting with my boss only.

        This Coordinator direct communication tool is restricted to the boss
        contact (``contact_id==1`` in the normal runtime). Scheduled meetings
        invite only my boss. If my boss asks me to schedule or host meetings
        with other people on their behalf, route that work through ``act``
        instead.

        Parameters
        ----------
        mode : str, optional
            ``"scheduled"`` (default) or ``"instant"``.
        subject : str | None, optional
            Meeting subject. Required for scheduled meetings.
        start : str | None, optional
            ISO-8601 start timestamp for scheduled meetings.
        duration_minutes : int, optional
            Meeting duration in minutes for scheduled meetings.
        timezone : str, optional
            Timezone name forwarded to Microsoft Graph.
        body_html : str | None, optional
            Meeting body, sent to Graph as HTML.
        location : str | None, optional
            Display-name location for the calendar event.
        """
        return await self._comms.create_teams_meet(
            mode=mode,
            subject=subject,
            start=start,
            duration_minutes=duration_minutes,
            timezone=timezone,
            attendee_contact_ids=(
                [self._boss_contact_id()] if mode == "scheduled" else None
            ),
            body_html=body_html,
            location=location,
        )

    @wraps(CommsPrimitives.send_unify_message)
    async def send_unify_message(
        self,
        *,
        content: str,
        contact_id: int | str,
        attachment_filepath: str | None = None,
    ) -> dict[str, Any]:
        return await self._comms.send_unify_message(
            content=content,
            contact_id=contact_id,
            attachment_filepath=attachment_filepath,
        )

    async def send_unify_message_to_boss(
        self,
        *,
        content: str,
        attachment_filepath: str | None = None,
    ) -> dict[str, Any]:
        """Send a Unify inbox message directly to my boss only.

        This Coordinator direct communication tool is restricted to the boss
        contact (``contact_id==1`` in the normal runtime). It cannot be used
        to message anyone else. If my boss asks me to draft or send Unify
        messages on their behalf, route that work through ``act`` instead.

        Parameters
        ----------
        content : str
            Message body to send to my boss.
        attachment_filepath : str | None, optional
            Workspace-relative path for one attachment.
        """
        return await self._comms.send_unify_message(
            content=content,
            contact_id=self._boss_contact_id(),
            attachment_filepath=attachment_filepath,
        )

    @wraps(CommsPrimitives.send_api_response)
    async def send_api_response(
        self,
        *,
        content: str,
        contact_id: int | str = 1,
        attachment_filepaths: list[str] | None = None,
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        return await self._comms.send_api_response(
            content=content,
            contact_id=contact_id,
            attachment_filepaths=attachment_filepaths,
            tags=tags,
        )

    async def send_api_response_to_boss(
        self,
        *,
        content: str,
        attachment_filepaths: list[str] | None = None,
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        """Reply to the pending API message with the boss as transcript anchor.

        This Coordinator direct communication tool never exposes a recipient
        choice. The response is anchored to the boss contact
        (``contact_id==1`` in the normal runtime). If my boss asks me to
        perform communication work on their behalf, route that work through
        ``act`` instead.

        Parameters
        ----------
        content : str
            Response text to send back to the waiting API caller.
        attachment_filepaths : list[str] | None, optional
            Workspace-local file paths to upload and attach to the API response.
        tags : list[str] | None, optional
            API routing tags to return; omit to echo inbound tags.
        """
        return await self._comms.send_api_response(
            content=content,
            contact_id=self._boss_contact_id(),
            attachment_filepaths=attachment_filepaths,
            tags=tags,
        )

    @wraps(CommsPrimitives.send_email)
    async def send_email(
        self,
        *,
        to: list[int | dict] | None = None,
        cc: list[int | dict] | None = None,
        bcc: list[int | dict] | None = None,
        subject: str,
        body: str,
        reply_all: bool = False,
        email_id_to_reply_to: str | None = None,
        thread_id: str | None = None,
        attachment_filepath: str | None = None,
    ) -> dict[str, Any]:
        return await self._comms.send_email(
            to=to,
            cc=cc,
            bcc=bcc,
            subject=subject,
            body=body,
            reply_all=reply_all,
            email_id_to_reply_to=email_id_to_reply_to,
            thread_id=thread_id,
            attachment_filepath=attachment_filepath,
        )

    async def send_email_to_boss(
        self,
        *,
        subject: str,
        body: str,
        email_id_to_reply_to: str | None = None,
        thread_id: str | None = None,
        attachment_filepath: str | None = None,
    ) -> dict[str, Any]:
        """Send an email directly to my boss only.

        This Coordinator direct communication tool is restricted to the boss
        contact (``contact_id==1`` in the normal runtime). It cannot be used
        to email anyone else, cc anyone else, bcc anyone else, or reply-all to
        a thread. If my boss asks me to draft or send email on their behalf,
        route that work through ``act`` instead.

        Parameters
        ----------
        subject : str
            Subject line for the email to my boss.
        body : str
            Email body content to send to my boss.
        email_id_to_reply_to : str | None, optional
            Existing boss email ID to reply to for threading.
        thread_id : str | None, optional
            Provider thread identifier shown on the inbound email.
        attachment_filepath : str | None, optional
            Workspace-relative path for one attachment.
        """
        return await self._comms.send_email(
            to=[self._boss_contact_id()],
            subject=subject,
            body=body,
            email_id_to_reply_to=email_id_to_reply_to,
            thread_id=thread_id,
            attachment_filepath=attachment_filepath,
        )

    @wraps(CommsPrimitives.make_call)
    async def make_call(
        self,
        *,
        contact_id: int | str,
        context: str,
        phone_number: str | None = None,
    ) -> dict[str, Any]:
        return await self._comms.make_call(
            contact_id=contact_id,
            context=context,
            phone_number=phone_number,
        )

    async def make_call_to_boss(
        self,
        *,
        context: str,
    ) -> dict[str, Any]:
        """Start an outbound phone call to my boss only.

        This Coordinator direct communication tool is restricted to the boss
        contact (``contact_id==1`` in the normal runtime). It cannot be used
        to call anyone else. If my boss asks me to call someone on their
        behalf, route that work through ``act`` instead.

        Parameters
        ----------
        context : str
            Mission briefing for the voice agent speaking with my boss.
        """
        return await self._comms.make_call(
            contact_id=self._boss_contact_id(),
            context=context,
        )

    @wraps(CommsPrimitives.make_whatsapp_call)
    async def make_whatsapp_call(
        self,
        *,
        contact_id: int | str,
        context: str,
        whatsapp_number: str | None = None,
    ) -> dict[str, Any]:
        return await self._comms.make_whatsapp_call(
            contact_id=contact_id,
            context=context,
            whatsapp_number=whatsapp_number,
        )

    async def make_whatsapp_call_to_boss(
        self,
        *,
        context: str,
    ) -> dict[str, Any]:
        """Start a WhatsApp voice call to my boss only.

        This Coordinator direct communication tool is restricted to the boss
        contact (``contact_id==1`` in the normal runtime). It cannot be used
        to call anyone else. If my boss asks me to call someone on their
        behalf, route that work through ``act`` instead.

        Parameters
        ----------
        context : str
            Mission briefing for the voice agent speaking with my boss.
        """
        return await self._comms.make_whatsapp_call(
            contact_id=self._boss_contact_id(),
            context=context,
        )

    async def join_google_meet(
        self,
        meet_url: str,
        context: str = "",
    ) -> dict[str, Any]:
        """Join a Google Meet call.

        This is the **only** way to join a Google Meet — it configures audio
        devices, establishes the voice pipeline, and dispatches the voice
        agent so the assistant can hear and speak in the meeting.  Never
        attempt to join a Meet URL via ``act``.

        Args:
            meet_url: The full Google Meet URL (e.g. https://meet.google.com/abc-defg-hij).
            context: Briefing for the voice agent about the meeting purpose,
                expected participants, topics to cover, and how to behave.
                Same guidance principles as make_call's context parameter.
        """
        if (
            self._cm.call_manager.has_active_call
            or self._cm.call_manager.has_active_google_meet
            or self._cm.call_manager.has_active_teams_meet
        ):
            return {
                "status": "error",
                "message": "A call or meeting is already active.",
            }

        if context:
            self._cm.call_manager.initial_notification = context

        from unify.conversation_manager.events import GoogleMeetReceived

        boss = (
            self._cm.contact_index.get_contact(
                contact_id=SESSION_DETAILS.boss_contact_id,
            )
            or {}
        )
        event = GoogleMeetReceived(contact=boss, meet_url=meet_url)
        await self._event_broker.publish(event.topic, event.to_json())
        return {"status": "ok", "message": f"Joining Google Meet at {meet_url}"}

    async def _leave_google_meet(self) -> dict[str, Any]:
        """Disconnect the assistant from the active Google Meet session.

        Internal helper routed through ``hang_up``; closes the browser and tears
        down the audio bridge.
        """
        if not self._cm.call_manager.has_active_google_meet:
            return {
                "status": "error",
                "message": "No active Google Meet session to leave.",
            }

        # Stop the browser agent immediately so the assistant disappears
        # from the Meet before the event handler's full cleanup pipeline.
        await self._notify_browser_meet_leave("googlemeet")

        from unify.conversation_manager.events import GoogleMeetEnded

        contact = self._cm.call_manager._disconnect_contact or {}
        event = GoogleMeetEnded(contact=contact)
        await self._event_broker.publish(event.topic, event.to_json())
        return {"status": "ok", "message": "Leaving Google Meet"}

    async def start_google_meet_screenshare(self) -> dict[str, Any]:
        """Share the assistant's desktop screen in the active Google Meet call.

        Opens a live view of the assistant's desktop and presents it to all
        meeting participants. Use this when participants need to see what the
        assistant is doing on its computer.
        """
        result = await self._cm.call_manager.start_gmeet_screenshare()
        if result:
            return {
                "status": "ok",
                "message": "Now presenting desktop in Google Meet.",
            }
        return {
            "status": "error",
            "message": "Failed to start screen sharing.",
        }

    async def stop_google_meet_screenshare(self) -> dict[str, Any]:
        """Stop sharing the assistant's desktop screen in Google Meet.

        Ends the current screen presentation. Meeting participants will no
        longer see the assistant's desktop.
        """
        result = await self._cm.call_manager.stop_gmeet_screenshare()
        if result:
            return {
                "status": "ok",
                "message": "Stopped presenting in Google Meet.",
            }
        return {
            "status": "error",
            "message": "Failed to stop screen sharing.",
        }

    async def join_teams_meet(
        self,
        meet_url: str,
        context: str = "",
    ) -> dict[str, Any]:
        """Join a Microsoft Teams meeting.

        This is the **only** way to join a Teams meeting — it configures audio
        devices, establishes the voice pipeline, and dispatches the voice
        agent so the assistant can hear and speak in the meeting.  Never
        attempt to join a Teams meeting URL via ``act``.

        Args:
            meet_url: The full Teams meeting URL (e.g.
                https://teams.microsoft.com/l/meetup-join/... or
                https://teams.live.com/meet/...).
            context: Briefing for the voice agent about the meeting purpose,
                expected participants, topics to cover, and how to behave.
                Same guidance principles as make_call's context parameter.
        """
        if (
            self._cm.call_manager.has_active_call
            or self._cm.call_manager.has_active_google_meet
            or self._cm.call_manager.has_active_teams_meet
        ):
            return {
                "status": "error",
                "message": "A call or meeting is already active.",
            }

        if context:
            self._cm.call_manager.initial_notification = context

        from unify.conversation_manager.events import TeamsMeetReceived

        boss = (
            self._cm.contact_index.get_contact(
                contact_id=SESSION_DETAILS.boss_contact_id,
            )
            or {}
        )
        event = TeamsMeetReceived(contact=boss, meet_url=meet_url)
        await self._event_broker.publish(event.topic, event.to_json())
        return {"status": "ok", "message": f"Joining Teams meeting at {meet_url}"}

    async def _leave_teams_meet(self) -> dict[str, Any]:
        """Disconnect the assistant from the active Teams meeting session.

        Internal helper routed through ``hang_up``; closes the browser and tears
        down the audio bridge.
        """
        if not self._cm.call_manager.has_active_teams_meet:
            return {
                "status": "error",
                "message": "No active Teams meeting session to leave.",
            }

        await self._notify_browser_meet_leave("teamsmeet")

        from unify.conversation_manager.events import TeamsMeetEnded

        contact = self._cm.call_manager._disconnect_contact or {}
        event = TeamsMeetEnded(contact=contact)
        await self._event_broker.publish(event.topic, event.to_json())
        return {"status": "ok", "message": "Leaving Teams meeting"}

    async def start_unify_meet(self, context: str = "") -> dict[str, Any]:
        """Ring my boss on Unify Meet (the in-app live call) and ask them to answer.

        Unify Meet is the in-app live call inside the Console - the canonical
        place to talk face to face (no phone credits, nothing to hold). Use this
        to move the conversation onto that call, e.g. to continue onboarding on
        the live call after verifying another channel.

        I cannot join my boss's browser for them: this rings them, and a pinned
        incoming-call window with an Answer button appears in their Console. When
        they answer, the call connects and I join automatically. If they do not
        answer shortly, I will be told to continue with them over text instead.

        Only one voice session can be active at a time, so if I am currently on a
        phone or WhatsApp call I must ``hang_up`` first, then ring the Meet.

        Args:
            context: A short briefing for how I should open the call once it is
                answered (why I'm calling / what we're continuing), spoken as a
                purposeful greeting. Same guidance principles as make_call's
                context parameter.
        """
        if (
            self._cm.call_manager.has_active_call
            or self._cm.call_manager.has_active_google_meet
            or self._cm.call_manager.has_active_teams_meet
        ):
            return {
                "status": "error",
                "message": (
                    "A call or meeting is already active. Hang up first, then "
                    "ring the Unify Meet."
                ),
            }
        return await self._cm.ring_unify_meet(context=context)

    async def hang_up(self) -> dict[str, Any]:
        """End the current call or meeting, whatever kind it is.

        Hangs up / leaves the live voice session — a phone call, WhatsApp call,
        Unify Meet, Google Meet, or Microsoft Teams meeting. Only one voice
        session can be active at a time, so this always targets that session.
        After it ends, the call-starting tools become available again.

        Use this when my boss asks me to hang up, end the call, leave the
        meeting, or when the conversation is clearly finished and it is natural
        to disconnect.
        """
        channel = self._cm.call_manager._call_channel
        has_meet = (
            self._cm.call_manager.has_active_google_meet
            or self._cm.call_manager.has_active_teams_meet
        )
        if not has_meet and (
            not self._cm.in_voice_session
            or channel
            not in (
                "phone_call",
                "whatsapp_call",
                "unify_meet",
            )
        ):
            return {
                "status": "error",
                "message": "No active voice session to end.",
            }

        # Defer the actual teardown until any explanatory line this turn has been
        # spoken, so the session never ends mid-utterance. The orchestration lives
        # in ``_run_llm`` (after the spoken guidance is published and delivered).
        self._cm._pending_hang_up = True
        self._cm._pending_hang_up_teardown = self._perform_hang_up_teardown
        return {
            "status": "ok",
            "message": "Acknowledged — I'll end the call once I've finished speaking.",
        }

    async def _perform_hang_up_teardown(self) -> dict[str, Any]:
        """Tear down the active voice session (run after the spoken line lands).

        Internal helper invoked by ``_run_llm`` once any explanatory guidance for
        the hang-up turn has been delivered; never called directly as a tool.
        """
        if self._cm.call_manager.has_active_google_meet:
            return await self._leave_google_meet()
        if self._cm.call_manager.has_active_teams_meet:
            return await self._leave_teams_meet()

        await self._cm.call_manager.end_call()
        # Wait for the outbound-only resource gate so a follow-on dial only
        # proceeds once a new assistant-initiated call can open cleanly.
        ready = await self._cm.call_manager.await_ready_for_outbound_call()
        return {
            "status": "ok",
            "message": (
                "Call ended; the line is clear and ready for a new call."
                if ready
                else "Call ended, but the voice line is still being prepared."
            ),
            "ready_for_outbound_call": ready,
        }

    async def start_teams_meet_screenshare(self) -> dict[str, Any]:
        """Share the assistant's desktop screen in the active Teams meeting.

        Opens a live view of the assistant's desktop and presents it to all
        meeting participants. Use this when participants need to see what the
        assistant is doing on its computer.
        """
        result = await self._cm.call_manager.start_teams_meet_screenshare()
        if result:
            return {
                "status": "ok",
                "message": "Now presenting desktop in Teams meeting.",
            }
        return {
            "status": "error",
            "message": "Failed to start screen sharing.",
        }

    async def stop_teams_meet_screenshare(self) -> dict[str, Any]:
        """Stop sharing the assistant's desktop screen in Teams meeting.

        Ends the current screen presentation. Meeting participants will no
        longer see the assistant's desktop.
        """
        result = await self._cm.call_manager.stop_teams_meet_screenshare()
        if result:
            return {
                "status": "ok",
                "message": "Stopped presenting in Teams meeting.",
            }
        return {
            "status": "error",
            "message": "Failed to stop screen sharing.",
        }

    async def _notify_browser_meet_leave(self, path_prefix: str) -> None:
        """Best-effort notify the agent-service to leave the active browser meet.

        ``path_prefix`` is the agent-service URL prefix (``"googlemeet"`` or
        ``"teamsmeet"``).  Failures are silently ignored — the event handler's
        cleanup pipeline still tears down the session even if this call fails.
        """
        session_id = self._cm.call_manager._meet_session_id
        if not session_id:
            return

        import aiohttp

        from unify.session_details import SESSION_DETAILS

        try:
            async with aiohttp.ClientSession() as session:
                await session.post(
                    f"http://localhost:3000/{path_prefix}/leave",
                    json={"sessionId": session_id},
                    headers={
                        "authorization": f"Bearer {SESSION_DETAILS.unify_key}",
                    },
                    timeout=aiohttp.ClientTimeout(total=30),
                )
        except Exception:
            pass

    async def act(
        self,
        *,
        query: str,
        requesting_contact_id: int,
        response_format: Optional[dict] = None,
        persist: bool = False,
        include_conversation_context: bool = True,
        llm_profile: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Engage with knowledge, resources, and the world beyond immediate conversations.

        This is the all-purpose method for any work that requires searching, retrieving,
        manipulating, or acting on information. Use ``act`` liberally — if it cannot
        help, it will simply report back. There is no penalty for speculative delegation.

        **Capabilities include:**

        - **Retrieval**: Search contact records, query knowledge bases, look up past
          conversations, find calendar events, search the web, retrieve files
        - **Action**: Update records, modify spreadsheets, control the desktop/web interface,
          schedule tasks, create reminders
        - **Combined**: Find information and act on it (e.g., "find David's email")

        **Excluded:** Do not use ``act`` for Google Meet or Microsoft Teams
        meeting operations — use the dedicated tools instead:
        ``join_google_meet`` / ``join_teams_meet`` to join, ``hang_up`` to leave
        the current meeting or call,
        ``start_google_meet_screenshare`` / ``start_teams_meet_screenshare`` to
        present the assistant's desktop, and
        ``stop_google_meet_screenshare`` / ``stop_teams_meet_screenshare`` to
        stop presenting.

        **When uncertain, call ``act``**: If you need information you don't have (like
        a contact's email address), call ``act`` to search for it. If ``act`` can't find
        it, it will tell you, and you can then ask the user.

        Args:
            query: Natural language request specifying what to do or find.
            requesting_contact_id: The contact_id of the person whose request or
                needs this action serves.  For responses to a contact's message,
                use that contact's ID.  For proactive actions benefiting a
                specific person, use their contact_id.  In ambiguous cases,
                choose the contact who most directly benefits from the action.
            response_format: An optional structured schema describing the shape of
                the result you need back.  When provided, the action is required to
                return a JSON object conforming to this schema (via a dedicated
                ``final_response`` tool) instead of free-form text.

                The schema uses a concise format where keys are field names and
                values describe their types:

                - Type strings: ``"string"``, ``"integer"``, ``"number"``,
                  ``"boolean"`` (shorthand ``"str"``, ``"int"``, etc. also work).
                - Nested objects: use a dict value, e.g.
                  ``{"address": {"city": "string", "zip": "string"}}``.
                - Arrays: use a single-element list whose element defines the item
                  schema, e.g. ``[{"name": "string", "email": "string"}]``.

                **Examples:**

                - Simple flat fields::

                      {"email": "string", "phone": "string"}

                - Nested with array::

                      {"contacts": [{"name": "string", "email": "string"}],
                       "total_count": "integer"}

                When omitted (the default), the action returns free-form text and
                the result is whatever the actor decides to report.
            persist: If True, the action runs as a **persistent session** that does
                not self-complete.  The actor stays alive after each response and
                waits for the next ``interject`` before continuing.  Use this for
                long-running interactive sessions (e.g. guided onboarding, live
                screen-sharing walkthroughs, multi-step workflows with a tight
                feedback loop between conversation and action).

                **Key differences from the default (persist=False):**

                - The action will **never** complete on its own.  You must
                  explicitly call ``stop_*`` to end the session.
                - Intermediate responses from the actor appear as **response**
                  events in the action's history (marked ``awaiting_input``).
                  Each response means the actor has finished its current turn
                  and is waiting for your next instruction via ``interject_*``.
                - Progress updates (notifications) may still arrive while the
                  actor is working, before it sends a response.

                The default (False) is a one-shot task: the actor works until
                done and the result arrives as an ``ActorResult``.
            include_conversation_context: Whether to pass the current conversation
                state to the action. When ``true`` (default), the action receives
                the full rendered conversation snapshot — messages, notifications,
                and in-flight actions — helping it understand the broader context.
                Set ``false`` when the action is self-contained and the query
                alone provides all necessary information (e.g. simple lookups,
                web searches, or factual questions). Subsequent steering calls
                (interject, ask) on this action will also skip context forwarding.
            llm_profile: Optional curated LLM profile for this action. Leave
                unset for the default actor profile, normally
                ``deepseek-v4-max@deepseek``. Use ``gpt_5_5_low``,
                ``gpt_5_5_medium``, or ``gpt_5_5_high`` only when the task or
                the user's wording warrants premium GPT-5.5 reasoning. If the
                user explicitly asks you to "use all of your thinking effort",
                "think as hard as possible", or similar, choose
                ``gpt_5_5_high``. GPT-5.5 profiles are substantially more
                expensive than the DeepSeek default: about 11.5x the
                input-token rate and 34.5x the output-token rate before any
                extra reasoning/output tokens from higher effort.

                Escalate the profile when retrying an action that shows
                concrete evidence of model/tool-use struggle, rather than
                retrying the same default profile repeatedly. Good escalation
                signals include a previous ``ActorResult`` ending in a
                tool-schema or tool-call formatting error, the same failed
                step recurring after a retry, repeated execution mistakes
                without new information being gathered, clear user frustration,
                or the user's explicit request for stronger/premium reasoning.
                Do not escalate solely because an action has been running for
                a long time; long-running data, coding, or browser work can be
                normal. If restarting after one of the concrete failure
                signals above, preserve the user's task and set
                ``llm_profile`` to ``gpt_5_5_medium`` or ``gpt_5_5_high``
                depending on difficulty and urgency.
        """
        global _next_handle_id

        import time as _bat_time

        _bat_t0 = _bat_time.perf_counter()

        def _bat_ms() -> str:
            return f"{(_bat_time.perf_counter() - _bat_t0) * 1000:.0f}ms"

        import logging as _bat_logging

        _bat_log = _bat_logging.getLogger("unify")
        _bat_log.debug(f"⏱️ [CM.act tool +{_bat_ms()}] entered")
        cm = self._cm

        suppression = cm.suppress_duplicate_commissioning_tool(
            tool_name="act",
            tool_args={
                "query": query,
                "requesting_contact_id": requesting_contact_id,
                "response_format": response_format,
                "persist": persist,
                "include_conversation_context": include_conversation_context,
                "llm_profile": llm_profile,
            },
        )
        if suppression is not None:
            return suppression

        # Override cost attribution for all nested LLM calls in this action.
        from unify.events.cost_attribution import COST_ATTRIBUTION
        from unify.session_details import SESSION_DETAILS

        # Per-user cost attribution only matters in org context (personal
        # accounts have a single user). Resolve the acting user here.
        effective_user_id = SESSION_DETAILS.user.id
        if SESSION_DETAILS.org_id is not None:
            contact = self._cm.contact_index.get_contact(
                contact_id=requesting_contact_id,
            )
            # Only trust user_id from system contacts (boss + provisioned org
            # members).  A contact from another org could carry a platform
            # user_id that doesn't belong to this org.
            attributed_user_id = (
                contact.get("user_id") if contact and contact.get("is_system") else None
            )
            effective_user_id = attributed_user_id or SESSION_DETAILS.user.id
            COST_ATTRIBUTION.set(
                (
                    [attributed_user_id]
                    if attributed_user_id
                    else [SESSION_DETAILS.user.id]
                ),
            )

        # Use the first sentence (everything before the first ".") as a
        # concise action label, falling back to the whole query when there is
        # no period.
        action_summary = query.split(".", 1)[0].strip() if query else ""
        action_label = f"Action: {action_summary}" if action_summary else None

        # Bind the billing context so all nested LLM calls in this action are
        # recorded as tool-driven work (source="tool") with the action label.
        # This must run for personal workspaces too: otherwise the action's
        # LLM spend inherits the conversation turn's "chat" context and shows
        # in the usage ledger as generic chat work instead of the action.
        try:
            import unillm

            unillm.set_billing_context(
                assistant_id=SESSION_DETAILS.assistant.agent_id,
                user_id=effective_user_id,
                organization_id=SESSION_DETAILS.org_id,
                source="tool",
                label=action_label,
            )
        except (ImportError, Exception):
            pass

        # Pass the fresh rendered state snapshot as context for the Actor,
        # unless the LLM opted out.
        parent_context = None
        if include_conversation_context:
            parent_context = (
                [_filter_cm_state_for_actor(self._cm._current_state_snapshot)]
                if self._cm._current_state_snapshot
                else None
            )
        _bat_log.debug(f"⏱️ [CM.act tool +{_bat_ms()}] parent context built")

        # Convert the LLM-provided schema dict into a Pydantic model that the
        # Actor's async tool loop uses for structured output validation.
        pydantic_response_format = None
        if response_format is not None:
            pydantic_response_format = schema_dict_to_pydantic(response_format)

        handle_id = _next_handle_id
        _next_handle_id += 1

        _bat_log.debug(f"⏱️ [CM.act tool +{_bat_ms()}] calling cm.actor.act()")
        handle = await cm.actor.act(
            query,
            _parent_chat_context=parent_context,
            response_format=pydantic_response_format,
            persist=persist,
            llm_profile=llm_profile,
        )
        _bat_log.debug(
            f"⏱️ [CM.act tool +{_bat_ms()}] cm.actor.act() returned handle",
        )

        initial_snapshot_state: SnapshotState | None = None
        if hasattr(self._cm, "_current_snapshot_state"):
            initial_snapshot_state = self._cm._current_snapshot_state

        self._cm.in_flight_actions[handle_id] = {
            "handle": handle,
            "query": query,
            "persist": persist,
            "llm_profile": llm_profile,
            "action_type": "act",
            "handle_actions": [
                {
                    "action_name": "act_started",
                    "query": query,
                    "timestamp": prompt_now(),
                },
            ],
            "initial_snapshot_state": initial_snapshot_state,
            "context_opted_in": include_conversation_context,
        }
        asyncio.create_task(
            managers_utils.actor_watch_result(
                handle_id,
                handle,
                action_type="act",
            ),
        )
        asyncio.create_task(
            managers_utils.actor_watch_notifications(handle_id, handle),
        )
        asyncio.create_task(
            managers_utils.actor_watch_clarifications(handle_id, handle),
        )
        _bat_log.debug(f"⏱️ [CM.act tool +{_bat_ms()}] watchers started")

        _bat_log.debug(f"⏱️ [CM.act tool +{_bat_ms()}] publishing ActorHandleStarted")
        await self._event_broker.publish(
            f"app:actor:actor_started_handle_{handle_id}",
            ActorHandleStarted(
                handle_id=handle_id,
                action_name="act",
                query=query,
                response_format=response_format,
            ).to_json(),
        )
        _bat_log.debug(f"⏱️ [CM.act tool +{_bat_ms()}] done, returning")

        return {"status": "acting", "query": query}

    async def _invoke_manager_action(
        self,
        *,
        manager: Any,
        method_name: str,
        text: str,
        action_type: str,
        response_format: Optional[dict] = None,
        include_conversation_context: bool = True,
    ) -> dict[str, Any]:
        """Shared lifecycle for direct manager tools (contact and transcript actions).

        Follows the same pattern as ``act``: store handle in
        ``in_flight_actions``, spawn watcher tasks, publish started event.
        """
        global _next_handle_id
        LOGGER.info(
            f"{ICONS['fast_path']} [FastPath] {action_type}: {text}",
        )

        parent_context = None
        if include_conversation_context:
            parent_context = (
                [_filter_cm_state_for_actor(self._cm._current_state_snapshot)]
                if self._cm._current_state_snapshot
                else None
            )

        pydantic_response_format = None
        if response_format is not None:
            pydantic_response_format = schema_dict_to_pydantic(response_format)

        cm = self._cm

        handle_id = _next_handle_id
        _next_handle_id += 1

        method = getattr(manager, method_name)
        handle = await method(
            text,
            response_format=pydantic_response_format,
            _parent_chat_context=parent_context,
        )

        initial_snapshot_state: SnapshotState | None = None
        if hasattr(cm, "_current_snapshot_state"):
            initial_snapshot_state = cm._current_snapshot_state

        cm.in_flight_actions[handle_id] = {
            "handle": handle,
            "query": text,
            "persist": False,
            "action_type": action_type,
            "handle_actions": [
                {
                    "action_name": f"{action_type}_started",
                    "query": text,
                    "timestamp": prompt_now(),
                },
            ],
            "initial_snapshot_state": initial_snapshot_state,
            "context_opted_in": include_conversation_context,
        }
        asyncio.create_task(
            managers_utils.actor_watch_result(
                handle_id,
                handle,
                action_type=action_type,
            ),
        )
        asyncio.create_task(
            managers_utils.actor_watch_notifications(handle_id, handle),
        )
        asyncio.create_task(
            managers_utils.actor_watch_clarifications(handle_id, handle),
        )

        await self._event_broker.publish(
            f"app:actor:actor_started_handle_{handle_id}",
            ActorHandleStarted(
                handle_id=handle_id,
                action_name=action_type,
                query=text,
                response_format=response_format,
            ).to_json(),
        )

        return {"status": "acting", "query": text}

    async def ask_about_contacts(
        self,
        *,
        text: str,
        response_format: Optional[dict] = None,
    ) -> dict[str, Any]:
        """
        Query contact records directly — names, emails, phone numbers, roles,
        relationships, and any other stored contact attributes.

        This is a **direct channel** to the contact management system, bypassing
        the general ``act`` pathway. Use it for any purely contact-related
        questions:

        - Looking up a specific contact's details
        - Finding contacts by attribute (role, location, company, etc.)
        - Checking if a contact exists
        - Listing or filtering contacts
        - Comparing contact records

        **Route here instead of ``act`` when the question is purely about
        contact data.** If the question also involves non-contact information
        (tasks, knowledge, transcripts, web, files, etc.) or requires
        cross-domain reasoning, use ``act`` instead.

        Args:
            text: Natural language question about contacts
                (e.g. "What is Sarah's email address?").
            response_format: Optional structured schema describing the shape of
                the result you need back. Same format as ``act``'s
                ``response_format`` — keys are field names, values are type
                strings (``"string"``, ``"integer"``, etc.), nested dicts, or
                single-element lists for arrays. When omitted, a free-form text
                answer is returned.
        """
        return await self._invoke_manager_action(
            manager=self._cm.contact_manager,
            method_name="ask",
            text=text,
            action_type="ask_about_contacts",
            response_format=response_format,
        )

    async def update_contacts(
        self,
        *,
        text: str,
        response_format: Optional[dict] = None,
    ) -> dict[str, Any]:
        """
        Create, edit, delete, or merge contact records directly.

        This is a **direct channel** to the contact management system, bypassing
        the general ``act`` pathway. Use it for any purely contact-related
        mutations:

        - Creating new contacts
        - Updating contact details (phone, email, address, role, bio, etc.)
        - Deleting contacts
        - Merging duplicate contacts

        **Route here instead of ``act`` when the request is purely about
        modifying contacts.** If the request also involves non-contact work
        or cross-domain operations, use ``act`` instead.

        Args:
            text: Natural language description of the contact change
                (e.g. "Add a new contact for John Smith, email john@acme.com").
            response_format: Optional structured schema describing the shape of
                the result you need back. Same format as ``act``'s
                ``response_format``. When omitted, a free-form text summary of
                the mutation is returned.
        """
        return await self._invoke_manager_action(
            manager=self._cm.contact_manager,
            method_name="update",
            text=text,
            action_type="update_contacts",
            response_format=response_format,
        )

    async def query_past_transcripts(
        self,
        *,
        text: str,
        response_format: Optional[dict] = None,
    ) -> dict[str, Any]:
        """
        Search and analyse past messages and conversation history directly.

        This is a **direct channel** to the transcript store, bypassing the
        general ``act`` pathway. Use it for any purely transcript-related
        questions:

        - Retrieving recent messages from a specific contact or channel
        - Searching past conversations for a keyword or topic
        - Summarising what was discussed in a previous exchange
        - Checking what someone said or when they last messaged
        - Comparing or filtering messages by date, medium, or sender

        **Route here instead of ``act`` when the question is purely about
        past messages or conversation history.** If the question also involves
        non-transcript information (contacts, knowledge, tasks, web, files,
        etc.) or requires cross-domain reasoning, use ``act`` instead.

        Args:
            text: Natural language question about past transcripts
                (e.g. "What did Bob say about the deadline yesterday?").
            response_format: Optional structured schema describing the shape of
                the result you need back. Same format as ``act``'s
                ``response_format`` — keys are field names, values are type
                strings (``"string"``, ``"integer"``, etc.), nested dicts, or
                single-element lists for arrays. When omitted, a free-form text
                answer is returned.
        """
        return await self._invoke_manager_action(
            manager=self._cm.transcript_manager,
            method_name="ask",
            text=text,
            action_type="query_past_transcripts",
            response_format=response_format,
        )

    # ── Computer fast-path tools ──────────────────────────────────────────

    async def _silent_interject_act_sessions(
        self,
        message: str,
    ) -> None:
        """Send a silent interjection to every in-flight ``act`` session,
        keeping the Actor informed without triggering an immediate LLM turn."""
        for hid, data in list(self._cm.in_flight_actions.items()):
            if data.get("action_type") != "act":
                continue
            handle = data.get("handle")
            if handle and not handle.done():
                try:
                    await handle.interject(
                        message,
                        trigger_immediate_llm_turn=False,
                        suppress_response_notification=True,
                    )
                except TypeError:
                    await handle.interject(message)

    async def _invoke_fast_path_action(
        self,
        *,
        coro,
        text: str,
        action_type: str,
    ) -> dict[str, Any]:
        """Shared lifecycle for desktop fast-path tools.

        Runs the desktop primitive call in the background and registers it as
        an in-flight action with the same lifecycle as ``act`` and the contact/
        transcript fast paths: ``ActorHandleStarted`` event, watcher tasks for
        result delivery, and automatic cleanup on completion.

        Interjects in-flight Actor desktop sessions twice:
        1. Immediately when the request is made (so the Actor knows what's happening)
        2. After the primitive completes with the result
        """
        global _next_handle_id
        LOGGER.info(
            f"{ICONS['fast_path']} [FastPath] {action_type}: {text}",
        )

        cm = self._cm
        handle_id = _next_handle_id
        _next_handle_id += 1

        async def _run():
            result = await coro
            summary = str(result) if result is not None else "done"
            LOGGER.info(
                f"{ICONS['fast_path']} [FastPath] {action_type} completed:\n"
                f"{summary}",
            )
            await self._silent_interject_act_sessions(
                f'[Fast-path result] {action_type}("{text}") completed. '
                f"Result: {result}\n\n"
                f"If this result looks wrong or incomplete — especially if "
                f"the task falls within your loaded guidance or requires "
                f"capabilities the fast path lacks (credentials, secrets, "
                f"multi-step workflows) — escalate by calling "
                f'notify({{"type": "escalation", "message": "<what you can '
                f'do better>"}}).  Otherwise, no action needed.',
            )
            return str(result) if result is not None else "done"

        handle = _DesktopActionHandle(asyncio.create_task(_run()))

        cm.in_flight_actions[handle_id] = {
            "handle": handle,
            "query": text,
            "persist": False,
            "action_type": action_type,
            "handle_actions": [
                {
                    "action_name": f"{action_type}_started",
                    "query": text,
                    "timestamp": prompt_now(),
                },
            ],
            "initial_snapshot_state": getattr(cm, "_current_snapshot_state", None),
            "context_opted_in": False,
        }
        asyncio.create_task(
            managers_utils.actor_watch_result(
                handle_id,
                handle,
                action_type=action_type,
            ),
        )

        await self._event_broker.publish(
            f"app:actor:actor_started_handle_{handle_id}",
            ActorHandleStarted(
                handle_id=handle_id,
                action_name=action_type,
                query=text,
            ).to_json(),
        )

        await self._silent_interject_act_sessions(
            f"[Fast-path request] The outer process is executing "
            f'{action_type}("{text}"). Do not replicate this action — it '
            f"is already in progress. You will see the result shortly.",
        )

        return {"status": "acting", "query": text}

    async def desktop_act(
        self,
        *,
        instruction: str,
    ) -> dict[str, Any]:
        """Execute a single desktop interaction on native UI (non-browser).

        Each call performs exactly **one** desktop interaction — one click,
        one keystroke, one window switch.

        **Only for interactions that cannot be done inside a web browser:**
        native application windows, terminal commands, file manager operations,
        system dialogs, or desktop UI elements outside any browser window.

        For any task involving a web browser — opening a browser, navigating,
        clicking web page elements, typing into web forms — use ``web_act``
        instead.

        Examples:

        - "Open the Terminal application"
        - "Switch to the File Manager window"
        - "Right-click the desktop background"
        - "Click the system tray notification"

        **Route through ``act`` or ``interject_*``** when the task requires
        more than one interaction, reasoning about *what* to do, or benefits
        from guidance and compositional functions.

        Args:
            instruction: A concrete single native desktop action to perform
                (e.g. "Open the Terminal application").
        """
        cp = self._cm.computer_primitives
        return await self._invoke_fast_path_action(
            coro=cp.desktop.act(instruction),
            text=instruction,
            action_type="desktop_act",
        )

    # ── Web fast-path tools ────────────────────────────────────────────

    def _resolve_or_create_web_session(self, session_id: int | None):
        """Return (handle, is_new) for an existing or freshly-created session."""
        cp = self._cm.computer_primitives

        async def _resolve():
            if session_id is not None:
                return cp.web.get_session(session_id), False
            handle = await cp.web.new_session(visible=True)
            return handle, True

        return _resolve()

    async def web_act(
        self,
        *,
        request: str,
        session_id: int | None = None,
    ) -> dict[str, Any]:
        """Execute a single browser interaction in a visible web session.

        Each call performs exactly **one** browser interaction — one click,
        one text entry, one scroll, or one navigation.  It bypasses the
        general ``act`` pathway and runs directly against a Chromium browser
        session visible on the desktop.

        A new browser session is created automatically when ``session_id``
        is omitted.  Pass a numeric ``session_id`` from
        ``<active_web_sessions>`` to continue working in an existing session.

        **Use ``desktop_act`` instead** only for native desktop interactions
        that cannot be done inside a browser (terminal, file manager, native
        app windows, system dialogs).

        **Use ``act`` or ``interject_*`` instead** when the task requires
        more than one interaction (e.g. click a button then type a value
        then click Save), or needs guidance / functions / knowledge.

        Args:
            request: Natural language description of a single browser action
                (e.g. "Navigate to example.com", "Click the Submit button").
            session_id: Optional numeric ID of an existing active web session
                to reuse.  When omitted a new visible session is created.
        """
        handle, is_new = await self._resolve_or_create_web_session(session_id)
        used_id = handle.session_id
        label = f"[session={used_id}, new={is_new}]"
        return await self._invoke_fast_path_action(
            coro=handle.act(request),
            text=f"{request} {label}",
            action_type="web_act",
        )

    async def close_web_session(
        self,
        *,
        session_id: int,
    ) -> dict[str, Any]:
        """Close a visible web browser session to free resources.

        Use after completing browser work to clean up.  Check
        ``<active_web_sessions>`` in the current state for valid IDs.

        Args:
            session_id: The numeric ID of the web session to close.
        """
        cp = self._cm.computer_primitives
        try:
            handle = cp.web.get_session(session_id)
        except ValueError:
            return {
                "status": "not_found",
                "session_id": session_id,
                "error": "No active web session with that ID.",
            }
        await handle.stop()
        return {"status": "closed", "session_id": session_id}

    async def set_boss_details(
        self,
        *,
        first_name: str | None = None,
        surname: str | None = None,
        phone_number: str | None = None,
        email_address: str | None = None,
    ) -> dict[str, Any]:
        """
        Update the boss contact's details.

        Use this when you learn the boss's name, phone number, or email
        address during conversation. Only provided fields are updated;
        omitted fields are left unchanged.

        Updating the boss's email address is especially important — once
        their email is on file and they create an account at unify.ai,
        the assistant will be automatically linked to their account.

        Args:
            first_name: The boss's first name.
            surname: The boss's surname / last name.
            phone_number: The boss's phone number.
            email_address: The boss's email address.
        """
        updates = {
            k: v
            for k, v in {
                "first_name": first_name,
                "surname": surname,
                "phone_number": phone_number,
                "email_address": email_address,
            }.items()
            if v is not None
        }
        if not updates:
            return {"status": "error", "error": "No fields provided to update."}

        self._cm.contact_index.contact_manager.update_contact(
            contact_id=SESSION_DETAILS.boss_contact_id,
            **updates,
        )
        return {"status": "updated", "updates": updates}

    async def deactivate_onboarding(self) -> dict[str, Any]:
        """Pause the global onboarding flow so my boss can use the platform normally.

        **Only call after my boss has verbally confirmed** they want to pause
        onboarding — not merely hinted, not while I am mid-explanation, and
        not when they are deferring a single checklist row (that is a per-row
        UI control, not a global pause).

        **Call when:** they clearly want to stop setup for now and have
        confirmed after I restate intent — e.g. "Yes, pause it", "Let's do
        that", "I'll finish setup later" in direct answer to my confirmation
        question.

        **Do not call when:** I am suggesting pause proactively; they asked
        what pause means; they said "skip this step" (row defer only); they
        have not answered a confirmation question; onboarding is already
        inactive; they only want help with a normal task while onboarding is
        still active (answer normally instead).

        After success: tell them onboarding is paused, they can ask for help
        normally, and they can resume later from the **Onboarding** tab in
        Assistant info or by asking me to continue setup.
        """
        return await self._cm._patch_coordinator_onboarding_active(
            active=False,
            clear_onboarding_step=True,
        )

    async def activate_onboarding(self) -> dict[str, Any]:
        """Resume the global onboarding checklist and nudges.

        **Only call after my boss has verbally confirmed** they want to return
        to setup — not when they are asking a one-off product question while
        onboarding is paused (answer normally without reactivating).

        **Call when:** they ask to resume or finish setup and confirm after I
        restate intent — e.g. "Yes, let's continue onboarding", "Take me back
        to the checklist".

        **Do not call when:** onboarding is already active; they have not
        confirmed; they only asked how to open the Onboarding tab (give
        directions instead); the Console **Return to onboarding** button is
        enough and they did not ask me to flip it.

        After success: confirm onboarding is live again and guide them to the
        first valid next step from my live onboarding progress block (if any).
        """
        return await self._cm._patch_coordinator_onboarding_active(active=True)

    async def set_onboarding_task_state(
        self,
        step_id: str,
        completed: bool,
    ) -> dict[str, Any]:
        """Mark one onboarding checklist step complete or incomplete.

        Use the ``step_id`` values from my live onboarding progress block
        (for example ``apps``, ``create-scheduled-task``). Communication
        rows and auto-triggered checklist actions complete on their own —
        if this tool returns an error, relay that explanation rather than
        guessing which steps are settable.

        **Call when:** I have finished onboarding work that the checklist
        cannot detect yet (semantic setup, guided walkthroughs outside
        Communication) and the user agrees the step is done — or when I need
        to undo a manual completion I set earlier.

        **Do not call when:** onboarding is inactive; the step is in the
        Communication section; the user only clicked a checklist row that
        auto-starts an action; or I have not actually done the work.

        After success: confirm briefly and continue from the refreshed
        onboarding progress block.
        """
        return await self._cm._patch_coordinator_onboarding_step_state(
            step_id=step_id,
            completed=completed,
        )

    async def wait(
        self,
        delay: int | None = None,
    ) -> dict[str, Any]:
        """
        Wait for more input without taking any action.

        PREFER THIS TOOL over sending messages in most situations. Call this tool:
        - After completing a request (let the user respond first)
        - When there are no NEW messages requiring response
        - When unsure whether to speak (when in doubt, wait)
        - To let the conversation end naturally

        The user should usually have the last word. Do not send follow-up
        messages, additional information, or "anything else?" prompts unless
        the user explicitly asks for more.

        Parameters
        ----------
        delay : int | None
            Seconds to wait before automatically waking up for another thinking
            turn.  When ``None`` (the default), wait indefinitely until the next
            external event (new message, action completion, etc.).  When set to a
            positive integer, the system schedules a follow-up thinking turn after
            that many seconds — useful for probing a long-running action or
            revisiting a situation after a reasonable interval.
        """
        self._cm._outbound_suppress_gen = self._cm._llm_gen
        return {"status": "waiting", "delay": delay}

    async def guide_voice_agent(
        self,
        *,
        message: str,
        fast_brain_guidance: str = "",
    ) -> dict[str, Any]:
        """
        Speak a line to the caller during a live call.

        On each turn I SPEAK (call this tool with ``message``) or WAIT (omit it
        and call ``wait``). The Voice Agent otherwise only fills the latency gap
        with a brief filler; everything the caller actually hears is the
        ``message`` I speak here. I call this **in parallel** with my action tool
        (``wait``, ``act``, ``send_sms``, etc.).

        I may optionally attach ``fast_brain_guidance``: a short note the Voice
        Agent may use to give a basic, direct reply to the caller's *very next*
        message (e.g. confirm a fact I have just told it), without waiting for me.
        It is never spoken on its own, the Voice Agent never volunteers it, and it
        is ONLY honored when ``message`` is also set — I can never hand over
        guidance without also speaking. It applies to the next moment only; my
        next spoken turn replaces or clears it.

        Write ``message`` in the language currently spoken on the call.

        Args:
            message: The exact words to speak to the caller now, spoken verbatim
                via TTS. Use **spoken prose** (no numbered lists, bullets, or
                outline labels — TTS reads them literally).
            fast_brain_guidance: Optional short note for the Voice Agent to give a
                basic direct reply to the caller's next message. Never spoken;
                only honored alongside ``message``. Include any "do not reveal
                unless…" constraint explicitly if the note is sensitive.
        """
        return {"status": "guidance_noted"}

    def _whatsapp_contact_label(self, contact_id: int) -> str:
        """Human-friendly name for a contact in the window-status appendix."""
        contact = None
        try:
            contact = self._cm.contact_index.get_contact(contact_id)
        except Exception:
            contact = None
        first = (contact or {}).get("first_name") if contact else None
        if first:
            return first
        if SESSION_DETAILS.is_coordinator and contact_id == self._boss_contact_id():
            return "my boss"
        return f"contact {contact_id}"

    def _whatsapp_window_doc_suffix(self) -> str:
        """Window-status appendix for the ``send_whatsapp`` tool docstring.

        Lists, by name, every contact whose 24-hour WhatsApp free-form window is
        currently known to be CLOSED (a send delivers only a placeholder) or OPEN
        (delivered verbatim). Returns ``""`` when nothing is known so the
        window-agnostic docstring guidance stands on its own.
        """
        cm = self._cm
        relevant: set[int] = set()
        if SESSION_DETAILS.is_coordinator:
            relevant.add(self._boss_contact_id())
        relevant.update(cm._pending_whatsapp_resends.keys())
        relevant.update(cm._whatsapp_window_open.keys())

        closed: list[str] = []
        open_: list[str] = []
        for cid in relevant:
            state = cm.whatsapp_window_state(cid)
            if state is None:
                continue
            (open_ if state else closed).append(self._whatsapp_contact_label(cid))

        if not closed and not open_:
            return ""

        lines = [
            "",
            "Current WhatsApp free-form window (decides whether a send arrives "
            "verbatim or only as a placeholder):",
        ]
        if closed:
            lines.append(
                "- CLOSED for: "
                + ", ".join(sorted(closed))
                + ". A send now delivers only a generic placeholder; my real "
                "text is queued to resend after they reply. I tell them to reply "
                "to the placeholder first and I do NOT claim the actual "
                "message/clue arrived.",
            )
        if open_:
            lines.append(
                "- OPEN for: "
                + ", ".join(sorted(open_))
                + ". A send now is delivered verbatim.",
            )
        return "\n".join(lines)

    def _with_whatsapp_window_doc(
        self,
        base: "Callable[..., Any]",
    ) -> "Callable[..., Any]":
        """Return ``base`` with the live window status appended to its docstring.

        Rebuilt per turn (``as_tools`` runs each turn), so the status reflects
        the latest known window state. Returns ``base`` unchanged when nothing is
        known, to avoid an empty appendix.
        """
        suffix = self._whatsapp_window_doc_suffix()
        if not suffix:
            return base

        @wraps(base)
        async def _send_whatsapp_window_aware(**kwargs: Any) -> Any:
            return await base(**kwargs)

        # Pin the schema signature to the bound method's (which already excludes
        # ``self``); without this, ``inspect.signature`` would unwrap past the
        # bound method to the underlying primitive and re-expose ``self``.
        _send_whatsapp_window_aware.__signature__ = inspect.signature(base)
        base_doc = inspect.getdoc(base) or ""
        _send_whatsapp_window_aware.__doc__ = f"{base_doc}\n{suffix}"
        return _send_whatsapp_window_aware

    def as_tools(self) -> dict[str, "Callable[..., Any]"]:
        """Return the static tools dict for start_async_tool_loop."""
        from unify.settings import SETTINGS

        is_coordinator = SESSION_DETAILS.is_coordinator
        tools: dict[str, Callable[..., Any]] = {
            "send_unify_message": (
                self.send_unify_message_to_boss
                if is_coordinator
                else self.send_unify_message
            ),
            "send_api_response": (
                self.send_api_response_to_boss
                if is_coordinator
                else self.send_api_response
            ),
            "wait": self.wait,
        }
        call_or_meet_in_progress = self._cm.in_voice_session
        # Assistant-initiated call tools are exposed only when a new outbound
        # call can actually open cleanly: no live session AND the worker has a
        # freshly prewarmed idle process ready. Inbound sessions bypass this
        # gate and are accepted by the runtime immediately.
        ready_to_start_call = (
            not call_or_meet_in_progress
            and self._cm.call_manager.is_ready_for_outbound_call
        )
        if not call_or_meet_in_progress:
            tools["join_google_meet"] = self.join_google_meet
            tools["join_teams_meet"] = self.join_teams_meet
            # Ringing the in-app Meet only signals the Console (the owner answers
            # in-browser), so unlike telephony call-start it needs no prewarmed
            # worker; expose it whenever no other voice session is live.
            tools["start_unify_meet"] = self.start_unify_meet
        else:
            # One voice session at a time; a single tool ends whichever is live
            # (phone, WhatsApp, Unify Meet, Google Meet, or Teams).
            tools["hang_up"] = self.hang_up
        if self._cm.call_manager.has_active_google_meet:
            if SESSION_DETAILS.assistant.desktop_url:
                if not self._cm.call_manager.has_gmeet_presenting:
                    tools["start_google_meet_screenshare"] = (
                        self.start_google_meet_screenshare
                    )
                else:
                    tools["stop_google_meet_screenshare"] = (
                        self.stop_google_meet_screenshare
                    )
        if self._cm.call_manager.has_active_teams_meet:
            if SESSION_DETAILS.assistant.desktop_url:
                if not self._cm.call_manager.has_teams_presenting:
                    tools["start_teams_meet_screenshare"] = (
                        self.start_teams_meet_screenshare
                    )
                else:
                    tools["stop_teams_meet_screenshare"] = (
                        self.stop_teams_meet_screenshare
                    )
        if self._cm.assistant_number:
            tools["send_sms"] = (
                self.send_sms_to_boss if is_coordinator else self.send_sms
            )
            if ready_to_start_call:
                tools["make_call"] = (
                    self.make_call_to_boss if is_coordinator else self.make_call
                )
        if self._cm.assistant_whatsapp_number:
            base_send_whatsapp = (
                self.send_whatsapp_to_boss if is_coordinator else self.send_whatsapp
            )
            tools["send_whatsapp"] = self._with_whatsapp_window_doc(base_send_whatsapp)
            if ready_to_start_call:
                tools["make_whatsapp_call"] = (
                    self.make_whatsapp_call_to_boss
                    if is_coordinator
                    else self.make_whatsapp_call
                )
        if self._cm.assistant_email:
            tools["send_email"] = (
                self.send_email_to_boss if is_coordinator else self.send_email
            )
        if self._cm.assistant_discord_bot_id:
            tools["send_discord_message"] = (
                self.send_discord_message_to_boss
                if is_coordinator
                else self.send_discord_message
            )
            if not is_coordinator:
                tools["send_discord_channel_message"] = (
                    self.send_discord_channel_message
                )
        if self._cm.assistant_slack_bot_user_id:
            tools["send_slack_message"] = (
                self.send_slack_message_to_boss
                if is_coordinator
                else self.send_slack_message
            )
            if not is_coordinator:
                tools["send_slack_channel_message"] = self.send_slack_channel_message
        if self._cm.assistant_has_teams:
            tools["send_teams_message"] = (
                self.send_teams_message_to_boss
                if is_coordinator
                else self.send_teams_message
            )
            if is_coordinator:
                tools["create_teams_meet"] = self.create_teams_meet_with_boss
            else:
                tools["create_teams_channel"] = self.create_teams_channel
                tools["create_teams_meet"] = self.create_teams_meet
        if getattr(self._cm.mode, "is_voice", False):
            tools["guide_voice_agent"] = self.guide_voice_agent
        if SETTINGS.DEMO_MODE:
            tools["set_boss_details"] = self.set_boss_details
        elif self._cm.initialized:
            tools["act"] = self.act
            tools["ask_about_contacts"] = self.ask_about_contacts
            tools["update_contacts"] = self.update_contacts
            tools["query_past_transcripts"] = self.query_past_transcripts
        # During onboarding, withhold a reference-quiz channel's send tool until
        # the user clicks its trigger row (this session) or the step durably
        # completes — so T-W1N cannot send an untagged clue proactively.
        if self._cm.coordinator_onboarding_active:
            for name in masked_reference_quiz_tools(
                self._cm.coordinator_onboarding_render,
                self._cm.onboarding_clicked_trigger_steps,
            ):
                tools.pop(name, None)
        if is_coordinator and SETTINGS.UNITY_CONSOLE_UI:
            if self._cm.coordinator_onboarding_active:
                tools["deactivate_onboarding"] = self.deactivate_onboarding
                tools["set_onboarding_task_state"] = self.set_onboarding_task_state
            else:
                tools["activate_onboarding"] = self.activate_onboarding
        return tools

    def build_action_steering_tools(self) -> dict[str, "Callable[..., Any]"]:
        """Build dynamic tools for steering in-flight actions.

        Conditionally generates pause/resume tools based on current state:
        - If action is paused: only generate resume_* (skip pause_*)
        - If action is running: only generate pause_* (skip resume_*)
        - If state unknown: only generate pause_* (default to running)
        """
        tools: dict[str, Callable[..., Any]] = {}

        for handle_id, handle_data in (self._cm.in_flight_actions or {}).items():
            query = handle_data.get("query", "")
            short_name = derive_short_name(query)
            handle = handle_data.get("handle")
            handle_actions = handle_data.get("handle_actions", [])

            # Check pause state to conditionally generate pause/resume tools
            is_paused = get_handle_paused_state(handle)

            pending_clarifications = [
                a
                for a in handle_actions
                if a.get("action_name") == "clarification_request"
                and not a.get("response")
            ]

            for op in STEERING_OPERATIONS:
                # Conditionally skip pause/resume based on current state
                # is_paused=True: skip pause, only offer resume
                # is_paused=False or None: skip resume, only offer pause (default to running)
                if op.name == "pause" and is_paused is True:
                    continue  # Already paused, don't offer pause
                if op.name == "resume" and is_paused is not True:
                    continue  # Not paused (running or unknown), don't offer resume

                if op.requires_clarification:
                    for clar in pending_clarifications:
                        call_id = clar.get("call_id", "")
                        suffix = safe_call_id_suffix(call_id)
                        tool_name = build_action_name(
                            op.name,
                            short_name,
                            handle_id,
                            suffix,
                        )
                        tool_fn = self._make_steering_tool(
                            handle_id,
                            handle,
                            op.name,
                            op.param_name,
                            op.get_docstring(),
                            query,
                            call_id,
                        )
                        tools[tool_name] = tool_fn
                else:
                    tool_name = build_action_name(op.name, short_name, handle_id)
                    tool_fn = self._make_steering_tool(
                        handle_id,
                        handle,
                        op.name,
                        op.param_name,
                        op.get_docstring(),
                        query,
                    )
                    tools[tool_name] = tool_fn

        return tools

    def build_completed_action_tools(self) -> dict[str, "Callable[..., Any]"]:
        """Build ask tools for completed actions.

        Completed actions preserve their trajectory and remain available
        for `ask` queries about their execution and results.
        """
        tools: dict[str, Callable[..., Any]] = {}

        for handle_id, handle_data in (self._cm.completed_actions or {}).items():
            query = handle_data.get("query", "")
            short_name = derive_short_name(query)
            handle = handle_data.get("handle")

            # ask tool — query the completed action's trajectory/results
            ask_op = OPERATION_MAP["ask"]
            tool_name = build_action_name(ask_op.name, short_name, handle_id)
            tool_fn = self._make_completed_action_ask_tool(
                handle_id,
                handle,
                ask_op.param_name,
                ask_op.get_docstring(),
                query,
            )
            tools[tool_name] = tool_fn

        return tools

    @staticmethod
    def _extract_tool_param_value(
        *,
        kwargs: dict[str, Any],
        primary_name: str,
        aliases: tuple[str, ...] = (),
    ) -> Any:
        """Extract a tool parameter value from kwargs using primary name then aliases."""
        if not primary_name:
            return ""
        for name in (primary_name, *aliases):
            if name in kwargs:
                return kwargs.get(name, "")
        return ""

    def _make_completed_action_ask_tool(
        self,
        handle_id: int,
        handle: Any,
        param_name: str,
        docstring: str,
        query: str,
    ) -> "Callable[..., Any]":
        """Create an ask tool closure for a completed action."""

        cm = self._cm
        event_broker = cm.event_broker
        ask_param_aliases = tuple(
            name for name in ("question", "query") if name != param_name
        )

        async def ask_completed_action(
            **kwargs: Any,
        ) -> dict[str, Any]:
            param_value = self._extract_tool_param_value(
                kwargs=kwargs,
                primary_name=param_name,
                aliases=ask_param_aliases,
            )

            # Get handle_data from completed_actions
            handle_data = cm.completed_actions.get(handle_id)

            # Record action with pending status
            if handle_data:
                handle_data["handle_actions"].append(
                    {
                        "action_name": f"ask_{handle_id}",
                        "query": param_value,
                        "status": "pending",
                        "timestamp": prompt_now(),
                    },
                )

            _handle = handle
            _param_value = param_value
            _handle_id = handle_id
            _parent_context = (
                [cm._current_state_snapshot] if cm._current_state_snapshot else None
            )

            async def _perform_ask_and_emit():
                await event_broker.publish(
                    "app:call:notification",
                    FastBrainNotification(
                        message=f"Ask dispatched on action: {_param_value[:200]}",
                        source="system",
                        contact={},
                    ).to_json(),
                )
                try:
                    ask_handle = await _handle.ask(
                        _param_value,
                        _parent_chat_context=_parent_context,
                    )
                    ask_result = await ask_handle.result()
                except Exception as e:
                    ask_result = f"Error: {e}"
                await event_broker.publish(
                    f"app:actor:handle_response_{_handle_id}",
                    ActorHandleResponse(
                        handle_id=_handle_id,
                        action_name="ask",
                        query=_param_value,
                        response=ask_result,
                        call_id="",
                    ).to_json(),
                )

            task = asyncio.create_task(_perform_ask_and_emit())
            cm._pending_steering_tasks.add(task)
            task.add_done_callback(cm._pending_steering_tasks.discard)

            return {
                "status": "ok",
                "operation": "ask",
                "result": (
                    "Query submitted. You will receive another turn "
                    "when the answer is ready."
                ),
            }

        # Build signature with proper parameter name
        if param_name:
            params = [
                inspect.Parameter(
                    param_name,
                    inspect.Parameter.KEYWORD_ONLY,
                    annotation=str,
                ),
            ]
        else:
            params = []

        ask_completed_action.__signature__ = inspect.Signature(params)
        base_doc = docstring or "Ask about this completed action."
        ask_completed_action.__doc__ = f"{base_doc}\n\nFor action: {query}"
        return ask_completed_action

    def _make_steering_tool(
        self,
        handle_id: int,
        handle: Any,
        operation: str,
        param_name: str,
        docstring: str,
        query: str,
        call_id: str | None = None,
    ) -> "Callable[..., Any]":
        """Create a closure for an action steering operation."""

        cm = self._cm
        # Use cm.event_broker to ensure the same broker is used throughout
        # (important for test patching)
        event_broker = cm.event_broker

        async def steering_tool(
            **kwargs: Any,
        ) -> dict[str, Any]:
            param_aliases: tuple[str, ...] = ()
            if operation == "ask":
                param_aliases = tuple(
                    name for name in ("question", "query") if name != param_name
                )
            param_value = self._extract_tool_param_value(
                kwargs=kwargs,
                primary_name=param_name,
                aliases=param_aliases,
            )

            handle_data = cm.in_flight_actions.get(handle_id)

            result = ""
            try:
                match operation:
                    case "ask":
                        # Record action with pending status - result will arrive async
                        if handle_data:
                            handle_data["handle_actions"].append(
                                {
                                    "action_name": f"ask_{handle_id}",
                                    "query": param_value,
                                    "status": "pending",
                                    "timestamp": prompt_now(),
                                },
                            )

                        # Capture values for the closure.
                        # Use the fresh rendered state snapshot (set by _run_llm before tools execute).
                        # Only pass context if the original action opted in.
                        _handle = handle
                        _param_value = param_value
                        _handle_id = handle_id
                        _ctx_opted_in = (
                            handle_data.get("context_opted_in", True)
                            if handle_data
                            else True
                        )
                        _parent_context = None
                        if _ctx_opted_in and cm._current_state_snapshot:
                            _parent_context = [cm._current_state_snapshot]

                        # Spawn background task to perform ask and emit result
                        async def _perform_ask_and_emit():
                            await event_broker.publish(
                                "app:call:notification",
                                FastBrainNotification(
                                    message=f"Ask dispatched on action: {_param_value[:200]}",
                                    source="system",
                                    contact={},
                                ).to_json(),
                            )
                            try:
                                # Start the ask operation (does the LLM roundtrip)
                                ask_handle = await _handle.ask(
                                    _param_value,
                                    _parent_chat_context=_parent_context,
                                )
                                # Await the result
                                ask_result = await ask_handle.result()
                            except Exception as e:
                                ask_result = f"Error: {e}"
                            # Emit ActorHandleResponse event to wake brain
                            await event_broker.publish(
                                f"app:actor:handle_response_{_handle_id}",
                                ActorHandleResponse(
                                    handle_id=_handle_id,
                                    action_name="ask",
                                    query=_param_value,
                                    response=ask_result,
                                    call_id="",
                                ).to_json(),
                            )

                        task = asyncio.create_task(_perform_ask_and_emit())
                        cm._pending_steering_tasks.add(task)
                        task.add_done_callback(
                            cm._pending_steering_tasks.discard,
                        )

                        # Return immediately - brain will be woken when result arrives
                        return {
                            "status": "ok",
                            "operation": "ask",
                            "result": (
                                "Query submitted. You will receive another turn "
                                "when the answer is ready."
                            ),
                        }

                    case "interject":
                        if handle_data:
                            handle_data["handle_actions"].append(
                                {
                                    "action_name": f"{operation}_{handle_id}",
                                    "query": param_value,
                                    "timestamp": prompt_now(),
                                },
                            )

                        # Only compute and send context diffs if the original
                        # action opted into conversation context.
                        parent_context_cont = None
                        _interject_ctx_opted_in = (
                            handle_data.get("context_opted_in", True)
                            if handle_data
                            else True
                        )

                        if _interject_ctx_opted_in:
                            initial_snapshot = (
                                handle_data.get("initial_snapshot_state")
                                if handle_data
                                else None
                            )
                            current_snapshot = getattr(
                                cm,
                                "_current_snapshot_state",
                                None,
                            )

                            if current_snapshot is not None:
                                diff_content = compute_snapshot_diff(
                                    initial_snapshot,
                                    current_snapshot,
                                )
                                if diff_content:
                                    parent_context_cont = [
                                        {
                                            "role": "user",
                                            "content": diff_content,
                                            "_cm_context_diff": True,
                                        },
                                    ]
                            elif cm._current_state_snapshot:
                                parent_context_cont = [cm._current_state_snapshot]

                        await handle.interject(
                            param_value,
                            _parent_chat_context_cont=parent_context_cont,
                        )
                        result = "Interjected successfully"
                    case "stop":
                        if handle_data:
                            handle_data["handle_actions"].append(
                                {
                                    "action_name": f"{operation}_{handle_id}",
                                    "query": param_value,
                                    "timestamp": prompt_now(),
                                },
                            )
                        await handle.stop(reason=param_value or None)
                        stopped = cm.in_flight_actions.pop(handle_id, None)
                        if stopped:
                            cm.completed_actions[handle_id] = stopped
                        result = "Action stopped"
                    case "pause":
                        if handle_data:
                            handle_data["handle_actions"].append(
                                {
                                    "action_name": f"{operation}_{handle_id}",
                                    "query": param_value,
                                    "timestamp": prompt_now(),
                                },
                            )
                        await handle.pause()
                        result = "Action paused"
                    case "resume":
                        if handle_data:
                            handle_data["handle_actions"].append(
                                {
                                    "action_name": f"{operation}_{handle_id}",
                                    "query": param_value,
                                    "timestamp": prompt_now(),
                                },
                            )
                        await handle.resume()
                        result = "Action resumed"
                    case "answer_clarification":
                        if handle_data:
                            handle_data["handle_actions"].append(
                                {
                                    "action_name": f"{operation}_{handle_id}",
                                    "query": param_value,
                                    "timestamp": prompt_now(),
                                },
                            )
                        if call_id:
                            await handle.answer_clarification(call_id, param_value)
                            result = "Clarification answered"
                        else:
                            result = "No clarification call_id available"
                    case _:
                        if handle_data:
                            handle_data["handle_actions"].append(
                                {
                                    "action_name": f"{operation}_{handle_id}",
                                    "query": param_value,
                                    "timestamp": prompt_now(),
                                },
                            )
                        result = f"Unknown operation: {operation}"
            except Exception as e:
                result = f"Error: {e}"

            return {"status": "ok", "operation": operation, "result": result}

        # Copy signature + docstring from the handle's method. Parameters
        # starting with _ are automatically hidden by method_to_schema.
        if handle is not None and hasattr(handle, operation):
            DynamicToolFactory._adopt_signature_and_annotations(
                getattr(handle, operation),
                steering_tool,
            )

        # Append action context so the CM knows which action this tool steers.
        # Preserve the docstring set by _adopt_signature_and_annotations (or
        # fall back to the docstring passed in from SteeringOperation).
        base_doc = inspect.getdoc(steering_tool) or docstring
        steering_tool.__doc__ = f"{base_doc}\n\nFor action: {query}"

        return steering_tool
