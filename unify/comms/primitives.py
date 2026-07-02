"""Shared assistant-owned communication domain.

`CommsPrimitives` is the single implementation of assistant-owned outbound
communication behavior. Two different runtime surfaces delegate into this same
module:

- `primitives.comms.*` inside CodeAct, `SingleFunctionActor`, and the offline
  task runner
- `ConversationManagerBrainActionTools` inside the live assistant runtime

Keeping the behavior here ensures that contact resolution, capability gating,
inline identifier attachment, transport calls, transcript/event publication,
and offline outbound-operation tracking stay consistent across live and
headless execution paths.
"""

from __future__ import annotations

import asyncio
import base64
import os
import uuid
from typing import TYPE_CHECKING, Any, Mapping

from unify.common.hierarchical_logger import DEFAULT_ICON
from unify.common.plain_text import normalize_outbound_plain_text
from unify.common.prompt_helpers import now as prompt_now
from unify.conversation_manager.cm_types import Medium
from unify.conversation_manager.domains import comms_utils
from unify.conversation_manager.event_broker import get_event_broker
from unify.conversation_manager.events import (
    ApiMessageSent,
    DiscordChannelMessageSent,
    DiscordMessageSent,
    EmailSent,
    Error,
    PhoneCallSent,
    SlackChannelMessageSent,
    SlackMessageSent,
    SMSSent,
    TeamsChannelCreated,
    TeamsChannelMessageSent,
    TeamsMeetCreated,
    TeamsMessageSent,
    UnifyMessageSent,
    WhatsAppCallInviteSent,
    WhatsAppCallSent,
    WhatsAppSent,
)
from unify.logger import LOGGER
from unify.manager_registry import ManagerRegistry
from unify.session_details import SESSION_DETAILS
from unify.comms.offline_support import (
    OfflineOutboundReservation,
    finalize_outbound_operation_failure,
    finalize_outbound_operation_success,
    reserve_outbound_operation,
)

if TYPE_CHECKING:
    from unify.conversation_manager.conversation_manager import ConversationManager
    from unify.conversation_manager.in_memory_event_broker import InMemoryEventBroker


_DETAIL_LABELS = {
    "email_address": "email address",
    "phone_number": "phone number",
    "whatsapp_number": "WhatsApp number",
    "discord_id": "Discord ID",
    "slack_user_id": "Slack user ID",
}

_VOICE_SESSION_CLEAR_TIMEOUT_SECONDS = 15.0
_VOICE_SESSION_CLEAR_POLL_SECONDS = 0.5


def _coerce_contact_id(value: Any) -> int:
    """Coerce a contact_id value to int."""
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except (TypeError, ValueError):
            pass
    raise TypeError(
        f"contact_id must be an integer, got {type(value).__name__}: {value!r}",
    )


def _get_contact_display_name(contact: dict | None) -> str:
    """Return a human-readable contact label for errors."""
    if not contact:
        return "unknown contact"
    first = contact.get("first_name") or ""
    last = contact.get("surname") or ""
    name = f"{first} {last}".strip()
    if name:
        return name
    return f"contact_id={contact.get('contact_id', 'unknown')}"


class CommsPrimitives:
    """Assistant-owned communication primitives shared by live and actor paths."""

    _PRIMITIVE_METHODS = (
        "send_sms",
        "send_whatsapp",
        "send_email",
        "make_call",
        "make_whatsapp_call",
        "send_unify_message",
        "send_api_response",
        "send_discord_message",
        "send_discord_channel_message",
        "send_slack_message",
        "send_slack_channel_message",
        "send_teams_message",
        "create_teams_channel",
        "create_teams_meet",
        "stop_inactivity_followups",
        "resume_inactivity_followups",
    )

    def __init__(
        self,
        *,
        conversation_manager: "ConversationManager | None" = None,
        event_broker: "InMemoryEventBroker | None" = None,
    ) -> None:
        self._cm = conversation_manager
        self._event_broker = event_broker or get_event_broker()

    def _assistant_number(self) -> str:
        if self._cm is not None:
            return getattr(self._cm, "assistant_number", "") or ""
        return SESSION_DETAILS.assistant.number or ""

    def _assistant_email(self) -> str:
        if self._cm is not None:
            return getattr(self._cm, "assistant_email", "") or ""
        return SESSION_DETAILS.assistant.email or ""

    def _assistant_whatsapp_number(self) -> str:
        if self._cm is not None:
            return getattr(self._cm, "assistant_whatsapp_number", "") or ""
        return SESSION_DETAILS.assistant.whatsapp_number or ""

    def _assistant_discord_bot_id(self) -> str:
        if self._cm is not None:
            return getattr(self._cm, "assistant_discord_bot_id", "") or ""
        return SESSION_DETAILS.assistant.discord_bot_id or ""

    def _assistant_slack_bot_user_id(self) -> str:
        if self._cm is not None:
            return getattr(self._cm, "assistant_slack_bot_user_id", "") or ""
        return SESSION_DETAILS.assistant.slack_bot_user_id or ""

    def _assistant_has_teams(self) -> bool:
        if self._cm is not None:
            return bool(getattr(self._cm, "assistant_has_teams", False))
        return (
            getattr(SESSION_DETAILS.assistant, "email_provider", "") == "microsoft_365"
        )

    def _allow_whatsapp_call_permission_probe(self) -> bool:
        if self._cm is None:
            return False
        truthy = {"1", "true", "yes", "on"}
        if (
            os.environ.get("WHATSAPP_CALL_PERMISSION_PROBE_ENABLED", "").lower()
            in truthy
        ):
            return True
        if os.environ.get("SELF_HOST", "").lower() in truthy:
            return True
        if os.environ.get("NEXT_PUBLIC_SELF_HOST", "").lower() in truthy:
            return True
        local_urls = (
            os.environ.get("ORCHESTRA_URL", ""),
            os.environ.get("COMMUNICATION_URL", ""),
            os.environ.get("UNITY_COMMS_URL", ""),
        )
        return any("127.0.0.1" in url or "localhost" in url for url in local_urls)

    def _voice_session_active(self) -> bool:
        if self._cm is None:
            return False
        call_manager = self._cm.call_manager
        return bool(
            call_manager.has_active_call
            or call_manager.has_active_google_meet
            or call_manager.has_active_teams_meet
            or call_manager._whatsapp_call_joining,
        )

    async def _wait_for_voice_session_to_clear(self) -> bool:
        if self._cm is None:
            return True
        call_manager = self._cm.call_manager
        clear_stale = getattr(call_manager, "_clear_stale_dispatch_state", None)
        deadline = (
            asyncio.get_running_loop().time() + _VOICE_SESSION_CLEAR_TIMEOUT_SECONDS
        )
        while self._voice_session_active():
            if callable(clear_stale):
                clear_stale()
            if not self._voice_session_active():
                return True
            if asyncio.get_running_loop().time() >= deadline:
                return False
            await asyncio.sleep(_VOICE_SESSION_CLEAR_POLL_SECONDS)
        return True

    async def _await_ready_for_outbound_call(self) -> bool:
        """Await until the voice worker can host an outbound call.

        Delegates to the call manager, which awaits the real resource — a
        freshly prewarmed idle worker process plus a drained prior session —
        with no fixed sleep. Belt-and-suspenders for the (rare) race where the
        tool was exposed at turn start but the resource was consumed before the
        handler ran. Returns True when an outbound call is safe, False on
        timeout.
        """
        if self._cm is None:
            return True
        await_ready = getattr(
            self._cm.call_manager,
            "await_ready_for_outbound_call",
            None,
        )
        if not callable(await_ready):
            return await self._wait_for_voice_session_to_clear()
        return await await_ready()

    def _onboarding_event_kwargs(self, medium: Medium) -> dict[str, str]:
        if self._cm is None:
            return {}
        consume = getattr(self._cm, "consume_pending_onboarding_outbound", None)
        if not callable(consume):
            return {}
        return consume(medium.value) or {}

    def _stash_whatsapp_resend_onboarding_kwargs(
        self,
        contact_id: int | None,
        metadata: dict[str, str],
    ) -> None:
        if self._cm is None or contact_id is None or not metadata:
            return
        stash = getattr(
            self._cm,
            "stash_pending_whatsapp_resend_onboarding_metadata",
            None,
        )
        if callable(stash):
            stash(contact_id, metadata)
            return
        pending = getattr(
            self._cm,
            "_pending_whatsapp_resend_onboarding_metadata",
            None,
        )
        if isinstance(pending, dict):
            pending[contact_id] = dict(metadata)

    def _consume_whatsapp_resend_onboarding_kwargs(
        self,
        contact_id: int | None,
    ) -> dict[str, str]:
        if self._cm is None or contact_id is None:
            return {}
        consume = getattr(
            self._cm,
            "consume_pending_whatsapp_resend_onboarding_metadata",
            None,
        )
        if callable(consume):
            return consume(contact_id) or {}
        pending = getattr(
            self._cm,
            "_pending_whatsapp_resend_onboarding_metadata",
            None,
        )
        if isinstance(pending, dict):
            return pending.pop(contact_id, {}) or {}
        return {}

    def _contact_manager(self):
        if (
            self._cm is not None
            and self._cm.contact_index.is_contact_manager_initialized
        ):
            return self._cm.contact_index.contact_manager
        return ManagerRegistry.get_contact_manager()

    @staticmethod
    def _as_contact_dict(contact: Any) -> dict | None:
        """Normalize pydantic/plain contact values to a plain dict."""
        if contact is None:
            return None
        if hasattr(contact, "model_dump"):
            return contact.model_dump()
        if isinstance(contact, dict):
            return dict(contact)
        return dict(contact)

    @staticmethod
    def _contact_dedupe_key(contact: dict) -> tuple[Any, ...]:
        """Build a stable de-duplication key for contact search results."""
        contact_id = contact.get("contact_id")
        if contact_id is not None:
            return ("contact_id", contact_id)
        return (
            "detail",
            contact.get("email_address"),
            contact.get("phone_number"),
            contact.get("whatsapp_number"),
            contact.get("discord_id"),
            contact.get("slack_user_id"),
            contact.get("first_name"),
            contact.get("surname"),
        )

    def _fallback_contacts(self) -> list[dict]:
        """Return fallback contacts cached on the live ContactIndex."""
        if self._cm is None:
            return []
        fallback = getattr(self._cm.contact_index, "_fallback_contacts", {}) or {}
        return [
            self._as_contact_dict(contact)
            for contact in fallback.values()
            if self._as_contact_dict(contact) is not None
        ]

    def _filter_contacts(
        self,
        *,
        field_name: str,
        value: str,
        limit: int = 10,
    ) -> list[dict]:
        """Search contacts by a single strong identifier."""
        contacts: list[dict] = []
        seen: set[tuple[Any, ...]] = set()

        for contact in self._fallback_contacts():
            if contact.get(field_name) != value:
                continue
            key = self._contact_dedupe_key(contact)
            if key in seen:
                continue
            seen.add(key)
            contacts.append(contact)
            if len(contacts) >= limit:
                return contacts

        escaped = value.replace("\\", "\\\\").replace("'", "\\'")
        try:
            result = self._contact_manager().filter_contacts(
                filter=f"{field_name} == '{escaped}'",
                limit=limit,
            )
        except Exception:
            return contacts

        for contact in result.get("contacts", []):
            contact_dict = self._as_contact_dict(contact)
            if contact_dict is None:
                continue
            key = self._contact_dedupe_key(contact_dict)
            if key in seen:
                continue
            seen.add(key)
            contacts.append(contact_dict)
            if len(contacts) >= limit:
                break
        return contacts

    def _get_contact(
        self,
        *,
        contact_id: int | None = None,
        phone_number: str | None = None,
        email: str | None = None,
        whatsapp_number: str | None = None,
        discord_id: str | None = None,
        slack_user_id: str | None = None,
    ) -> dict | None:
        """Resolve a contact by ID or a communication detail."""
        if self._cm is not None:
            try:
                contact = self._cm.contact_index.get_contact(
                    contact_id=contact_id,
                    phone_number=phone_number,
                    email=email,
                    whatsapp_number=whatsapp_number,
                    discord_id=discord_id,
                    slack_user_id=slack_user_id,
                )
                contact_dict = self._as_contact_dict(contact)
                if contact_dict is not None:
                    return contact_dict
            except Exception:
                pass

        try:
            if contact_id is not None:
                result = self._contact_manager().get_contact_info(contact_id)
                return self._as_contact_dict(result.get(contact_id))
        except Exception:
            return None

        if phone_number is not None:
            matches = self._filter_contacts(
                field_name="phone_number",
                value=phone_number,
                limit=1,
            )
        elif email is not None:
            matches = self._filter_contacts(
                field_name="email_address",
                value=email,
                limit=1,
            )
        elif whatsapp_number is not None:
            matches = self._filter_contacts(
                field_name="whatsapp_number",
                value=whatsapp_number,
                limit=1,
            )
        elif discord_id is not None:
            matches = self._filter_contacts(
                field_name="discord_id",
                value=discord_id,
                limit=1,
            )
        elif slack_user_id is not None:
            matches = self._filter_contacts(
                field_name="slack_user_id",
                value=slack_user_id,
                limit=1,
            )
        else:
            return None
        return matches[0] if matches else None

    def _find_conflicting_contact(
        self,
        *,
        field_name: str,
        value: str,
        exclude_contact_id: int,
    ) -> dict | None:
        """Return another contact that already owns the provided identifier."""
        for contact in self._filter_contacts(
            field_name=field_name,
            value=value,
            limit=10,
        ):
            if contact.get("contact_id") != exclude_contact_id:
                return contact
        return None

    def _assistant_anchor_contact(self) -> dict:
        """Return the assistant contact as a synthetic anchor for channel-only sends."""
        assistant_contact_id = SESSION_DETAILS.self_contact_id
        return self._get_contact(contact_id=assistant_contact_id) or {
            "contact_id": assistant_contact_id,
            "first_name": SESSION_DETAILS.assistant.first_name,
            "surname": SESSION_DETAILS.assistant.surname,
        }

    def _normalize_optional_contact(self, contact_id: int | str | None) -> dict | None:
        """Resolve an optional contact_id to a contact dict when provided."""
        if contact_id is None:
            return None
        return self._get_contact(contact_id=_coerce_contact_id(contact_id))

    def _check_outbound_allowed(self, contact: dict | None) -> str | None:
        """Check whether a contact may receive assistant-owned outbound comms."""
        if not contact:
            return "Contact not found"
        should_respond = contact.get("should_respond", False)
        if should_respond:
            return None
        contact_name = _get_contact_display_name(contact)
        return (
            f"Cannot send outbound communication to {contact_name}: "
            f"should_respond is False for this contact. "
            f"Check the contact's response_policy for details or ask your boss for guidance."
        )

    @staticmethod
    def _is_coordinator_boss_contact(contact_id: int | str) -> bool:
        if not SESSION_DETAILS.is_coordinator:
            return False
        try:
            return int(contact_id) == int(SESSION_DETAILS.boss_contact_id)
        except (TypeError, ValueError):
            return False

    def _resolve_or_attach_detail(
        self,
        *,
        contact: dict | None,
        contact_id: int,
        field_name: str,
        inline_value: str | None,
        medium_label: str,
    ) -> tuple[str | None, dict | None]:
        """Resolve a strong identifier, optionally attaching it to the contact."""
        if contact is None:
            return ("Contact not found", None)

        existing_value = contact.get(field_name)
        if existing_value:
            if inline_value and inline_value != existing_value:
                detail_label = _DETAIL_LABELS[field_name]
                return (
                    f"{_get_contact_display_name(contact)} already has "
                    f"{detail_label} {existing_value}. "
                    f"Do not overwrite identifiers during a {medium_label} send. "
                    f"Use `act` or `update_contacts` first if the contact really changed.",
                    contact,
                )
            return (None, contact)

        if not inline_value:
            detail_label = _DETAIL_LABELS[field_name]
            if self._is_coordinator_boss_contact(contact_id):
                return (
                    f"{_get_contact_display_name(contact)} does not have a {detail_label} on file. "
                    "Update the boss contact first, then retry.",
                    contact,
                )
            return (
                f"{_get_contact_display_name(contact)} does not have a {detail_label} on file. "
                f"Provide `{field_name}` in this send or update the contact first.",
                contact,
            )

        conflict = self._find_conflicting_contact(
            field_name=field_name,
            value=inline_value,
            exclude_contact_id=contact_id,
        )
        if conflict is not None:
            detail_label = _DETAIL_LABELS[field_name]
            return (
                f"Cannot use {detail_label} {inline_value} for "
                f"{_get_contact_display_name(contact)} because it already belongs to "
                f"{_get_contact_display_name(conflict)}. "
                f"Update the existing contact first instead of reassigning identifiers during a send.",
                contact,
            )

        self._contact_manager().update_contact(
            contact_id=contact_id,
            **{field_name: inline_value},
        )
        refreshed = self._get_contact(contact_id=contact_id) or {
            **contact,
            field_name: inline_value,
        }
        return (None, refreshed)

    async def _surface_comms_error(
        self,
        error_msg: str,
        topic: str,
        *,
        contact_id: int | None = None,
        medium: Medium | None = None,
        offline_reservation: OfflineOutboundReservation | None = None,
        attempted_content: str = "",
        receiver_ids: list[int] | None = None,
        target_metadata: Mapping[str, Any] | None = None,
        history_metadata: Mapping[str, Any] | None = None,
        attachments: list[dict] | None = None,
    ) -> dict[str, Any]:
        """Publish a comms error and surface it in the live conversation thread."""
        if offline_reservation is not None:
            finalize_outbound_operation_failure(
                offline_reservation,
                error=error_msg,
                attempted_content=attempted_content,
                receiver_ids=receiver_ids,
                target_metadata=target_metadata,
                metadata=history_metadata,
                attachments=attachments,
            )
        if self._cm is not None and contact_id is not None and medium is not None:
            self._cm.contact_index.push_message(
                contact_id=contact_id,
                sender_name="System",
                thread_name=medium,
                message_content=f"[Send Failed] {error_msg}",
                role="system",
                timestamp=prompt_now(as_string=False),
            )
        await self._event_broker.publish(topic, Error(error_msg).to_json())
        return {"status": "error", "error": error_msg}

    def _reserve_offline_operation(
        self,
        *,
        method_name: str,
        medium: Medium,
        target_kind: str,
        target_metadata: Mapping[str, Any],
        contact_id: int | None = None,
    ) -> tuple[OfflineOutboundReservation | None, dict[str, Any] | None]:
        """Reserve one durable offline outbound operation when headless tracking is active."""

        decision = reserve_outbound_operation(
            method_name=method_name,
            medium=medium,
            target_kind=target_kind,
            target_metadata=target_metadata,
            contact_id=contact_id,
        )
        return decision.reservation, decision.response

    def _record_offline_success(
        self,
        offline_reservation: OfflineOutboundReservation | None,
        *,
        attempted_content: str,
        receiver_ids: list[int] | None,
        target_metadata: Mapping[str, Any] | None = None,
        history_metadata: Mapping[str, Any] | None = None,
        attachments: list[dict] | None = None,
        provider_response: Mapping[str, Any] | None = None,
        status: str = "completed",
    ) -> None:
        """Persist offline assistant history and finalize the ledger on success."""

        finalize_outbound_operation_success(
            offline_reservation,
            attempted_content=attempted_content,
            receiver_ids=receiver_ids,
            target_metadata=target_metadata,
            metadata=history_metadata,
            attachments=attachments,
            provider_response=provider_response,
            status=status,
        )

    async def send_sms(
        self,
        *,
        contact_id: int | str,
        content: str,
        phone_number: str | None = None,
    ) -> dict[str, Any]:
        """Send an assistant-owned SMS message to an existing contact.

        The contact must already exist in the system.

        - If the contact already has a phone number on file, omit
          ``phone_number``.
        - If the contact is missing a phone number but you know it, pass it via
          ``phone_number`` and the contact record will be updated before the
          SMS is sent.
        - Do not supply a different phone number from the one already on file.
          Update the contact first, then retry.

        Parameters
        ----------
        contact_id : int | str
            Recipient contact from ``active_conversations`` or from contact
            search/create tools.
        content : str
            Text body to send.
        phone_number : str | None, optional
            Recipient phone number when the contact does not already have one on
            file.

        Returns
        -------
        dict[str, Any]
            ``{"status": "ok"}`` on success, or an error payload describing
            why the send was rejected or failed.
        """
        contact_id = _coerce_contact_id(contact_id)
        content = normalize_outbound_plain_text(content)
        offline_reservation = None
        contact = self._get_contact(contact_id=contact_id)

        outbound_error = self._check_outbound_allowed(contact)
        if outbound_error:
            return await self._surface_comms_error(
                outbound_error,
                "app:comms:sms_sent",
                contact_id=contact_id,
                medium=Medium.SMS_MESSAGE,
                offline_reservation=offline_reservation,
                attempted_content=content,
                receiver_ids=[contact_id],
                target_metadata={
                    "contact_id": contact_id,
                    "phone_number": (contact or {}).get("phone_number")
                    or phone_number
                    or "",
                },
                history_metadata={
                    "contact_display_name": _get_contact_display_name(contact),
                },
            )

        if not self._assistant_number():
            return await self._surface_comms_error(
                "You don't have a number, please provision one.",
                "app:comms:sms_sent",
                contact_id=contact_id,
                medium=Medium.SMS_MESSAGE,
                attempted_content=content,
                receiver_ids=[contact_id],
                target_metadata={
                    "contact_id": contact_id,
                    "phone_number": (contact or {}).get("phone_number")
                    or phone_number
                    or "",
                },
                history_metadata={
                    "contact_display_name": _get_contact_display_name(contact),
                },
            )

        detail_error, contact = self._resolve_or_attach_detail(
            contact=contact,
            contact_id=contact_id,
            field_name="phone_number",
            inline_value=phone_number,
            medium_label="SMS",
        )
        if detail_error:
            return await self._surface_comms_error(
                detail_error,
                "app:comms:sms_sent",
                contact_id=contact_id,
                medium=Medium.SMS_MESSAGE,
                offline_reservation=offline_reservation,
                attempted_content=content,
                receiver_ids=[contact_id],
                target_metadata={
                    "contact_id": contact_id,
                    "phone_number": (contact or {}).get("phone_number")
                    or phone_number
                    or "",
                },
                history_metadata={
                    "contact_display_name": _get_contact_display_name(contact),
                },
            )

        offline_reservation, offline_response = self._reserve_offline_operation(
            method_name="send_sms",
            medium=Medium.SMS_MESSAGE,
            target_kind="contact",
            target_metadata={
                "contact_id": (contact or {}).get("contact_id") or contact_id,
                "phone_number": (contact or {}).get("phone_number")
                or phone_number
                or "",
            },
            contact_id=(contact or {}).get("contact_id") or contact_id,
        )
        if offline_response is not None:
            return offline_response

        to_number = (contact or {}).get("phone_number")
        response = await comms_utils.send_sms_message_via_number(
            to_number=to_number,
            content=content,
        )
        if response.get("success"):
            fresh_contact = self._get_contact(phone_number=to_number) or contact or {}
            event = SMSSent(
                contact=fresh_contact,
                content=content,
                **self._onboarding_event_kwargs(Medium.SMS_MESSAGE),
            )
            await self._event_broker.publish("app:comms:sms_sent", event.to_json())
            self._record_offline_success(
                offline_reservation,
                attempted_content=content,
                receiver_ids=[fresh_contact.get("contact_id") or contact_id],
                target_metadata={
                    "contact_id": fresh_contact.get("contact_id") or contact_id,
                    "phone_number": to_number or "",
                },
                history_metadata={
                    "contact_display_name": _get_contact_display_name(fresh_contact),
                },
                provider_response=response,
            )
            return {"status": "ok"}

        if not self._assistant_number():
            error_msg = "You don't have a number, please provision one."
        else:
            error_msg = f"Failed to send sms to {to_number}"
        return await self._surface_comms_error(
            error_msg,
            "app:comms:sms_sent",
            contact_id=contact_id,
            medium=Medium.SMS_MESSAGE,
            offline_reservation=offline_reservation,
            attempted_content=content,
            receiver_ids=[contact_id],
            target_metadata={
                "contact_id": contact_id,
                "phone_number": to_number or phone_number or "",
            },
            history_metadata={
                "contact_display_name": _get_contact_display_name(contact),
            },
        )

    async def send_whatsapp(
        self,
        *,
        contact_id: int | str,
        content: str,
        whatsapp_number: str | None = None,
        attachment_filepath: str | None = None,
    ) -> dict[str, Any]:
        """Send an assistant-owned WhatsApp message to an existing contact.

        The contact must already exist in the system.

        - If the contact already has a WhatsApp number on file, omit
          ``whatsapp_number``.
        - If the contact is missing a WhatsApp number but you know it, pass it
          via ``whatsapp_number`` and the contact record will be updated before
          the message is sent.
        - Do not supply a different WhatsApp number from the one already on
          file. Update the contact first, then retry.

        WhatsApp accepts one attachment per outbound message. Inside the normal
        24-hour window, standard media types can be sent alongside the text. If
        WhatsApp falls back to a template notification instead of the verbatim
        body, a live assistant session can queue the original message for resend
        after the contact replies. Headless offline task runs cannot queue that
        follow-up, so they only report that a notification was sent.

        Delivery is NOT confirmed in the turn I call this tool. The send is
        handed off, not proven: WhatsApp either delivers the verbatim
        ``content`` (free-form, when the 24-hour window is open) or, when the
        window is closed, delivers only a generic template placeholder and the
        real body is queued to resend after the contact next replies. Which one
        happened is known only afterwards, when it surfaces as a
        ``[You WhatsApped <name>]`` transcript row (delivered verbatim) or a
        ``[You WhatsApped <name> (not delivered directly)]`` row (placeholder
        only — the body did NOT arrive). Therefore, in the same turn I call this
        tool, I say only that I am *sending* the message — never that it has
        arrived, is waiting, or is in their inbox. I confirm receipt only after
        the proof row appears; if it is marked ``(not delivered directly)`` I
        tell the contact a placeholder went out and they must reply to it before
        I can deliver the real content (clue, etc.).

        Parameters
        ----------
        contact_id : int | str
            Recipient contact from ``active_conversations`` or from contact
            search/create tools.
        content : str
            Message body to send.
        whatsapp_number : str | None, optional
            Recipient WhatsApp number when the contact does not already have one
            on file.
        attachment_filepath : str | None, optional
            Workspace-relative file path for one attachment to upload and send.

        Returns
        -------
        dict[str, Any]
            Status payload describing whether the send succeeded, failed, or
            entered the live resend flow.
        """
        contact_id = _coerce_contact_id(contact_id)
        content = normalize_outbound_plain_text(content)
        offline_reservation = None
        contact = self._get_contact(contact_id=contact_id)

        topic = "app:comms:whatsapp_sent"
        outbound_error = self._check_outbound_allowed(contact)
        if outbound_error:
            return await self._surface_comms_error(
                outbound_error,
                topic,
                contact_id=contact_id,
                medium=Medium.WHATSAPP_MESSAGE,
                offline_reservation=offline_reservation,
                attempted_content=content,
                receiver_ids=[contact_id],
                target_metadata={
                    "contact_id": contact_id,
                    "whatsapp_number": (contact or {}).get("whatsapp_number")
                    or whatsapp_number
                    or "",
                    "attachment_filepath": attachment_filepath or "",
                },
                history_metadata={
                    "contact_display_name": _get_contact_display_name(contact),
                },
            )

        if not self._assistant_whatsapp_number():
            return await self._surface_comms_error(
                "WhatsApp is not enabled for this assistant.",
                topic,
                contact_id=contact_id,
                medium=Medium.WHATSAPP_MESSAGE,
                attempted_content=content,
                receiver_ids=[contact_id],
                target_metadata={
                    "contact_id": contact_id,
                    "whatsapp_number": (contact or {}).get("whatsapp_number")
                    or whatsapp_number
                    or "",
                    "attachment_filepath": attachment_filepath or "",
                },
                history_metadata={
                    "contact_display_name": _get_contact_display_name(contact),
                },
            )

        detail_error, contact = self._resolve_or_attach_detail(
            contact=contact,
            contact_id=contact_id,
            field_name="whatsapp_number",
            inline_value=whatsapp_number,
            medium_label="WhatsApp",
        )
        if detail_error:
            return await self._surface_comms_error(
                detail_error,
                topic,
                contact_id=contact_id,
                medium=Medium.WHATSAPP_MESSAGE,
                offline_reservation=offline_reservation,
                attempted_content=content,
                receiver_ids=[contact_id],
                target_metadata={
                    "contact_id": contact_id,
                    "whatsapp_number": (contact or {}).get("whatsapp_number")
                    or whatsapp_number
                    or "",
                    "attachment_filepath": attachment_filepath or "",
                },
                history_metadata={
                    "contact_display_name": _get_contact_display_name(contact),
                },
            )

        offline_reservation, offline_response = self._reserve_offline_operation(
            method_name="send_whatsapp",
            medium=Medium.WHATSAPP_MESSAGE,
            target_kind="contact",
            target_metadata={
                "contact_id": (contact or {}).get("contact_id") or contact_id,
                "whatsapp_number": (contact or {}).get("whatsapp_number")
                or whatsapp_number
                or "",
                "attachment_filepath": attachment_filepath or "",
            },
            contact_id=(contact or {}).get("contact_id") or contact_id,
        )
        if offline_response is not None:
            return offline_response

        attachment = None
        media_url = None
        if attachment_filepath:
            from unify.file_manager.filesystem_adapters.local_adapter import (
                LocalFileSystemAdapter,
            )

            adapter = LocalFileSystemAdapter()
            try:
                abs_path = adapter._abspath(attachment_filepath)
                with open(abs_path, "rb") as file_handle:
                    file_contents = file_handle.read()
            except FileNotFoundError:
                return await self._surface_comms_error(
                    f"File not found: {attachment_filepath}",
                    topic,
                    contact_id=contact_id,
                    medium=Medium.WHATSAPP_MESSAGE,
                    offline_reservation=offline_reservation,
                    attempted_content=content,
                    receiver_ids=[contact_id],
                    target_metadata={
                        "contact_id": contact_id,
                        "whatsapp_number": (contact or {}).get("whatsapp_number")
                        or whatsapp_number
                        or "",
                        "attachment_filepath": attachment_filepath or "",
                    },
                    history_metadata={
                        "contact_display_name": _get_contact_display_name(contact),
                    },
                )
            except Exception as exc:
                return await self._surface_comms_error(
                    f"Failed to read file: {exc}",
                    topic,
                    contact_id=contact_id,
                    medium=Medium.WHATSAPP_MESSAGE,
                    offline_reservation=offline_reservation,
                    attempted_content=content,
                    receiver_ids=[contact_id],
                    target_metadata={
                        "contact_id": contact_id,
                        "whatsapp_number": (contact or {}).get("whatsapp_number")
                        or whatsapp_number
                        or "",
                        "attachment_filepath": attachment_filepath or "",
                    },
                    history_metadata={
                        "contact_display_name": _get_contact_display_name(contact),
                    },
                )

            attachment_filename = os.path.basename(attachment_filepath)
            upload_result = await comms_utils.upload_unify_attachment(
                file_content=file_contents,
                filename=attachment_filename,
            )
            if "error" in upload_result:
                return await self._surface_comms_error(
                    f"Failed to upload attachment: {upload_result['error']}",
                    topic,
                    contact_id=contact_id,
                    medium=Medium.WHATSAPP_MESSAGE,
                    offline_reservation=offline_reservation,
                    attempted_content=content,
                    receiver_ids=[contact_id],
                    target_metadata={
                        "contact_id": contact_id,
                        "whatsapp_number": (contact or {}).get("whatsapp_number")
                        or whatsapp_number
                        or "",
                        "attachment_filepath": attachment_filepath or "",
                    },
                    history_metadata={
                        "contact_display_name": _get_contact_display_name(contact),
                    },
                )

            attachment = upload_result
            attachment_id = attachment.get("id", "")
            attachment_target = f"Attachments/{attachment_id}_{attachment_filename}"
            try:
                import shutil

                attachment_dir = adapter._abspath("Attachments")
                os.makedirs(attachment_dir, exist_ok=True)
                shutil.copy2(
                    abs_path,
                    os.path.join(
                        attachment_dir,
                        f"{attachment_id}_{attachment_filename}",
                    ),
                )
            except Exception:
                pass
            attachment["filepath"] = attachment_target

            media_url = attachment.get("url") or attachment.get("gs_url")
            if media_url and media_url.startswith("gs://"):
                import aiohttp as _aiohttp

                from unify.conversation_manager.domains.comms_utils import (
                    _get_signed_url_from_gs_url,
                )

                async with _aiohttp.ClientSession() as session:
                    media_url = await _get_signed_url_from_gs_url(session, media_url)

        to_number = (contact or {}).get("whatsapp_number")
        response = await comms_utils.send_whatsapp_message(
            to_number=to_number,
            content=content,
            user_name=(contact or {}).get("first_name", ""),
            agent_name=SESSION_DETAILS.assistant.first_name,
            media_url=media_url,
        )
        if response.get("success"):
            via_template = response.get("method") == "template"
            delivered_content = response.get("delivered_body") or (
                None if via_template else content
            )
            fresh_contact = (
                self._get_contact(whatsapp_number=to_number) or contact or {}
            )
            attachments_for_event = [attachment] if attachment else []
            resolved_contact_id = fresh_contact.get("contact_id") or contact_id
            pending_resends = None
            automatic_resend_available = False
            if via_template and self._cm is not None:
                pending_resends = getattr(self._cm, "_pending_whatsapp_resends", None)
                automatic_resend_available = isinstance(pending_resends, dict)
            onboarding_kwargs = {}
            if via_template:
                if automatic_resend_available:
                    onboarding_kwargs = self._onboarding_event_kwargs(
                        Medium.WHATSAPP_MESSAGE,
                    )
                    self._stash_whatsapp_resend_onboarding_kwargs(
                        resolved_contact_id,
                        onboarding_kwargs,
                    )
                event_onboarding_kwargs = {}
            else:
                event_onboarding_kwargs = (
                    self._consume_whatsapp_resend_onboarding_kwargs(
                        resolved_contact_id,
                    )
                    or self._onboarding_event_kwargs(Medium.WHATSAPP_MESSAGE)
                )
            event = WhatsAppSent(
                contact=fresh_contact,
                content=content,
                via_template=via_template,
                delivered_content=delivered_content,
                attachments=attachments_for_event or None,
                **event_onboarding_kwargs,
            )
            await self._event_broker.publish(topic, event.to_json())
            self._record_offline_success(
                offline_reservation,
                attempted_content=content,
                receiver_ids=[resolved_contact_id],
                target_metadata={
                    "contact_id": resolved_contact_id,
                    "whatsapp_number": to_number or "",
                    "attachment_filepath": attachment_filepath or "",
                },
                history_metadata={
                    "contact_display_name": _get_contact_display_name(fresh_contact),
                    "via_template": via_template,
                    "automatic_resend_available": automatic_resend_available,
                },
                attachments=attachments_for_event or None,
                provider_response=response,
                status=(
                    "pending_resend"
                    if via_template and automatic_resend_available
                    else "completed"
                ),
            )
            if via_template:
                if (
                    automatic_resend_available
                    and pending_resends is not None
                    and resolved_contact_id is not None
                ):
                    pending_resends[resolved_contact_id] = content
                    return {
                        "status": "ok",
                        "pending_resend": True,
                        "note": (
                            "The message could not be delivered verbatim. "
                            "A notification was sent to the contact instead. "
                            "When they reply, you will be prompted to resend your message."
                        ),
                    }
                return {
                    "status": "ok",
                    "note": (
                        "The message could not be delivered verbatim. "
                        "A notification was sent to the contact instead. "
                        "Because this send ran without a live assistant session, "
                        "the original message was not queued for automatic resend."
                    ),
                }
            return {"status": "ok"}

        if not self._assistant_whatsapp_number():
            error_msg = "WhatsApp is not enabled for this assistant."
        else:
            error_msg = f"Failed to send WhatsApp message to {to_number}"
        return await self._surface_comms_error(
            error_msg,
            topic,
            contact_id=contact_id,
            medium=Medium.WHATSAPP_MESSAGE,
            offline_reservation=offline_reservation,
            attempted_content=content,
            receiver_ids=[contact_id],
            target_metadata={
                "contact_id": contact_id,
                "whatsapp_number": to_number or whatsapp_number or "",
                "attachment_filepath": attachment_filepath or "",
            },
            history_metadata={
                "contact_display_name": _get_contact_display_name(contact),
            },
            attachments=[attachment] if attachment else None,
        )

    async def send_discord_message(
        self,
        *,
        contact_id: int | str,
        content: str,
        discord_id: str | None = None,
    ) -> dict[str, Any]:
        """Send an assistant-owned Discord direct message to an existing contact.

        Use this for one-to-one Discord replies. For guild channel posts, use
        ``send_discord_channel_message`` instead.

        - If the contact already has a Discord ID on file, omit ``discord_id``.
        - If the contact is missing a Discord ID but you know it, pass it via
          ``discord_id`` and the contact record will be updated before the DM is
          sent.
        - Do not supply a different Discord ID from the one already on file.
          Update the contact first, then retry.

        Parameters
        ----------
        contact_id : int | str
            Recipient contact from ``active_conversations`` or from contact
            search/create tools.
        content : str
            Direct-message body to send.
        discord_id : str | None, optional
            Recipient Discord user ID when the contact does not already have one
            on file.

        Returns
        -------
        dict[str, Any]
            ``{"status": "ok"}`` on success, or an error payload describing
            why the DM could not be sent.
        """
        contact_id = _coerce_contact_id(contact_id)
        content = normalize_outbound_plain_text(content)
        offline_reservation = None
        contact = self._get_contact(contact_id=contact_id)
        topic = "app:comms:discord_message_sent"

        outbound_error = self._check_outbound_allowed(contact)
        if outbound_error:
            return await self._surface_comms_error(
                outbound_error,
                topic,
                contact_id=contact_id,
                medium=Medium.DISCORD_MESSAGE,
                offline_reservation=offline_reservation,
                attempted_content=content,
                receiver_ids=[contact_id],
                target_metadata={
                    "contact_id": contact_id,
                    "discord_id": (contact or {}).get("discord_id") or discord_id or "",
                },
                history_metadata={
                    "contact_display_name": _get_contact_display_name(contact),
                },
            )

        bot_id = self._assistant_discord_bot_id()
        if not bot_id:
            return await self._surface_comms_error(
                "Discord is not enabled for this assistant.",
                topic,
                contact_id=contact_id,
                medium=Medium.DISCORD_MESSAGE,
                attempted_content=content,
                receiver_ids=[contact_id],
                target_metadata={
                    "contact_id": contact_id,
                    "discord_id": (contact or {}).get("discord_id") or discord_id or "",
                },
                history_metadata={
                    "contact_display_name": _get_contact_display_name(contact),
                },
            )

        detail_error, contact = self._resolve_or_attach_detail(
            contact=contact,
            contact_id=contact_id,
            field_name="discord_id",
            inline_value=discord_id,
            medium_label="Discord",
        )
        if detail_error:
            return await self._surface_comms_error(
                detail_error,
                topic,
                contact_id=contact_id,
                medium=Medium.DISCORD_MESSAGE,
                offline_reservation=offline_reservation,
                attempted_content=content,
                receiver_ids=[contact_id],
                target_metadata={
                    "contact_id": contact_id,
                    "discord_id": (contact or {}).get("discord_id") or discord_id or "",
                },
                history_metadata={
                    "contact_display_name": _get_contact_display_name(contact),
                },
            )

        offline_reservation, offline_response = self._reserve_offline_operation(
            method_name="send_discord_message",
            medium=Medium.DISCORD_MESSAGE,
            target_kind="contact",
            target_metadata={
                "contact_id": (contact or {}).get("contact_id") or contact_id,
                "discord_id": (contact or {}).get("discord_id") or discord_id or "",
            },
            contact_id=(contact or {}).get("contact_id") or contact_id,
        )
        if offline_response is not None:
            return offline_response

        to_discord_id = (contact or {}).get("discord_id")
        response = await comms_utils.send_discord_message(
            to=to_discord_id,
            body=content,
            bot_id=bot_id,
        )
        if response.get("success"):
            fresh_contact = self._get_contact(discord_id=to_discord_id) or contact or {}
            event = DiscordMessageSent(
                contact=fresh_contact,
                content=content,
                **self._onboarding_event_kwargs(Medium.DISCORD_MESSAGE),
            )
            await self._event_broker.publish(topic, event.to_json())
            self._record_offline_success(
                offline_reservation,
                attempted_content=content,
                receiver_ids=[fresh_contact.get("contact_id") or contact_id],
                target_metadata={
                    "contact_id": fresh_contact.get("contact_id") or contact_id,
                    "discord_id": to_discord_id or "",
                },
                history_metadata={
                    "contact_display_name": _get_contact_display_name(fresh_contact),
                },
                provider_response=response,
            )
            return {"status": "ok"}

        if not bot_id:
            error_msg = "Discord is not enabled for this assistant."
        else:
            error_msg = "Failed to send Discord message"
        return await self._surface_comms_error(
            error_msg,
            topic,
            contact_id=contact_id,
            medium=Medium.DISCORD_MESSAGE,
            offline_reservation=offline_reservation,
            attempted_content=content,
            receiver_ids=[contact_id],
            target_metadata={
                "contact_id": contact_id,
                "discord_id": to_discord_id or discord_id or "",
            },
            history_metadata={
                "contact_display_name": _get_contact_display_name(contact),
            },
        )

    async def send_discord_channel_message(
        self,
        *,
        channel_id: str,
        content: str,
        guild_id: str = "",
        contact_id: int | str | None = None,
    ) -> dict[str, Any]:
        """Post an assistant-owned message into a Discord guild channel.

        Use this when the assistant should speak in a shared Discord channel
        rather than DM one person. The optional ``contact_id`` does not change
        where the message is sent; it only provides a contact anchor for
        transcript ownership and response-policy checks when the channel post is
        associated with a particular person or thread.

        If you want a one-to-one Discord reply, use
        ``send_discord_message`` instead.

        Parameters
        ----------
        channel_id : str
            Target Discord channel ID to post into.
        content : str
            Message body to publish.
        guild_id : str, optional
            Discord guild ID for transcript/event metadata when known.
        contact_id : int | str | None, optional
            Optional contact anchor for transcript ownership and response-policy
            checks.

        Returns
        -------
        dict[str, Any]
            Status payload with success or error details.
        """
        normalized_contact_id = (
            _coerce_contact_id(contact_id) if contact_id is not None else None
        )
        content = normalize_outbound_plain_text(content)
        offline_reservation = None
        resolved_contact = self._normalize_optional_contact(normalized_contact_id)
        if resolved_contact is not None:
            outbound_error = self._check_outbound_allowed(resolved_contact)
            if outbound_error:
                return await self._surface_comms_error(
                    outbound_error,
                    "app:comms:discord_channel_message_sent",
                    contact_id=resolved_contact.get("contact_id"),
                    medium=Medium.DISCORD_CHANNEL_MESSAGE,
                    offline_reservation=offline_reservation,
                    attempted_content=content,
                    receiver_ids=[resolved_contact.get("contact_id")],
                    target_metadata={
                        "channel_id": channel_id,
                        "guild_id": guild_id,
                        "contact_id": resolved_contact.get("contact_id"),
                    },
                    history_metadata={
                        "contact_display_name": _get_contact_display_name(
                            resolved_contact,
                        ),
                    },
                )
        anchor_contact = resolved_contact or self._assistant_anchor_contact()
        anchor_contact_id = anchor_contact.get("contact_id")
        bot_id = self._assistant_discord_bot_id()
        if not bot_id:
            return await self._surface_comms_error(
                "Discord is not enabled for this assistant.",
                "app:comms:discord_channel_message_sent",
                contact_id=anchor_contact_id,
                medium=Medium.DISCORD_CHANNEL_MESSAGE,
                attempted_content=content,
                receiver_ids=(
                    [anchor_contact_id] if anchor_contact_id is not None else None
                ),
                target_metadata={
                    "channel_id": channel_id,
                    "guild_id": guild_id,
                    "contact_id": anchor_contact_id,
                },
                history_metadata={
                    "contact_display_name": _get_contact_display_name(anchor_contact),
                },
            )
        offline_reservation, offline_response = self._reserve_offline_operation(
            method_name="send_discord_channel_message",
            medium=Medium.DISCORD_CHANNEL_MESSAGE,
            target_kind="discord_channel",
            target_metadata={
                "channel_id": channel_id,
                "guild_id": guild_id,
                "contact_id": normalized_contact_id,
            },
            contact_id=normalized_contact_id,
        )
        if offline_response is not None:
            return offline_response
        response = await comms_utils.send_discord_message(
            channel_id=channel_id,
            body=content,
            bot_id=bot_id,
        )
        if response.get("success"):
            event = DiscordChannelMessageSent(
                contact=anchor_contact,
                content=content,
                channel_id=channel_id,
                guild_id=guild_id,
                **self._onboarding_event_kwargs(Medium.DISCORD_CHANNEL_MESSAGE),
            )
            await self._event_broker.publish(
                "app:comms:discord_channel_message_sent",
                event.to_json(),
            )
            self._record_offline_success(
                offline_reservation,
                attempted_content=content,
                receiver_ids=(
                    [anchor_contact_id] if anchor_contact_id is not None else None
                ),
                target_metadata={
                    "channel_id": channel_id,
                    "guild_id": guild_id,
                    "contact_id": anchor_contact_id,
                },
                history_metadata={
                    "contact_display_name": _get_contact_display_name(anchor_contact),
                },
                provider_response=response,
            )
            return {"status": "ok"}

        if not bot_id:
            error_msg = "Discord is not enabled for this assistant."
        else:
            error_msg = "Failed to send Discord channel message"
        return await self._surface_comms_error(
            error_msg,
            "app:comms:discord_channel_message_sent",
            contact_id=anchor_contact_id,
            medium=Medium.DISCORD_CHANNEL_MESSAGE,
            offline_reservation=offline_reservation,
            attempted_content=content,
            receiver_ids=[anchor_contact_id] if anchor_contact_id is not None else None,
            target_metadata={
                "channel_id": channel_id,
                "guild_id": guild_id,
                "contact_id": anchor_contact_id,
            },
            history_metadata={
                "contact_display_name": _get_contact_display_name(anchor_contact),
            },
        )

    async def send_slack_message(
        self,
        *,
        contact_id: int | str,
        content: str,
        team_id: str,
        slack_user_id: str | None = None,
        thread_ts: str | None = None,
    ) -> dict[str, Any]:
        """Send an assistant-owned Slack direct message to an existing contact.

        Use this for one-to-one Slack DMs. For channel posts or threaded
        replies in a channel, use ``send_slack_channel_message`` instead.

        - If the contact already has a Slack user ID on file, omit
          ``slack_user_id``.
        - If the contact is missing a Slack user ID but you know it, pass it
          via ``slack_user_id`` and the contact record will be updated before
          the DM is sent.
        - Do not supply a different Slack user ID from the one already on
          file. Update the contact first, then retry.

        Parameters
        ----------
        contact_id : int | str
            Recipient contact from ``active_conversations`` or from contact
            search/create tools.
        content : str
            Direct-message body to send.
        team_id : str
            Slack workspace ID the bot token belongs to. Required to resolve
            the workteam-scoped bot token server-side.
        slack_user_id : str | None, optional
            Recipient Slack user ID when the contact does not already have
            one on file.
        thread_ts : str | None, optional
            Slack thread timestamp to reply inside an existing DM thread.

        Returns
        -------
        dict[str, Any]
            ``{"status": "ok"}`` on success, or an error payload describing
            why the DM could not be sent.
        """
        contact_id = _coerce_contact_id(contact_id)
        content = normalize_outbound_plain_text(content)
        offline_reservation = None
        contact = self._get_contact(contact_id=contact_id)
        topic = "app:comms:slack_message_sent"

        outbound_error = self._check_outbound_allowed(contact)
        if outbound_error:
            return await self._surface_comms_error(
                outbound_error,
                topic,
                contact_id=contact_id,
                medium=Medium.SLACK_MESSAGE,
                offline_reservation=offline_reservation,
                attempted_content=content,
                receiver_ids=[contact_id],
                target_metadata={
                    "contact_id": contact_id,
                    "team_id": team_id,
                    "slack_user_id": (contact or {}).get("slack_user_id")
                    or slack_user_id
                    or "",
                },
                history_metadata={
                    "contact_display_name": _get_contact_display_name(contact),
                },
            )

        bot_user_id = self._assistant_slack_bot_user_id()
        if not bot_user_id:
            return await self._surface_comms_error(
                "Slack is not enabled for this assistant.",
                topic,
                contact_id=contact_id,
                medium=Medium.SLACK_MESSAGE,
                attempted_content=content,
                receiver_ids=[contact_id],
                target_metadata={
                    "contact_id": contact_id,
                    "team_id": team_id,
                    "slack_user_id": (contact or {}).get("slack_user_id")
                    or slack_user_id
                    or "",
                },
                history_metadata={
                    "contact_display_name": _get_contact_display_name(contact),
                },
            )

        # Reach a workspace member we've never received a Slack message from:
        # when the contact has an email but no Slack user ID on file (and none
        # was supplied inline), resolve it from the email via the workspace
        # bot (users.lookupByEmail). _resolve_or_attach_detail then persists it
        # onto the contact, so this lookup happens at most once per contact.
        if (
            slack_user_id is None
            and contact is not None
            and not contact.get("slack_user_id")
            and contact.get("email_address")
        ):
            slack_user_id = await comms_utils.resolve_slack_user_id_by_email(
                team_id=team_id,
                email=contact["email_address"],
            )

        detail_error, contact = self._resolve_or_attach_detail(
            contact=contact,
            contact_id=contact_id,
            field_name="slack_user_id",
            inline_value=slack_user_id,
            medium_label="Slack",
        )
        if detail_error:
            return await self._surface_comms_error(
                detail_error,
                topic,
                contact_id=contact_id,
                medium=Medium.SLACK_MESSAGE,
                offline_reservation=offline_reservation,
                attempted_content=content,
                receiver_ids=[contact_id],
                target_metadata={
                    "contact_id": contact_id,
                    "team_id": team_id,
                    "slack_user_id": (contact or {}).get("slack_user_id")
                    or slack_user_id
                    or "",
                },
                history_metadata={
                    "contact_display_name": _get_contact_display_name(contact),
                },
            )

        offline_reservation, offline_response = self._reserve_offline_operation(
            method_name="send_slack_message",
            medium=Medium.SLACK_MESSAGE,
            target_kind="contact",
            target_metadata={
                "contact_id": (contact or {}).get("contact_id") or contact_id,
                "team_id": team_id,
                "slack_user_id": (contact or {}).get("slack_user_id")
                or slack_user_id
                or "",
                "thread_ts": thread_ts or "",
            },
            contact_id=(contact or {}).get("contact_id") or contact_id,
        )
        if offline_response is not None:
            return offline_response

        to_slack_user_id = (contact or {}).get("slack_user_id")
        response = await comms_utils.send_slack_message(
            team_id=team_id,
            user_id=to_slack_user_id,
            body=content,
            thread_ts=thread_ts,
        )
        if response.get("success"):
            fresh_contact = (
                self._get_contact(slack_user_id=to_slack_user_id) or contact or {}
            )
            event = SlackMessageSent(
                contact=fresh_contact,
                content=content,
                team_id=team_id,
                channel_id=response.get("channel_id", ""),
                thread_ts=thread_ts or "",
                **self._onboarding_event_kwargs(Medium.SLACK_MESSAGE),
            )
            await self._event_broker.publish(topic, event.to_json())
            self._record_offline_success(
                offline_reservation,
                attempted_content=content,
                receiver_ids=[fresh_contact.get("contact_id") or contact_id],
                target_metadata={
                    "contact_id": fresh_contact.get("contact_id") or contact_id,
                    "team_id": team_id,
                    "slack_user_id": to_slack_user_id or "",
                    "thread_ts": thread_ts or "",
                },
                history_metadata={
                    "contact_display_name": _get_contact_display_name(fresh_contact),
                },
                provider_response=response,
            )
            return {"status": "ok"}

        return await self._surface_comms_error(
            "Failed to send Slack message",
            topic,
            contact_id=contact_id,
            medium=Medium.SLACK_MESSAGE,
            offline_reservation=offline_reservation,
            attempted_content=content,
            receiver_ids=[contact_id],
            target_metadata={
                "contact_id": contact_id,
                "team_id": team_id,
                "slack_user_id": to_slack_user_id or slack_user_id or "",
                "thread_ts": thread_ts or "",
            },
            history_metadata={
                "contact_display_name": _get_contact_display_name(contact),
            },
        )

    async def send_slack_channel_message(
        self,
        *,
        channel_id: str,
        content: str,
        team_id: str,
        thread_ts: str | None = None,
        contact_id: int | str | None = None,
    ) -> dict[str, Any]:
        """Post an assistant-owned message into a Slack channel.

        Use this when the assistant should speak in a shared Slack channel
        rather than DM one person. When replying to an inbound channel
        message, pass the ``thread_ts`` surfaced on the inbound line so the
        reply lands in a thread under that message; for a top-level
        @mention the surfaced value is the original message's own id, which
        starts the thread. Omit ``thread_ts`` only to deliberately post a
        fresh message at the channel root. The optional
        ``contact_id`` does not change where the message is sent; it only
        provides a contact anchor for transcript ownership and response-
        policy checks when the channel post is associated with a particular
        person or thread.

        If you want a one-to-one Slack reply, use ``send_slack_message``
        instead.

        Parameters
        ----------
        channel_id : str
            Target Slack channel ID to post into.
        content : str
            Message body to publish.
        team_id : str
            Slack workspace ID (used to resolve the bot token server-side).
        thread_ts : str | None, optional
            Slack thread timestamp to reply inside a thread. Pass the value
            surfaced on the inbound message line to keep channel replies
            threaded under the original message.
        contact_id : int | str | None, optional
            Optional contact anchor for transcript ownership and response-
            policy checks.

        Returns
        -------
        dict[str, Any]
            Status payload with success or error details.
        """
        normalized_contact_id = (
            _coerce_contact_id(contact_id) if contact_id is not None else None
        )
        content = normalize_outbound_plain_text(content)
        offline_reservation = None
        resolved_contact = self._normalize_optional_contact(normalized_contact_id)
        if resolved_contact is not None:
            outbound_error = self._check_outbound_allowed(resolved_contact)
            if outbound_error:
                return await self._surface_comms_error(
                    outbound_error,
                    "app:comms:slack_channel_message_sent",
                    contact_id=resolved_contact.get("contact_id"),
                    medium=Medium.SLACK_CHANNEL_MESSAGE,
                    offline_reservation=offline_reservation,
                    attempted_content=content,
                    receiver_ids=[resolved_contact.get("contact_id")],
                    target_metadata={
                        "channel_id": channel_id,
                        "team_id": team_id,
                        "thread_ts": thread_ts or "",
                        "contact_id": resolved_contact.get("contact_id"),
                    },
                    history_metadata={
                        "contact_display_name": _get_contact_display_name(
                            resolved_contact,
                        ),
                    },
                )
        anchor_contact = resolved_contact or self._assistant_anchor_contact()
        anchor_contact_id = anchor_contact.get("contact_id")
        bot_user_id = self._assistant_slack_bot_user_id()
        if not bot_user_id:
            return await self._surface_comms_error(
                "Slack is not enabled for this assistant.",
                "app:comms:slack_channel_message_sent",
                contact_id=anchor_contact_id,
                medium=Medium.SLACK_CHANNEL_MESSAGE,
                attempted_content=content,
                receiver_ids=(
                    [anchor_contact_id] if anchor_contact_id is not None else None
                ),
                target_metadata={
                    "channel_id": channel_id,
                    "team_id": team_id,
                    "thread_ts": thread_ts or "",
                    "contact_id": anchor_contact_id,
                },
                history_metadata={
                    "contact_display_name": _get_contact_display_name(anchor_contact),
                },
            )
        offline_reservation, offline_response = self._reserve_offline_operation(
            method_name="send_slack_channel_message",
            medium=Medium.SLACK_CHANNEL_MESSAGE,
            target_kind="slack_channel",
            target_metadata={
                "channel_id": channel_id,
                "team_id": team_id,
                "thread_ts": thread_ts or "",
                "contact_id": normalized_contact_id,
            },
            contact_id=normalized_contact_id,
        )
        if offline_response is not None:
            return offline_response
        response = await comms_utils.send_slack_message(
            team_id=team_id,
            channel_id=channel_id,
            body=content,
            thread_ts=thread_ts,
        )
        if response.get("success"):
            event = SlackChannelMessageSent(
                contact=anchor_contact,
                content=content,
                team_id=team_id,
                channel_id=channel_id,
                thread_ts=thread_ts or "",
                **self._onboarding_event_kwargs(Medium.SLACK_CHANNEL_MESSAGE),
            )
            await self._event_broker.publish(
                "app:comms:slack_channel_message_sent",
                event.to_json(),
            )
            self._record_offline_success(
                offline_reservation,
                attempted_content=content,
                receiver_ids=(
                    [anchor_contact_id] if anchor_contact_id is not None else None
                ),
                target_metadata={
                    "channel_id": channel_id,
                    "team_id": team_id,
                    "thread_ts": thread_ts or "",
                    "contact_id": anchor_contact_id,
                },
                history_metadata={
                    "contact_display_name": _get_contact_display_name(anchor_contact),
                },
                provider_response=response,
            )
            return {"status": "ok"}

        return await self._surface_comms_error(
            "Failed to send Slack channel message",
            "app:comms:slack_channel_message_sent",
            contact_id=anchor_contact_id,
            medium=Medium.SLACK_CHANNEL_MESSAGE,
            offline_reservation=offline_reservation,
            attempted_content=content,
            receiver_ids=[anchor_contact_id] if anchor_contact_id is not None else None,
            target_metadata={
                "channel_id": channel_id,
                "team_id": team_id,
                "thread_ts": thread_ts or "",
                "contact_id": anchor_contact_id,
            },
            history_metadata={
                "contact_display_name": _get_contact_display_name(anchor_contact),
            },
        )

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
        """Send an assistant-owned Microsoft Teams message.

        Teams runs in three mutually-exclusive modes; exactly one applies per
        call:

        - **Chat reply** (existing 1:1, group, or meeting chat): pass
          ``chat_id``. ``contact_id`` must be a single value identifying the
          recipient for response-policy checks and transcript ownership.
        - **Channel post**: pass both ``team_id`` and ``channel_id``.
          ``contact_id`` must be a single value that anchors the post to a
          contact thread for response-policy checks and transcript
          attribution; it does not change where the message is delivered.
        - **New chat (find-or-create)**: omit ``chat_id``, ``team_id``, and
          ``channel_id``. Pass one or more ``contact_id`` recipients; a 1:1
          DM is created when exactly one recipient is supplied, and a group
          chat when two or more are supplied. Microsoft Graph dedupes 1:1
          chats, so repeat calls with the same single recipient reuse the
          same ``chat_id``. ``chat_topic`` is only valid when creating a
          group chat. Each recipient may be provided as a bare
          ``contact_id`` when the contact already has an ``email_address``
          on file, or as ``{"contact_id": ..., "email_address": ...}`` when
          the address needs to be attached during the send (same shape as
          ``send_email``). The first recipient anchors the transcript.

        Teams accepts one attachment per outbound message. Supply a workspace
        file path via ``attachment_filepath`` and Communication will upload
        it through OneDrive before posting.

        Parameters
        ----------
        contact_id : int | str | list[int | str | dict]
            Contact anchor(s) for the send. Accepts a single value for chat
            replies and channel posts, or a list for the find-or-create
            chat mode.
        content : str
            Message body to send.
        chat_id : str | None, optional
            Teams chat ID for replying into an existing chat.
        channel_id : str | None, optional
            Teams channel ID. Required together with ``team_id`` for channel
            posts.
        team_id : str | None, optional
            Teams team ID. Required together with ``channel_id`` for channel
            posts.
        chat_topic : str | None, optional
            Topic for a newly created group chat. Only valid in find-or-
            create mode with two or more recipients.
        attachment_filepath : str | None, optional
            Workspace-relative file path for one attachment to include with
            the message.

        Returns
        -------
        dict[str, Any]
            ``{"status": "ok"}`` on success, or an error payload describing
            why the send could not complete.
        """
        content = normalize_outbound_plain_text(content)
        offline_reservation = None
        is_channel = bool(channel_id and team_id)
        is_chat_reply = bool(chat_id) and not is_channel
        is_find_or_create = not is_channel and not is_chat_reply

        medium = Medium.TEAMS_CHANNEL_MESSAGE if is_channel else Medium.TEAMS_MESSAGE
        topic = (
            "app:comms:teams_channel_message_sent"
            if is_channel
            else "app:comms:teams_message_sent"
        )
        target_kind = "teams_channel" if is_channel else "contact"

        raw_contact_id = contact_id
        recipients_raw: list[int | str | dict] = (
            list(raw_contact_id)
            if isinstance(raw_contact_id, list)
            else [raw_contact_id]
        )
        if not recipients_raw:
            return await self._surface_comms_error(
                "At least one contact_id is required for Teams sends",
                topic,
                medium=medium,
                attempted_content=content,
                target_metadata={
                    "chat_id": chat_id or "",
                    "team_id": team_id or "",
                    "channel_id": channel_id or "",
                    "attachment_filepath": attachment_filepath or "",
                },
            )

        try:
            parsed_recipients: list[tuple[int, str | None]] = []
            for item in recipients_raw:
                if isinstance(item, dict):
                    raw = item.get("contact_id")
                    if raw is None:
                        raise TypeError(
                            "Teams recipient dict must include 'contact_id', "
                            f"got: {item!r}",
                        )
                    parsed_recipients.append(
                        (_coerce_contact_id(raw), item.get("email_address")),
                    )
                else:
                    parsed_recipients.append((_coerce_contact_id(item), None))
        except TypeError as exc:
            return await self._surface_comms_error(
                str(exc),
                topic,
                medium=medium,
                attempted_content=content,
                target_metadata={
                    "chat_id": chat_id or "",
                    "team_id": team_id or "",
                    "channel_id": channel_id or "",
                    "attachment_filepath": attachment_filepath or "",
                },
            )

        anchor_contact_id = parsed_recipients[0][0]
        contact = self._get_contact(contact_id=anchor_contact_id)

        def _target_metadata() -> dict[str, Any]:
            base: dict[str, Any] = {
                "contact_id": (contact or {}).get("contact_id") or anchor_contact_id,
                "attachment_filepath": attachment_filepath or "",
            }
            if is_channel:
                base["team_id"] = team_id or ""
                base["channel_id"] = channel_id or ""
            else:
                base["chat_id"] = chat_id or ""
            return base

        def _history_metadata() -> dict[str, Any]:
            return {
                "contact_display_name": _get_contact_display_name(contact),
            }

        if not self._assistant_has_teams():
            return await self._surface_comms_error(
                "Microsoft Teams is not enabled for this assistant.",
                topic,
                contact_id=anchor_contact_id,
                medium=medium,
                attempted_content=content,
                receiver_ids=[anchor_contact_id],
                target_metadata=_target_metadata(),
                history_metadata=_history_metadata(),
            )

        if not is_find_or_create and len(parsed_recipients) > 1:
            return await self._surface_comms_error(
                "Multiple contact_ids are only supported when creating a new "
                "Teams chat. For a chat reply or channel post, pass a single "
                "contact_id.",
                topic,
                contact_id=anchor_contact_id,
                medium=medium,
                attempted_content=content,
                receiver_ids=[anchor_contact_id],
                target_metadata=_target_metadata(),
                history_metadata=_history_metadata(),
            )

        if chat_topic and not is_find_or_create:
            return await self._surface_comms_error(
                "chat_topic is only valid when creating a new group chat "
                "(omit chat_id and team_id/channel_id, pass 2+ contact_ids).",
                topic,
                contact_id=anchor_contact_id,
                medium=medium,
                attempted_content=content,
                receiver_ids=[anchor_contact_id],
                target_metadata=_target_metadata(),
                history_metadata=_history_metadata(),
            )

        if chat_topic and is_find_or_create and len(parsed_recipients) < 2:
            return await self._surface_comms_error(
                "chat_topic is only valid for group chats (2+ recipients).",
                topic,
                contact_id=anchor_contact_id,
                medium=medium,
                attempted_content=content,
                receiver_ids=[anchor_contact_id],
                target_metadata=_target_metadata(),
                history_metadata=_history_metadata(),
            )

        outbound_error = self._check_outbound_allowed(contact)
        if outbound_error:
            return await self._surface_comms_error(
                outbound_error,
                topic,
                contact_id=anchor_contact_id,
                medium=medium,
                offline_reservation=offline_reservation,
                attempted_content=content,
                receiver_ids=[anchor_contact_id],
                target_metadata=_target_metadata(),
                history_metadata=_history_metadata(),
            )

        resolved_recipient_ids: list[int] = [anchor_contact_id]

        if is_find_or_create:
            member_emails: list[str] = []
            seen_emails: set[str] = set()
            resolved_recipient_ids = []
            for recipient_id, inline_email in parsed_recipients:
                recipient_contact = self._get_contact(contact_id=recipient_id)
                error, resolved_contact = self._resolve_or_attach_detail(
                    contact=recipient_contact,
                    contact_id=recipient_id,
                    field_name="email_address",
                    inline_value=inline_email,
                    medium_label="Teams",
                )
                if error:
                    return await self._surface_comms_error(
                        error,
                        topic,
                        contact_id=recipient_id,
                        medium=medium,
                        attempted_content=content,
                        receiver_ids=[anchor_contact_id],
                        target_metadata=_target_metadata(),
                        history_metadata=_history_metadata(),
                    )
                email_address = (resolved_contact or {}).get("email_address")
                if not email_address:
                    return await self._surface_comms_error(
                        f"Could not resolve email address for contact_id={recipient_id}",
                        topic,
                        contact_id=recipient_id,
                        medium=medium,
                        attempted_content=content,
                        receiver_ids=[anchor_contact_id],
                        target_metadata=_target_metadata(),
                        history_metadata=_history_metadata(),
                    )
                key = email_address.lower()
                if key in seen_emails:
                    continue
                seen_emails.add(key)
                member_emails.append(email_address)
                resolved_recipient_ids.append(
                    (resolved_contact or {}).get("contact_id") or recipient_id,
                )

            chat_type = "oneOnOne" if len(member_emails) == 1 else "group"
            create_response = await comms_utils.create_teams_chat(
                chat_type=chat_type,
                member_emails=member_emails,
                topic=chat_topic if chat_type == "group" else None,
            )
            if not create_response.get("success"):
                return await self._surface_comms_error(
                    create_response.get("error") or "Failed to create Teams chat",
                    topic,
                    contact_id=anchor_contact_id,
                    medium=medium,
                    attempted_content=content,
                    receiver_ids=[anchor_contact_id],
                    target_metadata=_target_metadata(),
                    history_metadata=_history_metadata(),
                )
            chat_id = create_response.get("chat_id")
            if not chat_id:
                return await self._surface_comms_error(
                    "Teams chat create returned no chat_id",
                    topic,
                    contact_id=anchor_contact_id,
                    medium=medium,
                    attempted_content=content,
                    receiver_ids=[anchor_contact_id],
                    target_metadata=_target_metadata(),
                    history_metadata=_history_metadata(),
                )
            contact = self._get_contact(contact_id=anchor_contact_id) or contact

        offline_reservation, offline_response = self._reserve_offline_operation(
            method_name="send_teams_message",
            medium=medium,
            target_kind=target_kind,
            target_metadata=_target_metadata(),
            contact_id=(contact or {}).get("contact_id") or anchor_contact_id,
        )
        if offline_response is not None:
            return offline_response

        outbound_attachments: list[dict] | None = None
        attachment_meta: dict[str, Any] | None = None
        if attachment_filepath:
            from unify.file_manager.filesystem_adapters.local_adapter import (
                LocalFileSystemAdapter,
            )

            adapter = LocalFileSystemAdapter()
            try:
                abs_path = adapter._abspath(attachment_filepath)
                with open(abs_path, "rb") as file_handle:
                    file_contents = file_handle.read()
            except FileNotFoundError:
                return await self._surface_comms_error(
                    f"File not found: {attachment_filepath}",
                    topic,
                    contact_id=anchor_contact_id,
                    medium=medium,
                    offline_reservation=offline_reservation,
                    attempted_content=content,
                    receiver_ids=[anchor_contact_id],
                    target_metadata=_target_metadata(),
                    history_metadata=_history_metadata(),
                )
            except Exception as exc:
                return await self._surface_comms_error(
                    f"Failed to read file: {exc}",
                    topic,
                    contact_id=anchor_contact_id,
                    medium=medium,
                    offline_reservation=offline_reservation,
                    attempted_content=content,
                    receiver_ids=[anchor_contact_id],
                    target_metadata=_target_metadata(),
                    history_metadata=_history_metadata(),
                )

            filename = os.path.basename(attachment_filepath)
            outbound_attachments = [
                {
                    "filename": filename,
                    "content_base64": base64.b64encode(file_contents).decode("ascii"),
                },
            ]
            attachment_meta = {"filename": filename, "filepath": attachment_filepath}

        response = await comms_utils.send_teams_message(
            chat_id=chat_id,
            team_id=team_id,
            channel_id=channel_id,
            body=content,
            attachments=outbound_attachments,
        )

        if response.get("success"):
            fresh_contact = (
                self._get_contact(
                    contact_id=(contact or {}).get("contact_id") or anchor_contact_id,
                )
                or contact
                or {}
            )
            attachments_for_event = [attachment_meta] if attachment_meta else None
            if is_channel:
                # Channel sends: unity has no local roster; let the downstream
                # receiver_ids derivation fall back to its default using the
                # message's target contact_id.
                event = TeamsChannelMessageSent(
                    contact=fresh_contact,
                    content=content,
                    channel_id=channel_id or "",
                    team_id=team_id or "",
                    attachments=attachments_for_event,
                    participants=[],
                )
            else:
                participants_list: list[int] = [SESSION_DETAILS.self_contact_id]
                for rid in resolved_recipient_ids:
                    if rid is not None and rid != SESSION_DETAILS.self_contact_id:
                        participants_list.append(rid)
                event = TeamsMessageSent(
                    contact=fresh_contact,
                    content=content,
                    chat_id=chat_id or "",
                    attachments=attachments_for_event,
                    participants=sorted(set(participants_list)),
                )
            await self._event_broker.publish(topic, event.to_json())
            self._record_offline_success(
                offline_reservation,
                attempted_content=content,
                receiver_ids=[fresh_contact.get("contact_id") or anchor_contact_id],
                target_metadata=_target_metadata(),
                history_metadata={
                    "contact_display_name": _get_contact_display_name(fresh_contact),
                },
                provider_response=response,
                attachments=[attachment_meta] if attachment_meta else None,
            )
            return {"status": "ok"}

        return await self._surface_comms_error(
            response.get("error") or "Failed to send Teams message",
            topic,
            contact_id=anchor_contact_id,
            medium=medium,
            offline_reservation=offline_reservation,
            attempted_content=content,
            receiver_ids=[anchor_contact_id],
            target_metadata=_target_metadata(),
            history_metadata=_history_metadata(),
            attachments=[attachment_meta] if attachment_meta else None,
        )

    async def create_teams_channel(
        self,
        *,
        team_id: str,
        display_name: str,
        description: str | None = None,
        membership_type: str = "standard",
        owner_contact_ids: list[int | str | dict] | None = None,
    ) -> dict[str, Any]:
        """Create a new channel inside an existing Microsoft Teams team.

        This is a structural operation separate from sending a message. After
        the channel is created, use ``send_teams_message`` with the returned
        ``team_id`` and ``channel_id`` to post into it.

        Membership modes:

        - ``"standard"``: open to all team members; no explicit owners.
        - ``"private"``: restricted to listed owners. ``owner_contact_ids``
          is required.
        - ``"shared"``: shareable with other teams/tenants. ``owner_contact_ids``
          is required.

        Each ``owner_contact_ids`` entry may be a bare ``contact_id`` when the
        contact has an ``email_address`` on file, or ``{"contact_id": ...,
        "email_address": ...}`` to attach the address during the call.

        Parameters
        ----------
        team_id : str
            ID of the existing team to create the channel within.
        display_name : str
            Display name for the new channel.
        description : str | None, optional
            Channel description.
        membership_type : str, optional
            ``"standard"`` (default), ``"private"``, or ``"shared"``.
        owner_contact_ids : list[int | str | dict] | None, optional
            Channel owners. Required for ``"private"`` and ``"shared"``;
            ignored for ``"standard"``.

        Returns
        -------
        dict[str, Any]
            ``{"status": "ok", "team_id": ..., "channel_id": ...}`` on
            success, or an error payload describing why the create failed.
        """
        topic = "app:comms:teams_channel_created"
        medium = Medium.TEAMS_CHANNEL_MESSAGE
        anchor_contact = self._assistant_anchor_contact()
        anchor_contact_id = anchor_contact.get("contact_id")

        def _target_metadata() -> dict[str, Any]:
            return {
                "team_id": team_id,
                "display_name": display_name,
                "description": description or "",
                "membership_type": membership_type,
            }

        def _history_metadata() -> dict[str, Any]:
            return {
                "contact_display_name": _get_contact_display_name(anchor_contact),
            }

        if not self._assistant_has_teams():
            return await self._surface_comms_error(
                "Microsoft Teams is not enabled for this assistant.",
                topic,
                contact_id=anchor_contact_id,
                medium=medium,
                attempted_content=f"create channel {display_name} in team {team_id}",
                target_metadata=_target_metadata(),
                history_metadata=_history_metadata(),
            )

        if membership_type not in ("standard", "private", "shared"):
            return await self._surface_comms_error(
                "membership_type must be 'standard', 'private', or 'shared'",
                topic,
                contact_id=anchor_contact_id,
                medium=medium,
                attempted_content=f"create channel {display_name} in team {team_id}",
                target_metadata=_target_metadata(),
                history_metadata=_history_metadata(),
            )

        requires_owners = membership_type != "standard"
        if requires_owners and not owner_contact_ids:
            return await self._surface_comms_error(
                f"{membership_type} channels require at least one owner "
                "(pass owner_contact_ids).",
                topic,
                contact_id=anchor_contact_id,
                medium=medium,
                attempted_content=f"create channel {display_name} in team {team_id}",
                target_metadata=_target_metadata(),
                history_metadata=_history_metadata(),
            )

        owner_emails: list[str] = []
        if owner_contact_ids:
            try:
                parsed_owners: list[tuple[int, str | None]] = []
                for item in owner_contact_ids:
                    if isinstance(item, dict):
                        raw = item.get("contact_id")
                        if raw is None:
                            raise TypeError(
                                "Owner dict must include 'contact_id', "
                                f"got: {item!r}",
                            )
                        parsed_owners.append(
                            (_coerce_contact_id(raw), item.get("email_address")),
                        )
                    else:
                        parsed_owners.append((_coerce_contact_id(item), None))
            except TypeError as exc:
                return await self._surface_comms_error(
                    str(exc),
                    topic,
                    contact_id=anchor_contact_id,
                    medium=medium,
                    attempted_content=(
                        f"create channel {display_name} in team {team_id}"
                    ),
                    target_metadata=_target_metadata(),
                    history_metadata=_history_metadata(),
                )

            seen_emails: set[str] = set()
            for owner_id, inline_email in parsed_owners:
                owner_contact = self._get_contact(contact_id=owner_id)
                error, resolved_owner = self._resolve_or_attach_detail(
                    contact=owner_contact,
                    contact_id=owner_id,
                    field_name="email_address",
                    inline_value=inline_email,
                    medium_label="Teams",
                )
                if error:
                    return await self._surface_comms_error(
                        error,
                        topic,
                        contact_id=owner_id,
                        medium=medium,
                        attempted_content=(
                            f"create channel {display_name} in team {team_id}"
                        ),
                        target_metadata=_target_metadata(),
                        history_metadata=_history_metadata(),
                    )
                email_address = (resolved_owner or {}).get("email_address")
                if not email_address:
                    return await self._surface_comms_error(
                        f"Could not resolve email address for owner contact_id={owner_id}",
                        topic,
                        contact_id=owner_id,
                        medium=medium,
                        attempted_content=(
                            f"create channel {display_name} in team {team_id}"
                        ),
                        target_metadata=_target_metadata(),
                        history_metadata=_history_metadata(),
                    )
                key = email_address.lower()
                if key in seen_emails:
                    continue
                seen_emails.add(key)
                owner_emails.append(email_address)

        response = await comms_utils.create_teams_channel(
            team_id=team_id,
            display_name=display_name,
            description=description,
            membership_type=membership_type,
            owner_emails=owner_emails if owner_emails else None,
        )

        if response.get("success"):
            channel_id = response.get("channel_id") or ""
            event = TeamsChannelCreated(
                contact=anchor_contact,
                team_id=team_id,
                channel_id=channel_id,
                display_name=display_name,
                description=description or "",
                membership_type=membership_type,
            )
            await self._event_broker.publish(topic, event.to_json())
            return {
                "status": "ok",
                "team_id": team_id,
                "channel_id": channel_id,
            }

        return await self._surface_comms_error(
            response.get("error") or "Failed to create Teams channel",
            topic,
            contact_id=anchor_contact_id,
            medium=medium,
            attempted_content=f"create channel {display_name} in team {team_id}",
            target_metadata=_target_metadata(),
            history_metadata=_history_metadata(),
        )

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
        """Create a Microsoft Teams meeting via Graph and return the join URL.

        Two explicit modes:

        - ``"scheduled"`` (default): creates a calendar event with an attached
          Teams meeting. ``subject`` is required. ``start`` defaults to five
          minutes from now (Graph requires a future start). ``end`` is computed
          as ``start + duration_minutes``. ``attendee_contact_ids`` are
          resolved to email addresses; Outlook invites are sent automatically.
          The meeting appears on calendars and generates invites.
        - ``"instant"``: creates a reusable Teams meeting with no calendar
          entry and no invites. ``subject`` is optional;
          ``start``/``duration_minutes``/``attendee_contact_ids`` are ignored
          (attendees are not contact-resolved in this mode). The returned
          ``join_web_url`` can be shared or passed to ``join_teams_meet``.

        ``body_html`` is forwarded verbatim — the communication service sends
        it to Graph with ``contentType=HTML``. Pass pre-rendered HTML or plain
        text (Graph will render plain text literally).

        Each ``attendee_contact_ids`` entry may be a bare ``contact_id`` when
        the contact has an ``email_address`` on file, or
        ``{"contact_id": ..., "email_address": ...}`` to attach the address
        during the call.

        Parameters
        ----------
        mode : str, optional
            ``"scheduled"`` (default) or ``"instant"``.
        subject : str | None, optional
            Meeting subject. Required for ``"scheduled"`` mode.
        start : str | None, optional
            ISO-8601 start timestamp. Defaults to ``now + 5min`` for
            ``"scheduled"`` mode; ignored for ``"instant"``.
        duration_minutes : int, optional
            Meeting duration in minutes; used to compute ``end`` for
            ``"scheduled"`` mode. Default 30.
        timezone : str, optional
            Timezone name forwarded to Graph. Default ``"UTC"``.
        attendee_contact_ids : list[int | str | dict] | None, optional
            Attendees for ``"scheduled"`` mode; ignored for ``"instant"``.
        body_html : str | None, optional
            Meeting body, sent to Graph as HTML.
        location : str | None, optional
            Display-name location for the calendar event (scheduled mode).

        Returns
        -------
        dict[str, Any]
            ``{"status": "ok", "mode", "join_web_url", "meeting_id",
            "event_id", "subject", "start", "end", "web_link"}`` on success,
            or an error payload describing why the create failed.
        """
        topic = "app:comms:teams_meet_created"
        medium = Medium.TEAMS_MEET
        anchor_contact = self._assistant_anchor_contact()
        anchor_contact_id = anchor_contact.get("contact_id")

        def _target_metadata() -> dict[str, Any]:
            return {
                "mode": mode,
                "subject": subject or "",
                "start": start or "",
                "duration_minutes": duration_minutes,
                "timezone": timezone,
            }

        def _history_metadata() -> dict[str, Any]:
            return {
                "contact_display_name": _get_contact_display_name(anchor_contact),
            }

        if not self._assistant_has_teams():
            return await self._surface_comms_error(
                "Microsoft Teams is not enabled for this assistant.",
                topic,
                contact_id=anchor_contact_id,
                medium=medium,
                attempted_content=f"create teams meeting '{subject or ''}'",
                target_metadata=_target_metadata(),
                history_metadata=_history_metadata(),
            )

        if mode not in ("instant", "scheduled"):
            return await self._surface_comms_error(
                "mode must be 'instant' or 'scheduled'",
                topic,
                contact_id=anchor_contact_id,
                medium=medium,
                attempted_content=f"create teams meeting '{subject or ''}'",
                target_metadata=_target_metadata(),
                history_metadata=_history_metadata(),
            )

        resolved_start: str | None = None
        resolved_end: str | None = None
        if mode == "scheduled":
            if not subject:
                return await self._surface_comms_error(
                    "scheduled mode requires a subject",
                    topic,
                    contact_id=anchor_contact_id,
                    medium=medium,
                    attempted_content="create teams meeting (missing subject)",
                    target_metadata=_target_metadata(),
                    history_metadata=_history_metadata(),
                )
            from datetime import datetime, timedelta, timezone as _tz

            if start:
                resolved_start = start
                try:
                    base = datetime.fromisoformat(start.replace("Z", "+00:00"))
                except ValueError:
                    return await self._surface_comms_error(
                        f"start is not a valid ISO-8601 timestamp: {start!r}",
                        topic,
                        contact_id=anchor_contact_id,
                        medium=medium,
                        attempted_content=f"create teams meeting '{subject}'",
                        target_metadata=_target_metadata(),
                        history_metadata=_history_metadata(),
                    )
            else:
                base = datetime.now(_tz.utc) + timedelta(minutes=5)
                resolved_start = base.isoformat()
            resolved_end = (base + timedelta(minutes=duration_minutes)).isoformat()

        attendee_emails: list[str] = []
        if mode == "scheduled" and attendee_contact_ids:
            try:
                parsed_attendees: list[tuple[int, str | None]] = []
                for item in attendee_contact_ids:
                    if isinstance(item, dict):
                        raw = item.get("contact_id")
                        if raw is None:
                            raise TypeError(
                                "Attendee dict must include 'contact_id', "
                                f"got: {item!r}",
                            )
                        parsed_attendees.append(
                            (_coerce_contact_id(raw), item.get("email_address")),
                        )
                    else:
                        parsed_attendees.append((_coerce_contact_id(item), None))
            except TypeError as exc:
                return await self._surface_comms_error(
                    str(exc),
                    topic,
                    contact_id=anchor_contact_id,
                    medium=medium,
                    attempted_content=f"create teams meeting '{subject}'",
                    target_metadata=_target_metadata(),
                    history_metadata=_history_metadata(),
                )

            seen_emails: set[str] = set()
            for attendee_id, inline_email in parsed_attendees:
                attendee_contact = self._get_contact(contact_id=attendee_id)
                error, resolved_attendee = self._resolve_or_attach_detail(
                    contact=attendee_contact,
                    contact_id=attendee_id,
                    field_name="email_address",
                    inline_value=inline_email,
                    medium_label="Teams",
                )
                if error:
                    return await self._surface_comms_error(
                        error,
                        topic,
                        contact_id=attendee_id,
                        medium=medium,
                        attempted_content=(f"create teams meeting '{subject}'"),
                        target_metadata=_target_metadata(),
                        history_metadata=_history_metadata(),
                    )
                email_address = (resolved_attendee or {}).get("email_address")
                if not email_address:
                    return await self._surface_comms_error(
                        f"Could not resolve email address for attendee contact_id={attendee_id}",
                        topic,
                        contact_id=attendee_id,
                        medium=medium,
                        attempted_content=(f"create teams meeting '{subject}'"),
                        target_metadata=_target_metadata(),
                        history_metadata=_history_metadata(),
                    )
                key = email_address.lower()
                if key in seen_emails:
                    continue
                seen_emails.add(key)
                attendee_emails.append(email_address)

        response = await comms_utils.create_teams_meet(
            mode=mode,
            subject=subject,
            start=resolved_start if mode == "scheduled" else start,
            end=resolved_end,
            timezone=timezone,
            attendees=attendee_emails if attendee_emails else None,
            body_html=body_html,
            location=location,
        )

        if response.get("success"):
            join_web_url = response.get("join_web_url") or ""
            meeting_id = response.get("meeting_id") or ""
            event_id = response.get("event_id") or ""
            resp_subject = response.get("subject") or (subject or "")
            resp_start = response.get("start") or (resolved_start or start or "")
            resp_end = response.get("end") or (resolved_end or "")
            web_link = response.get("web_link") or ""
            event = TeamsMeetCreated(
                contact=anchor_contact,
                mode=mode,
                subject=resp_subject,
                join_web_url=join_web_url,
                meeting_id=meeting_id,
                event_id=event_id,
                start=resp_start,
                end=resp_end,
                attendees=list(attendee_emails),
                web_link=web_link,
            )
            await self._event_broker.publish(topic, event.to_json())
            return {
                "status": "ok",
                "mode": mode,
                "join_web_url": join_web_url,
                "meeting_id": meeting_id,
                "event_id": event_id,
                "subject": resp_subject,
                "start": resp_start,
                "end": resp_end,
                "web_link": web_link,
            }

        return await self._surface_comms_error(
            response.get("error") or "Failed to create Teams meeting",
            topic,
            contact_id=anchor_contact_id,
            medium=medium,
            attempted_content=f"create teams meeting '{subject or ''}'",
            target_metadata=_target_metadata(),
            history_metadata=_history_metadata(),
        )

    async def send_unify_message(
        self,
        *,
        content: str,
        contact_id: int | str,
        attachment_filepath: str | None = None,
    ) -> dict[str, Any]:
        """Send an assistant-owned Unify inbox message to one contact.

        Use this when the assistant should message someone inside the Unify
        product rather than over an external channel like SMS or email. The
        recipient must already exist as a contact. You may include one local
        file attachment, which will be uploaded first and then referenced from
        the outbound message and transcript history.

        ``contact_id`` must be the integer id of an existing contact record.
        Shared-team routing tokens such as ``team:80`` are for memory and
        task destinations only; they are not valid here. To notify people who
        collaborate in a shared team, message each member's contact id
        individually. There is no comms primitive that posts to a team channel.

        Parameters
        ----------
        content : str
            Message body to send through the Unify platform.
        contact_id : int | str
            Integer contact id for the recipient. Not a ``team:<id>`` token.
        attachment_filepath : str | None, optional
            Workspace-local file path for one attachment to upload and include.

        Returns
        -------
        dict[str, Any]
            ``{"status": "ok"}`` on success, or an error payload describing
            why the message or attachment flow failed.
        """
        contact_id = _coerce_contact_id(contact_id)
        content = normalize_outbound_plain_text(content)
        offline_reservation, offline_response = self._reserve_offline_operation(
            method_name="send_unify_message",
            medium=Medium.UNIFY_MESSAGE,
            target_kind="contact",
            target_metadata={
                "contact_id": contact_id,
                "attachment_filepath": attachment_filepath or "",
            },
            contact_id=contact_id,
        )
        if offline_response is not None:
            return offline_response
        contact = self._get_contact(contact_id=contact_id)
        topic = "app:comms:unify_message_sent"

        if contact:
            outbound_error = self._check_outbound_allowed(contact)
            if outbound_error:
                return await self._surface_comms_error(
                    outbound_error,
                    topic,
                    contact_id=contact_id,
                    medium=Medium.UNIFY_MESSAGE,
                    offline_reservation=offline_reservation,
                    attempted_content=content,
                    receiver_ids=[contact_id],
                    target_metadata={
                        "contact_id": contact_id,
                        "attachment_filepath": attachment_filepath or "",
                    },
                    history_metadata={
                        "contact_display_name": _get_contact_display_name(contact),
                    },
                )

        attachment = None
        if attachment_filepath:
            from unify.file_manager.filesystem_adapters.local_adapter import (
                LocalFileSystemAdapter,
            )

            try:
                adapter = LocalFileSystemAdapter()
                adapter.get_file(attachment_filepath)
                abs_path = adapter._abspath(attachment_filepath)
                with open(abs_path, "rb") as file_handle:
                    file_contents = file_handle.read()

                file_size_mb = len(file_contents) / (1024 * 1024)
                if file_size_mb > 25:
                    return await self._surface_comms_error(
                        f"File too large: {file_size_mb:.1f}MB exceeds 25MB attachment limit.",
                        topic,
                        contact_id=contact_id,
                        medium=Medium.UNIFY_MESSAGE,
                        offline_reservation=offline_reservation,
                        attempted_content=content,
                        receiver_ids=[contact_id],
                        target_metadata={
                            "contact_id": contact_id,
                            "attachment_filepath": attachment_filepath or "",
                        },
                        history_metadata={
                            "contact_display_name": _get_contact_display_name(contact),
                        },
                    )

                attachment_filename = os.path.basename(attachment_filepath)
                upload_result = await comms_utils.upload_unify_attachment(
                    file_content=file_contents,
                    filename=attachment_filename,
                )
                if "error" in upload_result:
                    return await self._surface_comms_error(
                        f"Failed to upload attachment: {upload_result['error']}",
                        topic,
                        contact_id=contact_id,
                        medium=Medium.UNIFY_MESSAGE,
                        offline_reservation=offline_reservation,
                        attempted_content=content,
                        receiver_ids=[contact_id],
                        target_metadata={
                            "contact_id": contact_id,
                            "attachment_filepath": attachment_filepath or "",
                        },
                        history_metadata={
                            "contact_display_name": _get_contact_display_name(contact),
                        },
                    )

                attachment = upload_result
                attachment_id = attachment.get("id", "")
                attachment_target = f"Attachments/{attachment_id}_{attachment_filename}"
                try:
                    import shutil

                    attachment_dir = adapter._abspath("Attachments")
                    os.makedirs(attachment_dir, exist_ok=True)
                    shutil.copy2(
                        abs_path,
                        os.path.join(
                            attachment_dir,
                            f"{attachment_id}_{attachment_filename}",
                        ),
                    )
                except Exception:
                    pass
                attachment["filepath"] = attachment_target
            except FileNotFoundError:
                return await self._surface_comms_error(
                    f"File not found: {attachment_filepath}",
                    topic,
                    contact_id=contact_id,
                    medium=Medium.UNIFY_MESSAGE,
                    offline_reservation=offline_reservation,
                    attempted_content=content,
                    receiver_ids=[contact_id],
                    target_metadata={
                        "contact_id": contact_id,
                        "attachment_filepath": attachment_filepath or "",
                    },
                    history_metadata={
                        "contact_display_name": _get_contact_display_name(contact),
                    },
                )
            except Exception as exc:
                return await self._surface_comms_error(
                    f"Failed to read file: {exc}",
                    topic,
                    contact_id=contact_id,
                    medium=Medium.UNIFY_MESSAGE,
                    offline_reservation=offline_reservation,
                    attempted_content=content,
                    receiver_ids=[contact_id],
                    target_metadata={
                        "contact_id": contact_id,
                        "attachment_filepath": attachment_filepath or "",
                    },
                    history_metadata={
                        "contact_display_name": _get_contact_display_name(contact),
                    },
                )

        response = await comms_utils.send_unify_message(
            content=content,
            contact_id=contact_id,
            attachment=attachment,
        )
        if response.get("success"):
            fresh_contact = self._get_contact(contact_id=contact_id) or contact or {}
            event = UnifyMessageSent(
                contact=fresh_contact,
                content=content,
                attachments=[attachment] if attachment else [],
                **self._onboarding_event_kwargs(Medium.UNIFY_MESSAGE),
            )
            await self._event_broker.publish(topic, event.to_json())
            self._record_offline_success(
                offline_reservation,
                attempted_content=content,
                receiver_ids=[fresh_contact.get("contact_id") or contact_id],
                target_metadata={
                    "contact_id": fresh_contact.get("contact_id") or contact_id,
                    "attachment_filepath": attachment_filepath or "",
                },
                history_metadata={
                    "contact_display_name": _get_contact_display_name(fresh_contact),
                },
                attachments=[attachment] if attachment else None,
                provider_response=response,
            )
            return {"status": "ok"}

        return await self._surface_comms_error(
            "Failed to send unify message",
            topic,
            contact_id=contact_id,
            medium=Medium.UNIFY_MESSAGE,
            offline_reservation=offline_reservation,
            attempted_content=content,
            receiver_ids=[contact_id],
            target_metadata={
                "contact_id": contact_id,
                "attachment_filepath": attachment_filepath or "",
            },
            history_metadata={
                "contact_display_name": _get_contact_display_name(contact),
            },
            attachments=[attachment] if attachment else None,
        )

    async def send_api_response(
        self,
        *,
        content: str,
        contact_id: int | str = 1,
        attachment_filepaths: list[str] | None = None,
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        """Reply to the currently pending developer API message.

        This is a special assistant-owned reply path for requests that arrived
        through the developer API and are waiting for the assistant to complete
        them. It is not a general outbound messaging tool. If there is no
        pending API request in the live session, the method returns an ``ok``
        status with a note instead of creating a new outbound conversation.

        Parameters
        ----------
        content : str
            Response text to send back to the waiting API caller.
        contact_id : int | str, optional
            Contact anchor for transcript logging, defaulting to the boss
            contact.
        attachment_filepaths : list[str] | None, optional
            Workspace-local file paths to upload and attach to the API response.
        tags : list[str] | None, optional
            Optional response tags to persist alongside the completion.

        Returns
        -------
        dict[str, Any]
            ``{"status": "ok"}`` when the pending API message was completed, or
            a status payload explaining that there was no pending API message or
            why the response could not be sent.
        """
        contact_id = _coerce_contact_id(contact_id)
        api_message_id = getattr(self._cm, "_pending_api_message_id", None)
        if not api_message_id:
            return {"status": "ok", "note": "no pending api message"}

        contact = self._get_contact(contact_id=contact_id) or {"contact_id": contact_id}
        topic = "app:comms:api_message_sent"
        if tags is None:
            tags = getattr(self._cm, "_pending_api_message_tags", None) or []

        uploaded_attachments: list[dict] = []
        if attachment_filepaths:
            from unify.file_manager.filesystem_adapters.local_adapter import (
                LocalFileSystemAdapter,
            )

            for filepath in attachment_filepaths:
                try:
                    adapter = LocalFileSystemAdapter()
                    adapter.get_file(filepath)
                    abs_path = adapter._abspath(filepath)
                    with open(abs_path, "rb") as file_handle:
                        file_contents = file_handle.read()

                    upload_result = await comms_utils.upload_unify_attachment(
                        file_content=file_contents,
                        filename=os.path.basename(filepath),
                    )
                    if "error" in upload_result:
                        return await self._surface_comms_error(
                            f"Failed to upload attachment: {upload_result['error']}",
                            topic,
                            contact_id=contact_id,
                            medium=Medium.API_MESSAGE,
                        )

                    attachment_id = upload_result.get("id", "")
                    attachment_filename = os.path.basename(filepath)
                    attachment_target = (
                        f"Attachments/{attachment_id}_{attachment_filename}"
                    )
                    try:
                        import shutil

                        attachment_dir = adapter._abspath("Attachments")
                        os.makedirs(attachment_dir, exist_ok=True)
                        shutil.copy2(
                            abs_path,
                            os.path.join(
                                attachment_dir,
                                f"{attachment_id}_{attachment_filename}",
                            ),
                        )
                    except Exception:
                        pass
                    upload_result["filepath"] = attachment_target
                    uploaded_attachments.append(upload_result)
                except FileNotFoundError:
                    return await self._surface_comms_error(
                        f"File not found: {filepath}",
                        topic,
                        contact_id=contact_id,
                        medium=Medium.API_MESSAGE,
                    )
                except Exception as exc:
                    return await self._surface_comms_error(
                        f"Failed to read file: {exc}",
                        topic,
                        contact_id=contact_id,
                        medium=Medium.API_MESSAGE,
                    )

        result = await comms_utils.complete_api_message(
            api_message_id=api_message_id,
            response=content,
            attachments=uploaded_attachments or None,
            tags=tags or None,
        )
        if result.get("success"):
            event = ApiMessageSent(
                contact=contact,
                content=content,
                api_message_id=api_message_id,
                attachments=uploaded_attachments,
                tags=tags,
            )
            await self._event_broker.publish(topic, event.to_json())
            if self._cm is not None:
                self._cm._pending_api_message_id = None
                self._cm._pending_api_message_tags = None
            return {"status": "ok"}

        return await self._surface_comms_error(
            "Failed to send API response",
            topic,
            contact_id=contact_id,
            medium=Medium.API_MESSAGE,
        )

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
        """Send an assistant-owned email to explicit recipients or a reply thread.

        Each recipient in ``to``, ``cc``, and ``bcc`` can be provided in one of
        two ways:

        - Bare ``contact_id`` when the contact already has an email address on
          file.
        - ``{"contact_id": ..., "email_address": ...}`` when the contact is
          missing an email address but you know it and want to attach it during
          the send.

        Mixed recipient forms are allowed in the same call, and duplicate email
        addresses are collapsed automatically. Do not use the dict form to
        provide an email address that conflicts with the one already stored on
        the contact; update the contact first, then retry.

        ``reply_all=True`` is mutually exclusive with explicit ``to`` / ``cc`` /
        ``bcc`` lists. When reply-all is used, recipients are inferred from the
        referenced email thread instead.

        Parameters
        ----------
        to : list[int | dict] | None, optional
            Primary recipients.
        cc : list[int | dict] | None, optional
            Carbon-copy recipients.
        bcc : list[int | dict] | None, optional
            Blind-carbon-copy recipients.
        subject : str
            Subject line for the outgoing email.
        body : str
            Email body content to send.
        reply_all : bool, optional
            Reply to the referenced thread's participants instead of providing
            explicit recipients.
        email_id_to_reply_to : str | None, optional
            Explicit email ID to reply to for threading. Required for
            deterministic reply behavior when multiple candidate emails exist.
        thread_id : str | None, optional
            Provider thread identifier to target (for example Gmail threadId).
        attachment_filepath : str | None, optional
            Workspace-relative file path for one attachment to include.

        Returns
        -------
        dict[str, Any]
            ``{"status": "ok"}`` on success, or an error payload describing
            why the email could not be sent.
        """
        body = normalize_outbound_plain_text(body)
        from unify.file_manager.filesystem_adapters.local_adapter import (
            LocalFileSystemAdapter,
        )

        def _raw_recipients_for_history(items: list[int | dict] | None) -> list[dict]:
            serialized: list[dict] = []
            for item in items or []:
                if isinstance(item, dict):
                    serialized.append(
                        {
                            "contact_id": item.get("contact_id"),
                            "email_address": item.get("email_address") or "",
                        },
                    )
                elif isinstance(item, tuple) and len(item) == 2:
                    serialized.append(
                        {
                            "contact_id": item[0],
                            "email_address": item[1] or "",
                        },
                    )
                else:
                    serialized.append({"contact_id": item, "email_address": ""})
            return serialized

        def _first_contact_id_from_raw(items: list[int | dict] | None) -> int | None:
            for item in items or []:
                try:
                    if isinstance(item, dict):
                        raw_contact_id = item.get("contact_id")
                        if raw_contact_id is None:
                            continue
                        return _coerce_contact_id(raw_contact_id)
                    return _coerce_contact_id(item)
                except (TypeError, ValueError):
                    continue
            return None

        def _coerce_recipients(
            items: list | None,
        ) -> list[tuple[int, str | None]] | None:
            if items is None:
                return None
            result: list[tuple[int, str | None]] = []
            for item in items:
                if isinstance(item, dict):
                    contact_id = item.get("contact_id")
                    if contact_id is None:
                        raise TypeError(
                            f"Email recipient dict must include 'contact_id', got: {item!r}",
                        )
                    result.append(
                        (_coerce_contact_id(contact_id), item.get("email_address")),
                    )
                else:
                    result.append((_coerce_contact_id(item), None))
            return result

        topic = "app:comms:email_sent"
        offline_reservation = None
        initial_error_contact_id = (
            _first_contact_id_from_raw(to)
            or _first_contact_id_from_raw(cc)
            or _first_contact_id_from_raw(bcc)
        )
        try:
            to = _coerce_recipients(to)
            cc = _coerce_recipients(cc)
            bcc = _coerce_recipients(bcc)
        except TypeError as exc:
            return await self._surface_comms_error(
                str(exc),
                topic,
                contact_id=initial_error_contact_id,
                medium=Medium.EMAIL,
                offline_reservation=offline_reservation,
                attempted_content=f"Subject: {subject}\n\n{body}",
                receiver_ids=(
                    [initial_error_contact_id]
                    if initial_error_contact_id is not None
                    else None
                ),
                target_metadata={
                    "to": _raw_recipients_for_history(to),
                    "cc": _raw_recipients_for_history(cc),
                    "bcc": _raw_recipients_for_history(bcc),
                    "reply_all": reply_all,
                    "email_id_to_reply_to": email_id_to_reply_to or "",
                    "thread_id": thread_id or "",
                    "attachment_filepath": attachment_filepath or "",
                },
            )
        error_contact_id = (
            (to[0][0] if to else None)
            or (cc[0][0] if cc else None)
            or (bcc[0][0] if bcc else None)
        )

        if reply_all and (to or cc or bcc):
            return await self._surface_comms_error(
                "reply_all=True is mutually exclusive with to/cc/bcc. "
                "Either use reply_all to auto-populate recipients from the thread, "
                "or specify recipients explicitly.",
                topic,
                contact_id=error_contact_id,
                medium=Medium.EMAIL,
                offline_reservation=offline_reservation,
                attempted_content=f"Subject: {subject}\n\n{body}",
                receiver_ids=(
                    [error_contact_id] if error_contact_id is not None else None
                ),
                target_metadata={
                    "to": _raw_recipients_for_history(to),
                    "cc": _raw_recipients_for_history(cc),
                    "bcc": _raw_recipients_for_history(bcc),
                    "reply_all": reply_all,
                    "email_id_to_reply_to": email_id_to_reply_to or "",
                    "thread_id": thread_id or "",
                    "attachment_filepath": attachment_filepath or "",
                },
            )

        if not self._assistant_email():
            return await self._surface_comms_error(
                "You don't have an email address, please provision one.",
                topic,
                contact_id=error_contact_id,
                medium=Medium.EMAIL,
                attempted_content=f"Subject: {subject}\n\n{body}",
                receiver_ids=(
                    [error_contact_id] if error_contact_id is not None else None
                ),
                target_metadata={
                    "to": _raw_recipients_for_history(to),
                    "cc": _raw_recipients_for_history(cc),
                    "bcc": _raw_recipients_for_history(bcc),
                    "reply_all": reply_all,
                    "email_id_to_reply_to": email_id_to_reply_to or "",
                    "thread_id": thread_id or "",
                    "attachment_filepath": attachment_filepath or "",
                },
            )

        def _resolve_recipients(
            recipients: list[tuple[int, str | None]] | None,
        ) -> tuple[str | None, list[tuple[str, dict]]]:
            if not recipients:
                return (None, [])
            resolved_by_email: dict[str, dict] = {}
            for contact_id, inline_email in recipients:
                contact = self._get_contact(contact_id=contact_id)
                error, resolved_contact = self._resolve_or_attach_detail(
                    contact=contact,
                    contact_id=contact_id,
                    field_name="email_address",
                    inline_value=inline_email,
                    medium_label="email",
                )
                if error:
                    return (error, [])
                email_address = (resolved_contact or {}).get("email_address")
                if email_address and email_address not in resolved_by_email:
                    resolved_by_email[email_address] = resolved_contact or {}
            return (
                None,
                [(email, contact) for email, contact in resolved_by_email.items()],
            )

        final_to: list[str] = []
        final_cc: list[str] = []
        final_bcc: list[str] = []
        reply_email_id = email_id_to_reply_to
        primary_contact: dict | None = None
        to_resolved: list[tuple[str, dict]] = []
        cc_resolved: list[tuple[str, dict]] = []
        bcc_resolved: list[tuple[str, dict]] = []

        if reply_all:
            if self._cm is None:
                return await self._surface_comms_error(
                    "reply_all=True requires a live ConversationManager email thread.",
                    topic,
                    contact_id=error_contact_id,
                    medium=Medium.EMAIL,
                    offline_reservation=offline_reservation,
                    attempted_content=f"Subject: {subject}\n\n{body}",
                    receiver_ids=(
                        [error_contact_id] if error_contact_id is not None else None
                    ),
                    target_metadata={
                        "reply_all": True,
                        "email_id_to_reply_to": reply_email_id or "",
                        "thread_id": thread_id or "",
                        "attachment_filepath": attachment_filepath or "",
                    },
                )

            original_email = None
            all_emails = [
                entry.message
                for entry in self._cm.contact_index.global_thread
                if entry.medium == Medium.EMAIL
            ]
            if reply_email_id:
                for message in all_emails:
                    if getattr(message, "email_id", None) == reply_email_id:
                        original_email = message
                        break
            else:
                for message in reversed(all_emails):
                    if getattr(message, "name", None) == "You":
                        continue
                    if not getattr(message, "email_id", None):
                        continue
                    clean_subject = subject.removeprefix("Re: ").strip()
                    clean_message_subject = (
                        (getattr(message, "subject", "") or "")
                        .removeprefix(
                            "Re: ",
                        )
                        .strip()
                    )
                    if clean_subject == clean_message_subject or not clean_subject:
                        original_email = message
                        reply_email_id = message.email_id
                        break

            if original_email is None:
                return await self._surface_comms_error(
                    "reply_all=True but no email found to reply to. "
                    "Either provide email_id_to_reply_to or ensure there's a matching inbound email in the thread.",
                    topic,
                    contact_id=error_contact_id,
                    medium=Medium.EMAIL,
                    offline_reservation=offline_reservation,
                    attempted_content=f"Subject: {subject}\n\n{body}",
                    receiver_ids=(
                        [error_contact_id] if error_contact_id is not None else None
                    ),
                    target_metadata={
                        "reply_all": True,
                        "email_id_to_reply_to": reply_email_id or "",
                        "thread_id": thread_id or "",
                        "attachment_filepath": attachment_filepath or "",
                    },
                )

            assistant_email = self._assistant_email()
            original_to = getattr(original_email, "to", []) or []
            original_cc = getattr(original_email, "cc", []) or []

            sender_email = None
            for entry in self._cm.contact_index.global_thread:
                if entry.message is not original_email:
                    continue
                for role_contact_id, role in entry.contact_roles.items():
                    if role != "sender":
                        continue
                    sender_contact = self._get_contact(contact_id=role_contact_id)
                    if sender_contact:
                        sender_email = sender_contact.get("email_address")
                        primary_contact = sender_contact
                    break
                break

            if sender_email:
                final_to = [sender_email]

            all_original_recipients = set(original_to) | set(original_cc)
            if assistant_email:
                all_original_recipients.discard(assistant_email)
            if sender_email:
                all_original_recipients.discard(sender_email)
            final_cc = list(all_original_recipients)
        else:
            to_error, to_resolved = _resolve_recipients(to)
            if to_error:
                return await self._surface_comms_error(
                    to_error,
                    topic,
                    contact_id=error_contact_id,
                    medium=Medium.EMAIL,
                    offline_reservation=offline_reservation,
                    attempted_content=f"Subject: {subject}\n\n{body}",
                    receiver_ids=(
                        [error_contact_id] if error_contact_id is not None else None
                    ),
                    target_metadata={
                        "to": _raw_recipients_for_history(to),
                        "cc": _raw_recipients_for_history(cc),
                        "bcc": _raw_recipients_for_history(bcc),
                        "reply_all": False,
                        "email_id_to_reply_to": reply_email_id or "",
                        "thread_id": thread_id or "",
                        "attachment_filepath": attachment_filepath or "",
                    },
                )

            cc_error, cc_resolved = _resolve_recipients(cc)
            if cc_error:
                return await self._surface_comms_error(
                    cc_error,
                    topic,
                    contact_id=error_contact_id,
                    medium=Medium.EMAIL,
                    offline_reservation=offline_reservation,
                    attempted_content=f"Subject: {subject}\n\n{body}",
                    receiver_ids=(
                        [error_contact_id] if error_contact_id is not None else None
                    ),
                    target_metadata={
                        "to": _raw_recipients_for_history(to),
                        "cc": _raw_recipients_for_history(cc),
                        "bcc": _raw_recipients_for_history(bcc),
                        "reply_all": False,
                        "email_id_to_reply_to": reply_email_id or "",
                        "thread_id": thread_id or "",
                        "attachment_filepath": attachment_filepath or "",
                    },
                )

            bcc_error, bcc_resolved = _resolve_recipients(bcc)
            if bcc_error:
                return await self._surface_comms_error(
                    bcc_error,
                    topic,
                    contact_id=error_contact_id,
                    medium=Medium.EMAIL,
                    offline_reservation=offline_reservation,
                    attempted_content=f"Subject: {subject}\n\n{body}",
                    receiver_ids=(
                        [error_contact_id] if error_contact_id is not None else None
                    ),
                    target_metadata={
                        "to": _raw_recipients_for_history(to),
                        "cc": _raw_recipients_for_history(cc),
                        "bcc": _raw_recipients_for_history(bcc),
                        "reply_all": False,
                        "email_id_to_reply_to": reply_email_id or "",
                        "thread_id": thread_id or "",
                        "attachment_filepath": attachment_filepath or "",
                    },
                )

            final_to = [email for email, _contact in to_resolved]
            final_cc = [email for email, _contact in cc_resolved]
            final_bcc = [email for email, _contact in bcc_resolved]

            if to_resolved:
                primary_contact = to_resolved[0][1]
            elif cc_resolved:
                primary_contact = cc_resolved[0][1]
            elif bcc_resolved:
                primary_contact = bcc_resolved[0][1]

            if not final_to and not final_cc and not final_bcc:
                return await self._surface_comms_error(
                    "At least one recipient is required. Provide to, cc, or bcc, or use reply_all=True.",
                    topic,
                    contact_id=error_contact_id,
                    medium=Medium.EMAIL,
                    offline_reservation=offline_reservation,
                    attempted_content=f"Subject: {subject}\n\n{body}",
                    receiver_ids=(
                        [error_contact_id] if error_contact_id is not None else None
                    ),
                    target_metadata={
                        "to": _raw_recipients_for_history(to),
                        "cc": _raw_recipients_for_history(cc),
                        "bcc": _raw_recipients_for_history(bcc),
                        "reply_all": False,
                        "email_id_to_reply_to": reply_email_id or "",
                        "thread_id": thread_id or "",
                        "attachment_filepath": attachment_filepath or "",
                    },
                )

            if not reply_email_id and self._cm is not None:
                try:
                    all_emails = [
                        entry.message
                        for entry in self._cm.contact_index.global_thread
                        if entry.medium == Medium.EMAIL
                    ]
                    for message in reversed(all_emails):
                        if (
                            getattr(message, "name", None) != "You"
                            and getattr(
                                message,
                                "subject",
                                None,
                            )
                            == subject
                            and getattr(message, "email_id", None)
                        ):
                            reply_email_id = message.email_id
                            break
                except Exception:
                    pass

        if reply_email_id and not thread_id and self._cm is not None:
            for entry in self._cm.contact_index.global_thread:
                if entry.medium != Medium.EMAIL:
                    continue
                message = entry.message
                if getattr(message, "email_id", None) != reply_email_id:
                    continue
                thread_id = getattr(message, "thread_id", None) or None
                break

        final_subject = (
            f"Re: {subject}"
            if (reply_email_id or thread_id) and not subject.startswith("Re: ")
            else subject
        )
        offline_reservation, offline_response = self._reserve_offline_operation(
            method_name="send_email",
            medium=Medium.EMAIL,
            target_kind="email",
            target_metadata={
                "to": final_to,
                "cc": final_cc,
                "bcc": final_bcc,
                "reply_all": reply_all,
                "email_id_to_reply_to": reply_email_id or "",
                "thread_id": thread_id or "",
                "attachment_filepath": attachment_filepath or "",
            },
            contact_id=(primary_contact or {}).get("contact_id") or error_contact_id,
        )
        if offline_response is not None:
            return offline_response

        attachment = None
        attachment_meta = None
        if attachment_filepath:
            try:
                adapter = LocalFileSystemAdapter()
                adapter.get_file(attachment_filepath)
                abs_path = adapter._abspath(attachment_filepath)
                with open(abs_path, "rb") as file_handle:
                    file_contents = file_handle.read()

                file_size_mb = len(file_contents) / (1024 * 1024)
                if file_size_mb > 25:
                    return await self._surface_comms_error(
                        f"File too large for email: {file_size_mb:.1f}MB exceeds Gmail's 25MB attachment limit. "
                        "Consider sharing via Unify message instead.",
                        topic,
                        contact_id=error_contact_id,
                        medium=Medium.EMAIL,
                        offline_reservation=offline_reservation,
                        attempted_content=f"Subject: {final_subject}\n\n{body}",
                        receiver_ids=(
                            [error_contact_id] if error_contact_id is not None else None
                        ),
                        target_metadata={
                            "to": final_to,
                            "cc": final_cc,
                            "bcc": final_bcc,
                            "reply_all": reply_all,
                            "email_id_to_reply_to": reply_email_id or "",
                            "thread_id": thread_id or "",
                            "attachment_filepath": attachment_filepath or "",
                        },
                    )

                attachment_filename = os.path.basename(attachment_filepath)
                attachment_id = str(uuid.uuid4())
                attachment = {
                    "filename": attachment_filename,
                    "content_base64": base64.b64encode(file_contents).decode("utf-8"),
                }
                attachment_target = f"Attachments/{attachment_id}_{attachment_filename}"
                try:
                    import shutil

                    attachment_dir = adapter._abspath("Attachments")
                    os.makedirs(attachment_dir, exist_ok=True)
                    shutil.copy2(
                        abs_path,
                        os.path.join(
                            attachment_dir,
                            f"{attachment_id}_{attachment_filename}",
                        ),
                    )
                except Exception:
                    pass
                attachment_meta = {
                    "id": attachment_id,
                    "filename": attachment_filename,
                    "filepath": attachment_target,
                }
            except FileNotFoundError:
                return await self._surface_comms_error(
                    f"File not found: {attachment_filepath}",
                    topic,
                    contact_id=error_contact_id,
                    medium=Medium.EMAIL,
                    offline_reservation=offline_reservation,
                    attempted_content=f"Subject: {final_subject}\n\n{body}",
                    receiver_ids=(
                        [error_contact_id] if error_contact_id is not None else None
                    ),
                    target_metadata={
                        "to": final_to,
                        "cc": final_cc,
                        "bcc": final_bcc,
                        "reply_all": reply_all,
                        "email_id_to_reply_to": reply_email_id or "",
                        "thread_id": thread_id or "",
                        "attachment_filepath": attachment_filepath or "",
                    },
                )
            except Exception as exc:
                return await self._surface_comms_error(
                    f"Failed to read file: {exc}",
                    topic,
                    contact_id=error_contact_id,
                    medium=Medium.EMAIL,
                    offline_reservation=offline_reservation,
                    attempted_content=f"Subject: {final_subject}\n\n{body}",
                    receiver_ids=(
                        [error_contact_id] if error_contact_id is not None else None
                    ),
                    target_metadata={
                        "to": final_to,
                        "cc": final_cc,
                        "bcc": final_bcc,
                        "reply_all": reply_all,
                        "email_id_to_reply_to": reply_email_id or "",
                        "thread_id": thread_id or "",
                        "attachment_filepath": attachment_filepath or "",
                    },
                )

        history_receiver_ids = [
            contact.get("contact_id")
            for _email, contact in (to_resolved + cc_resolved + bcc_resolved)
            if contact.get("contact_id") is not None
        ]
        if not history_receiver_ids and primary_contact is not None:
            primary_contact_id = primary_contact.get("contact_id")
            if primary_contact_id is not None:
                history_receiver_ids = [primary_contact_id]

        response = await comms_utils.send_email_via_address(
            to=final_to,
            subject=final_subject,
            body=body,
            cc=final_cc or None,
            bcc=final_bcc or None,
            email_id=reply_email_id,
            thread_id=thread_id,
            attachment=attachment,
        )
        if response.get("success"):
            event = EmailSent(
                contact=primary_contact or {},
                body=body,
                subject=final_subject,
                email_id_replied_to=reply_email_id,
                thread_id=thread_id,
                attachments=[attachment_meta] if attachment_meta else [],
                to=final_to,
                cc=final_cc,
                bcc=final_bcc,
                **self._onboarding_event_kwargs(Medium.EMAIL),
            )
            await self._event_broker.publish(topic, event.to_json())
            self._record_offline_success(
                offline_reservation,
                attempted_content=f"Subject: {final_subject}\n\n{body}",
                receiver_ids=history_receiver_ids
                or ([error_contact_id] if error_contact_id is not None else None),
                target_metadata={
                    "to": final_to,
                    "cc": final_cc,
                    "bcc": final_bcc,
                    "reply_all": reply_all,
                    "email_id_to_reply_to": reply_email_id or "",
                    "thread_id": thread_id or "",
                    "attachment_filepath": attachment_filepath or "",
                },
                history_metadata={
                    "contact_display_name": _get_contact_display_name(primary_contact),
                    "reply_to_email_id": reply_email_id or None,
                    "thread_id": thread_id or None,
                },
                attachments=[attachment_meta] if attachment_meta else None,
                provider_response=response,
            )
            return {"status": "ok"}

        if not self._assistant_email():
            error_msg = "You don't have an email address, please provision one."
        else:
            recipients = final_to + final_cc + final_bcc
            error_msg = response.get("error", f"Failed to send email to {recipients}")
        return await self._surface_comms_error(
            error_msg,
            topic,
            contact_id=error_contact_id,
            medium=Medium.EMAIL,
            offline_reservation=offline_reservation,
            attempted_content=f"Subject: {final_subject}\n\n{body}",
            receiver_ids=history_receiver_ids
            or ([error_contact_id] if error_contact_id is not None else None),
            target_metadata={
                "to": final_to,
                "cc": final_cc,
                "bcc": final_bcc,
                "reply_all": reply_all,
                "email_id_to_reply_to": reply_email_id or "",
                "thread_id": thread_id or "",
                "attachment_filepath": attachment_filepath or "",
            },
            history_metadata={
                "contact_display_name": _get_contact_display_name(primary_contact),
                "reply_to_email_id": reply_email_id or None,
                "thread_id": thread_id or None,
            },
            attachments=[attachment_meta] if attachment_meta else None,
        )

    async def make_call(
        self,
        *,
        contact_id: int | str,
        context: str,
        phone_number: str | None = None,
    ) -> dict[str, Any]:
        """Start an assistant-owned outbound phone call to an existing contact.

        The contact must already exist in the system.

        - If the contact already has a phone number on file, omit
          ``phone_number``.
        - If the contact is missing a phone number but you know it, pass it via
          ``phone_number`` and the contact record will be updated before the
          call is placed.
        - Do not supply a different phone number from the one already on file.
          Update the contact first, then retry.

        Parameters
        ----------
        contact_id : int | str
            Person to call.
        context : str
            Mission briefing for the voice agent. This is the agent's main
            context for opening and handling the call, so include the purpose,
            key facts, questions to ask, tone, relationship context, and any
            constraints or fallback behavior.
        phone_number : str | None, optional
            Recipient phone number when the contact does not already have one on
            file.

        Returns
        -------
        dict[str, Any]
            ``{"status": "ok"}`` on success, or an error payload describing
            why the call could not be started.
        """
        contact_id = _coerce_contact_id(contact_id)
        offline_reservation = None

        if self._cm is not None and not await self._await_ready_for_outbound_call():
            return {
                "status": "retry_later_active_voice_session",
                "message": (
                    "The voice line isn't ready for a new call yet — a previous "
                    "session is still closing or the line is re-warming. Retry "
                    "shortly."
                ),
            }

        contact = self._get_contact(contact_id=contact_id)
        outbound_error = self._check_outbound_allowed(contact)
        if outbound_error:
            return await self._surface_comms_error(
                outbound_error,
                "app:comms:make_call",
                contact_id=contact_id,
                medium=Medium.PHONE_CALL,
                offline_reservation=offline_reservation,
                attempted_content="<Sending Call...>",
                receiver_ids=[contact_id],
                target_metadata={
                    "contact_id": contact_id,
                    "phone_number": (contact or {}).get("phone_number")
                    or phone_number
                    or "",
                },
                history_metadata={
                    "contact_display_name": _get_contact_display_name(contact),
                },
            )

        if not self._assistant_number():
            return await self._surface_comms_error(
                "You don't have a number, please provision one.",
                "app:comms:make_call",
                contact_id=contact_id,
                medium=Medium.PHONE_CALL,
                attempted_content="<Sending Call...>",
                receiver_ids=[contact_id],
                target_metadata={
                    "contact_id": contact_id,
                    "phone_number": (contact or {}).get("phone_number")
                    or phone_number
                    or "",
                },
                history_metadata={
                    "contact_display_name": _get_contact_display_name(contact),
                },
            )

        detail_error, contact = self._resolve_or_attach_detail(
            contact=contact,
            contact_id=contact_id,
            field_name="phone_number",
            inline_value=phone_number,
            medium_label="phone call",
        )
        if detail_error:
            return await self._surface_comms_error(
                detail_error,
                "app:comms:make_call",
                contact_id=contact_id,
                medium=Medium.PHONE_CALL,
                offline_reservation=offline_reservation,
                attempted_content="<Sending Call...>",
                receiver_ids=[contact_id],
                target_metadata={
                    "contact_id": contact_id,
                    "phone_number": (contact or {}).get("phone_number")
                    or phone_number
                    or "",
                },
                history_metadata={
                    "contact_display_name": _get_contact_display_name(contact),
                },
            )

        offline_reservation, offline_response = self._reserve_offline_operation(
            method_name="make_call",
            medium=Medium.PHONE_CALL,
            target_kind="contact",
            target_metadata={
                "contact_id": (contact or {}).get("contact_id") or contact_id,
                "phone_number": (contact or {}).get("phone_number")
                or phone_number
                or "",
            },
            contact_id=(contact or {}).get("contact_id") or contact_id,
        )
        if offline_response is not None:
            return offline_response

        to_number = (contact or {}).get("phone_number")
        LOGGER.debug(
            "%s [make_call] context=%r to_number_present=%s",
            DEFAULT_ICON,
            context,
            bool(to_number),
        )
        if self._cm is not None and context:
            self._cm.call_manager.initial_notification = context

        response = await comms_utils.start_call(to_number=to_number)
        if response.get("success"):
            fresh_contact = self._get_contact(phone_number=to_number) or contact or {}
            event = PhoneCallSent(
                contact=fresh_contact,
                provider_call_sid=response.get("call_sid", ""),
                **self._onboarding_event_kwargs(Medium.PHONE_CALL),
            )
            await self._event_broker.publish("app:comms:make_call", event.to_json())
            self._record_offline_success(
                offline_reservation,
                attempted_content="<Sending Call...>",
                receiver_ids=[fresh_contact.get("contact_id") or contact_id],
                target_metadata={
                    "contact_id": fresh_contact.get("contact_id") or contact_id,
                    "phone_number": to_number or "",
                },
                history_metadata={
                    "contact_display_name": _get_contact_display_name(fresh_contact),
                },
                provider_response=response,
            )
            return {"status": "ok"}

        if not self._assistant_number():
            error_msg = "You don't have a number, please provision one."
        else:
            error_msg = f"Failed to send call to {to_number}"
        return await self._surface_comms_error(
            error_msg,
            "app:comms:make_call",
            contact_id=contact_id,
            medium=Medium.PHONE_CALL,
            offline_reservation=offline_reservation,
            attempted_content="<Sending Call...>",
            receiver_ids=[contact_id],
            target_metadata={
                "contact_id": contact_id,
                "phone_number": to_number or phone_number or "",
            },
            history_metadata={
                "contact_display_name": _get_contact_display_name(contact),
            },
        )

    async def make_whatsapp_call(
        self,
        *,
        contact_id: int | str,
        context: str,
        whatsapp_number: str | None = None,
    ) -> dict[str, Any]:
        """Start an assistant-owned outbound WhatsApp voice call.

        The contact must already exist in the system.

        - If the contact already has a WhatsApp number on file, omit
          ``whatsapp_number``.
        - If the contact is missing a WhatsApp number but you know it, pass it
          via ``whatsapp_number`` and the contact record will be updated before
          the call is placed.
        - Do not supply a different WhatsApp number from the one already on
          file. Update the contact first, then retry.

        WhatsApp Business Calling may require the contact to approve calls from
        this business first. If permission has not yet been granted, this sends
        a WhatsApp call-permission request instead of placing the call. In a live
        assistant session, the call context is kept ready so the call is placed
        automatically after the contact taps "Allow calls" / "Call now".
        Headless offline task runs cannot queue that follow-up context, so they
        only report that an invite was sent.

        Parameters
        ----------
        contact_id : int | str
            Person to call.
        context : str
            Mission briefing for the voice agent. This is the agent's main
            context for opening and handling the call, so include the purpose,
            key facts, questions to ask, tone, relationship context, and any
            constraints or fallback behavior.
        whatsapp_number : str | None, optional
            Recipient WhatsApp number when the contact does not already have one
            on file.

        Returns
        -------
        dict[str, Any]
            Status payload with success, error, or live callback-follow-up
            details.
        """
        from unify.conversation_manager.domains.call_manager import make_room_name

        contact_id = _coerce_contact_id(contact_id)
        offline_reservation = None
        if not await self._await_ready_for_outbound_call():
            return {
                "status": "retry_later_active_voice_session",
                "message": (
                    "The voice line isn't ready for a new call yet — a previous "
                    "session is still closing or the line is re-warming. Retry "
                    "shortly after it clears."
                ),
            }

        contact = self._get_contact(contact_id=contact_id)
        outbound_error = self._check_outbound_allowed(contact)
        if outbound_error:
            return await self._surface_comms_error(
                outbound_error,
                "app:comms:whatsapp_call_sent",
                contact_id=contact_id,
                medium=Medium.WHATSAPP_CALL,
                offline_reservation=offline_reservation,
                attempted_content="<Sending WhatsApp Call...>",
                receiver_ids=[contact_id],
                target_metadata={
                    "contact_id": contact_id,
                    "whatsapp_number": (contact or {}).get("whatsapp_number")
                    or whatsapp_number
                    or "",
                },
                history_metadata={
                    "contact_display_name": _get_contact_display_name(contact),
                },
            )

        if not self._assistant_whatsapp_number():
            return await self._surface_comms_error(
                "You don't have a WhatsApp number configured.",
                "app:comms:whatsapp_call_sent",
                contact_id=contact_id,
                medium=Medium.WHATSAPP_CALL,
                attempted_content="<Sending WhatsApp Call...>",
                receiver_ids=[contact_id],
                target_metadata={
                    "contact_id": contact_id,
                    "whatsapp_number": (contact or {}).get("whatsapp_number")
                    or whatsapp_number
                    or "",
                },
                history_metadata={
                    "contact_display_name": _get_contact_display_name(contact),
                },
            )

        detail_error, contact = self._resolve_or_attach_detail(
            contact=contact,
            contact_id=contact_id,
            field_name="whatsapp_number",
            inline_value=whatsapp_number,
            medium_label="WhatsApp call",
        )
        if detail_error:
            return await self._surface_comms_error(
                detail_error,
                "app:comms:whatsapp_call_sent",
                contact_id=contact_id,
                medium=Medium.WHATSAPP_CALL,
                offline_reservation=offline_reservation,
                attempted_content="<Sending WhatsApp Call...>",
                receiver_ids=[contact_id],
                target_metadata={
                    "contact_id": contact_id,
                    "whatsapp_number": (contact or {}).get("whatsapp_number")
                    or whatsapp_number
                    or "",
                },
                history_metadata={
                    "contact_display_name": _get_contact_display_name(contact),
                },
            )

        offline_reservation, offline_response = self._reserve_offline_operation(
            method_name="make_whatsapp_call",
            medium=Medium.WHATSAPP_CALL,
            target_kind="contact",
            target_metadata={
                "contact_id": (contact or {}).get("contact_id") or contact_id,
                "whatsapp_number": (contact or {}).get("whatsapp_number")
                or whatsapp_number
                or "",
            },
            contact_id=(contact or {}).get("contact_id") or contact_id,
        )
        if offline_response is not None:
            return offline_response

        to_number = (contact or {}).get("whatsapp_number")
        assistant_id = str(SESSION_DETAILS.assistant.agent_id)
        room_name = make_room_name(assistant_id, "whatsapp_call")
        LOGGER.debug(
            f"{DEFAULT_ICON} [make_whatsapp_call] context: {context}, to_number: {to_number}",
        )

        if self._cm is not None:
            self._cm.call_manager._whatsapp_call_joining = True

        response = await comms_utils.start_whatsapp_call(
            to_number=to_number,
            agent_name=SESSION_DETAILS.assistant.name or "",
            room_name=room_name,
            allow_permission_probe=self._allow_whatsapp_call_permission_probe(),
            pending_call_context=context,
        )
        if not response.get("success"):
            if self._cm is not None:
                self._cm.call_manager._whatsapp_call_joining = False
            if not self._assistant_whatsapp_number():
                error_msg = "You don't have a WhatsApp number configured."
            else:
                error_msg = f"Failed to initiate WhatsApp call to {to_number}"
            return await self._surface_comms_error(
                error_msg,
                "app:comms:whatsapp_call_sent",
                contact_id=contact_id,
                medium=Medium.WHATSAPP_CALL,
                offline_reservation=offline_reservation,
                attempted_content="<Sending WhatsApp Call...>",
                receiver_ids=[contact_id],
                target_metadata={
                    "contact_id": contact_id,
                    "whatsapp_number": to_number or whatsapp_number or "",
                },
                history_metadata={
                    "contact_display_name": _get_contact_display_name(contact),
                },
            )

        fresh_contact = self._get_contact(whatsapp_number=to_number) or contact or {}
        method = response.get("method")
        if method == "direct":
            if self._cm is not None and context:
                self._cm.call_manager.initial_notification = context
            event = WhatsAppCallSent(
                contact=fresh_contact,
                **self._onboarding_event_kwargs(Medium.WHATSAPP_CALL),
            )
            await self._event_broker.publish(
                "app:comms:whatsapp_call_sent",
                event.to_json(),
            )
            self._record_offline_success(
                offline_reservation,
                attempted_content="<Sending WhatsApp Call...>",
                receiver_ids=[fresh_contact.get("contact_id") or contact_id],
                target_metadata={
                    "contact_id": fresh_contact.get("contact_id") or contact_id,
                    "whatsapp_number": to_number or "",
                },
                history_metadata={
                    "contact_display_name": _get_contact_display_name(fresh_contact),
                },
                provider_response=response,
            )
            return {"status": "ok"}

        pending_contexts = None
        automatic_callback_available = False
        if self._cm is not None:
            self._cm.call_manager._whatsapp_call_joining = False
            pending_contexts = getattr(
                self._cm,
                "_pending_whatsapp_call_contexts",
                None,
            )
            if context:
                if isinstance(pending_contexts, dict):
                    pending_contexts[contact_id] = context
                    automatic_callback_available = True
            else:
                automatic_callback_available = True

        pool_number = response.get("pool_number") or self._assistant_whatsapp_number()
        if (
            context
            and pool_number
            and method in {"invite", "invite_pending", "needs_reconciliation"}
        ):
            try:
                await comms_utils.store_pending_whatsapp_call_intent(
                    pool_number=pool_number,
                    contact_number=to_number,
                    context=context,
                )
            except Exception as exc:
                LOGGER.error(
                    f"{DEFAULT_ICON} Failed to persist pending WhatsApp call intent: {exc}",
                )

        if method == "invite_pending":
            return {
                "status": "ok",
                "pending_callback": automatic_callback_available,
                "note": (
                    "A WhatsApp call-permission request is already pending. "
                    "Do not send another request yet; wait for the contact to allow calls."
                ),
            }

        if method == "rejected":
            if isinstance(pending_contexts, dict):
                pending_contexts.pop(contact_id, None)
            if pool_number:
                try:
                    await comms_utils.clear_pending_whatsapp_call_intent(
                        pool_number=pool_number,
                        contact_number=to_number,
                    )
                except Exception as exc:
                    LOGGER.error(
                        f"{DEFAULT_ICON} Failed to clear pending WhatsApp call intent: {exc}",
                    )
            return {
                "status": "blocked",
                "note": (
                    "The contact rejected WhatsApp call permission. Do not place or request "
                    "another WhatsApp call unless they ask to try again."
                ),
            }

        if method == "needs_reconciliation":
            return {
                "status": "needs_reconciliation",
                "pending_callback": automatic_callback_available,
                "note": (
                    "WhatsApp reported a call-permission interaction without an accepted or "
                    "rejected payload. Do not place a direct call until permission is reconciled."
                ),
            }

        if method == "needs_permission":
            return {
                "status": "needs_permission",
                "pending_callback": automatic_callback_available,
                "note": (
                    "WhatsApp still requires the contact to approve calls from this business "
                    "before the call can ring. Tell the user to tap Allow calls / Call now in "
                    "WhatsApp, and do not claim that a call is ringing yet."
                ),
            }

        event = WhatsAppCallInviteSent(contact=fresh_contact)
        await self._event_broker.publish(
            "app:comms:whatsapp_call_invite_sent",
            event.to_json(),
        )
        self._record_offline_success(
            offline_reservation,
            attempted_content="<WhatsApp Call Invite Sent>",
            receiver_ids=[fresh_contact.get("contact_id") or contact_id],
            target_metadata={
                "contact_id": fresh_contact.get("contact_id") or contact_id,
                "whatsapp_number": to_number or "",
            },
            history_metadata={
                "contact_display_name": _get_contact_display_name(fresh_contact),
                "automatic_callback_available": automatic_callback_available,
            },
            provider_response=response,
            status="pending_callback" if automatic_callback_available else "completed",
        )
        if automatic_callback_available:
            return {
                "status": "ok",
                "pending_callback": True,
                "note": (
                    "WhatsApp requires the contact to approve calls from this business first. "
                    "A call-permission request was sent instead of the call. Tell the user "
                    "to tap Allow calls / Call now; when they approve, the call will be "
                    "placed automatically with your briefing context."
                ),
            }
        return {
            "status": "ok",
            "note": (
                "WhatsApp requires the contact to approve calls from this business first. "
                "A call-permission request was sent instead of the call. Because this send "
                "ran without a live assistant session, the callback was not queued with "
                "your briefing context."
            ),
        }

    # ------------------------------------------------------------------
    # Inactivity-followup opt-out primitives
    # ------------------------------------------------------------------

    async def stop_inactivity_followups(self) -> dict:
        """Stop sending inactivity re-engagement follow-ups to the boss.

        Call this when the boss explicitly declines further follow-ups
        (e.g. "no longer interested", "take me off your list", "stop
        contacting me"). Orchestra sets ``inactivity_followup_opted_out``
        so the daily inactivity-followup routine skips this Coordinator.

        Nothing is deleted — this only silences future follow-ups. Use
        :meth:`resume_inactivity_followups` if the boss later re-engages
        and wants to hear from us again.

        Returns:
            ``{"success": bool, "assistant_id": int | None}``
        """
        from unify.transcript_manager.activity_sync import (
            opt_out_of_inactivity_followups_via_orchestra,
        )

        agent_id = getattr(SESSION_DETAILS.assistant, "agent_id", None)
        success = opt_out_of_inactivity_followups_via_orchestra(agent_id)
        return {
            "success": bool(success),
            "assistant_id": int(agent_id) if agent_id is not None else None,
        }

    async def resume_inactivity_followups(self) -> dict:
        """Re-enable inactivity re-engagement follow-ups for the boss.

        Call this only when follow-ups were previously stopped (via
        :meth:`stop_inactivity_followups`) and the boss has now
        re-engaged and wants to keep hearing from us. Orchestra clears
        ``inactivity_followup_opted_out``.

        Safe to call when no opt-out is in effect — orchestra treats the
        clear as a no-op.

        Returns:
            ``{"success": bool, "assistant_id": int | None}``
        """
        from unify.transcript_manager.activity_sync import (
            opt_in_to_inactivity_followups_via_orchestra,
        )

        agent_id = getattr(SESSION_DETAILS.assistant, "agent_id", None)
        success = opt_in_to_inactivity_followups_via_orchestra(agent_id)
        return {
            "success": bool(success),
            "assistant_id": int(agent_id) if agent_id is not None else None,
        }
