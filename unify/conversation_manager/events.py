import json
import uuid
from collections.abc import Mapping as _Mapping
from typing import Optional, Any, ClassVar
from datetime import datetime
from dataclasses import dataclass, asdict, field

from pydantic import BaseModel

from unify.common.context_registry import ContextRegistry
from unify.common.prompt_helpers import now as prompt_now


def _coerce_int(value: Any) -> int | None:
    """Best-effort integer coercion for webhook / activation payloads.

    Lives in events.py because both the Pub/Sub ingress path and the
    in-process scheduler use it when constructing `TaskDue` from
    untyped JSON-derived dicts. Returns ``None`` for empty values or
    anything that won't parse cleanly as an int.
    """

    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def custom_dict_factory(kv):
    d = {}
    for k, v in kv:
        if isinstance(v, datetime):
            d[k] = v.isoformat()
        elif isinstance(v, BaseModel):
            d[k] = v.model_dump()
        else:
            d[k] = v
    return d


class _TruncatedReprMixin:
    """Mixin for events that need truncated repr (to avoid logging huge payloads)."""

    def __str__(self) -> str:
        return self._repr_truncated()

    def __repr__(self) -> str:
        return self._repr_truncated()

    def _repr_truncated(self) -> str:
        raise NotImplementedError


def _now_datetime() -> datetime:
    """Wrapper for prompt_now that returns datetime for dataclass default_factory."""
    return prompt_now(as_string=False)


@dataclass(kw_only=True)
class Event:
    timestamp: datetime = field(default_factory=_now_datetime)

    _registry: ClassVar[dict[str, "Event"]] = {}
    loggable: ClassVar[bool] = True
    content_logged: ClassVar[bool] = False
    prominent: ClassVar[bool] = False
    topic: ClassVar[str | None] = None

    def to_json(self):
        return json.dumps(self.to_dict())

    def to_dict(self):
        return {
            "event_name": self.__class__.__name__,
            "payload": asdict(self, dict_factory=custom_dict_factory),
        }

    def to_bus_event(self):
        from unify.events.event_bus import Event as BusEvent

        payload = self.to_dict()["payload"]
        return BusEvent(
            calling_id="",
            type="Comms",
            timestamp=self.timestamp.isoformat(),
            payload=payload,
            payload_cls=self.__class__.__name__,
        )

    @classmethod
    def from_dict(cls, data) -> "Event":
        import dataclasses

        target_cls = cls._registry.get(data["event_name"])
        if not target_cls:
            raise Exception(f"Class {data['event_name']} is not registered.")
        kwargs = data["payload"].copy()
        timestamp = kwargs.pop("timestamp")

        # Filter to only fields the target dataclass accepts
        valid_fields = {f.name for f in dataclasses.fields(target_cls)}
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_fields}

        return target_cls(
            **filtered_kwargs,
            timestamp=datetime.fromisoformat(timestamp),
        )

    @classmethod
    def from_json(cls, json_data):
        data = json.loads(json_data)
        return cls.from_dict(data)

    @classmethod
    def from_bus_event(cls, event):
        # Use mode="json" to ensure datetime objects are serialized to ISO strings,
        # which from_dict() expects for the timestamp field
        event_dump = event.model_dump(mode="json")
        data = {
            "event_name": event_dump["payload_cls"],
            "payload": event_dump["payload"],
        }
        return cls.from_dict(data)

    def __init_subclass__(cls):
        if cls.__name__ not in Event._registry:
            Event._registry[cls.__name__] = cls
        return cls


# --------------------------------------------------------------------------- #
# Comms Events
# --------------------------------------------------------------------------- #


@dataclass
class PhoneCallReceived(Event):
    topic: ClassVar[str | None] = "app:comms:call_received"
    prominent: ClassVar[bool] = True

    contact: dict
    conference_name: str = ""
    room_name: str | None = None
    call_session_id: str | None = None
    provider_call_sid: str | None = None


@dataclass
class PhoneCallAnswered(Event):
    topic: ClassVar[str | None] = "app:comms:call_answered"
    prominent: ClassVar[bool] = True

    contact: dict


@dataclass
class PhoneCallNotAnswered(Event):
    """Outbound call was not answered (no-answer, busy, failed, etc.)."""

    topic: ClassVar[str | None] = "app:comms:call_not_answered"
    prominent: ClassVar[bool] = True

    contact: dict
    reason: str = "no-answer"  # Twilio status: no-answer, busy, canceled, failed


@dataclass
class UnifyMeetReceived(Event):
    """Frontend/worker confirmed agent connected to room; begin LLM."""

    topic: ClassVar[str | None] = "app:comms:unify_meet_received"
    prominent: ClassVar[bool] = True

    contact: dict
    room_name: str | None = None
    opening_config: dict | None = None
    call_session_id: str | None = None


@dataclass
class PhoneCallStarted(Event):
    topic: ClassVar[str | None] = "app:comms:phone_call_started"
    prominent: ClassVar[bool] = True

    contact: dict


@dataclass
class UnifyMeetStarted(Event):
    """A web-based voice/video meeting session has started (no phone number).

    "contact" should reference the boss/user contact id (typically 1).
    """

    topic: ClassVar[str | None] = "app:comms:unify_meet_started"
    prominent: ClassVar[bool] = True

    contact: dict
    call_session_id: str | None = None


@dataclass
class InboundPhoneUtterance(Event):
    """Utterance received from the other party during a phone call."""

    topic: ClassVar[str | None] = "app:comms:phone_utterance"

    contact: dict
    content: str
    # Per-turn id from the voice agent, used to cancel exactly the slow-brain run
    # this turn spawns if the fast brain resolves the turn itself.
    turn_id: int | None = None


@dataclass
class InboundUnifyMeetUtterance(Event):
    """Utterance received from the other party during a web-based voice/video meeting."""

    topic: ClassVar[str | None] = "app:comms:unify_utterance"

    contact: dict
    content: str
    turn_id: int | None = None


@dataclass
class InboundWhatsAppCallUtterance(Event):
    """Utterance received from the other party during a WhatsApp voice call."""

    topic: ClassVar[str | None] = "app:comms:whatsapp_call_utterance"

    contact: dict
    content: str
    turn_id: int | None = None


@dataclass
class VoiceInterrupt(Event):
    """The caller barged in before the assistant finished its spoken line.

    Carries the part of the assistant's message that was NOT heard so the slow
    brain knows it was cut off and can decide whether to re-deliver it.
    """

    topic: ClassVar[str | None] = "app:comms:voice_interrupt"

    contact: dict
    spoken_prefix: str = ""
    unheard_remainder: str = ""


@dataclass
class FastBrainContinued(Event):
    """The fast brain resolved this turn itself (CONTINUE or SMALLTALK).

    Emitted when the fast brain takes full ownership of the turn's reply - either
    resuming an interrupted line verbatim (continuation) or fully answering a pure
    small-talk turn. Signals the CM to cancel the slow-brain run that was eagerly
    started for this user turn, so the slow brain does not also answer.
    """

    topic: ClassVar[str | None] = "app:comms:fast_brain_continued"

    contact: dict
    # The user turn the fast brain resolved; the CM cancels exactly that turn's
    # slow-brain run (wherever it sits in the debouncer queue).
    turn_id: int | None = None


@dataclass
class PhoneCallEnded(Event):
    topic: ClassVar[str | None] = "app:comms:phone_call_ended"
    prominent: ClassVar[bool] = True

    contact: dict


@dataclass
class UnifyMeetEnded(Event):
    """The web-based voice/video meeting session has ended."""

    topic: ClassVar[str | None] = "app:comms:unify_meet_ended"
    prominent: ClassVar[bool] = True

    contact: dict
    call_session_id: str | None = None


@dataclass
class GoogleMeetReceived(Event):
    """A request to join a Google Meet call via browser."""

    topic: ClassVar[str | None] = "app:comms:googlemeet_received"
    prominent: ClassVar[bool] = True

    contact: dict
    meet_url: str


@dataclass
class GoogleMeetStarted(Event):
    """The Google Meet browser session is active and audio bridge is running."""

    topic: ClassVar[str | None] = "app:comms:googlemeet_started"
    prominent: ClassVar[bool] = True

    contact: dict


@dataclass
class GoogleMeetEnded(Event):
    """The Google Meet browser session has ended."""

    topic: ClassVar[str | None] = "app:comms:googlemeet_ended"
    prominent: ClassVar[bool] = True

    contact: dict


@dataclass
class InboundGoogleMeetUtterance(Event):
    """Utterance received from a participant during a Google Meet call."""

    topic: ClassVar[str | None] = "app:comms:googlemeet_utterance"

    contact: dict
    content: str
    speaker_label: str | None = None
    participant_names: list[str] | None = None
    diarization_speaker_id: str | None = None
    turn_id: int | None = None


@dataclass
class OutboundGoogleMeetUtterance(Event):
    """Utterance sent by the assistant during a Google Meet call."""

    topic: ClassVar[str | None] = "app:comms:googlemeet_utterance"

    contact: dict
    content: str
    participant_names: list[str] | None = None


@dataclass
class GoogleMeetParticipantJoined(Event):
    """A participant joined the Google Meet session."""

    topic: ClassVar[str | None] = "app:comms:googlemeet_participant"

    contact: dict
    participant_name: str


@dataclass
class GoogleMeetParticipantLeft(Event):
    """A participant left the Google Meet session."""

    topic: ClassVar[str | None] = "app:comms:googlemeet_participant"

    contact: dict
    participant_name: str


@dataclass
class TeamsMeetReceived(Event):
    """A request to join a Microsoft Teams meeting via browser."""

    topic: ClassVar[str | None] = "app:comms:teamsmeet_received"
    prominent: ClassVar[bool] = True

    contact: dict
    meet_url: str


@dataclass
class TeamsMeetStarted(Event):
    """The Teams meeting browser session is active and audio bridge is running."""

    topic: ClassVar[str | None] = "app:comms:teamsmeet_started"
    prominent: ClassVar[bool] = True

    contact: dict


@dataclass
class TeamsMeetEnded(Event):
    """The Teams meeting browser session has ended."""

    topic: ClassVar[str | None] = "app:comms:teamsmeet_ended"
    prominent: ClassVar[bool] = True

    contact: dict


@dataclass
class InboundTeamsMeetUtterance(Event):
    """Utterance received from a participant during a Teams meeting."""

    topic: ClassVar[str | None] = "app:comms:teamsmeet_utterance"

    contact: dict
    content: str
    speaker_label: str | None = None
    participant_names: list[str] | None = None
    diarization_speaker_id: str | None = None
    turn_id: int | None = None


@dataclass
class OutboundTeamsMeetUtterance(Event):
    """Utterance sent by the assistant during a Teams meeting."""

    topic: ClassVar[str | None] = "app:comms:teamsmeet_utterance"

    contact: dict
    content: str
    participant_names: list[str] | None = None


@dataclass
class TeamsMeetParticipantJoined(Event):
    """A participant joined the Teams meeting session."""

    topic: ClassVar[str | None] = "app:comms:teamsmeet_participant"

    contact: dict
    participant_name: str


@dataclass
class TeamsMeetParticipantLeft(Event):
    """A participant left the Teams meeting session."""

    topic: ClassVar[str | None] = "app:comms:teamsmeet_participant"

    contact: dict
    participant_name: str


@dataclass
class RecordingReady(Event):
    """A call/meet recording has been processed and is available in GCS."""

    topic: ClassVar[str | None] = "app:comms:recording_ready"

    conference_name: str
    recording_url: str
    call_session_id: str | None = None
    provider_call_sid: str | None = None
    room_name: str | None = None


@dataclass
class SMSReceived(Event):
    topic: ClassVar[str | None] = "app:comms:msg_message"
    content_logged: ClassVar[bool] = True

    contact: dict
    content: str


@dataclass
class WhatsAppReceived(Event):
    topic: ClassVar[str | None] = "app:comms:whatsapp_message"
    content_logged: ClassVar[bool] = True

    contact: dict
    content: str
    attachments: list[dict] | None = None


@dataclass
class WhatsAppCallReceived(Event):
    """An inbound voice call initiated via WhatsApp Business calling."""

    topic: ClassVar[str | None] = "app:comms:whatsapp_call_received"
    prominent: ClassVar[bool] = True

    contact: dict
    conference_name: str = ""
    room_name: str | None = None
    call_session_id: str | None = None
    provider_call_sid: str | None = None


@dataclass
class WhatsAppCallStarted(Event):
    topic: ClassVar[str | None] = "app:comms:whatsapp_call_started"
    prominent: ClassVar[bool] = True

    contact: dict


@dataclass
class WhatsAppCallEnded(Event):
    topic: ClassVar[str | None] = "app:comms:whatsapp_call_ended"
    prominent: ClassVar[bool] = True

    contact: dict


@dataclass
class WhatsAppCallSent(Event):
    """Outbound WhatsApp call placed directly (permission was granted)."""

    topic: ClassVar[str | None] = "app:comms:whatsapp_call_sent"
    prominent: ClassVar[bool] = True

    contact: dict
    content: str = "<Sending WhatsApp Call...>"
    onboarding_trigger_step_id: str | None = None
    onboarding_reply_step_id: str | None = None
    onboarding_request_id: str | None = None
    onboarding_origin_event_id: str | None = None


@dataclass
class WhatsAppCallInviteSent(Event):
    """Call permission template sent (approval required before calling)."""

    topic: ClassVar[str | None] = "app:comms:whatsapp_call_invite_sent"
    prominent: ClassVar[bool] = True

    contact: dict
    content: str = (
        "<WhatsApp Call Permission Request Sent: waiting for the user to allow calls>"
    )


@dataclass
class WhatsAppCallAnswered(Event):
    """Outbound WhatsApp call was answered by the contact."""

    topic: ClassVar[str | None] = "app:comms:whatsapp_call_answered"
    prominent: ClassVar[bool] = True

    contact: dict


@dataclass
class WhatsAppCallNotAnswered(Event):
    """Outbound WhatsApp call was not answered."""

    topic: ClassVar[str | None] = "app:comms:whatsapp_call_not_answered"
    prominent: ClassVar[bool] = True

    contact: dict
    reason: str = "no-answer"


@dataclass
class WhatsAppCallPermissionResponse(Event):
    """Contact responded to a WhatsApp call permission request."""

    topic: ClassVar[str | None] = "app:comms:whatsapp_call_permission"
    prominent: ClassVar[bool] = True

    contact: dict
    accepted: bool
    status: str = ""


@dataclass
class DiscordMessageReceived(Event):
    """A direct message received from a user via a Discord bot."""

    topic: ClassVar[str | None] = "app:comms:discord_message"
    content_logged: ClassVar[bool] = True

    contact: dict
    content: str
    channel_id: str = ""
    bot_id: str = ""
    message_id: str = ""
    attachments: list[dict] = field(default_factory=list)


@dataclass
class DiscordChannelMessageReceived(Event):
    """A message received in a Discord guild channel via @mention."""

    topic: ClassVar[str | None] = "app:comms:discord_channel_message"
    content_logged: ClassVar[bool] = True

    contact: dict
    content: str
    channel_id: str = ""
    guild_id: str = ""
    bot_id: str = ""
    message_id: str = ""
    attachments: list[dict] = field(default_factory=list)


@dataclass
class DiscordMessageSent(Event):
    """A direct message sent to a user via a Discord bot."""

    topic: ClassVar[str | None] = "app:comms:discord_message_sent"
    content_logged: ClassVar[bool] = True

    contact: dict
    content: str
    onboarding_trigger_step_id: str | None = None
    onboarding_reply_step_id: str | None = None
    onboarding_request_id: str | None = None
    onboarding_origin_event_id: str | None = None


@dataclass
class DiscordChannelMessageSent(Event):
    """A message sent to a Discord guild channel via a bot."""

    topic: ClassVar[str | None] = "app:comms:discord_channel_message_sent"
    content_logged: ClassVar[bool] = True

    contact: dict
    content: str
    channel_id: str = ""
    guild_id: str = ""
    onboarding_trigger_step_id: str | None = None
    onboarding_reply_step_id: str | None = None
    onboarding_request_id: str | None = None
    onboarding_origin_event_id: str | None = None


@dataclass
class SlackMessageReceived(Event):
    """A direct message received from a user via a Slack app.

    ``routing_metadata`` carries Orchestra-side routing context (whether
    the route was created via token addressing, whether the assistant is
    the coordinator handling a misroute, etc.) and is surfaced to the
    assistant via the renderer.
    """

    topic: ClassVar[str | None] = "app:comms:slack_message"
    content_logged: ClassVar[bool] = True

    contact: dict
    content: str
    team_id: str = ""
    channel_id: str = ""
    bot_user_id: str = ""
    event_ts: str = ""
    thread_ts: str = ""
    message_id: str = ""
    attachments: list[dict] = field(default_factory=list)
    routing_metadata: dict = field(default_factory=dict)


@dataclass
class SlackChannelMessageReceived(Event):
    """A message received in a Slack channel via @mention of the app.

    ``routing_metadata`` carries Orchestra-side routing context (token
    used, thread inheritance, coordinator fallback, ambiguous-token
    fan-out hints) and is surfaced to the assistant via the renderer.
    """

    topic: ClassVar[str | None] = "app:comms:slack_channel_message"
    content_logged: ClassVar[bool] = True

    contact: dict
    content: str
    team_id: str = ""
    channel_id: str = ""
    bot_user_id: str = ""
    event_ts: str = ""
    thread_ts: str = ""
    message_id: str = ""
    attachments: list[dict] = field(default_factory=list)
    routing_metadata: dict = field(default_factory=dict)


@dataclass
class SlackMessageSent(Event):
    """A direct message sent to a Slack user via the app."""

    topic: ClassVar[str | None] = "app:comms:slack_message_sent"
    content_logged: ClassVar[bool] = True

    contact: dict
    content: str
    team_id: str = ""
    channel_id: str = ""
    thread_ts: str = ""
    onboarding_trigger_step_id: str | None = None
    onboarding_reply_step_id: str | None = None
    onboarding_request_id: str | None = None
    onboarding_origin_event_id: str | None = None


@dataclass
class SlackChannelMessageSent(Event):
    """A message sent to a Slack channel via the app."""

    topic: ClassVar[str | None] = "app:comms:slack_channel_message_sent"
    content_logged: ClassVar[bool] = True

    contact: dict
    content: str
    team_id: str = ""
    channel_id: str = ""
    thread_ts: str = ""
    onboarding_trigger_step_id: str | None = None
    onboarding_reply_step_id: str | None = None
    onboarding_request_id: str | None = None
    onboarding_origin_event_id: str | None = None


@dataclass
class TeamsMessageReceived(Event):
    """A message received in a Microsoft Teams chat (1:1, group, or meeting)."""

    topic: ClassVar[str | None] = "app:comms:teams_message"
    content_logged: ClassVar[bool] = True

    contact: dict
    content: str
    chat_id: str = ""
    message_id: str = ""
    chat_type: str | None = None
    chat_topic: str | None = None
    attachments: list[dict] = field(default_factory=list)
    participants: list[int] = field(default_factory=list)


@dataclass
class TeamsChannelMessageReceived(Event):
    """A message received in a Microsoft Teams channel."""

    topic: ClassVar[str | None] = "app:comms:teams_channel_message"
    content_logged: ClassVar[bool] = True

    contact: dict
    content: str
    channel_id: str = ""
    team_id: str = ""
    message_id: str = ""
    is_reply: bool = False
    parent_message_id: str | None = None
    thread_id: str = ""
    post_subject: str | None = None
    attachments: list[dict] = field(default_factory=list)
    participants: list[int] = field(default_factory=list)


@dataclass
class TeamsMessageSent(Event):
    """A message sent in a Microsoft Teams chat."""

    topic: ClassVar[str | None] = "app:comms:teams_message_sent"
    content_logged: ClassVar[bool] = True

    contact: dict
    content: str
    chat_id: str = ""
    attachments: list[dict] | None = None
    participants: list[int] = field(default_factory=list)


@dataclass
class TeamsChannelMessageSent(Event):
    """A message sent in a Microsoft Teams channel."""

    topic: ClassVar[str | None] = "app:comms:teams_channel_message_sent"
    content_logged: ClassVar[bool] = True

    contact: dict
    content: str
    channel_id: str = ""
    team_id: str = ""
    attachments: list[dict] | None = None
    participants: list[int] = field(default_factory=list)


@dataclass
class TeamsChannelCreated(Event):
    """A new Microsoft Teams channel was created by the assistant."""

    topic: ClassVar[str | None] = "app:comms:teams_channel_created"
    content_logged: ClassVar[bool] = True

    contact: dict
    team_id: str = ""
    channel_id: str = ""
    display_name: str = ""
    description: str = ""
    membership_type: str = "standard"


@dataclass
class TeamsMeetCreated(Event):
    """A Microsoft Teams meeting was created by the assistant via Graph.

    ``mode`` is ``"instant"`` (reusable ad-hoc meeting — ``meeting_id``
    populated, ``event_id``/``web_link`` empty) or ``"scheduled"`` (calendar
    event with attached Teams meeting — ``event_id``/``web_link`` populated,
    ``meeting_id`` empty). ``attendees`` lists the UPNs Graph was asked to
    invite (scheduled mode only).
    """

    topic: ClassVar[str | None] = "app:comms:teams_meet_created"
    content_logged: ClassVar[bool] = True

    contact: dict
    mode: str = "scheduled"
    subject: str = ""
    join_web_url: str = ""
    meeting_id: str = ""
    event_id: str = ""
    start: str = ""
    end: str = ""
    attendees: list[str] = field(default_factory=list)
    web_link: str = ""


@dataclass
class UnifyMessageReceived(Event):
    """A message was received via the Unify console chat interface.

    Attachments are downloaded asynchronously to the Attachments folder.
    Each attachment is a dict with keys: id, filename, gs_url, content_type, size_bytes.
    Files are saved to Attachments/{id}_{filename} and can be accessed via FileManager.
    """

    topic: ClassVar[str | None] = "app:comms:unify_message_message"
    content_logged: ClassVar[bool] = True

    contact: dict
    content: str
    # List of attachment dicts with full metadata (files saved to Attachments/).
    attachments: list[dict] = field(default_factory=list)


@dataclass
class PhoneCallSent(Event):
    topic: ClassVar[str | None] = "app:comms:make_call"
    prominent: ClassVar[bool] = True

    contact: dict
    content: str = "<Sending Call...>"
    provider_call_sid: str = ""
    onboarding_trigger_step_id: str | None = None
    onboarding_reply_step_id: str | None = None
    onboarding_request_id: str | None = None
    onboarding_origin_event_id: str | None = None


@dataclass
class OutboundPhoneUtterance(Event):
    """Utterance sent by the assistant during a phone call."""

    topic: ClassVar[str | None] = "app:comms:phone_utterance"

    contact: dict
    content: str


@dataclass
class OutboundUnifyMeetUtterance(Event):
    """Utterance sent by the assistant during a web-based voice/video meeting."""

    topic: ClassVar[str | None] = "app:comms:unify_utterance"

    contact: dict
    content: str


@dataclass
class OutboundWhatsAppCallUtterance(Event):
    """Utterance sent by the assistant during a WhatsApp voice call."""

    topic: ClassVar[str | None] = "app:comms:whatsapp_call_utterance"

    contact: dict
    content: str


@dataclass
class FastBrainNotification(Event):
    """Notification delivered to the fast brain during a voice call.

    ``message`` is injected as silent ``[notification]`` context for non-proactive
    sources. When ``should_speak`` is True, the fast brain speaks
    ``spoken_message`` if set, otherwise ``message``, verbatim via TTS.
    """

    topic: ClassVar[str | None] = "app:comms:assistant_notification"

    contact: dict
    message: str = ""
    should_speak: bool = False
    spoken_message: str = ""
    source: str = ""
    agent_service_url: str = ""
    llm_log_path: str = ""
    # Optional short note (bundled with a spoken line) the fast brain may use to
    # give a basic direct reply to the caller's next message. Never spoken on its
    # own; an empty value clears any prior note.
    fast_brain_guidance: str = ""


@dataclass
class AssistantTurnInjected(Event):
    """Assistant-authored voice-call context inserted without forcing speech."""

    topic: ClassVar[str | None] = "app:comms:assistant_turn_injected"

    contact: dict
    content: str
    source: str = ""
    schedule_proactive: bool = False


@dataclass
class ProactiveSpeechControl(Event):
    """External control for whether proactive voice-call speech may schedule."""

    topic: ClassVar[str | None] = "app:comms:proactive_speech_control"

    enabled: bool
    source: str = ""
    reason: str = ""
    schedule_now: bool = False


@dataclass
class FastBrainMoodClassified(Event):
    """Avatar mood classified from the current fast-brain voice transcript."""

    topic: ClassVar[str | None] = "app:comms:fast_brain_mood"

    contact: dict
    channel: str
    mood: str
    avatar_mood: str
    trigger_role: str
    trigger_utterance_id: str
    turn_index: int
    model: str = ""


@dataclass
class EmailReceived(Event):
    """An email was received from a contact.

    Attachments are downloaded asynchronously to the Attachments folder.
    Each attachment is a dict with keys: id, filename.
    """

    topic: ClassVar[str | None] = "app:comms:email_message"
    content_logged: ClassVar[bool] = True

    contact: dict
    subject: str
    body: str
    # Email provider identifier used for threading (e.g., RFC Message-ID header value).
    # This is *not* the TranscriptManager's auto-incremented message_id.
    email_id: Optional[str] = None
    # Provider conversation identifier used for deterministic thread targeting
    # (e.g., Gmail threadId).
    thread_id: str | None = None
    # List of attachment dicts with metadata (files saved to Attachments/).
    attachments: list[dict] = field(default_factory=list)
    # Recipients from the original email (for reply-all functionality)
    to: list[str] = field(default_factory=list)
    cc: list[str] = field(default_factory=list)
    bcc: list[str] = field(default_factory=list)


# assistant events
@dataclass
class SMSSent(Event):
    topic: ClassVar[str | None] = "app:comms:sms_sent"
    content_logged: ClassVar[bool] = True

    contact: dict
    content: str
    onboarding_trigger_step_id: str | None = None
    onboarding_reply_step_id: str | None = None
    onboarding_request_id: str | None = None
    onboarding_origin_event_id: str | None = None


@dataclass
class WhatsAppSent(Event):
    topic: ClassVar[str | None] = "app:comms:whatsapp_sent"
    content_logged: ClassVar[bool] = True

    contact: dict
    content: str
    via_template: bool = False
    delivered_content: str | None = None
    attachments: list[dict] | None = None
    onboarding_trigger_step_id: str | None = None
    onboarding_reply_step_id: str | None = None
    onboarding_request_id: str | None = None
    onboarding_origin_event_id: str | None = None


@dataclass
class UnifyMessageSent(Event):
    """A message was sent via the Unify console chat interface.

    Attachments are uploaded to GCS and copied to Attachments/{id}_{filename}.
    Each attachment is a dict with keys: id, filename, gs_url, content_type, size_bytes, filepath.
    """

    topic: ClassVar[str | None] = "app:comms:unify_message_sent"
    content_logged: ClassVar[bool] = True

    contact: dict
    content: str
    # List of attachment dicts with full metadata.
    attachments: list[dict] = field(default_factory=list)
    onboarding_trigger_step_id: str | None = None
    onboarding_reply_step_id: str | None = None
    onboarding_request_id: str | None = None
    onboarding_origin_event_id: str | None = None


@dataclass
class EmailSent(Event):
    """An email was sent to a contact.

    Each attachment is a dict with keys: id, filename, filepath.
    """

    topic: ClassVar[str | None] = "app:comms:email_sent"
    content_logged: ClassVar[bool] = True

    contact: dict
    subject: str
    body: str
    # Email provider identifier used for threading (e.g., RFC Message-ID header value).
    # This is *not* the TranscriptManager's auto-incremented message_id.
    email_id_replied_to: str | None = None
    # Provider conversation identifier targeted by this send (e.g., Gmail threadId).
    thread_id: str | None = None
    # List of attachment dicts with metadata.
    attachments: list[dict] = field(default_factory=list)
    # Recipients the email was sent to
    to: list[str] = field(default_factory=list)
    cc: list[str] = field(default_factory=list)
    bcc: list[str] = field(default_factory=list)
    onboarding_trigger_step_id: str | None = None
    onboarding_reply_step_id: str | None = None
    onboarding_request_id: str | None = None
    onboarding_origin_event_id: str | None = None


@dataclass
class ApiMessageReceived(Event):
    """A programmatic message received via the REST API.

    Attachments are downloaded to Attachments/{id}_{filename}.
    Each attachment is a dict with keys: id, filename, gs_url, content_type, size_bytes.
    Tags are opaque strings supplied by the developer for routing and context.
    """

    topic: ClassVar[str | None] = "app:comms:api_message_message"
    content_logged: ClassVar[bool] = True

    contact: dict
    content: str
    api_message_id: str = ""
    attachments: list[dict] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)


@dataclass
class ApiMessageSent(Event):
    """A response sent back to the developer via the REST API.

    Attachments are uploaded to GCS and copied to Attachments/{id}_{filename}.
    Each attachment is a dict with keys: id, filename, gs_url, content_type, size_bytes, filepath.
    Tags are typically echoed from the inbound message for routing.
    """

    topic: ClassVar[str | None] = "app:comms:api_message_sent"
    content_logged: ClassVar[bool] = True

    contact: dict
    content: str
    api_message_id: str = ""
    attachments: list[dict] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)


@dataclass
class UnknownContactCreated(Event):
    """A new contact was automatically created from an unknown inbound message.

    This event is published when an inbound SMS, email, or call arrives from
    a sender that is not in the Contacts table and not in the BlackList.

    The contact is created with:
    - Only the medium field populated (phone_number or email_address)
    - should_respond=False to prevent automatic responses
    - A response_policy guiding the assistant to seek boss guidance

    The ConversationManager should use this event to potentially notify the
    boss and ask for guidance on how to handle this new contact.
    """

    topic: ClassVar[str | None] = "app:comms:unknown_contact_created"

    contact: dict
    medium: str  # The communication medium (e.g., "sms_message", "email", "phone_call")
    message_preview: str = ""  # Optional preview of the initial message


@dataclass
class _SessionConfigBase(Event):
    """Base class for session configuration events (StartupEvent, AssistantUpdateEvent)."""

    loggable: ClassVar[bool] = False
    api_key: str
    medium: str
    assistant_id: str
    user_id: str
    assistant_first_name: str
    assistant_surname: str
    assistant_age: str
    assistant_nationality: str
    assistant_about: str
    assistant_number: str
    assistant_email: str
    user_first_name: str
    user_surname: str
    user_number: str
    user_email: str
    voice_id: str
    binding_id: str = ""
    voice_provider: str = "cartesia"
    assistant_whatsapp_number: str = ""
    assistant_discord_bot_id: str = ""
    assistant_slack_bot_user_id: str = ""
    assistant_is_coordinator: bool = False
    assistant_timezone: str = (
        ""  # IANA timezone identifier; default empty for backward compat
    )
    assistant_job_title: str = ""
    assistant_email_provider: str = "google_workspace"
    self_contact_id: int = 0
    boss_contact_id: int = 1
    desktop_mode: str = "ubuntu"
    desktop_url: str | None = None
    user_whatsapp_number: str = ""
    # Per-user desktop links: each {owner_user_id, url, os, filesys_sync}.
    user_desktops: list[dict[str, Any]] = field(default_factory=list)
    org_id: int | None = None
    org_name: str = ""
    team_ids: list[int] = field(default_factory=list)
    team_summaries: list[dict[str, Any]] = field(default_factory=list)
    is_coordinator: bool = False
    update_kind: str = "general"
    wake_reasons: list[dict[str, Any]] = field(default_factory=list)
    # Demo assistant metadata ID. If set, this is a demo session.
    # Unity derives demo_mode from (demo_id is not None).
    demo_id: int | None = None


@dataclass
class StartupEvent(_SessionConfigBase):
    """Initial session configuration sent when ConversationManager starts."""


@dataclass
class InitializationComplete(Event):
    """Published when ConversationManager has fully initialized all managers."""

    loggable: ClassVar[bool] = False


@dataclass
class AssistantUpdateEvent(_SessionConfigBase):
    """Updated session configuration sent to a running ConversationManager."""


@dataclass
class Ping(Event):
    loggable: ClassVar[bool] = False
    kind: str


@dataclass
class Error(Event):
    prominent: ClassVar[bool] = True

    message: str


@dataclass
class LogMessageResponse(Event):
    medium: str
    exchange_id: int


@dataclass
class ContactInfoResponse(Event):
    contact_details: dict[str, Any]


@dataclass(repr=False)
class StoreChatHistory(_TruncatedReprMixin, Event):
    chat_history: list[dict]

    def _repr_truncated(self) -> str:
        return f"{self.__class__.__name__}(chat_history_len={len(self.chat_history)})"


@dataclass(repr=False)
class GetChatHistory(_TruncatedReprMixin, Event):
    loggable: ClassVar[bool] = False
    chat_history: list[dict]

    def _repr_truncated(self) -> str:
        return f"{self.__class__.__name__}(chat_history_len={len(self.chat_history)})"


@dataclass(repr=False)
class GetBusEventsResponse(_TruncatedReprMixin, Event):
    loggable: ClassVar[bool] = False
    events: list[dict[str, Any]]

    def _repr_truncated(self) -> str:
        return f"{self.__class__.__name__}(events_len={len(self.events)})"


@dataclass
class PreHireMessage(Event):
    content: str
    role: str
    exchange_id: int


# --------------------------------------------------------------------------- #
# LLM inference events
# --------------------------------------------------------------------------- #
@dataclass
class LLMInput(Event):
    chat_history: list[dict]


@dataclass
class UpdateContactRollingSummaryResponse(Event):
    rolling_summaries: list[tuple[int, str]]


# --------------------------------------------------------------------------- #
# ConversationManagerHandle Events
# --------------------------------------------------------------------------- #


@dataclass
class NotificationInjectedEvent(Event):
    """Event to inject a notification into the ConversationManager."""

    content: str
    source: str
    target_conversation_id: str
    interjection_id: str = field(default_factory=lambda: str(uuid.uuid4().hex[:12]))
    pinned: bool = False


@dataclass
class NotificationUnpinnedEvent(Event):
    """Event to unpin a previously pinned interjection."""

    interjection_id: str
    target_conversation_id: str


# --------------------------------------------------------------------------- #
# Actor Events
# --------------------------------------------------------------------------- #


@dataclass(repr=False)
class ActorRequest(_TruncatedReprMixin, Event):
    """Event to ask or request the Actor to perform a task."""

    action_name: str
    query: str
    parent_chat_context: list[dict]

    def _repr_truncated(self) -> str:
        return (
            f"{self.__class__.__name__}(action_name={self.action_name}, "
            f"query={self.query}, "
            f"parent_chat_context_len={len(self.parent_chat_context)})"
        )


@dataclass
class ActorResponse(Event):
    """Event to respond to an Actor request."""

    handle_id: int
    action_name: str
    query: str
    response: str


@dataclass(repr=False)
class ActorHandleRequest(_TruncatedReprMixin, Event):
    """Event to any action on an existing Actor handle."""

    handle_id: int
    action_name: str
    query: str
    parent_chat_context: list[dict]

    def _repr_truncated(self) -> str:
        return (
            f"{self.__class__.__name__}(handle_id={self.handle_id}, "
            f"action_name={self.action_name}, "
            f"query={self.query}, "
            f"parent_chat_context_len={len(self.parent_chat_context)})"
        )


@dataclass
class ActorHandleResponse(Event):
    """Event to respond to an Actor handle request."""

    handle_id: int
    action_name: str
    query: str
    response: str
    call_id: str


@dataclass
class ActorResult(Event):
    """Event to the result of an Actor task."""

    handle_id: int
    success: bool
    result: dict | str | None = None
    error: str | None = None
    action_type: str = ""


@dataclass
class ActorClarificationRequest(Event):
    """Event to request clarification from the Actor."""

    handle_id: int
    query: str
    call_id: str


@dataclass
class ActorClarificationResponse(Event):
    """Event to respond to an Actor clarification request."""

    handle_id: int
    response: str
    call_id: str


@dataclass
class ActorNotification(Event):
    """Event to forward a notification from an Actor handle.

    Notifications carry status updates emitted explicitly by the actor via
    ``send_notification`` or ``notify()``.  The ``completed`` flag
    distinguishes a completion announcement from an in-progress update so
    downstream consumers (e.g. the fast brain) can render them with
    unambiguous prefixes.
    """

    handle_id: int
    response: str
    completed: bool = False


@dataclass
class ActorSessionResponse(Event):
    """Event signalling that a persistent actor session has completed a turn.

    Unlike ``ActorNotification``, a session response means the actor has
    finished its current work and is **waiting for the next instruction**
    (via ``interject``).  The ``content`` field carries the actor's output
    for this turn.
    """

    handle_id: int
    content: str


@dataclass
class ActorHandleStarted(Event):
    action_name: str
    handle_id: id
    query: str
    response_format: dict | None = None


@dataclass
class ComputerActCompleted(_TruncatedReprMixin, Event):
    """Fired when a visible computer session's act() completes anywhere in the
    system (CM fast path, CodeActActor, sub-agents).  Covers both desktop and
    web-vm sessions.  Carries the instruction and the agent's summary."""

    instruction: str = ""
    summary: str = ""


# --------------------------------------------------------------------------- #
# Desktop Lifecycle Events
# --------------------------------------------------------------------------- #


@dataclass
class AssistantDesktopReady(Event):
    """The assistant's managed VM desktop is reachable and ready for use.

    Published by the Communication service as a ``unity_system_event`` once
    the VM passes health checks.  Replaces the previous polling of
    ``/infra/vm/status``.
    """

    topic: ClassVar[str | None] = "app:comms:assistant_desktop_ready"

    binding_id: str = ""
    desktop_url: str = ""
    vm_type: str = ""


@dataclass
class TaskDue(Event):
    """A scheduled task activation became due for the live assistant.

    Communication publishes this payload either directly as a `unity_system_event`
    or indirectly via `StartupEvent.wake_reasons` during a cold start. The local
    install's `LocalActivationScheduler` also publishes the same event from an
    in-process asyncio timer. Unity uses the activation identity fields to
    reject stale deliveries before nudging the slow brain to execute the task.
    The packet also carries compact human-facing wake context so the slow and
    fast brains do not have to infer meaning from a bare task id alone.
    """

    topic: ClassVar[str | None] = "app:comms:task_due"

    task_id: int
    source_task_log_id: int
    activation_revision: str
    scheduled_for: str
    destination: str | None = None
    execution_mode: str = "live"
    source_type: str = "scheduled"
    task_label: str = ""
    task_summary: str = ""
    visibility_policy: str = "silent_by_default"
    recurrence_hint: str = "one_off"
    reason: str = ""

    @classmethod
    def from_dict(
        cls,
        payload: Any,
        *,
        reason: str = "",
    ) -> "TaskDue | None":
        """Build a `TaskDue` from a dict-shaped payload, or return ``None``.

        Centralises the field contract shared by every producer of
        `TaskDue`:

        - Communication's `unity_system_event` Pub/Sub payload
          (`ScheduledTaskDuePayload` shape).
        - Communication's cold-start `wake_reasons` dict shape (the same
          fields plus a leading ``type`` discriminator the caller is
          responsible for checking).
        - The local in-process `LocalActivationScheduler`, which converts
          a projected `TaskActivationSnapshot` into the same dict shape
          before calling this method.

        Returns ``None`` when any required identity field
        (``task_id``, ``source_task_log_id``, ``activation_revision``,
        ``scheduled_for``) is missing or malformed; the caller decides
        how to log / drop the event. Callers that dispatch from a
        heterogeneous wake-reason stream must still type-check the
        payload (e.g. ``payload.get("type") == "task_due"``) before
        calling this method.
        """

        if not isinstance(payload, _Mapping):
            return None
        task_id = _coerce_int(payload.get("task_id"))
        source_task_log_id = _coerce_int(payload.get("source_task_log_id"))
        activation_revision = str(payload.get("activation_revision") or "")
        scheduled_for = str(payload.get("scheduled_for") or "")
        if task_id is None or source_task_log_id is None:
            return None
        if not activation_revision or not scheduled_for:
            return None
        try:
            destination = ContextRegistry.canonical_destination(
                payload.get("destination"),
            )
        except ValueError:
            return None
        task_label = str(payload.get("task_label") or "")
        resolved_reason = reason or (
            f"Scheduled task '{task_label}' became due."
            if task_label
            else f"Scheduled task {task_id} became due."
        )
        return cls(
            task_id=task_id,
            source_task_log_id=source_task_log_id,
            activation_revision=activation_revision,
            scheduled_for=scheduled_for,
            destination=destination,
            execution_mode=str(payload.get("execution_mode") or "live"),
            source_type=str(payload.get("source_type") or "scheduled"),
            task_label=task_label,
            task_summary=str(payload.get("task_summary") or ""),
            visibility_policy=str(
                payload.get("visibility_policy") or "silent_by_default",
            ),
            recurrence_hint=str(payload.get("recurrence_hint") or "one_off"),
            reason=resolved_reason,
        )


@dataclass
class TaskTriggerRequested(Event):
    """A public REST API request asked Unity to start a task immediately."""

    topic: ClassVar[str | None] = "app:comms:task_trigger"

    task_id: int
    source_task_log_id: int | None = None
    destination: str | None = None
    source_ref: str = ""
    task_label: str = ""
    task_summary: str = ""
    reason: str = ""

    @classmethod
    def from_dict(
        cls,
        payload: Any,
        *,
        reason: str = "",
    ) -> "TaskTriggerRequested | None":
        """Build a REST task-trigger event from a dict-shaped payload."""

        if not isinstance(payload, _Mapping):
            return None
        task_id = _coerce_int(payload.get("task_id"))
        if task_id is None:
            return None
        try:
            destination = ContextRegistry.canonical_destination(
                payload.get("destination"),
            )
        except ValueError:
            return None
        task_label = str(payload.get("task_label") or "")
        resolved_reason = reason or (
            f"Task '{task_label}' was triggered via REST API."
            if task_label
            else f"Task {task_id} was triggered via REST API."
        )
        return cls(
            task_id=task_id,
            source_task_log_id=_coerce_int(payload.get("source_task_log_id")),
            destination=destination,
            source_ref=str(payload.get("source_ref") or ""),
            task_label=task_label,
            task_summary=str(payload.get("task_summary") or ""),
            reason=resolved_reason,
        )


@dataclass
class InactivityFollowup(Event):
    """Orchestra signalled that the user has been silent across all of
    their assistants for ``settings.inactivity_followup_days`` (orchestra
    side) and this Coordinator should compose a re-engagement message to
    the boss.

    Communication publishes this either as a ``unity_system_event`` to a
    hot pod's Pub/Sub topic or, on a cold start, as an entry in
    ``StartupEvent.wake_reasons``. The event itself carries no extra
    fields — the brain decides the variant (never-spoke vs spoke-before)
    by inspecting transcript history when the handler runs.
    """

    topic: ClassVar[str | None] = "app:comms:inactivity_followup"

    reason: str = ""


@dataclass
class AssistantPresenceObserved(Event):
    """Console observed the user viewing or interacting with this assistant.

    The adapters wake the runtime before publishing this event. Unity treats it
    as presence-only context: no LLM run is requested and no user-visible reply
    is composed.
    """

    topic: ClassVar[str | None] = "app:comms:assistant_presence_observed"
    loggable: ClassVar[bool] = False

    reason: str = ""
    source: str = ""
    page_visibility: str = ""
    occurred_at: str = ""


@dataclass
class CoordinatorDelegate(Event):
    """A Coordinator assigned asynchronous work to this colleague.

    Communication publishes this either as a ``unity_system_event`` to a hot
    pod or as a startup wake reason during cold start. The colleague's brain is
    responsible for carrying out the instruction through its own manager
    primitives; the Coordinator only receives dispatch-level confirmation.
    """

    topic: ClassVar[str | None] = "app:comms:coordinator_delegate"

    requested_by_assistant_id: str
    instruction: str
    intent: str = "general"
    dedupe_key: str | None = None
    related_context: dict[str, Any] | None = None
    reason: str = ""


@dataclass
class CoordinatorOnboardingEvent(Event):
    """Orchestra observed a user action that should be narrated in the
    Coordinator's onboarding conversation.

    Emitted only while the Coordinator is in
    ``Coordinator/State.onboarding_active`` (see
    ``coordinator_onboarding_event_service`` in orchestra) so day-to-day
    activity stays silent. Subtypes correspond to onboarding milestones
    with no other user-visible feedback channel: workspace OAuth landed,
    an integration secret was saved, or the user opened an onboarding
    session from the call-or-chat picker. The brain reacts with a
    one-line acknowledgement that names the thing that just happened
    and previews the next pending step — see the coordinator block in
    ``prompt_builders.build_system_prompt``.

    ``subtype`` is the canonical event taxonomy keyed off
    ``extra_event_fields.subtype`` on the Pub/Sub payload; ``details``
    carries optional structured context (secret name, specialist id,
    …) the brain can include in the acknowledgement when relevant.
    """

    topic: ClassVar[str | None] = "app:comms:coordinator_onboarding_event"

    subtype: str = ""
    message: str = ""
    details: dict = field(default_factory=dict)


@dataclass
class IntegrationToolsSyncRequested(Event):
    """A provider app needs its FunctionManager tools materialized or cleaned up."""

    topic: ClassVar[str | None] = "app:comms:integration_tools_sync_requested"

    app_slug: str
    app_display_name: str | None = None
    connection_id: str | None = None
    operation: str = "materialize"
    message: str = ""


@dataclass
class IntegrationToolsSyncCompleted(Event):
    """Unity finished preparing active provider tools for an app."""

    topic: ClassVar[str | None] = "app:comms:integration_tools_sync_completed"

    app_slug: str
    app_display_name: str | None = None
    connection_id: str | None = None
    tool_count: int | None = None
    operation: str = "materialize"
    message: str = ""


@dataclass
class IntegrationToolsSyncFailed(Event):
    """Unity failed to prepare active provider tools for an app."""

    topic: ClassVar[str | None] = "app:comms:integration_tools_sync_failed"

    app_slug: str
    app_display_name: str | None = None
    connection_id: str | None = None
    error: str = ""
    operation: str = "materialize"
    message: str = ""


@dataclass
class FileSyncComplete(Event):
    """The initial rclone bisync between the container and the managed VM has
    finished.  All files from the assistant's persistent disk are now available
    on the local filesystem."""

    topic: ClassVar[str | None] = "app:comms:file_sync_complete"


# --------------------------------------------------------------------------- #
# Meet Interaction Events (screen share / remote control)
# --------------------------------------------------------------------------- #


@dataclass
class AssistantScreenShareStarted(Event):
    """User enabled assistant screen sharing during a Unify Meet session.

    The assistant's desktop is now visible to the user.
    """

    topic: ClassVar[str | None] = "app:comms:assistant_screen_share_started"

    reason: str = ""


@dataclass
class AssistantScreenShareStopped(Event):
    """User disabled assistant screen sharing during a Unify Meet session.

    The assistant's desktop is no longer visible to the user.
    """

    topic: ClassVar[str | None] = "app:comms:assistant_screen_share_stopped"

    reason: str = ""


@dataclass
class UserScreenShareStarted(Event):
    """User started sharing their screen during a Unify Meet session.

    The user's screen is now being streamed to the assistant.
    """

    topic: ClassVar[str | None] = "app:comms:user_screen_share_started"

    reason: str = ""


@dataclass
class UserScreenShareStopped(Event):
    """User stopped sharing their screen during a Unify Meet session."""

    topic: ClassVar[str | None] = "app:comms:user_screen_share_stopped"

    reason: str = ""


@dataclass
class UserWebcamStarted(Event):
    """User enabled their webcam during a Unify Meet session.

    The user's webcam feed is now being streamed to the assistant.
    """

    topic: ClassVar[str | None] = "app:comms:user_webcam_started"

    reason: str = ""


@dataclass
class UserWebcamStopped(Event):
    """User disabled their webcam during a Unify Meet session."""

    topic: ClassVar[str | None] = "app:comms:user_webcam_stopped"

    reason: str = ""


@dataclass
class UserRemoteControlStarted(Event):
    """User took remote control of the assistant's desktop.

    The user now has mouse and keyboard control. The actor should pause
    computer-related execution to avoid conflicting with user input.
    """

    topic: ClassVar[str | None] = "app:comms:user_remote_control_started"

    reason: str = ""


@dataclass
class UserRemoteControlStopped(Event):
    """User released remote control of the assistant's desktop.

    The actor may resume computer-related execution.
    """

    topic: ClassVar[str | None] = "app:comms:user_remote_control_stopped"

    reason: str = ""


@dataclass
class UserFilesysAccessStarted(Event):
    """User (re-)enabled this assistant's access to their home filesystem.

    Carries ``user_id`` (the desktop owner) because one assistant may have
    several users' desktops linked; the live grant targets that user's link.
    """

    topic: ClassVar[str | None] = "app:comms:user_filesys_access_started"

    user_id: str = ""
    reason: str = ""


@dataclass
class UserFilesysAccessStopped(Event):
    """User revoked this assistant's access to their home filesystem.

    In-flight actors must stop reading from / writing back to that machine.
    Carries ``user_id`` (the desktop owner) so the revoke targets the right
    link when multiple users' desktops are linked.
    """

    topic: ClassVar[str | None] = "app:comms:user_filesys_access_stopped"

    user_id: str = ""
    reason: str = ""


@dataclass
class SyncContacts(Event):
    """Signal to re-sync system contacts from the API (assistant, user, org members)."""

    reason: str = ""


@dataclass
class BackupContactsEvent(Event):
    """
    Fallback contacts from inbound messages for use before ContactManager initializes.

    When an inbound message arrives before the ContactManager is ready, this event
    carries the contacts list so they can be cached locally in ContactIndex. Once
    ContactManager is initialized, this local cache is cleared and all contact
    lookups go through ContactManager.
    """

    loggable: ClassVar[bool] = False
    contacts: list[dict[str, Any]]


@dataclass
class LLMUserMessage(Event):
    content: str


@dataclass
class LLMAssistantMessage(Event):
    content: str


@dataclass
class SummarizeContext(Event):
    pass


@dataclass
class DirectMessageEvent(Event):
    """
    Send a message directly to the user via the current medium,
    bypassing the Main CM Brain's decision-making.

    Used by ConversationManagerHandle.ask for questions and acknowledgments
    that should be delivered verbatim without LLM processing.
    """

    content: str
    source: str = "system"
