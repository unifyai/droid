import asyncio
import contextlib
import json
import traceback
from datetime import datetime
from typing import Any, Callable, Optional, Reversible

from unify.logger import LOGGER
from unify.common.hierarchical_logger import DEFAULT_ICON
from unify.common.startup_timing import log_startup_timing
from unify.common.diagnostic_logging import staging_diagnostics_enabled
from unify.session_details import SESSION_DETAILS
from unify.coordinator_voice import resolve_runtime_voice
from unify.settings import SETTINGS
from unify.manager_registry import SingletonABCMeta
from unify.common.async_tool_loop import SteerableToolHandle
from unify.common.hierarchical_logger import SessionLogger
from unify.conversation_manager import assistant_jobs
from unify.conversation_manager.domains.call_manager import (
    CallConfig,
    LivekitCallManager,
)
from unify.conversation_manager.domains.contact_index import (
    ContactIndex,
    CommsMessage,
    Message,
)
from unify.conversation_manager.domains.brain import build_brain_spec
from unify.conversation_manager.domains.brain_action_tools import (
    ConversationManagerBrainActionTools,
)
from unify.conversation_manager.domains.brain_tools import ConversationManagerBrainTools
from unify.conversation_manager.domains.comms_utils import publish_system_error
from unify.conversation_manager.domains.event_handlers import EventHandler
from unify.conversation_manager.domains.renderer import Renderer
from unify.conversation_manager.events import *
from unify.integrations.sync_state import IntegrationSyncCoordinator
from unify.common.prompt_helpers import now as prompt_now

from unify.common.llm_client import new_llm_client
from unify.common.single_shot import single_shot_tool_decision
from unify.events.manager_event_logging import _EVENT_SOURCE
from unify.conversation_manager.domains.notifications import NotificationBar
from unify.conversation_manager.domains.utils import Debouncer, log_task_exc

from unify.memory_manager.memory_manager import MemoryManager
from unify.contact_manager.contact_manager import ContactManager
from unify.transcript_manager.transcript_manager import TranscriptManager
from unify.conversation_manager.cm_types import Medium, Mode, ScreenshotEntry
from unify.conversation_manager.cm_types.screenshot import (
    generate_screenshot_path,
    write_screenshot_to_disk,
)
from unify.actor.base import BaseActor
from unify.conversation_manager.domains.proactive_speech import ProactiveSpeech
from unify.conversation_manager.medium_scripts.common import FastBrainLogger
from unify.spending_limits import check_credit_gate_state

MAX_CONV_MANAGER_MSGS = 50
# Upper bound a deferred hang-up waits for its explanatory line to be spoken
# before tearing down anyway (guards a line that never surfaces). Generous so it
# rarely cuts off a legitimately-playing line.
_HANG_UP_SPEECH_TIMEOUT_S = 20.0
IDLE_SMALLTALK_RECENT_COMMS_SECONDS = 20.0
RECENT_TOOL_EXECUTIONS_LIMIT = 20
RECENT_TOOL_PREVIEW_CHARS = 500
CREDIT_GATE_REPLY_THROTTLE_SECONDS = 300
ONBOARDING_OUTBOUND_CONTEXT_TTL_SECONDS = 120
DEPLETED_CREDITS_SLOW_BRAIN_RESPONSE = (
    "Your credits are depleted, so I can't continue helping with setup or tasks "
    "until you top up. Please add credits in billing, then I'll pick this back up."
)
DEPLETED_CREDITS_EMAIL_SUBJECT = "Credits depleted"
COMMISSIONING_MUTATION_TOOL_NAMES = frozenset(
    {
        "act",
    },
)
COMMISSIONING_OUTBOUND_FOLLOWUP_EVENTS = frozenset(
    {
        "SMSSent",
        "WhatsAppMessageSent",
        "EmailSent",
        "UnifyMessageSent",
        "ApiMessageSent",
        "DiscordMessageSent",
        "DiscordChannelMessageSent",
        "TeamsMessageSent",
        "TeamsChannelMessageSent",
    },
)


def _idle_status_smalltalk_allowed(
    *,
    in_flight_actions: dict[int, dict],
    global_thread: Reversible[CommsMessage],
    inflight_voice_speech: str,
    now: datetime,
    recent_comms_seconds: float = IDLE_SMALLTALK_RECENT_COMMS_SECONDS,
) -> bool:
    if in_flight_actions:
        return False
    if inflight_voice_speech.strip():
        return False
    for message in reversed(global_thread):
        if not isinstance(message, CommsMessage):
            continue
        if getattr(message, "role", None) != "assistant":
            continue
        age_seconds = (now - message.timestamp).total_seconds()
        return age_seconds >= recent_comms_seconds
    return True


ACT_FOLLOWUP_ARGUMENT_DEFAULTS: dict[str, Any] = {
    "response_format": None,
    "persist": False,
    "include_conversation_context": True,
}


def _log_slow_brain_single_shot_failure(
    *,
    run_id: str,
    request_id: str,
    origin_event_name: str,
    message_count: int,
    tool_count: int,
    state_chars: int,
) -> None:
    if not staging_diagnostics_enabled():
        return
    LOGGER.exception(
        (
            "Slow-brain single-shot failed "
            "run_id=%s request_id=%s origin_event=%s "
            "message_count=%d tool_count=%d state_chars=%d"
        ),
        run_id,
        request_id or "-",
        origin_event_name or "-",
        message_count,
        tool_count,
        state_chars,
    )
    LOGGER.error(
        "Slow-brain single-shot traceback text:\n%s",
        traceback.format_exc(),
    )


def _append_context_to_state_message(message: dict, context: str) -> dict:
    if not context:
        return message
    updated = dict(message)
    content = updated.get("content")
    if isinstance(content, str):
        updated["content"] = f"{content}\n\n{context}"
        return updated
    if isinstance(content, list):
        updated["content"] = [
            *content,
            {"type": "text", "text": f"\n\n{context}"},
        ]
        return updated
    updated["content"] = f"{content or ''}\n\n{context}"
    return updated


def _render_action_context(
    in_flight_actions: dict,
    completed_actions: dict,
    notifications: list | None = None,
) -> str | None:
    """Build an action-status summary with recent progress for proactive speech."""
    lines: list[str] = []
    for handle_data in in_flight_actions.values():
        query = handle_data.get("query", "unknown")
        action_type = handle_data.get("action_type", "act")
        lines.append(f"- EXECUTING ({action_type}): {query}")
        for entry in handle_data.get("handle_actions", [])[-5:]:
            name = entry.get("action_name", "")
            if name in ("act_started", "desktop_act_started", "web_act_started"):
                continue
            detail = str(entry.get("query", ""))[:200]
            ts = entry.get("timestamp", "")
            lines.append(f"    [{ts}] {name}: {detail}")
    for handle_data in completed_actions.values():
        query = handle_data.get("query", "unknown")
        lines.append(f"- COMPLETED: {query}")
    if notifications:
        recent = [n for n in notifications[-5:] if n.type not in ("Meet",)]
        if recent:
            lines.append("")
            lines.append("Recent system events:")
            for n in recent:
                lines.append(f"  - [{n.type}] {n.content[:150]}")
    if not lines:
        return None
    header = "[action status] Current actions and recent progress:"
    return f"{header}\n" + "\n".join(lines)


class ConversationManager(metaclass=SingletonABCMeta):
    def __init__(
        self,
        event_broker,
        job_name: str,
        user_id: str,
        assistant_id: int | None,
        user_first_name: str,
        user_surname: str,
        assistant_first_name: str,
        assistant_surname: str,
        assistant_age: str,
        assistant_nationality: str,
        assistant_about: str,
        assistant_number: str,
        assistant_email: str,
        user_number: str,
        user_email: str = None,
        voice_provider: str = "cartesia",
        voice_id: str = None,
        assistant_timezone: str = "",
        assistant_whatsapp_number: str = "",
        assistant_discord_bot_id: str = "",
        assistant_email_provider: str = "",
        assistant_slack_bot_user_id: str = "",
        assistant_is_coordinator: bool = False,
        assistant_job_title: str = "",
        past_events: list | None = None,
        conv_context_length: int = 50,
        project_name: str = "Assistants",
        stop: asyncio.Event = None,
    ):
        # assistant details
        self.job_name = job_name
        self.user_id = user_id
        self.assistant_id = assistant_id
        self.assistant_first_name = assistant_first_name
        self.assistant_surname = assistant_surname
        self.assistant_age = assistant_age
        self.assistant_nationality = assistant_nationality
        self.assistant_timezone = assistant_timezone
        self.assistant_about = assistant_about
        self.assistant_job_title = assistant_job_title
        self.voice_provider = voice_provider
        self.voice_id = voice_id

        # contact data
        self.assistant_number = assistant_number
        self.assistant_email = assistant_email
        self.assistant_whatsapp_number = assistant_whatsapp_number
        self.assistant_discord_bot_id = assistant_discord_bot_id
        self.assistant_slack_bot_user_id = assistant_slack_bot_user_id
        self.is_coordinator = assistant_is_coordinator
        # Global onboarding scaffolding gate, mirrored from Orchestra's
        # ``Coordinator/State`` and refreshed on a short TTL. When False the
        # slow-brain drops all onboarding scaffolding. Defaults to True until
        # the first refresh resolves.
        self.coordinator_onboarding_active: bool = True
        # Precomputed depends_on-aware onboarding picture (steps + statuses
        # + valid next targets with nudge copy), mirrored from Orchestra so
        # the slow brain reads a standing progress block instead of
        # deriving "what's next". None outside active onboarding.
        self.coordinator_onboarding_render: dict[str, Any] | None = None
        # Trigger-step ids the user clicked in THIS session (ephemeral by
        # design): unlocks the matching reference-quiz comms tool until the
        # send durably completes the step. Lost on restart on purpose — the
        # row stays re-clickable, so a tool can never be permanently masked.
        self._onboarding_clicked_trigger_steps: set[str] = set()
        # Static, deployment-gated onboarding catalog (phases + steps + copy),
        # fetched once from Orchestra's canonical source of truth and reused for
        # every prompt build so console_ui never re-declares onboarding copy.
        self.onboarding_catalog: dict[str, Any] | None = None
        self._coordinator_state_checked_at: float = 0.0
        self.assistant_email_provider = assistant_email_provider
        self.user_first_name = user_first_name
        self.user_surname = user_surname
        self.user_number = user_number
        self.user_email = user_email

        # initialization state
        self.initialized: bool = False
        self.ready_for_brain: bool = True
        self.vm_ready: bool = False
        self.file_sync_complete: bool = False
        self.deployment_runtime_reconcile_status: Any | None = None
        # logging
        self.loop = asyncio.get_event_loop()
        self.project_name = project_name

        # inactivity & shutdown
        self.inactivity_timeout = 420  # 7 minutes in seconds
        self.inactivity_check_interval = 30  # seconds
        self.last_activity_time = self.loop.time()
        self.shutdown_reason: str | None = None
        self.stop = stop

        self.event_broker = event_broker

        # managers
        self.transcript_manager: TranscriptManager = None
        self.contact_manager: ContactManager = None
        self.memory_manager: MemoryManager = None
        self.actor: BaseActor | None = None

        self.debouncer = Debouncer(name="ConversationManager")

        # call manager - pass event_broker for socket IPC with voice agent subprocess
        self.call_manager = LivekitCallManager(self.get_call_config(), event_broker)
        self.call_manager.set_config_provider(self.get_call_config)
        self.call_manager.on_screenshot = self._buffer_screenshot
        self.call_manager.on_fast_brain_generating = self._on_fast_brain_generating
        self.call_manager.on_pipeline_quiescent = self._on_pipeline_quiescent

        # renderer
        self.prompt_renderer = Renderer()

        # state - TODO: put the state into a dict or state class
        # access is as a property with a lock, that is locked when an llm run
        # such that you can never modify state while the LLM is running (so actions do not break)
        self.mode: Mode = Mode.TEXT
        self.chat_history = []
        # Line this turn just decided to speak, not yet confirmed spoken. Injected
        # render-only into the next run's prompt (as a transient [You] row) so it
        # is not repeated; cleared once the real Outbound utterance lands. Never
        # written to the stored transcript.
        self._inflight_voice_speech: str = ""
        # Deferred session teardown: the hang_up tool records intent here rather
        # than ending the call immediately, so _run_llm can wait for the turn's
        # explanatory line to be spoken before tearing down (no mid-sentence cut).
        self._pending_hang_up: bool = False
        self._pending_hang_up_teardown: Callable | None = None
        # Set when an outbound voice utterance matching the just-published spoken
        # guidance lands (full line, or a barge-in truncated prefix). Used to gate
        # the deferred hang-up on speech actually being delivered.
        self._inflight_speech_delivered: asyncio.Event = asyncio.Event()
        # Call-session id of an in-flight Unify Meet ring awaiting an answer.
        # Cleared when the owner answers (UnifyMeetReceived) or the no-answer
        # timeout fires and falls the conversation back to text.
        self._pending_meet_ring: str | None = None
        self.contact_index = ContactIndex()
        self.notifications_bar = NotificationBar()
        self.integration_sync_coordinator = IntegrationSyncCoordinator()
        self.in_flight_actions: dict[
            int,
            dict,
        ] = (
            {}
        )  # dict[int, {"handle": "SteerableTool", "query": "str", "handle_actions": []}]
        self.completed_actions: dict[
            int,
            dict,
        ] = {}  # Finished actions, kept for post-completion ask() queries
        self._pending_steering_tasks: set[asyncio.Task] = (
            set()
        )  # Background tasks from async steering ops (e.g., ask_*)
        self.last_snapshot = prompt_now(as_string=False)
        self._current_snapshot = None
        self._current_state_snapshot = (
            None  # Fresh rendered state for tools during _run_llm
        )
        self._current_snapshot_state = (
            None  # SnapshotState with element tracking for incremental diff computation
        )

        # meet interaction state (screen share / webcam / remote control)
        self.assistant_screen_share_active: bool = False
        self.user_screen_share_active: bool = False
        self.user_webcam_active: bool = False
        self.user_remote_control_active: bool = False

        # screenshot buffer for slow brain visual context
        self._screenshot_buffer: list[ScreenshotEntry] = []
        # mapping from local_message_id (ephemeral CM counter) to
        # global message_id (persistent backend TM id), populated by
        # log_message() for post-hoc screenshot image updates.
        self._local_to_global_message_ids: dict[int, int] = {}
        # Per-destination transcript message ids for fanout writes.
        self._local_to_global_message_ids_by_destination: dict[
            int,
            dict[str | None, int],
        ] = {}
        # Primary destination used when one id is needed for compatibility paths.
        self._local_message_destinations: dict[int, str | None] = {}

        # mapping from conference_name/room_name to exchange_id, populated
        # at call/meet end so the async RecordingReady handler can resolve
        # the exchange without a database filter query.
        self._recording_exchange_ids: dict[str, int] = {}

        # proactive speech
        self.proactive_speech = ProactiveSpeech(
            model=SETTINGS.conversation.PROACTIVE_SPEECH_MODEL,
        )
        self._proactive_speech_task: asyncio.Task | None = None
        self._proactive_speech_gen: int = 0
        self._proactive_speech_enabled: bool = True
        self._voice_pipeline_quiescent = asyncio.Event()
        self._voice_pipeline_quiescent.set()
        self._proactive_logger = FastBrainLogger("ProactiveSpeech")

        # ask handles (for Actor actions)
        self.active_ask_handle: Optional["SteerableToolHandle"] = None

        # LLM run requests recorded during event handling (production path).
        # In step() mode, requests are recorded via a contextvar instead.
        self._pending_llm_requests: list[tuple[float, bool, bool]] = []
        self._pending_llm_request_meta: list[dict[str, Any]] = []
        self._current_event_trace: dict[str, str] | None = None
        self._event_trace_seq: int = 0
        self._llm_request_seq: int = 0
        self._llm_run_seq: int = 0
        self._llm_gen: int = 0
        self._outbound_suppress_gen: int = -1
        self._active_llm_trace_meta: dict[str, Any] | None = None
        self._credit_gate_reply_sent_at: dict[tuple[str, str], float] = {}
        self._recent_tool_executions: list[dict[str, Any]] = []
        self._recent_commissioning_successes: dict[str, int] = {}

        # WhatsApp messages that were sent via greeting template (outside 24h
        # window). When the contact replies, the brain is notified so it can
        # resend or rework the original message.  Maps contact_id → content.
        self._pending_whatsapp_resends: dict[int, str] = {}
        self._pending_whatsapp_resend_onboarding_metadata: dict[int, dict[str, str]] = (
            {}
        )

        # Best-effort estimate of whether each contact's 24-hour WhatsApp
        # free-form window is currently open, so the brain's send_whatsapp
        # docstring can warn it up front when an out-of-window send will only
        # deliver a generic template placeholder (not the verbatim body).
        # Maps contact_id → bool (absent = unknown). Seeded best-effort at
        # startup (Orchestra owns the authoritative window) and refreshed from
        # observed traffic: an inbound opens it, a templated outbound proves it
        # was closed, a free-form outbound proves it was open.
        self._whatsapp_window_open: dict[int, bool] = {}

        # Outbound WhatsApp call contexts stashed while awaiting call permission.
        # When the contact grants permission (taps "Call now"), the context is
        # injected as call_manager.initial_notification.  Maps contact_id → context.
        self._pending_whatsapp_call_contexts: dict[int, str] = {}
        self._pending_onboarding_outbound: dict[str, Any] | None = None
        self._startup_wake_reasons: list[dict[str, Any]] = []

        # Hierarchical session logger for consistent nested logging
        self._session_logger = SessionLogger("ConversationManager")
        self._session_logger.debug(
            "session_start",
            "ConversationManager session initialized",
        )

    def fast_brain_idle_smalltalk_allowed(self) -> bool:
        return _idle_status_smalltalk_allowed(
            in_flight_actions=self.in_flight_actions,
            global_thread=self.contact_index.global_thread,
            inflight_voice_speech=self._inflight_voice_speech,
            now=prompt_now(as_string=False),
        )

    def snapshot(self):
        self._current_snapshot = prompt_now(as_string=False)
        # Track how many notifications were present at snapshot time.
        # Any notifications appended while the LLM is running (e.g., an action that
        # completes very quickly) must remain visible for at least the NEXT LLM run.
        # Otherwise, `commit()` would immediately drop them and the LLM would never
        # see the result, which can cause repeated duplicate actions.
        self._snapshot_notif_count = len(self.notifications_bar.notifications)
        return self._current_snapshot

    def commit(self):
        self.last_snapshot = self._current_snapshot
        notifs = self.notifications_bar.notifications
        snap_n = int(getattr(self, "_snapshot_notif_count", 0) or 0)
        # Keep:
        # - pinned notifications
        # - notifications that were appended AFTER the last snapshot was taken
        #   (these arrived during the LLM run and must be shown next turn)
        self.notifications_bar.notifications = [
            n for i, n in enumerate(notifs) if n.pinned or i >= snap_n
        ]

    @staticmethod
    def _tool_result_is_error(result: Any) -> bool:
        return isinstance(result, dict) and "error_kind" in result

    @staticmethod
    def _preview_value(
        value: Any,
        *,
        max_chars: int = RECENT_TOOL_PREVIEW_CHARS,
    ) -> str:
        try:
            rendered = json.dumps(value, sort_keys=True, default=str)
        except Exception:
            rendered = repr(value)
        if len(rendered) <= max_chars:
            return rendered
        return rendered[: max_chars - 3] + "..."

    @staticmethod
    def _normalize_followup_tool_args(
        tool_name: str,
        tool_args: dict[str, Any] | None,
    ) -> dict[str, Any]:
        normalized = dict(tool_args or {})
        if tool_name == "act":
            for key, default_value in ACT_FOLLOWUP_ARGUMENT_DEFAULTS.items():
                normalized.setdefault(key, default_value)
        return normalized

    @classmethod
    def _commissioning_tool_fingerprint(
        cls,
        tool_name: str,
        tool_args: dict[str, Any] | None,
    ) -> str:
        stable_args = json.dumps(
            cls._normalize_followup_tool_args(tool_name, tool_args),
            sort_keys=True,
            default=str,
        )
        return f"{tool_name}:{stable_args}"

    def _is_immediate_commissioning_followup(self, origin_event_name: str) -> bool:
        return origin_event_name in COMMISSIONING_OUTBOUND_FOLLOWUP_EVENTS

    def suppress_duplicate_commissioning_tool(
        self,
        *,
        tool_name: str,
        tool_args: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        """Suppress immediate duplicate commissioning calls from outbound follow-ups."""
        if tool_name not in COMMISSIONING_MUTATION_TOOL_NAMES:
            return None
        trace_meta = self._active_llm_trace_meta or {}
        origin_event_name = str(trace_meta.get("origin_event_name") or "")
        if not self._is_immediate_commissioning_followup(origin_event_name):
            return None
        fingerprint = self._commissioning_tool_fingerprint(tool_name, tool_args)
        last_success_gen = self._recent_commissioning_successes.get(fingerprint)
        if last_success_gen != self._llm_gen - 1:
            return None
        return {
            "error_kind": "duplicate_suppressed",
            "message": (
                "Skipped duplicate commissioning tool call from immediate outbound "
                "follow-up event."
            ),
            "details": {
                "tool_name": tool_name,
                "origin_event_name": origin_event_name,
            },
        }

    def _record_recent_tool_executions(
        self,
        *,
        tools: list[Any],
        trace_meta: dict[str, Any],
    ) -> None:
        origin_event_name = str(trace_meta.get("origin_event_name") or "")
        for tool_exec in tools:
            tool_name = str(getattr(tool_exec, "name", ""))
            tool_args = getattr(tool_exec, "args", {}) or {}
            tool_result = getattr(tool_exec, "result", None)
            self._recent_tool_executions.append(
                {
                    "generation": self._llm_gen,
                    "origin_event_name": origin_event_name,
                    "tool_name": tool_name,
                    "args_preview": self._preview_value(tool_args),
                    "result_preview": self._preview_value(tool_result),
                },
            )
            if (
                tool_name in COMMISSIONING_MUTATION_TOOL_NAMES
                and not self._tool_result_is_error(tool_result)
            ):
                fingerprint = self._commissioning_tool_fingerprint(tool_name, tool_args)
                self._recent_commissioning_successes[fingerprint] = self._llm_gen
        if len(self._recent_tool_executions) > RECENT_TOOL_EXECUTIONS_LIMIT:
            self._recent_tool_executions = self._recent_tool_executions[
                -RECENT_TOOL_EXECUTIONS_LIMIT:
            ]
        for fingerprint, generation in list(
            self._recent_commissioning_successes.items(),
        ):
            if generation < self._llm_gen - 1:
                del self._recent_commissioning_successes[fingerprint]

    @property
    def assistant_has_teams(self) -> bool:
        """True when Microsoft Teams capabilities are available to this assistant.

        Derived from the email provider rather than a dedicated flag because
        Teams access is gated by the same MS365 OAuth grant that backs the
        assistant's email — provisioning the Teams scopes without an MS365
        mailbox is not a supported configuration. Update both `assistant_has_teams`
        and `_assistant_has_teams` in `unify.comms.primitives` together if a
        first-class Teams flag is ever introduced.
        """
        return self.assistant_email_provider == "microsoft_365"

    @property
    def in_voice_session(self) -> bool:
        """True when a voice call or meeting of any kind is live (or joining).

        A single predicate spanning every voice surface — phone calls, WhatsApp
        calls, Unify Meet, and browser meetings (Google Meet / Microsoft Teams).
        Only one such session can exist at a time, so the call-starting tools are
        withheld whenever this is True. This is the single source of truth shared
        by the tool set (`as_tools`) and the system prompt so the two can never
        disagree on what is available mid-call.
        """
        call_state = self.call_manager
        return (
            self.mode.is_voice
            or call_state.has_active_call
            or call_state.has_active_google_meet
            or call_state.has_active_teams_meet
            or call_state._whatsapp_call_joining
        )

    @property
    def session_logger(self) -> SessionLogger:
        """The hierarchical session logger for this ConversationManager instance."""
        return self._session_logger

    @property
    def computer_primitives(self):
        """Lazily resolve the ``ComputerPrimitives`` singleton via ManagerRegistry."""
        from unify.function_manager.primitives.runtime import ComputerPrimitives
        from unify.manager_registry import ManagerRegistry

        return ManagerRegistry.get_instance(ComputerPrimitives)

    @property
    def computer_fast_path_eligible(self) -> bool:
        """True when the CM should expose computer fast-path tools.

        Requires assistant screen share to be active.  The tools are available
        regardless of whether an in-flight ``act`` session has already invoked
        computer primitives — the prompt guides the LLM to spin up a concurrent
        ``act(persist=True)`` session when one isn't already running.
        """
        return self.assistant_screen_share_active

    def get_active_contact(self) -> dict | None:
        """Get the contact for the current active call, or fall back to the boss contact."""
        return self.call_manager.call_contact or self.contact_index.get_contact(
            contact_id=SESSION_DETAILS.boss_contact_id,
        )

    async def capture_assistant_screenshot(
        self,
        user_utterance: str,
        local_message_id: int | None = None,
        *,
        cached: bool = False,
    ) -> None:
        """Capture the assistant's screen and buffer it for the next slow brain turn.

        When *cached* is True, reads from the agent-service screenshot cache
        (~0 ms) instead of doing a live Playwright capture (~500 ms).  The
        cache is updated automatically after every Magnitude action.

        Runs the HTTP call in a thread to avoid event loop starvation — the
        main process event loop is shared with the actor and managers, which
        can saturate it during heavy async work.
        """
        import asyncio
        import time as _time
        from datetime import datetime, timezone

        import requests as _requests

        from unify.conversation_manager.medium_scripts.common import (
            _resolve_agent_service_url,
            _ensure_jpeg,
        )

        base_url = _resolve_agent_service_url()
        if cached:
            url = f"{base_url}/screenshot/latest"
        else:
            url = f"{base_url}/screenshot"
        auth_key = SESSION_DETAILS.unify_key

        def _sync_capture() -> dict | None:
            t0 = _time.monotonic()
            try:
                resp = _requests.post(
                    url,
                    json={},
                    headers={"authorization": f"Bearer {auth_key}"},
                    timeout=10,
                )
                total_ms = (_time.monotonic() - t0) * 1000
                if resp.status_code >= 400:
                    self._session_logger.warning(
                        "screenshot_capture",
                        f"Screenshot capture failed: HTTP {resp.status_code} "
                        f"url={url} total={total_ms:.0f}ms "
                        f"body={resp.text[:200]}",
                    )
                    return None
                data = resp.json()
                self._session_logger.debug(
                    "screenshot_capture",
                    f"Screenshot capture OK: url={url} "
                    f"total={total_ms:.0f}ms "
                    f"b64_len={len(data.get('screenshot', ''))}",
                )
                return data
            except Exception as e:
                total_ms = (_time.monotonic() - t0) * 1000
                self._session_logger.warning(
                    "screenshot_capture",
                    f"Screenshot capture error: {type(e).__name__}: {e} "
                    f"url={url} total={total_ms:.0f}ms",
                )
                return None

        data = await asyncio.to_thread(_sync_capture)
        if data and self.assistant_screen_share_active:
            b64 = data.get("screenshot")
            if b64:
                b64 = _ensure_jpeg(b64)
                self._screenshot_buffer.append(
                    ScreenshotEntry(
                        b64,
                        user_utterance,
                        datetime.now(timezone.utc),
                        "assistant",
                        local_message_id,
                    ),
                )

    def peek_screenshot_buffer(self) -> list[ScreenshotEntry]:
        """Return a snapshot of buffered screenshots without clearing.

        The buffer remains intact so that if the consuming operation
        (e.g. an LLM turn) is cancelled before completion, the next
        attempt will re-process the same screenshots.  Call
        :meth:`commit_screenshot_buffer` after all side effects have
        succeeded to remove the consumed entries.
        """
        return list(self._screenshot_buffer)

    def commit_screenshot_buffer(self, count: int) -> None:
        """Remove the first *count* entries from the screenshot buffer.

        Called after the LLM turn has successfully consumed and persisted
        the screenshots returned by :meth:`peek_screenshot_buffer`.
        Any screenshots that arrived *during* the turn (appended after the
        peek) are preserved for the next turn.
        """
        del self._screenshot_buffer[:count]

    async def _register_screenshots_background(
        self,
        screenshots: list[ScreenshotEntry],
        screenshot_paths: list[str],
    ) -> None:
        """Persist screenshots to disk and register with ImageManager / TM.

        Runs as a fire-and-forget background task after a successful LLM turn.
        None of these operations affect the LLM prompt or decision — they are
        purely persistence bookkeeping (disk write, image storage, transcript
        annotation).
        """
        source_labels = {"assistant": "Assistant's screen", "user": "User's screen"}

        # 0. Write screenshots to disk (entries with filepath already set by
        #    the fast brain are skipped — write_screenshot_to_disk is a no-op
        #    for those).
        for entry, path in zip(screenshots, screenshot_paths):
            if not entry.filepath:
                write_screenshot_to_disk(entry, path)

        # 1. Register with ImageManager to get persistent image_ids.
        image_ids: list[int] = []
        image_ids_by_destination: dict[str | None, list[int]] = {}
        implicit_destinations: list[str | None] = [None]
        try:
            from unify.manager_registry import ManagerRegistry
            from unify.common.context_registry import ContextRegistry

            image_manager = ManagerRegistry.get_image_manager()
            items = [
                {
                    "data": entry.b64,
                    "timestamp": entry.timestamp,
                    "filepath": path,
                }
                for entry, path in zip(screenshots, screenshot_paths)
            ]
            implicit_destinations = ContextRegistry.implicit_shared_destinations()
            for destination in implicit_destinations:
                destination_image_ids = await asyncio.to_thread(
                    image_manager.add_images,
                    items,
                    synchronous=True,
                    destination=destination,
                )
                image_ids_by_destination[destination] = destination_image_ids
            primary_destination = implicit_destinations[0]
            image_ids = image_ids_by_destination.get(primary_destination, [])
        except Exception as e:
            self._session_logger.warning(
                "screenshot_registration",
                f"ImageManager registration failed, skipping: {e}",
            )
            return

        # 2. Annotate CM Message objects with image_ids and build TM refs.
        msg_to_image_refs: dict[tuple[int, str | None], list[dict]] = {}
        for i, (entry, _path) in enumerate(zip(screenshots, screenshot_paths)):
            if entry.local_message_id is None or i >= len(image_ids):
                continue
            mid = entry.local_message_id
            img_id = image_ids[i]

            # Attach image_id to the Message object.
            for gte in self.contact_index.global_thread:
                msg = gte.message
                if isinstance(msg, Message) and msg.local_message_id == mid:
                    if not hasattr(msg, "image_ids") or msg.image_ids is None:
                        msg.image_ids = []
                    msg.image_ids.append(img_id)
                    break

            label = source_labels.get(entry.source, "Screenshot")
            for destination, destination_image_ids in image_ids_by_destination.items():
                if i >= len(destination_image_ids):
                    continue
                msg_to_image_refs.setdefault((mid, destination), []).append(
                    {
                        "raw_image_ref": {"image_id": destination_image_ids[i]},
                        "annotation": f"{label} -- '{entry.utterance}'",
                    },
                )

        # 3. Post-hoc update TM messages with AnnotatedImageRefs.
        if msg_to_image_refs and self.transcript_manager is not None:
            for (local_mid, destination), refs in msg_to_image_refs.items():
                destination_map = self._local_to_global_message_ids_by_destination.get(
                    local_mid,
                    {},
                )
                tm_msg_id = destination_map.get(destination)
                effective_destination = destination
                if tm_msg_id is None:
                    if destination is not None:
                        self._session_logger.warning(
                            "screenshot_tm_update",
                            (
                                "Skipping screenshot transcript update for "
                                f"local_mid={local_mid}, destination={destination!r}: "
                                "destination message mapping missing."
                            ),
                        )
                        continue
                    tm_msg_id = self._local_to_global_message_ids.get(local_mid)
                    effective_destination = self._local_message_destinations.get(
                        local_mid,
                        destination,
                    )
                if tm_msg_id is not None:
                    try:
                        await asyncio.to_thread(
                            self.transcript_manager.update_message_images,
                            tm_msg_id,
                            refs,
                            destination=effective_destination,
                        )
                    except Exception as e:
                        self._session_logger.warning(
                            "screenshot_tm_update",
                            f"TM image update failed for msg {tm_msg_id}: {e}",
                        )

    def _claim_pending_user_screenshot(self, local_message_id: int) -> None:
        """Stamp the most recent unclaimed user screenshot with the given local_message_id."""
        if self._screenshot_buffer:
            last = self._screenshot_buffer[-1]
            if last.source == "user" and last.local_message_id is None:
                self._screenshot_buffer[-1] = last._replace(
                    local_message_id=local_message_id,
                )

    def _buffer_screenshot(self, event_json: str) -> None:
        """Buffer a screenshot received from the fast brain via IPC.

        Accepts both user and assistant screenshots, distinguished by the
        ``source`` field in the JSON payload.  When a ``filepath`` is included,
        the file has already been written to disk by the fast brain.
        """
        import json as _json
        from datetime import datetime, timezone

        try:
            data = _json.loads(event_json)
            b64 = data.get("b64", "")
            utterance = data.get("utterance", "")
            source = data.get("source", "user")
            filepath = data.get("filepath")
            ts_str = data.get("timestamp")
            ts = (
                datetime.fromisoformat(ts_str) if ts_str else datetime.now(timezone.utc)
            )
            if b64:
                self._screenshot_buffer.append(
                    ScreenshotEntry(b64, utterance, ts, source, filepath=filepath),
                )
                self._session_logger.debug(
                    "screenshot_capture",
                    f"Buffered {source} screenshot #{len(self._screenshot_buffer)} "
                    f"for utterance: {utterance[:60]}...",
                )
        except Exception as e:
            self._session_logger.warning(
                "screenshot_capture",
                f"Error buffering screenshot: {e}",
            )

    def _active_voice_medium(self) -> Medium:
        """The Medium for the currently-active voice thread."""
        if self.call_manager.has_active_google_meet:
            return Medium.GOOGLE_MEET
        if self.call_manager.has_active_teams_meet:
            return Medium.TEAMS_MEET
        if self.mode == Mode.MEET:
            return Medium.UNIFY_MEET
        if self.call_manager._call_channel == "whatsapp_call":
            return Medium.WHATSAPP_CALL
        return Medium.PHONE_CALL

    def _stash_inflight_voice_speech(self, message: str) -> None:
        """Stash the line this turn just decided to speak, for a render-only overlay.

        This is the slow brain's in-flight-speech overlay. The next run may start
        before the real spoken ``[You]`` utterance is recorded; without seeing
        this line it would re-derive "was that actually spoken?" and repeat it.
        So we stash it here and inject it into the NEXT render as a transient
        ``[You @ ...]`` row (see ``_run_llm``) - indistinguishable from confirmed
        speech for that one call, so the model treats it as already said.

        Crucially this is NEVER written to the stored transcript: it is a
        one-shot, render-only mutation. Once the real utterance lands (the
        ``Outbound*Utterance`` event), this stash is cleared so future turns see
        only what was *actually* spoken (e.g. the truncated prefix after a
        barge-in, with the ``VoiceInterrupt`` note carrying the remainder).
        """
        self._inflight_voice_speech = (message or "").strip()

    def get_recent_voice_transcript(
        self,
        contact: dict | None = None,
        max_messages: int | None = None,
    ) -> tuple[list[dict], datetime | None]:
        """Extract recent voice transcript from the active conversation.

        Args:
            contact: Contact to get transcript for. Defaults to active contact.
            max_messages: Maximum number of messages to return. None for all.

        Returns:
            A tuple of (conversation_turns, last_message_timestamp) where:
            - conversation_turns: List of {"role": "user"|"assistant", "content": str}
            - last_message_timestamp: Timestamp of the last message, or None
        """
        conversation_turns: list[dict] = []
        last_message_timestamp: datetime | None = None

        if contact is None:
            contact = self.get_active_contact()

        if not contact:
            return conversation_turns, last_message_timestamp

        contact_id = contact.get("contact_id")
        conv_state = self.contact_index.get_conversation_state(contact_id)
        if not conv_state:
            return conversation_turns, last_message_timestamp

        voice_medium = self._active_voice_medium()
        voice_thread = self.contact_index.get_messages_for_contact(
            contact_id,
            voice_medium,
        )

        # Optionally limit to last N messages
        if max_messages is not None:
            voice_thread = voice_thread[-max_messages:]

        for msg in voice_thread:
            role = "assistant" if msg.name == "You" else "user"
            content = (msg.content or "").strip()

            # Skip system messages (e.g., "<Call Started>")
            if content.startswith("<") and content.endswith(">"):
                continue

            conversation_turns.append({"role": role, "content": content})

            if hasattr(msg, "timestamp") and msg.timestamp:
                last_message_timestamp = msg.timestamp

        return conversation_turns, last_message_timestamp

    def get_recent_transcript(
        self,
        contact: dict | None = None,
        max_messages: int | None = None,
    ) -> tuple[list[dict], datetime | None]:
        """Extract recent transcript from ALL threads for a contact.

        Unlike get_recent_voice_transcript which only looks at the voice thread,
        this method uses the global_thread which contains messages from ALL mediums
        (sms, unify, voice, email).

        Args:
            contact: Contact to get transcript for. Defaults to active contact.
            max_messages: Maximum number of messages to return. None for all.

        Returns:
            A tuple of (conversation_turns, last_message_timestamp) where:
            - conversation_turns: List of {"role": "user"|"assistant", "content": str}
            - last_message_timestamp: Timestamp of the last message, or None
        """
        conversation_turns: list[dict] = []
        last_message_timestamp: datetime | None = None

        if contact is None:
            contact = self.get_active_contact()

        if not contact:
            return conversation_turns, last_message_timestamp

        contact_id = contact.get("contact_id")
        conv_state = self.contact_index.get_conversation_state(contact_id)
        if not conv_state:
            return conversation_turns, last_message_timestamp

        global_thread = self.contact_index.get_messages_for_contact(contact_id)

        # Optionally limit to last N messages
        if max_messages is not None:
            global_thread = global_thread[-max_messages:]

        for msg in global_thread:
            # Skip non-communication messages (e.g., GuidanceMessage for internal orchestration)
            if not isinstance(msg, CommsMessage):
                continue

            # Handle both Message and EmailMessage types
            if hasattr(msg, "content"):
                content = (msg.content or "").strip()
            elif hasattr(msg, "body"):
                content = (msg.body or "").strip()
            else:
                continue

            # Skip system messages (e.g., "<Call Started>")
            if content.startswith("<") and content.endswith(">"):
                continue

            conversation_turns.append({"role": msg.role, "content": content})

            if hasattr(msg, "timestamp") and msg.timestamp:
                last_message_timestamp = msg.timestamp

        return conversation_turns, last_message_timestamp

    def _preprocess_messages(
        self,
        messages: str | dict | list,
    ) -> str | dict | list:
        """Keep only the latest state snapshot from message history.

        ConversationManager renders a full state snapshot each turn. We keep only the
        latest snapshot when calling the model, while preserving any system messages
        and user interjections.
        """
        if isinstance(messages, str):
            return messages
        if isinstance(messages, dict):
            return messages
        if not isinstance(messages, list):
            return messages

        try:
            # Find all state snapshot messages
            state_indices = [
                i
                for i, m in enumerate(messages)
                if isinstance(m, dict) and m.get("_cm_state_snapshot") is True
            ]
            if not state_indices:
                return messages

            # Keep only the latest state snapshot and non-state messages
            last_state = messages[state_indices[-1]]
            kept: list[dict] = []
            for m in messages:
                if not isinstance(m, dict):
                    continue
                role = m.get("role")
                if role == "system":
                    kept.append(m)
                elif role == "user" and not m.get("_cm_state_snapshot"):
                    kept.append(m)

            kept.append(last_state)
            return kept
        except Exception:
            return messages

    _USER_ORIGIN_EVENTS = frozenset(
        {
            "InboundPhoneUtterance",
            "InboundUnifyMeetUtterance",
            "InboundWhatsAppCallUtterance",
        },
    )

    async def cancel_slow_brain_run(self, turn_id) -> None:
        """Cancel exactly the slow-brain run spawned by ``turn_id``.

        Invoked when the fast brain resolves that turn itself
        (``FastBrainContinued`` - resuming an interrupted line or fully answering
        a small-talk turn): the eagerly-started run for that turn would otherwise
        also answer. Targets only that turn's run wherever it sits in the queue
        (no-op if it was already debounced out), so a prior still-thinking run or
        an unrelated act/SMS run is never cancelled. A run already in tool commit
        (speaking) is spared.
        """
        await self.debouncer.cancel_run_by_turn(turn_id)

    async def interject_or_run(
        self,
        content: str,
        triggering_contact_id: int | None = None,
        turn_id: int | None = None,
    ):
        """Interject the ask handle or run the LLM"""
        prev_utterance = getattr(self, "_last_inbound_utterance", None)
        self._last_inbound_utterance = content

        if self.active_ask_handle and not self.active_ask_handle.done():
            await self.active_ask_handle.interject(content)
        else:
            if self.mode.is_voice:
                running_origin = self.debouncer.running_task_trace_meta.get(
                    "origin_event_name",
                    "",
                )
                running_is_known_non_user = (
                    running_origin != ""
                    and running_origin not in self._USER_ORIGIN_EVENTS
                )
                has_running = (
                    self.debouncer.running_task is not None
                    and not self.debouncer.running_task.done()
                )
                # Deterministically preempt non-user slow-brain runs.
                # Only cancel when the running task is *known* to be a
                # non-user event. Unknown origin (empty) defaults to the
                # safe queue-of-2 behavior.
                cancel_running = has_running and running_is_known_non_user
            else:
                # Text mode: rapid messages should get fresh responses.
                cancel_running = True

            await self.request_llm_run(
                delay=0,
                cancel_running=cancel_running,
                triggering_contact_id=triggering_contact_id,
                is_user_origin=True,
                turn_id=turn_id,
            )

            if (
                self.mode.is_voice
                and not cancel_running
                and SETTINGS.conversation.SPEECH_URGENCY_PREEMPT_ENABLED
                and self.debouncer.running_task
                and not self.debouncer.running_task.done()
            ):
                stale_task = self.debouncer.running_task
                asyncio.create_task(
                    self._evaluate_speech_urgency(
                        content,
                        stale_task,
                        prev_utterance,
                    ),
                )

    async def _evaluate_speech_urgency(
        self,
        utterance: str,
        stale_task: asyncio.Task,
        previous_utterance: str | None = None,
    ) -> None:
        """Concurrent sidecar: evaluate whether *utterance* should preempt the slow brain."""
        from unify.conversation_manager.domains.speech_urgency import (
            SpeechUrgencyEvaluator,
        )

        trace_meta = self.debouncer.running_task_trace_meta
        origin_event = trace_meta.get("origin_event_name", "unknown")
        elapsed = self.loop.time() - self.debouncer.running_task_started_at

        actions_parts = []
        for info in self.in_flight_actions.values():
            action_type = info.get("action_type", "unknown")
            query = info.get("query", "")
            actions_parts.append(f"{action_type}: {query!r}")
        actions_summary = "; ".join(actions_parts) if actions_parts else "none"

        evaluator = SpeechUrgencyEvaluator(
            model=SETTINGS.conversation.FAST_BRAIN_MODEL,
        )
        decision = await evaluator.evaluate(
            utterance=utterance,
            origin_event=origin_event,
            elapsed_seconds=elapsed,
            actions_summary=actions_summary,
            previous_utterance=previous_utterance,
        )

        self._session_logger.debug(
            "speech_urgency",
            (
                f"Urgency eval: urgent={decision.urgent} "
                f"utterance={utterance!r} origin={origin_event} "
                f"elapsed={elapsed:.1f}s reasoning={decision.reasoning!r}"
            ),
        )

        if not decision.urgent:
            return

        # Only cancel if the same stale task is still the running task.
        # If it completed (or a new task started) the utterance is already
        # being processed — cancelling would be counterproductive.
        if self.debouncer.running_task is stale_task and not stale_task.done():
            self._session_logger.debug(
                "speech_urgency",
                "Preempting stale slow-brain run — pending task will promote",
            )
            stale_task.cancel()

    # this is non-blocking, it will quickly submit the
    # coro and return
    async def run_llm(
        self,
        delay: float = 0,
        cancel_running: bool = False,
        trace_meta: dict[str, str] | None = None,
        is_user_origin: bool = False,
    ):
        await self.debouncer.submit(
            self._run_llm_with_failure_notification,
            kwargs={"trace_meta": trace_meta or {}},
            delay=delay,
            cancel_running=cancel_running,
            label=(trace_meta or {}).get("origin_event_name", ""),
            trace_meta=trace_meta,
            is_user_origin=is_user_origin,
        )

    # Grace window for the owner to answer a Unify Meet ring before the
    # conversation falls back to text.
    _MEET_RING_TIMEOUT_S = 25.0

    async def ring_unify_meet(self, context: str = "") -> dict:
        """Ring the owner on Unify Meet and await an answer (no-answer -> text).

        Publishes a ``unify_meet_incoming`` signal so the Console shows a pinned
        incoming-call window with an Answer button. The assistant cannot join the
        owner's browser for them; when they answer, Console's normal connect flow
        lands here as ``UnifyMeetReceived``. ``context`` becomes a briefed opener
        so the answered call opens purposefully. If unanswered within
        ``_MEET_RING_TIMEOUT_S``, a notification tells the brain to continue over
        text.
        """
        import uuid

        from unify.conversation_manager.domains import comms_utils

        call_session_id = f"meet-ring-{uuid.uuid4().hex[:12]}"
        reason = (context or "").strip() or (
            "Continuing our conversation on the live call."
        )
        self._pending_meet_ring = call_session_id
        result = await comms_utils.send_unify_meet_ring(
            call_session_id=call_session_id,
            reason=reason,
        )
        if not result.get("success"):
            self._pending_meet_ring = None
            return {
                "status": "error",
                "message": "Could not ring the Unify Meet right now.",
            }
        asyncio.ensure_future(self._await_meet_ring_answer(call_session_id))
        return {
            "status": "ok",
            "message": (
                "Ringing my boss on Unify Meet — a pinged call window with an "
                "Answer button is now showing for them. I'll join when they answer."
            ),
        }

    async def _await_meet_ring_answer(self, call_session_id: str) -> None:
        """Fall back to text if a Unify Meet ring goes unanswered."""
        await asyncio.sleep(self._MEET_RING_TIMEOUT_S)
        if self._pending_meet_ring != call_session_id:
            return  # answered (or superseded) - nothing to do
        from unify.common.prompt_helpers import now as prompt_now

        self._pending_meet_ring = None
        self.notifications_bar.push_notif(
            "Comms",
            (
                "My Unify Meet call went unanswered. Continue with the boss here "
                "over the current text channel instead - do not keep waiting on "
                "the call."
            ),
            prompt_now(as_string=False),
        )
        await self.run_llm(
            trace_meta={"origin_event_name": "unify_meet_ring_unanswered"},
        )

    @staticmethod
    def _is_transient_llm_error(exc: BaseException) -> bool:
        """True if ``exc`` is a provider-side transient error after unillm retries.

        unillm (``retry_transient_400_async``) already retries these internally
        with exponential backoff. If one escapes, it means the provider stayed
        unhealthy for the whole retry budget — e.g. Anthropic HTTP 529
        ``overloaded_error`` surfaces as ``litellm.InternalServerError``.
        """
        import litellm

        return isinstance(
            exc,
            (
                litellm.InternalServerError,
                litellm.ServiceUnavailableError,
                litellm.RateLimitError,
            ),
        )

    async def _notify_fast_brain_of_slow_brain_failure(
        self,
        exc: BaseException,
    ) -> None:
        """Surface a slow-brain exhaustion failure to the fast brain.

        Publishes a ``FastBrainNotification`` with ``should_speak=True`` and
        a dedicated ``spoken_message`` so the fast brain utters the error via
        TTS directly (bypassing its own LLM, which may be hitting the same
        provider outage). Also cancels any pending proactive-speech loop so
        it stops emitting "still looking" filler for a request the slow brain
        has given up on.
        """
        spoken_message = (
            "Sorry, I'm having trouble thinking right now — "
            "could you say that again in a moment?"
        )
        notification_message = (
            f"Slow-brain turn failed after retries were exhausted "
            f"({type(exc).__name__}). The user's last request was not processed. "
            "Acknowledge the error and ask them to try again; do NOT claim you "
            "are still working on the prior request."
        )
        contact = self.get_active_contact()
        event = FastBrainNotification(
            contact=contact or {},
            message=notification_message,
            spoken_message=spoken_message,
            should_speak=True,
            source="slow_brain_failure",
        )
        self._session_logger.info(
            "slow_brain_failure",
            (
                f"Notifying fast brain of slow-brain failure: "
                f"{type(exc).__name__}: {exc}"
            ),
        )
        event_json = event.to_json()
        await self.event_broker.publish("app:call:notification", event_json)
        await self.event_broker.publish(
            "app:comms:assistant_notification",
            event_json,
        )

        with contextlib.suppress(Exception):
            await self.cancel_proactive_speech()

    async def _publish_slow_brain_fast_brain_guidance(
        self,
        *,
        message: str,
        slow_brain_log_path: str = "",
        fast_brain_guidance: str = "",
    ) -> None:
        """Publish a slow-brain spoken line (``guide_voice_agent``) to the fast brain.

        ``guide_voice_agent`` is speak-only, so this always publishes a spoken
        line (``should_speak=True``); there is no silent-guidance path.
        ``fast_brain_guidance`` rides bundled with the spoken line (a short note
        the fast brain may use for a basic direct reply to the next message); it
        is always sent so an empty value clears any stale note.
        """
        if not message:
            return
        contact = self.get_active_contact()
        event = FastBrainNotification(
            contact=contact,
            message=message,
            should_speak=True,
            source="slow_brain",
            llm_log_path=slow_brain_log_path,
            fast_brain_guidance=fast_brain_guidance,
        )
        self._session_logger.info(
            "call_notification",
            f"Guide FastBrain (speak): {message}",
        )
        event_json = event.to_json()
        await self.event_broker.publish(
            "app:call:notification",
            event_json,
        )
        await self.event_broker.publish(
            "app:comms:assistant_notification",
            event_json,
        )

    async def _perform_deferred_hang_up(self, *, awaiting_speech: bool) -> None:
        """Run a hang-up the ``hang_up`` tool deferred, after speech is delivered.

        When the same turn produced spoken guidance, wait for that line to land
        (the matching outbound utterance sets ``_inflight_speech_delivered``; a
        barge-in's truncated prefix counts too) before tearing the session down,
        so the call never ends mid-utterance. A timeout guards a line that never
        surfaces. Standalone hang-ups (no spoken line) tear down immediately.
        """
        teardown = self._pending_hang_up_teardown
        self._pending_hang_up = False
        self._pending_hang_up_teardown = None
        if teardown is None:
            return
        if awaiting_speech:
            try:
                await asyncio.wait_for(
                    self._inflight_speech_delivered.wait(),
                    timeout=_HANG_UP_SPEECH_TIMEOUT_S,
                )
            except asyncio.TimeoutError:
                self._session_logger.info(
                    "call_notification",
                    "Deferred hang-up proceeding without spoken-line ack (timeout)",
                )
        await teardown()

    async def _run_llm_with_failure_notification(
        self,
        trace_meta: dict[str, str] | None = None,
    ) -> list[str] | None:
        """Wrap ``_run_llm`` so transient provider failures reach the user.

        Previously, a failed slow-brain turn only produced a ``log_task_exc``
        line in the logs — the user was left in silence while
        ``ProactiveSpeech`` continued to emit "still looking…" filler. This
        wrapper catches transient LLM errors, publishes a
        ``FastBrainNotification`` so the fast brain explicitly apologises and
        asks the user to retry, then re-raises so the existing failure log is
        preserved.
        """
        try:
            return await self._run_llm(trace_meta=trace_meta)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if self.mode.is_voice and self._is_transient_llm_error(exc):
                with contextlib.suppress(Exception):
                    await self._notify_fast_brain_of_slow_brain_failure(exc)
            raise

    def _credit_gate_throttle_key(
        self,
        reply_context: dict[str, Any],
    ) -> tuple[str, str]:
        medium = str(reply_context.get("medium") or "")
        target = (
            reply_context.get("api_message_id")
            or reply_context.get("channel_id")
            or reply_context.get("chat_id")
            or reply_context.get("contact_id")
            or ""
        )
        return (medium, str(target))

    def _credit_gate_reply_is_throttled(
        self,
        reply_context: dict[str, Any],
    ) -> bool:
        if reply_context.get("medium") == Medium.API_MESSAGE.value:
            return False

        throttle_key = self._credit_gate_throttle_key(reply_context)
        last_sent_at = self._credit_gate_reply_sent_at.get(throttle_key)
        now = self.loop.time()
        if (
            last_sent_at is not None
            and now - last_sent_at < CREDIT_GATE_REPLY_THROTTLE_SECONDS
        ):
            return True

        self._credit_gate_reply_sent_at[throttle_key] = now
        return False

    async def _send_credit_gate_reply(
        self,
        reply_context: dict[str, Any],
    ) -> bool:
        medium = reply_context.get("medium")
        contact_id = reply_context.get("contact_id")
        tools = ConversationManagerBrainActionTools(self)

        previous_suppress_gen = self._outbound_suppress_gen
        self._outbound_suppress_gen = self._llm_gen
        try:
            if medium == Medium.UNIFY_MESSAGE.value and contact_id is not None:
                await tools.send_unify_message(
                    contact_id=contact_id,
                    content=DEPLETED_CREDITS_SLOW_BRAIN_RESPONSE,
                )
            elif medium == Medium.SMS_MESSAGE.value and contact_id is not None:
                await tools.send_sms(
                    contact_id=contact_id,
                    content=DEPLETED_CREDITS_SLOW_BRAIN_RESPONSE,
                )
            elif medium == Medium.WHATSAPP_MESSAGE.value and contact_id is not None:
                await tools.send_whatsapp(
                    contact_id=contact_id,
                    content=DEPLETED_CREDITS_SLOW_BRAIN_RESPONSE,
                )
            elif medium == Medium.EMAIL.value:
                email_id = reply_context.get("email_id")
                thread_id = reply_context.get("thread_id")
                if email_id:
                    await tools.send_email(
                        subject=DEPLETED_CREDITS_EMAIL_SUBJECT,
                        body=DEPLETED_CREDITS_SLOW_BRAIN_RESPONSE,
                        reply_all=True,
                        email_id_to_reply_to=email_id,
                        thread_id=thread_id,
                    )
                elif contact_id is not None:
                    await tools.send_email(
                        to=[contact_id],
                        subject=DEPLETED_CREDITS_EMAIL_SUBJECT,
                        body=DEPLETED_CREDITS_SLOW_BRAIN_RESPONSE,
                    )
                else:
                    return False
            elif medium == Medium.API_MESSAGE.value:
                await tools.send_api_response(
                    contact_id=contact_id or SESSION_DETAILS.boss_contact_id,
                    content=DEPLETED_CREDITS_SLOW_BRAIN_RESPONSE,
                    tags=reply_context.get("tags"),
                )
            elif medium == Medium.DISCORD_MESSAGE.value and contact_id is not None:
                await tools.send_discord_message(
                    contact_id=contact_id,
                    content=DEPLETED_CREDITS_SLOW_BRAIN_RESPONSE,
                )
            elif medium == Medium.DISCORD_CHANNEL_MESSAGE.value and reply_context.get(
                "channel_id",
            ):
                await tools.send_discord_channel_message(
                    channel_id=reply_context["channel_id"],
                    guild_id=reply_context.get("guild_id") or "",
                    contact_id=contact_id,
                    content=DEPLETED_CREDITS_SLOW_BRAIN_RESPONSE,
                )
            elif medium == Medium.SLACK_MESSAGE.value and contact_id is not None:
                await tools.send_slack_message(
                    contact_id=contact_id,
                    content=DEPLETED_CREDITS_SLOW_BRAIN_RESPONSE,
                    team_id=reply_context.get("team_id") or "",
                    thread_ts=reply_context.get("thread_ts"),
                )
            elif medium == Medium.SLACK_CHANNEL_MESSAGE.value and reply_context.get(
                "channel_id",
            ):
                await tools.send_slack_channel_message(
                    channel_id=reply_context["channel_id"],
                    team_id=reply_context.get("team_id") or "",
                    thread_ts=reply_context.get("thread_ts"),
                    contact_id=contact_id,
                    content=DEPLETED_CREDITS_SLOW_BRAIN_RESPONSE,
                )
            elif medium == Medium.TEAMS_MESSAGE.value and contact_id is not None:
                await tools.send_teams_message(
                    contact_id=contact_id,
                    content=DEPLETED_CREDITS_SLOW_BRAIN_RESPONSE,
                    chat_id=reply_context.get("chat_id"),
                )
            elif medium == Medium.TEAMS_CHANNEL_MESSAGE.value and reply_context.get(
                "channel_id",
            ):
                await tools.send_teams_message(
                    contact_id=contact_id or SESSION_DETAILS.boss_contact_id,
                    content=DEPLETED_CREDITS_SLOW_BRAIN_RESPONSE,
                    channel_id=reply_context.get("channel_id"),
                    team_id=reply_context.get("team_id"),
                )
            else:
                return False
        finally:
            self._outbound_suppress_gen = previous_suppress_gen

        return True

    async def _maybe_handle_depleted_credit_gate(
        self,
        trace_meta: dict[str, Any],
    ) -> bool:
        reply_context = trace_meta.get("credit_gate_reply_context")
        if not reply_context:
            return False

        credit_gate_state = await check_credit_gate_state()
        if credit_gate_state.allowed:
            return False

        if self._credit_gate_reply_is_throttled(reply_context):
            self._session_logger.info(
                "credit_gate",
                "Skipped repeated depleted-credit reply",
            )
            return True

        sent = await self._send_credit_gate_reply(reply_context)
        self._session_logger.info(
            "credit_gate",
            (
                "Served depleted-credit reply"
                if sent
                else "Skipped depleted-credit reply without a deliverable channel"
            ),
        )
        return True

    async def request_llm_run(
        self,
        delay=0,
        cancel_running=False,
        triggering_contact_id: int | None = None,
        is_user_origin: bool = False,
        credit_gate_reply_context: dict[str, Any] | None = None,
        turn_id: int | None = None,
    ) -> str:
        """Request an LLM run.

        The request is recorded and later scheduled by the event loop after
        the current event is handled.
        """
        self._llm_request_seq += 1
        request_id = f"llmreq-{self._llm_request_seq:06d}"
        event_trace = self._current_event_trace or {}
        request_meta = {
            "request_id": request_id,
            "origin_event_id": event_trace.get("event_id", ""),
            "origin_event_name": event_trace.get("event_name", ""),
            "triggering_contact_id": triggering_contact_id,
            "is_user_origin": is_user_origin,
            # Carried onto the debouncer task so the fast brain can cancel exactly
            # this turn's run by id. ``None`` for non-voice / non-user triggers,
            # which must never be matched by a fast-brain cancel.
            "turn_id": turn_id,
        }
        if credit_gate_reply_context is not None:
            request_meta["credit_gate_reply_context"] = credit_gate_reply_context
        self._pending_llm_requests.append((delay, cancel_running, is_user_origin))
        self._pending_llm_request_meta.append(request_meta)
        log_startup_timing(
            LOGGER,
            (
                "⏱️ [StartupTiming] first_reply.request_llm_run queued "
                "request_id=%s origin_event=%s delay=%s cancel_running=%s "
                "is_user_origin=%s pending=%d ready_for_brain=%s"
            ),
            request_id,
            request_meta["origin_event_name"] or "-",
            delay,
            cancel_running,
            is_user_origin,
            len(self._pending_llm_requests),
            self.ready_for_brain,
        )
        self._session_logger.debug(
            "llm_queue",
            (
                f"Queued slow-brain run request_id={request_id} "
                f"origin_event_id={request_meta['origin_event_id'] or '-'} "
                f"origin_event={request_meta['origin_event_name'] or '-'} "
                f"delay={delay} cancel_running={cancel_running} "
                f"is_user_origin={is_user_origin}"
            ),
        )
        return request_id

    async def flush_llm_requests(self) -> None:
        """Schedule any pending LLM runs recorded during event handling."""
        if not self._pending_llm_requests:
            return
        if not self.ready_for_brain:
            return

        requests = self._pending_llm_requests
        metas = self._pending_llm_request_meta

        # Prefer the newest user-origin request; fall back to the newest overall.
        selected_idx = len(requests) - 1
        for i in range(len(requests) - 1, -1, -1):
            if requests[i][2]:  # is_user_origin
                selected_idx = i
                break

        dropped_requests = len(requests) - 1
        delay, cancel_running, is_user_origin = requests[selected_idx]
        selected_meta = dict(metas[selected_idx]) if metas else {}

        self._pending_llm_requests.clear()
        self._pending_llm_request_meta.clear()

        self._llm_run_seq += 1
        run_id = f"llmrun-{self._llm_run_seq:06d}"
        selected_meta["run_id"] = run_id
        selected_meta["dropped_requests"] = str(dropped_requests)
        log_startup_timing(
            LOGGER,
            (
                "⏱️ [StartupTiming] first_reply.flush_llm_requests dispatch "
                "run_id=%s request_id=%s origin_event=%s dropped=%d delay=%s "
                "cancel_running=%s is_user_origin=%s"
            ),
            run_id,
            selected_meta.get("request_id", "-"),
            selected_meta.get("origin_event_name", "-") or "-",
            dropped_requests,
            delay,
            cancel_running,
            is_user_origin,
        )

        self._session_logger.debug(
            "llm_thinking",
            (
                f"Dispatching slow-brain run_id={run_id} "
                f"request_id={selected_meta.get('request_id', '-')} "
                f"origin_event_id={selected_meta.get('origin_event_id', '-') or '-'} "
                f"origin_event={selected_meta.get('origin_event_name', '-') or '-'} "
                f"dropped_requests={dropped_requests} delay={delay} "
                f"cancel_running={cancel_running} is_user_origin={is_user_origin}"
            ),
        )
        if await self._maybe_handle_depleted_credit_gate(selected_meta):
            log_startup_timing(
                LOGGER,
                (
                    "⏱️ [StartupTiming] first_reply.credit_gate_blocked "
                    "run_id=%s request_id=%s origin_event=%s"
                ),
                run_id,
                selected_meta.get("request_id", "-") or "-",
                selected_meta.get("origin_event_name", "-") or "-",
            )
            return

        log_startup_timing(
            LOGGER,
            (
                "⏱️ [StartupTiming] first_reply.run_llm_submitted "
                "run_id=%s request_id=%s origin_event=%s was_queued=%s mode=%s"
            ),
            run_id,
            selected_meta.get("request_id", "-") or "-",
            selected_meta.get("origin_event_name", "-") or "-",
            self.debouncer.was_queued,
            self.mode,
        )
        await self.run_llm(
            delay=delay,
            cancel_running=cancel_running,
            trace_meta=selected_meta,
            is_user_origin=is_user_origin,
        )

    async def _run_llm(self, trace_meta: dict[str, str] | None = None) -> list[str]:
        """Run a single LLM decision and return all tool names that were called."""
        import time as _rl_time

        from datetime import datetime, timezone

        from ..events.cost_attribution import COST_ATTRIBUTION

        _preamble_t0 = _rl_time.perf_counter()
        _last_preamble_step = _preamble_t0

        def _ms_since_start() -> str:
            return f"{(_rl_time.perf_counter() - _preamble_t0) * 1000:.0f}ms"

        def _mark_preamble_step() -> float:
            nonlocal _last_preamble_step
            now = _rl_time.perf_counter()
            elapsed_ms = (now - _last_preamble_step) * 1000
            _last_preamble_step = now
            return elapsed_ms

        trace_meta = trace_meta or {}

        # Resolve per-turn org member attribution (only meaningful in org
        # context, where a cost can be attributed to a specific member).
        attributed_user_id = None
        if SESSION_DETAILS.org_id is not None:
            triggering_contact_id = trace_meta.get("triggering_contact_id")
            if triggering_contact_id is not None:
                contact = self.contact_index.get_contact(
                    contact_id=triggering_contact_id,
                )
                if contact and contact.get("is_system"):
                    attributed_user_id = contact.get("user_id")
            if attributed_user_id:
                COST_ATTRIBUTION.set([attributed_user_id])
            else:
                COST_ATTRIBUTION.set([SESSION_DETAILS.user.id])

        # The acting user for this turn: the inbound message sender when it maps
        # to a system user (boss or provisioned org member), else the workspace
        # owner. Drives per-user linked-desktop resolution in the prompt so a
        # shared assistant reflects the *speaker's* machine, not the owner's.
        acting_user_id = attributed_user_id or SESSION_DETAILS.user.id

        # Re-bind the billing context for THIS turn so credit deductions are
        # attributed to the assistant (and the acting member, in org context).
        # This must run for personal workspaces too: the context set once at
        # init does not reliably propagate to the generation execution
        # context, so without this LLM transactions are recorded with a NULL
        # assistant_id and disappear when filtering usage by assistant.
        try:
            import unillm

            is_voice = self.mode.is_voice
            unillm.set_billing_context(
                assistant_id=SESSION_DETAILS.assistant.agent_id,
                user_id=acting_user_id,
                organization_id=SESSION_DETAILS.org_id,
                source="call" if is_voice else "chat",
                label="Voice reply" if is_voice else "Chat reply",
            )
        except (ImportError, Exception):
            pass
        _cost_attribution_ms = _mark_preamble_step()

        self._llm_gen += 1
        run_id = trace_meta.get("run_id", "llmrun-unknown")
        request_id = trace_meta.get("request_id", "")
        origin_event_id = trace_meta.get("origin_event_id", "")
        origin_event_name = trace_meta.get("origin_event_name", "")
        self._session_logger.debug(
            "llm_thinking",
            (
                f"Slow-brain run started run_id={run_id} "
                f"request_id={request_id or '-'} "
                f"origin_event_id={origin_event_id or '-'} "
                f"origin_event={origin_event_name or '-'} "
                f"was_queued={self.debouncer.was_queued} mode={self.mode}"
            ),
        )
        _run_metadata_ms = _mark_preamble_step()

        slow_brain_start_time = datetime.now(timezone.utc)

        screenshots = self.peek_screenshot_buffer()
        _screenshot_peek_ms = _mark_preamble_step()

        screenshot_paths = [
            s.filepath or generate_screenshot_path(s) for s in screenshots
        ]
        _screenshot_paths_ms = _mark_preamble_step()

        if screenshots:
            msg_to_paths: dict[int, list[str]] = {}
            for entry, path in zip(screenshots, screenshot_paths):
                if entry.local_message_id is not None:
                    msg_to_paths.setdefault(entry.local_message_id, []).append(path)
            if msg_to_paths:
                for gte in self.contact_index.global_thread:
                    msg = gte.message
                    if (
                        isinstance(msg, Message)
                        and msg.local_message_id in msg_to_paths
                    ):
                        msg.screenshots = msg_to_paths.pop(msg.local_message_id)
                    if not msg_to_paths:
                        break
        _screenshot_attach_ms = _mark_preamble_step()

        self.snapshot()
        _snapshot_ms = _mark_preamble_step()

        web_sessions = None
        if self.assistant_screen_share_active:
            cp = self.computer_primitives
            if cp is not None:
                try:
                    web_sessions = await cp.web.list_sessions_with_metadata(
                        visible_only=True,
                        active_only=True,
                    )
                except Exception:
                    web_sessions = cp.web.list_sessions(
                        visible_only=True,
                        active_only=True,
                    )
        _web_sessions_ms = _mark_preamble_step()

        log_startup_timing(
            LOGGER,
            (
                "⏱️ [StartupTiming] llm_preamble.setup.detail "
                "run_id=%s total=%.0fms cost_attribution=%.0fms metadata=%.0fms "
                "screenshot_peek=%.0fms screenshot_paths=%.0fms "
                "screenshot_attach=%.0fms snapshot=%.0fms web_sessions=%.0fms "
                "screenshots=%d global_thread=%d chat_history=%d "
                "screen_share_active=%s"
            ),
            run_id,
            (_rl_time.perf_counter() - _preamble_t0) * 1000,
            _cost_attribution_ms,
            _run_metadata_ms,
            _screenshot_peek_ms,
            _screenshot_paths_ms,
            _screenshot_attach_ms,
            _snapshot_ms,
            _web_sessions_ms,
            len(screenshots),
            len(self.contact_index.global_thread),
            len(self.chat_history),
            self.assistant_screen_share_active,
        )

        # Render-only overlay: inject the line we just decided to speak (not yet
        # confirmed spoken) as a transient `[You]` row so this render treats it as
        # already said and never repeats it, then remove it immediately. It is
        # never persisted, so future turns see only the actually-spoken transcript.
        _inflight_entry = None
        if self._inflight_voice_speech and self.mode.is_voice:
            _inflight_contact = self.get_active_contact()
            if _inflight_contact:
                _inflight_entry = self.contact_index.build_message(
                    contact_id=_inflight_contact.get("contact_id"),
                    sender_name="You",
                    thread_name=self._active_voice_medium(),
                    message_content=self._inflight_voice_speech,
                    role="assistant",
                )
                self.contact_index.global_thread.append(_inflight_entry)

        _t0 = _rl_time.perf_counter()
        try:
            snapshot_state = self.prompt_renderer.render_state(
                self.contact_index,
                self.notifications_bar,
                self.in_flight_actions,
                self.completed_actions,
                self.last_snapshot,
                recent_tool_executions=self._recent_tool_executions,
                assistant_screen_share_active=self.assistant_screen_share_active,
                user_screen_share_active=self.user_screen_share_active,
                user_webcam_active=self.user_webcam_active,
                user_remote_control_active=self.user_remote_control_active,
                google_meet_active=self.call_manager.has_active_google_meet,
                teams_meet_active=self.call_manager.has_active_teams_meet,
                active_web_sessions=web_sessions,
                managers_initialized=self.initialized,
                vm_ready=self.vm_ready,
                file_sync_complete=self.file_sync_complete,
                has_desktop=SESSION_DETAILS.assistant.has_managed_desktop,
            )
        finally:
            # render_state is synchronous, so the transient row is always the
            # last entry here; remove it so it never persists.
            if _inflight_entry is not None:
                gt = self.contact_index.global_thread
                if gt and gt[-1] is _inflight_entry:
                    gt.pop()
                else:
                    with contextlib.suppress(ValueError):
                        gt.remove(_inflight_entry)
        _render_ms = (_rl_time.perf_counter() - _t0) * 1000

        # Mirror the Coordinator's onboarding state (defer switch + the
        # precomputed progress render) so the prompt builder reads a
        # standing "what's done / what's next" block and can drop
        # scaffolding when deferred. TTL-cached + event-refreshed, so this
        # is a no-op on most turns.
        await self._refresh_coordinator_onboarding_state()

        _t0 = _rl_time.perf_counter()
        brain_spec = build_brain_spec(
            self,
            snapshot_state=snapshot_state,
            screenshots=screenshots,
            screenshot_paths=screenshot_paths,
            acting_user_id=acting_user_id,
        )
        _brain_spec_ms = (_rl_time.perf_counter() - _t0) * 1000

        if screenshots:
            self._session_logger.debug(
                "screen_share",
                f"Attaching {len(screenshots)} screenshot(s) to slow brain turn",
            )
        _t0 = _rl_time.perf_counter()
        input_message = brain_spec.state_message()
        integration_sync_context = self.integration_sync_coordinator.prompt_summary()
        input_message = _append_context_to_state_message(
            input_message,
            integration_sync_context,
        )
        _state_message_ms = (_rl_time.perf_counter() - _t0) * 1000
        _t0 = _rl_time.perf_counter()
        system_prompt = brain_spec.system_prompt
        _system_prompt_ref_ms = (_rl_time.perf_counter() - _t0) * 1000

        self._current_state_snapshot = input_message

        self._current_snapshot_state = snapshot_state

        reason = (trace_meta or {}).get("origin_event_name", "")
        self._session_logger.debug(
            "llm_thinking",
            f"LLM thinking... ({reason})" if reason else "LLM thinking...",
        )

        response_model = brain_spec.response_model

        _t0 = _rl_time.perf_counter()
        _tools_step_t0 = _t0
        brain_tools = ConversationManagerBrainTools(self)
        _brain_tools_init_ms = (_rl_time.perf_counter() - _tools_step_t0) * 1000
        _tools_step_t0 = _rl_time.perf_counter()
        action_tools = ConversationManagerBrainActionTools(self)
        _action_tools_init_ms = (_rl_time.perf_counter() - _tools_step_t0) * 1000
        _tools_step_t0 = _rl_time.perf_counter()
        brain_tool_dict = brain_tools.as_tools()
        _brain_tools_ms = (_rl_time.perf_counter() - _tools_step_t0) * 1000
        _tools_step_t0 = _rl_time.perf_counter()
        action_tool_dict = action_tools.as_tools()
        _action_tools_ms = (_rl_time.perf_counter() - _tools_step_t0) * 1000
        _tools_step_t0 = _rl_time.perf_counter()
        steering_tool_dict = action_tools.build_action_steering_tools()
        _steering_tools_ms = (_rl_time.perf_counter() - _tools_step_t0) * 1000
        _tools_step_t0 = _rl_time.perf_counter()
        completed_tool_dict = action_tools.build_completed_action_tools()
        _completed_tools_ms = (_rl_time.perf_counter() - _tools_step_t0) * 1000
        _tools_step_t0 = _rl_time.perf_counter()
        tools = {
            **brain_tool_dict,
            **action_tool_dict,
            **steering_tool_dict,
            **completed_tool_dict,
        }
        _tools_merge_ms = (_rl_time.perf_counter() - _tools_step_t0) * 1000

        _tools_step_t0 = _rl_time.perf_counter()
        if self.computer_fast_path_eligible:
            tools["desktop_act"] = action_tools.desktop_act
            tools["web_act"] = action_tools.web_act
            tools["close_web_session"] = action_tools.close_web_session
        _fast_path_tools_ms = (_rl_time.perf_counter() - _tools_step_t0) * 1000
        _tools_ms = (_rl_time.perf_counter() - _t0) * 1000
        log_startup_timing(
            LOGGER,
            (
                "⏱️ [StartupTiming] llm_preamble.tools.detail "
                "run_id=%s total=%.0fms brain_init=%.0fms action_init=%.0fms "
                "brain_tools=%.0fms action_tools=%.0fms steering=%.0fms "
                "completed=%.0fms merge=%.0fms fast_path=%.0fms "
                "brain_tool_count=%d action_tool_count=%d steering_tool_count=%d "
                "completed_tool_count=%d total_tool_count=%d"
            ),
            run_id,
            _tools_ms,
            _brain_tools_init_ms,
            _action_tools_init_ms,
            _brain_tools_ms,
            _action_tools_ms,
            _steering_tools_ms,
            _completed_tools_ms,
            _tools_merge_ms,
            _fast_path_tools_ms,
            len(brain_tool_dict),
            len(action_tool_dict),
            len(steering_tool_dict),
            len(completed_tool_dict),
            len(tools),
        )

        _t0 = _rl_time.perf_counter()
        _client_step_t0 = _t0
        client = new_llm_client(
            SETTINGS.UNIFY_MODEL,
            origin="ConversationManager",
            # Slow brain stays at "high"; the system default is "max" (used by
            # the CodeActActor). On deepseek-v4 "max" buys marginal gains on the
            # hardest tasks at extra latency, which the live conversation loop
            # cannot afford.
            reasoning_effort="high",
        )
        _new_client_ms = (_rl_time.perf_counter() - _client_step_t0) * 1000
        _client_step_t0 = _rl_time.perf_counter()
        if hasattr(client, "_pending_thinking_log"):
            parts = [
                p
                for p in [reason, "from queue" if self.debouncer.was_queued else ""]
                if p
            ]
            suffix = f" ({', '.join(parts)})" if parts else ""
            client._pending_thinking_log.set_thinking_context(suffix)
        _thinking_context_ms = (_rl_time.perf_counter() - _client_step_t0) * 1000
        _client_step_t0 = _rl_time.perf_counter()
        client.set_system_message(system_prompt.to_list())
        _set_system_ms = (_rl_time.perf_counter() - _client_step_t0) * 1000
        _client_step_t0 = _rl_time.perf_counter()
        client.set_prompt_caching(["system"])
        _prompt_caching_ms = (_rl_time.perf_counter() - _client_step_t0) * 1000
        _client_step_t0 = _rl_time.perf_counter()
        messages = self._preprocess_messages(self.chat_history + [input_message])
        _preprocess_messages_ms = (_rl_time.perf_counter() - _client_step_t0) * 1000
        _client_ms = (_rl_time.perf_counter() - _t0) * 1000
        log_startup_timing(
            LOGGER,
            (
                "⏱️ [StartupTiming] llm_preamble.client.detail "
                "run_id=%s total=%.0fms new_client=%.0fms thinking_context=%.0fms "
                "set_system=%.0fms prompt_caching=%.0fms preprocess_messages=%.0fms "
                "state_message=%.0fms system_prompt_ref=%.0fms chat_history=%d "
                "message_count=%d system_parts=%d state_chars=%d"
            ),
            run_id,
            _client_ms,
            _new_client_ms,
            _thinking_context_ms,
            _set_system_ms,
            _prompt_caching_ms,
            _preprocess_messages_ms,
            _state_message_ms,
            _system_prompt_ref_ms,
            len(self.chat_history),
            len(messages),
            len(system_prompt.to_list()),
            len(brain_spec.state_prompt),
        )

        _source_token = _EVENT_SOURCE.set("ConversationManager")

        _rl_t0 = _rl_time.perf_counter()

        def _rl_ms() -> str:
            return f"{(_rl_time.perf_counter() - _rl_t0) * 1000:.0f}ms"

        self._session_logger.debug(
            "perf",
            (
                f"[_run_llm preamble={_ms_since_start()}] "
                f"render_state={_render_ms:.0f}ms brain_spec={_brain_spec_ms:.0f}ms "
                f"tools={_tools_ms:.0f}ms client={_client_ms:.0f}ms | "
                f"calling single_shot_tool_decision ({len(tools)} tools, {len(messages)} msgs)"
            ),
        )
        log_startup_timing(
            LOGGER,
            (
                "⏱️ [StartupTiming] first_reply.llm_preamble "
                "run_id=%s duration=%s render_state=%.0fms brain_spec=%.0fms "
                "tools=%.0fms client=%.0fms tool_count=%d message_count=%d"
            ),
            run_id,
            _ms_since_start(),
            _render_ms,
            _brain_spec_ms,
            _tools_ms,
            _client_ms,
            len(tools),
            len(messages),
        )
        self._active_llm_trace_meta = trace_meta

        try:
            try:
                result = await single_shot_tool_decision(
                    client,
                    messages,
                    tools,
                    tool_choice="required" if tools else "auto",
                    response_format=response_model,
                    exclusive_tools={
                        "make_call",
                        "make_whatsapp_call",
                        "join_google_meet",
                        "join_teams_meet",
                    },
                    on_tool_execution_start=lambda: self._mark_tool_commit_started(
                        trace_meta,
                        run_id,
                    ),
                )
            except Exception:
                _log_slow_brain_single_shot_failure(
                    run_id=run_id,
                    request_id=request_id,
                    origin_event_name=origin_event_name,
                    message_count=len(messages),
                    tool_count=len(tools),
                    state_chars=len(input_message),
                )
                raise
        finally:
            self._active_llm_trace_meta = None
            if hasattr(client, "_pending_thinking_log"):
                client._pending_thinking_log.emit_fallback()
            _EVENT_SOURCE.reset(_source_token)
        tool_names = [t.name for t in result.tools]
        self._session_logger.debug(
            "perf",
            f"[_run_llm +{_rl_ms()}] single_shot returned tools={tool_names}",
        )
        log_startup_timing(
            LOGGER,
            "⏱️ [StartupTiming] first_reply.single_shot duration=%s run_id=%s tools=%s",
            _rl_ms(),
            run_id,
            tool_names,
        )
        self._record_recent_tool_executions(
            tools=result.tools,
            trace_meta=trace_meta or {},
        )

        # Extract structured output (thoughts)
        structured = result.structured_output
        thoughts = ""
        if structured is not None:
            thoughts = getattr(structured, "thoughts", "")

        # Handle guide_voice_agent tool calls for voice modes. The slow brain
        # either SPEAKs (guide_voice_agent with a message, spoken verbatim by the
        # fast brain subprocess) or WAITs (omits the tool). It may bundle an
        # optional fast_brain_guidance note alongside a spoken message — never on
        # its own — which the fast brain may use for a basic direct reply to the
        # caller's next message.
        if self.mode.is_voice:
            guidance_message = ""
            fast_brain_guidance = ""
            for tool_exec in result.tools:
                if tool_exec.name == "guide_voice_agent":
                    args = tool_exec.args or {}
                    guidance_message = args.get("message", "")
                    fast_brain_guidance = args.get("fast_brain_guidance", "")
                    break

            # A pending hang-up (recorded by the hang_up tool this turn) must not
            # tear down the session until the spoken line has been delivered,
            # otherwise the call ends mid-sentence. Reset the delivered signal
            # before publishing so we only observe THIS turn's delivery.
            if self._pending_hang_up and guidance_message:
                self._inflight_speech_delivered.clear()

            if guidance_message:
                pending = getattr(client, "_pending_thinking_log", None)
                slow_brain_log_path = (
                    pending.last_path or "" if pending is not None else ""
                )
                # guide_voice_agent is speak-only: every call is spoken. Guidance
                # rides bundled with the spoken line (never alone); a spoken turn
                # without guidance clears any stale note on the fast brain.
                await self._publish_slow_brain_fast_brain_guidance(
                    message=guidance_message,
                    slow_brain_log_path=slow_brain_log_path,
                    fast_brain_guidance=fast_brain_guidance,
                )
                # Stash the spoken line for a render-only overlay so the next run
                # (which may start before the real `[You]` utterance is recorded)
                # sees what this turn just decided to say, treats it as already
                # said, and does not repeat it. Cleared once the real utterance
                # lands so future turns see only what was actually spoken.
                self._stash_inflight_voice_speech(guidance_message)

            # Perform any deferred hang-up only after the spoken line has actually
            # been delivered (or a barge-in truncated it), so the session never
            # ends mid-utterance.
            if self._pending_hang_up:
                await self._perform_deferred_hang_up(
                    awaiting_speech=bool(guidance_message),
                )

        self._session_logger.debug(
            "llm_response",
            (
                f"run_id={run_id} thoughts: {thoughts[:100]}..."
                if len(thoughts) > 100
                else f"run_id={run_id} thoughts: {thoughts}"
            )
            + (f" | actions: {tool_names}" if tool_names else ""),
        )

        self._session_logger.debug(
            "perf",
            f"[_run_llm +{_rl_ms()}] voice notification done, committing",
        )
        self.commit()
        log_startup_timing(
            LOGGER,
            "⏱️ [StartupTiming] first_reply.commit completed run_id=%s elapsed=%s",
            run_id,
            _rl_ms(),
        )
        self._session_logger.debug("state_update", "Committing state")

        # Clear the temporary state snapshots now that tools have executed
        self._current_state_snapshot = None
        self._current_snapshot_state = None

        # The turn completed successfully — commit the screenshot buffer so
        # these entries are not re-processed on the next turn.  Any new
        # screenshots that arrived during this turn (appended after the peek)
        # are preserved.
        if screenshots:
            self.commit_screenshot_buffer(len(screenshots))
            asyncio.create_task(
                self._register_screenshots_background(
                    screenshots,
                    screenshot_paths,
                ),
            )

        # Build assistant message for chat history
        assistant_content = (
            structured.model_dump_json() if structured else result.text_response or ""
        )
        self.chat_history.append(input_message)
        self.chat_history.append({"role": "assistant", "content": assistant_content})

        # If the LLM called wait(delay=N), schedule a delayed follow-up turn.
        for tool_exec in result.tools:
            if tool_exec.name == "wait":
                delay = (tool_exec.args or {}).get("delay")
                msg = (
                    f"Decided to wait {delay} seconds"
                    if delay is not None
                    else "Decided to wait"
                )
                self._session_logger.info("wait", msg)
                if delay is not None:
                    await self.run_llm(delay=delay)
                break

        self._session_logger.debug(
            "perf",
            f"[_run_llm +{_rl_ms()}] post-processing done",
        )
        log_startup_timing(
            LOGGER,
            "⏱️ [StartupTiming] first_reply.post_processing completed run_id=%s elapsed=%s",
            run_id,
            _rl_ms(),
        )
        self._session_logger.debug(
            "llm_response",
            (f"Slow-brain run completed run_id={run_id} " f"tools={tool_names or '-'}"),
        )

        return tool_names

    def _mark_tool_commit_started(
        self,
        trace_meta: dict[str, str] | None,
        run_id: str,
    ) -> None:
        if trace_meta is not None:
            trace_meta["tool_commit_started"] = "true"
        running_meta = getattr(self.debouncer, "running_task_trace_meta", None)
        if isinstance(running_meta, dict) and running_meta.get("run_id") == run_id:
            running_meta["tool_commit_started"] = "true"
        self._session_logger.debug(
            "llm_thinking",
            f"Slow-brain run entered tool commit run_id={run_id}",
        )

    async def wait_for_events(self):
        async with self.event_broker.pubsub() as pubsub:
            await pubsub.psubscribe(
                "app:comms:*",
                "app:actor:*",
                "app:logging:message_logged",
                "app:managers:output",
            )

            # Initialization is triggered by StartupEvent handler which
            # sets details before starting init. Do not duplicate here.

            while True:
                msg = await pubsub.get_message(
                    timeout=2,
                    ignore_subscribe_messages=True,
                )

                if not msg:
                    continue
                self.last_activity_time = self.loop.time()
                # process events
                event = Event.from_json(msg["data"])
                channel = msg.get("channel", "")
                if isinstance(channel, bytes):
                    channel = channel.decode("utf-8", errors="replace")
                self._event_trace_seq += 1
                event_id = f"evt-{self._event_trace_seq:06d}"
                event_name = event.__class__.__name__
                self._current_event_trace = {
                    "event_id": event_id,
                    "event_name": event_name,
                }
                if event.__class__.loggable:
                    self._session_logger.debug(
                        "event_trace",
                        (
                            f"Processing event_id={event_id} "
                            f"event={event_name} channel={channel or '-'}"
                        ),
                    )
                try:
                    _event_t0 = self.loop.time()
                    await EventHandler.handle_event(
                        event,
                        self,
                    )
                    log_startup_timing(
                        LOGGER,
                        "⏱️ [StartupTiming] event.handle_event duration=%.2fs event_id=%s event=%s channel=%s",
                        self.loop.time() - _event_t0,
                        event_id,
                        event_name,
                        channel or "-",
                    )
                    _flush_t0 = self.loop.time()
                    await self.flush_llm_requests()
                    log_startup_timing(
                        LOGGER,
                        "⏱️ [StartupTiming] event.flush_llm_requests duration=%.2fs event_id=%s event=%s",
                        self.loop.time() - _flush_t0,
                        event_id,
                        event_name,
                    )
                except Exception as exc:
                    LOGGER.error(
                        f"⚠️ [EventLoop] Unhandled error processing "
                        f"event_id={event_id} event={event_name} "
                        f"channel={channel or '-'}: {exc}",
                        exc_info=True,
                    )
                    publish_system_error(
                        "An unexpected error occurred. The assistant is "
                        "attempting to recover.",
                        error_type="recovering",
                    )
                finally:
                    self._current_event_trace = None

    async def check_inactivity(self):
        """Monitor for inactivity and shut down gracefully after timeout.

        Activity is detected from three sources:
        - External pubsub messages (updated by wait_for_events via last_activity_time)
        - Internal EventBus publishes (LLM calls, tool-loop turns, manager methods)
        - Active work records for long-running local execution

        Ghost-publish detection: if pubsub is idle past the timeout but
        eventbus_idle stays suspiciously low for many consecutive checks,
        something is periodically resetting last_publish_monotonic without real
        user-facing activity. After ``_GHOST_PUBLISH_CHECKS`` consecutive such
        observations we shut down to prevent indefinite hangs.
        """
        import time as _time

        from unify.events.active_work import ACTIVE_WORK
        from unify.events.event_bus import EventBus

        _GHOST_PUBLISH_CHECKS = 20  # 20 * 30s = 10 minutes
        ghost_counter = 0

        while True:
            await asyncio.sleep(self.inactivity_check_interval)
            current_time = self.loop.time()
            pubsub_idle = current_time - self.last_activity_time
            monotonic_now = _time.monotonic()
            eventbus_idle = monotonic_now - EventBus.last_publish_monotonic
            idle_seconds = min(pubsub_idle, eventbus_idle)
            active_work = ACTIVE_WORK.snapshot()
            has_active_work = active_work.active_count > 0
            effective_idle_seconds = 0.0 if has_active_work else idle_seconds

            if (
                not has_active_work
                and pubsub_idle > self.inactivity_timeout
                and eventbus_idle < self.inactivity_timeout
            ):
                ghost_counter += 1
            else:
                ghost_counter = 0

            ghost_publish = ghost_counter >= _GHOST_PUBLISH_CHECKS

            if int(current_time) % 180 < self.inactivity_check_interval:
                extra = ""
                if ghost_counter > 0:
                    extra = f" ghost_count={ghost_counter}/{_GHOST_PUBLISH_CHECKS}"
                if has_active_work:
                    active_heartbeat_age = (
                        monotonic_now - active_work.newest_heartbeat_at
                        if active_work.newest_heartbeat_at is not None
                        else 0.0
                    )
                    extra += (
                        f" active_work_count={active_work.active_count}"
                        f" active_elapsed={active_work.oldest_elapsed_s:.1f}s"
                        f" active_heartbeat_age={active_heartbeat_age:.1f}s"
                    )
                self._session_logger.info(
                    "inactivity_check",
                    f"Idle check: pubsub_idle={pubsub_idle:.1f}s, "
                    f"eventbus_idle={eventbus_idle:.1f}s, "
                    f"min_idle={idle_seconds:.1f}s, "
                    f"effective_idle={effective_idle_seconds:.1f}s, "
                    f"timeout={self.inactivity_timeout}s{extra}",
                )

            if effective_idle_seconds > self.inactivity_timeout or ghost_publish:
                if ghost_publish:
                    log_str = (
                        f"Ghost-publish shutdown: pubsub_idle={pubsub_idle:.0f}s "
                        f"but eventbus_idle stuck at {eventbus_idle:.1f}s "
                        f"for {ghost_counter} consecutive checks "
                        f"(timeout={self.inactivity_timeout}s)"
                    )
                else:
                    self.shutdown_reason = "idle_timeout"
                    log_str = f"Inactivity timeout reached ({self.inactivity_timeout}s), requesting shutdown"
                LOGGER.info(f"{DEFAULT_ICON} {log_str}")
                self._session_logger.info("session_end", log_str)
                self.stop.set()
                await self.event_broker.aclose()
                break  # Exit the loop after triggering shutdown

    def set_details(self, payload: dict):
        """Populate assistant/user/voice details into SESSION_DETAILS."""
        self.user_id = payload["user_id"]
        self.assistant_id = int(payload["assistant_id"])
        self.assistant_first_name = payload["assistant_first_name"]
        self.assistant_surname = payload["assistant_surname"]
        self.assistant_age = payload["assistant_age"]
        self.assistant_nationality = payload["assistant_nationality"]
        self.assistant_timezone = payload.get("assistant_timezone", "")
        self.assistant_about = payload["assistant_about"]
        self.assistant_job_title = payload.get("assistant_job_title", "")
        self.assistant_number = payload["assistant_number"]
        self.assistant_email = payload["assistant_email"]
        self.assistant_email_provider = payload.get(
            "assistant_email_provider",
            "google_workspace",
        )
        self.self_contact_id = int(payload["self_contact_id"])
        self.boss_contact_id = int(payload["boss_contact_id"])
        self.assistant_whatsapp_number = payload.get("assistant_whatsapp_number", "")
        self.assistant_discord_bot_id = payload.get("assistant_discord_bot_id", "")
        self.assistant_slack_bot_user_id = payload.get(
            "assistant_slack_bot_user_id",
            "",
        )
        self.is_coordinator = bool(payload.get("assistant_is_coordinator", False))
        self.user_first_name = payload["user_first_name"]
        self.user_surname = payload["user_surname"]
        self.user_number = payload["user_number"]
        self.user_email = payload["user_email"]
        self.user_whatsapp_number = payload.get("user_whatsapp_number", "")
        # Only adopt voice from the payload when it carries a real value. A
        # sparse AssistantUpdateEvent (voice omitted / coerced None -> "") must
        # not wipe the assistant's current voice back to the provider default.
        if payload.get("voice_provider"):
            self.voice_provider = payload["voice_provider"]
        if payload.get("voice_id"):
            self.voice_id = payload["voice_id"]
        self.binding_id = payload.get("binding_id", "")
        self.desktop_mode = payload.get("desktop_mode", "ubuntu")
        self.user_desktops = payload.get("user_desktops") or []
        self.org_id: int | None = payload.get("org_id")
        self.org_name: str = payload.get("org_name", "")
        self.team_ids: list[int] = payload.get("team_ids") or []
        team_summaries = payload.get("team_summaries") or []
        is_coordinator = payload.get("is_coordinator", False)
        # Set API key on SESSION_DETAILS for runtime access
        if payload.get("api_key"):
            SESSION_DETAILS.unify_key = payload["api_key"]
        # Populate the global SessionDetails singleton
        SESSION_DETAILS.populate(
            agent_id=self.assistant_id,
            assistant_first_name=self.assistant_first_name,
            assistant_surname=self.assistant_surname,
            assistant_age=self.assistant_age,
            assistant_nationality=self.assistant_nationality,
            assistant_timezone=self.assistant_timezone,
            assistant_about=self.assistant_about,
            assistant_job_title=self.assistant_job_title,
            assistant_number=self.assistant_number,
            assistant_email=self.assistant_email,
            assistant_email_provider=self.assistant_email_provider,
            assistant_self_contact_id=self.self_contact_id,
            assistant_whatsapp_number=self.assistant_whatsapp_number,
            assistant_discord_bot_id=self.assistant_discord_bot_id,
            assistant_slack_bot_user_id=self.assistant_slack_bot_user_id,
            assistant_is_coordinator=self.is_coordinator,
            user_id=self.user_id,
            user_first_name=self.user_first_name,
            user_surname=self.user_surname,
            user_number=self.user_number,
            user_email=self.user_email,
            user_whatsapp_number=self.user_whatsapp_number,
            user_boss_contact_id=self.boss_contact_id,
            org_id=self.org_id,
            org_name=self.org_name,
            team_ids=self.team_ids,
            team_summaries=team_summaries,
            voice_provider=self.voice_provider,
            voice_id=self.voice_id,
            binding_id=self.binding_id,
            desktop_mode=self.desktop_mode,
            user_desktops=self.user_desktops,
            is_coordinator=is_coordinator,
        )
        self.team_summaries = SESSION_DETAILS.team_summaries
        # Export to env vars for subprocess inheritance
        SESSION_DETAILS.export_to_env()

    def get_details(self) -> dict:
        return {
            "job_name": self.job_name,
            "user_id": self.user_id,
            "assistant_id": self.assistant_id,
            "user_first_name": self.user_first_name,
            "user_surname": self.user_surname,
            "assistant_first_name": self.assistant_first_name,
            "assistant_surname": self.assistant_surname,
            "user_number": self.user_number,
            "assistant_number": self.assistant_number,
            "user_email": self.user_email,
            "assistant_email": self.assistant_email,
        }

    def get_call_config(self) -> CallConfig:
        # Resolve the voice from the live runtime source rather than a frozen
        # snapshot: the CM's own fields take precedence, but fall back to
        # SESSION_DETAILS.voice (populated from the OS env at boot in self-host,
        # where no StartupEvent ever arrives). Without this fallback an empty
        # CM voice field silently sends the provider default to the call agent.
        voice_provider = (
            self.voice_provider or SESSION_DETAILS.voice.provider or "cartesia"
        )
        voice_id = self.voice_id or SESSION_DETAILS.voice.id or ""
        voice_provider, voice_id = resolve_runtime_voice(
            is_coordinator=SESSION_DETAILS.is_coordinator,
            voice_provider=voice_provider,
            voice_id=voice_id,
        )
        return CallConfig(
            assistant_id=self.assistant_id,
            user_id=self.user_id,
            assistant_bio=self.assistant_about,
            assistant_number=self.assistant_number,
            voice_provider=voice_provider,
            voice_id=voice_id,
            assistant_name=f"{self.assistant_first_name} {self.assistant_surname}".strip(),
            job_name=self.job_name,
            is_coordinator=SESSION_DETAILS.is_coordinator,
        )

    async def _refresh_coordinator_onboarding_state(self) -> None:
        """Best-effort refresh of the cached onboarding state for the brain.

        Mirrors two things from Orchestra's ``Coordinator/State`` onto the
        session so ``build_brain_spec`` never has to derive anything:
          - ``coordinator_onboarding_active``: whether onboarding scaffolding
            is live (narration, progress block, tool masking).
          - ``coordinator_onboarding_render``: the precomputed
            depends_on-aware picture (steps + statuses + valid next
            targets with nudge copy) that drives the standing progress
            block.

        TTL-cached so we don't pay an HTTP round-trip every turn; the
        render is also refreshed in real time from each onboarding event
        (see ``set_coordinator_onboarding_render``), so this is mostly a
        backstop. Non-coordinator sessions and Console-less deployments
        skip it. Failures leave the previous values in place.
        """
        if not self.is_coordinator or not SETTINGS.UNITY_CONSOLE_UI:
            return
        import time as _time

        now = _time.monotonic()
        # Refresh more eagerly while actively onboarding (a render is
        # present) so "what's next" stays fresh during a fast-moving
        # setup conversation; back off once onboarding is done/deferred.
        ttl = 10.0 if self.coordinator_onboarding_active else 30.0
        # Stamp before the await so concurrent runs don't stampede the
        # endpoint; a failed fetch still respects the TTL backoff.
        if now - self._coordinator_state_checked_at < ttl:
            return
        self._coordinator_state_checked_at = now
        agent_id = SESSION_DETAILS.assistant.agent_id
        if agent_id is None:
            return
        import httpx as _httpx

        try:
            async with _httpx.AsyncClient(timeout=3.0) as client:
                resp = await client.get(
                    f"{SETTINGS.ORCHESTRA_URL}/assistant/{agent_id}/state",
                    headers={"Authorization": f"Bearer {SESSION_DETAILS.unify_key}"},
                )
                resp.raise_for_status()
                info = (resp.json() or {}).get("info") or {}
                # The onboarding catalog is static + deployment-gated; fetch it
                # once and reuse it for every subsequent prompt build.
                if self.onboarding_catalog is None:
                    cat_resp = await client.get(
                        f"{SETTINGS.ORCHESTRA_URL}/assistant/onboarding/catalog",
                        headers={
                            "Authorization": f"Bearer {SESSION_DETAILS.unify_key}",
                        },
                    )
                    cat_resp.raise_for_status()
                    catalog = (cat_resp.json() or {}).get("info") or {}
                    self.onboarding_catalog = (
                        catalog if isinstance(catalog, dict) else None
                    )
            self._apply_coordinator_state_info(info)
        except Exception as exc:
            LOGGER.warning(
                "Coordinator onboarding-state refresh failed; "
                "keeping previous values (active=%s): %s",
                self.coordinator_onboarding_active,
                exc,
            )

    def _apply_coordinator_state_info(self, info: dict[str, Any]) -> None:
        """Mirror onboarding gate + render from a Coordinator/State snapshot."""
        self.coordinator_onboarding_active = bool(info.get("onboarding_active"))
        render = info.get("onboarding")
        self.coordinator_onboarding_render = (
            render if isinstance(render, dict) else None
        )

    async def _patch_coordinator_onboarding_active(
        self,
        *,
        active: bool,
        clear_onboarding_step: bool = False,
    ) -> dict[str, Any]:
        """PATCH ``onboarding_active`` on Orchestra and refresh the session cache."""
        from unify.settings import SETTINGS

        if not self.is_coordinator or not SETTINGS.UNITY_CONSOLE_UI:
            return {
                "status": "error",
                "message": "Onboarding can only be toggled for the workspace Coordinator.",
            }
        agent_id = SESSION_DETAILS.assistant.agent_id
        if agent_id is None:
            return {
                "status": "error",
                "message": "Coordinator agent id is missing.",
            }
        body: dict[str, Any] = {"onboarding_active": active}
        if clear_onboarding_step:
            body["clear_onboarding_step"] = True
        import httpx as _httpx
        import time as _time

        try:
            async with _httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.patch(
                    f"{SETTINGS.ORCHESTRA_URL}/assistant/{agent_id}/state",
                    headers={
                        "Authorization": f"Bearer {SESSION_DETAILS.unify_key}",
                    },
                    json=body,
                )
            if resp.status_code == 403:
                return {
                    "status": "error",
                    "message": (
                        "I do not have permission to change onboarding state "
                        "in this workspace."
                    ),
                }
            resp.raise_for_status()
            info = (resp.json() or {}).get("info") or {}
            if isinstance(info, dict):
                self._apply_coordinator_state_info(info)
            self._coordinator_state_checked_at = _time.monotonic()
            if active:
                message = (
                    "Onboarding is live again — the setup checklist and nudges "
                    "are back."
                )
            else:
                message = (
                    "Onboarding is paused — they can use the platform normally "
                    "and resume setup anytime from the Onboarding tab or by "
                    "asking me."
                )
            return {
                "status": "ok",
                "message": message,
                "onboarding_active": self.coordinator_onboarding_active,
            }
        except Exception as exc:
            LOGGER.warning(
                "Coordinator onboarding_active PATCH failed (active=%s): %s",
                active,
                exc,
            )
            return {
                "status": "error",
                "message": f"Failed to update onboarding state: {exc}",
            }

    async def _patch_coordinator_onboarding_step_state(
        self,
        *,
        step_id: str,
        completed: bool,
    ) -> dict[str, Any]:
        """PATCH manual onboarding step completion on Orchestra."""
        from unify.settings import SETTINGS

        if not self.is_coordinator or not SETTINGS.UNITY_CONSOLE_UI:
            return {
                "status": "error",
                "message": (
                    "Onboarding step completion can only be changed for the "
                    "workspace Coordinator."
                ),
            }
        agent_id = SESSION_DETAILS.assistant.agent_id
        if agent_id is None:
            return {
                "status": "error",
                "message": "Coordinator agent id is missing.",
            }
        import httpx as _httpx
        import time as _time

        try:
            async with _httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.patch(
                    f"{SETTINGS.ORCHESTRA_URL}/assistant/{agent_id}/state",
                    headers={
                        "Authorization": f"Bearer {SESSION_DETAILS.unify_key}",
                    },
                    json={
                        "onboarding_step_completion": {
                            "step_id": step_id,
                            "completed": completed,
                        },
                    },
                )
            payload = resp.json() if resp.content else {}
            if resp.status_code == 403:
                return {
                    "status": "error",
                    "message": (
                        "I do not have permission to change onboarding step "
                        "completion in this workspace."
                    ),
                }
            if resp.status_code == 400:
                detail = payload.get("detail")
                if isinstance(detail, dict):
                    message = detail.get("message") or detail.get("code")
                else:
                    message = detail
                return {
                    "status": "error",
                    "message": message or "Invalid onboarding step completion.",
                    "step_id": step_id,
                }
            resp.raise_for_status()
            info = payload.get("info") or {}
            if isinstance(info, dict):
                self._apply_coordinator_state_info(info)
            self._coordinator_state_checked_at = _time.monotonic()
            completed_step_ids = (
                info.get("completed_step_ids")
                if isinstance(info.get("completed_step_ids"), list)
                else []
            )
            if completed:
                message = f"Marked '{step_id}' complete in the onboarding checklist."
            elif step_id in completed_step_ids:
                message = (
                    f"Removed my manual completion for '{step_id}', but the "
                    "checklist still shows it as done because the platform "
                    "detects it is already complete."
                )
            else:
                message = f"Marked '{step_id}' incomplete in the onboarding checklist."
            return {
                "status": "ok",
                "message": message,
                "step_id": step_id,
                "completed": completed,
            }
        except Exception as exc:
            LOGGER.warning(
                "Coordinator onboarding step completion PATCH failed "
                "(step_id=%s, completed=%s): %s",
                step_id,
                completed,
                exc,
            )
            return {
                "status": "error",
                "message": f"Failed to update onboarding step completion: {exc}",
                "step_id": step_id,
            }

    def set_coordinator_onboarding_render(self, render: Any) -> None:
        """Update the cached onboarding render from a fresh event payload.

        Onboarding events carry the same ``onboarding`` rendering the
        state endpoint returns, so the standing progress block stays
        current between TTL fetches the instant an event lands.
        """
        if isinstance(render, dict):
            self.coordinator_onboarding_render = render

    @property
    def onboarding_clicked_trigger_steps(self) -> set[str]:
        """Trigger-step ids clicked in this session (reference-quiz gating)."""
        return self._onboarding_clicked_trigger_steps

    def record_onboarding_trigger_clicked(self, step_id: str) -> None:
        """Mark a reference-quiz trigger row as clicked this session.

        Unlocks the matching comms send tool until the send durably completes
        the step. No-op for blank ids.
        """
        if isinstance(step_id, str) and step_id.strip():
            self._onboarding_clicked_trigger_steps.add(step_id.strip())

    def clear_onboarding_clicked_trigger_steps(self) -> None:
        """Forget this session's clicked trigger rows (e.g. on onboarding reset)."""
        self._onboarding_clicked_trigger_steps.clear()

    def set_pending_onboarding_outbound(
        self,
        details: dict[str, Any],
        *,
        origin_event_id: str = "",
    ) -> None:
        trigger_step_id = details.get("trigger_step_id")
        channel = details.get("channel")
        tool_name = details.get("tool_name")
        if not isinstance(trigger_step_id, str) or not trigger_step_id.strip():
            return
        if not isinstance(channel, str) or not channel.strip():
            return
        self._pending_onboarding_outbound = {
            "onboarding_trigger_step_id": trigger_step_id.strip(),
            "onboarding_reply_step_id": (
                details.get("reply_step_id").strip()
                if isinstance(details.get("reply_step_id"), str)
                else ""
            ),
            "onboarding_request_id": "",
            "onboarding_origin_event_id": origin_event_id,
            "channel": channel.strip(),
            "tool_name": tool_name.strip() if isinstance(tool_name, str) else "",
            "expires_at": self.loop.time() + ONBOARDING_OUTBOUND_CONTEXT_TTL_SECONDS,
        }

    def set_pending_onboarding_request_id(self, request_id: str) -> None:
        if self._pending_onboarding_outbound:
            self._pending_onboarding_outbound["onboarding_request_id"] = request_id

    def clear_pending_onboarding_outbound(self, step_id: str | None = None) -> None:
        if not self._pending_onboarding_outbound:
            return
        if (
            step_id
            and self._pending_onboarding_outbound.get(
                "onboarding_trigger_step_id",
            )
            != step_id
        ):
            return
        self._pending_onboarding_outbound = None

    def note_whatsapp_window_open(self, contact_id: int | None, is_open: bool) -> None:
        """Record the latest known WhatsApp free-form window state for a contact."""
        if contact_id is None:
            return
        self._whatsapp_window_open[int(contact_id)] = bool(is_open)

    def whatsapp_window_state(self, contact_id: int | None) -> bool | None:
        """Return True/False for the contact's WhatsApp window, or None if unknown.

        A pending template resend is authoritative proof the window is closed
        (the last send fell back to a placeholder and no reply has reopened it).
        Otherwise fall back to the last observed/seeded state.
        """
        if contact_id is None:
            return None
        cid = int(contact_id)
        if cid in self._pending_whatsapp_resends:
            return False
        return self._whatsapp_window_open.get(cid)

    async def seed_whatsapp_window(self, contact_id: int) -> None:
        """Best-effort: ask the gateway whether a contact's window is open.

        Used at startup so the brain knows up front whether a first send will
        deliver verbatim or only a placeholder. Failures are swallowed — the
        state simply stays unknown and the send_whatsapp docstring falls back to
        its window-agnostic guidance.
        """
        contact = self._get_contact_safe(contact_id)
        whatsapp_number = (contact or {}).get("whatsapp_number")
        if not whatsapp_number:
            return
        try:
            from unify.conversation_manager.domains import comms_utils

            is_open = await comms_utils.get_whatsapp_window(whatsapp_number)
        except Exception:
            is_open = None
        if is_open is not None:
            self.note_whatsapp_window_open(contact_id, is_open)

    def _get_contact_safe(self, contact_id: int) -> dict | None:
        try:
            return self.contact_index.get_contact(contact_id)
        except Exception:
            return None

    def stash_pending_whatsapp_resend_onboarding_metadata(
        self,
        contact_id: int,
        metadata: dict[str, str],
    ) -> None:
        if metadata:
            self._pending_whatsapp_resend_onboarding_metadata[contact_id] = dict(
                metadata,
            )

    def consume_pending_whatsapp_resend_onboarding_metadata(
        self,
        contact_id: int,
    ) -> dict[str, str] | None:
        return self._pending_whatsapp_resend_onboarding_metadata.pop(contact_id, None)

    def consume_pending_onboarding_outbound(self, medium: str) -> dict[str, str] | None:
        pending = self._pending_onboarding_outbound
        if not pending:
            return None
        if self.loop.time() > float(pending.get("expires_at", 0)):
            self._pending_onboarding_outbound = None
            return None
        expected_media = {
            "email": {"email"},
            "sms_message": {"sms_message"},
            "whatsapp_message": {"whatsapp_message"},
            "whatsapp_call": {"whatsapp_call"},
            "phone_call": {"phone_call"},
            "slack_message": {"slack_message", "slack_channel_message"},
            "discord_message": {"discord_message", "discord_channel_message"},
            # Workspace demo rows deliver their proof-of-completion summary over the unify_message medium.
            "workspace_mailbox": {"unify_message"},
            "workspace_drive": {"unify_message"},
            "workspace_calendar": {"unify_message"},
            "workspace_contacts": {"unify_message"},
            "workspace_tasks": {"unify_message"},
            "workspace_teams": {"unify_message"},
        }.get(str(pending.get("channel", "")), set())
        if medium not in expected_media:
            return None
        self._pending_onboarding_outbound = None
        return {
            key: value
            for key in (
                "onboarding_trigger_step_id",
                "onboarding_reply_step_id",
                "onboarding_request_id",
                "onboarding_origin_event_id",
            )
            if isinstance((value := pending.get(key)), str) and value
        }

    async def store_chat_history(self):
        if len(self.chat_history) >= 2:
            await self.event_broker.publish(
                "app:comms:chat_history",
                StoreChatHistory(chat_history=self.chat_history[-2:]).to_json(),
            )
            await asyncio.sleep(2)

    async def cleanup(self):
        """Clean up any running call processes and file sync."""
        await self.store_chat_history()
        local_ingress = getattr(self, "_local_comms_ingress", None)
        if local_ingress is not None:
            await local_ingress.stop()
        activation_materializer = getattr(self, "_activation_materializer", None)
        if activation_materializer is not None:
            try:
                await activation_materializer.stop()
            except Exception as exc:
                LOGGER.warning(
                    f"{DEFAULT_ICON} [ConversationManager] "
                    f"Failed to stop activation materializer: {exc}",
                )
        if self.call_manager.has_active_google_meet:
            await self.call_manager.cleanup_google_meet()
        elif self.call_manager.has_active_teams_meet:
            await self.call_manager.cleanup_teams_meet()
        else:
            await self.call_manager.cleanup_call_proc()
        await self.call_manager.cleanup_persistent_worker()

        await self._stop_file_sync()

        if self.job_name and self.assistant_id is not None:
            self._session_logger.debug(
                "session_end",
                f"Marking job {self.job_name} done",
            )
            mark_done_kwargs = {}
            if self.shutdown_reason:
                mark_done_kwargs["shutdown_reason"] = self.shutdown_reason
            assistant_jobs.mark_job_done(
                self.job_name,
                self.inactivity_timeout,
                **mark_done_kwargs,
            )
        self.stop.set()

    async def _stop_file_sync(self) -> None:
        """Stop file sync with managed VM."""
        if not self.initialized:
            return
        try:
            from unify.file_manager.managers.local import LocalFileManager

            local_fm = LocalFileManager()
            adapter = local_fm._adapter

            if not hasattr(adapter, "sync_started"):
                return

            if adapter.sync_started:
                LOGGER.debug(
                    f"{DEFAULT_ICON} [ConversationManager] Stopping file sync...",
                )
                await adapter.stop_sync()
                LOGGER.debug(f"{DEFAULT_ICON} [ConversationManager] File sync stopped")
        except Exception as e:
            LOGGER.error(
                f"{DEFAULT_ICON} [ConversationManager] Failed to stop file sync: {e}",
            )

    # Proactive speech related methods

    PROACTIVE_DEBOUNCE_SECONDS = 5

    def _on_fast_brain_generating(self) -> dict[str, bool]:
        """Called via IPC when the fast brain starts generating a reply.

        Restarts the proactive speech cycle so any in-flight decision is
        cancelled.  The quiescence gate in ``_proactive_speech_loop`` will
        prevent the countdown from starting until the pipeline is idle again.
        """
        if self._proactive_speech_enabled:
            asyncio.ensure_future(self.schedule_proactive_speech())
        return {"idle_smalltalk_allowed": self.fast_brain_idle_smalltalk_allowed()}

    def _on_pipeline_quiescent(self, quiescent: bool) -> None:
        """Called via IPC when the voice pipeline quiescence state changes."""
        if quiescent:
            self._voice_pipeline_quiescent.set()
        else:
            self._voice_pipeline_quiescent.clear()

    async def schedule_proactive_speech(self):
        """Cancel any pending proactive speech and start a fresh cycle.

        Called on every user/assistant utterance event to reset the silence
        timer.  Only operates in voice modes (call / meet).
        """
        self._proactive_speech_gen += 1
        my_gen = self._proactive_speech_gen
        await self.cancel_proactive_speech()

        if not self.mode.is_voice:
            return

        if not self._proactive_speech_enabled:
            return

        if self._proactive_speech_gen != my_gen:
            return

        self._proactive_speech_task = asyncio.create_task(
            self._proactive_speech_loop(my_gen),
        )
        self._proactive_speech_task.add_done_callback(log_task_exc)

    async def cancel_proactive_speech(self):
        if self._proactive_speech_task and not self._proactive_speech_task.done():
            if self._proactive_speech_task == asyncio.current_task():
                return

            self._proactive_speech_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._proactive_speech_task
            self._proactive_speech_task = None

    async def set_proactive_speech_enabled(self, enabled: bool):
        self._proactive_speech_enabled = enabled
        if not enabled:
            self._proactive_speech_gen += 1
            await self.cancel_proactive_speech()

    async def _proactive_speech_loop(self, gen: int = 0):
        _log = self._proactive_logger

        def _superseded() -> bool:
            return self._proactive_speech_gen != gen

        try:
            if not self._voice_pipeline_quiescent.is_set():
                _log.proactive_waiting_for_quiescence()
                await self._voice_pipeline_quiescent.wait()
                if _superseded():
                    return

            _log.proactive_debounce(self.PROACTIVE_DEBOUNCE_SECONDS)
            await asyncio.sleep(self.PROACTIVE_DEBOUNCE_SECONDS)

            if _superseded():
                return

            if not self._voice_pipeline_quiescent.is_set():
                _log.proactive_deferred("pipeline not quiescent")
                return

            # Gather context for the decision.
            conversation_turns, _ = self.get_recent_voice_transcript()

            # Attach the latest screenshot from each active visual source
            # so the proactive LLM can visually verify screen state.
            screenshots = self.peek_screenshot_buffer()
            latest_by_source: dict[str, ScreenshotEntry] = {}
            for entry in screenshots:
                latest_by_source[entry.source] = entry

            if latest_by_source:
                source_labels = {
                    "assistant": "Assistant's Screen",
                    "user": "User's Screen",
                    "webcam": "User's Webcam",
                }
                content_parts: list[dict] = []
                for source, entry in latest_by_source.items():
                    label = source_labels.get(source, "Screenshot")
                    content_parts.append(
                        {
                            "type": "text",
                            "text": (f'[{label}] User said: "{entry.utterance}"'),
                        },
                    )
                    content_parts.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{entry.b64}",
                            },
                        },
                    )
                conversation_turns.append(
                    {"role": "user", "content": content_parts},
                )
            else:
                active_visuals = []
                if self.user_screen_share_active:
                    active_visuals.append("the user is sharing their screen")
                if self.user_webcam_active:
                    active_visuals.append("the user's webcam is on")
                if self.assistant_screen_share_active:
                    active_visuals.append(
                        "the assistant's desktop is being shared",
                    )
                if active_visuals:
                    conversation_turns.append(
                        {
                            "role": "system",
                            "content": (
                                f"[context] "
                                f"{', '.join(active_visuals).capitalize()}."
                            ),
                        },
                    )

            # Nothing has been said yet — there is no silence to break.  The
            # cycle re-arms when a real utterance arrives.
            if not conversation_turns:
                return

            snapshot_state = self.prompt_renderer.render_state(
                self.contact_index,
                self.notifications_bar,
                self.in_flight_actions,
                self.completed_actions,
                self.last_snapshot,
                recent_tool_executions=self._recent_tool_executions,
                assistant_screen_share_active=self.assistant_screen_share_active,
                user_screen_share_active=self.user_screen_share_active,
                user_webcam_active=self.user_webcam_active,
                user_remote_control_active=self.user_remote_control_active,
                google_meet_active=self.call_manager.has_active_google_meet,
                teams_meet_active=self.call_manager.has_active_teams_meet,
                vm_ready=self.vm_ready,
                file_sync_complete=self.file_sync_complete,
                has_desktop=SESSION_DETAILS.assistant.has_managed_desktop,
            )
            brain_spec = build_brain_spec(self, snapshot_state=snapshot_state)

            action_context = _render_action_context(
                self.in_flight_actions,
                self.completed_actions,
                notifications=self.notifications_bar.notifications,
            )

            decision, llm_log_path = await self.proactive_speech.decide(
                conversation_turns,
                brain_spec.system_prompt.flatten(),
                action_context=action_context,
            )

            if _superseded():
                return

            _log.proactive_decision(
                decision.delay,
                decision.content,
            )

            # Wait the requested delay (cancellable if an utterance arrives).
            # `delay` is unbounded: a few seconds when someone is waiting on a
            # reply, many minutes during a focused collaborative silence.
            if decision.delay > 0:
                _log.proactive_speaking(decision.delay, decision.content)
                await asyncio.sleep(decision.delay)

            if _superseded():
                return

            # Do not pre-write to contact_index here: the line is recorded once,
            # via the actually-spoken Outbound utterance, only if it is genuinely
            # spoken (the fast brain discards proactive speech when the pipeline
            # is not quiescent). Pre-writing duplicated that record and logged
            # lines that were never said.
            contact = self.get_active_contact()

            event = FastBrainNotification(
                contact=contact or {},
                message=decision.content,
                should_speak=True,
                source="proactive_speech",
                llm_log_path=llm_log_path,
            )
            await self.event_broker.publish(
                "app:call:notification",
                event.to_json(),
            )
            _log.proactive_published(decision.content)

        except asyncio.CancelledError:
            _log.proactive_cancelled()
            raise
        except Exception as e:
            _log.proactive_error(str(e))
