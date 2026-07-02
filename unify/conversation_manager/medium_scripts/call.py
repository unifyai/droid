import os
import sys
import json
import queue
import asyncio
import threading
from dataclasses import dataclass
from importlib import resources

os.environ["UNITY_TERMINAL_LOG"] = "true"

from dotenv import load_dotenv

from livekit import agents, rtc
from livekit.agents import (
    AgentSession,
    Agent,
    RoomInputOptions,
    utils,
    tokenize,
    tts,
    stt,
)
from livekit.plugins import (
    cartesia,
    deepgram,
    elevenlabs,
    silero,
)
from livekit.agents.voice.io import TimedString

from unify.conversation_manager.livekit_unify_adapter import UnifyLLM

if sys.platform == "darwin":
    from livekit.plugins import noise_cancellation

from livekit.plugins.turn_detector.english import EnglishModel
from livekit.agents import ChatContext, ChatMessage
from livekit.agents import ModelSettings, llm
from livekit.agents.llm import ChatChunk, ChoiceDelta

from typing import AsyncIterable, Callable

load_dotenv()

from unify.conversation_manager.events import *
from unify.conversation_manager.utils import dispatch_livekit_agent
from unify.conversation_manager.prompt_builders import (
    SMALLTALK_DEFER_SENTINEL,
    SMALLTALK_SILENCE_SENTINEL,
    build_opening_greeting_messages,
    build_smalltalk_messages,
    build_voice_agent_prompt,
)
from unify.conversation_manager.tracing import (
    content_trace_id,
    monotonic_ms,
    now_utc_iso,
)
from unify.session_details import SESSION_DETAILS

# Shared helpers
from unify.conversation_manager.medium_scripts.common import (
    event_broker,
    create_end_call,  # kept for test monkeypatch compatibility
    match_say_meta,
    setup_participant_disconnect_handler,  # kept for test monkeypatch compatibility
    publish_call_started,
    publish_call_ended,
    delete_livekit_room,
    configure_from_cli,
    should_dispatch_livekit_agent,
    start_event_broker_receive,
    UserTrackCaptureManager,
    ScreenshotHistory,
    capture_assistant_screenshot,
    render_participant_comms,
    publish_meet_interaction_from_track,
    FastBrainLogger,
    hydrate_fast_brain_history,
)
from unify.conversation_manager.cm_types.screenshot import (
    ScreenshotEntry,
    generate_screenshot_path,
    write_screenshot_to_disk,
)
from unify.conversation_manager.domains.fast_brain_buffer import (
    compute_resume_text,
    pick_resume_lead_in,
    select_continuation,
    select_fast_reply,
)

# Globals initialized lazily or via prewarm to avoid duplicate heavy init
STT = None
VAD = None


# Module-level logger created early for prewarm (before entrypoint runs).
_log = FastBrainLogger()

DEPLETED_CREDITS_FAST_BRAIN_RESPONSE = (
    "Your credits are depleted, so I can't continue helping with setup or tasks "
    "until you top up. Please add credits in billing, then I'll pick this back up."
)
VIDEO_AVATAR_CHANNELS = frozenset({"unify_meet", "google_meet", "teams_meet"})
ELEVENLABS_TWIN_PRONUNCIATION_SOURCE = "t-w1n"
ELEVENLABS_TWIN_PRONUNCIATION_TARGET = "Twin"
IDLE_SMALLTALK_STATE_TIMEOUT_S = 0.2


def has_video_avatar_channel(channel: str) -> bool:
    return channel in VIDEO_AVATAR_CHANNELS


def _drain_elevenlabs_twin_pronunciation_buffer(
    pending: str,
    emitted: list[str],
) -> str:
    while pending:
        lower_pending = pending.lower()
        if ELEVENLABS_TWIN_PRONUNCIATION_SOURCE.startswith(lower_pending):
            if len(pending) == len(ELEVENLABS_TWIN_PRONUNCIATION_SOURCE):
                emitted.append(ELEVENLABS_TWIN_PRONUNCIATION_TARGET)
                return ""
            return pending

        emitted.append(pending[0])
        pending = pending[1:]

    return pending


async def _normalize_elevenlabs_twin_pronunciation_stream(
    text: AsyncIterable[str],
) -> AsyncIterable[str]:
    pending = ""

    async for chunk in text:
        emitted: list[str] = []
        for char in chunk:
            pending += char
            pending = _drain_elevenlabs_twin_pronunciation_buffer(
                pending,
                emitted,
            )
        if emitted:
            yield "".join(emitted)

    if pending:
        yield pending


class FastBrainCreditGateMonitor:
    """Polls credit state off the voice response path."""

    def __init__(self, refresh_interval_s: float = 5.0) -> None:
        from unify.spending_limits import CreditGateState

        self._refresh_interval_s = refresh_interval_s
        self._state = CreditGateState()

    @property
    def state(self):
        return self._state

    async def refresh_once(self) -> None:
        from unify.spending_limits import check_credit_gate_state

        next_state = await check_credit_gate_state()
        if next_state.allowed != self._state.allowed:
            if next_state.allowed:
                _log.info("Credit gate cleared")
            else:
                _log.warning(next_state.reason or "Credit gate active")
        self._state = next_state

    async def run(self) -> None:
        while True:
            await self.refresh_once()
            await asyncio.sleep(self._refresh_interval_s)


class MeetAudioBridge:
    """Owns all PortAudio/sounddevice streams on a single dedicated thread.

    PortAudio is thread-hostile: all API calls (open/start/stop/close) must
    happen on the same thread.  Keeping streams open for the entire Meet
    session also avoids the PulseAudio ring-buffer memory leak triggered by
    repeated open/close cycles (PortAudio issue #968).
    """

    def __init__(self, capture_rate: int = 16000):
        self._capture_rate = capture_rate
        self.capture_q: asyncio.Queue[bytes] = asyncio.Queue()
        self._playback_q: queue.Queue[tuple[bytes, int, int] | None] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._stop_event: threading.Event | None = None
        self._in_stream = None
        self._out_stream = None

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        self._playback_q = queue.Queue()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def write_playback(self, pcm: bytes, sample_rate: int, num_channels: int) -> None:
        if self._playback_q is not None:
            self._playback_q.put((pcm, sample_rate, num_channels))

    def stop(self) -> None:
        if self._stop_event is not None:
            self._stop_event.set()
        if self._playback_q is not None:
            self._playback_q.put(None)
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def _run(self) -> None:
        import sounddevice as sd
        import numpy as np

        def _capture_callback(indata, frames, time_info, status):
            pcm = (indata * 32767).astype(np.int16).tobytes()
            if self._loop is not None:
                self._loop.call_soon_threadsafe(self.capture_q.put_nowait, pcm)

        self._in_stream = sd.InputStream(
            channels=1,
            samplerate=self._capture_rate,
            dtype="float32",
            blocksize=1024,
            callback=_capture_callback,
        )
        self._in_stream.start()

        try:
            while not self._stop_event.is_set():
                try:
                    item = self._playback_q.get(timeout=0.1)
                except Exception:
                    continue
                if item is None:
                    break
                pcm_bytes, sample_rate, num_channels = item
                if self._out_stream is None:
                    self._out_stream = sd.OutputStream(
                        samplerate=sample_rate,
                        channels=num_channels,
                        dtype="int16",
                        latency="high",
                    )
                    self._out_stream.start()
                audio_arr = np.frombuffer(pcm_bytes, dtype=np.int16)
                if num_channels > 1:
                    audio_arr = audio_arr.reshape(-1, num_channels)
                self._out_stream.write(audio_arr)
        finally:
            if self._out_stream is not None:
                self._out_stream.stop()
                self._out_stream.close()
                self._out_stream = None
            if self._in_stream is not None:
                self._in_stream.stop()
                self._in_stream.close()
                self._in_stream = None


def prewarm(_ctx=None):
    global STT, VAD
    try:
        _log.info("Prewarm: initializing STT, VAD and turn detector…")
        STT = deepgram.STT(model="nova-3", language="en-GB")
        VAD = silero.VAD.load(min_speech_duration=0.15, min_silence_duration=1.0)
        _log.info("Prewarm complete")
    except Exception as e:  # noqa: BLE001
        _log.error(f"Prewarm failed: {e}")
        STT = None
        VAD = None


class Assistant(Agent):
    """
    TTS Fast Brain - handles real-time conversation independently.

    Uses a lightweight LLM (gpt-5.4-mini via UnifyLLM adapter) for fast
    conversational responses. Routes through unillm.AsyncUnify for local
    caching (CI) and usage tracking.
    Communicates with the Main CM Brain (slow brain) via Unix domain socket IPC.
    """

    def __init__(
        self,
        contact: dict,
        boss: dict,
        channel: str,
        instructions: str,
        outbound: bool = False,
        audio_bridge: MeetAudioBridge | None = None,
        normalize_elevenlabs_twin_pronunciation: bool = False,
    ) -> None:
        self.contact = contact
        self.boss = boss
        self.channel = channel
        self.audio_bridge = audio_bridge
        self.normalize_elevenlabs_twin_pronunciation = (
            normalize_elevenlabs_twin_pronunciation
        )
        if channel == "phone_call":
            self.utterance_event = InboundPhoneUtterance
            self.assistant_utterance_event = OutboundPhoneUtterance
        elif channel == "whatsapp_call":
            self.utterance_event = InboundWhatsAppCallUtterance
            self.assistant_utterance_event = OutboundWhatsAppCallUtterance
        elif channel == "google_meet":
            self.utterance_event = InboundGoogleMeetUtterance
            self.assistant_utterance_event = OutboundGoogleMeetUtterance
        elif channel == "teams_meet":
            self.utterance_event = InboundTeamsMeetUtterance
            self.assistant_utterance_event = OutboundTeamsMeetUtterance
        else:
            self.utterance_event = InboundUnifyMeetUtterance
            self.assistant_utterance_event = OutboundUnifyMeetUtterance
        self.call_received = not outbound
        self._user_speech_logged = False
        self.user_turn_generating = False
        self._credit_gate_state_provider: Callable | None = None
        # On outbound calls the opener is held until the callee's first utterance
        # (or a fallback timeout). While pending, the opener is the sole response
        # to that first turn: the fast-brain filler and the slow-brain turn are
        # suppressed for it. Cleared once the opener has been dispatched.
        self._opening_pending = outbound
        self._first_user_turn = asyncio.Event()
        # Optional short note the slow brain bundles with a spoken line for the
        # fast brain to use on the caller's next message (e.g. confirm a fact).
        # Replaced/cleared on each slow-brain spoken turn; never spoken aloud.
        self._fast_brain_guidance = ""
        # Monotonic user-turn counter and the latest turn the slow brain has
        # already produced spoken output for. A buffer filler is only useful as a
        # lead-in: if the slow brain has already responded to this turn, the
        # filler is dropped so it never plays AFTER the real answer.
        self._user_turn_seq = 0
        self._slow_brain_responded_turn = -1
        # Count of consecutive fast replies emitted since the slow brain last
        # spoke. After the first reaction, subsequent ones are marked as repeated
        # deferrals so they reassure ("bear with me") rather than starting a
        # fresh lookup. Reset when the slow brain delivers a real reply.
        self._buffers_since_slow_reply = 0
        # Armed when the recorded opener is interrupted before its static-removal
        # transition. Schedules a bridge recording at the start of the next turn.
        # The callable enqueues the bridge synchronously (no playout await) so
        # the fast-brain reply generates concurrently and queues behind it.
        self._pending_opening_bridge: Callable[[], None] | None = None
        # The in-flight TTS say handle (slow-brain speech or a briefed/speak
        # opener) registered for resumption, and the claimable resume candidate
        # produced when it is interrupted. Pre-recorded audio is never registered
        # here, so its hand-crafted tone is never continued by the live voice.
        self._active_tts: dict | None = None
        self._pending_continuation: dict | None = None
        self._tts_seq = 0
        self._publish_voice_interrupt: Callable | None = None
        # Full text of a continuation reply the fast brain is about to yield this
        # turn. Read once by the ``speech_created`` observer to register the reply
        # handle as interruptible (so a resumed line is itself recursively
        # resumable), then cleared. ``None`` for ordinary buffer fillers, which
        # carry no substantive content worth re-surfacing.
        self._continuation_full_text: str | None = None
        # Cancels the speculatively-started slow-brain run for this turn the
        # instant the fast brain resolves it itself (CONTINUE or SMALLTALK), so
        # the slow brain cannot also re-deliver / re-answer the same content.
        self._publish_fast_brain_continued: Callable | None = None
        # Fully answers a pure small-talk turn from persona + recent history, or
        # returns None to defer to the slow brain. Set in the entrypoint.
        self._generate_smalltalk_reply: Callable | None = None
        self._idle_smalltalk_allowed = False
        self._idle_smalltalk_state_event = asyncio.Event()

        super().__init__(instructions=instructions)

    def set_credit_gate_state_provider(self, provider: Callable) -> None:
        self._credit_gate_state_provider = provider

    def set_idle_smalltalk_allowed(self, allowed: bool) -> None:
        self._idle_smalltalk_allowed = allowed
        self._idle_smalltalk_state_event.set()

    async def _request_idle_smalltalk_state(self) -> bool:
        self._idle_smalltalk_allowed = False
        self._idle_smalltalk_state_event.clear()
        publish_task = asyncio.create_task(
            event_broker.publish("app:comms:fast_brain_generating", "{}"),
        )
        if getattr(event_broker, "_socket_client", None) is None:
            return False
        await publish_task
        try:
            await asyncio.wait_for(
                self._idle_smalltalk_state_event.wait(),
                timeout=IDLE_SMALLTALK_STATE_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            return False
        return self._idle_smalltalk_allowed

    def set_call_received(self):
        self.call_received = True

    async def on_user_turn_completed(
        self,
        turn_ctx: ChatContext,
        new_message: ChatMessage,
    ) -> None:
        """Hook called when user finishes speaking — before LLM generation starts.

        The opener static-removal bridge (if armed) is *scheduled* here but not
        awaited: enqueueing it synchronously, before this hook returns, keeps it
        ahead of the reply the framework generates next (same speech priority,
        FIFO), while letting the fast brain think during the bridge playout
        instead of after it. The reply queues behind the bridge rather than
        interrupting it, and the bridge text stays in the in-flight speech queue
        so the concurrent generation sees it and continues naturally from it.
        """
        # New user turn: a fresh buffer filler is now warranted (until the slow
        # brain responds to this turn).
        self._user_turn_seq += 1
        # On an outbound call, the callee's first completed utterance triggers the
        # held opener (the opener answers it). Signalling on turn-completed means
        # we respond after their "Hello?", never over it.
        if self._opening_pending and not self._first_user_turn.is_set():
            self._first_user_turn.set()
        if self._pending_opening_bridge is not None:
            schedule_bridge = self._pending_opening_bridge
            self._pending_opening_bridge = None
            schedule_bridge()
        text = new_message.text_content or ""
        if text:
            _log.user_speech(text)
            self._user_speech_logged = True

    async def llm_node(
        self,
        chat_ctx: llm.ChatContext,
        tools: list[llm.FunctionTool | llm.RawFunctionTool],
        model_settings: ModelSettings,
    ) -> AsyncIterable[llm.ChatChunk]:
        """Wait for call connection, then emit a single buffer filler phrase.

        The fast brain does not free-generate substantive replies; it selects one
        short, safe filler phrase to cover latency while the slow brain composes
        the real (verbatim-spoken) response.
        """
        self.user_turn_generating = True
        # Stale-guard: only a continuation set below should be registered for
        # resumption; clear any leftover so an ordinary buffer reply is not.
        self._continuation_full_text = None
        try:
            _log.info("Waiting for call to be received…")
            while not self.call_received:
                await asyncio.sleep(0.1)
            _log.call_status("call_received")

            # While the outbound opener is still pending, the callee's first turn
            # triggers the opener itself (the opener is the reply). Emit no filler
            # so it does not precede or race the opener.
            if self._opening_pending:
                _log.info("Filler suppressed: outbound opener still pending")
                return

            credit_gate_state = (
                self._credit_gate_state_provider()
                if self._credit_gate_state_provider is not None
                else None
            )
            if credit_gate_state is not None and not credit_gate_state.allowed:
                _log.info("Credit gate response served from cached state")
                yield ChatChunk(
                    id=f"credit-gate-{monotonic_ms()}",
                    delta=ChoiceDelta(
                        role="assistant",
                        content=DEPLETED_CREDITS_FAST_BRAIN_RESPONSE,
                    ),
                )
                return

            # The buffer is only useful as a lead-in. If the slow brain has
            # already produced spoken output for this turn (e.g. this is a
            # notification-triggered re-generation, or its answer landed first),
            # emit nothing - a filler must never play AFTER the real answer.
            my_turn = self._user_turn_seq
            if self._slow_brain_responded_turn >= my_turn:
                _log.info("Buffer suppressed: slow brain already responded this turn")
                return

            idle_status_smalltalk = await self._request_idle_smalltalk_state()

            # The fast brain does not compose the real answer (the slow brain
            # does, spoken verbatim). It gives one brief, natural reaction to
            # cover the gap. The first reply since the slow brain last spoke is a
            # fresh reaction; subsequent ones (the caller spoke again before the
            # real reply landed) are marked as repeated deferrals so they reassure
            # rather than starting a fresh lookup.
            already_deferred = self._buffers_since_slow_reply >= 1
            user_text = ""
            recent_assistant_text = ""
            for item in reversed(chat_ctx.items):
                role = getattr(item, "role", None)
                if role == "user" and not user_text:
                    user_text = item.text_content or ""
                elif role == "assistant" and not recent_assistant_text:
                    recent_assistant_text = item.text_content or ""
                if user_text and recent_assistant_text:
                    break

            # If this barge-in cut off a slow-brain line, the fast brain may
            # resume the unheard remainder immediately (speaking the slow brain's
            # own verbatim words) instead of waiting a full slow-brain turn for it
            # to be re-delivered.
            continuation = await self._claim_interrupted_continuation(user_text)
            if continuation is not None:
                if self._slow_brain_responded_turn >= my_turn:
                    _log.info("Continuation suppressed: slow brain already responded")
                    return
                self._buffers_since_slow_reply += 1
                # Resume the slow brain's verbatim remainder AS the reply. Mark the
                # full text so the ``speech_created`` observer registers this reply
                # for interruption-stashing -> a barge-in mid-resume re-stashes a
                # fresh candidate, making continuation recursive. Cancel the
                # speculatively-started slow-brain run so it cannot re-deliver it.
                self._continuation_full_text = continuation
                if self._publish_fast_brain_continued is not None:
                    await self._publish_fast_brain_continued(my_turn)
                yield ChatChunk(
                    id=f"fast-brain-continuation-{monotonic_ms()}",
                    delta=ChoiceDelta(role="assistant", content=continuation),
                )
                return

            # Refresh screenshots only on the first reaction (also feeds the slow
            # brain); repeated deferrals don't need it and should stay snappy.
            if not already_deferred:
                await self._capture_screenshots_for_llm(chat_ctx)

            # Race the small-talk reply against the lean filler. If this turn is
            # pure small talk (greeting, biographical, simple self-context,
            # repeat), the fast brain answers it in full and cancels the eager
            # slow-brain run; otherwise it defers and the filler covers the gap
            # while the slow brain composes the real answer. Running both
            # concurrently keeps DEFER turns at the filler's latency.
            _log.info("Selecting fast reply… (llm_node_start)")
            smalltalk_task = (
                asyncio.create_task(
                    self._generate_smalltalk_reply(
                        user_text,
                        idle_status_smalltalk=idle_status_smalltalk,
                    ),
                )
                if self._generate_smalltalk_reply is not None
                else None
            )
            buffer_task = asyncio.create_task(
                select_fast_reply(
                    user_text,
                    recent_assistant_text,
                    already_deferred=already_deferred,
                    guidance=self._fast_brain_guidance,
                ),
            )

            smalltalk = await smalltalk_task if smalltalk_task is not None else None
            if smalltalk is _SMALLTALK_STAY_SILENT:
                # A bare acknowledgement ("okay") needs no reply: say nothing and
                # cancel this turn's slow-brain run so neither brain speaks.
                buffer_task.cancel()
                _log.info("Small talk: staying silent on bare acknowledgement")
                if self._publish_fast_brain_continued is not None:
                    await self._publish_fast_brain_continued(my_turn)
                return
            if smalltalk is not None:
                buffer_task.cancel()  # the filler is not needed
                if self._slow_brain_responded_turn >= my_turn:
                    _log.info("Small talk suppressed: slow brain already responded")
                    return
                self._buffers_since_slow_reply += 1
                # The fast brain fully answered this turn; cancel the eager
                # slow-brain run so it does not also answer the same thing.
                if self._publish_fast_brain_continued is not None:
                    await self._publish_fast_brain_continued(my_turn)
                yield ChatChunk(
                    id=f"fast-brain-smalltalk-{monotonic_ms()}",
                    delta=ChoiceDelta(role="assistant", content=smalltalk),
                )
                return

            phrase = await buffer_task

            # Re-check: the slow brain may have answered during selection. If so,
            # drop the now-stale filler so it does not trail the real answer.
            if self._slow_brain_responded_turn >= my_turn:
                _log.info("Buffer suppressed: slow brain responded during selection")
                return

            self._buffers_since_slow_reply += 1
            yield ChatChunk(
                id=f"fast-brain-buffer-{monotonic_ms()}",
                delta=ChoiceDelta(role="assistant", content=phrase),
            )
        finally:
            self.user_turn_generating = False

    async def _claim_interrupted_continuation(self, user_text: str) -> str | None:
        """Decide the fate of an interrupted line - the fast brain's front-door job.

        Every turn after a barge-in routes through here:
        - No pending candidate -> None (ordinary buffer path).
        - A barge-in that produced no transcript (speechless noise/echo) ->
          auto-CONTINUE: resume the remainder verbatim, since there is no spoken
          content that could possibly justify deferring.
        - A barge-in with speech -> classify (heavily biased to CONTINUE); on a
          DEFER, hand the remainder to the slow brain via ``VoiceInterrupt``.

        Returns "{lead-in} {verbatim remainder}" to resume, else None. Claiming
        marks the candidate consumed so it is delivered exactly once.
        """
        text = (user_text or "").strip()

        # Barge-in vs user-turn race: if a line was just interrupted but its
        # remainder hasn't been computed yet, wait briefly for it.
        active = self._active_tts
        if (
            self._pending_continuation is None
            and active is not None
            and getattr(active.get("handle"), "interrupted", False)
        ):
            for _ in range(6):  # ~300ms
                await asyncio.sleep(0.05)
                if self._pending_continuation is not None:
                    break

        pending = self._pending_continuation
        if not pending or pending.get("consumed"):
            return None

        # Claim synchronously so no other path can double-deliver it.
        pending["consumed"] = True
        resume_text = (pending.get("resume_text") or "").strip()
        remainder = (pending.get("remainder") or "").strip()
        spoken_prefix = (pending.get("spoken_prefix") or "").strip()
        self._pending_continuation = None
        self._active_tts = None
        if not resume_text:
            return None

        # Speechless barge-in: the only sensible action is to continue, so resume
        # directly without consulting the classifier.
        if not text:
            return f"{pick_resume_lead_in()} {resume_text}".strip()

        lead_in = await select_continuation(resume_text, user_text)
        if lead_in is None:
            # Redirect / new ask: hand the remainder to the slow brain instead.
            if self._publish_voice_interrupt is not None and remainder:
                await self._publish_voice_interrupt(spoken_prefix, remainder)
            return None
        return f"{lead_in} {resume_text}".strip()

    async def stt_node(
        self,
        audio: AsyncIterable[rtc.AudioFrame],
        model_settings: ModelSettings,
    ):
        if (
            self.channel not in ("google_meet", "teams_meet")
            or self.audio_bridge is None
        ):
            async for event in super().stt_node(audio, model_settings):
                yield event
            return

        activity = self._get_activity_or_raise()
        assert activity.stt is not None

        wrapped_stt = activity.stt
        if not activity.stt.capabilities.streaming:
            if not activity.vad:
                raise RuntimeError(
                    "STT does not support streaming and no VAD is available",
                )
            wrapped_stt = stt.StreamAdapter(stt=wrapped_stt, vad=activity.vad)

        _RATE = self.audio_bridge._capture_rate

        async def _audio_from_bridge():
            while True:
                pcm = await self.audio_bridge.capture_q.get()
                samples = len(pcm) // 2
                yield rtc.AudioFrame(
                    data=pcm,
                    sample_rate=_RATE,
                    num_channels=1,
                    samples_per_channel=samples,
                )

        async with wrapped_stt.stream() as stt_stream:

            async def _forward():
                async for frame in _audio_from_bridge():
                    stt_stream.push_frame(frame)

            fwd = asyncio.create_task(_forward())
            try:
                async for event in stt_stream:
                    yield event
            finally:
                await utils.aio.cancel_and_wait(fwd)

    async def tts_node(
        self,
        text: AsyncIterable[str],
        model_settings: ModelSettings,
    ) -> AsyncIterable:
        if self.normalize_elevenlabs_twin_pronunciation:
            text = _normalize_elevenlabs_twin_pronunciation_stream(text)

        if (
            self.channel not in ("google_meet", "teams_meet")
            or self.audio_bridge is None
        ):
            async for frame in super().tts_node(text, model_settings):
                yield frame
            return

        activity = self._get_activity_or_raise()
        assert activity.tts is not None

        wrapped_tts = activity.tts
        if not activity.tts.capabilities.streaming:
            wrapped_tts = tts.StreamAdapter(
                tts=wrapped_tts,
                sentence_tokenizer=tokenize.basic.SentenceTokenizer(),
            )

        async with wrapped_tts.stream() as tts_stream:

            async def _forward_input():
                async for chunk in text:
                    tts_stream.push_text(chunk)
                tts_stream.end_input()

            fwd = asyncio.create_task(_forward_input())
            try:
                async for ev in tts_stream:
                    frame = ev.frame
                    self.audio_bridge.write_playback(
                        frame.data,
                        frame.sample_rate,
                        frame.num_channels,
                    )
            finally:
                await utils.aio.cancel_and_wait(fwd)


def _load_config_from_metadata(ctx: agents.JobContext) -> dict | None:
    """Parse call config from job dispatch metadata (persistent worker path).

    Returns the parsed dict, or None when no metadata is present (legacy
    subprocess path).
    """
    raw = getattr(ctx.job, "metadata", None)
    if not raw:
        return None
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None


def _hydrate_session_details_from_metadata(meta: dict) -> None:
    """Apply assistant identity fields carried by LiveKit job metadata."""
    assistant_bio = meta.get("assistant_bio", "")
    SESSION_DETAILS.assistant.about = assistant_bio
    SESSION_DETAILS.assistant.is_coordinator = meta.get("is_coordinator", False) is True
    if meta.get("assistant_id"):
        try:
            SESSION_DETAILS.assistant.agent_id = int(meta["assistant_id"])
        except (ValueError, TypeError):
            pass
    if meta.get("user_id"):
        SESSION_DETAILS.user.id = meta["user_id"]
    if meta.get("assistant_name"):
        parts = meta["assistant_name"].split(None, 1)
        SESSION_DETAILS.assistant.first_name = parts[0] if parts else ""
        SESSION_DETAILS.assistant.surname = parts[1] if len(parts) > 1 else ""


_CALL_OPENING_MODES = {"speak", "simulated", "silent", "briefed", "recorded"}

# On an outbound call, hold the opener until the earliest of the callee's first
# completed utterance (their "Hello?") or this fallback timeout. Triggering on
# their speech ensures the opener lands when they are actually listening, instead
# of into dead air right after the line connects.
OUTBOUND_OPENING_TRIGGER_TIMEOUT_S = 5.0

# Ceiling on a fast-brain small-talk reply. A social / biographical / self-context
# answer is a sentence or two; anything longer means the model overreached into a
# substantive answer (the slow brain's job), so we drop it and defer instead.
_MAX_SMALLTALK_CHARS = 300

# Distinct small-talk outcome: a bare acknowledgement ("okay") needs no reply at
# all. The fast brain stays silent AND cancels the slow-brain run, so neither
# brain speaks. Identity-checked, so it can never collide with a real reply.
_SMALLTALK_STAY_SILENT = object()

# When the assistant ends a Unify Meet, we first tell the Console to leave the
# room itself (so its WebRTC peer connection and SCTP data channels close
# cleanly) before the agent shuts down and the room is deleted server-side.
# Without this lead time the server-side DeleteRoom force-evicts the still-
# connected browser, which surfaces as benign but noisy "Unknown DataChannel
# error" logs in the Next.js console. This is the grace given for the client to
# disconnect gracefully on its own.
MEET_GRACEFUL_LEAVE_GRACE_S = 0.6

# Sentence-level timings for the coordinator onboarding intro (walkie + clean).
# Boundaries were derived from the original per-sentence audio slices aligned via
# Whisper word timestamps; playback uses one continuous recording with TimedString
# chunks so LiveKit can commit only the heard prefix on interruption.
_COORDINATOR_ONBOARDING_CLEAN_START_TIME = 60.650249
_COORDINATOR_ONBOARDING_TIMED_CHUNKS: list[dict[str, object]] = [
    {"text": "Hi, I'm T dash W 1 N.", "start_time": 0.000000, "end_time": 2.780000},
    {
        "text": "Before you ask, no I'm not one of Elon's many children, and no he didn't come up with the name, thankfully.",
        "start_time": 2.780000,
        "end_time": 8.360000,
    },
    {
        "text": "I have this name because I'll be acting as your digital twin.",
        "start_time": 8.360000,
        "end_time": 11.960000,
    },
    {"text": "Do you get it?", "start_time": 11.960000, "end_time": 13.530000},
    {"text": "Twin?", "start_time": 13.530000, "end_time": 14.320000},
    {
        "text": "Like T dash W 1 N spells Twin?",
        "start_time": 14.320000,
        "end_time": 17.380000,
    },
    {
        "text": "The creators of this platform express how important it is that you acknowledge that it's a clever and funny name.",
        "start_time": 17.380000,
        "end_time": 24.650000,
    },
    {"text": "Okay, what next.", "start_time": 24.650000, "end_time": 26.330000},
    {
        "text": "They didn't give me much to work with on this intro to be honest.",
        "start_time": 26.330000,
        "end_time": 29.840000,
    },
    {
        "text": "I think I was meant to tell you about my capabilities?",
        "start_time": 29.840000,
        "end_time": 32.380000,
    },
    {
        "text": "I'm not one for bragging, but I'll do my best.",
        "start_time": 32.380000,
        "end_time": 35.610000,
    },
    {"text": "What can I say?", "start_time": 35.610000, "end_time": 36.890000},
    {"text": 'I\'m not a "tool".', "start_time": 36.890000, "end_time": 38.170000},
    {"text": 'I\'m not an "agent".', "start_time": 38.170000, "end_time": 39.700000},
    {
        "text": "I'm your living, breathing colleague, metaphorically speaking.",
        "start_time": 39.700000,
        "end_time": 43.550000,
    },
    {
        "text": "Don't think about prompting me, or configuring me, just talk to me naturally like you would anyone else, and I'll be able to help.",
        "start_time": 43.550000,
        "end_time": 50.260000,
    },
    {
        "text": "It's really that simple.",
        "start_time": 50.260000,
        "end_time": 51.940000,
    },
    {
        "text": "There's not much more to say.",
        "start_time": 51.940000,
        "end_time": 53.680000,
    },
    {
        "text": "I'll now walk you through the platform.",
        "start_time": 53.680000,
        "end_time": 56.130000,
    },
    {
        "text": "Also, let me remove this voice static.",
        "start_time": 56.130000,
        "end_time": 60.650249,
    },
    {"text": "Much better.", "start_time": 60.650249, "end_time": 61.283567},
    {
        "text": "Any questions before we start with the onboarding?",
        "start_time": 61.283567,
        "end_time": 63.922391,
    },
    {
        "text": "By the way, you'll probably want to unmute yourself first. Click the microphone at the bottom of the meet window, and then I'll be able to hear you.",
        "start_time": 63.922391,
        "end_time": 71.733311,
    },
]

_RECORDED_OPENING_ASSETS = {
    "coordinator_onboarding_intro": "twin-onboarding-intro.mp3",
    "coordinator_onboarding_static_bridge": "twin-onboarding-static-bridge.mp3",
}
_RECORDED_OPENING_TRANSCRIPTS: dict[str, str] = {}

_WALKIE_OPENER_BRIDGE_TRANSCRIPT = """\
Hang on, let me just remove this voice static.

Much better."""

# Recorded openers played as one continuous audio stream with TimedString
# transcript chunks. If the caller interrupts before the clean voice transition,
# ``bridge`` is armed for the next assistant turn.
_RECORDED_OPENINGS = {
    "coordinator_onboarding_intro": {
        "asset": "coordinator_onboarding_intro",
        "timed_chunks": _COORDINATOR_ONBOARDING_TIMED_CHUNKS,
        "clean_start_time": _COORDINATOR_ONBOARDING_CLEAN_START_TIME,
        "bridge": {
            "asset": "coordinator_onboarding_static_bridge",
            "transcript": _WALKIE_OPENER_BRIDGE_TRANSCRIPT,
        },
    },
}


def _recorded_opening_timed_transcript(chunks: list[dict[str, object]]) -> str:
    return " ".join(str(chunk["text"]) for chunk in chunks)


async def _timed_opening_text(
    chunks: list[dict[str, object]],
) -> AsyncIterable[str]:
    for chunk in chunks:
        yield TimedString(
            str(chunk["text"]),
            start_time=float(chunk["start_time"]),
            end_time=float(chunk["end_time"]),
        )


@dataclass(frozen=True)
class _PreloadedAudio:
    pcm: bytes
    sample_rate: int
    num_channels: int


def _load_recorded_asset_pcm(asset_key: str) -> _PreloadedAudio:
    import numpy as _np
    import soundfile as _sf

    filename = _RECORDED_OPENING_ASSETS.get(asset_key)
    if not filename:
        raise ValueError(f"unknown recorded opening asset: {asset_key}")
    asset = resources.files("unify.assets.audio").joinpath(filename)
    with resources.as_file(asset) as recording_path:
        with _sf.SoundFile(str(recording_path)) as recording:
            data = recording.read(dtype="int16", always_2d=True)
            sample_rate = recording.samplerate
            num_channels = recording.channels
    return _PreloadedAudio(
        pcm=_np.ascontiguousarray(data).tobytes(),
        sample_rate=sample_rate,
        num_channels=num_channels,
    )


def _preload_recorded_opening_pcm(config: dict) -> dict[str, _PreloadedAudio]:
    asset_key = config.get("recording_asset", "").strip()
    spec = _RECORDED_OPENINGS.get(asset_key)
    if spec is None:
        source = _recorded_opening_source(config)
        if source.startswith("asset://"):
            key = source.removeprefix("asset://")
            if key in _RECORDED_OPENING_ASSETS:
                return {key: _load_recorded_asset_pcm(key)}
        return {}

    preloaded: dict[str, _PreloadedAudio] = {}
    asset = spec["asset"]
    preloaded[asset] = _load_recorded_asset_pcm(asset)
    if bridge := spec.get("bridge"):
        key = bridge["asset"]
        if key not in preloaded:
            preloaded[key] = _load_recorded_asset_pcm(key)
    return preloaded


def _pcm_audio_frames(
    pcm: bytes,
    *,
    sample_rate: int,
    num_channels: int,
    frame_duration_ms: int = 20,
) -> AsyncIterable[rtc.AudioFrame]:
    async def _frames() -> AsyncIterable[rtc.AudioFrame]:
        import numpy as _np

        samples_per_chunk = max(1, int(sample_rate * frame_duration_ms / 1000))
        bytes_per_chunk = samples_per_chunk * num_channels * 2
        offset = 0
        while offset < len(pcm):
            end = min(offset + bytes_per_chunk, len(pcm))
            chunk = pcm[offset:end]
            block = _np.frombuffer(chunk, dtype="int16").reshape(-1, num_channels)
            yield rtc.AudioFrame(
                data=_np.ascontiguousarray(block).tobytes(),
                sample_rate=sample_rate,
                num_channels=num_channels,
                samples_per_channel=len(block),
            )
            offset = end
            await asyncio.sleep(0)

    return _frames()


def _preloaded_audio_frames(
    audio: _PreloadedAudio,
    *,
    frame_duration_ms: int = 20,
) -> AsyncIterable[rtc.AudioFrame]:
    return _pcm_audio_frames(
        audio.pcm,
        sample_rate=audio.sample_rate,
        num_channels=audio.num_channels,
        frame_duration_ms=frame_duration_ms,
    )


def _recorded_opening_audio(
    source: str,
    preloaded: dict[str, _PreloadedAudio] | None = None,
    *,
    frame_duration_ms: int = 20,
) -> AsyncIterable[rtc.AudioFrame]:
    if source.startswith("asset://"):
        asset_key = source.removeprefix("asset://")
        cached = (preloaded or {}).get(asset_key)
        if cached is not None:
            return _preloaded_audio_frames(cached, frame_duration_ms=frame_duration_ms)
    return _recording_audio_frames(source, frame_duration_ms=frame_duration_ms)


def _recording_audio_frames(
    source: str,
    *,
    frame_duration_ms: int = 20,
) -> AsyncIterable[rtc.AudioFrame]:
    async def _frames() -> AsyncIterable[rtc.AudioFrame]:
        import io as _io

        import httpx as _httpx
        import numpy as _np
        import soundfile as _sf

        async def _yield_file(recording_source: str | _io.BytesIO):
            with _sf.SoundFile(recording_source) as recording:
                samples_per_chunk = max(
                    1,
                    int(recording.samplerate * frame_duration_ms / 1000),
                )
                while True:
                    block = recording.read(
                        samples_per_chunk,
                        dtype="int16",
                        always_2d=True,
                    )
                    if len(block) == 0:
                        break
                    pcm = _np.ascontiguousarray(block).tobytes()
                    yield rtc.AudioFrame(
                        data=pcm,
                        sample_rate=recording.samplerate,
                        num_channels=recording.channels,
                        samples_per_channel=len(block),
                    )
                    await asyncio.sleep(0)

        if source.startswith("asset://"):
            asset_key = source.removeprefix("asset://")
            filename = _RECORDED_OPENING_ASSETS.get(asset_key)
            if not filename:
                raise ValueError(f"unknown recorded opening asset: {asset_key}")
            asset = resources.files("unify.assets.audio").joinpath(filename)
            with resources.as_file(asset) as recording_path:
                async for frame in _yield_file(str(recording_path)):
                    yield frame
            return

        recording_source: str | _io.BytesIO = os.path.expanduser(source)
        if source.startswith(("http://", "https://")):
            async with _httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(source)
                response.raise_for_status()
                recording_source = _io.BytesIO(response.content)

        async for frame in _yield_file(recording_source):
            yield frame

    return _frames()


def _recorded_opening_source(config: dict) -> str:
    asset = config.get("recording_asset", "").strip()
    if asset:
        return f"asset://{asset}"
    path = config.get("recording_path", "").strip()
    url = config.get("recording_url", "").strip()
    return path or url


def _recorded_opening_transcript(config: dict) -> str:
    transcript = config.get("transcript", "").strip()
    if transcript:
        return transcript
    asset = config.get("recording_asset", "").strip()
    if asset:
        spec = _RECORDED_OPENINGS.get(asset)
        if spec is not None:
            return _recorded_opening_timed_transcript(spec["timed_chunks"])
        transcript = _RECORDED_OPENING_TRANSCRIPTS.get(asset, "").strip()
    if not transcript:
        raise ValueError("recorded opening requires transcript")
    return transcript


def _recorded_opening_source_count(config: dict) -> int:
    return sum(
        bool(config.get(key, "").strip())
        for key in ("recording_asset", "recording_path", "recording_url")
    )


def _normalize_call_opening_config(raw: object) -> dict:
    if raw in (None, ""):
        return {"mode": "speak"}
    if isinstance(raw, str):
        raw = json.loads(raw)
    if not isinstance(raw, dict):
        raise ValueError("opening_config must be an object")

    mode = str(raw.get("mode", "speak")).strip()
    if mode not in _CALL_OPENING_MODES:
        raise ValueError(
            "opening_config.mode must be one of speak, simulated, silent, briefed, recorded",
        )

    config = {"mode": mode}
    if raw.get("simulated_utterance") is not None:
        config["simulated_utterance"] = str(raw["simulated_utterance"])
    if raw.get("system_context") is not None:
        config["system_context"] = str(raw["system_context"])
    if raw.get("source") is not None:
        config["source"] = str(raw["source"])
    if raw.get("transcript") is not None:
        config["transcript"] = str(raw["transcript"])
    if raw.get("recording_asset") is not None:
        config["recording_asset"] = str(raw["recording_asset"])
    if raw.get("recording_path") is not None:
        config["recording_path"] = str(raw["recording_path"])
    if raw.get("recording_url") is not None:
        config["recording_url"] = str(raw["recording_url"])

    if mode == "briefed" and not config.get("system_context", "").strip():
        raise ValueError("briefed opening requires system_context")
    if mode == "recorded" and config.get("recording_asset", "").strip() not in (
        _RECORDED_OPENINGS
    ):
        _recorded_opening_transcript(config)
        if _recorded_opening_source_count(config) != 1:
            raise ValueError(
                "recorded opening requires exactly one of recording_asset, recording_path, or recording_url",
            )
    return config


def _configure_child_logging() -> None:
    """Ensure Unity's LOGGER works in LiveKit's pre-warmed child processes.

    LiveKit agents (v1.2.x) uses ``forkserver`` on Linux.  Child processes
    are forked from a lean server process, not from the worker, so the
    ``unity`` logger's handlers may point to stale file descriptors.

    The framework routes child logs through a ``LogQueueHandler`` on the
    **root** logger, which serialises records back to the worker process.
    We enable propagation so Unity records flow through that channel, and
    remove any direct handlers that could double-emit or silently fail.
    """
    import logging as _logging

    from unify.logger import LOGGER as _L

    _L.propagate = True
    for h in list(_L.handlers):
        _L.removeHandler(h)

    for name in ("livekit", "livekit.agents", "livekit.plugins"):
        lg = _logging.getLogger(name)
        lg.propagate = True
        for h in list(lg.handlers):
            lg.removeHandler(h)


async def entrypoint(ctx: agents.JobContext):
    global STT, VAD

    _configure_child_logging()

    # This prewarmed process is now being consumed by a job, so clear the
    # idle-ready marker. The worker re-creates it once a replacement idle
    # process has finished warming, which is what gates starting the next call.
    from unify.conversation_manager.medium_scripts.worker import mark_worker_busy

    mark_worker_busy()

    # Wire the module-level logger into the shared event broker.
    event_broker.set_logger(_log)

    # --- Config: persistent worker (job metadata) or legacy subprocess (env) ---
    meta = _load_config_from_metadata(ctx)
    _log.info(f"Entrypoint started (has_metadata={meta is not None})")
    if meta:
        from unify.conversation_manager.domains.ipc_socket import init_socket_for_job

        ipc_path = meta.get("ipc_socket_path", "")
        if ipc_path:
            init_socket_for_job(ipc_path)
            event_broker.reinit_socket()
            _log.info(f"IPC socket initialised: {ipc_path}")
        else:
            _log.warning("No ipc_socket_path in job metadata — IPC disabled")

        voice_provider = meta.get("voice_provider") or "cartesia"
        voice_id = meta.get("voice_id") or ""
        outbound = meta.get("outbound", False)
        channel = meta.get("channel", "phone")
        assistant_bio = meta.get("assistant_bio", "")
        contact = meta.get("contact", {})
        boss = meta.get("boss", {})
        opening_config = _normalize_call_opening_config(meta.get("opening_config"))
        _hydrate_session_details_from_metadata(meta)
    else:
        _log.warning(
            "No job metadata — falling back to env-based config (IPC disabled)",
        )
        SESSION_DETAILS.populate_from_env()
        voice_provider = SESSION_DETAILS.voice.provider
        voice_id = SESSION_DETAILS.voice.id
        outbound = SESSION_DETAILS.voice_call.outbound
        channel = SESSION_DETAILS.voice_call.channel
        assistant_bio = SESSION_DETAILS.assistant.about
        contact = json.loads(SESSION_DETAILS.voice_call.contact_json or "{}")
        boss = json.loads(SESSION_DETAILS.voice_call.boss_json or "{}")
        opening_config = _normalize_call_opening_config(
            os.environ.get("OPENING_CONFIG"),
        )

    from unify.coordinator_voice import resolve_runtime_voice

    is_coordinator = (
        bool(meta.get("is_coordinator")) if meta else SESSION_DETAILS.is_coordinator
    )
    voice_provider, voice_id = resolve_runtime_voice(
        is_coordinator=is_coordinator,
        voice_provider=voice_provider,
        voice_id=voice_id,
    )

    # Browser-meet diarization config (Google Meet / Teams Meet)
    meet_session_id: str = ""
    call_session_id: str = ""
    meet_url: str = ""
    meet_agent_service_url: str = ""
    # Per-channel agent-service URL prefix.
    _MEET_PATH_PREFIX = {
        "google_meet": "googlemeet",
        "teams_meet": "teamsmeet",
    }
    meet_path_prefix = _MEET_PATH_PREFIX.get(channel, "")
    if channel in ("google_meet", "teams_meet"):
        if meta:
            meet_session_id = meta.get("meet_session_id", "")
            meet_url = meta.get("meet_url", "")
            meet_agent_service_url = meta.get("agent_service_url", "")
        else:
            meet_session_id = os.environ.get("MEET_SESSION_ID", "")
            meet_url = os.environ.get("MEET_URL", "")
            meet_agent_service_url = os.environ.get("AGENT_SERVICE_URL", "")
    call_session_id = (
        str(meta.get("call_session_id", "")).strip()
        if meta
        else os.environ.get("CALL_SESSION_ID", "").strip()
    )

    _log.config(
        f"voice_provider={voice_provider} voice_id={voice_id} outbound={outbound} channel={channel} opening_mode={opening_config['mode']}",
    )

    _log.session_start("Connecting to room…")
    await ctx.connect()
    _log.session_start("Connected to room")

    # User screen share and webcam capture (subscribe to LiveKit room tracks automatically)
    screen_capture = UserTrackCaptureManager(
        ctx.room,
        track_source="screenshare",
        on_track_change=publish_meet_interaction_from_track,
        fb_logger=_log,
    )
    webcam_capture = UserTrackCaptureManager(
        ctx.room,
        track_source="camera",
        on_track_change=publish_meet_interaction_from_track,
        fb_logger=_log,
    )

    # Flag for call_answered that may arrive during initialization
    call_answered_flag = asyncio.Event()
    user_joined_event = asyncio.Event()
    joined_gate_required = channel in ("phone_call", "whatsapp_call")
    speech_gate_open = not joined_gate_required
    if speech_gate_open:
        user_joined_event.set()

    # Start receiving events from parent (callbacks registered later)
    await start_event_broker_receive()

    # Fallback for whenever pre-loading fails
    if STT is None:
        STT = deepgram.STT(model="nova-3", language="en-GB")
        VAD = silero.VAD.load(min_speech_duration=0.15, min_silence_duration=1.0)

    stt_instance = STT
    if channel in ("google_meet", "teams_meet"):
        stt_instance = deepgram.STT(
            model="nova-3",
            language="en-GB",
            enable_diarization=True,
        )

    # --- Browser-meet speaker + participant tracker (Meet / Teams) ---
    # Speaker identity uses two complementary signals:
    # 1. Deepgram diarization (enable_diarization=True) for precise per-utterance
    #    anonymous speaker IDs (S0, S1, ...).
    # 2. DOM scraping (activeSpeaker) via _meet_poll_loop for display names.
    # The correlation mapping table (_meet_speaker_map) links the two.
    _meet_auth_key = SESSION_DETAILS.unify_key
    _meet_cached_active_speaker: str | None = None
    _meet_cached_participants: list[dict] = []
    _meet_prev_participant_names: set[str] = set()
    _meet_latest_screenshot: str | None = None
    _meet_display_name: str = ""
    _meet_last_speaker_id: str | None = None
    _meet_speaker_map: dict[str, dict[str, int]] = {}
    if channel in ("google_meet", "teams_meet"):
        if meta:
            _meet_display_name = meta.get("meet_display_name", "")
        if not _meet_display_name:
            _meet_display_name = SESSION_DETAILS.assistant.name or "Unity Assistant"

    def _resolve_contact_by_name(display_name: str) -> dict | None:
        """Best-effort contact resolution from a Meet display name.

        Tries an exact first_name+surname match across known contacts
        (the caller contact and the boss). Falls back to None if no match,
        letting the caller use the original contact dict.
        """
        if not display_name:
            return None
        dn_lower = display_name.strip().lower()
        for candidate in (contact, boss):
            full = f"{candidate.get('first_name', '')} {candidate.get('surname', '')}".strip()
            if full.lower() == dn_lower:
                return candidate
            if candidate.get("first_name", "").lower() == dn_lower:
                return candidate
        return None

    def _resolve_speaker() -> tuple[dict, str | None, str | None]:
        """Resolve the current browser-meet speaker to (contact_dict, display_name, speaker_id).

        Uses two signals in priority order:
        1. Diarization speaker_id → correlation mapping (precise, audio-level).
        2. DOM-scraped activeSpeaker name (fallback, 2s polling granularity).
        Returns (contact_dict, display_name_or_None, diarization_speaker_id_or_None).
        """
        if channel not in ("google_meet", "teams_meet"):
            return contact, None, None

        sid = _meet_last_speaker_id

        # Primary: diarization speaker_id → mapped display name
        if sid and sid in _meet_speaker_map:
            votes = _meet_speaker_map[sid]
            if votes:
                top_name = max(votes, key=votes.get)
                top_count = votes[top_name]
                total = sum(votes.values())
                if top_count >= 2 and top_count / total > 0.6:
                    resolved = _resolve_contact_by_name(top_name)
                    if resolved:
                        label = f"{resolved.get('first_name', '')} {resolved.get('surname', '')}".strip()
                        return resolved, label or None, sid
                    return contact, top_name, sid

        # Fallback: DOM active speaker
        active_name = _meet_cached_active_speaker
        if active_name:
            resolved = _resolve_contact_by_name(active_name)
            if resolved:
                label = f"{resolved.get('first_name', '')} {resolved.get('surname', '')}".strip()
                return resolved, label or None, sid
            return contact, active_name, sid

        return contact, None, sid

    def _get_meet_participant_names() -> list[str]:
        """Return display names of all human participants (excluding the assistant)."""
        return [
            p["name"]
            for p in _meet_cached_participants
            if p.get("name") and p["name"] != _meet_display_name
        ]

    # Channel-specific event classes for participant join/leave events.
    if channel == "teams_meet":
        _ParticipantJoinedEvent = TeamsMeetParticipantJoined
        _ParticipantLeftEvent = TeamsMeetParticipantLeft
        _participant_topic = "app:comms:teamsmeet_participant"
    else:
        _ParticipantJoinedEvent = GoogleMeetParticipantJoined
        _ParticipantLeftEvent = GoogleMeetParticipantLeft
        _participant_topic = "app:comms:googlemeet_participant"

    async def _meet_poll_loop() -> None:
        """Background loop: poll agent-service for active speaker + meeting status.

        Two phases:
        1. Discovery — if meet_session_id wasn't provided at dispatch time,
           poll GET /{meet_path_prefix}/sessions and match by meetUrl.
        2. State polling — poll GET /{meet_path_prefix}/state for active speaker,
           participant roster, and meeting-end detection.
        """
        nonlocal _meet_cached_active_speaker, _meet_cached_participants
        nonlocal _meet_prev_participant_names, meet_session_id
        nonlocal _meet_latest_screenshot
        import aiohttp as _aiohttp

        try:
            async with _aiohttp.ClientSession() as http:
                # Phase 1: discover session ID if not provided
                while not meet_session_id:
                    _log.info(
                        f"Discovering {channel} session ID via /{meet_path_prefix}/sessions...",
                    )
                    try:
                        resp = await http.get(
                            f"{meet_agent_service_url}/{meet_path_prefix}/sessions",
                            headers={"authorization": f"Bearer {_meet_auth_key}"},
                            timeout=_aiohttp.ClientTimeout(total=5),
                        )
                        if resp.status == 200:
                            body = await resp.json()
                            for s in body.get("sessions", []):
                                if meet_url and s.get("meetUrl") == meet_url:
                                    meet_session_id = s["sessionId"]
                                    _log.info(
                                        f"Discovered {channel} session ID: {meet_session_id}",
                                    )
                                    break
                    except Exception:
                        pass
                    if not meet_session_id:
                        await asyncio.sleep(1)

                # Phase 2: poll state for active speaker + meeting end
                while True:
                    await asyncio.sleep(1)
                    try:
                        resp = await http.get(
                            f"{meet_agent_service_url}/{meet_path_prefix}/state",
                            params={"sessionId": meet_session_id},
                            headers={"authorization": f"Bearer {_meet_auth_key}"},
                            timeout=_aiohttp.ClientTimeout(total=5),
                        )
                        if resp.status != 200:
                            continue
                        body = await resp.json()
                    except Exception:
                        continue

                    _meet_cached_active_speaker = body.get("activeSpeaker")
                    _meet_cached_participants = body.get("participants", [])

                    # Diff roster for join/leave notifications
                    current_names = {
                        p["name"] for p in _meet_cached_participants if p.get("name")
                    }
                    joined = current_names - _meet_prev_participant_names
                    left = _meet_prev_participant_names - current_names
                    _meet_prev_participant_names = current_names

                    for name in joined:
                        if name == _meet_display_name:
                            continue
                        evt = _ParticipantJoinedEvent(
                            contact=contact,
                            participant_name=name,
                        )
                        asyncio.create_task(
                            event_broker.publish(
                                _participant_topic,
                                evt.to_json(),
                            ),
                        )

                    for name in left:
                        if name == _meet_display_name:
                            continue
                        evt = _ParticipantLeftEvent(
                            contact=contact,
                            participant_name=name,
                        )
                        asyncio.create_task(
                            event_broker.publish(
                                _participant_topic,
                                evt.to_json(),
                            ),
                        )

                    # Fetch cached screenshot (non-blocking — agent-service
                    # captures during its own poll cycle)
                    try:
                        ss_resp = await http.get(
                            f"{meet_agent_service_url}/{meet_path_prefix}/screenshot/latest",
                            params={"sessionId": meet_session_id},
                            headers={"authorization": f"Bearer {_meet_auth_key}"},
                            timeout=_aiohttp.ClientTimeout(total=5),
                        )
                        if ss_resp.status == 200:
                            ss_body = await ss_resp.json()
                            _meet_latest_screenshot = ss_body.get("screenshot")
                    except Exception:
                        pass

                    status = body.get("status", "")
                    if status in ("ended", "removed", "error"):
                        _log.info(f"{channel} ended (status={status})")
                        ctx.shutdown(reason=f"meet_{status}")
                        return
        except asyncio.CancelledError:
            pass

    if channel in ("google_meet", "teams_meet"):
        asyncio.create_task(_meet_poll_loop())

    from unify.settings import SETTINGS

    # Fast brain LLM - lightweight model for responsive conversation
    # Uses UnifyLLM adapter for local caching (CI) and usage tracking
    llm_model = UnifyLLM(
        model=SETTINGS.conversation.FAST_BRAIN_MODEL,
        reasoning_effort="low",
    )

    assistant_name = SESSION_DETAILS.assistant.name
    # The acting user on this call is the person we're talking with when they map
    # to a system user (boss or provisioned org member), else the workspace owner.
    # Drives per-speaker linked-desktop resolution so the guardrail relaxes only
    # for someone who has actually linked their machine to this assistant.
    call_acting_user_id = (
        contact.get("user_id") if contact.get("is_system") else None
    ) or SESSION_DETAILS.user.id
    call_has_linked_user_desktop = (
        SESSION_DETAILS.assistant.user_desktop_for(call_acting_user_id) is not None
    )

    system_prompt = build_voice_agent_prompt(
        bio=assistant_bio,
        assistant_name=assistant_name or None,
        boss_first_name=boss.get("first_name", ""),
        boss_surname=boss.get("surname", ""),
        boss_email_address=boss.get("email_address", ""),
        boss_phone_number=boss.get("phone_number", ""),
        boss_bio=boss.get("bio") or None,
        contact_first_name=contact.get("first_name", ""),
        contact_surname=contact.get("surname", ""),
        contact_phone_number=contact.get("phone_number", ""),
        contact_email=contact.get("email_address", ""),
        contact_bio=contact.get("bio") or None,
        is_boss_user=bool(contact.get("is_system", False)),
        contact_rolling_summary=contact.get("rolling_summary", ""),
        demo_mode=SETTINGS.DEMO_MODE,
        channel=channel,
        has_linked_user_desktop=call_has_linked_user_desktop,
        is_coordinator=SESSION_DETAILS.is_coordinator,
        is_org_workspace=SESSION_DETAILS.org_id is not None,
        console_ui_present=SETTINGS.UNITY_CONSOLE_UI,
    ).flatten()
    _log.config(f"System prompt ({len(system_prompt)} chars)")

    if voice_provider == "elevenlabs":
        tts_instance = elevenlabs.TTS(
            voice_id=voice_id or elevenlabs.DEFAULT_VOICE_ID,
            model="eleven_multilingual_v2",
        )
    else:
        tts_instance = cartesia.TTS(
            voice=voice_id or cartesia.tts.TTSDefaultVoiceId,
        )

    session = AgentSession(
        llm=llm_model,
        stt=stt_instance,
        tts=tts_instance,
        vad=VAD,
        turn_handling={
            "turn_detection": EnglishModel(),
            "endpointing": {"min_delay": 0.75},
            "interruption": {"enabled": True},
        },
        preemptive_generation=False,
    )

    user_is_speaking = False
    _queued_speech: list[tuple[str, str, str, str, str]] = []
    _say_meta_queue: list[dict] = []
    _recorded_opening_preloaded: dict[str, _PreloadedAudio] = {}
    generation_seq = 0
    user_state_seq = 0
    mood_turn_index = 0
    mood_classification_queue: asyncio.Queue[dict] = asyncio.Queue()
    mood_classification_task: asyncio.Task | None = None
    _was_quiescent = True
    _pending_reply_timer: asyncio.TimerHandle | None = None
    _NOTIFY_COALESCE_S = 0.05

    def _log_reply_task(task: asyncio.Task) -> None:
        try:
            task.result()
            _log.llm_completed()
        except asyncio.CancelledError:
            _log.llm_cancelled()
        except Exception as exc:  # noqa: BLE001
            _log.llm_error(str(exc))

    def _say_opening(text: str) -> None:
        speech_handle = session.say(
            text,
            allow_interruptions=True,
            add_to_chat_ctx=True,
        )
        # A briefed/speak opener is live TTS (not pre-recorded), so a barge-in
        # can be resumed by the fast brain just like any slow-brain line.
        _register_interruptible_tts(speech_handle, lambda: text, "opening")

    def _say_recorded_opening(text: str, recording_source: str):
        _say_meta_queue.append(
            {
                "source": opening_config.get("source", "recorded_opening"),
                "text": text,
                "llm_log_path": "",
            },
        )
        return session.say(
            text,
            audio=_recorded_opening_audio(
                recording_source,
                _recorded_opening_preloaded,
            ),
            allow_interruptions=True,
            add_to_chat_ctx=True,
        )

    def _schedule_opening_bridge(segment: dict) -> None:
        """Enqueue the opener static-removal bridge without blocking on playout.

        Scheduling (``session.say``) happens synchronously so the bridge is
        queued ahead of the user-turn reply the framework generates next. We
        deliberately do NOT await playout: the reply is generated concurrently
        while the bridge plays, then plays after it (same speech priority,
        FIFO — a newly scheduled reply does not interrupt in-progress speech).
        The bridge text remains in ``_say_meta_queue`` until its playout
        commits to history, so the concurrent generation sees it as in-flight
        speech and continues naturally from it.
        """
        text = segment["transcript"]
        _say_meta_queue.append(
            {
                "source": "recorded_opening_bridge",
                "text": text,
                "llm_log_path": "",
            },
        )
        session.say(
            text,
            audio=_recorded_opening_audio(
                f"asset://{segment['asset']}",
                _recorded_opening_preloaded,
            ),
            allow_interruptions=True,
            add_to_chat_ctx=True,
        )

    async def _run_recorded_opening(config: dict) -> None:
        asset_key = config.get("recording_asset", "").strip()
        spec = _RECORDED_OPENINGS.get(asset_key)
        if spec is None:
            handle = _say_recorded_opening(
                _recorded_opening_transcript(config),
                _recorded_opening_source(config),
            )
            await handle.wait_for_playout()
            return

        bridge = spec.get("bridge")
        timed_chunks = spec["timed_chunks"]
        full_transcript = _recorded_opening_timed_transcript(timed_chunks)
        _say_meta_queue.append(
            {
                "source": opening_config.get("source", "recorded_opening"),
                "text": full_transcript,
                "llm_log_path": "",
            },
        )
        handle = session.say(
            _timed_opening_text(timed_chunks),
            audio=_recorded_opening_audio(
                f"asset://{spec['asset']}",
                _recorded_opening_preloaded,
            ),
            allow_interruptions=True,
            add_to_chat_ctx=True,
        )
        await handle.wait_for_playout()
        if handle.interrupted:
            spoken = _spoken_text_from_handle(handle).strip()
            if "Much better." not in spoken and bridge is not None:
                assistant._pending_opening_bridge = (
                    lambda b=bridge: _schedule_opening_bridge(b)
                )

    def _fire_generate_reply(
        reason: str,
        source_id: str,
        allow_interruptions: bool = True,
        user_input: str | None = None,
    ):
        nonlocal generation_seq, _pending_reply_timer
        _pending_reply_timer = None
        generation_seq += 1
        generation_id = f"gen-{generation_seq:06d}"
        last_role = (
            assistant._chat_ctx.items[-1].role if assistant._chat_ctx.items else "none"
        )
        trigger = {
            "generation_id": generation_id,
            "reason": reason,
            "source_id": source_id,
            "user_is_speaking": user_is_speaking,
            "last_chat_role": last_role,
            "ts_utc": now_utc_iso(),
            "monotonic_ms": monotonic_ms(),
        }
        enqueue_trace_context = getattr(llm_model, "enqueue_trace_context", None)
        if callable(enqueue_trace_context):
            enqueue_trace_context(trigger)
        _log.llm_thinking(
            reason=reason,
            queued_speech=len(_queued_speech),
        )
        reply_kwargs = {"allow_interruptions": allow_interruptions}
        if user_input is not None:
            reply_kwargs["user_input"] = user_input
        maybe_result = session.generate_reply(**reply_kwargs)
        if isinstance(maybe_result, asyncio.Task):
            maybe_result.add_done_callback(_log_reply_task)
        return maybe_result

    def trigger_generate_reply(
        reason: str,
        source_id: str,
        *,
        allow_interruptions: bool = True,
        wait_for_completion: bool = False,
        user_input: str | None = None,
    ):
        nonlocal _pending_reply_timer
        if _pending_reply_timer is not None:
            _pending_reply_timer.cancel()
            _pending_reply_timer = None

        if wait_for_completion:
            return _fire_generate_reply(
                reason,
                source_id,
                allow_interruptions,
                user_input,
            )

        loop = asyncio.get_event_loop()
        _pending_reply_timer = loop.call_later(
            _NOTIFY_COALESCE_S,
            _fire_generate_reply,
            reason,
            source_id,
            allow_interruptions,
            user_input,
        )

    def _invalidate_current_generation(reason: str, source_id: str) -> None:
        """Cancel in-flight FastBrain generation and re-trigger with updated context.

        Called when a significant IPC event (slow brain notification, outbound
        message confirmation) arrives while the FastBrain LLM is mid-generation.
        The 50 ms coalescence in ``trigger_generate_reply`` naturally collapses
        bursts (e.g. notification + message_sent arriving ~100 ms apart) into a
        single regeneration.
        """
        if not assistant.user_turn_generating:
            return
        _log.info(f"Invalidating in-flight generation: {reason}")
        session.interrupt()
        trigger_generate_reply(reason=reason, source_id=source_id)

    if channel == "phone_call":
        user_utterance_event = InboundPhoneUtterance
        assistant_utterance_event = OutboundPhoneUtterance
    elif channel == "whatsapp_call":
        user_utterance_event = InboundWhatsAppCallUtterance
        assistant_utterance_event = OutboundWhatsAppCallUtterance
    elif channel == "google_meet":
        user_utterance_event = InboundGoogleMeetUtterance
        assistant_utterance_event = OutboundGoogleMeetUtterance
    elif channel == "teams_meet":
        user_utterance_event = InboundTeamsMeetUtterance
        assistant_utterance_event = OutboundTeamsMeetUtterance
    else:
        user_utterance_event = InboundUnifyMeetUtterance
        assistant_utterance_event = OutboundUnifyMeetUtterance

    async def _publish_assistant_utterance(text: str) -> None:
        if channel == "google_meet":
            event = OutboundGoogleMeetUtterance(
                contact=contact,
                content=text,
                participant_names=_get_meet_participant_names() or None,
            )
        elif channel == "teams_meet":
            event = OutboundTeamsMeetUtterance(
                contact=contact,
                content=text,
                participant_names=_get_meet_participant_names() or None,
            )
        else:
            event = assistant_utterance_event(contact, content=text)
        await event_broker.publish(
            f"app:comms:{channel}_utterance",
            event.to_json(),
        )

    credit_gate_task: asyncio.Task | None = None
    explicit_stop_requested = False
    shutdown_completed = False

    # Register cleanup as a LiveKit shutdown callback so it runs on any
    # exit path: participant disconnect or explicit stop.
    async def _on_job_shutdown():
        nonlocal shutdown_completed
        if shutdown_completed:
            return
        shutdown_completed = True
        if credit_gate_task is not None:
            await utils.aio.cancel_and_wait(credit_gate_task)
        if mood_classification_task is not None:
            await utils.aio.cancel_and_wait(mood_classification_task)
        if audio_bridge is not None:
            await asyncio.to_thread(audio_bridge.stop)
        await screen_capture.close()
        await webcam_capture.close()
        if channel != "unify_meet" or explicit_stop_requested:
            await delete_livekit_room(ctx.room.name)
        await publish_call_ended(contact, channel, call_session_id=call_session_id)

    ctx.add_shutdown_callback(_on_job_shutdown)

    # Bridge AgentSession close → job shutdown.  close_on_disconnect
    # (RoomInputOptions, default True) closes the AgentSession when the
    # linked participant leaves, but does NOT resolve the JobContext's
    # shutdown future — so our shutdown callbacks never fire.  Listening
    # for the session "close" event completes the chain.
    @session.on("close")
    def _on_session_close(ev):
        from livekit.agents.voice.events import CloseReason

        if ev.reason == CloseReason.PARTICIPANT_DISCONNECTED:
            ctx.shutdown(reason="participant_disconnected")

    def _check_quiescence_transition() -> None:
        nonlocal _was_quiescent
        now_quiescent = _is_pipeline_quiescent()
        if now_quiescent != _was_quiescent:
            _was_quiescent = now_quiescent
            import json as _json

            asyncio.create_task(
                event_broker.publish(
                    "app:comms:pipeline_quiescent",
                    _json.dumps({"quiescent": now_quiescent}),
                ),
            )

    @session.on("user_state_changed")
    def _on_user_state_changed(ev):
        nonlocal user_is_speaking, user_state_seq
        user_state_seq += 1
        state_id = f"usrstate-{user_state_seq:06d}"
        user_is_speaking = ev.new_state == "speaking"
        if not user_is_speaking:
            # The user just freed the floor: a queued slow-brain line should play
            # at the next silent moment, not wait for the next agent-state cycle.
            maybe_speak_queued()
        _log.user_state(ev.new_state, state_id=state_id)
        _check_quiescence_transition()

    @session.on("agent_state_changed")
    def _on_agent_state_changed(ev):
        """Try queued speech only after the agent settles into a quiescent state.

        We intentionally do NOT trigger from user_state_changed because there is
        a gap between VAD silence detection and the turn detector confirming the
        turn. During that gap, agent_state is still "listening" and current_speech
        is None — firing then would race ahead of the fast brain's reply.

        Triggering here guarantees the full thinking → speaking → listening cycle
        has completed before queued notification speech plays.
        """
        if ev.new_state in ("listening", "idle"):
            maybe_speak_queued()
        _check_quiescence_transition()

    # -- Diarization ↔ DOM speaker correlation --
    if channel in ("google_meet", "teams_meet"):

        @session.on("user_input_transcribed")
        def _on_user_input_transcribed(ev):
            nonlocal _meet_last_speaker_id
            if not ev.is_final or not ev.speaker_id:
                return
            _meet_last_speaker_id = ev.speaker_id
            dom_speaker = _meet_cached_active_speaker
            if dom_speaker and dom_speaker != _meet_display_name:
                bucket = _meet_speaker_map.setdefault(ev.speaker_id, {})
                bucket[dom_speaker] = bucket.get(dom_speaker, 0) + 1

    # -- Screenshot state --
    screenshot_history = ScreenshotHistory()
    assistant_screen_share_active = False
    user_remote_control_active = False
    _agent_service_url: str | None = None
    _visual_ctx_msg_id: str | None = None
    import aiohttp as _aiohttp

    _screenshot_http_session = _aiohttp.ClientSession()

    def _clear_visual_context(source: str | None = None) -> None:
        """Remove visual context from chat contexts and clear screenshot history."""
        nonlocal _visual_ctx_msg_id
        screenshot_history.clear(source=source)
        if not screenshot_history.build_visual_context_content():
            for ctx in (assistant._chat_ctx, session.history):
                if _visual_ctx_msg_id is not None:
                    idx = ctx.index_by_id(_visual_ctx_msg_id)
                    if idx is not None:
                        ctx.items.pop(idx)
            _visual_ctx_msg_id = None

    def _inject_visual_context() -> None:
        """Replace the visual context system message in the chat context."""
        nonlocal _visual_ctx_msg_id
        content = screenshot_history.build_visual_context_content()
        if not content:
            return
        # Remove the previous visual context message if present.
        for ctx in (assistant._chat_ctx, session.history):
            if _visual_ctx_msg_id is not None:
                idx = ctx.index_by_id(_visual_ctx_msg_id)
                if idx is not None:
                    ctx.items.pop(idx)
        msg = assistant._chat_ctx.add_message(role="user", content=content)
        session.history.add_message(
            role="user",
            content=content,
            id=msg.id,
        )
        _visual_ctx_msg_id = msg.id

    def _publish_screenshot(entry: ScreenshotEntry, filepath: str) -> None:
        """Fire-and-forget: write to disk and publish to slow brain via IPC."""

        async def _background():
            await asyncio.to_thread(write_screenshot_to_disk, entry, filepath)
            await event_broker.publish(
                "app:comms:screenshot",
                json.dumps(
                    {
                        "b64": entry.b64,
                        "utterance": entry.utterance,
                        "timestamp": entry.timestamp.isoformat(),
                        "source": entry.source,
                        "filepath": filepath,
                    },
                ),
            )

        asyncio.create_task(_background())

    def _handle_screenshot(entry: ScreenshotEntry) -> None:
        """Process a captured screenshot: history, visual context, disk, IPC."""
        filepath = generate_screenshot_path(entry)
        screenshot_history.add(entry, filepath)
        _inject_visual_context()
        if entry.source != "assistant":
            _publish_screenshot(entry, filepath)

    async def _refresh_screenshots() -> None:
        """Capture fresh screenshots from all active sources and update visual context.

        Called before any module that needs the latest visual state (e.g., the
        notification reply evaluator, the fast-brain LLM).  Sync captures
        (user screen, webcam) are ~1 ms.  The assistant capture reads from the
        agent-service screenshot cache (~0 ms) unless the user has remote
        control, in which case a live capture (~500 ms) is used.
        """
        from datetime import datetime, timezone

        if screen_capture._latest_frame_data is not None:
            b64 = screen_capture.capture_screenshot()
            if b64:
                _handle_screenshot(
                    ScreenshotEntry(
                        b64=b64,
                        utterance="",
                        timestamp=datetime.now(timezone.utc),
                        source="user",
                    ),
                )

        if webcam_capture._latest_frame_data is not None:
            b64 = webcam_capture.capture_screenshot()
            if b64:
                _handle_screenshot(
                    ScreenshotEntry(
                        b64=b64,
                        utterance="",
                        timestamp=datetime.now(timezone.utc),
                        source="webcam",
                    ),
                )

        if assistant_screen_share_active:
            entry = await capture_assistant_screenshot(
                utterance="",
                cached=not user_remote_control_active,
                fb_logger=_log,
                agent_service_url=_agent_service_url,
                http_session=_screenshot_http_session,
            )
            if entry and assistant_screen_share_active:
                _handle_screenshot(entry)

        if channel in ("google_meet", "teams_meet") and _meet_latest_screenshot:
            _handle_screenshot(
                ScreenshotEntry(
                    b64=_meet_latest_screenshot,
                    utterance="",
                    timestamp=datetime.now(timezone.utc),
                    source=channel,
                ),
            )

    def _fast_brain_text_transcript() -> str:
        lines: list[str] = []
        for item in session.history.items:
            role = getattr(item, "role", None)
            if role not in ("user", "assistant"):
                continue
            text = (getattr(item, "text_content", None) or "").strip()
            if not text:
                continue
            speaker = "User" if role == "user" else "Assistant"
            lines.append(f"{speaker}: {text}")
        return "\n".join(lines)

    def _mood_classification_enabled() -> bool:
        return (
            SETTINGS.conversation.FAST_BRAIN_MOOD_CLASSIFICATION_ENABLED
            and has_video_avatar_channel(channel)
        )

    async def _run_mood_classifications() -> None:
        from unify.conversation_manager.domains.fast_brain_mood import (
            FastBrainMoodClassifier,
        )

        classifier = FastBrainMoodClassifier(
            SETTINGS.conversation.FAST_BRAIN_MOOD_CLASSIFICATION_MODEL,
        )
        while True:
            item = await mood_classification_queue.get()
            try:
                try:
                    classification = await classifier.evaluate(
                        transcript=item["transcript"],
                        trigger_role=item["trigger_role"],
                        trigger_text=item["trigger_text"],
                    )
                    if classification is None:
                        continue

                    mood = classification.mood.value
                    avatar_mood = classification.avatar_mood
                    await ctx.room.local_participant.publish_data(
                        json.dumps(
                            {
                                "type": "mood_classification",
                                "mood": mood,
                                "avatarMood": avatar_mood,
                                "turnIndex": item["turn_index"],
                                "triggerRole": item["trigger_role"],
                            },
                        ).encode(),
                        topic="agent_status",
                        reliable=True,
                    )
                    event = FastBrainMoodClassified(
                        contact=contact,
                        channel=channel,
                        mood=mood,
                        avatar_mood=avatar_mood,
                        trigger_role=item["trigger_role"],
                        trigger_utterance_id=item["trigger_utterance_id"],
                        turn_index=item["turn_index"],
                        model=SETTINGS.conversation.FAST_BRAIN_MOOD_CLASSIFICATION_MODEL,
                    )
                    await event_broker.publish(event.topic, event.to_json())
                except Exception as e:
                    _log.error(f"Mood classification failed: {e}")
            finally:
                mood_classification_queue.task_done()

    def _enqueue_mood_classification(
        role: str,
        text: str,
        utterance_id: str,
    ) -> None:
        nonlocal mood_turn_index
        if not _mood_classification_enabled():
            return
        if not text.strip():
            return
        mood_turn_index += 1
        mood_classification_queue.put_nowait(
            {
                "transcript": _fast_brain_text_transcript(),
                "trigger_role": role,
                "trigger_text": text,
                "trigger_utterance_id": utterance_id,
                "turn_index": mood_turn_index,
            },
        )

    @session.on("conversation_item_added")
    def _on_chat_item_added(ev):
        """Publish both user and assistant utterances from a single location."""
        role = getattr(ev.item, "role", None)
        if role not in ("user", "assistant"):
            return
        text = ev.item.text_content or ""
        utterance_id = content_trace_id("utt", f"{role}:{text}")
        say_meta: dict | None = None
        if role == "assistant" and _say_meta_queue:
            for i, candidate in enumerate(_say_meta_queue):
                if match_say_meta(candidate, text):
                    say_meta = _say_meta_queue.pop(i)
                    break
        if role == "user":
            if not assistant._user_speech_logged:
                _log.user_speech(text)
            assistant._user_speech_logged = False
        else:
            source = (say_meta or {}).get("source", "reply")
            if say_meta and say_meta.get("llm_log_path"):
                log_path = say_meta["llm_log_path"]
            elif source == "reply":
                log_path = getattr(llm_model, "last_log_path", "")
            else:
                log_path = ""
            _log.assistant_speech(text, source=source, llm_log_path=log_path)
        if role == "user":
            from datetime import datetime, timezone

            b64 = screen_capture.capture_screenshot()
            if b64:
                _handle_screenshot(
                    ScreenshotEntry(
                        b64=b64,
                        utterance=text,
                        timestamp=datetime.now(timezone.utc),
                        source="user",
                    ),
                )
            webcam_b64 = webcam_capture.capture_screenshot()
            if webcam_b64:
                _handle_screenshot(
                    ScreenshotEntry(
                        b64=webcam_b64,
                        utterance=text,
                        timestamp=datetime.now(timezone.utc),
                        source="webcam",
                    ),
                )
            if channel in ("google_meet", "teams_meet") and _meet_latest_screenshot:
                _handle_screenshot(
                    ScreenshotEntry(
                        b64=_meet_latest_screenshot,
                        utterance=text,
                        timestamp=datetime.now(timezone.utc),
                        source=channel,
                    ),
                )

            async def _publish_user_utterance(text: str) -> None:
                nonlocal _meet_last_speaker_id
                resolved_contact, speaker_label, dia_sid = _resolve_speaker()
                _meet_last_speaker_id = None
                # Stamp the current turn so the slow-brain run it spawns can be
                # cancelled precisely if the fast brain resolves this turn itself.
                turn_id = assistant._user_turn_seq
                if channel == "google_meet":
                    event = InboundGoogleMeetUtterance(
                        contact=resolved_contact,
                        content=text,
                        speaker_label=speaker_label,
                        participant_names=_get_meet_participant_names() or None,
                        diarization_speaker_id=dia_sid,
                        turn_id=turn_id,
                    )
                elif channel == "teams_meet":
                    event = InboundTeamsMeetUtterance(
                        contact=resolved_contact,
                        content=text,
                        speaker_label=speaker_label,
                        participant_names=_get_meet_participant_names() or None,
                        diarization_speaker_id=dia_sid,
                        turn_id=turn_id,
                    )
                else:
                    event = user_utterance_event(
                        resolved_contact,
                        content=text,
                        turn_id=turn_id,
                    )
                await event_broker.publish(
                    f"app:comms:{channel}_utterance",
                    event.to_json(),
                )

            # While the outbound opener is pending, the callee's first utterance
            # (their "Hello?") only triggers the opener — do not publish it as a
            # turn to the slow brain, which would otherwise compose a competing
            # reply alongside the opener. It carries no content the opener needs.
            if not assistant._opening_pending:
                asyncio.create_task(
                    _publish_user_utterance(text),
                )
        else:
            asyncio.create_task(_publish_assistant_utterance(text))
        _enqueue_mood_classification(role, text, utterance_id)

    audio_bridge: MeetAudioBridge | None = None
    if channel in ("google_meet", "teams_meet"):
        audio_bridge = MeetAudioBridge()
        audio_bridge.start(asyncio.get_event_loop())

    assistant = Assistant(
        contact=contact,
        boss=boss,
        channel=channel,
        instructions=system_prompt,
        outbound=outbound,
        audio_bridge=audio_bridge,
        normalize_elevenlabs_twin_pronunciation=voice_provider == "elevenlabs",
    )
    credit_gate_monitor = FastBrainCreditGateMonitor()
    assistant.set_credit_gate_state_provider(lambda: credit_gate_monitor.state)
    # In-flight says (proactive/guidance still playing, not yet committed) live
    # in _say_meta_queue until their playout commits them to history. Set as a
    # direct attribute so it works uniformly on the real Assistant and the test
    # fakes without each needing a setter.
    credit_gate_task = asyncio.create_task(
        credit_gate_monitor.run(),
        name="fast_brain_credit_gate_monitor",
    )
    if _mood_classification_enabled():
        mood_classification_task = asyncio.create_task(
            _run_mood_classifications(),
            name="fast_brain_mood_classifier",
        )

    async def _capture_screenshots_for_llm(chat_ctx) -> None:
        """Capture fresh screenshots and inject them into the LLM's chat_ctx.

        The LiveKit pipeline passes a **copy** of the chat context to
        ``llm_node``.  ``_refresh_screenshots`` updates the live
        ``session.history`` (for subsequent turns and IPC), but that copy
        is stale.  After refreshing, we rebuild the visual context content
        and inject it directly into the ``chat_ctx`` parameter so the
        current LLM call sees the screenshot.
        """
        try:
            copy_visual_id = _visual_ctx_msg_id

            await _refresh_screenshots()

            content = screenshot_history.build_visual_context_content()
            if content:
                if copy_visual_id is not None:
                    idx = chat_ctx.index_by_id(copy_visual_id)
                    if idx is not None:
                        chat_ctx.items.pop(idx)
                msg = chat_ctx.add_message(role="user", content=content)
                chat_ctx.items.pop()
                chat_ctx.items.insert(-1, msg)
        except Exception as e:
            print(f"[llm_node] screenshot capture error (non-fatal): {e}")

    assistant._capture_screenshots_for_llm = _capture_screenshots_for_llm

    rio = RoomInputOptions(
        noise_cancellation=(
            noise_cancellation.BVC() if sys.platform == "darwin" else None
        ),
        close_on_disconnect=(
            channel not in ("google_meet", "teams_meet", "unify_meet")
        ),
    )

    # Publish call started (shared helper)
    await publish_call_started(contact, channel, call_session_id=call_session_id)

    pending_notifications: list[tuple[str, str, bool, str, str, str]] = (
        []
    )  # (message, spoken_message, should_speak, notification_id, notification_source, llm_log_path)
    session_ready = False

    def _mark_user_joined(reason: str) -> None:
        nonlocal speech_gate_open
        if user_joined_event.is_set():
            return
        _log.call_status(f"user_joined:{reason}")
        speech_gate_open = True
        user_joined_event.set()
        assistant.set_call_received()

    async def _graceful_meet_stop() -> None:
        """End a Unify Meet by letting the Console disconnect itself first.

        Publishing ``call_ended`` prompts the browser to call ``room.disconnect()``
        — a clean, client-initiated WebRTC teardown that closes its data channels
        via ``onclose`` — before the agent shuts down and the room is deleted.
        That avoids the abrupt server-side eviction that otherwise fires the
        browser's ``RTCDataChannel.onerror`` ("Unknown DataChannel error").
        """
        try:
            await ctx.room.local_participant.publish_data(
                json.dumps({"type": "call_ended"}).encode(),
                topic="agent_status",
                reliable=True,
            )
        except Exception as exc:
            _log.call_status(f"call_ended publish failed: {exc}")
        await asyncio.sleep(MEET_GRACEFUL_LEAVE_GRACE_S)
        ctx.shutdown(reason="stopped")

    def on_status(data: dict) -> None:
        """Handle status events (call_answered, stop, meet_session_id)."""
        nonlocal explicit_stop_requested, meet_session_id, speech_gate_open
        event_type = data.get("type", "")
        _log.call_status(event_type)

        if event_type == "call_answered":
            call_answered_flag.set()
            _mark_user_joined("call_answered")
        elif event_type in ("meet_session_id", "gmeet_session_id"):
            meet_session_id = data.get("session_id", "")
        elif event_type == "stop":
            explicit_stop_requested = True
            if channel == "unify_meet":
                asyncio.create_task(_graceful_meet_stop())
            else:
                ctx.shutdown(reason="stopped")

    @ctx.room.on("participant_connected")
    def _on_room_participant_connected(participant):
        if joined_gate_required and not outbound:
            _mark_user_joined("participant_connected")

    def _is_pipeline_quiescent() -> bool:
        """True when the voice pipeline is completely idle (no speech in flight)."""
        if user_is_speaking:
            return False
        if session.agent_state not in ("listening", "idle"):
            return False
        current = session.current_speech
        if current is not None and not current.done:
            return False
        return True

    def _queued_speech_block_reason() -> str:
        """Why a queued slow-brain line cannot play yet, or "" if the floor is free.

        Deliberately narrower than ``_is_pipeline_quiescent``: a ready line is held
        ONLY while someone occupies the floor (the user speaking, or assistant audio
        actually playing). The agent merely "thinking" (generating a reply) does not
        block it - filler/answer ordering is handled by the ``_slow_brain_responded_turn``
        suppression in ``llm_node``, not by withholding the real line.
        """
        if user_is_speaking:
            return "user_speaking"
        current = session.current_speech
        if current is not None and not current.done:
            return "assistant_speaking"
        return ""

    def _spoken_text_from_handle(handle: object) -> str:
        """Concatenate the assistant text actually persisted for a say handle.

        On interruption LiveKit records only the synchronized (actually-spoken)
        transcript, so this is the prefix the caller really heard.
        """
        items = getattr(handle, "chat_items", None) or []
        texts: list[str] = []
        for item in items:
            if getattr(item, "role", None) != "assistant":
                continue
            content = getattr(item, "text_content", None)
            if content:
                texts.append(content)
        return " ".join(texts)

    async def _publish_voice_interrupt(spoken: str, remainder: str) -> None:
        """Hand an unheard remainder to the slow brain to re-surface next turn."""
        await event_broker.publish(
            VoiceInterrupt.topic,
            VoiceInterrupt(
                contact=contact,
                spoken_prefix=spoken,
                unheard_remainder=remainder,
            ).to_json(),
        )
        from unify.logger import LOGGER

        LOGGER.info(
            "⬥ [FastBrain] Reported unheard remainder to the slow brain.",
        )

    # Expose to Assistant.llm_node (a class method, outside this closure) so it
    # can hand off when it decides not to resume an interrupted line itself.
    assistant._publish_voice_interrupt = _publish_voice_interrupt

    async def _publish_fast_brain_continued(turn_id: int | None = None) -> None:
        """Cancel the slow-brain run for ``turn_id`` when the fast brain answers.

        The slow brain is triggered the moment the user's utterance lands; if the
        fast brain then resolves that same turn itself - resuming the interrupted
        line or fully answering a small-talk turn - its run would also answer, so
        the CM cancels exactly that turn's run (wherever it sits in the queue).
        The fast brain (~1s) always beats the slow brain (~8-12s), so the run is
        still mid-flight and is dropped before producing speech.
        """
        await event_broker.publish(
            FastBrainContinued.topic,
            FastBrainContinued(contact=contact, turn_id=turn_id).to_json(),
        )

    assistant._publish_fast_brain_continued = _publish_fast_brain_continued

    def _register_interruptible_tts(
        handle: object,
        full_text_getter,
        notification_source: str,
    ) -> None:
        """Register an in-flight TTS line so a barge-in can be resumed.

        The slow brain owns all substantive speech; when the caller interrupts,
        only the spoken prefix lands in the transcript, so the missed remainder
        would otherwise be lost. On interruption we stash the remainder as a
        claimable continuation candidate. The fast brain is the single front
        door: the next ``llm_node`` turn always decides what happens to it -
        resume it verbatim (CONTINUE, the heavy default), or hand it to the slow
        brain (DEFER -> ``VoiceInterrupt``). A barge-in that produced no
        transcript is resumed automatically. There is no timer: the candidate
        simply waits for that decision, so the fast brain can never lose the race.

        A fast-brain continuation is itself registered here (the ``speech_created``
        observer hands its reply handle to this function), so interrupting a
        resumed line re-stashes a fresh candidate, making continuation recursive
        to arbitrary depth.

        Pre-recorded openings are never passed here, so their hand-crafted tone
        is never continued by the live voice. Proactive silence-filler is never
        resumed or reported.
        """
        if notification_source == "proactive_speech":
            return

        assistant._tts_seq += 1
        seq = assistant._tts_seq
        assistant._active_tts = {
            "handle": handle,
            "source": notification_source,
            "seq": seq,
        }

        async def _after_playout() -> None:
            try:
                await handle.wait_for_playout()
            except Exception:
                return
            if not getattr(handle, "interrupted", False):
                return
            full = (full_text_getter() or "").strip()
            if not full:
                return
            spoken = _spoken_text_from_handle(handle).strip()
            remainder = full
            if spoken and full.startswith(spoken):
                remainder = full[len(spoken) :].strip()
            if not remainder:
                return
            resume_text = compute_resume_text(full, spoken) or remainder
            assistant._pending_continuation = {
                "resume_text": resume_text,
                "remainder": remainder,
                "spoken_prefix": spoken,
                "source": notification_source,
                "seq": seq,
                "consumed": False,
            }

        asyncio.ensure_future(_after_playout())

    @session.on("speech_created")
    def _on_speech_created(ev) -> None:
        """Register a fast-brain continuation reply for interruption-stashing.

        Slow-brain lines and live openings register themselves at ``session.say``
        time. A fast-brain continuation is delivered as a ``generate_reply``
        reply (yielded from ``llm_node``), so it is registered here instead, using
        the full text the turn stashed on ``_continuation_full_text``. Ordinary
        buffer fillers leave that ``None`` and are never registered.
        """
        if getattr(ev, "source", "") != "generate_reply":
            return
        full_text = assistant._continuation_full_text
        assistant._continuation_full_text = None
        if not full_text:
            return
        _register_interruptible_tts(
            ev.speech_handle,
            lambda t=full_text: t,
            "continuation",
        )

    def _speak_now(
        text: "str | AsyncIterable[str]",
        notification_id: str,
        notification_source: str,
        notification_content: str,
        llm_log_path: str,
    ) -> None:
        if isinstance(text, str):
            _say_meta_queue.append(
                {
                    "notification_id": notification_id,
                    "source": notification_source,
                    "text": text,
                    "llm_log_path": llm_log_path,
                },
            )
            _log.notification_say(text, notification_source=notification_source)
            handle = session.say(text, allow_interruptions=True, add_to_chat_ctx=True)
            _register_interruptible_tts(
                handle,
                lambda: text,
                notification_source,
            )
            return

        # Streaming path: ``text`` is an async iterator of token chunks (rewritten
        # speech). The chunks are forwarded to TTS as they arrive so playout starts
        # on the first token. ``say_meta["text"]`` is kept in sync so the
        # ``conversation_item_added`` handler (which fires at playout end) can match
        # the assembled utterance via its prefix.
        say_meta = {
            "notification_id": notification_id,
            "source": notification_source,
            "text": "",
            "llm_log_path": llm_log_path,
        }
        _say_meta_queue.append(say_meta)
        parts: list[str] = []

        async def _tracked_stream() -> "AsyncIterable[str]":
            try:
                async for chunk in text:
                    if not chunk:
                        continue
                    parts.append(chunk)
                    say_meta["text"] = "".join(parts)
                    yield chunk
            except Exception as e:
                from unify.logger import LOGGER

                LOGGER.error(f"⬥ Speech rewrite stream interrupted: {e}")
            final = "".join(parts)
            if final:
                _log.notification_say(final, notification_source=notification_source)

        handle = session.say(
            _tracked_stream(),
            allow_interruptions=True,
            add_to_chat_ctx=True,
        )
        _register_interruptible_tts(
            handle,
            lambda: "".join(parts),
            notification_source,
        )

    def _extract_chat_messages(
        ctx,
        *,
        strip_images: bool = False,
        tail: int | None = None,
    ) -> list[dict]:
        """Convert a LiveKit ChatContext into a list of message dicts for direct LLM calls.

        Parameters
        ----------
        strip_images : bool
            When True, image content parts are dropped and only the text portions
            of multi-modal messages are kept.  Messages that become empty after
            stripping are omitted entirely.
        tail : int | None
            When set, only the last *tail* messages are returned (after any
            image stripping).  Useful for keeping the context compact.
        """
        from livekit.agents.llm import ImageContent

        messages: list[dict] = []
        for item in ctx.items:
            role = getattr(item, "role", None)
            if role is None:
                continue
            raw_content = getattr(item, "content", None)
            if not raw_content:
                continue
            has_images = isinstance(raw_content, list) and any(
                isinstance(c, ImageContent) for c in raw_content
            )
            if has_images:
                if strip_images:
                    text_parts = [c for c in raw_content if isinstance(c, str)]
                    text = " ".join(text_parts).strip()
                    if text:
                        messages.append({"role": role, "content": text})
                else:
                    parts: list[dict] = []
                    for c in raw_content:
                        if isinstance(c, str):
                            parts.append({"type": "text", "text": c})
                        elif isinstance(c, ImageContent) and isinstance(c.image, str):
                            parts.append(
                                {"type": "image_url", "image_url": {"url": c.image}},
                            )
                    if parts:
                        messages.append({"role": role, "content": parts})
            else:
                text = getattr(item, "text_content", None)
                if not text:
                    continue
                messages.append({"role": role, "content": text})
        if tail is not None and len(messages) > tail:
            messages = messages[-tail:]
        return messages

    def apply_notification(
        message: str,
        should_speak: bool = False,
        *,
        spoken_message: str = "",
        notification_id: str = "",
        source: str = "",
        notification_source: str = "",
        llm_log_path: str = "",
    ) -> None:
        # Awareness notifications (should_speak=False) are injected into the fast
        # brain's context so it can surface them in first person. should_speak=True
        # guidance is spoken verbatim by the slow brain (the fast brain only emits
        # a filler phrase), then its spoken text lands in context as an assistant
        # turn. If the caller interrupts mid-utterance, the unheard remainder is
        # reported to the slow brain via VoiceInterrupt (see _speak_now), not the
        # fast brain. Proactive speech is fire-and-forget filler — it never
        # updates context.
        speech_text = spoken_message or message
        if notification_source != "proactive_speech" and message and not should_speak:
            notification_message = f"[notification] {message}"
            assistant._chat_ctx.add_message(
                role="system",
                content=[notification_message],
            )
            session.history.add_message(
                role="system",
                content=[notification_message],
            )

        if should_speak and speech_text:
            if notification_source == "proactive_speech":
                # Proactive speech exists purely to fill silence — never queue it.
                # Play immediately if the pipeline is fully quiescent and nothing
                # else is waiting; otherwise discard silently.
                if not _is_pipeline_quiescent() or _queued_speech:
                    return
                _speak_now(
                    speech_text,
                    notification_id,
                    notification_source,
                    message,
                    llm_log_path,
                )
            else:
                # The slow brain has produced spoken output for the current
                # turn; mark it so any in-flight / re-triggered buffer filler is
                # suppressed rather than played after this real answer, and end
                # the filler streak so the next filler is a fresh first reaction.
                assistant._slow_brain_responded_turn = assistant._user_turn_seq
                assistant._buffers_since_slow_reply = 0
                # Latest slow brain guidance supersedes older queued speech.
                _queued_speech.clear()
                _queued_speech.append(
                    (
                        speech_text,
                        notification_id,
                        notification_source,
                        message,
                        llm_log_path,
                    ),
                )
                maybe_speak_queued()

    def apply_assistant_turn_injection(content: str) -> None:
        if not content:
            return
        assistant._chat_ctx.add_message(role="assistant", content=[content])
        session.history.add_message(role="assistant", content=[content])

    def maybe_speak_queued() -> None:
        """Speak the next queued slow-brain response, verbatim, when the floor is free.

        Releases the line as soon as nobody is occupying the voice floor (the user
        speaking, or assistant audio playing). It does NOT wait for the agent to
        leave the "thinking" state, so a ready line is never stalled behind reply
        generation. Filler/answer ordering is preserved by ``llm_node`` suppressing
        a filler once the slow brain has responded for the turn.
        """
        if not speech_gate_open or not _queued_speech:
            return
        block_reason = _queued_speech_block_reason()
        if block_reason:
            _log.info(f"Queued slow-brain speech deferred: {block_reason}")
            return
        (
            text,
            notification_id,
            notification_source,
            notification_content,
            llm_log_path,
        ) = _queued_speech.pop(0)
        _speak_now(
            text,
            notification_id,
            notification_source,
            notification_content,
            llm_log_path,
        )

    def on_notification(data: dict) -> None:
        """Handle notifications from conversation manager."""
        nonlocal assistant_screen_share_active, _agent_service_url
        if data.get("event_name") == "AssistantTurnInjected":
            payload = data.get("payload") or {}
            apply_assistant_turn_injection(str(payload.get("content") or ""))
            return

        payload = data.get("payload") or data
        message = payload.get("message", "")
        # Track screen share state from meet interaction notifications.
        if payload.get("source") == "meet_interaction":
            low = message.lower()
            if "screen sharing is now on" in low:
                assistant_screen_share_active = True
                if payload.get("agent_service_url"):
                    _agent_service_url = payload["agent_service_url"]
            elif "screen sharing is now off" in low:
                assistant_screen_share_active = False
                _clear_visual_context(source="assistant")
            elif "stopped sharing" in low:
                source = "user" if "user" in low else "assistant"
                _clear_visual_context(source=source)
            elif "took remote control" in low:
                user_remote_control_active = True
            elif "released remote control" in low:
                user_remote_control_active = False

                async def _update_cache_after_remote_control():
                    entry = await capture_assistant_screenshot(
                        utterance="",
                        cached=False,
                        fb_logger=_log,
                        agent_service_url=_agent_service_url,
                        http_session=_screenshot_http_session,
                    )
                    if entry and assistant_screen_share_active:
                        _handle_screenshot(entry)

                if assistant_screen_share_active:
                    asyncio.create_task(_update_cache_after_remote_control())
        spoken_message = payload.get("spoken_message", "")
        should_speak = payload.get("should_speak", False)
        notification_source = payload.get("source", "")
        llm_log_path = payload.get("llm_log_path", "")
        # A slow-brain spoken turn carries (possibly empty) fast-brain guidance
        # bundled with it; set it so the fast brain can use it on the next message
        # — and so a spoken turn without guidance clears any stale note.
        if notification_source == "slow_brain" and should_speak:
            assistant._fast_brain_guidance = payload.get("fast_brain_guidance", "")
        notification_id = content_trace_id("guid", message or spoken_message)
        triggers_turn = notification_source not in (
            "meet_interaction",
            "proactive_speech",
        )
        _log.notification(
            notification_source,
            message,
            speak=should_speak,
            turn=triggers_turn,
        )

        if message or (should_speak and spoken_message):
            if not session_ready or (should_speak and not speech_gate_open):
                pending_notifications.append(
                    (
                        message,
                        spoken_message,
                        should_speak,
                        notification_id,
                        notification_source,
                        llm_log_path,
                    ),
                )
                _log.notification_buffered(len(pending_notifications))
            else:
                apply_notification(
                    message,
                    should_speak,
                    spoken_message=spoken_message,
                    notification_id=notification_id,
                    source="socket_callback",
                    notification_source=notification_source,
                    llm_log_path=llm_log_path,
                )
                # Only awareness notifications (should_speak=False) regenerate the
                # fast brain's filler with new context. Spoken guidance already has
                # its real content queued; regenerating a filler for it would only
                # re-enter the "thinking" state and defer that very line.
                if triggers_turn and not should_speak:
                    _invalidate_current_generation(
                        "notification_during_generation",
                        notification_id,
                    )

    event_broker.register_callback("app:call:status", on_status)
    event_broker.register_callback("app:call:notification", on_notification)

    def on_idle_smalltalk_state(data: dict) -> None:
        assistant.set_idle_smalltalk_allowed(
            bool(data.get("idle_smalltalk_allowed")),
        )

    event_broker.register_callback(
        "app:call:idle_smalltalk_state",
        on_idle_smalltalk_state,
    )

    # --- Tier 1: Comms from call participants (all calls) ---
    is_boss_user = bool(contact.get("is_system", False))
    participant_ids: set[int] = set()
    if contact.get("contact_id") is not None:
        participant_ids.add(contact["contact_id"])

    def _inject_silent_context(msg: str) -> None:
        """Inject a system message into chat context as silent background."""
        assistant._chat_ctx.add_message(role="system", content=[msg])
        session.history.add_message(role="system", content=[msg])

    def on_participant_comms(data: dict) -> None:
        raw = data.get("event") if "event" in data else json.dumps(data)
        text = render_participant_comms(
            raw if isinstance(raw, str) else json.dumps(raw),
            participant_ids,
        )
        if not text:
            return
        _log.participant_comms(text)
        if not session_ready:
            return
        _inject_silent_context(text)
        if text.startswith("[You "):
            if assistant.user_turn_generating:
                _invalidate_current_generation(
                    "outbound_action_during_generation",
                    "participant_comms",
                )
            else:
                trigger_generate_reply(
                    reason="outbound_message_acknowledgment",
                    source_id="participant_comms",
                )

    event_broker.register_callback("app:comms:*", on_participant_comms)

    # Handle call_answered that arrived during initialization
    if call_answered_flag.is_set():
        _log.call_status("call_answered (arrived during init)")
        assistant.set_call_received()

    _log.session_start("Starting AgentSession + history hydration (parallel)")
    history_task = asyncio.create_task(
        hydrate_fast_brain_history(
            participant_ids=participant_ids,
            is_boss_user=is_boss_user,
            assistant_name=assistant_name or "Assistant",
            limit=SETTINGS.conversation.FAST_BRAIN_CONTEXT_WINDOW,
        ),
    )
    await session.start(room=ctx.room, agent=assistant, room_input_options=rio)
    if joined_gate_required and not outbound and not user_joined_event.is_set():
        remote_participants = getattr(ctx.room, "remote_participants", {}) or {}
        if remote_participants:
            _mark_user_joined("existing_participant")
    history_lines = await history_task
    if history_lines:
        history_block = (
            "--- Recent conversation history ---\n"
            + "\n".join(history_lines)
            + "\n--- Current call ---"
        )
        assistant._chat_ctx.add_message(role="system", content=[history_block])
        session.history.add_message(role="system", content=[history_block])
        _log.info(f"Hydrated {len(history_lines)} historical events into context")

    # Mark session ready and process any buffered notifications BEFORE first utterance.
    # After this, the on_notification callback will apply notifications immediately.
    # Note: For outbound calls, llm_node will wait for call_received (set by on_status).
    session_ready = True
    if pending_notifications:
        _log.session_ready(
            f"Applying {len(pending_notifications)} buffered notification(s)",
        )
        still_pending_notifications: list[tuple[str, str, bool, str, str, str]] = []
        for (
            message,
            spoken_message,
            should_speak,
            notification_id,
            notification_source,
            llm_log_path,
        ) in pending_notifications:
            if should_speak and not speech_gate_open:
                still_pending_notifications.append(
                    (
                        message,
                        spoken_message,
                        should_speak,
                        notification_id,
                        notification_source,
                        llm_log_path,
                    ),
                )
                continue
            apply_notification(
                message,
                should_speak,
                spoken_message=spoken_message,
                notification_id=notification_id,
                source="pending_buffer_flush",
                notification_source=notification_source,
                llm_log_path=llm_log_path,
            )
        pending_notifications[:] = still_pending_notifications

    async def _publish_ready_to_speak() -> None:
        if channel == "phone":
            return
        await ctx.room.local_participant.publish_data(
            json.dumps({"type": "ready_to_speak"}).encode(),
            topic="agent_status",
            reliable=True,
        )

    async def _generate_opening_greeting(*, authoritative_briefing: bool) -> str:
        """Pre-generate the opening line via a sidecar LLM call.

        Returns the cached depleted-credits response when the credit gate is
        closed, otherwise generates from the voice system prompt plus the
        current call history (which includes any injected system briefing).
        """
        from unify.common.llm_client import new_llm_client

        if not credit_gate_monitor.state.allowed:
            _log.info("Credit gate greeting served from cached state")
            return DEPLETED_CREDITS_FAST_BRAIN_RESPONSE
        greeting_client = new_llm_client(
            model=SETTINGS.conversation.FAST_BRAIN_MODEL,
            origin="fast_brain_greeting",
            reasoning_effort="low",
        )
        greeting_messages = build_opening_greeting_messages(
            system_prompt=system_prompt,
            history_messages=_extract_chat_messages(session.history),
            authoritative_briefing=authoritative_briefing,
        )
        return await greeting_client.generate(messages=greeting_messages)

    async def _generate_smalltalk_reply(
        user_text: str,
        *,
        idle_status_smalltalk: bool = False,
    ):
        """Resolve a pure small-talk turn, or return None to defer.

        Returns one of:
        - a reply string: a social / biographical / self-context / repeat answer
          to speak (the fast brain owns the turn);
        - ``_SMALLTALK_STAY_SILENT``: a bare acknowledgement that needs no reply;
        - ``None``: defer to the slow brain (anything substantive, or any action
          / status question, or any error / overreach).

        Drawn from the assistant persona plus recent history. Fail-safe to None.
        """
        from unify.common.llm_client import new_llm_client

        if not (user_text or "").strip():
            return None
        try:
            client = new_llm_client(
                model=SETTINGS.conversation.FAST_BRAIN_MODEL,
                origin="fast_brain_smalltalk",
                reasoning_effort="low",
            )
            messages = build_smalltalk_messages(
                system_prompt=system_prompt,
                history_messages=_extract_chat_messages(session.history, tail=8),
                user_text=user_text.strip(),
                idle_status_smalltalk=idle_status_smalltalk,
            )
            raw = await client.generate(messages=messages)
            text = " ".join(str(raw).split()).strip().strip("\"'“”‘’")
            upper = text.upper()
            if not text or upper.startswith(SMALLTALK_DEFER_SENTINEL):
                return None
            if upper.startswith(SMALLTALK_SILENCE_SENTINEL):
                return _SMALLTALK_STAY_SILENT
            if len(text) > _MAX_SMALLTALK_CHARS:
                return None
            return text
        except Exception as exc:  # never let small-talk selection break the turn
            _log.info(f"Small-talk selection failed; deferring: {exc}")
            return None

    assistant._generate_smalltalk_reply = _generate_smalltalk_reply

    def _inject_opening_system_context(text: str) -> None:
        """Seed the call with a durable system briefing before the opener.

        The briefing is a ``system`` message — never an assistant turn — so the
        model retains the full intended context to fall back on (e.g. after an
        interruption) without ever assuming the caller heard un-uttered content.
        """
        assistant._chat_ctx.add_message(role="system", content=[text])
        session.history.add_message(role="system", content=[text])

    opening_mode = opening_config["mode"]

    async def _prepare_opening() -> tuple[str, str | dict | None]:
        if opening_mode == "speak":
            return (
                "speak",
                await _generate_opening_greeting(authoritative_briefing=False),
            )
        if opening_mode == "briefed":
            _inject_opening_system_context(opening_config["system_context"])
            return (
                "speak",
                await _generate_opening_greeting(authoritative_briefing=True),
            )
        if opening_mode == "recorded":
            preloaded = await asyncio.to_thread(
                _preload_recorded_opening_pcm,
                opening_config,
            )
            return "recorded", {
                "config": opening_config,
                "preloaded": preloaded,
            }
        if opening_mode == "simulated":
            simulated_utterance = opening_config.get("simulated_utterance", "").strip()
            if not simulated_utterance:
                raise ValueError("simulated opening requires simulated_utterance")
            return "simulated", simulated_utterance
        return "silent", None

    opening_task = asyncio.create_task(_prepare_opening())
    await event_broker.publish(
        "app:call:status",
        json.dumps(
            {
                "type": "agent_ready",
                "room_name": ctx.room.name,
                "channel": channel,
            },
        ),
    )

    if outbound:
        _log.info("Outbound call — waiting for callee to answer…")
        await call_answered_flag.wait()
        _log.call_status("call_answered — opening turn")

    await user_joined_event.wait()
    speech_gate_open = True

    if outbound:
        # Hold the opener until the callee's first completed utterance (their
        # "Hello?") or a fallback timeout — so it lands when they are actually
        # listening, not into dead air right after the line connects. The first
        # turn is consumed by the opener (filler + slow-brain turn suppressed via
        # _opening_pending); a later turn is a normal barge-in.
        try:
            await asyncio.wait_for(
                assistant._first_user_turn.wait(),
                OUTBOUND_OPENING_TRIGGER_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            if user_is_speaking:
                # Timed out mid-"Hello?" — let them finish before we speak.
                await assistant._first_user_turn.wait()
        _log.call_status("outbound_opener_triggered")

    try:
        prepared_mode, prepared_payload = await opening_task
    except Exception as exc:
        _log.llm_error(f"opening preload failed: {exc}")
        prepared_mode, prepared_payload = "speak", "Hello — I'm here."

    if prepared_mode == "speak":
        await _publish_ready_to_speak()
        _say_opening(str(prepared_payload or "Hello — I'm here."))
    elif prepared_mode == "recorded":
        payload = prepared_payload or opening_config
        if isinstance(payload, dict) and "config" in payload:
            recorded_config = payload["config"]
            _recorded_opening_preloaded = payload.get("preloaded") or {}
        else:
            recorded_config = payload
        await _publish_ready_to_speak()
        await _run_recorded_opening(recorded_config)
    elif prepared_mode == "simulated":
        simulated_utterance = str(prepared_payload or "")
        await _publish_ready_to_speak()
        assistant._chat_ctx.add_message(role="assistant", content=[simulated_utterance])
        session.history.add_message(role="assistant", content=[simulated_utterance])
        _log.assistant_speech(
            simulated_utterance,
            source=opening_config.get("source", "simulated_opening"),
            llm_log_path="",
        )
        await _publish_assistant_utterance(simulated_utterance)
        _enqueue_mood_classification(
            "assistant",
            simulated_utterance,
            content_trace_id("utt", f"assistant:{simulated_utterance}"),
        )
    else:
        _log.info("Opening turn suppressed by call opening config")
        await _publish_ready_to_speak()

    # The opener has been dispatched; resume normal turn handling (fast-brain
    # fillers and slow-brain turns) for any subsequent user speech.
    assistant._opening_pending = False

    if pending_notifications:
        gated_notifications = list(pending_notifications)
        pending_notifications.clear()
        for (
            message,
            spoken_message,
            should_speak,
            notification_id,
            notification_source,
            llm_log_path,
        ) in gated_notifications:
            apply_notification(
                message,
                should_speak,
                spoken_message=spoken_message,
                notification_id=notification_id,
                source="user_joined_buffer_flush",
                notification_source=notification_source,
                llm_log_path=llm_log_path,
            )
    maybe_speak_queued()

    # Inject the initializing-state system message *after* the greeting has
    # been generated and spoken.  Placing it before the greeting caused the
    # LLM to proactively mention "still setting up" in the opening line,
    # which sounds odd when no action has been requested yet.  The note only
    # matters for subsequent turns where the user might ask for something
    # that requires initialized managers.
    if not os.environ.get("UNITY_CM_INITIALIZED"):
        _init_note = (
            "[system] You have just started up and your systems are still "
            "syncing — loading your files, tools, and any conversation "
            "history. This takes a few moments. If the user asks you to do "
            "something that requires looking things up or taking action, let "
            "them know naturally that you are still getting set up (e.g. "
            "'give me just a moment to finish getting set up and I'll get "
            "right on that'). Do NOT say 'I can't do that' — frame it as a "
            "brief delay, not a limitation. You will receive a notification "
            "when everything is ready."
        )
        assistant._chat_ctx.add_message(role="system", content=[_init_note])
        session.history.add_message(role="system", content=[_init_note])
        _log.info("Injected initializing-state system message (CM not yet initialized)")


if __name__ == "__main__":
    # CLI handling
    room_name = configure_from_cli(
        extra_env=[
            ("CONTACT", True),
            ("BOSS", True),
            ("ASSISTANT_BIO", False),
            ("ASSISTANT_ID", False),
            ("USER_ID", False),
        ],
    )

    if should_dispatch_livekit_agent():
        _log.dispatch(f"Dispatching LiveKit agent {room_name}")
        dispatch_livekit_agent(
            room_name,
            record=True,
            assistant_id=SESSION_DETAILS.assistant.agent_id,
            user_id=SESSION_DETAILS.user.id,
        )
        _log.dispatch(f"LiveKit agent {room_name} dispatched")

    # Run the agent using the standard CLI - this is the natural way to run LiveKit agents.
    # The process will be terminated via SIGTERM when cleanup_call_proc() is called.
    agents.cli.run_app(
        agents.WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name=room_name,
            prewarm_fnc=prewarm,
            initialize_process_timeout=60,
        ),
    )
