"""
tests/conversation_manager/test_medium_scripts.py
=======================================================

Tests for the medium scripts (call.py) that handle voice calls.

These scripts implement the "fast brain" Voice Agent that handles real-time
conversation while the Main CM Brain (slow brain) handles orchestration.

## Test Categories

### Unit Tests (no external dependencies)
- Assistant class initialization and state management
- Common helper functions (publish events, create_end_call, etc.)
- CLI argument parsing
- Voice prompt building
- Event type selection based on channel

### Integration Tests (require event broker)
- Event publishing flows
- Guidance subscription patterns
- Cross-thread event delivery

## Key Components Tested

1. **Assistant class** (call.py):
   - Initialization with contact/boss/channel/instructions
   - set_call_received() state transitions
   - Utterance event type selection

2. **Common helpers** (common.py):
   - publish_call_started / publish_call_ended
   - create_end_call with pre_shutdown_callback
   - configure_from_cli argument parsing

3. **Voice Agent prompt**:
   - build_voice_agent_prompt output structure
"""

import asyncio
import inspect
import json
from types import SimpleNamespace

import pytest
import pytest_asyncio

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def _preserve_global_logging_state():
    """Snapshot and restore global logging state around every test.

    Several tests here drive the real voice ``entrypoint``, which calls
    ``_configure_child_logging`` and, as a global side effect, flips
    ``LOGGER.propagate`` on and clears handlers (the forkserver relay
    fix). Without restoration that state leaks across tests in a
    single-process run and breaks order-dependent expectations — notably
    ``TestChildProcessLogging``, which asserts the default
    ``propagate=False`` precondition. Restoring per test keeps the file
    order-independent regardless of how it's run.
    """
    import logging

    from unify.logger import LOGGER

    _logger_names = ("livekit", "livekit.agents", "livekit.plugins")
    saved = {
        "unify": (LOGGER.propagate, list(LOGGER.handlers)),
        **{
            name: (
                logging.getLogger(name).propagate,
                list(logging.getLogger(name).handlers),
            )
            for name in _logger_names
        },
    }
    try:
        yield
    finally:
        LOGGER.propagate, LOGGER.handlers = saved["unify"]
        for name in _logger_names:
            lg = logging.getLogger(name)
            lg.propagate, lg.handlers = saved[name]


@pytest_asyncio.fixture
async def event_broker():
    """Local in-memory broker for testing."""
    from unify.conversation_manager.event_broker import create_event_broker

    broker = create_event_broker()
    yield broker
    await broker.aclose()


@pytest.fixture
def boss_contact():
    """Standard boss contact for testing."""
    return {
        "contact_id": 1,
        "first_name": "Test",
        "surname": "Boss",
        "phone_number": "+15555555555",
        "email_address": "boss@test.com",
    }


@pytest.fixture
def external_contact():
    """Standard external contact for testing."""
    return {
        "contact_id": 2,
        "first_name": "External",
        "surname": "Contact",
        "phone_number": "+15555555556",
        "email_address": "contact@test.com",
    }


# =============================================================================
# Unit Tests: Assistant Class (call.py - TTS mode)
# =============================================================================


class TestTTSAssistantClass:
    """Tests for the Assistant class in call.py (TTS mode)."""

    def test_assistant_initialization_phone_channel(self, boss_contact):
        """Assistant initializes correctly for phone channel."""
        from unify.conversation_manager.medium_scripts.call import Assistant

        assistant = Assistant(
            contact=boss_contact,
            boss=boss_contact,
            channel="phone_call",
            instructions="Test instructions",
            outbound=False,
        )

        assert assistant.contact == boss_contact
        assert assistant.boss == boss_contact
        assert assistant.channel == "phone_call"
        assert assistant.call_received is True  # inbound call, already received

    def test_assistant_initialization_unify_meet_channel(self, boss_contact):
        """Assistant initializes correctly for unify_meet channel."""
        from unify.conversation_manager.medium_scripts.call import Assistant

        assistant = Assistant(
            contact=boss_contact,
            boss=boss_contact,
            channel="unify_meet",
            instructions="Test instructions",
            outbound=False,
        )

        assert assistant.channel == "unify_meet"
        assert assistant.call_received is True

    def test_assistant_outbound_call_not_received_initially(self, boss_contact):
        """Outbound calls start with call_received=False."""
        from unify.conversation_manager.medium_scripts.call import Assistant

        assistant = Assistant(
            contact=boss_contact,
            boss=boss_contact,
            channel="phone_call",
            instructions="Test instructions",
            outbound=True,
        )

        assert assistant.call_received is False

    def test_assistant_set_call_received(self, boss_contact):
        """set_call_received() updates state correctly."""
        from unify.conversation_manager.medium_scripts.call import Assistant

        assistant = Assistant(
            contact=boss_contact,
            boss=boss_contact,
            channel="phone_call",
            instructions="Test instructions",
            outbound=True,
        )

        assert assistant.call_received is False
        assistant.set_call_received()
        assert assistant.call_received is True

    def test_assistant_utterance_event_type_phone(self, boss_contact):
        """Phone channel uses InboundPhoneUtterance."""
        from unify.conversation_manager.events import InboundPhoneUtterance
        from unify.conversation_manager.medium_scripts.call import Assistant

        assistant = Assistant(
            contact=boss_contact,
            boss=boss_contact,
            channel="phone_call",
            instructions="Test instructions",
        )

        assert assistant.utterance_event == InboundPhoneUtterance

    def test_assistant_utterance_event_type_meet(self, boss_contact):
        """Meet channel uses InboundUnifyMeetUtterance."""
        from unify.conversation_manager.events import InboundUnifyMeetUtterance
        from unify.conversation_manager.medium_scripts.call import Assistant

        assistant = Assistant(
            contact=boss_contact,
            boss=boss_contact,
            channel="unify_meet",
            instructions="Test instructions",
        )

        assert assistant.utterance_event == InboundUnifyMeetUtterance

    def test_assistant_outbound_utterance_event_type_phone(self, boss_contact):
        """Phone channel uses OutboundPhoneUtterance for assistant."""
        from unify.conversation_manager.events import OutboundPhoneUtterance
        from unify.conversation_manager.medium_scripts.call import Assistant

        assistant = Assistant(
            contact=boss_contact,
            boss=boss_contact,
            channel="phone_call",
            instructions="Test instructions",
        )

        assert assistant.assistant_utterance_event == OutboundPhoneUtterance

    def test_assistant_outbound_utterance_event_type_meet(self, boss_contact):
        """Meet channel uses OutboundUnifyMeetUtterance for assistant."""
        from unify.conversation_manager.events import OutboundUnifyMeetUtterance
        from unify.conversation_manager.medium_scripts.call import Assistant

        assistant = Assistant(
            contact=boss_contact,
            boss=boss_contact,
            channel="unify_meet",
            instructions="Test instructions",
        )

        assert assistant.assistant_utterance_event == OutboundUnifyMeetUtterance


class TestElevenLabsTwinPronunciation:
    """Tests for the ElevenLabs text-stream pronunciation normalizer."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("chunks", "expected"),
        [
            (["Say T-W1N now."], "Say Twin now."),
            (["Say t-w1n now."], "Say Twin now."),
            (["Say T-", "w1", "N now."], "Say Twin now."),
            (["T-W1N and t-W1n"], "Twin and Twin"),
            (["Almost T-W1 but not done"], "Almost T-W1 but not done"),
        ],
    )
    async def test_normalizes_twin_marker_across_stream_chunks(
        self,
        chunks,
        expected,
    ):
        from unify.conversation_manager.medium_scripts.call import (
            _normalize_elevenlabs_twin_pronunciation_stream,
        )

        async def _chunks():
            for chunk in chunks:
                yield chunk

        normalized = [
            chunk
            async for chunk in _normalize_elevenlabs_twin_pronunciation_stream(
                _chunks(),
            )
        ]

        assert "".join(normalized) == expected


# =============================================================================
# Unit Tests: Common Helpers
# =============================================================================


class TestCommonHelpers:
    """Tests for shared helper functions in common.py."""

    def test_should_dispatch_livekit_agent_with_dev_command(self, monkeypatch):
        """should_dispatch_livekit_agent returns True for 'dev' command."""
        from unify.conversation_manager.medium_scripts import common

        # Patch sys.argv in the common module's namespace
        monkeypatch.setattr(common.sys, "argv", ["call.py", "dev"])
        assert common.should_dispatch_livekit_agent() is True

    def test_should_dispatch_livekit_agent_with_connect_command(self, monkeypatch):
        """should_dispatch_livekit_agent returns True for 'connect' command."""
        from unify.conversation_manager.medium_scripts import common

        monkeypatch.setattr(common.sys, "argv", ["call.py", "connect"])
        assert common.should_dispatch_livekit_agent() is True

    def test_should_not_dispatch_livekit_agent_for_download_files(self, monkeypatch):
        """should_dispatch_livekit_agent returns False for 'download-files' command."""
        from unify.conversation_manager.medium_scripts import common

        monkeypatch.setattr(common.sys, "argv", ["call.py", "download-files"])
        assert common.should_dispatch_livekit_agent() is False

    def test_should_not_dispatch_livekit_agent_with_no_args(self, monkeypatch):
        """should_dispatch_livekit_agent returns False when no args provided."""
        from unify.conversation_manager.medium_scripts import common

        monkeypatch.setattr(common.sys, "argv", ["call.py"])
        assert common.should_dispatch_livekit_agent() is False


# =============================================================================
# Unit Tests: Event Publishing Helpers
# =============================================================================


@pytest.mark.asyncio
class TestEventPublishingHelpers:
    """Tests for event publishing helper functions.

    These tests patch common.event_broker to use our test fixture so we can
    verify the events are published correctly.
    """

    async def test_publish_call_started_phone_channel(
        self,
        event_broker,
        boss_contact,
        monkeypatch,
    ):
        """publish_call_started publishes PhoneCallStarted for phone channel."""
        from unify.conversation_manager.events import Event, PhoneCallStarted
        from unify.conversation_manager.medium_scripts import common

        # Patch the event_broker in common module to use our test fixture
        monkeypatch.setattr(common, "event_broker", event_broker)

        async with event_broker.pubsub() as pubsub:
            await pubsub.subscribe("app:comms:phone_call_started")

            await common.publish_call_started(boss_contact, "phone_call")

            msg = await pubsub.get_message(
                timeout=2.0,
                ignore_subscribe_messages=True,
            )
            assert msg is not None
            event = Event.from_json(msg["data"])
            assert isinstance(event, PhoneCallStarted)
            assert event.contact == boss_contact

    async def test_publish_call_started_meet_channel(
        self,
        event_broker,
        boss_contact,
        monkeypatch,
    ):
        """publish_call_started publishes UnifyMeetStarted for meet channel."""
        from unify.conversation_manager.events import Event, UnifyMeetStarted
        from unify.conversation_manager.medium_scripts import common

        monkeypatch.setattr(common, "event_broker", event_broker)

        async with event_broker.pubsub() as pubsub:
            await pubsub.subscribe("app:comms:unify_meet_started")

            await common.publish_call_started(boss_contact, "unify_meet")

            msg = await pubsub.get_message(
                timeout=2.0,
                ignore_subscribe_messages=True,
            )
            assert msg is not None
            event = Event.from_json(msg["data"])
            assert isinstance(event, UnifyMeetStarted)
            assert event.contact == boss_contact

    async def test_publish_call_ended_phone_channel(
        self,
        event_broker,
        boss_contact,
        monkeypatch,
    ):
        """publish_call_ended publishes PhoneCallEnded for phone channel."""
        from unify.conversation_manager.events import Event, PhoneCallEnded
        from unify.conversation_manager.medium_scripts import common

        monkeypatch.setattr(common, "event_broker", event_broker)

        async with event_broker.pubsub() as pubsub:
            await pubsub.subscribe("app:comms:phone_call_ended")

            await common.publish_call_ended(boss_contact, "phone_call")

            msg = await pubsub.get_message(
                timeout=2.0,
                ignore_subscribe_messages=True,
            )
            assert msg is not None
            event = Event.from_json(msg["data"])
            assert isinstance(event, PhoneCallEnded)
            assert event.contact == boss_contact

    async def test_publish_call_ended_meet_channel(
        self,
        event_broker,
        boss_contact,
        monkeypatch,
    ):
        """publish_call_ended publishes UnifyMeetEnded for meet channel."""
        from unify.conversation_manager.events import Event, UnifyMeetEnded
        from unify.conversation_manager.medium_scripts import common

        monkeypatch.setattr(common, "event_broker", event_broker)

        async with event_broker.pubsub() as pubsub:
            await pubsub.subscribe("app:comms:unify_meet_ended")

            await common.publish_call_ended(boss_contact, "unify_meet")

            msg = await pubsub.get_message(
                timeout=2.0,
                ignore_subscribe_messages=True,
            )
            assert msg is not None
            event = Event.from_json(msg["data"])
            assert isinstance(event, UnifyMeetEnded)
            assert event.contact == boss_contact


# =============================================================================
# Unit Tests: End Call Helper
# =============================================================================


@pytest.mark.asyncio
class TestEndCallHelper:
    """Tests for create_end_call helper.

    These tests patch common.event_broker to use our test fixture so we can
    verify the events are published correctly.
    """

    async def test_create_end_call_publishes_ended_event(
        self,
        event_broker,
        boss_contact,
        monkeypatch,
    ):
        """create_end_call returns function that publishes ended event."""
        from unify.conversation_manager.events import Event, PhoneCallEnded
        from unify.conversation_manager.medium_scripts import common

        # Patch the event_broker in common module to use our test fixture
        monkeypatch.setattr(common, "event_broker", event_broker)

        end_call = common.create_end_call(boss_contact, "phone_call")

        async with event_broker.pubsub() as pubsub:
            await pubsub.subscribe("app:comms:phone_call_ended")

            await end_call()

            msg = await pubsub.get_message(
                timeout=2.0,
                ignore_subscribe_messages=True,
            )
            assert msg is not None
            event = Event.from_json(msg["data"])
            assert isinstance(event, PhoneCallEnded)

    async def test_create_end_call_runs_pre_shutdown_callback(
        self,
        event_broker,
        boss_contact,
        monkeypatch,
    ):
        """create_end_call runs pre_shutdown_callback before shutdown."""
        from unify.conversation_manager.medium_scripts import common

        monkeypatch.setattr(common, "event_broker", event_broker)

        callback_called = {"value": False}

        def pre_shutdown():
            callback_called["value"] = True

        end_call = common.create_end_call(
            boss_contact,
            "phone_call",
            pre_shutdown_callback=pre_shutdown,
        )

        await end_call()

        assert callback_called["value"] is True

    async def test_create_end_call_handles_callback_error(
        self,
        event_broker,
        boss_contact,
        monkeypatch,
        caplog,
    ):
        """create_end_call continues even if callback raises."""
        import logging

        from unify.conversation_manager.medium_scripts import common

        monkeypatch.setattr(common, "event_broker", event_broker)

        def failing_callback():
            raise ValueError("Callback error")

        end_call = common.create_end_call(
            boss_contact,
            "phone_call",
            pre_shutdown_callback=failing_callback,
        )

        unity_logger = logging.getLogger("unify")
        unity_logger.addHandler(caplog.handler)
        caplog.handler.setLevel(logging.DEBUG)
        try:
            await end_call()
        finally:
            unity_logger.removeHandler(caplog.handler)

        assert "Error in pre-shutdown callback" in caplog.text


# =============================================================================
# Unit Tests: CLI Argument Parsing
# =============================================================================


class TestCLIArgumentParsing:
    """Tests for configure_from_cli argument parsing.

    These tests patch sys.argv in the common module's namespace since that's
    where configure_from_cli reads it from.
    """

    def test_configure_from_cli_with_full_args(self, monkeypatch):
        """configure_from_cli parses all arguments correctly."""
        from unify.conversation_manager.medium_scripts import common
        from unify.session_details import SESSION_DETAILS

        # Reset SESSION_DETAILS before test
        SESSION_DETAILS.reset()

        contact_json = json.dumps(
            {
                "contact_id": 1,
                "first_name": "Test",
                "surname": "User",
            },
        )
        boss_json = json.dumps(
            {
                "contact_id": 1,
                "first_name": "Boss",
                "surname": "Person",
            },
        )

        # argv[2] is the canonical room name from make_room_name() in call_manager
        monkeypatch.setattr(
            common.sys,
            "argv",
            [
                "call.py",
                "dev",
                "unity_test_assistant_id_phone",
                "elevenlabs",
                "voice123",
                "True",
                "phone_call",
                contact_json,
                boss_json,
                "Test assistant bio",
                "test_assistant_id",
                "test_user_id",
            ],
        )

        room_name = common.configure_from_cli(
            extra_env=[
                ("CONTACT", True),
                ("BOSS", True),
                ("ASSISTANT_BIO", False),
                ("ASSISTANT_ID", False),
                ("USER_ID", False),
            ],
        )

        assert room_name == "unity_test_assistant_id_phone"
        assert SESSION_DETAILS.voice.provider == "elevenlabs"
        assert SESSION_DETAILS.voice.id == "voice123"
        assert SESSION_DETAILS.voice_call.outbound is True
        assert SESSION_DETAILS.voice_call.channel == "phone_call"

    def test_configure_from_cli_meet_room_name(self, monkeypatch):
        """configure_from_cli returns the canonical room name for UnifyMeet calls."""
        from unify.conversation_manager.medium_scripts import common
        from unify.session_details import SESSION_DETAILS

        SESSION_DETAILS.reset()

        contact_json = json.dumps({"contact_id": 1, "first_name": "Test"})
        boss_json = json.dumps({"contact_id": 1, "first_name": "Boss"})

        # Simulate UnifyMeet call — argv[2] is the canonical room name
        # (same as phone calls; agent_name = room_name for all mediums)
        monkeypatch.setattr(
            common.sys,
            "argv",
            [
                "call.py",
                "dev",
                "unity_25_meet",
                "cartesia",
                "voice456",
                "False",
                "unify_meet",
                contact_json,
                boss_json,
                "Bio",
                "test_assistant_id",
                "test_user_id",
            ],
        )

        room_name = common.configure_from_cli(
            extra_env=[
                ("CONTACT", True),
                ("BOSS", True),
                ("ASSISTANT_BIO", False),
                ("ASSISTANT_ID", False),
                ("USER_ID", False),
            ],
        )

        assert room_name == "unity_25_meet"

    def test_configure_from_cli_defaults_none_voice_provider(self, monkeypatch):
        """configure_from_cli defaults 'None' voice provider to cartesia."""
        from unify.conversation_manager.medium_scripts import common
        from unify.session_details import SESSION_DETAILS

        SESSION_DETAILS.reset()

        contact_json = json.dumps({"contact_id": 1, "first_name": "Test"})
        boss_json = json.dumps({"contact_id": 1, "first_name": "Boss"})

        monkeypatch.setattr(
            common.sys,
            "argv",
            [
                "call.py",
                "dev",
                "12345",
                "None",  # Voice provider as "None" string
                "None",  # Voice ID as "None" string
                "False",
                "phone_call",
                contact_json,
                boss_json,
                "Bio",
                "test_assistant_id",
                "test_user_id",
            ],
        )

        common.configure_from_cli(
            extra_env=[
                ("CONTACT", True),
                ("BOSS", True),
                ("ASSISTANT_BIO", False),
                ("ASSISTANT_ID", False),
                ("USER_ID", False),
            ],
        )

        assert SESSION_DETAILS.voice.provider == "cartesia"
        assert SESSION_DETAILS.voice.id == ""


# =============================================================================
# Integration Tests: Guidance Channel Subscription
# =============================================================================


@pytest.mark.asyncio
class TestGuidanceChannelSubscription:
    """Tests for guidance channel subscription patterns."""

    async def test_guidance_channel_receives_call_guidance(self, event_broker):
        """Guidance channel receives FastBrainNotification events."""
        from unify.conversation_manager.events import FastBrainNotification, Event

        contact = {"contact_id": 1, "first_name": "Test"}

        async with event_broker.pubsub() as pubsub:
            await pubsub.subscribe("app:call:notification")

            event = FastBrainNotification(contact=contact, message="Test guidance")
            await event_broker.publish("app:call:notification", event.to_json())

            msg = await pubsub.get_message(
                timeout=2.0,
                ignore_subscribe_messages=True,
            )
            assert msg is not None
            received = Event.from_json(msg["data"])
            assert isinstance(received, FastBrainNotification)
            assert received.message == "Test guidance"

    async def test_status_channel_receives_stop_signal(self, event_broker):
        """Status channel receives stop signals."""
        async with event_broker.pubsub() as pubsub:
            await pubsub.subscribe("app:call:status")

            await event_broker.publish(
                "app:call:status",
                json.dumps({"type": "stop"}),
            )

            msg = await pubsub.get_message(
                timeout=2.0,
                ignore_subscribe_messages=True,
            )
            assert msg is not None
            data = json.loads(msg["data"])
            assert data["type"] == "stop"

    async def test_status_channel_receives_call_answered_signal(self, event_broker):
        """Status channel receives call_answered signals."""
        async with event_broker.pubsub() as pubsub:
            await pubsub.subscribe("app:call:status")

            await event_broker.publish(
                "app:call:status",
                json.dumps({"type": "call_answered"}),
            )

            msg = await pubsub.get_message(
                timeout=2.0,
                ignore_subscribe_messages=True,
            )
            assert msg is not None
            data = json.loads(msg["data"])
            assert data["type"] == "call_answered"


# =============================================================================
# Integration Tests: Utterance Event Publishing
# =============================================================================


@pytest.mark.asyncio
class TestUtteranceEventPublishing:
    """Tests for utterance event publishing patterns."""

    async def test_phone_utterance_published_to_correct_channel(
        self,
        event_broker,
        boss_contact,
    ):
        """Phone utterances are published to app:comms:phone_utterance."""
        from unify.conversation_manager.events import (
            Event,
            InboundPhoneUtterance,
        )

        async with event_broker.pubsub() as pubsub:
            await pubsub.subscribe("app:comms:phone_utterance")

            event = InboundPhoneUtterance(contact=boss_contact, content="Hello")
            await event_broker.publish("app:comms:phone_utterance", event.to_json())

            msg = await pubsub.get_message(
                timeout=2.0,
                ignore_subscribe_messages=True,
            )
            assert msg is not None
            received = Event.from_json(msg["data"])
            assert isinstance(received, InboundPhoneUtterance)
            assert received.content == "Hello"

    async def test_meet_utterance_published_to_correct_channel(
        self,
        event_broker,
        boss_contact,
    ):
        """Meet utterances are published to app:comms:meet_utterance."""
        from unify.conversation_manager.events import (
            Event,
            InboundUnifyMeetUtterance,
        )

        async with event_broker.pubsub() as pubsub:
            await pubsub.subscribe("app:comms:meet_utterance")

            event = InboundUnifyMeetUtterance(contact=boss_contact, content="Hello")
            await event_broker.publish("app:comms:meet_utterance", event.to_json())

            msg = await pubsub.get_message(
                timeout=2.0,
                ignore_subscribe_messages=True,
            )
            assert msg is not None
            received = Event.from_json(msg["data"])
            assert isinstance(received, InboundUnifyMeetUtterance)
            assert received.content == "Hello"


# =============================================================================
# Guidance tests: chat context and system messages
# =============================================================================


@pytest.mark.asyncio
async def test_simulated_opening_publishes_ready_before_utterance(monkeypatch):
    from livekit.agents import llm
    from unify.conversation_manager.medium_scripts import call as call_script

    sequence = []
    contact = {"contact_id": 1, "first_name": "User", "surname": "Example"}
    boss = {"contact_id": 1, "first_name": "User", "surname": "Example"}

    class _ImmediateAwaitable:
        def __await__(self):
            async def _done():
                return None

            return _done().__await__()

    class _FakeLocalParticipant:
        async def publish_data(self, payload, *, topic=None, reliable=False):
            sequence.append(("data", json.loads(payload.decode()), topic, reliable))

    class _FakeRoom:
        name = "fake-room"
        local_participant = _FakeLocalParticipant()

        def on(self, *args, **kwargs):
            return lambda fn: fn

    class _FakeJobContext:
        def __init__(self):
            self.room = _FakeRoom()
            self.job = SimpleNamespace(
                metadata=json.dumps(
                    {
                        "voice_provider": "cartesia",
                        "voice_id": "",
                        "outbound": False,
                        "channel": "unify_meet",
                        "contact": contact,
                        "boss": boss,
                        "assistant_bio": "Assistant bio",
                        "assistant_id": "123",
                        "user_id": "user-123",
                        "assistant_name": "Coordinator Unity",
                        "opening_config": {
                            "mode": "simulated",
                            "simulated_utterance": "Hi, I'm your coordinator unify.",
                            "source": "coordinator_onboarding_intro",
                        },
                    },
                ),
            )

        async def connect(self):
            return None

        def add_shutdown_callback(self, cb):
            pass

        def shutdown(self, reason=""):
            pass

    class _FakeEventBroker:
        def __init__(self):
            self.callbacks = {}

        def set_logger(self, fb_logger):
            pass

        def register_callback(self, channel, handler):
            self.callbacks[channel] = handler

        async def publish(self, channel, message):
            sequence.append(("broker", channel))
            return 1

    fake_session_holder = {}

    class _FakeSession:
        def __init__(self, *args, **kwargs):
            self._chat_ctx = llm.ChatContext()
            self.current_agent = None
            self._events = {}
            self.agent_state = "listening"
            self.current_speech = None
            fake_session_holder["session"] = self

        @property
        def history(self):
            return self._chat_ctx

        def on(self, event_name):
            def _decorator(fn):
                self._events[event_name] = fn
                return fn

            return _decorator

        async def start(self, room, agent, room_input_options=None):
            self.current_agent = agent

        def generate_reply(self, **kwargs):
            return _ImmediateAwaitable()

        def say(self, text, **kwargs):
            return _ImmediateAwaitable()

        def interrupt(self):
            pass

    class _FakeAssistant:
        def __init__(self, *args, **kwargs):
            self._chat_ctx = llm.ChatContext()
            self.call_received = True
            self.user_turn_generating = False

        def set_call_received(self):
            self.call_received = True

        def set_credit_gate_state_provider(self, provider):
            self.credit_gate_state_provider = provider

    async def _noop_async(*args, **kwargs):
        return None

    async def _noop_end_call():
        return None

    fake_session_details = SimpleNamespace(
        user=SimpleNamespace(id=None),
        assistant=SimpleNamespace(
            about="Assistant bio",
            is_coordinator=True,
            agent_id=None,
            name="Coordinator Unity",
            first_name="",
            surname="",
            user_desktop_for=lambda user_id: None,
        ),
        voice=SimpleNamespace(provider="cartesia", id=""),
        voice_call=SimpleNamespace(outbound=False, channel="unify_meet"),
        is_coordinator=True,
        org_id=None,
        unify_key="",
    )

    monkeypatch.setattr(call_script, "event_broker", _FakeEventBroker())
    monkeypatch.setattr(call_script, "SESSION_DETAILS", fake_session_details)
    monkeypatch.setattr(call_script, "AgentSession", _FakeSession)
    monkeypatch.setattr(call_script, "Assistant", _FakeAssistant)
    monkeypatch.setattr(call_script, "UnifyLLM", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        call_script,
        "build_voice_agent_prompt",
        lambda **kwargs: SimpleNamespace(flatten=lambda: "system prompt"),
    )
    monkeypatch.setattr(call_script, "start_event_broker_receive", _noop_async)
    monkeypatch.setattr(call_script, "publish_call_started", _noop_async)
    monkeypatch.setattr(
        call_script,
        "create_end_call",
        lambda *args, **kwargs: _noop_end_call,
    )
    monkeypatch.setattr(
        call_script,
        "setup_participant_disconnect_handler",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(call_script, "RoomInputOptions", lambda **kwargs: object())
    monkeypatch.setattr(call_script, "EnglishModel", lambda: object())
    monkeypatch.setattr(call_script.cartesia, "TTS", lambda **kwargs: object())
    monkeypatch.setattr(call_script.elevenlabs, "TTS", lambda **kwargs: object())
    if hasattr(call_script, "noise_cancellation"):
        monkeypatch.setattr(call_script.noise_cancellation, "BVC", lambda: object())
    monkeypatch.setattr(call_script, "STT", object())
    monkeypatch.setattr(call_script, "VAD", object())

    await call_script.entrypoint(_FakeJobContext())

    ready_index = next(
        i
        for i, item in enumerate(sequence)
        if item[0] == "data" and item[1] == {"type": "ready_to_speak"}
    )
    utterance_index = next(
        i
        for i, item in enumerate(sequence)
        if item[0] == "broker" and item[1] == "app:comms:unify_meet_utterance"
    )
    assert ready_index < utterance_index


@pytest.mark.asyncio
async def test_recorded_opening_uses_interruptible_audio_say(monkeypatch):
    from livekit.agents import llm
    from unify.conversation_manager.medium_scripts import call as call_script

    sequence = []
    audio_sources = []
    fake_audio = object()
    contact = {"contact_id": 1, "first_name": "User", "surname": "Example"}
    boss = {"contact_id": 1, "first_name": "User", "surname": "Example"}

    class _ImmediateAwaitable:
        def __await__(self):
            async def _done():
                return None

            return _done().__await__()

    class _FakeSpeechHandle:
        interrupted = False

        async def wait_for_playout(self):
            return None

        def done(self) -> bool:
            return False

        def interrupt(self, *, force: bool = False) -> "_FakeSpeechHandle":
            self.interrupted = True
            return self

    class _FakeLocalParticipant:
        async def publish_data(self, payload, *, topic=None, reliable=False):
            sequence.append(("data", json.loads(payload.decode()), topic, reliable))

    class _FakeRoom:
        name = "fake-room"
        local_participant = _FakeLocalParticipant()

        def on(self, *args, **kwargs):
            return lambda fn: fn

    class _FakeJobContext:
        def __init__(self):
            self.room = _FakeRoom()
            self.job = SimpleNamespace(
                metadata=json.dumps(
                    {
                        "voice_provider": "cartesia",
                        "voice_id": "",
                        "outbound": False,
                        "channel": "unify_meet",
                        "contact": contact,
                        "boss": boss,
                        "assistant_bio": "Assistant bio",
                        "assistant_id": "123",
                        "user_id": "user-123",
                        "assistant_name": "Coordinator Unity",
                        "is_coordinator": False,
                        "opening_config": {
                            "mode": "recorded",
                            "transcript": "Hi, I'm your coordinator unify.",
                            "recording_path": "/tmp/recorded-opener.wav",
                            "source": "coordinator_onboarding_intro",
                        },
                    },
                ),
            )

        async def connect(self):
            return None

        def add_shutdown_callback(self, cb):
            pass

        def shutdown(self, reason=""):
            pass

    class _FakeEventBroker:
        def set_logger(self, fb_logger):
            pass

        def register_callback(self, channel, handler):
            pass

        async def publish(self, channel, message):
            sequence.append(("broker", channel, message))
            return 1

        def reinit_socket(self):
            pass

    class _FakeSession:
        def __init__(self, *args, **kwargs):
            self._chat_ctx = llm.ChatContext()
            self.current_agent = None
            self._events = {}
            self.agent_state = "listening"
            self.current_speech = None
            self.say_calls = []
            self.generate_reply_calls = []

        @property
        def history(self):
            return self._chat_ctx

        def on(self, event_name):
            def _decorator(fn):
                self._events[event_name] = fn
                return fn

            return _decorator

        async def start(self, room, agent, room_input_options=None):
            self.current_agent = agent

        def generate_reply(self, **kwargs):
            self.generate_reply_calls.append(kwargs)
            return _ImmediateAwaitable()

        def say(self, text, **kwargs):
            sequence.append(("say", text, kwargs))
            self.say_calls.append((text, kwargs))
            return _FakeSpeechHandle()

        def interrupt(self):
            pass

    class _FakeAssistant:
        def __init__(self, *args, **kwargs):
            self._chat_ctx = llm.ChatContext()
            self.call_received = True
            self.user_turn_generating = False

        def set_call_received(self):
            self.call_received = True

        def set_credit_gate_state_provider(self, provider):
            self.credit_gate_state_provider = provider

    class _FakeCreditGateMonitor:
        state = SimpleNamespace(allowed=True)

        async def run(self):
            return None

    async def _noop_async(*args, **kwargs):
        return None

    async def _empty_history(*args, **kwargs):
        return []

    async def _noop_end_call():
        return None

    def _fake_recording_audio_frames(source, **_kwargs):
        audio_sources.append(source)
        return fake_audio

    fake_session_details = SimpleNamespace(
        user=SimpleNamespace(id="user-123"),
        assistant=SimpleNamespace(
            about="Assistant bio",
            is_coordinator=False,
            agent_id=None,
            name="Coordinator Unity",
            first_name="",
            surname="",
            user_desktop_for=lambda user_id: None,
        ),
        voice=SimpleNamespace(provider="cartesia", id=""),
        voice_call=SimpleNamespace(outbound=False, channel="unify_meet"),
        is_coordinator=False,
        org_id=None,
        unify_key="",
    )

    session_holder = {}

    def _fake_session_factory(*args, **kwargs):
        session = _FakeSession(*args, **kwargs)
        session_holder["session"] = session
        return session

    monkeypatch.setattr(call_script, "event_broker", _FakeEventBroker())
    monkeypatch.setattr(call_script, "SESSION_DETAILS", fake_session_details)
    monkeypatch.setattr(call_script, "AgentSession", _fake_session_factory)
    monkeypatch.setattr(call_script, "Assistant", _FakeAssistant)
    monkeypatch.setattr(call_script, "UnifyLLM", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        call_script,
        "build_voice_agent_prompt",
        lambda **kwargs: SimpleNamespace(flatten=lambda: "system prompt"),
    )
    monkeypatch.setattr(call_script, "start_event_broker_receive", _noop_async)
    monkeypatch.setattr(call_script, "hydrate_fast_brain_history", _empty_history)
    monkeypatch.setattr(call_script, "publish_call_started", _noop_async)
    monkeypatch.setattr(call_script, "publish_call_ended", _noop_async)
    monkeypatch.setattr(call_script, "delete_livekit_room", _noop_async)
    monkeypatch.setattr(
        call_script,
        "FastBrainCreditGateMonitor",
        _FakeCreditGateMonitor,
    )
    monkeypatch.setattr(
        call_script,
        "_recording_audio_frames",
        _fake_recording_audio_frames,
    )
    monkeypatch.setattr(
        call_script,
        "_preload_recorded_opening_pcm",
        lambda _config: {},
    )
    monkeypatch.setattr(
        call_script,
        "create_end_call",
        lambda *args, **kwargs: _noop_end_call,
    )
    monkeypatch.setattr(
        call_script,
        "setup_participant_disconnect_handler",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(call_script, "RoomInputOptions", lambda **kwargs: object())
    monkeypatch.setattr(call_script, "EnglishModel", lambda: object())
    monkeypatch.setattr(call_script.cartesia, "TTS", lambda **kwargs: object())
    monkeypatch.setattr(call_script.elevenlabs, "TTS", lambda **kwargs: object())
    if hasattr(call_script, "noise_cancellation"):
        monkeypatch.setattr(call_script.noise_cancellation, "BVC", lambda: object())
    monkeypatch.setattr(call_script, "STT", object())
    monkeypatch.setattr(call_script, "VAD", object())

    await call_script.entrypoint(_FakeJobContext())

    session = session_holder["session"]
    assert audio_sources == ["/tmp/recorded-opener.wav"]
    assert session.generate_reply_calls == []
    assert len(session.say_calls) == 1

    text, kwargs = session.say_calls[0]
    assert text == "Hi, I'm your coordinator unify."
    assert kwargs["audio"] is fake_audio
    assert kwargs["allow_interruptions"] is True
    assert kwargs["add_to_chat_ctx"] is True

    ready_index = next(
        i
        for i, item in enumerate(sequence)
        if item[0] == "data" and item[1] == {"type": "ready_to_speak"}
    )
    say_index = next(i for i, item in enumerate(sequence) if item[0] == "say")
    assert ready_index < say_index


@pytest.mark.asyncio
@pytest.mark.parametrize("interrupt_walkie", [True, False])
async def test_walkie_opener_arms_bridge_only_on_early_interruption(
    monkeypatch,
    interrupt_walkie,
):
    from livekit.agents import llm
    from unify.conversation_manager.medium_scripts import call as call_script

    sequence = []
    audio_sources = []
    fake_audio = object()
    # Only the first (walkie) segment is interrupted when interrupt_walkie.
    interrupt_flags = [True] if interrupt_walkie else []
    contact = {"contact_id": 1, "first_name": "User", "surname": "Example"}
    boss = {"contact_id": 1, "first_name": "User", "surname": "Example"}

    class _ImmediateAwaitable:
        def __await__(self):
            async def _done():
                return None

            return _done().__await__()

    class _FakeSpeechHandle:
        def __init__(self, interrupted, *, spoken_text: str = ""):
            self.interrupted = interrupted
            self._cancelled = False
            self.chat_items = (
                [SimpleNamespace(role="assistant", text_content=spoken_text)]
                if interrupted
                else []
            )

        async def wait_for_playout(self):
            return None

        def done(self) -> bool:
            return self._cancelled

        def interrupt(self, *, force: bool = False) -> "_FakeSpeechHandle":
            self._cancelled = True
            self.interrupted = True
            return self

    class _FakeLocalParticipant:
        async def publish_data(self, payload, *, topic=None, reliable=False):
            sequence.append(("data", json.loads(payload.decode()), topic, reliable))

    class _FakeRoom:
        name = "fake-room"
        local_participant = _FakeLocalParticipant()

        def on(self, *args, **kwargs):
            return lambda fn: fn

    class _FakeJobContext:
        def __init__(self):
            self.room = _FakeRoom()
            self.job = SimpleNamespace(
                metadata=json.dumps(
                    {
                        "voice_provider": "cartesia",
                        "voice_id": "",
                        "outbound": False,
                        "channel": "unify_meet",
                        "contact": contact,
                        "boss": boss,
                        "assistant_bio": "Assistant bio",
                        "assistant_id": "123",
                        "user_id": "user-123",
                        "assistant_name": "Coordinator Unity",
                        "is_coordinator": False,
                        "opening_config": {
                            "mode": "recorded",
                            "recording_asset": "coordinator_onboarding_intro",
                            "source": "coordinator_onboarding_intro",
                        },
                    },
                ),
            )

        async def connect(self):
            return None

        def add_shutdown_callback(self, cb):
            pass

        def shutdown(self, reason=""):
            pass

    class _FakeEventBroker:
        def set_logger(self, fb_logger):
            pass

        def register_callback(self, channel, handler):
            pass

        async def publish(self, channel, message):
            sequence.append(("broker", channel, message))
            return 1

        def reinit_socket(self):
            pass

    class _FakeSession:
        def __init__(self, *args, **kwargs):
            self._chat_ctx = llm.ChatContext()
            self.current_agent = None
            self._events = {}
            self.agent_state = "listening"
            self.current_speech = None
            self.say_calls = []
            self.speech_handles = []
            self.generate_reply_calls = []

        @property
        def history(self):
            return self._chat_ctx

        def on(self, event_name):
            def _decorator(fn):
                self._events[event_name] = fn
                return fn

            return _decorator

        async def start(self, room, agent, room_input_options=None):
            self.current_agent = agent

        def generate_reply(self, **kwargs):
            self.generate_reply_calls.append(kwargs)
            return _ImmediateAwaitable()

        def say(self, text, **kwargs):
            sequence.append(("say", text, kwargs))
            self.say_calls.append((text, kwargs))
            interrupted = interrupt_flags.pop(0) if interrupt_flags else False
            spoken_text = "Hi, I'm T dash W 1 N." if interrupted else ""
            handle = _FakeSpeechHandle(interrupted, spoken_text=spoken_text)
            self.speech_handles.append(handle)
            return handle

        def interrupt(self):
            pass

    class _FakeAssistant:
        def __init__(self, *args, **kwargs):
            self._chat_ctx = llm.ChatContext()
            self.call_received = True
            self.user_turn_generating = False
            self._pending_opening_bridge = None

        def set_call_received(self):
            self.call_received = True

        def set_credit_gate_state_provider(self, provider):
            self.credit_gate_state_provider = provider

    class _FakeCreditGateMonitor:
        state = SimpleNamespace(allowed=True)

        async def run(self):
            return None

    async def _noop_async(*args, **kwargs):
        return None

    async def _empty_history(*args, **kwargs):
        return []

    async def _noop_end_call():
        return None

    def _fake_recording_audio_frames(source, **_kwargs):
        audio_sources.append(source)
        return fake_audio

    fake_session_details = SimpleNamespace(
        user=SimpleNamespace(id="user-123"),
        assistant=SimpleNamespace(
            about="Assistant bio",
            is_coordinator=False,
            agent_id=None,
            name="Coordinator Unity",
            first_name="",
            surname="",
            user_desktop_for=lambda user_id: None,
        ),
        voice=SimpleNamespace(provider="cartesia", id=""),
        voice_call=SimpleNamespace(outbound=False, channel="unify_meet"),
        is_coordinator=False,
        org_id=None,
        unify_key="",
    )

    session_holder = {}

    def _fake_session_factory(*args, **kwargs):
        session = _FakeSession(*args, **kwargs)
        session_holder["session"] = session
        return session

    monkeypatch.setattr(call_script, "event_broker", _FakeEventBroker())
    monkeypatch.setattr(call_script, "SESSION_DETAILS", fake_session_details)
    monkeypatch.setattr(call_script, "AgentSession", _fake_session_factory)
    monkeypatch.setattr(call_script, "Assistant", _FakeAssistant)
    monkeypatch.setattr(call_script, "UnifyLLM", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        call_script,
        "build_voice_agent_prompt",
        lambda **kwargs: SimpleNamespace(flatten=lambda: "system prompt"),
    )
    monkeypatch.setattr(call_script, "start_event_broker_receive", _noop_async)
    monkeypatch.setattr(call_script, "hydrate_fast_brain_history", _empty_history)
    monkeypatch.setattr(call_script, "publish_call_started", _noop_async)
    monkeypatch.setattr(call_script, "publish_call_ended", _noop_async)
    monkeypatch.setattr(call_script, "delete_livekit_room", _noop_async)
    monkeypatch.setattr(
        call_script,
        "FastBrainCreditGateMonitor",
        _FakeCreditGateMonitor,
    )
    monkeypatch.setattr(
        call_script,
        "_recording_audio_frames",
        _fake_recording_audio_frames,
    )
    monkeypatch.setattr(
        call_script,
        "_preload_recorded_opening_pcm",
        lambda _config: {},
    )
    monkeypatch.setattr(
        call_script,
        "create_end_call",
        lambda *args, **kwargs: _noop_end_call,
    )
    monkeypatch.setattr(
        call_script,
        "setup_participant_disconnect_handler",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(call_script, "RoomInputOptions", lambda **kwargs: object())
    monkeypatch.setattr(call_script, "EnglishModel", lambda: object())
    monkeypatch.setattr(call_script.cartesia, "TTS", lambda **kwargs: object())
    monkeypatch.setattr(call_script.elevenlabs, "TTS", lambda **kwargs: object())
    if hasattr(call_script, "noise_cancellation"):
        monkeypatch.setattr(call_script.noise_cancellation, "BVC", lambda: object())
    monkeypatch.setattr(call_script, "STT", object())
    monkeypatch.setattr(call_script, "VAD", object())

    await call_script.entrypoint(_FakeJobContext())

    session = session_holder["session"]
    assert session.generate_reply_calls == []
    assert len(session.say_calls) == 1
    assert inspect.isasyncgen(session.say_calls[0][0])
    assert audio_sources[0] == "asset://coordinator_onboarding_intro"

    if interrupt_walkie:
        assert session.speech_handles[0].interrupted is True
        assert session.current_agent._pending_opening_bridge is not None
    else:
        assert session.speech_handles[0].interrupted is False
        assert session.current_agent._pending_opening_bridge is None


def test_recorded_opening_builtin_asset_validates_without_single_transcript():
    from unify.conversation_manager.medium_scripts import call as call_script

    # The builtin onboarding opener is segmented, so it must validate without a
    # single inline transcript or recording source.
    config = call_script._normalize_call_opening_config(
        {
            "mode": "recorded",
            "recording_asset": "coordinator_onboarding_intro",
            "source": "coordinator_onboarding_intro",
        },
    )
    assert config["recording_asset"] == "coordinator_onboarding_intro"


@pytest.mark.asyncio
async def test_preloaded_recorded_opening_frames_match_file_decode():
    from unify.conversation_manager.medium_scripts import call as call_script

    config = {
        "mode": "recorded",
        "recording_asset": "coordinator_onboarding_intro",
    }
    preloaded = call_script._preload_recorded_opening_pcm(config)
    assert len(preloaded) == 2
    assert set(preloaded) == {
        "coordinator_onboarding_intro",
        "coordinator_onboarding_static_bridge",
    }

    asset_key = "coordinator_onboarding_intro"
    file_frames = []
    async for frame in call_script._recording_audio_frames(f"asset://{asset_key}"):
        file_frames.append(frame)
    preloaded_frames = []
    async for frame in call_script._preloaded_audio_frames(preloaded[asset_key]):
        preloaded_frames.append(frame)

    assert len(preloaded_frames) == len(file_frames)
    assert preloaded_frames[0].sample_rate == file_frames[0].sample_rate
    assert preloaded_frames[-1].data == file_frames[-1].data


def test_walkie_opener_timed_chunks_cover_static_removal_transition():
    from unify.conversation_manager.medium_scripts import call as call_script

    spec = call_script._RECORDED_OPENINGS["coordinator_onboarding_intro"]
    chunks = spec["timed_chunks"]
    bridge = spec["bridge"]
    walkie = chunks[:20]
    clean = chunks[20:]

    assert len(walkie) == 20
    assert walkie[0]["text"] == "Hi, I'm T dash W 1 N."
    assert walkie[-1]["text"] == "Also, let me remove this voice static."
    for chunk in walkie:
        assert "Much better." not in chunk["text"]
    assert clean[0]["text"] == "Much better."
    assert float(clean[0]["start_time"]) == pytest.approx(
        call_script._COORDINATOR_ONBOARDING_CLEAN_START_TIME,
    )

    assert "remove this voice static" in bridge["transcript"]
    assert "Much better." in bridge["transcript"]

    for key in (spec["asset"], bridge["asset"]):
        assert key in call_script._RECORDED_OPENING_ASSETS

    combined = (
        call_script._recorded_opening_timed_transcript(chunks) + bridge["transcript"]
    )
    assert "Krispy Kreme" not in combined
    assert "Actually, first lets turn off this really annoying music." not in combined
    assert "There we go, now I'll pull up the platform." not in combined


def test_coordinator_onboarding_timed_chunks_match_audio():
    import soundfile as sf
    from importlib import resources

    from unify.conversation_manager.medium_scripts import call as call_script

    chunks = call_script._COORDINATOR_ONBOARDING_TIMED_CHUNKS
    assert len(chunks) == 23
    assert chunks[0]["text"] == "Hi, I'm T dash W 1 N."
    assert chunks[19]["text"] == "Also, let me remove this voice static."
    assert chunks[20]["text"] == "Much better."

    with resources.as_file(
        resources.files("unify.assets.audio") / "twin-onboarding-intro.mp3",
    ) as intro_path:
        full_duration = sf.info(intro_path).duration

    assert float(chunks[-1]["end_time"]) == pytest.approx(full_duration, abs=0.01)
    assert float(chunks[20]["start_time"]) == pytest.approx(
        call_script._COORDINATOR_ONBOARDING_CLEAN_START_TIME,
        abs=0.01,
    )

    for index in range(1, len(chunks)):
        assert float(chunks[index]["start_time"]) == pytest.approx(
            float(chunks[index - 1]["end_time"]),
            abs=1e-6,
        )


@pytest.mark.asyncio
async def test_on_user_turn_completed_schedules_pending_opening_bridge():
    from types import SimpleNamespace

    from unify.conversation_manager.medium_scripts import call as call_script

    scheduled: list[bool] = []

    # The bridge callable is synchronous: it enqueues the bridge say but does
    # NOT await playout, so the hook returns immediately and the reply is
    # generated concurrently (and queues behind the bridge).
    def _schedule_bridge() -> None:
        scheduled.append(True)

    self = SimpleNamespace(
        _pending_opening_bridge=_schedule_bridge,
        _user_speech_logged=False,
        _user_turn_seq=0,
        _opening_pending=False,
        _first_user_turn=asyncio.Event(),
    )
    new_message = SimpleNamespace(text_content="hi there")

    await call_script.Assistant.on_user_turn_completed(
        self,
        turn_ctx=None,
        new_message=new_message,
    )

    # The bridge is scheduled exactly once and disarmed so later turns are
    # normal.
    assert scheduled == [True]
    assert self._pending_opening_bridge is None
    assert self._user_speech_logged is True
    assert self._user_turn_seq == 1


@pytest.mark.asyncio
async def test_on_user_turn_completed_without_bridge_is_normal():
    from types import SimpleNamespace

    from unify.conversation_manager.medium_scripts import call as call_script

    self = SimpleNamespace(
        _pending_opening_bridge=None,
        _user_speech_logged=False,
        _user_turn_seq=0,
        _opening_pending=False,
        _first_user_turn=asyncio.Event(),
    )
    new_message = SimpleNamespace(text_content="hello")

    await call_script.Assistant.on_user_turn_completed(
        self,
        turn_ctx=None,
        new_message=new_message,
    )

    assert self._pending_opening_bridge is None
    assert self._user_speech_logged is True


@pytest.mark.asyncio
class TestFastBrainGuidanceFlow:
    """Coverage for guidance delivery in the TTS fast brain path."""

    async def test_notify_only_guidance_injects_context_without_direct_speech(
        self,
        monkeypatch,
    ):
        """Guidance with should_speak=False injects into both chat contexts as
        silent awareness context, without calling session.say() directly (only
        should_speak=True guidance is spoken)."""
        from livekit.agents import llm
        from unify.conversation_manager.medium_scripts import call as call_script

        contact = {
            "contact_id": 2,
            "first_name": "Caller",
            "surname": "Example",
            "phone_number": "+15550100002",
            "email_address": "caller@example.com",
        }
        boss = {
            "contact_id": 1,
            "first_name": "Manager",
            "surname": "Example",
            "phone_number": "+15550100001",
            "email_address": "manager@example.com",
        }

        class _ImmediateAwaitable:
            def __await__(self):
                async def _done():
                    return None

                return _done().__await__()

        class _FakeLocalParticipant:
            async def publish_data(self, *args, **kwargs):
                pass

        class _FakeRoom:
            name = "fake-room"
            local_participant = _FakeLocalParticipant()

            def on(self, *args, **kwargs):
                return lambda fn: fn

        class _FakeJobContext:
            def __init__(self):
                self.room = _FakeRoom()
                self.job = SimpleNamespace()

            async def connect(self):
                return None

            def add_shutdown_callback(self, cb):
                pass

            def shutdown(self, reason=""):
                pass

        class _FakeEventBroker:
            def __init__(self):
                self.callbacks = {}

            def set_logger(self, fb_logger):
                pass

            def register_callback(self, channel, handler):
                self.callbacks[channel] = handler

            async def publish(self, channel, message):
                return 1

        fake_session_holder = {}

        class _FakeSession:
            def __init__(self, *args, **kwargs):
                self._chat_ctx = llm.ChatContext()
                self.current_agent = None
                self._events = {}
                self.generate_reply_calls = 0
                self.say_calls = []
                self.agent_state = "listening"
                self.current_speech = None
                fake_session_holder["session"] = self

            @property
            def history(self):
                return self._chat_ctx

            def on(self, event_name):
                def _decorator(fn):
                    self._events[event_name] = fn
                    return fn

                return _decorator

            async def start(self, room, agent, room_input_options=None):
                self.current_agent = agent

            def generate_reply(self, **kwargs):
                self.generate_reply_calls += 1
                return _ImmediateAwaitable()

            def say(self, text, **kwargs):
                self.say_calls.append(text)
                return _ImmediateAwaitable()

            def interrupt(self):
                pass

        class _FakeAssistant:
            def __init__(self, *args, **kwargs):
                self._chat_ctx = llm.ChatContext()
                self.call_received = True
                self.user_turn_generating = False
                self._user_turn_seq = 0
                self._slow_brain_responded_turn = -1
                self._buffers_since_slow_reply = 0
                self._tts_seq = 0
                self._active_tts = None
                self._pending_continuation = None
                self._publish_voice_interrupt = None

            def set_call_received(self):
                self.call_received = True

            def set_credit_gate_state_provider(self, provider):
                self.credit_gate_state_provider = provider

        async def _noop_async(*args, **kwargs):
            return None

        async def _noop_end_call():
            return None

        fake_broker = _FakeEventBroker()
        fake_session_details = SimpleNamespace(
            populate_from_env=lambda: None,
            user=SimpleNamespace(id="user-123"),
            voice=SimpleNamespace(provider="cartesia", id=""),
            assistant=SimpleNamespace(
                about="Assistant bio",
                name="Ava",
                first_name="Assistant",
                surname="Example",
                agent_id=None,
                user_desktop_for=lambda user_id: None,
            ),
            voice_call=SimpleNamespace(
                outbound=False,
                channel="unify_meet",
                contact_json=json.dumps(contact),
                boss_json=json.dumps(boss),
            ),
            is_coordinator=False,
            org_id=None,
            unify_key="",
        )

        monkeypatch.setattr(call_script, "event_broker", fake_broker)
        monkeypatch.setattr(call_script, "SESSION_DETAILS", fake_session_details)
        monkeypatch.setattr(call_script, "AgentSession", _FakeSession)
        monkeypatch.setattr(call_script, "Assistant", _FakeAssistant)
        monkeypatch.setattr(call_script, "UnifyLLM", lambda *args, **kwargs: object())
        monkeypatch.setattr(
            call_script,
            "build_voice_agent_prompt",
            lambda **kwargs: SimpleNamespace(flatten=lambda: "system prompt"),
        )
        monkeypatch.setattr(call_script, "start_event_broker_receive", _noop_async)
        monkeypatch.setattr(call_script, "publish_call_started", _noop_async)
        monkeypatch.setattr(
            call_script,
            "create_end_call",
            lambda *args, **kwargs: _noop_end_call,
        )
        monkeypatch.setattr(
            call_script,
            "setup_participant_disconnect_handler",
            lambda *args, **kwargs: None,
        )
        monkeypatch.setattr(call_script, "RoomInputOptions", lambda **kwargs: object())
        monkeypatch.setattr(call_script, "EnglishModel", lambda: object())
        monkeypatch.setattr(call_script.cartesia, "TTS", lambda **kwargs: object())
        monkeypatch.setattr(call_script.elevenlabs, "TTS", lambda **kwargs: object())
        if hasattr(call_script, "noise_cancellation"):
            monkeypatch.setattr(call_script.noise_cancellation, "BVC", lambda: object())

        monkeypatch.setattr(call_script, "STT", object())
        monkeypatch.setattr(call_script, "VAD", object())

        import unify.common.llm_client as _llm_mod

        class _FakeGreetingClient:
            async def generate(self, **kwargs):
                return "Hello!"

        monkeypatch.setattr(
            _llm_mod,
            "new_llm_client",
            lambda *a, **kw: _FakeGreetingClient(),
        )

        await call_script.entrypoint(_FakeJobContext())

        session = fake_session_holder["session"]
        session.say_calls.clear()

        # Send notify-only guidance (should_speak=False)
        guidance_cb = fake_broker.callbacks["app:call:notification"]
        guidance_cb({"payload": {"message": "No, there is no contact named Bob."}})

        # Notification should be in both chat contexts
        mirror_texts = [
            item.text_content or ""
            for item in session._chat_ctx.items
            if getattr(item, "type", None) == "message"
        ]
        assert any("No, there is no contact named Bob." in txt for txt in mirror_texts)

        agent_texts = [
            item.text_content or ""
            for item in session.current_agent._chat_ctx.items
            if getattr(item, "type", None) == "message"
        ]
        assert any("No, there is no contact named Bob." in txt for txt in agent_texts)

        # say() must NOT fire directly — the structured notification evaluator
        # decides asynchronously whether to speak.
        assert (
            len(session.say_calls) == 0
        ), "Notify-only guidance must NOT trigger session.say() directly."

        from unify.conversation_manager.events import AssistantTurnInjected

        injected = AssistantTurnInjected(
            contact={"contact_id": 2},
            content="I just gave the prerecorded intro.",
            source="test",
        )
        guidance_cb(json.loads(injected.to_json()))

        agent_texts = [
            item.text_content or ""
            for item in session.current_agent._chat_ctx.items
            if getattr(item, "type", None) == "message"
        ]
        assert any("I just gave the prerecorded intro." in txt for txt in agent_texts)
        mirror_texts = [
            item.text_content or ""
            for item in session._chat_ctx.items
            if getattr(item, "type", None) == "message"
        ]
        assert any("I just gave the prerecorded intro." in txt for txt in mirror_texts)
        assert len(session.say_calls) == 0

    async def test_buffered_notification_is_applied_before_opening_greeting(
        self,
        monkeypatch,
    ):
        """Notifications that arrive before session_ready should land in greeting context."""

        from livekit.agents import llm

        if not hasattr(llm, "Tool"):
            llm.Tool = object  # type: ignore[attr-defined]
        from unify.conversation_manager.medium_scripts import call as call_script

        contact = {
            "contact_id": 2,
            "first_name": "Caller",
            "surname": "Example",
            "phone_number": "+15550100002",
            "email_address": "caller@example.com",
        }
        boss = {
            "contact_id": 1,
            "first_name": "Manager",
            "surname": "Example",
            "phone_number": "+15550100001",
            "email_address": "manager@example.com",
        }
        buffered_text = (
            "Background context: scheduled Morning briefing is due now "
            "(task_id=101, scheduled_for=2026-04-10T09:00:00+00:00)."
        )

        class _ImmediateAwaitable:
            def __await__(self):
                async def _done():
                    return None

                return _done().__await__()

        class _FakeLocalParticipant:
            async def publish_data(self, *args, **kwargs):
                pass

        class _FakeRoom:
            name = "fake-room"
            local_participant = _FakeLocalParticipant()

            def on(self, *args, **kwargs):
                return lambda fn: fn

        class _FakeJobContext:
            def __init__(self):
                self.room = _FakeRoom()
                self.job = SimpleNamespace()

            async def connect(self):
                return None

            def add_shutdown_callback(self, cb):
                pass

            def shutdown(self, reason=""):
                pass

        class _FakeEventBroker:
            def __init__(self):
                self.callbacks = {}

            def set_logger(self, fb_logger):
                pass

            def register_callback(self, channel, handler):
                self.callbacks[channel] = handler
                if channel == "app:call:notification":
                    handler({"payload": {"message": buffered_text}})

            async def publish(self, channel, message):
                return 1

        fake_session_holder = {}

        class _FakeSession:
            def __init__(self, *args, **kwargs):
                self._chat_ctx = llm.ChatContext()
                self.current_agent = None
                self._events = {}
                self.generate_reply_calls = 0
                self.say_calls = []
                self.agent_state = "listening"
                self.current_speech = None
                fake_session_holder["session"] = self

            @property
            def history(self):
                return self._chat_ctx

            def on(self, event_name):
                def _decorator(fn):
                    self._events[event_name] = fn
                    return fn

                return _decorator

            async def start(self, room, agent, room_input_options=None):
                self.current_agent = agent

            def generate_reply(self, **kwargs):
                self.generate_reply_calls += 1
                return _ImmediateAwaitable()

            def say(self, text, **kwargs):
                self.say_calls.append(text)
                return _ImmediateAwaitable()

            def interrupt(self):
                pass

        class _FakeAssistant:
            def __init__(self, *args, **kwargs):
                self._chat_ctx = llm.ChatContext()
                self.call_received = True
                self.user_turn_generating = False
                self._user_turn_seq = 0
                self._slow_brain_responded_turn = -1
                self._buffers_since_slow_reply = 0
                self._tts_seq = 0
                self._active_tts = None
                self._pending_continuation = None
                self._publish_voice_interrupt = None

            def set_call_received(self):
                self.call_received = True

            def set_credit_gate_state_provider(self, provider):
                self.credit_gate_state_provider = provider

        async def _noop_async(*args, **kwargs):
            return None

        async def _noop_end_call():
            return None

        fake_broker = _FakeEventBroker()
        fake_session_details = SimpleNamespace(
            populate_from_env=lambda: None,
            voice=SimpleNamespace(provider="cartesia", id=""),
            assistant=SimpleNamespace(
                about="Assistant bio",
                name="Ava",
                first_name="Assistant",
                surname="Example",
                agent_id=None,
                user_desktop_for=lambda user_id: None,
            ),
            user=SimpleNamespace(id="default"),
            is_coordinator=False,
            org_id=None,
            voice_call=SimpleNamespace(
                outbound=False,
                channel="unify_meet",
                contact_json=json.dumps(contact),
                boss_json=json.dumps(boss),
            ),
            unify_key="",
        )

        monkeypatch.setattr(call_script, "SESSION_DETAILS", fake_session_details)
        monkeypatch.setattr(call_script, "event_broker", fake_broker)
        monkeypatch.setattr(call_script, "Assistant", _FakeAssistant)
        monkeypatch.setattr(call_script, "AgentSession", _FakeSession)
        monkeypatch.setattr(call_script, "delete_livekit_room", _noop_async)
        monkeypatch.setattr(call_script, "publish_call_started", _noop_async)
        monkeypatch.setattr(call_script, "publish_call_ended", _noop_async)
        monkeypatch.setattr(call_script, "start_event_broker_receive", _noop_async)
        monkeypatch.setattr(
            call_script,
            "setup_participant_disconnect_handler",
            _noop_async,
        )
        monkeypatch.setattr(call_script, "create_end_call", _noop_end_call)
        monkeypatch.setattr(call_script, "dispatch_livekit_agent", _noop_async)
        monkeypatch.setattr(call_script, "RoomInputOptions", lambda **kwargs: object())
        monkeypatch.setattr(call_script, "EnglishModel", lambda: object())
        monkeypatch.setattr(call_script.cartesia, "TTS", lambda **kwargs: object())
        monkeypatch.setattr(call_script.elevenlabs, "TTS", lambda **kwargs: object())
        if hasattr(call_script, "noise_cancellation"):
            monkeypatch.setattr(
                call_script.noise_cancellation,
                "BVC",
                lambda: object(),
            )

        monkeypatch.setattr(call_script, "STT", object())
        monkeypatch.setattr(call_script, "VAD", object())

        captured = {}

        import unify.common.llm_client as _llm_mod

        class _FakeGreetingClient:
            async def generate(self, **kwargs):
                captured["messages"] = kwargs["messages"]
                return "Hello!"

        monkeypatch.setattr(
            _llm_mod,
            "new_llm_client",
            lambda *a, **kw: _FakeGreetingClient(),
        )

        await call_script.entrypoint(_FakeJobContext())

        serialized_messages = json.dumps(captured["messages"], default=str)
        assert buffered_text in serialized_messages

    async def test_unify_llm_preserves_base_system_prompt_with_notification(
        self,
        monkeypatch,
    ):
        """System instructions and notifications should both survive conversion."""
        from livekit.agents import llm
        from unify.conversation_manager import livekit_unify_adapter as adapter_module

        captured = {}

        class _DummyClient:
            def set_stream(self, enabled):
                captured["set_stream"] = enabled

            async def generate(self, **kwargs):
                captured["generate_kwargs"] = kwargs

                async def _empty_stream():
                    if False:
                        yield ""

                return _empty_stream()

        def _fake_new_llm_client(model, **kwargs):
            captured["client_model"] = model
            captured["client_kwargs"] = kwargs
            return _DummyClient()

        monkeypatch.setattr(adapter_module, "new_llm_client", _fake_new_llm_client)

        chat_ctx = llm.ChatContext()
        chat_ctx.add_message(role="system", content="BASE_PROMPT")
        chat_ctx.add_message(
            role="user",
            content="Do I have a contact named Bob?",
        )
        chat_ctx.add_message(
            role="system",
            content="[notification] No, there is no contact named Bob.",
        )
        chat_ctx.add_message(role="assistant", content="Let me check on that.")

        from unify.settings import SETTINGS

        stream = adapter_module.UnifyLLM(
            model=SETTINGS.conversation.FAST_BRAIN_MODEL,
        ).chat(
            chat_ctx=chat_ctx,
        )
        await stream._run()

        messages = captured["generate_kwargs"]["messages"]
        system_texts = [m["content"] for m in messages if m["role"] == "system"]
        all_system_text = "\n".join(system_texts)
        assert "BASE_PROMPT" in all_system_text
        assert "[notification] No, there is no contact named Bob." in all_system_text

    async def test_tts_guidance_received_while_user_speaking_is_replied_after_speech_ends(
        self,
        monkeypatch,
    ):
        """Guidance arriving mid-speech should be surfaced once speech ends."""
        from livekit.agents import llm
        from unify.conversation_manager.medium_scripts import call as call_script

        contact = {
            "contact_id": 2,
            "first_name": "Caller",
            "surname": "Example",
            "phone_number": "+15550100002",
            "email_address": "caller@example.com",
        }
        boss = {
            "contact_id": 1,
            "first_name": "Manager",
            "surname": "Example",
            "phone_number": "+15550100001",
            "email_address": "manager@example.com",
        }

        class _ImmediateAwaitable:
            def __await__(self):
                async def _done():
                    return None

                return _done().__await__()

        class _FakeLocalParticipant:
            async def publish_data(self, *args, **kwargs):
                pass

        class _FakeRoom:
            name = "fake-room"
            local_participant = _FakeLocalParticipant()

            def on(self, *args, **kwargs):
                return lambda fn: fn

        class _FakeJobContext:
            def __init__(self):
                self.room = _FakeRoom()
                self.job = SimpleNamespace()

            async def connect(self):
                return None

            def add_shutdown_callback(self, cb):
                pass

            def shutdown(self, reason=""):
                pass

        class _FakeEventBroker:
            def __init__(self):
                self.callbacks = {}

            def set_logger(self, fb_logger):
                pass

            def register_callback(self, channel, handler):
                self.callbacks[channel] = handler

            async def publish(self, channel, message):
                return 1

        fake_session_holder = {}

        class _FakeSession:
            def __init__(self, *args, **kwargs):
                self._chat_ctx = llm.ChatContext()
                self.current_agent = None
                self._events = {}
                self.generate_reply_calls = 0
                self.say_calls = []
                self.agent_state = "listening"
                self.current_speech = None
                fake_session_holder["session"] = self

            @property
            def history(self):
                return self._chat_ctx

            def on(self, event_name):
                def _decorator(fn):
                    self._events[event_name] = fn
                    return fn

                return _decorator

            async def start(self, room, agent, room_input_options=None):
                self.current_agent = agent

            def generate_reply(self, **kwargs):
                self.generate_reply_calls += 1
                return _ImmediateAwaitable()

            def say(self, text, **kwargs):
                self.say_calls.append(text)
                return _ImmediateAwaitable()

            def interrupt(self):
                pass

        class _FakeAssistant:
            def __init__(self, *args, **kwargs):
                self._chat_ctx = llm.ChatContext()
                self.call_received = True
                self.user_turn_generating = False
                self._user_turn_seq = 0
                self._slow_brain_responded_turn = -1
                self._buffers_since_slow_reply = 0
                self._tts_seq = 0
                self._active_tts = None
                self._pending_continuation = None
                self._publish_voice_interrupt = None

            def set_call_received(self):
                self.call_received = True

            def set_credit_gate_state_provider(self, provider):
                self.credit_gate_state_provider = provider

        async def _noop_async(*args, **kwargs):
            return None

        async def _noop_end_call():
            return None

        fake_broker = _FakeEventBroker()
        fake_session_details = SimpleNamespace(
            populate_from_env=lambda: None,
            voice=SimpleNamespace(provider="cartesia", id=""),
            voice_call=SimpleNamespace(
                outbound=False,
                channel="unify_meet",
                contact_json=json.dumps(contact),
                boss_json=json.dumps(boss),
            ),
            assistant=SimpleNamespace(
                about="Assistant bio",
                name="Ava",
                user_desktop_for=lambda user_id: None,
            ),
            user=SimpleNamespace(id="default"),
            is_coordinator=False,
            org_id=None,
            unify_key="",
        )

        monkeypatch.setattr(call_script, "event_broker", fake_broker)
        monkeypatch.setattr(call_script, "SESSION_DETAILS", fake_session_details)
        monkeypatch.setattr(call_script, "AgentSession", _FakeSession)
        monkeypatch.setattr(call_script, "Assistant", _FakeAssistant)
        monkeypatch.setattr(call_script, "UnifyLLM", lambda *args, **kwargs: object())
        monkeypatch.setattr(
            call_script,
            "build_voice_agent_prompt",
            lambda **kwargs: SimpleNamespace(flatten=lambda: "system prompt"),
        )
        monkeypatch.setattr(call_script, "start_event_broker_receive", _noop_async)
        monkeypatch.setattr(call_script, "publish_call_started", _noop_async)
        monkeypatch.setattr(
            call_script,
            "create_end_call",
            lambda *args, **kwargs: _noop_end_call,
        )
        monkeypatch.setattr(
            call_script,
            "setup_participant_disconnect_handler",
            lambda *args, **kwargs: None,
        )
        monkeypatch.setattr(call_script, "RoomInputOptions", lambda **kwargs: object())
        monkeypatch.setattr(call_script, "EnglishModel", lambda: object())
        monkeypatch.setattr(call_script.cartesia, "TTS", lambda **kwargs: object())
        monkeypatch.setattr(call_script.elevenlabs, "TTS", lambda **kwargs: object())
        if hasattr(call_script, "noise_cancellation"):
            monkeypatch.setattr(call_script.noise_cancellation, "BVC", lambda: object())

        monkeypatch.setattr(call_script, "STT", object())
        monkeypatch.setattr(call_script, "VAD", object())

        import unify.common.llm_client as _llm_mod

        class _FakeGreetingClient:
            async def generate(self, **kwargs):
                return "Hello!"

        monkeypatch.setattr(
            _llm_mod,
            "new_llm_client",
            lambda *a, **kw: _FakeGreetingClient(),
        )

        await call_script.entrypoint(_FakeJobContext())

        session = fake_session_holder["session"]
        session.say_calls.clear()
        guidance_cb = fake_broker.callbacks["app:call:notification"]
        agent_state_cb = session._events["agent_state_changed"]

        # User is speaking — guidance with should_speak=True arrives and is queued
        state_cb = session._events["user_state_changed"]
        state_cb(SimpleNamespace(new_state="speaking"))
        guidance_cb(
            {
                "payload": {
                    "message": "No, there's no contact named Bob.",
                    "should_speak": True,
                },
            },
        )
        assert (
            len(session.say_calls) == 0
        ), "Queued speech must not fire while user is speaking."

        # User stops speaking — say() must NOT fire from user_state_changed alone
        state_cb(SimpleNamespace(new_state="listening"))
        assert (
            len(session.say_calls) == 0
        ), "Queued speech must not fire from user_state_changed (race condition)."

        # Agent settles to listening — say() fires now
        agent_state_cb(SimpleNamespace(new_state="listening"))

        # maybe_speak_queued speaks synchronously; yield just in case.
        await asyncio.sleep(0)

        assert len(session.say_calls) == 1, (
            "Guidance that arrives while the user is speaking should be surfaced "
            "via session.say() after the agent settles to listening."
        )
        assert session.say_calls[0] == "No, there's no contact named Bob."

    async def test_queued_speech_waits_for_agent_thinking_and_speaking_cycle(
        self,
        monkeypatch,
    ):
        """Guidance arriving while the agent is thinking/speaking should wait
        for the full cycle to complete before session.say() fires."""
        from livekit.agents import llm
        from unify.conversation_manager.medium_scripts import call as call_script

        contact = {
            "contact_id": 2,
            "first_name": "Caller",
            "surname": "Example",
            "phone_number": "+15550100002",
            "email_address": "caller@example.com",
        }
        boss = {
            "contact_id": 1,
            "first_name": "Manager",
            "surname": "Example",
            "phone_number": "+15550100001",
            "email_address": "manager@example.com",
        }

        class _ImmediateAwaitable:
            def __await__(self):
                async def _done():
                    return None

                return _done().__await__()

        class _FakeLocalParticipant:
            async def publish_data(self, *args, **kwargs):
                pass

        class _FakeRoom:
            name = "fake-room"
            local_participant = _FakeLocalParticipant()

            def on(self, *args, **kwargs):
                return lambda fn: fn

        class _FakeJobContext:
            def __init__(self):
                self.room = _FakeRoom()
                self.job = SimpleNamespace()

            async def connect(self):
                return None

            def add_shutdown_callback(self, cb):
                pass

            def shutdown(self, reason=""):
                pass

        class _FakeEventBroker:
            def __init__(self):
                self.callbacks = {}

            def set_logger(self, fb_logger):
                pass

            def register_callback(self, channel, handler):
                self.callbacks[channel] = handler

            async def publish(self, channel, message):
                return 1

        fake_session_holder = {}

        class _FakeSession:
            def __init__(self, *args, **kwargs):
                self._chat_ctx = llm.ChatContext()
                self.current_agent = None
                self._events = {}
                self.generate_reply_calls = 0
                self.say_calls = []
                self.agent_state = "listening"
                self.current_speech = None
                fake_session_holder["session"] = self

            @property
            def history(self):
                return self._chat_ctx

            def on(self, event_name):
                def _decorator(fn):
                    self._events[event_name] = fn
                    return fn

                return _decorator

            async def start(self, room, agent, room_input_options=None):
                self.current_agent = agent

            def generate_reply(self, **kwargs):
                self.generate_reply_calls += 1
                return _ImmediateAwaitable()

            def say(self, text, **kwargs):
                self.say_calls.append(text)
                return _ImmediateAwaitable()

            def interrupt(self):
                pass

        class _FakeAssistant:
            def __init__(self, *args, **kwargs):
                self._chat_ctx = llm.ChatContext()
                self.call_received = True
                self.user_turn_generating = False
                self._user_turn_seq = 0
                self._slow_brain_responded_turn = -1
                self._buffers_since_slow_reply = 0
                self._tts_seq = 0
                self._active_tts = None
                self._pending_continuation = None
                self._publish_voice_interrupt = None

            def set_call_received(self):
                self.call_received = True

            def set_credit_gate_state_provider(self, provider):
                self.credit_gate_state_provider = provider

        async def _noop_async(*args, **kwargs):
            return None

        async def _noop_end_call():
            return None

        fake_broker = _FakeEventBroker()
        fake_session_details = SimpleNamespace(
            populate_from_env=lambda: None,
            voice=SimpleNamespace(provider="cartesia", id=""),
            voice_call=SimpleNamespace(
                outbound=False,
                channel="unify_meet",
                contact_json=json.dumps(contact),
                boss_json=json.dumps(boss),
            ),
            assistant=SimpleNamespace(
                about="Assistant bio",
                name="Ava",
                user_desktop_for=lambda user_id: None,
            ),
            user=SimpleNamespace(id="default"),
            is_coordinator=False,
            org_id=None,
            unify_key="",
        )

        monkeypatch.setattr(call_script, "event_broker", fake_broker)
        monkeypatch.setattr(call_script, "SESSION_DETAILS", fake_session_details)
        monkeypatch.setattr(call_script, "AgentSession", _FakeSession)
        monkeypatch.setattr(call_script, "Assistant", _FakeAssistant)
        monkeypatch.setattr(call_script, "UnifyLLM", lambda *args, **kwargs: object())
        monkeypatch.setattr(
            call_script,
            "build_voice_agent_prompt",
            lambda **kwargs: SimpleNamespace(flatten=lambda: "system prompt"),
        )
        monkeypatch.setattr(call_script, "start_event_broker_receive", _noop_async)
        monkeypatch.setattr(call_script, "publish_call_started", _noop_async)
        monkeypatch.setattr(
            call_script,
            "create_end_call",
            lambda *args, **kwargs: _noop_end_call,
        )
        monkeypatch.setattr(
            call_script,
            "setup_participant_disconnect_handler",
            lambda *args, **kwargs: None,
        )
        monkeypatch.setattr(call_script, "RoomInputOptions", lambda **kwargs: object())
        monkeypatch.setattr(call_script, "EnglishModel", lambda: object())
        monkeypatch.setattr(call_script.cartesia, "TTS", lambda **kwargs: object())
        monkeypatch.setattr(call_script.elevenlabs, "TTS", lambda **kwargs: object())
        if hasattr(call_script, "noise_cancellation"):
            monkeypatch.setattr(call_script.noise_cancellation, "BVC", lambda: object())

        monkeypatch.setattr(call_script, "STT", object())
        monkeypatch.setattr(call_script, "VAD", object())

        import unify.common.llm_client as _llm_mod

        class _FakeGreetingClient:
            async def generate(self, **kwargs):
                return "Hello!"

        monkeypatch.setattr(
            _llm_mod,
            "new_llm_client",
            lambda *a, **kw: _FakeGreetingClient(),
        )

        await call_script.entrypoint(_FakeJobContext())

        session = fake_session_holder["session"]
        session.say_calls.clear()
        guidance_cb = fake_broker.callbacks["app:call:notification"]
        agent_state_cb = session._events["agent_state_changed"]

        # Simulate agent in "thinking" state (processing a user turn)
        session.agent_state = "thinking"

        # Guidance arrives while agent is thinking
        guidance_cb(
            {
                "payload": {
                    "message": "It's at 3pm.",
                    "should_speak": True,
                },
            },
        )
        assert (
            len(session.say_calls) == 0
        ), "Queued speech must not fire while agent is thinking."

        # Agent starts speaking (its reply to the user)
        agent_state_cb(SimpleNamespace(new_state="speaking"))
        session.agent_state = "speaking"
        assert (
            len(session.say_calls) == 0
        ), "Queued speech must not fire while agent is speaking."

        # Agent finishes speaking → transitions to listening
        session.agent_state = "listening"
        agent_state_cb(SimpleNamespace(new_state="listening"))

        # maybe_speak_queued speaks synchronously; yield just in case.
        await asyncio.sleep(0)

        assert (
            len(session.say_calls) == 1
        ), "Queued speech should fire after agent returns to listening."
        assert session.say_calls[0] == "It's at 3pm."


@pytest.mark.asyncio
async def test_fastbrain_notification_records_only_silent_guidance():
    """should_speak guidance is no longer pre-written to contact_index (it is
    recorded once via the actually-spoken Outbound utterance); silent
    (should_speak=False) guidance is still recorded exactly once."""
    from unittest.mock import AsyncMock, MagicMock
    from unify.conversation_manager.cm_types import Mode
    from unify.conversation_manager.domains.event_handlers import EventHandler
    from unify.conversation_manager.events import FastBrainNotification

    handler = EventHandler._registry[FastBrainNotification]
    contact = {"contact_id": 1, "first_name": "Dan", "surname": "Lenton"}

    def _make_cm():
        return SimpleNamespace(
            contact_index=SimpleNamespace(
                get_contact=lambda contact_id=None: None,
                push_message=MagicMock(return_value=1),
            ),
            call_manager=SimpleNamespace(
                has_active_google_meet=False,
                has_active_teams_meet=False,
                _call_channel="phone_call",
            ),
            mode=Mode.CALL,
            schedule_proactive_speech=AsyncMock(),
        )

    # should_speak=True -> no contact_index pre-write; still re-arms proactive.
    cm = _make_cm()
    await handler(
        FastBrainNotification(
            contact=contact,
            message="Just sent the clue to your email.",
            should_speak=True,
            source="slow_brain",
        ),
        cm,
    )
    cm.contact_index.push_message.assert_not_called()
    cm.schedule_proactive_speech.assert_awaited()

    # should_speak=False -> recorded exactly once as guidance.
    cm = _make_cm()
    await handler(
        FastBrainNotification(
            contact=contact,
            message="Email received from Alice.",
            should_speak=False,
            source="slow_brain",
        ),
        cm,
    )
    assert cm.contact_index.push_message.call_count == 1
    _, kwargs = cm.contact_index.push_message.call_args
    assert kwargs.get("role") == "guidance"
    assert kwargs.get("message_content") == "Email received from Alice."


class TestSayMetaTextMatching:
    """Regression tests for _last_say_meta text matching.

    When session.say() and generate_reply produce concurrent TTS, the
    _last_say_meta (set by session.say) can be consumed by the wrong
    conversation_item_added event. The fix is to store the spoken text
    in the meta and only consume it when the utterance text matches.
    """

    def test_match_say_meta_exists(self):
        """The match_say_meta helper must exist in common.py."""
        from unify.conversation_manager.medium_scripts.common import match_say_meta

        assert callable(match_say_meta)

    def test_matching_text_consumes_meta(self):
        """When utterance text matches the session.say text, return the meta."""
        from unify.conversation_manager.medium_scripts.common import match_say_meta

        meta = {
            "guidance_id": "guid-abc",
            "source": "proactive_speech",
            "text": "Still there, Alex?",
        }
        result = match_say_meta(meta, "Still there, Alex?")
        assert result is not None
        assert result["guidance_id"] == "guid-abc"

    def test_different_text_does_not_consume_meta(self):
        """When utterance text differs from session.say text, return None.

        This is the exact bug: a fast brain response like
        'One moment - I am pulling that up.' should NOT consume meta
        set by session.say('Sure, go ahead - what is the task?').
        """
        from unify.conversation_manager.medium_scripts.common import match_say_meta

        meta = {
            "guidance_id": "guid-abc",
            "source": "proactive_speech",
            "text": "Sure, go ahead — what's the task?",
        }
        result = match_say_meta(meta, "One moment — I'm pulling that up.")
        assert result is None

    def test_none_meta_returns_none(self):
        """When no meta is set, return None regardless of text."""
        from unify.conversation_manager.medium_scripts.common import match_say_meta

        assert match_say_meta(None, "anything") is None

    def test_meta_without_text_key_always_matches(self):
        """Legacy meta dicts without a text key match any utterance
        (backward compatible with pre-fix code)."""
        from unify.conversation_manager.medium_scripts.common import match_say_meta

        meta = {"guidance_id": "guid-abc", "source": "slow_brain"}
        result = match_say_meta(meta, "Some text")
        assert result is not None


# =============================================================================
# Participant comms rendering
# =============================================================================


class TestParticipantCommsRendering:
    """Unit tests for render_participant_comms — verifies tag format and
    participant-match filtering for comms events on any call."""

    def _make_event_json(self, event):
        return event.to_json()

    # ── Outbound (assistant → participant) ──────────────────────────────

    def test_outbound_unify_message_to_participant_rendered(self):
        """When the slow brain sends a Unify message to the person on the
        call, the fast brain must see it so it can verbally acknowledge it.

        Without this, the caller receives a silent text message during a
        live voice conversation with no verbal indication from the
        assistant that anything was sent.
        """
        from unify.conversation_manager.events import UnifyMessageSent
        from unify.conversation_manager.medium_scripts.common import (
            render_participant_comms,
        )

        event = UnifyMessageSent(
            contact={"contact_id": 1, "first_name": "Alex", "surname": "Demo"},
            content="Here's the detailed breakdown of your OneDrive contents.",
        )
        result = render_participant_comms(event.to_json(), {1})
        assert result is not None, (
            "render_participant_comms must render outbound UnifyMessageSent "
            "to a call participant so the fast brain can acknowledge it verbally"
        )
        assert "Alex Demo" in result
        assert "OneDrive" in result

    def test_outbound_sms_to_participant_rendered(self):
        """Outbound SMS to a call participant should be visible to the fast
        brain for verbal acknowledgement."""
        from unify.conversation_manager.events import SMSSent
        from unify.conversation_manager.medium_scripts.common import (
            render_participant_comms,
        )

        event = SMSSent(
            contact={"contact_id": 5, "first_name": "Marcus", "surname": "Rivera"},
            content="Sent you the meeting link.",
        )
        result = render_participant_comms(event.to_json(), {5})
        assert result is not None, (
            "render_participant_comms must render outbound SMSSent "
            "to a call participant so the fast brain can acknowledge it verbally"
        )
        assert "Marcus Rivera" in result
        assert "meeting link" in result

    def test_outbound_message_to_non_participant_returns_none(self):
        """Outbound messages to contacts NOT on the call should still be
        invisible to the fast brain — only participant-targeted messages
        matter."""
        from unify.conversation_manager.events import UnifyMessageSent
        from unify.conversation_manager.medium_scripts.common import (
            render_participant_comms,
        )

        event = UnifyMessageSent(
            contact={"contact_id": 99, "first_name": "Alice", "surname": "Other"},
            content="Some message to a third party.",
        )
        result = render_participant_comms(event.to_json(), {1, 5})
        assert result is None

    # ── Inbound (participant → assistant) ───────────────────────────────

    def test_sms_from_participant_rendered_with_tag(self):
        from unify.conversation_manager.medium_scripts.common import (
            render_participant_comms,
        )
        from unify.conversation_manager.events import SMSReceived

        event = SMSReceived(
            contact={"contact_id": 5, "first_name": "Marcus", "surname": "Rivera"},
            content="Running late, be there in 10.",
        )
        result = render_participant_comms(event.to_json(), {5})
        assert result is not None
        assert result.startswith("[SMS from Marcus Rivera]")
        assert "Running late" in result

    def test_email_from_participant_rendered_with_tag(self):
        from unify.conversation_manager.medium_scripts.common import (
            render_participant_comms,
        )
        from unify.conversation_manager.events import EmailReceived

        event = EmailReceived(
            contact={"contact_id": 3, "first_name": "Sarah", "surname": "Chen"},
            subject="Updated agenda",
            body="See attached for the revised agenda.",
        )
        result = render_participant_comms(event.to_json(), {3})
        assert result is not None
        assert result.startswith("[Email from Sarah Chen]")
        assert "Updated agenda" in result

    def test_unify_message_from_participant_rendered_with_tag(self):
        from unify.conversation_manager.medium_scripts.common import (
            render_participant_comms,
        )
        from unify.conversation_manager.events import UnifyMessageReceived

        event = UnifyMessageReceived(
            contact={"contact_id": 7, "first_name": "Priya", "surname": "Sharma"},
            content="Check the shared doc.",
        )
        result = render_participant_comms(event.to_json(), {7})
        assert result is not None
        assert result.startswith("[Message from Priya Sharma]")
        assert "shared doc" in result

    def test_sms_from_non_participant_returns_none(self):
        from unify.conversation_manager.medium_scripts.common import (
            render_participant_comms,
        )
        from unify.conversation_manager.events import SMSReceived

        event = SMSReceived(
            contact={"contact_id": 99, "first_name": "Stranger", "surname": "Person"},
            content="Hello?",
        )
        result = render_participant_comms(event.to_json(), {5, 3})
        assert result is None

    def test_non_comms_event_returns_none(self):
        from unify.conversation_manager.medium_scripts.common import (
            render_participant_comms,
        )
        from unify.conversation_manager.events import ActorNotification

        event = ActorNotification(handle_id=1, response="Searching...")
        result = render_participant_comms(event.to_json(), {1, 5})
        assert result is None

    def test_multiple_participants_matched(self):
        from unify.conversation_manager.medium_scripts.common import (
            render_participant_comms,
        )
        from unify.conversation_manager.events import SMSReceived

        event = SMSReceived(
            contact={"contact_id": 3, "first_name": "Sarah", "surname": "Chen"},
            content="On my way.",
        )
        result = render_participant_comms(event.to_json(), {1, 3, 5})
        assert result is not None
        assert "[SMS from Sarah Chen]" in result


# =============================================================================
# Child Process Logging (forkserver compatibility)
# =============================================================================


class TestChildProcessLogging:
    """Verify _configure_child_logging fixes log propagation for LiveKit's
    forkserver child processes.

    LiveKit agents (v1.2.x) routes child-process logs through a
    LogQueueHandler on the **root** logger.  Unity's LOGGER defaults to
    propagate=False, which silently drops every record in the child.
    _configure_child_logging must flip propagation on and strip stale
    direct handlers so records reach the root relay.
    """

    def test_propagation_enabled_and_handlers_cleared(self):
        import logging

        from unify.logger import LOGGER

        from unify.conversation_manager.medium_scripts.call import (
            _configure_child_logging,
        )

        original_propagate = LOGGER.propagate
        original_handlers = list(LOGGER.handlers)
        try:
            assert (
                LOGGER.propagate is False
            ), "precondition: LOGGER.propagate should be False before the fix runs"

            _configure_child_logging()

            assert LOGGER.propagate is True
            assert LOGGER.handlers == []

            for name in ("livekit", "livekit.agents", "livekit.plugins"):
                lg = logging.getLogger(name)
                assert lg.propagate is True
                assert lg.handlers == []
        finally:
            LOGGER.propagate = original_propagate
            LOGGER.handlers = original_handlers

    def test_records_reach_root_logger_after_configure(self):
        import logging

        from unify.logger import LOGGER

        from unify.conversation_manager.medium_scripts.call import (
            _configure_child_logging,
        )

        original_propagate = LOGGER.propagate
        original_handlers = list(LOGGER.handlers)
        captured: list[logging.LogRecord] = []

        class _CaptureHandler(logging.Handler):
            def emit(self, record):
                captured.append(record)

        root = logging.getLogger()
        capture = _CaptureHandler()
        root.addHandler(capture)
        root_level = root.level
        root.setLevel(logging.NOTSET)
        try:
            _configure_child_logging()

            LOGGER.info("test-sentinel-message")

            assert any(
                r.message == "test-sentinel-message" for r in captured
            ), "LOGGER records must propagate to root after _configure_child_logging"
        finally:
            root.removeHandler(capture)
            root.setLevel(root_level)
            LOGGER.propagate = original_propagate
            LOGGER.handlers = original_handlers


# =============================================================================
# Fast brain resuming an interrupted slow-brain line
# =============================================================================


class TestFastBrainContinuation:
    """``Assistant._claim_interrupted_continuation`` exactly-once semantics."""

    def _assistant(self, boss_contact):
        from unify.conversation_manager.medium_scripts.call import Assistant

        return Assistant(
            contact=boss_contact,
            boss=boss_contact,
            channel="phone_call",
            instructions="x",
            outbound=False,
        )

    def _pending(self):
        return {
            "resume_text": "Next, click Connect Slack.",
            "remainder": "Next, click Connect Slack.",
            "spoken_prefix": "Your number is saved.",
            "seq": 1,
            "consumed": False,
        }

    @pytest.mark.asyncio
    async def test_claims_and_resumes_with_verbatim_remainder(
        self,
        boss_contact,
        monkeypatch,
    ):
        from unify.conversation_manager.medium_scripts import call as call_mod

        a = self._assistant(boss_contact)
        a._pending_continuation = self._pending()

        async def _fake(resume_text, user_text):
            return "Sorry — as I was saying,"

        monkeypatch.setattr(call_mod, "select_continuation", _fake)

        out = await a._claim_interrupted_continuation("okay")
        # Lead-in from the model + the slow brain's verbatim remainder.
        assert out == "Sorry — as I was saying, Next, click Connect Slack."
        # Claimed: cleared so the VoiceInterrupt handoff becomes a no-op.
        assert a._pending_continuation is None

    @pytest.mark.asyncio
    async def test_defer_hands_off_to_slow_brain(self, boss_contact, monkeypatch):
        from unify.conversation_manager.medium_scripts import call as call_mod

        a = self._assistant(boss_contact)
        a._pending_continuation = self._pending()
        published: dict = {}

        async def _pub(spoken, remainder):
            published["args"] = (spoken, remainder)

        a._publish_voice_interrupt = _pub

        async def _defer(resume_text, user_text):
            return None

        monkeypatch.setattr(call_mod, "select_continuation", _defer)

        out = await a._claim_interrupted_continuation("wait, stop")
        assert out is None
        # The remainder is handed to the slow brain instead (delivered once).
        assert published["args"] == (
            "Your number is saved.",
            "Next, click Connect Slack.",
        )
        assert a._pending_continuation is None

    @pytest.mark.asyncio
    async def test_no_pending_returns_none(self, boss_contact):
        a = self._assistant(boss_contact)
        assert a._pending_continuation is None
        assert await a._claim_interrupted_continuation("okay") is None

    @pytest.mark.asyncio
    async def test_already_consumed_returns_none(self, boss_contact, monkeypatch):
        from unify.conversation_manager.medium_scripts import call as call_mod

        a = self._assistant(boss_contact)
        pending = self._pending()
        pending["consumed"] = True
        a._pending_continuation = pending

        def _boom(*args, **kwargs):
            raise AssertionError(
                "must not select a continuation for a consumed candidate",
            )

        monkeypatch.setattr(call_mod, "select_continuation", _boom)
        assert await a._claim_interrupted_continuation("okay") is None

    @pytest.mark.asyncio
    async def test_speechless_barge_in_auto_continues(self, boss_contact, monkeypatch):
        """A barge-in with no transcript (noise/echo) resumes the remainder
        verbatim WITHOUT consulting the classifier - the only sensible action."""
        from unify.conversation_manager.medium_scripts import call as call_mod

        a = self._assistant(boss_contact)
        a._pending_continuation = self._pending()

        def _boom(*args, **kwargs):
            raise AssertionError("speechless barge-in must not consult the classifier")

        monkeypatch.setattr(call_mod, "select_continuation", _boom)

        out = await a._claim_interrupted_continuation("   ")
        assert out is not None
        assert out.endswith("Next, click Connect Slack.")
        # Claimed exactly once.
        assert a._pending_continuation is None

    @pytest.mark.asyncio
    async def test_llm_node_continue_marks_text_and_cancels_slow_brain(
        self,
        boss_contact,
        monkeypatch,
    ):
        """On CONTINUE, llm_node yields the verbatim remainder as the reply, marks
        it for interruption-stashing, and cancels the eager slow-brain run."""
        from unittest.mock import AsyncMock

        from livekit.agents import llm

        a = self._assistant(boss_contact)
        a.call_received = True
        resumed = "Sorry — as I was saying, Next, click Connect Slack."
        a._claim_interrupted_continuation = AsyncMock(return_value=resumed)
        a._publish_fast_brain_continued = AsyncMock()

        chunks = [chunk async for chunk in a.llm_node(llm.ChatContext(), [], None)]

        assert len(chunks) == 1
        assert chunks[0].delta.content == resumed
        # Marked so the speech_created observer registers the reply (recursion).
        assert a._continuation_full_text == resumed
        # Cancels exactly this turn's slow-brain run (by turn id) so it cannot
        # re-deliver.
        a._publish_fast_brain_continued.assert_awaited_once_with(a._user_turn_seq)


class TestFastBrainSmalltalk:
    """``llm_node`` small-talk branch: the fast brain fully answers a pure social
    turn and cancels the slow brain; otherwise it defers with a filler."""

    def _assistant(self, boss_contact):
        from unify.conversation_manager.medium_scripts.call import Assistant

        return Assistant(
            contact=boss_contact,
            boss=boss_contact,
            channel="phone_call",
            instructions="x",
            outbound=False,
        )

    @pytest.mark.asyncio
    async def test_smalltalk_answers_and_cancels_slow_brain(
        self,
        boss_contact,
        monkeypatch,
    ):
        from unittest.mock import AsyncMock

        from livekit.agents import llm

        from unify.conversation_manager.medium_scripts import call as call_mod

        a = self._assistant(boss_contact)
        a.call_received = True
        a._capture_screenshots_for_llm = AsyncMock()
        reply = "Doing great, thanks for asking! How are you?"
        a._generate_smalltalk_reply = AsyncMock(return_value=reply)
        a._publish_fast_brain_continued = AsyncMock()

        async def _filler(*args, **kwargs):
            return "One moment."

        monkeypatch.setattr(call_mod, "select_fast_reply", _filler)

        chunks = [chunk async for chunk in a.llm_node(llm.ChatContext(), [], None)]

        # The smalltalk reply is spoken (no filler), and this turn's slow-brain
        # run is cancelled by turn id.
        assert len(chunks) == 1
        assert chunks[0].delta.content == reply
        a._publish_fast_brain_continued.assert_awaited_once_with(a._user_turn_seq)

    @pytest.mark.asyncio
    async def test_defer_falls_back_to_filler_without_cancelling(
        self,
        boss_contact,
        monkeypatch,
    ):
        from unittest.mock import AsyncMock

        from livekit.agents import llm

        from unify.conversation_manager.medium_scripts import call as call_mod

        a = self._assistant(boss_contact)
        a.call_received = True
        a._capture_screenshots_for_llm = AsyncMock()
        # Not small talk -> defer; the filler covers the gap, slow brain proceeds.
        a._generate_smalltalk_reply = AsyncMock(return_value=None)
        a._publish_fast_brain_continued = AsyncMock()

        async def _filler(*args, **kwargs):
            return "One sec — pulling that up."

        monkeypatch.setattr(call_mod, "select_fast_reply", _filler)

        chunks = [chunk async for chunk in a.llm_node(llm.ChatContext(), [], None)]

        assert len(chunks) == 1
        assert chunks[0].delta.content == "One sec — pulling that up."
        # Slow brain must NOT be cancelled - it produces the real answer.
        a._publish_fast_brain_continued.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_idle_status_gate_is_passed_to_smalltalk(
        self,
        boss_contact,
        monkeypatch,
    ):
        from unittest.mock import AsyncMock

        from livekit.agents import llm

        from unify.conversation_manager.medium_scripts import call as call_mod

        a = self._assistant(boss_contact)
        a.call_received = True
        a._capture_screenshots_for_llm = AsyncMock()
        a._request_idle_smalltalk_state = AsyncMock(return_value=True)
        a._generate_smalltalk_reply = AsyncMock(return_value=None)
        a._publish_fast_brain_continued = AsyncMock()

        async def _filler(*args, **kwargs):
            return "One moment."

        monkeypatch.setattr(call_mod, "select_fast_reply", _filler)

        chunks = [chunk async for chunk in a.llm_node(llm.ChatContext(), [], None)]

        assert len(chunks) == 1
        assert chunks[0].delta.content == "One moment."
        a._generate_smalltalk_reply.assert_awaited_once_with(
            "",
            idle_status_smalltalk=True,
        )

    @pytest.mark.asyncio
    async def test_bare_acknowledgement_stays_silent(
        self,
        boss_contact,
        monkeypatch,
    ):
        """A bare ack ('okay') -> say nothing AND cancel the slow brain, so
        neither brain speaks (no parroting, no filler)."""
        from unittest.mock import AsyncMock

        from livekit.agents import llm

        from unify.conversation_manager.medium_scripts import call as call_mod

        a = self._assistant(boss_contact)
        a.call_received = True
        a._capture_screenshots_for_llm = AsyncMock()
        a._generate_smalltalk_reply = AsyncMock(
            return_value=call_mod._SMALLTALK_STAY_SILENT,
        )
        a._publish_fast_brain_continued = AsyncMock()

        async def _filler(*args, **kwargs):
            return "One moment."

        monkeypatch.setattr(call_mod, "select_fast_reply", _filler)

        chunks = [chunk async for chunk in a.llm_node(llm.ChatContext(), [], None)]

        # Nothing spoken (no parrot, no filler), but the slow brain is cancelled.
        assert chunks == []
        a._publish_fast_brain_continued.assert_awaited_once_with(a._user_turn_seq)


# =============================================================================
# Outbound opener: held until the callee's first utterance (or fallback)
# =============================================================================


class TestOutboundOpenerTrigger:
    """The outbound opener is held until the callee speaks; the first turn is
    consumed by the opener (no competing fast-brain filler)."""

    def _assistant(self, boss_contact, *, outbound: bool):
        from unify.conversation_manager.medium_scripts.call import Assistant

        return Assistant(
            contact=boss_contact,
            boss=boss_contact,
            channel="phone_call",
            instructions="x",
            outbound=outbound,
        )

    def test_opening_pending_only_for_outbound(self, boss_contact):
        assert self._assistant(boss_contact, outbound=True)._opening_pending is True
        assert self._assistant(boss_contact, outbound=False)._opening_pending is False

    @pytest.mark.asyncio
    async def test_first_user_turn_signalled_while_opening_pending(self, boss_contact):
        a = self._assistant(boss_contact, outbound=True)
        assert not a._first_user_turn.is_set()

        await a.on_user_turn_completed(None, SimpleNamespace(text_content="Hello?"))
        assert a._first_user_turn.is_set()

    @pytest.mark.asyncio
    async def test_first_user_turn_not_signalled_when_not_pending(self, boss_contact):
        a = self._assistant(boss_contact, outbound=False)  # _opening_pending False

        await a.on_user_turn_completed(None, SimpleNamespace(text_content="Hi"))
        assert not a._first_user_turn.is_set()

    @pytest.mark.asyncio
    async def test_llm_node_suppresses_filler_while_opening_pending(self, boss_contact):
        from livekit.agents import llm

        a = self._assistant(boss_contact, outbound=True)
        a.call_received = True  # past the connection wait
        assert a._opening_pending is True

        chunks = [chunk async for chunk in a.llm_node(llm.ChatContext(), [], None)]
        # The opener is the reply to the triggering turn; no filler is emitted.
        assert chunks == []
