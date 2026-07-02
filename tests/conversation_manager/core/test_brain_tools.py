"""
tests/conversation_manager/test_brain_tools.py
====================================================

Unit tests for ConversationManager brain tools.

Tests cover:
- ConversationManagerBrainTools (read-only inspection tools)
- ConversationManagerBrainActionTools (side-effecting action tools)

These tests verify the tool implementations directly, testing:
- Tool method signatures and return types
- Tool docstrings (important for LLM understanding)
- Dynamic tool generation for action steering
- Integration with ConversationManager state
"""

from __future__ import annotations

import asyncio
from datetime import datetime
import inspect
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from unify.common.llm_helpers import method_to_schema
from unify.contact_manager.simulated import SimulatedContactManager
from unify.conversation_manager.domains.brain_tools import (
    ConversationManagerBrainTools,
)
from unify.conversation_manager.domains.brain_action_tools import (
    ConversationManagerBrainActionTools,
)
from unify.file_manager.filesystem_adapters.local_adapter import (
    LocalFileSystemAdapter,
)
from unify.conversation_manager.domains.notifications import (
    NotificationBar,
)
from unify.conversation_manager.domains.contact_index import (
    ContactIndex,
)
from unify.conversation_manager.task_actions import (
    STEERING_OPERATIONS,
    parse_action_name,
)
from unify.session_details import SESSION_DETAILS

# =============================================================================
# Fixtures
# =============================================================================


def _setup_mock_contacts(
    contact_index,
    contacts: list[dict],
) -> SimulatedContactManager:
    """
    Set up a SimulatedContactManager with the given contacts on a ContactIndex.

    Returns the SimulatedContactManager for additional inspection if needed.
    """
    contact_manager = SimulatedContactManager()

    # Populate contacts - update system contacts (0, 1) and create others
    for contact_data in contacts:
        contact_id = contact_data["contact_id"]
        contact_manager.update_contact(
            contact_id=contact_id,
            first_name=contact_data.get("first_name"),
            surname=contact_data.get("surname"),
            email_address=contact_data.get("email_address"),
            phone_number=contact_data.get("phone_number"),
            should_respond=contact_data.get("should_respond", True),
        )

    contact_index.set_contact_manager(contact_manager)
    return contact_manager


@pytest.fixture
def mock_cm():
    """Create a minimal mock ConversationManager for testing."""
    from unify.conversation_manager.cm_types.mode import Mode

    cm = MagicMock()
    cm.mode = Mode.TEXT
    cm.contact_index = ContactIndex()
    cm.in_flight_actions = {}
    cm.completed_actions = {}
    cm.notifications_bar = NotificationBar()
    cm.chat_history = []
    cm.assistant_number = "+15555550000"
    cm.assistant_email = "assistant@test.com"
    cm.assistant_email_provider = "google_workspace"
    cm.assistant_whatsapp_number = ""
    cm.assistant_discord_bot_id = ""
    cm.assistant_slack_bot_user_id = ""
    cm.initialized = True
    cm.call_manager.has_active_call = False
    cm.call_manager.has_active_google_meet = False
    cm.call_manager.has_active_teams_meet = False
    cm.call_manager.has_gmeet_presenting = False
    cm.call_manager.has_teams_presenting = False
    cm.call_manager.is_ready_for_outbound_call = True
    cm.call_manager.await_ready_for_outbound_call = AsyncMock(return_value=True)
    cm.call_manager._meet_joining = False
    cm.call_manager._whatsapp_call_joining = False
    cm.call_manager._call_channel = None
    cm.in_voice_session = False
    cm.assistant_has_teams = False
    # Set up SimulatedContactManager (starts with system contacts 0 and 1)
    cm.contact_manager = _setup_mock_contacts(cm.contact_index, [])
    return cm


@pytest.fixture
def brain_tools(mock_cm):
    """Create ConversationManagerBrainTools instance."""
    return ConversationManagerBrainTools(mock_cm)


@pytest.fixture
def brain_action_tools(mock_cm):
    """Create ConversationManagerBrainActionTools instance."""
    # Patch the event broker to avoid actual pubsub
    with patch(
        "unify.conversation_manager.domains.brain_action_tools.get_event_broker",
    ) as mock_broker:
        mock_broker.return_value = MagicMock()
        mock_broker.return_value.publish = AsyncMock()
        tools = ConversationManagerBrainActionTools(mock_cm)
        yield tools


@pytest.fixture
def coordinator_session():
    previous_is_coordinator = SESSION_DETAILS.is_coordinator
    previous_boss_contact_id = SESSION_DETAILS.boss_contact_id
    SESSION_DETAILS.is_coordinator = True
    SESSION_DETAILS.boss_contact_id = 1
    yield
    SESSION_DETAILS.is_coordinator = previous_is_coordinator
    SESSION_DETAILS.boss_contact_id = previous_boss_contact_id


@pytest.fixture
def sample_contacts():
    """Sample contacts for testing."""
    return [
        {
            "contact_id": 1,
            "first_name": "Alice",
            "surname": "Smith",
            "phone_number": "+15551111111",
            "email_address": "alice@example.com",
            "should_respond": True,
        },
        {
            "contact_id": 2,
            "first_name": "Bob",
            "surname": "Johnson",
            "phone_number": "+15552222222",
            "email_address": "bob@example.com",
            "should_respond": True,
        },
    ]


# =============================================================================
# ConversationManagerBrainTools Tests
# =============================================================================


class TestCmGetMode:
    """Tests for cm_get_mode tool."""

    def test_returns_text_mode(self, brain_tools, mock_cm):
        """Returns 'text' when CM is in text mode."""
        from unify.conversation_manager.cm_types.mode import Mode

        mock_cm.mode = Mode.TEXT
        assert brain_tools.cm_get_mode() == "text"

    def test_returns_call_mode(self, brain_tools, mock_cm):
        """Returns 'call' when CM is in call mode."""
        from unify.conversation_manager.cm_types.mode import Mode

        mock_cm.mode = Mode.CALL
        assert brain_tools.cm_get_mode() == "call"

    def test_returns_meet_mode(self, brain_tools, mock_cm):
        """Returns 'meet' when CM is in meet mode."""
        from unify.conversation_manager.cm_types.mode import Mode

        mock_cm.mode = Mode.MEET
        assert brain_tools.cm_get_mode() == "meet"

    def test_converts_mode_to_string(self, brain_tools, mock_cm):
        """Converts mode to string regardless of type."""
        # Mode could be an enum or other type
        mock_cm.mode = MagicMock(__str__=lambda self: "custom_mode")
        result = brain_tools.cm_get_mode()
        assert isinstance(result, str)


class TestCmGetContact:
    """Tests for cm_get_contact tool."""

    def test_returns_contact_by_id(self, brain_tools, mock_cm, sample_contacts):
        """Returns contact when found by ID."""
        _setup_mock_contacts(mock_cm.contact_index, sample_contacts)
        result = brain_tools.cm_get_contact(1)
        assert result is not None
        assert result["contact_id"] == 1
        assert result["first_name"] == "Alice"

    def test_returns_none_for_unknown_id(self, brain_tools, mock_cm, sample_contacts):
        """Returns None when contact not found."""
        _setup_mock_contacts(mock_cm.contact_index, sample_contacts)
        result = brain_tools.cm_get_contact(999)
        assert result is None

    def test_excludes_threads_from_contact(self, brain_tools, mock_cm, sample_contacts):
        """Contact summary excludes thread data for efficiency."""
        _setup_mock_contacts(mock_cm.contact_index, sample_contacts)
        result = brain_tools.cm_get_contact(1)
        # get_contact uses model_dump(exclude={"threads", "global_thread"})
        assert "threads" not in result
        assert "global_thread" not in result


class TestCmListInFlightActions:
    """Tests for cm_list_in_flight_actions tool."""

    def test_returns_empty_list_when_no_actions(self, brain_tools, mock_cm):
        """Returns empty list when no in-flight actions."""
        mock_cm.in_flight_actions = {}
        result = brain_tools.cm_list_in_flight_actions()
        assert result == []

    def test_returns_action_summary(self, brain_tools, mock_cm):
        """Returns summary for each in-flight action."""
        mock_cm.in_flight_actions = {
            0: {"query": "Search for contacts", "handle_actions": []},
            1: {"query": "Send an email", "handle_actions": [{"action": "test"}]},
        }
        result = brain_tools.cm_list_in_flight_actions()
        assert len(result) == 2
        assert result[0]["handle_id"] == 0
        assert result[0]["query"] == "Search for contacts"
        assert result[0]["num_handle_actions"] == 0
        assert result[1]["handle_id"] == 1
        assert result[1]["query"] == "Send an email"
        assert result[1]["num_handle_actions"] == 1

    def test_handles_none_in_flight_actions(self, brain_tools, mock_cm):
        """Handles None in-flight actions gracefully."""
        mock_cm.in_flight_actions = None
        result = brain_tools.cm_list_in_flight_actions()
        assert result == []

    def test_handles_none_handle_actions(self, brain_tools, mock_cm):
        """Handles None handle_actions in action data."""
        mock_cm.in_flight_actions = {
            0: {"query": "Action", "handle_actions": None},
        }
        result = brain_tools.cm_list_in_flight_actions()
        assert result[0]["num_handle_actions"] == 0


class TestCmListNotifications:
    """Tests for cm_list_notifications tool."""

    def test_returns_empty_list_when_no_notifications(self, brain_tools, mock_cm):
        """Returns empty list when no notifications."""
        result = brain_tools.cm_list_notifications()
        assert result == []

    def test_returns_all_notifications(self, brain_tools, mock_cm, static_now):
        """Returns all notifications when pinned_only=False."""
        ts = static_now
        mock_cm.notifications_bar.push_notif("Type1", "Content1", ts)
        mock_cm.notifications_bar.push_notif("Type2", "Content2", ts, pinned=True)
        result = brain_tools.cm_list_notifications()
        assert len(result) == 2

    def test_filters_pinned_only(self, brain_tools, mock_cm, static_now):
        """Returns only pinned notifications when pinned_only=True."""
        ts = static_now
        mock_cm.notifications_bar.push_notif("Regular", "Not pinned", ts)
        mock_cm.notifications_bar.push_notif("Pinned", "Important", ts, pinned=True)
        result = brain_tools.cm_list_notifications(pinned_only=True)
        assert len(result) == 1
        assert result[0]["content"] == "Important"

    def test_converts_timestamp_to_isoformat(self, brain_tools, mock_cm):
        """Converts datetime timestamps to ISO format strings."""
        ts = datetime(2024, 1, 15, 10, 30, 0)
        mock_cm.notifications_bar.push_notif("Test", "Content", ts)
        result = brain_tools.cm_list_notifications()
        assert result[0]["timestamp"] == "2024-01-15T10:30:00"


class TestBrainToolsAsTools:
    """Tests for as_tools method."""

    def test_returns_dict_of_callables(self, brain_tools):
        """Returns dictionary mapping names to callable methods."""
        tools = brain_tools.as_tools()
        assert isinstance(tools, dict)
        assert all(callable(fn) for fn in tools.values())

    def test_contains_all_brain_tools(self, brain_tools):
        """Contains all expected brain tools."""
        tools = brain_tools.as_tools()
        expected = {
            "cm_get_mode",
            "cm_get_contact",
            "cm_list_in_flight_actions",
            "cm_list_notifications",
        }
        assert set(tools.keys()) == expected

    def test_tools_are_bound_methods(self, brain_tools):
        """Tools are bound to the BrainTools instance."""
        tools = brain_tools.as_tools()
        # Calling through the dict should work
        assert tools["cm_get_mode"]() == "text"


# =============================================================================
# ConversationManagerBrainActionTools Tests
# =============================================================================


class TestActionToolsAsTools:
    """Tests for action tools as_tools method."""

    def test_returns_dict_of_callables(self, brain_action_tools):
        """Returns dictionary mapping names to callable methods."""
        tools = brain_action_tools.as_tools()
        assert isinstance(tools, dict)
        assert all(callable(fn) for fn in tools.values())

    def test_contains_all_action_tools_when_fully_configured(
        self,
        brain_action_tools,
    ):
        """All comms tools present when assistant has both phone and email."""
        tools = brain_action_tools.as_tools()
        expected = {
            "join_google_meet",
            "join_teams_meet",
            "start_unify_meet",
            "send_sms",
            "send_unify_message",
            "send_api_response",
            "send_email",
            "make_call",
            "act",
            "ask_about_contacts",
            "update_contacts",
            "query_past_transcripts",
            "wait",
        }
        assert set(tools.keys()) == expected

    def test_excludes_phone_tools_without_number(self, mock_cm):
        """send_sms and make_call are excluded when assistant has no phone."""
        mock_cm.assistant_number = ""
        with patch(
            "unify.conversation_manager.domains.brain_action_tools.get_event_broker",
        ) as mock_broker:
            mock_broker.return_value = MagicMock()
            mock_broker.return_value.publish = AsyncMock()
            tools = ConversationManagerBrainActionTools(mock_cm).as_tools()
        assert "send_sms" not in tools
        assert "make_call" not in tools
        assert "send_email" in tools
        assert "send_unify_message" in tools

    def test_excludes_email_tool_without_email(self, mock_cm):
        """send_email is excluded when assistant has no email."""
        mock_cm.assistant_email = ""
        with patch(
            "unify.conversation_manager.domains.brain_action_tools.get_event_broker",
        ) as mock_broker:
            mock_broker.return_value = MagicMock()
            mock_broker.return_value.publish = AsyncMock()
            tools = ConversationManagerBrainActionTools(mock_cm).as_tools()
        assert "send_email" not in tools
        assert "send_sms" in tools
        assert "make_call" in tools
        assert "send_unify_message" in tools

    def test_excludes_all_comms_without_capabilities(self, mock_cm):
        """Only send_unify_message remains when assistant has no phone or email."""
        mock_cm.assistant_number = ""
        mock_cm.assistant_email = ""
        with patch(
            "unify.conversation_manager.domains.brain_action_tools.get_event_broker",
        ) as mock_broker:
            mock_broker.return_value = MagicMock()
            mock_broker.return_value.publish = AsyncMock()
            tools = ConversationManagerBrainActionTools(mock_cm).as_tools()
        assert "send_sms" not in tools
        assert "make_call" not in tools
        assert "send_email" not in tools
        assert "send_unify_message" in tools
        assert "wait" in tools

    def test_includes_whatsapp_and_discord_tools_when_enabled(self, mock_cm):
        """WhatsApp and Discord tools appear when those assistant capabilities exist."""
        mock_cm.assistant_whatsapp_number = "+15555557777"
        mock_cm.assistant_discord_bot_id = "discord-bot-123"
        with patch(
            "unify.conversation_manager.domains.brain_action_tools.get_event_broker",
        ) as mock_broker:
            mock_broker.return_value = MagicMock()
            mock_broker.return_value.publish = AsyncMock()
            tools = ConversationManagerBrainActionTools(mock_cm).as_tools()

        assert "send_whatsapp" in tools
        assert "make_whatsapp_call" in tools
        assert "send_discord_message" in tools
        assert "send_discord_channel_message" in tools

    def test_coordinator_comms_schemas_are_boss_only(
        self,
        brain_action_tools,
        mock_cm,
        coordinator_session,
    ):
        """Coordinator direct comms tools do not expose arbitrary recipients."""
        mock_cm.assistant_whatsapp_number = "+15555557777"
        mock_cm.assistant_discord_bot_id = "discord-bot-123"
        mock_cm.assistant_slack_bot_user_id = "slack-bot-123"
        mock_cm.assistant_has_teams = True

        tools = brain_action_tools.as_tools()

        assert "send_discord_channel_message" not in tools
        assert "send_slack_channel_message" not in tools
        assert "create_teams_channel" not in tools
        assert "create_teams_meet" in tools

        contact_id_tools = {
            "send_sms",
            "send_whatsapp",
            "send_discord_message",
            "send_slack_message",
            "send_teams_message",
            "send_unify_message",
            "send_api_response",
            "make_call",
            "make_whatsapp_call",
        }
        for tool_name in contact_id_tools:
            schema = method_to_schema(
                tools[tool_name],
                tool_name=tool_name,
                include_class_name=False,
            )
            props = schema["function"]["parameters"]["properties"]
            assert "contact_id" not in props
            assert "boss" in schema["function"]["description"].lower()

        email_schema = method_to_schema(
            tools["send_email"],
            tool_name="send_email",
            include_class_name=False,
        )
        email_props = email_schema["function"]["parameters"]["properties"]
        assert {"to", "cc", "bcc", "reply_all", "contact_id"}.isdisjoint(
            email_props,
        )
        assert {"subject", "body"}.issubset(email_props)

        teams_schema = method_to_schema(
            tools["send_teams_message"],
            tool_name="send_teams_message",
            include_class_name=False,
        )
        teams_props = teams_schema["function"]["parameters"]["properties"]
        assert {"contact_id", "team_id", "channel_id", "chat_topic"}.isdisjoint(
            teams_props,
        )
        assert {"content", "chat_id"}.issubset(teams_props)

        meet_schema = method_to_schema(
            tools["create_teams_meet"],
            tool_name="create_teams_meet",
            include_class_name=False,
        )
        meet_props = meet_schema["function"]["parameters"]["properties"]
        assert "attendee_contact_ids" not in meet_props
        assert "email_address" not in meet_props

    @pytest.mark.asyncio
    async def test_coordinator_comms_wrappers_force_boss_recipient(
        self,
        brain_action_tools,
        mock_cm,
        coordinator_session,
    ):
        """Coordinator exposed tools delegate to CommsPrimitives with the boss contact."""
        mock_cm.assistant_whatsapp_number = "+15555557777"
        mock_cm.assistant_discord_bot_id = "discord-bot-123"
        mock_cm.assistant_slack_bot_user_id = "slack-bot-123"
        mock_cm.assistant_has_teams = True
        tools = brain_action_tools.as_tools()

        brain_action_tools._comms.send_sms = AsyncMock(return_value={"status": "ok"})
        brain_action_tools._comms.send_whatsapp = AsyncMock(
            return_value={"status": "ok"},
        )
        brain_action_tools._comms.send_discord_message = AsyncMock(
            return_value={"status": "ok"},
        )
        brain_action_tools._comms.send_slack_message = AsyncMock(
            return_value={"status": "ok"},
        )
        brain_action_tools._comms.send_teams_message = AsyncMock(
            return_value={"status": "ok"},
        )
        brain_action_tools._comms.send_unify_message = AsyncMock(
            return_value={"status": "ok"},
        )
        brain_action_tools._comms.send_api_response = AsyncMock(
            return_value={"status": "ok"},
        )
        brain_action_tools._comms.send_email = AsyncMock(return_value={"status": "ok"})
        brain_action_tools._comms.make_call = AsyncMock(return_value={"status": "ok"})
        brain_action_tools._comms.make_whatsapp_call = AsyncMock(
            return_value={"status": "ok"},
        )
        brain_action_tools._comms.create_teams_meet = AsyncMock(
            return_value={"status": "ok"},
        )

        await tools["send_sms"](content="hello")
        brain_action_tools._comms.send_sms.assert_called_once_with(
            contact_id=SESSION_DETAILS.boss_contact_id,
            content="hello",
        )

        await tools["send_whatsapp"](content="hello")
        brain_action_tools._comms.send_whatsapp.assert_called_once_with(
            contact_id=SESSION_DETAILS.boss_contact_id,
            content="hello",
            attachment_filepath=None,
        )

        await tools["send_discord_message"](content="hello")
        brain_action_tools._comms.send_discord_message.assert_called_once_with(
            contact_id=SESSION_DETAILS.boss_contact_id,
            content="hello",
        )

        await tools["send_slack_message"](content="hello", team_id="T1")
        brain_action_tools._comms.send_slack_message.assert_called_once_with(
            contact_id=SESSION_DETAILS.boss_contact_id,
            content="hello",
            team_id="T1",
            thread_ts=None,
        )

        await tools["send_unify_message"](content="hello")
        brain_action_tools._comms.send_unify_message.assert_called_once_with(
            content="hello",
            contact_id=SESSION_DETAILS.boss_contact_id,
            attachment_filepath=None,
        )

        await tools["send_api_response"](content="done", tags=["route"])
        brain_action_tools._comms.send_api_response.assert_called_once_with(
            content="done",
            contact_id=SESSION_DETAILS.boss_contact_id,
            attachment_filepaths=None,
            tags=["route"],
        )

        await tools["make_call"](context="briefing")
        brain_action_tools._comms.make_call.assert_called_once_with(
            contact_id=SESSION_DETAILS.boss_contact_id,
            context="briefing",
        )

        await tools["make_whatsapp_call"](context="briefing")
        brain_action_tools._comms.make_whatsapp_call.assert_called_once_with(
            contact_id=SESSION_DETAILS.boss_contact_id,
            context="briefing",
        )

        await tools["send_email"](
            subject="Hi",
            body="Body",
            thread_id="gmail-thread-1",
        )
        brain_action_tools._comms.send_email.assert_called_once_with(
            to=[SESSION_DETAILS.boss_contact_id],
            subject="Hi",
            body="Body",
            email_id_to_reply_to=None,
            thread_id="gmail-thread-1",
            attachment_filepath=None,
        )

        await tools["send_teams_message"](
            content="hello",
            chat_id="chat-1",
        )
        brain_action_tools._comms.send_teams_message.assert_called_once_with(
            contact_id=SESSION_DETAILS.boss_contact_id,
            content="hello",
            chat_id="chat-1",
            attachment_filepath=None,
        )

        await tools["create_teams_meet"](
            subject="Sync",
        )
        brain_action_tools._comms.create_teams_meet.assert_called_once_with(
            mode="scheduled",
            subject="Sync",
            start=None,
            duration_minutes=30,
            timezone="UTC",
            attendee_contact_ids=[SESSION_DETAILS.boss_contact_id],
            body_html=None,
            location=None,
        )

    def test_regular_comms_schemas_keep_contact_addressed_surface(
        self,
        brain_action_tools,
        mock_cm,
    ):
        """Regular assistants keep the existing contact-addressed comms surface."""
        previous_is_coordinator = SESSION_DETAILS.is_coordinator
        SESSION_DETAILS.is_coordinator = False
        mock_cm.assistant_whatsapp_number = "+15555557777"
        mock_cm.assistant_discord_bot_id = "discord-bot-123"
        mock_cm.assistant_slack_bot_user_id = "slack-bot-123"
        mock_cm.assistant_has_teams = True
        try:
            tools = brain_action_tools.as_tools()
        finally:
            SESSION_DETAILS.is_coordinator = previous_is_coordinator

        assert "send_discord_channel_message" in tools
        assert "send_slack_channel_message" in tools
        assert "create_teams_channel" in tools

        sms_props = method_to_schema(
            tools["send_sms"],
            tool_name="send_sms",
            include_class_name=False,
        )["function"]["parameters"]["properties"]
        assert "contact_id" in sms_props

        email_props = method_to_schema(
            tools["send_email"],
            tool_name="send_email",
            include_class_name=False,
        )["function"]["parameters"]["properties"]
        assert {"to", "cc", "bcc", "reply_all"}.issubset(email_props)

        teams_props = method_to_schema(
            tools["send_teams_message"],
            tool_name="send_teams_message",
            include_class_name=False,
        )["function"]["parameters"]["properties"]
        assert {"contact_id", "team_id", "channel_id", "chat_topic"}.issubset(
            teams_props,
        )


class TestHangUpTool:
    """Tests for the universal hang_up tool and its exposure in as_tools."""

    def test_hang_up_exposed_only_during_voice_session(self, mock_cm):
        """hang_up appears while in a voice session; call-starting tools do not,
        and the standalone leave_* tools are gone."""
        mock_cm.in_voice_session = True
        mock_cm.call_manager._call_channel = "phone_call"
        with patch(
            "unify.conversation_manager.domains.brain_action_tools.get_event_broker",
        ) as mock_broker:
            mock_broker.return_value = MagicMock()
            mock_broker.return_value.publish = AsyncMock()
            tools = ConversationManagerBrainActionTools(mock_cm).as_tools()

        assert "hang_up" in tools
        assert "make_call" not in tools
        assert "join_google_meet" not in tools
        assert "leave_google_meet" not in tools
        assert "leave_teams_meet" not in tools

    def test_hang_up_absent_when_not_in_voice_session(self, brain_action_tools):
        """hang_up is not offered outside a live voice session."""
        tools = brain_action_tools.as_tools()
        assert "hang_up" not in tools
        assert "make_call" in tools

    @pytest.mark.asyncio
    async def test_hang_up_defers_then_teardown_ends_phone_call(
        self,
        brain_action_tools,
        mock_cm,
    ):
        """hang_up records a deferred intent (does NOT end the call itself); the
        teardown helper later routes a phone call to call_manager.end_call()."""
        mock_cm.in_voice_session = True
        mock_cm.call_manager._call_channel = "phone_call"
        mock_cm.call_manager.has_active_google_meet = False
        mock_cm.call_manager.has_active_teams_meet = False
        mock_cm.call_manager.end_call = AsyncMock()
        mock_cm.call_manager.await_ready_for_outbound_call = AsyncMock(
            return_value=True,
        )

        result = await brain_action_tools.hang_up()

        # Deferred: nothing torn down yet, intent recorded for _run_llm.
        assert result["status"] == "ok"
        assert mock_cm._pending_hang_up is True
        assert mock_cm._pending_hang_up_teardown is not None
        mock_cm.call_manager.end_call.assert_not_awaited()

        # The teardown helper performs the real end_call once invoked.
        teardown = await mock_cm._pending_hang_up_teardown()
        mock_cm.call_manager.end_call.assert_awaited_once()
        assert teardown["status"] == "ok"

    @pytest.mark.asyncio
    async def test_hang_up_teardown_routes_unify_meet_to_end_call(
        self,
        brain_action_tools,
        mock_cm,
    ):
        """A live Unify Meet teardown also routes through call_manager.end_call()."""
        mock_cm.in_voice_session = True
        mock_cm.call_manager._call_channel = "unify_meet"
        mock_cm.call_manager.has_active_google_meet = False
        mock_cm.call_manager.has_active_teams_meet = False
        mock_cm.call_manager.end_call = AsyncMock()
        mock_cm.call_manager.await_ready_for_outbound_call = AsyncMock(
            return_value=True,
        )

        await brain_action_tools.hang_up()
        mock_cm.call_manager.end_call.assert_not_awaited()

        await brain_action_tools._perform_hang_up_teardown()
        mock_cm.call_manager.end_call.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_hang_up_teardown_routes_google_meet_to_leave(
        self,
        brain_action_tools,
        mock_cm,
    ):
        """A live Google Meet teardown publishes GoogleMeetEnded (leave path)."""
        from unify.conversation_manager.events import GoogleMeetEnded

        mock_cm.in_voice_session = True
        mock_cm.call_manager._call_channel = "google_meet"
        mock_cm.call_manager.has_active_google_meet = True
        mock_cm.call_manager._disconnect_contact = {}
        mock_cm.call_manager.end_call = AsyncMock()
        brain_action_tools._notify_browser_meet_leave = AsyncMock()

        # hang_up defers; the leave only happens on teardown.
        await brain_action_tools.hang_up()
        brain_action_tools._notify_browser_meet_leave.assert_not_awaited()

        result = await brain_action_tools._perform_hang_up_teardown()

        brain_action_tools._notify_browser_meet_leave.assert_awaited_once_with(
            "googlemeet",
        )
        mock_cm.call_manager.end_call.assert_not_awaited()
        published_topics = [
            call.args[0]
            for call in brain_action_tools._event_broker.publish.await_args_list
        ]
        assert GoogleMeetEnded(contact={}).topic in published_topics
        assert result["status"] == "ok"

    @pytest.mark.asyncio
    async def test_hang_up_no_active_session_returns_error(
        self,
        brain_action_tools,
        mock_cm,
    ):
        """With nothing live, hang_up reports there is nothing to end."""
        mock_cm.in_voice_session = False
        mock_cm.call_manager._call_channel = None
        mock_cm.call_manager.end_call = AsyncMock()

        result = await brain_action_tools.hang_up()

        assert result["status"] == "error"
        mock_cm.call_manager.end_call.assert_not_awaited()

    def test_call_tools_hidden_when_worker_not_ready(self, mock_cm):
        """Between sessions, make_call/make_whatsapp_call are withheld until the
        voice worker has a freshly prewarmed idle process ready."""
        mock_cm.in_voice_session = False
        mock_cm.assistant_whatsapp_number = "+15555550000"
        mock_cm.call_manager.is_ready_for_outbound_call = False
        with patch(
            "unify.conversation_manager.domains.brain_action_tools.get_event_broker",
        ) as mock_broker:
            mock_broker.return_value = MagicMock()
            mock_broker.return_value.publish = AsyncMock()
            tools = ConversationManagerBrainActionTools(mock_cm).as_tools()

        assert "make_call" not in tools
        assert "make_whatsapp_call" not in tools
        # Text channels stay available regardless of voice-line readiness.
        assert "send_sms" in tools
        assert "send_whatsapp" in tools

    def test_call_tools_exposed_when_worker_ready(self, mock_cm):
        """With no live session and the worker ready, the call-starting tools
        are offered again."""
        mock_cm.in_voice_session = False
        mock_cm.assistant_whatsapp_number = "+15555550000"
        mock_cm.call_manager.is_ready_for_outbound_call = True
        with patch(
            "unify.conversation_manager.domains.brain_action_tools.get_event_broker",
        ) as mock_broker:
            mock_broker.return_value = MagicMock()
            mock_broker.return_value.publish = AsyncMock()
            tools = ConversationManagerBrainActionTools(mock_cm).as_tools()

        assert "make_call" in tools
        assert "make_whatsapp_call" in tools

    @pytest.mark.asyncio
    async def test_hang_up_teardown_awaits_readiness_before_returning(
        self,
        brain_action_tools,
        mock_cm,
    ):
        """The teardown helper ends the session, then awaits the worker readiness
        signal and reports whether a new call is safe."""
        mock_cm.in_voice_session = True
        mock_cm.call_manager._call_channel = "unify_meet"
        mock_cm.call_manager.has_active_google_meet = False
        mock_cm.call_manager.has_active_teams_meet = False
        mock_cm.call_manager.end_call = AsyncMock()
        mock_cm.call_manager.await_ready_for_outbound_call = AsyncMock(
            return_value=True,
        )

        await brain_action_tools.hang_up()
        result = await brain_action_tools._perform_hang_up_teardown()

        mock_cm.call_manager.end_call.assert_awaited_once()
        mock_cm.call_manager.await_ready_for_outbound_call.assert_awaited_once()
        assert result["status"] == "ok"
        assert result["ready_for_outbound_call"] is True

    @pytest.mark.asyncio
    async def test_hang_up_teardown_reports_not_ready_on_timeout(
        self,
        brain_action_tools,
        mock_cm,
    ):
        """If the worker never re-warms within the window, the teardown still ends
        the call but tells the brain the line is not ready for a new call yet."""
        mock_cm.in_voice_session = True
        mock_cm.call_manager._call_channel = "whatsapp_call"
        mock_cm.call_manager.has_active_google_meet = False
        mock_cm.call_manager.has_active_teams_meet = False
        mock_cm.call_manager.end_call = AsyncMock()
        mock_cm.call_manager.await_ready_for_outbound_call = AsyncMock(
            return_value=False,
        )

        await brain_action_tools.hang_up()
        result = await brain_action_tools._perform_hang_up_teardown()

        assert result["status"] == "ok"
        assert result["ready_for_outbound_call"] is False


class TestWaitTool:
    """Tests for wait tool."""

    @pytest.mark.asyncio
    async def test_returns_waiting_status(self, brain_action_tools):
        """Wait tool returns waiting status."""
        result = await brain_action_tools.wait()
        assert result == {"status": "waiting", "delay": None}

    def test_has_docstring(self, brain_action_tools):
        """Wait tool has descriptive docstring."""
        assert brain_action_tools.wait.__doc__ is not None
        assert "Wait" in brain_action_tools.wait.__doc__


class TestSendSmsTool:
    """Tests for send_sms tool."""

    @pytest.mark.asyncio
    async def test_requires_contact_id(self, brain_action_tools):
        """Raises TypeError if contact_id not provided."""
        with pytest.raises(TypeError):
            await brain_action_tools.send_sms(content="Hello")

    @pytest.mark.asyncio
    async def test_has_docstring(self, brain_action_tools):
        """Send SMS tool has descriptive docstring."""
        assert brain_action_tools.send_sms.__doc__ is not None
        assert "SMS" in brain_action_tools.send_sms.__doc__

    @pytest.mark.asyncio
    async def test_sends_sms_to_contact(
        self,
        brain_action_tools,
        mock_cm,
    ):
        """Sends SMS when given a valid contact_id with phone number."""
        contact = {
            "contact_id": 5,
            "first_name": "Test",
            "surname": "Person",
            "phone_number": "+1234567890",
            "should_respond": True,
        }
        _setup_mock_contacts(mock_cm.contact_index, [contact])

        result = await brain_action_tools.send_sms(
            contact_id=5,
            content="Hello",
        )

        assert result["status"] == "ok"

    @pytest.mark.asyncio
    async def test_returns_error_for_contact_without_phone(
        self,
        brain_action_tools,
        mock_cm,
    ):
        """Returns error when contact has no phone number."""
        contact_without_phone = {
            "contact_id": 5,
            "first_name": "NoPhone",
            "surname": "Person",
            "email_address": "nophone@example.com",
            "should_respond": True,
        }
        _setup_mock_contacts(mock_cm.contact_index, [contact_without_phone])

        result = await brain_action_tools.send_sms(
            contact_id=5,
            content="Hello",
        )

        assert result["status"] == "error"
        assert "does not have" in result["error"]
        assert "phone" in result["error"].lower()


class TestSendDiscordMessageTool:
    """Tests for send_discord_message tool metadata."""

    def test_uses_dm_only_signature(self, brain_action_tools):
        """Channel reply args stay on send_discord_channel_message only."""
        parameters = inspect.signature(
            brain_action_tools.send_discord_message,
        ).parameters

        assert "contact_id" in parameters
        assert "content" in parameters
        assert "discord_id" in parameters
        assert "channel_id" not in parameters
        assert "guild_id" not in parameters

    def test_docstring_points_channel_replies_to_channel_tool(self, brain_action_tools):
        """DM docstring steers channel posting to the dedicated channel tool."""
        doc = brain_action_tools.send_discord_message.__doc__
        assert doc is not None
        assert "send_discord_channel_message" in doc


class TestSendUnifyMessageTool:
    """Tests for send_unify_message tool."""

    def test_has_docstring(self, brain_action_tools):
        """Send Unify message tool has descriptive docstring."""
        assert brain_action_tools.send_unify_message.__doc__ is not None
        assert "Unify" in brain_action_tools.send_unify_message.__doc__

    def test_docstring_mentions_attachment(self, brain_action_tools):
        """Send Unify message docstring mentions attachment parameter."""
        doc = brain_action_tools.send_unify_message.__doc__
        assert "attachment" in doc.lower()

    @pytest.mark.asyncio
    async def test_returns_error_for_file_not_found(
        self,
        brain_action_tools,
        mock_cm,
        sample_contacts,
    ):
        """Returns error when attachment file not found."""
        _setup_mock_contacts(mock_cm.contact_index, sample_contacts)

        result = await brain_action_tools.send_unify_message(
            content="Here's the file",
            contact_id=1,
            attachment_filepath="/nonexistent/file.pdf",
        )

        assert result["status"] == "error"
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_returns_error_for_file_too_large(
        self,
        brain_action_tools,
        mock_cm,
        sample_contacts,
        tmp_path,
    ):
        """Returns error when attachment exceeds size limit."""
        _setup_mock_contacts(mock_cm.contact_index, sample_contacts)

        # Create a file larger than 25MB
        large_file = tmp_path / "large_file.bin"
        large_file.write_bytes(b"x" * (26 * 1024 * 1024))

        # Patch the class in its home module so the deferred import inside
        # send_unify_message picks up the rooted adapter.
        rooted = type(
            "RootedAdapter",
            (LocalFileSystemAdapter,),
            {
                "__init__": lambda self: LocalFileSystemAdapter.__init__(
                    self,
                    root=str(tmp_path),
                ),
            },
        )
        with patch(
            "unify.file_manager.filesystem_adapters.local_adapter.LocalFileSystemAdapter",
            rooted,
        ):
            result = await brain_action_tools.send_unify_message(
                content="Here's the file",
                contact_id=1,
                attachment_filepath="large_file.bin",
            )

        assert result["status"] == "error"
        assert "too large" in result["error"].lower()
        assert "25MB" in result["error"]

    @pytest.mark.asyncio
    async def test_send_with_attachment_success(
        self,
        brain_action_tools,
        mock_cm,
        sample_contacts,
        tmp_path,
    ):
        """Successfully sends message with attachment when file exists and upload succeeds."""
        _setup_mock_contacts(mock_cm.contact_index, sample_contacts)

        # Create a small test file
        test_file = tmp_path / "test_document.pdf"
        test_file.write_bytes(b"PDF content here")

        # Root the adapter at tmp_path so test files pass the subpath check
        with (
            patch(
                "unify.file_manager.filesystem_adapters.local_adapter.LocalFileSystemAdapter",
                lambda: LocalFileSystemAdapter(root=str(tmp_path)),
            ),
            patch(
                "unify.comms.primitives.comms_utils.upload_unify_attachment",
            ) as mock_upload,
            patch(
                "unify.comms.primitives.comms_utils.send_unify_message",
            ) as mock_send,
        ):
            # Configure mocks
            mock_upload.return_value = {
                "id": "test-uuid",
                "filename": "test_document.pdf",
                "url": "https://storage.googleapis.com/signed-url",
            }
            mock_send.return_value = {"success": True}

            result = await brain_action_tools.send_unify_message(
                content="Here's the document",
                contact_id=1,
                attachment_filepath=str(test_file),
            )

            assert result["status"] == "ok"

            # Verify upload was called with correct args
            mock_upload.assert_called_once()
            call_args = mock_upload.call_args
            assert call_args.kwargs["filename"] == "test_document.pdf"
            assert b"PDF content here" in call_args.kwargs["file_content"]

            # Verify send was called with the attachment
            mock_send.assert_called_once()
            send_args = mock_send.call_args
            assert send_args.kwargs["content"] == "Here's the document"
            assert send_args.kwargs["attachment"]["id"] == "test-uuid"
            assert send_args.kwargs["attachment"]["filename"] == "test_document.pdf"

    @pytest.mark.asyncio
    async def test_returns_error_when_upload_fails(
        self,
        brain_action_tools,
        mock_cm,
        sample_contacts,
        tmp_path,
    ):
        """Returns error when attachment upload fails."""
        _setup_mock_contacts(mock_cm.contact_index, sample_contacts)

        # Create a small test file
        test_file = tmp_path / "test_document.pdf"
        test_file.write_bytes(b"PDF content")

        # Root the adapter at tmp_path so test files pass the subpath check
        with (
            patch(
                "unify.file_manager.filesystem_adapters.local_adapter.LocalFileSystemAdapter",
                lambda: LocalFileSystemAdapter(root=str(tmp_path)),
            ),
            patch(
                "unify.comms.primitives.comms_utils.upload_unify_attachment",
            ) as mock_upload,
        ):
            mock_upload.return_value = {
                "success": False,
                "error": "Storage service unavailable",
            }

            result = await brain_action_tools.send_unify_message(
                content="Here's the document",
                contact_id=1,
                attachment_filepath=str(test_file),
            )

            assert result["status"] == "error"
            assert "upload" in result["error"].lower()


class TestSendEmailTool:
    """Tests for send_email tool."""

    @pytest.mark.asyncio
    async def test_requires_at_least_one_recipient(self, brain_action_tools):
        """Returns error if no recipients provided."""
        result = await brain_action_tools.send_email(subject="Test", body="Body")
        assert result["status"] == "error"
        assert "at least one recipient" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_reply_all_mutually_exclusive_with_recipients(
        self,
        brain_action_tools,
    ):
        """Returns error if reply_all=True and to/cc/bcc are also provided."""
        result = await brain_action_tools.send_email(
            to=[1],
            reply_all=True,
            subject="Test",
            body="Body",
        )
        assert result["status"] == "error"
        assert "mutually exclusive" in result["error"].lower()

    def test_has_docstring(self, brain_action_tools):
        """Send email tool has descriptive docstring."""
        assert brain_action_tools.send_email.__doc__ is not None
        assert "email" in brain_action_tools.send_email.__doc__.lower()

    def test_docstring_mentions_attachment(self, brain_action_tools):
        """Send email docstring mentions attachment parameter."""
        doc = brain_action_tools.send_email.__doc__
        assert "attachment" in doc.lower()

    def test_docstring_mentions_recipients(self, brain_action_tools):
        """Send email docstring mentions to/cc/bcc parameters."""
        doc = brain_action_tools.send_email.__doc__
        assert "to" in doc.lower()
        assert "cc" in doc.lower()
        assert "bcc" in doc.lower()

    @pytest.mark.asyncio
    async def test_resolves_contact_id_to_email(
        self,
        brain_action_tools,
        mock_cm,
        sample_contacts,
    ):
        """Resolves contact_id in to list to email address."""
        _setup_mock_contacts(mock_cm.contact_index, sample_contacts)

        with patch(
            "unify.comms.primitives.comms_utils.send_email_via_address",
        ) as mock_send:
            mock_send.return_value = {"success": True, "id": "sent-email-123"}

            result = await brain_action_tools.send_email(
                to=[1],  # contact_id
                subject="Test",
                body="Hello",
            )

            assert result["status"] == "ok"
            mock_send.assert_called_once()
            # Should resolve contact_id 1 to alice@example.com
            assert mock_send.call_args.kwargs["to"] == ["alice@example.com"]

    @pytest.mark.asyncio
    async def test_forwards_thread_id_separately_from_reply_email_id(
        self,
        brain_action_tools,
        mock_cm,
        sample_contacts,
    ):
        """Forwards provider thread targeting separately from RFC reply headers."""
        _setup_mock_contacts(mock_cm.contact_index, sample_contacts)

        with patch(
            "unify.comms.primitives.comms_utils.send_email_via_address",
        ) as mock_send:
            mock_send.return_value = {"success": True, "id": "sent-email-123"}

            result = await brain_action_tools.send_email(
                to=[1],
                subject="Re: Budget Discussion",
                body="Reply body",
                email_id_to_reply_to="<rfc-message-id@example.com>",
                thread_id="gmail-thread-123",
            )

            assert result["status"] == "ok"
            assert (
                mock_send.call_args.kwargs["email_id"] == "<rfc-message-id@example.com>"
            )
            assert mock_send.call_args.kwargs["thread_id"] == "gmail-thread-123"

    @pytest.mark.asyncio
    async def test_returns_error_for_file_not_found(
        self,
        brain_action_tools,
        mock_cm,
        sample_contacts,
    ):
        """Returns error when attachment file not found."""
        _setup_mock_contacts(mock_cm.contact_index, sample_contacts)

        result = await brain_action_tools.send_email(
            to=[1],
            subject="Test",
            body="Hello",
            attachment_filepath="/nonexistent/file.pdf",
        )

        assert result["status"] == "error"
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_returns_error_for_file_too_large(
        self,
        brain_action_tools,
        mock_cm,
        sample_contacts,
        tmp_path,
    ):
        """Returns error when attachment exceeds size limit."""
        _setup_mock_contacts(mock_cm.contact_index, sample_contacts)

        # Create a file larger than 25MB (use a sparse approach for speed)
        large_file = tmp_path / "large_file.bin"
        # Write 26MB of data
        large_file.write_bytes(b"x" * (26 * 1024 * 1024))

        # Root the adapter at tmp_path so test files pass the subpath check
        with patch(
            "unify.file_manager.filesystem_adapters.local_adapter.LocalFileSystemAdapter",
            lambda: LocalFileSystemAdapter(root=str(tmp_path)),
        ):
            result = await brain_action_tools.send_email(
                to=[1],
                subject="Test",
                body="Hello",
                attachment_filepath=str(large_file),
            )

        assert result["status"] == "error"
        assert "too large" in result["error"].lower()
        assert "25MB" in result["error"]

    @pytest.mark.asyncio
    async def test_send_with_attachment_success(
        self,
        brain_action_tools,
        mock_cm,
        sample_contacts,
        tmp_path,
    ):
        """Successfully sends email with attachment when file exists."""
        _setup_mock_contacts(mock_cm.contact_index, sample_contacts)

        # Create a small test file
        test_file = tmp_path / "report.pdf"
        test_file.write_bytes(b"PDF report content")

        # Root the adapter at tmp_path so test files pass the subpath check
        with (
            patch(
                "unify.file_manager.filesystem_adapters.local_adapter.LocalFileSystemAdapter",
                lambda: LocalFileSystemAdapter(root=str(tmp_path)),
            ),
            patch(
                "unify.comms.primitives.comms_utils.send_email_via_address",
            ) as mock_send,
        ):
            mock_send.return_value = {"success": True, "id": "sent-email-123"}

            result = await brain_action_tools.send_email(
                to=[1],
                subject="Quarterly Report",
                body="Please find the report attached.",
                attachment_filepath=str(test_file),
            )

            assert result["status"] == "ok"

            # Verify send was called with attachment
            mock_send.assert_called_once()
            call_args = mock_send.call_args
            assert call_args.kwargs["subject"] == "Quarterly Report"
            assert call_args.kwargs["attachment"] is not None
            assert call_args.kwargs["attachment"]["filename"] == "report.pdf"
            # Verify base64 content
            import base64

            decoded = base64.b64decode(call_args.kwargs["attachment"]["content_base64"])
            assert decoded == b"PDF report content"

    @pytest.mark.asyncio
    async def test_cc_only_email_valid(
        self,
        brain_action_tools,
        mock_cm,
        sample_contacts,
    ):
        """Accepts email with only CC recipients (empty TO is valid)."""
        _setup_mock_contacts(mock_cm.contact_index, sample_contacts)

        with patch(
            "unify.comms.primitives.comms_utils.send_email_via_address",
        ) as mock_send:
            mock_send.return_value = {"success": True, "id": "sent-email-123"}

            result = await brain_action_tools.send_email(
                cc=[1],  # Only CC, no TO
                subject="FYI",
                body="Just keeping you in the loop.",
            )

            assert result["status"] == "ok"
            mock_send.assert_called_once()
            assert mock_send.call_args.kwargs["to"] == []
            assert mock_send.call_args.kwargs["cc"] == ["alice@example.com"]

    @pytest.mark.asyncio
    async def test_bcc_only_email_valid(
        self,
        brain_action_tools,
        mock_cm,
        sample_contacts,
    ):
        """Accepts email with only BCC recipients."""
        _setup_mock_contacts(mock_cm.contact_index, sample_contacts)

        with patch(
            "unify.comms.primitives.comms_utils.send_email_via_address",
        ) as mock_send:
            mock_send.return_value = {"success": True, "id": "sent-email-123"}

            result = await brain_action_tools.send_email(
                bcc=[1],  # Only BCC
                subject="Private",
                body="Confidential message.",
            )

            assert result["status"] == "ok"
            mock_send.assert_called_once()
            assert mock_send.call_args.kwargs["to"] == []
            assert mock_send.call_args.kwargs["bcc"] == ["alice@example.com"]

    @pytest.mark.asyncio
    async def test_deduplicates_recipients(
        self,
        brain_action_tools,
        mock_cm,
        sample_contacts,
    ):
        """Deduplicates when same contact_id appears multiple times."""
        _setup_mock_contacts(mock_cm.contact_index, sample_contacts)

        with patch(
            "unify.comms.primitives.comms_utils.send_email_via_address",
        ) as mock_send:
            mock_send.return_value = {"success": True, "id": "sent-email-123"}

            # Provide same contact_id twice
            result = await brain_action_tools.send_email(
                to=[1, 1],  # Both resolve to alice@example.com
                subject="Test",
                body="Hello",
            )

            assert result["status"] == "ok"
            # Should deduplicate to single recipient
            assert len(mock_send.call_args.kwargs["to"]) == 1
            assert mock_send.call_args.kwargs["to"] == ["alice@example.com"]

    @pytest.mark.asyncio
    async def test_multiple_recipients_all_fields(
        self,
        brain_action_tools,
        mock_cm,
        sample_contacts,
    ):
        """Handles multiple contact_ids in to, cc, and bcc simultaneously."""
        _setup_mock_contacts(mock_cm.contact_index, sample_contacts)

        with patch(
            "unify.comms.primitives.comms_utils.send_email_via_address",
        ) as mock_send:
            mock_send.return_value = {"success": True, "id": "sent-email-123"}

            result = await brain_action_tools.send_email(
                to=[1],
                cc=[2],
                bcc=[1],  # duplicate with to — will deduplicate
                subject="Team Update",
                body="Update for everyone.",
            )

            assert result["status"] == "ok"
            mock_send.assert_called_once()
            assert len(mock_send.call_args.kwargs["to"]) == 1
            assert len(mock_send.call_args.kwargs["cc"]) == 1


class TestMakeCallTool:
    """Tests for make_call tool."""

    @pytest.mark.asyncio
    async def test_requires_contact_id(self, brain_action_tools):
        """Raises TypeError if contact_id not provided."""
        with pytest.raises(TypeError):
            await brain_action_tools.make_call()

    def test_has_docstring(self, brain_action_tools):
        """Make call tool has descriptive docstring."""
        assert brain_action_tools.make_call.__doc__ is not None
        assert "call" in brain_action_tools.make_call.__doc__.lower()
        assert (
            "Mission briefing for the voice agent"
            in brain_action_tools.make_call.__doc__
        )

    @pytest.mark.asyncio
    async def test_calls_contact(
        self,
        brain_action_tools,
        mock_cm,
    ):
        """Makes call when given a valid contact_id with phone number."""
        contact = {
            "contact_id": 5,
            "first_name": "Test",
            "surname": "Person",
            "phone_number": "+1234567890",
            "should_respond": True,
        }
        _setup_mock_contacts(mock_cm.contact_index, [contact])

        result = await brain_action_tools.make_call(
            contact_id=5,
            context="Calling to confirm the Thursday meeting",
        )

        assert result["status"] == "ok"

    @pytest.mark.asyncio
    async def test_returns_error_for_contact_without_phone(
        self,
        brain_action_tools,
        mock_cm,
    ):
        """Returns error when contact has no phone number."""
        contact_without_phone = {
            "contact_id": 5,
            "first_name": "NoPhone",
            "surname": "Person",
            "email_address": "nophone@example.com",
            "should_respond": True,
        }
        _setup_mock_contacts(mock_cm.contact_index, [contact_without_phone])

        result = await brain_action_tools.make_call(
            contact_id=5,
            context="Calling to confirm the Thursday meeting",
        )

        assert result["status"] == "error"
        assert "does not have" in result["error"]
        assert "phone" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_context_stores_initial_notification(
        self,
        brain_action_tools,
        mock_cm,
    ):
        """The context param stores initial_notification on call_manager
        before the call is placed, so CallManager can publish it to the
        fast brain after the subprocess spawns."""
        contact = {
            "contact_id": 5,
            "first_name": "Test",
            "surname": "Person",
            "phone_number": "+1234567890",
            "should_respond": True,
        }
        _setup_mock_contacts(mock_cm.contact_index, [contact])

        guidance_text = "Confirm the Thursday 3pm meeting"
        result = await brain_action_tools.make_call(
            contact_id=5,
            context=guidance_text,
        )

        assert result["status"] == "ok"
        assert mock_cm.call_manager.initial_notification == guidance_text

    @pytest.mark.asyncio
    async def test_context_is_required(self, brain_action_tools):
        """context is a required argument — calling without it raises TypeError."""
        with pytest.raises(TypeError):
            await brain_action_tools.make_call(contact_id=5)


class TestActTool:
    """Tests for act tool."""

    def test_has_docstring(self, brain_action_tools):
        """Act tool has descriptive docstring."""
        assert brain_action_tools.act.__doc__ is not None
        assert len(brain_action_tools.act.__doc__) > 10


# =============================================================================
# Dynamic Action Steering Tools Tests
# =============================================================================


class TestBuildActionSteeringTools:
    """Tests for build_action_steering_tools method."""

    def test_returns_empty_dict_when_no_in_flight_actions(
        self,
        brain_action_tools,
        mock_cm,
    ):
        """Returns empty dict when no in-flight actions."""
        mock_cm.in_flight_actions = {}
        tools = brain_action_tools.build_action_steering_tools()
        assert tools == {}

    def test_generates_tools_for_in_flight_action(self, brain_action_tools, mock_cm):
        """Generates steering tools for each in-flight action.

        Note: With pause/resume flipping, only ONE of pause/resume is generated
        depending on the handle's pause state. For a running action (default),
        only pause is available.
        """
        # Create a mock handle that appears to be running (not paused)
        mock_handle = MagicMock()
        mock_handle._pause_event = MagicMock()
        mock_handle._pause_event.is_set.return_value = True  # Running (not paused)

        mock_cm.in_flight_actions = {
            0: {
                "query": "List all contacts",
                "handle": mock_handle,
                "handle_actions": [],
            },
        }
        tools = brain_action_tools.build_action_steering_tools()

        # Should have tools for ask, stop, interject, pause (NOT resume when running)
        # (but NOT answer_clarification without pending clarifications)
        tool_names = list(tools.keys())

        # Should have pause but NOT resume when running
        assert any(
            "pause_" in n for n in tool_names
        ), "Should have pause tool when running"
        assert not any(
            "resume_" in n for n in tool_names
        ), "Should NOT have resume tool when running"

        # Other steering tools should be present
        assert any("ask_" in n for n in tool_names)
        assert any("stop_" in n for n in tool_names)
        assert any("interject_" in n for n in tool_names)

    def test_tool_names_follow_expected_format(self, brain_action_tools, mock_cm):
        """Tool names follow the build_action_name format."""
        mock_cm.in_flight_actions = {
            0: {
                "query": "Search web",
                "handle": MagicMock(),
                "handle_actions": [],
            },
        }
        tools = brain_action_tools.build_action_steering_tools()
        for name in tools.keys():
            # Should be parseable
            parsed = parse_action_name(name)
            assert parsed.operation in [op.name for op in STEERING_OPERATIONS]
            assert parsed.handle_id == 0

    def test_generates_answer_clarification_when_pending(
        self,
        brain_action_tools,
        mock_cm,
    ):
        """Generates answer_clarification tool when pending clarifications exist."""
        mock_cm.in_flight_actions = {
            0: {
                "query": "Do something",
                "handle": MagicMock(),
                "handle_actions": [
                    {
                        "action_name": "clarification_request",
                        "query": "Need more info?",
                        "call_id": "call_123",
                    },
                ],
            },
        }
        tools = brain_action_tools.build_action_steering_tools()
        answer_tools = [n for n in tools.keys() if "answer_clarification" in n]
        assert len(answer_tools) == 1

    def test_no_answer_clarification_without_pending(self, brain_action_tools, mock_cm):
        """No answer_clarification tool when no pending clarifications."""
        mock_cm.in_flight_actions = {
            0: {
                "query": "Do something",
                "handle": MagicMock(),
                "handle_actions": [],  # No pending clarifications
            },
        }
        tools = brain_action_tools.build_action_steering_tools()
        answer_tools = [n for n in tools.keys() if "answer_clarification" in n]
        assert len(answer_tools) == 0

    def test_skips_answered_clarifications(self, brain_action_tools, mock_cm):
        """Does not generate tool for already answered clarifications."""
        mock_cm.in_flight_actions = {
            0: {
                "query": "Do something",
                "handle": MagicMock(),
                "handle_actions": [
                    {
                        "action_name": "clarification_request",
                        "query": "Need info?",
                        "call_id": "call_answered",
                        "response": "Here's the answer",  # Already answered
                    },
                ],
            },
        }
        tools = brain_action_tools.build_action_steering_tools()
        answer_tools = [n for n in tools.keys() if "answer_clarification" in n]
        assert len(answer_tools) == 0

    def test_handles_multiple_actions(self, brain_action_tools, mock_cm):
        """Generates tools for multiple in-flight actions."""
        # Create mock handles
        mock_handle1 = MagicMock()
        mock_handle1._pause_event = MagicMock()
        mock_handle1._pause_event.is_set.return_value = True  # Running

        mock_handle2 = MagicMock()
        mock_handle2._pause_event = MagicMock()
        mock_handle2._pause_event.is_set.return_value = True  # Running

        mock_cm.in_flight_actions = {
            0: {
                "query": "Action one",
                "handle": mock_handle1,
                "handle_actions": [],
            },
            1: {
                "query": "Action two",
                "handle": mock_handle2,
                "handle_actions": [],
            },
        }
        tools = brain_action_tools.build_action_steering_tools()
        # Should have steering tools for both actions
        action0_tools = [n for n in tools.keys() if "__0" in n]
        action1_tools = [n for n in tools.keys() if "__1" in n]
        assert len(action0_tools) > 0
        assert len(action1_tools) > 0

    def test_paused_handle_only_shows_resume(self, brain_action_tools, mock_cm):
        """When handle is paused, only resume_* is generated (not pause_*)."""
        # Create a paused handle (pause_event is cleared)
        mock_handle = MagicMock()
        mock_handle._pause_event = MagicMock()
        mock_handle._pause_event.is_set.return_value = False  # Paused

        mock_cm.in_flight_actions = {
            0: {
                "query": "Paused action",
                "handle": mock_handle,
                "handle_actions": [],
            },
        }
        tools = brain_action_tools.build_action_steering_tools()
        tool_names = list(tools.keys())

        # Should have resume but NOT pause when paused
        assert any(
            "resume_" in n for n in tool_names
        ), "Should have resume tool when paused"
        assert not any(
            "pause_" in n for n in tool_names
        ), "Should NOT have pause tool when paused"

    @pytest.mark.asyncio
    async def test_storage_check_handle_paused_shows_resume(
        self,
        brain_action_tools,
        mock_cm,
    ):
        """_StorageCheckHandle forwards inner handle's pause state.

        Regression: _StorageCheckHandle didn't expose _pause_event, so
        get_handle_paused_state returned None (unknown). This caused
        build_action_steering_tools to always offer pause_* and never
        offer resume_* — even when the inner loop was paused. The CM's
        LLM would then call pause_* repeatedly while trying to resume,
        producing serial "Pause" events visible on the frontend.
        """
        from unify.actor.code_act_actor import _StorageCheckHandle

        inner_handle = MagicMock()
        inner_handle._pause_event = asyncio.Event()
        inner_handle._pause_event.set()  # Start running

        # Block result() so the lifecycle stays in phase 1 ("task")
        _block = asyncio.Event()

        async def _blocking_result():
            await _block.wait()
            return "done"

        inner_handle.result = _blocking_result
        inner_handle.done = MagicMock(return_value=False)

        async def _block_forever():
            await asyncio.Event().wait()
            return {}

        inner_handle.next_notification = _block_forever
        inner_handle.next_clarification = _block_forever

        mock_actor = MagicMock()
        wrapped = _StorageCheckHandle(inner=inner_handle, actor=mock_actor)

        try:
            # Pause the inner handle
            inner_handle._pause_event.clear()

            mock_cm.in_flight_actions = {
                0: {
                    "query": "Access Dan's Gmail",
                    "handle": wrapped,
                    "handle_actions": [],
                },
            }

            tools = brain_action_tools.build_action_steering_tools()
            tool_names = list(tools.keys())

            assert any(
                "resume_" in n for n in tool_names
            ), f"Should have resume tool when paused, got: {tool_names}"
            assert not any(
                "pause_" in n for n in tool_names
            ), f"Should NOT have pause tool when paused, got: {tool_names}"
        finally:
            _block.set()
            wrapped._lifecycle_task.cancel()
            try:
                await wrapped._lifecycle_task
            except (asyncio.CancelledError, Exception):
                pass

    def test_running_handle_only_shows_pause(self, brain_action_tools, mock_cm):
        """When handle is running, only pause_* is generated (not resume_*)."""
        # Create a running handle (pause_event is set)
        mock_handle = MagicMock()
        mock_handle._pause_event = MagicMock()
        mock_handle._pause_event.is_set.return_value = True  # Running

        mock_cm.in_flight_actions = {
            0: {
                "query": "Running action",
                "handle": mock_handle,
                "handle_actions": [],
            },
        }
        tools = brain_action_tools.build_action_steering_tools()
        tool_names = list(tools.keys())

        # Should have pause but NOT resume when running
        assert any(
            "pause_" in n for n in tool_names
        ), "Should have pause tool when running"
        assert not any(
            "resume_" in n for n in tool_names
        ), "Should NOT have resume tool when running"

    def test_handle_without_pause_event_shows_pause(self, brain_action_tools, mock_cm):
        """When handle has no _pause_event (unknown state), defaults to pause_*."""
        # Create a handle without _pause_event
        mock_handle = MagicMock(spec=[])  # No attributes

        mock_cm.in_flight_actions = {
            0: {
                "query": "Action with unknown state",
                "handle": mock_handle,
                "handle_actions": [],
            },
        }
        tools = brain_action_tools.build_action_steering_tools()
        tool_names = list(tools.keys())

        # Should default to showing pause (assume running) when state unknown
        assert any("pause_" in n for n in tool_names), "Should default to pause tool"
        assert not any(
            "resume_" in n for n in tool_names
        ), "Should NOT have resume tool when state unknown"

    def test_steering_tools_have_docstrings(self, brain_action_tools, mock_cm):
        """Generated steering tools have docstrings."""
        mock_cm.in_flight_actions = {
            0: {
                "query": "Test action",
                "handle": MagicMock(),
                "handle_actions": [],
            },
        }
        tools = brain_action_tools.build_action_steering_tools()
        for name, fn in tools.items():
            assert fn.__doc__ is not None, f"{name} should have docstring"
            assert (
                "Test action" in fn.__doc__
            ), f"{name} docstring should mention action"

    def test_handles_none_in_flight_actions(self, brain_action_tools, mock_cm):
        """Handles None in_flight_actions gracefully."""
        mock_cm.in_flight_actions = None
        tools = brain_action_tools.build_action_steering_tools()
        assert tools == {}


class TestMakeSteeringTool:
    """Tests for _make_steering_tool method."""

    @pytest.mark.asyncio
    async def test_ask_operation_calls_handle_ask(self, brain_action_tools, mock_cm):
        """Ask operation calls handle.ask with parameter."""
        mock_handle = MagicMock()
        mock_ask_handle = MagicMock()
        mock_ask_handle.result = AsyncMock(return_value="Answer")
        mock_handle.ask = AsyncMock(return_value=mock_ask_handle)

        mock_cm.in_flight_actions = {
            0: {
                "query": "Test",
                "handle": mock_handle,
                "handle_actions": [],
            },
        }
        mock_cm._pending_steering_tasks = set()
        mock_cm._current_state_snapshot = None
        mock_cm.event_broker.publish = AsyncMock()

        tool = brain_action_tools._make_steering_tool(
            handle_id=0,
            handle=mock_handle,
            operation="ask",
            param_name="query",
            docstring="Ask a question",
            query="Test",
        )
        result = await tool(query="What is the status?")

        # The ask operation spawns a background task — await it directly.
        for task in list(mock_cm._pending_steering_tasks):
            await task

        mock_handle.ask.assert_called_once()
        assert result["status"] == "ok"
        assert result["operation"] == "ask"

    @pytest.mark.asyncio
    async def test_stop_operation_calls_handle_stop(self, brain_action_tools, mock_cm):
        """Stop moves the action to completed_actions."""
        mock_handle = MagicMock()
        mock_handle.stop = AsyncMock()

        mock_cm.in_flight_actions = {
            0: {
                "query": "Test",
                "handle": mock_handle,
                "handle_actions": [],
            },
        }

        tool = brain_action_tools._make_steering_tool(
            handle_id=0,
            handle=mock_handle,
            operation="stop",
            param_name="reason",
            docstring="Stop the action",
            query="Test",
        )
        result = await tool(reason="No longer needed")
        mock_handle.stop.assert_called_once_with(reason="No longer needed")
        assert result["operation"] == "stop"
        assert 0 not in mock_cm.in_flight_actions
        assert 0 in mock_cm.completed_actions

    @pytest.mark.asyncio
    async def test_interject_operation_calls_handle_interject(
        self,
        brain_action_tools,
        mock_cm,
    ):
        """Interject operation calls handle.interject."""
        mock_handle = MagicMock()
        mock_handle.interject = AsyncMock()

        mock_cm.in_flight_actions = {
            0: {
                "query": "Test",
                "handle": mock_handle,
                "handle_actions": [],
            },
        }

        tool = brain_action_tools._make_steering_tool(
            handle_id=0,
            handle=mock_handle,
            operation="interject",
            param_name="message",
            docstring="Interject a message",
            query="Test",
        )
        result = await tool(message="Important update")
        mock_handle.interject.assert_called_once()
        assert result["operation"] == "interject"

    @pytest.mark.asyncio
    async def test_pause_operation_calls_handle_pause(
        self,
        brain_action_tools,
        mock_cm,
    ):
        """Pause operation calls handle.pause."""
        mock_handle = MagicMock()
        mock_handle.pause = AsyncMock()

        mock_cm.in_flight_actions = {
            0: {
                "query": "Test",
                "handle": mock_handle,
                "handle_actions": [],
            },
        }

        tool = brain_action_tools._make_steering_tool(
            handle_id=0,
            handle=mock_handle,
            operation="pause",
            param_name="",
            docstring="Pause the action",
            query="Test",
        )
        result = await tool()
        mock_handle.pause.assert_called_once()
        assert result["operation"] == "pause"

    @pytest.mark.asyncio
    async def test_resume_operation_calls_handle_resume(
        self,
        brain_action_tools,
        mock_cm,
    ):
        """Resume operation calls handle.resume."""
        mock_handle = MagicMock()
        mock_handle.resume = AsyncMock()

        mock_cm.in_flight_actions = {
            0: {
                "query": "Test",
                "handle": mock_handle,
                "handle_actions": [],
            },
        }

        tool = brain_action_tools._make_steering_tool(
            handle_id=0,
            handle=mock_handle,
            operation="resume",
            param_name="",
            docstring="Resume the action",
            query="Test",
        )
        result = await tool()
        mock_handle.resume.assert_called_once()
        assert result["operation"] == "resume"

    @pytest.mark.asyncio
    async def test_answer_clarification_calls_handle_method(
        self,
        brain_action_tools,
        mock_cm,
    ):
        """Answer clarification calls handle.answer_clarification."""
        mock_handle = MagicMock()
        mock_handle.answer_clarification = AsyncMock()

        mock_cm.in_flight_actions = {
            0: {
                "query": "Test",
                "handle": mock_handle,
                "handle_actions": [],
            },
        }

        tool = brain_action_tools._make_steering_tool(
            handle_id=0,
            handle=mock_handle,
            operation="answer_clarification",
            param_name="answer",
            docstring="Answer clarification",
            query="Test",
            call_id="call_123",
        )
        result = await tool(answer="Here is the answer")
        mock_handle.answer_clarification.assert_called_once_with(
            "call_123",
            "Here is the answer",
        )
        assert result["operation"] == "answer_clarification"

    @pytest.mark.asyncio
    async def test_records_intervention_in_handle_actions(
        self,
        brain_action_tools,
        mock_cm,
    ):
        """Steering operations record intervention in handle_actions."""
        mock_handle = MagicMock()
        mock_handle.pause = AsyncMock()

        mock_cm.in_flight_actions = {
            0: {
                "query": "Test",
                "handle": mock_handle,
                "handle_actions": [],
            },
        }

        tool = brain_action_tools._make_steering_tool(
            handle_id=0,
            handle=mock_handle,
            operation="pause",
            param_name="",
            docstring="Pause",
            query="Test",
        )
        await tool()

        actions = mock_cm.in_flight_actions[0]["handle_actions"]
        assert len(actions) == 1
        assert actions[0]["action_name"] == "pause_0"

    @pytest.mark.asyncio
    async def test_handles_operation_errors(self, brain_action_tools, mock_cm):
        """Handles errors in steering operations gracefully."""
        mock_handle = MagicMock()
        mock_handle.pause = AsyncMock(side_effect=RuntimeError("Test error"))

        mock_cm.in_flight_actions = {
            0: {
                "query": "Test",
                "handle": mock_handle,
                "handle_actions": [],
            },
        }

        tool = brain_action_tools._make_steering_tool(
            handle_id=0,
            handle=mock_handle,
            operation="pause",
            param_name="",
            docstring="Pause",
            query="Test",
        )
        result = await tool()
        assert "Error" in result["result"]


# =============================================================================
# Tool Docstring Quality Tests
# =============================================================================


class TestToolDocstrings:
    """Tests verifying tool docstrings are informative for LLM usage."""

    def test_brain_tools_have_docstrings(self, brain_tools):
        """All brain tools have docstrings."""
        tools = brain_tools.as_tools()
        for name, fn in tools.items():
            assert fn.__doc__ is not None, f"{name} missing docstring"
            assert len(fn.__doc__) > 10, f"{name} docstring too short"

    def test_action_tools_have_docstrings(self, brain_action_tools):
        """All action tools have docstrings."""
        tools = brain_action_tools.as_tools()
        for name, fn in tools.items():
            assert fn.__doc__ is not None, f"{name} missing docstring"
            assert len(fn.__doc__) > 10, f"{name} docstring too short"

    def test_send_email_docstring_mentions_parameters(self, brain_action_tools):
        """send_email docstring mentions recipients, subject and body."""
        doc = brain_action_tools.send_email.__doc__
        assert "to" in doc.lower()
        assert "cc" in doc.lower()
        assert "subject" in doc.lower()
        assert "body" in doc.lower()

    def test_act_docstring_is_comprehensive(self, brain_action_tools):
        """act tool has comprehensive docstring explaining capabilities."""
        doc = brain_action_tools.act.__doc__
        assert len(doc) > 100, "act docstring should be comprehensive"


# =============================================================================
# Integration Tests
# =============================================================================


class TestCompletedActionTools:
    """Tests for completed action tools (ask and close)."""

    def test_build_completed_action_tools_includes_ask_and_close(
        self,
        brain_action_tools,
        mock_cm,
    ):
        """build_completed_action_tools generates both ask_* and close_* tools."""
        mock_cm.completed_actions = {
            0: {
                "query": "Find contacts",
                "handle": MagicMock(),
                "handle_actions": [],
            },
        }

        tools = brain_action_tools.build_completed_action_tools()
        tool_names = list(tools.keys())

        ask_tools = [n for n in tool_names if n.startswith("ask_")]
        assert len(ask_tools) == 1, f"Expected 1 ask tool, got {ask_tools}"
        assert not any(
            n.startswith("close_") for n in tool_names
        ), f"close_* should not appear for completed actions: {tool_names}"

    def test_no_completed_actions_yields_no_tools(
        self,
        brain_action_tools,
        mock_cm,
    ):
        """Empty completed_actions yields no tools."""
        mock_cm.completed_actions = {}
        tools = brain_action_tools.build_completed_action_tools()
        assert len(tools) == 0

    def test_multiple_completed_actions_yield_tools_for_each(
        self,
        brain_action_tools,
        mock_cm,
    ):
        """Each completed action gets its own ask_* tool."""
        mock_cm.completed_actions = {
            0: {
                "query": "Find contacts",
                "handle": MagicMock(),
                "handle_actions": [],
            },
            1: {
                "query": "Create a task",
                "handle": MagicMock(),
                "handle_actions": [],
            },
        }

        tools = brain_action_tools.build_completed_action_tools()
        ask_tools = [n for n in tools if n.startswith("ask_")]
        assert len(ask_tools) == 2
        assert not any(n.startswith("close_") for n in tools)


class TestBrainToolsIntegration:
    """Integration tests for brain tools working together."""

    def test_brain_and_action_tools_have_distinct_names(
        self,
        brain_tools,
        brain_action_tools,
    ):
        """Brain tools and action tools have non-overlapping names."""
        brain_names = set(brain_tools.as_tools().keys())
        action_names = set(brain_action_tools.as_tools().keys())
        overlap = brain_names & action_names
        assert len(overlap) == 0, f"Overlapping tool names: {overlap}"

    def test_steering_tools_distinct_from_static_tools(
        self,
        brain_action_tools,
        mock_cm,
    ):
        """Dynamic steering tools don't overlap with static action tools."""
        mock_cm.in_flight_actions = {
            0: {
                "query": "Test",
                "handle": MagicMock(),
                "handle_actions": [],
            },
        }
        static_names = set(brain_action_tools.as_tools().keys())
        steering_names = set(brain_action_tools.build_action_steering_tools().keys())
        overlap = static_names & steering_names
        assert len(overlap) == 0, f"Overlapping tool names: {overlap}"


class TestOnboardingToggleTools:
    """Mutually exclusive activate/deactivate onboarding slow-brain tools."""

    def _action_tools(self, mock_cm):
        with patch(
            "unify.conversation_manager.domains.brain_action_tools.get_event_broker",
        ) as mock_broker:
            mock_broker.return_value = MagicMock()
            mock_broker.return_value.publish = AsyncMock()
            return ConversationManagerBrainActionTools(mock_cm)

    def test_deactivate_exposed_when_onboarding_active(
        self,
        mock_cm,
        coordinator_session,
    ):
        mock_cm.coordinator_onboarding_active = True
        with patch("unify.settings.SETTINGS") as settings:
            settings.UNITY_CONSOLE_UI = True
            settings.DEMO_MODE = False
            tools = self._action_tools(mock_cm).as_tools()
        assert "deactivate_onboarding" in tools
        assert "set_onboarding_task_state" in tools
        assert "activate_onboarding" not in tools

    def test_activate_exposed_when_onboarding_inactive(
        self,
        mock_cm,
        coordinator_session,
    ):
        mock_cm.coordinator_onboarding_active = False
        with patch("unify.settings.SETTINGS") as settings:
            settings.UNITY_CONSOLE_UI = True
            settings.DEMO_MODE = False
            tools = self._action_tools(mock_cm).as_tools()
        assert "activate_onboarding" in tools
        assert "deactivate_onboarding" not in tools
        assert "set_onboarding_task_state" not in tools

    def test_neither_tool_without_console_ui(
        self,
        mock_cm,
        coordinator_session,
    ):
        mock_cm.coordinator_onboarding_active = True
        with patch("unify.settings.SETTINGS") as settings:
            settings.UNITY_CONSOLE_UI = False
            settings.DEMO_MODE = False
            tools = self._action_tools(mock_cm).as_tools()
        assert "deactivate_onboarding" not in tools
        assert "set_onboarding_task_state" not in tools
        assert "activate_onboarding" not in tools

    def test_neither_tool_for_non_coordinator(self, mock_cm):
        mock_cm.coordinator_onboarding_active = True
        previous = SESSION_DETAILS.is_coordinator
        SESSION_DETAILS.is_coordinator = False
        try:
            with patch("unify.settings.SETTINGS") as settings:
                settings.UNITY_CONSOLE_UI = True
                settings.DEMO_MODE = False
                tools = self._action_tools(mock_cm).as_tools()
        finally:
            SESSION_DETAILS.is_coordinator = previous
        assert "deactivate_onboarding" not in tools
        assert "set_onboarding_task_state" not in tools
        assert "activate_onboarding" not in tools

    @pytest.mark.asyncio
    async def test_set_onboarding_task_state_calls_patch_helper(self, mock_cm):
        mock_cm._patch_coordinator_onboarding_step_state = AsyncMock(
            return_value={"status": "ok", "message": "done"},
        )
        tools = ConversationManagerBrainActionTools(mock_cm)
        result = await tools.set_onboarding_task_state("apps", True)
        mock_cm._patch_coordinator_onboarding_step_state.assert_awaited_once_with(
            step_id="apps",
            completed=True,
        )
        assert result["status"] == "ok"

    @pytest.mark.asyncio
    async def test_set_onboarding_task_state_surfaces_patch_error(self, mock_cm):
        mock_cm._patch_coordinator_onboarding_step_state = AsyncMock(
            return_value={
                "status": "error",
                "message": "Communication checklist steps complete automatically",
            },
        )
        tools = ConversationManagerBrainActionTools(mock_cm)
        result = await tools.set_onboarding_task_state("email-reference", True)
        assert result["status"] == "error"
        assert "Communication" in result["message"]

    @pytest.mark.asyncio
    async def test_deactivate_calls_patch_helper(self, mock_cm):
        mock_cm._patch_coordinator_onboarding_active = AsyncMock(
            return_value={"status": "ok", "message": "paused"},
        )
        tools = ConversationManagerBrainActionTools(mock_cm)
        result = await tools.deactivate_onboarding()
        mock_cm._patch_coordinator_onboarding_active.assert_awaited_once_with(
            active=False,
            clear_onboarding_step=True,
        )
        assert result["status"] == "ok"

    @pytest.mark.asyncio
    async def test_activate_calls_patch_helper(self, mock_cm):
        mock_cm._patch_coordinator_onboarding_active = AsyncMock(
            return_value={"status": "ok", "message": "live"},
        )
        tools = ConversationManagerBrainActionTools(mock_cm)
        result = await tools.activate_onboarding()
        mock_cm._patch_coordinator_onboarding_active.assert_awaited_once_with(
            active=True,
        )
        assert result["status"] == "ok"
