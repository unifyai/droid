"""Tests for email -> Slack user-ID resolution on ``send_slack_message``.

An org-wide Slack install means the assistant may need to DM a workspace
member it has never received a message from, so no ``slack_user_id`` is on
file. When the contact has an ``email_address``, the send path resolves the
Slack user ID via ``users.lookupByEmail`` (through the Communication
gateway), persists it onto the contact, and proceeds. The lookup only fires
when it's actually needed -- an existing ID, an inline ID, or a contact with
no email all skip it -- and an unresolved email surfaces the normal
"no identifier on file" error rather than a silent failure.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from unify.contact_manager.simulated import SimulatedContactManager
from unify.conversation_manager.domains.brain_action_tools import (
    ConversationManagerBrainActionTools,
)
from unify.conversation_manager.domains.contact_index import ContactIndex
from unify.conversation_manager.domains.notifications import NotificationBar

# =============================================================================
# Fixtures
# =============================================================================


def _setup_mock_contacts(
    contact_index: ContactIndex,
    contacts: list[dict],
) -> SimulatedContactManager:
    """Populate a SimulatedContactManager and wire it into the ContactIndex."""
    contact_manager = SimulatedContactManager()
    for contact_data in contacts:
        contact_manager.update_contact(
            contact_id=contact_data["contact_id"],
            first_name=contact_data.get("first_name"),
            surname=contact_data.get("surname"),
            email_address=contact_data.get("email_address"),
            slack_user_id=contact_data.get("slack_user_id"),
            should_respond=contact_data.get("should_respond", True),
        )
    contact_index.set_contact_manager(contact_manager)
    return contact_manager


@pytest.fixture
def mock_cm():
    cm = MagicMock()
    cm.mode = "text"
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
    cm.assistant_slack_bot_user_id = "UBOT"
    cm.call_manager.has_active_call = False
    cm.call_manager.has_active_google_meet = False
    cm.call_manager.has_active_teams_meet = False
    cm.call_manager.has_gmeet_presenting = False
    cm.call_manager.has_teams_presenting = False
    cm.call_manager._meet_joining = False
    cm.call_manager._whatsapp_call_joining = False
    cm.assistant_has_teams = False
    cm.contact_manager = _setup_mock_contacts(cm.contact_index, [])
    return cm


@pytest.fixture
def brain_action_tools(mock_cm):
    with patch(
        "unify.conversation_manager.domains.brain_action_tools.get_event_broker",
    ) as mock_broker:
        mock_broker.return_value = MagicMock()
        mock_broker.return_value.publish = AsyncMock()
        tools = ConversationManagerBrainActionTools(mock_cm)
        yield tools


_SEND_OK = {"success": True, "channel_id": "D123", "message_ts": "170.1"}


# =============================================================================
# Tests
# =============================================================================


class TestSendSlackEmailLookup:
    @pytest.mark.asyncio
    async def test_resolves_slack_id_by_email_when_missing(
        self,
        brain_action_tools,
        mock_cm,
    ):
        """No slack_user_id but an email on file -> resolve, persist, send."""
        _setup_mock_contacts(
            mock_cm.contact_index,
            [
                {
                    "contact_id": 5,
                    "first_name": "Alice",
                    "email_address": "alice@example.com",
                    "should_respond": True,
                },
            ],
        )

        with (
            patch(
                "unify.comms.primitives.comms_utils.resolve_slack_user_id_by_email",
                new=AsyncMock(return_value="U123"),
            ) as mock_resolve,
            patch(
                "unify.comms.primitives.comms_utils.send_slack_message",
                new=AsyncMock(return_value=_SEND_OK),
            ) as mock_send,
        ):
            result = await brain_action_tools.send_slack_message(
                contact_id=5,
                content="Hello",
                team_id="T1",
            )

        assert result["status"] == "ok"
        mock_resolve.assert_awaited_once()
        assert mock_resolve.await_args.kwargs == {
            "team_id": "T1",
            "email": "alice@example.com",
        }
        # The resolved ID is persisted and used as the send recipient.
        assert mock_cm.contact_index.get_contact(5)["slack_user_id"] == "U123"
        assert mock_send.await_args.kwargs["user_id"] == "U123"

    @pytest.mark.asyncio
    async def test_existing_slack_id_skips_lookup(
        self,
        brain_action_tools,
        mock_cm,
    ):
        """A slack_user_id already on file -> no email lookup."""
        _setup_mock_contacts(
            mock_cm.contact_index,
            [
                {
                    "contact_id": 5,
                    "first_name": "Alice",
                    "email_address": "alice@example.com",
                    "slack_user_id": "UEXISTING",
                    "should_respond": True,
                },
            ],
        )

        with (
            patch(
                "unify.comms.primitives.comms_utils.resolve_slack_user_id_by_email",
                new=AsyncMock(return_value="U123"),
            ) as mock_resolve,
            patch(
                "unify.comms.primitives.comms_utils.send_slack_message",
                new=AsyncMock(return_value=_SEND_OK),
            ) as mock_send,
        ):
            result = await brain_action_tools.send_slack_message(
                contact_id=5,
                content="Hello",
                team_id="T1",
            )

        assert result["status"] == "ok"
        mock_resolve.assert_not_awaited()
        assert mock_send.await_args.kwargs["user_id"] == "UEXISTING"

    @pytest.mark.asyncio
    async def test_inline_id_skips_lookup_and_persists(
        self,
        brain_action_tools,
        mock_cm,
    ):
        """An inline slack_user_id -> no email lookup, and it is persisted."""
        _setup_mock_contacts(
            mock_cm.contact_index,
            [
                {
                    "contact_id": 5,
                    "first_name": "Alice",
                    "email_address": "alice@example.com",
                    "should_respond": True,
                },
            ],
        )

        with (
            patch(
                "unify.comms.primitives.comms_utils.resolve_slack_user_id_by_email",
                new=AsyncMock(return_value="U123"),
            ) as mock_resolve,
            patch(
                "unify.comms.primitives.comms_utils.send_slack_message",
                new=AsyncMock(return_value=_SEND_OK),
            ),
        ):
            result = await brain_action_tools.send_slack_message(
                contact_id=5,
                content="Hello",
                team_id="T1",
                slack_user_id="UINLINE",
            )

        assert result["status"] == "ok"
        mock_resolve.assert_not_awaited()
        assert mock_cm.contact_index.get_contact(5)["slack_user_id"] == "UINLINE"

    @pytest.mark.asyncio
    async def test_no_email_no_id_errors_without_lookup(
        self,
        brain_action_tools,
        mock_cm,
    ):
        """Neither email nor slack_user_id -> error, and no lookup attempted."""
        _setup_mock_contacts(
            mock_cm.contact_index,
            [
                {
                    "contact_id": 5,
                    "first_name": "Alice",
                    "should_respond": True,
                },
            ],
        )

        with (
            patch(
                "unify.comms.primitives.comms_utils.resolve_slack_user_id_by_email",
                new=AsyncMock(return_value="U123"),
            ) as mock_resolve,
            patch(
                "unify.comms.primitives.comms_utils.send_slack_message",
                new=AsyncMock(return_value=_SEND_OK),
            ) as mock_send,
        ):
            result = await brain_action_tools.send_slack_message(
                contact_id=5,
                content="Hello",
                team_id="T1",
            )

        assert result["status"] == "error"
        mock_resolve.assert_not_awaited()
        mock_send.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_unresolved_email_errors(
        self,
        brain_action_tools,
        mock_cm,
    ):
        """Email present but lookup returns None -> normal missing-ID error."""
        _setup_mock_contacts(
            mock_cm.contact_index,
            [
                {
                    "contact_id": 5,
                    "first_name": "Alice",
                    "email_address": "ghost@example.com",
                    "should_respond": True,
                },
            ],
        )

        with (
            patch(
                "unify.comms.primitives.comms_utils.resolve_slack_user_id_by_email",
                new=AsyncMock(return_value=None),
            ) as mock_resolve,
            patch(
                "unify.comms.primitives.comms_utils.send_slack_message",
                new=AsyncMock(return_value=_SEND_OK),
            ) as mock_send,
        ):
            result = await brain_action_tools.send_slack_message(
                contact_id=5,
                content="Hello",
                team_id="T1",
            )

        assert result["status"] == "error"
        mock_resolve.assert_awaited_once()
        mock_send.assert_not_awaited()
