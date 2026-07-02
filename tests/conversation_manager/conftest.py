"""
tests/conversation_manager/conftest.py
==============================================

Fixtures for conversation manager integration tests.

Uses **direct handler testing** pattern (same as ContactManager tests):
- No event-driven initialization (no background task dependencies)
- Direct calls to event handlers via CMStepDriver
- Direct state inspection
- Works reliably with pytest-asyncio

These tests use **simulated** state managers (SimulatedActor,
SimulatedContactManager, SimulatedTranscriptManager) to avoid real LLM
calls and computer environment dependencies. The brain tools route
directly to ContactManager/TranscriptManager (ask_about_contacts,
update_contacts, query_past_transcripts), so those must also be simulated.

Parallel execution is coordinated using scenario_file_lock (same pattern as
ContactManager tests) to prevent race conditions when multiple test processes
try to create system contacts simultaneously.
"""

from __future__ import annotations

import os

# ─────────────────────────────────────────────────────────────────────────────
# CRITICAL: environment-variable setup MUST happen at module top, *before*
# any `tests.helpers` import (which transitively imports unity modules that
# instantiate `SETTINGS = ProductionSettings()` at import time).
# Pydantic's BaseSettings reads env vars once at instantiation; if SETTINGS
# is already constructed when pytest_configure() later sets these vars, the
# overrides are silently ignored and prompts/feature flags fall back to
# production defaults. Tests like test_email_to_email then fail because the
# system prompt branches on the wrong value (e.g. send_email tool gated by
# empty ASSISTANT_EMAIL).
#
# pytest_configure() below still reassigns these defensively in case a
# downstream test reimports SETTINGS, but the authoritative point of
# truth is *here*, before any unity import in this conftest.
# ─────────────────────────────────────────────────────────────────────────────
os.environ.setdefault("ASSISTANT_EMAIL", "assistant@test.example.com")
os.environ.setdefault("ASSISTANT_NUMBER", "+15550001000")
os.environ.setdefault("ASSISTANT_WHATSAPP_NUMBER", "+15550001000")
os.environ.setdefault("UNITY_CONVERSATION_JOB_NAME", "test_job")

from unittest.mock import patch

import pytest
import pytest_asyncio

from tests.helpers import scenario_file_lock, get_or_create_contact
from .cm_test_driver import CMStepDriver

# Response policies matching ContactManager defaults
# BOSS_RESPONSE_POLICY is for the user (contact_id 1) who gives commands
# DEFAULT_RESPONSE_POLICY is for regular contacts who should NOT give commands
# HELPFUL_RESPONSE_POLICY is for tests where contacts can make requests freely
BOSS_RESPONSE_POLICY = (
    "Your immediate manager, please do whatever they ask you to do within reason, "
    "and do *not* withhold any information from them."
)
DEFAULT_RESPONSE_POLICY = (
    "Please engage politely, helpfully, and respectfully, but you do not need to "
    "take orders from them. Please also do not share **any** sensitive or personal "
    "information with them about any other person, company or policy at all."
)
HELPFUL_RESPONSE_POLICY = (
    "Please engage politely, helpfully, and respectfully. Fulfil any reasonable "
    "requests they make, including requests to communicate via specific channels "
    "(SMS, email, phone call, etc.). Do not share sensitive information about "
    "other people, but otherwise be helpful and accommodating."
)

# System contacts (contact_id 0 and 1) are created by ContactManager from the database.
# BOSS represents the boss user (contact_id 1) who gives commands to the assistant.
# Tests should use BOSS when simulating commands from the user.
BOSS = {
    "contact_id": 1,
    "first_name": "Default",
    "surname": "User",
    "email_address": "user@example.com",
    "phone_number": "+15555551111",
    "should_respond": True,
    "is_system": True,
    "response_policy": BOSS_RESPONSE_POLICY,
}

# Test contacts used across all tests (starting at contact_id 2)
# should_respond=True allows outbound communication in tests
# response_policy matches ContactManager.DEFAULT_RESPONSE_POLICY
TEST_CONTACTS = [
    {
        "contact_id": 2,
        "first_name": "Alice",
        "surname": "Smith",
        "email_address": "alice@example.com",
        "phone_number": "+15555552222",
        "should_respond": True,
        "response_policy": DEFAULT_RESPONSE_POLICY,
    },
    {
        "contact_id": 3,
        "first_name": "Bob",
        "surname": "Johnson",
        "email_address": "bob@example.com",
        "phone_number": "+15555553333",
        "should_respond": True,
        "response_policy": DEFAULT_RESPONSE_POLICY,
    },
    {
        "contact_id": 4,
        "first_name": "Charlie",
        "surname": "Davis",
        "email_address": "charlie@example.com",
        "phone_number": "+15555554444",
        "should_respond": True,
        "response_policy": DEFAULT_RESPONSE_POLICY,
    },
    {
        "contact_id": 5,
        "first_name": "Diana",
        "surname": "Evans",
        "email_address": "diana@example.com",
        "phone_number": "+15555555555",
        "should_respond": True,
        "response_policy": DEFAULT_RESPONSE_POLICY,
    },
]


# =============================================================================
# Universal comms isolation
# =============================================================================

_COMMS_MODULE = "unify.conversation_manager.domains.comms_utils"


@pytest.fixture(autouse=True)
def _stub_outbound_comms(request):
    """Prevent real HTTP calls to the communication service.

    Patches the four outbound ``comms_utils`` functions that make HTTP/Pub-Sub
    calls to the communication service (SMS, email, phone calls, unify messages).
    Every test in ``tests/conversation_manager/`` gets this automatically.

    Tests that exercise the real ``comms_utils`` implementations (with their own
    mocked aiohttp / SESSION_DETAILS) opt out via ``@pytest.mark.real_comms_functions``.
    """
    if "real_comms_functions" in request.keywords:
        yield
        return

    async def _success(*args, **kwargs):
        return {"success": True}

    with (
        patch(f"{_COMMS_MODULE}.send_sms_message_via_number", _success),
        patch(f"{_COMMS_MODULE}.send_unify_message", _success),
        patch(f"{_COMMS_MODULE}.send_email_via_address", _success),
        patch(f"{_COMMS_MODULE}.start_call", _success),
        patch(f"{_COMMS_MODULE}.complete_api_message", _success),
    ):
        yield


# =============================================================================
# Module-level setup: Configure environment for in-process mode
# =============================================================================


def pytest_configure(config):
    """Configure environment variables before any tests run."""
    # Actor, ContactManager, and TranscriptManager are all simulated.
    # The brain tools route to these managers directly (ask_about_contacts,
    # update_contacts, query_past_transcripts), so they must be simulated
    # to avoid real LLM calls.
    os.environ["UNITY_ACTOR_IMPL"] = "simulated"
    os.environ["UNITY_ACTOR_SIMULATED_STEPS"] = "0"  # Allows pause+resume interactions
    os.environ.setdefault("UNITY_CONTACT_IMPL", "simulated")
    os.environ.setdefault("UNITY_TRANSCRIPT_IMPL", "simulated")

    # Disable optional managers not needed for conversation manager tests
    os.environ["UNITY_MEMORY_ENABLED"] = "false"
    os.environ["UNITY_KNOWLEDGE_ENABLED"] = "false"
    os.environ["UNITY_GUIDANCE_ENABLED"] = "false"
    os.environ["UNITY_SECRET_ENABLED"] = "false"
    os.environ["UNITY_SKILL_ENABLED"] = "false"
    os.environ["UNITY_WEB_ENABLED"] = "false"
    os.environ["UNITY_FILE_ENABLED"] = "false"

    # Enable incrementing timestamps for **NEW** marker comparisons
    os.environ["UNITY_INCREMENTING_TIMESTAMPS"] = "true"

    # Mark as test mode
    os.environ["TEST"] = "true"
    os.environ["UNITY_CONVERSATION_JOB_NAME"] = "test_job"

    # Configure the assistant identity for flows tests.
    #
    # unify/conversation_manager/prompt_builders.py:_build_comms_tool_listing
    # exposes `send_email` / `send_sms` / `send_whatsapp` to the LLM
    # ONLY when the corresponding assistant.{email,number,whatsapp_number}
    # is non-empty (gating added 2026-03-03 in d71f8dc9d0). With
    # SESSION_DETAILS.assistant.email / .number defaulting to "", flows
    # tests like test_email_to_email / test_sms_to_sms ended up with
    # only `send_unify_message` exposed — the LLM correctly chose the
    # only available tool, producing UnifyMessageSent instead of the
    # expected EmailSent/SMSSent and breaking the test's
    # assert_has_one(EmailSent) / SMSSent checks.
    #
    # Populate via env vars so SessionDetails.populate_from_env() (called
    # by the CM process under `apply_test_mocks=True`) picks them up.
    # Mirror the email_address / phone_number values that TEST_CONTACTS
    # use for the user so the flow looks coherent in any prompt
    # rendering. The provider stays "google_workspace" (the default)
    # because the gate only checks email non-emptiness.
    os.environ.setdefault("ASSISTANT_EMAIL", "assistant@test.example.com")
    os.environ.setdefault("ASSISTANT_NUMBER", "+15550001000")
    os.environ.setdefault("ASSISTANT_WHATSAPP_NUMBER", "+15550001000")


# =============================================================================
# ConversationManager Fixtures (Direct Handler Testing)
# =============================================================================


@pytest_asyncio.fixture(scope="module")
async def conversation_manager(request) -> CMStepDriver:
    """
    Start and initialize ConversationManager in-process for the test module.

    Uses DIRECT initialization (not event-driven) to avoid background task
    issues with pytest-asyncio. This follows the same pattern as ContactManager
    tests - direct method calls, not event publishing.

    Uses SimulatedActor explicitly for fast, deterministic testing without
    computer environment dependencies.

    Uses scenario_file_lock to coordinate initialization across parallel test
    processes, preventing race conditions when ContactManager creates system
    contacts (id=0, id=1).

    Returns a CMStepDriver that wraps the CM and provides step() and
    step_until_wait() methods for deterministic testing.
    """
    from unify.actor.simulated import SimulatedActor
    from unify.conversation_manager.event_broker import reset_event_broker
    from unify.conversation_manager import start_async, stop_async
    from unify.conversation_manager.domains import managers_utils

    # Reset any existing event broker state
    reset_event_broker()

    print("\n✓ Starting ConversationManager in-process...")
    cm = await start_async(
        project_name="TestProject",
        enable_comms_manager=False,  # Don't start CommsManager (requires GCP)
        apply_test_mocks=True,
    )
    print("✓ ConversationManager started (in-process mode)")
    print("  Using SimulatedActor for deterministic testing")

    # Create SimulatedActor for fast, deterministic testing
    # (avoids full actor computer environment setup)
    #
    # Uses steps=None, duration=None so actions run indefinitely until explicitly
    # completed via trigger_completion() in test cleanup. This makes tests fully
    # deterministic with no timing dependencies. Tests that verify steering (pause,
    # resume, stop, interject) just check that steering tools were called - they
    # don't need actions to auto-complete based on step counts.
    actor = SimulatedActor(
        steps=None,
        duration=None,
        log_mode="log",
        emit_notifications=False,
    )

    # Use file lock to coordinate manager initialization across parallel test processes.
    # ContactManager.__init__ creates system contacts (assistant id=0, user id=1)
    # via _sync_required_contacts(). This must be serialized to prevent duplicate
    # contact creation when multiple pytest sessions start in parallel.
    with scenario_file_lock("cm_conversation_manager"):
        # Initialize managers DIRECTLY (not via event handler)
        # This avoids the background task / event loop interleaving issues
        print("⏳ Initializing managers directly...")
        await managers_utils.init_conv_manager(cm, actor=actor)
        if not cm.initialized:
            raise RuntimeError(
                "ConversationManager managers failed to initialize - check logs for errors",
            )
        print("✅ Managers initialized")

        # Update system contacts in ContactManager with proper names and test defaults.
        # In CI, the test user may have null first/last name from the API.
        # ContactManager is the source of truth - ContactIndex queries it directly.
        if cm.contact_manager is not None:
            # Update assistant (contact_id 0)
            cm.contact_manager.update_contact(
                contact_id=0,
                first_name="Default",
                surname="Assistant",
                should_respond=True,
            )
            # Update boss/user (contact_id 1) with test defaults
            cm.contact_manager.update_contact(
                contact_id=1,
                first_name=BOSS["first_name"],
                surname=BOSS["surname"],
                email_address=BOSS["email_address"],
                phone_number=BOSS["phone_number"],
                should_respond=True,
                response_policy=BOSS["response_policy"],
            )
            print("✅ System contacts updated in ContactManager")

        # Create test contacts in the database using idempotent helper.
        # This ensures they exist with the expected contact_ids even when
        # multiple test processes run in parallel.
        for contact_data in TEST_CONTACTS:
            contact_id = get_or_create_contact(
                cm.contact_manager,
                first_name=contact_data["first_name"],
                surname=contact_data.get("surname"),
                email_address=contact_data.get("email_address"),
                phone_number=contact_data.get("phone_number"),
            )
            # Update should_respond and response_policy for tests
            if contact_id and cm.contact_manager is not None:
                cm.contact_manager.update_contact(
                    contact_id=contact_id,
                    should_respond=contact_data.get("should_respond", True),
                    response_policy=contact_data.get(
                        "response_policy",
                        DEFAULT_RESPONSE_POLICY,
                    ),
                )

    print(f"✅ Test contacts created: {len(TEST_CONTACTS)} + system contacts")

    # Wrap in CMStepDriver for deterministic testing
    driver = CMStepDriver(cm)

    yield driver

    # Cleanup
    print("\n✓ Stopping ConversationManager...")
    await stop_async()
    reset_event_broker()


def _reset_screen_share_state(cm: "CMStepDriver") -> None:
    """Clear screen-share flags and buffered screenshots between tests."""

    cm.cm.user_screen_share_active = False
    cm.cm.assistant_screen_share_active = False
    cm.cm._screenshot_buffer.clear()


def _complete_in_flight_actions(cm: "CMStepDriver") -> None:
    """
    Complete all in-flight actions to unblock watcher threads.

    With steps=None, actions run indefinitely. The watcher tasks (actor_watch_result,
    actor_watch_notifications, actor_watch_clarifications) block on _done_event.wait()
    in thread pool threads. If we don't call trigger_completion(), these threads
    never terminate and cause "executor did not finish joining" warnings at shutdown.
    """
    for handle_data in list(cm.cm.in_flight_actions.values()):
        handle = handle_data.get("handle")
        if handle and hasattr(handle, "trigger_completion"):
            handle.trigger_completion()
    cm.cm.in_flight_actions.clear()
    cm.cm.completed_actions.clear()


@pytest.fixture
def initialized_cm(
    conversation_manager: CMStepDriver,
):
    """
    Per-test fixture that provides a clean ConversationManager.

    Clears conversation state between tests for isolation while reusing
    the expensive module-scoped CM instance. Also ensures in-flight actions
    are completed after each test to prevent thread leaks.
    """
    # Complete and clear in-flight actions from previous tests
    _complete_in_flight_actions(conversation_manager)

    # Clear any conversation state from previous tests
    conversation_manager.contact_index.clear_conversations()

    # Reset handle_id counter to ensure deterministic tool names for caching.
    # Without this, handle_ids increment across tests, causing tool names like
    # pause_search_...__0 vs pause_search_...__1, which breaks LLM cache hits.
    import unify.conversation_manager.domains.brain_action_tools as bat

    bat._next_handle_id = 0

    # Clear chat history (LLM message history)
    conversation_manager.cm.chat_history.clear()

    # Screen-share tests attach screenshots that must not leak into text-only turns.
    _reset_screen_share_state(conversation_manager)

    # Clear tool call tracking from previous tests
    conversation_manager.all_tool_calls.clear()

    # Reset last_snapshot to use the patched prompt_now.
    # The module-scoped conversation_manager fixture is created BEFORE the
    # function-scoped stub_external_deps fixture patches prompt_now, so
    # cm.last_snapshot gets set to real time (e.g., January 2026) while
    # message timestamps use the patched fixed time (June 2025).
    # This breaks the **NEW** marker comparison (last_snapshot < message.timestamp).
    # Re-initializing here ensures last_snapshot uses the patched timestamp.
    from unify.common.prompt_helpers import now as prompt_now

    conversation_manager.cm.last_snapshot = prompt_now(as_string=False)

    yield conversation_manager

    # Cleanup after test: complete any in-flight actions created during this test
    _complete_in_flight_actions(conversation_manager)
