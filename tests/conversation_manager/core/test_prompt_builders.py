"""
tests/conversation_manager/core/test_prompt_builders.py
=======================================================

Unit tests for the ConversationManager prompt builder, focusing on
capability-gated sections (assistant phone / email).
"""

from __future__ import annotations

import pytest

from unify.conversation_manager.prompt_builders import (
    build_system_prompt,
    build_voice_agent_prompt,
)
from unify.session_details import TeamSummary

pytestmark = pytest.mark.no_unify_context

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BASE_KWARGS: dict = {
    "bio": "A helpful assistant.",
    "contact_id": 1,
    "first_name": "Alice",
    "surname": "Smith",
}


def _build(**overrides: object) -> str:
    """Build a system prompt with sensible defaults, returning flat text."""
    kwargs = {**_BASE_KWARGS, **overrides}
    return build_system_prompt(**kwargs).flatten()


def _build_voice(**overrides: object) -> str:
    """Build a voice-agent prompt with stable defaults."""
    kwargs = {
        "bio": "I help Acme configure its Unify team.",
        "assistant_name": "Avery",
        "boss_first_name": "Dana",
        "boss_surname": "Owner",
        **overrides,
    }
    return build_voice_agent_prompt(**kwargs).flatten()


# ---------------------------------------------------------------------------
# Tests – tool listing
# ---------------------------------------------------------------------------


class TestCommsToolListing:
    """The output-format section lists only the comms tools the assistant can use."""

    def test_all_tools_listed_when_fully_configured(self):
        prompt = _build(assistant_has_phone=True, assistant_has_email=True)
        assert "`send_sms`" in prompt
        assert "`send_email`" in prompt
        assert "`make_call`" in prompt
        assert "`send_unify_message`" in prompt

    def test_phone_tools_absent_without_phone(self):
        prompt = _build(assistant_has_phone=False, assistant_has_email=True)
        assert "`send_sms`: Send an SMS" not in prompt
        assert "`make_call`: Start an outbound" not in prompt
        assert "`send_email`" in prompt
        assert "`send_unify_message`" in prompt

    def test_email_tool_absent_without_email(self):
        prompt = _build(assistant_has_phone=True, assistant_has_email=False)
        assert "`send_email`: Send an email" not in prompt
        assert "`send_sms`" in prompt
        assert "`make_call`" in prompt
        assert "`send_unify_message`" in prompt

    def test_only_unify_message_without_any_capabilities(self):
        prompt = _build(assistant_has_phone=False, assistant_has_email=False)
        assert "`send_sms`: Send an SMS" not in prompt
        assert "`send_email`: Send an email" not in prompt
        assert "`make_call`: Start an outbound" not in prompt
        assert "`send_unify_message`" in prompt

    def test_coordinator_comms_listing_is_boss_only(self):
        prompt = _build(
            is_coordinator=True,
            assistant_has_phone=True,
            assistant_has_email=True,
            assistant_has_whatsapp=True,
            assistant_has_discord=True,
            assistant_has_slack=True,
            assistant_has_teams=True,
        )

        assert "`send_sms`: Send an SMS message to my boss only" in prompt
        assert "`send_email`: Send an email to my boss only" in prompt
        assert (
            "`send_unify_message`: Send a Unify platform message to my boss only"
            in prompt
        )
        assert "`send_slack_message`: Send a Slack DM to my boss only" in prompt
        assert (
            "`send_teams_message`: Send a Teams direct message to my boss only"
            in prompt
        )
        assert (
            "`create_teams_meet`: Create a Microsoft Teams meeting with my boss only"
            in prompt
        )
        assert "`send_slack_channel_message`" not in prompt
        assert "`create_teams_channel`" not in prompt

    def test_regular_comms_listing_keeps_contact_targeting(self):
        prompt = _build(
            is_coordinator=False,
            assistant_has_phone=True,
            assistant_has_email=True,
            assistant_has_slack=True,
            assistant_has_teams=True,
        )

        assert "`send_sms`: Send an SMS message to a contact" in prompt
        assert "`send_email`: Send an email to a contact" in prompt
        assert "`send_slack_channel_message`: Post into a Slack channel" in prompt
        assert "`create_teams_channel`: Create a new channel" in prompt

    def test_discord_channel_message_advertised_for_regular_assistant(self):
        prompt = _build(is_coordinator=False, assistant_has_discord=True)
        assert "`send_discord_message`: Send a Discord message to a contact" in prompt
        assert "`send_discord_channel_message`: Post into a Discord channel" in prompt

    def test_discord_channel_message_absent_for_coordinator(self):
        prompt = _build(is_coordinator=True, assistant_has_discord=True)
        assert "`send_discord_message`: Send a Discord direct message" in prompt
        assert "`send_discord_channel_message`" not in prompt


class TestVoiceSessionToolMasking:
    """While a voice session is live, call-starting tools are neither in the
    live tool set nor advertised, and a block explains they return on hang-up."""

    _CALL_TOOL_DESCRIPTIONS = (
        "Start an outbound phone call",
        "Start a WhatsApp voice call",
        "Join a Google Meet call via browser automation",
        "Join a Microsoft Teams meeting via browser automation",
    )

    def test_call_starting_tools_advertised_off_call(self):
        prompt = _build(
            assistant_has_phone=True,
            assistant_has_whatsapp=True,
            on_voice_call=False,
        )
        for desc in self._CALL_TOOL_DESCRIPTIONS:
            assert desc in prompt
        assert "Active voice session\n--------------------" not in prompt

    def test_call_starting_tools_withheld_on_call(self):
        prompt = _build(
            assistant_has_phone=True,
            assistant_has_whatsapp=True,
            is_voice_call=True,
            on_voice_call=True,
        )
        for desc in self._CALL_TOOL_DESCRIPTIONS:
            assert desc not in prompt
        # Text channels stay available mid-call.
        assert "`send_sms`: Send an SMS message to a contact" in prompt
        assert "`send_whatsapp`: Send a WhatsApp message to a contact" in prompt

    def test_active_voice_session_block_explains_return_on_hangup(self):
        prompt = _build(
            assistant_has_phone=True,
            assistant_has_whatsapp=True,
            is_voice_call=True,
            on_voice_call=True,
        )
        assert "Active voice session\n--------------------" in prompt
        assert "only be on ONE voice session at a time" in prompt
        assert "reappear automatically the moment this session ends" in prompt
        assert "`hang_up`" in prompt

    def test_coordinator_call_tools_withheld_on_call(self):
        prompt = _build(
            is_coordinator=True,
            assistant_has_phone=True,
            assistant_has_whatsapp=True,
            is_voice_call=True,
            on_voice_call=True,
        )
        assert "`make_call`: Start an outbound phone call to my boss only" not in prompt
        assert (
            "`make_whatsapp_call`: Start a WhatsApp voice call to my boss only"
            not in prompt
        )
        assert "Active voice session\n--------------------" in prompt

    def test_one_voice_session_rule_present_without_phone(self):
        # The mutual-exclusion rule is no longer gated on a stored phone number.
        prompt = _build(assistant_has_phone=False, assistant_has_whatsapp=False)
        assert "only be on ONE voice session at a time" in prompt


class TestAccessibleSpacesBlock:
    """The system prompt contains shared-team routing guidance."""

    def test_block_renders_after_bio(self):
        prompt = _build(
            bio="Assistant biography.",
            team_summaries=[
                TeamSummary(
                    team_id=3,
                    name="Repairs",
                    description="South-East repairs patch daily operations.",
                ),
            ],
        )

        assert "Bio\n---\nAssistant biography." in prompt
        assert "Accessible shared teams" in prompt
        assert (
            '- team:3 "Repairs" - South-East repairs patch daily operations.' in prompt
        )
        assert prompt.index("Bio\n---") < prompt.index("Accessible shared teams")
        assert prompt.index("Accessible shared teams") < prompt.index(
            "Onboarding reference",
        )


class TestCoordinatorPrompt:
    """Coordinator sessions use a unified base prompt plus org-context surfaces."""

    def test_org_coordinator_prompt_lists_org_roster_and_admin_tools(self):
        prompt = _build(
            is_coordinator=True,
            authorized_humans=[
                {
                    "first_name": "Dana",
                    "surname": "Owner",
                    "email": "dana@acme.com",
                    "is_admin": True,
                },
                {
                    "first_name": "Francis",
                    "surname": "Lead",
                    "email": "francis@acme.com",
                    "is_admin": False,
                },
            ],
        )

        assert "Authorized humans" in prompt
        assert "Dana Owner; email: dana@acme.com; role: admin" in prompt
        assert "Francis Lead; email: francis@acme.com; role: member" in prompt
        assert "**T-W1N admin tools:**" in prompt
        assert "`primitives.coordinator.list_org_members`" in prompt
        assert "always target the active workspace organization" in prompt
        assert "T-W1N\n----" in prompt
        assert "Role / specialization: Coordinator." in prompt
        assert "My identity" in prompt
        assert "I am T-W1N, Alice Smith's personal, private assistant" in prompt

    def test_personal_coordinator_uses_boss_details_and_routes_org_work_to_switch(
        self,
    ):
        prompt = _build(is_coordinator=True, is_org_workspace=False)

        assert "Boss details" in prompt
        assert "Authorized humans\n-----------------" not in prompt
        assert "Organization membership actions are unavailable" in prompt
        assert "switch to that organization's T-W1N" in prompt
        assert "list_accessible_organizations" not in prompt

    def test_regular_assistant_gets_twin_reference_block(self):
        prompt = _build()

        assert "T-W1N identity" in prompt
        assert "T-W1N is Alice Smith's personal, private assistant" in prompt
        assert "I propose handing it to T-W1N explicitly" in prompt
        assert "inviting, removing, or changing roles for colleagues" in prompt
        assert "creating or removing teams" in prompt
        assert (
            "placing shared credentials, integrations, or other org-level setup"
            in prompt
        )
        assert "I cannot forward it automatically" not in prompt

    def test_twin_handoff_guidance_is_absent_on_twin_sessions(self):
        coordinator_prompt = _build(is_coordinator=True)

        assert "My identity" in coordinator_prompt
        assert (
            "I am T-W1N, Alice Smith's personal, private assistant"
            in coordinator_prompt
        )
        assert (
            "T-W1N is Alice Smith's personal, private assistant"
            not in coordinator_prompt
        )
        assert "My onboarding flow (UI reference)" in coordinator_prompt
        # The flow reference no longer enumerates step titles (those live in
        # the render-driven progress block); it must still speak first-person.
        assert "Give T-W1N access to your workspace" not in coordinator_prompt
        assert "I propose handing it to T-W1N explicitly" not in coordinator_prompt

    def test_base_and_coordinator_share_restraint_but_keep_role_specific_sections(self):
        base_prompt = _build()
        coordinator_prompt = _build(is_coordinator=True)

        assert "Intent vs verified outcomes" in base_prompt
        assert "Intent vs verified outcomes" in coordinator_prompt
        assert "Console knowledge" in base_prompt
        assert "Console knowledge" not in coordinator_prompt
        assert "My Console literacy" in coordinator_prompt
        assert "Concurrent action and acknowledgment" in base_prompt
        assert "Concurrent action and acknowledgment" in coordinator_prompt
        assert "Onboarding reference" in base_prompt
        assert "Onboarding reference" not in coordinator_prompt

    def test_coordinator_direct_comms_guidance_is_boss_only(self):
        prompt = _build(
            is_coordinator=True,
            assistant_has_phone=True,
            assistant_has_email=True,
            assistant_has_whatsapp=True,
            assistant_has_discord=True,
            assistant_has_slack=True,
            assistant_has_teams=True,
        )

        assert "Boss-only direct communication" in prompt
        assert "only for communicating directly with my boss" in prompt
        assert "They do not accept ``contact_id``" in prompt
        assert "always target the boss contact (``contact_id==1``" in prompt
        assert (
            "Communication with anyone else is never handled by direct tools" in prompt
        )
        assert "delegated third-party communication work goes through ``act``" in prompt
        assert (
            "send a message, draft a reply, place a call, or invite someone else on their behalf"
            in prompt
        )
        assert "Direct tools never accept inline contact details" in prompt
        assert "update the boss contact record first" in prompt
        assert "contact_id=5" not in prompt
        assert "Use the contact_id visible in active_conversations" not in prompt
        assert "send_slack_channel_message" not in prompt

    def test_regular_direct_comms_guidance_keeps_contact_id_examples(self):
        prompt = _build(
            is_coordinator=False,
            assistant_has_phone=True,
            assistant_has_email=True,
            assistant_has_teams=True,
        )

        assert "Contact-addressed communication tools" in prompt
        assert "Use the contact_id visible in active_conversations" in prompt
        assert 'send_sms(contact_id=5, content="Hi"' in prompt
        assert "Boss-only direct communication" not in prompt


class TestPromptSectionOwnershipMatrix:
    """Role/mode/org combinations keep prompt section ownership boundaries stable."""

    def test_system_prompt_section_ownership_matrix(self):
        cases = (
            {
                "name": "regular_non_demo_no_org",
                "kwargs": {},
                "present": (
                    "Act capabilities\n----------------",
                    "Concurrent action and acknowledgment\n------------------------------------",
                ),
                "absent": (
                    "**T-W1N admin tools:**",
                    "Authorized humans\n-----------------",
                    "Demo mode\n---------",
                ),
            },
            {
                "name": "regular_non_demo_with_org",
                "kwargs": {},
                "present": (
                    "Act capabilities\n----------------",
                    "Concurrent action and acknowledgment\n------------------------------------",
                    "T-W1N identity\n--------------",
                ),
                "absent": (
                    "**T-W1N admin tools:**",
                    "Authorized humans\n-----------------",
                    "Demo mode\n---------",
                ),
            },
            {
                "name": "regular_demo_no_org",
                "kwargs": {"demo_mode": True},
                "present": ("Demo mode\n---------",),
                "absent": (
                    "**T-W1N admin tools:**",
                    "Authorized humans\n-----------------",
                    "Act capabilities\n----------------",
                    "Concurrent action and acknowledgment\n------------------------------------",
                ),
            },
            {
                "name": "regular_demo_with_org",
                "kwargs": {
                    "demo_mode": True,
                },
                "present": (
                    "Demo mode\n---------",
                    "T-W1N identity\n--------------",
                ),
                "absent": (
                    "**T-W1N admin tools:**",
                    "Authorized humans\n-----------------",
                    "Act capabilities\n----------------",
                    "Concurrent action and acknowledgment\n------------------------------------",
                ),
            },
            {
                "name": "coordinator_non_demo_with_org",
                "kwargs": {
                    "is_coordinator": True,
                },
                "present": (
                    "**T-W1N admin tools:**",
                    "Authorized humans\n-----------------",
                    "Act capabilities\n----------------",
                    "Concurrent action and acknowledgment\n------------------------------------",
                    "T-W1N\n----",
                    "My identity\n-----------",
                    "My Console literacy\n----------------------",
                    "Console account & org administration",
                    "Proactive meeting offers\n------------------------",
                ),
                "absent": (
                    "Demo mode\n---------",
                    "Onboarding reference\n--------------------",
                    "Console knowledge\n-----------------",
                ),
            },
            {
                "name": "coordinator_demo_with_org",
                "kwargs": {
                    "is_coordinator": True,
                    "demo_mode": True,
                },
                "present": (
                    "T-W1N\n----",
                    "My identity\n-----------",
                    "Authorized humans\n-----------------",
                    "Demo mode\n---------",
                ),
                "absent": (
                    "**T-W1N admin tools:**",
                    "Act capabilities\n----------------",
                    "Concurrent action and acknowledgment\n------------------------------------",
                ),
            },
            {
                "name": "coordinator_non_demo_personal_workspace",
                "kwargs": {
                    "is_coordinator": True,
                    "is_org_workspace": False,
                },
                "present": (
                    "**T-W1N admin tools:**",
                    "Boss details\n------------",
                    "Organization membership actions are unavailable",
                    "switch to that organization's T-W1N",
                    "T-W1N\n----",
                    "My identity\n-----------",
                    "My Console literacy\n----------------------",
                    "Console account & org administration",
                    "Proactive meeting offers\n------------------------",
                ),
                "absent": (
                    "Authorized humans\n-----------------",
                    "Onboarding reference\n--------------------",
                    "Console knowledge\n-----------------",
                ),
            },
        )

        for case in cases:
            prompt = _build(**case["kwargs"])
            for marker in case["present"]:
                assert (
                    marker in prompt
                ), f"{case['name']} missing expected marker: {marker}"
            for marker in case["absent"]:
                assert (
                    marker not in prompt
                ), f"{case['name']} unexpectedly contains marker: {marker}"


class TestCoordinatorVoicePrompt:
    """Coordinator voice calls use T-W1N intro scaffolding plus optional user about."""

    def test_regular_voice_prompt_unchanged_when_flag_is_false(self):
        omitted = _build_voice()
        explicit_false = _build_voice(is_coordinator=False)

        assert omitted == explicit_false
        assert "Coordinator voice role" not in omitted

    def test_coordinator_voice_prompt_uses_twin_intro_and_optional_user_about(self):
        prompt = _build_voice(is_coordinator=True)

        assert "T-W1N\n----" in prompt
        assert "Role / specialization: Coordinator." in prompt
        assert "About me\n--------\nI help Acme configure its Unify team." in prompt
        assert "Bio\n---" not in prompt
        assert "Coordinator voice role" not in prompt
        assert prompt.index("T-W1N\n----") < prompt.index("Brevity\n-------")

    def test_coordinator_voice_prompt_omits_user_about_when_empty(self):
        prompt = _build_voice(is_coordinator=True, bio="")

        assert "T-W1N\n----" in prompt
        assert "About me\n--------" not in prompt

    def test_coordinator_voice_prompt_excludes_slow_brain_literacy(self):
        prompt = _build_voice(is_coordinator=True)

        assert "T-W1N admin tools" not in prompt
        assert "Unify system literacy" not in prompt
        assert "Requirements discovery workflow" not in prompt
        assert "Tasks/Activations" not in prompt
        assert "Context taxonomy" not in prompt
        assert "`create_assistant`" not in prompt
        assert "`delete_team`" not in prompt
        assert "`remove_team_member`" not in prompt

    def test_coordinator_voice_prompt_excludes_navigation_maps(self):
        prompt = _build_voice(is_coordinator=True)

        # The identity block remains on the fast brain.
        assert "My identity" in prompt
        assert "I am T-W1N, Dana Owner's personal, private assistant" in prompt
        assert "T-W1N is Dana Owner's personal, private assistant" not in prompt
        # The console-literacy and onboarding-flow maps are deliberately NOT
        # given to the fast brain: holding the same navigation knowledge as
        # the slow brain let the Voice Agent freelance contradictory
        # "what's next / where do I click" answers. Those questions now defer
        # to the slow brain (RULE 2), which owns onboarding navigation.
        assert "My Console literacy" not in prompt
        assert "Left sidebar — selection drives everything" not in prompt
        assert "Console account & org administration" not in prompt
        assert "Two ways to accomplish org tasks" not in prompt
        assert "My onboarding flow (UI reference)" not in prompt
        assert "Console knowledge\n-----------------" not in prompt
        assert "My opening turn" not in prompt
        assert "Onboarding checklist" not in prompt
        assert "Step-by-step walkthrough pacing" not in prompt


# ---------------------------------------------------------------------------
# Tests – missing-capability notices
# ---------------------------------------------------------------------------


class TestMissingCapabilityNotices:
    """The prompt includes notices explaining which channels are unavailable."""

    def test_no_notices_when_fully_configured(self):
        prompt = _build(assistant_has_phone=True, assistant_has_email=True)
        assert "do not currently have a phone number configured" not in prompt
        assert "do not currently have an email address configured" not in prompt

    def test_missing_phone_notice_present(self):
        prompt = _build(assistant_has_phone=False, assistant_has_email=True)
        assert "do not currently have a phone number configured" in prompt
        assert "cannot send SMS messages or make phone calls" in prompt
        assert "do not currently have an email address configured" not in prompt

    def test_missing_email_notice_present(self):
        prompt = _build(assistant_has_phone=True, assistant_has_email=False)
        assert "do not currently have an email address configured" in prompt
        assert "cannot send or receive emails" in prompt
        assert "do not currently have a phone number configured" not in prompt

    def test_both_notices_when_no_capabilities(self):
        prompt = _build(assistant_has_phone=False, assistant_has_email=False)
        assert "do not currently have a phone number configured" in prompt
        assert "do not currently have an email address configured" in prompt


# ---------------------------------------------------------------------------
# Tests – communication guidelines adapt
# ---------------------------------------------------------------------------


class TestCommunicationGuidelinesAdapt:
    """Contact-action examples and should_respond channels adjust."""

    def test_inline_sms_example_present_with_phone(self):
        prompt = _build(assistant_has_phone=True)
        assert "send_sms(contact_id=5" in prompt

    def test_inline_sms_example_absent_without_phone(self):
        prompt = _build(assistant_has_phone=False)
        assert "send_sms(contact_id=5" not in prompt

    def test_inline_email_example_present_with_email(self):
        prompt = _build(assistant_has_email=True)
        assert "send_email(to=[{" in prompt

    def test_inline_email_example_absent_without_email(self):
        prompt = _build(assistant_has_email=False)
        assert "send_email(to=[{" not in prompt

    def test_should_respond_lists_all_channels(self):
        prompt = _build(assistant_has_phone=True, assistant_has_email=True)
        assert "I can send SMS, emails, unify messages, calls" in prompt

    def test_should_respond_omits_phone_channels(self):
        prompt = _build(assistant_has_phone=False, assistant_has_email=True)
        assert "I can send emails, unify messages" in prompt
        assert "I can send SMS, emails, unify messages, calls" not in prompt

    def test_should_respond_omits_email_channel(self):
        prompt = _build(assistant_has_phone=True, assistant_has_email=False)
        assert "I can send SMS, unify messages, calls" in prompt
        assert "I can send SMS, emails" not in prompt

    def test_teams_workspace_actions_are_not_marked_contact_addressed(self):
        prompt = _build(
            assistant_has_phone=True,
            assistant_has_email=True,
            assistant_has_teams=True,
        )
        contact_actions = prompt.split("**Contact actions:**")[1].split(
            "- If the contact is NOT in active_conversations at all",
        )[0]
        contact_addressed_line = next(
            line
            for line in contact_actions.splitlines()
            if "Contact-addressed communication tools" in line
        )

        assert "send_teams_message" in contact_addressed_line
        assert "create_teams_channel" not in contact_addressed_line
        assert "create_teams_meet" not in contact_addressed_line
        assert (
            "`create_teams_channel` and `create_teams_meet` are Teams workspace actions"
            in contact_actions
        )


# ---------------------------------------------------------------------------
# Tests – external app integration
# ---------------------------------------------------------------------------


class TestExternalAppIntegration:
    """The prompt includes guidance for external app integration via credentials + SDK."""

    def test_act_capabilities_has_external_apps_bullet(self):
        prompt = _build()
        assert "**External apps & services**" in prompt
        assert "stored credentials and the service's Python SDK" in prompt

    def test_act_capabilities_absent_in_demo_mode(self):
        prompt = _build(demo_mode=True)
        assert "**External apps & services**" not in prompt


class TestExternalResourcesActBlock:
    """External-resource work must go through ``act``."""

    def test_external_resources_block_present(self):
        prompt = _build()
        assert "External resources (use ``act``)" in prompt
        assert "Ground truth rule" in prompt
        assert "I do not answer from memory" in prompt

    def test_external_resources_block_absent_in_demo_mode(self):
        prompt = _build(demo_mode=True)
        assert "External resources (use ``act``)" not in prompt


# ---------------------------------------------------------------------------
# Tests – seeing/controlling the user's machine (screen-share → linked desktop)
# ---------------------------------------------------------------------------


class TestUserMachineAccess:
    """Precedence guidance for seeing/controlling the *user's* machine.

    Default (no linked desktop) is unchanged from the screen-share-only
    behaviour; a linked desktop unlocks the direct-control path while keeping
    screen share as the first option.
    """

    def test_block_absent_without_linked_desktop(self):
        prompt = _build(has_linked_user_desktop=False)
        assert "Seeing and controlling the user's machine" not in prompt
        # Default capability copy: assistant only controls its own computer.
        assert "I cannot control the user's computer — only my own" in prompt

    def test_linked_desktop_enables_direct_control_block(self):
        prompt = _build(has_linked_user_desktop=True)
        assert "Seeing and controlling the user's machine" in prompt
        # Screen share remains the first-priority option.
        assert "Active screen share / webcam first" in prompt
        assert "linked to me" in prompt
        # Capability bullet reflects the linked machine.
        assert "my boss's own machine, which they've linked to me" in prompt

    def test_screen_share_still_offered_with_linked_desktop(self):
        """Linking a desktop must not remove the screen-share-first guidance."""
        prompt = _build(has_linked_user_desktop=True)
        assert "Want to share your screen?" in prompt
        # Proactive meeting/screen-share offers are untouched.
        assert "Proactive meeting offers" in prompt

    def test_faq_reflects_linked_desktop(self):
        prompt = _build(has_linked_user_desktop=True)
        assert "you've linked a desktop to me" in prompt

    def test_acting_user_id_surfaced_for_targeting(self):
        """When linked + an acting user id is known, the block tells the model
        which user_id to target so a shared assistant drives the speaker's
        machine (not the owner's)."""
        prompt = _build(has_linked_user_desktop=True, acting_user_id="user-42")
        assert 'user_desktop.session(user_id="user-42")' in prompt
        assert "user_desktop.list_linked()" in prompt

    def test_acting_user_id_absent_keeps_block_generic(self):
        prompt = _build(has_linked_user_desktop=True, acting_user_id=None)
        assert "Seeing and controlling the user's machine" in prompt
        assert "user_desktop.session(user_id=" not in prompt


class TestPerUserDesktopResolution:
    """``AssistantDetails.user_desktop_for`` keys linked desktops by the acting
    user, so N users x M assistants resolves the speaker's own machine."""

    @staticmethod
    def _assistant_with_links() -> object:
        from unify.session_details import AssistantDetails, UserDesktopLink

        a = AssistantDetails()
        a.user_desktops = {
            "user-A": UserDesktopLink(
                owner_user_id="user-A",
                url="http://a",
                os="macos",
            ),
            "user-B": UserDesktopLink(
                owner_user_id="user-B",
                url="http://b",
                os="ubuntu",
            ),
        }
        return a

    def test_resolves_speakers_own_link(self):
        a = self._assistant_with_links()
        assert a.user_desktop_for("user-B").url == "http://b"
        assert a.user_desktop_for("user-A").url == "http://a"

    def test_unlinked_speaker_returns_none(self):
        a = self._assistant_with_links()
        assert a.user_desktop_for("user-C") is None

    def test_missing_user_id_returns_none(self):
        a = self._assistant_with_links()
        assert a.user_desktop_for(None) is None


# ---------------------------------------------------------------------------
# Tests – proactive meeting offers
# ---------------------------------------------------------------------------


class TestProactiveMeetingOffers:
    """The prompt encourages proactive meeting/screenshare suggestions."""

    def test_proactive_meeting_section_present(self):
        prompt = _build()
        assert "Proactive meeting offers" in prompt
        assert "screen sharing" in prompt.lower()

    def test_proactive_meeting_absent_in_demo_mode(self):
        prompt = _build(demo_mode=True)
        assert "Proactive meeting offers" not in prompt

    def test_proactive_meeting_absent_for_coordinator(self):
        prompt = _build(is_coordinator=True)
        assert "Proactive meeting offers" in prompt


# ---------------------------------------------------------------------------
# Tests – console knowledge
# ---------------------------------------------------------------------------


class TestConsoleKnowledge:
    """The prompt includes console UI knowledge for guiding users."""

    def test_console_knowledge_present(self):
        prompt = _build()
        assert "Console knowledge" in prompt
        assert "Integrations" in prompt
        assert "Contact Details" in prompt

    def test_console_knowledge_has_navigation_paths(self):
        prompt = _build()
        assert "open the **Integrations** tab" in prompt
        assert "⋮ → **Contact Details**" in prompt
        assert "profile menu" in prompt

    def test_console_knowledge_absent_in_demo_mode(self):
        prompt = _build(demo_mode=True)
        assert "Console knowledge" not in prompt


# ---------------------------------------------------------------------------
# Tests – demo mode adapts
# ---------------------------------------------------------------------------


class TestDemoModeAdapts:
    """Demo mode section adjusts available channel listing."""

    def test_demo_lists_all_channels(self):
        prompt = _build(
            demo_mode=True,
            assistant_has_phone=True,
            assistant_has_email=True,
        )
        assert "SMS, emails, unify messages, calls" in prompt

    def test_demo_omits_phone_channels(self):
        prompt = _build(
            demo_mode=True,
            assistant_has_phone=False,
            assistant_has_email=True,
        )
        assert "SMS" not in prompt.split("CAN do")[1].split("CANNOT do")[0]

    def test_demo_omits_email_channel(self):
        prompt = _build(
            demo_mode=True,
            assistant_has_phone=True,
            assistant_has_email=False,
        )
        assert "emails" not in prompt.split("CAN do")[1].split("CANNOT do")[0]


# ---------------------------------------------------------------------------
# Tests – Console-UI / onboarding gate (local mode)
# ---------------------------------------------------------------------------


class TestConsoleUIGate:
    """Console-UI knowledge and onboarding prompts are gated on
    ``console_ui_present`` so the public local install (no Console) gets a
    trimmed prompt with a local-mode note instead."""

    def test_regular_console_knowledge_present_by_default(self):
        prompt = _build(is_coordinator=False)
        assert "Console knowledge" in prompt
        assert "Interaction surface" not in prompt

    def test_regular_console_knowledge_absent_in_local_mode(self):
        prompt = _build(is_coordinator=False, console_ui_present=False)
        assert "Console knowledge" not in prompt
        assert "Interaction surface" in prompt

    def test_coordinator_console_blocks_present_by_default(self):
        prompt = _build(is_coordinator=True)
        assert "Console literacy" in prompt
        assert "onboarding flow (UI reference)" in prompt
        assert (
            "steps route through the Assistant info → Onboarding checklist first"
            in prompt
        )

    def test_coordinator_console_blocks_absent_in_local_mode(self):
        prompt = _build(is_coordinator=True, console_ui_present=False)
        assert "Console literacy" not in prompt
        assert "onboarding flow (UI reference)" not in prompt
        assert "Interaction surface" in prompt

    def test_voice_platform_knowledge_present_by_default(self):
        prompt = _build_voice(is_coordinator=False)
        assert "Platform knowledge" in prompt

    def test_voice_platform_knowledge_absent_in_local_mode(self):
        prompt = _build_voice(is_coordinator=False, console_ui_present=False)
        assert "Platform knowledge" not in prompt


# ---------------------------------------------------------------------------
# Tests - small-talk sidecar prompt
# ---------------------------------------------------------------------------


class TestSmalltalkMessages:
    """The small-talk sidecar lets the fast brain fully answer pure social /
    biographical / self-context / repeat turns, and DEFERs everything else."""

    def test_build_smalltalk_messages_structure(self):
        from unify.conversation_manager.prompt_builders import build_smalltalk_messages

        history = [
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "what did you just say?"},
        ]
        msgs = build_smalltalk_messages(
            system_prompt="PERSONA",
            history_messages=history,
            user_text="what did you just say?",
        )
        # Persona first, guardrail near the end, caller's latest line last.
        assert msgs[0] == {"role": "system", "content": "PERSONA"}
        assert msgs[-1] == {"role": "user", "content": "what did you just say?"}
        assert msgs[-2]["role"] == "system"
        # History is preserved between persona and guardrail.
        assert {"role": "assistant", "content": "Hi there!"} in msgs

    def test_idle_status_smalltalk_guidance_is_absent_by_default(self):
        from unify.conversation_manager.prompt_builders import build_smalltalk_messages

        msgs = build_smalltalk_messages(
            system_prompt="PERSONA",
            history_messages=[],
            user_text="what are you doing?",
        )

        system_text = "\n".join(m["content"] for m in msgs if m["role"] == "system")
        assert "Idle status small-talk is available" not in system_text
        assert "Mario Kart" not in system_text

    def test_idle_status_smalltalk_guidance_is_gated(self):
        from unify.conversation_manager.prompt_builders import build_smalltalk_messages

        msgs = build_smalltalk_messages(
            system_prompt="PERSONA",
            history_messages=[],
            user_text="what are you doing?",
            idle_status_smalltalk=True,
        )

        system_text = "\n".join(m["content"] for m in msgs if m["role"] == "system")
        assert "Idle status small-talk is available" in system_text
        assert "what are you doing" in system_text
        assert "Snake" in system_text
        assert "Sudoku" in system_text
        assert "Mario Kart" in system_text
        assert "Tetris" in system_text

    def test_smalltalk_guardrail_allows_social_bio_selfcontext_repeat(self):
        from unify.conversation_manager.prompt_builders import _SMALLTALK_GUARDRAIL

        g = _SMALLTALK_GUARDRAIL.lower()
        assert "tell me about yourself" in g
        assert "repeat" in g
        # Self-context is allowed but only when actually known.
        assert "local time" in g or "where you are" in g
        assert "do not guess" in g or "do not actually know" in g

    def test_smalltalk_guardrail_defers_substantive_and_mixed(self):
        from unify.conversation_manager.prompt_builders import (
            SMALLTALK_DEFER_SENTINEL,
            _SMALLTALK_GUARDRAIL,
        )

        assert SMALLTALK_DEFER_SENTINEL == "DEFER"
        g = _SMALLTALK_GUARDRAIL
        assert "DEFER" in g
        low = g.lower()
        # Data / tools / actions and mixed turns must defer.
        assert "calendar" in low and "inbox" in low
        assert "tool" in low
        assert "mixed" in low
        assert "when unsure" in low

    def test_smalltalk_guardrail_stays_silent_on_bare_acks(self):
        from unify.conversation_manager.prompt_builders import (
            SMALLTALK_SILENCE_SENTINEL,
            _SMALLTALK_GUARDRAIL,
        )

        assert SMALLTALK_SILENCE_SENTINEL == "SILENCE"
        g = _SMALLTALK_GUARDRAIL
        assert "SILENCE" in g
        low = g.lower()
        # Bare acknowledgements -> silence, never echoed back.
        assert "acknowledgement" in low
        assert "never echo" in low
        # The carve-out: an 'okay' that authorises an action is NOT silence.
        assert "authorises an action" in low or "authorizes an action" in low

    def test_smalltalk_guardrail_defers_action_and_status_questions(self):
        from unify.conversation_manager.prompt_builders import _SMALLTALK_GUARDRAIL

        low = _SMALLTALK_GUARDRAIL.lower()
        # The fast brain must never promise/report on an action it controls.
        assert "hang up" in low
        assert "are you calling me" in low
        assert "did you send it yet" in low
        assert "idle status small-talk" in low
        assert "never promise, claim, or report" in low

    def test_slow_brain_voice_guide_knows_idle_smalltalk_exception(self):
        prompt = _build(is_voice_call=True)
        assert "Idle small-talk exception" in prompt
        assert "playing Snake" in prompt
        assert "no in-flight action" in prompt


class TestOnboardingPromptLeakageGuard:
    """Onboarding and general restraint blocks must not invite parroting."""

    def test_conversational_restraint_forbids_prompt_leakage(self):
        prompt = _build()
        assert "No prompt leakage" in prompt
        assert "never quote, paraphrase, or summarize" in prompt

    def test_coordinator_onboarding_narration_forbids_parroting(self):
        prompt = _build(is_coordinator=True)
        assert "My onboarding narration" in prompt
        assert "internal guidance — I never repeat it to the user" in prompt
        assert "No genre lists, franchise names" in prompt

    def test_coordinator_onboarding_scaffolding_omitted_when_inactive(self):
        prompt = _build(
            is_coordinator=True,
            coordinator_onboarding_active=False,
        )
        assert "My onboarding narration" not in prompt
        assert "My onboarding progress (live)" not in prompt

    def test_coordinator_onboarding_scaffolding_present_when_active(self):
        prompt = _build(
            is_coordinator=True,
            coordinator_onboarding_active=True,
            coordinator_onboarding_render={
                "steps": [],
                "next_targets": [],
            },
        )
        assert "My onboarding narration" in prompt

    def test_reference_quiz_rules_omit_parrotable_franchise_lists(self):
        prompt = _build(is_coordinator=True)
        assert "Star Wars" not in prompt
        assert "Blade Runner" not in prompt
        assert "quick sci-fi quiz" in prompt
        assert "I NEVER list genres, franchises" in prompt
