"""Unit tests for the coordinator onboarding reset narration.

``_coordinator_onboarding_notification_text`` is a pure function that turns
one ``CoordinatorOnboardingEvent`` into the brain-facing instruction string.
The reset subtype is the one that must tell the brain a step is no longer
done so a later nudge cannot claim it complete.
"""

from __future__ import annotations

from unify.conversation_manager.domains.coordinator_onboarding import (
    _coordinator_onboarding_notification_text,
)
from unify.conversation_manager.events import CoordinatorOnboardingEvent


def test_reset_notification_marks_step_not_done_and_names_it() -> None:
    event = CoordinatorOnboardingEvent(
        subtype="onboarding_step_reset",
        message="User reset the 'workspace-mailbox' onboarding step.",
        details={"step_id": "workspace-mailbox"},
    )
    text = _coordinator_onboarding_notification_text(event)
    assert "onboarding_step_reset" in text
    assert "`workspace-mailbox`" in text
    # The brain must not treat the step as done and must not spontaneously act.
    assert "no longer complete" in text
    assert "never claim it is done" in text
    assert "live" in text


def test_reference_quiz_notification_stays_minimal() -> None:
    event = CoordinatorOnboardingEvent(
        subtype="reference_quiz_clue_requested",
        message="User clicked the email reference-quiz trigger.",
        details={
            "channel": "email",
            "tool_name": "send_email",
            "trigger_step_id": "email-reference",
            "reply_step_id": "email-reply",
            "interaction": {"type": "reference_quiz"},
        },
    )
    text = _coordinator_onboarding_notification_text(event)
    assert "reference_quiz_clue_requested" in text
    assert "sci-fi quote clue" in text
    assert "never list genres or franchises" in text
    assert "Star Wars" not in text
    assert "pop-culture" not in text
