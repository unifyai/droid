"""
tests/conversation_manager/core/test_onboarding_outbound_media.py
=================================================================

Drift-guard for the onboarding outbound-medium table baked into
``ConversationManager.consume_pending_onboarding_outbound``.

Orchestra derives onboarding step completion from durable, assistant-authored
transcript rows on a specific set of mediums per channel (onboarding_graph's
``_CHANNEL_TO_OUTBOUND_MEDIUMS`` merged with ``DEMO_TO_OUTBOUND_MEDIUMS``). Unity
must stamp its onboarding metadata onto exactly those mediums and no others, or a
channel silently stops auto-completing (metadata never lands on the row Orchestra
reads). Unify cannot import Orchestra, so the canonical mapping is mirrored here as
a golden constant; a change on either side must be applied to both.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from unify.conversation_manager.conversation_manager import ConversationManager

# Canonical channel -> accepted outbound mediums. Source of truth:
#   orchestra/services/onboarding_graph.py
#     _CHANNEL_TO_OUTBOUND_MEDIUMS   (reference-quiz comms channels)
#   + DEMO_TO_OUTBOUND_MEDIUMS       (workspace demo rows, proved over unify_message)
# Keep this mirror in lockstep with that mapping and with the expected_media
# table in ConversationManager.consume_pending_onboarding_outbound.
CANONICAL_CHANNEL_TO_MEDIA: dict[str, frozenset[str]] = {
    "email": frozenset({"email"}),
    "sms_message": frozenset({"sms_message"}),
    "whatsapp_message": frozenset({"whatsapp_message"}),
    "whatsapp_call": frozenset({"whatsapp_call"}),
    "phone_call": frozenset({"phone_call"}),
    "slack_message": frozenset({"slack_message", "slack_channel_message"}),
    "discord_message": frozenset({"discord_message", "discord_channel_message"}),
    "workspace_mailbox": frozenset({"unify_message"}),
    "workspace_drive": frozenset({"unify_message"}),
    "workspace_calendar": frozenset({"unify_message"}),
    "workspace_contacts": frozenset({"unify_message"}),
    "workspace_tasks": frozenset({"unify_message"}),
    "workspace_teams": frozenset({"unify_message"}),
}

_ALL_MEDIA: frozenset[str] = frozenset(
    medium for media in CANONICAL_CHANNEL_TO_MEDIA.values() for medium in media
)

_PENDING_METADATA = {
    "onboarding_trigger_step_id": "trigger-step",
    "onboarding_reply_step_id": "reply-step",
    "onboarding_request_id": "req-1",
    "onboarding_origin_event_id": "evt-1",
}


def _pending_stub(channel: str) -> SimpleNamespace:
    """A minimal stand-in exposing only what ``consume`` reads.

    ``consume_pending_onboarding_outbound`` touches ``self.loop.time()`` and
    ``self._pending_onboarding_outbound`` and nothing else, so it can run against
    a light namespace rather than a fully-constructed ConversationManager.
    """

    return SimpleNamespace(
        loop=SimpleNamespace(time=lambda: 0.0),
        _pending_onboarding_outbound={
            "channel": channel,
            "expires_at": 1e18,
            **_PENDING_METADATA,
        },
    )


@pytest.mark.parametrize("channel", sorted(CANONICAL_CHANNEL_TO_MEDIA))
def test_accepted_media_stamp_onboarding_metadata(channel: str) -> None:
    for medium in sorted(CANONICAL_CHANNEL_TO_MEDIA[channel]):
        stub = _pending_stub(channel)
        result = ConversationManager.consume_pending_onboarding_outbound(stub, medium)
        assert result == _PENDING_METADATA, (channel, medium)
        # A matched consume is single-shot: it clears the pending outbound.
        assert stub._pending_onboarding_outbound is None


@pytest.mark.parametrize("channel", sorted(CANONICAL_CHANNEL_TO_MEDIA))
def test_foreign_media_are_rejected(channel: str) -> None:
    for medium in sorted(_ALL_MEDIA - CANONICAL_CHANNEL_TO_MEDIA[channel]):
        stub = _pending_stub(channel)
        result = ConversationManager.consume_pending_onboarding_outbound(stub, medium)
        assert result is None, (channel, medium)
        # A non-match leaves the pending outbound intact for a later matching send.
        assert stub._pending_onboarding_outbound is not None
