"""Verify screen-share state does not leak across CM tests."""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from tests.conversation_manager.conftest import _reset_screen_share_state
from unify.conversation_manager.cm_types.screenshot import ScreenshotEntry


def _screenshot_entry() -> ScreenshotEntry:
    return ScreenshotEntry(
        b64="AAAA",
        utterance="test utterance",
        timestamp=datetime.now(timezone.utc),
        source="user",
    )


def test_reset_screen_share_state_helper_clears_buffer():
    cm = SimpleNamespace(
        user_screen_share_active=True,
        assistant_screen_share_active=True,
        _screenshot_buffer=[_screenshot_entry()],
    )
    driver = SimpleNamespace(cm=cm)

    _reset_screen_share_state(driver)  # type: ignore[arg-type]

    assert cm.user_screen_share_active is False
    assert cm.assistant_screen_share_active is False
    assert cm._screenshot_buffer == []


@pytest.mark.requires_orchestra
def test_previous_test_pollutes_screen_share_state(conversation_manager):
    conversation_manager.cm.user_screen_share_active = True
    conversation_manager.cm.assistant_screen_share_active = True
    conversation_manager.cm._screenshot_buffer.append(_screenshot_entry())

    assert conversation_manager.cm.user_screen_share_active is True
    assert conversation_manager.cm.assistant_screen_share_active is True
    assert len(conversation_manager.cm._screenshot_buffer) == 1


@pytest.mark.requires_orchestra
def test_initialized_cm_resets_screen_share_state(initialized_cm):
    assert initialized_cm.cm.user_screen_share_active is False
    assert initialized_cm.cm.assistant_screen_share_active is False
    assert initialized_cm.cm._screenshot_buffer == []
