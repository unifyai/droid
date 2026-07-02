"""Tests for outbound plain-text normalization."""

from __future__ import annotations

from unify.common.plain_text import normalize_outbound_plain_text


def test_collapses_hard_wrapped_paragraph() -> None:
    raw = (
        "This is part of onboarding — we're testing that email works\n"
        "with a quick sci-fi reference quiz. Reply with your guess:\n\n"
        '"I have a bad feeling about this."'
    )
    assert (
        normalize_outbound_plain_text(raw)
        == 'This is part of onboarding — we\'re testing that email works with a quick sci-fi reference quiz. Reply with your guess:\n\n"I have a bad feeling about this."'
    )


def test_preserves_existing_paragraph_breaks() -> None:
    raw = "First paragraph still fine.\n\nSecond paragraph here."
    assert normalize_outbound_plain_text(raw) == raw


def test_leaves_continuous_text_unchanged() -> None:
    raw = "Already one flowing line."
    assert normalize_outbound_plain_text(raw) == raw


def test_normalizes_crlf_and_trims_edges() -> None:
    raw = "  Hello\r\nworld  \r\n\r\n  Second block  "
    assert normalize_outbound_plain_text(raw) == "Hello world\n\nSecond block"


def test_empty_and_whitespace_only() -> None:
    assert normalize_outbound_plain_text("") == ""
    assert normalize_outbound_plain_text("   \n  \n  ") == ""


def test_preserves_bullet_list_items() -> None:
    raw = "- first thing\n- second thing"
    assert normalize_outbound_plain_text(raw) == raw


def test_preserves_numbered_list_items() -> None:
    raw = "1. First item\n2. Second item"
    assert normalize_outbound_plain_text(raw) == raw


def test_preserves_list_intro_line_before_bullets() -> None:
    raw = "Here are the steps:\n- first thing\n- second thing"
    assert normalize_outbound_plain_text(raw) == raw


def test_joins_line_after_colon_when_not_starting_a_list() -> None:
    raw = "Note: this continues\non the next line."
    assert (
        normalize_outbound_plain_text(raw) == "Note: this continues on the next line."
    )
