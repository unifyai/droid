"""Plain-text helpers for assistant-authored outbound messages."""

from __future__ import annotations

import re

_INLINE_WS = re.compile(r"[ \t]+")
_LIST_ITEM_RE = re.compile(
    r"^(?:" r"[-*•·]\s+" r"|\d+\.\s+" r"|\d+\)\s+" r"|-\s+\[[ xX]\]\s+" r")",
)


def _is_list_item(line: str) -> bool:
    return bool(_LIST_ITEM_RE.match(line.strip()))


def _should_join_wrapped_lines(previous: str, current: str) -> bool:
    if _is_list_item(previous) or _is_list_item(current):
        return False
    return True


def _normalize_paragraph_block(block: str) -> str:
    merged: list[str] = []
    for raw_line in block.split("\n"):
        line = _INLINE_WS.sub(" ", raw_line.strip())
        if not line:
            continue
        if merged and _should_join_wrapped_lines(merged[-1], line):
            merged[-1] = _INLINE_WS.sub(" ", f"{merged[-1]} {line}")
        else:
            merged.append(line)
    return "\n".join(merged)


def normalize_outbound_plain_text(text: str) -> str:
    """Collapse hard line wraps while preserving paragraph and list structure.

    LLMs often hard-wrap prose near 80 columns. Mail clients and chat surfaces
    reflow continuous text naturally, but they honor every ``\\n`` in plain text
    as a fixed break — which produces ragged edges on wide screens. This helper
    joins lines within a prose paragraph into one flowing line, keeps blank lines
    (two or more consecutive newlines) as ``\\n\\n`` paragraph separators, and
    leaves bullet/numbered list items on separate lines.
    """
    if not text:
        return text
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return ""
    paragraphs: list[str] = []
    for block in re.split(r"\n{2,}", normalized):
        paragraph = _normalize_paragraph_block(block)
        if paragraph:
            paragraphs.append(paragraph)
    return "\n\n".join(paragraphs)
