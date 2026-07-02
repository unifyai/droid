"""Gating of reference-quiz comms tools during onboarding.

During the communication phase of onboarding, each channel's send tool is
withheld from the slow brain until the user clicks that channel's
"Trigger ... from T-W1N" row (an ephemeral, this-session signal) or the step
is durably complete. This prevents T-W1N from proactively sending an untagged
clue before the click, which is what tags the outbound so the step can
auto-complete.

The masking decision is a pure, fail-open function of the onboarding render
(durable per-step status, supplied by Orchestra) plus the in-session set of
clicked trigger-step ids. Masking requires positive, fresh evidence; any
missing/None/malformed input masks nothing, so a tool can never be left
permanently hidden by a dropped event, race, or restart.
"""

from __future__ import annotations

from typing import Any

# Orchestra advertises the boss-facing call tools with a ``_to_boss`` suffix in
# the reference-quiz interaction; unity's tool registry uses the bare names.
_TOOL_NORMALIZATION = {
    "make_call_to_boss": "make_call",
    "make_whatsapp_call_to_boss": "make_whatsapp_call",
}


def masked_reference_quiz_tools(
    onboarding_render: dict[str, Any] | None,
    clicked_trigger_steps: set[str] | None,
) -> set[str]:
    """Return the unity send-tool names to withhold for this turn.

    A reference-quiz trigger step's tool is masked iff ``onboarding_active``
    is true, onboarding render is present, the step is not ``done``/``skipped``
    AND its trigger-step id is not in ``clicked_trigger_steps`` (clicked this
    session). Anything missing or malformed yields no masking (fail open).
    """
    if not isinstance(onboarding_render, dict):
        return set()
    clicked = clicked_trigger_steps or set()
    masked: set[str] = set()
    for step in onboarding_render.get("steps", []) or []:
        try:
            if not isinstance(step, dict) or step.get("kind") != "trigger":
                continue
            interaction = step.get("interaction") or {}
            if interaction.get("type") != "reference_quiz":
                continue
            if step.get("status") in ("done", "skipped"):
                continue
            if step.get("id") in clicked:
                continue
            tool = interaction.get("tool_name")
            if tool:
                masked.add(_TOOL_NORMALIZATION.get(tool, tool))
        except Exception:
            # Fail open per step: never let a parsing error hide a tool.
            continue
    return masked
