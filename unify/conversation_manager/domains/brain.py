from __future__ import annotations

import json
from dataclasses import dataclass, field
from time import perf_counter
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field, create_model

from unify.common.startup_timing import log_startup_timing
from unify.common.prompt_helpers import PromptParts
from unify.conversation_manager.prompt_builders import build_system_prompt
from unify.conversation_manager.runtime_status import (
    deployment_runtime_reconcile_prompt_note,
)
from unify.conversation_manager.cm_types import Mode, ScreenshotEntry
from unify.logger import LOGGER
from unify.session_details import SESSION_DETAILS

if TYPE_CHECKING:
    from unify.conversation_manager.conversation_manager import ConversationManager
    from unify.conversation_manager.domains.renderer import SnapshotState


def _build_response_models() -> dict[Mode, type[BaseModel]]:
    """
    Create response models for ConversationManager's main brain.

    All actions (comms, task steering, etc.) are now tool calls.
    The response model only captures the LLM's reasoning.

    Returns:
        dict: Response models for different modes (Mode.CALL, Mode.MEET, Mode.TEXT)
    """
    # Text mode: just thoughts
    TextResponse = create_model(
        "TextResponse",
        thoughts=(
            str,
            Field(..., description="Your concise reasoning before taking actions"),
        ),
        __base__=BaseModel,
    )

    # Voice mode uses the same response model as text mode.
    # Guidance is delivered via the standalone guide_voice_agent tool (called in
    # parallel with the action tool) rather than the response content, because
    # the model reliably populates tool arguments but intermittently skips the
    # content field when tool_choice is required.
    return {
        Mode.CALL: TextResponse,
        Mode.MEET: TextResponse,
        Mode.TEXT: TextResponse,
    }


# Cache the response models since they don't change
_RESPONSE_MODELS = _build_response_models()


def build_response_models() -> dict[Mode, type[BaseModel]]:
    """
    Public accessor for response models used by ConversationManager's brain.

    Returns cached models for different modes (Mode.CALL, Mode.MEET, Mode.TEXT).
    """
    return _RESPONSE_MODELS


@dataclass(frozen=True)
class BrainSpec:
    """
    Fully materialized inputs needed for a single Main CM Brain run.

    This is intentionally a small, explicit data structure so we can refactor
    how the brain is executed (plain generate vs async tool loop) without
    tangling prompt/model construction with execution and side effects.
    """

    system_prompt: PromptParts
    state_prompt: str
    response_model: type["BaseModel"]
    # Buffered screenshots captured during screen sharing, aligned with user turns.
    screenshots: list[ScreenshotEntry] = field(default_factory=list)
    # Relative file paths for each screenshot (parallel to screenshots list).
    screenshot_paths: list[str] = field(default_factory=list)

    def state_message(self) -> dict:
        # Mark this as a state snapshot so the async tool loop can treat it as
        # transient state (e.g., keep only the latest snapshot when generating).
        if not self.screenshots:
            return {
                "role": "user",
                "content": self.state_prompt,
                "_cm_state_snapshot": True,
            }

        # Build multimodal content: text state + screenshot blocks aligned with
        # the user utterances that triggered them.
        sources = {s.source for s in self.screenshots}
        if len(sources) > 1:
            header = (
                "The following screenshots were captured from multiple visual "
                "sources (desktop, user screen, meeting view, and/or webcam), "
                "each paired with what the user said at that moment. They are "
                "in chronological order."
            )
        elif "google_meet" in sources:
            header = (
                "The following screenshots were captured from the Google Meet "
                "call, showing the meeting view as you see it. They are paired "
                "with what was said at each moment and are in chronological order."
            )
        elif "teams_meet" in sources:
            header = (
                "The following screenshots were captured from the Microsoft "
                "Teams meeting, showing the meeting view as you see it. They "
                "are paired with what was said at each moment and are in "
                "chronological order."
            )
        elif "user" in sources:
            header = (
                "The following screenshots were captured from the user's screen "
                "during screen sharing, each paired with what the user said "
                "at that moment. They are in chronological order."
            )
        elif "webcam" in sources:
            header = (
                "The following frames were captured from the user's webcam, "
                "each paired with what the user said at that moment. "
                "They are in chronological order."
            )
        else:
            header = (
                "The following screenshots were captured from your desktop "
                "during screen sharing, each paired with what the user said "
                "at that moment. They are in chronological order."
            )

        content_parts: list[dict] = [
            {"type": "text", "text": self.state_prompt},
            {
                "type": "text",
                "text": (
                    f"\n\n<screen_share_snapshots>\n{header}\n"
                    "</screen_share_snapshots>"
                ),
            },
        ]
        source_labels = {
            "assistant": "Assistant's Screen",
            "user": "User's Screen",
            "webcam": "User's Webcam",
            "google_meet": "Google Meet",
            "teams_meet": "Microsoft Teams",
        }
        for i, entry in enumerate(self.screenshots, 1):
            label = source_labels.get(entry.source, "Screenshot")
            path_suffix = ""
            if i <= len(self.screenshot_paths):
                path_suffix = f" -- {self.screenshot_paths[i - 1]}"
            content_parts.append(
                {
                    "type": "text",
                    "text": (
                        f"\n[{label} - Screenshot {i}/{len(self.screenshots)}"
                        f"{path_suffix}] "
                        f'User said: "{entry.utterance}"'
                    ),
                },
            )
            content_parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{entry.b64}"},
                },
            )

        return {
            "role": "user",
            "content": content_parts,
            "_cm_state_snapshot": True,
        }


def build_brain_spec(
    cm: "ConversationManager",
    snapshot_state: "SnapshotState",
    screenshots: list[ScreenshotEntry] | None = None,
    screenshot_paths: list[str] | None = None,
    acting_user_id: str | None = None,
) -> BrainSpec:
    """
    Build the prompt + response model inputs for a single Main CM Brain run.

    The returned spec is *pure* (no side effects) and can be used by either the
    legacy single-shot generate path or the async tool loop path.

    Parameters
    ----------
    cm : ConversationManager
        The conversation manager instance.
    snapshot_state : SnapshotState
        Pre-rendered conversation state (caller computes this once and reuses it
        for both the BrainSpec and incremental-diff tracking).
    screenshots : list[ScreenshotEntry] | None
        Buffered screenshots from screen sharing (assistant and/or user), each
        paired with the user utterance that triggered capture and a timestamp.
    screenshot_paths : list[str] | None
        Relative file paths corresponding to each screenshot (parallel list).
    acting_user_id : str | None
        The user acting in this turn (the inbound message sender when it maps to
        a system user, else the workspace owner). Used to resolve the *speaker's*
        linked desktop. Falls back to the session owner when not provided.
    """
    acting_user_id = acting_user_id or SESSION_DETAILS.user.id
    from unify.settings import SETTINGS

    _brain_t0 = perf_counter()
    _last_step = _brain_t0

    def _mark_step() -> float:
        nonlocal _last_step
        now = perf_counter()
        elapsed_ms = (now - _last_step) * 1000
        _last_step = now
        return elapsed_ms

    prompt = snapshot_state.full_render
    _prompt_ms = _mark_step()

    boss_contact_id = SESSION_DETAILS.boss_contact_id
    boss_contact = cm.contact_index.get_contact(boss_contact_id) or {}
    _boss_contact_ms = _mark_step()
    is_internal_call = cm.mode.is_voice and bool(
        (cm.get_active_contact() or {}).get("is_system", False),
    )
    authorized_humans: list[dict] | None = None
    if cm.initialized:
        from unify.coordinator_manager.coordinator_manager import CoordinatorManager

        coordinator_manager = CoordinatorManager()
        if SESSION_DETAILS.is_coordinator and SESSION_DETAILS.org_id is not None:
            authorized_humans = coordinator_manager.get_org_members()

    _active_contact_ms = _mark_step()
    # Twin sessions carry fixed intro scaffolding in the prompt builder.
    # Regular assistants prepend job title into the bio block when set.
    _bio_parts: list[str] = []
    if not SESSION_DETAILS.is_coordinator:
        _job_title = (cm.assistant_job_title or "").strip()
        if _job_title:
            _bio_parts.append(f"Role / specialization: {_job_title}.")
    if cm.assistant_about:
        _bio_parts.append(cm.assistant_about)
    _bio_text = "\n".join(_bio_parts)
    _bio_ms = _mark_step()
    runtime_setup_note = deployment_runtime_reconcile_prompt_note(cm)
    _runtime_status_ms = _mark_step()
    _user_desktop_link = SESSION_DETAILS.assistant.user_desktop_for(acting_user_id)
    system_prompt = build_system_prompt(
        bio=_bio_text,
        contact_id=boss_contact_id,
        first_name=boss_contact.get("first_name") or "",
        surname=boss_contact.get("surname") or "",
        phone_number=boss_contact.get("phone_number"),
        email_address=boss_contact.get("email_address"),
        is_voice_call=cm.mode.is_voice,
        is_internal_call=is_internal_call,
        on_voice_call=cm.in_voice_session,
        outbound_voice_line_ready=cm.call_manager.is_ready_for_outbound_call,
        demo_mode=SETTINGS.DEMO_MODE,
        computer_fast_path=cm.computer_fast_path_eligible,
        assistant_has_phone=bool(cm.assistant_number),
        assistant_has_email=bool(cm.assistant_email),
        assistant_has_whatsapp=bool(cm.assistant_whatsapp_number),
        assistant_has_discord=bool(cm.assistant_discord_bot_id),
        assistant_has_slack=bool(cm.assistant_slack_bot_user_id),
        assistant_has_teams=bool(cm.assistant_has_teams),
        has_linked_user_desktop=_user_desktop_link is not None,
        user_filesys_consented=bool(
            _user_desktop_link and _user_desktop_link.filesys_sync,
        ),
        user_filesys_available=bool(
            _user_desktop_link and _user_desktop_link.filesys_available,
        ),
        acting_user_id=acting_user_id,
        runtime_setup_note=runtime_setup_note,
        team_summaries=getattr(cm, "team_summaries", []),
        is_coordinator=SESSION_DETAILS.is_coordinator,
        authorized_humans=authorized_humans,
        is_org_workspace=SESSION_DETAILS.org_id is not None,
        console_ui_present=SETTINGS.UNITY_CONSOLE_UI,
        coordinator_onboarding_active=cm.coordinator_onboarding_active,
        coordinator_onboarding_render=cm.coordinator_onboarding_render,
        coordinator_clicked_trigger_steps=cm.onboarding_clicked_trigger_steps,
        onboarding_catalog=cm.onboarding_catalog,
    )
    _system_prompt_ms = _mark_step()

    response_model = _RESPONSE_MODELS[cm.mode]
    _response_model_ms = _mark_step()

    # Validate we can JSON-encode state prompt early (helps catch accidental objects)
    json.dumps({"state_prompt": prompt})
    _json_validate_ms = _mark_step()

    spec = BrainSpec(
        system_prompt=system_prompt,
        state_prompt=prompt,
        response_model=response_model,
        screenshots=screenshots or [],
        screenshot_paths=screenshot_paths or [],
    )
    _spec_ms = _mark_step()

    log_startup_timing(
        LOGGER,
        (
            "⏱️ [StartupTiming] llm_preamble.brain_spec.detail "
            "total=%.0fms prompt_ref=%.0fms boss_contact=%.0fms "
            "active_contact=%.0fms bio=%.0fms runtime_status=%.0fms "
            "system_prompt=%.0fms response_model=%.0fms json_validate=%.0fms "
            "spec=%.0fms state_chars=%d system_chars=%d system_parts=%d "
            "screenshots=%d runtime_note=%s mode=%s"
        ),
        (perf_counter() - _brain_t0) * 1000,
        _prompt_ms,
        _boss_contact_ms,
        _active_contact_ms,
        _bio_ms,
        _runtime_status_ms,
        _system_prompt_ms,
        _response_model_ms,
        _json_validate_ms,
        _spec_ms,
        len(prompt),
        len(system_prompt.flatten()),
        len(system_prompt.to_list()),
        len(screenshots or []),
        runtime_setup_note is not None,
        cm.mode,
    )

    return spec
