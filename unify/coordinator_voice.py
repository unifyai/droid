"""Coordinator default voice resolution for Unity runtime."""

from __future__ import annotations

COORDINATOR_DEFAULT_VOICE_PROVIDER = "elevenlabs"
COORDINATOR_DEFAULT_VOICE_ID = "iP95p4xoKVk53GoZ742B"


def _normalize(value: object | None) -> str:
    return "" if value is None else str(value).strip()


def resolve_runtime_voice(
    *,
    is_coordinator: bool,
    voice_provider: object | None,
    voice_id: object | None,
) -> tuple[str, str]:
    """Return the TTS provider and voice id to use for one assistant session."""
    provider = _normalize(voice_provider)
    voice = _normalize(voice_id)
    if is_coordinator and not voice:
        return COORDINATOR_DEFAULT_VOICE_PROVIDER, COORDINATOR_DEFAULT_VOICE_ID
    return provider or "cartesia", voice
