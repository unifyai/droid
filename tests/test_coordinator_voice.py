from unify.coordinator_voice import (
    COORDINATOR_DEFAULT_VOICE_ID,
    COORDINATOR_DEFAULT_VOICE_PROVIDER,
    resolve_runtime_voice,
)


def test_resolve_runtime_voice_uses_coordinator_default_when_missing():
    provider, voice_id = resolve_runtime_voice(
        is_coordinator=True,
        voice_provider="",
        voice_id="",
    )
    assert provider == COORDINATOR_DEFAULT_VOICE_PROVIDER
    assert voice_id == COORDINATOR_DEFAULT_VOICE_ID
