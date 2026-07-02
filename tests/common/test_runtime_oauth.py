from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from unify.common import runtime_oauth
from unify.provider_proxy.session import ProxySession


def _future_expiry() -> str:
    return (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()


def _past_expiry() -> str:
    return (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()


class _FakeSecretManager:
    def __init__(self, secrets: dict[str, str]) -> None:
        self.secrets = secrets
        self.sync_calls: list[dict[str, Any]] = []
        self.on_sync = None

    def _get_secret_value(self, name: str) -> str | None:
        return self.secrets.get(name)

    def sync_assistant_secrets_if_stale(self, **kwargs: Any) -> bool:
        self.sync_calls.append(kwargs)
        if self.on_sync is not None:
            self.on_sync()
        return True


def _install_secret_manager(monkeypatch, sm: _FakeSecretManager) -> None:
    monkeypatch.setattr(runtime_oauth, "_get_secret_manager", lambda: sm)


def _install_fake_proxy(monkeypatch, nonce: str = "proxy-nonce") -> ProxySession:
    session = ProxySession(host="127.0.0.1", port=4321, nonce=nonce)
    monkeypatch.setattr(
        "unify.provider_proxy.proxy.ensure_proxy_running",
        lambda: session,
    )
    return session


# ── Real, trusted-runtime token getter ───────────────────────────────────────


def test_get_provider_access_token_supports_provider_alias(monkeypatch):
    sm = _FakeSecretManager(
        {
            "MICROSOFT_ACCESS_TOKEN": "fresh-ms-token",
            "MICROSOFT_TOKEN_EXPIRES_AT": _future_expiry(),
        },
    )
    _install_secret_manager(monkeypatch, sm)

    assert runtime_oauth.get_provider_access_token("graph") == "fresh-ms-token"
    assert sm.sync_calls[-1]["force"] is False


def test_get_provider_access_token_prefers_in_memory_oauth_store(monkeypatch):
    sm = _FakeSecretManager({"MICROSOFT_TOKEN_EXPIRES_AT": _future_expiry()})
    sm.get_oauth_token = lambda name: (  # type: ignore[attr-defined]
        "in-memory-token" if name == "MICROSOFT_ACCESS_TOKEN" else None
    )
    _install_secret_manager(monkeypatch, sm)

    assert runtime_oauth.get_provider_access_token("microsoft") == "in-memory-token"


def test_get_provider_access_token_unknown_provider_is_non_secret_error(monkeypatch):
    sm = _FakeSecretManager({})
    _install_secret_manager(monkeypatch, sm)

    with pytest.raises(ValueError, match="Unknown refresh-token OAuth provider"):
        runtime_oauth.get_provider_access_token("not-a-real-provider")


def test_get_provider_access_token_missing_token_raises_after_sync(monkeypatch):
    sm = _FakeSecretManager({"MICROSOFT_TOKEN_EXPIRES_AT": _future_expiry()})
    _install_secret_manager(monkeypatch, sm)

    with pytest.raises(ValueError, match="No access token is available"):
        runtime_oauth.get_provider_access_token("microsoft")

    assert sm.sync_calls[-1]["force"] is True


def test_get_provider_access_token_forces_sync_when_token_is_expired(monkeypatch):
    sm = _FakeSecretManager(
        {
            "MICROSOFT_ACCESS_TOKEN": "old-ms-token",
            "MICROSOFT_TOKEN_EXPIRES_AT": _past_expiry(),
        },
    )
    _install_secret_manager(monkeypatch, sm)

    def refresh_token() -> None:
        sm.secrets["MICROSOFT_ACCESS_TOKEN"] = "fresh-ms-token"
        sm.secrets["MICROSOFT_TOKEN_EXPIRES_AT"] = _future_expiry()

    sm.on_sync = refresh_token

    assert runtime_oauth.get_provider_access_token("microsoft") == "fresh-ms-token"
    assert sm.sync_calls[-1]["force"] is True


def test_get_provider_access_token_forces_sync_when_expiry_is_invalid(monkeypatch):
    sm = _FakeSecretManager(
        {
            "GOOGLE_ACCESS_TOKEN": "old-google-token",
            "GOOGLE_TOKEN_EXPIRES_AT": "not-a-date",
        },
    )
    _install_secret_manager(monkeypatch, sm)

    def refresh_token() -> None:
        sm.secrets["GOOGLE_ACCESS_TOKEN"] = "fresh-google-token"
        sm.secrets["GOOGLE_TOKEN_EXPIRES_AT"] = _future_expiry()

    sm.on_sync = refresh_token

    assert runtime_oauth.get_provider_access_token("google") == "fresh-google-token"
    assert sm.sync_calls[-1]["force"] is True


def test_get_provider_access_token_supports_multiple_providers(monkeypatch):
    sm = _FakeSecretManager(
        {
            "MICROSOFT_ACCESS_TOKEN": "fresh-ms-token",
            "MICROSOFT_TOKEN_EXPIRES_AT": _future_expiry(),
            "GOOGLE_ACCESS_TOKEN": "fresh-google-token",
            "GOOGLE_TOKEN_EXPIRES_AT": _future_expiry(),
        },
    )
    _install_secret_manager(monkeypatch, sm)

    assert runtime_oauth.get_provider_access_token("microsoft") == "fresh-ms-token"
    assert runtime_oauth.get_provider_access_token("google") == "fresh-google-token"


# ── Sandbox-facing capability handle (never a real token) ─────────────────────


def test_get_oauth_access_token_returns_proxy_handle_not_token(monkeypatch):
    session = _install_fake_proxy(monkeypatch, nonce="handle-123")
    # No secret manager access should be needed for the handle.
    assert runtime_oauth.get_oauth_access_token("microsoft") == "handle-123"
    assert runtime_oauth.get_oauth_access_token("google") == session.nonce


def test_get_oauth_access_token_validates_provider_before_proxy(monkeypatch):
    _install_fake_proxy(monkeypatch)
    with pytest.raises(ValueError, match="Unknown refresh-token OAuth provider"):
        runtime_oauth.get_oauth_access_token("not-a-real-provider")


def test_get_refresh_token_oauth_env_overlay_returns_proxy_endpoints(monkeypatch):
    session = _install_fake_proxy(monkeypatch, nonce="nonce-xyz")

    overlay = runtime_oauth.get_refresh_token_oauth_env_overlay()

    assert overlay == session.sandbox_env()
    assert overlay["WORKSPACE_PROXY_TOKEN"] == "nonce-xyz"
    assert overlay["MICROSOFT_GRAPH_BASE"].endswith("/microsoft/v1.0")
    assert overlay["GOOGLE_DRIVE_BASE"].endswith("/google/drive/v3")
    # No raw provider tokens are ever overlaid into the sandbox.
    assert "MICROSOFT_ACCESS_TOKEN" not in overlay
    assert "GOOGLE_ACCESS_TOKEN" not in overlay


def test_optimistic_returns_token_despite_stale_expiry(monkeypatch):
    # Expiry metadata says expired, but the token is still usable: optimistic
    # must return it (no pre-emptive gate) and not raise.
    sm = _FakeSecretManager(
        {
            "MICROSOFT_ACCESS_TOKEN": "still-valid",
            "MICROSOFT_TOKEN_EXPIRES_AT": _past_expiry(),
        },
    )
    _install_secret_manager(monkeypatch, sm)

    assert runtime_oauth.get_provider_access_token_optimistic("microsoft") == (
        "still-valid"
    )
    # Token present => debounced (not forced) sync.
    assert sm.sync_calls[-1]["force"] is False


def test_optimistic_returns_none_when_missing_and_forces_sync(monkeypatch):
    sm = _FakeSecretManager({})
    _install_secret_manager(monkeypatch, sm)

    assert runtime_oauth.get_provider_access_token_optimistic("microsoft") is None
    assert sm.sync_calls[-1]["force"] is True


def test_refresh_forces_sync_and_returns_new_token(monkeypatch):
    sm = _FakeSecretManager({"MICROSOFT_ACCESS_TOKEN": "old"})
    _install_secret_manager(monkeypatch, sm)
    sm.on_sync = lambda: sm.secrets.__setitem__("MICROSOFT_ACCESS_TOKEN", "new")

    assert runtime_oauth.refresh_provider_access_token("microsoft") == "new"
    assert sm.sync_calls[-1]["force"] is True


def test_refresh_token_oauth_token_names_are_sensitive_subset():
    names = runtime_oauth.refresh_token_oauth_token_names()
    assert names == {
        "MICROSOFT_ACCESS_TOKEN",
        "MICROSOFT_REFRESH_TOKEN",
        "GOOGLE_ACCESS_TOKEN",
        "GOOGLE_REFRESH_TOKEN",
    }
    # Expiry / granted-scope metadata is NOT sensitive and stays out of this set.
    assert "MICROSOFT_TOKEN_EXPIRES_AT" not in names
    assert "GOOGLE_GRANTED_SCOPES" not in names
