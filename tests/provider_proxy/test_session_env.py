from __future__ import annotations

from unify.provider_proxy import session as sess
from unify.provider_proxy.session import ProxySession, build_sandbox_env
from unify.secret_manager.secret_manager import SecretManager


def test_build_sandbox_env_strips_tokens_and_adds_proxy(monkeypatch):
    monkeypatch.setenv("MICROSOFT_ACCESS_TOKEN", "secret-ms")
    monkeypatch.setenv("GOOGLE_REFRESH_TOKEN", "secret-g")
    monkeypatch.setenv("PATH", "/usr/bin")
    sess.set_session(ProxySession(host="127.0.0.1", port=5555, nonce="N"))

    env = build_sandbox_env()

    assert "MICROSOFT_ACCESS_TOKEN" not in env
    assert "GOOGLE_REFRESH_TOKEN" not in env
    assert env["PATH"] == "/usr/bin"
    assert env["MICROSOFT_GRAPH_BASE"].endswith("/microsoft/v1.0")
    assert env["WORKSPACE_PROXY_TOKEN"] == "N"


def test_build_sandbox_env_without_session_still_strips_tokens(monkeypatch):
    monkeypatch.setenv("GOOGLE_ACCESS_TOKEN", "secret-g")
    monkeypatch.setattr(sess, "current_session", lambda: None)

    env = build_sandbox_env()

    assert "GOOGLE_ACCESS_TOKEN" not in env
    assert "WORKSPACE_PROXY_TOKEN" not in env


def test_secret_manager_sensitive_token_names_are_raw_tokens_only():
    names = SecretManager._sensitive_oauth_token_names()
    assert names == {
        "MICROSOFT_ACCESS_TOKEN",
        "MICROSOFT_REFRESH_TOKEN",
        "GOOGLE_ACCESS_TOKEN",
        "GOOGLE_REFRESH_TOKEN",
    }
