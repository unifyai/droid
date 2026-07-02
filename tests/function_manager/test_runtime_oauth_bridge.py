from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from unify.function_manager.function_manager import FunctionManager
from unify.function_manager import function_manager as function_manager_module


@dataclass
class _FakeSecretManager:
    sync_reasons: list[str] = field(default_factory=list)

    def _get_secret_value(self, name: str) -> str | None:
        values = {
            "GOOGLE_ACCESS_TOKEN": "google-token",
            "GOOGLE_TOKEN_EXPIRES_AT": "2999-01-01T00:00:00+00:00",
        }
        return values.get(name)

    def sync_assistant_secrets_if_stale(self, **kwargs) -> bool:
        self.sync_reasons.append(kwargs["reason"])
        return True


@pytest.mark.asyncio
async def test_shell_runtime_oauth_token_helper_uses_parent_rpc(monkeypatch):
    fake_secret_manager = _FakeSecretManager()
    monkeypatch.setattr(
        function_manager_module.ManagerRegistry,
        "get_secret_manager",
        lambda: fake_secret_manager,
    )
    # The sandbox-facing helper now returns the local proxy capability handle,
    # not a raw token. The parent RPC resolves it via the proxy session.
    from unify.provider_proxy.session import ProxySession

    session = ProxySession(host="127.0.0.1", port=4321, nonce="proxy-nonce")
    monkeypatch.setattr(
        "unify.provider_proxy.proxy.ensure_proxy_running",
        lambda: session,
    )

    fm = object.__new__(FunctionManager)
    result = await fm.execute_shell_script(
        implementation=(
            "#!/bin/sh\n"
            "token=$(unity-primitive runtime get_oauth_access_token "
            "--provider google --min_ttl_seconds 42)\n"
            'if [ "$token" = \'"proxy-nonce"\' ]; then echo "TOKEN_OK"; fi\n'
        ),
        language="sh",
    )

    assert result["error"] is None
    assert result["result"] == 0
    assert "TOKEN_OK" in result["stdout"]
    # The handle path does not touch the assistant-secret sync.
    assert fake_secret_manager.sync_reasons == []


def test_runtime_oauth_env_overlay_routes_through_runtime_helper(monkeypatch):
    from unify.common import runtime_oauth

    monkeypatch.setattr(
        runtime_oauth,
        "get_refresh_token_oauth_env_overlay",
        lambda: {"GOOGLE_ACCESS_TOKEN": "fresh-google-token"},
    )

    fm = object.__new__(FunctionManager)

    assert fm._get_runtime_oauth_env_overlay() == {
        "GOOGLE_ACCESS_TOKEN": "fresh-google-token",
    }


def test_venv_runtime_oauth_helper_uses_parent_rpc(monkeypatch):
    from unify.function_manager import venv_runner

    calls = []

    def fake_rpc_call_sync(path, kwargs):
        calls.append((path, kwargs))
        return "fresh-ms-token"

    monkeypatch.setattr(venv_runner, "rpc_call_sync", fake_rpc_call_sync)

    assert (
        venv_runner.get_oauth_access_token("microsoft", min_ttl_seconds=12)
        == "fresh-ms-token"
    )
    assert calls == [
        (
            "runtime.get_oauth_access_token",
            {"provider": "microsoft", "min_ttl_seconds": 12},
        ),
    ]
