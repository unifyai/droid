from __future__ import annotations

import httpx
import pytest

from unify.common import runtime_oauth
from unify.provider_proxy import proxy as pxy


def _client_factory(handler):
    def _make(follow_redirects: bool) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(handler))

    return _make


@pytest.mark.asyncio
async def test_forward_happy_path_no_refresh(monkeypatch):
    seen: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers.get("authorization"))
        return httpx.Response(200, json={"ok": True})

    monkeypatch.setattr(
        runtime_oauth,
        "get_provider_access_token_optimistic",
        lambda p: "good",
    )

    def _no_refresh(_p):
        raise AssertionError("refresh must not be called on 2xx")

    monkeypatch.setattr(runtime_oauth, "refresh_provider_access_token", _no_refresh)
    monkeypatch.setattr(pxy, "_make_client", _client_factory(handler))

    resp = await pxy._forward("microsoft", "GET", "v1.0/me/events", "", {}, None)
    assert resp.status_code == 200
    assert seen == ["Bearer good"]


@pytest.mark.asyncio
async def test_forward_retries_once_on_401_then_succeeds(monkeypatch):
    seen: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        auth = request.headers.get("authorization")
        seen.append(auth)
        if auth == "Bearer fresh":
            return httpx.Response(200, json={"ok": True})
        return httpx.Response(401, json={"error": "expired"})

    monkeypatch.setattr(
        runtime_oauth,
        "get_provider_access_token_optimistic",
        lambda p: "stale",
    )
    monkeypatch.setattr(
        runtime_oauth,
        "refresh_provider_access_token",
        lambda p: "fresh",
    )
    monkeypatch.setattr(pxy, "_make_client", _client_factory(handler))

    resp = await pxy._forward("microsoft", "GET", "v1.0/me/drive/root", "", {}, None)
    assert resp.status_code == 200
    assert seen == ["Bearer stale", "Bearer fresh"]


@pytest.mark.asyncio
async def test_forward_persistent_401_is_normalized_to_reconnect(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "nope"})

    monkeypatch.setattr(
        runtime_oauth,
        "get_provider_access_token_optimistic",
        lambda p: "t1",
    )
    monkeypatch.setattr(
        runtime_oauth,
        "refresh_provider_access_token",
        lambda p: "t2",
    )
    monkeypatch.setattr(pxy, "_make_client", _client_factory(handler))

    resp = await pxy._forward("google", "GET", "drive/v3/files", "", {}, None)
    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "authenticationRequired"


@pytest.mark.asyncio
async def test_forward_no_token_returns_reconnect_without_calling_upstream(monkeypatch):
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200)

    monkeypatch.setattr(
        runtime_oauth,
        "get_provider_access_token_optimistic",
        lambda p: None,
    )
    monkeypatch.setattr(runtime_oauth, "refresh_provider_access_token", lambda p: None)
    monkeypatch.setattr(pxy, "_make_client", _client_factory(handler))

    resp = await pxy._forward("microsoft", "GET", "v1.0/me/drive/root", "", {}, None)
    assert resp.status_code == 401
    assert calls["n"] == 0
    assert resp.json()["error"]["code"] == "authenticationRequired"
