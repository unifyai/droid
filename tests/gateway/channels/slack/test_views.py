"""Behavioural tests for ``unify.gateway.channels.slack.views``.

Covers the router contract plus the ``users.lookupByEmail`` reverse
lookup (helper + ``POST /user-by-email`` endpoint) that lets the
outbound pipeline reach a workspace member the bot has never received a
message from. The Slack HTTP call and the Orchestra bot-token resolution
are mocked -- this is a transport/auth proxy test, not a Slack behaviour
test.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from unify.gateway.channels.slack import auth_router
from unify.gateway.channels.slack import views as slack_views

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def _settings(monkeypatch: pytest.MonkeyPatch) -> None:
    stub_secret = SimpleNamespace(get_secret_value=lambda: "test-admin-key")
    monkeypatch.setattr(
        slack_views,
        "SETTINGS",
        SimpleNamespace(
            ORCHESTRA_URL="https://orchestra.example.com/v0",
            ORCHESTRA_ADMIN_KEY=stub_secret,
        ),
    )


@pytest.fixture
def app() -> FastAPI:
    app = FastAPI()
    app.include_router(auth_router, prefix="/slack")
    return app


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    return TestClient(app)


def _slack_response(payload: dict) -> MagicMock:
    resp = MagicMock(status_code=200)
    resp.json.return_value = payload
    return resp


def _async_httpx_client(response_mock: MagicMock) -> MagicMock:
    client = AsyncMock()
    client.__aenter__.return_value = client
    client.get.return_value = response_mock
    return client


# ---------------------------------------------------------------------------
# Router contract
# ---------------------------------------------------------------------------


def test_router_exposes_expected_paths() -> None:
    paths = sorted(
        (r.path, sorted(r.methods)) for r in auth_router.routes  # type: ignore[attr-defined]
    )
    assert paths == [
        ("/install", ["POST"]),
        ("/send", ["POST"]),
        ("/status", ["GET"]),
        ("/user-by-email", ["POST"]),
        ("/user-info", ["POST"]),
    ]


def test_router_importable_from_package_root() -> None:
    from unify.gateway.channels.slack import auth_router as exported

    assert exported is auth_router


# ---------------------------------------------------------------------------
# lookup_slack_user_id_by_email
# ---------------------------------------------------------------------------


class TestLookupSlackUserIdByEmail:
    @pytest.mark.asyncio
    async def test_resolves_id_on_ok(self) -> None:
        with (
            patch.object(
                slack_views,
                "_resolve_bot_token",
                new=AsyncMock(return_value="bot-token"),
            ),
            patch.object(
                slack_views.httpx,
                "AsyncClient",
                return_value=_async_httpx_client(
                    _slack_response({"ok": True, "user": {"id": "U123"}}),
                ),
            ),
        ):
            result = await slack_views.lookup_slack_user_id_by_email(
                "T1",
                "alice@example.com",
            )
        assert result == "U123"

    @pytest.mark.asyncio
    async def test_returns_none_when_user_not_found(self) -> None:
        with (
            patch.object(
                slack_views,
                "_resolve_bot_token",
                new=AsyncMock(return_value="bot-token"),
            ),
            patch.object(
                slack_views.httpx,
                "AsyncClient",
                return_value=_async_httpx_client(
                    _slack_response({"ok": False, "error": "users_not_found"}),
                ),
            ),
        ):
            result = await slack_views.lookup_slack_user_id_by_email(
                "T1",
                "ghost@example.com",
            )
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_scope_missing(self) -> None:
        with (
            patch.object(
                slack_views,
                "_resolve_bot_token",
                new=AsyncMock(return_value="bot-token"),
            ),
            patch.object(
                slack_views.httpx,
                "AsyncClient",
                return_value=_async_httpx_client(
                    _slack_response({"ok": False, "error": "missing_scope"}),
                ),
            ),
        ):
            result = await slack_views.lookup_slack_user_id_by_email(
                "T1",
                "alice@example.com",
            )
        assert result is None

    @pytest.mark.asyncio
    async def test_empty_email_short_circuits_without_token_resolution(self) -> None:
        resolve = AsyncMock(return_value="bot-token")
        with patch.object(slack_views, "_resolve_bot_token", new=resolve):
            result = await slack_views.lookup_slack_user_id_by_email("T1", "")
        assert result is None
        resolve.assert_not_awaited()


# ---------------------------------------------------------------------------
# POST /user-by-email
# ---------------------------------------------------------------------------


class TestUserByEmailEndpoint:
    def test_returns_resolved_id(self, client: TestClient, _settings: None) -> None:
        with (
            patch.object(
                slack_views,
                "_resolve_bot_token",
                new=AsyncMock(return_value="bot-token"),
            ),
            patch.object(
                slack_views.httpx,
                "AsyncClient",
                return_value=_async_httpx_client(
                    _slack_response({"ok": True, "user": {"id": "U123"}}),
                ),
            ),
        ):
            resp = client.post(
                "/slack/user-by-email",
                json={"team_id": "T1", "email": "alice@example.com"},
            )
        assert resp.status_code == 200
        assert resp.json() == {"slack_user_id": "U123"}

    def test_returns_null_when_unresolved(
        self,
        client: TestClient,
        _settings: None,
    ) -> None:
        with (
            patch.object(
                slack_views,
                "_resolve_bot_token",
                new=AsyncMock(return_value="bot-token"),
            ),
            patch.object(
                slack_views.httpx,
                "AsyncClient",
                return_value=_async_httpx_client(
                    _slack_response({"ok": False, "error": "users_not_found"}),
                ),
            ),
        ):
            resp = client.post(
                "/slack/user-by-email",
                json={"team_id": "T1", "email": "ghost@example.com"},
            )
        assert resp.status_code == 200
        assert resp.json() == {"slack_user_id": None}
