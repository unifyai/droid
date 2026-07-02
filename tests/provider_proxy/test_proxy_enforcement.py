from __future__ import annotations

import json

import httpx
import pytest

from unify.provider_proxy import filter as flt
from unify.provider_proxy import proxy as pxy
from unify.provider_proxy.policy import get_policy_store
from unify.provider_proxy.session import ProxySession, set_session

NONCE = "test-nonce"
_ALLOWED = {"root", "HR", "HRchild"}


class _Forwarder:
    """Records forwarded calls and returns canned upstream responses."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str]] = []

    async def __call__(
        self,
        provider,
        method,
        rest_path,
        query_string,
        incoming_headers,
        body,
        *,
        follow_redirects=False,
    ):
        self.calls.append((provider, method, rest_path))
        if rest_path == "v1.0/$batch":
            reqs = json.loads(body)["requests"]
            return httpx.Response(
                200,
                json={
                    "responses": [
                        {
                            "id": r["id"],
                            "status": 200,
                            "body": {
                                "value": [
                                    {"id": "HR", "parentReference": {"driveId": "D"}},
                                ],
                            },
                        }
                        for r in reqs
                    ],
                },
            )
        if method == "GET" and rest_path.endswith("/recent"):
            return httpx.Response(
                200,
                json={
                    "value": [
                        {"id": "HR", "parentReference": {"driveId": "D"}},
                        {"id": "FIN", "parentReference": {"driveId": "D"}},
                    ],
                },
            )
        if method == "GET" and rest_path.endswith("/children"):
            return httpx.Response(
                200,
                json={
                    "value": [
                        {"id": "HRchild", "parentReference": {"driveId": "D"}},
                        {"id": "FIN", "parentReference": {"driveId": "D"}},
                    ],
                },
            )
        return httpx.Response(200, json={"ok": rest_path})


# Path -> resolved node for Graph path addressing (None => nonexistent).
_PATH_NODES = {
    "Finance": {"drive_id": "D", "item_id": "FIN"},
    "Finance/report.xlsx": {"drive_id": "D", "item_id": "FIN"},
    "HR": {"drive_id": "D", "item_id": "HR"},
    "HR/notes.txt": {"drive_id": "D", "item_id": "HRchild"},
}


async def _fake_ms_get_by_path(drive_id, anchor_item_id, path):
    return _PATH_NODES.get(path)


async def _is_allowed(provider, drive_id, item_id):
    return item_id in _ALLOWED


@pytest.fixture()
def client(monkeypatch):
    set_session(ProxySession(host="127.0.0.1", port=0, nonce=NONCE))
    get_policy_store().set_policies(
        [{"provider": "microsoft", "default_allow": False, "decisions": []}],
    )
    fwd = _Forwarder()
    monkeypatch.setattr(pxy, "_forward", fwd)
    monkeypatch.setattr(pxy, "is_allowed", _is_allowed)
    monkeypatch.setattr(pxy, "ms_get_by_path", _fake_ms_get_by_path)
    monkeypatch.setattr(flt, "is_allowed", _is_allowed)

    transport = httpx.ASGITransport(app=pxy.build_app())
    c = httpx.AsyncClient(
        transport=transport,
        base_url="http://proxy",
        headers={"Authorization": f"Bearer {NONCE}"},
    )
    c._forwarder = fwd  # type: ignore[attr-defined]
    return c


@pytest.mark.asyncio
async def test_requires_valid_nonce(client):
    resp = await client.get(
        "/microsoft/v1.0/me/drive/recent",
        headers={"Authorization": "Bearer wrong"},
    )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_listing_masks_disallowed_items(client):
    resp = await client.get("/microsoft/v1.0/me/drive/recent")
    assert resp.status_code == 200
    assert [i["id"] for i in resp.json()["value"]] == ["HR"]


@pytest.mark.asyncio
async def test_get_masked_item_is_not_found_without_forwarding(client):
    resp = await client.get("/microsoft/v1.0/drives/D/items/FIN")
    assert resp.status_code == 404
    # Never forwarded upstream.
    assert client._forwarder.calls == []


@pytest.mark.asyncio
async def test_get_allowed_item_is_forwarded(client):
    resp = await client.get("/microsoft/v1.0/drives/D/items/HR")
    assert resp.status_code == 200
    assert client._forwarder.calls[-1][1] == "GET"


@pytest.mark.asyncio
async def test_rename_masked_item_is_blocked(client):
    resp = await client.patch(
        "/microsoft/v1.0/drives/D/items/FIN",
        json={"name": "x"},
    )
    assert resp.status_code == 404
    assert client._forwarder.calls == []


@pytest.mark.asyncio
async def test_rename_allowed_item_is_forwarded(client):
    resp = await client.patch(
        "/microsoft/v1.0/drives/D/items/HR",
        json={"name": "hr2"},
    )
    assert resp.status_code == 200
    assert client._forwarder.calls[-1] == (
        "microsoft",
        "PATCH",
        "v1.0/drives/D/items/HR",
    )


@pytest.mark.asyncio
async def test_create_under_masked_parent_is_forbidden(client):
    resp = await client.post(
        "/microsoft/v1.0/drives/D/items/FIN/children",
        json={"name": "folder", "folder": {}},
    )
    assert resp.status_code == 403
    assert client._forwarder.calls == []


@pytest.mark.asyncio
async def test_unknown_file_endpoint_is_denied(client):
    resp = await client.get("/microsoft/v1.0/me/drive/special")
    assert resp.status_code == 403
    assert client._forwarder.calls == []


@pytest.mark.asyncio
async def test_non_file_endpoint_passes_through(client):
    resp = await client.get("/microsoft/v1.0/me/events")
    assert resp.status_code == 200
    assert resp.json() == {"ok": "v1.0/me/events"}
    assert client._forwarder.calls[-1] == ("microsoft", "GET", "v1.0/me/events")


@pytest.mark.asyncio
async def test_path_get_masked_file_is_not_found(client):
    resp = await client.get("/microsoft/v1.0/me/drive/root:/Finance/report.xlsx")
    assert resp.status_code == 404
    assert client._forwarder.calls == []


@pytest.mark.asyncio
async def test_path_get_allowed_file_is_forwarded(client):
    resp = await client.get("/microsoft/v1.0/me/drive/root:/HR/notes.txt")
    assert resp.status_code == 200
    assert client._forwarder.calls != []


@pytest.mark.asyncio
async def test_path_create_new_file_in_allowed_folder_is_forwarded(client):
    # New file (path does not resolve) inside an allowed folder -> permitted.
    resp = await client.put(
        "/microsoft/v1.0/me/drive/root:/HR/new.txt:/content",
        content=b"data",
    )
    assert resp.status_code == 200
    assert client._forwarder.calls[-1][1] == "PUT"


@pytest.mark.asyncio
async def test_path_create_new_file_in_masked_folder_is_forbidden(client):
    resp = await client.put(
        "/microsoft/v1.0/me/drive/root:/Finance/new.txt:/content",
        content=b"data",
    )
    assert resp.status_code == 403
    assert client._forwarder.calls == []


@pytest.mark.asyncio
async def test_path_edit_existing_masked_file_is_blocked(client):
    resp = await client.patch(
        "/microsoft/v1.0/me/drive/root:/Finance/report.xlsx",
        json={"name": "x"},
    )
    assert resp.status_code == 404
    assert client._forwarder.calls == []


@pytest.mark.asyncio
async def test_path_children_in_masked_folder_are_filtered_out(client):
    resp = await client.get("/microsoft/v1.0/me/drive/root:/Finance:/children")
    assert resp.status_code == 200
    # Parent folder is masked, so children with no explicit allow are hidden.
    assert resp.json()["value"] == []


@pytest.mark.asyncio
async def test_batch_denies_masked_and_filters_forwarded(client):
    body = {
        "requests": [
            {"id": "1", "method": "GET", "url": "/drives/D/items/FIN"},
            {"id": "2", "method": "GET", "url": "/me/drive/recent"},
        ],
    }
    resp = await client.post("/microsoft/v1.0/$batch", json=body)
    assert resp.status_code == 200
    by_id = {r["id"]: r for r in resp.json()["responses"]}
    # Masked direct read is synthesized as 404 and never forwarded.
    assert by_id["1"]["status"] == 404
    # Forwarded listing is filtered.
    assert by_id["2"]["status"] == 200
    assert [i["id"] for i in by_id["2"]["body"]["value"]] == ["HR"]
