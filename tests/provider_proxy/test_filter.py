from __future__ import annotations

import pytest

from unify.provider_proxy import filter as flt
from unify.provider_proxy.classify import Locator
from unify.provider_proxy.policy import get_policy_store


@pytest.fixture(autouse=True)
def _reset_policy():
    get_policy_store().clear()
    yield
    get_policy_store().clear()


def _identity(url: str) -> str:
    return url


def _ms_item(item_id: str, drive_id: str = "D") -> dict:
    return {"id": item_id, "parentReference": {"driveId": drive_id}}


@pytest.mark.asyncio
async def test_no_policy_returns_payload_unchanged(monkeypatch):
    payload = {"value": [_ms_item("HR"), _ms_item("FIN")]}
    out = await flt.filter_listing(
        "microsoft",
        payload,
        Locator("D", "root"),
        _identity,
    )
    assert [i["id"] for i in out["value"]] == ["HR", "FIN"]


@pytest.mark.asyncio
async def test_children_inherit_parent_and_explicit_deny_masks(monkeypatch):
    get_policy_store().set_policies(
        [
            {
                "provider": "microsoft",
                "default_allow": False,
                "decisions": [
                    {"drive_id": "D", "item_id": "root", "allow": True},
                    {"drive_id": "D", "item_id": "FIN", "allow": False},
                ],
            },
        ],
    )
    # Parent (root) is allowed; children with no explicit decision inherit it.
    monkeypatch.setattr(flt, "is_allowed", _always(True))

    payload = {"value": [_ms_item("HR"), _ms_item("FIN"), _ms_item("X")]}
    out = await flt.filter_listing(
        "microsoft",
        payload,
        Locator("D", "root"),
        _identity,
    )

    # HR + X inherit allowed parent; FIN is explicitly denied.
    assert [i["id"] for i in out["value"]] == ["HR", "X"]


@pytest.mark.asyncio
async def test_children_hidden_when_parent_not_allowed(monkeypatch):
    get_policy_store().set_policies(
        [{"provider": "microsoft", "default_allow": False, "decisions": []}],
    )
    monkeypatch.setattr(flt, "is_allowed", _always(False))

    payload = {"value": [_ms_item("A"), _ms_item("B")]}
    out = await flt.filter_listing(
        "microsoft",
        payload,
        Locator("D", "root"),
        _identity,
    )
    assert out["value"] == []


@pytest.mark.asyncio
async def test_per_item_filtering_when_no_parent(monkeypatch):
    get_policy_store().set_policies(
        [{"provider": "microsoft", "default_allow": False, "decisions": []}],
    )

    async def _is_allowed(provider, drive_id, item_id):
        return item_id == "HR"

    monkeypatch.setattr(flt, "is_allowed", _is_allowed)

    payload = {"value": [_ms_item("HR"), _ms_item("FIN")]}
    out = await flt.filter_listing("microsoft", payload, None, _identity)
    assert [i["id"] for i in out["value"]] == ["HR"]


@pytest.mark.asyncio
async def test_google_files_field_filtered(monkeypatch):
    get_policy_store().set_policies(
        [{"provider": "google", "default_allow": False, "decisions": []}],
    )

    async def _is_allowed(provider, drive_id, item_id):
        return item_id == "keep"

    monkeypatch.setattr(flt, "is_allowed", _is_allowed)

    payload = {"files": [{"id": "keep"}, {"id": "drop"}]}
    out = await flt.filter_listing("google", payload, None, _identity)
    assert [f["id"] for f in out["files"]] == ["keep"]


@pytest.mark.asyncio
async def test_next_link_is_rewritten(monkeypatch):
    get_policy_store().set_policies(
        [{"provider": "microsoft", "default_allow": True, "decisions": []}],
    )
    monkeypatch.setattr(flt, "is_allowed", _always(True))

    payload = {
        "value": [_ms_item("A")],
        "@odata.nextLink": "https://graph.microsoft.com/v1.0/me/drive/root/children?$skip=1",
    }

    def rewrite(url: str) -> str:
        return url.replace(
            "https://graph.microsoft.com",
            "http://127.0.0.1:9/microsoft",
        )

    out = await flt.filter_listing("microsoft", payload, Locator("D", "root"), rewrite)
    assert out["@odata.nextLink"].startswith("http://127.0.0.1:9/microsoft/")


def _always(value: bool):
    async def _fn(provider, drive_id, item_id):
        return value

    return _fn
