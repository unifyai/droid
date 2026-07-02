"""Response filtering for proxied file listings and search results.

Given a raw provider listing/search payload, drop any item the allowlist masks
so the sandbox sees results indistinguishable from the item not existing. Also
provides the ``(drive_id, item_id)`` extraction used across the proxy.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from unify.provider_proxy.ancestry import child_allowed, is_allowed, policy_for
from unify.provider_proxy.classify import Locator

_MY_DRIVE = "my-drive"


def item_key(provider: str, raw: dict[str, Any]) -> Locator:
    """Extract the unified ``(drive_id, item_id)`` from a raw provider item."""
    if provider == "google":
        return Locator(str(raw.get("driveId") or _MY_DRIVE), str(raw.get("id") or ""))
    parent_ref = raw.get("parentReference") or {}
    return Locator(str(parent_ref.get("driveId") or ""), str(raw.get("id") or ""))


def _listing_field(provider: str, payload: dict[str, Any]) -> Optional[str]:
    if provider == "google" and isinstance(payload.get("files"), list):
        return "files"
    if isinstance(payload.get("value"), list):
        return "value"
    return None


async def filter_listing(
    provider: str,
    payload: dict[str, Any],
    parent: Optional[Locator],
    rewrite_url: Callable[[str], str],
) -> dict[str, Any]:
    """Return *payload* with masked items removed.

    When *parent* is known the parent's allow-state is resolved once and each
    child inherits it unless it carries an explicit decision. Otherwise (search,
    recent, sharedWithMe, cross-drive queries) each item is evaluated directly.
    """
    policy = policy_for(provider)
    if policy is None:
        return _rewrite_next_link(payload, rewrite_url)

    field = _listing_field(provider, payload)
    if field is None:
        return _rewrite_next_link(payload, rewrite_url)

    items = payload.get(field) or []
    parent_allowed: Optional[bool] = None
    if parent is not None:
        parent_allowed = await is_allowed(provider, parent.drive_id, parent.item_id)

    visible: list[dict[str, Any]] = []
    for raw in items:
        key = item_key(provider, raw)
        if parent_allowed is not None:
            allowed = child_allowed(policy, key.drive_id, key.item_id, parent_allowed)
        else:
            allowed = await is_allowed(provider, key.drive_id, key.item_id)
        if allowed:
            visible.append(raw)

    payload[field] = visible
    return _rewrite_next_link(payload, rewrite_url)


def _rewrite_next_link(
    payload: dict[str, Any],
    rewrite_url: Callable[[str], str],
) -> dict[str, Any]:
    link = payload.get("@odata.nextLink")
    if isinstance(link, str) and link:
        payload["@odata.nextLink"] = rewrite_url(link)
    return payload
