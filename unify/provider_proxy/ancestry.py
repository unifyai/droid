"""Provider item resolution + allowlist evaluation for the proxy.

Runs in the trusted runtime and holds the real provider OAuth token via
:func:`unify.common.runtime_oauth.get_provider_access_token`. Given a
``(drive_id, item_id)`` the helpers here resolve the item's ancestry chain from
the provider and evaluate it against the per-assistant allowlist in the
``PolicyStore``. The proxy uses these to decide whether a Drive/Graph item is
visible (reads) or writable (mutations), independently of the exact REST shape
the sandbox used.
"""

from __future__ import annotations

import threading
from typing import Any, Optional

import httpx

from unify.common.runtime_oauth import get_provider_access_token
from unify.provider_proxy.policy import WorkspaceFilePolicy, get_policy_store

_FOLDER_MIME = "application/vnd.google-apps.folder"
_GOOGLE_BASE = "https://www.googleapis.com/drive/v3"
_GRAPH_BASE = "https://graph.microsoft.com/v1.0"
_MY_DRIVE = "my-drive"
_MS_DEFAULT_DRIVE_ALIASES = frozenset({"", "me", "my-drive", "root", "default"})
_MAX_ANCESTRY_DEPTH = 40

# Module-level ancestry cache; the proxy process is scoped to a single
# assistant session, so a per-process cache is safe and short-lived.
_ANCESTRY_CACHE: dict[tuple[str, str, str], list[tuple[str, str]]] = {}
_CACHE_LOCK = threading.Lock()


class WorkspaceFileNotFound(Exception):
    """Raised when an item is absent or masked by the allowlist."""


def _headers(provider: str) -> dict[str, str]:
    token = get_provider_access_token(provider)
    return {"Authorization": f"Bearer {token}"}


# ── Google ───────────────────────────────────────────────────────────────


def _google_node(raw: dict[str, Any], drive_id: str) -> dict[str, Any]:
    parents = raw.get("parents") or []
    return {
        "drive_id": drive_id,
        "item_id": raw.get("id"),
        "name": raw.get("name"),
        "kind": "folder" if raw.get("mimeType") == _FOLDER_MIME else "file",
        "mime_type": raw.get("mimeType"),
        "web_url": raw.get("webViewLink"),
        "parent_id": parents[0] if parents else None,
    }


async def google_get(drive_id: str, item_id: str) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=30) as http:
        resp = await http.get(
            f"{_GOOGLE_BASE}/files/{item_id}",
            params={
                "fields": "id,name,mimeType,parents,webViewLink",
                "supportsAllDrives": "true",
            },
            headers=_headers("google"),
        )
    if resp.status_code == 404:
        raise WorkspaceFileNotFound(item_id)
    resp.raise_for_status()
    return _google_node(resp.json(), drive_id)


# ── Microsoft ──────────────────────────────────────────────────────────────


def _ms_node(raw: dict[str, Any], fallback_drive_id: str) -> dict[str, Any]:
    parent_ref = raw.get("parentReference") or {}
    drive_id = parent_ref.get("driveId") or fallback_drive_id
    return {
        "drive_id": drive_id,
        "item_id": raw.get("id"),
        "name": raw.get("name"),
        "kind": "folder" if raw.get("folder") else "file",
        "mime_type": (raw.get("file") or {}).get("mimeType"),
        "web_url": raw.get("webUrl"),
        "parent_id": parent_ref.get("id"),
    }


def _ms_item_url(drive_id: str, item_id: str) -> str:
    default_drive = drive_id in _MS_DEFAULT_DRIVE_ALIASES
    root_item = not item_id or item_id in ("root", "")
    if default_drive:
        base = f"{_GRAPH_BASE}/me/drive"
    else:
        base = f"{_GRAPH_BASE}/drives/{drive_id}"
    return f"{base}/root" if root_item else f"{base}/items/{item_id}"


async def ms_get(drive_id: str, item_id: str) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=30) as http:
        resp = await http.get(
            _ms_item_url(drive_id, item_id),
            params={"$select": "id,name,folder,file,parentReference,webUrl"},
            headers=_headers("microsoft"),
        )
    if resp.status_code == 404:
        raise WorkspaceFileNotFound(item_id)
    resp.raise_for_status()
    return _ms_node(resp.json(), drive_id)


def _ms_drive_base(drive_id: str) -> str:
    if drive_id in _MS_DEFAULT_DRIVE_ALIASES:
        return f"{_GRAPH_BASE}/me/drive"
    return f"{_GRAPH_BASE}/drives/{drive_id}"


def _ms_path_url(drive_id: str, anchor_item_id: Optional[str], path: str) -> str:
    base = _ms_drive_base(drive_id)
    anchor = (
        "root"
        if not anchor_item_id or anchor_item_id == "root"
        else f"items/{anchor_item_id}"
    )
    if path:
        return f"{base}/{anchor}:/{path}"
    return f"{base}/{anchor}"


async def ms_get_by_path(
    drive_id: str,
    anchor_item_id: Optional[str],
    path: str,
) -> Optional[dict[str, Any]]:
    """Resolve a Graph path-addressed item to a normalized node, or None if 404.

    ``path`` is kept URL-encoded exactly as received. Default fields (including
    ``parentReference``) are returned, so no ``$select`` (which would require the
    trailing-colon path-query form) is needed.
    """
    async with httpx.AsyncClient(timeout=30) as http:
        resp = await http.get(
            _ms_path_url(drive_id, anchor_item_id, path),
            headers=_headers("microsoft"),
        )
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return _ms_node(resp.json(), drive_id)


def parent_path(path: str) -> str:
    """Return the parent path of a ``/``-separated path (``""`` for top level)."""
    trimmed = (path or "").strip("/")
    if "/" in trimmed:
        return trimmed.rsplit("/", 1)[0]
    return ""


# ── Ancestry + enforcement ──────────────────────────────────────────────────


async def get_node(provider: str, drive_id: str, item_id: str) -> dict[str, Any]:
    if provider == "google":
        return await google_get(drive_id, item_id)
    return await ms_get(drive_id, item_id)


async def ancestry_chain(
    provider: str,
    drive_id: str,
    item_id: str,
) -> list[tuple[str, str]]:
    """Return ``(drive_id, id)`` tuples from *item_id* outward to the root.

    Uses provider-reported ids so the chain keys match Console-configured
    decisions even when the request used a ``me/drive`` alias.
    """
    cache_key = (provider, drive_id or "", item_id or "")
    with _CACHE_LOCK:
        cached = _ANCESTRY_CACHE.get(cache_key)
    if cached is not None:
        return cached

    chain: list[tuple[str, str]] = []
    current_drive = drive_id
    current_item = item_id
    seen: set[str] = set()
    for _ in range(_MAX_ANCESTRY_DEPTH):
        if not current_item or current_item in seen:
            break
        seen.add(current_item)
        try:
            node = await get_node(provider, current_drive, current_item)
        except WorkspaceFileNotFound:
            break
        resolved_drive = node.get("drive_id") or current_drive
        resolved_item = node.get("item_id") or current_item
        chain.append((resolved_drive or "", resolved_item))
        parent = node.get("parent_id")
        if not parent or parent == resolved_item:
            break
        current_drive = resolved_drive
        current_item = parent

    with _CACHE_LOCK:
        _ANCESTRY_CACHE[cache_key] = chain
    return chain


def policy_for(provider: str) -> Optional[WorkspaceFilePolicy]:
    return get_policy_store().get(provider)


async def is_allowed(provider: str, drive_id: str, item_id: str) -> bool:
    """Whether an item is accessible under the current allowlist.

    A provider with no configured policy is unrestricted (opt-in restriction).
    """
    policy = policy_for(provider)
    if policy is None:
        return True
    chain = await ancestry_chain(provider, drive_id, item_id)
    return policy.allows(chain)


def child_allowed(
    policy: WorkspaceFilePolicy,
    drive_id: str,
    child_id: str,
    parent_allowed: bool,
) -> bool:
    """Resolve a child's visibility given its parent's allow state.

    An explicit decision on the child wins; otherwise it inherits the parent.
    """
    explicit = policy.decision_for(drive_id, child_id)
    return explicit if explicit is not None else parent_allowed


def clear_ancestry_cache() -> None:
    with _CACHE_LOCK:
        _ANCESTRY_CACHE.clear()
