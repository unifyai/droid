"""Workspace file-access policy: semantics + runtime cache.

The allowlist is configured in Console and stored in Orchestra. This module
holds the runtime-side mirror: the pure evaluation logic (kept identical to
Orchestra's ``orchestra/web/api/assistant/file_access.py``) plus an in-memory
``PolicyStore`` populated by the assistant-secret/config sync.

Access for any item is resolved by walking from the item up its parent chain:
the nearest ancestor (or the item itself) carrying an explicit decision wins;
absent any decision, ``default_allow`` applies. A provider with *no* stored
policy is treated as unrestricted (the allowlist is opt-in), so connecting an
account does not silently sever access until the user configures restrictions.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Iterable

PROVIDERS = ("google", "microsoft")


def decision_key(drive_id: str, item_id: str) -> tuple[str, str]:
    """Stable lookup key for a decision / tree node."""
    return (drive_id or "", item_id or "")


def index_decisions(decisions: Iterable[dict[str, Any]]) -> dict[tuple[str, str], bool]:
    """Build a ``{(drive_id, item_id): allow}`` lookup from raw decisions."""
    index: dict[tuple[str, str], bool] = {}
    for entry in decisions or []:
        key = decision_key(entry.get("drive_id", ""), entry.get("item_id", ""))
        index[key] = bool(entry.get("allow"))
    return index


def evaluate_access(
    decisions: Iterable[dict[str, Any]],
    default_allow: bool,
    parent_chain: list[tuple[str, str]],
) -> bool:
    """Return ``True`` if the item at the head of *parent_chain* is allowed.

    ``parent_chain`` is ordered from the item itself outward to its root
    ancestor as ``(drive_id, item_id)`` tuples. The nearest entry carrying an
    explicit decision wins; absent any decision, ``default_allow`` applies.
    """
    index = index_decisions(decisions)
    for key in parent_chain:
        if key in index:
            return index[key]
    return bool(default_allow)


@dataclass(frozen=True)
class WorkspaceFilePolicy:
    """A single provider's resolved allowlist."""

    provider: str
    default_allow: bool = False
    decisions: tuple[dict[str, Any], ...] = field(default_factory=tuple)

    def allows(self, parent_chain: list[tuple[str, str]]) -> bool:
        """Evaluate access for an item given its (item-first) parent chain."""
        return evaluate_access(self.decisions, self.default_allow, parent_chain)

    def decision_for(self, drive_id: str, item_id: str) -> bool | None:
        """Return the explicit decision for an item, or ``None`` if unspecified."""
        return index_decisions(self.decisions).get(decision_key(drive_id, item_id))


class PolicyStore:
    """Thread-safe in-memory cache of per-provider allowlists.

    Populated by the config sync (see ``secret_manager``). ``get`` returns
    ``None`` for providers with no configured policy, which callers must treat
    as unrestricted access.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._by_provider: dict[str, WorkspaceFilePolicy] = {}

    def set_policies(self, policies: Iterable[dict[str, Any]]) -> None:
        """Replace the cache from a list of ``{provider, default_allow, decisions}``."""
        parsed: dict[str, WorkspaceFilePolicy] = {}
        for entry in policies or []:
            provider = entry.get("provider")
            if provider not in PROVIDERS:
                continue
            parsed[provider] = WorkspaceFilePolicy(
                provider=provider,
                default_allow=bool(entry.get("default_allow", False)),
                decisions=tuple(entry.get("decisions") or ()),
            )
        with self._lock:
            self._by_provider = parsed

    def get(self, provider: str) -> WorkspaceFilePolicy | None:
        with self._lock:
            return self._by_provider.get(provider)

    def clear(self) -> None:
        with self._lock:
            self._by_provider = {}


_POLICY_STORE = PolicyStore()


def get_policy_store() -> PolicyStore:
    """Return the process-wide policy store singleton."""
    return _POLICY_STORE
