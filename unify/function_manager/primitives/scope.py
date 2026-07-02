"""PrimitiveScope: the single knob for controlling which managers are exposed."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import FrozenSet

# Canonical manager aliases - the only valid values for scoped_managers.
# This is the authoritative list; ToolSurfaceRegistry.MANAGERS must match.
VALID_MANAGER_ALIASES: frozenset[str] = frozenset(
    {
        "comms",
        "contacts",
        "dashboards",
        "tasks",
        "transcripts",
        "knowledge",
        "secrets",
        "web",
        "data",
        "files",
        "workspace_email",
        "integrations",
        "computer",
        "actor",
        "coordinator",
    },
)
COORDINATOR_MANAGER_ALIAS = "coordinator"
_NON_COORDINATOR_MANAGER_ALIASES: frozenset[str] = frozenset(
    alias for alias in VALID_MANAGER_ALIASES if alias != COORDINATOR_MANAGER_ALIAS
)
_SCOPED_MANAGERS_BY_ROLE: dict[bool, frozenset[str]] = {
    True: VALID_MANAGER_ALIASES,
    False: _NON_COORDINATOR_MANAGER_ALIASES,
}


@dataclass(frozen=True, slots=True)
class PrimitiveScope:
    """
    Defines which managers are exposed in a deployment.

    This is the single source of truth for scoping. All downstream consumers
    (Primitives, StateManagerEnvironment, FunctionManager, prompt builders)
    read from this object.

    Attributes
    ----------
    scoped_managers : frozenset[str]
        Set of manager aliases to expose. Must be non-empty and contain only
        valid manager aliases from VALID_MANAGER_ALIASES.

    Examples
    --------
    # Files-only deployment
    scope = PrimitiveScope(scoped_managers=frozenset({"files"}))

    # Full deployment
    scope = PrimitiveScope(scoped_managers=VALID_MANAGER_ALIASES)
    """

    scoped_managers: "FrozenSet[str]"

    def __post_init__(self) -> None:
        """Validate scoped_managers."""
        if not self.scoped_managers:
            raise ValueError("scoped_managers must be non-empty")

        invalid = self.scoped_managers - VALID_MANAGER_ALIASES
        if invalid:
            raise ValueError(
                f"Invalid manager aliases: {sorted(invalid)}. "
                f"Valid aliases: {sorted(VALID_MANAGER_ALIASES)}",
            )

    @property
    def scope_key(self) -> str:
        """
        Stable, deterministic key for caching and registry lookups.

        Returns a sorted comma-separated string of manager aliases.
        """
        return ",".join(sorted(self.scoped_managers))

    def includes(self, manager_alias: str) -> bool:
        """Check if a manager alias is in scope."""
        return manager_alias in self.scoped_managers

    @classmethod
    def all_managers(cls) -> "PrimitiveScope":
        """Create a scope with all managers exposed."""
        return cls(scoped_managers=VALID_MANAGER_ALIASES)

    @classmethod
    def single(cls, manager_alias: str) -> "PrimitiveScope":
        """Create a scope with a single manager exposed."""
        return cls(scoped_managers=frozenset({manager_alias}))


def scoped_managers_for_role(*, is_coordinator: bool) -> frozenset[str]:
    """Return the canonical manager alias set for the active role."""
    return _SCOPED_MANAGERS_BY_ROLE[bool(is_coordinator)]


_DEFAULT_RUNTIME_SCOPES: dict[bool, PrimitiveScope] = {
    role: PrimitiveScope(scoped_managers=aliases)
    for role, aliases in _SCOPED_MANAGERS_BY_ROLE.items()
}


def default_runtime_scope(*, is_coordinator: bool | None = None) -> PrimitiveScope:
    """Return the role-scoped default primitive scope for runtime usage."""
    if is_coordinator is None:
        from unify.session_details import SESSION_DETAILS

        is_coordinator = bool(getattr(SESSION_DETAILS, "is_coordinator", False))
    return _DEFAULT_RUNTIME_SCOPES[bool(is_coordinator)]
