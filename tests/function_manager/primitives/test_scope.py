"""Tests for PrimitiveScope."""

import pytest

from unify.function_manager.primitives.scope import (
    PrimitiveScope,
    VALID_MANAGER_ALIASES,
    default_runtime_scope,
    scoped_managers_for_role,
)
from unify.session_details import SESSION_DETAILS

# ────────────────────────────────────────────────────────────────────────────
# PrimitiveScope validation tests
# ────────────────────────────────────────────────────────────────────────────


def test_valid_single_manager():
    """Can create scope with a single valid manager."""
    scope = PrimitiveScope(scoped_managers=frozenset({"files"}))
    assert scope.scoped_managers == frozenset({"files"})
    assert scope.includes("files")
    assert not scope.includes("contacts")


def test_valid_multiple_managers():
    """Can create scope with multiple valid managers."""
    scope = PrimitiveScope(scoped_managers=frozenset({"files", "contacts", "tasks"}))
    assert len(scope.scoped_managers) == 3
    assert scope.includes("files")
    assert scope.includes("contacts")
    assert scope.includes("tasks")
    assert not scope.includes("web")


def test_invalid_manager_raises():
    """Invalid manager alias raises ValueError."""
    with pytest.raises(ValueError, match="Invalid manager aliases"):
        PrimitiveScope(scoped_managers=frozenset({"invalid_manager"}))


def test_mixed_valid_invalid_raises():
    """Mixing valid and invalid aliases raises ValueError."""
    with pytest.raises(ValueError, match="Invalid manager aliases"):
        PrimitiveScope(scoped_managers=frozenset({"files", "not_a_manager"}))


def test_empty_scope_raises():
    """Empty scope raises ValueError."""
    with pytest.raises(ValueError, match="must be non-empty"):
        PrimitiveScope(scoped_managers=frozenset())


def test_scope_key_is_deterministic():
    """scope_key is deterministic regardless of insertion order."""
    scope1 = PrimitiveScope(scoped_managers=frozenset({"files", "contacts", "tasks"}))
    scope2 = PrimitiveScope(scoped_managers=frozenset({"tasks", "files", "contacts"}))
    assert scope1.scope_key == scope2.scope_key
    assert scope1.scope_key == "contacts,files,tasks"


def test_scope_key_single_manager():
    """scope_key works for single manager."""
    scope = PrimitiveScope.single("files")
    assert scope.scope_key == "files"


# ────────────────────────────────────────────────────────────────────────────
# Factory method tests
# ────────────────────────────────────────────────────────────────────────────


def test_all_managers_factory():
    """all_managers() creates scope with all valid managers."""
    scope = PrimitiveScope.all_managers()
    assert scope.scoped_managers == VALID_MANAGER_ALIASES


def test_single_factory():
    """single() creates scope with one manager."""
    scope = PrimitiveScope.single("files")
    assert scope.scoped_managers == frozenset({"files"})


def test_single_factory_invalid_raises():
    """single() with invalid manager raises ValueError."""
    with pytest.raises(ValueError, match="Invalid manager aliases"):
        PrimitiveScope.single("not_a_manager")


def test_single_factory_for_each_manager():
    """single() works for every valid manager alias."""
    for alias in VALID_MANAGER_ALIASES:
        scope = PrimitiveScope.single(alias)
        assert scope.scoped_managers == frozenset({alias})
        assert scope.includes(alias)


# ────────────────────────────────────────────────────────────────────────────
# Immutability and equality tests
# ────────────────────────────────────────────────────────────────────────────


def test_frozen_immutable():
    """PrimitiveScope is frozen (immutable)."""
    scope = PrimitiveScope(scoped_managers=frozenset({"files"}))
    with pytest.raises(AttributeError):
        scope.scoped_managers = frozenset({"contacts"})  # type: ignore


def test_scope_equality():
    """Two scopes with same managers are equal."""
    scope1 = PrimitiveScope(scoped_managers=frozenset({"files", "contacts"}))
    scope2 = PrimitiveScope(scoped_managers=frozenset({"contacts", "files"}))
    assert scope1 == scope2


def test_scope_inequality():
    """Two scopes with different managers are not equal."""
    scope1 = PrimitiveScope(scoped_managers=frozenset({"files"}))
    scope2 = PrimitiveScope(scoped_managers=frozenset({"contacts"}))
    assert scope1 != scope2


def test_scope_hashable():
    """PrimitiveScope can be used as dict key."""
    scope1 = PrimitiveScope(scoped_managers=frozenset({"files", "contacts"}))
    scope2 = PrimitiveScope(scoped_managers=frozenset({"contacts", "files"}))

    d = {scope1: "value1"}
    assert d[scope2] == "value1"  # Same scope should retrieve same value


# ────────────────────────────────────────────────────────────────────────────
# VALID_MANAGER_ALIASES tests
# ────────────────────────────────────────────────────────────────────────────


def test_valid_manager_aliases_contains_expected():
    """VALID_MANAGER_ALIASES contains expected managers."""
    expected = {
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
    }
    assert expected == VALID_MANAGER_ALIASES


def test_valid_manager_aliases_is_frozenset():
    """VALID_MANAGER_ALIASES is immutable."""
    assert isinstance(VALID_MANAGER_ALIASES, frozenset)


def test_computer_in_valid_aliases():
    """computer (ComputerPrimitives) is in VALID_MANAGER_ALIASES.

    ComputerPrimitives are indexed in Functions/Primitives alongside state
    managers, enabling discovery via FunctionManager and proper masking
    when promoted to an environment.
    """
    assert "computer" in VALID_MANAGER_ALIASES


def test_includes_returns_false_for_nonexistent():
    """includes() returns False for aliases not in scope."""
    scope = PrimitiveScope.single("files")
    # Check all other aliases are not included
    for alias in VALID_MANAGER_ALIASES - {"files"}:
        assert not scope.includes(alias)


def test_scoped_managers_for_role_excludes_coordinator_for_regular_role():
    scoped = scoped_managers_for_role(is_coordinator=False)
    assert "coordinator" not in scoped
    assert scoped == (VALID_MANAGER_ALIASES - {"coordinator"})


def test_scoped_managers_for_role_includes_coordinator_for_coordinator_role():
    scoped = scoped_managers_for_role(is_coordinator=True)
    assert scoped == VALID_MANAGER_ALIASES


def test_default_runtime_scope_respects_session_role_flag():
    SESSION_DETAILS.reset()
    try:
        SESSION_DETAILS.is_coordinator = False
        assert "coordinator" not in default_runtime_scope().scoped_managers

        SESSION_DETAILS.is_coordinator = True
        assert "coordinator" in default_runtime_scope().scoped_managers
    finally:
        SESSION_DETAILS.reset()
