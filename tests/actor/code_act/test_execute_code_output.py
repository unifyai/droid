"""
Tests for execute_code output propagation with state manager primitives.

These tests verify that when execute_code runs code that calls primitives
(e.g., primitives.contacts.ask) and awaits their results, the stdout and
result fields are properly captured and returned to the caller.

This is a critical integration point that is NOT covered by:
- test_sandbox.py (tests PythonExecutionSession but not primitives)
- tests/actor/state_managers/* (tests full actor.act() flow where LLM compensates)

The bug this catches: execute_code running `print(await handle.result())` but
returning empty stdout because the nested async tool loop result wasn't captured.
"""

from __future__ import annotations

from typing import Any, AsyncIterator

import pytest
import pytest_asyncio

from unify.actor.code_act_actor import CodeActActor
from unify.actor.execution import ExecutionResult, parts_to_text
from unify.actor.environments import StateManagerEnvironment
from unify.function_manager.primitives import Primitives
from unify.manager_registry import ManagerRegistry

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def configure_simulated_managers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Configure all managers to use simulated implementations."""
    from unify.settings import SETTINGS

    for impl_attr in [
        "contact",
        "task",
        "transcript",
        "knowledge",
        "guidance",
        "web",
    ]:
        monkeypatch.setenv(f"UNITY_{impl_attr.upper()}_IMPL", "simulated")
        if hasattr(SETTINGS, impl_attr):
            monkeypatch.setattr(
                getattr(SETTINGS, impl_attr),
                "IMPL",
                "simulated",
                raising=False,
            )

    # Enable optional managers
    for enabled_attr in ["guidance", "web", "knowledge"]:
        monkeypatch.setenv(f"UNITY_{enabled_attr.upper()}_ENABLED", "true")
        if hasattr(SETTINGS, enabled_attr):
            monkeypatch.setattr(
                getattr(SETTINGS, enabled_attr),
                "ENABLED",
                True,
                raising=False,
            )

    ManagerRegistry.clear()


@pytest_asyncio.fixture
async def actor_with_primitives(
    configure_simulated_managers,
) -> AsyncIterator[tuple[CodeActActor, Primitives]]:
    """Create a CodeActActor with primitives environment for direct execute_code testing."""
    from unify.function_manager.primitives import PrimitiveScope

    scope = PrimitiveScope(
        scoped_managers=frozenset({"contacts", "tasks", "transcripts", "knowledge"}),
    )
    primitives = Primitives(primitive_scope=scope)
    env = StateManagerEnvironment(primitives)
    actor = CodeActActor(environments=[env], function_manager=None)

    # Strip FunctionManager tools to focus on primitives
    act_tools = actor.get_tools("act")
    actor.add_tools("act", {"execute_code": act_tools["execute_code"]})

    try:
        yield actor, primitives
    finally:
        try:
            await actor.close()
        except Exception:
            pass


@pytest_asyncio.fixture
async def execute_code_tool(actor_with_primitives) -> tuple[Any, Primitives]:
    """Get the execute_code tool directly for isolated testing."""
    actor, primitives = actor_with_primitives
    tools = actor.get_tools("act")
    return tools["execute_code"], primitives


# ---------------------------------------------------------------------------
# Helper functions for handling ExecutionResult vs dict
# ---------------------------------------------------------------------------


def get_output_field(out: Any, field: str, default: Any = None) -> Any:
    """Get a field from execute_code output, handling both dict and ExecutionResult."""
    if isinstance(out, dict):
        return out.get(field, default)
    elif hasattr(out, field):
        return getattr(out, field, default)
    return default


def get_stdout_text(out: Any) -> str:
    """Extract stdout text from execute_code output."""
    stdout = get_output_field(out, "stdout", [])
    if isinstance(stdout, list):
        return parts_to_text(stdout)
    return str(stdout) if stdout else ""


def get_error(out: Any) -> str | None:
    """Get error from execute_code output."""
    return get_output_field(out, "error", None)


def get_result(out: Any) -> Any:
    """Get result from execute_code output."""
    return get_output_field(out, "result", None)


# ---------------------------------------------------------------------------
# Test: Runtime OAuth token helper exposed to real actor execute_code
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_code_oauth_helper_returns_proxy_handle(
    execute_code_tool: tuple[Any, Primitives],
) -> None:
    execute_code, _ = execute_code_tool

    # The sandbox helper returns the local workspace-proxy capability handle
    # (never a raw provider token); the same handle is exported to the sandbox
    # env alongside the proxy base URLs, which target the localhost proxy.
    out = await execute_code(
        "mock scenario: resolve the workspace proxy handle for both providers",
        """
import os
microsoft_handle = get_oauth_access_token("microsoft", min_ttl_seconds=123)
google_handle = get_oauth_access_token("google", min_ttl_seconds=456)
assert microsoft_handle == google_handle
assert microsoft_handle == os.environ["WORKSPACE_PROXY_TOKEN"]
assert os.environ["MICROSOFT_GRAPH_BASE"].endswith("/microsoft/v1.0")
assert os.environ["GOOGLE_DRIVE_BASE"].endswith("/google/drive/v3")
print("TOKEN_OK")
""",
        language="python",
        state_mode="stateless",
    )

    assert get_error(out) is None
    assert "TOKEN_OK" in get_stdout_text(out)


# ---------------------------------------------------------------------------
# Test: Basic stdout capture from primitives.*.ask().result()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_execute_code_can_branch_on_query_llm_structured_output(
    execute_code_tool,
    monkeypatch: pytest.MonkeyPatch,
):
    """execute_code can use typed semantic judgments in symbolic control flow."""
    execute_code, _primitives = execute_code_tool

    async def fake_query_llm(prompt: str, *, response_format=None, **kwargs):
        assert "Classify" in prompt
        return response_format(category="billing", needs_reply=True)

    import unify.common.reasoning as reasoning_module

    monkeypatch.setattr(reasoning_module, "query_llm", fake_query_llm)

    code = """
from pydantic import BaseModel

class Decision(BaseModel):
    category: str
    needs_reply: bool

Decision.model_rebuild()

decision = await query_llm(
    "Classify this email: Please approve the renewal quote.",
    response_format=Decision,
)
"queue_reply" if decision.category == "billing" and decision.needs_reply else "archive"
"""

    out = await execute_code(
        "test query_llm structured output",
        code,
        language="python",
        state_mode="stateless",
    )

    assert get_error(out) is None
    assert get_result(out) == "queue_reply"


@pytest.mark.asyncio
@pytest.mark.llm_call
@pytest.mark.timeout(120)
async def test_execute_code_captures_stdout_from_primitive_result(
    execute_code_tool,
):
    """
    CRITICAL TEST: Verifies stdout is captured when code prints a primitive result.

    This is the exact bug from the CI failure:
    - Code runs: handle = await primitives.contacts.ask(...); result = await handle.result(); print(result)
    - Expected: stdout contains the answer from ContactManager.ask
    - Bug: stdout was empty because result propagation failed
    """
    execute_code, primitives = execute_code_tool

    # Use a generic question that doesn't require seeded data
    code = """
handle = await primitives.contacts.ask("How many contacts are in the system?")
result = await handle.result()
print(f"ANSWER: {result}")
"""

    out = await execute_code(
        "test primitive stdout",
        code,
        language="python",
        state_mode="stateless",
    )

    error = get_error(out)
    assert error is None, f"Execution failed: {error}"

    stdout_text = get_stdout_text(out)

    # The stdout should contain the answer prefix
    assert (
        "ANSWER:" in stdout_text
    ), f"Expected 'ANSWER:' prefix in stdout, got: {stdout_text!r}"
    # The actual answer should be present (simulated manager returns something)
    assert len(stdout_text.strip()) > len(
        "ANSWER:",
    ), f"Expected stdout to contain actual result after 'ANSWER:', got: {stdout_text!r}"


@pytest.mark.asyncio
@pytest.mark.llm_call
@pytest.mark.timeout(120)
async def test_execute_code_result_field_populated_from_primitive(
    execute_code_tool,
):
    """
    Verifies the 'result' field is populated when code returns a primitive result.

    The execute_code docstring says:
    - result: The evaluated result of the last expression (Any), or None.

    When code ends with `await handle.result()`, that should be captured.
    """
    execute_code, primitives = execute_code_tool

    code = """
handle = await primitives.contacts.ask("List all contacts")
await handle.result()
"""

    out = await execute_code(
        "test result field",
        code,
        language="python",
        state_mode="stateless",
    )

    error = get_error(out)
    assert error is None, f"Execution failed: {error}"

    # The result field should contain the primitive's answer
    result = get_result(out)
    assert (
        result is not None
    ), f"Expected 'result' field to be populated, got None. Full output: {out}"


@pytest.mark.asyncio
@pytest.mark.llm_call
@pytest.mark.timeout(120)
async def test_execute_code_captures_stdout_with_multiple_prints(
    execute_code_tool,
):
    """Verifies multiple print statements are all captured in stdout."""
    execute_code, primitives = execute_code_tool

    code = """
print("BEFORE")
handle = await primitives.contacts.ask("How many contacts do we have?")
result = await handle.result()
print(f"RESULT: {result}")
print("AFTER")
"""

    out = await execute_code(
        "test multiple prints",
        code,
        language="python",
        state_mode="stateless",
    )

    error = get_error(out)
    assert error is None, f"Execution failed: {error}"

    stdout_text = get_stdout_text(out)

    assert "BEFORE" in stdout_text, f"Missing 'BEFORE' in stdout: {stdout_text!r}"
    assert "RESULT:" in stdout_text, f"Missing 'RESULT:' in stdout: {stdout_text!r}"
    assert "AFTER" in stdout_text, f"Missing 'AFTER' in stdout: {stdout_text!r}"


# ---------------------------------------------------------------------------
# Test: Sequential primitive calls
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.llm_call
@pytest.mark.timeout(180)
async def test_execute_code_sequential_primitive_calls(
    execute_code_tool,
):
    """Verifies two sequential primitive calls both have their results captured."""
    execute_code, primitives = execute_code_tool

    code = """
# First call
h1 = await primitives.contacts.ask("How many contacts exist?")
r1 = await h1.result()
print(f"FIRST: {r1}")

# Second call
h2 = await primitives.contacts.ask("List all contact names")
r2 = await h2.result()
print(f"SECOND: {r2}")
"""

    out = await execute_code(
        "test sequential calls",
        code,
        language="python",
        state_mode="stateless",
    )

    error = get_error(out)
    assert error is None, f"Execution failed: {error}"

    stdout_text = get_stdout_text(out)

    assert "FIRST:" in stdout_text, f"Missing first result in stdout: {stdout_text!r}"
    assert "SECOND:" in stdout_text, f"Missing second result in stdout: {stdout_text!r}"


# ---------------------------------------------------------------------------
# Test: Concurrent primitive calls
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.llm_call
@pytest.mark.timeout(180)
async def test_execute_code_concurrent_primitive_calls(
    execute_code_tool,
):
    """Verifies concurrent primitive calls via asyncio.gather work correctly."""
    execute_code, primitives = execute_code_tool

    code = """
import asyncio

h1 = await primitives.contacts.ask("Question 1: How many contacts?")
h2 = await primitives.contacts.ask("Question 2: List contact names")

# Wait for both concurrently
r1, r2 = await asyncio.gather(h1.result(), h2.result())

print(f"RESULT1: {r1}")
print(f"RESULT2: {r2}")
"""

    out = await execute_code(
        "test concurrent calls",
        code,
        language="python",
        state_mode="stateless",
    )

    error = get_error(out)
    assert error is None, f"Execution failed: {error}"

    stdout_text = get_stdout_text(out)

    assert "RESULT1:" in stdout_text, f"Missing RESULT1 in stdout: {stdout_text!r}"
    assert "RESULT2:" in stdout_text, f"Missing RESULT2 in stdout: {stdout_text!r}"


# ---------------------------------------------------------------------------
# Test: Different state managers
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.llm_call
@pytest.mark.timeout(120)
async def test_execute_code_tasks_manager_output(
    execute_code_tool,
):
    """Verifies stdout capture works with TaskScheduler primitives."""
    execute_code, primitives = execute_code_tool

    code = """
handle = await primitives.tasks.ask("What tasks are currently pending?")
result = await handle.result()
print(f"TASKS: {result}")
"""

    out = await execute_code(
        "test tasks manager",
        code,
        language="python",
        state_mode="stateless",
    )

    error = get_error(out)
    assert error is None, f"Execution failed: {error}"

    stdout_text = get_stdout_text(out)

    assert "TASKS:" in stdout_text, f"Missing 'TASKS:' in stdout: {stdout_text!r}"


@pytest.mark.asyncio
@pytest.mark.llm_call
@pytest.mark.timeout(120)
async def test_execute_code_transcripts_manager_output(
    execute_code_tool,
):
    """Verifies stdout capture works with TranscriptManager primitives."""
    execute_code, primitives = execute_code_tool

    code = """
handle = await primitives.transcripts.ask("What conversations happened today?")
result = await handle.result()
print(f"TRANSCRIPTS: {result}")
"""

    out = await execute_code(
        "test transcripts manager",
        code,
        language="python",
        state_mode="stateless",
    )

    error = get_error(out)
    assert error is None, f"Execution failed: {error}"

    stdout_text = get_stdout_text(out)

    assert (
        "TRANSCRIPTS:" in stdout_text
    ), f"Missing 'TRANSCRIPTS:' in stdout: {stdout_text!r}"


# ---------------------------------------------------------------------------
# Test: Return value vs print
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.llm_call
@pytest.mark.timeout(120)
async def test_execute_code_return_value_captured(
    execute_code_tool,
):
    """
    Verifies that when code returns a value (last expression), it's captured in 'result'.

    This tests the alternative pattern where code doesn't print but returns the value.
    """
    execute_code, primitives = execute_code_tool

    code = """
handle = await primitives.contacts.ask("List contacts")
await handle.result()  # This is the last expression, should be captured as result
"""

    out = await execute_code(
        "test return value",
        code,
        language="python",
        state_mode="stateless",
    )

    error = get_error(out)
    assert error is None, f"Execution failed: {error}"

    # The result field should capture the last expression's value
    result = get_result(out)
    assert (
        result is not None
    ), f"Expected 'result' to capture last expression value, got None"


# ---------------------------------------------------------------------------
# Test: Error propagation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_execute_code_primitive_error_propagates(
    execute_code_tool,
):
    """Verifies that errors from primitive calls are properly captured."""
    execute_code, primitives = execute_code_tool

    # This code intentionally accesses a non-existent manager attribute
    code = """
try:
    handle = await primitives.nonexistent_manager.ask("test")
    result = await handle.result()
    print(f"RESULT: {result}")
except AttributeError as e:
    print(f"ERROR: {e}")
"""

    out = await execute_code(
        "test error handling",
        code,
        language="python",
        state_mode="stateless",
    )

    # The code has a try/except, so it shouldn't have an error
    error = get_error(out)
    assert error is None, f"Unexpected execution error: {error}"

    stdout_text = get_stdout_text(out)

    # Should have caught and printed the error
    assert "ERROR:" in stdout_text, f"Missing error output in stdout: {stdout_text!r}"


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_execute_code_unhandled_error_captured(
    execute_code_tool,
):
    """Verifies unhandled exceptions are captured in the error field."""
    execute_code, primitives = execute_code_tool

    code = """
# This will raise an AttributeError
handle = await primitives.nonexistent_manager.ask("test")
"""

    out = await execute_code(
        "test unhandled error",
        code,
        language="python",
        state_mode="stateless",
    )

    # Should have an error captured
    error = get_error(out)
    assert (
        error is not None
    ), f"Expected error to be captured for invalid primitive access"
    assert "AttributeError" in str(
        error,
    ), f"Expected AttributeError in error field: {error}"


# ---------------------------------------------------------------------------
# Test: Stdout isolation (inner loop stdout shouldn't leak)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.llm_call
@pytest.mark.timeout(120)
async def test_execute_code_stdout_isolation_from_inner_loops(
    execute_code_tool,
):
    """
    Verifies that stdout from inner async tool loops doesn't leak to outer sandbox.

    When primitives.contacts.ask runs, it has its own async tool loop that may
    produce internal logging or output. This should NOT appear in the sandbox's
    stdout - only explicit print() calls in the user code should.
    """
    execute_code, primitives = execute_code_tool

    code = """
print("START")
handle = await primitives.contacts.ask("List contacts")
result = await handle.result()
print("END")
"""

    out = await execute_code(
        "test stdout isolation",
        code,
        language="python",
        state_mode="stateless",
    )

    error = get_error(out)
    assert error is None, f"Execution failed: {error}"

    stdout_text = get_stdout_text(out)

    # Should have START and END
    assert "START" in stdout_text, f"Missing 'START' in stdout: {stdout_text!r}"
    assert "END" in stdout_text, f"Missing 'END' in stdout: {stdout_text!r}"

    # Should NOT have internal async tool loop artifacts like tool call IDs,
    # LLM thinking, or internal logging markers
    internal_markers = [
        "toolu_",  # Anthropic tool call IDs
        "call_",  # OpenAI tool call IDs
        "[Tool",  # Internal logging
        "🔧",  # Internal emoji markers
        "🛠️",
    ]
    for marker in internal_markers:
        assert (
            marker not in stdout_text
        ), f"Found internal marker '{marker}' in stdout - inner loop leaking: {stdout_text!r}"


# ---------------------------------------------------------------------------
# Test: Stateful session preserves primitives across calls
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.llm_call
@pytest.mark.timeout(120)
async def test_execute_code_stateful_session_with_primitives(
    execute_code_tool,
):
    """Verifies stateful sessions maintain access to primitives across calls."""
    execute_code, primitives = execute_code_tool

    # First call: store result in a variable
    out1 = await execute_code(
        "first call",
        """
handle = await primitives.contacts.ask("How many contacts?")
stored_result = await handle.result()
print(f"STORED: {stored_result}")
""",
        language="python",
        state_mode="stateful",
        session_id=0,
    )

    error1 = get_error(out1)
    assert error1 is None, f"First call failed: {error1}"

    # Second call: access the stored variable
    out2 = await execute_code(
        "second call",
        """
print(f"RETRIEVED: {stored_result}")
""",
        language="python",
        state_mode="stateful",
        session_id=0,
    )

    error2 = get_error(out2)
    assert error2 is None, f"Second call failed: {error2}"

    stdout_text = get_stdout_text(out2)

    assert (
        "RETRIEVED:" in stdout_text
    ), f"Missing 'RETRIEVED:' in second call stdout: {stdout_text!r}"


# ---------------------------------------------------------------------------
# Test: ExecutionResult formatting for LLM
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.llm_call
@pytest.mark.timeout(120)
async def test_execution_result_includes_stdout_in_llm_content(
    execute_code_tool,
):
    """
    Verifies that ExecutionResult.to_llm_content() includes stdout when present.

    This is critical for the outer LLM to see the primitive results.
    """
    execute_code, primitives = execute_code_tool

    code = """
handle = await primitives.contacts.ask("List contacts")
result = await handle.result()
print(f"ANSWER: {result}")
"""

    out = await execute_code(
        "test llm content",
        code,
        language="python",
        state_mode="stateless",
    )

    error = get_error(out)
    assert error is None, f"Execution failed: {error}"

    # If out is already an ExecutionResult, use it directly
    # Otherwise wrap it
    if isinstance(out, ExecutionResult):
        exec_result = out
    elif isinstance(out, dict):
        stdout = out.get("stdout", [])
        if isinstance(stdout, list):
            exec_result = ExecutionResult(**out)
        else:
            pytest.skip("Output not in expected format for ExecutionResult test")
            return
    else:
        # Try to use it directly if it has to_llm_content
        if hasattr(out, "to_llm_content"):
            exec_result = out
        else:
            pytest.skip("Output not in expected format for ExecutionResult test")
            return

    llm_content = exec_result.to_llm_content()

    # Should have content blocks
    assert isinstance(llm_content, list), f"Expected list, got {type(llm_content)}"
    assert len(llm_content) > 0, "Expected at least one content block"

    # Combine all text content
    all_text = " ".join(
        block.get("text", "") for block in llm_content if block.get("type") == "text"
    )

    assert (
        "ANSWER:" in all_text
    ), f"Expected 'ANSWER:' in LLM content, got: {all_text!r}"


# ---------------------------------------------------------------------------
# Test: Handle result contains actual primitive answer
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.llm_call
@pytest.mark.timeout(120)
async def test_primitive_handle_result_contains_answer(
    actor_with_primitives,
):
    """
    Direct test that primitive handles return actual answers, not None.

    This bypasses execute_code entirely to verify the primitives work correctly.
    If this passes but execute_code tests fail, the bug is in execute_code.
    """
    actor, primitives = actor_with_primitives

    # Use a generic question that doesn't require seeded data
    # Call primitive directly (not through execute_code)
    handle = await primitives.contacts.ask("How many contacts are in the system?")
    result = await handle.result()

    # The result should be a non-empty string
    assert result is not None, "Primitive handle.result() returned None"
    assert isinstance(result, str), f"Expected string, got {type(result)}"
    assert len(result) > 0, "Primitive handle.result() returned empty string"


# ---------------------------------------------------------------------------
# Test: Verify the exact bug scenario from CI failure
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.llm_call
@pytest.mark.timeout(120)
async def test_exact_ci_failure_scenario(
    execute_code_tool,
):
    """
    Reproduces the exact scenario from the CI failure:
    test_contact_lookup_by_email_returns_phone

    The test expected Alice's phone number (15555552222) but got a hallucinated
    number (5551234567) because execute_code didn't propagate the actual result.
    """
    execute_code, primitives = execute_code_tool

    # The exact code pattern from the CI failure (using generic question since
    # we can't seed contacts in async context without special handling)
    code = """
handle = await primitives.contacts.ask(
    "How many contacts are in the system? Please provide a count."
)
result = await handle.result()
print(result)
"""

    out = await execute_code(
        "CI failure reproduction",
        code,
        language="python",
        state_mode="stateless",
    )

    error = get_error(out)
    assert error is None, f"Execution failed: {error}"

    stdout_text = get_stdout_text(out)

    # The stdout should contain SOMETHING (not be empty)
    assert (
        stdout_text.strip()
    ), f"stdout is empty - this is the bug! Expected primitive result, got nothing."

    # Note: We can't assert the exact content because simulated managers
    # may return different data. But stdout should NOT be empty.
    # If this test passes with non-empty stdout, the basic propagation works.
    # The CI failure happened because stdout was empty and the LLM hallucinated.
