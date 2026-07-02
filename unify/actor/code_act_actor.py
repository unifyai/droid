import asyncio
import contextvars
import copy
import functools
import inspect
import json
import sys
import traceback
import uuid
from secrets import token_hex as _token_hex
import logging
from typing import (
    Annotated,
    Any,
    Callable,
    Awaitable,
    Dict,
    NamedTuple,
    Optional,
    Type,
    Union,
    TYPE_CHECKING,
)
from pydantic import BaseModel

from unify.actor.base import BaseCodeActActor
from unify.common.context_dump import make_messages_safe_for_context_dump
from unify.actor.execution import (
    ExecutionResult,
    PackageOverlay,
    PythonExecutionSession,
    SessionExecutor,
    SessionKey,
    _CURRENT_ENVIRONMENTS,
    _CURRENT_PACKAGE_OVERLAY,
    _CURRENT_SANDBOX,
    _PARENT_CHAT_CONTEXT,
    _validate_execution_params,
)
from unify.common.async_tool_loop import (
    AsyncToolLoopHandle,
    SteerableToolHandle,
    start_async_tool_loop,
)
from unify.common.task_execution_context import (
    PostRunReviewContext,
    current_post_run_review_context,
    TaskExecutionDelegate,
    current_task_execution_delegate,
)
from unify.events.event_bus import EVENT_BUS, Event
from unify.common.llm_client import new_llm_client
from unify.common.act_llm_profiles import (
    CURRENT_ACT_LLM_PROFILE,
    resolve_act_llm_profile,
)
from unify.common.llm_helpers import methods_to_tool_dict
from unify.common.tool_spec import ToolSpec, llm_soft_required
from unify.function_manager.base import BaseFunctionManager
from unify.function_manager.primitives import ComputerPrimitives
from unify.actor.prompt_builders import build_code_act_prompt
from unify.events.manager_event_logging import log_manager_call
from unify.common._async_tool.loop_config import TOOL_LOOP_LINEAGE, _PENDING_LOOP_SUFFIX
from unify.common.hierarchical_logger import log_boundary_event
from unify.events.manager_event_logging import (
    new_call_id,
    publish_manager_method_event,
)
from unify.events.active_work import ACTIVE_WORK, ActiveWorkHandle
from unify.integrations.approval import build_pending_approval_payload
from unify.integrations.function_metadata import is_provider_backed_function

if TYPE_CHECKING:
    from unify.actor.environments.base import BaseEnvironment
    from unify.function_manager.function_manager import FunctionManager
    from unify.guidance_manager.guidance_manager import GuidanceManager


# ---------------------------------------------------------------------------
# Tool-policy type alias and sentinel
# ---------------------------------------------------------------------------

ToolPolicyFn = Callable[[int, Dict[str, Any]], tuple[str, Dict[str, Any]]]
"""Signature for a tool-policy callback.

Receives ``(step_index, tools_dict)`` and returns ``(tool_choice_mode,
filtered_tools_dict)`` where *tool_choice_mode* is ``"auto"`` or
``"required"``.
"""

_USE_DEFAULT: object = object()
"""Sentinel indicating 'use the built-in discovery-first tool policy'."""

_UNSET: object = object()
"""Sentinel indicating 'parameter was not explicitly provided'."""


class _ActiveWorkNotificationQueue:
    def __init__(
        self,
        target: asyncio.Queue[dict],
        active_work: ActiveWorkHandle,
    ) -> None:
        self._target = target
        self._active_work = active_work

    async def put(self, item: dict) -> None:
        self._active_work.record_user_notification()
        await self._target.put(item)

    def put_nowait(self, item: dict) -> None:
        self._active_work.record_user_notification()
        self._target.put_nowait(item)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._target, name)


def _resolve_param(explicit: object, code_value: object, default: object) -> object:
    """Three-tier resolution: explicit constructor arg > code config > hardcoded default."""
    if explicit is not _UNSET:
        return explicit
    if code_value is not None:
        return code_value
    return default


def _default_tool_policy(
    has_fm_tools: bool,
    has_gm_tools: bool,
    filter_tools: Callable[[Dict[str, Any]], Dict[str, Any]],
) -> ToolPolicyFn:
    """Build the default *discovery-first* tool policy.

    Until **both** a ``FunctionManager_*`` and a ``GuidanceManager_*`` tool
    have been called at least once, the LLM is restricted to only those
    discovery tools (with ``tool_choice="required"``).  Once both gates are
    satisfied the full (statically-filtered) tool set is returned with
    ``"auto"`` mode.

    When only one of the two manager tool families is present, that single
    family acts as the sole gate.  When neither is present the policy is a
    no-op pass-through.

    Parameters
    ----------
    has_fm_tools:
        Whether the base tool set contains any ``FunctionManager_*`` tools.
    has_gm_tools:
        Whether the base tool set contains any ``GuidanceManager_*`` tools.
    filter_tools:
        The static-filter callable (``_filter_tools``) that enforces
        ``can_compose`` / ``can_store`` / ``can_spawn_sub_agents``.
    """

    def _policy(
        step: int,
        tools: Dict[str, Any],
        called_tools: list[str],
    ) -> tuple[str, Dict[str, Any]]:
        filtered = filter_tools(tools)

        fm_satisfied = (not has_fm_tools) or any(
            t.startswith("FunctionManager_") for t in called_tools
        )
        gm_satisfied = (not has_gm_tools) or any(
            t.startswith("GuidanceManager_") for t in called_tools
        )

        if fm_satisfied and gm_satisfied:
            return "auto", filtered

        # Expose only the unsatisfied gate(s) and require a call.
        gated: Dict[str, Any] = {}
        if not fm_satisfied:
            gated.update(
                {
                    k: v
                    for k, v in filtered.items()
                    if isinstance(k, str) and k.startswith("FunctionManager_")
                },
            )
        if not gm_satisfied:
            gated.update(
                {
                    k: v
                    for k, v in filtered.items()
                    if isinstance(k, str) and k.startswith("GuidanceManager_")
                },
            )

        return ("required", gated) if gated else ("auto", filtered)

    return _policy


# ---------------------------------------------------------------------------
# Resolved session tuple returned by _resolve_session
# ---------------------------------------------------------------------------


class _ResolvedSession(NamedTuple):
    language: str
    venv_id: Optional[int]
    session_id: Optional[int]
    error: Optional[Dict[str, Any]]  # validation error dict, or None


# ---------------------------------------------------------------------------
# Agent context for tracking execution depth and providing handle access
# ---------------------------------------------------------------------------
from dataclasses import dataclass, field as dataclass_field


@dataclass
class AgentContext:
    """Runtime context for agent execution, accessible via get_current_agent_context().

    Attributes:
        depth: Nesting level (0 = root agent, 1 = first subagent, etc.)
        agent_id: Unique identifier for this agent run
        handle: Reference to the AsyncToolLoopHandle (for accessing history, etc.)
    """

    depth: int = 0
    agent_id: str = dataclass_field(default_factory=lambda: str(uuid.uuid4()))
    handle: "AsyncToolLoopHandle | None" = None
    proactive_storage_summaries: list[str] = dataclass_field(default_factory=list)


_CURRENT_AGENT_CONTEXT: contextvars.ContextVar[AgentContext] = contextvars.ContextVar(
    "code_act_agent_context",
    default=AgentContext(),
)


def get_current_agent_context() -> AgentContext:
    """Get the current agent execution context.

    Use this inside service methods to:
    - Check agent depth and prevent infinite recursion
    - Access the current agent's handle for message history, etc.

    Returns:
        AgentContext with depth, agent_id, and handle

    Example:
        ctx = get_current_agent_context()
        if ctx.depth >= 2:
            raise RuntimeError("Max depth exceeded")
        if ctx.handle:
            history = ctx.handle.get_history()
    """
    return _CURRENT_AGENT_CONTEXT.get()


logger = logging.getLogger(__name__)


class _CodeActEntrypointHandle(SteerableToolHandle):  # type: ignore[abstract-method]
    """Execute a FunctionManager entrypoint function without invoking the CodeAct LLM loop.

    TaskScheduler delegates task execution to an actor via:
    `primitives.actor.act(task_description, entrypoint=<function_id>, persist=False)`.

    When an `entrypoint` is provided, CodeActActor resolves the function by id,
    injects it into the sandbox namespace, and executes it in an asyncio task.
    """

    def __init__(
        self,
        *,
        entrypoint_id: int,
        execution_task: asyncio.Task[Any],
        on_finally: Optional[Callable[[], Awaitable[None]]] = None,
    ) -> None:
        self._entrypoint_id = int(entrypoint_id)
        self._execution_task = execution_task
        self._completion_event = asyncio.Event()
        self._result_str: Optional[str] = None
        self._error: Optional[BaseException] = None
        self._stopped = False
        self._on_finally = on_finally

        asyncio.create_task(self._monitor_execution())

    async def _monitor_execution(self) -> None:
        try:
            out = await self._execution_task
            if not self._stopped:
                self._result_str = str(out) if out is not None else ""
        except asyncio.CancelledError:
            self._stopped = True
            self._result_str = f"Entrypoint {self._entrypoint_id} was cancelled."
        except Exception as e:
            self._error = e
            self._result_str = f"Error: {e}"
        finally:
            if self._on_finally is not None:
                try:
                    await self._on_finally()
                except Exception:
                    pass
            self._completion_event.set()

    async def ask(
        self,
        question: str,
        *,
        _parent_chat_context: list[dict] | None = None,
    ) -> SteerableToolHandle:
        status = "completed" if self.done() else "still running"
        client = new_llm_client()
        client.set_system_message(
            "You are an AI assistant answering a status question about an in-flight entrypoint execution. "
            "Be brief and factual.",
        )
        msg = (
            f"Entrypoint {self._entrypoint_id} status: {status}.\n\n"
            f"User question: {question}"
        )
        return start_async_tool_loop(
            client=client,
            message=msg,
            tools={},
            loop_id=f"EntrypointQuestion({self._entrypoint_id})",
            max_consecutive_failures=1,
        )

    async def interject(
        self,
        message: str,
        *,
        _parent_chat_context_cont: list[dict] | None = None,
    ) -> None:
        # No-op for non-LLM entrypoint execution.
        pass

    async def stop(
        self,
        reason: Optional[str] = None,
    ) -> None:
        if self._completion_event.is_set():
            return
        self._stopped = True
        self._result_str = (
            f"Entrypoint {self._entrypoint_id} stopped."
            if not reason
            else f"Entrypoint {self._entrypoint_id} stopped: {reason}"
        )
        self._execution_task.cancel()
        try:
            await asyncio.wait_for(self._completion_event.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            pass

    async def pause(self) -> Optional[str]:
        return None

    async def resume(self) -> Optional[str]:
        return None

    def done(self) -> bool:
        return self._completion_event.is_set()

    async def result(self) -> str:
        await self._completion_event.wait()
        if self._error is not None:
            raise self._error
        return self._result_str or ""

    async def next_clarification(self) -> dict:
        await asyncio.Event().wait()
        return {}

    async def next_notification(self) -> dict:
        await asyncio.Event().wait()
        return {}

    async def answer_clarification(self, call_id: str, answer: str) -> None:
        return None


class _CodeActTaskExecutionDelegate:
    """Route durable task execution through the CodeActActor run that requested it."""

    def __init__(self, actor: "CodeActActor") -> None:
        self._actor = actor

    async def start_task_run(
        self,
        *,
        task_description: str,
        entrypoint: int | None,
        parent_chat_context: list[dict] | None,
        clarification_up_q: Optional[asyncio.Queue[str]],
        clarification_down_q: Optional[asyncio.Queue[str]],
        images: Any | None = None,
        **kwargs: Any,
    ) -> SteerableToolHandle:
        """Start one task run using this actor's CodeAct execution machinery."""

        _ = images
        task_guidelines = kwargs.pop("guidelines", None)
        return await self._actor.act(
            task_description,
            guidelines=task_guidelines,
            _parent_chat_context=parent_chat_context,
            _clarification_up_q=clarification_up_q,
            _clarification_down_q=clarification_down_q,
            entrypoint=entrypoint,
            persist=False,
            _reuse_actor_slot=entrypoint is not None,
            **kwargs,
        )


# ---------------------------------------------------------------------------
# Shared storage-review prompt sections
# ---------------------------------------------------------------------------

_DEFAULT_STORAGE_REVIEW_LABEL = "Storing reusable skills"
_DEFAULT_STORAGE_REVIEW_INSTRUCTIONS = (
    "Review the trajectory and store any reusable functions and compositional guidance."
)
MAX_OFFLINE_CERTIFICATION_REVISION_ATTEMPTS = 2


def _signature_compatible_kwargs(
    fn: Callable[..., Any],
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Return only the scheduler-supplied kwargs accepted by a callable."""

    signature = inspect.signature(fn)
    parameters = signature.parameters
    if any(
        param.kind is inspect.Parameter.VAR_KEYWORD for param in parameters.values()
    ):
        return dict(kwargs)
    accepted = {
        name
        for name, param in parameters.items()
        if param.kind
        in {
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        }
    }
    return {key: value for key, value in kwargs.items() if key in accepted}


_STORAGE_WHAT_CAN_BE_STORED = (
    "## What Can Be Stored\n\n"
    "Any code that executed successfully in `execute_code` during "
    "this trajectory can be stored as a function. Environment-provided "
    "namespaces (`primitives`, `primitives.computer`, `primitives.actor`) and "
    "other stored functions referenced in the code are automatically "
    "detected from the source and injected at runtime — you do not "
    "need to add imports or worry about whether these names will be "
    "available when the function runs later. Focus on whether a "
    "pattern is *worth* reusing, not whether it is *technically "
    "executable* in isolation.\n\n"
    "### Bake configuration into reusable callables\n\n"
    "Stored functions should NOT be verbatim copies of code blocks "
    "from the trajectory. During execution, the agent discovered the "
    "right combination of parameters, tool selections, and strategies "
    "through reasoning — that configuration knowledge is the valuable "
    "part. A stored function should **bake in** the hard-won "
    "configuration as fixed values and **expose** only the parts "
    "that genuinely vary between uses (typically the task-specific "
    "input). This produces a function that future callers can use "
    "without rediscovering the right setup.\n\n"
    "The same applies to trajectories that unrolled as "
    "`execute_code` -> observe -> agent reasoning -> `execute_code` loops. "
    "Do not assume the future workflow needs a full CodeActActor. First "
    "ask whether the agent's intermediate reasoning was open-ended "
    "planning or bounded semantic judgment inside an otherwise stable "
    "control flow. If it was bounded, distill the trajectory into one "
    "function: deterministic Python for control flow, managed primitives "
    "for side effects, and focused `query_llm(...)` calls with "
    "structured outputs, low temperature, and an explicit model for "
    "classification, summarization, drafting, ranking, or source "
    "selection. Leave it live-agent or guidance-driven when the reasoning "
    "involved changing tool discovery, unknown-state debugging, user "
    "clarification, or broad strategy selection.\n\n"
    "Semantic downgrades are bugs. When a live trajectory interpreted or "
    "produced unstructured data — classification, extraction, routing, "
    "summarization, drafting, rewriting, personalization, or other "
    "human-facing synthesis — the stored function should preserve that "
    "fuzzy step as `query_llm(...)` with a stable contract. Do not "
    "replace it with keyword ladders, regex classifiers, label-specific "
    "canned prose, or templates inferred from observed examples unless "
    "the user explicitly requested fixed deterministic rules/templates. "
    "Generalize by preserving the LLM call, not by memorizing the sample "
    "cases.\n\n"
    "### Preserving user-facing communication points\n\n"
    "When wrapping a workflow into a stored function, pay attention to "
    "points where the original code depended on the user being "
    "informed — especially states that block until the user takes an "
    "external action. If the original trajectory included a step like "
    '"notify the user about X, then wait for X to happen", the '
    "`notify()` call must survive into the stored function. Stripping "
    "it out creates a silent deadlock: the function blocks waiting for "
    "a condition the user does not know about.\n\n"
    "The general principle: a stored function inherits the execution "
    "environment's `notify()` helper. Any workflow state where "
    "progress depends on external human action (approving an auth "
    "prompt, granting a permission, confirming a destructive "
    "operation) must include a `notify()` call *before* entering the "
    "wait. Without it, the function waits indefinitely for something "
    "only the user can provide, and the user has no idea they need "
    "to act.\n\n"
    "### Durable task executor candidates\n\n"
    "A function intended to become a future TaskScheduler executor must "
    "preserve the observed live execution chain, not merely produce a "
    "plausible answer for the same example. Map each live trajectory step "
    "to the candidate code path that replaces or preserves it. Keep managed "
    "primitives, helper calls, validation gates, side-effect ordering, "
    "retries, cleanup, result shape, and failure semantics unless the "
    "candidate declares and validates an equivalent replacement. A live "
    "thinking step may become `query_llm(...)` only when it has a stable "
    "input/output contract and validation; if it required agentic "
    "exploration, preserve that substep or leave the task "
    "description-driven.\n\n"
    "Executor candidates may simplify incidental logging, formatting, dead "
    "exploratory branches, or duplicated setup, but they must not hardcode "
    "observations from live tool results, remove validation gates, reorder "
    "dependent side effects, discard recovery branches, replace managed "
    "tools with weaker ad hoc mechanisms, or replace semantic LLM work "
    "with brittle symbolic approximations. Store non-executor helpers and "
    "guidance freely; offline executor promotion requires separate "
    "certification.\n\n"
    "### Third-party package dependencies\n\n"
    "If the trajectory used `install_python_packages` and the function "
    "you want to store imports any of those packages (anything beyond "
    "the Python standard library and the environment-provided "
    "namespaces `primitives` and `pydantic`), the function **requires "
    "a virtual environment**. `FunctionManager_add_functions` will "
    "reject the function if third-party imports are detected without "
    "a `venv_id`.\n\n"
    "Workflow:\n"
    "1. Check existing venvs with `FunctionManager_list_venvs` — if "
    "one already declares the needed packages, reuse it.\n"
    "2. If no suitable venv exists, create one with "
    "`FunctionManager_add_venv`. Pass a minimal `pyproject.toml` "
    "string declaring only the packages the function actually "
    "imports. Example:\n\n"
    "```\n"
    "[project]\n"
    'name = "google-cloud-tools"\n'
    'version = "0.1.0"\n'
    'requires-python = ">=3.11"\n'
    "dependencies = [\n"
    '    "google-cloud-storage>=2.0.0",\n'
    "]\n"
    "```\n\n"
    "3. Pass the returned `venv_id` to "
    "`FunctionManager_add_functions(venv_id=<id>)`.\n\n"
    "Multiple functions that share the same dependency set should "
    "share a single venv. Do not create a separate venv per function "
    "when the dependency overlap is high — update an existing venv "
    "with `FunctionManager_update_venv` to add extra packages "
    "instead.\n\n"
)

_STORAGE_TWO_STORES = (
    "## Two Stores\n\n"
    "You have access to two complementary stores:\n\n"
    "### Function Store — the *what*\n\n"
    "The FunctionManager stores concrete, reusable function "
    "implementations — the building blocks. Each entry is a "
    "single callable with a clear name, docstring, and implementation.\n\n"
    "Actions:\n"
    "- **Add** a genuinely new, reusable function "
    "(`FunctionManager_add_functions`). If the function imports "
    "third-party packages, you **must** supply `venv_id`.\n"
    "- **Update** an existing function with a better implementation "
    "(`FunctionManager_add_functions` with `overwrite=True`).\n"
    "- **Merge** overlapping functions into one general-purpose function: "
    "add the merged version, then delete the old entries "
    "(`FunctionManager_delete_function`).\n"
    "- **Delete** functions that are redundant or superseded "
    "(`FunctionManager_delete_function`).\n"
    "- **Manage venvs**: create (`FunctionManager_add_venv`), list "
    "(`FunctionManager_list_venvs`), update "
    "(`FunctionManager_update_venv`), or delete "
    "(`FunctionManager_delete_venv`) virtual environments for "
    "functions with third-party dependencies. Link a function to a "
    "venv via `FunctionManager_set_function_venv` or pass `venv_id` "
    "directly to `FunctionManager_add_functions`.\n\n"
    "Do NOT store trivial one-liners, test scaffolding, or functions "
    "that are too task-specific to be reusable.\n\n"
    "### Guidance Store — the *how*\n\n"
    "The GuidanceManager stores procedural how-to entries: "
    "step-by-step instructions, standard operating procedures, "
    "software usage walkthroughs, and strategies for composing "
    "multiple functions together to accomplish broader tasks. "
    "Think of guidance entries as recipes or playbooks — they "
    "describe the procedure, decision points, and caveats, "
    "rather than containing executable code.\n\n"
    "In this storage-review context, guidance is most relevant "
    "when the trajectory reveals a non-obvious multi-step "
    "composition strategy that would be hard to rediscover. "
    "A single function call, a linear sequence of obvious steps, "
    "or a workflow fully explained by the individual function "
    "docstrings does NOT need guidance. Exception: if a simple "
    "operation required a non-obvious correction (an error-recovery "
    "loop, a silent failure mode, a precondition discovered through "
    "trial and error), the corrected approach IS worth storing as "
    "guidance — the value is in the insight, not the code "
    "complexity.\n\n"
    "Actions:\n"
    "- **Add** guidance for a genuinely non-trivial compositional "
    "workflow (`GuidanceManager_add_guidance`). Include `function_ids` "
    "to cross-reference the concrete functions it describes.\n"
    "- **Update** existing guidance that is incomplete or "
    "superseded (`GuidanceManager_update_guidance`).\n"
    "- **Delete** guidance that is obsolete or redundant "
    "(`GuidanceManager_delete_guidance`).\n\n"
    "Do NOT duplicate information that already lives in a "
    "function's docstring. Do NOT create guidance for simple or "
    "self-explanatory workflows.\n\n"
    "### Relationship between the two stores\n\n"
    "| Aspect | FunctionManager | GuidanceManager |\n"
    "|--------|----------------|----------------|\n"
    "| Granularity | Single callable | Multi-step workflow |\n"
    "| Content | Executable implementation | Natural-language recipe |\n"
    "| Analogy | A tool's docstring | A prompt that references tools |\n\n"
    "When a trajectory reveals both a useful function AND a non-trivial "
    "workflow that uses it, store the function first, then create a "
    "guidance entry referencing it via `function_ids`.\n\n"
)

_STORAGE_SUB_AGENT_PATTERNS = (
    "## Sub-Agent Delegation Patterns\n\n"
    "Calls to `primitives.actor.act(...)` in the trajectory are especially "
    "high-value storage candidates because they represent **pre-configured "
    "specialist agents**. Each `primitives.actor.act` invocation encodes a curated "
    "combination of `prompt_functions` (which tools the sub-agent sees), "
    "`guidelines` (how it should reason and compose those tools), "
    "`discovery_scope` (what it can find via search), and permission "
    "flags — together these define a specialist that can handle a "
    "particular *class* of tasks, not just the single task it was "
    "originally invoked for.\n\n"
    "### When to store\n\n"
    "Not every `primitives.actor.act` call is worth storing. Use this spectrum:\n\n"
    "- **Low value** — broad, unscoped delegation: all state managers "
    "in `prompt_functions`, generic or no `guidelines`, no "
    "`discovery_scope`, trivial `request`. This is just a passthrough "
    "that any future agent could reconstruct trivially.\n"
    "- **High value** — curated specialist: a carefully selected set of "
    "`prompt_functions`, detailed `guidelines` explaining how to compose "
    "those specific tools, a narrowed `discovery_scope`, and a non-trivial "
    "task that the sub-agent solved successfully. The configuration "
    "required real reasoning to discover and would be hard to "
    "rediscover from scratch.\n\n"
    "The more curation and domain knowledge went into the `primitives.actor.act` "
    "parameters, the more valuable it is to store.\n\n"
    "### What to bake in vs expose\n\n"
    "The parameters split naturally into two categories:\n\n"
    "- **Bake in** (agent specification): `guidelines`, "
    "`prompt_functions`, `discovery_scope`, `can_compose`, `can_store`, "
    "`can_spawn_sub_agents`, `timeout` — these define *what kind of "
    "specialist* this is and should be fixed in the stored function.\n"
    "- **Expose** (task specification): `request` — this defines *what "
    "to ask the specialist to do* and should be a parameter of the "
    "stored function.\n\n"
    "The result is a function that future callers can invoke with just "
    "a `request` string, without needing to know anything about the "
    "right tool selection, scoping, or behavioral guidelines.\n\n"
)

_STORAGE_BASE_INSTRUCTIONS = (
    "## Instructions\n\n"
    "1. Review the trajectory for reusable patterns.\n"
    "   Additionally, look for **pitfall patterns** — cases where the "
    "trajectory reveals that an obvious approach failed in a non-obvious "
    "way (a silent data loss, a missing property, a precondition the API "
    "doesn't enforce). These patterns have high reuse value even when the "
    "corrected code is simple: every future actor will attempt the obvious "
    "approach first. Indicators include error-recovery loops (the actor hit "
    "an exception, diagnosed the cause, and restructured the code to avoid "
    "it) and corrections that address a gap between a tool's apparent "
    "contract and its actual behavior. A brief guidance entry documenting "
    "the pitfall — what fails, why, and the correct approach — saves "
    "future sessions from repeating the same discovery cycle.\n"
    "2. Search the existing stores to understand what already exists "
    "(use the search/filter tools for each store).\n"
    "3. Decide what actions (if any) would improve the library. "
    "Prefer a clean, non-redundant library over a large one. "
    "Most trajectories will only warrant function changes, if "
    "anything at all. Add guidance only when a multi-step "
    "composition is genuinely non-obvious.\n"
    "4. **Delete superseded functions when you add a generalization.** "
    "When you store a new function that subsumes existing narrower "
    "variants (e.g. you add `greet(name, style)` while the store already "
    "has `greet_formal(name)` + `greet_casual(name)`), call "
    "`FunctionManager_delete_function` on the now-redundant entries by "
    "their `function_id` — leaving them in the library defeats the "
    "point of merging. The same applies to outright duplicates and to "
    "narrow special cases that the new function correctly handles.\n"
    "5. When done (or if there is nothing worth changing), respond "
    "with a brief summary of what you did (or that nothing was needed)."
)

# ---------------------------------------------------------------------------
# Shared storage tool construction
# ---------------------------------------------------------------------------


def _build_storage_tools(
    *,
    actor: "CodeActActor",
    ask_tools: dict,
    completed_tool_metadata: dict | None = None,
    task_entrypoint_review: dict[str, Any] | None = None,
) -> tuple[Dict[str, Callable], list[str]]:
    """Build the tool dict shared by both post-processing and proactive storage loops.

    Returns ``(tools, storage_active_lines)`` so callers can reference
    which inner tools are still actively reviewing skills.
    """
    fm = actor.function_manager
    gm = actor.guidance_manager

    tools: Dict[str, Callable] = {
        **methods_to_tool_dict(
            fm.search_functions,
            fm.filter_functions,
            fm.list_functions,
            fm.add_functions,
            fm.delete_function,
            fm.add_venv,
            fm.list_venvs,
            fm.get_venv,
            fm.update_venv,
            fm.delete_venv,
            fm.set_function_venv,
            fm.get_function_venv,
            gm.search,
            gm.filter,
            gm.get_guidance,
            gm.add_guidance,
            gm.update_guidance,
            gm.delete_guidance,
            include_class_name=True,
        ),
    }

    # ── Wire ask_about_completed_tool from snapshot ───────────────────

    _meta = completed_tool_metadata or {}
    storage_active_lines: list[str] = []
    storage_active_handles: Dict[str, Any] = {}
    dormant_lines: list[str] = []
    for name, fn in ask_tools.items():
        entry = None
        for _cid, _m in _meta.items():
            if _m.get("ask_fn") is fn:
                entry = _m
                break
        handle = entry.get("handle") if entry else None
        if handle is not None and hasattr(handle, "done") and not handle.done():
            storage_active_lines.append(f"- `{name}` [storage-active]")
            storage_active_handles[name] = handle
        else:
            dormant_lines.append(f"- `{name}`")

    if ask_tools:
        doc_sections: list[str] = [
            "Ask a follow-up question about a completed tool from the "
            "trajectory to inspect its internal reasoning or results.",
        ]
        if storage_active_lines:
            doc_sections.append(
                "\nStorage-active tools (background skill review in progress):\n"
                + "\n".join(storage_active_lines)
                + "\n\nThese tools have completed their primary task but are "
                "currently reviewing their execution for reusable skills. "
                "You can ask what they have stored or are considering.",
            )
        if dormant_lines:
            doc_sections.append(
                "\nCompleted tools (dormant):\n" + "\n".join(dormant_lines),
            )

        _ask_doc = "\n".join(doc_sections)

        async def ask_about_completed_tool(
            tool_name: str,
            question: str,
        ) -> str:
            fn = ask_tools.get(tool_name)
            if fn is None:
                return (
                    f"Tool '{tool_name}' not found. Available: {list(ask_tools.keys())}"
                )
            handle = await fn(question=question)
            if hasattr(handle, "result"):
                result = handle.result
                if callable(result):
                    result = result()
                if inspect.isawaitable(result):
                    result = await result
                return str(result)
            return str(handle)

        ask_about_completed_tool.__doc__ = _ask_doc
        tools["ask_about_completed_tool"] = ask_about_completed_tool

    # ── Wire steering tools for storage-active inner handles ──────────

    if storage_active_handles:
        _sa_handles = storage_active_handles
        _sa_listing = "\n".join(storage_active_lines)

        def _resolve_handle(tool_name: str) -> tuple[Any | None, str | None]:
            h = _sa_handles.get(tool_name)
            if h is None:
                avail = list(_sa_handles.keys())
                return None, (
                    f"Tool '{tool_name}' not found or no longer storage-active. "
                    f"Available: {avail}"
                )
            if hasattr(h, "done") and h.done():
                return None, (
                    f"Tool '{tool_name}' has already finished its storage review."
                )
            return h, None

        async def stop_inner_storage(tool_name: str, reason: str) -> str:
            """Stop an inner storage loop, preventing it from storing anything further.

            Use this when you have determined that the inner agent's storage
            would be redundant (e.g. you are storing a comprehensive function
            that already covers the inner agent's scope).
            """
            h, err = _resolve_handle(tool_name)
            if err:
                return err
            await h.stop(reason=reason)
            return f"Stopped inner storage for '{tool_name}': {reason}"

        stop_inner_storage.__doc__ += f"\n\nStorage-active tools:\n{_sa_listing}"

        async def interject_inner_storage(tool_name: str, message: str) -> str:
            """Inject a directive into an inner storage loop's conversation.

            Use this to provide context that should influence the inner
            loop's storage decisions (e.g. "The parent is storing a
            comprehensive function — only store yours if it is genuinely
            independent and reusable in isolation").
            """
            h, err = _resolve_handle(tool_name)
            if err:
                return err
            await h.interject(message)
            return f"Interjected into inner storage for '{tool_name}'."

        interject_inner_storage.__doc__ += f"\n\nStorage-active tools:\n{_sa_listing}"

        async def pause_inner_storage(tool_name: str) -> str:
            """Temporarily pause an inner storage loop.

            Use this to halt an inner loop while you make decisions,
            then resume it with ``resume_inner_storage``. The inner loop
            will not proceed until resumed (or until its timeout expires).
            """
            h, err = _resolve_handle(tool_name)
            if err:
                return err
            await h.pause()
            return f"Paused inner storage for '{tool_name}'."

        pause_inner_storage.__doc__ += f"\n\nStorage-active tools:\n{_sa_listing}"

        async def resume_inner_storage(tool_name: str) -> str:
            """Resume a previously paused inner storage loop.

            Call this after ``pause_inner_storage`` to let the inner
            loop continue its skill review.
            """
            h, err = _resolve_handle(tool_name)
            if err:
                return err
            await h.resume()
            return f"Resumed inner storage for '{tool_name}'."

        resume_inner_storage.__doc__ += f"\n\nStorage-active tools:\n{_sa_listing}"

        tools["stop_inner_storage"] = stop_inner_storage
        tools["interject_inner_storage"] = interject_inner_storage
        tools["pause_inner_storage"] = pause_inner_storage
        tools["resume_inner_storage"] = resume_inner_storage

    if task_entrypoint_review:
        attach_entrypoint = task_entrypoint_review.get("attach_entrypoint")
        promote_entrypoint_offline = task_entrypoint_review.get(
            "promote_entrypoint_offline",
        )
        metadata = dict(task_entrypoint_review.get("metadata") or {})
        task_id = metadata.get("task_id")
        instance_id = metadata.get("instance_id")
        task_name = metadata.get("task_name") or metadata.get("name") or "the task"

        async def attach_entrypoint_to_recurring_task(
            function_id: int,
            rationale: str,
            equivalence_manifest: dict[str, Any] | None = None,
            anti_oversimplification_checklist: dict[str, Any] | None = None,
        ) -> str:
            """Record a stored FunctionManager entrypoint candidate for future runs.

            Use this only after you have reviewed the completed trajectory and
            decided that the stored function captures a stable reusable workflow
            that preserves the observed operational contract. Calling this tool
            records the function as a symbolic executor candidate on future
            non-terminal instances; it does not promote those instances to
            offline delivery. Leaving the task description-driven is valid when
            future runs still need broad planning or tool discovery.
            """

            if not callable(attach_entrypoint):
                return "No task entrypoint attachment hook is available."
            return str(
                attach_entrypoint(
                    function_id=int(function_id),
                    rationale=str(rationale),
                    certification_metadata={
                        "equivalence_manifest": equivalence_manifest or {},
                        "anti_oversimplification_checklist": (
                            anti_oversimplification_checklist or {}
                        ),
                    },
                ),
            )

        attach_entrypoint_to_recurring_task.__doc__ += (
            f"\n\nCurrent task: {task_name} "
            f"(task_id={task_id}, completed instance_id={instance_id}). "
            "The tool only patches future non-terminal instances; it never "
            "rewrites the completed run or flips delivery to offline."
        )
        tools["attach_entrypoint_to_recurring_task"] = (
            attach_entrypoint_to_recurring_task
        )

        certification_revision_attempts = 0

        async def submit_offline_certification_evidence(
            function_id: int,
            certification_evidence: dict[str, Any],
            promotion_rationale: str | None = None,
        ) -> str:
            """Submit evidence for offline promotion of a symbolic task executor.

            Purpose:
            - Ask the scheduler-owned promotion gate whether a previously
              recorded FunctionManager entrypoint is safe and equivalent enough
              for future recurring task instances to run offline.
            - Submit structured evidence only. This tool does not certify by
              replaying work.

            Hard semantic rule:
            - This tool does not execute the entrypoint.
            - Do not use this tool to execute the entrypoint, replay live task
              steps, perform a dry-run, send messages, mutate external systems,
              fetch fresh expensive data, or make verification calls. The tool
              only passes evidence to the scheduler gate.

            Use this tool only when all of the following are true:
            - `attach_entrypoint_to_recurring_task(...)` has already recorded
              the symbolic executor candidate for future non-terminal instances.
            - You can provide complete evidence for the candidate's input,
              equivalence, side-effect, idempotency, cost, failure, observability,
              and managed-primitive contracts.
            - The candidate preserves the live run's managed primitive surface
              and operational behavior closely enough for offline delivery.

            Do not use this tool when:
            - The candidate is broad or still needs live planning, open-ended
              tool discovery, or user clarification.
            - Side effects are unclear, unsafe, not idempotent, or not covered
              by a concrete contract.
            - Verification would require token-heavy, costly, or effectful
              replay.
            - The function changes primitives, ordering, validation, recovery,
              output shape, failure behavior, or external data sources.
            - Evidence is incomplete or contradictory.

            Required evidence shape:
            `certification_evidence` must include:
            - `risk_classification`: one of `safe_noop`, `read_only`,
              `idempotent_effectful`, or `unsafe_effectful`.
            - `input_contract`: required runtime inputs and how future offline
              runs provide them.
            - `equivalence_contract`: mapping from the live task steps to the
              stored function paths, including result shape.
            - `managed_primitive_contract`: managed surfaces used by the live
              run and the candidate. It must include `preserved=True` and no
              `ad_hoc_replacements`.
            - `side_effect_contract`: side effects, ordering, and duplicate-run
              behavior.
            - `idempotency_contract`: why repeated/offline execution is safe or
              how duplicate effects are prevented.
            - `cost_contract`: bounded token/network/runtime cost. Include
              `bounded=True`.
            - `failure_contract`: preserved blocker, retry, validation, and
              recovery behavior.
            - `observability_contract`: what the offline run logs or returns so
              failures remain diagnosable.
            - `attestations`: booleans confirming no hardcoded live observations,
              no removed validation gates, no reordered side effects, no
              discarded recovery branches, no static runtime assumptions, and no
              ad hoc replacement of managed primitives.

            Managed primitive preservation rule:
            - Replacing live primitives with ad hoc logic is not equivalent and
              is not acceptable. For example, replacing `primitives.web.ask(...)`
              with custom `urllib` scraping, replacing contact/task/knowledge
              primitives with direct storage pokes, or bypassing validation and
              recovery primitives with local shortcuts must fail certification.
            - Positive pattern: wrap the same primitive sequence with stable
              parameters, expose only genuinely variable inputs, preserve side
              effect ordering, keep validation/recovery branches, and leave
              managed surfaces under their manager-owned primitives.
            - Antipatterns: hardcoding observations from the live run, flattening
              a multi-step primitive workflow into one request, using raw
              HTTP/storage access instead of manager APIs, dropping blocker
              handling, changing output shape, hiding side effects in helpers, or
              swapping in cheaper but behaviorally different data sources.

            Feedback and retries:
            - Rejection returns structured reasons. Use them to revise the stored
              function candidate or evidence, then resubmit only within the
              bounded review budget.
            - This post-run review allows at most two certification evidence
              submissions. When the budget is exhausted, leave future instances
              live. The symbolic candidate may remain recorded only if it is
              still useful as a helper.

            Fail-closed behavior:
            - Missing, contradictory, unsafe, high-risk, or primitive-changing
              evidence is rejected. Rejection never promotes offline delivery.
            """

            nonlocal certification_revision_attempts
            if not callable(promote_entrypoint_offline):
                return "No task offline-promotion hook is available."
            if (
                certification_revision_attempts
                >= MAX_OFFLINE_CERTIFICATION_REVISION_ATTEMPTS
            ):
                return str(
                    {
                        "outcome": "certification_revision_attempts_exhausted",
                        "task_id": task_id,
                        "completed_instance_id": instance_id,
                        "function_id": int(function_id),
                        "max_revision_attempts": (
                            MAX_OFFLINE_CERTIFICATION_REVISION_ATTEMPTS
                        ),
                        "next_action": (
                            "Leave future instances live unless a later task "
                            "run produces a better candidate."
                        ),
                    },
                )

            certification_revision_attempts += 1
            certification_metadata = {
                "certification_evidence": certification_evidence,
                "promotion_rationale": promotion_rationale or "",
                "certification_attempt": certification_revision_attempts,
                "max_revision_attempts": (MAX_OFFLINE_CERTIFICATION_REVISION_ATTEMPTS),
            }
            certification_result = {
                "evidence_based": True,
                "executed_entrypoint": False,
                "attempt": certification_revision_attempts,
                "max_revision_attempts": (MAX_OFFLINE_CERTIFICATION_REVISION_ATTEMPTS),
            }
            outcome = promote_entrypoint_offline(
                function_id=int(function_id),
                certification_metadata=certification_metadata,
                certification_result=certification_result,
            )
            remaining_attempts = max(
                0,
                MAX_OFFLINE_CERTIFICATION_REVISION_ATTEMPTS
                - certification_revision_attempts,
            )
            outcome["certification_attempt"] = certification_revision_attempts
            outcome["remaining_revision_attempts"] = remaining_attempts
            if outcome.get("outcome") == "certification_rejected":
                outcome["feedback"] = (
                    "Use rejection_reasons to revise the candidate function or "
                    "evidence. Do not execute the candidate through certification."
                )
                if remaining_attempts == 0:
                    outcome["certification_feedback_status"] = (
                        "revision_attempts_exhausted"
                    )
                    outcome["next_action"] = "Keep future task instances live."
            return str(outcome)

        submit_offline_certification_evidence.__doc__ += (
            f"\n\nCurrent task: {task_name} "
            f"(task_id={task_id}, completed instance_id={instance_id}). "
            "This tool may patch future non-terminal instances to offline "
            "delivery only after evidence-based certification passes."
        )
        tools["submit_offline_certification_evidence"] = (
            submit_offline_certification_evidence
        )

    return tools, storage_active_lines


# ---------------------------------------------------------------------------
# Storage check: start a review loop and return its handle
# ---------------------------------------------------------------------------


def _start_storage_check_loop(
    *,
    trajectory: list[dict],
    ask_tools: dict,
    completed_tool_metadata: dict | None = None,
    actor: "CodeActActor",
    original_result: str,
    parent_lineage: list[str] | None = None,
    stop_reason: str | None = None,
    proactive_summaries: list[str] | None = None,
    post_run_review_context: PostRunReviewContext | None = None,
) -> "AsyncToolLoopHandle | None":
    """Start a loop that reviews a completed trajectory for reusable knowledge.

    The loop maintains two complementary stores:

    * **FunctionManager** — stores the *what*: concrete, reusable function
      implementations (the building blocks).
    * **GuidanceManager** — stores the *how*: high-level guidance on
      composing multiple functions together to accomplish broader tasks
      (the recipes / playbooks).

    Both stores are required. Returns ``None`` when either manager is
    missing.
    """
    fm = actor.function_manager
    gm = actor.guidance_manager
    if fm is None or gm is None:
        return None
    task_entrypoint_review = (
        post_run_review_context.extensions.get("task_entrypoint_review")
        if post_run_review_context is not None
        else None
    )

    tools, storage_active_lines = _build_storage_tools(
        actor=actor,
        ask_tools=ask_tools,
        completed_tool_metadata=completed_tool_metadata,
        task_entrypoint_review=task_entrypoint_review,
    )

    # ── Build prompt ──────────────────────────────────────────────────

    trajectory_json = json.dumps(trajectory, indent=2, default=str)

    # Build optional section about inner storage loops.
    inner_storage_section = ""
    if storage_active_lines:
        inner_storage_section = (
            "## Inner Storage Loops\n\n"
            "Some inner tools from this trajectory are currently running "
            "their own background skill-review loops:\n\n"
            + "\n".join(storage_active_lines)
            + "\n\n"
            "These inner loops may be storing functions independently at a "
            "finer granularity. You can:\n"
            "- Query them via `ask_about_completed_tool`\n"
            "- Inject directives via `interject_inner_storage`\n"
            "- Stop them via `stop_inner_storage`\n"
            "- Pause/resume them via `pause_inner_storage` / "
            "`resume_inner_storage`\n\n"
            "Use these to coordinate storage decisions (e.g. stop an inner "
            "loop that would store something redundant, or interject context "
            "about what you plan to store at the higher level).\n\n"
        )

    # ── Proactive storage awareness ───────────────────────────────────
    proactive_storage_section = ""
    if proactive_summaries:
        summaries_text = "\n\n".join(
            f"**Proactive pass {i + 1}:**\n{s}"
            for i, s in enumerate(proactive_summaries)
        )
        proactive_storage_section = (
            "## Proactive Storage Already Performed\n\n"
            "The executing agent proactively triggered skill storage during "
            "this run via the `store_skills` tool. Below are the summaries "
            "from each proactive storage pass:\n\n"
            f"{summaries_text}\n\n"
            "Check the function and guidance stores to confirm what was "
            "already added. Do not duplicate existing entries. Focus on "
            "any additional reusable patterns — especially from sections "
            "of the trajectory *after* the last `store_skills` call — "
            "that the proactive passes may have missed.\n\n"
        )

    instructions = _STORAGE_BASE_INSTRUCTIONS
    if proactive_summaries:
        instructions = (
            "## Instructions\n\n"
            "1. Skill storage was proactively triggered during this run. "
            "Start by reviewing the proactive storage summaries above and "
            "checking the function and guidance stores to see what was "
            "already added.\n"
            "2. Search the existing stores to confirm exactly what was stored "
            "(use the search/filter tools for each store).\n"
            "3. Review the full trajectory — especially sections after the "
            "last `store_skills` call — for any additional reusable patterns "
            "the proactive passes may have missed.\n"
            "4. Do not duplicate entries that already exist. Only add, update, "
            "or merge if there is genuinely new value.\n"
            "5. When done (or if there is nothing more to add), respond "
            "with a brief summary of what you did (or that nothing additional "
            "was needed)."
        )

    stop_context_section = ""
    if stop_reason:
        stop_context_section = (
            "## Session Termination Context\n\n"
            "This session was explicitly stopped by the user. The stop reason "
            "provides important signal about whether the user intended the "
            "work to be saved:\n\n"
            f"> {stop_reason}\n\n"
            "Weigh this context when deciding what to store. If the reason "
            "indicates the user wanted the workflow remembered or saved, that "
            "is a strong positive signal — look for reusable patterns in the "
            "trajectory. If the reason indicates cancellation or abandonment, "
            "the trajectory is less likely to contain patterns worth "
            "persisting, though genuinely reusable sub-patterns may still "
            "be worth storing.\n\n"
        )

    task_entrypoint_section = ""
    if task_entrypoint_review:
        metadata = dict(task_entrypoint_review.get("metadata") or {})
        metadata_json = json.dumps(metadata, indent=2, default=str)
        task_entrypoint_section = (
            "## Recurring Task Entrypoint Review\n\n"
            "This trajectory completed a scheduled or triggered task that had "
            "no stored entrypoint when it ran. You must explicitly consider "
            "whether the successful run revealed a stable reusable workflow "
            "worth attaching to future task instances.\n\n"
            "No-op is valid: keep the task description-driven if future runs "
            "need broad planning, changing tool discovery, or open-ended "
            "judgment. If the workflow can be stabilized as code, it may still "
            "use focused `query_llm(...)` calls for bounded semantic substeps "
            "such as summarization, classification, ranking, drafting, or "
            "source selection.\n\n"
            "If you store a FunctionManager function and decide it is a stable "
            "candidate for future runs, call "
            "`attach_entrypoint_to_recurring_task(function_id=..., "
            "rationale=..., equivalence_manifest=..., "
            "anti_oversimplification_checklist=...)`. Calling that tool records "
            "a symbolic executor candidate on future non-terminal instances; it "
            "does not promote them to offline delivery. Do not call it unless "
            "the function has already been persisted and you have the numeric "
            "function_id.\n\n"
            "Executor candidates require an equivalence manifest. Include: "
            "required inputs; managed primitives/helpers/managers used; external "
            "capabilities; side effects and ordering; expected result shape; "
            "failure semantics; and a live-step to function-code-path mapping. "
            "The anti-oversimplification checklist must confirm there are no "
            "hardcoded observations from live tool results unless they are true "
            "task constants, no removed validation gates, no reordered side "
            "effects, no discarded recovery branches, and no replacement of "
            "runtime-dependent decisions with static assumptions. If the "
            "candidate materially changes primitives, inputs, ordering, or "
            "failure behavior, store it as a helper/guidance only and do not "
            "record it as the task executor candidate.\n\n"
            "Offline promotion is a separate evidence-based certification "
            "decision. Only call `submit_offline_certification_evidence(...)` "
            "after the candidate is recorded and you can provide complete "
            "evidence for equivalence, inputs, side effects, idempotency, cost, "
            "failure behavior, observability, and managed primitive "
            "preservation. The certification tool does not execute the "
            "entrypoint or replay live task steps. Replacing managed primitives "
            "with ad hoc logic is not equivalent: for example, do not replace "
            "`primitives.web.ask(...)` with custom scraping or manager-owned "
            "primitives with raw storage/HTTP shortcuts. Failed certification "
            "keeps future runs live while preserving the stored artifact for "
            "reuse.\n\n"
            "Task metadata:\n"
            f"```json\n{metadata_json}\n```\n\n"
        )

    system_prompt = (
        "You are a skill librarian. A CodeActActor has just completed a task. "
        "Your job is to review the execution trajectory and decide whether "
        "anything is worth persisting for future reuse. Often nothing is — "
        "that is perfectly fine.\n\n"
        "## Completed Trajectory\n\n"
        f"{trajectory_json}\n\n"
        "## Final Result\n\n"
        f"{original_result}\n\n"
        f"{stop_context_section}"
        f"{task_entrypoint_section}"
        f"{inner_storage_section}"
        f"{proactive_storage_section}"
        f"{_STORAGE_WHAT_CAN_BE_STORED}"
        f"{_STORAGE_TWO_STORES}"
        f"{_STORAGE_SUB_AGENT_PATTERNS}"
        f"{instructions}"
    )

    client = new_llm_client(actor._model)
    client.set_system_message(system_prompt)

    return start_async_tool_loop(
        client=client,
        message=(
            "Review the trajectory and store any reusable functions "
            "and compositional guidance."
        ),
        tools=tools,
        loop_id="StorageCheck(CodeActActor.act)",
        parent_lineage=parent_lineage,
    )


# ---------------------------------------------------------------------------
# Proactive storage: on-demand storage loop triggered from the doing loop
# ---------------------------------------------------------------------------


def _start_proactive_storage_loop(
    *,
    trajectory: list[dict],
    ask_tools: dict,
    completed_tool_metadata: dict | None = None,
    actor: "CodeActActor",
    request: str,
    parent_lineage: list[str] | None = None,
) -> "AsyncToolLoopHandle | None":
    """Start an on-demand storage review loop triggered mid-flight by the doing loop.

    Shares the same tool set and core prompt sections as the post-processing
    ``_start_storage_check_loop``, but uses a distinct prompt framing:
    the trajectory is partial (task still in progress), there is no final
    result, and the ``request`` parameter focuses the reviewer on specific
    skills worth storing.

    Returns ``None`` when either FunctionManager or GuidanceManager is
    missing.
    """
    fm = actor.function_manager
    gm = actor.guidance_manager
    if fm is None or gm is None:
        return None

    tools, storage_active_lines = _build_storage_tools(
        actor=actor,
        ask_tools=ask_tools,
        completed_tool_metadata=completed_tool_metadata,
    )

    # ── Build prompt ──────────────────────────────────────────────────

    trajectory_json = json.dumps(trajectory, indent=2, default=str)

    inner_storage_section = ""
    if storage_active_lines:
        inner_storage_section = (
            "## Inner Storage Loops\n\n"
            "Some inner tools from this trajectory are currently running "
            "their own background skill-review loops:\n\n"
            + "\n".join(storage_active_lines)
            + "\n\n"
            "These inner loops may be storing functions independently at a "
            "finer granularity. You can:\n"
            "- Query them via `ask_about_completed_tool`\n"
            "- Inject directives via `interject_inner_storage`\n"
            "- Stop them via `stop_inner_storage`\n"
            "- Pause/resume them via `pause_inner_storage` / "
            "`resume_inner_storage`\n\n"
            "Use these to coordinate storage decisions (e.g. stop an inner "
            "loop that would store something redundant, or interject context "
            "about what you plan to store at the higher level).\n\n"
        )

    instructions = (
        "## Instructions\n\n"
        "1. Review the trajectory so far, focusing on the storage request.\n"
        "2. Search the existing stores to understand what already exists "
        "(use the search/filter tools for each store).\n"
        "3. Decide what actions (if any) would improve the library based on "
        "the requested skill(s). Prefer a clean, non-redundant library over "
        "a large one.\n"
        "4. When done (or if there is nothing worth storing), respond "
        "with a brief, concrete summary of what you stored (function names, "
        "guidance titles) or that nothing was needed. This summary will be "
        "visible to both the executing agent and a follow-up storage review, "
        "so be specific."
    )

    system_prompt = (
        "You are a skill librarian. A CodeActActor is currently executing "
        "a task and has proactively requested skill storage. Your job is "
        "to review the execution trajectory so far and store the "
        "requested skill(s) for future reuse. Often nothing is worth "
        "storing — that is perfectly fine.\n\n"
        "## Storage Request\n\n"
        f"{request}\n\n"
        "## Trajectory So Far\n\n"
        f"{trajectory_json}\n\n"
        f"{inner_storage_section}"
        f"{_STORAGE_WHAT_CAN_BE_STORED}"
        f"{_STORAGE_TWO_STORES}"
        f"{_STORAGE_SUB_AGENT_PATTERNS}"
        f"{instructions}"
    )

    client = new_llm_client(actor._model)
    client.set_system_message(system_prompt)

    return start_async_tool_loop(
        client=client,
        message=(
            f"The executing agent has proactively requested skill storage: "
            f"{request!r}. Review the trajectory so far and store the "
            f"relevant skills."
        ),
        tools=tools,
        loop_id="ProactiveStorage(CodeActActor.act)",
        parent_lineage=parent_lineage,
    )


class _StorageCheckHandle(SteerableToolHandle):
    """Wraps an inner handle and runs a storage check after task completion.

    Lifecycle phases:

    * **task** -- the inner tool loop is running.  All steering methods
      forward to the inner handle.  Notifications from the inner handle
      are relayed to consumers.
    * **storage** -- the task has completed.  ``result()`` has already
      resolved with the original task result.  A second loop reviews the
      trajectory for reusable skills.  The handle remains live: steering
      methods (ask, interject, stop, pause, resume) operate on the
      storage loop, and ``done()`` returns ``False``.
    * **done** -- both phases have completed (or were stopped/skipped).
      ``done()`` returns ``True``.

    ``result()`` resolves at the end of Phase 1 — callers get the task
    result without waiting for storage.  ``done()`` reflects full
    lifecycle completion (including storage).  This means nested actor
    loops propagate results immediately while storage runs concurrently
    in the background.
    """

    def __init__(
        self,
        *,
        inner: "AsyncToolLoopHandle",
        actor: "CodeActActor",
        post_run_review_context: PostRunReviewContext | None = None,
    ) -> None:
        self._inner = inner
        self._actor = actor
        self._post_run_review_context = post_run_review_context
        self._notification_q: asyncio.Queue[dict] = asyncio.Queue()
        self._task_done_event = asyncio.Event()
        self._completion_event = asyncio.Event()
        self._original_result: Optional[str] = None
        self._storage_handle: Optional["AsyncToolLoopHandle"] = None
        self._phase: str = "task"  # "task" | "storage" | "done"
        self._stopped: bool = False
        self._stop_reason: Optional[str] = None
        self._active_relay: Optional[asyncio.Task] = None

        # Start the two-phase lifecycle manager.
        self._lifecycle_task = asyncio.create_task(self._run_lifecycle())

    # ── Internal helpers ──────────────────────────────────────────────

    @property
    def _pause_event(self):
        """Delegate to the active inner handle so get_handle_paused_state works."""
        handle = self._active_handle
        if handle is not None:
            return getattr(handle, "_pause_event", None)
        return None

    @property
    def _active_handle(self) -> Optional["SteerableToolHandle"]:
        """The currently active inner handle for steering delegation."""
        if self._phase == "task":
            return self._inner
        if self._phase == "storage":
            return self._storage_handle
        return None

    async def _relay_notifications_from(
        self,
        source: "SteerableToolHandle",
    ) -> None:
        """Forward notifications from *source* into our queue until cancelled."""
        try:
            while True:
                notif = await source.next_notification()
                await self._notification_q.put(notif)
        except asyncio.CancelledError:
            pass
        except Exception:
            pass

    async def _cancel_relay(self) -> None:
        """Cancel the active notification relay task, if any."""
        relay = self._active_relay
        if relay is not None and not relay.done():
            relay.cancel()
            try:
                await relay
            except (asyncio.CancelledError, Exception):
                pass
        self._active_relay = None

    # ── Lifecycle ─────────────────────────────────────────────────────

    async def _run_lifecycle(self) -> None:
        """Manage the two-phase lifecycle: task -> storage check -> done."""
        try:
            # ── Phase 1: task execution ───────────────────────────────
            self._active_relay = asyncio.create_task(
                self._relay_notifications_from(self._inner),
            )

            try:
                self._original_result = await self._inner.result()
                task_succeeded = not self._stopped
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                task_succeeded = False
                self._original_result = (
                    f"Error: inner task failed: {type(exc).__name__}: {exc}"
                )
                logger.error(
                    f"_StorageCheckHandle: inner result raised "
                    f"{type(exc).__name__}: {exc}",
                )
            await self._cancel_relay()
            self._task_done_event.set()

            # Snapshot trajectory and ask tools (client/messages are still
            # valid after result() returns -- cleanup only resets context
            # vars and releases the semaphore).
            trajectory: list[dict] = []
            ask_tools: dict = {}
            try:
                client = getattr(self._inner, "_client", None)
                if client is not None:
                    trajectory = make_messages_safe_for_context_dump(
                        list(getattr(client, "messages", []) or []),
                    )
            except Exception:
                pass
            try:
                _get_ask = getattr(
                    self._inner._task,
                    "get_ask_tools",
                    lambda: {},
                )
                ask_tools = _get_ask()
            except Exception:
                pass
            completed_tool_metadata: dict = {}
            try:
                _get_meta = getattr(
                    self._inner._task,
                    "get_completed_tool_metadata",
                    lambda: {},
                )
                completed_tool_metadata = _get_meta()
            except Exception:
                pass

            # ── Phase 2: storage check ────────────────────────────────
            self._phase = "storage"

            _sc_suffix = _token_hex(2)
            _sc_call_id = new_call_id()
            _sc_parent = TOOL_LOOP_LINEAGE.get([])
            _sc_parent_lineage = (
                list(_sc_parent) if isinstance(_sc_parent, list) else []
            )
            _sc_hierarchy = [
                *_sc_parent_lineage,
                f"StorageCheck(CodeActActor.act)({_sc_suffix})",
            ]
            _sc_lineage_token = TOOL_LOOP_LINEAGE.set(_sc_hierarchy)
            _sc_suffix_token = _PENDING_LOOP_SUFFIX.set(_sc_suffix)

            try:
                active_review_context = (
                    self._post_run_review_context if task_succeeded else None
                )
                review_display_label = (
                    active_review_context.display_label
                    if active_review_context is not None
                    else _DEFAULT_STORAGE_REVIEW_LABEL
                )
                review_instructions = (
                    active_review_context.instructions
                    if active_review_context is not None
                    else _DEFAULT_STORAGE_REVIEW_INSTRUCTIONS
                )
                await publish_manager_method_event(
                    _sc_call_id,
                    "CodeActActor",
                    "StorageCheck",
                    phase="incoming",
                    display_label=review_display_label,
                    hierarchy=_sc_hierarchy,
                    instructions=review_instructions,
                )

                proactive_summaries: list[str] = []
                try:
                    _ctx = _CURRENT_AGENT_CONTEXT.get(None)
                    if _ctx is not None:
                        proactive_summaries = list(
                            _ctx.proactive_storage_summaries,
                        )
                except Exception:
                    pass

                storage_handle = _start_storage_check_loop(
                    trajectory=trajectory,
                    ask_tools=ask_tools,
                    completed_tool_metadata=completed_tool_metadata,
                    actor=self._actor,
                    original_result=str(self._original_result),
                    parent_lineage=_sc_parent_lineage,
                    stop_reason=self._stop_reason,
                    proactive_summaries=proactive_summaries or None,
                    post_run_review_context=active_review_context,
                )

                if storage_handle is None:
                    await publish_manager_method_event(
                        _sc_call_id,
                        "CodeActActor",
                        "StorageCheck",
                        phase="outgoing",
                        display_label=review_display_label,
                        hierarchy=_sc_hierarchy,
                    )
                else:
                    self._storage_handle = storage_handle
                    try:
                        await self._storage_handle.result()
                    except Exception as exc:
                        logger.warning(
                            f"StorageCheck failed: {type(exc).__name__}: {exc}",
                        )

                    await publish_manager_method_event(
                        _sc_call_id,
                        "CodeActActor",
                        "StorageCheck",
                        phase="outgoing",
                        display_label=review_display_label,
                        hierarchy=_sc_hierarchy,
                    )
            finally:
                _PENDING_LOOP_SUFFIX.reset(_sc_suffix_token)
                TOOL_LOOP_LINEAGE.reset(_sc_lineage_token)

        except asyncio.CancelledError:
            pass
        except Exception:
            pass
        finally:
            self._phase = "done"
            self._task_done_event.set()
            self._completion_event.set()

    # ── Steering: phase-aware forwarding ──────────────────────────────

    async def ask(
        self,
        question: str,
        *,
        _parent_chat_context: list[dict] | None = None,
        **kwargs,
    ) -> "SteerableToolHandle":
        # Task and done phases: ask about the completed/running task.
        if self._phase != "storage":
            return await self._inner.ask(
                question,
                _parent_chat_context=_parent_chat_context,
                **kwargs,
            )

        # ── Storage phase: thin routing loop ──────────────────────────
        inner_ref = self._inner
        storage_ref = self._storage_handle
        pcc = _parent_chat_context

        async def ask_about_task(question: str) -> str:
            """Ask a question about the **completed task** itself.

            Use this for anything related to:
            - What the task was and what the agent did to accomplish it
            - The reasoning, tool calls, or intermediate steps taken
            - The final result or output of the task
            - Errors or issues encountered during execution

            This queries the full execution trajectory of the finished
            task, NOT the skill-storage process that is running now.
            """
            h = await inner_ref.ask(question, _parent_chat_context=pcc)
            return await h.result()

        async def ask_about_skill_storage(question: str) -> str:
            """Ask a question about the **ongoing skill storage** process.

            Use this for anything related to:
            - Which functions are being considered for storage
            - What the skill librarian has stored, merged, or deleted so far
            - Progress or status of the skill consolidation review
            - Decisions about whether a function is worth keeping

            This queries the live storage-check loop that is reviewing
            the completed trajectory for reusable patterns, NOT the
            original task itself.
            """
            if storage_ref is not None:
                h = await storage_ref.ask(question, _parent_chat_context=pcc)
                return await h.result()
            return "Skill storage has not started yet."

        routing_tools: Dict[str, Callable] = {
            "ask_about_task": ask_about_task,
            "ask_about_skill_storage": ask_about_skill_storage,
        }

        routing_client = new_llm_client()
        routing_client.set_system_message(
            "You are answering a question about an agent that has completed "
            "its primary task and is now reviewing its execution trajectory "
            "to store reusable skills.\n\n"
            "You have two tools:\n"
            "- ask_about_task: for questions about the completed task, its "
            "approach, reasoning, or result\n"
            "- ask_about_skill_storage: for questions about the ongoing "
            "skill consolidation process\n\n"
            "Route the question to the appropriate tool. If the question "
            "spans both topics, call both tools and synthesize the answers.",
        )

        return start_async_tool_loop(
            client=routing_client,
            message=question,
            tools=routing_tools,
            loop_id="Question(StorageCheck.routing)",
        )

    async def interject(
        self,
        message: str,
        *,
        _parent_chat_context_cont: list[dict] | None = None,
        **kwargs,
    ) -> None:
        handle = self._active_handle
        if handle is not None:
            return await handle.interject(
                message,
                _parent_chat_context_cont=_parent_chat_context_cont,
                **kwargs,
            )

    async def stop(self, reason: Optional[str] = None, **kwargs) -> None:
        self._stopped = True
        self._stop_reason = reason
        handle = self._active_handle
        if handle is not None:
            await handle.stop(reason=reason, **kwargs)

    async def pause(self, **kwargs) -> Optional[str]:
        handle = self._active_handle
        if handle is not None:
            return await handle.pause(**kwargs)
        return None

    async def resume(self, **kwargs) -> Optional[str]:
        handle = self._active_handle
        if handle is not None:
            return await handle.resume(**kwargs)
        return None

    # ── Completion ────────────────────────────────────────────────────

    def done(self) -> bool:
        return self._completion_event.is_set()

    async def result(self) -> str:
        await self._task_done_event.wait()
        if self._stopped and self._stop_reason:
            return (
                f"Task stopped as requested. Reason: {self._stop_reason}\n"
                f"Background skill storage is reviewing the completed work."
            )
        return self._original_result or ""

    # ── Events ────────────────────────────────────────────────────────

    async def next_clarification(self) -> dict:
        handle = self._active_handle
        if handle is not None:
            return await handle.next_clarification()
        # Done: block forever (no more clarifications expected).
        await asyncio.Event().wait()
        return {}

    async def next_notification(self) -> dict:
        return await self._notification_q.get()

    async def answer_clarification(self, call_id: str, answer: str) -> None:
        handle = self._active_handle
        if handle is not None:
            return await handle.answer_clarification(call_id, answer)

    def get_history(self) -> list[dict]:
        return self._inner.get_history()


# ---------------------------------------------------------------------------
# Code synthesis helpers for execute_function
# ---------------------------------------------------------------------------


def _synthesize_python_call(
    *,
    function_name: str,
    call_kwargs: Dict[str, Any],
    function_manager: Optional["FunctionManager"] = None,
) -> str:
    """Build a Python code snippet that calls *function_name* with *call_kwargs*.

    This function only synthesises the **code string** — it does not handle
    dependency injection.  Transitive dependencies (both bare compositional
    functions and dotted environment namespaces like ``actor`` or
    ``primitives``) are injected into the sandbox namespace *before* this
    code runs, through a separate path:

    * When the LLM discovers a function via ``FunctionManager_search_functions``
      (or filter/list), the FM's ``_inject_callables_for_functions`` calls
      ``_inject_dependencies``, which resolves every entry in ``depends_on``
      and places the result into the sandbox's ``global_state``.
    * Environment namespaces (``actor``, ``primitives``, etc.) are also
      already present in the sandbox if the CodeActActor was constructed
      with the corresponding environments.

    So by the time this synthesised code executes, all names the function
    references — whether bare helpers or dotted environment calls — are
    already available in scope.

    Resolution order for the *function itself*:
    1. Emit a plain call expression.  The sandbox namespace already contains
       environment-injected callables and previously-discovered FM functions,
       so this is the common-case fast path.
    2. If the FunctionManager has a stored implementation, prepend it as a
       preamble (defining the function) so the call works even in a fresh
       stateless session where discovery hasn't run yet.

    The call expression is always the **last expression** so that
    ``PythonExecutionSession``'s REPL semantics return its value (including
    steerable handles from primitives).
    """
    kwargs_repr = repr(call_kwargs) if call_kwargs else "{}"
    # Determine whether the function is async by inspecting the stored impl.
    # Default to ``await`` — environment-injected callables (primitives,
    # manager methods) are async, and ``await`` on a sync return value
    # produces a clear ``TypeError`` rather than silently discarding a
    # coroutine.
    is_async = True
    preamble = ""

    if function_manager is not None:
        func_data = function_manager._get_function_data_by_name(name=function_name)
        if func_data is None and getattr(
            function_manager,
            "_include_primitives",
            False,
        ):
            func_data = function_manager._get_primitive_data_by_name(name=function_name)
        if func_data is None and getattr(
            function_manager,
            "_include_primitives",
            False,
        ):
            get_stored_primitive = getattr(
                function_manager,
                "_get_stored_primitive_data_by_name",
                None,
            )
            if callable(get_stored_primitive):
                func_data = get_stored_primitive(name=function_name)

        if func_data is not None:
            impl = func_data.get("implementation")
            if impl and isinstance(impl, str) and impl.strip():
                is_async = "async def" in impl
                # Strip @custom_function decorators (not available in sandbox).
                from unify.function_manager.function_manager import (
                    _strip_custom_function_decorators,
                )

                preamble = _strip_custom_function_decorators(impl) + "\n\n"
            elif func_data.get("is_primitive"):
                is_async = True

    call_expr = f"{'await ' if is_async else ''}{function_name}(**{kwargs_repr})"
    return f"{preamble}{call_expr}"


def _synthesize_shell_call(
    *,
    function_name: str,
    call_kwargs: Dict[str, Any],
    function_manager: Optional["FunctionManager"] = None,
) -> str:
    """Build a shell script that runs the stored function with *call_kwargs*.

    Shell functions must have a stored implementation in the FunctionManager.
    ``call_kwargs`` are exported as environment variables before sourcing the
    implementation.
    """
    impl: str | None = None
    if function_manager is not None:
        func_data = function_manager._get_function_data_by_name(name=function_name)
        if func_data is not None:
            impl = func_data.get("implementation")

    if not impl or not isinstance(impl, str) or not impl.strip():
        raise ValueError(
            f"Shell function '{function_name}' has no stored implementation.",
        )

    # Export kwargs as environment variables.
    exports: list[str] = []
    for k, v in (call_kwargs or {}).items():
        escaped = str(v).replace("'", "'\\''")
        exports.append(f"export {k}='{escaped}'")

    parts = exports + [impl]
    return "\n".join(parts)


class CodeActActor(BaseCodeActActor):
    """
    An actor that uses a conversational tool loop and a stateful code execution
    sandbox to accomplish tasks. It acts as a baseline for code-centric agents.
    """

    def __init__(
        self,
        *,
        environments: Optional[list["BaseEnvironment"]] = None,
        function_manager: Optional["FunctionManager"] = None,
        guidance_manager: Optional["GuidanceManager"] = None,
        can_compose: object = _UNSET,
        can_store: object = _UNSET,
        timeout: object = _UNSET,
        model: object = _UNSET,
        preprocess_msgs: Optional[Callable[[list[dict]], list[dict]]] = None,
        prompt_caching: object = _UNSET,
        guidelines: object = _UNSET,
        tool_policy: Union[ToolPolicyFn, None, object] = _USE_DEFAULT,
    ):
        """
        Initializes the CodeActActor.

        Args:
            environments: List of execution environments to install. Each environment
                injects a namespace into the sandbox (e.g. ``primitives``,
                ``primitives.computer``, ``primitives.actor``). Pass ``None`` or ``[]``
                for a bare actor with no environments.
            function_manager: Manages a library of reusable functions. Exposes read-only tools
                (list_functions, search_functions, filter_functions) to the LLM.
                The LLM can call these tools to discover and retrieve reusable function implementations.
            guidance_manager: Manages high-level guidance entries that describe *how* to
                compose functions together for tasks. Exposes read/write tools in the
                post-completion storage check loop alongside FunctionManager tools.
            can_compose: Whether the LLM can write and execute arbitrary code via
                ``execute_code``. Set to False for function-execution-only mode.
            can_store: Whether a post-completion review loop should run to
                identify and store reusable functions and guidance from the
                trajectory. Storage is always deferred to a dedicated second
                loop after the main task completes — the main loop never
                exposes storage tools.
            timeout: Maximum seconds for individual code execution in sessions.
            model: Optional LLM model identifier. If None, uses SETTINGS.UNIFY_MODEL.
            preprocess_msgs: Optional callback to modify messages before each LLM call.
                Receives a list of message dicts and returns a modified list.
                Useful for pruning old messages, adding context, or transforming content.
            prompt_caching: Optional list of cache targets (e.g. ["system", "messages"]).
                Enables Anthropic prompt caching for the specified components to reduce
                costs and latency. Valid values: "tools", "system", "messages".
            guidelines: Persistent behavioral guidelines applied to every ``act()``
                invocation.  Per-invocation ``guidelines`` passed to ``act()`` are
                appended after these, so the constructor value acts as a baseline
                and ``act()`` adds task-specific refinements on top.
            tool_policy: Controls per-turn dynamic tool filtering and tool-choice mode.
                - ``_USE_DEFAULT`` (default): uses the built-in "discovery-first"
                  policy that requires both a FunctionManager and a GuidanceManager
                  discovery call before unlocking the full tool set.
                - A custom ``ToolPolicyFn`` callable: receives ``(step, tools)`` and
                  returns ``(mode, filtered_tools)``.  Static filters (``can_compose``,
                  ``can_store``, etc.) are always applied before the custom policy sees
                  the tools.
                - ``None``: no dynamic policy; only the static ``can_compose`` /
                  ``can_store`` filters apply.
        """
        super().__init__(
            environments=environments or [],
            function_manager=function_manager,
            guidance_manager=guidance_manager,
        )

        can_compose = can_compose if can_compose is not _UNSET else True
        can_store = can_store if can_store is not _UNSET else True
        timeout = timeout if timeout is not _UNSET else 3600.0
        model = model if model is not _UNSET else None
        prompt_caching = (
            prompt_caching
            if prompt_caching is not _UNSET
            else ("system", "tools", "messages")
        )
        guidelines = guidelines if guidelines is not _UNSET else None
        self._base_guidelines = guidelines

        # Collect function_ids from all environments, split by context, and set
        # them on the FunctionManager via setters. This prevents overlap between
        # prompt-injected environment tools and FunctionManager-discoverable
        # functions. We update in-place rather than replacing the FM instance so
        # that callers who pass a custom FM (e.g., SimulatedFunctionManager) keep
        # their instance intact.
        if self.function_manager is not None:
            _excl_primitive: set[int] = set()
            _excl_compositional: set[int] = set()
            for env in self.environments.values():
                for tool_meta in env.get_tools().values():
                    if tool_meta.function_id is not None:
                        if tool_meta.function_context == "primitive":
                            _excl_primitive.add(tool_meta.function_id)
                        elif tool_meta.function_context == "compositional":
                            _excl_compositional.add(tool_meta.function_id)

            if _excl_primitive:
                self.function_manager.exclude_primitive_ids = frozenset(
                    _excl_primitive,
                )
            if _excl_compositional:
                self.function_manager.exclude_compositional_ids = frozenset(
                    _excl_compositional,
                )
            try:
                from unify.integration_status import build_function_filter_scope

                function_scope = build_function_filter_scope()
                if function_scope:
                    current = getattr(self.function_manager, "filter_scope", None)
                    self.function_manager.filter_scope = (
                        f"({current}) and ({function_scope})"
                        if current
                        else function_scope
                    )
            except Exception:
                pass

        if self.guidance_manager is not None:
            try:
                from unify.integration_status import build_guidance_filter_scope

                guidance_scope = build_guidance_filter_scope()
                if guidance_scope:
                    current = getattr(self.guidance_manager, "filter_scope", None)
                    self.guidance_manager.filter_scope = (
                        f"({current}) and ({guidance_scope})"
                        if current
                        else guidance_scope
                    )
            except Exception:
                pass

        # Create persistent pools that survive across act() calls
        from unify.function_manager.function_manager import VenvPool
        from unify.function_manager.shell_pool import ShellPool

        self._venv_pool = VenvPool()
        self._shell_pool = ShellPool()
        self._session_executor = SessionExecutor(
            venv_pool=self._venv_pool,
            shell_pool=self._shell_pool,
            environments=self.environments,
            computer_primitives=self._computer_primitives,
            function_manager=self.function_manager,
            timeout=timeout,
        )

        # Session name registry: name -> (language, venv_id, session_id)
        self._session_names: Dict[str, SessionKey] = {}
        # Reverse map: (language, venv_id, session_id) -> set(names)
        self._session_names_rev: Dict[SessionKey, set[str]] = {}
        # Actor-level session cap (global across languages for this actor instance).
        self._max_sessions_total: int = 20
        self._next_session_id: dict[tuple[str, Optional[int]], int] = {}

        self.can_compose: bool = bool(can_compose)
        self.can_store: bool = bool(can_store)
        self.tool_policy: Union[ToolPolicyFn, None, object] = tool_policy
        self._model = model
        self._preprocess_msgs = preprocess_msgs
        self._prompt_caching = prompt_caching
        self._computer_tools = (
            self._get_computer_tools()
        )  # Register stable tools once; per-call sandboxes are bound via _CURRENT_SANDBOX.
        self.add_tools("act", self._build_tools())

        self._main_event_loop: Optional[asyncio.AbstractEventLoop] = None
        try:
            self._main_event_loop = asyncio.get_running_loop()
        except RuntimeError:
            pass

        # Concurrency guard: limit active sandboxes per actor instance.
        self._act_semaphore = asyncio.Semaphore(20)
        # Timeout used when acquiring the semaphore (prevents unbounded waits).
        self._act_semaphore_timeout_s: float = 30.0
        self._active_work_heartbeat_interval_s: float = 60.0
        self._active_work_fallback_initial_delay_s: float = 120.0
        self._active_work_fallback_repeat_interval_s: float = 300.0

    # ───────────────────────── Session name registry ─────────────────────── #

    def _register_session_name(
        self,
        *,
        name: str,
        language: str,
        venv_id: int | None,
        session_id: int,
    ) -> None:
        key: SessionKey = (language, venv_id, int(session_id))
        existing = self._session_names.get(name)
        if existing is not None and existing != key:
            raise ValueError(
                f"Session name {name!r} is already bound to {existing}, cannot rebind to {key}.",
            )
        self._session_names[name] = key
        self._session_names_rev.setdefault(key, set()).add(name)

    def _resolve_session_name(self, name: str) -> SessionKey | None:
        return self._session_names.get(name)

    def _get_session_name(
        self,
        *,
        language: str,
        venv_id: int | None,
        session_id: int,
    ) -> str | None:
        key: SessionKey = (language, venv_id, int(session_id))
        names = self._session_names_rev.get(key)
        if not names:
            return None
        # Prefer stable ordering for determinism.
        return sorted(names)[0]

    def _unregister_session_name(self, name: str) -> None:
        key = self._session_names.pop(name, None)
        if key is None:
            return
        names = self._session_names_rev.get(key)
        if names is not None:
            names.discard(name)
            if not names:
                self._session_names_rev.pop(key, None)

    def _unregister_all_names_for_session(self, *, key: SessionKey) -> None:
        names = self._session_names_rev.pop(key, None)
        if not names:
            return
        for n in list(names):
            self._session_names.pop(n, None)

    def _count_active_sessions_total(self) -> int:
        # Count unique in-process python sessions + persistent pool sessions.
        n = 0
        try:
            n += len(
                self._session_executor._python_sessions,
            )  # pylint: disable=protected-access
        except Exception:
            pass
        try:
            n += len(self._shell_pool.get_active_sessions())
        except Exception:
            pass
        try:
            n += len(self._venv_pool.list_active_sessions())
        except Exception:
            pass
        return n

    def _session_exists(
        self,
        *,
        language: str,
        venv_id: int | None,
        session_id: int,
    ) -> bool:
        if language == "python":
            if venv_id is None:
                return self._session_executor.has_python_session(
                    session_id=int(session_id),
                    venv_id=None,
                )
            # venv-backed python session exists if pool has it active
            try:
                return (int(venv_id), int(session_id)) in set(
                    self._venv_pool.list_active_sessions(),
                )
            except Exception:
                return False
        # shell
        try:
            return self._shell_pool.has_session(
                language=language,  # type: ignore[arg-type]
                session_id=int(session_id),
            )
        except Exception:
            return False

    def _validate_execution_params(
        self,
        *,
        state_mode: str,
        session_id: int | None,
        session_name: str | None,
        language: str,
        venv_id: int | None = None,
    ) -> dict | None:
        return _validate_execution_params(
            state_mode=state_mode,
            session_id=session_id,
            session_name=session_name,
            language=language,
            venv_id=venv_id,
            resolve_session_name=self._resolve_session_name,
            get_session_name_for_id=lambda l, v, s: self._get_session_name(
                language=l,
                venv_id=v,
                session_id=s,
            ),
            session_exists=lambda l, v, s: self._session_exists(
                language=l,
                venv_id=v,
                session_id=s,
            ),
            max_sessions_total=self._max_sessions_total,
            active_session_count=self._count_active_sessions_total(),
        )

    def _resolve_session(
        self,
        *,
        state_mode: str,
        language: str,
        session_id: int | None,
        session_name: str | None,
        venv_id: int | None,
    ) -> _ResolvedSession:
        """Resolve/allocate a session and validate execution params.

        Handles the full session resolution flow used by both ``execute_code``
        and ``execute_function``:

        1. For stateful mode: resolve an existing session name, allocate a new
           session id, or default to session 0.
        2. Register session name aliases when both name and id are provided.
        3. Validate the resulting execution parameters.

        Returns a ``_ResolvedSession`` named tuple.  If ``error`` is not
        ``None``, the caller should return it as the tool result immediately.
        """
        # Resolve / allocate sessions for stateful.
        if state_mode == "stateful":
            if session_name:
                resolved = self._resolve_session_name(session_name)
                if resolved is not None:
                    language, venv_id, session_id = resolved
                elif session_id is None:
                    key = (
                        str(language),
                        int(venv_id) if venv_id is not None else None,
                    )
                    next_id = self._next_session_id.get(key, 1)
                    session_id = next_id
                    self._next_session_id[key] = next_id + 1
                    self._register_session_name(
                        name=session_name,
                        language=str(language),
                        venv_id=venv_id,
                        session_id=int(session_id),
                    )
            elif session_id is None:
                session_id = 0

        # If name + id are both set but not registered yet, register alias.
        if state_mode == "stateful" and session_name and session_id is not None:
            if self._resolve_session_name(session_name) is None:
                self._register_session_name(
                    name=session_name,
                    language=str(language),
                    venv_id=venv_id,
                    session_id=int(session_id),
                )

        # Validate.
        err = self._validate_execution_params(
            state_mode=state_mode,
            session_id=session_id,
            session_name=session_name,
            language=str(language),
            venv_id=venv_id,
        )

        return _ResolvedSession(
            language=str(language),
            venv_id=venv_id,
            session_id=session_id,
            error=err,
        )

    async def _execute_on_surface(
        self,
        *,
        surface_name: str,
        code: str,
        language: str,
        state_mode: str,
        session_id: int | None,
        session_name: str | None,
        venv_id: int | None,
        user_id: str | None,
    ) -> dict[str, Any]:
        """Run code on a non-local surface (assistant desktop or user desktop).

        Remote surfaces are stateless one-shots: sessions and venvs are
        local-only concepts, so a session/venv request is rejected with a
        structured error the model can self-correct against, rather than being
        silently ignored.
        """
        import time as _surface_time

        from unify.actor.execution.surface import ExecutionSurface
        from unify.actor.execution.targets import (
            TargetUnavailableError,
            get_target,
        )

        def _err(message: str, suggestion: str) -> dict[str, Any]:
            return {
                "stdout": "",
                "stderr": "",
                "result": None,
                "error": message,
                "suggestion": suggestion,
                "language": language,
                "state_mode": state_mode,
                "session_id": None,
                "session_name": None,
                "venv_id": None,
                "session_created": False,
                "duration_ms": 0,
                "surface": surface_name,
            }

        try:
            surface = ExecutionSurface(surface_name)
        except ValueError:
            return _err(
                f"Unknown surface: {surface_name!r}",
                "Use one of: 'local', 'assistant_desktop', 'user_desktop'.",
            )

        if (
            state_mode != "stateless"
            or session_id is not None
            or session_name is not None
            or venv_id is not None
        ):
            return _err(
                f"Surface {surface_name!r} supports only stateless execution.",
                "Remove state_mode/session_id/session_name/venv_id (remote "
                "surfaces are stateless), or use surface='local' for sessions "
                "and venvs.",
            )

        t0 = _surface_time.perf_counter()
        try:
            target = get_target(
                surface,
                user_id=user_id,
                session_executor=self._session_executor,
                function_manager=self.function_manager,
            )
            await target.ensure_ready()
            if language == "python":
                res = await target.run_python(code)
            else:
                res = await target.run_shell(code)
        except TargetUnavailableError as e:
            return _err(
                str(e),
                "Check that the desktop is linked, reachable, and (for the "
                "user desktop) that the user has granted access.",
            )
        except ValueError as e:
            return _err(
                str(e),
                "Adjust the request to match the surface's capabilities.",
            )

        return {
            "stdout": res.stdout,
            "stderr": res.stderr,
            "result": res.result if res.result is not None else res.returncode,
            "error": res.error,
            "returncode": res.returncode,
            "language": language,
            "state_mode": "stateless",
            "session_id": None,
            "session_name": None,
            "venv_id": None,
            "session_created": False,
            "duration_ms": int((_surface_time.perf_counter() - t0) * 1000),
            "surface": surface_name,
        }

    def _get_computer_tools(self) -> Dict[str, Callable]:
        """Extracts computer-related methods from the desktop namespace."""
        if not self._computer_primitives:
            return {}
        desktop = self._computer_primitives.desktop
        return {
            "navigate": desktop.navigate,
            "act": desktop.act,
            "observe": desktop.observe,
        }

    def _get_extra_ask_tools(self) -> Dict[str, Callable] | None:
        """Build domain-specific ask tools for handle.ask() inspection loops."""
        if self._computer_primitives is None:
            return None

        computer_query = self._computer_primitives.desktop.query

        async def ask_computer_progress(
            question: str,
            *,
            _parent_chat_context: list[dict] | None = None,
        ) -> str:
            """Inspect the in-flight computer action loop via browser-agent memory.

            Use this to check progress/state of ongoing ``session.act(...)``
            work when the inspected transcript lacks enough detail (for example,
            placeholders or terse summaries).             This is memory/history introspection,
            not a fresh page read and not a way to trigger new actions.
            """
            _ = _parent_chat_context
            # This tool is offered to every ``handle.ask()`` inspection loop
            # whenever computer primitives exist, but the work being inspected
            # is often not a browser/computer ``session.act`` (e.g. an SFTP file
            # sync). In those cases the computer-agent backend may be unreachable
            # or have nothing to report. A read-only progress probe failing must
            # not look like an error: degrade to a plain answer so it cannot burn
            # the inspection loop's failure budget or be mistaken for the
            # inspected task failing.
            try:
                return await computer_query(question)
            except Exception as exc:
                return (
                    "No computer-agent progress is available to inspect "
                    f"({type(exc).__name__}). There may be no active browser/"
                    "computer session for this work (for example, a file sync "
                    "or shell command runs outside the computer agent). This is "
                    "not a failure of the underlying task."
                )

        return {"ask_computer_progress": ask_computer_progress}

    async def _run_active_work_heartbeat(
        self,
        active_work: ActiveWorkHandle,
        notification_q: asyncio.Queue[dict] | None,
    ) -> None:
        try:
            while True:
                await asyncio.sleep(self._active_work_heartbeat_interval_s)
                active_work.heartbeat()
                if (
                    notification_q is not None
                    and active_work.fallback_notification_due(
                        initial_delay_s=self._active_work_fallback_initial_delay_s,
                        repeat_interval_s=self._active_work_fallback_repeat_interval_s,
                    )
                ):
                    await notification_q.put(
                        {
                            "type": "notification",
                            "message": "Still working on the code step...",
                            "source": "active_work",
                            "completed": False,
                            "active_work_id": active_work.work_id,
                        },
                    )
                    active_work.record_fallback_notification()
        except asyncio.CancelledError:
            pass

    def _build_tools(self) -> Dict[str, Callable[..., Awaitable[Any]]]:
        """Builds the dictionary of tools available to the LLM."""

        @llm_soft_required(thought="")
        async def execute_code(
            thought: Annotated[
                str,
                "A brief, first-person, one-sentence explanation of what this "
                'code does and why you are running it right now (e.g. "Loading '
                'the data and computing the summary the user asked for."). Shown '
                "to the user as the rationale for this step; always provide it.",
            ],
            code: Optional[str] = None,
            *,
            language: str,
            state_mode: str = "stateless",
            session_id: int | None = None,
            session_name: str | None = None,
            venv_id: int | None = None,
            surface: str = "local",
            user_id: str | None = None,
            _notification_up_q: asyncio.Queue[dict] | None = None,
            _parent_chat_context: list[dict] | None = None,
        ) -> Any:
            """
            Execute arbitrary code in a specified language and state mode.

            **IMPORTANT — single-call rule**: If the task requires only a
            single function or primitive call with no surrounding logic,
            use ``execute_function`` instead. ``execute_code`` is for
            **multi-step composition** — conditional logic, loops, or
            combining multiple primitives/functions where intermediate
            results are needed within the same code block.

            Key concepts
            -----------
            - **language**: "python" | "bash" | "zsh" | "sh" | "powershell"
            - **surface**: which machine to run on.
              - "local" (default): the local host itself — the only surface that
                supports stateful sessions and venvs.
              - "assistant_desktop": the assistant's managed VM.
              - "user_desktop": the user's own linked machine, when the user has
                granted access (pass ``user_id`` to disambiguate when more than
                one user desktop is linked).
              Remote surfaces ("assistant_desktop"/"user_desktop") are **stateless
              one-shots**: ``state_mode`` must be "stateless" and ``session_id`` /
              ``session_name`` / ``venv_id`` must be omitted. Use them to run a
              shell command or a self-contained Python snippet on that machine.
            - **state_mode**:
              - "stateless": fresh execution; no persistence of intermediate variables.
                Environment globals and FunctionManager-discovered functions are
                always available.
              - "stateful": persistent session; state accumulates across calls
              - "read_only": reads from an existing session but does not persist changes
            - **session_id/session_name**:
              - only meaningful for stateful/read_only
              - for stateful: if omitted, defaults to **session_id=0** (the default session)
              - to create an additional stateful session, provide a fresh `session_name` (recommended)
                or an explicit `session_id` > 0
              - **Python session_id=0** is special:
                - If this tool is called from inside a running CodeAct `act()` loop, session 0 maps to the
                  **current per-call Python sandbox** (shared via the ContextVar binding).
                - If no sandbox is bound (e.g. calling the tool directly in a unit test), session 0 behaves
                  like a normal in-process Python session managed by the SessionExecutor.

            Best practices
            --------------
            - Use **stateful** when doing multi-step work (cd then ls; load data then analyze).
            - Use **stateless** for one-off checks or when you need isolation.
            - Use **read_only** to "peek" without mutating state (what-if exploration).
            - Use `list_sessions()` and `inspect_state()` to decide which session to use.

            Output
            ------
            Returns either a dict or an ExecutionResult object with the following fields:

            - **stdout**: For in-process Python, a List[TextPart | ImagePart] preserving
              rich output (text and images from print()/display()). For shell or venv
              execution, a plain string.
            - **stderr**: Same format as stdout (list for in-process Python, string otherwise).
            - **result**: The evaluated result of the last expression (Any), or None.
              If the last expression is a steerable handle, it is automatically
              adopted by the outer loop for mid-flight steering.
            - **error**: Error message string if execution failed, otherwise None.
            - **language**: The language used for execution.
            - **state_mode**: The state mode used ("stateless", "stateful", or "read_only").
            - **session_id**: The session ID (int) if stateful/read_only, otherwise None.
            - **session_name**: The session name alias if one was assigned, otherwise None.
            - **venv_id**: The virtual environment ID if applicable, otherwise None.
            - **session_created**: True if a new session was created by this call.
            - **duration_ms**: Execution duration in milliseconds.

            Runtime credential helpers
            --------------------------
            Python execution globals include ``get_oauth_access_token(provider)``
            for connected-account (BYOD) OAuth. It returns a local capability
            handle (not a raw token) to use with the workspace proxy base URLs
            (``MICROSOFT_GRAPH_BASE`` / ``GOOGLE_DRIVE_BASE`` / ``GOOGLE_API_BASE``);
            the proxy injects the real token and enforces the file-access
            allowlist. Static API keys and provider SDKs that read credentials
            from the environment may still use ``os.environ`` after checking
            available secret names.

            For in-process Python execution with rich output, the result is wrapped in an
            ExecutionResult object (a Pydantic model implementing FormattedToolResult).
            """
            _ = thought  # Thought is logged by the LLM; not used programmatically.
            if code is None or code.strip() == "":
                return {
                    "stdout": "",
                    "stderr": "",
                    "result": None,
                    "error": None,
                    "language": language,
                    "state_mode": state_mode,
                    "session_id": session_id,
                    "session_name": session_name,
                    "venv_id": venv_id,
                    "session_created": False,
                    "duration_ms": 0,
                    "surface": surface,
                }

            # ──────────────────────────────────────────────────────────────
            # Boundary wrapper: execute_code (lineage + events + terminal log)
            # ──────────────────────────────────────────────────────────────

            _suffix = _token_hex(2)
            _call_id = new_call_id()
            _parent = TOOL_LOOP_LINEAGE.get([])
            _parent_lineage = list(_parent) if isinstance(_parent, list) else []
            _hierarchy = [*_parent_lineage, f"execute_code({_suffix})"]
            # Establish a boundary lineage frame so nested calls (e.g., FunctionManager-injected
            # functions calling state managers) keep a consistent parent->child chain.
            _lineage_token = TOOL_LOOP_LINEAGE.set(_hierarchy)

            async def _pub_safe(**payload: Any) -> None:
                try:
                    await publish_manager_method_event(
                        _call_id,
                        "CodeActActor",
                        "execute_code",
                        hierarchy=_hierarchy,
                        display_label="Running code",
                        **payload,
                    )
                except Exception as e:
                    log_boundary_event(
                        "->".join(_hierarchy),
                        f"Warning: failed to publish event: {type(e).__name__}: {e}",
                        icon="⚠️",
                        level="warning",
                    )

            try:
                await _pub_safe(phase="incoming")
            except Exception:
                pass
            log_boundary_event("->".join(_hierarchy), "Executing code...", icon="🛠️")

            out: dict[str, Any] | None = None
            tb_str: str | None = None
            exec_exc: Exception | None = None

            active_work = ACTIVE_WORK.begin(
                label="execute_code",
                metadata={
                    "language": language,
                    "state_mode": state_mode,
                    "session_id": session_id,
                    "session_name": session_name,
                    "venv_id": venv_id,
                    "thought": thought[:500],
                },
            )
            heartbeat_task: asyncio.Task[None] | None = None
            try:
                heartbeat_task = asyncio.create_task(
                    self._run_active_work_heartbeat(active_work, _notification_up_q),
                )
                notification_q = (
                    _ActiveWorkNotificationQueue(_notification_up_q, active_work)
                    if _notification_up_q is not None
                    else None
                )
                sandbox_id = None
                try:
                    from unify.manager_registry import ManagerRegistry

                    # Keep generated code's normal environment-based credential
                    # path fresh at the execution boundary.  The SecretManager
                    # gate is debounced, so repeated execute_code calls only pay
                    # a cheap timestamp check within the TTL window.
                    ManagerRegistry.get_secret_manager().sync_assistant_secrets_if_stale(
                        ttl_seconds=60.0,
                        reason="execute_code",
                    )
                except Exception:
                    logger.warning(
                        "execute_code assistant secret sync failed",
                        exc_info=True,
                    )

                # Route non-local surfaces (assistant/user desktop) through the
                # execution targets. Remote surfaces are stateless, so they skip
                # the local session-resolution and pool machinery entirely.
                if surface != "local":
                    out = await self._execute_on_surface(
                        surface_name=surface,
                        code=code,
                        language=str(language),
                        state_mode=state_mode,
                        session_id=session_id,
                        session_name=session_name,
                        venv_id=venv_id,
                        user_id=user_id,
                    )
                    return out

                _rs = self._resolve_session(
                    state_mode=state_mode,
                    language=str(language),
                    session_id=session_id,
                    session_name=session_name,
                    venv_id=venv_id,
                )
                language, venv_id, session_id = (
                    _rs.language,
                    _rs.venv_id,
                    _rs.session_id,
                )
                if _rs.error is not None:
                    out = _rs.error
                    return out

                # Inject per-tool notification queue into bound sandbox so notify() works.
                try:
                    sb_for_notifs = _CURRENT_SANDBOX.get()
                    sandbox_id = getattr(sb_for_notifs, "id", None)
                    if notification_q is not None:
                        sb_for_notifs.global_state["__notification_up_q__"] = (
                            notification_q
                        )
                except Exception:
                    pass

                # Execute via SessionExecutor. Route primitives if available in current sandbox.
                primitives = None
                computer_primitives = self._computer_primitives
                try:
                    sb = _CURRENT_SANDBOX.get()
                    primitives = sb.global_state.get("primitives")
                except Exception:
                    pass

                _pcc_token = _PARENT_CHAT_CONTEXT.set(_parent_chat_context)
                try:
                    try:
                        out = await self._session_executor.execute(
                            code=code,
                            language=str(language),  # type: ignore[arg-type]
                            state_mode=state_mode,  # type: ignore[arg-type]
                            session_id=session_id,
                            venv_id=venv_id,
                            primitives=primitives,
                            computer_primitives=computer_primitives,
                            notification_q=notification_q,
                        )
                    except Exception as e:
                        exec_exc = e
                        tb = traceback.format_exc()
                        tb_str = tb
                        out = {
                            "stdout": "",
                            "stderr": "",
                            "result": None,
                            "error": tb,
                            "language": language,
                            "state_mode": state_mode,
                            "session_id": session_id,
                            "session_name": session_name,
                            "venv_id": venv_id,
                            "session_created": False,
                            "duration_ms": 0,
                        }
                finally:
                    _PARENT_CHAT_CONTEXT.reset(_pcc_token)

                # Enrich with session name.
                if out.get("session_id") is not None:
                    out["session_name"] = self._get_session_name(
                        language=str(out.get("language")),
                        venv_id=out.get("venv_id"),
                        session_id=int(out["session_id"]),
                    )
                else:
                    out["session_name"] = None

                # Wrap in-process Python results in ExecutionResult for proper LLM
                # image formatting. In-process Python has stdout as List[OutputPart];
                # venv/shell have strings.
                if out.get("language") == "python" and isinstance(
                    out.get("stdout"),
                    list,
                ):
                    out = ExecutionResult(**out)

                return out
            finally:
                active_work.end()
                if heartbeat_task is not None and not heartbeat_task.done():
                    heartbeat_task.cancel()
                    try:
                        await heartbeat_task
                    except (asyncio.CancelledError, Exception):
                        pass
                try:
                    _out_err = (
                        (
                            out.get("error")
                            if isinstance(out, dict)
                            else getattr(out, "error", None)
                        )
                        if out is not None
                        else None
                    )
                    if _out_err:
                        await _pub_safe(
                            phase="outgoing",
                            status="error",
                            error=str(_out_err),
                            error_type=(
                                type(exec_exc).__name__
                                if exec_exc is not None
                                else "Error"
                            ),
                            traceback=(tb_str or "")[:2000],
                        )
                    else:
                        await _pub_safe(phase="outgoing", status="ok")
                except Exception:
                    pass
                try:
                    TOOL_LOOP_LINEAGE.reset(_lineage_token)
                except Exception:
                    pass

        # ───────────────────────── Package installation tool ────────────────── #

        async def install_python_packages(
            packages: list[str],
        ) -> dict:
            """Install Python packages into the current execution environment.

            **CRITICAL**: You MUST use this tool whenever you need a Python
            package that is not already available. Do NOT attempt to install
            packages yourself via ``execute_code`` (e.g.
            ``subprocess.run(["pip", "install", ...])``, ``!pip install``,
            ``uv pip install``, or any other shell-based installation).  Direct
            installs bypass the managed overlay and will leave residual
            packages that pollute subsequent sessions.

            Packages are installed into a temporary directory that is
            automatically cleaned up when this task finishes.  They are
            immediately importable in any subsequent ``execute_code`` Python
            call for the remainder of this task, but they do **not** persist
            beyond it.

            Parameters
            ----------
            packages : list[str]
                One or more package specifiers, using the same syntax accepted
                by ``pip install`` / ``uv pip install``.  Examples:

                - ``["pandas"]`` — latest version from PyPI
                - ``["pandas==2.1.0"]`` — exact version pin
                - ``["pandas>=2.0,<3.0"]`` — version range
                - ``["pandas[sql]"]`` — with extras
                - ``["requests", "beautifulsoup4"]`` — multiple packages
                - ``["git+https://github.com/user/repo.git"]`` — from a Git repository
                - ``["./some_wheel.whl"]`` — from a local file

            Returns
            -------
            dict
                - **success** (bool): Whether installation succeeded.
                - **stdout** (str): Installer standard output.
                - **stderr** (str): Installer standard error (contains
                  resolution/download progress and any error messages).
                - **packages** (list[str]): The specifiers that were requested.

            Notes
            -----
            - If a requested package conflicts with a system dependency, the
              system version takes precedence and the import will resolve to
              the pre-installed version.  This is by design — the runtime
              environment must remain stable.
            - If installation fails, inspect ``stderr`` for resolution errors
              and adjust your specifiers (e.g. relax a version constraint).
            """
            try:
                sb = _CURRENT_SANDBOX.get()
                overlay: PackageOverlay | None = sb.global_state.get(
                    "__package_overlay__",
                )
            except Exception:
                overlay = None

            if overlay is None:
                return {
                    "success": False,
                    "stdout": "",
                    "stderr": (
                        "Package installation is not available outside of an "
                        "active act() session."
                    ),
                    "packages": list(packages),
                }

            return overlay.install(packages)

        tools: Dict[str, Callable[..., Awaitable[Any]]] = {
            "execute_code": ToolSpec(fn=execute_code),
            "install_python_packages": ToolSpec(
                fn=install_python_packages,
                display_label="Installing Python packages",
            ),
        }

        # FunctionManager read tools: thin wrappers that inject callables
        # into the sandbox and return only metadata to the LLM. Docstrings
        # are inherited from the base class (the single source of truth).
        if self.function_manager:

            async def FunctionManager_search_functions(
                query: str,
                n: int = 5,
                include_implementations: bool = True,
                _return_callable: bool = False,
                _namespace: Optional[Dict[str, Any]] = None,
                _also_return_metadata: bool = False,
            ) -> Any:
                sb = _CURRENT_SANDBOX.get()
                before = set(sb.global_state.keys())
                result = self.function_manager.search_functions(
                    query=query,
                    n=n,
                    include_implementations=include_implementations,
                    _return_callable=True,
                    _namespace=sb.global_state,
                    _also_return_metadata=True,
                )
                new_keys = set(sb.global_state.keys()) - before
                if new_keys:
                    self._session_executor.register_fm_globals(
                        {k: sb.global_state[k] for k in new_keys},
                    )
                return result["metadata"]

            FunctionManager_search_functions.__doc__ = (
                BaseFunctionManager.search_functions.__doc__
            )

            async def FunctionManager_filter_functions(
                filter: Optional[str] = None,
                offset: int = 0,
                limit: int = 100,
                include_implementations: bool = True,
                _return_callable: bool = False,
                _namespace: Optional[Dict[str, Any]] = None,
                _also_return_metadata: bool = False,
            ) -> Any:
                sb = _CURRENT_SANDBOX.get()
                before = set(sb.global_state.keys())
                result = self.function_manager.filter_functions(
                    filter=filter,
                    offset=offset,
                    limit=limit,
                    include_implementations=include_implementations,
                    _return_callable=True,
                    _namespace=sb.global_state,
                    _also_return_metadata=True,
                )
                new_keys = set(sb.global_state.keys()) - before
                if new_keys:
                    self._session_executor.register_fm_globals(
                        {k: sb.global_state[k] for k in new_keys},
                    )
                return result["metadata"]

            FunctionManager_filter_functions.__doc__ = (
                BaseFunctionManager.filter_functions.__doc__
            )

            async def FunctionManager_list_functions(
                include_implementations: bool = False,
                _return_callable: bool = False,
                _namespace: Optional[Dict[str, Any]] = None,
                _also_return_metadata: bool = False,
            ) -> Any:
                sb = _CURRENT_SANDBOX.get()
                before = set(sb.global_state.keys())
                result = self.function_manager.list_functions(
                    include_implementations=include_implementations,
                    _return_callable=True,
                    _namespace=sb.global_state,
                    _also_return_metadata=True,
                )
                new_keys = set(sb.global_state.keys()) - before
                if new_keys:
                    self._session_executor.register_fm_globals(
                        {k: sb.global_state[k] for k in new_keys},
                    )
                return result["metadata"]

            FunctionManager_list_functions.__doc__ = (
                BaseFunctionManager.list_functions.__doc__
            )

            tools["FunctionManager_search_functions"] = ToolSpec(
                fn=FunctionManager_search_functions,
                display_label="Searching for relevant skills",
            )
            tools["FunctionManager_filter_functions"] = ToolSpec(
                fn=FunctionManager_filter_functions,
                display_label="Filtering saved skills",
            )
            tools["FunctionManager_list_functions"] = ToolSpec(
                fn=FunctionManager_list_functions,
                display_label="Listing existing skills",
            )

            fm = self.function_manager
            tools.update(
                methods_to_tool_dict(
                    ToolSpec(
                        fn=fm.add_functions,
                        display_label="Adding functions to the library",
                    ),
                    ToolSpec(
                        fn=fm.delete_function,
                        display_label="Deleting functions from the library",
                    ),
                    include_class_name=True,
                ),
            )

        # FunctionManager read tools (search/filter/list) use custom wrappers
        # that inject callables into the sandbox. All other FM/GM tools below
        # are plain CRUD with no sandbox side-effects.
        if self.guidance_manager:
            gm = self.guidance_manager
            tools.update(
                methods_to_tool_dict(
                    ToolSpec(
                        fn=gm.search,
                        display_label="Searching for relevant guidance",
                    ),
                    ToolSpec(fn=gm.filter, display_label="Filtering saved guidance"),
                    ToolSpec(
                        fn=gm.get_guidance,
                        display_label="Reading a full guidance entry",
                    ),
                    ToolSpec(fn=gm.add_guidance, display_label="Saving new guidance"),
                    ToolSpec(
                        fn=gm.update_guidance,
                        display_label="Updating saved guidance",
                    ),
                    ToolSpec(
                        fn=gm.delete_guidance,
                        display_label="Deleting saved guidance",
                    ),
                    include_class_name=True,
                ),
            )

        # ── Proactive skill storage tool ──────────────────────────────
        if self.function_manager and self.guidance_manager:
            _actor_ref = self

            async def store_skills(request: str) -> Any:
                """Proactively store reusable skills from the current execution trajectory.

                Triggers a skill-storage review of the trajectory so far. A dedicated
                reviewer will examine the execution history and store any reusable
                functions and compositional guidance based on your request.

                Use this when you have just completed a complex subtask and recognize
                a reusable pattern worth preserving — for example, a non-obvious
                configuration of primitives.actor.act, a multi-step workflow, or a
                function that bakes in hard-won configuration.

                Parameters
                ----------
                request : str
                    Describe the skill(s) you want stored. Be specific about which
                    part of the trajectory contains the reusable pattern and what
                    makes it valuable. For example: "Store the email lookup function
                    that uses primitives.contacts.ask with the scoped discovery_scope"
                    or "Store the multi-step data pipeline that combines file parsing
                    with knowledge update."

                Returns
                -------
                str
                    A summary of what was stored (functions and/or guidance entries),
                    or a note that nothing was worth storing.
                """
                ctx = get_current_agent_context()
                handle = ctx.handle
                if handle is None:
                    return "No active execution context to snapshot."

                _client = getattr(handle, "_client", None)
                _trajectory = (
                    make_messages_safe_for_context_dump(
                        list(getattr(_client, "messages", []) or []),
                    )
                    if _client
                    else []
                )

                _task = getattr(handle, "_task", None)
                _ask_tools = (
                    _task.get_ask_tools()
                    if _task and hasattr(_task, "get_ask_tools")
                    else {}
                )
                _completed_meta = (
                    _task.get_completed_tool_metadata()
                    if _task and hasattr(_task, "get_completed_tool_metadata")
                    else {}
                )

                _ps_call_id = new_call_id()
                _ps_parent = TOOL_LOOP_LINEAGE.get([])
                _ps_parent_lineage = (
                    list(_ps_parent) if isinstance(_ps_parent, list) else []
                )
                _ps_suffix = _token_hex(2)
                _ps_hierarchy = [
                    *_ps_parent_lineage,
                    f"ProactiveStorage(CodeActActor.act)({_ps_suffix})",
                ]

                await publish_manager_method_event(
                    _ps_call_id,
                    "CodeActActor",
                    "ProactiveStorage",
                    phase="incoming",
                    display_label="Proactive skill storage",
                    hierarchy=_ps_hierarchy,
                    instructions=request,
                )

                storage_handle = _start_proactive_storage_loop(
                    trajectory=_trajectory,
                    ask_tools=_ask_tools,
                    completed_tool_metadata=_completed_meta,
                    actor=_actor_ref,
                    request=request,
                    parent_lineage=_ps_parent_lineage,
                )

                if storage_handle is None:
                    await publish_manager_method_event(
                        _ps_call_id,
                        "CodeActActor",
                        "ProactiveStorage",
                        phase="outgoing",
                        display_label="Proactive skill storage",
                        hierarchy=_ps_hierarchy,
                    )
                    return (
                        "Skill storage unavailable "
                        "(FunctionManager or GuidanceManager missing)."
                    )

                _orig_result_fn = storage_handle.result

                async def _tracking_result():
                    try:
                        result = await _orig_result_fn()
                        ctx.proactive_storage_summaries.append(result)
                        return result
                    finally:
                        await publish_manager_method_event(
                            _ps_call_id,
                            "CodeActActor",
                            "ProactiveStorage",
                            phase="outgoing",
                            display_label="Proactive skill storage",
                            hierarchy=_ps_hierarchy,
                        )

                storage_handle.result = _tracking_result  # type: ignore[assignment]

                return storage_handle

            tools["store_skills"] = store_skills

        if self.function_manager:

            async def execute_function(
                function_name: str,
                call_kwargs: Optional[Dict[str, Any]] = None,
                *,
                language: str = "python",
                state_mode: str = "stateless",
                session_id: int | None = None,
                session_name: str | None = None,
                _notification_up_q: asyncio.Queue[dict] | None = None,
                _parent_chat_context: list[dict] | None = None,
            ) -> Any:
                """
                Execute a single function or primitive by name.

                **This is the preferred tool for any task that maps to a single
                function or primitive call.** Use it instead of ``execute_code``
                whenever the task can be accomplished by invoking one callable
                with keyword arguments — no surrounding Python logic needed.

                Why prefer this tool
                --------------------
                ``execute_function`` **structurally guarantees** that the
                returned handle is exposed to the outer loop for steering
                (ask, stop, pause, resume, interject). When you write the
                same call inside ``execute_code``, the handle is only
                adopted if it happens to be the last expression — a pattern
                that is easy to break by adding prints, notifications, or
                error handling around the call.

                When to use ``execute_function`` vs ``execute_code``
                ----------------------------------------------------
                - **Single primitive call** (e.g. ``primitives.contacts.ask``,
                  ``primitives.web.ask``, ``primitives.knowledge.update``)
                  → always ``execute_function``.
                - **Single stored function call** (discovered via
                  FunctionManager) → always ``execute_function``.
                - **Multi-step composition**, conditional logic, loops,
                  or any code that genuinely needs to combine multiple
                  calls or process intermediate results
                  → use ``execute_code``.

                Resolution order
                ----------------
                1. The current sandbox namespace (environment-injected callables,
                   previously discovered FunctionManager functions, etc.).
                2. The FunctionManager store (by exact name lookup).
                3. If neither matches, a ``NameError`` is raised naturally.

                Key concepts
                ------------
                - **language**: ``"python"`` | ``"bash"`` | ``"zsh"`` | ``"sh"`` | ``"powershell"``
                - **state_mode**:
                  - ``"stateless"``: no session; clean execution; no persistence
                  - ``"stateful"``: persistent session; state accumulates
                  - ``"read_only"``: reads from an existing session but does not
                    persist changes
                - **session_id / session_name**: only meaningful for
                  stateful / read_only (same semantics as ``execute_code``)

                Parameters
                ----------
                function_name : str
                    Exact name of the function or primitive to execute.
                    For primitives, use the dotted path as it appears in the
                    sandbox (e.g. ``"primitives.contacts.ask"``).
                call_kwargs : dict, optional
                    Keyword arguments to pass to the function.

                Returns
                -------
                dict | ExecutionResult
                    Same shape as ``execute_code`` output (stdout, stderr, result,
                    error, language, state_mode, session_id, session_name,
                    session_created, duration_ms).
                """
                call_kwargs = call_kwargs or {}
                resolved_venv_id: int | None = None
                function_data: dict[str, Any] | None = None
                get_function_data = getattr(
                    self.function_manager,
                    "_get_function_data_by_name",
                    None,
                )
                if callable(get_function_data):
                    function_data = get_function_data(name=function_name)
                if function_data is None:
                    get_stored_primitive = getattr(
                        self.function_manager,
                        "_get_stored_primitive_data_by_name",
                        None,
                    )
                    if callable(get_stored_primitive):
                        function_data = get_stored_primitive(name=function_name)
                stored_venv_id = (
                    function_data.get("venv_id")
                    if isinstance(function_data, dict)
                    and not function_data.get("is_primitive")
                    else None
                )
                if stored_venv_id is not None:
                    resolved_venv_id = int(stored_venv_id)

                import time as _ef_time
                import logging as _ef_logging

                _ef_t0 = _ef_time.perf_counter()
                _ef_log = _ef_logging.getLogger("unify")

                def _ef_ms():
                    return f"{(_ef_time.perf_counter() - _ef_t0) * 1000:.0f}ms"

                _ef_log.debug(
                    f"⏱️ [execute_function +{_ef_ms()}] entered: {function_name}",
                )

                # ── Synthesize the code string ────────────────────────────
                code: str | None = None

                if str(language) == "python":
                    code = _synthesize_python_call(
                        function_name=function_name,
                        call_kwargs=call_kwargs,
                        function_manager=self.function_manager,
                    )
                else:
                    # Shell: look up the stored implementation and append it
                    # with kwargs serialised as environment variables.
                    code = _synthesize_shell_call(
                        function_name=function_name,
                        call_kwargs=call_kwargs,
                        function_manager=self.function_manager,
                    )
                _ef_log.debug(
                    f"⏱️ [execute_function +{_ef_ms()}] code synthesized",
                )

                # ── Lineage boundary ─────────────────────────────────────
                _ef_suffix = _token_hex(2)
                _ef_call_id = new_call_id()
                _ef_parent = TOOL_LOOP_LINEAGE.get([])
                _ef_parent_lineage = (
                    list(_ef_parent) if isinstance(_ef_parent, list) else []
                )
                _ef_hierarchy = [
                    *_ef_parent_lineage,
                    f"execute_function({function_name})({_ef_suffix})",
                ]
                _ef_lineage_token = TOOL_LOOP_LINEAGE.set(_ef_hierarchy)

                async def _ef_pub_safe(**payload: Any) -> None:
                    try:
                        await publish_manager_method_event(
                            _ef_call_id,
                            "CodeActActor",
                            "execute_function",
                            hierarchy=_ef_hierarchy,
                            display_label=f"Running: {function_name}",
                            **payload,
                        )
                    except Exception as e:
                        log_boundary_event(
                            "->".join(_ef_hierarchy),
                            f"Warning: failed to publish event: {type(e).__name__}: {e}",
                            icon="⚠️",
                            level="warning",
                        )

                _ef_log.debug(
                    f"⏱️ [execute_function +{_ef_ms()}] lineage boundary (incoming) start",
                )
                try:
                    await _ef_pub_safe(phase="incoming")
                except Exception:
                    pass
                _ef_log.debug(
                    f"⏱️ [execute_function +{_ef_ms()}] lineage boundary (incoming) done",
                )
                log_boundary_event(
                    "->".join(_ef_hierarchy),
                    f"Executing function {function_name}...",
                    icon="🛠️",
                )

                # ── Session resolution + execution (shared with execute_code) ──
                out: dict[str, Any] | None = None
                tb_str: str | None = None
                exec_exc: Exception | None = None

                active_work = ACTIVE_WORK.begin(
                    label="execute_function",
                    metadata={
                        "function_name": function_name,
                        "language": language,
                        "state_mode": state_mode,
                        "session_id": session_id,
                        "session_name": session_name,
                        "venv_id": resolved_venv_id,
                    },
                )
                heartbeat_task: asyncio.Task[None] | None = None
                try:
                    heartbeat_task = asyncio.create_task(
                        self._run_active_work_heartbeat(
                            active_work,
                            _notification_up_q,
                        ),
                    )
                    notification_q = (
                        _ActiveWorkNotificationQueue(_notification_up_q, active_work)
                        if _notification_up_q is not None
                        else None
                    )
                    sandbox_id = None
                    _rs = self._resolve_session(
                        state_mode=state_mode,
                        language=str(language),
                        session_id=session_id,
                        session_name=session_name,
                        venv_id=resolved_venv_id,
                    )
                    language, resolved_venv_id, session_id = (
                        _rs.language,
                        _rs.venv_id,
                        _rs.session_id,
                    )
                    if _rs.error is not None:
                        out = _rs.error
                        return out

                    # Inject per-tool notification queue into bound sandbox.
                    try:
                        sb_for_notifs = _CURRENT_SANDBOX.get()
                        sandbox_id = getattr(sb_for_notifs, "id", None)
                        if notification_q is not None:
                            sb_for_notifs.global_state["__notification_up_q__"] = (
                                notification_q
                            )
                    except Exception:
                        pass

                    # Resolve primitives from current sandbox.
                    primitives = None
                    computer_primitives = self._computer_primitives
                    try:
                        sb = _CURRENT_SANDBOX.get()
                        primitives = sb.global_state.get("primitives")
                    except Exception:
                        pass

                    if (
                        isinstance(function_data, dict)
                        and function_data.get("is_primitive")
                        and is_provider_backed_function(function_data)
                    ):
                        _ef_log.debug(
                            f"⏱️ [execute_function +{_ef_ms()}] "
                            "provider primitive direct execute start",
                        )
                        try:
                            direct_result = (
                                await self.function_manager.execute_function(
                                    function_name=function_name,
                                    call_kwargs=call_kwargs,
                                    target_venv_id=None,
                                    state_mode=state_mode,  # type: ignore[arg-type]
                                    session_id=session_id or 0,
                                    extra_namespaces=(
                                        {"primitives": primitives}
                                        if primitives is not None
                                        else None
                                    ),
                                    _parent_chat_context=_parent_chat_context,
                                )
                            )
                            if (
                                isinstance(direct_result, dict)
                                and direct_result.get("status")
                                == "confirmation_required"
                            ):
                                direct_result = build_pending_approval_payload(
                                    function_name=function_name,
                                    function_data=function_data,
                                    call_kwargs=call_kwargs,
                                    provider_envelope=direct_result,
                                )
                                if notification_q is not None:
                                    await notification_q.put(direct_result)
                            out = {
                                "stdout": [],
                                "stderr": [],
                                "result": direct_result,
                                "error": None,
                                "language": "python",
                                "state_mode": state_mode,
                                "session_id": session_id,
                                "session_name": session_name,
                                "venv_id": resolved_venv_id,
                                "session_created": False,
                                "duration_ms": int(
                                    (_ef_time.perf_counter() - _ef_t0) * 1000,
                                ),
                            }
                            _ef_log.debug(
                                f"⏱️ [execute_function +{_ef_ms()}] "
                                "provider primitive direct execute done",
                            )
                        except Exception:
                            exec_exc = sys.exc_info()[1]
                            tb = traceback.format_exc()
                            tb_str = tb
                            out = {
                                "stdout": [],
                                "stderr": [],
                                "result": None,
                                "error": tb,
                                "language": "python",
                                "state_mode": state_mode,
                                "session_id": session_id,
                                "session_name": session_name,
                                "venv_id": resolved_venv_id,
                                "session_created": False,
                                "duration_ms": int(
                                    (_ef_time.perf_counter() - _ef_t0) * 1000,
                                ),
                            }
                    else:
                        _ef_log.debug(
                            f"⏱️ [execute_function +{_ef_ms()}] sandbox.execute start",
                        )
                        _pcc_token = _PARENT_CHAT_CONTEXT.set(_parent_chat_context)
                        try:
                            try:
                                out = await self._session_executor.execute(
                                    code=code,
                                    language=str(language),  # type: ignore[arg-type]
                                    state_mode=state_mode,  # type: ignore[arg-type]
                                    session_id=session_id,
                                    venv_id=resolved_venv_id,
                                    primitives=primitives,
                                    computer_primitives=computer_primitives,
                                    notification_q=notification_q,
                                )
                                _ef_log.debug(
                                    f"⏱️ [execute_function +{_ef_ms()}] sandbox.execute done",
                                )
                            except Exception as e:
                                exec_exc = e
                                tb = traceback.format_exc()
                                tb_str = tb
                                out = {
                                    "stdout": "",
                                    "stderr": "",
                                    "result": None,
                                    "error": tb,
                                    "language": language,
                                    "state_mode": state_mode,
                                    "session_id": session_id,
                                    "session_name": session_name,
                                    "venv_id": resolved_venv_id,
                                    "session_created": False,
                                    "duration_ms": 0,
                                }
                        finally:
                            _PARENT_CHAT_CONTEXT.reset(_pcc_token)

                    # Enrich with session name.
                    if out.get("session_id") is not None:
                        out["session_name"] = self._get_session_name(
                            language=str(out.get("language")),
                            venv_id=out.get("venv_id"),
                            session_id=int(out["session_id"]),
                        )
                    else:
                        out["session_name"] = None

                    # Wrap in-process Python results in ExecutionResult.
                    if out.get("language") == "python" and isinstance(
                        out.get("stdout"),
                        list,
                    ):
                        out = ExecutionResult(**out)

                    # When the execution produced a bare SteerableToolHandle
                    # with no meaningful side output, return the handle directly
                    # so the core loop adopts it via the bare-handle path
                    # (no intermediate LLM turn required).
                    _ef_result_val = (
                        out.get("result")
                        if isinstance(out, dict)
                        else getattr(out, "result", None)
                    )
                    if isinstance(_ef_result_val, SteerableToolHandle):
                        _ef_stdout = (
                            out.get("stdout")
                            if isinstance(out, dict)
                            else getattr(out, "stdout", None)
                        )
                        _ef_stderr = (
                            out.get("stderr")
                            if isinstance(out, dict)
                            else getattr(out, "stderr", None)
                        )
                        _ef_error = (
                            out.get("error")
                            if isinstance(out, dict)
                            else getattr(out, "error", None)
                        )
                        _has_side_output = bool(
                            (
                                _ef_stdout
                                and (
                                    isinstance(_ef_stdout, str)
                                    and _ef_stdout.strip()
                                    or isinstance(_ef_stdout, list)
                                    and _ef_stdout
                                )
                            )
                            or (
                                _ef_stderr
                                and (
                                    isinstance(_ef_stderr, str)
                                    and _ef_stderr.strip()
                                    or isinstance(_ef_stderr, list)
                                    and _ef_stderr
                                )
                            )
                            or _ef_error,
                        )
                        if not _has_side_output:
                            _ef_log.debug(
                                f"⏱️ [execute_function +{_ef_ms()}] "
                                f"returning bare handle (no side output)",
                            )
                            return _ef_result_val

                    _ef_log.debug(
                        f"⏱️ [execute_function +{_ef_ms()}] returning result",
                    )
                    return out
                finally:
                    active_work.end()
                    if heartbeat_task is not None and not heartbeat_task.done():
                        heartbeat_task.cancel()
                        try:
                            await heartbeat_task
                        except (asyncio.CancelledError, Exception):
                            pass
                    _ef_log.debug(
                        f"⏱️ [execute_function +{_ef_ms()}] lineage boundary (outgoing) start",
                    )
                    try:
                        _out_err = (
                            (
                                out.get("error")
                                if isinstance(out, dict)
                                else getattr(out, "error", None)
                            )
                            if out is not None
                            else None
                        )
                        if _out_err:
                            await _ef_pub_safe(
                                phase="outgoing",
                                status="error",
                                error=str(_out_err),
                                error_type=(
                                    type(exec_exc).__name__
                                    if exec_exc is not None
                                    else "Error"
                                ),
                                traceback=(tb_str or "")[:2000],
                            )
                        else:
                            await _ef_pub_safe(phase="outgoing", status="ok")
                    except Exception:
                        pass
                    _ef_log.debug(
                        f"⏱️ [execute_function +{_ef_ms()}] lineage boundary (outgoing) done",
                    )
                    try:
                        TOOL_LOOP_LINEAGE.reset(_ef_lineage_token)
                    except Exception:
                        pass

            def _ef_display_label(tc: dict) -> str:
                try:
                    args = json.loads(tc.get("function", {}).get("arguments", "{}"))
                    return args.get("function_name", "execute_function")
                except Exception:
                    return "execute_function"

            tools["execute_function"] = ToolSpec(
                fn=execute_function,
                display_label=_ef_display_label,
            )

        # ───────────────────────── Session management tools ────────────────── #

        async def list_sessions(detail: str = "summary") -> Dict[str, Any]:
            """
            List all active sessions across all languages (Python + shell).

            Use this tool whenever you need to choose which session to use for a
            subsequent `execute_code(..., state_mode="stateful"/"read_only")` call.

            Parameters
            ----------
            detail:
                Controls how much information is returned per session:
                - "summary": metadata + a short `state_summary` string (default)
                - "full": best-effort enrichment using cheap inspection where available

            Returns
            -------
            dict:
                {"sessions": [ ... ]} where each entry includes (best-effort):
                - language: "python" | "bash" | "zsh" | "sh" | "powershell"
                - session_id: int (scoped per language + venv_id)
                - venv_id: int | None (Python only)
                - session_name: optional human-friendly alias (if registered)
                - created_at / last_used: timestamps when available
                - state_summary: a short human-readable summary (e.g. "3 names", "cwd=/repo")

            Notes
            -----
            - Session IDs are **scoped per (language, venv_id)**, so `python` session 0 and
              `bash` session 0 can coexist.
            - The default per-call Python sandbox is exposed as `python` session_id=0 (venv_id=None)
              when it is bound for the current call.
            """
            detail = (detail or "summary").strip()

            sessions: list[dict[str, Any]] = []

            # Default sandbox (current act sandbox) as python session 0 (venv_id=None).
            try:
                sb = _CURRENT_SANDBOX.get()
                sessions.append(
                    {
                        "language": "python",
                        "session_id": 0,
                        "venv_id": None,
                        "session_name": self._get_session_name(
                            language="python",
                            venv_id=None,
                            session_id=0,
                        ),
                        "created_at": None,
                        "last_used": None,
                        "state_summary": f"{len(sb.global_state)} globals",
                    },
                )
            except Exception:
                pass

            # In-process python sessions created via SessionExecutor.
            for s in self._session_executor.list_in_process_python_sessions():
                s = dict(s)
                s["session_name"] = self._get_session_name(
                    language="python",
                    venv_id=s.get("venv_id"),
                    session_id=int(s["session_id"]),
                )
                sessions.append(s)

            # Venv sessions.
            try:
                for s in self._venv_pool.get_all_sessions():
                    s = dict(s)
                    s["session_name"] = self._get_session_name(
                        language="python",
                        venv_id=s.get("venv_id"),
                        session_id=int(s["session_id"]),
                    )
                    sessions.append(s)
            except Exception:
                pass

            # Shell sessions.
            try:
                for s in self._shell_pool.get_all_sessions():
                    s = dict(s)
                    s["session_name"] = self._get_session_name(
                        language=str(s.get("language")),
                        venv_id=None,
                        session_id=int(s["session_id"]),
                    )
                    sessions.append(s)
            except Exception:
                pass

            if detail == "full":
                # Best-effort enrich state_summary with inspection where cheap.
                for s in sessions:
                    try:
                        if (
                            s.get("language") == "python"
                            and s.get("venv_id") is not None
                        ):
                            st = await self._venv_pool.get_session_state(
                                venv_id=int(s["venv_id"]),
                                session_id=int(s["session_id"]),
                                function_manager=self.function_manager,
                                detail="summary",
                            )
                            if isinstance(st, dict) and "count" in st:
                                s["state_summary"] = f'{st["count"]} names'
                        elif s.get("language") in ("bash", "zsh", "sh", "powershell"):
                            st = await self._shell_pool.get_session_state(
                                language=s["language"],
                                session_id=int(s["session_id"]),
                                detail="summary",
                            )
                            if isinstance(st, dict) and "summary" in st:
                                s["state_summary"] = st["summary"]
                    except Exception:
                        continue

            return {"sessions": sessions}

        async def inspect_state(
            session_name: str | None = None,
            session_id: int | None = None,
            language: str | None = None,
            venv_id: int | None = None,
            detail: str = "summary",
        ) -> Dict[str, Any]:
            """
            Inspect the state of a specific session (Python or shell).

            This tool is for debugging and for deciding whether to:
            - continue in the same session (stateful)
            - start a fresh session (stateful without session_id/session_name)
            - run a one-off (stateless)
            - do a what-if (read_only)

            Parameters
            ----------
            session_name:
                Optional human-friendly alias for a session (preferred when available).
            session_id + language (+ optional venv_id):
                Directly identify a session. `session_id` is scoped per (language, venv_id).
            detail:
                "summary" | "names" | "full"
                - Prefer "summary" when you just need quick context.
                - Use "names" to see what variables exist without dumping values.
                - Use "full" sparingly (can be large; values are truncated/redacted best-effort).

            Defaults
            --------
            If no session selector is provided, this inspects the **current per-call Python sandbox**
            (python session_id=0, venv_id=None) when bound.

            Returns
            -------
            dict with:
            - session: {language, session_id, session_name, venv_id}
            - state: implementation-specific state representation (Python vars; shell cwd/env/functions/aliases)
            """
            detail = (detail or "summary").strip()

            # Resolve session.
            resolved: SessionKey | None = None
            if session_name:
                resolved = self._resolve_session_name(session_name)
                if resolved is None:
                    return {
                        "error": f"Session {session_name!r} not found",
                        "error_type": "validation",
                    }
            elif session_id is not None and language is not None:
                resolved = (str(language), venv_id, int(session_id))

            # Default: current sandbox.
            if resolved is None:
                try:
                    sb = _CURRENT_SANDBOX.get()
                except Exception as e:
                    return {
                        "error": f"No sandbox bound: {type(e).__name__}",
                        "error_type": "internal",
                    }

                names: list[str] = []
                full_map: dict[str, str] = {}
                for k, v in sb.global_state.items():
                    if not isinstance(k, str) or k.startswith("_"):
                        continue
                    if callable(v) or isinstance(v, type):
                        continue
                    names.append(k)
                    if detail == "full":
                        try:
                            s = repr(v)
                            if len(s) > 500:
                                s = s[:500] + "..."
                        except Exception:
                            s = f"<{type(v).__name__}>"
                        full_map[k] = s

                names = sorted(names)
                state_obj: dict[str, Any]
                if detail == "full":
                    state_obj = {"variables": full_map, "functions": []}
                else:
                    state_obj = {"variables": names, "functions": []}

                return {
                    "session": {
                        "language": "python",
                        "session_id": 0,
                        "session_name": self._get_session_name(
                            language="python",
                            venv_id=None,
                            session_id=0,
                        ),
                        "venv_id": None,
                    },
                    "state": state_obj,
                }

            lang, resolved_venv_id, sid = resolved

            # Python venv-backed
            if lang == "python" and resolved_venv_id is not None:
                st = await self._venv_pool.get_session_state(
                    venv_id=int(resolved_venv_id),
                    session_id=int(sid),
                    function_manager=self.function_manager,
                    detail=detail,
                )
                return {
                    "session": {
                        "language": "python",
                        "session_id": int(sid),
                        "session_name": self._get_session_name(
                            language="python",
                            venv_id=int(resolved_venv_id),
                            session_id=int(sid),
                        ),
                        "venv_id": int(resolved_venv_id),
                    },
                    "state": st,
                }

            # Python in-process session (SessionExecutor)
            if lang == "python" and resolved_venv_id is None:
                key = (None, int(sid))
                sb = self._session_executor._python_sessions.get(
                    key,
                )  # pylint: disable=protected-access
                if sb is None:
                    return {
                        "error": f"Python session {sid} not found",
                        "error_type": "validation",
                    }
                names: list[str] = []
                full_map: dict[str, str] = {}
                for k, v in sb.global_state.items():
                    if not isinstance(k, str) or k.startswith("_"):
                        continue
                    if callable(v) or isinstance(v, type):
                        continue
                    names.append(k)
                    if detail == "full":
                        try:
                            s = repr(v)
                            if len(s) > 500:
                                s = s[:500] + "..."
                        except Exception:
                            s = f"<{type(v).__name__}>"
                        full_map[k] = s
                names = sorted(names)
                state_obj = {
                    "variables": full_map if detail == "full" else names,
                    "functions": [],
                }
                return {
                    "session": {
                        "language": "python",
                        "session_id": int(sid),
                        "session_name": self._get_session_name(
                            language="python",
                            venv_id=None,
                            session_id=int(sid),
                        ),
                        "venv_id": None,
                    },
                    "state": state_obj,
                }

            # Shell
            st = await self._shell_pool.get_session_state(
                language=lang,  # type: ignore[arg-type]
                session_id=int(sid),
                detail=detail,
            )
            return {
                "session": {
                    "language": str(lang),
                    "session_id": int(sid),
                    "session_name": self._get_session_name(
                        language=str(lang),
                        venv_id=None,
                        session_id=int(sid),
                    ),
                    "venv_id": None,
                },
                "state": st,
            }

        async def close_session(
            session_name: str | None = None,
            session_id: int | None = None,
            language: str | None = None,
            venv_id: int | None = None,
        ) -> Dict[str, Any]:
            """
            Close a specific session and free resources.

            Use this to proactively manage resources when you are done with a session.
            This operation is **idempotent**: closing an already-closed/non-existent session
            returns `closed=False, reason="not_found"` rather than raising.

            Parameters
            ----------
            session_name:
                Preferred: close by human-friendly alias.
            session_id + language (+ optional venv_id):
                Close by canonical identity.

            Returns
            -------
            dict:
                - closed: bool
                - reason: "success" | "not_found" | "error"
                - session: {language, session_id, session_name}
            """
            resolved: SessionKey | None = None
            if session_name:
                resolved = self._resolve_session_name(session_name)
                if resolved is None:
                    return {
                        "closed": False,
                        "reason": "not_found",
                        "session": {
                            "language": language,
                            "session_id": session_id,
                            "session_name": session_name,
                        },
                    }
            elif session_id is not None and language is not None:
                resolved = (str(language), venv_id, int(session_id))
            else:
                return {
                    "closed": False,
                    "reason": "error",
                    "error": "Must provide session_name or (language + session_id).",
                }

            lang, resolved_venv_id, sid = resolved
            closed = False

            if lang == "python" and resolved_venv_id is not None:
                closed = await self._venv_pool.close_session(
                    venv_id=int(resolved_venv_id),
                    session_id=int(sid),
                )
            elif lang == "python" and resolved_venv_id is None:
                closed = await self._session_executor.close_in_process_python_session(
                    session_id=int(sid),
                    venv_id=None,
                )
            else:
                closed = await self._shell_pool.close_session(language=lang, session_id=int(sid))  # type: ignore[arg-type]

            # Unregister all aliases for this session.
            self._unregister_all_names_for_session(
                key=(str(lang), resolved_venv_id, int(sid)),
            )

            return {
                "closed": bool(closed),
                "reason": "success" if closed else "not_found",
                "session": {
                    "language": str(lang),
                    "session_id": int(sid),
                    "session_name": session_name
                    or self._get_session_name(
                        language=str(lang),
                        venv_id=resolved_venv_id,
                        session_id=int(sid),
                    ),
                },
            }

        async def close_all_sessions() -> Dict[str, Any]:
            """
            Close all active sessions across all languages.

            This is a blunt cleanup tool. Prefer `close_session(...)` when you only
            want to discard a specific polluted/unused session.

            Returns
            -------
            dict:
                - closed_count: int
                - languages: list[str] (languages that had sessions closed)
                - details: per-language counts
            """
            closed_counts: dict[str, int] = {
                "python": 0,
                "bash": 0,
                "zsh": 0,
                "sh": 0,
                "powershell": 0,
            }

            # Close in-process python sessions.
            for s in list(self._session_executor.list_in_process_python_sessions()):
                sid = int(s.get("session_id", 0))
                if await self._session_executor.close_in_process_python_session(
                    session_id=sid,
                    venv_id=None,
                ):
                    closed_counts["python"] += 1
                    self._unregister_all_names_for_session(key=("python", None, sid))

            # Close venv python sessions.
            for vid, sid in list(self._venv_pool.list_active_sessions()):
                if await self._venv_pool.close_session(
                    venv_id=int(vid),
                    session_id=int(sid),
                ):
                    closed_counts["python"] += 1
                    self._unregister_all_names_for_session(
                        key=("python", int(vid), int(sid)),
                    )

            # Close shell sessions.
            for lang, sid in list(self._shell_pool.get_active_sessions()):
                if await self._shell_pool.close_session(
                    language=lang,
                    session_id=int(sid),
                ):
                    closed_counts[str(lang)] = closed_counts.get(str(lang), 0) + 1
                    self._unregister_all_names_for_session(
                        key=(str(lang), None, int(sid)),
                    )

            # Clear any remaining aliases.
            self._session_names.clear()
            self._session_names_rev.clear()

            closed_total = sum(closed_counts.values())
            langs = [k for k, v in closed_counts.items() if v > 0]
            return {
                "closed_count": closed_total,
                "languages": langs,
                "details": closed_counts,
            }

        tools["list_sessions"] = ToolSpec(
            fn=list_sessions,
            display_label="Listing active sessions",
        )
        tools["inspect_state"] = ToolSpec(
            fn=inspect_state,
            display_label="Inspecting session state",
        )
        tools["close_session"] = ToolSpec(
            fn=close_session,
            display_label="Closing a session",
        )
        tools["close_all_sessions"] = ToolSpec(
            fn=close_all_sessions,
            display_label="Closing all sessions",
        )

        # ───────────────────── Package installation tool ───────────────── #

        async def install_python_packages(
            packages: list[str],
        ) -> Dict[str, Any]:
            """
            Install Python packages into the current execution environment.

            **You MUST use this tool whenever you need a Python package that is
            not already available.**  Do NOT attempt to install packages via
            ``execute_code`` (e.g. ``!pip install ...``, ``subprocess.run(["pip",
            ...])``), ``uv pip install``, or any other shell-based method.
            Doing so bypasses the managed overlay and will leave the
            environment in an inconsistent state.

            Accepts the full range of pip/uv package specifiers:

            - ``"pandas"`` (latest version)
            - ``"pandas==2.1.0"`` (exact version)
            - ``"pandas>=2.0,<3.0"`` (version range)
            - ``"pandas[sql]"`` (extras)
            - ``"git+https://github.com/user/repo.git"`` (VCS)
            - ``"git+https://github.com/user/repo.git@branch"`` (VCS + ref)
            - ``"./path/to/wheel.whl"`` (local file)

            Installed packages become immediately importable in subsequent
            ``execute_code`` Python calls within the same trajectory.

            **Automatic cleanup**: all packages installed via this tool are
            automatically removed when the current act() trajectory completes.
            They do NOT persist across trajectories.  If a future trajectory
            needs the same package, it must install it again.

            Parameters
            ----------
            packages : list[str]
                One or more package specifiers (see examples above).

            Returns
            -------
            dict
                - **success** (bool): True if installation succeeded.
                - **stdout** (str): Installer standard output.
                - **stderr** (str): Installer standard error (includes
                  resolution details and any warnings).
                - **packages** (list[str]): The specifiers that were requested.
            """
            overlay = _CURRENT_PACKAGE_OVERLAY.get()
            if overlay is None:
                return {
                    "success": False,
                    "stdout": "",
                    "stderr": "No package overlay is bound for this trajectory.",
                    "packages": packages,
                }
            return overlay.install(packages)

        tools["install_python_packages"] = ToolSpec(
            fn=install_python_packages,
            display_label="Installing Python packages",
        )

        return tools

    async def _repair_symbolic_entrypoint(
        self,
        *,
        entrypoint_id: int,
        request: str | dict | list[str | dict],
        entrypoint_kwargs: dict[str, Any],
        failure: BaseException,
        repair_context: dict[str, Any] | None,
    ) -> str:
        """Run a bounded review loop that can repair a failing symbolic executor."""

        fm = self.function_manager
        if fm is None:
            raise RuntimeError(
                "Cannot repair symbolic entrypoint without FunctionManager.",
            )

        function_snapshot = fm.filter_functions(
            filter=f"function_id == {int(entrypoint_id)}",
            _also_return_metadata=True,
        )
        tools = methods_to_tool_dict(
            fm.search_functions,
            fm.filter_functions,
            fm.list_functions,
            fm.add_functions,
            fm.delete_function,
            fm.add_venv,
            fm.list_venvs,
            fm.get_venv,
            fm.update_venv,
            fm.delete_venv,
            fm.set_function_venv,
            fm.get_function_venv,
            include_class_name=True,
        )
        client = new_llm_client(self._model)
        client.set_system_message(
            "You are repairing a stored symbolic task executor. The function "
            "must preserve the task contract, managed primitives, deterministic "
            "inputs, side-effect ordering, and failure semantics. Prefer updating "
            "the existing function with overwrite=True. Do not replace managed "
            "primitives with ad hoc weaker implementations.",
        )
        message = (
            "A symbolic task executor failed certification or execution.\n\n"
            f"Task request:\n{request}\n\n"
            "Deterministic entrypoint kwargs:\n"
            f"```json\n{json.dumps(entrypoint_kwargs, indent=2, default=str)}\n```\n\n"
            "Function snapshot:\n"
            f"```json\n{json.dumps(function_snapshot, indent=2, default=str)}\n```\n\n"
            "Repair context:\n"
            f"```json\n{json.dumps(repair_context or {}, indent=2, default=str)}\n```\n\n"
            f"Failure: {type(failure).__name__}: {failure}\n\n"
            "Repair the stored function if possible, then briefly summarize the "
            "equivalence rationale and the change made. If it cannot be repaired "
            "without changing the task contract, say so without promoting it."
        )
        handle = start_async_tool_loop(
            client=client,
            message=message,
            tools=tools,
            loop_id=f"SymbolicEntrypointRepair({entrypoint_id})",
            max_consecutive_failures=2,
        )
        result = await handle.result()
        return str(result)

    @functools.wraps(BaseCodeActActor.act, updated=())
    @log_manager_call(
        "CodeActActor",
        "act",
        payload_key="request",
        display_label=lambda kw: "Session" if kw.get("persist") else "Taking action",
        forward_kwargs=("persist",),
    )
    async def act(
        self,
        request: str | dict | list[str | dict],
        *,
        guidelines: Optional[str] = None,
        clarification_enabled: bool = True,
        response_format: Optional[Type[BaseModel]] = None,
        _parent_chat_context: list[dict] | None = None,
        _clarification_up_q: Optional[asyncio.Queue[str]] = None,
        _clarification_down_q: Optional[asyncio.Queue[str]] = None,
        _call_id: Optional[str] = None,
        _reuse_actor_slot: bool = False,
        entrypoint: Optional[int] = None,
        entrypoint_args: Optional[list[Any]] = None,
        entrypoint_kwargs: Optional[dict[str, Any]] = None,
        persist: Optional[bool] = None,
        can_compose: Optional[bool] = None,
        can_store: Optional[bool] = None,
        llm_profile: Optional[str] = None,
        **kwargs,
    ) -> SteerableToolHandle:
        if not self._main_event_loop:
            self._main_event_loop = asyncio.get_running_loop()

        import time as _act_time

        _act_t0 = _act_time.perf_counter()

        def _act_ms() -> str:
            return f"{(_act_time.perf_counter() - _act_t0) * 1000:.0f}ms"

        logger.debug(f"⏱️ [CodeActActor.act +{_act_ms()}] entered")

        entrypoint_repair_attempts = int(
            kwargs.pop("entrypoint_repair_attempts", 0) or 0,
        )
        entrypoint_repair_context = kwargs.pop("entrypoint_repair_context", None)

        effective_can_compose = (
            self.can_compose if can_compose is None else bool(can_compose)
        )
        effective_can_store = self.can_store if can_store is None else bool(can_store)
        act_llm_profile = resolve_act_llm_profile(llm_profile)

        # can_compose=False requires a FunctionManager so the LLM has execute_function
        # and the discovery tools available. Without it there are no usable tools.
        if not effective_can_compose and self.function_manager is None:
            raise RuntimeError(
                "CodeActActor cannot run with can_compose=False: "
                "function_manager is required so execute_function and "
                "FunctionManager discovery tools are available.",
            )

        initial_prompt = (
            "This is an interactive session. Acknowledge that you are ready and "
            "wait for the user to provide instructions via interjection."
        )

        # Clarification queues:
        # - When enabled, we ensure the handle has queues (either provided by caller or newly created).
        # - When disabled, we do not provide queues and we do not wire queue injection into environments.
        clarification_up_q: Optional[asyncio.Queue[str]]
        clarification_down_q: Optional[asyncio.Queue[str]]
        if clarification_enabled:
            clarification_up_q = _clarification_up_q or asyncio.Queue()
            clarification_down_q = _clarification_down_q or asyncio.Queue()
        else:
            clarification_up_q = None
            clarification_down_q = None

        # Create per-call environments so clarification queues are not stored on shared actor environments.
        logger.debug(f"⏱️ [CodeActActor.act +{_act_ms()}] copying environments")
        sandbox_envs: Dict[str, "BaseEnvironment"] = {}
        try:
            from unify.actor.environments.base import (
                _CompositeEnvironment as _CompositeEnv,
            )
            from unify.actor.environments import (
                ComputerEnvironment as _ComputerEnvironment,
                StateManagerEnvironment as _StateManagerEnvironment,
            )
        except Exception:
            _CompositeEnv = None  # type: ignore
            _ComputerEnvironment = None  # type: ignore
            _StateManagerEnvironment = None  # type: ignore

        for ns, env in self.environments.items():
            # Prefer explicit reconstruction for known env types.
            try:
                if _CompositeEnv is not None and isinstance(env, _CompositeEnv):
                    sandbox_envs[ns] = _CompositeEnv(
                        env.sub_environments,
                        clarification_up_q=clarification_up_q,
                        clarification_down_q=clarification_down_q,
                    )
                    continue
                if _ComputerEnvironment is not None and isinstance(
                    env,
                    _ComputerEnvironment,
                ):
                    sandbox_envs[ns] = _ComputerEnvironment(
                        env._computer_primitives,
                        clarification_up_q=clarification_up_q,
                        clarification_down_q=clarification_down_q,
                    )
                    continue
                if _StateManagerEnvironment is not None and isinstance(
                    env,
                    _StateManagerEnvironment,
                ):
                    sandbox_envs[ns] = _StateManagerEnvironment(
                        env.get_instance(),
                        clarification_up_q=clarification_up_q,
                        clarification_down_q=clarification_down_q,
                    )
                    continue
            except Exception:
                pass

            # Fallback: shallow-copy and set private queue attrs on the copy only.
            try:
                env_copy = copy.copy(env)
                if hasattr(env_copy, "_clarification_up_q"):
                    setattr(env_copy, "_clarification_up_q", clarification_up_q)
                if hasattr(env_copy, "_clarification_down_q"):
                    setattr(env_copy, "_clarification_down_q", clarification_down_q)
                sandbox_envs[ns] = env_copy
            except Exception:
                sandbox_envs[ns] = env

        # Concurrency/backpressure guard for externally started actor runs.
        logger.debug(
            f"⏱️ [CodeActActor.act +{_act_ms()}] envs copied, preparing actor slot",
        )
        acquired_actor_slot = False
        if not _reuse_actor_slot:
            try:
                await asyncio.wait_for(
                    self._act_semaphore.acquire(),
                    timeout=float(getattr(self, "_act_semaphore_timeout_s", 30.0)),
                )
                acquired_actor_slot = True
            except asyncio.TimeoutError:
                raise RuntimeError(
                    "CodeActActor is at capacity (too many concurrent sessions). "
                    "Try again later or reduce concurrency.",
                )
        logger.debug(
            f"⏱️ [CodeActActor.act +{_act_ms()}] actor slot ready, creating sandbox",
        )
        sandbox = PythonExecutionSession(
            computer_primitives=self._computer_primitives,
            environments=sandbox_envs,
            venv_pool=self._venv_pool,
            shell_pool=self._shell_pool,
        )
        # Note: __notification_up_q__ is injected dynamically by execute_code
        # when it receives a _notification_up_q from the async tool loop.

        token = _CURRENT_SANDBOX.set(sandbox)
        env_token = _CURRENT_ENVIRONMENTS.set(sandbox_envs)
        llm_profile_token = CURRENT_ACT_LLM_PROFILE.set(act_llm_profile)

        # Set agent context for depth tracking and handle access
        parent_ctx = _CURRENT_AGENT_CONTEXT.get()
        new_ctx = AgentContext(
            depth=parent_ctx.depth + 1,
            agent_id=str(uuid.uuid4()),
            handle=None,  # Will be set after handle is created
        )
        ctx_token = _CURRENT_AGENT_CONTEXT.set(new_ctx)

        # Per-trajectory package overlay: lazily installs packages into a
        # temporary directory on sys.path and cleans them up when act() ends.
        # Created after AgentContext so it can use agent_id for directory naming,
        # and after _CURRENT_PACKAGE_OVERLAY is readable so child overlays
        # discover their parent's directory for hierarchical nesting.
        pkg_overlay = PackageOverlay(agent_id=new_ctx.agent_id)
        pkg_overlay_token = _CURRENT_PACKAGE_OVERLAY.set(pkg_overlay)

        # Mutable ref populated after handle creation so _cleanup can deregister.
        _registered_queue: list[asyncio.Queue | None] = [None]

        async def _cleanup() -> None:
            if (
                _registered_queue[0] is not None
                and self._computer_primitives is not None
            ):
                self._computer_primitives.deregister_interject_queue(
                    _registered_queue[0],
                )
            try:
                pkg_overlay.cleanup()
            except Exception:
                pass
            try:
                _CURRENT_PACKAGE_OVERLAY.reset(pkg_overlay_token)
            except Exception:
                pass
            try:
                # Best-effort cleanup
                if hasattr(sandbox, "close") and callable(getattr(sandbox, "close")):
                    await sandbox.close()  # type: ignore[misc]
            except Exception:
                pass
            try:
                _CURRENT_SANDBOX.reset(token)
            except Exception:
                pass
            try:
                _CURRENT_ENVIRONMENTS.reset(env_token)
            except Exception:
                pass
            try:
                CURRENT_ACT_LLM_PROFILE.reset(llm_profile_token)
            except Exception:
                pass
            try:
                _CURRENT_AGENT_CONTEXT.reset(ctx_token)
            except Exception:
                pass
            if acquired_actor_slot:
                try:
                    self._act_semaphore.release()
                except Exception:
                    pass

        task_execution_delegate: TaskExecutionDelegate = _CodeActTaskExecutionDelegate(
            self,
        )

        # If an explicit FunctionManager entrypoint is provided (e.g., TaskScheduler task execution),
        # bypass the CodeAct LLM loop and run the function directly.
        if entrypoint is not None:
            entrypoint_id = int(entrypoint)
            args = list(entrypoint_args or [])
            kwargs_for_entrypoint = dict(entrypoint_kwargs or {})

            async def _run_entrypoint_once() -> Any:
                fm = self.function_manager
                if fm is None:
                    raise RuntimeError(
                        "CodeActActor cannot execute entrypoint: function_manager is None",
                    )

                out = fm.filter_functions(
                    filter=f"function_id == {entrypoint_id}",
                    _return_callable=True,
                    _namespace=sandbox.global_state,
                    _also_return_metadata=True,
                )
                metadata = []
                if isinstance(out, dict):
                    metadata = list(out.get("metadata") or [])
                if not metadata:
                    raise ValueError(
                        f"Entrypoint function_id {entrypoint_id} not found in FunctionManager.",
                    )
                fn_name = metadata[0].get("name")
                if not isinstance(fn_name, str) or not fn_name.strip():
                    raise ValueError(
                        f"Entrypoint {entrypoint_id} has no valid function name.",
                    )
                fn = sandbox.global_state.get(fn_name)
                if fn is None:
                    raise ValueError(
                        f"Entrypoint {entrypoint_id} ({fn_name}) was not injected into the sandbox namespace.",
                    )

                compatible_kwargs = _signature_compatible_kwargs(
                    fn,
                    kwargs_for_entrypoint,
                )
                res = fn(*args, **compatible_kwargs)
                if inspect.isawaitable(res):
                    res = await res
                return res

            async def _run_entrypoint() -> Any:
                attempts_remaining = max(0, entrypoint_repair_attempts)
                while True:
                    try:
                        return await _run_entrypoint_once()
                    except Exception as exc:
                        if attempts_remaining <= 0:
                            raise
                        attempts_remaining -= 1
                        await self._repair_symbolic_entrypoint(
                            entrypoint_id=entrypoint_id,
                            request=request,
                            entrypoint_kwargs=kwargs_for_entrypoint,
                            failure=exc,
                            repair_context=(
                                entrypoint_repair_context
                                if isinstance(entrypoint_repair_context, dict)
                                else None
                            ),
                        )

            delegate_token = current_task_execution_delegate.set(
                task_execution_delegate,
            )
            try:
                entry_task = asyncio.create_task(_run_entrypoint())
                entry_handle = _CodeActEntrypointHandle(
                    entrypoint_id=entrypoint_id,
                    execution_task=entry_task,
                    on_finally=_cleanup,
                )
            finally:
                current_task_execution_delegate.reset(delegate_token)
            return entry_handle

        # Build the tool set for this call. When can_compose=False the LLM
        # can_compose=False: specialist may only discover and execute stored
        # functions — no arbitrary code, no function persistence.
        # can_store=False: function/guidance library is read-only.
        # Session tools are kept because execute_function supports the same
        # session/state_mode semantics.
        _compose_only_tools = {
            "execute_code",
            "install_python_packages",
        }
        _store_only_tools = {
            "store_skills",
            "FunctionManager_add_functions",
            "FunctionManager_delete_function",
        }

        def _filter_tools(tool_dict: Dict[str, Any]) -> Dict[str, Any]:
            """Apply static per-call filters (can_compose, can_store)."""
            out = dict(tool_dict)
            if not effective_can_compose:
                for name in _compose_only_tools:
                    out.pop(name, None)
                for name in _store_only_tools:
                    out.pop(name, None)
            if not effective_can_store:
                for name in _store_only_tools:
                    out.pop(name, None)
            return out

        base_tools = _filter_tools(self.get_tools("act"))

        # When execute_code is masked (can_compose=False), strip any
        # execute_code references from execute_function's docstring so the
        # LLM has no awareness that a code sandbox exists.
        if "execute_function" in base_tools and "execute_code" not in base_tools:
            _ef = base_tools["execute_function"]
            (_ef.fn if isinstance(_ef, ToolSpec) else _ef).__doc__ = (
                "Execute a known function by name and return its result.\n"
                "\n"
                "The function is resolved from the sandbox namespace or looked up\n"
                "in the FunctionManager by exact name. Functions discovered via the\n"
                "FunctionManager discovery tools are automatically available.\n"
                "\n"
                "Workflow\n"
                "-------\n"
                "1. Discover functions via ``FunctionManager_search_functions``,\n"
                "   ``FunctionManager_filter_functions``, or\n"
                "   ``FunctionManager_list_functions``.\n"
                "2. Pick the best match by name.\n"
                "3. Call ``execute_function(function_name=..., call_kwargs=...)``.\n"
                "\n"
                "Key concepts\n"
                "------------\n"
                '- **language**: ``"python"`` | ``"bash"`` | ``"zsh"`` | '
                '``"sh"`` | ``"powershell"``\n'
                "- **state_mode**:\n"
                '  - ``"stateless"``: no session; clean execution; no persistence\n'
                '  - ``"stateful"``: persistent session; state accumulates\n'
                '  - ``"read_only"``: reads from an existing session but does not\n'
                "    persist changes\n"
                "- **session_id / session_name**: only meaningful for\n"
                "  stateful / read_only\n"
                "\n"
                "Parameters\n"
                "----------\n"
                "function_name : str\n"
                "    Exact name of the function to execute.\n"
                "call_kwargs : dict, optional\n"
                "    Keyword arguments to pass to the function.\n"
                'language : str, default ``"python"``\n'
                "    Language of the function.\n"
                'state_mode : str, default ``"stateless"``\n'
                "    Execution state mode.\n"
                "session_id : int | None\n"
                "    Session ID for stateful/read_only modes.\n"
                "session_name : str | None\n"
                "    Human-friendly session alias.\n"
                "venv_id : int | None\n"
                "    Virtual environment ID (Python only).\n"
                "\n"
                "Returns\n"
                "-------\n"
                "dict | ExecutionResult\n"
                "    Same shape as code execution output (stdout, stderr, result,\n"
                "    error, language, state_mode, session_id, session_name, venv_id,\n"
                "    session_created, duration_ms).\n"
            )

        integration_summary = ""
        try:
            from unify.integration_status import enabled_summary_for_prompt

            integration_summary = enabled_summary_for_prompt()
        except Exception:
            integration_summary = ""
        effective_guidelines = (
            "\n\n".join(
                filter(None, [self._base_guidelines, guidelines, integration_summary]),
            )
            or None
        )

        logger.debug(f"⏱️ [CodeActActor.act +{_act_ms()}] building system prompt")
        system_prompt = build_code_act_prompt(
            environments=sandbox_envs,
            tools=base_tools,
            can_store=effective_can_store,
            guidelines=effective_guidelines,
            discovery_first_policy=self.tool_policy is _USE_DEFAULT,
        )
        logger.debug(
            f"⏱️ [CodeActActor.act +{_act_ms()}] prompt built "
            f"({len(system_prompt)} chars, {len(base_tools)} tools)",
        )

        # Tool policy controls which tools are visible per turn, and whether a
        # tool call is required.  The static _filter_tools (can_compose,
        # can_store, can_spawn_sub_agents) is always applied regardless of
        # the dynamic policy.
        if self.tool_policy is None:
            # No dynamic policy -- only static filtering on every turn.
            def _static_only_policy(step: int, tools: Dict[str, Any]):
                return "auto", _filter_tools(tools)

            tool_policy: Optional[ToolPolicyFn] = _static_only_policy
        elif self.tool_policy is _USE_DEFAULT:
            # Default discovery-first policy (both FM and GM gates).
            _has_fm_tools = any(
                isinstance(k, str) and k.startswith("FunctionManager_")
                for k in base_tools.keys()
            )
            _has_gm_tools = any(
                isinstance(k, str) and k.startswith("GuidanceManager_")
                for k in base_tools.keys()
            )
            tool_policy = _default_tool_policy(
                _has_fm_tools,
                _has_gm_tools,
                _filter_tools,
            )
        else:
            # Custom caller-provided policy.  Wrap it so that _filter_tools
            # is always applied first (static filters are never bypassed).
            _user_policy = self.tool_policy

            def _wrapped_policy(step: int, tools: Dict[str, Any]):
                return _user_policy(step, _filter_tools(tools))

            tool_policy = _wrapped_policy

        # Build an LLM client for this act() call. The profile is per-call so
        # concurrent runs on the same actor can use different models safely.
        client_model = act_llm_profile.model or self._model
        client = new_llm_client(client_model, **act_llm_profile.client_kwargs)
        if system_prompt:
            client.set_system_message(system_prompt)

        tools = dict(base_tools)

        # Build event bus callbacks for clarification and notification tools
        # (the loop creates the tools; we just provide the event hooks).
        _clar_queues = None
        _on_clar_req = None
        _on_clar_ans = None
        if clarification_up_q is not None and clarification_down_q is not None:
            _clar_queues = (clarification_up_q, clarification_down_q)

            async def _on_clar_req(q: str):
                try:
                    await EVENT_BUS.publish(
                        Event(
                            type="ManagerMethod",
                            calling_id=_call_id,
                            payload={
                                "manager": "CodeActActor",
                                "method": "act",
                                "action": "clarification_request",
                                "question": q,
                            },
                        ),
                    )
                except Exception:
                    pass

            async def _on_clar_ans(ans: str):
                try:
                    await EVENT_BUS.publish(
                        Event(
                            type="ManagerMethod",
                            calling_id=_call_id,
                            payload={
                                "manager": "CodeActActor",
                                "method": "act",
                                "action": "clarification_answer",
                                "answer": ans,
                            },
                        ),
                    )
                except Exception:
                    pass

        async def _on_notify(message: str):
            try:
                await EVENT_BUS.publish(
                    Event(
                        type="ManagerMethod",
                        calling_id=_call_id,
                        payload={
                            "manager": "CodeActActor",
                            "method": "act",
                            "action": "notification",
                            "message": message,
                        },
                    ),
                )
            except Exception:
                pass

        logger.debug(f"⏱️ [CodeActActor.act +{_act_ms()}] starting async tool loop")
        delegate_token = current_task_execution_delegate.set(task_execution_delegate)
        try:
            handle = start_async_tool_loop(
                client,
                request or initial_prompt,
                tools,
                loop_id=f"CodeActActor.act",
                parent_chat_context=_parent_chat_context,
                interrupt_llm_with_interjections=True,
                log_steps=True,
                tool_policy=tool_policy,
                response_format=response_format,
                persist=persist,
                preprocess_msgs=self._preprocess_msgs,
                prompt_caching=self._prompt_caching,
                extra_ask_tools=self._get_extra_ask_tools(),
                extra_compression_tools=(
                    ["store_skills"] if effective_can_store else None
                ),
                clarification_queues=_clar_queues,
                on_clarification_request=_on_clar_req,
                on_clarification_answer=_on_clar_ans,
                on_notify=_on_notify,
            )
        finally:
            current_task_execution_delegate.reset(delegate_token)
        logger.debug(
            f"⏱️ [CodeActActor.act +{_act_ms()}] loop started, returning handle",
        )

        # Wrap result() to run cleanup when the loop finishes
        _original_result = handle.result

        async def _result_with_cleanup() -> str:
            delegate_token = current_task_execution_delegate.set(
                task_execution_delegate,
            )
            try:
                try:
                    return await _original_result()
                finally:
                    current_task_execution_delegate.reset(delegate_token)
            finally:
                await _cleanup()

        handle.result = _result_with_cleanup  # type: ignore[assignment]

        # Wrap pause()/resume() to propagate to the browser agent
        if self._computer_primitives is not None:
            _cp: ComputerPrimitives = self._computer_primitives
            _original_pause = handle.pause
            _original_resume = handle.resume

            async def _pause_with_propagation(**kwargs: Any) -> None:
                await _original_pause(**kwargs)
                await _cp.pause()

            async def _resume_with_propagation(**kwargs: Any) -> None:
                await _cp.resume()
                await _original_resume(**kwargs)

            handle.pause = _pause_with_propagation  # type: ignore[assignment]
            handle.resume = _resume_with_propagation  # type: ignore[assignment]

            # Register the loop's interject queue so environmental state
            # changes (e.g. user remote control) are broadcast to this actor.
            _cp.register_interject_queue(handle._queue)
            _registered_queue[0] = handle._queue

        # Update agent context with handle reference
        new_ctx.handle = handle

        post_run_review_context = current_post_run_review_context.get()

        # Wrap in StorageCheckHandle for post-completion function review.
        if effective_can_store or post_run_review_context is not None:
            handle = _StorageCheckHandle(
                inner=handle,
                actor=self,
                post_run_review_context=post_run_review_context,
            )

        return handle

    async def close(self):
        """Shuts down the actor and its associated resources gracefully."""
        # Close any in-process session sandboxes owned by the session executor.
        try:
            await self._session_executor.close()
        except Exception:
            pass

        # Clear session name registry.
        try:
            self._session_names.clear()
            self._session_names_rev.clear()
        except Exception:
            pass

        # Close the pools (terminates persistent subprocess/session connections)
        await self._venv_pool.close()
        await self._shell_pool.close()

        # The ComputerPrimitives backend is a process-wide singleton (one VM,
        # one screen).  Individual actors must not tear it down — the process
        # owns the lifecycle.
