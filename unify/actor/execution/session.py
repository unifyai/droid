"""Execution sessions and the unified SessionExecutor.

Provides PythonExecutionSession (in-process stateful sandbox),
SessionExecutor (multi-language, multi-session orchestrator),
and related validation / shell helpers.
"""

from __future__ import annotations

import asyncio
import ast
import contextvars
import logging
import json
import shlex
import sys
import traceback
import types
import uuid
from datetime import datetime, timezone
from typing import (
    Any,
    Callable,
    Dict,
    Literal,
    Optional,
    Tuple,
    TYPE_CHECKING,
)

from unify.function_manager.primitives import ComputerPrimitives
from unify.common.hierarchical_logger import DEFAULT_ICON

from .capture import _stdout_parts, capture_sandbox_output
from .types import TextPart

if TYPE_CHECKING:
    from unify.actor.environments.base import BaseEnvironment
    from unify.function_manager.function_manager import FunctionManager


logger = logging.getLogger(__name__)


def _with_shell_env_overlay(
    command: str,
    env_overlay: dict[str, str],
    *,
    language: str,
) -> str:
    if not env_overlay:
        return command
    if language == "powershell":
        assignments = "\n".join(
            f"$env:{key} = {json.dumps(value)}" for key, value in env_overlay.items()
        )
    else:
        assignments = "\n".join(
            f"export {key}={shlex.quote(value)}" for key, value in env_overlay.items()
        )
    return f"{assignments}\n{command}"


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
SupportedShellLanguage = Literal["bash", "zsh", "sh", "powershell"]
SupportedLanguage = Literal["python", "bash", "zsh", "sh", "powershell"]
StateMode = Literal["stateful", "read_only", "stateless"]
SessionKey = Tuple[str, Optional[int], int]  # (language, venv_id, session_id)


# ---------------------------------------------------------------------------
# ContextVars
# ---------------------------------------------------------------------------
_CURRENT_SANDBOX: contextvars.ContextVar["PythonExecutionSession"] = (
    contextvars.ContextVar(
        "code_act_current_sandbox",
    )
)

_PARENT_CHAT_CONTEXT: contextvars.ContextVar[list | None] = contextvars.ContextVar(
    "code_act_parent_chat_context",
    default=None,
)

_CURRENT_ENVIRONMENTS: contextvars.ContextVar[dict] = contextvars.ContextVar(
    "code_act_current_environments",
    default={},
)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------
def _validation_error(
    *,
    message: str,
    suggestion: str,
    state_mode: str,
    session_id: int | None,
    session_name: str | None,
    language: str,
    venv_id: int | None = None,
) -> dict:
    return {
        "error": message,
        "error_type": "validation",
        "suggestion": suggestion,
        "received": {
            "state_mode": state_mode,
            "session_id": session_id,
            "session_name": session_name,
            "language": language,
            "venv_id": venv_id,
        },
    }


def _validate_execution_params(
    *,
    state_mode: str,
    session_id: int | None,
    session_name: str | None,
    language: str,
    venv_id: int | None = None,
    supported_languages: tuple[str, ...] = (
        "python",
        "bash",
        "zsh",
        "sh",
        "powershell",
    ),
    # Name resolution/lookup is actor-owned, so validation accepts callables.
    resolve_session_name: Optional[Callable[[str], Optional[SessionKey]]] = None,
    get_session_name_for_id: Optional[
        Callable[[str, Optional[int], int], Optional[str]]
    ] = None,
    session_exists: Optional[Callable[[str, Optional[int], int], bool]] = None,
    max_sessions_total: Optional[int] = None,
    active_session_count: Optional[int] = None,
) -> dict | None:
    """
    Validate state_mode + session selection rules for execute_code.

    Returns:
        None if valid, otherwise a structured validation error dict.

    Notes:
        This function intentionally returns structured errors (not exceptions)
        so the LLM can self-correct deterministically.
    """
    if language not in supported_languages:
        return _validation_error(
            message=f"Unsupported language: {language!r}",
            suggestion=f"Use one of: {sorted(supported_languages)}",
            state_mode=state_mode,
            session_id=session_id,
            session_name=session_name,
            language=language,
            venv_id=venv_id,
        )

    if state_mode not in ("stateful", "read_only", "stateless"):
        return _validation_error(
            message=f"Unsupported state_mode: {state_mode!r}",
            suggestion="Use one of: 'stateful', 'read_only', 'stateless'",
            state_mode=state_mode,
            session_id=session_id,
            session_name=session_name,
            language=language,
            venv_id=venv_id,
        )

    # Stateless must not reference sessions.
    if state_mode == "stateless" and (
        session_id is not None or session_name is not None
    ):
        return _validation_error(
            message="Cannot use state_mode='stateless' with a session.",
            suggestion="Remove session_id/session_name or switch to state_mode='stateful' or 'read_only'.",
            state_mode=state_mode,
            session_id=session_id,
            session_name=session_name,
            language=language,
            venv_id=venv_id,
        )

    # Read-only requires an existing session.
    if state_mode == "read_only" and (session_id is None and session_name is None):
        return _validation_error(
            message="Cannot use state_mode='read_only' without specifying a session.",
            suggestion="Provide session_id or session_name (must refer to an existing session), or use state_mode='stateless'.",
            state_mode=state_mode,
            session_id=session_id,
            session_name=session_name,
            language=language,
            venv_id=venv_id,
        )

    # If both are present, ensure they match.
    if session_id is not None and session_name is not None:
        if resolve_session_name is None:
            return _validation_error(
                message="Cannot validate session_name against session_id (no resolver configured).",
                suggestion="Specify only one of session_id or session_name.",
                state_mode=state_mode,
                session_id=session_id,
                session_name=session_name,
                language=language,
                venv_id=venv_id,
            )
        key = resolve_session_name(session_name)
        if key is None:
            # For stateful, the caller may choose to create+bind; for read_only it must exist.
            if state_mode == "read_only":
                return _validation_error(
                    message=f"Session name {session_name!r} not found for read_only execution.",
                    suggestion="Use an existing session_name (see list_sessions) or specify an existing session_id.",
                    state_mode=state_mode,
                    session_id=session_id,
                    session_name=session_name,
                    language=language,
                    venv_id=venv_id,
                )
        else:
            resolved_language, resolved_venv_id, resolved_session_id = key
            if (
                resolved_language != language
                or resolved_venv_id != venv_id
                or resolved_session_id != session_id
            ):
                return _validation_error(
                    message=(
                        f"session_id and session_name refer to different sessions. "
                        f"{session_name!r} resolves to {(resolved_language, resolved_venv_id, resolved_session_id)} "
                        f"but received {(language, venv_id, session_id)}."
                    ),
                    suggestion="Specify only one of session_id or session_name, or make them consistent.",
                    state_mode=state_mode,
                    session_id=session_id,
                    session_name=session_name,
                    language=language,
                    venv_id=venv_id,
                )

    # If session_name is provided alone:
    if session_name is not None and session_id is None:
        if resolve_session_name is None:
            return _validation_error(
                message="Cannot resolve session_name (no resolver configured).",
                suggestion="Specify session_id instead, or configure a session registry.",
                state_mode=state_mode,
                session_id=session_id,
                session_name=session_name,
                language=language,
                venv_id=venv_id,
            )
        key = resolve_session_name(session_name)
        if key is None and state_mode == "read_only":
            return _validation_error(
                message=f"Session name {session_name!r} not found for read_only execution.",
                suggestion="Use an existing session_name (see list_sessions) or specify an existing session_id.",
                state_mode=state_mode,
                session_id=session_id,
                session_name=session_name,
                language=language,
                venv_id=venv_id,
            )

    # Optional: enforce a global session cap (actor-owned pools, so this is per-actor).
    # Note: `stateful` with no session specified means "use the default session" (session_id=0),
    # so it does NOT create a new session and should not be rejected by the global cap here.
    #
    # We only apply the cap when the call would CREATE a new stateful session:
    # - session_name is provided but does not resolve (new session would be allocated)
    if (
        max_sessions_total is not None
        and active_session_count is not None
        and state_mode == "stateful"
        and session_id is None
        and session_name is not None
        and resolve_session_name is not None
        and resolve_session_name(session_name) is None
        and active_session_count >= max_sessions_total
    ):
        return _validation_error(
            message=f"Session limit exceeded: {active_session_count} active sessions (max {max_sessions_total}).",
            suggestion="Close an existing session (close_session) or reuse an existing session_id/session_name.",
            state_mode=state_mode,
            session_id=session_id,
            session_name=session_name,
            language=language,
            venv_id=venv_id,
        )

    # If an explicit session_id is provided for stateful execution, enforce limits when it would
    # CREATE a new session (important for in-process Python sessions where pool-level limits may not apply).
    if (
        max_sessions_total is not None
        and active_session_count is not None
        and state_mode == "stateful"
        and session_id is not None
        and session_exists is not None
        and active_session_count >= max_sessions_total
    ):
        try:
            exists = bool(session_exists(language, venv_id, session_id))
        except Exception:
            exists = False
        if not exists:
            return _validation_error(
                message=f"Session limit exceeded: {active_session_count} active sessions (max {max_sessions_total}).",
                suggestion="Close an existing session (close_session) or reuse an existing session_id/session_name.",
                state_mode=state_mode,
                session_id=session_id,
                session_name=session_name,
                language=language,
                venv_id=venv_id,
            )

    # If session_id is provided and we can validate existence for read_only, do so.
    if (
        state_mode == "read_only"
        and session_id is not None
        and session_exists is not None
    ):
        if not session_exists(language, venv_id, session_id):
            name_hint = None
            if get_session_name_for_id is not None:
                name_hint = get_session_name_for_id(language, venv_id, session_id)
            hint = f" (known name: {name_hint!r})" if name_hint else ""
            return _validation_error(
                message=f"Session {(language, venv_id, session_id)} does not exist for read_only execution{hint}.",
                suggestion="Use list_sessions to find an existing session, or switch to state_mode='stateful' to create a new session.",
                state_mode=state_mode,
                session_id=session_id,
                session_name=session_name,
                language=language,
                venv_id=venv_id,
            )

    return None


# ---------------------------------------------------------------------------
# PythonExecutionSession
# ---------------------------------------------------------------------------
class PythonExecutionSession:
    """
    A stateful execution environment for running Python code asynchronously.

    This class maintains a persistent global state across multiple executions,
    capturing stdout, stderr, return values, and exceptions in a structured format.

    It can optionally use pools for persistent subprocess connections (VenvPool
    for Python venvs, ShellPool for shell sessions), enabling state to be
    preserved across multiple function calls.
    """

    def __init__(
        self,
        computer_primitives: Optional[ComputerPrimitives] = None,
        environments: Optional[Dict[str, "BaseEnvironment"]] = None,
        venv_pool: Optional[Any] = None,
        shell_pool: Optional[Any] = None,
    ):
        """
        Initializes the execution environment.

        Args:
            computer_primitives: An instance of ComputerPrimitives to be injected into the
                             global state, making computer tools available.
            environments: Optional mapping of environment namespaces to environments. If
                provided, each environment instance is injected into globals.
            venv_pool: Optional VenvPool for persistent Python venv connections.
                If provided, venv-backed functions will use persistent connections
                that maintain state across calls.
            shell_pool: Optional ShellPool for persistent shell session connections.
                If provided, shell functions will use persistent sessions.
        """
        from unify.function_manager.execution_env import create_execution_globals

        self.id: str = str(uuid.uuid4())
        self._module_name: str = f"__sandbox_{self.id}__"

        # Register the sandbox globals as a proper module in sys.modules.
        _mod = types.ModuleType(self._module_name)
        _initial = create_execution_globals()
        _mod.__dict__.update(_initial)
        sys.modules[self._module_name] = _mod
        self.global_state: Dict[str, Any] = _mod.__dict__
        self.global_state["__name__"] = self._module_name

        # Expose sandbox metadata to user code (best-effort; callers may ignore).
        self.global_state["__sandbox_id__"] = self.id

        # Notification queue is injected dynamically by execute_code when it
        # receives a _notification_up_q from the async tool loop:
        # sandbox.global_state["__notification_up_q__"] = <asyncio.Queue>
        #
        # Provide a user-driven progress helper:
        #   notify({"type": "...", ...})
        # This helper is intentionally synchronous; it uses put_nowait.
        # Notifications bubble up through the async tool loop to the outer handle.
        def notify(payload: dict) -> None:
            try:
                q = self.global_state.get("__notification_up_q__")
                if q is None:
                    return
                # Queue is expected to be an asyncio.Queue[dict]
                q.put_nowait(payload)
            except Exception:
                return

        self.global_state["notify"] = notify

        # Inject pools into namespace (for function proxies to use)
        if venv_pool is not None:
            self.global_state["__venv_pool__"] = venv_pool
        if shell_pool is not None:
            self.global_state["__shell_pool__"] = shell_pool

        if environments:
            for namespace, env in environments.items():
                try:
                    # Use get_sandbox_instance() if available (for filtered primitives),
                    # otherwise fall back to get_instance()
                    if hasattr(env, "get_sandbox_instance"):
                        instance = env.get_sandbox_instance()
                    else:
                        instance = env.get_instance()
                    self.global_state[namespace] = instance
                except Exception:
                    # Keep sandbox usable even if a non-critical environment fails to inject.
                    continue

        # Backward-compat: if computer_primitives was passed directly and no
        # "primitives" namespace is present, inject a Primitives wrapper so
        # primitives.computer.* calls work.
        if computer_primitives and "primitives" not in self.global_state:
            from unify.function_manager.primitives import Primitives, PrimitiveScope

            self.global_state["primitives"] = Primitives(
                primitive_scope=PrimitiveScope(
                    scoped_managers=frozenset({"computer"}),
                ),
            )

    async def close(self) -> None:
        """
        Best-effort cleanup for an ephemeral sandbox instance.

        Notes
        -----
        - Pools (venv/shell) are owned by the actor and are not closed here.
        - This method is safe to call multiple times.
        """
        try:
            sys.modules.pop(self._module_name, None)
            self.global_state.clear()
        except Exception as e:
            try:
                logger.warning(
                    f"{DEFAULT_ICON} PythonExecutionSession.close() failed: {e}",
                    exc_info=True,
                )
            except Exception:
                pass

    async def execute(self, code: str, *, timeout: float | None = None) -> dict:
        """
        Executes a string of Python code within the sandbox's stateful environment.

        Returns a dict with:
            stdout: list[OutputPart] - structured output parts (TextPart, ImagePart)
            stderr: list[OutputPart] - structured error output parts
            result: Any - return value of the last expression
            error: str | None - traceback if an exception occurred
        """
        result = None
        error = None

        with capture_sandbox_output() as (stdout_parts, stderr_parts, display_fn):
            # Inject display function into globals
            self.global_state["display"] = display_fn

            try:
                # Guardrails: prevent agent code from accidentally shadowing critical
                # injected environment globals (common failure mode in LLM-generated code).
                #
                # We do this via an AST rewrite (not brittle string heuristics):
                # rewrite any assignment targets named `primitives`
                # to `_primitives_local`.
                #
                # This preserves the injected globals for the rest of the session.
                try:

                    class _ShadowingGuard(ast.NodeTransformer):
                        _REMAP = {
                            "primitives": "_primitives_local",
                        }

                        def visit_Name(self, node: ast.Name) -> ast.AST:  # noqa: N802
                            # Only rewrite *assignments* (Store context). Loads are preserved.
                            if (
                                isinstance(node.ctx, ast.Store)
                                and node.id in self._REMAP
                            ):
                                return ast.copy_location(
                                    ast.Name(id=self._REMAP[node.id], ctx=node.ctx),
                                    node,
                                )
                            return node

                        def visit_Global(
                            self,
                            node: ast.Global,
                        ) -> ast.AST:  # noqa: N802
                            node.names = [self._REMAP.get(n, n) for n in node.names]
                            return node

                        def visit_Nonlocal(
                            self,
                            node: ast.Nonlocal,
                        ) -> ast.AST:  # noqa: N802
                            node.names = [self._REMAP.get(n, n) for n in node.names]
                            return node

                    tree = ast.parse(code)
                    tree = _ShadowingGuard().visit(tree)  # type: ignore[assignment]
                    ast.fix_missing_locations(tree)
                    code = ast.unparse(tree)
                except Exception:
                    # Best-effort only; if rewriting fails, proceed with original code.
                    pass

                is_empty_or_comment_only = all(
                    line.strip() == "" or line.strip().startswith("#")
                    for line in code.splitlines()
                )
                if is_empty_or_comment_only:
                    code += "\npass"

                tree = ast.parse(code)
                top_level_assign_targets = set()
                for node in tree.body:
                    if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                        targets = []
                        if isinstance(node, ast.Assign):
                            targets.extend(node.targets)
                        else:
                            targets.append(node.target)

                        for target in targets:
                            if isinstance(target, ast.Name):
                                top_level_assign_targets.add(target.id)
                            elif isinstance(target, ast.Tuple):
                                for elt in target.elts:
                                    if isinstance(elt, ast.Name):
                                        top_level_assign_targets.add(elt.id)

                    elif isinstance(node, (ast.Import, ast.ImportFrom)):
                        for alias in node.names:
                            top_level_assign_targets.add(
                                alias.asname or alias.name.split(".")[0],
                            )

                    elif isinstance(
                        node,
                        (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
                    ):
                        top_level_assign_targets.add(node.name)

                # REPL semantics: implicitly return the last expression's value
                # so callers can read it from the `result` field.
                if tree.body and isinstance(tree.body[-1], ast.Expr):
                    tree.body[-1] = ast.Return(value=tree.body[-1].value)
                    ast.fix_missing_locations(tree)
                    code = ast.unparse(tree)

                async_code = "async def __exec_wrapper():\n"
                if top_level_assign_targets:
                    async_code += f"    global {', '.join(sorted(list(top_level_assign_targets)))}\n"

                async_code += "".join(f"    {line}\n" for line in code.splitlines())

                # Inject a custom print function that writes directly to our capture
                # list via ContextVar, bypassing sys.stdout entirely. This is
                # necessary because pytest's live logging feature can replace
                # sys.stdout during LOGGER.info() calls, breaking our StreamRouter.
                _gs_builtins = self.global_state.get("__builtins__", {})
                if isinstance(_gs_builtins, dict):
                    _original_print = _gs_builtins.get("print")

                    def _sandbox_print(
                        *args,
                        sep=" ",
                        end="\n",
                        file=None,
                        flush=False,
                    ):
                        # If file is explicitly specified, use the original print
                        if file is not None:
                            if _original_print:
                                return _original_print(
                                    *args,
                                    sep=sep,
                                    end=end,
                                    file=file,
                                    flush=flush,
                                )
                            return
                        # Otherwise, write directly to our capture list via ContextVar
                        try:
                            parts = _stdout_parts.get()
                        except LookupError:
                            # No capture context - fall back to original print
                            if _original_print:
                                return _original_print(
                                    *args,
                                    sep=sep,
                                    end=end,
                                    flush=flush,
                                )
                            return
                        # Format the output like standard print
                        output = sep.join(str(arg) for arg in args) + end
                        # Merge consecutive text writes into a single TextPart
                        if parts and isinstance(parts[-1], TextPart):
                            last = parts[-1]
                            parts[-1] = TextPart(text=last.text + output)
                        else:
                            parts.append(TextPart(text=output))

                    self.global_state["__builtins__"]["print"] = _sandbox_print

                # Wrap primitives with ContextForwardingProxy if parent
                # chat context is available, so inner tool loops receive it.
                _pcc = _PARENT_CHAT_CONTEXT.get()
                _orig_prims = self.global_state.get("primitives")
                if _pcc is not None and _orig_prims is not None:
                    from unify.function_manager.primitives.context_proxy import (
                        ContextForwardingProxy,
                    )

                    self.global_state["primitives"] = ContextForwardingProxy(
                        _orig_prims,
                        _parent_chat_context=_pcc,
                    )

                try:
                    exec(async_code, self.global_state)
                    execution = self.global_state["__exec_wrapper"]()
                    if timeout is None:
                        result = await execution
                    else:
                        result = await asyncio.wait_for(execution, timeout=timeout)
                finally:
                    if _orig_prims is not None:
                        self.global_state["primitives"] = _orig_prims

            except asyncio.TimeoutError:
                error = f"Python execution timed out after {timeout}s"
            except Exception:
                error = traceback.format_exc()
            finally:
                if "__exec_wrapper" in self.global_state:
                    del self.global_state["__exec_wrapper"]

        return {
            "stdout": stdout_parts,
            "stderr": stderr_parts,
            "result": result,
            "error": error,
        }


# ---------------------------------------------------------------------------
# SessionExecutor
# ---------------------------------------------------------------------------
class SessionExecutor:
    """
    Unified execution engine for multi-language, multi-session CodeAct execution.

    Notes
    -----
    - Python (in-process) sessions are backed by persistent PythonExecutionSession instances.
    - Shell sessions use ShellPool (persistent) and ephemeral subprocesses for stateless.
    - Venv-backed Python sessions are supported when venv_id is provided.
    """

    def __init__(
        self,
        *,
        venv_pool: Any,
        shell_pool: Any,
        environments: Optional[Dict[str, "BaseEnvironment"]] = None,
        computer_primitives: Optional[ComputerPrimitives] = None,
        function_manager: Optional["FunctionManager"] = None,
        timeout: Optional[float] = None,
    ) -> None:
        self._venv_pool = venv_pool
        self._shell_pool = shell_pool
        self._environments = environments or {}
        self._computer_primitives = computer_primitives
        self._function_manager = function_manager
        self._timeout = timeout

        # In-process Python sessions keyed by (venv_id=None, session_id).
        self._python_sessions: Dict[
            Tuple[Optional[int], int],
            PythonExecutionSession,
        ] = {}
        self._python_session_meta: Dict[Tuple[Optional[int], int], dict[str, str]] = {}

        self._fm_globals: Dict[str, Any] = {}

    def register_fm_globals(self, globals_dict: Dict[str, Any]) -> None:
        self._fm_globals.update(globals_dict)

    def _inject_fm_globals(self, sb: PythonExecutionSession) -> None:
        if self._fm_globals:
            sb.global_state.update(self._fm_globals)

    def has_python_session(
        self,
        *,
        session_id: int,
        venv_id: int | None = None,
    ) -> bool:
        return (venv_id, session_id) in self._python_sessions

    def list_in_process_python_sessions(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for (venv_id, session_id), sb in list(self._python_sessions.items()):
            meta = self._python_session_meta.get((venv_id, session_id)) or {}
            out.append(
                {
                    "language": "python",
                    "venv_id": venv_id,
                    "session_id": int(session_id),
                    "created_at": meta.get("created_at"),
                    "last_used": meta.get("last_used"),
                    "state_summary": "active",
                },
            )
        return out

    async def close_in_process_python_session(
        self,
        *,
        session_id: int,
        venv_id: int | None = None,
    ) -> bool:
        key = (venv_id, int(session_id))
        sb = self._python_sessions.pop(key, None)
        self._python_session_meta.pop(key, None)
        if sb is None:
            return False
        try:
            await sb.close()
        except Exception:
            pass
        return True

    async def close(self) -> None:
        # Close in-process python sandboxes; pools are owned by the actor.
        for sb in list(self._python_sessions.values()):
            try:
                await sb.close()
            except Exception:
                pass
        self._python_sessions.clear()
        self._python_session_meta.clear()

    async def execute(
        self,
        *,
        code: str,
        language: SupportedLanguage,
        state_mode: StateMode,
        session_id: int | None,
        venv_id: int | None,
        primitives: Any = None,
        computer_primitives: Any = None,
        notification_q: asyncio.Queue[dict] | None = None,
    ) -> Dict[str, Any]:
        import time as _se_time
        import logging as _se_logging

        _se_t0 = _se_time.perf_counter()
        _se_log = _se_logging.getLogger("unify")

        def _se_ms():
            return f"{(_se_time.perf_counter() - _se_t0) * 1000:.0f}ms"

        _se_log.debug(
            f"⏱️ [SessionExecutor.execute +{_se_ms()}] entered "
            f"(lang={language}, state_mode={state_mode}, session_id={session_id})",
        )

        started = datetime.now(timezone.utc)
        t0 = started.timestamp()

        # Ensure the localhost provider proxy is up before any code runs so its
        # base URLs / nonce are available to both in-process and subprocess
        # backends (connected-provider REST is reached only through it).
        try:
            from unify.provider_proxy.proxy import ensure_proxy_running

            ensure_proxy_running()
        except Exception:
            _se_log.warning("provider proxy failed to start", exc_info=True)

        # Default: use actor computer primitives (if any).
        if computer_primitives is None:
            computer_primitives = self._computer_primitives

        def _runtime_oauth_env_overlay() -> dict[str, str]:
            # The parent execute_code boundary already performs the generic
            # debounced secret sync.  This overlay is the subprocess-specific
            # bridge: venv and shell sessions may be long-lived, so they need
            # current rotating OAuth env vars injected for each execution.
            if self._function_manager is None:
                return {}
            getter = getattr(
                self._function_manager,
                "_get_runtime_oauth_env_overlay",
                None,
            )
            if getter is None:
                return {}
            return getter()

        async def _execute_in_python_session(
            sb: PythonExecutionSession,
        ) -> Dict[str, Any]:
            notification_sentinel = object()
            previous_notification_q = sb.global_state.get(
                "__notification_up_q__",
                notification_sentinel,
            )
            if notification_q is not None:
                sb.global_state["__notification_up_q__"] = notification_q
            else:
                sb.global_state.pop("__notification_up_q__", None)
            try:
                return await sb.execute(code, timeout=self._timeout)
            finally:
                if previous_notification_q is notification_sentinel:
                    sb.global_state.pop("__notification_up_q__", None)
                else:
                    sb.global_state["__notification_up_q__"] = previous_notification_q

        # ─── Python ────────────────────────────────────────────────────────
        if language == "python":
            # Special-case: session 0 is the *current bound sandbox* when present.
            if state_mode == "stateful" and venv_id is None and session_id == 0:
                try:
                    sb0 = _CURRENT_SANDBOX.get()
                    self._inject_fm_globals(sb0)
                    _se_log.debug(
                        f"⏱️ [SessionExecutor.execute +{_se_ms()}] bound sandbox (session 0), executing",
                    )
                    res = await _execute_in_python_session(sb0)
                    _se_log.debug(
                        f"⏱️ [SessionExecutor.execute +{_se_ms()}] bound sandbox done",
                    )
                    return {
                        **res,
                        "language": language,
                        "state_mode": state_mode,
                        "session_id": 0,
                        "venv_id": None,
                        "session_created": False,
                        "duration_ms": int(
                            (datetime.now(timezone.utc).timestamp() - t0) * 1000,
                        ),
                    }
                except Exception:
                    # If no sandbox is bound, fall back to executor-managed session 0.
                    pass
            # Stateless: fresh in-process sandbox per call.
            if state_mode == "stateless" and venv_id is None:
                _se_log.debug(
                    f"⏱️ [SessionExecutor.execute +{_se_ms()}] creating stateless sandbox",
                )
                sb = PythonExecutionSession(
                    computer_primitives=computer_primitives,
                    environments=self._environments,
                    venv_pool=self._venv_pool,
                    shell_pool=self._shell_pool,
                )
                _se_log.debug(
                    f"⏱️ [SessionExecutor.execute +{_se_ms()}] sandbox created, injecting globals",
                )
                self._inject_fm_globals(sb)
                _se_log.debug(
                    f"⏱️ [SessionExecutor.execute +{_se_ms()}] globals injected, executing code",
                )
                try:
                    res = await _execute_in_python_session(sb)
                    _se_log.debug(
                        f"⏱️ [SessionExecutor.execute +{_se_ms()}] code execution done",
                    )
                finally:
                    try:
                        await sb.close()
                    except Exception:
                        pass
                return {
                    **res,
                    "language": language,
                    "state_mode": state_mode,
                    "session_id": None,
                    "venv_id": venv_id,
                    "session_created": False,
                    "duration_ms": int(
                        (datetime.now(timezone.utc).timestamp() - t0) * 1000,
                    ),
                }

            # If a venv_id is provided, use persistent subprocess sessions.
            if venv_id is not None:
                implementation = _wrap_code_as_async_function(code)
                if state_mode == "stateless":
                    if self._function_manager is None:
                        raise RuntimeError(
                            "function_manager is required for python venv execution",
                        )
                    out = await self._function_manager.execute_in_venv(
                        venv_id=int(venv_id),
                        implementation=implementation,
                        call_kwargs={},
                        is_async=True,
                        primitives=primitives,
                        computer_primitives=computer_primitives,
                        env_overlay=_runtime_oauth_env_overlay(),
                    )
                    return {
                        **out,
                        "language": language,
                        "state_mode": state_mode,
                        "session_id": None,
                        "venv_id": venv_id,
                        "session_created": False,
                        "duration_ms": int(
                            (datetime.now(timezone.utc).timestamp() - t0) * 1000,
                        ),
                    }

                if session_id is None:
                    raise ValueError(
                        "session_id is required for venv-backed python execution",
                    )
                # Determine whether this is a new persistent session.
                existed_before = (int(venv_id), int(session_id)) in set(
                    self._venv_pool.list_active_sessions(),
                )
                if state_mode == "stateful":
                    # Persistent venv workers keep their process environment
                    # across calls.  Pass the OAuth overlay so SDK/default-env
                    # credential paths see fresh access tokens without the actor
                    # manually exporting anything.
                    out = await self._venv_pool.execute_in_venv(
                        venv_id=int(venv_id),
                        implementation=implementation,
                        call_kwargs={},
                        is_async=True,
                        session_id=int(session_id),
                        primitives=primitives,
                        computer_primitives=computer_primitives,
                        function_manager=self._function_manager,
                        timeout=self._timeout,
                        env_overlay=_runtime_oauth_env_overlay(),
                    )
                    return {
                        **out,
                        "language": language,
                        "state_mode": state_mode,
                        "session_id": session_id,
                        "venv_id": venv_id,
                        "session_created": not existed_before,
                        "duration_ms": int(
                            (datetime.now(timezone.utc).timestamp() - t0) * 1000,
                        ),
                    }

                if state_mode == "read_only":
                    # Snapshot state from persistent session, then run in one-shot subprocess.
                    if self._function_manager is None:
                        raise RuntimeError(
                            "function_manager is required for venv read_only execution",
                        )
                    initial_state = await self._venv_pool.get_connection_state(
                        venv_id=int(venv_id),
                        function_manager=self._function_manager,
                        session_id=int(session_id),
                        timeout=10.0,
                    )
                    # Read-only venv execution runs in a one-shot subprocess
                    # seeded from persistent state, but still receives the same
                    # runtime OAuth overlay before code executes.
                    out = await self._function_manager.execute_in_venv(
                        venv_id=int(venv_id),
                        implementation=implementation,
                        call_kwargs={},
                        is_async=True,
                        initial_state=initial_state,
                        primitives=primitives,
                        computer_primitives=computer_primitives,
                        env_overlay=_runtime_oauth_env_overlay(),
                    )
                    return {
                        **out,
                        "language": language,
                        "state_mode": state_mode,
                        "session_id": session_id,
                        "venv_id": venv_id,
                        "session_created": False,
                        "duration_ms": int(
                            (datetime.now(timezone.utc).timestamp() - t0) * 1000,
                        ),
                    }

                raise ValueError(
                    f"Unsupported state_mode for python venv: {state_mode}",
                )

            # In-process persistent sessions (venv_id is None).
            if session_id is None:
                raise ValueError(
                    "session_id is required for in-process python stateful/read_only execution",
                )

            key = (venv_id, int(session_id))
            if state_mode == "stateful":
                created = False
                if key not in self._python_sessions:
                    self._python_sessions[key] = PythonExecutionSession(
                        computer_primitives=computer_primitives,
                        environments=self._environments,
                        venv_pool=self._venv_pool,
                        shell_pool=self._shell_pool,
                    )
                    created = True
                    now = datetime.now(timezone.utc).isoformat()
                    self._python_session_meta[key] = {
                        "created_at": now,
                        "last_used": now,
                    }
                sb = self._python_sessions[key]
                self._inject_fm_globals(sb)
                res = await _execute_in_python_session(sb)
                meta = self._python_session_meta.get(key)
                if meta is not None:
                    meta["last_used"] = datetime.now(timezone.utc).isoformat()
                return {
                    **res,
                    "language": language,
                    "state_mode": state_mode,
                    "session_id": session_id,
                    "venv_id": venv_id,
                    "session_created": created,
                    "duration_ms": int(
                        (datetime.now(timezone.utc).timestamp() - t0) * 1000,
                    ),
                }

            if state_mode == "read_only":
                # Create a throwaway sandbox seeded with current state.
                if key not in self._python_sessions:
                    raise ValueError(
                        f"Python session {key} not found for read_only execution",
                    )
                base = self._python_sessions[key]
                sb = PythonExecutionSession(
                    computer_primitives=computer_primitives,
                    environments=self._environments,
                    venv_pool=self._venv_pool,
                    shell_pool=self._shell_pool,
                )
                try:
                    # Shallow copy globals to allow read access while avoiding persistence.
                    sb.global_state.update(dict(base.global_state))
                    self._inject_fm_globals(sb)
                    res = await _execute_in_python_session(sb)
                finally:
                    try:
                        await sb.close()
                    except Exception:
                        pass
                return {
                    **res,
                    "language": language,
                    "state_mode": state_mode,
                    "session_id": session_id,
                    "venv_id": venv_id,
                    "session_created": False,
                    "duration_ms": int(
                        (datetime.now(timezone.utc).timestamp() - t0) * 1000,
                    ),
                }

            raise ValueError(
                f"Unsupported state_mode for python in-process: {state_mode}",
            )

        # ─── Shell ─────────────────────────────────────────────────────────
        # Stateless: ephemeral subprocess (no pool/session).
        if state_mode == "stateless":
            out = await _execute_shell_stateless(language=language, command=code)
            return {
                **out,
                "language": language,
                "state_mode": state_mode,
                "session_id": None,
                "venv_id": None,
                "session_created": False,
                "duration_ms": int(
                    (datetime.now(timezone.utc).timestamp() - t0) * 1000,
                ),
            }

        if session_id is None:
            raise ValueError(
                "session_id is required for shell stateful/read_only execution",
            )

        # Persistent shell session.
        if state_mode == "stateful":
            existed_before = self._shell_pool.has_session(
                language=language,  # type: ignore[arg-type]
                session_id=int(session_id),
            )
            # Shells are especially prone to stale env because exports persist
            # inside the session.  We both pass an env overlay to the pool and
            # prepend explicit assignments to the command so the current command
            # and future commands in the same shell agree on the refreshed token
            # values.
            res = await self._shell_pool.execute(
                language=language,  # type: ignore[arg-type]
                command=_with_shell_env_overlay(
                    code,
                    _runtime_oauth_env_overlay(),
                    language=str(language),
                ),
                session_id=int(session_id),
                timeout=self._timeout,
                env=_runtime_oauth_env_overlay(),
            )
            return {
                "stdout": res.stdout,
                "stderr": res.stderr,
                "result": res.exit_code,
                "error": res.error,
                "language": language,
                "state_mode": state_mode,
                "session_id": session_id,
                "venv_id": None,
                "session_created": not existed_before,
                "duration_ms": int(
                    (datetime.now(timezone.utc).timestamp() - t0) * 1000,
                ),
            }

        if state_mode == "read_only":
            # Snapshot persistent state, restore into ephemeral session, execute, then discard.
            from unify.function_manager.shell_session import ShellSession

            sess = await self._shell_pool.get_session(
                language=language,  # type: ignore[arg-type]
                session_id=int(session_id),
            )
            snap = await sess.snapshot_state()
            tmp = ShellSession(language=language)  # type: ignore[arg-type]
            await tmp.start()
            try:
                restore_res = await tmp.restore_state(snap)
                if restore_res.error:
                    return {
                        "stdout": restore_res.stdout,
                        "stderr": restore_res.stderr,
                        "result": restore_res.exit_code,
                        "error": restore_res.error,
                        "language": language,
                        "state_mode": state_mode,
                        "session_id": session_id,
                        "venv_id": None,
                        "session_created": False,
                        "duration_ms": int(
                            (datetime.now(timezone.utc).timestamp() - t0) * 1000,
                        ),
                    }
                res = await tmp.execute(code, timeout=self._timeout)
                return {
                    "stdout": res.stdout,
                    "stderr": res.stderr,
                    "result": res.exit_code,
                    "error": res.error,
                    "language": language,
                    "state_mode": state_mode,
                    "session_id": session_id,
                    "venv_id": None,
                    "session_created": False,
                    "duration_ms": int(
                        (datetime.now(timezone.utc).timestamp() - t0) * 1000,
                    ),
                }
            finally:
                try:
                    await tmp.close()
                except Exception:
                    pass

        raise ValueError(f"Unsupported state_mode for shell: {state_mode}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _wrap_code_as_async_function(code: str) -> str:
    """
    Wrap an arbitrary code snippet into a single async function definition.

    This is required for the venv runner protocol, which expects exactly one function
    definition in the provided source.
    """
    # Ensure non-empty body.
    body = code if code.strip() else "pass"
    indented = "\n".join(
        ("    " + line) if line.strip() else "    " for line in body.splitlines()
    )
    return "async def __unity_code_act__():\n" + indented + "\n"


async def _execute_shell_stateless(
    *,
    language: SupportedLanguage,
    command: str,
) -> Dict[str, Any]:
    """
    Execute shell code in an ephemeral subprocess (stateless).
    """
    if language == "python":
        raise ValueError("Shell stateless executor called with language='python'")

    # Build command line consistent with ShellSession's non-interactive choices.
    if language == "bash":
        argv = ["/bin/bash", "--norc", "--noprofile", "-c", command]
    elif language == "zsh":
        argv = ["/bin/zsh", "--no-rcs", "--no-globalrcs", "-c", command]
    elif language == "sh":
        argv = ["/bin/sh", "-c", command]
    elif language == "powershell":
        argv = ["pwsh", "-NoProfile", "-NoLogo", "-Command", command]
    else:
        raise ValueError(f"Unsupported shell language: {language}")

    try:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_b, stderr_b = await proc.communicate()
        return {
            "stdout": (stdout_b or b"").decode(errors="replace"),
            "stderr": (stderr_b or b"").decode(errors="replace"),
            "result": int(proc.returncode or 0),
            "error": None,
        }
    except Exception as e:
        return {
            "stdout": "",
            "stderr": "",
            "result": None,
            "error": f"{type(e).__name__}: {e}",
        }
