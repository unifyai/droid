from __future__ import annotations

import asyncio
import functools
import logging
import os
from threading import Lock
from time import monotonic
from typing import Any, Callable, Dict, List, Optional, Type
from pydantic import BaseModel

import unisdk
from unify.common.llm_client import new_llm_client
from unify.common.log_utils import log as unity_log

logger = logging.getLogger(__name__)
from ..common.llm_helpers import methods_to_tool_dict
from ..common.tool_spec import ToolSpec
from ..common.async_tool_loop import (
    start_async_tool_loop,
    SteerableToolHandle,
    TOOL_LOOP_LINEAGE,
)
from ..settings import SETTINGS
from ..common.read_only_ask_guard import ReadOnlyAskGuardHandle
from ..events.event_bus import EVENT_BUS, Event
from ..events.manager_event_logging import log_manager_call
from ..common.tool_outcome import ToolError, ToolErrorException, ToolOutcome
from ..common.embed_utils import ensure_vector_column
from ..common.context_store import TableStore
from ..common.model_to_fields import model_to_fields
from .types import Secret
from .base import BaseSecretManager
from .prompt_builders import build_ask_prompt, build_update_prompt
from ..common.filter_utils import normalize_filter_expr
from ..common.search_utils import is_plain_identifier
from ..common.federated_search import (
    SOURCE_FIELD,
    FederatedSearchContext,
    federated_filter,
    federated_ranked_search,
)
from ..common.context_registry import (
    ContextRegistry,
    PERSONAL_DESTINATION,
    TEAM_CONTEXT_PREFIX,
    TableContext,
)

SECRETS_TABLE = "Secrets"  # pragma: allowlist secret
DESTINATION_FILTER_PATTERN = (
    r"destination\s*==\s*(?P<quote>['\"])(?P<destination>[^'\"]+)(?P=quote)"
)


class SecretManager(BaseSecretManager):
    """
    Manage personal and shared credentials without exposing raw values to LLMs.

    LLM-facing reads merge the assistant's personal vault with every reachable
    shared team vault. Writes accept a destination and persist to exactly one
    vault. Runtime credential lookups also read exactly one vault; a missing
    credential in the requested destination raises instead of falling back to
    another scope.
    """

    class Config:
        required_contexts = [
            TableContext(
                name="Secrets",
                description="Key-value secrets with descriptions and embeddings.",
                fields=model_to_fields(Secret),
                unique_keys={"secret_id": "int", "name": "str"},
                auto_counting={"secret_id": None},
            ),
        ]

    def __init__(self) -> None:
        super().__init__()
        self._ctx = ContextRegistry.get_context(self, SECRETS_TABLE)
        self._assistant_secret_sync_lock = Lock()
        self._last_assistant_secret_sync_success_at: float | None = None
        self._last_assistant_secret_sync_failure_at: float | None = None
        # Raw provider access/refresh tokens live here only -- never in the
        # Secrets context, .env, or os.environ -- so sandboxed actor code cannot
        # read them and bypass the workspace file-access allowlist. The trusted
        # provider proxy reads them via ``get_oauth_token``.
        self._oauth_tokens: dict[str, str] = {}

        # Ensure storage/schema exists deterministically (idempotent)
        self._provision_storage()

        # Public tools
        ask_tools: Dict[str, Callable] = {
            **methods_to_tool_dict(
                ToolSpec(
                    fn=self._list_columns,
                    display_label="Listing credential fields",
                ),
                ToolSpec(
                    fn=self._filter_secrets,
                    display_label="Filtering credentials",
                ),
                ToolSpec(
                    fn=self._search_secrets,
                    display_label="Searching credentials",
                ),
                ToolSpec(
                    fn=self._list_secret_keys,
                    display_label="Listing credential names",
                ),
                include_class_name=False,
            ),
        }
        self.add_tools("ask", ask_tools)
        update_tools: Dict[str, Callable] = {
            **methods_to_tool_dict(
                ToolSpec(fn=self.ask, display_label="Querying credentials"),
                ToolSpec(
                    fn=self._create_secret,
                    display_label="Storing a new credential",
                ),
                ToolSpec(fn=self._update_secret, display_label="Updating a credential"),
                ToolSpec(fn=self._delete_secret, display_label="Deleting a credential"),
                include_class_name=False,
            ),
        }
        self.add_tools("update", update_tools)

        # .env sync: create file if missing and backfill existing secrets as KEY=VALUE
        try:
            self._ensure_dotenv_synced_on_init()
        except Exception:
            # Best-effort – local file sync must never break construction
            pass

    # --------------------- Storage provisioning helper --------------------- #
    def _provision_storage(self) -> None:
        """Ensure Secrets context and schema exist and required vectors are present."""
        self._store = TableStore(
            self._ctx,
            unique_keys={"secret_id": "int", "name": "str"},
            auto_counting={"secret_id": None},
            description="Key-value secrets with descriptions and embeddings.",
            fields=model_to_fields(Secret),
        )

    def warm_embeddings(self) -> None:
        for context in self._read_secret_contexts():
            self._ensure_description_vector(context)

    def _secret_context_for_root(self, root_context: str) -> str:
        """Return the concrete Secrets context under a registry root."""
        return f"{root_context.strip('/')}/{SECRETS_TABLE}"

    def _effective_destination(self, destination: str | None) -> str | None:
        """Resolve implicit task-scoped credential routing."""
        raw_destination = destination or os.environ.get("TASK_DESTINATION")
        return ContextRegistry.canonical_destination(raw_destination)

    def _secret_context_for_destination(self, destination: str | None) -> str:
        """Resolve a public destination into one concrete Secrets context."""
        destination = self._effective_destination(destination)
        root_context = ContextRegistry.write_root(
            self,
            SECRETS_TABLE,
            destination=destination,
        )
        return self._secret_context_for_root(root_context)

    def _destination_for_context(self, context: str) -> str:
        """Return the public destination label for a concrete Secrets context."""
        if context.startswith(TEAM_CONTEXT_PREFIX):
            parts = context.split("/")
            if len(parts) >= 2:
                return f"team:{parts[1]}"
        return PERSONAL_DESTINATION

    def _split_destination_filter(
        self,
        filter_expr: str | None,
    ) -> tuple[str | None, set[str]]:
        """Extract destination equality predicates from a public filter."""
        if not filter_expr:
            return filter_expr, set()

        import re

        destinations = {
            match.group("destination")
            for match in re.finditer(DESTINATION_FILTER_PATTERN, filter_expr)
        }
        if not destinations:
            return filter_expr, set()

        cleaned = re.sub(
            rf"\s+and\s+{DESTINATION_FILTER_PATTERN}",
            "",
            filter_expr,
        )
        cleaned = re.sub(
            rf"{DESTINATION_FILTER_PATTERN}\s+and\s+",
            "",
            cleaned,
        )
        cleaned = re.sub(DESTINATION_FILTER_PATTERN, "", cleaned).strip()
        return cleaned or None, destinations

    def _read_secret_contexts(self) -> list[str]:
        """Return personal-first concrete Secrets contexts visible to the assistant."""
        return list(
            dict.fromkeys(
                self._secret_context_for_root(root)
                for root in ContextRegistry.read_roots(self, SECRETS_TABLE)
            ),
        )

    def _is_personal_context(self, context: str) -> bool:
        """Return whether a concrete Secrets context is the personal vault."""
        return context == self._ctx

    @functools.cache
    def _ensure_description_vector(self, context: str) -> None:
        """Ensure the description embedding exists for one Secrets context."""
        try:
            ensure_vector_column(
                context,
                embed_column="description_emb",
                source_column="description",
                derived_expr=None,
            )
        except Exception:
            pass

    @functools.wraps(BaseSecretManager.clear, updated=())
    def clear(self) -> None:
        unisdk.delete_context(self._ctx)

        # Force re-provisioning even if previously ensured
        self._ctx = ContextRegistry.refresh(self, SECRETS_TABLE)

        # Re-create schema and vectors
        self._provision_storage()

        # Verify the context is visible before attempting reads
        try:
            import time as _time  # local import

            for _ in range(3):
                try:
                    unisdk.get_fields(context=self._ctx)
                    break
                except Exception:
                    _time.sleep(0.05)
        except Exception:
            pass

    # --------------------- Internal helpers (LLM client/policies) --------------------- #

    @staticmethod
    def _default_ask_tool_policy(
        step_index: int,
        current_tools: Dict[str, Any],
    ) -> tuple[str, Dict[str, Any]]:
        """Default ask-side tool policy (no-op, retain current tools)."""
        return ("auto", current_tools)

    @staticmethod
    def _default_update_tool_policy(
        step_index: int,
        current_tools: Dict[str, Any],
    ) -> tuple[str, Dict[str, Any]]:
        """Require 'ask' on the first step (if enabled); auto thereafter."""
        from unify.settings import SETTINGS

        if (
            SETTINGS.FIRST_MUTATION_TOOL_IS_ASK
            and step_index < 1
            and "ask" in current_tools
        ):
            return ("required", {"ask": current_tools["ask"]})
        return ("auto", current_tools)

    # --------------------- Internal helpers (assistant secret sync) ---------- #

    # Allowlist for ``_sync_assistant_secrets``.  Limited to OAuth tokens that
    # Communication writes to Orchestra's ``AssistantSecret`` table after each
    # Google / Microsoft OAuth callback — those are the only secrets THIS sync
    # is responsible for transporting.  Console-pasted integration secrets
    # (HubSpot, Employment Hero, Matterport, Webex, Salesforce …) live in the
    # ``/Secrets`` context directly and reach ``os.environ`` via
    # ``_sync_dotenv``.  They neither need nor go through this sync; mixing
    # them in here causes the cleanup loop to wipe them, which is the bug
    # ``61141bba2`` patched.  Concern separation enforced explicitly: keep
    # this allowlist OAuth-only so the bug class can't reappear.
    _BUILTIN_OAUTH_SECRET_ALLOWLIST = frozenset(
        {
            "GOOGLE_ACCESS_TOKEN",
            "GOOGLE_REFRESH_TOKEN",
            "GOOGLE_TOKEN_EXPIRES_AT",
            "GOOGLE_GRANTED_SCOPES",
            "MICROSOFT_ACCESS_TOKEN",
            "MICROSOFT_REFRESH_TOKEN",
            "MICROSOFT_TOKEN_EXPIRES_AT",
            "MICROSOFT_GRANTED_SCOPES",
        },
    )

    # Backwards-compatible alias for the (small number of) call sites and
    # tests that read ``OAUTH_SECRET_ALLOWLIST`` directly.  Identical to
    # the built-in set above.
    OAUTH_SECRET_ALLOWLIST = _BUILTIN_OAUTH_SECRET_ALLOWLIST

    @classmethod
    def _resolve_secret_allowlist(cls) -> frozenset[str]:
        """Return assistant-secret names owned by runtime OAuth sync.

        The set is intentionally limited to refresh-token OAuth metadata.
        Console-pasted integration credentials already live in the local
        ``Secrets`` context and reach ``os.environ`` through ``_sync_dotenv``.
        """
        try:
            from unify.common.runtime_oauth import refresh_token_oauth_secret_names

            return (
                cls._BUILTIN_OAUTH_SECRET_ALLOWLIST | refresh_token_oauth_secret_names()
            )
        except Exception:
            return cls._BUILTIN_OAUTH_SECRET_ALLOWLIST

    @classmethod
    def _sensitive_oauth_token_names(cls) -> frozenset[str]:
        """Raw access/refresh token names held in-memory only (never in env)."""
        try:
            from unify.common.runtime_oauth import refresh_token_oauth_token_names

            return refresh_token_oauth_token_names()
        except Exception:
            return frozenset(
                {
                    "GOOGLE_ACCESS_TOKEN",
                    "GOOGLE_REFRESH_TOKEN",
                    "MICROSOFT_ACCESS_TOKEN",
                    "MICROSOFT_REFRESH_TOKEN",
                },
            )

    def get_oauth_token(self, name: str) -> str | None:
        """Return a raw provider token from the in-memory OAuth store.

        Trusted-runtime accessor used by :func:`runtime_oauth.get_provider_access_token`.
        Never expose the returned value to sandboxed code.
        """
        return self._oauth_tokens.get(name)

    def _sync_assistant_secrets(self) -> None:
        """Mirror runtime OAuth assistant secrets from Orchestra into local state.

        Orchestra is the platform source of truth for assistant-level OAuth
        secrets written outside this Unity process. Communication refresh jobs
        persist updated access tokens there; this method pulls those values into
        Unity's local ``Secrets`` context, then updates ``.env``/``os.environ``
        so generated code and provider SDKs can use normal environment-based
        credential discovery.

        The sync is intentionally allowlisted. We mirror refresh-token OAuth
        keys, but we do not copy arbitrary assistant secrets into the runtime.
        Failures are best-effort: callers use ``sync_assistant_secrets_if_stale``
        as the observable gate.
        """
        from ..session_details import SESSION_DETAILS

        agent_id = SESSION_DETAILS.assistant.agent_id
        if agent_id is None:
            return

        base_url = SETTINGS.ORCHESTRA_URL
        admin_key = SETTINGS.ORCHESTRA_ADMIN_KEY.get_secret_value()
        if not base_url or not admin_key:
            return

        try:
            from unisdk.utils import http

            resp = http.get(
                f"{base_url}/admin/assistant",
                headers={"Authorization": f"Bearer {admin_key}"},
                params={"agent_id": int(agent_id), "from_fields": "secrets"},
                timeout=15,
            )
            if resp.status_code != 200:
                return
            payload = resp.json()
            items = payload.get("info", [])
            if not items:
                return
            secrets_dict: dict = items[0].get("secrets") or {}
        except Exception:
            return

        # Allowlist is intentionally OAuth-only; integration secrets do not flow
        # through this sync because they already live in the local Secrets
        # context and are exported by _sync_dotenv.
        active_allowlist = self._resolve_secret_allowlist()
        sensitive = self._sensitive_oauth_token_names()

        written = 0
        for name, value in secrets_dict.items():
            if name not in active_allowlist:
                continue
            if not isinstance(value, str) or not value:
                continue
            # Raw access/refresh tokens are held in-memory only, never mirrored
            # to the Secrets context / .env / os.environ, so the sandbox cannot
            # read them. Non-sensitive OAuth metadata (expiry, granted scopes)
            # still flows to the context/env for scope and freshness checks.
            if name in sensitive:
                self._oauth_tokens[name] = value
                written += 1
                continue
            try:
                existing = unisdk.get_logs(
                    context=self._ctx,
                    filter=f"name == {name!r}",
                    limit=1,
                    return_ids_only=True,
                )
                description = "System-managed OAuth credential (auto-synced)"
                if existing:
                    unisdk.update_logs(
                        logs=[existing[0]],
                        context=self._ctx,
                        entries={"value": value, "description": description},
                        overwrite=True,
                    )
                else:
                    unity_log(
                        context=self._ctx,
                        name=name,
                        value=value,
                        description=description,
                        new=True,
                        mutable=True,
                        stamp_authoring=True,
                    )
                self._env_set(name, value)
                written += 1
            except Exception:
                continue

        logger.info(
            "[integrations] sync: agent_id=%s orchestra_keys=%d wrote=%d",
            agent_id,
            len(secrets_dict),
            written,
        )

        # Stale-cleanup is limited to the OAuth secrets owned by this sync.
        # Console-pasted integration credentials live in the same local Secrets
        # context but are not removed based on the admin assistant payload.
        for stale_name in active_allowlist - secrets_dict.keys():
            if stale_name in sensitive:
                self._oauth_tokens.pop(stale_name, None)
                continue
            try:
                ids = unisdk.get_logs(
                    context=self._ctx,
                    filter=f"name == {stale_name!r}",
                    limit=1,
                    return_ids_only=True,
                )
                if ids:
                    unisdk.delete_logs(context=self._ctx, logs=ids[0])
                    self._env_remove(stale_name)
            except Exception:
                continue

    def _sync_workspace_file_policy(self) -> None:
        """Mirror the workspace file-access allowlist into the runtime policy store.

        Orchestra owns the per-assistant, per-provider Drive/SharePoint
        allowlist (configured in Console). The localhost provider proxy reads it
        from this in-process cache to enforce access on every Drive/Graph call;
        this keeps that cache current. Best-effort: failures leave the prior
        cache.
        """
        from ..session_details import SESSION_DETAILS

        agent_id = SESSION_DETAILS.assistant.agent_id
        if agent_id is None:
            return

        base_url = SETTINGS.ORCHESTRA_URL
        admin_key = SETTINGS.ORCHESTRA_ADMIN_KEY.get_secret_value()
        if not base_url or not admin_key:
            return

        from unisdk.utils import http

        from unify.provider_proxy.policy import get_policy_store

        resp = http.get(
            f"{base_url}/admin/assistant/{int(agent_id)}/workspace-file-access",
            headers={"Authorization": f"Bearer {admin_key}"},
            timeout=15,
        )
        if resp.status_code != 200:
            return
        info = resp.json().get("info") or {}
        get_policy_store().set_policies(info.get("policies") or [])

    def sync_assistant_secrets_if_stale(
        self,
        ttl_seconds: float = 60.0,
        *,
        force: bool = False,
        reason: str = "runtime",
        failure_cooldown_seconds: float = 10.0,
    ) -> bool:
        """Pull assistant secrets through one debounced runtime sync gate.

        This is the single runtime entry point for keeping Unity's local secret
        state close to Orchestra without adding a network round trip to every
        actor operation.  Normal callers, including ``execute_code``, call with
        ``force=False`` and therefore only perform the expensive Orchestra pull
        once per ``ttl_seconds`` window.  Forced callers use this when freshness
        matters more than debounce, such as SecretManager construction,
        ``primitives.secrets.ask(...)``, assistant-update events, or an OAuth
        helper detecting a missing/near-expiry access token.

        Returns ``True`` only when this invocation actually ran the sync work.
        Returns ``False`` when the success debounce or failure cooldown skipped
        work, or when the wrapped sync raised an exception.
        """
        now = monotonic()
        if not force:
            last_success = self._last_assistant_secret_sync_success_at
            if last_success is not None and now - last_success < ttl_seconds:
                return False
            last_failure = self._last_assistant_secret_sync_failure_at
            if (
                last_failure is not None
                and now - last_failure < failure_cooldown_seconds
            ):
                return False

        with self._assistant_secret_sync_lock:
            now = monotonic()
            if not force:
                last_success = self._last_assistant_secret_sync_success_at
                if last_success is not None and now - last_success < ttl_seconds:
                    return False
                last_failure = self._last_assistant_secret_sync_failure_at
                if (
                    last_failure is not None
                    and now - last_failure < failure_cooldown_seconds
                ):
                    return False
            try:
                self._sync_assistant_secrets()
                self._sync_dotenv()
                self._sync_workspace_file_policy()
            except Exception:
                self._last_assistant_secret_sync_failure_at = monotonic()
                logger.warning(
                    "[integrations] assistant secret sync failed reason=%s",
                    reason,
                    exc_info=True,
                )
                return False
            self._last_assistant_secret_sync_success_at = monotonic()
            self._last_assistant_secret_sync_failure_at = None
            logger.info(
                "[integrations] assistant secret sync complete reason=%s",
                reason,
            )
            return True

    def _get_secret_value(self, name: str) -> str | None:
        try:
            rows = unisdk.get_logs(
                context=self._ctx,
                filter=f"name == {name!r}",
                limit=1,
                from_fields=["name", "value"],
            )
            if rows:
                value = (rows[0].entries or {}).get("value")
                if isinstance(value, str) and value:
                    return value
        except Exception:
            pass
        value = os.environ.get(name)
        return value if value else None

    # --------------------- Internal helpers (.env sync) --------------------- #
    def _dotenv_path(self) -> str:
        """Return the path to the .env file used for local sync.

        Honors UNITY_SECRET_DOTENV_PATH from SETTINGS; defaults to ".env" in CWD.
        """
        import os as _os

        if SETTINGS.secret.DOTENV_PATH:
            return SETTINGS.secret.DOTENV_PATH
        from unify.file_manager.settings import get_local_root

        return _os.path.join(get_local_root(), ".env")

    def _sync_dotenv(self) -> None:
        """Fetch all secrets from the backend and merge into the local .env file.

        Ensures that secrets added externally (e.g. via the Console UI, which
        writes directly to Orchestra) are available as environment variables
        for code executed via ``os.environ``.
        """
        try:
            rows = unisdk.get_logs(context=self._ctx)
        except Exception:
            rows = []
        name_to_value: Dict[str, str] = {}
        for lg in rows:
            try:
                nm = (lg.entries or {}).get("name")
                val = (lg.entries or {}).get("value")
                if isinstance(nm, str) and nm and isinstance(val, str):
                    name_to_value[nm] = val
            except Exception:
                continue

        if name_to_value:
            self._env_merge_and_write(add_or_update=name_to_value, remove_keys=None)

    def _ensure_dotenv_synced_on_init(self) -> None:
        """Create .env if missing, pull assistant OAuth tokens, and merge all secrets."""
        path = self._dotenv_path()
        # Ensure directory exists
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        except Exception:
            pass
        # Ensure file exists
        if not os.path.exists(path):
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("")

        self.sync_assistant_secrets_if_stale(force=True, reason="secret_manager_init")

    @staticmethod
    def _parse_env_lines(lines: List[str]) -> Dict[str, int]:
        """Return mapping of existing KEY -> line index for a simple .env file."""
        import re

        key_to_idx: Dict[str, int] = {}
        for idx, raw in enumerate(lines):
            m = re.match(r"\s*([A-Za-z_][A-Za-z0-9_/]*)\s*=", raw)
            if m:
                key_to_idx[m.group(1)] = idx
        return key_to_idx

    def _env_merge_and_write(
        self,
        add_or_update: Dict[str, str] | None,
        remove_keys: List[str] | None,
    ) -> None:
        """Merge provided updates/removals into the .env file atomically."""
        path = self._dotenv_path()
        try:
            with open(path, "r", encoding="utf-8") as fh:
                lines = fh.read().splitlines()
        except FileNotFoundError:
            lines = []

        key_to_idx = self._parse_env_lines(lines)

        # Remove keys first
        if remove_keys:
            rm = set(remove_keys)

            def _keep(i: int, s: str) -> bool:
                for k, j in key_to_idx.items():
                    if j == i and k in rm:
                        return False
                return True

            lines = [s for i, s in enumerate(lines) if _keep(i, s)]
            key_to_idx = self._parse_env_lines(lines)

        # Add or update keys
        if add_or_update:
            for key, value in add_or_update.items():
                line = f"{key}={value}"
                if key in key_to_idx:
                    lines[key_to_idx[key]] = line
                else:
                    lines.append(line)
                os.environ[key] = value

        if remove_keys:
            for key in remove_keys:
                os.environ.pop(key, None)

        with open(path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + ("\n" if lines else ""))

    def _env_set(self, name: str, value: str) -> None:
        """Set or update one KEY=VALUE line in .env."""
        self._env_merge_and_write({name: value}, remove_keys=None)

    def _env_remove(self, name: str) -> None:
        """Remove one KEY from .env (if present)."""
        self._env_merge_and_write(add_or_update=None, remove_keys=[name])

    # --------------------- Public API --------------------- #
    async def from_placeholder(
        self,
        text: str,
        *,
        destination: str | None = None,
    ) -> str:
        """Resolve ${name} placeholders in text to raw secret values (no LLM).

        Parameters
        ----------
        text : str
            Input string that may contain placeholders like "${api_key}".
        destination : str | None, default None
            Credential vault to read from. Defaults to the active task
            destination when one is set, otherwise to the personal vault.

        Returns
        -------
        str
            String with placeholders substituted with their secret values.
        """
        return self._resolve_placeholders(text, destination=destination)

    async def to_placeholder(self, text: str) -> str:
        """Convert secret values in text to placeholders.

        Parameters
        ----------
        text : str
            The text to convert secret values to placeholders.

        Returns
        -------
        str
            The text with secret values converted to placeholders.
        """
        value_to_name: Dict[str, str] = {}
        for context in self._read_secret_contexts():
            try:
                rows = unisdk.get_logs(
                    context=context,
                    from_fields=["name", "value"],
                )
            except Exception:
                rows = []

            for lg in rows:
                try:
                    nm = (lg.entries or {}).get("name")
                    val = (lg.entries or {}).get("value")
                    if isinstance(nm, str) and nm and isinstance(val, str) and val:
                        if val in value_to_name:
                            if nm < value_to_name[val]:
                                value_to_name[val] = nm
                        else:
                            value_to_name[val] = nm
                except Exception:
                    continue

        # Replace longer values first to avoid partial overlaps
        import re

        ordered_values = sorted(value_to_name.keys(), key=len, reverse=True)
        result = text
        for val in ordered_values:
            name = value_to_name[val]
            pattern = re.escape(val)
            placeholder = f"${{{name}}}"
            result = re.sub(pattern, placeholder, result)

        return result

    @functools.wraps(BaseSecretManager.ask, updated=())
    @log_manager_call(
        "SecretManager",
        "ask",
        payload_key="question",
        display_label="Checking credentials",
    )
    async def ask(
        self,
        text: str,
        *,
        response_format: Optional[Type[BaseModel]] = None,
        _return_reasoning_steps: bool = False,
        _parent_chat_context: Optional[List[Dict[str, Any]]] = None,
        _clarification_up_q: Optional[asyncio.Queue[str]] = None,
        _clarification_down_q: Optional[asyncio.Queue[str]] = None,
        _call_id: Optional[str] = None,
    ) -> SteerableToolHandle:
        self.sync_assistant_secrets_if_stale(force=True, reason="secret_ask")

        # First, replace any known raw secret values with placeholders
        try:
            text = await self.to_placeholder(text)
        except Exception:
            pass

        client = new_llm_client()

        # Build tools for read-only inspection
        tools = dict(self.get_tools("ask"))
        _clar_queues = None
        _on_clar_req = None
        _on_clar_ans = None
        if _clarification_up_q is not None and _clarification_down_q is not None:
            from ..common.llm_helpers import make_request_clarification_tool

            _clar_queues = (_clarification_up_q, _clarification_down_q)
            tools["request_clarification"] = make_request_clarification_tool(None, None)

            async def _on_clar_req(q: str):
                try:
                    await EVENT_BUS.publish(
                        Event(
                            type="ManagerMethod",
                            calling_id=_call_id,
                            payload={
                                "manager": "SecretManager",
                                "method": "ask",
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
                                "manager": "SecretManager",
                                "method": "ask",
                                "action": "clarification_answer",
                                "answer": ans,
                            },
                        ),
                    )
                except Exception:
                    pass

        # System message via prompt builder
        client.set_system_message(
            build_ask_prompt(tools=tools).to_list(),
        )

        handle = start_async_tool_loop(
            client,
            text,
            tools,
            loop_id=f"{self.__class__.__name__}.{self.ask.__name__}",
            parent_lineage=TOOL_LOOP_LINEAGE.get([]),
            parent_chat_context=_parent_chat_context,
            tool_policy=self._default_ask_tool_policy,
            response_format=response_format,
            handle_cls=(
                ReadOnlyAskGuardHandle if SETTINGS.UNITY_READONLY_ASK_GUARD else None
            ),
            clarification_queues=_clar_queues,
            on_clarification_request=_on_clar_req,
            on_clarification_answer=_on_clar_ans,
        )

        if _return_reasoning_steps:
            original_result = handle.result

            async def wrapped_result():
                answer = await original_result()
                return answer, client.messages

            handle.result = wrapped_result  # type: ignore

        return handle

    @functools.wraps(BaseSecretManager.update, updated=())
    @log_manager_call(
        "SecretManager",
        "update",
        payload_key="request",
        display_label="Updating credentials",
    )
    async def update(
        self,
        text: str,
        *,
        response_format: Optional[Type[BaseModel]] = None,
        _return_reasoning_steps: bool = False,
        _parent_chat_context: Optional[List[Dict[str, Any]]] = None,
        _clarification_up_q: Optional[asyncio.Queue[str]] = None,
        _clarification_down_q: Optional[asyncio.Queue[str]] = None,
        _call_id: Optional[str] = None,
    ) -> SteerableToolHandle:
        # First, replace any known raw secret values with placeholders
        try:
            text = await self.to_placeholder(text)
        except Exception:
            pass

        client = new_llm_client()

        tools = dict(self.get_tools("update"))
        _clar_queues = None
        _on_clar_req = None
        _on_clar_ans = None
        if _clarification_up_q is not None and _clarification_down_q is not None:
            from ..common.llm_helpers import make_request_clarification_tool

            _clar_queues = (_clarification_up_q, _clarification_down_q)
            tools["request_clarification"] = make_request_clarification_tool(None, None)

            async def _on_clar_req(q: str):
                try:
                    await EVENT_BUS.publish(
                        Event(
                            type="ManagerMethod",
                            calling_id=_call_id,
                            payload={
                                "manager": "SecretManager",
                                "method": "update",
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
                                "manager": "SecretManager",
                                "method": "update",
                                "action": "clarification_answer",
                                "answer": ans,
                            },
                        ),
                    )
                except Exception:
                    pass

        client.set_system_message(
            build_update_prompt(tools=tools).to_list(),
        )

        handle = start_async_tool_loop(
            client,
            text,
            tools,
            loop_id=f"{self.__class__.__name__}.{self.update.__name__}",
            parent_lineage=TOOL_LOOP_LINEAGE.get([]),
            parent_chat_context=_parent_chat_context,
            tool_policy=self._default_update_tool_policy,
            response_format=response_format,
            clarification_queues=_clar_queues,
            on_clarification_request=_on_clar_req,
            on_clarification_answer=_on_clar_ans,
        )

        if _return_reasoning_steps:
            original_result = handle.result

            async def wrapped_result():
                answer = await original_result()
                return answer, client.messages

            handle.result = wrapped_result  # type: ignore

        return handle

    # --------------------- Tools (read-only) --------------------- #
    def get_credential(
        self,
        integration: str,
        *,
        destination: str | None = None,
    ) -> str:
        """Return one raw credential from the vault named by destination.

        Parameters
        ----------
        integration : str
            Secret name to resolve.
        destination : str | None, default None
            ``"personal"`` or ``"team:<id>"``. ``None`` inherits the active
            task destination when one is set, otherwise personal.

        Raises
        ------
        KeyError
            If the credential is not stored in the resolved vault.
        """
        context = self._secret_context_for_destination(destination)
        rows = unisdk.get_logs(
            context=context,
            filter=f"name == {integration!r}",
            limit=1,
        )
        if not rows:
            resolved_destination = (
                self._effective_destination(destination) or PERSONAL_DESTINATION
            )
            raise KeyError(
                f"No credential named {integration!r} in {resolved_destination!r}.",
            )
        value = (rows[0].entries or {}).get("value")
        if not isinstance(value, str):
            resolved_destination = (
                self._effective_destination(destination) or PERSONAL_DESTINATION
            )
            raise KeyError(
                f"No credential named {integration!r} in {resolved_destination!r}.",
            )
        return value

    def _resolve_placeholders(
        self,
        text: str,
        *,
        destination: str | None = None,
    ) -> str:
        """Return a copy of text with ${name} placeholders replaced by values.

        This helper performs direct Unify reads and never emits logs/events.
        Unknown names are left unchanged.
        """
        import re

        def repl(match: "re.Match[str]") -> str:
            name = match.group(1)
            try:
                return self.get_credential(name, destination=destination)
            except KeyError:
                pass
            return match.group(0)  # leave placeholder as-is when missing

        return re.sub(r"\$\{([^}]+)\}", repl, text)

    def _list_columns(
        self,
        *,
        include_types: bool = True,
    ) -> Dict[str, Any] | List[str]:
        """Return available columns for the secrets table.

        Parameters
        ----------
        include_types : bool, default True
            When True, returns a mapping ``{column_name: column_type}``.
            When False, returns a list of column names only.

        Returns
        -------
        Dict[str, Any] | List[str]
            Column map when ``include_types=True``; otherwise a list of names.
        """
        cols = self._store.get_columns()
        return cols if include_types else list(cols)

    def _sanitize_secret_references(
        self,
        references: Optional[Dict[str, str]],
    ) -> Optional[Dict[str, str]]:
        """Return a safe subset of references limited to description-based terms.

        Only allows:
        - Plain identifier "description"; or
        - Derived expressions whose placeholders are exclusively {description}.
        Any other term is dropped to avoid embedding sensitive columns like "value".
        """
        if not references:
            return references

        allowed: Dict[str, str] = {}
        for source_expr, ref_text in references.items():
            try:
                if is_plain_identifier(source_expr):
                    if source_expr == "description":
                        allowed[source_expr] = ref_text
                    continue

                # Derived expression – verify placeholders are only {description}
                import re as _re

                placeholders = _re.findall(
                    r"\{\s*([a-zA-Z_][\w]*)\s*\}",
                    source_expr or "",
                )
                if placeholders and all(ph == "description" for ph in placeholders):
                    allowed[source_expr] = ref_text
            except Exception:
                # Skip malformed expressions defensively
                continue

        return allowed or None

    def _search_secrets(
        self,
        *,
        references: Optional[Dict[str, str]] = None,
        k: int = 10,
    ) -> List[Secret]:
        """Semantic search over secrets using the description embedding.

        Parameters
        ----------
        references : Dict[str, str] | None, default None
            Mapping of source expressions to reference text. For this manager
            use a column name like ``"description"`` to search over secret descriptions.
            When None or empty, returns most-recent rows.
        k : int, default 10
            Maximum number of results to return. Must be <= 1000.

        Returns
        -------
        List[Secret]
            Up to ``k`` redacted Secret models (``value`` is never populated).
        """
        # Sanitize references to avoid embedding sensitive fields like "value"
        safe_refs = self._sanitize_secret_references(references)

        contexts: list[FederatedSearchContext] = []
        for context in self._read_secret_contexts():
            self._ensure_description_vector(context)
            contexts.append(
                FederatedSearchContext(
                    context=context,
                    source=self._destination_for_context(context),
                    # Never return the secret value.
                    allowed_fields=["secret_id", "name", "description"],
                ),
            )
        rows = federated_ranked_search(
            contexts,
            safe_refs,
            limit=k,
            backfill=True,
        )
        return [
            Secret(
                secret_id=(
                    int(r.get("secret_id")) if r.get("secret_id") is not None else -1
                ),
                name=r.get("name"),
                value="",
                description=r.get("description") or "",
                destination=r.get(SOURCE_FIELD) or PERSONAL_DESTINATION,
            )
            for r in rows
        ]

    def _filter_secrets(
        self,
        *,
        filter: Optional[str] = None,
        offset: int = 0,
        limit: int = 100,
    ) -> List[Secret]:
        """Filter secrets using a boolean expression evaluated per row.

        Parameters
        ----------
        filter : str | None, default None
            A Python expression evaluated with column names in scope (e.g.,
            ``"name == 'unify_key'"``). When None, returns all rows.
        offset : int, default 0
            Zero-based index of the first result to include.
        limit : int, default 100
            Maximum number of rows to return. Must be <= 1000.

        Returns
        -------
        List[Secret]
            Matching Secret models with ``value`` redacted.
        """
        normalized = normalize_filter_expr(filter)
        normalized, destination_filter = self._split_destination_filter(normalized)
        contexts = []
        for context in self._read_secret_contexts():
            destination = self._destination_for_context(context)
            if destination_filter and destination not in destination_filter:
                continue
            contexts.append(
                FederatedSearchContext(
                    context=context,
                    source=destination,
                    # Never expose values in read tools.
                    allowed_fields=["secret_id", "name", "description"],
                ),
            )
        rows = federated_filter(
            contexts,
            filter=normalized,
            offset=offset,
            limit=limit,
        )
        return [
            Secret(
                secret_id=(
                    int(row.get("secret_id"))
                    if row.get("secret_id") is not None
                    else -1
                ),
                name=row.get("name"),
                value="",
                description=row.get("description") or "",
                destination=row.get(SOURCE_FIELD) or PERSONAL_DESTINATION,
            )
            for row in rows
        ]

    def _list_secret_keys(self) -> List[str]:
        """Return all available secret names (keys) stored in Unify.

        Returns
        -------
        List[str]
            Sorted, unique list of secret names currently present in storage.
        """
        names: set[str] = set()
        for context in self._read_secret_contexts():
            try:
                rows = unisdk.get_logs(
                    context=context,
                    from_fields=["name"],
                )
            except Exception:
                rows = []
            for lg in rows:
                nm = (lg.entries or {}).get("name")
                if isinstance(nm, str) and nm:
                    names.add(nm)
        return sorted(names)

    # --------------------- Tools (mutations) --------------------- #
    def _invalidate_credential_processes(self) -> None:
        """Drop stateful execution pools so subprocesses reopen with fresh env."""
        from unify.function_manager.function_manager import VenvPool

        VenvPool.invalidate_all_pools()

    def _create_secret(
        self,
        *,
        name: str,
        value: str,
        description: Optional[str] = None,
        destination: str | None = None,
    ) -> ToolOutcome | ToolError:
        """Create and persist a new secret.

        Parameters
        ----------
        name : str
            Unique identifier for the secret. Used as the placeholder name.
        value : str
            Raw secret value to store (never exposed to LLMs).
        description : str | None, default None
            Optional human-readable description.
        destination : str | None, default None
            Where the credential is stored. Pass ``"personal"`` (the default)
            for credentials only you should use: your own personal API key,
            your individual OAuth tokens, anything tied to your identity. Pass
            ``"team:<id>"`` for a team service account or shared credential
            that every member of the team should be able to use: the team
            Slack bot token, the shared SendGrid API key, the team's
            integration service account. Personal credentials never leak into
            team memory; team credentials never leak into your local ``.env``
            mirror. The set of available ``team:<id>`` values, each with a
            name and a description naming the team / domain the credential
            pool belongs to, is rendered in the *Accessible shared teams*
            block of your system prompt; read that block before choosing.
            The privacy floor: when in doubt between personal and a team,
            pick personal, because sharing a credential is harder to undo than
            re-sharing later. When confidence is low and the credential would
            land in a team, call ``request_clarification`` instead of
            guessing toward the wider audience.
            When running inside a task, omitting this argument inherits the
            task destination so task-owned credentials stay with the task's
            vault.

        Returns
        -------
        ToolOutcome | ToolError
            A standard outcome dict, or a structured tool error for invalid
            destinations.
        """
        assert name and value, "Both name and value are required."
        try:
            context = self._secret_context_for_destination(destination)
        except ToolErrorException as exc:
            return exc.payload

        # Enforce uniqueness of name
        existing = unisdk.get_logs(
            context=context,
            filter=f"name == {name!r}",
            limit=1,
            return_ids_only=True,
        )
        assert not existing, f"Secret with name '{name}' already exists."

        # Write secret (store raw value in backend, but never surface to LLM)
        entries = {
            "name": name,
            "value": value,
            "description": description or "",
        }
        unity_log(
            context=context,
            **entries,
            new=True,
            mutable=True,
            stamp_authoring=True,
        )

        try:
            if self._is_personal_context(context):
                self._env_set(name, value)
        except Exception:
            pass
        finally:
            self._invalidate_credential_processes()

        return {"outcome": "secret created", "details": {"name": name}}

    def _update_secret(
        self,
        *,
        name: str,
        value: Optional[str] = None,
        description: Optional[str] = None,
        destination: str | None = None,
    ) -> ToolOutcome | ToolError:
        """Update fields of an existing secret.

        Parameters
        ----------
        name : str
            Secret name to update.
        value : str | None, default None
            New raw value (optional). When provided it overwrites the existing value.
        description : str | None, default None
            New description (optional).
        destination : str | None, default None
            Which copy of the credential to update. Defaults to ``"personal"``
            (your private credential). Passing ``"team:<id>"`` rotates the
            shared credential in that team and is visible to every member;
            pooled subprocesses pick up the new value on next invocation. See
            the *Accessible shared teams* block in your system prompt for
            the available teams and their descriptions. Inside a task, an
            omitted destination inherits the task destination.

        Returns
        -------
        ToolOutcome | ToolError
            Outcome dict or structured tool error for invalid destinations.
        """
        try:
            context = self._secret_context_for_destination(destination)
        except ToolErrorException as exc:
            return exc.payload

        # Find target log id
        ids = unisdk.get_logs(
            context=context,
            filter=f"name == {name!r}",
            limit=2,
            return_ids_only=True,
        )
        if not ids:
            raise ValueError(f"No secret found with name '{name}'.")
        if len(ids) > 1:
            raise RuntimeError(f"Multiple secrets found with name '{name}'.")
        log_id = ids[0]

        updates: Dict[str, Any] = {}
        if description is not None:
            updates["description"] = description
        if value is not None:
            updates["value"] = value

        if not updates:
            raise ValueError("No updates provided.")

        unisdk.update_logs(
            logs=[log_id],
            context=context,
            entries=updates,
            overwrite=True,
        )

        try:
            if value is not None and self._is_personal_context(context):
                self._env_set(name, value)
        except Exception:
            pass
        finally:
            self._invalidate_credential_processes()

        return {"outcome": "secret updated", "details": {"name": name}}

    def _delete_secret(
        self,
        *,
        name: str,
        destination: str | None = None,
    ) -> ToolOutcome | ToolError:
        """Delete a secret by name.

        Parameters
        ----------
        name : str
            The secret name to remove.
        destination : str | None, default None
            Which copy of the credential to remove. Defaults to ``"personal"``.
            Passing ``"team:<id>"`` removes the shared credential from the
            team for every member, breaking any team integration that depends
            on it; do not delete a shared credential unless the team decision
            is to rotate or retire the integration. See the *Accessible shared teams* block in your system prompt. Inside a task, an omitted
            destination inherits the task destination.

        Returns
        -------
        ToolOutcome | ToolError
            Outcome dict or structured tool error for invalid destinations.
        """
        try:
            context = self._secret_context_for_destination(destination)
        except ToolErrorException as exc:
            return exc.payload

        ids = unisdk.get_logs(
            context=context,
            filter=f"name == {name!r}",
            limit=2,
            return_ids_only=True,
        )
        if not ids:
            raise ValueError(f"No secret found with name '{name}'.")
        if len(ids) > 1:
            raise RuntimeError(f"Multiple secrets found with name '{name}'.")
        unisdk.delete_logs(context=context, logs=ids[0])
        try:
            if self._is_personal_context(context):
                self._env_remove(name)
        except Exception:
            pass
        finally:
            self._invalidate_credential_processes()
        return {"outcome": "secret deleted", "details": {"name": name}}
