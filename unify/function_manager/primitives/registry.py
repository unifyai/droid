"""
ToolSurfaceRegistry: Single source of truth for manager/primitive configuration.

This module centralizes all manager definitions, primitive method discovery,
prompt context generation, and primitive row collection for FunctionManager indexing.

No other module should define manager lists or primitive configurations.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type, TYPE_CHECKING

from unify.function_manager.hash_utils import stable_hash_for_rows
from unify.function_manager.primitives.scope import PrimitiveScope

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# =============================================================================
# ManagerSpec - Per-manager configuration
# =============================================================================


@dataclass(frozen=True, slots=True)
class ManagerSpec:
    """
    Configuration for a single manager in the tool surface.

    This is the authoritative specification for each manager, containing:
    - Identity (alias, registry key, class path)
    - Method exclusions
    - Prompt metadata (domain, description, use_when, examples, priority)
    - Sandbox namespace root (the top-level key in the sandbox's global_state)
    """

    manager_alias: str
    manager_registry_key: str
    primitive_class_path: str
    # The top-level key under which this manager's methods appear in a
    # CodeActActor sandbox's global_state.  All managers live under
    # ``"primitives"`` (accessed as ``primitives.<alias>.<method>``).
    # Used by construct_sandbox_root() to map dotted depends_on entries
    # back to the class that provides them.
    sandbox_root: str = "primitives"
    excluded_methods: frozenset[str] = field(default_factory=frozenset)
    priority: int = 99
    domain: str = ""
    description: str = ""
    use_when: str = ""
    examples: str = ""
    special_note: str | None = None


# =============================================================================
# Common excluded methods (applies to all managers)
# =============================================================================

_COMMON_EXCLUDED_METHODS: frozenset[str] = frozenset(
    {
        # State management / lifecycle
        "clear",
        # Internal helpers
        "add_tools",
        "get_tools",
    },
)


# =============================================================================
# Canonical Manager Registry (SINGLE SOURCE OF TRUTH)
# =============================================================================

_MANAGER_SPECS: tuple[ManagerSpec, ...] = (
    ManagerSpec(
        manager_alias="comms",
        manager_registry_key="",
        primitive_class_path="unify.comms.primitives.CommsPrimitives",
        excluded_methods=frozenset(),
        priority=4,
        domain="Assistant-Owned Communication",
        description=(
            "Send assistant-owned outbound messages and calls across SMS, email, "
            "WhatsApp, Discord, Unify, and API-response channels"
        ),
        use_when=(
            "The assistant needs to proactively contact people or post into "
            "assistant-owned communication channels as part of a task"
        ),
        examples=(
            "'Text Alice that the meeting moved', 'Email all shortlisted leads', "
            "'Reply in the Discord channel', 'Send a Unify update to the team lead'"
        ),
    ),
    ManagerSpec(
        manager_alias="contacts",
        manager_registry_key="contacts",
        primitive_class_path="unify.contact_manager.contact_manager.ContactManager",
        excluded_methods=frozenset({"filter_contacts", "update_contact"}),
        priority=3,
        domain="People & Relationships",
        description="People, organizations, contact records (names, emails, phones, roles, locations)",
        use_when="Questions about specific people, contact info, 'who is X?', 'find contact in Y location'",
        examples="'Who is our contact at Acme Corp?', 'Find Alice's email', 'Contacts in Berlin?'",
    ),
    ManagerSpec(
        manager_alias="dashboards",
        manager_registry_key="dashboards",
        primitive_class_path="unify.dashboard_manager.dashboard_manager.DashboardManager",
        excluded_methods=frozenset(),
        priority=8,
        domain="Visualizations & Dashboards",
        description=(
            "Create HTML visualization tiles (charts, plots, KPI cards, tables) "
            "and compose them into dashboard grid layouts"
        ),
        use_when=(
            "Generate any visualization, chart, plot, dashboard, KPI card, "
            "or interactive HTML output for the user to view"
        ),
        examples=(
            "'Plot repairs by category', 'Show me a bar chart of revenue', "
            "'Create a dashboard with KPIs and charts', "
            "'Generate an interactive table of orders'"
        ),
        special_note=(
            "The actor has full creative freedom over tile HTML -- any "
            "HTML/CSS/JS that renders in a browser works. PREFER live tiles: "
            "declare data_bindings (FilterBinding, ReduceBinding, JoinBinding, "
            "JoinReduceBinding) as the single source of truth and provide an "
            "on_data JS callback that receives fetched data keyed by alias. "
            "Console auto-generates bridge calls from the serialized bindings "
            "-- the actor never writes bridge API calls. Each binding is "
            "auto-validated via the corresponding DataManager method before "
            "the tile is stored. Compose tiles into dashboards with "
            "create_dashboard(). Live data tiles are essential for production: "
            "large datasets, joins, aggregation, and frequently updated data. "
            "Only bake data into HTML for very small static snapshots."
        ),
    ),
    ManagerSpec(
        manager_alias="data",
        manager_registry_key="data",
        primitive_class_path="unify.data_manager.data_manager.DataManager",
        excluded_methods=frozenset(),
        priority=9,
        domain="Data Operations & Ingestion",
        description=(
            "Low-level and high-level data operations on any Unify context "
            "(filter, search, reduce, join, ingest, vectorize)"
        ),
        use_when=(
            "Data operations, aggregations, "
            "data ingestion from APIs/warehouses, cross-context joins"
        ),
        examples=(
            "'Filter rows where amount > 1000', 'Join repairs with telematics', "
            "'Sum revenue by region', 'Ingest this API data'"
        ),
        special_note=(
            "DataManager operates on any Unify context path. "
            "Use `describe_table` and `list_tables` to discover schemas and available tables."
        ),
    ),
    ManagerSpec(
        manager_alias="transcripts",
        manager_registry_key="transcripts",
        primitive_class_path="unify.transcript_manager.transcript_manager.TranscriptManager",
        excluded_methods=frozenset(),
        priority=2,
        domain="Conversation History",
        description="Past messages, conversation history, communication records (chat/SMS/email)",
        use_when="Questions about past communications, 'what did X say?', 'last message about Y?', 'conversation with Z?'",
        examples="'What did Bob say yesterday?', 'Last SMS with Alice?', 'Messages mentioning budget?'",
    ),
    ManagerSpec(
        manager_alias="knowledge",
        manager_registry_key="knowledge",
        primitive_class_path="unify.knowledge_manager.knowledge_manager.KnowledgeManager",
        excluded_methods=frozenset({"filter", "search"}),
        priority=1,
        domain="Facts, Policies & Domain Knowledge",
        description="Organizational facts, policies, procedures, reference material, documentation, stored information",
        use_when="Questions about company policies, operational procedures, reference docs, 'what is our X policy?', 'summarize Y procedure'",
        examples="'What's our return policy?', 'Summarize onboarding procedure', 'Office hours?', 'Warranty terms for X?'",
    ),
    ManagerSpec(
        manager_alias="tasks",
        manager_registry_key="tasks",
        primitive_class_path="unify.task_scheduler.task_scheduler.TaskScheduler",
        excluded_methods=frozenset({"get_active_singleton_handle"}),
        priority=4,
        domain="Durable Work & Tracking",
        description="Task management, assignments, deadlines, priorities",
        use_when="Questions about tasks/work items, 'what's due?', 'tasks assigned to X?', 'high-priority items?'",
        examples="'What tasks are due today?', 'Show Alice's open tasks', 'List high-priority items'",
    ),
    ManagerSpec(
        manager_alias="secrets",
        manager_registry_key="secrets",
        primitive_class_path="unify.secret_manager.secret_manager.SecretManager",
        excluded_methods=frozenset(),
        priority=8,
        domain="Credentials & Secrets",
        description="API keys, passwords, tokens, credentials",
        use_when="Managing credentials, API keys, secrets (rarely used in plans)",
        examples="Rarely used directly in plans",
    ),
    ManagerSpec(
        manager_alias="web",
        manager_registry_key="web_search",
        primitive_class_path="unify.web_searcher.web_searcher.WebSearcher",
        excluded_methods=frozenset(),
        priority=5,
        domain="Time-Sensitive & External Research",
        description="Quick one-off internet queries against the public web (headlines, weather, definitions, current events). Not for gated sites, browser automation, or multi-step web workflows — use Tavily + SecretManager + ComputerPrimitives directly for those",
        use_when="Fast, simple public-web lookups: current events, weather, news, definitions, stock prices, quick factual questions",
        examples="'What is the Eisenhower Matrix?', 'Weather in Berlin today?', 'Latest AI news?', 'Current stock price?'",
    ),
    ManagerSpec(
        manager_alias="files",
        manager_registry_key="files",
        primitive_class_path="unify.file_manager.managers.file_manager.FileManager",
        excluded_methods=frozenset(
            {
                "exists",
                "list",
                "ingest_files",
                "export_file",
                "export_directory",
                "rename_file",
                "move_file",
                "delete_file",
                "sync",
            },
        ),
        priority=7,
        domain="File Operations & Document Parsing",
        description=(
            "File-specific operations: describing storage layout, listing files, "
            "parsing documents, rendering PDFs/Excel sheets as images"
        ),
        use_when=(
            "Questions about files themselves (metadata, layout, parsing), "
            "file-path-based context resolution, document rendering"
        ),
        examples=(
            "'Parse the attached PDF', 'What's in document X?', "
            "'Describe the schema of report.xlsx', 'Render page 3 of the report'"
        ),
    ),
    ManagerSpec(
        manager_alias="workspace_email",
        manager_registry_key="",
        primitive_class_path=(
            "unify.workspace_email.workspace_email_manager.WorkspaceEmailManager"
        ),
        excluded_methods=frozenset(),
        priority=4,
        domain="Connected Workspace Mailbox (Gmail / Outlook)",
        description=(
            "Send from and read the user's CONNECTED Google Workspace / "
            "Microsoft 365 mailbox (the account linked via OAuth). Sends as the "
            "connected account, not the assistant's own managed mailbox."
        ),
        use_when=(
            "The task is explicitly about the user's own connected inbox: "
            "reading/searching messages in their linked Gmail/Outlook, or "
            "sending an email that must come FROM the user's connected account"
        ),
        examples=(
            "'Check my inbox for anything from the landlord', "
            "'Search my email for the Q3 invoice', "
            "'Reply from my own account to the client thread'"
        ),
        special_note=(
            "Distinct from `primitives.comms.send_email`, which sends AS THE "
            "ASSISTANT from its managed mailbox and is wired into the contact "
            "graph. Use `workspace_email` only to act on the user's own "
            "connected account. Recipients here are plain email-address "
            "strings, not contact ids. Requires a connected workspace account "
            "with the relevant Gmail/Graph scopes; errors clearly if none is "
            "connected."
        ),
    ),
    ManagerSpec(
        manager_alias="integrations",
        manager_registry_key="",
        primitive_class_path="unify.integrations.primitives.IntegrationPrimitives",
        excluded_methods=frozenset(
            {
                "list_connected",
                "search_tools",
                "get_tool_schema",
                "execute_tool",
                "manage_connection",
                "resolve_tool_id",
                "callable_for_tool",
            },
        ),
        priority=6,
        domain="Integration App Discovery & Connected SaaS Apps",
        description=(
            "`primitives.integrations.search_integrations` is the app discovery "
            "surface for both Native Unity-deploy packages and Third-party "
            "provider-backed apps. It answers support, deployment activation, "
            "provider connection, and sync/materialization status. Provider "
            "tools execute through Orchestra policy/audit after materialization; "
            "native package functions keep their existing custom function, MCP, "
            "guidance, and computer-use routing."
        ),
        use_when=(
            "The user asks to read or act in connected external apps such as "
            "HubSpot, Salesforce, Google Drive, Gmail, Slack, Notion, Clay, "
            "or a native deployment package. Also use it when a user asks "
            "whether an integration is available, native vs third-party, "
            "enabled for this assistant, connected, missing secrets/scopes, "
            "expired, or blocked by policy."
        ),
        examples=(
            "'Find recent HubSpot leads', 'Search Salesforce opportunities', "
            "'Look up Google Drive files modified this week', 'Search Gmail "
            "for invoices from Acme', 'Send this approved update to Slack', "
            "'Which CRM integrations are connected?'"
        ),
        special_note=(
            "Use `search_integrations` to answer whether an app is supported, "
            "Native or Third-party, enabled/connected, or still syncing. "
            "FunctionManager search remains the only actor-facing discovery "
            "surface for executable functions/tools. Native apps become "
            "executable through deployment-enabled package functions; "
            "Third-party apps become executable after connection and provider "
            "tool materialization. Inspect each row's docstring and argspec "
            "before calling, and require user confirmation for write, "
            "destructive, bulk-export, or sensitive actions."
        ),
    ),
    ManagerSpec(
        manager_alias="coordinator",
        manager_registry_key="",  # No ManagerRegistry getter - singleton via metaclass
        primitive_class_path="unify.coordinator_manager.coordinator_manager.CoordinatorManager",
        excluded_methods=frozenset(),
        priority=6,
        domain="Coordinator Workspace Administration",
        description=(
            "Coordinator-only workspace lifecycle operations for assistants, teams, "
            "membership, and Coordinator/State onboarding mode"
        ),
        use_when=(
            "The session is running as a Coordinator and needs privileged setup "
            "or lifecycle mutations on colleagues and shared workspaces"
        ),
        examples=(
            "'Create the Regional Ops Manager colleague', "
            "'Add the supervisor to Patch 4 workspace', "
            "'Mark the first setup slice ready once validation passes'"
        ),
    ),
    ManagerSpec(
        manager_alias="computer",
        manager_registry_key="",  # No ManagerRegistry getter - singleton via metaclass
        primitive_class_path="unify.function_manager.primitives.runtime.ComputerPrimitives",
        excluded_methods=frozenset(),
        priority=10,
        domain="Web & Desktop Control",
        description="Browser automation, web navigation, computer use actions, reasoning",
        use_when="Web automation, browser control, navigating websites, extracting web content",
        examples="'Navigate to example.com', 'Click the login button', 'Extract page content'",
    ),
    ManagerSpec(
        manager_alias="actor",
        manager_registry_key="",  # No ManagerRegistry getter - stateless, constructed directly
        primitive_class_path="unify.actor.environments.actor._ActorRunner",
        excluded_methods=frozenset(),
        priority=11,
        domain="Actor Delegation",
        description="Spawn focused sub-actors for isolated multi-step sub-tasks",
        use_when="Task decomposition, isolated reasoning, parallel sub-tasks, context isolation",
        examples="'Delegate research to a sub-actor', 'Spawn an actor to handle data processing'",
    ),
)

# Build lookup dicts for fast access
_MANAGER_BY_ALIAS: Dict[str, ManagerSpec] = {
    spec.manager_alias: spec for spec in _MANAGER_SPECS
}

# Reverse mapping: primitive_class_path -> manager_alias (for deriving alias from stored primitive_class)
_CLASS_PATH_TO_ALIAS: Dict[str, str] = {
    spec.primitive_class_path: spec.manager_alias for spec in _MANAGER_SPECS
}

# sandbox_root -> [specs]: maps each sandbox namespace root to its manager specs.
# Used by construct_sandbox_root() to materialise environment namespaces at
# runtime.  The keys here must stay in sync with the env_namespaces frozenset
# in FunctionManager.add_functions (the storage-time AST detection side).
_SANDBOX_ROOTS: Dict[str, list[ManagerSpec]] = {}
for _spec in _MANAGER_SPECS:
    _SANDBOX_ROOTS.setdefault(_spec.sandbox_root, []).append(_spec)


def construct_sandbox_root(
    root_name: str,
    *,
    primitive_scope: Optional[PrimitiveScope] = None,
) -> Optional[Any]:
    """Construct a fresh root namespace object for a sandbox namespace key.

    This is the factory used by ``FunctionManager._inject_dependencies``
    to satisfy *dotted* entries in a stored function's ``depends_on`` list.
    When a function declares a dependency like ``"primitives.actor.act"`` or
    ``"primitives.contacts.ask"``, ``_inject_dependencies`` extracts the
    root segment (``"primitives"``) and calls this function to obtain a
    live ``Primitives`` instance that provides those methods.

    The returned object is **stateless** — it does not require any ambient
    ContextVars or parent actor state.  This is essential because stored
    functions may be executed outside of a live ``CodeActActor`` sandbox
    (e.g. by the storage-check loop's ``FunctionManager_add_functions``
    tool or by a future caller that only has a FunctionManager).

    Returns ``None`` when *root_name* does not match any known sandbox root.
    """
    specs = _SANDBOX_ROOTS.get(root_name)
    if not specs:
        return None

    from unify.function_manager.primitives.runtime import Primitives

    return Primitives(primitive_scope=primitive_scope)


# =============================================================================
# Routing Guidance for Commonly Confused Manager Pairs
# =============================================================================

_ROUTING_GUIDANCE: List[Dict[str, Any]] = [
    {
        "managers": {"data", "files"},
        "title": "`primitives.data.*` vs `primitives.files.*`",
        "guidance": [
            (
                "data",
                "Use for **ALL analytical and data operations** -- filtering, searching, "
                "aggregating, joining, and data ingestion. Works on "
                "any Unify context (`Data/*`, `Files/*`, `Knowledge/*`). Use when the question "
                "is about DATA INSIDE tables, regardless of where those tables came from. "
                "Also use for ingesting data from APIs/warehouses.",
            ),
            (
                "files",
                "Use for **file-specific operations** -- describing the storage layout of "
                "an uploaded/received file, listing files in the file registry, parsing "
                "documents, rendering PDFs/Excel sheets as images for visual inspection. "
                "Use when the question is about FILES themselves (metadata, layout, parsing) "
                "or when you need file-path-based context resolution.",
            ),
        ],
        "examples": [
            (
                "What tables exist under Data/examplehousing?",
                "data",
                "primitives.data.list_tables(prefix=...)",
            ),
            (
                "Describe the schema of the repairs table",
                "data",
                "primitives.data.describe_table(...)",
            ),
            (
                "What's in the uploaded PDF?",
                "files",
                "primitives.files.describe(file_path=...)",
            ),
            (
                "Render page 3 of the report as an image",
                "files",
                "primitives.files.render_pdf(...)",
            ),
        ],
    },
]


# =============================================================================
# Example generator mapping (manager_alias -> list of function names)
# =============================================================================

_EXAMPLE_GENERATORS: Dict[str, List[str]] = {
    "contacts": [
        "get_primitives_contact_ask_example",
        "get_primitives_contact_update_example",
    ],
    "tasks": [
        "get_primitives_task_execute_example",
    ],
    "knowledge": [
        "get_primitives_knowledge_ask_example",
        "get_primitives_knowledge_update_example",
    ],
    "transcripts": [
        "get_primitives_transcript_ask_example",
    ],
    "web": [
        "get_primitives_web_ask_example",
    ],
    "secrets": [
        "get_primitives_secrets_ask_example",
        "get_primitives_secrets_update_example",
    ],
    "files": [
        "get_primitives_files_render_extract_example",
        "get_primitives_files_describe_example",
        "get_primitives_files_reduce_example",
        "get_primitives_files_filter_example",
        "get_primitives_files_search_example",
    ],
    "data": [
        "get_primitives_data_filter_example",
        "get_primitives_data_reduce_example",
        "get_primitives_data_ingest_example",
    ],
    "dashboards": [
        "get_primitives_dashboards_baked_in_example",
        "get_primitives_dashboards_live_data_example",
        "get_primitives_dashboards_rich_live_data_example",
        "get_primitives_dashboards_composition_example",
    ],
    "integrations": [
        "get_primitives_integrations_function_manager_search_example",
        "get_primitives_integrations_catalog_status_example",
        "get_primitives_integrations_materialized_schema_example",
        "get_primitives_integrations_activation_state_example",
        "get_primitives_integrations_confirmation_example",
    ],
}


# =============================================================================
# Stable ID Generation (Matching Old Format)
# =============================================================================


def _get_stable_id(class_name: str, method_name: str) -> int:
    """
    Generate a stable integer ID from class.method name.

    Uses a hash-based approach so IDs are:
    - Deterministic (same name → same ID)
    - Stable across code changes (no positional dependencies)
    - Unique (collision-resistant within practical limits)

    Args:
        class_name: Short class name (e.g., "ContactManager")
        method_name: Method name (e.g., "ask")

    Returns:
        A stable non-negative integer ID within signed 32-bit range.
    """
    key = f"{class_name}.{method_name}"
    # Use first 4 bytes of SHA256, masked to signed int32 range (0x7FFFFFFF)
    # to avoid PostgreSQL integer overflow when sorting by function_id
    digest = hashlib.sha256(key.encode()).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


# =============================================================================
# NumPy-style docstring helpers
# =============================================================================


def _get_line_indent(line: str) -> int:
    """Count the number of leading spaces in a line."""
    return len(line) - len(line.lstrip())


def _is_param_declaration_at_level(line: str, base_indent: int) -> bool:
    """Detect whether a line is a parameter declaration at the given indent level.

    A declaration is at ``base_indent`` and matches one of:
    - ``name : type``  (standard single-param)
    - ``name1, name2``  (grouped params sharing a description)
    - ``name1 / name2 : type``  (slash-separated grouped params)
    - ``name``  (bare name, no type annotation)
    """
    stripped = line.strip()
    if not stripped:
        return False
    indent = _get_line_indent(line)
    if indent != base_indent:
        return False
    # Standard "name : type" format
    if " : " in stripped:
        return True
    # Grouped comma-separated names: "a, b, c"
    if "," in stripped:
        parts = [p.strip().rstrip(",") for p in stripped.split(",") if p.strip()]
        if all(re.match(r"^[a-zA-Z_]\w*$", p) for p in parts):
            return True
    # Bare single name
    if re.match(r"^[a-zA-Z_]\w*$", stripped):
        return True
    return False


def _extract_param_names(declaration_line: str) -> list[str]:
    """Extract parameter names from a NumPy-style declaration line.

    Handles: ``name : type``, ``name1, name2``, ``name1 / name2 : type``.
    """
    stripped = declaration_line.strip()
    name_part = stripped.split(" : ")[0] if " : " in stripped else stripped
    # Handle "/" separator (e.g., "_clarification_up_q / _clarification_down_q")
    if " / " in name_part:
        names = [n.strip() for n in name_part.split("/") if n.strip()]
    else:
        names = [n.strip().rstrip(",") for n in name_part.split(",") if n.strip()]
    return names


# =============================================================================
# ToolSurfaceRegistry - Main API
# =============================================================================


class ToolSurfaceRegistry:
    """
    Central registry for tool surface configuration.

    This class provides the API for:
    - Getting manager specs filtered by scope
    - Discovering primitive methods for a manager
    - Generating prompt context and examples
    - Collecting primitive rows for FunctionManager indexing
    - Building row filters for scoped queries
    """

    # Class-level references to canonical data
    MANAGERS = _MANAGER_SPECS
    ROUTING_GUIDANCE = _ROUTING_GUIDANCE
    EXAMPLE_GENERATORS = _EXAMPLE_GENERATORS

    def __init__(self) -> None:
        """Initialize the registry."""
        # Cache for dynamically loaded classes (avoids repeated imports)
        self._class_cache: Dict[str, Type] = {}

    def manager_specs(self, primitive_scope: PrimitiveScope) -> List[ManagerSpec]:
        """
        Get manager specs for a given scope, sorted by priority.

        Args:
            primitive_scope: The scope defining which managers are exposed.

        Returns:
            List of ManagerSpec for exposed managers, sorted by priority.
        """
        specs = [
            spec
            for spec in _MANAGER_SPECS
            if spec.manager_alias in primitive_scope.scoped_managers
        ]
        return sorted(specs, key=lambda s: s.priority)

    def get_manager_spec(self, manager_alias: str) -> Optional[ManagerSpec]:
        """Get a single manager spec by alias (includes ComputerPrimitives)."""
        return _MANAGER_BY_ALIAS.get(manager_alias)

    def get_function_id(self, manager_alias: str, method_name: str) -> int:
        """Compute the stable function_id for a primitive method.

        This returns the same ID that ``collect_primitives`` produces for rows
        stored in the builtins primitives catalogue, without requiring a DB
        round-trip.

        Args:
            manager_alias: Canonical manager alias (e.g., ``"contacts"``).
            method_name: Method name (e.g., ``"ask"``).

        Returns:
            Stable non-negative integer ID (deterministic, hash-based).

        Raises:
            ValueError: If *manager_alias* is not a known manager.
        """
        spec = _MANAGER_BY_ALIAS.get(manager_alias)
        if spec is None:
            raise ValueError(f"Unknown manager alias: {manager_alias!r}")
        class_name = spec.primitive_class_path.rsplit(".", 1)[-1]
        return _get_stable_id(class_name, method_name)

    def _load_manager_class(self, class_path: str) -> Optional[Type]:
        """Dynamically load a manager class for introspection."""
        if class_path in self._class_cache:
            return self._class_cache[class_path]

        try:
            module_path, class_name = class_path.rsplit(".", 1)
            import importlib

            module = importlib.import_module(module_path)
            cls = getattr(module, class_name)
            self._class_cache[class_path] = cls
            return cls
        except Exception as e:
            logger.warning(f"Failed to load class {class_path}: {e}")
            return None

    def primitive_methods(self, *, manager_alias: str) -> List[str]:
        """
        Get the list of primitive methods for a manager.

        Discovers methods from @abstractmethod definitions on Base* classes,
        minus common exclusions and per-manager exclusions.

        For ComputerPrimitives, uses the class's _PRIMITIVE_METHODS constant
        since methods are dynamically created via setattr in __init__.

        Args:
            manager_alias: The canonical manager alias.

        Returns:
            Sorted list of method names exposed as primitives.
        """
        spec = _MANAGER_BY_ALIAS.get(manager_alias)
        if not spec:
            logger.warning(f"Unknown manager alias: {manager_alias}")
            return []

        cls = self._load_manager_class(spec.primitive_class_path)
        if cls is None:
            return []

        # Combine common exclusions with per-manager exclusions
        exclude = _COMMON_EXCLUDED_METHODS | spec.excluded_methods

        # Classes that declare _PRIMITIVE_METHODS explicitly (e.g. because
        # methods are dynamically created or not discoverable via @abstractmethod)
        # use that constant as the authoritative method list.
        if hasattr(cls, "_PRIMITIVE_METHODS"):
            return sorted([m for m in cls._PRIMITIVE_METHODS if m not in exclude])

        methods = []

        # Standard case: find @abstractmethod definitions in Base* classes
        for base in cls.__mro__:
            base_name = base.__name__
            # Look for Base* classes (e.g., BaseContactManager, BaseFileManager)
            if not base_name.startswith("Base"):
                continue
            # Skip the root BaseStateManager - we want the specific manager's base
            if base_name == "BaseStateManager":
                continue

            for name, method in vars(base).items():
                if name.startswith("_"):
                    continue
                if name in exclude:
                    continue
                if getattr(method, "__isabstractmethod__", False):
                    if name not in methods:
                        methods.append(name)

        return sorted(methods)

    def tool_names(self, primitive_scope: PrimitiveScope) -> List[str]:
        """
        Get fully-qualified tool names for a scope.

        Args:
            primitive_scope: The scope defining which managers are exposed.

        Returns:
            List of tool names like "primitives.contacts.ask".
        """
        names = []
        for alias in sorted(primitive_scope.scoped_managers):
            for method in self.primitive_methods(manager_alias=alias):
                names.append(f"primitives.{alias}.{method}")
        return names

    def tool_metadata(
        self,
        primitive_scope: PrimitiveScope,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get metadata for all tools in scope.

        Args:
            primitive_scope: The scope defining which managers are exposed.

        Returns:
            Dict mapping tool name to metadata (is_impure, is_steerable).
        """
        metadata = {}
        for name in self.tool_names(primitive_scope):
            # Conservative: all state manager primitives are impure and steerable
            metadata[name] = {
                "is_impure": True,
                "is_steerable": True,
            }
        return metadata

    @staticmethod
    def _resolve_base_method(cls: Optional[Type], method_name: str):
        """Find the canonical method object from the Base* class in the MRO.

        Returns the method object or None.
        """
        if cls is None:
            return None
        for base in cls.__mro__:
            if base.__name__.startswith("Base") and method_name in vars(base):
                return vars(base)[method_name]
        return getattr(cls, method_name, None)

    @staticmethod
    def _format_method_signature(cls: Optional[Type], method_name: str) -> str:
        """Extract a compact signature string for a method.

        Strips ``self`` and any private ``_``-prefixed parameters (internal
        wiring like ``_parent_chat_context`` or ``_clarification_up_q``).

        Falls back to ``...`` when the class is unavailable or introspection
        fails for any reason.
        """
        method = ToolSurfaceRegistry._resolve_base_method(cls, method_name)
        if method is None:
            return "..."
        try:
            sig = inspect.signature(method)
            params = [
                p
                for name, p in sig.parameters.items()
                if name != "self" and not name.startswith("_")
            ]
            return str(sig.replace(parameters=params))
        except (ValueError, TypeError):
            return "..."

    @staticmethod
    def _extract_method_docstring(cls: Optional[Type], method_name: str) -> str:
        """Extract the full docstring for a method from the Base* class.

        Unwraps ``functools.wraps`` / ``__wrapped__`` to reach the original
        docstring.  Returns an empty string when unavailable.
        """
        method = ToolSurfaceRegistry._resolve_base_method(cls, method_name)
        if method is None:
            return ""
        fn = method
        while hasattr(fn, "__wrapped__"):
            fn = fn.__wrapped__
        return inspect.getdoc(fn) or ""

    @staticmethod
    def _extract_summary_and_params(docstring: str) -> str:
        """Extract the first paragraph and Parameters block from a docstring.

        Returns a compact version containing the summary (up to the first
        blank line) and the NumPy-style ``Parameters`` section, omitting
        Returns, Raises, Examples, Notes, other verbose sections, and any
        internal ``_``-prefixed parameters.
        """
        if not docstring:
            return ""

        lines = docstring.splitlines()

        # First paragraph (up to first blank line after content)
        summary_lines: list[str] = []
        i = 0
        for i, line in enumerate(lines):
            if not line.strip():
                if summary_lines:
                    break
            else:
                summary_lines.append(line)

        # Find Parameters section
        params_lines: list[str] = []
        in_params = False
        for j in range(i, len(lines)):
            stripped = lines[j].strip()
            if stripped == "Parameters":
                in_params = True
                continue
            if in_params and stripped.startswith("---"):
                continue
            if in_params:
                # Stop at next section header
                if (
                    stripped
                    and not stripped[0].isspace()
                    and not stripped.startswith("-")
                ):
                    if j + 1 < len(lines) and lines[j + 1].strip().startswith(
                        "---",
                    ):
                        break
                    if re.match(r"^[A-Z][a-zA-Z\s]+$", stripped) and len(stripped) < 30:
                        break
                params_lines.append(lines[j])

        # Filter out internal _-prefixed parameter entries.
        # Detect the base indentation level from the first non-empty line.
        base_indent = 0
        for line in params_lines:
            if line.strip():
                base_indent = _get_line_indent(line)
                break

        filtered_params: list[str] = []
        skip_entry = False
        for line in params_lines:
            if _is_param_declaration_at_level(line, base_indent):
                names = _extract_param_names(line)
                skip_entry = bool(names) and all(n.startswith("_") for n in names)
            if not skip_entry:
                filtered_params.append(line)

        result = "\n".join(summary_lines)
        params = "\n".join(filtered_params).rstrip()
        if params:
            result += "\n\nParameters\n----------\n" + params
        return result

    @staticmethod
    def _filter_internal_params_from_docstring(docstring: str) -> str:
        """Remove ``_``-prefixed parameter entries from a full docstring.

        Scans the Parameters section and drops any entry whose parameter
        name starts with ``_`` (including its indented description lines).
        All other docstring content is preserved unchanged.
        """
        if not docstring or "Parameters" not in docstring:
            return docstring

        lines = docstring.splitlines()
        result_lines: list[str] = []
        in_params = False
        skip_entry = False
        param_base_indent: int = 0

        i = 0
        while i < len(lines):
            stripped = lines[i].strip()

            # Detect start of Parameters section
            if not in_params and stripped == "Parameters":
                # Check for the dashes line that follows
                if i + 1 < len(lines) and lines[i + 1].strip().startswith("---"):
                    in_params = True
                    result_lines.append(lines[i])
                    result_lines.append(lines[i + 1])
                    # Detect base indent from the first param declaration
                    for k in range(i + 2, len(lines)):
                        if lines[k].strip():
                            param_base_indent = _get_line_indent(lines[k])
                            break
                    i += 2
                    continue
                else:
                    result_lines.append(lines[i])
                    i += 1
                    continue

            if in_params:
                # Detect next section header (end of Parameters)
                if (
                    stripped
                    and _get_line_indent(lines[i]) <= param_base_indent
                    and not stripped.startswith("-")
                ):
                    is_section = False
                    if i + 1 < len(lines) and lines[i + 1].strip().startswith("---"):
                        is_section = True
                    if re.match(r"^[A-Z][a-zA-Z\s]+$", stripped) and len(stripped) < 30:
                        is_section = True

                    if is_section:
                        in_params = False
                        skip_entry = False
                        result_lines.append(lines[i])
                        i += 1
                        continue

                if _is_param_declaration_at_level(lines[i], param_base_indent):
                    names = _extract_param_names(lines[i])
                    skip_entry = bool(names) and all(n.startswith("_") for n in names)

                if not skip_entry:
                    result_lines.append(lines[i])
            else:
                result_lines.append(lines[i])

            i += 1

        return "\n".join(result_lines)

    def prompt_context(self, primitive_scope: PrimitiveScope) -> str:
        """
        Generate prompt context for exposed managers.

        Structure: routing guidance first (which manager to pick), then
        detailed method documentation. This ensures the LLM reads selection
        criteria before wading through method signatures.

        Args:
            primitive_scope: The scope defining which managers are exposed.

        Returns:
            Formatted prompt context string.
        """
        specs = self.manager_specs(primitive_scope)
        if not specs:
            return ""

        lines = ["### State manager primitives (`primitives.*`)\n"]
        lines.append(
            "Each manager owns a specific domain of the assistant's durable state. "
            "Choose the right manager for your task:\n",
        )

        # ── Section 1: Brief manager overview (routing-focused) ──
        for spec in specs:
            lines.append(f"\n**{spec.domain}** → `primitives.{spec.manager_alias}`")
            lines.append(f"- {spec.description}")
            if spec.use_when:
                lines.append(f"- **Use when**: {spec.use_when}")
            if spec.examples:
                lines.append(f"- **Examples**: {spec.examples}")
            if spec.special_note:
                lines.append(f"- **Note**: {spec.special_note}")

        # ── Section 2: Manager selection priorities ──
        if len(specs) > 1:
            lines.append("\n**Manager Selection Priorities**:")
            lines.append(
                "1. **knowledge** takes priority for organizational policies, procedures, company facts, internal documentation",
            )
            lines.append(
                "2. **transcripts** for historical communications (what was said/written)",
            )
            lines.append("3. **contacts** for people/relationship information")
            lines.append("4. **tasks** for work items, deadlines, assignments")
            lines.append(
                "5. **web** for current external information (weather, news, real-time data)",
            )
            lines.append(
                "7. **files** when dealing with specific documents or file-level operations",
            )

        # ── Section 3: Routing guidance for commonly confused pairs ──
        exposed_aliases = primitive_scope.scoped_managers
        for guidance in _ROUTING_GUIDANCE:
            if guidance["managers"].issubset(exposed_aliases):
                lines.append(f"\n**CRITICAL: {guidance['title']} Routing**:")
                lines.append(
                    "These managers serve DIFFERENT purposes - do not confuse them:",
                )
                for alias, desc in guidance["guidance"]:
                    lines.append(f"- **`primitives.{alias}.*`**: {desc}")
                if guidance.get("examples"):
                    lines.append("\n**Examples**:")
                    for question, mgr, call in guidance["examples"]:
                        lines.append(f'  - "{question}" → `{call}` ({mgr})')

        # ── Section 4: General rules ──
        if len(specs) > 1:
            lines.append("\n**General Rules**:")
            lines.append(
                "- All manager calls return a steerable handle; default to returning the handle as the last expression so outer-loop steering/progress stays available. Await `.result()` only for immediate in-code composition",
            )
            lines.append(
                "- Calls to `primitives.*` are nested tool loops; emit `notify({...})` before each call, and emit a concrete completion update when you await and continue with more steps",
            )
            lines.append(
                "- Notification messages should be user-facing progress summaries, not low-level technical diagnostics",
            )
            lines.append(
                "- If a manager asks for clarification, wait for the user response and answer via the handle's API",
            )
            lines.append(
                "- Prefer `ask(...)` for read-only queries; only use `update(...)`/`execute(...)` when mutations are needed",
            )
            lines.append(
                "- Don't call `ask(...)` before `update(...)`/`execute(...)` on the same manager to pre-check state; mutation methods already inspect existing records, so a preemptive read is duplicative",
            )
            lines.append(
                "- When in doubt between managers, prefer the most specific domain match",
            )

        # ── Section 5: Detailed method documentation ──
        lines.append("\n---\n")
        lines.append("### Method Reference\n")

        for spec in specs:
            mgr_cls = self._load_manager_class(spec.primitive_class_path)
            method_names = self.primitive_methods(manager_alias=spec.manager_alias)
            if not method_names:
                continue

            lines.append(f"\n#### `primitives.{spec.manager_alias}`")

            for method_name in method_names:
                sig_str = self._format_method_signature(mgr_cls, method_name)
                full_doc = self._extract_method_docstring(mgr_cls, method_name)
                compact_doc = self._extract_summary_and_params(full_doc)
                lines.append(f"\n**`.{method_name}{sig_str}`**")
                if compact_doc:
                    for doc_line in compact_doc.splitlines():
                        lines.append(f"  {doc_line}")

        return "\n".join(lines)

    def prompt_examples(self, primitive_scope: PrimitiveScope) -> str:
        """
        Get concatenated examples for exposed managers.

        Args:
            primitive_scope: The scope defining which managers are exposed.

        Returns:
            Formatted examples string.
        """
        try:
            from unify.actor.prompt_examples import get_example_function_map
        except ImportError:
            logger.warning("Could not import prompt_examples module")
            return ""

        fn_map = get_example_function_map()
        examples = []

        for alias in sorted(primitive_scope.scoped_managers):
            fn_names = _EXAMPLE_GENERATORS.get(alias, [])
            for fn_name in fn_names:
                fn = fn_map.get(fn_name)
                if fn:
                    try:
                        example = fn()
                        if example:
                            examples.append(example.strip())
                    except Exception as e:
                        logger.warning(f"Error generating example {fn_name}: {e}")

        return "\n\n".join(examples)

    def computer_prompt_context(self) -> str:
        """
        Generate prompt context for ComputerPrimitives methods.

        Introspects the ComputerBackend abstract class to extract method
        signatures and docstrings, then formats them into markdown documentation
        similar to state manager prompt context.

        Returns:
            Markdown-formatted documentation for computer methods, or empty string
            if ComputerBackend cannot be loaded.
        """
        try:
            from unify.function_manager.computer_backends import ComputerBackend
            from unify.function_manager.primitives.runtime import ComputerPrimitives
        except ImportError:
            logger.warning("Could not import ComputerBackend or ComputerPrimitives")
            return ""

        method_names = ComputerPrimitives._PRIMITIVE_METHODS

        lines = ["### Computer Method Reference\n"]
        lines.append(
            "These methods are available on `primitives.computer.desktop.*` "
            "(singleton desktop) and on session handles returned by "
            "`primitives.computer.web.new_session()`.\n",
        )

        for method_name in method_names:
            if hasattr(ComputerBackend, method_name):
                source_cls = ComputerBackend
            else:
                source_cls = ComputerPrimitives

            sig_str = self._format_method_signature(source_cls, method_name)
            full_doc = self._extract_method_docstring(source_cls, method_name)
            full_doc = self._filter_internal_params_from_docstring(full_doc)
            lines.append(f"\n**`.{method_name}{sig_str}`**")
            if full_doc:
                for doc_line in full_doc.splitlines():
                    lines.append(f"  {doc_line}")

        return "\n".join(lines)

    def computer_prompt_examples(self) -> str:
        """
        Get concatenated examples for computer methods.

        Delegates to the get_computer_examples() function in prompt_examples.py,
        which returns pre-defined examples for computer primitives usage.

        Returns:
            Formatted examples string from prompt_examples.py, or empty string
            if the module cannot be imported or the function fails.
        """
        try:
            from unify.actor.prompt_examples import get_computer_examples
        except ImportError:
            logger.warning("Could not import prompt_examples module")
            return ""

        try:
            return get_computer_examples()
        except Exception as e:
            logger.warning(f"Error generating computer examples: {e}")
            return ""

    def _get_method_metadata(
        self,
        cls: Type,
        method_name: str,
        class_name: str,
        manager_alias: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Extract metadata (signature, docstring) from a class method.

        Handles functools.wraps by looking for __wrapped__ attribute.
        For ComputerPrimitives dynamic methods, looks up docstrings from backend classes.

        Args:
            cls: The class containing the method.
            method_name: Name of the method to introspect.
            class_name: Short class name (used for stable ID generation).
            manager_alias: The canonical alias (e.g. "contacts") for
                building the qualified name ``primitives.{alias}.{method}``.

        Returns:
            Primitive metadata dict, or None if method not found.
        """
        method = getattr(cls, method_name, None)

        # ComputerPrimitives methods live on the backend/session, not on the
        # class itself.  Look up docstrings from the ComputerBackend ABC.
        if method is None and class_name == "ComputerPrimitives":
            if method_name in cls._PRIMITIVE_METHODS:
                from unify.function_manager.computer_backends import ComputerBackend

                backend_method = getattr(ComputerBackend, method_name, None)
                if backend_method:
                    docstring = inspect.getdoc(backend_method) or ""
                    try:
                        signature = str(inspect.signature(backend_method))
                    except (ValueError, TypeError):
                        signature = "(...)"
                else:
                    docstring = ""
                    signature = "(...)"

                qualified_name = f"primitives.{manager_alias}.{method_name}"
                return {
                    "name": qualified_name,
                    "function_id": _get_stable_id(class_name, method_name),
                    "argspec": signature,
                    "docstring": docstring,
                    "embedding_text": (
                        f"Function Name: {qualified_name}\n"
                        f"Signature: {signature}\n"
                        f"Docstring: {docstring}"
                    ),
                    "implementation": None,
                    "is_primitive": True,
                    "depends_on": [],
                    "precondition": None,
                    "verify": False,
                    "guidance_ids": [],
                    "primitive_class": cls.__module__ + "." + cls.__name__,
                    "primitive_method": method_name,
                }
            return None

        if method is None:
            return None

        # Unwrap functools.wraps to get original function metadata
        fn = method
        while hasattr(fn, "__wrapped__"):
            fn = fn.__wrapped__

        qualified_name = f"primitives.{manager_alias}.{method_name}"

        try:
            signature = str(inspect.signature(fn))
        except (ValueError, TypeError):
            signature = "(...)"

        docstring = inspect.getdoc(fn) or ""

        return {
            "name": qualified_name,
            "function_id": _get_stable_id(class_name, method_name),
            "argspec": signature,
            "docstring": docstring,
            "embedding_text": (
                f"Function Name: {qualified_name}\n"
                f"Signature: {signature}\n"
                f"Docstring: {docstring}"
            ),
            "implementation": None,
            "is_primitive": True,
            "depends_on": [],
            "precondition": None,
            "verify": False,
            "guidance_ids": [],
            "primitive_class": cls.__module__ + "." + cls.__name__,
            "primitive_method": method_name,
        }

    def collect_primitives(
        self,
        primitive_scope: Optional[PrimitiveScope] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Introspect primitive classes and return their metadata.

        Auto-discovers primitive methods from each class by inspecting @abstractmethod
        definitions on base classes (minus per-class exclusions), then extracts
        signature and docstring information.

        Each primitive receives a stable `function_id` derived from a hash of its
        fully-qualified name (e.g., "ContactManager.ask"). This ensures IDs are:
        - Deterministic across runs
        - Stable when methods are added/removed (no positional dependencies)
        - Consistent across all deployments

        Args:
            primitive_scope: Optional scope to filter managers. If None, collects all
                           primitives (state managers and ComputerPrimitives).

        Returns:
            Dict mapping qualified_name (e.g. "primitives.contacts.ask") to primitive
            metadata suitable for insertion into the Functions/Primitives context.
        """
        primitives: Dict[str, Dict[str, Any]] = {}

        # Determine which specs to process
        if primitive_scope is None:
            specs_to_process = _MANAGER_SPECS
        else:
            specs_to_process = [
                spec
                for spec in _MANAGER_SPECS
                if spec.manager_alias in primitive_scope.scoped_managers
            ]

        for spec in specs_to_process:
            cls = self._load_manager_class(spec.primitive_class_path)
            if cls is None:
                continue

            class_name = cls.__name__
            method_names = self.primitive_methods(manager_alias=spec.manager_alias)

            for method_name in method_names:
                metadata = self._get_method_metadata(
                    cls,
                    method_name,
                    class_name,
                    spec.manager_alias,
                )
                if metadata is not None:
                    primitives[metadata["name"]] = metadata

        logger.debug(f"Collected {len(primitives)} primitives")
        return primitives

    def compute_primitives_hash(
        self,
        primitives: Optional[Dict[str, Dict[str, Any]]] = None,
        primitive_scope: Optional[PrimitiveScope] = None,
    ) -> str:
        """
        Compute a stable hash of all primitive signatures.

        Used to detect when primitives have changed (docstrings updated, methods
        added/removed) and a sync is needed.

        Args:
            primitives: Optional pre-collected primitives dict. If provided, uses these
                       directly without re-collecting.
            primitive_scope: Optional scope to filter managers. Only used if primitives
                           is not provided.

        Returns:
            16-character hex hash string.
        """
        if primitives is None:
            primitives = self.collect_primitives(primitive_scope)

        return stable_hash_for_rows(
            primitives.values(),
            fields=("name", "argspec", "docstring"),
            digest_chars=16,
            projection="delimited",
        )

    def compute_hash_for_manager(self, manager_alias: str) -> str:
        """
        Compute hash for a single manager's primitives.

        Args:
            manager_alias: The manager to hash.

        Returns:
            16-character hex hash string.
        """
        scope = PrimitiveScope(scoped_managers=frozenset({manager_alias}))
        return self.compute_primitives_hash(primitive_scope=scope)

    def primitive_row_filter(self, primitive_scope: PrimitiveScope) -> str:
        """
        Build a Unify filter expression for scoped primitive queries.

        Uses primitive_class (which is already stored in Function model) to filter,
        avoiding the need for a separate primitive_manager field.

        Args:
            primitive_scope: The scope defining which managers to include.

        Returns:
            Filter expression using membership over primitive class paths.
        """
        # Collect class paths for scoped managers
        class_paths = []
        for alias in primitive_scope.scoped_managers:
            spec = _MANAGER_BY_ALIAS.get(alias)
            if spec:
                class_paths.append(spec.primitive_class_path)

        if not class_paths:
            return "False"
        class_list = ", ".join(json.dumps(cp) for cp in sorted(class_paths))
        return f"primitive_class in [{class_list}]"


# =============================================================================
# Module-level convenience functions (matching old API)
# =============================================================================


def collect_primitives() -> Dict[str, Dict[str, Any]]:
    """
    Introspect all registered primitive classes and return their metadata.

    This is a convenience function matching the old module-level API.
    Delegates to ToolSurfaceRegistry.collect_primitives().

    Returns:
        Dict mapping qualified_name (e.g. "primitives.contacts.ask") to primitive
        metadata suitable for insertion into the Functions/Primitives context.
    """
    return get_registry().collect_primitives()


def compute_primitives_hash(primitives: Dict[str, Dict[str, Any]]) -> str:
    """
    Compute a stable hash of all primitive signatures.

    This is a convenience function matching the old module-level API.
    Delegates to ToolSurfaceRegistry.compute_primitives_hash().

    Args:
        primitives: Dict from collect_primitives().

    Returns:
        16-character hex hash string.
    """
    return get_registry().compute_primitives_hash(primitives=primitives)


def get_primitive_sources() -> List[tuple[Type, List[str]]]:
    """
    Get all primitive classes with their discovered method names.

    Returns a list of (class, method_names) tuples. This provides access to
    the auto-discovered primitives for code that needs to iterate over them.

    Returns:
        List of (class_object, [method_names]) tuples.
    """
    registry = get_registry()
    result = []
    for spec in _MANAGER_SPECS:
        cls = registry._load_manager_class(spec.primitive_class_path)
        if cls is not None:
            methods = registry.primitive_methods(manager_alias=spec.manager_alias)
            result.append((cls, methods))
    return result


# Module-level singleton for convenience
_registry_instance: Optional[ToolSurfaceRegistry] = None


def get_registry() -> ToolSurfaceRegistry:
    """Get the singleton registry instance."""
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = ToolSurfaceRegistry()
    return _registry_instance
