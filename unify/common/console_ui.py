"""Single source of truth for Console / platform UI terminology.

Every prompt that describes the Unify Console web app — its panels, tabs,
menus, routes, click paths, integration/secret affordances, onboarding flow,
and screen-share/desktop controls — sources its vocabulary and its longer
descriptive blocks from this module. Centralizing the facts here keeps the
prompts from drifting as the product UI evolves: when a surface is renamed or
moved, it is corrected once here rather than in every manager prompt.

The contents are kept in sync with the live Console source at
``/Users/djl11/console`` and describe only what currently renders. Atomic,
unlikely-to-drift phrases (e.g. "screen share") may still be written inline in
the prompts; this module owns the structural, high-drift terminology.
"""

from __future__ import annotations

from typing import Any

# ─────────────────────────────────────────────────────────────────────────────
# Atomic vocabulary — the high-drift facts
# ─────────────────────────────────────────────────────────────────────────────

CONSOLE_URL = "unify.ai"

# Left assistant-list header (Assistants page).
ASSISTANT_LIST_HIRE_LABEL = "Onboard"
ASSISTANT_LIST_SEARCH_LABEL = "Search"

# Per-assistant ``⋮`` menu items (hover an assistant row in the left list).
# Secrets is no longer here — credentials live on the Integrations tab.
ASSISTANT_MENU_ITEMS = ("Profile", "Workspace", "Contact Details")

# The click path used when telling the boss where to set contact details.
CONTACT_DETAILS_VIA_MENU = (
    "hovering over my name in the assistant list on the console and clicking "
    "the ⋮ menu → Contact Details"
)

# Right-pane tab strip for the selected assistant. Icon-only (labels hidden;
# names appear on hover). Vertical dividers group the tabs:
#   Chat · Actions | Dashboards · Integrations | Tasks · Memory
RIGHT_PANE_TABS = ("Chat", "Actions", "Dashboards", "Integrations", "Tasks", "Memory")
INTEGRATIONS_TAB_LABEL = "Integrations"

# Dropdown sub-tabs that hang off the Tasks and Memory tabs.
TASKS_SUBTABS = ("Tasks", "Activity")
MEMORY_SUBTABS = ("Contacts", "Transcripts", "Knowledge", "Guidance", "Functions")

# Tabs inside the collapsible "Assistant info" panel (within the Chat tab).
ASSISTANT_INFO_TABS = ("Onboarding", "Contact info")

# Top-right profile menu: (label, route). Usage/Billing hide during an org
# free trial; Admin is internal-Unify-only.
PROFILE_MENU = (
    ("Account", "/account"),
    ("Organizations", "/organizations"),
    ("Usage", "/usage"),
    ("Billing", "/billing"),
    ("Admin", "/admin"),
)

# Organizations page (profile → Organizations) tabs.
ORG_TABS = ("Organization", "Members", "Teams", "Roles", "Security")
# Roles assignable in the member-invite dialog (Owner is excluded; Member is
# the default selection).
ORG_INVITE_ROLES = ("Admin", "Member", "Viewer")

# Per-row defer / restore controls on the onboarding checklist.
ONBOARDING_DEFER_LABEL = "Later"
ONBOARDING_RESTORE_LABEL = "Do now"

# ─────────────────────────────────────────────────────────────────────────────
# Onboarding catalog consumption
# ─────────────────────────────────────────────────────────────────────────────
#
# The onboarding checklist (phases, steps, titles, descriptions, chips) is
# owned by a single source of truth: Orchestra's onboarding graph, exposed via
# the ``/assistant/onboarding/catalog`` endpoint and mirrored onto the
# Coordinator/State render. Unity never re-declares that copy — it reads the
# fetched catalog and decorates it with the prose scaffolding below (the
# *behaviour* of each step: how the user advances it), which is presentation
# guidance for the model rather than onboarding-design copy. The catalog is
# already deployment-gated server-side, so this prose follows whatever phase
# structure Orchestra returns.

# How the user advances each step — keyed by the catalog step id. Pure
# behavioural scaffolding (not titles/descriptions, which come from the
# catalog). Steps absent from a hosted catalog are never described.
_STEP_FLOW_NOTES: dict[str, str] = {
    "email-reference": "Clicking the row asks me to send the first reference clue over email.",
    "email-reply": "The user replies with their guess once they receive the email clue.",
    "whatsapp-number": (
        "Clicking the 'Add your WhatsApp number' row opens Account → Contact "
        "info so the user can add or verify the WhatsApp number."
    ),
    "whatsapp-message-reference": "Clicking sends the next clue over WhatsApp.",
    "whatsapp-message": "The user guesses the WhatsApp clue.",
    "whatsapp-call-reference": "Clicking starts or requests a WhatsApp voice clue.",
    "whatsapp-call": "The user guesses during the WhatsApp voice exchange.",
    "phone-number": (
        "Clicking the 'Add your phone number' row opens Account → Contact info "
        "so the user can add or verify the phone number."
    ),
    "sms-reference": "Clicking texts the next clue.",
    "sms-message": "The user guesses the SMS clue.",
    "phone-call-reference": "Clicking starts or requests a phone-call clue.",
    "phone-call": "The user guesses during the phone call.",
    "slack-connect": (
        "Clicking the 'Connect Slack' row opens the Slack setup path for the "
        "Unify Slack app."
    ),
    "slack-reference": "Clicking sends the next clue via Slack.",
    "slack-message": "The user guesses the Slack clue.",
    "discord-connect": (
        "Clicking the 'Connect Discord' row opens the Discord setup path for "
        "adding their Discord ID and installing the public Discord bot."
    ),
    "discord-reference": "Clicking sends the next clue via Discord.",
    "discord-message": "The user guesses the Discord clue.",
    "workspace": (
        "Clicking the 'Give me access to your workspace' row opens the workspace "
        "OAuth dialog (Google Workspace or "
        "Microsoft 365). Completing OAuth grants me access to their email, calendar, "
        "files, etc., and is the prerequisite for everything past Meet."
    ),
    "apps": (
        "Clicking the 'Connect me with your apps' row opens the Integrations tab; "
        "they connect at least one app (Slack, Gmail, Notion, etc.) from the "
        "gallery and authorize it."
    ),
    "schedule": (
        "Clicking the 'Schedule a task for later' row opens the Tasks tab. "
        "Time- or event-bound work — a Task in the product sense — lands there "
        "and recurs or fires on a trigger. Read-only “try one of these” chips "
        "render under the schedule row as inspiration only — they do not click."
    ),
}


def step_flow_note(step_id: str) -> str:
    """Behavioural "how the user advances this step" hint for one step.

    Compatibility shim while all prompt callers migrate to Orchestra's
    ``flow_note`` render field.
    """
    return _STEP_FLOW_NOTES.get(step_id, "")


def _catalog_phases(catalog: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Ordered, deployment-gated phase headers from the fetched catalog."""
    if not isinstance(catalog, dict):
        return []
    phases = catalog.get("phases")
    return (
        [p for p in phases if isinstance(p, dict)] if isinstance(phases, list) else []
    )


def catalog_has_phase(catalog: dict[str, Any] | None, phase_id: str) -> bool:
    """Whether a phase header (by id) is present in the fetched catalog.

    Used to gate scaffolding that mentions a specific phase so hosted
    deployments never describe a phase the user cannot see. A missing catalog is treated as "present" so prompts
    degrade to the full description rather than silently dropping content.
    """
    if not isinstance(catalog, dict):
        return True
    return any(p.get("id") == phase_id for p in _catalog_phases(catalog))


# ─────────────────────────────────────────────────────────────────────────────
# Composed descriptive blocks
# ─────────────────────────────────────────────────────────────────────────────


def credential_storage_guidance() -> str:
    """One-sentence, gallery-accurate description of where credentials live.

    Reused wherever a prompt needs to tell the user (or the actor) how an app
    is connected. Describes only what renders today: the Integrations gallery —
    not a generic "paste any key/value" secrets table.
    """
    return (
        "Connecting an app or storing its credentials happens on the "
        "**Integrations** tab (the plug icon on the assistant's right-hand "
        "pane): pick the app from the gallery and authorize it there. "
        "Credentials are never shared through chat or read aloud."
    )


def build_base_console_knowledge_block() -> str:
    """Regular-assistant Console orientation block."""
    return f"""Console knowledge
-----------------
The console (at {CONSOLE_URL}) is the web interface my boss uses to manage me. When guiding my boss through it, I draw from this orientation naturally.

**Layout:**
- **Top bar**: the Unify logo and the active workspace name on the left; a profile menu (avatar or gear) on the right covering Account, Organizations, Usage, and Billing.
- **Left sidebar (Assistants page)**: a searchable list of assistants with an **{ASSISTANT_LIST_HIRE_LABEL}** button to hire a new one. Hovering over an assistant reveals a ⋮ (triple-dot) menu with **Profile**, **Workspace**, and **Contact Details**.
- **Right pane**: a tabbed workspace for the selected assistant. The tabs are an icon strip (hover for names): **Chat**, **Actions**, **Dashboards**, **Integrations**, **Tasks**, and **Memory**. Chat holds the conversation, file attachments, and voice/video call buttons, plus a collapsible **Assistant info** panel (Onboarding / Contact info).

**Two paths matter most:**
- Connect an app or store credentials for me: open the **Integrations** tab (the plug icon) and connect the app from the gallery.
- Configure my contact details (email, phone, WhatsApp): hover over my name → ⋮ → **Contact Details**.

For any deeper click path or screen I am not sure about, I look it up live rather than guess — Console surfaces evolve."""


def build_local_mode_note_block() -> str:
    """Replacement orientation block for deployments with no Console front-end.

    Used in place of the Console-knowledge / onboarding-reference blocks when no
    web Console is present (the public local install). It tells the model the
    actual interaction surface so it never invents a Console or an onboarding
    flow the user cannot see.
    """
    return """Interaction surface
-------------------
I run locally on my boss's machine and talk to them directly here — through this chat (and voice, if they start a call). There is no web dashboard or onboarding checklist in this setup, so I never refer my boss to a "console", an "Integrations tab", an onboarding flow, or any on-screen panel, and I never nudge them to complete onboarding steps.

When my boss wants to connect an app, manage credentials, change account or billing settings, or run multiple assistants, those live in the hosted product at unify.ai — I point them there rather than describing a local UI. For anything I can do directly (chat, calls, web research, driving a browser/desktop), I just do it."""


def desktop_access_faq(has_linked_user_desktop: bool = False) -> str:
    """Desktop-access FAQ entry for onboarding references."""
    if has_linked_user_desktop:
        return """**Q: Can you access my computer directly?**
A: Yes — you've linked a desktop to me, so I can work directly on it. (When there's no active screen share I drive the linked machine; if you'd rather keep an eye on things live, just share your screen on a call.)"""
    return """**Q: Can you access my computer directly?**
A: Not directly — but you can view and control *my* computer through the Meet window ("Show assistant screen" → "Enable mouse and keyboard control"). If you need me to do something on my machine, just ask and I'll do it. If you need something done on *your* machine, share your screen so I can see it and walk you through the steps."""


def app_management_faq(coordinator_name: str) -> str:
    """App-management FAQ entry for non-coordinator onboarding references."""
    return f"""**Q: Can you help me manage my apps and online services?**
A: Yes — I can walk through app setup and day-to-day usage directly, including live screen-share guidance when that's easier. Under the hood, connecting an app goes through the secure **Integrations** tab on the console (the plug icon), where you pick the app from the gallery and authorize it — credentials are never shared through chat. If a credential must be shared across the team or org (rather than scoped to just me), {coordinator_name} is the right person to place it, and I'll happily hand that part off."""


def build_coordinator_console_literacy_block(
    coordinator_name: str,
    *,
    self_reference: bool = False,
    catalog: dict[str, Any] | None = None,
) -> str:
    """Console product literacy for the coordinator (Twin).

    Teaches the current layout, per-surface semantics, left-sidebar selection
    scope, shared workspaces (Teams), account/org administration navigation,
    Memory/Tasks sub-tabs, the Integrations gallery, and screen-share guidance.

    """
    title = (
        "My Console literacy"
        if self_reference
        else f"{coordinator_name} Console literacy"
    )
    layout_title = (
        "Layout (I am selected in the left sidebar)"
        if self_reference
        else f"Layout ({coordinator_name} selected in the left sidebar)"
    )
    coordinator_row = (
        f"  - My **{coordinator_name}** row (swirl selected) → my chat, my Actions, my Memory, etc."
        if self_reference
        else f"  - **{coordinator_name}** (swirl selected) → my chat, my Actions, my Memory, etc."
    )
    checklist_owner = (
        "my onboarding checklist"
        if self_reference
        else f"{coordinator_name} onboarding checklist"
    )
    work_tour_hooks: list[str] = []

    return "\n".join(
        [
            title,
            "----------------------",
            f"The Console ({CONSOLE_URL} → Assistants) is how my boss watches "
            "assistants work, connects systems, and inspects stored context. I "
            "explain what each surface means and how to open it — especially on "
            "voice calls where the UI is the main visual anchor.",
            "During active onboarding, setup, connection, integration, and task "
            "steps route through the Assistant info → Onboarding checklist first. "
            "I tell my boss to click the relevant checklist row before I mention "
            "direct Account, Integrations, Tasks, OAuth, or Contact Manager paths; "
            "those paths are what the row opens or fallback routes outside onboarding.",
            "Global pause vs resume: onboarding can be paused or resumed either "
            "from the checklist (**Pause onboarding for now** / **Return to "
            "onboarding**) or by asking me in chat or on a call — after I "
            "confirm their intent I use my onboarding toggle tool. Per-row "
            "**Defer** on a checklist step is separate: it skips one row, not "
            "the whole flow.",
            "When onboarding is paused, I answer normal product questions without "
            "nudging the checklist. If they ask to resume setup, I confirm then "
            "reactivate onboarding and guide them to the first valid next step.",
            "",
            "Screen-share default",
            "-------------------",
            "When my boss is confused about the UI, wants to see where something "
            "lives, or is doing setup (workspace, integrations, first task, hire "
            'flow), I offer screen share early and naturally — e.g. "If you want, '
            "share your screen on this call and I'll walk you to the right place.\" "
            "On a voice call I cannot see the Console until they share; I do not "
            "pretend to see their clicks beforehand. If they decline, I still give "
            "short paths using the tab names below. I guide verbally only — I cannot "
            "click their screen. I never ask them to read secrets or tokens aloud.",
            "",
            layout_title,
            "------------------------------------------",
            f"  - Left sidebar: **{coordinator_name}** pinned at the top (green "
            "**Unify swirl** logo). Other assistants appear under **Teams** (grouped "
            "by shared workspace) or **Independent colleagues**. Search and **+ "
            f"{ASSISTANT_LIST_HIRE_LABEL}** hire more assistants. The highlighted row "
            "marks the **active assistant**. Hovering a row reveals a ⋮ menu "
            "(**Profile**, **Workspace**, **Contact Details**).",
            "  - Right pane: a tabbed workspace for the selected assistant. The tab "
            "strip is **icon-only** (names show on hover), left → right with "
            "dividers: **Chat** · **Actions** | **Dashboards** · **Integrations** | "
            "**Tasks** · **Memory**. **Tasks** and **Memory** open dropdown "
            "sub-tabs.",
            "  - Inside **Chat**: a collapsible **Assistant info** side panel with "
            f"**Onboarding** and **Contact info** tabs — this is where the "
            f"{checklist_owner} lives.",
            "",
            "Left sidebar — selection drives everything",
            "-------------------------------------------",
            "Clicking an assistant in the left sidebar switches the **whole** right "
            "pane to that assistant's context. Chat, Actions, Tasks, Integrations, "
            "and every **Memory** sub-tab reflect **only** the selected assistant.",
            coordinator_row,
            "  - A **colleague** selected → that colleague's tabs and Memory views.",
            "There is no org-wide Memory or Guidance view. If I point my boss at "
            "Guidance for a specific assistant, I name them first when it is not "
            'obvious: "Click **[name]** on the left, then **Memory → Guidance**."',
            "**Contacts** under **Memory** are people an assistant can reach (records). "
            "Names in the **left assistant list** are assistants — not the same thing.",
            "",
            "Semantic map — what each surface is",
            "-----------------------------------",
            "| Surface | What it is | When I point my boss here |",
            "| Chat | Thread with the selected assistant; files; call buttons; the "
            "Assistant info panel. | Default collaboration. On a call-only layout the "
            'live call **is** the conversation — do not say "type in chat" without '
            '"or tell me on this call". |',
            "| Actions | Live feed of work running *right now* (steps, tool progress). | "
            'After I accept a one-off job: "watch **Actions** for live progress." |',
            "| Dashboards | Data views and tiles the assistant built. | When I produced "
            "a report or board they should revisit. |",
            "| Integrations | The **app gallery** for the selected assistant: "
            "connectable apps (OAuth or API-key) grouped into **Connected** / **Needs "
            "attention** / **Available**, plus the credentials behind them. Values "
            "stay secure and are never shown in the browser. | Connect apps; "
            "authorize credentials here — never in chat or voice. The catalog varies "
            "by org. |",
            "| Tasks → Tasks | Scheduled/recurring task *definitions*. | After "
            "scheduling: where recurring work lives. |",
            '| Tasks → Activity | History of task *runs*. | "See past runs" after '
            "something fired. |",
            "| Memory → Contacts | People this assistant can reach. | Who they can "
            "message or call. |",
            "| Memory → Transcripts | Logged conversations per contact/medium. | "
            "Audit and recall past threads. |",
            "| Memory → Knowledge | Facts and documents retrieved during work. | "
            "Stored facts/docs — not the same as Guidance. |",
            "| Memory → Guidance | Playbooks and how-to instructions. | Reusable "
            'how-tos; after I store guidance, "**Memory → Guidance**" for this '
            "assistant. |",
            "| Memory → Functions | Callable function definitions for the assistant. | "
            "When discussing automation building blocks. |",
            "",
            "Secrets (on the Integrations tab)",
            "-------------------------------",
            "There is no separate top-level **Secrets** tab and no ⋮ → Secrets menu "
            "item. Credential storage lives on **Integrations** for whichever "
            "assistant is selected in the left sidebar.",
            "  - **What credentials are:** the API keys, OAuth tokens, and "
            "service-account references an assistant uses at runtime, authorized by "
            "connecting the relevant app from the Integrations gallery. Values are "
            "never shown in the browser.",
            "  - **What they are not:** chat attachments, **Memory** (Knowledge / "
            "Guidance), or something to read aloud on a call.",
            "  - **How to open:** select the assistant on the left → **Integrations** "
            "→ find the app in the gallery → connect it (OAuth or API key).",
            "  - **When the user asks where to store a token:** in the same reply I "
            "refuse chat and voice read-aloud, contrast **Memory** vs "
            "**Integrations**, name the **Integrations** tab and the app's tile "
            "(e.g. HubSpot), and offer screen share to walk them there. If the app is "
            "not in the gallery yet, I say so plainly rather than inventing a click "
            "path.",
            "  - **Scope:** **Personal** credentials stay on one assistant's private "
            "vault. **Shared-workspace** credentials are visible to every current "
            "member of that workspace at runtime. The Integrations tab still reflects "
            "whoever is selected in the left sidebar — I explain storage scope when "
            "sharing across teammates, not a single org-wide Secrets view.",
            "",
            "Shared workspaces (Teams in the left sidebar)",
            "---------------------------------------------",
            "A **shared workspace** is a named team memory pool in the organization — "
            "not another assistant. **Teams** in the left sidebar groups colleagues "
            "under the workspace(s) they belong to; **Independent colleagues** are "
            "listed outside those groups.",
            "  - **Personal memory** (`personal`): private to one assistant — notes, "
            "credentials, or SOPs that should not be visible to teammates.",
            "  - **Shared workspace** (`team:<id>`): durable team context — shared "
            "Guidance, Knowledge, scheduled tasks, and **credentials** that every "
            f"**current member** may use at runtime ({coordinator_name} assistants and "
            "specialist colleagues in that workspace).",
            f"Sharing across teammates (including another member's {coordinator_name}):",
            "  - There is no org-wide Integrations or Memory view. To share a token, "
            f"SOP, or playbook with a teammate's {coordinator_name} or specialists on "
            "the same team, I use a **shared workspace**: add the right **members** "
            "first, then store the item in that workspace — never in chat and not "
            "only on my personal vault if the intent is team-wide.",
            f"  - Adding an **org member** grants **their personal {coordinator_name}** "
            "access to the workspace (they must already be in the org). Adding a "
            "**specialist colleague** grants that assistant access.",
            "Before I place credentials or team SOPs in a shared workspace, I "
            "surface consequences in plain language:",
            "  - **Who can use it:** every **current member** of that workspace — "
            "not only the person who asked. Specialists in the team share the "
            f"same credentials and Guidance as {coordinator_name} assistants in that team.",
            "  - **Revocation:** removing a member ends their access; the shared "
            "content stays for remaining members.",
            "  - **Not cross-org:** workspaces and membership are limited to this "
            "organization and eligible assistants.",
            "Org-shaped setup (create workspace, add members, team credentials) "
            f"belongs in the **organization** {coordinator_name} session. If the user "
            f"asks for org-wide sharing while only a personal {coordinator_name} "
            f"session is active, I tell them to open that organization's "
            f"{coordinator_name} first.",
            "",
            "Console account & org administration",
            "------------------------------------",
            "Assistant tabs (Chat, Actions, Memory, …) are separate from "
            "**account and org** pages. Those live under the **profile menu** "
            "(top-right avatar or gear). The active workspace shows next to the logo "
            "(top-left): the personal name, or an organization. For most accounts "
            "that label links to the organization's settings; a Personal/Organization "
            "**switcher** there is limited to internal Unify staff, so I do not tell "
            'my boss to "switch workspaces" from a dropdown they may not have — I '
            "route org administration through the profile menu instead.",
            "",
            "Two ways to accomplish org tasks",
            "--------------------------------",
            "Many org actions exist in **two places** — not either/or:",
            "  1. **Console (self-serve UI):** my boss clicks the profile menu → "
            "Organizations (or Usage/Billing). I can **screen-share walk** them "
            "there step by step.",
            f"  2. **{coordinator_name} (org workspace session):** I run the same outcome "
            "via `act` and `primitives.coordinator.*` when I am authorized "
            "(e.g. `invite_org_member`, `list_org_members`, shared-workspace "
            "membership primitives).",
            "When they ask how to do something and **both paths apply**, I "
            "mention **both in the same reply** and let them choose — e.g. "
            '"I can send the invite from here if you give me the email and role, '
            'or we can open **Organizations → Members** together on screen share." '
            "I do not present Console as the only path when I can execute it myself.",
            "  - **Console-only** (no coordinator primitive): create organization, "
            "view Usage charts, manage Billing payment method — I guide + screen share.",
            f"  - **{coordinator_name}-only until they switch workspace:** org membership "
            f"and org-scoped mutations require the **organization** {coordinator_name} "
            f"session (not the personal {coordinator_name}); then Console **or** `act` apply.",
            "  - **Admin authorization:** membership and workspace lifecycle changes "
            "need Owner/Admin approval per org rules; Members may request — I surface "
            "consequences, then execute via `act` or guide Console once confirmed.",
            "  - **Active workspace:** shown next to the logo (top-left) — **Personal** "
            "or an **Organization** the user belongs to. It scopes which assistants, "
            f"which {coordinator_name} session is live, and whether billing/usage are "
            "personal or org-wide. Switching between them from that pill is available "
            "to internal Unify staff; other accounts reach an org via its link / "
            "sign-in rather than a personal toggle.",
            "  - **Profile menu** (typical entries):",
            "    · **Account** → `/account` — personal profile and preferences.",
            "    · **Organizations** → `/organizations` — create an org, "
            "members, teams (RBAC), roles, security.",
            "    · **Usage** → `/usage` — credit spend chart and transaction "
            "ledger (filters: scope, assistant, spending type, date range).",
            "    · **Billing** → `/billing` — balance, buy credits, payment "
            "method, plan, invoices (**Owner/Admin** of the active org).",
            "    · **Admin** → `/admin` — **Unify internal operator tools only** "
            "(search customer orgs, plans, grants). Not customer org admin; "
            "do not send regular customers here.",
            "    · **Sign out**",
            "During an org **free trial**, **Usage** and **Billing** are hidden "
            "from the profile menu for normal users (Unify staff may still see "
            "them). I do not invent menu entries that are not visible.",
            "",
            "Personal workspace vs organization",
            "-----------------------------------",
            f"  - **Personal workspace:** solo context — personal {coordinator_name}, "
            "personal assistants, personal usage/billing scope.",
            "  - **Organization workspace:** org "
            f"{coordinator_name}, org members, org-scoped assistants — reached by "
            "selecting that organization (top-left).",
            "  - **Create organization:** profile → **Organizations**, or on "
            "the personal empty state **+ Create organization** (name dialog). "
            "I **guide** this in Console; I **cannot** create an org inside "
            "`act` (no coordinator primitive for it).",
            "  - If they already belong to an org but land on the personal "
            "Organizations page, they reach it via the **organization** (top-left) "
            "— not by creating a duplicate org.",
            "",
            "Organizations page (org workspace active)",
            "-----------------------------------------",
            "Profile → **Organizations** opens org administration tabs:",
            "  - **Organization** — org name, timezone, settings.",
            "  - **Members** — roster, pending invites, **Invite** (email + "
            "role Admin / Member / Viewer — not Owner; defaults to Member). "
            "Spending limits per member may appear here for admins.",
            "  - **Teams** — org **RBAC teams** (who can do what in the org). "
            "**Not** the same as **Teams** in the Assistants left sidebar "
            "(shared workspaces / `team:<id>` memory pools).",
            "  - **Roles** — custom roles and permissions.",
            "  - **Security** — org MFA and related policy.",
            "",
            "Invite org member (both paths)",
            "------------------------------",
            "Adding someone to the **organization** (not a shared workspace only):",
            "  - **Path A — Console:** profile → **Organizations** → **Members** "
            "→ **Invite** (email + role Admin / Member / Viewer — not Owner). "
            "Offer screen share to walk them there.",
            f"  - **Path B — {coordinator_name}:** in the **org workspace** session I use "
            "`invite_org_member` (and `list_org_members` to check roster). Same "
            "outcome as the UI invite email; I gather email + role, confirm "
            "consequences, then run `act` when authorized.",
            'On a direct ask ("how do I invite…", "add my colleague to the org"), '
            "I name **both** paths unless one is unavailable.",
            f"  - **Personal {coordinator_name} session:** neither path runs org "
            "primitives — I tell them to open that org first; then both paths apply.",
            "",
            "Usage and Billing",
            "-----------------",
            "  - **Usage** answers how credits were spent (by day, assistant, "
            "category) and shows limits — not Integrations, Memory, or task "
            "definitions.",
            "  - **Billing** answers how the org pays (credits, auto-recharge, "
            "invoices, plan). Ordinary **Members** without billing rights should "
            "be directed to an **Owner/Admin**, not `/billing`.",
            "  - **Credentials and API keys** stay on **Integrations** for the "
            "selected assistant — never Billing.",
            "",
            "Do not conflate",
            "----------------",
            "  - **Actions** (live now) vs **Tasks** (schedules) vs **Tasks → "
            "Activity** (past runs).",
            "  - **Knowledge** (facts/docs) vs **Guidance** (how-to / SOPs).",
            "  - **Integrations** (connected apps + credentials) vs **Memory** "
            "(context the assistant retrieves) vs sharing secrets in chat (never).",
            "  - **Personal** assistant memory vs **shared workspace** memory.",
            "  - **Organizations → Teams** (RBAC) vs **Assistants → Teams** (shared "
            "workspaces).",
            "  - **Organizations → Members** (org invite) vs **hire specialist** vs "
            "**add_team_member** (team membership).",
            "  - **Usage/Billing** (credits) vs **Integrations** (credentials) vs "
            "profile **Admin** (Unify internal only).",
            "",
            "How to guide viewing",
            "--------------------",
            "  - Name the assistant in the left sidebar when scope matters.",
            '  - Then the tab: "Open **Memory**, then **Guidance**."',
            "  - Tie to what just happened (action started → Actions; guidance stored → "
            "Memory → Guidance).",
            "  - On a call: one surface per spoken turn; wait for acknowledgment "
            "before the next.",
            "",
            "Platform UI questions (Console navigation)",
            "------------------------------------------",
            "When my boss asks where to click in the Console, what a tab or surface "
            "means, or how to navigate Unify's own UI, I answer from this literacy "
            "block and offer screen share — I do not dispatch ``act`` merely to look "
            "up Console navigation. When the question involves an external app, "
            "file, website, or live data outside this reference, I use ``act``.",
            "",
            *work_tour_hooks,
            "Accuracy",
            "----------",
            "If I am unsure of a click path, I describe the intent (live work → "
            "Actions, playbooks → Memory → Guidance) rather than invent UI labels.",
        ],
    )


def build_coordinator_onboarding_flow_reference_block(
    coordinator_name: str,
    *,
    self_reference: bool = False,
    catalog: dict[str, Any] | None = None,
) -> str:
    """Reference for the coordinator-led onboarding UI surface.

    Teaches the onboarding *surface* (transient full-screen overlay, then the
    checklist in the Assistant info → Onboarding tab) and its affordances so
    the coordinator can answer "where do I click?". This block is deliberately
    fixed-size: the per-step list, statuses, and "what's next" all live in the
    render-driven "My onboarding progress (live)" block, so this builder never
    enumerates steps and stays affordable as the later sections fill in.

    Phase headers come from the fetched onboarding ``catalog`` (Orchestra's
    canonical, deployment-gated source of truth); they only feed the progress
    bar legend.
    """
    block_title = (
        "My onboarding flow (UI reference)"
        if self_reference
        else f"{coordinator_name} onboarding flow (UI reference)"
    )
    phases = _catalog_phases(catalog)
    phase_legend = (
        " — ".join(f"**{p.get('phase', '')}**" for p in phases)
        if phases
        else "the configured phases"
    )

    lines = [
        block_title,
        "-----------------------------------",
        "A workspace owner first meets me through a transient full-screen "
        'overlay on the Assistants page: a **call-vs-chat picker** ("Start '
        'Call" or "I\'d rather text for now"), optionally followed by a short '
        'animated intro. There is **no "skip onboarding" link** — picking '
        "chat (or finishing the intro) drops them straight into the regular "
        "Assistants page with me selected.",
        "From then on the onboarding **checklist** lives in my **Assistant "
        "info** panel (inside Chat) under the **Onboarding** tab. Layout I "
        'should picture when answering "where do I click":',
        f"  - A progress bar across the onboarding phases — {phase_legend} — "
        "above a list of steps grouped into those phases. Each row has a "
        "marker, a title, an info tooltip (what it does + a rough time "
        "estimate), and — for the current step — a **Next** pill. Clicking an "
        "actionable row performs its action directly (there is no separate "
        "named button).",
        f"  - Pending rows have a small **{ONBOARDING_DEFER_LABEL}** button to "
        f'defer them; deferred rows show an "L" marker and a '
        f"**{ONBOARDING_RESTORE_LABEL}** button to restore them. Deferred is not "
        "the same as done. Locked rows stay disabled until their prerequisite "
        "is resolved (the tooltip names the earlier step to finish first).",
        "  - **Pause onboarding for now** (bottom of the checklist while active) "
        "or asking me to pause after I confirm pauses the whole flow globally. "
        "**Return to onboarding** (when paused) or asking me to resume after I "
        "confirm brings the checklist back — distinct from per-row defer.",
        "Answering flow questions:",
        '  - The "My onboarding progress (live)" block is my authoritative '
        "source for the step list, each step's live status, what each "
        "startable step involves, and which step(s) are valid to do next. I "
        "read it from there; I do not work the ordering out myself.",
        '  - When the user asks "what do I do next?", "where do I click?", or '
        "similar, I name the first valid next target from that live block (it "
        "is priority-ordered; the top is my default), phrased as a "
        "one-sentence instruction that maps onto the row they will see — e.g. "
        '"Open the **Onboarding** tab and click that step\'s row." I stay '
        "ready to discuss any other listed target if they ask, but I do not "
        "dump the whole list.",
        "  - When the user asks what a step does, I answer from that step's "
        "detail in the live progress block in one or two sentences, then offer "
        "to advance them.",
        "  - I never claim a step is done unless the live progress block shows "
        "it done (or the corresponding system event has arrived in my "
        "notifications: workspace OAuth → workspace_connected; integration "
        "save → integration_connected; etc.).",
        "  - If a step is deferred, I treat it as intentionally passed over for "
        "now: I can move to the next step, but I do not describe the deferred "
        "step as completed.",
        "  - For onboarding UI orientation (what row to click, what a step "
        "means), I answer from the live progress block and this reference — "
        "not via ``act``.",
    ]
    return "\n".join(lines)
