"""
Runtime primitives interface for state managers.

This module provides:
- `ComputerPrimitives` - Computer use (web/desktop) control capabilities
- `Primitives` - Scoped runtime interface for accessing state manager primitives
- `_AsyncPrimitiveWrapper` - Async wrapper for sync managers

All manager configuration (aliases, excluded methods, class paths) is defined in
`unify.function_manager.primitives.registry`. This module only handles runtime instantiation
and async wrapping.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import threading
from typing import Any, Callable, Optional, TYPE_CHECKING

from unify.function_manager.primitives.scope import (
    PrimitiveScope,
    VALID_MANAGER_ALIASES,
    default_runtime_scope,
)
from unify.function_manager.primitives.registry import (
    get_registry,
    _CLASS_PATH_TO_ALIAS,
)
from unify.manager_registry import SingletonABCMeta
from unify.integrations.function_metadata import is_provider_backed_function

if TYPE_CHECKING:
    from unify.comms.primitives import CommsPrimitives
    from unify.function_manager.computer_backends import ComputerBackend
    from unify.contact_manager.contact_manager import ContactManager
    from unify.transcript_manager.transcript_manager import TranscriptManager
    from unify.knowledge_manager.knowledge_manager import KnowledgeManager
    from unify.task_scheduler.task_scheduler import TaskScheduler
    from unify.secret_manager.secret_manager import SecretManager
    from unify.web_searcher.web_searcher import WebSearcher

logger = logging.getLogger(__name__)


# =============================================================================
# ComputerPrimitives - Computer Use (Web/Desktop) Control
# =============================================================================


# Default agent-service URL for local development
DEFAULT_AGENT_SERVER_URL = "http://localhost:3000"

# Gate that blocks lazy MagnitudeBackend initialization until the managed VM is
# confirmed ready.  Set immediately for localhost / mock (no VM to wait for),
# or by _startup_sequence after log_job_startup confirms VM readiness.
_vm_ready = threading.Event()


_COMPUTER_METHODS = (
    "act",
    "observe",
    "query",
    "navigate",
    "get_links",
    "get_content",
    "get_screenshot",
    # Low-level actions (bypass LLM planning)
    "click",
    "double_click",
    "right_click",
    "drag",
    "scroll",
    "type_text",
    "press_enter",
    "press_tab",
    "press_backspace",
    "select_all",
    "press_key",
    "switch_tab",
    "close_tab",
    "new_tab",
    "go_back",
    "wait_for",
    "save_browser_state",
    "solve_captcha",
    "execute_actions",
)

_DESKTOP_METHODS = tuple(
    name for name in _COMPUTER_METHODS if name not in ("get_content", "solve_captcha")
)
_WEB_SESSION_METHODS = _COMPUTER_METHODS


def _publish_desktop_invoked(method_name: str) -> None:
    """Fire-and-forget EventBus publish for desktop primitive invocations."""
    try:
        from unify.events.event_bus import EVENT_BUS, Event

        asyncio.get_running_loop().create_task(
            EVENT_BUS.publish(
                Event(type="DesktopPrimitiveInvoked", payload={"method": method_name}),
            ),
        )
    except Exception:
        pass


def _publish_user_file_access(
    user_id: str,
    operation: str,
    path: str,
    dest: str = "",
) -> None:
    """Fire-and-forget audit publish for user-home filesystem access."""
    try:
        from unify.events.event_bus import EVENT_BUS, Event

        asyncio.get_running_loop().create_task(
            EVENT_BUS.publish(
                Event(
                    type="UserDesktopFileAccess",
                    payload={
                        "user_id": user_id,
                        "operation": operation,
                        "path": path,
                        "dest": dest,
                    },
                ),
            ),
        )
    except Exception:
        pass


def _publish_computer_act_completed(instruction: str, result: "ActResult") -> None:
    """Fire-and-forget EventBus publish when a visible session's act() completes."""
    try:
        from unify.events.event_bus import EVENT_BUS, Event

        asyncio.get_running_loop().create_task(
            EVENT_BUS.publish(
                Event(
                    type="ComputerActCompleted",
                    payload={
                        "instruction": instruction,
                        "summary": result.summary,
                    },
                ),
            ),
        )
    except Exception:
        pass


def _is_dead_session_error(e) -> bool:
    """Return True if the error indicates the session/browser is permanently gone."""
    from unify.function_manager.computer_backends import ComputerAgentError

    if not isinstance(e, ComputerAgentError):
        return False
    if e.error_type == "session_not_found":
        return True
    msg = (e.message or "").lower()
    return any(
        pattern in msg
        for pattern in (
            "browser has been closed",
            "context has been closed",
            "page has been closed",
        )
    )


def _make_session_method(
    method_name: str,
    owner: "ComputerPrimitives",
    session_resolver,
    *,
    mode: str = "",
    on_session_dead=None,
):
    """Build a wrapped async method that routes through a session.

    ``session_resolver`` is an async callable returning a ``ComputerSession``.
    Shared by ``_ComputerNamespace`` (lazy singleton) and ``WebSessionHandle``
    (bound instance).

    ``on_session_dead`` is an optional callback invoked when a request fails
    with a terminal session error (session removed, browser closed).
    """
    from unify.function_manager.computer_backends import ComputerSession

    is_desktop = mode == "desktop"

    if method_name == "get_screenshot":

        async def screenshot_wrapper(*args, **kwargs):
            kwargs.pop("_clarification_up_q", None)
            kwargs.pop("_clarification_down_q", None)
            if not _vm_ready.is_set():
                ready = await asyncio.to_thread(_vm_ready.wait, 300)
                if not ready:
                    raise RuntimeError(
                        "Managed VM did not become ready within 5 minutes",
                    )
            import base64, io
            from PIL import Image as _Image

            session = await session_resolver()
            try:
                b64 = await session.get_screenshot()
            except Exception as e:
                if on_session_dead and _is_dead_session_error(e):
                    on_session_dead()
                raise
            if is_desktop:
                _publish_desktop_invoked(method_name)
            return _Image.open(io.BytesIO(base64.b64decode(b64)))

        screenshot_wrapper.__name__ = method_name
        from unify.function_manager.computer_backends import ComputerBackend

        screenshot_wrapper.__doc__ = (
            getattr(ComputerBackend, method_name, None).__doc__
            or getattr(ComputerSession, method_name, None).__doc__
        )
        return screenshot_wrapper

    async def wrapper(*args, **kwargs):
        import time as _w_time
        import logging as _w_logging

        _w_t0 = _w_time.perf_counter()
        _w_log = _w_logging.getLogger("unify")

        def _w_ms():
            return f"{(_w_time.perf_counter() - _w_t0) * 1000:.0f}ms"

        _w_log.debug(
            f"⏱️ [desktop.{method_name} +{_w_ms()}] entered",
        )
        kwargs.pop("_clarification_up_q", None)
        kwargs.pop("_clarification_down_q", None)
        if not _vm_ready.is_set():
            _w_log.debug(f"⏱️ [desktop.{method_name} +{_w_ms()}] waiting for _vm_ready")
            ready = await asyncio.to_thread(_vm_ready.wait, 300)
            _w_log.debug(
                f"⏱️ [desktop.{method_name} +{_w_ms()}] _vm_ready resolved (ready={ready})",
            )
            if not ready:
                raise RuntimeError("Managed VM did not become ready within 5 minutes")
        else:
            _w_log.debug(f"⏱️ [desktop.{method_name} +{_w_ms()}] _vm_ready already set")
        if method_name in owner._SECRET_INJECTED_METHODS and args:
            _w_log.debug(f"⏱️ [desktop.{method_name} +{_w_ms()}] resolving secrets")
            resolved = await owner.secret_manager.from_placeholder(args[0])
            args = (resolved,) + args[1:]
            _w_log.debug(f"⏱️ [desktop.{method_name} +{_w_ms()}] secrets resolved")
        _w_log.debug(f"⏱️ [desktop.{method_name} +{_w_ms()}] session_resolver start")
        session = await session_resolver()
        _w_log.debug(
            f"⏱️ [desktop.{method_name} +{_w_ms()}] session resolved (id={getattr(session, '_session_id', '?')})",
        )
        _w_log.debug(
            f"⏱️ [desktop.{method_name} +{_w_ms()}] calling session.{method_name}",
        )
        try:
            result = await getattr(session, method_name)(*args, **kwargs)
        except Exception as e:
            if on_session_dead and _is_dead_session_error(e):
                on_session_dead()
            raise
        _w_log.debug(
            f"⏱️ [desktop.{method_name} +{_w_ms()}] session.{method_name} returned",
        )
        if is_desktop:
            _publish_desktop_invoked(method_name)
        is_visible = getattr(session, "_mode", "") in ("desktop", "web-vm")
        if is_visible and method_name == "act":
            _publish_computer_act_completed(args[0] if args else "", result)
        return result

    wrapper.__name__ = method_name
    # Prefer the rich docstrings from ComputerBackend ABC over ComputerSession's terse ones
    from unify.function_manager.computer_backends import ComputerBackend

    wrapper.__doc__ = (
        getattr(ComputerBackend, method_name, None).__doc__
        or getattr(ComputerSession, method_name, None).__doc__
    )
    return wrapper


def _make_user_desktop_method(
    method_name: str,
    owner: "ComputerPrimitives",
    session_resolver,
    user_id: str,
):
    """Build a wrapped async method routing to a *user's* linked desktop.

    Mirrors ``_make_session_method`` but targets a per-user agent-service
    backend instead of the managed VM.  Differences: it does not gate on the
    managed-VM readiness event (the user's machine has its own liveness,
    enforced by the tunnel), it never publishes desktop-invoked events (those
    drive the assistant's own live view), and any terminal connection error is
    routed through ``owner._handle_user_desktop_error`` so a dropped tunnel
    triggers a reconnect on the next call.
    """
    from unify.function_manager.computer_backends import (
        ComputerBackend,
        ComputerSession,
    )

    if method_name == "get_screenshot":

        async def screenshot_wrapper(*args, **kwargs):
            kwargs.pop("_clarification_up_q", None)
            kwargs.pop("_clarification_down_q", None)
            import base64, io
            from PIL import Image as _Image

            session = await session_resolver()
            try:
                b64 = await session.get_screenshot()
            except Exception as e:
                owner._handle_user_desktop_error(user_id, e)
                raise
            return _Image.open(io.BytesIO(base64.b64decode(b64)))

        screenshot_wrapper.__name__ = method_name
        screenshot_wrapper.__doc__ = (
            getattr(ComputerBackend, method_name, None).__doc__
            or getattr(ComputerSession, method_name, None).__doc__
        )
        return screenshot_wrapper

    async def wrapper(*args, **kwargs):
        kwargs.pop("_clarification_up_q", None)
        kwargs.pop("_clarification_down_q", None)
        if method_name in owner._SECRET_INJECTED_METHODS and args:
            resolved = await owner.secret_manager.from_placeholder(args[0])
            args = (resolved,) + args[1:]
        session = await session_resolver()
        try:
            return await getattr(session, method_name)(*args, **kwargs)
        except Exception as e:
            owner._handle_user_desktop_error(user_id, e)
            raise

    wrapper.__name__ = method_name
    wrapper.__doc__ = (
        getattr(ComputerBackend, method_name, None).__doc__
        or getattr(ComputerSession, method_name, None).__doc__
    )
    return wrapper


class _ComputerNamespace:
    """Thin wrapper that routes method calls to a lazily-created singleton session.

    Used for ``primitives.computer.desktop``.
    """

    def __init__(self, owner: "ComputerPrimitives", mode: str):
        self._owner = owner
        self._mode = mode

        async def _resolve():
            return await owner.backend.get_session(mode)

        methods = _DESKTOP_METHODS if mode == "desktop" else _WEB_SESSION_METHODS
        for name in methods:
            setattr(self, name, _make_session_method(name, owner, _resolve, mode=mode))


class WebSessionHandle:
    """Wrapped browser session returned by ``primitives.computer.web.new_session()``.

    Exposes ``act``, ``observe``, ``query``, ``navigate``, ``get_links``,
    ``get_content``, and ``get_screenshot`` plus ``stop()`` for explicit
    lifecycle management.  Unlike ``primitives.computer.desktop``, web
    sessions retain ``get_content()`` because they operate on real browser
    pages rather than the noVNC viewer surface.
    """

    def __init__(
        self,
        session: "ComputerSession",
        owner: "ComputerPrimitives",
        session_id: int,
    ):
        self._session = session
        self._owner = owner
        self._session_id = session_id
        self._agent_session_id: str = session._session_id
        self._label = f"Web {session_id}"
        self._active = True

        def _mark_inactive():
            self._active = False

        async def _resolve():
            return self._session

        for name in _WEB_SESSION_METHODS:
            setattr(
                self,
                name,
                _make_session_method(
                    name,
                    owner,
                    _resolve,
                    on_session_dead=_mark_inactive,
                ),
            )

    @property
    def session_id(self) -> int:
        """Numeric session identifier (0, 1, 2, ...)."""
        return self._session_id

    @property
    def label(self) -> str:
        """Human-readable label shown as a visual badge in the browser."""
        return self._label

    @property
    def visible(self) -> bool:
        """Whether this session renders on the VM desktop (``web-vm`` mode)."""
        return self._session._mode == "web-vm"

    @property
    def active(self) -> bool:
        """Whether this session is still running (``stop()`` has not been called)."""
        return self._active

    async def stop(self):
        """Stop the browser session and release resources."""
        self._active = False
        await self._session.stop()


class _WebSessionFactory:
    """Factory for independent browser sessions.

    Accessed as ``primitives.computer.web``.  Has no default session --
    every browser session is explicitly created via ``new_session()`` and
    must be stopped when no longer needed.
    """

    def __init__(self, owner: "ComputerPrimitives"):
        self._owner = owner
        self._handles: list[WebSessionHandle] = []
        self._next_id: int = 0

    async def new_session(
        self,
        visible: bool = True,
        *,
        storage_state_name: str | None = None,
    ) -> WebSessionHandle:
        """Create a new independent browser session.

        Each call spawns a fresh Chromium process with its own browsing
        context (cookies, storage, etc.).  Multiple sessions can run in
        parallel without interfering with each other.

        Parameters
        ----------
        visible : bool, default True
            If True, the browser window renders on the VM desktop (visible
            via noVNC) but is controlled entirely via CDP -- no mouse or
            keyboard involvement.  The user can un-minimize the window in
            noVNC to observe the session in real time.

            If False, the browser runs headless on the host machine for
            fast background lookups where visibility is unnecessary.
        storage_state_name : str, optional
            Name of a previously-saved browser-state file (created via
            ``session.save_browser_state(name)``). When provided, the new
            Chromium context boots with the persisted cookies +
            localStorage + sessionStorage already populated, so the
            session starts in an authenticated state. Only honoured for
            ``visible=False`` (i.e. ``mode='web'``); the web-vm path on
            the managed VM has its own auth flow.

        Returns
        -------
        WebSessionHandle
            Session handle with methods: ``act``, ``observe``, ``query``,
            ``navigate``, ``get_links``, ``get_content``, ``get_screenshot``,
            and ``stop``.  Call ``stop()`` when done to release resources.
        """
        sid = self._next_id
        self._next_id += 1
        label = f"Web {sid}"
        mode = "web-vm" if visible else "web"
        session = await self._owner.backend.create_session(
            mode,
            label=label,
            storage_state_name=storage_state_name,
        )
        handle = WebSessionHandle(session, self._owner, session_id=sid)
        self._handles.append(handle)
        return handle

    def list_sessions(
        self,
        visible_only: bool = False,
        active_only: bool = False,
    ) -> list[WebSessionHandle]:
        """List web sessions created across the entire system.

        Returns the actual ``WebSessionHandle`` objects from a global
        registry shared by all actors (``ComputerPrimitives`` is a
        singleton).  Use this for cross-actor coordination — e.g. to
        check how many browser sessions are currently active before
        creating new ones.

        Parameters
        ----------
        visible_only : bool, default False
            When True, only return sessions running on the VM desktop
            (``visible=True`` at creation time).  Excludes headless
            background sessions.
        active_only : bool, default False
            When True, only return sessions where ``stop()`` has not
            been called.

        Returns
        -------
        list[WebSessionHandle]
            Matching session handles.  Each handle can be used to call
            ``act``, ``observe``, ``navigate``, ``stop``, etc.
        """
        result = self._handles
        if visible_only:
            result = [h for h in result if h.visible]
        if active_only:
            result = [h for h in result if h.active]
        return list(result)

    def get_session(
        self,
        session_id: int,
        *,
        active_only: bool = True,
    ) -> WebSessionHandle:
        """Return an existing web session handle by numeric ID.

        This is the ergonomic reattachment API for browser sessions created in
        earlier turns or by other actors.  The ``session_id`` matches the
        handle's numeric ``session_id`` property and the IDs shown in
        ``<active_web_sessions>`` snapshots.

        Parameters
        ----------
        session_id : int
            Numeric session identifier (0, 1, 2, ...).
        active_only : bool, default True
            When True, only return sessions that have not been stopped.

        Returns
        -------
        WebSessionHandle
            The existing matching session handle.

        Raises
        ------
        ValueError
            If no matching session exists, or if the matching session is
            inactive while ``active_only`` is True.
        """
        for handle in self._handles:
            if handle.session_id != session_id:
                continue
            if active_only and not handle.active:
                raise ValueError(
                    f"Web session {session_id} exists but is inactive. "
                    "Use list_sessions(active_only=True) to discover reusable sessions.",
                )
            return handle

        raise ValueError(
            f"No web session with id {session_id}. "
            "Use list_sessions() to inspect the available handles.",
        )

    async def list_sessions_with_metadata(
        self,
        visible_only: bool = False,
        active_only: bool = False,
    ) -> list[dict]:
        """Like ``list_sessions`` but includes URL metadata for each session.

        Returns a list of dicts with keys ``handle``, ``url``, ``label``,
        and ``session_id``.
        """
        sessions = self.list_sessions(
            visible_only=visible_only,
            active_only=active_only,
        )
        result = []
        for h in sessions:
            url = ""
            try:
                url = await h._session.get_current_url()
            except Exception:
                pass
            result.append(
                {
                    "handle": h,
                    "session_id": h.session_id,
                    "label": h.label,
                    "url": url,
                },
            )
        return result


class UserDesktopHandle:
    """Control handle for a single user's linked local desktop.

    Returned by ``primitives.computer.user_desktop.session(user_id=...)``.
    Exposes the desktop method set (``act``, ``observe``, ``click``,
    ``type_text``, ``get_screenshot``, ...) routed to a dedicated
    agent-service backend connected to that user's machine over the reverse
    tunnel.  This is the user's *own* computer — not the assistant's managed
    VM and not the surface shown in the Console live view.  Every call
    re-checks that the user still consents to remote control, so a runtime
    revocation or a dropped tunnel fails fast rather than acting on a stale
    connection.
    """

    def __init__(
        self,
        owner: "ComputerPrimitives",
        user_id: str,
        link: Any,
    ):
        self._owner = owner
        self._user_id = user_id
        self._link = link
        self._label = f"UserDesktop {user_id}"

        async def _resolve():
            owner._assert_user_desktop_allowed(user_id)
            backend = owner._get_user_desktop_backend(link)
            return await backend.get_session("desktop")

        for name in _DESKTOP_METHODS:
            setattr(
                self,
                name,
                _make_user_desktop_method(name, owner, _resolve, user_id),
            )

    @property
    def user_id(self) -> str:
        """User who owns the machine this handle controls."""
        return self._user_id

    @property
    def os(self) -> str:
        """Operating system reported for the linked machine."""
        return getattr(self._link, "os", "")

    @property
    def label(self) -> str:
        """Human-readable label for this user-desktop connection."""
        return self._label


class _UserDesktopFilesNamespace:
    """On-demand access to a user's own home filesystem.

    Accessed as ``primitives.computer.user_desktop.files``.  This is the
    supported way to read, fetch, or "sync" a user's desktop files — do **not**
    copy their files in by hand with shell ``cp``/``scp``/``rclone``.  Unlike
    the live remote-control methods on a desktop session, this reads and writes
    individual paths in the user's home directory over SFTP, pulled only when
    asked.  Pulled files are staged under ``~/Unity/Remote/<user_id>/``, a
    read-only mirror of their home.  Writebacks never overwrite the user's
    originals — edited content is saved as a timestamped copy the user can
    review.  Requires the user to have enabled filesystem access for this
    assistant in the Console.
    """

    def __init__(self, owner: "ComputerPrimitives"):
        self._owner = owner

    async def _client(self, user_id: str | None) -> tuple[str, Any]:
        link = self._owner._resolve_user_desktop_link(user_id)
        target_uid = link.owner_user_id
        self._owner._assert_user_filesys_allowed(target_uid)
        if not getattr(link, "filesys_available", False):
            raise ValueError(
                f"Filesystem access is not enabled for user {target_uid!r}. "
                "Ask them to turn on filesystem access for this assistant in "
                "the Console.",
            )
        client = await self._owner._get_user_home_sftp(target_uid, link)
        return target_uid, client

    async def list(self, path: str = "", user_id: str | None = None) -> list[str]:
        """Browse a home-relative directory on the user's machine (no copy).

        This is the cheap, mounting-like way to explore their files: it reads
        only directory metadata over SFTP and copies nothing. Start here, then
        ``pull`` the specific entries whose contents you actually need.

        ``path`` is relative to the user's home (``""`` lists the home root).
        Returns names (directories carry a trailing ``/``). Pair with ``pull``
        to bring a single entry into the local mirror under
        ``~/Unity/Remote/<user_id>/``; do not shell out to copy them.
        """
        target_uid, client = await self._client(user_id)
        entries = await client.list_dir(path)
        _publish_user_file_access(target_uid, "list", path or "")
        return entries

    async def pull(self, path: str, user_id: str | None = None) -> str:
        """Fetch one file from the user's home into the local staging mirror.

        This is the default, on-demand way to access a user's desktop file:
        ``list`` to find it, then ``pull`` it when you need its contents or want
        to edit it. Do **not** copy it in by hand with shell
        ``cp``/``scp``/``rclone``.

        ``path`` is relative to the user's home.  The file is staged at
        ``~/Unity/Remote/<user_id>/<path>`` (a read-only mirror of their home)
        and that absolute local path is returned, ready to read or parse.
        Noise (caches, dependency trees, VCS metadata) and credential dirs
        (``.ssh``, ``.gnupg``, ``.aws``, …) are skipped, so pulling a directory
        won't drag in its dependencies or secrets; ``list`` still shows the
        full tree.
        """
        target_uid, client = await self._client(user_id)
        local_path = await client.pull(path)
        _publish_user_file_access(target_uid, "pull", path)
        return local_path

    async def sync(self, path: str = "", user_id: str | None = None) -> list[str]:
        """Bulk-mirror a whole home subtree into the local mirror at once.

        Reach for this only when you genuinely need *every* file under a
        subtree; otherwise prefer ``list`` + ``pull``, which fetch just what's
        needed and are far faster. Do **not** shell out
        (``find``/``cat``/``tar``/``rclone``) to copy their files.

        ``path`` is relative to the user's home. Scope it to a subtree (e.g.
        ``"Documents"``); ``""`` mirrors the entire home, which can be very
        large and slow (caches, dependency trees and credential dirs like
        ``.ssh``/``.gnupg``/``.aws`` are skipped, but it is still a heavy
        operation — confirm the user really wants everything). Every staged
        file lands under ``~/Unity/Remote/<user_id>/`` (a read-only mirror of
        their home); the list of absolute local paths is returned.
        """
        target_uid, client = await self._client(user_id)
        staged = await client.sync(path)
        _publish_user_file_access(target_uid, "sync", path or "")
        return staged

    async def push(
        self,
        local_path: str,
        dest_path: str,
        user_id: str | None = None,
    ) -> str:
        """Write a local file back to the user's home as a timestamped copy.

        ``dest_path`` is the home-relative path the content corresponds to.  The
        user's original is never overwritten; the new content is saved alongside
        it under a review folder.  Returns the remote path of the saved copy.
        """
        target_uid, client = await self._client(user_id)
        remote_dest = await client.push(local_path, dest_path)
        _publish_user_file_access(target_uid, "push", dest_path, dest=remote_dest)
        return remote_dest


class _UserDesktopFactory:
    """Namespace for controlling users' own linked local desktops.

    Accessed as ``primitives.computer.user_desktop``.  Distinct from
    ``primitives.computer.desktop`` (the assistant's managed VM, always shown
    in the Console live view): this drives a *user's* physical machine,
    exposed over a reverse tunnel and linked by that user in the Console.

    Default posture: operate your *own* desktop.  Only reach for a user's
    machine when that user has linked it **and** has explicitly asked you to
    act on it, and proceed with care — it is their personal computer.
    """

    def __init__(self, owner: "ComputerPrimitives"):
        self._owner = owner
        self._files: Optional[_UserDesktopFilesNamespace] = None

    @property
    def files(self) -> "_UserDesktopFilesNamespace":
        """On-demand access to the user's home filesystem (pull/push/list)."""
        if self._files is None:
            self._files = _UserDesktopFilesNamespace(self._owner)
        return self._files

    def list_linked(self) -> list[dict]:
        """List the user desktops linked to this assistant.

        Returns one dict per linked machine with keys ``user_id``, ``os``,
        ``filesys_sync`` (whether the user enabled home filesystem access), and
        ``filesys_available`` (whether that access is actually usable right now,
        i.e. the device has registered its SFTP tunnel).  Use this to discover
        whose machines are available before calling ``session()`` or
        ``files.pull()``.  An empty list means no user has linked a desktop and
        only the assistant's own desktop is controllable.

        This call is **synchronous** — do not ``await`` it::

            linked = primitives.computer.user_desktop.list_linked()  # sync
        """
        from unify.session_details import SESSION_DETAILS

        return [
            {
                "user_id": uid,
                "os": link.os,
                "filesys_sync": link.filesys_sync,
                "filesys_available": link.filesys_available,
            }
            for uid, link in SESSION_DETAILS.assistant.user_desktops.items()
        ]

    def session(self, user_id: str | None = None) -> "UserDesktopHandle":
        """Return a control handle for a user's linked local desktop.

        Parameters
        ----------
        user_id : str, optional
            Which user's machine to control.  Defaults to the session's
            primary user, falling back to the sole linked desktop when only
            one exists.  When several users have linked desktops it must be
            given explicitly — pass the acting user's id (surfaced in the
            system prompt) to target the person currently being helped.  Must
            match a desktop the user linked to this assistant in the Console.

        Returns
        -------
        UserDesktopHandle
            Handle exposing the desktop method set routed to the user's
            machine.

        Raises
        ------
        ValueError
            If no matching linked desktop exists (or the target is ambiguous
            across multiple linked users).
        PermissionError
            If the user has revoked live remote-control consent.

        Notes
        -----
        ``session()`` is **synchronous** and returns a handle immediately; the
        handle's methods are **async** and must be awaited::

            ud = primitives.computer.user_desktop.session(user_id=...)  # sync
            shot = await ud.get_screenshot()                            # async
        """
        link = self._owner._resolve_user_desktop_link(user_id)
        target_uid = link.owner_user_id
        self._owner._assert_user_desktop_allowed(target_uid)
        return UserDesktopHandle(self._owner, target_uid, link)


class ComputerPrimitives(metaclass=SingletonABCMeta):
    """Multi-mode computer control interface.

    Two attributes:

    - ``primitives.computer.desktop`` -- singleton namespace for full desktop
      control (mouse/keyboard via noVNC).  Methods: ``act``, ``observe``,
      ``query``, ``navigate``, ``get_links``, and ``get_screenshot``.
    - ``primitives.computer.web`` -- factory for independent browser sessions.
      Call ``new_session(visible=True/False)`` to create a session handle with
      the desktop method set plus ``get_content()`` and ``stop()``.  Use
      ``get_session(session_id)`` or ``list_sessions()`` to reattach to an
      existing browser session.

    Singleton via ``SingletonABCMeta`` / ``ManagerRegistry``.  All actors
    (including nested sub-agents) share the same backend connection.
    """

    _DYNAMIC_METHODS = (
        "act",
        "observe",
        "query",
        "navigate",
        "get_links",
        "get_content",
    )
    _LOW_LEVEL_METHODS = (
        "click",
        "double_click",
        "right_click",
        "drag",
        "scroll",
        "type_text",
        "press_enter",
        "press_tab",
        "press_backspace",
        "select_all",
        "press_key",
        "switch_tab",
        "close_tab",
        "new_tab",
        "go_back",
        "wait_for",
        "save_browser_state",
        "solve_captcha",
        "execute_actions",
    )
    _PRIMITIVE_METHODS = _DYNAMIC_METHODS + ("get_screenshot",) + _LOW_LEVEL_METHODS
    _SECRET_INJECTED_METHODS = frozenset({"act", "observe", "type_text"})

    @staticmethod
    def _resolve_container_url(explicit_url: str | None) -> str | None:
        """Resolve the agent-service container URL.

        Returns ``None`` when the managed VM hasn't been confirmed ready yet
        (``_vm_ready`` is unset), signalling that no URL is available.  Once
        ``_vm_ready`` is set the method returns the real desktop-derived URL
        or falls back to ``DEFAULT_AGENT_SERVER_URL`` for local/mock setups.
        """
        if explicit_url is not None and explicit_url != DEFAULT_AGENT_SERVER_URL:
            return explicit_url
        try:
            from unify.session_details import SESSION_DETAILS

            if SESSION_DETAILS.assistant.desktop_url:
                from urllib.parse import urlparse

                parsed = urlparse(SESSION_DETAILS.assistant.desktop_url)
                return f"{parsed.scheme}://{parsed.netloc}/api"
            if not _vm_ready.is_set():
                return None
        except Exception:
            pass
        return DEFAULT_AGENT_SERVER_URL

    def __init__(
        self,
        computer_mode: str = "magnitude",
        *,
        container_url: str | None = None,
        local_url: str | None = None,
        connect_now: bool = False,
        # Legacy compat: callers that pass agent_server_url get it mapped to container_url
        agent_server_url: str | None = None,
        **_kwargs,
    ):
        resolved_container = container_url or self._resolve_container_url(
            agent_server_url,
        )
        logger.info(
            f"ComputerPrimitives init: container_url={resolved_container}, mode={computer_mode}",
        )

        self._computer_mode = computer_mode
        self._computer_kwargs_map = {
            "magnitude": {
                "container_url": resolved_container,
                "local_url": local_url,
            },
            "mock": {},
        }

        self._secret_manager = None
        self._backend = None
        self._desktop_ns: Optional[_ComputerNamespace] = None
        self._web_factory: Optional[_WebSessionFactory] = None
        self._user_desktop_factory: Optional[_UserDesktopFactory] = None
        # Dedicated agent-service backends per linked user desktop, keyed by
        # the resolved ``scheme://netloc/api`` URL.
        self._user_desktop_backends: dict[str, Any] = {}
        # On-demand SFTP clients for users' home filesystems, keyed by user_id.
        self._user_home_sftp: dict[str, Any] = {}
        # Users who have revoked live remote-control of their own desktop.
        self._user_desktop_revoked: set[str] = set()
        # Users who have revoked live access to their own home filesystem. This
        # is a separate consent from remote-control: a user may permit one
        # without the other, and either can be withdrawn mid-session.
        self._user_filesys_revoked: set[str] = set()
        self._pending_url_mappings: dict[str, str] | None = None

        self._interject_queues: set[asyncio.Queue] = set()
        self._user_remote_control_active: bool = False

        if computer_mode == "mock":
            _vm_ready.set()

        if connect_now:
            _ = self.backend

    @property
    def secret_manager(self):
        if self._secret_manager is None:
            from unify.manager_registry import ManagerRegistry

            self._secret_manager = ManagerRegistry.get_secret_manager()
        return self._secret_manager

    @property
    def backend(self) -> "ComputerBackend":
        if self._backend is None:
            from unify.function_manager.computer_backends import (
                MagnitudeBackend,
                MockComputerBackend,
            )

            fresh_url = self._resolve_container_url(None)
            if fresh_url is not None and fresh_url != DEFAULT_AGENT_SERVER_URL:
                self._computer_kwargs_map["magnitude"]["container_url"] = fresh_url

            effective_url = self._computer_kwargs_map["magnitude"].get("container_url")
            logger.info(
                f"Creating {self._computer_mode} backend: fresh_url={fresh_url}, "
                f"container_url={effective_url}",
            )

            if self._computer_mode == "magnitude":
                self._backend = MagnitudeBackend(
                    **self._computer_kwargs_map["magnitude"],
                )
            elif self._computer_mode == "mock":
                self._backend = MockComputerBackend(
                    **self._computer_kwargs_map.get("mock", {}),
                )
            else:
                raise ValueError(f"Unknown computer_mode: '{self._computer_mode}'.")
            self._backend._on_session_closed = self._invalidate_web_session
            if self._pending_url_mappings is not None:
                self._backend._url_mappings = self._pending_url_mappings
        return self._backend

    @property
    def url_mappings(self) -> dict[str, str] | None:
        if self._backend is not None:
            return self._backend._url_mappings
        return self._pending_url_mappings

    @url_mappings.setter
    def url_mappings(self, mappings: dict[str, str] | None) -> None:
        self._pending_url_mappings = mappings
        if self._backend is not None:
            self._backend._url_mappings = mappings

    @staticmethod
    def mark_ready() -> None:
        """Signal that the agent-service is available and ready for requests.

        In production this is called by the VM startup sequence after the
        managed container boots.  Tests or sandboxes that manage their own
        agent-service should call this after the service is listening.
        """
        _vm_ready.set()

    def _invalidate_web_session(self, agent_session_id: str) -> None:
        """Mark the WebSessionHandle matching the agent-service UUID as inactive."""
        if self._web_factory is None:
            return
        for h in self._web_factory._handles:
            if h._agent_session_id == agent_session_id:
                h._active = False

    # ── Sub-namespace properties ─────────────────────────────────────────

    @property
    def desktop(self) -> _ComputerNamespace:
        """Desktop control namespace (mouse/keyboard via noVNC)."""
        if self._desktop_ns is None:
            self._desktop_ns = _ComputerNamespace(self, "desktop")
        return self._desktop_ns

    @property
    def web(self) -> _WebSessionFactory:
        """Factory for independent browser sessions.

        Call ``new_session(visible=True/False)`` to create a session, or
        ``get_session(session_id)`` / ``list_sessions()`` to reattach to an
        existing one.
        """
        if self._web_factory is None:
            self._web_factory = _WebSessionFactory(self)
        return self._web_factory

    @property
    def user_desktop(self) -> _UserDesktopFactory:
        """Namespace for controlling users' own linked local desktops.

        Distinct from ``desktop`` (the assistant's managed VM, always shown
        in the Console live view).  Default posture: operate your *own*
        desktop; only drive a user's machine when they have linked it and
        explicitly asked you to act on it.
        """
        if self._user_desktop_factory is None:
            self._user_desktop_factory = _UserDesktopFactory(self)
        return self._user_desktop_factory

    # ── User-desktop resolution / consent ────────────────────────────────

    _USER_DESKTOP_REVOKED_MSG = (
        "The user has revoked remote control of their own desktop. Stop any "
        "actions targeting their machine immediately. You may continue "
        "working on your own desktop. Do not retry user-desktop actions until "
        "the user explicitly re-enables control."
    )

    def _resolve_user_desktop_link(self, user_id: str | None = None) -> Any:
        """Resolve the ``UserDesktopLink`` for the target user.

        Defaults to the current session's primary user.  Falls back to the
        sole linked desktop when there is exactly one and no explicit user
        was requested.
        """
        from unify.session_details import SESSION_DETAILS

        desktops = SESSION_DETAILS.assistant.user_desktops
        if not desktops:
            raise ValueError(
                "No user desktop is linked to this assistant. The user must "
                "link their machine in the Console before it can be "
                "controlled.",
            )
        if user_id is not None:
            link = desktops.get(user_id)
            if link is None:
                raise ValueError(
                    f"No linked desktop for user {user_id!r}. "
                    f"Linked users: {sorted(desktops)}.",
                )
            return link
        session_uid = getattr(SESSION_DETAILS.user, "id", None)
        if session_uid and session_uid in desktops:
            return desktops[session_uid]
        if len(desktops) == 1:
            return next(iter(desktops.values()))
        raise ValueError(
            "Multiple users have linked desktops; specify which one via "
            "user_id. Pass the acting user's id (the person currently being "
            f"helped, surfaced in the system prompt). Linked users: "
            f"{sorted(desktops)}.",
        )

    def _get_user_desktop_backend(self, link: Any) -> "ComputerBackend":
        """Lazily create and cache a backend for a user's desktop tunnel."""
        from urllib.parse import urlparse

        from unify.function_manager.computer_backends import MagnitudeBackend

        parsed = urlparse(link.url)
        api_url = f"{parsed.scheme}://{parsed.netloc}/api"
        backend = self._user_desktop_backends.get(api_url)
        if backend is None:
            backend = MagnitudeBackend(container_url=api_url, local_url=None)
            self._user_desktop_backends[api_url] = backend
        return backend

    async def _get_user_home_sftp(self, user_id: str, link: Any) -> Any:
        """Lazily create and cache an on-demand SFTP client for a user's home.

        Gated on the link's standing ``filesys_sync`` flag and registered tunnel
        coordinates; the per-link private key is fetched on demand inside the
        client so it never reaches the pod env via ``user_desktops``.
        """
        from unify.file_manager.sync.user_sftp import UserHomeSFTP

        client = self._user_home_sftp.get(user_id)
        if client is None:
            client = UserHomeSFTP(user_id, link)
            if not await client.setup():
                raise RuntimeError(
                    f"Could not open the home filesystem for user {user_id!r}. "
                    "Ensure their device is online and filesystem access is "
                    "enabled in the Console.",
                )
            self._user_home_sftp[user_id] = client
        return client

    def _assert_user_desktop_allowed(self, user_id: str) -> None:
        """Raise if the user has revoked live remote-control of their desktop."""
        if user_id in self._user_desktop_revoked:
            raise PermissionError(
                f"User {user_id!r} has revoked live remote-control of their "
                "desktop. Do not attempt further actions on it until they "
                "re-enable control.",
            )

    def _assert_user_filesys_allowed(self, user_id: str) -> None:
        """Raise if the user has revoked live access to their home filesystem."""
        if user_id in self._user_filesys_revoked:
            raise PermissionError(
                f"User {user_id!r} has revoked access to their home "
                "filesystem. Do not attempt further reads or writebacks on it "
                "until they re-enable access.",
            )

    def _handle_user_desktop_error(self, user_id: str, e: Exception) -> None:
        """Drop cached user-desktop backends on a terminal connection error.

        A dropped tunnel surfaces as a dead-session error; clearing the cache
        forces a fresh connection attempt on the next call rather than
        repeatedly hitting a closed socket.
        """
        if _is_dead_session_error(e):
            self._user_desktop_backends.clear()
            for client in self._user_home_sftp.values():
                client.cleanup()
            self._user_home_sftp.clear()

    def revoke_user_desktop_control(
        self,
        user_id: str,
        conversation_context: str | None = None,
    ) -> None:
        """Revoke live remote-control of a user's desktop and notify actors.

        Called when a user withdraws consent mid-session (e.g. via a Console
        toggle).  Broadcasts an interjection so in-flight actors stop acting
        on that machine.  The standing Console link is unaffected — control
        can be re-enabled with ``grant_user_desktop_control``.
        """
        self._user_desktop_revoked.add(user_id)
        payload = self._build_interjection(
            self._USER_DESKTOP_REVOKED_MSG,
            conversation_context,
        )
        for q in self._interject_queues:
            q.put_nowait(payload)

    def grant_user_desktop_control(self, user_id: str) -> None:
        """Re-enable live remote-control of a user's desktop after a revoke."""
        self._user_desktop_revoked.discard(user_id)

    _USER_FILESYS_REVOKED_MSG = (
        "The user has revoked access to their home filesystem. Stop any reads "
        "or writebacks targeting their files immediately. You may continue "
        "other work. Do not retry filesystem actions on their machine until "
        "the user explicitly re-enables access."
    )

    def revoke_user_filesys_access(
        self,
        user_id: str,
        conversation_context: str | None = None,
    ) -> None:
        """Revoke live access to a user's home filesystem and notify actors.

        Called when a user withdraws filesystem consent mid-session (e.g. via a
        Console toggle).  Broadcasts an interjection so in-flight actors stop
        reading from or writing back to that machine, and drops any cached SFTP
        client so a later re-grant reconnects cleanly.  The standing Console
        link is unaffected — access can be re-enabled with
        ``grant_user_filesys_access``.
        """
        self._user_filesys_revoked.add(user_id)
        client = self._user_home_sftp.pop(user_id, None)
        if client is not None:
            client.cleanup()
        payload = self._build_interjection(
            self._USER_FILESYS_REVOKED_MSG,
            conversation_context,
        )
        for q in self._interject_queues:
            q.put_nowait(payload)

    def grant_user_filesys_access(self, user_id: str) -> None:
        """Re-enable live access to a user's home filesystem after a revoke."""
        self._user_filesys_revoked.discard(user_id)

    # ── Steering control (not exposed as actor tools) ────────────────────

    async def pause(self) -> None:
        """Pause the underlying browser agent's action loop.

        Called programmatically by the actor's steering mechanism when the
        CodeActActor is paused.  This is not an actor tool — it is never
        listed in ``_DYNAMIC_METHODS`` or ``_PRIMITIVE_METHODS``.
        """
        await self.backend.pause()

    async def resume(self) -> None:
        """Resume the underlying browser agent's action loop.

        Called programmatically by the actor's steering mechanism when the
        CodeActActor is resumed.
        """
        await self.backend.resume()

    # ── Interject queue registry ─────────────────────────────────────────

    _REMOTE_CONTROL_STARTED_MSG = (
        "The user has taken remote control of the desktop. They are now "
        "able to operate the mouse and keyboard directly, so the screen "
        "state may diverge from what you last observed. This is not cause "
        "for alarm — the user may be collaborating, demonstrating "
        "something, or performing a task themselves. Use your judgement: "
        "if you are unsure whether to proceed with computer actions, "
        "request clarification rather than guessing."
    )
    _REMOTE_CONTROL_STOPPED_MSG = (
        "The user has released remote control of the desktop. The screen "
        "state may have changed since your last observation. Before "
        "continuing any computer-related work, take a fresh screenshot "
        "to re-orient yourself."
    )

    @staticmethod
    def _build_interjection(base_msg: str, context: str | None) -> dict:
        msg = base_msg
        if context:
            msg = f"{base_msg}\n\nRecent conversation context:\n{context}"
        return {"message": msg}

    def register_interject_queue(self, queue: asyncio.Queue) -> None:
        """Register an actor's interject queue for environmental broadcasts.

        If the user currently has remote control, the queue immediately
        receives the corresponding interjection so late-arriving actors
        are informed of the current state.
        """
        self._interject_queues.add(queue)
        if self._user_remote_control_active:
            queue.put_nowait(
                self._build_interjection(self._REMOTE_CONTROL_STARTED_MSG, None),
            )

    def deregister_interject_queue(self, queue: asyncio.Queue) -> None:
        """Remove an actor's interject queue from the registry."""
        self._interject_queues.discard(queue)

    def set_user_remote_control(
        self,
        active: bool,
        conversation_context: str | None = None,
    ) -> None:
        """Update remote-control state and broadcast to all registered actors.

        Parameters
        ----------
        active:
            Whether the user currently has remote control.
        conversation_context:
            Optional snippet of recent conversation to include in the
            interjection, giving actors immediate context on *why* the
            user is taking or releasing control.
        """
        self._user_remote_control_active = active
        base = (
            self._REMOTE_CONTROL_STARTED_MSG
            if active
            else self._REMOTE_CONTROL_STOPPED_MSG
        )
        payload = self._build_interjection(base, conversation_context)
        for q in self._interject_queues:
            q.put_nowait(payload)


# =============================================================================
# Async Wrapper for Sync Managers
# =============================================================================


class _AsyncPrimitiveWrapper:
    """
    Wrapper that provides async versions of sync manager methods.

    Delegates to the original manager without modifying it,
    ensuring internal code using the manager synchronously continues to work.

    Uses asyncio.to_thread() for sync methods to avoid blocking the event loop.
    """

    def __init__(self, manager: Any, manager_alias: str):
        """
        Initialize the wrapper.

        Args:
            manager: The original sync manager instance.
            manager_alias: The manager alias to look up primitive methods.
        """
        object.__setattr__(self, "_wrapped_manager", manager)
        object.__setattr__(self, "_manager_alias", manager_alias)
        # Get primitive methods from registry
        registry = get_registry()
        object.__setattr__(
            self,
            "_primitive_methods",
            set(registry.primitive_methods(manager_alias=manager_alias)),
        )

    def __getattr__(self, name: str) -> Any:
        """
        Get an attribute - returns async wrapper for primitive methods, else delegates.
        """
        attr = getattr(self._wrapped_manager, name)

        # Only wrap methods that are in our primitive methods set
        if name not in self._primitive_methods:
            return attr

        # Non-callable attributes pass through directly
        if not callable(attr):
            return attr

        # Create async wrapper that uses to_thread for sync methods
        @functools.wraps(attr)
        async def async_method_wrapper(*args, **kwargs):
            if asyncio.iscoroutinefunction(attr):
                return await attr(*args, **kwargs)
            else:
                return await asyncio.to_thread(attr, *args, **kwargs)

        return async_method_wrapper


def _create_async_wrapper(manager: Any, manager_alias: str) -> _AsyncPrimitiveWrapper:
    """
    Create an async wrapper for a sync manager.

    Args:
        manager: The original sync manager instance.
        manager_alias: The manager alias for registry lookup.

    Returns:
        An async wrapper around the manager.
    """
    return _AsyncPrimitiveWrapper(manager, manager_alias)


# =============================================================================
# Manager Registry Key Mapping
# =============================================================================

# Maps manager_alias to ManagerRegistry getter method name.
# Empty string means direct construction (e.g. singleton via metaclass).
_ALIAS_TO_GETTER: dict[str, str] = {
    "comms": "",
    "contacts": "get_contact_manager",
    "dashboards": "get_dashboard_manager",
    "data": "get_data_manager",
    "transcripts": "get_transcript_manager",
    "knowledge": "get_knowledge_manager",
    "tasks": "get_task_scheduler",
    "secrets": "get_secret_manager",
    "web": "get_web_searcher",
    "files": "get_file_manager",
    "workspace_email": "",
    "integrations": "",
    "computer": "",
    "actor": "",
    "coordinator": "",
}

# Managers that need async wrapping (sync implementations)
_SYNC_MANAGERS: frozenset[str] = frozenset(
    {"dashboards", "data", "files", "coordinator"},
)


# =============================================================================
# Primitives Runtime Class (Scoped)
# =============================================================================


class Primitives:
    """
    Scoped runtime interface to all primitives (state managers and computer).

    Only managers in the provided `primitive_scope` are accessible.
    Attempting to access an out-of-scope manager raises AttributeError.

    Most managers are obtained via ManagerRegistry typed methods to respect
    IMPL settings (real vs simulated). ComputerPrimitives is constructed
    directly (singleton via metaclass).

    Sync managers (DataManager, FileManager) are wrapped with async interfaces
    for consistency - the LLM can safely use `await` on all primitives.

    Usage:
        scope = PrimitiveScope(scoped_managers=frozenset({"files", "contacts"}))
        primitives = Primitives(primitive_scope=scope)

        # Accessible:
        await primitives.files.describe(file_path="...")
        await primitives.contacts.ask(text="...")

        # Raises AttributeError:
        primitives.tasks  # not in scope
    """

    def __init__(self, *, primitive_scope: Optional[PrimitiveScope] = None) -> None:
        """
        Initialize primitives with the given scope.

        Args:
            primitive_scope: Defines which managers are accessible.
                           If None, uses role-gated default runtime scope.
        """
        self._primitive_scope = primitive_scope or default_runtime_scope()
        # Lazy-initialized manager instances
        self._managers: dict[str, Any] = {}

    @property
    def primitive_scope(self) -> PrimitiveScope:
        """The scope controlling which managers are accessible."""
        return self._primitive_scope

    def _get_manager(self, alias: str) -> Any:
        """
        Get or create a manager instance by alias.

        Raises AttributeError if alias is not in scope.
        """
        if alias not in self._primitive_scope.scoped_managers:
            available = sorted(self._primitive_scope.scoped_managers)
            raise AttributeError(
                f"primitives.{alias} is not available in this scope. "
                f"Available managers: {available}",
            )

        if alias in self._managers:
            return self._managers[alias]

        getter_name = _ALIAS_TO_GETTER.get(alias)
        if getter_name is None:
            raise AttributeError(f"Unknown manager alias: {alias}")

        if getter_name == "":
            # Direct construction via primitive_class_path from the registry.
            from unify.function_manager.primitives.registry import _MANAGER_BY_ALIAS

            spec = _MANAGER_BY_ALIAS.get(alias)
            if spec is None:
                raise AttributeError(f"No ManagerSpec for alias: {alias}")
            cls = get_registry()._load_manager_class(spec.primitive_class_path)
            if cls is None:
                raise AttributeError(
                    f"Could not load class for alias {alias!r}: "
                    f"{spec.primitive_class_path}",
                )
            manager = cls()
        else:
            from unify.manager_registry import ManagerRegistry

            getter = getattr(ManagerRegistry, getter_name)
            manager = getter()

        # Wrap sync managers with async interface
        if alias in _SYNC_MANAGERS:
            manager = _create_async_wrapper(manager, alias)

        self._managers[alias] = manager
        return manager

    def __getattr__(self, name: str) -> Any:
        """Attribute access for manager retrieval."""
        if name in VALID_MANAGER_ALIASES:
            return self._get_manager(name)

        raise AttributeError(f"'Primitives' object has no attribute '{name}'")

    # Convenience properties for type hints (IDE support)
    # These are optional and just provide better autocomplete

    @property
    def comms(self) -> "CommsPrimitives":
        """Assistant-owned communication primitives."""
        return self._get_manager("comms")

    @property
    def contacts(self) -> "ContactManager":
        """Contact management primitives (ask, update)."""
        return self._get_manager("contacts")

    @property
    def dashboards(self) -> "_AsyncPrimitiveWrapper":
        """Dashboard primitives (create_tile, create_dashboard, etc.)."""
        return self._get_manager("dashboards")

    @property
    def data(self) -> "_AsyncPrimitiveWrapper":
        """Data operations primitives (filter, search, reduce, join, etc.)."""
        return self._get_manager("data")

    @property
    def transcripts(self) -> "TranscriptManager":
        """Transcript management primitives (ask)."""
        return self._get_manager("transcripts")

    @property
    def knowledge(self) -> "KnowledgeManager":
        """Knowledge management primitives (ask, update, refactor)."""
        return self._get_manager("knowledge")

    @property
    def tasks(self) -> "TaskScheduler":
        """Task scheduling primitives (ask, update, execute)."""
        return self._get_manager("tasks")

    @property
    def secrets(self) -> "SecretManager":
        """Secret management primitives (ask, update)."""
        return self._get_manager("secrets")

    @property
    def web(self) -> "WebSearcher":
        """Web search primitives (ask)."""
        return self._get_manager("web")

    @property
    def files(self) -> "_AsyncPrimitiveWrapper":
        """File management primitives (describe, reduce, filter_files, etc.)."""
        return self._get_manager("files")

    @property
    def computer(self) -> "ComputerPrimitives":
        """Computer use primitives (act, navigate, observe, query, etc.)."""
        return self._get_manager("computer")

    @property
    def workspace_email(self) -> Any:
        """Connected workspace mailbox send/read (Gmail/Outlook impersonation)."""
        return self._get_manager("workspace_email")

    @property
    def integrations(self) -> Any:
        """Provider-backed SaaS app integrations."""
        return self._get_manager("integrations")

    @property
    def actor(self) -> Any:
        """Actor delegation primitives (run)."""
        return self._get_manager("actor")

    @property
    def coordinator(self) -> "_AsyncPrimitiveWrapper":
        """Coordinator admin primitives (assistant/team lifecycle)."""
        return self._get_manager("coordinator")


# =============================================================================
# Primitive Callable Resolution (for FunctionManager execution)
# =============================================================================


def get_primitive_callable(
    primitive_data: dict[str, Any],
    *,
    primitives: Optional[Primitives] = None,
) -> Optional[Callable]:
    """
    Resolve a primitive metadata dict to its actual callable.

    Uses the ``primitives`` instance (which handles all aliases uniformly)
    when available, falling back to a default-scoped Primitives instance.

    Args:
        primitive_data: Primitive metadata with primitive_class and primitive_method.
        primitives: Scoped Primitives instance for resolution.

    Returns:
        The callable method, or None if resolution fails.
    """
    class_path = primitive_data.get("primitive_class")
    method_name = primitive_data.get("primitive_method")

    if not class_path or not method_name:
        return None

    manager_alias = _CLASS_PATH_TO_ALIAS.get(class_path)
    if not manager_alias:
        return None

    # Use provided primitives instance, or construct a default-scoped one.
    if primitives is None:
        primitives = Primitives()

    manager = getattr(primitives, manager_alias, None)
    if manager is None:
        return None
    if manager_alias == "integrations" and is_provider_backed_function(primitive_data):
        return manager.callable_for_tool(primitive_data)
    return getattr(manager, method_name, None)
