"""Localhost policy-enforcing proxy for Microsoft Graph and Google Drive.

The ``execute_code`` sandbox holds no provider OAuth token. Instead it points
its base URL at this proxy and authorizes with the per-process capability nonce.
The proxy runs in the trusted runtime, swaps the nonce for the real upstream
token, and enforces the per-assistant file-access allowlist:

- file listings/search: masked items are dropped from the response;
- direct reads of a masked item: 404 (indistinguishable from not-found);
- writes into/onto a masked location: 403;
- unrecognized file/drive endpoints: 403 (default-deny);
- everything non-file (calendar, mail, contacts, ...): passed straight through.

The proxy is started lazily, in-process, on a background thread bound to
loopback, and its base URLs + nonce are published to the sandbox environment.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import socket
import threading
import time
from typing import Any, Optional

import httpx
import uvicorn
from fastapi import FastAPI, Request
from starlette.responses import JSONResponse, Response

from unify.common import runtime_oauth
from unify.provider_proxy.ancestry import is_allowed, ms_get_by_path, parent_path
from unify.provider_proxy.classify import (
    KIND_BATCH,
    KIND_FILE_READ,
    KIND_FILE_WRITE,
    KIND_NON_FILE,
    KIND_UNKNOWN,
    Classification,
    Locator,
    classify,
)
from unify.provider_proxy.filter import filter_listing
from unify.provider_proxy.session import ProxySession, current_session, set_session

logger = logging.getLogger(__name__)

_UPSTREAM = {
    "microsoft": "https://graph.microsoft.com",
    "google": "https://www.googleapis.com",
}
_HOP_BY_HOP = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "transfer-encoding",
        "upgrade",
        "content-encoding",
        "content-length",
        "host",
    },
)


def _not_found() -> JSONResponse:
    return JSONResponse(
        status_code=404,
        content={"error": {"code": "itemNotFound", "message": "Item not found."}},
    )


def _forbidden(message: str) -> JSONResponse:
    return JSONResponse(
        status_code=403,
        content={"error": {"code": "accessDenied", "message": message}},
    )


def _base_headers(incoming: dict[str, str]) -> dict[str, str]:
    return {
        k: v
        for k, v in incoming.items()
        if k.lower() not in _HOP_BY_HOP and k.lower() != "authorization"
    }


def _make_client(follow_redirects: bool) -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=120, follow_redirects=follow_redirects)


def _reconnect_response(provider: str) -> httpx.Response:
    name = "Microsoft" if provider == "microsoft" else "Google"
    return httpx.Response(
        401,
        json={
            "error": {
                "code": "authenticationRequired",
                "message": (
                    f"The connected {name} account could not be authenticated "
                    "(token expired and could not be refreshed). Reconnect the "
                    "account from the Integrations tab and retry."
                ),
            },
        },
    )


async def _forward(
    provider: str,
    method: str,
    rest_path: str,
    query_string: str,
    incoming_headers: dict[str, str],
    body: Optional[bytes],
    *,
    follow_redirects: bool = False,
) -> httpx.Response:
    """Forward to the upstream provider, injecting the real token.

    Uses the token optimistically (no pre-emptive expiry gate) and, if the
    provider rejects it with 401, forces one refresh from Orchestra and retries
    once. A persistent 401 is normalized to a clean "reconnect account" 401 so
    stale-expiry metadata never surfaces as an opaque 500.
    """
    url = f"{_UPSTREAM[provider]}/{rest_path}"
    if query_string:
        url = f"{url}?{query_string}"
    base_headers = _base_headers(incoming_headers)

    token = runtime_oauth.get_provider_access_token_optimistic(provider)
    if not token:
        token = runtime_oauth.refresh_provider_access_token(provider)
    if not token:
        return _reconnect_response(provider)

    async with _make_client(follow_redirects) as http:
        resp = await http.request(
            method,
            url,
            headers={**base_headers, "Authorization": f"Bearer {token}"},
            content=body,
        )
        if resp.status_code == 401:
            new_token = runtime_oauth.refresh_provider_access_token(provider)
            if new_token and new_token != token:
                resp = await http.request(
                    method,
                    url,
                    headers={**base_headers, "Authorization": f"Bearer {new_token}"},
                    content=body,
                )
            if resp.status_code == 401:
                return _reconnect_response(provider)
    return resp


def _passthrough_response(resp: httpx.Response) -> Response:
    headers = {k: v for k, v in resp.headers.items() if k.lower() not in _HOP_BY_HOP}
    return Response(
        content=resp.content,
        status_code=resp.status_code,
        headers=headers,
        media_type=resp.headers.get("content-type"),
    )


def _rewriter(provider: str, session: ProxySession):
    upstream = _UPSTREAM[provider]
    proxy_root = session.provider_root(provider)

    def rewrite(url: str) -> str:
        if url.startswith(upstream):
            return proxy_root + url[len(upstream) :]
        return url

    return rewrite


def _write_destinations(
    c: Classification,
    query: dict[str, str],
    body_json: Optional[dict[str, Any]],
) -> list[Locator]:
    """Destination parents implied by a move/copy/create, for allow-checking."""
    dests: list[Locator] = []
    default_drive = c.target.drive_id if c.target else "my-drive"
    if c.provider == "microsoft" and isinstance(body_json, dict):
        ref = body_json.get("parentReference") or {}
        if ref.get("id"):
            dests.append(
                Locator(str(ref.get("driveId") or default_drive), str(ref["id"])),
            )
    if c.provider == "google":
        if isinstance(body_json, dict):
            for pid in body_json.get("parents") or []:
                dests.append(Locator(default_drive, str(pid)))
        for key in ("addParents", "parents"):
            raw = query.get(key)
            if raw:
                dests.extend(Locator(default_drive, pid) for pid in raw.split(","))
    return dests


async def _resolve_ids(
    provider: str,
    loc: Optional[Locator],
) -> tuple[Optional[tuple[str, str]], bool]:
    """Resolve a locator to concrete ``(drive_id, item_id)``.

    Returns ``(ids, existed)``. For id-addressed locators, ``existed`` is True
    (existence is enforced downstream by ancestry). For Graph path-addressed
    locators, the path is resolved via the provider; ``existed`` is False when
    the item does not exist (used to distinguish create-vs-edit).
    """
    if loc is None:
        return (None, False)
    if provider == "microsoft" and loc.is_path:
        node = await ms_get_by_path(loc.drive_id, loc.anchor_item_id, loc.path or "")
        if node is None:
            return (None, False)
        return ((node["drive_id"], node["item_id"]), True)
    return ((loc.drive_id, loc.item_id), True)


async def _resolve_parent_ids(
    provider: str,
    loc: Locator,
) -> Optional[tuple[str, str]]:
    """Resolve the parent folder of a (possibly nonexistent) path locator."""
    if provider != "microsoft" or not loc.is_path:
        return None
    node = await ms_get_by_path(
        loc.drive_id,
        loc.anchor_item_id,
        parent_path(loc.path or ""),
    )
    if node is None:
        return None
    return (node["drive_id"], node["item_id"])


async def _handle_write(
    c: Classification,
    query: dict[str, str],
    body: Optional[bytes],
) -> Optional[JSONResponse]:
    """Return a deny response if the write touches a masked location, else None.

    An allowed folder always accepts new files (create checks the parent
    folder); editing an existing item checks that item (so an explicitly-masked
    sensitive file stays blocked for writes too).
    """
    if c.parent is not None:
        pids, _ = await _resolve_ids(c.provider, c.parent)
        if pids is not None and not await is_allowed(c.provider, *pids):
            return _forbidden("Destination folder is not accessible.")
    if c.target is not None:
        tids, existed = await _resolve_ids(c.provider, c.target)
        if existed and tids is not None:
            if not await is_allowed(c.provider, *tids):
                return _not_found()
        else:
            # Creating a new item at a path: an allowed parent folder permits it.
            pids = await _resolve_parent_ids(c.provider, c.target)
            if pids is not None and not await is_allowed(c.provider, *pids):
                return _forbidden("Destination folder is not accessible.")
    body_json: Optional[dict[str, Any]] = None
    if body:
        try:
            body_json = json.loads(body)
        except (ValueError, TypeError):
            body_json = None
    for dest in _write_destinations(c, query, body_json):
        if not await is_allowed(c.provider, dest.drive_id, dest.item_id):
            return _forbidden("Destination folder is not accessible.")
    return None


async def _handle_batch(
    provider: str,
    incoming_headers: dict[str, str],
    body: Optional[bytes],
    session: ProxySession,
) -> Response:
    try:
        payload = json.loads(body or b"{}")
        requests = list(payload.get("requests") or [])
    except (ValueError, TypeError):
        return JSONResponse(status_code=400, content={"error": "invalid batch body"})

    synth: dict[str, dict[str, Any]] = {}
    forward: list[dict[str, Any]] = []
    listing_parent: dict[str, Optional[Locator]] = {}

    for sub in requests:
        sub_id = str(sub.get("id"))
        url = str(sub.get("url") or "")
        method = str(sub.get("method") or "GET").upper()
        rel = url.split("?", 1)
        path = rel[0].lstrip("/")
        q = _parse_query(rel[1] if len(rel) > 1 else "")
        c = classify(provider, method, path, q)

        if c.kind == KIND_UNKNOWN:
            synth[sub_id] = {"id": sub_id, "status": 403, "body": _forbidden("").body}
            continue
        if c.kind == KIND_FILE_WRITE:
            deny = await _handle_write(c, q, _encode_body(sub.get("body")))
            if deny is not None:
                synth[sub_id] = {"id": sub_id, "status": deny.status_code}
                continue
        elif c.kind == KIND_FILE_READ and c.target is not None:
            tids, existed = await _resolve_ids(provider, c.target)
            if existed and tids is not None and not await is_allowed(provider, *tids):
                synth[sub_id] = {"id": sub_id, "status": 404}
                continue
        if c.kind == KIND_FILE_READ and c.is_listing:
            parent_loc = c.parent
            if parent_loc is not None and parent_loc.is_path:
                pids, _ = await _resolve_ids(provider, parent_loc)
                parent_loc = Locator(pids[0], pids[1]) if pids is not None else None
            listing_parent[sub_id] = parent_loc
        forward.append(sub)

    upstream_by_id: dict[str, dict[str, Any]] = {}
    if forward:
        resp = await _forward(
            provider,
            "POST",
            "v1.0/$batch",
            "",
            incoming_headers,
            json.dumps({"requests": forward}).encode(),
        )
        try:
            for item in resp.json().get("responses") or []:
                upstream_by_id[str(item.get("id"))] = item
        except (ValueError, TypeError):
            pass

    rewrite = _rewriter(provider, session)
    responses: list[dict[str, Any]] = []
    for sub in requests:
        sub_id = str(sub.get("id"))
        if sub_id in synth:
            responses.append(synth[sub_id])
            continue
        item = upstream_by_id.get(sub_id)
        if item is None:
            responses.append({"id": sub_id, "status": 502})
            continue
        if sub_id in listing_parent and isinstance(item.get("body"), dict):
            item["body"] = await filter_listing(
                provider,
                item["body"],
                listing_parent[sub_id],
                rewrite,
            )
        responses.append(item)

    return JSONResponse(content={"responses": responses})


def _encode_body(body: Any) -> Optional[bytes]:
    if body is None:
        return None
    if isinstance(body, (bytes, bytearray)):
        return bytes(body)
    return json.dumps(body).encode()


def _parse_query(query_string: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for part in query_string.split("&"):
        if not part:
            continue
        key, _, value = part.partition("=")
        out[key] = value
    return out


async def _dispatch(provider: str, rest_path: str, request: Request) -> Response:
    session = current_session()
    if session is None:
        return JSONResponse(status_code=503, content={"error": "proxy not ready"})

    auth = request.headers.get("authorization", "")
    if auth != f"Bearer {session.nonce}":
        return JSONResponse(status_code=401, content={"error": "invalid proxy token"})

    if provider not in _UPSTREAM:
        return _not_found()

    method = request.method.upper()
    query = dict(request.query_params)
    query_string = request.url.query
    body = await request.body()
    incoming_headers = dict(request.headers)

    c = classify(provider, method, rest_path, query)

    if c.kind == KIND_NON_FILE:
        resp = await _forward(
            provider,
            method,
            rest_path,
            query_string,
            incoming_headers,
            body or None,
            follow_redirects=False,
        )
        return _passthrough_response(resp)

    if c.kind == KIND_UNKNOWN:
        return _forbidden(
            "This file/drive endpoint is not permitted through the proxy.",
        )

    if c.kind == KIND_BATCH:
        return await _handle_batch(provider, incoming_headers, body, session)

    if c.kind == KIND_FILE_WRITE:
        deny = await _handle_write(c, query, body or None)
        if deny is not None:
            return deny
        resp = await _forward(
            provider,
            method,
            rest_path,
            query_string,
            incoming_headers,
            body or None,
        )
        return _passthrough_response(resp)

    # KIND_FILE_READ
    if c.is_listing:
        # Resolve a path-addressed parent folder to concrete ids for filtering.
        parent_loc = c.parent
        if parent_loc is not None and parent_loc.is_path:
            pids, _ = await _resolve_ids(provider, parent_loc)
            parent_loc = Locator(pids[0], pids[1]) if pids is not None else None
        resp = await _forward(
            provider,
            method,
            rest_path,
            query_string,
            incoming_headers,
            None,
        )
        if resp.status_code >= 400:
            return _passthrough_response(resp)
        try:
            payload = resp.json()
        except (ValueError, TypeError):
            return _passthrough_response(resp)
        filtered = await filter_listing(
            provider,
            payload,
            parent_loc,
            _rewriter(provider, session),
        )
        return JSONResponse(content=filtered, status_code=resp.status_code)

    if c.target is not None:
        tids, existed = await _resolve_ids(provider, c.target)
        if existed and tids is not None and not await is_allowed(provider, *tids):
            return _not_found()
        resp = await _forward(
            provider,
            method,
            rest_path,
            query_string,
            incoming_headers,
            None,
            follow_redirects=c.is_content,
        )
        return _passthrough_response(resp)

    # Root/drive listings and drive metadata: no item to mask.
    resp = await _forward(
        provider,
        method,
        rest_path,
        query_string,
        incoming_headers,
        None,
    )
    return _passthrough_response(resp)


def build_app() -> FastAPI:
    app = FastAPI(title="unify-provider-proxy")

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.api_route(
        "/{provider}/{rest_path:path}",
        methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
    )
    async def handle(provider: str, rest_path: str, request: Request) -> Response:
        return await _dispatch(provider, rest_path, request)

    return app


# ── Lifecycle ────────────────────────────────────────────────────────────────

_START_LOCK = threading.Lock()
_SERVER: Optional[uvicorn.Server] = None


def _pick_port() -> int:
    configured = os.environ.get("WORKSPACE_PROXY_PORT")
    if configured and configured.isdigit():
        return int(configured)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def ensure_proxy_running() -> ProxySession:
    """Start the localhost proxy once and return its live session."""
    session = current_session()
    if session is not None:
        return session
    with _START_LOCK:
        session = current_session()
        if session is not None:
            return session

        global _SERVER
        host = "127.0.0.1"
        port = _pick_port()
        nonce = secrets.token_urlsafe(32)
        session = ProxySession(host=host, port=port, nonce=nonce)

        config = uvicorn.Config(
            build_app(),
            host=host,
            port=port,
            log_level="warning",
            access_log=False,
        )
        server = uvicorn.Server(config)
        thread = threading.Thread(
            target=server.run,
            name="provider-proxy",
            daemon=True,
        )
        thread.start()

        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            if getattr(server, "started", False):
                break
            time.sleep(0.05)
        if not getattr(server, "started", False):
            raise RuntimeError("provider proxy failed to start within 10s")

        _SERVER = server
        set_session(session)
        # Publish base URLs + nonce to the parent env so in-process actor code
        # and freshly spawned subprocesses can target the proxy.
        for key, value in session.sandbox_env().items():
            os.environ[key] = value
        # Defense-in-depth: ensure no raw provider token lingers in the parent
        # environment (e.g. from a stale .env) where in-process actor code could
        # read it. The real token is read from SecretManager's in-memory store,
        # not os.environ, so scrubbing here is safe.
        from unify.provider_proxy.session import provider_token_env_keys

        for key in provider_token_env_keys():
            os.environ.pop(key, None)
        logger.info("provider proxy listening on %s", session.root_url)
        return session
