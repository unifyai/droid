"""Runtime helpers for refresh-token backed OAuth credentials.

SecretManager owns storage and synchronization: it mirrors allowlisted assistant
secrets from Orchestra into the local ``Secrets`` context, ``.env``, and
``os.environ``.  This module owns the runtime interpretation of those mirrored
values: provider aliases, access-token/expiry secret names, freshness checks,
and the sandbox helper exposed to actor-written Python.

The split is deliberate and security-relevant. ``get_provider_access_token(...)``
returns the REAL bearer token and is TRUSTED-RUNTIME ONLY (the localhost
provider proxy and first-party managers). The sandbox-facing
``get_oauth_access_token(...)`` never returns a real token: it returns a local
capability handle (the proxy nonce) to use against the localhost proxy base
URLs. Raw provider tokens are kept out of ``os.environ``/``.env`` and the
``Secrets`` context so that neither subprocess nor in-process actor code can read
them and bypass the file-access allowlist; connected-provider REST is reached
only through the proxy, which injects the real token and enforces the allowlist.
"""

import inspect
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True)
class OAuthProviderMetadata:
    """Runtime metadata for a refresh-token backed OAuth provider."""

    canonical_name: str
    aliases: tuple[str, ...]
    access_token_secret: str
    refresh_token_secret: str | None = None
    expiry_secret: str | None = None
    granted_scopes_secret: str | None = None
    docs_label: str = ""

    @property
    def secret_names(self) -> frozenset[str]:
        return frozenset(
            name
            for name in (
                self.access_token_secret,
                self.refresh_token_secret,
                self.expiry_secret,
                self.granted_scopes_secret,
            )
            if name
        )


_OAUTH_PROVIDER_METADATA: dict[str, OAuthProviderMetadata] = {
    "google": OAuthProviderMetadata(
        canonical_name="google",
        aliases=("google", "gmail", "google_workspace", "drive"),
        access_token_secret="GOOGLE_ACCESS_TOKEN",
        refresh_token_secret="GOOGLE_REFRESH_TOKEN",
        expiry_secret="GOOGLE_TOKEN_EXPIRES_AT",
        granted_scopes_secret="GOOGLE_GRANTED_SCOPES",
        docs_label="Google APIs",
    ),
    "microsoft": OAuthProviderMetadata(
        canonical_name="microsoft",
        aliases=("microsoft", "msft", "ms365", "microsoft_365", "graph"),
        access_token_secret="MICROSOFT_ACCESS_TOKEN",
        refresh_token_secret="MICROSOFT_REFRESH_TOKEN",
        expiry_secret="MICROSOFT_TOKEN_EXPIRES_AT",
        granted_scopes_secret="MICROSOFT_GRANTED_SCOPES",
        docs_label="Microsoft Graph",
    ),
}
_OAUTH_PROVIDER_ALIASES: dict[str, str] = {
    alias.strip().lower().replace("-", "_"): metadata.canonical_name
    for metadata in _OAUTH_PROVIDER_METADATA.values()
    for alias in metadata.aliases
}


def _resolve_oauth_provider(provider: str) -> OAuthProviderMetadata:
    if not isinstance(provider, str) or not provider.strip():
        supported = ", ".join(sorted(_OAUTH_PROVIDER_METADATA))
        raise ValueError(
            "A refresh-token OAuth provider name is required. "
            f"Supported providers: {supported}",
        )
    normalized = provider.strip().lower().replace("-", "_")
    canonical = _OAUTH_PROVIDER_ALIASES.get(normalized, normalized)
    metadata = _OAUTH_PROVIDER_METADATA.get(canonical)
    if metadata is None:
        supported = ", ".join(sorted(_OAUTH_PROVIDER_METADATA))
        raise ValueError(
            f"Unknown refresh-token OAuth provider {provider!r}. "
            f"Supported providers: {supported}",
        )
    return metadata


def refresh_token_oauth_secret_names() -> frozenset[str]:
    names: set[str] = set()
    for metadata in _OAUTH_PROVIDER_METADATA.values():
        names.update(metadata.secret_names)
    return frozenset(names)


def refresh_token_oauth_token_names() -> frozenset[str]:
    """Return only the raw access/refresh token secret names (sensitive subset).

    These must never be mirrored to the Secrets context, ``.env`` or
    ``os.environ``; they are held in SecretManager's in-memory OAuth store.
    """
    names: set[str] = set()
    for metadata in _OAUTH_PROVIDER_METADATA.values():
        names.add(metadata.access_token_secret)
        if metadata.refresh_token_secret:
            names.add(metadata.refresh_token_secret)
    return frozenset(names)


def _get_secret_manager():
    from unify.manager_registry import ManagerRegistry

    return ManagerRegistry.get_secret_manager()


def _get_secret_value(secret_manager, name: str) -> str | None:
    getter = getattr(secret_manager, "_get_secret_value", None)
    if callable(getter):
        value = getter(name)
        if isinstance(value, str) and value:
            return value
    value = os.environ.get(name)
    return value if value else None


def _parse_expiry(value: str) -> datetime:
    if value.isdigit():
        return datetime.fromtimestamp(int(value), tz=timezone.utc)
    normalized = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _token_expires_within(
    secret_manager,
    metadata: OAuthProviderMetadata,
    min_ttl_seconds: int,
) -> bool:
    if metadata.expiry_secret is None:
        return False
    expiry_value = _get_secret_value(secret_manager, metadata.expiry_secret)
    if not expiry_value:
        return True
    try:
        expiry = _parse_expiry(expiry_value)
    except ValueError:
        return True
    remaining = (expiry - datetime.now(timezone.utc)).total_seconds()
    return remaining <= min_ttl_seconds


def _read_access_token(secret_manager: Any, name: str) -> str | None:
    """Read a real access token, preferring the in-memory OAuth store.

    Raw provider tokens are deliberately kept out of the ``Secrets`` context,
    ``.env`` and ``os.environ`` so sandboxed actor code cannot read them. They
    live only in SecretManager's in-memory OAuth store; fall back to the legacy
    secret lookup for environments/tests that still populate it.
    """
    getter = getattr(secret_manager, "get_oauth_token", None)
    if callable(getter):
        value = getter(name)
        if isinstance(value, str) and value:
            return value
    return _get_secret_value(secret_manager, name)


def get_provider_access_token(provider: str, *, min_ttl_seconds: int = 300) -> str:
    """Return a current REAL OAuth access token for a refresh-token provider.

    TRUSTED-RUNTIME ONLY. This returns the actual bearer token and must never be
    exposed to the ``execute_code`` sandbox. It is used by the localhost
    provider proxy and by first-party managers (e.g. workspace email) that run
    in the trusted parent process. Sandboxed code uses
    :func:`get_oauth_access_token`, which returns a local capability handle.

    If the current token is missing or expires within ``min_ttl_seconds``, an
    assistant-secret sync from Orchestra is forced before returning a token.
    """
    metadata = _resolve_oauth_provider(provider)
    secret_manager = _get_secret_manager()
    token = _read_access_token(secret_manager, metadata.access_token_secret)
    needs_force_sync = token is None or _token_expires_within(
        secret_manager,
        metadata,
        min_ttl_seconds,
    )
    secret_manager.sync_assistant_secrets_if_stale(
        ttl_seconds=60.0,
        force=needs_force_sync,
        reason=f"oauth_access_token:{metadata.canonical_name}",
    )
    token = _read_access_token(secret_manager, metadata.access_token_secret)
    if not token:
        raise ValueError(
            f"No access token is available for refresh-token OAuth provider "
            f"{metadata.canonical_name!r}.",
        )
    if _token_expires_within(secret_manager, metadata, min_ttl_seconds):
        raise ValueError(
            f"The access token for refresh-token OAuth provider "
            f"{metadata.canonical_name!r} is expired or near expiry after sync.",
        )
    return token


def get_provider_access_token_optimistic(provider: str) -> str | None:
    """Return the current access token WITHOUT a pre-emptive expiry gate.

    TRUSTED-RUNTIME ONLY. Unlike :func:`get_provider_access_token`, this does not
    refuse a token whose stored ``*_TOKEN_EXPIRES_AT`` looks stale/missing: it
    trusts the provider to reject a genuinely-expired token (the proxy then
    forces a refresh and retries once). This avoids blocking valid tokens on
    stale expiry metadata. Performs a debounced sync (forced only when no token
    is cached). Returns None if no token is available.
    """
    metadata = _resolve_oauth_provider(provider)
    secret_manager = _get_secret_manager()
    token = _read_access_token(secret_manager, metadata.access_token_secret)
    secret_manager.sync_assistant_secrets_if_stale(
        ttl_seconds=60.0,
        force=token is None,
        reason=f"oauth_optimistic:{metadata.canonical_name}",
    )
    return _read_access_token(secret_manager, metadata.access_token_secret)


def refresh_provider_access_token(provider: str) -> str | None:
    """Force a secret sync from Orchestra and return the freshest token, or None.

    TRUSTED-RUNTIME ONLY. Used by the proxy after the provider rejects a token
    with 401. This pulls whatever the platform refresh job has persisted to
    Orchestra; it does not itself call the provider's token endpoint, so it only
    recovers when Orchestra already holds a newer token (e.g. the 30-minute
    refresh cron has run). If it still returns a stale token, the proxy surfaces
    a clean "reconnect account" 401.
    """
    metadata = _resolve_oauth_provider(provider)
    secret_manager = _get_secret_manager()
    secret_manager.sync_assistant_secrets_if_stale(
        ttl_seconds=0.0,
        force=True,
        reason=f"oauth_refresh_on_401:{metadata.canonical_name}",
    )
    return _read_access_token(secret_manager, metadata.access_token_secret)


def get_oauth_access_token(provider: str, *, min_ttl_seconds: int = 300) -> str:
    """
    Authorize provider REST calls from ``execute_code`` via the local proxy.

    This does NOT return a raw provider access token. It returns a local
    capability handle (the workspace proxy nonce) to place in the
    ``Authorization: Bearer ...`` header. You must ALSO point your base URL at
    the local proxy so the request is authorized and policy-enforced:

    - Microsoft Graph: base URL ``os.environ["MICROSOFT_GRAPH_BASE"]`` (drop-in
      for ``https://graph.microsoft.com/v1.0``).
    - Google APIs: ``os.environ["GOOGLE_DRIVE_BASE"]`` (drop-in for
      ``https://www.googleapis.com/drive/v3``) or ``GOOGLE_API_BASE`` for other
      Google services.

    The proxy swaps this handle for the real upstream token and enforces the
    per-assistant file-access allowlist. Calling the provider hosts directly
    (``graph.microsoft.com`` / ``www.googleapis.com``) with this handle will
    fail: the sandbox holds no real token by design.

    Parameters
    ----------
    provider:
        Provider name or alias. Built-in aliases include ``"microsoft"``,
        ``"graph"``, ``"google"``, ``"gmail"``, and ``"drive"``.
    min_ttl_seconds:
        Accepted for signature compatibility; token freshness is handled by the
        proxy on each upstream call.

    Examples
    --------
    Multiple providers can be used in one sandbox; request each explicitly::

        microsoft_token = get_oauth_access_token("microsoft")
        google_token = get_oauth_access_token("google")

    Anti-patterns
    -------------
    - Do not call ``graph.microsoft.com`` / ``www.googleapis.com`` directly; use
      the proxy base URLs above.
    - Do not print, log, return, or store this handle.
    """
    _resolve_oauth_provider(provider)
    from unify.provider_proxy.proxy import ensure_proxy_running

    return ensure_proxy_running().nonce


def get_refresh_token_oauth_env_overlay() -> dict[str, str]:
    """Return the proxy base URLs + nonce to overlay into subprocess sandboxes.

    Venv and persistent shell sessions may outlive the parent process's last
    environment update, so each execution overlays the current workspace proxy
    endpoints. The sandbox is never given raw OAuth tokens: connected-provider
    REST (files and non-file) is reached through the localhost proxy, which
    injects the real token and enforces the file-access allowlist.
    """
    from unify.provider_proxy.proxy import ensure_proxy_running

    return dict(ensure_proxy_running().sandbox_env())


def get_oauth_prompt_context() -> str:
    """Return actor-facing documentation for OAuth runtime helpers."""
    doc = inspect.getdoc(get_oauth_access_token) or ""
    signature = (
        f"def {get_oauth_access_token.__name__}"
        f"{inspect.signature(get_oauth_access_token)}"
    )
    return (
        "### OAuth Access Token Helper: `get_oauth_access_token(...)`\n\n"
        "`get_oauth_access_token(...)` is available inside `execute_code` "
        "Python sessions and stored Python functions. It is a normal sandbox "
        "helper, not a JSON tool call.\n\n"
        f"```python\n{signature}\n```\n\n"
        f"{doc}"
    )
