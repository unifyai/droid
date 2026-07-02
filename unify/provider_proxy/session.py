"""Process-wide state for the localhost provider proxy.

Holds the bound host/port and the per-process capability nonce, and derives the
base URLs handed to the ``execute_code`` sandbox. This module is intentionally
dependency-light so it can be imported from both the trusted runtime and the
credential helpers without creating import cycles.

The sandbox never receives a real provider OAuth token. Instead it authorizes
against this local proxy with ``nonce`` and points its base URL at the proxy;
the trusted proxy swaps the nonce for the real upstream token and enforces the
file-access allowlist before anything leaves the process.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass

# Sensitive provider credential env vars that must never reach the sandbox
# (venv/shell subprocesses or in-process actor code). These are the raw tokens
# whose presence would let actor code call Drive/Graph directly and bypass the
# allowlist. Kept in lockstep with the built-in providers in ``runtime_oauth``.
_PROVIDER_TOKEN_ENV_KEYS: frozenset[str] = frozenset(
    {
        "MICROSOFT_ACCESS_TOKEN",
        "MICROSOFT_REFRESH_TOKEN",
        "GOOGLE_ACCESS_TOKEN",
        "GOOGLE_REFRESH_TOKEN",
    },
)


def provider_token_env_keys() -> frozenset[str]:
    """Return the credential env var names that must be scrubbed from sandboxes."""
    return _PROVIDER_TOKEN_ENV_KEYS


@dataclass(frozen=True)
class ProxySession:
    """A live localhost proxy binding plus its capability nonce."""

    host: str
    port: int
    nonce: str

    @property
    def root_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def provider_root(self, provider: str) -> str:
        return f"{self.root_url}/{provider}"

    @property
    def microsoft_graph_base(self) -> str:
        # Drop-in replacement for https://graph.microsoft.com/v1.0
        return f"{self.provider_root('microsoft')}/v1.0"

    @property
    def google_drive_base(self) -> str:
        # Drop-in replacement for https://www.googleapis.com/drive/v3
        return f"{self.provider_root('google')}/drive/v3"

    @property
    def google_api_base(self) -> str:
        # Drop-in replacement for https://www.googleapis.com
        return self.provider_root("google")

    def sandbox_env(self) -> dict[str, str]:
        """Env vars handed to the sandbox so SDKs/HTTP target the proxy."""
        return {
            "WORKSPACE_PROXY_URL": self.root_url,
            "WORKSPACE_PROXY_TOKEN": self.nonce,
            "MICROSOFT_GRAPH_BASE": self.microsoft_graph_base,
            "GOOGLE_DRIVE_BASE": self.google_drive_base,
            "GOOGLE_API_BASE": self.google_api_base,
        }


_LOCK = threading.Lock()
_SESSION: ProxySession | None = None


def set_session(session: ProxySession) -> None:
    global _SESSION
    with _LOCK:
        _SESSION = session


def current_session() -> ProxySession | None:
    with _LOCK:
        return _SESSION


def build_sandbox_env(base: dict[str, str] | None = None) -> dict[str, str]:
    """Return a subprocess env with provider tokens removed and proxy vars added.

    Copies the current process environment (or *base*), strips the raw provider
    credential vars, and overlays the proxy base URLs + nonce. Used at every
    sandbox spawn site so no subprocess ever inherits a usable file token.
    """
    env = dict(os.environ if base is None else base)
    for key in _PROVIDER_TOKEN_ENV_KEYS:
        env.pop(key, None)
    session = current_session()
    if session is not None:
        env.update(session.sandbox_env())
    return env
