"""Localhost policy-enforcing proxy for connected workspace providers.

Fronts the full Microsoft Graph and Google Drive REST APIs for the
``execute_code`` sandbox. The sandbox never receives a real provider OAuth
token; it targets the proxy base URLs and authorizes with a per-process nonce.
The proxy (trusted runtime) injects the real token upstream and enforces the
per-assistant file-access allowlist, so masked files/folders are invisible in
listings/search and not-found on direct access, while non-file provider APIs
pass straight through.
"""

from unify.provider_proxy.policy import (
    PolicyStore,
    WorkspaceFilePolicy,
    evaluate_access,
    get_policy_store,
)
from unify.provider_proxy.proxy import ensure_proxy_running
from unify.provider_proxy.session import (
    ProxySession,
    build_sandbox_env,
    current_session,
    provider_token_env_keys,
)

__all__ = [
    "PolicyStore",
    "ProxySession",
    "WorkspaceFilePolicy",
    "build_sandbox_env",
    "current_session",
    "ensure_proxy_running",
    "evaluate_access",
    "get_policy_store",
    "provider_token_env_keys",
]
