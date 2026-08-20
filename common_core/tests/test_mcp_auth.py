import asyncio
import time

import jwt
import pytest

from common_core.auth import AuthError
from common_core.config import AuthConfig
from common_core.mcp_auth import (
    MCPBearerTokenVerifier,
    ToolAuthError,
    ToolContextGuard,
    build_mcp_auth,
)


def _token(secret: str, **claims: object) -> str:
    payload = {"exp": int(time.time()) + 3600, **claims}
    return jwt.encode(payload, secret, algorithm="HS256")


def test_guard_disabled_still_requires_scope() -> None:
    guard = ToolContextGuard(AuthConfig(mode="disabled"))
    ctx = guard.resolve(
        tenant_id="t1",
        kb_id="kb1",
        request_id="r1",
        session_id="s1",
    )
    assert ctx.tenant_id == "t1"
    assert ctx.request_id == "r1"
    assert ctx.user_id == "dev-user"

    with pytest.raises(ToolAuthError) as exc:
        guard.resolve(tenant_id="t1", kb_id="kb1")
    assert exc.value.status_code == 400


def test_guard_verifies_token_and_fills_missing_scope_from_claims() -> None:
    secret = "test-secret-key-0123456789abcdef"
    guard = ToolContextGuard(
        AuthConfig(jwt_secret=secret, jwt_issuer="issuer", jwt_audience="audience")
    )
    token = _token(
        secret,
        sub="user-1",
        tenant_id="tenant-1",
        kb_id="kb-1",
        request_id="req-1",
        iss="issuer",
        aud="audience",
    )
    ctx = guard.resolve(auth_token=token, request_id="")
    assert ctx.tenant_id == "tenant-1"
    assert ctx.kb_id == "kb-1"
    assert ctx.request_id == "req-1"
    assert ctx.user_id == "user-1"


def test_guard_rejects_token_scope_mismatch() -> None:
    secret = "test-secret-key-0123456789abcdef"
    guard = ToolContextGuard(AuthConfig(jwt_secret=secret))
    token = _token(secret, sub="user-1", tenant_id="tenant-1", kb_id="kb-1")
    with pytest.raises(ToolAuthError) as exc:
        guard.resolve(
            auth_token=token,
            tenant_id="tenant-2",
            kb_id="kb-1",
            request_id="r1",
        )
    assert exc.value.status_code == 403


def test_guard_rejects_missing_token_when_auth_enabled() -> None:
    guard = ToolContextGuard(AuthConfig(jwt_secret="test-secret"))
    with pytest.raises(ToolAuthError) as exc:
        guard.resolve(tenant_id="t1", kb_id="kb1", request_id="r1")
    assert exc.value.status_code == 401


def test_mcp_bearer_verifier_returns_access_token() -> None:
    pytest.importorskip("mcp")
    secret = "test-secret-key-0123456789abcdef"
    verifier = MCPBearerTokenVerifier(AuthConfig(jwt_secret=secret))
    token = _token(secret, sub="user-1", tenant_id="t1", roles=["agent"])

    access = asyncio.run(verifier.verify_token(token))
    assert access is not None
    assert access.client_id == "user-1"
    assert access.scopes == ["agent"]

    bad = asyncio.run(verifier.verify_token("not-a-jwt"))
    assert bad is None


def test_build_mcp_auth_wires_jwt_and_skips_disabled() -> None:
    pytest.importorskip("mcp")
    verifier, auth_settings = build_mcp_auth(
        AuthConfig(mode="jwt", jwt_secret="test-secret-key-0123456789abcdef")
    )
    assert verifier is not None
    assert auth_settings is not None
    assert str(auth_settings.issuer_url)
    assert str(auth_settings.resource_server_url)

    disabled_verifier, disabled_settings = build_mcp_auth(AuthConfig(mode="disabled"))
    assert disabled_verifier is None
    assert disabled_settings is None
