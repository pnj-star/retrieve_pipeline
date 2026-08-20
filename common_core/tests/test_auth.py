import time

import jwt
import pytest

from common_core.auth import AuthError, IdentityClaims, TokenVerifier, parse_bearer_token
from common_core.config import AuthConfig


def _token(secret: str, **claims: object) -> str:
    payload = {"exp": int(time.time()) + 3600, **claims}
    return jwt.encode(payload, secret, algorithm="HS256")


def test_parse_bearer_token() -> None:
    assert parse_bearer_token("Bearer abc") == "abc"
    assert parse_bearer_token("bearer  abc") == "abc"
    assert parse_bearer_token("Basic abc") is None
    assert parse_bearer_token(None) is None


def test_verifier_returns_identity_and_context() -> None:
    secret = "test-secret-key-0123456789abcdef"
    verifier = TokenVerifier(
        AuthConfig(jwt_secret=secret, jwt_issuer="issuer", jwt_audience="audience")
    )
    token = _token(
        secret,
        sub="user-1",
        tenant_id="tenant-1",
        kb_id="kb-1",
        iss="issuer",
        aud="audience",
    )
    identity = verifier.identity(token)
    assert isinstance(identity, IdentityClaims)
    assert identity.sub == "user-1"
    assert identity.tenant_id == "tenant-1"
    ctx = identity.to_context()
    assert ctx.user_id == "user-1"
    assert ctx.tenant_id == "tenant-1"
    assert ctx.kb_id == "kb-1"


def test_verifier_rejects_expired_token() -> None:
    secret = "test-secret-key-0123456789abcdef"
    verifier = TokenVerifier(AuthConfig(jwt_secret=secret))
    token = jwt.encode(
        {"sub": "u", "exp": int(time.time()) - 10},
        secret,
        algorithm="HS256",
    )
    with pytest.raises(AuthError):
        verifier.verify(token)


def test_verifier_fails_closed_without_key() -> None:
    verifier = TokenVerifier(AuthConfig(jwt_secret=""))
    with pytest.raises(AuthError):
        verifier.verify("not-a-real-jwt")
