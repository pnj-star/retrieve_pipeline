from common_core.context import AgentContext


def test_from_claims_maps_identity_fields() -> None:
    ctx = AgentContext.from_claims(
        {"tenant_id": "t1", "kb_id": "kb1", "session_id": "s1", "request_id": "r1"}
    )
    assert ctx.tenant_id == "t1"
    assert ctx.kb_id == "kb1"
    assert ctx.session_id == "s1"
    assert ctx.request_id == "r1"


def test_from_claims_defaults_to_empty_string() -> None:
    ctx = AgentContext.from_claims({})
    assert ctx.tenant_id == ""
    assert ctx.kb_id == ""
    assert ctx.user_id == ""


def test_from_claims_maps_sub_to_user_id() -> None:
    ctx = AgentContext.from_claims({"sub": "user-1"})
    assert ctx.user_id == "user-1"
