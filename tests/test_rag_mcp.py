"""MCP 入口契约测试：验证 rag_retrieve 工具的入参透传、返回契约、
上下文预算能力、鉴权接线与传输层 auth 配置。"""

import asyncio

import pytest

pytest.importorskip("mcp")

from common_core.config import AuthConfig
from common_core.context import AgentContext
from rag_skill.mcp import create_mcp_server
from rag_skill.results import RetrieveResult, RetrieveStatus


class FakePipeline:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def retrieve_context(
        self,
        query: str,
        context: AgentContext | None = None,
        **kwargs,
    ) -> RetrieveResult:
        self.calls.append({"op": "retrieve_context", "query": query, "context": context, **kwargs})
        return RetrieveResult(
            RetrieveStatus.RETRIEVED,
            docs=[{"content": "doc", "score": 0.9}],
            rewritten_query=kwargs.get("rewrite_query") or query,
            message="ok",
        )


def _payload(result):
    return result[1] if isinstance(result, tuple) else result


def _run(coro):
    return asyncio.run(coro)


def test_mcp_server_exposes_rag_tools() -> None:
    server = create_mcp_server(pipeline=FakePipeline(), auth=AuthConfig(mode="disabled"))
    names = {tool.name for tool in _run(server.list_tools())}
    assert "rag_retrieve" in names
    assert "rag_answer" not in names


def test_rag_retrieve_accepts_traceparent_and_returns_trace_id() -> None:
    fake = FakePipeline()
    server = create_mcp_server(pipeline=fake, auth=AuthConfig(mode="disabled"))
    result = _run(
        server.call_tool(
            "rag_retrieve",
            {
                "query": "policy lookup",
                "tenant_id": "t2",
                "kb_id": "kb2",
                "request_id": "r2",
                "traceparent": "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",
            },
        )
    )
    payload = _payload(result)
    assert payload["docs"][0]["content"] == "doc"
    assert "context_text" in payload
    assert payload["context_text"]
    assert "trace_id" in payload
    assert payload["status"] == "retrieved"


def test_rag_retrieve_passes_scope_and_top_k() -> None:
    fake = FakePipeline()
    server = create_mcp_server(pipeline=fake, auth=AuthConfig(mode="disabled"))
    result = _run(
        server.call_tool(
            "rag_retrieve",
            {
                "query": "policy lookup",
                "tenant_id": "t2",
                "kb_id": "kb2",
                "request_id": "r2",
                "top_k": 5,
            },
        )
    )
    payload = _payload(result)
    assert payload["count"] == 1
    assert payload["docs"][0]["content"] == "doc"
    assert fake.calls[-1]["top_k"] == 5
    assert fake.calls[-1]["context"].tenant_id == "t2"
    assert fake.calls[-1]["op"] == "retrieve_context"


def test_rag_retrieve_accepts_budget_params() -> None:
    fake = FakePipeline()
    server = create_mcp_server(pipeline=fake, auth=AuthConfig(mode="disabled"))
    result = _run(
        server.call_tool(
            "rag_retrieve",
            {
                "query": "policy lookup",
                "tenant_id": "t4",
                "kb_id": "kb4",
                "request_id": "r4",
                "context_max_chars": 200,
                "context_max_tokens": 512,
                "max_doc_chars": 300,
                "max_doc_tokens": 150,
            },
        )
    )
    payload = _payload(result)
    assert payload["ok"] is True
    assert payload["count"] == 1
    assert "context_text" in payload
    assert payload["context_text"]
    assert payload["cache_hit"] is False


def test_rag_retrieve_forwards_query_rewrite_options() -> None:
    fake = FakePipeline()
    server = create_mcp_server(pipeline=fake, auth=AuthConfig(mode="disabled"))
    result = _run(
        server.call_tool(
            "rag_retrieve",
            {
                "query": "原始问题",
                "tenant_id": "t1",
                "kb_id": "kb1",
                "request_id": "r1",
                "query_rewrite_mode": "query_expansion",
                "rewrite_query": "改写后的问题",
            },
        )
    )
    payload = _payload(result)

    call = fake.calls[-1]
    assert call["query_rewrite_mode"] == "query_expansion"
    assert call["rewrite_query"] == "改写后的问题"
    assert payload["rewritten_query"] == "改写后的问题"


class EmptyPipeline(FakePipeline):
    async def retrieve_context(
        self,
        query: str,
        context: AgentContext | None = None,
        **kwargs,
    ) -> RetrieveResult:
        self.calls.append({"op": "retrieve_context", "query": query, "context": context, **kwargs})
        return RetrieveResult(
            RetrieveStatus.NO_CONTEXT,
            docs=[],
            rewritten_query=query,
            message="no context",
        )


def test_rag_retrieve_reports_no_context_status() -> None:
    server = create_mcp_server(pipeline=EmptyPipeline(), auth=AuthConfig(mode="disabled"))
    result = _run(
        server.call_tool(
            "rag_retrieve",
            {
                "query": "nothing here",
                "tenant_id": "t3",
                "kb_id": "kb3",
                "request_id": "r3",
            },
        )
    )
    payload = _payload(result)
    assert payload["count"] == 0
    assert payload["status"] == "no_context"


def test_mcp_server_wires_transport_auth_when_jwt() -> None:
    server = create_mcp_server(
        pipeline=FakePipeline(),
        auth=AuthConfig(mode="jwt", jwt_secret="test-secret-key-0123456789abcdef"),
    )
    assert server._token_verifier is not None
    assert server.settings.auth is not None


def test_mcp_server_skips_transport_auth_when_disabled() -> None:
    server = create_mcp_server(
        pipeline=FakePipeline(),
        auth=AuthConfig(mode="disabled"),
    )
    assert server._token_verifier is None
    assert server.settings.auth is None
