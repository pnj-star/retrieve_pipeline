"""MCP 入口契约测试：验证 rag_answer / rag_retrieve 工具的入参透传、
返回契约、鉴权接线与传输层 auth 配置。"""

import asyncio

import pytest

pytest.importorskip("mcp")

from common_core.config import AuthConfig
from common_core.context import AgentContext
from rag_skill.mcp import create_mcp_server
from rag_skill.results import RagResult, RagStatus


class FakePipeline:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def retrieve(
        self,
        query: str,
        context: AgentContext | None = None,
        **kwargs,
    ) -> list[dict]:
        self.calls.append({"op": "retrieve", "query": query, "context": context, **kwargs})
        trace = kwargs.get("rewrite_trace")
        if isinstance(trace, dict):
            trace["rewritten_query"] = kwargs.get("rewrite_query") or query
        return [{"content": "doc", "score": 0.9}]

    async def answer(
        self,
        query: str,
        context: AgentContext | None = None,
        **kwargs,
    ) -> RagResult | str:
        self.calls.append({"op": "answer", "query": query, "context": context, **kwargs})
        return RagResult(
            RagStatus.ANSWERED,
            "ok",
            [],
            "answer-ok",
            rewritten_query=kwargs.get("rewrite_query") or query,
        )


def _payload(result):
    return result[1] if isinstance(result, tuple) else result


def _run(coro):
    return asyncio.run(coro)


def test_mcp_server_exposes_rag_tools() -> None:
    server = create_mcp_server(pipeline=FakePipeline(), auth=AuthConfig(mode="disabled"))
    names = {tool.name for tool in _run(server.list_tools())}
    assert {"rag_answer", "rag_retrieve"} <= names


def test_rag_answer_passes_scope_to_pipeline() -> None:
    fake = FakePipeline()
    server = create_mcp_server(pipeline=fake, auth=AuthConfig(mode="disabled"))
    result = _run(
        server.call_tool(
            "rag_answer",
            {
                "query": "what is the policy?",
                "tenant_id": "t1",
                "kb_id": "kb1",
                "request_id": "r1",
                "session_id": "s1",
            },
        )
    )
    payload = _payload(result)
    assert payload["answer"] == "answer-ok"
    assert payload["status"] == "answered"
    assert payload["message"] == "ok"
    assert payload["docs"] == []
    assert payload["tenant_id"] == "t1"
    assert payload["request_id"] == "r1"

    call = fake.calls[-1]
    assert call["context"] == AgentContext(
        tenant_id="t1", kb_id="kb1", session_id="s1", request_id="r1", user_id="dev-user"
    )


def test_rag_answer_accepts_traceparent_and_returns_trace_id() -> None:
    fake = FakePipeline()
    server = create_mcp_server(pipeline=fake, auth=AuthConfig(mode="disabled"))
    result = _run(
        server.call_tool(
            "rag_answer",
            {
                "query": "what is the policy?",
                "tenant_id": "t1",
                "kb_id": "kb1",
                "request_id": "r1",
                "traceparent": (
                    "00-4bf92f3577b34da6a3ce929d0e0e4736-"
                    "00f067aa0ba902b7-01"
                ),
            },
        )
    )
    payload = _payload(result)
    # 无论 OTel 是否启用，返回契约都必须稳定且带 trace_id 字段
    assert payload["answer"] == "answer-ok"
    assert payload["status"] == "answered"
    assert "trace_id" in payload


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
    assert "trace_id" in payload


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


def test_rag_answer_forwards_generation_options() -> None:
    fake = FakePipeline()
    server = create_mcp_server(pipeline=fake, auth=AuthConfig(mode="disabled"))
    _run(
        server.call_tool(
            "rag_answer",
            {
                "query": "policy lookup",
                "tenant_id": "t4",
                "kb_id": "kb4",
                "request_id": "r4",
                "temperature": 0.3,
                "max_tokens": 128,
                "context_max_chars": 200,
                "context_max_tokens": 512,
                "max_doc_chars": 300,
                "max_doc_tokens": 150,
                "prompt_template": "Use {context}",
            },
        )
    )

    call = fake.calls[-1]
    assert call["temperature"] == 0.3
    assert call["max_tokens"] == 128
    assert call["context_max_chars"] == 200
    assert call["context_max_tokens"] == 512
    assert call["max_doc_chars"] == 300
    assert call["max_doc_tokens"] == 150
    assert call["prompt_template"] == "Use {context}"


def test_rag_answer_forwards_query_rewrite_options() -> None:
    fake = FakePipeline()
    server = create_mcp_server(pipeline=fake, auth=AuthConfig(mode="disabled"))
    result = _run(
        server.call_tool(
            "rag_answer",
            {
                "query": "原始问题",
                "tenant_id": "t1",
                "kb_id": "kb1",
                "request_id": "r1",
                "query_rewrite_mode": "llm_rewrite",
                "rewrite_query": "改写后的问题",
            },
        )
    )
    payload = _payload(result)

    call = fake.calls[-1]
    assert call["query_rewrite_mode"] == "llm_rewrite"
    assert call["rewrite_query"] == "改写后的问题"
    assert payload["rewritten_query"] == "改写后的问题"


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
    async def retrieve(self, query: str, context: AgentContext | None = None, **kwargs):
        self.calls.append({"op": "retrieve", "query": query, "context": context, **kwargs})
        return []


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
