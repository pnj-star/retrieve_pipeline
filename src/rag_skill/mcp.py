"""可复用 RAG 管线的 MCP 服务器入口。

每一个 MCP 工具都必须传入 ``tenant_id``、``kb_id`` 和 ``request_id``；
当 ``AUTH_MODE=jwt`` 时还要求 ``auth_token`` 携带与之匹配的
``tenant_id``/``kb_id`` claims。解析出的 ``AgentContext`` 会传入管线，
保证缓存、指标与下游日志都保持租户隔离。
"""

from __future__ import annotations

from typing import Any

from common_core.auth import AuthConfig
from common_core.mcp_auth import ToolContextGuard, build_mcp_auth

from .pipeline import RagPipeline
from .results import RagResult, RagStatus


def _load_pipeline(pipeline: RagPipeline | None) -> RagPipeline:
    """返回显式传入的管线，否则从环境构建一个默认管线。"""
    if pipeline is not None:
        return pipeline
    from .builder import build_pipeline

    return build_pipeline()


def create_mcp_server(
    pipeline: RagPipeline | None = None,
    *,
    auth: AuthConfig | None = None,
    name: str = "rag-skill",
    instructions: str = (
        "Tenant-scoped RAG tools. Every call must pass tenant_id, kb_id, and "
        "request_id; when AUTH_MODE=jwt, also pass a JWT auth_token whose "
        "tenant_id and kb_id claims match the requested scope. rag_answer "
        "returns status (answered|answered_cache|no_context|guard_blocked|error), "
        "message, docs, and answer."
    ),
) -> Any:
    """构建一个暴露 ``rag_retrieve`` 与 ``rag_answer`` 工具的 FastMCP 服务器。

    - pipeline: 可注入自定义管线（测试场景），默认从环境构建；
    - auth: 鉴权配置，默认 ``AuthConfig.from_env()``；
    - name / instructions: 服务器名称与面向 MCP 客户的说明。
    """
    from mcp.server.fastmcp import FastMCP

    pipeline = _load_pipeline(pipeline)
    auth_config = auth or AuthConfig.from_env()
    # 工具级上下文守卫：解析并校验租户作用域（JWT 时校验 claims 一致性）
    guard = ToolContextGuard(config=auth_config)
    token_verifier, auth_settings = build_mcp_auth(auth_config)
    mcp_kwargs: dict[str, Any] = {}
    if auth_settings is not None:
        # JWT 模式下在传输层（transport）接入 token 校验
        mcp_kwargs["token_verifier"] = token_verifier
        mcp_kwargs["auth"] = auth_settings
    server = FastMCP(name=name, instructions=instructions, **mcp_kwargs)

    @server.tool()
    async def rag_retrieve(
        query: str,
        tenant_id: str,
        kb_id: str,
        request_id: str,
        auth_token: str | None = None,
        session_id: str = "",
        user_id: str = "",
        collection_name: str | None = None,
        top_k: int | None = None,
        filter_expr: str | None = None,
        query_rewrite_mode: str | None = None,
        rewrite_query: str | None = None,
    ) -> dict[str, Any]:
        """只检索租户范围内的文档，不生成回答。"""
        # 解析租户作用域：校验 JWT（若启用），得到隔离的 AgentContext
        context = guard.resolve(
            tenant_id=tenant_id,
            kb_id=kb_id,
            request_id=request_id,
            session_id=session_id,
            user_id=user_id,
            auth_token=auth_token,
        )
        rewrite_trace: dict[str, Any] = {}
        docs = await pipeline.retrieve(
            query,
            context,
            collection_name=collection_name,
            top_k=top_k,
            filter_expr=filter_expr,
            query_rewrite_mode=query_rewrite_mode,
            rewrite_query=rewrite_query,
            rewrite_trace=rewrite_trace,
        )
        return {
            "ok": True,
            "status": "no_context" if not docs else "retrieved",
            "tenant_id": context.tenant_id,
            "kb_id": context.kb_id,
            "request_id": context.request_id,
            "user_id": context.user_id,
            "count": len(docs),
            "docs": docs,
            "rewritten_query": rewrite_trace.get("rewritten_query", "") or query,
        }

    @server.tool()
    async def rag_answer(
        query: str,
        tenant_id: str,
        kb_id: str,
        request_id: str,
        auth_token: str | None = None,
        session_id: str = "",
        user_id: str = "",
        system_prompt: str = "",
        top_k: int | None = None,
        empty_answer: str = "No relevant context found.",
        filter_expr: str | None = None,
        min_relevance: float | None = None,
        enable_guard: bool = True,
        prompt_template: str | None = None,
        context_max_chars: int | None = None,
        context_max_tokens: int | None = None,
        max_doc_chars: int | None = None,
        max_doc_tokens: int | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        query_rewrite_mode: str | None = None,
        rewrite_query: str | None = None,
    ) -> dict[str, Any]:
        """检索、重排并生成回答（走完整 RAG 管线）。

        ``prompt_template`` 必须包含 ``{context}``（可含 ``{query}``）；
        ``system_prompt`` 会作为额外指令追加到生成提示，不替换默认模板。
        ``query_rewrite_mode`` 可覆盖配置的改写策略（off / identity /
        llm_rewrite / query_expansion）；显式传 ``rewrite_query`` 时跳过改写。
        """
        context = guard.resolve(
            tenant_id=tenant_id,
            kb_id=kb_id,
            request_id=request_id,
            session_id=session_id,
            user_id=user_id,
            auth_token=auth_token,
        )
        result = await pipeline.answer(
            query,
            context,
            system_prompt=system_prompt,
            top_k=top_k,
            empty_answer=empty_answer,
            filter_expr=filter_expr,
            min_relevance=min_relevance,
            enable_guard=enable_guard,
            prompt_template=prompt_template,
            context_max_chars=context_max_chars,
            context_max_tokens=context_max_tokens,
            max_doc_chars=max_doc_chars,
            max_doc_tokens=max_doc_tokens,
            temperature=temperature,
            max_tokens=max_tokens,
            query_rewrite_mode=query_rewrite_mode,
            rewrite_query=rewrite_query,
        )
        # 兼容底层返回纯字符串的情况（例如自定义管线直接返回回答文本）
        if isinstance(result, str):
            result = RagResult(
                RagStatus.ANSWERED,
                "",
                [],
                result,
                rewritten_query=rewrite_query or query,
            )
        return {
            "ok": result.ok,
            "status": result.status,
            "message": result.message,
            "docs": result.docs,
            "count": len(result.docs),
            "answer": result.answer,
            "tenant_id": context.tenant_id,
            "kb_id": context.kb_id,
            "request_id": context.request_id,
            "user_id": context.user_id,
            "rewritten_query": getattr(result, "rewritten_query", "") or query,
        }

    return server


def main() -> None:
    """通过 stdio 运行 MCP 服务器，供 MCP 兼容客户端发现与调用。"""
    create_mcp_server().run(transport="stdio")


__all__ = ["create_mcp_server", "main"]
