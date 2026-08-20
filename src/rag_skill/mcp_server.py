"""构建暴露 rag_retrieve / rag_answer 的 FastMCP 服务器。

把 MCP 服务器（工具 + 鉴权 + trace 接线）与命令行入口拆开：本模块等价于
参考工程里的 ``tools_server.py``，只负责 ``create_mcp_server()``；入口
``mcp.py`` 等价于 ``streamable_server.py``，负责解析参数并启动。
"""

from __future__ import annotations

import logging
from typing import Any

from common_core import telemetry
from common_core.auth import AuthConfig
from common_core.instrumentation import current_trace_id, trace_node
from common_core.mcp_auth import ToolContextGuard, build_mcp_auth

from .pipeline import RagPipeline
from .results import RagResult, RagStatus

logger = logging.getLogger(__name__)


def _load_pipeline(pipeline: RagPipeline | None) -> RagPipeline:
    """返回显式传入的管线，否则从环境构建一个默认管线。"""
    if pipeline is not None:
        return pipeline
    from .builder import build_pipeline

    return build_pipeline()


def _warmup_grid_connections(pipeline: RagPipeline) -> None:
    """服务启动时尽力预热网格后端连接，把冷启动提前到服务 ready 之前。

    首次 ``rag_retrieve`` 会同时连 Milvus 并加载 embedding 模型，这个窗口
    可能撞上 httpx "client has been closed"（连接被关的竞态）。这里在 accept
    请求前先把 Milvus 连上，失败只记告警、不阻断启动（运行时仍会惰性重连）。
    """
    store = getattr(pipeline, "vector", None)
    if store is None or not hasattr(store, "connect"):
        return
    try:
        store.connect()
        # 连上后顺带预加载默认集合，避免首次请求在 collection.load() 上撞竞态。
        # 非默认集合（调用方显式传 collection_name）仍按需惰性加载。
        default_collection = getattr(getattr(store, "config", None), "text_collection", "") or None
        if default_collection and hasattr(store, "_ensure"):
            store._ensure(default_collection)
        logger.info(
            "rag_skill warmup: connected to Milvus (collection=%s)",
            default_collection or "none",
        )
    except Exception:  # noqa: BLE001 - 预热失败不应阻止服务启动
        logger.warning(
            "rag_skill warmup: Milvus connect failed (lazy retry on first request); "
            "check MILVUS_HOST/MILVUS_PORT",
            exc_info=True,
        )


def _attach_trace(traceparent: str | None):
    """解析并挂载上游 W3C traceparent 为当前 OTel 上下文。

    返回恢复上下文所需的 token；解析失败或未启用追踪时返回 None（no-op）。
    """
    return telemetry.set_current_context(telemetry.parse_traceparent(traceparent))


def create_mcp_server(
    pipeline: RagPipeline | None = None,
    *,
    auth: AuthConfig | None = None,
    name: str = "rag-skill",
    host: str | None = None,
    port: int | None = None,
    streamable_path: str = "/streamable",
    sse_path: str = "/sse",
    log_level: str = "INFO",
    debug: bool = False,
    instructions: str = (
        "Tenant-scoped RAG tools. Every call must pass tenant_id, kb_id, and "
        "request_id; when AUTH_MODE=jwt, also pass a JWT auth_token whose "
        "tenant_id and kb_id claims match the requested scope. rag_answer "
        "returns status (answered|answered_cache|no_context|guard_blocked|error), "
        "message, docs, and answer. Optional: pass an upstream W3C traceparent "
        "header to keep the distributed trace linked; the response includes a "
        "trace_id for correlation in logs and tracing backends."
    ),
) -> Any:
    """构建暴露 rag_retrieve / rag_answer 的 FastMCP 服务器。"""
    from mcp.server.fastmcp import FastMCP

    pipeline = _load_pipeline(pipeline)
    _warmup_grid_connections(pipeline)
    auth_config = auth or AuthConfig.from_env()
    guard = ToolContextGuard(config=auth_config)
    token_verifier, auth_settings = build_mcp_auth(auth_config)
    mcp_kwargs: dict[str, Any] = {
        "streamable_http_path": streamable_path,
        "sse_path": sse_path,
        "log_level": log_level,
        "debug": debug,
    }
    if auth_settings is not None:
        mcp_kwargs["token_verifier"] = token_verifier
        mcp_kwargs["auth"] = auth_settings
    if host is not None:
        mcp_kwargs["host"] = host
    if port is not None:
        mcp_kwargs["port"] = port
    server = FastMCP(name=name, instructions=instructions, **mcp_kwargs)

    @server.tool()
    async def rag_retrieve(
        query: str,
        tenant_id: str,
        kb_id: str,
        request_id: str,
        auth_token: str | None = None,
        traceparent: str | None = None,
        session_id: str = "",
        user_id: str = "",
        collection_name: str | None = None,
        top_k: int | None = None,
        filter_expr: str | None = None,
        query_rewrite_mode: str | None = None,
        rewrite_query: str | None = None,
    ) -> dict[str, Any]:
        """只检索租户范围内的文档，不生成回答。"""
        context = guard.resolve(
            tenant_id=tenant_id,
            kb_id=kb_id,
            request_id=request_id,
            session_id=session_id,
            user_id=user_id,
            auth_token=auth_token,
        )
        trace_token = _attach_trace(traceparent)
        trace_id_value: str | None = None
        rewrite_trace: dict[str, Any] = {}
        try:
            with trace_node(
                "rag_retrieve",
                tenant_id=context.tenant_id,
                kb_id=context.kb_id,
                request_id=context.request_id,
            ):
                trace_id_value = current_trace_id()
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
        finally:
            telemetry.reset_context(trace_token)
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
            "trace_id": trace_id_value,
        }

    @server.tool()
    async def rag_answer(
        query: str,
        tenant_id: str,
        kb_id: str,
        request_id: str,
        auth_token: str | None = None,
        traceparent: str | None = None,
        session_id: str = "",
        user_id: str = "",
        collection_name: str | None = None,
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
        """检索、重排并生成回答（走完整 RAG 管线）。"""
        context = guard.resolve(
            tenant_id=tenant_id,
            kb_id=kb_id,
            request_id=request_id,
            session_id=session_id,
            user_id=user_id,
            auth_token=auth_token,
        )
        trace_token = _attach_trace(traceparent)
        trace_id_value: str | None = None
        try:
            with trace_node(
                "rag_answer",
                tenant_id=context.tenant_id,
                kb_id=context.kb_id,
                request_id=context.request_id,
            ):
                trace_id_value = current_trace_id()
                result = await pipeline.answer(
                    query,
                    context,
                    collection_name=collection_name,
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
                if isinstance(result, str):
                    result = RagResult(
                        RagStatus.ANSWERED,
                        "",
                        [],
                        result,
                        rewritten_query=rewrite_query or query,
                    )
        finally:
            telemetry.reset_context(trace_token)
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
            "trace_id": trace_id_value,
        }

    return server


__all__ = ["create_mcp_server"]
