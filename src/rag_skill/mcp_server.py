"""构建只暴露 rag_retrieve 的 FastMCP 服务器。

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
from common_core.rag.assembly import (
    DEFAULT_MAX_CONTEXT_CHARS,
    build_context_text,
    clean_markdown,
)

from .pipeline import RagPipeline
from .results import RetrieveStatus

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


def _format_doc(index: int, doc: dict[str, Any]) -> str:
    """把单篇检索文档渲染成一行上下文片段（供 build_context_text 使用）。"""
    content = str(doc.get("parent_content", "") or doc.get("content", "") or "")
    fields = {
        "content": clean_markdown(content) if content.strip() else None,
        "source": doc.get("source"),
        "category": doc.get("category"),
        "score": doc.get("score"),
    }
    parts = [f"[{index}]"]
    for field_name, value in fields.items():
        if value is not None and str(value).strip():
            parts.append(f"{field_name}: {value}")
    return " | ".join(parts)


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
        "Tenant-scoped RAG retrieval tool. Every call must pass tenant_id, kb_id, "
        "and request_id; when AUTH_MODE=jwt, also pass a JWT auth_token whose "
        "tenant_id and kb_id claims match the requested scope. rag_retrieve only "
        "returns raw docs plus a budgeted context_text, and never generates an "
        "answer. Optional: pass an upstream W3C traceparent header to keep the "
        "distributed trace linked; the response includes a trace_id for "
        "correlation in logs and tracing backends."
    ),
) -> Any:
    """构建只暴露 rag_retrieve 的 FastMCP 服务器。"""
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
        min_relevance: float | None = None,
        context_max_chars: int | None = None,
        context_max_tokens: int | None = None,
        max_doc_chars: int | None = None,
        max_doc_tokens: int | None = None,
    ) -> dict[str, Any]:
        """只检索租户范围内的文档，不生成回答。

        走完整检索管道：检索缓存命中（query → 精排后达标文档）直接返回；
        未命中则查询改写 → 混合检索（稀疏+稠密）→ RRF → 精排 → 阈值筛选，
        达标文档回写检索缓存。返回原始 ``docs`` 供 agent 做多来源融合 / 复核，
        同时用共享的 ``build_context_text`` 产出一段受预算约束、可直接喂给 LLM
        的 ``context_text``，避免 top-k 文档把 agent 的上下文窗口打爆。不做回答生成。
        """
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
                "rag_retrieve",
                tenant_id=context.tenant_id,
                kb_id=context.kb_id,
                request_id=context.request_id,
            ):
                trace_id_value = current_trace_id()
                result = await pipeline.retrieve_context(
                    query,
                    context,
                    collection_name=collection_name,
                    top_k=top_k,
                    filter_expr=filter_expr,
                    query_rewrite_mode=query_rewrite_mode,
                    rewrite_query=rewrite_query,
                    min_relevance=min_relevance,
                )
        finally:
            telemetry.reset_context(trace_token)
        docs = result.docs
        if docs:
            context_text, _sources = build_context_text(
                docs,
                max_chars=(
                    DEFAULT_MAX_CONTEXT_CHARS
                    if context_max_chars is None
                    else context_max_chars
                ),
                max_tokens=context_max_tokens,
                max_doc_chars=max_doc_chars,
                max_doc_tokens=max_doc_tokens,
                format_doc=_format_doc,
            )
        else:
            context_text = ""
        return {
            "ok": result.status != RetrieveStatus.ERROR,
            "status": result.status,
            "message": result.message,
            "cache_hit": result.cache_hit,
            "tenant_id": context.tenant_id,
            "kb_id": context.kb_id,
            "request_id": context.request_id,
            "user_id": context.user_id,
            "count": len(docs),
            "docs": docs,
            "context_text": context_text,
            "rewritten_query": result.rewritten_query or query,
            "trace_id": trace_id_value,
        }

    return server


__all__ = ["create_mcp_server"]
