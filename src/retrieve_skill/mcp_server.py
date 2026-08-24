"""构建只暴露 rag_retrieve 的 FastMCP 服务器。

把 MCP 服务器（工具 + 鉴权 + trace 接线）与命令行入口拆开：本模块等价于
参考工程里的 ``tools_server.py``，只负责 ``create_mcp_server()``；入口
``mcp.py`` 等价于 ``streamable_server.py``，负责解析参数并启动。
"""

from __future__ import annotations

import logging
from typing import Any

from starlette.requests import Request
from starlette.responses import JSONResponse

from common_core import telemetry
from common_core.auth import AuthConfig
from common_core.instrumentation import current_trace_id, trace_node
from common_core.mcp_auth import ToolContextGuard, build_mcp_auth

from .pipeline import RagPipeline
from .results import RetrieveStatus

logger = logging.getLogger(__name__)


def _health_payload(
    name: str,
    runtime: Any,
    config_source: str | None,
) -> dict[str, Any]:
    """构造 /health 响应；配置不完整时仍返回 ok 便于运维继续排查。"""
    from common_core.config import config_fingerprint, redacted_snapshot

    if runtime is None:
        return {
            "status": "ok",
            "service": name,
            "tools": ["rag_retrieve"],
            "config": {
                "complete": True,
                "fingerprint": None,
                "source": config_source or "unknown",
                "masked": {},
            },
        }
    complete = True
    try:
        runtime.validate(require_embedding=True, require_llm=False)
    except Exception:  # noqa: BLE001 - health 只报告状态，不因配置问题崩溃
        complete = False
    return {
        "status": "ok",
        "service": name,
        "tools": ["rag_retrieve"],
        "config": {
            "complete": complete,
            "fingerprint": config_fingerprint(runtime),
            "source": config_source or "process-env",
            "masked": redacted_snapshot(runtime),
        },
    }


def _load_pipeline(pipeline: RagPipeline | None, metrics: Any = None) -> RagPipeline:
    """返回显式传入的管线，否则从环境构建一个默认管线。"""
    if pipeline is not None:
        return pipeline
    from .builder import build_pipeline

    return build_pipeline(metrics=metrics)


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
            "retrieve_skill warmup: connected to Milvus (collection=%s)",
            default_collection or "none",
        )
    except Exception:  # noqa: BLE001 - 预热失败不应阻止服务启动
        logger.warning(
            "retrieve_skill warmup: Milvus connect failed (lazy retry on first request); "
            "check MILVUS_HOST/MILVUS_PORT",
            exc_info=True,
        )


def _warmup_cache_connection(pipeline: RagPipeline) -> None:
    """启动时探测 Redis，缓存不可用时显式告警而不是无感知降级。"""
    cache = getattr(pipeline, "cache", None)
    if cache is None or not hasattr(cache, "ping"):
        return
    try:
        cache_available = bool(cache.ping())
    except Exception:  # noqa: BLE001 - 探测失败按不可用处理
        cache_available = False
    if cache_available:
        logger.info("retrieve_skill warmup: Redis available; retrieval cache enabled")
    else:
        logger.warning(
            "retrieve_skill warmup: Redis unavailable; retrieval cache will degrade "
            "to live search on every request. Check REDIS_HOST/REDIS_PORT before production."
        )


def _attach_trace(traceparent: str | None):
    """解析并挂载上游 W3C traceparent 为当前 OTel 上下文。

    返回恢复上下文所需的 token；解析失败或未启用追踪时返回 None（no-op）。
    """
    return telemetry.set_current_context(telemetry.parse_traceparent(traceparent))


def _transport_auth_token() -> str | None:
    """HTTP 传输层校验通过后，从 FastMCP 上下文取回原始 JWT。"""
    try:
        from mcp.server.auth.middleware.auth_context import get_access_token

        access_token = get_access_token()
    except Exception:  # noqa: BLE001 - 非 HTTP 传输/未启用鉴权时优雅降级
        return None
    return access_token.token if access_token is not None else None


def create_mcp_server(
    pipeline: RagPipeline | None = None,
    *,
    auth: AuthConfig | None = None,
    metrics: Any = None,
    runtime: Any = None,
    config_source: str | None = None,
    name: str = "retrieve-skill",
    host: str | None = None,
    port: int | None = None,
    streamable_path: str = "/streamable",
    sse_path: str = "/sse",
    log_level: str = "INFO",
    debug: bool = False,
    instructions: str = (
        "租户隔离的 RAG 检索工具。只做检索、不做回答生成：输入 query + tenant_id + kb_id + request_id，"
        "返回精排后达到相关性阈值的父块粒度文档 docs（含命中的子块 id 列表 child_ids），绝不会生成回答。"
        "若 AUTH_MODE=jwt 开启，HTTP 调用通过 Authorization Bearer 头传递 JWT，也可在工具参数里传 auth_token；"
        "其中 tenant_id / kb_id claims 需与请求参数一致。"
        "可选参数支持按集合检索（collection_name）、控制返回条数（top_k）、追加过滤（filter_expr）、"
        "覆盖相关性阈值（min_relevance）与上下文预算（context_max_tokens / max_doc_tokens、"
        "context_max_chars / max_doc_chars）。"
        "可选传入上游 W3C traceparent 以串联分布式调用链路，"
        "响应会携带 trace_id 供日志与链路追踪关联排查。"
    ),
) -> Any:
    """构建只暴露 rag_retrieve 的 FastMCP 服务器。"""
    from mcp.server.fastmcp import FastMCP

    pipeline = _load_pipeline(pipeline, metrics)
    _warmup_grid_connections(pipeline)
    _warmup_cache_connection(pipeline)
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

    @server.custom_route("/health", methods=["GET"])
    async def health_check(request: Request) -> JSONResponse:
        """K8s readiness / 运维探活：暴露脱敏配置状态与指纹。"""
        return JSONResponse(_health_payload(name, runtime, config_source))

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
        context_max_tokens: int | None = None,
        max_doc_tokens: int | None = None,
        context_max_chars: int | None = None,
        max_doc_chars: int | None = None,
    ) -> dict[str, Any]:
        """只检索租户范围内的文档，不生成回答。

            走完整检索管道：检索缓存命中后回源校验精排达标的父块引用；未命中则
            查询改写 → 混合检索（稀疏+稠密）→ RRF → 精排 → 阈值筛选，达标子块
            按 parent_id 去重后回写缓存。随后从权威父块存储批量加载正文，并做
            单篇与总量的 token/字符预算截断。返回父块粒度 ``docs``，不做回答生成。

        Args:
            query: 用户原始问题文本。
            tenant_id: 租户 ID，用于租户隔离与鉴权校验。
            kb_id: 知识库 ID，用于知识库隔离。
            request_id: 本次请求唯一标识，供日志追踪。
            auth_token: JWT 令牌；HTTP 调用也可通过 Authorization Bearer 头传递。
            traceparent: 上游 W3C traceparent 头，串联分布式调用链路。
            session_id: 会话 ID，用于日志关联。
            user_id: 用户 ID，用于日志关联。
            collection_name: 目标 Milvus 集合名；不传用环境默认集合。
            top_k: 每路检索返回候选条数；不传用环境默认值。
            filter_expr: 附加业务过滤表达式（Milvus 布尔语法）。
            query_rewrite_mode: 查询改写策略，由调用方 agent 根据用户问题特征选择：
                "off" — 问题简短明确，不做改写；
                "llm_rewrite" — 问题口语化或有省略指代，LLM 改写成规范检索查询；
                "query_expansion" — 一句话含多个子问题或话题宽泛，LLM 生成多条变体分别检索后合并提升召回。
                不传则使用服务端配置默认值。
            rewrite_query: 显式传入改写后的查询文本；传了即跳过内部改写。
            min_relevance: 本次精排相关性阈值覆盖；不传用环境默认值。
            context_max_tokens: 父块正文总 token 预算；不传默认 6000。
            max_doc_tokens: 单篇父块正文 token 上限；不传约为总预算一半。
            context_max_chars: 可选的父块正文字符总预算；与 token 预算同时生效。
            max_doc_chars: 可选的单篇父块字符上限。

        Returns:
            结构化字典：ok 表示没有内部异常；status 区分 retrieved /
            retrieved_cache / no_context / error；docs 是父块粒度最终上下文，
            每项通常含 id/parent_id、content、child_ids、score、ce_score、
            tenant_id、kb_id 和 doc_version。count 等于 len(docs)；
            diagnostics 提供数量级摘要；trace_id 用于串联日志。
        """
        token = _transport_auth_token() or auth_token
        context = guard.resolve(
            tenant_id=tenant_id,
            kb_id=kb_id,
            request_id=request_id,
            session_id=session_id,
            user_id=user_id,
            auth_token=token,
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
                    context_max_tokens=context_max_tokens,
                    max_doc_tokens=max_doc_tokens,
                    context_max_chars=context_max_chars,
                    max_doc_chars=max_doc_chars,
                )
        except Exception as exc:  # 管线异常转成稳定结构化契约，不把堆栈炸给调用方
            logger.exception(
                "rag_retrieve failed tenant=%s kb=%s request=%s",
                context.tenant_id,
                context.kb_id,
                context.request_id,
            )
            record_error = getattr(metrics, "record_node_error", None)
            if record_error is not None:
                record_error(
                    "rag_retrieve",
                    tenant_id=context.tenant_id,
                    kb_id=context.kb_id,
                )
            return {
                "ok": False,
                "status": RetrieveStatus.ERROR,
                "message": f"{type(exc).__name__}: {exc}",
                "cache_hit": False,
                "tenant_id": context.tenant_id,
                "kb_id": context.kb_id,
                "request_id": context.request_id,
                "user_id": context.user_id,
                "count": 0,
                "docs": [],
                "rewritten_query": query,
                "diagnostics": {},
                "trace_id": trace_id_value,
            }
        finally:
            telemetry.reset_context(trace_token)
        return {
            "ok": result.status != RetrieveStatus.ERROR,
            "status": result.status,
            "message": result.message,
            "cache_hit": result.cache_hit,
            "tenant_id": context.tenant_id,
            "kb_id": context.kb_id,
            "request_id": context.request_id,
            "user_id": context.user_id,
            "count": len(result.docs),
            "docs": result.docs,
            "rewritten_query": result.rewritten_query or query,
            "diagnostics": result.diagnostics,
            "trace_id": trace_id_value,
        }

    return server


__all__ = ["create_mcp_server"]
