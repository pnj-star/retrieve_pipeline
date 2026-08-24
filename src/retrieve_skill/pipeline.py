"""基于 common_core provider 的可执行检索管线。

RagPipeline 串联整个 RAG 检索流程：
  检索缓存检查 → 查询改写（默认 off）→ 混合检索（稀疏 + 稠密）
  → RRF 融合 → 交叉编码器精排 → 相关性阈值判断 → 回写父块引用缓存
  → 批量回源父块、版本校验与 token 预算截断
并在过程中上报各环节的耗时与错误指标。回答生成与护栏归属 agent 编排层，
``common_core.rag`` 提供答案侧公共机制供其复用。
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import time
from collections.abc import Iterable
from typing import Any

from common_core.config import RuntimeConfig, env_int
from common_core.context import AgentContext
from common_core.instrumentation import trace_node
from common_core.observability import Observability
from common_core.providers import LocalEmbedder, MilvusVectorStore, OpenAICompatibleLLM, RedisCache
from common_core.providers.vector import build_filter_expr

from .parent_docs import (
    assemble_parent_refs,
    build_parent_refs,
    default_token_counter,
    filter_parent_refs,
    DEFAULT_MAX_CONTEXT_TOKENS,
    DEFAULT_MAX_DOC_TOKENS,
    validate_parent_refs,
)
from .results import RetrieveResult, RetrieveStatus
from .stages import (
    QueryRewriteResult,
    QueryRewriter,
    Reranker,
    RetrievalCache,
)

logger = logging.getLogger(__name__)

# 混合检索默认返回的文本字段。不同集合的 schema 可通过
# runtime.vector.text_output_fields 或调用方显式传入 output_fields 覆盖，避免查询报错。
# id 为 Milvus 主键（子块身份），始终随结果返回（见 providers.vector._to_docs），
# 供评估/回答节点回查具体子块；parent_id 依赖集合 schema，需按需加入
# MILVUS_OUTPUT_FIELDS（默认 .env.example 已包含），缺该字段的集合请勿硬编码。
DEFAULT_TEXT_OUTPUT_FIELDS: tuple[str, ...] = (
    "id",
    "content",
    "parent_id",
    "chunk_index",
    "tenant_id",
    "kb_id",
    "doc_version",
)

# retrieve_context 的内部契约：先取齐聚合父块所需字段，再按请求投影。
# 换集合时必须保证这些标量字段存在；score / ce_score 由检索和精排阶段生成。
INTERNAL_CHILD_OUTPUT_FIELDS: tuple[str, ...] = (
    "id",
    "content",
    "parent_id",
    "chunk_index",
    "tenant_id",
    "kb_id",
    "doc_version",
)

_WHITESPACE_PATTERN = re.compile(r"\s+")


def _normalize_cache_text(value: Any) -> str:
    """缓存 material 使用的稳定文本规范化：压缩空白并去除首尾空白。"""
    return _WHITESPACE_PATTERN.sub(" ", str(value or "")).strip()


def _prompt_digest(*values: Any) -> str:
    """把 prompt 配置压缩成短摘要，避免完整 prompt 撑爆缓存 material。"""
    raw = "\x1f".join(str(value or "") for value in values)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _callable_identity(value: Any) -> str:
    """生成可调用对象的稳定身份；自定义评分后端应显式提供版本号。"""
    name = getattr(
        value,
        "__qualname__",
        getattr(value, "__name__", type(value).__name__),
    )
    module = getattr(value, "__module__", type(value).__module__)
    return f"{module}.{name}"


def _finite_score(value: Any) -> float | None:
    """读取有限数值分数；缺失、非数或 NaN/inf 都视为无效。"""
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    return score if math.isfinite(score) else None


def _best_ce_score(docs: list[dict[str, Any]]) -> float | None:
    """取候选集中最高有效精排分；只用于诊断，不参与质量判断。"""
    scores = [
        score
        for doc in docs
        if (score := _finite_score(doc.get("ce_score"))) is not None
    ]
    return max(scores) if scores else None


def merge_retrieval_docs(
    results: Iterable[Iterable[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """把多个查询的检索结果按文档身份合并，保留首次出现顺序与最高分数。

    参数:
        results: 多个检索批次的可迭代对象；每个批次是文档字典的列表。
            文档按 "id" 字段判重，若 "id" 缺失则改用
            (tenant_id, kb_id, source, content) 的组合作为身份键。

    返回:
        合并去重后的文档列表；同身份文档保留首次出现顺序，各分数取最高值。
    """
    merged: dict[Any, dict[str, Any]] = {}
    for batch in results:
        for doc in batch:
            key = doc.get("id")
            if not isinstance(key, (str, int)):
                key = tuple(
                    str(doc.get(field_name) or "")
                    for field_name in ("tenant_id", "kb_id", "source", "content")
                )
            existing = merged.get(key)
            if existing is None:
                merged[key] = dict(doc)
                continue
            for score_field in ("score", "hybrid_score", "dense_score", "sparse_score"):
                if score_field not in doc:
                    continue
                try:
                    new_score = float(doc[score_field])
                    old_score = float(existing.get(score_field) or 0)
                except (TypeError, ValueError):
                    continue
                if new_score > old_score:
                    existing[score_field] = doc[score_field]
    return list(merged.values())


class RagPipeline:
    """RAG 检索管线。

    所有组件依赖都可以传入（便于测试注入 fake），未传入的参数会从
    runtime 配置构建默认实现：
      - llm: OpenAI 兼容 LLM（用于查询改写）
      - vector: Milvus 向量库（混合检索）
      - embedder: 本地 Embedding 模型
      - cache: Redis 底层缓存（服务于检索缓存）
      - reranker / retrieval_cache: 可选的重排与检索缓存阶段
    """

    def __init__(
        self,
        runtime: RuntimeConfig | None = None,
        *,
        llm: OpenAICompatibleLLM | None = None,
        vector: MilvusVectorStore | None = None,
        embedder: LocalEmbedder | None = None,
        cache: RedisCache | None = None,
        metrics: Observability | None = None,
        reranker: Reranker | None = None,
        retrieval_cache: RetrievalCache | None = None,
        parent_store: Any | None = None,
        query_rewriter: QueryRewriter | None = None,
        min_relevance: float | None = None,
        tenant_filter: bool = True,
        default_output_fields: Iterable[str] | None = None,
        data_version: str | None = None,
        count_tokens: Any | None = None,
        default_context_max_tokens: int | None = None,
        default_max_doc_tokens: int | None = None,
    ) -> None:
        """组装整条 RAG 管线：未显式注入的组件从 runtime 配置默认构建。

        参数:
            runtime: common_core 的运行配置；未传则从环境变量构建。
            llm: OpenAI 兼容 LLM 实例，用于查询改写；未传用 runtime.llm 默认构建。
            vector: Milvus 向量库实例，用于混合检索；未传则默认构建。
            embedder: 本地 Embedding 模型，用于生成查询向量；未传则默认构建。
            cache: Redis 底层缓存实例（供检索缓存复用连接）；未传则默认构建。
            metrics: 可观测性 / 指标上报对象；为 None 则不上报指标。
            reranker: 交叉编码器精排器；None 表示不启用精排。
            retrieval_cache: 检索结果缓存实例（query → 精排后父块引用）；None 表示不启用。
            parent_store: 父块权威内容存储；retrieve_context 必须配置。
            query_rewriter: 查询改写器；None 则用默认 QueryRewriter。
            min_relevance: 精排后的最低相关性阈值；None 时用 runtime 默认值。
            tenant_filter: 是否强制按 tenant_id/kb_id 做隔离过滤，默认 True。
            default_output_fields: 检索默认返回的字段；None 用 runtime/内置默认值。
            data_version: 知识库数据版本；变化后检索缓存自然失效。
            default_context_max_tokens: 默认父块上下文 token 预算。
            default_max_doc_tokens: 默认单篇父块 token 预算；None 表示自动取总量一半。
        """
        self.runtime = runtime or RuntimeConfig.from_env()
        self.metrics = metrics
        self.llm = llm or OpenAICompatibleLLM(self.runtime.llm, metrics=metrics)
        self.vector = vector or MilvusVectorStore(
            self.runtime.vector,
            max_workers=self.runtime.retrieval.hybrid_max_workers,
        )
        self.embedder = embedder or LocalEmbedder(config=self.runtime.llm)
        self.cache = cache or RedisCache(self.runtime.cache)
        self.reranker = reranker
        self.retrieval_cache = retrieval_cache
        self.parent_store = parent_store
        if (
            self.retrieval_cache is not None
            and getattr(self.retrieval_cache, "metrics", None) is None
            and metrics is not None
        ):
            self.retrieval_cache.metrics = metrics
        self.query_rewriter = query_rewriter or QueryRewriter(
            self.llm,
            self.runtime.query_rewrite,
            metrics=metrics,
        )
        # 精排后的最低相关性阈值：显式参数优先，否则取运行时配置默认值。
        self.min_relevance = (
            min_relevance
            if min_relevance is not None
            else self.runtime.retrieval.min_relevance
        )
        self.tenant_filter = tenant_filter
        self.data_version = data_version
        self.count_tokens = count_tokens or default_token_counter
        self.default_context_max_tokens = (
            default_context_max_tokens
            if default_context_max_tokens is not None
            else env_int("CONTEXT_MAX_TOKENS", default=DEFAULT_MAX_CONTEXT_TOKENS)
        )
        configured_max_doc_tokens = (
            default_max_doc_tokens
            if default_max_doc_tokens is not None
            else env_int("MAX_DOC_TOKENS", default=0)
        )
        self.default_max_doc_tokens = (
            configured_max_doc_tokens if configured_max_doc_tokens > 0 else None
        )
        self.default_output_fields = tuple(
            default_output_fields
            if default_output_fields is not None
            else (self.runtime.vector.text_output_fields or DEFAULT_TEXT_OUTPUT_FIELDS)
        )

    def _context_labels(self, context: AgentContext | None) -> tuple[str, str, str, str]:
        """取出用于指标 / 日志打点的租户上下文标签。

        参数:
            context: agent 上下文；可能为 None（无租户信息）。

        返回:
            (tenant_id, kb_id, request_id, session_id) 四元组，缺省时为空字符串。
        """
        if context is None:
            return "", "", "", ""
        return (
            context.tenant_id,
            context.kb_id,
            context.request_id,
            context.session_id,
        )

    def _observe(self, node: str, start: float, context: AgentContext | None) -> None:
        """记录某个环节（node）的耗时指标。

        参数:
            node: 环节名称，如 "retrieve" / "query_rewrite" / "rerank"。
            start: 该环节开始时刻（time.perf_counter() 的返回值）。
            context: agent 上下文，用于提取租户标签；可为 None。
        """
        if self.metrics is None:
            return
        tenant_id, kb_id, _request_id, _session_id = self._context_labels(context)
        self.metrics.record_node_duration(
            node,
            time.perf_counter() - start,
            tenant_id=tenant_id,
            kb_id=kb_id,
        )

    def _observe_error(self, node: str, context: AgentContext | None) -> None:
        """记录某个环节（node）的错误指标。

        参数:
            node: 环节名称。
            context: agent 上下文，用于提取租户标签；可为 None。
        """
        if self.metrics is None:
            return
        tenant_id, kb_id, _request_id, _session_id = self._context_labels(context)
        self.metrics.record_node_error(
            node,
            tenant_id=tenant_id,
            kb_id=kb_id,
        )

    def _build_scope_filter(
        self,
        context: AgentContext | None,
        filter_expr: str | None,
    ) -> str:
        """合并租户 / 知识库隔离条件与可选的业务过滤表达式。

        最终产出形如 `tenant_id == "x" and kb_id == "y" and (业务条件)` 的表达式，
        传给 Milvus 做过滤检索，确保检索永远不会跨越租户范围。

        参数:
            context: agent 上下文；非空且 tenant_filter 开启时生成租户隔离条件。
            filter_expr: 调用方传入的额外业务过滤表达式；为空字符串则忽略。

        返回:
            合并后的 Milvus 过滤表达式字符串；无任何条件时返回空字符串。
        """
        parts: list[str] = []
        if self.tenant_filter and context is not None:
            # 强制 tenant_id + kb_id 精确匹配，租赁户隔离的核心防线
            scope_expr = build_filter_expr(
                {
                    "tenant_id": context.tenant_id,
                    "kb_id": context.kb_id,
                }
            )
            if scope_expr:
                parts.append(scope_expr)
        extra = (filter_expr or "").strip()
        if extra:
            parts.append(f"({extra})")
        return " and ".join(parts)

    async def _resolve_rewrite(
        self,
        query: str,
        context: AgentContext | None,
        *,
        query_rewrite_mode: str | None,
        rewrite_query: str | None,
    ) -> QueryRewriteResult:
        """只解析一次“最终生效的查询”，并上报改写耗时。

        - 显式传入了 ``rewrite_query`` 时直接采用它，跳过 LLM；
        - 否则按配置的改写策略执行一次查询改写。
        该方法被 ``retrieve()`` 统一调用，保证整条管线在同一处集中处理改写，
        避免检索侧与潜在调用方出现逻辑分叉。

        参数:
            query: 原始用户查询文本。
            context: agent 上下文，改写器可能按租户取改写配置。
            query_rewrite_mode: 本次显式指定的改写模式；None 时按配置决定。
            rewrite_query: 显式传入的改写后查询；非空时直接采用并跳过 LLM。

        返回:
            QueryRewriteResult，包含最终生效查询、改写模式与变体列表等信息。
        """
        if rewrite_query is not None and str(rewrite_query).strip():
            effective_query = str(rewrite_query).strip()
            return QueryRewriteResult(
                mode="explicit",
                original_query=query,
                rewritten_query=effective_query,
                query_variants=[effective_query],
            )
        start_rewrite = time.perf_counter()
        try:
            return await self.query_rewriter.rewrite(
                query,
                context,
                mode=query_rewrite_mode,
            )
        except Exception:
            self._observe_error("query_rewrite", context)
            raise
        finally:
            self._observe("query_rewrite", start_rewrite, context)

    async def _retrieve_resolved(
        self,
        query: str,
        context: AgentContext | None,
        result: QueryRewriteResult,
        *,
        collection: str,
        top_k: int,
        output_fields: Iterable[str],
        filter_expr: str | None,
    ) -> list[dict[str, Any]]:
        """基于已经解析好的改写结果执行稀疏 + 稠密的混合检索。

        改写结果里通常只有一个查询；展开模式（query_expansion）会带多个变体。
        - 单个查询：直接做一次混合检索；
        - 多个变体：对每个变体各检索一次，再用 ``merge_retrieval_docs`` 按文档
          身份合并、取最高分，最终截断到 top_k，提升召回覆盖面。

        参数:
            query: 原始用户查询文本（用于追溯）。
            context: agent 上下文，用于租户隔离过滤。
            result: 已解析好的改写结果，含最终查询与查询变体列表。
            collection: 目标 Milvus 集合名。
            top_k: 每个查询返回的候选条数。
            output_fields: 检索需要返回的字段列表。
            filter_expr: 额外的业务过滤表达式；为空则不附加。

        返回:
            融合/合并并按 top_k 截断后的候选文档列表。
        """
        scope_filter = self._build_scope_filter(context, filter_expr)
        field_list = list(output_fields)
        search_queries = result.query_variants or [result.rewritten_query]
        if len(search_queries) == 1:
            embedding = await self.embedder.embed(search_queries[0])
            return await self.vector.a_search_hybrid(
                collection,
                search_queries[0],
                embedding,
                top_k=top_k,
                rrf_top_k=self.runtime.retrieval.rrf_top_k,
                rrf_k=self.runtime.retrieval.rrf_k,
                output_fields=field_list,
                filter_expr=scope_filter,
            )
        batches: list[list[dict[str, Any]]] = []
        for search_query in search_queries:
            embedding = await self.embedder.embed(search_query)
            batch = await self.vector.a_search_hybrid(
                collection,
                search_query,
                embedding,
                top_k=top_k,
                rrf_top_k=self.runtime.retrieval.rrf_top_k,
                rrf_k=self.runtime.retrieval.rrf_k,
                output_fields=field_list,
                filter_expr=scope_filter,
            )
            batches.append(batch)
        return merge_retrieval_docs(batches)[:top_k]

    async def retrieve(
        self,
        query: str,
        context: AgentContext | None = None,
        *,
        collection_name: str | None = None,
        top_k: int | None = None,
        output_fields: Iterable[str] | None = None,
        filter_expr: str | None = None,
        query_rewrite_mode: str | None = None,
        rewrite_query: str | None = None,
        rewrite_trace: dict[str, Any] | None = None,
        _resolved_rewrite: QueryRewriteResult | None = None,
    ) -> list[dict[str, Any]]:
        """执行查询改写（默认关闭）与混合检索，返回候选文档列表。

        查询改写支持 ``off`` / ``llm_rewrite`` / ``query_expansion``；
        改写失败自动回退原始查询，不影响检索可用性。显式传入 ``rewrite_query``
        时跳过改写，直接用该文本检索。改写明细会写入 ``rewrite_trace``（若有）。

        参数:
            query: 用户查询文本。
            context: agent 上下文；可为 None（表示无租户隔离）。
            collection_name: 覆盖默认的 Milvus 集合名；None 用 runtime 配置。
            top_k: 返回候选条数；None 用 runtime 默认值。
            output_fields: 覆盖默认返回字段；None 用管线默认字段。
            filter_expr: 额外的业务过滤表达式。
            query_rewrite_mode: 本次指定的改写模式；None 按配置决定。
            rewrite_query: 显式改写后的查询；非空时跳过内置改写。
            rewrite_trace: 可选字典，改写明细会写入其中供排查。
            _resolved_rewrite: 内部复用已解析改写结果，调用方通常不需要传。

        返回:
            候选文档字典列表（每项含检索字段与分数等信息）。
        """
        collection = collection_name or self.runtime.vector.text_collection
        if not collection:
            raise ValueError("text_collection is not configured")
        tenant_id, kb_id, request_id, _session_id = self._context_labels(context)
        trace = {} if rewrite_trace is None else rewrite_trace
        start = time.perf_counter()
        try:
            # 给检索打 span（含租户上下文），异常时 span 记 ERROR 后向上抛
            with trace_node(
                "retrieve",
                tenant_id=tenant_id,
                kb_id=kb_id,
                request_id=request_id,
            ):
                if _resolved_rewrite is not None:
                    result = _resolved_rewrite
                else:
                    result = await self._resolve_rewrite(
                        query,
                        context,
                        query_rewrite_mode=query_rewrite_mode,
                        rewrite_query=rewrite_query,
                    )
                trace.update(result.to_trace())
                docs = await self._retrieve_resolved(
                    query,
                    context,
                    result,
                    collection=collection,
                    top_k=top_k or self.runtime.retrieval.top_k,
                    output_fields=(
                        list(output_fields)
                        if output_fields is not None
                        else list(self.default_output_fields)
                    ),
                    filter_expr=filter_expr,
                )
        except Exception:
            self._observe_error("retrieve", context)
            raise
        finally:
            self._observe("retrieve", start, context)
        if not docs and self.metrics is not None:
            # 检索为空时单独打点，方便监控无内容命中率
            tenant_id, kb_id, _request_id, _session_id = self._context_labels(context)
            self.metrics.record_retrieval_empty(
                tenant_id=tenant_id,
                kb_id=kb_id,
            )
        return docs

    def _cache_material(
        self,
        *,
        query: str,
        context: AgentContext | None,
        collection_name: str | None,
        top_k: int | None,
        filter_expr: str | None,
        min_relevance: float,
        query_rewrite_mode: str | None = None,
        rewrite_query: str | None = None,
    ) -> str:
        """构造 v3 父块引用缓存的完整检索签名。

        material 只包含影响候选集合的语义；tenant/kb 由底层 RedisCache 追加，
        request/session/user 与展示投影、上下文预算不参与 key。
        """
        effective_mode = self.query_rewriter.resolve_mode(
            context,
            requested_mode=query_rewrite_mode,
        )
        explicit_rewrite = _normalize_cache_text(rewrite_query) or None
        rewrite_config = self.query_rewriter.config
        needs_rewriter_signature = not explicit_rewrite and effective_mode != "off"
        rerank_config = self.runtime.retrieval
        custom_scoring = getattr(self.reranker, "score_fn", None)
        explicit_version = str(getattr(self.reranker, "scoring_version", "") or "")
        if custom_scoring:
            scoring_backend = f"custom:{_callable_identity(custom_scoring)}"
            scoring_version = explicit_version or "unversioned"
        else:
            scoring_backend = "builtin_cross_encoder"
            scoring_version = "rank_docs_v1"
        reranker_payload: dict[str, Any] = {
            "model": str(
                getattr(self.reranker, "model_name", rerank_config.rerank_model)
                or rerank_config.rerank_model
            ),
            "top_k": int(
                getattr(self.reranker, "top_k", rerank_config.rerank_top_k)
            ),
            "ce_weight": float(
                getattr(self.reranker, "ce_weight", rerank_config.rerank_ce_weight)
            ),
            "retrieval_weight": float(
                getattr(
                    self.reranker,
                    "retrieval_weight",
                    rerank_config.rerank_retrieval_weight,
                )
            ),
            "backend": scoring_backend,
            "scoring_version": scoring_version,
        }
        payload = {
            "schema": "rag_retrieval_cache_v3",
            "query": _normalize_cache_text(query),
            "rewrite": {
                # 显式改写时 LLM 配置不影响本次召回，因此不把 rewriter 签名掺进来。
                "mode": "explicit" if explicit_rewrite else effective_mode,
                "explicit_query": explicit_rewrite,
            },
            "collection": collection_name or self.runtime.vector.text_collection,
            "top_k": top_k or self.runtime.retrieval.top_k,
            "rrf_top_k": self.runtime.retrieval.rrf_top_k,
            "rrf_k": self.runtime.retrieval.rrf_k,
            "filter_expr": _normalize_cache_text(filter_expr),
            "min_relevance": float(min_relevance),
            "embedding_model": str(
                getattr(self.embedder, "model_name", "")
                or self.runtime.llm.embedding_model
            ),
            "reranker": reranker_payload,
            # off / explicit 不执行 LLM 改写；rewriter 配置变化不应打散缓存。
            "query_rewriter": None if not needs_rewriter_signature else {
                "model": (
                    getattr(rewrite_config, "llm_model", "")
                    or self.runtime.llm.model
                ),
                "temperature": float(getattr(rewrite_config, "temperature", 0.0)),
                "max_tokens": int(getattr(rewrite_config, "max_tokens", 256)),
                "expand_count": int(getattr(rewrite_config, "expand_count", 2)),
                "prompt_digest": _prompt_digest(
                    getattr(rewrite_config, "rewrite_prompt", ""),
                    getattr(rewrite_config, "expansion_prompt", ""),
                ),
            },
            "data_version": str(self.data_version or ""),
            "parent_store": {
                "schema": "rag_parent_ref_cache_v1",
                "type": type(self.parent_store).__name__ if self.parent_store else "",
                "database": str(getattr(getattr(self.parent_store, "config", None), "database", "") or ""),
                "table": str(getattr(getattr(self.parent_store, "config", None), "table", "") or ""),
                "status": str(getattr(getattr(self.parent_store, "config", None), "status", "") or "active"),
            },
        }
        return json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    async def _rerank_docs(
        self,
        docs: list[dict[str, Any]],
        query: str,
        context: AgentContext | None,
        *,
        tenant_id: str,
        kb_id: str,
        request_id: str,
    ) -> list[dict[str, Any]]:
        """对混合检索结果做交叉编码器精排；模型故障向上抛出。"""
        if self.reranker is None:
            return docs
        start_rerank = time.perf_counter()
        try:
            with trace_node(
                "rerank",
                tenant_id=tenant_id,
                kb_id=kb_id,
                request_id=request_id,
            ):
                ranked = await self.reranker.arank(
                    query,
                    docs,
                    tenant_id=tenant_id,
                    kb_id=kb_id,
                )
        except Exception:
            self._observe_error("rerank", context)
            raise
        finally:
            self._observe("rerank", start_rerank, context)
        return list(ranked or [])

    @staticmethod
    def _project_parent_docs(
        docs: list[dict[str, Any]],
        output_fields: Iterable[str] | None,
    ) -> tuple[list[dict[str, Any]], list[str]]:
        """在父块聚合后投影请求字段；缺失字段会被记录而不是静默编造。"""
        if output_fields is None:
            return docs, []
        fields = list(dict.fromkeys(str(field) for field in output_fields if str(field)))
        if not fields:
            return [], []
        projected: list[dict[str, Any]] = []
        missing: set[str] = set()
        for doc in docs:
            item: dict[str, Any] = {}
            for field in fields:
                if field in doc:
                    item[field] = doc[field]
                else:
                    missing.add(field)
            projected.append(item)
        return projected, sorted(missing)

    def _diagnostics(
        self,
        *,
        reason: str | None = None,
        candidates: list[dict[str, Any]] | None = None,
        children: list[dict[str, Any]] | None = None,
        parents: list[dict[str, Any]] | None = None,
        parent_refs: list[dict[str, Any]] | None = None,
        missing_parent_count: int | None = None,
        version_mismatch_count: int | None = None,
        context_tokens: int | None = None,
        missing_output_fields: list[str] | None = None,
    ) -> dict[str, Any]:
        """构造不含候选正文的检索诊断摘要。"""
        payload: dict[str, Any] = {}
        if reason:
            payload["reason"] = reason
        if candidates is not None:
            payload["candidate_count"] = len(candidates)
            best = _best_ce_score(candidates)
            if best is not None:
                payload["best_ce_score"] = best
        if children is not None:
            payload["qualified_child_count"] = len(children)
        if parents is not None:
            payload["parent_doc_count"] = len(parents)
        if parent_refs is not None:
            payload["qualified_parent_ref_count"] = len(parent_refs)
        if missing_parent_count is not None:
            payload["missing_parent_count"] = missing_parent_count
        if version_mismatch_count is not None:
            payload["version_mismatch_count"] = version_mismatch_count
        if context_tokens is not None:
            payload["context_tokens"] = context_tokens
        if missing_output_fields:
            payload["missing_output_fields"] = missing_output_fields
        return payload

    async def retrieve_context(
        self,
        query: str,
        context: AgentContext | None = None,
        *,
        collection_name: str | None = None,
        top_k: int | None = None,
        output_fields: Iterable[str] | None = None,
        filter_expr: str | None = None,
        query_rewrite_mode: str | None = None,
        rewrite_query: str | None = None,
        min_relevance: float | None = None,
        context_max_chars: int | None = None,
        max_doc_chars: int | None = None,
        context_max_tokens: int | None = None,
        max_doc_tokens: int | None = None,
    ) -> RetrieveResult:
        """执行完整检索侧管道，返回通过精排阈值的父块上下文。

        成功结果先以父块引用形式写入缓存，再从 MySQL 回源父块正文。缓存命中时
        不再改写、召回或精排，但仍会回源 MySQL 并重做版本校验与 token 截断。
        低于阈值、reranker/parent store 未配置或故障时不返回候选原文，也不写缓存。

        参数:
            query: 用户查询文本。
            context: agent 上下文，用于租户/知识库隔离和缓存隔离。
            collection_name/top_k/filter_expr/query_rewrite_mode/rewrite_query:
                影响底层候选集合的参数。
            output_fields: 最终父块文档的字段投影；不参与底层检索与缓存 key。
            min_relevance: 本次精排相关性阈值覆盖；None 用管线默认。
            context_max_chars / max_doc_chars: 可选字符预算。
            context_max_tokens / max_doc_tokens: 父块上下文 token 预算。

        返回:
            ``RetrieveResult``：成功路径的 docs 是父块粒度上下文；失败和无上下文
            路径的 docs 为空。
        """
        tenant_id, kb_id, request_id, _session_id = self._context_labels(context)
        threshold = (
            float(min_relevance)
            if min_relevance is not None
            else float(self.min_relevance)
        )
        if threshold < 0.0 or threshold > 1.0:
            return RetrieveResult(
                RetrieveStatus.ERROR,
                docs=[],
                rewritten_query=query,
                message="min_relevance must be between 0 and 1",
                diagnostics=self._diagnostics(reason="invalid_threshold"),
            )
        if self.reranker is None:
            return RetrieveResult(
                RetrieveStatus.ERROR,
                docs=[],
                rewritten_query=query,
                message="reranker is required for retrieve_context",
                diagnostics=self._diagnostics(reason="reranker_not_configured"),
            )
        if self.parent_store is None:
            return RetrieveResult(
                RetrieveStatus.ERROR,
                docs=[],
                rewritten_query=query,
                message="parent store is required for retrieve_context",
                diagnostics=self._diagnostics(reason="parent_store_not_configured"),
            )
        if context is None or not tenant_id or not kb_id:
            return RetrieveResult(
                RetrieveStatus.ERROR,
                docs=[],
                rewritten_query=query,
                message="tenant_id and kb_id are required for parent retrieval",
                diagnostics=self._diagnostics(reason="invalid_scope"),
            )
        context_max_tokens = (
            self.default_context_max_tokens
            if context_max_tokens is None
            else context_max_tokens
        )
        if max_doc_tokens is None:
            max_doc_tokens = self.default_max_doc_tokens

        cache_material = self._cache_material(
            query=query,
            context=context,
            collection_name=collection_name,
            top_k=top_k,
            filter_expr=filter_expr,
            min_relevance=threshold,
            query_rewrite_mode=query_rewrite_mode,
            rewrite_query=rewrite_query,
        )

        # 第一步：命中父块引用缓存后，仍回源 MySQL 并重建展示视图。
        if self.retrieval_cache is not None:
            cached = self.retrieval_cache.get(cache_material, context=context)
            refs = (
                validate_parent_refs(
                    cached,
                    threshold=threshold,
                    context=context,
                )
                if cached is not None
                else None
            )
            if refs is not None:
                start_fetch = time.perf_counter()
                try:
                    parent_rows = await self.parent_store.aget_parent_blocks(
                        [ref["parent_id"] for ref in refs],
                        context=context,
                    )
                except Exception as exc:
                    self._observe_error("parent_fetch", context)
                    logger.exception("parent store failed; returning fail-closed result")
                    return RetrieveResult(
                        RetrieveStatus.ERROR,
                        docs=[],
                        rewritten_query=rewrite_query or query,
                        message=f"parent store unavailable: {exc}",
                        diagnostics=self._diagnostics(
                            reason="parent_store_unavailable",
                            parent_refs=refs,
                        ),
                    )
                finally:
                    self._observe("parent_fetch", start_fetch, context)

                start_assembly = time.perf_counter()
                parents, assembly_stats = assemble_parent_refs(
                    refs,
                    parent_rows,
                    context=context,
                    count_tokens=self.count_tokens,
                    context_max_chars=context_max_chars,
                    max_doc_chars=max_doc_chars,
                    context_max_tokens=context_max_tokens,
                    max_doc_tokens=max_doc_tokens,
                )
                self._observe("parent_aggregation", start_assembly, context)
                if not parents:
                    self.retrieval_cache.delete(cache_material, context=context)
                    reason = (
                        "parent_version_mismatch"
                        if assembly_stats["version_mismatch_count"]
                        else "parent_not_found"
                    )
                    return RetrieveResult(
                        RetrieveStatus.NO_CONTEXT,
                        docs=[],
                        rewritten_query=rewrite_query or query,
                        message="cached parents are no longer available",
                        diagnostics=self._diagnostics(
                            reason=reason,
                            parent_refs=refs,
                            parents=[],
                        ),
                    )
                projected, missing_fields = self._project_parent_docs(
                    parents,
                    output_fields,
                )
                # MySQL 中部分父块已删除或版本变化时，同步清理失效引用。
                if (
                    assembly_stats["missing_parent_count"]
                    or assembly_stats["version_mismatch_count"]
                ):
                    current_refs = filter_parent_refs(refs, parent_rows, context=context)
                    if current_refs:
                        self.retrieval_cache.put(cache_material, current_refs, context=context)
                    else:
                        self.retrieval_cache.delete(cache_material, context=context)
                return RetrieveResult(
                    RetrieveStatus.RETRIEVED_CACHE,
                    docs=projected,
                    rewritten_query=rewrite_query or query,
                    cache_hit=True,
                    message="retrieved from parent reference cache",
                    diagnostics=self._diagnostics(
                        parents=parents,
                        parent_refs=refs,
                        missing_output_fields=missing_fields,
                    ),
                )
            if cached is not None:
                logger.warning("invalid parent reference cache treated as miss")

        # 第二步：缓存未命中 → 查询改写 + 混合检索 + RRF + top_k。
        rewrite_trace: dict[str, Any] = {}
        docs = await self.retrieve(
            query,
            context,
            collection_name=collection_name,
            top_k=top_k,
            output_fields=INTERNAL_CHILD_OUTPUT_FIELDS,
            filter_expr=filter_expr,
            query_rewrite_mode=query_rewrite_mode,
            rewrite_query=rewrite_query,
            rewrite_trace=rewrite_trace,
        )
        rewritten_query = rewrite_trace.get("rewritten_query") or query

        # 第三步：混合检索两边都空 → 没有可用上下文。
        if not docs:
            return RetrieveResult(
                RetrieveStatus.NO_CONTEXT,
                docs=[],
                rewritten_query=rewritten_query,
                message="no context",
                diagnostics=self._diagnostics(
                    reason="empty_recall",
                    candidates=[],
                    children=[],
                ),
            )

        # 第四步：交叉编码器精排；模型故障必须 fail-closed。
        try:
            ranked = await self._rerank_docs(
                docs,
                rewritten_query,
                context,
                tenant_id=tenant_id,
                kb_id=kb_id,
                request_id=request_id,
            )
        except Exception as exc:
            logger.exception("rerank failed; returning fail-closed result")
            return RetrieveResult(
                RetrieveStatus.ERROR,
                docs=[],
                rewritten_query=rewritten_query,
                message=f"reranker unavailable: {exc}",
                diagnostics=self._diagnostics(
                    reason="reranker_unavailable",
                    candidates=docs,
                    children=[],
                ),
            )

        scored_children: list[dict[str, Any]] = []
        seen_child_ids: set[str] = set()
        for doc in ranked:
            child_id = str(doc.get("id", "") or "").strip()
            content = str(doc.get("content", "") or "").strip()
            ce_score = _finite_score(doc.get("ce_score"))
            if (
                not child_id
                or child_id in seen_child_ids
                or not content
                or ce_score is None
                or ce_score < 0.0
                or ce_score > 1.0
            ):
                return RetrieveResult(
                    RetrieveStatus.ERROR,
                    docs=[],
                    rewritten_query=rewritten_query,
                    message="reranker returned an invalid scored child",
                    diagnostics=self._diagnostics(
                        reason="invalid_reranked_child",
                        candidates=docs,
                        children=[],
                    ),
                )
            seen_child_ids.add(child_id)
            scored_children.append(dict(doc))

        qualified_children = [
            doc for doc in scored_children
            if float(doc["ce_score"]) >= threshold
        ]
        if not qualified_children:
            return RetrieveResult(
                RetrieveStatus.NO_CONTEXT,
                docs=[],
                rewritten_query=rewritten_query,
                message="no relevant docs above threshold",
                diagnostics=self._diagnostics(
                    reason="below_threshold",
                    candidates=scored_children,
                    children=[],
                ),
            )

        # 第五步：按 parent_id 去重，生成精排后的父块引用。
        parent_refs = build_parent_refs(qualified_children)

        # 第六步：批量回源权威父块；MySQL 故障必须 fail-closed。
        start_fetch = time.perf_counter()
        try:
            parent_rows = await self.parent_store.aget_parent_blocks(
                [ref["parent_id"] for ref in parent_refs],
                context=context,
            )
        except Exception as exc:
            self._observe_error("parent_fetch", context)
            logger.exception("parent store failed; returning fail-closed result")
            return RetrieveResult(
                RetrieveStatus.ERROR,
                docs=[],
                rewritten_query=rewritten_query,
                message=f"parent store unavailable: {exc}",
                diagnostics=self._diagnostics(
                    reason="parent_store_unavailable",
                    parent_refs=parent_refs,
                ),
            )
        finally:
            self._observe("parent_fetch", start_fetch, context)

        valid_parent_refs = filter_parent_refs(
            parent_refs,
            parent_rows,
            context=context,
        )
        if not valid_parent_refs:
            version_mismatch = any(
                str(row.get("doc_version", "") or "").strip()
                != str(ref.get("doc_version", "") or "").strip()
                for ref, row in (
                    (ref, parent_rows.get(ref["parent_id"]))
                    for ref in parent_refs
                )
                if row is not None
            )
            return RetrieveResult(
                RetrieveStatus.NO_CONTEXT,
                docs=[],
                rewritten_query=rewritten_query,
                message="no available parent docs",
                diagnostics=self._diagnostics(
                    reason=(
                        "parent_version_mismatch"
                        if version_mismatch
                        else "parent_not_found"
                    ),
                    candidates=scored_children,
                    children=qualified_children,
                    parents=[],
                ),
            )

        # 只缓存 MySQL 当前确认存在的父块引用。
        if self.retrieval_cache is not None:
            self.retrieval_cache.put(cache_material, valid_parent_refs, context=context)

        # 第七步：每次按请求预算重建父块视图，保证不同展示需求共享同一份候选集。
        start_assembly = time.perf_counter()
        parents, assembly_stats = assemble_parent_refs(
            valid_parent_refs,
            parent_rows,
            context=context,
            count_tokens=self.count_tokens,
            context_max_chars=context_max_chars,
            max_doc_chars=max_doc_chars,
            context_max_tokens=context_max_tokens,
            max_doc_tokens=max_doc_tokens,
        )
        projected, missing_fields = self._project_parent_docs(parents, output_fields)
        self._observe("parent_aggregation", start_assembly, context)
        return RetrieveResult(
            RetrieveStatus.RETRIEVED,
            docs=projected,
            rewritten_query=rewritten_query,
            message="ok",
            diagnostics=self._diagnostics(
                candidates=scored_children,
                children=qualified_children,
                parents=parents,
                **assembly_stats,
                missing_output_fields=missing_fields,
            ),
        )
