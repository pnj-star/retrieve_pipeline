"""基于 common_core provider 的可执行检索管线。

RagPipeline 串联整个 RAG 检索流程：
  检索缓存检查 → 查询改写（默认 off）→ 混合检索（稀疏 + 稠密）
  → RRF 融合 → 交叉编码器精排 → 相关性阈值判断 → 回写检索缓存
并在过程中上报各环节的耗时与错误指标。回答生成与护栏归属 agent 编排层，
``common_core.rag`` 提供答案侧公共机制供其复用。
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable
from typing import Any

from common_core.config import RuntimeConfig
from common_core.context import AgentContext
from common_core.instrumentation import trace_node
from common_core.observability import Observability
from common_core.providers import LocalEmbedder, MilvusVectorStore, OpenAICompatibleLLM, RedisCache
from common_core.providers.vector import build_filter_expr

from .results import RetrieveResult, RetrieveStatus
from .stages import (
    QueryRewriteResult,
    QueryRewriter,
    Reranker,
    RetrievalCache,
    judge_relevance,
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
    "source",
    "category",
    "parent_content",
    "parent_title",
    "chunk_index",
    "tenant_id",
    "kb_id",
)


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
        query_rewriter: QueryRewriter | None = None,
        min_relevance: float | None = None,
        tenant_filter: bool = True,
        default_output_fields: Iterable[str] | None = None,
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
            retrieval_cache: 检索结果缓存实例（query → 精排后文档）；None 表示不启用。
            query_rewriter: 查询改写器；None 则用默认 QueryRewriter。
            min_relevance: 精排后的最低相关性阈值；None 时用 runtime 默认值。
            tenant_filter: 是否强制按 tenant_id/kb_id 做隔离过滤，默认 True。
            default_output_fields: 检索默认返回的字段；None 用 runtime/内置默认值。
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

    def _cache_query_key(
        self,
        query: str,
        *,
        mode: str,
        rewrite_query: str | None,
    ) -> str:
        """构造响应缓存用的 key，决定哪些请求共享/隔离同一个缓存。

        设计要点：
        - ``off`` 保持原查询为 key，这样缓存命中时连改写都不用跑；
        - 显式传入的 ``rewrite_query`` 会把 key 绑定到该改写文本上，避免不同
          改写内容之间互相串缓存；
        - 会改变查询文本的改写模式（如 ``llm_rewrite``）也单独开一个桶，
          保证同一个原始问题不会因模式不同而错误地命中同一条缓存。

        参数:
            query: 原始用户查询文本。
            mode: 最终生效的改写模式（off/llm_rewrite/query_expansion 等）。
            rewrite_query: 显式传入的改写后查询；None 表示没有。

        返回:
            用于区分缓存分桶的字符串 key。
        """
        key = str(query).strip()
        explicit = None if rewrite_query is None else str(rewrite_query).strip()
        if explicit:
            key = f"{key}\x1f{explicit}"
        elif mode != "off":
            key = f"{key}\x1f{mode}"
        return key

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

    def _build_rewrite_cache_key(
        self,
        query: str,
        context: AgentContext | None,
        *,
        query_rewrite_mode: str | None,
        rewrite_query: str | None,
    ) -> str:
        """构造检索缓存 key：基于原始查询 + 改写模式分桶。

        缓存命中时跳过改写，因此 key 必须在改写运行前就能计算出来，不能依赖
        改写后的文本（除非调用方显式传入 rewrite_query，那种情况直接以其为 key）。
        """
        effective_mode = self.query_rewriter.resolve_mode(
            context,
            requested_mode=query_rewrite_mode,
        )
        return self._cache_query_key(
            query,
            mode=effective_mode,
            rewrite_query=rewrite_query,
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
        """对混合检索结果做交叉编码器精排；未配置重排器则原样透传。"""
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
    ) -> RetrieveResult:
        """执行完整的"检索 → 精排 → 阈值"管道，并做检索缓存（query → 达标文档）。

        流程与企业检索标准一致：
        1. 先查检索缓存（key = 原始查询 + 改写模式分桶，按 tenant/kb 隔离）。
           命中 → 直接返回缓存的精排后文档（status=retrieved_cache）。
        2. 未命中 → 查询改写（off/llm_rewrite/query_expansion）。
        3. 混合检索（稀疏 + 稠密）：两者都空 → status=no_context。
        4. 至少一方有结果 → RRF 融合 → top_k。
        5. 交叉编码器精排，与相关性阈值比较：全部低于阈值 → status=no_context。
        6. 有达标文档 → 回写检索缓存 → status=retrieved。

        参数:
            query: 用户查询文本。
            context: agent 上下文；可为 None（表示无租户隔离）。
            collection_name/top_k/output_fields/filter_expr/query_rewrite_mode/
                rewrite_query: 语义与 ``retrieve()`` 一致。
            min_relevance: 本次精排相关性阈值覆盖；None 用管线默认。

        返回:
            ``RetrieveResult``：docs 始终携带检索/精排后的候选文档，
            是否达到阈值由 status 表达。
        """
        tenant_id, kb_id, request_id, _session_id = self._context_labels(context)
        cache_key = self._build_rewrite_cache_key(
            query,
            context,
            query_rewrite_mode=query_rewrite_mode,
            rewrite_query=rewrite_query,
        )

        # 第一步：先查检索缓存；命中则跳过改写、混合检索与精排。
        if self.retrieval_cache is not None:
            cached = self.retrieval_cache.get(cache_key, context=context)
            if cached is not None:
                return RetrieveResult(
                    RetrieveStatus.RETRIEVED_CACHE,
                    docs=cached,
                    rewritten_query=rewrite_query or query,
                    cache_hit=True,
                    message="retrieved from cache",
                )

        # 第二步：缓存未命中 → 查询改写 + 混合检索 + RRF + top_k。
        rewrite_trace: dict[str, Any] = {}
        docs = await self.retrieve(
            query,
            context,
            collection_name=collection_name,
            top_k=top_k,
            output_fields=output_fields,
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
            )

        # 第四步：交叉编码器精排。
        ranked = await self._rerank_docs(
            docs,
            rewritten_query,
            context,
            tenant_id=tenant_id,
            kb_id=kb_id,
            request_id=request_id,
        )

        # 第五步：相关性阈值判断与达标文档筛选。
        threshold = min_relevance if min_relevance is not None else self.min_relevance
        if self.reranker is not None:
            if judge_relevance(ranked, threshold):
                # 没有任何文档达标 → no_context，但仍返回全部候选供 agent 判断是否转人工。
                return RetrieveResult(
                    RetrieveStatus.NO_CONTEXT,
                    docs=ranked,
                    rewritten_query=rewritten_query,
                    message="no relevant docs above threshold",
                )
            # 只保留达标文档：交给 agent 并写入缓存（缓存价值 = 精排后达标文档）。
            ranked = [
                doc for doc in ranked
                if float(doc.get("ce_score", 0.0)) >= threshold
            ]

        # 第六步：有达标文档 → 回写检索缓存（命中即复用精排结果）。
        if self.retrieval_cache is not None:
            self.retrieval_cache.put(cache_key, ranked, context=context)
        return RetrieveResult(
            RetrieveStatus.RETRIEVED,
            docs=ranked,
            rewritten_query=rewritten_query,
            message="ok",
        )
