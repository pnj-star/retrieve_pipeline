"""基于 common_core provider 的可执行检索与回答管线。

RagPipeline 串联整个 RAG 执行流程：
  响应缓存检查 → 查询改写（默认 off）→ 混合检索（稀疏 + 稠密）
  → RRF 融合 → 交叉编码器精排
  → 相关性阈值判断 → 拼上下文 → 生成 → 质量护栏 → 回写缓存
并在过程中上报各环节的耗时与错误指标。
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable
from typing import Any, Callable, Sequence

from common_core.config import RuntimeConfig
from common_core.context import AgentContext
from common_core.instrumentation import trace_node
from common_core.observability import Observability
from common_core.providers import LocalEmbedder, MilvusVectorStore, OpenAICompatibleLLM, RedisCache
from common_core.providers.vector import build_filter_expr
from common_core.telemetry import trace_id

from .results import RagResult, RagStatus
from .stages import (
    GenerationConfig,
    GuardConfig,
    QueryRewriteResult,
    QueryRewriter,
    Reranker,
    ResponseCache,
    build_context_text,
    clean_markdown,
    generate_answer,
    guard_generation,
    judge_relevance,
)
from .stages.assembly import DEFAULT_MAX_CONTEXT_CHARS
from .stages.generation import DEFAULT_PROMPT_TEMPLATE
from .tokenization import build_token_counter

logger = logging.getLogger(__name__)

# 混合检索默认返回的文本字段。不同集合的 schema 可通过
# runtime.vector.text_output_fields 或调用方显式传入 output_fields 覆盖，避免查询报错。
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


def _format_doc(index: int, doc: dict[str, Any]) -> str:
    """把单个文档格式化为给 LLM 看的一行上下文片段。

    优先使用父块内容（parent_content），其次使用当前块内容（content）；
    再用 source / category / score 补充来源与相关性信息。

    参数:
        index: 文档在上下文中的序号，输出时作为 [index] 前缀。
        doc: 单篇检索返回的文档字典，支持 parent_content、content、
            source、category、score 等字段。

    返回:
        格式化后的单行上下文文本。
    """
    content = str(doc.get("parent_content", "") or doc.get("content", "") or "")
    return _render_fields(
        index,
        {
            "content": clean_markdown(content) if content.strip() else None,
            "source": doc.get("source"),
            "category": doc.get("category"),
            "score": doc.get("score"),
        },
    )


def _render_fields(index: int, fields: dict[str, Any]) -> str:
    """把有序字段渲染成 `[index] | 字段: 值` 的单行上下文片段。

    所有调用方共用的渲染骨架：字段值为空或空白时跳过，避免输出 ``field: `` 空壳。
    """
    parts = [f"[{index}]"]
    for field_name, value in fields.items():
        if value is not None and str(value).strip():
            parts.append(f"{field_name}: {value}")
    return " | ".join(parts)


def format_context(
    docs: Iterable[dict[str, Any]],
    *,
    include_fields: tuple[str, ...] = ("content", "source", "score"),
    max_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
    prefix_blocks: Sequence[str] | None = None,
    source_label: str = "source",
    max_doc_chars: int | None = None,
    max_doc_tokens: int | None = None,
    max_tokens: int | None = None,
    count_tokens: Callable[[str], int] | None = None,
) -> str:
    """用自定义字段集把文档拼接成上下文文本（测试 / 调试辅助函数）。

    预算参数与 ``build_context_text`` 对齐：max_chars / max_tokens 控制总体上限，
    max_doc_chars / max_doc_tokens 控制单篇（或同一父块合并后）上限；
    传了 max_tokens 且未传 count_tokens 时，自动使用 build_token_counter()。

    参数:
        docs: 待拼接的文档字典可迭代对象。
        include_fields: 每个文档要输出的字段名列表，按顺序渲染成 "字段: 值"。
        max_chars: 整个上下文的最大字符数上限。
        prefix_blocks: 放在上下文最前面的固定文本块（如系统指引），可为 None。
        source_label: 主标识字段名，在 build_context_text 内用于控制文档分组。
        max_doc_chars: 单篇（或合并后同源块）的最大字符数；None 表示不限制。
        max_doc_tokens: 单篇（或合并后同源块）的最大 token 数；None 表示不限制。
        max_tokens: 整个上下文的最大 token 数；需配合 count_tokens 使用。
        count_tokens: 把字符串换算成 token 数的函数；未传且用了 max_tokens 时自动构建。

    返回:
        拼接好的上下文文本字符串。
    """
    def _format(index: int, doc: dict[str, Any]) -> str:
        """把单个文档按 include_fields 渲染成一行，前面带 [序号]。

        参数:
            index: 当前文档序号。
            doc: 单个文档字典。

        返回:
            格式化后的单行文本。
        """
        return _render_fields(index, {name: doc.get(name) for name in include_fields})

    if max_tokens is not None and count_tokens is None:
        count_tokens = build_token_counter()

    return build_context_text(
        docs,
        max_chars=max_chars,
        prefix_blocks=prefix_blocks,
        source_label=source_label,
        format_doc=_format,
        max_doc_chars=max_doc_chars,
        max_doc_tokens=max_doc_tokens,
        max_tokens=max_tokens,
        count_tokens=count_tokens,
    )[0]


class RagPipeline:
    """RAG 检索 / 回答管线。

    所有组件依赖都可以传入（便于测试注入 fake），未传入的参数会从
    runtime 配置构建默认实现：
      - llm: OpenAI 兼容 LLM
      - vector: Milvus 向量库（混合检索）
      - embedder: 本地 Embedding 模型
      - cache: Redis 底层缓存（服务于响应缓存）
      - reranker / response_cache / guard_config: 可选的重排、缓存与护栏阶段
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
        response_cache: ResponseCache | None = None,
        guard_config: GuardConfig | None = None,
        query_rewriter: QueryRewriter | None = None,
        min_relevance: float | None = None,
        tenant_filter: bool = True,
        count_tokens: Callable[[str], int] | None = None,
        default_output_fields: Iterable[str] | None = None,
    ) -> None:
        """组装整条 RAG 管线：未显式注入的组件从 runtime 配置默认构建。

        参数:
            runtime: common_core 的运行配置；未传则从环境变量构建。
            llm: OpenAI 兼容 LLM 实例，用于生成回答；未传用 runtime.llm 默认构建。
            vector: Milvus 向量库实例，用于混合检索；未传则默认构建。
            embedder: 本地 Embedding 模型，用于生成查询向量；未传则默认构建。
            cache: Redis 底层缓存实例（供响应缓存复用连接）；未传则默认构建。
            metrics: 可观测性 / 指标上报对象；为 None 则不上报指标。
            reranker: 交叉编码器精排器；None 表示不启用精排。
            response_cache: 响应缓存实例；None 表示不启用响应缓存。
            guard_config: 质量护栏配置；None 表示默认无护栏配置。
            query_rewriter: 查询改写器；None 则用默认 QueryRewriter。
            min_relevance: 精排后的最低相关性阈值；None 时用 runtime 默认值。
            tenant_filter: 是否强制按 tenant_id/kb_id 做隔离过滤，默认 True。
            count_tokens: 字符串转 token 数函数，用于 token 预算控制；None 用默认。
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
        self.response_cache = response_cache
        # 响应缓存若未显式传入指标对象，则复用管线的全局指标，保证缓存指标统一上报。
        if (
            self.response_cache is not None
            and getattr(self.response_cache, "metrics", None) is None
            and metrics is not None
        ):
            self.response_cache.metrics = metrics
        self.guard_config = guard_config
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
        self.count_tokens = count_tokens
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
        该方法被 ``retrieve()`` 和 ``_answer_impl()`` 共用，保证整条管线
        在同一处统一处理改写，避免两处逻辑分叉。

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
        - ``off`` / ``identity`` 保持原查询为 key，这样缓存命中时连改写都不用跑；
        - 显式传入的 ``rewrite_query`` 会把 key 绑定到该改写文本上，避免不同
          改写内容之间互相串缓存；
        - 会改变查询文本的改写模式（如 ``llm_rewrite``）也单独开一个桶，
          保证同一个原始问题不会因模式不同而错误地命中同一条缓存。

        参数:
            query: 原始用户查询文本。
            mode: 最终生效的改写模式（off/identity/llm_rewrite/query_expansion 等）。
            rewrite_query: 显式传入的改写后查询；None 表示没有。

        返回:
            用于区分缓存分桶的字符串 key。
        """
        key = str(query).strip()
        explicit = None if rewrite_query is None else str(rewrite_query).strip()
        if explicit:
            key = f"{key}\x1f{explicit}"
        elif mode not in ("off", "identity"):
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

        查询改写支持 ``off`` / ``identity`` / ``llm_rewrite`` / ``query_expansion``；
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

    async def answer(
        self,
        query: str,
        context: AgentContext | None = None,
        *,
        collection_name: str | None = None,
        system_prompt: str = "",
        top_k: int | None = None,
        output_fields: Iterable[str] | None = None,
        filter_expr: str | None = None,
        query_rewrite_mode: str | None = None,
        rewrite_query: str | None = None,
        empty_answer: str = "No relevant context found.",
        temperature: float | None = None,
        max_tokens: int | None = None,
        min_relevance: float | None = None,
        guard_config: GuardConfig | None = None,
        enable_guard: bool = True,
        prompt_template: str | None = None,
        context_max_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
        context_max_tokens: int | None = None,
        count_tokens: Callable[[str], int] | None = None,
        max_doc_chars: int | None = None,
        max_doc_tokens: int | None = None,
    ) -> RagResult:
        """完整执行一次的 RAG 回答；对异常做兜底，绝不向上抛错。

        任何内部异常都会被转换为 ``status=error`` 的 RagResult，并记录日志与
        指标，保证调用方（如 MCP 层）始终拿到稳定的返回契约。

        参数:
            query: 用户的问题文本。
            context: agent 上下文；可为 None。
            system_prompt: 附加在生成 prompt 里的系统指令，追加在模板之后。
            top_k: 检索候选条数；None 用默认值。
            output_fields: 检索返回字段覆盖；None 用管线默认。
            filter_expr: 额外业务过滤表达式。
            query_rewrite_mode: 改写模式；None 按配置决定。
            rewrite_query: 显式改写后的查询；非空时跳过内置改写。
            collection_name: 覆盖默认的 Milvus 集合名；None 用 runtime 配置。
            empty_answer: 无可用上下文时返回的占位回答文案。
            temperature: LLM 采样温度；None 用模型/配置默认。
            max_tokens: 生成最大 token 数；None 用模型/配置默认。
            min_relevance: 本次最低相关性阈值覆盖；None 用管线默认。
            guard_config: 本次护栏配置覆盖；None 用管线默认。
            enable_guard: 是否启用护栏阶段，默认 True。
            prompt_template: 覆盖默认生成 prompt 模板。
            context_max_chars: 上下文最大字符数。
            context_max_tokens: 上下文最大 token 数（需 count_tokens 支持）。
            count_tokens: 字符串转 token 数函数。
            max_doc_chars: 单篇文档最大字符数。
            max_doc_tokens: 单篇文档最大 token 数。

        返回:
            包含状态、回答、候选文档与改写信息的 RagResult。
        """
        tenant_id, kb_id, request_id, session_id = self._context_labels(context)
        if self.metrics is not None:
            self.metrics.begin_run()
        start = time.perf_counter()
        try:
            result = await self._answer_impl(
                query,
                context,
                collection_name=collection_name,
                system_prompt=system_prompt,
                top_k=top_k,
                output_fields=output_fields,
                filter_expr=filter_expr,
                query_rewrite_mode=query_rewrite_mode,
                rewrite_query=rewrite_query,
                empty_answer=empty_answer,
                temperature=temperature,
                max_tokens=max_tokens,
                min_relevance=min_relevance,
                guard_config=guard_config,
                enable_guard=enable_guard,
                prompt_template=prompt_template,
                context_max_chars=context_max_chars,
                context_max_tokens=context_max_tokens,
                count_tokens=count_tokens,
                max_doc_chars=max_doc_chars,
                max_doc_tokens=max_doc_tokens,
            )
        except Exception as exc:
            self._observe_error("answer", context)
            logger.exception(
                "rag.answer.error tenant_id=%s kb_id=%s request_id=%s session_id=%s",
                tenant_id,
                kb_id,
                request_id,
                session_id,
            )
            result = RagResult(RagStatus.ERROR, str(exc), [], "")
            result.rewritten_query = rewrite_query or query
        self._observe("answer", start, context)
        if self.metrics is not None:
            self.metrics.end_run(
                route=result.status,
                tenant_id=tenant_id,
                kb_id=kb_id,
            )
        logger.info(
            "rag.answer.end status=%s tenant_id=%s kb_id=%s request_id=%s session_id=%s",
            result.status,
            tenant_id,
            kb_id,
            request_id,
            session_id,
        )
        return result

    async def _answer_impl(
        self,
        query: str,
        context: AgentContext | None,
        *,
        collection_name: str | None,
        system_prompt: str,
        top_k: int | None,
        output_fields: Iterable[str] | None,
        filter_expr: str | None,
        query_rewrite_mode: str | None,
        rewrite_query: str | None,
        empty_answer: str,
        temperature: float | None,
        max_tokens: int | None,
        min_relevance: float | None,
        guard_config: GuardConfig | None,
        enable_guard: bool,
        prompt_template: str | None,
        context_max_chars: int,
        context_max_tokens: int | None,
        count_tokens: Callable[[str], int] | None,
        max_doc_chars: int | None,
        max_doc_tokens: int | None,
    ) -> RagResult:
        """RAG 核心流水线（不含 answer 的异常兜底包装）。

        参数语义与 ``answer()`` 完全一致，全部为关键字参数，唯一的区别是：
        这里不做异常兜底，内部异常会直接向上抛出，由外层 ``answer()``
        统一转换成 ``status=error`` 的稳定返回。
        """
        tenant_id, kb_id, request_id, session_id = self._context_labels(context)
        logger.info(
            "rag.answer.start tenant_id=%s kb_id=%s request_id=%s session_id=%s trace_id=%s",
            tenant_id,
            kb_id,
            request_id,
            session_id,
            trace_id(),
        )

        # 第一步：先构造缓存 key（不预跑改写），命中即返回，避免缓存命中时也调用改写模型。
        effective_mode = self.query_rewriter.resolve_mode(
            context,
            requested_mode=query_rewrite_mode,
        )
        cache_key = self._cache_query_key(
            query,
            mode=effective_mode,
            rewrite_query=rewrite_query,
        )

        # 第二步：检查响应缓存（按租户隔离）。命中则不再走任何 provider。
        if self.response_cache is not None:
            cached = self.response_cache.get(cache_key, context=context)
            if cached is not None:
                return RagResult(
                    RagStatus.ANSWERED_CACHE,
                    "answered from cache",
                    [],
                    cached,
                    rewritten_query=rewrite_query or query,
                )

        # 第三步：缓存未命中，才执行一次改写。
        rewrite_result = await self._resolve_rewrite(
            query,
            context,
            query_rewrite_mode=query_rewrite_mode,
            rewrite_query=rewrite_query,
        )
        rewritten_query = rewrite_result.rewritten_query or query

        # 第四步：混合检索（复用已解析的改写结果，避免二次改写）
        docs = await self.retrieve(
            query,
            context,
            collection_name=collection_name,
            top_k=top_k,
            output_fields=output_fields,
            filter_expr=filter_expr,
            query_rewrite_mode=query_rewrite_mode,
            rewrite_query=rewrite_query,
            _resolved_rewrite=rewrite_result,
        )
        if not docs:
            # 检索为空：没有可用上下文
            return RagResult(
                RagStatus.NO_CONTEXT,
                empty_answer,
                [],
                empty_answer,
                rewritten_query=rewritten_query,
            )

        # 第五步：交叉编码器精排（若配置了重排器）
        ranked = docs
        if self.reranker is not None:
            start_rerank = time.perf_counter()
            try:
                with trace_node(
                    "rerank",
                    tenant_id=tenant_id,
                    kb_id=kb_id,
                    request_id=request_id,
                ):
                    ranked = await self.reranker.arank(
                        rewritten_query,
                        docs,
                        tenant_id=tenant_id,
                        kb_id=kb_id,
                    )
            except Exception:
                self._observe_error("rerank", context)
                raise
            finally:
                self._observe("rerank", start_rerank, context)
            ranked = list(ranked or [])

        # 第六步：相关性阈值判断。低于阈值视为没有可用上下文，
        # 但候选文档（ranked）会随结果返回，便于外层 agent 决定是否转人工。
        threshold = min_relevance if min_relevance is not None else self.min_relevance
        if self.reranker is not None and judge_relevance(ranked, threshold):
            return RagResult(
                RagStatus.NO_CONTEXT,
                empty_answer,
                ranked,
                empty_answer,
                rewritten_query=rewritten_query,
            )

        # 第七步：把精排后的文档拼成给 LLM 的上下文文本（去重、清 markdown、截断）
        max_chars = (
            DEFAULT_MAX_CONTEXT_CHARS
            if context_max_chars is None
            else context_max_chars
        )
        token_counter = count_tokens if count_tokens is not None else self.count_tokens
        if context_max_tokens is not None and token_counter is None:
            logger.warning(
                "rag.answer.token_budget_without_counter tenant_id=%s kb_id=%s "
                "request_id=%s; falling back to char budget",
                tenant_id,
                kb_id,
                request_id,
            )
            context_max_tokens = None
        context_text, _sources = build_context_text(
            ranked,
            max_chars=max_chars,
            max_tokens=context_max_tokens,
            count_tokens=token_counter,
            max_doc_chars=max_doc_chars,
            max_doc_tokens=max_doc_tokens,
            format_doc=_format_doc,
        )
        # 第八步：选择生成 / 护栏节点
        cfg = guard_config if guard_config is not None else self.guard_config
        node_name = "guard" if (enable_guard and cfg is not None) else "generate"

        async def _generate(q: str, ctx_text: str, guard_reason: str) -> str:
            """真正调用 LLM 生成回答；护栏重试时会回传上次未通过的原因。

            参数:
                q: 要回答的用户查询。
                ctx_text: 已拼好的上下文文本。
                guard_reason: 上一次护栏未通过的原因；为空表示首次生成。

            返回:
                LLM 生成的回答字符串。
            """
            instructions = system_prompt.strip()
            if guard_reason:
                instructions = (
                    f"{instructions}\n\n{guard_reason}".strip()
                    if instructions
                    else guard_reason
                )
            return await generate_answer(
                self.llm,
                q,
                context_text=ctx_text,
                config=GenerationConfig(
                    prompt_template=prompt_template or DEFAULT_PROMPT_TEMPLATE,
                    temperature=temperature,
                    max_tokens=max_tokens,
                ),
                extra_prompt=instructions,
                # 生成失败视为硬错误，交由上层 answer 兜底为 status=error
                raise_on_error=True,
            )

        start_generate = time.perf_counter()
        try:
            if node_name == "guard":
                # 走护栏：生成 + LLM 评审 + 按需重试
                response, guard_result, _attempts = await guard_generation(
                    self.llm,
                    generate=_generate,
                    query=query,
                    context_text=context_text,
                    config=cfg,
                )
                if self.metrics is not None:
                    self.metrics.record_guard(
                        "pass" if guard_result.passed else "fail_exhausted",
                        tenant_id=tenant_id,
                        kb_id=kb_id,
                    )
                if not guard_result.passed:
                    # 护栏不通过：拦截结果，同时返回候选文档与（未过审的）回答
                    return RagResult(
                        RagStatus.GUARD_BLOCKED,
                        guard_result.reason or "guard_blocked",
                        ranked,
                        response,
                        rewritten_query=rewritten_query,
                    )
            else:
                # 未启用护栏：直接生成
                response = await _generate(query, context_text, "")
        except Exception:
            self._observe_error(node_name, context)
            raise
        finally:
            self._observe(node_name, start_generate, context)

        # 第七步：把结果写回响应缓存（仅成功回答会被缓存）
        if self.response_cache is not None:
            self.response_cache.put(cache_key, response, context=context)
        return RagResult(
            RagStatus.ANSWERED,
            "ok",
            ranked,
            response,
            rewritten_query=rewritten_query,
        )
