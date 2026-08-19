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
    """把多个查询的检索结果按文档身份合并，保留首次出现顺序与最高分数。"""
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
    """
    content = str(doc.get("parent_content", "") or doc.get("content", "") or "")
    parts = [f"[{index}]"]
    if content.strip():
        parts.append(f"content: {clean_markdown(content)}")
    for field_name in ("source", "category", "score"):
        value = doc.get(field_name)
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
    """
    def _format(index: int, doc: dict[str, Any]) -> str:
        parts = [f"[{index}]"]
        for field_name in include_fields:
            value = doc.get(field_name)
            if value is not None and str(value).strip():
                parts.append(f"{field_name}: {value}")
        return " | ".join(parts)

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
        """取出用于指标 / 日志打点的租户上下文标签。"""
        if context is None:
            return "", "", "", ""
        return (
            context.tenant_id,
            context.kb_id,
            context.request_id,
            context.session_id,
        )

    def _observe(self, node: str, start: float, context: AgentContext | None) -> None:
        """记录某个环节（node）的耗时指标。"""
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
        """记录某个环节（node）的错误指标。"""
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
        """Resolve the effective query (explicit rewrite or configured strategy) once,
        reporting rewrite timing. Reused by both retrieve() and _answer_impl()."""
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
        """Build the response-cache key. Default/identity keeps the original query,
        so cache hits skip re-rewriting; explicit rewrite queries and rewrite-changing
        modes get their own bucket and never collide on the same original query."""
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
        """Run hybrid search from an already-resolved rewrite result."""
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
        """
        tenant_id, kb_id, request_id, session_id = self._context_labels(context)
        if self.metrics is not None:
            self.metrics.begin_run()
        start = time.perf_counter()
        try:
            result = await self._answer_impl(
                query,
                context,
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
        """RAG 核心流水线（不含 answer 的异常兜底包装）。"""
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
            """真正调用 LLM 生成回答；护栏重试时会回传上次未通过的原因。"""
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
