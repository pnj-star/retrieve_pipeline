"""管线契约测试：全部使用 fake provider，不依赖外部的 Milvus / Redis / LLM 服务。"""

from __future__ import annotations

import asyncio
from typing import Callable

from common_core.config import RuntimeConfig, VectorStoreConfig
from common_core.context import AgentContext
from rag_skill.builder import build_pipeline, build_runtime
from rag_skill.pipeline import DEFAULT_TEXT_OUTPUT_FIELDS, RagPipeline, format_context
from rag_skill.results import RagStatus
from rag_skill.stages import (
    GuardConfig,
    QueryRewriteConfig,
    QueryRewriter,
    Reranker,
    ResponseCache,
)


class FakeEmbedder:
    def __init__(self) -> None:
        self.queries: list[str] = []

    async def embed(self, text: str) -> list[float]:
        self.queries.append(text)
        return [0.1] * 3


class FakeVector:
    def __init__(self, docs: list[dict] | None = None) -> None:
        self.docs = list(docs or [])
        self.calls: list[dict] = []

    async def a_search_hybrid(
        self,
        collection_name: str,
        query: str,
        embedding: list[float],
        **kwargs,
    ) -> list[dict]:
        self.calls.append(
            {
                "collection_name": collection_name,
                "query": query,
                "embedding": embedding,
                **kwargs,
            }
        )
        return list(self.docs)


class FakeLLM:
    def __init__(
        self,
        response: str = "good answer",
        review: dict | None = None,
        *,
        fail_chat: bool = False,
    ) -> None:
        self.response = response
        self.review = review if review is not None else {"pass": True}
        self.fail_chat = fail_chat
        self.chat_calls: list[tuple[list[dict], dict]] = []
        self.json_calls: list[tuple[list[dict], str]] = []

    async def chat(self, messages: list[dict], *, system_prompt: str = "", **kwargs) -> str:
        if self.fail_chat:
            raise RuntimeError("llm down")
        self.chat_calls.append((messages, {"system_prompt": system_prompt, **kwargs}))
        return self.response

    async def chat_json(
        self,
        messages: list[dict],
        *,
        system_prompt: str = "",
        **kwargs,
    ) -> dict:
        self.json_calls.append((messages, system_prompt))
        return dict(self.review)


class FakeCache:
    def __init__(self) -> None:
        self.store: dict[str, str] = {}

    def key(
        self,
        scope: str,
        material: str,
        *,
        tenant_id: str = "",
        kb_id: str = "",
    ) -> str:
        return f"{tenant_id}:{kb_id}:{scope}:{material}"

    def get(
        self,
        scope: str,
        material: str,
        *,
        tenant_id: str = "",
        kb_id: str = "",
    ) -> str | None:
        return self.store.get(self.key(scope, material, tenant_id=tenant_id, kb_id=kb_id))

    def set(
        self,
        scope: str,
        material: str,
        value: str,
        *,
        ttl: int | None = None,
        tenant_id: str = "",
        kb_id: str = "",
    ) -> bool:
        self.store[self.key(scope, material, tenant_id=tenant_id, kb_id=kb_id)] = value
        return True

    def delete(
        self,
        scope: str,
        material: str,
        *,
        tenant_id: str = "",
        kb_id: str = "",
    ) -> bool:
        key = self.key(scope, material, tenant_id=tenant_id, kb_id=kb_id)
        return self.store.pop(key, None) is not None


class FakeMetrics:
    def record_cache(
        self,
        result: str,
        tenant_id: str = "",
        kb_id: str = "",
    ) -> None:
        return None


class FakeReranker:
    def __init__(self, docs: list[dict] | None = None) -> None:
        self.docs = list(docs or [])
        self.calls: list[tuple[str, str, str]] = []

    async def arank(
        self,
        query: str,
        docs: list[dict],
        *,
        tenant_id: str = "",
        kb_id: str = "",
    ) -> list[dict]:
        self.calls.append((query, tenant_id, kb_id))
        return list(self.docs)


def make_runtime() -> RuntimeConfig:
    return RuntimeConfig(vector=VectorStoreConfig(text_collection="test_kb"))


def make_pipeline(
    vector: FakeVector,
    *,
    llm: FakeLLM | None = None,
    query_rewriter: QueryRewriter | None = None,
    cache: FakeCache | None = None,
    response_cache: ResponseCache | None = None,
    reranker: FakeReranker | None = None,
    guard_config: GuardConfig | None = None,
    min_relevance: float | None = None,
    tenant_filter: bool = True,
    metrics: FakeMetrics | None = None,
    default_output_fields: tuple[str, ...] | None = None,
    count_tokens: Callable[[str], int] | None = None,
) -> RagPipeline:
    cache = cache or FakeCache()
    return RagPipeline(
        make_runtime(),
        llm=llm or FakeLLM(),
        vector=vector,
        embedder=FakeEmbedder(),
        query_rewriter=query_rewriter,
        cache=cache,
        response_cache=response_cache,
        reranker=reranker,
        guard_config=guard_config,
        min_relevance=min_relevance,
        tenant_filter=tenant_filter,
        metrics=metrics,
        default_output_fields=default_output_fields,
        count_tokens=count_tokens,
    )


def make_context(**overrides: str) -> AgentContext:
    values = {
        "tenant_id": "t1",
        "kb_id": "kb1",
        "session_id": "s1",
        "request_id": "r1",
        "user_id": "u1",
    }
    values.update(overrides)
    return AgentContext(**values)


def test_build_runtime_stays_business_free() -> None:
    runtime = build_runtime(
        env={
            "LLM_MODEL": "generic-model",
            "MILVUS_TEXT_COLLECTION": "generic_kb",
            "REDIS_KEY_PREFIX": "demo",
        }
    )
    assert runtime.llm.model == "generic-model"
    assert runtime.vector.text_collection == "generic_kb"
    assert runtime.cache.key_prefix == "demo"
    assert "mushroom" not in str(runtime)


def test_build_pipeline_constructs_providers() -> None:
    pipeline = build_pipeline(runtime=build_runtime(env={}))
    assert pipeline.llm is not None
    assert pipeline.vector is not None
    assert pipeline.embedder is not None
    assert pipeline.cache is not None


def test_build_pipeline_defaults_include_full_status_stages() -> None:
    pipeline = build_pipeline(runtime=build_runtime(env={}))
    assert isinstance(pipeline.response_cache, ResponseCache)
    assert isinstance(pipeline.reranker, Reranker)
    assert isinstance(pipeline.guard_config, GuardConfig)
    assert pipeline.min_relevance == 0.70
    assert pipeline.count_tokens is not None


def test_build_pipeline_can_skip_default_stages() -> None:
    pipeline = build_pipeline(runtime=build_runtime(env={}), include_defaults=False)
    assert pipeline.response_cache is None
    assert pipeline.reranker is None
    assert pipeline.guard_config is None


def test_format_context_keeps_relevant_fields() -> None:
    text = format_context(
        [
            {"content": "first chunk", "source": "kb.md", "score": 0.9},
            {"content": "second chunk", "source": "kb.md", "score": 0.7},
        ]
    )
    assert "[1]" in text
    assert "first chunk" in text
    assert "score: 0.9" in text


def test_format_context_forwards_budget_params(monkeypatch) -> None:
    captured: dict = {}

    def fake_build_context_text(docs, **kwargs):
        captured.update(kwargs)
        return "context", ["source"]

    monkeypatch.setattr(
        "rag_skill.pipeline.build_context_text",
        fake_build_context_text,
    )

    text = format_context(
        [{"content": "doc"}],
        max_chars=120,
        prefix_blocks=["P" * 20],
        source_label="url",
        max_doc_chars=40,
        max_doc_tokens=30,
        max_tokens=200,
        count_tokens=lambda value: len(value),
    )

    assert text == "context"
    assert captured["max_chars"] == 120
    assert captured["prefix_blocks"] == ["P" * 20]
    assert captured["source_label"] == "url"
    assert captured["max_doc_chars"] == 40
    assert captured["max_doc_tokens"] == 30
    assert captured["max_tokens"] == 200
    assert captured["count_tokens"] is not None


def test_format_context_defaults_token_counter_for_token_budget(monkeypatch) -> None:
    captured: dict = {}

    def fake_build_context_text(docs, **kwargs):
        captured.update(kwargs)
        return "context", ["source"]

    monkeypatch.setattr(
        "rag_skill.pipeline.build_context_text",
        fake_build_context_text,
    )
    monkeypatch.setattr("rag_skill.pipeline.build_token_counter", lambda: len)

    text = format_context(
        [{"content": "doc"}],
        max_tokens=100,
    )

    assert text == "context"
    assert captured["count_tokens"] is len
    assert captured["max_tokens"] == 100


def test_answer_returns_cache_hit_without_calling_providers() -> None:
    llm = FakeLLM()
    vector = FakeVector([{"content": "doc"}])
    cache = FakeCache()
    response_cache = ResponseCache(cache, min_cache_chars=0)
    pipeline = make_pipeline(
        vector,
        llm=llm,
        cache=cache,
        response_cache=response_cache,
    )
    ctx = make_context()
    response_cache.put("what is policy", "cached answer " * 3, context=ctx)

    result = asyncio.run(pipeline.answer("what is policy", ctx))

    assert result.status == RagStatus.ANSWERED_CACHE
    assert result.answer == "cached answer " * 3
    assert vector.calls == []
    assert llm.chat_calls == []


def test_answer_reports_no_context_when_retrieval_empty() -> None:
    pipeline = make_pipeline(
        FakeVector(),
        response_cache=ResponseCache(FakeCache(), min_cache_chars=0),
    )

    result = asyncio.run(
        pipeline.answer("missing", make_context(), empty_answer="没有找到相关内容")
    )

    assert result.status == RagStatus.NO_CONTEXT
    assert result.docs == []
    assert result.answer == "没有找到相关内容"


def test_answer_blocks_below_relevance_threshold() -> None:
    llm = FakeLLM()
    vector = FakeVector([{"id": "1", "content": "weak doc"}])
    reranker = FakeReranker(
        [{"id": "1", "content": "weak doc", "ce_score": 0.1, "score": 0.1}]
    )
    pipeline = make_pipeline(vector, llm=llm, reranker=reranker, min_relevance=0.5)

    result = asyncio.run(pipeline.answer("query", make_context()))

    assert result.status == RagStatus.NO_CONTEXT
    assert llm.chat_calls == []
    assert result.docs[0]["ce_score"] == 0.1


def test_guard_blocked_returned_without_caching() -> None:
    llm = FakeLLM(response="绝对最低价，无副作用")
    cache = FakeCache()
    pipeline = make_pipeline(
        FakeVector([{"id": "1", "content": "doc", "ce_score": 0.9}]),
        llm=llm,
        cache=cache,
        response_cache=ResponseCache(cache, min_cache_chars=0),
        reranker=FakeReranker([{"id": "1", "content": "doc", "ce_score": 0.9}]),
        guard_config=GuardConfig(),
    )

    result = asyncio.run(pipeline.answer("query", make_context()))

    assert result.status == RagStatus.GUARD_BLOCKED
    assert "绝对" in result.answer
    assert cache.store == {}


def test_answer_passes_guard_and_caches() -> None:
    llm = FakeLLM(response="请参照平台规则，以实际结算单为准。")
    cache = FakeCache()
    pipeline = make_pipeline(
        FakeVector([{"id": "1", "content": "doc", "ce_score": 0.9}]),
        llm=llm,
        cache=cache,
        response_cache=ResponseCache(cache, min_cache_chars=10),
        reranker=FakeReranker([{"id": "1", "content": "doc", "ce_score": 0.9}]),
        guard_config=GuardConfig(),
    )

    result = asyncio.run(pipeline.answer("query", make_context()))

    assert result.status == RagStatus.ANSWERED
    assert result.answer == "请参照平台规则，以实际结算单为准。"
    assert len(cache.store) == 1


def test_retrieve_injects_tenant_and_kb_filter() -> None:
    vector = FakeVector([{"id": "1", "content": "doc"}])
    pipeline = make_pipeline(vector)

    asyncio.run(pipeline.retrieve("query", make_context(), filter_expr='category == "x"'))

    assert (
        vector.calls[-1]["filter_expr"]
        == 'tenant_id == "t1" and kb_id == "kb1" and (category == "x")'
    )


def test_retrieve_can_disable_tenant_filter() -> None:
    vector = FakeVector([{"id": "1", "content": "doc"}])
    pipeline = make_pipeline(vector, tenant_filter=False)

    asyncio.run(pipeline.retrieve("query", make_context(), filter_expr='category == "x"'))

    assert vector.calls[-1]["filter_expr"] == '(category == "x")'


def test_retrieve_uses_default_output_fields() -> None:
    vector = FakeVector([{"id": "1", "content": "doc"}])
    pipeline = make_pipeline(vector)

    asyncio.run(pipeline.retrieve("query", make_context()))

    assert vector.calls[-1]["output_fields"] == list(DEFAULT_TEXT_OUTPUT_FIELDS)


def test_retrieve_uses_configured_output_fields() -> None:
    vector = FakeVector([{"id": "1", "content": "doc"}])
    runtime = RuntimeConfig(
        vector=VectorStoreConfig(
            text_collection="test_kb",
            text_output_fields=("id", "content"),
        )
    )
    pipeline = RagPipeline(
        runtime,
        llm=FakeLLM(),
        vector=vector,
        embedder=FakeEmbedder(),
        cache=FakeCache(),
    )

    asyncio.run(pipeline.retrieve("query", make_context()))

    assert vector.calls[-1]["output_fields"] == ["id", "content"]


def test_retrieve_accepts_explicit_default_output_fields() -> None:
    vector = FakeVector([{"id": "1", "content": "doc"}])
    pipeline = make_pipeline(vector, default_output_fields=("id", "content"))

    asyncio.run(pipeline.retrieve("query", make_context()))

    assert vector.calls[-1]["output_fields"] == ["id", "content"]


def test_retrieve_default_off_uses_original_query() -> None:
    vector = FakeVector([{"id": "1", "content": "doc"}])
    pipeline = make_pipeline(vector)

    asyncio.run(pipeline.retrieve("原始问题", make_context(), rewrite_trace={}))

    assert vector.calls[-1]["query"] == "原始问题"


def test_retrieve_explicit_rewrite_query_skips_rewriting() -> None:
    vector = FakeVector([{"id": "1", "content": "doc"}])
    pipeline = make_pipeline(vector)
    trace: dict = {}

    asyncio.run(
        pipeline.retrieve(
            "原始问题",
            make_context(),
            rewrite_query="改写后的问题",
            rewrite_trace=trace,
        )
    )

    assert vector.calls[-1]["query"] == "改写后的问题"
    assert trace["rewritten_query"] == "改写后的问题"
    assert trace["mode"] == "explicit"


def test_retrieve_query_rewrite_mode_overrides_and_writes_trace() -> None:
    llm = FakeLLM("市场行情与政策")
    rewriter = QueryRewriter(llm, QueryRewriteConfig(mode="off"))
    vector = FakeVector([{"id": "1", "content": "doc"}])
    pipeline = make_pipeline(vector, query_rewriter=rewriter)
    trace: dict = {}

    docs = asyncio.run(
        pipeline.retrieve(
            "了解行情",
            make_context(),
            query_rewrite_mode="llm_rewrite",
            rewrite_trace=trace,
        )
    )

    assert docs[0]["id"] == "1"
    assert vector.calls[-1]["query"] == "市场行情与政策"
    assert trace["mode"] == "llm_rewrite"
    assert trace["rewritten_query"] == "市场行情与政策"
    assert trace["original_query"] == "了解行情"


def test_answer_uses_assembly_and_generation_stages() -> None:
    llm = FakeLLM()
    pipeline = make_pipeline(
        FakeVector([{"id": "1", "content": "doc", "score": 0.9}]),
        llm=llm,
    )

    result = asyncio.run(
        pipeline.answer(
            "query",
            make_context(),
            system_prompt="business rules",
        )
    )

    assert result.status == RagStatus.ANSWERED
    messages, kwargs = llm.chat_calls[0]
    assert messages == [{"role": "user", "content": "query"}]
    assert "[1]" in kwargs["system_prompt"]
    assert "content: doc" in kwargs["system_prompt"]
    assert "business rules" in kwargs["system_prompt"]


def test_answer_exposes_rewritten_query() -> None:
    llm = FakeLLM()
    pipeline = make_pipeline(
        FakeVector([{"id": "1", "content": "doc", "score": 0.9}]),
        llm=llm,
    )

    result = asyncio.run(
        pipeline.answer(
            "原始问题",
            make_context(),
            rewrite_query="改写后的问题",
        )
    )

    assert result.status == RagStatus.ANSWERED
    assert result.rewritten_query == "改写后的问题"
    # 生成仍针对原始用户问题，不因改写而偏离
    messages, _kwargs = llm.chat_calls[0]
    assert messages == [{"role": "user", "content": "原始问题"}]


def test_answer_generation_failure_is_error_and_not_cached() -> None:
    llm = FakeLLM(fail_chat=True)
    cache = FakeCache()
    pipeline = make_pipeline(
        FakeVector([{"id": "1", "content": "doc"}]),
        llm=llm,
        cache=cache,
        response_cache=ResponseCache(cache, min_cache_chars=0),
    )

    result = asyncio.run(pipeline.answer("query", make_context()))

    assert result.status == RagStatus.ERROR
    assert cache.store == {}


def test_answer_defaults_context_max_chars_when_none() -> None:
    llm = FakeLLM()
    pipeline = make_pipeline(
        FakeVector([{"id": "1", "content": "doc", "score": 0.9}]),
        llm=llm,
    )

    result = asyncio.run(
        pipeline.answer("query", make_context(), context_max_chars=None)
    )

    assert result.status == RagStatus.ANSWERED


def test_answer_forwards_token_and_doc_budget(monkeypatch) -> None:
    captured: dict = {}

    def fake_build_context_text(docs, **kwargs):
        captured.update(kwargs)
        return "context", ["source"]

    monkeypatch.setattr(
        "rag_skill.pipeline.build_context_text",
        fake_build_context_text,
    )
    pipeline = make_pipeline(FakeVector([{"content": "doc"}]))

    asyncio.run(
        pipeline.answer(
            "query",
            make_context(),
            context_max_tokens=123,
            max_doc_chars=45,
            max_doc_tokens=67,
            count_tokens=lambda text: len(text),
        )
    )

    assert captured["max_tokens"] == 123
    assert captured["count_tokens"] is not None
    assert captured["max_doc_chars"] == 45
    assert captured["max_doc_tokens"] == 67


def test_answer_falls_back_to_chars_without_token_counter(monkeypatch) -> None:
    captured: dict = {}

    def fake_build_context_text(docs, **kwargs):
        captured.update(kwargs)
        return "context", ["source"]

    monkeypatch.setattr(
        "rag_skill.pipeline.build_context_text",
        fake_build_context_text,
    )
    warnings: list[str] = []
    monkeypatch.setattr(
        "rag_skill.pipeline.logger.warning",
        lambda *args: warnings.append(str(args)),
    )
    pipeline = make_pipeline(FakeVector([{"content": "doc"}]))

    asyncio.run(
        pipeline.answer("query", make_context(), context_max_tokens=123)
    )

    assert captured["max_tokens"] is None
    assert any("token_budget_without_counter" in warning for warning in warnings)


def test_pipeline_injects_metrics_into_response_cache() -> None:
    metrics = FakeMetrics()
    cache = FakeCache()
    response_cache = ResponseCache(cache, min_cache_chars=0)
    pipeline = make_pipeline(
        FakeVector([{"id": "1"}]),
        cache=cache,
        response_cache=response_cache,
        metrics=metrics,
    )

    assert pipeline.response_cache.metrics is metrics


def test_answer_caches_under_effective_query_and_avoids_double_rewrite() -> None:
    """缓存 key 按原始查询 + 改写策略分桶；缓存命中时不再触发改写。"""
    rewriter_llm = FakeLLM("rewritten query")
    rewriter = QueryRewriter(rewriter_llm, QueryRewriteConfig(mode="llm_rewrite"))
    gen_llm = FakeLLM(response="answer text")
    cache = FakeCache()
    vector = FakeVector([{"id": "1", "content": "doc", "score": 0.9}])
    pipeline = make_pipeline(
        vector,
        llm=gen_llm,
        query_rewriter=rewriter,
        cache=cache,
        response_cache=ResponseCache(cache, min_cache_chars=0),
    )
    ctx = make_context()

    result = asyncio.run(
        pipeline.answer("original question", ctx, query_rewrite_mode="llm_rewrite")
    )

    assert result.status == RagStatus.ANSWERED
    assert result.rewritten_query == "rewritten query"
    # 改写只解析一次，检索阶段不再重复调用改写 LLM
    assert len(rewriter_llm.chat_calls) == 1
    # 生成仍针对原始用户问题
    assert gen_llm.chat_calls[0][0] == [
        {"role": "user", "content": "original question"}
    ]
    (cached_key,) = cache.store.keys()
    assert "original question" in cached_key
    assert "llm_rewrite" in cached_key

    # 第二次请求命中基于改写查询的缓存，不再调用检索/生成
    vector.calls.clear()
    gen_llm.chat_calls.clear()
    hit = asyncio.run(
        pipeline.answer("original question", ctx, query_rewrite_mode="llm_rewrite")
    )
    assert hit.status == RagStatus.ANSWERED_CACHE
    assert vector.calls == []
    assert gen_llm.chat_calls == []


def test_answer_different_explicit_rewrites_do_not_share_cache_key() -> None:
    """同一原始查询的不同显式改写应使用不同的缓存 key，避免互相命中。"""
    gen_llm = FakeLLM(response="answer text")
    cache = FakeCache()
    vector = FakeVector([{"id": "1", "content": "doc", "score": 0.9}])
    pipeline = make_pipeline(
        vector,
        llm=gen_llm,
        cache=cache,
        response_cache=ResponseCache(cache, min_cache_chars=0),
    )
    ctx = make_context()

    asyncio.run(
        pipeline.answer(
            "original",
            ctx,
            rewrite_query="rewrite A",
        )
    )
    asyncio.run(
        pipeline.answer(
            "original",
            ctx,
            rewrite_query="rewrite B",
        )
    )

    assert len(cache.store) == 2
    keys = sorted(cache.store.keys())
    assert "rewrite A" in keys[0]
    assert "rewrite B" in keys[1]
