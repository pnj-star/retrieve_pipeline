"""管线契约测试：全部使用 fake provider，不依赖外部的 Milvus / Redis / LLM 服务。"""

from __future__ import annotations

import asyncio

from common_core.config import RuntimeConfig, VectorStoreConfig
from common_core.context import AgentContext
from retrieve_skill.builder import build_pipeline, build_retrieval_cache, build_runtime
from retrieve_skill.pipeline import DEFAULT_TEXT_OUTPUT_FIELDS, RagPipeline
from retrieve_skill.results import RetrieveStatus
from retrieve_skill.stages import (
    QueryRewriteConfig,
    QueryRewriter,
    Reranker,
    RetrievalCache,
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
    retrieval_cache: RetrievalCache | None = None,
    reranker: FakeReranker | None = None,
    min_relevance: float | None = None,
    tenant_filter: bool = True,
    metrics: FakeMetrics | None = None,
    default_output_fields: tuple[str, ...] | None = None,
) -> RagPipeline:
    cache = cache or FakeCache()
    return RagPipeline(
        make_runtime(),
        llm=llm or FakeLLM(),
        vector=vector,
        embedder=FakeEmbedder(),
        query_rewriter=query_rewriter,
        cache=cache,
        retrieval_cache=retrieval_cache,
        reranker=reranker,
        min_relevance=min_relevance,
        tenant_filter=tenant_filter,
        metrics=metrics,
        default_output_fields=default_output_fields,
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


def test_build_pipeline_defaults_include_stages() -> None:
    pipeline = build_pipeline(runtime=build_runtime(env={}))
    assert isinstance(pipeline.retrieval_cache, RetrievalCache)
    assert isinstance(pipeline.reranker, Reranker)
    assert pipeline.min_relevance == 0.70


def test_build_pipeline_can_skip_default_stages() -> None:
    pipeline = build_pipeline(runtime=build_runtime(env={}), include_defaults=False)
    assert pipeline.reranker is None


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


def test_build_pipeline_defaults_include_retrieval_cache() -> None:
    pipeline = build_pipeline(runtime=build_runtime(env={}))
    assert isinstance(pipeline.retrieval_cache, RetrievalCache)


def test_build_retrieval_cache_uses_runtime_ttl() -> None:
    cache = build_retrieval_cache(runtime=build_runtime(env={"REDIS_DEFAULT_TTL": "120"}))
    assert cache.default_ttl == 120
    assert (
        build_retrieval_cache(
            runtime=build_runtime(env={}),
            default_ttl=30,
        ).default_ttl
        == 30
    )


def test_retrieve_context_cache_hit_skips_providers() -> None:
    vector = FakeVector([{"id": "1", "content": "doc", "ce_score": 0.95}])
    retrieval_cache = RetrievalCache(FakeCache())
    pipeline = make_pipeline(
        vector,
        retrieval_cache=retrieval_cache,
        reranker=FakeReranker([{"id": "1", "content": "doc", "ce_score": 0.95}]),
    )
    ctx = make_context()
    retrieval_cache.put("what is policy", [{"id": "1", "content": "cached doc"}], context=ctx)

    result = asyncio.run(pipeline.retrieve_context("what is policy", ctx))

    assert result.status == RetrieveStatus.RETRIEVED_CACHE
    assert result.cache_hit is True
    assert result.docs == [{"id": "1", "content": "cached doc"}]
    assert vector.calls == []


def test_retrieve_context_miss_rewrites_retrieves_reranks_and_caches() -> None:
    rewriter_llm = FakeLLM("rewritten query")
    rewriter = QueryRewriter(rewriter_llm, QueryRewriteConfig(mode="llm_rewrite"))
    vector = FakeVector(
        [
            {"id": "1", "content": "high", "score": 0.8},
            {"id": "2", "content": "low", "score": 0.5},
        ]
    )
    reranker = FakeReranker(
        [
            {"id": "1", "content": "high", "score": 0.9, "ce_score": 0.95},
            {"id": "2", "content": "low", "score": 0.5, "ce_score": 0.2},
        ]
    )
    cache = FakeCache()
    retrieval_cache = RetrievalCache(cache)
    pipeline = make_pipeline(
        vector,
        query_rewriter=rewriter,
        retrieval_cache=retrieval_cache,
        reranker=reranker,
        min_relevance=0.5,
    )
    ctx = make_context()

    result = asyncio.run(
        pipeline.retrieve_context("original", ctx, query_rewrite_mode="llm_rewrite")
    )

    assert result.status == RetrieveStatus.RETRIEVED
    assert result.rewritten_query == "rewritten query"
    # 改写在缓存未命中后、混合检索前发生一次
    assert len(rewriter_llm.chat_calls) == 1
    assert vector.calls[-1]["query"] == "rewritten query"
    # 只返回精排后达标文档（低于阈值的 doc2 被过滤掉）
    assert [doc["id"] for doc in result.docs] == ["1"]
    # 检索缓存回写的是精排后达标文档
    cached = retrieval_cache.get("original\u001fllm_rewrite", context=ctx)
    assert cached is not None
    assert [doc["id"] for doc in cached] == ["1"]


def test_retrieve_context_empty_hybrid_returns_no_context() -> None:
    pipeline = make_pipeline(
        FakeVector(),
        retrieval_cache=RetrievalCache(FakeCache()),
        reranker=FakeReranker(),
    )

    result = asyncio.run(pipeline.retrieve_context("nothing", make_context()))

    assert result.status == RetrieveStatus.NO_CONTEXT
    assert result.docs == []


def test_retrieve_context_all_below_threshold_returns_no_context_and_skips_cache() -> None:
    vector = FakeVector([{"id": "1", "content": "weak", "score": 0.3}])
    reranker = FakeReranker(
        [{"id": "1", "content": "weak", "score": 0.3, "ce_score": 0.1}]
    )
    cache = FakeCache()
    retrieval_cache = RetrievalCache(cache)
    pipeline = make_pipeline(
        vector,
        retrieval_cache=retrieval_cache,
        reranker=reranker,
        min_relevance=0.5,
    )
    ctx = make_context()

    result = asyncio.run(pipeline.retrieve_context("query", ctx))

    assert result.status == RetrieveStatus.NO_CONTEXT
    # no_context 时仍返回候选文档供 agent 判断是否转人工
    assert result.docs[0]["id"] == "1"
    assert cache.store == {}


def test_retrieve_context_no_reranker_passes_docs_through() -> None:
    vector = FakeVector([{"id": "1", "content": "doc", "score": 0.9}])
    cache = FakeCache()
    retrieval_cache = RetrievalCache(cache)
    pipeline = make_pipeline(vector, retrieval_cache=retrieval_cache)

    result = asyncio.run(pipeline.retrieve_context("query", make_context()))

    assert result.status == RetrieveStatus.RETRIEVED
    assert [doc["id"] for doc in result.docs] == ["1"]
    assert cache.store
