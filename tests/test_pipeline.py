"""管线契约测试：全部使用 fake provider，不依赖外部的 Milvus / Redis / LLM 服务。"""

from __future__ import annotations

import asyncio
import json

from common_core.config import RuntimeConfig, VectorStoreConfig
from common_core.context import AgentContext
from retrieve_skill.builder import build_pipeline, build_retrieval_cache, build_runtime
from retrieve_skill.parent_store import MySQLParentStore, ParentStoreConfig
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


class FakeParentStore:
    def __init__(self, rows: dict[str, dict] | None = None) -> None:
        self.rows = dict(rows or {})
        self.calls: list[list[str]] = []

    async def aget_parent_blocks(self, parent_ids: list[str], *, context) -> dict[str, dict]:
        assert context.tenant_id == "t1"
        assert context.kb_id == "kb1"
        unique_ids = list(dict.fromkeys(parent_ids))
        self.calls.append(unique_ids)
        return {parent_id: self.rows[parent_id] for parent_id in unique_ids if parent_id in self.rows}


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
    parent_store: FakeParentStore | None = None,
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
        parent_store=parent_store or FakeParentStore(),
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


def make_child(
    child_id: str,
    *,
    parent_id: str = "parent-1",
    content: str = "cached doc",
    score: float = 0.9,
    ce_score: float = 0.95,
) -> dict:
    return {
        "id": child_id,
        "content": content,
        "parent_id": parent_id,
        "chunk_index": 0,
        "tenant_id": "t1",
        "kb_id": "kb1",
        "doc_version": 1,
        "score": score,
        "ce_score": ce_score,
    }


def make_parent_store(parent_id: str = "parent-1", content: str = "cached doc"):
    return FakeParentStore(
        {
            parent_id: {
                "parent_id": parent_id,
                "tenant_id": "t1",
                "kb_id": "kb1",
                "title": "Policy title",
                "content": content,
                "doc_version": 1,
            }
        }
    )


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
    assert pipeline.parent_store is not None


def test_build_pipeline_defaults_include_stages() -> None:
    pipeline = build_pipeline(runtime=build_runtime(env={}))
    assert isinstance(pipeline.retrieval_cache, RetrievalCache)
    assert pipeline.retrieval_cache.scope == "rag_retrieval_cache_v3"
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
    child = make_child("chunk-1")
    vector = FakeVector([dict(child)])
    retrieval_cache = RetrievalCache(FakeCache())
    parent_store = make_parent_store(content="authoritative parent content")
    reranker = FakeReranker([dict(child)])
    pipeline = make_pipeline(
        vector,
        retrieval_cache=retrieval_cache,
        reranker=reranker,
        parent_store=parent_store,
    )
    ctx = make_context()
    first = asyncio.run(pipeline.retrieve_context("what is policy", ctx))
    assert first.status == RetrieveStatus.RETRIEVED
    assert first.docs[0]["id"] == "parent-1"
    assert first.docs[0]["child_ids"] == ["chunk-1"]
    assert first.docs[0]["content"] == "authoritative parent content"

    vector.calls.clear()
    reranker.calls.clear()

    result = asyncio.run(pipeline.retrieve_context("what is policy", ctx))

    assert result.status == RetrieveStatus.RETRIEVED_CACHE
    assert result.cache_hit is True
    assert result.docs[0]["id"] == "parent-1"
    assert result.docs[0]["child_ids"] == ["chunk-1"]
    # 缓存只保留父块引用，命中后仍以 MySQL 当前正文重建结果。
    assert result.docs[0]["content"] == "authoritative parent content"
    assert vector.calls == []
    assert reranker.calls == []


def test_retrieve_context_miss_rewrites_retrieves_reranks_and_caches_parent_refs() -> None:
    rewriter_llm = FakeLLM("rewritten query")
    rewriter = QueryRewriter(rewriter_llm, QueryRewriteConfig(mode="llm_rewrite"))
    vector = FakeVector(
        [
            make_child("1", parent_id="p1", score=0.8),
            make_child("2", parent_id="p2", content="weak", score=0.5, ce_score=0.2),
        ]
    )
    reranker = FakeReranker(
        [
            make_child("1", parent_id="p1", score=0.9),
            make_child("2", parent_id="p2", content="weak", score=0.5, ce_score=0.2),
        ]
    )
    parent_store = FakeParentStore(
        {
            "p1": {
                "parent_id": "p1",
                "tenant_id": "t1",
                "kb_id": "kb1",
                "content": "high parent",
                "doc_version": 1,
            }
        }
    )
    cache = FakeCache()
    retrieval_cache = RetrievalCache(cache)
    pipeline = make_pipeline(
        vector,
        query_rewriter=rewriter,
        retrieval_cache=retrieval_cache,
        reranker=reranker,
        min_relevance=0.5,
        parent_store=parent_store,
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
    # 只返回达标子块对应的父块；低分子块不参与父块引用。
    assert [doc["id"] for doc in result.docs] == ["p1"]
    # 检索缓存回写的是父块引用，不含父块正文。
    assert len(cache.store) == 1
    cached = json.loads(next(iter(cache.store.values())))
    assert [doc["parent_id"] for doc in cached] == ["p1"]
    assert all("content" not in doc for doc in cached)


def test_retrieve_cache_material_resolves_defaults_and_separates_recall_params() -> None:
    pipeline = make_pipeline(FakeVector())
    ctx = make_context()
    kwargs = {
        "context": ctx,
        "collection_name": None,
        "top_k": None,
        "filter_expr": None,
        "min_relevance": 0.7,
    }

    implicit_default = pipeline._cache_material(query=" policy ", **kwargs)
    explicit_default = pipeline._cache_material(
        query="policy",
        collection_name=pipeline.runtime.vector.text_collection,
        top_k=pipeline.runtime.retrieval.top_k,
        filter_expr="",
        min_relevance=0.7,
        context=ctx,
    )

    payload = json.loads(implicit_default)
    assert payload["schema"] == "rag_retrieval_cache_v3"
    assert payload["query_rewriter"] is None
    assert payload["parent_store"]["schema"] == "rag_parent_ref_cache_v1"
    assert payload["query"] == "policy"
    assert payload["top_k"] == pipeline.runtime.retrieval.top_k
    assert implicit_default == explicit_default
    changed = pipeline._cache_material(
        query="policy",
        top_k=payload["top_k"] + 1,
        filter_expr='category == "x"',
        min_relevance=0.8,
        context=ctx,
        collection_name=None,
    )
    assert implicit_default != changed


def test_cache_material_includes_mysql_identity_and_active_rewriter_only() -> None:
    pipeline = make_pipeline(FakeVector())
    ctx = make_context()
    configured_parent_store = pipeline.parent_store
    pipeline.parent_store = MySQLParentStore(
        ParentStoreConfig(
            database="rag_test",
            table="rag_parent_block",
            status="active",
        )
    )
    try:
        off_material = pipeline._cache_material(
            query="query",
            context=ctx,
            collection_name=None,
            top_k=None,
            filter_expr=None,
            min_relevance=0.7,
            query_rewrite_mode="off",
        )
        rewrite_material = pipeline._cache_material(
            query="query",
            context=ctx,
            collection_name=None,
            top_k=None,
            filter_expr=None,
            min_relevance=0.7,
            query_rewrite_mode="llm_rewrite",
        )
    finally:
        pipeline.parent_store = configured_parent_store

    off_payload = json.loads(off_material)
    rewrite_payload = json.loads(rewrite_material)
    assert off_payload["parent_store"] == {
        "schema": "rag_parent_ref_cache_v1",
        "type": "MySQLParentStore",
        "database": "rag_test",
        "table": "rag_parent_block",
        "status": "active",
    }
    assert off_payload["query_rewriter"] is None
    assert isinstance(rewrite_payload["query_rewriter"], dict)
    assert off_material != rewrite_material


def test_cached_children_support_different_budgets_and_projections() -> None:
    children = [
        make_child(
            f"chunk-{index}",
            content=f"part {index}",
            score=0.9,
            ce_score=0.9,
        )
        for index in (1, 2)
    ]
    for index, child in enumerate(children):
        child["chunk_index"] = index
    vector = FakeVector(children)
    reranker = FakeReranker([dict(child) for child in children])
    parent_store = make_parent_store(content="a" * 80)
    pipeline = make_pipeline(
        vector,
        retrieval_cache=RetrievalCache(FakeCache()),
        reranker=reranker,
        parent_store=parent_store,
    )
    ctx = make_context()

    wide = asyncio.run(
        pipeline.retrieve_context(
            "query",
            ctx,
            context_max_chars=1000,
            max_doc_chars=1000,
            output_fields=("id", "content", "child_ids"),
        )
    )
    assert wide.status == RetrieveStatus.RETRIEVED

    vector.calls.clear()
    reranker.calls.clear()
    narrow = asyncio.run(
        pipeline.retrieve_context(
            "query",
            ctx,
            context_max_chars=40,
            max_doc_chars=25,
            output_fields=("id", "content"),
        )
    )

    assert narrow.status == RetrieveStatus.RETRIEVED_CACHE
    assert narrow.cache_hit is True
    assert vector.calls == []
    assert reranker.calls == []
    assert len(wide.docs[0]["content"]) > len(narrow.docs[0]["content"])
    assert "child_ids" not in narrow.docs[0]


def test_corrupted_parent_ref_cache_is_treated_as_miss() -> None:
    child = make_child("chunk-1")
    vector = FakeVector([dict(child)])
    cache = FakeCache()
    retrieval_cache = RetrievalCache(cache)
    reranker = FakeReranker([dict(child)])
    parent_store = make_parent_store()
    pipeline = make_pipeline(
        vector,
        retrieval_cache=retrieval_cache,
        reranker=reranker,
        parent_store=parent_store,
    )
    ctx = make_context()
    material = pipeline._cache_material(
        query="query",
        context=ctx,
        collection_name=None,
        top_k=None,
        filter_expr=None,
        min_relevance=pipeline.min_relevance,
    )
    assert retrieval_cache.put(material, [dict(child)], context=ctx)

    key = retrieval_cache.cache.key(
        retrieval_cache.scope,
        material,
        tenant_id=ctx.tenant_id,
        kb_id=ctx.kb_id,
    )
    cache.store[key] = "{not-json"
    result = asyncio.run(pipeline.retrieve_context("query", ctx))

    assert result.status == RetrieveStatus.RETRIEVED
    assert result.cache_hit is False
    assert len(vector.calls) == 1
    assert len(reranker.calls) == 1
    assert parent_store.calls == [["parent-1"]]


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
    assert result.docs == []
    assert result.diagnostics["reason"] == "below_threshold"
    assert result.diagnostics["candidate_count"] == 1
    assert cache.store == {}


def test_retrieve_context_without_reranker_fails_closed() -> None:
    vector = FakeVector([{"id": "1", "content": "doc", "score": 0.9}])
    cache = FakeCache()
    retrieval_cache = RetrievalCache(cache)
    pipeline = make_pipeline(vector, retrieval_cache=retrieval_cache)

    result = asyncio.run(pipeline.retrieve_context("query", make_context()))

    assert result.status == RetrieveStatus.ERROR
    assert result.docs == []
    assert result.diagnostics["reason"] == "reranker_not_configured"
    assert cache.store == {}


def test_retrieve_context_reranker_failure_fails_closed_and_skips_cache() -> None:
    class FailingReranker(FakeReranker):
        async def arank(self, *args, **kwargs):
            raise RuntimeError("cross encoder down")

    vector = FakeVector([{"id": "1", "content": "doc", "score": 0.9}])
    cache = FakeCache()
    pipeline = make_pipeline(
        vector,
        retrieval_cache=RetrievalCache(cache),
        reranker=FailingReranker(),
    )

    result = asyncio.run(pipeline.retrieve_context("query", make_context()))

    assert result.status == RetrieveStatus.ERROR
    assert result.docs == []
    assert result.diagnostics["reason"] == "reranker_unavailable"
    assert cache.store == {}


def test_retrieve_context_parent_store_failure_fails_closed() -> None:
    class FailingParentStore:
        async def aget_parent_blocks(self, parent_ids, *, context):
            raise RuntimeError("mysql down")

    child = make_child("chunk-1")
    pipeline = make_pipeline(
        FakeVector([dict(child)]),
        reranker=FakeReranker([dict(child)]),
        parent_store=FailingParentStore(),
    )

    result = asyncio.run(pipeline.retrieve_context("query", make_context()))

    assert result.status == RetrieveStatus.ERROR
    assert result.docs == []
    assert result.diagnostics["reason"] == "parent_store_unavailable"


def test_retrieve_context_version_mismatch_returns_no_context_and_skips_cache() -> None:
    child = make_child("chunk-1", ce_score=0.9)
    stale_parent = make_parent_store()
    stale_parent.rows["parent-1"]["doc_version"] = 2
    pipeline = make_pipeline(
        FakeVector([dict(child)]),
        reranker=FakeReranker([dict(child)]),
        parent_store=stale_parent,
        retrieval_cache=RetrievalCache(FakeCache()),
    )
    cache = pipeline.cache

    result = asyncio.run(pipeline.retrieve_context("query", make_context()))

    assert result.status == RetrieveStatus.NO_CONTEXT
    assert result.docs == []
    assert result.diagnostics["reason"] == "parent_version_mismatch"
    assert cache.store == {}


def test_cache_hit_removes_stale_parent_refs_but_keeps_current_ones() -> None:
    children = [
        make_child("chunk-1", parent_id="p1"),
        make_child("chunk-2", parent_id="p2"),
    ]
    vector = FakeVector(children)
    reranker = FakeReranker([dict(child) for child in children])
    parent_store = FakeParentStore(
        {
            "p1": {
                "parent_id": "p1",
                "tenant_id": "t1",
                "kb_id": "kb1",
                "content": "current p1",
                "doc_version": 1,
            },
            "p2": {
                "parent_id": "p2",
                "tenant_id": "t1",
                "kb_id": "kb1",
                "content": "soon deleted p2",
                "doc_version": 1,
            },
        }
    )
    cache = FakeCache()
    retrieval_cache = RetrievalCache(cache)
    pipeline = make_pipeline(
        vector,
        reranker=reranker,
        parent_store=parent_store,
        retrieval_cache=retrieval_cache,
    )
    ctx = make_context()

    first = asyncio.run(pipeline.retrieve_context("query", ctx))
    # 预置缓存后，把 p2 从 MySQL 中移除，模拟父块在 TTL 内被删除。
    first_refs = json.loads(next(iter(cache.store.values())))
    assert [ref["parent_id"] for ref in first_refs] == ["p1", "p2"]
    parent_store.rows.pop("p2")

    second = asyncio.run(pipeline.retrieve_context("query", ctx))
    refreshed_refs = json.loads(next(iter(cache.store.values())))

    assert first.status == RetrieveStatus.RETRIEVED
    assert second.status == RetrieveStatus.RETRIEVED_CACHE
    assert [doc["id"] for doc in second.docs] == ["p1"]
    assert [ref["parent_id"] for ref in refreshed_refs] == ["p1"]
