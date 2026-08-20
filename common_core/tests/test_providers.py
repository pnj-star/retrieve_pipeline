import common_core.providers.cache as cache_module

from common_core.config import CacheConfig, LLMConfig
from common_core.providers import (
    OpenAICompatibleLLM,
    RedisCache,
    build_filter_expr,
    response_ttl_for,
    rrf_fuse,
)


def test_rrf_fuse_ranks_docs_hit_by_both_routes_first() -> None:
    dense = [
        {"id": "a", "content": "a", "score": 0.9},
        {"id": "b", "content": "b", "score": 0.8},
    ]
    sparse = [
        {"id": "b", "content": "b", "bm25_score": 2.0},
        {"id": "c", "content": "c", "bm25_score": 1.0},
    ]
    fused = rrf_fuse(dense, sparse)
    assert [doc["id"] for doc in fused] == ["b", "a", "c"]
    assert "fusion_score" in fused[0]
    assert "dense_rank" in fused[0]
    assert "sparse_rank" in fused[0]


def test_build_filter_expr_handles_scalar_and_list_filters() -> None:
    expr = build_filter_expr(
        {
            "category": "knowledge",
            "product_id": ["p1", "p2"],
            "tag": ["x", "y"],
        }
    )
    assert 'category == "knowledge"' in expr
    assert '"p1", "p2"' in expr
    assert expr.startswith("category == ")


def test_redis_cache_key_is_tenant_and_kb_scoped() -> None:
    cache = RedisCache(CacheConfig(key_prefix="shop"))
    key_a = cache.key("resp", "hello world", tenant_id="t1", kb_id="kb1")
    key_b = cache.key("resp", "hello world", tenant_id="t1", kb_id="kb2")
    key_c = cache.key("resp", "hello world")
    assert key_a.startswith("shop:resp:t1:kb1:")
    assert key_a != key_b
    assert key_a != key_c


def test_response_ttl_shortens_truncated_replies() -> None:
    assert response_ttl_for("ok") == 300
    assert response_ttl_for("a reasonably complete answer") == 1800


def test_llm_client_holds_config_without_importing_openai() -> None:
    llm = OpenAICompatibleLLM(
        LLMConfig(base_url="http://llm.local/v1", model="generic-model")
    )
    assert llm.config.base_url == "http://llm.local/v1"
    assert llm.config.model == "generic-model"


def test_redis_cache_shared_mode_reuses_client(monkeypatch) -> None:
    created: list[object] = []

    def fake_factory(config):
        created.append(config)
        return object()

    monkeypatch.setattr(cache_module, "_shared_clients", {})
    monkeypatch.setattr(cache_module, "_create_redis_client", fake_factory)
    first = RedisCache(CacheConfig(host="cache", port=6379), shared=True)
    second = RedisCache(CacheConfig(host="cache", port=6379), shared=True)
    assert first.client() is second.client()
    assert len(created) == 1

    standalone = RedisCache(CacheConfig(host="cache", port=6379))
    assert standalone.client() is not first.client()
    assert len(created) == 2
