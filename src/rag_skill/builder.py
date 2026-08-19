"""从环境变量（或显式传入参数）组装 runtime、provider、cache、reranker、guard 等实例。

本模块是整个 rag_skill 的装配层：把 common_core 提供的通用能力
（LLM、向量库、Embedding、Redis 缓存）与 rag_skill 自己的阶段
（重排器、响应缓存、人工交接存储）组合成一个可直接运行的 RagPipeline。
"""

from __future__ import annotations

import os
from typing import Any

from common_core.config import RuntimeConfig, load_env_files
from common_core.observability import Observability
from common_core.providers import LocalEmbedder, MilvusVectorStore, OpenAICompatibleLLM, RedisCache

from .stages import GuardConfig, RedisHandoffStore, Reranker, ResponseCache
from .stages.rerank import DEFAULT_RERANK_MODEL
from .tokenization import build_token_counter


def build_runtime(
    *,
    env: dict[str, str] | None = None,
    dotenv_paths: tuple[str, ...] | list[str] | None = None,
    override: bool = False,
) -> RuntimeConfig:
    """从环境变量构建统一的运行时配置。

    - env: 可选，用字典显式提供/覆盖环境变量；
    - dotenv_paths: 可选的 .env 文件路径列表；
    - override: 加载 .env 时是否覆盖已存在的环境变量。
    """
    if dotenv_paths:
        load_env_files(*dotenv_paths, override=override)
    source = dict(env) if env is not None else None
    return RuntimeConfig.from_env(env=source)


def build_llm(
    runtime: RuntimeConfig | None = None,
    *,
    metrics: Observability | None = None,
) -> OpenAICompatibleLLM:
    """构建 OpenAI 兼容的 LLM 客户端，未传 runtime 时自动从环境构建。"""
    runtime = runtime or build_runtime()
    return OpenAICompatibleLLM(runtime.llm, metrics=metrics)


def build_vector(runtime: RuntimeConfig | None = None) -> MilvusVectorStore:
    """构建 Milvus 向量库客户端，支持混合检索（稀疏 + 稠密）。"""
    runtime = runtime or build_runtime()
    return MilvusVectorStore(
        runtime.vector,
        max_workers=runtime.retrieval.hybrid_max_workers,
    )


def build_embedder(runtime: RuntimeConfig | None = None) -> LocalEmbedder:
    """构建本地 Embedding 模型，用于把查询转成稠密向量。"""
    runtime = runtime or build_runtime()
    return LocalEmbedder(config=runtime.llm)


def build_cache(
    runtime: RuntimeConfig | None = None,
    *,
    shared: bool = True,
) -> RedisCache:
    """构建底层 Redis 缓存（响应缓存与人工交接存储共用）。"""
    runtime = runtime or build_runtime()
    return RedisCache(runtime.cache, shared=shared)


def build_reranker(
    runtime: RuntimeConfig | None = None,
    *,
    metrics: Observability | None = None,
    **overrides: Any,
) -> Reranker:
    """构建交叉编码器重排器。

    模型名取自环境变量 ``RERANKER_MODEL``（默认 bge-reranker-base），
    设备取 ``RERANKER_DEVICE``；可通过 overrides 覆盖 top_k、score_fn 等。
    """
    runtime = runtime or build_runtime()
    return Reranker(
        model_name=os.getenv("RERANKER_MODEL", DEFAULT_RERANK_MODEL),
        device=os.getenv("RERANKER_DEVICE"),
        top_k=runtime.retrieval.rerank_top_k,
        metrics=metrics,
        **overrides,
    )


def build_response_cache(
    runtime: RuntimeConfig | None = None,
    *,
    cache: RedisCache | None = None,
    metrics: Observability | None = None,
    **overrides: Any,
) -> ResponseCache:
    """构建租户隔离的响应缓存，复用底层 Redis 缓存连接。"""
    runtime = runtime or build_runtime()
    return ResponseCache(
        cache or build_cache(runtime),
        metrics=metrics,
        **overrides,
    )


def build_handoff_store(
    runtime: RuntimeConfig | None = None,
    *,
    cache: RedisCache | None = None,
    **overrides: Any,
) -> RedisHandoffStore:
    """构建 Redis 人工交接存储（low-relevance 等场景下记录待人工处理）。"""
    runtime = runtime or build_runtime()
    return RedisHandoffStore(
        cache or build_cache(runtime),
        **overrides,
    )


def build_pipeline(
    runtime: RuntimeConfig | None = None,
    *,
    metrics: Observability | None = None,
    include_defaults: bool = True,
    **overrides: Any,
) -> Any:
    """组装完整的 RagPipeline。

    - include_defaults=True 时自动注入响应缓存、重排器与默认护栏配置；
      测试场景可传 False 跳过这些阶段（置为 None）。
    - overrides 中的键会覆盖默认构造的组件（如传入自定义 reranker、count_tokens 等 RagPipeline 参数）；
    - count_tokens 推荐使用 ``build_token_counter()``：tiktoken 可用时按真实 token 计数，
      不可用时回退为字符数。
    """
    from .pipeline import RagPipeline

    kwargs = dict(overrides)
    if include_defaults:
        default_response_cache = build_response_cache(runtime, metrics=metrics)
        kwargs.setdefault(
            "response_cache",
            default_response_cache,
        )
        # 底层 Redis 缓存与响应缓存共用同一连接
        kwargs.setdefault("cache", default_response_cache.cache)
        kwargs.setdefault(
            "reranker",
            build_reranker(runtime, metrics=metrics),
        )
        kwargs.setdefault("guard_config", GuardConfig())
        kwargs.setdefault("count_tokens", build_token_counter())
    return RagPipeline(
        runtime=runtime,
        metrics=metrics,
        **kwargs,
    )
