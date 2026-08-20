"""从环境变量（或显式传入参数）组装 runtime、provider、cache、reranker 等实例。

本模块是整个 rag_skill 的装配层：把 common_core 提供的通用能力
（LLM、向量库、Embedding、Redis 缓存）与 rag_skill 自己的阶段
（重排器、响应缓存）组合成一个可直接运行的 RagPipeline。
"""

from __future__ import annotations

import os
from typing import Any

from common_core.config import RuntimeConfig, load_env_files
from common_core.observability import Observability
from common_core.providers import LocalEmbedder, MilvusVectorStore, OpenAICompatibleLLM, RedisCache

from .stages import Reranker, RetrievalCache
from .tokenization import build_token_counter


# 模型应尽量从本地 HuggingFace 缓存加载，避免首次推理去访问 HF_ENDPOINT
# （如 hf-mirror.com）做联网检查而卡住几十秒。默认关联网检查，仅用本地缓存；
# 若确实需要联网下载，可显式设 HF_HUB_OFFLINE=0 / TRANSFORMERS_OFFLINE=0。
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def build_runtime(
    *,
    env: dict[str, str] | None = None,
    dotenv_paths: tuple[str, ...] | list[str] | None = None,
    override: bool = False,
) -> RuntimeConfig:
    """从环境变量构建统一的运行时配置。

    参数:
        env: 可选字典，显式提供/覆盖环境变量后传给 RuntimeConfig。
        dotenv_paths: 可选的 .env 文件路径列表，存在则先加载。
        override: 加载 .env 时是否覆盖已存在的环境变量，默认 False。

    返回:
        统一的 RuntimeConfig 配置对象。
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
    """构建 OpenAI 兼容的 LLM 客户端，未传 runtime 时自动从环境构建。

    参数:
        runtime: 运行时配置；None 时用 build_runtime() 从环境构建。
        metrics: 可观测性 / 指标对象，挂载到 LLM 上；可为 None。

    返回:
        OpenAICompatibleLLM 实例。
    """
    runtime = runtime or build_runtime()
    return OpenAICompatibleLLM(runtime.llm, metrics=metrics)


def build_vector(runtime: RuntimeConfig | None = None) -> MilvusVectorStore:
    """构建 Milvus 向量库客户端，支持混合检索（稀疏 + 稠密）。

    参数:
        runtime: 运行时配置；None 时从环境构建。

    返回:
        配置好并发度的 MilvusVectorStore 实例。
    """
    runtime = runtime or build_runtime()
    return MilvusVectorStore(
        runtime.vector,
        max_workers=runtime.retrieval.hybrid_max_workers,
    )


def build_embedder(runtime: RuntimeConfig | None = None) -> LocalEmbedder:
    """构建本地 Embedding 模型，用于把查询转成稠密向量。

    参数:
        runtime: 运行时配置；None 时从环境构建。

    返回:
        LocalEmbedder 实例。
    """
    runtime = runtime or build_runtime()
    return LocalEmbedder(config=runtime.llm)


def build_cache(
    runtime: RuntimeConfig | None = None,
    *,
    shared: bool = True,
) -> RedisCache:
    """构建底层 Redis 缓存（响应缓存共用）。

    参数:
        runtime: 运行时配置；None 时从环境构建。
        shared: 是否使用共享连接，默认 True。

    返回:
        RedisCache 实例。
    """
    runtime = runtime or build_runtime()
    return RedisCache(runtime.cache, shared=shared)


def build_reranker(
    runtime: RuntimeConfig | None = None,
    *,
    metrics: Observability | None = None,
    **overrides: Any,
) -> Reranker:
    """构建交叉编码器重排器。

    模型名、融合权重与 top_k 均取自 ``runtime.retrieval``（其默认值由
    common_core 的 RetrievalConfig 提供，来源 RERANKER_MODEL、
    RERANKER_CE_WEIGHT、RERANKER_RETRIEVAL_WEIGHT、RERANK_TOP_K）。
    设备仍取 ``RERANKER_DEVICE``；可通过 overrides 覆盖 top_k、score_fn 等。

    参数:
        runtime: 运行时配置；None 时从环境构建（用于取 rerank_top_k）。
        metrics: 可观测性 / 指标对象；可为 None。
        overrides: 透传给 Reranker 构造器的额外参数（可覆盖 top_k、score_fn 等）。

    返回:
        配置好的 Reranker 实例。
    """
    runtime = runtime or build_runtime()
    return Reranker(
        model_name=runtime.retrieval.rerank_model,
        device=os.getenv("RERANKER_DEVICE"),
        top_k=runtime.retrieval.rerank_top_k,
        ce_weight=runtime.retrieval.rerank_ce_weight,
        retrieval_weight=runtime.retrieval.rerank_retrieval_weight,
        metrics=metrics,
        **overrides,
    )


def build_retrieval_cache(
    runtime: RuntimeConfig | None = None,
    *,
    cache: RedisCache | None = None,
    metrics: Observability | None = None,
    **overrides: Any,
) -> RetrievalCache:
    """构建租户隔离的检索缓存（query → 精排后达标文档），复用底层 Redis 连接。

    检索缓存（``rag_retrieval``）只负责检索侧：命中时直接复用精排后文档，
    跳过改写、混合检索与精排。

    参数:
        runtime: 运行时配置；None 时从环境构建。
        cache: 底层 Redis 缓存实例；None 时用 build_cache(runtime) 构建。
        metrics: 可观测性 / 指标对象；可为 None。
        overrides: 透传给 RetrievalCache 构造器的额外参数。

    返回:
        配置好的 RetrievalCache 实例。
    """
    runtime = runtime or build_runtime()
    return RetrievalCache(
        cache or build_cache(runtime),
        metrics=metrics,
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

    - include_defaults=True 时自动注入检索缓存、重排器与 token 计数；
      测试场景可传 False 跳过这些阶段（置为 None）。
    - overrides 中的键会覆盖默认构造的组件（如传入自定义 reranker、count_tokens 等 RagPipeline 参数）；
    - count_tokens 推荐使用 ``build_token_counter()``：tiktoken 可用时按真实 token 计数，
      不可用时回退为字符数。

    参数:
        runtime: 运行时配置；None 时从环境构建。
        metrics: 可观测性 / 指标对象；可为 None。
        include_defaults: 是否自动注入检索缓存、重排器与 token 计数；
            测试场景可传 False 跳过这些阶段（对应组件置为 None）。
        overrides: 透传给 RagPipeline 构造器的参数；与默认注入的组件冲突时以显式传入为准。

    返回:
        组装好的 RagPipeline 实例。
    """
    from .pipeline import RagPipeline

    kwargs = dict(overrides)
    if include_defaults:
        default_cache = build_cache(runtime)
        kwargs.setdefault("cache", default_cache)
        kwargs.setdefault(
            "retrieval_cache",
            build_retrieval_cache(runtime, cache=default_cache, metrics=metrics),
        )
        kwargs.setdefault(
            "reranker",
            build_reranker(runtime, metrics=metrics),
        )
        kwargs.setdefault("count_tokens", build_token_counter())
    return RagPipeline(
        runtime=runtime,
        metrics=metrics,
        **kwargs,
    )
