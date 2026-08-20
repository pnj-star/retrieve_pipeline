"""提供者实现与适配器（providers）。
本子包将 abstract 协议（protocols.py）的接口实现为可用的具体适配器：
- cache.py：基于 Redis 的多租户缓存；
- llm.py：OpenAI 兼容对话与本地 Embedding；
- vector.py：基于 Milvus 的混合检索。
加载本子包时不会引入重任务依赖（redis/pymilvus/openai 均在需要时才导入）。
"""

from .cache import RedisCache, response_ttl_for
from .llm import LocalEmbedder, OpenAICompatibleLLM
from .vector import MilvusVectorStore, build_filter_expr, rrf_fuse

__all__ = [
    "LocalEmbedder",
    "MilvusVectorStore",
    "OpenAICompatibleLLM",
    "RedisCache",
    "build_filter_expr",
    "response_ttl_for",
    "rrf_fuse",
]
