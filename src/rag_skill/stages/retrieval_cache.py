"""租户隔离的检索缓存：缓存 query → 精排后达到阈值的文档列表。

检索缓存保存的是检索管道产出的原始文档（JSON 列表），key 基于规范化查询 +
改写模式分桶，按 tenant/kb 隔离。命中时直接复用精排结果，跳过改写、混合检索
与精排；文档更新后需要主动失效（见 ``delete``）。回答缓存归属 agent 编排层，
不属于本 skill。
"""

from __future__ import annotations

import json
import logging
from typing import Any

from common_core.context import AgentContext
from common_core.observability import Observability
from common_core.providers import RedisCache

logger = logging.getLogger(__name__)


class RetrievalCache:
    """检索结果缓存：读写均按租户隔离，值以 JSON 形式存储文档列表。"""

    def __init__(
        self,
        cache: RedisCache | None = None,
        *,
        scope: str = "rag_retrieval",
        default_ttl: int | None = 600,
        metrics: Observability | None = None,
    ) -> None:
        """初始化检索缓存。

        参数:
            cache: 底层 Redis 适配器；为 None 时缓存退化（不读写）。
            scope: Redis key 中的命名空间，默认 "rag_retrieval"。
            default_ttl: 默认缓存过期秒数；None 时用底层配置。
            metrics: 可观测性 / 指标对象；可为 None。
        """
        self.cache = cache
        self.scope = scope
        self.default_ttl = default_ttl
        self.metrics = metrics

    def _labels(self, context: AgentContext | None) -> tuple[str, str]:
        if context is None:
            return "", ""
        return context.tenant_id, context.kb_id

    def get(
        self,
        query: str,
        *,
        context: AgentContext | None = None,
    ) -> list[dict[str, Any]] | None:
        """读取缓存；未命中返回 None。

        参数:
            query: 规范化后的查询文本（作为缓存 key）。
            context: agent 上下文；非空时用于租户隔离。

        返回:
            缓存的文档列表；未命中或不可解析时为 None。
        """
        if self.cache is None:
            return None
        tenant_id, kb_id = self._labels(context)
        raw = self.cache.get(
            self.scope,
            query.strip(),
            tenant_id=tenant_id,
            kb_id=kb_id,
        )
        if raw is None:
            if self.metrics is not None:
                self.metrics.record_cache(
                    "miss",
                    tenant_id=tenant_id,
                    kb_id=kb_id,
                )
            return None
        try:
            docs = json.loads(raw)
        except (TypeError, ValueError) as exc:
            logger.debug("retrieval cache value unparsable, ignoring: %s", exc)
            return None
        if self.metrics is not None:
            self.metrics.record_cache(
                "hit" if docs else "miss",
                tenant_id=tenant_id,
                kb_id=kb_id,
            )
        return docs if isinstance(docs, list) else None

    def put(
        self,
        query: str,
        docs: list[dict[str, Any]],
        *,
        context: AgentContext | None = None,
        ttl: int | None = None,
    ) -> bool:
        """写入缓存；空文档 / 底层不可用时返回 False。

        参数:
            query: 规范化后的查询文本（作为缓存 key）。
            docs: 精排后达到阈值的文档列表。
            context: agent 上下文；非空时用于租户隔离。
            ttl: 本次缓存的过期秒数；None 时用 default_ttl。

        返回:
            True 表示写入成功；False 表示被跳过或写入失败。
        """
        if self.cache is None:
            return False
        if not docs:
            return False
        tenant_id, kb_id = self._labels(context)
        ok = self.cache.set(
            self.scope,
            query.strip(),
            json.dumps(docs, ensure_ascii=False, default=str),
            ttl=ttl if ttl is not None else self.default_ttl,
            tenant_id=tenant_id,
            kb_id=kb_id,
        )
        if self.metrics is not None:
            self.metrics.record_cache(
                "write" if ok else "error",
                tenant_id=tenant_id,
                kb_id=kb_id,
            )
        return ok

    def delete(
        self,
        query: str,
        *,
        context: AgentContext | None = None,
    ) -> bool:
        """删除指定查询的缓存（文档更新后主动失效）。"""
        if self.cache is None:
            return False
        tenant_id, kb_id = self._labels(context)
        return self.cache.delete(
            self.scope,
            query.strip(),
            tenant_id=tenant_id,
            kb_id=kb_id,
        )


__all__ = ["RetrievalCache"]
