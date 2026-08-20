"""基于 common_core Redis 适配器的租户隔离响应缓存。

对同一 (tenant_id, kb_id, 规范化 query) 缓存回答：命中时直接返回缓存，
跳过检索与生成；写入时按回答长度决定是否缓存，避免缓存过短的兜底文案。
"""

from __future__ import annotations

from typing import Any

from common_core.context import AgentContext
from common_core.observability import Observability
from common_core.providers import RedisCache, response_ttl_for


class ResponseCache:
    """RAG 回答缓存：读写均按租户隔离，并上报缓存指标。"""

    def __init__(
        self,
        cache: RedisCache | None = None,
        *,
        scope: str = "rag_response",
        default_ttl: int | None = None,
        min_cache_chars: int = 20,
        metrics: Observability | None = None,
    ) -> None:
        """初始化响应缓存：底层存储、key 命名空间、TTL 与指标对象均可注入。

        参数:
            cache: 底层 Redis 适配器；为 None 时缓存退化（不读写）。
            scope: Redis key 中的命名空间，用于区分不同用途，默认 "rag_response"。
            default_ttl: 默认缓存过期秒数；None 时按回答长度自动推断。
            min_cache_chars: 回答达到该长度才允许写入缓存，避免缓存过短兜底文案。
            metrics: 可观测性 / 指标对象；可为 None。
        """
        self.cache = cache  # 底层 Redis 适配器；为 None 时缓存退化（不读写）
        self.scope = scope  # Redis key 中的命名空间，用于区分不同用途
        self.default_ttl = default_ttl
        self.min_cache_chars = min_cache_chars  # 回答达到该长度才允许写入缓存
        self.metrics = metrics

    def _labels(self, context: AgentContext | None) -> tuple[str, str]:
        """取出指标打点用的租户标签。

        参数:
            context: agent 上下文；可为 None（无租户信息）。

        返回:
            (tenant_id, kb_id) 二元组，缺省时为空字符串。
        """
        if context is None:
            return "", ""
        return context.tenant_id, context.kb_id

    def get(self, query: str, *, context: AgentContext | None = None) -> str | None:
        """读取缓存；未命中返回 None。

        参数:
            query: 规范化后的查询文本（作为缓存 key）。
            context: agent 上下文；非空时用于租户隔离。

        返回:
            缓存的回答文本；未命中时为 None。
        """
        if self.cache is None:
            return None
        tenant_id, kb_id = self._labels(context)
        value = self.cache.get(
            self.scope,
            query.strip(),
            tenant_id=tenant_id,
            kb_id=kb_id,
        )
        if self.metrics is not None:
            self.metrics.record_cache(
                "hit" if value is not None else "miss",
                tenant_id=tenant_id,
                kb_id=kb_id,
            )
        return value

    def put(
        self,
        query: str,
        response: str,
        *,
        context: AgentContext | None = None,
        ttl: int | None = None,
    ) -> bool:
        """写入缓存；回答过短或底层不可用时返回 False。

        参数:
            query: 规范化后的查询文本（作为缓存 key）。
            response: 要缓存的回答文本。
            context: agent 上下文；非空时用于租户隔离。
            ttl: 本次缓存的过期秒数；None 时用 default_ttl 或自动推断。

        返回:
            True 表示写入成功；False 表示被跳过或写入失败。
        """
        if self.cache is None:
            return False
        if len(response.strip()) < self.min_cache_chars:
            # 过短的回答（多为兜底文案）不缓存
            if self.metrics is not None:
                tenant_id, kb_id = self._labels(context)
                self.metrics.record_cache(
                    "skip",
                    tenant_id=tenant_id,
                    kb_id=kb_id,
                )
            return False
        tenant_id, kb_id = self._labels(context)
        ok = self.cache.set(
            self.scope,
            query.strip(),
            response,
            ttl=ttl or self.default_ttl or response_ttl_for(response),
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

    def delete(self, query: str, *, context: AgentContext | None = None) -> bool:
        """删除指定查询的缓存（例如知识更新后主动失效）。

        参数:
            query: 规范化后的查询文本（作为缓存 key）。
            context: agent 上下文；非空时用于租户隔离。

        返回:
            True 表示删除成功；False 表示底层不可用或删除失败。
        """
        if self.cache is None:
            return False
        tenant_id, kb_id = self._labels(context)
        return self.cache.delete(
            self.scope,
            query.strip(),
            tenant_id=tenant_id,
            kb_id=kb_id,
        )
