"""人工交接（handoff）：记录、存储与模板选择。

本阶段不含业务逻辑：记录只携带通用字段，由实例决定使用的模板、关键词与
持久化后端。持久化写入失败只记日志、绝不抛错；实例可自行实现一个持久化
writer 并放到 Redis 兜底之前（见 ChainHandoffStore）。
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, Sequence

from common_core.providers import RedisCache

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class HandoffRecord:
    """一条人工交接记录：问题、原因、候选文档/图片与 TTL。"""

    id: str
    query: str
    reason: str
    intent: str = "standard"  # 意图分类，由实例约定
    candidate_docs: list[dict[str, Any]] = field(default_factory=list)
    candidate_images: list[dict[str, Any]] = field(default_factory=list)
    ttl_seconds: int = 3600

    def to_json(self) -> str:
        """序列化为 JSON；ensure_ascii=False 保留中文原文。"""
        return json.dumps(
            {
                "id": self.id,
                "query": self.query,
                "reason": self.reason,
                "intent": self.intent,
                "candidate_docs": self.candidate_docs,
                "candidate_images": self.candidate_images,
                "ttl_seconds": self.ttl_seconds,
            },
            ensure_ascii=False,
            default=str,
        )


def record_from_json(raw: str) -> HandoffRecord:
    """从 JSON 字符串还原 HandoffRecord。"""
    data = json.loads(raw)
    return HandoffRecord(
        id=str(data.get("id", "")),
        query=str(data.get("query", "")),
        reason=str(data.get("reason", "")),
        intent=str(data.get("intent", "standard")),
        candidate_docs=list(data.get("candidate_docs", []) or []),
        candidate_images=list(data.get("candidate_images", []) or []),
        ttl_seconds=int(data.get("ttl_seconds", 3600)),
    )


class HandoffStore(Protocol):
    """交接记录的存储协议：实现方决定底层后端。"""

    def save(self, record: HandoffRecord) -> bool: ...

    async def asave(self, record: HandoffRecord) -> bool: ...

    def get(self, record_id: str) -> HandoffRecord | None: ...

    def delete(self, record_id: str) -> bool: ...


class RedisHandoffStore:
    """基于 Redis 的交接存储；失败只记日志，绝不抛错。"""

    def __init__(
        self,
        cache: RedisCache | None = None,
        *,
        scope: str = "handoff",
    ) -> None:
        self.cache = cache or RedisCache()
        self.scope = scope

    def _key(self, record_id: str) -> str:
        """组装 Redis key：key_prefix + scope + record_id。"""
        return f"{self.cache.config.key_prefix}:{self.scope}:{record_id}"

    def save(self, record: HandoffRecord) -> bool:
        """写入记录（带 TTL），失败返回 False 而非抛错。"""
        try:
            client = self.cache.client()
            client.setex(
                self._key(record.id),
                record.ttl_seconds,
                record.to_json(),
            )
            return True
        except Exception as exc:
            logger.warning("handoff Redis save failed: %s", exc)
            return False

    async def asave(self, record: HandoffRecord) -> bool:
        """异步保存：把阻塞式 save 放到线程池执行。"""
        return await asyncio.to_thread(self.save, record)

    def get(self, record_id: str) -> HandoffRecord | None:
        """读取记录；不存在或出错时返回 None。"""
        try:
            client = self.cache.client()
            raw = client.get(self._key(record_id))
            return record_from_json(raw) if raw else None
        except Exception as exc:
            logger.warning("handoff Redis get failed: %s", exc)
            return None

    def delete(self, record_id: str) -> bool:
        """删除记录，返回是否删除成功。"""
        try:
            client = self.cache.client()
            return bool(client.delete(self._key(record_id)))
        except Exception as exc:
            logger.warning("handoff Redis delete failed: %s", exc)
            return False


class ChainHandoffStore:
    """按顺序尝试多个存储；第一个保存成功的生效（后续作为兜底）。"""

    def __init__(self, stores: Sequence[Any]) -> None:
        self.stores = list(stores)

    def save(self, record: HandoffRecord) -> bool:
        for store in self.stores:
            try:
                if store.save(record):
                    return True
            except Exception as exc:
                logger.warning("handoff store save raised: %s", exc)
        return False

    async def asave(self, record: HandoffRecord) -> bool:
        return await asyncio.to_thread(self.save, record)

    def get(self, record_id: str) -> HandoffRecord | None:
        for store in self.stores:
            try:
                record = store.get(record_id)
            except Exception as exc:
                logger.warning("handoff store get raised: %s", exc)
                continue
            if record is not None:
                return record
        return None

    def delete(self, record_id: str) -> bool:
        for store in self.stores:
            try:
                if store.delete(record_id):
                    return True
            except Exception as exc:
                logger.warning("handoff store delete raised: %s", exc)
        return False


def build_handoff_record(
    query: str,
    reason: str,
    *,
    docs: Sequence[dict[str, Any]] = (),
    images: Sequence[dict[str, Any]] = (),
    intent: str = "standard",
    doc_limit: int = 3,
    doc_chars: int = 500,
    image_limit: int = 5,
    id_prefix: str = "handoff",
    ttl_seconds: int = 3600,
) -> HandoffRecord:
    """构造交接记录：候选文档 / 图片做截断，自动生成带前缀的唯一 id。"""
    record_id = f"{id_prefix}:{uuid.uuid4().hex[:12]}"
    candidate_docs = [
        {
            "content": str(doc.get("content", ""))[:doc_chars],
            "score": doc.get("score", 0),
        }
        for doc in list(docs)[:doc_limit]
    ]
    candidate_images = [
        {
            "url": image.get("image_url", ""),
            "description": image.get("description", ""),
        }
        for image in list(images)[:image_limit]
    ]
    return HandoffRecord(
        id=record_id,
        query=query,
        reason=reason,
        intent=intent,
        candidate_docs=candidate_docs,
        candidate_images=candidate_images,
        ttl_seconds=ttl_seconds,
    )


def default_template_selector(
    templates: dict[str, str],
) -> Callable[[str, bool, str], str]:
    """返回一个模板选择器：优先精确命中 reason，其次空上下文，最后 confidence。

    选择顺序：
      1. 传入了 reason 且在模板表里 → 用该 reason 对应的模板；
      2. 没有候选文档且模板表里有 "empty" → 用空上下文模板；
      3. 否则 → 用 "confidence" 模板（可能为空字符串）。
    """

    def select(query: str, has_docs: bool, reason: str = "") -> str:
        if reason and reason in templates:
            return templates[reason]
        if not has_docs and "empty" in templates:
            return templates["empty"]
        return templates.get("confidence", "")

    return select


async def persist_handoff(store: Any, record: HandoffRecord) -> bool:
    """按存储协议持久化交接记录，不阻塞事件循环。

    优先用异步 asave；没有 asave 时把阻塞式 save 放到线程池执行。
    """
    if hasattr(store, "asave"):
        return bool(await store.asave(record))
    return bool(await asyncio.to_thread(store.save, record))