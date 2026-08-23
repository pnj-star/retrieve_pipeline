
"""RAG 检索工具返回的结构化结果契约。

status 枚举是契约中稳定的机器可读部分。``docs`` 始终存放经过管线保留的文档
（在 ``no_context`` 场景下则为候选项），供 agent 编排层做多来源融合、复核或转人工。
检索管道不生成回答，因此这里只有检索侧状态与结果。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class RetrieveStatus:
    """检索阶段（rag_retrieve）的结果状态枚举。

    检索管道只负责把"精排后达到相关性阈值的文档"交给 agent，不做回答生成。
        RETRIEVED: 正常检索到并精排后达到阈值。
        RETRIEVED_CACHE: 命中检索缓存（query → 精排后达标文档）。
        NO_CONTEXT: 检索为空 / 混合检索两边都空 / 精排后没有文档达到阈值。
        ERROR: 管线内部异常。
    """

    RETRIEVED = "retrieved"  # 检索 + 精排后达标
    RETRIEVED_CACHE = "retrieved_cache"  # 命中检索缓存
    NO_CONTEXT = "no_context"  # 没有达到阈值的文档
    ERROR = "error"  # 管线内部异常


@dataclass(slots=True)
class RetrieveResult:
    """检索管道结果：状态、精排后的候选文档、是否命中缓存。

    ``docs`` 始终携带检索/精排后的候选文档（即使 ``status`` 为 ``no_context``，
    便于调用方 agent 结合候选自行判断是否转人工）。是否达到阈值由 ``status``
    表达：``retrieved`` / ``retrieved_cache`` 表示有达标文档可用，
    ``no_context`` 表示没有达标文档。
    """

    status: str
    docs: list[dict[str, Any]] = field(default_factory=list)
    rewritten_query: str = ""
    cache_hit: bool = False
    message: str = ""

    @property
    def ok(self) -> bool:
        """是否未发生内部异常（仅 ERROR 视为失败）。"""
        return self.status != RetrieveStatus.ERROR

    @property
    def has_context(self) -> bool:
        """是否有可供 agent 直接使用的达标文档。"""
        return self.status in (RetrieveStatus.RETRIEVED, RetrieveStatus.RETRIEVED_CACHE)


__all__ = ["RetrieveResult", "RetrieveStatus"]
