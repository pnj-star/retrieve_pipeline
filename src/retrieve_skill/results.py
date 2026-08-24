
"""RAG 检索工具返回的结构化结果契约。

status 枚举是契约中稳定的机器可读部分。``docs`` 只携带通过精排阈值并完成
父块聚合的上下文；低于阈值或精排器故障时一律不返回候选原文，避免低置信内容
进入下游 LLM。检索管道不生成回答，因此这里只有检索侧状态与结果。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class RetrieveStatus:
    """检索阶段（rag_retrieve）的结果状态枚举。

    检索管道只负责把"精排后达到相关性阈值的文档"交给 agent，不做回答生成。
        RETRIEVED: 正常检索到并精排后达到阈值。
        RETRIEVED_CACHE: 命中合格父块引用缓存并回源重建后的上下文。
        NO_CONTEXT: 检索为空 / 混合检索两边都空 / 精排后没有文档达到阈值。
        ERROR: 管线内部异常。
    """

    RETRIEVED = "retrieved"  # 检索 + 精排后达标
    RETRIEVED_CACHE = "retrieved_cache"  # 命中检索缓存
    NO_CONTEXT = "no_context"  # 没有达到阈值的文档
    ERROR = "error"  # 管线内部异常


@dataclass(slots=True)
class RetrieveResult:
    """检索管道结果：状态、达标父块上下文、是否命中缓存与诊断信息。

    ``docs`` 只在成功路径携带父块粒度文档；``no_context`` 和 ``error`` 路径
    一律为空列表，防止未达标的子块被下游模型当作事实依据。诊断字段只保留
    数量和分数等摘要，不携带候选正文。

    字段:
        status: 机器可读状态，取值见 RetrieveStatus。
        docs: 最终父块上下文；成功时非空，失败或无相关内容时为空。
        rewritten_query: 实际用于检索的查询；未改写时等于原始 query。
        cache_hit: 是否命中父块引用缓存。命中后仍会回源 MySQL 校验正文和版本。
        message: 给人和日志看的简短状态说明。
        diagnostics: 数量、分数、缺失原因等排障摘要；不包含候选正文。
    """

    status: str
    docs: list[dict[str, Any]] = field(default_factory=list)
    rewritten_query: str = ""
    cache_hit: bool = False
    message: str = ""
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """是否未发生内部异常（仅 ERROR 视为失败）。"""
        return self.status != RetrieveStatus.ERROR

    @property
    def has_context(self) -> bool:
        """是否有可供 agent 直接使用的达标文档。"""
        return self.status in (RetrieveStatus.RETRIEVED, RetrieveStatus.RETRIEVED_CACHE)


__all__ = ["RetrieveResult", "RetrieveStatus"]
