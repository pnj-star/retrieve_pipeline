"""RAG 工具返回的结构化结果契约。

status 枚举是契约中稳定的机器可读部分。``docs`` 始终存放经过管线保留的文档
（在 ``no_context`` 场景下则为候选文档），``answer`` 存放生成或缓存的回答。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class RagStatus:
    """RAG 处理结果的机器可读状态枚举。"""

    ANSWERED = "answered"  # 正常生成完成
    ANSWERED_CACHE = "answered_cache"  # 命中响应缓存，直接返回
    NO_CONTEXT = "no_context"  # 没有检索到足够相关的上下文
    GUARD_BLOCKED = "guard_blocked"  # 输出被质量护栏拦截
    ERROR = "error"  # 管线内部异常


@dataclass(slots=True)
class RagResult:
    """RAG 返回结果：状态、说明、精排后的文档与最终回答。"""

    status: str
    message: str
    docs: list[dict[str, Any]] = field(default_factory=list)
    answer: str = ""

    @property
    def ok(self) -> bool:
        """是否成功：仅 ERROR 状态视为失败。"""
        return self.status != RagStatus.ERROR

    def to_dict(self) -> dict[str, Any]:
        """转成字典，便于在 MCP 工具返回值中序列化。"""
        return {
            "status": self.status,
            "message": self.message,
            "docs": self.docs,
            "answer": self.answer,
        }


__all__ = ["RagResult", "RagStatus"]