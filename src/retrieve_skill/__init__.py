"""基于 common_core provider 的可复用 RAG 检索技能包。

对外统一导出：组装函数（build_runtime / build_pipeline）、管线（RagPipeline）、
检索结果契约（RetrieveResult / RetrieveStatus）、MCP 入口
（create_mcp_server / main），以及检索侧阶段（重排、查询改写、检索缓存）。

回答侧的上下文组装 / 生成 / 护栏位于 ``common_core.rag``，
供 retrieve_skill 之外的 agent / 多 agent 框架复用。
"""

from .builder import build_pipeline, build_runtime
from .mcp import create_mcp_server, main
from .pipeline import RagPipeline
from .results import RetrieveResult, RetrieveStatus
from .stages import (
    Reranker,
    RetrievalCache,
    judge_relevance,
    rank_docs,
)

__all__ = [
    "RagPipeline",
    "Reranker",
    "RetrievalCache",
    "RetrieveResult",
    "RetrieveStatus",
    "build_pipeline",
    "build_runtime",
    "create_mcp_server",
    "judge_relevance",
    "main",
    "rank_docs",
]

__version__ = "0.1.0"
