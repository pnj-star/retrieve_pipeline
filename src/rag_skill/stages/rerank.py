"""交叉编码器重排与相关性门控。

本阶段不含业务逻辑：模型名、设备、分数混合权重与 top-k 均为可配置项；
调用方可以注入 ``score_fn`` 用于测试或接入不同的重排后端。
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Sequence

logger = logging.getLogger(__name__)

DEFAULT_RERANK_MODEL = "BAAI/bge-reranker-base"  # 默认交叉编码器模型
DEFAULT_CE_WEIGHT = 0.6  # 融合分中交叉编码器分数的权重
DEFAULT_RETRIEVAL_WEIGHT = 0.4  # 融合分中检索原始分数的权重


def judge_relevance(docs: Sequence[dict[str, Any]], min_relevance: float) -> bool:
    """相关性门控：返回 True 表示没有文档达到相关性阈值（管线返回 no_context）。

    缺失 ``ce_score`` 按 0.0 处理，因此透传（未重排）的结果永远不会被
    误判为相关。
    """
    if not docs:
        return True
    best = max((float(doc.get("ce_score", 0.0)) for doc in docs), default=0.0)
    return best < min_relevance


def rank_docs(
    docs: Sequence[dict[str, Any]],
    query: str = "",
    *,
    scores: Sequence[float] | None = None,
    top_k: int = 3,
    ce_weight: float = DEFAULT_CE_WEIGHT,
    retrieval_weight: float = DEFAULT_RETRIEVAL_WEIGHT,
) -> list[dict[str, Any]]:
    """给文档附加 ``ce_score`` 与融合分数，排序后截断到 top_k。

    当没有 query 或没有 scores 时，前 top_k 个文档原样透传，
    保证检索上下文不会丢失（例如模型不可用时的优雅降级）。
    """
    docs = list(docs)
    if not docs:
        return []
    if not query.strip() or scores is None:
        return docs[:top_k]

    scored: list[dict[str, Any]] = []
    for index, doc in enumerate(docs):
        raw = float(scores[index]) if index < len(scores) else 0.0
        ce_score = max(0.0, min(1.0, raw))  # 交叉编码器分数限制在 [0, 1]
        retrieval_score = float(doc.get("score", 0.0) or 0.0)
        item = dict(doc)
        item["ce_score"] = ce_score
        # 融合分 = 交叉编码器分×权重 + 检索分×权重
        item["score"] = ce_weight * ce_score + retrieval_weight * retrieval_score
        scored.append(item)

    scored.sort(key=lambda item: item.get("score", 0.0), reverse=True)
    return scored[:top_k]


class Reranker:
    """懒加载的交叉编码器重排器，模型不可用时优雅透传（return passthrough）。"""

    def __init__(
        self,
        model_name: str = DEFAULT_RERANK_MODEL,
        *,
        device: str | None = None,
        top_k: int = 3,
        score_fn: Callable[[str, Sequence[str]], Sequence[float]] | None = None,
        metrics: Any = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.top_k = top_k
        self.score_fn = score_fn  # 可注入的评分函数（测试 / 替代后端）
        self.metrics = metrics
        self._model: Any = None
        self._available: bool | None = None  # None=未尝试，True=可用，False=不可用

    def _load_model(self) -> Any | None:
        """首次使用时才加载模型；加载失败则标记不可用并返回 None。"""
        if self._available is False:
            return None
        if self._model is None:
            try:
                import torch
                from sentence_transformers import CrossEncoder

                device = self.device or (
                    "cuda" if torch.cuda.is_available() else "cpu"
                )
                logger.info("Loading reranker (%s) on %s", self.model_name, device)
                self._model = CrossEncoder(self.model_name, device=device)
                self._available = True
            except Exception as exc:
                # 模型（或依赖）不可用：降级为透传，不阻塞管线
                logger.warning(
                    "Reranker model unavailable, using passthrough: %s", exc
                )
                self._available = False
                self._model = None
        return self._model

    def _scores(
        self,
        query: str,
        contents: Sequence[str],
    ) -> Sequence[float] | None:
        """计算每条内容的相似度分数；无模型 / 无 score_fn 时返回 None。"""
        if self.score_fn is not None:
            return list(self.score_fn(query, contents))
        model = self._load_model()
        if model is None:
            return None
        return list(model.predict([(query, content) for content in contents]))

    def rank(
        self,
        query: str,
        docs: Sequence[dict[str, Any]],
        *,
        tenant_id: str = "",
        kb_id: str = "",
    ) -> list[dict[str, Any]]:
        """对文档做重排：计算分数、融合排序并截断到 top_k。"""
        contents = [
            str(doc.get("parent_content", "") or doc.get("content", "") or "")
            for doc in docs
        ]
        scores = self._scores(query, contents)
        ranked = rank_docs(
            docs,
            query,
            scores=scores,
            top_k=self.top_k,
        )
        if scores is not None and self.metrics is not None:
            # 上报本次重排的最佳分数，供相关性监控使用
            try:
                best = max(
                    (float(doc.get("ce_score", 0.0)) for doc in ranked),
                    default=0.0,
                )
                self.metrics.record_rerank_best(
                    best,
                    tenant_id=tenant_id,
                    kb_id=kb_id,
                )
            except Exception:
                logger.debug("rerank metric record failed (ignored)", exc_info=True)
        return ranked

    async def arank(
        self,
        query: str,
        docs: Sequence[dict[str, Any]],
        *,
        tenant_id: str = "",
        kb_id: str = "",
    ) -> list[dict[str, Any]]:
        """异步重排：把阻塞式计算放到线程池，避免卡住事件循环。"""
        return await asyncio.to_thread(
            self.rank,
            query,
            list(docs),
            tenant_id=tenant_id,
            kb_id=kb_id,
        )
