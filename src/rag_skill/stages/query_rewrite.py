"""查询改写阶段：identity / LLM 改写 / 查询扩展。

默认 mode 为 off，完全保持现有检索行为；启用后，改写只影响检索与精排使用的
查询文本，不改变最终回答针对原始用户问题的语义。LLM 调用失败时自动回退为
原始查询，避免让一个可选的增强环节拖垮整条 RAG 链路。
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from common_core.config import QueryRewriteConfig
from common_core.context import AgentContext

logger = logging.getLogger(__name__)

VALID_QUERY_REWRITE_MODES = frozenset(
    {"off", "identity", "llm_rewrite", "query_expansion"}
)

DEFAULT_QUERY_REWRITE_PROMPT = (
    "你是企业知识库检索的查询改写助手。请把用户问题改写成适合向量检索和"
    "关键词检索的查询：保留原始意图，补全省略的指代或上下文，去掉寒暄。"
    "只输出改写后的查询文本，不要解释、引号或编号。"
)

DEFAULT_QUERY_EXPANSION_PROMPT = (
    "你是企业知识库检索的查询扩展助手。请为给定查询生成 {count} 个不同角度的"
    "检索查询，覆盖同义词、常见说法、术语和具体化表达。每个查询单独一行，"
    "不要编号、解释或引号。"
)


def normalize_query_rewrite_mode(mode: str | None, default: str = "off") -> str:
    """规范化改写策略；未知值回退到 default。"""
    value = str(mode or "").strip().lower()
    return value if value in VALID_QUERY_REWRITE_MODES else default


def _strip_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3:
            return "\n".join(lines[1:-1]).strip()
    return stripped


def _clean_query_line(line: str) -> str:
    cleaned = line.strip()
    if not cleaned:
        return ""
    cleaned = re.sub(r"^\s*(?:[-*•]|\d+[.)、])\s*", "", cleaned).strip()
    return cleaned.strip("\"'“”‘’「」")


def _parse_query_variants(raw: str) -> list[str]:
    """解析 LLM 输出的多个查询，兼容 JSON、代码块与逐行文本。"""
    text = _strip_fences(raw)
    if not text:
        return []
    if text.startswith("["):
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = None
        if isinstance(data, list):
            return [str(item).strip() for item in data if str(item).strip()]
    if text.startswith("{"):
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict):
            for key in ("queries", "variants", "query"):
                value = data.get(key)
                if isinstance(value, list):
                    return [str(item).strip() for item in value if str(item).strip()]
                if isinstance(value, str) and value.strip():
                    return [value.strip()]
    variants: list[str] = []
    for line in text.splitlines():
        cleaned = _clean_query_line(line)
        if cleaned and cleaned not in variants:
            variants.append(cleaned)
    return variants


@dataclass(slots=True)
class QueryRewriteResult:
    """一次查询改写的结果，供检索与排查使用。"""

    mode: str
    original_query: str
    rewritten_query: str
    query_variants: list[str] = field(default_factory=list)
    error: str = ""

    def to_trace(self) -> dict[str, Any]:
        trace: dict[str, Any] = {
            "mode": self.mode,
            "original_query": self.original_query,
            "rewritten_query": self.rewritten_query,
            "query_variants": list(self.query_variants),
        }
        if self.error:
            trace["error"] = self.error
        return trace


class QueryRewriter:
    """根据配置选择查询改写策略，并在失败时回退到原始查询。"""

    def __init__(
        self,
        llm: Any,
        config: QueryRewriteConfig | None = None,
        *,
        metrics: Any = None,
    ) -> None:
        self.llm = llm
        self.config = config or QueryRewriteConfig()
        self.metrics = metrics

    def resolve_mode(
        self,
        context: AgentContext | None = None,
        requested_mode: str | None = None,
    ) -> str:
        explicit = normalize_query_rewrite_mode(requested_mode, default=self.config.mode)
        if requested_mode is not None:
            return explicit
        if context is not None and self.config.scoped_modes:
            candidates = (
                f"{context.tenant_id}/{context.kb_id}",
                f"{context.tenant_id}/*",
                f"*/{context.kb_id}",
                "*/*",
            )
            for scope in candidates:
                mode = self.config.scoped_modes.get(scope)
                if mode:
                    return normalize_query_rewrite_mode(mode, default=self.config.mode)
        return normalize_query_rewrite_mode(
            self.config.mode,
            default="off",
        )

    def _llm_for_rewrite(self) -> Any:
        """llm_model 为空时复用管线 LLM，否则构造同配置但不同模型的客户端。"""
        if not self.config.llm_model:
            return self.llm
        current_model = getattr(getattr(self.llm, "config", None), "model", "")
        if current_model == self.config.llm_model:
            return self.llm
        from common_core.config import LLMConfig
        from common_core.providers import OpenAICompatibleLLM

        base = getattr(self.llm, "config", None)
        return OpenAICompatibleLLM(
            LLMConfig(
                base_url=getattr(base, "base_url", "") if base else "",
                api_key=getattr(base, "api_key", "") if base else "",
                model=self.config.llm_model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                timeout_seconds=getattr(base, "timeout_seconds", 30.0) if base else 30.0,
            ),
            metrics=self.metrics,
        )

    async def rewrite(
        self,
        query: str,
        context: AgentContext | None = None,
        *,
        mode: str | None = None,
    ) -> QueryRewriteResult:
        resolved = self.resolve_mode(context, requested_mode=mode)
        try:
            return await self._rewrite_impl(query, resolved)
        except Exception as exc:  # 改写永远不阻塞检索
            logger.warning(
                "query_rewrite.fallback mode=%s error=%s query=%s",
                resolved,
                exc,
                query,
            )
            return QueryRewriteResult(
                mode=resolved,
                original_query=query,
                rewritten_query=query,
                query_variants=[query],
                error=str(exc) or exc.__class__.__name__,
            )

    async def _rewrite_impl(
        self,
        query: str,
        mode: str,
    ) -> QueryRewriteResult:
        if mode in {"off", "identity"}:
            return QueryRewriteResult(
                mode=mode,
                original_query=query,
                rewritten_query=query,
                query_variants=[query],
            )
        if mode == "llm_rewrite":
            return await self._rewrite_single(query, mode)
        if mode == "query_expansion":
            return await self._expand_query(query, mode)
        # 配置打了未知值：按 off 处理，保持行为可控。
        return QueryRewriteResult(
            mode="off",
            original_query=query,
            rewritten_query=query,
            query_variants=[query],
            error=f"unknown_mode:{mode}",
        )

    async def _rewrite_single(
        self,
        query: str,
        mode: str,
    ) -> QueryRewriteResult:
        prompt = self.config.rewrite_prompt or DEFAULT_QUERY_REWRITE_PROMPT
        raw = await self._llm_for_rewrite().chat(
            [{"role": "user", "content": query}],
            system_prompt=prompt,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        variants = _parse_query_variants(raw)
        rewritten = variants[0] if variants else ""
        if not rewritten or rewritten.lower() == query.strip().lower():
            rewritten = query
            error = "" if not variants else "rewrite_unchanged"
        else:
            error = ""
        return QueryRewriteResult(
            mode=mode,
            original_query=query,
            rewritten_query=rewritten,
            query_variants=[rewritten],
            error=error,
        )

    async def _expand_query(
        self,
        query: str,
        mode: str,
    ) -> QueryRewriteResult:
        count = max(1, self.config.expand_count)
        prompt = (
            self.config.expansion_prompt
            or DEFAULT_QUERY_EXPANSION_PROMPT
        ).format(count=count)
        raw = await self._llm_for_rewrite().chat(
            [{"role": "user", "content": query}],
            system_prompt=prompt,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        expansions = [
            item
            for item in _parse_query_variants(raw)
            if item and item.lower() != query.strip().lower()
        ][:count]
        if not expansions:
            return QueryRewriteResult(
                mode=mode,
                original_query=query,
                rewritten_query=query,
                query_variants=[query],
                error="no_expansions",
            )
        variants = [query, *expansions]
        return QueryRewriteResult(
            mode=mode,
            original_query=query,
            rewritten_query=expansions[0],
            query_variants=variants,
        )


__all__ = [
    "DEFAULT_QUERY_EXPANSION_PROMPT",
    "DEFAULT_QUERY_REWRITE_PROMPT",
    "QueryRewriteConfig",
    "QueryRewriteResult",
    "QueryRewriter",
    "VALID_QUERY_REWRITE_MODES",
    "normalize_query_rewrite_mode",
]
