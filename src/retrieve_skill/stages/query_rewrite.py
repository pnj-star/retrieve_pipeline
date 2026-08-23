"""查询改写阶段：LLM 改写 / 查询扩展。

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

from common_core.config import VALID_QUERY_REWRITE_MODES, QueryRewriteConfig
from common_core.context import AgentContext

logger = logging.getLogger(__name__)

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
    """规范化改写策略；未知值回退到 default。

    参数:
        mode: 原始改写模式（可能为 None 或未知值）。
        default: 未知 / 空值时的回退模式，默认 "off"。

    返回:
        归一化后的合法改写模式字符串。
    """
    value = str(mode or "").strip().lower()
    return value if value in VALID_QUERY_REWRITE_MODES else default


def _strip_fences(text: str) -> str:
    """去掉 LLM 返回内容外围的 ``` 代码块围栏，只留内部正文。

    参数:
        text: LLM 返回的原始内容。

    返回:
        去掉外层围栏后的正文；无围栏时原样返回去首尾空白的文本。
    """
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3:
            return "\n".join(lines[1:-1]).strip()
    return stripped


def _clean_query_line(line: str) -> str:
    """清理单条查询：去掉列表前缀（- * • 或 1. 1、）与多余引号。

    参数:
        line: 单条查询的原始文本行。

    返回:
        清理后的查询文本；空行返回空字符串。
    """
    cleaned = line.strip()
    if not cleaned:
        return ""
    cleaned = re.sub(r"^\s*(?:[-*•]|\d+[.)、])\s*", "", cleaned).strip()
    return cleaned.strip("\"'“”‘’「」")


def _parse_query_variants(raw: str) -> list[str]:
    """解析 LLM 输出的多个查询，兼容 JSON、代码块与逐行文本。

    参数:
        raw: LLM 返回的原始文本。

    返回:
        解析出的查询文本去重列表；无法解析时为空列表。
    """
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
    """一次查询改写的结果，供检索与排查使用。

    字段:
        mode: 本次生效的改写模式。
        original_query: 改写前的原始用户查询。
        rewritten_query: 改写后实际用于检索的主查询（回退时等于原查询）。
        query_variants: 检索使用的查询变体列表（展开模式含多个）。
        error: 改写过程中的错误信息；无错误时为空字符串。
    """

    mode: str
    original_query: str
    rewritten_query: str
    query_variants: list[str] = field(default_factory=list)
    error: str = ""

    def to_trace(self) -> dict[str, Any]:
        """把这次改写结果拍平成日志 / 排障用的字典。

        返回:
            包含 mode / original_query / rewritten_query / query_variants
            （以及可选的 error）的字典。
        """
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
        """持有改写要用的 LLM、配置与指标对象；未传配置时用默认值。

        参数:
            llm: 用于改写的底层 LLM 客户端。
            config: 查询改写配置；None 用默认 QueryRewriteConfig()。
            metrics: 可观测性 / 指标对象；可为 None。
        """
        self.llm = llm
        self.config = config or QueryRewriteConfig()
        self.metrics = metrics

    def resolve_mode(
        self,
        context: AgentContext | None = None,
        requested_mode: str | None = None,
    ) -> str:
        """综合请求参数与配置，算出本次实际生效的改写模式。

        优先级：调用方显式传的 ``requested_mode`` > 租户/KB 作用域的
        ``scoped_modes`` > 全局默认 ``mode``。归一化后返回一个合法模式。

        参数:
            context: agent 上下文；非空时用于按租户/KB 查 scoped_modes。
            requested_mode: 调用方显式指定的模式；None 时按配置决定。

        返回:
            归一化后的合法改写模式。
        """
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
        """llm_model 为空时复用管线 LLM，否则构造同配置但不同模型的客户端。

        返回:
            本次改写实际要用的 LLM 客户端。
        """
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
        """对外入口：改写一条查询，任何异常都回退为原查询，绝不阻塞检索。

        参数:
            query: 原始用户查询。
            context: agent 上下文；可为 None。
            mode: 本次显式指定的改写模式；None 时按配置决定。

        返回:
            QueryRewriteResult；异常时改写结果回退到原查询并记录 error。
        """
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
        """按解析出的模式分发到具体的改写实现。

        参数:
            query: 原始用户查询。
            mode: 已解析并归一化的改写模式。

        返回:
            QueryRewriteResult，含改写后的查询与变体列表。
        """
        if mode == "off":
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
        """单查询改写：让 LLM 产出一个更适合检索的查询文本。

        参数:
            query: 原始用户查询。
            mode: 改写模式（用于结果记录）。

        返回:
            QueryRewriteResult，改写失败则回退到原查询并记录 error。
        """
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
        """查询扩展：让 LLM 产出多个不同角度的检索查询以提升召回。

        参数:
            query: 原始用户查询。
            mode: 改写模式（用于结果记录）。

        返回:
            QueryRewriteResult；主查询取首个扩展，query_variants 含原查询与扩展。
        """
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
