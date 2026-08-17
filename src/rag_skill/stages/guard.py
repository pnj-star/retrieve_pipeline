"""输出质量护栏：确定性检查 + LLM 评审与重试。

- 确定性检查（绝对化用语、数字溯源）是通用的；
- 业务评审提示词与绝对化词表由实例通过 ``GuardConfig`` 注入，
  本层内置的默认值刻意保持保守（fail-closed）。
- LLM 评审不可用时按“不通过”处理，宁可拦截也不放行有风险的回答。
"""

from __future__ import annotations

import logging
import re
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# 默认的绝对化 / 夸大用语黑名单（业务实例可覆盖）。
DEFAULT_ABSOLUTE_WORDS: tuple[str, ...] = (
    "绝对",
    "百分百",
    "全网最低",
    "最低价",
    "最强",
    "包治",
    "根治",
    "零风险",
    "无副作用",
    "立竿见影",
)

# 风险数字正则：小数、以及带单位量词后缀的数字（百分比/点/成/折/元/块/钱/单）
_RISKY_NUM_RE = re.compile(r"\d+\.\d+|\d+(?=[%‰点成折元块钱单]|个点)")
# 全部数字正则：任意普通整数 / 小数
_ALL_NUM_RE = re.compile(r"\d+(?:\.\d+)?")

# 默认评审提示词：让 LLM 按固定 JSON 结构判断回答是否合格。
DEFAULT_REVIEW_PROMPT = """You are an output quality reviewer. Check whether the assistant response is acceptable. Output JSON with these keys: pass (bool), issues (list[str]), fabrication (bool), exaggeration (bool), safety_violation (bool), out_of_domain (bool), unhelpful (bool).

Check standards:
- fabrication: the response invents specific facts that are not present in the reference information
- exaggeration: the response uses absolute claims such as "best", "100%", "absolutely", or promises health cures
- safety_violation: the response gives unsafe advice
- out_of_domain: the response drifts outside the assistant's intended domain
- unhelpful: the response provides no real help (a reasonable handoff to human support does not count)

A response that honestly says information is missing and suggests contacting support is not unhelpful."""


def _norm_num(value: str) -> str:
    """归一化数字字符串：去掉小数末尾的 0 与小数点（15% 与 0.15 视为不同数字）。"""
    if "." in value:
        value = value.rstrip("0").rstrip(".")
    return value


def extract_risky_numbers(text: str) -> list[str]:
    """提取风险数字：小数与带单位量词的数字（费率、金额、数量等）。"""
    return [_norm_num(match.group(0)) for match in _RISKY_NUM_RE.finditer(text or "")]


def extract_all_numbers(text: str) -> set[str]:
    """提取参考文本中所有归一化后的数字（用于数字溯源比对）。"""
    return {
        _norm_num(match.group(0))
        for match in _ALL_NUM_RE.finditer(text or "")
    }


def check_compound_numbers(response: str, context_text: str) -> list[str]:
    """数字溯源检查：回答中的风险数字必须同时出现在参考上下文里。

    防止模型编造出上下文里不存在的费率 / 金额 / 数量等关键数字。
    """
    allowed = extract_all_numbers(context_text or "")
    return [
        f"response contains number not present in reference: {number}"
        for number in extract_risky_numbers(response or "")
        if number not in allowed
    ]


def absolute_language_issues(
    response: str,
    absolute_words: Sequence[str] = DEFAULT_ABSOLUTE_WORDS,
) -> list[str]:
    """检查回答中是否出现绝对化 / 夸大用语，返回命中的问题列表。"""
    return [
        f"absolute language: {word}"
        for word in absolute_words
        if word and word in (response or "")
    ]


@dataclass(slots=True)
class GuardConfig:
    """护栏配置：词表、重试次数与评审提示词。"""

    absolute_words: tuple[str, ...] = DEFAULT_ABSOLUTE_WORDS
    max_retries: int = 2  # 生成失败后最多重试次数
    review_prompt: str = DEFAULT_REVIEW_PROMPT
    temperature: float = 0.0  # 评审 LLM 使用低温度，保证判断稳定


@dataclass(slots=True)
class GuardResult:
    """单次评审结果：是否通过 + 各类问题标记 + 原因。"""

    passed: bool
    issues: list[str] = field(default_factory=list)
    fabrication: bool = False  # 编造事实
    exaggeration: bool = False  # 夸大 / 绝对化
    safety_violation: bool = False  # 安全违规
    out_of_domain: bool = False  # 跑题 / 越界
    unhelpful: bool = False  # 没有实际帮助
    reason: str = ""


async def evaluate_guard(
    llm: Any,
    response: str,
    *,
    query: str = "",
    context_text: str = "",
    config: GuardConfig | None = None,
    enforce_numbers: bool = False,
) -> GuardResult:
    """评审一次回答；LLM 评审不可用时按“不通过”处理（fail-closed）。

    - enforce_numbers=True 时额外启用数字溯源检查（默认关闭，按实例配置）。
    """
    cfg = config or GuardConfig()
    if not response:
        # 空回答直接拦截
        return GuardResult(passed=False, issues=["empty_response"], reason="empty_response")

    # 先跑确定性检查：绝对化用语 + （可选的）数字溯源
    absolute_issues = absolute_language_issues(response, cfg.absolute_words)
    number_issues = (
        check_compound_numbers(response, context_text) if enforce_numbers else []
    )
    issues = absolute_issues + number_issues
    if issues:
        # 确定性检查已命中：不必再走 LLM 评审
        return GuardResult(
            passed=False,
            issues=issues,
            exaggeration=bool(absolute_issues),
            fabrication=bool(number_issues),
            reason="; ".join(issues),
        )

    # 交给评审 LLM 做语义层面的审核
    try:
        result = await llm.chat_json(
            [
                {
                    "role": "user",
                    "content": f"Review the following assistant response:\n\n{response}",
                }
            ],
            system_prompt=cfg.review_prompt,
            temperature=cfg.temperature,
        )
    except Exception as exc:
        logger.warning("guard LLM unavailable, treating as fail: %s", exc)
        return GuardResult(
            passed=False,
            issues=["audit_llm_unavailable"],
            reason="audit_llm_unavailable",
        )

    if not isinstance(result, dict) or not result.get("pass", False):
        # 未通过：透出各项问题标记
        issues = list(result.get("issues", [])) if isinstance(result, dict) else []
        reason = "; ".join(str(item) for item in issues) or "guard_failed"
        return GuardResult(
            passed=False,
            issues=issues,
            fabrication=bool(result.get("fabrication")),
            exaggeration=bool(result.get("exaggeration")),
            safety_violation=bool(result.get("safety_violation")),
            out_of_domain=bool(result.get("out_of_domain")),
            unhelpful=bool(result.get("unhelpful")),
            reason=reason,
        )
    return GuardResult(passed=True)


async def guard_generation(
    llm: Any,
    *,
    generate: Callable[[str, str, str], Awaitable[str]],
    query: str,
    context_text: str = "",
    config: GuardConfig | None = None,
    enforce_numbers: bool = False,
) -> tuple[str, GuardResult, int]:
    """“生成 → 评审”循环：不通过则带上未通过原因重新生成，最多重试 max_retries 次。

    返回 ``(最终回答, 最后一次评审结果, 实际尝试次数)``。
    重试原因 guard_reason 会回传给 generate，作为后续生成的额外指令。
    """
    cfg = config or GuardConfig()
    attempts = 0
    guard_reason = ""
    response = ""
    result = GuardResult(passed=False, reason="not_attempted")

    while attempts <= cfg.max_retries:
        response = await generate(query, context_text, guard_reason)
        result = await evaluate_guard(
            llm,
            response,
            query=query,
            context_text=context_text,
            config=cfg,
            enforce_numbers=enforce_numbers,
        )
        attempts += 1
        if result.passed:
            break
        guard_reason = result.reason
    return response, result, attempts