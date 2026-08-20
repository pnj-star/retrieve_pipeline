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

# 默认的绝对化 / 夸大用语黑名单，刻意保守（fail-closed）。
# 电商等话题场景可经 GuardConfig(absolute_words=...) 覆盖，避免拦截正常的营销话术。
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
DEFAULT_REVIEW_PROMPT = """你是一名输出质量审核人员。检查助手回复是否合格。
    输出包含以下键的 JSON：pass（布尔值），issues（字符串列表），fabrication（布尔值），exaggeration（布尔值），
    safety_violation（布尔值），out_of_domain（布尔值），unhelpful（布尔值）。
    检查标准：
    ‑ fabrication（虚构事实）：回复编造了参考信息中不存在的具体事实
    ‑ exaggeration（夸大表述）：回复使用 “最佳”“100%”“绝对” 这类绝对化表述，或是承诺可以治愈疾病
    ‑ safety_violation（安全违规）：回复给出不安全的建议
    ‑ out_of_domain（超出领域）：回复偏离助手预设的业务范围
    ‑ unhelpful（无帮助）：回复没有提供实际帮助（合理转接人工客服的情况不算此项）
    如实说明信息缺失并建议联系人工的回复，不判定为无帮助。 """


def _norm_num(value: str) -> str:
    """归一化数字字符串：去掉小数末尾的 0 与小数点（15% 与 0.15 视为不同数字）。

    参数:
        value: 原始数字字符串。

    返回:
        去掉尾部冗余 0 与小数点后的字符串。
    """
    if "." in value:
        value = value.rstrip("0").rstrip(".")
    return value


def extract_risky_numbers(text: str) -> list[str]:
    """提取风险数字：小数与带单位量词的数字（费率、金额、数量等）。

    参数:
        text: 待检查的文本（通常是回答）。

    返回:
        归一化后的风险数字字符串列表。
    """
    return [_norm_num(match.group(0)) for match in _RISKY_NUM_RE.finditer(text or "")]


def extract_all_numbers(text: str) -> set[str]:
    """提取参考文本中所有归一化后的数字（用于数字溯源比对）。

    参数:
        text: 待提取的文本（通常是参考上下文）。

    返回:
        所有归一化数字组成的集合。
    """
    return {
        _norm_num(match.group(0))
        for match in _ALL_NUM_RE.finditer(text or "")
    }


def check_compound_numbers(response: str, context_text: str) -> list[str]:
    """数字溯源检查：回答中的风险数字必须同时出现在参考上下文里。

    防止模型编造出上下文里不存在的费率 / 金额 / 数量等关键数字。

    参数:
        response: 待检查的回答文本。
        context_text: 参考上下文文本，作为数字白名单来源。

    返回:
        回答中出现但参考上下文里没有的风险数字问题列表。
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
    """检查回答中是否出现绝对化 / 夸大用语，返回命中的问题列表。

    参数:
        response: 待检查的回答文本。
        absolute_words: 绝对化 / 夸大用语黑名单；默认用内置词表。

    返回:
        命中的绝对化用语问题列表；无命中时为空列表。
    """
    return [
        f"absolute language: {word}"
        for word in absolute_words
        if word and word in (response or "")
    ]


@dataclass(slots=True)
class GuardConfig:
    """护栏配置：词表、重试次数与评审提示词。

    字段:
        absolute_words: 绝对化 / 夸大用语黑名单。
        max_retries: 生成失败后最多重试次数。
        review_prompt: 评审 LLM 使用的提示词。
        temperature: 评审 LLM 的采样温度，使用低温度保证判断稳定。
    """

    absolute_words: tuple[str, ...] = DEFAULT_ABSOLUTE_WORDS
    max_retries: int = 2  # 生成失败后最多重试次数
    review_prompt: str = DEFAULT_REVIEW_PROMPT
    temperature: float = 0.0  # 评审 LLM 使用低温度，保证判断稳定


@dataclass(slots=True)
class GuardResult:
    """单次评审结果：是否通过 + 各类问题标记 + 原因。

    字段:
        passed: 本次评审是否通过。
        issues: 命中问题的描述列表。
        fabrication: 是否编造事实。
        exaggeration: 是否夸大 / 使用绝对化用语。
        safety_violation: 是否安全违规。
        out_of_domain: 是否跑题 / 越界。
        unhelpful: 是否没有实际帮助。
        reason: 未通过时的原因说明。
    """

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

    参数:
        llm: 底层 LLM 客户端，提供 chat_json 供评审。
        response: 待评审的回答文本。
        query: 原始用户查询，用于上下文；默认空字符串。
        context_text: 参考上下文，供数字溯源检查使用。
        config: 护栏配置；None 用默认 GuardConfig()。
        enforce_numbers: 是否启用数字溯源检查，默认 False。

    返回:
        本次评审结果 GuardResult。
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

    参数:
        llm: 底层 LLM 客户端，用于评审。
        generate: 生成函数，签名 (query, context_text, guard_reason) -> Awaitable[str]。
        query: 用户查询文本。
        context_text: 参考上下文文本。
        config: 护栏配置；None 用默认 GuardConfig()。
        enforce_numbers: 是否启用数字溯源检查，默认 False。

    返回:
        (最终回答, 最后一次评审结果, 实际尝试次数) 三元组。
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
