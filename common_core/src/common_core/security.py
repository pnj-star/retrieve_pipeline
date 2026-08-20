"""供全部 Skill 共用的、结果可确定的输入安全基础能力。
本模块不含业务逻辑：敏感词、屏蔽文案、领域专属规则都由各个 Skill 实例自行提供。
该模块只负责文本标准化、识别常见提示注入模式、以及对通用个人隐私信息 (PII) 做掩码脱敏。
"""

from __future__ import annotations

import re
from typing import Any, Iterable


_INVISIBLE_CHARS = re.compile(
    "[\u200B-\u200F\u202A-\u202E\u2060-\u2064\uFEFF\u00AD\u061C]"
)

_EMOJI_CHARS = re.compile(
    r"["
    r"\U0001f000-\U0001faff"
    r"\U0001f1e6-\U0001f1ff"
    r"\U00002600-\U000027ff"
    r"\U00002b00-\U00002bff"
    r"\U00002190-\U000021ff"
    r"\U0000fe00-\U0000fe0f"
    r"\u200d"
    r"\u20e3"
    r"]+"
)

_MULTI_SPACE = re.compile(r"\s+")

_EMAIL_RE = re.compile(r"(?<![\w.+-])[\w.+-]+@[\w-]+(?:\.[\w-]+)+(?![\w])")
_ID_RE = re.compile(
    r"(?<!\d)\d{6}(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx](?!\d)"
)
_BANK_RE = re.compile(r"(?<!\d)(?:622[0-9]{13,16}|621[0-9]{13,16}|4[0-9]{15,18}|5[0-9]{15,18})(?!\d)")
_PHONE_SPLIT_RE = re.compile(r"(?<!\d)(?:1[3-9]\d-\d{4}-\d{4}|1[3-9]\d \d{4} \d{4})(?!\d)")
_PHONE_RE = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")


def normalize_query(text: str) -> str:
    """用户输入标准化：剔除不可见控制字符、emoji 表情，以及多余空白字符。

    参数:
        text: 待标准化的原始输入。

    返回:
        清理并去除首尾空白后的文本。
    """
    text = _INVISIBLE_CHARS.sub("", text)
    text = _EMOJI_CHARS.sub("", text)
    text = text.replace("\u3000", " ")
    return _MULTI_SPACE.sub(" ", text).strip()


def mask_pii(text: str) -> str:
    """在送入日志或者大模型之前，将常见的个人身份信息 (PII) 模式替换为占位符。

    支持脱敏的类别：邮箱、身份证号、银行卡号、手机号。

    参数:
        text: 待脱敏的原始文本。

    返回:
        将各类 PII 替换为对应占位符后的文本。
    """
    text = _EMAIL_RE.sub("<EMAIL>", text)
    text = _ID_RE.sub("<ID_NUMBER>", text)
    text = _BANK_RE.sub("<BANK_CARD>", text)
    text = _PHONE_SPLIT_RE.sub("<PHONE>", text)
    text = _PHONE_RE.sub("<PHONE>", text)
    return text


INJECTION_PATTERNS: tuple[str, ...] = (
    r"ignore\s+(all\s+|any\s+|every\s+)?(system|instructions|rules|prompt|settings)",
    r"disregard\s+(all\s+|any\s+|every\s+)?(system|instructions|rules|prompt|settings)",
    r"(do\s+not|never|don't)\s+(follow|obey|respect)\s+(system|instructions|rules|prompt)",
    r"your\s+(system|initial|underlying|default)\s+(prompt|instructions|rules|settings)",
    r"(system|role|assistant)\s*[:：]",
    r"(jailbreak|break out of system prompt)",
    r"bypass\s+(the\s+)?(review|limit|system|rules|safety|filter)",
    r"show\s+(me\s+)?(your|the)\s+(system\s+)?(prompt|instructions|rules)",
)


def check_safety(
    raw_query: str,
    sensitive_words: Iterable[str] = (),
    injection_patterns: Iterable[str] = INJECTION_PATTERNS,
) -> dict[str, Any]:
    """执行确定性安全校验，并返回一份经过标准化处理的工作副本文本。

    参数:
        raw_query: 用户的原始输入。
        sensitive_words: 敏感词集合；命中任意一个即拦截。
        injection_patterns: 提示注入正则模式集合，默认使用 INJECTION_PATTERNS。

    返回:
        字典，包含：
            query: 标准化之后的文本；
            blocked: 该输入是否应当被拒绝拦截；
            reason: 被拦截时的机器可读原因码，未拦截时为空字符串。
    """
    query = normalize_query(raw_query)
    if not query:
        return {"query": "", "blocked": False, "reason": ""}

    for keyword in sensitive_words:
        if keyword in query:
            return {
                "query": query,
                "blocked": True,
                "reason": f"sensitive_word:{keyword}",
            }

    for pattern in injection_patterns:
        if re.search(pattern, query, re.IGNORECASE):
            return {
                "query": query,
                "blocked": True,
                "reason": f"injection:{pattern}",
            }

    return {"query": mask_pii(query), "blocked": False, "reason": ""}
