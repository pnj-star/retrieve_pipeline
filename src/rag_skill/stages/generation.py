"""基于 LLM 的回答生成：可配置提示词与兜底（fallback）。

支持普通对话与流式两种模式；LLM 调用失败时可回退到中性兜底文案，
或按调用方要求把错误向上抛出（作为硬失败）。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, AsyncIterator

logger = logging.getLogger(__name__)

# 默认生成提示词：只依据参考信息回答，不编造事实，输出纯文本。
# 必须包含 {context} 占位符；{query} 可选。
DEFAULT_PROMPT_TEMPLATE = """
仅依据下文参考信息回答用户问题，不得虚构具体事实，使用纯文本，不使用 markdown 格式。
参考信息:{context}"""

# 无参考信息时的兜底提示词：诚实说明没有相关信息，并在涉及具体产品/服务时建议联系客服。
DEFAULT_FALLBACK_PROMPT_TEMPLATE = """
未查询到该用户问题对应的参考信息，请如实回复暂无相关具体信息，如果问题涉及特定产品或服务，建议联系人工支持。
用户问题:{query}"""

# LLM 调用失败时的中性兜底回答。
DEFAULT_FALLBACK_RESPONSE = (
    "很抱歉，我目前无法生成回答。 "
    "请稍后重试或调用其他工具或联系人工支持。"
)


@dataclass(slots=True)
class GenerationConfig:
    """生成配置：提示词模板与生成参数。

    字段:
        prompt_template: 有上下文时使用的模板，必须含 {context} 占位符。
        fallback_prompt_template: 无上下文时使用的模板。
        fallback_response: LLM 失败时返回的中性兜底文案。
        temperature: LLM 采样温度；None 用模型默认。
        max_tokens: 生成最大 token 数；None 用模型默认。
    """

    prompt_template: str = DEFAULT_PROMPT_TEMPLATE  # 有上下文时使用的模板
    fallback_prompt_template: str = DEFAULT_FALLBACK_PROMPT_TEMPLATE  # 无上下文时使用的模板
    fallback_response: str = DEFAULT_FALLBACK_RESPONSE  # LLM 失败时返回的文案
    temperature: float | None = None
    max_tokens: int | None = None


def build_messages(
    query: str,
    context_text: str,
    config: GenerationConfig,
    *,
    extra_prompt: str = "",
) -> tuple[list[dict[str, str]], str]:
    """构造 ``(messages, system_prompt)``。

    有上下文时用主模板，无上下文时用兜底模板；extra_prompt（如系统指令、
    护栏重试原因）会追加到 system_prompt 末尾。

    参数:
        query: 用户查询文本。
        context_text: 已拼好的上下文文本；为空时走兜底模板。
        config: 生成配置，决定使用哪个模板。
        extra_prompt: 追加到 system_prompt 末尾的额外指令；默认空字符串。

    返回:
        (messages, system_prompt) 二元组，可直接交给 LLM 调用。
    """
    has_context = bool(context_text and context_text.strip())
    if has_context:
        system_prompt = config.prompt_template.format(
            context=context_text,
            query=query,
        )
    else:
        system_prompt = config.fallback_prompt_template.format(
            query=query,
            context=context_text,
        )
    if extra_prompt:
        system_prompt = f"{system_prompt}\n\n{extra_prompt}"
    return [{"role": "user", "content": query}], system_prompt


async def generate_answer(
    llm: Any,
    query: str,
    *,
    context_text: str = "",
    config: GenerationConfig | None = None,
    extra_prompt: str = "",
    stream: bool = False,
    raise_on_error: bool = False,
) -> str:
    """生成回答（流式或非流式），失败时返回中性兜底文案。

    - raise_on_error=True：调用方（如管线）需要把 LLM 失败当作硬错误处理；
      默认为 False，通用技能路径保留软兜底。

    参数:
        llm: 底层 LLM 客户端，提供 chat / stream_chat。
        query: 用户查询文本。
        context_text: 已拼好的上下文文本；可为空。
        config: 生成配置；None 用默认 GenerationConfig()。
        extra_prompt: 追加到 system_prompt 的额外指令。
        stream: 是否启用流式生成，默认 False。
        raise_on_error: LLM 失败时是否向上抛异常，默认 False（软兜底）。

    返回:
        生成的回答文本；失败且未抛错时返回配置的中性兜底文案。
    """
    cfg = config or GenerationConfig()
    messages, system_prompt = build_messages(
        query,
        context_text,
        cfg,
        extra_prompt=extra_prompt,
    )
    try:
        if stream:
            # 流式：逐块累加后合并
            chunks: list[str] = []
            async for chunk in stream_answer(
                llm,
                messages,
                system_prompt,
                cfg,
            ):
                chunks.append(chunk)
            response = "".join(chunks).strip()
        else:
            response = await llm.chat(
                messages,
                system_prompt=system_prompt,
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
            )
            response = str(response or "").strip()
        return response or cfg.fallback_response
    except Exception:
        if raise_on_error:
            raise
        logger.warning(
            "大模型LLM 生成失败，启用中性兜底回复",
            exc_info=True,
        )
        return cfg.fallback_response


async def stream_answer(
    llm: Any,
    messages: list[dict[str, str]],
    system_prompt: str,
    config: GenerationConfig,
) -> AsyncIterator[str]:
    """从底层 LLM 逐块产出文本。

    参数:
        llm: 底层 LLM 客户端，提供 stream_chat。
        messages: 对话消息列表（通常只有一条用户消息）。
        system_prompt: 已构造好的系统提示词。
        config: 生成配置，提供 temperature / max_tokens。

    返回:
        逐块产出文本字符串的异步迭代器。
    """
    async for chunk in llm.stream_chat(
        messages,
        system_prompt=system_prompt,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    ):
        yield str(chunk)
