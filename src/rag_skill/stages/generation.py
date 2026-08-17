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
DEFAULT_PROMPT_TEMPLATE = """Answer the user's question based only on the reference information below. Do not fabricate specific facts. Use plain text without markdown formatting.

Reference information:
{context}"""

# 无参考信息时的兜底提示词：诚实说明没有相关信息，并在涉及具体产品/服务时建议联系客服。
DEFAULT_FALLBACK_PROMPT_TEMPLATE = """No reference information was found for the user's question. Answer honestly that no specific information is available, and suggest contacting support when the question concerns a specific product or service.

User question:
{query}"""

# LLM 调用失败时的中性兜底回答。
DEFAULT_FALLBACK_RESPONSE = (
    "Sorry, I cannot generate an answer right now. "
    "Please try again later or contact support."
)


@dataclass(slots=True)
class GenerationConfig:
    """生成配置：提示词模板与生成参数。"""

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
            "LLM generation failed, using neutral fallback",
            exc_info=True,
        )
        return cfg.fallback_response


async def stream_answer(
    llm: Any,
    messages: list[dict[str, str]],
    system_prompt: str,
    config: GenerationConfig,
) -> AsyncIterator[str]:
    """从底层 LLM 逐块产出文本。"""
    async for chunk in llm.stream_chat(
        messages,
        system_prompt=system_prompt,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    ):
        yield str(chunk)