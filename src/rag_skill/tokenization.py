"""token 计数封装：优先使用 tiktoken，缺失时回退字符计数。"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any, Callable

logger = logging.getLogger(__name__)

DEFAULT_TIKTOKEN_ENCODING = "cl100k_base"


def make_token_counter(encoding: Any) -> Callable[[str], int]:
    """用已加载的 tiktoken encoding 构造 ``str -> token 数`` 的计数器。"""
    return lambda text: len(encoding.encode(text))


@lru_cache(maxsize=8)
def build_token_counter(
    model: str | None = None,
    *,
    fallback: Callable[[str], int] | None = None,
) -> Callable[[str], int]:
    """返回可注入 ``count_tokens`` 的计数函数。

    - model 传入时按模型名选编码（例如 ``gpt-4o``）；
    - 未传 model 时使用 ``cl100k_base``；
    - 未安装 tiktoken 或编码不可用时，回退到 fallback（默认按字符数）。
    """
    counter = fallback or len
    try:
        import tiktoken
    except ImportError:
        logger.warning(
            "tiktoken is not installed; token counter falls back to character count"
        )
        return counter
    try:
        if model:
            encoding = tiktoken.encoding_for_model(model)
        else:
            encoding = tiktoken.get_encoding(DEFAULT_TIKTOKEN_ENCODING)
    except Exception:
        logger.warning(
            "tiktoken encoding unavailable for model=%r; "
            "token counter falls back to character count",
            model,
            exc_info=True,
        )
        return counter
    return make_token_counter(encoding)


__all__ = ["build_token_counter", "make_token_counter"]
