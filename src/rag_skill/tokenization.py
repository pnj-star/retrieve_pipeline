"""token 计数封装：优先使用 tiktoken，缺失时回退字符计数。"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any, Callable

logger = logging.getLogger(__name__)

DEFAULT_TIKTOKEN_ENCODING = "cl100k_base"


def make_token_counter(encoding: Any) -> Callable[[str], int]:
    """用已加载的 tiktoken encoding 构造 ``str -> token 数`` 的计数器。

    参数:
        encoding: 已加载的 tiktoken 编码对象，用于把文本切成 token。

    返回:
        接收文本、返回 token 数的函数。
    """
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

    参数:
        model: 可选的模型名，用于选择对应的 tiktoken 编码。
        fallback: 可选的备用计数函数；tiktoken 不可用时调用它。默认按字符数。

    返回:
        可注入 ``count_tokens`` 的计数函数。
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
