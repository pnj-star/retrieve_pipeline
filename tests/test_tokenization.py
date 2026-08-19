"""token 计数封装的单元测试。"""

from __future__ import annotations

import sys

from rag_skill.tokenization import build_token_counter, make_token_counter


class FakeEncoding:
    def __init__(self, token_count: int) -> None:
        self.token_count = token_count

    def encode(self, text: str) -> list[int]:
        assert isinstance(text, str)
        return list(range(self.token_count))


def test_make_token_counter_counts_encoded_items() -> None:
    counter = make_token_counter(FakeEncoding(token_count=4))
    assert counter("anything") == 4


def test_build_token_counter_uses_tiktoken_when_available() -> None:
    try:
        import tiktoken  # noqa: F401
    except ImportError:
        return

    counter = build_token_counter()
    assert callable(counter)
    assert counter("hello world") > 0


def test_build_token_counter_falls_back_without_tiktoken(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "tiktoken", None)
    counter = build_token_counter(fallback=lambda text: len(text) * 2)
    assert counter("abcd") == 8
