"""QueryRewriter 契约测试：策略选择、作用域覆盖与失败回退。"""

from __future__ import annotations

import asyncio

import pytest

from common_core.config import QueryRewriteConfig
from common_core.context import AgentContext
from rag_skill.stages import QueryRewriteResult, QueryRewriter
from rag_skill.stages.query_rewrite import normalize_query_rewrite_mode


class FakeLLM:
    def __init__(self, response: str = "", *, fail: bool = False) -> None:
        self.response = response
        self.fail = fail

    async def chat(self, messages, *, system_prompt: str = "", **kwargs) -> str:
        if self.fail:
            raise RuntimeError("llm down")
        return self.response


def make_context(tenant_id: str = "t1", kb_id: str = "kb1") -> AgentContext:
    return AgentContext(
        tenant_id=tenant_id,
        kb_id=kb_id,
        session_id="s1",
        request_id="r1",
        user_id="u1",
    )


def run(coro):
    return asyncio.run(coro)


def test_normalize_mode_defaults_to_off() -> None:
    assert normalize_query_rewrite_mode(None) == "off"
    assert normalize_query_rewrite_mode("LLM_REWRITE") == "llm_rewrite"
    assert normalize_query_rewrite_mode("bogus") == "off"


def test_off_and_identity_return_original_query() -> None:
    for mode in ("off", "identity"):
        rewriter = QueryRewriter(FakeLLM(), QueryRewriteConfig(mode=mode))
        result = run(rewriter.rewrite("  帮我查一下退款政策  ", make_context()))
        assert result.mode == mode
        assert result.rewritten_query == "  帮我查一下退款政策  "
        assert result.query_variants == ["  帮我查一下退款政策  "]


def test_llm_rewrite_uses_first_line_of_llm_output() -> None:
    llm = FakeLLM("退款的申请流程和条件")
    rewriter = QueryRewriter(llm, QueryRewriteConfig(mode="llm_rewrite"))
    result = run(
        rewriter.rewrite("我该怎么退这个订单的钱？", make_context(), mode="llm_rewrite")
    )
    assert result.mode == "llm_rewrite"
    assert result.rewritten_query == "退款的申请流程和条件"
    assert llm.fail is False


def test_llm_rewrite_falls_back_to_original_on_error() -> None:
    rewriter = QueryRewriter(
        FakeLLM(fail=True),
        QueryRewriteConfig(mode="llm_rewrite"),
    )
    result = run(rewriter.rewrite("原始问题", make_context(), mode="llm_rewrite"))
    assert result.rewritten_query == "原始问题"
    assert result.query_variants == ["原始问题"]
    assert result.error


def test_llm_rewrite_unchanged_output_falls_back() -> None:
    rewriter = QueryRewriter(
        FakeLLM("原始问题"),
        QueryRewriteConfig(mode="llm_rewrite"),
    )
    result = run(rewriter.rewrite("原始问题", make_context(), mode="llm_rewrite"))
    assert result.rewritten_query == "原始问题"
    assert result.error == "rewrite_unchanged"


def test_query_expansion_returns_original_plus_variants() -> None:
    rewriter = QueryRewriter(
        FakeLLM("按订单编号查退款\n查看退货页面"),
        QueryRewriteConfig(mode="query_expansion", expand_count=2),
    )
    result = run(
        rewriter.rewrite("怎么查退款？", make_context(), mode="query_expansion")
    )
    assert result.mode == "query_expansion"
    assert result.rewritten_query == "按订单编号查退款"
    assert result.query_variants == ["怎么查退款？", "按订单编号查退款", "查看退货页面"]


def test_query_expansion_falls_back_when_no_variants() -> None:
    rewriter = QueryRewriter(
        FakeLLM("怎么查退款？"),
        QueryRewriteConfig(mode="query_expansion"),
    )
    result = run(
        rewriter.rewrite("怎么查退款？", make_context(), mode="query_expansion")
    )
    assert result.rewritten_query == "怎么查退款？"
    assert result.query_variants == ["怎么查退款？"]
    assert result.error == "no_expansions"


def test_scoped_mode_overrides_default() -> None:
    rewriter = QueryRewriter(
        FakeLLM(),
        QueryRewriteConfig(
            mode="off",
            scoped_modes={"t1/kb1": "identity", "t2/*": "query_expansion"},
        ),
    )
    assert rewriter.resolve_mode(make_context("t1", "kb1")) == "identity"
    assert rewriter.resolve_mode(make_context("t2", "kb9")) == "query_expansion"
    # 未匹配作用域时回退默认 off
    assert rewriter.resolve_mode(make_context("t9", "kb9")) == "off"


def test_raw_model_router_works_in_async_context() -> None:
    """确保 QueryRewriter 可直接用于异步上下文，返回稳定契约。"""
    rewriter = QueryRewriter(FakeLLM("改写后的问题"), QueryRewriteConfig())
    result = run(rewriter.rewrite("原始问题", make_context(), mode="llm_rewrite"))
    assert isinstance(result, QueryRewriteResult)
    assert result.rewritten_query == "改写后的问题"
