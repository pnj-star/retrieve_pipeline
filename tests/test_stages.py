"""各阶段（stages）单元测试：覆盖重排、上下文组装、生成、护栏与
响应缓存的契约行为，使用 fake LLM / 缓存 / 存储。"""

import asyncio

from common_core.context import AgentContext
from rag_skill.stages import (
    GenerationConfig,
    GuardConfig,
    Reranker,
    ResponseCache,
    build_context_text,
    check_compound_numbers,
    clean_markdown,
    dedupe_docs,
    evaluate_guard,
    extract_images,
    generate_answer,
    guard_generation,
    judge_relevance,
    rank_docs,
)


class FakeLLM:
    def __init__(
        self,
        response: str = "generic answer",
        reviews: list[dict] | None = None,
        fail_chat: bool = False,
    ) -> None:
        self.response = response
        self.review_results = list(reviews) if reviews else [{"pass": True}]
        self.fail_chat = fail_chat
        self.chat_calls: list[tuple[list[dict], str]] = []
        self.json_calls: list[tuple[list[dict], str]] = []

    async def chat(
        self,
        messages: list[dict],
        *,
        system_prompt: str = "",
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict | None = None,
    ) -> str:
        self.chat_calls.append((messages, system_prompt))
        if self.fail_chat:
            raise RuntimeError("llm down")
        return self.response

    async def chat_json(
        self,
        messages: list[dict],
        *,
        system_prompt: str = "",
        temperature: float | None = None,
    ) -> dict:
        self.json_calls.append((messages, system_prompt))
        return self.review_results.pop(0) if self.review_results else {"pass": True}

    async def stream_chat(
        self,
        messages: list[dict],
        *,
        system_prompt: str = "",
        temperature: float | None = None,
        max_tokens: int | None = None,
    ):
        for chunk in ("part1 ", "part2"):
            yield chunk


class FakeCache:
    def __init__(self) -> None:
        self.store: dict[str, str] = {}

    def key(
        self,
        scope: str,
        material: str,
        *,
        tenant_id: str = "",
        kb_id: str = "",
    ) -> str:
        return f"{tenant_id}:{kb_id}:{scope}:{material}"

    def get(
        self,
        scope: str,
        material: str,
        *,
        tenant_id: str = "",
        kb_id: str = "",
    ) -> str | None:
        return self.store.get(self.key(scope, material, tenant_id=tenant_id, kb_id=kb_id))

    def set(
        self,
        scope: str,
        material: str,
        value: str,
        *,
        ttl: int | None = None,
        tenant_id: str = "",
        kb_id: str = "",
    ) -> bool:
        self.store[self.key(scope, material, tenant_id=tenant_id, kb_id=kb_id)] = value
        return True

    def delete(
        self,
        scope: str,
        material: str,
        *,
        tenant_id: str = "",
        kb_id: str = "",
    ) -> bool:
        key = self.key(scope, material, tenant_id=tenant_id, kb_id=kb_id)
        return self.store.pop(key, None) is not None
class FakeMetrics:
    def __init__(self) -> None:
        self.cache_events: list[tuple[str, str, str]] = []

    def record_cache(
        self,
        result: str,
        tenant_id: str = "",
        kb_id: str = "",
    ) -> None:
        self.cache_events.append((result, tenant_id, kb_id))


def test_rank_docs_sorts_and_truncates() -> None:
    docs = [
        {"id": "a", "content": "first"},
        {"id": "b", "content": "second"},
    ]
    ranked = rank_docs(docs, "query", scores=[0.5, 0.9], top_k=1)
    assert len(ranked) == 1
    assert ranked[0]["id"] == "b"
    assert ranked[0]["ce_score"] == 0.9


def test_rank_docs_passthrough_without_query() -> None:
    docs = [{"id": "a"}, {"id": "b"}]
    ranked = rank_docs(docs, "", top_k=1)
    assert ranked == [docs[0]]
    assert "ce_score" not in ranked[0]


def test_judge_relevance_gate() -> None:
    assert judge_relevance([], 0.15) is True
    assert judge_relevance([{"ce_score": 0.1}], 0.15) is True
    assert judge_relevance([{"ce_score": 0.2}], 0.15) is False


def test_reranker_uses_injected_score_fn() -> None:
    reranker = Reranker(top_k=2, score_fn=lambda q, contents: [0.2, 0.9])
    ranked = reranker.rank("q", [{"id": "a"}, {"id": "b"}])
    assert ranked[0]["id"] == "b"
    assert ranked[0]["ce_score"] == 0.9


def test_clean_markdown_strips_markers() -> None:
    assert clean_markdown("**bold** and `code`\n# title") == "bold and code\ntitle"


def test_build_context_text_dedupes_and_truncates() -> None:
    docs = [
        {"parent_title": "t1", "parent_content": "A" * 100, "source": "kb.md"},
        {"parent_title": "t1", "parent_content": "B" * 100, "source": "kb.md"},
        {"parent_title": "t2", "parent_content": "C" * 100, "source": "kb2.md"},
    ]
    context, sources = build_context_text(docs, max_chars=250)
    assert "[Source 1]" in context
    assert "[Source 2]" in context
    assert "B" not in context
    assert len(sources) == 2


def test_build_context_text_truncates_first_doc_instead_of_empty_context() -> None:
    docs = [{"parent_content": "A" * 9000, "source": "kb.md"}]
    context, sources = build_context_text(docs, max_chars=120, max_doc_chars=8000)
    assert context
    assert len(context) <= 120
    assert sources == ["kb.md"]


def test_build_context_text_counts_prefix_and_separators_in_budget() -> None:
    docs = [{"content": "a" * 100}, {"content": "b" * 100}]
    context, sources = build_context_text(
        docs,
        max_chars=300,
        prefix_blocks=["P" * 80],
        max_doc_chars=200,
    )
    assert len(context) <= 300
    assert sources == ["", "", ""]


def test_build_context_text_max_doc_chars_keeps_other_sources() -> None:
    docs = [
        {"parent_content": "A" * 2000, "source": "a"},
        {"parent_content": "B" * 100, "source": "b"},
    ]
    context, sources = build_context_text(
        docs,
        max_chars=600,
        max_doc_chars=200,
        format_doc=lambda index, doc: doc.get("parent_content", ""),
    )
    assert "B" in context
    assert sources == ["a", "b"]
    assert len(context) <= 600


def test_build_context_text_token_budget_with_counter() -> None:
    def fake_tokens(text: str) -> int:
        return len(text)

    context, sources = build_context_text(
        [{"content": "a" * 100}, {"content": "b" * 100}],
        prefix_blocks=["P" * 20],
        max_tokens=280,
        count_tokens=fake_tokens,
        max_doc_chars=200,
    )
    assert len(context) <= 280
    assert len(sources) == 3


def test_build_context_text_caps_oversized_prefix() -> None:
    context, sources = build_context_text(
        [{"content": "x"}],
        max_chars=30,
        prefix_blocks=["P" * 100],
        max_doc_chars=100,
    )
    assert len(context) <= 30
    assert sources == [""]


def test_dedupe_docs_prefers_parent_id_over_title() -> None:
    docs = [
        {"parent_id": "p1", "parent_title": "same", "content": "A"},
        {"parent_id": "p1", "parent_title": "same", "content": "B"},
    ]
    deduped = dedupe_docs(docs)
    assert len(deduped) == 1
    assert "A" in deduped[0]["content"]


def test_dedupe_docs_keeps_same_title_with_different_parent_id() -> None:
    docs = [
        {"parent_id": "p1", "parent_title": "政策 A", "content": "第一份"},
        {"parent_id": "p2", "parent_title": "政策 A", "content": "第二份"},
    ]
    deduped = dedupe_docs(docs)
    assert len(deduped) == 2
    assert [doc["parent_id"] for doc in deduped] == ["p1", "p2"]


def test_dedupe_docs_falls_back_to_source_and_title() -> None:
    docs = [
        {"source": "a.md", "parent_title": "常见问题", "content": "A"},
        {"source": "a.md", "parent_title": "常见问题", "content": "B"},
        {"source": "b.md", "parent_title": "常见问题", "content": "C"},
    ]
    deduped = dedupe_docs(docs)
    assert [doc["source"] for doc in deduped] == ["a.md", "b.md"]
    assert [doc["content"] for doc in deduped] == ["A", "C"]


def test_dedupe_docs_explicit_key_takes_priority() -> None:
    docs = [
        {"parent_id": "p1", "category": "x", "content": "A"},
        {"parent_id": "p2", "category": "x", "content": "B"},
    ]
    deduped = dedupe_docs(docs, dedupe_key=lambda doc: doc["category"])
    assert len(deduped) == 1
    assert deduped[0]["content"] == "A"


def test_build_context_text_max_doc_tokens_limits_single_doc() -> None:
    docs = [
        {"content": "A" * 200, "source": "a"},
        {"content": "B" * 200, "source": "b"},
    ]
    context, sources = build_context_text(
        docs,
        max_tokens=400,
        count_tokens=len,
        max_doc_tokens=100,
    )
    assert len(context) <= 400
    assert len(sources) == 2
    for block in context.split("\n\n---\n\n"):
        assert len(block) <= 100
    assert " [truncated]" in context


def test_build_context_text_merges_child_chunks_in_chunk_index_order() -> None:
    docs = [
        {
            "parent_id": "p1",
            "parent_title": "退款规则",
            "content": "末尾",
            "chunk_index": 2,
            "source": "kb.md",
        },
        {
            "parent_id": "p1",
            "parent_title": "退款规则",
            "content": "开头",
            "chunk_index": 1,
            "source": "kb.md",
        },
    ]
    context, sources = build_context_text(docs, max_chars=200)
    assert context.index("开头") < context.index("末尾")
    assert "（退款规则）" in context
    assert sources == ["kb.md"]


def test_build_context_text_uses_parent_content_when_present() -> None:
    docs = [
        {"parent_id": "p1", "parent_content": "完整父内容", "content": "子内容"},
        {"parent_id": "p1", "content": "另一个子块"},
    ]
    context, sources = build_context_text(docs, max_chars=200)
    assert "完整父内容" in context
    assert "子内容" not in context
    assert sources == [""]


def test_build_context_text_dedupes_duplicate_child_chunks() -> None:
    docs = [
        {"parent_id": "p1", "parent_title": "规则", "content": "相同块", "chunk_index": 1},
        {"parent_id": "p1", "parent_title": "规则", "content": "相同块", "chunk_index": 2},
        {"parent_id": "p1", "parent_title": "规则", "content": "不同块", "chunk_index": 3},
    ]
    context, sources = build_context_text(docs, max_chars=200)
    assert context.count("相同块") == 1
    assert "不同块" in context
    assert sources == [""]


def test_extract_images_dedupes() -> None:
    images = [{"image_url": "img1"}, {"image_url": "img1"}, {"url": "img2"}]
    docs = [{"image_urls": ["img3", "img1"]}]
    assert extract_images(docs, images, max_images=2) == ["img1", "img2"]


def test_generate_answer_uses_prompt_template() -> None:
    llm = FakeLLM(response="ok answer")
    answer = asyncio.run(
        generate_answer(
            llm,
            "question",
            context_text="ctx",
            config=GenerationConfig(prompt_template="Use {context}"),
        )
    )
    assert answer == "ok answer"
    assert llm.chat_calls[0][1] == "Use ctx"


def test_generate_answer_fallback_without_context() -> None:
    llm = FakeLLM(response="fallback text")
    cfg = GenerationConfig(
        fallback_prompt_template="Q: {query}",
        fallback_response="neutral",
    )
    answer = asyncio.run(generate_answer(llm, "q", config=cfg))
    assert answer == "fallback text"
    assert "Q: q" in llm.chat_calls[0][1]


def test_generate_answer_neutral_when_llm_fails() -> None:
    llm = FakeLLM(fail_chat=True)
    answer = asyncio.run(
        generate_answer(
            llm,
            "q",
            config=GenerationConfig(fallback_response="neutral"),
        )
    )
    assert answer == "neutral"


def test_generate_answer_aggregates_stream() -> None:
    llm = FakeLLM()
    answer = asyncio.run(
        generate_answer(llm, "q", config=GenerationConfig(), stream=True)
    )
    assert answer == "part1 part2"


def test_absolute_language_fails_deterministically() -> None:
    result = asyncio.run(
        evaluate_guard(
            FakeLLM(),
            "this is absolutely the best",
            config=GuardConfig(absolute_words=("absolutely",)),
        )
    )
    assert not result.passed
    assert result.exaggeration


def test_compound_number_check_blocks_conversion() -> None:
    llm = FakeLLM()
    ok = asyncio.run(
        evaluate_guard(
            llm,
            "the rate is 15%",
            context_text="authoritative rate is 15%",
            enforce_numbers=True,
        )
    )
    assert ok.passed
    bad = asyncio.run(
        evaluate_guard(
            llm,
            "the rate is 0.15",
            context_text="authoritative rate is 15%",
            enforce_numbers=True,
        )
    )
    assert not bad.passed
    assert bad.fabrication


def test_check_compound_numbers_returns_issues() -> None:
    issues = check_compound_numbers("0.15 and 10%", "10%")
    assert any("0.15" in issue for issue in issues)


def test_guard_generation_retries_until_pass() -> None:
    llm = FakeLLM(
        reviews=[
            {"pass": False, "issues": ["vague"]},
            {"pass": False, "issues": ["too short"]},
            {"pass": True},
        ]
    )
    seen_reasons: list[str] = []

    async def generate(query: str, context_text: str, guard_reason: str) -> str:
        seen_reasons.append(guard_reason)
        return f"answer {len(seen_reasons)}"

    response, result, attempts = asyncio.run(
        guard_generation(
            llm,
            generate=generate,
            query="q",
            config=GuardConfig(max_retries=2),
        )
    )
    assert result.passed
    assert attempts == 3
    assert seen_reasons[0] == ""
    assert "vague" in seen_reasons[1]
    assert "too short" in seen_reasons[2]
    assert response == "answer 3"

def test_response_cache_tenant_isolation() -> None:
    cache = FakeCache()
    rc = ResponseCache(cache)
    ctx = AgentContext(tenant_id="t1", kb_id="kb1")
    assert rc.put("question", "a longer cached answer here", context=ctx)
    assert (
        rc.get("question", context=ctx)
        == "a longer cached answer here"
    )
    assert (
        rc.get("question", context=AgentContext(tenant_id="t2", kb_id="kb1"))
        is None
    )


def test_response_cache_skips_short_fallback() -> None:
    rc = ResponseCache(FakeCache())
    assert rc.put("q", "short") is False


def test_response_cache_records_metrics() -> None:
    metrics = FakeMetrics()
    cache = FakeCache()
    rc = ResponseCache(cache, metrics=metrics)
    ctx = AgentContext(tenant_id="t1", kb_id="kb1")

    assert rc.get("question", context=ctx) is None
    assert rc.put("question", "a sufficiently long answer", context=ctx)
    assert rc.put("short", "x", context=ctx) is False

    assert [event[0] for event in metrics.cache_events] == ["miss", "write", "skip"]
    assert metrics.cache_events[1][1:] == ("t1", "kb1")

    assert rc.get("question", context=ctx) is not None
    assert [event[0] for event in metrics.cache_events] == [
        "miss",
        "write",
        "skip",
        "hit",
    ]
