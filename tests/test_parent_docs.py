"""父块聚合模块的单测：多子块合并、child_ids、预算截断。"""

from __future__ import annotations

from retrieve_skill.parent_docs import aggregate_parent_docs


def _child(
    id: str,
    *,
    parent_id: str = "",
    content: str = "text",
    chunk_index: int = 0,
    source: str = "a.md",
    tenant: str = "t",
    kb: str = "k",
    score: float | None = None,
):
    doc = {
        "id": id,
        "parent_id": parent_id,
        "content": content,
        "chunk_index": chunk_index,
        "source": source,
        "tenant_id": tenant,
        "kb_id": kb,
    }
    if score is not None:
        doc["score"] = score
    return doc


def test_same_parent_children_merge_into_one_doc():
    docs = [
        _child("c1", parent_id="p1", content="first", chunk_index=0, score=0.9),
        _child("c2", parent_id="p1", content="second", chunk_index=1, score=0.7),
        _child("c3", parent_id="p2", content="other", chunk_index=0, score=0.5),
    ]
    result = aggregate_parent_docs(docs)
    assert [d["id"] for d in result] == ["p1", "p2"]
    p1 = result[0]
    assert p1["child_ids"] == ["c1", "c2"]
    assert p1["score"] == 0.9
    assert "first" in p1["content"]
    assert "second" in p1["content"]


def test_children_merge_in_chunk_index_order():
    docs = [
        _child("c2", parent_id="p", content="second", chunk_index=1),
        _child("c1", parent_id="p", content="first", chunk_index=0),
    ]
    result = aggregate_parent_docs(docs)
    assert result[0]["content"] == "first\n\nsecond"


def test_doc_without_parent_id_becomes_own_parent():
    result = aggregate_parent_docs([_child("c1", content="solo", source="b.md")])
    assert len(result) == 1
    assert result[0]["id"] == "c1"
    assert result[0]["child_ids"] == ["c1"]


def test_total_budget_truncates_later_docs():
    docs = [
        _child("c1", parent_id="p1", content="a" * 100),
        _child("c2", parent_id="p2", content="b" * 100),
        _child("c3", parent_id="p3", content="c" * 100),
    ]
    result = aggregate_parent_docs(docs, max_chars=120)
    # 每篇被截到总量一半（60），三篇里只有前两篇能放进 120 的总预算。
    assert [d["id"] for d in result] == ["p1", "p2"]
    assert result[0]["id"] == "p1"


def test_per_doc_budget_truncates_single_long_parent():
    docs = [_child("c1", parent_id="p1", content="a" * 100)]
    result = aggregate_parent_docs(docs, max_doc_chars=20)
    assert len(result) == 1
    assert len(result[0]["content"]) <= 20


def test_empty_input_returns_empty():
    assert aggregate_parent_docs([]) == []
