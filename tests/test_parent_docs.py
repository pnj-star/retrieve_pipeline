"""Parent reference grouping, validation, and authoritative assembly tests."""

from __future__ import annotations

from common_core.context import AgentContext

from retrieve_skill.parent_docs import (
    assemble_parent_refs,
    build_parent_refs,
    filter_parent_refs,
    validate_parent_refs,
)


def _child(
    child_id: str,
    parent_id: str,
    *,
    ce_score: float,
    score: float = 0.8,
    doc_version: int = 1,
) -> dict:
    return {
        "id": child_id,
        "content": f"{child_id} text",
        "parent_id": parent_id,
        "tenant_id": "t1",
        "kb_id": "kb1",
        "doc_version": doc_version,
        "score": score,
        "ce_score": ce_score,
    }


def _context() -> AgentContext:
    return AgentContext(tenant_id="t1", kb_id="kb1")


def test_build_parent_refs_groups_and_ranks_parents() -> None:
    refs = build_parent_refs([
        _child("c1", "p2", ce_score=0.7),
        _child("c2", "p1", ce_score=0.95),
        _child("c3", "p1", ce_score=0.9),
        _child("c4", "", ce_score=0.99),
    ])

    assert [ref["parent_id"] for ref in refs] == ["p1", "p2"]
    assert refs[0]["child_ids"] == ["c2", "c3"]
    assert refs[0]["ce_score"] == 0.95
    assert refs[0]["doc_version"] == 1
    assert all("content" not in ref for ref in refs)
    assert all("chunk_indexes" not in ref for ref in refs)
    assert all("best_child_id" not in ref for ref in refs)


def test_validate_parent_refs_enforces_threshold_and_scope() -> None:
    valid = build_parent_refs([_child("c1", "p1", ce_score=0.9)])
    assert validate_parent_refs(valid, threshold=0.8, context=_context()) == [
        {
            "tenant_id": "t1",
            "kb_id": "kb1",
            "parent_id": "p1",
            "child_ids": ["c1"],
            "ce_score": 0.9,
            "score": 0.8,
            "doc_version": 1,
        }
    ]

    below = build_parent_refs([_child("c1", "p1", ce_score=0.7)])
    assert validate_parent_refs(below, threshold=0.8, context=_context()) is None

    other_kb = [dict(valid[0], kb_id="other")]
    assert validate_parent_refs(other_kb, threshold=0.8, context=_context()) is None


def test_assemble_parent_refs_checks_version_and_budget() -> None:
    ref = {
        "tenant_id": "t1",
        "kb_id": "kb1",
        "parent_id": "p1",
        "child_ids": ["c1", "c2"],
        "ce_score": 0.9,
        "score": 0.8,
        "doc_version": 2,
    }
    rows = {
        "p1": {
            "tenant_id": "t1",
            "kb_id": "kb1",
            "title": "title",
            "content": "a" * 100,
            "doc_version": 2,
        },
        "stale": {
            "tenant_id": "t1",
            "kb_id": "kb1",
            "content": "old",
            "doc_version": 1,
        },
    }
    parents, stats = assemble_parent_refs(
        [ref],
        rows,
        context=_context(),
        count_tokens=len,
        context_max_tokens=50,
        max_doc_tokens=30,
    )

    assert len(parents) == 1
    assert parents[0]["id"] == "p1"
    assert parents[0]["child_ids"] == ["c1", "c2"]
    assert "[truncated]" in parents[0]["content"]
    assert stats["context_tokens"] <= 50

    stale_ref = dict(ref, parent_id="stale")
    stale_parents, stale_stats = assemble_parent_refs(
        [stale_ref],
        rows,
        context=_context(),
        count_tokens=len,
    )
    assert stale_parents == []
    assert stale_stats["version_mismatch_count"] == 1


def test_filter_parent_refs_keeps_current_rows_only() -> None:
    refs = [
        {"parent_id": "current", "doc_version": 2},
        {"parent_id": "missing", "doc_version": 1},
        {"parent_id": "stale", "doc_version": 1},
    ]
    rows = {
        "current": {
            "tenant_id": "t1",
            "kb_id": "kb1",
            "content": "ok",
            "doc_version": 2,
        },
        "stale": {
            "tenant_id": "t1",
            "kb_id": "kb1",
            "content": "ok",
            "doc_version": 3,
        },
    }
    filtered = filter_parent_refs(refs, rows, context=_context())
    assert [ref["parent_id"] for ref in filtered] == ["current"]
