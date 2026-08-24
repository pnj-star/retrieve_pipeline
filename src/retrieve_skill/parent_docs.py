"""精排后的父块引用与权威父块内容组装。

Milvus 只保存子块和 ``parent_id``；Redis 只缓存精排达标后的轻量父块引用。
父块正文始终从 authoritative store（当前为 MySQL）回源，并在组装时校验租户、
知识库、版本与 token / char 预算。
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any, Mapping, Sequence

DEFAULT_MAX_CONTEXT_TOKENS = 6000
DEFAULT_MAX_DOC_TOKENS = 3000
_TOKEN_ENCODINGS: dict[str, Any] = {}


def default_token_counter(text: str) -> int:
    """Count tokens with tiktoken; callers can inject a model-specific counter."""
    import tiktoken

    encoding_name = os.getenv("CONTEXT_TOKEN_ENCODING", "cl100k_base")
    encoding = _TOKEN_ENCODINGS.get(encoding_name)
    if encoding is None:
        encoding = tiktoken.get_encoding(encoding_name)
        _TOKEN_ENCODINGS[encoding_name] = encoding
    return len(encoding.encode(text))


def build_parent_refs(docs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group qualified children into ranked parent references.

    The reference list is the only retrieval result stored in Redis. Parent
    text is intentionally omitted so MySQL remains the authoritative source.
    """
    grouped: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for position, doc in enumerate(docs):
        parent_id = str(doc.get("parent_id", "") or "").strip()
        if not parent_id:
            continue
        scope = "\x00".join(
            str(doc.get(field, "") or "")
            for field in ("tenant_id", "kb_id")
        )
        key = f"{scope}\x00{parent_id}"
        if key not in grouped:
            grouped[key] = {
                "tenant_id": str(doc.get("tenant_id", "") or ""),
                "kb_id": str(doc.get("kb_id", "") or ""),
                "parent_id": parent_id,
                "child_ids": [],
                "first_position": position,
                "ce_score": 0.0,
                "score": 0.0,
                "doc_version": doc.get("doc_version"),
            }
            order.append(key)
        group = grouped[key]
        child_id = str(doc.get("id", "") or "").strip()
        ce_score = float(doc["ce_score"])
        score = float(doc.get("score", 0.0) or 0.0)
        group["child_ids"].append(child_id)
        if ce_score == group["ce_score"]:
            group["doc_version"] = doc.get("doc_version")
        group["ce_score"] = max(float(group["ce_score"]), ce_score)
        group["score"] = max(float(group["score"]), score)

    refs = []
    for key in order:
        item = grouped[key]
        item.pop("first_position")
        refs.append(item)
    refs.sort(
        key=lambda item: (
            -float(item["ce_score"]),
            -float(item["score"]),
            str(item["parent_id"]),
        )
    )
    return refs


def validate_parent_refs(
    docs: Any,
    *,
    threshold: float,
    context: Any | None = None,
) -> list[dict[str, Any]] | None:
    """Validate cached parent references; invalid data is treated as a miss."""
    if not isinstance(docs, list) or not docs:
        return None
    validated: list[dict[str, Any]] = []
    seen_parents: set[str] = set()
    for item in docs:
        if not isinstance(item, dict):
            return None
        parent_id = str(item.get("parent_id", "") or "").strip()
        child_ids_raw = item.get("child_ids")
        ce_score = item.get("ce_score")
        doc_version = item.get("doc_version")
        if (
            not parent_id
            or parent_id in seen_parents
            or not isinstance(child_ids_raw, list)
            or not doc_version
        ):
            return None
        child_ids = [str(child or "").strip() for child in child_ids_raw]
        if not child_ids or any(not child for child in child_ids):
            return None
        if len(set(child_ids)) != len(child_ids):
            return None
        try:
            ce_score_value = float(ce_score)
            doc_version_value = int(str(doc_version).strip())
        except (TypeError, ValueError):
            return None
        if not 0.0 <= ce_score_value <= 1.0 or ce_score_value < threshold:
            return None
        score_value = 0.0
        if item.get("score") is not None:
            try:
                score_value = float(item["score"])
            except (TypeError, ValueError):
                return None
        if context is not None:
            if (
                str(item.get("tenant_id", "") or "") != context.tenant_id
                or str(item.get("kb_id", "") or "") != context.kb_id
            ):
                return None
        validated.append({
            "tenant_id": context.tenant_id if context else str(item.get("tenant_id", "") or ""),
            "kb_id": context.kb_id if context else str(item.get("kb_id", "") or ""),
            "parent_id": parent_id,
            "child_ids": child_ids,
            "ce_score": ce_score_value,
            "score": score_value,
            "doc_version": doc_version_value,
        })
        seen_parents.add(parent_id)
    return validated


def _truncate_to_limit(
    text: str,
    limit: int,
    measure: Callable[[str], int],
) -> str:
    """Keep the longest prefix that fits the measured budget."""
    if limit <= 0 or not text:
        return ""
    if measure(text) <= limit:
        return text
    marker = " [truncated]"
    if measure(marker) <= limit:
        low, high = 0, len(text)
        while low < high:
            mid = (low + high + 1) // 2
            if measure(text[:mid] + marker) <= limit:
                low = mid
            else:
                high = mid - 1
        if low:
            return text[:low] + marker
    low, high = 0, len(text)
    while low < high:
        mid = (low + high + 1) // 2
        if measure(text[:mid]) <= limit:
            low = mid
        else:
            high = mid - 1
    return text[:low]


def assemble_parent_refs(
    refs: Sequence[dict[str, Any]],
    parent_rows: Mapping[str, dict[str, Any]],
    *,
    context: Any,
    count_tokens: Callable[[str], int] | None = None,
    context_max_tokens: int | None = None,
    max_doc_tokens: int | None = None,
    context_max_chars: int | None = None,
    max_doc_chars: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Hydrate parent refs and apply token and optional character budgets."""
    measure = count_tokens or default_token_counter
    token_budget = (
        context_max_tokens
        if context_max_tokens is not None
        else DEFAULT_MAX_CONTEXT_TOKENS
    )
    per_doc_tokens = (
        max_doc_tokens
        if max_doc_tokens is not None and max_doc_tokens > 0
        else max(1, token_budget // 2)
    )
    use_chars = context_max_chars is not None
    char_budget = context_max_chars if use_chars else 0
    per_doc_chars = (
        max(1, max_doc_chars)
        if max_doc_chars is not None and max_doc_chars > 0
        else None
    )

    parents: list[dict[str, Any]] = []
    missing_parent_count = 0
    version_mismatch_count = 0
    remaining_tokens = token_budget
    remaining_chars = char_budget
    for ref in refs:
        parent_id = str(ref.get("parent_id", "") or "")
        row = parent_rows.get(parent_id)
        if not row:
            missing_parent_count += 1
            continue
        if context is not None and (
            str(row.get("tenant_id", "") or "") != context.tenant_id
            or str(row.get("kb_id", "") or "") != context.kb_id
        ):
            missing_parent_count += 1
            continue
        try:
            row_version = int(str(row.get("doc_version", "") or "").strip())
            ref_version = int(str(ref.get("doc_version", "") or "").strip())
        except (TypeError, ValueError):
            version_mismatch_count += 1
            continue
        if row_version != ref_version:
            version_mismatch_count += 1
            continue
        content = str(row.get("content", "") or "")
        if not content.strip():
            missing_parent_count += 1
            continue

        if per_doc_chars is not None:
            content = _truncate_to_limit(content, per_doc_chars, len)
        content = _truncate_to_limit(content, per_doc_tokens, measure)
        content_tokens = measure(content)
        content_chars = len(content)
        if content_tokens > remaining_tokens:
            break
        if use_chars and content_chars > remaining_chars:
            break

        source = (
            str(row.get("source_uri", "") or "")
            or str(row.get("source_id", "") or "")
            or str(row.get("source_type", "") or "")
        )
        parent = {
            "id": parent_id,
            "parent_id": parent_id,
            "child_ids": list(ref.get("child_ids", [])),
            "content": content,
            "parent_title": row.get("title"),
            "summary": row.get("summary"),
            "source": source,
            "source_type": row.get("source_type"),
            "category": row.get("category"),
            "tenant_id": context.tenant_id if context else ref.get("tenant_id"),
            "kb_id": context.kb_id if context else ref.get("kb_id"),
            "doc_version": row_version,
            "content_sha256": row.get("content_sha256"),
            "visibility": row.get("visibility"),
            "score": ref.get("score"),
            "ce_score": ref.get("ce_score"),
        }
        parents.append(parent)
        remaining_tokens -= content_tokens
        if use_chars:
            remaining_chars -= content_chars

    stats = {
        "missing_parent_count": missing_parent_count,
        "version_mismatch_count": version_mismatch_count,
        "context_tokens": measure("".join(str(item.get("content", "")) for item in parents)),
    }
    return parents, stats


def filter_parent_refs(
    refs: Sequence[dict[str, Any]],
    parent_rows: Mapping[str, dict[str, Any]],
    *,
    context: Any,
) -> list[dict[str, Any]]:
    """Keep only refs whose authoritative parent row is current and usable."""
    valid: list[dict[str, Any]] = []
    for ref in refs:
        parent_id = str(ref.get("parent_id", "") or "")
        row = parent_rows.get(parent_id)
        if not row:
            continue
        if context is not None and (
            str(row.get("tenant_id", "") or "") != context.tenant_id
            or str(row.get("kb_id", "") or "") != context.kb_id
        ):
            continue
        try:
            row_version = int(str(row.get("doc_version", "") or "").strip())
            ref_version = int(str(ref.get("doc_version", "") or "").strip())
        except (TypeError, ValueError):
            continue
        if row_version != ref_version or not str(row.get("content", "") or "").strip():
            continue
        valid.append(dict(ref))
    return valid

__all__ = [
    "DEFAULT_MAX_CONTEXT_TOKENS",
    "DEFAULT_MAX_DOC_TOKENS",
    "assemble_parent_refs",
    "build_parent_refs",
    "default_token_counter",
    "filter_parent_refs",
    "validate_parent_refs",
]
