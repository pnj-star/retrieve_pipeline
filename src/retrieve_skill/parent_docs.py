"""父块粒度聚合：把精排后达标的子块合并成一个父块文档。

检索管线的 ``docs`` 是"子块粒度"的结果：同一个父块可能被切成一串 chunk 且
全部命中。直接把这批子块交给 agent 会带来两处问题：一是上下文窗口被撑爆，
二是一段完整依据被拆成很多碎片。本模块把这些命中子块按稳定父块身份合并成
一个"父块文档"，再做单篇与总量的预算截断，保证返回的 docs 既是完整可读的整段依据、又不会把 agent 的上下文窗口打爆。

每个父块文档保留 ``child_ids``（命中的子块主键列表）与 ``parent_id``，供上层
评估 agent 据此回查具体子块核对召回是否正确，也兼容 toolbench 用子块 id 做
golden 召回匹配。
"""

from __future__ import annotations

from typing import Any, Sequence

from common_core.rag.assembly import DEFAULT_MAX_CONTEXT_CHARS, clean_markdown


def parent_keys_of(doc: dict[str, Any]) -> tuple[str, ...]:
    """子块归属的父块身份主键，用于把同父块子块聚成一组。

    按可靠性排序：parent_id 最可靠，其次 parent_title + source，最后退回
    parent_content / content 前 60 字符的内容指纹。返回所有候选主键，
    取第一个非空作为分组键；全部为空时视为无父块身份（调用方跳过聚合）。

    参数:
        doc: 单个子块文档字典。

    返回:
        可用于分组的身份主键列表（从最可靠到最弱）。
    """
    scope = "\x00".join(
        str(doc.get(field, "") or "")
        for field in ("tenant_id", "kb_id", "source")
    )
    parent_id = str(doc.get("parent_id", "") or "").strip()
    if parent_id:
        return (f"{scope}\x00parent:{parent_id}",)
    parent_title = str(doc.get("parent_title", "") or "").strip()
    if parent_title:
        return (f"{scope}\x00title:{parent_title}",)
    content = str(doc.get("parent_content", "") or doc.get("content", "") or "").strip()
    if content:
        return (f"{scope}\x00content:{content[:60]}",)
    return ()


def _merge_child_content(group: Sequence[dict[str, Any]]) -> str:
    """把同一父块的多个子块正文按 chunk 顺序合并去重，剔除空块。"""
    ranked = sorted(
        enumerate(group),
        key=lambda item: _chunk_sort_key(item[1], item[0]),
    )
    seen: set[str] = set()
    parts: list[str] = []
    for _position, doc in ranked:
        text = clean_markdown(str(doc.get("content", "") or "")).strip()
        if text and text not in seen:
            seen.add(text)
            parts.append(text)
    return "\n\n".join(parts)


def _chunk_sort_key(doc: dict[str, Any], position: int) -> tuple[int, int]:
    """返回可排序的 chunk 序号；缺失或非法时回退到输入顺序（视为排最后）。"""
    raw = doc.get("chunk_index")
    try:
        return (int(str(raw).strip()), position)
    except (TypeError, ValueError):
        return (2**31, position)


def _parent_content(group: Sequence[dict[str, Any]]) -> str:
    """取父块全文；找不到时退回子块合并结果。"""
    for doc in group:
        text = str(doc.get("parent_content", "") or "").strip()
        if text:
            return text
    return _merge_child_content(group)


def _max_number(*values: Any) -> float | None:
    """取这些值里合法的最大数值；全都非法时返回 None。"""
    numbers = []
    for value in values:
        if value is None:
            continue
        try:
            numbers.append(float(value))
        except (TypeError, ValueError):
            continue
    return max(numbers) if numbers else None


def _aggregate_group(
    key: str,
    group: Sequence[dict[str, Any]],
) -> dict[str, Any] | None:
    """把同一父块的一组子块合并成一个父块文档。"""
    first = group[0]
    content = _parent_content(group)
    if not content.strip():
        return None
    child_ids = [
        str(doc.get("id", "") or "")
        for doc in group
        if str(doc.get("id", "") or "").strip()
    ]
    doc = {
        "id": str(first.get("parent_id", "") or first.get("id", "") or ""),
        "parent_id": str(first.get("parent_id", "") or ""),
        "child_ids": child_ids,
        "content": content,
        "parent_title": first.get("parent_title"),
        "source": first.get("source"),
        "category": first.get("category"),
        "chunk_index": first.get("chunk_index"),
        "tenant_id": first.get("tenant_id"),
        "kb_id": first.get("kb_id"),
    }
    score = _max_number(*(d.get("score") for d in group))
    ce_score = _max_number(*(d.get("ce_score") for d in group))
    if score is not None:
        doc["score"] = score
    if ce_score is not None:
        doc["ce_score"] = ce_score
    return doc


def _truncate(
    text: str,
    limit: int,
) -> str:
    """在预算内保留前缀；预算不够放标记时退化为纯前缀。"""
    if limit <= 0 or not text:
        return ""
    if len(text) <= limit:
        return text
    marker = " [truncated]"
    if len(marker) <= limit:
        low, high = 0, len(text)
        while low < high:
            mid = (low + high + 1) // 2
            if len(text[:mid] + marker) <= limit:
                low = mid
            else:
                high = mid - 1
        if low:
            return text[:low] + marker
    low, high = 0, len(text)
    while low < high:
        mid = (low + high + 1) // 2
        if len(text[:mid]) <= limit:
            low = mid
        else:
            high = mid - 1
    return text[:low]


def aggregate_parent_docs(
    docs: Sequence[dict[str, Any]],
    *,
    max_chars: int | None = None,
    max_doc_chars: int | None = None,
) -> list[dict[str, Any]]:
    """把子块 docs 聚合成父块粒度并做预算截断。

    参数:
        docs: 精排后达标的子块文档列表；同一父块的多个子块会合并成一篇。
        max_chars: 合并后 docs 里正文的字符总预算；None 用默认值。
        max_doc_chars: 单篇父块文档正文的字符上限；None 时用总量的一半。

    返回:
        父块粒度去重后的文档列表，按首次出现的父块顺序排列。
    """
    if not docs:
        return []
    groups: dict[str, list[dict[str, Any]]] = {}
    order: list[str] = []
    for doc in docs:
        keys = parent_keys_of(doc)
        if not keys:
            continue
        key = keys[0]
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(doc)

    aggregated: list[dict[str, Any]] = []
    for key in order:
        parent = _aggregate_group(key, groups[key])
        if parent is None:
            continue
        content = str(parent.get("content", "") or "")

        # 单篇预算：显式传了单篇上限则用之，否则取总量预算的一半，避免一篇
        # 超长父块独占全部配额。
        total_chars = max_chars if max_chars is not None else DEFAULT_MAX_CONTEXT_CHARS
        per_chars = (
            max(1, max_doc_chars)
            if max_doc_chars is not None and max_doc_chars > 0
            else max(1, total_chars // 2)
        )
        if len(content) > per_chars:
            content = _truncate(content, per_chars)

        parent["content"] = content if content.strip() else ""
        aggregated.append(parent)

    # 总量预算：按聚合后的顺序逐篇扣减，超过总量就丢弃后面的篇幅（保留已
    # 放行的父块完整性）。
    total = max_chars if max_chars is not None else DEFAULT_MAX_CONTEXT_CHARS
    if total <= 0:
        return aggregated
    remaining = total
    kept: list[dict[str, Any]] = []
    for parent in aggregated:
        content = str(parent.get("content", "") or "")
        cost = len(content) if content else 0
        if cost == 0:
            continue
        if cost > remaining:
            break
        kept.append(parent)
        remaining -= cost
    return kept


__all__ = ["DEFAULT_MAX_CONTEXT_CHARS", "aggregate_parent_docs"]
