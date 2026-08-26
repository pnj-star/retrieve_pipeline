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
    """用 tiktoken 计算文本 token 数。

    参数:
        text: 待计数的任意文本；None 会先转成空字符串。

    返回:
        token 数量。编码器按 CONTEXT_TOKEN_ENCODING 配置选择，
        默认 cl100k_base；同一编码会进程内缓存，避免重复构建。

    说明:
        如果目标模型有专用 tokenizer，可在 pipeline 中注入 count_tokens 覆盖本函数。
    """
    import tiktoken

    encoding_name = os.getenv("CONTEXT_TOKEN_ENCODING", "cl100k_base")
    encoding = _TOKEN_ENCODINGS.get(encoding_name)
    if encoding is None:
        encoding = tiktoken.get_encoding(encoding_name)
        _TOKEN_ENCODINGS[encoding_name] = encoding
    return len(encoding.encode(text))


def build_parent_refs(docs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """把精排达标的子块聚合成带排序信息的轻量父块引用。

    处理流程:
    1. 忽略没有 parent_id 的异常子块；
    2. 用 tenant_id + kb_id + parent_id 作为分组键，避免不同作用域同名 ID 串数据；
    3. 同一父块下收集 child_ids，并保留最高 ce_score 和最高粗检索 score；
    4. 先按首次出现顺序稳定建组，最后按精排分、检索分和 parent_id 排序。

    参数:
        docs: 已经通过相关性阈值的子块文档序列。每个文档必须能读取
            parent_id 和 ce_score，通常还带有 id、tenant_id、kb_id、score、doc_version。

    返回:
        父块引用列表。每项包含 tenant_id、kb_id、parent_id、child_ids、ce_score、
        score 和 doc_version；故意不包含正文，保证 Redis 只存轻量引用，
        MySQL 始终是父块正文的唯一权威来源。

    异常:
        KeyError: 文档缺少 ce_score 字段。
        ValueError/TypeError: 分数字段无法转成 float。
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
        # ce_score 相同只是保持最后一次读到的版本；正常情况下同组子块来自
        # 同一父块快照。这里不引入额外排序规则，避免改变既有行为。
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
    """校验缓存中的父块引用；任何结构或质量异常都视为缓存未命中。

    校验内容包括：整体必须是列表、每一项必须是字典且字段类型合法、
    parent_id 不重复、child_ids 非空且无重复、ce_score 在阈值内且为有限数值、
    doc_version 可以转成整数；传入 context 时还会校验租户/知识库一致。

    参数:
        docs: Redis 反序列化得到的候选父块引用，可能是损坏或过期结构。
        threshold: 本次请求使用的最低精排相关性阈值。
        context: agent 上下文；非空时用于强制校验 tenant_id 和 kb_id。

    返回:
        校验通过并归一化后的引用列表；任何一项无效则整体返回 None，
        让上层把它当作缓存 miss 并走完整检索，而不是带着可疑数据继续。
    """
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
    """截断文本，保留不超过测量预算的最长前缀。

    使用二分查找而不是逐字符缩短，避免长父块在每次请求中被大量重复计数。
    当预算足够容纳截断标记时会追加 " [truncated]"；预算极小时返回空字符串。

    参数:
        text: 原始文本。
        limit: 最大允许长度，单位由 measure 决定（通常是 token 或字符）。
        measure: 长度测量函数，输入文本并返回非负数量，例如 token 计数器或 len。

    返回:
        不超过 limit 的截断后文本。limit <= 0 或原文本为空时返回空字符串。
    """
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
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """回源后的父块引用组装成展示上下文，并应用 token 预算。

    处理流程:
    1. 按排序后的 refs 顺序查找权威父块行；缺失、跨租户、版本不一致或正文为空
       的引用都会被跳过并计入诊断统计；
    2. 每篇父块先按单篇 token 上限截断；
    3. 若当前父块会超过上下文总预算，停止继续添加后续父块；
    4. 输出回答侧可直接使用的父块视图和数量/token 统计。

    参数:
        refs: 已通过过滤的轻量父块引用，顺序代表最终优先级。
        parent_rows: MySQL 回源结果，key 是 parent_id，value 是父块字段字典。
        context: agent 上下文，用于再次确认父块所属租户和知识库。
        count_tokens: 自定义 token 计数函数；None 用默认 tiktoken 计数器。
        context_max_tokens: 所有父块正文的总 token 预算；None 用默认 6000。
        max_doc_tokens: 单篇父块 token 上限；None 或 <=0 时默认取总预算的一半。

    返回:
        (parents, stats) 二元组。parents 是可直接给 agent/LLM 的父块文档列表；
        stats 包含 missing_parent_count、version_mismatch_count 和实际 content token 总量。
    """
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

    parents: list[dict[str, Any]] = []
    missing_parent_count = 0
    version_mismatch_count = 0
    remaining_tokens = token_budget
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

        content = _truncate_to_limit(content, per_doc_tokens, measure)
        content_tokens = measure(content)
        if content_tokens > remaining_tokens:
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
    """只保留权威父块仍然存在、版本一致且可用的引用。

    该方法用于回写 Redis 前清理失效数据。判断条件和最终组装保持一致：
    父块必须存在于 parent_rows、属于当前租户/知识库、doc_version 与缓存引用一致，
    并且正文不为空。

    参数:
        refs: 待过滤的父块引用序列。
        parent_rows: 批量回源得到的父块字典。
        context: agent 上下文；非空时用于租户和知识库一致性检查。

    返回:
        仍可使用的父块引用浅拷贝列表。顺序与入参 refs 保持一致，
        方便调用方直接替换旧缓存。
    """
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
