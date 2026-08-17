"""上下文组装阶段：markdown 清理、去重、截断与图片提取。

最终产出一段可直接拼进 LLM 提示词的上下文文本，以及对应的来源列表。
"""

from __future__ import annotations

import re
from typing import Any, Callable, Iterable, Sequence

# 拼入上下文的默认字符上限，防止超出模型上下文窗口
DEFAULT_MAX_CONTEXT_CHARS = 8000


def clean_markdown(text: str) -> str:
    """去掉知识块通常携带的 markdown 标记。

    处理三类常见标记：加粗（**x**）、行首标题（#）、反引号（`）。
    """
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.M)
    text = text.replace("`", "")
    return text


def dedupe_docs(
    docs: Sequence[dict[str, Any]],
    *,
    dedupe_key: Callable[[dict[str, Any]], str] | None = None,
) -> list[dict[str, Any]]:
    """按父块标题（parent_title）去重，或用调用方提供的 key 去重。

    默认 key 优先取父块标题，其次取内容前 60 个字符，避免同一父块
    的多个子块重复进入上下文。
    """
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for doc in docs:
        if dedupe_key is not None:
            key = dedupe_key(doc)
        else:
            parent_title = str(doc.get("parent_title", "") or "")
            content = str(doc.get("parent_content", "") or doc.get("content", "") or "")
            key = parent_title or content[:60]
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(doc)
    return result


def build_context_text(
    docs: Sequence[dict[str, Any]],
    *,
    max_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
    prefix_blocks: Sequence[str] | None = None,
    source_label: str = "source",
    format_doc: Callable[[int, dict[str, Any]], str] | None = None,
) -> tuple[str, list[str]]:
    """把文档组装成 ``(context_text, sources)`` 供 LLM 生成使用。

    - prefix_blocks: 放在正文之前的固定信息块（例如权威的 SQL 事实）；
    - format_doc: 自定义单条文档的格式化函数，默认输出 `[Source N] 内容`；
    - sources: 与切块顺序一致的来源列表（prefix 块对应的来源为空字符串）。
    """
    deduped = dedupe_docs(docs)
    parts: list[str] = []
    sources: list[str] = []
    total_len = 0

    for index, doc in enumerate(deduped):
        content = clean_markdown(
            str(doc.get("parent_content", "") or doc.get("content", "") or "")
        )
        if format_doc is not None:
            chunk = format_doc(len(parts) + 1, doc)
        else:
            chunk = f"[Source {len(parts) + 1}] {content}"
        if total_len + len(chunk) > max_chars:
            # 超出字符上限：截断，不再追加后续文档
            break
        parts.append(chunk)
        sources.append(str(doc.get(source_label, "") or ""))
        total_len += len(chunk)

    context_text = "\n\n---\n\n".join(parts)
    if prefix_blocks:
        prefix = "\n\n---\n\n".join(str(block) for block in prefix_blocks)
        context_text = prefix + ("\n\n---\n\n" + context_text if context_text else "")
        sources = [""] * len(prefix_blocks) + sources
    return context_text, sources


def extract_images(
    docs: Sequence[dict[str, Any]],
    images: Sequence[dict[str, Any]] = (),
    *,
    max_images: int = 5,
    url_fields: Iterable[str] = ("image_url", "url"),
) -> list[str]:
    """收集图片结果与文本文档中的图片 URL，保持顺序并去重。"""
    urls: list[str] = []
    for image in images:
        for field in url_fields:
            url = str(image.get(field, "") or "")
            if url and url not in urls:
                urls.append(url)
    for doc in docs:
        for url in doc.get("image_urls", []) or []:
            if url and url not in urls:
                urls.append(url)
    return urls[:max_images]