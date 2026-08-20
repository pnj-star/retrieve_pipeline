"""上下文组装阶段：markdown 清理、去重、截断与图片提取。

最终产出一段可直接拼进 LLM 提示词的上下文文本，以及对应的来源列表。

整体思路：把"检索/精排出来的文档"变成"给 LLM 的提示词上下文"，中间要解决
四个问题——去掉知识块常见的 markdown 噪音、把同一个父块的多个子块合并去重、
控制总长度不超出模型上下文窗口、必要时预留 prefix（权威事实）的位置。
"""

from __future__ import annotations

import re
from typing import Any, Callable, Iterable, Sequence

from common_core.config import RetrievalConfig


_ASSEMBLY_DEFAULTS = RetrievalConfig()
DEFAULT_MAX_CONTEXT_CHARS = _ASSEMBLY_DEFAULTS.assembly_max_context_chars

# 上下文各块之间的分隔符（同时计入预算）。
CONTEXT_SEPARATOR = "\n\n---\n\n"
# 截断时追加的提示标记，告知 LLM 这段内容被砍断了。
TRUNCATION_MARKER = " [truncated]"


def clean_markdown(text: str) -> str:
    """去掉知识块通常携带的 markdown 标记。

    处理三类常见标记：加粗（**x**）、行首标题（#）、反引号（`）。
    顺序很重要：先处理**加粗**和**行首标题**（结构化标记），最后再删反引号，
    避免行内代码的内容里残留下别的标记。

    逻辑：
    1. 把 ``**x**`` 里的加粗符号剥掉只留内容；
    2. 把行首的 ``#``~``######`` 标题标记删掉（用 re.M 逐行匹配）；
    3. 全局删除反引号。

    参数:
        text: 待清理的原始文本（常为知识块正文）。

    返回:
        清理掉加粗、行首标题与反引号后的文本。
    """
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.M)
    text = text.replace("`", "")
    return text


def _truncate_to_limit(
    text: str,
    limit: int,
    measure: Callable[[str], int],
) -> str:
    """在指定预算内保留文本前缀，预算不足以放截断标记时退化为纯前缀。

    存在的意义：预算不可溢出，但要尽量多留内容。measure 决定"计量单位"
    （字符数 len 或 token 数 count_tokens），因此本函数在字符模式和 token
    模式下通用。

    逻辑：
    1. 预算<=0 或文本为空 → 直接返回空串（起点保护）；
    2. 整段就已经在预算内 → 原样返回，不截断；
    3. 预算够放截断标记时：用二分查找"前缀 + 标记"不超过预算的最大长度，
       （二分保证结果一定合法，避免逐字符试带来的 O(n) 开销）；
    4. 预算连标记都放不下 → 退化为纯前缀（不带标记），同样二分。

    参数:
        text: 待截断的文本。
        limit: 允许的最大计量值（字符数或 token 数）上限。
        measure: 把字符串换算成计量值的函数，如 len 或 count_tokens。

    返回:
        不超过 limit 的截断结果；必要时会带 "[truncated]" 标记。
    """
    if limit <= 0 or not text:
        return ""
    if measure(text) <= limit:
        return text

    marker = TRUNCATION_MARKER
    if measure(marker) <= limit:
        # 场景 A：能带上"已截断"标记。二分寻找最大的 low，使 text[:low] + marker 仍不超预算。
        low, high = 0, len(text)
        while low < high:
            mid = (low + high + 1) // 2  # 上取整，防止死循环
            if measure(text[:mid] + marker) <= limit:
                low = mid
            else:
                high = mid - 1
        if low:
            return text[:low] + marker

    # 场景 B：预算太小，标记都放不下。返回纯前缀 text[:low]（可能为空）。
    low, high = 0, len(text)
    while low < high:
        mid = (low + high + 1) // 2
        if measure(text[:mid]) <= limit:
            low = mid
        else:
            high = mid - 1
    return text[:low]


def _render_doc(
    index: int,
    doc: dict[str, Any],
    format_doc: Callable[[int, dict[str, Any]], str] | None,
) -> str:
    """渲染单条上下文；默认格式优先使用父块全文。

    - 有 format_doc：完全交给调用方格式化（如带上 score / source 附注）；
    - 无 format_doc：输出 ``[Source N] 清理后的正文``，正文优先取父块
      parent_content（整篇），取不到再退回当前块 content。

    参数:
        index: 文档在上下文中的序号，用作 [Source N] 前缀。
        doc: 单篇检索返回的文档字典。
        format_doc: 可选的自定义格式化函数，(index, doc) -> str。

    返回:
        渲染好的单条上下文文本。
    """
    if format_doc is not None:
        return format_doc(index, doc)
    content = clean_markdown(
        str(doc.get("parent_content", "") or doc.get("content", "") or "")
    )
    return f"[Source {index}] {content}"


def _parent_scope(doc: dict[str, Any]) -> str:
    """父块分组作用域：tenant / kb / source，避免不同租户或文档误合并。

    为什么要有它：同一个 parent_title 或长得很像的内容可能出现在不同的
    租户 / 知识库 / 文档里，光按 title 分组会把它们错误地并成一组。
    用 NUL 字符（\x00）连接三段作为分隔符是安全的——正常文本不会包含 NUL。

    参数:
        doc: 单篇文档字典。

    返回:
        由 tenant_id / kb_id / source 拼接成的分组作用域字符串。
    """
    parts = []
    for field in ("tenant_id", "kb_id", "source"):
        parts.append(str(doc.get(field, "") or ""))
    return "\x00".join(parts)


def _stable_parent_key(doc: dict[str, Any]) -> str:
    """返回稳定父块身份；有 parent_id 时优先，否则回退 source + title / content。

    身份"稳定性"决定去重/分组是否可靠，回退链，按可靠性递减：
    1. 最可靠：parent_id（父块的唯一主键）→ scope + parent:xxx；
    2. 其次：parent_title（标题通常唯一）→ scope + title:xxx；
    3. 再其次：内容前 60 字符作为指纹 → scope + content:xxx[:60]；
    4. 全都拿不到 → 返回 ""（调用方会跳过它，不参与去重/分组）。

    参数:
        doc: 单篇文档字典。

    返回:
        稳定父块身份 key；无法识别时返回空字符串。
    """
    parent_id = str(doc.get("parent_id", "") or "").strip()
    scope = _parent_scope(doc)  # 先算作用域，保证不同租户/文档不会互相碰撞
    if parent_id:
        return f"{scope}\x00parent:{parent_id}"
    parent_title = str(doc.get("parent_title", "") or "").strip()
    if parent_title:
        return f"{scope}\x00title:{parent_title}"
    content = str(doc.get("parent_content", "") or doc.get("content", "") or "").strip()
    if not content:
        return ""
    return f"{scope}\x00content:{content[:60]}"


def _group_docs(docs: Sequence[dict[str, Any]]) -> list[tuple[str, list[dict[str, Any]]]]:
    """按稳定父块身份分组，保留首次出现的父块顺序。

    用途：同一个父块可能被切成多个 chunk 且都命中了检索，把它们归到一组，
    后面 _render_group 再决定"用父块全文"还是"把子块按顺序合并"。

    逻辑：用 dict 存组 + 独立 order 列表记住第一次出现的顺序；
    无法生成身份 key 的文档直接跳过（不会成为一组）。

    参数:
        docs: 待分组的文档序列。

    返回:
        [(身份 key, 该组文档列表), ...] 列表，按首现顺序排列。
    """
    groups: dict[str, list[dict[str, Any]]] = {}
    order: list[str] = []
    for doc in docs:
        key = _stable_parent_key(doc)
        if not key:
            continue
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(doc)
    return [(key, groups[key]) for key in order]


def _chunk_index(doc: dict[str, Any], position: int) -> tuple[int, int]:
    """返回可排序的 chunk 序号；缺失或非数字时回退到输入顺序。

    返回 ``(序号, 原始位置)`` 二元组，方便 sorted() 稳定排序：
    - 有合法 chunk_index → 按它排序；
    - 没有 / 非法 → 用 2**31（视为"排在最后"）+ 原始位置兜底，
      保证同批非法序号之间仍按输入顺序排，不出现随机抖动。

    参数:
        doc: 单篇文档字典，读取其 chunk_index 字段。
        position: 文档在输入里的原始位置，作为同序号时的稳定兜底。

    返回:
        (排序号, 原始位置) 二元组，可直接交给 sorted() 的 key。
    """
    raw = doc.get("chunk_index")
    try:
        return (int(str(raw).strip()), position)
    except (TypeError, ValueError):
        return (2**31, position)


def _render_group(
    index: int,
    group: list[dict[str, Any]],
    format_doc: Callable[[int, dict[str, Any]], str] | None,
) -> str:
    """渲染同一父块的子块组；有 parent_content 时用父块全文，否则合并子块。

    两条渲染路径：
    1. 父块路线：组里只要有任意文档带 parent_content（整篇正文）→ 直接用父块
       全文（最多信息量，且不会重复），避免把子块一个个拼出来又长又碎；
    2. 子块路线：没有 parent_content → 把子块按 chunk_index 排好序，
       去掉空块后用空行合并成一段；有 format_doc 时再造一个
       "synthetic" 文档（content=合并结果）交回给调用方格式化，
       否则输出 ``[Source N]（标题）\n合并内容``。

    参数:
        index: 组在上下文中的序号，用作 [Source N] 前缀。
        group: 同一父块的子块文档列表。
        format_doc: 可选的自定义格式化函数，(index, doc) -> str。

    返回:
        渲染好的该组上下文文本；无可用内容时返回空字符串。
    """
    # 父块路线：找到第一个带 parent_content 的文档（用它的全文，忽略其余子块内容）
    parent_doc = next(
        (doc for doc in group if str(doc.get("parent_content", "") or "").strip()),
        None,
    )
    if parent_doc is not None:
        if format_doc is not None:
            return format_doc(index, parent_doc)
        content = clean_markdown(str(parent_doc.get("parent_content", "") or ""))
        return f"[Source {index}] {content}"

    # 子块路线：先按 chunk_index 排序（同序号保持输入顺序），再逐块清理并合并且去空
    children = sorted(
        enumerate(group),
        key=lambda item: _chunk_index(item[1], item[0]),
    )
    contents = [
        clean_markdown(str(doc.get("content", "") or "")).strip()
        for _position, doc in children
    ]
    contents = [
        item
        for item in contents
        if item
    ]
    # 子块按"清理后的完整内容"去重并保序，避免重叠切窗产生的内容重复喂给 LLM
    seen: set[str] = set()
    deduped: list[str] = []
    for item in contents:
        if item not in seen:
            seen.add(item)
            deduped.append(item)
    contents = deduped
    if not contents:
        return ""
    combined = "\n\n".join(contents)
    if format_doc is not None:
        # 合成一个"看起来像单篇文档"的 dict，让调用方的格式函数无脑复用
        synthetic = dict(group[0])
        synthetic["content"] = combined
        return format_doc(index, synthetic)
    title = str(group[0].get("parent_title", "") or "").strip()
    prefix = f"[Source {index}]" + (f"（{title}）" if title else "")
    return f"{prefix}\n{combined}"


def dedupe_docs(
    docs: Sequence[dict[str, Any]],
    *,
    dedupe_key: Callable[[dict[str, Any]], str] | None = None,
) -> list[dict[str, Any]]:
    """按稳定父块身份去重，或用调用方提供的 key 去重。

    默认 key 优先取 parent_id；缺失时依次回退到
    tenant + kb + source + parent_title、tenant + kb + source + 内容前 60 个字符，
    避免同一父块的多个子块重复进入上下文，同时防止不同租户或文档误合并。

    逻辑：遍历并维护一个 seen 集合；key 为空或已见过 → 跳过，否则保留。
    保留的是"该父块第一个出现的文档"。

    参数:
        docs: 待去重的文档序列。
        dedupe_key: 可选的自定义身份 key 函数；未传时使用稳定父块身份。

    返回:
        去重后的文档列表（保留首现顺序）。
    """
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for doc in docs:
        if dedupe_key is not None:
            key = dedupe_key(doc)
        else:
            key = _stable_parent_key(doc)
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
    max_doc_chars: int | None = None,
    max_doc_tokens: int | None = None,
    max_tokens: int | None = None,
    count_tokens: Callable[[str], int] | None = None,
) -> tuple[str, list[str]]:
    """把文档组装成 ``(context_text, sources)`` 供 LLM 生成使用。

    这是上下文组装的核心，核心逻辑是**预算核算**：总预算、单篇上限、分隔符、
    prefix 全部用同一个"计量单位"（字符或 token）统一扣减，保证最终产物
    一定 <= 总预算。

    参数：
    - docs: 检索 / 精排后的文档列表；同一父块的多个子块会先合并再渲染。
    - max_chars: 整体字符预算，也是未启用 token 模式时的默认总上限，默认 8000。
    - prefix_blocks: 放在正文之前的固定信息块（例如权威的 SQL 事实），
      同样计入整体预算，超长时会截断而不是丢弃正文。
    - source_label: 生成 sources 时从文档中读取来源字段的字段名，默认 "source"。
    - format_doc: 自定义单条文档（或合并后的父块组）的格式化函数，
      签名是 ``(index, doc) -> str``；默认输出 `[Source N] 内容`。
    - max_doc_chars: 单篇文档字符上限；None 时在字符模式下使用 max_chars // 2，
      避免一篇长文档独占上下文配额。
    - max_doc_tokens: token 模式下单篇文档 token 上限；None 时默认 max_tokens // 2。
    - max_tokens: token 模式的总预算；需同时传入 count_tokens 才按真实 token 计量。
    - count_tokens: 文本转 token 数的计数器，例如 ``lambda s: len(encode(s))``；
      传了 max_tokens 但未传 count_tokens 时，按字符数作为计量单位（非真实 token）。

    返回：
    - context_text: 拼好并截断后的上下文文本，分隔符和 prefix 都计入总预算。
    - sources: 与上下文切块顺序一一对应的来源列表；prefix 块对应空字符串。
    """
    # ---------- 第一步：确定"计量单位"与总预算 ----------
    # token 模式（同时给了 max_tokens 和 count_tokens）才真正按 token 计量；
    # 否则退化为字符模式，以 len 为计量函数。这让本函数只写一套逻辑。
    budget = max_tokens if max_tokens is not None else max_chars
    measure = (
        count_tokens
        if max_tokens is not None and count_tokens is not None
        else len
    )
    if budget <= 0:
        return "", []

    # ---------- 第二步：确定"单篇文档"上限 ----------
    # 目的：防止一篇超长文档（比如 parent_content 有一万多字）把整个预算吃光，
    # 导致别的来源进不来。token 模式用 token 上限，字符模式用字符上限。
    # 用 /2 是经验值：给单篇文档最多一半预算，剩下的留给其他文档和分隔符。
    if max_tokens is not None and count_tokens is not None:
        # token 模式的单篇上限：
        #   - 字符上限只做额外收紧（per_doc_chars），主上限是 per_doc_limit（token）
        per_doc_chars = (
            max(1, max_doc_chars)
            if max_doc_chars is not None and max_doc_chars > 0
            else None
        )
        per_doc_limit = (
            max(1, max_doc_tokens)
            if max_doc_tokens is not None and max_doc_tokens > 0
            else max(1, max_tokens // 2)
        )
    else:
        # 字符模式：单篇上限即 max_doc_chars（未指定则取总预算的一半）
        per_doc_chars = (
            max(1, max_doc_chars)
            if max_doc_chars is not None and max_doc_chars > 0
            else max(1, max_chars // 2)
        )
        per_doc_limit = None

    # ---------- 第三步：处理 prefix 固定信息块 ----------
    # prefix 也占用预算，且优先级最高——先扣掉它的长度，剩下的才是正文预算；
    # 若 prefix 本身超长，直接截断它而非丢弃，保证权威事实尽量保留。
    separator_cost = measure(CONTEXT_SEPARATOR)
    prefix_text = CONTEXT_SEPARATOR.join(
        str(block) for block in (prefix_blocks or ())
    )
    if measure(prefix_text) > budget:
        prefix_text = _truncate_to_limit(prefix_text, budget, measure)
    remaining = budget - measure(prefix_text)

    # ---------- 第四步：逐组合并渲染并做预算核算 ----------
    parts: list[str] = []  # 已放行的上下文片段
    sources: list[str] = []  # 与 parts 一一对应的来源（prefix 时段为空）
    for _key, group in _group_docs(docs):
        # 4.1 把这一组（同一父块）渲染成一段文本
        chunk = _render_group(len(parts) + 1, group, format_doc)

        # 4.2 先做"单篇上限"截断：早截早省，避免长文独占配额
        if per_doc_chars is not None and len(chunk) > per_doc_chars:
            chunk = _truncate_to_limit(chunk, per_doc_chars, len)
        if per_doc_limit is not None and measure(chunk) > per_doc_limit:
            chunk = _truncate_to_limit(chunk, per_doc_limit, measure)
        if not chunk:
            continue  # 截断后为空（极限情况）则跳过这篇

        # 4.3 核算"放不放行"：每段之间要有分隔符，分隔符也算开销。
        #     若剩余预算连分隔符+内容都装不下 → 直接停（break）。
        overhead = separator_cost if (prefix_text or parts) else 0
        room = remaining - overhead
        if room <= 0:
            break
        if measure(chunk) > room:
            # 内容比剩余空间大：截断到刚好塞下，而不是整个丢弃
            chunk = _truncate_to_limit(chunk, room, measure)
        if not chunk:
            continue

        # 4.4 放行：记录片段 + 来源，扣减剩余预算
        parts.append(chunk)
        sources.append(str(group[0].get(source_label, "") or ""))
        remaining -= overhead + measure(chunk)

    # ---------- 第五步：拼装最终文本并对齐 sources ----------
    # 正文各段用分隔符连接；有 prefix 时把它放在最前，并与正文之间也补分隔符。
    # sources 对齐规则：prefix 有多少块，sources 前面就补多少个空串占位。
    context_text = CONTEXT_SEPARATOR.join(parts)
    if prefix_text:
        context_text = prefix_text + (
            CONTEXT_SEPARATOR + context_text if context_text else ""
        )
        sources = [""] * len(prefix_blocks or ()) + sources
    return context_text, sources


def extract_images(
    docs: Sequence[dict[str, Any]],
    images: Sequence[dict[str, Any]] = (),
    *,
    max_images: int = 5,
    url_fields: Iterable[str] = ("image_url", "url"),
) -> list[str]:
    """收集图片结果与文本文档中的图片 URL，保持顺序并去重。

    逻辑：先从图片检索结果（images）里按 url_fields 依次找 url，
    再从文本文档（docs）的 image_urls 字段里收集，两轮都保持原始顺序且
    只保留第一次出现的 URL；最后截断到 max_images 张。

    参数:
        docs: 文本文档列表，读取其 image_urls 字段。
        images: 图片检索结果列表；图片 URL 优先从这里收集。
        max_images: 返回的最大图片数量上限，默认 5。
        url_fields: 从图片结果中尝试读取 URL 的字段名顺序。

    返回:
        去重并按顺序截断后的图片 URL 列表。
    """
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
