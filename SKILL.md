---
name: retrieve_skill
description: 从知识库检索可追溯文档的 RAG 检索能力。当 agent 需要基于知识库回答问题时，调用 rag_retrieve 获取达标文档用于生成回答；回答生成与护栏不属于本工具职责。提供租户隔离、混合检索、RRF 融合、交叉编码器精排与检索缓存。
---

# retrieve_skill

`rag_retrieve` 是面向 agent 的检索工具。传入用户查询与租户上下文，返回按父块聚合、预算化、达到相关性阈值的 `docs`，供你用于回答生成或多来源融合。回答生成、护栏与低置信兜底由调用方 agent 完成，本工具不做。

## 什么时候用

- 用户问题需要从知识库召回依据来回答时调用 `rag_retrieve`。
- 需要把检索结果与另一路数据（如 SQL 查询结果）融合成一版回答时，以返回的 `docs` 为依据。
- 不需要挨个手写嵌入、向量检索、精排与缓存；这些由本工具在单次调用内完成。

## 调用方式

必传参数：`query`、`tenant_id`、`kb_id`、`request_id`。

可选参数：`auth_token`、`session_id`、`user_id`、`collection_name`、`top_k`、`filter_expr`、`min_relevance`、`context_max_chars`、`max_doc_chars`。

查询改写相关可选参数：

- `query_rewrite_mode`: 本次调用覆盖改写策略。**由你（agent）根据用户问题的特征选择**，不传则使用服务端默认值：

  | 值 | 什么时候传 | 效果 |
  |---|---|---|
  | `"off"` | 用户问题简短明确、语义清晰（如"退货政策是什么"） | 不做任何改写，原始 query 直接检索 |
  | `"llm_rewrite"` | 用户问题口语化、有省略指代或多轮上下文（如"那个东西怎么弄"） | LLM 把口语改写成适合检索的规范查询 |
  | `"query_expansion"` | 用户一句话包含多个子问题、话题宽泛或表述模糊（如"最近有什么活动优惠之类的"） | LLM 生成多条不同角度的检索变体，分别检索后合并去重，提升召回 |

  **注意**：`query_expansion` 会多花 LLM 调用成本和延迟，只在确实需要扩展召回时使用。
- `rewrite_query`: 显式传入改写后的查询文本；传了即跳过改写，直接用它检索。你可以先把改写做好，再交给本工具。

鉴权：启用 JWT 时，远程 HTTP 调用用 `Authorization: Bearer <jwt>` 传递令牌（也可省略并用 `auth_token` 参数传同一个 JWT）。令牌中的 `tenant_id` / `kb_id` claims 必须与调用参数一致，否则调用失败。

## 处理流程

```text
query + tenant_id + kb_id + request_id
→ 检索缓存检查（key: 规范化 query + 改写模式分桶，按 tenant/kb 隔离）
   ├─ 命中 → status=retrieved_cache，直接返回缓存中精排后达标文档
   └─ 未命中 → 继续
→ 查询改写（默认 off，可选 llm_rewrite / query_expansion）
   └─ 启用 → 用改写后 query / 扩展变体继续，实际检索文本见 rewritten_query
→ 稀疏检索 + 稠密检索
   ├─ 两者都为空 → status=no_context，结束
   └─ 任一有结果 → 下一步
→ RRF 融合，得到候选文档
→ 交叉编码器精排，得到统一 relevance 分数
→ 与 RETRIEVAL_MIN_RELEVANCE 比较
   ├─ 没有文档过阈值 → status=no_context，结束
   └─ 有文档过阈值 → 只保留达标文档 → 按父块聚合（含 child_ids）→ 预算截断 → 回写检索缓存 → status=retrieved
```

返回的 `rewritten_query` 是本次实际用于检索的查询文本：改冖关闭（`off`）时为原始 `query`，显式传 `rewrite_query` 时为其本身。

## 返回值

```json
{
  "status": "retrieved",
  "ok": true,
  "docs": [],
  "rewritten_query": "用户问题",
  "cache_hit": false,
  "count": 0,
  "tenant_id": "t1",
  "kb_id": "kb1",
  "request_id": "req-1",
  "user_id": "",
  "trace_id": "",
  "message": ""
}
```

| 字段 | 说明 |
| --- | --- |
| `status` | 机器可读状态码（权威信号），见下表 `status` 取值 |
| `ok` | 是否无内部异常，等价于 `status != "error"` |
| `docs` | 精排后的达标文档，按父块聚合去重（`no_context` 时为候选项） |
| `rewritten_query` | 本次实际用于检索的查询文本 |
| `cache_hit` | 是否命中检索缓存，等价于 `status == "retrieved_cache"` |
| `count` | 达标文档数量，等价于 `len(docs)` |
| `tenant_id` / `kb_id` / `request_id` / `user_id` | 本次实际生效的调用上下文回显，用于确认鉴权作用域与日志/追踪关联 |
| `trace_id` | 本次调用在追踪后端中的关联标识，供跨链路排障 |
| `message` | 状态的人类可读说明（如缓存命中 / no_context 原因） |

`docs[*]` 字段：

| 字段 | 说明 |
| --- | --- |
| `id` | 父块主键（优先取 `parent_id`，缺失时回退到命中的子块 id）；据此回查父块/子块 |
| `child_ids` | 该父块下所有命中子块的主键列表；供核对子块级召回 |
| `parent_id` / `parent_title` | 父块主键 / 标题，可配（`MILVUS_OUTPUT_FIELDS`） |
| `content` | 父块全文（同一父块多个命中子块合并去重后的整段依据），已做预算截断 |
| `source` / `category` / `chunk_index` | 来源文件、分类、父块内首个命中子块序号 |
| `ce_score` | 交叉编码器精排原始分数，裁剪到 `[0,1]` |
| `score` | 精排后融合分 = `ce_weight*ce_score + retrieval_weight*retrieval_score`；聚合时取组内最高分 |
| `tenant_id` / `kb_id` | 命中文档所属作用域 |

## 按状态处理结果

| status | 含义 | 调用方行为建议 |
| --- | --- | --- |
| `retrieved` | 正常检索并精排后达到阈值 | 直接使用 `docs` 进入生成节点 |
| `retrieved_cache` | 命中检索缓存（query → 精排后达标文档） | 直接复用 `docs` |
| `no_context` | 检索为空或没有文档过相关性阈值 | `docs` 为候选；由你决定转人工、澄清问题或二次检索 |
| `error` | 管线内部异常 | 视为失败，重试或上报；用 `trace_id` / `request_id` 排障 |

需要多来源融合（如再查 MySQL/NL2SQL）时，以 `docs` 里的父块文档作为依据，与外部数据统一交给生成节点；`child_ids` 只用于回查核对，不作为生成正文本体重复拼入。

## 缓存与成本

- 检索缓存默认开启（Redis，`rag_retrieval` 命名空间）。同 query + tenant/kb 命中直接返回精排后达标文档，跳过改写、混合检索与精排，是省成本的关键。
- `no_context` 不写入缓存：无达标文档的结论不会缓存，知识库更新后的同类问题仍会重新检索。
- 本工具不做回答缓存；回答缓存归属调用方编排层。

## 常见参数说明

- `context_max_chars`: 合并后 `docs` 里正文的字符总预算，默认 `8000`。
- `max_doc_chars`: 单篇父块文档正文上限，不传时默认约为整体预算的一半。
- `filter_expr`: 附加的业务过滤表达式，会与租户 / 知识库隔离条件一起下发到向量库。

## 边界

- 只做检索，返回 `docs`，不做回答生成。
- 不包含 LangGraph 编排、业务词表、话术、知识摄取策略。
- 回答生成、护栏与低置信兜底决策由调用方 agent 完成；其中回答侧公共机制（上下文拼装、生成、护栏）见仓库顶层 `common_core.rag`，可按需引用。
