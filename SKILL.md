---
name: retrieve_skill
description: 从知识库检索可追溯文档的 RAG 检索能力。当 agent 需要基于知识库回答问题时，调用 rag_retrieve 获取达标父块文档用于生成回答；回答生成与护栏不属于本工具职责。提供租户隔离、混合检索、RRF 融合、交叉编码器精排、MySQL 父块回源与检索缓存。
---

# retrieve_skill

`rag_retrieve` 是面向 agent 的检索工具。传入用户查询与租户上下文，返回按父块聚合、预算化、达到相关性阈值的 `docs`；父块正文来自权威 MySQL 存储，供你用于回答生成或多来源融合。回答生成、护栏与低置信兜底由调用方 agent 完成，本工具不做。

## 什么时候用

- 用户问题需要从知识库召回依据来回答时调用 `rag_retrieve`。
- 需要把检索结果与另一路数据（如 SQL 查询结果）融合成一版回答时，以返回的 `docs` 为依据。
- 不需要挨个手写嵌入、向量检索、精排与缓存；这些由本工具在单次调用内完成。

## 调用方式

必传参数：`query`、`tenant_id`、`kb_id`、`request_id`。

可选参数：`auth_token`、`session_id`、`user_id`、`collection_name`、`top_k`、`filter_expr`、`min_relevance`、`context_max_tokens`、`max_doc_tokens`、`context_max_chars`、`max_doc_chars`。

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
→ 检索缓存检查（v3 完整检索签名，父块引用结果；tenant/kb 由 Redis key 外层隔离）
   ├─ 命中 → 校验父块引用与阈值 → MySQL 回源校验 → token/字符预算重建 → status=retrieved_cache
   └─ 未命中 → 继续
→ 查询改写（默认 off，可选 llm_rewrite / query_expansion）
   └─ 启用 → 用改写后 query / 扩展变体继续，实际检索文本见 rewritten_query
→ 稀疏检索 + 稠密检索
   ├─ 两者都为空 → status=no_context，结束
   └─ 任一有结果 → 下一步
→ RRF 融合，得到候选文档
→ 交叉编码器精排，得到统一 relevance 分数
   └─ 精排器故障 → status=error，docs=[]，不写缓存
→ 与 RETRIEVAL_MIN_RELEVANCE 比较
   ├─ 没有子块过阈值 → status=no_context，docs=[]，不写缓存
   └─ 有子块过阈值 → 子块去重 → 按 parent_id 去重并构建引用
      → MySQL 批量回源父块 → 版本/状态校验 → 父块引用回写 Redis
      → token/字符预算截断 → status=retrieved
```

返回的 `rewritten_query` 是本次实际用于检索的查询文本：改写关闭（`off`）时为原始 `query`，显式传 `rewrite_query` 时为其本身。

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
  "diagnostics": {},
  "message": ""
}
```

| 字段 | 说明 |
| --- | --- |
| `status` | 机器可读状态码（权威信号），见下表 `status` 取值 |
| `ok` | 是否无内部异常，等价于 `status != "error"` |
| `docs` | 精排后达标且 MySQL 当前可用的父块上下文；`no_context` / `error` 时为空 |
| `rewritten_query` | 本次实际用于检索的查询文本 |
| `cache_hit` | 是否命中检索缓存，等价于 `status == "retrieved_cache"` |
| `count` | 达标文档数量，等价于 `len(docs)` |
| `tenant_id` / `kb_id` / `request_id` / `user_id` | 本次实际生效的调用上下文回显，用于确认鉴权作用域与日志/追踪关联 |
| `trace_id` | 本次调用在追踪后端中的关联标识，供跨链路排障 |
| `diagnostics` | 不含候选正文的检索摘要：原因、各级数量、缺失父块/版本冲突数、token 预算等 |
| `message` | 状态的人类可读说明（如缓存命中 / no_context 原因） |

`docs[*]` 字段：

| 字段 | 说明 |
| --- | --- |
| `id` | 父块主键；来自 Milvus 子块 metadata 中的 `parent_id` |
| `child_ids` | 该父块下所有命中子块的主键列表；供核对子块级召回 |
| `parent_id` / `parent_title` | 父块主键 / MySQL 中的标题 |
| `content` | MySQL 权威父块正文，已做 token/字符预算截断 |
| `source` / `source_type` / `category` | MySQL 中保存的来源信息 |
| `doc_version` | 命中时 MySQL 父块的当前版本 |
| `ce_score` | 该父块下达标子块的最高交叉编码器分数 |
| `score` | 对应最高分子块的融合检索分 |
| `tenant_id` / `kb_id` | 命中文档所属作用域 |

## 按状态处理结果

| status | 含义 | 调用方行为建议 |
| --- | --- | --- |
| `retrieved` | 正常检索并精排后达到阈值 | 直接使用 `docs` 进入生成节点 |
| `retrieved_cache` | 命中合格父块引用缓存并回源重建视图 | 直接复用 `docs` |
| `no_context` | 检索为空或没有文档过相关性阈值 | `docs` 为空；由你决定转人工、澄清问题或二次检索 |
| `error` | 管线内部异常 | 视为失败，重试或上报；用 `trace_id` / `request_id` 排障 |

需要多来源融合（如再查 MySQL/NL2SQL）时，以 `docs` 里的父块文档作为依据，与外部数据统一交给生成节点；`child_ids` 只用于回查核对，不作为生成正文本体重复拼入。

## 缓存与成本

- 检索缓存默认开启（Redis，`rag_retrieval_cache_v3` 命名空间）。key 覆盖查询、改写语义、集合、top-k/RRF、过滤、阈值、embedding/reranker 配置、parent store 标识和数据版本；命中后跳过改写、混合检索与精排，但仍回源 MySQL。
- Redis 缓存的是精排后达标的**父块引用**：`tenant_id`、`kb_id`、`parent_id`、`child_ids`、`ce_score`、`score`、`doc_version`，不保存父块正文。每次命中都会用 MySQL 当前正文和请求预算重建最终视图。
- `no_context` 不写入缓存：无达标文档的结论不会缓存，知识库更新后的同类问题仍会重新检索。
- 本工具不做回答缓存；回答缓存归属调用方编排层。

## 常见参数说明

- `context_max_tokens`: 所有父块正文的 token 总预算，默认 `6000`。
- `max_doc_tokens`: 单篇父块正文 token 上限，不传时默认约为总预算一半。
- `context_max_chars` / `max_doc_chars`: 可选字符预算；显式传入时会与 token 预算同时生效。
- `filter_expr`: 附加的业务过滤表达式，会与租户 / 知识库隔离条件一起下发到向量库。

## 边界

- 只做检索，返回 `docs`，不做回答生成。
- 不包含 LangGraph 编排、业务词表、话术、知识摄取策略。
- 回答生成、护栏与低置信兜底决策由调用方 agent 完成；其中回答侧公共机制（上下文拼装、生成、护栏）见仓库顶层 `common_core.rag`，可按需引用。
