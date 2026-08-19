---
name: rag_skill
description: 基于 common_core 组件实现的可复用 RAG 能力，提供租户隔离、混合检索、RRF 融合、交叉编码器精排、生成护栏与响应缓存。当 agent 或其他服务需要从知识库检索并生成可追溯答案时，通过 rag_answer / rag_retrieve 调用。
---

# rag_skill

`rag_skill` 是通用 RAG 能力层，不包含 LangGraph 编排，也不包含业务话术。它接收 `query + tenant_id + kb_id + request_id`，内部完成缓存、混合检索、融合、精排、阈值判断、生成和护栏，最后返回稳定契约。

## 处理流程

```text
query + tenant_id + kb_id + request_id
→ 响应缓存检查
   ├─ 命中 → status=answered_cache，直接返回
   └─ 未命中 → 继续
→ 查询改写（默认 off，可选 identity / llm_rewrite / query_expansion）
   ├─ 关闭 → 直接使用原始 query
   └─ 启用 → 用改写后 query / 扩展变体继续，实际检索文本写入 rewritten_query
→ 稀疏检索 + 稠密检索（各取 top-k）
   ├─ 两者都为空 → status=no_context，结束
   └─ 任一有结果 → 下一步
→ RRF 融合，得到候选文档
→ 交叉编码器精排，得到统一 relevance 分数
→ 与 RETRIEVAL_MIN_RELEVANCE 比较
   ├─ 没有文档过阈值 → status=no_context，结束
   └─ 有文档过阈值 → 拼上下文 → 生成 → 护栏
→ status=answered + message + docs + answer
```

## 文件布局

- `src/rag_skill/builder.py`: 从环境变量构建 runtime、provider、cache、reranker、guard 等实例。
- `src/rag_skill/pipeline.py`: `RagPipeline` 统一执行检索、重排、生成、缓存和指标。
- `src/rag_skill/mcp.py`: MCP 工具入口 `rag_answer` / `rag_retrieve`。
- `src/rag_skill/results.py`: `RagResult` / `RagStatus` 返回契约。
- `src/rag_skill/stages/`: 缓存、上下文组装、生成、护栏、重排、人工交接等阶段。
- `tests/`: 使用 fake provider 的契约测试，不依赖外部服务。

## MCP 工具

`rag_answer` 必传参数：`query`、`tenant_id`、`kb_id`、`request_id`。

可选参数：`auth_token`、`session_id`、`user_id`、`system_prompt`、`top_k`、`empty_answer`、`filter_expr`、`min_relevance`、`enable_guard`、`prompt_template`、`context_max_chars`、`context_max_tokens`、`max_doc_chars`、`max_doc_tokens`、`temperature`、`max_tokens`。

查询改写相关可选参数（`rag_answer` 与 `rag_retrieve` 均支持）：

- `query_rewrite_mode`: 本次调用覆盖改写策略，可选 `off` / `identity` / `llm_rewrite` / `query_expansion`；不传则使用 `RETRIEVAL_QUERY_REWRITE_MODE`（及租户 / 知识库作用域覆盖）。
- `rewrite_query`: 显式传入改写后的查询文本；传了即跳过改写，直接用它检索。调用方可以先把改写做好，再交给 RAG。

注意：

- `prompt_template` 必须包含 `{context}`，可以使用 `{query}`；不传时使用默认模板。
- `system_prompt` 作为额外指令追加到生成提示，不替换默认模板。
- `context_max_chars` 控制拼入上下文的字符上限，默认 `8000`。
- `context_max_tokens` 控制 token 上限；`build_pipeline()` 默认自动注入 `build_token_counter()`（tiktoken 可用时按真实 token 计数，否则回退字符数），也可用 `build_pipeline(count_tokens=...)` 显式覆盖；`format_context(docs, max_tokens=...)` 未传 `count_tokens` 时同样自动使用默认计数器。
- `max_doc_chars` / `max_doc_tokens` 限制单篇文档（或同一父块合并后）的上限，不传时默认约为整体预算的一半。
- 启用 JWT 时 `auth_token` 中的 `tenant_id` / `kb_id` claims 必须与调用参数一致。

返回值契约（`rag_answer`）：

| 字段 | 说明 |
| --- | --- |
| `status` | `answered` / `answered_cache` / `no_context` / `guard_blocked` / `error` |
| `message` | 状态说明 |
| `docs` | 精排后的文档；`no_context` / `guard_blocked` 时返回候选文档 |
| `answer` | 生成或缓存的回答 |
| `rewritten_query` | 本次实际用于检索的查询文本；未启用改写时等于原始 `query` |

`rag_retrieve` 只做检索，返回 `status`（`retrieved` / `no_context`）、`count`、`docs`。

## 配置

常用环境变量（前缀与 `common_core` 一致）：

- `MILVUS_TEXT_COLLECTION` / `MILVUS_IMAGE_COLLECTION`: 文本 / 图片集合。
- `MILVUS_OUTPUT_FIELDS`: 检索返回的文本字段，逗号分隔。不配置时默认 `id,content,source,category,parent_content,parent_title,chunk_index,tenant_id,kb_id`；不同集合 schema 可通过该变量覆盖，避免查询报错。若集合含 `parent_id` 且希望按稳定父块身份去重，请通过该变量把 `parent_id` 加进输出字段。
- `RETRIEVAL_TOP_K` / `RETRIEVAL_MIN_RELEVANCE` / `RETRIEVAL_RRF_TOP_K` / `RETRIEVAL_RRF_K`: 检索与精排阈值参数。
- `RETRIEVAL_QUERY_REWRITE_MODE`: 查询改写策略，默认 `off`；可选 `off` / `identity` / `llm_rewrite` / `query_expansion`。
- `RETRIEVAL_QUERY_REWRITE_LLM_MODEL`: 改写专用模型，为空则复用管线 LLM。
- `RETRIEVAL_QUERY_REWRITE_TEMPERATURE` / `RETRIEVAL_QUERY_REWRITE_MAX_TOKENS`: 改写调用参数。
- `RETRIEVAL_QUERY_REWRITE_EXPAND_COUNT`: `query_expansion` 生成的扩展变体数量，默认 `2`。
- `RETRIEVAL_QUERY_REWRITE_PROMPT`: 自定义单条改写 prompt（`llm_rewrite`）。
- `RETRIEVAL_QUERY_REWRITE_EXPANSION_PROMPT`: 自定义扩展 prompt，需含 `{count}` 占位符。
- `RETRIEVAL_QUERY_REWRITE_SCOPES`: 按作用域覆盖策略，格式 `t1/kb1=llm_rewrite,t2/*=query_expansion`；支持 `tenant/kb`、`tenant/*`、`*/kb`、`*/*`。
- `REDIS_HOST` / `REDIS_PORT` / `REDIS_PASSWORD` / `REDIS_KEY_PREFIX`: 响应缓存。
- `METRICS_ENABLED` / `METRICS_PREFIX` / `METRICS_PORT`: Prometheus 指标。缓存命中、写入、跳过、失败统一记录到 `<prefix>_cache_results_total`，labels 为 `result` / `tenant_id` / `kb_id`。
- `AUTH_MODE` / `AUTH_JWT_SECRET`: 工具鉴权，默认 `jwt`。

一套可运行的占位符配置见仓库根目录的 `.env.example`（只含占位符，不含真实密钥）。

默认开关与成本：

- 守卫（生成护栏）：默认关闭。每次回答若开启护栏，都会在生成后再做一次 LLM 评审，不合格最多重试 2 次，延迟和 token 成本显著上升；多 agent 工具场景建议按需 `enable_guard=True` 或 `build_pipeline(guard_config=GuardConfig())` 显式开启。
- 响应缓存：默认开启（Redis）。同 query + tenant/kb 命中直接返回，是省成本的关键；需要最新检索时可显式关闭或调低命中优先级。
- 查询改写：默认 `off`。启用 `llm_rewrite` / `query_expansion` 会增加一次 LLM 调用，只影响检索文本，不改变最终回答面向用户问题的语义。

## 使用示例

```python
from common_core.context import AgentContext
from rag_skill.builder import build_pipeline, build_runtime
from rag_skill.tokenization import build_token_counter

runtime = build_runtime()
pipeline = build_pipeline(runtime, count_tokens=build_token_counter("gpt-4o"))
ctx = AgentContext(tenant_id="merchant", kb_id="merchant_kb", request_id="req-1")
result = await pipeline.answer("当前支持的售后流程是什么？", context=ctx)
print(result.status, result.answer)
```

## 边界

- 不包含 LangGraph：编排和业务节点由 `instances/*` 或外层 agent 完成。
- 不包含业务词表、话术、知识摄取策略：这些由实例配置注入。
- `common_core` 提供通用 provider、鉴权、上下文、指标；`rag_skill` 只做 RAG 语义。
