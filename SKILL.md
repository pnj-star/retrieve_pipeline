---
name: rag_skill
description: 基于 common_core 组件实现的可复用 RAG 能力，提供租户隔离、混合检索、RRF 融合、交叉编码器精排与响应缓存。当 agent 或其他服务需要从知识库检索可追溯文档时，通过 rag_retrieve 调用；回答生成归属 agent 编排层。
---

# rag_skill

`rag_skill` 是检索专用 RAG 能力层，不包含 LangGraph 编排，也不包含业务话术。它接收 `query + tenant_id + kb_id + request_id`，内部完成缓存、混合检索、融合、精排、阈值判断，对外只暴露 `rag_retrieve`，返回原始文档与受预算约束的 `context_text`。回答生成与护栏归属 agent 编排层，`common_core.rag` 提供答案侧公共机制供 agent/多 agent 框架复用。

## 处理流程

```text
query + tenant_id + kb_id + request_id
→ 检索缓存检查（key: 规范化 query + 改写模式分桶，按 tenant/kb 隔离）
   ├─ 命中 → status=retrieved_cache，直接返回缓存中精排后达标文档
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
   └─ 有文档过阈值 → 只保留达标文档 → 回写检索缓存 → 生成受预算约束的 context_text → status=retrieved
→ 返回 docs + context_text（不做回答生成；生成由 agent 编排层负责）
```

## 文件布局

- `src/rag_skill/builder.py`: 从环境变量构建 runtime、provider、cache、reranker 等实例。
- `src/rag_skill/pipeline.py`: `RagPipeline` 统一执行改写、混合检索、RRF 融合、重排、阈值与检索缓存。
- `src/rag_skill/mcp.py`: MCP 工具入口，只暴露 `rag_retrieve`。
- `src/rag_skill/results.py`: `RetrieveResult` / `RetrieveStatus` 返回契约。
- `src/rag_skill/stages/`: 查询改写、重排、检索缓存等检索侧阶段。
- `tests/`: 使用 fake provider 的契约测试，不依赖外部服务。

## MCP 工具

`rag_skill` 对外只暴露 `rag_retrieve`，不做回答生成。必传参数：`query`、`tenant_id`、`kb_id`、`request_id`。

可选参数：`auth_token`、`session_id`、`user_id`、`collection_name`、`top_k`、`filter_expr`、`min_relevance`、`context_max_chars`、`context_max_tokens`、`max_doc_chars`、`max_doc_tokens`。

查询改写相关可选参数：

- `query_rewrite_mode`: 本次调用覆盖改写策略，可选 `off` / `identity` / `llm_rewrite` / `query_expansion`；不传则使用 `RETRIEVAL_QUERY_REWRITE_MODE`（及租户 / 知识库作用域覆盖）。
- `rewrite_query`: 显式传入改写后的查询文本；传了即跳过改写，直接用它检索。调用方可以先把改写做好，再交给 RAG。

注意：

- `context_max_chars` 控制拼入 `context_text` 的字符上限，默认 `8000`。
- `context_max_tokens` 控制 token 上限；`build_pipeline()` 默认自动注入 `build_token_counter()`（tiktoken 可用时按真实 token 计数，否则回退字符数），也可用 `build_pipeline(count_tokens=...)` 显式覆盖；`format_context(docs, max_tokens=...)` 未传 `count_tokens` 时同样自动使用默认计数器。
- `max_doc_chars` / `max_doc_tokens` 限制单篇文档（或同一父块合并后）的上限，不传时默认约为整体预算的一半。
- 启用 JWT 时 `auth_token` 中的 `tenant_id` / `kb_id` claims 必须与调用参数一致。

返回值契约（`rag_retrieve`）：

| 字段 | 说明 |
| --- | --- |
| `status` | 机器可读状态码，见下表 |
| `docs` | 精排后的文档；`no_context` 时返回候选文档 |
| `context_text` | 受预算约束、可直接喂给 agent LLM 的上下文文本 |
| `rewritten_query` | 本次实际用于检索的查询文本；改写关闭 / `identity` 时为原始 `query`；显式传 `rewrite_query` 时即为所传文本 |
| `cache_hit` | 是否命中检索缓存 |
| `message` | 状态的人类可读说明（如缓存命中 / no_context 原因） |

`status` 标准值（`rag_retrieve`）：

| status | 含义 | 调用方行为建议 |
| --- | --- | --- |
| `retrieved` | 正常检索并精排后达到阈值 | 直接使用 `docs` / `context_text` |
| `retrieved_cache` | 命中检索缓存（query → 精排后达标文档） | 直接复用 `docs` / `context_text` |
| `no_context` | 检索为空或没有文档过相关性阈值 | 可结合候选 `docs` 自行判断是否转人工或澄清用户问题 |
| `error` | 管线内部异常 | 视为失败，重试或上报 |

回答生成、护栏与低置信兜底决策均不属于本 skill 的职责：`rag_skill` 只返回
原始 `docs` + 预算化的 `context_text`，由调用方 agent（例如多 agent 框架中的
回答节点、评估节点）自行完成多来源融合、生成、校验与转人工。

## 可复用库：common_core.rag

回答侧公共机制已下沉到工作区顶层的 `common_core.rag`，供 skill 之外的
agent / 多 agent 框架复用（`rag_skill` 本身不内置 `common_core` 副本，运行时
通过 `pythonpath` / 启动脚本引用顶层 `common_core/src`）：

- `common_core.rag.assembly`: `build_context_text` / `clean_markdown` / `dedupe_docs` / `extract_images`。
- `common_core.rag.generation`: `generate_answer` / `stream_answer` / `GenerationConfig`。
- `common_core.rag.guard`: `guard_generation` / `evaluate_guard` / `GuardConfig`。

这些机制不属于检索管道本身，因此不再放在 `rag_skill` 源码里；需要回答生成时，
由多 agent 框架的节点从 `common_core.rag` 按需引用。

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
- `REDIS_HOST` / `REDIS_PORT` / `REDIS_PASSWORD` / `REDIS_KEY_PREFIX`: 检索缓存（`rag_retrieval`）使用的 Redis 配置。
- `METRICS_ENABLED` / `METRICS_PREFIX` / `METRICS_PORT`: Prometheus 指标。缓存命中、写入、跳过、失败统一记录到 `<prefix>_cache_results_total`，labels 为 `result` / `tenant_id` / `kb_id`。
- `AUTH_MODE` / `AUTH_JWT_SECRET`: 工具鉴权，默认 `jwt`。

一套可运行的占位符配置见仓库根目录的 `.env.example`（只含占位符，不含真实密钥）。

默认开关与成本：

- 检索缓存：默认开启（Redis，`rag_retrieval` 命名空间）。同 query + tenant/kb 命中直接返回精排后达标文档，跳过改写、混合检索与精排，是省成本的关键；文档更新后需主动失效（`RetrievalCache.delete` / 知识治理 hook）。本 skill 不做回答缓存，回答缓存归属 agent 编排层。
- 查询改写：默认 `off`。启用 `llm_rewrite` / `query_expansion` 会增加一次 LLM 调用（仅在检索缓存未命中后发生），只影响检索文本，不改变最终回答面向用户问题的语义。

## 使用示例

```python
from common_core.context import AgentContext
from rag_skill.builder import build_pipeline, build_runtime
from rag_skill.tokenization import build_token_counter

runtime = build_runtime()
pipeline = build_pipeline(runtime, count_tokens=build_token_counter("gpt-4o"))
ctx = AgentContext(tenant_id="merchant", kb_id="merchant_kb", request_id="req-1")
docs = await pipeline.retrieve("当前支持的售后流程是什么？", context=ctx)
print(docs)
```

## 边界

- 不包含 LangGraph：编排和业务节点由 `instances/*` 或外层 agent 完成。
- 不包含业务词表、话术、知识摄取策略：这些由实例配置注入。
- `common_core` 提供通用 provider、鉴权、上下文、指标；`rag_skill` 只做 RAG 语义。
