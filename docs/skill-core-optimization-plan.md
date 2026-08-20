# rag_skill 核心优化计划书（skill 内部）

> 版本：v1.4  
> 日期：2026-08-18  
> 状态：已实施  
> 定位：只优化 `rag_skill` 自身可复用能力，不把多 Agent 编排、评测、可观测、合规、模型服务化等宿主平台事项塞进 skill。

> 当前状态说明：本计划为历史文档。`rag_skill` 现已收敛为检索专用，MCP 只暴露
> `rag_retrieve`；上下文组装、生成、护栏已下沉到工作区顶层的 `common_core.rag`，
> 供 agent / 多 agent 框架复用（`rag_skill` 不再内置 `common_core` 副本）。文中
> `rag_answer` 与 skill 内置的生成/护栏改动视为已废除，请以 `SKILL.md` 与 `common_core.rag` 为准。

## 1. 背景与目标

目标：让 `rag_skill` 成为边界清晰、接口稳定、预算安全、身份可靠的可复用 RAG 能力包。

1. 任何调用方（MCP、Python API、业务节点）都能稳定组装上下文；
2. 检索结果的去重身份稳定，不靠标题猜；
3. 预算控制既支持字符，也支持真实 token；
4. 不再出现"第一个大文档撑爆阈值导致空上下文 / 独占预算 / 单块关键召回被丢弃"的问题。

## 2. 现状结论

本轮优化范围（`rag_skill` 内部）已完成：

- 去重身份改为 `parent_id` 优先，保留显式 `dedupe_key` 覆盖能力；
- token 预算端到端接入 `build_context_text` -> `RagPipeline.answer()` -> MCP；
- 单文档 / 单父块组有独立字符与 token 上限，不再独占整体预算；
- 父子块命中时按 `chunk_index` 合并，未命中子块也有完整可读上下文；
- 默认 `MILVUS_OUTPUT_FIELDS` 保持不变，不写死 `parent_id`，避免 schema 不含该字段的现有集合查询报错；接入方按需通过该变量补充。

边界（本轮不做）：ingestion 侧需稳定写入 `parent_id + chunk_index`；评测、链路追踪、知识治理等宿主平台事项不在 skill 内。

## 3. 本轮明确不做

- LangGraph / 业务节点编排（`instances/*` 与宿主框架职责）；
- RAG 评测 / golden set / RAGAS（属于独立 `eval_skill` 或评测平台）；
- OpenTelemetry 链路追踪、审计日志、IAM/SSO、PII 全链路脱敏；
- 向量库 HA、限流、熔断、embedding / reranker 推理服务化；
- 知识生命周期、版本管理、缓存主动失效（属于 ingest / 知识治理层）。

## 4. 优化项 A：稳定去重身份（P0）

### 目标

`dedupe_docs` 不再把标题作为第一身份；有稳定父块 ID 时优先用 ID。

### 改动

- `assembly.py` 新增 `_stable_parent_key(doc)`，默认 key 优先级改为：
  1. `parent_id` 存在时：`tenant_id + kb_id + source + parent_id`；
  2. 否则 `tenant_id + kb_id + source + parent_title`；
  3. 否则 `tenant_id + kb_id + source + content[:60]`。
- 保留 `dedupe_key` 参数，显式传入时优先级最高；
- 结果保持输入顺序（即精排后的顺序），同名但不同父块的文档不再互相误去重；
- 默认检索输出字段不新增 `parent_id`；集合 schema 含该字段时，由 `MILVUS_OUTPUT_FIELDS` 显式补充。

### 示例

```text
doc1: parent_id=p1, title=政策 A
doc2: parent_id=p1, title=政策 B   -> 同一父块，只保留先到者
doc3: parent_id=p2, title=政策 A   -> 保留（同名但不同父块）
doc4: 无 parent_id, source=a.md, title=常见问题 -> 按 a.md + 常见问题 去重
```

### 测试

- 同 `parent_id` 去重；
- 同 `parent_title`、不同 `parent_id` 都保留；
- 无 ID 时回退 `source + parent_title`；
- 显式 `dedupe_key` 仍然优先。

## 5. 优化项 B：token 预算端到端（P0）

### 目标

让 `max_tokens` 从底层函数一路到 `RagPipeline.answer()` 和 MCP 工具，并补上单文档 token 上限。

### 改动

- `build_context_text` 增加 `max_doc_tokens: int | None = None`；
  - token 模式（`max_tokens + count_tokens`）下，默认单文档上限为 `max_tokens // 2`；
  - 字符模式维持 `max_doc_chars or max_chars // 2`。
- `RagPipeline.__init__` 增加可选 `count_tokens: Callable[[str], int] | None`，作为默认 token 计数注入点；
- `RagPipeline.answer()` / `_answer_impl()` 增加：
  - `context_max_tokens`
  - `count_tokens`（按调用覆盖）
  - `max_doc_chars`
  - `max_doc_tokens`
- `format_context()` 签名与 `build_context_text` 预算参数对齐，转发 `max_chars`、`max_tokens`、`max_doc_*`、`prefix_blocks` 等控制参数；
- `mcp.rag_answer` 增加 `context_max_tokens`、`max_doc_chars`、`max_doc_tokens`；`count_tokens` 不可序列化，由 `build_pipeline()` 默认注入 `build_token_counter()`，也可显式覆盖；
- 新增 `rag_skill/tokenization.py`：`build_token_counter()` 优先使用 tiktoken 按模型计数，缺失时回退字符数；`make_token_counter()` 支持注入自定义 encoding；
- 如果调用方传了 `context_max_tokens` 但 pipeline 没有计数器：记录 warning 并回退字符预算，不做"伪 token"。

### 测试

- pipeline 层把 token 参数透传给 `build_context_text`；
- MCP 层转发 `context_max_tokens` 等预算参数；
- token 模式下默认单文档上限避免第一篇吃光预算；
- 默认 token 计数器：tiktoken 可用时返回真实 token 数，不可用时回退字符数；
- 不传 `count_tokens` 时自动注入默认计数器：tiktoken 可用则真实 token 计数，否则回退字符。

## 6. 优化项 C：父子块命中窗口扩展（P1，本轮）

### 目标

解决"一个父块只喂第一个子块可能丢上下文"的幻觉风险；有 `parent_content` 时直接用父块全文，缺失时再按 `chunk_index` 合并子块。

### 改动

- `_group_docs()` 按稳定父块身份分组，并保留输入（精排后）顺序；
- `_render_group()` 有 `parent_content` 时直接用父块全文；没有时按 `chunk_index` 排序并合并子块，默认输出类似 `[Source N]（父标题）chunk1 ... chunk2 ...`；
- 整组仍受单文档 / 单组上限与总预算约束，超长时截断而不是丢弃；
- 来源仍取该组第一条文档的 source；
- 该改动需要 ingestion 侧稳定写入 `parent_id + chunk_index`（schema 契约）。

### 测试

- 同父块子块按 `chunk_index` 排序合并、来源唯一；
- `parent_content` 存在时优先全文，不重复拼子块；
- 无 `parent_id` 时回退行为不变。

## 7. 文件改动清单

| 文件 | 改动 |
| --- | --- |
| `rag_skill/src/rag_skill/stages/assembly.py` | 稳定身份、父子块分组、字符 / token 单文档上限 |
| `rag_skill/src/rag_skill/pipeline.py` | 透传 token / 单文档参数、`count_tokens` 注入 |
| `rag_skill/src/rag_skill/builder.py` | `build_pipeline()` 默认注入 `build_token_counter()` |
| `rag_skill/src/rag_skill/tokenization.py`（新增） | tiktoken 封装与字符回退 |
| `rag_skill/src/rag_skill/mcp.py` | `rag_answer` 暴露新预算参数 |
| `rag_skill/SKILL.md` | 参数与默认值说明 |
| `rag_skill/tests/test_stages.py` | 去重与 token 单文档上限用例 |
| `rag_skill/tests/test_pipeline.py` | pipeline 透传用例 |
| `rag_skill/tests/test_rag_mcp.py` | MCP 参数转发用例 |
| `rag_skill/tests/test_tokenization.py`（新增） | token 计数器封装用例 |

## 8. 验收标准

- [x] 全仓 `pytest` 通过；
- [x] 现有调用方式不破坏：`build_context_text(docs, max_chars=...)`、`pipeline.answer(...)`、MCP 原有参数均可继续使用；
- [x] 同名不同 `parent_id` 的文档不会互相误去重；
- [x] `context_max_tokens` 在 Python API 与 MCP 均可生效；
- [x] 超长首文档、超长 prefix、token 单文档上限均有单测覆盖；
- [x] 文档中的参数默认值与代码一致。

## 9. 建议排期

| 步骤 | 内容 | 预估 |
| --- | --- | --- |
| 1 | A：稳定去重身份 + 单测 | 0.5 人日 |
| 2 | B：token 预算接入 pipeline / MCP + 单测 | 0.5 - 1 人日 |
| 3 | 文档同步 + 全仓回归 | 0.25 - 0.5 人日 |
| 合计 |  | 1.25 - 2 人日 |

## 10. 执行顺序建议

1. 先做 A（纯 `assembly.py`，风险最小）；
2. 再做 B（改接口与 MCP，需要同步 SKILL.md）；
3. 再做 C（父子块合并，依赖 A 提供的稳定分组身份）。

> 本轮 A + B + C 的目标是让 `rag_skill` 具备"预算安全 + 身份稳定 + 父子块可读"的完整上下文组装能力；更远期的 score-aware 选择、查询改写、评测回归都不属于本次交付。
