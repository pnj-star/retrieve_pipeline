# retrieve_skill

基于 `common_core` 组件的可复用 RAG 检索能力层。它不包含 LangGraph 编排，也不包含业务话术；只做检索语义，对外暴露检索工具 `rag_retrieve` 与进程内 `RagPipeline`。回答生成与护栏归属 agent 编排层，本仓库顶层 `common_core.rag` 提供回答侧公共机制。

本文件面向想复用 / 扩展 / 部署该 skill 的开发者。如果你只是要调用它做检索（agent / MCP 调用方），看 [SKILL.md](SKILL.md) 的调用契约即可。

## 能力

- 租户 / 知识库隔离的混合检索（稀疏 + 稠密）。
- RRF 融合、交叉编码器精排、相关性阈值判断。
- 检索缓存：完整检索签名 → 精排达标子块对应的父块引用；低于阈值、reranker 或父块存储失败不写缓存。
- 查询改写（`off` / `llm_rewrite` / `query_expansion`），也支持调用方显式传入已改写查询。
- 子块按 `parent_id` 去重聚合；MySQL 批量回源权威父块正文后做 token 预算截断。

## 文件布局

- `src/retrieve_skill/builder.py`: 从环境变量构建 runtime、provider、cache、reranker 等实例。
- `src/retrieve_skill/pipeline.py`: `RagPipeline` 统一执行改写、混合检索、RRF 融合、重排、阈值、检索缓存与父块聚合。
- `src/retrieve_skill/parent_store.py`: MySQL 父块批量读取组件。
- `src/retrieve_skill/parent_docs.py`: 父块引用构建、校验、版本检查与预算组装。
- `src/retrieve_skill/mcp.py`: MCP 工具入口，只暴露 `rag_retrieve`。
- `src/retrieve_skill/results.py`: `RetrieveResult` / `RetrieveStatus` 返回契约。
- `src/retrieve_skill/stages/`: 查询改写、重排、检索缓存等检索侧阶段。
- `tests/`: 使用 fake provider 的契约测试，不依赖外部服务。

## 安装

`retrieve_skill` 依赖 `common-core[llm,vector,cache]`；启用 MySQL 父块回源时还需要
`pymysql` 和 `DBUtils`。本地开发时从仓库内安装：

```powershell
cd D:\my_project\Skill\retrieve_skill
pip install -e ../common_core --no-deps
pip install -e . --no-deps
pip install ".[mysql]"
```

发布后改用私有源版本锁定，见 [docs/deployment.md](/D:/my_project/Skill/docs/deployment.md) 的依赖交付章节。

## 使用示例

```python
from common_core.context import AgentContext
from retrieve_skill.builder import build_pipeline, build_runtime

runtime = build_runtime()
pipeline = build_pipeline(runtime)
ctx = AgentContext(tenant_id="merchant", kb_id="merchant_kb", request_id="req-1")
result = await pipeline.retrieve_context("当前支持的售后流程是什么？", context=ctx)
print(result.status, result.docs)
```

## 配置

常用环境变量（前缀与 `common_core` 一致），一套可运行的占位符配置见仓库根目录的 `.env.example`（只含占位符，不含真实密钥）：

- `MILVUS_TEXT_COLLECTION` / `MILVUS_IMAGE_COLLECTION`: 文本 / 图片集合。
- `MILVUS_OUTPUT_FIELDS`: 普通 `retrieve()` 的返回字段。`retrieve_context()` 内部固定请求 `id,content,parent_id,chunk_index,tenant_id,kb_id,doc_version`；Milvus 只保存子块正文和定位字段，父块正文由 MySQL 回源。
- `MYSQL_HOST` / `MYSQL_PORT` / `MYSQL_DATABASE` / `MYSQL_USER` / `MYSQL_PASSWORD`: 父块权威库连接。
- `RAG_PARENT_TABLE` / `RAG_PARENT_STATUS`: 父块表名和可见状态过滤，默认 `rag_parent_block` / `active`。
- `CONTEXT_MAX_TOKENS` / `MAX_DOC_TOKENS`: 父块上下文总预算与单篇预算；调用方也可在每次请求覆盖。
- `RETRIEVAL_TOP_K` / `RETRIEVAL_MIN_RELEVANCE` / `RETRIEVAL_RRF_TOP_K` / `RETRIEVAL_RRF_K`: 检索与精排阈值参数。
- `RETRIEVAL_QUERY_REWRITE_MODE`: 查询改写策略，默认 `off`；可选 `off` / `llm_rewrite` / `query_expansion`。调用方显式传入 `rewrite_query` 时会跳过内部 LLM 改写。
- `RETRIEVAL_QUERY_REWRITE_LLM_MODEL`: 改写专用模型，为空则复用管线 LLM。
- `RETRIEVAL_QUERY_REWRITE_TEMPERATURE` / `RETRIEVAL_QUERY_REWRITE_MAX_TOKENS`: 改写调用参数。
- `RETRIEVAL_QUERY_REWRITE_EXPAND_COUNT`: `query_expansion` 生成的扩展变体数量，默认 `2`。
- `RETRIEVAL_QUERY_REWRITE_PROMPT`: 自定义单条改写 prompt（`llm_rewrite`）。
- `RETRIEVAL_QUERY_REWRITE_EXPANSION_PROMPT`: 自定义扩展 prompt，需含 `{count}` 占位符。
- `RETRIEVAL_QUERY_REWRITE_SCOPES`: 按作用域覆盖策略，格式 `t1/kb1=llm_rewrite,t2/*=query_expansion`；支持 `tenant/kb`、`tenant/*`、`*/kb`、`*/*`。
- `KNOWLEDGE_DATA_VERSION`: 参与检索缓存 v3 签名；知识数据批量更新后递增可避免旧结果命中。
- `REDIS_HOST` / `REDIS_PORT` / `REDIS_PASSWORD` / `REDIS_KEY_PREFIX`: 检索缓存使用的 Redis 配置。
- `METRICS_ENABLED` / `METRICS_PREFIX` / `METRICS_PORT`: Prometheus 指标。缓存命中、写入、跳过、失败统一记录到 `<prefix>_cache_results_total`，labels 为 `result` / `tenant_id` / `kb_id`。
- `AUTH_MODE` / `AUTH_JWT_SECRET`: 工具鉴权，默认 `jwt`。
- `AUTH_JWKS_URL`: 企业 IdP 的 JWKS 端点（如 Keycloak / Auth0 / Okta 的 `certs` URL）。配置后无需静态公钥，服务按 `AUTH_JWKS_LIFESPAN` 缓存公钥集合，遇到未知 `kid` 或缓存过期自动刷新，签名密钥轮换不需要重启 MCP 进程。

### 默认开关与成本

- 检索缓存：默认开启（Redis，`rag_retrieval_cache_v3` 命名空间）。key 覆盖查询、改写语义、召回参数、过滤条件、质量阈值、embedding/reranker 配置、parent store 标识和数据版本；Redis 只保存精排达标的父块引用，不保存父块正文。命中后仍回源 MySQL 校验版本并重建展示视图。文档更新后需主动失效（`RetrievalCache.delete` / 更新 `KNOWLEDGE_DATA_VERSION` / 递增 `doc_version`）。本 skill 不做回答缓存。
- 查询改写：默认 `off`。启用 `llm_rewrite` / `query_expansion` 会增加一次 LLM 调用（仅在检索缓存未命中后发生），只影响检索文本。

## 启动 MCP

```powershell
cd D:\my_project\Skill\retrieve_skill
python -m retrieve_skill.mcp --transport streamable-http --host 127.0.0.1 --port 8000
```

也支持 stdio / sse 传输。具体交付与环境加载方式见 [docs/deployment.md](/D:/my_project/Skill/docs/deployment.md)。

## 测试

```powershell
python -B -m pytest -q -p no:cacheprovider tests
```

## 回答侧复用：common_core.rag

回答生成的公共机制在仓库顶层 `common_core.rag`，供 skill 之外的 agent / 多 agent 框架复用：

- `common_core.rag.assembly`: `build_context_text` / `clean_markdown` / `dedupe_docs` / `extract_images`。
- `common_core.rag.generation`: `generate_answer` / `stream_answer` / `GenerationConfig`。
- `common_core.rag.guard`: `guard_generation` / `evaluate_guard` / `GuardConfig`。

## 边界

- 不包含 LangGraph：编排和业务节点由 `instances/*` 或外层 agent 完成。
- 不包含业务词表、话术、知识摄取策略：这些由实例配置注入。
- `common_core` 提供通用 provider、鉴权、上下文、指标；`retrieve_skill` 只做 RAG 检索语义。
