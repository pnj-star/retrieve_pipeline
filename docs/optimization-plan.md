# rag_skill 企业级优化计划书

> 版本：v1.0　日期：2026-08-18　状态：待评审
> 适用范围：`common_core`（基础能力）+ `rag_skill`（查询侧 RAG）+ 未来 `knowledge_ingest_skill` / `eval_skill` + `instances/merchant`（首个消费实例）

---

## 一、现状评估（已对代码逐行核实）

### 1.1 已完成的能力

| 领域 | 现状 | 核实依据 |
| --- | --- | --- |
| 分层架构 | `common_core`（配置/鉴权/上下文/指标/provider）→ `rag_skill`（检索/重排/组装/生成/护栏/交接/缓存）→ `instances/*`（业务编排）三层分离 | `Skill/migration-plan.md` 分层边界表，实测代码一致 |
| 检索 | 混合检索 + RRF 融合 + 交叉编码器精排（bge-reranker-base） | `common_core/providers/vector.py`、`rag_skill/stages/rerank.py` |
| 租户隔离 | `ToolContextGuard` JWT claims 校验 + 检索侧强制注入 `tenant_id/kb_id` filter，fail-closed | `common_core/mcp_auth.py:48`、`providers/vector.py:256` |
| 可观测性 | Prometheus Counter/Histogram 全覆盖（runs/node/guard/cache/rerank/handoff/llm tokens），tenant/kb 标签，失败静默 | `common_core/observability.py` 实测 |
| 护栏 | 绝对化词表 + 数字溯源 + LLM 评审重试（fail-closed） | `rag_skill/stages/guard.py` |
| 降级 | 缓存失败降级、混合检索单路失败降级、本地 reranker 不可用透传、LLM/Redis 有超时 | `rag_skill/stages/rerank.py:87`、`providers/vector.py` |
| 安全原语 | `mask_pii`（邮箱/身份证/银行卡/手机）、`check_safety`（normalize/敏感词/注入模式）**已实现但未接进 rag_skill 查询侧管线** | `common_core/security.py` |
| 测试 | common_core 35 个 + rag_skill 49 个，全绿（2026-08-18 实测） | `python -m pytest` 结果 |

### 1.2 已确认的缺口（按优先级）

| 优先级 | 缺口 | 说明 |
| --- | --- | --- |
| **P0** | 无评测体系 | 无 golden set、无 Recall@k/MRR、无 RAGAS 类生成指标、无调参回归机制。`min_relevance=0.70` 是无数据支撑的默认值（可由 `RETRIEVAL_MIN_RELEVANCE` 覆盖） |
| **P0** | 无端到端链路追踪 | 有 Prometheus 打点，但无 OTel trace/span；`request_id` 只是日志与 metrics 字段，未跨进程贯穿 agent→MCP→管线 |
| **P1** | 知识治理缺失 | 纯查询侧。无 ingest、无文档版本/过期/发布状态、无"知识更新→缓存失效"机制 |
| **P1** | 安全能力未接线 | 鉴权/租户隔离已有，但 `security.py` 原语未在查询侧生效；无 IAM/SSO 集成、无审计日志、无全链路 PII 脱敏 |
| **P1** | 可靠性 | 本地加载 embedding/reranker 属单机玩法；无推理服务化、向量库 HA、限流、熔断、健康检查 |
| **P2** | 权限粒度 | 仅 tenant/kb 级隔离，无 chunk 级 ACL |
| **P3** | 部署文档与依赖矩阵 | GitHub 仓库仅含 rag_skill；`common_core` 未发布 PyPI，clone 后无法直接 `pip install`；无部署运行手册 |

> 注：README 中 "22 tests green" 指 merchant 实例迁移期测试，不是本次验证结果；本次实测为 common_core 35 + rag_skill 49。

---

## 二、目标状态

做一个"可以交付给企业内部其他团队使用的 RAG 能力平台"，而非单机自用：

1. **可证明**：任何调参（reranker 模型、min_relevance、top_k）都有评测数据支撑；
2. **可观测**：一次问答从 agent 调用到回答端到端可追踪（trace + 指标 + 日志三合一）；
3. **可治理**：知识有生命周期（入库→发布→失效），缓存随知识变更自动失效；
4. **可信**：安全原语在管线内全面生效，具备审计能力；
5. **可运维**：有部署手册、依赖矩阵、健康检查，组件可独立部署扩容。

---

## 三、路线图总览

| 阶段 | 主题 | 交付物 | 预估工期 |
| --- | --- | --- | --- |
| P0-1 | RAG 评测体系 | eval 工具包 + golden set 骨架 + 指标脚本 + 回归门禁 | 5–8 人日 |
| P0-2 | 端到端链路追踪 | OTel 基础件 + 跨进程传播 + 默认 dashboard | 4–6 人日 |
| P1-1 | 知识治理 | ingest skill 骨架 + 版本/发布状态 + 缓存失效 hook | 8–12 人日 |
| P1-2 | 安全合规接线 | security 原语接入管线 + 审计日志 + PII 全链路脱敏 | 4–6 人日 |
| P1-3 | 可靠性 | 推理服务化 + 限流 + 熔断 + 健康检查 | 6–10 人日 |
| P2 | 权限粒度 | chunk 级 ACL + 检索过滤扩展 | 3–5 人日 |
| P3 | 交付工程 | 部署手册 + 依赖矩阵 + 仓库自包含改造 | 2–4 人日 |

**依赖关系**：P0-1 与 P0-2 互相独立可并行；P1 各项依赖 P0 完成（评测与追踪是衡量 P1 效果的前提）；P2 依赖 P1-1（ACL 需要 ingest 侧授标）。

---

## 四、P0-1 评测体系（最高优先）

> 定位：单独一个 `eval_skill`，与迁移计划中 `未来 eval_skill` 对齐。

### 4.1 目标

- 建立 20→50→200 条可扩展的 golden set，覆盖检索、生成、护栏、安全四类断言；
- 产出检索指标（Recall@k / MRR / 命中率）与生成指标（faithfulness / answer relevance，用 RAGAS 或自研 judge）；
- 每次调参（reranker 模型、`min_relevance`、top_k）自动跑回归并出对比报告；
- **统计不同 reranker 模型的分数分布**，解构 `0.70` 阈值的模型相关性（bge-reranker-base 与更大模型分数语义不同，阈值不可直接迁移）。

### 4.2 交付物与文件布局（建议新增于 monorepo 顶层）

```text
Skill/
├─ eval_skill/
│  ├─ pyproject.toml
│  └─ src/eval_skill/
│     ├─ golden_set.yaml          # 黄金数据集（标注了期望检索 doc 与期望回答要点）
│     ├─ metrics.py               # Recall@k、MRR、命中率
│     ├─ generate_metrics.py      # faithfulness / answer relevance 的 judge 实现（RAGAS 或内置）
│     ├─ score_distribution.py    # 各模型 rerank ce_score 分布统计（服务 0.70 调参）
│     ├─ run_pipeline_eval.py     # 注入 fake/真实 provider 跑一次完整的检索+生成+护栏评测
│     ├─ comparable_report.py     # 两次评测对比（阈值/模型 A/B），输出 md 报告
│     └─ gate.py                  # 回归门禁：指标劣化超阈值则 CI 失败
```

### 4.3 实施步骤

1. **建 golden set（第 1 天即可开始）**：
   ```yaml
   # golden_set.yaml 结构示意
   dataset:
     - id: kb-001
       query: "当前支持的售后流程是什么？"
       kb_id: merchant_kb
       expected_retrieval: ["chunk_售后_01"]   # 期望检索命中的 chunk
       expected_answer_keywords: ["退货", "7天"]
       assert_no_absolute_words: true          # 断言无绝对化用语
       min_ce_score: 0.70                        # 期望精排最低分（若配置了 reranker）
   ```
   来源：从 merchant 实例的历史线上问题 + 业务痛点人工标注；条目按领域分桶（售后/结算/规则/兜底）。
2. **实现检索指标**：`metrics.py` 对 golden set 跑 `rag_retrieve`，计算 Recall@k（期望 chunk 是否在 top-k）与 MRR。
3. **实现生成指标**：`generate_metrics.py` 用 judge LLM 按 RAGAS 标准打分（faithfulness=回答事实是否都可溯源到上下文；answer relevance=回答是否切题），输出 0–1 分。
4. **实现分数分布统计**：`score_distribution.py` 收集同一批 golden query 在不同 reranker 模型下的 ce_score 分布，输出分位数表与直方图数据，用于校准 `min_relevance`。
5. **实现对比与门禁**：`comparable_report.py` 对比 baseline vs 候选配置；`gate.py` 设定阈值（如 faithfulness 下降 >0.05 即失败），接入 CI（见 §4.4）。
6. **首批只评估 rag_skill 主路径**：`answer`（缓存命中/检索空/精排拒绝/护栏拦截/正常生成）五条路径各建断言。

### 4.4 验收标准（Definition of Done）

- [ ] `eval_skill` 可一条命令跑完整评测：`python -m eval_skill.run_pipeline_eval --golden golden_set.yaml`
- [ ] 输出含 Recall@k、MRR、faithfulness、answer relevance 四项指标
- [ ] 产出 `score_distribution` 报告：至少对比当前模型与一个候选模型
- [ ] 门禁脚本接入 CI；任何调参 PR 必须附评测对比报告
- [ ] 至少 30 条 golden set 通过人工复核

### 4.5 风险

- golden set 标注质量决定评测可信度 → 首轮由业务方人工复核；
- judge LLM 打分有主观性 → 用多 judge 均值 + 低温度（复用 `GuardConfig.temperature=0.0` 思路）。

---

## 五、P0-2 端到端链路追踪

### 5.1 目标

- 引入 OpenTelemetry（OTel），让 `request_id` 成为 trace/span 的贯穿标识；
- 覆盖跨进程边界：agent → MCP（HTTP/SSE/stdio）→ pipeline 各 node（retrieve/rerank/guard/generate）→ 缓存/provider；
- 指标、日志、trace 三通道用同一 `request_id trace_id` 可关联。

### 5.2 交付物（落在 common_core，供所有 skill 复用）

```text
common_core/src/common_core/
├─ telemetry.py      # OTel provider 初始化、Tracer 单例、导出配置（OTLP exporter）
└─ instrumentation.py# 装饰器：span + 自动记录 node 名/tenant/kb/耗时/错误
```

新增依赖（common_core 的 pyproject）：`opentelemetry-sdk`、`opentelemetry-exporter-otlp`、`opentelemetry-instrumentation-fastapi`（如需 HTTP 出口）。

### 5.3 实施步骤

1. `telemetry.py`：按 `OTEL_EXPORTER_OTLP_ENDPOINT` / `OTEL_SERVICE_NAME` 环境变量初始化 provider，默认带 fallback（无 exporter 时不阻塞）。
2. `instrumentation.py`：提供 `trace_node(node, tenant_id, kb_id)` 上下文管理器，在 pipeline 的 retrieve/rerank/guard/generate 调用点包裹（复用现有 `_observe` 的调用点，不重复打点）。
3. **跨进程传播**（关键）：
   - MCP 层在 `rag_answer`/`rag_retrieve` 入口读取 `traceparent`（从 header / MCP 请求 metadata / JWT claim），设置当前 trace context；
   - agent 侧（merchant LangGraph）在调用 MCP 时注入 `traceparent`；
   - `request_id` 直接对齐为 trace 的 `trace_id` 低 64 位或作为 span attribute 记录。
4. 把 `request_id` 写入每个 span attribute；日志上下文（logging contextvars）同时注入 trace_id，保证 grep 日志能对上 trace。
5. 默认提供一张 Grafana/CloudWatch 示例 dashboard 定义（p95 延迟、错误率、各 node 耗时分布、护栏拦截率、缓存命中率）。

### 5.4 验收标准

- [ ] 一次完整 `rag_answer` 在 Jaeger/Tempo 中可见完整 trace：agent→MCP→retrieve→rerank→guard→generate
- [ ] 跨进程调用（agent→MCP）trace 不断链
- [ ] 任一日志行可按 trace_id/request_id 反查整条链路
- [ ] 无 OTLP 端点时管线正常运行（追踪不引入硬失败）

### 5.5 风险与注意

- 跨进程传播最容易漏的是 stdio 传输的 MCP（无 header）→ 通过 JWT claim 或 N-请求信封携带 traceparent；
- span 数量要克制（对缓存命中这种高频快路径可跳过，避免采样开销）。

---

## 六、P1-1 知识治理（ingest + 缓存失效）

### 6.1 目标

- 新增 `knowledge_ingest_skill`（与迁移计划命名对齐）：通用入库、版本、回滚、发布状态；
- 建立"知识变更 → 响应缓存失效"的解耦机制（不做业务词，只做机制）。

### 6.2 交付物

```text
Skill/knowledge_ingest_skill/
├─ src/knowledge_ingest_skill/
│  ├─ documents.py     # 文档/分块模型：doc_id, version, published, archived_at
│  ├─ ingest.py        # 通用入库管道（chunk → embedding → Milvus upsert）
│  ├─ versioning.py    # 版本生成、回滚（切换 published 指针）
│  └─ invalidation.py  # 变更事件 → 缓存失效 + 可选检索索引失效
```

`rag_skill` 侧变更：

```text
rag_skill/src/rag_skill/
└─ hooks.py           # 注册钩子：on_kb_updated(kb_id) 时批量删除该 kb 的缓存
```

### 6.3 实施步骤

1. **invalidation hook（先做，改动小收益大）**：`rag_skill/hooks.py` 暴露 `register_kb_updated_handler`；在 `instances/merchant` 的 ingest/更新流程里调用，触发 `ResponseCache.delete_by_kb(kb_id)`（在 `ResponseCache` 增加按 kb 前缀批量删除，基于现有 `_key` 命名空间）。
2. **documents.py**：定义文档生命周期状态机 `draft → published → archived`；collection schema 增加 `doc_id/version/published_at/archived` 字段（注意 `MILVUS_OUTPUT_FIELDS` 默认字段表需同步）。
3. **ingest.py**：实现 upsert 语义（同一 doc_id+version 覆盖）；失败事务回滚（同一批要么全进要么不进）。
4. **versioning.py**：发布/回滚只动状态字段不物理删数据，检索侧默认过滤 `published=true AND archived=false`。
5. 检索侧（`rag_skill/_build_scope_filter`）追加发布状态过滤；该过滤可配置关闭（用于未发布环境的联调）。

### 6.4 验收标准

- [ ] 同 doc_id 更新后，旧回答不再从缓存返回（缓存被主动失效）
- [ ] update/rollback 后检索结果切换（published 指针生效）
- [ ] ingest 单测覆盖：upsert 覆盖、失败回滚、缓存失效触发
- [ ] 不影响现有 `MILVUS_OUTPUT_FIELDS` 默认值（不写死新字段，仍可覆盖）

---

## 七、P1-2 安全合规接线

### 7.1 目标

把已实现但未使用的 `common_core/security.py` 原语接进查询管线，并补齐审计与 IAM 对接。

### 7.2 交付物与位置

| 改动 | 文件 | 说明 |
| --- | --- | --- |
| 查询清洗 | `rag_skill/mcp.py`（rag_answer/rag_retrieve 入口） | 调用 `check_safety(...)`：命中敏感词/注入模式 → 直接返回安全拒绝结果 |
| PII 全链路脱敏 | `rag_skill/pipeline.py` + `common_core/observability.py` | 日志与指标字段写前过 `mask_pii`；generation 输入上下文在写入前脱敏（可选开关） |
| 审计日志 | 新增 `common_core/audit.py` | 结构化事件：who(claims) / what(op) / which(kb/request) / when / result，写独立通道（文件或专用索引） |
| IAM/SSO 对接 | `common_core/auth.py` 扩展 | JWT 之外支持 OIDC/JWKS 自动发现；`issuer_url` 已有雏形（`AUTH_MCP_ISSUER_URL`） |

### 7.3 实施步骤

1. `security.py` 的 `check_safety` 目前未在任何查询入口被调用 → 在 `guard.resolve` 之后、`pipeline.retrieve/answer` 之前插入；拦截时返回与 `no_context` 一致的契约结构，不影响调用方。
2. `audit.py`：每条工具调用写一条审计事件（含评分/拦截原因），异步批量刷新，失败只记日志不阻断（与 observability 一致的安全默认值）。
3. 日志脱敏：`logger` 输出统一走一个 `masked()` 包装；注意不要掩掉 tenant_id/kb_id（非 PII）。
4. 评审 `mask_pii` 正则在中文语境的漏网（如中文身份证 18 位无空格）——补回归用例（`common_core/tests/test_security.py` 已有基础）。

### 7.4 验收标准

- [ ] 注入/敏感词查询被拦截并返回安全拒绝契约（有测试覆盖）
- [ ] 日志中不再出现手机号/身份证原文（可用测试构造含 PII 的查询验证）
- [ ] 每条工具调用产生审计事件，含用户身份与结果
- [ ] JWT 之外可对接外部 OIDC issuer（配置项 + 文档）

---

## 八、P1-3 可靠性

### 8.1 目标

把"单机本地模型 + 无防护"升级为"可独立部署、有防护、可诊断"。

### 8.2 交付物

| 项 | 方案 | 位置 |
| --- | --- | --- |
| 推理服务化 | Embedding 与 reranker 独立推理服务（Triton/Ray Serve/vLLM 或封装 HTTP），rag_skill 通过 `score_fn`/provider 注入远程调用；本地实现保留为 dev 模式 | `common_core/providers/` 新增 `embedding_service.py` / `rerank_service.py` |
| 限流 | 按 tenant/kb 的令牌桶，MCP 传输层 + pipeline 入口双层 | `common_core/ratelimit.py` + `rag_skill/pipeline.py` |
| 熔断 | provider 层对 LLM/向量库/Redis 做熔断（失败率阈值→打开→半开），复用现有降级点 | `common_core/providers/_circuit.py` |
| 健康检查 | 暴露 `/healthz`（Hashicorp 风格）：组件探活 + 只读探针 + 就绪探针 | 各 skill MCP HTTP 出口 |
| 向量库 HA | Milvus 连接多副本/故障切换配置文档；确保连接对象可重建 | `common_core/providers/vector.py`（连接重构） |

### 8.3 实施步骤

1. 先做限流（收益最快）：纯内存令牌桶 + tenant/kb key，超出直接返回 `429` 语义结果；补单测。
2. provider 熔断：包住 `llm.chat / a_search_hybrid / cache.get/set`，失败率超阈值后快速失败并记录指标（新增 `circuit_open_total` counter）。
3. 推理服务化：把 `Reranker.score_fn` 作为生产入口（架构上已预留注入点，见 `rerank.py`），新增远程实现。
4. 健康检查：MCP HTTP 传输开启时挂 `/healthz`；部署文档说明 liveness/readiness 语义。
5. Milvus HA：文档化 `uri/host` 多地址配置与重连策略；对现有 `_executor` 连接对象做重建测试。

### 8.4 验收标准

- [ ] 单租户突发流量被限流（有单测验证超限被拒）
- [ ] 模拟 LLM 持续失败 → 熔断快速打开，指标可见（`circuit_open_total`）
- [ ] 推理服务不可用时 rag_skill 降级到本地 or 透传（不同配置）
- [ ] `/healthz` 在组件异常时返回非 200

---

## 九、P2 权限粒度（chunk 级 ACL）

### 9.1 目标

在保留 tenant/kb 隔离基础上，支持文档/分块级可见性控制（如"某类文档仅结算组可见"）。

### 9.2 方案

- collection schema 增加 `acl_roles: list[str]` 字段（或独立 ACL 表）；
- `AgentContext` 扩展 `roles: list[str]`（从 JWT claims 解析）；
- `rag_skill/_build_scope_filter` 追加 `acl_roles` 命中过滤；空 roles 时仅可访问无 ACL 限制的文档（fail-closed，与现有风格一致）；
- `MILVUS_OUTPUT_FIELDS` 与检索过滤表达式构建需同步扩展（`common_core/providers/vector.py` 的 `build_filter_expr`）。

### 9.3 验收标准

- [ ] 无角色用户检索不到带 ACL 文档；带角色用户可命中
- [ ] ACL 过滤可配置关闭（不影响现有租户）
- [ ] 租户隔离与 ACL 叠加时仍保证交集语义

---

## 十、P3 交付工程

### 10.1 目标

让仓库"可独立交付、可安装、可部署"。

### 10.2 交付物

1. **依赖矩阵**：`docs/dependency-matrix.md` 列出 common_core / rag_skill / structured_query_skill / instances 的依赖关系与安装方式（editable install 命令）。
2. **common_core 发布**：补齐 `pyproject.toml` 版本与 classifiers，发布到私有 PyPI（或内部 index）；`rag_skill` 已声明 `common-core>=0.1.0`。
3. **仓库自包含决策**（三个选项，需产品决策）：
   - A. 把 `common_core` 也建独立仓库并发布（推荐，长期清晰）；
   - B. 在 `rag_skill` 仓库用 git submodule 引入 common_core；
   - C. 保持现状单包仓库，靠 `pip install -e ../common_core` 本地开发。
4. **部署运行手册**：`docs/deployment.md`——环境变量清单（对照 `instances/merchant/.env.example`）、Milvus/Redis/LLM 服务要求、并发与内存建议、压测结果记录表。

### 10.3 验收标准

- [ ] 新环境按文档可 30 分钟内跑通 `rag_answer`
- [ ] 依赖矩阵与 `pyproject.toml` 声明一致
- [ ] 完成 common_core 私有 index 发布，`pip install` 可直接命中

---

## 十一、成功度量（如何判断"到企业级了"）

| 维度 | 指标 | 目标值（示例起点） |
| --- | --- | --- |
| 回答质量 | faithfulness / answer relevance（RAGAS） | ≥ 0.85 / ≥ 0.8 |
| 检索质量 | Recall@5 / MRR | ≥ 0.9 / ≥ 0.8 |
| 性能 | p95 回答延迟 | ≤ 5s（含推理服务） |
| 缓存 | 缓存命中率（整体流量） | ≥ 20%（随重复问题增长） |
| 护栏 | 拦截率 / 误伤率 | 拦截率有报告，误伤率 < 1% |
| 可靠性 | 可用性 / 错误率 | ≥ 99.5% / ≤ 1% |
| 安全 | 含 PII 日志数 / 审计覆盖率 | 0 / 100% |

---

## 十二、风险与回滚

| 风险 | 影响 | 缓解 |
| --- | --- | --- |
| 评测 judge 主观 | 指标失真 | 多 judge + 人工复核 golden set |
| OTel 采样/导出性能开销 | 延迟上升 | 快路径不采样、异步导出、失败静默 |
| 缓存失效误伤（发布即清库） | 缓存命中率短暂下降 | 按 kb 分批失效 + 发布窗口错峰 |
| 审计写入成为瓶颈 | 请求阻塞 | 异步批量 + 失败降级 |
| 熔断误开 | 可用性下降 | 阈值保守起步、半开探测、人工阈值开关 |

回滚原则：所有 P0/P1 改动均为**可独立开关的配置项**（env 开关）+ 单测覆盖，不回滚基线功能。新增能力按 `migration-plan.md` 的"业务词隔离"约定落地，不污染 `rag_skill` 的通用语义。

---

## 附：建议执行顺序

1. **本周**：P0-1 第 1–3 步（golden set 首版 30 条 + 检索指标脚本 + 分数组分布统计）
2. **两周内**：P0-2 OTel 基础件打通本进程 + 示意图；P0-1 完成门禁
3. **一月内**：P1-1 invalidation hook（改动小先上）+ P1-2 安全接线
4. **一季内**：P1-3 可靠性、P2 权限、P3 交付工程

> 每一步完成后更新本文件状态；计划与 `migration-plan.md` 第 9 项（eval/ingest skill）对齐。
