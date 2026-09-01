# retrieve_skill

可独立复用的 RAG 检索组件。它只做“检索并返回权威父块上下文”，不包含回答生成、LangGraph 编排或业务话术；既可作为本地 Python 组件调用，也可以通过 MCP 工具 `rag_retrieve` 对外提供服务。

调用方契约见 [SKILL.md](SKILL.md)，完整配置见 [.env.example](.env.example) 与 [common_core 配置文档](../common_core/docs/config.md)。

## 能力

- Milvus Dense Embedding + BM25 混合召回，RRF 融合，Cross-Encoder 精排，相关性阈值门控。
- 多租户隔离：JWT 鉴权、Milvus 过滤、MySQL 回源三层校验。
- 子块召回 + 父块回源：Milvus 只存子块，MySQL 保存权威父块；回源时校验 tenant / kb / status / doc_version。
- 父块引用级 Redis 缓存：缓存签名覆盖查询、改写、过滤、模型与数据版本；命中后仍回源 MySQL 重建上下文。
- 查询改写 / 查询扩展、token 预算截断、Prometheus 指标、结构化错误契约与关键依赖故障 fail-closed。

## 处理流程

```mermaid
flowchart TD
    A["调用方 / Agent"] --> B["rag_retrieve / retrieve_context"]
    B --> C["JWT 鉴权与租户校验"]
    C --> D["检索缓存检查"]

    D -->|"缓存命中"| E["MySQL 父块回源"]
    D -->|"缓存未命中"| F["查询改写 / 查询扩展"]

    F --> G["Milvus BM25 + Dense 混合召回"]
    G --> H["RRF 融合"]
    H --> I["Cross-Encoder 精排"]
    I --> J{"是否达到相关性阈值?"}

    J -->|"否"| K["返回 no_context"]
    J -->|"是"| L["按 parent_id 聚合父块"]
    L --> M["MySQL 批量回源父块"]
    M --> N["校验 tenant / kb / status / doc_version"]
    N --> O["写入 Redis 父块引用缓存"]
    O --> P["Token 预算截断与字段投影"]
    P --> Q["返回父块 docs"]

    E --> P
```

## 快速上手

组件依赖 `common_core`，已发布到私有源可直接安装：

```powershell
pip install "retrieve-skill[mysql]"
```

monorepo 本地开发时可执行：

```powershell
pip install -e ../common_core --no-deps
pip install -e . --no-deps
pip install ".[mysql]"
```



```python
import asyncio

from common_core.context import AgentContext
from retrieve_skill.builder import build_pipeline

pipeline = build_pipeline()
ctx = AgentContext(tenant_id="", kb_id="", request_id="")

async def main() -> None:
    result = await pipeline.retrieve_context("当前支持的售后流程是什么？", context=ctx)
    print(result.status, result.docs)

asyncio.run(main())
```

## Docker 运行

镜像默认以 `streamable-http` 方式监听 `0.0.0.0:8000`，入口命令是 `retrieve-skill-mcp`。

```bash
# 本地构建
docker build -t retrieve-skill:0.1.1 .

docker run --rm -p 8000:8000 --env-file .env retrieve-skill:0.1.1
```

推送 tag `v0.1.1` 后可从 GHCR 直接拉取：
```bash
docker pull ghcr.io/<owner>/retrieve_pipeline:v0.1.1
docker run --rm -p 8000:8000 --env-file .env ghcr.io/<owner>/retrieve_pipeline:v0.1.1
```



## 边界
- 不包含回答生成、LangGraph 编排、业务词表与知识摄取。
- 不提供 Milvus / MySQL 写入能力，知识写入归属入库链路。
- 回答侧公共机制由外层 `common_core.rag` 按需复用。
