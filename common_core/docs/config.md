# common_core 配置契约

这是所有 AI skill 配置的**唯一契约入口**。契约约束键名、类型、必填/可选与敏感标记；真实值只存在于各 skill 的 `.env` 或进程环境，绝不进入本库或仓库。

## 加载顺序

配置来源优先级从高到低固定为：

1. 系统环境变量（`os.environ`）
2. 当前 skill 部署目录的 `.env`
3. 代码中的默认值

只由可执行入口（各 skill 的 `main()`）显式加载 `.env`（`override=False`），`common_core` 不主动读取自身安装目录下的 `.env`。每个 MCP 进程只加载一份 env 源；同名键（如 `LLM_MODEL`）在不同 skill 进程里允许不同值，互不冲突。

## 完整键表

标记说明：`必填` 列里的 `*` 表示所有 skill 都要；`rag`/`sql` 表示仅该 skill 要求。`敏感` 列标记为 `是` 的键必须脱敏，且 `.env.example` 只允许出现占位符。

| 键 | 类型 | 默认值 | 必填 | 敏感 |
| --- | --- | --- | --- | --- |
| `LLM_BASE_URL` | str | `""` | `*` | 否 |
| `LLM_API_KEY` | str | `""` | `*` | 是 |
| `LLM_MODEL` | str | `""` | `*` | 否 |
| `LLM_TEMPERATURE` | float | `0.0` | 可选 | 否 |
| `LLM_MAX_TOKENS` | int | `2048` | 可选 | 否 |
| `LLM_TIMEOUT_SECONDS` | float | `30.0` | 可选 | 否 |
| `EMBEDDING_MODEL` | str | `""` | `rag` | 否 |
| `MILVUS_HOST` | str | `localhost` | `rag` | 否 |
| `MILVUS_PORT` | int | `19530` | 可选 | 否 |
| `MILVUS_USER` | str | `""` | 可选 | 否 |
| `MILVUS_PASSWORD` | str | `""` | 可选 | 是 |
| `MILVUS_SECURE` | bool | `false` | 可选 | 否 |
| `MILVUS_DB` | str | `""` | 可选 | 否 |
| `MILVUS_TEXT_COLLECTION` | str | `""` | `rag` | 否 |
| `MILVUS_IMAGE_COLLECTION` | str | `""` | 可选 | 否 |
| `MILVUS_DIM` | int | `0` | 可选 | 否 |
| `MILVUS_OUTPUT_FIELDS` | csv | `[]` | 可选 | 否 |
| `REDIS_HOST` | str | `localhost` | `rag` | 否 |
| `REDIS_PORT` | int | `6379` | 可选 | 否 |
| `REDIS_PASSWORD` | str | `""` | 可选 | 是 |
| `REDIS_DB` | int | `0` | 可选 | 否 |
| `REDIS_DEFAULT_TTL` | int | `1800` | 可选 | 否 |
| `REDIS_KEY_PREFIX` | str | `rag` | 可选 | 否 |
| `REDIS_SOCKET_TIMEOUT` | float | `3.0` | 可选 | 否 |
| `AUTH_MODE` | enum | `jwt` | `*` | 否 |
| `AUTH_JWT_SECRET` | str | `""` | `*`（jwt 需其一） | 是 |
| `AUTH_JWT_PUBLIC_KEY` | str | `""` | `*`（jwt 需其一） | 是 |
| `AUTH_JWT_ALGORITHMS` | csv | `HS256` | 可选 | 否 |
| `AUTH_JWT_ISSUER` | str | `""` | 可选 | 否 |
| `AUTH_JWT_AUDIENCE` | str | `""` | 可选 | 否 |
| `AUTH_MCP_ISSUER_URL` | str | `""` | 可选 | 否 |
| `AUTH_MCP_RESOURCE_SERVER_URL` | str | `""` | 可选 | 否 |
| `METRICS_ENABLED` | bool | `false` | 可选 | 否 |
| `METRICS_PREFIX` | str | `app` | 可选 | 否 |
| `METRICS_PORT` | int | `9090` | 可选 | 否 |
| `METRICS_BIND` | str | `127.0.0.1` | 可选 | 否 |
| `RETRIEVAL_TOP_K` | int | `20` | 可选 | 否 |
| `RETRIEVAL_MIN_RELEVANCE` | float | `0.70` | 可选 | 否 |
| `RETRIEVAL_HYBRID_WORKERS` | int | `16` | 可选 | 否 |
| `RETRIEVAL_ASSEMBLY_MAX_CHARS` | int | `8000` | 可选 | 否 |
| `RRF_TOP_K` | int | `20` | 可选 | 否 |
| `RRF_K` | int | `60` | 可选 | 否 |
| `RERANK_TOP_K` | int | `3` | 可选 | 否 |
| `RETRIEVAL_QUERY_REWRITE_MODE` | enum | `off` | 可选 | 否 |
| `RETRIEVAL_QUERY_REWRITE_LLM_MODEL` | str | `""` | 可选 | 否 |
| `RETRIEVAL_QUERY_REWRITE_TEMPERATURE` | float | `0.0` | 可选 | 否 |
| `RETRIEVAL_QUERY_REWRITE_MAX_TOKENS` | int | `256` | 可选 | 否 |
| `RETRIEVAL_QUERY_REWRITE_EXPAND_COUNT` | int | `2` | 可选 | 否 |
| `RETRIEVAL_QUERY_REWRITE_PROMPT` | str | `""` | 可选 | 否 |
| `RETRIEVAL_QUERY_REWRITE_EXPANSION_PROMPT` | str | `""` | 可选 | 否 |
| `RETRIEVAL_QUERY_REWRITE_SCOPES` | str | `""` | 可选 | 否 |
| `RERANKER_MODEL` | str | `BAAI/bge-reranker-base` | `rag` | 否 |
| `RERANKER_DEVICE` | str | `""` | 可选 | 否 |
| `RERANKER_CE_WEIGHT` | float | `0.6` | 可选 | 否 |
| `RERANKER_RETRIEVAL_WEIGHT` | float | `0.4` | 可选 | 否 |
| `SQL_HOST` | str | `localhost` | `sql` | 否 |
| `SQL_PORT` | int | `3306` | 可选 | 否 |
| `SQL_USER` | str | `""` | `sql` | 否 |
| `SQL_PASSWORD` | str | `""` | `sql` | 是 |
| `SQL_DATABASE` | str | `""` | `sql` | 否 |
| `SQL_CHARSET` | str | `utf8mb4` | 可选 | 否 |
| `SQL_CONNECT_TIMEOUT` | float | `3.0` | 可选 | 否 |
| `SQL_READ_TIMEOUT` | float | `5.0` | 可选 | 否 |
| `SQL_POOL_MAX_CONNECTIONS` | int | `10` | 可选 | 否 |
| `SQL_POOL_MIN_CACHED` | int | `2` | 可选 | 否 |
| `SQL_POOL_MAX_CACHED` | int | `5` | 可选 | 否 |
| `SQL_POOL_BLOCKING` | bool | `true` | 可选 | 否 |
| `SQL_DEFAULT_MAX_ROWS` | int | `20` | 可选 | 否 |

## 校验规则

- `RuntimeConfig.validate()` 启动时 fail-fast，缺任一必填键抛 `ConfigError`，附带完整缺失键清单。
- `AUTH_MODE` 只允许 `jwt` / `disabled`；`jwt` 模式必须提供 `AUTH_JWT_SECRET` 或 `AUTH_JWT_PUBLIC_KEY` 之一。
- 其余读取失败（整数、布尔解析失败）回退默认值，不中断启动。

## 一致性

CI 用 `common_core/scripts/check_env_examples.py` 对比每个 skill 的 `.env.example` 与上表：缺必填键、出现未声明键、或敏感键写成真实值即失败。修改键名/必填/敏感属性时，必须同步 `config.py`、`config.md` 与该脚本。

## 环境文件解析

`resolve_env_file()` 按“显式 `--env-file` 参数 > `*_ENV_FILE` 环境变量 > 当前目录 `.env`”解析，交给 `load_env_files()`（`override=False`）。启动日志会打印配置来源与脱敏配置指纹（见 `config_fingerprint()` / `log_config_audit()`）。
