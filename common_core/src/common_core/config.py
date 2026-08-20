""" 供所有业务能力共用、基于环境变量实现的基础配置组件。
本模块刻意做到与具体业务解耦：仅定义通用配置板块
（大模型、向量库、缓存、鉴权、监控指标、检索）。
各个业务能力基于这些基础组件组装自身领域配置，并传入各自专属的环境变量前缀。
"""

from __future__ import annotations

import os
import hashlib
import hmac
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

_TRUE_VALUES = {"1", "true", "yes", "y", "on"}
_FALSE_VALUES = {"0", "false", "no", "n", "off", ""}


class ConfigError(ValueError):
    """当运行时配置缺失或无效时抛出的异常。"""


_SENSITIVE_FRAGMENTS = (
    "API_KEY",
    "PASSWORD",
    "JWT_SECRET",
    "PUBLIC_KEY",
    "TOKEN",
    "SECRET",
)


def env_str(key: str, default: str = "", env: Mapping[str, str] | None = None) -> str:
    """从环境变量（或传入的字典）读取字符串配置。

    参数:
        key: 环境变量名。
        default: 读取失败或值为 None 时返回的默认值，默认空字符串。
        env: 可选的配置字典；为 None 时读取当前进程的 os.environ。

    返回:
        取到的字符串值；值为 None 或读取异常时返回 default。
    """
    source = os.environ if env is None else env
    try:
        value = source.get(key)
    except Exception:
        return default
    return default if value is None else str(value)


def env_int(key: str, default: int = 0, env: Mapping[str, str] | None = None) -> int:
    """从环境变量读取整数配置。

    参数:
        key: 环境变量名。
        default: 无法转换为整数时返回的默认值，默认 0。
        env: 可选的配置字典；为 None 时读取当前进程的 os.environ。

    返回:
        解析出的整数；转换失败或值为空时返回 default。
    """
    try:
        return int(env_str(key, "", env=env))
    except (TypeError, ValueError):
        return default


def env_float(key: str, default: float = 0.0, env: Mapping[str, str] | None = None) -> float:
    """从环境变量读取浮点数配置。

    参数:
        key: 环境变量名。
        default: 无法转换为浮点数时返回的默认值，默认 0.0。
        env: 可选的配置字典；为 None 时读取当前进程的 os.environ。

    返回:
        解析出的浮点数；转换失败或值为空时返回 default。
    """
    try:
        return float(env_str(key, "", env=env))
    except (TypeError, ValueError):
        return default


def env_bool(key: str, default: bool = False, env: Mapping[str, str] | None = None) -> bool:
    """从环境变量读取布尔配置。

    支持的真值：1/true/yes/y/on；支持的假值：0/false/no/n/off/空。

    参数:
        key: 环境变量名。
        default: 值无法识别为真或假时返回的默认值，默认 False。
        env: 可选的配置字典；为 None 时读取当前进程的 os.environ。

    返回:
        布尔值；值不在真/假集合内时返回 default。
    """
    raw = env_str(key, "", env=env).strip().lower()
    if raw in _TRUE_VALUES:
        return True
    if raw in _FALSE_VALUES:
        return False
    return default


def env_list(
    key: str,
    default: Sequence[str] = (),
    separator: str = ",",
    env: Mapping[str, str] | None = None,
) -> list[str]:
    """从环境变量读取按分隔符切分的列表配置。

    参数:
        key: 环境变量名。
        default: 值为空时返回的默认列表，默认空元组。
        separator: 列表元素间的分隔符，默认逗号。
        env: 可选的配置字典；为 None 时读取当前进程的 os.environ。

    返回:
        切分并去除首尾空白后的字符串列表；值为空时返回 default 的拷贝。
    """
    raw = env_str(key, "", env=env).strip()
    if not raw:
        return list(default)
    return [item.strip() for item in raw.split(separator) if item.strip()]


def env_scoped_modes(
    key: str,
    env: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """解析形如 `t1/kb1=llm_rewrite,t2/*=query_expansion` 的作用域覆盖配置。"""
    raw = env_str(key, "", env=env).strip()
    result: dict[str, str] = {}
    if not raw:
        return result
    for item in raw.split(","):
        item = item.strip()
        if not item or "=" not in item:
            continue
        scope, mode = item.split("=", 1)
        scope = scope.strip()
        mode = mode.strip().lower()
        if scope and mode:
            result[scope] = mode
    return result


def load_env_files(
    *paths: str,
    override: bool = False,
    env: Mapping[str, str] | None = None,
) -> list[str]:

    """ 加载 dotenv 环境变量文件，无需硬编码项目根目录。
    返回实际被加载的文件路径列表。
    如果未安装 python‑dotenv 库，该函数不执行任何操作；以此保证核心模块导入时保持轻量。
    """

    try:
        from dotenv import load_dotenv
    except ImportError:
        return []

    loaded: list[str] = []
    for path in paths:
        if path:
            load_dotenv(path, override=override)
            loaded.append(path)
    return loaded


def resolve_env_file(
    env_file: str | None = None,
    *,
    env_key: str = "",
    default: str = ".env",
) -> str | None:
    """在不硬编码项目根目录的前提下，确定要加载的 env 文件路径。

    优先级：显式传入的 env_file 参数 > 环境变量 env_key 指定的值 > 当前工作目录下的默认路径。
    返回 None 表示无需加载 env 文件，直接以进程环境变量为准。

    参数:
        env_file: 显式指定的 env 文件路径。
        env_key: 用于查找 env 文件路径的环境变量名。
        default: 默认环境文件名，默认 ".env"。

    返回:
        解析出的 env 文件路径；都没有匹配时返回 None。
    """
    if env_file:
        return str(env_file)
    if env_key:
        candidate = os.environ.get(env_key)
        if candidate:
            return candidate
    path = Path(default)
    if path.is_file():
        return str(path)
    return None


@dataclass(slots=True)
class LLMConfig:
    """兼容 OpenAI 接口的对话模型，以及本地 Embedding 嵌入模型相关配置。"""

    base_url: str = ""
    api_key: str = ""
    model: str = ""
    embedding_model: str = ""
    temperature: float = 0.0
    max_tokens: int = 2048
    timeout_seconds: float = 30.0

    @classmethod
    def from_env(
        cls,
        prefix: str = "LLM_",
        embedding_key: str = "EMBEDDING_MODEL",
        env: Mapping[str, str] | None = None,
    ) -> "LLMConfig":
        """从环境变量构造一个对话模型配置。

        参数:
            prefix: 环境变量前缀，默认 "LLM_"。
            embedding_key: 嵌入模型所在的环境变量名，默认 "EMBEDDING_MODEL"。
            env: 可选的配置字典；为 None 时读取当前进程的 os.environ。

        返回:
            填充好各项参数的 LLMConfig 实例。
        """
        return cls(
            base_url=env_str(f"{prefix}BASE_URL", env=env),
            api_key=env_str(f"{prefix}API_KEY", env=env),
            model=env_str(f"{prefix}MODEL", env=env),
            embedding_model=env_str(embedding_key, env=env),
            temperature=env_float(f"{prefix}TEMPERATURE", env=env),
            max_tokens=env_int(f"{prefix}MAX_TOKENS", default=2048, env=env),
            timeout_seconds=env_float(f"{prefix}TIMEOUT_SECONDS", default=30.0, env=env),
        )

    def validate(
        self,
        prefix: str = "LLM_",
        embedding_key: str = "EMBEDDING_MODEL",
        require_embedding: bool = False,
    ) -> list[str]:
        """检查对话模型配置是否完整，返回缺失项的清单。

        参数:
            prefix: 环境变量前缀，用于拼出缺失项的键名。
            embedding_key: 嵌入模型环境变量名，用于拼出缺失项的键名。
            require_embedding: 为 True 时要求必须配置 embedding 模型。

        返回:
            缺失配置键的名称列表；全部完整时为空列表。
        """
        missing: list[str] = []
        if not self.base_url:
            missing.append(f"{prefix}BASE_URL")
        if not self.api_key:
            missing.append(f"{prefix}API_KEY")
        if not self.model:
            missing.append(f"{prefix}MODEL")
        if require_embedding and not self.embedding_model:
            missing.append(embedding_key)
        return missing


@dataclass(slots=True)
class VectorStoreConfig:
    """兼容Milvus的向量数据库配置。集合(Collection)名称默认不设置；业务模块会传入自身对应的部署名称。"""

    host: str = "localhost"
    port: int = 19530
    user: str = ""
    password: str = ""
    secure: bool = False
    db_name: str = ""
    text_collection: str = ""
    image_collection: str = ""
    dim: int = 0
    text_output_fields: tuple[str, ...] = ()

    @classmethod
    def from_env(
        cls,
        prefix: str = "MILVUS_",
        env: Mapping[str, str] | None = None,
    ) -> "VectorStoreConfig":
        """从环境变量构造一个向量库配置。

        参数:
            prefix: 环境变量前缀，默认 "MILVUS_"。
            env: 可选的配置字典；为 None 时读取当前进程的 os.environ。

        返回:
            填充好各项参数的 VectorStoreConfig 实例。
        """
        return cls(
            host=env_str(f"{prefix}HOST", default="localhost", env=env),
            port=env_int(f"{prefix}PORT", default=19530, env=env),
            user=env_str(f"{prefix}USER", env=env),
            password=env_str(f"{prefix}PASSWORD", env=env),
            secure=env_bool(f"{prefix}SECURE", env=env),
            db_name=env_str(f"{prefix}DB", env=env),
            text_collection=env_str(f"{prefix}TEXT_COLLECTION", env=env),
            image_collection=env_str(f"{prefix}IMAGE_COLLECTION", env=env),
            dim=env_int(f"{prefix}DIM", env=env),
            text_output_fields=tuple(
                env_list(f"{prefix}OUTPUT_FIELDS", env=env)
            ),
        )

    def validate(self, prefix: str = "MILVUS_") -> list[str]:
        """检查向量库配置是否完整，返回缺失项的清单。

        参数:
            prefix: 环境变量前缀，用于拼出缺失项的键名。

        返回:
            缺失配置键的名称列表；全部完整时为空列表。
        """
        missing: list[str] = []
        if not self.host:
            missing.append(f"{prefix}HOST")
        if not self.text_collection:
            missing.append(f"{prefix}TEXT_COLLECTION")
        return missing


@dataclass(slots=True)
class CacheConfig:
    """兼容 Redis 的缓存配置。"""

    host: str = "localhost"
    port: int = 6379
    password: str = ""
    db: int = 0
    default_ttl: int = 1800
    key_prefix: str = "rag"
    socket_timeout: float = 3.0

    @classmethod
    def from_env(
        cls,
        prefix: str = "REDIS_",
        env: Mapping[str, str] | None = None,
    ) -> "CacheConfig":
        """从环境变量构造一个缓存配置。

        参数:
            prefix: 环境变量前缀，默认 "REDIS_"。
            env: 可选的配置字典；为 None 时读取当前进程的 os.environ。

        返回:
            填充好各项参数的 CacheConfig 实例。
        """
        return cls(
            host=env_str(f"{prefix}HOST", default="localhost", env=env),
            port=env_int(f"{prefix}PORT", default=6379, env=env),
            password=env_str(f"{prefix}PASSWORD", env=env),
            db=env_int(f"{prefix}DB", env=env),
            default_ttl=env_int(f"{prefix}DEFAULT_TTL", default=1800, env=env),
            key_prefix=env_str(f"{prefix}KEY_PREFIX", default="rag", env=env),
            socket_timeout=env_float(f"{prefix}SOCKET_TIMEOUT", default=3.0, env=env),
        )


@dataclass(slots=True)
class AuthConfig:
    """JWT 校验配置。密钥为空时，由调用方自行决定校验策略。"""

    mode: str = "jwt"
    jwt_secret: str = ""
    jwt_public_key: str = ""
    jwt_algorithms: tuple[str, ...] = ("HS256",)
    jwt_issuer: str = ""
    jwt_audience: str = ""
    mcp_issuer_url: str = ""
    mcp_resource_server_url: str = ""

    @classmethod
    def from_env(
        cls,
        prefix: str = "AUTH_",
        env: Mapping[str, str] | None = None,
    ) -> "AuthConfig":
        """从环境变量构造一个鉴权配置。

        参数:
            prefix: 环境变量前缀，默认 "AUTH_"。
            env: 可选的配置字典；为 None 时读取当前进程的 os.environ。

        返回:
            填充好各项参数的 AuthConfig 实例。
        """
        return cls(
            mode=env_str(f"{prefix}MODE", default="jwt", env=env).strip().lower(),
            jwt_secret=env_str(f"{prefix}JWT_SECRET", env=env),
            jwt_public_key=env_str(f"{prefix}JWT_PUBLIC_KEY", env=env),
            jwt_algorithms=tuple(
                env_list(f"{prefix}JWT_ALGORITHMS", default=("HS256",), env=env)
            ),
            jwt_issuer=env_str(f"{prefix}JWT_ISSUER", env=env),
            jwt_audience=env_str(f"{prefix}JWT_AUDIENCE", env=env),
            mcp_issuer_url=env_str(f"{prefix}MCP_ISSUER_URL", env=env),
            mcp_resource_server_url=env_str(f"{prefix}MCP_RESOURCE_SERVER_URL", env=env),
        )

    def validate(self, prefix: str = "AUTH_") -> list[str]:
        """检查鉴权配置是否合法完整，返回缺失项的清单。

        参数:
            prefix: 环境变量前缀，用于拼出缺失项的键名。

        返回:
            缺失配置键的名称列表；全部完整时为空列表。

        异常:
            ConfigError: mode 值非法（非 jwt/disabled）时抛出。
        """
        missing: list[str] = []
        if self.mode not in {"jwt", "disabled"}:
            raise ConfigError(
                f"Unsupported {prefix}MODE={self.mode!r}; expected jwt or disabled"
            )
        if self.mode == "jwt" and not self.jwt_secret and not self.jwt_public_key:
            missing.append(f"{prefix}JWT_SECRET or {prefix}JWT_PUBLIC_KEY")
        return missing


@dataclass(slots=True)
class MetricsConfig:
    """业务指标开关、HTTP 服务绑定配置、指标名称前缀。"""

    enabled: bool = False
    prefix: str = "app"
    port: int = 9090
    bind: str = "127.0.0.1"

    @classmethod
    def from_env(
        cls,
        prefix: str = "METRICS_",
        env: Mapping[str, str] | None = None,
    ) -> "MetricsConfig":
        """从环境变量构造一个指标配置。

        参数:
            prefix: 环境变量前缀，默认 "METRICS_"。
            env: 可选的配置字典；为 None 时读取当前进程的 os.environ。

        返回:
            填充好各项参数的 MetricsConfig 实例。
        """
        return cls(
            enabled=env_bool(f"{prefix}ENABLED", env=env),
            prefix=env_str(f"{prefix}PREFIX", default="app", env=env),
            port=env_int(f"{prefix}PORT", default=9090, env=env),
            bind=env_str(f"{prefix}BIND", default="127.0.0.1", env=env),
        )


@dataclass(slots=True)
class RetrievalConfig:
    """RAG 类业务模块共用的检索流水线调优配置。

    本类是所有检索与精排调优参数的**唯一契约默认值来源**：代码里出现
    的交叉编码器模型名、融合权重等默认值，都应引用这里的字段，而不是在
    业务逻辑中散落魔法数字。缺省配置时（未注入环境变量）即使用这些默认值，
    从而保证“开箱即用”的行为与历史上硬编码的字面量完全一致。
    """

    top_k: int = 20
    rrf_top_k: int = 20
    rrf_k: int = 60
    rerank_top_k: int = 3
    min_relevance: float = 0.70
    hybrid_max_workers: int = 16
    # 交叉编码器模型名：懒加载重排器时用于创建 CrossEncoder。
    # 默认与历史一致（bge-reranker-base），可由 RERANKER_MODEL 覆盖。
    rerank_model: str = "BAAI/bge-reranker-base"
    # 交叉编码器分数在融合分中的权重，默认 0.6。
    rerank_ce_weight: float = 0.6
    # 检索原始分数在融合分中的权重，默认 0.4。
    rerank_retrieval_weight: float = 0.4
    # 拼入 LLM 提示词的上下文总字符预算，防止超出模型上下文窗口。
    assembly_max_context_chars: int = 8000

    @classmethod
    def from_env(
        cls,
        prefix: str = "RETRIEVAL_",
        env: Mapping[str, str] | None = None,
    ) -> "RetrievalConfig":
        """从环境变量构造一份检索配置。

        参数:
            prefix: 主键前缀；默认 "RETRIEVAL_"，用于 TOP_K 等以该前缀开头的键。
            env: 可选的配置字典；为 None 时读取当前进程的 os.environ。

        返回:
            填充好各项参数的 RetrievalConfig 实例。
        """
        return cls(
            top_k=env_int(f"{prefix}TOP_K", default=20, env=env),
            rrf_top_k=env_int("RRF_TOP_K", default=20, env=env),
            rrf_k=env_int("RRF_K", default=60, env=env),
            rerank_top_k=env_int("RERANK_TOP_K", default=3, env=env),
            min_relevance=env_float(
                f"{prefix}MIN_RELEVANCE",
                default=0.70,
                env=env,
            ),
            hybrid_max_workers=env_int(
                f"{prefix}HYBRID_WORKERS", default=16, env=env
            ),
            # 重排相关键采用独立前缀 RERANKER_（与 rag_skill 的 .env 约定一致）。
            rerank_model=env_str(
                "RERANKER_MODEL",
                default="BAAI/bge-reranker-base",
                env=env,
            ),
            rerank_ce_weight=env_float(
                "RERANKER_CE_WEIGHT",
                default=0.6,
                env=env,
            ),
            rerank_retrieval_weight=env_float(
                "RERANKER_RETRIEVAL_WEIGHT",
                default=0.4,
                env=env,
            ),
            assembly_max_context_chars=env_int(
                f"{prefix}ASSEMBLY_MAX_CHARS",
                default=8000,
                env=env,
            ),
        )


@dataclass(slots=True)
class QueryRewriteConfig:
    """查询改写（Query Rewriting）配置。

    mode 支持 off / identity / llm_rewrite / query_expansion，默认 off；
    scoped_modes 允许按 ``tenant/kb``、``tenant/*`` 或 ``*/kb`` 粒度覆盖默认模式。
    """

    mode: str = "off"
    llm_model: str = ""
    temperature: float = 0.0
    max_tokens: int = 256
    expand_count: int = 2
    rewrite_prompt: str = ""
    expansion_prompt: str = ""
    scoped_modes: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_env(
        cls,
        prefix: str = "RETRIEVAL_QUERY_REWRITE_",
        env: Mapping[str, str] | None = None,
    ) -> "QueryRewriteConfig":
        """从环境变量构造一个查询改写配置。

        参数:
            prefix: 环境变量前缀，默认 "RETRIEVAL_QUERY_REWRITE_"。
            env: 可选的配置字典；为 None 时读取当前进程的 os.environ。

        返回:
            填充好各项参数的 QueryRewriteConfig 实例。
        """
        return cls(
            mode=env_str(f"{prefix}MODE", default="off", env=env).strip().lower(),
            llm_model=env_str(f"{prefix}LLM_MODEL", env=env),
            temperature=env_float(f"{prefix}TEMPERATURE", env=env),
            max_tokens=env_int(f"{prefix}MAX_TOKENS", default=256, env=env),
            expand_count=env_int(f"{prefix}EXPAND_COUNT", default=2, env=env),
            rewrite_prompt=env_str(f"{prefix}PROMPT", env=env),
            expansion_prompt=env_str(f"{prefix}EXPANSION_PROMPT", env=env),
            scoped_modes=env_scoped_modes(f"{prefix}SCOPES", env=env),
        )


@dataclass(slots=True)
class RuntimeConfig:
    """通用运行时各配置块的聚合总配置。"""

    llm: LLMConfig = field(default_factory=LLMConfig)
    vector: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    auth: AuthConfig = field(default_factory=AuthConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    query_rewrite: QueryRewriteConfig = field(default_factory=QueryRewriteConfig)

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "RuntimeConfig":
        """从环境变量构造一个聚合的运行时配置。

        参数:
            env: 可选的配置字典；为 None 时读取当前进程的 os.environ。

        返回:
            聚合了所有配置板块的 RuntimeConfig 实例。
        """
        return cls(
            llm=LLMConfig.from_env(env=env),
            vector=VectorStoreConfig.from_env(env=env),
            cache=CacheConfig.from_env(env=env),
            auth=AuthConfig.from_env(env=env),
            metrics=MetricsConfig.from_env(env=env),
            retrieval=RetrievalConfig.from_env(env=env),
            query_rewrite=QueryRewriteConfig.from_env(env=env),
        )

    def validate(self, *, require_embedding: bool = False) -> None:
        """当必需的运行时配置缺失时快速失败。

        进程环境变量是最终数据源；本方法只报告缺失的键，不会尝试加载文件。

        参数:
            require_embedding: 为 True 时要求必须配置 embedding 模型。

        异常:
            ConfigError: 任一必需配置键缺失时抛出。
        """
        missing: list[str] = []
        missing.extend(
            self.llm.validate(require_embedding=require_embedding)
        )
        missing.extend(self.vector.validate())
        missing.extend(self.auth.validate())
        if missing:
            raise ConfigError(
                "Missing required configuration keys: " + ", ".join(missing)
            )


def is_sensitive_key(key: str) -> bool:
    """判断一个配置键是否携带必须脱敏的敏感值。

    参数:
        key: 配置键名。

    返回:
        键名包含 API_KEY/PASSWORD/JWT_SECRET/PUBLIC_KEY/TOKEN/SECRET 等敏感片段时返回 True。
    """
    upper = key.upper()
    return any(fragment in upper for fragment in _SENSITIVE_FRAGMENTS)


def mask_key_value(key: str, value: str) -> str:
    """对单个配置值做日志与健康检查输出的脱敏处理。

    敏感键值显示为 ``<redacted>``；短的普通值全部打星；
    较长的普通值仅保留前两位与后两位字符。

    参数:
        key: 配置键名。
        value: 要脱敏的原始值。

    返回:
        脱敏后的字符串；值为空时返回空字符串。
    """
    if not value:
        return ""
    if is_sensitive_key(key):
        return "<redacted>"
    if len(value) <= 4:
        return "*" * len(value)
    return f"{value[:2]}***{value[-2:]}"


def config_snapshot(runtime: RuntimeConfig) -> list[tuple[str, str]]:
    """从运行时配置生成一份有序、有效的键值快照。

    参数:
        runtime: 聚合了各配置板块的运行时配置对象。

    返回:
        按板块顺序排列的 (键, 值) 字符串元组列表。
    """
    rows: list[tuple[str, str]] = []
    for key in ("base_url", "api_key", "model", "embedding_model"):
        rows.append(("LLM_" + key.upper(), getattr(runtime.llm, key)))
    for key in ("host", "port", "text_collection", "image_collection"):
        rows.append(("MILVUS_" + key.upper(), getattr(runtime.vector, key)))
    for key in ("host", "port", "db"):
        rows.append(("REDIS_" + key.upper(), getattr(runtime.cache, key)))
    for key in ("mode", "jwt_secret", "jwt_public_key"):
        rows.append(("AUTH_" + key.upper(), getattr(runtime.auth, key)))
    rows.append(("RETRIEVAL_TOP_K", runtime.retrieval.top_k))
    rows.append(("RETRIEVAL_MIN_RELEVANCE", runtime.retrieval.min_relevance))
    return [(k, str(v)) for k, v in rows]


def config_fingerprint(
    runtime: RuntimeConfig,
    *,
    secret: str | None = None,
) -> str:
    """基于生效（非敏感）配置值生成稳定的指纹。

    敏感键被排除，确保指纹绝不内嵌凭据。
    当传入 ``secret`` 时，摘要会使用 HMAC 签名，可独立于配置值进行轮换。

    参数:
        runtime: 聚合了各配置板块的运行时配置对象。
        secret: 可选的 HMAC 签名字段；为 None 时使用普通 SHA-256。

    返回:
        十六进制指纹字符串。
    """
    parts: list[str] = []
    for key, value in config_snapshot(runtime):
        if is_sensitive_key(key):
            continue
        parts.append(f"{key}={value}")
    payload = "&".join(parts).encode("utf-8")
    if secret:
        return hmac.new(secret.encode("utf-8"), payload, hashlib.sha256).hexdigest()
    return hashlib.sha256(payload).hexdigest()


def log_config_audit(
    runtime: RuntimeConfig,
    *,
    source: str = "process-env",
    logger: Any = None,
    require_embedding: bool = False,
) -> str:
    """打印一条经过脱敏的启动日志，并返回配置指纹。

    日志会记录配置来源、指纹以及有效（非敏感）键的数量，
    方便运维确认实际生效的配置来源，同时不暴露任何敏感值。

    参数:
        runtime: 聚合了各配置板块的运行时配置对象。
        source: 配置来源标识，默认 "process-env"。
        logger: 待使用的日志器；为 None 时使用模块默认日志器。
        require_embedding: 校验时是否要求配置 embedding 模型。

    返回:
        配置指纹字符串。
    """
    import logging

    log = logger or logging.getLogger("common_core.config")
    runtime.validate(require_embedding=require_embedding)
    fingerprint = config_fingerprint(runtime)
    snapshot = config_snapshot(runtime)
    log.info(
        "config source=%s fingerprint=%s effective_keys=%d required=complete",
        source,
        fingerprint,
        len(snapshot),
    )
    return fingerprint


def redacted_snapshot(runtime: RuntimeConfig) -> dict[str, str]:
    """返回一份完全脱敏的配置快照，可安全地暴露给健康检查接口。

    参数:
        runtime: 聚合了各配置板块的运行时配置对象。

    返回:
        键到脱敏值的字典，所有敏感值均已被掩码处理。
    """
    return {key: mask_key_value(key, value) for key, value in config_snapshot(runtime)}
