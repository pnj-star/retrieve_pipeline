"""共用运行时（common_core）：供多个 skill 共享的底层组件集合。
本包不依赖具体业务，提供配置、鉴权、上下文、可观测性、安全审查与数据提供者的基础组件，是各 skill 的地基。
导出名用于方便业务模块从包顶层直接引入常用类与函数。
"""

from .auth import AuthError, IdentityClaims, TokenVerifier
from .config import (
    AuthConfig,
    CacheConfig,
    LLMConfig,
    MetricsConfig,
    RetrievalConfig,
    RuntimeConfig,
    VectorStoreConfig,
)
from .context import AgentContext
from .mcp_auth import (
    MCPBearerTokenVerifier,
    ToolAuthError,
    ToolContextGuard,
    build_mcp_auth,
    resolve_tool_context,
)
from .observability import Observability
from .protocols import QueryRequest, QueryResult
from .security import INJECTION_PATTERNS, check_safety, mask_pii, normalize_query

__version__ = "0.1.0"

__all__ = [
    "AuthConfig",
    "AuthError",
    "AgentContext",
    "CacheConfig",
    "INJECTION_PATTERNS",
    "IdentityClaims",
    "LLMConfig",
    "MCPBearerTokenVerifier",
    "MetricsConfig",
    "Observability",
    "QueryRequest",
    "QueryResult",
    "RetrievalConfig",
    "RuntimeConfig",
    "ToolAuthError",
    "ToolContextGuard",
    "TokenVerifier",
    "VectorStoreConfig",
    "build_mcp_auth",
    "check_safety",
    "mask_pii",
    "normalize_query",
    "resolve_tool_context",
]
