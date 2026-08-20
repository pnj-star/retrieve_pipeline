"""用于 agent / MCP 工具调用的、框架无关的鉴权与作用域防护模块。
各个 Skill 会使用本模块，把每次调用携带的鉴权 token、租户参数，转换成经过校验的 AgentContext 对象。
本模块刻意不引入任何 MCP SDK 依赖，因此普通 Python 工作进程也可以复用同一套鉴权规则、做单元测试
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .auth import AuthConfig, AuthError, IdentityClaims, TokenVerifier, parse_bearer_token
from .context import AgentContext


class ToolAuthError(AuthError):
    """当工具调用未通过身份认证，或者超出 token 允许的权限作用域时抛出该异常。"""


@dataclass(slots=True)
class ToolContextGuard:
    """校验调用方的 token，并将其绑定到本次请求对应的工具作用域。
    在环境配置 AUTH_MODE=disabled（鉴权关闭）模式下，该守卫仍然强制要求传入 tenant、kb、request id；
    以此保证下游的缓存、指标统计、审计链路始终拥有完整的作用域上下文。
    """

    config: AuthConfig = field(default_factory=AuthConfig)
    require_tenant_id: bool = True
    require_kb_id: bool = True
    require_request_id: bool = True
    verifier: TokenVerifier | None = None

    def __post_init__(self) -> None:
        if self.verifier is None:
            self.verifier = TokenVerifier(self.config)

    @classmethod
    def from_env(
        cls,
        prefix: str = "AUTH_",
        env: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> "ToolContextGuard":
        """从环境变量构造一个工具作用域守卫。

        参数:
            prefix: 鉴权配置的环境变量前缀，默认 "AUTH_"。
            env: 可选的配置字典；为 None 时读取当前进程的 os.environ。
            **kwargs: 透传给 ToolContextGuard 的其它参数（如 require_tenant_id）。

        返回:
            配置好的 ToolContextGuard 实例。
        """
        return cls(AuthConfig.from_env(prefix=prefix, env=env), **kwargs)

    def resolve(
        self,
        *,
        tenant_id: str = "",
        kb_id: str = "",
        request_id: str = "",
        session_id: str = "",
        user_id: str = "",
        auth_token: str | None = None,
        authorization: str | None = None,
    ) -> AgentContext:
        """校验调用方身份并返回经过确认的作用域上下文；失败时以明确的 AuthError 安全关闭。

        AUTH_MODE=disabled 时走开发模式，仅校验必填的作用域参数；
        否则校验 JWT，并核对调用方传人的 tenant/kb 是否与 token 声明一致。

        参数:
            tenant_id: 调用方声明的租户 ID。
            kb_id: 调用方声明的知识库 ID。
            request_id: 请求 ID。
            session_id: 会话 ID。
            user_id: 用户 ID（JWT 场景下以 token 声明为准）。
            auth_token: 直接传入的 JWT，优先于 authorization 头。
            authorization: HTTP Authorization 头的原始值。

        返回:
            经过校验的 AgentContext。

        异常:
            ToolAuthError: token 缺失/无效，或 tenant/kb 与 token 作用域不匹配时抛出。
        """
        if self.config.mode == "disabled":
            return self._dev_context(
                tenant_id=tenant_id,
                kb_id=kb_id,
                request_id=request_id,
                session_id=session_id,
                user_id=user_id,
            )
        raw = (auth_token or "").strip() or parse_bearer_token(authorization)
        if not raw:
            raise ToolAuthError(
                "Missing auth_token or Authorization bearer token; tool calls require JWT authentication.",
                401,
            )
        try:
            identity = self.verifier.identity(raw)
        except AuthError as exc:
            raise ToolAuthError(exc.message, exc.status_code) from None
        return self._claims_context(
            identity=identity,
            tenant_id=tenant_id,
            kb_id=kb_id,
            request_id=request_id,
            session_id=session_id,
            user_id=user_id,
        )

    def _dev_context(
        self,
        *,
        tenant_id: str,
        kb_id: str,
        request_id: str,
        session_id: str,
        user_id: str,
    ) -> AgentContext:
        """在鉴权关闭模式下构造开发用的上下文，仅校验必填作用域。

        参数:
            tenant_id/kb_id/request_id: 必填的作用域参数。
            session_id/user_id: 可选的会话与用户信息。

        返回:
            开发模式的 AgentContext；user_id 缺省为 "dev-user"。

        异常:
            ToolAuthError: 任一必填作用域缺失时抛出。
        """
        tenant_id = tenant_id.strip()
        kb_id = kb_id.strip()
        request_id = request_id.strip()
        if self.require_tenant_id and not tenant_id:
            raise ToolAuthError("tenant_id is required for tool calls.", 400)
        if self.require_kb_id and not kb_id:
            raise ToolAuthError("kb_id is required for tool calls.", 400)
        if self.require_request_id and not request_id:
            raise ToolAuthError("request_id is required for tool calls.", 400)
        return AgentContext(
            tenant_id=tenant_id,
            kb_id=kb_id,
            session_id=session_id.strip(),
            request_id=request_id,
            user_id=user_id.strip() or "dev-user",
        )

    def _claims_context(
        self,
        *,
        identity: IdentityClaims,
        tenant_id: str,
        kb_id: str,
        request_id: str,
        session_id: str,
        user_id: str,
    ) -> AgentContext:
        """把 JWT 身份声明与调用方传入的作用域合并成最终上下文。

        以调用方传入的 tenant/kb 优先，缺失时回退到 token 声明；
        两者都存在且不一致时判定越权。

        参数:
            identity: 已校验的身份声明。
            tenant_id/kb_id/request_id/session_id/user_id: 调用方传入的作用域参数。

        返回:
            合并并校验后的 AgentContext。

        异常:
            ToolAuthError: tenant/kb 越权或必填作用域缺失时抛出。
        """
        tenant_id = tenant_id.strip()
        kb_id = kb_id.strip()
        request_id = request_id.strip()

        resolved_tenant = tenant_id or identity.tenant_id
        if identity.tenant_id and resolved_tenant and identity.tenant_id != resolved_tenant:
            raise ToolAuthError(
                "tenant_id does not match the authenticated token scope.",
                403,
            )
        if self.require_tenant_id and not resolved_tenant:
            raise ToolAuthError("tenant_id is required for tool calls.", 400)

        resolved_kb = kb_id or identity.kb_id
        if identity.kb_id and resolved_kb and identity.kb_id != resolved_kb:
            raise ToolAuthError("kb_id does not match the authenticated token scope.", 403)
        if self.require_kb_id and not resolved_kb:
            raise ToolAuthError("kb_id is required for tool calls.", 400)

        resolved_request = request_id or identity.request_id
        if self.require_request_id and not resolved_request:
            raise ToolAuthError("request_id is required for tool calls.", 400)

        return AgentContext(
            tenant_id=resolved_tenant,
            kb_id=resolved_kb,
            session_id=session_id.strip() or identity.session_id,
            request_id=resolved_request,
            user_id=identity.sub or user_id.strip() or "jwt-user",
        )


def resolve_tool_context(
    config: AuthConfig | None = None,
    *,
    prefix: str = "AUTH_",
    env: dict[str, str] | None = None,
    **kwargs: Any,
) -> AgentContext:
    """针对 ToolContextGuard.resolve() 的便捷封装函数。"""
    if config is None:
        config = AuthConfig.from_env(prefix=prefix, env=env)
    return ToolContextGuard(config).resolve(**kwargs)


class MCPBearerTokenVerifier:
    """FastMCP 的 Bearer‑token 协议适配器，适配 HTTP、SSE 两种传输方式。
    JWT 合法时，verify_token 返回 SDK 的 AccessToken 对象；
    如果处于鉴权关闭模式，则返回 None。这样 HTTP 部署场景，即便 MCP 进程没有配置 JWKS 公钥集合 / 密钥，服务依然可以正常运行。
    """

    def __init__(
        self,
        config: AuthConfig | None = None,
        *,
        prefix: str = "AUTH_",
        env: dict[str, str] | None = None,
    ) -> None:
        """初始化 MCP Bearer-token 校验器。

        参数:
            config: 鉴权配置；为 None 时从环境变量构造。
            prefix: 鉴权配置的环境变量前缀，默认 "AUTH_"。
            env: 可选的配置字典；为 None 时读取当前进程的 os.environ。
        """
        if config is None:
            config = AuthConfig.from_env(prefix=prefix, env=env)
        self.config = config
        self.verifier = TokenVerifier(config)

    async def verify_token(self, token: str) -> Any | None:
        """校验传入的 Bearer token 并转换为 MCP 的 AccessToken 对象。

        参数:
            token: 待校验的 JWT 字符串。

        返回:
            鉴权关闭模式下返回 AccessToken 对象；JWT 合法时返回对应 AccessToken；
            校验失败或 MCP SDK 未安装时返回 None。
        """
        if self.config.mode == "disabled":
            return self._access_token(
                token=token,
                client_id="dev-user",
                scopes=["agent"],
            )
        try:
            identity = self.verifier.identity(token)
        except AuthError:
            return None
        return self._access_token(
            token=token,
            client_id=identity.sub or "jwt-user",
            scopes=list(identity.roles) or ["agent"],
        )

    @staticmethod
    def _access_token(*, token: str, client_id: str, scopes: list[str]) -> Any | None:
        """构造 MCP 的 AccessToken 对象；SDK 未安装时返回 None。

        参数:
            token: 原始 token 字符串。
            client_id: 客户端标识。
            scopes: 授权的作用域列表。

        返回:
            MCP AccessToken 对象；SDK 不可用时返回 None。
        """
        try:
            from mcp.server.auth.provider import AccessToken
        except ImportError:
            return None
        return AccessToken(token=token, client_id=client_id, scopes=scopes)


def build_mcp_auth(
    config: AuthConfig | None = None,
    *,
    prefix: str = "AUTH_",
    env: dict[str, str] | None = None,
) -> tuple[Any | None, Any | None]:
    """JWT 鉴权模式下，返回元组 (token_verifier, FastMCP AuthSettings)。
    当配置 AUTH_MODE=disabled（鉴权关闭）时，函数返回 (None, None)，让传输层中间件放行请求；
    但每个工具调用层面的 ToolContextGuard 守卫，仍然要求每次调用都携带 tenant、kb、request 作用域参数。
    """
    if config is None:
        config = AuthConfig.from_env(prefix=prefix, env=env)
    if config.mode != "jwt":
        return None, None
    verifier = MCPBearerTokenVerifier(config)
    from mcp.server.auth.settings import AuthSettings

    auth_settings = AuthSettings(
        issuer_url=config.mcp_issuer_url or "http://localhost",
        resource_server_url=config.mcp_resource_server_url or "http://localhost",
    )
    return verifier, auth_settings
