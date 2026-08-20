"""与框架解耦的 JWT 身份校验模块。本模块刻意不引入 langgraph_sdk，也不依赖任何 Web 服务框架。
在 LangGraph 内部运行的业务模块，可以把这套基础能力接入自身的鉴权装饰器；普通独立服务可以直接使用 TokenVerifier 类。
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from typing import Any, Mapping

from .config import AuthConfig
from .context import AgentContext


class AuthError(Exception):
    """当 token 缺失、格式错误、已过期或无效时抛出的异常，带有 HTTP 状态码以便上层转换为相应响应。
    """

    def __init__(self, message: str, status_code: int = 401) -> None:
        """初始化鉴权异常。
参数:
    message: 人可读的错误信息。
    status_code: 应返回的 HTTP 状态码，默认 401。
        """
        super().__init__(message)
        self.message = message
        self.status_code = status_code


def parse_bearer_token(authorization: str | None) -> str | None:
    """从 Authorization 头中提取 Bearer token。
参数:
    authorization: HTTP Authorization 头的原始值（可为 None）。
返回:
    成功时返回 token 字符串；头缺失/格式不正（非 Bearer 或空）时返回 None。
    """
    if not authorization:
        return None
    parts = authorization.split(" ", 1)
    if (
        len(parts) == 2
        and parts[0].strip().lower() == "bearer"
        and parts[1].strip()
    ):
        return parts[1].strip()
    return None


def decode_public_key(key: str) -> str:
    """返回 PEM 文本，当输入是被 base64 包裹的 PEM 时自动解码。
参数:
    key: 原始密钥，可能是 PEM 文本或 base64 字符串。
返回:
    解码后的 PEM 文本；若不是可解码的 base64 PEM，原样返回输入。
    """
    if "-----BEGIN" in key:
        return key
    try:
        decoded = base64.b64decode(key).decode("utf-8")
    except Exception:
        return key
    return decoded if "-----BEGIN" in decoded else key


@dataclass(frozen=True)
class IdentityClaims:
    """鉴权后的身份声明集合，封装了用户与分层边界信息。
属性:
    sub: 主体，通常代表用户 ID。
    tenant_id: 租户 ID，数据隔离边界。
    kb_id: 知识库 ID。
    session_id: 会话 ID。
    request_id: 请求 ID。
    roles: 角色/范围集合，可来自 roles/scope/scp 声明。
    extra: 其他未知自定义声明的保留字典。
    """

    sub: str
    tenant_id: str = ""
    kb_id: str = ""
    session_id: str = ""
    request_id: str = ""
    roles: tuple[str, ...] = ()
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "IdentityClaims":
        """从解码后的 JWT payload 构造 IdentityClaims。
参数:
    payload: pyjwt 解码出的字典。
返回:
    填充好身份信息的 IdentityClaims 实例。
        """
        raw_roles = (
            payload.get("roles") or payload.get("scope") or payload.get("scp") or ()
        )
        if isinstance(raw_roles, str):
            roles = tuple(item.strip() for item in raw_roles.split() if item.strip())
        else:
            roles = tuple(str(item) for item in raw_roles)

        extra = {
            key: value
            for key, value in payload.items()
            if key
            not in {
                "sub",
                "tenant_id",
                "kb_id",
                "session_id",
                "request_id",
                "roles",
                "scope",
                "scp",
            }
        }
        return cls(
            sub=str(payload.get("sub") or ""),
            tenant_id=str(payload.get("tenant_id") or ""),
            kb_id=str(payload.get("kb_id") or ""),
            session_id=str(payload.get("session_id") or ""),
            request_id=str(payload.get("request_id") or ""),
            roles=roles,
            extra=extra,
        )

    def to_context(self) -> AgentContext:
        """转换为 AgentContext，用于在整个处理链路中传递身份与隔离边界。
返回:
    以 sub 作为 user_id、其余字段对应填充 的 AgentContext。
        """
        return AgentContext(
            tenant_id=self.tenant_id,
            kb_id=self.kb_id,
            session_id=self.session_id,
            request_id=self.request_id,
            user_id=self.sub,
        )


class TokenVerifier:
    """使用 HS256 或者非对称密钥校验 JWT，支持可选的 iss、aud 校验。    """

    def __init__(
        self,
        config: AuthConfig | None = None,
        *,
        key: str | None = None,
        algorithms: tuple[str, ...] | list[str] | None = None,
        issuer: str | None = None,
        audience: str | None = None,
    ) -> None:
        """初始化 token 校验器。
参数:
    config: JWT 相关配置，包含密钥、算法、issuer/audience 等。
    key: 显式提供的密钥；优先级高于 config 中的密钥。
    algorithms: 允许的签名算法集合；None 时用 config.jwt_algorithms。
    issuer: 必须匹配的发行方；None 时用 config。
    audience: 必须匹配的受众；None 时用 config。
        """
        cfg = config or AuthConfig()
        self.config = cfg
        if key is not None:
            self._key = key
        elif cfg.jwt_secret:
            self._key = cfg.jwt_secret
        elif cfg.jwt_public_key:
            self._key = decode_public_key(cfg.jwt_public_key)
        else:
            self._key = ""
        self._algorithms = tuple(algorithms or cfg.jwt_algorithms)
        self._issuer = cfg.jwt_issuer if issuer is None else issuer
        self._audience = cfg.jwt_audience if audience is None else audience

    def verify(self, token: str) -> dict[str, Any]:
        """校验并解码 JWT。
参数:
    token: 待校验的 JWT 字符串。
返回:
    解码后的 payload 字典。
异常:
    AuthError: 密钥未配置、token 过期或无效时抛出。
        """
        if not self._key:
            raise AuthError("Auth key is not configured; refusing to verify tokens.")

        import jwt as pyjwt

        kwargs: dict[str, Any] = {"algorithms": self._algorithms, "options": {"require": ["exp"]}}
        if self._issuer:
            kwargs["issuer"] = self._issuer
        if self._audience:
            kwargs["audience"] = self._audience
        try:
            return pyjwt.decode(token, self._key, **kwargs)
        except pyjwt.ExpiredSignatureError:
            raise AuthError("Token expired.") from None
        except Exception as exc:
            raise AuthError(f"Invalid token: {exc}") from None

    def identity(self, token: str) -> IdentityClaims:
        """校验 token 并转换为 IdentityClaims。
返回:
    身份声明对象。
        """
        return IdentityClaims.from_payload(self.verify(token))

    def context(self, token: str) -> AgentContext:
        """校验 token 并直接转换为 AgentContext。
返回:
    包含身份与隔离边界的 AgentContext。
        """
        return self.identity(token).to_context()


def verify_token(
    token: str,
    config: AuthConfig | None = None,
    **kwargs: Any,
) -> IdentityClaims:
    """快捷口：校验 token 并返回 IdentityClaims。
参数:
    token: 待校验的 JWT。
    config: 可选配置，None 时使用默认值。
    **kwargs: 其他传递给 TokenVerifier 的参数。
返回:
    身份声明对象。
    """
    return TokenVerifier(config, **kwargs).identity(token)
