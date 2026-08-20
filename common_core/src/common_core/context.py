from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentContext:
    """AgentContext是贯穿整个请求生命周期、用于多租户隔离的上下文对象。它把 JWT 鉴权解析出的身份信息（租户/知识库/会话/请求）打包成不可变对象，贯穿检索、重排、缓存、指标打点等所有环节。

设计要点：
- frozen 冻结的 dataclass：字段一旦创建便不可修改，天然线程安全、可哈希；
- 所有字段默认为空字符串，避免 None 判断；
- tenant_id / kb_id 是隔离边界，所有数据访问和缓存键都必须带上它们。

属性:
    tenant_id: 租户 ID，数据隔离的第一层边界。 JWT claims 中解析得到；为空时无法隔离数据。
    kb_id: 知识库 ID，在 tenant_id 基础上进一步划分｢知识库｣粒度。每个知识库是独立的检索/缓存命名空间。
    session_id: 会话 ID，用于把同一会话的所有请求归组，便于做会话级上下文与指标聚合。
    request_id: 请求 ID，用于链路追踪（span/trace）。一次用户请求对应一个唯一的 request_id。
    user_id: 用户 ID，通常取自 JWT 的 sub（subject）声明，用于按用户维度审计与个性化。
    """

    tenant_id: str
    kb_id: str
    session_id: str = ""
    request_id: str = ""
    user_id: str = ""

    @classmethod
    def from_claims(cls, claims: dict[str, object]) -> "AgentContext":
        """从 JWT claims 字典构造一个 AgentContext。

参数:
    claims: 由 TokenVerifier 解码出的 JWT payload。期望包含 tenant_id / kb_id / session_id / request_id 等自定义声明，sub（subject，通常代表用户ID）用作 user_id。

返回:
    一个填充好各字段的 AgentContext 实例。任何字段缺失都会被安全地替换为空字符串。
        """
        return cls(
            tenant_id=str(claims.get("tenant_id") or ""),
            kb_id=str(claims.get("kb_id") or ""),
            session_id=str(claims.get("session_id") or ""),
            request_id=str(claims.get("request_id") or ""),
            user_id=str(claims.get("sub") or claims.get("user_id") or ""),
        )
