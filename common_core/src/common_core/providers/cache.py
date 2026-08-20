"""兼容 Redis 协议的缓存适配器，支持多租户隔离的键命名空间。"""

from __future__ import annotations

import hashlib
import logging
import threading
from typing import Any

from ..config import CacheConfig

logger = logging.getLogger(__name__)

_shared_clients: dict[tuple[Any, ...], Any] = {}
_shared_clients_lock = threading.Lock()


def _create_redis_client(config: CacheConfig) -> Any:
    """根据配置创建一个 redis 客户端，并启用 decode_responses 以直接返回 str。
参数:
    config: 缓存配置，包含 host/port/password/db等。
返回:
    已初始化的 redis.Redis 客户端实例。
    """
    import redis

    return redis.Redis(
        host=config.host,
        port=config.port,
        password=config.password or None,
        db=config.db,
        decode_responses=True,
        socket_connect_timeout=config.socket_timeout,
        protocol=2,
    )


def response_ttl_for(
    response: str,
    *,
    short_ttl: int = 300,
    long_ttl: int = 1800,
    max_short_len: int = 20,
) -> int:
    """对可疑的短响应使用较短的缓存 TTL，其余情况使用较长 TTL。
参数:
    response: 要缓存的响应文本。
    short_ttl: 短响应的有效期（秒），默认 300。
    long_ttl: 长响应的有效期（秒），默认 1800。
    max_short_len: 判断“短响应”的字符上限，默认 20。
返回:
    应用到该响应的 TTL（秒数）。
    """
    return long_ttl if len(response.strip()) >= max_short_len else short_ttl


class RedisCache:
    """基于 Redis 的缓存实现，支持多租户/多知识库隔离、可共享连接、容错降级。
所有读写操作在 Redis 不可用时都会安全降级（输出空值/返回 False）而不会抛出异常，避免缓存失效影响主流程。    """
    def __init__(
        self,
        config: CacheConfig | None = None,
        *,
        host: str | None = None,
        port: int | None = None,
        password: str | None = None,
        db: int | None = None,
        key_prefix: str | None = None,
        client: Any | None = None,
        shared: bool = False,
    ) -> None:
        """初始化 Redis 缓存。
参数:
    config: 缓存配置；为 None 时使用默认配置。
    host/port/password/db/key_prefix: 可选的个别覆盖值，优先级高于 config。
    client: 预置的 redis 客户端，供测试注入。
    shared: 为 True 时从过程级共享连接池提取客户端，节省连接。
        """
        self.config = config or CacheConfig()
        if host is not None:
            self.config.host = host
        if port is not None:
            self.config.port = port
        if password is not None:
            self.config.password = password
        if db is not None:
            self.config.db = db
        if key_prefix is not None:
            self.config.key_prefix = key_prefix
        self._client: Any = client
        self.shared = shared

    @classmethod
    def _shared_key(cls, config: CacheConfig) -> tuple[Any, ...]:
        """生成标识一个共享连接的全部键字段疏组合，用于共享连接池的字典键。

        参数:
            config: 缓存配置。

        返回:
            一个用于字典查找的元组。
        """
        return (
            config.host,
            config.port,
            config.db,
            config.key_prefix,
            config.password,
            config.socket_timeout,
        )

    def client(self) -> Any:
        """获取客户端，缺少时按 shared 标志创建或获取共享或独立实例。
返回:
    redis 客户端实例。
        """
        if self._client is None:
            if self.shared:
                key = self._shared_key(self.config)
                with _shared_clients_lock:
                    if key not in _shared_clients:
                        _shared_clients[key] = _create_redis_client(self.config)
                    self._client = _shared_clients[key]
            else:
                self._client = _create_redis_client(self.config)
        return self._client

    def _safe_client(self) -> Any:
        """安全获取客户端；创建失败时记录调试日志并返回 None（让调用者降级）。
返回:
    客户端实例或 None。
        """
        try:
            return self.client()
        except Exception as exc:
            logger.debug("redis client unavailable: %s", exc)
            return None

    def key(
        self,
        scope: str,
        material: str,
        *,
        tenant_id: str = "",
        kb_id: str = "",
    ) -> str:
        """构建缓存键：缓存换算 material 为 SHA-256 摘要，降低巨长内容对键长的影响。
参数:
    scope: 缓存领域/用途（如 "rerank"/"rewrite"）。
    material: 缓存的原始内容，用于计算摘要。
    tenant_id/kb_id: 隔离边界，不同租户/知识库不会互相命中缓存。
返回:
    组合好的缓存键字符串。
        """
        parts = [self.config.key_prefix, scope]
        if tenant_id:
            parts.append(tenant_id)
        if kb_id:
            parts.append(kb_id)
        digest = hashlib.sha256(str(material).strip().encode("utf-8")).hexdigest()
        return ":".join(parts + [digest])

    def get(
        self,
        scope: str,
        material: str,
        *,
        tenant_id: str = "",
        kb_id: str = "",
    ) -> str | None:
        """读取缓存。
返回:
    命中时返回存储的字符串；未命中/客户端不可用时返回 None。
        """
        client = self._safe_client()
        if client is None:
            return None
        try:
            return client.get(
                self.key(scope, material, tenant_id=tenant_id, kb_id=kb_id)
            )
        except Exception as exc:
            logger.debug("cache get failed (degraded to None): %s", exc)
            return None

    def set(
        self,
        scope: str,
        material: str,
        value: str,
        *,
        ttl: int | None = None,
        tenant_id: str = "",
        kb_id: str = "",
    ) -> bool:
        """写入缓存并设定 TTL。
参数:
    ttl: 有效期（秒）；None 时用配置的 default_ttl。
返回:
    写入成功返回 True；失败或客户端不可用返回 False。
        """
        client = self._safe_client()
        if client is None:
            return False
        try:
            client.setex(
                self.key(scope, material, tenant_id=tenant_id, kb_id=kb_id),
                ttl if ttl is not None else self.config.default_ttl,
                value,
            )
            return True
        except Exception as exc:
            logger.debug("cache set failed (ignored): %s", exc)
            return False

    def delete(
        self,
        scope: str,
        material: str,
        *,
        tenant_id: str = "",
        kb_id: str = "",
    ) -> bool:
        """删除缓存键。
返回:
    删除成功返回 True；未命中或客户端不可用返回 False。
        """
        client = self._safe_client()
        if client is None:
            return False
        try:
            return bool(
                client.delete(
                    self.key(scope, material, tenant_id=tenant_id, kb_id=kb_id)
                )
            )
        except Exception as exc:
            logger.debug("cache delete failed (ignored): %s", exc)
            return False

    def ping(self) -> bool:
        """检测客户端与 Redis 连接是否可用。
返回:
    可用时返回 True；否则返回 False。
        """
        client = self._safe_client()
        if client is None:
            return False
        try:
            return bool(client.ping())
        except Exception:
            return False

    def close(self) -> None:
        """释放资源。共享连接时仅清空本实例引用，不关闭共享客户端；独立连接时关闭并清空。
        """
        if self.shared:
            self._client = None
            return
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                logger.debug("redis close failed (ignored)", exc_info=True)
            self._client = None
