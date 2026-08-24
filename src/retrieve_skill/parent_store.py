"""MySQL-backed parent block store used after child reranking."""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from typing import Any, Mapping

from common_core.config import env_int, env_str
from common_core.context import AgentContext

logger = logging.getLogger(__name__)

_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_PARENT_FIELDS = (
    "parent_id",
    "title",
    "content",
    "summary",
    "category",
    "source_type",
    "source_id",
    "source_uri",
    "visibility",
    "doc_version",
    "content_sha256",
)


@dataclass(slots=True)
class ParentStoreConfig:
    """Connection and table settings for the authoritative parent store."""

    host: str = "127.0.0.1"
    port: int = 3306
    database: str = ""
    user: str = ""
    password: str = ""
    charset: str = "utf8mb4"
    connect_timeout: int = 3
    read_timeout: int = 5
    pool_max_connections: int = 10
    pool_min_cached: int = 1
    pool_max_cached: int = 5
    pool_blocking: bool = True
    table: str = "rag_parent_block"
    status: str = "active"

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "ParentStoreConfig":
        return cls(
            host=env_str("MYSQL_HOST", default="127.0.0.1", env=env),
            port=env_int("MYSQL_PORT", default=3306, env=env),
            database=env_str("MYSQL_DATABASE", env=env),
            user=env_str("MYSQL_USER", env=env),
            password=env_str("MYSQL_PASSWORD", env=env),
            charset=env_str("MYSQL_CHARSET", default="utf8mb4", env=env),
            connect_timeout=env_int("MYSQL_CONNECT_TIMEOUT", default=3, env=env),
            read_timeout=env_int("MYSQL_READ_TIMEOUT", default=5, env=env),
            pool_max_connections=env_int(
                "MYSQL_POOL_MAX_CONNECTIONS", default=10, env=env
            ),
            pool_min_cached=env_int("MYSQL_POOL_MIN_CACHED", default=1, env=env),
            pool_max_cached=env_int("MYSQL_POOL_MAX_CACHED", default=5, env=env),
            table=env_str("RAG_PARENT_TABLE", default="rag_parent_block", env=env),
            status=env_str("RAG_PARENT_STATUS", default="active", env=env),
        )

    def validate(self) -> list[str]:
        missing = [
            name
            for name, value in (
                ("MYSQL_HOST", self.host),
                ("MYSQL_DATABASE", self.database),
                ("MYSQL_USER", self.user),
                ("RAG_PARENT_TABLE", self.table),
                ("RAG_PARENT_STATUS", self.status),
            )
            if not str(value).strip()
        ]
        if not _IDENTIFIER_PATTERN.fullmatch(self.table):
            missing.append("RAG_PARENT_TABLE (invalid identifier)")
        return missing


class MySQLParentStore:
    """Batch loader for canonical parent documents stored in MySQL."""

    def __init__(
        self,
        config: ParentStoreConfig | None = None,
        *,
        pool: Any = None,
    ) -> None:
        self.config = config or ParentStoreConfig.from_env()
        self._pool_obj = pool

    def _ensure_pool(self) -> Any:
        if self._pool_obj is not None:
            return self._pool_obj
        missing = self.config.validate()
        if missing:
            raise RuntimeError(f"parent store is not configured: {', '.join(missing)}")
        try:
            import pymysql
            from dbutils.pooled_db import PooledDB
        except ImportError as exc:
            raise RuntimeError(
                "MySQL parent store requires 'retrieve-skill[mysql]'"
            ) from exc

        self._pool_obj = PooledDB(
            creator=pymysql,
            maxconnections=self.config.pool_max_connections,
            mincached=self.config.pool_min_cached,
            maxcached=self.config.pool_max_cached,
            blocking=self.config.pool_blocking,
            host=self.config.host,
            port=self.config.port,
            user=self.config.user,
            password=self.config.password or None,
            database=self.config.database,
            charset=self.config.charset,
            connect_timeout=self.config.connect_timeout,
            read_timeout=self.config.read_timeout,
            autocommit=True,
            cursorclass=pymysql.cursors.DictCursor,
        )
        return self._pool_obj

    def _query(self, parent_ids: list[str], context: AgentContext) -> dict[str, dict[str, Any]]:
        conn = None
        placeholders = ",".join("%s" for _ in parent_ids)
        sql = f"""
            SELECT {', '.join(_PARENT_FIELDS)}
            FROM {self.config.table}
            WHERE tenant_id = %s
              AND kb_id = %s
              AND status = %s
              AND parent_id IN ({placeholders})
        """
        try:
            conn = self._ensure_pool().connection()
            with conn.cursor() as cursor:
                cursor.execute(
                    sql,
                    (context.tenant_id, context.kb_id, self.config.status, *parent_ids),
                )
                rows = list(cursor.fetchall())
        finally:
            if conn is not None:
                conn.close()

        return {
            str(row.get("parent_id", "") or ""): dict(row)
            for row in rows
            if str(row.get("parent_id", "") or "").strip()
        }

    async def aget_parent_blocks(
        self,
        parent_ids: list[str],
        *,
        context: AgentContext,
    ) -> dict[str, dict[str, Any]]:
        """Read active parents in one query; missing IDs are omitted."""
        unique_ids = list(dict.fromkeys(str(item).strip() for item in parent_ids if str(item).strip()))
        if not unique_ids:
            return {}
        if not context.tenant_id or not context.kb_id:
            raise ValueError("parent store requires tenant_id and kb_id")
        return await asyncio.to_thread(self._query, unique_ids, context)

    def close(self) -> None:
        if self._pool_obj is not None:
            self._pool_obj.close()
            self._pool_obj = None


__all__ = ["MySQLParentStore", "ParentStoreConfig"]
