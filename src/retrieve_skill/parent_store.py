"""子块精排后使用的 MySQL 父块权威存储。

整体流程：
1. 子块在 Milvus 中完成召回和精排后，只携带 ``parent_id``；
2. 本模块按租户、知识库和状态过滤，批量读取这些 ``parent_id`` 对应的权威父块；
3. 调用方再根据父块版本、正文和 token 预算组装最终上下文。

这里刻意不提供写入能力，避免检索服务绕过知识入库链路修改父块数据。
"""

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
    "tenant_id",
    "kb_id",
    "parent_id",
    "title",
    "content",
    "summary",
    "category",
    "source_type",
    "source_id",
    "source_uri",
    "visibility",
    "status",
    "doc_version",
    "content_sha256",
)


@dataclass(slots=True)
class ParentStoreConfig:
    """权威父块存储的连接与表配置。

    字段:
        host / port: MySQL 服务地址和端口。
        database: 包含父块表的数据库名。
        user / password: 连接 MySQL 的账号信息；密码可为空（例如本地免密配置）。
        charset: 连接字符集，默认使用 utf8mb4 以兼容中文和 emoji。
        connect_timeout / read_timeout: 建连和单次查询的超时秒数，
            避免一次父块回源拖慢整条检索请求。
        pool_max_connections / pool_min_cached / pool_max_cached /
            pool_blocking: DBUtils 连接池容量和取连接时的阻塞策略。
        table: 权威父块表名。SQL 中会拼接表名，因此必须是合法标识符。
        status: 允许返回的父块业务状态，默认只读 active 数据。
    """

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
        """从环境变量构建配置。

        参数:
            env: 可选环境变量映射；None 时读取当前进程环境变量。

        返回:
            ParentStoreConfig 实例；缺失的必填项会在 validate() 中报告。
        """
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
        """校验启动和执行 SQL 所需的关键配置。

        返回:
            缺失或非法配置名列表；列表为空表示配置可用。
            这里不直接抛异常，便于上层汇总所有问题后给出更清晰的错误。
        """
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
    """批量加载 MySQL 中权威父块内容的只读存储。

    典型流程：
    1. 检索管线把精排达标子块的 ``parent_id`` 去重；
    2. 调用 aget_parent_blocks() 在线程池中执行一次 IN 查询；
    3. 结果以 ``parent_id -> 父块字段字典`` 的形式返回，缺失 ID 不存在映射。
    """

    def __init__(
        self,
        config: ParentStoreConfig | None = None,
        *,
        pool: Any = None,
    ) -> None:
        """初始化存储；连接池延迟到第一次真正查询时创建。

        参数:
            config: MySQL 配置；None 时从环境变量构建。
            pool: 可注入的连接池或具备 connection()/close() 能力的测试替身；
                注入后不再创建真实 DBUtils 连接池。
        """
        self.config = config or ParentStoreConfig.from_env()
        self._pool_obj = pool

    def _ensure_pool(self) -> Any:
        """获取或惰性创建 MySQL 连接池。

        返回:
            当前可用的连接池对象。

        异常:
            RuntimeError: 配置缺失，或未安装 pymysql/dbutils 依赖。
        """
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
        """同步执行一次父块批量查询。

        流程:
        1. 按 parent_ids 数量生成 SQL 占位符，避免字符串拼值造成注入风险；
        2. 强制叠加 tenant_id/kb_id/status 条件，保证多租户隔离；
        3. 从连接池取连接、执行查询并确保 finally 中归还连接；
        4. 把行数据转换成以 parent_id 为键的字典，方便调用方随机访问。

        参数:
            parent_ids: 需要加载的父块主键列表；调用方应先去重。
            context: agent 上下文，提供租户和知识库隔离条件。

        返回:
            ``dict[parent_id, row]``；数据库中不存在或被过滤掉的 ID 不出现在结果中。
        """
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
        """异步批量读取当前有效的父块内容。

        参数:
            parent_ids: 精排达标子块指向的父块 ID 列表；允许重复或包含空值，
                方法内部会去重和清洗。
            context: agent 上下文；tenant_id 和 kb_id 必须同时存在，
                否则宁可失败也不跨租户查询。

        返回:
            ``dict[parent_id, 父块字段字典]``。缺失 ID 会被省略，由调用方根据
            版本校验和统计信息决定如何降级或清理缓存。

        异常:
            ValueError: context 中缺少 tenant_id 或 kb_id。
        """
        unique_ids = list(dict.fromkeys(str(item).strip() for item in parent_ids if str(item).strip()))
        if not unique_ids:
            return {}
        if not context.tenant_id or not context.kb_id:
            raise ValueError("parent store requires tenant_id and kb_id")
        return await asyncio.to_thread(self._query, unique_ids, context)

    def close(self) -> None:
        """关闭已创建的连接池；尚未创建连接池时是安全的 no-op。"""
        if self._pool_obj is not None:
            self._pool_obj.close()
            self._pool_obj = None


__all__ = ["MySQLParentStore", "ParentStoreConfig"]
