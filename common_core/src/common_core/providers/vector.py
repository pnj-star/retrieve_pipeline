"""兼容 Milvus 的向量存储封装，支持稠密向量 + BM25 混合检索。该封装层做通用设计：集合名称、向量字段、返回字段全部由调用方传入。业务模块自行管理对应的 schema 结构以及业务过滤字段。"""

from __future__ import annotations

import asyncio
import logging
import math
from collections.abc import Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from typing import Any

from ..config import VectorStoreConfig

logger = logging.getLogger(__name__)


def _escape(value: str) -> str:
    """转义字符串中的反斜柱与双引号，使其可安全地嵌入 Milvus 表达式。
参数:
    value: 待嵌入的字符串。
返回:
    转义后的字符串。
    """
    return str(value).replace("\\", "\\\\").replace('"', '\\"')

def build_filter_expr(
    filters: Mapping[str, str | Iterable[str]] | None
) -> str:  # todo 后续了解
    """根据字段过滤器构建通用的 Milvus 查询表达式。字符串值会生成等值判断；可迭代对象会生成 field in [...] 包含判断。
参数:
    filters: 字段名到值的映射，值可为字符串或可迭代对象。
返回:
    组合好的 Milvus 表达式；无有效过滤时返回空字符串。
    """
    parts: list[str] = []
    for field_name, value in (filters or {}).items():
        if not field_name:
            continue
        if isinstance(value, str):
            if value:
                parts.append(f'{field_name} == "{_escape(value)}"')
        else:
            values = [
                str(item)
                for item in value
                if item is not None and str(item).strip()
            ]
            if values:
                quoted = ", ".join(f'"{_escape(item)}"' for item in values)
                parts.append(f"{field_name} in [{quoted}]")
    return " and ".join(parts) if parts else ""


def rrf_fuse(
    dense_results: list[dict[str, Any]],
    sparse_results: list[dict[str, Any]],
    k: int = 60,
) -> list[dict[str, Any]]:
    """对稠密向量检索结果与 BM25 关键词检索结果执行 RRF（倒数秩融合）。
原理：同一文档在两路搜索中的秩位越靠前，融合分 1/(k+rank+1) 越高；最终按融合分降序返回。
参数:
    dense_results: 稠密检索结果列表。
    sparse_results: BM25 检索结果列表。
    k: RRF 平滑系数，默认 60。
返回:
    融合后按 fusion_score 降序排列的文档列表。
    """
    fused: dict[str, dict[str, Any]] = {}

    def _merge(doc: dict[str, Any], rank: int, via: str) -> None:
        """把单路检索结果合并进融合字典。

        参数:
            doc: 单路检索到的一篇文档。
            rank: 该文档在对应检索路中的名次。
            via: 来源标识（"dense"/"sparse"）。

        提交（无返回值）：累加融合分并记录来源名次。
        """
        doc_id = str(doc["id"])
        item = fused.setdefault(doc_id, dict(doc, fusion_score=0.0))
        item["fusion_score"] += 1.0 / (k + rank + 1)
        item[f"{via}_rank"] = rank
        if via == "dense":
            item["score"] = max(item.get("score", 0.0), float(doc.get("score", 0.0)))
        else:
            bm25 = float(doc.get("bm25_score", 0.0))
            item["score"] = max(item.get("score", 0.0), 1.0 / (1.0 + math.exp(-bm25)))

    for index, doc in enumerate(dense_results):
        _merge(doc, index, "dense")
    for index, doc in enumerate(sparse_results):
        _merge(doc, index, "sparse")

    return sorted(fused.values(), key=lambda item: item["fusion_score"], reverse=True)


class MilvusVectorStore:
    """基于 Milvus 的向量存储适配器，提供稠密、BM25、混合三种检索方式的同步与异步接口，可选配置图片检索。    """
    def __init__(
        self,
        config: VectorStoreConfig | None = None,
        *,
        host: str | None = None,
        port: int | None = None,
        user: str | None = None,
        password: str | None = None,
        secure: bool | None = None,
        db_name: str | None = None,
        text_collection: str | None = None,
        image_collection: str | None = None,
        dim: int | None = None,
        max_workers: int = 16,
    ) -> None:
        """初始化向量存储。
参数:
    config: 向量库配置；None 时用默认值。
    host/port/user/password/secure/db_name/text_collection/image_collection/dim: 可选覆盖值，优先级高于 config。
    max_workers: 混合检索并行执行的线程数。
        """
        base = config or VectorStoreConfig()
        self.config = replace(
            base,
            host=host or base.host,
            port=port if port is not None else base.port,
            user=base.user if user is None else user,
            password=base.password if password is None else password,
            secure=base.secure if secure is None else secure,
            db_name=base.db_name if db_name is None else db_name,
            text_collection=(
                base.text_collection if text_collection is None else text_collection
            ),
            image_collection=(
                base.image_collection if image_collection is None else image_collection
            ),
            dim=base.dim if dim is None else dim,
        )
        self._connected = False
        self._collections: dict[str, Any] = {}
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="common_core_milvus",
        )

    def connect(self) -> None:
        """建立与 Milvus 的连接（仅此一次），并按配置填充 user/secure/db_name 等。
        """
        if self._connected:
            return
        from pymilvus import connections

        conn_args: dict[str, Any] = {
            "alias": "default",
            "host": self.config.host,
            "port": self.config.port,
        }
        if self.config.user:
            conn_args["user"] = self.config.user
            conn_args["password"] = self.config.password
            conn_args["secure"] = self.config.secure
        if self.config.db_name:
            conn_args["db_name"] = self.config.db_name
        connections.connect(**conn_args)
        self._connected = True

    def _ensure(self, collection_name: str) -> Any:
        """确保集合已加载并返回缓存的 Collection 对象，缺少时自动连接并加载。
参数:
    collection_name: 要访问的集合名称。
返回:
    已加载的 pymilvus.Collection 实例。
异常:
    ValueError: 集合不存在时抛出。
        """
        if collection_name in self._collections:
            return self._collections[collection_name]
        self.connect()
        from pymilvus import Collection, utility

        if not utility.has_collection(collection_name, using="default"):
            raise ValueError(f"Milvus collection not found: {collection_name}")
        collection = Collection(collection_name, using="default")
        collection.load()
        self._collections[collection_name] = collection
        return collection

    def _to_docs(
        self,
        results: Any,
        output_fields: list[str],
        score_key: str,
    ) -> list[dict[str, Any]]:
        """把 Milvus 的原始 hit 转换为统一的字典列表，并批量读出 output_fields 指定的字段。
参数:
    results: Milvus search 返回的结果。
    output_fields: 要包含在输出中的字段列表。
    score_key: 分数存放的字段名（如 score/bm25_score）。
返回:
    每项包含 id、score 与 output_fields 的字典列表。
        """
        docs: list[dict[str, Any]] = []
        for hits in results:
            for hit in hits:
                item: dict[str, Any] = {
                    "id": str(hit.id),
                    score_key: float(hit.distance),
                }
                for field_name in output_fields:
                    if field_name == "id":
                        continue
                    entity = hit.entity
                    if isinstance(entity, dict):
                        item[field_name] = entity.get(field_name)
                    elif hasattr(entity, "get"):
                        item[field_name] = entity.get(field_name)
                    else:
                        item[field_name] = None
                docs.append(item)
        return docs

    def search_dense(
        self,
        collection_name: str,
        embedding: list[float],
        *,
        top_k: int = 20,
        output_fields: Iterable[str] | None = None,
        filter_expr: str | None = None,
        anns_field: str = "embedding",
    ) -> list[dict[str, Any]]:
        """稠密向量检索，使用 COSINE 相似度。
参数:
    collection_name: 集合名。
    embedding: 查询向量。
    top_k: 返回结果上限。
    output_fields: 返回字段（默认仅 id）。
    filter_expr: 可选的过滤表达式。
    anns_field: 向量字段名，默认 "embedding"。
返回:
    字典列表。
        """
        collection = self._ensure(collection_name)
        fields = list(output_fields or ["id"])
        results = collection.search(
            data=[embedding],
            anns_field=anns_field,
            param={"metric_type": "COSINE", "params": {"nprobe": 16}},
            limit=top_k,
            expr=filter_expr or "",
            output_fields=fields,
        )
        return self._to_docs(results, fields, "score")

    def search_bm25(
        self,
        collection_name: str,
        query: str,
        *,
        top_k: int = 20,
        output_fields: Iterable[str] | None = None,
        filter_expr: str | None = None,
        anns_field: str = "sparse",
    ) -> list[dict[str, Any]]:
        """BM25 关键词检索。
参数:
    collection_name: 集合名。
    query: 查询文本。
    top_k: 返回结果上限。
    output_fields: 返回字段。
    filter_expr: 可选过滤表达式。
    anns_field: 稀疏字段名，默认 "sparse"。
返回:
    字典列表，分数放在 bm25_score 字段。
        """
        collection = self._ensure(collection_name)
        fields = list(output_fields or ["id"])
        results = collection.search(
            data=[query],
            anns_field=anns_field,
            param={"metric_type": "BM25"},
            limit=top_k,
            expr=filter_expr or "",
            output_fields=fields,
        )
        return self._to_docs(results, fields, "bm25_score")

    def search_hybrid(
        self,
        collection_name: str,
        query: str,
        embedding: list[float] | None,
        *,
        top_k: int = 20,
        rrf_top_k: int = 20,
        rrf_k: int = 60,
        output_fields: Iterable[str] | None = None,
        filter_expr: str | None = None,
    ) -> list[dict[str, Any]]:
        """混合检索：并行执行稠密向量检索与 BM25 检索，缺少一路时降级为单路，两路都有时用 RRF 融合。
参数:
    collection_name: 集合名。
    query: 查询文本（用于 BM25）。
    embedding: 查询向量（用于稠密），可为 None。
    top_k: 各路排名检索的结果数上限。
    rrf_top_k: 融合后返回的结果上限。
    rrf_k: RRF 平滑系数。
    output_fields: 返回字段。
    filter_expr: 可选过滤表达式，稀疏检索出现空结果时不会回退到无过滤检索。
返回:
    字典列表。
        """
        self._ensure(collection_name)

        def safe_dense() -> list[dict[str, Any]]:
            """安全地执行稠密检索：失败时记录告警并降级为空列表。"""
            try:
                return self.search_dense(
                    collection_name,
                    embedding or [],
                    top_k=top_k,
                    output_fields=output_fields,
                    filter_expr=filter_expr,
                )
            except Exception as exc:
                logger.warning("dense search failed, degraded: %s", exc)
                return []

        def safe_bm25() -> list[dict[str, Any]]:
            """安全地执行 BM25 检索：失败时记录告警并降级为空列表。"""
            try:
                return self.search_bm25(
                    collection_name,
                    query,
                    top_k=top_k,
                    output_fields=output_fields,
                    filter_expr=filter_expr,
                )
            except Exception as exc:
                logger.warning("bm25 search failed, degraded: %s", exc)
                return []

        dense_future = self._executor.submit(safe_dense) if embedding else None
        sparse_future = (
            self._executor.submit(safe_bm25) if query and query.strip() else None
        )
        dense_results = dense_future.result() if dense_future else []
        sparse_results = sparse_future.result() if sparse_future else []

        if not dense_results and not sparse_results:
            if filter_expr:
                logger.warning(
                    "hybrid search empty with filter %r; refusing unscoped fallback",
                    filter_expr,
                )
            return []
        if not dense_results:
            for doc in sparse_results:
                doc.setdefault(
                    "score",
                    1.0 / (1.0 + math.exp(-float(doc.get("bm25_score", 0.0)))),
                )
            return sparse_results[:rrf_top_k]
        if not sparse_results:
            return dense_results[:rrf_top_k]
        return rrf_fuse(dense_results, sparse_results, k=rrf_k)[:rrf_top_k]

    def search_image(
        self,
        embedding: list[float],
        *,
        top_k: int = 20,
        output_fields: Iterable[str] = ("id", "image_url", "description"),
        filter_expr: str | None = None,
    ) -> list[dict[str, Any]]:
        """使用图片向量进行图像检索，集合名由配置的 image_collection 提供。
参数:
    embedding: 图片查询向量。
    top_k: 返回结果上限。
    output_fields: 返回字段，默认 id/image_url/description。
    filter_expr: 可选过滤表达式。
返回:
    字典列表。
异常:
    ValueError: 未配置 image_collection 时抛出。
        """
        if not self.config.image_collection:
            raise ValueError("image_collection is not configured.")
        return self.search_dense(
            self.config.image_collection,
            embedding,
            top_k=top_k,
            output_fields=output_fields,
            filter_expr=filter_expr,
        )

    async def a_search_dense(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        """稠密检索的异步版，在辅助线程执行。

        参数:
            *args/**kwargs: 与 ``search_dense`` 相同的参数。

        返回:
            检索结果的字典列表。
        """
        return await asyncio.to_thread(self.search_dense, *args, **kwargs)

    async def a_search_hybrid(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        """混合检索的异步版，在辅助线程执行。

        参数:
            *args/**kwargs: 与 ``search_hybrid`` 相同的参数。

        返回:
            检索结果的字典列表。
        """
        return await asyncio.to_thread(self.search_hybrid, *args, **kwargs)

    def close(self) -> None:
        """释放资源：关闭线程池并断开 Milvus 连接（若已建立）。
        """
        self._executor.shutdown(wait=False, cancel_futures=True)
        if self._connected:
            try:
                from pymilvus import connections

                connections.disconnect("default")
            except Exception:
                logger.debug("milvus disconnect failed (ignored)", exc_info=True)
            self._connected = False
