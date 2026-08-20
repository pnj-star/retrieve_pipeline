from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class QueryRequest:
    """一次具体的数据/检索查询请求。它是各个提供者（向量库、缓存、LLM 等）之间传递的统一请求对象，用数据类地加工（dataclass）描述请求的身份边界与操作内容。

属性:
    tenant_id: 租户 ID，数据隔离的第一层边界，提供者必须按它筛选数据。
    kb_id: 知识库 ID，在租户内进一步划分检索空间。
    action: 要执行的操作名称（如 "retrieve"/"query_expansion"）。
    params: 操作的参数字典，由上下游商定字段名与含义。
    """
    tenant_id: str
    kb_id: str
    action: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryResult:
    """查询的统一返回结果，包含值得信任的事实列表与来源描述。

属性:
    facts: 检索到的事实/片段列表，各项为字典（通常包含同源的 content 等）。
    source: 结果的来源标识，用于引用与审计。
    confidence: 结果的稳定性置信度，0～1 之间，默认 1.0。
    error: 非空时表示查询失败，该字段携带错误信息；为 None 表示成功。
    """
    facts: list[dict[str, Any]]
    source: str
    confidence: float = 1.0
    error: str | None = None


class DataProvider(Protocol):
    """数据/检索提供者的接口约定。实现者必须提供异步 query 方法，接受一个 QueryRequest 并返回 QueryResult。"""
    async def query(self, request: QueryRequest) -> QueryResult:
        """异步执行一次数据/检索查询。

        参数:
            request: 统一的查询请求对象，包含身份边界与操作内容。

        返回:
            查询结果 QueryResult。
        """
        ...


class LLMProvider(Protocol):
    """大语言模型提供者的接口约定。实现者必须提供异步 complete 方法，接受 message 列表并返回生成的文本。"""
    async def complete(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        """异步完成一次对话生成。

        参数:
            messages: 对话消息列表，各项为 {"role": ..., "content": ...} 字典。
            **kwargs: 其它可选生成参数。

        返回:
            生成的文本字符串。
        """
        ...


class Embedder(Protocol):
    """向量化器提供者的接口约定。实现者必须提供单条与批量的异步向量化方法：embed 与 embed_many。"""
    async def embed(self, text: str) -> list[float]:
        """异步对单条文本进行向量化。

        参数:
            text: 待向量化的文本。

        返回:
            该文本对应的浮点向量。
        """
        ...

    async def embed_many(self, texts: list[str]) -> list[list[float]]:
        """异步对一批文本进行向量化。

        参数:
            texts: 待向量化的文本列表。

        返回:
            与输入一一对应的向量列表。
        """
        ...


class Cache(Protocol):
    """缓存提供者的接口约定。实现者必须提供 get/set 方法，以 scope+material 结合 tenant/kb 生成归一缓存键。"""
    def get(self, scope: str, material: str, *, tenant_id: str = "", kb_id: str = "") -> str | None:
        """读取缓存。

        参数:
            scope: 缓存领域/用途。
            material: 缓存的原始内容。
            tenant_id/kb_id: 隔离边界。

        返回:
            命中时返回存储的字符串；未命中时返回 None。
        """
        ...

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
        """写入缓存并设置 TTL。

        参数:
            scope: 缓存领域/用途。
            material: 缓存的原始内容。
            value: 要写入的值。
            ttl: 有效期（秒）；为 None 时使用实现自带的默认值。
            tenant_id/kb_id: 隔离边界。

        返回:
            写入成功返回 True，失败返回 False。
        """
        ...


class VectorStore(Protocol):
    """向量库提供者的接口约定。实现者必须提供异步混合检索 a_search_hybrid，支持稀疏+稠密融合与过滤表达式。"""
    async def a_search_hybrid(
        self,
        collection_name: str,
        query: str,
        embedding: list[float] | None,
        *,
        top_k: int = 20,
        rrf_top_k: int = 20,
        rrf_k: int = 60,
        output_fields: list[str] | None = None,
        filter_expr: str | None = None,
    ) -> list[dict[str, Any]]:
        """异步执行稠密+稀疏混合检索。

        参数:
            collection_name: 集合名。
            query: 查询文本（用于稀疏检索）。
            embedding: 查询向量（用于稠密检索），可为 None。
            top_k: 各路排名检索的结果数上限。
            rrf_top_k: 融合后返回的结果上限。
            rrf_k: RRF 融合平滑系数。
            output_fields: 返回字段列表。
            filter_expr: 可选的过滤表达式。

        返回:
            检索结果的字典列表。
        """
        ...
