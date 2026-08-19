r"""rag_skill 查询改写真实链路冒烟。

依赖（本地已具备）：Docker 内的 Milvus + Redis、缓存的
``BAAI/bge-small-zh-v1.5`` embedding 模型、pymilvus / sentence_transformers。

用途：在真实 Milvus（稠密 + BM25 混合检索）与真实 embedding 下，验证查询改写
开 / 关无回归，以及 ``llm_rewrite`` / ``query_expansion`` 真正进入检索并透出
``rewritten_query``。LLM 改写文本用可控桩代替（无真实 LLM 凭据）。

运行（请先启动 Docker 内的 Milvus 与 Redis）：
    $env:HF_HUB_OFFLINE='1'
    & D:\python\python.exe rag_skill\scripts\smoke_query_rewrite.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "..", "..", "common_core", "src"),
)

import asyncio
from typing import Any

from common_core.config import CacheConfig, RuntimeConfig, VectorStoreConfig
from common_core.config import LLMConfig
from common_core.context import AgentContext
from common_core.providers import OpenAICompatibleLLM
from rag_skill.pipeline import RagPipeline


MILVUS_HOST = "127.0.0.1"
MILVUS_PORT = 19531
REDIS_HOST = "127.0.0.1"
REDIS_PORT = 6380
EMBED_MODEL = "BAAI/bge-small-zh-v1.5"
COLLECTION = "smoke_rewrite_kb"
TENANT = "t1"
KB = "kb1"
DIM = 512

LLM_BASE_URL = os.getenv("RAG_LLM_BASE_URL", "https://api.deepseek.com/v1")
LLM_API_KEY = os.getenv("RAG_LLM_API_KEY", "")
LLM_MODEL = os.getenv("RAG_LLM_MODEL", "deepseek-v4-flash")
USE_REAL_LLM = bool(LLM_API_KEY)


DOCS: list[dict[str, Any]] = [
    {
        "id": 1,
        "content": "退货政策：用户可在签收后 7 天内申请退货，需保持商品完好。",
        "source": "after_sales.md",
        "category": "售后",
        "parent_title": "退货政策",
        "parent_content": "退货政策：用户可在签收后 7 天内申请退货，需保持商品完好。",
        "chunk_index": 0,
        "tenant_id": TENANT,
        "kb_id": KB,
    },
    {
        "id": 2,
        "content": "售后流程：退款申请提交后，平台在 1-3 个工作日内完成审核。",
        "source": "after_sales.md",
        "category": "售后",
        "parent_title": "售后流程",
        "parent_content": "售后流程：退款申请提交后，平台在 1-3 个工作日内完成审核。",
        "chunk_index": 1,
        "tenant_id": TENANT,
        "kb_id": KB,
    },
    {
        "id": 3,
        "content": "食品安全：请勿食用野生不认识的蘑菇，以免中毒。",
        "source": "safety.md",
        "category": "规则",
        "parent_title": "食品安全",
        "parent_content": "食品安全：请勿食用野生不认识的蘑菇，以免中毒。",
        "chunk_index": 0,
        "tenant_id": TENANT,
        "kb_id": KB,
    },
]


class RewriteStubLLM:
    """改写专用桩：每次调用返回预设的单条改写文本。"""

    def __init__(self, response: str = "") -> None:
        self.response = response

    async def chat(
        self,
        messages: list[dict],
        *,
        system_prompt: str = "",
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        return self.response


def _make_runtime() -> RuntimeConfig:
    return RuntimeConfig(
        llm=LLMConfig(
            base_url=LLM_BASE_URL,
            api_key=LLM_API_KEY,
            model=LLM_MODEL,
            embedding_model=EMBED_MODEL,
        ),
        vector=VectorStoreConfig(
            host=MILVUS_HOST,
            port=MILVUS_PORT,
            text_collection=COLLECTION,
        ),
        cache=CacheConfig(host=REDIS_HOST, port=REDIS_PORT),
    )


def _create_collection() -> None:
    from pymilvus import (
        Collection,
        CollectionSchema,
        DataType,
        FieldSchema,
        Function,
        FunctionType,
        connections,
        utility,
    )

    connections.connect(alias="default", host=MILVUS_HOST, port=str(MILVUS_PORT))
    if utility.has_collection(COLLECTION):
        utility.drop_collection(COLLECTION)

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(
            name="content",
            dtype=DataType.VARCHAR,
            max_length=65535,
            enable_analyzer=True,
            analyzer_params={"type": "chinese"},
        ),
        FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=128),
        FieldSchema(name="parent_title", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="parent_content", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="chunk_index", dtype=DataType.INT64),
        FieldSchema(name="tenant_id", dtype=DataType.VARCHAR, max_length=128),
        FieldSchema(name="kb_id", dtype=DataType.VARCHAR, max_length=128),
        FieldSchema(
            name="embedding",
            dtype=DataType.FLOAT_VECTOR,
            dim=DIM,
        ),
        FieldSchema(name="sparse", dtype=DataType.SPARSE_FLOAT_VECTOR),
    ]
    schema = CollectionSchema(fields=fields)
    schema.add_function(
        Function(
            name="bm25",
            input_field_names=["content"],
            output_field_names=["sparse"],
            function_type=FunctionType.BM25,
        )
    )
    collection = Collection(name=COLLECTION, schema=schema)
    collection.create_index(
        "embedding",
        {
            "index_type": "HNSW",
            "metric_type": "COSINE",
            "params": {"M": 16, "efConstruction": 128},
        },
    )
    collection.create_index(
        "sparse",
        {
            "index_type": "SPARSE_INVERTED_INDEX",
            "metric_type": "BM25",
            "params": {"drop_ratio_build": 0.2},
        },
    )
    collection.load()
    connections.disconnect("default")


def _embed_docs() -> list[list[float]]:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBED_MODEL, local_files_only=True)
    return model.encode(
        [doc["content"] for doc in DOCS],
        normalize_embeddings=True,
    ).tolist()


def _insert_docs(vectors: list[list[float]]) -> None:
    from pymilvus import Collection, connections

    connections.connect(alias="default", host=MILVUS_HOST, port=str(MILVUS_PORT))
    collection = Collection(name=COLLECTION)
    rows: list[dict[str, Any]] = []
    for doc, vector in zip(DOCS, vectors):
        row = dict(doc)
        row["embedding"] = vector
        rows.append(row)
    collection.insert(rows)
    collection.flush()
    connections.disconnect("default")


def _ctx() -> AgentContext:
    return AgentContext(
        tenant_id=TENANT,
        kb_id=KB,
        session_id="smoke",
        request_id="smoke-1",
        user_id="smoke",
    )


async def _retrieve(
    pipeline: RagPipeline,
    *,
    query_rewrite_mode: str,
) -> tuple[list[str], dict[str, Any]]:
    """跑一次真实检索，返回命中文档 id 与改写明细。"""
    trace: dict[str, Any] = {}
    docs = await pipeline.retrieve(
        "怎么申请退货退款？",
        _ctx(),
        query_rewrite_mode=query_rewrite_mode,
        rewrite_trace=trace,
    )
    return [str(doc.get("id")) for doc in docs], trace


async def main() -> None:
    print("[setup] create collection + insert docs ...")
    _create_collection()
    _insert_docs(_embed_docs())

    print("[setup] build pipeline (real Milvus + real embedding + Redis) ...")
    runtime = _make_runtime()
    if USE_REAL_LLM:
        rewrite_llm: Any = OpenAICompatibleLLM(runtime.llm)
    else:
        rewrite_llm = RewriteStubLLM()
    pipeline = RagPipeline(runtime, llm=rewrite_llm)

    results: dict[str, Any] = {}

    if not USE_REAL_LLM:
        rewrite_llm.response = "退货退款申请流程是什么"
    off_ids, off_trace = await _retrieve(
        pipeline, query_rewrite_mode="off"
    )
    results["off"] = {"ids": off_ids, "rewritten_query": off_trace["rewritten_query"]}

    identity_ids, identity_trace = await _retrieve(
        pipeline, query_rewrite_mode="identity"
    )
    results["identity"] = {
        "ids": identity_ids,
        "rewritten_query": identity_trace["rewritten_query"],
    }

    if not USE_REAL_LLM:
        rewrite_llm.response = "七天无理由退货退款申请"
    llm_ids, llm_trace = await _retrieve(
        pipeline, query_rewrite_mode="llm_rewrite"
    )
    results["llm_rewrite"] = {
        "ids": llm_ids,
        "rewritten_query": llm_trace["rewritten_query"],
        "variants": llm_trace["query_variants"],
    }

    if not USE_REAL_LLM:
        rewrite_llm.response = "七天无理由退货\n退货退款审核流程\n售后退款政策"
    exp_ids, exp_trace = await _retrieve(
        pipeline, query_rewrite_mode="query_expansion"
    )
    results["query_expansion"] = {
        "ids": exp_ids,
        "rewritten_query": exp_trace["rewritten_query"],
        "variants": exp_trace["query_variants"],
    }

    print("\n===== 冒烟结果 =====")
    for mode, data in results.items():
        print(f"[{mode}] rewritten_query={data['rewritten_query']!r}")
        extras = {k: v for k, v in data.items() if k not in ("ids", "rewritten_query")}
        if extras:
            print(f"       {extras}")
        print(f"       hit_ids={data['ids']}")

    # 断言（真实链路回归）
    assert results["off"]["ids"], "off 模式应能命中"
    assert (
        results["off"]["ids"] == results["identity"]["ids"]
    ), "identity 与 off 应保持原查询，召回一致（无回归）"
    if USE_REAL_LLM:
        assert results["llm_rewrite"]["rewritten_query"], "llm_rewrite 应有改写结果"
        assert len(results["llm_rewrite"]["variants"]) >= 1
        assert len(results["query_expansion"]["variants"]) >= 2
    else:
        assert results["llm_rewrite"]["rewritten_query"] == "七天无理由退货退款申请"
        assert results["query_expansion"]["rewritten_query"] == "七天无理由退货"
    assert results["llm_rewrite"]["ids"], "llm_rewrite 应能命中"
    assert len(results["query_expansion"]["ids"]) == len(
        set(results["query_expansion"]["ids"])
    ), "query_expansion 多变体合并后不应有重复 id"
    print(
        "\nSMOKE PASS: off/identity 无回归，改写与扩展均真实进入 Milvus 检索。"
        + ("（使用真实 DeepSeek 模型改写）" if USE_REAL_LLM else "（使用可控桩）")
    )


if __name__ == "__main__":
    asyncio.run(main())
