r"""rag_skill 真实链路端到端冒烟（answer 全流程 + 响应缓存命中）。

在真实 Milvus（稠密 + BM25 混合检索）与真实 embedding 下跑通
``answer()`` 的 retrieve -> 拼上下文 -> LLM 生成，并验证 Redis 响应缓存：

    1) 第一次 answer -> status=answered（真实检索 + 生成回答）
    2) 第二次同 query + tenant/kb -> status=answered_cache（命中缓存，不再检索 / 生成）

生成 / 改写模型默认用可控桩；设置 RAG_LLM_BASE_URL / RAG_LLM_API_KEY / RAG_LLM_MODEL
时改用真实 OpenAI 兼容模型。

运行（请先启动 Docker 内的 Milvus 与 Redis）：
    $env:HF_HUB_OFFLINE='1'
    & D:\python\python.exe rag_skill\scripts\smoke_answer_e2e.py
"""

from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "..", "..", "common_core", "src"),
)

from common_core.context import AgentContext
from common_core.providers import OpenAICompatibleLLM, RedisCache
from rag_skill.pipeline import RagPipeline
from rag_skill.results import RagStatus
from rag_skill.stages import ResponseCache

import smoke_query_rewrite as sq

ANSWER_COLLECTION = "smoke_answer_kb"


class AnswerStubLLM:
    """生成专用桩：固定返回一段回答，并记录调用次数。"""

    def __init__(self, response: str = "") -> None:
        self.response = response or "根据售后政策，用户可在签收后 7 天内申请退货退款。"
        self.chat_calls = 0

    async def chat(
        self,
        messages: list[dict],
        *,
        system_prompt: str = "",
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        self.chat_calls += 1
        return self.response


def _ctx() -> AgentContext:
    return AgentContext(
        tenant_id=sq.TENANT,
        kb_id=sq.KB,
        session_id="smoke-answer",
        request_id="smoke-answer-1",
        user_id="smoke",
    )


async def main() -> None:
    sq.COLLECTION = ANSWER_COLLECTION
    print(f"[setup] create collection '{ANSWER_COLLECTION}' + insert docs ...")
    sq._create_collection()
    sq._insert_docs(sq._embed_docs())

    print("[setup] build pipeline (real Milvus + real embedding + Redis cache) ...")
    runtime = sq._make_runtime()
    gen_llm = OpenAICompatibleLLM(runtime.llm) if sq.USE_REAL_LLM else AnswerStubLLM()
    cache = RedisCache(runtime.cache)
    pipeline = RagPipeline(
        runtime,
        llm=gen_llm,
        cache=cache,
        response_cache=ResponseCache(cache, min_cache_chars=0),
    )
    ctx = _ctx()

    print("\n[call 1] first answer (should hit real Milvus + generate) ...")
    first = await pipeline.answer("怎么申请退货退款？", ctx)
    print(f"          status={first.status} answer={first.answer!r}")
    assert first.status == RagStatus.ANSWERED, first.status
    assert first.answer, "回答不应为空"
    if isinstance(gen_llm, AnswerStubLLM):
        assert gen_llm.chat_calls == 1, "首次应生成一次"

    print("[call 2] same query again (should hit Redis cache) ...")
    second = await pipeline.answer("怎么申请退货退款？", ctx)
    print(f"          status={second.status} answer={second.answer!r}")
    assert second.status == RagStatus.ANSWERED_CACHE, second.status
    assert second.answer == first.answer, "缓存回答应与首次一致"

    if isinstance(gen_llm, AnswerStubLLM):
        assert gen_llm.chat_calls == 1, "第二次命中缓存，不应再触发生成"
    else:
        print("          (使用真实 LLM，跳过生成计数断言)")

    print(
        "\nSMOKE PASS: answer 全链路（检索->生成）与 Redis 响应缓存命中均验证通过。"
        + ("（使用真实 LLM）" if sq.USE_REAL_LLM else "（使用可控桩）")
    )


if __name__ == "__main__":
    asyncio.run(main())
