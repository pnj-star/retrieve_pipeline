"""基于 common_core provider 的可复用 RAG 管线技能包。

对外统一导出：组装函数（build_runtime / build_pipeline）、管线（RagPipeline）、
返回契约（RagResult / RagStatus）、MCP 入口（create_mcp_server / main），
以及各阶段函数（重排、上下文组装、生成、护栏、人工交接、缓存）。
"""

from .builder import build_pipeline, build_runtime
from .mcp import create_mcp_server, main
from .pipeline import RagPipeline
from .results import RagResult, RagStatus
from .stages import (
    ChainHandoffStore,
    DEFAULT_ABSOLUTE_WORDS,
    GenerationConfig,
    GuardConfig,
    GuardResult,
    HandoffRecord,
    RedisHandoffStore,
    Reranker,
    ResponseCache,
    absolute_language_issues,
    build_context_text,
    build_handoff_record,
    check_compound_numbers,
    clean_markdown,
    dedupe_docs,
    default_template_selector,
    evaluate_guard,
    extract_all_numbers,
    extract_images,
    extract_risky_numbers,
    generate_answer,
    guard_generation,
    judge_relevance,
    persist_handoff,
    rank_docs,
    stream_answer,
)
from .tokenization import build_token_counter, make_token_counter

__all__ = [
    "ChainHandoffStore",
    "DEFAULT_ABSOLUTE_WORDS",
    "GenerationConfig",
    "GuardConfig",
    "GuardResult",
    "HandoffRecord",
    "RagPipeline",
    "RagResult",
    "RagStatus",
    "RedisHandoffStore",
    "Reranker",
    "ResponseCache",
    "absolute_language_issues",
    "build_context_text",
    "build_handoff_record",
    "build_pipeline",
    "build_runtime",
    "build_token_counter",
    "check_compound_numbers",
    "clean_markdown",
    "create_mcp_server",
    "dedupe_docs",
    "default_template_selector",
    "evaluate_guard",
    "extract_all_numbers",
    "extract_images",
    "extract_risky_numbers",
    "generate_answer",
    "guard_generation",
    "judge_relevance",
    "main",
    "make_token_counter",
    "persist_handoff",
    "rank_docs",
    "stream_answer",
]

__version__ = "0.1.0"
