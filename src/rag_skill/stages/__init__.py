"""可复用 RAG 管线阶段（stages）。

每个阶段都不含业务逻辑：提示词、词表、模板与渲染规则由实例
（skill instances）注入。本层只实现通用机制：
重排（rerank）、上下文组装（assembly）、生成（generation）、
护栏（guard）与响应缓存（cache）。
"""

from .assembly import (
    build_context_text,
    clean_markdown,
    dedupe_docs,
    extract_images,
)
from .cache import ResponseCache
from .generation import GenerationConfig, generate_answer, stream_answer
from .guard import (
    DEFAULT_ABSOLUTE_WORDS,
    GuardConfig,
    GuardResult,
    absolute_language_issues,
    check_compound_numbers,
    evaluate_guard,
    extract_all_numbers,
    extract_risky_numbers,
    guard_generation,
)
from .query_rewrite import (
    QueryRewriteConfig,
    QueryRewriteResult,
    QueryRewriter,
)
from .rerank import Reranker, judge_relevance, rank_docs

__all__ = [
    "DEFAULT_ABSOLUTE_WORDS",
    "GenerationConfig",
    "GuardConfig",
    "GuardResult",
    "QueryRewriteConfig",
    "QueryRewriteResult",
    "QueryRewriter",
    "Reranker",
    "ResponseCache",
    "absolute_language_issues",
    "build_context_text",
    "check_compound_numbers",
    "clean_markdown",
    "dedupe_docs",
    "evaluate_guard",
    "extract_all_numbers",
    "extract_images",
    "extract_risky_numbers",
    "generate_answer",
    "guard_generation",
    "judge_relevance",
    "rank_docs",
    "stream_answer",
]
