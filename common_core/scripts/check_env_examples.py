"""CI guard: keep every skill ``.env.example`` aligned with the config contract.

The contract is defined here (Key name, required for which skill, sensitive,
expected type). It is the single source the examples and ``config.md`` must
agree with. Running this in CI fails the build when an example:

- omits a required shared key,
- declares an unknown key (typo / drift from the contract), or
- ships a real-looking secret instead of a placeholder.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Canonical contract. Keys used by skills beyond the common shared set are
# listed as optional so they are still spell-checked.
CONTRACT: dict[str, dict[str, str]] = {
    # (required-for, sensitive, type)
    "LLM_BASE_URL": {"required": {"common_core", "rag_skill"}, "sensitive": False, "type": "str"},
    "LLM_API_KEY": {"required": {"common_core", "rag_skill"}, "sensitive": True, "type": "str"},
    "LLM_MODEL": {"required": {"common_core", "rag_skill"}, "sensitive": False, "type": "str"},
    "LLM_TEMPERATURE": {"required": set(), "sensitive": False, "type": "float"},
    "LLM_MAX_TOKENS": {"required": set(), "sensitive": False, "type": "int"},
    "LLM_TIMEOUT_SECONDS": {"required": set(), "sensitive": False, "type": "float"},
    "EMBEDDING_MODEL": {"required": {"rag_skill"}, "sensitive": False, "type": "str"},
    "MILVUS_HOST": {"required": {"rag_skill"}, "sensitive": False, "type": "str"},
    "MILVUS_PORT": {"required": set(), "sensitive": False, "type": "int"},
    "MILVUS_USER": {"required": set(), "sensitive": False, "type": "str"},
    "MILVUS_PASSWORD": {"required": set(), "sensitive": True, "type": "str"},
    "MILVUS_SECURE": {"required": set(), "sensitive": False, "type": "bool"},
    "MILVUS_DB": {"required": set(), "sensitive": False, "type": "str"},
    "MILVUS_TEXT_COLLECTION": {"required": {"rag_skill"}, "sensitive": False, "type": "str"},
    "MILVUS_IMAGE_COLLECTION": {"required": set(), "sensitive": False, "type": "str"},
    "MILVUS_DIM": {"required": set(), "sensitive": False, "type": "int"},
    "MILVUS_OUTPUT_FIELDS": {"required": set(), "sensitive": False, "type": "csv"},
    "REDIS_HOST": {"required": {"rag_skill"}, "sensitive": False, "type": "str"},
    "REDIS_PORT": {"required": set(), "sensitive": False, "type": "int"},
    "REDIS_PASSWORD": {"required": set(), "sensitive": True, "type": "str"},
    "REDIS_DB": {"required": set(), "sensitive": False, "type": "int"},
    "REDIS_DEFAULT_TTL": {"required": set(), "sensitive": False, "type": "int"},
    "REDIS_KEY_PREFIX": {"required": set(), "sensitive": False, "type": "str"},
    "REDIS_SOCKET_TIMEOUT": {"required": set(), "sensitive": False, "type": "float"},
    "AUTH_MODE": {"required": {"common_core", "rag_skill"}, "sensitive": False, "type": "enum"},
    "AUTH_JWT_SECRET": {"required": set(), "sensitive": True, "type": "str"},
    "AUTH_JWT_PUBLIC_KEY": {"required": set(), "sensitive": True, "type": "str"},
    "AUTH_JWT_ALGORITHMS": {"required": set(), "sensitive": False, "type": "csv"},
    "AUTH_JWT_ISSUER": {"required": set(), "sensitive": False, "type": "str"},
    "AUTH_JWT_AUDIENCE": {"required": set(), "sensitive": False, "type": "str"},
    "AUTH_MCP_ISSUER_URL": {"required": set(), "sensitive": False, "type": "str"},
    "AUTH_MCP_RESOURCE_SERVER_URL": {"required": set(), "sensitive": False, "type": "str"},
    "METRICS_ENABLED": {"required": set(), "sensitive": False, "type": "bool"},
    "METRICS_PREFIX": {"required": set(), "sensitive": False, "type": "str"},
    "METRICS_PORT": {"required": set(), "sensitive": False, "type": "int"},
    "METRICS_BIND": {"required": set(), "sensitive": False, "type": "str"},
    "RETRIEVAL_TOP_K": {"required": set(), "sensitive": False, "type": "int"},
    "RETRIEVAL_MIN_RELEVANCE": {"required": set(), "sensitive": False, "type": "float"},
    "RETRIEVAL_HYBRID_WORKERS": {"required": set(), "sensitive": False, "type": "int"},
    "RETRIEVAL_ASSEMBLY_MAX_CHARS": {"required": set(), "sensitive": False, "type": "int"},
    "RRF_TOP_K": {"required": set(), "sensitive": False, "type": "int"},
    "RRF_K": {"required": set(), "sensitive": False, "type": "int"},
    "RERANK_TOP_K": {"required": set(), "sensitive": False, "type": "int"},
    "RETRIEVAL_QUERY_REWRITE_MODE": {"required": set(), "sensitive": False, "type": "enum"},
    "RETRIEVAL_QUERY_REWRITE_LLM_MODEL": {"required": set(), "sensitive": False, "type": "str"},
    "RETRIEVAL_QUERY_REWRITE_TEMPERATURE": {"required": set(), "sensitive": False, "type": "float"},
    "RETRIEVAL_QUERY_REWRITE_MAX_TOKENS": {"required": set(), "sensitive": False, "type": "int"},
    "RETRIEVAL_QUERY_REWRITE_EXPAND_COUNT": {"required": set(), "sensitive": False, "type": "int"},
    "RETRIEVAL_QUERY_REWRITE_PROMPT": {"required": set(), "sensitive": False, "type": "str"},
    "RETRIEVAL_QUERY_REWRITE_EXPANSION_PROMPT": {"required": set(), "sensitive": False, "type": "str"},
    "RETRIEVAL_QUERY_REWRITE_SCOPES": {"required": set(), "sensitive": False, "type": "str"},
    # skill-specific keys
    "RERANKER_MODEL": {"required": {"rag_skill"}, "sensitive": False, "type": "str"},
    "RERANKER_DEVICE": {"required": set(), "sensitive": False, "type": "str"},
    "RERANKER_CE_WEIGHT": {"required": set(), "sensitive": False, "type": "float"},
    "RERANKER_RETRIEVAL_WEIGHT": {"required": set(), "sensitive": False, "type": "float"},
    "SQL_HOST": {"required": {"structured_query_skill"}, "sensitive": False, "type": "str"},
    "SQL_PORT": {"required": set(), "sensitive": False, "type": "int"},
    "SQL_USER": {"required": {"structured_query_skill"}, "sensitive": False, "type": "str"},
    "SQL_PASSWORD": {"required": {"structured_query_skill"}, "sensitive": True, "type": "str"},
    "SQL_DATABASE": {"required": {"structured_query_skill"}, "sensitive": False, "type": "str"},
    "SQL_CHARSET": {"required": set(), "sensitive": False, "type": "str"},
    "SQL_CONNECT_TIMEOUT": {"required": set(), "sensitive": False, "type": "float"},
    "SQL_READ_TIMEOUT": {"required": set(), "sensitive": False, "type": "float"},
    "SQL_POOL_MAX_CONNECTIONS": {"required": set(), "sensitive": False, "type": "int"},
    "SQL_POOL_MIN_CACHED": {"required": set(), "sensitive": False, "type": "int"},
    "SQL_POOL_MAX_CACHED": {"required": set(), "sensitive": False, "type": "int"},
    "SQL_POOL_BLOCKING": {"required": set(), "sensitive": False, "type": "bool"},
    "SQL_DEFAULT_MAX_ROWS": {"required": set(), "sensitive": False, "type": "int"},
}

PLACEHOLDER_MARKERS = ("replace-me", "changeme", "your-", "<", ">")


def parse_env(path: Path) -> set[str]:
    keys: set[str] = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key = line.split("=", 1)[0].strip()
        if key:
            keys.add(key)
    return keys


def check_example(path: Path, skill: str, errors: list[str]) -> None:
    keys = parse_env(path)
    for required, meta in CONTRACT.items():
        if skill in meta["required"] and required not in keys:
            errors.append(f"{path.name}[{skill}]: missing required key {required}")
    for key in sorted(keys):
        if key not in CONTRACT:
            errors.append(f"{path.name}: unknown key {key} not in contract")
            continue
        if CONTRACT[key]["sensitive"]:
            value = ""
            for raw in path.read_text(encoding="utf-8").splitlines():
                if raw.startswith(key + "="):
                    value = raw.split("=", 1)[1].strip()
                    break
            if value and not any(m in value.lower() for m in PLACEHOLDER_MARKERS):
                errors.append(f"{path.name}: sensitive key {key} holds a real-looking value in an example")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    errors: list[str] = []
    examples = {
        "common_core": args.root / "common_core" / ".env.example",
        "rag_skill": args.root / "rag_skill" / ".env.example",
        "structured_query_skill": args.root / "structured_query_skill" / ".env.example",
    }
    for skill, path in examples.items():
        if not path.is_file():
            errors.append(f"missing example for {skill}: {path}")
            continue
        check_example(path, skill, errors)
    if errors:
        print("\n".join(f"[fail] {e}" for e in errors))
        return 1
    print("env examples aligned with config contract")
    return 0


if __name__ == "__main__":
    sys.exit(main())
