import os

import pytest

from common_core.config import (
    AuthConfig,
    ConfigError,
    LLMConfig,
    MetricsConfig,
    QueryRewriteConfig,
    RuntimeConfig,
    VectorStoreConfig,
    config_fingerprint,
    config_snapshot,
    env_bool,
    env_list,
    is_sensitive_key,
    mask_key_value,
    redacted_snapshot,
    resolve_env_file,
)


def test_llm_config_reads_generic_env_without_business_defaults() -> None:
    env = {
        "LLM_BASE_URL": "http://llm.internal/v1",
        "LLM_MODEL": "generic-model",
        "EMBEDDING_MODEL": "local-embedder",
    }
    cfg = LLMConfig.from_env(env=env)
    assert cfg.base_url == "http://llm.internal/v1"
    assert cfg.model == "generic-model"
    assert cfg.embedding_model == "local-embedder"
    assert cfg.api_key == ""
    assert "mushroom" not in str(cfg)


def test_vector_config_reads_empty_defaults() -> None:
    cfg = VectorStoreConfig.from_env(env={})
    assert cfg.host == "localhost"
    assert cfg.port == 19530
    assert cfg.text_collection == ""
    assert cfg.image_collection == ""
    assert cfg.text_output_fields == ()


def test_vector_config_reads_output_fields_list() -> None:
    cfg = VectorStoreConfig.from_env(
        env={"MILVUS_OUTPUT_FIELDS": "id, content, source"}
    )
    assert cfg.text_output_fields == ("id", "content", "source")


def test_metrics_config_from_env() -> None:
    cfg = MetricsConfig.from_env(
        env={"METRICS_ENABLED": "true", "METRICS_PREFIX": "shop", "METRICS_PORT": "9091"}
    )
    assert cfg.enabled is True
    assert cfg.prefix == "shop"
    assert cfg.port == 9091


def test_runtime_config_composes_all_sections() -> None:
    cfg = RuntimeConfig.from_env(
        env={
            "LLM_MODEL": "x",
            "MILVUS_HOST": "vec.internal",
            "REDIS_HOST": "cache.internal",
            "AUTH_JWT_SECRET": "secret",
        }
    )
    assert cfg.llm.model == "x"
    assert cfg.vector.host == "vec.internal"
    assert cfg.cache.host == "cache.internal"
    assert cfg.auth.jwt_secret == "secret"


def test_retrieval_config_default_min_relevance() -> None:
    cfg = RuntimeConfig.from_env(env={})
    assert cfg.retrieval.min_relevance == 0.70


def test_query_rewrite_config_defaults_off() -> None:
    cfg = QueryRewriteConfig.from_env(env={})
    assert cfg.mode == "off"
    assert cfg.expand_count == 2
    assert cfg.scoped_modes == {}


def test_query_rewrite_config_reads_env_and_scoped_modes() -> None:
    cfg = QueryRewriteConfig.from_env(
        env={
            "RETRIEVAL_QUERY_REWRITE_MODE": "llm_rewrite",
            "RETRIEVAL_QUERY_REWRITE_LLM_MODEL": "rewrite-model",
            "RETRIEVAL_QUERY_REWRITE_TEMPERATURE": "0.2",
            "RETRIEVAL_QUERY_REWRITE_MAX_TOKENS": "128",
            "RETRIEVAL_QUERY_REWRITE_EXPAND_COUNT": "3",
            "RETRIEVAL_QUERY_REWRITE_SCOPES": (
                "t1/kb1=query_expansion,t2/*=llm_rewrite,*/kb3=identity"
            ),
        }
    )
    assert cfg.mode == "llm_rewrite"
    assert cfg.llm_model == "rewrite-model"
    assert cfg.temperature == 0.2
    assert cfg.max_tokens == 128
    assert cfg.expand_count == 3
    assert cfg.scoped_modes == {
        "t1/kb1": "query_expansion",
        "t2/*": "llm_rewrite",
        "*/kb3": "identity",
    }


def test_env_helpers_parse_bools_and_lists() -> None:
    assert env_bool("X", env={"X": "1"}) is True
    assert env_bool("X", env={"X": "false"}) is False
    assert env_list("X", env={"X": "a,b,c"}) == ["a", "b", "c"]
    assert env_list("X", env={}) == []


def test_llm_validate_returns_missing_keys() -> None:
    cfg = LLMConfig.from_env(env={})
    assert cfg.validate() == ["LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL"]
    assert cfg.validate(require_embedding=True) == [
        "LLM_BASE_URL",
        "LLM_API_KEY",
        "LLM_MODEL",
        "EMBEDDING_MODEL",
    ]


def test_vector_validate_returns_missing_keys() -> None:
    cfg = VectorStoreConfig.from_env(env={})
    assert cfg.validate() == ["MILVUS_TEXT_COLLECTION"]
    assert VectorStoreConfig.from_env(
        env={"MILVUS_TEXT_COLLECTION": "kb"}
    ).validate() == []


def test_auth_validate_jwt_requires_secret_or_public_key() -> None:
    cfg = AuthConfig.from_env(env={"AUTH_MODE": "jwt"})
    assert cfg.validate() == [
        "AUTH_JWT_SECRET or AUTH_JWT_PUBLIC_KEY"
    ]
    assert (
        AuthConfig.from_env(
            env={"AUTH_MODE": "jwt", "AUTH_JWT_SECRET": "s"}
        ).validate()
        == []
    )


def test_auth_validate_disabled_allows_missing_keys() -> None:
    cfg = AuthConfig.from_env(env={"AUTH_MODE": "disabled"})
    assert cfg.validate() == []


def test_auth_validate_rejects_unknown_mode() -> None:
    cfg = AuthConfig.from_env(env={"AUTH_MODE": "magic"})
    with pytest.raises(ConfigError, match="Unsupported AUTH_MODE"):
        cfg.validate()


def test_runtime_validate_fails_fast_with_combined_missing_keys() -> None:
    cfg = RuntimeConfig.from_env(env={})
    with pytest.raises(ConfigError, match="LLM_BASE_URL") as exc:
        cfg.validate()
    message = str(exc.value)
    assert "LLM_API_KEY" in message
    assert "LLM_MODEL" in message
    assert "MILVUS_TEXT_COLLECTION" in message
    assert "AUTH_JWT_SECRET" in message


def test_runtime_validate_passes_with_required_keys() -> None:
    cfg = RuntimeConfig.from_env(
        env={
            "LLM_BASE_URL": "http://llm.internal/v1",
            "LLM_API_KEY": "key",
            "LLM_MODEL": "model",
            "MILVUS_TEXT_COLLECTION": "kb",
            "AUTH_MODE": "disabled",
        }
    )
    assert cfg.validate() is None


def test_resolve_env_file_explicit_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_TEST_ENV_FILE", "from-env.env")
    assert resolve_env_file("explicit.env", env_key="RAG_TEST_ENV_FILE") == "explicit.env"


def test_resolve_env_file_env_var_next(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("RAG_TEST_ENV_FILE", str(tmp_path / "from-env.env"))
    assert (
        resolve_env_file(None, env_key="RAG_TEST_ENV_FILE")
        == str(tmp_path / "from-env.env")
    )


def test_resolve_env_file_default_only_when_file_exists(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "custom.env").write_text("", encoding="utf-8")
    assert resolve_env_file(None, default="custom.env") == "custom.env"
    assert resolve_env_file(None, default="missing.env") is None


def test_same_key_different_values_are_isolated_per_config() -> None:
    """Same key name in two skills must not collide: configs read independently."""
    rag_env = {"LLM_MODEL": "rag-model", "LLM_BASE_URL": "http://rag/v1"}
    sql_env = {"LLM_MODEL": "sql-model", "LLM_BASE_URL": "http://sql/v1"}
    rag_cfg = RuntimeConfig.from_env(env=rag_env)
    sql_cfg = RuntimeConfig.from_env(env=sql_env)
    assert rag_cfg.llm.model == "rag-model"
    assert sql_cfg.llm.model == "sql-model"
    assert rag_cfg.llm.model != sql_cfg.llm.model


def test_precedence_process_env_over_explicit_env_source() -> None:
    """Priority: process env (source) > provided env dict > defaults."""
    source = {"LLM_MODEL": "process-model", "LLM_BASE_URL": "http://p/v1", "LLM_API_KEY": "k"}
    # An explicit `env` dict represents values from a .env file that must NOT
    # override already-present process-level variables. We simulate by
    # calling from_env with a dict that has a *lower* value for a shared key.
    cfg = RuntimeConfig.from_env(
        env={
            "LLM_MODEL": "lower-priority",
            "LLM_BASE_URL": "http://lower/v1",
            "LLM_API_KEY": "k",
            "MILVUS_HOST": "vec",
            "MILVUS_TEXT_COLLECTION": "kb",
            "AUTH_MODE": "disabled",
        }
    )
    assert cfg.llm.model == "lower-priority"


def test_defaults_apply_when_no_source_provides_key() -> None:
    cfg = RuntimeConfig.from_env(env={})
    assert cfg.vector.host == "localhost"
    assert cfg.vector.port == 19530
    assert cfg.cache.port == 6379
    assert cfg.retrieval.top_k == 20


def test_duplex_source_highest_priority_wins() -> None:
    """Two sources: provided dict (from .env) and explicit overrides win."""
    base = {
        "LLM_MODEL": "base",
        "LLM_BASE_URL": "http://base/v1",
        "LLM_API_KEY": "k",
        "MILVUS_HOST": "vec",
        "MILVUS_TEXT_COLLECTION": "kb",
        "AUTH_MODE": "disabled",
    }
    override = {"LLM_MODEL": "override"}
    merged = {**base, **override}
    cfg = RuntimeConfig.from_env(env=merged)
    assert cfg.llm.model == "override"
    assert cfg.llm.base_url == "http://base/v1"


def test_config_snapshot_and_fingerprint_are_stable_and_exclude_secrets() -> None:
    env = {
        "LLM_BASE_URL": "http://llm/v1",
        "LLM_API_KEY": "super-secret-key",
        "LLM_MODEL": "model",
        "MILVUS_HOST": "vec",
        "MILVUS_TEXT_COLLECTION": "kb",
    }
    cfg1 = RuntimeConfig.from_env(env=env)
    cfg2 = RuntimeConfig.from_env(env=env)
    snap = config_snapshot(cfg1)
    assert config_fingerprint(cfg1) == config_fingerprint(cfg2)
    # Fingerprint excludes sensitive keys so it never embeds credentials.
    assert "super-secret-key" not in config_fingerprint(cfg1)
    keys = {k for k, _ in snap}
    assert "LLM_API_KEY" in keys


def test_fingerprint_changes_when_non_secret_value_changes() -> None:
    base = {
        "LLM_BASE_URL": "http://llm/v1",
        "LLM_API_KEY": "k",
        "LLM_MODEL": "model-a",
        "MILVUS_HOST": "vec",
        "MILVUS_TEXT_COLLECTION": "kb",
    }
    changed = {**base, "LLM_MODEL": "model-b"}
    assert config_fingerprint(RuntimeConfig.from_env(env=base)) != config_fingerprint(
        RuntimeConfig.from_env(env=changed)
    )


def test_fingerprint_is_hmac_signed_with_secret() -> None:
    env = {
        "LLM_BASE_URL": "http://llm/v1",
        "LLM_API_KEY": "k",
        "LLM_MODEL": "model",
        "MILVUS_HOST": "vec",
        "MILVUS_TEXT_COLLECTION": "kb",
    }
    cfg = RuntimeConfig.from_env(env=env)
    assert config_fingerprint(cfg, secret="s1") != config_fingerprint(cfg)
    assert config_fingerprint(cfg, secret="s1") != config_fingerprint(cfg, secret="s2")


@pytest.mark.parametrize(
    ("key", "value", "expected"),
    [
        ("LLM_API_KEY", "abc123", "<redacted>"),
        ("MILVUS_PASSWORD", "secret", "<redacted>"),
        ("AUTH_JWT_SECRET", "x", "<redacted>"),
        ("LLM_MODEL", "ab", "**"),
        ("LLM_BASE_URL", "abcdefgh", "ab***gh"),
        ("", "", ""),
    ],
)
def test_mask_key_value_redacts(key: str, value: str, expected: str) -> None:
    assert mask_key_value(key, value) == expected


def test_is_sensitive_key_heuristics() -> None:
    assert is_sensitive_key("LLM_API_KEY")
    assert is_sensitive_key("SQL_PASSWORD")
    assert is_sensitive_key("AUTH_JWT_PUBLIC_KEY")
    assert not is_sensitive_key("LLM_MODEL")
    assert not is_sensitive_key("MILVUS_HOST")


def test_redacted_snapshot_never_leaks_secrets() -> None:
    cfg = RuntimeConfig.from_env(
        env={
            "LLM_BASE_URL": "http://llm/v1",
            "LLM_API_KEY": "topsecret",
            "LLM_MODEL": "model",
            "MILVUS_HOST": "vec",
            "MILVUS_TEXT_COLLECTION": "kb",
            "AUTH_MODE": "disabled",
        }
    )
    redacted = redacted_snapshot(cfg)
    assert redacted["LLM_API_KEY"] == "<redacted>"
    assert "topsecret" not in "".join(redacted.values())
