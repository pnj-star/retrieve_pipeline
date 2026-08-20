from prometheus_client import CollectorRegistry

from common_core.config import MetricsConfig
from common_core.observability import Observability


def _regulated(prefix: str = "shop", enabled: bool = True) -> tuple[Observability, CollectorRegistry]:
    registry = CollectorRegistry(auto_describe=True)
    return (
        Observability(
            MetricsConfig(enabled=enabled, prefix=prefix),
            registry=registry,
        ),
        registry,
    )


def test_metrics_use_configured_prefix_and_tenant_labels() -> None:
    obs, registry = _regulated()
    obs.record_run("chat", tenant_id="t1", kb_id="kb1")
    obs.record_tokens("prompt", "model-x", 10, tenant_id="t1", kb_id="kb1")

    assert (
        registry.get_sample_value(
            "shop_runs_total",
            {"route": "chat", "tenant_id": "t1", "kb_id": "kb1"},
        )
        == 1.0
    )
    assert (
        registry.get_sample_value(
            "shop_llm_tokens_total",
            {"type": "prompt", "model": "model-x", "tenant_id": "t1", "kb_id": "kb1"},
        )
        == 10.0
    )


def test_metrics_are_silent_when_disabled() -> None:
    obs, registry = _regulated(enabled=False)
    obs.record_run("chat", tenant_id="t1")
    assert registry.get_sample_value("shop_runs_total") is None


def test_record_methods_never_raise_on_bad_values() -> None:
    obs, _ = _regulated()
    obs.record_tokens("prompt", "model", "not-a-number")
    obs.record_rerank_best("nope")
    obs.record_node_duration("node", -1)


def test_cache_metric_recorded() -> None:
    obs, registry = _regulated()
    obs.record_cache("hit", tenant_id="t1", kb_id="kb1")
    obs.record_cache("miss", tenant_id="t1", kb_id="kb1")

    assert (
        registry.get_sample_value(
            "shop_cache_results_total",
            {"result": "hit", "tenant_id": "t1", "kb_id": "kb1"},
        )
        == 1.0
    )
    assert (
        registry.get_sample_value(
            "shop_cache_results_total",
            {"result": "miss", "tenant_id": "t1", "kb_id": "kb1"},
        )
        == 1.0
    )
