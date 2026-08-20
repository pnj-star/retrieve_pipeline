"""Prometheus 业务指标采集组件，支持可配置指标前缀，自动附加 tenant、kb 标签。
和旧版只针对 agent 的指标模块不同，本实现是一个类：每个 Skill 可以独立实例化，传入自己的指标注册器 (registry)、指标前缀、总开关。
当指标采集被关闭时，所有记录方法都是安全空操作，永远不会抛出异常。
"""

from __future__ import annotations

import contextvars
import logging
import threading
import time
from typing import Any

from prometheus_client import CollectorRegistry, Counter, Histogram, start_http_server

from .config import MetricsConfig

logger = logging.getLogger(__name__)


class Observability:
    """指标采集器：封装了一组 Prometheus 指标（运行时长、转交、守护、重排分数、token 消耗等）。所有记录方法都经过 _safe 保护，在关闭或异常时安静不影响业务。    """
    def __init__(
        self,
        config: MetricsConfig | None = None,
        *,
        prefix: str | None = None,
        enabled: bool | None = None,
        port: int | None = None,
        bind: str | None = None,
        registry: CollectorRegistry | None = None,
    ) -> None:
        """初始化指标采集器并去创建所有指标。
参数:
    config: 指标配置。
    prefix: 指标名前缀；None 时用 config.prefix。
    enabled: 总开关；None 时用 config.enabled。
    port: HTTP 指标服务端口。
    bind: HTTP 指标服务绑定地址。
    registry: 指定的 Prometheus 注册器；None 时自动创建。
        """
        cfg = config or MetricsConfig()
        self.prefix = prefix or cfg.prefix or "app"
        self.enabled = bool(enabled if enabled is not None else cfg.enabled)
        self.port = port if port is not None else cfg.port
        self.bind = bind if bind is not None else cfg.bind
        self._registry = registry or CollectorRegistry(auto_describe=True)
        self._run_start: contextvars.ContextVar[float | None] = contextvars.ContextVar(
            f"{self.prefix}_run_start", default=None
        )
        self._server_started = False
        self._server_lock = threading.Lock()
        self._build_metrics()

    @classmethod
    def from_env(
        cls,
        prefix: str = "METRICS_",
        registry: CollectorRegistry | None = None,
    ) -> "Observability":
        """从环境变量构造指标采集器。
参数:
    prefix: 环境变量前缀，默认 METRICS_。
    registry: 指定注册器。
返回:
    配置好的 Observability 实例。
        """
        return cls(MetricsConfig.from_env(prefix=prefix), registry=registry)

    def _name(self, suffix: str) -> str:
        """组合前缀与后缀生成 Prometheus 指标名。
参数:
    suffix: 指标后缀。
返回:
    f“{prefix}_{suffix}” 形式的指标名。
        """
        return f"{self.prefix}_{suffix}"

    def _build_metrics(self) -> None:
        """初始化所有 Prometheus 指标（Counter/Histogram），并备好各类分组标签。

        提交（无返回值）：创建运行次数、运行时长、转交、护持结果、
        重排分数、检索空、token 消耗、节点耗时、节点错误、缓存结果等指标。
        """
        tenant_labels = ("tenant_id", "kb_id")
        self._runs_total = Counter(
            self._name("runs_total"),
            "Completed agent runs by terminal route",
            ("route", *tenant_labels),
            registry=self._registry,
        )
        self._run_duration = Histogram(
            self._name("run_duration_seconds"),
            "End-to-end agent run duration",
            tenant_labels,
            buckets=(
                0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 20.0, 40.0, 60.0
            ),
            registry=self._registry,
        )
        self._handoffs_total = Counter(
            self._name("handoffs_total"),
            "Human handoffs by reason",
            ("reason", *tenant_labels),
            registry=self._registry,
        )
        self._guard_results_total = Counter(
            self._name("guard_results_total"),
            "Guard outcomes (pass / retry / fail_exhausted)",
            ("result", *tenant_labels),
            registry=self._registry,
        )
        self._rerank_best_ce = Histogram(
            self._name("rerank_best_ce_score"),
            "Best reranker ce_score per run (0-1)",
            tenant_labels,
            buckets=(
                0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0
            ),
            registry=self._registry,
        )
        self._retrieval_empty_total = Counter(
            self._name("retrieval_empty_total"),
            "Retrieval runs returning zero text docs",
            tenant_labels,
            registry=self._registry,
        )
        self._llm_tokens_total = Counter(
            self._name("llm_tokens_total"),
            "LLM tokens consumed",
            ("type", "model", *tenant_labels),
            registry=self._registry,
        )
        self._node_duration = Histogram(
            self._name("node_duration_seconds"),
            "Per-node execution time",
            ("node", *tenant_labels),
            buckets=(
                0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0
            ),
            registry=self._registry,
        )
        self._node_errors_total = Counter(
            self._name("node_errors_total"),
            "Node exceptions",
            ("node", *tenant_labels),
            registry=self._registry,
        )
        self._handoff_persist_failures_total = Counter(
            self._name("handoff_persist_failures_total"),
            "Handoff persist failures",
            tenant_labels,
            registry=self._registry,
        )
        self._cache_results_total = Counter(
            self._name("cache_results_total"),
            "Response cache outcomes (hit/miss/write/skip/error)",
            ("result", *tenant_labels),
            registry=self._registry,
        )

    def _labels(self, tenant_id: str = "", kb_id: str = "", **extra: Any) -> dict[str, str]:
        """构建带有 tenant/kb 与额外标签的标签字典，为记录方法提供统一内容。

        参数:
            tenant_id: 租户 ID 标签。
            kb_id: 知识库 ID 标签。
            **extra: 其它标签，值会被转为字符串。

        返回:
            可用于 Prometheus labels() 的标签字典。
        """
        return {
            "tenant_id": tenant_id or "",
            "kb_id": kb_id or "",
            **{key: str(value) for key, value in extra.items()},
        }

    def _safe(self, fn: Any, *args: Any, **kwargs: Any) -> None:
        """安全执行一个指标操作：关闭时直接返回，异常时记录调试日志，决不向上抛出。
参数:
    fn: 待执行的指标操作快括。
    *args/**kwargs: 传给 fn 的参数。
        """
        if not self.enabled:
            return
        try:
            fn(*args, **kwargs)
        except Exception:
            logger.debug("metrics record failed (ignored)", exc_info=True)

    def begin_run(self) -> None:
        """记录一次运行的开始时刻（仅在未开始时设置，避免重复进入覆盖）。

        提交（无返回值）。
        """
        if self._run_start.get() is None:
            self._run_start.set(time.perf_counter())

    def end_run(
        self,
        route: str = "",
        tenant_id: str = "",
        kb_id: str = "",
    ) -> None:
        """结束一次运行：统计运行次数与时长（若有开始时刻）并清空状态。
参数:
    route: 终端路径。
    tenant_id/kb_id: 隔离标签。
        """
        labels = self._labels(tenant_id=tenant_id, kb_id=kb_id, route=route or "unknown")
        self._safe(self._runs_total.labels(**labels).inc)
        start = self._run_start.get()
        if start is not None:
            duration_labels = self._labels(tenant_id=tenant_id, kb_id=kb_id)
            self._safe(
                self._run_duration.labels(**duration_labels).observe,
                time.perf_counter() - start,
            )
            self._run_start.set(None)

    def record_run(self, route: str, tenant_id: str = "", kb_id: str = "") -> None:
        """记录一次运行次数。

        参数:
            route: 运行所走的终端路径。
            tenant_id/kb_id: 隔离标签。

        提交（无返回值）。
        """
        labels = self._labels(tenant_id=tenant_id, kb_id=kb_id, route=route or "unknown")
        self._safe(self._runs_total.labels(**labels).inc)

    def record_handoff(self, reason: str, tenant_id: str = "", kb_id: str = "") -> None:
        """记录一次人工转交次数。

        参数:
            reason: 转交原因。
            tenant_id/kb_id: 隔离标签。

        提交（无返回值）。
        """
        labels = self._labels(
            tenant_id=tenant_id, kb_id=kb_id, reason=reason or "unknown"
        )
        self._safe(self._handoffs_total.labels(**labels).inc)

    def record_guard(self, result: str, tenant_id: str = "", kb_id: str = "") -> None:
        """记录守护结果（pass/retry/fail_exhausted）次数。

        参数:
            result: 守护结果类型。
            tenant_id/kb_id: 隔离标签。

        提交（无返回值）。
        """
        labels = self._labels(
            tenant_id=tenant_id, kb_id=kb_id, result=result or "unknown"
        )
        self._safe(self._guard_results_total.labels(**labels).inc)

    def record_rerank_best(self, score: float, tenant_id: str = "", kb_id: str = "") -> None:
        """记录本次运行中重排最佳 ce_score（限住在 0～1 范围）。
参数:
    score: 原始得分，会被剪辑到 [0,1]。
        """
        try:
            clamped = max(0.0, min(1.0, float(score)))
        except (TypeError, ValueError):
            return
        labels = self._labels(tenant_id=tenant_id, kb_id=kb_id)
        self._safe(self._rerank_best_ce.labels(**labels).observe, clamped)

    def record_retrieval_empty(self, tenant_id: str = "", kb_id: str = "") -> None:
        """记录一次检索返回零条文本的情况。

        参数:
            tenant_id/kb_id: 隔离标签。

        提交（无返回值）。
        """
        labels = self._labels(tenant_id=tenant_id, kb_id=kb_id)
        self._safe(self._retrieval_empty_total.labels(**labels).inc)

    def record_tokens(
        self,
        token_type: str,
        model: str,
        count: int | float,
        tenant_id: str = "",
        kb_id: str = "",
    ) -> None:
        """记录工具消耗的 token 数量。
参数:
    token_type: prompt/completion 等类别。
    model: 消耗所属的模型名。
    count: 要累加的 token 数；<=0 时忽略。
        """
        try:
            n = int(count)
        except (TypeError, ValueError):
            return
        if n <= 0:
            return
        labels = self._labels(
            tenant_id=tenant_id,
            kb_id=kb_id,
            type=token_type or "unknown",
            model=model or "unknown",
        )
        self._safe(self._llm_tokens_total.labels(**labels).inc, n)

    def record_node_duration(
        self, node: str, seconds: float, tenant_id: str = "", kb_id: str = ""
    ) -> None:
        """记录单个节点的执行时长。
参数:
    node: 节点名。
    seconds: 执行时长（秒）。
        """
        labels = self._labels(tenant_id=tenant_id, kb_id=kb_id, node=node or "unknown")
        self._safe(self._node_duration.labels(**labels).observe, seconds)

    def record_node_error(self, node: str, tenant_id: str = "", kb_id: str = "") -> None:
        """记录节点异常次数。

        参数:
            node: 节点名。
            tenant_id/kb_id: 隔离标签。

        提交（无返回值）。
        """
        labels = self._labels(tenant_id=tenant_id, kb_id=kb_id, node=node or "unknown")
        self._safe(self._node_errors_total.labels(**labels).inc)

    def record_handoff_persist_failure(self, tenant_id: str = "", kb_id: str = "") -> None:
        """记录一次转交持久化失败。

        参数:
            tenant_id/kb_id: 隔离标签。

        提交（无返回值）。
        """
        labels = self._labels(tenant_id=tenant_id, kb_id=kb_id)
        self._safe(self._handoff_persist_failures_total.labels(**labels).inc)

    def record_cache(self, result: str, tenant_id: str = "", kb_id: str = "") -> None:
        """记录缓存结果（hit/miss/write/skip/error）次数。

        参数:
            result: 缓存结果类型。
            tenant_id/kb_id: 隔离标签。

        提交（无返回值）。
        """
        labels = self._labels(
            tenant_id=tenant_id,
            kb_id=kb_id,
            result=result or "unknown",
        )
        self._safe(self._cache_results_total.labels(**labels).inc)

    def start_server(self) -> None:
        """启动 Prometheus HTTP 指标服务（仅在启用且未启动时）；失败不致命。

        提交（无返回值）。
        """
        if not self.enabled:
            return
        with self._server_lock:
            if self._server_started:
                return
            try:
                start_http_server(self.port, addr=self.bind, registry=self._registry)
                self._server_started = True
                logger.info(
                    "Prometheus metrics server started on %s:%d (prefix=%s)",
                    self.bind,
                    self.port,
                    self.prefix,
                )
            except Exception as exc:
                logger.warning("metrics server failed to start (non-fatal): %s", exc)
