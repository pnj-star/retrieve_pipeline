"""OpenTelemetry 链路追踪基础件（telemetry 模块）。

目标：
- 提供统一的 Tracer / span 入口，service name、OTLP 导出地址都走环境变量；
- 未配置导出器（或 SDK 未安装）时自动退化为 no-op，绝不让追踪成为硬依赖、
  拖垮主链路；
- 提供 W3C ``traceparent`` 的注入 / 提取工具，为后续跨进程
  （agent → MCP → 管线）链路传播做准备。

环境变量（沿用 OTel 惯例）：
- ``OTEL_SERVICE_NAME``：服务名，缺省 ``rag-skill``；
- ``OTEL_TRACES_EXPORTER``：``otlp`` / ``console`` / ``none``，缺省 ``none``；
- ``OTEL_EXPORTER_OTLP_ENDPOINT``：OTLP 导出地址；为 ``otlp`` 时的目标端点。

设计约束：所有初始化与工具函数都是"失败静默"的——初始化异常只记 warning，
绝不抛出；未初始化 / 未装 SDK 时返回 None 或 no-op，调用方无需判空兜底。
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# OTel SDK 可选引入：装了就启用，没装（或损坏）就整模块退化为 no-op。
try:
    from opentelemetry import context as _otel_context
    from opentelemetry import propagate as _otel_propagate
    from opentelemetry import trace as _otel_trace

    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

    _OTEL_AVAILABLE = True
except Exception:  # pragma: no cover - 取决于运行环境是否安装 SDK
    _OTEL_AVAILABLE = False

_initialized = False
_initialized_error: Exception | None = None


def init_telemetry(
    *,
    service_name: str | None = None,
    endpoint: str | None = None,
) -> None:
    """初始化全局 TracerProvider（幂等，可多次调用）。

    仅在"SDK 可用且配置了导出器"时才真正设置全局 provider；
    其余情况保留 OTel 默认 no-op provider，避免把 span 无限积压在内存里。

    参数:
        service_name: 服务名；为 None 时读取 OTEL_SERVICE_NAME，缺省 "rag-skill"。
        endpoint: OTLP 导出地址；为 None 时读取 OTEL_EXPORTER_OTLP_ENDPOINT。

    提交（无返回值）。
    """
    global _initialized, _initialized_error
    if _initialized:
        return
    _initialized = True
    if not _OTEL_AVAILABLE:
        logger.debug("opentelemetry 未安装，链路追踪退化为 no-op")
        return

    service = service_name or os.getenv("OTEL_SERVICE_NAME") or "rag-skill"
    exporter = os.getenv("OTEL_TRACES_EXPORTER") or "none"
    if exporter == "none":
        return

    try:
        if exporter == "otlp":
            otlp_endpoint = (
                endpoint
                or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
                or "http://localhost:4318/v1/traces"
            )
            span_processor = BatchSpanProcessor(
                OTLPSpanExporter(endpoint=otlp_endpoint)
            )
        elif exporter == "console":
            from opentelemetry.sdk.trace.export import ConsoleSpanExporter

            span_processor = BatchSpanProcessor(ConsoleSpanExporter())
        else:
            logger.warning("未知的 OTEL_TRACES_EXPORTER=%r，跳过初始化", exporter)
            return

        provider = TracerProvider(
            resource=Resource.create({"service.name": service})
        )
        provider.add_span_processor(span_processor)
        _otel_trace.set_tracer_provider(provider)
        logger.info("OpenTelemetry 已初始化 service=%s exporter=%s", service, exporter)
    except Exception as exc:  # 初始化失败也不能影响主链路
        _initialized_error = exc
        logger.warning("OpenTelemetry 初始化失败，追踪退化为 no-op: %s", exc)


def get_tracer(scope: str = "rag") -> Any | None:
    """返回指定 scope 的 Tracer；未初始化 / 不可用时返回 None。

    首次调用会触发惰性初始化（读环境变量），幂等。

    参数:
        scope: 追踪器的作用域名，默认 "rag"。

    返回:
        指定 scope 的 Tracer，或 SDK 不可用/初始化失败时的 None。
    """
    if not _initialized:
        init_telemetry()
    if not _OTEL_AVAILABLE or _initialized_error is not None:
        return None
    return _otel_trace.get_tracer(scope)


def inject_traceparent() -> str | None:
    """把当前上下文序列化为 W3C ``traceparent`` 请求头，供下游传播。

    当前没有有效 span 时返回 None（调用方跳过注入）。

    返回:
        W3C ``traceparent`` 字符串；不可用时返回 None。
    """
    if not _OTEL_AVAILABLE:
        return None
    carrier: dict[str, str] = {}
    _otel_propagate.inject(carrier)
    return carrier.get("traceparent")


def parse_traceparent(header: str | None) -> Any | None:
    """解析收到的 ``traceparent`` 头，返回 OTel 上下文对象。

    把返回的上下文 ``attach`` 为当前上下文后，后续新建的 span 会成为它的
    子 span，从而实现跨进程链路不断。解析失败 / 头部非法返回 None。

    参数:
        header: 收到的 ``traceparent`` 头字符串。

    返回:
        解析出的 OTel 上下文对象；失败或 SDK 不可用时返回 None。
    """
    if not header or not _OTEL_AVAILABLE:
        return None
    try:
        return _otel_propagate.extract({"traceparent": header})
    except Exception:
        logger.debug("traceparent 解析失败（忽略）: %r", header, exc_info=True)
        return None


def set_current_context(ctx: Any) -> Any | None:
    """把解析出的上游上下文设为当前上下文，返回可恢复的 token。

    返回的 token 应在请求结束后传给 ``reset_context`` 恢复现场。

    参数:
        ctx: 由 ``parse_traceparent`` 解析出的上下文对象。

    返回:
        用于恢复上下文的 token；ctx 为空或 SDK 不可用时返回 None。
    """
    if ctx is None or not _OTEL_AVAILABLE:
        return None
    return _otel_context.attach(ctx)


def reset_context(token: Any | None) -> None:
    """恢复 attach 之前的上下文。

    参数:
        token: ``set_current_context`` 返回的 token。

    提交（无返回值）。
    """
    if token is not None and _OTEL_AVAILABLE:
        _otel_context.detach(token)


def trace_id() -> str | None:
    """返回当前 span 的 trace_id（32 位十六进制小写），用于日志关联。

    没有活跃 span（或追踪未启用）时返回 None；日志里按
    ``trace_id=None`` 处理即可，不影响业务。

    返回:
        32 位十六进制 trace_id 字符串；无活跃 span 或 SDK 不可用时返回 None。
    """
    if not _OTEL_AVAILABLE:
        return None
    try:
        ctx = _otel_trace.get_current_span().get_span_context()
    except Exception:
        return None
    if not ctx.is_valid:
        return None
    return f"{ctx.trace_id:032x}"


def is_enabled() -> bool:
    """追踪是否真正可用（SDK 已装且初始化成功）。

    返回:
        SDK 可用且初始化成功时为 True，否则为 False。
    """
    if not _initialized:
        init_telemetry()
    return _OTEL_AVAILABLE and _initialized_error is None
