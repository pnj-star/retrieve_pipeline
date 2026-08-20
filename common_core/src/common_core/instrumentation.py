"""管线的 span 打点工具（instrumentation 模块）。

给每个 node（retrieve / rerank / guard / generate ...）开启一个 span，
自动带上 node / tenant_id / kb_id / request_id 属性；异常时记录并标记
span 为 ERROR 后向上抛（不吞异常）。追踪未启用时是 no-op，零开销。
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

from . import telemetry

# OTel SDK 可用时才引入 Status，避免在未安装时 import 报错。
try:
    from opentelemetry.trace import Status, StatusCode

    _ERROR_STATUS = Status(StatusCode.ERROR)
except Exception:  # pragma: no cover
    _ERROR_STATUS = None


@contextmanager
def trace_node(
    node: str,
    *,
    tenant_id: str = "",
    kb_id: str = "",
    request_id: str = "",
) -> Iterator[Any]:
    """给一个 node 开启 span；异常时标记 ERROR 后向上抛。

    用法（同步 / 异步均可，OTel 基于 contextvars 自动传播）：
        with trace_node("retrieve", tenant_id=..., kb_id=..., request_id=...):
            docs = await vector.a_search_hybrid(...)

    追踪未启用时等价于 ``nullcontext()``，yield 出 None。

    参数:
        node: 节点名（如 retrieve/rerank/guard/generate），用于命名 span。
        tenant_id/kb_id/request_id: 附加到 span 的隔离与追踪属性。

    返回:
        一个上下文管理器；yield 出当前 span（追踪未启用时为 None）。
    """
    tracer = telemetry.get_tracer("rag")
    if tracer is None:
        yield None
        return
    with tracer.start_as_current_span(node) as span:
        span.set_attribute("node", node)
        if tenant_id:
            span.set_attribute("tenant_id", tenant_id)
        if kb_id:
            span.set_attribute("kb_id", kb_id)
        if request_id:
            span.set_attribute("request_id", request_id)
        try:
            yield span
        except Exception as exc:
            span.record_exception(exc)
            if _ERROR_STATUS is not None:
                span.set_status(_ERROR_STATUS)
            raise  # 不吞异常：指标/日志/上层兜底仍按原逻辑处理


def current_trace_id() -> str | None:
    """返回当前 span 的 trace_id，供日志打点做关联（16 位十六进制）。"""
    return telemetry.trace_id()
