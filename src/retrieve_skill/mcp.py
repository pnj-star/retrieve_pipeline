"""retrieve_skill MCP 服务端入口（薄层）。

参考 ``streamable_server.py``：本文件只负责命令行参数解析、加载配置并调用
``server.run(...)`` 启动服务；服务器构建与工具定义在 ``mcp_server.py`` 里。
支持 stdio / sse / streamable_http 三种传输方式。入口必须通过已安装的包调用：
``python -m retrieve_skill.mcp`` 或 ``retrieve-skill-mcp``，不再直接引用
工作区里的 ``common_core/src`` 源码目录；也支持 IDE 把本文件作为脚本直接运行
（此时仅移除脚本目录遮蔽，包名统一由 editable 安装解析，避免旧版安装包遮蔽）。
脚本方式运行时不传 ``--transport`` 会默认启动 streamable-http，因此 IDE 里
按 Run 即开即用；已安装包 / ``python -m`` 方式仍默认 stdio。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

if not __package__:
    # 脚本方式直接运行（如 PyCharm Run 按钮）时，Python 会把脚本所在目录
    # 放进 sys.path[0]。该目录里的 mcp.py 会遮蔽第三方 mcp 包，因此只移除它；
    # common_core / retrieve_skill 由 editable 安装提供，不再手动拼源码路径。
    script_dir = str(Path(sys.argv[0]).resolve().parent)
    sys.path[:] = [
        entry
        for entry in sys.path
        if entry and str(Path(entry).resolve()) != script_dir
    ]

from common_core.observability import Observability

if __package__:
    from .mcp_server import create_mcp_server
else:
    from retrieve_skill.mcp_server import create_mcp_server


def _normalize_transport(value: str) -> str:
    """把 CLI 里两种拼法并归成一个：``streamable-http`` / ``streamable_http``。"""
    return "streamable-http" if value.replace("_", "-") == "streamable-http" else value


def _run_server(server: Any, transport: str) -> None:
    """按传输方式启动 FastMCP 服务（stdio / sse / streamable-http）。"""
    import asyncio

    if transport == "stdio":
        server.run(transport="stdio")
        return
    if transport == "sse":
        asyncio.run(server.run_sse_async())
        return
    asyncio.run(server.run_streamable_http_async())


def _build_runtime(env_file: str | None) -> tuple[Any, str]:
    """解析并加载 .env，返回校验后的 RuntimeConfig 与配置来源。

    环境文件优先级：--env-file > RETRIEVE_SKILL_ENV_FILE > 当前工作目录的 .env；
    找不到时再兜底读取 retrieve_skill 项目根目录下的 .env。CLI 与 IDE 一键启动
    两个入口共用这里，保证配置（Milvus / Redis / Embedding 等）加载一致。
    """
    from pathlib import Path

    from common_core.config import (
        RuntimeConfig,
        load_env_files,
        log_config_audit,
        resolve_env_file,
    )

    resolved = resolve_env_file(env_file, env_key="RETRIEVE_SKILL_ENV_FILE")
    if resolved is None:
        # 直接以脚本方式运行（例如 IDE Run 按钮）时，工作目录不一定是项目根目录。
        # 兜底用 retrieve_skill 包根目录下的 .env，避免配置找不到。
        fallback = Path(__file__).resolve().parents[2] / ".env"
        if fallback.is_file():
            resolved = str(fallback)
    if resolved:
        load_env_files(resolved)
    runtime = RuntimeConfig.from_env()
    # retrieve_skill needs a local embedding model for dense retrieval.
    # LLM 三件套只在查询改写实际启用 LLM 模式时才要求，off 环境可留空。
    runtime.validate(require_embedding=True, require_llm=False)
    source = "env:" + resolved if resolved else "process-env"
    log_config_audit(
        runtime,
        source=source,
        require_embedding=True,
        require_llm=False,
    )
    return runtime, source


def main() -> None:
    """运行 MCP 服务端并加载配置，支持多种传输方式。

    默认传输：以脚本方式直接运行（IDE Run 按钮）时为 ``streamable-http``，
    方便一键启动 HTTP 服务；以已安装包 / ``python -m`` 方式运行时为 ``stdio``，
    适合被本地 MCP client 拉起。显式传 ``--transport`` 仍可覆盖。
    ``--transport sse`` / ``--transport streamable-http``（或 ``streamable_http``）
    会启动一个 HTTP 服务并暴露 URL，供远程评测平台或其它 MCP client 连接。
    缺少必需配置时快速报错终止。
    """
    import argparse

    # ``__package__`` 为真表示以包模块 / 安装后的 console script 方式执行，
    # 保持 MCP 客户端常见的 stdio 默认；否则视为 IDE 本地调试，默认起 HTTP。
    default_transport = "streamable-http" if not __package__ else "stdio"
    parser = argparse.ArgumentParser(prog="retrieve-skill-mcp")
    parser.add_argument(
        "--env-file",
        default=None,
        help="Path to a .env file (default: RETRIEVE_SKILL_ENV_FILE or ./.env)",
    )
    parser.add_argument(
        "--transport",
        default=default_transport,
        choices=["stdio", "sse", "streamable-http", "streamable_http"],
        help=(
            f"Transport to expose (default: {default_transport}). "
            "sse / streamable-http start an HTTP server."
        ),
    )
    parser.add_argument(
        "--host",
        default=None,
        help="Bind host for HTTP transports (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Bind port for HTTP transports (default: 8000)",
    )
    parser.add_argument(
        "--path",
        default=None,
        help="HTTP endpoint path for streamable-http (default /streamable) or sse (default /sse)",
    )
    args = parser.parse_args()

    runtime, config_source = _build_runtime(args.env_file)
    metrics = Observability(runtime.metrics)
    metrics.start_server()

    transport = _normalize_transport(args.transport)
    if args.host is not None or args.port is not None or args.path is not None:
        host = "127.0.0.1" if args.host is None else args.host
        port = 8000 if args.port is None else args.port
        streamable_path = args.path or "/streamable"
        sse_path = args.path or "/sse"
    else:
        host, port, streamable_path, sse_path = None, None, "/streamable", "/sse"
    server = create_mcp_server(
        host=host,
        port=port,
        streamable_path=streamable_path,
        sse_path=sse_path,
        metrics=metrics,
        runtime=runtime,
        config_source=config_source,
    )
    _run_server(server, transport)


if __name__ == "__main__":
    raise SystemExit(main())


def serve_default(
    *,
    transport: str = "streamable-http",
    host: str = "127.0.0.1",
    port: int = 8000,
    path: str = "/streamable",
) -> None:
    """直接启动默认 MCP 服务，供 IDE 一键运行 / 本机调试。

    不解析命令行参数，写死默认传输方式与端点，方便在 PyCharm 里以脚本方式
    （Script path）一键运行即启动 streamable-http。生产 / 复用场景仍走
    ``main()`` 的 CLI 参数（``python -m retrieve_skill.mcp``）。同样会先加载
    .env 并校验配置，保证 Milvus / Redis / Embedding 与 CLI 启动一致。
    """
    runtime, config_source = _build_runtime(None)
    metrics = Observability(runtime.metrics)
    metrics.start_server()
    server = create_mcp_server(
        host=host,
        port=port,
        streamable_path=path,
        sse_path="/sse",
        metrics=metrics,
        runtime=runtime,
        config_source=config_source,
    )
    _run_server(server, transport)


__all__ = ["create_mcp_server", "main", "serve_default"]
