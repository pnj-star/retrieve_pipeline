"""rag_skill MCP 服务端入口（薄层）。

参考 ``streamable_server.py``：本文件只负责命令行参数解析、加载配置并调用
``server.run(...)`` 启动服务；服务器构建与工具定义在 ``mcp_server.py`` 里。
支持 stdio / sse / streamable_http 三种传输方式。
"""

from __future__ import annotations

if __name__ == "__main__" and not __package__:
    # 直接以脚本方式运行（例如 IDE 的 Run 按钮执行 python .../mcp.py）时，
    # 当前文件被当作顶层脚本，__package__ 为空，相对导入无法解析。这里把本包
    # 与同仓库 common_core 的源码目录加入 import 路径后，以包模块方式重进入口。
    import sys
    from pathlib import Path

    mcp_file = Path(__file__).resolve()
    for src_dir in (
        mcp_file.parents[1],                            # rag_skill/src
        mcp_file.parents[2] / "common_core" / "src",    # 同仓库 common_core/src
    ):
        if src_dir.exists():
            sys.path.insert(0, str(src_dir))
    # 脚本所在目录（src/rag_skill）会被 Python 自动放到 sys.path 最前面，
    # 里面有 mcp.py，会让 `import mcp`（第三方 MCP 库）解析到本文件，
    # 从而触发相对导入报错。这里把脚本目录从 sys.path 移除，让 mcp 落回真实库。
    script_dir = str(mcp_file.parents[0])
    if script_dir in sys.path:
        sys.path.remove(script_dir)

    from rag_skill.mcp import main

    raise SystemExit(main())

from typing import Any

from .mcp_server import create_mcp_server


def _normalize_transport(value: str) -> str:
    """把 CLI 里两种拼法并归成一个：``streamable-http`` / ``streamable_http``。"""
    return "streamable-http" if value.replace("_", "-") == "streamable-http" else value


def main() -> None:
    """运行 MCP 服务端并加载配置，支持多种传输方式。

    ``--transport stdio``（默认）走标准输入输出，适合被本地 MCP client 拉起；
    ``--transport sse`` / ``--transport streamable-http``（或 ``streamable_http``）
    会启动一个 HTTP 服务并暴露 URL，供远程评测平台或其它 MCP client 连接。

    环境文件优先级：--env-file > RAG_SKILL_ENV_FILE > 当前工作目录的 .env；
    找不到时再兜底读取 rag_skill 项目根目录下的 .env，避免 IDE Run 因配置
    缺失静默/报错。缺少必需配置时快速报错终止。
    """
    import argparse
    import asyncio
    from pathlib import Path

    from common_core.config import (
        RuntimeConfig,
        load_env_files,
        log_config_audit,
        resolve_env_file,
    )

    parser = argparse.ArgumentParser(prog="rag-skill-mcp")
    parser.add_argument(
        "--env-file",
        default=None,
        help="Path to a .env file (default: RAG_SKILL_ENV_FILE or ./.env)",
    )
    parser.add_argument(
        "--transport",
        default="stdio",
        choices=["stdio", "sse", "streamable-http", "streamable_http"],
        help="Transport to expose (default: stdio). sse / streamable-http start an HTTP server.",
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

    env_file = resolve_env_file(args.env_file, env_key="RAG_SKILL_ENV_FILE")
    if env_file is None:
        # 直接以脚本方式运行（例如 IDE Run 按钮）时，工作目录不一定是项目根目录。
        # 兜底用 rag_skill 包根目录下的 .env，避免配置找不到。
        fallback = Path(__file__).resolve().parents[2] / ".env"
        if fallback.is_file():
            env_file = str(fallback)
    if env_file:
        load_env_files(env_file)
    runtime = RuntimeConfig.from_env()
    # rag_skill needs a local embedding model for dense retrieval.
    runtime.validate(require_embedding=True)
    log_config_audit(
        runtime,
        source="env:" + env_file if env_file else "process-env",
        require_embedding=True,
    )

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
    )

    if transport == "stdio":
        server.run(transport="stdio")
        return
    if transport == "sse":
        asyncio.run(server.run_sse_async())
        return
    asyncio.run(server.run_streamable_http_async())


if __name__ == "__main__" and __package__:
    # 以 `python -m rag_skill.mcp` 方式运行时（__package__ 非空），顶层脚本
    # bootstrap 不会触发，这里兜底调用入口；以 `python mcp.py` 方式运行时，
    # 已在文件顶部 bootstrap 里通过 raise SystemExit(main()) 退出，不会走到这里。
    raise SystemExit(main())


__all__ = ["create_mcp_server", "main"]
