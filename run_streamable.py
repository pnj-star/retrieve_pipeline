
from __future__ import annotations

import os
import sys
from pathlib import Path


# 模型已缓存在本地（bge-small-zh-v1.5 / bge-reranker-base）。
# 强制离线加载，避免 SentenceTransformer/CrossEncoder 在首次请求时去访问
# HF_ENDPOINT（例如 hf-mirror.com）做 HEAD 检查而卡住几十秒、导致 MCP
# 会话超时被关（报 "Cannot send a request, as the client has been closed"）。
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def main() -> None:
    """以 streamable-http 启动 MCP 服务端（委托给 rag_skill.mcp.main）。"""
    # 把源码目录加进 import 路径，让 `import rag_skill` / `import common_core` 命中。
    root = Path(__file__).resolve().parent
    for src_dir in (
        root / "src",
        root / "common_core" / "src",
    ):
        if src_dir.exists() and str(src_dir) not in sys.path:
            sys.path.insert(0, str(src_dir))

    from rag_skill.mcp import main as mcp_main


    args = list(sys.argv[1:]) if len(sys.argv) > 1 else [
        "--transport", "streamable-http",
        "--host", "127.0.0.1",
        "--port", "8000",
    ]
    sys.argv = [sys.argv[0], *args]
    raise SystemExit(mcp_main())


if __name__ == "__main__":
    main()
