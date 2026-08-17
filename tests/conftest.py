"""pytest 全局配置：把 common_core 与 rag_skill 的 src 目录加入模块搜索路径。"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1].parent
sys.path.insert(0, str(ROOT / "common_core" / "src"))
sys.path.insert(0, str(ROOT / "rag_skill" / "src"))
