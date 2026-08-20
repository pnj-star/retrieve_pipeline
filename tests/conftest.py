"""pytest 全局配置：把 common_core 与 rag_skill 的 src 目录加入模块搜索路径。"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# common_core 是工作区共享库，位于 rag_skill 的上级目录（D:\my_project\Skill\common_core）。
sys.path.insert(0, str(ROOT.parent / "common_core" / "src"))
sys.path.insert(0, str(ROOT / "src"))
