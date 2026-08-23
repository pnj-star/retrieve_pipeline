"""pytest 全局配置：common_core 与 retrieve_skill 均通过已安装包加载。

retrieve_skill 的 ``src`` 由 pyproject.toml 的 pytest pythonpath 提供；
common_core 由 ``pip install``（editable 或发布包）提供，不再注入源码路径。
"""
