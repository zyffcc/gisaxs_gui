# Calibration application

Use cases 编排图像加载、运行校准、导入/导出、读取 detector catalog 和应用几何
参数。application 不导入 PyQt，也不直接访问文件系统或 `global_params`；所有外部
行为均通过 `ports/`。
