# Presentation

本目录拥有 Trainset 的 PyQt 页面、交互式画布与结果渲染。页面只暴露控件、signals 和展示
状态；legacy controller 暂时负责把页面事件桥接到现有 application/legacy workflows。
BornAgain、TensorFlow、文件读写、项目配置持久化和后台进程不得进入本目录。

顶层 workflow shell 与五个步骤分别由 `views/page_view.py`、`dataset_page_view.py`、
`preview_page_view.py`、`model_page_view.py`、`run_page_view.py` 和
`monitor_page_view.py` 维护。步骤内部的 catalog/plugin 参数编辑器、交互画布、任务状态和
模型层内容按运行时数据注入；View 文件只维护静态控件和布局，不包含业务流程。
