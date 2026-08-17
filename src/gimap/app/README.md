# `app`

`app` 是极薄的 composition root，负责构建 adapters、注入 use cases、管理应用启动
生命周期和顶层 navigation。

它可以在装配阶段同时引用 presentation、application ports 和具体 adapters，但不得
包含科学计算、feature 工作流、TensorFlow inference 或 BornAgain simulation。
`main.py` 在渐进迁移完成前仍是现有启动入口。
