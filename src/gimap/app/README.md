# `app`

`app` 是极薄的 composition root，负责构建 adapters、注入 use cases、管理应用启动
生命周期和顶层 navigation。

它可以在装配阶段同时引用 presentation、application ports 和具体 adapters，但不得
包含科学计算、feature 工作流、TensorFlow inference 或 BornAgain simulation。
`main.py` 在渐进迁移完成前仍是现有启动入口。

`main_window.py` 拥有 application shell 的侧边导航、workspace 容器和 feature page
装配。它是允许依赖 feature presentation 的 composition boundary；共享 UI 组件仍留在
`app/presentation`，并保持不依赖任何 feature。

`runtime.py` 只装配 feature ViewModels/ViewBindings、连接顶层 navigation 和控制延迟启动；
Fitting session 与跨 workspace 参数分别由 `fitting_session.py` 和
`workspace_parameters.py` 协调。旧 `MainController` 名称仅为兼容 re-export。
