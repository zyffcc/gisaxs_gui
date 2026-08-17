# App presentation

这里保存由 GIMaP application shell 所有、可被多个 feature presentation 使用的通用
PyQt5 视觉组件。组件只负责布局、展示状态和发出 UI signals，不包含科学计算、文件读取、
JobRunner/process 管理或 feature workflow。

Feature-specific widgets 和 ViewModels 仍应留在对应 feature。该目录不是通用业务代码或
scientific shared kernel，也不得被 application/domain 导入。
