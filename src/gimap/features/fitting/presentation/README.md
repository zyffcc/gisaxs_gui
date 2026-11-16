# Fitting presentation

本层保存 framework-neutral ViewModel state、Qt View 和 ViewBinding。ViewModel 只管理 UI
state、commands 和 use-case 结果映射，不执行 scientific calculation、具体 I/O 或 dialog。

`workspace.py` 的 `FittingWorkspace` 把两个页面放进 Fitting：Single analysis（`single/page.py` 的 `FitPage`）
和 In-situ series（`single/series_page.py` 的 `FitSeriesPage`），由 `workspace_context.py` 切换。
`control_view_factory.py` 只建一个空的 `gisaxsFittingPage`，由 workspace 填充。静态布局在 `views/*_view.py`。

`view_binding.py` 的 `FittingViewBinding` 只管页面之外的部分：1D Predict 窗口（`workflow_v5_dialog.py`）、
工程参数和关闭时保存的会话。这些模块不调用具体 adapter、TensorFlow、BornAgain 或 scientific runtime，
也不得在 presentation 内复制 application workflow。
