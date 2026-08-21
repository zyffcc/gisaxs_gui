# Fitting presentation

本层保存 framework-neutral ViewModel state、Qt View 和 ViewBinding。ViewModel 只管理 UI
state、commands 和 use-case 结果映射，不执行 scientific calculation、具体 I/O 或 dialog。

`GisaxsFittingWorkspace` 和 feature-owned cards 只重组、渲染现有 Qt controls。AI controls、
global parameter editors、preview cards 与 splitter state 按明确 presentation 职责分开；这些
模块不调用具体 adapter、TensorFlow、BornAgain 或 scientific runtime，也不得在
presentation 内复制 application workflow。

`views/fitting_page_view.py` 是 Fitting 原始控件的 Python View，
`views/fitting_workspace_view.py` 是最终左右 splitter 和 section hierarchy 的 Python View；
`views/detector_parameters_dialog_view.py` 是 Detector Parameters 辅助窗体的静态布局来源。
`control_view_factory.py` 只负责实例化 View 和暴露稳定属性，cards
负责动态 AI/模型参数与 plot controls。

`views/independent_image_window_view.py` 和 `views/independent_fit_window_view.py` 拥有两个独立绘图
窗口的固定外壳；Matplotlib canvas、toolbar 和运行时 actions 通过明确 layout host 注入。
