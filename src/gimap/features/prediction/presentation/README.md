# Prediction presentation

保存 typed UI state、ViewModel commands 和 Qt workers。ViewModel 不操作 widget/dialog，
不执行 preprocessing、TensorFlow inference 或具体文件 I/O。

`GisaxsPredictWorkspace` 拥有 Prediction 页面布局，`cards.py`、`control_style.py` 和
`preview_layout.py` 只重组、渲染现有 Qt controls，不调用 controller、use case 或科学运行时。
主窗口 View factory 和 legacy controller 是当前兼容边界，不得在 presentation 内复制其 workflow。

`views/prediction_page_view.py` 是 legacy binding 所需原始控件的 Python View，
`views/prediction_workspace_view.py` 是最终可见 section hierarchy 的 Python View。
`control_view_factory.py` 只负责实例化 View 和暴露兼容属性，`workspace.py` 只把同一控件实例
装入语义化 host。Controller 属性命名仍是明确兼容边界，
不得重新引入手写的第二套静态控件工厂。

多文件结果区及其 Export、Distribution Heatmap、Parameter Trend 辅助窗体也分别由
`views/multifile_results_widget_view.py`、`export_dialog_view.py`、
`distribution_heatmap_dialog_view.py`、`parameter_trend_dialog_view.py` 持有静态布局。
结果 model/filter、Matplotlib canvas 与数据驱动的曲线仍由运行时注入，不进入 View。
