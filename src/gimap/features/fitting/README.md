# Fitting feature

Fitting 按 feature-first 方式渐进迁移。`domain` 保存可脱离 Qt、文件系统和外部
runtime 验证的科学数据结构与数值运算；`application` 编排 use cases 和 ports；
`infrastructure` 实现文件及模型 adapters；`presentation` 保存 ViewModel 和 Qt view binding。

Fitting workspace layout、Input/Cut/Run/Model cards、Preview controls、typed state 和 ViewModel
均位于 `presentation/`。生产运行时直接构造 `FittingViewBinding`；旧
`controllers/fitting_controller.py` 与 feature `legacy_bridge.py` 只保留外部 import 兼容，
不再承载运行时实现。
