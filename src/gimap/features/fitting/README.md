# Fitting feature

Fitting 按 feature-first 方式组织。`domain` 保存可脱离 Qt、文件系统和外部
runtime 验证的科学数据结构与数值运算；`application` 编排 use cases 和 ports；
`infrastructure` 实现文件及模型 adapters；`presentation` 保存 ViewModel 和 Qt view binding。

Fitting 只拟合 1D 曲线：输入是 Analyze 写出的 `*_fit_input.dat`（`q I σ pixels` + `# observation`）
或任意曲线文件；探测器读取、几何、切割都在 Analyze。Single analysis 拟合一条曲线，In-situ series
拟合一个曲线序列（Live Watch / Process Existing Sequence，Recipe 版本化）。

Workspace layout（Curve 卡片、Components / Global / Refine / 1D Predict、曲线图与 plot options）、
typed state 和 ViewModel 均位于 `presentation/`。生产运行时直接构造 `FittingViewBinding`
（由一组 mixin 组成）；旧的 `controllers/fitting_controller.py`、`utils/*.py`、
feature `legacy_bridge.py` 与 Cut & Fitting 的探测器部分已删除。
