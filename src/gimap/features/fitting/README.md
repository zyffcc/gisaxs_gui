# Fitting feature

Fitting 按 feature-first 方式组织。`domain` 保存可脱离 Qt、文件系统和外部
runtime 验证的科学数据结构与数值运算；`application` 编排 use cases 和 ports；
`infrastructure` 实现文件及模型 adapters；`presentation` 保存 ViewModel 和 Qt view binding。

Fitting 只拟合 1D 曲线：输入是 Analyze 写出的 `*_fit_input.dat`（`q I σ pixels` + `# observation`）
或 `.dat` / `.txt` 曲线文件（`application/models.py` 的 `CURVE_SUFFIXES`；不读 `.csv`）；探测器读取、几何、
切割都在 Analyze。Single analysis 拟合一条曲线（`presentation/single/page.py` 的 `FitPage`：
Curve → Model → Fit → Results），In-situ series 用 Single 的模型拟合一个文件夹里的每条曲线
（`presentation/single/series_page.py` 的 `FitSeriesPage`：Curves → Start → Results）。

显示的页面、静态布局（`presentation/views/`）、typed state 和 ViewModel 均位于 `presentation/`。旧的单曲线页
（Components / Global / Refine / 1D Predict 标签，`FittingViewBinding`）不再显示，但仍在后台构造：它的 binding
运行 1D Predict 窗口（Tools ▸ 1D Predict — Fit Many Curves…）并跟随新页面的曲线与模型
（`presentation/workspace.py` 的 `attach_legacy`）。旧的 `controllers/fitting_controller.py`、`utils/*.py`、
feature `legacy_bridge.py` 与 Cut & Fitting 的探测器部分已删除。`bootstrap.create_quick_fit()` 把数值物理拟合
交给组合根，供自动分析和 Batch Export 使用。界面与调用链见 `docs/ui/workspaces/fitting.md`。
