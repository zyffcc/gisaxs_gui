# Fitting feature

Fitting 按 feature-first 方式组织。`domain` 保存可脱离 Qt、文件系统和外部
runtime 验证的科学数据结构与数值运算；`application` 编排 use cases 和 ports；
`infrastructure` 实现文件及模型 adapters；`presentation` 保存 ViewModel 和 Qt view binding。

Fitting 只拟合 1D 曲线：输入是 Analyze 写出的 `*_fit_input.dat`（`q I σ pixels` + `# observation`）
或 `.dat` / `.txt` 曲线文件（`application/models.py` 的 `CURVE_SUFFIXES`；不读 `.csv`）；探测器读取、几何、
切割都在 Analyze。Single analysis 拟合一条曲线（`presentation/single/page.py` 的 `FitPage`：
Curve → Model → Fit → Results），In-situ series 用 Single 的模型拟合一个文件夹里的每条曲线
（`presentation/single/series_page.py` 的 `FitSeriesPage`：Curves → Start → Results）。

显示的页面、静态布局（`presentation/views/`）、typed state 和 ViewModel 均位于 `presentation/`。
`presentation/view_binding.py` 的 `FittingViewBinding` 只剩页面之外的部分：1D Predict 窗口（Fitting ▸ 1D Predict，
以及 Single analysis 的按钮；输入是 Single analysis 的曲线及其两半、拟合范围和排除的点）、工程参数
（`app/workspace_parameters.py`）和关闭时保存的会话。旧的单曲线页（Components / Global / Refine / 1D Predict 标签、
`bindings/` 下的 53 个 mixin）与旧的 In-situ 页（Recipe、Live Watch）已于 2026-11-16 删除：它们早已不显示。
旧的 `controllers/fitting_controller.py`、`utils/*.py`、
feature `legacy_bridge.py` 与 Cut & Fitting 的探测器部分已删除。

Fit 步骤只有三个方法：Predict parameters（唯一的网络方法）、Refine the values in Model 和 Search the ranges, then
refine。Predict parameters 的调用链为 `presentation/single/predict.py` → `application/single_predict.py`
（`predict_request`：Particles 决定组成，Model 的角度给出切线的 q_z，sizes 与 coupling 取自 Model）→ JobRunner →
`infrastructure/adapters/posterior_job.py` → `domain/posterior/refine.predict`（唯一的路径：网络的重要性加权样本、
每个模式最小二乘精修、χ²/N > 2 时在范围内搜索、ΔBIC ≤ 20 的解做局部后验，按 evidence 或 BIC 比较）。
未拟合的 GISAXS 切线由 `presentation/single/method_hint.py` 推荐 Predict parameters。In-situ series 的
Predict each frame 在 `presentation/single/series_predict.py`。1D Predict 批量窗口（`workflow_v5_dialog.py`）也只有这一种方法：
`application/workflow_v5.validate_options` 把旧设置（V5、Single RC specialist、physical fit、fit1d）一律打开为
Predict parameters，fit1d 的切线 qz 保留为 `cut_qz`。`bootstrap.create_curve_fit()` 把同一方法交给组合根，供自动分析、
Batch Export、Process with AI 和命令行使用。

fit1d 工具包（`utils/gisaxs1d_gui_kit`，Predict parameters (fit1d)、1D Predict 的 fit1d）和快速物理拟合
（Find the particle shape、1D Predict 的 Physical fit、`create_quick_fit()`）已在 0.2.0 删除，留在仓库历史里；
`tools/benchmarks/fit1d_recovery.py` 等对照脚本需要另给一份工具包（`tools/benchmarks/README.md`）。
界面与调用链见 `docs/ui/workspaces/fitting.md`。
