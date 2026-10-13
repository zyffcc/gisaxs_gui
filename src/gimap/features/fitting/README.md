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
交给组合根，供自动分析和 Batch Export 使用。

主单曲线 `FitPage` 已有独立的 `fit1d` 方法：Model 给定 1–4 群 Sphere / Vertical cylinder（可重复），
Fit 中选择 fit1d，所有群在同一条曲线上共同拟合；general = `cond_v2_ft4`，au_np_si = `cond_v2_ftr1`，
默认 auto，可切 fast / robust。GISAXS 的真空 qz（nm⁻¹）需手填，未给时为 0 并提示；SAXS 忽略 qz、
不带结构因子与低 q 项。仅使用 Model 的组分，参数值、范围、固定项与 Global 设置不传给 v2。

调用链为 `presentation/single/fit1d.py` → `application/single_fit1d.py` → JobRunner →
`infrastructure/adapters/single_fit1d.py`，无 V5 / TensorFlow 资产依赖。`FitSession.native_result` 保存原 v2
总曲线、分量、effective σ、参数误差 / 状态和提示，绘图、Results 与 CSV / JSON 导出均使用这些原值。
无 σ 时明确提示 Poisson 假设；相对系统误差下限与 σ 按平方和合并。v2 是数目加权截断高斯尺寸分布与
Hosemann 准晶（w = w_rel·D），q 用 nm⁻¹，长度 nm；与手动模型的正演和宽度约定不同，映射只用作手动初值。
改 Model、曲线选择或 fit1d 设置会使原结果过期：旧 χ² / 表注明 previous fit1d，图显示手动起点，拒绝导出过期结果；
恢复原条件或重拟合后可保存。测试在 `tests/test_fit_page_fit1d.py`。

1D Predict 批量窗口的第四种方法 `fit1d`（`infrastructure/adapters/fit1d_method.py`）调用同一个 gisaxs1d 工具包
（`application/workflow_v5.fit1d_kit_home()`：环境变量 `GISAXS1D_KIT_HOME`，否则 `utils/gisaxs1d_gui_kit`）。工具包是
研究代码，只在 job 进程里导入；批量路线 `run_workflow_job` 逐曲线处理，默认 fast。
行里的曲线和分量是工具包原结果（约定写在每行的 `unit_contract` / `validation_scope`，标记写在 `warnings`），
选中行后保留原 v2 图，映射的 Components 仅作手动起点。界面与调用链见 `docs/ui/workspaces/fitting.md`。
