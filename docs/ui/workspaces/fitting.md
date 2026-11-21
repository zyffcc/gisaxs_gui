<!-- 2026-11-18: one network method (Predict parameters); fit1d and Find the particle shape removed. -->
# Fitting 界面与交互说明

- **Status**: Current
- **Scope**: Fitting workspace（1D 曲线拟合：单条曲线与 in-situ 曲线序列）的布局、控件映射与手动验收
- **Related code**:
  [`src/gimap/features/fitting/presentation/`](../../../src/gimap/features/fitting/presentation/)、
  [`src/gimap/app/main_window.py`](../../../src/gimap/app/main_window.py)
- **NXcanSAS**: `.h5` 曲线由 `infrastructure/adapters/nxcansas_curves.py` 读取（`InSituSourceFrame.frame_index` 是文件中的
  第几条曲线；Q 的单位来自文件，1/m 换算为 1/nm；Idev 为 σ），`series_frames.is_curve_file` / `is_frame` 按内容区分 NXcanSAS
  与探测器帧；单条分析的 Open Curve… 对多条曲线的文件用 `QInputDialog` 问选哪一条（`page.open_curve_file`），Import Curves…
  取全部（`CurveTablePreview.curve_titles`）；Save Data and Fit… 可存为 NXcanSAS（`application/fit_nxcansas.py`）。
  测试 [`tests/test_nxcansas.py`](../../../tests/test_nxcansas.py)。
- **数据与保存**（2026-11-21）：`.csv` 按表头列名选 q、I、σ（`infrastructure/adapters/curve_columns.py`；说不清时
  `COLUMNS_TO_CHOOSE`，页面用 `presentation/curve_columns_dialog.py` 问）；保存 ▸ 解… 与数据与拟合记录里的 `predict`
  （`application/solution_export.py`、`single/solutions_export.py`）；项目保存拟合结果与解（`application/fit_state.py`、
  `single/project.py`、`single/series_project.py`）；预测的阶段与停止后保留已完成的解（`posterior_job.StageReport`、
  `refine.predict` 的 `progress` / `partial`，只报告、不改计算；`single/predict_progress.py`）；精修来源与解的编号、标记
  （`single/solution_texts.py`、`single/equivalent_solutions.py`）；保存图带残差（`single/figure_export.py`、
  `adapters/curve_figures.py`）；撤销 / 重做说明是哪一步（`single/history.py`）。测试
  [`tests/test_fit_data_files.py`](../../../tests/test_fit_data_files.py)、
  [`tests/test_fit_results_kept.py`](../../../tests/test_fit_results_kept.py)。
- **Related tests**:
  [`tests/test_fit_page.py`](../../../tests/test_fit_page.py)、
  [`tests/test_fit_page_predict.py`](../../../tests/test_fit_page_predict.py)、
  [`tests/test_ml1d_posterior.py`](../../../tests/test_ml1d_posterior.py)、
  [`tests/test_fit_engine.py`](../../../tests/test_fit_engine.py)、
  [`tests/test_fit_series.py`](../../../tests/test_fit_series.py)、
  [`tests/test_fit_series_halves.py`](../../../tests/test_fit_series_halves.py)、
  [`tests/test_fit_series_table.py`](../../../tests/test_fit_series_table.py)、
  [`tests/test_signed_cut.py`](../../../tests/test_signed_cut.py)、
  [`tests/test_fitting_presentation.py`](../../../tests/test_fitting_presentation.py)、
  [`tests/test_fitting_insitu_series.py`](../../../tests/test_fitting_insitu_series.py)、
  [`tests/test_analyze_fit_input_contract.py`](../../../tests/test_analyze_fit_input_contract.py)、
  [`tests/test_ui_workspace_layouts.py`](../../../tests/test_ui_workspace_layouts.py)、
  [`tests/test_release2_defaults_template.py`](../../../tests/test_release2_defaults_template.py)（起始模型的模板）、
  [`tests/test_release2_fitting_cylinder_key.py`](../../../tests/test_release2_fitting_cylinder_key.py)（每种形状一个块）
- **Last verified**: 2026-10-14（两半、In-situ series 与从 Analyze 接收曲线各节对照代码；其余沿用 2026-10-01）

## 分工：Analyze 出曲线，Fitting 拟合曲线

原 “Cut & Fitting” 页的探测器部分（图像预览、center / Yoneda、cut 选区、mask / gap fill /
threshold / flip、stack 与 q 轴设置）已全部下线，由 [Analyze](analyze.md) 承担：

| 原 Cut & Fitting 功能 | 现在的位置 |
| --- | --- |
| 打开 CBF / NXS / TIFF、上一张 / 下一张 | Analyze 文件列表 |
| detector 几何、beam centre | Analyze 的 instrument profile 与 Beam centre 菜单 |
| Find Yoneda & Set Cut、cut 厚度 | Analyze 自动水平 cut（可拖动 / 手动行区间） |
| Optimize Center X | Analyze ▸ Beam centre ▸ Refine x by Symmetry |
| 负像素 / 坏点 mask 与 3 px guard | Analyze ▸ Mask & corrections 步骤 ▸ 折叠的 Corrections ▸ Gap guard（默认 3 px，可保存） |
| Stack 多帧 | Analyze ▸ Data 步骤 ▸ 折叠的 Frames ▸ Sum（N 帧相加） |
| Mirror fill / Flip UD | 不再提供：Analyze 使用 canonical 方向，缺口不参与平均、不跨缺口插值 |
| In-situ CBF / NXS 序列 | Analyze ▸ Send Series to Fitting（先导出曲线，再在 Fitting 拟合） |

q 由 Analyze 的精确 grazing-incidence 公式给出；与旧 Fitting 的近似 q 在 GISAXS 范围内
相差不到 1 px，因此没有保留“旧 q 公式”开关。

## 布局

```text
Fitting
├── Single analysis（single/page.py，布局 views/fit_page_view.py + fit_steps_view.py）
│   ├── 命令栏：Open Curve…（菜单 Load Model…）、Undo / Redo、曲线名、Fit、Stop、Save ▾
│   ├── 左：步骤栏 Curve → Model → Fit → Results，下面是当前步骤的控件
│   └── 右：曲线图（点、模型、Terms 各项、Exclude、橙色拟合范围可拖动、Log q / Log I、±q 选择）+ 残差图
└── In-situ series（single/series_page.py，布局 views/fit_series_view.py）
    ├── 命令栏：Choose Folder…、文件夹（长路径只显示最后三级，悬停看全路径）、Start / Pause / Stop、Save ▾
    ├── 左：步骤栏 Curves → Start → Results，下面是当前步骤的控件
    └── 右：选中帧的曲线与模型；下面是一个值随帧的变化（点 ± 1σ 误差棒，或 χ²ᵣ）
```

菜单栏的 **Fitting** 菜单列出两页的命令：Open Curve…、Fit a Folder of Curves…、Fit（Ctrl+Return）、Stop Fit、Load Model… /
Save Model…、Save Data and Fit…、Save Plot…、In-situ Series ▸ Start / Stop、1D Predict — Fit Many Curves…；
**Edit ▸ Undo / Redo** 撤销 Single analysis 里模型的改动。条目在 `app/presentation/workspace_menus.py`（`FITTING_MENU`、
`EDIT_MENU`），由组合根 `app/workspace_commands.py` 注入；从别的页打开时先切到 Fitting，键仍归页面所有。

- **Curve**：卡片（`single/curve_info.py`）写点数、q 范围（nm⁻¹；Å⁻¹ 在卡片提示里）、“σ from the file”；没有 σ 时卡片里一行警告写明各方法怎么做
  （“⚠ No σ in the file. Predict parameters: a relative σ of … %, estimated from the scatter of ln I about its trend. Refine and
  Search: ln I is fitted, every point weighted alike.”），Fit 步骤里按所选方法说（Predict：`curve_intake.SIGMA_ESTIMATED`，
  和 1D Predict 的记录同一句；其余：“⚠ No σ in the file: every point weighted alike”）。曲线接收时的说明统一由
  `application/curve_intake.py` 给出（`IntakeNote`：`q_unit` 未注明单位而 q 范围不合理、`sigma_estimated`、`nonpositive`
  负值和零点、`unreadable_rows`），页面用 `intake_texts.py` 翻译；页面上 `curve_notes()` 返回它们，Results 步骤的
  `fitCurveNotes` 在数值旁边显示警告，1D Predict 的行和记录里是 `curve_notes`。文件名只在命令栏。
  步骤标题下的说明行只在步骤栏和控件没说的时候出现（`show_line`，空时隐藏），用法写在控件提示里；切线两半（Mean of both halves / Both halves on |q| / q > 0 half /
  q < 0 half，`single_fit.SIDES`）。平均（`shared/signed_cut.py`）只在另一半两侧 2.5 个步长内都有测量点处求平均，其余处
  （另一半的缺口、另一半到头之后）用这一半原样，不跨缺口插值。Predict parameters 只能拟合一条曲线（`ONE_CURVE_METHODS`）：
  选着 Both halves on |q| 时 Fit 先把下拉换成平均，并说 “Both halves on |q| cannot be fitted by this method: the mean of
  both halves is used.”；Refine 与 Search 照旧拟合两半；
  拟合范围（nm⁻¹，或拖动图上的橙色区域；Whole Curve）；折叠的 *File* 里是 *q unit of files without one in their header*：
  文件头写明 q 单位的文件按写明的单位读（Analyze 的 `_fit_input.dat`：1/A），这个设置只用于没写明的文件，原位序列也用它
  （规则 `application/curve_units.q_unit_read`，在 `LoadCurve` 里执行；页面部分在 `single/curve_unit.py`）。Curve 卡片最后一行
  “q read in Å⁻¹: stated in the file / not stated in the file: the setting for files without one”；改设置只重开没写明单位的曲线；
  同时写明两种单位的文件打不开（“Could not open …: The header states q both in 1/A and in 1/nm; …”）。拟合记录的 `curve` 有
  `q_unit_of_file`、`q_unit_from`；项目存 `curve.unit`（读取用的单位）和 `unit_setting`。旧项目：没写明单位的文件按存的单位打开，
  与以前相同；文件头写明了别的单位时按文件头读，项目的提示里说明（`SAVED_OTHER`）。测试
  `tests/test_release3_curve_units.py`。
  **排除点**：图上方的 Exclude 打开后，点一个点把它排除在拟合之外（再点一次恢复），或拖一个框排除
  框里的点；排除的点画成红色 ×，Curve 步骤写出“Points left out of the fit: N”和 Include All。
  排除按 q 记住（随曲线、项目保存），拟合记录里写 `left_out_q_nm^-1`。
- **Model**：每种颗粒一张卡片（类型、Distance D 开关、删除），每个参数一行：值、上次拟合的 ± 误差
  （或“到达边界”）、fit 勾选（自由 / 固定）；勾选 *Ranges* 后每行下面显示 min – max。尺寸分布
  宽度统一用相对值 σR/R、σh/h、σD/D。另有背景、分辨率峰 A/(1+(|q|/w)^ν)、因子 k 的卡片。
- **两种用法，两个入口**：看一条曲线用 Single analysis；一个序列或一批样品用 In-situ series。Curve 步骤在没有曲线时
  并排给出 Open Curve… 和 Fit a Folder of Curves…；拟合后 Results 有 Fit Every Curve in This Folder…（同一文件夹、
  同类文件名，每条曲线从这个模型开始，见 In-situ series）。批量的推荐流程：先在这里拟合一条有代表性的曲线（预测 → 精修），
  再拟合整个文件夹。
- **Fit**（`views/fit_step_view.py`，行为在 `single/run.py`）：一个 Method 下拉、一个按钮，只有三个方法（`METHODS`）：
  Predict parameters（唯一的网络方法：GIMaP 的建议网络 + 后验 + 最小二乘，`single/predict.py`、
  `application/single_predict.py`、`adapters/posterior_job.py`、`domain/posterior/refine.predict`）、Refine the values in Model
  （局部有界最小二乘）、Search the ranges, then refine（差分进化 + 三个起点精修）。fit1d 工具包（Predict parameters (fit1d)）
  与快速物理拟合（Find the particle shape (no AI)）已在 0.2.0 删除。打开曲线时页面替它选（`single/method_hint.py`）：
  未拟合的 GISAXS 线切选 Predict parameters，下面一行写 “Recommended for this curve: a GISAXS cut not fitted yet.”；
  其余曲线保持原来的选择（GIWAXS I(q) 从不推荐）；网络缺失时不推荐，选了 Predict 时下面一行写 `predict.MISSING`（缺少的文件名）。
  预测之后自动换成 Refine。Predict 下面只多一行 Particles（`single_predict.PARTICLES`：The particles in Model /
  Spheres and upright cylinders (1–2) / Any particles (1–3)；Model 里没有颗粒时按第二项）。Predict 只有一条路径，
  没有“解的得出方式”选项。进度条与 Stop：精修 / 范围搜索保留目前最好的值，停止 Predict 不产生新结果。
  折叠的 *Options* 只有 Evaluations at most (Refine, Search) 和 1D Predict Batch Window…。
- **Results**（`views/fit_results_view.py`，`single/result_actions.py`）：自上而下——质量（预测之后：几个解、最可能的已在
  Model，Refine 给出质量和误差）；下一步按钮 Refine These Values（预测之后）、Fit Again（同一方法、用 Model 现在的值）、
  Fit Every Curve in This Folder…；Fit details… 浮窗里是边界与强相关（只列前两对，其余悬停），预测之后是 Model 中那个解的
  精修值及其区间、χ²ᵣ 的点数与标记（`single/predict.posterior_lines`，文字在 `single/predict_texts.py`，名字和句子在 `shared/fit_names.py`）；值的表格：选中行后 Fix / Let It Fit / Range…（右键同样），下次
  Fit Again 即用；折叠的 Other solutions（p 列为概率；Predict 和 Analyze 的解）；Save…；日志。
- **边界**（`domain/fit_edges.py`，`application/single_fit.fit_resolving`）：Model 的固定值和上下限是硬限制。Refine 与
  Search 之后停在范围边上的值只报告（“… stopped at a bound of its range: widen the range or fix it.”），不自动放宽范围，
  也不自动固定（`fit_edges.resolve(..., rounds=0)`；没有勾选框）。scale 为 0（不需要的项）和 k 不报告。Predict 不使用
  Model 的范围：它的值保持在网络先验的盒子里（`refine._box`），处于边界的值带标记 `at_bound:NAME`。
- **误差**：协方差 pinv(JᵀJ)·s²，J 用中心差分；差分步长与列的单位取每个参数自己的大小（为 0 时取它的
  范围），所以一个跑到 1e30 的分辨率峰振幅不会把 R 的步长变成 1e24、也不会把其他参数的误差压成 0。
- 每次拟合、选用解、载入模型都是一步撤销；模型、方法、选择和最近的曲线下次启动时恢复。
- 拟合属于开始时的那条曲线和那些点：拟合后改了两半、拟合范围或排除的点，Fit 与 Results 步骤标为
  “Changed after the fit: Fit again”，Solutions 的 χ² 也注明是之前那些点上的值；打开另一条曲线会停止正在进行的拟合，
  之后才结束的拟合不保留（并提示）。
- 保存是“全有或全无”：表格与 JSON 记录先写成 `.part` 再替换；写失败时原来的文件不变，并说明原因（例如文件只读）。
- `Ctrl+O` 在 Fitting 页打开曲线（Open Curve…），在其他页打开数据。
- 1D Predict 窗口（`view_binding.py` 的 `FittingViewBinding`）拟合 Single analysis 的曲线：从文件重新读取（保留 Analyze
  写的探测器列像素数），按页面选的两半取符号，按拟合范围（|q|）和排除的点取列。旧的单曲线页与旧的 In-situ 页
  （Recipe、Live Watch）原本仍在后台构造但不显示，2026-11-16 删除。
- 手动模型存在用户数据目录的 `model_parameters.json` 里（File ▸ Save / Load Labs Parameters… 带着它），每个组分每种形状一个块（`sphere`、`cylinder`、
  `vertical_cylinder`）；形状名不论怎么写（“Vertical Cylinder”、`vertical_cylinder`）都指同一个块
  （`model_parameters_store.shape_block_name`）。早先的版本只把名字转成小写，在 1D Predict 窗口里选中一个竖直圆柱的解之后，
  文件里会多出一个 `vertical cylinder` 块，而 `vertical_cylinder` 还是原来的数。这样的文件照常读取（两个块都在时以
  `vertical_cylinder` 为准，另一个只补它没有的值），读取不改文件；下次保存该形状的值时多余的块去掉。回归测试 `tests/test_release2_fitting_cylinder_key.py`。
- `config/model_parameters.json` 是这个存储的出厂模板：用户数据目录里没有 `model_parameters.json` 时复制过去，之后不再动它。
  它就是 Fitting 的起始模型（`single/session.py` 的 `starting_model()`，即 `fitting/domain/fit_model.py` 里 `new_component`
  和 `FitModel` 的默认值）按存储的单位写出来的样子，所以改了 `fit_model.py` 的默认值，也要改这个 JSON：
  `tests/test_release2_defaults_template.py` 在两者一致之前会失败。里面不放某次拟合留下的数。

模型与拟合引擎在 domain：`fit_model.py`（参数、相对 σ 与手动模型 σ 的换算、`evaluate` 与
`scattering_model` 同一公式和采样）、`fit_engine.py`（拟合、误差、相关、边界）；Analyze 结果或
1D Predict 的解经 `native_solution.py` 换成模型（与该解曲线的最大相对差写在状态里）。

手动拟合（精修、搜索）另有（《物理与方法》9.6、9.14–9.16 节）：
- `domain/model_library.py`：核壳球、Percus–Yevick 硬球、二维六方次晶（只给精修和搜索；预测参数和 `cut_model` 不变）；
  `fit_model.FAMILIES` 多了 `core_shell_sphere`，`Component.interference` 选干涉函数（`NETWORK_FAMILIES` 是网络认识的）。
- `domain/fit_links.py`（名称 `R_1`、安全的表达式求值、链接、不等式）和 `domain/linked_fit.py`（拟合里的链接与不等式；
  没有链接的模型不进入这里，数值与之前完全相同，预测参数的结果用 `tests/test_ml1d_posterior.py` 核对）。
- `domain/beam_tail.py`：尾项范围按曲线设定（`single_fit.tail_for_curve`，在打开曲线、开始精修或搜索、序列每帧时调用，
  `fit_resolving` 本身不改范围）和标记；`series_fit.flagged_fit`。
- `domain/fit_profile.py`、`application/fit_uncertainty.py`（相关矩阵、强相关对、记录）、`application/fit_report.py`（报告）。
- 界面：`single/parameter_row.py`（每个参数一行）、`single/model_links.py`（链接与约束卡片）、`single/tail_ranges.py`、
  `single/uncertainty.py` 与 `views/fit_uncertainty_view.py`（不确定度与报告窗口）。
- 测试：`tests/test_fit_links.py`、`test_fit_model_library.py`（对照 SasView 公式、Wertheim 解析解和 BornAgain）、
  `test_fit_uncertainty.py`、`test_fit_model_editor_links.py`。

## 主单曲线页的 Predict parameters

- **请求**（`application/single_predict.predict_request`）：曲线的点（q 为 nm⁻¹，σ 可缺）、Particles 给出的组成
  （`model`：Model 的 1–3 群按半径排序，每群的类型与是否带次晶；`spheres_cylinders` / `any`：网络在允许的组成里选头部最可能的
  三种）、Model 的角度给出的切线 q_z（αi、αf、λ 为 0 时 q_z = 0，有效界面）、Model 的 sizes 与 coupling。
- **路径**（`domain/posterior/refine.predict`，唯一的一条）：每种组成 1000 个网络样本按重要性加权；每种组成最重的 3 个模式
  用最小二乘精修；最好的两种组成都精修不到 χ²/N 2 以下时，从最重的 16 个样本出发在范围内搜索一次；与最好的解 BIC 相差 20
  以内的解（最多 4 个）做局部后验（每轮 400 个样本）；按 evidence 比较，局部抽样有效样本少于 10 时按 BIC 比较、区间取最小二乘误差
  （标记 `bic`）。无 σ 时按 ln I 围绕局部趋势的离散取相对 σ（至少 1 %）。
- **进程边界**：`presentation/single/predict.py` → JobRequest → `infrastructure/adapters/posterior_job.run_predict`。
  网络按记录 `flow_v1.json` 的 SHA-256 校验后才用；Stop 取消该 job，不产生新结果。解进 Results 的 Other solutions，
  最可能的进 Model。测试 `tests/test_fit_page_predict.py`、`tests/test_ml1d_posterior.py`。

## 从 Analyze 接收曲线

- **Send to Fitting**：Analyze 写出 `<stem>_fit_input.dat`（`q I sigma pixels` 四列，q 单位 Å⁻¹，
  首部带 `# observation:` JSON），切到 Fitting 并载入。Fit 按钮菜单选择送出的半边，对应 Curve 步骤的
  “切线的两半”：

  | Analyze 选择 | Fitting 的两半 |
  | --- | --- |
  | 两种颜色的 \|qy\|（默认） | Both halves on \|q\| |
  | 左右对称平均 | Mean of both halves |
  | qy < 0 | q < 0 half |
  | qy > 0 | q > 0 half |

  `# observation:` 里有 `"halves"`（Analyze 选的是平均或某一半时）。数据旁边写不了时 Analyze 询问另存的文件夹。
  GIWAXS 的 I(q) 打开时页面上方出现提示（`workspace.NOTE_TEXT`）：Fitting 的模型是颗粒形状因子，不描述衍射峰，不推荐
  方法；峰用 Analyze ▸ Batch Export ▸ Peaks of the ring regions。
- **Results ▸ Fit details ▸ Show in Fitting**：同上载入曲线，再把该解换成模型（`single_fit.model_from_solution`：自动分析的解
  来自 Predict parameters，`physics_backend` 为 `cut_model`，就是 Fitting 自己的模型，原样载入），状态里写出与该解曲线的最大相对差。

- **Send Series to Fitting…**：先用 Batch Export 对话框的拟合模式导出列表中所有帧（可 Sum N，GISAXS 选 Halves of the
  cut），写进 `<名称>_fitting/`，完成后打开 In-situ series，folder 指向 `<名称>_fitting/fit_input`，pattern 为
  `*_fit_input.dat`，曲线已列出；序列的两半取这些文件 `# observation:` 里的 `"halves"`。
- GISAXS 水平 cut 的 observation `source = native_detector_columns`：每个探测器列的平均计数、
  Poisson σ 与有效像素数。（0.2.0 删除的 Single RC specialist 曾据此恢复原 CBF 路径的计数约定。）

## 1D Predict 窗口

- 1D Predict 窗口只有一种方法：Predict parameters（`adapters/posterior_job.fit_and_write`；Parameters… 里没有 Fit method
  下拉）。Parameters…：Complete composition（最多 3 个框，各群都带次晶；全为 Auto / unused 时网络在全部组成中选头部最可能的三种）、
  q unit of files without one in their header、q sides、Cut qz (nm⁻¹)（`cut_qz`；不勾选为有效界面）、Relative σ added to counting σ
  （默认 0.10）、Absolute σ added to counting σ（只用于带探测器列像素数的当前切线；没有 σ 的曲线用 `curve_intake.relative_sigma`，
  即 Fit 页 Predict 的估计）、Save Settings。未注明单位的文件默认按 Å⁻¹ 读（与 Fit 页同一个默认值）；文件由 Fit 页的读取器
  `load_xy_any` 读入。行带 probability、posterior、warnings、ess；Stage 列为 Network + posterior。
- 旧设置：General V5、Single RC specialist、Physical fit (numerical) 与 fit1d 都已于 0.2.0 删除；保存的旧方法键
  （model、stable、experimental、fit1d）按 `application/workflow_v5.RETIRED_METHODS` 打开为 posterior，其选项（`RETIRED`）丢弃，
  fit1d 的 `fit1d_qz` 保留为 `cut_qz`。
- 单条曲线的网络建议在 Fit 步骤里（Predict parameters），它的解进 Results 的 Other solutions 表（p 列为概率），Model 中
  解的精修值及区间、χ²ᵣ 所依据的点数与提示写在 Fit details…（`single/predict.posterior_lines`、`single/predict_texts.py`）。
- `Fit ▸ Options ▸ 1D Predict Batch Window…`（菜单 Tools ▸ 1D Predict — Fit Many Curves…，Fitting 菜单里同名的一项也是它）打开 1D Predict 窗口：
  Add curves…（若干曲线文件）、Use current cut、Parameters…（组成、q 单位与两半、q_z、σ）、Fit curve、Cancel、
  Open results。每次运行写到 GIMaP 用户数据文件夹下的 `AI_Fitting_Output/v5_<日期>_<时间>_<微秒>/`。
  - **q 的单位**：文件头写明单位的文件按写明的单位读（Analyze 的 `_fit_input.dat` 首行 `(q in 1/A)`、
    `# columns: qy (1/A) …`）；设置 *q unit of files without one in their header* 只用于没写明的文件。识别在
    `infrastructure/adapters/curve_units.py`（`LocalCurveRepository.stated_q_unit` → `LoadCurve.stated_q_unit` →
    `FittingViewModel.stated_q_unit`，由 `view_binding.py` 交给窗口），规则 `application/curve_units.q_unit_read`（经
    `workflow_v5.q_unit_of_file`，与 Single analysis、原位序列同一条）由窗口（`presentation/workflow_v5_units.py`：按钮下一行、表格的 Curve / side 列）和 job（`run_workflow_job`）共用；
    `request.json`、`fitting_run_summary.json` 的 `q_units`，`batch_summary.json` 与候选行的 `q_unit_read` / `q_unit_from`
    记下每条曲线按什么单位读。同时写明两种单位的文件不拟合。测试 `tests/test_release3_predict_units.py`。

## In-situ series

用 Single analysis 里的模型拟合一个文件夹里的每条曲线（`application/series_fit.py`）。

- **探测器帧**（`app/series_frames.py`、`single/series_frames.py`、`application/series_frames.py`）：Files 填探测器文件的
  pattern（`*.cbf`、`*.tif`、`*.edf`、`*.nxs` …），或选一个没有曲线、只有探测器帧的文件夹（自动换成 `*.<后缀>`），就列出
  每一帧（NeXus / HDF5 源的每一帧，Lambda 模块组算一个源；Analyze 的 Sum N 把相邻帧合成一组）。每一帧按 Analyze 列出帧
  时的设置切线——与 Batch Export 同一个请求（仪器配置、αi、束心、掩膜与校正、坏像素、切线宽度，钉住时切在屏幕上那一帧的
  行）、同一个 `AnalyzeFrame`、同一个 `fit_input_curve` / `fit_input_observation`（两半取 Analyze 送去拟合时的选择），
  得到与 `<stem>_fit_input.dat` 相同的曲线（q 为 Å⁻¹）；只在内存里，数据旁边不写任何文件，每帧只切一次。Curves 步骤的卡片写出
  用的是哪一帧的设置（仪器、αi、Sum、钉住的行）；Analyze 里还没有设置好的帧时说明原因、Start 不开始；Take Analyze's Set-up
  重新采用 Analyze 现在的设置并重新切。记录 `listing.detector_frames`、`listing.cut_as_in_analyze`（Analyze 的设置记录
  `gimap-analyze-settings`、钉住的切线、两半、Sum、设置所在的帧、αi）。Watch、阶段、两半、逐帧预测都照常工作。
- **Curves**：Choose Folder…（或 Analyze ▸ Send Series to Fitting）；Files 的 pattern（默认
  `*_fit_input.dat`）、Also in subfolders；Frames 第一帧 – 最后一帧、每隔几帧（编号取文件名最后一个
  数字，自然顺序）；Watch for new curves：运行中每 2 秒看一次文件夹，大小和时间两次相同的新文件才拿来拟合，读过之后
  又变了的文件重新排队，文件夹读不了时等它回来（“The folder of the curves cannot be read: waiting for it …”）。
  列表里每帧前面标 ✓ / ✗ / ·。
- **q 的单位**（Curves 步骤，`single/series_units.py`）：每个文件按文件头写明的单位读，没写明的按 Single analysis 的
  *q unit of files without one in their header*（Start 时的值，整次运行不变）。Curves 步骤按单位和来源数出列出的曲线
  （“2 curves: q in Å⁻¹, as their header states” / “… no unit in their header (Single analysis ▸ Curve ▸ File)”），并点名
  同时写明两种单位的文件——它们不拟合，帧写出原因。探测器帧按 Å⁻¹ 切，不写这一行。
- **Halves of the cut**（Curves 步骤，`single/series_halves.py`）：每帧只拟合一条曲线——Mean of both halves / q > 0 half /
  q < 0 half（`SERIES_SIDES`），从不两半一起。值的来源依次是（`series_fit.series_side`）：这个序列里自己选的（或项目里存的）
  → 列出的第一个拟合输入 `# observation:` 里的 `"halves"`（Analyze 写的）→ Single analysis 的两半（不是 Both halves on
  |q| 时）→ 平均；不是自己选的时下面一行写出来源（“from the fit inputs written by Analyze” / “as Analyze sends cuts to Fitting”（直接读的探测器帧） / “as in Single analysis” /
  “as saved in the project”）。打开另一个文件夹时自己的选择作废，同一文件夹重新列出时保留。曲线只有一侧 q 时下拉
  不可用，它（和它的标签）的提示说 “These curves have q on one side only: there are no halves to choose.”。所选的一半没有点的帧失败，原因为
  “No points on the q > 0 half” / “No points on the q < 0 half”，不会改用另一半（`MissingHalf`，预览也这样画）。
  两半随项目保存，写进记录（`settings.side`、`side_from`），并写在 Start 的摘要里。
- **Start**：Single analysis 的模型、拟合范围和排除的点，以及上面的两半（摘要写在这里，Edit in Single
  Analysis 回去改）；每帧从上一帧的结果开始（默认，上一帧没收敛时退回 Single 的模型）或都从 Single
  的模型开始；方法 Refine（快）、Search the ranges, then refine（慢）或 **Predict each frame**。阶段还没找到就开始的
  运行不用阶段（“No stages for this run: it started before they were found.”），运行结束后再找。各方法都保留模型的固定值和上下限。
- **Predict each frame**（`single/series_predict.py`）：每帧一个 Predict parameters job（`predict_request(data, start,
  "model")`：组成取这一帧起始模型的 1–3 群；少于 1 群或多于 3 群时不开始，`NOT_PREDICT`），保留的解再在起始模型的固定值和
  范围下用 Refine 精修（`series_fit._fit_frame`），表格的值是精修后的值和 1σ 误差。**Prefer the solution that continues the
  previous frame**（默认勾选；起点为 Single 的模型、或新阶段的第一帧时不跟随）：在概率至少为最可能解的 `CLOSE_ENOUGH` = 0.2 倍的解里，
  取尺寸和间距（ln R、ln D 之差，按半径对应各群）与上一帧最接近的一个（`closest`）；明显更可能的解仍被采用。记录
  `settings.method` = `predict`、`settings.follow`。
- **运行**：曲线在界面线程读入，拟合在后台一帧一帧做；Pause / Stop；选中最后一帧时图跟着运行走，
  选了别的帧就停在那里。Start 之前点列表中的帧就能看到它的曲线和 Single 的模型（预览）；开始后各帧都按这次运行的设置画。
  运行期间不能打开项目（先 Stop）。
- **Results**：“拟合 N 帧 · 失败 · 未收敛”；每帧一行（#、χ²ᵣ、每个自由值 ± 1σ，列名与 Results 步骤
  相同，如 “1·Sphere R (nm)”，最后两列 Points 与 Halves；失败的帧整行写 “✗ 原因”）；点一行看那一帧；Open Frame in
  Single Analysis 把该帧与它的拟合模型送回单条分析；折叠的日志。下面的趋势图选一个值（或 χ²ᵣ），画成点和 ±1σ 误差棒；
  坐标轴标题保持英文（图的内容），Log I 切换后误差棒重画，主题切换后重画。图标题用序列名
  （`series_text.series_name`：`…/sampleA/gimap_analysis/fit_input` 为 “sampleA”，其下的 Batch Export 子文件夹为
  “sampleA · <子文件夹>”）。GIWAXS 的 I(q) 序列跑完后，页面上方的提示回来并说有几帧是差的拟合。
- **Save ▾**：Table of Every Frame…（CSV：frame、file、chi2_reduced、log_rmse、converged、points、halves_used、每个值与
  `_error`）旁边写同名 `.json` 记录（`"schema": "gimap_series_fit_v2"`：起始模型、`settings.side` 与 `side_from`、范围、
  排除的点、方法、`q_unit_of_files`（Start 时没写明单位的文件所用的设置）、各状态的帧数、失败的帧与原因、`poor_fit`
  （χ²ᵣ > 10 的帧）、`per_frame` 每帧的状态、原因、点数、所用的两半、`q_unit` 与 `q_unit_from`，以及 `listing`：帧范围、阶段、Watch 加进来的帧 `added_by_watch`）；
  Trend Plot…、Selected Frame's Plot…（PNG / SVG）。
- **未保存的结果**：有拟合到底、还没保存的帧时，再次 Start、列出别的曲线或打开项目之前都会问
  “The results of the last run are not saved. Save the table of every frame first?”（Save / Discard / Cancel；
  Cancel 什么都不改）。
- 文件夹、pattern、帧范围与选择下次启动时恢复，也随项目（.gimap）保存。
- **阶段与异常帧**（`series_fit.stages_of_curves`，内核 `shared/series_stages`）：列出曲线后在后台按 Single 的两半、范围、
  排除点比较全部曲线；Curves 步骤写出阶段与异常帧，帧列表按阶段着色；Leave out the odd frames（默认开）；Start 的第三个
  选项“上一帧的结果；每个新阶段从 Single 的模型开始”；趋势图按阶段着色。见 [compare.md](compare.md)。

## 固定值、硬范围和结果浏览（2026-11-15）

- `domain/fit_limits.py` 保留建议中的固定值和范围；`fit_engine` 使用按列缩放的 BVLS 求解有界强度项，并保留可行初始曲线作为最优结果候选。
- `fit_resolving` 只给出边界诊断，不修改模型范围。序列沿用同一规则。
- `single/equivalent_solutions.py` 用表格下拉框整套切换完整的解（Predict parameters 或 Analyze 的），过期后不可切换，导出记录选择。
- `single/plotting.py` 把各群粒子与全局项显示开关接入右键。`Fit details…` 非模态浮窗收纳详细诊断；预测之后方法换成 Refine。

## 验证

- 单元与界面：上面的 Related tests，以及 `tests/test_analyze_workspace.py` 中经真实窗口的
  Analyze → Fitting（单条与序列）端到端测试。
- 发布回放：从项目根目录运行 `python tools/check_stable_predict_workflow.py --mode all`：
  worker（固定 native fixture、文本 batch、取消）、UI（Analyze 打开真实 CBF → 拖到验证过的带 →
  Refine x by Symmetry → Send to Fitting → Single analysis 的拟合范围 → 1D Predict）。单帧 in-situ 那部分跑在已删除的
  旧 In-situ 页上，2026-11-16 随该页去掉。保存的几何只读入
  内存 profile，不修改用户设置。2026-09-28：Analyze 的 1148 列与旧探测器路径的固定样本逐列强度、
  像素数完全相同，q 差不到一列；学习分支运行后按结构残差转数值拟合，lnRMSE 0.166／0.148
  （旧记录 0.171／0.155）。这些回放不代表独立样品泛化验证。

## 手动验收清单

- [ ] Analyze 打开 CBF 后 Send to Fitting：切到 Fitting，Curve 步骤显示 `_fit_input.dat`、点数与
      q 范围；Send to Fitting ▾ 的四种半边选择对应 Curve 步骤的 Halves of the cut；
- [ ] Open Curve… 打开两列 / 三列 / 四列 `.dat` / `.txt` 曲线和按表头读的 `.csv`（命令行的 horizontal.csv、保存的
      fit.csv）；表头说不清时弹出 Choose Columns；坏文件或其他格式给出明确错误；
- [ ] 步骤栏 Curve、Model、Fit、Results 四步都能点开，每步写出状态；窄窗口下命令栏的 Fit 与 Save 不被截断；
- [ ] 三种方法（Predict parameters / Refine the values in Model / Search the ranges, then refine）在 Method 下拉里，
      都由同一个 Fit 按钮（Ctrl+Return）启动；Predict 下面只有 Particles，Options 里只有最多计算次数与 1D Predict 批量窗口；
      未拟合的 GISAXS 线切自动选 Predict 并写 Recommended；精修的 Stop 保留最优值，Predict 的 Stop 保留已完成采样的解（标为不完整）；
- [ ] 范围太窄的 R（例如 1–2 nm，真值 3 nm）：Predict 和 Refine 都保留用户范围，
      边界说明在拟合详情中；固定值不变；
- [ ] Results 选中一个值 → Fix → Fit Again：该值固定参与下一次拟合；Fit Every Curve in This Folder… 打开 In-situ series
      并列出同一文件夹的同类文件；
      Options ▸ 1D Predict Batch Window… 打开 1D Predict 窗口，概率与提示在 Stage 列的提示里；
- [ ] Predict parameters 的三种 Particles 都能运行；解进 Other solutions 并带 p，Fit details… 写出后验区间与标记；
      Model 里的角度为 0 时按有效界面，填了 αi、αf、λ 时按切线的 q_z；
- [ ] Halves of the cut 的四个选项（Mean of both halves / Both halves on |q| / q > 0 half / q < 0 half）切换后，
      Curve 步骤的点数和图上的点随之更新；Log q / Log I 切换正常；
- [ ] 拟合后 Model 里每个自由值显示 ± 误差；仍在边上的值 Results 用橙色说明；参数输入不静默舍入小值；
- [ ] Send Series to Fitting 导出所有帧后打开 In-situ series，folder（`<名称>_fitting/fit_input`）与 pattern 已填好、曲线已列出；
- [ ] Single 选 Both halves on |q| 后用 Predict parameters：Fit 时下拉换成 Mean of both halves，
      状态行说明原因；Refine 与 Search 仍拟合两半；
- [ ] In-situ series 的 Halves of the cut：Send Series 时选 q > 0 half，打开后下拉为 q > 0 half，下面写 “from the fit inputs
      written by Analyze”；只有一侧 q 的曲线时下拉不可用并说明；所选一半没有点的帧标 ✗，原因 “No points on the q > 0 half”；
- [ ] 结果表最后两列为 Points、Halves；保存的 CSV 有 points、halves_used，JSON 的 `schema` 为 `gimap_series_fit_v2`，
      `settings.side_from` 写出两半的来源；
- [ ] 有拟合完、未保存的帧时，再次 Start、换文件夹或打开项目会问 Save / Discard / Cancel；Cancel 不改变任何东西；
- [ ] GIWAXS I(q) 送到 Fitting：页面上方出现提示；该序列跑完后提示回来并说差的拟合有几帧；
- [ ] Start 按自然顺序拟合，坏文件记为失败并继续；Watch for new curves 只拟合新写完的曲线；
- [ ] 每帧的误差合理（不出现 ±1e-20 或空白），趋势图的误差棒与表一致；
- [ ] Exclude：点选 / 框选排除、再点恢复、Include All；排除的点不参与拟合，序列也用同样的排除；
- [ ] Single / In-situ 来回切换不重置曲线、模型或结果表；
- [ ] Save ▾ ▸ Data and Fit… 写出 CSV（q 的 nm⁻¹ 与 Å⁻¹、I、σ、模型、残差、各项）和同名 JSON 记录；
- [ ] Save ▾ ▸ Plot… 写出的 PNG 或 SVG 与屏幕上的图层一致；Save ▾ ▸ Model… 写出的 JSON 能用 Load Model… 读回；
- [ ] 切换浅色／深色主题时曲线图背景、坐标轴与图例同时变化，导出图不受主题影响。
