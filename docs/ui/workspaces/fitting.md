# Fitting 界面与交互说明

- **Status**: Current
- **Scope**: Fitting workspace（1D 曲线拟合：单条曲线与 in-situ 曲线序列）的布局、控件映射与手动验收
- **Related code**:
  [`src/gimap/features/fitting/presentation/`](../../../src/gimap/features/fitting/presentation/)、
  [`src/gimap/app/main_window.py`](../../../src/gimap/app/main_window.py)
- **Related tests**:
  [`tests/test_fit_page.py`](../../../tests/test_fit_page.py)、
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

- **Curve**：文件、点数、q 范围（nm⁻¹ 与 Å⁻¹）、有无 σ；切线两半（Mean of both halves / Both halves on |q| / q > 0 half /
  q < 0 half，`single_fit.SIDES`）。平均（`shared/signed_cut.py`）只在另一半两侧 2.5 个步长内都有测量点处求平均，其余处
  （另一半的缺口、另一半到头之后）用这一半原样，不跨缺口插值。Find the particle shape、AI proposal、fit1d 只能拟合一条曲线：
  选着 Both halves on |q| 时 Fit 先把下拉换成平均，并说 “Both halves on |q| cannot be fitted by this method: the mean of
  both halves is used.”；Refine 与 Search 照旧拟合两半；
  拟合范围（nm⁻¹，或拖动图上的橙色区域；Whole Curve）；折叠的 *File* 里选文件的 q 单位。
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
- **Fit**（`views/fit_step_view.py`，行为在 `single/run.py`）：一个 Method 下拉、一个按钮。方法：Predict parameters
  (fit1d)、Refine the values in Model（局部有界最小二乘）、Search the ranges, then refine（差分进化 + 三个起点精修）、
  Find the particle shape (no AI)、AI proposal (1D Predict)。打开曲线时页面替它选好：未拟合的 GISAXS 线切选 Predict 并在
  下面一行写 “Recommended for this curve”；其余曲线保持原来的选择（GIWAXS I(q) 从不选粒子搜索）；Predict 或寻找形状
  之后自动换成 Refine。Predict 下面只多一行 Particles（尝试组成 / 模型中的粒子），寻找形状下面一行 Families。
  **Resolve values at the edge of their range automatically**（默认勾选，记住；In-situ series 跟随它）：见下面“边界”。
  进度条与 Stop：精修 / 范围搜索保留目前最好的值，取消寻找解或 fit1d 不产生新结果。不常用的都在折叠的 *Options*：
  fit1d 的网络、搜索、曲线类型、qz、系统误差，最多计算次数，1D Predict 批量窗口。
- **Results**（`views/fit_results_view.py`，`single/result_actions.py`）：自上而下——质量（fit1d 时第二行说明 Model 只是
  起点）；下一步按钮 Refine These Values（预测或寻找形状之后）、Fit Again（同一方法、用 Model 现在的值）、Fit Every Curve
  in This Folder…；绿色 “Done automatically” 列出拟合自己处理的边界；橙色只写真正要知道的（kit 的提示、仍在边上的值、
  强相关——只列前两对，其余悬停）；值的表格：选中行后 Fix / Let It Fit / Range…（右键同样），下次 Fit Again 即用；
  折叠的 Other solutions / Other compositions 与 Constraints and ranges；Save…；日志。
- **边界**（`domain/fit_edges.py`、`domain/fit1d_edges.py`）：拟合后停在范围边上的值由拟合自己处理。先放宽，最多 2 轮
  （尺寸、距离、分辨率峰宽 ×2 / ÷2；默认范围和用户范围一样放宽；用户固定的值从不动；改动后 χ² 不比之前差 2 % 以上才保留，
  否则保留原拟合并说明）。放宽后仍在边上的值（或分布宽度 σR/R、w/D 停在下限、D = 2R 这类不能放宽的边）说明曲线朝那个方向
  是平的，只给出一个界：拟合把它向内一步步移动（尺寸 ÷/× 1.25，相对宽度 ∓ 0.05），固定它、其余重新拟合，只要总 χ² 与最佳
  相差不超过 1σ 容差 max(1, χ²/dof) 就继续；报告为其中**最不极端**的值，标成单侧界（“≥ 0.549”，状态 one-sided bound，
  约束来源 `bound_upper` / `bound_lower`；精修时该值在 Model 里固定）。连一步都明显变差、而边又是模型自身的极限（如 σR/R
  0.6）时，说明曲线想要超出模型的值、模型可能缺了什么。scale 为 0（不需要的项）和 k 不处理。全部写在 Results 的
  “Done automatically”、`native_result.fit1d_bounds.actions` 和序列记录 `per_frame[].resolved`。
- **误差**：协方差 pinv(JᵀJ)·s²，J 用中心差分；差分步长与列的单位取每个参数自己的大小（为 0 时取它的
  范围），所以一个跑到 1e30 的分辨率峰振幅不会把 R 的步长变成 1e24、也不会把其他参数的误差压成 0。
- 每次拟合、选用解、载入模型都是一步撤销；模型、方法、选择和最近的曲线下次启动时恢复。
- 拟合属于开始时的那条曲线和那些点：拟合后改了两半、拟合范围或排除的点，Fit 与 Results 步骤标为
  “Changed after the fit: Fit again”，Solutions 的 χ² 也注明是之前那些点上的值；打开另一条曲线会停止正在进行的拟合，
  之后才结束的拟合不保留（并提示）。
- 保存是“全有或全无”：表格与 JSON 记录先写成 `.part` 再替换；写失败时原来的文件不变，并说明原因（例如文件只读）。
- `Ctrl+O` 在 Fitting 页打开曲线（Open Curve…），在其他页打开数据。
- 旧的单曲线页仍在后台（不显示）：它的 binding 运行 1D Predict 窗口，并跟随新页面的曲线与模型，见
  `workspace.attach_legacy`。旧的 In-situ 页（Recipe、Live Watch）不再显示，由下面的新页面代替。
- 旧页的模型存在用户数据目录的 `model_parameters.json` 里，每个组分每种形状一个块（`sphere`、`cylinder`、
  `vertical_cylinder`）；形状名不论怎么写（“Vertical Cylinder”、`vertical_cylinder`）都指同一个块
  （`model_parameters_store.shape_block_name`）。早先的版本只把名字转成小写，在 1D Predict 窗口里选中一个竖直圆柱的解之后，
  文件里会多出一个 `vertical cylinder` 块，而 `vertical_cylinder` 还是原来的数。这样的文件照常读取（两个块都在时以
  `vertical_cylinder` 为准，另一个只补它没有的值），读取不改文件；下次保存该形状的值时多余的块去掉。旧页按形状下拉框里
  选中的条目读形状，不读框里显示的文字（中文界面下那些文字是翻译过的）。回归测试 `tests/test_release2_fitting_cylinder_key.py`。
- `config/model_parameters.json` 是这个存储的出厂模板：用户数据目录里没有 `model_parameters.json` 时复制过去，之后不再动它。
  它就是 Fitting 的起始模型（`single/session.py` 的 `starting_model()`，即 `fitting/domain/fit_model.py` 里 `new_component`
  和 `FitModel` 的默认值）按存储的单位写出来的样子，所以改了 `fit_model.py` 的默认值，也要改这个 JSON：
  `tests/test_release2_defaults_template.py` 在两者一致之前会失败。里面不放某次拟合留下的数。

模型与拟合引擎在 domain：`fit_model.py`（参数、相对 σ 与手动模型 σ 的换算、`evaluate` 与
`scattering_model` 同一公式和采样）、`fit_engine.py`（拟合、误差、相关、边界）；Analyze 结果或
1D Predict 的解经 `native_solution.py` 换成模型（与该解曲线的最大相对差写在状态里）。

## 主单曲线页的 fit1d

- **Predict**：Method 选 Predict parameters (fit1d)。**Particles**（`fit1d_composition`）：`auto:sphere,vc:3`、
  `auto:vc:3` = 工具包拟合这些形状的每种组成并按 BIC 排序，最好的一种再按“边界”处理；`model` = Model 中 1–4 群
  Sphere / Vertical cylinder（允许同类重复），带上 Model 的约束。未编辑的 Model 打开 GISAXS 线切时默认 `auto:sphere,vc:3`。
- **其他组成**（只在 `auto:`）：Results 的折叠 Other compositions 列出至多 6 种（χ²/点、ΔBIC、“与 … 分不开”、标记），
  选中一行显示每个参数的值 ± 误差、等价解范围和状态；Use This Composition 以它的值为起点、用这个组成重新预测（边界照样
  自动处理），列表保留。
- 网络：general = `cond_v2_ft4`，au_np_si = `cond_v2_ftr1`。主单页 search 默认 **Auto**，另有 Fast / Robust；
  curve type 为 GISAXS / SAXS，systematic error 为相对误差下限。Yoneda 切线的真空 qz（nm⁻¹）需手填，
  不自动从 Analyze 带入；不勾选按 0 计算，含球时给提示。SAXS 忽略 qz、关闭结构因子和低 q 项。
- **约束**（`domain/fit1d_bounds.py`）：组成为 `model` 时，Model 中固定的值和改过的范围（与默认范围不同的）
  作为工具包的 `{'fixed'}` / `{'range'}` 约束传入：R、D 用 nm，σR/R 原样，σD/D 换成 w/D = σD/D·√(2π)，固定的
  background 也传；群按工具包编号（先球后竖直圆柱，各按 Model 顺序）。默认范围留给工具包，由曲线的 q 范围决定
  （R ∈ [0.5/q_max, 5/q_min]、D ∈ [2R, 40R]）。范围超出曲线能分辨的 R / D 时拟合前提示。Constraints and ranges 与导出的
  JSON（`native_result.fit1d_constraints`：`given`、`used`、来源 model / widened / limit）写出用到的每个约束；
  `native_result.fit1d_bounds`：`measurable`、`actions`（自动做了什么）、`edges`（仍在边上的）、`refits`。
  v2 为数目加权截断高斯尺寸分布（相对宽度）和 Hosemann 准晶 w = w_rel·D，q 用 nm⁻¹、长度 nm；
  球用 |q| = √(q∥² + qz²)，竖直圆柱用 q∥。手动模型的正演与宽度约定不同，映射只作为后续手动精修的起点。
- **原结果保留**：图上的总曲线、Terms 各群分量、残差、Results 的 χ²/点及参数误差 / 状态来自原 v2 结果。
  σ 使用文件误差与系统误差下限的平方和；无 σ 时明确提示 Poisson 假设。状态表与悬停范围说明等价解、弱约束、
  触边界和无贡献的群；全部提示保留。Save Data and Fit 的 CSV / JSON 保存原曲线、分量、effective σ、参数和提示。
- **过期结果**：改 Model、选区或 fit1d 设置后，旧表和 χ² 标为 previous fit1d；当前图标为手动初始模型，
  不再宣称显示原 v2 曲线，禁止把过期结果作为当前 fit 导出。恢复相同条件可恢复有效状态，或重新拟合。
- **进程边界**：`presentation/single/fit1d.py` → `application/single_fit1d.py` 的 JobRequest →
  `infrastructure/adapters/single_fit1d.py` → 工具包。只在 JobRunner 子进程导入研究代码，Stop 终止该 job；
  不依赖 V5 / TensorFlow 资产。工具包位置为 `utils/gisaxs1d_gui_kit` 或 `GISAXS1D_KIT_HOME`。
  回归测试为 `tests/test_fit_page_fit1d.py`。

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
  GIWAXS 的 I(q) 打开时页面上方出现提示（`workspace.NOTE_TEXT`）：Fitting 的模型是颗粒形状因子，不描述衍射峰，不自动
  做形状搜索；峰用 Analyze ▸ Batch Export ▸ Peaks of the ring regions。
- **Results ▸ Fit details ▸ Show in Fitting**：同上载入曲线，再把该解换成模型（`native_solution.py`：
  尺寸与相对 σ 换算、各 scale 在该解自己的曲线上求解），状态里写出与该解曲线的最大相对差。

- **Send Series to Fitting…**：先用 Batch Export 对话框的拟合模式导出列表中所有帧（可 Sum N，GISAXS 选 Halves of the
  cut），写进 `<名称>_fitting/`，完成后打开 In-situ series，folder 指向 `<名称>_fitting/fit_input`，pattern 为
  `*_fit_input.dat`，曲线已列出；序列的两半取这些文件 `# observation:` 里的 `"halves"`。
- GISAXS 水平 cut 的 observation `source = native_detector_columns`：每个探测器列的平均计数、
  Poisson σ 与有效像素数。V5 据此恢复与原 CBF 路径相同的 counting contract
  （`native_cbf_columns`、`valid_pixel_counts`、working tolerance），所以 stable 学习分支的
  适用条件和结果与旧路径一致；缺少计数的普通文本曲线会走数值回退并注明原因。
- 学习分支（Single RC specialist）的验证范围：每侧 450–700 个原生列、q 覆盖到 4.0–4.3 nm⁻¹、
  每列 3–12 个有效像素、单帧（不相加）。Analyze 默认切满探测器宽度、5 行带，对 P03 这类数据要
  得到同样的观测：在 Analyze 把水平带拖到 6 行且避开模块缝隙（例如 1171–1176），在 Fitting 把
  fitting range 设为 |q| ≤ 4.23 nm⁻¹（0.423 Å⁻¹）。不满足时结果照常给出，Stage 显示数值回退及原因。

## Fit 方法（1D Predict）

- 1D Predict 窗口的 `Fit curve` 按 Parameters… 里保存的 Fit method 执行，默认 **General V5 (experimental)**；**Single RC
  specialist (experimental)** 只在显式选择一个 Random cylinder（`components=[2]`）且曲线带原生
  计数时使用局部网络，否则数值回退；第三种 **Physical fit (numerical)** 不用网络。Stage 列显示 Neural model / Model + amplitude /
  Numerical fallback，悬停可见原因；不显示伪概率。
- 第四种 **fit1d (network + physical fit)** 调用外部的 gisaxs1d 工具包（放在 `utils/gisaxs1d_gui_kit`，或用环境变量
  `GISAXS1D_KIT_HOME` 指定；只需 numpy、scipy）。找不到工具包时，任务以一句说明放在哪里的消息失败。
  - 组分必须由用户在 Complete composition 里给定（Sphere=1、Vertical cylinder=3；没有 Random cylinder）：
    一条曲线识别不了组分，空着或含 2 时 `validate_options` 报错。不能固定 σ res（可以固定 ν res）。
  - Parameters… 里 fit1d 的设置：search（`fit1d_policy`：fast / auto / robust）、curve type（`fit1d_mode`：
    gisaxs / saxs）、network（`fit1d_model`：general / au_np_si）、Cut qz（`fit1d_qz`，nm⁻¹；不勾选时为 0，
    含球时行的 warnings 里注明）、systematic error（`fit1d_sigma_floor`，相对值）。
  - 每个 q 半边一行（rank 1）。物理是工具包的 v2：数目加权的尺寸分布、Hosemann 准晶 w = w_rel·D、球按
    |q| = √(q∥² + qz²)。`native_fit` / `display_fit` 是工具包自己在测量点上的曲线；`params.sigma_D` 是
    `legacy_sigma_D = w_rel/√(2π)`（在手动模型里给出相同准晶阻尼的值），工具包自己的值、1σ 误差、等价解范围和
    每个参数的状态在 `components[i].fit1d` 与行的 `fit1d`。多解 / 触边界 / 弱约束 / 模型未在误差内描述曲线都写进
    `warnings`（Stage 列悬停可见）。选中 fit1d 行后保留原 v2 总曲线与分量；复制到 Components 的模型只是
    手动精修初值，不用它重画替代原结果。批量窗口默认 fast，主单页默认 auto。
  - 代码：`infrastructure/adapters/fit1d_method.py`（只在 job 进程里导入工具包）；测试 `tests/test_fit1d_method.py`
    （没有工具包时跳过拟合部分）。
- 单条曲线的 AI 建议在新页面 Fit 步骤里（AI proposal），它的解进 Results 的 Solutions 表。
- `Fit ▸ Options ▸ 1D Predict Batch Window…`（菜单 Tools ▸ 1D Predict — Fit Many Curves…）打开 1D Predict 窗口：
  Add curves…（若干曲线文件）、Use current cut、Parameters…（方法、组分、q 单位与两半）、Fit curve、Cancel、
  Open results。每次运行写到启动 GIMaP 的文件夹下的 `AI_Fitting_Output/v5_<日期>_<时间>/`。

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
- **Halves of the cut**（Curves 步骤，`single/series_halves.py`）：每帧只拟合一条曲线——Mean of both halves / q > 0 half /
  q < 0 half（`SERIES_SIDES`），从不两半一起。值的来源依次是（`series_fit.series_side`）：这个序列里自己选的（或项目里存的）
  → 列出的第一个拟合输入 `# observation:` 里的 `"halves"`（Analyze 写的）→ Single analysis 的两半（不是 Both halves on
  |q| 时）→ 平均；不是自己选的时下面一行写出来源（“from the fit inputs written by Analyze” / “as Analyze sends cuts to Fitting”（直接读的探测器帧） / “as in Single analysis” /
  “as saved in the project”）。打开另一个文件夹时自己的选择作废，同一文件夹重新列出时保留。曲线只有一侧 q 时下拉
  不可用并说 “These curves have q on one side only: there are no halves to choose.”。所选的一半没有点的帧失败，原因为
  “No points on the q > 0 half” / “No points on the q < 0 half”，不会改用另一半（`MissingHalf`，预览也这样画）。
  两半随项目保存，写进记录（`settings.side`、`side_from`），并写在 Start 的摘要里。
- **Start**：Single analysis 的模型、拟合范围和排除的点，以及上面的两半（摘要写在这里，Edit in Single
  Analysis 回去改）；每帧从上一帧的结果开始（默认，上一帧没收敛时退回 Single 的模型）或都从 Single
  的模型开始；方法 Refine（快）、Search the ranges, then refine（慢）或 **Predict each frame (fit1d)**。阶段还没找到就开始的
  运行不用阶段（“No stages for this run: it started before they were found.”），运行结束后再找。边界的处理跟随 Single 的
  Resolve… 勾选。
- **Predict each frame (fit1d)**（`single/series_fit1d.py`，`domain/fit1d_follow.py`）：每帧一个 fit1d job（粒子、固定值和
  范围用 Single 的模型，设置用 Single 的 Fit 步骤，边界同样自动处理），表格的值是 fit1d 的值和 1σ 误差（σD/D = w/D/√(2π)），
  帧图画 fit1d 自己的 v2 曲线。**Prefer the solution that continues the previous frame**（默认勾选；起点为 Single 的模型、或新阶段
  的第一帧时不跟随）：每帧再拟合一次，所有参数限制在上一帧值附近的窗口里（R、D × / ÷ 1.5，相对宽度 ± 0.15；用户固定的值不变，
  用户范围取交集；值停在窗口边上时窗口照样放宽），若它的 χ² 不比自由拟合差过工具包跟踪器的同盆容差
  max(25, 0.02 N)·max(1, χ²/N)，就采用它（等价解中最平滑的序列），否则采用自由拟合并写明“样品在这里变了”。工具包自己的
  series1d 跟踪器只处理探测器帧，曲线文件夹在工具包里是逐条独立拟合，所以这条规则在这里实现。记录：`settings.method`
  = `fit1d`、`settings.follow`、`settings.fit1d`，每帧 `followed_previous` 与 `resolved`。
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
  排除的点、方法、`q_unit_of_files`、各状态的帧数、失败的帧与原因、`poor_fit`（χ²ᵣ > 10 的帧）、`per_frame` 每帧的
  状态、原因、点数、所用的两半，以及 `listing`：帧范围、阶段、Watch 加进来的帧 `added_by_watch`）；
  Trend Plot…、Selected Frame's Plot…（PNG / SVG）。
- **未保存的结果**：有拟合到底、还没保存的帧时，再次 Start、列出别的曲线或打开项目之前都会问
  “The results of the last run are not saved. Save the table of every frame first?”（Save / Discard / Cancel；
  Cancel 什么都不改）。
- 文件夹、pattern、帧范围与选择下次启动时恢复，也随项目（.gimap）保存。
- **阶段与异常帧**（`series_fit.stages_of_curves`，内核 `shared/series_stages`）：列出曲线后在后台按 Single 的两半、范围、
  排除点比较全部曲线；Curves 步骤写出阶段与异常帧，帧列表按阶段着色；Leave out the odd frames（默认开）；Start 的第三个
  选项“上一帧的结果；每个新阶段从 Single 的模型开始”；趋势图按阶段着色。见 [compare.md](compare.md)。

## 验证

- 单元与界面：上面的 Related tests，以及 `tests/test_analyze_workspace.py` 中经真实窗口的
  Analyze → Fitting（单条与序列）端到端测试。
- 发布回放：从项目根目录运行 `python tools/check_stable_predict_workflow.py --mode all`：
  worker（固定 native fixture、文本 batch、取消）、UI（Analyze 打开真实 CBF → 拖到验证过的带 →
  Refine x by Symmetry → Send to Fitting → fitting range → 1D Predict）与单帧 in-situ。保存的几何只读入
  内存 profile，不修改用户设置。2026-09-28：Analyze 的 1148 列与旧探测器路径的固定样本逐列强度、
  像素数完全相同，q 差不到一列；学习分支运行后按结构残差转数值拟合，lnRMSE 0.166／0.148
  （旧记录 0.171／0.155）。这些回放不代表独立样品泛化验证。

## 手动验收清单

- [ ] Analyze 打开 CBF 后 Send to Fitting：切到 Fitting，Curve 步骤显示 `_fit_input.dat`、点数与
      q 范围；Send to Fitting ▾ 的四种半边选择对应 Curve 步骤的 Halves of the cut；
- [ ] Open Curve… 打开两列 / 三列 / 四列 `.dat` / `.txt` 曲线；坏文件或其他格式（包括 `.csv`）给出明确错误；
- [ ] 步骤栏 Curve、Model、Fit、Results 四步都能点开，每步写出状态；窄窗口下命令栏的 Fit 与 Save 不被截断；
- [ ] 五种方法（Predict parameters (fit1d) / Refine the values in Model / Search the ranges, then refine /
      Find the particle shape (no AI) / AI proposal (1D Predict)）在 Method 下拉里，都由同一个 Fit 按钮（Ctrl+Return）启动；
      未拟合的 GISAXS 线切自动选 Predict 并写 Recommended；精修的 Stop 保留最优值，fit1d 的 Stop 不产生新结果；
- [ ] 范围太窄的 R（例如 1–2 nm，真值 3 nm）：Predict 和 Refine 都自己放宽并重拟合，Results 的 “Done automatically”
      写出放宽前后的边和 χ²；不用点任何按钮；取消勾选 Resolve… 后不再处理，值停在边上并有提示；
- [ ] Results 选中一个值 → Fix → Fit Again：该值固定参与下一次拟合；Fit Every Curve in This Folder… 打开 In-situ series
      并列出同一文件夹的同类文件；
      Options ▸ 1D Predict Batch Window… 打开 1D Predict 窗口，回退原因在 Stage 列的提示里；
- [ ] fit1d 允许 1–4 群球 / 竖直圆柱、两网络和 GISAXS / SAXS；qz 手填、σ 说明准确，Terms 能显示原 v2 分量；
      Results 状态及 CSV / JSON 保留原结果；改 Model / 选区 / qz / mode / network / search / 误差下限后标记过期并拒绝导出；
- [ ] Halves of the cut 的四个选项（Mean of both halves / Both halves on |q| / q > 0 half / q < 0 half）切换后，
      Curve 步骤的点数和图上的点随之更新；Log q / Log I 切换正常；
- [ ] 拟合后 Model 里每个自由值显示 ± 误差；仍在边上的值 Results 用橙色说明；参数输入不静默舍入小值；
- [ ] Send Series to Fitting 导出所有帧后打开 In-situ series，folder（`<名称>_fitting/fit_input`）与 pattern 已填好、曲线已列出；
- [ ] Single 选 Both halves on |q| 后用 Find the particle shape / AI proposal / fit1d：Fit 时下拉换成 Mean of both halves，
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
