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
  [`tests/test_fitting_presentation.py`](../../../tests/test_fitting_presentation.py)、
  [`tests/test_fitting_insitu_series.py`](../../../tests/test_fitting_insitu_series.py)、
  [`tests/test_analyze_fit_input_contract.py`](../../../tests/test_analyze_fit_input_contract.py)、
  [`tests/test_ui_workspace_layouts.py`](../../../tests/test_ui_workspace_layouts.py)
- **Last verified**: 2026-10-01

## 分工：Analyze 出曲线，Fitting 拟合曲线

原 “Cut & Fitting” 页的探测器部分（图像预览、center / Yoneda、cut 选区、mask / gap fill /
threshold / flip、stack 与 q 轴设置）已全部下线，由 [Analyze](analyze.md) 承担：

| 原 Cut & Fitting 功能 | 现在的位置 |
| --- | --- |
| 打开 CBF / NXS / TIFF、上一张 / 下一张 | Analyze 文件列表 |
| detector 几何、beam centre | Analyze 的 instrument profile 与 Beam centre 菜单 |
| Find Yoneda & Set Cut、cut 厚度 | Analyze 自动水平 cut（可拖动 / 手动行区间） |
| Optimize Center X | Analyze ▸ Beam centre ▸ Refine x by Symmetry |
| 负像素 / 坏点 mask 与 3 px guard | Analyze 选项 ▸ Gap guard（默认 3 px，可保存） |
| Stack 多帧 | Analyze 选项 ▸ Frames ▸ Sum N frames |
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

- **Curve**：文件、点数、q 范围（nm⁻¹ 与 Å⁻¹）、有无 σ；切线两半（平均 / |q| 上两半 / 一半）；
  拟合范围（nm⁻¹，或拖动图上的橙色区域；Whole Curve）；折叠的 *File* 里选文件的 q 单位。
  **排除点**：图上方的 Exclude 打开后，点一个点把它排除在拟合之外（再点一次恢复），或拖一个框排除
  框里的点；排除的点画成红色 ×，Curve 步骤写出“Points left out of the fit: N”和 Include All。
  排除按 q 记住（随曲线、项目保存），拟合记录里写 `left_out_q_nm^-1`。
- **Model**：每种颗粒一张卡片（类型、Distance D 开关、删除），每个参数一行：值、上次拟合的 ± 误差
  （或“到达边界”）、fit 勾选（自由 / 固定）；勾选 *Ranges* 后每行下面显示 min – max。尺寸分布
  宽度统一用相对值 σR/R、σh/h、σD/D。另有背景、分辨率峰 A/(1+(|q|/w)^ν)、因子 k 的卡片。
- **Fit**：一个按钮、四种方法——精修当前值（局部，有界最小二乘）、在范围内搜索再精修（差分进化 +
  三个起点精修）、寻找颗粒形状（不用 AI，与 Analyze 自动分析同一数值拟合）、AI 建议（1D Predict）。
  进度条与 Stop（保留目前最好的值）；各组分的 scale、背景、峰 A 每步用非负最小二乘精确求解。
  *Advanced*：最多计算次数、Fit Many Curves…（1D Predict 批量窗口）。
- **Results**：χ²ᵣ（有 σ 时）、点数、自由参数、是否收敛；到达边界、强相关（|ρ| ≥ 0.95）、χ²ᵣ 远大于 1
  的警告；每个值 ± 误差；Solutions 表（Use This Solution）；Save Data and Fit…（CSV + JSON 记录）、
  Save Plot…、Save Model…；折叠的日志。形状搜索或 AI 建议之后还没有精修时，这里说明“最好的解已在
  Model 里，Fit（Refine）给出它的质量和误差”，而不是“还没有拟合”。
- **误差**：协方差 pinv(JᵀJ)·s²，J 用中心差分；差分步长与列的单位取每个参数自己的大小（为 0 时取它的
  范围），所以一个跑到 1e30 的分辨率峰振幅不会把 R 的步长变成 1e24、也不会把其他参数的误差压成 0。
- 每次拟合、选用解、载入模型都是一步撤销；模型、方法、选择和最近的曲线下次启动时恢复。
- 旧的单曲线页仍在后台（不显示）：它的 binding 运行 1D Predict 窗口，并跟随新页面的曲线与模型，见
  `workspace.attach_legacy`。旧的 In-situ 页（Recipe、Live Watch）不再显示，由下面的新页面代替。

模型与拟合引擎在 domain：`fit_model.py`（参数、相对 σ 与手动模型 σ 的换算、`evaluate` 与
`scattering_model` 同一公式和采样）、`fit_engine.py`（拟合、误差、相关、边界）；Analyze 结果或
1D Predict 的解经 `native_solution.py` 换成模型（与该解曲线的最大相对差写在状态里）。

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

- **Results ▸ Fit details ▸ Show in Fitting**：同上载入曲线，再把该解换成模型（`native_solution.py`：
  尺寸与相对 σ 换算、各 scale 在该解自己的曲线上求解），状态里写出与该解曲线的最大相对差。

- **Send Series to Fitting…**：先按 Export All 的对话框导出列表中所有帧（可 Sum N），完成后
  打开 In-situ series，folder 指向导出目录，pattern 为 `*_fit_input.dat`，曲线已列出。
- GISAXS 水平 cut 的 observation `source = native_detector_columns`：每个探测器列的平均计数、
  Poisson σ 与有效像素数。V5 据此恢复与原 CBF 路径相同的 counting contract
  （`native_cbf_columns`、`valid_pixel_counts`、working tolerance），所以 stable 学习分支的
  适用条件和结果与旧路径一致；缺少计数的普通文本曲线会走数值回退并注明原因。
- 学习分支（Single RC specialist）的验证范围：每侧 450–700 个原生列、q 覆盖到 4.0–4.3 nm⁻¹、
  每列 3–12 个有效像素、单帧（不相加）。Analyze 默认切满探测器宽度、5 行带，对 P03 这类数据要
  得到同样的观测：在 Analyze 把水平带拖到 6 行且避开模块缝隙（例如 1171–1176），在 Fitting 把
  fitting range 设为 |q| ≤ 4.23 nm⁻¹（0.423 Å⁻¹）。不满足时结果照常给出，Stage 显示数值回退及原因。

## Fit 方法（1D Predict）

- `Fit curve` 按保存的 Fit method 执行，默认 **General V5 (experimental)**；**Single RC
  specialist (experimental)** 只在显式选择一个 Random cylinder（`components=[2]`）且曲线带原生
  计数时使用局部网络，否则数值回退。Stage 列显示 Neural model / Model + amplitude /
  Numerical fallback，悬停可见原因；不显示伪概率。
- 单条曲线的 AI 建议在新页面 Fit 步骤里（AI proposal），它的解进 Results 的 Solutions 表。
- `Fit ▸ Advanced ▸ Fit Many Curves…`（菜单 Tools ▸ Fit Settings & Batch…）打开 1D Predict 窗口：方法、
  候选、参数与文本批处理。

## In-situ series

用 Single analysis 里的模型拟合一个文件夹里的每条曲线（`application/series_fit.py`）。

- **Curves**：Choose Folder…（或 Analyze ▸ Send Series to Fitting）；Files 的 pattern（默认
  `*_fit_input.dat`）、Also in subfolders；Frames 第一帧 – 最后一帧、每隔几帧（编号取文件名最后一个
  数字，自然顺序）；Watch for new curves：运行中每 2 秒找新写完的曲线并接着拟合。列表里每帧前面
  标 ✓ / ✗ / ·。
- **Start**：Single analysis 的模型、两半、拟合范围和排除的点（摘要写在这里，Edit in Single
  Analysis 回去改）；每帧从上一帧的结果开始（默认，上一帧没收敛时退回 Single 的模型）或都从 Single
  的模型开始；方法 Refine（快）或 Search the ranges, then refine（慢）。
- **运行**：曲线在界面线程读入，拟合在后台一帧一帧做；Pause / Stop；选中最后一帧时图跟着运行走，
  选了别的帧就停在那里。
- **Results**：“拟合 N 帧 · 失败 · 未收敛”；每帧一行（#、χ²ᵣ、每个自由值 ± 1σ，列名与 Results 步骤
  相同，如 “1·Sphere R (nm)”）；点一行看那一帧；Open Frame in Single Analysis 把该帧与它的拟合模型
  送回单条分析；折叠的日志。下面的趋势图选一个值（或 χ²ᵣ），画成点和 ±1σ 误差棒；坐标轴标题保持
  英文（图的内容），Log I 切换后误差棒重画。
- **Save ▾**：Table of Every Frame…（CSV：frame、file、chi2_reduced、log_rmse、converged、每个值与
  `_error`）旁边写同名 `.json` 记录（起始模型、两半、范围、排除的点、方法、失败的帧与原因）；
  Trend Plot…、Selected Frame's Plot…（PNG / SVG）。
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

- [ ] Analyze 打开 CBF 后 Send to Fitting：切到 Fitting，Curve 卡片显示 `_fit_input.dat`、点数、
      q 范围与 observation；四种半边选择对应的 q display 正确；
- [ ] Open Curve… 打开两列 / 三列 / 四列 `.dat` / `.txt` 曲线；坏文件或其他格式给出明确错误；
- [ ] Components、Global、Refine、1D Predict 四个标签始终可见、无滚动箭头；1D Predict 按钮不被截断；
- [ ] Fit curve / AI guess only / Physical fit (no AI) / Stop / Fit settings & batch… 正常；幅度校准开关
      独立，回退原因可见；
- [ ] Overlay ±q as |q| 中 +q 蓝、−q 红，嵌入图与 Open plot 一致；Log X 在 signed 模式下为 symlog；
- [ ] Global Search / Local Refine 写回选中参数；参数输入不静默舍入小值；
- [ ] Send Series to Fitting 导出所有帧后打开 In-situ series，folder 与 pattern 已填好、曲线已列出；
- [ ] Start 按自然顺序拟合，坏文件记为失败并继续；Watch for new curves 只拟合新写完的曲线；
- [ ] 每帧的误差合理（不出现 ±1e-20 或空白），趋势图的误差棒与表一致；
- [ ] Exclude：点选 / 框选排除、再点恢复、Include All；排除的点不参与拟合，序列也用同样的排除；
- [ ] Single / In-situ 来回切换不重置曲线、模型或结果表；
- [ ] Export Data 的三种表示与页面状态一致，header 可追溯；
- [ ] Export Plot… 写出的 PNG／TIFF／SVG／PDF 与屏幕上的图层一致，一栏宽、600 dpi；
- [ ] 切换浅色／深色主题时曲线图背景、坐标轴与图例同时变化，导出图不受主题影响。
