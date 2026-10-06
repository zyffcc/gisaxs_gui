# Analyze 工作区

- 状态：打开数据后的主工作区，统一处理 GISAXS 与 GIWAXS（原 WAXS 页面与原 Cut & Fitting 的探测器部分已并入）。
  Fitting 只拟合曲线。最近验证：2026-09-29（Windows，offscreen 截图 + 812 个测试）。
- 调用链：`AnalyzePage → AnalyzeViewModel → AnalyzeFrame / CorrectSeriesFrame / ExportAnalysis / series_map → ports`。
- 静态布局：`presentation/views/analyze_page_view.py`（命令栏、三栏、状态行）、`analyze_steps_view.py`（步骤栏与各步页面）、
  `options_panel_view.py`（Frames / Corrections / GIWAXS cuts 分区）、`series_view.py`（Series 标签页）。
- 行为（mixin，`presentation/page.py` 组合）：`bindings/workspace.py`（步骤、反馈、来源与掩膜叠加、自动分析）、
  `display.py`、`beam_center.py`、`options.py`、`batch_watch.py`、`profile_actions.py`、`series.py`。
- 共享组件：`app/presentation/components/`（DetectorView、CurvePlot、StepRail、SegmentedControl、EmptyState、Toast）。
- 自动化：`presentation/automation.py` 的 `AnalyzeAutomation` 让自动分析和 AI 像用户一样操作本页（每个改动同步到控件、
  重新分析一次再回调）；曲线一律以 q（Å⁻¹）返回。
- 相关测试：`test_analyze_*.py`、`test_guided_start.py`、`test_guided_gisaxs.py`、`test_series_map.py`、`test_bad_pixels.py`、
  `test_public_data.py`。

## 布局

| 区域 | 内容 |
|---|---|
| 命令栏 | Open…（下拉 Open Folder…）· ‹ 文件 ›（列出多于一个文件时出现）· Auto / GISAXS / GIWAXS · αi · Run Automatic Analysis（运行时旁边出现 Stop）· Ask AI… · Export ▾ · Batch Export…（列出多于一帧时出现）· Send to Fitting ▾ |
| 左：步骤 | 1 Data · 2 Geometry · 3 Mask & corrections · 4 Cuts · 5 Results · 6 Export；每步显示状态（✓ / ! / ✕ / 进行中）与一句结论，点开是该步的少数常用控件，高级项折叠 |
| 中：图像 | Detector / q map、Log、色图、Auto levels、Zoom、Fit view、Sources（每条曲线来自图上的哪里，可点曲线只看它的来源）；光标读数（过长时截断，完整内容在提示中，不会改变视图大小） |
| 右：标签页 | Curves（两个曲线图）· Results（自动分析或 AI 的结果与证据）· Series（帧 × q 热图） |
| 状态行 | 文字 + 进度条 + Cancel；写出文件时右下角 toast，可直接打开文件夹 |

## 默认流程（零点击）

```text
打开 / 拖入文件或文件夹（CBF、NXS 多模块拼接、TIFF、EDF）→ 立即显示文件名、探测器、帧数（Data 步骤）
按探测器名 + 帧尺寸匹配仪器配置 ──没有──→ 提示条：Find Calibration Automatically / Enter Geometry / Calibrate
    ↓ 有效像素：整数计数探测器的负值是缝隙码；浮点帧（扣暗场）的负值是数据
    ↓ 自动排除孤立的热像素、死像素和有缺陷的探测器行 / 列（Mask 步骤可关）
    ↓ Sum N frames（可选）→ 减背景（可选）→ gap guard（默认 3 px）→ 有效强度范围（可选）
最大散射角 ≤ 20° → GISAXS；否则 GIWAXS（命令栏可强制）
GISAXS：Yoneda 行的 I(qy) + 束流中心列的 I(qz) + qy–qz 图
GIWAXS：I(q)、面内 / 面外扇区、自定义扇区、q 框、最强环的 I(χ)、q∥–qz 图
```

## 自动分析（不需要 AI）

Run Automatic Analysis（Results 步骤可写束线时笔记：αi、能量、像素尺寸、标定位置）：

- GIWAXS：几何（配置 / 附近的标定文件或标样图像）→ 序列取末 10 帧 → 峰表 → 面内 / 面外 → 最强可靠环的取向与 Scherrer 尺寸。
- GISAXS：几何 → Yoneda 切线（找不到时提示检查 αi）→ 按左右对称移动中心列 → 选择两半（都可用且一致就平均，
  一半被遮挡 / 有缝隙 / 太短就用另一半，并写出理由）→ 面内间距 2π/q*（只有 shoulder 时标为提示）→ 数值物理拟合
  （球、竖直圆柱、随机圆柱，含尺寸分布与间距 D，也从间距起步）。多个模型 χ² 相差 10 % 以内时，结果里会问你选哪个模型，
  并指出与间距一致的解。
- Results 标签页：GISAXS 为切线与两半、间距、拟合图（数据 + 选中的解，Log q / Log I）与解的表格；Save Fitted Curve…、
  Save Fit Table…、Refine in Fitting。GIWAXS 为峰表、检查、几何、序列对比。两者都有 Save Report…（网页或 Markdown）。
- 表格下面是折叠的 **Fit details**（默认关；打开后下一次报告也保持打开），跟随选中的行，只在打开时绘制：
  峰——拟合用到的点、高斯 + 局部线性背景、q / d / FWHM 及误差、峰高、面积、背景与斜率、S/N、χ²ᵣ、拟合窗口、
  Scherrer 尺寸、寻峰与拟合方法，以及该峰在 q 图上测到 / 阴影 / 没测到的部分；
  模型解——每个参数（R、σR/R、h、D、σD/D、权重、振幅、常数背景、分辨率）、χ²、是否收敛、计算次数与用时、
  q 范围、D 的初值、算法、警告，和 **Show in Fitting**（在拟合中打开曲线并画出这个解；拟合的步骤栏显示“当前画出的解”，
  Export Data… 保存拟合与参数）。数据在报告里（`peaks[].fit`、`peak_search`、`gisaxs.fit.solutions[]`、`gisaxs.fit.native`），
  给 AI 的精简报告去掉数组。
- 只有你能回答的问题（αi、能量、标定文件等）在 Results 步骤里变成输入框，回答后 Run Again；步骤、标记和状态行中的
  问题数相同（同一个值的几个问题算一个）。
- 结果属于运行时的那一帧：运行期间文件列表锁定（变灰），新列出的帧排在后面；运行结束后换到别的帧，Results 步骤说明
  显示的是哪一帧的结果；回到该帧时恢复（最近 24 个文件）。序列只对运行用到的帧显示 ✓。进度面板的 Discard 也会清掉
  Results。报告标题为 GISAXS / GIWAXS / Geometry report，保存在数据旁边，写完有 toast 可打开文件夹。
- 进度（和 AI 面板一样）：运行时右侧切到 Results，顶部是进度面板——“正在：……”一句话说明当前步骤和已用时间，阶段清单
  （读取帧 · 几何 · 序列中的帧 · 峰 / Yoneda 切线 · 面内面外 / 间距 · 取向与尺寸 / 模型拟合）逐项打勾并显示用时，
  不需要的阶段标为“不需要”，进度条按阶段前进。
- Stop（进度面板和命令栏）：在下一步开始前结束。正在进行的一步不能打断——此时显示“正在停止：等待‘用标样拟合几何’完成
  （已用 12 s；这一步可能需要一分钟左右）”，慢的步骤（几何拟合、模型拟合、查找标定文件、求和很多帧）会说明大概要多久。
  停止后已得到的结果（峰、几何、标定记录、决定与理由）留在 Results 中，面板提供 Save Report…（保存）和 Discard（丢弃）；
  不操作就一直保留。报告中 `stopped` 记下没有开始的那一步。

Ask AI… 用同样的工具（GIWAXS 或 GISAXS 的结果选项随当前模式变化），改动以卡片形式预览 / 应用 / 撤销。

## Series（原位 / 批量）

- 热图生成后自动给出**阶段与异常帧**（彩色阶段条、边界虚线、红箭头；Stages 行；折叠的“变化了什么，以及异常帧”；
  Change along the series；Stages as Table；Batch Export 可跳过异常帧；Send to Compare）。方法与验收见 [compare.md](compare.md)。

- 列出多个文件、打开文件夹或多帧 NeXus 后，Series ▸ 选择曲线（只有一帧时这里只说明怎样得到序列，控件隐藏；Export 在有热图后出现）（GISAXS 默认水平切线，GIWAXS 默认 I(q)）▸ Build Map：
  每帧（或每组相加帧）用当前设置处理，热图逐步长出来（至多每秒重画一次）；和 Batch Export 共用同一个运行器
  （`bindings/batch_run.py`）：长序列在多个低优先级进程里同时处理，面板显示进度、剩余时间，可暂停 / 停止（已处理的行保留）。
- Batch Export 运行时同样把每帧的行实时加进 Series 热图，下方曲线跟随最新的一帧（点热图某行即停止跟随）；
  速度 Gentle / Balanced / Fast（`application/batch_job.py: frames_at_once`：核心数、一半空闲内存、每帧约 100 B/像素；
  短批次不开进程）；结果按帧顺序折叠（表格、从上一帧开始的拟合），工作进程意外退出时其余帧回到程序内逐帧处理。
  “every n frame”只取每 n 帧，长序列先快速看全貌（Lambda 9M 每帧约 1.7 s；序列行不计算 q 图）。
- 热图就是一个 DetectorView：Log、色图、色阶可调；读数显示帧、文件、x 与强度。
- 热图属于建图时的文件列表：之后再列出文件，热图保留并注明 “Map of the earlier list (N frames)”；Clear 清空列表时
  热图一并清除（正在建的图也停止）。
- 水平带选帧（下方左图画出该帧曲线，Open 在 Analyze 中打开该帧），竖直带选 q 窗口（下方右图画出强度随帧的变化，
  默认放在序列变化最显著的 q 处）；两条带都能拖动。
- Export ▾：Map as CSV Table…（首行 x，首列帧号，每帧一行 `#` 注明文件）、Map as Figure…、所选帧曲线、强度随帧。
- 右下图的下拉：Mean intensity（q 窗口内的平均强度随帧）或 Peak position / FWHM / area / height（窗口内的峰：
  扣去直线背景后的质心、2.355 σ、面积、峰高，每帧一个值）；Export ▾ → Peak Table of Every Frame…（CSV，一帧一行）。
  曲线下拉里有每个区域的 I(q) 和 I(χ)，所以原位序列可以按任意区域看。

## 束流中心

- 默认用仪器配置里标定过的中心，不受文件头影响（设置 ▸ Analyze 可改为信任文件头）。
- Geometry 步骤 ▸ Beam centre：拖动图上的青色十字、Pick on Image、Enter Coordinates…、Use File Header Centre、
  Refine x by Symmetry（GISAXS：在水平切带内找左右对称轴，只改 x）。改动对之后的文件保持；Save to Profile 写入配置，
  Back to Profile Centre 放弃。
- 本次会话设的中心只用于同一探测器尺寸（行 × 列）的帧；尺寸不同的帧回到配置的中心。尺寸随中心写入设置文件与项目
  （`beam_center_shape_px`）。

## αi 与模式

- 命令栏的 αi：没有输入时显示 “αi from profile”；为本次会话输入的值以橙色标出，右键 ▸ Back to Profile 回到配置的值。
- 模式固定为 GISAXS 或 GIWAXS 而帧看起来是另一种时，状态行会提醒；Auto 每帧重新判断。

## 送到 Fitting

Send to Fitting 把完整的带符号水平切线（GIWAXS 为 I(q)）写到 `gimap_analysis/<stem>_fit_input.dat`
（`q I sigma pixels`，q 单位 Å⁻¹，`# observation:` 记录来源、gap guard、相加帧数）并在 Fitting 中打开；下拉选择两半的用法
（Both halves on |qy| / Mean / qy < 0 / qy > 0），与 Cuts 步骤的 Halves 同步。Send Series to Fitting… 导出全部帧后打开
Fitting ▸ In-situ series。

## Mask & corrections（预处理）

- 摘要：排除的像素总数；其中探测器缝隙与标记像素、本帧找到的热 / 死像素、手绘或载入的掩膜、gap guard、强度范围、
  背景各多少，以及从镜像一侧补齐了多少像素。Show Masked Pixels on the Image：红色 = 排除，琥珀色圆圈 = 热 / 死像素，
  青色 = 由镜像补齐（已参与计算）。
- Leave out hot and dead pixels（默认开，记住）：只标记孤立像素——比 8 个邻居都亮很多（计数探测器用 Poisson 标准差，
  浮点帧用全帧稳健噪声），或读数为 0 而邻居都 ≥ 20 counts；峰、条纹、beam stop 边缘都保留。
- Fill gaps from the mirror side（只在 GIWAXS 显示，默认关）：GIWAXS 对 ±q∥ 对称；没有数据的像素（模块缝隙、掩膜、热像素）取它关于
  束流中心列的镜像位置的值（两列之间插值，两列都须有数据）。镜像也没有数据的地方保持空缺。补齐的像素计数写进导出记录，
  Fitting 的计数模型标记为不再成立（`counting_model_valid: false`）。
- Masks you draw：Rectangle（在探测器图上点两个对角）、Polygon（逐点单击，双击或 Enter 闭合，Backspace 删最后一点，Esc 取消）；
  列表与 Remove Selected / Clear Masks 只在有掩膜时出现；Save… 存为 JSON（规范像素坐标），Load… 载入 JSON 或同尺寸的掩膜图（EDF / TIFF，非零 = 掩膜，pyFAI 约定）。
  掩膜在图上以红色轮廓显示，立即从所有曲线中排除。
- Corrections（折叠区）：背景帧（缩放、帧号）、有效原始强度 Min / Max、gap guard（0–20 px，记住）。
  顺序：探测器无效像素 → 热 / 死像素 → 掩膜 → 背景 → gap guard → 强度范围 → 镜像补齐 → 强度校正；曲线、q 图、展开图与导出都用处理后的帧。
- Intensity corrections (GIWAXS)（折叠区，只在 GIWAXS 显示，默认关，`domain/intensity.py`）：立体角 cos³2θ、偏振（pyFAI 公式，因子 f）、
  薄膜吸收（厚度 t、衰减长度 L，相对 αf = αi，不含折射）；I / factor，q 不变。计数帧的方差按 value / factor 传递
  （`BinnedMean.add(..., scale)`），导出记录的 `intensity_corrections` 写明公式与参数，Fitting 的计数模型标记为不成立。
- Undo / Redo（命令栏 ↶ ↷，Ctrl+Z / Ctrl+Shift+Z / Ctrl+Y，`setup_history.py`）：设置的每次改动是一步，同类快速改动合并
  （但抵消上一步的改动——比如刚加上又删掉——单独成一步）。
- 色阶（`components/levels.py`）：图像旁是显示值的直方图，两个手柄是绝对位置（拖动快速调整，滚轮放大色标做精细调整）；
  Levels ▾：每帧自动（规则 1–99.7 % / 0.1–99.9 % / 5–95 % / 最小–最大 / 均值 ± 3σ）、输入 Min / Max（强度单位）、自动一次；
  拖动或输入后固定，换帧、重新分析、实时序列热图都不再重置；探测器、q 图、展开图、序列热图各自保存；
  批量导出的图片可选“每帧自己的上下限”或“屏幕上的上下限，所有帧相同”（写入批处理 JSON）。
- 标记（`components/marks.py`、`bindings/marks.py`）：每个图像视图的 Marks ▾ 和每个曲线图的眼睛按钮，按类显示 / 隐藏
  （光束中心、地平线、切线带、掩膜与区域、像素叠加层、热 / 死像素、q 框、色标、窗口带），全部显示 / 隐藏；隐藏选择按视图
  记在设置 `analyze.hidden_marks`；自己画的可从同一菜单删除（掩膜、全部区域、q 框，可撤销）。

## GIWAXS：区域与切线（Cuts 步骤）

- 三种视图（图像区上方）：Detector、q map（q∥ 即 qr / qxy 对 qz）、Cake（展开图：χ 对 q，环是竖线、区域是矩形）。
  Save ▾ → View as Figure…（当前视图，带色标）/ View Data as CSV…（q 图或展开图的数据表，带坐标轴）。
- 区域列表：整环（Full ring）、面内带、面外带、I(χ) 的环（标准切线，自动分析和 AI 也用它们）、自定义扇区（AI 设置时出现），
  以及你添加的区域。每个区域 = q 范围 × χ 范围，可选“两侧”（±χ，折叠到 |χ|）；每个区域给出两条曲线：沿 q 的 I(q)
  （上图）和沿 χ 的 I(χ)（下图）。勾选框控制显示；同一区域在曲线、q 图轮廓、展开图矩形和 Sources 中颜色相同。
- 自定义切线（点击即可，自动对准峰）：“点击图像添加切线”下的三个按钮，再在探测器图、q 图或展开图上点一下——
  - Ring（环）：点一个峰 → 该峰的 q 窗口 × 所有 χ，用于 I(χ)（取向）；也可以点上方 I(q) 曲线图上的峰。
  - Sector（扇区）：点一个方向 → 所有 q × 该 χ 附近（默认 ±5°，点在斑点上时按斑点的方位宽度），用于沿该方向的 I(q)。
  - Spot（斑点）：点一个斑点 → q 窗口 × χ 窗口，得到该斑点的 I(q) 与 I(χ)。
  对准规则（`domain/region_pick.py`）：从点击处沿平滑曲线爬到最近的极大值；背景取两侧最低点连线；中心 = 半高以上部分的
  质心，宽度 = FWHM，窗口 = 中心 ± FWHM，不越过两侧最低点（不吃进相邻峰）。低于 4 倍点间噪声、不足两个原始 bin 高于半高、
  在数据边缘被截断、或离点击处太远（q 的 2 % / χ 6°）的都不算峰——此时取点击处附近固定宽度（q ±1.5 %），并在提示中说明。
  点击处的 χ 带（±10°）没有数据或没有峰时改用整环。直接用像素（不是展开图网格），精度与数据相同，约 20–300 ms。
  新区域以位置命名（`Ring q 0.998`、`Sector |χ| 85°`），提示给出峰位、d、FWHM，可撤销。
- 精确修改：选中区域后在下方输入“中心 ± 半宽”（q 四位小数，χ 0.1°），或勾选 All q / All χ / 两侧；下面一行显示实际范围和
  d = 2π/q 的范围。Snap to Peak 把区域重新对准离中心最近的峰（可在 1.5 倍区域宽度内寻找）。
- Add ▾：Draw on the Cake View（在展开图上点两个对角画矩形；在 χ = 0 一侧的矩形自动变成两侧区域）、最强峰处的环、
  面内带（|χ| 70–90°）、面外带（|χ| 0–20°）、整个图样，以及 Save Cuts… / Load Cuts…（区域集合存为 JSON，
  `{"format": "gimap-cut-regions", "regions": […]}`，下一组数据直接载入；载入替换你添加的区域，标准切线不变）。
  也可在 Cake 视图直接拖动 / 缩放矩形。I(χ) 的环也可在展开图上拖动，或点 Strongest Ring 回到自动。
- More cut settings（折叠）：Radial bins、I(q) 横轴 q 或 2θ、q 框（q∥ × qz，在 q 图上可拖动）。
- 两个曲线图右上角 Save ▾：Plot as Figure…（论文图）/ Curves as Data…（该图的每条曲线一个 CSV + JSON 记录）。
- 曲线的 x 有正负时（GISAXS 水平切线的 qy、环的 I(χ)），曲线图上方出现 ± / + / − / |qy|：两侧、只看正半轴、只看负半轴、
  两侧都画在 |x| 上（负半轴为另一深浅的虚线，比较两侧是否一致）。只改变显示和 Plot as Figure…，导出的数据不变。
  图较窄时（例如 1280 px 窗口）标题换行显示在图上方，这些选项收进 **⋯** 菜单；右键菜单有 Reset View、对数轴、
  Plot as Figure…、Curves as Data…、Copy Image、Copy Data，右键拖动缩放。
- 放大：每个图像和曲线图都有 Zoom（放大镜）——按下后拖出矩形放大；在任何视图中按住 Shift 拖动也可以；Reset / Fit view
  恢复全部。Fitting 的曲线图（以及原位序列的图）直接拖出矩形放大，双击恢复；重新拟合、重画数据时保持放大，载入新曲线时恢复。
- AI 工具 `set_cut_regions` 设置同一个区域列表（可预览、应用、撤销）。

## 曲线上的点从哪里来

- 每个点 = 一个 bin 内有效像素的平均强度。bin 不比一个像素细（自动 bins 按像素的 q 步长选）；像素太少的 bin
  （少于 max(8, 中位数的 10 %)，多在光束附近和区域边缘）沿 q 与相邻 bin 合并，x 与强度按像素加权，不跨越缝隙。
  I(χ) 不合并：取向分析依赖 1° 的 bin；区域的 I(χ) 在小 q 处用更宽的 χ bin（不窄于该 q 处的一个像素）。
- 自动排除（Mask 步骤可关）：孤立的热 / 死像素，以及有缺陷的探测器行 / 列——一行（列）中散布的、比上下（左右）两行
  亮得多的像素（例：Lambda 上两行像素达 4·10⁵ 计数，而图样只有 0–几个计数，成对出现，孤立像素检测看不到）。
  连续的棒、Yoneda 带、地平线边缘不受影响。Mask 摘要中单独列出这些像素数。
- σ：整数计数帧且未减背景时用泊松误差 √sum / n；浮点（扣暗场）帧、减背景或镜像补齐后改用 bin 内像素离散度 s / √n，
  拟合输入的 `observation.counting_model_valid` 为 false。
- 对数纵轴的范围跟随曲线主体，个别接近 0 的点（扣暗场后的噪声）不再把轴拉到很低；刻度标签按可用高度取
  每个数量级的 1 / 1-3 / 1-2-5 / 1…9，矮图上不再叠在一起。
- 没有像素的 bin（探测器缝隙、被掩膜的 χ 范围、缺失楔）没有点；曲线在这里断开，不画跨过缺口的直线
  （屏幕与 Plot as Figure… 相同；导出的数据不变）。
- 以上都处理后仍然抖动的是真实噪声（例：P08 平板探测器扣暗场后每像素噪声约 25 计数，面内扇区的信号约 10）。
  要更平滑的曲线，加宽区域的 χ 范围或调小 Radial bins（点少、每点像素多），误差棒随之变小；程序不对曲线做平滑。

## 导出（Export 步骤）

- Export Curves…：每条曲线一个 CSV（`x, I, sigma, pixels`）+ `_analysis.json`（来源、几何、中心来源、校正、热 / 死像素数、切割）。
- Save Image…（探测器图或 q 图，含色标）、Save q-Map Data…（规则 q 网格上的强度 CSV）、Save Plots…；PNG / TIFF / SVG / PDF。
- Batch Export…：列表中多于一帧时，命令栏出现 Batch Export…（描边强调色），Data 步骤文件列表下出现
  “Batch Export N Frames…”，Series 标签页 Build Map 旁也有，并提示一次“设置好一帧，批量导出全部”；快捷键 Ctrl+Shift+E；
  也在 Export 步骤第一个按钮、Export ▾ 菜单和开始页。列表中的每一帧用当前设置处理，
  一个对话框选完。上方：帧（每 n 帧取一帧）与设置摘要（Load / Save Settings…）；中间四组，每个选项右边写出它会生成的
  文件（含子文件夹和扩展名，随格式变化）；下方：保存到的文件夹（可选以数据命名的子文件夹）。

  | 组 | 选项 | 文件 |
  |---|---|---|
  | 数据（数值），格式 CSV（逗号）/ TXT（制表符）/ DAT（空格） | 所有帧：每条曲线一个表格（第一列 x，之后每帧一列） | `<名称>_<曲线>_frames.csv` |
  | | 每帧：各条曲线（x、I、σ、像素数）+ 设置 JSON | `curves/<帧>_<曲线>.csv` |
  | | 每帧：拟合用曲线 | `fit_input/<帧>_fit_input.dat` |
  | | 每帧：q 图数值 / χ–q 展开图数值 | `maps/<帧>_qmap.csv`、`maps/<帧>_cake.csv` |
  | 图片，格式 PNG / TIFF / SVG / PDF | 每帧：探测器图像的图片；每帧：q 图的图片（分别勾选） | `images/<帧>_detector.png`、`images/<帧>_qmap.png` |
  | 转换格式的探测器帧，格式 TIFF 32 位 / EDF / NumPy / HDF5 | 每帧：探测器数据（按读取原样：模块拼接、帧求和）——即格式转换 | `frames/<帧>.tif` |
  | 拟合（可选） | 见下 | `<名称>_peak_fits.csv` / `_model_fits.csv` + 同名 `.png` |

  选表格与逐帧曲线时，在列表中勾选要哪些曲线（All / None）。表格中同一几何的帧 x 完全相同；否则插值到第一帧的 x，
  表头写明，不外推。文件夹里总会写 `README.txt`（每种文件是什么、曲线键对应的名称、单位）和 `<名称>_batch.json`
  （选项、帧列表、失败的帧、写出的文件、第一帧的完整设置）。
- 批量拟合（可选，默认不拟合）：
  - GIWAXS——“环区域中的峰”：每个有 q 窗口的区域（切线 ▸ 环 / 斑点）是一个峰；在该区域的 χ 范围内，按探测器 q 步长取
    I(q)，窗口为区域窗口两侧各加一个窗口宽度（背景才能确定），拟合 Pseudo-Voigt / 高斯 / 洛伦兹 + 直线背景（加权最小二乘，
    误差按 √χ²ᵣ 放大）。表中每帧一行、每个峰：q ± 误差、d = 2π/q、FWHM ± 误差、面积 ± 误差、峰高、η、χ²ᵣ、说明
    （不显著、太宽、点太少时写原因，该帧不影响下一帧的起点）。
  - GISAXS——“水平切线的颗粒模型”：拟合用曲线（水平切线选定的一侧）交给 Fitting 的快速物理拟合（球 / 随机取向圆柱 / 竖直
    圆柱，含尺寸分布与间距 D；模型可选自动比较或指定一种），表中每帧一行：模型、χ²、log RMSE 与各组分参数（nm）。
    约半分钟一帧。
  - 起始值：上一帧的结果（缓慢变化的序列，峰或颗粒随之移动；GISAXS 取上一帧的模型与 D）/ 第一帧的结果（所有帧同一起点，
    帧之间互不影响）/ 每帧重新寻找。
  - “在当前帧上试拟合”：用这些选项拟合当前显示的帧，结果直接显示在对话框中，满意再批量。
  - 可选同时保存每帧的拟合曲线 `fits/<帧>_<曲线>_fit.csv`（q、数据、拟合）；拟合值随帧的变化图 `<名称>_peak_fits.png`。
  - 更复杂的拟合（Fitting 的 AI 辅助模型、每帧各自的输入选择）：Send Series to Fitting… → 拟合 ▸ 原位序列。
- 逐帧在后台处理，状态行显示进度、剩余时间和是否在拟合，可取消（已写的保留，表格包含已处理的帧）。选项、格式和文件夹会
  记住（设置节 `analyze_batch`），下次打开对话框即是上次的选择。只需要 q 图或 q 图图片时才计算 q 图（P03 Lambda 9M 约 2 s/帧）。
- 开始页 Batch Export…：选一个原始数据文件夹 → 自动加载并分析第一帧（几何按探测器匹配，或设置文件中的配置）→
  直接弹出 Batch Export 对话框 → Export。
- Save Settings… / Load Settings…（Export ▾ 和对话框中）：`{"format": "gimap-analyze-settings"}` JSON，包含模式、仪器配置
  （名称和几何，换电脑也能用；缺失时自动加入）、αi、中心、求和、校正（缺陷像素、掩膜形状与文件、镜像补齐、有效范围、背景）、
  GIWAXS 切线（条带宽度、bins、x 轴、环窗口、扇区、q 框、区域）、GISAXS 切线和导出选项。载入后所有控件同步；找不到的
  掩膜或背景文件会列出并忽略。
- Send Series to Fitting…：同一个对话框的拟合模式（只写每帧的拟合输入），完成后在拟合 ▸ 原位序列中打开。
- Watch：监视文件夹，新帧写完后自动加入；Export ▸ Export New Frames While Watching 时自动导出。

## 界面语言

设置 ▸ Appearance ▸ Language，或菜单 View ▸ Language：English（默认）/ 中文。命令、步骤、按钮、菜单、表头、数值框的单位后缀和运行中生成的句子
切换为中文；切换语言时当前页面的步骤说明、图标题和状态行随之重写。数值、单位、文件名，以及作为数据的表头（序列名、
参数名）不翻译。

## 手动验收清单

1. 开始页 ▸ Open Files…：选完文件立即跳到 Analyze，Data 步骤显示文件名、探测器、帧数，图像随后出现。
2. GISAXS 帧（例如 `tests/data/external/gisaxs_galaxi/`，配 `galaxi_bornagain.poni` 的几何）▸ Run Automatic Analysis：
   Results 显示 Yoneda 切线、两半的理由、间距（约 53.5 nm，shoulder）、拟合表，并提示选择模型；Save Fitted Curve… 有 toast。
3. Sources 打开后点击某条曲线：图上只显示它的来源区域。
4. Mask 步骤：P03 Pilatus 帧显示找到的热像素数；取消 Leave out hot and dead pixels 后这些像素回到曲线里。
5. 列出一个文件夹 ▸ Series ▸ Build Map：热图逐步生成，Cancel 后保留已有行；拖动竖直带，右下图更新；Export ▸ Map as CSV Table…。
6. GIWAXS 帧（例如 `tests/data/external/giwaxs_p08_mapi/` 配其 .poni 的几何）：Cuts ▸ Ring，在 q 图上点 q≈1.0 的环：
   出现 `Ring q 0.998`，提示给出峰位、d、FWHM；上下图出现该区域的 I(q)、I(χ)（同色）；Sector 点在面内方向得到 |χ| 80–90°；
   切到 Cake，拖动矩形，列表与曲线随之更新；在下方输入中心 ± 半宽，Snap to Peak 回到峰上；Add ▸ Save Cuts… / Load Cuts…。
7. Export ▸ Batch Export…：对话框显示帧数、设置摘要，勾选“每条曲线一个表格”和某个区域的两条曲线，选文件夹，Export；
   状态行显示进度和剩余时间；文件夹中出现 `<名称>_region1_frames.csv`（每帧一列）和 `<名称>_batch.json`；再次打开对话框时
   是上次的选择。Save Settings… 后重启，开始页 Batch Export… 选同类数据的文件夹，Load Settings…，Export。
8. Mask 步骤：画一个矩形，列表与图上出现；勾选 Fill gaps from the mirror side，摘要显示补齐的像素数，Show Masked Pixels
   中该区域为青色。
9. 设置 ▸ Language ▸ 中文：界面文字变为中文；切回 English 后完全恢复。
