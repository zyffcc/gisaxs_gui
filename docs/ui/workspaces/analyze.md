# Analyze 工作区

- 状态：打开数据后的主工作区，统一处理 GISAXS 与 GIWAXS（原 WAXS 页面与原 Cut & Fitting 的探测器部分已并入）。
  Fitting 只拟合曲线。最近验证：2026-10-14（macOS：NeXus、Series、Batch Export、送到 Fitting 各节逐条对照代码与测试；
  其余各节沿用 2026-09-29 Windows 上的验证）。
- 调用链：`AnalyzePage → AnalyzeViewModel → AnalyzeFrame / CorrectSeriesFrame / ExportAnalysis / series_map → ports`。
- 静态布局：`presentation/views/analyze_page_view.py`（命令栏、三栏、状态行）、`analyze_steps_view.py`（步骤栏、帧条与各步页面）、
  `options_panel_view.py`（Frames / Corrections / GIWAXS cuts 分区）、`series_view.py`（Series 标签页）、
  `batch_export_view.py`（Batch Export 对话框）、`frame_import_dialog_view.py`（导入帧选择）、
  `series_fit_view.py`（Series 下的批量拟合预览）。
- 行为（mixin，`presentation/page.py` 组合）：`bindings/workspace.py`（步骤、反馈、来源与掩膜叠加、自动分析）、
  `display.py`、`beam_center.py`、`options.py`、`data_step.py`（源的说明、帧条）、`file_list.py`、`batch_watch.py`（导出、监视文件夹）、
  `watch_series.py`（完整新帧追加到序列热图）、`frame_import.py`（后台检查源、虚拟帧列表）、
  `fit_handover.py`（Send to Fitting）、`batch_export.py` / `batch_run.py`、`profile_actions.py`、`series.py`（及 `series_display.py`、
  `series_pick.py`、`series_exports.py`、`series_stages.py`、`series_fit_results.py`）。视图模型的帧与源：
  `frames_model.py`、`sources_model.py`、`frame_selection.py`、`frame_sequences.py`、`folder_watch.py`。
- 共享组件：`app/presentation/components/`（DetectorView、CurvePlot、StepRail、SegmentedControl、EmptyState、Toast）。
- 自动化：`presentation/automation.py` 的 `AnalyzeAutomation` 让自动分析和 AI 像用户一样操作本页（每个改动同步到控件、
  重新分析一次再回调）；曲线一律以 q（Å⁻¹）返回。
- 相关测试：`test_analyze_*.py`（NeXus 源：`test_analyze_nexus_sources.py`）、`test_live_following.py`、`test_guided_start.py`、
  `test_guided_gisaxs.py`、`test_series_map.py`、`test_series_map_display.py`、`test_series_cut.py`、`test_batch_export.py`、
  `test_batch_flow.py`、`test_batch_halves.py`、`test_batch_names.py`、`test_handover.py`、`test_bad_pixels.py`、`test_public_data.py`。

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
打开 / 拖入文件或文件夹（CBF、NeXus/HDF5、TIFF、EDF）→ 后台检查源 → 多帧源选择导入范围 → Data 中每帧一行
按探测器名 + 帧尺寸匹配仪器配置 ──没有──→ 提示条：Find Calibration Automatically / Enter Geometry / Calibrate
    ↓ 有效像素：整数计数探测器的负值是缝隙码；浮点帧（扣暗场）的负值是数据
    ↓ 自动排除孤立的热像素、死像素和有缺陷的探测器行 / 列（Mask 步骤可关）
    ↓ Sum N frames（可选）→ 减背景（可选）→ gap guard（默认 3 px）→ 有效强度范围（可选）
最大散射角 ≤ 20° → GISAXS；否则 GIWAXS（命令栏可强制）
GISAXS：Yoneda 行的 I(qy) + 束流中心列的 I(qz) + qy–qz 图
GIWAXS：I(q)、面内 / 面外扇区、自定义扇区、q 框、最强环的 I(χ)、q∥–qz 图
```

## Data 步骤：源、帧与 NeXus / HDF5

- 后缀：CBF、TIFF、EDF，以及 NeXus / HDF5 的 `.nxs`、`.nx`、`.nx5`、`.h5`、`.hdf5`、`.hdf`。NeXus / HDF5 的“源”由内容决定，
  文件名只作提议（`shared/detector_io/nexus_layout.py`）：
  - 一个文件一帧；一个文件装整个序列（`(扫描点, 帧, 行, 列)` 的网格按存储顺序当作一串帧，并说明）；
  - 每个探测器模块一个文件（P03 Lambda 9M）：只差 `_mNN` 的文件，只有数据集路径、模块尺寸相同且模块位置
    （探测器组的 `translation/distance`）两两不同时才算一组，按位置拼接；不符合的各自单独读，并说明原因；
  - master 加数据文件（Eiger 的 `data_NNNNNN` 链接或虚拟数据集）。
  - 图像数据集：文件标的默认图 → 探测器组的 `data` → 编号序列 / 虚拟数据集 → 其他像图像的数据集；掩膜、平场、暗场
    不当作图像。几个同样可能的图像时显示一个并列出其余的（Format Converter 和 Calibration 可以选）。
- **统一帧列表**：Data 中每个已导入的帧一行（实际帧号 + 源名称）；普通多文件和多帧 NeXus 的操作相同。
  模块组、master 和其数据文件先合并为一个源，再按帧列出，不重复计数；模块组的源名称去掉 `_mNN`。
  不含探测器图像的 HDF5 文件（模型权重、结果表）不从文件夹列出。行的提示写出帧号、由哪些文件组成和读取器的说明。
  右键 **Remove Source from List**（或列表中的 Delete / Backspace）移除该源的全部帧，原始文件不变；
  **Choose Frames…** 可重新选择要保留的帧。
  打开的文件夹自己没有帧、下一层有
  （探测器各写一个子文件夹时）就打开下一层，并提示打开了哪些子文件夹。
- **导入方法**：Open、Open Folder、拖放、Recent 和主页入口都在后台检查数据，可点 **Cancel Import**。
  多帧源出现 **Import Frames**：All Frames、Frame Range（首帧、末帧、Every）或 Specific Frames（逗号分隔的帧号）；
  每个源分别选择。界面帧号从 1 数。Data 的 **Choose Frames…** 修改已有选择，复杂的大范围可保留当前选择。
  导入只保存帧引用，虚拟列表不会为所有帧创建控件或一次解码整个堆栈；打开哪一帧才读取哪一帧。
  Build Map、Batch Export、Send Series to Fitting 处理已导入的帧，再应用各自的范围、间隔和 Sum N；
  不连续选择不会跨过未导入帧相加，真实帧号、时间和来源不重新编号。选择改变后清除旧热图和拟合预览。
  项目保存导入选择；All Frames 在监视模式下包含之后到达的新帧，显式选择保持原范围。
- 文件列表上方：源的组成（“11 module files · 240 frames · stitched”）、方向（“P03 Lambda: transposed and flipped (rotated
  by 90°)” 或 “Image as stored in the file”）和读取器的说明；缺了东西时用警告色：缺模块文件、模块帧数不同（只用每个模块
  都有的帧）、缺数据文件、文件还在写、写入程序没关文件就结束了（`h5clear -s`）、有几个图像数据集。只说明“怎样理解这些
  文件”的几条（同一模块位置、不属于这一组、帧的网格）不把 Data 步骤标成警告。
- 方向：Lambda 服务器写的文件（模块组；或探测器组有 `translation/distance`、形如 `1556X516` 的 `layout`、`description` /
  `type` / `local_name` 含 “lambda”）照旧做 P03 变换；其他 NeXus / HDF5 图像按存储原样读（第 0 行在上）。实际做的变换在
  `metadata["transformations"]`。
- 头文件的值按 `units` 换算（曝光 ms → s，能量 eV / keV，长度 mm / cm / m），不认识的单位不取；Data 步骤的
  “File header” 写出能量、距离和曝光（s）。无符号整数数据中该类型的最大值表示“无数据”，计入掩膜。
- 帧条：多帧的源在步骤上方显示 “Frame [4] of 12”（从 1 数）；Page Up / Page Down（或命令栏 ‹ ›）只走已导入的帧，到头再换源。
  导出文件名里的 `_frameNNNN` 仍从 0 数。
- 读不了时说原因：还没有帧、正被别的程序写着、被写入程序遗弃、压缩过滤器需要 `hdf5plugin`、要的帧超出（不会改读最后一帧）、
  文件损坏或不完整。新句子同时要放进 `frames.readable_read_error`、`texts.READ_ERRORS`、中文表和
  `tests/test_release_analyze_i18n.py`。
- 帧的文件（名称、修改时间、大小）变了，F5、换帧和监视文件夹都会重新读这一帧；没变就不读。

## 自动分析（不需要 AI）

Run Automatic Analysis（Results 步骤可写束线时笔记：αi、能量、像素尺寸、标定位置）：

- GIWAXS：几何（配置 / 附近的标定文件或标样图像）→ 序列取最后五分之一的帧求和（至多 10 帧；不到 10 帧只取最后一帧，
  `pipeline.default_sum`）→ 峰表 → 面内 / 面外 → 最强可靠环的取向与 Scherrer 尺寸。“与序列开头比较”用和末态同样多的帧。
  自动分析在 Data 步骤设的 Sum N 不带进 Series 热图、Batch Export 和 Send Series（对话框和 Series 标签页说明已关掉；
  要相加就自己在 Data ▸ Sum N 设）。
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
  Save ▾ ▸ Data and Fit… 保存拟合与参数）。数据在报告里（`peaks[].fit`、`peak_search`、`gisaxs.fit.solutions[]`、`gisaxs.fit.native`），
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
- **整个序列一条切线**（`frames_model.series_cut`，默认开）：切线是自动的时候，热图、Batch Export、Send Series 和监视文件夹
  自动导出的每一帧都用屏幕上这一帧的切线——GISAXS 为水平带的行和竖直切线的列，GIWAXS 为 I(χ) 的环窗口；相邻两行的
  差别于是来自样品，而不是切在哪里。按钮下一句写明（“Cut of the series — horizontal cut: rows 202–206 for every frame”，
  行号是带覆盖的像素行，两端都含）；Open 用同一条切线打开该行的帧。Series ▸ Options ▸ Find the cut again in every frame
  （Batch Export 对话框里也有同一个勾选框）改为每帧重新找；自己设的切线本来就每帧相同（“cuts as set by you, the same for
  every frame”）。状态存在设置 `analyze.pin_series_cuts`、设置文件和项目里（文件没写这一项时不改当前状态）。
- **真实 x**：每个点画在它测到的 x 上（只为显示重新取等距网格，至多 8192 列），缺口保持缺口、不连线；导出用测到的 x 和值。
- **帧轴**：真实帧号，从 1 数、整数刻度——一个多帧文件时是文件内的帧号，几个文件时是在列表中的位置。
  Options ▸ Frame axis ▸ Time (s)：帧带时间时按时间（每帧的时间戳；或帧号 × 曝光；几个多帧文件时为文件开始时间 +
  帧号 × 曝光），没有时间时说明。
  导出热图图片时，每行等高，刻度标注这一行的真实帧号或时间；跳帧或不等间隔采集的图片不表示实际时间间距。
  CSV 和旁边的 JSON 保留每行的真实帧号、时间和来源。
- **帧范围**：Options ▸ Frames 第一帧 to 最后一帧（“the last” 即列表的最后一帧）；不是全部帧时按钮下面和折叠标题都写出
  （“Build Map takes frames 3 to 5 only (Options).”）；Clear 恢复为全部。
- **两半**（只对 GISAXS 水平切线）：热图旁 ± / + / − / |x| / mean（两侧原样、一侧、两侧都画在 |x| 上、两侧平均）；
  阶段、轨迹和导出都跟随。
- **标题行**写出曲线、两半、帧数、切线、相加和时间的来源；建图之后改了设置或曲线下拉，警告色提示
  “Built with other settings — Build Map again.”。**Wide** 让 Series 标签页变宽（图像缩到最窄），
  热图下的帧曲线与序列轨迹始终上下排列，两个图都随页面宽度放大。
- 热图就是一个 DetectorView：Log、色图、色阶可调；读数显示帧、文件、x 与强度。
- 热图属于建图时的文件列表：之后再列出文件，热图保留并注明 “Map of the earlier list (N frames)”；Clear 清空列表时
  热图一并清除（正在建的图也停止）。监视文件夹期间，兼容当前设置的完整新帧会追加到图中，见下面的 Watch。
- 水平带选帧（下方第一幅图画出该帧曲线，Open 在 Analyze 中打开该帧），竖直带选 q 窗口（再下方画出强度随帧的变化，
  默认放在序列变化最显著的 q 处）；两条带都能拖动。
- **Results: batch fitting** 在 Series 下方：Batch Export 开启 Fitting 后，逐帧显示参数表、状态与失败原因；
  点击一行可看实际拟合点与模型叠图，Parameter trend 查看所选参数随真实帧号的变化。
  未开启拟合曲线文件导出也有 GUI 预览，预览使用同一次拟合的结果；q 为 Å⁻¹，尺寸为 nm。
  运行、暂停、停止和结束后的已完成结果均可查看。曲线点的缓存限制为 128 帧或 32 MiB，
  更早的点写入临时缓存，选择时在后台读取；Clear、改变导入选择、下一次拟合或关闭页面会清理缓存。
- Export ▾ 的文件按序列、曲线和两半命名（`bindings/series_exports.py`）：Map as CSV Table… `<序列>_<曲线>[_<两半>]_map.csv`
  （首行 x，首列帧号，每帧一行 `#` 注明文件）、Map as Figure… `…_map.png`、Trace as Figure / Table… `…_trace_<量>.csv`、
  Peak Table of Every Frame… `…_peaks.csv`、Stages as Table… `…_stages.csv`；所选帧的曲线用 Batch Export 给这一帧的名字
  `<帧>_<曲线>[_<两半>].csv / .png`（帧号从 0 数）。每个表的 `#` 头写出曲线、两半、帧、序列的切线、相加和帧号的含义，
  旁边写同名 `.json`（`application/series_map.series_record`：建图的设置、每行的文件、帧号、时间）。
- 序列轨迹图的下拉：Mean intensity（q 窗口内的平均强度随帧）或 Peak position / width / area / height（窗口内的峰：
  扣去直线背景后的质心、2.355 σ、面积、峰高，每帧一个值）；窗口落在缺口里或数据之外时没有值，图上说明，Trace as Table 拒绝；
  窗口比周围合并的 bin 还窄时取最近的点并注明。曲线下拉里有每个区域的 I(q) 和 I(χ)，所以原位序列可以按任意区域看。

## 束流中心

- 默认用仪器配置里标定过的中心，不受文件头影响（设置 ▸ Analyze 可改为信任文件头）。
- Geometry 步骤 ▸ Beam centre：拖动图上的青色十字、Pick on Image、Enter Coordinates…、Use File Header Centre、
  Refine x by Symmetry / Make Left and Right Symmetric（GISAXS：先在当前水平切带的有效数据中全局粗定位左右对称轴，
  再精修，只改 x；列强度与显示曲线一致，取有效像素均值。精修按离中心距离的倒数加权，使不同倍数的距离区间获得
  相近权重，避免远端大量点压过低 q 点；这近似对应 GISAXS 的对数 |qy| 分布。候选中心使用同一组实测配对点，
  不跨缝隙插值；离中心不足 1 px 的点不参与。精修停止容差 0.0001 px，不代表实验定位精度）。
  搜索在后台运行，同一按钮可取消；
  数据覆盖不足或存在多个同样可靠的轴时保留原中心并说明原因。改动对之后的文件保持；Save to Profile 写入配置，
  Back to Profile Centre 放弃。
- 本次会话设的中心只用于同一探测器尺寸（行 × 列）的帧；尺寸不同的帧回到配置的中心。尺寸随中心写入设置文件与项目
  （`beam_center_shape_px`）。
- Geometry… 和 Enter Coordinates… 的窗口保持打开，不阻挡曲线操作。中心 x/y 显示四位小数（Beam centre 按钮和状态消息也是四位），Centre step 设置上下按钮、
  方向键和滚轮每次移动的像素步长（最小 0.0001 px，两个窗口共用并记住）。Apply Centre 应用本次会话的中心并刷新曲线，
  可反复调整和应用；Save Profile 保存全部几何参数，窗口仍保持打开。Close 保留已应用的中心，放弃尚未应用的输入。
  勾选 Live Preview 后，输入中心、上下按钮或滚轮调整会自动刷新曲线，步长仍由 Centre step 设置。
  连续移动合并刷新；当前计算完成后只计算最新的中心，保留曲线的手动缩放。关闭预览或窗口会取消尚未提交的刷新。
  切换文件或帧、清空和关闭页面时关闭调整窗口，避免将旧帧的输入应用到另一个探测器。

## αi 与模式

- 命令栏的 αi：没有输入时显示 “αi from profile”；为本次会话输入的值以橙色标出，右键 ▸ Back to Profile 回到配置的值。
- 模式固定为 GISAXS 或 GIWAXS 而帧看起来是另一种时，状态行会提醒；Auto 每帧重新判断。

## 送到 Fitting

Send to Fitting（命令栏和 Export 步骤的按钮都走 `bindings/fit_handover.py`）把完整的带符号水平切线（GIWAXS 为 I(q)）写到
`gimap_analysis/<stem>_fit_input.dat`（`q I sigma pixels`，q 单位 Å⁻¹；`# observation:` 记录来源、gap guard、相加帧数，
选了平均或一半时还有 `"halves"`，多帧的源有 `frame_index`、`frame_count`；`#` 行写出帧 “frame 4 of 12 (index 3)”、相加和文件）
并在 Fitting 中打开；文件先写成 `.<名称>.part` 再改名，监视的文件夹不会读到写了一半的文件。下拉选择两半的用法
（Both halves on |qy| / Mean / qy < 0 / qy > 0），与 Cuts 步骤的 Halves 同步。数据旁边写不了时询问另存到哪里，写成功的
文件夹本次会话记住（`write_elsewhere`，每次最多问两次）。Send Series to Fitting… 见下面的导出一节。

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
  亮得多的像素（例：一台 Lambda 探测器上有两行像素的读数比图样高几个数量级，成对出现，孤立像素检测看不到）。
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

  选表格与逐帧曲线时，在列表中勾选要哪些曲线（All / None）。表格（`application/batch_tables.common_x`，8 位有效数字）：
  各帧的点完全相同就并排；只差舍入或缺几列（缝隙、掩膜）时每个点放在自己 x 的行上，不插值；网格真的不同时才线性插值到
  第一个有点的帧的网格（跨过它自己的缺口按步长补齐），不跨缺口、不外推，没有任何帧有值的行不写；表头第三行写明是哪一种。
  文件夹里总会写 `README.txt`（每种文件是什么、曲线键对应的名称、单位、切线、相加、失败的帧）和 `<名称>_batch.json`
  （`gimap_analyze_batch_2`：选项、帧列表与帧号、失败的帧和原因、写出的每个文件、序列的切线和每帧的切线、来源、
  之前的导出被替换还是保留、有模型拟合时的 `model_fit` 与 `fit_failed`、第一帧的完整设置）。
- 名称：文件名里的帧号从 0 数（`<扫描>_frame0003_sum3_…`，`application/export_record.frame_stem`；单帧文件没有 `_frame`）；
  表头、标签、图标题从 1 数（“scan_00007 frame 4 of 12”）。模块组按扫描名命名，不带 `_m01`，记录里列出每个模块文件。
  每个曲线文件的 `#` 行写出 `# frame: 4 of 12 (index 3)`、相加的帧和读取的文件。
- 对话框在开始之前说：帧数、曲线、切线和两半、大约多大多久（`batch_summary.py`）。Cuts 一行写出每帧用的切线，旁边是
  Find the cut again in every frame；屏幕上的帧在去掉自动分析的 Sum 重新读的时候，这一行说 “reading”，Export 等它读完。
  写不了的文件夹立即说明并关掉 Export；上次批处理的文件夹若属于别的数据会注明，Next to the Data 回到数据旁边。
  文件夹里已有同一数据的导出（或拟合用曲线）时，Export 问 Replace / Keep Both in a New Subfolder / Cancel：只有 Replace
  删掉的每一种文件这次都会重写时，回车才是 Replace，否则回车是 Keep Both，问题里写出 Replace 会删掉哪些种类的文件。
  只删目标文件夹里的文件（记录里的 `../`、绝对路径和指向外面的链接都不碰）。
- 结束后状态行与面板说：导出了几帧、写到哪里、用时，模型拟合成功几帧、其余为什么（最多三种原因），失败的帧逐行列出；
  有帧失败、有拟合没有解或被停止时不以成功结束（警告色；拟合失败时 toast 可打开文件夹）。
- 批量拟合（可选，默认不拟合）：
  - GIWAXS——“环区域中的峰”：每个有 q 窗口的区域（切线 ▸ 环 / 斑点）是一个峰；在该区域的 χ 范围内，按探测器 q 步长取
    I(q)，窗口为区域窗口两侧各加一个窗口宽度（背景才能确定），拟合 Pseudo-Voigt / 高斯 / 洛伦兹 + 直线背景（加权最小二乘，
    误差按 √χ²ᵣ 放大）。表中每帧一行、每个峰：q ± 误差、d = 2π/q、FWHM ± 误差、面积 ± 误差、峰高、η、χ²ᵣ、说明
    （不显著、太宽、点太少时写原因，该帧不影响下一帧的起点）。
  - GISAXS——“水平切线的颗粒模型”：带符号的水平切线按 Halves of the cut（对话框 *Curve for Fitting* 组：两半的平均 /
    q > 0 半边 / q < 0 半边，`BatchChoices.fit_halves`）交给 Fitting 的快速物理拟合（球 / 随机取向圆柱 / 竖直圆柱，含尺寸分布与
    间距 D；模型可选自动比较或指定一种；多于 1000 个点时合并相邻点），表中每帧一行：`halves`、`points`、模型、χ²、log RMSE 与
    各组分参数（nm）。约半分钟一帧。两半永远不一起拟合。下拉的初值：Cuts 步骤选的是平均或某一半时用它，否则用上次批处理的选择。
    所选的一半没有点（“No points on the q > 0 half”）、整条切线没有大于 0 的强度时，这一帧不拟合，原因写进表、记录和面板；
    选平均而某帧只有一半时拟合那一半，结束时说 “N frame(s) on one half alone”。拟合用曲线文件仍含两半（带符号 qy），所选的
    两半写在 `# observation:` 的 `"halves"` 里。
  - 起始值：上一帧的结果（缓慢变化的序列，峰或颗粒随之移动；GISAXS 取上一帧的模型与 D）/ 第一帧的结果（所有帧同一起点，
    帧之间互不影响）/ 每帧重新寻找。
  - “在当前帧上试拟合”：用这些选项拟合当前显示的帧，结果直接显示在对话框中，满意再批量。
  - 可选同时保存每帧的拟合曲线 `fits/<帧>_<曲线>_fit.csv`（峰）或 `fits/<帧>_model_fit.csv`（颗粒模型：拟合用到的点的
    q (1/nm)、I、σ 与拟合，注释写出两半和准备说明）；拟合值随帧的变化图 `<名称>_peak_fits.png` / `_model_fits.png`。
    高斯和洛伦兹峰的 η 固定为 0 / 1，不参与拟合。
  - 更复杂的拟合（Fitting 的 AI 辅助模型、每帧各自的输入选择）：Send Series to Fitting… → 拟合 ▸ 原位序列。
- 逐帧在后台处理，状态行显示进度、剩余时间和是否在拟合，可取消（已写的保留，表格包含已处理的帧）。选项、格式和文件夹会
  记住（设置节 `analyze_batch`），下次打开对话框即是上次的选择。只需要 q 图或 q 图图片时才计算 q 图（P03 Lambda 9M 约 2 s/帧）。
- 开始页 Batch Export…：选一个原始数据文件夹 → 自动加载并分析第一帧（几何按探测器匹配，或设置文件中的配置）→
  直接弹出 Batch Export 对话框 → Export。
- Save Settings… / Load Settings…（Export ▾ 和对话框中）：`{"format": "gimap-analyze-settings"}` JSON，包含模式、仪器配置
  （名称和几何，换电脑也能用；缺失时自动加入）、αi、中心、求和、校正（缺陷像素、掩膜形状与文件、镜像补齐、有效范围、背景）、
  GIWAXS 切线（条带宽度、bins、x 轴、环窗口、扇区、q 框、区域）、GISAXS 切线和导出选项。载入后所有控件同步；找不到的
  掩膜或背景文件会列出并忽略。
- Send Series to Fitting…：同一个对话框的拟合模式（只写每帧的拟合输入和 Halves of the cut），总是写进自己的子文件夹
  `<名称>_fitting/`（`application/batch_folder.folder_name`；Keep Both 时 `<名称>_fitting_2`），与同一数据的 Batch Export
  不共用 README 和记录；完成后在拟合 ▸ 原位序列中打开 `<名称>_fitting/fit_input`。它的两半选择也会记住。
- Watch：每 2 秒看一次文件夹（在后台线程上看，`folder_watch.py`、`sources_model.py`）。新文件列出并显示最新的；已列出的源
  多了帧，同样逐帧处理；Export ▸ Export New Frames While Watching 时每个完整的帧导出一次。
  - **实时热图**（`bindings/watch_series.py`）：已有的 Series 图继续追加完整新帧，不需要再点 Build Map；开启自动导出时，
    还没有图则用所选曲线自动建立。自动导出和热图行复用同一次处理结果；关闭自动导出时，已有图仍在后台处理新帧。
    启动监视时新列出的已有文件不当作新测量帧；旧图会提示来自之前的列表，可 Build Map 纳入这些帧。
    没有原始曲线行或曲线来源不兼容的历史图保留提示。首帧分析结束后立即处理等候的导出，不依赖下一次轮询。
    沿用建图时的切线、Sum N、每 n 帧和帧范围；改变几何、处理设置或曲线后提示重新建图，旧任务不混入新图。
    带参考峰原位校正的 Batch Export 图会提示重新建图，不追加常规监视处理的新行；避免把已校正和未校正的数据混在一起。
    点选热图行会停止跟随最新行，并保留选中的行、q 窗口和缩放；Series ▸ Options ▸ Show the newest frame 可以恢复跟随。
    Stop Watch、Clear 和关闭页面丢弃迟到的热图结果。
  - **相加的末尾短组**：新帧补齐原分组后，重新处理这一组并替换短组行，分组起点不移动。自动导出的完整组可包含开始
    监视前已在短组中的帧，文件名里的 `_sumN` 和 JSON 都记录实际相加的帧；已经完成的完整组不重复导出。
  - **边写边跟**：每次看都在变的源（探测器一秒一帧写进一个文件或一组模块文件）在增长时就跟着：每次只交出上一次看时
    已经在文件里的帧，最新的一帧等下一次（或文件不再变）；Data 步骤写出 “Still growing: frame N is the newest complete one”。
    这期间开始的 Series 热图和 Batch Export 也只取已交出的帧（`sources_model.frames_to_run`）。
  - 模块组完整且静止约 10 s（5 次）后才导出；同一文件夹别的扫描有同样多的模块文件时不必等，较少时等 15 次。缺模块文件
    或模块文件还读不了时不导出，并在状态行说明（只说一次）。
  - 被别的程序占着写的文件每次再问；写入程序没关文件就结束了时，状态行用警告色说明怎样修复。master 晚于数据文件到达时
    合成一行；Clear / Remove 之后又增长的源重新列出。

## 界面语言

设置 ▸ Appearance ▸ Language，或菜单 View ▸ Language：English（默认）/ 中文。命令、步骤、按钮、菜单、表头、数值框的单位后缀和运行中生成的句子
切换为中文；切换语言时当前页面的步骤说明、图标题和状态行随之重写。数值、单位、文件名，以及作为数据的表头（序列名、
参数名）不翻译。

## 手动验收清单

1. 开始页 ▸ Open Files…：选完文件立即跳到 Analyze，Data 步骤显示文件名、探测器、帧数，图像随后出现。
2. GISAXS 帧（例如 `tests/data/external/gisaxs_galaxi/`，配 `galaxi_bornagain.poni` 的几何）▸ Run Automatic Analysis：
   Results 显示 Yoneda 切线、两半的理由、间距（约 55.6 nm，shoulder）、拟合表，并提示选择模型；Save Fitted Curve… 有 toast。
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
10. Open Folder `TestSAXSdata/lmbdp03/`：11 个模块文件只占一行（扫描名，不带 `_m01`，提示里列出 11 个文件）；Data 步骤写
    “11 module files · 1 frame · stitched” 和 “P03 Lambda: transposed and flipped (rotated by 90°)”，File header 的曝光以 s 计。
    在临时目录的副本里删掉一个模块文件再打开：警告色说缺哪个模块，Data 步骤标为警告。
11. Format Converter 写出的多帧 HDF5 容器（或任何 `(N, 行, 列)` 的 `.h5`）：方向与写出时相同，Data 步骤写 “Image as stored in the
    file”；步骤上方出现 “Frame [1] of N”（帧号框最大为 N），Page Down 先走帧再换文件。
12. 一个多帧序列 ▸ Series ▸ Build Map：帧轴从 1 起、整数刻度；按钮下写出序列的切线（“Cut of the series — …”）；Options ▸ Frames
    3 to 5 后按钮下和折叠标题写出范围，Clear 后回到全部；Frame axis ▸ Time (s)（帧没有时间时说明）；GISAXS 水平切线时热图旁
    有 ± / + / − / |x| / mean，选 + 后导出名带 `_positive`；勾选 Find the cut again in every frame 后标题行提示
    “Built with other settings — Build Map again.”；Export ▸ Map as CSV Table… 旁边有同名 `.json`。
13. 一个 ≥ 10 帧的序列先 Run Automatic Analysis，再 Batch Export…：对话框说自动分析的 Sum 已关掉，Cuts 一行先说 reading、读完
    写出行号；GISAXS 时 *Curve for Fitting* 组有 Halves of the cut。导出到同一文件夹再导一次：问 Replace / Keep Both，回车为
    Replace；只导表格到存着图片的文件夹时回车为 Keep Both。之后 Send Series to Fitting… 写进 `<名称>_fitting/`，刚才的导出
    原样还在，Fitting ▸ In-situ series 打开它的 `fit_input/`。文件名带 `_frame0003`，表头和图标题写 “frame 4 of 12”。
14. Watch 一个临时文件夹，另一个进程每秒往一个 HDF5 文件追加一帧：文件出现后列出并跟着增长，Data 步骤写
    “Still growing: frame N is the newest complete one”；停止追加后最后一帧也交出；勾选 Export New Frames While Watching
    时每帧只导出一次。
