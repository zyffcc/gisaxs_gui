# XRR Series Extractor 界面与交互说明

- **Status**: Current
- **Scope**: Tools 中 XRR series 提取窗口的控件映射、导航状态和手动验收
- **Related code**:
  [`src/gimap/features/xrr/presentation/`](../../../src/gimap/features/xrr/presentation/)、
  [`xrr_series_dialog_view.py`](../../../src/gimap/features/xrr/presentation/views/xrr_series_dialog_view.py)、
  [`pixel_coordinates.py`](../../../src/gimap/features/xrr/presentation/pixel_coordinates.py)（窗口里的像素坐标）
- **Related tests**:
  [`tests/test_xrr_presentation.py`](../../../tests/test_xrr_presentation.py)、
  [`tests/test_xrr_nexus_series.py`](../../../tests/test_xrr_nexus_series.py)、
  [`tests/test_xrr_series_record.py`](../../../tests/test_xrr_series_record.py)、
  [`tests/test_release2_xrr_window.py`](../../../tests/test_release2_xrr_window.py)（窗口里的像素坐标）、
  [`tests/test_ui_source_of_truth.py`](../../../tests/test_ui_source_of_truth.py)
- **Last verified**: 2026-10-14（数据来源与记录一节对照代码；像素坐标一节 2026-10-13；其余沿用 2026-08-25）

## 当前状态

通过 **Tools → XRR Series Extractor…** 或 `Ctrl+Shift+R` 打开独立、非模态窗口。打开、处理和
关闭该工具都不改变主窗口当前的页面（Analyze、Fitting、Compare、2D Prediction 或 Trainset Build）。

左侧参数区只有一个垂直滚动容器；ROI、Run、Export 和 JobStatus 位于固定命令区，在
1280×800、1440×900 和 1920×1080 下保持可见。右侧 `Live frame` 与 `XRR points` 标签位置稳定。
worker progress 同时刷新当前 detector/ROI 和累计曲线，但不会调用 `setCurrentIndex`，因此用户查看
曲线或表格时不会被每帧刷新拉回 detector。

几何的来源（2026-10-06）：窗口打开时用上次**应用**的几何标定（距离、能量、像素、中心）预填，字段旁标
“(from last calibration)”，并显示 “Last calibration: <图像> · <时间>”；标定针对的帧尺寸与序列不同时只用它的能量并说明。
Load first frame 不会覆盖你输入或点选的值：文件中的值与输入不同时只在摘要里写 “The file says: …”；文件没有中心时
用图像中点（标为 image center，警告色）。摘要按表单顺序列出哪些值来自文件、上次标定、图像中点或内置默认值。

导出：默认文件名 `<源文件夹>/<源文件名>_xrr.csv`；旁边写同名 `.json` 记录（序列、θ 起点与步长或 NXS 数据集、实际用到的
距离 / 能量 / 波长 / 像素 / 中心 / 方向及各自来源、ROI 半径与聚合方式、列的含义、qz 与 ROI 中心的公式）；完成后 toast 可打开文件夹。
qz 的计算不变。可以把 NXS / CBF 文件或 CBF 文件夹拖进窗口（自动设置数据类型并载入第一帧）；Browse… 从当前序列或上次的文件夹开始。

数据来源（2026-10-14，`infrastructure/adapters/detector_series.py`，经共用的 NeXus 读取器）：Browse… 选一个 NeXus / HDF5
文件（任何 HDF5 后缀）或 CBF 文件，旁边的 **Folder…** 选序列所在的文件夹（其中的 NeXus / HDF5 文件逐个读，或其中符合
pattern 的 CBF）；所选文件读不了时数据类型自动换成能读的。一组模块文件每帧拼成一张；一个文件有多帧时每帧一个点；一个文件
只有一帧时，同一文件夹里编号相同方式的单帧文件（同后缀、同帧尺寸）就是这个序列，编号像序列但对不上的文件（帧尺寸不同、
不是一帧）被列出并排除（按起点和步长给角度时，排除的文件之后的帧按位置取角度）。模块帧数不同时只用每个模块都有的帧，
并说明。导出的 `.json` 记录多了 `source_layout`：组成的说明、源和帧的个数、读到的每个文件、读取器的警告；运行结束的状态行
在 “Extracted N XRR point(s).” 之后附上这些警告。

窗口顶部有标题；图跟随浅色 / 深色主题（保存的图用浅色）。Esc 关闭窗口，提取运行中会先询问；
界面文字（按钮、空状态、下拉项）有中文；表头中的量（`Theta (°)`、`ROI x`）、数值、单位、文件名与
NXS 数据集路径不翻译。

## 像素坐标（2026-10-13）

窗口里所有位置都用规范像素坐标，与 Analyze 和 Geometry Calibration 相同：像素角点在整数上，第 j 个像素覆盖 [j, j+1]，
第一个像素的中心是 (0.5, 0.5)，x 从左边、y 从上边数。这包括 Beam center X / Y 两栏（不论来自上次标定、文件、图像中点还是输入）、
Pick、预览图的坐标轴和像素位置、十字和 ROI 圆、结果表的 ROI x / ROI y 两列，以及摘要里 “The file says … / The last calibration says …”
的中心。同一个标定在三个窗口里显示同样的数，可以直接对照或抄写。

- 提取内部（`SpecularGeometry.beam_center_*_px`、`XrrPoint.roi_center_*_px`）、设置里的 `detector.beam_center_x / _y`、
  导出的 CSV（`roi_center_x_px / _y_px`）和 JSON 记录（`geometry.beam_center_x_px / _y_px`）仍按“像素中心从 0 数”，
  比窗口里的数小 0.5 px。JSON 记录在顶层的 `center_convention`（`name` 是 `index`，和导出的标定文件用同一个键名）里写明；
  没有这个键的旧记录含义相同。提取本身没有变：同一个物理位置得到的 ROI 像素、强度、角度和 qz 与以前相同，CSV 与以前逐字节相同。
  程序不会把导出的 CSV 或记录再读回来。
- 屏幕上的数和提取用的数只在 `presentation/pixel_coordinates.py` 换算（`shown_coordinate`、`shown_roi_center`、
  `shown_image_center`、`extraction_coordinate`、`image_extent`），算式在 `shared/geometry/conventions.py`。
  新加一处显示或输入位置的控件时走这些函数。中心两栏和 ROI x / ROI y 的表头有提示，说明是哪一套像素。
- 图像中点是 (列数/2, 行数/2)，例如 981 × 1043（列 × 行）的帧是 490.50 / 521.50；和以前的 `(W−1)/2` 是同一个点。
- Pick 得到的就是点到的位置：在一个像素里任何地方点一下，半径 0 时读到的就是这个像素；十字和 ROI 圆画在数值所指的位置，
  圆里的像素中心就是被读的像素。以前（窗口按像素中心从 0 数、预览图的坐标轴却是 0…W）提取用的是点击处右下方半个像素的位置。
- 以前的版本里手工输入过的中心，现在要在 x 和 y 上各加 0.5 px 才是同一个点；从上次标定带过来的中心和图像中点由窗口换算，不用改。
- 中心两栏只有两位小数（Analyze 的 Beam centre 和 Enter Coordinates 显示四位，Geometry Calibration 的结果三位：同一个数，位数不同）。
  两栏把 Analyze 持有的数（索引 + 0.5）按两位小数的 Qt 输入框取舍；标定存下的中心恰好落在两个刻度正中（x.xx5）时，
  送去提取的数可能与以前差 0.01 px。
- 没有载入帧、也没有可用的标定时，两栏显示内置的 0.00 / 0.00（标 built-in default），现在指的是帧的左上角。

## 已知限制：NeXus 头里的光束中心（2026-10-13）

不是这一轮改出来的，是核对文档时确认的。NeXus 头里的 `beam_center_x / _y` 仍原样当作载入后那一帧的（列, 行）使用
（`bindings/geometry_sources.py` 读 `metadata["beam_center_x_px"] / ["beam_center_y_px"]`），只是显示时加 0.5；
载入器对图像做的转置和上下翻转没有用到它上面。Analyze 对单模块 NeXus 的头中心是按转置和翻转换算的
（`shared/detector_io/loading.py` 的 `header_beam_center`；多模块拼接时 Analyze 不给头中心）。

- 后果：文件写了头中心时，两栏标 “from file”，并且盖掉上次标定的中心（摘要里写 “The last calibration says: …”），
  而这个位置可能离直射束很远。用一个编造的 NeXus 文件核对过：头指向存储图像里的一个像素，载入后 Analyze 的头中心落在这个像素的角上，
  XRR 窗口读到的是另一个像素。公开数据里的 NXS 没有头中心，没有用真实数据核对过；CBF 序列不受影响（CBF 头不用来取中心）。
- 另外差半个像素：XRR 把头里的数当作“像素中心从 0 数”，Analyze 把同一个数当作规范坐标。
- 改任何一边都会改变提取结果，由用户决定。手册第 11 节和 `CHANGELOG.md` 的已知限制写了这条；
  `tests/test_release2_docs_decisions.py` 里有测试用编造的文件复现它：改好之后那项测试会失败，那时删掉这三处的限制。

## 控件映射

| 用户任务 | View object | 行为 |
| --- | --- | --- |
| 选择 NeXus / HDF5 文件、CBF 文件或序列文件夹 | `source_picker`（其中的 Browse…）、`source_folder_button`（Folder…） | QFileDialog 位于 dialog；实际 discover/load 经 application port |
| 指定 source type 与 CBF glob | `source_kind_combo`、`pattern_edit` | Auto/NXS/CBF；CBF 使用自然排序 |
| 读取首帧 | `inspect_button` | QThread 中加载一帧，回填可用 metadata 和 detector preview |
| 指定样品角序列 | `angle_mode_combo`、`theta_start_spin`、`theta_step_spin`、`angle_dataset_edit` | 线性序列或 NXS motor dataset |
| 设置固定 geometry | `distance_spin`、`energy_spin`、`pixel_x_spin`、`pixel_y_spin` | 单位分别为 mm、keV、µm、µm |
| 设置 direct-beam center | `center_x_spin`、`center_y_spin`、`pick_center_button` | 显式 Pick command；点击 detector，Esc 取消 |
| 设置反射方向 | `direction_combo` | detector y 向上或向下移动 |
| 定义 ROI intensity | `radius_spin`、`aggregation_combo` | radius 0 单点；圆形邻域 Sum/Mean |
| 运行、取消和导出 | `run_button`、`job_status`、`export_button` | 独立进程逐帧处理；CSV 由 infrastructure adapter 写入 |
| 查看当前 detector/ROI | `live_panel` | 只显示有界降采样投影；ROI 坐标来自全分辨率计算 |
| 查看曲线和逐点表 | `curve_panel`、`results_table` | 可选 log intensity；刷新不改变当前 tab |

所有数值框和下拉框安装 safe-wheel behavior。普通滚轮滚动左侧参数区，只有控件已聚焦且按住
Alt/Option 时才会改变数值。

## 手动验收清单

- [ ] Tools 菜单和 `Ctrl+Shift+R` 只打开一个 modeless XRR window，主 workspace 不跳转；
- [ ] 选择任一 `_mN.nxs` 后显示正确 frame count，每个 frame 只形成一个拼接 detector image；
- [ ] Folder… 选一个单帧 NeXus 文件的文件夹：每个文件一个点；编号相同但帧尺寸不同的文件被列出并排除；导出的 JSON 有
      `source_layout`，列出读到的每个文件；
- [ ] CBF folder/pattern 使用自然排序并且每个 CBF 形成一个 point；
- [ ] Linear 与 NXS motor dataset 两种 theta source 的值和单位正确；
- [ ] Load first frame 回填有效 energy、distance、pixel size、beam center metadata；缺失 center 时使用图像中心
      (列数/2, 行数/2)；
- [ ] Pick direct-beam center 显示选中态、cross cursor，点击回填 x/y（就是点到的位置；半径 0 时读到的是点到的那个像素），
      Esc 可取消；
- [ ] Apply 过的标定：Geometry Calibration、Analyze 和这个窗口对同一个中心显示同样的数（不差 0.5 px）；
- [ ] up/down direction、distance、pixel Y 和 theta 对 ROI 轨迹的影响符合 scientific contract；
- [ ] radius 0、circle Sum、circle Mean 和 invalid/masked pixels 结果正确；
- [ ] Run 后每帧显示 detector、ROI、theta 和进度，GUI 保持响应，Cancel 可停止；
- [ ] 在运行中切换到 XRR points 后，后续 frame progress 不切回 Live frame；
- [ ] Log intensity 只改变曲线显示，不改变 point/CSV；
- [ ] CSV 包含 theta、qz、intensity、ROI center 和 valid-pixel count；CSV 的 ROI center 比表格里的小 0.5 px，
      JSON 记录有 `center_convention`；
- [ ] 1280×800、1440×900、1920×1080 下 Run/Cancel 始终可见，无水平裁切和双重滚动。
