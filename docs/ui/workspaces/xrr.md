# XRR Series Extractor 界面与交互说明

- **Status**: Current
- **Scope**: Tools 中 XRR series 提取窗口的控件映射、导航状态和手动验收
- **Related code**:
  [`src/gimap/features/xrr/presentation/`](../../../src/gimap/features/xrr/presentation/)、
  [`xrr_series_dialog_view.py`](../../../src/gimap/features/xrr/presentation/views/xrr_series_dialog_view.py)
- **Related tests**:
  [`tests/test_xrr_presentation.py`](../../../tests/test_xrr_presentation.py)、
  [`tests/test_ui_source_of_truth.py`](../../../tests/test_ui_source_of_truth.py)
- **Last verified**: 2026-08-25

## 当前状态

通过 **Tools → XRR Series Extractor…** 或 `Ctrl+Shift+R` 打开独立、非模态窗口。打开、处理和
关闭该工具都不改变主窗口当前的 Fitting、Compare、Prediction、Trainset 或 WAXS 页面。

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

窗口顶部有标题；图跟随浅色 / 深色主题（保存的图用浅色）。Esc 关闭窗口，提取运行中会先询问；
界面文字（按钮、空状态、下拉项）有中文；表头中的量（`Theta (°)`、`ROI x`）、数值、单位、文件名与
NXS 数据集路径不翻译。

## 控件映射

| 用户任务 | View object | 行为 |
| --- | --- | --- |
| 选择 NXS module 或 CBF file/folder | `source_picker` | QFileDialog 位于 dialog；实际 discover/load 经 application port |
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
- [ ] CBF folder/pattern 使用自然排序并且每个 CBF 形成一个 point；
- [ ] Linear 与 NXS motor dataset 两种 theta source 的值和单位正确；
- [ ] Load first frame 回填有效 energy、distance、pixel size、beam center metadata；缺失 center 时使用图像中心；
- [ ] Pick direct-beam center 显示选中态、cross cursor，点击回填 x/y，Esc 可取消；
- [ ] up/down direction、distance、pixel Y 和 theta 对 ROI 轨迹的影响符合 scientific contract；
- [ ] radius 0、circle Sum、circle Mean 和 invalid/masked pixels 结果正确；
- [ ] Run 后每帧显示 detector、ROI、theta 和进度，GUI 保持响应，Cancel 可停止；
- [ ] 在运行中切换到 XRR points 后，后续 frame progress 不切回 Live frame；
- [ ] Log intensity 只改变曲线显示，不改变 point/CSV；
- [ ] CSV 包含 theta、qz、intensity、ROI center 和 valid-pixel count；
- [ ] 1280×800、1440×900、1920×1080 下 Run/Cancel 始终可见，无水平裁切和双重滚动。
