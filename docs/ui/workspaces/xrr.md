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
关闭该工具都不改变主窗口当前的 Fitting、Prediction、Trainset、Classification 或 WAXS 页面。

左侧参数区只有一个垂直滚动容器；ROI、Run、Export 和 JobStatus 位于固定命令区，在
1280×800、1440×900 和 1920×1080 下保持可见。右侧 `Live frame` 与 `XRR points` 标签位置稳定。
worker progress 同时刷新当前 detector/ROI 和累计曲线，但不会调用 `setCurrentIndex`，因此用户查看
曲线或表格时不会被每帧刷新拉回 detector。

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
