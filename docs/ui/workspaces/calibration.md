# Calibration 界面与交互说明

- **Status**: Current
- **Scope**: Geometry Calibration 的 PyQt presentation 所有权、控件映射与手动验收
- **Related code**:
  [`src/gimap/features/calibration/presentation/`](../../../src/gimap/features/calibration/presentation/)、
  [`geometry_calibration_dialog_view.py`](../../../src/gimap/features/calibration/presentation/views/geometry_calibration_dialog_view.py)
- **Related tests**:
  [`tests/test_calibration_presentation.py`](../../../tests/test_calibration_presentation.py)、
  [`tests/test_calibration_feature.py`](../../../tests/test_calibration_feature.py)、
  [`tests/test_calibration.py`](../../../tests/test_calibration.py)
- **Last verified**: 2026-08-18

## 当前状态

Geometry Calibration 的静态 widget hierarchy、splitter、objectName、tab order 和默认控件值
以 feature-owned `presentation/views/geometry_calibration_dialog_view.py` 为唯一来源。
`presentation/dialog.py` 只绑定
ViewModel、signals 和运行时组件：

```text
PyQt Dialog → CalibrationViewModel → application use cases → ports
                              └────→ pure domain rules
```

Matplotlib canvas/toolbar 使用 Python View 中的 `calibrationToolbarHost` 与
`calibrationFigureHost` 在运行时安装；标准品和 detector model 的选项由 ViewModel 动态填充。
Qt signals、`QFileDialog` 和 `QMessageBox` 留在 dialog。路径规范化
通过 application port；standard detection、理论环 geometry、manual refinement 和显著差异
阈值位于 domain，并由 ViewModel commands 调用。

旧的 `ui/geometry_calibration_dialog.py` 与 `calibration/*.py` 兼容别名已删除；调用方直接导入
`src.gimap.features.calibration.presentation.dialog`。

使用上的要点（2026-10-06）：

- Auto Calibration、Cancel 与任务状态固定在左栏底部，不随表单滚走；预览的工具栏在窄窗口时移到自己的一行。
- 工具栏的缩放 / 平移打开时，点击预览不会移动手动精修的点。
- “Advanced manual refinement” 只展开或收起这一节；手动模式只由 Manual refine 或 “Use manual values”
  开启。Reset to fitted 回到拟合得到的解（导出或应用之后也可以）；只有改过的手动值才写入并标为手动调整。
- 预览的空状态是界面文字（中文不会显示成方框）；图跟随浅色 / 深色主题，Save 写出的图用浅色。
- Esc 关闭窗口；运行中按 Esc 会先询问，重新打开的窗口不会自己关闭。从 Analyze 打开的标定窗口关闭后释放。
- Apply 只出一个 toast（中心、距离，以及“已保存为仪器配置 …，分析会自动使用”），不再另弹消息框；覆盖已有配置前仍会询问。
  Apply 同时记下标定图像的尺寸与探测器（XRR 据此判断能否沿用）。Export 后的 toast 可打开文件夹；toast 显示在右侧，不挡按钮。
- 可以拖入探测器图像（.nxs / .cbf / .tif / .edf，载入）或导出的 .json（导入）；Open… / Import… 从当前图像或上次的文件夹开始。

## 控件映射

| 功能/控件区域 | Python View object / 当前位置 | 行为 |
| --- | --- | --- |
| Calibration image path/Open、energy、standard、distance、detector | `calibration_input_section` | loader、detector auto-detection 和 defaults 不变 |
| pixel X/Y、custom range、background、log/mask/rings | `calibration_advanced_section` | 默认折叠；metadata 缺失或 Custom detector 时自动展开 |
| Auto Calibration、Cancel、progress、stage label | `calibration_run_section` + `job_status` | 原 worker、cancel boundary 和 progress callbacks 不变 |
| Matplotlib toolbar/canvas、overlay actions/legend | `calibration_preview_panel` + dynamic hosts | ring/center rendering、zoom 和 drag behavior 不变 |
| selected solution labels、candidate table | `calibration_results_section` | candidate order、confidence 和 residual 不变 |
| manual center/distance/ring fitting | `calibration_manual_section` | 与原 Manual refine toggle 同步，折叠不重置值 |
| Import/Export Calibration、Apply、Close | `calibration_export_section` | JSON format、AppContext 写入和 main-window sync 不变 |

所有原有显式 `objectName` 均保持不变。旧实现没有设置按钮快捷键，本次也没有新增或改变；
`calibrationApplied` signal、CBF/NXS filter、错误标题/文案和 JSON v1 schema 保持原行为。

## 手动验收清单

- [ ] Open CBF/NXS、粘贴路径和 ambiguous NXS dataset 选择正常；
- [ ] energy、standard、estimated distance、range 和 detector defaults 与配置契约一致；
- [ ] 无 pixel metadata 或选择 Custom detector 时 Advanced 自动展开；
- [ ] Advanced 折叠/展开不改变 pixel、distance bounds 或 overlay toggles；
- [ ] Preview 保持原 image orientation、log display、mask 和 ring colors；
- [ ] Auto Calibration 可启动，JobStatus 显示 stage/progress；
- [ ] Cancel 在当前 numerical step 后安全停止；
- [ ] candidate table 选择可更新 overlays 和 result labels；
- [ ] Reset view、Clean image、Focus image、Manual refine 正常；
- [ ] 手动拖动 center、编辑 distance、Fit selected ring 正常；
- [ ] Apply 仍执行显著差异确认并同步 WAXS geometry；
- [ ] Import/Export Calibration 保持原 JSON schema。
