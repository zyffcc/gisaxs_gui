# Calibration 布局迁移记录

- **Status**: Current
- **Scope**: Geometry Calibration 的 PyQt presentation 所有权、控件映射与手动验收
- **Related code**:
  [`src/gimap/features/calibration/presentation/`](../../../src/gimap/features/calibration/presentation/)、
  [`geometry_calibration_dialog_view.py`](../../../src/gimap/features/calibration/presentation/views/geometry_calibration_dialog_view.py)、
  [`ui/geometry_calibration_dialog.py`](../../../ui/geometry_calibration_dialog.py)
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

旧文件 `ui/geometry_calibration_dialog.py` 只保留 9 行 import compatibility，直接 re-export
feature 的三个 class。生产 caller `ui/menu_manager.py` 已直接按需导入 feature-owned dialog；
旧路径仅服务尚未迁移的外部脚本和回归测试，不再被生产代码调用。

## 控件映射

| 迁移前控件/区域 | Python View object / 迁移后位置 | 行为 |
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
- [ ] energy、standard、estimated distance、range 和 detector defaults 与迁移前一致；
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
