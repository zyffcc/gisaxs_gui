# Calibration 布局迁移记录

## 控件映射

| 迁移前控件/区域 | 迁移后位置 | 行为 |
| --- | --- | --- |
| Calibration image path/Open、energy、standard、distance、detector | `Input / ParameterSection` | loader、detector auto-detection 和 defaults 不变 |
| pixel X/Y、custom range、background、log/mask/rings | `Advanced configuration / AdvancedSection` | 默认折叠；metadata 缺失或 Custom detector 时自动展开 |
| Auto Calibration、Cancel、progress、stage label | `Run / ParameterSection + JobStatus` | 原 worker、cancel boundary 和 progress callbacks 不变 |
| Matplotlib toolbar/canvas、overlay actions/legend | `Preview / PlotPanel` | ring/center rendering、zoom 和 drag behavior 不变 |
| selected solution labels、candidate table | `Results / ParameterSection` | candidate order、confidence 和 residual 不变 |
| manual center/distance/ring fitting | `Advanced manual refinement / AdvancedSection` | 与原 Manual refine toggle 同步，折叠不重置值 |
| Import/Export Calibration、Apply、Close | `Export / ParameterSection` | JSON format、AppContext 写入和 main-window sync 不变 |

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
