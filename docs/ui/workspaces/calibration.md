# Calibration 界面与交互说明

- **Status**: Current
- **Scope**: Geometry Calibration 的 PyQt presentation 所有权、控件映射与手动验收
- **Related code**:
  [`src/gimap/features/calibration/presentation/`](../../../src/gimap/features/calibration/presentation/)、
  [`geometry_calibration_dialog_view.py`](../../../src/gimap/features/calibration/presentation/views/geometry_calibration_dialog_view.py)
- **Related tests**:
  [`tests/test_calibration_presentation.py`](../../../tests/test_calibration_presentation.py)、
  [`tests/test_calibration_feature.py`](../../../tests/test_calibration_feature.py)、
  [`tests/test_calibration.py`](../../../tests/test_calibration.py)、
  [`tests/test_release2_calibration_window.py`](../../../tests/test_release2_calibration_window.py)（窗口里的像素坐标）、
  [`tests/test_release2_calibration_edges.py`](../../../tests/test_release2_calibration_edges.py)（图像边缘、中心在探测器外）、
  [`tests/test_release2_calibration_files.py`](../../../tests/test_release2_calibration_files.py)（导出文件的格式）、
  [`tests/test_release2_calibration_estimate.py`](../../../tests/test_release2_calibration_estimate.py)（Estimated distance 填什么）
- **Last verified**: 2026-10-13

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

## 光束中心用哪套坐标（2026-10-13）

窗口里显示和输入的光束中心（结果栏、候选解表、手动精修的 X / Y、预览图的坐标轴和中心线、Apply 之后的提示）用规范像素坐标，
和 Analyze 相同：像素角点在整数上，第一个像素的中心是 (0.5, 0.5)，x 从左边、y 从上边数，见 `shared/geometry/conventions.py`。
同一个仪器配置在两处显示同样的数，可以直接对照或抄写。

- 改的只是显示和输入。Apply 保存的几何（仪器配置）和以前在数值上相同，已保存的配置不用重做。
- 标定引擎内部和导出的 `.json` 仍按“像素中心从 0 数”记中心（`CalibrationCandidate.center_x_px / center_y_px`），
  比窗口里显示的数小 0.5 px；导出文件在 `metadata.center_convention` 里写明这一点，旧版本导出的文件照样能读。
- 屏幕上的数和引擎的数只在 `presentation/pixel_coordinates.py`（`shown_center`、`fitted_center`）换算，
  算式在 `shared/geometry/conventions.py`。新加一处显示或输入中心的控件时走这两个函数。
- 以前的版本在窗口里显示的是引擎的数，比 Analyze 小 0.5 px；手抄过中心的记录要按这个差别核对。

## 已知限制：Auto Detect 可能把错的标样排在第一（2026-10-13）

载入图像时看的是图像的整个路径（文件名和它上面各级文件夹的名字，不分大小写；标样的写法见
`domain/standard_selection.py` 的 `STANDARD_SOURCE_ALIASES`，如 `agbh`、`lab6`、`ceo2`）：

- 路径里只出现一种标样：自动选中它；
- 出现两种以上（例如 LaB₆ + CeO₂ 混合标样的名字）：Standard 设为 Auto Detect，状态行写 “multiple standard names found”；
- 一种都没有：Standard 保持原样，新开的窗口是 Auto Detect。

Auto Detect 时引擎用所有已知标样去试，再按环的匹配打分。打分可能让另一种标样的解排在第一。

- 见过的例子：一张山嵛酸银（AgBh）图像，路径里没有标样，Estimated distance 为空（新安装现在就是这样）。排第一的是 LaB₆
  在短得多的距离上的解，AgBh 的解排在候选表的最后；手动选 AgBh、并且 Estimated distance 为空时结果正确。
- **估计距离错了会带偏结果，选对了标样也一样。** 搜索偏向 Estimated distance 附近的解（`domain/candidate_ranker.py`）。
  同一张图上选 AgBh：框里填 2000 mm 时排第一的是约 1830 mm 的解，而实际距离约 1630 mm（候选表里没有正确的解）；
  填 1200 mm 时排第一的是约 1175 mm 的解（正确的解排第二）；留空（0，显示 Optional）或填 1400–1900 mm 时排第一的是正确的解。
  Auto Detect 时，估计值接近实际值（这张图上 1500–1800 mm）能把正确的解排到第一，填 2000 mm 时排第一的仍是 LaB₆。
- 用户该怎么做：知道标样时自己选；知道装置的大概距离就填进 Estimated distance，不知道就留着 Optional；
  载入图像后框里出现不是自己输入的数时（见下一节），核对它是不是这套装置的距离，不是就清成 0；Apply 之前核对距离，
  并看画出的环是否落在图像的环上。手册第 9 节和 `CHANGELOG.md` 的已知限制写了同样的话。
- 搜索和排序是打分算法的问题，按用户的决定这一版只写进文档，之后再改。
  `tests/test_release2_docs_decisions.py` 里有测试在 `tests/data/external/` 的公开 AgBh 图上复现这两种情况
  （把图像拷到路径里不含标样的地方再标定）：改好之后那些测试会失败，那时把它们改成回归测试，
  并删掉手册、`CHANGELOG.md` 和本节里的这条限制。

## Estimated distance 里填什么（2026-10-13）

框里是 0 时显示 “Optional”（中文界面 “可选”），搜索就不带估计值。内置的 2000 mm 不再出现在框里：以前新安装载入图像时
会把它填进去，上面“选了 AgBh 却排出约 1830 mm”的情况在新安装上不用用户做任何事就会发生。现在每次载入图像，
窗口按下面的顺序决定框里的数（`bindings/image_loading.py` 的 `_prefill_estimated_distance`，
回归测试 `tests/test_release2_calibration_estimate.py`）：

1. 图像文件写明的距离。实际上只有 NeXus 文件的 `/entry/instrument/detector/distance`；CBF 头的 `Detector_distance` 和
   EDF 头的 `SampleDistance` 常常是过期值，不用（`shared/detector_io/metadata.py` 把它们放在 `header_distance_m`）。
   框显示不了的值（大于 100 m、不是有限的数、四舍五入到两位小数后是 0）当作文件没有写；以前会显示成框的上限 100000.00 mm。
2. 否则是上一次 Apply 的标定的距离，本次或以前的会话都算。设置里的 `detector.distance` 在每个安装上都有数
   （内置的模拟几何 2000 mm，见 `integrations/state/default_settings.py`），数本身说明不了有没有做过标定；
   Apply 同时写下的 `system.geometry_calibration` 才说明（`SettingsGeometryAdapter.current_geometry`，
   XRR 窗口用同样的判断）。存下的数读不出来时当作没有。
3. 两者都没有：用户自己输入的值留着（载入第一张图之前输入的、用户清成 0 的也算）；窗口为上一张图填的值退回 0。

还要知道：

- 第 1、2 种来源会盖掉用户输入的值和用户清空的框，每载入一张图都再填一次。上一次 Apply 的标定可能属于别的装置或别的探测器
  （Apply 的记录里有 `image_shape` 和 `detector`，窗口不比较），窗口也不说明填进去的数来自文件还是来自标定。
- 0.1.0 之前的源码版本留下的设置（从 `config/user_parameters.json` 导入）里可能有 Apply 写下的几何，却没有
  `system.geometry_calibration`：按没有做过标定处理，框显示 Optional，在这一版里按一次 Apply 之后恢复。
- NeXus 的距离没有 `units` 时，共用的读取器按米读。窗口只是不显示超出范围的值；一个错的、但看起来合理的数仍会被填进去。

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
`calibrationApplied` signal、错误标题/文案和 JSON v1 schema 保持原行为；Open 的过滤器是 NXS、CBF、TIFF、EDF
（`bindings/image_loading.py` 的 `CALIBRATION_IMAGE_FILTER`）。

## 手动验收清单

- [ ] Open（NXS / CBF / TIFF / EDF）、粘贴路径和 ambiguous NXS dataset 选择正常；
- [ ] energy、standard、estimated distance、range 和 detector defaults 与配置契约一致；
- [ ] 无 pixel metadata 或选择 Custom detector 时 Advanced 自动展开；
- [ ] Advanced 折叠/展开不改变 pixel、distance bounds 或 overlay toggles；
- [ ] Preview 保持原 image orientation、log display、mask 和 ring colors；
- [ ] Auto Calibration 可启动，JobStatus 显示 stage/progress；
- [ ] Cancel 在当前 numerical step 后安全停止；
- [ ] candidate table 选择可更新 overlays 和 result labels；
- [ ] Reset view、Clean image、Focus image、Manual refine 正常；
- [ ] 手动拖动 center、编辑 distance、Fit selected ring 正常；
- [ ] 结果栏和手动精修里的光束中心，与 Apply 之后 Analyze 对同一个仪器配置显示的数相同（不差 0.5 px）；
- [ ] 路径里不含标样的图像：Standard 留在 Auto Detect 时核对排第一的标样和距离（见上面的已知限制）；
- [ ] 新安装（没有 Apply 过）载入图像后 Estimated distance 显示 Optional；Apply 之后载入下一张图，框里是刚应用的距离；
      框里的数（上次标定的、NeXus 文件的、自己输入的）偏离实际距离时，选对标样也要核对距离；
- [ ] Apply 仍执行显著差异确认，把几何存为仪器配置，并通知 Analyze 改用它；
- [ ] Import/Export Calibration 保持原 JSON schema。
