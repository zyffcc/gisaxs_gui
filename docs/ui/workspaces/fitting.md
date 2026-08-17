# Fitting 布局迁移记录

## 控件映射

| 迁移前控件/区域 | 迁移后位置 | 行为 |
| --- | --- | --- |
| `GisaxsInputCard` | `Input / ParameterSection` | image/stack 加载、auto show、flip、display range 不变 |
| `CutLineCard` | `Configure / ParameterSection` | detector center、cut line、step/reset 和 Cut command 不变 |
| `ModelParameterCard` | `Advanced model configuration / AdvancedSection` | component add/remove 和所有参数对象不变；默认折叠 |
| `FittingControlsCard` | `Run / ParameterSection` | current/external curve、manual、AI、refinement、profiles 和 constraints 不变 |
| `DetectorPreviewCard` | `Preview / PlotPanel` | drag/drop、double-click、orientation 和 cut overlay 不变 |
| `PlotPreviewCard` | `Results / PlotPanel` | measured curve、components、fit curve 和 axes 不变 |
| `FittingPlotControlsCard` | `Advanced plot controls / AdvancedSection` | fitting region、sampling 和 plot display 不变 |
| `FittingTextBrowser`/`StatusCard` | `Log / AdvancedSection` | manual、AI 与 in-situ message sink 不变 |
| `FittingExportButton`、`fitExportPlotButton` | `Export / ParameterSection` | 复用原按钮实例和 signal，不复制 export command |

本轮没有修改 `FittingViewModel`、use case、controller workflow 或 fitting domain。现有 controller
仍作为 legacy Qt signal bridge；布局层没有新增 controller/ViewModel orchestration。AI Fast、
Balanced、Exhaustive、manual fitting 和 in-situ 继续调用迁移前相同的 application 行为。

## 手动验收清单

- [ ] CBF/NXS/TIFF 和 stack 加载、上一张/下一张、auto show 正常；
- [ ] flip、log、auto scale、colormap 和显示范围保持原值；
- [ ] center auto finding、detector parameters、cut geometry 和 Cut 正常；
- [ ] Advanced model configuration 折叠/展开不改变 component 数量或参数；
- [ ] current curve 与 external 1D curve 选择、log X/Y、normalize 正常；
- [ ] manual fit、Auto-K、Auto Refine、Clear 正常；
- [ ] AI model refresh/open、constraint、Fast/Full、Stop 和 advanced constraints 正常；
- [ ] detector Preview 的 drag/drop、double-click 和 overlay 正常；
- [ ] Results 中实验曲线、各 component、resolution 和总拟合曲线一致；
- [ ] fitting region、data points、plot options 折叠/展开不重置；
- [ ] Run Log 继续显示 manual、AI 和 in-situ 进度；
- [ ] Export Data 与 Export Plot 输出格式和文件内容不变；
- [ ] in-situ 三文件以上运行、取消、单文件失败继续和恢复正常。
