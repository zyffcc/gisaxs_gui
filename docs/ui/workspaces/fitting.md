# Fitting 布局迁移记录

- **Status**: Current
- **Scope**: Cut & Fitting workspace 的 PyQt layout ownership、控件映射与手动验收
- **Related code**:
  [`src/gimap/features/fitting/presentation/`](../../../src/gimap/features/fitting/presentation/)、
  [`ui/components/main_window_components.py`](../../../ui/components/main_window_components.py)、
  [`ui/main_window.py`](../../../ui/main_window.py)
- **Related tests**:
  [`tests/test_fitting_presentation.py`](../../../tests/test_fitting_presentation.py)、
  [`tests/test_fitting_view_model.py`](../../../tests/test_fitting_view_model.py)、
  [`tests/test_ui_workspace_layouts.py`](../../../tests/test_ui_workspace_layouts.py)
- **Last verified**: 2026-08-18

## 当前状态

Fitting 的唯一 workspace layout 实现已由 feature presentation 拥有，并按 Input、Cut、Run、
Model、Preview 和 Workspace 职责拆分。Run 区域中的 AI controls 与 global parameter editors
也分别具有明确模块：

```text
feature-owned Python View factory
                 ↓
feature-owned GisaxsFittingWorkspace
                 ↓
FittingViewBinding → FittingViewModel → application use cases
```

`ui/components/main_window_components.py` 不再定义 Fitting workspace、cards 或 plot controls；
它只从 feature public presentation API 导入并在 application shell 中组装。旧 import path
继续返回相同的 feature-owned classes，不存在第二套页面实现。

Fitting controls 的构造现由 `presentation/control_view_factory.py` 单一拥有；
`Ui_MainWindow.setupUi` 只调用该工厂并继续提供 binding 所需的相同属性名；页面文本、模型
选项和默认显示值也由同模块的 `translate_fitting_controls` 单一设置。生产运行时直接构造
`FittingViewBinding`；它负责 Qt signals、dialogs 和绘图，而科学计算、NXS/CBF/TIFF I/O、
AI、in-situ persistence、模型参数和外部 runtime 均通过 ViewModel/use cases/ports。旧
`FittingController` 名称只作为兼容别名。

静态 UI 现在有明确的 Python View source：

- `views/fitting_page_view.py` 保存 legacy binding 仍需访问的原始控件、objectName、模型选项和
  数值默认值；`QRangeSlider` 使用 feature-owned custom widget；
- `views/fitting_workspace_view.py` 保存实际可见的左右 splitter，以及 Input、Configure、
  Advanced、Run、Preview、Results、Log、Export section hierarchy；
- `views/detector_parameters_dialog_view.py` 保存 detector 参数窗；
- `views/independent_image_window_view.py` 与 `views/independent_fit_window_view.py` 保存两个独立绘图
  窗口的固定外壳和筛选控件，Matplotlib canvas、toolbar 与 actions 仍注入明确 host。

`control_view_factory.py` 只负责 Python View 实例化和兼容属性转发。各 card
仍负责运行时重组原控件和动态 AI/模型参数，但不再创建第二套 workspace section shell。

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

本次所有权迁移没有修改科学算法、参数、单位或 fitting domain 行为。
所有 View 控件、objectName、button instances、默认参数、快捷操作和 signal targets 均保持不变。
旧 controller 文件不再承载实现；布局层没有新增 controller/ViewModel 双重
orchestration。AI Fast、Balanced、Exhaustive、manual fitting 和 in-situ 继续调用迁移前相同的
application 行为。

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
