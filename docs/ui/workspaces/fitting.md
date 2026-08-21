# Fitting 界面与交互说明

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
- **Last verified**: 2026-08-21

## 当前状态

Fitting 的唯一 workspace layout 实现已由 feature presentation 拥有。当前界面不再把所有参数
同时铺开，而是按实验人员的真实流程组织：

```text
Fitting
├── Single analysis
│   └── Import data → Experiment setup → Yoneda & cut → Fit
└── In-situ series
    └── Source → Preprocess → Geometry → Yoneda & cut → Fit → Results
```

顶部 `Single analysis / In-situ series` 是两个稳定工作上下文。切换上下文只改变当前可见页面，
不会重置 Single 的左侧步骤、Detector/Curve 标签、参数或图像，也不会重置 In-situ 的 Recipe 和
结果列表。Single 的 Load Mode 只有 Single/Stack；实时和批量序列只从顶部 In-situ context 进入。

In-situ 使用显式 Recipe 交接：先在 Single 分析一个代表文件，再点击 `Use current Single setup`
创建只读 `Recipe v1`。In-situ 中可以点击各流程节点调整 preprocessing、geometry、cut、tracking、
fit initial values、refinement 或失败策略；保存时创建新版本，并明确应用到 future、selected +
future 或 all frames，且不会反向覆盖 Single。Live/Batch 控件直接位于页面，旧 runner dialog 已
删除。若用户在捕获后改过 Single model，必须显式重新捕获，不能静默用错模型。结果行记录 Recipe
版本和 load/preprocess/geometry/cut/fit 的独立状态。
完整契约见
[`../../architecture/insitu-series-workflow.md`](../../architecture/insitu-series-workflow.md)。

```text
Import data → Experiment setup → Yoneda & cut → Fit
                                               ├─ Plot current model
                                               ├─ Refine
                                               └─ AI assisted
```

左侧是一次只展示一个任务的操作轨道，工作流导航固定在滚动区上方；右侧是持续可见的工作画布，
使用稳定的 `Detector` 与 `Curve` 两个标签页。Curve 在同一 canvas 上按 `Data only / Compare /
Model only` 切换实验数据和模型图层，cut 与 fitting 不再各自占用一个会跳动的页面。导航步骤始终
可以点击，并且只改变左侧任务；右侧页面由用户独立控制。完成状态仍只由成功/失败结果驱动，
不会把一次未成功的按钮点击误记为完成。
可切换 `Guided` / `Compact`，熟练用户隐藏说明后仍保留快捷导航。Detector 参数直接位于 Setup
任务内，不再要求打开单独 dialog，也不显示面向开发者的“每次实验设置一次”备注。Image display
和 preprocessing 常驻 detector preview 旁；只有 remote cache、step sizes、AI tuning、专家绘图
参数和日志等低频内容采用 progressive disclosure。

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

- `views/fitting_page_view.py` 保存 ViewBinding 使用的原始控件、objectName、模型选项和
  数值默认值；`QRangeSlider` 使用 feature-owned custom widget；
- `views/fitting_workspace_view.py` 保存实际可见的左右 splitter，以及 Input、Configure、
  Advanced、Run、Preview、Results、Log、Export section hierarchy；
- `views/detector_parameters_dialog_view.py` 仅保留旧 detector dialog 的兼容 View；当前 workspace
  使用 `presentation/detector_setup_panel.py` 内嵌参数编辑器；
- `views/independent_image_window_view.py` 与 `views/independent_fit_window_view.py` 保存两个独立绘图
  窗口的固定外壳和筛选控件，Matplotlib canvas、toolbar 与 actions 仍注入明确 host。

`control_view_factory.py` 只负责 Python View 实例化和兼容属性转发。各 card
仍负责运行时重组原控件和动态 AI/模型参数，但不再创建第二套 workspace section shell。

现代化视觉层由 `presentation/fitting_theme.qss` 单一拥有；流程展示与纯状态转换分别由
`presentation/workflow_header.py` 和 `presentation/workflow_state.py` 拥有。它们只处理展示和 UI state，不包含 scientific
calculation、文件格式判断、TensorFlow inference 或 fitting orchestration。

当前主流程的交互约定如下：

- 文件选择、路径回车、Previous/Next 均立即显示图像；`auto_show` JSON compatibility 字段继续保留，
  但不会让手动导入回到“只加载不显示”的状态；
- `Previous / Next / 当前位置 / Show` 始终位于文件输入下一行，不受 Advanced 折叠影响；
- detector preview 顶部提供显式 `Pick center`、`Select cut region`、`Reset view`、`Open viewer`；
  主预览可直接单击中心或拖选区域，`Esc` 取消，不再要求用户发现右键隐藏手势；
- Setup 的 `Show detector axes in q` 提供 `qy / signed qr` 水平坐标选择，纵轴固定为 `qz`。Detector
  使用真实二维 q 网格绘制而不是 min/max extent 拉伸；主预览、独立 viewer、center、框选和 Yoneda
  cut 共用最近 detector-cell 吸附。切换 q 坐标时保留原 detector 区域并刷新已有 cut；
- Auto scale、Vmin/Vmax、Log intensity、Color map、center/cut overlay 位于预览右侧 Image display
  inspector；mask、gap fill、threshold 和 Flip UD 位于其下方始终可见的 Preprocessing 区；Input
  不再保留空的重复 display/preprocessing 折叠栏；
- detector 数据遵守 [`../../architecture/scientific-data-flow.md`](../../architecture/scientific-data-flow.md)：
  Flip UD、threshold/mask 和 mirror-fill 共同生成唯一 AnalysisImage，Preview、Yoneda/center finding、
  ROI/cut 与 fitting 使用同一 revision；colormap、vmin/vmax、log intensity 和 overlay 只属于
  DisplayState；
- 独立 2D/1D 窗口是嵌入 Detector/Curve 的放大投影，不是第二套页面：2D 共用 AnalysisImage 与
  DetectorDisplayState，1D 共用 CurvePlotSpec 与 CurveViewState。主界面或独立窗口修改显示选项会
  双向同步；只有 zoom/pan、窗口大小和临时工具模式保持为窗口局部状态；
- 修改 center、cut geometry、sampling 或 detector 参数会先更新 draft；如果已有 cut，则在防抖后
  用新参数重算该 cut，但保持当前 tab、scroll 和 focus；尚无 cut 时不会隐式启动 cut 或 fitting；
- 显式 cut 成功后进入 `Curve / Data only`；显式 `Plot Current Model` 成功后进入
  `Curve / Compare`。失败和自动刷新都保持当前页并显示 inline error；
- 用户导航把 Yoneda 与 cut 合并为一个连续任务页；内部仍分别保存 `center` 与 `cut` 的完成、失败和
  stale 状态。`Find Yoneda & Set Cut` 使用可调的 `Auto horizontal cut thickness`，默认 5 px，
  表示围绕 Yoneda 位置平均的 detector rows；显式 `Extract / Update Cut` 才生成 1D curve；
- Fit 主任务使用 `Components / Global / Data & refine / Auto fit` 四个同级标签页。`Plot Current Model`
  是标签页外的常驻主命令，修改 component 或 global 参数后无需切换工作面即可重新绘制；
- Global 的 `Default step` 列就是数值增量设置入口，修改后保存到 UI preferences。Resolution Sigma
  的内建默认步长是 `0.0001`，Reset 恢复内建值；
- 结果区顶部只暴露一个 `q display` 选择以及 `Log X / Log Y / Normalize`。`Signed ±q` 保留符号，
  `Positive +q`、`Negative −q` 和 `Negative as |q|` 提供单支选择，`Overlay ±q as |q|` 与
  `Average ±q` 提供折叠/平均；Signed 或 Negative 模式勾选 Log X 时自动使用 symlog，已转为
  正 `|q|` 的模式使用普通 log，不再暴露 Branch、Combine、X scale 三个互相耦合的内部维度；
  Overlay 中 +q 固定使用蓝色、镜像 −q 固定使用红色，并在嵌入图与独立图中保持一致；
- `Detector / Curve` 标签栏始终位于同一位置；q display、curve layers 和 inline feedback 属于
  Curve 页面内容，不得出现在标签栏之前或在切换页面时推动导航栏；
- 两个 preview tab 的 layout hint 只由当前可见页决定；Curve 中展开 Advanced plot controls
  只能改变 Curve 自身的高度/滚动范围，不得改变 Detector 的几何；
- 页面 spin box 和 combo box 采用公共 safe-wheel 行为：普通滚轮滚动页面，只有控件获得焦点且
  按住 Alt/Option 时才修改输入；
- 参数 Enter/结束编辑立即提交；方向键、按钮箭头和有意滚轮连续修改采用 `220 ms` trailing
  debounce。完整规则见
  [`../../architecture/ui-interaction-contract.md`](../../architecture/ui-interaction-contract.md)；
- Export Data 先显示 source 与 `Data used for fitting / Prepared full / Raw signed` 三种明确表示，
  文件 header 记录 branch、combination、X scale、ROI 和参数快照。

## 控件映射

| 功能/控件区域 | 当前位置 | 行为 |
| --- | --- | --- |
| `GisaxsInputCard` | `Input` | import 与 load mode 直接可见；Previous/Next/position/Show 常驻；手动导入立即预览 |
| `CutLineCard` | `Setup / Yoneda & Cut` | detector 参数内嵌；q 轴可选 qy/qr、纵轴为 qz；Yoneda、center、cut geometry 与两个显式命令位于同一任务页；自动水平 cut 厚度默认 5 px |
| `ModelParameterCard` | `Fit / Components` | component add/remove 和所有参数对象直接可见，不再藏在 Advanced |
| `FittingControlsCard` | `Fit / Components / Global / Data & refine / Auto fit` | `Plot Current Model` 常驻；k/BG/resolution default steps、current/external curve、AI、refinement、profiles 和 constraints 保持可达 |
| `DetectorPreviewCard` | `Detector` tab | 增加显式 center/region toolbar 和右侧 Display inspector；保留 drag/drop、double-click、orientation、overlay 与 empty state |
| 旧 `CutCurvePreviewCard` | 合并到 `Curve` tab | 不再保留第二套 dialog/canvas；显式 cut 显示 `Data only` |
| `PlotPreviewCard` | `Curve` tab | `Data only / Compare / Model only`、单一 q display、Log X/Log Y/Normalize 与结果状态常驻 |
| `FittingPlotControlsCard` | `Advanced plot controls / AdvancedSection` | fitting region、sampling 和 plot display 不变 |
| `FittingTextBrowser`/`StatusCard` | `Log / AdvancedSection` | manual、AI 与 in-situ message sink 不变 |
| `FittingExportButton`、`fitExportPlotButton` | `Curve / Export` | Data export 明确选择 raw/prepared/fitting-range；Plot export 保留原 command |
| `Single analysis / In-situ series` | Fitting 顶部 context switch | 切换稳定上下文，不清空任一页面状态 |
| `InSituSeriesPage` | `In-situ series` | 可点击逐帧 workflow、Recipe、内嵌 Live/Batch、Preview/Frames/Log 和统一 JobStatus |

页面布局和 presentation ownership 不修改科学算法、参数、单位或 fitting domain 行为。
所有 View 控件、objectName、button instances、默认参数、快捷操作和 signal targets 均保持不变。
旧 controller 文件不再承载实现；布局层没有新增 controller/ViewModel 双重
orchestration。AI Fast、Balanced、Exhaustive、manual fitting 和 in-situ 调用相同的
application 行为。

## 手动验收清单

- [ ] CBF/NXS/TIFF 和 stack 加载、路径回车、上一张/下一张均立即显示，position 正确；
- [ ] Show 始终可见，恢复 `auto_show=false` 的旧 session 后手动导入仍立即显示；
- [ ] Auto Show 每次启动均勾选；Show 与文件导航常驻，旧 session 的 false 不覆盖启动默认；
- [ ] preview 旁 Image display 和 Preprocessing 无需展开即可操作；flip、threshold、gap fill、log、
      auto scale、colormap 和显示范围都能刷新图像；
- [ ] 开启 Flip UD 或 mirror-fill 后，Detector preview、Find Yoneda、ROI/cut 和 in-situ auto cut
      使用同一个 AnalysisImage revision；改变 colormap/log/vmin/vmax 不改变 revision；
- [ ] Setup 内 detector parameters 可完整编辑和应用，不弹出独立 dialog，不显示开发备注；
- [ ] 开启 q axes 后二维图按真实 qy/qz 或 signed-qr/qz 网格绘制；切换 qy/qr 时 center、选区和已有
      Yoneda cut 保持同一 detector cells，标签、数值与曲线横坐标同步；
- [ ] 主预览 Pick center 单击、Select cut region 拖选、Esc 取消和独立窗口原交互都正常；
- [ ] Yoneda & Cut 在同一任务页；自动水平 cut 厚度默认 5 px、可修改并跨启动保留；只改 center/region
      不切 tab；已有 cut 时防抖更新，无 cut 时不隐式执行 cut/fitting；
- [ ] 成功 Cut 自动进入 Curve/Data only，失败不切页；成功 Plot Current Model 进入 Curve/Compare；
- [ ] 点击 Import/Setup/Yoneda & cut/Fit 只切换左侧内容，在 Detector 或 Curve 上都不重置右侧 tab；
- [ ] Components、Global、Data & refine、Auto fit 四个标签始终可到达；`Plot Current Model` 在四页中
      都可直接点击；Global 的 default step 可编辑并保存，Resolution Sigma 内建值为 0.0001；
- [ ] Components 中 particle 新增/移除、shape 与对应参数页一致；
- [ ] 滚动左侧页面经过数值框不会误改值；focus + Alt/Option + wheel 可以有意调整；
- [ ] 数值参数 Enter 立即更新；连续方向键/Alt-wheel 仅在停止约 220 ms 后提交最终值，交互无卡顿；
- [ ] current curve 与 external 1D curve 选择正常；q display 六种用户模式含义明确；
- [ ] Signed 保留负 q；Negative as |q|/Overlay 显式折叠；Average ±q 只输出重叠域平均；ROI、
      preview、拟合输入和 export 对同一模式的解释一致；
- [ ] 未勾选 Log X 时线性；Signed/Negative 勾选后为 symlog；正 q/折叠 q 勾选后为普通 log；
      Log Y 与 Normalize 正常；
- [ ] 连续切换 Detector、Curve 时标签栏的纵向位置不变；q display 和 curve layers 只在曲线页面
      内容中出现，错误/状态 banner 也不推动标签栏；
- [ ] Curve 展开 Advanced plot controls 后切回 Detector，Detector 宽高与滚动范围不受隐藏页影响；
- [ ] Plot Current Model、Auto-K、Auto Refine、Clear 正常；成功绘图进入结果页，失败显示 inline error；
- [ ] Manual / AI assisted 切换不重置当前 curve、model 或 constraint state；
- [ ] AI model refresh/open、constraint、Fast/Full、Stop 和 advanced constraints 正常；
- [ ] detector Preview 的 drag/drop、double-click 和 overlay 正常；
- [ ] Curve 的 Data only、Compare、Model only 分别显示正确图层，实验曲线、各 component、resolution
      和总拟合曲线一致；
- [ ] fitting region、data points、plot options 折叠/展开不重置；
- [ ] Run Log 继续显示 manual、AI 和 in-situ 进度；
- [ ] Export Data 的 raw/prepared/fitting-range 与页面状态一致，header 可追溯；Export Plot 正常；
- [ ] in-situ 三文件以上运行、取消、单文件失败继续和恢复正常。
- [ ] Single/In-situ 来回切换不重置左侧步骤、Detector/Curve 当前标签、Recipe 或结果表；
- [ ] Single Load Mode 只有 Single/Stack；In-situ 页面无需切换 Single mode 即可选择 folder 并运行；
- [ ] 点击 Source/Preprocess/Geometry/Yoneda & cut/Fit/Results 只切参数页，Start/Pause/Stop 位置不变；
- [ ] Frames 选中任一行后，各流程节点显示该帧真实成功、失败或跳过状态；
- [ ] 未加载代表文件时不能创建 Recipe；创建后显示版本和来源；In-situ policy 修改产生下一版本；
- [ ] Recipe 的 future/selected/all scope 明确，In-situ 修改不会改变 Single 控件；
- [ ] Guided/Compact 偏好可保存；workflow 仅在成功后完成，上游参数改变后下游显示 stale。
