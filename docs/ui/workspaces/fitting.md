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
- **Last verified**: 2026-09-22

## 当前状态

Fitting 的唯一 workspace layout 实现已由 feature presentation 拥有。当前界面不再把所有参数
同时铺开，而是按实验人员的真实流程组织：

```text
Fitting
├── Single analysis
│   └── Import data → Experiment setup → Yoneda & cut → Fit
└── In-situ series
    └── Source → Analysis settings → Results
```

顶部 `Single analysis / In-situ series` 是两个稳定工作上下文。切换上下文只改变当前可见页面，
不会重置 Single 的左侧步骤、Detector/Curve 标签、参数或图像，也不会重置 In-situ 的 Recipe 和
结果列表。Single 的 Load Mode 只有 Single/Stack；实时和批量序列只从顶部 In-situ context 进入。

In-situ 使用显式 Recipe 交接：先在 Single 分析一个代表文件，再点击 `Use current setup`
创建只读 `Recipe v1`。In-situ 中在 Analysis settings 调整预测模式、模型条件、失败策略，
展开 Detector / cut settings 调整 preprocessing、geometry、cut 与 tracking；保存时创建新版本，并明确应用到 future、selected +
future 或 all frames，且不会反向覆盖 Single。Live/Batch 控件直接位于页面，旧 runner dialog 已
删除。1D workflow 使用捕获的方法和模型条件，Single 后续修改不影响正在运行的序列。结果行记录 Recipe
版本和 load/preprocess/geometry/cut/fit 的独立状态。Source 以 acquisition root 为入口，可递归
监控子文件夹，并显式选择 `CBF images` 或 `NXS module series`。NXS 默认等待 11 个 module files
形成完整组，随后按内部 frame 逐帧排队；已存在 NXS 中追加 frame 与新建 NXS 组使用同一发现流程。
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
- Fit 主任务使用 `Components / Global / Data & refine / 1D Predict` 四个同级标签页。`Plot Current Model`
  只在手工模型标签页显示；默认进入 1D Predict，以 Fit curve 为主操作；
- `Data & refine` 提供 `Global Search` 与 `Local Refine` 两个独立入口。两者都打开非阻塞参数弹窗，
  顶部明确显示输入来自 current cut 或 imported 1D data 及实际点数；表格允许逐项选择优化参数并
  编辑 Min/Max。Global 使用数据/q-window 驱动的宽范围、differential-evolution evaluation budget
  和 local starts，并在每个候选上先消去线性幅度；默认约 16384 evaluations、3 starts。Local 使用
  当前值附近的 polishing 范围；运行期间显示所处阶段、总进度和当前/最佳 logRMSE，并可停止，完成后
  把选中参数写回 Components/Global 控件。Global 的 Target logRMSE 默认是 `0`（不提前停止），避免
  沿用 AI refine 的宽松阈值而在搜索尚未收敛时结束；
- `1D Predict` 的 Fit curve 按保存的 Fit method 执行，默认恢复为 **General V5 (experimental)**，
  提出多组分候选；这不表示旧模型的拟合质量或泛化问题已解决。**Single RC specialist
  (experimental)** 仅在已知单随机圆柱、Complete composition 显式选择一个 Random cylinder
  （`components=[2]`）且原生 CBF 计数范围适用时使用局部网络；默认只校准线性幅度。
  **Calibrate intensity amplitudes** 是该分支的独立开关。General predict only 使用旧通用候选网络；
- Parameters / batch… 打开候选曲线、参数和批处理界面；已保存的方法和 Recipe 不静默改写。
  使用局部分支前需检查 Fit method 与完整组分；已有 in-situ Recipe 用 1D parameters… 另存
  未来帧设置。内部 `method="stable"` 为兼容旧记录保留，不表示模型已获通用稳定性认证。
  该分支遇到 Auto／`[]` 组分会数值回退并只比较单组分族，不保证混合物覆盖；缺少原生计数的
  文本、固定 resolution、其他完整组分、threshold／镜像替换／stack 或曲线不符也会回退。
  Stage 显示 Neural model／Model + amplitude／Numerical fallback，悬停可见原因与限制。
  不显示伪概率，也不以 0.05 判定通过；选择行直接显示保存的 native forward，不经旧手工模型
  重算。当前快速分支不提供通用组分识别或多峰概率保证。模型与开发记录随
  `modules/Fitting_1D_Model/Workflow_v5` 迁移，完整用法见该目录 README_zh.md；
- Components/Global 参数输入保留至少 12 位小数；进入页面、打开搜索弹窗或焦点切换不会把已加载的
  小参数（例如 `10⁻⁸` 量级的 `D/sigma_D`）静默舍入并写回配置；
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

## CBF 自动 Yoneda 与 Center X 对称优化（2026-09-21）

Single 加载 CBF 后默认运行现有 Find Yoneda & Set Cut。`Find Yoneda automatically on CBF load`
可关闭并保存偏好；手动改 Center Vertical 或拖选 cut 区域仍有效。in-situ 帧加载继续遵守 Recipe，
不被 Single 的自动定位开关覆盖。

在步骤 3 Yoneda & Cut 的 Extract / Update Cut 后新增 **Optimize Center X**。
点击后在当前水平带内寻找左右对称轴，保存 detector.beam_center_x（q=0），更新 cut Center X，
并重新提取。保持 Yoneda 高度、带宽和 Beam center Y。q 模式保持原 detector 行区间。
结果横幅显示前后损失；损失是稳健 asinh 强度差，不是拟合误差或概率。

算法使用相同 AnalysisImage 和 pixel-cut 行边界，按行中位数降低孤立热像素影响；负 CBF 像素与
NaN 不参与，缺口不跨越插值。搜索使用固定半径、至少 60% 配对覆盖和最多 ±80 px 范围，
粗搜索后进行亚像素一维优化。无结构、范围太窄、最优点触及边界会提示调整选区。
镜像填补开启时先提示关闭，避免把补出来的对称当作测量证据。

验证：`python tools/check_cbf_center_workflow.py`（真实 CBF 到单条拟合）、
`--geometry-only`（手动 Yoneda / q 模式 / 延迟回调）、`--insitu`（真实 CBF 序列一帧）。
脚本读取上次保存的 beam / detector 到内存副本，使用独立 in-situ 测试缓存，不覆盖用户保存的几何或序列记录。
结果与已知质量问题见 `validation/center_symmetry/REPORT_zh.md`。

## 控件映射

| 功能/控件区域 | 当前位置 | 行为 |
| --- | --- | --- |
| `GisaxsInputCard` | `Input` | import 与 load mode 直接可见；Previous/Next/position/Show 常驻；手动导入立即预览 |
| `CutLineCard` | `Setup / Yoneda & Cut` | detector 参数内嵌；q 轴可选 qy/qr、纵轴为 qz；Yoneda、center、cut geometry 与两个显式命令位于同一任务页；自动水平 cut 厚度默认 5 px |
| `ModelParameterCard` | `Fit / Components` | component add/remove 和所有参数对象直接可见，不再藏在 Advanced |
| `FittingControlsCard` | `Fit / Components / Global / Data & refine / 1D Predict` | 默认 1D Predict 一键拟合；手工页保留 Plot Current Model 与传统参数/优化入口 |
| `DetectorPreviewCard` | `Detector` tab | 增加显式 center/region toolbar 和右侧 Display inspector；保留 drag/drop、double-click、orientation、overlay 与 empty state |
| 旧 `CutCurvePreviewCard` | 合并到 `Curve` tab | 不再保留第二套 dialog/canvas；显式 cut 显示 `Data only` |
| `PlotPreviewCard` | `Curve` tab | `Data only / Compare / Model only`、单一 q display、Log X/Log Y/Normalize 与结果状态常驻 |
| `FittingPlotControlsCard` | `Advanced plot controls / AdvancedSection` | fitting region、sampling 和 plot display 不变 |
| `FittingTextBrowser`/`StatusCard` | `Log / AdvancedSection` | manual、AI 与 in-situ message sink 不变 |
| `FittingExportButton`、`fitExportPlotButton` | `Curve / Export` | Data export 明确选择 raw/prepared/fitting-range；Plot export 保留原 command |
| `Single analysis / In-situ series` | Fitting 顶部 context switch | 切换稳定上下文，不清空任一页面状态 |
| `InSituSeriesPage` | `In-situ series` | 可点击逐帧 workflow、Recipe、内嵌 Live/Batch、Preview/Frames/Log 和统一 JobStatus |

In-situ Preview 右侧也有独立的 `Image display` inspector，包含 Auto scale、Log intensity、
Vmin/Vmax、Color map、Center 和 Cut ROI。它只控制当前 In-situ 图像的投影，不改变 Single 的
display widgets，不写入 Recipe，也不会启动 preprocessing、cut 或 fitting。

RC specialist 使用 NumPy／SciPy 运行时和明确的 nm 单位；默认 General V5 及训练使用 TensorFlow。
宽范围数值回退仍使用历史 V5 求积，结果会标注其范围限制；传统手工模型仍在手工页使用原算法。
新入口通过隔离进程执行。Recipe 保存实际科学几何精度，而非 spinbox 的舍入显示值，真正的
用户编辑仍被捕获；同一帧的 Single 与 in-situ 应保留相同 q、观测、像素计数和 ROI。

2026-09-22 早期 RC 分支开发帧回放的双侧 lnRMSE 为 0.17120／0.15523；worker 双侧计算约 0.68 s，
完整 GUI 点击约 3.70 s，单帧 in-situ 约 3.07 s。不同计时范围不可混用，数值回退耗时另计。
这些历史指标不属于恢复后的 General V5 默认方法，也不证明未知组分或新样品泛化。
本次范围纠正未改网络权重；既有测试通过只支持其声明范围。此前默认 Stable 的表述已撤回，历史证据见
[`RELEASE_zh.md`](../../../modules/Fitting_1D_Model/Workflow_v5/development/evidence/stable_blue_20260922/RELEASE_zh.md)。
从项目根目录运行 `python tools/check_stable_predict_workflow.py --mode all` 可重放真实按钮、
单帧 in-situ、worker、文本 batch 与取消检查；这些开发回放不代表独立样品泛化验证。

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
- [ ] Components、Global、Data & refine、1D Predict 四个标签始终可到达；默认进入 1D Predict；
      `Plot Current Model` 在手工页可点击；Global 的 default step 可编辑并保存，Resolution Sigma 内建值为 0.0001；
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
- [ ] Plot Current Model、Auto-K、Clear 正常；Global Search 与 Local Refine 分别打开宽范围/精修范围
      弹窗，勾选 `Use current cut` 时使用当前 cut，未勾选时使用 imported 1D data；bounds 包含当前
      值，完成后写回选中参数；
- [ ] Manual / AI assisted 切换不重置当前 curve、model 或 constraint state；
- [ ] 默认 General V5 标为 experimental；局部 RC specialist 也标为 experimental，只有显式单 RC 才用神经快速路径；
- [ ] Fit curve / General predict only / Quick physical fit / Stop / Parameters 正常；幅度校准开关独立，回退原因可见，误差不伪装为概率；
- [ ] 同一 CBF 在 Single／in-situ 下原生 q、强度、计数与 ROI 一致；Recipe 不舍入实际科学几何；
- [ ] CBF 坏点及 guard 在 Detector Preprocessing 中统一屏蔽；cut/拟合/in-situ 使用同一 revision 的原生有效点，500 点只用于正演显示；参数中的 Fit method 可用于文本 batch 与未来 in-situ 帧；
- [ ] 保留候选全曲线 logRMSE，说明其中包含测量噪声；按用户最新目标检查峰位、整体形状和残差，不再以 0.05 强制判定通过/失败；
- [ ] Add curves 多选文本文件后一次 Fit N files 可完成，坏文件记录失败并继续；
- [ ] detector Preview 的 drag/drop、double-click 和 overlay 正常；
- [ ] Curve 的 Data only、Compare、Model only 分别显示正确图层，实验曲线、各 component、resolution
      和总拟合曲线一致；
- [ ] fitting region、data points、plot options 折叠/展开不重置；
- [ ] Run Log 继续显示 manual、AI 和 in-situ 进度；
- [ ] Export Data 的 raw/prepared/fitting-range 与页面状态一致，header 可追溯；Export Plot 正常；
- [ ] in-situ 三文件以上运行、取消、单文件失败继续和恢复正常。
- [ ] Single/In-situ 来回切换不重置左侧步骤、Detector/Curve 当前标签、Recipe 或结果表；
- [ ] Single Load Mode 只有 Single/Stack；In-situ 页面无需切换 Single mode 即可选择 folder 并运行；
- [ ] In-situ Source 可选择 acquisition root、CBF/NXS、pattern 和递归子目录；CBF 子目录中新文件可入队；
- [ ] NXS 等待配置的 module 数且各 module frame count 一致；同一组追加内部 frame 和新建另一组
      NXS 都只把新增 frame 入队，不重复已处理 frame；
- [ ] In-situ Preview 的 Image display 可独立调节 log、色图、范围和 overlay；调节后不改变 Recipe、
      AnalysisImage revision、当前 Preview/Frames/Log 标签或 Single 当前页面；
- [ ] 点击 Source/Analysis settings/Results 只切参数页，Start/Pause/Stop 位置不变；
- [ ] Frames 选中任一行后，各流程节点显示该帧真实成功、失败或跳过状态；
- [ ] 未加载代表文件时不能创建 Recipe；创建后显示版本和来源；In-situ policy 修改产生下一版本；
- [ ] Recipe 的 future/selected/all scope 明确，In-situ 修改不会改变 Single 控件；
- [ ] Guided/Compact 偏好可保存；workflow 仅在成功后完成，上游参数改变后下游显示 stale。

## Interactive detector inspection

The detector toolbar's **Interactive** button opens the shared pixel viewer with wheel zoom, drag pan,
ROI, crosshair, full-resolution unlogged intensity readout and histogram/LUT. **Apply ROI** uses the
existing detector-region handler. Log intensity and histogram range changes update the workspace
controls; custom gradient presets are inspection-only and do not change the export colormap.
The existing **Open** window and publication/export actions remain available. Nonuniform q-space
continues in the existing detector view; the interactive pixel window disables stale selection when
q-space is active. GISAXS stack playback is not yet connected to this window.
