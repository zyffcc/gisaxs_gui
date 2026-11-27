# Prediction 界面与交互说明

- **Status**: Current
- **Scope**: GIMaP Prediction workspace 的 PyQt layout ownership、控件映射与手动验收
- **Related code**:
  [`src/gimap/features/prediction/presentation/`](../../../src/gimap/features/prediction/presentation/)、
  [`src/gimap/app/main_window.py`](../../../src/gimap/app/main_window.py)、
  [`src/gimap/app/window_view.py`](../../../src/gimap/app/window_view.py)
- **Related tests**:
  [`tests/test_prediction_presentation.py`](../../../tests/test_prediction_presentation.py)、
  [`tests/test_prediction_multifile_presentation.py`](../../../tests/test_prediction_multifile_presentation.py)、
  [`tests/test_prediction_watch.py`](../../../tests/test_prediction_watch.py)、
  [`tests/test_prediction_view_model.py`](../../../tests/test_prediction_view_model.py)、
  [`tests/test_ui_workspace_layouts.py`](../../../tests/test_ui_workspace_layouts.py)
- **Last verified**: 2026-11-26

## 当前状态

Prediction 已按真实操作顺序重做为现代双栏工作台。唯一 workspace layout 实现仍由 feature
拥有；`workspace.py` 组装既有控件，`workbench_layout.py` 只负责左侧流程栏和右侧画布，
`workflow_components.py` 提供模式切换、可导航三步状态、折叠区域和空状态，
`workflow_state.py` 保存不依赖点击历史的 typed workflow snapshot，样式集中在
`prediction_theme.qss`：

```text
feature-owned Python View factory
                 ↓
feature-owned GisaxsPredictWorkspace / PredictionWorkbenchLayout
                 ↓
PredictionViewBinding → PredictionViewModel → application use cases
```

Application shell（`src/gimap/app/main_window.py`）不定义 Prediction workspace 或专用 cards，
只从 feature public presentation API 导入并组装。旧的 `ui/` 与 `controllers/` 兼容别名已删除。

Prediction controls 的构造现由 `presentation/control_view_factory.py` 单一拥有；
`Ui_MainWindow.setupUi` 只调用该工厂并继续提供 binding 所需的相同属性名；页面文本、默认
显示值和 tab 标题也由同模块的 `translate_prediction_controls` 单一设置。生产运行时直接构造
`PredictionViewBinding`；旧 `GisaxsPredictController` 名称只是兼容别名。

静态 UI 现在有明确的 Python View source：

- `views/prediction_page_view.py` 保存 ViewBinding 使用的原始控件、objectName、默认值和
  GISAXS/Predict-2D tabs；
- `views/prediction_workspace_view.py` 保存用户实际看到的 Input、Configure、Advanced、Preview、
  Run、Results、Export section hierarchy；
- `views/multifile_results_widget_view.py` 保存多文件结果过滤、排序、表格和 action bar；
- `views/export_dialog_view.py`、`views/distribution_heatmap_dialog_view.py`、
  `views/parameter_trend_dialog_view.py` 保存相应辅助窗体的静态布局，Matplotlib canvas 仍在运行时
  注入各自的 `plot_host`。

`control_view_factory.py` 只实例化 `PredictionPageView` 并把既有属性暴露给 application shell。
`workspace.py` 通过 `PredictionWorkspaceView` 的语义化 host 重组这些同一控件实例；
module.yaml 驱动的参数和预测结果仍由运行时组件维护。页面的信息架构不再是需要从上滚到下
的六个等权区域，而是：

```text
左侧操作栏                         右侧工作台
1. Import data                    Input preview / Prediction result
   Single file | Folder batch    Display / zoom / curve controls
2. Import model                  Export results
   Basic setup + Import          Activity log（默认折叠）
   Advanced（默认折叠）
--------------------------------  Folder batch results
3. Predict（底部常驻）             filter / sort / trend / export
   readiness + Predict / Stop
```

运行 prediction 后，Preprocessed 展示所选 module 的 YAML 顺序以及每一步的真实输出缩略图；点击任一
缩略图会在主画布显示该步。selector 固定在主图上方，不需要先滚过主图；同名的重复步骤以
`(1/2)`、`(2/2)` 区分。原始强度步骤只在 preview 使用对数色阶以显示弱散射，mask/cut 使用的 `-1`
sentinel 也只在 preview 中作为无效像素隐藏；两种显示处理都不会改写送入模型的 tensor。结果页复用
该次 inference 已捕获的 snapshots，不会为了画图再次运行 preprocessing。YAML 声明的 mask 或
preprocessing 配置无效时，workflow 显示错误并停止，不再用隐式 fallback 产生看似成功但科学输入
不一致的预测。

联合分布页从 module output-axis contract 读取物理轴。Au 模型显示 `p(H,R)`（H 0.1–30 nm、R
0.05–15 nm），顶部/侧边 marginal 以及单独的 `p(H)`、`p(R)` 曲线共享同一组 bin centers；导出曲线
时也写出这些物理坐标，不再导出 0–29 的数组下标。

单文件模式只显示 detector file 和 Stack；Folder batch 只显示 folder、Range、
Files per prediction 和 Watch for new frames / Keep up with the detector，并实时显示文件数、job 数和无法组成完整 stack 的尾部文件数。
Watch：`bindings/frame_watch.py` 在批量跑完已有帧后继续看文件夹（`QFileSystemWatcher` 加每秒一次，列目录在 `TaskRunner` 里），
新帧写完整后（`application/frame_watch.FrameWatch`：EDF 看文件头的 Size，其他文件看两次相同）追加一行并交给
`batch_manager.MultiFilePredictManager`（队列，`follow` 时等待新帧）；Keep up（`KeepUp`）最多让两叠等待模型（`can_take` 时一次给一叠），更早的计数跳过。
文件夹变化后的下一次查看至少隔上一次查看所用的时间（`_watch_soon` 的间隔，至少 `SOON_MS`）：几万帧的文件夹列一次要零点几秒，一次接一次地看会让一个线程一直忙、与窗口争解释器；`adapters/fabio_images.look` 用 `_suffix` 和先拼好的文件夹前缀代替 `os.path.splitext` / `os.path.join`（52 000 个文件 0.13 s → 0.05 s）。批量结果按状态计数（`result_models.counts`，加行和改状态时增量维护），打开的趋势和分布窗口至多每 `WINDOWS_MS` 重画一次（`results_widget`）：5 万行时加一行 14.6 → 0.2 ms，改一次状态 17.6 → 约 2 ms，`tests/test_prediction_multifile_presentation.py`。
Show the newest result（Batch results 里、当前结果名称旁，默认勾选）：`bindings/live_canvas.py` 在任何文件夹批量运行时把最新完成的结果
画到画布（`_display_prediction(live=True)`：不写日志、保留正在看的子面板、不抢主 tab，只有一轮的第一个结果切到 Prediction result），
至多每 `SHOW_MS` 一次；页面不可见时不画；运行中用户点了某行就取消勾选。只有结果、没有 Input preview 时画布也离开空状态。
两种模式共用原有 prediction command 和数据结构，不是两套预测实现。

右侧复用 app presentation design system 的 `PlotPanel`。无输入时只显示下一步；输入加载
成功后自动显示 Input preview；预测成功或选择已完成的 batch row 后自动显示 Prediction
result。二维输出保留 AutoScale、LogScale、Colormap 和 Zoom，手动 Vmin/Vmax 收入
`Manual color range`；曲线输出只显示 Log X、Log Y、AutoScale 和轴范围，不混入无意义的
二维色阶。Input 与 Prediction export 是当前 tab 的上下文 action。

顶部工作流状态来自 input/model/framework/job/result 的成功状态，不再根据“点击过按钮”
推断完成。步骤可点击或通过键盘跳转；Predict/Stop 位于固定底部 action area。

单文件预测在后台线程运行（`presentation/prediction_worker.py`），窗口不会冻结；运行时显示 Stop。
Stop 后 Predict 显示 “Stopping…” 并保持禁用，直到模型进程结束，因此同一时间最多一个模型进程。
运行期间改了帧、module 或模型时，结果被丢弃并给出警告 toast，不会把两帧的结果混在一个视图里。
活动日志经信号写入（线程安全），Show Log 会滚动到日志处。界面上的图跟随主题，Export 写出的图
仍是原来的浅色出版样式。关闭 GIMaP 时若预测在运行，退出确认会列出 “2D Prediction”。1280 px 宽
的窗口下视图与面板上下排列。

输入接受 CBF、EDF、TIFF（`domain/sequences.FRAME_SUFFIXES`）：按钮为 “Choose frame…”，对话框标题 “Select Detector Frame”，
提示写明三种格式；空状态的按钮直接打开选择框；也可以把帧拖到图区，别的文件给出 “Unsupported Format”。帧由
`infrastructure/adapters/detector_images.py` 经 `shared/detector_io.load_detector_image` 读取：存储值不变，读取器标为无数据且
值 ≥ 0 的像素（存为 0 的 Pilatus 缝隙、EDF Dummy）给 −1；P03 CBF 与原 fabio 读取逐像素相同
（`tests/test_prediction_frames.py`）。文件夹扫描取多数编号帧的格式；叠加只取与第一帧同格式的文件。

预处理入口带 `context` 参数时，`ProfileFrameContexts`（bootstrap 注入 `AppContext.instrument_profiles`）给出帧的事实、
匹配仪器配置的几何与 `q_map()`；普通值写进第一步的 `frame_context`，从而进入导出记录。`io.geometry: required` 缺几何时
拒绝预测（`application/use_cases.PreparePredictionInput`）。模型在 `AppContext.model_jobs`（`integrations/jobs/kept_process.py`）
的常驻进程中运行，`integrations/tensorflow/worker.py` 保留最近两个模型；退出时 `window.py` 与 `_cancel_model_jobs` 关掉它。导出结果：保存对话框建议 `<输入文件名>_prediction.png`（PNG），数据以同名写在旁边，
另写 `<同名>.json` 记录（模块、模型路径、框架、运行时版本、预处理步骤、输入文件与叠加数），完成后 toast 可打开文件夹，
失败时显示错误。状态栏只显示当前打开的 Labs 页面自己的消息，切换语言后随之翻译。

## 控件映射

| 功能/控件区域 | 当前位置 | 行为 |
| --- | --- | --- |
| single/multi mode | `1. Import data / segmented selector` | 文案改为 Single file / Folder batch；原 mode signal 不变 |
| file picker + Stack | `Single file page` | 只在单文件模式显示；路径和 stacking 语义不变 |
| folder picker + Range + Every | `Folder batch page` | 显示为 Files per prediction；inclusive range 与 stacking 语义不变，新增 job summary |
| Watch for new frames + Keep up with the detector | `Folder batch page`，Every 下面一行 | 只在 Folder batch 显示；Keep up 默认勾选，只在 Watch 勾选时可用；运行中可勾选或取消 |
| Show the newest result | `Batch results`，当前结果名称同一行右侧 | 默认勾选；批量运行时画布跟随最新结果（≤ 1 次/s）；运行中点选一行即取消勾选 |
| module selector + model import/status | `2. Import model / Basic` | 主任务前置；`module.yaml`、manifest validation、model discovery 和 lazy loading 不变 |
| framework、reload、edit config | `Advanced model configuration` | 默认折叠；原 framework 检查和 config actions 不变 |
| DESY Model Library | `Advanced model sources` | 默认折叠；URL 和 Browse Models action 不变 |
| readiness、Predict、Stop | `sticky 3. Predict` | 原 signal、worker process 和 stop behavior 不变；真实状态驱动，Stop 运行时显示 |
| GISAXS/Predict-2D tabs 与 graphics views | shared `PlotPanel` | Input preview / Prediction result；按 widget identity 切换，不依赖可见文案 |
| display / curve controls | 当前 output 的 inspector | 二维 quick controls + 折叠手动范围；curve-only axis controls |
| 空 graphics view | `PlotPanel EmptyState` + canvas overlay | 只提供下一步提示，不拦截鼠标，不参与计算 |
| run log | `Activity log` | 默认折叠；错误与运行消息仍保留 |
| multi-file result widget | embedded `Batch results` | 复用原 filter/sort/trend/export 和 result selection，不再作为唯一外部窗口 |
| GISAXS Export、Predict-2D Export | PlotPanel contextual actions | Input 加载后可导出 input；预测完成后可导出 prediction |

页面布局和 presentation ownership 不修改 Prediction domain、use case、ViewModel、TensorFlow adapter 或
预测 workflow。TensorFlow 仍按需加载并通过 worker 边界运行；单文件和多文件
结果继续使用原数据结构。为保持 UI 契约，所有 View 控件、objectName、button instances、
tab 顺序及 `gisaxsPredictImageShowTabWidget` 均未拆开或替换。

## 手动验收清单

- [ ] Single file 只显示 detector file 和 Stack；
- [ ] 页面初始显示 Import detector data，不显示无效 display/export controls；
- [ ] 文件选择或路径回车成功后自动显示 Input preview，失败时 workflow 不前进；
- [ ] Folder batch 只显示 folder、Range 和 Files per prediction；
- [ ] Folder batch summary 的 file/job/skipped 数量与实际 grouping 一致；
- [ ] Watch for new frames：空文件夹也能开始；写入的新帧写完后出现新行并被预测；取消勾选后跑完已取用的帧即结束，Stop 立即结束；
- [ ] Keep up：帧比模型快时最多两帧等待、更早的跳过，状态栏给出跳过的帧数；模型够快时不跳过；不勾选时依次预测每一帧；
- [ ] Show the newest result：批量运行时画布每秒至多更新一次为最新结果，正在看的子面板不跳走；运行中点选一行后取消勾选并停在该行；
- [ ] Open batch results 能滚动到嵌入式结果表；
- [ ] module.yaml reload、module selector、framework 状态和 Edit Config 正常；
- [ ] Import Model、model manifest error 和 compatibility message 正常；
- [ ] Advanced model configuration 和 Advanced model sources 折叠/展开不改变已选 module/model；
- [ ] GISAXS Preview 的 current、display limits、log、colormap 和 zoom 正常；
- [ ] Preprocessed steps 顺序与 module.yaml 一致，重复步骤标签可区分，mask/cut 区域不会显示成高强度；
- [ ] 缺失 mask、无效 resize method 或未知 step 会停止预测并显示错误；
- [ ] Predict readiness 四项状态、Predict 与 Stop 正常；
- [ ] Predict 常驻可见，Stop 仅在 batch job 运行时显示；
- [ ] 成功加载 model 后 workflow 进入 Predict，失败或取消不显示完成；
- [ ] 预测成功后自动显示 Prediction result；
- [ ] 2D result 的 AutoScale、LogScale、manual range、colormap 和 zoom 正常；
- [ ] curve 的 Log X、Log Y、AutoScale 和手动轴范围正常且不显示色阶；
- [ ] 单文件结果与固定 fixture 数值一致；
- [ ] 多文件 batch 顺序、失败提示、aggregation 和结果窗口一致；
- [ ] Activity log 保留原进度、异常和完成消息；
- [ ] Input export 在 input 加载后可用，Prediction export 只在结果完成后可用；
- [ ] GISAXS Export 与 Predict-2D Export 使用原文件格式和内容。

## SEM 粒径与预测对比

预测结果区的 **SEM comparison…（SEM 对比…）** 可把当前半径分布带入 SEM 工具，选择 SEM 图像
或粒径 CSV 后直接生成并导出对比图。独立的 **Tools → SEM Particle Analysis…** 也可在没有模型
结果时提取、统计和导出 SEM 粒径。校准、分割审查、CSV 映射、归一化与来源记录见
[SEM 粒径分析与预测对比](sem.md)。
