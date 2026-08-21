# Prediction 界面与交互说明

- **Status**: Current
- **Scope**: GIMaP Prediction workspace 的 PyQt layout ownership、控件映射与手动验收
- **Related code**:
  [`src/gimap/features/prediction/presentation/`](../../../src/gimap/features/prediction/presentation/)、
  [`ui/components/main_window_components.py`](../../../ui/components/main_window_components.py)、
  [`ui/main_window.py`](../../../ui/main_window.py)
- **Related tests**:
  [`tests/test_prediction_presentation.py`](../../../tests/test_prediction_presentation.py)、
  [`tests/test_prediction_multifile_presentation.py`](../../../tests/test_prediction_multifile_presentation.py)、
  [`tests/test_prediction_view_model.py`](../../../tests/test_prediction_view_model.py)、
  [`tests/test_ui_workspace_layouts.py`](../../../tests/test_ui_workspace_layouts.py)
- **Last verified**: 2026-08-19

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

`ui/components/main_window_components.py` 不再定义 Prediction workspace 或专用 cards，只从
feature public presentation API 导入并在 application shell 中组装，因此旧 import path 仍返回
同一个 feature-owned class，不存在第二套页面实现。

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

单文件模式只显示 detector file 和 Stack；Folder batch 只显示 folder、Range 和
Files per prediction，并实时显示文件数、job 数和无法组成完整 stack 的尾部文件数。
两种模式共用原有 prediction command 和数据结构，不是两套预测实现。

右侧复用 app presentation design system 的 `PlotPanel`。无输入时只显示下一步；输入加载
成功后自动显示 Input preview；预测成功或选择已完成的 batch row 后自动显示 Prediction
result。二维输出保留 AutoScale、LogScale、Colormap 和 Zoom，手动 Vmin/Vmax 收入
`Manual color range`；曲线输出只显示 Log X、Log Y、AutoScale 和轴范围，不混入无意义的
二维色阶。Input 与 Prediction export 是当前 tab 的上下文 action。

顶部工作流状态来自 input/model/framework/job/result 的成功状态，不再根据“点击过按钮”
推断完成。步骤可点击或通过键盘跳转；Predict/Stop 位于固定底部 action area，Stop 只在
batch job 运行时出现。

## 控件映射

| 功能/控件区域 | 当前位置 | 行为 |
| --- | --- | --- |
| single/multi mode | `1. Import data / segmented selector` | 文案改为 Single file / Folder batch；原 mode signal 不变 |
| file picker + Stack | `Single file page` | 只在单文件模式显示；路径和 stacking 语义不变 |
| folder picker + Range + Every | `Folder batch page` | 显示为 Files per prediction；inclusive range 与 stacking 语义不变，新增 job summary |
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
- [ ] Open batch results 能滚动到嵌入式结果表；
- [ ] module.yaml reload、module selector、framework 状态和 Edit Config 正常；
- [ ] Import Model、model manifest error 和 compatibility message 正常；
- [ ] Advanced model configuration 和 Advanced model sources 折叠/展开不改变已选 module/model；
- [ ] GISAXS Preview 的 current、display limits、log、colormap 和 zoom 正常；
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
