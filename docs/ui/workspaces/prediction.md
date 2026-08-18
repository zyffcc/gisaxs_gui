# Prediction 布局迁移记录

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
- **Last verified**: 2026-08-18

## 当前状态

Prediction 的唯一 workspace layout 实现位于 feature 的 `presentation/workspace.py`，专用
cards、响应式 control styling 和 preview layout 也由同一 feature 拥有：

```text
feature-owned Python View factory
                 ↓
feature-owned GisaxsPredictWorkspace
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

- `views/prediction_page_view.py` 保存与 legacy binding 兼容的原始控件、objectName、默认值和
  GISAXS/Predict-2D tabs；
- `views/prediction_workspace_view.py` 保存用户实际看到的 Input、Configure、Advanced、Preview、
  Run、Results、Export section hierarchy；
- `views/multifile_results_widget_view.py` 保存多文件结果过滤、排序、表格和 action bar；
- `views/export_dialog_view.py`、`views/distribution_heatmap_dialog_view.py`、
  `views/parameter_trend_dialog_view.py` 保存相应辅助窗体的静态布局，Matplotlib canvas 仍在运行时
  注入各自的 `plot_host`。

`control_view_factory.py` 只实例化 `PredictionPageView` 并把既有属性暴露给 application shell。
`workspace.py` 通过 `PredictionWorkspaceView` 的语义化 host
重组这些同一控件实例；module.yaml 驱动的参数和预测结果仍由运行时组件维护。

## 控件映射

| 迁移前控件/区域 | 迁移后位置 | 行为 |
| --- | --- | --- |
| single/multi mode、file/folder、range/every | `Input / ParameterSection` | 单文件、多文件、inclusive range 和 stacking 语义不变 |
| module、framework、model import/status | `Configure / ParameterSection` | `module.yaml`、manifest validation、model discovery 和 lazy loading 不变 |
| DESY Model Library | `Advanced model sources / AdvancedSection` | 默认折叠；URL 和 Browse Models action 不变 |
| GISAXS/Predict-2D tabs 与 graphics views | `Preview / PlotPanel` | 保留原主 `QTabWidget`，binding 的 tab selection 和 inner result tabs 不变 |
| readiness labels、Predict、Stop | `Run / ParameterSection` | 原 signal、worker process 和 stop behavior 不变 |
| run log、Show Multi-File Results | `Results / ParameterSection` | 原 text browser 与 multi-file aggregation action 不变 |
| GISAXS Export、Predict-2D Export | `Export / ParameterSection` | 复用原按钮实例；格式、当前图像和结果输出不变 |

本次所有权迁移没有修改 Prediction domain、use case、ViewModel、TensorFlow adapter 或
预测 workflow。TensorFlow 仍按需加载并通过 worker 边界运行；单文件和多文件
结果继续使用原数据结构。为保持 UI 契约，所有 View 控件、objectName、button instances、
tab 顺序及 `gisaxsPredictImageShowTabWidget` 均未拆开或替换。

## 手动验收清单

- [ ] Single File 与 Multi Files 切换时 input controls 显示正确；
- [ ] file/folder picker、range、Every 和 Show Multi-File Results 正常；
- [ ] module.yaml reload、module selector、framework 状态和 Edit Config 正常；
- [ ] Import Model、model manifest error 和 compatibility message 正常；
- [ ] Advanced model sources 折叠/展开不改变已选 module/model；
- [ ] GISAXS Preview 的 current、display limits、log、colormap 和 zoom 正常；
- [ ] Predict readiness 四项状态、Predict 与 Stop 正常；
- [ ] Predict-2D 结果 tab、curve、display limits、colormap 和 zoom 正常；
- [ ] 单文件结果与迁移前 fixture 数值一致；
- [ ] 多文件 batch 顺序、失败提示、aggregation 和结果窗口一致；
- [ ] Run Log 保留原进度、异常和完成消息；
- [ ] GISAXS Export 与 Predict-2D Export 使用原文件格式和内容。
