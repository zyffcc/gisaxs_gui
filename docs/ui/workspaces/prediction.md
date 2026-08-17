# Prediction 布局迁移记录

## 控件映射

| 迁移前控件/区域 | 迁移后位置 | 行为 |
| --- | --- | --- |
| single/multi mode、file/folder、range/every | `Input / ParameterSection` | 单文件、多文件、inclusive range 和 stacking 语义不变 |
| module、framework、model import/status | `Configure / ParameterSection` | `module.yaml`、manifest validation、model discovery 和 lazy loading 不变 |
| DESY Model Library | `Advanced model sources / AdvancedSection` | 默认折叠；URL 和 Browse Models action 不变 |
| GISAXS/Predict-2D tabs 与 graphics views | `Preview / PlotPanel` | 保留原主 `QTabWidget`，controller 的 tab selection 和 inner result tabs 不变 |
| readiness labels、Predict、Stop | `Run / ParameterSection` | 原 controller signal、worker process 和 stop behavior 不变 |
| run log、Show Multi-File Results | `Results / ParameterSection` | 原 text browser 与 multi-file aggregation action 不变 |
| GISAXS Export、Predict-2D Export | `Export / ParameterSection` | 复用原按钮实例；格式、当前图像和结果输出不变 |

本轮没有修改 Prediction domain、use case、ViewModel、TensorFlow adapter 或 controller 的预测
workflow。TensorFlow 仍按需加载并通过原 worker 边界运行；单文件和多文件结果继续使用原数据
结构。为了保持 controller 兼容，`gisaxsPredictImageShowTabWidget` 没有拆开或替换。

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
