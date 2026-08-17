# Classification 布局迁移记录

## 控件映射

| 迁移前控件/区域 | 迁移后位置 | 行为 |
| --- | --- | --- |
| class cards、Scan & Import、dataset table/QC | `Input / ParameterSection` | source、include/exclude、QC、search/filter 和 drag/drop 不变 |
| sample browser、image controls、quality list | `Preview / PlotPanel` | sample order、shape、log/colormap/range 和 Fit 不变 |
| 1D/2D pipeline、normalize/log、input summary | `Configure / ParameterSection` | shared preprocessing config 和 feature matrix 语义不变 |
| smoothing、resize rows/cols | `Advanced preprocessing / AdvancedSection` | 默认折叠；值和 input shape 行为不变 |
| algorithm selection/table | `Algorithms > Configure` | recommended/all/clear/defaults 和 classifier parameter dialogs 不变 |
| validation、seed、ranking、PCA/UMAP | `Advanced validation and projection / AdvancedSection` | 默认折叠；split、ranking 和 projection 行为不变 |
| Run Comparison、Cancel、status/progress | `Run / ParameterSection + JobStatus` | 仍由现有 `ClassificationViewModel` 和 JobRunner 执行 |
| leaderboard、confusion、metrics、misclassified、embedding、prediction | `Results / ParameterSection` | ranking、tables、active model 和 predictions 不变 |
| Save Active Model、Export Results、Export Prediction CSV | `Export / ParameterSection` | 复用原按钮实例和保存/导出 adapters |
| operation log | `Log / AdvancedSection` | 原 log browser 不变，默认折叠 |

本轮没有修改 Classification domain、feature extraction、classifier adapters、降维算法、排名
规则、模型格式或 application use cases。Legacy controller 仍只把 Qt events 映射到已经存在的
`ClassificationViewModel`；共享 `JobStatus` 通过旧 `runStatusLabel` 和 `taskProgressBar` 别名
保持 controller 兼容。

## 手动验收清单

- [ ] Add Class、Scan & Import、drag/drop 和 class source editing 正常；
- [ ] dataset search/filter、include/exclude/remove/open/copy/export list 正常；
- [ ] Preview previous/next、log、colormap、auto/manual range、Fit 正常；
- [ ] 1D/2D preprocessing、normalize/log 配置与迁移前一致；
- [ ] Advanced preprocessing 折叠/展开不重置 smoothing 或 resize；
- [ ] classifier recommended/all/clear/defaults 与 parameter dialogs 正常；
- [ ] validation、folds、repeat、seed、ranking、PCA/UMAP 配置不变；
- [ ] Run Comparison 使用 worker process，Cancel 和异常不会退出 GUI；
- [ ] JobStatus 的 running/succeeded/failed 和 0–100 progress 正确；
- [ ] leaderboard ranking、confusion normalization、per-class metrics 和 misclassified 正确；
- [ ] embedding 仍区分 visualization-only t-SNE 与可保存 projection；
- [ ] active model、Predict New Data、prediction table 正常；
- [ ] model/result/prediction 导出格式和内容不变；
- [ ] New/Load/Save Session 与 operation log 正常。
