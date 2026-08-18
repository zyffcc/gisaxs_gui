# Classification 布局迁移记录

- 状态：当前页面与五个静态 panel 均已由 Classification feature 的 Python Views 拥有。
- 当前调用链：`ClassificationPage → ClassificationViewBinding → ClassificationViewModel → application use cases`。
- 页面静态外壳：`src/gimap/features/classification/presentation/views/classification_page_view.py`。
- 页面行为与 View binding：`src/gimap/features/classification/presentation/page.py`。
- 页面样式：`src/gimap/features/classification/presentation/styles/classification_page.qss`。
- 兼容入口：`ui/classification_page.py`。
- 最近验证：2026-08-18。

当前页面、`ClassificationViewBinding` 和 ViewModel 均由 app composition root 显式装配。
Binding 只承担 Qt 信号、dialogs、表格/图表渲染和 ViewModel state 映射；数据导入、训练、
embedding、prediction 和 artifact I/O 均经 application ports。旧 controller 名称只返回
同一个 binding，不存在第二套页面、样式或 orchestration。

`Ui_MainWindow` 只保留无业务控件的 `classificationPage` host。启动时不再创建随后会被
删除的旧 import、降维和 classification widgets；app composition root 把唯一的 feature-owned
页面装入 host。`ClassificationViewBinding.initialize()` 复用注入实例并连接 workflow signals。
Binding 的日志直接写入注入页面的 `logTextBrowser`；主窗口不再安装任何 Classification widget
alias。Binding 强制要求页面依赖，也不再包含 host/layout
fallback；旧 import/降维/classifier 属性、隐藏 class list 及其他占位控件均已删除。

页面顶层 header、workflow stepper、四个滚动 workspace、Input/Preview/Configure/Results/
Export section 和 Log 折叠区以 `classification_page_view.py` 为来源。Dataset、Inspection、
Preprocessing、Experiment/Run 和 Results 的静态控件分别位于同目录的五个命名明确的 panel
Python Views。`page.py` 不再包含对应 `_build_*_panel` 实现，只创建 panel、保留兼容属性并
处理 drag/drop、step state、responsive splitter 和动态 class cards。

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

页面迁移没有修改 Classification domain、feature extraction、classifier adapters、降维算法、
排名规则、模型格式或 application use cases。所有原 objectName、signals、drag/drop、按钮实例
和输入输出控件保持不变。`ClassificationViewBinding` 只把 Qt events 映射到
`ClassificationViewModel`；共享 `JobStatus` 通过旧 `runStatusLabel` 和 `taskProgressBar` 属性
保持 UI 契约兼容。

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
