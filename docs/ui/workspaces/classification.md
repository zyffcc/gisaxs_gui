# Classification 工作台

- **Status**: Current
- **Scope**: Classification 的导入、预处理、降维、人工标注、辅助分组、监督训练和新数据分类界面
- **Related code**: `src/gimap/features/classification/`
- **Related tests**: `tests/test_classification_*.py`、`tests/test_ui_workspace_layouts.py`、`tests/test_ui_source_of_truth.py`
- **Last verified**: 2026-08-24

## 用户目标与主流程

工作台接受 1D 曲线、2D 图像或二者混合的数据；数据可以有既有标签，也可以完全无标签。主流程固定为：

```text
1 Data → 2 Prepare → 3 Explore & label → 4 Train & review → 5 Apply & export
```

工作流将“观察数据”和“训练分类器”解耦。导入和降维不要求标签；只有训练阶段要求至少两个已接受的类别。普通参数提交不会隐式启动降维、聚类或训练，也不会自动切换步骤。

顶部深色 workflow header 与 Fitting 工作台使用同一套视觉语言，同时严格区分两种状态：蓝色选中态只表示当前正在查看的步骤；available/running/complete/stale/error/blocked 则来自实际 data、embedding、accepted labels、training result 与 prediction artifact。点击步骤只导航，不会伪造完成状态。`Guided / Compact` 只切换说明密度，不改变流程和数据。

## 1 Data：先接住数据

`Add data` 是默认入口：可以选择若干文件或一个文件夹，导入时按无标签数据处理。用户也可以用 `Add labeled source` 明确创建已有类别，或用 `Scan and import` 检查配置后的 sources。拖放文件默认无标签；拖放文件夹时由用户确认文件夹名是已接受标签，还是仅作为来源名。

每个 source 的 `label_mode` 明确记录标签含义：

| `label_mode` | 含义 | 是否参与训练 |
| --- | --- | --- |
| `accepted` | 文件夹/来源名称是用户确认的真实类别 | 是 |
| `provisional` | 文件夹名称只是待确认建议 | 否，接受后才参与 |
| `unlabeled` | 来源不携带类别信息 | 否 |

混合 1D/2D 数据不会被拒绝。顶部 `Active data group` 按数据维度建立兼容组；Preview、Prepare、Explore 和 Train 只消费当前组，避免把不同科学语义的 feature vector 混进同一次计算。切换组只改变工作上下文，不重置页面步骤。

## 2 Prepare：建立确定性特征配方

Prepare 为当前组配置 1D 插值或 2D crop/resize、normalize、log transform 等预处理。主配置默认可见；smoothing 和目标尺寸放在 Advanced。点击 `Continue to explore` 只导航到 Explore，不隐式计算。

同一组的降维、分组和训练都通过同一份 preprocessing config 构造 feature matrix。工作台没有改变既有 1D/2D 数值预处理定义、array orientation 或 model package 中保存的 recipe。

## 3 Explore & label：人机协同闭环

Explore 包含两个互不替代的动作：

- `Build 2D map`：用 PCA、UMAP 或 t-SNE 生成二维视图。无标签样本也可运行。
- `Suggest groups`：在完整 feature matrix 上用 HDBSCAN 或 K-Means 提出复核组，不把二维图坐标当作分类输入。

散点图支持点击、框选、`Ctrl+wheel` 缩放、`Esc` 清空选择、双击联动预览。颜色可以按已接受标签、建议分组、来源、QC 或预测结果切换。

人工决策保持显式：

1. 选中一个或多个样本；
2. 输入真实类别并 `Apply class label`，或接受算法给出的 selected/all suggestions；
3. 不确定的数据使用 `Keep unlabeled`；
4. 只有 `label_status=accepted` 的样本进入训练。

HDBSCAN 的 noise 显示为 `Noise / review`，仍需人工处理。算法建议保存在 `suggested_label` / `suggestion_source`，不会静默覆盖 accepted label。

## 4 Train & review：只用确认标签

训练按钮只有在当前组至少有两个已接受类别且数据质量允许时启用。模型比较仍通过 JobRunner worker 执行；validation、seed、ranking 和可保存 projection 位于 Advanced。结果区展示 leaderboard、confusion matrix、per-class metrics 和 misclassified samples，用户显式选择 active model。

改变 active group、preprocessing、样本 include 状态或标签后，已有结果会标为 outdated，但界面不会自动重跑或跳页。

## 5 Apply & export：模型消费区

Apply 可使用当前 active model 或载入已有 model package，对新文件执行 package 内保存的 preprocessing 和 projection。预测表与训练评估分离，避免把“模型评审”和“批量应用”混成同一个结果页。Export 提供 active model、训练结果 CSV 和 prediction CSV。

## 状态与持久化

Session 保存 source 配置、预处理/验证/算法参数，以及按文件路径记录的 `label_overrides`。建议标签与 accepted label 在运行时保持独立；重新导入同一路径后应用人工覆盖。Embedding 和 clustering 是可重算的视图/建议，不作为第二份科学数据源。

当前调用链为：

```text
ClassificationPage
  → ClassificationViewBinding
  → ClassificationViewModel
  → application use cases / ports
  → infrastructure adapters and JobRunner workers
```

Presentation 只负责 Qt signals、dialogs、选择联动和 state rendering；导入、feature construction、降维、分组、训练、预测与 artifact I/O 均通过 application API。
分组建议的外部运行能力由 application-owned `ClusteringPort` 定义并由 infrastructure adapter 实现；source/session 配置新增的 `label_mode` 是 accepted/provisional/unlabeled 语义的唯一持久化字段。

## 响应式与手动验收

页面在 1280×800 使用纵向 Data/Explore splitter，在 1440×900 和 1920×1080 使用横向布局；workflow header 可由用户在 Guided/Compact 间切换说明密度。一个步骤只保留一个主要滚动容器；顶部 active group 和五步导航位置稳定。

- [ ] 可分别导入 labeled folders、provisional folders、unlabeled files/folders；
- [ ] 同批 1D/2D 自动出现两个 active groups，组间 preview/selection 不串数据；
- [ ] 无标签组可以完成 feature construction、PCA/UMAP 和 grouping；
- [ ] scatter 点击、框选、Fit、缩放、清空和预览联动正常；
- [ ] suggestion 不会自动成为训练标签；Apply/Accept/Keep unlabeled 后训练状态即时更新；
- [ ] workflow 的导航选中态与 verified progress 独立；切换步骤不改变 complete/stale/blocked；
- [ ] 参数编辑和自动刷新不切 tab、不滚动、不启动长任务；
- [ ] accepted labels 满足条件后可训练，结果表和 active model 正常；
- [ ] 新数据分类复用 model package recipe，结果与训练 review 分区；
- [ ] Session 重载后 sources、label modes 和 manual label overrides 恢复；
- [ ] 1280×800、1440×900、1920×1080 无水平裁切、核心命令可见、无重复框套框。
