# Classification feature

Classification 按 `presentation → application → domain` 组织；`infrastructure` 实现数据文件、
ML runtime、JobRunner 与模型文件 ports。工作台的 public application API 支持：

- 导入已标注、暂定标签或无标签的 1D/2D 数据；
- 按 1D/2D compatibility group 构造 feature matrix；
- 无标签降维和 clustering suggestions；
- 人工 assign、clear、accept suggested labels；
- 仅使用 accepted labels 的模型训练，以及 model apply/export。

Application 通过 `ClusteringPort` 隔离可选聚类 runtime，source/session 的 `label_mode` 明确区分
accepted、provisional 与 unlabeled。Classification 页面和专属样式由本 feature 拥有；生产运行时直接构造
`ClassificationViewBinding`。降维、clustering 和训练通过 JobRunner ports 执行。旧
`ui.classification_page` 与 `controllers.classification_*` 名称仅作为兼容入口。完整用户工作流和
session 字段说明见 `docs/ui/workspaces/classification.md`。顶部 workflow header 的状态由真实
data/embedding/label/model artifact 驱动，导航位置不会改变步骤完成度。
