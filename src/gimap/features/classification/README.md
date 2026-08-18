# Classification feature

分类功能按 `presentation → application → domain` 组织；`infrastructure` 实现数据文件、
ML runtime、JobRunner 与模型文件 ports。Classification 页面和专属样式由本 feature
拥有；生产运行时直接构造 `ClassificationViewBinding`，训练/降维通过 JobRunner ports。
旧 `ui.classification_page` 与 `controllers.classification_*` 名称仅作为兼容入口。
