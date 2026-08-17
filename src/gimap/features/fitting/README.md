# Fitting feature

Fitting 按 feature-first 方式渐进迁移。`domain` 保存可脱离 Qt、文件系统和外部
runtime 验证的科学数据结构与数值运算；`application` 编排 use cases 和 ports；
`infrastructure` 实现文件及模型 adapters；`presentation` 保存 ViewModel 和 Qt bridge。

旧入口仍位于 `controllers/fitting_controller.py`，迁移期间由兼容调用委托给本 feature。
