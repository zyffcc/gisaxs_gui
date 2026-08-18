# Trainset target boundary

domain 拥有 geometry、plugin definitions 与稳定参数模型；application 提供 generation、
project storage、simulation ports/use cases 和 simulation orchestration；infrastructure
拥有配置序列化、数据生成与预处理、grid cache、local/Slurm backend、portable job package
及 Keras adapters；presentation 拥有嵌入式 PyQt 页面和交互画布。

生产调用链为 `TrainsetBuildPage → TrainsetViewBinding → TrainsetViewModel → use cases`；
旧 `TrainsetController` 仅是外部 import 兼容别名。

顶层 `trainset` 包与 `ui.trainset_build_page` 仅保留兼容模块名。导出的 portable job 会同时
携带这些兼容入口和 `src/gimap` 实现，因此既保持旧脚本路径，也不再让新 feature 反向依赖
顶层 legacy package。
