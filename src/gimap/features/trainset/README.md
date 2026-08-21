# Trainset feature

domain 拥有 geometry、plugin definitions 与稳定参数模型；application 提供 generation、
project storage、simulation ports/use cases 和 simulation orchestration；infrastructure
拥有配置序列化、数据生成与预处理、grid cache、local/Slurm backend、portable job package
及 Keras adapters；presentation 拥有嵌入式 PyQt 页面和交互画布。

生产调用链为 `TrainsetBuildPage → TrainsetViewBinding → TrainsetViewModel → use cases`；
顶层 `TrainsetController` 仅是 public import alias。

顶层 `trainset` 包与 `ui.trainset_build_page` 仅提供兼容模块名。导出的 portable job 会同时
携带这些 aliases 和 `src/gimap` 实现，以保持已有脚本路径；feature 不反向依赖顶层包。
