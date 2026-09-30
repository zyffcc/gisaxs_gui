# Trainset feature

domain 拥有 geometry、plugin definitions 与稳定参数模型；application 提供 generation、
project storage、simulation ports/use cases 和 simulation orchestration；infrastructure
拥有配置序列化、数据生成与预处理、grid cache、local/Slurm backend、portable job package
及 Keras adapters；presentation 拥有嵌入式 PyQt 页面和交互画布。

生产调用链为 `TrainsetBuildPage → TrainsetViewBinding → TrainsetViewModel → use cases`。

旧的顶层 `trainset` 包、`TrainsetController` 与 `ui.trainset_build_page` 兼容别名已删除。导出的
portable job 只携带 `src/gimap` 实现；已注册模型的 `preprocess.py` 依赖的
`src.gimap.features.trainset.infrastructure.adapters.dataset_generator` 保持为稳定路径。
