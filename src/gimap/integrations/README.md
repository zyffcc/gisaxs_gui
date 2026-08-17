# `integrations`

`integrations` 保存可被多个 adapter 使用的外部系统集成支持，例如 BornAgain、
TensorFlow/Keras、文件格式或本地文件系统的低层封装。Feature-specific adapter 仍应放在
对应 feature 的 `infrastructure/adapters/` 中。

本层可以依赖外部库以及稳定 domain/application contracts，但不得依赖 presentation，
不得包含 feature 工作流，也不得直接进行用户交互。
