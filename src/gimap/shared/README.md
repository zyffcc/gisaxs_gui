# `shared`

`shared` 不是默认放置位置，也不是新的 `utils/`。只有至少两个 feature 已经稳定需要
同一项领域能力，并且语义、边界和 ownership 明确时，才能在这里提取 shared
scientific kernel。

Shared domain primitives 可以依赖 Python 标准库、NumPy，以及适合稳定数值原语的
SciPy；不得依赖 PyQt/PySide、TensorFlow/Keras、BornAgain 或具体文件系统实现。
禁止为了“未来可能复用”提前创建 abstraction。
