# Trainset ports

`SimulationPort` 接收普通 dict scientific input，返回 NumPy intensity array。
具体 BornAgain runtime 只能存在于 integration adapter 的 worker process。

`ModelContractPort` 隔离模型 runtime forward-pass 验证；application tests 使用 fake port，
TensorFlow 只允许由 infrastructure worker 加载。
