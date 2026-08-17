# TensorFlow integration

该 adapter 在 spawned worker process 中按需加载 TensorFlow 模型并执行预测。主 GUI
进程只处理可序列化的请求、结果和运行状态，不直接持有 TensorFlow/Keras 对象。

模型 artifact/manifest 的发现与静态校验在 `manifest.py` 完成，不需要导入 TensorFlow；
runtime 兼容性检查在隔离 worker 中完成。
