# Calibration domain

这里复用已通过回归测试的纯 scientific kernel：数据模型、标准表、几何换算、
预处理、径向分析、峰检测/匹配、中心估计、优化和候选排序。它们只依赖 Python、
NumPy / SciPy，不依赖 PyQt、文件读写、`global_params` 或应用运行时。

当前这些算法的 canonical implementation 仍在旧 `calibration` 包；本层提供目标
架构下的稳定 domain 入口。包含文件 fingerprint I/O 的 legacy `CalibrationEngine`
不被声明为纯 domain，而是暂由 infrastructure adapter 隔离。
