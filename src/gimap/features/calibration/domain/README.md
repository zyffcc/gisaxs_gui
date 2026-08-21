# Calibration domain

这里复用已通过回归测试的纯 scientific kernel：数据模型、标准表、几何换算、
standard detection、ring overlay geometry、manual refinement、显著差异判断、预处理、
径向分析、峰检测/匹配、中心估计、优化和候选排序。它们只依赖 Python、
NumPy / SciPy，不依赖 PyQt、文件读写、`global_params` 或应用运行时。

本层提供稳定 domain API。包含文件 fingerprint I/O 的 `CalibrationEngine` 不被声明为纯
domain，而由 infrastructure adapter 隔离。
