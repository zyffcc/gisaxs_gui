# Calibration domain

这里复用已通过回归测试的纯 scientific kernel：数据模型、标准表、几何换算、
standard detection、ring overlay geometry、manual refinement、显著差异判断、预处理、
径向分析、峰检测/匹配、中心估计、优化和候选排序。它们只依赖 Python、
NumPy / SciPy，不依赖 PyQt、文件读写、`global_params` 或应用运行时。

本层提供稳定 domain API。包含文件 fingerprint I/O 的 `CalibrationEngine` 不被声明为纯
domain，而由 infrastructure adapter 隔离。

掠入射标样图（硅片上的 AgBH）：`gisaxs_features.py` 找镜面反射杆（列中位数）并在找环和拟合时去掉，判断光束上方的
径向峰是不是水平亮带（Yoneda 带）而不是环，`LitHalf` 只取样品不遮挡的那一半图做方位分位数；`center_polish.py`
把粗的弧投票中心按亮像素径向剖面的锐度磨到一个像素以内（只对掠入射图）。`peak_matcher.py` 对 AgBH 去掉偏离共同距离的
级次，并给出各级环是否齐全（`order_completeness`，不是记录的字段，只用于排序）。距离范围在生成、拟合和最终筛选里都是
硬限制（`OUT_OF_RANGE`）。
