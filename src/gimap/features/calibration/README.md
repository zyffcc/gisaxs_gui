# Calibration feature

Geometry Calibration 由 domain、application、infrastructure 和 presentation 组成。
`ui.geometry_calibration_dialog` 只提供 feature-owned dialog 的 public import compatibility。

依赖方向为 `presentation -> application -> domain`，具体图像读取、JSON、
settings、路径、calibration engine 与文件读写由 infrastructure adapters 实现
ports。
