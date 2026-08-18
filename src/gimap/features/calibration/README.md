# Calibration feature

Geometry Calibration 按渐进式迁移组织为 domain、application、infrastructure 和
presentation。旧 `calibration` 包继续作为已验证 scientific kernel；
`ui.geometry_calibration_dialog` 只保留为 feature-owned dialog 的 import compatibility。

依赖方向为 `presentation -> application -> domain`，具体图像读取、JSON、
settings、路径、legacy calibration engine 与文件读写由 infrastructure adapters 实现
ports。
