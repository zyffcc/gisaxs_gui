# Calibration feature

Geometry Calibration 按渐进式迁移组织为 domain、application、infrastructure 和
presentation。旧 `calibration` 包及 `ui.geometry_calibration_dialog` 入口继续有效；
迁移期间它们作为兼容层或已验证 scientific kernel 保留。

依赖方向为 `presentation -> application -> domain`，具体图像读取、JSON、
`global_params` 与 legacy calibration engine 由 infrastructure adapters 实现 ports。
