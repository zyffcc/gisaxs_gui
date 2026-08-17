# Calibration presentation

`CalibrationViewModel` 持有 UI state 和 commands，并只调用 application use cases。
现有 PyQt dialog 保留原布局、交互和导入路径，逐步改为经 ViewModel 加载图像、
运行算法、导入/导出和应用参数。`QMessageBox` / `QFileDialog` 仍只在 dialog 中。
