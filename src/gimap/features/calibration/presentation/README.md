# Calibration presentation

`views/geometry_calibration_dialog_view.py` 是静态布局的唯一 Python 来源。
`dialog.py` 绑定 ViewModel、worker signals，并把 Matplotlib canvas/toolbar 安装到 View 中
命名明确的 hosts。路径、standard detection、ring geometry、manual refinement 和显著差异
判断不在 dialog 中实现；`QMessageBox` / `QFileDialog` 与 Matplotlib 渲染仍只在 dialog 中。

旧的 `ui.geometry_calibration_dialog` 别名已删除；调用方直接导入本目录的 `dialog.py`。
