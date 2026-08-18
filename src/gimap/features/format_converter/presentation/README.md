# Presentation

`views/format_converter_dialog_view.py`、`views/folder_import_dialog_view.py` 和
`views/conversion_progress_dialog_view.py` 分别是主对话框、目录导入窗和进度窗的 Python 布局
唯一来源。`dialog.py` 注入 ViewModel 并连接 commands；共享 `JobStatus` 由进度 View 放置。
`FormatConverterViewModel` 保存输入源 UI state，并通过 use cases 执行检查、
预览、估算和转换。`QFileDialog` 与 `QMessageBox` 留在 dialog；文件读写和格式规则不在
presentation 中实现。

旧入口 `ui.format_converter_dialog` 只 re-export 本目录中的类，供尚未迁移的 caller
继续使用，不再拥有第二套 dialog 实现。
