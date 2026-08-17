# Presentation

`FormatConverterViewModel` 保存输入源 UI state，并通过 use cases 执行检查、预览、估算
和转换。现有 `ui.format_converter_dialog.FormatConverterDialog` 保持原入口；本目录的
`dialog.py` 提供新路径到 legacy dialog 的临时兼容 bridge。
