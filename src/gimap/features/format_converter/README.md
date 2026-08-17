# Format Converter feature

本 feature 是第一个从 legacy 代码渐进迁移到 `src/gimap` 的功能。依赖方向为：

```text
presentation → application → domain
infrastructure → implements application ports
```

`utils.format_converter` 和 `ui.format_converter_dialog` 暂时保留为兼容入口。核心转换
不依赖 QApplication；Qt dialog 通过 ViewModel 调用 application use cases。
