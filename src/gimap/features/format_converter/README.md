# Format Converter feature

Format Converter 的 dialog、ViewModel、转换 use case、格式规则和文件 adapter 均由本
feature 拥有。依赖方向为：

```text
presentation → application → domain
infrastructure → implements application ports
```

`utils.format_converter` 和 `ui.format_converter_dialog` 只提供 public import alias。PyQt
dialog 的唯一实现通过 ViewModel 调用 application use cases；核心转换不依赖 QApplication。
