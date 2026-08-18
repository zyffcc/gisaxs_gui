# Format Converter feature

本 feature 是第一个从 legacy 代码渐进迁移到 `src/gimap` 的功能。依赖方向为：

```text
presentation → application → domain
infrastructure → implements application ports
```

`utils.format_converter` 暂时保留 domain/infrastructure 旧 API，
`ui.format_converter_dialog` 暂时保留为 presentation import 兼容入口。PyQt dialog 的
唯一实现已经由本 feature 拥有，并通过 ViewModel 调用 application use cases；核心
转换不依赖 QApplication。
