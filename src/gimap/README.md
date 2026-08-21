# `gimap` 生产源码

这里是 GIMaP 的生产源码根目录。`main.py` 通过 `app/` composition root 创建应用上下文、
主窗口和各 feature；业务实现由对应 feature 拥有。

目录职责：

- `app/`：composition root、启动装配和全局 navigation；
- `features/`：按业务 feature 组织 presentation、application、domain 和 infrastructure；
- `shared/`：至少被两个 feature 稳定需要且 ownership 明确的 scientific kernel；
- `integrations/`：外部 runtime、engine 和 I/O 系统的集成支持；
- `plugins/`：插件契约、发现和生命周期支持。

Feature 内部依赖方向必须为：

```text
presentation → application → domain

infrastructure → implements application ports
```

具体规则见 `docs/architecture/overview.md` 和
`docs/architecture/dependency-rules.md`。
