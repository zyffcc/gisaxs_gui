# `features`

每个业务能力在这里拥有独立 feature，例如 fitting、prediction、trainset、
classification、WAXS 或 calibration。创建实际 feature 时采用：

```text
feature_name/
    presentation/
    application/
        ports/
    domain/
    infrastructure/
        adapters/
```

默认调用链为 `PyQt View → ViewModel → Use Case`。Application 依赖 domain；
infrastructure 实现 application ports。跨 feature 协作只能通过真正的 public
application API、明确 port/interface 或 ownership 清晰的 shared scientific kernel。
