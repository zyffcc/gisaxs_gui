# Fitting presentation

本层保存 framework-neutral ViewModel state 和后续 Qt bridge。ViewModel 只管理 UI state、
commands 和 use-case 结果映射，不执行 scientific calculation、具体 I/O 或 dialog。
现有 controller 在迁移期作为 Qt 信号、dialog 和绘图兼容桥。
