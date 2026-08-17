# AppContext 与 global_params 渐进迁移

## 当前 `core/global_params.py` 的职责

现有 `GlobalParameterManager` 同时负责：

- beam、detector、sample、trainset、fitting 等用户参数的内存存储；
- `default_parameters.json` 与 `user_parameters.json` 的初始化、读取和写入；
- 嵌套参数路径访问；
- Qt 参数变化信号；
- QWidget 注册和值同步；
- legacy controller registry 与 controller 同步；
- process-wide singleton 生命周期。

这些职责暂不一次性拆除。旧 controller 可以继续使用 `global_params`；新 feature 只依赖
`SettingsRepository`、`SessionRepository` 和显式传入的 `AppContext`。

## 第一版边界

- `SettingsRepository`：跨启动持久化的用户设置，JSON 仍是原有顶层 module mapping，
  不增加 schema envelope，也不改变嵌套参数名称。
- `SessionRepository`：新的、可替换的临时 session 持久化边界。
- `ProjectState`：当前项目路径、dirty flag、metadata 和 feature state 聚合。
- `FeatureState`：feature state 的 `snapshot` / `restore` 最小协议。
- `AppContext`：由 `main.py` 创建一次，通过构造函数传给新架构 feature。

`GlobalParamsSettingsRepository` 是迁移适配器。它不会创建另一个 singleton，只包装现有
manager。迁移完成前，Qt signal、widget registry 和 controller registry 仍属于 legacy
`global_params`，不进入新的 repository contract。
