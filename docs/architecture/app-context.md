# AppContext 与应用状态边界

> **Status**：Current
>
> **Scope**：应用级依赖注入、settings、session、preferences 和 project state
>
> **Related code**：`src/gimap/app/context.py`、`src/gimap/app/bootstrap.py`、
> `src/gimap/integrations/state/`
>
> **Last verified**：2026-08-20

## AppContext

`main.py` 的 composition root 为一个应用进程创建一个 `AppContext`，再通过构造函数把它交给
workspace 和 feature。Context 当前包含：

- `SettingsRepository`：跨启动持久化的用户设置；
- `SessionRepository`：项目与 feature 会话快照；
- `UserPreferencesRepository`：纯 UI/交互偏好；
- `ProjectParametersRepository`：项目参数文件边界；
- `JobRunner`：可取消、可超时的后台任务边界；
- `ProjectState`：当前项目路径、dirty flag、metadata 和 feature state 聚合。

Feature 不得创建自己的全局 context 或 application singleton。Application 和 ViewModel 只依赖
所需的 repository/port，不应把完整 Context 当作 service locator 到处传递。

## `core/global_params.py` 兼容边界

`GlobalParameterManager` 仍为现有 JSON 参数格式和 Qt 参数注册提供兼容：

- beam、detector、sample、trainset、fitting 等用户参数的内存存储；
- `default_parameters.json` 与 `user_parameters.json` 的初始化、读取和写入；
- 嵌套参数路径访问；
- Qt 参数变化信号；
- QWidget 注册和值同步；
- public controller alias 的注册和值同步；
- process-wide singleton 生命周期。

`GlobalParamsSettingsRepository` 只把该 manager 映射为 `SettingsRepository`，不会创建第二个
singleton，也不会改变 `default_parameters.json` 或 `user_parameters.json` 的顶层 mapping、
嵌套参数名和序列化格式。Qt signals、widget registry 和 controller alias registry 不属于
repository contract，不能泄漏到 application/domain。
