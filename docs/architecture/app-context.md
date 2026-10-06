# AppContext 与用户状态

> **Status**：Current
>
> **Scope**：应用级依赖注入、设置、偏好、会话、仪器配置、项目文件，以及它们存在哪里
>
> **Related code**：`src/gimap/app/context.py`、`src/gimap/app/bootstrap.py`、`src/gimap/app/project.py`、
> `src/gimap/integrations/state/`（`user_store.py`）
>
> **Related tests**：`tests/test_architecture_dependencies.py::test_settings_live_in_the_user_store_only`、`tests/conftest.py`
>
> **Last verified**：2026-10-06（逐条对照代码）

## AppContext

`main.py` 的 `main()` 调用 `create_app_context()`（`src/gimap/app/bootstrap.py`）为一个进程创建一个 `AppContext`，
再经构造函数交给页面和 feature 的 `bootstrap.py`。字段：

- `settings`：`SettingsRepository`，跨启动的用户设置（分 section）；
- `preferences`：`UserPreferencesRepository`，界面与交互偏好（主题、字号、语言、最近的文件、Fitting 表格选择…）；
- `session`：`SessionRepository`，`ProjectState` 的快照（上次的会话）；
- `jobs`：`JobRunner`（`LocalProcessJobRunner`），可取消、可超时的子进程任务；
- `project_parameters`：`ProjectParametersRepository`，用户选择的参数 JSON 文件的读写；
- `instrument_profiles`：`InstrumentProfileRepository`，按探测器名 + 帧尺寸匹配的仪器配置（标定结果会写入）；
- `data_dir`：用户数据文件夹（内存中的测试 context 为 `None`）；
- `project_state`：`ProjectState`，当前项目路径、dirty、metadata 与已注册的 feature state。

feature 不创建自己的全局 context 或单例；application 和 ViewModel 只拿自己需要的 repository / port，
不把整个 context 当 service locator 传来传去。

## 用户数据文件夹

**没有 `core/` 包，也没有全局参数单例。** 设置和偏好都在一个 `UserStore`：用户数据文件夹里的 `settings.json`
（带 `schema_version` 的 JSON，原子写入），`StoreSettingsRepository` 和 `StorePreferencesRepository` 共用它。

文件夹（`user_data_dir()`）：`$GIMAP_HOME`（若设置）> Windows 上 `%APPDATA%\GIMaP` > 其他系统
`$XDG_CONFIG_HOME/gimap` 或 `~/.config/gimap`。程序文件夹因此只读、更新不会覆盖用户设置。里面有：

| 文件 | 内容 |
| --- | --- |
| `settings.json` | 所有设置与偏好（`UserStore`） |
| `session.json` | 上次的会话（`JsonSessionRepository`） |
| `instrument_profiles.json` | 仪器配置（`JsonInstrumentProfileRepository`） |
| `model_parameters.json` | Fitting 的模型参数；缺少时从程序自带的 `config/model_parameters.json` 复制 |
| `last_analyze_setup.json` | Analyze 上次的设置（掩膜、校正、切线区域…） |
| `logs/` | `ErrorGuard` 的错误日志 |

**运行 `main.py` 或任何调用 `create_app_context()` 的脚本都会读写真实的用户数据文件夹。** 做实验时把 `GIMAP_HOME`
设到临时文件夹。只用内存 repository（`InMemorySettingsRepository`、`InMemorySessionRepository`、
`InMemoryUserPreferencesRepository`，如 `tools/offscreen_smoke.py`）还不够：Fitting 的模型参数仍走 `user_data_dir()`，
缺少时会写入 `model_parameters.json`，所以单独运行 smoke 也要设 `GIMAP_HOME`。`tests/conftest.py` 为每次 pytest 运行
把 `GIMAP_HOME` 设成新的临时文件夹；`tools/check.py` 在未设置时也这样做（对它启动的每一步都有效）。
`create_standalone_legacy_context()`（不经主窗口打开的对话框）用内存中的会话和仪器配置，设置与偏好仍在用户文件夹。

## `config/`

`config/` 只有 JSON，没有 Python 模块；`src/gimap` 不 `import core` 或 `config`（测试检查）。

- `config/detectors.json`：运行时读取的探测器定义（Geometry Calibration，`features/calibration/infrastructure/adapters/local.py`）；
- `config/model_parameters.json`：Fitting 模型参数的出厂模板，用户文件夹里没有 `model_parameters.json` 时复制过去；
- `config/user_parameters.json`、`config/user_settings.json`、`config/instrument_profiles.json`（以及旧的
  `.gimap_cache/session.json`）：store 出现之前的旧文件。`create_app_context()` 每次都调用 `migrate_legacy_files`：
  用户文件夹里还没有 `settings.json` 时从前两个导入设置与偏好；`instrument_profiles.json`、`session.json`、
  `model_parameters.json` 缺少时从旧位置复制。目标存在后不再读取旧文件，也从不修改它们。

## 项目文件（`.gimap`）

File 菜单保存的项目是一个 JSON 文件（`src/gimap/app/project.py`）：Analyze 的帧和整套设置、Fitting 的 Single 与
In-situ series 状态、Compare 的设置与序列（来自 Analyze 的图存在旁边的 `<name>.compare.npz`）。路径原样保存；
打开时报告已不存在的文件。它与 `session.json`（自动保存的上次会话）是两回事。

## Fitting 的交互偏好

Fitting 的 Local Refine / Global Search 表格选择、可编辑 bounds 和全局搜索预算是 `preferences` 里的交互偏好，
不进入科学参数 JSON、项目或会话的科学记录。Local 用 `manual_auto_refine`，Global 用 `manual_global_search_v2`，
搜索预算用 `manual_parameter_search_v2`；分开的 key 避免宽范围在两种模式间串用，`_v2` 不复用旧版 Sobol 搜索保存的窄 bounds。
