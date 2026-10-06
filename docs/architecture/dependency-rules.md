# 依赖规则

> **Status**：Current
>
> **Scope**：`src/` 生产代码的依赖方向、模块大小和由测试强制的边界
>
> **Related tests**：`tests/test_architecture_dependencies.py`、`tests/test_ui_source_of_truth.py`、
> `tests/test_ui_design_system.py`、`tools/offscreen_smoke.py`
>
> **Last verified**：2026-10-06（逐条对照上述测试）

文档与测试不一致时以测试为准。下面第一节是测试会挡住的硬规则；其余是设计意图，靠 review 维护。

## 测试强制的规则

| 规则 | 测试 |
| --- | --- |
| `src/` 下每个 `.py` 不超过 **600 行**（`utils/ML_Fitting_1D_GISAXS` 研究代码不算）。为了模块可读、可 review；按职责拆分，不为凑行数切碎内聚的代码。 | `test_runtime_python_modules_do_not_regrow_into_monoliths` |
| 每个 feature 的 `view_model.py` 不超过 **300 行**。 | `test_feature_view_models_stay_within_review_threshold` |
| **domain** 不导入 PyQt5/6、PySide2/6、TensorFlow、Keras、BornAgain。 | `test_domain_files_do_not_import_forbidden_runtimes` |
| **application** 不导入路径中含 `presentation`、`infrastructure` 的模块，也不导入 PyQt/PySide、TensorFlow、Keras、BornAgain。 | `test_application_cannot_depend_on_presentation`、`test_application_files_do_not_import_gui_runtimes_or_concrete_adapters` |
| **presentation**（任何路径含 `presentation` 的文件，包括 `app/presentation`）不导入路径中含 `domain`、`infrastructure` 或 `adapters` 的模块，也不导入顶层 `utils`。可以导入本 feature 的 application、`src.gimap.shared`、`src.gimap.app`（AppContext、ports、jobs）和 `src.gimap.app.presentation`。 | `test_presentation_does_not_import_domain_directly`、`test_presentation_files_do_not_import_concrete_adapters`、`test_presentation_files_do_not_import_legacy_utils_package` |
| **静态视图** `presentation/views/*_view.py` 更严：不导入 application、domain、infrastructure、controllers、TensorFlow、Keras、BornAgain；文件集合必须与 `EXPECTED_VIEWS_BY_OWNER` 完全一致（新增、删除、改名都要改表）。不再有 Qt Designer（`.ui`、`_generated`、`tools/compile_ui.py`）。 | `tests/test_ui_source_of_truth.py` |
| **feature 之间不互相导入**：`src/gimap/features/X/` 里不能出现 `src.gimap.features.Y`（Y ≠ X）。 | `test_features_do_not_import_other_feature_internals` |
| `src/gimap/shared` 和 `src/gimap/app/presentation` 都不导入 `src.gimap.features`。 | `test_shared_does_not_import_feature_implementations`、`test_ui_design_system.py::test_shared_components_construct_without_feature_or_scientific_dependencies` |
| `QFileDialog`、`QMessageBox` 只在路径含 `presentation` 的模块里导入（`app/menus.py` 等经 `app/presentation/app_dialogs.py`）。 | `test_qt_file_and_message_dialogs_are_confined_to_presentation` |
| 删除的顶层包不回来：`calibration/`、`controllers/`、`trainset/`、`ui/`、`WAXS/` 下没有 `.py`；`utils/` 顶层除 `__init__.py` 外没有 `.py`；`core/`、`config/` 没有 `.py`；没有 `legacy_bridge.py` / `legacy_controller.py`；`src/gimap` 不 `import` 顶层 `calibration`、`controllers`、`trainset`、`ui`、`utils`、`waxs`、`core`、`config`。 | `test_deleted_compatibility_aliases_do_not_regrow`、`test_new_source_does_not_depend_on_legacy_compatibility_packages`、`test_settings_live_in_the_user_store_only`、`test_production_source_does_not_use_internal_legacy_bridges` |
| 每个 `.qss` 在浅色和深色主题下都能被 Qt 解析。 | `test_every_style_sheet_parses_in_qt_in_both_themes` |
| 真实主窗口能 offscreen 启动：6 个页面、fitting / prediction / trainset 三个 binding、一个 Compare 页面。 | `tools/offscreen_smoke.py`（`tools/check.py` 的一步） |

检查方式是扫描 `import` 语句的模块路径段，所以 `from ..domain import x` 和 `import src.gimap.features.y` 都会被发现。

## 方向

```text
presentation → application → domain
infrastructure → 实现 application 的 ports（并使用 domain）
组合根（src/gimap/app/、main.py、各 feature 的 bootstrap.py）→ 创建 adapters 并注入
```

- **domain**：纯科学含义。可用标准库、NumPy，以及语义稳定的 SciPy；不接收也不返回 Qt 对象、张量、BornAgain 对象或文件句柄。
- **application**：用例与 ports；输入输出与框架无关；不碰控件、对话框和具体 adapter。每个新用例都有测试（尽量用 fake port）。
- **infrastructure**：文件、BornAgain、TensorFlow/Keras 等具体实现，放在 `infrastructure/adapters/`；在边界处转换外部类型；不显示对话框。
- **presentation**：页面、ViewModel、bindings。ViewModel 只管界面状态、命令、调用用例和把结果变成显示状态；
  ViewModel 和 bindings 不做科学计算、不写具体文件系统实现、不编排第二层工作流。用户选的路径作为请求的一部分交给 application。

## 跨 feature 协作

feature 不导入另一个 feature。需要组合时在组合根注入：`src/gimap/app/main_window.py`（页面、Run Automatic Analysis 放进 Analyze、
Process with AI）、`menus.py`（Tools 窗口）、`window_view.py`（经典 Fitting / Prediction 控件）、`runtime.py`（Labs 与 Fitting 的 binding）、
`headless_assistant.py`（`tools/gimap_agent.py` 用）。例：Fitting 的快速物理拟合由 `create_quick_fit()` 创建，注入 Fitting workspace、
自动分析与 Process with AI（assistant）和 Analyze 的批量拟合（`page.set_model_fitter`）；Calibration 的
`create_headless_calibration()` 同样注入自动分析与 Process with AI。

多个 feature 需要同一项稳定科学能力时，提取到 `src/gimap/shared/`（现有 `detector_io`、`geometry`、`series_stages`、
`figures.py`、`file_paths.py`），而不是让一个 feature 调另一个的用例。`shared/` 不是 `utils/`：至少两个稳定使用方、
语义和 ownership 明确才提取；模块名说明职责，不用 `utils.py`、`helpers.py`、`common.py`、`misc.py`。

## 科学行为

架构、UI、性能和维护性修改不得静默改变科学结果：数值定义、参数含义、单位、数组方向、约束、排序、拟合和预处理行为。
科学行为修改是单独的任务，有专门的测试，不藏在移动、改名或抽取里。修改已有行为前先用 focused tests 固定它。

## Review 时问

- 代码属于哪个 feature？依赖方向对吗？对话框只在 presentation 吗？
- BornAgain、TensorFlow 和文件系统细节在 ports / adapters 后面吗？
- 跨 feature 的组合在组合根吗？shared 抽取有两个稳定使用方吗？
- 模块接近 600 行（或 `view_model.py` 接近 300 行）时，是否按职责拆分了，而不是硬切？
- 科学输出没变吗？每个新用例有测试吗？
