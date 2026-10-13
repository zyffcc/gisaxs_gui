# GIMaP 桌面 UI 设计原则

- **Status**: Current
- **Scope**: 所有 workspace、page、dialog、工具窗口和公共组件的反馈、布局、颜色与语言
- **Related code**: [`src/gimap/app/presentation/`](../../src/gimap/app/presentation/)（`components/`、`theme/`、`i18n/`）
- **Related tests**: `tests/test_ui_design_system.py`（含 QSS 在两种主题下都能被 Qt 解析）、
  `tests/test_ui_workspace_layouts.py`、`tests/test_i18n.py`、`tests/test_wave2b_analyze_i18n.py`、
  `tests/test_review_labs_status.py`
- **Last verified**: 2026-10-06（逐条对照代码）

组件 API 见 [`ui-components.md`](ui-components.md)；各 workspace 的控件与调用链见 `docs/ui/workspaces/`。

## 意图

1. **用户从不需要猜某件事有没有发生。** 每个操作都有可见结果：正在做、做完了（发现了什么）、
   失败了（为什么、怎么办）、或为什么什么也没有。
2. **长时间的工作看得见、停得下。** 有进度，能取消（能暂停更好），不冻结界面，取消后界面回到可用状态。
3. 用户随时能回答：我在哪个任务？现在要做什么？结果和下一步在哪里？

## 反馈：用哪个组件

| 情况 | 做法 |
| --- | --- |
| 读取、计算中 | `StepRail.set_state(key, "busy")`，结束后 `ok` / `warn` / `error` 并给一行 detail（发现了什么）；状态来自真实结果 |
| 长任务 | 在后台运行（`TaskRunner` 或 `JobRunner`），`JobStatus` 显示进度、Pause / Cancel；`TaskRunner` 不会中断工作，取消要把标志传进函数并在循环里检查 |
| 写了文件 | 文件确实写成后 `show_toast(page, "Saved …", level="ok", action=("Open Folder", …))`；`export_plot` 抛 `OSError` 时报错，不说 “Saved” |
| 图或表是空的 | `CurvePlot.set_empty_text(...)`、`EmptyState`、`ResultTable` 的空状态：说明为什么空、下一步做什么 |
| 出错 | `ErrorBanner`（页内）或 `level="error"` 的 toast（直到关闭）；说出原因和可做的事 |
| 不常用的选项 | `AdvancedSection` 折叠，但可发现；核心流程不需要展开它 |

## 单层画布，默认可用

- 一个视觉区域最多一层带背景或边框的容器；层级靠留白、字号、字重和分隔线，不靠 `Section → Card → GroupBox`
  的嵌套边框。`QGroupBox` 只在边界本身有意义时用。
- 不同时显示含义重复的 workspace、section、card、group 标题。
- 输入、主要参数、预览、主命令和当前结果默认可见。每个任务只有一个视觉主操作；核心命令不只存在于右键、
  双击、手势或 tooltip。
- 界面上的说明帮助用户做决定；不显示开发者备注或 “给 agent 的话”。

## 响应式布局

- 用 layout、`QSizePolicy`、stretch 和 `sizeHint` 表达意图，不用手工坐标；不把 `minimumHeight` 与
  `maximumHeight` 锁成同一个动态值，不用固定高度掩盖布局问题（图标、短按钮、工具栏除外）。
- 一个方向只有一个主要滚动容器；窄窗口退化为单列或外层滚动，不隐藏功能（`FlowLayout` 让按钮行换行）。
- `QStackedWidget` / tab 的尺寸跟随当前页；导航（tab、步骤栏、侧栏）位置稳定，不因条件控件的显示而移动。
- 至少在 1280×800、1440×900、1920×1080 下检查主操作和当前结果可达。

## 科学图像与曲线

- 显示控制（log、色图、色阶、叠加层）紧挨图像且默认可发现；纯显示操作不触发切线、拟合、页面跳转，
  也不改变科学数组；改变计算输入后只把下游结果标为过期或重算。
- `Pick center`、选区等直接操作有明确按钮、选中态和 Esc 取消。
- 有正负两半的曲线（GISAXS 的 qy、GIWAXS 的 χ）：`CurvePlot` 提供 ±、+、−、|x|（两半叠在 |x| 上，负半虚线）；
  log 轴隐藏非正值。经典 Fitting 页面（Matplotlib）的 Signed ±q 与 Negative −q 在 Log X 时用 symlog。
  预览、拟合区间、拟合输入和导出对同一 q 模式的解释一致。

## 颜色与主题

- **界面颜色来自主题 token**（`theme/tokens.py`：`LIGHT` 与 `DARK` 键完全相同）：`.qss` 模板里写 `@token@`；
  代码里用语义属性 `set_role(widget, "muted")`（`gimapRole`）和 `set_state(widget, name, value)`；绘制时用
  `theme_color(name)`。feature 的 `.qss` 用 `style_widget(root, path)` 在填充子控件之前设置。
  不要为界面元素写 `setStyleSheet("color: #…")`。`.qss` 里现有的字面颜色只用在两种主题下都是深色的底上（侧栏和
  Fitting workflow header 上的白字、overlay 的深色底牌）；新规则不要再加。
- **数据颜色可以写字面值**：曲线（`CurvePlot` 的 `CURVE_COLORS`）、阶段（`STAGE_COLORS`）、拟合曲线等，
  两种主题相同；它们作为列表或表格里的文字时用 `text_color` / `stage_text_color` 保证可读。
- 浅色、深色都要看。`.qss` 必须能被 Qt 解析（CSS 才有的写法如 `:not()` 会让 Qt 丢掉整张样式表；测试检查）。
- **导出的图总用浅色配色**：pyqtgraph 图经 `export_plot`；屏幕上跟随主题的 Matplotlib 图（`theme_figure`）
  在 `with exported_colors(figure):` 里写文件。

## 语言（i18n）

- 代码里写英文；中文在 `src/gimap/app/presentation/i18n/` 的分区表（`zh_analyze.py`、`zh_shell.py` …），
  由 `zh.py` 合并成 `ZH`，按英文原文精确匹配：改了英文就要改表。数值、单位、文件名、符号不翻译。
- 静态文字（标签、按钮、tab、下拉项、表头、spin box 前后缀、占位符、tooltip、菜单）由遍历器在窗口或菜单
  显示时翻译，不需要手动处理。
- **运行时拼出的文字**用 `tr(text)` 或 `trf(template, **values)`：模板（含 `{字段}`）是表里的键，中文保留相同字段；
  不写 `tr(f"…")`，也不先格式化再翻译（`test_wave2b_analyze_i18n.py` 扫描 Analyze presentation；
  `test_review_labs_status.py`、`test_review_stages_cross.py` 检查 Labs 与 Compare 的文字都在表里）。
- **切换语言后**，`language_changed()` 发出，`MainWindowComponents.refresh_language`（`src/gimap/app/main_window.py`）
  对 `_language_pages()` 中每个页面（Start、Analyze、自动分析、Compare、Fitting 及其页面、Labs 的 view binding、
  assistant、打开的 Tools 窗口）调用
  `refresh_language()`。新页面或工具窗口若拼出文字，要实现 `refresh_language()` 并保留英文原始状态以便重拼。
- 表头是数据名（序列、样品、参数名）的表格或树设 `setProperty(DATA_HEADERS, True)`（`"gimapDataHeaders"`），
  它们不被翻译。
- `CurvePlot.set_title` / `set_empty_text`、`show_toast`、`JobStatus.set_state`、`enable_table_copy` 的菜单
  自己 `tr` 传入的英文。
- application 与 domain 的消息、曲线标题保持英文（导出记录和 assistant 读取它们）；presentation 在显示时翻译
  （Analyze 在 `presentation/texts.py`）。中文字体没有 “▸”“▾”，中文里显示为 “›”“▼”。

## 每次 UI 修改

1. 先写下任务流和容器树，找出重复标题、嵌套边框、被折叠的核心功能和固定尺寸。
2. 优先复用公共组件，但不为复用多加一层容器；布局修改不夹带科学算法修改。
3. 检查键盘焦点、safe-wheel、默认 / 空 / 错误状态、长文本，两种主题、两种语言。
4. 新页面要有 offscreen 构造测试；新增、删除或改名 `*_view.py` 要更新
   `tests/test_ui_source_of_truth.py` 的 `EXPECTED_VIEWS_BY_OWNER` 和 `docs/ui/workspaces/` 的说明。

以下任一项出现时，UI 修改不算完成：核心流程需要展开 Advanced；两层以上连续边框；内容被固定高度裁切；
命令只能靠手势找到；1440×900 下主操作或当前结果不可达；调整显示参数触发计算、跳页或改变数据；
操作后用户看不出发生了什么；长任务停不下来。
