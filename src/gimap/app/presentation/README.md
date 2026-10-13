# App presentation

这里是 GIMaP 应用外壳和多个 feature 共用的 PyQt5 组件。组件只负责布局、显示状态和发出 UI signal，
不做科学计算、文件读取、JobRunner / 进程管理或 feature 工作流，也不导入任何 feature
（`tests/test_ui_design_system.py` 检查）。规则与 API：`docs/architecture/ui-design-principles.md`、
`docs/architecture/ui-components.md`。（2026-10-06 核对）

## 主题（`theme/`）

- `tokens.py`：浅色 / 深色两套语义色（`surface`、`text_muted`、`accent`、`warning_soft`、`plot_bg` …）与字号刻度。
  两套键完全相同；`.qss` 里写 `@token@`。现有的字面颜色只在两种主题下都是深色的底上：侧栏（`@rail@`）和
  Fitting 的 workflow header 上的白字、`gimapRole="overlay"` 的深色底牌（`base.qss` 5 处、`fitting_theme.qss` 25 处）。
  新规则不要再加字面颜色。
- `base.qss`：应用级样式表模板，`ThemeManager.apply()` 一次性设置到 `QApplication`（Fusion + QPalette）。
  控件通过语义属性选择外观：`gimapRole="muted|caption|strong|title|heading|display|success|warning|error|hint|chip|step…"`、
  `gimapRole="primary|danger|success"`（按钮）、`card`、`gimapSection`、`segment` 等。
- Feature 自己的规则放在 feature 的 `*.qss` 模板中，用 `style_widget(root, path)` 设置到页面根控件，
  切换主题时自动重新渲染。请在页面**填充子控件之前**设置（只 polish 一次）。每个 `.qss` 都必须能被 Qt 解析（测试检查两种主题）。
- `set_role()` / `set_state()` 修改语义属性并刷新样式；绘制时用 `theme_color(name)`。界面元素不要用
  `setStyleSheet("color: #…")`；只有数据本身的颜色（曲线、阶段、类别色块）才写字面值。
- `figures.py`：屏幕上的 Matplotlib 图跟随主题（`theme_figure`）；写文件时用 `with exported_colors(figure):` 保持浅色配色。
- `appearance.py`：用户选择的主题与字号（`appearance.theme`、`appearance.font_pt`），即时生效并保存。
- 缩放交给 Qt high-DPI（`main.py` 设置 `PassThrough`），没有按分辨率的 profile 或自定义缩放。

## 语言（`i18n/`）

`i18n/__init__.py`：`tr`、`trf`、`language_changed()`、`DATA_HEADERS` 和翻译静态控件的遍历器；中文表按区域分在
`zh_*.py`，由 `zh.py` 合并为 `ZH`（后合并的表覆盖同名键）。用法见设计原则的 “语言” 一节。

## Shell

- `navigation.py`：`NAVIGATION_ITEMS`（key、标题、图标、分组）与 `NavigationSidebar`（图标 + 文字，可折叠为图标栏；
  发出 `pageRequested(key)`）。Workspaces：Start、Analyze、Fitting、Compare；Labs：2D Prediction、Trainset Build。
- `home_page.py`：Start 页面（打开数据、选任务卡片、问 AI），只发出请求，由主窗口决定打开哪个工作区。
- `menu_bar.py`：`MainMenuBar`（File / View / Tools / Help）只认识 `MenuCommands` 中注入的可调用对象；缺少的命令不显示。
  `ToolWindows` 管理单实例非模态工具窗口（Geometry Calibration、Format Converter、XRR Series Extractor）。命令的组合在 `app/menus.py`。
- `settings_dialog.py` + `views/settings_dialog_view.py`：Appearance / Analyze / Data 三页，外加 feature 提供的页（如 Assistant），改动立即生效。
- `app_dialogs.py`：shell 用的文件选择与提示框（`QFileDialog` / `QMessageBox` 只允许出现在 presentation）。
- `task_runner.py`：`TaskRunner`，线程池上的后台任务，同一 key 只保留最新结果。
- `stage_text.py`：阶段与奇异帧的界面文字（Analyze ▸ Series、Fitting ▸ In-situ series、Compare 共用）；`recent_items.py`：最近文件的显示名。
- `layout_metrics.py`：一套逻辑像素尺寸 `LAYOUT` 与窗口放置函数；`layout_primitives.py`：紧凑的按钮 / 输入框高度。
- `views/main_window_view.py`：页面栈（各 feature 的 host）、菜单栏与状态栏；`app/window_view.py` 装配 Fitting / Prediction
  的经典控件；`app/main_window.py` 装配其余页面与侧栏、按 key 切换页面，并在切换语言后调用各页面的 `refresh_language()`。

## 组件（`components/`）

`CurvePlot`、`DetectorView`（及 `levels`、`marks`、`ShapeLayer`、`box_zoom`）、`StepRail`、`show_toast` / `Toast`、`JobStatus`、
`EmptyState`、`ErrorBanner`、`SegmentedControl`、`RowGroups` 与阶段颜色、`table_copy.enable_table_copy`、`plot_export.export_plot`、
`page_status.PageStatus`、`ParameterSection` / `AdvancedSection`、`FilePicker`、`ResultTable`、`PlotPanel`、`FlowLayout`、
`ScientificImageViewer`、安全滚轮输入框。`collapsible_card.py` 的 `CollapsibleCardFrame` 是 Fitting 与 Prediction 经典页面共用的
持久化折叠卡片。各自的用途与 API 见 `docs/architecture/ui-components.md`。
