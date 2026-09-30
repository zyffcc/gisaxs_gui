# App presentation

这里保存由 GIMaP application shell 所有、可被多个 feature presentation 使用的通用 PyQt5
组件。组件只负责布局、展示状态和发出 UI signals，不包含科学计算、文件读取、JobRunner /
process 管理或 feature workflow，也不导入任何 feature module（`tests/test_ui_design_system.py` 检查）。

## 主题（`theme/`）

- `tokens.py`：浅色 / 深色两套语义色（`surface`、`text_muted`、`accent`、`warning_soft` …）与字号刻度。
  两套键完全相同；样式表里只写 `@token@`，从不写字面颜色。
- `base.qss`：应用级样式表模板，`ThemeManager.apply()` 一次性设置到 `QApplication`（Fusion + QPalette）。
  控件通过语义属性选择外观：`gimapRole="muted|caption|strong|title|heading|display|success|warning|error|hint|chip|step…"`、
  `gimapRole="primary|danger|success"`（按钮）、`card`、`gimapSection`、`segment` 等。
- Feature 自己的规则放在 feature 的 `*.qss` 模板中，用 `style_widget(root, path)` 设置到页面根控件，
  切换主题时自动重新渲染。请在页面**填充子控件之前**设置（只 polish 一次）。
- `set_role()` / `set_state()` 修改语义属性并刷新样式；不要再用 `setStyleSheet("color: #…")`
  （只有数据本身的颜色，例如类别色块，才可以内联）。
- `appearance.py`：用户选择的主题与字号（`appearance.theme`、`appearance.font_pt`），即时生效并保存。
- 缩放交给 Qt high-DPI（`main.py` 设置 `PassThrough`），不再有按分辨率的 profile 或自定义缩放。

## Shell

- `navigation.py`：`NAVIGATION_ITEMS`（key、标题、图标、分组）与 `NavigationSidebar`
  （图标 + 文字，可折叠为图标栏；发出 `pageRequested(key)`）。
- `menu_bar.py`：`MainMenuBar`（File / View / Tools / Help）只认识 `MenuCommands` 中注入的可调用对象；
  缺少的命令不显示。`ToolWindows` 管理单实例非模态工具窗口。命令的组合在 `app/menus.py`。
- `settings_dialog.py` + `views/settings_dialog_view.py`：Appearance / Analyze / Data 三页，改动立即生效。
- `app_dialogs.py`：shell 使用的文件选择与提示框（QFileDialog / QMessageBox 只允许出现在 presentation）。
- `layout_metrics.py`：一套逻辑像素尺寸 `LAYOUT` 与窗口放置函数；`layout_primitives.py`：紧凑的按钮 /
  输入框高度。
- `views/main_window_view.py`：页面栈（各 feature 的 host）、菜单栏与状态栏；`app/window_view.py`
  装配 Fitting / Prediction 的经典控件；`app/main_window.py` 装配其余页面与侧栏并按 key 切换页面。

## 组件（`components/`）

`DetectorView`（探测器图、色标、切带、可拖动束流中心、q 框、点选模式）、`CurvePlot`（跟随主题的曲线图）、
`SegmentedControl`（与 QComboBox 子集兼容的分段按钮）、`ParameterSection` / `AdvancedSection`、
`JobStatus` / `ErrorBanner` / `EmptyState`、`ResultTable`、`FilePicker`、`ScientificImageViewer`、
安全滚轮输入框。`CollapsibleCardFrame` 是 Fitting 与 Prediction 共用的持久化折叠卡片。
