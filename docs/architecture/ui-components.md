# GIMaP 公共 Presentation 组件

- **Status**: Current
- **Scope**: 跨 feature 的 PyQt 组件：用途、主要 API、何时用
- **Related code**: [`src/gimap/app/presentation/components/`](../../src/gimap/app/presentation/components/)、
  [`task_runner.py`](../../src/gimap/app/presentation/task_runner.py)
- **Related tests**: `tests/test_ui_design_system.py`、`test_components_polish.py`、`test_wave2_plots_readout.py`、
  `test_table_copy.py`、`test_levels.py`、`test_marks.py`、`test_wave2b_compare_export.py`
- **Last verified**: 2026-10-06（逐条对照代码）

视觉与反馈规则见 [`ui-design-principles.md`](ui-design-principles.md)。

## 放在哪里、能做什么

```text
src/gimap/app/presentation/components/       跨 feature 的公共组件（本文）
src/gimap/features/<feature>/presentation/   feature 自己的页面与组件
```

公共组件只管控件、布局、显示状态和 signal；不做科学计算、不读写文件、不调用 ViewModel / use case，
也不导入任何 feature（`tests/test_ui_design_system.py` 检查 `app/presentation` 不导入 `gimap.features`）。
只有一个 feature 用的组件留在该 feature；两个以上 feature 的语义稳定相同时再提到这里。

大多数组件从 `src.gimap.app.presentation.components` 导入（见其 `__init__.py` 的 `__all__`）；`table_copy`、
`plot_export`、`page_status`、`levels`、`marks` 从各自模块导入，`TaskRunner` 从 `src.gimap.app.presentation.task_runner`
（也由 `src.gimap.app.presentation` 导出）。

## 图与图像

| 组件 | 用途与主要 API |
| --- | --- |
| `CurvePlot(title="", parent=None, *, log_y=True, log_x=None, sides=True)` | 一维曲线（pyqtgraph）。`set_curves([(name, x, y), …], colors, *, markers, legend)`、`set_labels`、`set_title`（给英文，显示时 `tr`）、`clear_curves`、`has_curves`、`show_window(low, high)` / `windowChanged`、`curveClicked`、`positionClicked`、`figure_state()`。表头有 Log I、Log q（`log_x` 不为 `None` 时）、正负两半（x 有正有负时出现；`set_side`：both / positive / negative / folded；`sides=False` 关闭）、Zoom、Reset、Marks；窄时收进 “⋯”。 |
| ↳ `set_empty_text(text)` | 没有曲线时在图中央显示一句灰字（给英文，切换语言后自动重译；`""` 关闭）。每个可能为空的图都应说明为什么空、下一步做什么。 |
| ↳ 读数 | 光标在图区时，角落显示 `q = … Å⁻¹ · I = …`（量和单位取自轴标签，log 轴还原为真实值，每秒最多 30 次）。 |
| ↳ `add_save_menu()` | 表头加 “Save” 按钮：Plot as Figure… / Curves as Data…（发出 `saveFigureRequested` / `saveDataRequested`，由页面选路径并写文件）、Copy Image、Copy Data（`copy_data`：制表符分隔、带表头）。右键菜单有同样的项。 |
| ↳ `dispose()` | 在父控件销毁前删除 pyqtgraph 视图，否则可能使解释器崩溃。 |
| `DetectorView` | 探测器帧或 q 图。`set_image(data, *, valid, rect, y_down=True, title, x_label, y_label, keep_view, context)`（规范像素坐标：第 0 行在上）；自带 log、色图、色阶条（`levels.py`：拖动、输入、每帧自动或固定，按 `context` 分别记住）、Marks 菜单（`marks.py`）、光标读数（`set_readout(formatter)` 追加 q 等）。叠加：切带（`show_horizontal_band` / `show_vertical_band`）、可拖动束流中心（`show_beam_center`、`beamCenterMoved`、`set_pick_mode`）、q 框（`show_box`、`boxChanged`）、地平线、标记点、标签图。`display_state()` 交给图像导出；也要 `dispose()`。显示选择从不改变传入的数组。 |
| `ShapeLayer` | `DetectorView` 上的形状：轮廓、可编辑矩形、画新掩膜（矩形 / 多边形 / 点，Esc 取消）。 |
| `RowGroups(view)` | 在图像上按行分组（序列的阶段）：右边色条、组界虚线、组号、奇异帧红箭头；`show(edges, x_range, odd_rows=…)`。 |
| `stage_color(n)` / `STAGE_COLORS` / `stage_text_color(n)` / `text_color(c)` | 阶段颜色：图像色条和曲线用 `stage_color`（两种主题相同）；列表、表格里的名字用 `stage_text_color` / `text_color`（深色主题更亮、浅色主题保证 ≥ 4.5:1）。 |
| `plot_export.export_plot(plot, path, *, width=1600)` | 把 `CurvePlot`（或 pyqtgraph `PlotWidget` / `PlotItem`）写成 SVG，或 1600 px 宽的 PNG 等图像；**总用浅色主题的图配色**（导出时临时换色，之后还原，失败也还原），曲线颜色不变。什么也没写成时抛 `OSError`——调用方此时不能说 “Saved”。 |
| `box_zoom`：`install_box_zoom`、`zoom_button`、`MplBoxZoom` | 框选放大：所有 pyqtgraph 视图 Shift+拖动；Zoom 按钮；Matplotlib 画布拖框、双击还原。 |
| `ScientificImageViewer` | 像素图像查看器（导出在公共 API 里）；目前没有页面使用，只有 `tests/test_scientific_image_viewer.py`。新页面用 `DetectorView`。 |

## 流程与反馈

| 组件 | 用途与主要 API |
| --- | --- |
| `StepRail(steps)` | 竖排步骤栏，`steps = [(key, title), …]`。`set_state(key, state, detail="")`，state 为 `pending` / `ok` / `warn` / `error` / `busy`，detail 是一行发现（“PILATUS 2M · 1 frame”）；`set_current(key)`；点击发出 `stepChosen(key)`。状态必须来自真实结果，不伪造完成。 |
| `show_toast(parent, text, *, level="info", action=None, timeout_ms=None)` / `Toast` | 浮在页面右下角的非阻塞通知，不抢焦点。level：`info` / `ok` / `warning` / `error`；默认停留：error 直到关闭、warning 10 s、其他 5 s（`timeout_ms=0` 直到关闭）。最多同时 3 条，相同文字只显示一次。`action=(title, callback)` 加一个按钮，写文件后用 `("Open Folder", 打开所在文件夹)`。文字给英文，在这里 `tr`。 |
| `JobStatus` | 长任务状态：`set_state(state, message, *, progress)`，state 为 idle / queued / running / paused / succeeded / failed / cancelled / timed_out，`progress=None` 为不定进度；Pause / Cancel / Details 按钮发出 `pauseRequested(bool)` / `cancelRequested` / `detailsRequested`；`set_actions_visible(...)`。 |
| `EmptyState` / `ErrorBanner` | 无输入或无结果时的引导（`set_content(title, message, action_text)`、`actionRequested`）；页面内横幅（`set_level`：info / warning / error / success，其他值当作 error；`set_message(title, message)`、`dismissed`、`detailsRequested`）。 |
| `page_status.PageStatus(page, status_signal)` | 共用一条状态栏的页面（Labs：2D Prediction、Trainset Build）各自记住最后一条消息，页面再次显示时恢复，避免显示另一个页面的消息。 |
| `TaskRunner.submit(key, fn, *, on_done, on_error)` | 在线程池上运行 `fn()`，在 GUI 线程回调；同一 key 的新提交使旧结果作废（拖动切带、翻帧只显示最新结果）。**不会中断正在运行的工作**：要能停止，把取消标志传进 `fn`。另有 `cancel(key)`（只是忽略它的结果）、`is_busy()`、`busy_changed`、`wait`、`shutdown`。 |

## 输入、表格与布局

| 组件 | 用途与主要 API |
| --- | --- |
| `SegmentedControl` | 几个互斥选项的一排按钮；API 与 `QComboBox` 常用部分相同（`addItem(text, data)`、`findData`、`currentData`、`setCurrentIndex`、`activated`、`currentIndexChanged`、`setItemToolTip`），可直接替换小下拉框。 |
| `table_copy.enable_table_copy(table)` | 任意 `QTableView` / `QTableWidget`：Ctrl+C 复制选中行（无选择时整表），右键 Copy Rows / Copy Table（表格已有自己的菜单时只加 Ctrl+C）。制表符分隔、带表头；表头的 `Qt.UserRole` 可给出带单位的完整列名。 |
| `ParameterSection` / `AdvancedSection` | 常规区块（标题、说明、header action）；低频选项的折叠区（`expandedChanged`、`set_expanded`）。核心流程不能依赖展开 Advanced。 |
| `FilePicker`、`ResultTable`、`PlotPanel`、`FlowLayout` | 路径输入（`browseRequested`、`clearRequested`）；只读结果表与空状态；带工具栏和空状态的绘图容器；放不下时换行的按钮行。 |
| `SafeWheelSpinBox` / `SafeWheelDoubleSpinBox` / `SafeWheelComboBox`、`install_safe_wheel_behavior(root)` | 普通滚轮滚动页面；只有获得焦点并按住 Alt/Option 时滚轮才改值。静态页面构造完调用一次，运行时新建的子树再调用一次（可重复调用）。 |
| `CollapsibleCardFrame`（`app/presentation/collapsible_card.py`） | Fitting 与 Prediction 经典页面共用的持久化折叠卡片。 |

## 新增公共组件

1. 至少两个 feature 已稳定需要相同语义，或属于全应用一致的安全 / 可访问性规则；
2. 在 `components/` 建一个说明职责的模块（不叫 `utils.py`、`common.py`），只发 intent signal、只收显示状态；
3. 从 `components/__init__.py` 导出（常用的也从 `src/gimap/app/presentation/__init__.py` 导出）；
4. 加 offscreen 构造与 signal / state 测试，并让一个调用方采用；
5. 用户可见的英文进 `i18n/zh*.py`，颜色用主题 token（见设计原则）。

只是外观相似、语义不同的控件留在各自 feature，不要用一堆 flag 做成万能组件。
