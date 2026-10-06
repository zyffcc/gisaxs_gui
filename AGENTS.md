# GIMaP 项目说明（给改代码和处理数据的 agent）

GIMaP 是掠入射 X 射线散射（GISAXS、GIWAXS）的 PyQt5 桌面分析软件，使用者是做散射实验的科学家：
数值对不对、从哪里来，比功能多少更重要。

- **工作区**：
  - **Analyze**：打开帧 → 几何 → 掩膜与校正 → 切线 → 结果 → 导出；不需要 AI 的 Run Automatic Analysis；Series 热图；
    Batch Export。
  - **Fitting**：一维模型拟合，可以拟合单条曲线或原位序列。
  - **Compare**：多个序列或样品的阶段、异常帧、谁和谁相像。
- **Labs**：2D Prediction、Trainset Build。
- **Tools 菜单**：Geometry Calibration、Format Converter（含 Convert Current File）、XRR Series Extractor、
  1D Predict — Fit Many Curves（机器学习一维拟合，调用研究代码）、Process with AI、Settings。
- 入口 `main.py`；开发解释器 `D:\conda\envs\GUI\python.exe`。

## 先知道的几件事

- **工作区里有用户自己的工作。** 常常有几百个未提交的修改；未跟踪的文件里既有代码依赖的新源文件和测试，
  也有 `_handoff/` 这类个人文件。除非用户要求，不改 git 状态：
  - 不 commit、push、stash、reset，不 checkout 文件；
  - 不写 index：改名用普通 `mv`，不用 `git mv`。
- **用户数据只读，而且常常未发表**（如 `E:\ExpData`）。
  - 输出写到临时目录或用户指定的位置。
  - 从用户数据得到的数值、文件名不写进会提交的文件（代码、测试、文档）。唯一的例外是用户要求的回归案例
    `docs/agents/eval/*.json`。
  - 两个容易踩的地方：
    - Analyze 的 Export、Batch Export、Send to Fitting 和监视文件夹时的自动导出，默认都写到数据旁边的
      `gimap_analysis/`。拿用户数据驱动界面代码时，要改目标目录。
    - 运行 `main.py` 或调用 `create_app_context()` 会读写真实的用户数据目录 `%APPDATA%\GIMaP`。
      改代码、做实验时总是把 `GIMAP_HOME` 设到临时目录。只用内存仓库不够：Fitting 会直接往
      `user_data_dir()` 写 `model_parameters.json`。
- **规则以测试为准。** 文档和测试说法不同时，以 `tests/test_architecture_dependencies.py`、
  `tests/test_ui_source_of_truth.py`、`tests/test_ui_design_system.py` 为准。

## 代码结构

- **按功能组织**：每个功能在 `src/gimap/features/<feature>/` 下，含以下几部分：
  - `domain`：纯科学逻辑；
  - `application`：用例与端口；
  - `infrastructure`：适配器；
  - `presentation`：Qt 界面；
  - `bootstrap.py`。

  现有功能：analyze、fitting、compare、assistant（自动分析与 AI）、calibration、format_converter、xrr、
  prediction、trainset。
- **组合根**在 `src/gimap/app/`：`main_window.py`、`menus.py`、`runtime.py`、`bootstrap.py`、`headless_assistant.py`。
  功能之间不互相导入，需要组合时由组合根注入。例如：
  - Fitting 的快速物理拟合经 `create_quick_fit()` 交给 assistant；
  - Run Automatic Analysis 写在 assistant 里，由 `main_window` 放进 Analyze 的 Results 步骤。
- **其他共用部分**：
  - `src/gimap/app/presentation/`：应用壳与共用界面，包括 `components/`、`theme/`、`i18n/` 和 `task_runner.py`；
  - `src/gimap/shared/`：探测器读取、几何、`series_stages`（阶段与异常帧）、`figures`；
  - `src/gimap/integrations/`：用户数据（UserStore）、任务、BornAgain、TensorFlow；
  - `utils/ML_Fitting_1D_GISAXS/`：研究代码，GUI 只按路径用子进程调用它。
- **测试强制的规则。** 理由：模块要能读完、能审查，各层要能单独替换。
  - 每个 `src` 模块 ≤ 600 行，每个功能的 `view_model.py` ≤ 300 行。好几个模块已接近上限，
    加代码前先按职责拆分，不要为了凑行数切碎。
  - 导入限制：
    - presentation 不导入 domain、infrastructure；
    - `*_view.py` 不导入 application、domain、infrastructure，并且登记在 `tests/test_ui_source_of_truth.py`；
    - application 不导入 PyQt、presentation、infrastructure、TensorFlow、BornAgain；
    - domain 不导入 PyQt、TensorFlow、BornAgain；
    - `QFileDialog` / `QMessageBox` 只出现在 presentation；
    - shared 和 app/presentation 不导入任何功能；
    - `src/gimap` 不导入顶层的 utils、ui、controllers、calibration、trainset、waxs、core、config；
    - 不要在顶层的 `utils/`（`__init__.py` 除外）、`core/`、`config/`、`WAXS/` 下新建 `.py`。
  - 每个 `.qss` 在浅色和深色下都要能被 Qt 解析。Qt 只认 `:!checked` 这种写法，不认 `:not()`；
    一处写错，整张样式表都会被丢掉。
- **惯用写法**：静态布局放在 `presentation/views/*_view.py`，行为写成 mixin（Analyze 在 `bindings/`）。照着相邻代码写。
- **文档**：`docs/architecture/overview.md`、`scientific-data-flow.md`、`dependency-rules.md`；
  各工作区的调用链、文件和测试见 `docs/ui/workspaces/<名称>.md`。

## 科学数据契约

- 显示上的优化不改科学数值、单位、数组方向和数据谱系。
- **单位**：Analyze 里 q 用 Å⁻¹；Fitting 内部用 nm⁻¹，界面标明单位；送去拟合的 `<stem>_fit_input.dat` 用 Å⁻¹。
- **规范像素坐标**：像素角点在整数上，第 0 行在上，光束中心是直射束位置；像素（第 i 行、第 j 列）的中心在
  (x, y) = (j+0.5, i+0.5)。坐标换算只用 `shared/geometry/conventions.py`，否则会差半个像素。
  - 例外：Trainset 故意沿用生成训练数据时的约定（`shared/geometry/legacy_conventions.py`，q 用 nm⁻¹）。
    改它是科学上的决定（见 `docs/architecture/geometry.md`），未经用户同意不要统一。
- **存储值**：读取器保留存储值（`metadata["stored_dtype"]`）。整数计数探测器的负值是缝隙码；浮点帧（已扣暗场或背景）的
  负值是数据（`valid_pixels(..., negatives_valid=True)`）。
- **坏像素**：每帧自动排除以下几类，不会整行整列去掉；数量写进导出元数据 `bad_pixels_left_out`：
  - 孤立的热像素；
  - 死像素（只在计数探测器上判断）；
  - 缺陷探测器行 / 列上零散的亮像素。

  代码在 `analyze/domain/bad_pixels.py`；可以在 Mask 步骤或设置 `analyze.bad_pixels` 里关掉。
- **来源与记录**：每条曲线都能在图上显示来源（Sources）。曲线导出旁边写 `<stem>_analysis.json`（设置、几何、校正）；
  q 图、cake 的 CSV 和 `_fit_input.dat` 把来源写在 `#` 头行里。新增一种数据导出时，至少要带上来源和设置。
- **强度校正**：GIWAXS 强度校正（立体角、偏振、薄膜吸收）默认关闭。开启时 I / factor、q 不变，光子计数的方差按
  value / factor 传递；公式和参数写进 JSON（`analyze/domain/intensity.py`）。
- **导出的图要用浅色配色**：
  - pyqtgraph 的图用 `components/plot_export.export_plot`，它会自动切到浅色；
  - 屏幕上按主题着色的 Matplotlib 图，要在 `with theme.figures.exported_colors(figure):` 里保存；
  - 新建的出版图用 `shared/figures.save_figure`：默认浅色，中文自动换中文字体。

## 界面

目标：用户永远不用猜一个操作有没有生效；长任务可以停；常用操作一眼可见，少用的选项折叠但找得到。
不需要 AI 的流程必须完整可用：Run Automatic Analysis（GIWAXS 与 GISAXS）、Series 热图、Batch Export、Compare。

- **现成的部件**（在 `app/presentation/components/`）：
  - 步骤状态用 `StepRail.set_state`；进度、暂停、取消用 `JobStatus`；后台工作用 `TaskRunner`，不要阻塞界面线程；
  - 写完文件用 `show_toast(..., action=("Open Folder", …))` 告知；
  - 空状态用 `EmptyState` 和 `CurvePlot.set_empty_text`；少用的选项放进 `AdvancedSection`；
  - 结果表格用 `enable_table_copy`；多个页面共用一个状态栏时用 `PageStatus`。
- **颜色**：
  - 界面框架的颜色用 theme token：QSS 里写 `@token@`；代码里用 `theme.set_role(widget, role)` 设 `gimapRole`，
    用 `theme_color(name)` 取颜色；
  - 在代码里设的颜色要连接 `theme_manager().changed`，跟着主题切换；
  - 数据的颜色（曲线、阶段）可以直接写，但要在两种主题下都看得清。
- **语言**：代码里写英文，中文在 `i18n/zh_*.py` 各区域的表里，由 `zh.py` 合并；键是英文原文，`{字段}` 保持一致；
  一张表放满了就新建一张。
  - 窗口显示时，静态文字由遍历器自动翻译。
  - 运行时拼出来的文字用 `tr(text)` 或 `trf(template, **values)`：不要写 `tr(f"…")`，也不要先格式化再翻译。
  - 拼文字的页面实现 `refresh_language()`，并要列在 `src/gimap/app/main_window.py` 的 `MainWindowComponents._language_pages()` 里，切换语言后主窗口才会调用它；
    组件和对话框直接连接 `language_changed()`。
  - 数值、单位、文件名、模型名不翻译。作为数据的表头：`table.setProperty(DATA_HEADERS, True)`。
  - application / domain 的消息保持英文，因为导出记录和 AI 助手要读它们；presentation 在显示时翻译。
    带数值的消息：application 提供模板，presentation 反查模板后用 `trf`，参见 `compare/presentation/errors.py` 和
    `analyze/presentation/texts.py`。

## 测试与检查

- 改完先跑相关的 focused tests，界面测试在 offscreen 下运行（`QT_QPA_PLATFORM=offscreen`）。
  `tests/conftest.py` 已经准备好以下几样：
  - 临时的 `GIMAP_HOME` 和 QApplication；
  - 字体（含微软雅黑）和浅色主题；
  - 测试结束后删除窗口。
- 写测试时要注意：
  - 单独建的页面用完调用 `dispose()`（Analyze、Fitting 的单条与序列页、Compare）；
  - MainWindow 关闭时会自己 dispose 页面，但还有任务在跑时会先弹确认框；
  - 模态框会卡住 offscreen 测试：monkeypatch `QMessageBox` 的静态方法、`QFileDialog.get*`，以及任何会 `exec_()` 的对话框。
- 完整检查：`python tools/check.py`。
  - 主测试集（`tests/`，约 14 分钟）、offscreen smoke、ruff 是三道关，任何一道失败，退出码就是 1。
  - 研究代码的测试（`utils/ML_Fitting_1D_GISAXS/tests`）只报告、不算关，Windows 上默认跳过，因为它导入了 POSIX 专用的模块。
  - check.py 自己把 `GIMAP_HOME` 设到临时目录。`--` 后面的参数交给 pytest，例如 `python tools/check.py -- -x -k analyze`。
  - 直接运行 `python -m pytest` 只跑 `tests/`。
- ruff 只选了 E9、F63、F7、F82：语法错误、无效比较和控制流、未定义的名字。它通过不代表风格没问题，
  比如未使用的导入不会被查出来。不要跑会改动大量文件的格式化工具，那会碰到用户未提交的文件。
- **数据**：
  - 公开的真实数据在 `tests/data/external/`（GALAXI GISAXS、pyFAI AgBh、P08 与 Xeuss GIWAXS），
    几何和许可证见 `tests/data/external/README.md`；
  - `TestSAXSdata/`（P03 的 Pilatus CBF、Lambda NXS 帧和几条一维曲线）也在仓库里。
- **测试之外的截图脚本**，要做 `main()` 和 conftest 做的事：
  - 设临时的 `GIMAP_HOME`；
  - offscreen 下设 `QT_QPA_FONTDIR=C:/Windows/Fonts`，并载入一个中文字体；
  - `app.setFont(QFont("Segoe UI"))`、`apply_theme(mode, 9.0)`；
  - 然后浅色、深色、中文、英文都看一遍。

## 处理散射数据（不是改代码时）

先读 `docs/agents/giwaxs-playbook.md` 或 `gisaxs-playbook.md`。

- **起点**：`python tools/gimap_agent.py auto <图像...> --notes "<用户笔记原文>" --out <临时或用户指定的文件夹>`。
  - 耗时：给了 `.poni` 的单帧约 10 秒；要从标样图标定或读大的 NeXus 序列时，要几十秒到几分钟。
  - 输入和已保存的仪器配置相同，结果就能复现。
  - 处理数据时不要改 `GIMAP_HOME`：gimap_agent 要读那里保存的仪器配置。总是给 `--out`，不给时它会写进用户数据目录。
  - 每个样品的 `report.md` 写出它做的每个默认决定和理由（**Decisions**），以及只有用户能回答的问题（**Needs attention**）。

  它是起点，不是终点：按用户的问题需要追到哪一步就追到哪一步。手册列出了默认决定最容易不适合的地方。
- **转述数字之前先读 Decisions。** 下面几处，数字可能没错，回答的却不是用户的问题：
  - **测量技术**：先应用几何再判断。探测器最远的角超过 2θ = 20° 是 GIWAXS，否则是 GISAXS。没有几何时什么都不分析；
    判断不了时按 GIWAXS 跑并提问。用 `--technique` 可以强制。
  - **几何来源**：顺序是 `--calibration` → 笔记里点名的文件 → 已保存的仪器配置 → 在帧附近搜索。
    - 仪器配置只按探测器名和图像尺寸匹配，可能来自别的实验；
    - 搜索也可能挑到别的实验的标定。“quality: good” 只说明标定和它自己的标样一致，不说明它属于这组数据；
    - `.poni` 和 `.json` 直接采用，不再核对。

    有把握时直接给 `--calibration`；怀疑时加 `--no-saved-profiles` 或 `--recalibrate`。
  - **帧**：序列默认取最后 10 帧求和（改用 `--frame N --sum K`）。
  - **环**（GIWAXS）：只分析面积最大的 3 个没有警告的峰（改用 `--rings N`）。
  - **αi**：没有时先用 0°，并提问（补 `--incidence-deg`）。
- **退出码**：
  - 0：没有待回答的问题。
  - 2：Needs attention 里有问题，可能是：
    - 缺一个值，报告会写出补这个值的选项；
    - 需要判断的问题，比如 GISAXS 选哪个模型，这是正常结果；
    - 没找到几何，什么都没分析。
  - 1：至少一帧失败，其他帧照样有报告。1 优先于 2。
- **底线**（都有理由）：
  - **测量值只来自工具。** 界面和 JSON 记录能复现的就是这些值；自己算的派生量要写出算式和用到的工具数值。
  - **数据目录只读。** 那是用户的、常常是共享的、未发表的数据；输出写到 `--out`。
  - **不编参数。** 能量猜错，所有 q 都跟着缩放；αi 猜错，qz 跟着偏。去头文件、笔记、日志、幻灯片里找，
    找不到就问，并说清找过哪里。
  - **分清发现和假设。** 物相在用户给出材料、或者数据毫无歧义之前，都只是假设。
- **维护**：有确定答案的判断出错时，改代码，并用合成数据或公开数据加测试：
  - GIWAXS 的规则测试加在 `tests/test_assistant_rules.py`，GISAXS 的加在 `tests/test_guided_gisaxs.py`；
  - 用户要求整次运行的回归时，加一个 `docs/agents/eval/` 案例（只支持 GIWAXS，由 `tools/eval_giwaxs_agent.py` 运行）。

  需要判断力的新情况写进手册，不要写成硬性拦截。
