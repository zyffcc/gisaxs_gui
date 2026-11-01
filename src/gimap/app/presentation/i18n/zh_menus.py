"""Chinese texts of the main menu bar for the release: Edit, Analyze and Fitting, the View sections, the
mnemonics fixed in Tools and File, why a menu entry cannot run now (``app/workspace_commands.py``), and the
Help menu with its windows (``presentation/help_documents.py``, ``help_dialogs.py``, ``app/help_commands.py``).

Keyed by the exact English text of the code (``presentation/menu_bar.py``, ``presentation/workspace_menus.py``).
A mnemonic is written ``(&X)`` after the Chinese text, before an ellipsis. Tooltips the pages already have are
in their own tables and are not repeated here.
"""

MENUS_ZH = {
    # -- the menu bar ------------------------------------------------------------------------------
    "&Edit": "编辑(&E)",
    "&Analyze": "分析(&A)",
    "F&itting": "拟合(&I)",
    "Workspaces": "工作区",
    "Labs": "实验室",
    # -- File, Tools ------------------------------------------------------------------------------
    "&Open Curve…": "打开曲线(&O)…",  # File ▸ Open Data on Fitting's Single analysis (``Open Curve…``: its buttons)
    "Appearance, analysis defaults, the AI assistant and the user data folder":
        "外观、分析默认值、AI 助手和用户数据文件夹",
    # -- Edit -------------------------------------------------------------------------------------
    "&Undo: {what}": "撤销(&U)：{what}",
    "&Redo: {what}": "重做(&R)：{what}",
    "Undo the last change on the page shown: the set-up in Analyze, the model in Fitting's Single analysis":
        "撤销当前页面上的上一次修改：分析页的设置，或拟合单条分析的模型",
    "Redo what was undone last (Ctrl+Y or Ctrl+Shift+Z)": "重做上一次撤销的修改（Ctrl+Y 或 Ctrl+Shift+Z）",
    # -- Analyze ----------------------------------------------------------------------------------
    "Run &Automatic Analysis": "运行自动分析(&A)",
    "S&top Automatic Analysis": "停止自动分析(&T)",
    "&Repeat Analysis": "重新分析(&R)",
    "Analyse the frame shown again with the set-up as it is": "按现在的设置重新分析显示的帧",
    "&Export Curves…": "导出曲线(&E)…",
    "&Batch Export…": "批量导出(&B)…",
    "Batch Export a F&older…": "批量导出一个文件夹(&O)…",
    "A folder of raw frames straight to curves: Batch Export opens once its first frame is analysed":
        "把一个文件夹的原始帧直接变成曲线：第一帧分析完后打开批量导出",
    "Send to &Fitting": "发送到拟合(&F)",
    "Send Series to Fitti&ng…": "把序列发送到拟合(&N)…",
    "Send Series to &Compare": "把序列发送到比较(&C)",
    "&Set-up Files": "设置文件(&S)",
    "Settings, cut regions and masks as files, for the next data set": "把设置、切线区域和掩膜存为文件，供下一组数据使用",
    "Save &Settings…": "保存设置(&S)…",
    "Load S&ettings…": "载入设置(&E)…",
    "Save &Cuts…": "保存切线(&C)…",
    "Load C&uts…": "载入切线(&U)…",
    "The GIWAXS cut regions as a .json file": "把 GIWAXS 切线区域存为 .json 文件",
    "Cut regions saved before (.json)": "以前保存的切线区域（.json）",
    "Save &Masks…": "保存掩膜(&M)…",
    "Load M&asks…": "载入掩膜(&A)…",
    "Fi&gures": "图与数据(&G)",
    "The view as a figure, or its data as a table": "把视图存为图片，或把它的数据存为表格",
    "Save &Image…": "保存图像(&I)…",
    "Save &q-Map Data…": "保存 q 图数据(&Q)…",
    "Save &View Data…": "保存当前视图数据(&V)…",
    "C&lear Frame List": "清空帧列表(&L)",
    # -- Fitting ----------------------------------------------------------------------------------
    "Fit a Fol&der of Curves…": "拟合一个文件夹的曲线(&D)…",
    "&Fit": "拟合(&F)",
    "S&top Fit": "停止拟合(&T)",
    "&Load Model…": "载入模型(&L)…",
    "Save &Model…": "保存模型(&M)…",
    "Save Data &and Fit…": "保存数据与拟合(&A)…",
    "Save &Plot…": "保存图(&P)…",
    "&In-situ Series": "原位序列(&I)",
    "Every curve of a folder fitted with the model of Single analysis": "用单条分析的模型拟合一个文件夹里的每条曲线",
    "&Start": "开始(&S)",
    "S&top": "停止(&T)",
    "1D Predict — Fit Many &Curves…": "一维预测——拟合多条曲线(&C)…",
    # File ▸ Labs Parameters (presentation/menu_bar.py)
    "The settings of 2D Prediction and Trainset Build, and Fitting's model": "二维预测和训练集构建的设置，以及拟合的模型",
    "Load the settings of the Labs pages (2D Prediction, Trainset Build) and Fitting's model from a JSON file; "
    "Fitting's model is replaced": "从 JSON 文件载入实验室页面（二维预测、训练集构建）的设置和拟合的模型；拟合的模型会被替换",
    "Save the settings of the Labs pages and Fitting's model to a JSON file": "把实验室页面的设置和拟合的模型保存为 JSON 文件",
    # -- why an entry cannot run now (app/workspace_commands.py) ----------------------------------
    "Set the geometry first (Geometry step).": "请先设置几何（几何步骤）。",
    "The automatic analysis is running.": "自动分析正在运行。",
    "The automatic analysis is not running.": "自动分析没有在运行。",
    "The automatic analysis works on the frame shown: stop it first.": "自动分析正在处理显示的帧：请先停止它。",
    "List two or more frames first (a folder, several files or a NeXus series).":
        "请先列出两帧或更多（一个文件夹、几个文件或一个 NeXus 序列）。",
    "Build the Series map first (Series tab ▸ Build Map).": "请先生成序列热图（序列标签页 ▸ 生成热图）。",
    "Draw a mask first.": "请先画一个掩膜。",
    "A fit is running.": "正在拟合。",
    "No fit is running.": "没有在运行的拟合。",
    "The In-situ series is not running.": "原位序列没有在运行。",
    "Undo and Redo work on Analyze and on Fitting's Single analysis.": "撤销和重做作用于分析页和拟合的单条分析。",
    "Nothing to undo on this page.": "这个页面没有可撤销的修改。",
    "Nothing to redo on this page.": "这个页面没有可重做的修改。",
    "The frame shown is not analysed: it could not be read, or its analysis is still running.":
        "显示的帧还没有分析结果：它无法读取，或者分析还在进行。",
    "This entry could not check whether it can run; the details are in the error log.":
        "这一项无法检查自己现在能否运行；详细信息写在错误日志里。",
    # -- Help (presentation/menu_bar.py, help_documents.py, help_dialogs.py; app/help_commands.py) ---------
    "&Physics and Methods": "物理与方法(&P)",
    "&Keyboard Shortcuts…": "键盘快捷键(&K)…",
    "E&xample Data": "示例数据(&X)",
    "&Report a Problem…": "报告问题(&R)…",
    "Open Error &Log Folder": "打开错误日志文件夹(&L)",
    "Check for Upda&tes…": "检查更新(&T)…",
    "How to &Cite GIMaP…": "如何引用 GIMaP(&C)…",
    "Third-party &Notices": "第三方声明(&N)",
    "The manual of GIMaP, opened in your web browser": "GIMaP 的用户手册，在网页浏览器中打开",
    "How GIMaP computes what it shows: geometry, q, corrections, cuts and models (in your web browser)":
        "GIMaP 怎样算出它显示的数值：几何、q、校正、切线和模型（在网页浏览器中打开）",
    "Every key of the menus and of the pages, in a table you can copy": "菜单和各页面的全部快捷键，列在一张可以复制的表里",
    "Public example frames that come with GIMaP: one is copied to a folder of your choice and opened in Analyze":
        "GIMaP 自带的公开示例帧：选一个复制到你指定的文件夹，并在分析页中打开",
    "Copies the versions of GIMaP and of your system (no paths, no data) and opens a new issue on GitHub in your browser":
        "复制 GIMaP 和系统的版本信息（不含路径和数据），并在浏览器中打开 GitHub 上的新问题页面",
    "The folder of errors.log, where every unexpected error is written": "errors.log 所在的文件夹，每个意外错误都写在这里",
    "The reference to give when you publish results obtained with GIMaP": "发表用 GIMaP 得到的结果时应当引用的文献",
    "What GIMaP and its Windows package contain that is not GIMaP's own work, and the licences":
        "GIMaP 及其 Windows 便携包中不属于 GIMaP 自己的内容，以及它们的许可证",
    "You are running {name} {version}. This opens the releases page on GitHub in your browser (newest first, "
    "pre-releases included); GIMaP itself makes no network request.":
        "你正在运行 {name} {version}。这一项在浏览器中打开 GitHub 上的发布页（最新的在前，含预发布版）；GIMaP 本身不发出任何网络请求。",
    # the documents
    "Physics and Methods": "物理与方法",
    "The physics and methods manual was not found:\n{path}": "找不到物理与方法手册：\n{path}",
    "Third-party Notices": "第三方声明",
    "The third-party notices were not found:\n{path}": "找不到第三方声明：\n{path}",
    # Keyboard Shortcuts
    "Keyboard Shortcuts": "键盘快捷键",
    "The keys of the menus work on every page. The keys of a page or a window act while it has the keyboard focus: "
    "after switching pages, click into the page first. Ctrl+C copies the selected rows.":
        "菜单的快捷键在每个页面上都起作用。页面或窗口的快捷键只在它有键盘焦点时起作用：切换页面之后，先在页面里点一下。"
        "Ctrl+C 复制选中的行。",
    "Where": "在哪里",
    "Keys": "按键",
    "What": "作用",
    "{menu} menu": "{menu}菜单",
    "Tables": "表格",
    "Analyze, drawing a polygon mask": "分析，画多边形掩膜时",
    "Trainset Build (the image)": "训练集构建（图像）",
    "Previous imported frame": "上一个已导入的帧",
    "Next imported frame (through the frames of a source, then on to the next source)":
        "下一个已导入的帧（先走完一个数据源的帧，再到下一个数据源）",
    "Undo a change of the set-up (menu: Edit ▸ Undo)": "撤销对设置的一次修改（菜单：编辑 ▸ 撤销）",
    "Redo it (menu: Edit ▸ Redo)": "重做（菜单：编辑 ▸ 重做）",
    "Export the curves of this frame (menu: Analyze ▸ Export Curves…)": "导出这一帧的曲线（菜单：分析 ▸ 导出曲线…）",
    "Batch Export… (menu: Analyze ▸ Batch Export…)": "批量导出…（菜单：分析 ▸ 批量导出…）",
    "Analyse the frame again (menu: Analyze ▸ Repeat Analysis)": "重新分析这一帧（菜单：分析 ▸ 重新分析）",
    "Remove the selected source (all its frames), mask or cut region from its list (the files themselves are not "
    "touched)": "从列表中移除选中的数据源（它的全部帧）、掩膜或切线区域（文件本身不受影响）",
    "Stop picking the beam centre on the image; cancel drawing a mask": "停止在图上点选光束中心；取消正在画的掩膜",
    "Close the polygon (from three points on)": "闭合多边形（至少三个点）",
    "Take back the last point": "去掉最后一个点",
    "Fit (menu: Fitting ▸ Fit)": "拟合（菜单：拟合 ▸ 拟合）",
    "Undo a change of the model (menu: Edit ▸ Undo)": "撤销对模型的一次修改（菜单：编辑 ▸ 撤销）",
    "Remove the selected series from the table": "从表格中移除选中的序列",
    "Copy the selected rows with their column names": "复制选中的行，带列名",
    "Close the window (it asks first while a calibration runs)": "关闭窗口（标定正在运行时会先询问）",
    "Stop picking the direct-beam centre on the preview; when not picking, close the window":
        "停止在预览图上点选直射束中心；没有在点选时关闭窗口",
    "Leave the drawing mode: picking the direct beam, the ROI or a mask shape": "退出绘制模式：点选直射束、画 ROI 或掩膜形状",
    # How to Cite, About
    "How to Cite GIMaP": "如何引用 GIMaP",
    "If you publish results obtained with GIMaP, please cite it as below and name the version you used (the export "
    "records of GIMaP write it as software_version).":
        "如果你发表用 GIMaP 得到的结果，请按下面的格式引用它，并写明你用的版本（GIMaP 的导出记录里以 software_version 写明）。",
    "GIMaP has no DOI yet: cite the repository and the version.": "GIMaP 还没有 DOI：请引用仓库地址和版本。",
    "Copy BibTeX": "复制 BibTeX",
    "The reference is on the clipboard.": "引用文字已复制到剪贴板。",
    "The BibTeX entry is on the clipboard.": "BibTeX 条目已复制到剪贴板。",
    "MIT licence; the Windows package also contains components under other licences — see Help ▸ Third-party Notices":
        "MIT 许可证；Windows 便携包里还有使用其他许可证的组件——见 帮助 ▸ 第三方声明",
    "Copied to the clipboard.": "已复制到剪贴板。",
    # Example Data, the error log, Report a Problem
    "GISAXS — Ag &Nanoparticles (GALAXI)…": "GISAXS——Ag 纳米颗粒（GALAXI）(&N)…",
    "SAXS — Ag&Bh Calibrant (pyFAI)…": "SAXS——AgBh 标样（pyFAI）(&B)…",
    "GIWAXS — &MAPbI₃ Film (DESY P08)…": "GIWAXS——MAPbI₃ 薄膜（DESY P08）(&M)…",
    "GIWAXS — &PbTe, Tilted Detector (Xeuss)…": "GIWAXS——PbTe，倾斜探测器（Xeuss）(&P)…",
    "GISAXS of Ag nanoparticles, GALAXI (Jülich), Pilatus 1M, with its calibration (.poni). Source: BornAgain test "
    "data. Licence: GPL-3.0": "Ag 纳米颗粒的 GISAXS，GALAXI（于利希），Pilatus 1M，带它的标定（.poni）。来源：BornAgain 测试数据。"
                              "许可证：GPL-3.0",
    "Silver behenate calibration image, ESRF BM26, Pilatus 1M, with its .poni. Source: pyFAI test images. Licence: "
    "MIT (pyFAI), to be confirmed for the image files": "山嵛酸银标定图像，ESRF BM26，Pilatus 1M，带它的 .poni。"
                                                        "来源：pyFAI 测试图像。许可证：MIT（pyFAI），图像文件本身的许可证待确认",
    "GIWAXS of a MAPbI₃ perovskite film, DESY P08, PerkinElmer 2048 × 2048, with its LaB₆ calibration (.poni). "
    "Source: pygid usage examples, Zenodo 10.5281/zenodo.17466183. Licence: CC BY 4.0":
        "MAPbI₃ 钙钛矿薄膜的 GIWAXS，DESY P08，PerkinElmer 2048 × 2048，带它的 LaB₆ 标定（.poni）。"
        "来源：pygid 使用示例，Zenodo 10.5281/zenodo.17466183。许可证：CC BY 4.0",
    "GIWAXS of PbTe nanoplatelets, Xeuss 2.0 lab source, Pilatus 300K, 6° detector tilt, with its .poni and mask. "
    "Source: pygid usage examples, Zenodo 10.5281/zenodo.17466183. Licence: CC BY 4.0":
        "PbTe 纳米片的 GIWAXS，Xeuss 2.0 实验室光源，Pilatus 300K，探测器倾斜 6°，带它的 .poni 和掩膜。"
        "来源：pygid 使用示例，Zenodo 10.5281/zenodo.17466183。许可证：CC BY 4.0",
    "Example Data": "示例数据",
    "GIMaP copies the example {name} to this folder and opens it in Analyze. What you export is written next to "
    "the copy, never into the program folder.": "GIMaP 把示例 {name} 复制到这个文件夹，并在分析页中打开。导出的文件写在副本旁边，"
                                                "不会写进程序文件夹。",
    "Copy and Open": "复制并打开",
    "The example data are not in this installation of GIMaP (looked in {folder}).":
        "这个 GIMaP 安装里没有示例数据（查找的位置：{folder}）。",
    "The example could not be copied: {error}": "无法复制示例：{error}",
    "Copied the example to {folder} and opened it in Analyze.": "已把示例复制到 {folder}，并在分析页中打开。",
    "Copied the example to {folder}. Analyze is busy: open {name} from there when it is free.":
        "已把示例复制到 {folder}。分析页正忙：等它空闲后从那里打开 {name}。",
    "Error Log": "错误日志",
    "No error has been logged.": "还没有记录过任何错误。",
    "The versions of GIMaP and of your system (no paths, no data) are on the clipboard: paste them into the report.":
        "GIMaP 和系统的版本信息（不含路径和数据）已复制到剪贴板：请把它粘贴到问题报告里。",
}

__all__ = ["MENUS_ZH"]
