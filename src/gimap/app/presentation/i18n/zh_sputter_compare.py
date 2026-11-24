"""Chinese texts of Sputter ▸ Compare… (Compare Scans).

Keys are the exact English of ``features/sputter`` (``presentation/views/sputter_compare_view.py``,
``presentation/compare_dialog.py``, ``domain/compare.py``); templates keep their ``{fields}``. Numbers, units, scan
names and file names are not translated.
"""

SPUTTER_COMPARE_ZH = {
    # -- the button and the dialog's layout (sputter_compare_view.py) ------------------------------------------------
    "Compare…": "比较…",
    "Put the results of several scans side by side against the deposited amount: sizes, coverage, film thickness and "
    "the growth stages, for example the same metal on two polymers.":
        "把几条扫描的结果按沉积量并排比较：尺寸、覆盖率、膜厚和生长阶段，例如同一种金属在两种聚合物上。",
    "Compare Scans": "比较扫描",
    "Nominal deposit": "名义沉积量",
    "Time": "时间",
    "Each line is the solution shown of each window (as chosen on the main screen); the film thickness, the grain "
    "height from its fringe, the optical density and the lateral scale are read from the data without the model (the "
    "report). GIMaP computes none of the values. Tick, rename or add results at the left.":
        "每条线是各时间窗所显示的解（与主屏所选的一致）；膜厚、由晶粒条纹读出的高度、光学致密度和横向尺度不经模型、"
        "直接从数据读出（来自报告）。这些数值都不是 GIMaP 计算的。在左边勾选、改名或添加结果。",
    "Tick the results to compare; double-click a name to change it (the name is used in the plots and files).":
        "勾选要比较的结果；双击名称可以改名（图和文件里用这个名称）。",
    "Add Result…": "添加结果…",
    "A result folder analysed before, also from another results folder.": "以前分析过的结果文件夹，也可以在别的结果文件夹里。",
    "Take the selected result off the list (its files stay).": "把所选结果从列表里拿掉（文件保留）。",
    "Horizontal axis": "横轴",
    "The nominal deposit of each window (rate × time): scans at different rates are compared at the same amount of "
    "metal.": "每个时间窗的名义沉积量（速率 × 时间）：速率不同的扫描在相同金属量处比较。",
    "The time since the onset of the deposition.": "从沉积开始算起的时间。",
    "Growth": "生长",
    "The package's growth descriptors of each result: the metal amount at coverage 0.5 and at the film onset, the "
    "growth laws of size and distance, the stages.":
        "分析包给出的各结果的生长描述量：覆盖率达到 0.5 时和成膜时的金属量、尺寸与间距的生长规律、各生长阶段。",
    "Write the curves and the growth table (CSV) and the plots (PNG, PDF) into a folder.":
        "把曲线和生长表（CSV）以及图（PNG、PDF）写进一个文件夹。",
    # -- behaviour (compare_dialog.py) -------------------------------------------------------------------------------
    "Comparison written to {folder}": "比较结果已写入 {folder}",
    "Tick at least one result to export.": "至少勾选一个结果才能导出。",
    "Unit": "单位",
    "Add the Folder of a Result": "添加一个结果文件夹",
    "Export the Comparison to a Folder": "把比较结果导出到文件夹",
    # -- the quantities, the axis and the growth table (domain/compare.py) -------------------------------------------
    "Film thickness (fringes)": "膜厚（条纹）",
    "Grain height (its fringe)": "晶粒高度（晶粒条纹）",
    "Optical density (α/α_c)²": "光学致密度 (α/α_c)²",
    "Lateral scale 2π/q": "横向尺度 2π/q",
    "Nominal deposit (nm)": "名义沉积量（nm）",
    "Time since the onset (s)": "沉积开始后的时间（s）",
    "{stage} starts at (metal amount, nm)": "{stage}开始于（金属量，nm）",
    "Tick at least one result to compare.": "至少勾选一个结果来比较。",
}
