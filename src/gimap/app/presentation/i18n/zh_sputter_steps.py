"""Chinese texts of the step tree of Sputter ▸ Analyze, the joint fit's progress, the continuous metal layer and
Details ▸ Storage.

Keys are the exact English of ``features/sputter`` (``presentation/step_texts.py``, ``step_pictures.py``,
``storage_panel.py``, ``live_status.py``, ``views/sputter_steps_view.py``, ``application/step_details.py``); templates
keep their ``{fields}``. The package's one-line summaries are matched against ``SUMMARY_TEMPLATES`` and said here with
the same numbers; file and network names are data.
"""

SPUTTER_STEPS_ZH = {
    # -- the continuous metal layer (tile, curve, column) -------------------------------------------------------------
    "Continuous layer t_b": "连续金属层 t_b",
    "Thickness t_b of the continuous metal layer under the grains (0: separate islands).":
        "晶粒下面连续金属层的厚度 t_b（0：彼此分开的岛）。",
    # -- the joint fit at work (live_status.py) --------------------------------------------------------------------
    "Joint fit: iteration {iteration} of {total} · {phase}: {done} of {windows} windows":
        "联合精修：第 {iteration}/{total} 次迭代 · {phase}：{done}/{windows} 个时间窗",
    "Joint fit: preparing · {done} of {windows} windows": "联合精修：准备中 · {done}/{windows} 个时间窗",
    "Joint fit: the starting point · {done} of {windows} windows": "联合精修：计算起点 · {done}/{windows} 个时间窗",
    # -- the export (export_flow.py, application/client.py) ------------------------------------------------------------
    "Export…": "导出…",
    "Write the result shown for a paper or a talk: figures as PDF, SVG and PNG (300 dpi), the solution shown in each "
    "window as a CSV with units, every solution, and a methods text with this run's numbers.":
        "把显示的结果写成可用于论文或报告的文件：图（PDF、SVG 和 300 dpi 的 PNG）、每个时间窗显示的解（带单位的 CSV）、"
        "全部解，以及带本次运行数字的方法说明。",
    "Choose the Folder for the Export": "选择导出到的文件夹",
    "Exporting {scan}…": "正在导出 {scan}…",
    "Exported {count} files to {folder}": "已导出 {count} 个文件到 {folder}",
    "This analysis package cannot export (it is older than the export): update GIMaP or choose a newer package.":
        "这个分析包不能导出（它早于导出功能）：请更新 GIMaP，或选择较新的分析包。",
    "The export stopped with an error; its last lines are under Details.": "导出出错停止；最后几行在“详情”里。",
    "The result shown has no window with a solution to export.": "显示的结果里没有带解的时间窗，无法导出。",
    "derivatives": "求导数",
    "trial step": "试探步",
    # -- the titles of the steps -----------------------------------------------------------------------------------
    "Find the beam centre": "确定束心",
    "Find the pixel mask": "生成像素掩膜",
    "Read and bin the frames": "读入帧并按时间归并",
    "Find the start of the deposition": "确定沉积开始时刻",
    "Lay out the time windows": "划分时间窗",
    "The networks' solutions": "各网络的解",
    "Compare the networks": "比较网络",
    "Joint fit of the whole scan": "整条扫描联合精修",
    "Prepare (every window tries its neighbours' solutions)": "准备（每个时间窗试用相邻时间窗的解）",
    "Iterations": "迭代",
    "Write the result": "写出结果",
    "Pictures of the main screen": "主屏图像",
    # -- the package's summaries -------------------------------------------------------------------------------------
    "found in the frames: column {col}, row {row}": "由帧求得：列 {col}，行 {row}",
    "given: column {col}, row {row}": "给定：列 {col}，行 {row}",
    "{dead} dead, {hot} hot, {shadow} beamstop / holder pixels besides {file}":
        "{file} 之外：坏点 {dead}，热点 {hot}，挡束器 / 支架 {shadow}",
    "{file} only (no automatic mask)": "仅用 {file}（无自动掩膜）",
    "{read} of {total} frames in {seconds} s, {bins} bins; disk {disk} s per frame (the disk is the slow part)":
        "{total} 帧中读入 {read} 帧，用时 {seconds} s，{bins} 个时间段；磁盘每帧 {disk} s（慢在磁盘）",
    "{read} of {total} frames in {seconds} s, {bins} bins; disk {disk} s per frame":
        "{total} 帧中读入 {read} 帧，用时 {seconds} s，{bins} 个时间段；磁盘每帧 {disk} s",
    "{read} of {total} frames in {seconds} s": "{total} 帧中读入 {read} 帧，用时 {seconds} s",
    "found when the windows are made ({error})": "划分时间窗时确定（{error}）",
    "{count} windows of 1 s, {first}-{last} s, every {step} s": "{count} 个 1 s 时间窗，{first}–{last} s，每 {step} s 一个",
    "{count} windows of 1 s, {first}-{last} s": "{count} 个 1 s 时间窗，{first}–{last} s",
    "median fit deviation {deviation} (best candidate), {windows} windows, {seconds} s":
        "拟合偏差中位数 {deviation}（最佳候选），{windows} 个时间窗，{seconds} s",
    "{iterations} iterations, median fit deviation {deviation}, metal amount {kappa} x nominal, {minutes} min":
        "{iterations} 次迭代，拟合偏差中位数 {deviation}，金属量为名义值的 {kappa} 倍，{minutes} 分钟",
    "{iterations} iterations, median fit deviation {deviation}, {minutes} min":
        "{iterations} 次迭代，拟合偏差中位数 {deviation}，{minutes} 分钟",
    "{iterations} iterations with {workers} processes": "{iterations} 次迭代，{workers} 个进程",
    "{windows} windows, {solutions} solutions tried in {seconds} s": "{windows} 个时间窗，{seconds} s 内试了 {solutions} 个解",
    "{family} fits better ({deviations}); its result is shown first": "{family} 更吻合（{deviations}）；先显示它的结果",
    "{family} fits better ({deviations})": "{family} 更吻合（{deviations}）",
    "mode {mode}: the network's solutions and their time path": "{mode} 模式：网络的解及其时间路径",
    "median fit deviation {deviations}": "拟合偏差中位数 {deviations}",
    "{windows} windows": "{windows} 个时间窗",
    "{onset} s ({how})": "{onset} s（{how}）",
    "imported before": "此前已读入",
    "imported before: no frame read again": "此前已读入：不再重读任何帧",
    "run with the network (mode full)": "与网络一起运行（完整模式）",
    "did not finish (see the log of this step): the network's solutions are shown":
        "没有完成（见这一步的日志）：显示网络的解",
    "stopped: see the log of this step": "已停止：见这一步的日志",
    "see the log of the import": "见读入的日志",
    "not found in the frames": "帧里没有找到",
    "given": "给定",
    "dark first frames: from the lit bare plateau": "最初几帧是暗的：取自照亮后的裸基底平台",
    "the rate rises above 1.05 x the 0.3-1.0 s baseline": "计数率升到 0.3–1.0 s 基线的 1.05 倍以上",
    # -- what a step found (step_texts.py, step_pictures.py) ----------------------------------------------------------
    "Beam centre ≈ ({col}, {row}) px · detector binned {binning} × {binning}":
        "束心 ≈ ({col}, {row}) px · 探测器按 {binning} × {binning} 合并",
    "{frames} frames summed · {masked} of {pixels} pixels left out by the mask (found in these frames: {dead} dead, "
    "{hot} hot, {shadow} of beamstop and holder) · dashed: the sample horizon":
        "{frames} 帧求和 · 掩膜去掉 {pixels} 个像素中的 {masked} 个（其中由这些帧找出的：坏点 {dead}，热点 {hot}，"
        "挡束器及支架 {shadow}）· 虚线：样品地平线",
    "{read} of {total} frames read in {time}": "{total} 帧中读入 {read} 帧，用时 {time}",
    "every frame read": "读入了全部帧",
    "only the frames of {windows} time windows, every {step} s": "只读 {windows} 个时间窗用到的帧，每 {step} s 一个",
    "{mb} MB read": "读入 {mb} MB",
    "Frames read: {ranges}": "读入的帧：{ranges}",
    "{ranges} … ({count} runs in all)": "{ranges} …（共 {count} 段）",
    "Start of the deposition: {onset} s ({how})": "沉积开始：{onset} s（{how}）",
    "the rule alone: {onset} s": "仅按规则：{onset} s",
    "baseline {baseline} (the mean rate {first}–{last} s)": "基线 {baseline}（{first}–{last} s 的平均计数率）",
    "{count} time windows, {first}–{last} s, every {step} s": "{count} 个时间窗，{first}–{last} s，每 {step} s 一个",
    "at most {count}": "最多 {count} 个",
    "{family} {deviation} (chosen)": "{family} {deviation}（选用）",
    "{done} of at most {total} iterations · median fit deviation {first} → {last} · {time}":
        "{done} 次迭代（最多 {total} 次）· 拟合偏差中位数 {first} → {last} · {time}",
    "This result has no record of its steps: it was made by an older analysis package (the result itself is "
    "complete).": "这个结果没有各步骤的记录：它由较早的分析包生成（结果本身是完整的）。",
    "This step saved no details.": "这一步没有保存细节。",
    "Click a step marked › to see what it found.": "点带 › 的步骤，查看它的产出。",
    "Back to the Analysis": "回到分析",
    "Count rate (onset band)": "计数率（判定沉积开始的区域）",
    "Baseline": "基线",
    "The rule alone": "仅按规则",
    "Median fit deviation": "拟合偏差中位数",
    "Iteration": "迭代次数",
    "Solution after iteration {iteration}": "第 {iteration} 次迭代后的解",
    "The details of this step are not in {folder} (the file {name} is missing).":
        "{folder} 里没有这一步的细节（缺少文件 {name}）。",
    "The details of this step cannot be read: {reason}": "无法读取这一步的细节：{reason}",
    # -- the live view and the main screen (views) ---------------------------------------------------------------------
    "Show the Preliminary Result": "显示初步结果",
    "The steps of the analysis at work and what each has found (the live view).": "正在进行的分析的各步骤及其产出（实时视图）。",
    "Close what this step found and follow the analysis again.": "关闭这一步的产出，回到跟随分析。",
    # -- Details ▸ Storage (views/sputter_steps_view.py, storage_panel.py) --------------------------------------------
    "Storage": "存储",
    "Remove the imports chosen in the table (the scan is read again the next time it is analysed).":
        "删除表中选中的读入数据（下次分析该扫描时会重新读入）。",
    "Remove Unused For": "删除未使用超过",
    "Remove the imports no analysis has used for at least this many days.": "删除至少这么多天没有被任何分析用过的读入数据。",
    "Limit of the imports": "读入数据上限",
    "When the imports together take more than this, the ones not used for the longest time are removed when an "
    "analysis starts and ends — never the one in use, never a result. 0: no limit.":
        "读入数据合计超过这个大小时，分析开始和结束时删除最久没用过的——绝不删除正在使用的，也绝不删除结果。0：不设上限。",
    "Scan import": "扫描的读入数据",
    "Size": "大小",
    "Last used": "上次使用",
    " days": " 天",
    "No limit": "不设上限",
    "Imports: {size} in {count} · {folder}": "读入数据：{count} 份，共 {size} · {folder}",
    "Results: {size} in {folder} (never removed automatically)": "结果：{folder} 中共 {size}（从不自动删除）",
    "This analysis package keeps no imports GIMaP can list.": "这个分析包没有 GIMaP 能列出的读入数据。",
    "Adding up the sizes…": "正在统计大小…",
    "{label} (in use)": "{label}（正在使用）",
    "local folder": "本机文件夹",
    "user data folder (older GIMaP)": "用户数据文件夹（旧版 GIMaP）",
    "Remove Imports": "删除读入数据",
    "Remove {count} imports ({size})? Their scans are read again from the frames the next time they are analysed; the "
    "results stay.": "删除 {count} 份读入数据（{size}）？下次分析这些扫描时会重新从帧读入；结果保留。",
    "No import has gone unused for {days} days.": "没有 {days} 天未被使用的读入数据。",
    "Removed {count} imports ({size}).": "已删除 {count} 份读入数据（{size}）。",
    "{label} could not be removed: {reason}": "无法删除 {label}：{reason}",
}

__all__ = ["SPUTTER_STEPS_ZH"]
