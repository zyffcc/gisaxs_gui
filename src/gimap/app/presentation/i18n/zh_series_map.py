"""Chinese for the Series tab of Analyze: the map, its axes and views, the cut of a series, its exports."""

SERIES_MAP_ZH = {
    "This map uses batch series corrections. Watching cannot append uncorrected frames: "
    "Build Map again or rerun Batch Export with the same corrections.":
        "此热图使用了批量序列校正。监视时不能加入未校正的帧：请重新生成热图，或用相同校正重新运行批量导出。",
    "Could not append a frame to the Series map: {error}": "无法将一帧加入序列热图：{error}",
    "Could not append {name} to the Series map: {error}": "无法将 {name} 加入序列热图：{error}",
    # -- one cut for a series (analyze/presentation/frames_model.py: ``series_cut_text``) ---------------------------
    "horizontal cut: rows {low}–{high} for every frame": "水平切线：每一帧都取第 {low}–{high} 行",
    "horizontal cut: found again in each frame": "水平切线：在每一帧里重新找",
    "I(χ) ring: q {low}–{high} Å⁻¹ for every frame": "I(χ) 的环：每一帧都取 q {low}–{high} Å⁻¹",
    "I(χ) ring: the strongest ring of each frame": "I(χ) 的环：每一帧各取自己最强的环",
    "cuts as set by you, the same for every frame": "切线按你设定的位置，每一帧相同",
    "Cut of the series — {cut}.": "序列的切线——{cut}。",
    "Find the cut again in every frame": "在每一帧里重新找切线",
    "Off (the usual way): every frame of a Series map, a Batch Export and a watched folder is cut where the frame "
    "on screen is — the Yoneda rows of a GISAXS frame, the ring of I(χ) of a GIWAXS frame — so a row differs from "
    "the next by the sample, not by where it was cut. On: the cut is found in each frame anew.":
        "不勾选（通常的做法）：序列热图、批量导出和监视文件夹里的每一帧，都在当前显示这一帧的切线位置上切——"
        "GISAXS 帧的 Yoneda 行、GIWAXS 帧 I(χ) 的环——这样相邻两行的差别来自样品，而不是切的位置不同。"
        "勾选：每一帧各自重新找切线。",
    # A frame cut at the cut of the series (application/use_cases.held_cut): the Cuts step and the plot titles
    # (bindings/workspace.CUT_SOURCES, bindings/display.cut_lines).
    "Cut of the series (the cut of one frame, held for every frame)": "序列的切线（某一帧的切线，用于每一帧）",
    "the cut of the series": "序列的切线",
    "cut of the series": "序列的切线",
    "Vertical cut I(qz) over columns {columns} (the cut of the series).": "竖直切线 I(qz)：第 {columns} 列（序列的切线）。",

    # -- what Build Map would stack (analyze/presentation/series_texts.py, bindings/series.py) ----------------------
    "{frames} frames in {groups} sums of {n}: Build Map stacks one curve per sum.":
        "{frames} 帧分成 {groups} 组，每组 {n} 帧求和：“生成热图”每组叠一条曲线。",
    "{frames} frames in {groups} sums of {n}: Build Map stacks one curve per sum; the last sums only {last}.":
        "{frames} 帧分成 {groups} 组，每组 {n} 帧求和：“生成热图”每组叠一条曲线；最后一组只有 {last} 帧。",
    "Sum {n} of the automatic analysis was switched off for the series (Data ▸ Sum N to sum again).":
        "自动分析设置的 {n} 帧求和已为这个序列关闭（要再求和：数据 ▸ 求和）。",
    "Frames {first} to {last} hold fewer than two frames: a series map needs at least two.":
        "第 {first} 至 {last} 帧不足两帧：序列热图至少需要两帧。",

    # -- what the map is (the line above it) ------------------------------------------------------------------------
    "{n} frames of {name}": "{name} 的 {n} 帧",
    "{n} of {listed} frames of {name}": "{name} 的 {listed} 帧中的 {n} 帧",
    "{n} rows of {name}": "{name} 的 {n} 行",
    "each row the sum of {n} frames": "每行是 {n} 帧之和",
    "each row the sum of {n} frames; the last row sums only {last} of {n} frames":
        "每行是 {n} 帧之和；最后一行只有 {n} 帧中的 {last} 帧",
    "time from the frames' time stamps": "时间取自各帧的时间戳",
    "time = frame × exposure": "时间 = 帧号 × 曝光时间",
    "{axis} > 0 half": "{axis} > 0 半边",
    "{axis} < 0 half": "{axis} < 0 半边",
    "both halves on |{axis}|": "两半都画在 |{axis}| 上",
    "mean of both halves": "两半取平均",
    "Built with other settings — Build Map again.": "这张热图是用另一套设置生成的——请再点“生成热图”。",
    "The Curve box shows another curve ({curve}) — Build Map stacks it.":
        "“曲线”框里现在是另一条曲线（{curve}）——点“生成热图”才会叠它。",

    # -- frames, sums, the frame axis -------------------------------------------------------------------------------
    "Frame {n} of {total}: {name}": "第 {n} 帧（共 {total} 帧）：{name}",
    "sum of {n}": "{n} 帧之和",
    "sum of {n} of {group}": "仅 {n} 帧之和（应为 {group} 帧）",
    "Incomplete sums": "不完整的求和",
    "Frame number": "帧号",
    "Time (s)": "时间 (s)",
    "Frame axis": "帧轴",
    "What the frame axis of the map and of the traces counts in: the number of the frame (in its file, "
    "or in the list), or the time since the first row when the frames carry one":
        "热图和随帧变化曲线的帧轴按什么计数：帧号（文件内的帧号，或在列表中的序号），或自第一行起的时间（帧带有时间信息时）",
    "The frames of this map carry no time (no time stamp per frame, no exposure time of one file).":
        "这张热图的帧没有时间信息（没有逐帧的时间戳，也不是同一文件里带曝光时间的帧）。",

    # -- the halves of a signed cut (views/series_view.py) ----------------------------------------------------------
    "The halves of the signed cut this map stacks: ± both as measured, + or − one half, |x| both on |x|, or "
    "their mean. The stages, the traces and the exports follow it.":
        "这张热图叠的是带正负号切线的哪一半：± 两半按测量的原样，+ 或 − 只取一半，|x| 两半都画在 |x| 上，"
        "或取两半的平均。阶段、随帧变化的曲线和导出都随之改变。",
    "Both halves, on the signed axis (as measured)": "两半都显示，横轴带正负号（按测量的原样）",
    "Only the half with x > 0": "只取 x > 0 的一半",
    "Only the half with x < 0, on |x|": "只取 x < 0 的一半，画在 |x| 上",
    "Both halves on |x|: every measured point of the two": "两半都画在 |x| 上：两半的每个测量点都保留",
    "The mean of both halves on |x|; where one half has no points (a detector gap), the other half alone":
        "两半在 |x| 上取平均；某一半没有数据点的地方（探测器缝隙），只用另一半",

    # -- the options, folded under the plots ------------------------------------------------------------------------
    "Options: frames, cut, axes": "选项：帧、切线、坐标轴",
    "Which frames Build Map takes and where they are cut; what the frame axis counts in; what the stages "
    "compare. The cut holds for a Batch Export and a watched folder too.":
        "“生成热图”取哪些帧、在哪里切；帧轴按什么计数；阶段比较什么范围。切线的选择同样用于批量导出和监视文件夹。",
    "the last": "最后一帧",
    "The first frame of the list a map takes (the frames before it are left out)": "热图从列表的第几帧开始（它之前的帧不取）",
    "The last frame a map takes; at its least value: the last frame listed": "热图取到第几帧；取最小值时：到列表的最后一帧",
    "Stages compare": "阶段比较",
    "The whole x axis": "整个 x 轴",
    "The x range shown in the map": "热图当前显示的 x 范围",
    "What the stages and odd frames are found in: every x of the map, or only the x range the map shows "
    "(zoom the map: the stages follow)":
        "在什么范围内寻找阶段和异常帧：热图的全部 x，或只是热图当前显示的 x 范围（缩放热图，阶段随之更新）",
    "Show the picked frame's image": "同时显示所选帧的图像",
    "Picking a row of the map also shows that frame in Analyze (its image and curves); a large frame takes "
    "a moment to read. Off: Open shows it.":
        "在热图上选中一行时，也在分析页显示这一帧（图像和曲线）；大的帧需要一点时间读取。不勾选：点“打开”才显示。",
    "Wide": "加宽",
    "More room for this tab: the image beside it shrinks to its least width, so the map and both plots can "
    "be seen together. Click again for the image.":
        "给这一页更多空间：旁边的图像缩到最小宽度，热图和两张曲线图可以同时看到。再点一次恢复图像。",

    # -- stages of a short map --------------------------------------------------------------------------------------
    "Stages need {n} frames or more (this map has {rows}).": "寻找阶段至少需要 {n} 帧（这张热图只有 {rows} 帧）。",
    "Stages are found in a map of {n} frames or more (this map has {rows}).":
        "至少 {n} 帧的热图才会寻找阶段（这张热图只有 {rows} 帧）。",
    "Change along the series: needs a map of five frames or more": "沿序列的变化：需要至少五帧的热图",

    # -- after the review: rows that are sums, a frame range, the time of a file, an empty window --------------------
    "{rows} rows (sums of {n} frames) × {points} points. Click or drag the horizontal band to pick a row, drag "
    "the vertical band (and its edges) to pick a {axis} window; Open shows the row's frames in Analyze.":
        "{rows} 行（每行 {n} 帧之和）× {points} 个点。点击或拖动水平带选择一行，拖动竖直带（及其边缘）选择 {axis} 窗口；"
        "“打开”在分析页显示这一行的帧。",
    "Series map: {n} rows (sums of {group} frames) of {curve}": "序列热图：{curve}，{n} 行（每行 {group} 帧之和）",
    "Series map: {n} rows (sums of {group} frames) of {curve}; {failed} failed: {names}":
        "序列热图：{curve}，{n} 行（每行 {group} 帧之和）；{failed} 行失败：{names}",
    "{curve} — {n} rows (sums of {group})": "{curve} — {n} 行（每行 {group} 帧之和）",
    "Build Map takes frames {first} to {last} only (Options).": "“生成热图”只取第 {first} 至 {last} 帧（见选项）。",
    "Build Map takes frames {first} to the last only (Options).": "“生成热图”只取第 {first} 帧至最后一帧（见选项）。",
    "Options: frames {first} to {last}, cut, axes": "选项：第 {first} 至 {last} 帧、切线、坐标轴",
    "Options: frames {first} to the last, cut, axes": "选项：第 {first} 帧至最后一帧、切线、坐标轴",
    "time = start of the file + frame × exposure": "时间 = 文件开始时间 + 帧号 × 曝光时间",
    "I at {axis} = {x} (the point nearest to the window)": "{axis} = {x} 处的 I（离窗口最近的点）",
    "No measured point in this window (a detector gap, or beyond the data): move the band onto the data.":
        "这个窗口里没有测得的点（探测器缝隙，或超出数据范围）：请把竖直带移到数据上。",

    # -- the peak in the window: a moment width, not a fitted FWHM ---------------------------------------------------
    "Peak width (2.355 σ)": "峰宽（2.355 σ）",
    "In the x window of the vertical band, for every frame: the mean intensity, or the peak found there "
    "above a straight background — its centroid, its moment width (2.355 × the standard deviation inside "
    "the window: the FWHM of a Gaussian peak, larger for a wider window), its area and its height. "
    "A fitted FWHM with its error: Batch Export ▸ Fit ▸ Peaks.":
        "在竖直带的 x 窗口内，对每一帧：平均强度，或其中直线背景之上的峰——质心位置、矩宽度（窗口内标准差的 2.355 倍："
        "对高斯峰就是 FWHM，窗口越宽它越大）、面积和峰高。带误差的拟合 FWHM：批量导出 ▸ 拟合 ▸ 峰。",
    "CSV: frame, file, and the peak in the x window — centroid position, moment width (2.355 σ), area and height":
        "CSV：帧、文件，以及 x 窗口内的峰——质心位置、矩宽度（2.355 σ）、面积和峰高",

    # -- Export ▾ ---------------------------------------------------------------------------------------------------
    "Selected Frame's Curve as Figure…": "所选帧的曲线（图）…",
    "Selected Frame's Curve as Table…": "所选帧的曲线（表格）…",
    "Trace as Figure…": "随帧变化的曲线（图）…",
    "Trace as Table…": "随帧变化的曲线（表格）…",
    "CSV: the value of the trace plot for every frame": "CSV：右下图里每一帧的数值",
    "Export the Frame's Curve": "导出这一帧的曲线",
    "Export the Trace": "导出随帧变化的曲线",
    "The trace has no values yet.": "随帧变化的曲线还没有数值。",
    "Could not export the curve: {error}": "无法导出曲线：{error}",
    "Could not export the trace: {error}": "无法导出随帧变化的曲线：{error}",
}

__all__ = ["SERIES_MAP_ZH"]
