"""Chinese for Analyze (with Run Automatic Analysis, Batch Export, Geometry…): the shorter on-screen texts and the tooltips
that took over the explanations which used to stand under the controls (2026-11 small-text review)."""

TIPS_ANALYZE_ZH = {
    # -- the command bar, the Data step, the plots -------------------------------------------------------------
    "No data yet": "还没有数据",
    "Open or drop detector frames (CBF, NeXus/HDF5, TIFF, EDF) or a folder.":
        "打开或拖入探测器帧（CBF、NeXus/HDF5、TIFF、EDF）或文件夹。",
    "Pixel size {size} µm": "像素尺寸 {size} µm",
    "1 frame imported": "已导入 1 帧",
    "I(q) of the regions": "各区域的 I(q)",
    # -- Geometry, Mask, Cuts ----------------------------------------------------------------------------------------
    "Instrument profile: the detector geometry used for q — distance, beam centre, wavelength, αi":
        "仪器配置：用于计算 q 的探测器几何——距离、光束中心、波长、αi",
    "Drag the cyan cross on the image to move the centre; the change holds for every file of this detector until "
    "Back to Profile Centre (Save to Profile keeps it).":
        "拖动图上的青色十字来移动中心；在“回到配置中的中心”之前，这一改动对这个探测器的所有文件有效"
        "（“保存到配置”会永久保留）。",
    "Pixels without data — detector gaps, bad pixels, your masks — are left out of every curve before any cut; "
    "the corrections below apply to the whole frame.":
        "没有数据的像素（探测器缝隙、坏像素、你的掩膜）在任何切线之前就从每条曲线中排除；下面的校正作用于整帧。",
    "Valid raw intensity:": "有效原始强度：",
    "Pixels whose raw value is outside Min–Max are ignored": "原始值在最小–最大范围之外的像素被忽略",
    "The horizontal cut at the Yoneda band gives the in-plane structure, the vertical cut the out-of-plane one. "
    "Make left and right symmetric before averaging the halves. Drag the orange band on the image to move the "
    "horizontal cut (its edges change its height), or double-click a column to move the vertical cut.":
        "Yoneda 带处的水平切线给出面内结构，竖直切线给出面外结构。先使左右对称，再对两半取平均。"
        "拖动图上的橙色带移动水平切线（拖动边缘可改变高度），或双击某一列移动竖直切线。",
    "Every region is cut along q (upper plot) and along χ (lower plot). Tick a region to show its curves and outline; "
    "select it to change it below. Delete removes a region you added. Drag a region on the Cake view (χ against q) "
    "to move or resize it; the q map shows it in the same colour, Sources shows its pixels on the detector.":
        "每个区域沿 q（上图）和沿 χ（下图）各切一条曲线。勾选区域以显示其曲线和轮廓；选中后可在下方修改。"
        "按 Delete 删除你添加的区域。在展开图（χ 对 q）上拖动区域可移动或改变大小；q 图用同样的颜色显示它，"
        "“来源”在探测器图上显示它的像素。",
    "Every χ above the horizon; its bins and the q or 2θ axis: More cut settings.":
        "地平线以上所有 χ；分箱和 q 或 2θ 坐标在“更多切线设置”里。",
    # -- Export --------------------------------------------------------------------------------------------------------
    "The detector image or q map as shown, with a colour bar (PNG, TIFF, SVG, PDF), to a file you choose.":
        "按当前显示保存探测器图像或 q 图，带色标（PNG、TIFF、SVG、PDF），保存到你选的文件。",
    "The intensity on a regular q grid (qy–qz or q∥–qz) as a CSV table with its axes, to a file you choose.":
        "规则 q 网格（qy–qz 或 q∥–qz）上的强度，带坐标轴的 CSV 表，保存到你选的文件。",
    "The upper and lower plots as figures, one column wide, to a file you choose.":
        "上下两个曲线图保存为单栏宽度的图，保存到你选的文件。",
    # -- the Series tab --------------------------------------------------------------------------------------------
    "Several frames needed: list more files, open a folder or a multi-frame NeXus file.":
        "需要多帧：列出更多文件、打开文件夹或多帧 NeXus 文件。",
    # -- Batch Export ------------------------------------------------------------------------------------------------
    "Curves to save:": "要保存的曲线：",
    "For the tables of all frames and the per-frame curves": "用于所有帧的表格和逐帧曲线",
    "A fit takes one curve on |qy|: the mean of the two halves of the horizontal cut, or one half — never both "
    "halves together. Fitting ▸ In-situ series starts with this choice. Used when the curve for Fitting is saved "
    "(Data, above) or a particle model is fitted (below).":
        "拟合只用 |qy| 上的一条曲线：水平切线两半的平均，或其中一半——不会把两半放在一起拟合。拟合 ▸ 原位序列 "
        "从这个选择开始。保存拟合用曲线（上方“数据”）或拟合颗粒模型（下方）时才用到。",
    # -- Geometry… -------------------------------------------------------------------------------------------------
    "Beam centre in pixel-edge coordinates: the first pixel's centre is 0.5.": "光束中心用像素边界坐标：第一个像素的中心是 0.5。",
    "Beam centre = direct beam in pixels, pixel j spanning [j, j+1] (the centre of the first pixel is 0.5).":
        "光束中心 = 直射束的像素位置，第 j 个像素覆盖 [j, j+1]（第一个像素的中心是 0.5）。",
    "Use this centre for this session and redraw the curves": "在本次会话中使用这个中心并重画曲线",
    "Save all geometry values for this detector: the profile is used for every frame from this detector with this size":
        "保存此探测器的全部几何参数：这个配置会用于该探测器、该尺寸的每一帧",
    # -- Run Automatic Analysis: the Results step and tab ---------------------------------------------------------------
    "Beamtime notes (optional)": "束线时笔记（可选）",
    "αi, energy, pixel size and where the calibration is — whatever you know.":
        "αi、能量、像素尺寸、标定文件在哪——知道什么就写什么。",
    "q, FWHM in Å⁻¹ · d in Å · L in nm (Scherrer, a lower bound)": "q、FWHM：Å⁻¹ · d：Å · L：nm（Scherrer 下限）",
    "Phases are not assigned automatically: ask the AI or compare q with the lines of your material. Hover a cell "
    "for the whole sentence; the saved report has the full table.":
        "物相不会自动指认：可以问 AI，或把 q 与你的材料的衍射线对比。鼠标停在单元格上可看完整说明；保存的报告中有完整的表格。",
    "Missing wedge (red dashed next to qz): |χ| {where} Å⁻¹ not measured — Hermans f is biased low.":
        "缺失楔形区（qz 旁的红色虚线）：|χ| {where}（q 以 Å⁻¹ 计）未测量——Hermans f 偏低。",
    "Ring q ≈ {q} Å⁻¹: f = {f} (random ring: {random}) — {texture}": "环 q ≈ {q} Å⁻¹：f = {f}（随机取向：{random}）—— {texture}",
    "A random ring would give f = {random} here.": "随机取向的环在这里会给出 f = {random}。",
    "{what}: {decision}": "{what}：{decision}",
    "Halves: {halves}": "两半：{halves}",
    "2π/q* = {d} nm at the peak of the cut (|qy| = {q} Å⁻¹): not necessarily the particle spacing.":
        "2π/q* = {d} nm，位于切线的峰处（|qy| = {q} Å⁻¹）：不一定是粒子间距。",
    "2π/q* = {d} nm at a shoulder of the cut (|qy| = {q} Å⁻¹, no resolved peak): not necessarily the particle spacing.":
        "2π/q* = {d} nm，位于切线的肩峰处（|qy| = {q} Å⁻¹，没有分辨出的峰）：不一定是粒子间距。",
    "Not determined by the curve (at a search bound) — {rows}": "不由曲线确定（处于搜索边界）——{rows}",
    "Reported: rank {rank} ({model}), the best-ranked converged solution within {percent} of the first's χ²/N with "
    "no non-linear value at a search bound.":
        "报告第 {rank} 名（{model}）：χ²/N 在第一名的 {percent} 以内、已收敛且没有非线性参数处于搜索边界的解中排名最靠前的一个。",
    "Fitted: {curve} ({points} points) · R, h and D in nm.": "已拟合：{curve}（{points} 个点）· R、h、D 单位为 nm。",
    "Numerical fits of single particle families with size dispersity and a paracrystal distance D, a screening step: "
    "choose the model from what you know of the sample and confirm it in Fitting.":
        "对单一粒子类型做数值拟合（含尺寸分布和次晶距离 D），只是一次筛选：请根据你对样品的了解选择模型，并在拟合中确认。",
}
