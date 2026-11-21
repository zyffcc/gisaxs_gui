"""Chinese texts of the GISAXS cuts in Analyze's Cuts step: the table of cuts, adding and drawing them, the editor, the
question when a frame of another set-up comes, and the messages of a cut (analyze/presentation/views/gisaxs_cuts_view.py,
bindings/gisaxs_cuts.py, bindings/gisaxs_cut_display.py, domain/gisaxs_cuts.py)."""

GISAXS_CUTS_ZH = {
    # the panel (views/gisaxs_cuts_view.py)
    "List of cuts": "切线列表",
    "Angles (°)": "角度（°）",
    "Pixels (px)": "像素（px）",
    "Horizontal Band — I(qy)": "水平带 — I(qy)",
    "Vertical Band — I(qz)": "竖直带 — I(qz)",
    "Box — I(qy) or I(qz)": "框 — I(qy) 或 I(qz)",
    "Sector — I(qyz) in a direction": "扇区 — 某一方向上的 I(qyz)",
    "Line in Any Direction — I(s)": "任意方向的线 — I(s)",
    "Horizontal": "水平",
    "Vertical": "竖直",
    "Box": "框",
    "Line": "线",
    "Click a height on the image or the q map: a horizontal band there, I(qy)":
        "在图像或 q 图上点一个高度：在那里放一条水平带，得到 I(qy)",
    "Click a position on the image or the q map: a vertical band there, I(qz)":
        "在图像或 q 图上点一个位置：在那里放一条竖直带，得到 I(qz)",
    "Click two opposite corners: a box in qy × qz, cut along qy (or qz in the editor)":
        "点两个对角：qy × qz 中的一个框，沿 qy 切（可在编辑区改为沿 qz）",
    "Click in the direction of the cut: every q within ±5° of it, from the direct beam":
        "朝切线方向点一下：从直射束出发、该方向 ±5° 内的所有 q",
    "Click the two ends of a line in any direction: I along it, ±2 pixels across":
        "点任意方向一条线的两端：沿线的 I，横向 ±2 像素",
    "Every cut gives one curve: horizontal bands and boxes along qy in the upper plot, vertical bands along qz in the "
    "lower plot, sectors and lines in the lower plot's list. Tick a cut to show its curve and outline (detector and q "
    "map, same colour), double-click a name to rename it, select a cut to change it below. The first two rows are the "
    "automatic cuts; Fitting takes the horizontal one.":
        "每条切线给出一条曲线：水平带和沿 qy 的框在上图，竖直带和沿 qz 的在下图，扇区和线在下图的列表里。"
        "勾选一条切线即显示它的曲线和轮廓（探测器图和 q 图上同一颜色），双击名称可改名，选中后在下方修改。"
        "前两行是自动切线；拟合使用水平的那一条。",
    "The unit to type and show positions in. Cuts are kept in q, so they hold for every frame of this geometry. Exit "
    "angles: αf of qz, and 2θf of qy on the sample horizon; pixels: rows on the beam-centre column and columns on the "
    "horizon row.":
        "输入和显示位置所用的单位。切线以 q 保存，因此对这一几何的每一帧都成立。出射角：qz 对应 αf，"
        "qy 对应样品水平线上的 2θf；像素：光束中心所在列上的行，以及水平线所在行上的列。",
    "The cuts are in q: kept, they cut this frame at the same q; removed, only the automatic cuts stay.":
        "切线以 q 定义：保留时，在这一帧的同一 q 处切；移除后只剩自动切线。",
    "Keep Cuts": "保留切线",
    "Remove Cuts": "移除切线",
    "Draw on the image or the q map:": "在图像或 q 图上画：",
    "Cut": "切线",
    "Kind and position": "类型与位置",
    "Duplicate": "复制",
    "A copy of the selected cut, to move to another position": "复制所选切线，以便移到另一位置",
    "Remove the selected cut (the automatic cuts stay; untick them to hide them)":
        "移除所选切线（自动切线保留；取消勾选可隐藏它们）",
    "Profile along qy (mean over qz)": "沿 qy 的曲线（对 qz 求平均）",
    "Profile along qz (mean over qy)": "沿 qz 的曲线（对 qy 求平均）",
    "Every q": "所有 q",
    "The sector reaches from the direct beam to the edge of the frame": "扇区从直射束一直延伸到帧的边缘",
    # the rows and the editor (bindings/gisaxs_cuts.py)
    "Vertical cut": "竖直切线",
    "Auto I(qy)": "自动 I(qy)",
    "Auto I(qz)": "自动 I(qz)",
    "Band I(qy)": "带 I(qy)",
    "Band I(qz)": "带 I(qz)",
    "Box I(qy)": "框 I(qy)",
    "Sector I(qyz)": "扇区 I(qyz)",
    "Line I(s)": "线 I(s)",
    "Start point": "起点",
    "End point": "终点",
    "χ from the qz axis (the sign of qy), qyz = √(qy² + qz²) from the direct beam; bins one pixel wide "
    "(k·p/D = {step} Å⁻¹).":
        "χ 从 qz 轴量起（符号同 qy），qyz = √(qy² + qz²) 从直射束量起；每个区间一个像素宽（k·p/D = {step} Å⁻¹）。",
    "Points as ({axes}), kept in q; I against the distance s from the start point, in bins one pixel wide "
    "(k·p/D = {step} Å⁻¹).":
        "点按 ({axes}) 给出，以 q 保存；I 对离起点的距离 s，每个区间一个像素宽（k·p/D = {step} Å⁻¹）。",
    "Open a GISAXS frame with a geometry first.": "请先打开一帧有几何的 GISAXS 图像。",
    "{name} added: type its position below the table, or drag it on the q map.":
        "已添加 {name}：在表格下方输入它的位置，或在 q 图上拖动它。",
    "{name} added: type its position below the table.": "已添加 {name}：在表格下方输入它的位置。",
    "Removed {name} (Ctrl+Z brings it back).": "已移除 {name}（Ctrl+Z 可恢复）。",
    "{n} cut(s) were made on another set-up (other {changes}). They are kept in q and cut this frame at the same q "
    "until you choose.":
        "有 {n} 条切线是在另一套设置上做的（{changes}不同）。在你选择之前，它们以 q 保留，并在这一帧的同一 q 处切。",
    "This frame has another {changes} than the cuts were made on: keep or remove them in the Cuts step.":
        "这一帧的{changes}与做切线时的不同：请在“切线”步骤中保留或移除它们。",
    "Show Cuts": "显示切线",
    "Cuts kept for this set-up.": "已为这套设置保留切线。",
    "Cuts removed: only the automatic cuts stay (Ctrl+Z brings them back).": "已移除切线：只剩自动切线（Ctrl+Z 可恢复）。",
    "frame size": "帧尺寸",
    "wavelength": "波长",
    "detector tilt": "探测器倾斜",
    # curves and drawing (bindings/gisaxs_cut_display.py)
    "Click the height of the band on the image or the q map (Esc cancels).": "在图像或 q 图上点出水平带的高度（Esc 取消）。",
    "Click the position of the band on the image or the q map (Esc cancels).": "在图像或 q 图上点出竖直带的位置（Esc 取消）。",
    "Click two opposite corners of the box (Esc cancels).": "点出框的两个对角（Esc 取消）。",
    "Click in the direction of the sector (Esc cancels).": "朝扇区的方向点一下（Esc 取消）。",
    "Click the two ends of the line (Esc cancels).": "点出线的两端（Esc 取消）。",
    "Cuts along qz": "沿 qz 的切线",
    "Sectors": "扇区",
    "Lines": "线",
    "I(qy) of the cuts": "各切线的 I(qy)",
    "I(qz) of the cuts": "各切线的 I(qz)",
    "I(qyz) of the sectors": "各扇区的 I(qyz)",
    "I(s) along the lines": "沿各线的 I(s)",
    "Which cuts the lower plot shows": "下图显示哪些切线",
    # messages of a cut (domain/gisaxs_cuts.py)
    "Cut “{name}”: the {what} range is empty.": "切线“{name}”：{what} 范围为空。",
    "Cut “{name}”: unknown kind {kind}.": "切线“{name}”：未知类型 {kind}。",
    "Cut “{name}”: a horizontal band needs its qz range.": "切线“{name}”：水平带需要 qz 范围。",
    "Cut “{name}”: a vertical band needs its qy range.": "切线“{name}”：竖直带需要 qy 范围。",
    "Cut “{name}”: a box needs its qy and qz ranges.": "切线“{name}”：框需要 qy 和 qz 范围。",
    "Cut “{name}”: a box is cut along qy or qz.": "切线“{name}”：框只能沿 qy 或 qz 切。",
    "Cut “{name}”: a line needs a start and an end point (qy, qz).": "切线“{name}”：线需要起点和终点 (qy, qz)。",
    "Cut “{name}”: the line has no length.": "切线“{name}”：线的长度为零。",
    "Cut “{name}”: the half width of a line must be above 0.": "切线“{name}”：线的半宽必须大于 0。",
    "Cut “{name}”: no valid pixels above the horizon in it.": "切线“{name}”：其中没有水平线以上的有效像素。",
}

__all__ = ["GISAXS_CUTS_ZH"]
