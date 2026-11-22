"""Chinese of the fixes after the third user test: the question when frames are opened over listed ones, the
symmetry of a CIF phase from its space group, the assignment in the peak table, the pole-figure verdict, the
Mask & corrections step and the RMS of a calibration."""

REVIEW_PHASES_ZH = {
    # opening frames while others are listed (bindings/frame_import.py)
    "Answer the open question about the frames listed first.": "请先回答关于已列出帧的问题。",
    # the symmetry of a CIF phase (domain/crystal/cif.py, space_groups.py; indexing_texts.py, indexing_panel.py)
    "{group}positions only: no symmetry operations, every hkl, no intensities":
        "{group}仅位置：没有对称操作，列出所有 hkl，没有强度",
    "{group}{count} operations from the space group (none in the CIF)": "{group}由空间群生成的 {count} 个操作（CIF 中没有）",
    "Formula: {formula}": "化学式：{formula}",
    "See the Symmetry column.": "见“对称性”一列。",
    "{count} symmetry operations generated from the space group {symbol} (No. {number}): the file lists none.":
        "由空间群 {symbol}（第 {number} 号）生成了 {count} 个对称操作：文件中没有列出。",
    "The file does not say which origin choice of {symbol} its coordinates use: choice {choice} was taken (it puts "
    "the atoms on fewer sites).":
        "文件没有说明坐标用的是 {symbol} 的哪种原点选择：采用了选择 {choice}（这样原子占据的位置更少）。",
    "The space-group symbol “{symbol}” cannot be read.": "无法识别空间群符号“{symbol}”。",
    "The space group “{symbol}” is not in GIMaP's table (a setting it does not know).":
        "空间群“{symbol}”不在 GIMaP 的表中（未收录的设置）。",
    "The cell does not have the shape the space group {symbol} needs.": "晶胞的形状不符合空间群 {symbol} 的要求。",
    "The space-group symbol {symbol} and number {number} disagree.": "空间群符号 {symbol} 与编号 {number} 不一致。",
    "There is no space group number {number}.": "不存在第 {number} 号空间群。",
    "The file names no space group.": "文件没有给出空间群。",
    "No symmetry operations: positions only. Every hkl of the cell is listed, without intensities or extinctions, "
    "and rings assigned to this phase are tentative.":
        "没有对称操作：仅给出位置。列出晶胞的所有 hkl，不含强度和消光；指认到此物相的衍射环只是暂定。",
    # the assignment in the peak table of Results (assistant/presentation/guided_tables.py)
    "phase": "物相",
    "The phase of the nearest reflection within max(3 σ(q0), FWHM/2), from the phases given (Analyze ▸ Phases). ? "
    "marks an assignment to a phase without symmetry operations (positions only): tentative.":
        "在 max(3 σ(q0), FWHM/2) 之内最近的衍射所属的物相，来自给定的物相（分析 ▸ 物相）。? 表示指认到没有对称操作"
        "（仅位置）的物相：暂定。",
    "The reflection (one per family) at q_hkl; the tooltip lists the others within the window.":
        "q_hkl 处的衍射（每族一个）；提示中列出窗口内的其他衍射。",
    "q0 − q_hkl: how far the peak lies from the reflection (the window that accepted it in the tooltip).":
        "q0 − q_hkl：峰离该衍射有多远（接受它的窗口见提示）。",
    "Tentative: this phase gives positions only (no symmetry operations).": "暂定：此物相只给出位置（没有对称操作）。",
    "Also within the window:\n{others}": "窗口内还有：\n{others}",
    "q_hkl = {q_hkl} Å⁻¹; window ±{window} Å⁻¹": "q_hkl = {q_hkl} Å⁻¹；窗口 ±{window} Å⁻¹",
    # the pole figure (indexing_texts.py)
    "(at its bound)": "（在边界上）",
    "Not determined by this fit: {what} at its bound. Best determined: Hermans f = {f} from the measured bins.":
        "此拟合未能确定：{what}停在边界上。最可靠的是只用测到的分档得到的 Hermans f = {f}。",
    "oriented fraction": "取向分数",
    "direction χ0": "方向 χ0",
    # Mask & corrections (bindings/workspace.py)
    "intensity × {factor}": "强度 × {factor}",
    "Lorentz factor": "Lorentz 因子",
    # the calibration result (calibration/presentation/residual_text.py)
    "Ring-point RMS:": "环上点 RMS：",
    "Ring-radius RMS:": "环半径 RMS：",
    "RMS distance of the points sampled on the matched rings from the rings of this geometry, with the detector tilt.":
        "在匹配环上取的各点到此几何（含探测器倾斜）所给环的均方根距离。",
    "The same points without the tilt: {flat} px.": "不计倾斜时，同样这些点为 {flat} px。",
    "RMS difference between the radius of each matched ring (azimuthal average about the centre) and the radius this "
    "geometry predicts.": "各匹配环的半径（绕中心作方位角平均）与此几何预测的半径之差的均方根。",
}

__all__ = ["REVIEW_PHASES_ZH"]
