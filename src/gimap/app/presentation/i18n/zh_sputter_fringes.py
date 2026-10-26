"""Chinese texts of the island height H from the q_z fringes on the Sputter page.

The Size Trend panel and the Windows columns (``features/sputter``: ``presentation/fringe_panel.py``,
``results_panel.py``), the background run (``presentation/fringe_flow.py``, ``application/fringes.py``) and the
Reliability paragraph with the statuses of a time group (``domain/fringes.py``). Numbers, units, symbols and file names
stay as they are.
"""

SPUTTER_FRINGES_ZH = {
    # -- Size Trend and Windows ------------------------------------------------------------------------------------
    "H, island height from q_z fringes (nm)": "H，由 q_z 干涉条纹得到的岛高度（nm）",
    "H from q_z fringes (68%)": "由 q_z 干涉条纹得到的 H（68%）",
    "H, one side of the beam only": "H，只来自光束一侧",
    "nominal δ = rate × (t − t0)": "名义厚度 δ = 速率 × (t − t0)",
    "below the range: H < {bound} nm": "低于测量范围：H < {bound} nm",
    "H fringes (nm)": "H 条纹（nm）",
    "H fringes 68% (nm)": "H 条纹 68% 区间（nm）",
    "from the minima of known order": "由已知级次的极小值得到",
    "from the phase-locked fit": "由锁相拟合得到",
    # -- statuses of a time group ----------------------------------------------------------------------------------
    "H measured": "已测得 H",
    "seen on one side of the beam only": "只在光束一侧看到",
    "the two sides of the beam disagree": "光束两侧的结果不一致",
    "moving minima of unknown order": "有移动的极小值，但级次无法确定",
    "below the range: the first minimum is above the top of the detector's q_z range":
        "低于测量范围：第一个极小值还在探测器 q_z 范围的上端之外",
    "no moving minimum": "没有移动的极小值",
    # -- the background run ----------------------------------------------------------------------------------------
    "Measuring the island height from the q_z fringes in the background…": "正在后台由 q_z 干涉条纹测量岛高度…",
    "Island height from the q_z fringes: {status} ({folder})": "由 q_z 干涉条纹得到的岛高度：{status}（{folder}）",
    "Island height from the q_z fringes: {error}": "由 q_z 干涉条纹得到的岛高度：{error}",
    "The island height from the q_z fringes could not be measured (exit code {code}): {reason}":
        "无法由 q_z 干涉条纹测量岛高度（退出码 {code}）：{reason}",
    "The fringe analysis is not in this GIMaP folder: {folder} is missing.":
        "这个 GIMaP 文件夹中没有干涉条纹分析：缺少 {folder}。",
    # -- Reliability -----------------------------------------------------------------------------------------------
    "<b>Island height H from the q_z fringes.</b>": "<b>由 q_z 干涉条纹得到的岛高度 H。</b>",
    "H is the vertical size of the islands or of the layer, from the positions of the minima of the vertical cuts at "
    "low q_y that move to lower q_z while the metal grows (H = 2π n / Q, Q = kz(αi) + kz(αf) inside the metal). It "
    "equals the film thickness only after a continuous film has formed. It can be measured only while a minimum lies "
    "inside the detector's q_z range ({low}–{high} nm⁻¹ inside the metal); before the first minimum enters it, H is "
    "below {bound} nm.":
        "H 是岛或金属层在竖直方向上的尺寸，由低 q_y 处竖直切线中极小值的位置得到；金属生长时，这些极小值移向更低的 "
        "q_z（H = 2π n / Q，Q = kz(αi) + kz(αf)，取金属内部的值）。只有形成连续膜之后，H 才等于膜厚。只有当极小值"
        "落在探测器的 q_z 范围内（金属内部 {low}–{high} nm⁻¹）时才能测量；第一个极小值进入该范围之前，H 小于 "
        "{bound} nm。",
    "H is measured in {measured} of {groups} time groups ({first}–{last} s; {one_side} of them on one side only).":
        "在 {groups} 个时间组中有 {measured} 个测得 H（{first}–{last} s；其中 {one_side} 个只来自光束一侧）。",
    "No minimum moves through the range like the fringe of a growing layer: H is not measured.":
        "没有极小值像生长层的干涉条纹那样在范围内移动：未测得 H。",
    "{count} time groups show moving minima whose order cannot be told: H is not given there.":
        "{count} 个时间组中有移动的极小值，但无法确定其级次：这些时间组不给出 H。",
    "Refraction-corrected with αc = {alpha_c}° ± {error}° ({source}).": "已做折射校正，αc = {alpha_c}° ± {error}°（{source}）。",
    "A straight line through the measured groups: dH/dt = {v} ± {error} nm/s; it reaches H = 0 at t = {t0} s (the "
    "deposition start the fringes imply, for information).":
        "对测得的时间组作直线拟合：dH/dt = {v} ± {error} nm/s；直线在 t = {t0} s 处 H = 0（干涉条纹所指示的沉积开始"
        "时刻，仅供参考）。",
    "The intervals hold the minima's position errors, the error of αc and a systematic error that grows when less than "
    "about one fringe period is in the range; the slab phase assumed (minima at H Q = 2π n) may leave a constant "
    "offset of about 1 nm.":
        "区间包含极小值位置的误差、αc 的误差，以及范围内不足约一个条纹周期时增大的系统误差；所假定的平板相位（极小值"
        "位于 H Q = 2π n）可能留下约 1 nm 的固定偏差。",
    "fitted to the fringes of the accepted time groups": "由采纳的时间组的干涉条纹拟合得到",
    "fitted so that coexisting orders give the same H": "拟合使同时出现的各级次给出相同的 H",
    "given by the user": "由用户给定",
    "bulk value of {metal}": "{metal} 的块体值",
}

__all__ = ["SPUTTER_FRINGES_ZH"]
