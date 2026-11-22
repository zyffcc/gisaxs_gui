"""Fitting's fit of several curves together, the substrate and level of Model (four-term DWBA), the posterior check by
MCMC in the Uncertainty and Report window, and Analyze ▸ Send Both Cuts to Fitting."""

JOINT_FIT_ZH = {
    # menus
    "Send Bot&h Cuts to Fitting": "把两条切线都发送到拟合(&H)",
    "The horizontal and the vertical cut of this GISAXS frame, fitted together in Fitting (one χ²): the vertical cut "
    "sees the particles' height":
        "把这一 GISAXS 帧的水平切线和垂直切线一起送到拟合中联合拟合（一个 χ²）：垂直切线能看到颗粒的高度",
    "Fit Several Curves To&gether…": "多条曲线联合拟合(&G)…",
    "One χ² over several curves — a horizontal and a vertical cut of one frame (Analyze ▸ Send Both Cuts to Fitting) "
    "or curve files — with values shared or each curve's own":
        "多条曲线共用一个 χ²：同一帧的水平切线和垂直切线（分析 ▸ 把两条切线都发送到拟合），或曲线文件；参数可共享，也可各曲线自有",
    "Both cuts are in Fitting ▸ Fit Several Curves Together — the curve files are in {folder}":
        "两条切线已在 拟合 ▸ 多条曲线联合拟合 中——曲线文件在 {folder}",
    "Could not send both cuts: {error}": "无法发送两条切线：{error}",
    "Both cuts are those of a GISAXS frame: this frame has no horizontal and vertical cut.":
        "两条切线指 GISAXS 帧的水平切线和垂直切线：这一帧没有这两条切线。",
    "The vertical cut of this frame is empty: set it in the Cuts step first.": "这一帧的垂直切线是空的：请先在切线步骤中设置。",
    # the dialog
    "Fit Several Curves Together": "多条曲线联合拟合",
    "Curves": "曲线",
    "Kind": "类型",
    "Points": "点数",
    "Geometry": "几何",
    "Add Files…": "添加文件…",
    "Add Curve of the Page": "添加本页曲线",
    "Remove": "移除",
    "Curve files: a horizontal cut, a vertical cut (Analyze writes its geometry in the file), or any other curve":
        "曲线文件：水平切线、垂直切线（分析会把它的几何写进文件）或其他曲线",
    "The curve open in Single analysis, with the angles of its Model": "单曲线分析中打开的曲线，带其模型中的角度",
    "Horizontal cut": "水平切线",
    "Vertical cut": "垂直切线",
    "Other curve (q_z = 0)": "其他曲线（q_z = 0）",
    "Horizontal cut: I(|q_y|) at one exit angle. Vertical cut: I(q_z) at one in-plane q, in the four-term DWBA (needs "
    "the substrate). Other curve: at q_z = 0.":
        "水平切线：一个出射角下的 I(|q_y|)。垂直切线：一个面内 q 下的 I(q_z)，用四项 DWBA（需要基底）。其他曲线：在 q_z = 0。",
    "Incidence angle of the frame (°)": "这一帧的入射角（°）",
    "Exit angle of the horizontal cut's band (°); 0 with αi = 0: q_z = 0": "水平切线所在条带的出射角（°）；与 αi = 0 一起为 0：q_z = 0",
    "Wavelength (nm)": "波长（nm）",
    "In-plane q of the vertical cut's column band (nm⁻¹), on the sample horizon": "垂直切线所在列条带在样品地平线处的面内 q（nm⁻¹）",
    "The film's critical angle when Analyze refracted the cut's q_z (°); 0: vacuum q_z":
        "分析对切线 q_z 做了折射修正时薄膜的临界角（°）；0：真空中的 q_z",
    "αc of q_z (°)": "q_z 的 αc（°）",
    "Substrate": "基底",
    "The substrate under the particles (all curves): a vertical cut needs it; a horizontal cut with both angles is then "
    "in the four-term DWBA as well":
        "颗粒下面的基底（所有曲线共用）：垂直切线需要它；给了两个角度的水平切线也随之用四项 DWBA",
    "Parameters": "参数",
    "Shared": "共享",
    "Scales (Int, background, tail A) are each curve's own, solved for each curve. A height of upright cylinders is "
    "shared: only a vertical cut sees it.":
        "标度（Int、背景、束尾 A）由各曲线自有，并对各曲线分别求解。竖直圆柱的高度是共享的：只有垂直切线能看到它。",
    "Refine: from the values in Model": "精修：从模型中的值出发",
    "Fit Together": "联合拟合",
    "Show": "显示",
    "Use in Model": "用到模型",
    "The fitted values of the first horizontal cut (shared ones included) into the Model of Single analysis":
        "把第一条水平切线的拟合值（含共享值）放进单曲线分析的模型",
    "Save Record…": "保存记录…",
    "Save Record": "保存记录",
    "Every curve, its model and errors, the shared values, χ² and the covariance (JSON)":
        "每条曲线及其模型和误差、共享值、χ² 与协方差（JSON）",
    "Add Curves": "添加曲线",
    "Single analysis has no curve open.": "单曲线分析中没有打开曲线。",
    "{name} was not added: {reason}": "未添加 {name}：{reason}",
    "αi {ai}°, αf {af}°, λ {wl} nm": "αi {ai}°，αf {af}°，λ {wl} nm",
    "αi {ai}°, q∥ {qp} nm⁻¹, λ {wl} nm": "αi {ai}°，q∥ {qp} nm⁻¹，λ {wl} nm",
    "Add the curves, then Fit Together: each curve is shown here with its model.": "先添加曲线，再点联合拟合：每条曲线和它的模型显示在这里。",
    "Fitting the curves together …": "正在联合拟合这些曲线…",
    "The fit failed: {reason}": "拟合失败：{reason}",
    "The curves were fitted together.": "曲线已联合拟合。",
    "χ²ᵣ {chi2} over {points} points of {curves} curves and {free} free values · {outcome} · {evaluations} evaluations, "
    "{seconds:.1f} s":
        "χ²ᵣ {chi2}：{curves} 条曲线的 {points} 个点，{free} 个自由值 · {outcome} · {evaluations} 次计算，{seconds:.1f} s",
    "stopped": "已停止",
    "converged": "已收敛",
    "not converged": "未收敛",
    "{name}: Σr²/N = {value} ({points} points, {weighting})": "{name}：Σr²/N = {value}（{points} 个点，{weighting}）",
    "The curves are weighted differently (some by σ, some by ln I): their χ² are not comparable.":
        "各曲线的权重方式不同（有的按 σ，有的按 ln I）：它们的 χ² 不可比。",
    "At a bound of its range: {names}": "位于范围边界：{names}",
    "The values of the joint fit are in Model.": "联合拟合的值已放进模型。",
    "Could not save {name}: {reason}": "无法保存 {name}：{reason}",
    "Saved {name}.": "已保存 {name}。",
    "{name}: a vertical cut needs the substrate (choose a material or type δ and β)":
        "{name}：垂直切线需要基底（选一种材料，或输入 δ 和 β）",
    "{name}: a vertical cut needs αi and λ": "{name}：垂直切线需要 αi 和 λ",
    "{name}: no point of the vertical cut lies above the sample horizon": "{name}：垂直切线没有点位于样品地平线以上",
    "{name}: fewer than 3 points to fit": "{name}：可拟合的点少于 3 个",
    "Add at least one curve": "至少添加一条曲线",
    "{name}: a vertical cut at q∥ = {q} nm⁻¹ lies on or near the specular rod, which the particles' model leaves out: "
    "take the vertical cut beside it in Analyze (Cuts step), e.g. at the structure peak.":
        "{name}：q∥ = {q} nm⁻¹ 处的垂直切线在镜面反射杆上或其附近，而颗粒模型不含镜面反射：请在分析（切线步骤）中把垂直切线放到它旁边，"
        "例如结构峰处。",
    "The curves have no more points than free values": "这些曲线的点数不多于自由值的个数",
    "Scales (Int, background, tail A) are each curve's own: they cannot be shared": "标度（Int、背景、束尾 A）由各曲线自有，不能共享",
    "A vertical cut takes spheres and upright cylinders (the four-term DWBA of particles on the substrate); this model "
    "has another family":
        "垂直切线只接受球和竖直圆柱（基底上颗粒的四项 DWBA）；这个模型有别的粒子类型",
    "Upright cylinders in a vertical cut need their height h (and σh/h)": "垂直切线中的竖直圆柱需要高度 h（和 σh/h）",
    # Model: substrate and level
    "Substrate:": "基底：",
    "Unknown": "未知",
    "Glass (plate)": "玻璃（平板玻璃）",
    "Typed δ, β": "输入 δ、β",
    "The substrate under the particles. Unknown: the spheres at q_z = 0, at the Born term or with two paths and a free "
    "r, as αi, αf and |r| say. A material of the table or δ and β typed: with αi and αf the spheres are in the "
    "four-term DWBA with the substrate's Fresnel coefficients (|r| and arg r are then not used). The table holds the "
    "bare materials (Henke's tables, 4.9–30 keV); an oxide, a polymer film or roughness are not part of it.":
        "颗粒下面的基底。未知：按 αi、αf 和 |r| 的设置，球在 q_z = 0、在 Born 项或用两条路径加自由的 r。选表中的材料或输入 δ 和 β："
        "给出 αi 和 αf 时，球用基底菲涅耳系数的四项 DWBA（此时不用 |r| 和 arg r）。表中是裸材料（Henke 表，4.9–30 keV），"
        "不含氧化层、聚合物膜或粗糙度。",
    "δ of the substrate at the cut's wavelength (n = 1 − δ + iβ)": "切线波长下基底的 δ（n = 1 − δ + iβ）",
    "β of the substrate at the cut's wavelength": "切线波长下基底的 β",
    "Level: {level}": "层次：{level}",
    "Level: {level} on {substrate} (δ = {delta}, β = {beta} at λ = {wavelength} nm)":
        "层次：{level}，基底 {substrate}（λ = {wavelength} nm 时 δ = {delta}，β = {beta}）",
    "Level: {level}. The substrate is used once αi and αf are given.": "层次：{level}。给出 αi 和 αf 后才用基底。",
    "The substrate cannot be used: {reason}": "无法使用基底：{reason}",
    "Type δ (above 0) and β of the substrate at the cut's wavelength.": "请输入切线波长下基底的 δ（大于 0）和 β。",
    "Effective interface (q_z = 0)": "有效界面（q_z = 0）",
    "Born term at the cut's q_z": "切线 q_z 处的 Born 项",
    "Two paths, free r": "两条路径，自由的 r",
    "Four-term DWBA, substrate known": "四项 DWBA（已知基底）",
    "αi = αf = 0: spheres at q_z = 0 (effective interface approximation, no substrate needed). With αi, αf and λ of the "
    "cut: direct and exit-reflected path, with a free effective reflection coefficient r = |r|·e^(i·arg r); free |r| "
    "and arg r to fit it. With a known substrate as well: the four-term DWBA. Upright cylinders are not affected.":
        "αi = αf = 0：球在 q_z = 0（有效界面近似，不需要基底）。给出切线的 αi、αf 和 λ：直射路径和出射反射路径，"
        "有效反射系数 r = |r|·e^(i·arg r) 可自由拟合；放开 |r| 和 arg r 即可拟合它。再给出已知基底：四项 DWBA。竖直圆柱不受影响。",
    # the posterior check
    "Posterior check (MCMC)": "后验检验（MCMC）",
    "Steps": "步数",
    "Walkers": "行走者",
    "Seed": "种子",
    "Run Check": "运行检验",
    "Steps of every walker; the first half is discarded (burn-in)": "每个行走者的步数；前一半丢弃（预烧）",
    "Walkers of the ensemble (even); 0: four per value, at least 16": "系综中的行走者数（偶数）；0：每个值四个，至少 16 个",
    "auto": "自动",
    "Seed of the random numbers (the same seed, the same run)": "随机数种子（同一种子，同一次运行）",
    "Value": "值",
    "Checked": "被检验值",
    "Reference 68 %": "参照 68 %",
    "MCMC median": "MCMC 中位数",
    "τ (steps)": "τ（步）",
    "Trace of": "轨迹：",
    "Save Check…": "保存检验…",
    "Save Check": "保存检验",
    "The run's settings, intervals and diagnostics beside the values it checks (JSON)": "运行设置、区间和诊断，以及被检验的值（JSON）",
    "Nothing to check: put a solution of Predict parameters in Model, or fit with Refine or Search first.":
        "没有可检验的：先把预测参数的一个解放进模型，或先用精修或搜索拟合。",
    "Checks the solution of Predict parameters in Model: its posterior (the model error and the scales integrated out, "
    "the network's prior), against the median and intervals Predict gave.":
        "检验模型中预测参数的解：它的后验（对模型误差和标度积分，用网络的先验），与预测给出的中位数和区间对照。",
    "Checks the last Refine or Search: its likelihood (Σr² scaled by its χ²ᵣ, the scales solved), against its value "
    "± 1σ (± 1.96σ for 95 %).":
        "检验最近一次精修或搜索：它的似然（Σr² 按其 χ²ᵣ 缩放，标度求解），与其值 ± 1σ（95 % 用 ± 1.96σ）对照。",
    "A check only: the values and intervals in Results are not changed.": "这只是检验：结果中的值和区间不变。",
    "The model changed since the run: the check is of the model it was started for.": "运行后模型已改变：检验针对的是启动时的模型。",
    "Running {steps} steps of {walkers} walkers …": "正在运行 {walkers} 个行走者的 {steps} 步…",
    "{steps} steps of {walkers} walkers in {seconds:.1f} s, acceptance {acceptance:.2f}.":
        "{walkers} 个行走者 {steps} 步，用时 {seconds:.1f} s，接受率 {acceptance:.2f}。",
    "Stopped after {steps} steps: the steps done are shown.": "在第 {steps} 步停止：显示已完成的步。",
    "⚠ The run's 68 % interval does not overlap the one checked: {names}.": "⚠ 本次运行的 68 % 区间与被检验的区间不重叠：{names}。",
    "Converged: {kept} steps kept (the first half discarded) ≥ {factor} × the longest autocorrelation time ({tau} "
    "steps), R̂ ≤ {rhat}; about {samples} independent samples.":
        "已收敛：保留 {kept} 步（丢弃前一半）≥ {factor} × 最长自相关时间（{tau} 步），R̂ ≤ {rhat}；约 {samples} 个独立样本。",
    "Not converged: {kept} steps kept, the longest autocorrelation time {tau} steps (needs ≥ {factor} × of it), R̂ up "
    "to {rhat} (needs < {limit}). Run longer before reading the intervals.":
        "未收敛：保留 {kept} 步，最长自相关时间 {tau} 步（需要 ≥ 它的 {factor} 倍），R̂ 最高 {rhat}（需要 < {limit}）。"
        "请先运行更长时间再读区间。",
    "The check failed: {reason}": "检验失败：{reason}",
    "Run the check: the walkers' values step by step are drawn here.": "运行检验：这里逐步画出行走者的值。",
    "median of the walkers": "行走者中位数",
    "16 % and 84 % of the walkers": "行走者的 16 % 和 84 %",
    "walker {number}": "行走者 {number}",
    "walkers 1–{count}": "行走者 1–{count}",
    "step": "步",
    "The fit has no free value besides the scales: nothing to sample": "除标度外这个拟合没有自由值：无可抽样",
    "The run was stopped before it had steps to report": "运行在有可报告的步之前就停止了",
}

__all__ = ["JOINT_FIT_ZH"]
