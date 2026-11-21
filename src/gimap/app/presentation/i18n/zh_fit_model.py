"""Fitting's Model step (one line per parameter, links and constraints, the model library), the Uncertainty and
Report window and the beam tail's ranges and flags."""
FIT_MODEL_ZH = {
    # the parameter lines
    "min": "下限",
    "max": "上限",
    "Name in links: {name}": "链接中的名称：{name}",
    "Link this value to another parameter or to an expression of others (e.g. 2*R_1); it then follows them":
        "把这个值链接到另一个参数或其他参数的表达式（如 2*R_1）；之后它随之变化",
    "{name} = an expression of other parameters, e.g. 2*R_1, sigma_R_1, R_1 + t_shell_1.\n"
    "Names: {names}. Functions: sqrt, exp, log, sin, cos, abs, min, max; pi.":
        "{name} = 其他参数的表达式，例如 2*R_1、sigma_R_1、R_1 + t_shell_1。\n"
        "名称：{names}。函数：sqrt、exp、log、sin、cos、abs、min、max；pi。",
    "Linked: = {expression} (follows the other values; not fitted)": "已链接：= {expression}（随其他值变化，不参与拟合）",
    "Same as": "等于",
    "Expression…": "表达式…",
    "Unlink": "取消链接",
    "Link": "链接",
    "Could not link: {reason}": "无法链接：{reason}",
    "Could not add the constraint: {reason}": "无法添加约束：{reason}",
    "Links or constraints that no longer hold were removed: {items}": "已删除不再成立的链接或约束：{items}",
    # the model library
    "Core–shell sphere": "核壳球",
    "For Refine and Search: Predict parameters does not propose this family": "用于精修和搜索：预测参数不会给出这一类粒子",
    "1D paracrystal": "一维次晶",
    "2D hexagonal paracrystal": "二维六方次晶",
    "Hard spheres (Percus–Yevick)": "硬球（Percus–Yevick）",
    "Interference": "干涉函数",
    "Interference between neighbours (a structure factor): choose its kind below the parameters":
        "近邻之间的干涉（结构因子）：在参数下方选择类型",
    "1D paracrystal: neighbours along a line (Hosemann). 2D hexagonal paracrystal: a disordered hexagonal lattice in the "
    "plane, averaged over its orientation. Hard spheres: the Percus–Yevick liquid of radius R_HS and volume fraction φ. "
    "The last two are for Refine and Search only.":
        "一维次晶：近邻沿一条线排列（Hosemann）。二维六方次晶：平面内无序的六方格子，对取向取平均。硬球：半径 R_HS、"
        "体积分数 φ 的 Percus–Yevick 液体。后两种只用于精修和搜索。",
    "Thickness of the shell: the outer radius is R + t": "壳层厚度：外半径为 R + t",
    "Scattering contrast of the core relative to that of the shell (0: a hollow shell, 1: a homogeneous sphere of radius "
    "R + t)": "核的散射衬度与壳层衬度之比（0：空心壳；1：半径为 R + t 的均匀球）",
    "Hard-sphere radius: half the closest distance between the centres (Percus–Yevick)":
        "硬球半径：中心间最近距离的一半（Percus–Yevick）",
    "Volume fraction of the hard spheres (Percus–Yevick)": "硬球的体积分数（Percus–Yevick）",
    # links and constraints
    "Links and constraints": "链接与约束",
    "A linked value follows others (its “=” button); a constraint keeps a fitted value on one side of an expression. "
    "Both are kept in the model, the saved model and the fit's record.":
        "链接的值随其他值变化（用该行的“=”按钮）；约束让拟合值保持在某个表达式的一侧。两者都保存在模型、保存的模型文件和拟合记录中。",
    "Particles do not overlap (D ≥ 2R, R_HS ≥ R)": "粒子不重叠（D ≥ 2R，R_HS ≥ R）",
    "While both are fitted: the distance D of a paracrystal is at least the particles' diameter 2R (2(R + t) for "
    "core–shell spheres), and the hard-sphere radius R_HS at least R (+ t)":
        "两者都拟合时：次晶的间距 D 不小于粒子直径 2R（核壳球为 2(R + t)），硬球半径 R_HS 不小于 R（+ t）",
    "Remove this constraint": "删除这条约束",
    "Add Constraint…": "添加约束…",
    "Add Constraint": "添加约束",
    "A fitted value ≥ or ≤ an expression of others, e.g. D_2 >= 2*R_2 + 3": "拟合值 ≥ 或 ≤ 其他参数的表达式，例如 D_2 >= 2*R_2 + 3",
    "A constraint keeps a fitted value on one side of an expression of others, e.g.\n"
    "D_2 >= 2*R_2 + 3   or   R_2 <= 0.5*R_1\nNames: {names}.":
        "约束让拟合值保持在其他参数表达式的一侧，例如\nD_2 >= 2*R_2 + 3   或   R_2 <= 0.5*R_1\n名称：{names}。",
    # the Uncertainty and Report window
    "Uncertainty and Report…": "不确定度与报告…",
    "Uncertainty and Report": "不确定度与报告",
    "The errors and correlation matrix of the last Refine or Search, the χ² profile of one value, and a fit report to "
    "save": "上一次精修或搜索的误差和相关矩阵、单个参数的 χ² 剖面，以及可保存的拟合报告",
    "Errors and correlations": "误差与相关",
    "χ² profile": "χ² 剖面",
    "Fit report": "拟合报告",
    "Correlation matrix": "相关矩阵",
    "Name in links": "链接名",
    "relative": "相对误差",
    "Run Profile": "计算剖面",
    "Format": "格式",
    "Plain text": "纯文本",
    "Save Report…": "保存报告…",
    "Save Report": "保存报告",
    "Values of the grid: ±3 errors around the fitted value, inside its range": "网格点数：拟合值 ±3 倍误差之内，且在范围内",
    "Fit first (Refine or Search): the errors and the correlations are those of the last fit, of the values it found. "
    "Predict parameters gives its intervals in Fit details.":
        "请先拟合（精修或搜索）：误差和相关性来自上一次拟合得到的值。预测参数的区间见拟合详情。",
    "{chi} {value} over {points} points, {free} free values. Errors: 1σ from the covariance (JᵀJ)⁻¹·s², "
    "s² = Σr²/(N − p); linked values by propagation.":
        "{chi} {value}，{points} 个点，{free} 个自由量。误差：协方差 (JᵀJ)⁻¹·s² 给出的 1σ，s² = Σr²/(N − p)；"
        "链接的值按误差传递计算。",
    "Strongly correlated (|r| > {limit}): {pairs}": "强相关（|r| > {limit}）：{pairs}",
    "No pair of free values is correlated beyond |r| = {limit}.": "没有一对自由量的相关超过 |r| = {limit}。",
    "not defined: the curve does not see this direction": "无定义：曲线对这个方向没有响应",
    "Fitting the other values at each point of the grid …": "正在网格的每个点上重新拟合其他值 …",
    "The profile failed: {reason}": "剖面计算失败：{reason}",
    "68 % interval from Δχ² = 1 (Δχ² in units of the best fit's χ²ᵣ): {low} – {high} (best {best}; ± from the "
    "covariance {error}).": "由 Δχ² = 1 得到的 68 % 区间（Δχ² 以最佳拟合的 χ²ᵣ 为单位）：{low} – {high}（最佳 {best}；"
    "协方差给出的 ± {error}）。",
    "open (Δχ² stays below 1 on the grid)": "开放（网格内 Δχ² 始终低于 1）",
    "A point of the grid fits better than the fit itself (Δχ² below 0): Refine again from there.":
        "网格上有一点比拟合结果更好（Δχ² 小于 0）：请从那里再精修一次。",
    "Stopped: the values done so far are shown.": "已停止：显示已完成的点。",
    "Not determined by the curve: the error is not finite or above 10× the value": "曲线无法确定这个值：误差不是有限数，或超过数值的 10 倍",
    "⚠ Beam tail: {name} {what}. Narrow the tail's range, fix it, or switch it off (A = 0, fixed).":
        "⚠ 束尾：{name}{what}。请收窄束尾范围、固定它，或关闭它（A = 0 并固定）。",
    "is at an edge of its range": "停在范围边界",
    "is not determined": "无法确定",
    "⚠ Not determined by the curve (error not finite or above 10× the value): {names}.":
        "⚠ 曲线无法确定（误差不是有限数或超过数值的 10 倍）：{names}。",
    "Predict parameters knows spheres, random and vertical cylinders with the 1D paracrystal: Model has a core–shell "
    "sphere or another interference function. Use Refine or Search, or set Particles to a choice.":
        "预测参数只认识球、随机取向圆柱和竖直圆柱（带一维次晶）：模型里有核壳球或其他干涉函数。请用精修或搜索，或把粒子设为一个选择。",
    # the beam tail's ranges, series flags
    "The beam tail's ranges were set from this curve: A ≤ {amplitude}, w {low}–{high} nm⁻¹, ν ≤ {exponent} (Ranges "
    "shows them in Model).": "束尾的范围已按这条曲线设置：A ≤ {amplitude}，w {low}–{high} nm⁻¹，ν ≤ {exponent}（勾选“范围”可在模型中看到）。",
    "flagged": "已标记",
    "{count} flagged": "{count} 帧已标记",
    "Flagged (a value not determined, or the beam tail at an edge of its range): frames {frames}":
        "已标记（有值无法确定，或束尾停在范围边界）：第 {frames} 帧",
    "Flagged: {values}": "已标记：{values}",
    "{flagged} of the fitted frames are flagged: a value the curve does not determine, or the beam tail at an edge of "
    "its range.": "拟合的帧中有 {flagged} 帧被标记：有曲线无法确定的值，或束尾停在范围边界。",
}
