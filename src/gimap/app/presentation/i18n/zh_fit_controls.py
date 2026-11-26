"""Concise fitting feedback and complete-solution selection."""
FIT_CONTROLS_ZH = {
    "did not improve on the start values": "未能比起始值更好（保留起始值）",
    "The fit did not improve on the start values: they are kept.": "拟合未能比起始值更好：保留起始值。",
    "Only {points} points in the fitting range for {free} free values: a fit needs at least {minimum}. Fix some values or widen the fitting range.":
        "拟合范围内只有 {points} 个点，却有 {free} 个自由量：拟合至少需要 {minimum} 个点。请固定一些量或放宽拟合范围。",
    "No point of the curve has an intensity above zero: there is nothing a model can be fitted to.":
        "曲线中没有强度大于零的点：没有可供模型拟合的数据。",
    "Too few points to fit (at least 3 in the fitting range)": "点太少，无法拟合（拟合范围内至少要有 3 个点）",
    "{count} points with a zero, negative or missing σ were left out: they cannot be weighted. The other {points} points are fitted with the σ of the file.":
        "有 {count} 个点的 σ 为零、为负或缺失，已被略去：它们无法加权。其余 {points} 个点按文件中的 σ 拟合。",
    "No upper limit": "不设上限",
    "fit1d cannot preserve {parameter} in its native model. Use Refine or Search for this constraint.": "fit1d 的原生模型无法保留 {parameter} 约束。请使用精修或范围搜索。",
    "fit1d supports a fixed background, not a background range. Use Refine or Search for this constraint.": "fit1d 支持固定背景，但不支持背景范围约束。请使用精修或范围搜索。",
    "SAXS has no structure factor: choose GISAXS to constrain D or its disorder.": "SAXS 没有结构因子：请选 GISAXS 以约束 D 或间距无序度。",
    "SAXS has no low-q term: choose GISAXS to constrain its exponent.": "SAXS 没有低 q 项：请选 GISAXS 以约束其指数。",
    "fit1d applies the structure factor to every population: enable it in Model or choose SAXS.": "fit1d 对所有粒子群应用结构因子：请在模型中启用它，或选择 SAXS。",
    "fit1d low-q exponent must be within 0.5–12.": "fit1d 的低 q 指数必须在 0.5–12 范围内。",
    "No candidate preserves Model's constraints. Use Refine or Search with the current model.": "没有候选解能保留模型的约束。请使用当前模型进行精修或范围搜索。",
    "Fit details…": "拟合详情…",
    "Fit details": "拟合详情",
    "Best solution": "最佳解",
    "Equivalent solution {index}": "等价解 {index}",
    "Show components": "显示组分",
    "Choose a complete solution: all parameters and curves change together.": "选择一整套解：所有参数和曲线一起切换。",
    "Prediction finished: χ²/point {chi2}.": "预测完成：χ²/点 {chi2}。",
    "Resolve default ranges during composition search": "搜索组成时处理默认范围边界",
    "Only automatic composition searches may extend the predictor's default ranges. Model's fixed values and min–max ranges are always kept. See Fit details for boundary diagnostics.": "只有自动搜索组成时，才可扩展预测器的默认范围。模型中的固定值和上下限始终保留。边界诊断见拟合详情。",
    "The computer ran out of memory. Close other programs and run it again.":
        "电脑的内存已用完。请关闭其他程序后再运行一次。",
    "Model has fixed values or custom ranges. Choose The particles in Model to keep them.": "模型有固定值或自定义范围。请选择“模型中的粒子”以保留这些约束。",
}
