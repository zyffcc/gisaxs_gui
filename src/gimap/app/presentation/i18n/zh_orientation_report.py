"""GIWAXS ring orientation fit results and uncertainty estimates."""

ORIENTATION_REPORT_ZH = {
    "Ring q ≈ {q} Å⁻¹: pole fit not determined ({reasons}). Measured f = {f}": "环 q ≈ {q} Å⁻¹：极图拟合未能确定取向（{reasons}）。实测 f = {f}",
    "no clear preferred orientation; the strongest measured bin is at |χ| ≈ {chi}°":
        "择优取向不明显，最强的实测分箱位于 |χ| ≈ {chi}°",
    "no single preferred orientation: the maximum at |χ| ≈ {chi}° and the measured moment disagree":
        "没有单一的择优取向：|χ| ≈ {chi}° 处的极大值与实测二阶矩指示的方向不一致",
    "fit parameters are not identifiable": "拟合参数不能独立确定",
    "Measured f = {value}": "实测 f = {value}",
    " [{low}, {high}] (systematic)": " [{low}, {high}]（系统范围）",
    "Fitted Hermans f = {f} (missing wedge filled, oriented part {f_oriented})":
        "拟合 Hermans f = {f}（补全缺失楔，取向部分为 {f_oriented}）",
    "stat": "统计",
    "Measured N(χ)": "实测 N(χ)",
    "The total sin χ weighted intensity after background subtraction is nonpositive.": "扣除背景后，按 sin χ 加权的总强度不大于零。",
    "The missing wedge covers |χ| < {angle}°. The measured f excludes these orientations.": "缺失楔覆盖 |χ| < {angle}°。实测 f 不含这些取向。",
    "The measured f excludes the missing wedge (red dashed near qz), |χ| {where} Å⁻¹.": "实测 f 不含缺失楔（qz 附近的红色虚线），|χ| {where} Å⁻¹。",
    "Pole figure at q = {q} Å⁻¹": "q = {q} Å⁻¹ 处的极图",
    "At q = {q} Å⁻¹ the pole fit is not determined ({bounds}).": "q = {q} Å⁻¹ 处的极图拟合未能确定取向（{bounds}）。",
    "Exclude pixels with αf < {angle}° ({source}). Polarisation: {factor}.": "排除 αf < {angle}° 的像素（{source}）。偏振：{factor}。",
    "statistical ± and systematic range for f": "统计误差 ± 及 f 的系统变化范围",
    "Binned means sample the pole density. Hermans f and the pole fit use (I − b) sin χ. The systematic range varies the exit threshold, background and missing-wedge fill.":
        "分箱均值对应极密度。Hermans f 和极图拟合都使用 (I − b) sin χ。系统范围反映出射角阈值、背景和缺失楔补全方法的影响。",

    "Gaussian mosaic + isotropic": "高斯镶嵌分布 + 各向同性",
    "Pole fit not determined ({reasons}).": "极图拟合未能确定取向（{reasons}）。",
    "fit did not converge": "拟合未收敛",
    "reduced χ² exceeds 5": "约化 χ² 大于 5",
    "FWHM exceeds 100°": "半高宽大于 100°",
    "isotropic fraction at a bound": "各向同性分数达到边界",
    "width at a bound": "宽度达到边界",
    "direction at a bound": "方向达到边界",
    "Pole f = {f}, oriented fraction = {fraction}. 1 − φ is conditional on the background b.":
        "极图 f = {f}，取向分数 = {fraction}。1 − φ 取决于背景 b。",
    "Background {value} counts/pixel ({source}). 1 − φ is conditional on the background b.":
        "背景 {value} 计数/像素（{source}）。1 − φ 取决于背景 b。",
    "Excluded αf < {angle}°. Polarisation {factor}.": "排除 αf < {angle}° 的像素。偏振因子 {factor}。",
    "Measured f systematic range [{low}, {high}]": "实测 f 的系统变化范围 [{low}, {high}]",
    "Some systematic estimates are unavailable.": "部分系统变化估计无法计算。",
    "Inspect the profile and background in Phases ▸ Pole Figure.": "请在“物相 ▸ 极图”中检查剖面和背景。",
}
