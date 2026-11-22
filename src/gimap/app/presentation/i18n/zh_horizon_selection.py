"""Ring orientation selection and sensitivity notes."""

HORIZON_SELECTION_ZH = {
    "Excluded |χ| {low}–{high}° next to the exit-angle cut at {fraction} of the median net ring intensity.":
        "出射角截止处还排除了 |χ| {low}–{high}° 的分箱，阈值为环净强度中位数的 {fraction}。",
    "The pole fit is undetermined. The systematic range uses measured ±χ partners only and leaves the missing wedge empty.":
        "极图拟合未能确定取向。系统范围仅使用 ±χ 两侧可测的分箱，缺失楔留空。",
    "Binned means sample the pole density. Hermans f and the pole fit use (I − b) sin χ. "
    "The systematic range varies the exit-angle cut, horizon intensity threshold, background and missing-wedge fill.":
        "分箱均值对应极密度。Hermans f 和极图拟合都使用 (I − b) sin χ。"
        "系统范围考察出射角、地平线强度阈值、背景和缺失楔补全方法的影响。",
    "The systematic range for f varies the background, exit-angle cut, horizon intensity threshold and wedge fill.":
        "f 的系统范围考察背景、出射角、地平线强度阈值和缺失楔补全方法的影响。",
}
