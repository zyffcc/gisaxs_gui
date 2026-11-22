"""Chinese text for Predict parameters results in a GISAXS automatic analysis."""

GISAXS_POSTERIOR_ZH = {
    "Predict parameters on {curve} compared the solutions by BIC weight after evidence sampling failed.":
        "证据抽样失败，预测参数对 {curve} 的解按 BIC 权重作了比较。",
    "{n} solutions have a BIC weight of at least {limit} ({solutions}).":
        "有 {n} 个解的 BIC 权重至少为 {limit}（{solutions}）。",
    "Evidence sampling failed. The solutions are ranked by normalised BIC weights.":
        "证据抽样失败。各解按归一化的 BIC 权重排序。",
    "{n} solutions have BIC weights ≥ {limit}. Select a row to compare their fits.":
        "{n} 个解的 BIC 权重 ≥ {limit}。选择表格行来比较拟合曲线。",
    "{model} R {radius} nm (BIC weight {p}, χ²ᵣ {chi2})": "{model} R {radius} nm（BIC 权重 {p}，χ²ᵣ {chi2}）",
    "CSV: every solution with its BIC weight, parameters (nm), intervals and χ²ᵣ.":
        "CSV：每个解的 BIC 权重、参数（nm）、区间和 χ²ᵣ。",
    "Posterior median {value} (68 % {low}–{high})": "后验中位数 {value}（68 % 区间 {low}–{high}）",
    "{points} points, |qy| {low}–{high} Å⁻¹ ({low_nm}–{high_nm} nm⁻¹). {screen}. {algorithm}. "
    "BIC fallback intervals use least-squares errors.":
        "{points} 个点，|qy| 为 {low}–{high} Å⁻¹（{low_nm}–{high_nm} nm⁻¹）。{screen}。{algorithm}。"
        "回退到 BIC 后，报告区间采用最小二乘误差。",
}
