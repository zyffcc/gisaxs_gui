"""Chinese translations: first-user review fixes (curves).

What Fitting says of a curve as it takes it in (``fitting/application/curve_intake.py``; the Fit page's Curve card,
Fit step and Results, the 1D Predict window): the unit of q assumed for a file that states none, the σ estimated for
a file without one, negative and zero intensities."""

REVIEW_CURVES_ZH: dict[str, str] = {
    "The file states no unit of q and was read in {unit}: |q| then reaches {q_max} nm⁻¹ ({q_max_a} Å⁻¹), "
    "beyond what a GISAXS or SAXS cut covers. If its q is in {other}, every length found (R, D) is 10× "
    "too small. Set “q unit of files without one in their header” to {other} (Fit page: Curve ▸ File; 1D Predict: "
    "Parameters…), or state the unit in the file's header, for example “{example}”.":
        "文件没有注明 q 的单位，按 {unit} 读入：|q| 于是达到 {q_max} nm⁻¹（{q_max_a} Å⁻¹），超出了 GISAXS 或 SAXS "
        "切线能覆盖的范围。如果它的 q 其实是 {other}，得到的每个长度（R、D）都小了 10 倍。把“文件头未注明时的 q 单位”"
        "设为 {other}（拟合页：曲线 ▸ 文件；一维预测：参数…），或在文件头里注明单位，例如“{example}”。",
    "The file states no unit of q and was read in {unit}: |q| then reaches only {q_max} nm⁻¹ "
    "({q_max_a} Å⁻¹), less than a GISAXS or SAXS cut covers. If its q is in {other}, every length found "
    "(R, D) is 10× too large. Set “q unit of files without one in their header” to {other} (Fit page: Curve ▸ File; "
    "1D Predict: Parameters…), or state the unit in the file's header, for example “{example}”.":
        "文件没有注明 q 的单位，按 {unit} 读入：|q| 于是最大只有 {q_max} nm⁻¹（{q_max_a} Å⁻¹），小于 GISAXS 或 SAXS "
        "切线覆盖的范围。如果它的 q 其实是 {other}，得到的每个长度（R、D）都大了 10 倍。把“文件头未注明时的 q 单位”"
        "设为 {other}（拟合页：曲线 ▸ 文件；一维预测：参数…），或在文件头里注明单位，例如“{example}”。",
    "No σ in the file: a relative σ of {percent} % is used, estimated from the scatter of ln I about its trend":
        "文件里没有 σ：使用 {percent} % 的相对 σ，由 ln I 相对其趋势的离散估计",
    "⚠ No σ in the file. Predict parameters: a relative σ of {percent} %, estimated from the scatter of "
    "ln I about its trend. Refine and Search: ln I is fitted, every point weighted alike.":
        "⚠ 文件里没有 σ。预测参数：使用 {percent} % 的相对 σ，由 ln I 相对其趋势的离散估计。精修和搜索：拟合 ln I，"
        "每个点权重相同。",
    "No σ in the file: Refine and Search give every point the same relative weight (ln I is fitted).":
        "文件里没有 σ：精修和搜索给每个点相同的相对权重（拟合 ln I）。",
    "No σ in the file: Predict parameters takes a relative σ from the scatter of ln I about its trend "
    "(at least 1 %), the same in the 1D Predict window.":
        "文件里没有 σ：预测参数由 ln I 相对其趋势的离散取一个相对 σ（至少 1 %），一维预测窗口里相同。",
    "{count} of the {points} fitted points have a negative or zero intensity ({negative} negative, "
    "{zero} zero): they are kept and weighted by their σ. The model cannot go below zero, so a "
    "background subtracted too far shows as a poor χ².":
        "参与拟合的 {points} 个点里有 {count} 个强度为负或为零（{negative} 个为负，{zero} 个为零）：它们保留，"
        "按各自的 σ 加权。模型不能低于零，所以背景扣得过多会表现为 χ² 很差。",
    "{count} points with a negative or zero intensity ({negative} negative, {zero} zero) were "
    "left out: without σ the fit is on ln I, which they do not have.":
        "{count} 个强度为负或为零的点（{negative} 个为负，{zero} 个为零）没有参与拟合：没有 σ 时拟合的是 ln I，"
        "这些点没有 ln I。",
    "{count} rows of the file could not be read as numbers and were left out.":
        "文件里有 {count} 行读不成数字，已略去。",
    "Only for Use current cut on a curve Analyze wrote with the pixel counts of its detector columns: "
    "added in quadrature to the counting σ. A curve without σ gets a relative σ estimated from the scatter "
    "of ln I about its trend, as on the Fit page.":
        "只用于“使用当前切线”，并且曲线是分析写出的、带有探测器各列像素数时：与计数 σ 按平方和相加。没有 σ 的曲线"
        "由 ln I 相对其趋势的离散估计一个相对 σ，和拟合页相同。",
}
