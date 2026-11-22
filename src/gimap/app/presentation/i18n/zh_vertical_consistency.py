"""Height range check in the joint-fit progress, summary and report."""

VERTICAL_CONSISTENCY_ZH = {
    "The fitted height assumes a dilute particle layer and no correlations between height and radius.":
        "拟合高度以颗粒层稀疏且颗粒高度与半径不相关为前提。",
    "Checking the height on a second vertical range …": "正在用另一个垂直切线范围核对高度……",
    "Both vertical ranges contain the same usable points. The height comparison is unavailable.":
        "两个垂直切线范围包含相同的可用点，无法比较不同范围的高度。",
    "The second vertical range could not be fitted. The height comparison is unavailable.":
        "另一个垂直切线范围未能完成拟合，无法比较高度。",
    "The height comparison was stopped.": "高度比较已停止。",
    "{height} is {selected} nm at the selected cutoff and {alternate} nm at a cutoff of {multiple} αc.":
        "在选定的出射角下限下，{height} 为 {selected} nm；下限为 {multiple} αc 时，{height} 为 {alternate} nm。",
    "{height} depends on the vertical fit range. The change exceeds twice the combined height uncertainty "
    "or 2% of the selected height.":
        "{height} 随垂直切线拟合范围而变。差值超过两次拟合的高度合成标准不确定度的两倍，或超过选定范围拟合高度的 2%。",
}
