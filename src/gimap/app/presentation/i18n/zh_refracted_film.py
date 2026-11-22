"""Film optical constants and the GIWAXS intensity correction."""

REFRACTED_FILM_ZH = {
    "Divide each pixel by the film factor relative to the specular exit angle.":
        "将每个像素的强度除以薄膜因子，因子以镜面出射方向为基准。",
    "Film model": "薄膜模型",
    "Enter δ and β": "输入 δ 和 β",
    "Film CIF": "薄膜 CIF",
    "Thick film limit": "厚膜极限",
    "Choose CIF…": "选择 CIF…",
    "Enter value": "输入数值",
    "Enter both optical constants to apply the correction.": "输入两个光学常数后应用校正。",
    "Enter both film optical constants, δ and β.": "请输入薄膜的两个光学常数 δ 和 β。",
    "Film δ must be finite and film β must be finite and nonnegative.": "薄膜的 δ 必须为有限值，β 必须为非负有限值。",
    "Enter a positive film thickness or select the thick film limit.": "请输入正的膜厚或选择厚膜极限。",
    "The film CIF does not provide optical constants at this wavelength.": "无法从薄膜的 CIF 获取该波长下的光学常数。",
    "Fitting film thickness requires one film and incidence angle across all rings.":
        "拟合膜厚时，各衍射环必须采用相同的薄膜光学常数和入射角。",
}
