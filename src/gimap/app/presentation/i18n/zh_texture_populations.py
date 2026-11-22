"""Independent mosaic widths and phase fractions in the crystal orientation fit."""

TEXTURE_POPULATIONS_ZH = {
    "N(χ) = net intensity × sin χ": "N(χ) = 净强度 × sin χ",
    "{plane} ∥ substrate · mosaic FWHM {width}° · {fraction}% of phase":
        "{plane} ∥ 衬底 · 镶嵌展宽（半高宽）{width}° · 占该物相 {fraction}%",
    "Total oriented {oriented}% · isotropic {isotropic}%":
        "取向晶粒合计 {oriented}% · 各向同性晶粒 {isotropic}%",
    "Film exit transmission |T|² from CIF composition · δ {delta} · β {beta}":
        "按 CIF 组成计算的薄膜出射透射因子 |T|² · δ {delta} · β {beta}",
    "Film exit transmission unavailable": "缺少计算薄膜出射透射因子所需的数据",
    "Only one pole direction constrains this phase. Two distinct fibre axes cannot be established.":
        "该物相只有一个独立的极点方向，无法据此确定两根不同的纤维轴。",
}
