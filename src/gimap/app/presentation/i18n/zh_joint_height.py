"""Vertical fit range and interpretation of the fitted height."""

JOINT_HEIGHT_ZH = {
    "Needs attention": "需要留意",
    "Lowest exit angle of the vertical cut": "垂直切线的最低出射角",
    "Multiple of the substrate critical angle at each cut’s wavelength. Set to 0 to keep all valid exit angles.":
        "以各条切线波长下的基底临界角为单位。设为 0 可保留所有有效出射角。",
    "Lowest exit angle of the vertical cut ({name}) = {multiple} αc ({degrees}°).":
        "垂直切线（{name}）的最低出射角为 {multiple} αc（{degrees}°）。",
    "{name} has fewer than 3 usable points above the exit-angle limit. Lower the limit or widen the cut.":
        "{name} 在出射角下限以上的可用点少于 3 个。请降低下限或扩大切线范围。",
    "The lowest exit angle must be a finite, non-negative multiple of αc.":
        "最低出射角必须是 αc 的有限非负倍数。",
    "Each curve has its own intensity scale, background and beam tail. Upright cylinders share their height and height spread.":
        "各条曲线分别拟合强度标度、背景和束尾。竖直圆柱的高度和高度分布宽度由各条曲线共享。",
    "The model omits correlations between height and radius.": "模型未考虑高度与半径之间的相关性。",
    "Vertical reduced χ² = {value}": "垂直切线的约化 χ² = {value}",
    "{name} has a vertical reduced χ² of {chi2} (> 3). The model omits the effective refractive index "
    "of the particle layer. For dense or high-Z layers the fitted height is likely underestimated. "
    "See Physics and Methods §9.18 for height biases in the BornAgain benchmark.":
        "{name} 的垂直切线约化 χ² 为 {chi2}（> 3）。模型未计入颗粒层的有效折射率，致密层或高原子序数元素层的拟合高度很可能偏低。"
        "BornAgain 基准中的高度偏差见《物理与方法》9.18 节。",
    "The angles and wavelength set the scattering paths for spheres and for upright cylinders with a height "
    "distribution in DA. Choose a substrate for the four-term DWBA. With an unknown substrate, |r| and arg r "
    "set the effective reflection coefficient.":
        "角度和波长确定球以及 DA 中有高度分布的竖直圆柱的散射路径。选择基底可使用四项 DWBA。基底未知时，|r| 和 arg r 决定有效反射系数。",
}
