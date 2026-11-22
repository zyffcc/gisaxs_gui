"""Chinese text for ring-point calibration residuals."""

CALIBRATION_RESIDUALS_ZH = {
    "Automatic fit residual": "自动拟合残差",
    "Ring-point RMS {before} px before refinement, {after} px after":
        "环上点的 RMS 在精修前为 {before} px，精修后为 {after} px",
    "Ring-point RMS {rms} px": "环上点的 RMS 为 {rms} px",
    "Ring-radius RMS {rms} px": "环半径的 RMS 为 {rms} px",
    "none fits well (best {standard}: {rings} rings, ring-radius RMS {rms} px): the image may show another standard, "
    "the energy or pixel size may be wrong, or the rings are too weak":
        "均不吻合（最佳为 {standard}，{rings} 个环，环半径 RMS 为 {rms} px）：可能是其他标样、能量或像素尺寸有误，或衍射环太弱",
    "clear: {standard} ({rings} rings, ring-radius RMS {rms} px) beats {second} ({second_rings} rings, ring-radius RMS {second_rms} px)":
        "明确：{standard}（{rings} 个环，环半径 RMS 为 {rms} px）优于 {second}（{second_rings} 个环，环半径 RMS 为 {second_rms} px）",
    "clear: {standard} ({rings} rings, ring-radius RMS {rms} px)": "明确：{standard}（{rings} 个环，环半径 RMS 为 {rms} px）",
    "RMS difference between observed and predicted ring radii at the automatic ring-matching step "
    "(azimuthal average).":
        "自动匹配衍射环时，观测环半径（方位角平均）与预测环半径之差的均方根。",
    "RMS distance of the fitted ring points from the rings of this geometry.":
        "拟合所用的环上点与此几何预测的衍射环之间距离的均方根。",
    "Before refinement, the RMS on these points was {rms} px.":
        "精修前，同一组点的 RMS 为 {rms} px。",
    "Fitting the same points with zero tilt gives {rms} px.":
        "在倾斜为零时拟合同一组点，RMS 为 {rms} px。",
}
