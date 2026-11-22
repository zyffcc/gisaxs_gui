"""Pixel corrections and the exit-angle selection for GIWAXS ring profiles."""

ORIENTATION_REDUCTION_ZH = {
    "The Lorentz exclusion half-width must be finite and nonnegative.": "洛伦兹排除带半宽必须是有限的非负数。",
    "The orientation minimum exit angle must be between 0° and 90°.": "取向分析的最低出射角必须在 0° 到 90° 之间。",
    "I (summed counts)": "I（计数之和）",
    "Intensity (see legend)": "强度（见图例）",
    "Select the ring again to measure its background before the Lorentz correction.":
        "请重新选择衍射环，以测量洛伦兹校正前的背景。",
    "No valid pixels for the corrected map or radial curve.": "校正后的图和径向曲线没有有效像素。",
    "{name} (sum)": "{name}（总和）",
    "{name} (mean)": "{name}（均值）",
    "Lorentz factor for peak areas": "峰面积的洛伦兹因子",
    "Weight crystalline intensities by cos αi |qy|/k. The radial curve sums these pixels in each q bin.":
        "用 cos αi |qy|/k 对晶体散射强度加权。径向曲线在每个 q 区间对这些像素求和。",
    "Exclude |qy|/k below this many horizontal pixel widths divided by distance.":
        "排除 |qy|/k 小于此像素数乘以水平像素宽度再除以样品到探测器距离的区域。",
    "Incidence-plane half-width": "入射面排除带半宽",
    "Orientation minimum αf": "取向分析的最低出射角 αf",
    "Automatic uses twice the film critical angle, or twice the incidence angle.":
        "自动取薄膜临界角的两倍；未给临界角时取入射角的两倍。",
    "Polarise ring profiles by default": "默认对环剖面作偏振校正",
    "Use f = 0.99 for orientation when the frame polarisation is off.": "整帧偏振校正关闭时，取向分析使用 f = 0.99。",
    "Ring orientation uses polarisation by default. Map and radial intensity corrections are optional.":
        "环取向分析默认作偏振校正。图像和径向曲线的强度校正可按需开启。",
    "Optional upper χ limit after the minimum exit angle in Mask & Corrections.":
        "在“掩膜与校正”中指定最低出射角后，还可在这里限制拟合的 χ 上限。",
}
