"""Chinese for what the GIWAXS automatic analysis says about the limits of its numbers (texture rule, the bins next
to the sample horizon, the error model of the peak fits, an αi set in Analyze's command bar)."""

GIWAXS_AUTO_ZH = {
    # A ring's texture sentence and notes (assistant/domain/orientation.py), shown through guided_words.
    "no clear preferred orientation (f − f_iso = {f}, within ±{margin}): the ring is strongest at |χ| ≈ {chi}° but "
    "nearly as intense elsewhere":
        "择优取向不明显（f − f_iso = {f}，在 ±{margin} 以内）：环在 |χ| ≈ {chi}° 处最强，但在其他位置几乎同样强",
    "no single preferred orientation: strongest at |χ| ≈ {chi}°, but f − f_iso = {f} puts most of the intensity "
    "nearer the sample plane":
        "没有单一的择优取向：在 |χ| ≈ {chi}° 处最强，但 f − f_iso = {f} 表明大部分强度更靠近样品平面",
    "no single preferred orientation: strongest at |χ| ≈ {chi}°, but f − f_iso = {f} puts most of the intensity "
    "nearer the surface normal":
        "没有单一的择优取向：在 |χ| ≈ {chi}° 处最强，但 f − f_iso = {f} 表明大部分强度更靠近表面法线",
    "Leaving out |χ| ≥ {chi}°, next to the sample horizon, changes f − f_iso from {a} to {b} ({n}σ_f): f depends on "
    "those bins. There the exit angle αf is smallest, and absorption in the film, the transmission function or a "
    "shadow can lower the intensity; GIMaP corrects none of them.":
        "去掉紧邻样品地平线的 |χ| ≥ {chi}° 后，f − f_iso 从 {a} 变为 {b}（{n}σ_f）：f 依赖这些区间。那里的出射角 αf "
        "最小，薄膜中的吸收、透射函数或阴影都可能降低强度；GIMaP 对这些都不做校正。",
    "Without those bins the texture rule (|f − f_iso| against {margin}) would give another verdict.":
        "去掉这些区间后，织构判据（|f − f_iso| 与 {margin} 比较）会给出另一个结论。",
    # Where αi came from (assistant/application/pipeline.py).
    "from Analyze's command bar (the αi set there holds for every frame until it is changed)": "来自分析的命令栏（那里设定的 αi 对之后的每一帧都有效，直到改动）",
    "from given; it replaces the {value}° set in Analyze's command bar": "来自给定值；它取代了在分析命令栏中设定的 {value}°",
    "from the notes; it replaces the {value}° set in Analyze's command bar": "来自笔记；它取代了在分析命令栏中设定的 {value}°",
    # The errors a peak was fitted with (assistant/presentation/guided_details.py).
    "Poisson (counting) error": "泊松（计数）误差",
    "pixel-scatter error (statistical, not counting noise)": "像素离散误差（统计误差，不是计数噪声）",
    "statistical error": "统计误差",
}

__all__ = ["GIWAXS_AUTO_ZH"]
