"""Chinese text for the Predict parameters sampling notice."""

PREDICT_SAMPLING_ZH = {
    "This cut has {count} fitted points. The validation grid has 397 equally spaced points "
    "from 0.02 to 2 nm⁻¹ (Δq = 0.005 nm⁻¹).":
        "此切线有 {count} 个点参与拟合。验证网格在 0.02 至 2 nm⁻¹ 之间有 397 个等间距点（Δq = 0.005 nm⁻¹）。",
    "At the intensity maximum (|q| = {peak} nm⁻¹), Δq is {spacing} nm⁻¹ "
    "({ratio} times the validation spacing).":
        "在强度最大处（|q| = {peak} nm⁻¹），Δq 为 {spacing} nm⁻¹，是验证网格间距的 {ratio} 倍。",
    "Fewer than 80 fitted points. Check whether the measured features are resolved.":
        "参与拟合的点少于 80 个。请检查数据点能否分辨曲线上的特征。",
    "With more than 400 fitted points, sampling may fail and the fits may be compared by BIC. "
    "Try rebinning to 150–250 log-spaced points.":
        "参与拟合的点多于 400 个时，抽样可能失败，此时将用 BIC 比较各解。可尝试按对数间距重新分箱为 150–250 个点。",
}
