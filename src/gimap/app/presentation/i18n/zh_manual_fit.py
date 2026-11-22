"""Size distributions, q resolution and weighting of manual GISAXS fits."""

MANUAL_FIT_ZH = {
    "Unrecognized GIMaP fit model format.": "无法识别的 GIMaP 拟合模型格式。",
    "Choose the size distribution for this population. The parameter labels give its location and width. "
    "Results also show the mean radius and relative spread by number and by intensity.":
        "选择本粒子组的尺寸分布。参数名称标明分布的位置和宽度。结果还分别给出按数量和按强度加权的平均半径和相对标准差。",
    "Log-normal, by number (median)": "对数正态，按数量（中位数）",
    "Schulz (Gamma), by number": "舒尔茨（伽马），按数量",
    "Gaussian, by number (tabulated)": "高斯，按数量（查表）",
    "Log-normal, by number (mean)": "对数正态，按数量（均值）",
    "Median R": "半径中位数",
    "Median h": "高度中位数",
    "Median radius of the log-normal distribution": "对数正态分布的半径中位数",
    "Median height of the log-normal distribution": "对数正态分布的高度中位数",
    "Standard deviation of ln R": "ln R 的标准差",
    "Standard deviation of ln h": "ln h 的标准差",
    "Resolution": "分辨率",
    "Constant σ_q": "恒定 σ_q",
    "Resolution from the file": "文件中的分辨率",
    "Weights": "权重",
    "Relative (σ = I)": "相对（σ = I）",
    "Poisson (σ = √I)": "泊松（σ = √I）",
    "None (fit ln I)": "无（拟合 ln I）",
    "Refine and Search smear the particle terms using constant σ_q or each point’s dQ.":
        "精修和搜索使用恒定 σ_q 或各点的 dQ 对粒子散射项进行高斯卷积。",
    "Refine and Search use these weights. Without a file σ, the default fits ln I with equal weights.":
        "精修和搜索使用所选的权重。默认选项在文件未提供 σ 时对 ln I 等权拟合。",
    "dQ from the file": "dQ 来自文件",
    "Refine and Search use {weights}.": "精修和搜索使用的权重为{weights}。",
    "{count} points with I ≤ 0 were left out for the selected weights.": "按所选权重拟合时排除了 {count} 个 I ≤ 0 的点。",
    "Resolution from the file needs a dQ or sigma_q column, or NXcanSAS Qdev.":
        "使用文件中的分辨率需要 dQ 或 sigma_q 列，或 NXcanSAS 文件中的 Qdev。",
    "Every fitted point needs a finite, non-negative dQ for resolution from the file.":
        "使用文件中的分辨率时，每个拟合点的 dQ 都须为有限的非负数。",
    "Resolution from the file needs fitted points with dQ.": "使用文件中的分辨率需要有 dQ 的拟合点。",
    "A resolution column cannot be used as q, intensity or sigma.": "分辨率列不能用作 q、强度或 σ 列。",
    "Each curve must have one resolution column.": "每条曲线只能指定一个分辨率列。",
    "The unit of dQ must be an inverse length.": "dQ 的单位必须是长度的倒数。",
    "dQ and q must have the same length": "dQ 和 q 的点数必须相同",
    "Resolution (dQ)": "分辨率（dQ）",
    "The dQ column supplies the Gaussian width at each point when you choose Resolution from the file.":
        "选择“文件中的分辨率”时，dQ 列提供每个点的高斯标准差。",
    "Each NXcanSAS SASdata group supplies one curve. Idev gives the intensity uncertainty and Qdev gives "
    "the q resolution when present. The preview shows the first curve.":
        "每个 NXcanSAS SASdata 组提供一条曲线。如有 Idev 或 Qdev，则分别用作强度不确定度或 q 分辨率。此处预览第一条曲线。",
}
