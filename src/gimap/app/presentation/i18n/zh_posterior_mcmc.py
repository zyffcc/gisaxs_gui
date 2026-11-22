"""Chinese text for the independent ensembles of the MCMC posterior check."""

POSTERIOR_MCMC_ZH = {
    "a second sampler of the same posterior (same likelihood, model error and prior); "
    "checks the Monte Carlo accuracy of this mode":
        "用第二种抽样器对同一后验抽样（似然、模型误差和先验相同），检验此模态的蒙特卡罗精度",
    "The reference intervals are least-squares intervals from the BIC fallback.":
        "参照区间是回退到 BIC 时给出的最小二乘区间。",
    "Walkers in each independent ensemble (even). Auto uses at least 16 and four per value.":
        "每个独立系综的行走者数量（偶数）。自动设置为每个参数至少 4 个，合计至少 16 个。",
    "Running {ensembles} ensembles of {walkers} walkers for {steps} steps each …":
        "正在运行 {ensembles} 个系综，每个系综有 {walkers} 个行走者，各走 {steps} 步…",
    "{ensembles} ensembles of {walkers} walkers, {steps} steps each in {seconds:.1f} s, acceptance {acceptance:.2f}.":
        "{ensembles} 个系综，每个系综有 {walkers} 个行走者，各走 {steps} 步，用时 {seconds:.1f} 秒，接受率 {acceptance:.2f}。",
    "Could not find finite posterior density for every starting walker": "未能为每个行走者找到后验密度有限的起点",
    "The data selection changed. The displayed check uses the points selected when it started.":
        "已更改选取的数据。这里显示的检验使用运行开始时选取的点。",
}
