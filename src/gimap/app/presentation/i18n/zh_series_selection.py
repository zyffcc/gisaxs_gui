"""Chinese text for solution selection and uncertainty in Fitting's in-situ series."""

SERIES_SELECTION_ZH = {
    "Previous result, Predict at chain breaks": "上一帧的结果；链断开处用“预测参数”",
    "Refine again in a backward pass": "反向再精修一遍",
    "Retry with Predict after a failure, non-convergence, a new stage, or χ²ᵣ above both 2 and 3× the previous value. Keep the better fit.":
        "若拟合失败、未收敛、进入新阶段，或 χ²ᵣ 超过 2 且超过前一帧的 3 倍，则用“预测参数”重试，并保留较好的拟合结果。",
    "Predict solutions for each frame using the particles in Single analysis.": "按单条分析中的粒子模型逐帧预测各个可能解。",
    "Keep the nearest solution with at least 20% of the largest probability.": "在概率达到最高概率 20% 的解中，保留最接近上一帧的解。",
    "Smooth across frames (hidden Markov model)": "跨帧平滑（隐马尔可夫模型）",
    "Use all frames to choose the most probable sequence of Predict solutions.": "用各帧的数据选出概率最高的预测解序列。",
    "Log step per frame (s)": "逐帧对数步长 (s)",
    "A log step of 0.1 allows about 10% change per frame in one coordinate.":
        "步长 0.1 对应单个坐标逐帧约 10% 的变化尺度。",
    "Least squares": "最小二乘",
    "Overlay the chosen Predict solution’s least-squares values as small open symbols.": "用较小的空心符号叠加所选预测解的最小二乘值。",
    "Predict least squares": "预测解最小二乘值",
    "Posterior median (68%)": "后验中位数（68% 区间）",
    "Refined fit (±1σ)": "精修值（±1σ）",
    "Chosen by": "选解规则",
    "Rank": "排名",
    "p chosen": "所选概率",
    "p max": "最高概率",
    "p smoothed": "平滑概率",
    "How the Predict solution was selected. A discarded retry did not supply the fitted values.":
        "本帧预测解的选择依据。标记为未采用的重试没有用于最终拟合结果。",
    "Rank of the chosen solution by its probability in this frame (1 is most probable).":
        "所选解按单帧概率的排名（1 为概率最高的解）。",
    "Probability of the chosen solution using this frame alone.": "仅用本帧数据得到的所选解概率。",
    "Largest solution probability using this frame alone.": "仅用本帧数据得到的各解中的最高概率。",
    "Probability of the chosen solution given the series, from forward–backward.": "通过前向–后向算法利用整个序列计算的所选解概率。",
    "The 68% interval is conditional on the chosen solution. It excludes uncertainty about which solution to use.":
        "68% 区间以所选解为条件，不包含选择不同解带来的不确定度。",
    "closest to frame {n}": "最接近第 {n} 帧",
    "most probable": "概率最高",
    "HMM": "隐马尔可夫模型",
    "{rule} (discarded retry)": "{rule}（重试未采用）",
    "forward": "正向",
    "Predict failed": "预测失败",
    "fit failure, non-convergence, new stage, or χ²ᵣ above {floor} and {ratio}× the previous χ²ᵣ":
        "拟合失败、未收敛、进入新阶段，或 χ²ᵣ 超过 {floor} 且超过前一帧的 {ratio} 倍",
    "not chained": "未使用链式拟合",
    "closest eligible solution (p ≥ {fraction} p max)": "满足 p ≥ {fraction} p max 的解中最接近参考帧者",
    "HMM (s = {scale}, ε = {penalty})": "HMM（s = {scale}，ε = {penalty}）",
    "Chain breaks at: {reason}; Predict solution chosen by: {choice}.": "链断开条件：{reason}；预测解的选择规则：{choice}。",
    "Series report…": "序列报告…",
    "Save Series Report": "保存序列报告",
    "Markdown (*.md)": "Markdown 文档 (*.md)",
    "Smoothing the Predict solutions across frames…": "正在跨帧平滑预测解…",
    "Series smoothing failed: {error}": "序列平滑失败：{error}",
    "Predict returned no valid set of mode probabilities for this frame": "此帧的预测没有得到有效的各解概率。",
    "The continuity scale must be positive": "连续性步长必须大于零。",
    "This project did not record Predict solution probabilities or its selection distance.":
        "该项目没有记录预测解的概率或选解时使用的距离。",
}
