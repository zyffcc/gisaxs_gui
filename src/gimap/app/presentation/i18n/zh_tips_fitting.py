"""Chinese for Fitting, In-situ series, 1D Predict and Compare: the shorter on-screen texts and the tooltips
that took over the explanations which used to stand under the controls (2026-11 small-text review)."""

TIPS_FITTING_ZH = {
    # -- Single analysis: the steps, the curve card, the status line ---------------------------------------
    "{points} points fitted": "拟合 {points} 个点",
    "Starting model, not set up yet": "初始模型，尚未设置",
    "Fit finished: {state}.": "拟合完成：{state}。",
    "No fit yet.": "还没有拟合。",
    "{points} points, |q| {low}–{high} nm⁻¹; {fitted} in the fitting range":
        "{points} 个点，|q| {low}–{high} nm⁻¹；拟合范围内 {fitted} 个",
    "σ from the file": "σ 来自文件",
    "⚠ No σ in the file: every point weighted alike (ln I fitted)": "⚠ 文件里没有 σ：每个点权重相同（拟合 ln I）",
    "⚠ No σ in the file: every point weighted alike": "⚠ 文件里没有 σ：每个点权重相同",
    "Residuals: {formula}. Each point is weighted by 1/σ from the file; fit1d adds the systematic error floor "
    "(Options).": "残差：{formula}。每个点按文件中的 1/σ 加权；fit1d 另加系统误差下限（选项）。",
    "Only the points inside the fitting range are fitted. Type the limits, or drag the orange band on the plot.":
        "只拟合拟合范围内的点。输入上下限，或在图上拖动橙色区域。",
    "Add a particle family to the model. Not sure which? Fit ▸ Find the particle shape (no AI) tries every family.":
        "往模型里添加一种颗粒。不确定是哪种？拟合 ▸ 寻找颗粒形状（不用 AI）会逐一尝试每一种。",
    "fit1d: a relative systematic error added to σ in quadrature (without σ in the file, to Poisson errors). 2–5 % "
    "suits bright measured curves.":
        "fit1d：按平方和加到 σ 上的相对系统误差（文件里没有 σ 时加到泊松误差上）。很亮的实测曲线适合 2–5 %。",
    "Select values, then Fix, Let It Fit or Range… (or right-click); Fit Again uses them.":
        "选中数值，再点“固定”“参与拟合”或“范围…”（也可以右键）；“再拟合一次”会用上它们。",
    # -- In-situ series --------------------------------------------------------------------------------------
    "Shown after Start": "开始后显示",
    "No curves listed — choose a folder, or Send Series to Fitting in Analyze":
        "还没有列出曲线——选择一个文件夹，或在分析中点“把序列发送到拟合”",
    "Start would fit this starting guess to every frame.": "现在开始会用这个初始猜测拟合每一帧。",
    # -- 1D Predict ------------------------------------------------------------------------------------------
    "Experimental: candidates are starts to check, not results.": "实验性：候选只是待检查的起点，不是结果。",
    "General V5 proposes several compositions; the single-RC specialist needs a known single random cylinder. Both "
    "are experimental. logRMSE and RMS/σ include the measurement noise and are not calibrated probabilities. The "
    "model is chosen in Parameters… ▸ Fit method.":
        "通用 V5 会给出多种组成；单 RC 专用模型要求已知只有一个随机取向圆柱。两者都是实验性的。logRMSE 和 RMS/σ "
        "含测量噪声，不是校准过的概率。模型在 参数… ▸ 拟合方法 里选择。",
    "Input: the current cut of Fitting (q in nm⁻¹); each side fitted separately":
        "输入：拟合页的当前切线（q 为 nm⁻¹）；两侧分别拟合",
    "Ready.": "就绪。",
    "Finished in {seconds} s · {failures} failed files · {count} candidates saved.":
        "用时 {seconds} s 完成 · {failures} 个文件失败 · 保存了 {count} 个候选。",
    "q in {unit}, as their header states": "q 按文件头注明的 {unit}",
    "q in {unit}, no unit in their header (Parameters…)": "q 按 {unit}，文件头未注明单位（参数…）",
    # -- Compare ---------------------------------------------------------------------------------------------
    "Send a Series map from Analyze, or add or drop curve files. One series is described; two or more are compared.":
        "从分析页发送序列热图，或添加、拖放曲线文件。一个序列给出描述；两个或更多相互比较。",
    "Add a series: Analyze's Series map (Analyze ▸ Series ▸ Send to Compare, once per sample), a folder of curve "
    "files (x and I columns; a folder Analyze wrote asks which kind of curve) or chosen curve files. Folders and "
    "files can also be dropped on the page.":
        "添加一个序列：分析页的序列热图（分析 ▸ 序列 ▸ 发送到比较，每个样品一次）、一个曲线文件夹（x、I 两列；"
        "分析写出的文件夹会询问用哪一种曲线）或若干选中的曲线文件。文件夹和文件也可以直接拖放到本页。",
    "Each series is one run or one sample: its frames in order. Two or more are compared with each other; one "
    "alone is described (odd frames, stages). Double-click a name to rename the series.":
        "每个序列是一次测量或一个样品：按顺序的各帧。两个或更多时相互比较；只有一个时给出它的描述（异常帧、阶段）。"
        "双击名称可以重命名序列。",
    "Every series: its odd frames, its stages, and the frame by which half / 90 % of its change had happened (Half "
    "done / 90 % done). How far the end states are apart: the table below.":
        "每个序列：异常帧、阶段，以及变化完成一半 / 90 % 时所在的帧（完成一半 / 完成 90 %）。终态相差多远：见下面的表。",
    "End states {end} % apart": "终态相差 {end} %",
    "Method: log I on the shared q range, odd frames left out": "方法：在共同的 q 范围上比较 log I，排除异常帧",
    "1 odd frame left out (below).": "排除了 1 个异常帧（见下方）。",
    "{n} odd frames left out (below).": "排除了 {n} 个异常帧（见下方）。",
}
