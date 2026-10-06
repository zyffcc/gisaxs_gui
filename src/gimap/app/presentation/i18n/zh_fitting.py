"""Chinese texts of the single-curve Fitting page (merged into ``zh.ZH``; keyed by the exact English
text in the code, like the rest of the table). Symbols, units and numbers are not translated."""

FITTING_ZH = {
    # -- command bar and plot -------------------------------------------------------------
    "Open a 1D curve (q, I, σ) — Analyze ▸ Send to Fitting opens its cut here":
        "打开一条一维曲线（q、I、σ）——在分析中点“发送到拟合”会把切线打开到这里",
    "Load Model…": "载入模型…",
    "A model saved from Fitting (JSON), for this curve": "从拟合保存的模型（JSON），用于这条曲线",
    "Undo the last change of the model (Ctrl+Z)": "撤销模型的上一次修改（Ctrl+Z）",
    "No curve yet — open one, or send a cut from Analyze": "还没有曲线——打开一条，或从分析发送一条切线",
    "Run the method chosen in the Fit step (Ctrl+Return)": "运行“拟合”步骤里选的方法（Ctrl+Return）",
    "Data and Fit…": "数据与拟合…",
    "Plot…": "曲线图…",
    "Model…": "模型…",
    "Terms": "各项",
    "Also draw each term of the model: the particles, the background, the resolution peak":
        "同时画出模型的每一项：颗粒、背景、分辨率峰",
    "not fitted": "未拟合的点",
    "model": "模型",
    "residual": "残差",
    "resolution peak": "分辨率峰",
    # -- steps ------------------------------------------------------------------------------
    "Open a curve": "打开曲线",
    "Open a curve (q, I, σ), or send a cut from Analyze ▸ Send to Fitting.":
        "打开一条曲线（q、I、σ），或在分析里点“发送到拟合”。",
    "{name} · {points} points fitted": "{name} · 拟合 {points} 个点",
    "no particle": "没有颗粒",
    "{free} free": "{free} 个自由参数",
    "{names}; {free} values are fitted.": "{names}；拟合 {free} 个值。",
    "A starting model: set values you know, or let Fit ▸ Find the particle shape choose the family.":
        "这是初始模型：填入已知的值，或用 拟合 ▸ 寻找颗粒形状 来选择颗粒类型。",
    "Fitting …": "正在拟合…",
    "Stopped: the best values so far are kept": "已停止：保留了目前最好的值",
    "Choose a method, then Fit": "选择方法，然后拟合",
    "Errors, solutions, Save": "误差、解、保存",
    "After a fit": "拟合之后",
    # -- 1 curve --------------------------------------------------------------------------
    "A 1D curve: columns q, I and optionally σ (.dat, .txt, .csv). Analyze ▸ Send to Fitting opens its cut here.":
        "一维曲线：q、I 两列，可选 σ（.dat、.txt、.csv）。在分析里点“发送到拟合”会把切线打开到这里。",
    "Halves of the cut": "切线的两半",
    "A cut through the beam has q < 0 and q > 0: fit their mean, both on |q|, or one half":
        "穿过光束的切线有 q < 0 和 q > 0 两半：拟合两半的平均、|q| 上的两半，或其中一半",
    "Both halves on |q|": "|q| 上的两半",
    "q > 0 half": "q > 0 一半",
    "q < 0 half": "q < 0 一半",
    "Fitting range": "拟合范围",
    "Whole Curve": "整条曲线",
    "Fit every point of the curve": "拟合曲线的所有点",
    "Or drag the orange band on the plot. Only the points inside it are fitted.":
        "也可以在图上拖动橙色区域。只拟合区域内的点。",
    "File": "文件",
    "q in the file": "文件里的 q",
    "Curves from Analyze are in Å⁻¹; other files may be in nm⁻¹": "分析输出的曲线用 Å⁻¹；其他文件可能用 nm⁻¹",
    "No curve yet.": "还没有曲线。",
    "{name}\n{points} points, |q| {low}–{high} nm⁻¹ ({low_a}–{high_a} Å⁻¹); {fitted} in the fitting range.":
        "{name}\n{points} 个点，|q| {low}–{high} nm⁻¹（{low_a}–{high_a} Å⁻¹）；拟合范围内 {fitted} 个。",
    "σ from the file: the fit weights each point by 1/σ.": "σ 来自文件：拟合按 1/σ 给每个点加权。",
    "No σ in the file: every point gets the same relative weight (ln I is fitted).":
        "文件里没有 σ：每个点的相对权重相同（拟合 ln I）。",
    "Could not open {name}: {reason}": "无法打开 {name}：{reason}",
    "Opened {name}: {points} points": "已打开 {name}：{points} 个点",
    "Opened {name}.": "已打开 {name}。",
    # -- 2 model --------------------------------------------------------------------------
    "Add Particle ▾": "添加颗粒 ▾",
    "Add a particle family to the model": "往模型里添加一种颗粒",
    "Ranges": "范围",
    "Show the min–max range of every parameter: the fit keeps each value inside it, and “Search the ranges” searches across it":
        "显示每个参数的最小–最大范围：拟合让每个值保持在范围内，“在范围内搜索”会在整个范围里搜索",
    "Tick “fit” for the values the fit may change. Not sure which particle? Fit ▸ Find the particle shape tries every family.":
        "勾选“拟合”的值才会被拟合改变。不确定是哪种颗粒？拟合 ▸ 寻找颗粒形状 会逐一尝试。",
    "fit": "拟合",
    "Ticked: the fit may change this value; unticked: it stays fixed": "勾选：拟合可以改变这个值；不勾选：保持固定",
    "Range of this parameter: the fit keeps it inside, “Search the ranges” searches across it":
        "这个参数的范围：拟合让它保持在范围内，“在范围内搜索”会在整个范围里搜索",
    "at a bound": "到达边界",
    "The fit stopped at a bound of the range: widen the range or fix this value": "拟合停在了范围的边界：放宽范围或固定这个值",
    "1σ error of the last fit": "上次拟合的 1σ 误差",
    "Distance D": "间距 D",
    "Interference between neighbours: a paracrystal of distance D and disorder σD/D":
        "相邻颗粒之间的干涉：间距 D、无序度 σD/D 的准晶模型",
    "Remove this component": "删除这个组分",
    "No particle yet: add one above, or let Fit ▸ Find the particle shape choose.":
        "还没有颗粒：在上面添加一个，或让 拟合 ▸ 寻找颗粒形状 来选择。",
    "Background and resolution peak": "背景与分辨率峰",
    "I(q) = BG + k·[Σ particles + A/(1 + (|q|/w)^ν)]: the peak is the direct and reflected beam's tail and the resolution near q = 0":
        "I(q) = BG + k·[Σ 颗粒 + A/(1 + (|q|/w)^ν)]：这个峰是直射束和反射束的尾部以及 q = 0 附近的分辨率",
    "Intensity of this component (the form factor is 1 at q = 0)": "这个组分的强度（形状因子在 q = 0 处为 1）",
    "Mean radius": "平均半径",
    "Relative spread of the radius (Gaussian)": "半径的相对分布宽度（高斯）",
    "Mean height (length) of the cylinders": "圆柱的平均高度（长度）",
    "Relative spread of the height (Gaussian)": "高度的相对分布宽度（高斯）",
    "Mean distance between neighbours (paracrystal)": "相邻颗粒的平均间距（准晶）",
    "Relative disorder of the distance": "间距的相对无序度",
    "Constant added to the whole curve": "加在整条曲线上的常数",
    "Peak A": "峰 A",
    "Height of the resolution peak A / (1 + (|q|/w)^ν) at q = 0": "分辨率峰 A / (1 + (|q|/w)^ν) 在 q = 0 处的高度",
    "Peak w": "峰 w",
    "Width w of the resolution peak": "分辨率峰的宽度 w",
    "Peak ν": "峰 ν",
    "Exponent ν of the resolution peak (its tail)": "分辨率峰的指数 ν（决定尾部）",
    "Factor k": "因子 k",
    "Overall factor of everything but the background (usually fixed at 1)": "除背景外所有项的总因子（通常固定为 1）",
    "Loaded the model from {name}.": "已从 {name} 载入模型。",
    "Could not load the model: {reason}": "无法载入模型：{reason}",
    "Could not use this solution: {reason}": "无法使用这个解：{reason}",
    "{model} is in Model; this model draws the same curve.": "{model} 已放入模型；这个模型画出的是同一条曲线。",
    "{model} is in Model as a start: this model differs from the solution by up to {percent} % (the Vertical Cylinder here weights radii by R⁴). Fit to refine it.":
        "{model} 已作为初值放入模型：这个模型与该解最多相差 {percent} %（这里的竖直圆柱按 R⁴ 给半径加权）。请拟合来精修。",
    # -- 3 fit ----------------------------------------------------------------------------
    "Refine the current values": "精修当前的值",
    "Least squares from the values in Model: fast; finds the nearest good fit.": "从模型里的值出发做最小二乘：快，找到最近的好拟合。",
    "Search the ranges, then refine": "在范围内搜索，再精修",
    "Tries values across the min–max range of every free parameter first; slower, for a poor start.":
        "先在每个自由参数的最小–最大范围内尝试取值；较慢，适合初值不好时。",
    "Find the particle shape (no AI)": "寻找颗粒形状（不用 AI）",
    "Fits sphere, random cylinder and vertical cylinder from several starts and lists the solutions.":
        "从多个初值分别拟合球、随机取向圆柱和竖直圆柱，并列出各个解。",
    "AI proposal (1D Predict)": "AI 建议（一维预测）",
    "The V5 model proposes compositions and parameters, then corrects them numerically.": "V5 模型给出组成和参数，再做数值校正。",
    "Families": "颗粒类型",
    "Try every family": "逐一尝试每种类型",
    "The families in Model": "模型里的类型",
    "Refine": "精修",
    "Search the ranges": "在范围内搜索",
    "Find the particle shape": "寻找颗粒形状",
    "AI proposal": "AI 建议",
    "Stop the fit; the best values so far are kept": "停止拟合；保留目前最好的值",
    "Advanced": "高级",
    "Evaluations at most": "最多计算次数",
    "How many model evaluations a fit may use (automatic: by the method and free parameters)":
        "一次拟合最多可以计算模型多少次（自动：按方法和自由参数个数决定）",
    "Fit Many Curves…": "拟合多条曲线…",
    "1D Predict: a list of curve files fitted one after another, with their results": "一维预测：逐条拟合一组曲线文件，并列出结果",
    "Open a curve first.": "请先打开一条曲线。",
    "Too few points in the fitting range: widen it in the Curve step.": "拟合范围内的点太少：请在“曲线”步骤里放宽范围。",
    "Finding the particle shape is not available here: choose another method.": "这里无法寻找颗粒形状：请选择其他方法。",
    "Add a particle in Model, or use Find the particle shape.": "请在模型里添加颗粒，或使用“寻找颗粒形状”。",
    "{method}: {points} points, {free} free parameters": "{method}：{points} 个点，{free} 个自由参数",
    "Stopping … the best values so far are kept.": "正在停止…保留目前最好的值。",
    "The fit failed: {reason}": "拟合失败：{reason}",
    "Stopped: the best values so far are kept ({quality}).": "已停止：保留了目前最好的值（{quality}）。",
    "Fitted: {quality}.": "拟合完成：{quality}。",
    "No solution: the curve may be too short or too noisy for these families.": "没有解：对这些颗粒类型来说，曲线可能太短或噪声太大。",
    "{count} solutions; the best, {model} (χ²ᵣ {chi2}), is in Model. Fit again to refine it, or choose another in Results.":
        "{count} 个解；最好的 {model}（χ²ᵣ {chi2}）已放入模型。再拟合一次来精修，或在结果里选择另一个。",
    "{method}: {count} solutions": "{method}：{count} 个解",
    "1D Predict is not available in this installation": "这个安装里没有一维预测",
    "1D Predict failed": "一维预测失败",
    # -- 4 results ------------------------------------------------------------------------
    "value": "值",
    "Solutions": "解",
    "Use This Solution": "使用这个解",
    "Put the selected solution into Model (Undo brings the previous model back)": "把选中的解放入模型（撤销可恢复之前的模型）",
    "Save Data and Fit…": "保存数据与拟合…",
    "CSV: q, I, σ, the model, the residuals and each term; a JSON record of the model and the fit next to it":
        "CSV：q、I、σ、模型、残差和各项；旁边附模型与拟合的 JSON 记录",
    "Save Model…": "保存模型…",
    "The model (values, fit/fixed, ranges) as JSON, to load for another curve": "模型（值、拟合/固定、范围）存为 JSON，可用于另一条曲线",
    "stopped": "已停止",
    "{chi} {value} · {points} points · {free} free · {state}": "{chi} {value} · {points} 个点 · {free} 个自由参数 · {state}",
    "No fit yet: Fit shows here how good it is and the error of every value.": "还没有拟合：拟合后这里显示拟合质量和每个值的误差。",
    "After a fit: its quality, the errors and the solutions to compare.": "拟合之后：拟合质量、误差和可比较的解。",
    "{quality}\nlog RMSE {rmse} · {evaluations} evaluations · {seconds} s · {method}":
        "{quality}\nlog RMSE {rmse} · 计算 {evaluations} 次 · {seconds} s · {method}",
    "The model was changed after this fit: the errors belong to the fitted values.": "这次拟合之后模型改过了：误差对应的是拟合出的值。",
    "{name} stopped at a bound of its range: widen the range or fix it.": "{name} 停在了范围的边界：请放宽范围或固定它。",
    "Strongly correlated, so the data do not separate them (fix one, or read their errors as a range): {pairs}.":
        "强相关，数据无法把它们分开（固定其中一个，或把误差当作取值范围看）：{pairs}。",
    "χ²ᵣ well above 1: the model misses features of the curve, or σ is underestimated.": "χ²ᵣ 远大于 1：模型没描述出曲线的某些特征，或 σ 估计偏小。",
    "fixed": "固定",
    "From {source}.": "来自 {source}。",
    "shape search": "形状搜索",
    "This model differs from the solution by up to {percent} %.": "这个模型与该解最多相差 {percent} %。",
    "{model} is in Model (Undo brings the previous one back).": "{model} 已放入模型（撤销可恢复之前的模型）。",
    "Saved {name} and its record {record}.": "已保存 {name} 及其记录 {record}。",
    "Saved {name}.": "已保存 {name}。",
    "The plot as shown: PNG (image) or SVG (vector)": "当前显示的图：PNG（图片）或 SVG（矢量）",
    "Open Curve": "打开曲线",
    "Load Model": "载入模型",
    "Save Data and Fit": "保存数据与拟合",
    "Save Plot": "保存曲线图",
    "Save Model": "保存模型",
    # -- 2026-10-05 polish ------------------------------------------------------------------------
    "Add Particle": "添加颗粒",
    "Changed after the fit: Fit again": "拟合后已更改：请重新拟合",
    "Residuals: {formula}": "残差：{formula}",
    "{count} solutions to compare": "{count} 个解可以比较",
    "{count} solutions · best {model} {chi} {value}": "{count} 个解 · 最佳 {model} {chi} {value}",
    "The fitting range or the left-out points changed after this fit: Fit again for its quality and errors.":
        "这次拟合之后，拟合范围或被排除的点已改变：请重新拟合以得到它的质量和误差。",
    "The trend of the chosen value appears here after Start": "开始后，所选数值的趋势会显示在这里",
    "The range and left-out points of the series are kept; the series now starts from this frame's model (Undo brings the previous one back).":
        "保留了序列的范围和被排除的点；序列现在从这一帧的模型开始（撤销可恢复之前的模型）。",
    "The range and left-out points of the series are kept.": "保留了序列的范围和被排除的点。",
    "The fit of {name} ended after another curve was opened; its result was not kept.":
        "{name} 的拟合在打开另一条曲线之后才结束，其结果没有保留。",
    # -- static texts of the windows (2026-10-05) ---------------------------------------------------
    "1D Predict": "一维预测",
    "Load curves → Fit → compare candidates. General V5 proposes multiple compositions. The single-RC specialist requires a known single random cylinder and is experimental.":
        "载入曲线 → 拟合 → 比较候选。通用 V5 会给出多种组成。单 RC 专用模型要求已知只有一个随机取向圆柱，仍是实验性的。",
    "Current cut — original measured points; positive and negative sides are fitted separately.":
        "当前切线——原始测量点；正负两侧分别拟合。",
    "Curve / side": "曲线 / 侧",
    "Stage": "阶段",
    "Ready. Automatic conditions are estimates; candidates are not calibrated probabilities.":
        "就绪。自动条件只是估计；候选不是校准过的概率。",
    # -- review round (2026-10-06) -------------------------------------------------------------------
    "From Analyze: Fit to refine it": "来自分析：请拟合来精修",
    "The halves, the fitting range or the left-out points changed after the search: the χ² of the solutions is of the points before.":
        "搜索之后，所用的半侧、拟合范围或被排除的点已改变：这些解的 χ² 仍是之前那些点上的值。",
    "The file is read-only": "文件是只读的",
    "the In-situ series was not restored: a series is running": "原位序列没有恢复：有序列正在运行",
    "Stop {jobs} first, then open the project.": "请先停止{jobs}，再打开项目。",
    # -- second wave (2026-10-06) ------------------------------------------------------------------
    "Undo the last change of the model, the fitting range or the left-out points, or the last fit (Ctrl+Z)":
        "撤销模型、拟合范围或被排除点的上一次修改，或上一次拟合（Ctrl+Z）",
    "Open a curve: the residuals of the model appear here.": "打开曲线后，这里显示模型的残差。",
    "Choose a method; the button below runs it (Ctrl+Return).": "选择一种方法；用下面的按钮运行（Ctrl+Return）。",
    "One curve at a time here: {name} is open. Fit ▸ Advanced ▸ Fit Many Curves… takes several.":
        "这里一次只打开一条曲线：已打开 {name}。拟合 ▸ 高级 ▸ 拟合多条曲线… 可以处理多条。",
    "A 1D curve: columns q, I and optionally σ (.dat, .txt). Analyze ▸ Send to Fitting opens its cut here.":
        "一维曲线：q、I 两列，可选 σ（.dat、.txt）。在分析里点“发送到拟合”会把切线打开到这里。",
    "No data rows found (at least two numeric columns are needed).": "未找到数据行（至少需要两列数字）。",
    "No (q, I[, σ]) rows could be read.": "没有读取到任何 (q, I[, σ]) 数据行。",
    "Too few valid points (fewer than 2).": "有效数据点太少（少于 2 个）。",
    "The selected frame and its fit appear here.": "所选帧及其拟合会显示在这里。",
    "1D Predict · fit curves & batch": "一维预测 · 拟合曲线与批处理",
    "Prediction settings — saved for single curves and future batch / in-situ runs": "预测设置——保存后用于单条曲线以及之后的批处理 / 原位运行",
    "Auto / unused": "自动 / 不用",
    "Complete composition": "完整组成",
    "Fit method": "拟合方法",
    "Text-file q unit": "文本文件的 q 单位",
    "q sides": "q 的两侧",
    "Each half separately": "两半分别拟合",
    "General V5: improve fit with four numerical steps": "通用 V5：再用四个数值步骤改进拟合",
    "Calibrate intensity amplitudes": "校准强度幅值",
    "The single-RC specialist can adjust particle, background and resolution amplitudes while keeping the neural shape parameters fixed. Broader fitting may still run when curve agreement is poor.":
        "单 RC 专用模型可以调整颗粒、背景和分辨率的幅值，同时保持神经网络给出的形状参数不变。曲线吻合较差时，仍可能运行范围更广的拟合。",
    "General V5 (experimental)": "通用 V5（实验性）",
    "Single RC specialist (experimental)": "单 RC 专用模型（实验性）",
    "Physical fit (numerical)": "物理拟合（数值）",
    "General V5 proposes multiple compositions but remains experimental. The specialist requires Complete composition = one Random cylinder and eligible native CBF counts. Other inputs, fixed resolution and poor curve agreement use numerical fallback; that fallback does not make the specialist a general model. Scores are not probabilities.":
        "通用 V5 会给出多种组成，但仍是实验性的。专用模型要求“完整组成”只有一个随机取向圆柱，并且输入是符合条件的原始 CBF 计数。其他输入、固定的分辨率或曲线吻合较差时会改用数值回退；这种回退并不会让专用模型变成通用模型。得分不是概率。",
    "Fix σ res (nm⁻¹)": "固定 σ res（nm⁻¹）",
    "Fix ν res": "固定 ν res",
    "General V5: 0.007–0.013 nm⁻¹. RC specialist / physical fit: 0.001–0.1 nm⁻¹; fixed resolution uses numerical fallback.":
        "通用 V5：0.007–0.013 nm⁻¹。RC 专用模型 / 物理拟合：0.001–0.1 nm⁻¹；固定分辨率时改用数值回退。",
    "General V5: 5–10. RC specialist / physical fit: 1–20; fixed resolution uses numerical fallback.":
        "通用 V5：5–10。RC 专用模型 / 物理拟合：1–20；固定分辨率时改用数值回退。",
    "Auto: 0.1% peak": "自动：峰值的 0.1%",
    "Auto: measured max": "自动：测量最大值",
    "Relative σ (if missing)": "相对 σ（文件里没有时）",
    "Absolute σ floor": "σ 绝对下限",
    "Intensity normalizer": "强度归一化值",
    "Discover combinations": "搜索的组合数",
    "Condition best combinations": "细化的最佳组合数",
    "In-situ · 1D prediction parameters": "原位 · 一维预测参数",
    "In-situ prediction parameters": "原位预测参数",
    "Save creates a new settings snapshot for future frames. Completed frames are unchanged.":
        "保存会为之后的帧建立新的设置快照；已完成的帧不变。",
    "Edit known components / resolution, or leave them automatic.": "编辑已知的组分 / 分辨率，或保持自动。",
    "S: sphere; RC: random cylinder; VC: vertical cylinder. Repeated types are distinct components.":
        "S：球；RC：随机取向圆柱；VC：竖直圆柱。重复的类型是不同的组分。",
    "Natural-log RMSE on original positive-intensity observations only.": "只在原始的正强度观测点上计算的自然对数 RMSE。",
    "RMS of (forward − observed)/sigma, including negative observations; not a calibrated probability.":
        "(正演 − 观测)/sigma 的均方根，包括负的观测值；不是校准过的概率。",
    "Add curves (or Use current cut), then Fit curve.\nSelect a candidate to see its fit here.":
        "添加曲线（或使用当前切线），然后拟合曲线。\n选一个候选，在这里查看它的拟合。",
    "Model + amplitude": "模型 + 幅值",
    "Neural model": "神经网络模型",
    "Numerical fallback": "数值回退",
    "Stage: {stage}": "阶段：{stage}",
    "Reason: {reason}": "原因：{reason}",
    "Natural-log RMSE over the measured curve, including measurement noise. Review peak positions, overall shape and residuals; no mandatory cutoff is applied.":
        "在整条测量曲线上计算的自然对数 RMSE，包含测量噪声。请检查峰位、整体形状和残差；不设强制阈值。",
    "Lengths: nm. sigma_R/h/D: relative standard deviations.\nMixture weights are not posterior probabilities.\nResolution sigma: nm^-1; nu: dimensionless.":
        "长度：nm。sigma_R/h/D：相对标准差。\n混合权重不是后验概率。\n分辨率 sigma：nm^-1；nu：无量纲。",
    "Select one or more 1D curves": "选择一条或多条一维曲线",
    "{count} file(s): {names}": "{count} 个文件：{names}",
    "Fit {count} files": "拟合 {count} 个文件",
    "Current cut — native points, q converted to nm⁻¹ by the fitting workspace.": "当前切线——原始数据点，q 由拟合工作区换算为 nm⁻¹。",
    "A fitting / in-situ job is already running. Finish or cancel it first.": "已有拟合 / 原位任务在运行。请先等它完成或取消。",
    "Load a curve or select files first": "请先载入曲线或选择文件",
    "Starting the experimental single-RC specialist… Checking the known composition and input scope.":
        "正在启动实验性的单 RC 专用模型…正在检查已知组成和输入范围。",
    "Starting numerical physical fitting…": "正在开始数值物理拟合…",
    "Loading experimental General V5… First run includes loading and compilation.": "正在载入实验性的通用 V5…第一次运行包括载入和编译。",
    "Finished in {seconds} s · {failures} failed files. {quality}": "用时 {seconds} s 完成 · {failures} 个文件失败。{quality}",
    "{count} candidates saved. Review curve shape and residuals; observed-data scores include noise and are not probabilities.":
        "已保存 {count} 个候选。请检查曲线形状和残差；基于观测数据的得分包含噪声，不是概率。",
    "Settings saved. Existing in-situ recipes keep their captured settings.": "设置已保存。已有的原位配方保留它们记录时的设置。",
    "Cancelling… completed file results remain saved.": "正在取消…已完成文件的结果仍会保留。",
    "Saved settings v{version} for future frames.": "已为之后的帧保存设置 v{version}。",
}

__all__ = ["FITTING_ZH"]
