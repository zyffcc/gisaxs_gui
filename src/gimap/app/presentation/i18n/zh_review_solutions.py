"""Chinese translations: first-user review fixes (solutions).

What GIMaP says of a solution of Predict parameters (``shared/fit_names.py``: one name per quantity, one sentence
per flag; ``fitting/presentation/single/predict_texts.py``; the automatic analysis's fit table): its values with
their intervals, its probability, the points its χ²ᵣ is of, its notes. The names of the Model table (Tail ν, σ_q,
σR/R (1)) are names and stay as they are."""

REVIEW_SOLUTIONS_ZH: dict[str, str] = {
    # shared/fit_names.py
    "the curve says little about {name} (its 68 % interval spans most of its allowed range)":
        "曲线对 {name} 约束很少（其 68 % 区间覆盖了允许范围的大部分）",
    "{name} stopped at a bound of its range": "{name} 停在其范围的边界上",
    "{name} are strongly correlated": "{name} 强相关",
    "few effective samples of the posterior: the intervals are rough, and wider than a converged sampling would give":
        "后验的有效样本很少：区间只是粗略的，比收敛的抽样给出的更宽",
    "the posterior's sampling did not converge: the solutions are compared by χ² and their number of values (BIC), "
    "and the intervals are the least-squares errors":
        "后验抽样没有收敛：各个解按 χ² 和数值个数（BIC）比较，区间取自最小二乘误差",
    "the values are those of a fit with the model error added to σ: the fit with the curve's σ alone lay outside "
    "the posterior":
        "这些数值来自把模型误差加进 σ 的拟合：只用曲线自身 σ 的拟合落在后验之外",
    "population {name} has Scale 0: this solution does not use it": "第 {name} 种颗粒的比例为 0：这个解没有用到它",
    "{name}: q_z·R is above 1.7 at the cut's q_z, where the form factor at q_z = 0 (the default effective interface) "
    "biases the radius, usually too small; give both angles of the cut (αi, αf) so that spheres are fitted with the "
    "exit-reflected path (random cylinders stay at q_z = 0)":
        "{name}：在切线的 q_z 处 q_z·R 大于 1.7，此时在 q_z = 0 处计算形状因子（默认的有效界面近似）会使半径有偏差，"
        "通常偏小；请给出切线的两个角度（αi、αf），使球按出射反射路径拟合（随机取向圆柱仍取 q_z = 0）",
    "the refined value lies at the edge of the posterior: the model describes this curve only within the model "
    "error ({names})":
        "精修值位于后验的边缘：模型只在模型误差之内描述这条曲线（{names}）",
    "χ²ᵣ = Σ((I − model)² / σ²) / (N − number of free values), over the N fitted points, with the curve's σ":
        "χ²ᵣ = Σ((I − 模型)² / σ²) / (N − 自由数值的个数)，对拟合的 N 个点求和，σ 取曲线自身的 σ",
    "{fitted} of {given} points": "{given} 个点中的 {fitted} 个",
    "{count} below the network's q window": "{count} 个低于网络的 q 窗口",
    "{count} without a usable value": "{count} 个没有可用的值",
    "Background, beam tail and resolution: {notes}": "背景、光束尾项和分辨率：{notes}",
    "only this composition was tried": "只尝试了这一种组成",
    # fitting/presentation/single/predict_texts.py
    "Values in Model with their 68 % posterior intervals (refined value [interval]): {values}.":
        "模型中的数值及其 68 % 后验区间（精修值 [区间]）：{values}。",
    "Values in Model with 68 % intervals from the least-squares errors (refined value [interval]): {values}.":
        "模型中的数值及由最小二乘误差得到的 68 % 区间（精修值 [区间]）：{values}。",
    "Only this composition was tried: no probability.": "只尝试了这一种组成：不给概率。",
    "Probability {p} among the solutions of the one composition that was tried.":
        "在所尝试的那一种组成的各个解中，概率为 {p}。",
    "Probability {p}, relative to the compositions that were tried.": "概率 {p}，相对于所尝试的各种组成。",
    "χ²ᵣ is over {points}.": "χ²ᵣ 的点数：{points}。",
    "Differs from solution {other} in: {values}.": "与第 {other} 个解的不同之处：{values}。",
    "{name} {value} (there {other})": "{name} {value}（那个解为 {other}）",
    "Refine fitted {refined} points; Predict parameters fitted {fitted} of {given} ({why}): the two χ²ᵣ are not of "
    "the same points.":
        "精修拟合了 {refined} 个点；预测参数拟合了 {given} 个点中的 {fitted} 个（{why}）：两个 χ²ᵣ 不是同一批点的。",
    "another selection of points": "选取的点不同",
    "p: the posterior probability of the solution; probabilities are relative to the compositions that were tried.":
        "p：这个解的后验概率；概率是相对于所尝试的各种组成而言的。",
    "Every population's value, in the order of the radii; –: a population without a paracrystal.":
        "每种颗粒的数值，按半径从小到大；–：没有次晶的颗粒。",
    "No error: the value is at a bound of its range.": "没有误差：数值在其范围的边界上。",
    "No error: the Scale of this population is 0, so its values change nothing.":
        "没有误差：这种颗粒的比例为 0，它的数值不改变曲线。",
    "No error: Tail A is 0, so this value changes nothing.": "没有误差：尾 A 为 0，这个数值不改变曲线。",
    "No error: the curve does not constrain this value.": "没有误差：曲线不约束这个数值。",
    "1 solution": "1 个解",
    "{count} solutions": "{count} 个解",
    # fitting/presentation/single: results.py, run.py, page.py
    "{solutions} below; the best is in Model. Fit (Refine) gives its quality and the error of every value.":
        "下面有 {solutions}；最好的已放入模型。点“拟合”（精修）会给出它的质量和每个值的误差。",
    "{solutions} to compare.": "{solutions}可以比较。",
    "{solutions} to compare": "{solutions}可以比较",
    "{solutions} · best {model} {chi} {value}": "{solutions} · 最佳 {model} {chi} {value}",
    "{fitted} of {given} points fitted": "拟合 {given} 个点中的 {fitted} 个",
    "{value} over {points}": "{value}，{points}",
    "{solutions}; the best, {model} (χ²ᵣ {chi2}), is in Model. Fit again to refine it, or choose another in Results.":
        "{solutions}；最好的 {model}（χ²ᵣ {chi2}）已放入模型。再拟合一次来精修，或在结果里选择另一个。",
    "{method}: {solutions}": "{method}：{solutions}",
    "χ²ᵣ of Analyze's fit (Predict parameters), over the points Analyze fitted. Fit (Refine) gives it for the points "
    "here.":
        "分析页拟合（预测参数）的 χ²ᵣ，按分析页所拟合的点计算。在这里拟合（精修）给出这里各点的 χ²ᵣ。",
    # assistant/presentation: guided_gisaxs.py, guided_details.py
    "p: the posterior probability of the solution, relative to the compositions that were tried; the solutions are "
    "sorted by it, most probable first. Another solution with p of 0.2 or more: the curve alone does not decide the "
    "model.":
        "p：这个解的后验概率，相对于所尝试的各种组成；各个解按它排序，最可能的在前。另有解的 p 不低于 0.2 时：仅凭这条曲线"
        "无法确定模型。",
    "R (nm): the particle radius of every population, in the order of the radii.":
        "R (nm)：每种颗粒的半径，按半径从小到大。",
    "h (nm): the cylinder height of every population, in the order of the radii (– for a sphere).":
        "h (nm)：每种颗粒的圆柱高度，按半径从小到大（球为 –）。",
    "D (nm): the paracrystal distance of every population, in the order of the radii (– without a paracrystal).":
        "D (nm)：每种颗粒的次晶距离，按半径从小到大（没有次晶为 –）。",
}
