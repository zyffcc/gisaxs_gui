"""Chinese wording for model probabilities and competing GISAXS parameter values."""

SAME_MODEL_ZH = {
    "probability {weight}": "概率 {weight}",
    "BIC weight {weight}": "BIC 权重 {weight}",
    "model probability {weight} (this solution {share})": "模型概率 {weight}（此解 {share}）",
    "model BIC weight {weight} (this solution {share})": "模型 BIC 权重 {weight}（此解 {share}）",
    "model probability": "模型概率",
    "model BIC weight": "模型 BIC 权重",
    "this solution": "此解",
    "Predict parameters: GIMaP's network proposes the particles (one or two populations of spheres and upright "
    "cylinders) and their values. Each solution is refined by least squares. Select a solution to see its fitted curve.":
        "预测参数：GIMaP 的网络给出粒子模型（由一或两个球形或直立圆柱形粒子群组成）及其参数值，"
        "再用最小二乘法精修每个解。选择一个解可查看其拟合曲线。",
    "p is the posterior probability of this solution among the compositions tried. The most probable solution is "
    "listed first. A model's probability is the sum of the probabilities of its solutions.":
        "p 是此解在所尝试的各组成中的后验概率。概率最高的解排在首行。一个模型的概率等于该模型各解的概率之和。",
    "p: the posterior probability of the solution; probabilities are relative to the compositions that were tried. "
    "A model's probability is the sum of the probabilities of its solutions.":
        "p 是此解在所尝试的各组成中的后验概率。一个模型的概率等于该模型各解的概率之和。",
    "most probable fit {model} R {radius} nm ({weight}, χ²ᵣ {chi2})":
        "最可能的拟合解 {model}，R {radius} nm（{weight}，χ²ᵣ {chi2}）",
    "{model} R {radius} nm ({weight}, χ²ᵣ {chi2})": "{model}，R {radius} nm（{weight}，χ²ᵣ {chi2}）",
    "Competing models and their probabilities: {models}. Select a row to compare their fits.":
        "候选模型及其概率：{models}。选择表格行可比较拟合曲线。",
    "Competing models and their BIC weights: {models}. Select a row to compare their fits.":
        "候选模型及其 BIC 权重：{models}。选择表格行可比较拟合曲线。",
    "The curve alone does not decide the model.": "仅凭这条曲线无法确定模型。",
    "Model probabilities: {models}.": "各模型的概率：{models}。",
    "Model BIC weights: {models}.": "各模型的 BIC 权重：{models}。",
    "The curve alone does not decide the model. Model probabilities: {models}.":
        "仅凭这条曲线无法确定模型。各模型的概率：{models}。",
    "The curve alone does not decide the model. Model BIC weights: {models}.":
        "仅凭这条曲线无法确定模型。各模型的 BIC 权重：{models}。",
    "{n} solutions of this model with overlapping 68 % intervals.": "该模型有 {n} 个解的 68 % 区间重叠。",
    "{fit_values}. {n} solutions of this model with overlapping 68 % intervals.":
        "{fit_values}。该模型有 {n} 个解的 68 % 区间重叠。",
    "{model}: {sizes}, χ²ᵣ = {chi2}, {fit_weight}": "{model}：{sizes}，χ²ᵣ = {chi2}，{fit_weight}",
    "Parameter values": "参数值",
    "Two solutions of the same model give {name} values of {first_value} and {second_value}.":
        "同一模型的两个解给出的 {name} 分别为 {first_value} 和 {second_value}。",
    "{value} nm (68 % interval unavailable)": "{value} nm（无 68 % 区间）",
    "The curve does not decide between these values.": "仅凭这条曲线无法确定应取哪个值。",
    "In Fitting, refine from each start and compare the fitted curves and parameter values.":
        "在拟合工作区分别从各初值开始精修，比较拟合曲线和参数值。",
}
