"""Chinese strings for the Predict parameters diagnostics and exports."""

PREDICT_REPORT_ZH = {
    "Network-proposal effective sample size {ess}.": "网络候选样本的有效样本数为 {ess}。",
    "Values in Model with reported 68 % intervals. Least-squares errors are used where available — {values}.":
        "模型中的数值及其 68 % 区间。有最小二乘误差的参数按该误差计算区间：{values}。",
    "the posterior's sampling did not converge: the solutions are compared by χ² and their number of values "
    "(BIC), and the intervals use least-squares errors where available": "后验抽样未收敛：按 χ² 和参数个数（BIC）比较各解，有最小二乘误差的参数用该误差计算区间",
    "± half the width of the reported 68 % interval [{low}, {high}] from the network proposals.":
        "± 为网络候选样本给出的 68 % 区间 [{low}, {high}] 宽度的一半。",
    "Sampled central 68 % interval [{low}, {high}].": "抽样得到的中央 68 % 区间 [{low}, {high}]。",
    "Export each solution's probability or BIC weight, χ²ᵣ, flags, least-squares values, posterior quantiles and fitted curves.":
        "导出各解的概率或 BIC 权重、χ²ᵣ、标记、最小二乘值、后验分位数和拟合曲线。",
    "Measured q_z for level-1 flags (nm⁻¹)": "用于第一层提示的实测 q_z（nm⁻¹）",
    "Measured q_z for level-1 flags must be within (0, 10] nm⁻¹.": "用于第一层提示的实测 q_z 须在 (0, 10] nm⁻¹ 内。",
    "Network-proposal log evidence {z} ± {error} (bootstrap Monte Carlo error).":
        "网络候选样本的对数证据 {z} ± {error}（自助法估计的蒙特卡罗误差）。",
    "± half the width of the 68 % posterior interval [{low}, {high}], extended to include the least-squares value.":
        "± 为 68 % 后验区间 [{low}, {high}] 宽度的一半，区间已扩展至包含最小二乘值。",
    "After Refine or Search, ± gives the least-squares error. For Predict parameters, "
    "± gives half the width of the reported 68 % interval.": "精修或搜索后，± 为最小二乘误差。预测参数后，± 为所报告的 68 % 区间宽度的一半。",
    "Values in Model with their 68 % posterior intervals extended to include the least-squares value — {values}.":
        "模型中的数值及其 68 % 区间（区间已扩展至包含最小二乘值）：{values}。",
    "The particles in Model uses the families in radius order, with their paracrystals. "
    "For a choice of particles, the network proposes compositions. Evidence or BIC weights rank the fits.":
        "使用模型中的颗粒时，按半径顺序采用各颗粒类型及其准晶设置。若允许选择颗粒，网络提出候选组成，按证据或 BIC 权重排列拟合结果。",
    "The network proposes particles and sizes. Least squares refines the values. Evidence compares the solutions, "
    "with BIC weights if sampling fails. Refine afterwards to use the limits in Model.":
        "网络提出颗粒组成和尺寸，再用最小二乘精修数值。按证据比较各解；抽样失败时用 BIC 权重。之后可在模型的范围内精修。",
    "{name}: intervals not calibrated at this q_z·R (DWBA data)": "{name}：在此 q_z·R 处区间未经校准（DWBA 数据）",
    "{name}: the second level does not describe this q_z·R; use the third level or the four-term DWBA":
        "{name}：第二层模型无法描述此 q_z·R，请使用第三层或四项 DWBA",
    "{name}: may stand for a sphere of this radius at high q_z·R": "{name}：在较高 q_z·R 处可能代表具有该半径的球",
    "interval coverage: {name}": "区间覆盖率：{name}",
    "level 2 fails: {name}": "第二层模型失效：{name}",
    "possible sphere: {name}": "可能是球：{name}",
    "Reflected beam present in the data": "数据中有反射束",
    "BIC weight": "BIC 权重",
    "BIC weight {weight} among the solutions compared.": "在所比较的解中，BIC 权重为 {weight}。",
    "posterior median": "后验中位数",
    "least-squares\nvalue": "最小二乘值",
    "Posterior median. Hover over a value for its central 68 % interval.": "后验中位数。将鼠标悬停在数值上可查看中央 68 % 区间。",
    "least-squares value": "最小二乘值",
    "Posterior median [central 68 % interval]": "后验中位数 [中央 68 % 区间]",
    "Posterior median [central 68 % interval] — {values}.": "后验中位数 [中央 68 % 区间]：{values}。",
    "Composition BIC weight {weight}.": "组成的 BIC 权重为 {weight}。",
    "Composition log evidence {z} ± {error} (bootstrap Monte Carlo error).": "组成的对数证据 {z} ± {error}（自助法蒙特卡罗误差）。",
    "Mode log evidence {z} ± {error} (bootstrap Monte Carlo error).": "局部模式的对数证据 {z} ± {error}（自助法估计的蒙特卡罗误差）。",
    "Composition probability {p} (68 % Monte Carlo range {low}–{high}).": "组成概率 {p}（68 % 蒙特卡罗范围 {low}–{high}）。",
    "Effective sample size {ess}.": "有效样本数 {ess}。",
    "Posterior quantiles use tempered importance weights.": "后验分位数使用调温的重要性权重。",
    "Incomplete. BIC weights refer to the finished solutions.": "未完成。BIC 权重仅针对已完成的解。",
    "Stopped. {solutions} kept after sampling {finished} of {total} solutions. BIC weights refer to the finished solutions.":
        "已停止。已对 {total} 个解中的 {finished} 个抽样，保留 {solutions}。BIC 权重仅针对已完成的解。",
    "{model} (BIC weight {p}), {params} nm, χ²ᵣ = {chi}": "{model}（BIC 权重 {p}），{params} nm，χ²ᵣ = {chi}",
    "Each candidate has an evidence-based probability or a BIC weight. logRMSE and RMS/σ include the measurement noise.":
        "每个候选解给出由证据计算的概率或 BIC 权重。logRMSE 和 RMS/σ 包含测量噪声。",
}
