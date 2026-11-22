"""Analyze's phase-aware crystal orientation fit."""

CRYSTAL_ORIENTATION_ZH = {
    "Crystal orientation (from phase)": "晶体取向（依据物相）",
    "Fit Crystal Orientation": "拟合晶体取向",
    "Show ring": "显示衍射环",
    "Fit the assigned rings of each phase together, using each ring’s local background.":
        "各物相的已指认衍射环联合拟合，每个环使用其局部背景。",
    "Export the displayed ring as CSV and the phase fits as JSON": "将所示衍射环导出为 CSV，各物相的拟合结果导出为 JSON",
    "Fitting crystal orientations …": "正在拟合晶体取向…",
    "No assigned rings could be fitted.": "没有可拟合的已指认衍射环。",
    "Orientation determined": "已确定取向",
    "Orientation not determined": "未能确定取向",
    "Mosaic FWHM {width}° · oriented {oriented}% · isotropic {isotropic}%":
        "镶嵌展宽（半高宽）{width}° · 取向晶粒 {oriented}% · 各向同性晶粒 {isotropic}%",
    "Runner-up {axes} · ΔBIC {delta} · joint χ²red {joint} · ring χ²red {ring}":
        "次优轴 {axes} · ΔBIC {delta} · 联合拟合约化 χ² {joint} · 此环约化 χ² {ring}",
    "Crystal Hermans f {crystal} · measured ring f {ring}": "晶体 Hermans 因子 f {crystal} · 此环实测 f {ring}",
    "Joint fit": "联合拟合",
    "Isotropic": "各向同性",
    "Shares of the oriented population {shares}": "取向晶粒中各纤维轴所占比例 {shares}",
    "Free reflection weights": "自由拟合的反射权重",
    "Kinematic reflection weights": "运动学反射权重",
    "No usable ring assigned to this phase.": "该物相没有可用的已指认衍射环。",
    "Use a distinct name for each visible phase.": "请为每个显示的物相使用不同的名称。",
    "The radial curve is unavailable.": "没有可用的径向曲线。",
    "no assigned rings": "没有已指认的衍射环",
    "no assigned ring has a usable profile and a reflection in its extraction window":
        "没有已指认的衍射环同时具有可用剖面和提取窗口内的反射",
    "the isotropic model has the lowest BIC": "各向同性模型的 BIC 最小",
    "joint or ring reduced χ² exceeds 5": "联合拟合或单环的约化 χ² 大于 5",
    "the competing orientation is within 6 BIC units or unavailable": "次优模型与最优模型的 BIC 差小于 6，或没有可比较的模型",
    "fit parameters are not identifiable": "无法由数据独立确定各拟合参数",
    "the width or isotropic limit is at a bound": "展宽达到边界或模型达到各向同性极限",
}
