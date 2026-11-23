"""CIF composition checks and sample-angle evidence."""

PHASE_GEOMETRY_ZH = {
    "Expanded sites per cell: {sites}. CIF sum formula: {formula}. Z = {z}.":
        "每晶胞展开的原子：{sites}。CIF 总化学式：{formula}。Z = {z}。",
    "Expanded sites differ from the CIF sum formula × Z. Check the site model before using its reflection intensities.":
        "展开后的原子数与 CIF 总化学式乘以 Z 不符。使用计算反射强度前，请核查原子占位模型。",
    "Expanded site ratios differ from the CIF sum formula. Z was not given.":
        "展开后的原子数之比与 CIF 总化学式不符。文件未给出 Z。",
    "The CIF formula-unit count Z must be positive. Check Z before using the bulk formula.":
        "CIF 每晶胞的化学式单元数 Z 必须为正。使用总化学式前请核对 Z。",
    "Roll assuming a surface-normal fibre axis": "假定纤维轴沿表面法线得到的滚转角",
    "The rings show a fibre tilt without an independently resolved surface tilt.":
        "衍射环显示纤维轴倾斜，表面倾角缺乏独立测量依据。",
    "The run assumes that the fibre axis is the surface normal.": "本次运行假定纤维轴沿表面法线。",
    "Single specular-like spot": "单个疑似镜面反射斑",
    "a single specular-like spot": "单个疑似镜面反射斑",
    "the shadow edge": "阴影边缘",
    "the Yoneda band": "Yoneda 带",
    "The estimate draws {weight}% of its weight from {feature}.": "{feature}占该估计权重的 {weight}%。",
    "{reference} and frame αi = {value}° ({method}) differ by more than 3σ.":
        "{reference} 与帧测量值 αi = {value}°（{method}）相差超过 3σ。",
    "{reference} and frame αi = {value}° ({method}) agree within 3σ.":
        "{reference} 与帧测量值 αi = {value}°（{method}）在 3σ 内相符。",
    "frame αi = {value}° ({method}).": "帧测量值 αi = {value}°（{method}）。",
    "The frame estimate is unresolved ({method}).": "未能确定帧测量值（{method}）。",
    "combined": "综合估计",
    "horizon": "阴影边缘",
    "specular": "镜面反射斑",
    "yoneda": "Yoneda 带",
    "disagreement": "估计不一致",
    "The reduction uses the measured angle.": "约化采用测得的角度。",
    "Select Keep entered αi or pass --keep-incidence to override.":
        "若要保留输入值，请选择“保留输入的 αi”或使用 --keep-incidence。",
    "The measured angle supplies the missing incidence.": "采用测量值补全入射角。",
    "The reduction retains the nominal angle.": "约化采用参考角度。",
    "The user selected the nominal angle.": "用户选择保留参考角度。",
    "{feature} supplies {weight}% of the incidence weight.": "入射角估计中，{feature}占权重的 {weight}%。",
    "unspecified": "未给出",
    "unavailable": "不可用",
}
