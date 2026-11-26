"""Chinese text for Fitting ▸ Rank Incidence Angles… (views/angle_rank_view.py, single/angle_rank_dialog.py)."""

ANGLE_RANK_ZH = {
    "Ran&k Incidence Angles…": "入射角排序(&K)…",
    "Which incidence angle to measure next: what an exposure at each angle would add to the fit of this page, and a "
    "record of the angle chosen":
        "下一次测哪个入射角：在每个角度曝光一次能为本页的拟合增加多少信息，并记录所选的角度",
    "Rank Incidence Angles": "入射角排序",
    "Which incidence angle to measure next: what an exposure at each angle would add to what the fit in Single "
    "analysis already knows (Physics and Methods 9.20). The score is only as good as the model.":
        "下一次测哪个入射角：在每个角度曝光一次，能在单条分析中的拟合已知信息之上增加多少（物理与方法 9.20 节）。"
        "评分的好坏取决于模型。",
    "The candidate incidence angles in degrees, separated by commas": "候选入射角（度），用逗号分隔",
    "The exposure of one measurement at a candidate angle (s)": "在候选角度测一次的曝光时间（s）",
    "the curve's own angle and exposure": "曲线自己的角度和曝光时间",
    "What was measured so far, as angle:exposure pairs (0.4:10, 0.1:1); empty: the angle and exposure of the curve "
    "fitted":
        "到目前为止测过的，写成 角度:曝光 对（0.4:10, 0.1:1）；留空：所拟合曲线的角度和曝光时间",
    "The exposure the fitted curve's intensity is per (s): its counts per pixel were collected in this time":
        "所拟合曲线的强度对应的曝光时间（s）：它每像素的计数是在这段时间内收集的",
    "The model-error floor f in % of each point: beyond 1/f² counts the model, not the photons, limits the error":
        "每个点的模型误差下限 f（%）：计数超过 1/f² 后，限制误差的是模型而不是光子",
    "The height of the beam at the sample (mm): the footprint is height / sin αi": "样品处光束的高度（mm）：光斑长度为 高度 / sin αi",
    "The length of the sample along the beam (mm): the footprint ends there": "样品沿光束方向的长度（mm）：光斑到此为止",
    "mm beam on": "mm 光束，样品",
    "mm sample": "mm",
    "Candidate angles (°)": "候选角度（°）",
    "Exposure per angle (s)": "每个角度的曝光（s）",
    "Measured so far": "已测",
    "Exposure of the curve (s)": "曲线的曝光（s）",
    "Model-error floor (%)": "模型误差下限（%）",
    "Footprint": "光斑",
    "Particles buried in a film": "埋在膜中的颗粒",
    "A fraction of the particles lies in a film on the substrate (a cellulose film under a spray), fewer with depth as "
    "exp(−z/ζ): the angles below and above the film's critical angle tell how deep.":
        "一部分颗粒位于基底上的膜中（喷涂下的纤维素膜），数量随深度按 exp(−z/ζ) 减少：膜临界角上下的角度能告诉它们埋得多深。",
    "Some particles are buried in a film": "部分颗粒埋在膜中",
    "δ of the film at the wavelength of the curve (cellulose at 10 keV: 3.0e-6)": "膜在曲线波长下的 δ（10 keV 的纤维素：3.0e-6）",
    "β of the film at the wavelength of the curve": "膜在曲线波长下的 β",
    "The share of the particles in the film (%)": "膜中颗粒所占的比例（%）",
    "ζ: how deep the buried particles lie, the depth where their number falls to 1/e (nm)":
        "ζ：埋入颗粒的深度，数量降到 1/e 处的深度（nm）",
    "Film δ": "膜 δ",
    "Film β": "膜 β",
    "Buried share (%)": "埋入比例（%）",
    "Depth ζ (nm)": "深度 ζ（nm）",
    "Rank Angles": "排序",
    "Score every candidate angle with the fit in Single analysis": "用单条分析中的拟合给每个候选角度打分",
    "Rank": "排名",
    "αi (°)": "αi（°）",
    "Gain": "增益",
    "Counts": "计数",
    "Note of the operator (optional)": "操作员备注（可选）",
    "Record Choice": "记录选择",
    "Write the angle chosen in the table, every candidate's scores and the fit into angle_decisions.jsonl next to the "
    "curve (one line per choice)":
        "把表中选中的角度、每个候选的得分和拟合写入曲线旁的 angle_decisions.jsonl（每次选择一行）",
    "Write the angles as numbers above 0 and below 90, separated by commas.": "请把角度写成大于 0、小于 90 的数，用逗号分隔。",
    "Write what was measured as angle:exposure pairs, separated by commas (0.4:10, 0.1:1).":
        "请把已测的写成 角度:曝光 对，用逗号分隔（0.4:10, 0.1:1）。",
    "Load a curve and fit it in Single analysis first.": "请先载入一条曲线，并在单条分析中拟合。",
    "Best next: αi {angle}° (gain {gain} nats). {count} angles scored in {seconds} s.":
        "下一个最佳：αi {angle}°（增益 {gain} nat）。{count} 个角度用时 {seconds} s。",
    "αi {angle}° recorded in {name}": "αi {angle}° 已记录在 {name}",
    "σ/θ of {name}": "{name} 的 σ/θ",
    "Could not write the record: {error}": "无法写入记录：{error}",
    "buried fraction": "埋入比例",
    "burial depth ζ": "埋入深度 ζ",
    "The fit has no incidence angle: Rank Incidence Angles needs αi of the curve (Model ▸ Geometry).":
        "这个拟合没有入射角：入射角排序需要曲线的 αi（模型 ▸ 几何）。",
    "Rank Incidence Angles needs a known substrate (Model ▸ Substrate): the four-term DWBA gives how the intensity "
    "changes with the incidence angle.":
        "入射角排序需要已知的基底（模型 ▸ 基底）：由四项 DWBA 给出强度如何随入射角变化。",
    "The model has no free shape or spacing parameter to rank angles for.": "模型中没有可用于排序的自由形状或间距参数。",
}

__all__ = ["ANGLE_RANK_ZH"]
