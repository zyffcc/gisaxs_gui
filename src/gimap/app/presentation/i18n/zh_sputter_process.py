"""Chinese texts of the Sputter workspace's component separation and growth-process analysis.

Keys are the exact English of ``features/sputter`` (``presentation/views/sputter_main_view.py``,
``views/sputter_page_view.py``, ``components_view.py``, ``process_tab.py``, ``domain/process.py``; the growth-law
prior of Analyze options: ``application/client.py``, ``result_tabs.py``); templates keep
their ``{fields}``. Numbers, units, element and material names are not translated.
"""

SPUTTER_PROCESS_ZH = {
    # -- the picture switch and the Components picture (sputter_main_view.py, components_view.py) -------------------
    "The model image of the solution shown split into its physical terms (grains, film, substrate, …), and how "
    "each term's share changes.":
        "把所选解的模型图拆成各个物理项（晶粒、膜、基底……），并给出每一项占比的变化。",
    "{name} · {share}": "{name}：{share}",
    "Share of the model counts": "占模型计数的比例",
    "This result does not split the model into its physical terms: analyze it again with the current analysis "
    "package.": "这个结果没有把模型拆成物理项：请用当前的分析包重新分析。",
    "Also: {names}": "另有：{names}",
    # -- the physical terms (domain/process.py) -----------------------------------------------------------------------
    "Grains and islands": "晶粒与岛",
    "Large clusters": "大团簇",
    "Film fluctuation and network": "膜的涨落与网络",
    "Voids under the film": "膜下空洞",
    "Buried clusters": "埋入的团簇",
    "Metal-rich domains in the polymer": "聚合物中的富金属畴",
    "Rough top of the continuous layer": "连续层的粗糙上表面",
    "Clusters in the pores": "孔内团簇",
    "Clusters on the cap layer": "盖层上的团簇",
    "Metal network in the pores": "孔内金属网络",
    "Substrate seen through the metal": "透过金属看到的基底",
    "Fluorescence": "荧光",
    "Beam-path background": "光路背景",
    # -- the stages (domain/process.py) -------------------------------------------------------------------------------
    "Separate islands": "孤立的岛",
    "Coalescence": "聚并",
    "Continuous film": "连续膜",
    "After the deposition": "停镀之后",
    "Clusters": "团簇",
    "Metal film": "金属膜",
    # -- the growth descriptors (domain/process.py) -------------------------------------------------------------------
    "Metal amount at coverage 0.5": "覆盖率达到 0.5 时的金属量",
    "Metal amount at which the film forms": "成膜时的金属量",
    "Distance D at 1 nm of metal (D = D₁ h^p)": "1 nm 金属时的间距 D（D = D₁ h^p）",
    "Growth exponent p of D": "D 的生长指数 p",
    "Radius R at 1 nm of metal (R = R₁ h^q)": "1 nm 金属时的半径 R（R = R₁ h^q）",
    "Growth exponent q of R": "R 的生长指数 q",
    "Distance D once the film has formed": "成膜后的间距 D",
    "Radius R once the film has formed": "成膜后的半径 R",
    "Metal amount when the deposition stopped": "停镀时的金属量",
    "Coarsening after the stop (change of ln D per decade of time)": "停镀后的粗化（每十倍时间 ln D 的变化）",
    "Metal amount at half of the cap layer term": "盖层项达到一半时的金属量",
    "Metal amount at half of the film term": "膜项达到一半时的金属量",
    "Metal amount at the last window": "最后一个时间窗的金属量",
    "Median fit deviation": "拟合偏差中位数",
    "Steps back in R, D or coverage between windows": "相邻时间窗间 R、D 或覆盖率的回退次数",
    # -- Details ▸ Growth Process (sputter_page_view.py, process_tab.py) ---------------------------------------------
    "Growth Process": "生长过程",
    "Stages of the growth": "生长阶段",
    "Growth laws": "生长规律",
    "Physical terms of the model": "模型的物理项",
    "From the solutions: {source}": "依据的解：{source}",
    "joint fit of the whole scan": "整条扫描的联合精修",
    "smooth path through the windows' solutions": "贯穿各时间窗的平滑路径",
    "Metal amount (nm)": "金属量（nm）",
    "Quantity": "量",
    "This scan": "这条扫描",
    "{system}, {scans} scans": "{system}，{scans} 条扫描",
    "Here": "位置",
    "Same metal and substrate": "同一金属与基底",
    "within the range": "在范围内",
    "below the range": "低于范围",
    "above the range": "高于范围",
    "{median} ({low} – {high})": "{median}（{low} – {high}）",
    "Metal amounts are equivalent thicknesses of the metal (nm), independent of the X-ray wavelength. The range is "
    "that of the same metal on the same substrate over the scans the analysis package was built from (development "
    "reference: no independent characterisation).":
        "金属量是金属的等效厚度（nm），与 X 射线波长无关。范围取自构建分析包时所用的、同一金属在同一基底上的扫描"
        "（开发参考：没有独立表征）。",
    "This result has no analysis of the growth process: analyze the scan again with the current analysis package.":
        "这个结果没有生长过程分析：请用当前的分析包重新分析这条扫描。",
    "The analysis of the growth process is made for in-situ scans.": "生长过程分析只针对原位扫描。",
    # -- Details ▸ Analyze options: the growth-law prior (sputter_main_view.py, client.py, result_tabs.py) -----------
    "Growth-law prior": "生长规律先验",
    "On": "开",
    "Off": "关",
    "On where it passed a held-out test (Ag on PS), off for the other systems.":
        "在通过留一检验的体系（PS 上的 Ag）打开，其他体系关闭。",
    "Before the film forms, the whole-scan refinement keeps the size and the distance near the growth laws found for "
    "the metal on this substrate.":
        "成膜之前，联合精修让尺寸和间距靠近同一金属在这种基底上的生长规律。",
    "The whole-scan refinement without the growth laws.": "联合精修不用生长规律。",
    "The growth-law prior must be auto, on or off.": "生长规律先验必须是 auto、on 或 off。",
    "Growth-law prior ({setting}): used, laws of {system}": "生长规律先验（{setting}）：已使用，{system} 的规律",
    "Growth-law prior ({setting}): not used": "生长规律先验（{setting}）：未使用",
    # -- Details ▸ Report, the import row (result_tabs.py) --------------------------------------------------------
    "{count} damaged frames left out: {frames}": "{count} 帧损坏，已略过：{frames}",
}
