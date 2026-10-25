"""Chinese texts of the Sputter workspace's literature method and Literature tab (``features/sputter``).

The explanation "Literature method (mass balance)", the Literature tab (``presentation/literature_tab.py``), the
literature rows of the Windows table and the report part (``domain/literature.py``). The toolkit's own sentences
(its checks and notes) are in ``zh_sputter_toolkit.py``. Symbols, grades, citation keys and numbers stay as they
are.
"""

SPUTTER_LITERATURE_ZH = {
    # -- the explanation, the result line, the table -----------------------------------------------------------
    "literature method (mass balance)": "文献方法（质量守恒）",
    "no percolation found": "没有找到渗流",
    "percolation (literature method) at δ = {delta} nm, t = {time} s": "渗流（文献方法）在 δ = {delta} nm，t = {time} s",
    "percolation (literature method) at δ ≤ {delta} nm": "渗流（文献方法）在 δ ≤ {delta} nm",
    "percolation (literature method) not determined: from δ = {low} nm or later":
        "渗流（文献方法）无法确定：在 δ = {low} nm 或更晚",
    "2R/D below 1 up to δ = {delta} nm (no percolation while a peak is resolved)":
        "直到 δ = {delta} nm 2R/D 都小于 1（峰可分辨期间没有渗流）",
    "2π/q* after the percolation": "渗流之后的 2π/q*",
    "Literature method": "文献方法",
    "R literature (nm)": "R 文献方法（nm）",
    "D literature (nm)": "D 文献方法（nm）",
    "2R/D literature": "2R/D 文献方法",
    "Coverage literature": "覆盖率 文献方法",
    "2R/D network": "2R/D 网络",
    "Coverage network": "覆盖率 网络",
    "η network": "η 网络",
    "Literature note": "文献方法说明",
    "after percolation": "渗流之后",
    "host peak": "宿主（聚合物）峰",
    "outlier": "离群",
    "late peak": "晚出现的峰",
    "no peak": "无峰",
    "no data": "无数据",
    "before deposition": "沉积开始前",
    "beam may be off the sample": "光斑可能移出样品",
    # -- the trend ---------------------------------------------------------------------------------------------
    "after percolation: D = 2π/q* goes on, R is not given (film)": "渗流之后：D = 2π/q* 继续给出，不给 R（膜）",
    "68% interval of the percolation (scatter, rate ±{rate_error}%, t0 ±{t0_error} s)":
        "渗流的 68% 区间（离散、速率 ±{rate_error}%、t0 ±{t0_error} s）",
    "percolation not determined: from {time} s or later": "渗流无法确定：在 {time} s 或更晚",
    "network path: film onsets about as consistent as this one": "网络路径：与此起点同样相容的成膜起点",
    "network path: onset not established (no film fits about as well)": "网络路径：起点未确立（不成膜也同样相容）",
    "network path: no film onset; one from {time} s or later is not excluded": "网络路径：没有成膜起点；不排除在 {time} s 或更晚开始",
    "literature method: the islands hold δ by construction": "文献方法：按构造岛中的金属就是 δ",
    "percolation {time} s (literature method)": "渗流 {time} s（文献方法）",
    "δ, nominal thickness (nm)": "δ，名义厚度（nm）",
    # -- the Literature tab ------------------------------------------------------------------------------------
    "Literature": "文献对照",
    "Predict a scan or open a results folder: its comparison with published growth appears here.":
        "预测一个扫描或打开结果文件夹：这里显示它与已发表生长数据的对照。",
    "The comparison with the literature uses the physical files: choose Physical path or Literature method under More "
    "options ▸ Explanation.":
        "与文献的对照使用物理文件：在“更多选项 ▸ 解释”中选择“物理路径”或“文献方法”。",
    "This result was made before the literature method existed: enter the deposition rate again under More options (or "
    "press Predict) to add it.":
        "这个结果是在有文献方法之前得到的：在“更多选项”中重新输入沉积速率（或按“预测”）即可加上。",
    "The literature method needs the deposition rate (for example from a quartz balance): enter it under More options ▸ "
    "Deposition rate (nm/s), and the start of the deposition t0 if you know it. The files are rewritten without "
    "reading the frames again. Until then only the monotony of D is tested.":
        "文献方法需要沉积速率（例如来自石英晶体微天平）：在“更多选项 ▸ 沉积速率（nm/s）”中输入，知道的话再输入沉积开始 t0。"
        "文件会重写，不重新读帧。在此之前只检验 D 是否单调。",
    "<b>{metal} on {polymer}</b> · published percolation band of {host}: {low}–{high} nm · deposition rate {rate} "
    "nm/s, t0 = {t0} s":
        "<b>{polymer} 上的 {metal}</b> · {host} 的已发表渗流区间：{low}–{high} nm · 沉积速率 {rate} nm/s，t0 = {t0} s",
    " (assumed: the first frame)": "（假定：第一帧）",
    " · deposition ended at t_end = {t_end} s: later windows stay at δ = {delta} nm":
        " · 沉积在 t_end = {t_end} s 结束：之后的时间窗停留在 δ = {delta} nm",
    " · deposition ended at t_end = {t_end} s, after the last window: δ = rate × (t − t0) in every window":
        " · 沉积在 t_end = {t_end} s 结束，在最后一个时间窗之后：每个时间窗中 δ = 速率 × (t − t0)",
    "a polymer of unknown type": "种类未知的聚合物",
    "D = 2π/q* (literature method)": "D = 2π/q*（文献方法）",
    "trend of D (non-decreasing)": "D 的趋势（非减）",
    "peaks not used (host, outlier, late, beam off the sample)": "未使用的峰（宿主、离群、晚出现、光斑移出样品）",
    "68% interval (scatter, rate ±{rate_error}%, t0 ±{t0_error} s)": "68% 区间（离散、速率 ±{rate_error}%、t0 ±{t0_error} s）",
    "percolation not determined: from δ = {low} nm or later": "渗流无法确定：在 δ = {low} nm 或更晚",
    "D of the network": "网络的 D",
    "Spacing D against δ": "间距 D 随 δ",
    "published 2R/D, islands and coalescence ({refs})": "已发表的 2R/D，岛与聚并阶段（{refs}）",
    "percolation: 2R = D": "渗流：2R = D",
    "literature method": "文献方法",
    "literature method, trend": "文献方法，趋势",
    "network": "网络",
    "2R/D against δ": "2R/D 随 δ",
    "published coverage at percolation {low}–{high}": "已发表的渗流时覆盖率 {low}–{high}",
    "published coverage at its δ": "已发表的覆盖率（在其 δ 处）",
    "coverage": "覆盖率",
    "Coverage against δ": "覆盖率随 δ",
    "literature method: 1 by construction": "文献方法：按构造为 1",
    "{value}: flagged below (engineering choice)": "{value}：低于此值提示（工程上的选择）",
    "network: η = t_eq/δ": "网络：η = t_eq/δ",
    "Share of the nominal metal in the network's islands": "网络所见岛中的金属占名义量的比例",
    "published percolation, {host}": "已发表的渗流，{host}",
    "percolation (literature method) δ = {delta} nm": "渗流（文献方法）δ = {delta} nm",
    "TS onset of the network path δ = {delta} nm": "网络路径的 TS 起点 δ = {delta} nm",
    # -- the report and the checks table (domain/literature.py) ------------------------------------------------
    "<b>Literature method (mass balance), the default view.</b> As in the P03 in-situ GISAXS studies, R is not "
    "fitted: it follows from the nominal thickness δ = rate × (t − t0) (constant after the end of the deposition) and "
    "the spacing D = 2π/q* of the direct peak by mass balance for hemispheres (h = R) on a hexagonal lattice, "
    "δ = 2.418·R³/D² (grade <b>Lit</b>). It agrees with the literature by "
    "construction; the Literature tab shows the independent tests (D against δ, the percolation against the published "
    "band, the network's 2R/D and mass balance). The simulation-based typical errors do not apply: D = 2π/q* can "
    "be off by more than 40% for D &gt; 10 nm once coalescence starts, and R follows D as D^(2/3) and the rate as "
    "rate^(1/3).":
        "<b>文献方法（质量守恒），默认视图。</b>与 P03 原位 GISAXS 研究相同，R 不是拟合出来的：它由名义厚度 "
        "δ = 速率 × (t − t0)（沉积结束后保持不变）和直接峰的"
        "间距 D = 2π/q* 按六角排列半球（h = R）的质量守恒算出，δ = 2.418·R³/D²（等级 <b>Lit</b>）。它按构造与文献一致；"
        "“文献对照”页给出独立的检验（D 随 δ、渗流与已发表区间、网络的 2R/D 和质量守恒）。基于模拟的典型误差不适用于这些值："
        "聚并开始后、D &gt; 10 nm 时 D = 2π/q* 的误差可超过 40%，R 随 D 按 D^(2/3)、随速率按 rate^(1/3) 变化。",
    "Percolation (2R/D = 1) at δ = {delta} nm, t = {time} s (68% interval {low}–{high} nm from the scatter of the "
    "windows, the rate ±{rate_error}% and t0 ±{t0_error} s; a systematic error of 2π/q* is not in it); after it "
    "D = 2π/q* stays and R is not given (film).":
        "渗流（2R/D = 1）在 δ = {delta} nm，t = {time} s（68% 区间 {low}–{high} nm，含时间窗的离散、速率 ±{rate_error}% 和 "
        "t0 ±{t0_error} s；不含 2π/q* 的系统误差）；此后 D = 2π/q* 继续给出，不给 R（膜）。",
    "Percolation not determined: within the scatter of the windows, the rate ±{rate_error}% and t0 ±{t0_error} s, "
    "2R/D reaches 1 from δ = {low} nm, or not before the peak is lost at δ = {last} nm. R is given throughout.":
        "渗流无法确定：在时间窗的离散、速率 ±{rate_error}% 和 t0 ±{t0_error} s 的范围内，2R/D 可能从 δ = {low} nm 起达到 1，"
        "也可能直到峰消失的 δ = {last} nm 都达不到。R 在整个过程中都给出。",
    "2R/D is at or above 1 from the first window with a peak after nucleation: percolation at δ ≤ {delta} nm "
    "(t = {time} s).":
        "从成核之后第一个有峰的时间窗起 2R/D 就已达到或超过 1：渗流在 δ ≤ {delta} nm（t = {time} s）。",
    "2R/D stays below 1 (at most {ratio}) up to δ = {delta} nm, where the peak is lost, also with a {rate_error}% "
    "higher rate and t0 {t0_error} s earlier: no percolation while a peak is resolved.":
        "2R/D 一直小于 1（至多 {ratio}），直到峰消失的 δ = {delta} nm；速率高 {rate_error}%、t0 早 {t0_error} s 时也是如此："
        "峰可分辨期间没有渗流。",
    "Checks against the literature: {consistent} consistent, {inconsistent} inconsistent, {open} not testable "
    "(Literature tab).":
        "与文献的检验：{consistent} 项一致，{inconsistent} 项不一致，{open} 项无法检验（见“文献对照”页）。",
    "<b>Literature method.</b> It needs the deposition rate: enter it under More options ▸ Deposition rate (nm/s). The "
    "files are rewritten without reading the frames again.":
        "<b>文献方法。</b>它需要沉积速率：在“更多选项 ▸ 沉积速率（nm/s）”中输入。文件会重写，不重新读帧。",
    "What is compared": "比较的内容",
    "Verdict": "结论",
    "Measured": "测得",
    "Expected (literature)": "预期（文献）",
    "References": "参考文献",
    "inconsistent": "不一致",
    "not testable": "无法检验",
    "Caveats and notes": "注意事项与说明",
}

__all__ = ["SPUTTER_LITERATURE_ZH"]
