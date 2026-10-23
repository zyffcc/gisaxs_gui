"""Chinese texts of the Sputter workspace's reliability report (``features/sputter/domain``).

The domain writes the report from English templates (``reliability_report(result, text)``); the page fills
these translated templates with the same values. Grades, symbols and numbers stay as they are.
"""

SPUTTER_REPORT_ZH = {
    "{name}: how reliable are the sizes?": "{name}：这些尺寸有多可靠？",
    "{windows} time windows; {film} late windows use the TS film model (characteristic length L instead of island "
    "R/D).": "{windows} 个时间窗；其中 {film} 个后期时间窗用 TS 膜模型（用特征长度 L 代替岛的 R/D）。",
    "Windows with a value": "有值的时间窗",
    "By grade": "按等级",
    "Typical error of these values": "这些值的典型误差",
    "Order of two windows correct (≥1.2×)": "两个时间窗的大小顺序正确（≥1.2 倍）",
    "Window-to-window scatter (data)": "相邻时间窗的离散（数据）",
    "R, equivalent hemisphere radius": "R，等效半球半径",
    "D, centre-to-centre spacing": "D，中心间距",
    "Grades:": "等级：",
    "<b>Check against the data.</b> {outside} of {checked} D values differ from the direct 2π/q* by more than the "
    "simulated 90% range (D/(2π/q*) outside {low}–{high}); they are circled in red and should not be used.":
        "<b>与数据核对。</b>{checked} 个 D 值中有 {outside} 个与直接值 2π/q* 的差别超出模拟的 90% 范围"
        "（D/(2π/q*) 不在 {low}–{high} 内）；它们用红圈标出，不应使用。",
    "<b>How to read this.</b> In the trend each window's value is a faint point and the line is their smoothed "
    "trend; the points are not error bars. The typical error of each value (simulation-based, about a 68% band) is in "
    "the Windows table.":
        "<b>怎么读。</b>趋势图中每个时间窗的值是一个淡色点，线是它们的平滑趋势；这些点不是误差棒。每个值的典型误差"
        "（基于模拟，约 68% 区间）在“时间窗”表中。",
    "<b>How to read this.</b> Error bars in the trend are the simulation-based typical error of the value's source "
    "(about a 68% band).": "<b>怎么读。</b>趋势图中的误差棒是该值来源基于模拟的典型误差（约 68% 区间）。",
    "About 10–20% of values carry a larger error (P90 ≈ 50–65%) because another structure fits the counts equally "
    "well; grade B marks windows where such an alternative was found. Trends are more reliable than absolute values: "
    "two windows whose true R differ by 20% are put in the right order in about 92–96% of simulated cases. The "
    "window-to-window scatter is measured on your data from neighbouring windows with independent photons.":
        "约 10–20% 的值误差更大（P90 ≈ 50–65%），因为另一种结构同样能拟合计数；等级 B 标出找到了这种另一解的时间窗。"
        "趋势比绝对值可靠：真实 R 相差 20% 的两个时间窗，在约 92–96% 的模拟情形中顺序正确。相邻时间窗的离散是在你的数据上"
        "用光子相互独立的相邻时间窗测得的。",
    "<b>Model conditions.</b> R is the base radius of a flat-based hemisphere (height h = R). Real clusters may be "
    "taller (h/R between 1 and 2); absolute R depends on this condition, trends much less. D is given only where a "
    "correlation peak is resolved; × marks the direct value 2π/q*. Burial depth and polymer thickness are not "
    "reliable (order accuracy 52–80%) and are only listed as conditional values.":
        "<b>模型条件。</b>R 是平底半球的底面半径（高 h = R）。真实团簇可能更高（h/R 在 1 到 2 之间）；R 的绝对值依赖这个条件，"
        "趋势受影响小得多。只有分辨出关联峰时才给 D；× 标出直接值 2π/q*。埋入深度和聚合物厚度不可靠（顺序准确率 52–80%），"
        "只作为有条件的值列出。",
    "The networks were trained for Ag and Au; a Cu series is read with the Ag model, so its sizes are a reference, "
    "not a measurement.": "网络是为 Ag 和 Au 训练的；Cu 序列用 Ag 模型读取，所以它的尺寸是参考，不是测量值。",
    # -- the physical path ---------------------------------------------------------------------------------------
    "<b>Physical path of the network.</b> One path of stages through time — pre-nucleation, islands, coalescence, film — "
    "in which a film does not revert to islands. R and D are the network estimate in every island window (grade P): "
    "the source does not switch between refined and network values from window to window (refined values stay in "
    "the Windows table). The trend lines are smoothed over neighbouring island windows. {changed} of {windows} "
    "windows differ from the per-window rule.":
        "<b>网络的物理路径。</b>随时间只走一条阶段路径——成核前、岛、聚并、膜——膜不会退回成岛。每个岛时间窗的 R 和 D 都是网络估计"
        "（等级 P）：来源不会在相邻时间窗之间在精修值和网络值之间切换（精修值保留在“时间窗”表中）。趋势线在相邻的岛时间窗上平滑。"
        "{windows} 个时间窗中有 {changed} 个与逐窗规则不同。",
    "Film from {onset} s on this path (onset between {low} and {high} s).":
        "这条路径从 {onset} s 起成膜（起点在 {low} 到 {high} s 之间）。",
    "Film from {onset} s on this path.": "这条路径从 {onset} s 起成膜。",
    "No film window on this path.": "这条路径上没有膜时间窗。",
    "Back-and-forth changes before (per-window rule) and after the physical path:":
        "来回变化：用物理路径之前（逐窗规则）和之后：",
    "Back and forth": "来回变化",
    "Per-window rule": "逐窗规则",
    "Physical path": "物理路径",
    "Stage steps back": "阶段后退",
    "Film → islands": "膜 → 岛",
    "R zig-zags": "R 锯齿",
    "D zig-zags": "D 锯齿",
    "The intensity is modulated with a period of about {period} s (autocorrelation peak {acf}, {cycles} cycles), "
    "possibly by a sample rotation: the data do not tell the source. A single window shorter than this period sees "
    "part of one modulation cycle; the physical path and the trend lines average over it.":
        "强度以约 {period} s 的周期调制（自相关峰 {acf}，{cycles} 个周期），可能是样品旋转：数据不能告诉调制的来源。短于这个周期的"
        "单个时间窗只看到一个调制周期的一部分；物理路径和趋势线对它取平均。",
    "No periodic intensity modulation was found.": "没有发现周期性的强度调制。",
    "<b>Metal thickness.</b> t_eq is the island-volume equivalent thickness: the metal in the visible islands, for "
    "hemispheres (h = R) on a hexagonal arrangement with spacing D. It is a conditional value with an error of about "
    "a factor 2–3, and it is not given in film windows.":
        "<b>金属厚度。</b>t_eq 是岛体积等效厚度：可见岛中的金属，按半球（h = R）以间距 D 六角排列计算。它是有条件的值，误差约 2–3 倍，"
        "膜时间窗中不给出。",
    "Nominal thickness = rate × (t − t0) with rate {rate} nm/s and t0 = {t0} s: what was deposited, not what is "
    "measured.": "名义厚度 = 速率 × (t − t0)，速率 {rate} nm/s，t0 = {t0} s：这是沉积的量，不是测得的量。",
    "Give the deposition rate (More options) to draw the nominal thickness rate × (t − t0).":
        "给出沉积速率（更多选项）即可画出名义厚度 速率 × (t − t0)。",
    "In film windows the thickness is not determined: the current model cannot measure the thickness of a continuous "
    "film (in simulation its estimate equals the prior). A model of a film with particles on it is planned.":
        "膜时间窗中厚度未确定：现有模型测不出连续膜的厚度（在模拟中它的估计等于先验）。膜上带颗粒的模型正在计划中。",
    "Mass check at the film onset ({time} s): island-volume thickness {teq} nm against the nominal {nominal} nm "
    "(ratio {ratio}). A ratio far below 1 means that much of the metal is not in the visible islands (smaller "
    "clusters, embedded metal, a wetting layer) or that the hemisphere condition does not hold.":
        "成膜起点（{time} s）的质量核对：岛体积厚度 {teq} nm，名义厚度 {nominal} nm（比值 {ratio}）。比值远小于 1 说明大部分金属"
        "不在可见的岛中（更小的团簇、埋入的金属、润湿层），或半球条件不成立。",
    "Notes of the model:": "模型的说明：",
    # -- the time-consistent path, the learned selector -----------------------------------------------------------
    "never": "从不",
    "D (peak windows)": "D（有峰时间窗）",
    "passed": "通过",
    "not passed": "未通过",
    "<b>Time-consistent path.</b> All network explanations of every window were considered and one path through "
    "time was chosen with registered physical rules (sizes and spacing do not shrink quickly; a film does not revert "
    "to islands; D should agree with 2π/q*). {changed} of {windows} windows changed from the per-window choice. Grade "
    "<b>S</b> ({supported} R values): the data support the chosen explanation (within 2 nats of the best). Grade "
    "<b>T</b> ({imposed} R values): the rules imposed it against the data — read these as the story, not as a "
    "measurement. Film from {film} on this path; when the observation-rule labels switch back and forth, this time is "
    "uncertain. The path can only choose among the network's explanations: where none fits (e.g. a steep drop that "
    "every explanation shows) the drop stays.":
        "<b>时间一致路径。</b>考虑了每个时间窗的全部网络解释，并按登记的物理规则选出一条随时间的路径（尺寸和间距不会迅速变小；"
        "膜不会退回成岛；D 应与 2π/q* 一致）。{windows} 个时间窗中有 {changed} 个与逐窗选择不同。等级 <b>S</b>（{supported} 个 R 值）："
        "数据支持所选解释（与最优相差 2 nats 以内）。等级 <b>T</b>（{imposed} 个 R 值）：规则违背数据强加了它——把这些当作故事，"
        "不要当作测量。这条路径从 {film} 起成膜；观测规则的标签来回切换时，这个时间不确定。路径只能在网络的解释中选择："
        "没有一个解释合适时（例如每个解释都显示的陡降），陡降会保留。",
    "<b>Exploratory — not more accurate.</b> In the registered test on 900 simulated growth windows this path was "
    "<i>not</i> more accurate than the per-window choice (R wrong by &gt;30%: 26% vs 28% of windows; median R error "
    "16% vs 15%; D unchanged), and it erased a real 25% dip of R in 93% of the dip windows. Use it to see one "
    "consistent picture, not as a more reliable measurement. The correct explanation was among the network's "
    "candidates in about 92% of windows; a better way of choosing it is being developed.":
        "<b>探索性——并不更准。</b>在 900 个模拟生长时间窗的登记测试中，这条路径<i>并不</i>比逐窗选择更准（R 错 &gt;30% 的时间窗："
        "26% 对 28%；R 的中位误差 16% 对 15%；D 不变），并且在 93% 的下凹时间窗中抹掉了 R 真实的 25% 下凹。用它看一幅一致的图景，"
        "不要把它当作更可靠的测量。在约 92% 的时间窗中，正确解释在网络的候选之中；更好的选择方法正在开发。",
    "<b>Learned selector.</b> For every window the network offers four explanations (mixture components); a model "
    "trained on simulated growth series picks one from their network weights, the agreement of D with 2π/q* and how "
    "well each agrees with the explanations of the three windows before and after. {low} windows have a selection "
    "probability below 0.5.":
        "<b>学习选择器。</b>网络为每个时间窗给出四种解释（混合分量）；一个在模拟生长序列上训练的模型根据它们的网络权重、D 与 2π/q* "
        "的一致程度，以及与前后各三个时间窗解释的一致程度从中选一个。{low} 个时间窗的选择概率低于 0.5。",
    "Independent simulated TEST": "独立的模拟测试（TEST）",
    "Network top: wrong &gt;30% / median error": "网络首选：错 &gt;30% / 中位误差",
    "Selector: wrong &gt;30% / median error": "选择器：错 &gt;30% / 中位误差",
    "Best possible (oracle)": "最好可能（oracle）",
    "Registered test (wrong-branch share cut by at least 30% and median error not larger): {verdict}. The selector is "
    "therefore offered as an option, not as the default, unless both pass. Its typical errors in the trend come from "
    "that test. On real scans compare its D with 2π/q* (red circles).":
        "登记的测试（错分支比例至少降低 30%，且中位误差不变大）：{verdict}。因此除非两项都通过，选择器只作为选项提供，不作默认。"
        "趋势图中它的典型误差来自这个测试。在实测扫描上请把它的 D 与 2π/q* 比较（红圈）。",
}

__all__ = ["SPUTTER_REPORT_ZH"]
