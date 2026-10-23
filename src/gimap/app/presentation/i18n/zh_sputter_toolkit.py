"""Chinese texts of the Sputter toolkit's own sentences (``features/sputter/presentation/toolkit_texts.py``).

The toolkit writes its notes, reasons and literature checks in English from templates; the page matches each
sentence against them (``texts.note_text``) and fills the same values into these. Symbols, column names,
citation keys and numbers stay as they are.
"""

SPUTTER_TOOLKIT_ZH = {
    't_eq = (2/3)*pi*R^3*exp(3*sigma^2) / ((sqrt(3)/2)*D^2): island-volume equivalent thickness of the '
    'visible particles, conditional on flat-based hemispheres (h=R) on a hexagonal lattice; not the '
    'deposited amount. Band t_eq/f..t_eq*f with f = exp(sqrt(9*eR^2 + 4*eD^2)), eR and eD = ln(1 + typical '
    'relative error) of the network R and D.':
        't_eq = (2/3)*pi*R^3*exp(3*sigma^2) / ((sqrt(3)/2)*D^2)：可'
        '见颗粒的岛体积等效厚度，以平底半球（h=R）六角排列为条件；不是沉积量。区间 t_eq/f..t_eq*f，f '
        '= exp(sqrt(9*eR^2 + 4*eD^2))，eR、eD = ln(1 + 网络 R、D 的典型相对'
        '误差)。',
    'nominal deposited thickness = rate * (t - t0) from the given deposition rate (e.g. QCM); not measured '
    'by GISAXS.':
        '名义沉积厚度 = 速率 * (t - t0)，由给出的沉积速率（例如 QCM）算出；不是 GISAXS 测得的。',
    'nominal deposited thickness = rate * (t - t0) from the given deposition rate (e.g. QCM); not measured '
    'by GISAXS. t0 not given: assumed 0 s = the first frame; the start of the deposition is not known from '
    'the data (each 1 s of error in t0 shifts the nominal thickness by {rate} nm).':
        '名义沉积厚度 = 速率 * (t - t0)，由给出的沉积速率（例如 QCM）算出；不是 GISAXS 测得的。'
        '没有给出 t0：假定为 0 s，即第一帧；数据不能告诉沉积从何时开始（t0 每差 1 s，名义厚度差 {rate'
        '} nm）。',
    'no deposition rate given: nominal thickness not computed':
        '没有给出沉积速率：未计算名义厚度',
    'not determined by GISAXS: the TS film thickness and metal fraction are not identifiable (in simulation '
    "the network's posterior equals its prior); see L_TS_nm for the lateral texture length":
        'GISAXS 无法确定：TS 膜厚和金属分数不可辨识（在模拟中网络的后验等于先验）；横向纹理长度见 L_TS_n'
        'm',
    'first window of the TS-network part of the path (rule stage coalescence or coalescence-or-film '
    'candidate); a connected film is not proven':
        '路径上属于 TS 网络部分的第一个时间窗（规则阶段为聚并或“聚并或成膜”候选）；并未证明是连通的膜',
    'the network of the path module was not run on this window in this folder (binned/frames_alt.csv '
    'missing; run analyze again on this folder to add it)':
        '路径模块的网络没有在本文件夹的这个时间窗上运行（缺少 binned/frames_alt.csv；对这个文件夹再'
        '运行一次 analyze 即可补上）',
    'film-type (TS layer) window on the physical path: island R/D not defined (see L_TS_nm)':
        '物理路径上的膜类（TS 层）时间窗：岛的 R/D 没有定义（见 L_TS_nm）',
    'pre-nucleation on the physical path: no island estimate':
        '物理路径上处于成核前：没有岛的估计',
    'no window of the series has a resolved stage: no island estimate':
        '序列中没有一个时间窗的阶段已分辨：没有岛的估计',
    'fewer than 1000 counts in the window (shutter closed or beam off): no stage, no values':
        '时间窗内计数少于 1000（快门关闭或没有光束）：没有阶段，没有数值',
    'no resolved peak: D not reported':
        '没有分辨出峰：不报告 D',
    'no estimate':
        '没有估计',
    'outside simulated 90% range of D/(2pi/q*)':
        '超出 D/(2pi/q*) 的模拟 90% 范围',
    'Model assumption: during deposition the TS onset is monotone (no return from film to islands); the '
    'stage path is the non-decreasing L1 fit of the rule stages, and unresolved windows cast no vote.':
        '模型假设：沉积过程中 TS 起点是单调的（不会从膜回到岛）；阶段路径是规则阶段的非减 L1 拟合，未分辨的时间窗'
        '不参与。',
    "stage_path 'film' is the rule's coalescence-or-film candidate (a connected film is not proven); "
    'film=true marks every window on the TS network (coalescence or film).':
        "stage_path 的 'film' 是规则的“聚并或成膜”候选（并未证明是连通的膜）；film=true 标"
        '出 TS 网络上的每个时间窗（聚并或膜）。',
    'R_trend_nm / D_trend_nm: local log-linear (Theil-Sen) fit of ln R / ln D over the island windows in a '
    'centred span of {span} s (max of the modulation period and 2.5 s), evaluated at each window.':
        'R_trend_nm / D_trend_nm：在以每个时间窗为中心、宽 {span} s（调制周期与 2.5 '
        's 中较大者）的范围内，对岛时间窗的 ln R / ln D 做局部对数线性（Theil-Sen）拟合，在该时间'
        '窗取值。',
    'R_nm / D_nm are network estimates in every window whose path stage is islands (grade P); refined values '
    'stay in refined_R_nm / refined_D_nm. Pre-nucleation windows have no island values.':
        '路径阶段为岛的每个时间窗中，R_nm / D_nm 都是网络估计（等级 P）；精修值保留在 refined_R_'
        'nm / refined_D_nm。成核前的时间窗没有岛的数值。',
    'TS film windows: no thickness or metal fraction is reported (not identifiable); L_TS_nm is the lateral '
    'texture length, not an island spacing.':
        'TS 膜时间窗：不报告厚度和金属分数（不可辨识）；L_TS_nm 是横向纹理长度，不是岛间距。',
    'Cu approximated with the Ag model, for reference only (errors not validated).':
        'Cu 用 Ag 模型近似，仅供参考（误差未经验证）。',
    '{count} windows with an unresolved rule stage carry the stage of the neighbouring resolved window '
    '(stage_imputed=true; stage_path_counts_imputed).':
        '{count} 个规则阶段未分辨的时间窗沿用相邻已分辨时间窗的阶段（stage_imputed=true；sta'
        'ge_path_counts_imputed）。',
    '{count} windows with fewer than {counts} counts have no stage and no values.':
        '{count} 个计数少于 {counts} 的时间窗没有阶段，也没有数值。',
    'TS onset not established: no TS window at all disagrees with {no_ts} resolved windows, within {slack} '
    'of the {best} at the chosen onset ({onset} s); the series may stay in the island phase.':
        'TS 起点未确立：完全没有 TS 时间窗时与 {no_ts} 个已分辨时间窗不符，与所选起点（{onset} s'
        '）的 {best} 个只差不到 {slack} 个；序列可能一直处于岛阶段。',
    'No TS onset on the path, but onsets from {start} s disagree with at most {most} resolved windows: a '
    'late TS onset is not excluded.':
        '路径上没有 TS 起点，但从 {start} s 起的起点至多与 {most} 个已分辨时间窗不符：不能排除较晚'
        '的 TS 起点。',
    'The path reaches coalescence but never the film stage: film_onset_s is the onset of TS-network '
    '(coalescence) windows, not of a continuous film.':
        '路径到达聚并，但从未到达膜阶段：film_onset_s 是 TS 网络（聚并）时间窗的起点，不是连续膜的起点。',
    'The intensity dips every {period} s with unequal depths; the pattern repeats exactly after {repeat} s, '
    'which may be one turn of the sample.':
        '强度每 {period} s 出现一次深浅不一的下凹；图样要到 {repeat} s 后才完全重复，这可能是样品'
        '转一圈的时间。',
    'The smoothing span ({span} s) covers the whole island phase ({start}-{end} s): R_trend_nm / D_trend_nm '
    'are one log-linear fit over it; the per-window R_nm / D_nm (grade P) show more.':
        '平滑宽度（{span} s）覆盖了整个岛阶段（{start}-{end} s）：R_trend_nm / D_t'
        'rend_nm 是在其上的一次对数线性拟合；逐窗的 R_nm / D_nm（等级 P）显示得更多。',
    'needs a deposition rate: the literature method computes R from the nominal thickness delta = rate*(t - '
    't0); without it only the monotony of D is tested (in time)':
        '需要沉积速率：文献方法由名义厚度 delta = 速率*(t - t0) 算出 R；没有速率时只检验 D 是否单'
        '调（随时间）',
    'fewer than {minimum} windows with a resolved peak of the metal: no literature-method values':
        '金属的可分辨峰少于 {minimum} 个时间窗：没有文献方法的数值',
    'literature method of the P03 in-situ GISAXS studies: D = 2*pi/q* of the direct peak, R = '
    '(delta*D^2/2.418)^(1/3) for hemispheres (h=R) on a hexagonal lattice, coverage = '
    '(2*pi/sqrt(3))*(R/D)^2; percolation where the trend of 2R/D reaches 1; after it this tool keeps D = '
    '2*pi/q* but gives no R (Abr16; the P03 studies go on reporting R and 2R/D)':
        'P03 原位 GISAXS 研究的文献方法：D = 直接峰的 2*pi/q*，R = (delta*D^2/2.'
        '418)^(1/3)（六角排列的半球，h=R），覆盖率 = (2*pi/sqrt(3))*(R/D)^2；渗流在'
        ' 2R/D 趋势达到 1 处；此后本工具保留 D = 2*pi/q*，但不再给 R（Abr16；P03 的研究在'
        '渗流之后仍报告 R 和 2R/D）',
    'fewer than 1000 counts in the window: no value':
        '时间窗内计数少于 1000：没有数值',
    'before the start of the deposition (delta = 0): no island value':
        '沉积开始之前（delta = 0）：没有岛的数值',
    'no resolved peak: D and R not given':
        '没有分辨出峰：不给 D 和 R',
    'peak far above every literature D(delta) at this delta: probably the host, not used':
        '峰远高于该 delta 下所有文献 D(delta)：可能来自宿主（聚合物），不使用',
    'peak before the start of the deposition: the host, not used':
        '沉积开始之前就有的峰：来自宿主（聚合物），不使用',
    'peak at the spacing of the host peak and far from the trend of D: probably the host, not used':
        '峰位于宿主（聚合物）峰的间距，且远离 D 的趋势：可能来自宿主，不使用',
    'peak more than a factor 2 from the trend of D: another feature, not used':
        '峰与 D 的趋势相差超过 2 倍：另一种特征，不使用',
    'peak more than a factor 2 above the preceding peaks, beyond any growth since t0: a jump, not growth; '
    'not used':
        '峰比之前的峰高出 2 倍以上，超出自 t0 以来可能的生长：是跳变，不是生长；不使用',
    'peak after a long stretch without a peak: another structure, not used':
        '长时间无峰之后出现的峰：另一种结构，不使用',
    'after the percolation of the literature method: D = 2*pi/q* kept, R not given (film)':
        '文献方法的渗流之后：保留 D = 2*pi/q*，不给 R（膜）',
    'no deposition rate: only D is given':
        '没有沉积速率：只给出 D',
    'D (= 2*pi/q*) does not decrease with delta':
        'D（= 2*pi/q*）不随 delta 减小',
    'Percolation by the literature method against the published band':
        '文献方法的渗流与已发表区间比较',
    'TS onset of the physical path against the published band':
        '物理路径的 TS 起点与已发表区间比较',
    'Network 2R/D and coverage against the literature':
        '网络的 2R/D 和覆盖率与文献比较',
    'Network mass balance eta = t_eq/delta':
        '网络的质量守恒 eta = t_eq/delta',
    'TS film thickness grows with delta':
        'TS 膜厚随 delta 增长',
    'needs a deposition rate':
        '需要沉积速率',
    'fewer than {minimum} windows with a resolved peak of the metal':
        '金属的可分辨峰少于 {minimum} 个时间窗',
    'Spearman rho = {rho} over {windows} windows; the binned median D drops by {drop}% between delta {low} '
    'and {high} nm, beyond the scatter':
        'Spearman rho = {rho}（{windows} 个时间窗）；分箱中位 D 在 delta {low'
        '} 到 {high} nm 之间下降 {drop}%，超出离散',
    'Spearman rho = {rho} over {windows} windows; the binned median D drops by {drop}% between {low} and '
    '{high} s, beyond the scatter':
        'Spearman rho = {rho}（{windows} 个时间窗）；分箱中位 D 在 {low} 到 {h'
        'igh} s 之间下降 {drop}%，超出离散',
    'Spearman rho = {rho} over {windows} windows; no drop of the binned median D beyond the scatter (largest '
    '{drop}%)':
        'Spearman rho = {rho}（{windows} 个时间窗）；分箱中位 D 没有超出离散的下降（最大'
        ' {drop}%）',
    'D grows or stays with delta after nucleation (delta > {start} nm); only during nucleation a slight '
    'decrease is reported':
        '成核之后（delta > {start} nm）D 随 delta 增大或不变；只有成核期报道过轻微下降',
    'D grows or stays with time during the deposition; only during nucleation a slight decrease is reported':
        '沉积过程中 D 随时间增大或不变；只有成核期报道过轻微下降',
    '{count} peak windows were not used (host peak, outlier, jump or late peak; see lit_status).':
        '{count} 个有峰时间窗未使用（宿主峰、离群、跳变或晚出现的峰；见 lit_status）。',
    'Without a rate the nucleation phase is not known: early windows are included.':
        '没有速率时不知道成核期：早期时间窗也计入。',
    '{count} peak windows after a stretch without a peak (t = {low}-{high} s) were not used: their median '
    '2*pi/q* is {median} nm, {percent}% of the end of the trend ({end} nm).':
        '长时间无峰之后的 {count} 个有峰时间窗（t = {low}-{high} s）未使用：它们的 2*pi/'
        'q* 中位数为 {median} nm，是趋势末端（{end} nm）的 {percent}%。',
    'delta = {delta} nm at t = {time} s (68% interval {low}-{high} nm with the noise, the rate '
    '±{rate_error}% and t0 ±{t0_error} s)':
        'delta = {delta} nm，t = {time} s（68% 区间 {low}-{high} nm，含'
        '噪声、速率 ±{rate_error}% 和 t0 ±{t0_error} s）',
    'not determined: within the noise, the rate ±{rate_error}% and t0 ±{t0_error} s, 2R/D reaches 1 from '
    'delta = {low} nm or not before the peak phase ends (delta = {last} nm, t = {time} s)':
        '无法确定：在噪声、速率 ±{rate_error}% 和 t0 ±{t0_error} s 的范围内，2R/D '
        '可能从 delta = {low} nm 起达到 1，也可能直到峰阶段结束（delta = {last} nm，'
        't = {time} s）都达不到',
    'delta <= {delta} nm: 2R/D is at or above 1 from the first window with a peak after nucleation (t = '
    '{time} s)':
        'delta <= {delta} nm：从成核之后第一个有峰的时间窗（t = {time} s）起 2R/D 就'
        '已达到或超过 1',
    '2R/D stays below 1 (at most {ratio}) up to delta = {delta} nm (t = {time} s), where the peak phase '
    'ends, also with a {rate_error}% higher rate and t0 {t0_error} s earlier':
        '2R/D 一直小于 1（至多 {ratio}），直到峰阶段结束的 delta = {delta} nm（t = '
        '{time} s）；速率高 {rate_error}%、t0 早 {t0_error} s 时也是如此',
    'percolation at delta = {low}-{high} nm for {host} (published values: {count}, {gisaxs} of them by '
    'GISAXS; each with its printed ± or ±{tol}%)':
        '{host} 的渗流在 delta = {low}-{high} nm（已发表数值 {count} 个，其中 {'
        'gisaxs} 个来自 GISAXS；每个数值按其标注的 ±，未标注的按 ±{tol}%）',
    'percolation at delta = {low}-{high} nm for {host} (published values: {count}, {gisaxs} of them by '
    'GISAXS; each with its printed ± or ±{tol}%); scaled to {rate} nm/s with chi = 0.07-0.17: '
    '{scaled_low}-{scaled_high} nm':
        '{host} 的渗流在 delta = {low}-{high} nm（已发表数值 {count} 个，其中 {'
        'gisaxs} 个来自 GISAXS；每个数值按其标注的 ±，未标注的按 ±{tol}%）；按 chi = 0.'
        '07-0.17 换算到 {rate} nm/s：{scaled_low}-{scaled_high} nm',
    'no published percolation threshold for {host}':
        '{host} 没有已发表的渗流阈值',
    '68% interval: a moving-block bootstrap of the scatter of the windows around their local median, added '
    'to the trend of D and fitted again, evaluated with the rate {rate_error}% higher and t0 {t0_error} s '
    'earlier for the lower end and the other way round for the upper end (unless given, an engineering '
    'choice; the toolkit options --rate-error and --t0-error set them). A systematic error of 2*pi/q* is not '
    'in it.':
        '68% 区间：对时间窗相对其局部中位数的离散做移动块自助重抽样，叠加到 D 的趋势上再拟合；下端按速率高 {ra'
        'te_error}%、t0 早 {t0_error} s 计算，上端反之（未给出时这两个误差是工程上的选择；工具'
        '包选项 --rate-error 和 --t0-error 可以设定）。2*pi/q* 的系统误差不在其中。',
    'The trend of 2R/D stays within 5% of 1 from delta = {low} to {high} nm: there a small change of the '
    'rate or t0 moves the crossing far.':
        '从 delta = {low} 到 {high} nm，2R/D 的趋势与 1 相差不到 5%：在这一段，速率或'
        ' t0 的小变化就会让交点移动很远。',
    'Earlier than every published value of this band.':
        '早于该区间内所有已发表数值。',
    'Later than every published value of this band.':
        '晚于该区间内所有已发表数值。',
    'Only a bound is measured, and it overlaps the band: the comparison is not decided.':
        '只测得一个界限，且与区间重叠：比较无法判定。',
    'Only one published value for this class: the comparison rests on it and its ±{tol}%.':
        '这一类只有一个已发表数值：比较只依据它及其 ±{tol}%。',
    'With H/R = {f} (Cu on PS-b-PEO), R is {factor} times this R and 2R/D reaches 1 at delta = {delta} nm.':
        '按 H/R = {f}（PS-b-PEO 上的 Cu），R 是这里 R 的 {factor} 倍，2R/D 在 '
        'delta = {delta} nm 达到 1。',
    'With H/R = {f} (Cu on PS-b-PEO), R is {factor} times this R and 2R/D stays below 1 in the peak phase.':
        '按 H/R = {f}（PS-b-PEO 上的 Cu），R 是这里 R 的 {factor} 倍，在峰阶段内 2'
        'R/D 一直小于 1。',
    'With f = 1 and with f = {f} the percolation is outside the band on opposite sides (earlier with f = 1, '
    'later with the published f): the direction depends on the island shape.':
        '按 f = 1 和按 f = {f}，渗流都在区间之外，但在相反的两侧（f = 1 时偏早，按已发表的 f 时偏'
        '晚）：方向取决于岛的形状。',
    'With f = 1 and with f = {f} the comparison gives different verdicts: it depends on the island shape and '
    'is not decided.':
        '按 f = 1 和按 f = {f} 比较得到不同的结论：结论取决于岛的形状，无法判定。',
    'The same holds with f = {f}.':
        '按 f = {f} 也是如此。',
    'TS onset at t = {time} s, delta = {delta} nm (onsets within the slack: {low}-{high} nm)':
        'TS 起点 t = {time} s，delta = {delta} nm（容差内的起点：{low}-{high'
        '} nm）',
    'no TS onset on the path; onsets from delta = {low} nm are within the slack':
        '路径上没有 TS 起点；从 delta = {low} nm 起的起点在容差之内',
    'no TS onset on the path up to delta = {delta} nm':
        '直到 delta = {delta} nm 路径上都没有 TS 起点',
    'The TS onset is the first window the path gives to the TS network (coalescence or film candidate); a '
    'connected film is not proven.':
        'TS 起点是路径交给 TS 网络的第一个时间窗（聚并或成膜候选）；并未证明是连通的膜。',
    'The onset is not established: no TS window at all is within the slack.':
        '起点未确立：完全没有 TS 时间窗也在容差之内。',
    '2R/D median {median} (10-90%: {low}-{high}), coverage median {c_median} (10-90%: {c_low}-{c_high}) over '
    '{windows} island windows with delta >= 1 nm':
        '2R/D 中位数 {median}（10-90%：{low}-{high}），覆盖率中位数 {c_median}'
        '（10-90%：{c_low}-{c_high}），基于 delta >= 1 nm 的 {windows} 个'
        '岛时间窗',
    'fewer than {minimum} island windows with network R and D at delta >= 1 nm':
        'delta >= 1 nm 时有网络 R 和 D 的岛时间窗少于 {minimum} 个',
    '2R/D {low}-{high} after nucleation (mass-balance studies: Au 0.8-0.95, Ag 0.66-0.95; independent: Cu on '
    'graphene 0.55-0.7), 1 at percolation; coverage {c_low}-{c_high} at delta >= 1 nm (0.5-0.65 at 1 nm, '
    '0.58 at 3 nm for Ag on SiO2, 0.76-0.96 at 4 nm for Au on polymers), 0.56-0.9 at percolation':
        '成核之后 2R/D 为 {low}-{high}（质量守恒研究：Au 0.8-0.95，Ag 0.66-0.95'
        '；独立测量：石墨烯上的 Cu 0.55-0.7），渗流时为 1；delta >= 1 nm 时覆盖率为 {c_l'
        'ow}-{c_high}（1 nm 时 0.5-0.65，SiO2 上的 Ag 在 3 nm 时 0.58，聚合'
        '物上的 Au 在 4 nm 时 0.76-0.96），渗流时 0.56-0.9',
    'The network R comes from the shape of the scattering, not from the nominal thickness: an independent '
    'test.':
        '网络的 R 来自散射的形状，不来自名义厚度：这是独立的检验。',
    'coverage = (2*pi/sqrt(3))*(R/D)^2 = 0.907*(2R/D)^2 for equal hemispheres on a hexagonal lattice: it '
    'follows from 2R/D, so the two are one test.':
        '覆盖率 = (2*pi/sqrt(3))*(R/D)^2 = 0.907*(2R/D)^2（六角排列的等大半球）'
        '：它由 2R/D 决定，两者是同一个检验。',
    'Only one of the two published ranges holds the network value; the published ranges of 2R/D and coverage '
    'do not map onto each other exactly.':
        '网络的值只落在两个已发表范围之一内；2R/D 和覆盖率的已发表范围并不能完全互相换算。',
    'median {median} (10-90%: {low}-{high}) over {windows} island windows with delta >= 1 nm; with the t_eq '
    'band {band_low}-{band_high}':
        '中位数 {median}（10-90%：{low}-{high}），基于 delta >= 1 nm 的 {wi'
        'ndows} 个岛时间窗；按 t_eq 区间为 {band_low}-{band_high}',
    '1 by construction in the literature method; embedding in glassy polymers holds 0.6-6%; sticking below 1 '
    'is possible on polymers but not measured for sputtering; below {threshold} is flagged (engineering '
    'choice)':
        '文献方法中按构造为 1；玻璃态聚合物中的埋入占 0.6-6%；聚合物上粘附系数可能小于 1，但溅射条件下未见测量'
        '；低于 {threshold} 时提示（工程上的选择）',
    'Even the upper end of the t_eq band holds less than half of the nominal metal: most of the deposited '
    'metal is not in the islands the network sees, or the rate, t0 or the hemisphere condition is off.':
        '连 t_eq 区间的上端也不到名义金属量的一半：沉积的金属大部分不在网络看到的岛中，或者速率、t0 或半球条件有'
        '误。',
    'The islands hold more metal than was deposited: check the rate and t0.':
        '岛中的金属多于沉积的量：请检查速率和 t0。',
    'eta uses the same network R and D as the 2R/D check: the two are not independent of each other.':
        'eta 和 2R/D 检验用的是同样的网络 R 和 D：两者并不互相独立。',
    'network film thickness {t_low}-{t_high} nm (metal-equivalent {m_low}-{m_high} nm) while delta grows '
    'from {d_low} to {d_high} nm: slope {slope}':
        '网络膜厚 {t_low}-{t_high} nm（金属当量 {m_low}-{m_high} nm），而 del'
        'ta 从 {d_low} 增到 {d_high} nm：斜率 {slope}',
    'fewer than 5 TS windows over at least 2 nm of delta':
        'TS 时间窗少于 5 个，或 delta 跨度不到 2 nm',
    'no value: the network film thickness is not identifiable':
        '没有数值：网络的膜厚不可辨识',
    'after percolation the metal-equivalent film thickness grows about linearly with delta (slope near 1)':
        '渗流之后金属当量膜厚随 delta 近似线性增长（斜率接近 1）',
    'The TS film thickness is not determined: in simulation the network posterior equals its prior, so a '
    'constant value says nothing about the sample.':
        'TS 膜厚未确定：在模拟中网络的后验等于先验，所以一个不变的值说明不了样品。',
    'The literature-method R agrees with the literature by construction: it uses the same mass balance with '
    'the nominal thickness. The independent tests are the shape of D(delta), the percolation delta against '
    'the published band, and the 2R/D and mass balance of the network.':
        '文献方法的 R 按构造与文献一致：它用的是同一个基于名义厚度的质量守恒。独立的检验是 D(delta) 的形状、'
        '渗流 delta 与已发表区间的比较，以及网络的 2R/D 和质量守恒。',
    'R scales as f^(-1/3) with the shape factor f = H/R (here f = 1): the literature reports flatter Ag on '
    'oxides and f = 2.75 for Cu on PS-b-PEO.':
        'R 随形状因子 f = H/R 按 f^(-1/3) 变化（这里 f = 1）：文献报道氧化物上的 Ag 更扁，'
        'PS-b-PEO 上的 Cu 为 f = 2.75。',
    'delta = rate*(t - t0) with the given rate ({rate} nm/s), e.g. from a QCM; GISAXS does not measure it. '
    'The percolation interval allows the rate ±{rate_error}% and t0 ±{t0_error} s.':
        'delta = 速率*(t - t0)，速率取给定的 {rate} nm/s（例如来自 QCM）；GISAXS '
        '不测量它。渗流区间允许速率有 ±{rate_error}%、t0 有 ±{t0_error} s 的误差。',
    't0 not given: assumed 0 s (the first frame); delta is an upper bound if the deposition started at or '
    'after the first frame, and each 1 s of error in t0 shifts delta by {rate} nm.':
        '没有给出 t0：假定为 0 s（第一帧）；若沉积在第一帧或之后才开始，delta 是上限；t0 每差 1 s，d'
        'elta 差 {rate} nm。',
    'D = 2*pi/q* can be off by more than 40% for D > 10 nm once coalescence starts (Hod17).':
        'D = 2*pi/q* 在聚并开始后、D > 10 nm 时误差可超过 40%（Hod17）。',
    'The condensation (sticking) coefficient of these metals on these polymers is not known; the literature '
    'method assumes that all deposited metal stays on the surface.':
        '这些金属在这些聚合物上的凝结（粘附）系数未知；文献方法假定沉积的金属全部留在表面。',
    'Published thresholds come from other chambers and rates; the rate dependence (delta_c ~ F^-chi, chi '
    '0.07-0.17) is weaker than differences between chambers.':
        '已发表的阈值来自别的腔体和速率；速率依赖（delta_c ~ F^-chi，chi 0.07-0.17）比腔体之'
        '间的差别弱。',
    'Cu: the literature method needs no network, but the TS onset and the network checks use the Ag model.':
        'Cu：文献方法不需要网络，但 TS 起点和网络的检验用的是 Ag 模型。',
    '{count} peak windows probably belong to the host (median D {D} nm) and are not used.':
        '{count} 个有峰时间窗可能来自宿主聚合物（中位 D {D} nm），未使用。',
    'Polymer type not given: the band of all polymers is used for {metal}.':
        '没有给出聚合物种类：{metal} 用所有聚合物的区间。',
    'No published percolation data for {polymer}: the band of {host} is used.':
        '{polymer} 没有已发表的渗流数据：用 {host} 的区间。',
    'The band of {host} includes published values for {polymer}.':
        '{host} 的区间包含 {polymer} 的已发表数值。',
}

__all__ = ["SPUTTER_TOOLKIT_ZH"]
