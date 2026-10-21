"""Chinese for fit1d (Predict parameters) and the Fit / Results steps of Fitting's single-curve page and the series:
the method, what the fits resolve by themselves, the values in Results, the compositions tried, following the
previous frame (fitting/presentation/single/*.py, views/fit_*_view.py, application/single_fit1d.py,
domain/fit*_edges.py, domain/fit1d_follow.py)."""

FIT1D_ZH = {
    'Which files of the folder are curves (* stands for any text), or detector frames (*.cbf, *.tif, *.edf, *.nxs …): each frame is then cut as Analyze cuts it':
        '文件夹里哪些文件是曲线（* 代表任意文字），或者探测器帧（*.cbf、*.tif、*.edf、*.nxs …）：这时每一帧都按 Analyze 的方式切线',
    "Take Analyze's Set-up":
        '采用 Analyze 的设置',
    "Cut the frames with Analyze's set-up as it is now (geometry, masks and corrections, the cut, Sum N, the halves sent to Fitting). Set it up on one frame of the series in Analyze first.":
        '用 Analyze 现在的设置切每一帧（几何、掩膜与校正、切线、N 帧相加、送去拟合的两半）。请先在 Analyze 里用这个序列的一帧设置好。',
    'Each frame is cut as Analyze cuts {frame}: instrument {profile}, αi {incidence}°, Sum {sum}{pinned}.':
        '每一帧都按 Analyze 切 {frame} 的方式切线：仪器 {profile}，αi {incidence}°，{sum} 帧相加{pinned}。',
    ', the cut at rows {low}–{high} of that frame':
        '，切线在那一帧的第 {low}–{high} 行',
    'The frames cannot be cut yet: {why}':
        '这些帧还不能切线：{why}',
    "Analyze's set-up is taken: the frames are cut again with it.":
        '已采用 Analyze 的设置：各帧按它重新切线。',
    '{count} detector frames listed; each is cut as in Analyze.':
        '列出了 {count} 个探测器帧；每一帧都按 Analyze 的方式切线。',
    'Analyze is not open.':
        'Analyze 还没有打开。',
    "Analyze has no frame set up yet: open one frame of this series in Analyze, set its geometry, masks and cut, then take Analyze's set-up here.":
        'Analyze 里还没有设置好的帧：请在 Analyze 中打开这个序列的一帧，设好几何、掩膜和切线，再在这里采用 Analyze 的设置。',
    'Nothing to fit: this frame has no usable curve.':
        '没有可拟合的：这一帧没有可用的曲线。',
    'Another solution fits equally well:':
        '另有同样好的解：',
    'Not pinned down by this curve:':
        '这条曲线定不住：',
    'Populations without contribution:':
        '没有贡献的群：',
    'and {count} more (hover)':
        '另有 {count} 对（悬停查看）',
    'Predicted by fit1d.':
        '由 fit1d 预测。',
    'Choose a method and press Fit; values that stop at an edge of their range are dealt with by the fit.':
        '选择方法后点拟合；停在范围边上的数值由拟合自动处理。',
    'Predict parameters (fit1d)':
        '预测参数（fit1d）',
    'The particles in Model':
        '模型中的粒子',
    'Try 1–3 vertical cylinders':
        '尝试 1–3 群竖直圆柱',
    'Try spheres and vertical cylinders (1–3)':
        '尝试球和竖直圆柱（1–3 群）',
    "Trying compositions: the Model's fixed values and ranges are not passed (the populations differ between compositions). Use a candidate, then predict again with its particles to use them.":
        '尝试多种组成时，模型中的固定值和范围不会传入（不同组成的群不一样）。先采用一个候选，再用它的粒子重新预测，才会用上这些设定。',
    'Model has no particles: add Sphere / Vertical cylinder particles, or choose “Try …” compositions.':
        '模型中没有粒子：添加球 / 竖直圆柱，或选择“尝试……”的组成。',
    'Trying compositions with fit1d':
        '正在用 fit1d 尝试多种组成',
    'Fitting the particles in Model with fit1d':
        '正在用 fit1d 拟合模型中的粒子',
    'Loading fit1d':
        '正在载入 fit1d',
    'fit1d finished':
        'fit1d 完成',
    'No σ in the file: fit1d assumes Poisson errors; the selected systematic error floor is added in quadrature.':
        '文件里没有 σ：fit1d 假定泊松误差，并按平方和加上所选的系统误差下限。',
    '{name}: your range {low}–{high}{unit} is narrower than what this curve can measure ({measurable_low}–{measurable_high}{unit}). If the fit stops at an edge of your range, widen it.':
        '{name}：你的范围 {low}–{high}{unit} 比这条曲线能测的范围（{measurable_low}–{measurable_high}{unit}）窄。如果拟合停在你的范围边上，请放宽它。',
    '{name}: your range reaches {side_value}{unit}, beyond what this curve can resolve ({measurable_low}–{measurable_high}{unit}); values out there are not determined by it.':
        '{name}：你的范围延伸到 {side_value}{unit}，超出了这条曲线能分辨的范围（{measurable_low}–{measurable_high}{unit}）；那里的数值不由这条曲线决定。',
    'upper':
        '上',
    'lower':
        '下',
    'The low-q term {name} stops at the {side} edge of its range.':
        '低 q 项 {name} 停在其范围的{side}边。',
    'Constraints passed to fit1d: {items}.':
        '传给 fit1d 的约束：{items}。',
    ' (Model)':
        '（模型）',
    ' (widened here)':
        '（此处放宽）',
    'No constraints: fit1d searched its default ranges for this q range.':
        '没有约束：fit1d 在这个 q 范围的默认范围内搜索。',
    'This q range measures R {r_low}–{r_high} nm and D {d_low}–{d_high} nm.':
        '这个 q 范围能测 R {r_low}–{r_high} nm、D {d_low}–{d_high} nm。',
    'fit1d tried {count} compositions; the best, {composition} (χ²/point {chi2}), is in Results and in Model. Compare the candidates in Results.':
        'fit1d 尝试了 {count} 种组成；最好的 {composition}（χ²/点 {chi2}）已放进结果和模型。请在结果中比较各候选。',
    '{state}: {low}–{high} {unit}':
        '{state}：{low}–{high} {unit}',
    'ok':
        '正常',
    'Every composition fit1d tried, best (lowest BIC) first. Compositions within ΔBIC ≤ 10 of the best cannot be told apart on this curve: choose from what you know of the sample.':
        'fit1d 尝试过的每一种组成，最好的（BIC 最低）在前。与最好者相差 ΔBIC ≤ 10 的组成在这条曲线上分不开：请根据你对样品的了解来选择。',
    'composition':
        '组成',
    'note':
        '备注',
    'parameter':
        '参数',
    'value ± error':
        '值 ± 误差',
    'equivalent range':
        '等价解范围',
    'in Results':
        '已在结果中',
    'indistinguishable from {others}':
        '与 {others} 分不开',
    'a parameter at an edge':
        '有参数触边界',
    'not all pinned down':
        '有参数定不住',
    'misses the curve':
        '未描述好曲线',
    'unusual curve':
        '曲线异常',
    'robust search':
        '稳健搜索',
    'Several compositions fit equally well; the curve alone does not decide between them.':
        '几种组成拟合得一样好；仅凭这条曲线无法在它们之间做出选择。',
    'The model does not describe the curve within the error bars (quality flag).':
        '模型没有在误差范围内描述曲线（质量标记）。',
    'The curve is unlike the curves the tool was tested on (out-of-distribution flag).':
        '这条曲线与工具测试过的曲线不同（分布外标记）。',
    'No error bars were given: Poisson errors were assumed.':
        '没有给出误差棒：假定为泊松误差。',
    'Method':
        '方法',
    'Refine the values in Model':
        '精修模型中的数值',
    'Finds the particles and their sizes from the curve (network proposals and a physical fit), with errors and equivalent solutions. The first step for a GISAXS cut.':
        '从曲线找出粒子及其尺寸（网络给初值，再做物理拟合），给出误差和等价解。GISAXS 线切的第一步。',
    "Least squares from the values in Model, with Fitting's own model: fast; finds the nearest good fit.":
        '从模型中的数值出发，用拟合页自己的模型做最小二乘：快，找到最近的好拟合。',
    'Recommended for this curve: a GISAXS cut not fitted yet.':
        '推荐用于这条曲线：尚未拟合的 GISAXS 线切。',
    'Particles':
        '粒子',
    'Try …: fit1d fits every composition of these shapes and lists them as candidates, best first. The particles in Model: fit1d fits these, keeping the values you fixed and the ranges you set.':
        '尝试……：fit1d 拟合这些形状的每一种组成，按好坏列为候选。模型中的粒子：fit1d 拟合这些粒子，保留你固定的值和设定的范围。',
    'Resolve values at the edge of their range automatically':
        '自动处理停在范围边上的数值',
    'When a value stops at the edge of its range, widen that side (or fix a size spread at its lower limit), fit again and keep it if the fit is not worse. Results lists what was done.':
        '数值停在范围边上时，放宽那一边（或把尺寸分布宽度固定在下限），重新拟合；拟合不变差就保留。结果页列出做了什么。',
    'Options':
        '选项',
    'Predict (fit1d)':
        '预测（fit1d）',
    'Evaluations at most (Refine, Search)':
        '最多计算次数（精修、搜索）',
    '1D Predict Batch Window…':
        '一维预测批量窗口…',
    '1D Predict: a list of curve files fitted one after another, with their results. To fit every curve of a folder with this model, use Fit Every Curve in This Folder…':
        '一维预测：逐个拟合一组曲线文件并列出结果。要用这个模型拟合一个文件夹里的每条曲线，请用“拟合此文件夹的每条曲线…”',
    'Fit a Folder of Curves…':
        '拟合一个文件夹的曲线…',
    'In-situ series: every curve of a folder (a series, or many samples), each fitted with the model of this page — fit one curve here first, or let each curve be predicted':
        '原位序列：一个文件夹里的每条曲线（一个序列，或许多样品），每条都用本页的模型拟合——先在这里拟合一条曲线',
    'Refine These Values':
        '精修这些数值',
    "Refine the values now in Model with Fitting's own model (least squares)":
        '用拟合页自己的模型精修模型中现在的数值（最小二乘）',
    'Fit Again':
        '再拟合一次',
    'Run the same method again, with the values, fixed values and ranges now in Model':
        '用模型中现在的数值、固定值和范围，再运行一次同样的方法',
    'Fit Every Curve in This Folder…':
        '拟合此文件夹的每条曲线…',
    'In-situ series: every curve of this folder, each fitted from this model':
        '原位序列：此文件夹里的每条曲线，都从这个模型开始拟合',
    'Fix':
        '固定',
    'Keep the selected values as they are in the next fit':
        '下次拟合时保持所选数值不变',
    'Let It Fit':
        '参与拟合',
    'Let the next fit change the selected values':
        '下次拟合时让所选数值可变',
    'Range…':
        '范围…',
    'Show the min–max range of the selected value in Model':
        '在模型中显示所选数值的最小–最大范围',
    'Select values to fix them or let them fit, then Fit Again; right-click for the same.':
        '选中数值后可以固定或让它参与拟合，再点“再拟合一次”；右键菜单也可以。',
    'Other solutions':
        '其他解',
    'Use This Composition':
        '使用这个组成',
    'Predict again with this composition (its values as the start; values at an edge resolved) — Undo brings the previous model back':
        '用这个组成重新预测（以它的数值为起点，自动处理停在边上的数值）——撤销可恢复之前的模型',
    'Constraints and ranges':
        '约束与范围',
    'Done automatically:':
        '已自动处理：',
    '{names}: fixed in Model. Fit Again to keep them there.':
        '{names}：已在模型中固定。点“再拟合一次”即按固定值拟合。',
    '{names}: the next fit changes them. Fit Again to fit them.':
        '{names}：下次拟合会改变它们。点“再拟合一次”即参与拟合。',
    'The values selected are not in Model (a global term of fit1d): set them in Model.':
        '所选数值不在模型中（fit1d 的全局项）：请在模型中设置。',
    'The range of {name} is shown in Model: change its min or max there, then Fit Again.':
        '{name} 的范围已在模型中显示：在那里修改最小或最大值，再点“再拟合一次”。',
    'Other solutions ({count})':
        '其他解（{count}）',
    'Other compositions ({count})':
        '其他组成（{count}）',
    "Compositions tried: {composition}; the best, {fitted}, is shown (Model's values and ranges not used).":
        '尝试的组成：{composition}；显示最好的 {fitted}（未使用模型的数值和范围）。',
    'A parameter stopped at the edge of its range: fitting again with it resolved':
        '有参数停在范围边上：处理后重新拟合',
    'a smaller size spread cannot be told apart from none on this curve':
        '这条曲线分不出更小的尺寸分布宽度和没有分布',
    'a better order cannot be told apart on this curve':
        '这条曲线分不出更好的有序度',
    'D = 2R means touching particles':
        'D = 2R 表示粒子相接触',
    '{name} stopped at the {side} edge of the default range ({old}{unit}): widened to {new}{unit} and fitted again (χ²/point {before} → {after}).':
        '{name} 停在默认范围的{side}边（{old}{unit}）：放宽到 {new}{unit} 并重新拟合（χ²/点 {before} → {after}）。',
    '{name} stopped at the {side} edge of your range ({old}{unit}): widened to {new}{unit} and fitted again (χ²/point {before} → {after}).':
        '{name} 停在你设定范围的{side}边（{old}{unit}）：放宽到 {new}{unit} 并重新拟合（χ²/点 {before} → {after}）。',
    'Widening or fixing {names} made the fit worse (χ²/point {before} → {after}): the fit before is kept, with them at the edge.':
        '放宽或固定 {names} 使拟合变差（χ²/点 {before} → {after}）：保留之前的拟合，它们仍在边上。',
    '{name} = {value}{unit} is still at the {side} edge of its range: the curve does not determine it beyond there.':
        '{name} = {value}{unit} 仍在范围的{side}边：这条曲线定不住超出那里的值。',
    '{name} stopped at the {side} edge of its range ({old}): widened to {new} and fitted again (χ²ᵣ {before} → {after}).':
        '{name} 停在范围的{side}边（{old}）：放宽到 {new} 并重新拟合（χ²ᵣ {before} → {after}）。',
    '{name} = {value} is still at the {side} edge of its range: the curve does not determine it beyond there.':
        '{name} = {value} 仍在范围的{side}边：这条曲线定不住超出那里的值。',
    'Widening {count} parameters made the fit worse (χ²ᵣ {before} → {after}): the fit before is kept, with them at the edge.':
        '放宽 {count} 个参数使拟合变差（χ²ᵣ {before} → {after}）：保留之前的拟合，它们仍在边上。',
    '{name}: every value {relation} {value}{unit} fits the curve as well (χ² within {tolerance} of the best), so the curve gives only this bound. Reported at {value}{unit}, the least extreme of them; the other values are fitted with it there.':
        '{name}：所有 {relation} {value}{unit} 的值都拟合得一样好（χ² 与最佳相差不超过 {tolerance}），曲线只给出这个界。报告为其中最不极端的 {value}{unit}，其他数值在此值下重新拟合。',
    '{name} = {value}{unit} sits at the {side} limit of the model, and the curve wants a value beyond it: the model may be missing something (another population, the background, the resolution).':
        '{name} = {value}{unit} 停在模型的{side}限上，曲线还想要超出它的值：模型可能缺了什么（另一群粒子、背景或分辨率项）。',
    '{name}: every value {relation} {value} fits the curve as well (χ² within {tolerance} of the best), so the curve gives only this bound. Reported at {value}, the least extreme of them, and fixed there in Model; the other values are fitted with it.':
        '{name}：所有 {relation} {value} 的值都拟合得一样好（χ² 与最佳相差不超过 {tolerance}），曲线只给出这个界。报告为其中最不极端的 {value}，并在模型中固定于此；其他数值随之拟合。',
    'A value stays at the edge of its range: finding the least extreme value the curve allows':
        '有数值仍停在范围边上：寻找曲线允许的最不极端的值',
    'Fitting the continuation of the previous frame':
        '正在拟合上一帧的延续',
    'Followed the previous frame: its continuation fits as well as a free fit (χ² {follow} against {free}, tolerance {tolerance}).':
        '跟随上一帧：它的延续与自由拟合一样好（χ² {follow} 对 {free}，容差 {tolerance}）。',
    'Did not follow the previous frame: a free fit is better by more than the tolerance (χ² {free} against {follow}, tolerance {tolerance}) — the sample changed here.':
        '未跟随上一帧：自由拟合好得超过容差（χ² {free} 对 {follow}，容差 {tolerance}）——样品在这里变了。',
    'one-sided bound':
        '单侧界',
    ' (near the previous frame)':
        '（上一帧附近）',
    ' (a bound: every larger value fits as well)':
        '（界：更大的值都一样好）',
    ' (a bound: every smaller value fits as well)':
        '（界：更小的值都一样好）',
    'Predict each frame (fit1d)':
        '逐帧预测（fit1d）',
    "fit1d on every frame with the particles, fixed values and ranges of Single analysis' model and the settings of its Fit step; values at an edge are resolved as there":
        '对每一帧做 fit1d：粒子、固定值和范围用单条分析的模型，设置用它的拟合步骤；停在边上的数值同样自动处理',
    'Prefer the solution that continues the previous frame':
        '优先选择延续上一帧的解',
    'When a frame has solutions that fit equally well, keep the one closest to the previous frame (the smoothest series); a clearly better other solution is still taken, and said':
        '一帧有几个同样好的解时，保留最接近上一帧的那个（序列最平滑）；明显更好的其他解仍会被采用，并注明',
    "Predict each frame needs Sphere / Vertical cylinder particles in Single analysis' model.":
        '逐帧预测需要单条分析的模型中有球 / 竖直圆柱粒子。',
}

__all__ = ["FIT1D_ZH"]
