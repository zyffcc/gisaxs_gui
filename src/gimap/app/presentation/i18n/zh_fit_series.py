"""Chinese for the halves of a signed cut in Fitting and for Fitting ▸ In-situ series."""

FIT_SERIES_ZH = {
    # A fit takes one curve of a cut: the mean of its halves, or one half (shared/signed_cut.py,
    # fitting/application/single_fit.py and series_fit.py write these in English; the pages translate them).
    "Both halves on |q| cannot be fitted by this method: the mean of both halves is used.":
        "这个方法不能拟合“|q| 上的两半”：改用两半取平均。",
    "No points on the q > 0 half": "q > 0 一半没有数据点",
    "No points on the q < 0 half": "q < 0 一半没有数据点",
    "too few points in the fitting range": "拟合范围内的数据点太少",
    "the curve could not be read": "无法读取这条曲线",
    "No curves to find stages in.": "没有可用来划分阶段的曲线。",
    # What the curve of a fit is made of (the note of shared/signed_cut.prepare_cut). The count is shown by its
    # form (assistant/domain/gisaxs_cut.HOLES_SENTENCE); the sentences without a value are entries as written.
    "{averaged} points averaged, {single} from one half alone ({inside} of them inside that range, where the other half has no points)":
        "{averaged} 个点是两半的平均，{single} 个点只来自一半（其中 {inside} 个在该范围内，另一半在那里没有数据点）",
    "the qy > 0 half has no points": "qy > 0 这一半没有数据点",
    "the qy < 0 half has no points": "qy < 0 这一半没有数据点",
    "the qy > 0 half is too sparse to average with": "qy > 0 这一半的数据点太稀疏，无法与另一半取平均",
    "the qy < 0 half is too sparse to average with": "qy < 0 这一半的数据点太稀疏，无法与另一半取平均",
    # The quick physical fit (fitting/bootstrap.create_quick_fit) on a curve with too few points; shown with trf.
    "Only {points} points on the q > 0 half: this fit needs at least {minimum}":
        "q > 0 一半只有 {points} 个数据点：这种拟合至少需要 {minimum} 个",
    "Only {points} points on the q < 0 half: this fit needs at least {minimum}":
        "q < 0 一半只有 {points} 个数据点：这种拟合至少需要 {minimum} 个",
    "Only {points} points in the mean of both halves: this fit needs at least {minimum}":
        "两半的平均只有 {points} 个数据点：这种拟合至少需要 {minimum} 个",
    "Only {points} points in the curve: this fit needs at least {minimum}":
        "曲线只有 {points} 个数据点：这种拟合至少需要 {minimum} 个",
    # -- In-situ series: its own halves (fitting/presentation/single/series_halves.py) ---------------------
    "Which half of each cut is fitted, in every frame of the series: the mean of both halves, or one half.":
        "序列里每一帧拟合切线的哪一半：两半的平均，或其中一半。",
    "from the fit inputs written by Analyze": "来自分析写出的拟合输入文件",
    "as Analyze sends cuts to Fitting": "按分析送去拟合时的设置",
    "as in Single analysis": "与单条分析相同",
    "as saved in the project": "按项目中保存的",
    "{side} ({source})": "{side}（{source}）",
    "{side}; {range}": "{side}；{range}",
    "These curves have q on one side only: there are no halves to choose.": "这些曲线只有一侧的 q：没有两半可选。",
    "Single analysis fits both halves on |q|; a series fits one curve per frame: their mean.":
        "单条分析拟合的是“|q| 上的两半”；序列每帧只拟合一条曲线：用两半的平均。",
    # -- the Curves and Start steps -------------------------------------------------------------------
    "A folder of curves, e.g. the gimap_analysis/fit_input folder of Analyze's Batch Export or Send Series to Fitting, or a folder of detector frames (CBF, TIFF, EDF, NeXus): each frame is then cut as Analyze cuts it.":
        "一个曲线文件夹，例如分析的批量导出或“把序列发送到拟合”写出的 gimap_analysis/fit_input 文件夹；或一个探测器帧文件夹（CBF、TIFF、EDF、NeXus）：这时每一帧都按分析的方式切线。",
    "The folder of the curves, e.g. the gimap_analysis/fit_input folder Analyze writes":
        "曲线所在的文件夹，例如分析写出的 gimap_analysis/fit_input 文件夹",
    "While the series runs, fit curves that appear in the folder (e.g. written by Analyze during a measurement). Can be switched while the series runs.":
        "序列运行时，也拟合文件夹里新出现的曲线（例如测量时分析写出的曲线）。序列运行中也可以勾选或取消。",
    "The series uses the model, fitting range and left-out points of Single analysis: set them there on one representative curve":
        "序列使用单条分析里的模型、拟合范围和排除的点：请在那里用一条有代表性的曲线设置好",
    "{count} curves listed, in {place} of the chosen folder.": "列出了 {count} 条曲线，位于所选文件夹下的 {place}。",
    "No curves here, but in {places}: choose one of them, or tick “Also in subfolders”.":
        "这里没有曲线，但 {places} 里有：请选择其中一个，或勾选“包括子文件夹”。",
    "cannot be used: frames {frames}": "无法使用：帧 {frames}",
    "No stages for this run: it started before they were found.": "这次运行没有阶段：阶段还没找到就开始了。",
    "No stages are known: every frame starts from the previous frame's result.":
        "没有可用的阶段：每一帧都从上一帧的结果开始。",
    "Model not set up yet: fit one frame in Single analysis": "模型还没有设置：请先在单条分析中拟合一帧",
    "The model is still the starting guess: nothing was set up or fitted in Single analysis. Start would fit this guess to every frame.":
        "模型还是初始猜测：单条分析里没有设置或拟合过。现在开始，会用这个猜测去拟合每一帧。",
    "Set Up on Frame {n} in Single Analysis": "在单条分析中用第 {n} 帧设置",
    # -- a frame: its line in the list, the plot and the note above it ----------------------------------
    "poor fit": "拟合差",
    "q > 0 half only": "只有 q > 0 一半",
    "q < 0 half only": "只有 q < 0 一半",
    "{points} points": "{points} 个点",
    "{count} left out": "排除 {count} 个",
    "{single} from one half alone": "{single} 个只来自一半",
    "{points} points, {single} of them from one half alone": "{points} 个点，其中 {single} 个只来自一半",
    "Frame {frame} cannot be fitted: {reason}": "第 {frame} 帧无法拟合：{reason}",
    "Curve file was not found: {path}": "找不到曲线文件：{path}",
    # -- the run and Watch (series_watch.py) --------------------------------------------------------------
    "Waiting for new curves in {folder} … (Stop to end)": "正在等待 {folder} 里的新曲线…（点“停止”结束）",
    "The folder of the curves cannot be read: waiting for it … (Stop to end)":
        "曲线所在的文件夹无法读取：正在等待它恢复…（点“停止”结束）",
    "{done} frames · waiting for new curves": "{done} 帧 · 正在等待新曲线",
    "{count} curves changed after they were read: fitted again.": "{count} 条曲线在读取之后又有改动：已重新拟合。",
    "{count} of them are outside the chosen frames and are not fitted.": "其中 {count} 条不在所选的帧里，不拟合。",
    "Series stopped: {count} frames fitted, {failed} failed.": "序列已停止：拟合了 {count} 帧，失败 {failed} 帧。",
    # the end of a run with poor fits (single/series_page.END_SENTENCES)
    "Series done: {count} frames fitted, {poor} of them poor fits (χ²ᵣ above {limit}), {failed} failed.":
        "序列完成：拟合了 {count} 帧，其中 {poor} 帧拟合差（χ²ᵣ 大于 {limit}），失败 {failed} 帧。",
    "Series stopped: {count} frames fitted, {poor} of them poor fits (χ²ᵣ above {limit}), {failed} failed.":
        "序列已停止：拟合了 {count} 帧，其中 {poor} 帧拟合差（χ²ᵣ 大于 {limit}），失败 {failed} 帧。",
    "The results of the last run are not saved.\nSave the table of every frame first?":
        "上一次运行的结果还没有保存。\n要先保存每一帧的结果表吗？",
    # -- the Results step (series_results.py) -------------------------------------------------------------
    "Points": "点数",
    "Halves": "两半",
    "mean": "平均",
    "The points the frame was fitted on": "这一帧拟合所用的点数",
    "What those points are: the mean of both halves of the cut, or one half": "这些点是什么：切线两半的平均，或其中一半",
    "{count} fitted": "拟合 {count} 帧",
    "{count} failed": "失败 {count} 帧",
    "{count} not converged": "未收敛 {count} 帧",
    "{count} stopped": "已停止 {count} 帧",
    "{count} poor fits": "拟合差 {count} 帧",
    "Poor fit (χ²ᵣ above {limit}): frames {frames}": "拟合差（χ²ᵣ 大于 {limit}）：帧 {frames}",
    "{points} points per frame": "每帧 {points} 个点",
    "{low}–{high} points per frame": "每帧 {low}–{high} 个点",
    "Changed after the fit: Start again": "拟合之后有改动：请重新开始",
    "The model was the starting guess of Single analysis: it was never set up or fitted there.":
        "模型是单条分析的初始猜测：没有在那里设置或拟合过。",
    "Logarithmic axis of the value": "该值用对数坐标轴",
    # -- Single analysis: both halves on |q|, each in its colour (single/points.py) -------------------------
    "measured (+q)": "测量（+q）",
    "measured (−q)": "测量（−q）",
}

__all__ = ["FIT_SERIES_ZH"]
