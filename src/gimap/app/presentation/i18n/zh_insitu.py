"""Chinese of the in-situ series work: the timeline under the Series map, the online watch (pause, settle time,
the map while watching, the log of skipped files, the results database), the chained fit of an In-situ series
(Predict where the chain breaks, the backward pass, the flags) and Compare's Results Database…."""

INSITU_ZH = {
    # -- the timeline (analyze/presentation/bindings/series_timeline.py, views/series_view.py) --
    "χ² to the frame before (timeline)": "与前一帧的 χ²（时间线）",
    "χ² to the reference frame (timeline)": "与参考帧的 χ²（时间线）",
    "χ² to the frame before": "与前一帧的 χ²",
    "χ² to frame {n}": "与第 {n} 帧的 χ²",
    "jumps": "跳变",
    "The timeline needs a map of two frames or more": "时间线需要至少两帧的热图",
    "χ² of two frames: their difference in units of the noise of each x column (taken from the series), about 1 for "
    "frames that differ by noise alone. Dashed lines: stage boundaries; ▲ jumps; × odd frames. Click a point to show "
    "that frame.":
        "两帧的 χ²：两帧之差以每个 x 列的噪声（由序列本身估计）为单位，只差噪声的两帧约为 1。虚线：阶段边界；▲ 跳变；"
        "× 异常帧。点击某点显示那一帧。",
    "Timeline as Table…": "时间线（表格）…",
    "CSV: every frame's χ² to the frame before and to the reference frame, the jumps, its stage and whether it is odd; "
    "a JSON record next to it":
        "CSV：每一帧与前一帧和与参考帧的 χ²、跳变、所在阶段以及是否异常；旁边附 JSON 记录",
    "Reference frame": "参考帧",
    "The frame the timeline “χ² to the reference frame” compares every frame with (its number on the frame axis; the "
    "nearest frame of the map when it holds no frame of that number)":
        "时间线“与参考帧的 χ²”用来与每一帧比较的帧（按帧轴上的编号；热图中没有该编号时取最近的一帧）",
    "Export the Timeline": "导出时间线",
    "Could not export the timeline: {error}": "无法导出时间线：{error}",
    # -- the online watch (analyze/presentation/views/analyze_steps_view.py, bindings/watch_online.py) --
    "Pause Watching": "暂停监视",
    "No looks at the folder until you resume; what is written meanwhile is listed then":
        "继续之前不再查看文件夹；其间写入的文件在继续后列出",
    "Add New Frames to the Series Map": "把新帧加入序列热图",
    "Reduce every complete new frame with this set-up and append its curve to the Series map, its traces and timeline "
    "(only curves are kept, not frames)":
        "用当前设置约简每个写完的新帧，并把它的曲线加入序列热图、随帧变化的曲线和时间线（只保留曲线，不保留帧）",
    "Settle Time": "静置时间",
    "Two Looks (2 s)": "两次查看（2 秒）",
    "How long a new file must stay unchanged before it is read: a frame still being written is never read half-written":
        "新文件要保持不变多久才读取：正在写入的帧不会被读到一半",
    "Record Every Frame in the Results Database": "把每一帧记入结果数据库",
    "Write the curve of every frame of the map into gimap_analysis/gimap_series.sqlite next to the data (Compare reads "
    "it)":
        "把热图中每一帧的曲线写入数据旁边的 gimap_analysis/gimap_series.sqlite（比较页可以读取）",
    "Skipped Files…": "跳过的文件…",
    "Every file the watch could not take, when and why": "监视时未能处理的每个文件、时间和原因",
    "Fit New Frames in Fitting…": "在拟合中拟合新帧…",
    "Export every new frame and open the exports in Fitting ▸ In-situ series, which fits each new frame from the result "
    "of the one before":
        "导出每个新帧，并在“拟合 ▸ 原位序列”中打开导出的文件，每个新帧从上一帧的结果开始拟合",
    "Save Results Database…": "保存结果数据库…",
    "Every frame's curve, with its frame, file, time and the settings of the map, into a results database "
    "(gimap_series.sqlite): Compare reads it, and a watch can add to it":
        "把每一帧的曲线连同帧号、文件、时间和热图的设置写入结果数据库（gimap_series.sqlite）：比较页可以读取，监视时可以继续写入",
    "Watching paused: frames written meanwhile are listed when you resume.": "监视已暂停：其间写入的帧在继续后列出。",
    "Watching {folder} again …": "继续监视 {folder} …",
    "Skipped Files": "跳过的文件",
    "No file was skipped while watching.": "监视时没有跳过任何文件。",
    "{time}  {file}  {what}: {reason}": "{time}  {file}  {what}：{reason}",
    "waiting": "等待中",
    "failed": "失败",
    "no curve": "没有曲线",
    "Could not write the results database: {error}": "无法写入结果数据库：{error}",
    "Saved {n} frames to {name}": "已把 {n} 帧保存到 {name}",
    "Watch a folder first: Fit New Frames in Fitting fits the frames written into it.":
        "请先监视一个文件夹：“在拟合中拟合新帧”拟合写入该文件夹的帧。",
    "Save Results Database": "保存结果数据库",
    "Results database": "结果数据库",
    # -- the chained fit of an In-situ series (fitting/presentation/views/fit_series_view.py, single/series_chain.py) --
    "Fits as NXcanSAS…": "拟合结果（NXcanSAS）…",
    "For every fitted frame the points fitted and the model, one HDF5 file (SasView and other NXcanSAS programs open it)":
        "每个已拟合帧的拟合数据点和模型，写入一个 HDF5 文件（SasView 等 NXcanSAS 程序可以打开）",
    "Results Database…": "结果数据库…",
    "Every frame's fit into the series' results database (gimap_series.sqlite), beside the curves Analyze recorded":
        "把每一帧的拟合写入序列的结果数据库（gimap_series.sqlite），与分析页记录的曲线放在一起",
    "The previous result; Predict parameters where the chain breaks": "上一帧的结果；链断开处用“预测参数”",
    "Then a backward pass: each frame from the result of the frame after it": "然后反向再拟合一遍：每一帧从下一帧的结果开始",
    "Once every frame is done, from the last but one to the first, each frame is refined from the result of the frame "
    "after it; the better of its two fits is kept (its row says “backward”)":
        "所有帧拟合完后，从倒数第二帧到第一帧，每一帧从下一帧的结果开始精修；保留两次拟合中较好的一次（该行注明“反向”）",
    "Draw the trend against the frame, or against the time since the first fitted frame (when the curves say when their "
    "frames were measured)":
        "按帧或按距第一个已拟合帧的时间绘制趋势（曲线文件写明测量时间时）",
    "from the previous frame, Predict where the chain breaks": "从上一帧开始，链断开处预测",
    "Flags": "标记",
    "at a bound": "在边界上",
    "beam tail": "光束拖尾",
    "undetermined": "未确定",
    "chain broke": "链断开",
    "backward": "反向",
    "Backward pass: {count} frames, each from the result of the frame after it.":
        "反向拟合：{count} 帧，每一帧从下一帧的结果开始。",
    "Backward pass: {count} frames fitted better from the frame after them.": "反向拟合：{count} 帧从下一帧开始拟合得更好。",
    "Predict parameters needs 1–3 particles: where the chain breaks the frame is refined only.":
        "“预测参数”需要 1–3 种颗粒：链断开处只做精修。",
    "No fitted frame to save.": "没有可保存的已拟合帧。",
    "Saved the fits of {count} frames to {name}.": "已把 {count} 帧的拟合保存到 {name}。",
    "Saved {count} frames' fits to {name}.": "已把 {count} 帧的拟合保存到 {name}。",
    "Save Fits as NXcanSAS": "把拟合保存为 NXcanSAS",
    # -- Compare ▸ Add Series ▸ Results Database… (compare/presentation/series_input.py, application/use_cases.py) --
    "A series Analyze recorded in a results database (gimap_series.sqlite): every frame's curve, without its frames":
        "分析页记录在结果数据库（gimap_series.sqlite）中的序列：每一帧的曲线，不需要原始帧",
    "Results Database": "结果数据库",
    "{name} holds {n} curves of series. One is one series — which one?": "{name} 中有 {n} 条序列曲线。一条就是一个序列——选哪一条？",
    "{series} · {curve} · {n} frames": "{series} · {curve} · {n} 帧",
    "Results Database of a Series": "序列的结果数据库",
    "The results database holds fewer than two frames of {name} · {curve}.": "结果数据库中 {name} · {curve} 不到两帧。",
    "{name} is not a GIMaP results database ({error}).": "{name} 不是 GIMaP 结果数据库（{error}）。",
    "{name} was written by a newer GIMaP (database schema {schema}); this version reads schema {known}.":
        "{name} 由更新版本的 GIMaP 写入（数据库结构 {schema}）；本版本只能读取结构 {known}。",
    "{name} holds no series.": "{name} 中没有序列。",
    "{name} holds no curve “{curve}” of the series “{series}”.": "{name} 中没有序列“{series}”的曲线“{curve}”。",
}

__all__ = ["INSITU_ZH"]
