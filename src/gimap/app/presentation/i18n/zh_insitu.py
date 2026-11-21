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
    "Fitting ▸ In-situ series lists the curves written so far: tick Watch for new curves and press Start — every new "
    "frame is then fitted from the result of the one before.":
        "“拟合 ▸ 原位序列”列出了目前写好的曲线：勾选“监视新曲线”并按“开始”——之后每个新帧都从上一帧的结果开始拟合。",
    "Save Results Database": "保存结果数据库",
    "Results database": "结果数据库",
}

__all__ = ["INSITU_ZH"]
