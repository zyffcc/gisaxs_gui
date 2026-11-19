"""Chinese texts of the live view of the Sputter main screen: what a running Analyze is doing and has found so far.

Keys are the exact English of ``features/sputter`` (``presentation/views/sputter_live_view.py``, ``live_panel.py``,
``live_pictures.py``); templates keep their ``{fields}``. Numbers, units and the names of the networks are data.
The axis titles shared with the main screen (exit angle, count rate, time in the scan) are in ``zh_sputter_client.py``.
"""

SPUTTER_LIVE_ZH = {
    # -- the steps (views/sputter_live_view.py) ------------------------------------------------------------------------
    "Analysis at work": "分析进行中",
    "Read the frames": "读入帧",
    "Find the solutions": "求解结构",
    "Write the report": "写报告",
    "Draw the images": "绘制图像",
    "Click a step that has started to see what it found.": "点一个已开始的步骤，可以查看它的产出。",
    # -- what each step does (live_panel.py) -----------------------------------------------------------------------
    "Every detector frame is projected onto a grid of q_y and exit angle, without the pixels the mask leaves out, and "
    "the frames are summed in steps of 0.5 s. Left: the detector summed over sampled frames — red: left out by the "
    "mask, +: the direct beam, dashed: the sample horizon. Right: the frames read last. Below: the count rate of every "
    "frame read so far; it rises as the metal is deposited.":
        "把探测器的每一帧投影到 q_y–出射角网格上（掩膜去掉的像素不计），再按 0.5 s 归并。左：抽样帧求和后的探测器——"
        "红色为掩膜去掉的像素，+ 为直射束，虚线为样品地平线。右：最近读入的帧。下：已读入各帧的计数率，"
        "金属沉积后随之上升。",
    "For every time window the network proposes candidate structures; the physical model draws each as an image and "
    "the one closest to the measured image is kept. First the network input of every window is prepared and the "
    "network runs once for all of them; when the mode refines, every window has its candidates before the "
    "refinements start. Above: the window solved last, measured and model image. Below: the sizes found so far.":
        "网络为每个时间窗给出若干候选结构；物理模型把每个候选画成图样，留下与实测图最接近的那个。"
        "先为每个时间窗准备网络输入，网络一次处理全部时间窗；需要精修时，先给每个时间窗求出候选，再逐个精修。"
        "上：最近求解的时间窗，实测图与模型图。下：目前已求出的尺寸。",
    "The solutions of all windows go into a table, the plots of their evolution and a report.":
        "所有时间窗的解写入表格、演化图和报告。",
    "For every window the measured image and the image of every candidate are drawn for the main screen. The result "
    "opens when this step is done.":
        "为每个时间窗绘制实测图和各候选的模型图，供主屏显示。这一步完成后打开结果。",
    # -- the lines of the steps, the captions -----------------------------------------------------------------------
    "Step {index} of {count}: {name}": "第 {index} 步（共 {count} 步）：{name}",
    "Running for {time}": "已运行 {time}",
    "Beam centre ({col}, {row}) px": "束心 ({col}, {row}) px",
    "{done} of {total} frames read": "已读入 {done} / {total} 帧",
    "{frames} frames, {first}–{last} s": "{frames} 帧，{first}–{last} s",
    "{frames} frames, {first}–{last} s, deposition from {onset} s": "{frames} 帧，{first}–{last} s，沉积自 {onset} s 开始",
    "Read before for this scan": "此扫描此前已读入",
    "{solved} of {total} time windows": "已求解 {solved} / {total} 个时间窗",
    "Network {index} of {count}: {what}": "网络 {index}/{count}：{what}",
    "Network inputs ready for {done} of {total} time windows": "已准备 {done} / {total} 个时间窗的网络输入",
    "The network proposes candidates for {total} time windows…": "网络正在为 {total} 个时间窗给出候选结构…",
    "{solved} of {total} time windows solved ({images} model images computed)":
        "已求解 {solved} / {total} 个时间窗（已计算 {images} 幅模型图）",
    "{solved} of {total} time windows solved, {done} refined ({images} model images computed)":
        "已求解 {solved} / {total} 个时间窗，已精修 {done} 个（已计算 {images} 幅模型图）",
    "Whole-scan refinement: iteration {done} of at most {total}": "整条扫描联合精修：第 {done} 轮（最多 {total} 轮）",
    "The measured and the model image of a time window appear here once the first one is solved.":
        "求出第一个时间窗后，这里显示它的实测图与模型图。",
    "{family} fits the data better": "{family} 与数据更吻合",
    "{family} fits the data better (fit deviation {deviation})": "{family} 与数据更吻合（拟合偏差 {deviation}）",
    "{drawn} of {total} windows drawn": "已绘制 {drawn} / {total} 个时间窗",
    "{frames} frames summed · {masked} pixels left out by the mask (found in these frames: {dead} dead, {hot} hot, "
    "{shadow} of beamstop and holder)":
        "{frames} 帧求和 · 掩膜去掉 {masked} 个像素（其中由这些帧找出的：坏点 {dead}，热点 {hot}，挡束器及支架 {shadow}）",
    "fit deviation {deviation}": "拟合偏差 {deviation}",
    "The first time windows are being solved…": "正在求解最初的几个时间窗…",
    "The frames are being read…": "正在读入帧…",
    "This analysis package does not show what it finds while it works: its result opens at the end.":
        "这个分析包不显示中间结果：分析完成后打开结果。",
    # -- the pictures (live_pictures.py) ----------------------------------------------------------------------------
    "Detector, summed": "探测器（求和）",
    "Frames {first}–{last} s": "{first}–{last} s 的帧",
    "Frames read": "已读入的帧",
    "Count rate per pixel": "每像素计数率",
    "Start of the deposition": "沉积开始",
    # -- a sample measured after its deposition (live_static.py) ---------------------------------------------------
    "Read the frames of the sample": "读入样品的帧",
    "Find its solution": "求解结构",
    "Add it to the series": "归入系列",
    "Every frame of the sample is projected onto a grid of q_y and exit angle, without the pixels the mask leaves out; "
    "a frame with the direct or the specular beam not blocked is left out. A sample measured after its deposition has "
    "no frames from before it: the substrate part of its image comes from frames of another piece of the same "
    "substrate without metal (the bare substrate), or, without them, from the model of Si. Left: the detector summed — "
    "red: left out by the mask, +: the direct beam, dashed: the sample horizon. Right: the usable frames summed. "
    "Below: the series so far.":
        "把样品的每一帧投影到 q_y–出射角网格上（掩膜去掉的像素不计）；直射束或镜面反射没有挡住的帧不用。"
        "沉积之后测量的样品没有沉积之前的帧：它图像里的基底部分来自另一片同种基底、没有镀金属的帧（裸基底），"
        "没有给出时用 Si 的模型。左：探测器求和——红色为掩膜去掉的像素，+ 为直射束，虚线为样品地平线。右：可用帧之和。"
        "下：目前的系列。",
    "For each usable frame the network proposes candidate structures; the physical model draws each as an image and "
    "the one closest to the measured frame is kept. Above: the frame solved last, measured and model image. Below: "
    "the series so far, with every frame of this sample at its nominal thickness; the sample's value is their median. "
    "Quick and Refine solve four of the usable frames, Full every one.":
        "网络为每个可用帧给出若干候选结构；物理模型把每个候选画成图样，留下与实测帧最接近的那个。"
        "上：最近求解的帧，实测图与模型图。下：目前的系列，这个样品的每一帧都画在它的名义厚度处；样品的数值取它们的中位数。"
        "快速和精修模式取可用帧中的 4 帧，完整模式取全部。",
    "The sample's values — the median over its frames — go into the series, with its report and images.":
        "样品的数值（各帧的中位数）连同报告和图像归入系列。",
    "{usable} of {frames} frames usable": "{frames} 帧中 {usable} 帧可用",
    "Bare substrate: {usable} of {frames} frames usable": "裸基底：{frames} 帧中 {usable} 帧可用",
    "{solved} of {total} frames solved": "已求解 {solved} / {total} 帧",
    "{solved} of {total} frames solved ({total} of the {usable} usable)":
        "已求解 {solved} / {total} 帧（取 {usable} 个可用帧中的 {total} 帧）",
    "Usable frames of the sample, summed": "样品可用帧之和",
    "Usable frames of the bare substrate, summed": "裸基底可用帧之和",
    "left out: {name} ({reason})": "未用：{name}（{reason}）",
    "{count} saturated pixels: direct or specular beam not blocked": "{count} 个饱和像素：直射束或镜面反射没有挡住",
    "bare substrate: another piece without metal": "裸基底：另一片未镀金属的基底",
    "no bare substrate: the film is taken to lie on Si": "没有裸基底：认为膜长在 Si 上",
    "Frame {frame}": "第 {frame} 帧",
    "network": "网络",
    "refined": "精修",
    "The first frames are being solved…": "正在求解最初的几帧…",
    # -- the bar, After Deposition (views/sputter_main_view.py, static_flow.py) ------------------------------------
    "Nominal (nm)": "名义厚度（nm）",
    # -- reading the frames from disk and the processes (reading_texts.py) ------------------------------------------
    "disk {seconds} s per frame ({share} % of the time)": "磁盘每帧 {seconds} s（占用时的 {share} %）",
    "disk {seconds} s per frame": "磁盘每帧 {seconds} s",
    "Reading {done} of {total} frames": "正在读入第 {done} 帧（共 {total} 帧）",
    "Processes: {count}": "进程数：{count}",
    # -- what the analysis does now, in one line (live_status.py) --------------------------------------------------
    "Reading frames: {done} of {total}": "正在读入帧：{done} / {total}",
    "Reading the frames: finding the beam centre and the pixel mask": "正在读入帧：寻找束心，生成像素掩膜",
    "{frames} frames read; the networks start": "已读入 {frames} 帧，网络即将开始",
    "Starting network {family}": "正在启动网络 {family}",
    "Starting network {family} (the first of {count} to compare)": "正在启动网络 {family}（要比较的 {count} 个网络中的第一个）",
    "Network {family}: starting on {total} time windows": "网络 {family}：开始处理 {total} 个时间窗",
    "Network {family}: reading the network's input, window {done} of {total}":
        "网络 {family}：正在准备网络输入，第 {done}/{total} 个时间窗",
    "Network {family}: the network proposes candidates for {total} windows":
        "网络 {family}：网络正在为 {total} 个时间窗给出候选结构",
    "Network {family}: drawing the candidates with the forward model, window {done} of {total}":
        "网络 {family}：正在用正向模型绘制候选，第 {done}/{total} 个时间窗",
    "Network {family}: refining the best candidate of each window, {done} of {total}":
        "网络 {family}：正在精修各时间窗的最佳候选，{done}/{total}",
    "Joint fit: iteration {done} of at most {total}": "联合精修：第 {done} 次迭代（最多 {total} 次）",
    "Comparing the networks: {scores} → {family}": "比较网络：{scores} → {family}",
    "Network {family}: every window solved": "网络 {family}：所有时间窗已求解",
    "Preliminary result shown; the refinement of every window starts": "已显示初步结果；开始逐个精修时间窗",
    "Network {family}: refining every window": "网络 {family}：正在逐个精修时间窗",
    "Network {family}: refining every window, then the joint fit of the whole scan":
        "网络 {family}：正在逐个精修时间窗，之后联合精修整条扫描",
    "Joint fit: preparing (each window tries its neighbours' solutions, {windows} windows)":
        "联合精修：准备中（每个时间窗试用相邻时间窗的解，共 {windows} 个时间窗）",
    "Joint fit: working on iteration 1 of {total}": "联合精修：正在进行第 1/{total} 次迭代",
    "Joint fit: iteration {iteration} of {total}": "联合精修：第 {iteration}/{total} 次迭代",
    "median fit deviation {median}": "拟合偏差中位数 {median}",
    "{elapsed} so far": "已用 {elapsed}",
    "{elapsed} so far, {left}": "已用 {elapsed}，{left}",
    "about {minutes} min left": "约剩 {minutes} 分钟",
    "about {hours} h {minutes} min left": "约剩 {hours} 小时 {minutes} 分钟",
    "less than a minute left": "剩不到 1 分钟",
    "Drawing the main screen: window {drawn} of {total}": "正在绘制主屏：第 {drawn}/{total} 个时间窗",
    "Joint fit, iteration {iteration} of {total}": "联合精修 第 {iteration}/{total} 次",
    "Reading the frames from this disk is the slow part (a USB or network drive, or an on-access virus scan of every "
    "file). Copying the scan folder to a local SSD first makes the import faster.":
        "慢在从这个磁盘读帧（USB 或网络驱动器，或者杀毒软件在访问时扫描每个文件）。"
        "先把扫描文件夹复制到本机 SSD，读入会快得多。",
}

__all__ = ["SPUTTER_LIVE_ZH"]
