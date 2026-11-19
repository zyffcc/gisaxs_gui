"""Chinese texts of After Deposition in the Sputter workspace: the sample list, Analyze over it and the report
(``static_list.py``, ``static_queue.py``, ``static_export.py``, ``application/static_samples.py`` and
``application/static_report.py``). Merged into the main table by ``zh.py``."""

SPUTTER_SAMPLES_ZH = {
    # -- the bar and the list (views/sputter_main_view.py, views/sputter_samples_view.py) ---------------------------
    "Add Samples…": "添加样品…",
    "A folder of samples (each folder in it with CBF frames is one sample), one sample's folder or one CBF file. The "
    "nominal thickness, HiPIMS and the bare substrate are read from the folder names.":
        "一个装着多个样品的文件夹（其中每个含 CBF 帧的文件夹算一个样品）、单个样品的文件夹或单个 CBF 文件。"
        "名义厚度、是否 HiPIMS 和裸基底从文件夹名里读出。",
    "Show or hide the list of samples beside the main screen.": "显示或隐藏主屏旁边的样品列表。",
    "Sample List": "样品列表",
    "Select None": "全不选",
    "Remove the Samples Not Selected": "移除没勾选的样品",
    "Add Samples… takes a folder of samples — each folder in it with CBF frames is one sample, however deep they lie "
    "— or one sample's folder. The nominal thickness is read from the name (10Au, H10Au for HiPIMS, 6.5nm); a folder "
    "named like the part before it (850 for 850_10Au) is the bare substrate.":
        "“添加样品…”可以选一个装着多个样品的文件夹——其中每个含 CBF 帧的文件夹算一个样品，帧放得多深都行——"
        "也可以选单个样品的文件夹。名义厚度从名字里读出（10Au，HiPIMS 写作 H10Au，6.5nm）；"
        "名字等于厚度前面那部分的文件夹（850_10Au 对应 850）就是裸基底。",
    "Samples measured after their deposition: each one's nominal thickness takes the place of the rate, and every "
    "series is shown against it.":
        "沉积之后测量的样品：用各自的名义厚度代替速率，每个系列按名义厚度显示。",
    "CBF File…": "CBF 文件…",
    "Choose a Folder of Samples or One Sample's Folder": "选择装着多个样品的文件夹或单个样品的文件夹",
    "Choose a CBF File": "选择 CBF 文件",
    "{added} samples added in {series} series.": "已添加 {added} 个样品，分属 {series} 个系列。",
    "{added} samples added in {series} series; {left} left out (see the list).":
        "已添加 {added} 个样品，分属 {series} 个系列；{left} 个暂不分析（见列表）。",
    "No CBF frames were found in {folder}.": "在 {folder} 里没有找到 CBF 帧。",
    "These samples are in the list already.": "这些样品已经在列表里了。",
    "No samples yet: Add Samples… takes a folder of samples.": "还没有样品：“添加样品…”可以选一个装着多个样品的文件夹。",
    "{samples} samples in {series} series · {analysed} analysed": "{samples} 个样品，{series} 个系列 · 已分析 {analysed} 个",
    "{samples} samples in {series} series · {analysed} analysed · {left} left out":
        "{samples} 个样品，{series} 个系列 · 已分析 {analysed} 个 · {left} 个暂不分析",
    "{series} — bare: {bare} · {analysed}/{count} analysed": "{series} — 裸基底：{bare} · 已分析 {analysed}/{count}",
    "none, on Si": "无，直接在 Si 上",
    "At work…": "分析中…",
    "Fit {fit}": "偏差 {fit}",
    "Left out": "暂不分析",
    "Thickness?": "缺厚度",
    "No frames": "没有帧",
    "Not Au": "不是 Au",
    "{source}\n{frames} CBF frames · bare substrate: {bare}": "{source}\n{frames} 个 CBF 帧 · 裸基底：{bare}",
    "Analyze This Sample": "分析这个样品",
    "Analyze This Series": "分析这个系列",
    "Show Result": "显示结果",
    "Remove This Series": "移除这个系列",
    "Bare Substrate…": "裸基底…",
    "No Bare Substrate (on Si)": "不用裸基底（直接在 Si 上）",
    "Choose the Folder of the Bare Substrate": "选择裸基底的文件夹",
    "Not analysed yet: tick it and press Analyze.": "还没分析：勾选后按“分析”。",
    # -- why a sample is left out (application/static_samples.py) ----------------------------------------------------
    "No CBF frames in this folder": "这个文件夹里没有 CBF 帧",
    "The name gives no thickness: enter it": "名字里没有厚度：请填写",
    "Only Au films can be analysed after their deposition": "沉积后的分析只适用于 Au 膜",
    # -- Details ▸ Static Frames (views/sputter_static_view.py, static_flow.py) --------------------------------------
    "Samples measured after their deposition (choose After Deposition in the bar and add them there): the folder their "
    "series go to, the mode and the geometry of their frames. Each series is shown on the main screen against the "
    "nominal thickness.":
        "沉积之后测量的样品（在输入栏选“沉积后”，在那里添加样品）：各系列的存放文件夹、模式和这些帧的几何。"
        "每个系列在主屏上按名义厚度显示。",
    "Results folder": "结果文件夹",
    "The folder the series go to: each series of the list (the part of the names before the thickness, and the "
    "deposition method) is analysed into a folder of its own in it, named after the series. A series already there is "
    "added to.":
        "各系列存放的文件夹：列表里的每个系列（名字里厚度前面的部分，加上沉积方式）分析进其中一个以系列命名的子文件夹。"
        "已有的系列会继续往里加样品。",
    "Frames of another piece of the same substrate without metal, measured like the samples: they give the substrate "
    "part of their images. An in-situ scan takes this from its own frames before the deposition; a sample measured "
    "afterwards has none, so it needs another piece. None: the film is taken to lie directly on Si.":
        "另一片同种基底、没有镀金属的帧，测法与样品相同：用作样品图像里的基底部分。原位扫描用的是它自己沉积开始之前的帧；"
        "沉积之后测量的样品没有这样的帧，所以要另一片。不用裸基底：认为膜直接长在 Si 上。",
    "Choose the Folder the Series Go To": "选择各系列存放的文件夹",
    "Au films measured after the deposition, analysed sample by sample into their series.":
        "沉积之后测量的 Au 膜，逐个样品分析，归入各自的系列。",
    "Au films measured after the deposition at an incidence angle near {alpha}°, analysed sample by sample into their "
    "series.": "沉积之后在入射角约 {alpha}° 下测量的 Au 膜，逐个样品分析，归入各自的系列。",
    "Replace the Samples?": "替换这些样品？",
    "{names}: analysed before from other frames or values. Analysing again replaces them in their series.":
        "{names}：以前用别的帧或数值分析过。再分析一次会替换它们在系列里的结果。",
    "The frames of the sample were not found: its folder or CBF file is not there.":
        "没有找到样品的帧：它的文件夹或 CBF 文件不在了。",
    "The frames of the bare substrate were not found: choose its folder again, or none.":
        "没有找到裸基底的帧：请重新选择它的文件夹，或者不用裸基底。",
    # -- Analyze over the list (static_queue.py) ----------------------------------------------------------------------
    "Sample {index} of {count}: {name}": "第 {index}/{count} 个样品：{name}",
    "Done: {done} samples analysed.": "完成：分析了 {done} 个样品。",
    "Done: {done} samples analysed, {failed} failed (the reason of each is in its row's tooltip).":
        "完成：分析了 {done} 个样品，{failed} 个失败（原因见各行的提示）。",
    "No sample was analysed: {failed} failed (the reason of each is in its row's tooltip).":
        "没有分析成任何样品：{failed} 个失败（原因见各行的提示）。",
    "Stopped after {done} samples. The samples analysed stay in their series.":
        "分析了 {done} 个样品后停止。已分析的样品留在各自的系列里。",
    "Add samples first: Add Samples… takes a folder of samples.": "请先添加样品：“添加样品…”可以选一个装着多个样品的文件夹。",
    "Tick the samples to analyse in the list (each needs its nominal thickness).":
        "请在列表里勾选要分析的样品（每个都要有名义厚度）。",
    "{done} samples analysed into {series} series.": "已把 {done} 个样品分析进 {series} 个系列。",
    # -- Export… of the report (static_export.py) ---------------------------------------------------------------------
    "Write the report of every series of the sample list: one HTML file with each series' values, the stage of each "
    "sample and the figures of the fits, and a CSV with every sample.":
        "写出样品列表里所有系列的报告：一个 HTML 文件，含各系列的数值、每个样品所处的阶段和拟合图；另有一个列出全部样品的 CSV。",
    "Choose the Folder for the Report": "选择存放报告的文件夹",
    "Writing the report of {series} series…": "正在写 {series} 个系列的报告…",
    "Report of {series} series written to {folder}": "{series} 个系列的报告已写到 {folder}",
    "No series of the list has analysed samples yet: analyse them first.": "列表里还没有分析过的系列：请先分析。",
    # -- the report (application/static_report.py) --------------------------------------------------------------------
    "separate islands": "分立的岛",
    "grains in contact": "相互接触的晶粒",
    "grains on a thin layer": "薄底层上的晶粒",
    "continuous film": "连续膜",
    "film with a density below 1": "密度低于 1 的膜",
    "Au films after their deposition: structure from GISAXS": "沉积后的 Au 膜：由 GISAXS 得到的结构",
    "{samples} samples in {series} series · {date} · GIMaP {version} · analysis package {package}":
        "{samples} 个样品，{series} 个系列 · {date} · GIMaP {version} · 分析包 {package}",
    "Geometry: {geometry} · mode {mode}": "几何：{geometry} · 模式 {mode}",
    "Each value is the median over the usable frames of a sample. The fit deviation is the deviance per detector cell "
    "with a 5 % allowance for the model: up to 4 is a good fit, 4 to 10 a fair one, above 10 the model does not "
    "describe the frames and the values are only an order of magnitude.":
        "每个数值是一个样品各可用帧的中位数。拟合偏差是每个探测器格的偏差（含 5 % 的模型误差）：不超过 4 为好，"
        "4 到 10 尚可，超过 10 说明模型描述不了这些帧，数值只能当量级看。",
    "The metal amount h_m comes from the shape of the pattern, not from an absolute intensity (the incident flux is not "
    "known): within about 30 % of the nominal thickness is normal.":
        "金属等效厚度 h_m 来自图样的形状，不是绝对强度（入射通量未知）：和名义厚度相差 30 % 以内属于正常。",
    "grains joined into a network": "连成网络的晶粒",
    "Each series is in two parts. Separate particles: the grains stand apart or touch without joining across the "
    "picture — their radius, height, spacing and the gap between them. Connected film: a continuous layer under the "
    "grains, or grains joined into a network — there are no separate particles any more, so the film's thickness, "
    "density and height are given, and the radius and spacing are those of the relief of its surface.":
        "每个系列分两部分。孤立颗粒：晶粒彼此分开，或相互接触但没有连成贯穿整个图的网络——给出它们的半径、高度、间距和相邻颗粒的间隙。"
        "连成的膜：晶粒下有连续底层，或晶粒连成网络——这时已经没有单个颗粒，所以给出膜的厚度、密度和高度，半径和间距是膜表面起伏的尺度。",
    "Stage, as the Real Space picture of the main screen draws the solution: separate islands, grains in contact "
    "(touching, no cluster across the picture) or grains joined into a network (a cluster across it); with a "
    "continuous layer, grains on a thin layer while the layer is thinner than the grains stand above it, else a "
    "continuous film — a film with a density below 1 when its density is below 0.95 of the bulk metal's.":
        "阶段与主屏“实空间”图画出的解一致：分立的岛、相互接触的晶粒（接触，但没有贯穿整个图的团簇）或连成网络的晶粒（有团簇贯穿全图）；"
        "有连续底层时，底层比晶粒高出底层的部分薄为“薄底层上的晶粒”，否则为“连续膜”——底层密度低于块体金属的 0.95 时为“密度低于 1 的膜”。",
    "Separate particles": "孤立颗粒",
    "Connected film": "连成的膜",
    "The film is connected from {nominal} nm on.": "从 {nominal} nm 起连成膜。",
    "The two parts against the nominal thickness": "两部分随名义厚度的变化",
    "Height H (nm)": "高度 H（nm）",
    "Spacing D (nm)": "间距 D（nm）",
    "Gap D − 2R (nm)": "间隙 D − 2R（nm）",
    "Film thickness (nm)": "膜厚（nm）",
    "Film density": "膜密度",
    "Film height (nm)": "膜高（nm）",
    "Relief height (nm)": "起伏高度（nm）",
    "Relief radius (nm)": "起伏半径（nm）",
    "Relief spacing (nm)": "起伏间距（nm）",
    "Spacing D": "间距 D",
    "Film thickness": "膜厚",
    "Film height": "膜高",
    "Relief height": "起伏高度",
    "Relief spacing": "起伏间距",
    "No independent measurement (AFM, SEM, XRR) checked these values.": "这些数值没有经过独立测量（AFM、SEM、XRR）核对。",
    "Series {series}": "系列 {series}",
    "Bare substrate: {bare} · folder: {folder}": "裸基底：{bare} · 文件夹：{folder}",
    "none (the film directly on Si)": "无（膜直接在 Si 上）",
    "Measured and model images, their ratio and two cuts, sample by sample": "逐个样品的实测图、模型图、两者之比和两条切线",
    "h_m (nm)": "h_m（nm）",
}

__all__ = ["SPUTTER_SAMPLES_ZH"]
