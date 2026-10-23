"""Chinese texts of the Sputter workspace (page, Predict and PyTorch's installation, results, sketch).

Keys are the exact English of ``features/sputter`` (its templates keep their ``{fields}``). The reliability
report's sentences are in ``zh_sputter_report.py``.
"""

SPUTTER_ZH = {
    # -- the shell -------------------------------------------------------------------------------------------
    "Sputter": "溅射",
    "In-situ sputter series: particle size and spacing over time, and how reliable they are":
        "原位溅射序列：颗粒尺寸和间距随时间的变化，以及它们有多可靠",
    "Sputter prediction": "溅射预测",
    # -- the page: fixed texts -------------------------------------------------------------------------------
    "<b>Sputter series</b><br>Particle size R and spacing D over time, the metal thickness where it can be stated "
    "(island-volume equivalent; nominal, from a deposition rate), and how far each value can be trusted.":
        "<b>溅射序列</b><br>颗粒尺寸 R 和间距 D 随时间的变化，能给出时的金属厚度（岛体积等效；由沉积速率得到的名义厚度），"
        "以及每个值有多可信。",
    "Folder with the CBF frames of one scan": "一个扫描的 CBF 帧所在的文件夹",
    "Choose the folder of a scan: its first frame tells the detector and the time.":
        "选择一个扫描的文件夹：它的第一帧给出探测器和时间。",
    "Sample and geometry": "样品与几何",
    "Metal": "金属",
    "Cu — approximated with the Ag model, reference only": "Cu——用 Ag 模型近似，仅供参考",
    "Cu is read with the Ag model (the networks were trained for Ag and Au): its sizes are for reference only.":
        "Cu 用 Ag 模型读取（网络是为 Ag 和 Au 训练的）：它的尺寸仅供参考。",
    "X-rays": "X 射线",
    "Incidence αi (°)": "入射角 αi（°）",
    "Beam centre (px)": "光束中心（px）",
    "col": "列",
    "row": "行",
    "required": "必填",
    "Beam centre in pixels as Analyze shows it (pixel corners on whole numbers, row 0 at the top). Empty: estimated "
    "from the frames.": "光束中心，像素坐标与分析（Analyze）中显示的相同（像素角在整数上，第 0 行在上）。留空：由帧估计。",
    "Save Geometry as Preset…": "把几何存为预设…",
    "Keep this geometry for every scan of this detector and flat field in a date range (a beamtime): such scans are "
    "filled in from it.": "为这台探测器和这个平场在某个日期范围（一次束线时间）内的每个扫描保存这套几何：这些扫描会按它填写。",
    "Read the frames and predict the structure of every time window. The first time, PyTorch is installed into "
    "GIMaP's Python before (about 1.2 GB).": "读取各帧并预测每个时间窗的结构。第一次会先把 PyTorch 装进 GIMaP 的 Python（约 1.2 GB）。",
    "Messages of the model and of the installation": "模型和安装过程的消息",
    "No result yet: choose a scan folder and press Predict.": "还没有结果：选择扫描文件夹，然后按“预测”。",
    "More options": "更多选项",
    "Polymer": "聚合物",
    "Sample roll (°)": "样品滚转角（°）",
    "Mask": "掩膜文件",
    "optional detector mask (.npy)": "可选的探测器掩膜（.npy）",
    "Time window (s)": "时间窗（s）",
    "Deposition rate (nm/s)": "沉积速率（nm/s）",
    "Start of deposition t0 (s)": "沉积开始 t0（s）",
    "Deposition end (s)": "沉积结束（s）",
    "optional": "可选填",
    "end of the scan": "扫描结束时",
    "Deposition rate (for example from a quartz balance): the nominal thickness rate × (t − t0) is drawn with the "
    "metal thickness, and the literature method (a check of the network) and the Literature tab use it. Changing it "
    "later does not read the frames again.":
        "沉积速率（例如来自石英晶体微天平）：名义厚度 速率 × (t − t0) 画在金属厚度图里，文献方法（对网络的核对）和“文献对照”页"
        "也用它。之后修改它不会重新读帧。",
    "Time (s) of the scan at which the deposition started (the nominal thickness is 0 there).":
        "扫描中沉积开始的时间（s）（名义厚度在此为 0）。",
    "Time (s) of the scan at which the deposition ended (the shutter was closed): the nominal thickness stays at "
    "rate × (t_end − t0) after it, also in the literature method. Empty: the deposition lasts to the end of the scan.":
        "扫描中沉积结束（挡板关上）的时间（s）：此后名义厚度保持在 速率 × (t_end − t0)，文献方法中也是如此。留空：沉积持续到"
        "扫描结束。",
    "Save results in": "结果保存在",
    "Each scan and geometry gets its own folder in here; the same request reuses it.":
        "每个扫描和几何在这里有自己的文件夹；相同的请求会重用它。",
    "Open Results Folder…": "打开结果文件夹…",
    "Add Refined Windows…": "加入精修时间窗…",
    "Merge physical-refinement windows (one JSON per window, e.g. from the cluster) into the current result.":
        "把物理精修的时间窗（每窗一个 JSON，例如来自集群）合并进当前结果。",
    "Explanation": "解释",
    "Which explanation of the windows is shown.\nPhysical path (network; the default): one path of stages through "
    "time, network sizes in every island window, no switching between sources (grade P).\nLiterature method (a check "
    "of the network; needs a deposition rate): R from the nominal thickness and D = 2π/q* by mass balance, as in the "
    "P03 in-situ GISAXS studies (grade Lit).\nPer window: highest network weight, or the "
    "converged refinement (grades A/B/C).\nLearned selector: trained on simulated growth series (grade L; its tested "
    "accuracy is on the Reliability tab).\nTime-consistent path (exploratory): registered physical rules; not more "
    "accurate in simulation (grades S/T).":
        "显示时间窗的哪一种解释。\n物理路径（网络；默认）：随时间只走一条阶段路径，每个岛时间窗都用网络尺寸，不在来源之间切换"
        "（等级 P）。\n文献方法（对网络的核对；需要沉积速率）：与 P03 原位 GISAXS 研究相同，由名义厚度和 D = 2π/q* 按质量守恒"
        "算出 R（等级 Lit）。\n"
        "逐窗：网络权重最高的解，或收敛的精修（等级 A/B/C）。\n学习选择器：在模拟生长序列上训练（等级 L；测试准确率见“可靠性”页）。\n"
        "时间一致路径（探索性）：按登记的物理规则；在模拟中并不更准（等级 S/T）。",
    "Physical path (network)": "物理路径（网络）",
    "Literature method (mass balance)": "文献方法（质量守恒）",
    "Per window (network / refined)": "逐窗（网络 / 精修）",
    "Learned selector": "学习选择器",
    "Time-consistent path (exploratory)": "时间一致路径（探索性）",
    "Logarithmic size axis": "尺寸轴取对数",
    "Size Trend": "尺寸趋势",
    "Structure Sketch": "结构示意",
    "Windows": "时间窗",
    "Reliability": "可靠性",
    "<p>Choose a scan folder and press Predict, or open a results folder (More options). Each R and D value gets a "
    "grade: <b>P</b> network estimate on the physical path (the default), <b>Lit</b> literature method (R from the "
    "nominal thickness and 2π/q*; a check of the network that needs a deposition rate), <b>A</b> refined (unique), "
    "<b>B</b> refined (an equally good alternative exists), <b>C</b> network estimate only.</p>":
        "<p>选择扫描文件夹并按“预测”，或打开一个结果文件夹（更多选项）。每个 R 和 D 值都有等级：<b>P</b> 物理路径上的网络估计"
        "（默认），<b>Lit</b> 文献方法（由名义厚度和 2π/q* 算出 R；对网络的核对，需要沉积速率），<b>A</b> 精修（唯一），<b>B</b> "
        "精修（存在同样好的另一解），<b>C</b> 仅网络估计。</p>",
    # -- the folder, the sources of the values ------------------------------------------------------------------
    "Choose the Scan Folder": "选择扫描文件夹",
    "Reading the header of the first frame…": "正在读取第一帧的文件头…",
    "exposure {exposure} s": "曝光 {exposure} s",
    " every {period} s": "，每 {period} s 一帧",
    "header: {energy} keV": "文件头：{energy} keV",
    "NeXus file: {energy} keV": "NeXus 文件：{energy} keV",
    "flat field: E = {energy} keV": "平场：E = {energy} keV",
    "No CBF frames in this folder.": "这个文件夹里没有 CBF 帧。",
    "from the file header": "来自文件头",
    "from the scan's NeXus file": "来自该扫描的 NeXus 文件",
    "as last used for this folder": "沿用这个文件夹上次的值",
    "from preset {name}": "来自预设 {name}",
    "last used — please check": "上次使用的值——请核对",
    "not in the files — please enter": "文件中没有——请填写",
    "not in the files: 0 assumed — please check": "文件中没有：假定为 0——请核对",
    "empty: estimated from the frames": "留空：由帧估计",
    "entered": "已填写",
    "the file header suggests {energy} keV": "文件头提示 {energy} keV",
    "the scan's NeXus file suggests {energy} keV": "该扫描的 NeXus 文件提示 {energy} keV",
    "the flat-field file name says E = {energy} keV: the detector's calibration energy, not necessarily the beam's":
        "平场文件名写着 E = {energy} keV：这是探测器的标定能量，不一定是光束的能量",
    "Predict starts once the header of the first frame is read…": "读完第一帧的文件头后开始预测…",
    "The result shown is of another scan: the deposition rate, t0 and end are kept for the next Predict of the "
    "chosen folder.": "显示的结果属于另一个扫描：沉积速率、t0 和沉积结束时间留给所选文件夹的下一次预测。",
    "The end of the deposition must be after its start t0 (or empty).": "沉积结束时间必须晚于开始时间 t0（或留空）。",
    "Choose a scan folder whose frames name the detector's serial number first: a preset is found again by the "
    "serial number, the flat field and the date.": "请先选择一个帧里写有探测器序列号的扫描文件夹：预设按序列号、平场和日期找回。",
    "Geometry preset {name} saved: scans of this detector from {start} to {end} are filled in from it.":
        "几何预设 {name} 已保存：这台探测器从 {start} 到 {end} 的扫描会按它填写。",
    "Save Geometry as Preset": "把几何存为预设",
    "The geometry on the page (X-rays, distance, incidence, beam centre, sample roll) is used for every scan of this "
    "detector and flat field taken in these days.":
        "页面上的几何（X 射线、距离、入射角、光束中心、样品滚转角）用于这台探测器和这个平场在这些日期内的每个扫描。",
    "for example the beamtime": "例如束线时间",
    "Detector serial number": "探测器序列号",
    "Flat field": "平场",
    "First day": "第一天",
    "Last day": "最后一天",
    "Choose a Detector Mask": "选择探测器掩膜",
    "Save Results In": "结果保存位置",
    # -- the request -------------------------------------------------------------------------------------------
    "Choose the scan folder that holds the CBF frames.": "请选择存放 CBF 帧的扫描文件夹。",
    "Give the photon energy or the wavelength.": "请给出光子能量或波长。",
    "Give the sample-detector distance and the incidence angle (both positive).": "请给出样品-探测器距离和入射角（都为正）。",
    "Give both beam-centre coordinates, or neither (automatic estimate).": "请给出光束中心的两个坐标，或都不给（自动估计）。",
    "Metal must be Ag, Au or Cu.": "金属必须是 Ag、Au 或 Cu。",
    "The time window must be positive.": "时间窗必须为正。",
    "The deposition rate must be positive (or empty).": "沉积速率必须为正（或留空）。",
    # -- the environment, PyTorch's installation ------------------------------------------------------------------
    "Model: bundled with GIMaP · Python: {python} · PyTorch {version}: ready":
        "模型：随 GIMaP 提供 · Python：{python} · PyTorch {version}：已就绪",
    "Model: bundled with GIMaP · Python: {python} · PyTorch is installed on the first Predict ({size}, needs the "
    "internet)": "模型：随 GIMaP 提供 · Python：{python} · 第一次预测时安装 PyTorch（{size}，需要联网）",
    "Model: bundled with GIMaP · Python: {python} · PyTorch is checked when you press Predict":
        "模型：随 GIMaP 提供 · Python：{python} · 按“预测”时检查 PyTorch",
    "about 1.2 GB": "约 1.2 GB",
    "The Sputter model is not in this GIMaP folder: {folder} is missing or incomplete. Update GIMaP "
    "(Update_GIMaP.bat or a new download); the model is part of it.":
        "这个 GIMaP 文件夹里没有溅射模型：{folder} 不存在或不完整。请更新 GIMaP（Update_GIMaP.bat 或重新下载）；模型是它的一部分。",
    "Checking PyTorch…": "正在检查 PyTorch…",
    "Step 1 of 2: installing PyTorch ({size}, the first time only)…": "第 1/2 步：正在安装 PyTorch（{size}，只在第一次）…",
    "Step 2 of 2: predicting the structure…": "第 2/2 步：正在预测结构…",
    "Predicting the structure…": "正在预测结构…",
    "Preparing pip…": "正在准备 pip…",
    "Looking for {package}…": "正在查找 {package}…",
    "Downloading {package} ({size})…": "正在下载 {package}（{size}）…",
    "Downloading {package}: {percent}%": "正在下载 {package}：{percent}%",
    "{package} is already downloaded": "{package} 已经下载过",
    "Installing the packages (this cannot be stopped)…": "正在安装软件包（这一步不能停止）…",
    "PyTorch is installed": "PyTorch 已安装",
    "Stopping once pip has written the packages…": "pip 写完软件包后停止…",
    "PyTorch could not be downloaded: there is no connection to the Python package index (pypi.org). Connect to the "
    "internet or set the proxy (HTTPS_PROXY), then press Predict again. To install by hand: {command}":
        "无法下载 PyTorch：连不上 Python 软件包索引（pypi.org）。请联网或设置代理（HTTPS_PROXY），然后再按“预测”。手动安装：{command}",
    "PyTorch could not be installed: there is no permission to write into this Python ({prefix}). Start GIMaP from a "
    "folder you can write to (Start_GIMaP.bat), or install by hand as an administrator: {command}":
        "无法安装 PyTorch：没有写入这个 Python（{prefix}）的权限。请从你能写入的文件夹启动 GIMaP（Start_GIMaP.bat），"
        "或以管理员身份手动安装：{command}",
    "PyTorch could not be installed: the disk is full. It needs about 1.5 GB of free space ({prefix}); free some "
    "space, then press Predict again.": "无法安装 PyTorch：磁盘已满。需要约 1.5 GB 可用空间（{prefix}）；请腾出空间，然后再按“预测”。",
    "pip found no PyTorch for this Python ({version}). Start GIMaP with Start_GIMaP.bat, which sets up a Python that "
    "fits, or install PyTorch by hand: {command}":
        "pip 找不到适合这个 Python（{version}）的 PyTorch。请用 Start_GIMaP.bat 启动 GIMaP（它会配置合适的 Python），"
        "或手动安装 PyTorch：{command}",
    "This Python has no pip, and pip could not be added (ensurepip). Install pip for {python}, then press Predict "
    "again, or install by hand: {command}": "这个 Python 没有 pip，也无法补上（ensurepip）。请为 {python} 安装 pip 后再按“预测”，"
                                            "或手动安装：{command}",
    "PyTorch is installed but does not load ({detail}). On Windows this usually means that the Microsoft Visual C++ "
    "Redistributable is missing (install it from Microsoft), or that the installation is damaged: reinstall it with "
    "{command}": "PyTorch 已安装但无法加载（{detail}）。在 Windows 上这通常是缺少 Microsoft Visual C++ 可再发行组件"
                 "（从 Microsoft 安装），或安装已损坏：用 {command} 重新安装",
    "This GIMaP cannot install packages into itself. Install PyTorch by hand: {command}":
        "这个 GIMaP 不能往自身安装软件包。请手动安装 PyTorch：{command}",
    "Installing PyTorch automatically is switched off (GIMAP_SPUTTER_NO_INSTALL=1). Install it by hand: {command}":
        "自动安装 PyTorch 已关闭（GIMAP_SPUTTER_NO_INSTALL=1）。请手动安装：{command}",
    "Stopped before PyTorch was installed. Press Predict to start again.": "在 PyTorch 装好之前已停止。按“预测”重新开始。",
    "pip stopped with exit code {code}; its last lines are under Details. To install by hand: {command}":
        "pip 以退出码 {code} 结束；最后几行在“详情”里。手动安装：{command}",
    # -- the run -----------------------------------------------------------------------------------------------
    "Reading the frames": "正在读帧",
    "Summing the counts into time windows": "正在把计数合并成时间窗",
    "Direct features of each window": "正在计算每个时间窗的直接特征",
    "Running the networks": "正在运行网络",
    "Done": "完成",
    "{stage}: {done} of {total} frames": "{stage}：{done} / {total} 帧",
    "{count} damaged frames left out": "{count} 个损坏的帧已略去",
    "Stopped. Finished steps are kept in the results folder and are reused on the next run.":
        "已停止。已完成的步骤保留在结果文件夹中，下次运行时会重用。",
    "The Sputter model stopped with an error (exit code {code}); its last lines are under Details.":
        "溅射模型出错结束（退出码 {code}）；最后几行在“详情”里。",
    "Done: {windows} time windows.": "完成：{windows} 个时间窗。",
    "The structure of {name} is ready.": "{name} 的结构已得出。",
    "The prediction failed.": "预测失败。",
    "Drawing the nominal thickness again (no frame is read)…": "正在重画名义厚度（不读帧）…",
    "Merging the refined windows…": "正在合并精修时间窗…",
    "Folder of Refined Window Files": "精修时间窗文件所在的文件夹",
    "Open Sputter Results Folder": "打开溅射结果文件夹",
    "{folder} has no {table} (press Predict, or choose a results folder).":
        "{folder} 里没有 {table}（请按“预测”，或选择一个结果文件夹）。",
    # -- the result line, the table ------------------------------------------------------------------------------
    "{windows} windows · R in {R} · D in {D}": "{windows} 个时间窗 · 有 R 的 {R} 个 · 有 D 的 {D} 个",
    "refined A+B: R {R}, D {D}": "精修 A+B：R {R}，D {D}",
    "film from {onset} s": "从 {onset} s 起成膜",
    " ({low}–{high} s)": "（{low}–{high} s）",
    "no film window": "没有膜时间窗",
    "intensity modulation period ≈ {period} s (sample rotation?)": "强度调制周期 ≈ {period} s（可能是样品旋转）",
    "Cu: approximated with the Ag model, for reference only.": "Cu：用 Ag 模型近似，仅供参考。",
    "Stage (rule)": "阶段（规则）",
    "Stage (path)": "阶段（路径）",
    "Model (path)": "模型（路径）",
    "R grade": "R 等级",
    "R typ. error": "R 典型误差",
    "R trend (nm)": "R 趋势（nm）",
    "D grade": "D 等级",
    "D typ. error": "D 典型误差",
    "D trend (nm)": "D 趋势（nm）",
    "D check": "D 核对",
    "t_eq range (nm)": "t_eq 范围（nm）",
    "Nominal thickness (nm)": "名义厚度（nm）",
    "Film thickness": "膜厚",
    "TS length L (nm)": "TS 特征长度 L（nm）",
    "Refined R (nm)": "精修 R（nm）",
    "Refined D (nm)": "精修 D（nm）",
    "Refinement": "精修状态",
    "Alternatives": "其他解数",
    "Changed by the path": "被路径改动",
    "Path": "路径",
    "Data cost (nats)": "数据代价（nats）",
    "Selector P": "选择器 P",
    "Component": "分量",
    "R note": "R 说明",
    "D note": "D 说明",
    # -- the trend ---------------------------------------------------------------------------------------------
    "time (s)": "时间（s）",
    "D, centre-to-centre spacing (peak windows)": "D，中心间距（有峰时间窗）",
    "TS length L (film windows)": "TS 特征长度 L（膜时间窗）",
    "Metal thickness (nm)": "金属厚度（nm）",
    "TS length L (network)": "TS 特征长度 L（网络）",
    "trend (smoothed)": "趋势（平滑）",
    "t_eq: island-volume equivalent (conditional)": "t_eq：岛体积等效（有条件）",
    "nominal: rate × (t − t0)": "名义：速率 × (t − t0)",
    "deposition ended {time} s": "沉积结束 {time} s",
    "deposition ended {time} s, after the last window": "沉积结束 {time} s，在最后一个时间窗之后",
    "deposition ended {time} s, before the first window": "沉积结束 {time} s，在第一个时间窗之前",
    "film: thickness not determined": "膜：厚度未确定",
    "No thickness value in this result.": "这个结果里没有厚度值。",
    "film onset {time} s": "成膜起点 {time} s",
    "D vs 2π/q* outside simulated 90% range": "D 与 2π/q* 之比超出模拟的 90% 范围",
    "2π/q* (direct)": "2π/q*（直接）",
    "Plotting is unavailable: {error}": "无法绘图：{error}",
    "refined, unique": "精修，唯一",
    "refined, alternative exists": "精修，存在另一解",
    "network only": "仅网络",
    "time-consistent, data-supported": "时间一致，数据支持",
    "time-consistent, constraint-driven": "时间一致，约束推出",
    "learned selector": "学习选择器选出",
    "network estimate, physical path": "网络估计，物理路径",
    # -- the sketch ----------------------------------------------------------------------------------------------
    "Time": "时间",
    "Median of ±5 windows": "±5 个时间窗的中位数",
    "Median over neighbouring windows of the same model (about ±2.5 s): hides single-window jumps, keeps the trend. "
    "Untick to see each window's own values.": "同一模型相邻时间窗的中位数（约 ±2.5 s）：去掉单窗跳变，保留趋势。取消勾选可看每个时间窗自己的值。",
    "Show model burial depth (unreliable)": "显示模型埋入深度（不可靠）",
    "Save Storyboard…": "保存分镜图…",
    "Open a result to see the structure.": "打开一个结果以查看结构。",
    "Sketch of the fitted model, not an image: flat-based hemispheres (h = R) with the estimated mean size and size "
    "spread; spacing from the model D only where it agrees with the peak, otherwise 2π/q*. Films are drawn without "
    "particles (a film with particles is not modelled) and their thickness is not determined. Burial depth is not "
    "reliable.": "拟合模型的示意图，不是图像：平底半球（h = R），用估计的平均尺寸和尺寸分布；只有模型 D 与峰一致时才用它作间距，"
                 "否则用 2π/q*。膜画成不带颗粒的连续膜（膜上有颗粒的情形没有建模），膜厚未确定。埋入深度不可靠。",
    "Drawing is unavailable: {error}": "无法绘制：{error}",
    "No window with a size estimate.": "没有带尺寸估计的时间窗。",
    "Save Sketch": "保存示意图",
    "Save Storyboard": "保存分镜图",
    "<b>t = {time}</b> · {stage} · <b>R = {R} nm</b> {error} (grade {grade}) · spacing <b>{spacing}</b> · size "
    "spread σ = {sigma} · spacing disorder {disorder}{coverage}":
        "<b>t = {time}</b> · {stage} · <b>R = {R} nm</b> {error}（等级 {grade}）· 间距 <b>{spacing}</b> · 尺寸分布 σ = {sigma} · "
        "间距无序度 {disorder}{coverage}",
    "<b>t = {time}</b> · {stage} · continuous film · thickness not determined":
        "<b>t = {time}</b> · {stage} · 连续膜 · 厚度未确定",
    "<b>t = {time}</b> · {stage} · continuous film: characteristic length <b>L = {L} nm</b> · thickness not "
    "determined": "<b>t = {time}</b> · {stage} · 连续膜：特征长度 <b>L = {L} nm</b> · 厚度未确定",
    " · coverage ≈ {coverage}% <i>(model-derived)</i>": " · 覆盖率 ≈ {coverage}% <i>（由模型推出）</i>",
    "no size estimate in this window": "这个时间窗没有尺寸估计",
    "(assumed)": "（假定）",
    "pre-nucleation": "成核前",
    "islands": "岛",
    "coalescence": "聚并",
    "film": "膜",
    "pre-nucleation (candidate)": "成核前（候选）",
    "separate islands (candidate)": "分立岛（候选）",
    "coalescence (candidate)": "聚并（候选）",
    "coalescence or film (candidate)": "聚并或成膜（候选）",
    "continuous film (candidate)": "连续膜（候选）",
    "not determined": "未确定",
    "consistent": "一致",
    "outside the simulated 90% range": "超出模拟的 90% 范围",
    "side view (cut)": "侧视（截面）",
    "polymer (thickness not to scale)": "聚合物（厚度不按比例）",
    "top view": "俯视",
    "burial {depth} nm\n(unreliable)": "埋入 {depth} nm\n（不可靠）",
    "thickness not determined": "厚度未确定",
    "no size estimate": "没有尺寸估计",
    "spacing not measured\n(illustration)": "间距未测得\n（示意）",
    "continuous film": "连续膜",
    "t = {time} s · {stage}\nR = {R} nm ({grade}) · spacing {spacing} nm":
        "t = {time} s · {stage}\nR = {R} nm（{grade}）· 间距 {spacing} nm",
    "t = {time} s · film\nL = {L} nm": "t = {time} s · 膜\nL = {L} nm",
    "t = {time} s · film": "t = {time} s · 膜",
    "(R and spacing inconsistent)": "（R 与间距不一致）",
    "{name}: sketch of the fitted structure (hemispheres h = R; spacing from consistent D or 2π/q*; burial {burial})":
        "{name}：拟合结构示意（半球 h = R；间距取一致的 D 或 2π/q*；埋入深度{burial}）",
    "shown, unreliable": "已显示，不可靠",
    "not shown": "未显示",
    "Films are drawn without particles (not modelled) and their thickness is not determined. Spacing '?' = no "
    "resolved peak (illustration). Times on a logarithmic grid": "膜画成不带颗粒（未建模），膜厚未确定。间距“?”= 没有分辨出的峰（示意）。时间取对数网格",
    "values are medians of ±5 neighbouring windows.": "数值为相邻 ±5 个时间窗的中位数。",
    "single-window values.": "数值为单个时间窗的值。",
    "late window: continuous-film (TS) model; L is a characteristic length, not an island spacing":
        "后期时间窗：连续膜（TS）模型；L 是特征长度，不是岛间距",
    "film window of the physical path: a continuous film; its thickness is not determined":
        "物理路径上的膜时间窗：连续膜；厚度未确定",
    "model D {D} nm disagrees with the peak; spacing drawn from 2π/q*": "模型 D {D} nm 与峰不一致；间距按 2π/q* 画",
    "no resolved peak: spacing not measured, islands drawn 4R apart for illustration only":
        "没有分辨出的峰：间距未测得，岛按 4R 的间距画，仅作示意",
    "islands would overlap at this R and spacing (coverage {coverage}% > {limit}%): R or the spacing is not reliable "
    "here": "按这个 R 和间距岛会重叠（覆盖率 {coverage}% > {limit}%）：这里的 R 或间距不可靠",
    "median of {count} neighbouring windows of the same model": "同一模型 {count} 个相邻时间窗的中位数",
    "model D (grade {grade})": "模型 D（等级 {grade}）",
    "direct 2π/q*": "直接 2π/q*",
    "not measured": "未测得",
    "TS length L": "TS 特征长度 L",
}

__all__ = ["SPUTTER_ZH"]
