"""Chinese texts of the Sputter workspace's main screen, of Analyze and of Static Frames (the client analysis package).

Keys are the exact English of ``features/sputter`` (``presentation/views/sputter_main_view.py``,
``views/sputter_static_view.py``, ``client_flow.py``, ``client_panel.py``, ``client_texts.py``, ``static_flow.py``,
``real_space_view.py``,
``pattern_view.py``, ``evolution_view.py``, ``application/client.py``, ``application/client_static.py``,
``application/client_progress.py``, ``domain/client.py``); templates keep their ``{fields}``. Numbers, units, element
and material names, scan and sample names, file names and the package's own lines are not translated.
"""

SPUTTER_CLIENT_ZH = {
    # -- the picture: patterns or real space (views/sputter_main_view.py, real_space_view.py) --------------------------
    "Patterns": "图样",
    "The measured image beside the model image of the solution shown.": "实测图与所选解的模型图并排。",
    "Real Space": "实空间",
    "The solution shown drawn in real space: a cut through a row of grains and the view from above, in nm.":
        "把所选的解画成实空间形貌：穿过一排晶粒的截面和俯视图，单位 nm。",
    "x (nm)": "x（nm）",
    "y (nm)": "y（nm）",
    "height (nm)": "高度（nm）",
    "continuous layer {thickness} nm": "连续底层 {thickness} nm",
    "Grains placed at random around the distance D; radius, size distribution, height and base layer are "
    "those of the solution: {name}":
        "晶粒位置按间距 D 随机排布；半径、尺寸分布、高度和底层取自所选的解：{name}",
    "Separate islands": "分立的岛",
    "Coalescing: neighbouring grains touch": "聚并中：相邻晶粒相接",
    "Connected network across the view (percolated)": "连成贯穿视场的网络（已逾渗）",
    "Continuous layer of {thickness} nm; the grains on it: {grains}": "连续底层 {thickness} nm；其上的晶粒：{grains}",
    "separate": "彼此分开",
    "touching their neighbours": "与相邻晶粒相接",
    "connected across the view": "连成贯穿视场的网络",
    "cross-section of the TiO₂ layer": "TiO₂ 层的截面",
    "top of a particle (enlarged)": "一个颗粒的顶部（放大）",
    "depth below the particle tops (nm)": "颗粒顶面以下的深度（nm）",
    "pore metal around {depth} nm (width {width} nm)": "孔内金属集中在 {depth} nm 深处（宽 {width} nm）",
    "cap on {cover} of the tops, {thickness} nm": "帽层覆盖 {cover} 的颗粒顶面，厚 {thickness} nm",
    "no cap layer": "没有帽层",
    "Of the {deposited} nm deposited, as dense-metal thickness: islands {islands} nm · cap {cap} nm · "
    "pores {pores} nm. TiO₂ particles of {particle} nm radius; positions drawn at random: {name}":
        "已沉积 {deposited} nm，折合成致密金属厚度：岛 {islands} nm · 帽层 {cap} nm · 孔内 {pores} nm。TiO₂ 颗粒半径 "
        "{particle} nm；位置为随机示意：{name}",
    "This solution has no arrangement to draw: the package gave neither a grain radius and distance nor where "
    "the metal sits in a porous layer.":
        "这个解没有可画的形貌：分析包既没有给出晶粒的半径和间距，也没有给出金属在多孔层里的位置。",
    # -- the input bar -----------------------------------------------------------------------------------------------
    "Rate (nm/s)": "速率（nm/s）",
    "Deposition rate in nm/s: needed, the solutions depend on the metal deposited at each time.":
        "沉积速率（nm/s）：必填，各个解依赖每个时刻已沉积的金属量。",
    "QCM reading": "QCM 读数",
    "Ticked: the number is the reading of the quartz balance (QCM), which the analysis converts to the rate at "
    "the sample. Not ticked: the rate at the sample itself.":
        "勾选：这个数是石英晶体微天平（QCM）的读数，分析会把它换算成样品处的速率。不勾选：这个数就是样品处的速率。",
    "Read the frames, find the solutions of every time window and draw each as an image beside the measured one.":
        "读取各帧，求出每个时间窗的解，并把每个解画成图像，与实测图并排。",
    "Edit": "编辑",
    "Show the geometry in Details to enter or change it.": "在“详情”里显示几何，以便填写或修改。",
    "Open Result…": "打开结果…",
    "Show the result in a folder analysed before.": "显示以前分析过的文件夹里的结果。",
    "Main Screen": "主屏",
    "The geometry, the options of the analysis, the report and the earlier method with its tabs.":
        "几何、分析的选项、报告，以及早先的方法和它的各个页面。",
    # "X-ray energy" (the first of the names) is in another table
    "Geometry to enter: {names}": "还需填写的几何：{names}",
    "sample-detector distance": "样品到探测器的距离",
    "incidence angle": "入射角",
    "both beam-centre coordinates": "光束中心的两个坐标",
    "beam centre ({col}, {row}) px": "光束中心（{col}, {row}）px",
    "beam centre from the frames": "光束中心由帧估计",
    "roll {roll}°": "滚转角 {roll}°",
    # -- the main screen ---------------------------------------------------------------------------------------------
    "Previous time window": "上一个时间窗",
    "Next time window": "下一个时间窗",
    "Time window shown": "显示的时间窗",
    "t = {time} s · {deposit} nm deposited": "t = {time} s · 已沉积 {deposit} nm",
    "Time window {index} of {count}.": "第 {index} 个时间窗，共 {count} 个。",
    "Time window {index} of {count}: {onset} s after the start of the deposition.":
        "第 {index} 个时间窗，共 {count} 个：沉积开始后 {onset} s。",
    "Solution shown": "显示的解",
    "The number beside each solution is its fit deviation: lower is better.": "每个解旁边的数字是它的拟合偏差：越小越好。",
    "Back to Default": "恢复默认解",
    "Show again the solution the smooth evolution over the scan selects for this window.":
        "重新显示全扫描的平滑演化为这个时间窗选出的解。",
    "Candidate {rank}": "候选 {rank}",
    "Refined": "精修解",
    "{name}: fit deviation {deviation}": "{name}：拟合偏差 {deviation}",
    "{name}: fit deviation {deviation}, weight of the network {weight}":
        "{name}：拟合偏差 {deviation}，网络给的权重 {weight}",
    "Radius R": "半径 R",
    "Height H": "高度 H",
    "Gap": "间隙",
    "Size distribution width": "尺寸分布宽度",
    "Coverage": "金属覆盖率",
    "Radius of the grains.": "晶粒的半径。",
    "Height of the grains.": "晶粒的高度。",
    "Centre-to-centre distance of neighbouring grains.": "相邻晶粒的中心间距。",
    "Gap between neighbouring grains: D − 2R.": "相邻晶粒之间的间隙：D − 2R。",
    "Width σ of the log-normal distribution of the grain radius.": "晶粒半径对数正态分布的宽度 σ。",
    "Share of the surface that the metal covers (0 to 1).": "金属覆盖的表面份额（0 到 1）。",
    "Fit deviation (lower is better)": "拟合偏差（越小越好）",
    "How far the model image is from the measured image, per cell and with an allowance for the model: it compares "
    "the solutions of one measured image.":
        "模型图与实测图相差多少（按图像的每一格计，并为模型留有余量）：用来比较同一幅实测图的各个解。",
    "This time window has no solution: {reason}": "这个时间窗没有解：{reason}",
    "Measured image": "实测图",
    "Model image: {name}": "模型图：{name}",
    "Model image": "模型图",
    "Exit angle αf (°)": "出射角 αf（°）",
    "Count rate": "计数率",
    "Time in the scan (s)": "扫描中的时间（s）",
    "Size (nm)": "尺寸（nm）",
    # -- the empty states --------------------------------------------------------------------------------------------
    "No result yet": "尚无结果",  # its sentence is below, with the static frames it names
    "The analysis package is needed": "需要分析包",
    # -- Details: Analyze options ------------------------------------------------------------------------------------
    "Analyze options": "分析选项",
    "Quick": "快速",
    "Full": "完整",
    "Quick for Si and TiO₂, Refine for polymer substrates.": "Si 和 TiO₂ 用“快速”，聚合物基底用“精修”。",
    "The network's candidates, each drawn by the physical model: about a second per time window.":
        "网络给出的候选解，每个都用物理模型画成图：每个时间窗约一秒。",
    "Also refines the solution of every window (minutes per window on a polymer).":
        "再对每个时间窗的解做精修（聚合物上每个时间窗要几分钟）。",
    "A longer refinement of every window and of the whole scan together: hours.":
        "对每个时间窗做更长的精修，并把整条扫描放在一起精修：要几个小时。",
    "Start of deposition (s)": "沉积开始（s）",
    "Deposition duration (s)": "沉积持续时间（s）",
    "Polymer film thickness (nm)": "聚合物膜厚（nm）",
    "from the data": "由数据确定",
    "Time of the scan (s) at which the deposition started. Empty: found in the data.":
        "扫描中沉积开始的时间（s）。留空：由数据确定。",
    "How long the deposition lasted after its start (s): the metal amount stays constant afterwards. Empty: to the "
    "end of the scan.": "沉积开始后持续了多久（s）：之后金属量不再增加。留空：持续到扫描结束。",
    "Thickness of the polymer film on the wafer. Empty: the analysis uses its own value for this polymer.":
        "晶圆上聚合物膜的厚度。留空：分析对这种聚合物用它自己的值。",
    "Analysis package": "分析包",
    "The folder of the analysis package (it holds code and models).": "分析包的文件夹（里面有 code 和 models）。",
    "{folder} (from GIMAP_SPUTTER_CLIENT)": "{folder}（来自 GIMAP_SPUTTER_CLIENT）",
    "{folder} (in the program folder)": "{folder}（在程序文件夹里）",
    "not found": "未找到",
    "Choose the Folder of the Analysis Package": "选择分析包的文件夹",
    "Open Report": "打开报告",
    "The report of the result shown: every quantity with its range and where it comes from.":
        "所显示结果的报告：每个量的数值、范围和来源。",
    "Open the folder of the result shown in the file manager.": "在文件管理器里打开所显示结果的文件夹。",
    "Open the Folder of a Result": "打开结果所在的文件夹",
    "Messages of the analysis": "分析过程的消息",
    "This result has no report file.": "这个结果没有报告文件。",
    "This folder holds a result of the earlier method: it is shown in Details.":
        "这个文件夹里是早先方法的结果：已在“详情”里显示。",
    "Earlier method: the networks bundled with GIMaP": "早先的方法：GIMaP 自带的网络",
    "Predict reads the scan with the networks bundled with GIMaP (Ag and Au on polymers) and fills the tabs at the "
    "right. Analyze, above, is the newer analysis.":
        "“预测”用 GIMaP 自带的网络（聚合物上的 Ag 和 Au）读取扫描，并填写右边的各个页面。上方的“分析”是较新的分析。",
    # -- Analyze: the job status -------------------------------------------------------------------------------------
    "Analyze starts once the header of the first frame is read…": "读完第一帧的文件头后开始分析…",
    "Starting the analysis…": "正在启动分析…",
    "Step {index} of {total}: {stage}…": "第 {index} 步，共 {total} 步：{stage}…",
    "Step {index} of {total}: {stage} — already done for this scan": "第 {index} 步，共 {total} 步：{stage}——这个扫描已经做过",
    "Finding the solutions of every time window": "正在求每个时间窗的解",  # “Writing the report” is in another table
    "Drawing the measured and the model images": "正在绘制实测图和模型图",
    "This request was analysed before: its result is shown ({windows} time windows).":
        "这个请求以前分析过：显示的是它已有的结果（{windows} 个时间窗）。",
    "The analysis of {name} is ready.": "{name} 的分析已完成。",
    "The analysis failed.": "分析失败。",
    "The choice could not be saved: {reason}": "无法保存所选的解：{reason}",
    # -- Analyze: problems and failures ------------------------------------------------------------------------------
    "Give the deposition rate (nm/s): the rate at the sample, or the reading of the quartz balance with QCM reading "
    "ticked.": "请给出沉积速率（nm/s）：样品处的速率，或者勾选“QCM 读数”后填石英晶体微天平的读数。",
    "Choose the substrate.": "请选择基底。",
    "The mode must be auto, quick, refine or full.": "模式必须是 auto、quick、refine 或 full。",
    "The deposition duration must be positive (or empty).": "沉积持续时间必须为正（或留空）。",
    "The polymer film thickness must be positive (or empty).": "聚合物膜厚必须为正（或留空）。",
    "The analysis package was not found. Choose its folder (it holds code and models), or set the environment "
    "variable GIMAP_SPUTTER_CLIENT to it.":
        "没有找到分析包。请选择它的文件夹（里面有 code 和 models），或把环境变量 GIMAP_SPUTTER_CLIENT 设为这个文件夹。",
    "{folder} is not an analysis package: {missing} is missing.": "{folder} 不是分析包：缺少 {missing}。",
    "The analysis stopped with an error ({status}); its last lines are under Details.":
        "分析出错结束（{status}）；最后几行在“详情”里。",
    "Stopped. Frames that were read completely are kept and reused by the next run of the same request.":
        "已停止。已经读完的帧会保留，同一请求下次运行时重用。",
    "The analysis finished, but this package does not write the data of the main screen (main/main.json): a newer "
    "package is needed. Its report is in {folder}.":
        "分析已完成，但这个分析包不写主屏用的数据（main/main.json）：需要更新的分析包。它的报告在 {folder}。",
    "The mask file cannot be read: {file}": "无法读取掩膜文件：{file}",
    "{folder} holds no result of the analysis (main/main.json).": "{folder} 里没有分析结果（main/main.json）。",
    "The result in {folder} cannot be read: {reason}": "无法读取 {folder} 里的结果：{reason}",
    "The images of window {window} are not in {folder}.": "{folder} 里没有时间窗 {window} 的图像。",
    "The images of window {window} cannot be read: {reason}": "无法读取时间窗 {window} 的图像：{reason}",
    "The analysis gave no solution for this window.": "分析没有给出这个时间窗的解。",
    # -- the main screen: a porous substrate ---------------------------------------------------------------------------
    "Island radius (relative)": "岛半径（相对值）",
    "Island share": "岛份额",
    "Cap share": "帽层份额",
    "Vertical correlation": "竖直关联长度",
    "Lateral correlation": "横向关联长度",
    "Film share": "成膜份额",
    "Share": "份额",
    "Radius of the resolvable islands. On a porous substrate it is a relative size: how it changes over the scan "
    "counts, not its value.": "可分辨的岛的半径。在多孔基底上它是相对尺寸：有意义的是它在扫描中的变化，而不是数值本身。",
    "Share of the metal that is in resolvable islands. Early in the deposition it can be above 1.":
        "金属中处在可分辨的岛里的份额。沉积初期可以大于 1。",
    "Share of the optically active metal that is in the cap layer on the particles of the substrate (0 to 1).":
        "光学上起作用的金属中，处在基底颗粒顶部帽层里的份额（0 到 1）。",
    "Vertical correlation length of the metal network in the pores.": "孔内金属网络的竖直关联长度。",
    "Lateral correlation length of the metal network in the pores.": "孔内金属网络的横向关联长度。",
    "Share of the scattering of the layer that leaves through the metal film (0 to 1).":
        "这一层的散射中经金属膜出射的份额（0 到 1）。",
    # -- the main screen: a series of static samples -------------------------------------------------------------------
    "Previous sample of the series": "上一个样品",
    "Next sample of the series": "下一个样品",
    "Sample shown": "显示的样品",
    "{label} · nominal {thickness} nm": "{label} · 名义厚度 {thickness} nm",
    "Sample {index} of {count}.": "第 {index} 个样品，共 {count} 个。",
    "Sample {index} of {count}: {frames} frames.": "第 {index} 个样品，共 {count} 个：用了 {frames} 帧。",
    "Substrate: {substrate}.": "基底：{substrate}。",
    "This sample has no solution: {reason}": "这个样品没有解：{reason}",
    "Range: {low} – {high}": "范围：{low} – {high}",
    "Nominal thickness": "名义厚度",
    "Model solution": "模型解",
    "Choose the folder of a scan, give the metal, the substrate and the deposition rate, and press Analyze. Open "
    "Result… shows a folder analysed before. Frames measured after the deposition are analysed in Details, under "
    "Static Frames.":
        "选择一个扫描的文件夹，给出金属、基底和沉积速率，然后按“分析”。“打开结果…”可以显示以前分析过的文件夹。"
        "沉积之后测的帧在“详情”的“静态帧”里分析。",
    # -- Details: Static Frames ----------------------------------------------------------------------------------------
    "Static Frames (after deposition)": "静态帧（沉积之后测量）",
    "Frames of a sample measured after its deposition: each sample is analysed into a series, which the main screen "
    "shows against the nominal thickness.": "样品镀完之后测的帧：每个样品分析后归入一个系列，主屏按名义厚度显示这个系列。",
    "Au films measured after the deposition, one sample at a time into a series.":
        "沉积之后测量的 Au 膜，每次分析一个样品，归入一个系列。",
    "Au films measured after the deposition at an incidence angle near {alpha}°, one sample at a time into a series.":
        "沉积之后在入射角约 {alpha}° 下测量的 Au 膜，每次分析一个样品，归入一个系列。",
    "folder, CBF file or zip archive": "文件夹、CBF 文件或 zip 压缩包",
    "File…": "文件…",
    "Choose one CBF file or a zip archive.": "选择单个 CBF 文件或一个 zip 压缩包。",
    "The GISAXS frames of one sample: a folder (every CBF file in it), one CBF file, or a zip archive. For an "
    "archive, the folder inside it follows after two colons: archive.zip::folder/":
        "一个样品的 GISAXS 帧：一个文件夹（其中的全部 CBF 文件）、单个 CBF 文件，或一个 zip 压缩包。"
        "压缩包内的文件夹写在两个冒号之后：archive.zip::folder/",
    "Sample name": "样品名",
    "The name of the sample in the series: letters, digits, - and _ (anything else becomes _). Analysing a name "
    "again with other frames or values replaces that sample.":
        "样品在系列里的名字：字母、数字、- 和 _（其他字符换成 _）。同一个名字用别的帧或数值再分析一次，会替换原来的样品。",
    "The nominal thickness of the film (nm): the position of the sample on the axis of the series.":
        "膜的名义厚度（nm）：样品在系列横轴上的位置。",
    "Bare substrate": "空基底",
    "Frames of the bare substrate, measured like the sample: they are used as the substrate part of its image. "
    "Empty: the film is taken to lie directly on Si.":
        "空基底的帧，测法与样品相同：用作样品图像里的基底部分。留空：认为膜直接长在 Si 上。",
    "Series folder": "系列文件夹",
    "The folder of the series. Each sample analysed into it is added to its report and to what the main screen "
    "shows; choosing the folder of an existing series adds to that series.":
        "系列的文件夹。分析进这个文件夹的每个样品都会加入它的报告和主屏显示的内容；选择已有系列的文件夹，就是往那个系列里加样品。",
    "The network's candidates, each drawn by the physical model, on a few frames of the sample; the best is kept.":
        "在样品的几帧上，把网络给出的候选解逐个用物理模型画成图，保留最好的一个。",
    "Also refines the solution with a short fit.": "再用一次较短的拟合精修这个解。",
    "A longer fit, on every usable frame of the sample.": "更长的拟合，用样品的全部可用帧。",
    "Geometry of these frames (an empty field leaves the value of the package)": "这些帧的几何（留空的项用分析包的值）",
    "of the package": "用分析包的值",
    "of the package (.npy)": "用分析包的掩膜（.npy）",
    "Beam centre in pixels as Analyze shows it (pixel corners on whole numbers, row 0 at the top). Empty: the value "
    "of the package.": "光束中心，单位像素，与“分析”工作区显示的一致（像素角点在整数上，第 0 行在最上面）。留空：用分析包的值。",
    "Masks of the Package": "分析包自带的掩膜",
    "The pixel masks the analysis package brings, by file name: choosing one fills the field.":
        "分析包自带的像素掩膜，按文件名列出：选一个就填进掩膜栏。",
    "The analysis package brings no masks": "分析包没有自带掩膜",
    "Package Values": "分析包的数值",
    "Fill the geometry fields with the values the analysis package uses for static frames.":
        "把分析包处理静态帧时所用的数值填进几何各栏。",
    "Analyze Sample": "分析这个样品",
    "Read the frames of this sample, find its solution and add it to the series; the main screen then shows the "
    "series.": "读取这个样品的帧，求出它的解并加入系列；随后主屏显示这个系列。",
    "Choose the Folder of the Frames": "选择帧所在的文件夹",
    "Choose a CBF File or a Zip Archive": "选择 CBF 文件或 zip 压缩包",
    "Choose the Folder of the Series": "选择系列的文件夹",
    "{file} holds no CBF frames.": "{file} 里没有 CBF 帧。",
    "Folder Inside the Archive": "压缩包里的文件夹",
    "The archive holds frames in several folders. Frames to use:": "压缩包里有几个文件夹含有帧。要用的帧：",
    "(frames at the top of the archive)": "（压缩包顶层的帧）",
    # -- Analyze Sample: the job status, problems and failures ---------------------------------------------------------
    "Finding the solution of the sample": "正在求这个样品的解",
    "Writing the report and the images of the series": "正在写系列的报告和图像",
    "Done: {samples} samples in the series.": "完成：系列里有 {samples} 个样品。",
    "This sample was analysed before: the series is shown ({samples} samples).":
        "这个样品以前分析过：显示的是已有的系列（{samples} 个样品）。",
    "Sample {name} is in the series.": "样品 {name} 已加入系列。",
    "Replace the Sample?": "替换这个样品？",
    "The series already has a sample named {name}, analysed from other frames or values. Analysing it again "
    "replaces that sample in the series.": "系列里已经有名为 {name} 的样品，是用别的帧或数值分析的。再分析一次会替换系列里的这个样品。",
    "Choose the frames of the sample: a folder, one CBF file or a zip archive.":
        "请选择样品的帧：一个文件夹、单个 CBF 文件或一个 zip 压缩包。",
    "The frames of the bare substrate were not found: choose a folder, one CBF file or a zip archive, or leave the "
    "field empty.": "没有找到空基底的帧：请选择一个文件夹、单个 CBF 文件或一个 zip 压缩包，或把这一栏留空。",
    "Give the sample a name with letters or digits.": "请给样品起一个含字母或数字的名字。",
    "A sample name cannot end in _ref: the frames of a bare substrate are kept under such a name.":
        "样品名不能以 _ref 结尾：空基底的帧就保存在这样的名字下。",
    "Give the nominal thickness of the sample (nm).": "请给出样品的名义厚度（nm）。",
    "Choose the folder of the series.": "请选择系列的文件夹。",
    "The mode must be quick, refine or full.": "模式必须是 quick、refine 或 full。",
    "The X-ray energy and the sample-detector distance must be positive (or empty).":
        "X 射线能量和样品到探测器的距离必须为正（或留空）。",
    "This analysis package has no entry for static frames: a newer package is needed.":
        "这个分析包没有处理静态帧的入口：需要更新的分析包。",
    "{folder} is not empty and holds no series of static frames: choose an empty folder or the folder of a series.":
        "{folder} 不是空文件夹，里面也没有静态帧的系列：请选择一个空文件夹，或一个系列的文件夹。",
    "The series in {folder} was analysed elsewhere and the frames imported for it are not on this computer: a sample "
    "cannot be added to it. Choose a new folder for the series.":
        "{folder} 里的系列是在别处分析的，为它导入的帧不在这台电脑上：不能再往里加样品。请为系列另选一个新文件夹。",
    "The series in {folder} was analysed with another pixel mask, and a series has one mask: choose a new folder for "
    "the series.": "{folder} 里的系列是用另一个像素掩膜分析的，而一个系列只能用一个掩膜：请为系列另选一个新文件夹。",
    "The data of the series in {folder} already hold a sample named {name}: give this sample another name.":
        "{folder} 里系列的数据中已经有名为 {name} 的样品：请给这个样品换一个名字。",
    "The analysis of the static frames stopped with an error ({status}); its last lines are in the log.":
        "静态帧的分析出错结束（{status}）；最后几行在日志里。",
    "Stopped. The samples analysed before stay in the series.": "已停止。之前分析的样品仍在系列里。",
}
