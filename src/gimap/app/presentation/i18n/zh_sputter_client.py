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
    "One growing film: the grains of the earlier windows, grown, merged where D grows and joined by new "
    "nuclei where it shrinks; radius, size distribution, height and base layer are those of the solution: "
    "{name}":
        "同一片膜的生长：晶粒沿用前面各窗的晶粒，D 变大处合并，D 变小处出现新核；半径、尺寸分布、高度和底层取自所选的解：{name}",
    "Connected film {thickness} nm; channels down to the substrate: {share} % of the view":
        "连通的膜 {thickness} nm；通到基底的沟道占视场的 {share} %",
    "Connected network across the view; channels down to the substrate: {share} % of the view":
        "贯穿视场的连通网络；通到基底的沟道占视场的 {share} %",
    "Separate islands": "分立的岛",
    "Coalescing: neighbouring grains touch": "聚并中：相邻晶粒相接",
    "Connected network across the view (percolated)": "连成贯穿视场的网络（已逾渗）",
    "Continuous film {thickness} nm, grains on top: {grains}": "连续膜 {thickness} nm，其上的晶粒：{grains}",
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
    "Polymer: a polymer film of another or unknown kind, a block copolymer for instance. The model describes every "
    "polymer as one film layer; which polymer it is changes the metal's sizes by a few percent.":
        "Polymer：其他或不确定种类的聚合物膜，例如嵌段共聚物。模型把任何聚合物都描述成一层膜；聚合物种类对金属尺寸的影响只有几个百分点。",
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
    ("network result: the joint fit of the whole scan did not finish (Details ▸ Logs), so the evolution "
     "can jump between windows"):
        "网络结果：整条扫描联合精修没有完成（见 详情 ▸ 日志），演化可能在相邻时间窗之间跳动",
    "network result: the joint fit of the whole scan is running and replaces it when it is done":
        "网络初步结果：整条扫描联合精修进行中，完成后自动替换",
    "The analysis failed.": "分析失败。",
    "The choice could not be saved: {reason}": "无法保存所选的解：{reason}",
    # -- Analyze: problems and failures ------------------------------------------------------------------------------
    "Give the deposition rate (Å/s or nm/s): the rate at the sample, or the reading of the quartz balance with QCM "
    "reading ticked.": "请给出沉积速率（Å/s 或 nm/s）：样品处的速率，或者勾选“QCM 读数”后填石英晶体微天平的读数。",
    "Choose the substrate.": "请选择基底。",
    "The mode must be auto, quick, refine or full.": "模式必须是 auto、quick、refine 或 full。",
    "The deposition duration must be positive (or empty).": "沉积持续时间必须为正（或留空）。",
    "The polymer film thickness must be positive (or empty).": "聚合物膜厚必须为正（或留空）。",
    "The analysis package was not found. Choose its folder (it holds code and models), or set the environment "
    "variable GIMAP_SPUTTER_CLIENT to it.":
        "没有找到分析包。请选择它的文件夹（里面有 code 和 models），或把环境变量 GIMAP_SPUTTER_CLIENT 设为这个文件夹。",
    "{folder} is not an analysis package: {missing} is missing.": "{folder} 不是分析包：缺少 {missing}。",
    "Pixel mask: {file}.": "像素掩膜：{file}。",
    "Pixel mask: {file} and the automatic mask from the frames ({shadow} pixels of beamstop and holder, {dead} dead, "
    "{hot} hot).": "像素掩膜：{file}，加上从帧本身得出的自动掩膜（挡块及支撑杆 {shadow} 个像素，死像素 {dead} 个，热像素 {hot} 个）。",
    "Pixel mask: the automatic mask from the frames ({shadow} pixels of beamstop and holder, {dead} dead, {hot} hot); "
    "{file} does not fit this detector.": "像素掩膜：从帧本身得出的自动掩膜（挡块及支撑杆 {shadow} 个像素，死像素 {dead} 个，热像素 {hot} 个）；{file} 与这台探测器不符。",
    "{folder} is not an analysis package ({missing} is missing): the one that comes with GIMaP is used.":
        "{folder} 不是分析包（缺少 {missing}）：改用随 GIMaP 提供的分析包。",
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
    "In Situ: choose the folder of a scan recorded during the deposition, give the metal, the substrate and the "
    "deposition rate, and press Analyze. After Deposition: Add Samples… lists the samples of a folder, their nominal "
    "thickness read from the folder names, and Analyze analyses the ticked ones. Open Result… shows a folder analysed "
    "before.":
        "原位：选择沉积过程中记录的扫描文件夹，给出金属、基底和沉积速率，然后按“分析”。沉积后：“添加样品…”列出一个文件夹里的样品，"
        "名义厚度从文件夹名里读出，按“分析”分析勾选的样品。“打开结果…”可以显示以前分析过的文件夹。",
    # -- Details: Static Frames ----------------------------------------------------------------------------------------
    "Static Frames (after deposition)": "静态帧（沉积之后测量）",
    "Sample": "样品",
    "Bare substrate": "裸基底",
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
    # -- the bar: what it analyses (views/sputter_main_view.py) ----------------------------------------------------
    "In Situ": "原位",
    "After Deposition": "沉积后",
    "Frames recorded during the deposition: the deposition rate gives the metal deposited at each time.":
        "沉积过程中记录的帧：由沉积速率得到每个时刻已沉积的金属量。",
    # -- Analyze Sample: the job status, problems and failures ---------------------------------------------------------
    "Finding the solution of the sample": "正在求这个样品的解",
    "Writing the report and the images of the series": "正在写系列的报告和图像",
    "Give the sample a name with letters or digits.": "请给样品起一个含字母或数字的名字。",
    "A sample name cannot end in _ref: the frames of a bare substrate are kept under such a name.":
        "样品名不能以 _ref 结尾：裸基底的帧就保存在这样的名字下。",
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
    "Fitting the samples of the series together stopped with an error ({status}); the samples keep their own "
    "solutions. The last lines are in the log.":
        "系列样品的联合拟合出错结束（{status}）；各样品保留各自的解。最后几行在日志里。",
    # -- the bar: the deposition rate and its unit (views/sputter_main_view.py, rate_flow.py) -------------------------
    "Rate": "速率",
    "Deposition rate, in the unit chosen beside it: needed, the solutions depend on the metal deposited at each time.":
        "沉积速率，单位在旁边选择：必填，各个解依赖每个时刻已沉积的金属量。",
    "Unit of the deposition rate: a quartz balance (QCM) shows Å/s; 1 nm/s = 10 Å/s.":
        "沉积速率的单位：石英晶体微天平（QCM）显示的是 Å/s；1 nm/s = 10 Å/s。",
    "Check the Deposition Rate": "请核对沉积速率",
    "This rate is unusual for sputtering ({nm} nm/s = {angstrom} Å/s at the sample). Is the unit right?":
        "这个速率对溅射来说不寻常（样品处 {nm} nm/s = {angstrom} Å/s）。单位对吗？",
    "Analyze Anyway": "仍然分析",
    "≈ {thickness} nm deposited at the last frame ({rate} nm/s at the sample)":
        "到最后一帧约沉积 {thickness} nm（样品处 {rate} nm/s）",
    "The geometry and the options of Analyze, and the windows, the report and the log of the result shown.":
        "几何与“分析”的选项，以及所示结果的时间窗、报告和日志。",
    # -- Details: the parallel processes (workers_flow.py) and the time per step (result_tabs.py) ---------------------
    "Parallel processes": "并行进程数",
    "Automatic ({count})": "自动（{count}）",
    "Automatic": "自动",
    "How many processes the analysis runs at once (Analyze and Static Frames). Automatic: one per core, two cores left "
    "free and at most 16, fewer when the free memory is short; worked out again when the analysis starts. More "
    "processes do not help when reading the frames from the disk is the slow part.":
        "分析同时运行几个进程（“分析”和静态帧都用这个设置）。自动：每个核心一个进程，留出两个核心，最多 16 个；"
        "空闲内存不够时更少；每次开始分析时重新计算。如果慢在从磁盘读帧，增加进程也没有帮助。",
    "Time per step": "各步用时",
    "Step": "步骤",
    "Time (m:ss)": "用时（分:秒）",
    "All steps": "全部步骤",
    "{read} of {total} frames read": "已读入 {read} 帧（共 {total} 帧）",
    "{read} frames read": "已读入 {read} 帧",
    # -- the solutions (client_texts.py) ------------------------------------------------------------------------------
    "Joint fit (whole scan)": "联合精修解",
}
