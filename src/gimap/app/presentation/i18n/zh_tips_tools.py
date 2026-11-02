"""Chinese for the shell, Start, Settings, Process with AI, Help and the tool windows: the shorter on-screen texts and the tooltips
that took over the explanations which used to stand under the controls (2026-11 small-text review)."""

TIPS_TOOLS_ZH = {
    # -- Start page (home_page.py): one line per card, the whole text in the card's tooltip ---------------
    "Peaks, orientation, crystallite size — no AI needed.": "峰、取向、晶粒尺寸——无需 AI。",
    "Yoneda cut, spacing and a model fit — no AI needed.": "Yoneda 切线、间距和模型拟合——无需 AI。",
    "Frame × q map, stages, odd frames, Batch Export — and Compare.": "帧 × q 热图、阶段、异常帧、批量导出——以及比较。",
    "Distance and beam centre from a standard (AgBh, LaB6, CeO2).": "由标样（AgBh、LaB6、CeO2）得到距离和光束中心。",
    "CBF, TIFF, EDF, NeXus / HDF5 — single frames or whole series": "CBF、TIFF、EDF、NeXus / HDF5 —— 单帧或整个序列",
    "The AI (Claude, DeepSeek, Qwen, OpenAI, a local model … — chosen in Settings) works with the same tools as you, "
    "with what you wrote as its task; its changes come back as cards to preview, apply or undo.":
        "AI（Claude、DeepSeek、通义千问、OpenAI、本地模型……在设置中选择）使用和你一样的工具，以你写的内容为任务；"
        "它的改动以卡片形式返回，可预览、应用或撤销。",
    # -- Settings ▸ Analyze (views/settings_dialog_view.py): the whole paragraph is the checkbox's tooltip ----
    "Off (recommended): new frames keep the beam centre of your calibrated instrument profile.":
        "关闭（推荐）：新帧保留你标定的仪器配置中的光束中心。",
    # -- Settings ▸ Assistant (assistant/presentation/settings_page.py, provider_section.py, code_section.py) ---
    "Each run sends the frame's status and reduced curves (numbers) to the chosen provider — never the detector file.":
        "每次运行把帧的状态和约化曲线（数字）发送给所选服务商 —— 绝不发送探测器文件。",
    "Other AI provider — DeepSeek, Qwen, OpenAI, Ollama … (below)": "其他 AI 服务商 —— DeepSeek、Qwen、OpenAI、Ollama …（见下方）",
    "Higher effort lets Claude think longer per step: slower, and more of your plan's usage or more tokens (other "
    "providers use their model's own setting).":
        "更高的思考强度让 Claude 每一步想得更久：更慢，也会用掉更多订阅额度或更多 token（其他服务商使用其模型自己的设置）。",
    "The start dialog can change the brain and the permission for a single run.": "开始对话框可以为单次运行更改 AI 引擎和权限。",
    "Added to every run; the AI can read and search the paths you write here.": "加入每次运行；AI 可以读取和搜索这里写的路径。",
    "An API key is billed per token and is separate from a Claude plan. A key saved here takes precedence over the "
    "ANTHROPIC_API_KEY environment variable. The panel shows the tokens and the estimated cost of every run.":
        "API 密钥按 token 计费，与 Claude 订阅无关。保存在这里的密钥优先于 ANTHROPIC_API_KEY 环境变量。面板会显示每次运行的 token 数和估计费用。",
    "Saved keys are kept in plain text in the user data folder (readable by your account only).":
        "保存的密钥以明文存放在用户数据文件夹中（仅你的账户可读）。",
    "A key saved here takes precedence over the provider's environment variable.": "保存在这里的密钥优先于服务商的环境变量。",
    "The model must support tool calling (function calling): the test sends one tiny request with a tool.":
        "模型必须支持工具调用（function calling）：测试会发送一个带工具的极小请求。",
    "Claude Code gets only the GIMaP tools — no shell, files or web.": "Claude Code 只能使用 GIMaP 的工具 —— 没有命令行、文件或网络。",
    "Program: claude.exe of Claude Code. The Claude desktop app keeps one under "
    "%LOCALAPPDATA%\\Packages\\Claude_…\\LocalCache\\Roaming\\Claude\\claude-code.":
        "程序：Claude Code 的 claude.exe。Claude 桌面应用在 "
        "%LOCALAPPDATA%\\Packages\\Claude_…\\LocalCache\\Roaming\\Claude\\claude-code 下自带一个。",
    # -- Process with AI, the start dialog (assistant/presentation/start_dialog.py) -------------------------
    "Sent to the AI: the frame's status and reduced curves as numbers, never the detector file.":
        "发送给 AI 的：帧的状态和约化曲线（数字），绝不发送探测器文件。",
    "The AI operates the Analyze tools on this frame while you watch each step, then reports every result ticked "
    "above; what cannot be determined is reported with the reason (e.g. no signal above the noise).":
        "AI 在这一帧上操作分析工具，你可以看着每一步，然后报告上面勾选的每个结果；无法确定的内容会连同原因一起报告（例如没有高于噪声的信号）。",
    "No frame open — open one in Analyze first.": "没有打开的帧 —— 请先在分析中打开一帧。",
    "The AI's changes come back as cards: look at the picture, then apply or dismiss each one.":
        "AI 的修改以卡片返回：看过图片后再逐个应用或忽略。",
    "Send a small q-map image (qualitative; Claude only)": "发送一张小的 q 图（仅定性；仅 Claude）",
    "The AI gets a small picture of the q map to judge it by eye; every number still comes from the tools. Only the "
    "Claude brains read images.": "AI 会收到一张小的 q 图，用来目测判断；所有数值仍来自工具。只有 Claude 引擎能读图像。",
    # -- Help (help_dialogs.py) ------------------------------------------------------------------------------
    "Menu keys work on every page; page keys only while the page has the focus (click into it first). Ctrl+C copies "
    "the selected rows.": "菜单快捷键在每个页面都有效；页面快捷键只在页面有焦点时有效（先在页面里点一下）。Ctrl+C 复制选中的行。",
    "Please cite GIMaP as below, with the version you used (software_version in the export records).":
        "请按下面的格式引用 GIMaP，并写明所用版本（导出记录中的 software_version）。",
    # -- Geometry Calibration (features/calibration/presentation) ---------------------------------------------
    "Image loaded{notes}. Click Auto Calibration.": "已载入图像{notes}。请点击“自动标定”。",
    "Few azimuths carry matched ring points, so the centre and the distance are poorly constrained. Check that the "
    "matched (yellow) rings lie on the rings of the image before Apply.":
        "匹配到的环点只覆盖很少的方位角，中心和距离约束不足。应用之前请核对匹配的（黄色）环是否落在图像的环上。",
    "Detector overrides (pixel size), custom distance bounds for the search, and preview overlays.":
        "探测器参数覆盖（像素尺寸）、搜索用的自定义距离范围和预览叠加。",
    "Fine-tune the selected solution: centre, distance or which detected ring belongs to which theoretical peak.":
        "微调所选解：中心、距离，或哪个检测到的环对应哪个理论峰。",
    # -- Format Converter (features/format_converter/presentation) -------------------------------------------
    "NXS / HDF5, CBF or TIFF detector files, in any combination; single images and multi-frame files are recognised "
    "automatically. The sources are only read, never changed.":
        "NXS / HDF5、CBF 或 TIFF 探测器文件，可任意组合；自动识别单张图像和多帧文件。只读取源文件，不会改动它们。",
    "No input files yet — add NXS, CBF or TIFF files or a folder.": "还没有输入文件 —— 请添加 NXS、CBF 或 TIFF 文件或文件夹。",
    "Ranges, custom expressions (e.g. 1, 5, 8–20) or every Nth frame, for multi-frame NXS inputs.":
        "多帧 NXS 输入可按范围、自定义表达式（例如 1, 5, 8–20）或每 N 帧选择。",
    # -- XRR Series Extractor (features/xrr/presentation) -----------------------------------------------------
    "Stream GIWAXS/GISAXS detector frames and extract the moving specular reflection.":
        "逐帧读取 GIWAXS/GISAXS 探测器图像，提取移动的镜面反射。",
    "One NXS / HDF5 file, a CBF file, or the folder of the series. Module files are stitched per frame; a file "
    "holding one frame stands for the series of files numbered like it.":
        "一个 NXS / HDF5 文件、一个 CBF 文件，或序列所在的文件夹。模块文件按帧拼接；只含一帧的文件代表与它编号方式相同的一系列文件。",
    "Theta is the sample-table angle; the beam and the detector stay fixed.": "Theta 是样品台的角度；光束和探测器保持不动。",
    "The center is the direct-beam position at theta = 0. Pick it on the preview if needed.":
        "中心是 theta = 0 时直射束的位置。需要时可以在预览上点选。",
    "Radius 0 reads one pixel; a larger radius integrates a circle around the calculated specular beam.":
        "半径 0 只读一个像素；更大的半径对计算出的镜面反射周围的圆形区域积分。",
    "Not in the file — check: {fields}": "文件里没有，请核对：{fields}",
    "{fields} (defaults)": "{fields}（默认值）",
    "{fields} (image center)": "{fields}（图像中心）",
    # -- SEM Particle Analysis (features/sem/presentation) ---------------------------------------------------
    "Scale from these values only — never read from the image.": "标尺只取自这里输入的数值，不从图像读取。",
    "Enter nm per pixel, or a scale-bar length in nm and in pixels. Scale-bar text, magnification or DPI in the "
    "image are never used.": "输入 nm/像素，或比例尺的长度（nm 和像素）。图中的比例尺文字、放大倍数或 DPI 都不会被使用。",
    "Threshold / watershed works on the stored image intensities (no rescaling).":
        "阈值 / 分水岭根据图像的存储灰度工作（不做缩放）。",
    "Measured. Review the outlines and the particle table.": "测量完成。请检查分割轮廓和逐颗粒测量表。",
    "R is the equivalent-circle radius of the projected area.": "R 是投影面积的等效圆半径。",
    "CSV sizes are converted to radius in nm using the selected column, quantity, and unit.":
        "CSV 粒径按所选列、半径或直径及单位换算为 nm 半径。",
    "Original measurements and every comparison setting are included in the export.":
        "导出结果包含原始测量值和全部比较设置。",
    "SEM image": "SEM 图像",
}
