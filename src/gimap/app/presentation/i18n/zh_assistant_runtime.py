"""Chinese for the sentences the automatic analysis composes at run time (Results tab, outcome lines, answer form)."""

ASSISTANT_RUNTIME_ZH = {
    # -- automatic analysis, second wave (2026-10-06) ----------------------------------------------
    "Calibration File": "选择标定文件",
    "an image of a standard, or a .poni file": "标样图像，或 .poni 文件",
    "Images of a standard and .poni files": "标样图像和 .poni 文件",
    "GIMaP calibration": "GIMaP 标定",
    "All files": "所有文件",
    "Or press Enter in any of the fields": "也可以在任一输入框中按回车",
    "Choose the calibration file: an image of a standard or a .poni file (the dialog starts in the frame's folder)":
        "选择标定文件：标样图像或 .poni 文件（对话框从这一帧所在的文件夹开始）",
    "Web page with pictures": "带图片的网页",
    "Markdown text": "Markdown 文本",
    "both": "两者",
    "OOP only": "仅 OOP",
    "IP only": "仅 IP",
    "neither": "均未见",
    "n/a": "无法判断",
    "check": "待查",
    "reliable": "可靠",
    "spike": "尖峰",
    "halo": "晕环",
    "weak": "弱",
    "no fit": "未拟合",
    "edge": "边缘",
    "q and FWHM in Å⁻¹, d in Å, L (Scherrer size, a lower bound) in nm. OOP: out-of-plane (χ ≈ 0°), IP: in-plane (χ ≈ ±90°), *: one sector only. Hover a cell for the whole sentence; the saved report has the full table.":
        "单位：q 与 FWHM 为 Å⁻¹，d 为 Å，L（Scherrer 尺寸，下限）为 nm。OOP：面外（χ ≈ 0°），IP：面内（χ ≈ ±90°），*：只有一个扇区。鼠标停在单元格上可看完整说明；保存的报告中有完整的表格。",
    "artefact (spike)": "伪影（尖峰）",
    "halo, not a crystal peak": "晕环，不是晶体峰",
    "weak, tentative": "弱，待定",
    "fit failed: check the image": "拟合失败：请检查图像",
    "at the end of the data: check": "位于数据末端：请检查",
    "The peak lies within 1.5 widths of the end of the measured q range: its shape and position may be cut off. Look at I(q) before using it.":
        "这个峰距测量 q 范围的末端不到 1.5 个峰宽：它的形状和位置可能被截断。使用前请查看 I(q)。",
    "Where the peak is: the centre of a Gaussian fitted on a local linear background to the radial I(q) of the whole detector (all χ).":
        "峰的位置：在整个探测器（所有 χ）的径向 I(q) 上，以局部线性背景拟合的高斯峰的中心。",
    "The lattice spacing of this reflection: d = 2π / q.": "这条衍射对应的晶面间距：d = 2π / q。",
    "The width of the fitted peak (full width at half maximum). It includes the instrument's broadening.":
        "拟合峰的宽度（半高全宽），其中包含仪器展宽。",
    "Whether this is a crystal peak: spikes (one hot pixel, a streak or a flat-topped box from one detector row), broad halos (amorphous order) and weak peaks are flagged.":
        "是否为晶体峰：尖峰（单个热像素、条纹或探测器某一行造成的平顶方块）、宽的晕环（非晶有序）和弱峰都会被标出。",
    "Scherrer: L = 2π·0.9 / FWHM. The instrument's broadening is not removed, so it is a lower bound (≥) of the crystallite size.":
        "Scherrer：L = 2π·0.9 / FWHM。没有扣除仪器展宽，因此它是晶粒尺寸的下限（≥）。",
    "The net intensity per pixel at this q in the out-of-plane sector (χ ≈ 0°, along the surface normal) compared with the in-plane sector (χ ≈ ±90°).":
        "在这个 q 处，面外扇区（χ ≈ 0°，沿表面法线）与面内扇区（χ ≈ ±90°）每像素净强度的比较。",
    "mainly out-of-plane": "主要在面外",
    "mainly in-plane": "主要在面内",
    "both sectors (no strong preference)": "两个扇区都有（无明显偏好）",
    "out-of-plane only": "仅在面外",
    "in-plane only": "仅在面内",
    "not detected in either sector": "两个扇区都未检测到",
    "out-of-plane only (the other sector is in a shadow)": "仅面外（另一个扇区在阴影中）",
    "in-plane only (the other sector is in a shadow)": "仅面内（另一个扇区在阴影中）",
    "cannot tell here (shadow / not measured)": "此处无法判断（阴影 / 未测量）",
    "not measured at this q": "此 q 处未测量",
    "in-plane only measured": "只测量了面内",
    "out-of-plane only measured": "只测量了面外",
    "only the in-plane sector is usable here: the other is shadowed (peak seen)": "此处只有面内扇区可用：另一个在阴影中（看到峰）",
    "only the in-plane sector is usable here: the other is shadowed (no peak)": "此处只有面内扇区可用：另一个在阴影中（没有峰）",
    "only the out-of-plane sector is usable here: the other is shadowed (peak seen)": "此处只有面外扇区可用：另一个在阴影中（看到峰）",
    "only the out-of-plane sector is usable here: the other is shadowed (no peak)": "此处只有面外扇区可用：另一个在阴影中（没有峰）",
    "only the in-plane sector is measured here (peak seen)": "此处只测量了面内扇区（看到峰）",
    "only the in-plane sector is measured here (no peak)": "此处只测量了面内扇区（没有峰）",
    "only the out-of-plane sector is measured here (peak seen)": "此处只测量了面外扇区（看到峰）",
    "only the out-of-plane sector is measured here (no peak)": "此处只测量了面外扇区（没有峰）",
    "not measured: neither sector reaches this q on the detector": "未测量：两个扇区在探测器上都达不到这个 q",
    "not comparable: the in-plane sector is shadowed and the out-of-plane sector is shadowed":
        "无法比较：面内扇区在阴影中，面外扇区也在阴影中",
    "not comparable: the in-plane sector is shadowed and the out-of-plane sector is not measured":
        "无法比较：面内扇区在阴影中，面外扇区未测量",
    "not comparable: the in-plane sector is not measured and the out-of-plane sector is shadowed":
        "无法比较：面内扇区未测量，面外扇区在阴影中",
    "a spike one or two bins wide far above the background: hot pixels, a module edge or a zinger rather than diffraction (check whether the calibration image shows it too)":
        "一个只有一两个 bin 宽、远高于背景的尖峰：更可能是热像素、模块边缘或宇宙射线（zinger），而不是衍射（请检查标定图像中是否也有）",
    "its shape could not be fitted: look at the image at this q (module gap, detector edge, overlap)":
        "无法拟合它的形状：请在图像上查看这个 q 处（模块缝隙、探测器边缘、重叠）",
    "weak (3–5σ): tentative, may be noise": "弱（3–5σ）：待定，可能是噪声",
    "a broad halo (amorphous or liquid-like order), not a crystalline reflection": "宽的晕环（非晶或类液体有序），不是晶体衍射",
    "{ranges} at q {q}": "{ranges}（q {q}）",
    "Shadow (orange on the map): |χ| {where} Å⁻¹. The intensity there is far below the diffuse background, so these pixels count as unmeasured.":
        "阴影（图上橙色）：|χ| {where}（q 以 Å⁻¹ 计）。那里的强度远低于漫散射背景，因此这些像素按未测量处理。",
    "Missing wedge (red dashed next to qz): |χ| {where} Å⁻¹ is not measured, so orientations closest to the surface normal are missing and Herman's f is biased low.":
        "缺失楔形区（qz 旁的红色虚线）：|χ| {where}（q 以 Å⁻¹ 计）未测量，因此最接近表面法线的取向缺失，Herman 取向因子 f 偏低。",
    "Ring q ≈ {q} Å⁻¹: f = {f} (a random ring would give {random} here) — {texture}":
        "环 q ≈ {q} Å⁻¹：f = {f}（随机取向的环在这里会给出 {random}）—— {texture}",
    "Ring q ≈ {q} Å⁻¹: f = {f} — {texture}": "环 q ≈ {q} Å⁻¹：f = {f} —— {texture}",
    "Ring q ≈ {q} Å⁻¹: orientation not determined — only {covered} of the range Herman's f needs is measured ({gaps}). Hover for details.":
        "环 q ≈ {q} Å⁻¹：取向无法确定 —— Herman 取向因子 f 所需的范围只测到了 {covered}（{gaps}）。鼠标悬停可看详情。",
    "Ring q ≈ {q} Å⁻¹: orientation not determined — only {covered} of the range Herman's f needs is measured. Hover for details.":
        "环 q ≈ {q} Å⁻¹：取向无法确定 —— Herman 取向因子 f 所需的范围只测到了 {covered}。鼠标悬停可看详情。",
    "Ring q ≈ {q} Å⁻¹: {reason}": "环 q ≈ {q} Å⁻¹：{reason}",
    "in a shadow at |χ| {ranges}": "|χ| {ranges} 在阴影中",
    "missing wedge below {angle}°": "{angle}° 以下为缺失楔形区",
    "isotropic within the noise (random orientation)": "在噪声范围内各向同性（随机取向）",
    "oriented along the surface normal (out-of-plane, χ ≈ 0°)": "沿表面法线取向（面外，χ ≈ 0°）",
    "oriented in the sample plane (in-plane, χ ≈ ±90°)": "在样品平面内取向（面内，χ ≈ ±90°）",
    "{title}: {why}": "{title}：{why}",
    "q = {q} Å⁻¹: {caveat}": "q = {q} Å⁻¹：{caveat}",
    "Note: {warning}": "注意：{warning}",
    "Note: {warning} — the line check above already confirms the calibration.": "注意：{warning} —— 上面的谱线检查已经确认了标定。",
    "{what}: {decision} — {why}": "{what}：{decision} —— {why}",
    "instrument profile “{name}”": "仪器配置“{name}”",
    "Calibration candidates": "标定候选",
    "a saved calibration (not re-checked against a standard image)": "已保存的标定（没有再用标样图像检查）",
    "kept the instrument profile “{name}” this detector already has": "沿用了这台探测器已有的仪器配置“{name}”",
    "a saved profile is the person's own calibration (recalibrate to replace it)": "保存的仪器配置是你自己的标定（重新标定可以替换它）",
    "A series of {total} frames: these results are frames {first}–{last} (the final state).":
        "共 {total} 帧的序列：这些结果来自第 {first}–{last} 帧（最终状态）。",
    "at the start": "开头",
    "at the end": "结尾",
    "change": "变化",
    "Frames 1–10 summed.": "第 1–10 帧求和。",
    "The frames in the results above.": "上面结果所用的帧。",
    "Reliable peaks only. Peaks closer than half their width are the same line; it shifted if it moved by more than 0.3 % in q. \"weak\": a tentative peak.":
        "只计可靠的峰。相距不到半个峰宽的峰视为同一条线；q 变化超过 0.3 % 时算作移动。“弱”：待定的峰。",
    "present at both": "两端都有",
    "appeared": "出现",
    "disappeared": "消失",
    "grew (weak at the start)": "增强（开头较弱）",
    "faded (weak at the end)": "减弱（结尾较弱）",
    "shifted {move} from {q}": "移动 {move}（原在 {q}）",
    "moved? {move}: one line shifting further than its width, or one line replacing another":
        "可能移动 {move}：一条线移动得比它的宽度更远，或一条线取代了另一条",
    "Beam centre on the symmetry axis: x = {x} px ({shift} px).": "对称轴上的光束中心：x = {x} px（{shift} px）。",
    "D ≈ 2π/q* = {d} nm (maximum at |qy| = {q} Å⁻¹).": "D ≈ 2π/q* = {d} nm（极大值位于 |qy| = {q} Å⁻¹）。",
    "A shoulder at |qy| = {q} Å⁻¹ (2π/q ≈ {d} nm): a hint, not a resolved peak.":
        "|qy| = {q} Å⁻¹ 处有一个肩峰（2π/q ≈ {d} nm）：只是提示，不是分辨出的峰。",
    "Fitted: {curve} ({points} points); R, h and D in nm. Numerical fits of single particle families with size dispersity and a paracrystal distance D; choose the model from what you know of the sample.":
        "已拟合：{curve}（{points} 个点）；R、h 和 D 的单位为 nm。对单一粒子类型做数值拟合，含尺寸分布和次晶距离 D；请根据你对样品的了解选择模型。",
    "R (nm): the particle radius.": "R (nm)：粒子半径。",
    "One or more shape/resolution parameters reached a search bound; values may be unidentifiable.":
        "有形状 / 分辨率参数到达了搜索边界；这些数值可能无法确定。",
    "h (nm): the cylinder height (— for a sphere).": "h (nm)：圆柱高度（球体为 —）。",
    "D (nm): the paracrystal distance between particles.": "D (nm)：粒子之间的次晶距离。",
    "No detector image is open.": "没有打开探测器图像。",
    "Give the path of a detector image.": "请给出探测器图像的路径。",
    "Check that GIMaP can read this file.": "请检查 GIMaP 能否读取这个文件。",
    "Show that file in Analyze again and run the analysis again.": "请在分析中重新显示那个文件，然后再运行一次分析。",
    "Check the frame in the GUI.": "请在界面中检查这一帧。",
    "Neither the options, the notes nor an instrument profile give αi, so 0° is used. Ring positions |q| barely change, but qz shifts by about k·sin αi (≈0.04 Å⁻¹ at 0.4° and 12 keV) and the missing wedge moves.":
        "选项、笔记和仪器配置都没有给出 αi，因此使用 0°。环的位置 |q| 几乎不变，但 qz 会偏移约 k·sin αi（0.4°、12 keV 时约 0.04 Å⁻¹），缺失楔形区也会移动。",
    "Beamtime notes, the logbook or the slides usually state it (typically 0.1–0.5°); otherwise ask.":
        "实验笔记、记录本或幻灯片通常会写明（一般为 0.1–0.5°）；否则请询问。",
    "The person declined saving the geometry, so nothing is in q.": "你没有同意保存几何参数，因此无法换算到 q。",
    "Approve the geometry, or calibrate in Tools ▸ Geometry Calibration.": "请确认几何参数，或在 工具 ▸ 几何标定 中标定。",
    "No calibration candidate gave a good geometry (see decisions for each one).": "没有一个标定候选给出好的几何参数（每个候选的情况见决策）。",
    "Name the calibration image or file used at the beamtime; the notes usually say which.":
        "请指明实验时使用的标定图像或文件；笔记中通常会写明。",
    "The folders around the frame could not be searched.": "无法搜索这一帧周围的文件夹。",
    "Name the calibration file.": "请指明标定文件。",
    "An image of AgBh, LaB6, CeO2 or a LaB6+CeO2 mixture taken with this detector, or a .poni / GIMaP calibration file; a log or the beamtime notes usually name it.":
        "用这台探测器拍摄的 AgBh、LaB6、CeO2 或 LaB6+CeO2 混合物的图像，或 .poni / GIMaP 标定文件；日志或实验笔记中通常会写明。",
    "Neither the images' headers, the options nor the notes give the energy; calibration needs it.":
        "图像头信息、选项和笔记都没有给出能量；标定需要它。",
    "Beamtime notes or the logbook; P03 GIWAXS is often 11.8 or 12.4 keV but never guess.":
        "查看实验笔记或记录本；P03 的 GIWAXS 常用 11.8 或 12.4 keV，但不要猜。",
    "The detector's pixel size: Pilatus 172 µm, Eiger 75 µm, Lambda 55 µm; the notes or the detector name usually say which.":
        "探测器的像素尺寸：Pilatus 172 µm、Eiger 75 µm、Lambda 55 µm；笔记或探测器名称通常会说明。",
    "The notes or the file name usually say which standard it is.": "笔记或文件名通常会说明是哪种标样。",
    "Beamtime notes or the logbook.": "实验笔记或记录本。",
    "Look at the horizontal cut: its halves should mirror each other.": "查看水平切线：它的两半应当互为镜像。",
    "No Yoneda band was found above the horizon; the horizontal cut sits just above it.":
        "在样品地平线上方没有找到 Yoneda 带；水平切线位于地平线正上方。",
    "Check αi (the horizon moves with it), or drag the yellow band on the image to the Yoneda row.":
        "请检查 αi（地平线随它移动），或在图像上把黄色条带拖到 Yoneda 所在的行。",
    "The best fit did not converge.": "最佳拟合没有收敛。",
    "Open the curve in Fitting (Send to Fitting) and refine it with a chosen model.": "在拟合中打开这条曲线（发送到拟合），用选定的模型精修。",
    "{texture} — but the strongest measured |χ| ({chi}°) borders an unmeasured range, so the true maximum may lie inside it":
        "{texture} —— 但测得的最强处（|χ| = {chi}°）紧邻未测量的范围，真正的极大值可能就在其中",
    "no single preferred orientation: maxima of similar height at |χ| ≈ {a}° and {b}°":
        "没有单一的择优取向：在 |χ| ≈ {a}° 和 {b}° 处有高度相近的极大值",
    "weak or no preferred orientation (f = {f}): the ring is strongest at |χ| ≈ {chi}° but nearly as intense elsewhere":
        "择优取向很弱或没有（f = {f}）：环在 |χ| ≈ {chi}° 处最强，但在其他位置几乎同样强",
    "tilted: maximum at χ ≈ ±{chi}°": "倾斜取向：极大值位于 χ ≈ ±{chi}°",
    "Herman's f is sin χ weighted: it assumes the film is isotropic in its plane (fibre texture).":
        "Herman 取向因子 f 以 sin χ 加权：它假定薄膜在其平面内各向同性（纤维织构）。",
    "The radial background under the ring ({background} counts/pixel) was subtracted.":
        "已扣除环下方的径向背景（{background} counts/pixel）。",
    "|χ| {where} is shadowed: the intensity there is below {fraction} of the diffuse background at this q (a shadow, absorber or insensitive detector area), so it counts as unmeasured.":
        "|χ| {where} 在阴影中：那里的强度低于此 q 处漫散射背景的 {fraction}（阴影、吸收体或探测器的不灵敏区域），因此按未测量处理。",
    "|χ| < {angle}° is not measured (missing wedge); f then leaves out the orientations closest to the surface normal and is biased low.":
        "|χ| < {angle}° 未测量（缺失楔形区）；f 因此缺少最接近表面法线的取向，数值偏低。",
    "A random (isotropic) ring measured over the same |χ| would give f = {f}: compare f with that, not with 0.":
        "在相同 |χ| 范围内测得的随机（各向同性）环会给出 f = {f}：应把 f 与这个值比较，而不是与 0 比较。",
    "Only {n} measured χ bins in this ring.": "这个环中只有 {n} 个测得的 χ 区间。",
    "The ring is not above the radial background: at most {snr}σ in any 1° χ bin.":
        "这个环没有高出径向背景：任一 1° 的 χ 区间中最多 {snr}σ。",
    "Only {covered} of the orientation range Herman's f weighs (sin χ) is measured at this q (|χ| {where}); f and the orientation distribution need most of it, above all near the sample plane. Within the measured part the ring is strongest at |χ| ≈ {chi}°.":
        "此 q 处只测到了 Herman 取向因子 f 所加权（sin χ）的取向范围的 {covered}（|χ| {where}）；f 和取向分布需要其中的大部分，尤其是靠近样品平面的部分。在测得的部分中，环在 |χ| ≈ {chi}° 处最强。",
    "Only the {where} half is on the detector.": "探测器上只有 {where} 这一半。",
    "The {side} half reaches only |qy| = {reach} Å⁻¹ (the {other} half {other_reach} Å⁻¹), so the {used} half is used.":
        "{side} 这一半只到 |qy| = {reach} Å⁻¹（{other} 这一半到 {other_reach} Å⁻¹），因此使用 {used} 这一半。",
    "The {side} half has points in only {coverage} of its |qy| range (detector gaps or the beam-stop shadow; the {other} half: {other_coverage}), so the {used} half is used.":
        "{side} 这一半只在其 |qy| 范围的 {coverage} 内有数据点（探测器缝隙或束流挡块的阴影；{other} 这一半：{other_coverage}），因此使用 {used} 这一半。",
    "The halves differ by {percent} % (median) up to |qy| = {q} Å⁻¹ even after the symmetry correction (a shadow, an absorber or real in-plane anisotropy): both are kept and the {better} half, the better covered one, is fitted.":
        "即使经过对称校正，两半在 |qy| = {q} Å⁻¹ 以内仍相差 {percent} %（中位数）（阴影、吸收体或真实的面内各向异性）：两半都保留，并拟合覆盖更好的 {better} 这一半。",
    "Both halves are usable (points in {a} and {b} of their |qy| ranges) and {agree}: they are averaged where both exist, which halves the noise{extend}.":
        "两半都可用（分别在各自 |qy| 范围的 {a} 和 {b} 内有数据点），并且{agree}：在两半都有数据的地方取平均，噪声减半{extend}。",
    "agree within {percent} %": "相差在 {percent} % 以内",
    "overlap too little to compare": "重叠太少，无法比较",
    "; beyond {q} Å⁻¹ the longer {side} half continues alone to {reach} Å⁻¹":
        "；超过 {q} Å⁻¹ 后，较长的 {side} 这一半单独延续到 {reach} Å⁻¹",
    "mean of both halves up to |qy| = {q} Å⁻¹": "两半的平均，至 |qy| = {q} Å⁻¹",
    "then the {side} half alone to {q} Å⁻¹": "之后只用 {side} 这一半，至 {q} Å⁻¹",
    "the {side} half": "{side} 这一半",
    "{group} neighbouring points merged ({count} points for the fit)": "每 {group} 个相邻点合并为一个（拟合用 {count} 个点）",
    "{n} solutions of {families} particle families fit within {percent} of the best χ²: the curve alone does not decide the model.":
        "{n} 个解（{families} 种粒子类型）的 χ² 都在最佳值的 {percent} 以内：仅凭这条曲线无法确定模型。",
    "Solutions whose D agrees with the observed spacing: {solutions}. Choose the particle shape you expect and refine that model in Fitting; compare the fitted curves.":
        "D 与观测到的间距一致的解：{solutions}。请选择你预期的粒子形状，在拟合中精修这个模型，并比较拟合曲线。",
    "Choose the particle shape you expect and refine that model in Fitting; compare the fitted curves.":
        "请选择你预期的粒子形状，在拟合中精修这个模型，并比较拟合曲线。",
    "The symmetry axis is {shift} px from the calibrated centre: check the calibration or the sample alignment.":
        "对称轴与标定的中心相差 {shift} px：请检查标定或样品的对准。",
    "No calibration file and no image of a standard GIMaP can fit was found ({n} folders searched).":
        "没有找到标定文件，也没有找到 GIMaP 能拟合的标样图像（搜索了 {n} 个文件夹）。",
    "GIMaP reduces this frame as {mode} and could not switch to GISAXS.": "GIMaP 按 {mode} 处理这一帧，且无法切换到 GISAXS。",
    "GIMaP reduces this frame as {mode} and could not switch to GIWAXS.": "GIMaP 按 {mode} 处理这一帧，且无法切换到 GIWAXS。",
    "{name} has no pixel size in its header, so it cannot be calibrated.": "{name} 的头信息中没有像素尺寸，因此无法标定。",
    "The frame changed during the run: it started on {start}, Analyze now shows {now}.":
        "运行期间显示的帧变了：运行开始于 {start}，分析现在显示的是 {now}。",
    "Bounded soft-L1 least_squares with weighted nonnegative linear-amplitude least squares at each shape evaluation":
        "有界 soft-L1 least_squares，每次形状评估时用加权非负线性振幅最小二乘",
    "{name} and the frame give no energy.": "{name} 和这一帧都没有给出能量。",
    "good: {lines} lines of the standard land within {error} of their q on average":
        "好：标样的 {lines} 条谱线平均落在其 q 的 {error} 以内",
    "usable: {lines} lines within {error} of their q; peak positions good to about that":
        "可用：{lines} 条谱线在其 q 的 {error} 以内；峰位精度大致如此",
    "doubtful: the standard's lines are {error} off on average; try another image or standard":
        "存疑：标样的谱线平均偏离 {error}；请换一张图像或另一种标样",
    "none fits well: the best, {standard}, still puts its lines {error} off their q; the image may show another standard, or the energy or pixel size is wrong":
        "都拟合得不好：最好的 {standard} 的谱线仍偏离其 q {error}；图像可能是另一种标样，或者能量或像素尺寸有误",
    "clear: {standard} — {lines} lines within {error} of their q": "明确：{standard} —— {lines} 条谱线在其 q 的 {error} 以内",
    "ambiguous: {standard} ({error}) and {second} ({second_error}) place their lines about equally well; check a log or ask the user":
        "不确定：{standard}（{error}）和 {second}（{second_error}）的谱线位置吻合得差不多；请查看日志或询问用户",
    "none fits well (best {standard}: {rings} rings, rms {rms} px): the image may show another standard, the energy or pixel size may be wrong, or the rings are too weak":
        "都拟合得不好（最好的 {standard}：{rings} 个环，rms {rms} px）：图像可能是另一种标样，能量或像素尺寸可能有误，或者环太弱",
    "clear: {standard} ({rings} rings, rms {rms} px) beats {second} ({second_rings} rings, rms {second_rms} px)":
        "明确：{standard}（{rings} 个环，rms {rms} px）优于 {second}（{second_rings} 个环，rms {second_rms} px）",
    "clear: {standard} ({rings} rings, rms {rms} px)": "明确：{standard}（{rings} 个环，rms {rms} px）",
    "clear: {standard}, and its distance agrees with the expected {distance} mm":
        "明确：{standard}，其距离与预期的 {distance} mm 一致",
    "probably {second}: its distance agrees with the expected {distance} mm, although {standard} matches as many rings":
        "可能是 {second}：其距离与预期的 {distance} mm 一致，尽管 {standard} 匹配的环一样多",
    "ambiguous: {standard} and {second} fit about equally well; check the distance in the header or a log, or ask the user":
        "不确定：{standard} 和 {second} 拟合得差不多；请查看头信息或日志中的距离，或询问用户",
    "good: several rings match the standard with a small residual": "好：多个环与标样匹配，残差小",
    "unreliable: fewer than two rings matched, the distance is ambiguous": "不可靠：匹配的环少于两个，距离不确定",
    "doubtful: few rings or a large residual; compare the alternatives or another image": "存疑：环太少或残差大；请比较其他候选或另一张图像",
    "calibrated from {name}": "由 {name} 标定",
    "could not use {name}": "无法使用 {name}",
    "rejected {name} ({standard})": "未采用 {name}（{standard}）",
    "rejected {name}": "未采用 {name}",
    "{name} ({standard}) failed": "{name}（{standard}）失败",
    "skipped {name}": "跳过 {name}",
    "from {name}": "来自 {name}",
    "it could not be read": "无法读取它",
    "it is a {kind} file, not a calibration": "它是 {kind} 文件，不是标定",
    "{shape} pixels, the frame has {frame}: another detector": "{shape} 像素，而这一帧是 {frame}：另一台探测器",
    "made for {pixel} µm pixels, the frame has {frame_pixel} µm: another detector":
        "为 {pixel} µm 像素制作，而这一帧是 {frame_pixel} µm：另一台探测器",
    "made for {shape} pixels, the frame has {frame}": "为 {shape} 像素制作，而这一帧是 {frame}",
    "no standard fitted": "没有拟合出标样",
    "ranked by the search": "按搜索的顺序",
    "standard images are ranked: GIMaP can fit the standard, same detector kind as the frame (file type and module series), 'giwaxs'/'waxs' in the name, 'final'/'redone' names, then closeness in time (before the frame preferred); a multi-module NeXus series is listed once (modules = number of files)":
        "标样图像的排序依据：GIMaP 能拟合这种标样，与这一帧是同类探测器（文件类型和模块系列），文件名含 'giwaxs'/'waxs'，文件名含 'final'/'redone'，然后是时间上的接近程度（优先选这一帧之前的）；多模块 NeXus 序列只列一次（modules = 文件数）",
    "from given": "由你给出",
    "from the notes": "来自笔记",
    "from the image header": "来自图像头信息",
    "from given (the image header has none)": "由你给出（图像头信息中没有）",
    "from the notes (the image header has none)": "来自笔记（图像头信息中没有）",
    "a saved calibration file (not re-checked against a standard image)": "已保存的标定文件（没有再用标样图像检查）",
    "pyFAI calibration (.poni)": "pyFAI 标定（.poni）",
    "detector image": "探测器图像",
    # -- the questions before the AI reads files or changes the set-up (any provider) ----------------
    "The AI wants to change something": "AI 想要修改设置",
    "The AI wants to look at your files": "AI 想要查看你的文件",
    "Allow the AI to do this?": "允许 AI 这样做吗？",
    # -- agent tooling (2026-10-06) ----------------------------------------------------------------
    "named by the user (--calibration, then the notes): tried before the instrument profile and the search":
        "由你指定（先 --calibration，再是笔记）：在仪器配置和自动搜索之前先试",
    "from the notes (the only calibrant they name)": "来自笔记（笔记中只提到这一种标样）",
    "none from the notes": "不从笔记中选取",
    "the notes name {n}: {names}": "笔记中提到 {n} 种：{names}",
    "compared ({name})": "逐一比较各标样（{name}）",
    "the notes name {noted}, the file name {from_name}": "笔记写的是 {noted}，文件名写的是 {from_name}",
    "from given (the image header agrees)": "由你给出（与图像头信息一致）",
    "given; the image header says {header} µm, the given value is used": "由你给出；图像头信息为 {header} µm，使用你给出的值",
    "did not keep the instrument profile '{name}'": "未沿用仪器配置“{name}”",
    "made for {pixel} µm pixels, the given pixel size is {given} µm": "为 {pixel} µm 像素制作，而你给出的像素尺寸是 {given} µm",
    "The calibration given was not used ({reason}); the geometry comes from {origin} instead.":
        "没有使用你给出的标定（{reason}）；几何改为来自 {origin}。",
    "Check that file (detector, energy, standard), or give the one used at the beamtime.":
        "请检查这个文件（探测器、能量、标样），或给出束线实验时实际使用的标定。",
    "Technique": "测量技术",
    "GIMaP's Auto detection could not classify this frame ({kind}), so the GIWAXS procedure ran.":
        "GIMaP 的自动识别无法判断这一帧的类型（{kind}），因此运行了 GIWAXS 流程。",
    "Give the technique (GIWAXS or GISAXS); the notes or the set-up (detector distance) say which.":
        "请给出测量技术（GIWAXS 或 GISAXS）；笔记或实验设置（探测器距离）会说明是哪一种。",
    "The calibration given was not used ({reason}), and no other calibration gave a good geometry.":
        "没有使用你给出的标定（{reason}），其他标定也都没有给出好的几何。",
    "Check that file (detector, energy, standard, pixel size), or give the one used at the beamtime.":
        "请检查这个文件（探测器、能量、标样、像素尺寸），或给出束线实验时实际使用的标定。",
    "a spike-like artefact (flat top, edges sharper than one bin): one detector row or column, a module edge or a gap among the averaged pixels rather than diffraction (look at the image at this q)":
        "类似尖峰的伪影（顶部平坦、边缘比一个 bin 还陡）：更可能来自探测器的一行或一列、模块边缘或参与平均的像素中的缝隙，而不是衍射（请在图像上查看这个 q 处）",
    "a spike-like artefact (flat top, edges sharper than one bin)": "类似尖峰的伪影（顶部平坦、边缘比一个 bin 还陡）",
}

__all__ = ["ASSISTANT_RUNTIME_ZH"]
