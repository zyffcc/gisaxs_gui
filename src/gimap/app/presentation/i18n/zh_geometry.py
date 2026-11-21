"""Chinese for the detector geometry: the tilt fit and the energy source of Geometry Calibration
(``calibration/presentation/bindings/energy_source.py``, ``result_preview.py``, ``domain/tilt_fit.py``), the
Detector geometry dialog of Analyze with its wavelength source and tilt fields (``analyze/presentation/geometry_dialog.py``),
Load / Export .poni (``bindings/geometry_files.py``, ``application/geometry_files.py``), the transmission technique
(``application/technique.py``), the profile of another detector (``application/geometry_resolution.py``), opening
frames while others are listed (``bindings/frame_import.py``, ``application/frame_import.py``) and the geometry lines
of Batch Export (``application/batch_geometry.py``)."""

GEOMETRY_ZH = {
    # -- Geometry Calibration: energy and tilt ------------------------------------------------
    "Required": "必填",
    "Refine rot1, rot2": "精修 rot1、rot2",
    "Fit the detector tilt (pyFAI's rot1 and rot2) with the centre and the distance when three or more rings "
    "are matched; rot3 (a turn about the beam) does not change the rings and stays 0":
        "匹配到三个或更多环时，与中心和距离一起拟合探测器倾斜（pyFAI 的 rot1 和 rot2）；rot3（绕光束的转动）不改变环，保持为 0",
    "Detector tilt:": "探测器倾斜：",
    "From the file header.": "来自文件头。",
    "From the companion NXS file.": "来自配套的 NXS 文件。",
    "Entered by you.": "由你输入。",
    "From the imported calibration.": "来自导入的标定。",
    "Not in the file: enter the X-ray energy (it sets the wavelength of the calibration).":
        "文件中没有：请输入 X 射线能量（它决定标定的波长）。",
    "Not in this file: the energy entered for the image before is used — check it.":
        "此文件中没有：沿用为上一张图像输入的能量——请核对。",
    "not refined": "未精修",
    "Tick Refine rot1, rot2 and calibrate again; the tilt needs three or more matched rings.":
        "勾选“精修 rot1、rot2”后重新标定；倾斜需要三个或更多匹配的环。",
    "pyFAI's rot1 and rot2 (rot3 = {rot3}°), fitted with the centre and the distance.":
        "pyFAI 的 rot1 和 rot2（rot3 = {rot3}°），与中心和距离一起拟合。",
    "Ring points: RMS {tilted} px tilted, {flat} px without the tilt.":
        "环上的点：倾斜几何的 RMS 为 {tilted} px，不计倾斜时为 {flat} px。",
    "1σ: {rot1}°, {rot2}° ({method}).": "1σ：{rot1}°、{rot2}°（{method}）。",
    "pyFAI geometry (*.poni)": "pyFAI 几何文件 (*.poni)",
    "The detector tilt was not refined: it needs at least three matched rings.": "未精修探测器倾斜：至少需要三个匹配的环。",
    "The detector tilt was not refined: too few ring points.": "未精修探测器倾斜：环上的点太少。",
    "Refining detector tilt": "精修探测器倾斜",
    "Image loaded{notes}. Enter the X-ray energy, then click Auto Calibration.":
        "已载入图像{notes}。请输入 X 射线能量，然后点击“自动标定”。",
    # -- Analyze: Detector geometry dialog ------------------------------------------------------
    "Energy": "能量",
    "Detector tilt rot1": "探测器倾斜 rot1",
    "Detector tilt rot2": "探测器倾斜 rot2",
    "Rotation about the beam rot3": "绕光束的转动 rot3",
    "X-ray wavelength and energy: E = hc/λ (12.398 keV·Å); typing one sets the other":
        "X 射线波长和能量：E = hc/λ（12.398 keV·Å）；输入其一即得另一个",
    "pyFAI's detector rotations (as in a .poni): rot1 about the vertical axis, rot2 about the horizontal axis, rot3 "
    "about the beam; 0 is a detector perpendicular to the beam. The beam centre stays the direct beam and the "
    "distance the distance to it":
        "pyFAI 的探测器转角（与 .poni 相同）：rot1 绕竖直轴，rot2 绕水平轴，rot3 绕光束；全为 0 即探测器垂直于光束。"
        "光束中心仍是直射束位置，距离仍是到直射束的距离",
    "Read a pyFAI .poni (distance, PONI, rot1–rot3, pixel sizes, wavelength) into these fields":
        "把 pyFAI .poni（距离、PONI、rot1–rot3、像素尺寸、波长）读入这些字段",
    "Write these values as a pyFAI .poni (version 2.1) for pyFAI, GIXSGUI and other programs":
        "把这些值写成 pyFAI .poni（2.1 版），供 pyFAI、GIXSGUI 等程序使用",
    "Load .poni…": "载入 .poni…",
    "Export .poni…": "导出 .poni…",
    "Wavelength from the file header.": "波长来自文件头。",
    "Wavelength from {name}.": "波长来自 {name}。",
    "Wavelength of profile “{name}”.": "波长来自配置“{name}”。",
    "Wavelength entered by you.": "波长由你输入。",
    "No wavelength or energy is known for this frame: enter one (q scales with it).":
        "这一帧没有已知的波长或能量：请输入一个（q 随它缩放）。",
    "Enter the wavelength or the energy first: the file states none.": "请先输入波长或能量：文件中没有给出。",
    # -- Analyze: Load / Export .poni ------------------------------------------------------------
    "Use the geometry of a pyFAI .poni file (distance, beam, tilt, pixel size, wavelength) for this detector":
        "为这台探测器使用 pyFAI .poni 文件的几何（距离、光束、倾斜、像素尺寸、波长）",
    "Write the geometry in use as a pyFAI .poni file": "把正在使用的几何写成 pyFAI .poni 文件",
    "Load .poni": "载入 .poni",
    "Export .poni": "导出 .poni",
    "Could not read the .poni file: {error}": "无法读取 .poni 文件：{error}",
    "{problem}\n\nUse this geometry for the frame shown anyway?": "{problem}\n\n仍然把这个几何用于当前显示的帧吗？",
    "{name} not used: {problem}": "未使用 {name}：{problem}",
    "Read {name}: check the values, then Save Profile.": "已读取 {name}：请核对数值，然后点“保存配置”。",
    "Geometry from {name} saved as instrument profile “{profile}” and used for this detector.":
        "{name} 的几何已保存为仪器配置“{profile}”，并用于这台探测器。",
    "Geometry written to {name}.": "几何已写入 {name}。",
    "There is no geometry to export: load a .poni, enter the geometry or calibrate.":
        "没有可导出的几何：请载入 .poni、输入几何或标定。",
    "Could not write the .poni file: {error}": "无法写入 .poni 文件：{error}",
    "The .poni file is for {poni_shape} frames; this frame is {shape}.": "这个 .poni 文件用于 {poni_shape} 的帧；这一帧是 {shape}。",
    "The .poni file has {poni_um} µm pixels; the frame header says {frame_um} µm.":
        "这个 .poni 文件的像素是 {poni_um} µm；帧的文件头写的是 {frame_um} µm。",
    "The .poni file states no wavelength: enter the wavelength or the energy.": "这个 .poni 文件没有给出波长：请输入波长或能量。",
    "The .poni file lacks Distance, Poni1, Poni2 or the pixel size.": ".poni 文件缺少 Distance、Poni1、Poni2 或像素尺寸。",
    "{name} beside the frame": "帧旁边的 {name}",
    # -- Analyze: a profile of another detector ----------------------------------------------------
    "Profile “{name}” is for {profile_shape} frames; this {shape} frame is from another detector, so it is not used.":
        "配置“{name}”用于 {profile_shape} 的帧；这一 {shape} 的帧来自另一台探测器，所以不使用它。",
    "Use Profile Anyway": "仍然使用该配置",
    "Use the profile picked for frames of another size on these frames too": "把为另一尺寸的帧选的配置也用于这些帧",
    "Use “{name}” Anyway": "仍然使用“{name}”",
    "Profile “{name}” is used for {shape} frames too, as you asked.": "按你的要求，配置“{name}”也用于 {shape} 的帧。",
    # -- Analyze: the transmission technique ---------------------------------------------------------
    "SAXS/WAXS (transmission)": "SAXS/WAXS（透射）",
    "SAXS/WAXS in transmission: full rings about the beam, I(q) and I(χ) over 360°, no horizon or Yoneda band (a "
    "calibrant, or a sample in transmission)":
        "透射 SAXS/WAXS：绕光束的完整圆环，I(q) 和 I(χ) 覆盖 360°，没有样品地平线和 Yoneda 带（标样，或透射测量的样品）",
    "Decide from the largest scattering angle on the detector; without an incidence angle, a calibrant or complete "
    "rings mean transmission":
        "按探测器上最大的散射角判断；没有入射角时，标样或完整的圆环意味着透射",
    "Taken as SAXS/WAXS in transmission: the file name names a calibration standard and no incidence angle is known.":
        "按透射 SAXS/WAXS 处理：文件名指明了标样，且没有已知的入射角。",
    "Taken as SAXS/WAXS in transmission: {n} complete rings about the direct beam and no incidence angle is known.":
        "按透射 SAXS/WAXS 处理：绕直射束有 {n} 个完整的环，且没有已知的入射角。",
    "No valid pixels.": "没有有效像素。",
    "technique by angle (transmission for a calibrant)": "按角度判断技术（标样按透射）",
    # -- Analyze: opening frames while others are listed ------------------------------------------
    "Open Frames": "打开帧",
    "{count} frames are listed. Replace them with the frames opened now, or add these to the list?":
        "列表中已有 {count} 帧。用现在打开的帧替换它们，还是把这些帧加到列表中？",
    "Add to List": "加到列表",
    "These frames are {shape} ({detector}); the frames listed are {listed}. Frames of another detector are not added "
    "to this list: choose Replace, or Analyze ▸ Clear first.":
        "这些帧是 {shape}（{detector}）；列表中的帧是 {listed}。另一台探测器的帧不会加入这个列表：请选择“替换”，"
        "或先点 分析 ▸ 清除。",
    "detector not named": "未写明探测器",
    # -- Batch Export: the geometry of each folder -----------------------------------------------------
    "{n} frame(s) in {folder}: geometry from {poni} beside them.": "{folder} 中的 {n} 帧：几何来自它们旁边的 {poni}。",
    "{n} frame(s) in {folder}: {source}, as the frame on screen.": "{folder} 中的 {n} 帧：{source}，与当前显示的帧相同。",
    "{n} frame(s) in {folder}: the profile picked or the one that fits their detector and size, read frame by frame; "
    "a frame without one gives no q curves.":
        "{folder} 中的 {n} 帧：逐帧使用所选配置或与其探测器和尺寸相符的配置；没有配置的帧不产生 q 曲线。",
    "No geometry: q curves cannot be made. Load a .poni, enter the geometry or calibrate first.":
        "没有几何：无法得到 q 曲线。请先载入 .poni、输入几何或标定。",
    "no geometry": "无几何",
    "geometry in use": "正在使用的几何",
}

__all__ = ["GEOMETRY_ZH"]
