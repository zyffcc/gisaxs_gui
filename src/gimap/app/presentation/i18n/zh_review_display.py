"""Chinese translations: first-user review fixes (display).

The header's detector distance against the calibration in use (``shared/geometry/header_check.py``: Analyze's Data
step and the automatic analysis), a calibrant frame in transmission (``assistant/application/ring_frames.py``) and a
calibration of another folder that contradicts the frame (``assistant/application/calibration_match.py``)."""

REVIEW_DISPLAY_ZH: dict[str, str] = {
    "Detector distance": "探测器距离",
    "The file header gives a detector distance of {header} mm, the calibration in use {used} mm ({percent} % apart).":
        "文件头给出的探测器距离是 {header} mm，正在使用的标定是 {used} mm（相差 {percent} %）。",
    "Check that the calibration belongs to this frame. A header can hold a nominal distance; then the calibration "
    "stands and this line can be ignored.":
        "请确认这份标定属于这一帧。文件头里的距离可能只是名义值；那样的话以标定为准，这一行可以忽略。",
    "a transmission frame of a calibrant or powder ({rings} complete rings about the beam centre)":
        "标样或粉末的透射帧（围绕光束中心有 {rings} 个完整的环）",
    "no incidence angle is known and the frame shows {rings} sharp rings concentric with the direct beam that run "
    "through every direction the detector covers, as a powder in transmission gives; a grazing-incidence frame is "
    "shadowed below the sample horizon":
        "入射角未知，而这一帧有 {rings} 个与直射束同心的锐环，在探测器覆盖的每个方向上都有，正如粉末透射测量给出的那样；"
        "掠入射帧在样品地平线以下会被样品遮挡",
    "This frame looks like a transmission measurement of a calibrant or powder ({rings} complete rings about the "
    "beam centre, no incidence angle), not like a GISAXS sample: its cuts were written, but no particle model was "
    "fitted.":
        "这一帧看起来是标样或粉末的透射测量（围绕光束中心有 {rings} 个完整的环，没有入射角），不像 GISAXS 样品："
        "切线已写出，但没有拟合颗粒模型。",
    "If it is a calibrant, use it in Tools ▸ Geometry Calibration. If it is a grazing-incidence frame, give αi or "
    "the technique and run again.":
        "如果它是标样，请在 工具 ▸ 几何标定 中使用它。如果它是掠入射帧，请给出 αi 或测量技术后再运行。",
    "the frame is a transmission frame of a calibrant or powder, not a GISAXS sample":
        "这一帧是标样或粉末的透射帧，不是 GISAXS 样品",
    "{name} lies in another folder and was made for a detector of {shape} pixels, the frame has {frame_shape}: "
    "another experiment's calibration":
        "{name} 在另一个文件夹里，是为 {shape} 像素的探测器做的，而这一帧是 {frame_shape}：是别的实验的标定",
    "{name} was found by the search in another folder than the frame's and was made at {energy} keV, while the "
    "frame was measured at {frame_energy} keV: its distance and beam centre are used with the frame's energy.":
        "{name} 是搜索在帧所在文件夹之外找到的，是在 {energy} keV 下做的，而这一帧是在 {frame_energy} keV 下测量的："
        "采用了它的距离和光束中心，能量用这一帧的。",
    "{name}: made at another energy than the frame's": "{name}：是在与这一帧不同的能量下做的",
}
