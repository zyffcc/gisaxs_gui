"""Chinese for what GIMaP says about a detector tilt its flat geometry leaves out (a ``.poni``'s rotations): the
frame's message in Analyze (``analyze/application/flat_detector.py``) and the automatic analysis's open point and
decision (``assistant/application/pipeline_geometry.py``), shown through ``texts.message_text`` and
``guided_words.words``."""

FLAT_DETECTOR_ZH = {
    "The calibration describes a detector tilted by {tilt}°; GIMaP's geometry is a flat detector perpendicular to "
    "the beam, so the tilt is left out and |q| is off by up to {error} % on this frame.":
        "标定描述的探测器倾斜了 {tilt}°；GIMaP 的几何是垂直于光束的平板探测器，因此略去了这一倾斜，这一帧上 |q| "
        "最多偏差 {error} %。",
    "The calibration describes a detector tilted by {tilt}°; GIMaP's geometry is a flat detector perpendicular to "
    "the beam, so the tilt is left out and |q| is off by up to {error} % over the valid detector area.":
        "标定描述的探测器倾斜了 {tilt}°；GIMaP 的几何是垂直于光束的平板探测器，因此略去了这一倾斜，在探测器的有效"
        "区域内 |q| 最多偏差 {error} %。",
    "Peak positions carry this error, which grows away from the beam. GIMaP does not model a detector tilt: for "
    "exact q, reduce the frame with the full .poni geometry (e.g. pyFAI) or calibrate with an untilted detector.":
        "峰位带有这一误差，离光束越远越大。GIMaP 不模拟探测器倾斜：需要准确的 q 时，请用完整的 .poni 几何（例如 "
        "pyFAI）约化这一帧，或用不倾斜的探测器标定。",
    "GIMaP's geometry is a flat detector perpendicular to the beam; leaving the tilt out changes |q| by at most "
    "{error} % over the valid detector area (an open point above {limit_deg}° or {limit_error} %)":
        "GIMaP 的几何是垂直于光束的平板探测器；略去这一倾斜，在探测器的有效区域内 |q| 最多改变 {error} %（倾斜超过 "
        "{limit_deg}° 或误差超过 {limit_error} % 时列为待处理）",
    "{tilt}° left out": "略去 {tilt}°",
    "Detector tilt": "探测器倾斜",
    "tilt {tilt}° left out: |q| up to {error} % off": "略去了 {tilt}° 的倾斜：|q| 最多偏差 {error} %",
    "Geometry from “{name}”: {detail}; {tilt}.": "几何来自“{name}”：{detail}；{tilt}。",
}

FLAT_DETECTOR_ZH.update({
    'detector rotation about the beam': '探测器绕光束的旋转',
    "The calibration's rotation about the beam (Rot3 = {rotation}°) is left out. Rot3 alone does not change |q|, but it rotates χ and changes the directional assignment of qy/qz and in-plane/out-of-plane cuts.": '略去了标定中绕光束的旋转（Rot3 = {rotation}°）。单独的 Rot3 不改变 |q|，但会旋转 χ，改变 qy/qz 以及面内/面外切线的方向归属。',
    'Do not use the |q| error as an orientation error bound. Check direction-dependent results with the full .poni geometry (e.g. pyFAI); GIMaP has not applied this rotation.': '不要把 |q| 误差当作方向误差上限。请用完整的 .poni 几何（例如 pyFAI）核对依赖方向的结果；GIMaP 尚未应用这一旋转。',
    'Rot3 = {rotation}° left out': '略去 Rot3 = {rotation}°',
    'no detector-normal tilt': '探测器法线没有倾斜',
    'Rot3 {rotation}° left out: χ and qy/qz directions affected': '略去了 Rot3 {rotation}°：χ 和 qy/qz 方向受影响',
})

__all__ = ["FLAT_DETECTOR_ZH"]
