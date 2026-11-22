"""Chinese messages for pyFAI detector orientation."""

GEOMETRY_ORIENTATION_ZH = {
    "detector orientation": "探测器方向",
    "The .poni detector orientation must be 1, 2, 3 or 4.": ".poni 的探测器方向必须为 1、2、3 或 4。",
    "A flipped .poni detector orientation needs the detector dimensions. Supply max_shape or open the frame.":
        ".poni 的探测器坐标轴翻转需要图像尺寸。请在文件中给出 max_shape，或打开对应的图像。",
    "The detector dimensions must be positive.": "探测器的行数和列数必须为正数。",
    "The .poni Detector_config with orientation must be valid JSON.":
        ".poni 中包含方向的 Detector_config 必须采用有效的 JSON 格式。",
    "The .poni file gives conflicting detector orientations.": ".poni 文件中的探测器方向设置互相冲突。",
}
