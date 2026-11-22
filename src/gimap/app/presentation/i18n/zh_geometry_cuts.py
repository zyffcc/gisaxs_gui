"""Placement of the GISAXS vertical band and specular-rod warnings in the joint fit."""

ZH_GEOMETRY_CUTS = {
    "Detector rotations, pixel sizes, distance limits and preview overlays.": "探测器旋转角、像素尺寸、距离范围和预览叠加层。",
    "Fixed rot3": "固定 rot3",
    "rot3 fixed at {rot3}°": "rot3 固定为 {rot3}°",
    "Fit rot1 and rot2 with the centre and distance when three or more rings match. The fit holds rot3 fixed.":
        "匹配到至少三个环时，将 rot1、rot2 与中心和距离一起拟合。拟合时 rot3 保持固定。",
    "Rotation about the incident beam. Enter the angle measured from a GISAXS specular rod before calibration.":
        "绕入射束的转角。请在标定前输入从 GISAXS 镜面杆测得的角度。",
    "The fit refined rot1 and rot2 with the centre and distance. It held rot3 fixed at {rot3}°.":
        "拟合同时精修了 rot1、rot2、中心和距离。rot3 固定为 {rot3}°。",
    "The frame is too narrow for the default vertical band to clear the specular rod by 15 columns.":
        "图像宽度不足，默认垂直条带与镜面杆之间无法留出 15 列。",
    "Vertical cut I(qz) over columns {columns} (automatic).": "垂直切线 I(qz) 取自第 {columns} 列（自动选择）。",
    "The vertical band of {name} crosses the specular rod (qy = 0). The particle model excludes this "
    "scattering. Move the band beside the rod in Analyze and extract the cut again.":
        "{name} 的垂直条带经过镜面杆（qy = 0）。颗粒模型不包含这部分散射。请在分析工作区将条带移到镜面杆旁边并重新提取切线。",
    "The vertical band has no unique intersection with the sample horizon. Its in-plane q is undefined.":
        "垂直条带与样品地平线没有唯一交点，无法确定其面内 q。",
}
