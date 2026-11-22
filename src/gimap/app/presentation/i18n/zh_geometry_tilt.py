"""Chinese text for tilted detector cuts, mirror filling and XRR geometry."""

ZH_GEOMETRY_TILT = {
    "The exit angle does not cross the beam-centre column.": "该出射角在光束中心列上没有交点。",
    "GISAXS cuts need the q maps of the frame.": "提取 GISAXS 切线需要该帧的 q 图。",
    "Detector rotations": "探测器旋转角",
    "Cuts are stored in q. Exit angles give αf for qz and 2θf on the sample horizon for qy. Pixel units give "
    "rows along the beam-centre column and columns along the calculated horizon.":
        "切线以 q 保存。使用出射角时，qz 对应 αf，qy 对应样品地平线上的 2θf。使用像素时，"
        "行坐标沿光束中心列读取，列坐标沿计算得到的地平线读取。",
    "pyFAI orientation": "pyFAI 探测器朝向",
    "Unsupported detector geometry in the file ({values}). Check the values shown before extraction.":
        "文件中的探测器几何参数不受支持（{values}）。提取前请核对当前显示的数值。",
    "Unsupported detector geometry in the last calibration ({values}). Check the values shown before extraction.":
        "上次标定中的探测器几何参数不受支持（{values}）。提取前请核对当前显示的数值。",
    "Up in the laboratory frame": "实验室坐标系中向上",
    "Down in the laboratory frame": "实验室坐标系中向下",
    "pyFAI rotations in degrees. Check these values for a tilted or rolled detector.":
        "pyFAI 的旋转角，单位为度。探测器倾斜或绕光束旋转时，请核对这些数值。",
    "The detector horizon changes by more than half a row across this frame. "
    "The Yoneda search and automatic horizontal cut use detector rows. "
    "Add a horizontal band in q to select a constant exit angle.":
        "样品地平线在整帧宽度内的高度变化超过半个像素行。Yoneda 带搜索和自动水平切线按探测器行取值。"
        "要选取固定出射角，请在 q 空间添加水平带。",
    "The horizon has no unique row at the beam column. The horizontal cut uses the middle row.":
        "地平线在光束中心列没有唯一的行坐标。水平切线取图像中间行。",
    "Set αi in the geometry for correct qz.": "请在几何设置中填写 αi，以计算正确的 qz。",
    "Assumes left–right symmetry of the sample. An unmeasured pixel takes the interpolated intensity "
    "at the opposite in-plane angle and the same exit angle. Every contributing pixel must be measured. "
    "The export records how many pixels were filled.":
        "假设样品的散射左右对称。缺失像素取相同出射角、相反面内角处的插值强度，且参与插值的各个像素都须有实测值。"
        "导出文件记录填充的像素数。",
}
