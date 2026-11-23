"""Chinese texts of the Sputter workspace: the shell, Details (the geometry and where each value comes from, More
options), the request, and Details ▸ Windows, Report and Log.

Keys are the exact English of ``features/sputter`` (its templates keep their ``{fields}``). The main screen, Analyze
and Static Frames are in ``zh_sputter_client.py``, the live view in ``zh_sputter_live.py``.
"""

SPUTTER_ZH = {
    # -- the shell -------------------------------------------------------------------------------------------
    "Sputter": "溅射",
    "In-situ sputter series: grain size, height and spacing over time, fitted window by window":
        "原位溅射序列：逐时间窗拟合的晶粒尺寸、高度和间距随时间的变化",
    "Sputter analysis": "溅射分析",
    # -- the page: fixed texts -------------------------------------------------------------------------------
    "Folder with the CBF frames of one scan": "一个扫描的 CBF 帧所在的文件夹",
    "Sample and geometry": "样品与几何",
    "Metal": "金属",
    "X-rays": "X 射线",
    "Incidence αi (°)": "入射角 αi（°）",
    "Beam centre (px)": "光束中心（px）",
    "col": "列",
    "row": "行",
    "required": "必填",
    "Beam centre in pixels as Analyze shows it (pixel corners on whole numbers, row 0 at the top). Empty: estimated "
    "from the frames.": "光束中心，像素坐标与分析（Analyze）中显示的相同（像素角在整数上，第 0 行在上）。留空：由帧估计。",
    "Save Geometry as Preset…": "把几何存为预设…",
    "Keep this geometry for every scan of this detector and flat field in a date range (a beamtime): such scans are "
    "filled in from it.": "为这台探测器和这个平场在某个日期范围（一次束线时间）内的每个扫描保存这套几何：这些扫描会按它填写。",
    "More options": "更多选项",
    "Polymer": "聚合物",
    "Sample roll (°)": "样品滚转角（°）",
    "Mask": "掩膜文件",
    "optional detector mask (.npy)": "可选的探测器掩膜（.npy）",
    "optional": "可选填",
    "end of the scan": "扫描结束时",
    "Save results in": "结果保存在",
    "Windows": "时间窗",
    # -- the folder, the sources of the values ------------------------------------------------------------------
    "Choose the Scan Folder": "选择扫描文件夹",
    "Reading the header of the first frame…": "正在读取第一帧的文件头…",
    "exposure {exposure} s": "曝光 {exposure} s",
    " every {period} s": "，每 {period} s 一帧",
    "No CBF frames in this folder.": "这个文件夹里没有 CBF 帧。",
    "from the file header": "来自文件头",
    "from the scan's NeXus file": "来自该扫描的 NeXus 文件",
    "as last used for this folder": "沿用这个文件夹上次的值",
    "from preset {name}": "来自预设 {name}",
    "last used — please check": "上次使用的值——请核对",
    "not in the files: 0 assumed — please check": "文件中没有：假定为 0——请核对",
    "entered": "已填写",
    "Choose a scan folder whose frames name the detector's serial number first: a preset is found again by the "
    "serial number, the flat field and the date.": "请先选择一个帧里写有探测器序列号的扫描文件夹：预设按序列号、平场和日期找回。",
    "Geometry preset {name} saved: scans of this detector from {start} to {end} are filled in from it.":
        "几何预设 {name} 已保存：这台探测器从 {start} 到 {end} 的扫描会按它填写。",
    "Save Geometry as Preset": "把几何存为预设",
    "The geometry on the page (X-rays, distance, incidence, beam centre, sample roll) is used for every scan of this "
    "detector and flat field taken in these days.":
        "页面上的几何（X 射线、距离、入射角、光束中心、样品滚转角）用于这台探测器和这个平场在这些日期内的每个扫描。",
    "for example the beamtime": "例如束线时间",
    "Detector serial number": "探测器序列号",
    "Flat field": "平场",
    "First day": "第一天",
    "Last day": "最后一天",
    "Choose a Detector Mask": "选择探测器掩膜",
    "Save Results In": "结果保存位置",
    # -- the request -------------------------------------------------------------------------------------------
    "Choose the scan folder that holds the CBF frames.": "请选择存放 CBF 帧的扫描文件夹。",
    "Give the photon energy or the wavelength.": "请给出光子能量或波长。",
    "Give the sample-detector distance and the incidence angle (both positive).": "请给出样品-探测器距离和入射角（都为正）。",
    "Give both beam-centre coordinates, or neither (automatic estimate).": "请给出光束中心的两个坐标，或都不给（自动估计）。",
    "Metal must be Ag, Au or Cu.": "金属必须是 Ag、Au 或 Cu。",
    # -- the run -----------------------------------------------------------------------------------------------
    "Reading the frames": "正在读帧",
    "Done": "完成",
    "Done: {windows} time windows.": "完成：{windows} 个时间窗。",
    # -- the result line, the table ------------------------------------------------------------------------------
    "Nominal thickness (nm)": "名义厚度（nm）",
    "Component": "分量",
    # -- the trend ---------------------------------------------------------------------------------------------
    "time (s)": "时间（s）",
    "Plotting is unavailable: {error}": "无法绘图：{error}",
    # -- the sketch ----------------------------------------------------------------------------------------------
    "islands": "岛",
    "film": "膜",
    "side view (cut)": "侧视（截面）",
    "polymer (thickness not to scale)": "聚合物（厚度不按比例）",
    "Metal inside the polymer: not resolved by these data, not drawn": "聚合物里的金属：这些数据分辨不出，没有画",
    "Metal inside the polymer: not in the model, not drawn": "聚合物里的金属：模型里没有，没有画",
    "top view": "俯视",
    "not measured": "未测得",
    # -- Details: the tabs about the result shown (views/sputter_page_view.py, result_tabs.py) ------------------------
    "Analyze a scan or open a result": "请分析一个扫描，或打开一个结果",
    "Logs": "日志",
    "Each scan and geometry gets its own folder in here; the same request shows the result it has.":
        "每个扫描和几何在这里有自己的文件夹；同一个请求会显示它已有的结果。",
    "Fit deviation": "拟合偏差",
    "Solution": "解",
    "no solution: {reason}": "无解：{reason}",
    "Network used: {family}": "所用网络：{family}",
    "Networks tried (fit deviation): {tried}": "试过的网络（拟合偏差）：{tried}",
    "Mode: {mode}": "模式：{mode}",
    "Status: {status}": "状态：{status}",
    "Run time: {time}": "运行时间：{time}",
    "{minutes} min": "{minutes} 分钟",
    "This result has no report (REPORT.md).": "这个结果没有报告（REPORT.md）。",
    "This result has no log files.": "这个结果没有日志文件。",
}

__all__ = ["SPUTTER_ZH"]
