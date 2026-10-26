"""Chinese texts of Sputter ▸ Show in Analyze: one time window's raw frames, summed, in Analyze.

The menu entry and the problems of reading a window (``features/sputter``: ``presentation/window_menu.py``,
``domain/windows.py``), why Analyze cannot take the frames now (``features/analyze``: ``bindings/frame_import.py``) and
the notice of the hand-over (``app/sputter_handover.py``). Numbers, units, file and window names stay as they are.
"""

SPUTTER_HANDOVER_ZH = {
    # -- the Sputter page ------------------------------------------------------------------------------------------
    "Show in Analyze": "在分析中查看",
    "{folder} does not say which scan it was predicted from (gimap_request.json): its frames cannot be shown.":
        "{folder} 没有记录它是由哪个扫描预测的（gimap_request.json）：无法显示它的帧。",
    "The scan folder of this result is not there: {folder}": "这个结果的扫描文件夹不在了：{folder}",
    "{folder} has no binned windows (binned/chunk_*.json): its frames cannot be shown.":
        "{folder} 中没有分箱的时间窗（binned/chunk_*.json）：无法显示它的帧。",
    "Window {window} is not among the binned windows of {folder}.": "时间窗 {window} 不在 {folder} 的分箱时间窗中。",
    "The binned windows of {folder} do not name the frames of window {window} (raw_members).":
        "{folder} 的分箱时间窗没有列出时间窗 {window} 的帧（raw_members）。",
    "{missing} of the {count} frames of window {window} are not in {folder} (first missing: {name}).":
        "时间窗 {window} 的 {count} 帧中有 {missing} 帧不在 {folder} 中（第一个缺少的：{name}）。",
    # -- Analyze cannot take the frames now ------------------------------------------------------------------------
    "The automatic analysis works on the frame in Analyze: stop it first.": "自动分析正在处理分析页中的帧：请先停止它。",
    "A batch (Batch Export or a Series map) is running in Analyze: wait until it is done.":
        "分析页中有批处理（批量导出或序列热图）正在运行：请等它完成。",
    "Analyze is watching a folder: stop watching first.": "分析页正在监视文件夹：请先停止监视。",
    # -- the notice of the hand-over -------------------------------------------------------------------------------
    "Window {window} ({span}): the sum of its {count} frames is shown in Analyze.":
        "时间窗 {window}（{span}）：分析页显示的是它 {count} 帧之和。",
    "Window {window} ({span}): its one frame is shown in Analyze.": "时间窗 {window}（{span}）：分析页显示的是它唯一的一帧。",
    "Instrument profile “{name}”: distance {distance} mm, λ {wavelength} Å, as in Sputter.":
        "仪器配置“{name}”：距离 {distance} mm，λ {wavelength} Å，与溅射页相同。",
    "No instrument profile has the Sputter geometry: enter distance {distance} mm and λ {wavelength} Å "
    "(Geometry step ▸ Geometry…).":
        "没有仪器配置与溅射页的几何相同：请输入距离 {distance} mm 和 λ {wavelength} Å（几何步骤 ▸ 几何…）。",
    "The Sputter result does not record its distance and wavelength.": "这个溅射结果没有记录距离和波长。",
    "αi {alpha}° as in Sputter.": "αi {alpha}°，与溅射页相同。",
    "Beam centre ({x}, {y}) px as in Sputter.": "光束中心 ({x}, {y}) px，与溅射页相同。",
    "Beam centre: Analyze's own (the Sputter model estimated ({x}, {y}) px).":
        "光束中心：用分析页自己的（溅射模型估计为 ({x}, {y}) px）。",
    "Beam centre: Analyze's own (the Sputter model estimated its own).": "光束中心：用分析页自己的（溅射模型自行估计了光束中心）。",
    "The frames listed in Analyze before were replaced.": "分析页中原先列出的帧已被替换。",
    "Not shown in Analyze: {reason}": "没有在分析页中显示：{reason}",
}

__all__ = ["SPUTTER_HANDOVER_ZH"]
