"""Chinese for NeXus / HDF5 sources: what a file or a set of files consists of, and why one cannot be read.

The keys are the exact English templates: those of ``shared/detector_io/nexus_layout.py`` (``SUMMARY_TEMPLATES``,
``WARNING_TEMPLATES``, ``ERROR_TEMPLATES``), and Analyze's own sentences about a source (the Data step, the
file list, the frame stepper, the folder watch). File names, dataset paths, module labels (m07) and numbers are
filled in afterwards and are never translated.
"""

NEXUS_ZH = {
    # -- what a source consists of (NexusLayout.summary) ---------------------------------------------------
    "{modules} module files · {frames} frames · stitched": "{modules} 个模块文件 · {frames} 帧 · 已拼接",
    "{modules} module files · 1 frame · stitched": "{modules} 个模块文件 · 1 帧 · 已拼接",
    "master + {files} data files · {frames} frames": "主文件 + {files} 个数据文件 · {frames} 帧",
    "master + {files} data files · 1 frame": "主文件 + {files} 个数据文件 · 1 帧",
    "master + 1 data file · {frames} frames": "主文件 + 1 个数据文件 · {frames} 帧",
    "master + 1 data file · 1 frame": "主文件 + 1 个数据文件 · 1 帧",
    "1 file · {frames} frames": "1 个文件 · {frames} 帧",
    "1 file · 1 frame": "1 个文件 · 1 帧",
    "no detector image": "没有探测器图像",
    # -- warnings of a layout (NexusLayout.warnings) ---------------------------------------------------------
    "Module files hold different numbers of frames ({detail}; the others {full}): frames after frame {frames} "
    "are left out": "各模块文件的帧数不同（{detail}；其余为 {full}）：第 {frames} 帧之后的帧不使用",
    "Module file {names} is missing: that part of the detector is left out":
        "缺少模块文件 {names}：探测器的这一部分没有数据",
    "Module files {names} are missing: those parts of the detector are left out":
        "缺少模块文件 {names}：探测器的这些部分没有数据",
    "Module file {names} cannot be read yet: that part of the detector is left out":
        "模块文件 {names} 还无法读取：探测器的这一部分没有数据",
    "Module file {names} was left open for writing and has not changed for a while: the program that wrote it seems "
    "to have ended without closing it, and that part of the detector is left out. Running h5clear -s on a copy of it "
    "makes readable what was stored in it":
        "模块文件 {names} 仍标记为“正在写入”，但已有一段时间没有变化：写入它的程序似乎没有关闭文件就结束了，"
        "探测器的这一部分因此没有数据。对它的副本运行 h5clear -s，就能读出已经存入的内容",
    "Module files {names} were left open for writing and have not changed for a while: the programs that wrote them "
    "seem to have ended without closing them, and those parts of the detector are left out. Running h5clear -s on "
    "copies of them makes readable what was stored in them":
        "模块文件 {names} 仍标记为“正在写入”，但已有一段时间没有变化：写入它们的程序似乎没有关闭文件就结束了，"
        "探测器的这些部分因此没有数据。对它们的副本运行 h5clear -s，就能读出已经存入的内容",
    "{first} and {second} are named like modules of one detector but hold the same module position: each file is "
    "read on its own": "{first} 和 {second} 的文件名像同一个探测器的模块，但记录的模块位置相同：每个文件单独读取",
    "{name} is named like a detector module of this scan but does not fit the set (dataset, module shape or module "
    "position): each file is read on its own":
        "{name} 的文件名像这次扫描的探测器模块，但与这组文件不符（数据集、模块形状或模块位置）：每个文件单独读取",
    "{count} image datasets in this file: {dataset} is shown, not {others} (Format Converter and Geometry Calibration "
    "let you choose the dataset)":
        "此文件中有 {count} 个图像数据集：显示的是 {dataset}，不是 {others}（在“格式转换”和“几何标定”中可以选择数据集）",
    "{dataset} is named like a mask or a correction, not like a detector image: it is read because this file holds "
    "no other image": "{dataset} 的名字像掩膜或校正数据，不像探测器图像：此文件中没有别的图像，所以读取它",
    "Dataset {wanted} is not in this file: {dataset} is read instead": "此文件中没有数据集 {wanted}：改为读取 {dataset}",
    "Data file {names} is missing: frames after frame {frames} cannot be read":
        "缺少数据文件 {names}：第 {frames} 帧之后的帧无法读取",
    "Data files {names} are missing: frames after frame {frames} cannot be read":
        "缺少多个数据文件（{names}）：第 {frames} 帧之后的帧无法读取",
    "A {grid} grid of frames is read in storage order as {frames} frames": "{grid} 的帧网格按存储顺序读作 {frames} 帧",
    "{name} is still being written: it is read again when it changes": "{name} 还在写入：它变化后会重新读取",
    "{name} was not closed by the program that wrote it and has not changed for a while: it is read as it is, and "
    "its last frame may be incomplete": "写入 {name} 的程序没有关闭它，而它已有一段时间没有变化：按现状读取，最后一帧可能不完整",
    "{count} module files, but {other} in the same folder has {others}: {names} may be missing here (that part of "
    "the detector is then left out)":
        "这里有 {count} 个模块文件，而同一文件夹中的 {other} 有 {others} 个：这里可能缺少 {names}（探测器的这一部分就没有数据）",
    "{dataset} is read as {frames} frames of {rows} × {columns} pixels — if these are {columns} frames stored last, "
    "this file is not supported yet":
        "{dataset} 被读作 {frames} 帧，每帧 {rows} × {columns} 像素 —— 如果它其实是 {columns} 帧、帧存在最后一维，目前还不支持这种文件",
    # -- why a frame cannot be read (ERROR_TEMPLATES, and the two sentences of Analyze's frame reader) ---------
    "{name} is being written by another program and cannot be read until it is closed.":
        "{name} 正被另一个程序写入，在它被关闭之前无法读取。",
    "{name} was left open for writing and has not changed for a while: the program that wrote it seems to have "
    "ended without closing it. It cannot be read as it is; running h5clear -s on a copy of it makes readable what "
    "was stored in it.":
        "{name} 仍标记为“正在写入”，但已有一段时间没有变化：写入它的程序似乎没有关闭文件就结束了。"
        "它现在这样无法读取；对它的副本运行 h5clear -s，就能读出已经存入的内容。",
    "{name} holds no detector image.": "{name} 中没有探测器图像。",
    "frame {frame} of {frames}": "第 {frame} 帧，共 {frames} 帧",
    "Dataset {dataset} cannot be read as detector frames: its shape is {shape} and its type {dtype}.":
        "数据集 {dataset} 不能作为探测器帧读取：它的形状是 {shape}，类型是 {dtype}。",
    "{dataset} is a group of the file, not a dataset.": "{dataset} 是文件中的一个组，不是数据集。",
    "{name} holds no frames yet.": "{name} 中还没有帧。",
    "{name} is compressed with a filter this installation cannot read: install the Python package hdf5plugin.":
        "{name} 使用了当前安装无法读取的压缩过滤器：请安装 Python 包 hdf5plugin。",
    # -- the Data step: the source shown ------------------------------------------------------------------------
    "Image as stored in the file": "图像按文件中存储的方向显示",
    "P03 Lambda: transposed and flipped (rotated by 90°)": "P03 Lambda：转置并上下翻转（即旋转 90°）",
    "Frames listed: {n}": "列表中的帧：{n}",
    "Still growing: frame {n} is the newest complete one": "还在增长：第 {n} 帧是目前最新的完整帧",
    "{text}.": "{text}。",
    "of {n}": "共 {n} 帧",
    "The frame shown (Page Up / Page Down show the previous / next one)":
        "当前显示的帧（Page Up / Page Down 显示上一帧 / 下一帧）",
    "Open or drop detector frames (CBF, NeXus/HDF5, TIFF, EDF) or a folder. Files that belong together open as one "
    "row: module files (…_m01.nxs … _m11.nxs) are stitched, a master is read with its data files. Every frame in "
    "them is a frame of the series.":
        "打开或拖入探测器帧（CBF、NeXus/HDF5、TIFF、EDF）或文件夹。属于同一组的文件只占一行：模块文件"
        "（…_m01.nxs … _m11.nxs）会被拼接，主文件会连同它的数据文件一起读取。其中的每一帧都是序列中的一帧。",
    "No new detector frames ({formats}) found.": "没有找到新的探测器帧（{formats}）。",
    "{folder} holds no frames itself: its subfolders {names} were opened":
        "{folder} 本身没有帧：已打开它的这些子文件夹：{names}",
    "{folder} holds no frames itself: its subfolder {names} was opened": "{folder} 本身没有帧：已打开它的子文件夹 {names}",
    # -- the file list and the frame stepper --------------------------------------------------------------------
    "{name} · {frames} frames": "{name} · {frames} 帧",
    "{files} and {n} more": "{files}，另有 {n} 个",
    "frame {i} / {n}": "第 {i} / {n} 帧",
    "Frame {i} of {n} of {name} — Page Up / Page Down show the previous / next frame":
        "{name} 的第 {i} 帧，共 {n} 帧 — Page Up / Page Down 显示上一帧 / 下一帧",
    "Previous frame (Page Up)": "上一帧（Page Up）",
    "Next frame (Page Down)": "下一帧（Page Down）",
    # -- the folder watch ------------------------------------------------------------------------------------------
    "{name}: {n} new frames": "{name}：新增 {n} 帧",
    "{name}: 1 new frame": "{name}：新增 1 帧",
    "Watching {folder}: {reason}": "正在监视 {folder}：{reason}",
    "{name} is not exported yet: {reason}": "{name} 还没有导出：{reason}",
    "{name} is not exported yet: its module files are still arriving ({n} so far)":
        "{name} 还没有导出：它的模块文件还在陆续到达（目前 {n} 个）",
    "{name} is not exported yet: {n} of the {total} module files that the other scans in this folder have are here":
        "{name} 还没有导出：此文件夹中其他扫描有 {total} 个模块文件，它目前只有 {n} 个",
    # -- the formats named where data are opened ---------------------------------------------------------------------
    "Open or drop detector files (CBF, NeXus/HDF5, TIFF, EDF) to start": "打开或拖入探测器文件（CBF、NeXus/HDF5、TIFF、EDF）开始",
    "Open detector frames (CBF, NeXus/HDF5, TIFF, EDF) — Ctrl+O. You can also drop files or folders here.":
        "打开探测器帧（CBF、NeXus/HDF5、TIFF、EDF）—— Ctrl+O。也可以把文件或文件夹拖到这里。",
    "Open detector frames (CBF, NeXus/HDF5, TIFF, EDF) in Analyze": "在分析中打开探测器帧（CBF、NeXus/HDF5、TIFF、EDF）",
    "Detector frames (CBF, NeXus/HDF5, TIFF, EDF), opened in Analyze — Ctrl+O":
        "探测器帧（CBF、NeXus/HDF5、TIFF、EDF），在分析中打开 — Ctrl+O",
    "Nothing opened: drop detector frames (CBF, NeXus/HDF5, TIFF, EDF), a folder of them or a project (.gimap)":
        "没有打开任何内容：请拖入探测器帧（CBF、NeXus/HDF5、TIFF、EDF）、包含它们的文件夹或项目（.gimap）",
}

__all__ = ["NEXUS_ZH"]
