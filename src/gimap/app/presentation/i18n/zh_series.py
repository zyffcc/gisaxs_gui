"""Chinese for series work in Analyze: the Series map, Batch Export and Send Series to Fitting."""

SERIES_ZH = {
    # -- Batch Export / Send Series to Fitting: the halves of the signed GISAXS cut a fit takes -----------------
    # (the dialog: analyze/presentation/views/batch_export_view.py and batch_texts.py; "Halves of the cut",
    # "Mean of both halves", "q > 0 half" and "q < 0 half" are in the tables of Analyze and Fitting)
    "Curve for Fitting": "拟合用曲线",
    "A fit takes one curve on |qy|: the mean of the two halves of the horizontal cut, or one half — never both "
    "halves together. Fitting ▸ In-situ series starts with this choice.":
        "拟合只用 |qy| 上的一条曲线：水平切线两半的平均，或其中一半——不会把两半放在一起拟合。拟合 ▸ 原位序列 从这个选择开始。",
    "A fit takes the mean of both halves of each cut. The file written for Fitting keeps both halves, with signed "
    "qy, so the choice can still be changed in Fitting.":
        "拟合用每条切线两半的平均。写给拟合的文件保留两半（qy 带符号），所以在拟合里还可以改这个选择。",
    "A fit takes the q > 0 half of each cut. The file written for Fitting keeps both halves, with signed qy, so "
    "the choice can still be changed in Fitting.":
        "拟合用每条切线 q > 0 的一半。写给拟合的文件保留两半（qy 带符号），所以在拟合里还可以改这个选择。",
    "A fit takes the q < 0 half of each cut. The file written for Fitting keeps both halves, with signed qy, so "
    "the choice can still be changed in Fitting.":
        "拟合用每条切线 q < 0 的一半。写给拟合的文件保留两半（qy 带符号），所以在拟合里还可以改这个选择。",
    # … and of a batch that fits a model without saving the curve for Fitting: no file to speak of
    "A fit takes the mean of both halves of each cut.": "拟合用每条切线两半的平均。",
    "A fit takes the q > 0 half of each cut.": "拟合用每条切线 q > 0 的一半。",
    "A fit takes the q < 0 half of each cut.": "拟合用每条切线 q < 0 的一半。",
    "Used when the curve for Fitting is saved (Data, above) or a particle model is fitted (below).":
        "保存拟合用曲线（上方“数据”）或拟合颗粒模型（下方）时才用到。",
    "Each frame: the model is fitted to the mean of both halves of the horizontal cut (Halves of the cut, above); "
    "up to half a minute per frame.":
        "每帧：对水平切线两半的平均拟合模型（见上方“切线的两半”）；每帧最多约半分钟。",
    "Each frame: the model is fitted to the q > 0 half of the horizontal cut (Halves of the cut, above); up to "
    "half a minute per frame.":
        "每帧：对水平切线 q > 0 的一半拟合模型（见上方“切线的两半”）；每帧最多约半分钟。",
    "Each frame: the model is fitted to the q < 0 half of the horizontal cut (Halves of the cut, above); up to "
    "half a minute per frame.":
        "每帧：对水平切线 q < 0 的一半拟合模型（见上方“切线的两半”）；每帧最多约半分钟。",
    "Each frame: the curve for Fitting (q in Å⁻¹, I, σ, pixels) → {file}. The folder then opens in Fitting ▸ "
    "In-situ series.":
        "每帧：拟合用曲线（q 单位 Å⁻¹、I、σ、像素数）→ {file}。随后在 拟合 ▸ 原位序列 中打开这个文件夹。",
    "Fitting ▸ In-situ series fits particle form factors, made for the horizontal cut of a GISAXS frame: they do not "
    "describe diffraction peaks. To follow the peaks of this series use Batch Export ▸ Peaks of the ring regions.":
        "拟合 ▸ 原位序列 拟合的是颗粒形状因子，用于 GISAXS 帧的水平切线：它们不能描述衍射峰。"
        "要跟踪这个序列中的峰，请用“批量导出 ▸ 环区域中的峰”。",
    # What a trial fit was made of (Try on This Frame), composed with trf.
    "Fitted: {halves}, {points} points": "拟合所用：{halves}，{points} 个点",
    "Fitted: {halves}, {points} points ({merged} neighbouring points merged into one)":
        "拟合所用：{halves}，{points} 个点（每 {merged} 个相邻点合并为一个）",
    "Fitted: {points} points": "拟合所用：{points} 个点",
    "{reason} in this frame: choose the mean of both halves or the other half (Halves of the cut, above).":
        "这一帧 {reason}：请选择两半取平均或另一半（见上方“切线的两半”）。",
    # How a batch with a model fit ended (status line, toast, the panel), composed with trf.
    "model fit: {fitted} of {frames} frames fitted": "模型拟合：{frames} 帧中拟合了 {fitted} 帧",
    "model fit: {fitted} of {frames} frames fitted — {why}": "模型拟合：{frames} 帧中拟合了 {fitted} 帧 —— {why}",
    "{n} frame(s): {reason}": "{n} 帧：{reason}",
    "{n} frame(s) on one half alone": "{n} 帧只用了一半",
    "{first}; {second}": "{first}；{second}",
    # Why a frame has no model fit. The application and the fit say these in English (they go into the table and
    # the record of the batch); the two “No points on the q … half” sentences and the “Only {points} points …”
    # forms of the quick fit are in zh_fit_series.py.
    "Every supplied sigma must be finite and positive": "给出的每个 σ 都必须是有限的正数",
    "This frame has no curve to fit.": "这一帧没有可拟合的曲线。",
    "no particle-model fit in this session": "本次会话中没有颗粒模型拟合",
    "no solution": "没有解",
    # A frame without counts on what was to be fitted (a closed shutter): the batch's own reason (NO_INTENSITY).
    "The horizontal cut has no intensity above zero": "水平切线上没有大于零的强度",
    "No intensity above zero on the q > 0 half": "q > 0 一半没有大于零的强度",
    "No intensity above zero on the q < 0 half": "q < 0 一半没有大于零的强度",
    # What the quick fit itself refuses a curve with (shared/signed_cut.prepare_cut, then prepare_sides of Fitting
    # without its “positive: ”).
    "The horizontal cut has no measured points.": "水平切线没有测量点。",
    "duplicate q values; combine replicates with their uncertainties first": "有重复的 q 值；请先把重复测量连同其不确定度合并",
    "at least one positive intensity is required": "至少要有一个大于零的强度",
    "q and intensity must have equal lengths and contain finite values": "q 和强度必须长度相同，且都是有限值",
    "No nonzero q points in the selected side": "所选的一侧没有 q 不为零的点",
    "The particle-model fit is not available in this session.": "本次会话中无法使用颗粒模型拟合。",
    "No region with a q window to fit: pick rings in Cuts (Ring) first.": "没有带 q 窗口的区域可拟合：请先在“切线”（环）中选环。",

    # -- Batch Export / Send Series to Fitting: what the dialog says before Export ------------------------------
    # (analyze/presentation/batch_summary.py, batch_dialog.py, views/batch_export_view.py, bindings/batch_export.py;
    # "Cuts" and "Files go to {folder}" are in the tables of Analyze)
    "Find the cut again in every frame": "在每一帧里重新找切线",
    "Off: every frame is cut where the frame on screen is cut, so the curves of a series can be compared point by "
    "point. On: the cut is looked for in each frame on its own (it may sit on other rows in a weak frame); the "
    "record of the batch has the cut of every frame either way.":
        "不勾选：每一帧都在当前显示这一帧的切线位置上切，序列里的曲线可以逐点比较。勾选：每一帧各自重新找切线"
        "（信号弱的帧可能落在别的行上）；两种情况下批量记录里都有每一帧的切线位置。",
    "Next to the Data": "数据旁边",
    "Save into the folder gimap_analysis next to the data files": "保存到数据文件旁边的 gimap_analysis 文件夹",
    "Summary": "概要",
    "This folder cannot be written ({reason}): choose another one.": "这个文件夹无法写入（{reason}）：请另选一个。",
    "a file of that name is in the way": "已有一个同名的文件",
    "there is no such drive or folder": "没有这个盘或文件夹",
    "no permission to write there": "没有写入权限",
    "This folder already holds an export of these data ({when}, {frames} frames): Export asks whether to replace it "
    "or to keep both.":
        "这个文件夹里已有这组数据的一次导出（{when}，{frames} 帧）：点“导出”时会问是替换它还是两者都保留。",
    "This folder already holds {n} curves for Fitting: Export asks whether to replace them or to keep both.":
        "这个文件夹里已有 {n} 条拟合用曲线：点“导出”时会问是替换它们还是两者都保留。",
    "This is the folder of the last batch, which was made of other data.": "这是上一次批量导出用的文件夹，那次导出的是另一组数据。",
    "“{folder}” already holds an export of these data, made {when} ({frames} frames).":
        "“{folder}”里已有这组数据的一次导出（{when}，{frames} 帧）。",
    "“{folder}” already holds {n} curves for Fitting.": "“{folder}”里已有 {n} 条拟合用曲线。",
    "Fitting ▸ In-situ series lists every curve of that folder: the old and the new ones would be fitted as one series.":
        "拟合 ▸ 原位序列 会列出该文件夹里的全部曲线：新旧曲线会被当成同一个序列来拟合。",
    "Replace removes the {n} files of the earlier export first.": "“替换”会先删除之前那次导出的 {n} 个文件。",
    "It was written by an earlier version of GIMaP, whose record lists only its tables: the files of single frames stay.":
        "它由较早版本的 GIMaP 写成，记录里只列出了汇总表格：逐帧的文件会留在原处。",
    "Keep Both writes this export into the new subfolder “{name}”.": "“两者都保留”会把这次导出写到新的子文件夹“{name}”里。",
    "Replace": "替换",
    "Keep Both in a New Subfolder": "两者都保留（写到新的子文件夹）",
    "Replace removes the {n} files of the earlier export first, among them {m} files of kinds this export does not "
    "write again: {kinds}.":
        "“替换”会先删除之前那次导出的 {n} 个文件，其中 {m} 个属于这次导出不会再写的类型：{kinds}。",
    "tables of all frames": "所有帧的汇总表格",
    "curves of single frames": "逐帧曲线",
    "curves for Fitting": "拟合用曲线",
    "q maps": "q 图",
    "cakes": "cake 图",
    "pictures": "图片",
    "converted frames": "转换后的探测器帧",
    "fit tables": "拟合结果表",
    "fitted curves": "拟合曲线",
    ", ": "、",  # between the kinds of file named in a row
    "the frame on screen is read again without the sum …": "正在不求和重新读取屏幕上的帧 …",
    "{n} s": "{n} 秒",
    "{n} min": "{n} 分钟",
    "{n} h": "{n} 小时",
    "1 frame": "1 帧",
    "{count} frames": "{count} 帧",
    "{count} of {total} frames (Every {n})": "共 {total} 帧中的 {count} 帧（每 {n} 帧取一帧）",
    "{count} sums of {size} frames": "{count} 组，每组 {size} 帧求和",
    "{count} of {total} sums of {size} frames (Every {n})": "共 {total} 组中的 {count} 组，每组 {size} 帧求和（每 {n} 组取一组）",
    "+ {n} more": "等另外 {n} 条",
    "curves: {names}": "曲线：{names}",
    "a fit takes: {halves}": "拟合使用：{halves}",
    "about {time}": "约 {time}",
    "about {size} per frame, {total} in all": "每帧约 {size}，共约 {total}",
    "about {total} in all": "共约 {total}",
    "README.txt in the folder says what each file is": "文件夹里的 README.txt 说明每个文件是什么",
    "sum(s): {count} of {total}": "求和组：共 {total} 组中的 {count} 组",
    "Export 1 Sum": "导出 1 组求和",
    "Export {count} Sums": "导出 {count} 组求和",
    "Batch Export {count} Frames in Sums of {n}…": "批量导出 {count} 帧（每 {n} 帧求和）…",
    "Sum {n} of the automatic analysis was switched off for the series (Data ▸ Sum N to sum again).":
        "自动分析设置的 {n} 帧求和已为这个序列关闭（要再求和：数据 ▸ 求和）。",
    "{groups} sums of {n} frames ({frames} frames)": "{groups} 组，每组 {n} 帧求和（共 {frames} 帧）",
    "{groups} sums of {n} frames ({frames} frames; the last sum has only {k})":
        "{groups} 组，每组 {n} 帧求和（共 {frames} 帧；最后一组只有 {k} 帧）",
    "{groups} sums of {n} frames ({frames} frames; {short} sums at the end of a source hold fewer)":
        "{groups} 组，每组 {n} 帧求和（共 {frames} 帧；其中 {short} 组在各自数据源的末尾，帧数不足）",
    "{frames} frames from 1 scan of {modules} module files": "1 次扫描（{modules} 个模块文件）中的 {frames} 帧",
    "{name}: a module file is missing or cannot be read yet — every frame lacks that part of the detector.":
        "{name}：有模块文件缺失或暂时读不了——每一帧都缺探测器的这一部分。",
    "{name} is still arriving: the batch takes the {frames} frames that are there now.":
        "{name} 还在陆续到达：这次批量只处理现在已有的 {frames} 帧。",
    "… and {n} more sources that are not whole": "… 还有 {n} 个不完整的数据源",

    # -- running a batch and its end (bindings/batch_run.py, bindings/batch_panel.py, batch_texts.py) -------------
    "Choose Another Folder…": "另选文件夹…",
    "The earlier export in {folder} could not be removed: {error}": "无法删除 {folder} 里之前的导出：{error}",
    "Series map {done}/{total}": "序列热图 {done}/{total}",
    "Series map {done}/{total}: {name} …": "序列热图 {done}/{total}：{name} …",
    "{done} of {total} frames in {time}": "共 {total} 帧中的 {done} 帧，用时 {time}",
    "1 frame in {time}": "1 帧，用时 {time}",
    "{frame}: {reason}": "{frame}：{reason}",
    "… and {n} more: README.txt in the folder has every one": "… 另有 {n} 帧：文件夹里的 README.txt 列出了全部",
    "tables and records {n}": "表格和记录 {n}",
    "{n} files written: {where}": "已写入 {n} 个文件：{where}",
    "{n} of {total} frames failed": "{total} 帧中有 {n} 帧失败",
    "Batch Export stopped: the folder takes no more files": "批量导出已停止：文件夹无法再写入文件",
    "Batch Export stopped: the folder takes no more files ({error})": "批量导出已停止：文件夹无法再写入文件（{error}）",
    "Batch Export ended: {n} of {total} frames failed": "批量导出结束：{total} 帧中有 {n} 帧失败",
    "Batch Export stopped: {done} of {total} frames exported": "批量导出已停止：{total} 帧中导出了 {done} 帧",
    "Batch Export: the tables of all frames could not be written ({error})": "批量导出：所有帧的汇总表格无法写入（{error}）",
    "Stopped: {folder} takes no more files ({error}); {done} of {total} frames were exported":
        "已停止：{folder} 无法再写入文件（{error}）；{total} 帧中导出了 {done} 帧",
    "Exported {done}/{total} frames to {folder}, but the tables of all frames could not be written: {error}":
        "已导出 {done}/{total} 帧到 {folder}，但所有帧的汇总表格无法写入：{error}",
    "Stopped: {done} of {total} frames exported to {folder} in {seconds} s ({left} not started)":
        "已停止：{total} 帧中导出了 {done} 帧到 {folder}，用时 {seconds} s（{left} 帧未开始）",
}

__all__ = ["SERIES_ZH"]
