"""Chinese for hand-overs between workspaces (Analyze to Fitting and Compare, projects, the Start page)."""

HANDOVER_ZH = {
    "Series not opened in Fitting — the exported curves are in {folder}.": "序列没有在拟合中打开——导出的曲线位于 {folder}。",
    # -- Compare: a folder Analyze wrote holds several kinds of curves ---------------------------------
    "{folder} holds {n} kinds of curves ({kinds}): choose one of them.": "{folder} 中有 {n} 种曲线（{kinds}）：请选择其中一种。",
    "{folder} holds no curves “{kind}” ({kinds}).": "{folder} 中没有“{kind}”这种曲线（现有：{kinds}）。",
    "The table {name} holds fewer than two frames.": "表格 {name} 中的帧不足两个。",
    "Curves of the Folder": "文件夹中的曲线",
    "The curve files of a folder (x and I columns), in counting order: one series. A folder Analyze wrote "
    "(a Batch Export, its curves, gimap_analysis) holds several kinds of curves: you choose one":
        "一个文件夹里的曲线文件（x、I 两列），按编号顺序：作为一个序列。分析页写出的文件夹（批量导出、其中的 curves、"
        "gimap_analysis）含有多种曲线：选择其中一种",
    "{folder} holds {n} kinds of curves. One kind is one series — which one?":
        "{folder} 中有 {n} 种曲线。一种曲线是一个序列——选哪一种？",
    "{key} — {title} · {n} frames": "{key} — {title} · {n} 帧",
    "{key} · {n} frames": "{key} · {n} 帧",
    "the curve for Fitting ({title})": "拟合用曲线（{title}）",
    "the curve for Fitting": "拟合用曲线",
    "other files": "其他文件",
    "no table of frames: a header row and at least three rows of x and one column per frame":
        "不是逐帧表格：需要一行表头、至少三行 x，以及每帧一列",
    "not a table of numbers ({error})": "不是数值表格（{error}）",
    "Nothing added: no curve of {folder} was chosen.": "没有添加任何内容：没有选择 {folder} 中的曲线。",
    "In Analyze, open a sample and build its map in the Series tab, then Send to Compare; repeat for every sample. "
    "A Batch Export folder of Analyze, or any folder of curve files (x and I columns), can be added too, or dropped "
    "on this page: one kind of curve is one series, and a folder with several asks which.":
        "在分析页打开一个样品，在序列标签页生成热图，再点“发送到比较”；每个样品重复一次。分析页批量导出的文件夹，"
        "或任何曲线文件（x、I 两列）的文件夹，也可以添加或直接拖放到本页：一种曲线是一个序列，"
        "文件夹里有多种曲线时会询问用哪一种。",
    # -- Fitting: a curve that is the I(q) of a GIWAXS frame ----------------------------------------------
    "A GIWAXS I(q), not a GISAXS cut": "这是 GIWAXS 的 I(q)，不是 GISAXS 切线",
    "GIWAXS I(q) curves, not GISAXS cuts": "这些是 GIWAXS 的 I(q) 曲线，不是 GISAXS 切线",
    "Fitting's models are particle form factors for the horizontal cut of a GISAXS frame: they do not "
    "describe diffraction peaks, so no particle-shape search is set up for this curve. For its peaks use "
    "Analyze ▸ Batch Export ▸ Peaks of the ring regions.":
        "拟合页的模型是颗粒形状因子，用于 GISAXS 帧的水平切线：它们不能描述衍射峰，所以没有为这条曲线预设颗粒形状搜索。"
        "要拟合它的峰，请用“分析 ▸ 批量导出 ▸ 环区域中的峰”。",
    "Fitting's models are particle form factors for the horizontal cut of a GISAXS frame: they do not "
    "describe diffraction peaks, and Start would fit one to every frame. To follow the peaks through the "
    "series use Analyze ▸ Batch Export ▸ Peaks of the ring regions.":
        "拟合页的模型是颗粒形状因子，用于 GISAXS 帧的水平切线：它们不能描述衍射峰，而“开始”会用这样的模型去拟合每一帧。"
        "要跟踪整个序列中的峰，请用“分析 ▸ 批量导出 ▸ 环区域中的峰”。",
    "{name} is a GIWAXS I(q): no particle-shape search was set up.": "{name} 是 GIWAXS 的 I(q)：没有预设颗粒形状搜索。",
    "Start fitted a particle form factor to every frame ({fitted} fitted). Such models do not describe "
    "diffraction peaks: read these values with care. To follow the peaks through the series use "
    "Analyze ▸ Batch Export ▸ Peaks of the ring regions.":
        "“开始”用颗粒形状因子拟合了每一帧（拟合了 {fitted} 帧）。这类模型不能描述衍射峰：请谨慎看待这些数值。"
        "要跟踪整个序列中的峰，请用“分析 ▸ 批量导出 ▸ 环区域中的峰”。",
    "Start fitted a particle form factor to every frame: {poor} of the {fitted} frames fitted are poor "
    "fits (χ²ᵣ above {limit}). Such models do not describe diffraction peaks: these values are no "
    "result. To follow the peaks through the series use Analyze ▸ Batch Export ▸ Peaks of the ring "
    "regions.":
        "“开始”用颗粒形状因子拟合了每一帧：拟合的 {fitted} 帧中有 {poor} 帧拟合很差（χ²ᵣ 大于 {limit}）。"
        "这类模型不能描述衍射峰：这些数值不是结果。要跟踪整个序列中的峰，请用“分析 ▸ 批量导出 ▸ 环区域中的峰”。",
    "Close this note": "关闭这条提示",
    "The former fitting page could not take this curve: {error}": "旧版拟合页无法载入这条曲线：{error}",
    "The former fitting page could not take this model: {error}": "旧版拟合页无法载入这个模型：{error}",
    # -- Analyze to Fitting: where the file went ---------------------------------------------------------
    "Sent to Fitting — the curve file is in {folder}": "已发送到拟合——曲线文件位于 {folder}",
    "Opened {name} in Fitting. The folder of the data cannot be written: the file is in {folder}":
        "已在拟合中打开 {name}。数据所在的文件夹不可写：文件位于 {folder}",
    "Cannot write in {folder} ({error}); export to…": "无法写入 {folder}（{error}）；导出到…",
    # Analyze ▸ Send to Fitting on a GIWAXS frame (analyze/presentation/page.FIT_TIP_GIWAXS)
    "Open the I(q) in Fitting (q in Å⁻¹). Fitting's models are particle form factors for the horizontal cut of a "
    "GISAXS frame: the peaks of a GIWAXS frame are fitted in Batch Export ▸ Peaks of the ring regions.":
        "在拟合中打开 I(q)（q 以 Å⁻¹ 为单位）。拟合页的模型是颗粒形状因子，用于 GISAXS 帧的水平切线：GIWAXS 帧的峰"
        "请在“批量导出 ▸ 环区域中的峰”里拟合。",
    # Export New Frames While Watching when no folder takes the files (analyze/presentation/bindings/batch_watch.py)
    "Export New Frames While Watching is off: the folder of the data cannot be written ({error}) and no other "
    "folder was chosen.": "“监视时导出新帧”已关闭：数据所在的文件夹不可写（{error}），也没有选择其他文件夹。",
    # -- the Start page: a series -------------------------------------------------------------------------
    "CBF, TIFF, EDF, NeXus / HDF5 — one file per frame, or a whole series in one file; the module files of a "
    "scan are stitched": "CBF、TIFF、EDF、NeXus / HDF5——每帧一个文件，或整个序列存在一个文件里；一次扫描的各模块文件会自动拼接",
    "The frames of a folder: one file per frame, or the module files of one scan":
        "一个文件夹里的帧：每帧一个文件，或一次扫描的各模块文件",
    "Chosen frames — or one NeXus / HDF5 file that holds the whole series":
        "选中的若干帧——或一个包含整个序列的 NeXus / HDF5 文件",
}

__all__ = ["HANDOVER_ZH"]
