"""Chinese texts of the NXcanSAS (HDF5) export and import of curves: Analyze's Export step and Batch Export, Fitting's
Open Curve, Import Curves and Save Data and Fit."""

NXCANSAS_ZH = {
    # Analyze ▸ Export (analyze/presentation/views/analyze_steps_view.py)
    "Also as NXcanSAS (.h5)": "同时导出为 NXcanSAS（.h5）",
    "Export Curves…, Save the Curves of the Plot and Export New Frames While Watching also write "
    "<frame>_nxcansas.h5: the curves against q (Q in 1/Å, I, σ, the wavelength, distance and αi, and the record) as "
    "NXcanSAS (HDF5), for SasView and other NeXus programs. I(χ) and 2θ curves stay CSV only.":
        "导出曲线…、保存这张图的曲线和监视时导出新帧同时写出 <frame>_nxcansas.h5：以 q 为横轴的曲线"
        "（Q 单位 1/Å、I、σ、波长、距离和 αi，以及记录），格式为 NXcanSAS（HDF5），供 SasView 和其他 NeXus 程序读取。"
        "I(χ) 和 2θ 曲线仍只写 CSV。",
    # Batch Export (analyze/presentation/views/batch_export_view.py, batch_summary.py)
    "Each frame: its q curves as NXcanSAS (.h5)": "每帧：以 q 为横轴的曲线，NXcanSAS 格式（.h5）",
    "One HDF5 file per frame for SasView and other NeXus programs: every chosen curve against q with σ, the "
    "wavelength, distance and αi, and the record (I(χ) and 2θ curves are not in it)":
        "每帧一个 HDF5 文件，供 SasView 和其他 NeXus 程序读取：所选的每条以 q 为横轴的曲线（含 σ）、波长、距离和 αi，"
        "以及记录（不含 I(χ) 和 2θ 曲线）",
    "NXcanSAS files": "NXcanSAS 文件",
    # Fitting (fitting/presentation/single/page.py, curve_import.py)
    "Choose a Curve": "选择曲线",
    "{name} holds {count} curves. Which one is to be fitted?": "{name} 含有 {count} 条曲线。要拟合哪一条？",
    "NXcanSAS file: every curve in it (each SASdata group: Q, I and Idev as σ) is imported; the rows shown are those "
    "of the first.":
        "NXcanSAS 文件：导入其中的每一条曲线（每个 SASdata 组：Q、I，Idev 作为 σ）；表中显示的是第一条曲线的行。",
}

__all__ = ["NXCANSAS_ZH"]
