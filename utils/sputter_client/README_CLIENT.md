# Analysis package for the GIMaP Sputter page (release)

GIMaP runs this package for **Sputter > Analyze** (in-situ scans) and **Details > Static Frames** (samples measured
after their deposition). It needs Python 3.9 or newer with NumPy, SciPy and Matplotlib (`requirements.txt`); GIMaP
uses its own Python.

Without GIMaP:

    run.cmd analyze --frames <folder of CBF frames> --label <new name> --metal Ag --substrate Si --rate-nm-s 0.25 \
        --out <output folder> -- --sdd-mm <distance> (--beam-col <col> --beam-row <row> | --auto-centre)
    run.cmd static --frames <folder | file.cbf | archive.zip::folder/> --label <name> --nominal-nm 8 --out <series>

(`run.sh` on Linux and macOS.) The output folder holds `analysis.json`, `REPORT.md` and `main/` (the solutions of
every time window with the measured and model images), which GIMaP shows.

Metals and substrates: Au and Ag on Si and on porous TiO2, Ag on polymer films (PS, PET, PSBMA, PNO), Cu on PEO;
static frames: Au films at an incidence angle near 0.40 deg. The values are those of a physical model whose images
reproduce the measured ones; they were not compared with an independent measurement of the samples.

GIMaP Sputter 页面使用的分析包：原位扫描用“分析”，沉积后测量的样品用“详情 ▸ 静态帧”。需要 Python 3.9 以上和
NumPy、SciPy、Matplotlib；数值是能重现实测图样的物理模型给出的，没有与样品的独立测量比较过。
