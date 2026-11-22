"""Chinese of Analyze's Phases window: reflection overlay, ring assignment and pole figure (``bindings/indexing.py``,
``indexing_panel.py``, ``indexing_texts.py``, ``views/indexing_view.py``)."""

INDEXING_ZH = {
    # the toolbar button and the window
    "Phases": "物相",
    "Overlay the reflections of known phases (a unit cell or a CIF) on the q map, the detector and I(q); assign the "
    "rings; fit the pole figure of a ring":
        "把已知物相（晶胞或 CIF）的衍射点叠加到 q 图、探测器图像和 I(q) 上；指认衍射环；拟合一个环的极图",
    "Phases and Indexing": "物相与指标化",
    "Reflections": "衍射点",
    "Pole Figure": "极图",
    "Add from CIF…": "从 CIF 添加…",
    "Read the cell, the symmetry operations and the atom sites of a CIF": "读取 CIF 中的晶胞、对称操作和原子位置",
    "Add Cell": "添加晶胞",
    "A phase from a unit cell typed in, with its lattice centring": "由输入的晶胞和点阵心型建立物相",
    "Show": "显示",
    "Cell (Å, °)": "晶胞（Å，°）",
    "Symmetry": "对称性",
    "Selected Phase": "所选物相",
    "Centring": "心型",
    "Lattice centring of a cell typed in (the reflections it extinguishes)": "输入晶胞的点阵心型（决定哪些衍射消光）",
    "How the crystallites lie: at random, or at random about the surface normal with a plane parallel to the "
    "substrate or an axis along the normal (a fibre texture)":
        "晶粒的取向：完全随机；或绕表面法线随机，某晶面平行于衬底、或某晶轴沿法线（纤维织构）",
    "Powder (random orientation)": "粉末（随机取向）",
    "Plane (hkl) parallel to the substrate": "晶面 (hkl) 平行于衬底",
    "Axis [uvw] along the surface normal": "晶轴 [uvw] 沿表面法线",
    "Overlay": "叠加",
    "Reflections up to this |q|": "列出 |q| 不超过此值的衍射",
    "q max": "q 上限",
    "Shift the fibre spots for refraction in the film": "按薄膜中的折射移动纤维织构衍射点",
    "Place each spot where refraction puts it (transmitted channel), from αi and the film's αc; with Analyze's "
    "refraction correction on, the q map already shows the true qz of that channel":
        "由 αi 和薄膜的 αc 把每个衍射点放到折射后的位置（透射通道）；Analyze 已开启折射校正时，q 图显示的就是该通道的真实 qz",
    "Analyze's αc": "Analyze 的 αc",
    "Critical angle of the film; at 0 the one of Analyze's refraction correction": "薄膜的临界角；为 0 时用 Analyze 折射校正中的值",
    "Also the reflected channel (DWBA)": "同时显示反射通道（DWBA）",
    "The spot of the wave reflected by the substrate first, k_z,f = q_z + k_z,i, drawn hollow":
        "先经衬底反射的波产生的衍射点，k_z,f = q_z + k_z,i，以空心符号显示",
    "hkl labels": "hkl 标注",
    "αc of the film": "薄膜的 αc",
    "Fibre spots (q∥, qz ≥ 0) or powder rings of every phase shown; m: reflections at the position; |F|²: their "
    "mean; I: m·|F|² relative to the strongest of the phase.":
        "每个显示物相的纤维织构衍射点（q∥，qz ≥ 0）或粉末衍射环；m：落在该位置的衍射数；|F|²：其平均值；"
        "I：m·|F|²，以该物相最强者为 100。",
    "Pole Figure of This Ring": "这个环的极图",
    "Move the I(χ) ring onto the selected reflection and fit its pole figure, with χ0 of the reflection":
        "把 I(χ) 的环移到所选衍射上，并以该衍射的 χ0 拟合极图",
    "Direction": "方向",
    "Fit up to χ": "拟合到 χ",
    "The local linear background of a Gaussian fitted to I(q) at the ring": "在环处对 I(q) 拟合高斯峰得到的局部线性背景",
    "Fit Pole Figure": "拟合极图",
    "The pole figure as CSV and the indexing record (.json) beside it": "极图写为 CSV，旁边写指标化记录（.json）",
    "Gaussian + isotropic": "高斯 + 各向同性",
    "Maier–Saupe + isotropic": "Maier–Saupe + 各向同性",
    "Along the normal (χ0 = 0°)": "沿法线（χ0 = 0°）",
    "In the plane (χ0 = 90°)": "在面内（χ0 = 90°）",
    "Set χ0": "指定 χ0",
    "Fitted direction": "拟合方向",
    "N(χ) ∝ crystallites per degree": "N(χ) ∝ 每度的晶粒数",
    "not fitted": "未拟合",
    # messages of the window
    "Add Phase from CIF": "从 CIF 添加物相",
    "CIF files (*.cif);;All files (*)": "CIF 文件 (*.cif);;所有文件 (*)",
    "Could not read the CIF: {error}": "无法读取 CIF：{error}",
    "Read {name} from {file}.": "已从 {file} 读取 {name}。",
    "Phase {n}": "物相 {n}",
    "The orientation needs indices other than 0 0 0.": "取向需要不全为 0 的指数。",
    "The overlay needs a GIWAXS frame with a geometry.": "叠加需要一帧有几何的 GIWAXS 图像。",
    "Select a reflection in the table first.": "请先在表中选择一个衍射。",
    "Fitting the pole figure …": "正在拟合极图 …",
    "Fit a pole figure first.": "请先拟合极图。",
    "Export Pole Figure": "导出极图",
    "CSV files (*.csv)": "CSV 文件 (*.csv)",
    "Wrote {name} and its record": "已写入 {name} 及其记录",
    "Wrote {files}.": "已写入 {files}。",
    "Phases: {error}": "物相：{error}",
    "reflected channel": "反射通道",
    # composed sentences (indexing_texts.py)
    "({h} {k} {l}) parallel to the substrate": "({h} {k} {l}) 平行于衬底",
    "[{h} {k} {l}] along the surface normal": "[{h} {k} {l}] 沿表面法线",
    "powder (random orientation)": "粉末（随机取向）",
    "{group}{count} operations from the CIF": "{group}CIF 中的 {count} 个对称操作",
    "lattice centring {centring}": "点阵心型 {centring}",
    "{atoms} atom sites: |F|² from the atoms": "{atoms} 个原子位置：|F|² 由原子计算",
    "no atoms: every allowed reflection has intensity 1": "没有原子：每个允许的衍射强度都取 1",
    "No ring of I(χ): open a GIWAXS frame and drag the purple band over I(q) onto a ring.":
        "没有 I(χ) 的环：打开一帧 GIWAXS 图像，把 I(q) 上的紫色带拖到一个环上。",
    "Pole figure, q = {low}–{high} Å⁻¹": "极图，q = {low}–{high} Å⁻¹",
    "Ring of I(χ): q = {low}–{high} Å⁻¹ (drag the purple band over I(q), or choose a reflection).":
        "I(χ) 的环：q = {low}–{high} Å⁻¹（拖动 I(q) 上的紫色带，或选择一个衍射）。",
    "{model}: oriented fraction {fraction}, FWHM {fwhm}°, direction χ0 = {chi0}°":
        "{model}：取向分数 {fraction}，半高宽 {fwhm}°，方向 χ0 = {chi0}°",
    "Hermans f = {f} (fitted distribution, missing wedge filled; oriented part {f_oriented}); from the measured bins "
    "{f_measured}": "Hermans f = {f}（拟合的分布，缺失楔由模型补全；取向部分 {f_oriented}）；只用测到的分档为 {f_measured}",
    "2D order 2<cos²χ> − 1 = {order}; χ²_red = {chi2} over {points} bins up to χ = {chi_max}°; weighted coverage "
    "{coverage}": "二维取向参数 2<cos²χ> − 1 = {order}；χ²_red = {chi2}，共 {points} 个分档，到 χ = {chi_max}°；加权覆盖 {coverage}",
    "local linear background of a Gaussian fit to I(q)": "I(q) 高斯拟合的局部线性背景",
    "mean of I(q) beside the window": "窗口两侧 I(q) 的平均值",
}

__all__ = ["INDEXING_ZH"]
