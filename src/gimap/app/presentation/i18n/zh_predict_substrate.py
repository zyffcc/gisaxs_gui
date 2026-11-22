"""Known substrate and cut geometry in Predict parameters."""

PREDICT_SUBSTRATE_ZH = {
    "substrate": "基底",
    "file optics unused": "未采用文件中的基底光学参数",
    "Unsupported substrate material {material}.": "不支持基底材料 {material}。",
    "The substrate needs a known material or finite optical constants δ > 0 and β ≥ 0.":
        "请提供支持的基底材料，或有限的光学常数 δ > 0 和 β ≥ 0。",
    "Each specified film needs its thickness in nm and finite optical constants δ and β.":
        "每层薄膜均须给出膜厚（nm）及有限的光学常数 δ 和 β。",
    "The substrate description contains unrecognized layer or optical fields.": "基底描述含有无法识别的膜层或光学参数。",
    "The material table does not cover the cut energy.": "该切线的能量超出材料表范围。",
    "The notes give conflicting substrate materials.": "笔记中给出的基底材料互相矛盾。",
    "Give one material or a complete optical description.": "请指定一种材料或给出完整的光学参数。",
    "With both cut angles and a known substrate, Predict and Refine use four-term DWBA. "
    "The material table covers 4.9–30 keV. To enter a film, open Cut geometry and substrate in Fit.":
        "两个切线角度和基底已知时，预测和精修采用四项 DWBA。材料表覆盖 4.9–30 keV。若有薄膜，在拟合步骤打开“切线几何与基底”。",
    "at the fourth level": "在第四层",
    "four-term DWBA on {substrate}": "{substrate} 上的四项 DWBA",
    "Cut geometry and substrate…": "切线几何与基底…",
    "Cut geometry and substrate": "切线几何与基底",
    "Substrate δ": "基底 δ",
    "Substrate β": "基底 β",
    "Incidence angle αi (°)": "入射角 αi（°）",
    "Exit angle αf (°)": "出射角 αf（°）",
    "Photon energy (keV)": "光子能量（keV）",
    "Top layer": "表层薄膜",
    "Layer thickness (nm)": "膜厚（nm）",
    "Layer δ": "薄膜 δ",
    "Layer β": "薄膜 β",
    "Enter both cut angles between 0° and 90° for the known substrate.": "请为已知基底输入切线的两个角度，均须在 0° 和 90° 之间。",
    "Enter a finite number for {field}.": "请为{field}输入有限数值。",
    "Wavelength and photon energy must be positive.": "波长和光子能量须为正数。",
    "Cut angles must be at least 0° and below 90°.": "切线角度须大于或等于 0° 且小于 90°。",
    "The material table covers 4.9–30 keV. Enter δ and β at this energy.":
        "材料表的范围是 4.9–30 keV。请手动输入此能量下的 δ 和 β。",
    "A substrate needs δ > 0 and β ≥ 0": "基底须满足 δ > 0 且 β ≥ 0",
    "A layer needs a finite thickness > 0 nm": "膜厚须大于 0 nm 且为有限值",
    "A layer needs finite δ ≥ 0 and β ≥ 0": "薄膜的 δ 和 β 须为有限的非负数",
}
