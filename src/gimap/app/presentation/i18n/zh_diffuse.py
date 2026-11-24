"""Chinese texts of the diffuse term of a rough interface in Fitting (``fit_model.DIFFUSE``): its values, its curve in
the plot and why its values have no error when it is off."""

DIFFUSE_ZH = {
    "Diffuse A": "漫散射 A",
    "Diffuse w": "漫散射 w",
    "Diffuse H": "漫散射 H",
    "Height at q = 0 of the diffuse scattering of a rough interface A_d (1 + (|q|/w)²)^−(1+H); 0: no diffuse term":
        "粗糙界面漫散射 A_d (1 + (|q|/w)²)^−(1+H) 在 q = 0 处的高度；0 表示没有漫散射项",
    "Width w = 1/ξ of the diffuse term, ξ the correlation length of the roughness": "漫散射项的宽度 w = 1/ξ，ξ 为粗糙度的相关长度",
    "Hurst exponent H of the roughness: the diffuse term falls as q^−2(1+H) (H = 1: also the Debye–Bueche form of large "
    "inhomogeneities)": "粗糙度的 Hurst 指数 H：漫散射项按 q^−2(1+H) 下降（H = 1 时也是大尺度不均匀性的 Debye–Bueche 形式）",
    "diffuse scattering": "漫散射",
    "No error: Diffuse A is 0, so this value changes nothing.": "没有误差：漫散射 A 为 0，这个数值不改变曲线。",
}

__all__ = ["DIFFUSE_ZH"]
