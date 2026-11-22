"""Orientation components pooled with the isotropic fraction."""

CELL_ORIENTATION_ZH = {
    "Resolved oriented {oriented}% · isotropic including near-random {isotropic}%":
        "可分辨的取向晶粒 {oriented}% · 各向同性晶粒（含近随机组分）{isotropic}%",
    "Near-random · {reason}": "近随机组分 · {reason}",
    "isotropic replacement within 6 BIC units": "换成各向同性组分后的 BIC 差值小于 6",
    "fraction trades with the isotropic fraction within 3σ": "在 3σ 内无法将其分数与各向同性分数区分开",
    "no population can be separated from the isotropic fraction": "无法从各向同性分数中分辨出取向组分",
    "Profile-model ΔBIC {delta} · joint χ²red {joint} · ring χ²red {ring}":
        "分布拟合模型 ΔBIC {delta} · 联合 χ²red {joint} · 该环 χ²red {ring}",
}
