"""Decision to average the overlapping halves of a GISAXS cut."""

SIGNED_CUT_ZH = {
    "CSV columns give q (Å⁻¹), I, marginal σ, fitted I and the σ used for fitting.":
        "CSV 列依次为 q（Å⁻¹）、I、各点的边际标准差 σ、拟合 I 和拟合所用的 σ。",
    "Both halves are usable (points in {a} and {b} of their |qy| ranges) and {agree}: they are averaged where both "
    "exist{extend}.": "两半都可用（分别在各自 |qy| 范围的 {a} 和 {b} 内有数据点），并且{agree}："
    "在两半都有数据的地方取平均{extend}。",
}
