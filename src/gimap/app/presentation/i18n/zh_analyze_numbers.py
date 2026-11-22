"""Analyze measurements in result labels and summaries."""

ANALYZE_NUMBERS_ZH = {
    "{estimate} (68% interval)": "{estimate}（68% 区间）",
    "{model} (BIC weight {p}), {params}, χ²ᵣ = {chi}": "{model}（BIC 权重 {p}），{params}，χ²ᵣ = {chi}",
    "Most probable: {model} (p = {p}), {params}, χ²ᵣ = {chi}": "最可能：{model}（p = {p}），{params}，χ²ᵣ = {chi}",
    "Measured αi  {estimate}°": "实测 αi  {estimate}°",
    "A_abs = {estimate} (1σ)\n{points} points from {low} to {high} Å⁻¹":
        "A_abs = {estimate}（1σ）\n{points} 个点，范围 {low} 至 {high} Å⁻¹",
    "Saved A_abs = {estimate} · {standard} · {date}": "已保存 A_abs = {estimate} · {standard} · {date}",
    "{name}: q = {q} Å⁻¹ (d = {d} Å), FWHM {w} Å⁻¹, area {a}, χ²ᵣ {chi}":
        "{name}：q = {q} Å⁻¹（d = {d} Å），FWHM {w} Å⁻¹，面积 {a}，χ²ᵣ {chi}",
}
