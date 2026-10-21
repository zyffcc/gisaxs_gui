# GISAXS 处理手册

适用对象与底线同 [GIWAXS 手册](giwaxs-playbook.md)：测量值只来自工具；数据目录只读；不编参数；分清发现和假设。

## 基线

```bash
python tools/gimap_agent.py auto <图像> [<图像> ...] --notes "<用户笔记原文>" --out <文件夹>
```

GIMaP 应用几何后会自动识别 GISAXS（探测器没有超过 2θ = 20°）；要强制时加 `--technique gisaxs`。
GUI 的 Run Automatic Analysis 做同样的事。基线按以下顺序进行。除了第 2 步，每一步都是一个可以单独调用的工具；
第 2 步的切线由 GISAXS 还原自动放在 Yoneda 带，要换位置用 `set_gisaxs_cuts`。

1. **几何**：顺序同 GIWAXS（`--calibration`、笔记里点名的文件、仪器配置、在附近搜索）。搜索采用的标定在
   Decisions 里有一行 “calibration found by the search”，写出它的完整路径和位置：先确认它属于这次测量。
   没有 αi 时先用 0°，并在 Needs attention 里提问（退出码 2）。运行中没有人可问，由调用它的 agent 补
   `--incidence-deg` 后重跑。
2. **水平切线 I(qy)**：放在 Yoneda 带（αf ≈ αc 处的漫散射极大）。找不到 Yoneda 带时放在样品地平线上方，并提示检查 αi。
3. **光束中心列**：直射束通常被 beam stop 挡住，所以用 I(qy) 的左右对称性确定（`refine_beam_center_symmetry`）。
   先用当前切带的有效像素在探测器范围内粗定位，再精修 x。列强度取有效像素均值，与显示曲线一致；
   精修按离中心距离的倒数加权，平衡近中心和远端不同倍数的距离区间，让低 q 点保持影响力
   （近似对应 GISAXS 的对数 |qy| 分布）。候选中心使用同一组实测配对点，不跨缝隙插值，离中心不足 1 px 的点不参与。
   停止容差 0.0001 px 是数值容差，不是实验不确定度；y 保持原值。
   数据不足或对称轴有歧义时不应用结果，先检查切带和掩膜。
   偏离标定中心超过 20 px 时提示检查标定或样品对准。
4. **两半**（`choose_halves` → `set_halves`）：
   - 两半都可用且一致（中位 |ln(I₋/I₊)| ≤ 0.15，约 16 %）：取平均；平均区之外用较长的那一半。平均不跨另一半的缺口插值：
     另一半在附近没有测量点处用这一半原样，有缺口时报告写出平均了几个点、几个点只来自一半（`shared/signed_cut.py`）。
   - 一半被遮挡、有缝隙或太短：只用另一半。判据是覆盖 < 60 % 且比另一半低 10 个百分点以上，
     或者到达的 |qy| 不到另一半的 30 %。
   - 两半不一致：两半都保留（|qy| 上两种颜色），只拟合覆盖更好的那一半。

   理由都写进报告。
5. **面内间距**（`in_plane_spacing`）：I(qy) 在 qy = 0 之外的极大给出 D = 2π/q*。只有斜率变化（shoulder）时，
   写成“提示，不是测量”。
6. **拟合**（`fit_horizontal_cut`，用 Fitting 的数值物理拟合）：球、竖直圆柱、随机取向圆柱，含尺寸分布和类晶间距 D；
   也会从第 5 步的间距出发。最多给出 5 个互不相同的解：report.md 和 fit_solutions.csv 全部列出，而在 MCP 或 call 里
   直接调用 `fit_horizontal_cut` 只返回前 3 个。`--no-fit` 只准备切线，不拟合。

输出：`report.md/json`、`horizontal.csv`、`vertical.csv`、`fit_curve.csv`（q、I、σ、I_fit）、`fit_solutions.csv`
（每个解的每个组分一行：R、h、D 单位 nm；sigma_R、sigma_h、sigma_D 是相对宽度 σ/值，没有单位）和 `qmap.png`。
给了 `.poni` 的一帧约 25 秒。

## 该怀疑的地方

| 看什么 | 为什么 | 怎么做 |
|---|---|---|
| 模型的选择 | 几个解的 χ² 相差 10 % 以内时，曲线本身定不了模型。所以 Needs attention 里的 “model” 是**正常结果**，不是失败 | 看哪个解的 D 和 shoulder / 极大一致；用户给了材料或形状就按它选；在 Fitting 里精修并比较拟合曲线 |
| χ² 最好的解没有粒子间关联 | D ≥ 250 nm（搜索上限 500 nm 的一半；下面 GALAXI 的 500 nm 就是这种）说明该解在 q 范围内没有粒子间关联 | 不要把这样的 D 当成测得的距离 |
| 切线位置 | Yoneda 带之外的切线会混进别的散射 | `get_curve vertical` 看 Yoneda 带和镜面峰的位置是否与 αi、λ 一致；`set_gisaxs_cuts` 换位置 |
| 两半的决定 | 遮挡、吸收体或真实的面内各向异性都会让两半不同 | 读报告里的理由；必要时在 Fitting 里分别拟合两半 |
| 垂直切线 I(qz) | 已导出（`vertical.csv`），但没有分析（膜厚、Kiessig 条纹、Yoneda 带与 αc 是否吻合） | 问题需要时自己分析，并写出依据 |
| 不同 qz 的切线 | shoulder 位置不随 qz 变 → 是面内间距 | 在 Yoneda 带和镜面峰之间再切一条比较 |
| 对称轴偏移大 | 标定或样品对准可能有问题 | 对比标定中心和对称轴，说明差多少 |
| 序列 | 原位实验看变化；默认只分析终态（最后五分之一的帧求和，至多 10 帧；不到 10 帧只取最后一帧） | GUI 的 Series 热图（帧 × qy，每帧用同一条切线）和 Compare；命令行用 `--frame 1 --sum K`（K 同终态的求和帧数）对比首末帧；一帧一个文件的序列加 `--series` |

## 公开参考数据

`tests/data/external/gisaxs_galaxi/`（BornAgain 的 GALAXI 示例）：模型是银球，R = 5.75 nm（对数正态分布，σ 0.4），
径向类晶间距 53.6 nm（见 `tests/data/external/README.md`）。

```bash
python tools/gimap_agent.py auto tests/data/external/gisaxs_galaxi/galaxi_data.tif \
    --calibration tests/data/external/gisaxs_galaxi/galaxi_bornagain.poni --incidence-deg 0.463 --out <文件夹>
```

基线结果（2026-10-14，macOS，退出码 2，只剩 “model” 这一个待回答的问题）：

- Yoneda 带在第 605–609 行；对称轴在 597.3 px；两半取平均（到 |qy| = 0.179 Å⁻¹，之后 qy < 0 一半单独到 0.278 Å⁻¹；
  有 15 个点落在另一半的缺口里，只来自一半）；
- shoulder 在 |qy| = 0.0113 Å⁻¹，2π/q ≈ 55.6 nm；
- χ² 最好的是没有粒子间关联的随机圆柱解（D = 500 nm，在搜索上限）；
- 与间距一致的是球解：R ≈ 9.4 nm、D ≈ 48 nm。它的 D 接近模型的 53.6 nm；R 不能直接和模型的 5.75 nm 比，
  因为模型是对数正态分布（σ 0.4），GIMaP 用的是高斯 σR/R，宽分布下散射偏重大颗粒。

这个例子正好说明：选模型要看间距和用户知道的形状，不能只看 χ²。
