# GISAXS 处理手册（给 agent：Codex、Claude Code、GUI 里的 AI、脚本）

与 GIWAXS 手册同一原则：**判断交给代码，好奇心留给 agent。** 底线四条相同（测量值只来自工具；
数据目录只读；不编参数；分清发现和假设）。

## 1. 第一层：基线

```bash
python tools/gimap_agent.py auto <图像> [<图像> ...] --technique gisaxs --notes "<用户笔记原文>"
```

基线按顺序做（每一步都是一个工具调用，GUI 的 Run Automatic Analysis 也一样）：

1. 几何：仪器配置 / 附近的标定文件或标样图像（同 GIWAXS）。没有 αi 就问。
2. 水平切线 I(qy)：放在 Yoneda 带（αf ≈ αc 的漫散射极大）；找不到 Yoneda 时放在样品地平线上方，
   并在 “Needs attention” 里提示检查 αi。
3. 光束中心列：直射束通常被 beam stop 挡住，用 I(qy) 的左右对称性确定（`refine_beam_center_symmetry`）；
   偏离标定中心超过 20 px 会提示检查标定或样品对准。
4. 两半（`choose_halves` → `set_halves`）：两半都可用且一致（差 < 15 %）就平均，平均区之外用更长的那一半；
   一半被遮挡、有缝隙或太短就只用可用的一半；两半不一致就两半都保留（|qy| 两种颜色）。理由写进报告。
5. 面内间距（`in_plane_spacing`）：I(qy) 在 qy = 0 之外的极大 → D = 2π/q*；只有斜率变化（shoulder）时
   写成“提示，不是测量”。
6. 拟合（`fit_horizontal_cut`，Fitting 的数值物理拟合）：球、竖直圆柱、随机取向圆柱，含尺寸分布和
   间距 D；也从第 5 步的间距起步。给出所有互不相同的解。

输出：`report.md/json`、`horizontal.csv`、`vertical.csv`、`fit_curve.csv`（q、I、σ、I_fit）、
`fit_solutions.csv`（每个解的参数，nm）和 `qmap.png`。

## 2. 第二层：在基线之上

| 看什么 | 为什么 | 怎么做 |
|---|---|---|
| 模型之间的差别 | χ² 相差 10 % 以内时曲线本身不能决定模型 | 看哪个解的 D 与 shoulder / 极大一致；用户给了材料或形状时按其选；在 Fitting 里精修 |
| D 接近搜索上限（几百 nm） | 该解在 q 范围内没有粒子间关联 | 不要把它当测得的距离 |
| 切线位置 | Yoneda 带之外的切线混入别的散射 | `get_curve vertical` 看 Yoneda 与镜面峰位置是否与 αi、λ 一致；`set_gisaxs_cuts` 换位置 |
| 不同 qz 的切线 | shoulder 位置不随 qz 变 → 面内间距 | 在 Yoneda 与镜面之间再切一条比较 |
| 序列怎么变 | 原位实验看变化 | GUI 的 Series 热图（帧 × qy），或 `--frame` 对比首末 |

## 3. 已知参考

`tests/data/external/gisaxs_galaxi/`（BornAgain 的 GALAXI 示例）：BornAgain 模型为 Ag 球，
径向准晶间距 53.6 nm。GIMaP 基线：Yoneda 行 ≈ 607，对称轴 596.3 px（文献 597.1），两半平均，
shoulder 在 |qy| = 0.01175 Å⁻¹（2π/q ≈ 53.5 nm），球解 R ≈ 9.4 nm、D ≈ 53.4 nm，与随机圆柱解 χ² 相近
（报告会要求选择模型）。
