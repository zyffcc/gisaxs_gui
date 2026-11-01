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
   αi 的来源顺序同 GIWAXS（选项、笔记、分析命令栏里设定的 αi、仪器配置）；之前已用 `set_incidence_angle` 或在命令栏设定的
   αi 会被采用，不再提问。都没有 αi 时先用 0°，并在 Needs attention 里提问（退出码 2）。运行中没有人可问，由调用它的 agent 补
   `--incidence-deg` 后重跑。`.poni` 的探测器倾斜被略去（GIMaP 的探测器是垂直于光束的平板）：倾斜 > 1° 或 |q| 的最大
   相对误差 > 0.5 % 时 Needs attention 里有 “detector tilt”（退出码 2，没有选项能补），否则 Decisions 写出误差；
   GISAXS 的小角度下误差通常很小，数值见 `report.json` 的 `geometry.flat_detector`。误差是倾斜本身的代价：比较的是
   `.poni` 和由它换算的平板几何，第 3 步对称轴移动光束中心不改变它（公开 AgBh 帧：0.08 %；拿移动后的中心比较会得到 4.1 %）。
   Decisions 还总有两行说明数值是什么：`q`（运动学值；折射主要移动 q_z，Yoneda 带的最多，量级为 k·sin αi，并给出这一帧的数）
   和 `uncertainties`（切线的 σ 只是统计误差，写出误差模型；快速拟合不给 R、h、D 的 ±）。
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
5. **特征长度 2π/q\***（`in_plane_spacing`；报告里的一节是 “Length scale 2π/q\*”，决策项是 `length scale`）：
   I(qy) 在 qy = 0 之外的极大（没有时用斜率变化，即 shoulder）给出 2π/q*。报告写成 “a length scale, not necessarily
   the particle spacing; check it with a model fit”（决策 `length scale` 写出位置，理由只说一次它是什么）：形状因子的拐折或直射束拖尾也会形成 shoulder。BornAgain 模拟的
   GALAXI 模型（真实峰间距 53.6 nm）上 shoulder 给出 73.9 nm，去掉干涉的同样颗粒也有 shoulder（60.6 nm）。
   转述时**不要**把它叫作间距。记录里的键仍是 `spacing`、`distance_nm`。
6. **拟合**（`fit_horizontal_cut`，用 Fitting 的数值物理拟合）：球、竖直圆柱、随机取向圆柱，含尺寸分布和类晶距离 D；
   也会从第 5 步的 2π/q* 出发。最多给出 5 个互不相同的解：report.md 和 fit_solutions.csv 全部列出，而在 MCP 或 call 里
   直接调用 `fit_horizontal_cut` 只返回前 3 个，外加 `chosen`。停在搜索边界上的非线性参数列在每个解的 `at_bounds` 里
   （表里写 “D at its upper bound”），它们不由曲线确定。报告的解（`chosen`）是 χ²/N 在第一个解的 10 % 以内、
   已收敛且没有非线性参数处于边界的排名最靠前的一个；都碰到边界时报告第一个，Needs attention 里有一条 “fit” 说明。
   振幅、背景 BG 和直射束拖尾振幅 A 按非负最小二乘求解，不算边界：等于 0 时 Decisions 有一条 `fit caveat`
   （“background BG at 0, the non-negativity limit”；A = 0 时 σ_Res、ν_Res 不影响曲线）。报告的解已收敛但
   χ²/N > 10（与 Fitting 序列 “拟合差” 的 χ²ᵣ 限值相同）时 Needs attention 里有一条 “fit”：模型远离曲线，R、D 不是测量值
   （公开 AgBh 标样：χ²/N = 472，是布拉格环而不是颗粒）。
   这是筛选，不是尺寸测量（手册第 13 章）。`--no-fit` 只准备切线，不拟合。

输出：`report.md/json`、`horizontal.csv`、`vertical.csv`、`fit_curve.csv`（q、I、σ、I_fit：报告的那个解）、`fit_solutions.csv`
（每个解的每个组分一行：R、h、D 单位 nm；sigma_R、sigma_h、sigma_D 是相对宽度 σ/值，没有单位；`at_bounds` 写
`D:upper` 这样处于边界的数值，`chosen` 标出报告的解）和 `qmap.png`。
给了 `.poni` 的一帧约 25 秒。

## 该怀疑的地方

| 看什么 | 为什么 | 怎么做 |
|---|---|---|
| 模型的选择 | 解按稳健的失配量排序（σ 归一化残差 r 的平均 Huber 损失，δ = 2），表里的 χ²/N 是 r² 的平均，没有自由度修正（不同于 Fitting 的 χ²ᵣ）。几个解的 χ²/N 相差 10 % 以内时，曲线本身定不了模型。所以 Needs attention 里的 “model” 是**正常结果**，不是失败。BornAgain 模拟的 14 条球切线上，快速拟合只有 1 次把球排第一，最好的球解 R 中位偏大 11 %（−29 % 到 +26 %），D 没有还原出来 | 用户给了材料或形状就按它选；在 Fitting 里精修并比较拟合曲线和残差。D 与 2π/q* 接近不是证据 |
| 停在搜索边界上的数值 | `at_bounds` 里的数值（例如 D = 500 nm 的上界）由边界而不是数据决定；排第一的解常常这样 | 转述报告的 `chosen` 解；第一个解处于边界时要说出来，不要引用边界上的值 |
| 报告的解没有粒子间关联 | D ≥ 250 nm（搜索上限 500 nm 的一半）说明该解在 q 范围内没有粒子间关联 | 不要把这样的 D 当成测得的距离 |
| 切线位置 | Yoneda 带之外的切线会混进别的散射 | `get_curve vertical` 看 Yoneda 带和镜面峰的位置是否与 αi、λ 一致；`set_gisaxs_cuts` 换位置 |
| 两半的决定 | 遮挡、吸收体或真实的面内各向异性都会让两半不同 | 读报告里的理由；必要时在 Fitting 里分别拟合两半 |
| 垂直切线 I(qz) | 已导出（`vertical.csv`），但没有分析（膜厚、Kiessig 条纹、Yoneda 带与 αc 是否吻合） | 问题需要时自己分析，并写出依据 |
| 不同 qz 的切线 | shoulder 位置随 qz 移动 → 更像球的形状因子；不随 qz 变 → 面内结构（干涉，或竖直圆柱的横向形状因子），仍不等于间距 | 在 Yoneda 带和镜面峰之间再切一条比较；间距要由模型拟合判断 |
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

- Yoneda 带在第 605–609 行；对称轴在 597.26 px；两半取平均（到 |qy| = 0.179 Å⁻¹，之后 qy < 0 一半单独到 0.278 Å⁻¹；
  有 15 个点落在另一半的缺口里，只来自一半）；
- shoulder 在 |qy| = 0.0113 Å⁻¹，2π/q* = 55.6 nm（特征长度；接近模型的 53.6 nm 只是巧合，去掉干涉的模拟也有 shoulder）；
- 排第一的随机圆柱解 D 在搜索上界（500 nm）、σh/h 也在上界，第二个解的 σh/h 在下界；
- 报告的解（`chosen`，第 3 名）是球：R = 9.44 nm、D = 48.3 nm、χ²/N 1.36（第一个解 1.26 的 1.1 倍以内）。
  R 不能直接和模型的 5.75 nm 比，因为模型是对数正态分布（σ 0.4），GIMaP 用的是高斯 σR/R，宽分布下散射偏重大颗粒；
  模拟上快速拟合的球 R 也系统偏大（手册第 13 章）。

这个例子正好说明：选模型要看用户知道的形状，并在 Fitting 里确认；不能只看排序或 χ²/N，也不能把 2π/q* 当间距。
