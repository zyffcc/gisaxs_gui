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
   Decisions 里有一行 “calibration found by the search”，写出它的完整路径和位置：先确认它属于这次测量。它不在帧所在的
   文件夹（或其子文件夹）里时，同一项也列在 Needs attention 里（退出码 2）；`.poni` 点名的探测器尺寸与帧不同时不采用
   （“the calibration does not match this frame”）；对称轴离标定中心超过 100 px 时不应用并提问。`--incidence-deg` 必须在
   (0°, 90°) 内，`--calibration` 的文件必须存在，否则以退出码 1 结束。CSV 的列名是 ASCII：`qy (A^-1), I, sigma`。
   αi 的来源顺序同 GIWAXS（选项、笔记、分析命令栏里设定的 αi、仪器配置）；之前已用 `set_incidence_angle` 或在命令栏设定的
   αi 会被采用，不再提问。都没有 αi 时先用 0°，并在 Needs attention 里提问（退出码 2）。运行中没有人可问，由调用它的 agent 补
   `--incidence-deg` 后重跑。`.poni` 的探测器倾斜（`Rot1`、`Rot2`、`Rot3`）是几何的一部分：每个像素放在 pyFAI
   放的位置，q 是倾斜探测器的值。Decisions 里有一行 “detector tilt”，Geometry 一节写出倾斜角，数值在 `report.json`
   的 `geometry.tilt_deg` 和 `geometry.rotations_deg`；第 3 步对称轴移动光束中心时倾斜保留。几何标定不拟合倾斜，
   `.poni` 的 `orientation` 项不读。
   Decisions 还总有两行说明数值是什么：`q`（运动学值；折射主要移动 q_z，Yoneda 带的最多，量级为 k·sin αi，并给出这一帧的数）
   和 `uncertainties`（切线的 σ 只是统计误差，写出误差模型；R、h、D 的区间是给定这些 σ 的 Predict parameters 后验）。
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
6. **拟合**（`fit_horizontal_cut`，用 Fitting 的 Predict parameters，即 `create_curve_fit()`，约 15 秒）：GIMaP 的网络给出组成
   和数值，每个解用最小二乘精修并按后验概率加权。没有给组成（`components`）时，网络在一种或两种球和竖直圆柱、带或不带次晶
   的组成中选；随机取向圆柱只在给定组成时才用。已知波长、入射角和 Yoneda 带出射角，并且会话或元数据给出基底时，使用第四层的四项 DWBA。
   可用 `--substrate Si` 或在笔记中单独写一行 `基底 = Si`。基底未知、两个角度已知时，用切线真空 q_z 处的 Born 项（第二层）；任一角未知时用有效界面（第一层，q_z = 0）。第 5 步的 2π/q* 不作起点。最多给出 5 个
   互不相同的解，最可能的在前：report.md 和 fit_solutions.csv 全部列出，而在 MCP 或 call 里直接调用 `fit_horizontal_cut`
   只返回前 3 个，外加 `chosen`。报告的解（`chosen`）就是最可能的那个，带它的概率和 R、h、D 的 68 % 后验区间（决策
   `fit intervals`；区间不含几何和模型选择）。另一个解的概率 ≥ 0.2（`ASK_PROBABILITY`）时，Needs attention 里有一条 “model”：
   曲线本身定不了模型。停在范围边界上的非线性参数列在每个解的 `at_bounds` 里（表里写 “D at its upper bound”），它们不由
   曲线确定；报告的解有这样的值时 Needs attention 里有一条 “fit”。振幅、背景 BG 和直射束拖尾振幅 A 按非负最小二乘求解，
   不算边界：等于 0 时 Decisions 有一条 `fit caveat`（“Background at 0, the non-negativity limit”；Tail A = 0 时 Tail w、Tail ν
   不影响曲线）。报告的解已收敛但 χ²ᵣ > 10（与 Fitting 序列 “拟合差” 的 χ²ᵣ 限值相同）时 Needs attention 里有一条 “fit”：
   试过的组成都描述不了这条切线，R、D 不是测量值（粉末或标样的布拉格环会这样）。切线在 Yoneda 带上且知道 αi 时，切线的
   q_z = k(sin αi + sin αf) 及角度、波长记在 `gisaxs.fit.cut_qz`；未知时该项为 `null`。`gisaxs.fit.substrate_level` 和 Decisions 均写明拟合层级。
   NeXus 文件中的基底材料不受支持或薄膜参数不完整时，`gisaxs.fit.substrate_metadata_warning` 和 Decisions 写明原因，采用与已知角度对应的层级。
   当前条目下 `sample/substrate/top_layer` 需给出 `delta`、`beta`，以及 `thickness_nm` 或带长度单位的 `thickness`（用户手册 5.5 节）。
   在第二层，球或随机取向圆柱的 q_z·R > 1.2 时，`qzR_cov` 提示区间可能低估不确定度；q_z·R > 2 时，`qzR_fail` 提示模型误差较大。
   第二层的竖直圆柱在 q_z·R > 2 时还会标记 `cyl_sphere`，即使没有高度，也要考虑半径相近的球。转述 R 时要说明这些标记。
   基底未知时，可在 Fitting 的预测参数中填入 αi、αf、λ，以第三层拟合有效反射系数。第三层在 q_z·R > 1.7 时也会给出覆盖率提示。
   第四层按裸界面的菲涅耳系数或分层基底的 Parratt 系数计算反射项，不产生 Born 近似标记。
   网络只凭曲线选组成，所以要用用户知道的形状在 Fitting 里确认（手册第 13 章）。`--no-fit` 只准备切线，不拟合。

输出：`report.md/json`、`horizontal.csv`、`vertical.csv`、`fit_curve.csv`（q、I、σ、I_fit：报告的那个解）、`fit_solutions.csv`
（每个解的每个组分一行：概率；R、h、D 单位 nm；sigma_R、sigma_h、sigma_D 是相对宽度 σ/值，没有单位；R、h、D 的 68 %
区间；`at_bounds` 写 `D:upper` 这样处于边界的数值，`chosen` 标出报告的解）和 `qmap.png`。
拟合一步约 15 秒。

## 该怀疑的地方

| 看什么 | 为什么 | 怎么做 |
|---|---|---|
| 模型的选择 | 解按后验概率排序，表里的 χ²ᵣ 与 Fitting 一样，用拟合点上 σ 归一化残差的平方和除以自由度。另一个解的概率 ≥ 0.2 时曲线本身定不了模型，所以 Needs attention 里的 “model” 是**正常结果**，不是失败。即使一个模型占了几乎全部概率，也要留意形状混淆：公开的 GALAXI 银球帧在第二层拟合中给出一群球和一群竖直圆柱（见下） | 用户给了材料或形状就按它选：在 Fitting 里用 Predict parameters 的 *The particles in Model* 拟合那个模型，再精修并比较拟合曲线和残差。D 与 2π/q* 接近不是证据 |
| 停在范围边界上的数值 | `at_bounds` 里的数值由边界而不是数据决定 | 转述报告的 `chosen` 解；它处于边界时要说出来，不要引用边界上的值 |
| 报告的解没有粒子间关联 | D ≥ 250 nm 说明该解在 q 范围内没有粒子间关联（Decisions 的 `fit caveat`） | 不要把这样的 D 当成测得的距离 |
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

用上述命令得到的第二层结果：

- Yoneda 带在第 605–609 行；对称轴在 597.26 px；两半取平均（到 |qy| = 0.179 Å⁻¹，之后 qy < 0 一半单独到 0.278 Å⁻¹；
  有 15 个点落在另一半的缺口里，只来自一半）；
- shoulder 在 |qy| = 0.0113 Å⁻¹，2π/q* = 55.6 nm（特征长度；接近模型的 53.6 nm 只是巧合，去掉干涉的模拟也有 shoulder）；
- 报告的解（`chosen`）为带次晶的球和竖直圆柱，R 分别为 3.28 和 5.99 nm，球的 D = 48.9 nm。χ²ᵣ = 1.05，598 个点中 596 个参与拟合。
  在 q_z = 0.5226 nm⁻¹，报告标记球半径的区间覆盖率（`qzR_cov:R_1`），并提示竖直圆柱可能代表半径相近的球（`cyl_sphere:R_2`，q_z·R = 3.13）。

这个例子正好说明：选模型要看用户知道的形状（这里是球），并在 Fitting 里确认；不能只看概率或 χ²ᵣ，也不能把 2π/q* 当间距。
