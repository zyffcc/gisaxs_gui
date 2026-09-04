# Fitting 科学模型与数值契约

- **Status**：Current
- **Scope**：Fitting 一维散射模型、分量累加、q 单位和绘图数据对齐
- **Related code**：`src/gimap/features/fitting/domain/scattering_model.py`、
  `src/gimap/features/fitting/domain/curve_transformations.py`、
  `src/gimap/features/fitting/domain/manual_refinement.py`、
  `src/gimap/features/fitting/presentation/bindings/detector_display.py`、
  `utils/ML_Fitting_1D_GISAXS/Training/prediction_curve_io.py`
- **Related tests**：`tests/test_fitting_domain_scattering_model.py`、
  `tests/test_fitting_curve_rendering.py`、
  `tests/test_fitting_domain_constraints_scoring.py`、
  `tests/test_fitting_domain_manual_refinement.py`、
  `tests/test_ai_curve_loader.py`
- **Last verified**：2026-09-04

本文是 GIMaP Fitting 科学模型的权威说明。修改公式、参数顺序、单位、采样方式、分量缩放
或 q–intensity 对齐行为前，必须同步更新本文并增加固定数值回归测试。

## 单位与参数顺序

模型内部统一使用 `q` 的 `nm⁻¹` 数值。输入为 `Å⁻¹` 时：

```text
q_model[nm⁻¹] = 10 × q_source[Å⁻¹]
```

每个粒子的参数按选择顺序排列，最后接全局参数：

```text
Sphere:            Int, R, sigma_R, D, sigma_D
Cylinder:          Int, R, sigma_R, h, sigma_h, D, sigma_D
Vertical Cylinder: Int, R, sigma_R, D, sigma_D
Global:            BG, sigma_Res, nu_Res, int_Res, k
```

## 总强度与分量累加

当前混合模型为：

```text
I_model(q) = BG
             + K(k) × [Σᵢ Intᵢ Pᵢ(q) Sᵢ(q; Dᵢ, sigma_Dᵢ)
                        + int_Res R(q; sigma_Res, nu_Res)]
```

其中兼容缩放函数为：

```text
K(k) = k,  k > 0
K(k) = 1,  k is None or k ≤ 0
```

因此背景 `BG` 不乘 `k`；每个粒子分量和 resolution 分量都乘相同的 `K(k)`。展示给用户的
分解必须满足以下逐点恒等式：

```text
Total = BG_total + Resolution + Σ Particleᵢ
```

禁止为红色总曲线和虚线分量分别实现两套累加逻辑。`mixed_model_components()` 暴露的
`total` 是绘图和分解校验的唯一总曲线来源。

## Resolution 分量

```text
R(q; sigma_Res, nu_Res)
    = 1 / [1 + (|q| / sigma_Res)^nu_Res]

I_resolution(q)
    = K(k) × int_Res × R(q; sigma_Res, nu_Res)
```

当 `sigma_Res ≤ 0` 或 `nu_Res ≤ 0` 时，当前兼容行为令 `R(q)=1`；当 `int_Res=0` 时，
resolution 分量为零。这里的 Resolution 是一个加性分量，不是对粒子曲线执行的高斯卷积。

## Sphere form factor

令 `x=qR`，归一化球振幅为：

```text
F_sphere(q, R) = 3 [sin(x) - x cos(x)] / x³
F_sphere(0, R) = 1
```

半径多分散性使用截断到 `R≥0` 并归一化的高斯采样：

```text
P_sphere(q) = Σⱼ wⱼ F_sphere(q, Rⱼ)²
```

## Cylinder form factor

对圆柱轴与 `q` 的夹角 `alpha`：

```text
F_cylinder(q; R, h, alpha)
    = [2 J₁(qR sin(alpha)) / (qR sin(alpha))]
      × sinc(qh cos(alpha) / 2)

sinc(x) = sin(x) / x
```

`R`、`h` 分别按截断高斯采样，随机取向采用 `sin(alpha)` 权重：

```text
P_cylinder(q) = ⟨F_cylinder²⟩_(R,h,alpha)
```

当前采样点数和截断范围属于数值定义的一部分，不能作为普通性能优化静默改变。

## Vertical Cylinder form factor

当前 qz=0 参考实现为：

```text
P_vertical(q)
    = 10⁻⁶ Σⱼ wⱼ [Rⱼ J₁(qRⱼ) / q]²
```

该分量当前把 `sigma_R` 解释为相对宽度，并使用 `R × sigma_R` 作为高斯采样的绝对标准差。
这是现有参数语义，除非有独立科学变更任务和回归基线，否则不得与 Sphere/Cylinder 的
绝对宽度语义合并。

## 一维结构因子

当 `D=0` 或 `sigma_D=0` 时，结构因子关闭并令 `S(q)=1`。否则：

```text
phi(q) = exp[-pi q² sigma_D²]

S(q) = (1 - phi²)
       / [1 + phi² - 2 phi cos(qD)]
```

分母绝对值小于 `1e-15` 时使用 `1e-15`，这是当前数值稳定约定。

## q–intensity 对齐契约

所有曲线数组都必须被视为点对：

```text
(q[0], I[0]), (q[1], I[1]), …, (q[n], I[n])
```

过滤、正负分支选择、fold、排序、ROI 和删除点必须对 `q`、`I` 以及 source-branch metadata
应用同一个索引。禁止：

- 只排序 `q` 而不以相同顺序排序 `I`；
- 把已经按 prepared q 计算的模型强度重新配到 raw q；
- 把 prepared model intensity 再与 raw data 执行第二次 fold/sort；
- 嵌入图和独立窗口分别重新处理科学数组。

正确的数据流是：

```text
raw paired curve
    ↓ one paired preparation (branch / fold / sort / ROI)
prepared q + measured I + source sign
    ├─→ evaluate every model component on this exact prepared q
    └─→ build one CurvePlotSpec
             ├─→ embedded curve view
             └─→ independent curve window
```

Fold overlay 可以让 `+q` 与 `−q` 共享相同的 `|q|` 横坐标，但必须保留 source sign 供颜色、
导出和诊断使用。对于仅依赖 `|q|` 的当前模型，相同 `|q|` 上的正负分支模型值必须相等。

## AI proposal 与传统优化交接

`Fast Predict` 只执行神经网络 proposal、后验采样和物理 forward verification，用于给出模型拓扑与
initial parameters；它本身不执行数值精修。其输入源与手动 fitting 完全一致：勾选
`Use current cut` 时只使用当前 cut（原生 nm⁻¹），未勾选时只使用 imported 1D data，并在 AI
preprocessing 后将 Å⁻¹ 转为模型的 nm⁻¹。不得从 `self.q`、旧 ROI 缓存或另一种数据源隐式 fallback。

候选结果同时显示 physics fit likelihood 与 model sampling probability。二者用于排序和诊断，不应
解释为结构真值的校准置信度。选择候选会把其 topology、component 参数与 global 参数加载为当前手动
模型；随后可选择 `Local Refine` 在窄范围 polishing，或选择 `Global Search` 在相同 topology 上进行
宽范围搜索。这样模型负责提出可行 basin/组分组合，传统优化负责实际曲线收敛。

checkpoint 的实际 tensor signature 是 inference contract。旧 production checkpoint 的
`d_spacing_rule` 宽度为 3，当前 schema 宽度为 4；前三个 rule id 语义不变时，只允许裁掉全零的新增
尾列。若用户启用了旧模型不认识的新 rule，必须明确拒绝，不能静默重映射。其他 tensor shape
不匹配同样必须报错。

AI full-profile 的 least-squares 使用数值 Jacobian；一次 scipy function evaluation 会触发约
`n_variables + 1` 次 residual call。stall patience 按估算 function evaluations 计数，而不是按底层
residual calls 计数，并在相同 optimization q grid 上比较初始值和后续最佳值，避免高维问题只做一两
次有效迭代就被误判停滞。最终候选仍在完整 q grid 上重新计分，只有 score 不变差时才接受精修结果。

## Global Search 与 Local Refine

`Global Search` 与 `Local Refine` 是两个明确分开的入口。勾选
`Use current cut` 时输入来自当前 cut；未勾选时只使用已导入的 1D data，不在两者之间隐式
fallback。输入继续使用当前 q branch/fold/average、排除点和 fitting ROI，随后仅保留有限且
强度为正的点；少于 8 个点时不得启动。

目标函数是实验强度与模型强度的 log residual。优化向量只包含弹窗中勾选的参数，其他参数
保持当前值；每个 bounds 必须包含当前值。局部精修的内建默认范围用于 polishing：`R/h/D/k` 为当前值的
±20%，`Int/BG/int_Res` 和各 sigma 为 ±50%，`nu_Res` 为 ±25%，并保持非负物理边界。
零值使用小的非负可编辑窗口，`nu_Res` 下限为 0.1。

局部 least-squares 不直接使用物理数值作为优化坐标，而是把每个选中参数按其 bounds 映射到
`[0, 1]`。因此 `xtol` 衡量的是相对于用户范围的无量纲步长；同时释放 `10⁻⁸` 量级的参数与
`10⁴` 量级的 `k` 时，不会因为原始尺度差异产生假性收敛。流程始终保留已评估的最佳参数，终止点
不得比初始参数更差。

全局搜索使用固定 seed、Latin-hypercube 初始化的 differential evolution 探索非线性形状、尺寸、
structure 和 resolution 参数。对每个候选，`Int_i`、`int_Res` 与 `BG` 的线性幅度先用带 bounds 的
迭代加权线性最小二乘消元；这避免把大量预算浪费在 `k × Int_i` 的退化方向。演化完成后，从 score
最好的若干候选分别执行上述归一化局部精修。跨度达到 100 倍且上下界均为正的范围按对数映射，避免
候选集中在大数值端。默认使用约 16384 次全局 evaluation、3 个局部起点，每个起点最多 80 次 local
nfev；这些预算都可在弹窗中修改。

全局默认 bounds 同时包含当前值并参考实际 `q_model` window：尺寸和分布宽度至少覆盖当前 q window
可辨认的宽尺度范围，`D/sigma_D` 可跨越当前值附近的错误 basin，`nu_Res` 至少覆盖 `0.5–30`；
`Int/int_Res/k` 对非零当前值允许 `10⁻⁴×–10⁴×`，`BG` 上限还参考当前有效强度的下四分位数。
它们是搜索窗口，不是实验先验，用户应按样品知识收紧。

Global 默认勾选所有 component geometry、distribution、structure、resolution 和线性幅度参数，只不
勾选全局 `k`；`k` 与 component intensity 完全相关，不应同时释放。Local 默认仍只勾选 `Int`、
`R/h` 与 `BG`，保持精修语义。
用户修改过的勾选状态可以复用，但只有参数当前值未变化时才复用之前的绝对 bounds；当前值变化
后重新围绕新值生成默认范围。Global 与 Local 分别保存选择和 bounds，避免宽搜索范围误用于精修。

Components 与 Global 数值控件至少保留 12 位小数，足以表达接近零但非零的 structure/resolution
参数。参数编辑追踪必须在 JSON 值加载后建立 baseline；仅打开页面、切换焦点或关闭窗口不得把
高精度参数按控件的临时默认值重新写回。

仓库回归样例 `TestSAXSdata/Cut_Data.txt` 按真实 `Import 1D` 路径读取：源 q 为 Å⁻¹，模型输入转换
为 nm⁻¹（乘 10），并过滤唯一的非正强度点。固定的 3 Sphere 起点 logRMSE 为约 `0.174029`；默认
16384 evaluation、3 个局部起点、每起点 80 nfev 必须降至 `0.05` 以下，当前固定 seed 回归约为
`0.03838`。该真实曲线没有可追溯 parameter ground truth，因此这里只验证 empirical curve fit，不把
得到的多解参数解释为结构真值。这个基准不改变 scattering 公式或参数语义。

## 修改门禁

任何影响本文内容的修改必须同时满足：

- 说明是 bug fix、参数语义变更还是新科学模型；
- 使用固定输入验证 `Total = BG + Resolution + Σ Particle`；
- 验证 `q`、实验强度、模型强度和 branch metadata 长度及顺序一致；
- 对现有可信数据做修改前后数值比较；
- 科学行为确实改变时，不得伪装成 UI 或结构整理。
