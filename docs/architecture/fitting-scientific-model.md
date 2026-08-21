# Fitting 科学模型与数值契约

- **Status**：Current
- **Scope**：Fitting 一维散射模型、分量累加、q 单位和绘图数据对齐
- **Related code**：`src/gimap/features/fitting/domain/scattering_model.py`、
  `src/gimap/features/fitting/domain/curve_transformations.py`、
  `src/gimap/features/fitting/presentation/bindings/detector_display.py`
- **Related tests**：`tests/test_fitting_domain_scattering_model.py`、
  `tests/test_fitting_curve_rendering.py`
- **Last verified**：2026-08-20

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

## 修改门禁

任何影响本文内容的修改必须同时满足：

- 说明是 bug fix、参数语义变更还是新科学模型；
- 使用固定输入验证 `Total = BG + Resolution + Σ Particle`；
- 验证 `q`、实验强度、模型强度和 branch metadata 长度及顺序一致；
- 对现有可信数据做修改前后数值比较；
- 科学行为确实改变时，不得伪装成 UI 或结构整理。
