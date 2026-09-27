# 输出单位契约 v5-output-r3

这是输出层修订。基础网络、校正网络权重、正演方程、34组合×6头和20次神经校正预算与原版相同。

## 输入和输出单位

| 字段 | 单位或含义 |
|---|---|
| 输入 `q` | nm⁻¹ |
| 输入 `observed` | V5相对强度：整体尺度k固定为1，单组分形状因子F(0)=1 |
| 输入 `sigma` | 观测强度标准差，与observed同单位，不是相对误差比例 |
| `parameters.R/h/D` | nm；不生效的h、D为null |
| `parameters.sigma_R` | 无量纲σR/R，**不是nm** |
| `parameters.sigma_h` | 无量纲σh/h；h未生效时为null |
| `parameters.sigma_D` | 无量纲σD/D；D未生效时为null |
| `absolute_widths_nm.sigma_R/sigma_h/sigma_D` | 上述比例乘相应长度，得到nm标准差；用于解释或对接采用绝对宽度的接口 |
| `weight` | 模型混合权重，总和为1；不是组分概率、质量分数或体积分数 |
| `global_parameters.rho_BG` | 无量纲；BG=rho_BG×有效q点上P的中位数 |
| `global_parameters.sigma_Res` | nm⁻¹，分辨率背景项的q尺度 |
| `global_parameters.nu_Res` | 无量纲指数 |
| `global_parameters.rho_Res` | 无量纲相对幅值系数，不是绝对intRes |

例如R=20、sigma_R=0.15表示平均半径20nm、半径标准差3nm。传回本包V5 forward时保持sigma_R=0.15；不要把3当成相对宽度传入。NPZ中的params/globals仍是[0,1]归一化坐标，不是JSON里的物理值。

正演为 `I=P+intRes*g+BG`，`P=sum(w_i*F_i*S_i)`，`g(q)=1/(1+(q/sigma_Res)^nu_Res)`。`intRes=rho_Res*max(P[first5])/max(g[first5])`，first5指输入有效q点中的前五点。背景和幅值依赖原生q网格，不能无条件作为新网格上的绝对常数复用。分辨率项不启用时，其三个JSON参数为null。

不自动换算q或强度；不能把任意实验计数直接视为符合V5尺度。数值反归一化采用NumPy float64，和旧TF float32输出可能有微小舍入差异；本次与已有JSON的最大相对差约5.09×10⁻⁷。

## r3候选输出规则

单位与r1/r2相同。现在每种组合下可有多套parameter_solutions；原参数不平均、不改写。去重只在同一组合内进行，要求排列不变的活跃参数状态RMS≤1e-7且有效q点最大log曲线差≤1e-5。不同参数但相似曲线不删除。相同粒子的交换顺序、无效参数变化不构成新物理解。

条件模式使用完整组分多重集合；具体命令、层级输出及质量规则限制见README_zh.md。参数候选不是已校准后验样本，也不是穷尽参数空间。物理JSON导出字段沿用已验证的r1契约，本轮另验证新层级下的参数正演往返。
