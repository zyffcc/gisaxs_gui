# 多解反演真实曲线案例清单

- **Status**：pre-freeze working inventory；尚无案例获准作为论文 blind test
- **Scope**：记录已经暴露给开发流程的曲线，以及仍需 provenance / prior-use 审计的本地候选文件
- **Related code**：`utils/ML_Fitting_1D_GISAXS/PosteriorV8/observed_curve_io.py`
- **Related tests**：`utils/ML_Fitting_1D_GISAXS/tests/test_posterior_v8_observed_curve_io.py`
- **Last verified**：2026-09-03

## 已知开发案例

| 文件 | SHA-256 | 点数 | 状态 |
|---|---|---:|---|
| `utils/ML_Fitting_1D_GISAXS/Training/Cut_Data_unmasked.txt` | `f2f6336ad6124438eefe8bf2b76654f4f35ded93305cdcced01ac4243385d8d4` | 963 data rows | `development_seen` |

该 Cut_Data 曲线已经进入 V5.1 开发评估，因此不得重新命名为冻结后的盲测。它仍可用于回归、失败分析和
GUI 演示，但所有结果必须明确标为 development case。

## 本地候选文件：资格未定

| 文件 | SHA-256 | 行数 | 当前限制 |
|---|---|---:|---|
| `TestSAXSdata/real_nano_HR.dat` | `acb4508d502d00e1af8c1476ef282cb5ecfb7b8aebc7c7486dd3ed4c19ee9bdf` | 594 | provenance、测量不确定度和历史使用尚未审计 |
| `TestSAXSdata/Cut_Data.txt` | `eabc32e99e81c75f8523119767e187bf2e4911f56f8544fcda544f7ba7020769` | 51 | 横轴语义、provenance 和历史使用尚未审计 |
| `TestSAXSdata/test_1d_data.dat` | `baec087330ebc288190f15320eb2b16112cb540379fdb44e71c026d78a483d57` | 22 | README 已将其用于旧 CPU benchmark，因此不是未见 blind case |

“资格未定”不表示这些文件会进入论文。必须先由数据所有者补齐样品、仪器、q 单位、强度与误差列、预处理、
许可和既往调参使用记录；任何已用于选择架构、范围、阈值或 checkpoint 的曲线只能进入开发集合。

## 正式 blind case 的冻结条件

在打开曲线内容前发布不可变 inventory 与 SHA，冻结纳入/排除规则、每例用户范围的制定者和依据、方法与候选
顺序随机化、专家评分 rubric、人工 GUI baseline 的操作者/时间/尝试预算，以及失败和 abstention 分母。
若没有可追溯 ground truth，只报告 empirical-forward residual、范围/物理合规、候选多样性与稳定性、运行时间
和盲评一致性，不报告参数准确率或“找回真实结构”。
