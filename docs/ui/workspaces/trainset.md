# Trainset 布局迁移记录

## 控件映射

| 迁移前控件/区域 | 迁移后位置 | 行为 |
| --- | --- | --- |
| real scattering reference | `Input / ParameterSection` | reference 仍只用于几何、ROI 和 mask 指导，不作为模拟训练图像 |
| beam/detector、ROI、particle population、dataset sampling | `Configure / ParameterSection` | 字段 path、默认值、单位和采样含义不变 |
| mask、interference、layers/substrate | `Advanced configuration / AdvancedSection` | 默认折叠；所有原表格、draw action 和 constraints 保留 |
| Full detector/ROI/Masked image/Mask only | `Preview / PlotPanel` | canvas、坐标方向、beam/ROI/mask 交互不变 |
| Local Preview 的 range、coverage、三种 update action | `Run / ParameterSection` | BornAgain cache 和 realization 行为不变 |
| background、noise、mask & transforms | `Advanced preprocessing / AdvancedSection` | preprocessing 顺序、range 和 enabled state 不变 |
| Range impact/Pipeline stages/Diagnostics | `Preview / PlotPanel` | 三组比较、histogram、coverage 和 readiness table 不变 |
| training controls | `Model Design > Configure / ParameterSection` | batch、epoch、optimizer、learning rate、scheduler 不变 |
| layer editor | `Model Design > Advanced model architecture / AdvancedSection` | layer 类型、顺序和 tensor contract 不变 |
| model contract、forward-pass validation | `Preview` 与 `Run` sections | 原 validation command 不变 |
| Local/Maxwell tabs | `Run / ParameterSection` | local process、package generation 和 disabled Maxwell submit 行为不变 |
| local activity/progress | shared `JobStatus` | controller 的旧 label/progress 接口通过别名保留，百分比仍为 0–100 |
| package manifest/tree | `Export / ParameterSection` | package 路径、manifest 和 Slurm 文件不变 |
| process output | `Monitor & Results > Log / AdvancedSection` | 原 append、refresh 和 sync 路径不变 |
| metrics/register model | `Monitor & Results > Results / ParameterSection` | metric columns 和 model registration 不变 |

Trainset 原有五步导航继续有效；本轮只在各步骤内部建立 Input → Configure → Preview →
Run → Results → Export 的统一信息层级。现有 legacy controller 仍是兼容桥，没有新增第二层
workflow orchestration，也没有改动 BornAgain、preprocessing、generation 或 training 算法。

## 手动验收清单

- [ ] 加载 reference 后 detector preview、beam center pick、ROI draw 正常；
- [ ] Basic 与 Advanced 折叠/展开不重置 detector、mask、particle、layer 或 dataset 值；
- [ ] 四个 design preview tabs 的方向、坐标和 mask overlay 正确；
- [ ] Local Preview 可更新比较、强制重算 BornAgain、刷新 noise/mask realization；
- [ ] Advanced preprocessing 的背景、Gaussian、Poisson、mask、log、normalize、crop 顺序不变；
- [ ] Range impact、Pipeline stages、Diagnostics 均能显示原结果；
- [ ] Model layer 增删、排序和 forward-pass validation 正常；
- [ ] Prepare local package、small physical test、full generation、training、I/O smoke test 正常；
- [ ] JobStatus 在 running、paused、succeeded、failed 时显示正确，Pause/Stop safely 仍可用；
- [ ] package tree、job log、metrics table 和 Register best model 正常；
- [ ] Save/Load、Remember changes、Reset defaults 和 step readiness 状态不变；
- [ ] Maxwell 远端提交仍按原设计保持禁用，package export 可用。
