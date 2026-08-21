# Trainset 界面与交互说明

- 状态：Trainset 页面和交互画布由 Trainset feature 拥有；顶层 UI 路径仅为 public alias。
- 当前调用链：`TrainsetBuildPage → TrainsetViewBinding → TrainsetViewModel → application use cases/ports`。
- Python workflow shell：`src/gimap/features/trainset/presentation/views/page_view.py`。
- Dataset Design 布局：`src/gimap/features/trainset/presentation/views/dataset_page_view.py`。
- Local Preview、Model Design、Local Run、Monitor & Results 布局分别位于
  `preview_page_view.py`、`model_page_view.py`、`run_page_view.py`、
  `monitor_page_view.py`。
- 页面行为与动态 workspace panels：`src/gimap/features/trainset/presentation/page.py`。
- 兼容入口：`ui/trainset_build_page.py`。
- 最近验证：2026-08-19。

## 当前现代化工作流

Trainset 保留五个具有明确产物的步骤：

```text
1 Dataset Design → 2 Local Preview → 3 Model Design → 4 Local Run → 5 Monitor & Results
```

第一步不再把 detector、particle 和 sampling 参数铺成一条超长表单，而是在同一
Configure section 内分为 `Geometry + ROI`、`Particle population` 和
`Sampling + files`。右侧 detector/ROI/mask preview 始终保留。底部 action bar 只显示
当前步骤相关的全局动作；Load/Save 始终可用，Validate design 和 Prepare job package
分别只在需要的步骤出现。字段、默认值和命令实例未复制。

`TrainsetViewBinding` 只负责页面 signal、控件值映射、文件对话框和结果渲染；配置、preview、
simulation、job package、local/Slurm workflow 和模型注册均经 `TrainsetViewModel` 调用
application use cases/ports。旧 `TrainsetController` 名称只 re-export 同一个 binding，不存在
第二套页面或运行时实现。

`Ui_MainWindow` 只保留无业务控件的 `trainsetBuildPage` host。启动时不再创建随后会被
隐藏的 2,022 行旧 beam、detector、particle、preprocessing 和 generation widgets；app
composition root 把唯一的 feature-owned 页面装入 host，并通过构造函数注入
`TrainsetViewBinding`。Binding 强制要求页面和 ViewModel 依赖，不再包含 host/layout fallback、
页面创建或兼容 widget 清理逻辑。

Physical-background 参数定义由 trainset domain 单一拥有；public alias `trainset.config` 与
feature presentation 引用同一个对象，因此参数键、范围、精度和帮助文本保持不变。

## 控件映射

| 功能/控件区域 | 当前位置 | 行为 |
| --- | --- | --- |
| real scattering reference | `Input / ParameterSection` | reference 仍只用于几何、ROI 和 mask 指导，不作为模拟训练图像 |
| beam/detector、ROI | `Configure > Geometry + ROI` | 字段 path、默认值、单位和几何含义不变 |
| particle population | `Configure > Particle population` | form factor、分布、constraints 和参数表不变 |
| dataset sampling/files/split | `Configure > Sampling + files` | 样本数、shard 和 train/validation/test 语义不变 |
| mask、interference、layers/substrate | `Advanced configuration / AdvancedSection` | 默认折叠；所有原表格、draw action 和 constraints 保留 |
| Full detector/ROI/Masked image/Mask only | `Preview / PlotPanel` | canvas、坐标方向、beam/ROI/mask 交互不变 |
| Local Preview 的 range、coverage、三种 update action | `Run / ParameterSection` | BornAgain cache 和 realization 行为不变 |
| background、noise、mask & transforms | `Advanced preprocessing / AdvancedSection` | preprocessing 顺序、range 和 enabled state 不变 |
| Range impact/Pipeline stages/Diagnostics | `Preview / PlotPanel` | 三组比较、histogram、coverage 和 readiness table 不变 |
| training controls | `Model Design > Configure / ParameterSection` | batch、epoch、optimizer、learning rate、scheduler 不变 |
| layer editor | `Model Design > Advanced model architecture / AdvancedSection` | layer 类型、顺序和 tensor contract 不变 |
| model contract、forward-pass validation | `Preview` 与 `Run` sections | 原 validation command 不变 |
| Local/Maxwell tabs | `Run / ParameterSection` | local process、package generation 和 disabled Maxwell submit 行为不变 |
| local activity/progress | shared `JobStatus` | binding 的 label/progress 映射保持百分比 0–100 |
| package manifest/tree | `Export / ParameterSection` | package 路径、manifest 和 Slurm 文件不变 |
| process output | `Monitor & Results > Log / AdvancedSection` | 原 append、refresh 和 sync 路径不变 |
| metrics/register model | `Monitor & Results > Results / ParameterSection` | metric columns 和 model registration 不变 |

Trainset 原有五步导航继续有效；当前页面在各步骤内部建立 Input → Configure → Preview →
Run → Results → Export 的统一信息层级，并用 contextual action bar 强化当前阶段。顶层
controller 文件仅为 public alias，没有第二层 workflow orchestration。页面布局不改动 BornAgain、preprocessing、generation、
training、module.yaml、project YAML/JSON、HDF5 schema 或模型格式；所有 objectName、signals、
快捷键、错误行为和内嵌样式保持不变。

页面顶层标题、项目名、验证状态、五步导航、五个固定 page host 和底部 action bar 现在由
`views/page_view.py` 单一维护，`TrainsetBuildPage` 注入各步骤内容，不再保留顶层
`_build()` 实现。Dataset Design 的 Input / Configure / Advanced / Preview 容器和 splitter
也已迁入独立 Python View。其余四个步骤也各有独立 View，负责 section hierarchy、滚动区、
固定表格、按钮和语义 host。catalog/plugin 参数字段、交互画布、BornAgain 比较和任务状态组件
仍按运行时数据注入；这些动态 presentation 不复制到另一套静态实现。

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
- [ ] 三个 Dataset Design 配置页签切换不重置任何字段；
- [ ] 每个步骤只突出与当前阶段相关的全局动作；
- [ ] Maxwell 远端提交仍按原设计保持禁用，package export 可用。
