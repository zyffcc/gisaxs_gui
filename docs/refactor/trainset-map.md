# Trainset 边界调整地图

## 当前迁移结果

Trainset 的 domain/application/infrastructure/presentation 边界已经落地：

- stable plugin definitions、geometry、parameter rules、physical background 和 model contract
  属于 domain；presentation 通过 application `TrainsetUiCatalog` 获取展示清单，不直接读取
  domain registry；
- preview、what-if、design、configuration、simulation orchestration、job package、local process、
  remote Slurm、metrics 和 model registration 均有 application port/use case；
- YAML/JSON、dataset generation、grid cache、QProcess、remote transfer、portable package 与 Keras
  model building 属于 adapters；
- `BornAgainSimulator` 仅在 composition/worker adapter 中创建，application 只依赖
  `SimulationPort`；
- feature-owned ViewModel/page/ViewBinding 负责 UI state 与展示，旧 controller 名称和顶层
  `trainset` 包只保留 Qt/import/monkeypatch 兼容路径。

当前 focused Trainset 基线为 `50 passed`；完整仓库基线见
[`remaining-work.md`](remaining-work.md)。

## 当前模块归属

| 现有模块 | 主要职责 | 目标边界 | 本阶段处理 |
| --- | --- | --- | --- |
| `trainset/geometry.py` | ROI→球面角、q vectors | domain | 实现迁入 `features/trainset/domain`，旧名 re-export |
| `trainset/plugins.py` | 稳定 plugin definitions/registry | domain | 实现迁入 domain，旧名 re-export |
| `trainset/config.py` | 默认值、同步、验证、YAML/JSON I/O | infrastructure | 实现迁入 `adapters/configuration.py`；旧模块为 alias，格式保持 |
| `trainset/generator.py` | 参数采样、预处理、模拟调度、HDF5 | infrastructure | 实现迁入 `adapters/dataset_generator.py`；只接收 simulation port |
| `trainset/simulation.py` | 多组分模拟调度、structure factor | application | 实现迁入 `application/simulation.py`；不导入 concrete integration |
| `trainset/grid_cache.py` | form-factor cache 与插值 | infrastructure | 实现迁入 `adapters/grid_cache.py`；cache build 显式接收 port |
| `trainset/modeling.py` | Keras 模型构造 | infrastructure | 迁入 Keras adapter，旧名 re-export |
| `trainset/backends.py` | local/Slurm process 与传输 | infrastructure | 实现迁入 `adapters/job_backends.py`；旧模块为 alias |
| `trainset/job_package.py` | 自包含作业文件写入 | infrastructure/composition | 实现迁入 `adapters/portable_job_package.py`；生成脚本显式装配 BornAgain adapter |

## 依赖规则

- domain 允许 stdlib、NumPy；禁止 Qt、BornAgain、TensorFlow、文件系统实现。
- application generation use case 只依赖 `DatasetGenerationPort`、`TrainsetConfigRepository` 与 `SimulationPort`。
- `BornAgainSimulator` 只能在 GUI/CLI composition root 创建，随后以 `SimulationPort` 注入。
- `trainset/generator.py`、`trainset/simulation.py`、`trainset/grid_cache.py` 不得 import BornAgain integration 或 `bornagain` 包。
- exported job script 是独立 composition root，可以创建 `BornAgainSimulator`，但实际 `import bornagain` 仍仅发生在 integration worker process。

## 保持不变

- 参数名称、范围同步、随机种子、Latin hypercube/grid/random sampling、physical constraints。
- ROI orientation、q-space 几何、mask/preprocessing、structure factor、grid cache key 与 HDF5 schema。
- 项目 YAML/JSON 与 job package 文件名、命令行参数、Slurm 行为。
- BornAgain 24.1 数值由原 `api_24_1` worker binding 继续提供。

## 后续兼容层

旧 `trainset` 顶层包继续是公共入口，但所有模块均只做 re-export 或模块 alias。模块 alias
特意保留 `trainset.grid_cache` 等旧 monkeypatch/import 语义。Portable job 同时复制顶层兼容包
与 `src/gimap`，因此已生成脚本使用的模块名、CLI 参数和部署结构不变；新代码直接依赖 feature
application/domain/infrastructure owner，不反向导入顶层包。
