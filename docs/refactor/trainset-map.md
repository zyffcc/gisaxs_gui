# Trainset 边界调整地图

## 当前模块归属

| 现有模块 | 主要职责 | 目标边界 | 本阶段处理 |
| --- | --- | --- | --- |
| `trainset/geometry.py` | ROI→球面角、q vectors | domain | 实现迁入 `features/trainset/domain`，旧名 re-export |
| `trainset/plugins.py` | 稳定 plugin definitions/registry | domain | 实现迁入 domain，旧名 re-export |
| `trainset/config.py` | 默认值、同步、验证、YAML/JSON I/O | legacy mixed | 新 application config repository seam；旧格式保持 |
| `trainset/generator.py` | 参数采样、预处理、模拟调度、HDF5 | legacy mixed/application facade | 去除 concrete BornAgain；新增 generation port adapter |
| `trainset/simulation.py` | 多组分模拟调度、structure factor | application/domain seam | 只接受 `SimulationPort`，不导入 integration |
| `trainset/grid_cache.py` | form-factor cache 与插值 | infrastructure | cache build 显式接收 `SimulationPort` |
| `trainset/modeling.py` | Keras 模型构造 | infrastructure | 迁入 Keras adapter，旧名 re-export |
| `trainset/backends.py` | local/Slurm process 与传输 | infrastructure | legacy 入口暂留 |
| `trainset/job_package.py` | 自包含作业文件写入 | infrastructure/composition | 生成脚本显式装配 BornAgain adapter |

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

旧 `trainset` 顶层包继续是公共入口；geometry/plugins/modeling 当前为薄 re-export。`config.py`、`generator.py`、cache/backends/job package 尚未整体搬移，避免破坏已生成作业和外部脚本；新调用应优先经过 feature application API。
