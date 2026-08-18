# GIMaP 重构完成状态与兼容层清单

- **状态**：当前架构目标已完成，保留渐进式兼容层
- **审计日期**：2026-08-18
- **范围**：feature ownership、依赖方向、状态注入、外部运行时隔离和 legacy 入口
- **原则**：兼容层不是第二套实现；只有 caller 和外部 import path 都完成审计后才能删除

## 当前结论

目录与依赖所有权已经收口：Format Converter、Calibration、Fitting、Prediction、Trainset、
Classification 和 WAXS 的页面及业务实现均由 `src/gimap` 中对应 owner 持有。顶层
`controllers/`、`ui/`、`calibration/`、`trainset/` 和 `WAXS/WAXS.py` 只保留兼容入口或
设计/数据资产；架构测试禁止新源码反向导入这些兼容包。

科学计算、文件/目录读写、模型发现与加载、BornAgain、TensorFlow、后台进程、参数文件和
持久化均已从 View/ViewModel 的直接职责中移出。新调用链遵循：

```text
PyQt View + feature ViewBinding → ViewModel → Application Use Case → Domain
                                      ↓
                                     Port
                                      ↑
                     Infrastructure Adapter
```

`AppContext` 显式注入 settings、session、user preferences、project parameters 与
`JobRunner`；新 feature 不创建全局 context 或新的全局单例。架构测试禁止新源码反向依赖
legacy compatibility package，也禁止 presentation 直接导入 feature infrastructure/adapter。

## 已完成的关键边界

- **Format Converter**：页面、状态、格式规则、转换 use case 和本地文件 adapter 全部由 feature
  持有；旧 dialog 仅 re-export；核心转换可无 `QApplication` 测试。
- **Calibration**：standard detection、ring geometry、manual refinement、significant-change 和
  scientific kernel 归入 domain/application；图像与 JSON 通过 ports/adapters；旧路径保留。
- **Fitting**：约束、评分、curve/ROI/cut/in-situ 数学、模型与 q-space 调用、文件加载/导出、远程
  cache、参数快照、AI artifacts/catalog、日志、模型参数和依赖探测均已有 command/use case 或
  port/adapter；NXS 序列检查也已通过 repository port；ViewModel 已按 storage、in-situ、
  scientific 职责拆分。
- **Prediction**：模型清单、单/多文件序列规则、图像/mask、array export 与预测工作流已分层；
  文件和导出状态分别由子 ViewModel 持有。
- **Trainset**：配置、sampling、preview、simulation orchestration、grid cache、job package、
  local/Slurm backend、模型 contract/registration 已分层；BornAgain 只通过 `SimulationPort`。
- **Classification**：数据加载、embedding、训练、artifact 保存与 presentation 已隔离；训练通过
  `JobRunner`，application 测试使用 fake classifier。
- **WAXS**：文件加载、geometry、mask、integration、batch、export 和 presentation 已按 map
  迁移；旧 `WAXS/WAXS.py` 是 standalone compatibility launcher。
- **App shell/state**：主窗口、菜单、设置、布局和资源由 app owner 持有；用户偏好与 project
  parameters 通过 repository 注入，现有 JSON 格式保持不变。
- **Python View ownership**：应用外壳、七个 workspace、Trainset 五个步骤和固定辅助窗体共有
  37 个 feature-owned Python View；显式 inventory 与依赖测试保护单一事实来源。
- **Integrations**：BornAgain 和 TensorFlow 延迟到 worker process 加载；缺失、损坏或 worker
  崩溃不会终止 GUI 主进程。

## 当前大型 presentation binding 与兼容层

| 层 | 当前规模 | 保留原因 | 允许的后续处理 |
| --- | ---: | --- | --- |
| Fitting `view_binding.py` | 17,301 行 | widget、Matplotlib、Qt signal 与历史动态 UI 契约；科学与 I/O 已有 seam | 只按有 characterization test 的 UI 状态组逐个拆分 |
| Prediction `view_binding.py` | 3,709 行 | 单/多文件页面渲染与旧 signal/objectName 契约 | 按 module/preview/batch rendering 拆分 |
| Trainset `view_binding.py` | 1,678 行 | 页面映射、dialogs 与任务状态展示 | 保持 JobRunner/port 边界，按 design/local/remote UI 区块拆分 |
| Classification `view_binding.py` | 1,748 行 | 页面渲染、图表和 Qt worker signals | 不得把训练或 artifact I/O 放回 presentation |
| `ApplicationRuntime` | 281 行 | workspace composition、navigation 和启动顺序 | 保持 composition-only；session/parameter coordination 已拆出 |
| 顶层 wrappers/aliases | 通常数行 | 用户脚本、插件、portable job、动态 import 和 monkeypatch 兼容 | 经过 deprecation 与外部 caller 审计后独立删除 |

这些 binding 行数超过 review 阈值，但已经由测试和依赖门禁约束。机械地把同一 Qt 类切成多个
无语义文件不会改善架构，反而会扩大动态 signal 回归风险。因此它们不是本轮必须清零的 P0
架构 violation。生产代码已经不再导入任何 `legacy_bridge` 或顶层 `controllers` 路径；这些
文件现在仅为薄 re-export。

## 仍可单独安排的维护项

以下是可选的 P2 清理，不阻塞当前重构完成状态：

- adaptive stack、range slider、detector trigger 和 parameter trigger 已迁入 app/Fitting
  presentation owner；旧 `utils` 路径现在仅为 5–8 行 re-export。参数触发核心、旧信号兼容和
  diagnostics 已按职责拆分，核心文件为 335 行。
- 菜单视图实现已迁入 feature-free 的 `app/presentation/menu_manager.py`；app 根路径仅负责注入
  Calibration/Format Converter dialog factories，`ui/menu_manager.py` 为兼容导出。所有
  `QFileDialog`/`QMessageBox` 均由架构测试限制在 presentation。
- Fitting infrastructure 的 `legacy_*` adapters 仍委托既有 `utils` scientific/model 实现；
  adapter 正是隔离这些稳定实现的边界。只有数值回归 fixture 完整时才应替换内部实现。
- feature-owned 大型页面可按稳定 UI 区块继续缩小，但必须保持视觉、快捷键、objectName、信号和
  scientific output，不以文件行数作为拆分依据。
- Fitting、Prediction、Classification 与 Trainset 仍有按当前结果临时构造的 workflow dialogs；
  它们不是主页面所有权的 P0 问题。只有 widget hierarchy 稳定后，才按一个 dialog 一个
  seam 提取为 feature-owned Python View，禁止一次性搬迁全部瞬时 dialogs。

## 兼容入口保留标准

几行的 re-export/module alias 不是重复实现，也不是应直接删除的“无用文件”。删除前必须同时满足：

- 仓库内 caller 已迁移；
- 旧入口与新 owner 均有测试；
- 已评估用户脚本、插件、portable job 和第三方 import path；
- 删除后完整测试、架构测试和离屏启动通过；

才能在独立变更中删除兼容入口。CBF、NXS、HDF5、图片 fixture 与示例数据必须保留。

## 验证基线

- 完整测试：`427 passed, 1 skipped`；唯一 skip 是外部 TFRecord shards 不存在；
- Python View inventory：37 个 View 均具有明确 owner，并通过依赖门禁；
- shared path primitive 另有 Unicode、`file://`、环境变量和旧入口回归测试；
- TensorFlow 数值兼容测试在子进程加载 runtime，pytest 主进程不再混载 TF 与 PyQt；
- pytest session 会在最终 GC 前显式释放 offscreen Qt/Matplotlib resources；
- `python -m ruff check .`、`python -m compileall -q src/gimap main.py` 与
  `git diff --check` 通过；
- 离屏真实主窗口完成延迟初始化：5 个 workspace pages、4 个 feature ViewBindings、默认 page
  index 2 均验证成功。
