# GIMaP 工程约定

本文件适用于整个仓库。Codex 和其他 coding agents 后续修改代码时必须遵守这些约定。

## 保护用户已有工作

修改文件之前必须运行：

```bash
git status
git diff
```

所有已有 tracked 和 untracked changes 都应视为用户工作。必须保留无关修改，并绕开
发生冲突的内容。不得覆盖、删除或清理与当前任务无关的文件。

未经用户明确授权，禁止执行：

```bash
git reset --hard
git restore .
git checkout -- .
git clean
git stash
```

除非用户明确要求，否则不得 commit 或 push。

## 目标架构

采用 feature-first modular monolith。目标 feature 边界包括 fitting、prediction、
trainset、classification、WAXS 和 calibration。每个 feature 内采用：

```text
presentation/
application/
    ports/
domain/
infrastructure/
    adapters/
```

这是渐进式迁移目标。不得创建空架构目录，不得移动无关源码，不得进行 big-bang
rewrite。

不要重新建立一个以全局 `controllers/`、`services/`、`models/` 或 `utils/` 为核心的
架构。Feature-first ownership 优先于全局技术分层。

## 依赖方向

所有新增和迁移代码必须遵循：

```text
presentation → application → domain

infrastructure → implements application ports
```

- Presentation 使用 PyQt views 和 ViewModels；
- Application 包含 framework-neutral use cases 和 application-owned ports；
- Domain 是不依赖 GUI、ML runtime、simulation engine 和 I/O infrastructure 的
  Python；允许使用 Python 标准库、NumPy，以及适用于稳定 scientific primitive 的
  SciPy；
- Infrastructure 包含 BornAgain、TensorFlow/Keras、文件系统、存储格式和其他外部
  依赖的 adapters；
- 小型 composition root 可以构建 adapters 并注入 use cases，但 application 和
  domain 不得导入具体 adapters。

## Domain 限制

Domain 禁止导入或依赖：

- PyQt 或 PySide；
- TensorFlow 或 Keras；
- BornAgain；
- presentation code；
- controllers、widgets、dialogs、windows 或其他 GUI-specific modules；
- 具体 infrastructure 或文件系统 implementations。

Domain values 和 APIs 必须与 Qt objects、tensors、BornAgain objects 和 file handles
保持独立。

“Domain 为纯 Python”表示框架和外部运行时无关，不表示只能使用 Python 标准库。
Domain 明确允许 NumPy；允许 SciPy 用于语义稳定且适合放入 domain 的 scientific
primitive。引入其他数值库前必须进行 architecture review。

## Application 限制

Application 不得依赖 PyQt，也不得操作 `QWidget`、`QMessageBox`、`QFileDialog` 或
其他 GUI object。Application 不得直接调用 BornAgain、TensorFlow 或具体文件系统 API。

每个 application action 应建模为具有 framework-neutral input/output 的 use case。
外部能力必须通过 ports 注入。每个新的 application use case 必须有测试。

## Presentation 与 ViewModel

`QMessageBox`、`QFileDialog` 和其他直接用户交互只能位于 presentation。

ViewModel 可以：

- 保存 UI state；
- 暴露 commands；
- 调用 use case；
- 接收结果并转换为 display state。

ViewModel 禁止负责：

- scientific calculation；
- TensorFlow inference；
- BornAgain simulation；
- 具体文件系统实现；
- `QMessageBox`、`QFileDialog` 或 widget manipulation。

View 负责渲染 ViewModel state 和用户 dialogs，use case 负责工作流编排。

新 presentation 默认采用：

```text
PyQt View → ViewModel → Use Case
```

Legacy Controller 可以暂时存在，但新代码不得同时发展 Controller 和 ViewModel 两层
orchestration，也不得建立 `View → Controller → ViewModel → Use Case`。确实需要保留
controller 时，它只能承担极薄的 composition 或 navigation 职责，不得包含工作流编排、
科学计算、外部引擎调用或具体 I/O。

## Ports 与 adapters

Ports 属于 feature 的 `application/ports/`，具体 implementations 属于
`infrastructure/adapters/`。

可使用以下 interfaces：

- `SimulationPort`；
- `PredictionModelPort`；
- `FileRepositoryPort`；
- `DatasetStoragePort`。

可使用以下 adapters：

- `BornAgainSimulationAdapter`；
- `TensorFlowModelAdapter`；
- `LocalFileSystemAdapter`。

Port 描述 application 真正需要的能力，不能完整照搬外部库。Adapter 在边界处转换
外部类型和异常。Use-case tests 应使用 fake 或 test double。

## Feature 边界

禁止导入另一个 feature 的 presentation、controller、ViewModel、adapter 或内部实现。
以下模式明确禁止：

```text
prediction → FittingController.SomeHelper
```

跨 feature 复用只允许通过：

- public application API；
- 明确的 port/interface；
- 具有清晰所有权的稳定 shared domain 或 scientific primitive。

跨 feature application API 仅用于真正的业务协作，不能作为普通代码复用通道。多个
feature 复用稳定数学或科学能力时，应优先提取为具有明确 ownership 的 shared
scientific kernel，而不是通过另一个 feature 的 use case 间接调用。例如 prediction
和 fitting 应共同依赖 q-space scientific kernel，而不是让 prediction 调用
FittingUseCase 来完成 q conversion。

`shared/` 不是默认放置位置。只有至少两个 feature 已经稳定需要同一项领域能力，并且
语义、边界和 ownership 明确时，才能提取 shared abstraction。禁止为了“未来可能复用”
提前创建 shared code，也禁止让 `shared/` 成为新的垃圾桶。

禁止新增 catch-all modules：`utils.py`、`helpers.py`、`common.py`、`misc.py`。
模块名称必须表达明确职责。

## 渐进式重构

禁止 big-bang rewrite。每次 refactor 优先只处理一个 dependency seam，不得在请求范围
之外顺便进行无关清理。

采用以下顺序：

```text
legacy code
    ↓
introduce dependency seam
    ↓
introduce new abstraction / use case
    ↓
migrate caller
    ↓
verify behavior
    ↓
remove legacy implementation
```

Legacy architecture 可以暂时违反目标架构，但：

- 新代码不得新增 violation；
- 新代码不得扩大已有 violation；
- 每次 refactor 必须减少或缩小 dependency violation；
- 只有 caller 已迁移且行为验证通过后，才能删除 legacy code。

## 保持科学行为不变

结构重构不得改变：

- numerical definitions 或结果；
- parameter meanings；
- units；
- array orientation；
- constraints；
- ranking；
- fitting behavior；
- preprocessing behavior。

只有任务明确要求时才能修改科学行为。科学行为修改必须与结构重构隔离，并通过适当
测试和 scientific review 验证。

## 文件大小与内聚性

- 新手写 Python 文件通常应保持在 400 行以内；
- Controller 和 ViewModel 通常应保持在 300 行以内；
- 这些是 architecture-review 阈值，不是机械硬限制；
- 禁止仅为满足行数要求而进行没有明确职责边界的拆分。

## 验证要求

每个新的 application use case 都必须增加测试。交付重构前：

- 运行与修改行为相关的 focused tests；
- 可行时运行仓库统一验证命令；
- 移动计算逻辑时比较可信科学输出；
- 检查 `git status` 和 `git diff`，确认修改范围；
- 应明确报告 legacy violations，不得静默扩大 lint 或架构豁免范围。

完整架构说明位于：

- `docs/architecture/overview.md`；
- `docs/architecture/dependency-rules.md`。
