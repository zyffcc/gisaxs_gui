# GIMaP 目标架构概览

> 状态说明：本文档定义渐进式重构的目标架构。当前源码仍属于 legacy
> architecture，尚未完全符合这些规则。

## 架构风格

GIMaP 将逐步演进为 **feature-first modular monolith（按功能组织的模块化单体）**。
它仍然是一个桌面应用和一个可部署仓库，但每个面向用户的功能都拥有自己的
presentation、application、domain 和 infrastructure 代码。

目标结构如下：

```text
features/
    fitting/
        presentation/
        application/
            ports/
        domain/
        infrastructure/
            adapters/
    prediction/
        presentation/
        application/
        domain/
        infrastructure/
    trainset/
    classification/
    waxs/
    calibration/
```

这只是渐进式迁移的目标，不代表现在就要创建空目录，或一次性移动现有源码。

优先采用 feature-first 的原因包括：

- 将一个用户工作流及其实现代码放在一起；
- 为 fitting、prediction、trainset、classification、WAXS 和 calibration
  建立清晰的所有权边界；
- 减少对全局 `controllers/`、`services/`、`models/` 和 `utils/` 技术目录的依赖，
  避免其职责随时间逐渐模糊；
- 允许单个 feature 通过 dependency seam 迁移，而不影响整个 GUI；
- 便于编写聚焦测试，也便于将后续 coding-agent 任务控制在清晰范围内。

## 依赖方向

目标依赖方向为：

```text
presentation → application → domain

infrastructure → implements application ports
```

依赖必须指向策略和科学含义所在的内层。Domain 不应了解 UI 框架、持久化、
BornAgain 或 TensorFlow。Application 通过 ports 协调 domain 行为。
Infrastructure 提供 ports 的具体 adapters。Presentation 负责在 PyQt 事件与
application 输入输出之间进行转换。

应用启动处可以有一个很小的 composition root，用于构建 adapters 并将其注入
use cases。但这不允许 application 或 domain 导入具体 infrastructure 实现。

## 各层职责

### Presentation

Presentation 包含 PyQt views、widgets、dialogs、presenters 和 ViewModels，负责：

- 渲染 UI state 并收集用户输入；
- 将 Qt signals 映射为 ViewModel commands；
- 展示进度、校验消息、错误和 use-case 结果；
- 承担 `QMessageBox`、`QFileDialog` 等用户交互；
- 将 presentation 特有的数据转换为 application request objects。

Presentation 可以依赖本 feature 的 public application API，但不能包含科学计算、
BornAgain simulation、TensorFlow inference 或具体文件系统实现。

新 presentation 默认采用以下调用链：

```text
PyQt View
    ↓
ViewModel
    ↓
Use Case
```

Legacy Controller 可以在迁移期间暂时存在，但不能继续发展成与 ViewModel 并列的第二层
orchestration。新代码不得建立 `View → Controller → ViewModel → Use Case` 链路。如果
确实需要保留 controller，它只能承担极薄的 composition 或 navigation 职责，不得包含
工作流编排、科学计算、外部引擎调用或具体 I/O。

### Application

Application 包含 use cases、request/result 类型、工作流编排策略和 ports，负责：

- 表达用户或批处理工作流可以执行的操作；
- 协调 domain objects 和 application ports；
- 在需要时定义事务、进度、取消和错误边界；
- 返回不依赖 UI 框架、可供 presentation 展示的结果；
- 提供本 feature 的 public application API。

Application 不依赖 PyQt，也不能操作 `QWidget`、`QMessageBox`、`QFileDialog` 或
其他 GUI 对象。Use case 描述工作流，不负责对话框的外观，也不决定原生文件选择器
应当如何打开。

每个新的 application use case 都必须有测试。

### Domain

Domain 是不依赖 GUI、ML runtime、simulation engine 和 I/O infrastructure 的
Python 代码，可以包含 entities、value objects、validation rules、constraints、units
和稳定 scientific primitives。“纯 Python”在这里表示框架无关，而不是只能使用
Python 标准库。

Domain 允许使用：

- Python 标准库；
- NumPy；
- SciPy，但仅用于语义稳定、适合放在 domain 的 scientific primitive。

Domain 禁止导入：

- PyQt 或 PySide；
- TensorFlow 或 Keras；
- BornAgain；
- presentation、controller、widget、dialog 或其他 GUI-specific modules；
- 具体 infrastructure 或 filesystem implementations。

Domain 代码应当是确定性的，并且无需启动 GUI 或安装外部科学计算引擎即可测试。

### Infrastructure

Infrastructure 包含外部系统的具体 adapters，负责：

- BornAgain simulations；
- TensorFlow/Keras 模型加载和 inference；
- 本地文件系统和 dataset persistence；
- serialization formats 和其他外部 I/O；
- 将外部库的数据和异常转换为 application-level 类型。

Infrastructure 实现由 application 定义的 ports。它不能定义 feature 的工作流策略，
也不能直接调用 presentation。

## ViewModel 与 use case

ViewModel 属于 presentation。它可以：

- 保存 UI state；
- 暴露 commands；
- 调用 application use case；
- 接收结果并转换为展示状态。

ViewModel 不得负责：

- scientific calculation；
- TensorFlow inference；
- BornAgain simulation；
- 具体文件系统访问；
- `QMessageBox` 或其他 widget interaction。

Dialogs 和 message boxes 必须留在 view/presentation 边界。ViewModel 可以暴露错误
状态或类似事件的结果，再由 view 决定如何展示。

Use case 负责 application 工作流编排。它接收与框架无关的输入，使用 domain 逻辑
和注入的 ports，并返回与框架无关的结果。它不应知道由哪个 widget、window 或
dialog 发起调用。

## Ports 与 adapters

Ports 用于将 application 工作流与容易变化的外部依赖隔离。Port 属于 application，
因为由 use case 定义它需要什么能力。具体 adapter 属于 infrastructure，因为 adapter
描述该能力如何实现。

每个 feature 内推荐采用：

```text
application/
    ports/

infrastructure/
    adapters/
```

Port 示例：

```text
SimulationPort
PredictionModelPort
FileRepositoryPort
DatasetStoragePort
```

Adapter 示例：

```text
BornAgainSimulationAdapter
TensorFlowModelAdapter
LocalFileSystemAdapter
```

Use-case tests 可以使用内存 fake 或 test double 替代这些 adapters。Port 应描述
application 真正需要的能力，不能照搬 BornAgain、TensorFlow 或操作系统的全部 API。

## Feature 边界

一个 feature 禁止导入另一个 feature 的 presentation、controller、adapter 或内部实现。
例如 prediction 不得通过调用 `FittingController.SomeHelper` 复用 fitting 的 helper。

跨 feature 复用只允许通过：

- public application API；
- 明确的 port 或 interface；
- 具有清晰所有权的稳定 shared domain/scientific primitive。

跨 feature application API 调用只适用于真正的业务协作。如果复用的是稳定的数学或
科学能力，应优先提取为具有明确所有权的 shared scientific kernel，而不是通过另一个
feature 的 use case 间接调用。例如 q-space conversion 不应通过 FittingUseCase 复用：

```text
Prediction ─┐
            ↓
       shared scientific
          q-space
            ↑
Fitting ────┘
```

`shared/` 不是默认放置位置。只有至少两个 feature 已经稳定需要同一项领域能力，且其
语义、边界和 ownership 都明确时，才允许提取 shared abstraction。禁止为了“未来可能
复用”而提前创建 shared code。

Shared code 必须有明确的科学或 application 职责。禁止新增名为 `utils.py`、
`helpers.py`、`common.py` 或 `misc.py` 的 catch-all modules，也禁止让 `shared/` 成为
新的 catch-all directory。

## 当前 legacy architecture 与目标架构的关系

当前仓库包含大型 controllers、GUI-aware modules、全局技术目录、对外部库的直接调用
和共享全局状态。部分 legacy code 目前必然违反目标依赖方向。这些 violation 是后续
迁移的输入，不能作为在新代码中继续复制相同耦合的理由。

迁移期间：

- 已有 violation 可以保留到对应 feature 开始迁移；
- 新代码不得新增 violation，也不得扩大已有 violation；
- 每次 refactor 都应减少 dependency violation；
- compatibility shim 可以暂时连接 legacy caller 和新 use case；
- 只有 caller 已迁移且行为验证通过后，才删除 legacy implementation。

## 渐进式迁移

GIMaP 禁止 big-bang rewrite。默认迁移顺序为：

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

每次重构优先遵循 **one dependency seam per refactor**。不要因为附近代码同样陈旧，
就在一个任务中合并多个无关 feature 的迁移。

移动行为之前，应增加 characterization tests 或记录可信输出。结构重构必须保持：

- numerical definitions 和结果；
- 参数含义和单位；
- array orientation；
- constraints 和 ranking；
- fitting behavior；
- preprocessing behavior。

科学行为修改必须作为独立、明确的任务，并经过适当的科学验证。

## 文件大小指导

新手写 Python 文件通常应控制在 400 行以内。Controller 和 ViewModel 通常应控制在
300 行以内。这些是 architecture review 阈值，不是机械硬限制。职责内聚的模块可以在
有明确理由时超过阈值；禁止为了满足行数要求而进行没有意义的拆分。
