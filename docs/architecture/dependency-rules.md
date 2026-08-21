# 依赖规则

> **Status**：Current
>
> **Scope**：`src/gimap` 生产代码的允许依赖和禁止依赖
>
> **Last verified**：2026-08-20

这些规则适用于全部 `src/gimap` 生产代码。顶层 public import alias 可以向当前 owner 转发，但
`src/gimap` 禁止反向导入 `controllers`、`ui`、`trainset`、`calibration`、`WAXS` 或
`utils`；架构测试会阻止这类依赖重新出现。

## 允许的依赖

| 来源 | 可以依赖 | 用途 |
|---|---|---|
| `presentation` | 本 feature 的 public `application` API | 调用 use cases 并展示结果 |
| `application` | `domain`、application-owned ports | 在不引入框架细节的情况下编排工作流 |
| `domain` | Python 标准库、NumPy，以及适用于稳定 scientific primitive 的 SciPy | 表达稳定科学含义 |
| `infrastructure` | `application` ports、`domain`、外部库 | 实现外部能力 |
| composition root | presentation、application、具体 adapters | 构建并注入对象关系 |

核心方向为：

```text
presentation → application → domain

infrastructure → implements application ports
```

## 禁止的依赖

### Domain

Domain 禁止导入或依赖：

- PyQt 或 PySide；
- TensorFlow 或 Keras；
- BornAgain；
- presentation code；
- controllers、widgets、dialogs、windows 或其他 GUI-specific modules；
- infrastructure adapters；
- 具体文件系统行为。

Domain 明确允许：

- Python 标准库；
- NumPy；
- SciPy，但仅限语义稳定且适合成为 domain scientific primitive 的数值能力。

因此，“domain 为纯 Python”表示它不依赖 GUI、ML runtime、simulation engine 和 I/O
infrastructure，并不表示它只能使用 Python 标准库。引入其他数值库前需要 architecture
review，确认其稳定性和 domain 适用性。

Domain API 不得接受或返回 Qt objects、TensorFlow tensors、BornAgain objects、
widgets 或 open file handles 作为 domain model 的组成部分。

### Application

Application 禁止：

- 导入 PyQt 或 PySide；
- 创建、检查或修改 `QWidget` instances；
- 调用 `QMessageBox`、`QFileDialog` 或其他 GUI dialogs；
- 导入具体 infrastructure adapters；
- 直接调用 BornAgain 或 TensorFlow；
- 通过原生文件对话框选择具体路径；
- 依赖另一个 feature 的 presentation、controller 或内部模块。

Application 可以为 simulation、inference 或 storage 定义 port，并通过 dependency
injection 接收其实现。

### Presentation

Presentation 是唯一允许通过 `QMessageBox`、`QFileDialog` 和类似 GUI API 与用户
交互的层。Presentation 禁止执行：

- scientific calculation；
- TensorFlow inference；
- BornAgain simulation；
- 具体文件系统实现；
- 应属于 use case 的工作流编排。

Presentation 可以收集用户选择的路径，并将路径作为 application request 的一部分。
实际 repository/storage 操作必须通过 application port 完成。

### ViewModels

ViewModel 只允许负责：

- UI state；
- commands；
- 调用 use cases；
- 将 use-case results 转换为 display state。

ViewModel 禁止负责：

- scientific calculations；
- TensorFlow inference；
- BornAgain simulation；
- 具体文件系统 implementations；
- `QMessageBox`、`QFileDialog` 或 widget manipulation。

### ViewBinding 与 public Controller aliases

Presentation 的调用链必须是：

```text
PyQt View → ViewModel → Use Case
```

Feature-owned `ViewBinding` 可以连接 widget signals、dialogs、rendering 与
ViewModel。ViewBinding 视为 View 的实现细节，只能做控件值映射和展示，不得调用具体 adapter、
执行科学计算或形成第二层 workflow orchestration。生产代码不得同时保留同一页面的
ViewBinding 实现和 Controller 实现。

顶层 Controller import path 只能薄 re-export 当前 feature owner。生产代码不得建立以下链路：

```text
View → Controller → ViewModel → Use Case
```

Presentation 中确有 composition 或 navigation 对象时，它不得
包含 application workflow orchestration、scientific calculation、BornAgain/TensorFlow
调用或具体文件系统实现。

## 跨 feature 规则

一个 feature 禁止直接依赖另一个 feature 的：

- presentation；
- controller 或 ViewModel；
- infrastructure adapter；
- private/internal implementation。

以下形式明确禁止：

```text
prediction → FittingController.SomeHelper
```

跨 feature 协作只允许通过：

1. 由提供方 feature 维护的 public application API；
2. 明确的 port 或 interface；
3. 具有明确所有权的稳定 shared domain/scientific primitive。

Public application API 不是普通代码复用通道。跨 feature application API 调用只适用
于真正的业务协作。如果多个 feature 复用的是稳定数学或科学能力，应优先提取为具有
明确 ownership 的 shared scientific kernel，而不是让一个 feature 间接调用另一个
feature 的 use case。例如：

```text
不好：Prediction → FittingUseCase → q conversion

推荐：Prediction ─┐
                  ↓
             shared scientific q-space
                  ↑
       Fitting ────┘
```

`shared/` 不是默认放置位置。只有至少两个 feature 已经稳定需要相同领域能力，并且该
能力的语义、边界和 ownership 明确时，才能提取 shared abstraction。禁止为了“未来
可能复用”而提前创建 shared code，也禁止将 `shared/` 变成新的 catch-all directory。

禁止通过全局技术目录绕开 feature 边界。不得新增 `utils.py`、`helpers.py`、
`common.py` 或 `misc.py`，因为这些名称没有表达明确职责。模块名称必须说明其负责的
操作或科学概念。

## Ports/adapters 规则

Ports 是 application-owned interfaces，adapters 是 infrastructure-owned
implementations。

Feature 内推荐结构：

```text
application/
    ports/

infrastructure/
    adapters/
```

常见 ports 包括：

- `SimulationPort`；
- `PredictionModelPort`；
- `FileRepositoryPort`；
- `DatasetStoragePort`。

常见 adapters 包括：

- `BornAgainSimulationAdapter`；
- `TensorFlowModelAdapter`；
- `LocalFileSystemAdapter`。

具体规则：

- use case 只能依赖 port，不能依赖具体 adapter；
- adapter 实现 application 所需能力，并在边界处转换外部类型；
- BornAgain、TensorFlow/Keras 和具体文件系统 imports 必须位于 infrastructure
  adapters；
- port 不得暴露外部库特有类型，除非该类型已经是稳定的 domain primitive；
- adapter 不得调用 presentation 或展示 dialogs；
- adapter 的构建属于 composition root；
- use-case tests 应尽可能使用 fake 或 test double。

## Use-case 规则

每个新的 application use case 都必须有测试。Use case 应当：

- 表达一个内聚的用户操作或批处理操作；
- 接受 framework-neutral input；
- 协调 domain logic 和 injected ports；
- 返回 framework-neutral results 或 application errors；
- 避免 widget state 和具体外部库细节。

## 科学行为规则

架构、UI、性能和维护性修改不得静默改变科学结果或语义，包括：

- numerical definitions；
- parameter meanings；
- units；
- array orientation；
- constraints；
- ranking；
- fitting behavior；
- preprocessing behavior。

任何科学行为修改都必须有明确任务范围、专门测试和适当科学 review。禁止将行为修改
隐藏在移动、重命名、提取或 dependency inversion 中。

## 模块大小与内聚性

- 新手写 Python 文件通常不超过 400 行；
- Controller 和 ViewModel 通常不超过 300 行；
- 以上数值用于触发 architecture review，并非机械硬限制；
- 禁止仅为满足行数要求，将内聚逻辑拆成无意义的碎片。

## Architecture violation 处理规则

所有 `src/gimap` 生产代码必须符合这些规则：

- 不得引入新的 violation；
- 修改碰到已有 violation 时，不得扩大其影响范围；
- public import alias 必须保持薄转发，不能承载业务实现；
- 修复 violation 前先用 focused tests 固定科学和用户可见行为；
- 一个任务只处理请求范围内的职责，不同时改写无关 feature。

## Review checklist

接受新增或修改代码之前，应检查：

- 代码是否归属于一个 feature，而不是全局技术目录？
- 所有依赖是否指向允许的方向？
- PyQt 用户交互是否仅存在于 presentation？
- BornAgain、TensorFlow 和文件系统细节是否位于 ports/adapters 后方？
- 跨 feature 复用是否通过 public API、port 或稳定 primitive？
- 跨 feature API 是否代表真实业务协作，而不是绕路复用数学能力？
- Shared abstraction 是否已有至少两个稳定使用方，并具备明确语义和 ownership？
- 模块名称是否表达明确职责？
- 每个新 use case 是否有测试？
- 科学输出和语义是否保持不变？
- 本次修改是否保持职责内聚，没有混入无关 feature？
