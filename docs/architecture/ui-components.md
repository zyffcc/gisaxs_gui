# GIMaP 公共 Presentation 组件约定

- **Status**: Current
- **Scope**: 应用级 PyQt 公共组件的职责、调用方式和新增流程
- **Related code**: [`src/gimap/app/presentation/components/`](../../src/gimap/app/presentation/components/)
- **Related tests**: [`tests/test_ui_design_system.py`](../../tests/test_ui_design_system.py)
- **Last verified**: 2026-08-19

视觉层级、响应式布局、折叠规则和截图验收以
[`ui-design-principles.md`](ui-design-principles.md) 为准。公共组件复用不得制造额外的可见容器层。

## 组件放在哪里

```text
src/gimap/app/presentation/components/       跨 feature 的稳定公共视觉组件
src/gimap/features/<feature>/presentation/   feature 专属页面与组件
```

公共组件是应用 presentation API，不是新的 `shared/` 垃圾桶。它们只能负责 widget hierarchy、
layout、视觉状态、可访问性和通用输入安全；不得知道 fitting、prediction 等业务名词，也不得调用
ViewModel、use case、文件系统或科学 runtime。

## 当前公共 API

| 组件 | 用途 | 页面应该提供的内容 |
| --- | --- | --- |
| `ParameterSection` | Input、Configure、Results 等常规区块 | 标题、说明和 feature widgets |
| `AdvancedSection` | 低频参数的 progressive disclosure | feature-owned 参数控件 |
| `FilePicker` | 路径输入与 browse/clear intent | dialog 处理和路径校验 |
| `PlotPanel` | toolbar、canvas host、empty state | 实际 canvas 与 plot command |
| `ResultTable` | 一致的只读结果表和空状态 | headers 与 display rows |
| `JobStatus` | queued/running/success/error/cancel 状态 | Job/ViewModel state |
| `EmptyState` | 没有输入或结果时的引导 | 文案与 action intent |
| `ErrorBanner` | 页面内 warning/error/success | 用户可理解的消息 |
| `SafeWheelSpinBox` / `SafeWheelDoubleSpinBox` / `SafeWheelComboBox` | 防止滚动页面时误改输入 | range、value、业务 label |
| `install_safe_wheel_behavior` | 保护现有或动态输入子树 | 传入页面或新建 subtree root |

统一从 public API 导入：

```python
from src.gimap.app.presentation import (
    FilePicker,
    ParameterSection,
    SafeWheelDoubleSpinBox,
    install_safe_wheel_behavior,
)
```

禁止 feature 直接导入另一个 feature 的 presentation 组件。若视觉组件只有一个 feature 使用，
应先保留在该 feature；当至少两个 feature 的语义和行为已经稳定相同，再提升到 app presentation。

## 数值输入的滚轮规则

普通滚轮永远优先滚动最近的 `QAbstractScrollArea`，避免鼠标经过输入框时意外改变科学参数。
只有输入控件已经获得焦点，并且用户按住 Alt/Option，滚轮才改变数值或下拉选择。

静态页面在完成控件构造后调用一次：

```python
install_safe_wheel_behavior(page)
```

运行时新建的 particle editor 等动态 subtree 也必须对新 subtree 再调用一次。该函数可重复调用，
guard 会由 root 保持，避免 Qt event filter 被垃圾回收。

## 新增公共组件流程

1. 证明需求已经跨至少两个 feature 稳定复用，或属于全应用一致的安全/可访问性规则；
2. 在 `components/` 创建职责明确的模块，禁止 `utils.py`、`common.py` 等名称；
3. 保持 component 无业务逻辑，只发出 intent signal 或接收 display state；
4. 从两个 `__init__.py` public API 显式导出；
5. 在 `tests/test_ui_design_system.py` 增加 offscreen construction、signal/state test；
6. 有明显视觉状态时更新 `src/gimap/app/presentation/showcase.py`；
7. 让一个 caller 采用该组件，并运行 feature、design-system 和 architecture tests。

若只是视觉相似但业务语义不同，保留 feature-owned component，不要通过大量 flags 做成万能组件。
