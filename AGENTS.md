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

迁移历史 Qt 页面时，可以使用 feature-owned `ViewBinding` 连接既有 widget signals、dialogs、
rendering 与 ViewModel。ViewBinding 属于 View 的实现细节，只能做控件值映射和展示；不得
绕过 ViewModel 调用 use case，不得执行科学计算或具体 I/O，也不得与同一页面的 Controller
实现并存。

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

## Python View 与 UI source of truth

GIMaP 不再使用 Qt Designer `.ui` 或 pyuic 生成文件。每个稳定页面、dialog 和 window 必须由
对应 feature 中独立、可读的 Python View 文件拥有：

```text
presentation/
    page.py         # 页面组合、依赖注入和信号绑定
    views/          # 独立 Python 页面、panel、dialog/window 布局
    components/     # 仅在该 feature 内复用的视觉组件
    view_model.py
    styles/         # feature-owned QSS
```

应用外壳的 Python View 位于 `src/gimap/app/presentation/views/`。Feature View 位于
`src/gimap/features/<feature>/presentation/views/`。禁止重新建立包含所有业务页面的单一
monolithic Python UI 文件。

- View 只定义 PyQt widget hierarchy、layout、objectName、tab order、视觉属性和展示绑定；
- Matplotlib canvas、动态参数编辑器等运行时组件通过命名明确的 host widget 注入；
- View 不导入 ViewModel、application、domain、controller、文件系统或外部科学 runtime；
- `page.py`/`dialog.py` 负责注入 ViewModel、连接 commands 和把 state 渲染到 View；
- 每个独立页面或稳定 dialog/window 使用职责明确的 Python 文件；页面变大时按可识别视觉区域
  拆为 `views/` 或 `components/`，禁止形成新的千行通用 UI 文件；
- 禁止把业务流程、科学计算、文件读写或进程管理放进 View；
- 禁止保留同一页面的 `.ui`、pyuic 生成文件和 Python View 三套实现；
- UI 迁移必须保持 objectName、快捷键、默认值、tab order、signals 和视觉行为，并增加离屏
  characterization tests；
- `tests/test_ui_source_of_truth.py` 维护显式 Python View inventory 和依赖门禁；新增、删除或
  重命名 View 必须同步清单并说明 owner；
- 禁止重新新增 legacy `.ui`、pyuic 输出或 UI 编译步骤；历史视觉参考应使用文档或截图保存。

## 文档治理

文档必须与代码保持同步，但禁止为没有文档影响的修改制造无意义的文档变更。不同目录
承担不同职责：

- `docs/architecture/`：当前架构、目标架构和稳定依赖规则；
- `docs/refactor/`：只读审计、迁移地图、阶段状态和 compatibility layer；
- `docs/ui/`：workspace 信息架构、控件映射和手动验收清单；
- `docs/development.md`：开发环境、依赖安装、统一检查命令和平台差异；
- `docs/adr/`：需要长期保留原因和权衡的重要架构决策；仅在确有此类决策时创建。

禁止在多个文档中复制同一套完整规则。详细依赖规则以
`docs/architecture/dependency-rules.md` 为唯一权威来源；本文件只保留 coding agent
必须执行的摘要。其他文档应链接到权威来源，而不是复制后各自演化。

新的重要文档应在开头明确：

- `Status`：`Current`、`Target`、`Historical` 或 `Draft`；
- `Scope`；
- `Related code`；
- `Related tests`；
- `Last verified`。

历史审计和迁移地图不得描述成当前架构。重构完成后，应更新剩余 compatibility layer，
或者把阶段文档明确标记为 `Historical`。不得使用没有具体对象和条件的“以后处理”或
“已经完成”。

发生以下变化时，必须在同一任务中更新对应的权威文档：

- feature boundary 或 dependency direction 变化；
- application port、public application API 或配置格式变化；
- 用户工作流、启动流程或后台任务模型变化；
- BornAgain、TensorFlow 或其他外部依赖的安装和兼容方式变化；
- compatibility layer、legacy entry point 或 public API 被增加、迁移或删除。

文档质量要求：

- 架构图优先使用 Markdown Mermaid，确保图和文字可以一起 review；
- 仓库内链接使用相对路径，文件名、模块名和命令必须与当前仓库一致；
- 不得写入本地绝对路径、access token、用户配置、临时输出或机器专属信息；
- 移动或删除文件时必须检索并修正文档中的失效引用；
- 新文档必须有明确读者和职责，不得创建内容重叠的 architecture 文档；
- `shared/`、ports、feature ownership 等术语必须与架构文档保持同一含义。

每次交付必须明确报告 `Documentation impact`：更新了哪些文档；如果没有更新，说明为何
不需要；同时说明是否改变了 public API、配置格式、用户工作流、依赖方向或兼容层。

## 验证要求

每个新的 application use case 都必须增加测试。交付重构前：

- 运行与修改行为相关的 focused tests；
- 可行时运行仓库统一验证命令；
- 移动计算逻辑时比较可信科学输出；
- 检查 `git status` 和 `git diff`，确认修改范围；
- 检查文档影响和仓库内链接，避免实现与文档状态漂移；
- 应明确报告 legacy violations，不得静默扩大 lint 或架构豁免范围。

完整架构说明位于：

- `docs/architecture/overview.md`；
- `docs/architecture/dependency-rules.md`。
