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

## 当前架构

采用 feature-first modular monolith。当前 feature 边界包括 format converter、fitting、
prediction、trainset、classification、WAXS 和 calibration。每个 feature 内采用：

```text
presentation/
application/
    ports/
domain/
infrastructure/
    adapters/
```

目录必须对应真实职责和运行代码。不得创建空架构目录，不得移动无关源码，也不得在单个
任务中混入无关的大范围改写。

不要重新建立一个以全局 `controllers/`、`services/`、`models/` 或 `utils/` 为核心的
架构。Feature-first ownership 优先于全局技术分层。

`src/gimap` 生产代码不得反向导入顶层 `controllers`、`ui`、`trainset`、`calibration`、
`WAXS` 或 `utils` 兼容包。旧路径只能向 feature/app/shared owner 单向转发；不得在兼容文件中
新增业务实现。`utils/ML_Fitting_1D_GISAXS` 仅作为专用 TensorFlow worker/training bundle
维护，不是新增通用 helper 的位置。

## 依赖方向

所有生产代码必须遵循：

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

Presentation 不得直接导入本 feature 的 domain。需要展示的稳定 DTO、枚举或只读能力应由
application 的 public API 明确导出；presentation 通过 application 获取它们，避免绕过
use case 和 application boundary。

Presentation 采用：

```text
PyQt View → ViewModel → Use Case
```

Feature-owned `ViewBinding` 可以连接 widget signals、dialogs、rendering 与 ViewModel。
ViewBinding 属于 View 的实现细节，只能做控件值映射和展示；不得绕过 ViewModel 调用 use
case，不得执行科学计算或具体 I/O。顶层旧 import path 只能薄 re-export 当前 owner，不能
形成第二套 Controller orchestration。

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

## 保持科学行为不变

架构、UI、性能和维护性修改不得静默改变：

- numerical definitions 或结果；
- parameter meanings；
- units；
- array orientation；
- constraints；
- ranking；
- fitting behavior；
- preprocessing behavior。

只有任务明确要求时才能修改科学行为。科学行为修改必须单独说明，并通过适当测试和
scientific review 验证。

## 文件大小与内聚性

- 新手写 Python 文件通常应保持在 400 行以内；
- Controller 和 ViewModel 通常应保持在 300 行以内；
- 这些是 architecture-review 阈值，不是机械硬限制；
- 禁止仅为满足行数要求而进行没有明确职责边界的拆分。
- `dialog.py`、`page.py`、`view_binding.py` 等 public presentation entrypoint 只负责组合、
  依赖注入和稳定 re-export，不得重新承载完整页面实现；架构测试以 600 行作为入口退化门禁；
- 仓库 runtime Python 文件另有 600 行 monolith 安全门禁；确有高内聚理由需要超过时，必须
  先完成 architecture review 并记录显式例外，禁止通过压缩排版或无语义切片绕过；
- 页面事件和展示绑定按职责放入命名明确的 `presentation/bindings/` 模块；大型 Python View
  按有语义的视觉 section 或 component 拆分；
- 禁止使用 `part1.py`、`part2.py` 等仅按大小切割、无法表达职责的模块名。

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
- UI 维护必须审计 objectName、快捷键、默认值、tab order 和 signals；有意改变交互或视觉行为
  时必须更新离屏测试与 workspace 文档；
- `tests/test_ui_source_of_truth.py` 维护显式 Python View inventory 和依赖门禁；新增、删除或
  重命名 View 必须同步清单并说明 owner；
- 禁止重新新增 `.ui`、pyuic 输出或 UI 编译步骤；视觉参考应使用文档或截图保存。

## 公共 Presentation 组件

修改或新增页面前，必须先检查 `src/gimap/app/presentation/components/` 的公共组件 API，
优先复用已有 `ParameterSection`、`AdvancedSection`、`FilePicker`、`PlotPanel`、
`ResultTable`、`JobStatus`、`EmptyState`、`ErrorBanner` 和 safe-wheel numeric inputs。

- 公共组件只能包含跨 feature 稳定的视觉、布局、可访问性和输入安全行为，不得包含科学计算、
  use-case 调用、文件格式判断或 feature 状态；
- feature presentation 只能通过 `src.gimap.app.presentation` 或其 `components` public API 导入，
  禁止导入另一个 feature 的私有 presentation component；
- 至少两个 feature 已有稳定相同需求，或属于全应用必须一致的交互安全规则时，才提升为公共组件；
  禁止为了“将来可能复用”提前抽象；
- 新公共组件必须从 `components/__init__.py` 和 `app/presentation/__init__.py` 显式导出，增加
  offscreen test；有视觉状态时同步 showcase；
- 动态创建的 spin box/combo box 必须再次调用 `install_safe_wheel_behavior`；普通滚轮用于滚动页面，
  只有控件获得焦点且按住 Alt/Option 时才允许滚轮改值；
- 公共组件目录不得出现 `utils.py`、`helpers.py`、`common.py` 等 catch-all 文件。

详细组件清单、选择规则和新增流程见
[`docs/architecture/ui-components.md`](docs/architecture/ui-components.md)。

## 现代 UI 与布局门禁

任何 workspace、page、dialog 或公共组件的新增与重构，都必须先阅读并遵守
[`docs/architecture/ui-design-principles.md`](docs/architecture/ui-design-principles.md)。该文档是
视觉层级、响应式布局、progressive disclosure 和 UI 验收的唯一权威来源。

- 页面采用单层画布，通过留白、标题、对齐和分隔线建立层级；同一视觉区域最多一层带边框或背景
  的容器，禁止 `Section → Card → GroupBox → Frame` 连续显示“框套框”；
- Input、主要参数、Preview、主命令和 Results 必须默认可见。Advanced 只允许低频或专家选项，
  标准工作流不得依赖展开面板；
- 面向开发者的需求备注、实现说明和用户对 agent 的指导不得放入 UI；
- 可变内容必须由 layout、`QSizePolicy` 和当前内容的 size hint 管理；禁止把动态页的 minimum 与
  maximum height 锁为同一个值，禁止用固定高度掩盖裁切问题；
- 一个方向只保留一个主要滚动容器；tab/stack 必须跟随当前页自然高度；
- tab 和步骤导航必须保持位置、顺序稳定；仅当前页需要的 toolbar、banner、filter 必须位于 tab
  内容内部或固定占位区，不得因显示/隐藏而把导航栏推上推下；
- 图像显示控制紧邻 Preview，核心手势必须同时有显式 command；纯显示操作不得触发 cut、fit 或
  页面跳转；
- 曲线 q/log 控件必须使用用户任务语言。Signed q 的 Log X 使用 symlog，折叠后的 `|q|` 才使用
  普通 log；不得把 branch、combination、axis scale 三个底层维度直接堆给用户；
- 每次 UI 修改必须检查 1280×800、1440×900、1920×1080 的逻辑 viewport，运行 offscreen test，
  并用截图检查裁切、重复边框、核心命令可见性和唯一主操作。

## 科学数据流门禁

Detector image 和其他会被多个科学步骤消费的数据必须遵守
[`docs/architecture/scientific-data-flow.md`](docs/architecture/scientific-data-flow.md)。

- 导入数据保存为不可变 RawImage；scientific preprocessing 必须从 RawImage 确定性生成唯一的
  AnalysisImage，禁止在上一版处理结果上累计变换；
- Preview、Yoneda/center finding、ROI、cut、fitting、batch analysis 和 processed export 默认且
  只能消费 AnalysisImage，禁止缺失时静默回退到 RawImage；
- Flip、threshold/mask、detector correction 和 mirror-fill 属于 scientific preprocessing；
  colormap、vmin/vmax、auto scale、log intensity、zoom 和 overlay 属于 DisplayState；
- DisplayState 不得改变科学数组、触发 scientific command 或使分析结果失效；preprocessing 改变
  必须产生新 revision，并把依赖旧 revision 的 center、cut 和 fitting result 标记为 stale；
- 嵌入 preview 与独立窗口只能是同一 AnalysisImage/CurvePlotSpec 和 typed display state 的两个
  projection；禁止在独立窗口重复过滤、归一化或保存第二份科学状态。独立窗口可单独拥有 zoom、
  pan、窗口几何和临时工具模式；
- `Overlay ±q` 必须保留正负 source branch metadata，并以稳定、可辨识的不同颜色展示；
- RawImage 只能用于重新 preprocessing、明确的 Raw Preview/Raw Export 或具名诊断；禁止通过
  presentation display helper 给 application/domain 提供计算输入；
- 裸数组兼容字段只能是 AnalysisImage 的只读别名，不能拥有第二份状态。代码必须使用语义明确的
  data-flow API，并用测试证明 preview 和下游算法消费同一 revision。

## Fitting 科学模型门禁

Fitting 的总强度、Sphere/Cylinder/Vertical Cylinder form factor、结构因子、resolution、参数
顺序、q 单位和分量缩放以
[`docs/architecture/fitting-scientific-model.md`](docs/architecture/fitting-scientific-model.md)
为唯一权威说明。

当前核心公式为：

```text
I_model(q) = BG + K(k) × [Σᵢ Intᵢ Pᵢ(q) Sᵢ(q) + int_Res R(q)]
R(q) = 1 / [1 + (|q| / sigma_Res)^nu_Res]
F_sphere(q,R) = 3[sin(qR) - qR cos(qR)] / (qR)^3
P_sphere(q) = <F_sphere(q,R)^2>
phi(q) = exp(-pi q^2 sigma_D^2)
S(q) = (1 - phi^2) / [1 + phi^2 - 2 phi cos(qD)]
```

`BG` 不乘 `k`；粒子分量与 resolution 分量乘相同的 `K(k)`。完整的圆柱公式、采样定义、
参数语义和边界行为只在上述科学契约中维护，禁止在其他文档复制另一套定义。

- 总曲线必须逐点满足 `Total = BG_total + Resolution + Σ Particle`；
- 模型的所有分量必须在用于绘图的同一个 prepared q 数组上计算；
- q、intensity 和 source-branch metadata 的过滤、fold 与排序必须使用同一个索引；
- 禁止将 prepared model intensity 与 raw q 重新配对或再次独立 fold/sort；
- 修改公式、采样、参数语义、单位或累加顺序时，必须同步更新公式文档并增加固定数值回归测试。

## 参数提交与导航门禁

所有 workspace 和 dialog 必须遵守
[`docs/architecture/ui-interaction-contract.md`](docs/architecture/ui-interaction-contract.md)。

- workflow step、workflow completion 和 preview tab 必须是独立状态；点击左侧任务不得重置右侧视图；
- 参数编辑、鼠标选区和自动刷新不得隐式切 tab、滚动、抢焦点或打开 dialog；
- 数值输入 Enter/结束编辑立即提交，方向键和有意滚轮采用默认 `220 ms` trailing debounce；普通
  滚轮不得改值；相同 draft 不得重复提交；
- 轻量 preview 可以节流刷新，scientific commit 必须通过 ViewModel command/use case；长任务必须
  由显式 command 通过 JobRunner 启动；
- 只有显式 Run/Plot/Extract 等命令成功产生有效结果后，才允许主动揭示结果页；失败和自动刷新
  必须保留用户当前 view；
- 已有派生结果可以在相关参数 commit 后防抖重算；尚无结果时不得把普通参数修改升级为隐式运行；
- Qt signal 合并应复用公共 `ParameterCommitCoordinator`，不得在各 feature 重复 timer glue，也不得
  把 scientific calculation 放入公共 coordinator。

## 文档治理

文档必须与代码保持同步，但禁止为没有文档影响的修改制造无意义的文档变更。不同目录
承担不同职责：

- `docs/architecture/`：当前架构、科学契约和稳定依赖规则；
- `docs/ui/`：workspace 信息架构、控件映射和手动验收清单；
- `docs/development.md`：开发环境、依赖安装、统一检查命令和平台差异；
- `docs/adr/`：需要长期保留原因和权衡的重要架构决策；仅在确有此类决策时创建。

禁止在多个文档中复制同一套完整规则。详细依赖规则以
`docs/architecture/dependency-rules.md` 为唯一权威来源；本文件只保留 coding agent
必须执行的摘要。其他文档应链接到权威来源，而不是复制后各自演化。

新的重要文档应在开头明确：

- `Status`：`Current` 或 `Draft`；
- `Scope`；
- `Related code`；
- `Related tests`；
- `Last verified`。

发生以下变化时，必须在同一任务中更新对应的权威文档：

- feature boundary 或 dependency direction 变化；
- application port、public application API 或配置格式变化；
- 用户工作流、启动流程或后台任务模型变化；
- BornAgain、TensorFlow 或其他外部依赖的安装和兼容方式变化；
- public import alias、entry point 或 public API 被增加、修改或删除。

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

每个新的 application use case 都必须增加测试。交付修改前：

- 运行与修改行为相关的 focused tests；
- 可行时运行仓库统一验证命令；
- 移动计算逻辑时比较可信科学输出；
- 检查 `git status` 和 `git diff`，确认修改范围；
- 检查文档影响和仓库内链接，避免实现与文档状态漂移；
- 应明确报告 architecture violations，不得静默扩大 lint 或架构豁免范围。

完整架构说明位于：

- `docs/architecture/overview.md`；
- `docs/architecture/dependency-rules.md`。

## In-situ 序列分析契约

Fitting 的实时、历史回看和批量处理必须遵守
`docs/architecture/insitu-series-workflow.md`：

- In-situ 复用单文件的 preprocessing、cut 和 fitting use cases，不得复制科学算法；
- 单文件配置只能经用户显式操作创建不可变 Recipe 快照；
- In-situ 修改必须创建新 Recipe 版本，不得反向或隐式同步到 Single analysis；
- Recipe 更新必须声明 `future`、`selected_and_future` 或 `all` 作用范围；
- display-only state 不进入 Recipe；worker 和持久化数据必须可 JSON 序列化；
- 相同源数据与 Recipe 经 Single/In-situ 入口执行时必须保持数值兼容。
- Single analysis 的 Load Mode 只能包含 Single/Stack；Live、历史回看和批量序列操作只能位于
  feature-owned In-situ 页面，禁止重新增加 In-situ mode、轮询 timer 或第二个 runner dialog；
- In-situ 页面采用 `Source → Preprocess → Geometry → Yoneda & cut → Fit → Results` 可点击流程。
  点击节点只负责导航参数；完成/失败状态必须来自实际 frame record，不能来自点击历史；
- Live 与 Batch 必须共享 Recipe、folder/pattern 输入、JobStatus、预览和结果缓存；不得复制单帧
  preprocessing、cut、fit 算法或维护第二份运行状态。
