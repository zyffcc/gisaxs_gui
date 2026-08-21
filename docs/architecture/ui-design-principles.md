# GIMaP 桌面 UI 设计原则

- **Status**: Current
- **Scope**: 所有 workspace、page、dialog 和共享 presentation 组件的视觉层级、布局与验收规则
- **Related code**: [`src/gimap/app/presentation/`](../../src/gimap/app/presentation/)、[`src/gimap/features/`](../../src/gimap/features/)
- **Related tests**: [`tests/test_ui_design_system.py`](../../tests/test_ui_design_system.py)、[`tests/test_ui_workspace_layouts.py`](../../tests/test_ui_workspace_layouts.py)
- **Last verified**: 2026-08-19

## 设计目标

GIMaP 的桌面界面应具有现代 Web 应用相同的**信息清晰度、空间节奏和渐进披露**，但仍保留
桌面科学软件的原生键盘操作、精确数值输入和高密度数据展示。这里的“像 HTML”不表示把页面
改成网页，也不表示依赖大量圆角卡片；它表示用户能立即回答三个问题：

1. 我现在处于哪个任务？
2. 当前必须完成什么？
3. 结果和下一步在哪里？

PyQt5 Qt Widgets 已具备自动布局、伸缩策略和样式能力。默认继续使用 PyQt5；只有明确的产品
需求或平台生命周期评审才允许迁移 PyQt6/PySide。不得把拥挤、裁切或僵硬布局归因于框架版本，
也不得以框架迁移替代正确的 layout、size policy 和 information architecture。

## 单层画布，而不是“框套框”

页面使用一个主画布。视觉层级主要由留白、标题字号、字重、对齐和分隔线建立，而不是由边框建立。

```text
Workspace canvas
  Page/task heading
  Section heading
  Controls or content
  Subheading
  Controls or content
```

强制规则：

- 同一视觉区域最多允许一层带背景或边框的容器；禁止 `Section → Card → GroupBox → Frame`
  连续嵌套并让每层都显示边框；
- `QGroupBox` 只在其边界本身表达必要语义时使用。普通表单分组优先使用无边框 `QWidget`、标题
  `QLabel` 和 layout；
- 相关控件优先共享同一 section，通过 8/12/16/24 px 的空间节奏区分层级；
- 状态、警告和空状态可以有有色表面；普通参数区不应全部变成独立卡片；
- 不同时显示含义重复的 workspace 标题、section 标题、card 标题和 group 标题。

## 默认状态必须可用

Progressive disclosure 只用于低频、危险或专家级选项。核心工作流不能依赖默认折叠的面板。

- Input、主要参数、Preview、主命令和当前结果默认可见；
- `Advanced` 内的内容即使永远不展开，也不能阻塞一次标准工作流；
- 禁止创建内容为空、内容已被 reparent 或展开后被固定高度裁切的 disclosure；
- 核心 command 不得只存在于右键菜单、双击、隐藏手势或 tooltip；这些只能作为快捷方式；
- 每个任务只有一个视觉主操作。次要操作使用普通按钮或文字按钮；危险操作必须有清晰语义；
- 说明文字必须帮助用户决策。面向开发者的需求备注、实现解释和“告诉 agent 的话”不得显示在 UI。

## 响应式 Qt Widgets 布局

所有页面使用 layout 管理尺寸，不使用手工坐标。内容的高度由当前可见内容决定，并允许外层滚动。

- 使用 `QSizePolicy`、stretch factor、`sizeHint()` 和 `minimumSizeHint()` 表达意图；
- 禁止为可变内容把 `minimumHeight` 与 `maximumHeight` 锁为同一动态计算值；这会导致字体、DPI、
  翻译或内容变化时被裁切；
- 禁止使用固定高度掩盖 layout 问题。确有固定尺寸的对象仅限 icon、短按钮、toolbar 或明确尺寸的
  preview placeholder；
- 一个页面方向上只保留一个主要滚动容器。避免 scroll area 内再嵌套 scroll area；
- 宽屏可使用双列表单或 inspector，窄屏必须能退化为单列或外层滚动，不得隐藏功能；
- 控件的最小宽度只保证可输入，不应把左侧工作区撑到挤压 preview；
- `QStackedWidget` 和 tab 页的尺寸必须跟随当前页，不得被隐藏页或固定最大高度控制。
- 每个 tab 的 disclosure、toolbar、表格和结果内容只拥有本页几何；展开隐藏页内容后切换标签，
  新页的 `sizeHint`、`minimumSizeHint` 和滚动范围必须恢复为新页自身的值。不得用所有 tab 的
  最大内容高度作为共享最小高度；这类问题应通过 current-page-aware container 解决，而不是
  给页面写死高度；
- tab、步骤导航和 workspace navigation 属于稳定坐标系：标签栏位置和顺序不得因当前页的 toolbar、
  banner、筛选器或结果状态显示/隐藏而移动。条件控件必须放在对应 tab 内容内部、固定占位区或
  overlay 中，禁止插在持久导航之前；

Qt 官方的 [Layout Management](https://doc.qt.io/qt-6/layout.html) 说明 layout 会根据
`sizePolicy`、minimum size、stretch 和内容变化自动重新分配空间。实现与 review 应以该模型为准。

## 科学图像与曲线的交互

Preview 是工作流的一等区域，不是参数页面下方的附属结果。

- 图像显示控制紧邻图像，以可见 inspector 或 toolbar 呈现；不得在 Input 和 Preview 各复制一套；
- `Auto scale`、强度 log、vmin/vmax、colormap、中心和 cut overlay 等高频显示控制默认可发现；
- preprocessing 与原始显示参数明确分组，但只有低频 preprocessing 可以放入 Advanced；
- `Pick center`、`Select region` 等直接操作必须有显式按钮、选中态、光标/提示和 Esc 取消；
- 纯显示操作不得隐式执行 cut、fit 或切换结果页；改变计算输入后只标记下游结果 stale；
- 曲线 toolbar 使用用户任务语言，不暴露底层算法参数的笛卡尔积。

包含正负 q 的曲线遵循以下界面语义：

- `Signed ±q` 保留符号；勾选 Log X 时使用 symmetric-log；
- `Positive +q` 或已经折叠到 `|q|` 的数据可以使用普通 log；
- `Negative −q` 保留负号，Log X 使用 symmetric-log；
- fold/overlay/average 是用户选择的 q 展示与数据准备模式，不作为三个互相冲突的下拉框暴露。

Matplotlib 官方 [Symlog scale](https://matplotlib.org/stable/gallery/scales/symlog_demo.html)
明确将 symlog 定义为覆盖负值的对数扩展，并在零附近使用有限的线性区。实现必须保持 preview、
fitting region、拟合输入和 export 对同一 q 模式的解释一致。

## 页面任务结构

复杂 workspace 优先采用稳定的任务导航和当前任务内容区：

```text
Input → Setup → Locate/Select → Run → Results → Export
```

导航表示位置，不伪造完成状态。完成、失败和 stale 必须来自真实结果判断。熟练用户可以任意跳转；
引导模式只增加说明，不改变功能可达性。

一个任务内部如果有多个对等工作面，例如 `Components / Global / Manual fit / Auto fit`，使用同级
tabs 或 segmented navigation。不得把常用工作面放在超长表单下方，也不得把它们混入 Advanced。

## 文案与命名

- 使用用户能执行的动词：`Import data`、`Find Yoneda`、`Extract cut`、`Run fit`；
- 避免 `Para.`、`Widget`、`Method 2` 等实现语言；
- 单位始终在 label 中清楚显示；
- helper text 最多解释一个选择的影响，不重复标题，不陈述显而易见的事实；
- tooltip 用于补充精确含义，不得成为发现核心功能的唯一方式。

## 每次 UI 修改的必做流程

1. 先画出当前页面的任务流和容器树，标记重复标题、嵌套边框、折叠核心功能和固定尺寸；
2. 列出 Basic 与 Advanced；证明每个 Advanced 项确实低频；
3. 优先复用公共组件，但不得为了复用制造新的容器层；
4. 保持 View → ViewModel → Use Case 依赖，布局修改不夹带科学算法修改；
5. 对至少 1280×800、1440×900、1920×1080 三种逻辑 viewport 做离屏或人工检查；
6. 验证键盘焦点、safe-wheel、默认状态、空状态、错误状态和长文本；
7. 截图检查以下问题：核心命令是否首屏可见、是否有裁切、是否有重复框线、视觉主操作是否唯一；
8. 更新对应 workspace 控件映射和手动验收清单。

## Review 门禁

以下任一项出现时，UI change 不得视为完成：

- 核心工作流需要展开 Advanced；
- 可见区域出现两层以上连续边框容器；
- 内容因固定最大高度而裁切；
- 页面存在空 disclosure、重复控件或只有手势才能找到的命令；
- 1440×900 下主操作或当前结果不可达；
- 调整显示参数触发科学计算、页面跳转或修改原始数据；
- 新页面没有 offscreen construction test 和对应手动验收清单。
