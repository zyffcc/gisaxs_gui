# GIMaP Workspace 信息架构与交互设计

## 1. 文档范围

本文只描述信息架构和交互目标，不定义新的科学行为，也不改变任何参数、单位、默认值、
array orientation、preprocessing、ranking、simulation 或 fitting 语义。

覆盖以下七个用户工作区：

1. Format Converter；
2. Calibration；
3. Trainset；
4. Fitting；
5. Prediction；
6. Classification；
7. WAXS。

目标不是让七个页面视觉上完全相同，而是让用户在任意 workspace 都能快速回答：

- 输入是什么；
- 当前可以配置什么；
- 运行前能预览什么；
- 哪个操作会启动任务；
- 任务现在是什么状态；
- 结果在哪里；
- 如何导出。

## 2. 当前 GUI 的主要信息架构问题

### 2.1 相同职责使用不同容器

- Format Converter 使用三步 `QStackedWidget` 向导；
- Calibration 使用左右 splitter 和多个 `QGroupBox`；
- Trainset 使用五步列表、stack、内部 tabs 和页面底部动作区；
- Fitting 使用左侧长控制列和右侧多个可折叠 preview/log card；
- Prediction 使用 Input/Model/Run/Results cards；
- Classification 使用 Dataset/Preprocessing/Algorithms/Results stepper；
- WAXS 使用预览区加 Display/Mask/Geometry/ROI/Integration/Batch tabs。

这些结构本身都合理，但相同概念没有稳定位置。例如 Run 状态有时位于按钮旁、有时在底部、
有时在独立 Monitor 页面；Export 有时在 toolbar、有时在 Results tab、有时混在 Configure。

### 2.2 Basic 与 Advanced 边界不一致

- Calibration 已有可折叠 Advanced Settings；
- Format Converter 只有 container option 被标为 Advanced，但 dtype、metadata 和 naming
  的高级程度没有统一说明；
- WAXS 将 display、geometry、mask、ROI 全部同级展示；
- Trainset 的日常参数、科学模型细节和运行基础设施参数处在同一视觉层级；
- Fitting 与 Prediction 在 card 内继续堆叠低频参数；
- Classification 的算法、validation 和 projection 同时展开。

### 2.3 重复控件与重复状态

- 文件路径 + Browse/Open 在所有 workspace 重复；
- Auto scale、Log scale、Colormap 在 Fitting、Prediction、Trainset preview 和 WAXS 重复；
- Vmin/Vmax、mask threshold、beam center、pixel size 和 detector distance 在多个页面重复；
- Start/Run/Predict/Integrate/Generate、Pause、Stop/Cancel 的状态呈现不一致；
- 每页分别维护进度条、状态文字和日志窗口；
- Export Image、Export Curve、Export CSV/JSON/Model/Project 的位置和命名不一致；
- 空预览通常只是普通 `QLabel`，错误有时用 dialog、有时写日志、有时只改状态文字。

## 3. 统一 Workspace 框架

每个 workspace 采用同一概念顺序，但可以根据任务规模使用 card、stepper 或 splitter：

```text
Workspace header
  ├─ title / short purpose
  ├─ current project or input summary
  └─ compact workspace status

Input → Configure → Preview → Run → Results → Export

Persistent bottom area
  ├─ Job Center summary
  ├─ workspace log toggle
  └─ latest error / warning / success
```

### 3.1 六个阶段的职责

| 阶段 | 内容 | 不应包含 |
| --- | --- | --- |
| Input | 文件、目录、project、module/model、frame/range 选择 | 科学计算、运行实现 |
| Configure | 会影响结果的参数；Basic 默认可见，Advanced 折叠 | 结果表、长日志 |
| Preview | 输入图、ROI、mask、预处理阶段、模型/任务摘要 | 文件写入、隐式长任务 |
| Run | readiness、主动作、取消/暂停、任务摘要 | 重复配置表单 |
| Results | 数值、图、候选、metrics、失败项 | 文件对话框 |
| Export | 导出格式、目标、命名和显式导出动作 | 科学结果重算 |

### 3.2 Basic / Advanced 判定规则

Basic 参数满足至少一项：

- 每次典型任务都需要确认；
- 对主要结果语义有直接影响；
- 新用户能够根据实验信息明确填写；
- 错误值会阻止 Run readiness。

Advanced 参数满足至少一项：

- 有稳定默认值且多数任务不需要修改；
- 用于兼容特殊 detector、文件格式或旧模型；
- 属于优化器、采样器、solver、cache、HPC 或显示微调；
- 需要领域专家判断；
- 只影响性能、展示或诊断，而不改变主要操作路径。

Advanced 只是信息层级，不改变参数默认值、可用性或持久化 key。

## 4. Format Converter

### 4.1 目标流程

| 阶段 | 内容 |
| --- | --- |
| Input | Add files、Add folder、Use current file；显示 source/type/frame count/status |
| Configure | frame selection、output format、destination、naming、dtype、metadata |
| Preview | 选中 source 的 First/Middle/Last frame、dataset、shape/dtype、预计输出量 |
| Run | readiness 摘要、Convert、Pause、Cancel、progress |
| Results | succeeded/failed item、输出路径、conversion report |
| Export | 转换产物本身；打开目标目录、查看 report/metadata |

### 4.2 Basic 与 Advanced

Basic：输入文件、是否包含 source、frame mode、output format、destination、命名模板、
preserve original values、是否写 metadata。

Advanced：手写 NXS dataset、Custom frame 表达式、Every N、uint16 scale/clip、collision
suffix policy、per-image JSON、single HDF5 container。

### 4.3 当前重复与整理方向

- Input 页和 Selection 页都展示 source/type/frame/selection，应使用同一 source model，
  Input 负责添加，Configure 负责批量选择；
- dataset selector 应靠近选中 source 的 preview，而不是成为独立底部状态；
- conversion progress dialog 应使用统一 `JobStatus` 视觉语义，但仍保留现有 conversion
  worker/ViewModel 行为。

## 5. Calibration

### 5.1 目标流程

| 阶段 | 内容 |
| --- | --- |
| Input | calibration image、energy、standard、detector model |
| Configure | estimated distance/range、pixel size、background、display overlays |
| Preview | detector image、mask、rings、candidate overlays、manual center marker |
| Run | Auto Calibration、Cancel、stage/progress |
| Results | selected candidate、candidate table、residual/confidence、manual refinement |
| Export | Apply to project、Save calibration JSON、Load existing calibration |

### 5.2 Basic 与 Advanced

Basic：image、energy、standard、detector model、optional estimated distance、Auto Calibration。

Advanced：custom pixel X/Y、custom distance bounds、background subtraction、log intensity、
invalid-pixel mask、ring overlays、manual candidate refinement。

### 5.3 当前重复与整理方向

- `pixel_label`、`detector_label` 和 `detector_combo` 存在信息/选择重叠；目标是把自动识别
  结果作为只读 Input metadata，把 override 放在 Advanced；
- stage label + progress 使用统一 JobStatus；
- candidates、selected result 和 manual refinement 归入 Results，不与 Preview 平级散落。

## 6. Trainset

### 6.1 目标流程

| 阶段 | 内容 |
| --- | --- |
| Input | project、reference scattering file、project load/save、output workspace |
| Configure | beam/detector、ROI、mask、form/structure factor、layers、sampling、preprocessing、model/training、backend |
| Preview | full detector、ROI、masked image、mask-only、pipeline stages、parameter coverage、manual simulation |
| Run | Validate、preview simulation、prepare package、small physical test、generate dataset、train、Maxwell actions |
| Results | validation gate、job state/log、metrics、generated paths、best model |
| Export | project YAML/JSON、portable job package、dataset/results sync、register prediction module |

### 6.2 Basic 与 Advanced

Basic：project name、reference file、energy/wavelength、detector geometry、ROI、particle type、
parameter ranges、sample count、split、preprocessing preset、batch size/epochs、local output folder。

Advanced：custom mask regions、physical background components、noise distributions、transform
细节、structure factor/layer细节、grid cache、model layer editor、optimizer/scheduler、Python
executable、smoke-test size、HPC/Slurm/Maxwell 参数。

### 6.3 当前重复与整理方向

- 页面顶部 Validate/Load/Save/Preview/Prepare/Maxwell 与各 step 内动作重复；目标是顶部只保留
  project actions，运行操作集中在 Run；
- Local progress、Monitor job state、job log 应接入统一 Job Center；Monitor 页面继续展示
  trainset-specific metrics；
- Full detector/ROI/mask preview 使用统一 PlotPanel 外壳，但保留 `ArrayCanvas` 行为；
- Model contract、validation gate 和 package tree 是 Result/diagnostic，不应与 Basic 配置竞争。

## 7. Fitting

### 7.1 目标流程

| 阶段 | 内容 |
| --- | --- |
| Input | scattering file/stack/in-situ sequence、1D curve、detector parameters |
| Configure | center/cut/ROI、model components、constraints、manual/AI fitting profile、plot options |
| Preview | detector image、cut region、1D curve、model curve、residual/parameter preview |
| Run | Cut/Show、Manual Fit、AI Fit、Auto Refine、in-situ workflow、cancel/progress |
| Results | fitted parameters、score/chi-square、AI candidates/ranking、in-situ trend |
| Export | curve/plot、fit result、candidate/session/in-situ result |

### 7.2 Basic 与 Advanced

Basic：input、load mode、center/cut bounds、model selection、主要参数值/范围、fit method、
Manual/AI run、基本 plot selection。

Advanced：mirror-gap/mask threshold、step sizes、solver tolerances、AI sampling/refinement
budgets、fixed combinations、resolution/background细节、plot sampling、in-situ recovery/cache。

### 7.3 当前重复与整理方向

- Detector Preview、Plot Preview、Plot Controls、Run Log 是四张独立右侧 card；目标保持右侧
  preview 主区，但将 Plot Controls 放入 Preview Advanced，将 Run Log 移入统一日志区；
- `CardFrame`、`CollapsibleCardFrame`、`SectionCard` 与新公共 section 语义重复，逐页替换，
  不一次删除 legacy classes；
- Manual/AI/in-situ 三条 Run 路径必须共享 readiness/status 语言，但不合并 scientific use case。

## 8. Prediction

### 8.1 目标流程

| 阶段 | 内容 |
| --- | --- |
| Input | single file/folder、range、Every/stack、module、model |
| Configure | preprocessing/module config、framework/model compatibility、display options |
| Preview | input image、preprocessing stages、model contract/status |
| Run | Predict、Stop、single/multi progress、readiness |
| Results | parameter/image/curve output、multi-file table/trend/heatmap、errors |
| Export | current image/result、JSONL/JPG/ASCII、selected multi-file results |

### 8.2 Basic 与 Advanced

Basic：single/multi mode、file/folder、module、model、range、Every、Predict。

Advanced：framework override、module reload/edit、custom preprocessing、input/output color limits、
colormap、multi-result filters 和低频 export options。

### 8.3 当前重复与整理方向

- Input card 同时容纳 single 与 multi 路径；使用模式切换只展示当前相关 FilePicker；
- Module、framework、model readiness 统一为 Configure summary；
- Run card 的四个 readiness labels 合并为一个 JobStatus/readiness list；
- Results/Preview 当前同卡，目标是 Preview 显示输入/预处理，Results 显示模型输出。

## 9. Classification

### 9.1 目标流程

| 阶段 | 内容 |
| --- | --- |
| Input | labeled dataset sources、scan/import、sample table、QC |
| Configure | preprocessing、validation、algorithms、projection、ranking metric |
| Preview | selected sample、class balance、input shape/memory、quality issues |
| Run | compare classifiers、embedding、predict new data、cancel/progress |
| Results | overview、confusion matrix、per-class metrics、misclassified、embedding、prediction |
| Export | result CSV、prediction CSV、active model、selected file list |

### 9.2 Basic 与 Advanced

Basic：class sources、include/exclude samples、preprocessing preset、recommended algorithms、
validation method/folds、ranking metric、Run Comparison。

Advanced：resize/smoothing/log细节、算法参数编辑、projection、UMAP/t-SNE 参数、class
weight/validation repeat、embedding color mode、legacy model load。

### 9.3 当前重复与整理方向

- Dataset step 内 summary、quality、inspection 分散；目标是 Input + Preview 两列；
- Algorithm table、validation 和 projection 同时展开；projection 与 per-algorithm parameters
  默认进入 Advanced；
- 独立 operation log 使用统一日志区；Results tabs 保持，因为不同结果视图语义清晰；
- Run Embedding 位于 Results tab，但它实际启动 Job，应显示在 Run secondary actions，并将结果
  导航到 Embedding tab。

## 10. WAXS

### 10.1 目标流程

| 阶段 | 内容 |
| --- | --- |
| Input | scattering file/frame、batch folder/pattern |
| Configure | geometry、mask、ROI/cut、integration、display、batch export options |
| Preview | 2D detector/q-space image、cut overlay、1D curve |
| Run | Integrate、batch Start/Pause/Stop、progress |
| Results | current 1D curve、batch item/failure summary、output paths |
| Export | current image、current curve、batch images/curves/background-subtracted matrices |

### 10.2 Basic 与 Advanced

Basic：file/frame、auto/log/colormap、cut type/interactive selection、integration mode/bins、
Integrate、batch folder/output和export choice。

Advanced：manual vmin/vmax、bad-pixel threshold、full detector geometry、manual q limits、
manual line/circle coordinates、x-axis mode、smoothing、file pattern。

### 10.3 当前重复与整理方向

- toolbar 和 Display tab 重复 Auto scale/Log/Colormap；保留 toolbar Basic 控件，详细 color
  limits 进入 Advanced，二者仍绑定同一状态；
- Geometry、Mask、ROI、Integration 六个同级 tabs 造成频繁跳转；目标以 Configure section
  顺序排列，Geometry/Mask/Manual coordinates 可折叠；
- batch export 和 current export 使用同一 Export section 语义，但保持不同 use case；
- 页面底部 status/progress 接入统一 JobStatus，batch item summary 放 Results。

## 11. 公共组件职责

第 14 步的共享组件只负责结构和展示，不读取 `global_params`，不调用 ViewModel，不执行
文件 I/O、simulation、TensorFlow、BornAgain 或科学计算。

| 组件 | 统一职责 |
| --- | --- |
| `ParameterSection` | 有标题、说明、内容区和可选 header actions 的基础 section |
| `AdvancedSection` | 默认折叠的低频参数容器；不改变子控件值 |
| `FilePicker` | path text + Browse/Clear；只发 signal，不打开 dialog、不读文件 |
| `PlotPanel` | title、toolbar slot、canvas slot、empty overlay；不绘制科学数据 |
| `ResultTable` | 一致的 header/selection/empty presentation |
| `JobStatus` | state、message、progress、Pause/Cancel/Details signals |
| `EmptyState` | icon/title/message/optional action signal |
| `ErrorBanner` | error/warning/info/success banner 与 dismiss/details signal |

## 12. 统一 Job Center

### 12.1 位置与层级

Job Center 是主窗口级可折叠底部区域，而不是每个 workspace 各自实现的进程管理器。
默认只显示一行摘要；存在多个任务、失败或用户主动展开时显示任务列表。

```text
Collapsed:  [Running 2]  Prediction 48% · Trainset queued        [Open]

Expanded:
Feature | Job | State | Progress | Elapsed | Message | Actions
```

### 12.2 显示字段

- job id；
- feature/workspace；
- short operation label；
- queued/running/paused/succeeded/failed/cancelled/timed_out；
- progress fraction 与 message；
- start/elapsed/end time；
- error summary；
- Pause/Resume、Cancel、Open Result、Show Details。

Job Center 只展示 application/JobRunner 提供的可序列化状态，不持有 process、worker、
TensorFlow model 或 BornAgain object。首轮组件可以使用展示数据 model 和 signals，实际全局
job registry 在单独任务中接入。

## 13. 统一日志区

- 主窗口底部提供可折叠 Log 区，与 Job Center 相邻；
- 支持 All/当前 workspace/当前 job 过滤；
- 每条记录包含 timestamp、level、feature、job id、message；
- 默认显示 INFO 以上，debug/details 按需展开；
- ErrorBanner 只显示当前需要用户处理的问题，完整 traceback 放日志详情；
- 页面原有日志控件在迁移期可作为兼容 sink，但新 layout 不再各自发明不同日志样式。

## 14. 状态显示层级

状态使用三层，避免同一信息重复出现：

1. Workspace header：只显示整体 readiness，例如 `Ready`、`Input missing`、`Running`；
2. `JobStatus`：显示当前任务进度、动作和短消息；
3. Job Center/Log：显示跨页面任务、历史、错误详情。

颜色不能成为唯一信息来源；所有状态必须有文本和稳定状态名。错误应保留原异常信息，
但 presentation 负责将其组织为用户可理解的 summary。

## 15. 分页迁移原则

每个页面独立执行：

```text
record old widget mapping
  ↓
wrap/reparent existing widgets into shared components
  ↓
preserve objectName and signal connections
  ↓
verify ViewModel/controller contract
  ↓
offscreen construction test
  ↓
manual acceptance checklist
```

优先 reparent 现有 widget，而不是重新创建同名业务控件。不得在视觉任务中修改 domain、
scientific algorithms、default values、units、serialization format 或 ViewModel commands。
每页完成时必须输出旧控件到新 section 的映射，以及包含 Input、Configure、Preview、Run、
Results、Export 的手动验收清单。
