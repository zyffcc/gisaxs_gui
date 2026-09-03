# GIMaP Agent Working Agreement

本文件适用于整个仓库。目标不是规定每一步怎么做，而是让 agent 在长期迭代中持续交付清晰、
可靠、容易继续修改的软件。

## 默认工作方式

- 先理解用户要得到的结果，再选择最小、完整的实现路径。
- 对明确的 change/build/fix 请求，直接完成范围内的本地修改和非破坏性验证；不要为普通实现步骤
  反复请求确认。
- 对 explain/review/diagnose 请求只检查和报告，除非用户同时要求修改。
- 只在会显著改变产品行为、科学定义、公开接口、数据兼容性或任务范围时停下来询问。
- 一个任务聚焦一个用户结果和一个主要 owner。不要顺手重写无关 feature，也不要把重构、视觉
  改版和科学行为变化混成一次大改。
- 允许做有助于完成当前任务的小型整理，但不要为“未来也许需要”提前搭架构。
- 修改过的代码应至少和修改前一样容易阅读。若新增间接层、状态或抽象不能消除更大的复杂度，
  就不要增加它。

## 保护用户工作

修改文件前运行 `git status` 和 `git diff`。已有 tracked 与 untracked changes 都视为用户工作：
保留无关修改，遇到重叠时绕开或明确说明。

未经用户明确授权，不得使用会丢弃工作区内容的命令，包括：

```text
git reset --hard
git restore .
git checkout -- .
git clean
git stash
```

除非用户明确要求，否则不要 commit 或 push。

## 架构一页版

GIMaP 是 feature-first modular monolith。用户功能由 `src/gimap/features/<feature>/` 拥有；应用壳
和跨 feature 的 UI 基础设施由 `src/gimap/app/` 拥有；只有已经被多个 feature 稳定复用、语义
明确的能力才进入 `src/gimap/shared/`。

生产代码的依赖方向是：

```text
PyQt View → ViewModel → Application Use Case → Domain
Infrastructure Adapter ─implements→ Application Port
```

- Presentation 负责 widget、展示状态、信号映射和用户 dialog。
- Application 负责 framework-neutral workflow、DTO 和 ports。
- Domain 负责科学与业务规则，不依赖 GUI、I/O 或外部 runtime；可以使用标准库、NumPy，以及
  适合稳定 scientific primitive 的 SciPy。
- Infrastructure 负责 BornAgain、TensorFlow/Keras、文件系统、存储格式和进程等外部能力。
- Composition root 可以装配 adapter 与 use case，但 application/domain 不导入具体 adapter。

必须保持的边界：

- Presentation 不直接导入本 feature 的 domain，也不直接构造 infrastructure adapter。
- Application 不依赖 PyQt、具体文件系统、BornAgain 或 TensorFlow。
- Domain 不依赖 presentation、application orchestration、GUI object、文件句柄或具体 adapter。
- Feature 不导入另一个 feature 的 presentation、ViewModel、controller、adapter 或内部实现。
  真正的跨 feature 协作使用 public application API、port 或稳定 shared primitive。
- `src/gimap` 生产代码不反向导入顶层 `controllers`、`ui`、`trainset`、`calibration`、`WAXS`
  或 `utils` 兼容包。旧路径只能薄转发当前 owner，不能承载新业务实现。
- `utils/ML_Fitting_1D_GISAXS` 只是专用 TensorFlow worker/training bundle，不是通用工具目录。

详细依赖规则以 `docs/architecture/dependency-rules.md` 为准；ownership 不清楚时先查看
`docs/architecture/overview.md`。

## 控制复杂度，而不是堆规则

- 每项状态、公式和 workflow 只保留一个 source of truth；不要新增平行实现或第二份可变状态。
- 优先复用当前 owner 的 public API。不要通过兼容层、另一个 feature 的 use case 或全局技术目录
  绕路复用代码。
- 局部重复可以暂时存在。至少两个稳定调用方且语义一致后，才提取 shared abstraction 或公共组件。
- 不新增 `utils.py`、`helpers.py`、`common.py`、`misc.py` 等 catch-all module；名称应表达具体职责。
- 不为了满足架构图创建空目录、无行为 wrapper、无实际调用方的 port 或 speculative interface。
- 不用 `part1.py`、`part2.py` 或压缩排版规避文件过大。按真实职责拆分，或者保留高内聚代码并说明
  理由。
- 新手写 Python 文件通常控制在 400 行以内，Controller/ViewModel 通常控制在 300 行以内；
  runtime Python 文件和 public presentation entrypoint 以 600 行为安全门禁。这些数字用于触发
  review，不是机械切割目标。
- 注释解释“为什么”以及科学约束，不重复代码已经清楚表达的“做什么”。

## 需要先停一下的架构闸门

正常的范围内修改可以直接推进。只有出现以下情况时，先给出简短影响分析与可选方案：

- 新增跨 feature 依赖或 shared abstraction；
- 新增 application port、public API、配置格式、持久化格式或兼容入口；
- 改变 feature ownership、依赖方向或后台任务模型；
- 有意改变科学公式、单位、参数语义、数组方向、约束、ranking、fitting 或 preprocessing；
- 为完成任务必须明显扩大到另一个独立用户流程；
- 需要新增生产依赖，或无法在现有边界内给代码找到清楚 owner。

不要因为代码旧、任务较大或存在多个合理实现就自动停下。先检查现有模式和测试，能安全做出局部
决定时就继续。

## UI 工作

GIMaP 使用 feature-owned Python View 作为 UI 唯一事实来源，不使用 Qt Designer `.ui`、pyuic
生成文件或 UI 编译步骤。

- View 只拥有 widget hierarchy、layout、objectName、tab order 和视觉默认值。
- ViewModel 管理展示状态并调用 use case，不做科学计算、具体 I/O、TensorFlow/BornAgain 调用或
  widget manipulation。
- `QMessageBox`、`QFileDialog` 等直接交互只存在于 presentation。
- 修改页面前先查看 `src/gimap/app/presentation/components/`，复用已经稳定的公共组件；不要
  为单一 feature 提前提升公共组件。
- 新增、删除或重命名 View 时同步 `tests/test_ui_source_of_truth.py` 的 inventory。
- UI 变更应检查 1280×800、1440×900、1920×1080 viewport，运行 offscreen tests，并用截图确认
  没有裁切、框套框、导航跳动或重复主操作。

按需阅读，而不是把全部 UI 规则复制到这里：

- 页面布局或视觉层级：`docs/architecture/ui-design-principles.md`
- 公共组件：`docs/architecture/ui-components.md`
- 参数提交、滚轮、刷新或导航：`docs/architecture/ui-interaction-contract.md`
- Python View ownership：`docs/architecture/ui-source-of-truth.md`
- 具体 workspace 行为：`docs/ui/workspaces/<feature>.md`

## 科学行为与数据

重构、UI、性能和维护性修改默认不得改变 numerical result、参数含义、单位、数组方向、约束、
ranking、fitting 或 preprocessing。不要用“清理代码”的名义偷偷改变科学行为。

若任务明确要求科学变化：

- 将变化与无关重构分开；
- 先固定旧行为或建立可信基准；
- 增加有代表性的数值回归测试与边界测试；
- 更新唯一权威科学文档，并在交付中明确说明变化。

修改以下领域前必须阅读对应契约：

- detector image、preprocessing、preview/cut/fitting 数据谱系：
  `docs/architecture/scientific-data-flow.md`
- fitting 公式、参数、q 处理和分量：`docs/architecture/fitting-scientific-model.md`
- fitting live/history/batch：`docs/architecture/insitu-series-workflow.md`
- XRR series extraction：`docs/architecture/xrr-series-workflow.md`
- app settings/session/project state：`docs/architecture/app-context.md`

契约正文只在这些文档中维护，不在 `AGENTS.md` 或其他说明中复制第二套版本。

## 验证

验证应与风险匹配，但不能省略与修改行为直接相关的检查：

- 先运行 focused tests，快速获得反馈；
- 每个新 application use case 都要有测试，优先使用 fake/test double；
- 移动或重构科学逻辑时比较可信输出；
- UI 修改运行相关 offscreen tests，并按上面的 viewport 做视觉检查；
- 可行时运行仓库统一验证：`python tools/check.py`；
- 不通过扩大 lint、架构或测试豁免来让修改过关；
- 交付前再次检查 `git status` 和 `git diff`，确认没有范围外修改。

如果完整验证受环境或外部 runtime 限制，运行能够运行的部分，并准确说明未运行内容和原因。

## DESY Maxwell 远程工作

- 当用户把当前任务明确放在 DESY Maxwell 范围内时，使用本机已有 SSH 配置连接
  `zhaiyufe@max-wgs.desy.de`。连接中断后可以直接重连，不必为普通的任务范围内检查重复询问。
- Maxwell 登录使用交互式认证。不得把密码、一次性秘钥、access token 或其他凭据写入仓库、脚本、
  SSH 配置、shell history、日志或交付内容，也不得把它们提交到版本控制。
- 遇到 `OTP(mfa.desy.de)`、凭据失效或认证被拒绝时立即暂停认证流程，在当前 Codex task 中向用户
  请求新的 OTP。不得猜测、重复试用已经失败或可能过期的 OTP。
- 用户已授权仅为 Maxwell OTP 请求发送通知到 `yufeng.zhai@desy.de`。只有在当前环境已有可用且已
  连接的邮件发送能力时才可使用；否则在当前 Codex task 中请求。邮件中不得包含密码、OTP、token、
  私钥、实验数据或远程日志内容。
- `max-wgs` 是登录节点，只用于检查、文件管理和 Slurm 操作；训练、数据生成和其他重计算必须提交到
  Slurm worker，不得直接在登录节点运行。
- Maxwell 的 `/home/zhaiyufe` 容量有限，只用于轻量代码、配置和必要的小文件。训练集、模型、
  checkpoint、缓存、大型日志及其他主要产物默认写入 `/data/dust/user/zhaiyufe/`；提交任务前检查
  所有输出参数和 Slurm 日志路径，避免把大文件写入 home。
- 成功登录不等于获得无限远程操作授权。提交或取消作业、修改远程代码/数据、安装依赖和删除内容仍
  必须属于用户当前任务的明确范围；操作前先检查远程路径和状态，并保护已有工作。

## 文档

只有代码变化影响稳定事实时才更新文档，不为普通内部实现制造文档噪音：

- 架构、依赖、port 或 public application API 变化更新 `docs/architecture/`；
- 用户 workflow 或控件映射变化更新 `docs/ui/`；
- 安装、依赖和统一命令变化更新 `docs/development.md`；
- 只有需要长期保留原因与权衡的重要决策才创建 ADR。

不要在多份文档复制完整规则。重要新文档应包含 Status、Scope、Related code、Related tests 和
Last verified。移动或删除文件时检查仓库内引用；文档中不写本地绝对路径、token 或机器专属信息。

## 交付格式

最终报告保持简洁并包含：

- 完成了什么；
- 运行了哪些验证及结果；
- `Documentation impact`：更新了哪些文档，或为什么不需要更新；
- 是否改变 public API、配置格式、用户 workflow、依赖方向或兼容层；
- 已知风险或未完成项；没有则明确说没有。

清晰的代码、通过的测试和诚实的影响说明，比更长的规则清单更重要。
