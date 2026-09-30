# GIMaP 项目说明（给 Codex、Claude Code 等 agent）

GIMaP 是 Python / PyQt5 的掠入射 X 射线散射（GISAXS、GIWAXS）分析桌面软件：打开探测器帧 →
几何 → 掩膜与校正 → 切线 → 结果 → 导出，另有 Fitting（一维模型拟合）和 Labs（二维预测、训练集、分类）。
入口 `main.py`；开发解释器 `D:\conda\envs\GUI\python.exe`。

## 代码结构（架构测试会检查）

- `src/gimap/features/<feature>/` 拥有该功能的 `domain`（纯科学逻辑）、`application`（用例、端口，
  presentation 只从这里导入）、`infrastructure`（适配器）、`presentation`（Qt）和 `bootstrap.py`。
- feature 之间不互相导入；需要组合时在 `src/gimap/app/`（组合根：`main_window.py`、`runtime.py`、
  `headless_assistant.py`）里注入，例如 Fitting 的快速物理拟合经 `create_quick_fit()` 注入 assistant。
- `src/gimap/app/presentation/` 放应用壳和共用组件（`components/`：DetectorView、CurvePlot、StepRail、
  Toast…；`theme/` 的 token + QSS；`i18n/` 的界面语言）；`src/gimap/shared/` 放稳定复用能力（探测器读取、几何）。
- 单个模块 ≤ 600 行；静态布局放在 `presentation/views/*_view.py`，并登记在 `tests/test_ui_source_of_truth.py`；
  行为放在 `bindings/` 的 mixin 里。
- 顶层 `core/`、`config/` 仍被组合根使用；`utils/ML_Fitting_1D_GISAXS/` 是研究代码，GUI 只以子进程调用。
- 架构与契约文档在 `docs/architecture/`（先看 `overview.md`、`scientific-data-flow.md`、`ui-design-principles.md`）。

## 科学数据契约

- 显示优化不改科学数值、单位、数组方向和数据谱系。Analyze 里 q 用 Å⁻¹（Fitting 内部 nm⁻¹，界面标明单位）；
  规范像素坐标：第 0 行在上，光束中心 = 直射束位置。
- 读取器保留存储值（`metadata["stored_dtype"]`）。整数计数探测器的负值是缝隙码；浮点帧（扣暗场/背景）
  的负值是数据（`valid_pixels(..., negatives_valid=True)`）。孤立的热像素、死像素在每帧自动排除
  （`domain/bad_pixels.py`，可在 Mask 步骤关闭），数量记入导出元数据。
- 每条曲线都能在图上显示来源（Sources）；导出旁边写 JSON 记录（设置、几何、校正）。
- GIWAXS 强度校正（立体角、偏振、薄膜吸收）默认关闭；开启时 I / factor、q 不变，计数误差按 value / factor 传递，
  JSON 记录写明公式与参数（`domain/intensity.py`）。

## UI 约定

- 用户每一步都要有反馈：加载状态、步骤栏状态、进度条 + 取消、写文件后的 toast（可打开文件夹）。
  常用操作放在显眼处，高级选项折叠但可发现；新功能出现时能被注意到，也能关掉。
- 颜色用 theme token（浅色/深色都要看）；界面文字在代码里写英文，中文在 `i18n/zh.py`（按英文原文精确匹配，
  改了英文要同步改表）。数值、单位、文件名不翻译。
- 不需要 AI 的流程必须完整：Run Automatic Analysis（GIWAXS 与 GISAXS）、Series 热图、批量导出都不依赖模型。

## 测试与检查

- 修改后跑相关 focused tests；UI 用 `QT_QPA_PLATFORM=offscreen`。完整检查：`python tools/check.py`
  （pytest + offscreen smoke + ruff）。`tests/research` 在 Windows 上有已知失败，交付时说明环境限制。
- 真实公开数据在 `tests/data/external/`（GALAXI GISAXS、pyFAI AgBh、P08 与 Xeuss GIWAXS，许可证见其 README）；
  本地 `TestSAXSdata/` 有 P03 Pilatus/Lambda 帧，不存在时相关测试跳过。
- 截图检查：offscreen 下设 `QT_QPA_FONTDIR=C:/Windows/Fonts`、`app.setFont(QFont("Segoe UI"))` 并 `apply_theme()`。

## 工作规则

- 保留用户已有的 tracked / untracked 修改；未经要求不 commit、push、stash、reset。
- 不写 git index：改名用普通 `mv`，不用 `git mv`。
- 用户数据目录（如 `E:\ExpData`）只读；输出写到临时目录或用户指定的位置。

## 处理散射数据（不是改代码时）

- GIWAXS 读 `docs/agents/giwaxs-playbook.md`，GISAXS 读 `docs/agents/gisaxs-playbook.md`。第一层是基线：
  `python tools/gimap_agent.py auto <图像...> --notes "<用户笔记原文>"`（GISAXS 加 `--technique gisaxs`），
  转述各样品的 `report.md` 并回答其中 “Needs attention” 的问题。第二层：基线的决定只是默认值，
  能力够就继续追查（序列前后对比、共有的线、峰的比值、切线位置与两半、模型之间的差别、笔记和幻灯片）。
- 底线四条：测量值只来自工具（派生量写出来源）；数据目录只读；不编参数；分清发现和假设。
- 有确定答案的判断出错时，写进代码和 `tests/test_assistant_rules.py`；需要判断力的新情况写进手册，
  不要变成硬规则。
