# GIWAXS 处理手册（给 agent：Codex、Claude Code、GUI 里的 Claude、脚本）

原则：**判断交给代码，好奇心留给 agent。**

- GIMaP 的工具负责有确定答案的判断：标定好不好、哪个峰是伪影、一个环能不能算取向、
  笔记里有没有唯一的 αi。它们给出结论和理由，任何模型读到的都一样。
- agent 负责有问题要问的部分：看什么、比较什么、结果对用户的问题意味着什么。

手册分两层。第一层每个 agent 都做，弱 agent 做到这里就合格；第二层是能力够的 agent
在基线之上该做的事。基线的每个决定都是**默认值，不是上限**。

## 0. 底线（所有 agent，只有这四条）

1. **测量值只来自工具**。自己算的派生量（比值、d、晶格常数）要写出用了哪些工具数值。
2. **数据目录只读**。用户没要求就不导出、不改校正。
3. **不编参数**。能量、αi、距离、束心、像素尺寸找不到就问，并说清找过哪里。
4. **分清发现和假设**。数据显示了什么是发现；可能意味着什么是假设，要标明并写出依据
   （如 Δq）。物相只有在用户给了材料、或数据毫无歧义时才当结论。

## 1. 第一层：基线

```bash
python tools/gimap_agent.py auto <图像> [<图像> ...] --notes "<用户给的笔记原文>"
```

- NeXus 多模块序列（`*_m01.nxs … *_m11.nxs`）只给 `_m01`；多个样品一次给全
  （同一探测器只标定一次）。解释器：`D:\conda\envs\GUI\python.exe`。
- 输出在 GIMaP 用户数据目录 `assistant_runs/cli/<时间>/`（或 `--out`），每个样品一个
  `report.md`（另有 `report.json`、曲线 CSV 和 `qmap.png`：q 图上画出分析过的环，白 = 测到、橙 = 阴影、
  红虚线 = 没测到），外加 `summary.md`。四个 Lambda 9M 样品（每个 403 帧）约 80 秒。
- GUI 里的 Claude 和 MCP 客户端用同一个工具 `run_standard_pipeline`（可选，参数同下表）。

| 退出码 | 状态 | 做什么 |
|---|---|---|
| 0 | OK | 转述 `report.md`，数字不改 |
| 2 | NEEDS INPUT | 回答 “Needs attention” 的每一条：每条写了该加的参数（如 `--incidence-deg`）。按第 4 节去找值，找到就重跑，找不到按第 5 节问用户 |
| 1 | FAILED | 把错误原样告诉用户 |

弱 agent 到此为止也合格：基线里所有容易犯的错都已经在代码里挡住了（见第 4 节）。

## 2. 第二层：在基线之上

先读 `report.md` 的 “决策 / Decisions”：每个默认值（哪几帧、哪几个环、哪个标定）是否适合用户的
问题？然后看基线看不到的东西，只要它和问题有关：

| 看什么 | 为什么 | 怎么做 |
|---|---|---|
| 序列怎么变 | 原位实验的意义在变化，终态不是全部 | `auto … --frame 1 --sum 10` 与默认的末帧对比；需要时取中间几段 |
| 所有样品 / 帧都有的线 | 可能来自衬底、窗口、坏点，而不是样品 | 汇总表 “Lines at the same q”；对比初态、标定图、空衬底 |
| 多个峰偏同一相对量 | 几何问题（样品与标样位置不同、αi），不是晶格 | 用户给了材料时，把测到的 q 和已知线逐条比：同一比例的偏移指向几何 |
| 峰的比值 | 晶系、层状、六方的线索 | 只用可靠峰；写出算式（如 3.408/2.942 = 1.158） |
| 被跳过的峰 | 基线只分析最强的 3 个可靠峰（加上用户点名的环） | `ring_orientation` / `crystallite_size` 指定 q |
| 缺失楔边上的极大 | 真实极大可能落在测不到的 χ 里 | `get_curve azimuthal` 看 I(χ)；`set_custom_sector` 只取测到的区域 |
| 笔记、日志、幻灯片 | αi、材料、实验设计常只写在那里 | 读文本（`.pptx/.odp` 是 zip，文字在 slide XML 的 `<a:t>`）；截图要多模态或问人 |
| 用户到底想知道什么 | 决定哪些结果值得追 | 不清楚就问 |

更细的操作：

```bash
python tools/gimap_agent.py status <图像>            # GIMaP 看到的：探测器、帧数、头文件、有没有几何
python tools/gimap_agent.py find-calibration <图像>  # 周围的标定、标样图、日志，已排序并附理由
python tools/gimap_agent.py tools                    # 所有工具的定义
python tools/gimap_agent.py call <图像> steps.json   # 按顺序执行 [{"tool": ..., "args": {...}}]
```

或者用 MCP（第 7 节）在同一帧上连续调用任意工具。

**实例（Yuxin，P03 2021-11，Cu 溅射到 PEO）**：

- 基线给出 4 个样品的峰、f 和尺寸，并挡住了坏点、晕和覆盖不足。
- 第二层又多做了三件事，都是基线做不到的：
  - 读了 `data analysis.pptx`，知道样品是 PEO + Cu；
  - 对比初态和终态，看出 2.56、3.90 在沉积前就有，2.94、3.41 是沉积中出现的；
  - 算出 3.408/2.942 = 1.158 ≈ fcc (111)/(200) 的 1.155，对应 a ≈ 3.70 Å，比 Cu 大 2.3%。
    于是提出假设：样品和标样位置不同。这是需要核对的假设，不是结论。
- 真实的 Claude Code 运行（GUI 助手，Opus 5.5，16 次工具调用，约 4 分钟）还自己发现了一件事：
  - χ > 57° 的区域在所有 q 上都比漫散射背景低 20–150 倍，是阴影；
  - 基线原来的 f = 0.42，以及非晶晕 “面外/面内 = 40”，都是这个阴影造成的，不是取向。
  - 这是有确定答案的判断，所以按第 9 节写进了代码，现在基线也不会再错（第 4 节 “阴影” 一行）。

## 3. 基线的默认决定（都能改）

| 步 | 默认 | 怎么改 |
|---|---|---|
| 能量 / αi / 像素 | 参数 > 头文件 > 笔记里唯一的值；笔记里有两个不同值就不取，改为提问 | `--energy-kev`、`--incidence-deg`、`--pixel-size-um` |
| 几何 | 已有仪器配置就保留；否则在周围找标定（同目录、父目录、兄弟目录、上两层整棵树） | `--calibration <文件>`、`--recalibrate` |
| 标定候选顺序 | GIMaP 能拟合 > 同类探测器文件 > 名字含 giwaxs/waxs > final/redone > 样品之前测的；多模块序列算一个；最多 2 个文件 + 3 张图 | `--calibration` |
| 标样 | 文件名（含 `lab6_ceo2` 混合）> 全部拟合比较 | `--standard` |
| 标定验收 | 标样的线落在的 q：平均误差 ≤ 0.2% 好，≤ 0.5% 可用；不看像素残差 | — |
| 帧 | 序列取末 10 帧求和（终态） | `--frame N --sum K`（负数从末尾数） |
| 模式 | 自动识别不是 GIWAXS 就切到 GIWAXS | GUI 里分析 GISAXS |
| 环和尺寸 | 面积最大的 3 个可靠峰 + 用户点名的环；晕不给尺寸；sin χ 加权覆盖 < 80% 不给 f，阴影区算未测 | `--rings`；或直接调工具 |

代码：`src/gimap/features/assistant/application/pipeline.py`，判断规则在 `domain/`。

## 4. 情况 → 怎么处理

来自真实数据。“代码” 一栏是工具已经做的，“agent” 一栏是还要做的。

| 情况 | 怎么认出来 | 代码 | agent |
|---|---|---|---|
| 没有几何 | status `geometry: null` | 在周围找标定、拟合、验收 | 失败时看 Decisions；用户给了路径就 `--calibration` |
| 标定在别的目录、文件夹名不一致（`lmbd` ↔ `lmbdp03`，`p2m` ↔ `embl_2m`） | 标定在 `../../Calibration/...` | 按文件类型、模块序列、图像尺寸匹配，不看文件夹名 | 汇报用了哪一个 |
| 多个候选（`_00001/_00002`、`redone`、`redone_final`） | 列表里同一标样多个版本 | final > redone > 普通，样品之前 > 之后 | 同上 |
| 混合标样 | 名字含 `lab6_ceo2` | 两套线一起拟合 | — |
| 像素残差大（rms 4.2 px） | 宽角、探测器倾斜 | 以线位置为准（本例 0.06%） | 不要因为 rms 大就拒绝 |
| 束心在探测器外 | “limited azimuthal coverage” | 线检查确认后注明 | 照写 |
| 能量 | NeXus 头文件（11.8 keV） | 自动读 | TIFF 没有 → 找笔记/日志 |
| αi 缺失 | Needs attention: “incidence angle αi” | 先用 0° 出结果，同时提问 | 查笔记、日志、**幻灯片**（本例 0.4° 只在 `.odp` 截图里） |
| 像素尺寸缺失 | TIFF 无头文件 | 提问 | Pilatus 172 µm、Eiger 75 µm、Lambda 55 µm |
| 原位序列 | `frames: 403` | 末 10 帧求和 | 第二层：对比初态 |
| 扇区覆盖不到 | “only the in-plane sector is measured here” / “not measured” | 明说，不给比值 | 不要说成 “只在面内” |
| 环覆盖太少 | “Only …% of the orientation range Herman's f weighs (sin χ) is measured” | f 按 sin χ 加权，靠近面内的部分最重要；加权覆盖 < 80% 不给 f，给出测到的范围 | 需要时看 I(χ) 原始曲线 |
| 阴影 / 遮挡 / 不灵敏区 | 强度低于同一 q 漫散射背景的 20%（本例 χ > 57°，低 20–150 倍） | 这些 χ 算未测；扇区在阴影里就不比较（“the other is shadowed”）；部分覆盖时给出同范围各向同性环的 f 作对照 | 看标定图同一区域是否也暗；需要时加掩模或用 `set_custom_sector` 只取亮区 |
| 缺失楔 | “\|χ\| < 13° is not measured” | f 偏低，照注 | 照写；第二层考虑极大在楔里 |
| 两个等高极大 / f ≈ 0 | “no single preferred orientation” | 不说取向 | 照写 |
| 坏点 / 模块边缘 | caveat “a spike”（本例 4.893） | 不做环和尺寸，不进序列判断 | 标为伪影 |
| 非晶晕 | caveat “a broad halo”（FWHM > 15% q） | 不给尺寸 | 可以说有晕 |
| 晕上的尖峰 | 晕和尖峰都列出 | 都保留 | 分开描述 |
| 所问 q 没有峰 | ring 结果 `analysed: no peak at q = …` | 分析最近的特征并说明 | 不要当成所问的峰 |
| 所有样品共有的线 | 汇总表 “Lines at the same q” | 列出 | 第二层：追查来源 |
| 中文 Windows GBK 乱码 | UnicodeEncodeError | CLI 强制 UTF-8 | 自己写脚本也要 UTF-8 |
| 大文件慢 | 2.4 GB NeXus | 加载超时 15 分钟 | 等，不要中途重跑 |

## 5. 问用户

一次问完：缺什么、找过哪里、可选项（带文件时间）。例如：

> 这组数据没有入射角 αi。我查了：命令行笔记、`Calibration/` 下 60 张标定图的文件名、4 个 `.log`
> （只有时间戳）。`DFG_Nov2021_calib.odp` 里可能有，但是截图，我读不了。请告诉我 αi（常见 0.1–0.5°）。

能从文件得到的值（距离、束心）不要问。

## 6. 汇报

1. **基线**：标定（文件、标样、线误差）、αi 和能量的来源、分析了哪些帧；峰表、面内/面外、
   取向（f 和原话、覆盖率、缺失楔）、尺寸下限——照 `report.md`。
2. **发现**：第二层看到的东西，每条写出依据（哪个工具、哪些数值）。
3. **假设**：标明 “假设”，写出支持的数据和怎样能证实或否定。
4. **未解决**：Needs attention 和你没能回答的问题。

## 7. 工具和接入

- **Codex**：读仓库根的 `AGENTS.md`，它指向本手册。不用配置就能用命令行。
  需要在同一帧上连续调用工具时加 MCP（`~/.codex/config.toml`）：

  ```toml
  [mcp_servers.gimap]
  command = 'D:\conda\envs\GUI\python.exe'
  args = ['E:\PythonCode\gisaxs_gui\tools\gimap_agent.py', 'mcp']
  startup_timeout_sec = 60
  tool_timeout_sec = 900
  ```

  或 `codex mcp add gimap -- D:\conda\envs\GUI\python.exe E:\PythonCode\gisaxs_gui\tools\gimap_agent.py mcp`
  （再把 `tool_timeout_sec` 调到 900：标定一个大 NeXus 要一分钟）。
- **Claude Code**：`claude mcp add gimap -- D:\conda\envs\GUI\python.exe E:\PythonCode\gisaxs_gui\tools\gimap_agent.py mcp`
- **MCP 工具**：`open_frame(path, notes)`，然后是 GUI 助手的全部工具：`run_standard_pipeline`（可加
  `out_dir` 写文件）、`find_peaks`、`ring_orientation`、`set_frame`、`calibrate_geometry` 等。
- **GUI**：Process with Claude 里的 Claude 有同一个 `run_standard_pipeline`，可以先拿基线，再自由用其他工具。

## 8. 评估（两层分开评）

```bash
python tools/eval_giwaxs_agent.py baseline     # 在真实数据上跑基线，检查不该错的事
python tools/eval_giwaxs_agent.py report <agent 的报告.md>   # 给 agent 写的报告打分
```

- **第一层：不犯错。**
  - `baseline` 检查：用对了标定文件、线误差 ≤ 0.2%、坏点被标出、没有假序列、2.94 环的 f 在合理范围、
    给了 αi 就没有未解决问题。
  - `report` 检查报告里的错误说法（六方序列、把坏点当衍射峰、给晕算晶粒、13% 覆盖报 f）。
    命中的句子会打印出来，由人确认——否定句（“没有六方序列”）不算。
- **第二层：发现更多。** `report` 给 “发现” 打分：缺失楔的影响、初态与终态的对比、共有线、fcc 比值或
  q 偏移、阴影区。只能作为参考，最终要人读报告。
- 上面那次真实运行的得分：错误 0，第一层事实 4/4，第二层 8/8。评分时只给 agent 自己写的报告打分，
  不要把工具日志一起放进去：日志里的原始输出会被误判成 agent 的说法。
- 案例和规则在 `docs/agents/eval/yuxin-p03-2021.json`，可以加新案例。

## 9. 维护：遇到新情况

- **有确定答案的判断出错**（如把坏点当峰）：
  - 用 `call` 复现；
  - 把判断写进 `src/gimap/features/assistant/domain/` 的函数，输出一句带理由的结论；
  - 在 `tests/test_assistant_rules.py` 加测试，在第 4 节加一行。
- **需要判断力的新情况**（如一种新的实验设计）：写进第 2 节的表，不要写成代码或硬规则。
- 不要为了弱模型把强模型能做的事禁掉：弱模型靠第一层的代码保护，第二层留给能做的 agent。
