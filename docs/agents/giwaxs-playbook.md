# GIWAXS 处理手册

适用于命令行 agent（Codex、Claude Code、脚本）和 MCP 客户端。GISAXS 见 [gisaxs-playbook.md](gisaxs-playbook.md)，
仓库规则见 `AGENTS.md`。GIMaP 里的 Process with AI 不读这份手册：它的提示在
`src/gimap/features/assistant/application/prompts.py`，改这里的规则时要同步改那里。

## 底线

1. **测量值只来自工具。** 界面和 JSON 记录能复现的就是这些值。自己算的派生量（比值、d、晶格常数）要写出算式和
   用到的工具数值。
2. **数据目录只读。** 那是用户的、常常未发表的数据；所有输出写到 `--out`。命令行和 MCP 本来就不往数据旁边写；
   GUI 里则不同：`export_results` 写到数据旁边的 `gimap_analysis/`，只在用户要文件时调用。GUI 里的 `use_geometry`
   会把几何存成磁盘上的仪器配置，之后命令行默认会用它（Decisions 会写明）。
3. **不编参数。** 能量猜错，所有 q 都跟着缩放；αi 猜错，qz 跟着偏。去头文件、笔记、日志、幻灯片里找，
   找不到就问，并说清找过哪里。
4. **分清发现和假设。** 数据显示了什么是发现；可能意味着什么是假设，要写出依据和怎样证实。
   物相在用户给出材料、或者数据毫无歧义之前，都只是假设。

## 基线：做了什么，能信到什么程度

```bash
python tools/gimap_agent.py auto <图像> [<图像> ...] --notes "<用户笔记原文>" --out <文件夹>
```

- NeXus 给任意一个模块文件即可（如 `_m01`）：同前缀的 `_m02`、`_m03` 等探测器模块会拼成一张图，帧在文件内部；
  同一组模块只给一次。多个样品一次给全：同一批里尺寸相同的帧沿用前面从文件或标样图得到的标定。
- 每个样品写一个 `<out>/<名字>/` 文件夹，包含：
  - `report.md`、`report.json`；
  - `radial.csv`、`in_plane.csv`、`out_of_plane.csv`、`azimuthal_last_ring.csv`；
  - `qmap.png`：q 图上画出分析过的环，白色是测到的，橙色是阴影，红色虚线是没测到的。

  外加整批的 `summary.md` 和 `summary.json`。`<名字>` 是文件名去掉扩展名：不同文件夹里同名的帧会写进同一个文件夹，
  互相覆盖而不报错。这种情况要分开跑，各用一个 `--out`。
- 耗时：给了 `.poni` 的单帧约 10 秒；要从标样图标定或读大的 NeXus 时，一帧几十秒到一分多钟；
  四个 Lambda 9M 原位样品（每个 403 帧）约 2.5 分钟。
- 退出码：
  - **0**：全部分析完，没有待回答的问题。
  - **2**：至少一个样品有 “Needs attention”。可能是缺一个值（会写出补这个值的选项）、一个选项定不了的判断，
    或者没找到几何、什么也没分析。
  - **1**：至少一帧失败，比如读不了的文件；其他帧照样有报告。1 优先于 2：有失败也有待回答的问题时退出码是 1，
    仍要读其他样品的 Needs attention。

基线的价值在于两点：判断标准始终一致（下面的阈值），每个默认决定和理由都记在 `report.md` 的 **Decisions** 里。
它对用户的问题未必合适。转述数字之前先读 Decisions，下面几处最容易“数字没错，问题不对”：

| Decision | 默认 | 什么时候不合适 | 怎么改 |
|---|---|---|---|
| technique | GIMaP 应用几何后自动识别：探测器任一角超过 2θ = 20° 就是 GIWAXS，否则 GISAXS | 小探测器或长距离的 GIWAXS 会被认成 GISAXS | `--technique giwaxs` |
| geometry | 依次尝试：`--calibration` → 笔记里点名的标定文件 → 已保存的仪器配置 → 在附近搜索（同目录及子目录、父目录和兄弟目录、往上两三层；2 个标定文件 + 3 张标样图） | 仪器配置只按探测器名和图像尺寸匹配，不管是哪次束线实验；搜索也可能挑到别的实验的标定，“quality: good” 只说明标定和它自己的标样一致；`.poni` 和 GIMaP 的 `.json` 直接采用，不再用标样核对 | `--calibration <文件>`、`--no-saved-profiles`、`--recalibrate` |
| frames | 序列取最后 10 帧求和（终态） | 问题关心的是过程，或者开头 | `--frame N --sum K`（负数从末尾数；只给 `--frame` 时仍从该帧起求和 10 帧，单帧用 `--sum 1`） |
| rings | 面积最大的 3 个没有警告的峰算取向和尺寸 | 用户问的是别的环 | `--rings N`；指定某个 q 要用 `call` 或 MCP 的 `ring_orientation`、`crystallite_size` |
| 能量 / αi / 像素 | 能量：选项 > 头文件 > 笔记（只认 “12.4 keV” 这种写法；波长要自己换算后传 `--energy-kev`）> 标定文件的波长（这时 Decisions 里没有 energy 一行，能量来源就是几何来源）。αi：选项 > 笔记 > 仪器配置（只在几何也取自仪器配置时），都没有时先用 0° 并提问。像素：选项 > 头文件 > 笔记。笔记里的值只有唯一时才取 | 笔记里有两个不同的值，或者有效值只在幻灯片、日志里 | `--energy-kev`、`--incidence-deg`、`--pixel-size-um` |
| 标样 | `--standard` > 笔记里唯一点名的标样 > 文件名 > 所有标样拟合比较（笔记和文件名不一致时也比较） | 混合标样的写法：`LaB6+CeO2` 是混合，`LaB6, CeO2` 是两种 | `--standard` |

标定验收看的是标样的线落在的 q：平均误差 ≤ 0.2 % 为好，≤ 0.5 % 可用（需要至少 3 条线）。测到 3 条以上线时不看
像素残差：宽角和探测器倾斜时 rms 会很大（几个像素），线的位置仍然可以很准；少于 3 条线时才退回看环数和像素残差（≤ 1.5 px）。

## 基线没看的地方：判断力最有用的地方

- **峰只在径向 I(q) 上找。** 只出现在某个扇区的峰会漏掉：对 `in_plane`、`out_of_plane` 曲线也跑 `find_peaks`。
- **只有终态。** 原位实验的意义在变化：`auto --frame 1 --sum 10 --out A` 加上默认的那次运行，再比较两份
  `report.json`。GUI 的 “Compare with the Start of the Series” 用同一套规则：只比较可靠峰的有无和位置（appeared、
  disappeared、shifted、moved?；grew 是开始时弱、结束时可靠，faded 相反），不比较强度。对两份 report.json 可以直接调用
  `src.gimap.features.assistant.application.series_markdown(start, end)` 得到同一张表。
- **本批所有样品都有的线**（≥ 2 个样品时 summary.md 的 “Lines at the same q”，±0.3 %，包括带提示的峰）可能来自衬底、
  窗口或探测器，而不是样品：对比初态、标定图和空衬底。
- **几个峰按同一比例偏移**，指向几何问题（样品与标样的位置即距离不同，或能量不对），而不是晶格变化；
  αi 主要移动 qz，几乎不改 |q|。用户给了材料时，把测到的 q 和已知线逐条比较。
- **峰的比值是晶系的线索。** 基线只检查层状和六方序列（2 % 以内），从不判定物相。
  只用可靠峰，并写出算式，例如 q₂ / q₁ = 1.158，与 fcc (200)/(111) 的 1.155 相比。
- **Scherrer 尺寸是下限**：基线没有扣除仪器展宽。有标样的峰宽时，用 `crystallite_size(q_center, instrumental_fwhm=…)`
  扣除（高斯平方相减），并写出宽度从哪里来。
- **缺失楔边上的极大值**：真正的极大可能落在测不到的 χ 里。
  先 `ring_orientation(q_center=…)`（它设定这个环的 q 窗口），再 `get_curve azimuthal` 看 I(χ)；
  用 `set_custom_sector` 只取测到的区域。
- **笔记、日志、幻灯片**：αi、材料和实验设计常常只写在那里。GIMaP 的工具只读图像、`.poni` 和文本或日志文件；
  `.pptx` / `.odp` 要自己读（文字分别在 `ppt/slides/*.xml` 和 `content.xml` 里），截图需要多模态或问人。

## 报告里的提示语：含义和阈值

数据和某个提示矛盾时，拿工具的证据说话（`get_curve`、调整 `min_snr` 或背景窗口后的 `find_peaks`），并写出来。
如果它是有确定答案的判断，就去改规则（见文末“维护”）。

| 提示 | 阈值 | 排除了什么 |
|---|---|---|
| weak | SNR < 5（3–5σ 视为暂定） | 不算可靠峰 |
| a spike | 窄到分辨率极限（FWHM < 3 个 bin），且高于局部背景 5 倍 | 当作热像素、模块边缘或宇宙射线；不做环、尺寸和序列 |
| a spike-like artefact (flat top …) | 平顶宽 ≥ 15 个 bin，至少一侧边缘在 1 个 bin 内陡升 | 探测器的一行或一列、模块边缘；同上处理。窄于 15 个 bin 的方块不会被识别 |
| a broad halo | FWHM > q 的 15 % | 非晶有序；不给晶粒尺寸 |
| shadowed | 强度低于同一 q 漫散射背景的 20 % | 这些 χ 算作没测到；扇区落在阴影里就不比较 |
| Only …% of the orientation range | sin χ 加权覆盖 < 80 % | 不给 Herman f，只给测到的范围 |
| not measured (missing wedge) | 靠近 qz 的 \|χ\| 没有像素 | f 偏低，照实写 |
| no single preferred orientation | 两个极大高度差在 15 % 以内、相距 ≥ 30° | 不说取向 |
| weak or no preferred orientation | \|f\| < 0.1 | — |
| mainly in-plane / out-of-plane | 两个扇区之比 ≥ 2 | — |
| its shape could not be fitted | 高斯拟合失败，但信号 ≥ 2 × min_snr（默认 6σ） | 不算可靠峰；看图像上这个 q（模块缝隙、探测器边缘、重叠） |
| `at_edge`（GUI 显示 “at the end of the data: check”） | 峰在 q 范围末端 1.5 个峰宽以内 | 不排除：仍算可靠峰，可能被选去算环和尺寸；形状、位置和宽度可能被截断，要自己核对 |

所问的 q 处没有峰时：`ring_orientation` 的结果里 `analysed` 字段写 “no peak at q = … the nearest feature … was
analysed”；`crystallite_size` 不提示，直接用 2 个峰宽内最近的峰，要核对它返回的 q。两者分析的都可能不是所问的峰。

## 问用户

一次问完：缺什么、找过哪里、有哪些候选（附文件时间）。能从文件得到的值（距离、束心）不要问。例如：

> 这组数据没有入射角 αi。我查过：你的笔记、`Calibration/` 下 60 张标定图的文件名、4 个 `.log`（只有时间戳）。
> `…_calib.odp` 里可能有，但那是截图，我读不了。请告诉我 αi（常见 0.1–0.5°）。

## 汇报

围绕用户的问题来组织，不必套固定格式，但要做到：

- 每个数都能追到工具或来源；
- 派生量写出算式；
- 写明标定（文件、标样、线误差）、αi 和能量从哪里来、分析的是哪些帧；
- 假设要标为假设，并写出支持它的数据和证实或否定的方法；
- 列出 Needs attention 里和你没能回答的问题；
- 需要时指向 `qmap.png` 和 CSV。

## 工具和接入

```bash
python tools/gimap_agent.py status <图像>            # GIMaP 看到的：探测器、帧数、头文件、有没有几何
python tools/gimap_agent.py find-calibration <图像>  # 周围的标定文件、标样图、日志，已排序并附理由
python tools/gimap_agent.py tools                    # 所有工具的定义（34 个）
python tools/gimap_agent.py call <图像> steps.json   # 按顺序执行 [{"tool": ..., "args": {...}}]，--out 指定输出
```

每次 `call` 都是一个新会话：步骤里先放 `run_standard_pipeline`（带 calibration、incidence_deg 等）或 `use_geometry`，
再放 `find_peaks`、`ring_orientation`、`crystallite_size`；`crystallite_size` 需要同一会话里先对同一条曲线跑过
`find_peaks`。MCP 的会话在两次 `open_frame` 之间一直保持。

- **MCP**：Codex 写在 `~/.codex/config.toml`，或用 `claude mcp add`：

  ```toml
  [mcp_servers.gimap]
  command = 'D:\conda\envs\GUI\python.exe'
  args = ['E:\PythonCode\gisaxs_gui\tools\gimap_agent.py', 'mcp']
  startup_timeout_sec = 60
  tool_timeout_sec = 900   # 标定一个大的 NeXus 要一分钟以上
  ```

  ```bash
  claude mcp add gimap -- D:/conda/envs/GUI/python.exe E:/PythonCode/gisaxs_gui/tools/gimap_agent.py mcp
  ```

  （在 bash 里用正斜杠：反斜杠会被吞掉。长调用超时时，调大 `MCP_TOOL_TIMEOUT`，单位毫秒。）
  `--no-saved-profiles` 只能写在启动参数里（`… gimap_agent.py mcp --no-saved-profiles`），会话中不能切换。

  - 先用 `open_frame(path, notes, out_dir)` 打开帧，然后可以连续调用所有工具。
  - `run_standard_pipeline` 的参数：
    - 和命令行同名的：`calibration`、`standard`、`energy_kev`、`incidence_deg`、`pixel_size_um`、`frame`、`rings`、
      `recalibrate`、`technique`；
    - 和命令行不同的：`sum_frames`（对应 `--sum`）、`fit`（对应 `--no-fit`）、`out_dir`；
    - 没有 notes 参数，笔记来自 `open_frame`。
  - 同一会话里用 `technique=X` 跑过一次后，之后不指定 technique 的运行也沿用 X。
- **GUI**：Analyze 的 Run Automatic Analysis 不需要 AI，跑的是同一套流程。
  Process with AI（Tools 菜单，或 Analyze 的 Ask AI…）可以先拿基线，再自由使用其他工具；
  它的 `run_standard_pipeline` 在 Analyze 处于 Auto 时跑 GIWAXS（已选模式时跑所选模式），所以 GISAXS 任务要传 `technique`。

## 评估

```bash
python tools/eval_giwaxs_agent.py baseline             # 在案例数据上跑基线，检查不该错的事
python tools/eval_giwaxs_agent.py report <报告.md>      # 给 agent 写的报告打分（错误说法、发现）
```

- 案例和规则在 `docs/agents/eval/*.json`。现有的一个案例需要用户的本地数据（只读）。
- `report` 只给 agent 自己写的报告打分。不要把工具日志放进去，日志里的原始输出会被误当成 agent 的说法。
  命中的句子会打印出来，由人确认。只有规则的 `unless` 里列了否定词时，否定句才不算（如 “没有六方序列”）；
  “4.89 不是衍射峰” 这类句子仍会被标出，要人判断。
- 已知不过的一项：“spike at 4.893 flagged”。逐帧排除缺陷探测器行之后，原来那个窄尖峰已经被去掉；留下的平顶方块在
  q ≈ 4.875 处，已被标为伪影。是否改案例的预期，由用户决定。其余检查都应通过，以实际运行 `baseline` 的结果为准。

## 维护：遇到新情况

- **有确定答案的判断出错**（如把坏点当成峰）：
  - 用 `call` 复现；
  - 把判断写进 `src/gimap/features/assistant/domain/` 的函数，输出一句带理由的结论；
  - 用合成数据在 `tests/test_assistant_rules.py` 加测试；整次运行的回归加一个 eval 案例；
  - 在上面的提示语表里补一行。
- **需要判断力的新情况**（如一种新的实验设计）：写进“基线没看的地方”，不要写成代码里的硬性拦截。
