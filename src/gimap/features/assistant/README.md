# Assistant feature（Process with AI）

三种“大脑”（Settings ▸ Assistant 或开始对话框里的 Brain 选择）：

- **Claude Code（默认）**：用本机的 Claude Code 和用户自己的 Claude 订阅（Pro / Max），不需要 API key。
  GIMaP 在 127.0.0.1 的随机端口起一个一次性的 MCP 端点（`LocalMcpServer`，带随机 token），以
  `claude -p`（stream-json 输入输出）无头运行 Claude Code，只开放 GIMaP 的工具（`--tools ""` 关掉
  shell / 文件 / 网络，`--strict-mcp-config`、`--setting-sources=`、`--disable-slash-commands`、
  `--permission-mode dontAsk`），并从子进程环境里去掉 `ANTHROPIC_*` 和父 Claude 会话的变量，
  所以用的是 `claude auth login` 登录的账号额度，而不是按 token 计费。
- **Claude API**：API key，按 token 计费（`AnthropicAssistantLlm`）。
- **其他 AI 提供方（OpenAI 兼容）**：`infrastructure/openai_compat_llm.py` 用官方 `openai` SDK 接
  `chat.completions`（流式、工具调用、DeepSeek / Qwen 的 `reasoning_content` 显示为思考、用量含缓存命中、
  老服务器不认 `stream_options` 时自动去掉重试、参数不是 JSON 时重问、错误翻成 `LlmError`）。运行循环
  不变：适配器把 Anthropic 块（text / tool_use / tool_result）和聊天消息（assistant.tool_calls / tool）
  来回翻译。预设在 `application/providers.py`（OpenAI、DeepSeek、通义千问、Kimi、智谱 GLM、Gemini 兼容
  端点、OpenRouter、硅基流动、Azure OpenAI、Ollama、LM Studio、自定义），只给地址、key 的环境变量和几个
  模型建议；设置页可“Get List”向提供方要模型列表，“Test Connection”发一次带工具的小请求。key 按提供方
  存在用户数据目录 `assistant_provider_keys.json`（`ProviderKeyStore`，也认各家的环境变量）。设置页的
  这一节在 `presentation/provider_section.py`。

在 Analyze 打开并分析一张 GIWAXS 帧后，点 **Process with AI…**：选择要得到的结果
（峰位表、面内 / 面外取向、单个环的取向分布、晶粒尺寸）、写下样品说明和权限，Claude 通过
工具调用操作 Analyze（切换模式、改扇区、设 I(χ) 窗口……），GUI 实时跟随；右侧 Claude 面板
显示思考摘要、每一步工具调用和最后的报告。做不到的项目标成 `not_available` 并写清原因
（信号不足、覆盖缺失、缺少几何、缺少工具）。

## 没有几何的帧

没有匹配的仪器配置时不报错，而是自己找标定：用户在备注 / 长期指令里写的文件或文件夹先用
（`path_candidates` 从中文句子里也能取出路径，按磁盘上真实存在的最长路径解析）；
`find_calibration_files`（`LocalFileExplorer`：帧所在文件夹及子文件夹、上一级及全部同级文件夹、
上两级及其子文件夹、上三级中名字像 calib / log / standard 的文件夹；有条目数与时间上限）和
`search_files`（任一上级文件夹或用户给的文件夹，可按名字过滤）列出 `.poni`、GIMaP 标定 `.json`、
标样图像（按文件名 / 文件夹名识别 AgBH、LaB6、CeO₂；Si、Al₂O₃、Cr₂O₃ 只识别不拟合，作衬底时
不误判）和日志。读取规则（`calibration_tools.py`）：帧往上两级内、搜索找到的、用户点名的文件
直接读；更远的文件确认模式下先问、全自动模式直接读；隐藏 / 系统文件夹（.ssh、AppData、Windows…，
只看与帧共有的上级以下部分）和非图像 / 标定 / 文本类型永不读。`inspect_file` 读图像头、`.poni`
（pyFAI 的 Fit2D 换算得到直射束中心）、标定 JSON 和日志里与几何有关的行；
`calibrate_geometry` 用 Calibration feature 的 `HeadlessCalibration`（app 组合根注入，
不跨 feature 导入）拟合，并检查标样的线实际落在哪个 q（`line_check`，平均相对误差 ≤ 0.2% 为好；
宽角倾斜探测器上像素残差会误导，所以不以它为准）；标样不明时 `standard="compare"` 每种支持的标样
（含 LaB6 + CeO₂ 混合）各拟合一次，按线位置误差给出判断；NeXus 多模块序列只列一次；`ask_user` 弹出“Claude asks”列表（带文件时间和文件夹，可输入数值），
只在数字判断不了时用；
`use_geometry`（写操作，确认模式先问）通过 `AnalyzeAutomation.use_geometry` 存为仪器配置并重新分析。
报告里有“Geometry used for this frame”和“Calibration fits”两张表。`auto` 标样可能拟合错
（合成 AgBH 上选成 LaB6、34 mm），所以提示词要求按文件名选标样。

## 可预览的操作（AI 的回答不只是文字）

`application/operations.py` + `operation_tools.py`：模型对 Analyze 设置的每次改动（扇区、自定义扇区、
q 框、帧、模式、αi、径向 bin、有效强度范围）都记成 `Operation`，带“恢复原状”的参数（从改动前的状态
算出，`inverse_arguments`），所以能精确撤销。`propose_operations` 让模型只提建议：每条先预览（应用 →
截 q 图 → 恢复，按给出的顺序累积），再作为卡片等用户处理。权限 `PERMISSION_PREVIEW`（“先预览”）：
运行结束时把模型改过的设置全部恢复，每个设置只留最后的净改动；探索步骤、回到起点的改动、看不出效果
的改动（`setting_state` 比较改动前后的有效值，`no_effect`）都标为 `superseded` 不显示；模型对某个设置
明确提了建议（`propose_operations`），这个设置就只显示建议——它结束时停在的值可能只是最后一次尝试。
面板的 **Changes** 区（`presentation/operation_cards.py`）：图、标题、原因、效果，Apply / Dismiss /
Undo、Undo All；控制器在 GUI 线程用一个新的 `ToolCatalog` 执行。面板底部可以追问
（`AssistantController.follow_up`：同一帧、同一设置的新运行，带上次报告的摘要），回答可以带新卡片。
真实运行（Opus 5.5，样品 117）用 `propose_operations` 提了“近面内替代扇区 χ 36–46°”“用强度下限近似
掩掉阴影（10 帧求和）”这类有理由的卡片，并用 `note_missing_capability` 记下缺“按 χ–q 区域掩膜”。

## 引导式页面（不用 AI）

`presentation/guided_page.py`（文字和小部件在 `guided_text.py`）：一个按钮在 Analyze 的当前帧上跑
`StandardPipeline`（工作线程，GUI 操作经 `GuiBridge`），五步显示：数据（笔记里自动认出 αi / 能量 /
像素）、几何（来源 + 质量徽章）、检查（spike、晕；阴影和缺失楔各一行（`coverage_notes`，逐环的原句
在提示里）；q 图上画出分析过的环：白 = 测到、橙 = 阴影、红虚线 = 没测到，`AnalyzeAutomation.preview_png(
rings=ring_overlays(report))`）、结果（峰表：简短的面内/面外说明 + “可信 / 伪影 / 晕”，表头提示每列怎么算；
选中一个峰，右边的 q 图画出它的环；数据末端的峰标“check”；环取向，尺寸；序列可“与开始对比”，
`series_changes` 按峰宽配对（相距小于半个 FWHM 算同一条线）给出出现 / 消失 / 变强 / 变弱 / 移动，
消失和出现相距 1% 以内合成一行“moved?”）、报告（“Save Report…”存成一个网页：`guided_report.py`，
q 图带环、I(q) 标出峰（`AnalyzeAutomation.curve_png`）、序列对比表，图片内嵌，任何浏览器能开；也可存 Markdown）。
只有人知道的值变成输入框，“Run Again” 用这些回答，之前的回答保留。由 `bootstrap.create_guided_page`
组装，`app/main_window.py` 放进侧栏（Start 页在 `app/presentation/home_page.py`）。

## 基线和探索：判断交给代码，好奇心留给 agent

`StandardPipeline`（`application/pipeline.py`）把标准 GIWAXS 流程写成代码：状态 → 几何（已有
仪器配置，或周围找到的最佳标定，按标样线位置验收）→ 序列末 10 帧求和 → 找峰 → 面内 / 面外 →
最强可靠峰（加上用户点名的环）的环取向与尺寸。每个决定连同理由记进 `decisions`；只有人知道的值
（αi、能量、像素、选哪个标定）进 `needs_attention`，并写明用哪个参数回答。它调用的就是
`ToolCatalog` 的工具，所以数字与 GUI 完全一致。`pipeline_report.py` 把结果写成 Markdown（先问题、后数字）。

它是**可选的基线，不是流程上限**：在 GUI 里是工具 `run_standard_pipeline`（完整表格存进
`RunResults.pipeline`，模型拿到的是精简版；内部的写操作照样按权限询问，用户拒绝后不再追问），
模型拿到基线后继续用其他工具追查（系统提示第 6 条：序列前后、共有的线、同比例偏移、缺失楔边上的
极大）。提示词只约束结果可信（测量值来自工具、派生量写来源、假设要标明），不约束探索路径。

## 不开窗口：命令行、Codex 和其他 agent

`src/gimap/app/headless_assistant.py` 在 offscreen 的 Analyze 页上组装同一套工具（只读用户已存的
仪器配置，新配置只在内存），`tools/gimap_agent.py` 是命令行入口（`auto`、`status`、
`find-calibration`、`tools`、`call`、`mcp`）；`mcp` 用 `infrastructure/mcp_stdio.py` 以 stdio
提供 MCP（与 Claude Code 大脑用的 HTTP 端点共用 `mcp_protocol.McpDispatcher`；工具就是
`open_frame` 加上 GUI 助手的全部工具，`run_standard_pipeline` 多一个 `out_dir`）。给 agent 的
操作手册：`docs/agents/giwaxs-playbook.md`（两层：基线 / 在基线之上；仓库根 `AGENTS.md` 指向它，
Codex 会读）。评估：`tools/eval_giwaxs_agent.py`（`baseline` 在真实数据上检查基线不该错的事，
`report` 给 agent 写的报告找错误说法、数第二层的发现），案例在 `docs/agents/eval/`。

## 调教在哪里

- `application/prompts.py`：`SYSTEM_PROMPT`（工作规则：第 2 条是可选的基线 `run_standard_pipeline`，第 3 条找标定，第 6 条是“固定流程看不到的东西”）、`AGENT_TOOLS_NOTE`
  （Claude Code 模式的工具前缀说明）、`REMINDER`、`task_message`（每次运行的任务消息）。
- `application/tool_specs.py`：模型看到的每个工具的说明和参数。
- `application/calibration_tools.py`：读文件的规则（`OPEN_LEVELS`、确认）、标样比较的判断
  （`compare_verdict`）、拟合好坏的阈值（`GOOD_RELATIVE_DQ`、`GOOD_RINGS`、`GOOD_RMS_PX`，在 `domain/calibration_quality.py`）。
- `infrastructure/local_files.py`：自动搜索的范围和上限；`domain/calibration_files.py`：标样名、
  文件分类、日志关键词、路径识别。
- 代码里的判断（弱模型也不会错的部分）：`domain/peaks.py` 的 `caveat`（spike、broad 晕、
  fit_failed、weak；序列提示只用可靠峰）、`domain/size.py`（晕不给尺寸）、`domain/orientation.py`
  （sin χ 加权覆盖 < 80% 不给 f、低于漫散射背景 20% 的阴影区算未测、扇区在阴影里不比较、两个等高极大或 f ≈ 0 不说取向）、`domain/notes.py`（从笔记取唯一的 αi /
  能量 / 像素）、`application/pipeline.py`（候选顺序、验收、帧选择）。回归测试在
  `tests/test_assistant_rules.py`、`tests/test_assistant_pipeline.py`。
- 不改代码的调教：Settings ▸ Assistant ▸ Standing instructions（每次运行都加进任务消息）。

## 分层

| 层 | 内容 |
|---|---|
| `domain/` | 纯 numpy / scipy 的 GIWAXS 指标：`find_peaks`（SNIP 背景、高斯 + 线性拟合、SNR 与标记）、`compare_sectors`（面内 / 面外净强度）、`ring_orientation`（I(χ) 覆盖、极大、Herman 因子 ± MC 误差）、`scherrer_size`（相干长度下限） |
| `application/` | `RunAssistantTask`（API：自有模型循环、提醒、重试、停止）、`RunAgentTask`（Claude Code：它负责循环，GIMaP 逐个回答工具调用）、`ToolCatalog`（工具校验、写操作确认、结果收集）、`tool_specs`、固定 system prompt、ports（`AssistantLlm`、`AgentRuntime`、`AnalysisWorkbench`、`Confirmer`、`ResultStore`、`RunEvents`） |
| `infrastructure/` | `ClaudeCodeAgent`（找 CLI：PATH、`~/.local/bin`、Claude 桌面版自带的 `%APPDATA%\Claude\claude-code\<版本>\claude.exe`、npm；`status()` 读版本与登录状态、`open_login()`）、`LocalMcpServer`（MCP Streamable HTTP，JSON 响应，只监听本机、校验 token 与 Origin）、`AnthropicAssistantLlm`（官方 SDK，流式、adaptive thinking + 摘要、prompt caching、`fallbacks="default"`）、`ApiKeyStore`、`JsonResultStore`（表格、运行记录、缺失能力清单） |
| `presentation/` | `AssistantController`（守护线程运行、Dock 面板、Brain 检查、保存报告）、`GuiBridge` / `GuiWorkbench`（跨线程操作 Analyze）、开始对话框、面板、设置页（`code_section.py`：Claude Code 程序、模型、登录）、报告 HTML |

`bootstrap.py` 组装以上部分；`src/gimap/app/main_window.py` 把控制器接到 Analyze 页面的按钮、
Tools 菜单和 Settings ▸ Assistant。Analyze 只暴露 `AnalyzePage.automation()`
（`presentation/automation.py`），不依赖本 feature。

## 约定

- 报告里的数值表只来自工具结果（`RunResults`），不是模型转述的文字。
- 写文件或改校正（`export_results`、`set_valid_intensity_range`）在“确认”模式下先弹窗询问；
  拒绝后模型收到 `declined` 结果，不重试。“全自动”模式不询问，但每一步都记录在面板里。
- 发送给 API 的只有状态和约化曲线（数字）；只有勾选“允许看图”时才发送一张小的 q 图。
- 每次运行的完整记录（步骤、结果、对话）写入用户数据目录的 `assistant_runs/`；
  模型标记的缺失能力追加到 `assistant_feature_requests.jsonl`，供以后补工具。
- API key 可存在用户数据目录（`claude_api_key.txt`），也可用环境变量 `ANTHROPIC_API_KEY`；
  需要安装 `anthropic` SDK（`python -m pip install anthropic`）。
- Claude Code 大脑需要 Claude Code 已安装（Claude 桌面版自带一份）并登录一次：
  Settings ▸ Assistant ▸ Sign In…（运行 `claude auth login --claudeai`）。面板显示
  Claude Code 报告的额度提示（`rate_limit_event`）和按 API 价格估算的费用（订阅内不计费）。
  这是给用户自己用的；别人使用需要他们自己的登录或 API key。
