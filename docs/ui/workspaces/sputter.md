# 溅射（Sputter）

- **Status**: Current（2026-11-14 页面重写：打开时是**主屏**（“分析” Analyze，调用随仓库的分析包；实测图与模型图并排、
  候选解、数值块、演化），第二个视图是**详情**（Details）；同日补充：多孔基底结果的数值块、“静态帧（沉积之后测量）”、
  主屏图样区的“图样 ｜ 实空间”切换；2026-11-15 分析运行时主屏显示**实时视图**；2026-11-16 输入栏加入**原位 ｜ 沉积后**
  切换；2026-11-16 早先的方法“预测”（Predict，随仓库的 PyTorch 网络、文献方法、干涉条纹求 H、在分析中查看）整个删除：
  详情只为“分析”服务——左栏几何、分析选项、静态帧、更多选项（掩膜、样品滚转角、结果位置），右边关于所示结果的
  **时间窗 / 报告 / 日志**三个标签页；速率加单位 Å/s ｜ nm/s（默认 Å/s）、不寻常速率的询问和到最后一帧的名义厚度；
  联合精修解 `joint` 作为一个解；实空间俯视图把有连续底层的解画成闭合的膜；2026-11-17 分析选项加**并行进程数**，实时视图和
  报告写出从磁盘读帧的耗时、慢盘提示与各步用时；同日一行**状态**说分析此刻在做什么（实时视图顶部、任务状态、初步结果下的
  提示行），初步结果的演化图上用虚线画联合精修当前的解；2026-11-18 实时视图左栏改为**步骤树**（较新的分析包；点带 › 的步骤看它的产出），
  联合精修在一次迭代内部也报告进度与剩余时间，**连续金属层 t_b** 作为数值块、演化曲线和时间窗一列，详情加**步骤**标签页和**存储**区，
  读入数据（缓存）改放本机文件夹并有大小上限）
- **科学说明**：各量的定义、规则和阈值见 [Physics and Methods Reference](../../Physics_Manual.md) 第 12 章；本文写调用链、
  文件与界面约定。
- **Scope**: 原位溅射序列逐时间窗的结构（分析包）：晶粒半径 R、高度 H、中心间距 D、间隙、尺寸分布宽度、覆盖率，
  每个时间窗若干候选解，实测图与模型图并排；沉积之后测量的样品逐个分析进一个系列。
- **Related code**: [`src/gimap/features/sputter/`](../../../src/gimap/features/sputter/)（说明见其 `README.md`）、
  `app/main_window.py`（`_create_sputter_page`、`running_jobs`、`_language_pages`、`shutdown`）、
  `app/project.py`（`missing`：也查主屏结果的文件夹）、
  `app/presentation/i18n/zh_sputter.py`（详情、几何、标签页）、`zh_sputter_client.py`（主屏、分析与速率）、`zh_sputter_live.py`、`zh_sputter_steps.py`（步骤树、存储、连续金属层）；
  主屏与分析：`domain/client.py`、`application/client.py`、`application/client_progress.py`、`application/use_cases.py`
  （`SputterService`、扫描的结果文件夹）、`infrastructure/client.py`、`infrastructure/process.py`、
  `presentation/views/sputter_main_view.py`、`presentation/client_flow.py`、`rate_flow.py`（速率与单位）、`client_panel.py`、
  `client_texts.py`、`pattern_view.py`、`evolution_view.py`、`real_space_view.py` 与 `domain/real_space.py`（实空间示意）；
  详情：`presentation/views/sputter_page_view.py`、`geometry.py`、`result_tabs.py`（时间窗、报告、日志）、`project_state.py`；
  并行进程数：`presentation/workers_flow.py`；读帧耗时的文字：`presentation/reading_texts.py`；
  实时视图：`application/client_live.py`、`application/live_progress.py`（初步结果之后的精修、联合精修、计数的速度）、
  `presentation/live_panel.py`、`live_status.py`（状态行、联合精修的虚线）、`live_pictures.py`、`live_static.py`（沉积后的样品）、
  `presentation/views/sputter_live_view.py`；步骤树：`application/live_steps.py`、`application/step_details.py`、
  `infrastructure/trace_files.py`、`presentation/live_steps_panel.py`、`step_tree.py`、`step_texts.py`、`step_pictures.py`、
  `presentation/views/sputter_steps_view.py`；存储：`application/storage.py`、`infrastructure/cache_store.py`、
  `presentation/storage_panel.py`；静态帧：`application/client_static.py`、`infrastructure/client_static.py`、
  `presentation/static_flow.py`、`static_list.py`、`static_queue.py`、`static_export.py`、`application/static_samples.py`、
  `application/static_report.py`、`presentation/views/sputter_static_view.py`、`views/sputter_samples_view.py`；分析包随仓库提供（`utils/sputter_client/`，
  研究代码的发布版：只有这些命令用到的文件，不含开发时所用私有数据的名字，`MANIFEST.json` 与
  `tests/test_sputter_client_bundle.py` 检查它；`GIMAP_SPUTTER_CLIENT` 或用户选的文件夹优先）
- **Related tests**: [`tests/test_sputter_feature.py`](../../../tests/test_sputter_feature.py)（几何的来源与预设、扫描的结果
  文件夹（与删除“预测”之前的路径相同）、进程树、文件头）、
  [`tests/test_sputter_page.py`](../../../tests/test_sputter_page.py)（详情的组成、表单与来源、每个文件夹记住的值、中文、
  关闭窗口；假的文件头读取：[`tests/sputter_fixtures.py`](../../../tests/sputter_fixtures.py)）、
  [`tests/test_sputter_details.py`](../../../tests/test_sputter_details.py)（时间窗 / 报告 / 日志标签页、联合精修解、速率单位、
  不寻常速率的询问、到最后一帧的名义厚度、旧版留下的偏好和项目）、
  [`tests/test_sputter_client.py`](../../../tests/test_sputter_client.py)（分析包的读取器、命令行与环境、进度行、服务；
  多孔基底与静态帧的读取、静态帧的命令行与服务；实空间示意的排布）、
  [`tests/test_sputter_main_screen.py`](../../../tests/test_sputter_main_screen.py)（主屏：数值块、换解、保存选择、空状态、
  分析与取消、语言、项目；多孔基底的数值块、输入栏的“沉积后”与静态帧系列、实空间示意的切换与闭合的膜、长路径不撑宽详情左栏；
  假的分析包进程与编造的结果：[`tests/sputter_client_fixtures.py`](../../../tests/sputter_client_fixtures.py)）、
  [`tests/test_sputter_live.py`](../../../tests/test_sputter_live.py)（实时视图）、
  [`tests/test_sputter_status.py`](../../../tests/test_sputter_status.py)（状态行的各种事件与剩余时间、任务状态、初步结果上
  联合精修的虚线及其消失、中文）、
  [`tests/test_sputter_reading.py`](../../../tests/test_sputter_reading.py)（并行进程数到请求和命令行、读帧耗时与慢盘提示、
  报告的各步用时、中文）、
  [`tests/test_sputter_steps.py`](../../../tests/test_sputter_steps.py)（步骤树、步骤的产出、联合精修迭代内的进度、结果的
  步骤标签页、连续金属层、实时视图只留所示的图）、
  [`tests/test_sputter_storage.py`](../../../tests/test_sputter_storage.py)（本机文件夹、用户数据文件夹里旧的读入数据、
  上限与最久未用者先删、存储区）、
  [`tests/test_sputter_client_bundle.py`](../../../tests/test_sputter_client_bundle.py)（随仓库的分析包）
- **Last verified**: 2026-11-18（步骤树、步骤的产出、联合精修迭代内的进度、连续金属层、存储：离线测试，编造的事件、细节文件与
  缓存；另在离屏窗口里浅色英文、深色中文截图核对）；2026-11-17（状态行与联合精修的虚线：离线测试，编造的事件与结果，另在离屏窗口里浅色、深色截图核对）；
  2026-11-17（并行进程数、读帧耗时、各步用时：离线测试，编造的事件与结果）；2026-11-16（删除“预测”、新的详情、速率单位、联合精修解、闭合的膜：离线测试，编造的结果）；
  2026-11-16（输入栏的“沉积后”与静态帧的实时视图：离线测试；另在本机用一份真实的沉积后样品在离屏窗口里经输入栏走完三步
  并截图，数值不进仓库）；2026-11-15（实时视图）；2026-11-14（主屏与分析）

## 页面的两个视图

`views/sputter_page_view.py` 从上到下：**输入栏**（两个视图都显示）和一个 `QStackedWidget`（`view_stack`）。

- **输入栏**（`views/sputter_main_view.py` 的 `_input_bar`）：最左是切换**原位 ｜ 沉积后**（`kind_switch`，
  `KIND_CHOICES`），其后的输入在 `input_stack` 的两页里（不显示的那页大小策略设为 `Ignored`，不撑宽输入栏）。“沉积后”一页见
  “静态帧”一节；“原位”一页：扫描文件夹（`scan_picker`）、金属 `sample_metal`（Ag / Au / Cu）、基底 `substrate`、速率
  `sample_rate` 与它的单位 `rate_unit`（`RATE_UNITS`：Å/s、nm/s，默认 Å/s）、**QCM 读数** `qcm_check`、主按钮**分析**
  `analyze_button`；下一行是一句几何 `geometry_summary`（能量、距离、入射角、光束中心、滚转角；缺什么就只说“还需填写的几何：…”，
  选了文件夹后用警示色）和**编辑**（切到详情、滚到几何区、光标放在第一个空字段）、**打开结果…**、**导出…**（`export_flow.py`：选文件夹，后台用分析包的 `iampred.export` 写 `<扫描名>_export`——演化图和实测/模型图的 PDF、SVG、300 dpi PNG，每窗显示的解（含用户手选）及全部解的 CSV，方法说明；详情的“打开报告 / 打开文件夹”旁也有）、视图开关 `view_switch`
  （主屏 / 详情）；再下一行是扫描的一句话 `scan_summary`（帧数、探测器、曝光、时间，有速率时加“到最后一帧约沉积 … nm”）；
  右边是分析的 `JobStatus`（`analyze_status`：取消、详情——详情按钮切到详情视图并显示分析的消息）。
- **主屏**（`_main_view`，默认）：没有结果时是空状态（`client_empty`：没有分析包时说明怎么给出它，按钮“选择文件夹…”；
  有分析包时说明怎么得到结果，按钮“打开结果…”）。有结果时（`client_view`）：时间窗选择（上一个、滑块、下一个、
  “t = … s · 已沉积 … nm”）与切换**图样 ｜ 实空间**（`image_switch`）；实测图与模型图（`PatternPair`），或所选解的实空间
  示意（`RealSpacePicture`，同一个位置，`pattern_host` 是两页的 `QStackedWidget`）；解的选择（`SegmentedControl`，页面按时间窗的
  条目动态建立：候选 1…3、精修解、联合精修解，各带拟合偏差）、**恢复默认解**、一句说明；数值块（半径 R、高度 H、间距 D、间隙、
  尺寸分布宽度、覆盖率；没有的量不显示；连续金属层 t_b（`base_nm`）只在结果的 `wanted` 列出它时显示，`ClientResult.shows`）
  和所显示解的拟合偏差；演化图（`EvolutionPlot`；t_b 用暗金色画在尺寸轴上）。
- **详情**（`_details_view`）：左栏（`details_scroll`）依次是**样品与几何**（X 射线、距离、入射角、光束中心，各带来源一行；
  把几何存为预设）、**分析选项**（`_analysis_section`：模式（自动 / 快速 / 精修 / 完整）、并行进程数（见下）、沉积开始、沉积持续时间、聚合物膜厚
  （只在基底是聚合物膜时显示）、分析包的文件夹与“选择文件夹…”、打开报告、打开文件夹、分析的消息 `client_log`）、
  **静态帧（沉积之后测量）**（`sputter_static_view.py`，默认收起）、**更多选项**（默认收起：掩膜与“分析包自带的掩膜”、样品
  滚转角及其来源、结果保存在）。右边 `tabs`（`result_tabs.py`，读所示结果的文件夹：`ClientService.texts` →
  `ClientFolderReader.texts`）：
  - **时间窗**（`windows_table`）：每个时间窗一行（静态系列每个样品一行）——时间或样品、名义厚度、结果有值的数值块各一列、
    拟合偏差、哪一个解；没有解的时间窗在“解”一列写原因；主屏换解后这一行跟着变；`enable_table_copy`。
  - **报告**：`analysis.json` 的要点（所用网络、试过的网络及拟合偏差、模式、像素掩膜、各步骤运行时间之和）在上；然后
    **各步用时**（`report_steps`，`result_tabs.steps_html`，富文本小表）：`steps` 里每一步及其秒数（m:ss，一小时以上
    h:mm:ss；同名步骤可出现两次，例如初步结果和最终结果各一次 `present (main)`；括号里的标签是分析包的原词），读帧那一行加上
    `reading` 的“已读入 a 帧（共 b 帧）· 磁盘每帧 x s（占用时的 y %）”（读过的扫描另写“此扫描此前已读入”；`reading.workers`
    与 `workers` 不同时写导入的进程数），最后一行合计，表下“进程数：`workers`”；`reading_was_slow` 时一条提示
    （`report_disk_note`，`ErrorBanner` 警告级，与实时视图相同的文字）；没有这些键的旧结果只列有的行。`REPORT.md` 用
    `QTextBrowser.setMarkdown` 显示在最下（搜索路径是结果文件夹，报告里的图能显示）。
  - **日志**：`analyze_*.log`，按 `analysis.json` 里各步骤的顺序，下拉框选一份（日志很长时只显示最后 400 000 个字符）。
  - 没有结果时每页只有一行“请分析一个扫描，或打开一个结果”（`NO_RESULT`）。
  标签页名用 “Logs”：单独的 “Log” 在翻译表里已是“对数”。
  - **步骤**（`tab_steps`，第四个标签页，`live_steps_panel.show_result_steps`）：左边一行说明和所示结果的步骤树
    （`ClientService.steps`：`trace/gimap_steps.json`，没有时由 `analysis.json` 和 `trace/` 里的细节文件重建），右边所点步骤的
    产出（点时读取，换结果时丢掉）；没有 `trace/` 的结果一行说它由较早的分析包生成。
- **存储**（`storage_panel.py`，`_storage_section`，左栏最后，默认收起）：展开时在后台（任务键 `sputter_storage`）列出每份读入
  数据（标签，运行中的那份加“（正在使用）”；大小；上次使用；本机文件夹或“用户数据文件夹（旧版 GIMaP）”）、合计与本机文件夹、
  结果根的大小（“从不自动删除”）。**删除所选**、**删除未使用超过 n 天**先询问（`QMessageBox.question`），从不删除正在使用的那份。
  **读入数据上限**（GB，偏好 `cache_limit_gb`，默认 5，0 为“不设上限”）随每次分析放进 `ClientRequest.cache_limit`。
- **并行进程数**（`workers_flow.py` 的 `WorkersMixin`，`workers_box`：`ShownComboBox`，显示或打开时发 `aboutToShow`）：
  第一项“自动（N）”，N = `ClientService.automatic_workers()`（即运行器的 `workers()`，`infrastructure/process.py`
  `default_workers`：min(16, 核心数 − 2)，再受空闲内存限制），每次显示或打开选择框时重算；其后 1 … `ClientService.cores()`。
  选择存在偏好 `sputter_page.workers`（数字或 `auto`；大于本机核心数的数字按核心数算）。分析或静态帧开始时
  `run_workers()` 把数字放进 `ClientRequest.workers` / `StaticRequest.workers`（自动时就在这一刻算出），运行器给
  `--workers <该数>`（为 None 时用运行器自己的数）；这个数不进请求的代码和标签（同一请求换进程数仍显示已有结果）。
  `ClientStarted.workers` 把它交给实时视图。

1440×900 和 1280×800 下主屏不需要展开任何东西（`test_the_main_screen_fits_the_window…`）。

## 主屏与分析（Analyze）

1. **分析**：`SputterClientMixin.analyze`（`client_flow.py`）→ 文件头没读完时排队（`_run_queued` / `_forget_queued`）→ `ClientRequest.problems()`（文件夹、速率 > 0、能量或波长、距离与入射角、光束中心
   两个都给或都不给、金属、基底、模式、持续时间与膜厚为正）和 `ClientService.package().problem` 先在界面线程检查 →
   样品处的速率（`ClientRequest.rate_at_sample`：QCM 读数 × 0.6）不在 0.005–1 nm/s 之内时问一次单位对不对
   （`rate_flow.RateMixin._confirm_rate`：**仍然分析** / 取消）→ 记住各值（几何 `_remember`；基底、速率及单位、QCM、开始、
   持续时间、膜厚按文件夹记在偏好 `client_folders`，基底、模式、QCM、单位另记为 `client_last`）→ TaskRunner 键
   `sputter_client` 提交 `ClientService.analyze(request, service.results_folder(request, 结果根), emit, should_stop)`。
   速率按所选单位换算成 nm/s（Å/s ÷ 10，取 12 位有效数字）放进请求；旧版记住的速率（没有单位）按 nm/s 读回。
2. **服务**（`application/client.py`）：输出文件夹 = 该扫描的结果文件夹（`<结果根>/<扫描>-<代码>`，`use_cases.scan_results_folder`：
   文件夹、金属和几何的哈希，并保留删除“预测”之前一并哈希的值 `FOLDER_VALUES`——聚合物 PSBMA、时间窗 0.5 s、旧模型的版本——
   使以前分析过的扫描仍能找回结果）下的
   `iampred-<代码>`，代码是导入依赖的全部值 + 模式 + 分析包版本的 SHA-1 前 8 位；这个文件夹已有完整结果
   （`analysis.json` 的 `status == "ok"` 且 `main/main.json` 有时间窗）时直接读它，不运行；否则 `ClientRunner.run` →
   停止抛 `Stopped`；退出码非 0 或 `status` 不是 `ok` 抛 `ClientFailed`（消息带 `status`，详情是失败那一步日志的最后几行，
   没有 `analysis.json` 时是进程输出的最后几行，例如分析包说的“no network for … on …”）；`status` 是 `ok` 但没有
   `main/main.json`（旧版分析包）抛 `ClientFailed(CLIENT_NO_MAIN)`。
3. **进程**（`infrastructure/client.py` 的 `ClientPackageRunner`）：做分析包 `run.cmd analyze` 做的事——
   `console_python() -u -m iampred.analyze --frames … --label gui_<扫描>_<代码> --metal … --substrate …
   (--rate-nm-s | --qcm-nm-s) … --alpha-i … --out … --mode … --workers N --data <数据根> -- --sdd-mm …
   (--energy-ev | --wavelength-nm) … (--beam-col … --beam-row … | --auto-centre) --roll … [--mask …] [--t-dep …]
   [--t-sput …] [--poly-t …]`，工作目录 `<包>/code/iam`，环境 `IAM_GIMAP`、`IAM_DIST`、`IAM_DATA`、`IAM_WORK`、`IAM_RAW`、
   `IAMPRED_MODELS`、`PYTHONPATH=<包>/code/iam`、`TZ=Europe/Berlin`、`OMP_NUM_THREADS=1`，加上 `child_environment` 的
   UTF-8、无缓冲、不用用户 site。`run_streaming` 每 0.2 s 检查停止并结束整棵进程树。
   - **标签**：分析包对已导入过的标签不再导入（不管这次给了什么参数），所以标签的代码包含导入依赖的全部值
     （`ClientRequest.intake_values`：文件夹、金属、基底、速率及是否 QCM、几何、掩膜、开始、持续时间、膜厚）；模式不在内。
   - **光束中心**：页面用 GIMaP 的像素坐标，交给分析包前减 0.5（`index_center_from_canonical`）；留空时给 `--auto-centre`（页面说“由帧估计”；否则分析包会用它自己束线的默认值）。
   - **滚转角**总是给出（留空为 0，与页面说的一致）；**掩膜**留空时不给 `--mask`，分析包用它自带的像素掩膜。
   - **数据与工作文件夹**：`<本机文件夹>/sputter_client/<包版本前 12 位>/data/<package-mask 或 mask-<掩膜 SHA-256 前 12 位>>`
     和 `…/work`（分析包要求一个数据文件夹只用一种像素掩膜）。包版本 = `MANIFEST.json` 的 SHA-256，没有时是文件指纹。
     本机文件夹（`infrastructure/cache_store.py` `local_data_dir`）：Windows `%LOCALAPPDATA%\GIMaP`，macOS
     `~/Library/Caches/GIMaP`，其他 `$XDG_CACHE_HOME/gimap`（`~/.cache/gimap`），设了 `GIMAP_HOME` 时 `<GIMAP_HOME>/cache`——
     不放在用户数据文件夹：Windows 上那是漫游配置文件，在很多研究所电脑上是有配额的慢速网络目录。以前放在
     `<用户数据>/sputter_client/` 的读入数据不搬动也不丢：请求的读入在那里已完成而本机没有时，`roots` 直接用那里的数据文件夹；
     静态系列的数据同样先找本机、再找那里（`storage_folder`）；存储区把它们列为“用户数据文件夹（旧版 GIMaP）”。
   - **上限**：每次运行前后，GIMaP 自己的读入数据（本机和用户数据文件夹里的，按标记 `<标签>.gimap.json` 的时间算最近使用，
     每次运行都更新它）合计超过 `ClientRequest.cache_limit`（默认 5 GB，0 不限）时，从最久未用的删起，直到不超过；不删本次请求的，
     也从不删结果。删了时消息里有一行 `CACHES_REMOVED`。
   - **中断的导入**：导入被取消后，标签文件夹留在数据文件夹里，分析包会永远拒绝这个标签（“exists: never overwritten”）。
     GIMaP 在导入那一步说“done”时写标记 `<数据根>/<标签>.gimap.json`；下次运行前，标签文件夹存在但没有 `scan.json` 或没有
     标记时把它删掉（只删 `gui_` 开头、GIMaP 自己建的那个文件夹）。
   - 运行前写 `<输出>/gimap_request.json`（请求、标签、分析包路径与版本、Python、命令行）。
4. **进度**（`application/client_progress.py`）：`[n/N] <step> ...`、`done in … s` / `FAILED (…)`、`status: …`；任务状态说
   “第 n 步，共 N 步：…”，进度条显示已完成的步骤所占的份额，加上正在运行的步骤内部由实时事件得知的份额（`within`）。`@live {json}` 行是实时视图的事件（见下节），
   在后台线程里读好它的小数组文件后作为 `LiveEvent` 交给界面，不进日志，也不进失败时显示的最后几行；其余的行进日志
   （详情 ▸ 分析选项）。
5. **结果**（`ClientFolderReader` → `domain/client.py`）：`analysis.json`（状态、模型族、模式、各文件路径）、
   `main/main.json`（`wanted`、`windows[]`：`t`、`t_rel`、`h_nom`、`arrays`、`entries[]`、`pick`、`smooth`；或 `{t, error}`）、
   `main/w_<t>.npz`（`q`、`af`、`measured`、`models`、`labels`）。没有的量就是没有（数值块隐藏、演化线断开）；`arrays` 只接受
   纯文件名；`{t, error}` 的时间窗在滑块上保留，图的位置说明原因。**打开结果…**接受输出文件夹、它的 `main` 文件夹，或
   扫描的结果文件夹（取其中最新的 `iampred-*`）；其他文件夹（例如早先方法“预测”的结果）不打开。
6. **显示哪个解**：默认是 `smooth`（全扫描平滑演化选出的；分析包做了全扫描联合精修时是标签和种类都为 `joint` 的
   联合精修解，名为“联合精修解”，与精修解一样没有网络权重），没有时是 `pick`（拟合偏差最小），再没有时是偏差最小的条目。
   用户换解后，选择写进 `main/choices.json`（只记与默认不同的时间窗；结果文件夹不可写时写到
   `<用户数据>/sputter_client/choices/`），再次打开时读回；演化图跟着选择变。
7. **图**：`PatternPair`（Matplotlib）把每幅图画在结果自己的格子上（出射角不等距；q 全为正时横轴按 log₁₀ 画，刻度写成
   10ⁿ）：图上的每个点取它所在格子的值（`raster_cells`，最近格、不插值），再用 `imshow` 画出。两幅图共用坐标轴和一个
   `LogNorm`，色标上下限是实测图正值的 1 % 和 99.7 % 分位（换解时不变）；没有数据的格子留空。格点不变时（同一个结果的
   各时间窗），换时间窗或换解只把新的值放进这两幅图和色标（`_update`），不重建坐标轴——每次约 0.1 s，跟得上滑块；语言或主题
   变了才整幅重画。`EvolutionPlot`：R、H、D、间隙（nm）和覆盖率对扫描中的时间，当前时间窗用竖线标出，点击选最近的时间窗。
   两者的边距按像素固定（坐标轴不随数值跳动，上下两排图左边对齐），随主题重画（`restyle_figure`），不改数组；中文字体只给
   已知的几个标题（`cjk_fallback(figure, texts)`：遍历整幅图会把每个刻度都建出来，一次要 0.1 s）。用 Matplotlib 而不是
   `DetectorView` / `CurvePlot`：格点不等距、两幅图要共用一个色标，并且本页的测试不逐个 `dispose()`（pyqtgraph 视图需要）。

## 实时视图（分析运行时的主屏）

分析一次要几分钟到几十分钟（读帧、精修最慢）。从分析包开始运行（`ClientStarted`，不是直接显示已有结果）到它结束，主屏的
`result_stack` 显示第三页 `live_view`（`views/sputter_live_view.py`；行为在 `live_panel.py` 的 `LivePanelMixin`）：

- **左栏**：`StepRail` 四步——读入帧、求解结构、写报告、绘制图像——各带状态（按 `ClientProgress` 的步骤行；有两个网络时，
  第二个网络跑完、`chosen` 到达之前“求解结构”仍在进行）和一行产出：已读入 n / N 帧 → “N 帧，起–止 s，沉积自 … s 开始”
  （或“此扫描此前已读入”）；“网络 i/n：已求解 k / m 个时间窗” → “<网络> 与数据更吻合（拟合偏差 …）”；“已绘制 k / m 个时间窗”。
  下面是已运行时间和进程数（“进程数：n”，取 `reading.workers`，没有时取 GIMaP 交给分析包的数）。点一个已开始的步骤显示它的产出；点正在进行的步骤回到跟随分析；点尚未开始的步骤不变。
- **右边**：“第 n 步（共 4 步）：…”、这一步做什么（一段话），一幅图、图下一行、一幅曲线：
  - 读入帧：`FramesPicture`——左边是自动掩膜所用抽样帧的求和（分析包按 4×4 像素合并后的 log₁₀ 计数），掩膜去掉的部分
    红色半透明、直射束 “+”、样品地平线虚线；右边是最近读入的一块帧（128 帧）在 q_y–出射角网格上的计数率（两侧各自的格子，
    中间的空隙留空，对数色标取 1 % 和 99.7 % 分位）；图下是束心（GIMaP 像素坐标）与掩膜去掉的像素数。`TracePlot`：已读入各帧的
    每像素计数率对扫描时间，导入找到沉积开始时用虚线标出。
  - 求解结构、写报告、绘制图像：最近求解的时间窗（正在工作的网络的；数据选定网络后是选定的那个的）的实测图与模型图
    （复用 `PatternPair`，同一套格子与色标规则），图下一行
    “t = … s · 候选 k（或精修解）· 半径 R … · 高度 H … · 间距 D … · 覆盖率 … · 拟合偏差 …”；演化图复用 `EvolutionPlot`，
    画到目前为止各时间窗拟合偏差最小的解（正在工作的网络；选定网络后画选定的那个），最近的时间窗用竖线标出。
- **从磁盘读帧**（2026-11-17）：较新的分析包在每个 `frames` 事件里加 `frames`（这一块的帧数）、`read_s`（这块的进程
  打开、读取帧文件的秒数）、`wall_s`（这块的总秒数：读 + 解码 + 投影）、`mb`；导入结束时发 `reading`（`frames_read`、
  `frames_total`、`workers`、`seconds`、`read_s_per_frame`、`read_share`、`mb_read`、`slow`）。`LiveRun` 累加各块
  （`read_frames`、`read_s`、`wall_s`、`read_mb`），`disk_per_frame` / `disk_share` 优先取 `reading` 的值。读帧这一步的行变成
  “正在读入第 a 帧（共 b 帧）· 磁盘每帧 x s（占用时的 y %）”，读完后在“N 帧，…”后面保留磁盘这一段。每帧 ≥ 0.15 s 且占比
  ≥ 50 %（`SLOW_DISK_S_PER_FRAME`、`SLOW_DISK_SHARE`，与分析包的阈值相同），或 `reading.slow` 为真时，说明下方出现警告级
  `ErrorBanner`（`live_disk_note`）：慢在从这个磁盘读帧（USB／网络驱动器或杀毒软件逐个扫描文件），先把扫描文件夹复制到本机
  SSD。之后各步都保留。没有这些字段的旧版分析包：行与以前相同，不出提示。
- **效率**：事件在后台线程里读好数组（每个几十到几百 kB，`LIVE_ARRAYS_LIMIT` 以上不读），界面线程只更新 `LiveRun`；
  图由单发定时器最多每 `REDRAW_MS`（400 ms）画一次，只画变了的部分；帧图和实测／模型图在格子不变时只换数值。
  `LiveRun` 只留每个网络最近求解的一个时间窗和最近一块帧的数组，其余只留数值。
- **结束**：成功时显示结果；失败或停止时回到运行前显示的（原来的结果或空状态）。不发事件的旧版分析包只有左栏的步骤，
  右边说明它不显示中间结果。
- **网络之间不空白**：第二个网络刚开始、还没有解时，图和演化仍显示上一个有解的网络，直到新网络的第一个时间窗到达。
- **求解结构的各个阶段**（2026-11-16）：一个网络依次是准备网络输入（每个时间窗一次）、网络一次给出全部时间窗的候选、
  模型画出每个候选、（精修模式）逐窗精修、（完整模式）整条扫描联合精修。分析包每个工作进程对自己那一份每秒最多发一个
  `progress` 事件（`done`、`total`，画图与精修另带已计算的模型图数），`LiveRun` 按网络和阶段把各进程的最新计数相加
  （`phase`、`phase_count`）。步骤行按阶段说：“已准备 k / m 个时间窗的网络输入”“网络正在为 m 个时间窗给出候选结构…”
  “已求解 k / m 个时间窗（已计算 n 幅模型图）”“已求解 k / m 个时间窗，已精修 j 个（已计算 n 幅模型图）”“整条扫描联合
  精修：第 i 轮（最多 N 轮）”；两个网络时前面加“网络 i/n：”。第一个时间窗求出之前，图的位置也显示这一行。精修模式下
  分析包先给每个时间窗求出候选（演化图很快画满），再逐个精修（每个精修只从它自己时间窗的最佳候选出发，结果与顺序无关）。
- **状态行**（2026-11-17，`live_status.py` 的 `status_line` 与 `LiveStatusMixin`）：一行白话说分析此刻在做什么，取自最近的
  事件（`LiveRun.activity`：最近一个被接受的事件的种类；步骤行到达时 `note_step()` 设为 `step`）：读帧“正在读入帧：a / b”、
  束心与掩膜、“已读入 N 帧，网络即将开始”、“正在启动网络 …”、各网络阶段（“网络 …：正在准备网络输入 / 网络正在给出候选 /
  正在用正向模型绘制候选 / 正在精修各时间窗的最佳候选，第 k/m 个时间窗”）、“比较网络：a 21.5, b 12.5 → b”、初步结果与逐窗
  精修开始、联合精修（`joint_stage` 的 pool：“准备中（每个时间窗试用相邻时间窗的解，共 n 个时间窗）”；fit 之后第一次迭代
  完成前“正在进行第 1/n 次迭代 · 已用 m:ss”，之后“第 k/n 次迭代 · 拟合偏差中位数 … · 已用 m:ss，约剩 … 分钟”）、
  “正在绘制主屏：第 k/m 个时间窗”。剩余时间：计数类按第一个和最近一个事件之间的速度（`LiveRun.paces`、`left`，两事件相隔
  至少 `PACE_MIN_S` = 1 s 才估计，减去最近事件之后已过的时间；不到 1 分钟说“剩不到 1 分钟”）；联合精修按
  `seconds / iteration × (total − iteration)`（`joint_left`：迭代全部做完所需时间，提前收敛会更早结束），已用时间是
  `seconds` 加上事件到达后的时间（`joint_elapsed`）。事件没有比步骤行更多的信息时（`activity == "step"`、旧版分析包）显示
  步骤行（“第 n 步，共 N 步：…”）。这一行在实时视图顶部（`live_status`，`ElidedLabel`，`gimapRole` = `info`），在任务状态里
  （`STATUS_FIELD`；取消后的“正在停止…”不被覆盖；同样的内容不重设），初步结果显示在主屏时也在时间窗选择下面的提示行
  （`run_note`，只在这时显示）。重画定时器（≤ 400 ms）和每秒的时钟更新它。初步结果之后、写报告之前（`AFTER_PRELIMINARY`），
  任务状态的进度条跟随这一行所说的份额（阶段的 k/m、联合精修的 k/n；没有份额时为忙碌条），不再停在初步结果画完时的 100 %。
  同一网络再次收到 `windows`（初步结果之后的逐窗精修）时，它的阶段计数、速度和联合精修从头开始（`_restart`），解保留。
- **联合精修的虚线**：每个 `joint` 事件（每次迭代后各时间窗当前的 R、H、D、间隙、覆盖率…）由 `LiveRun.joint_points` 按时间
  排好，`EvolutionPlot.set_overlay((时间, 各量, 名称))` 以各量自己的颜色画成虚线，图例最后一项用灰色虚线写“联合精修 第 k/n 次”，
  下一次迭代替换它。画在主屏的演化图上——只在主屏显示的就是本次运行的初步结果时（`_live_preliminary`，`_preliminary_shown`）；
  否则（没有初步结果的模式、或初步结果读不出）画在实时视图的演化图上（网络相同时）。数值块不显示联合精修的值：卡片只说
  所选的那个解，两个结果不混在同一排数字里。运行结束（`_live_end` → `_live_status_end`）或主屏换成别的结果
  （`show_client_result`）时虚线去掉；最终结果本身含联合精修解。多孔基底和静态系列不画。
- **进度条**：`ClientProgress.within` 是正在运行的步骤内部已完成的份额（`LiveRun.step_share`：读帧按帧数，求解按各网络
  阶段的大致权重 `PHASE_WEIGHTS`，绘图按窗数），进度条在步骤内部也前进；求解那一步在第一个网络结束时就打印 done，第二个
  网络运行时进度条仍按 `within` 前进，直到数据选定网络。
- **沉积后的样品**（`live_static.py` 的 `LiveStaticMixin`，`LiveRun(kind="static")`）：三步——读入样品的帧、求解结构、
  归入系列（`STATIC_LIVE_STEPS`；“绘制图像”隐藏）。读入帧：左边探测器求和（给了裸基底时先是裸基底的，再是样品的），右边
  样品可用帧之和；图下“N 帧中 n 帧可用 · 裸基底：另一片未镀金属的基底（或：没有裸基底，认为膜长在 Si 上）· 未用：<文件>
  （分析包说的原因）”；下面是目前的系列。求解结构：最近求解的一帧的实测图与模型图，图下“第 k 帧 · 网络（或精修）· 半径 R … ·
  拟合偏差 …”；演化图是此前已分析的样品加上这个样品已求解的各帧（都画在名义厚度处，经中位数连线——结果里样品的数值就是各帧的
  中位数），最近一帧标出。步骤行“N 帧中 n 帧可用”“已求解 k / m 帧”（快速、精修模式只取可用帧中的 4 帧，这时加上
  “取 n 个可用帧中的 m 帧”）。
- **步骤树**（`application/live_steps.py` 的 `StepTree`，`presentation/step_tree.py` 的 `StepChecklist`，`live_steps_panel.py`
  的 `LiveStepsMixin`，布局 `views/sputter_steps_view.py`）：较新的分析包发 `step` 事件后（`StepTree.reported`），左栏的四个
  步骤行换成步骤树：○ 待做、● 进行中（下面一行是状态行，见上；联合精修时是“联合精修：第 k/n 次迭代 · 求导数：12/20 个时间窗 ·
  已用 6:12，约剩 9 分钟”；第一次迭代前是“联合精修：计算起点 · 4/20 个时间窗”，准备和起点阶段不报剩余时间）、✓ 完成（右边是这一步的一行结论）、✗ 失败、– 跳过。父步骤在它或子步骤进行中 / 失败时展开，其余时候
  收起（▸ / ▾ 可手动切换）。标题用 `STEP_TITLES` 的译文（网络的步骤用网络名）；分析包写的英文结论与 `step_texts.SUMMARY_TEMPLATES`
  匹配，用同样的数值显示译文，其余照原样。带细节文件的步骤标题后有“›”，点它在右半边打开，标题下是这一步完整的一行结论（树里会截断；点本身没有细节的父步骤——读入帧、整条扫描联合精修——会展开它并显示第一个有细节的子步骤）（`live_right` 第 2 页：
  `step_pictures.StepDetailPicture`，由 `ClientService.step_detail` 依次在结果的 `trace/` 和读入的 `<数据根>/<标签>.trace/`
  里找）：束心与掩膜——合并后的探测器图、掩膜去掉的份额（红）、束心 +、样品地平线（虚线），束心按 GIMaP 像素坐标
  (col + 0.5) × 合并倍数给出；读帧——数字和读入的帧段；沉积开始——计数率随时间、基线及其取值区间、开始时刻（规则自己的结果
  不同时用点线）；时间窗——每个时间窗一个刻度；比较网络——各网络拟合偏差中位数的条形（选用的用强调色）；联合精修——每次迭代的
  拟合偏差中位数，以及最后一次迭代后的解（R、H、D、t_b，覆盖率用右轴）。“回到分析”关闭它并丢掉数组；打开着的联合精修每次
  `joint` 事件后重读。初步结果显示时，提示行旁的**步骤**（`steps_button`）回到实时视图，实时视图顶部的**显示初步结果**
  （`live_preliminary_button`）回去。运行以结果结束时，树写进 `trace/gimap_steps.json`（`ClientService.save_steps`）。
  较早的分析包没有 `step` 事件：保留四个步骤行。
- **联合精修迭代内的进度**（`application/live_progress.py`）：`joint_progress`（`family`、`iteration`：正在计算的那次、`total`、
  `phase`：`derivatives` 求导数（一次迭代的大部分时间）/ `step` 试探步 / `pool` 准备、`done`、`windows`、`seconds`：从联合精修开始
  起的秒数）。剩余时间 `joint_batch_left` = 每个时间窗的秒数（这一批从第一个到最近一个事件；一次迭代求导数的第一个事件则从上一次
  迭代结束算起）× 这一批剩下的时间窗 + 每次迭代的秒数（已完成迭代的平均，没有时用这一批的速度 × 时间窗数）× 之后的迭代次数，
  再减去最近事件之后已过的时间。任务状态的进度条：之前的迭代加上本次迭代的份额（求导数按 0.9 次迭代算，`joint_batch_share`）。
- **内存**：实时视图只留它显示的图（最近一批帧、每个网络最近求解的时间窗、一个打开的步骤细节）；细节文件点开时读、关闭或换结果时
  丢掉。

**分析包的约定**（研究代码 `iampred.live`，2026-11-15）：每一步的进程从环境变量 `IAMPRED_LIVE` 得到一个文件夹，在其中写
`<时间>_<pid>.json`（和需要时同名 `.npz`，先写临时名再改名）；`iampred.analyze` 在步骤运行时每 0.5 s 把新事件打印成一行
`@live {json}`（加 `step` 和 `arrays`：npz 的绝对路径），结束时删掉 `<输出>/live/`。事件：`centre`、`detector`、`frames`、
`reading`、`imported`（读入帧）；`models`、`windows`、`window`、`chosen`（求解；`models`、`chosen` 由 `analyze` 自己发）；`drawing`、
`drawn`（绘图）。字段见 `application/client_live.py` 的说明。事件不改变任何结果（同一扫描有无事件，`windows.csv` 逐字节相同）。
`progress`（求解：`family`、`phase`、`done`、`total`、`worker`，`images` / `evaluations`）见上。精修模式下 `analyze` 还发
`preliminary`（`family`、`folder`：初步结果已写好）和 `refine`（`family`、`refine`、`iterations`）；较新的分析包在联合精修中发
`joint_stage`（`family`、`stage`：`pool` 或 `fit`、`windows`、`total` = 迭代次数）和每次迭代后的 `joint`（`family`、
`iteration`、`total`、`seconds`：从开始迭代起的秒数、`median`：拟合偏差中位数、`windows`：各时间窗 `{t, fit_deviation, R_nm,
H_nm, D_nm, gap_nm, width, coverage, amount_nm, base_nm}`）。缺的字段照样显示能显示的部分。
2026-11-17 起的分析包还发 `step`（`id`、`state`、`title`、`summary`、`detail`，见上）和 `joint_progress`，把每一步的细节写进
`<输出>/trace/`（JSON，图像为压缩的 float16 `.npz`：`detector`、`reading`、`onset`、`windows`、`compare`、`joint`），读入的细节
还留在 `<数据根>/<标签>.trace/` 并复制进每个结果；`main.json` 的 `wanted` 列出 `base_nm`（晶粒下连续金属层的厚度，0 为分开的岛）。
`iampred.static` 同样转发：`detector`（`role`：`sample` 或 `bare substrate`）、`sample`（可用帧之和、帧数、未用的帧及原因）、
`solving`（要求解的帧数）、`frame`（一帧的解，`stage`：`network` 或 `refined`，带实测图与模型图）。

主屏的约定：

- 主屏上没有等级、状态标签，也没有“可靠 / 不可靠”一类的话；数据分不开的解就是并列的候选解。分析包自己的说明
  （`note`：开发参考、未经独立表征验证）写在手册里，不在主屏上。
- 沉积速率是必填的；勾选 QCM 读数时用 `--qcm-nm-s`（分析包换算到样品处），否则 `--rate-nm-s`。
- 分析包支持哪些金属／基底组合由分析包自己说（不在 GIMaP 里再写一份表）；不支持时它的那句话进任务状态的详情。
- 聚合物膜厚：分析包只在基底不在它自己的聚合物表里时才要求 `--poly-t`，而命令行接受的五种聚合物（PS、PET、PSBMA、
  PNO、PEO）都在表里，所以膜厚是可选的，只在基底是这五种之一时显示和传递。
- 项目文件：`project_state()["client"]`（基底、QCM、模式、速率／开始／持续时间／膜厚字段、结果文件夹、是否在详情视图）；
  没有这一项的旧项目清空主屏。偏好 `sputter_page` 另有 `client_package`（用户选的分析包文件夹）、`last_client_result`
  （下次启动时打开）。

## 多孔基底（Porous substrate）

分析包对多孔 TiO₂ 上的金属给出另一组量（`main.json` 的 `wanted`），没有晶粒点阵，所以没有高度、间距、间隙和覆盖率：岛半径
（相对值）、尺寸分布宽度、岛份额 `f_islands`（可以大于 1）、帽层份额 `cap_share`、孔内金属网络的竖直与横向关联长度
`depth_nm`、`xi_nm`、成膜份额 `film_share`。`ClientResult.porous`（任一条目有这几个键）决定显示方式：数值块是这七个
（半径一块写“岛半径（相对值）”，提示说它只看相对变化）；演化图左边是三个长度（岛半径、竖直与横向关联长度，nm），右边是三个
份额（纵轴上限至少到最大值，岛份额大于 1 时不出界）。

## 静态帧（Static Frames，沉积之后测量）

沉积之后测量的样品没有沉积速率，也没有沉积之前的帧，所以输入栏的“沉积后”一页（`kind_switch` 选 `static`）不问单个样品，而是
**添加样品…**（`static_add_button`，菜单：文件夹… / CBF 文件…；不再打开 zip 包）、一行概况（`static_summary`：几个样品、几个系列、
已分析几个、几个暂不分析）和**样品列表**开关（`static_list_button`）。列表（`views/sputter_samples_view.py`，行为在
`static_list.py`）在主屏和详情的左边，与它们同在一个分割器里，只在“沉积后”显示：

- **找样品**（`application/static_samples.py`，`find_samples`）：选的文件夹里每个下面有 CBF 帧的子文件夹算一个样品（帧多深都行，
  文件名含 GIWAXS 的不算）；里面只有一个说明不了什么的文件夹时（压缩包解压进同名文件夹）往里找，至多三层；子文件夹的名字都没有
  厚度时，选的文件夹本身是一个样品（同级里与前缀同名的文件夹作裸基底）；单个 CBF 文件也是一个样品。
- **从名字读**（`name_parts`）：后面跟 `nm` 或金属符号的数字是名义厚度（`10Au`、`6.5nm`、`6p5_Au`），紧挨着的 `H` / `HiPIMS`
  为 HiPIMS、`DC` 为 DC；前面的部分是前缀（基底）。与前缀同名、没有厚度的文件夹是这些样品的裸基底（不列为样品）；同一前缀里
  有 HiPIMS 样品时，其余的记为 DC（`complete`）。前缀加沉积方式是**系列**（`850 · HiPIMS`），每个系列分析进
  `<结果文件夹>/<系列名>`（`series_slug`）。名字说不出厚度、或写着别的金属时，样品列出但不勾选，状态栏写原因（缺厚度 / 不是 Au /
  没有帧），提示里是全文；不猜。
- **列表**（`QTreeWidget`）：系列行（跨列：系列名、裸基底、已分析几个；勾选框勾选整组）下面是样品行——勾选框、名字、名义厚度
  （双击可改；补上厚度的样品自动勾上）、状态（等待中 / 分析中 / 偏差 x.xx / 失败，失败原因在提示里）。选一个样品，主屏显示它的
  系列并停在这个样品（`show_client_result(…, sample=…)`）；主屏上沿系列移动，列表跟着选中（`_choose_row_of_window`）。右键：
  分析这个样品 / 这个系列、显示结果、整组的裸基底…/不用裸基底、打开文件夹、移除；“编辑”菜单：全选、全不选、移除没勾选的、
  全部移除。列表记在偏好 `static_samples` 和项目的 `static.samples` 里。

**分析**（`static_queue.py`）：按列表显示的顺序依次分析勾选且有厚度的样品。先查所有样品共用的值（结果文件夹、模式、几何），
不对就不开始；单个样品的值不对（文件夹没了）就标为失败、跳过。要替换以前用别的帧或数值分析过的样品时，问一次，列出全部名字。
每个样品是 `static_flow.py` 里的一次运行：任务状态写“第 i/n 个样品：名字 — 步骤”，进度条覆盖整个列表，行状态跟着变，主屏显示
这个样品的实时视图。某个失败就标出来、接着下一个（只有一个样品时原因写在任务状态里，同以前）；取消停止当前和后面的样品。
结束后任务状态汇总，主屏显示最后分析的样品所在的系列，提示条给出结果文件夹。

**导出…**（`static_export.py`，显示的是静态系列时；按钮提示随之改变）：问一个文件夹，在后台写 `samples_report/report.html`
（`application/static_report.py`：每个系列分“孤立颗粒”和“连成的膜”两部分，判据与主屏的实空间图和数值块标题相同（`domain/real_space.py`：
有连续底层，或晶粒连成贯穿全图的网络，就是连成的膜，`joined`）。孤立颗粒的表列 R、高度 H、间距 D、间隙 D − 2R、覆盖率；连成的膜的表列
膜厚（底层）、膜密度、膜高、起伏高度、起伏半径、起伏间距——膜里没有单个颗粒；两张表都有名义厚度、阶段、h_m、拟合偏差（按 ≤4 / ≤10 /
更大着色）。另有一张两部分随名义厚度的图（`static_figures.py`，matplotlib Agg，同一横轴）和分析包的 `patterns.png`，以 base64 嵌入；
文字用界面语言）和 `samples.csv`（每行带所属部分）。阶段（`stage`）：分立的岛 / 相互接触的晶粒 / 连成网络的晶粒（实空间图的排布）；
有连续底层时，底层比晶粒高出底层的部分薄为薄底层上的晶粒，否则为连续膜，底层密度低于 0.95 时为密度低于 1 的膜。

任务状态是输入栏的同一个（`static_status` 就是 `analyze_status`，`_status_owner` 记住最近是哪一种分析在说话）。几何一行这时说这些帧
的几何。所选种类记在偏好 `analysis_kind` 和项目里。详情里一节（`sputter_static_view.py`，默认收起）只留**结果文件夹**（各系列的
子文件夹放在里面）、模式（快速 / 精修 / 完整）、这些帧的几何（能量、距离、光束中心、滚转角、掩膜；留空用分析包自己的值，“分析包
的值”把它们填进来；“分析包自带的掩膜”按文件名列出包里的 `.npy`）和日志。运行时主屏显示实时视图的三步（见“实时视图”）。

1. `StaticService.analyze`（`application/client_static.py`）→ `StaticPackageRunner`（`infrastructure/client_static.py`）：
   `python -u -m iampred.static --frames … --label … --nominal-nm … --out <系列> [--reference-frames …] [--sample …]
   [--substrate …] --mode … --workers N [--sdd-mm …] [--energy-ev …] [--beam-row … --beam-col …] [--roll …] [--mask …]`，
   环境同“分析”，另加 `IAM_STATIC_DATA`。分析包的默认几何从它 `static` 模块的源码里读（不导入）；没有这个模块的包不能
   分析静态帧，页面这样说。
2. **系列**是一个输出文件夹（`<结果文件夹>/<系列名>`）；GIMaP 在里面写 `gimap_series.json`（编号、像素掩膜、每个样品由什么分析
   得到）。导入的帧在 `<用户数据>/sputter_client/static/<系列编号>`，与原位的数据文件夹分开（像素掩膜不同）。同一个样品名换了帧或
   数值再分析时，先问（一次问全部），再把旧的导入和系列里的那个样品删掉重做；完全相同的请求直接显示系列，不再运行分析包。
3. 结束后主屏显示整个系列：样品代替时间窗（选择器写“样品名 · 名义厚度 … nm”），每个样品一个解（`模型解`，各帧的中位数），
   演化图横轴是名义厚度，不出现“时间”字样；数值的范围（各帧与网络区间）在数值块的提示里。**打开结果…**也能打开系列。
4. 一次只运行一个分析：一种在跑时“分析”不可用（切到另一种也一样）；所选的包不认识静态帧时，“沉积后”下的“分析”不可用。
   结果文件夹、模式、几何、掩膜记为上次用的值，整节、样品列表和所选种类进项目文件（旧项目为“原位”，列表为空）。

## 实空间示意（Real Space）

主屏时间窗选择右边的切换 **图样 ｜ 实空间**：实空间把所选的解画成形貌（`presentation/real_space_view.py`；排布是纯 Python 的
`domain/real_space.py`）。换时间窗、换解，示意跟着变；只在这一页显示时才画。

- 用到的值：半径 R、间距 D、尺寸分布宽度、高度 H，以及分析包另给的 `base_nm`（晶粒下面连续底层的厚度，0 为分立的岛；
  不在 `wanted` 里、没有数值块，读取器为示意单独保留它：`SKETCH_KEYS`）。没有高度时晶粒画成半球。
- 左边截面：穿过一排晶粒，长 4 D；衬底（聚合物膜按 `FILM_SUBSTRATES` 画成聚合物，厚度不按比例），连续底层，晶粒画成半径为
  r、高为 (H − base)·r/R 的半椭圆；横竖同一比例（nm）。右边俯视：边长 7 D 的方块，晶粒在间距 D 的六角排布上各自随机偏移约
  0.12 D（没有长程有序，与模型的干涉函数一致），半径取自以 R 为中位数的对数正态分布。
- 随机数只有一个种子、每次按同样的顺序取，所以换解或换时间窗只改变大小，不改变“骰子”。示意下面一行说明位置是随机排布的，
  大小取自所选的解。
- **整条扫描是同一片膜的生长**（`domain.real_space.grown`，2026-11-20 用户：“溅射过程肯定是连续的…第二个图是基于第一个图
  的加性的”）：主屏把各时间窗所选解的值按时间顺序（沉积后的系列样品：同一基底、按名义厚度）交给示意
  （`client_panel._space_series`）。视场边长整条扫描共用（`series_field`：7 × 最大 D，最小 D 时超过 `MAX_GROWN` 个晶粒就缩小）。
  逐窗：所画覆盖率低于 `ROUND_UP`（0.45）时，解的数密度 2/(√3 D²) 要更多晶粒就在空位加新核（每个核试 `NUCLEUS_TRIES` 个
  随机位置、取离已有晶粒最远的），要更少就让最接近相接的岛成对合成一个圆岛（面积相加、面积加权的中心）；之后不再增减晶粒、
  也不移动，只长大。半径 = s·exp(宽度·z)，公共尺度 s 使晶粒的并集覆盖视场的目标份额（`_cover_scale`：格点上
  min_i |g − x_i|·exp(−宽度·z_i) 的分位数）；目标是解的覆盖率（没有时按 R、D 估），有连续底层时至少 `CLOSED_COVER`（0.85），
  且不低于前一窗。于是早期是分立的岛，覆盖率约一半后长成蠕虫状团簇，再成连通的膜；这时 `RealSpace.channels` 为真，
  底层只铺在金属的覆盖范围内，其余是通到基底的沟道（`height_map`、`profile` 在那里为 0，图上方写沟道占视场的份额）。
  2026-11-20 用户：“最终阶段应该是连续的沟道，而不是 nm 颗粒”。截面是俯视图中线的切线（`cut`）。整条序列计算一次
  （`lru_cache`），换窗不重算。“连续膜／网络”的判断（数值方块的标题）也用这片生长的膜。
- **聚并与逾渗**：相接的晶粒（中心距不大于两半径之和）连成一个团簇（`clusters`，并查集），画成一整块——先画所有轮廓，
  再把填充盖在上面，团簇内部不留边线；截面里相接的晶粒帽同样连成一片。连通视场两条对边的团簇（`spanning`）画得深一些。
  图上方一行说这一时刻的排布（`arrangement`）：分立的岛 / 聚并中：相邻晶粒相接 / 连成贯穿视场的网络（已逾渗）。
  有连续底层（`base_nm` > 0，`RealSpace.continuous`）的解是闭合的膜：俯视图整个方块铺成底层的金属色（与截面里的底层同色），
  晶粒画成其上的起伏（颜色稍深 `RELIEF`、细轮廓线），图上方一行改说“连续膜 … nm，其上的晶粒：…”。这些都由所选解的 R、D、分布宽度和底层厚度在几何上推出。
- **多孔 TiO₂**：分析包为 TiO₂ 模型族的每个解另给出金属的位置（`porous_geometry`：颗粒半径 75 nm、孔内金属的深度与宽度、
  帽层覆盖率与厚度、岛／帽层／孔内的金属量、团簇高度与间距；读取器经 `SKETCH_KEYS` 保留，不进数值块）。左边是层的
  截面，按比例：TiO₂ 颗粒（`layer_particles`：略留空隙的六角排布，深度到孔内金属以下），孔内金属按深度的高斯
  （`pore_density`）画成颗粒后面的色带，并标出中心深度；最上一排按帽层覆盖率给一部分颗粒戴上帽层。右边放大一个颗粒的
  顶部：帽层按厚度、团簇按半径、高度和间距。下面一行是已沉积量和折合成致密金属的岛、帽层、孔内厚度。没有这些位置的解
  （旧的分析包）说明没有可画的形貌。

## 记住的内容（偏好 `sputter_page`）

`folders`（每个扫描文件夹上次分析的几何、金属、掩膜；最多 300 个）、`last`（上次的值；掩膜和样品滚转角不从这里带到别的
文件夹）、`presets`（用户的几何预设：名称、探测器序列号、平场文件名、起止日期和几何）、`last_scan`、`results_root`、
`advanced_open`。主屏的分析另有：`client_folders`（每个扫描文件夹上次分析的基底、速率及其单位 `unit`、是否 QCM 读数、模式、
沉积开始、持续时间、膜厚；最多 300 个；分析过的文件夹再选中时取回它自己的值，没分析过的文件夹不沿用上一个扫描的速率、
开始、持续时间和膜厚，但用户刚输入、还没分析的值保留——`_client_owner`）、`client_last`（上次的基底、模式、QCM、单位）、
`client_package`（用户选的分析包文件夹）、`last_client_result`（下次启动时打开）、`analysis_kind`、`static_last`。
旧版为“预测”记下的值（`folders` 里的聚合物、时间窗、速率、t0、`t_end`，顶层的 `last_result`、`python`、`toolkit`、`device`）
读取时忽略；顶层那几个在下次写偏好时去掉（`geometry.DROPPED_KEYS`）。这些值来自用户的实验，只存在用户数据文件夹，不进仓库。
