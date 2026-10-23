# 溅射（Sputter）

- **Status**: Current（2026-10-22 重新设计：模型随仓库提供，选文件夹后按“预测”即可；2026-10-23 加入文献方法和“文献对照”页；
  同日改为物理路径始终是默认解释、文献方法作为对网络的核对，并加入可选的沉积结束时间）
- **Scope**: 原位溅射序列（聚合物上的 Ag、Au；Cu 用 Ag 模型近似）逐时间窗的颗粒尺寸 R、间距 D、生长阶段、金属厚度
  的替代量，以及它们的可靠性
- **Related code**: [`src/gimap/features/sputter/`](../../../src/gimap/features/sputter/)（说明见其 `README.md`）、
  `app/main_window.py`（`_create_sputter_page`、`running_jobs`、`_language_pages`、`shutdown`）、
  `app/presentation/i18n/zh_sputter.py`、`zh_sputter_report.py`、`zh_sputter_literature.py`、`zh_sputter_toolkit.py`；
  模型 `utils/sputter_toolkit/`（GUI 只按路径以子进程调用；文献方法在 `sputtergisaxs/sputter_literature.py`，已发表数值在
  `configs/literature_reference_v1.json`）
- **Related tests**: [`tests/test_sputter_feature.py`](../../../tests/test_sputter_feature.py)、
  [`tests/test_sputter_page.py`](../../../tests/test_sputter_page.py)（假的安装器、模型与文件头读取：
  [`tests/sputter_fixtures.py`](../../../tests/sputter_fixtures.py)）、
  [`tests/test_sputter_toolkit_bundle.py`](../../../tests/test_sputter_toolkit_bundle.py)（工具包，含文献方法）
- **Last verified**: 2026-10-23（离线测试；编造的扫描名与数值；文献方法另在本机真实结果的副本上核对过，数值不进仓库）

## 调用链

1. **数据 ▸ 选择…**：`SputterGeometryMixin.set_scan_folder` → TaskRunner 键 `sputter_inspect` →
   `SputterService.inspect` → `CbfHeaderInspector`（只读第一帧的文本头，几 kB）→ `ScanInfo` →
   `application.scan.prefill`（每组几何值的来源：本文件夹上次的值 → 匹配的用户预设 → 文件头 `Wavelength` 行或同扫描
   NeXus 的能量 → 上次使用的值；平场文件名里的 `E<eV>` 是探测器标定能量，只写在字段旁边，不填入；样品滚转角和掩膜不取
   “上次使用的值”，没有时滚转角为 0 并标“假定”）→ 表单和来源标签。读头期间用户改过的字段保留用户的值。
2. **预测**：`SputterPredictMixin.predict` → 所选文件夹的文件头还没读完（`header_ready()` 为假）时先排队，读完、表单
   填好后再开始；文件头说没有帧（`ScanInfo.problem`）时拒绝 → `AnalyzeRequest.problems()`、`toolkit_problem()` 先在界面线程检查 →
   记住各值（`remember`）→ TaskRunner 键 `sputter_predict` 只提交一个任务 `SputterService.predict`：
   `installer.verify()`（子进程 `import torch`，`PYTHONNOUSERSITE=1`）→ 缺少时 `installer.install()`（pip，见下）→
   再验证 → `runner.analyze()`（`BundledToolkit`：控制台版 `sys.executable`、`--device cpu`、
   `--workers max(1, min(8, 核数−1))`、光束中心 −0.5 px 换成工具包的像素中心索引、`--rate-nm-s/--t0-s/--t-end-s`；先写
   `gimap_request.json`）→ 读结果（默认 `physical`）。事件经 `eventReceived` 信号回到界面线程：`PredictStep`、
   `InstallProgress`、工具包的输出行（`ProgressTracker`：`SPUTTER_PROGRESS`、`INTAKE_PLAN`、`INTAKE_CHUNK`、
   `INTAKE_EXCLUDED`）。失败包装成 `Failure`（消息 + pip 或模型的最后几行，放进“详情”）。
3. **取消**：`_stop = True`；`process.run_streaming` 每 0.2 s 检查一次（不等输出行），用 `taskkill /T /F` 结束整棵进程树。
   pip 写入软件包（`Installing collected packages`）之后不再停止，等 pip 结束；关闭窗口时 `dispose()` 设为 `"now"`，
   连这一步也停止。
4. **改沉积速率、t0 或沉积结束时间**（已有结果时）：只在显示的结果属于所选扫描时（`StructureResult.scan_folder` 读自
   `gimap_request.json`；不知道扫描的结果属于打开它时所选的文件夹）`rebuild_physical` → 工具包
   `physical --folder OUT [--rate-nm-s] [--t0-s] [--t-end-s]`，不读帧；否则这些值留给所选扫描的下一次预测。比较的是数值
   （`thickness_values`：有速率没有 t0 时 t0 = 0；没有速率时 t0 和结束时间都不触发重写），结束时间不晚于 t0 时提示错误
   （`deposition_end_ok`），重写期间按下的“预测”排在它之后。显示的解释不变（传当前的 `variant`；去掉速率后文献方法不再提供，
   `SputterService.open` 退回 `physical`）。**加入精修时间窗…**：`table` 再 `physical`。
5. **结果**：`StructureFolderReader.available`（`physical` 始终第一，是默认解释；`literature` 在摘要
   `literature.status == "ok"` 时紧跟其后，作为对网络的核对，从不作为默认；然后 `window` → `selected` → `sequence`）→
   `show_result`：结果行（成膜起点或文献方法
   的渗流、“强度调制周期（可能是样品旋转）”、Cu 说明）、`TrendPlot`（物理路径和文献方法：淡色逐窗点 + 趋势线；有速率时上轴给
   δ，给出沉积结束时间时刻度只到 rate × (t_end − t0)；金属厚度栏：t_eq 带、名义厚度线（沉积结束后为平台，并标出结束时间）、
   膜时间窗阴影和“film: thickness not determined”，文献方法下只画 δ；成膜起点或渗流及区间）、
   `LiteratureTab`（δ 上四幅图：D 与已发表 D(δ)、2R/D、覆盖率、η；下面是检验表、注意事项和参考文献）、时间窗表（只列有值的列）、
   可靠性报告（领域层按英文模板写，界面用 `texts.text` 翻译；工具包自己的句子用 `texts.note_text` 按
   `toolkit_texts.TOOLKIT_TEMPLATES` 匹配后翻译）、结构示意（膜时间窗画连续膜，不画颗粒）。

## 文献方法（`domain/literature.py`、`presentation/literature_tab.py`）

- 工具包有沉积速率时写 `structure_physical.csv` 的 `delta_nm`、`D_lit_nm`、`D_lit_trend_nm`、`R_lit_nm`、`R_lit_trend_nm`、
  `two_R_over_D_lit(_trend)`、`coverage_lit`、`two_R_over_D_net`、`coverage_net`、`eta_net`、`lit_status`、`lit_reason`，以及摘要
  的 `literature`（状态、渗流及 68% 区间、已发表区间、参考曲线、检验、说明、参考文献）。渗流的 `state`：`reached`、
  `first_window`、`not_determined`（区间 `[下端, None]`，没有时间）、`not_reached`；区间含窗间离散和速率、t0 的误差
  （`rate_error`、`t0_error_s`）。`percolation()` 对开放的一端返回 None，不用渗流时间代替；`TrendPlot._interval` 和
  `LiteratureTab._marks` 把它画到峰阶段末尾（`last_time_s`、`last_delta_nm`）并加斜线。
- `literature_rows(rows, summary)` 把 `R_lit_nm`/`D_lit_nm`（及趋势）当作 R/D 显示，等级 `Lit`；渗流时间之后的每个时间窗
  `film` 为真（也包括无峰和未使用的峰）；渗流之后 D = 2π/q* 继续给出，R 不给。`lit_status` 保留每个窗的原因（宿主峰、
  离群、跳变显示为离群、晚出现的峰），不被渗流覆盖。网络的形状量（σ、埋深、L）在这个解释里不显示，网络值仍在 `network_*`
  列和其他解释里。
- 新速率或去掉速率：`_thickness_changed` 传当前的解释；去掉速率后文献方法不再提供，`rebuild_physical` 打开默认的物理路径。
  旧结果摘要里的 `default_ready` 不再使用，工具包也不再写它。
- 沉积结束时间 t_end（挡板手动关闭，扫描可以在沉积结束后继续）：工具包的 `nominal_thickness_nm` = rate·clip(t − t0, 0,
  t_end − t0)，文献方法用同一个 δ（t_end 之后不变；跳变、渗流及其区间、TS 起点的 δ 都按沉积时间 min(t, t_end) 计算），摘要的
  `thickness` 和 `literature` 记 `t_end_s`、`t_end_given`，`thickness.t_end_note` 说明给出或未给出。`domain/physical.deposition_end`
  读它；可靠性页的名义厚度句子在给出时说明平台；“文献对照”页标题写明之后的时间窗停留在哪个 δ，TS 起点的 δ 也截在 t_end。
- 文献方法的 R 按构造与文献一致；页面和手册都这样说明，独立检验是 D(δ)、渗流与已发表区间、网络的 2R/D（覆盖率由它决定，
  算一项）和质量守恒。
- 可靠性页的工具包说明用 `note=note_text` 翻译（带数值的句子也按模板匹配）。上方的 δ 轴是子坐标轴：
  `theme/figures.apply_figure_colors` 也给 `child_axes` 着色。

## 安装 PyTorch（`infrastructure/installer.py`、`pip_output.py`）

- 命令：`python -m pip install --disable-pip-version-check --no-input --no-user -c <临时约束：numpy==已装版本> -r
  requirements-sputter.txt`，pip ≥ 24.1 时加 `--progress-bar raw`。环境：`PYTHONNOUSERSITE=1`、去掉 `PYTHONPATH`；
  解释器在 `.gimap` 里时 `PIP_CACHE_DIR=.gimap/cache/pip`。
- 没有 pip（`uv venv` 建的环境）：先 `python -m ensurepip --upgrade --default-pip`。
- 进度：`Collecting` → 查找；`Downloading …whl (大小)` → 下载；`Progress a of b` → 百分比；`Using cached`；
  `Installing collected packages` → 不可取消；`Successfully installed`。
- 失败分类（按 pip 的最后几行）：磁盘满 → 网络（先于“找不到版本”，因为连不上时 pip 也这样说）→ 权限 →
  没有合适版本 → 没有 pip → 其他（退出码）；导入失败（DLL）在验证时判断。每条消息都给出手动命令。
- `GIMAP_SPUTTER_NO_INSTALL=1`：不调用 pip（测试、管理员）。
- pip 写入软件包时被结束（关闭窗口）：Python 旁边留下 `gimap_sputter_install.unfinished`；之后 PyTorch 加载失败时，下一次
  预测用 `--force-reinstall --no-deps` 把 `requirements-sputter.txt` 的软件包重装一遍（不碰 NumPy 等依赖：GIMaP 正在用它们）。
  `ensurepip` 超时或无法启动时报“没有 pip”。

## 记住的内容（偏好 `sputter_page`）

`folders`（每个扫描文件夹上次预测的几何、金属、聚合物、掩膜、时间窗、沉积速率、t0、沉积结束时间 `t_end`；最多 300 个）、
`last`（上次的值，不含速率、t0 和 `t_end`；掩膜和样品滚转角不从这里带到别的文件夹）、`presets`（用户的几何预设：名称、探测器序列号、平场文件名、起止日期和几何）、`last_scan`、
`results_root`、`last_result`（下次启动时打开）、`advanced_open`。旧版的 `python`、`toolkit`、`device` 不再使用；旧版的
`sdd`、`alpha`、`energy`、`window`、`mask` 作为“上次的值”读入一次。这些值来自用户的实验，只存在用户数据文件夹，
不进仓库。

## 诚实性约定

- 金属膜厚度：现有模型测不出（TS 网络在模拟中后验等于先验）。只给岛体积等效厚度 t_eq（有条件，膜时间窗不给）和
  名义厚度 速率 × (t − t0)（给出沉积结束时间时此后不变）；膜时间窗写“厚度未确定”。“膜 + 稀疏颗粒”的新模型另行开发。
- Cu 可以选，但处处标明“用 Ag 模型近似，仅供参考”（结果行、报告、示意图、保存的图像和分镜图）。
- 结果文件夹的代码包含模型版本（`BundledToolkit.model_id()`：`MANIFEST.json` 的 SHA-256，没有时用文件指纹），换了模型
  不会重用旧模型的网络输出；`gimap_request.json` 记 `model_id` 和所选的 `scan_folder`。
- 物理路径不在相邻时间窗之间切换精修值和网络值（等级 P）；精修值留在表里。强度的周期性调制只称“强度调制周期（可能是样品旋转）”：
  有的扫描调制来源不明，不断言是旋转。
- 文献对照的已发表数值都带引用键、DOI 和来源类型（正文/表、图读、计算、综述、摘要）；用户的速率只来自界面输入，不写进仓库。
- 页面的方法是网络：物理路径始终是默认解释；文献方法只作为对它的核对（第二个选择和“文献对照”页），按构造与文献一致。
