# 溅射（Sputter）

- **Status**: Current（2026-10-22 重新设计：模型随仓库提供，选文件夹后按“预测”即可）
- **Scope**: 原位溅射序列（聚合物上的 Ag、Au；Cu 用 Ag 模型近似）逐时间窗的颗粒尺寸 R、间距 D、生长阶段、金属厚度
  的替代量，以及它们的可靠性
- **Related code**: [`src/gimap/features/sputter/`](../../../src/gimap/features/sputter/)（说明见其 `README.md`）、
  `app/main_window.py`（`_create_sputter_page`、`running_jobs`、`_language_pages`、`shutdown`）、
  `app/presentation/i18n/zh_sputter.py`、`zh_sputter_report.py`；模型 `utils/sputter_toolkit/`（另一分支加入，
  GUI 只按路径以子进程调用）
- **Related tests**: [`tests/test_sputter_feature.py`](../../../tests/test_sputter_feature.py)、
  [`tests/test_sputter_page.py`](../../../tests/test_sputter_page.py)（假的安装器、模型与文件头读取：
  [`tests/sputter_fixtures.py`](../../../tests/sputter_fixtures.py)）
- **Last verified**: 2026-10-22（离线测试；编造的扫描名与数值；没有用真实数据，没有联网安装）

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
   `--workers max(1, min(8, 核数−1))`、光束中心 −0.5 px 换成工具包的像素中心索引、`--rate-nm-s/--t0-s`；先写
   `gimap_request.json`）→ 读结果（默认 `physical`）。事件经 `eventReceived` 信号回到界面线程：`PredictStep`、
   `InstallProgress`、工具包的输出行（`ProgressTracker`：`SPUTTER_PROGRESS`、`INTAKE_PLAN`、`INTAKE_CHUNK`、
   `INTAKE_EXCLUDED`）。失败包装成 `Failure`（消息 + pip 或模型的最后几行，放进“详情”）。
3. **取消**：`_stop = True`；`process.run_streaming` 每 0.2 s 检查一次（不等输出行），用 `taskkill /T /F` 结束整棵进程树。
   pip 写入软件包（`Installing collected packages`）之后不再停止，等 pip 结束；关闭窗口时 `dispose()` 设为 `"now"`，
   连这一步也停止。
4. **改沉积速率或 t0**（已有结果时）：只在显示的结果属于所选扫描时（`StructureResult.scan_folder` 读自
   `gimap_request.json`；不知道扫描的结果属于打开它时所选的文件夹）`rebuild_physical` → 工具包
   `physical --folder OUT [--rate-nm-s] [--t0-s]`，不读帧；否则这些值留给所选扫描的下一次预测。比较的是数值
   （有速率没有 t0 时 t0 = 0），重写期间按下的“预测”排在它之后。**加入精修时间窗…**：`table` 再 `physical`。
5. **结果**：`StructureFolderReader`（`physical` → `window` → `selected` → `sequence`）→ `show_result`：结果行（成膜起点、
   旋转周期、Cu 说明）、`TrendPlot`（物理路径：淡色逐窗点 + 趋势线；金属厚度栏：t_eq 带、名义厚度线、膜时间窗阴影和
   “film: thickness not determined”、成膜起点及区间）、时间窗表（只列有值的列）、可靠性报告（领域层按英文模板写，
   界面用 `texts.text` 翻译）、结构示意（膜时间窗画连续膜，不画颗粒）。

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

`folders`（每个扫描文件夹上次预测的几何、金属、聚合物、掩膜、时间窗、沉积速率、t0；最多 300 个）、`last`（上次的值，
不含速率和 t0；掩膜和样品滚转角不从这里带到别的文件夹）、`presets`（用户的几何预设：名称、探测器序列号、平场文件名、起止日期和几何）、`last_scan`、
`results_root`、`last_result`（下次启动时打开）、`advanced_open`。旧版的 `python`、`toolkit`、`device` 不再使用；旧版的
`sdd`、`alpha`、`energy`、`window`、`mask` 作为“上次的值”读入一次。这些值来自用户的实验，只存在用户数据文件夹，
不进仓库。

## 诚实性约定

- 金属膜厚度：现有模型测不出（TS 网络在模拟中后验等于先验）。只给岛体积等效厚度 t_eq（有条件，膜时间窗不给）和
  名义厚度 速率 × (t − t0)；膜时间窗写“厚度未确定”。“膜 + 稀疏颗粒”的新模型另行开发。
- Cu 可以选，但处处标明“用 Ag 模型近似，仅供参考”（结果行、报告、示意图、保存的图像和分镜图）。
- 结果文件夹的代码包含模型版本（`BundledToolkit.model_id()`：`MANIFEST.json` 的 SHA-256，没有时用文件指纹），换了模型
  不会重用旧模型的网络输出；`gimap_request.json` 记 `model_id` 和所选的 `scan_folder`。
- 物理路径不在相邻时间窗之间切换精修值和网络值（等级 P）；精修值留在表里。样品旋转造成的强度周期在结果行和报告里给出。
