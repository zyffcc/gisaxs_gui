# 溅射（Sputter）

- **Status**: Current（2026-10-22 重新设计：模型随仓库提供，选文件夹后按“预测”即可；2026-10-23 加入文献方法和“文献对照”页，
  并加入可选的沉积结束时间；文献方法有数值（有沉积速率、金属峰的时间窗足够）时是默认解释，网络的物理路径是第二个解释，
  “文献对照”页比较两者；2026-10-26 加入由 q_z 干涉条纹得到的岛高度 H，显示结果后在后台测量；2026-11-14 页面重写：
  打开时是**主屏**（“分析” Analyze，调用仓库外的分析包；实测图与模型图并排、候选解、数值块、演化），原来的页面整个收进第二个
  视图**详情**（Details），行为不变——见下面“主屏与分析（Analyze）”；同日补充：多孔基底结果的数值块、详情里的“静态帧
  （沉积之后测量）”、主屏图样区的“图样 ｜ 实空间”切换——见“多孔基底”“静态帧”“实空间示意”三节）
- **科学说明**：各量的定义、规则和阈值见 [Physics and Methods Reference](../../Physics_Manual.md) 第 12 章；本文写调用链、
  文件与界面约定。
- **Scope**: 原位溅射序列逐时间窗的结构。主屏（分析包）：晶粒半径 R、高度 H、中心间距 D、间隙、尺寸分布宽度、覆盖率，
  每个时间窗若干候选解，实测图与模型图并排；详情里的早先方法（随仓库的网络：聚合物上的 Ag、Au；Cu 用 Ag 模型近似）：
  颗粒尺寸 R、间距 D、生长阶段、金属厚度的替代量，以及它们的可靠性
- **Related code**: [`src/gimap/features/sputter/`](../../../src/gimap/features/sputter/)（说明见其 `README.md`）、
  `app/main_window.py`（`_create_sputter_page`、`running_jobs`、`_language_pages`、`shutdown`）、
  `app/sputter_handover.py`（在分析中查看，经 `HandoverMixin`）、`app/project.py`（`missing`：也查主屏结果的文件夹）、
  `app/presentation/i18n/zh_sputter.py`、`zh_sputter_report.py`、`zh_sputter_literature.py`、`zh_sputter_toolkit.py`、
  `zh_sputter_handover.py`、`zh_sputter_fringes.py`、`zh_sputter_client.py`（主屏与分析）；
  主屏与分析：`domain/client.py`、`application/client.py`、`application/client_progress.py`、`infrastructure/client.py`、
  `presentation/views/sputter_main_view.py`、`presentation/client_flow.py`、`client_panel.py`、`client_texts.py`、
  `pattern_view.py`、`evolution_view.py`、`real_space_view.py` 与 `domain/real_space.py`（实空间示意）；静态帧：
  `application/client_static.py`、`infrastructure/client_static.py`、`presentation/static_flow.py`、
  `presentation/views/sputter_static_view.py`；分析包不在仓库里（`GIMAP_SPUTTER_CLIENT`、用户选的文件夹或 `utils/sputter_client/`）；
  由 q_z 干涉条纹得到的岛高度：`domain/fringes.py`、`application/fringes.py`、`infrastructure/fringes.py`、
  `presentation/fringe_flow.py`、`presentation/fringe_panel.py`，工具 `utils/sputter_fringes/`（numpy/scipy，子进程）；
  模型 `utils/sputter_toolkit/`（GUI 只按路径以子进程调用；文献方法在 `sputtergisaxs/sputter_literature.py`，已发表数值在
  `configs/literature_reference_v1.json`）
- **Related tests**: [`tests/test_sputter_feature.py`](../../../tests/test_sputter_feature.py)、
  [`tests/test_sputter_page.py`](../../../tests/test_sputter_page.py)（假的安装器、模型与文件头读取：
  [`tests/sputter_fixtures.py`](../../../tests/sputter_fixtures.py)）、
  [`tests/test_sputter_toolkit_bundle.py`](../../../tests/test_sputter_toolkit_bundle.py)（工具包，含文献方法）、
  [`tests/test_sputter_show_in_analyze.py`](../../../tests/test_sputter_show_in_analyze.py)（在分析中查看：读时间窗、菜单、
  交接；真实分析页上用编造的 TIFF 帧核对求和）、
  [`tests/test_sputter_fringes.py`](../../../tests/test_sputter_fringes.py)（干涉条纹工具：合成的生长层、无条纹的切线、
  命令行；数据：[`tests/sputter_fringe_data.py`](../../../tests/sputter_fringe_data.py)）、
  [`tests/test_sputter_fringes_page.py`](../../../tests/test_sputter_fringes_page.py)（页面：H 栏、时间窗列、可靠性段落、后台测量）、
  [`tests/test_sputter_client.py`](../../../tests/test_sputter_client.py)（分析包的读取器、命令行与环境、进度行、服务；
  多孔基底与静态帧的读取、静态帧的命令行与服务；实空间示意的排布）、
  [`tests/test_sputter_main_screen.py`](../../../tests/test_sputter_main_screen.py)（主屏：数值块、换解、保存选择、空状态、
  分析与取消、语言、项目；多孔基底的数值块、静态帧系列与“分析这个样品”、实空间示意的切换、长路径不撑宽详情左栏；
  假的分析包进程与编造的结果：[`tests/sputter_client_fixtures.py`](../../../tests/sputter_client_fixtures.py)）
- **Last verified**: 2026-11-14（主屏与分析：离线测试；多孔基底、静态帧和实空间示意另用本机几份真实结果的副本在离屏窗口里
  看过，数值不进仓库；另在本机用真实的旧版分析包核对过命令行和环境——给一个不存在的帧文件夹，
  分析包解析完全部参数后拒绝它，不读帧；主屏另用本机一份真实结果的副本看过，数值不进仓库）；2026-11-01（状态行、默认解释、
  工作进程数和光斑移出样品的规则对照代码）；其余 2026-10-23（离线测试；编造的扫描名与数值；文献方法另在本机真实结果的副本上
  核对过，数值不进仓库）

## 页面的两个视图

`views/sputter_page_view.py` 从上到下：**输入栏**（两个视图都显示）和一个 `QStackedWidget`（`view_stack`）。

- **输入栏**（`views/sputter_main_view.py` 的 `_input_bar`）：扫描文件夹（`scan_picker`，原来在“数据”区）、金属
  `sample_metal`（Ag / Au / Cu，与详情里早先方法的 `metal` 是同一个选择，互相同步；早先方法的下拉项仍把 Cu 标成“用 Ag 模型
  近似”）、基底 `substrate`、速率 `sample_rate` 与 **QCM 读数** `qcm_check`、主按钮**分析** `analyze_button`；下一行是一句几何
  `geometry_summary`（能量、距离、入射角、光束中心、滚转角；缺什么就只说“还需填写的几何：…”，选了文件夹后用警示色）和
  **编辑**（切到详情、滚到几何区、光标放在第一个空字段）、**打开结果…**、视图开关 `view_switch`（主屏 / 详情）；再下一行是
  扫描的一句话 `scan_summary`；右边是分析的 `JobStatus`（`analyze_status`：取消、详情——详情按钮切到详情视图并显示日志）。
- **主屏**（`_main_view`，默认）：没有结果时是空状态（`client_empty`：没有分析包时说明怎么给出它，按钮“选择文件夹…”；
  有分析包时说明怎么得到结果，按钮“打开结果…”）。有结果时（`client_view`）：时间窗选择（上一个、滑块、下一个、
  “t = … s · 已沉积 … nm”）与切换**图样 ｜ 实空间**（`image_switch`）；实测图与模型图（`PatternPair`），或所选解的实空间
  示意（`RealSpacePicture`，同一个位置，`pattern_host` 是两页的 `QStackedWidget`）；解的选择（`SegmentedControl`，页面按时间窗的条目动态建立：
  候选 1…3、精修解，各带拟合偏差）、**恢复默认解**、一句说明；数值块（半径 R、高度 H、间距 D、间隙、尺寸分布宽度、覆盖率；
  没有的量不显示）和所显示解的拟合偏差；演化图（`EvolutionPlot`）。
- **详情**（`_details_view`）：原来的整个页面——左栏（标题、样品与几何、**分析选项**、早先的方法“预测”及其任务状态、
  静态帧（`sputter_static_view.py`，默认收起）、更多选项）和右边五个标签页。分析选项（`_analysis_section`）：模式（自动 / 快速 / 精修 / 完整）、沉积开始、沉积持续时间、
  聚合物膜厚（只在基底是聚合物膜时显示）、分析包的文件夹与“选择文件夹…”、打开报告、打开文件夹（所显示结果的）、日志。

原有控件的属性名都没有变（`scan_picker`、`metal`、`rate`、`job_status`、`predict_button`、`tabs` …）；“预测”不再是主按钮样式
（页面的主操作是“分析”）。1440×900 和 1280×800 下主屏不需要展开任何东西（`test_the_main_screen_fits_the_window…`）；
详情视图右边的标签页最小宽度是原来就有的。

## 主屏与分析（Analyze）

1. **分析**：`SputterClientMixin.analyze`（`client_flow.py`）→ 文件头没读完时排队（与“预测”相同的 `_run_queued` /
   `_forget_queued`，协作式 `super()`）→ `ClientRequest.problems()`（文件夹、速率 > 0、能量或波长、距离与入射角、光束中心
   两个都给或都不给、金属、基底、模式、持续时间与膜厚为正）和 `ClientService.package().problem` 先在界面线程检查 →
   记住各值（几何照“预测”的办法 `_remember`；基底、速率、QCM、开始、持续时间、膜厚按文件夹记在偏好 `client_folders`，
   基底、模式、QCM 另记为 `client_last`）→ TaskRunner 键 `sputter_client` 提交 `ClientService.analyze(request,
   service.results_folder(旧请求), emit, should_stop)`。
2. **服务**（`application/client.py`）：输出文件夹 = 该扫描的结果文件夹（`<结果根>/<扫描>-<旧请求的代码>`）下的
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
   - **光束中心**：页面用 GIMaP 的像素坐标，交给分析包前减 0.5（`index_center_from_canonical`，与随仓库的工具包相同：两者
     把它交给同一个 `prepare`）；留空时给 `--auto-centre`（页面说“由帧估计”；否则分析包会用它自己束线的默认值）。
   - **滚转角**总是给出（留空为 0，与页面说的一致）；**掩膜**留空时不给 `--mask`，分析包用它自带的像素掩膜。
   - **数据与工作文件夹**：`<用户数据>/sputter_client/<包版本前 12 位>/data/<package-mask 或 mask-<掩膜 SHA-256 前 12 位>>`
     和 `…/work`（分析包要求一个数据文件夹只用一种像素掩膜）。包版本 = `MANIFEST.json` 的 SHA-256，没有时是文件指纹。
   - **中断的导入**：导入被取消后，标签文件夹留在数据文件夹里，分析包会永远拒绝这个标签（“exists: never overwritten”）。
     GIMaP 在导入那一步说“done”时写标记 `<数据根>/<标签>.gimap.json`；下次运行前，标签文件夹存在但没有 `scan.json` 或没有
     标记时把它删掉（只删 `gui_` 开头、GIMaP 自己建的那个文件夹）。
   - 运行前写 `<输出>/gimap_request.json`（请求、标签、分析包路径与版本、Python、命令行）。
4. **进度**（`application/client_progress.py`）：`[n/N] <step> ...`、`done in … s` / `FAILED (…)`、`status: …`；任务状态说
   “第 n 步，共 N 步：…”，进度条显示已完成的步骤所占的份额（步骤内部没有进度）。其余的行进日志（详情 ▸ 分析选项）。
5. **结果**（`ClientFolderReader` → `domain/client.py`）：`analysis.json`（状态、模型族、模式、各文件路径）、
   `main/main.json`（`wanted`、`windows[]`：`t`、`t_rel`、`h_nom`、`arrays`、`entries[]`、`pick`、`smooth`；或 `{t, error}`）、
   `main/w_<t>.npz`（`q`、`af`、`measured`、`models`、`labels`）。没有的量就是没有（数值块隐藏、演化线断开）；`arrays` 只接受
   纯文件名；`{t, error}` 的时间窗在滑块上保留，图的位置说明原因。**打开结果…**接受输出文件夹、它的 `main` 文件夹，或
   扫描的结果文件夹（取其中最新的 `iampred-*`）；只有早先方法结果的文件夹改在详情里打开。
6. **显示哪个解**：默认是 `smooth`（全扫描平滑演化选出的），没有时是 `pick`（拟合偏差最小），再没有时是偏差最小的条目。
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

详情里一节（`sputter_static_view.py`，默认收起；行为在 `static_flow.py`）：一个样品的帧（文件夹、一个 CBF 文件，或 zip 包加
`::` 后的包内文件夹；选 zip 时列出包里有 CBF 的文件夹让用户选）、样品名、名义厚度（nm）、可选的空基底帧、系列文件夹、
模式（快速 / 精修 / 完整），这些帧的几何（能量、距离、光束中心、滚转角、掩膜；留空用分析包自己的值，“分析包的值”把它们
填进来；“分析包自带的掩膜”按文件名列出包里的 `.npy`），按钮**分析这个样品**及其任务状态和日志。

1. `StaticService.analyze`（`application/client_static.py`）→ `StaticPackageRunner`（`infrastructure/client_static.py`）：
   `python -u -m iampred.static --frames … --label … --nominal-nm … --out <系列> [--reference-frames …] [--sample …]
   [--substrate …] --mode … --workers N [--sdd-mm …] [--energy-ev …] [--beam-row … --beam-col …] [--roll …] [--mask …]`，
   环境同“分析”，另加 `IAM_STATIC_DATA`。分析包的默认几何从它 `static` 模块的源码里读（不导入）；没有这个模块的包不能
   分析静态帧，页面这样说。
2. **系列**是一个输出文件夹；GIMaP 在里面写 `gimap_series.json`（编号、像素掩膜、每个样品由什么分析得到）。导入的帧在
   `<用户数据>/sputter_client/static/<系列编号>`，与原位的数据文件夹分开（像素掩膜不同）。同一个样品名换了帧或数值再分析时，
   先问，再把旧的导入和系列里的那个样品删掉重做；完全相同的请求直接显示系列。
3. 结束后主屏显示整个系列：样品代替时间窗（选择器写“样品名 · 名义厚度 … nm”），每个样品一个解（`模型解`，各帧的中位数），
   演化图横轴是名义厚度，不出现“时间”字样；数值的范围（各帧与网络区间）在数值块的提示里。**打开结果…**也能打开系列。
4. “分析”和“分析这个样品”不同时运行：一个在跑时另一个等待。系列文件夹、模式、几何、掩膜记为上次用的值，整节进项目文件。

## 实空间示意（Real Space）

主屏时间窗选择右边的切换 **图样 ｜ 实空间**：实空间把所选的解画成形貌（`presentation/real_space_view.py`；排布是纯 Python 的
`domain/real_space.py`）。换时间窗、换解，示意跟着变；只在这一页显示时才画。

- 用到的值：半径 R、间距 D、尺寸分布宽度、高度 H，以及分析包另给的 `base_nm`（晶粒下面连续底层的厚度，0 为分立的岛；
  不在 `wanted` 里、没有数值块，读取器为示意单独保留它：`SKETCH_KEYS`）。没有高度时晶粒画成半球。
- 左边截面：穿过一排晶粒，长 4 D；衬底（聚合物膜按 `FILM_SUBSTRATES` 画成聚合物，厚度不按比例），连续底层，晶粒画成半径为
  r、高为 (H − base)·r/R 的半椭圆；横竖同一比例（nm）。右边俯视：边长 7 D 的方块，晶粒在间距 D 的六角排布上各自随机偏移约
  0.12 D（没有长程有序，与模型的干涉函数一致），半径取自以 R 为中位数的对数正态分布；有底层时整片铺一层浅色。
- 随机数只有一个种子、每次按同样的顺序取，所以换解或换时间窗只改变大小，不改变“骰子”。示意下面一行说明位置是随机排布的，
  大小取自所选的解。
- **聚并与逾渗**：相接的晶粒（中心距不大于两半径之和）连成一个团簇（`clusters`，并查集），画成一整块——先画所有轮廓，
  再把填充盖在上面，团簇内部不留边线；截面里相接的晶粒帽同样连成一片。连通视场两条对边的团簇（`spanning`）画得深一些。
  图上方一行说这一时刻的排布（`arrangement`）：分立的岛 / 聚并中：相邻晶粒相接 / 连成贯穿视场的网络（已逾渗）；底层
  至少 0.3 nm 时改说“连续底层 … nm；其上的晶粒：…”。这些都由所选解的 R、D、分布宽度和底层厚度在几何上推出。
- **多孔 TiO₂**：分析包为 TiO₂ 模型族的每个解另给出金属的位置（`porous_geometry`：颗粒半径 75 nm、孔内金属的深度与宽度、
  帽层覆盖率与厚度、岛／帽层／孔内的金属量、团簇高度与间距；读取器经 `SKETCH_KEYS` 保留，不进数值块）。左边是层的
  截面，按比例：TiO₂ 颗粒（`layer_particles`：略留空隙的六角排布，深度到孔内金属以下），孔内金属按深度的高斯
  （`pore_density`）画成颗粒后面的色带，并标出中心深度；最上一排按帽层覆盖率给一部分颗粒戴上帽层。右边放大一个颗粒的
  顶部：帽层按厚度、团簇按半径、高度和间距。下面一行是已沉积量和折合成致密金属的岛、帽层、孔内厚度。没有这些位置的解
  （旧的分析包）说明没有可画的形貌。

## 调用链（详情：早先的方法）

1. **输入栏的扫描文件夹 ▸ 选择…**（两种方法共用）：`SputterGeometryMixin.set_scan_folder` → TaskRunner 键 `sputter_inspect` →
   `SputterService.inspect` → `CbfHeaderInspector`（只读第一帧的文本头，几 kB）→ `ScanInfo` →
   `application.scan.prefill`（每组几何值的来源：本文件夹上次的值 → 匹配的用户预设 → 文件头 `Wavelength` 行或同扫描
   NeXus 的能量 → 上次使用的值；平场文件名里的 `E<eV>` 是探测器标定能量，只写在字段旁边，不填入；样品滚转角和掩膜不取
   “上次使用的值”，没有时滚转角为 0 并标“假定”）→ 表单和来源标签。读头期间用户改过的字段保留用户的值。
2. **预测**：`SputterPredictMixin.predict` → 所选文件夹的文件头还没读完（`header_ready()` 为假）时先排队，读完、表单
   填好后再开始；文件头说没有帧（`ScanInfo.problem`）时拒绝 → `AnalyzeRequest.problems()`、`toolkit_problem()` 先在界面线程检查 →
   记住各值（`remember`）→ TaskRunner 键 `sputter_predict` 只提交一个任务 `SputterService.predict`：
   `installer.verify()`（子进程 `import torch`，`PYTHONNOUSERSITE=1`）→ 缺少时 `installer.install()`（pip，见下）→
   再验证 → `runner.analyze()`（`BundledToolkit`：控制台版 `sys.executable`、`--device cpu`、
   `--workers`：`toolkit.default_workers()` = max(1, min(8, 核数−1))，并且不多于空闲内存装得下的个数
   （(可用内存 − 1 GB) / 0.4 GB，没有 psutil 时只看核数）、光束中心 −0.5 px 换成工具包的像素中心索引、`--rate-nm-s/--t0-s/--t-end-s`；先写
   `gimap_request.json`）→ 读结果（默认 `physical`）。事件经 `eventReceived` 信号回到界面线程：`PredictStep`、
   `InstallProgress`、工具包的输出行（`ProgressTracker`：`SPUTTER_PROGRESS`、`INTAKE_PLAN`、`INTAKE_CHUNK`、
   `INTAKE_EXCLUDED`）。失败包装成 `Failure`（消息 + pip 或模型的最后几行，放进“详情”）。
3. **取消**：`_stop = True`；`process.run_streaming` 每 0.2 s 检查一次（不等输出行），用 `taskkill /T /F` 结束整棵进程树。
   pip 写入软件包（`Installing collected packages`）之后不再停止，等 pip 结束；关闭窗口时 `dispose()` 设为 `"now"`，
   连这一步也停止。
4. **改沉积速率、t0 或沉积结束时间**（已有结果时）：只在显示的结果属于所选扫描时（`StructureResult.scan_folder` 读自
   `gimap_request.json`；不知道扫描的结果属于打开它时所选的文件夹）`rebuild_physical` → 工具包
   `physical --folder OUT [--rate-nm-s] [--t0-s] [--t-end-s]`，不读帧；否则这些值留给所选扫描的下一次预测。比较的是数值
   （`thickness_values`：有速率没有 t0 时 t0 = 0；没有速率时 t0 和结束时间也比较——摘要照样记录它们，结束时间也在尺寸趋势里标出——
   改了就重写，字段和文件夹记忆始终与结果一致），结束时间不晚于 t0 时提示错误
   （`deposition_end_ok`），重写期间按下的“预测”排在它之后。显示的解释不变（传当前的 `variant`；去掉速率后文献方法不再提供，
   `SputterService.open` 退回 `physical`）。**加入精修时间窗…**：先按同样的规则检查字段（`_thickness_refused`），再 `table`
   和 `physical`。
5. **结果**：`StructureFolderReader.available`（摘要 `literature.status == "ok"` 时 `literature` 排第一，是默认解释，
   `physical` 紧跟其后；否则 `physical` 第一；然后 `window` → `selected` → `sequence`）→
   `show_result`：结果行（成膜起点或文献方法
   的渗流、“强度调制周期（可能来自样品台平移）”——读 `modulation_period_s`，旧结果读 `rotation_period_s`（`modulation_period`）——、
   光斑可能移出样品的时间窗数（`beam_off_count`；`beam_off_record` 的 `rule` 为 `extreme_dip_only`、即没有调制时说
   “or interrupted”，1 个时用单数）、Cu 说明）、`TrendPlot`（物理路径和文献方法：淡色逐窗点 + 趋势线，
   `beam_off_sample` 的时间窗在 R、D、TS 长度 L 和 t_eq 上画成空心灰点“beam may be off the sample (stage translation)”
   （没有调制时为“beam may be off the sample or interrupted”），淡色点、t_eq 线不含它们，文献方法下
   D 栏用它们的 2π/q*；有速率时上轴给
   δ，给出沉积结束时间时刻度只到 rate × (t_end − t0)，结束在第一个时间窗之前时不画 δ 轴；金属厚度栏：t_eq 带、名义厚度线（沉积
   结束后为平台，结束时间落在各时间窗之间时画竖线，在最后一个之后或第一个之前时只写进图例，时间轴不被拉长）、
   膜时间窗阴影和“film: thickness not determined”，文献方法下只画 δ；成膜起点或渗流及区间）、
   `LiteratureTab`（δ 上四幅图：D 与已发表 D(δ)、2R/D、覆盖率、η；`lit_status` 为 `beam_off` 的峰与其他未使用的峰一样画空心圈；
   下面是检验表、注意事项和参考文献）、时间窗表（只列有值的列；“Beam may be off the sample”列的悬停提示是原因，“Beam note”列是
   `beam_off_reason`）、
   可靠性报告（领域层按英文模板写，界面用 `texts.text` 翻译；工具包自己的句子用 `texts.note_text` 按
   `toolkit_texts.TOOLKIT_TEMPLATES` 匹配后翻译）、结构示意（膜时间窗画连续膜，不画颗粒；`beam_off_sample` 的时间窗
   按自己的数值画并加说明，`smoothed_rows` 的中位数和 `storyboard_indices` 不用它们）。
6. **在分析中查看**（`presentation/window_menu.py`）：时间窗表一行（`customContextMenuRequested`，表格原有的复制行／复制表格
   留在同一菜单）、结构示意（`SketchView.windowMenuRequested`：显示的时间窗）、尺寸趋势（画布的右键位置经
   `mouseEventCoords` 换成时间，取时间上最近的时间窗）的右键菜单 → `show_window_in_analyze(frame)` →
   `SputterService.window` → `BinnedWindowReader`（`infrastructure/windows.py`：在 `binned/chunk_NNNNN.json` 里按 `frame`
   找到这一行，`raw_members` 只接受纯文件名，帧在 `gimap_request.json` 的 `scan_folder` 里；`intake/geometry.json` 的
   `db_col/db_row` 是模型自己估计的光束中心，+0.5 px 换成 GIMaP 坐标，只写进通知；问题用 `domain/windows.py` 的模板，
   在溅射页上说）→ 组合根注入的 `set_analyze_target` = `SputterHandoverMixin._show_sputter_window`
   （`app/sputter_handover.py`）：`AnalyzePage.summed_locked()`（自动分析、批处理、监视文件夹时拒绝，通知说明原因）→
   `matching_profile`（距离 ±0.5 mm、波长 ±0.1%，同探测器优先；不新建配置）→ `AnalyzePage.show_summed`
   （`analyze/presentation/bindings/frame_import.py`：清空列表 → 非交互导入 → 列出后设求和 N = 帧数、αi、给出时的光束中心
   （对所有形状）、配置，再显示第一帧：一次分析就是全部帧之和）→ 通知（时间窗、求和帧数、几何、是否替换了原先的帧；没有配置时
   一直显示，写出应输入的距离和波长）。

7. **由 q_z 干涉条纹得到的岛高度**（`presentation/fringe_flow.py`）：`show_result` → `_update_fringes` →
   `FringeService.cached`（结果文件夹的 `fringes_v1/`，再看用户数据文件夹的 `sputter_fringes/<结果文件夹>/`；
   `FringeFolderStore.fresh`：工具源码的 SHA-256、`binned/run.json` 的 SHA-256、金属、光斑可能移出样品的时间窗都相同）→
   没有新鲜的输出而结果有 `binned/run.json` 时，TaskRunner 键 `sputter_fringes` 提交 `FringeService.measure` →
   `FringeTool`（控制台版 `sys.executable` 运行 `utils/sputter_fringes/fringes.py --folder --out --group-seconds 1.0
   --toolkit --metal`，`run_streaming`，新的结果或关闭页面时停止）→ 完成时若仍显示同一结果，重画尺寸趋势（`TrendPlot.draw`
   的 `fringes`：`fringe_panel.draw_heights`，简单视图也显示）、时间窗表（`table_columns(result, fringes)` 在 D 列之后插入
   两列，`height_cells` 取时间上最近的时间组）和可靠性报告（`fringe_paragraph`）。失败只写进“详情”的日志。

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
  读它；可靠性页的名义厚度句子在给出时说明平台；“文献对照”页标题写明之后的时间窗停留在哪个 δ（结束在最后一个时间窗之后时
  改说每个时间窗都是 rate × (t − t0)，工具包的说明用 `NOTE_T_END_AFTER`），TS 起点的 δ 也截在 t_end。工具包命令行在读帧之前
  拒绝不是正数的速率和不在 t0 之后的结束时间（`_check_thickness`）。
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
`sdd`、`alpha`、`energy`、`window`、`mask` 作为“上次的值”读入一次。主屏的分析另有：`client_folders`（每个扫描文件夹上次分析的
基底、速率、是否 QCM 读数、模式、沉积开始、持续时间、膜厚；最多 300 个；分析过的文件夹再选中时取回它自己的值，
没分析过的文件夹不沿用上一个扫描的速率、开始、持续时间和膜厚，但用户刚输入、还没分析的值保留——`_client_owner`）、
`client_last`（上次的基底、模式、QCM）、
`client_package`（用户选的分析包文件夹）、`last_client_result`（下次启动时打开）。这些值来自用户的实验，只存在用户数据文件夹，
不进仓库。

## 诚实性约定（详情：早先的方法）

- 金属膜厚度：现有模型测不出（TS 网络在模拟中后验等于先验）。只给岛体积等效厚度 t_eq（有条件，膜时间窗不给）和
  名义厚度 速率 × (t − t0)（给出沉积结束时间时此后不变）；膜时间窗写“厚度未确定”。“膜 + 稀疏颗粒”的新模型另行开发。
- Cu 可以选，但处处标明“用 Ag 模型近似，仅供参考”（结果行、报告、示意图、保存的图像和分镜图）。
- 结果文件夹的代码包含模型版本（`BundledToolkit.model_id()`：`MANIFEST.json` 的 SHA-256，没有时用文件指纹），换了模型
  不会重用旧模型的网络输出；`gimap_request.json` 记 `model_id` 和所选的 `scan_folder`。
- 物理路径不在相邻时间窗之间切换精修值和网络值（等级 P）；精修值留在表里。强度的周期性调制称“强度调制周期（可能来自样品台
  平移）”：沉积时样品台来回平移（防止光束损伤同一处），行程两端光斑可能到达样品边缘或移出样品；只凭数据不能证明来源，所以用问句。
- 光斑可能移出样品（工具包 `sputter_offsample.py`）：只凭强度判断——有调制时，下凹的核心是低于前后各 max(调制周期, 2 s) 内水平
  （相邻时间窗 ln 强度的 90% 分位数，取较低一侧）50% 的时间窗；没有调制时只有低于前后各 10 s 内水平 20% 的极深下凹才算核心。
  从核心向两侧延伸到仍低于各自水平 80% 的相邻时间窗，整个下凹都标出（只延伸有核心的下凹）；每侧要有至少 3 个有数据的时间窗，
  序列首尾不判断；不读样品台位置。这些时间窗保留数值，但不参与阶段投票，不进趋势线、来回变化计数、质量守恒、文献方法的 D 趋势和
  对网络的检验；页面把它们画成空心灰点、在结果行计数、在可靠性报告里说明（`physical.BEAM_OFF*`）。
- 文献对照的已发表数值都带引用键、DOI 和来源类型（正文/表、图读、计算、综述、摘要）；用户的速率只来自界面输入，不写进仓库。
- 默认解释：文献方法有数值时是它（`StructureFolderReader.available` 把 `literature` 排在 `physical` 前面），否则是网络的物理路径；
  文献方法的 R 按构造与文献一致，独立的检验是 D(δ)、渗流与已发表区间、网络的 2R/D 和质量守恒（“文献对照”页）。
