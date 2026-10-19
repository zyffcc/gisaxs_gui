# SEM 粒径分析与预测对比

- **Status**: Current
- **Scope**: 独立 SEM 图像粒径提取、粒径 CSV 导入及可选的预测半径分布对比
- **Related code**: [`src/gimap/features/sem/`](../../../src/gimap/features/sem/)
- **Related tests**: [`test_sem_science.py`](../../../tests/test_sem_science.py)、
  [`test_sem_presentation.py`](../../../tests/test_sem_presentation.py)、
  [`test_sem_integration.py`](../../../tests/test_sem_integration.py)
- **Last verified**: 2026-10-19

## 打开工具

**Tools → SEM Particle Analysis…（工具 → SEM 粒径分析…）** 打开独立窗口。可以只分析 SEM
图像或已测量的粒径 CSV；这条流程不需要模型文件、TensorFlow 或预测结果。

在 **Labs → 2D Prediction（实验室 → 二维预测）** 完成预测后，点击结果区的
**SEM comparison…（SEM 对比…）**，再选择 SEM 图像或粒径 CSV。当前结果具有明确的 nm
半径轴时，会带入该次预测的半径分布及输入、模型、预处理和运行时来源。带入的数据是复制的
快照；之后切换预测页的输入不会改写已经打开的对比。预测页更换输入、模块、模型或预处理后，
旧结果不会被当作当前预测再次带入。

**Remove prediction（移除预测）** 保留 SEM 测量结果，重新生成独立的 SEM 分布。从 Tools
菜单打开这个工具时，也会移除复用窗口中原先附带的预测。

## 从图像提取

1. 点击 **Open SEM image…（打开 SEM 图像…）**，选择 TIFF、PNG、JPEG 或 BMP。
2. 在 **Image calibration（图像标尺）** 中明确输入 `nm per pixel`；也可以选择
   **Scale-bar length / pixels（标尺长度 / 像素长度）**，填写已知标尺的物理长度和像素长度。
   换算为 `nm_per_pixel = 标尺长度_nm / 标尺长度_px`。软件不自动读取图中文字、倍率或图片 DPI
   来猜测标尺。切换到另一来源文件时，标尺数值清空，必须按该来源重新设置；未设置正值时不能提取。
3. 在 **Segmentation and image ROI（分割与图像 ROI）** 中，用 ROI 排除标尺、文字、边框和
   不属于颗粒的区域。ROI 使用原图像素边界坐标 `(left, top, right, bottom)`；第零行在上，
   像素中心位于 `(column + 0.5, row + 0.5)`。
4. 选择亮颗粒或暗颗粒，设置阈值与筛选条件，点击 **Extract particles（提取颗粒）**。
5. 查看 **SEM image / outlines（SEM 图像 / 轮廓）** 和 **Particle measurements（颗粒测量）**，
   检查哪些连通区域被保留、哪些被拒绝及其理由，再使用分布或导出结果。

### 选择检测方法

**Particle detection（颗粒检测）** 提供三种方法。默认保留 **Threshold / watershed**；选择
**StarDist (local AI)** 使用本地实例分割，或用 **Existing instance labels** 测量已有分割标签。
切换方法会使旧测量失效；不会在 AI 失败时自动改用另一种算法。

- **StarDist（本地 AI）**：指定装有 StarDist/TensorFlow 的本地 Python，以及包含 `config.json`
  和 `weights_best.h5`（或 `weights_last.h5`）的本地模型文件夹。资源检查只查看本地文件；
  缺资源时不能提取，旁边的 Browse 可选择路径。推理在独立 CPU 子进程中运行，主 GUI 不导入
  TensorFlow 或 StarDist。图像不上传，推理不会联网下载权重。
- **阈值 / 分水岭**：继续使用下面列出的存储强度阈值、亮暗选择和距离分水岭，不需要 AI 运行时。
- **已有实例标签**：先打开对应 SEM 图像，再选择整数 TIFF/PNG 标签文件。零为背景，每个颗粒有
  不同的正整数 ID；RGB 伪彩色图、浮点标签和多页文件不会被当作实例标签。尺寸可与 SEM 全图一致，
  也可恰好与明确选择的 ROI 一致。全图标签会显式应用 ROI，ROI 大小的标签放到指定原点；
  原 ID 保留，相邻 ID 不重新合并。跨 ROI 的对象仍按共同边界筛选设置处理，裁剪 ID 写入记录。
  这条路径可测量外部 StarDist 或人工修正的标签，但不自动确认其与当前 SEM 图来自同一来源。

本地默认搜索目录为 `~/.cache/gimap/sem/stardist/`（也可通过 `GIMAP_SEM_STARDIST_HOME`
设置）：Python 位于 `runtime/bin/python`（Windows 为 `runtime/Scripts/python.exe`），默认模型
位于 `models/2D_versatile_fluo/`。也可以在界面选择其他解释器和自有模型。运行时是可选依赖，
不要求改动主 GUI 环境；本轮验证组合为 StarDist 0.9.2、CSBDeep 0.8.2、TensorFlow 2.15.1。
可用该独立环境的 Python 执行 `python -m pip install -r requirements-sem-stardist.txt`；
清单只安装运行时包，不包含或下载模型权重。主程序的 `requirements.txt` 不要求安装 StarDist。

官方 `2D_versatile_fluo` 权重来自荧光细胞核训练，**不是经过 SEM 专门验证的模型**。适合程度需
通过真实图像轮廓和标注评估，检测数变多不等于准确率变高；自有 SEM 模型可通过本地文件夹加载。
来源与许可证见 [StarDist](https://github.com/stardist/stardist) 和
[官方预训练权重仓库](https://github.com/stardist/stardist-models)。

**Instance AI options（实例 AI 选项）** 可设置模型输入缩放、概率阈值、NMS 重叠阈值及归一化
分位数。概率/NMS 显示 **Model default** 时采用模型保存的实际阈值；采用值仍会写入 JSON。
归一化仅在选定 ROI 内进行；暗颗粒先反转灰度方向，再按分位数归一化。非有限像素用于推理时由
有限像素中位数填补，最终标签在这些像素上置零，过程会记录。它们不参与模型归一化分位数。

AI 输入缩放和物理标尺是两件事：StarDist 将预测轮廓恢复到原 ROI 像素网格，再放回全图。
测量面积始终是这些原网格标签的像素数；不会再按 AI 缩放系数修改面积或半径，也不会对 AI 标签
再次应用 Otsu 或连通域。标尺、ROI、面积/圆度/边界筛选仍在三种方法间共用。

AI 导出记录包括原网格标签、实际归一化/亮暗/缩放/分块参数、概率/NMS、版本、执行设备以及模型
配置/阈值/权重的 SHA-256；可用的模型来源说明也会导出。记录只引用并校验模型，不复制模型权重。

灰度图保留存储强度用于分割；彩色图按记录的 RGB 亮度公式转成灰度。屏幕和轮廓图的强度拉伸只
帮助看清图像，不改变阈值输入、面积或测量值。

### 测量量与分割设置

提取量为**二维投影的面积等效圆半径**：

```text
A_nm² = A_px × nm_per_pixel²
R_nm  = sqrt(A_px / pi) × nm_per_pixel
D_nm  = 2 × R_nm
```

其中面积是分割区域包含的像素数。界面分布横轴为半径 `R`，粒子表同时给出半径与直径。
非圆形颗粒也按同面积圆换算，因此比较和导入时必须保持半径、直径的定义一致。

| 设置 | 实际作用与审查重点 |
| --- | --- |
| Bright / Dark particles | 选择颗粒相对背景更亮或更暗；亮颗粒取高于阈值的像素，暗颗粒取低于或等于阈值的像素。 |
| Otsu threshold | 在当前 ROI 的有限灰度值上自动计算阈值；应在轮廓图中检查背景与颗粒是否被正确分开。 |
| Manual threshold | 按图像存储强度明确指定阈值，便于检查和复现。 |
| Split touching particles (watershed) | 可选的距离变换标记分水岭；峰间距会影响粘连区域被分成几个对象，需检查过分割和漏分割。 |
| Minimum / Maximum area | 用像素面积筛选候选区域；可排除小噪声或大块背景，也可能排除真实颗粒。 |
| Minimum circularity | 按记录的周长算法及 `4 pi A / perimeter²` 筛选，表中保留圆度值。 |
| Exclude particles touching the ROI boundary | 可排除被 ROI 截断的颗粒；被排除的对象及理由仍可审查。 |

分割效果取决于对比度、背景、粘连、形状与标尺设置。轮廓图和粒子表用于判断当前参数是否适合这张
图像；自动阈值或分水岭成功运行，并不代表每个 SEM 图像都已准确分割。所有对象都被筛掉时，仍会
保留候选粒子、拒绝理由与轮廓，允许导出审查资料；此时没有粒径分布图。

## 导入已经测量的 CSV

点击 **Open sizes CSV…（打开粒径 CSV…）**，在 **CSV measurement settings（CSV 测量设置）**
中明确选择：

- 数值所在的列，以及数据是 **Radius（半径）** 还是 **Diameter（直径）**；
- 原始单位：`nm`、`µm` 或 `pixels`；像素单位还需要该来源的显式 `nm per pixel` 标尺；
- 分隔符和表头前跳过的行数。TSV 可使用制表符 `\t`；无表头数据按 `Column 1` 等列名选择。

读取列名后不会自动选择测量列。源文件的列名属于数据，中英文切换不会翻译或改写它们。点击
**Import particle sizes（导入粒径）** 后，直径按除以二转换成半径，微米按单位换算成纳米；原列、
数量定义、单位和换算因子写入导出记录。缺失、非数值、非有限或非正值的行会被拒绝，理由可在表中查看。

## 分布与可选的预测对比

默认 **Probability (same bins)（概率，同一分箱）**：SEM 半径与预测使用相同的半径范围和分箱边界。
SEM 按颗粒数形成概率质量；预测按原轴中点形成的分箱边界，将各箱内部视作均匀的概率质量重新分箱。
两者分别对选定范围内保留的计数或概率质量归一化到总和为一。范围外的 SEM 颗粒数和预测概率质量
会在界面说明和 JSON 中记录。

**SEM radius scale（SEM 半径缩放系数）** 默认是一。需要额外缩放时应显式设置；这个系数只作用于
比较用的 SEM 半径，原始测量半径仍保留在粒子 CSV 中。修改分箱、范围、缩放或显示模式后，点击
**Update distribution（更新分布）** 重新计算，再导出。

可选的 **Paper display (independent peak scaling)（论文显示，独立峰高缩放）** 使用六十个 SEM 显示
分箱；预测使用其原半径轴，概率归一化后扣除最小值，再按公共分箱 SEM 分布的峰高缩放。该模式用于
明确要求的显示方式，曲线幅度不再能直接作为同一分箱下的概率比较。设置、缩放因子、扣最小值及
显示数组都会导出；不做半径轴平移，也不自动套用某次实验的缩放设置。

## 导出与来源

点击 **Export SEM results…（导出 SEM 结果…）**，选择保存文件夹。软件在其中建立新的
`SEM_analysis_<时间>/` 子目录；遇到已有的同名结果文件会拒绝覆盖。源图像和源 CSV 保持只读，
完成后提示可打开输出文件夹。

| 文件 | 内容 |
| --- | --- |
| `sem_distribution.png`、`.pdf`、`.svg` | 浅色分布图；有附带预测时包含对比曲线。 |
| `sem_particles.csv` | 每个对象的接受状态、半径、直径，以及图像提取时的面积、质心、圆度和拒绝理由。 |
| `sem_distribution.csv` | 公共分箱边界、SEM 概率质量和可选的预测概率质量。 |
| `sem_display_arrays.npz` | 实际用于显示的分箱和曲线数组，以及原始测量半径。 |
| `sem_overlay.png` / `.pdf` / `.svg` | 图像分割轮廓、接受与拒绝对象及 ROI，供人工审查。 |
| `sem_segmentation.npz` | 提取所用的原分辨率灰度图像、标签数组和轮廓 RGBA；灰度强度未应用显示拉伸。 |
| `sem_instance_labels.tif` | 原分辨率 int32 实例标签，保留接受及拒绝对象的原 ID，可直接在“已有实例标签”模式重用。 |
| `sem_analysis.json` | 来源路径、SHA-256、读取方式、标尺、ROI、阈值、筛选条件、数量统计、比较与归一化设置、可选预测来源及输出清单。 |

CSV 的 `#` 头行也写入来源、测量和比较设置。只有图像提取产生分割图与标签；CSV 输入不生成图像
轮廓。没有接受的粒子时，仅导出可用的粒子表、来源记录及图像分割审查资料。

## 取消、换源与退出

读取、提取、更新分布与导出在后台执行。**Cancel（取消）** 会使当前结果失效，晚到的回调不会覆盖
新来源或新设置；提取会在检查点响应停止。底层读写或单次数值计算可能需要先完成当前阶段，已经
开始的文件写入也可能继续完成，取消不会删除已写出的文件。

StarDist 取消会终止并回收本地模型子进程；替换请求串行等待旧子进程停止，不同时加载多个重模型。
运行超时、资源缺失或模型文件在推理过程中改变会报错，不输出伪装为成功的回退测量。

切换来源或修改测量设置后，旧测量不再可导出；修改比较设置后需先更新分布。关闭忙碌的工具窗口会
取消待显示的结果并等待后台结束；在结束前重新打开，窗口保持可用，不会被旧任务再次关闭。
退出 GIMaP 的任务确认列表包含 SEM。若确认退出后后台任务仍未停止，主窗口会保留并提示稍后重试，
避免先释放工作线程依赖的窗口和信号。

## 开发调用链与检查

```text
Tools 菜单 / Prediction 的 SEM comparison 信号
    → app.menus.ApplicationMenus.open_sem
    → sem.bootstrap.create_sem_window / create_sem_service
    → presentation.SemWindow + SemViewModel
    → application.SemAnalysisService
    → domain.segmentation / domain.comparison
      或 application.SemInstanceSegmenter → local subprocess / existing labels
         → domain.measure_labels（原像素网格统一测量）
    → infrastructure.readers / exporter / figures
```

Prediction 到 SEM 的连接由组合根注入，两个 feature 不相互导入。SEM 的 service 和数值逻辑独立于
TensorFlow；仅附带已有预测数组时才生成对比。静态布局在
[`sem_window_view.py`](../../../src/gimap/features/sem/presentation/views/sem_window_view.py)，
后台任务使用共用 `TaskRunner`，窗口退出前等待工作结束并释放绘图与主题连接。

Focused tests 使用合成图像、合成 CSV 和临时用户目录，覆盖面积及单位、阈值/分水岭与筛选、归一化与
导出来源、独立 GUI、预测快照、过时任务、语言切换、取消及退出边界：

```sh
QT_QPA_PLATFORM=offscreen python -m pytest tests/test_sem_science.py tests/test_sem_advanced.py tests/test_sem_presentation.py tests/test_sem_integration.py
```
