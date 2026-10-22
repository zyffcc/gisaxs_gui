# Compare 与阶段（Stages）

- **Status**: Current（2026-10-01 取代 Labs ▸ Classification）
- **Scope**: 一组曲线的异常帧与阶段（Analyze ▸ Series、Fitting ▸ In-situ series），以及多个序列 / 样品的比较（Compare 页）
- **Related code**:
  [`src/gimap/shared/series_stages/`](../../../src/gimap/shared/series_stages/)（纯 NumPy 内核，三处共用）、
  [`src/gimap/features/compare/`](../../../src/gimap/features/compare/)、
  `analyze/application/series_stages.py` + `analyze/presentation/bindings/series_stages.py`、
  `fitting/application/series_fit.py`（`stages_of_curves`、`next_start`）+ `fitting/presentation/single/series_stages.py`、
  `app/presentation/stage_text.py`（界面文字）、`app/presentation/components/row_groups.py`（热图上的阶段条）
- **Related tests**: `tests/test_series_stages.py`、`tests/test_release3_odd_frames.py`、`tests/test_series_stages_page.py`、
  `tests/test_compare.py`、`tests/test_fit_series.py::test_stages_and_odd_frames_steer_the_run`
- **Last verified**: 2026-10-22（合成 GISAXS 原位序列中一帧的热列：异常帧的尖峰量，对照代码）；2026-10-14（读 Analyze 文件夹的几种曲线、阶段导出的横轴：对照代码）；2026-10-01（合成序列，以及本地的
  原位 GIWAXS 序列；本地数据和从中得到的数值不进仓库）

## 为什么不是“分类”

原 Classification 页是通用机器学习工作台（原始像素特征、要标签、10 种分类器），和几何、掩膜、q 无关，
结果也不回到流程里。散射用户真正要问的是：这个序列什么时候变了、变得多快？哪些帧不该算？几个样品哪些
一样？这些都不需要标签，也不需要 AI。新功能一律用 Analyze 约化出的曲线（同一几何、掩膜、坏像素、切线）。

## 方法（`shared/series_stages`）

1. **可比较的曲线**：log10 I，只用（几乎）每帧都有数据的 q 点；默认去掉每帧的平均水平（只比形状），整体强度单独保存。
2. **异常帧先找**：一帧与它之前的 3 帧和之后的 3 帧（各取中位数）都不一致，且远超过该处帧间的变化。两种量，满足一种即可：
   - 整条曲线（RMS，两侧中较近的一侧）：超过该处帧间变化的 5 倍、超过典型帧间变化的 5 倍；
   - 尖峰：在差别最大的约 1 % 的点上（至少 3 点），与两侧的差别**方向相同**，并超过这些点在该处帧间变化的 8 倍。
     只看整条曲线时尖峰被稀释：1 % 的点上亮 25 倍的热列（log10 差 1.4）整条曲线的 RMS 只有约 0.14，在噪声较大的序列里
     不到一步帧间变化的 4 倍，以前因此漏掉，还被切成了一个阶段。“方向相同”把样品的真实变化排除在外：持续的变化与前后
     两侧的差别方向相反，突变与其中一侧一致。纯噪声在这些点上可达约 3.5 倍（高斯与低计数泊松、5–3200 帧实测），
     所以阈值取 8；10 倍的尖峰才让整条曲线移动约一步帧间变化。

   差别集中在约 1 % 的点上 → “多半是探测器”（建议在 Mask 步骤遮掉）。异常帧不再充当邻居：去掉已找到的异常帧再看它们的
   邻居（最多 3 遍），所以连续两三帧的伪影、序列一端的两帧都能找到（以前只标出中间或最外的一帧）；连续四帧以上
   （序列两端三帧以上）一样的是一个状态。序列两端只有一侧可比：开头或末尾一两帧里的变化分不出是样品还是伪影，
   照旧标为异常帧并写出原因。
   不先排除的话，一个异常帧会自己成为一个主成分，还会被切成一个假的阶段，并进入终态（最后 10 帧的平均）。
3. **主成分**：其余帧的 PCA（解释 95 % 方差，最多 10 个），所有帧投影；每个成分的正负号定为“从开始到结束上升”。
4. **阶段**：主成分得分按帧序做最优的分段直线拟合（动态规划，每段至少 5 帧，超过 1500 帧时按块平均）。
   每多一段，至少要多解释一段时剩余变化的 5 %，并且多于噪声能解释的（BIC 式惩罚，噪声取自二阶差分）。
   一个阶段是描述，不等于相：平滑的变化也会在“变化速度拐弯”处被切开；阶段之间“哪里增长、哪里减少”（相对曲线其余部分）
   才说明结构是否改变。
5. **进度**：主成分 1 完成一半 / 90 % 变化时的帧。

## Analyze ▸ Series

Build Map（或填满热图的 Batch Export）结束后在后台计算（几百帧、每帧约两千个点的序列约 1 秒）：

- 热图右侧彩色阶段条、阶段边界虚线、红色箭头标出异常帧（Marks ▸ Stages 可隐藏）；
- 控件下一行：**Stages** Auto (n) / 1…8，以及各阶段的帧范围、异常帧数；
- 折叠的 *What changes, and the odd frames*：各阶段的典型帧、每个边界处增长 / 减少最多的 q、到哪帧完成一半 / 90 %、
  比较的 q 范围、每个异常帧的原因；**Leave the odd frames out of Batch Export**；
- 右下图 *Change along the series*：主成分 1 对帧，按阶段着色，异常帧为红 ×；
- Export ▸ **Stages as Table…**：每帧的阶段、是否异常及原因、主成分得分、整体强度，旁边 JSON 记录（方法、各阶段、变化、异常帧）；
  变化和异常帧的原因用热图自己的横轴（χ、qy 或 q）表述；阶段寻找失败时清掉旧的阶段，导出给出失败的原因；
- **Send to Compare**：把这张热图加入 Compare。

## Fitting ▸ In-situ series

列出曲线后在后台读入全部曲线，按 Single analysis 的两半、范围和排除点比较：Curves 步骤显示阶段与异常帧，
帧列表按阶段着色、异常帧标 “odd”；**Leave out the odd frames**（默认开）；Start 多一个选项
**The previous result; the Single model at each new stage**；趋势图按阶段着色。

## Compare 页

- **Series**：Analyze 的 Series 热图（Send to Compare，或 Add Series ▸ The Series Map of Analyze）、一个曲线文件夹、或若干曲线文件
  （两列以上 q、I；`# columns:` 或表头写 nm⁻¹ 时换算成 Å⁻¹；按文件名中的编号排序）。可重命名（换行与多余空格合成一个空格，
  清空则恢复原名）、移除（Remove Selected 或 Delete 键）。名字很长时图、结果表和摘要只显示各序列不同的部分
  （`abc_xy_runA_10s_5hz_012_00001` 与 `abc_xy_runB_10s_5hz_020_00001` 显示为 `runA_012`、`runB_020`），
  Series 表、提示、CSV 与 JSON 保留完整的名字。没有序列时右侧是空状态说明；每次操作只有一个 toast。
  Analyze 写的文件夹（Batch Export 文件夹、它的 `curves/`、`gimap_analysis/`）里有好几种曲线：读 README 与文件名分出各种
  曲线（`compare/infrastructure/folders.py`），一种曲线是一个序列，有几种时问选哪一种（拟合用曲线叫 “the curve for Fitting
  (…)”，不属于 Analyze 导出的文件叫 “other files”）；序列名为 “<样品或批次> · <曲线键>”（如 `radial`、`other`）。批次的帧表格
  `<名称>_<曲线>_frames.*` 也能读（至少三行 x，每帧一列）。
  也可以把文件夹（每个文件夹一个序列）或曲线文件（合成一个序列）拖到页面上；其他文件（项目、探测器图像）交给主窗口按平常方式打开，
  混在一起拖入时说明哪些没有加入。同一个热图或文件夹不会重复加入。Remove All 立即清空，toast 上的 Undo 可把它们放回。
  所有表格都可以 Ctrl+C / 右键复制行（带列名）。
- **Compare**：所有序列都覆盖的 q 范围（可收窄，*Whole Range* 复原）、只比形状（默认）、终态取最后 N 帧（默认 10）。
  任何改动后约 0.3 s 自动重新比较（几个各有几百帧的序列需要几秒，后台）。范围框的上下限取自共同的 q 网格（向外取整到显示的四位小数），
  步长约为范围的 1 %。比较失败时（例如范围内没有共同的点）Results 说明原因和怎么改，不会停在 “Comparing …”。
- **Results**：分组（三个以上序列时，按终态 Ward 聚类，在合并距离跳变 ≥ 2 倍处切开，差别 < 2 % 时不分组）、
  差别最大的序列（终态与开始时各一句）、每个序列的帧数 / 异常帧 / 阶段 / 完成一半 / 完成 90 %、终态或开始时的差别矩阵（%）。
- 右侧：每个序列变化了多少（帧或在序列中的比例，可选成分；× 为异常帧，颜色同序列）、沿两个主要变化方向的路径
  （○ 起点、■ 终点；只有一个主要变化时显示说明而不是空坐标）、终态曲线。保存的图总用浅色配色（深色界面下也是），
  写失败时说明原因，成功后 toast 可打开文件夹。
- Save：每个序列的表 / 每一帧的表（CSV + JSON 记录）、三张图；项目（.gimap）保存设置与序列（曲线文件按路径，
  Analyze 热图存到 `<项目>.compare.npz`）。

## 读结果时要注意的两件事

这两条来自在真实原位序列上的使用，写成一般的规则；具体数值属于用户的数据，不记在这里。

- **时有时无的探测器伪峰会变成异常帧。** 某个 q 处的窄伪峰只在部分帧出现时，这些帧会被标为异常帧，原因写的是
  “差别集中在很少的点上，多半是探测器”。处理办法是在 Analyze 的 Mask 步骤遮掉那里再重建热图，而不是把这些帧当成
  样品的变化。真正属于样品的突变（例如某一帧里一个峰突然消失）也会被标为异常帧，原因里写的是哪个 q 处变了多少。
- **比较范围要避开探测器伪影和阴影区。** 在全 q 范围比较时，一个序列可能因为伪峰或阴影区看起来离群；把 Compare
  步骤的 q 范围收窄到干净的部分后，终态的差别可能小得多。这时再看“开始时”的差别矩阵：差别可能在起点，
  不在终态。下结论之前两张矩阵（终态、开始时）都要看。

## 手动验收

- [ ] 序列热图生成后出现阶段行与彩条；Auto / 手选阶段数即时切换；异常帧红箭头与原因一致；
- [ ] Batch Export 勾选跳过异常帧后少导出相应帧；Stages as Table 写出 CSV + JSON；
- [ ] Send to Compare 后切到 Compare，序列名可改，范围 / 只比形状 / 终态帧数改动后自动重算；
- [ ] Compare 的表格、三张图、Save 菜单、项目保存与重开；浅色 / 深色、中文。
- [ ] Add Series ▸ Folder of Curves… 选一个 Batch Export 文件夹：问选哪一种曲线（中文界面下全是中文），选后序列名为
      “<批次> · <曲线键>”；选 `fit_input/` 文件夹直接成为一个序列。
