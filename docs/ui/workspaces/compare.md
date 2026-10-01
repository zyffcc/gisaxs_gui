# Compare 与阶段（Stages）

- **Status**: Current（2026-10-01 取代 Labs ▸ Classification）
- **Scope**: 一组曲线的异常帧与阶段（Analyze ▸ Series、Fitting ▸ In-situ series），以及多个序列 / 样品的比较（Compare 页）
- **Related code**:
  [`src/gimap/shared/series_stages/`](../../../src/gimap/shared/series_stages/)（纯 NumPy 内核，三处共用）、
  [`src/gimap/features/compare/`](../../../src/gimap/features/compare/)、
  `analyze/application/series_stages.py` + `analyze/presentation/bindings/series_stages.py`、
  `fitting/application/series_fit.py`（`stages_of_curves`、`next_start`）+ `fitting/presentation/single/series_stages.py`、
  `app/presentation/stage_text.py`（界面文字）、`app/presentation/components/row_groups.py`（热图上的阶段条）
- **Related tests**: `tests/test_series_stages.py`、`tests/test_series_stages_page.py`、`tests/test_compare.py`、
  `tests/test_fit_series.py::test_stages_and_odd_frames_steer_the_run`
- **Last verified**: 2026-10-01（合成序列 + Yuxin 四个原位 GIWAXS 样品，本地数据不进仓库）

## 为什么不是“分类”

原 Classification 页是通用机器学习工作台（原始像素特征、要标签、10 种分类器），和几何、掩膜、q 无关，
结果也不回到流程里。散射用户真正要问的是：这个序列什么时候变了、变得多快？哪些帧不该算？几个样品哪些
一样？这些都不需要标签，也不需要 AI。新功能一律用 Analyze 约化出的曲线（同一几何、掩膜、坏像素、切线）。

## 方法（`shared/series_stages`）

1. **可比较的曲线**：log10 I，只用（几乎）每帧都有数据的 q 点；默认去掉每帧的平均水平（只比形状），整体强度单独保存。
2. **异常帧先找**：一帧与它之前的 3 帧和之后的 3 帧都不一致（两侧中较近的一侧的距离），且超过该处帧间变化的 5 倍、
   超过典型帧间变化的 5 倍。差别集中在约 1 % 的点上 → “多半是探测器”（建议在 Mask 步骤遮掉）。不先排除的话，
   一个异常帧会自己成为一个主成分，还会被切成一个假的阶段。
3. **主成分**：其余帧的 PCA（解释 95 % 方差，最多 10 个），所有帧投影；每个成分的正负号定为“从开始到结束上升”。
4. **阶段**：主成分得分按帧序做最优的分段直线拟合（动态规划，每段至少 5 帧，超过 1500 帧时按块平均）。
   每多一段，至少要多解释一段时剩余变化的 5 %，并且多于噪声能解释的（BIC 式惩罚，噪声取自二阶差分）。
   一个阶段是描述，不等于相：平滑的变化也会在“变化速度拐弯”处被切开；阶段之间“哪里增长、哪里减少”（相对曲线其余部分）
   才说明结构是否改变。
5. **进度**：主成分 1 完成一半 / 90 % 变化时的帧。

## Analyze ▸ Series

Build Map（或填满热图的 Batch Export）结束后在后台计算（403 帧 × 2175 点约 1 秒）：

- 热图右侧彩色阶段条、阶段边界虚线、红色箭头标出异常帧（Marks ▸ Stages 可隐藏）；
- 控件下一行：**Stages** Auto (n) / 1…8，以及各阶段的帧范围、异常帧数；
- 折叠的 *What changes, and the odd frames*：各阶段的典型帧、每个边界处增长 / 减少最多的 q、到哪帧完成一半 / 90 %、
  比较的 q 范围、每个异常帧的原因；**Leave the odd frames out of Batch Export**；
- 右下图 *Change along the series*：主成分 1 对帧，按阶段着色，异常帧为红 ×；
- Export ▸ **Stages as Table…**：每帧的阶段、是否异常及原因、主成分得分、整体强度，旁边 JSON 记录（方法、各阶段、变化、异常帧）；
- **Send to Compare**：把这张热图加入 Compare。

## Fitting ▸ In-situ series

列出曲线后在后台读入全部曲线，按 Single analysis 的两半、范围和排除点比较：Curves 步骤显示阶段与异常帧，
帧列表按阶段着色、异常帧标 “odd”；**Leave out the odd frames**（默认开）；Start 多一个选项
**The previous result; the Single model at each new stage**；趋势图按阶段着色。

## Compare 页

- **Series**：Analyze 的 Series 热图（Send to Compare，或 Add Series ▸ The Series Map of Analyze）、一个曲线文件夹、或若干曲线文件
  （两列以上 q、I；`# columns:` 或表头写 nm⁻¹ 时换算成 Å⁻¹；按文件名中的编号排序）。可重命名、移除。
- **Compare**：所有序列都覆盖的 q 范围（可收窄，*Whole Range* 复原）、只比形状（默认）、终态取最后 N 帧（默认 10）。
  任何改动后约 0.3 s 自动重新比较（四个样品约 4 s，后台）。
- **Results**：分组（三个以上序列时，按终态 Ward 聚类，在合并距离跳变 ≥ 2 倍处切开，差别 < 2 % 时不分组）、
  差别最大的序列（终态与开始时各一句）、每个序列的帧数 / 异常帧 / 阶段 / 完成一半 / 完成 90 %、终态或开始时的差别矩阵（%）。
- 右侧：每个序列变化了多少（帧或在序列中的比例，可选成分）、沿两个主要变化方向的路径、终态曲线。
- Save：每个序列的表 / 每一帧的表（CSV + JSON 记录）、三张图；项目（.gimap）保存设置与序列（曲线文件按路径，
  Analyze 热图存到 `<项目>.compare.npz`）。

## Yuxin 数据上的读法（2026-10-01，本地验证）

- 样品 116：异常帧 169、176（q≈4.87 的探测器伪峰时有时无）和 403（2.95 Å⁻¹ 峰掉 99 %）；一个主成分占 97 %；
  3 个阶段 1–52 / 53–170 / 171–403，2.95 Å⁻¹ 峰在第二段出现、第三段变强。
- 四个样品全 q 范围比较时 120 看似离群（终态差 72–114 %），但把 q 限在 1.65–4.25（去掉 4.3 以上的阴影区和 4.89 伪峰）
  后终态只差 3–5 %，**开始时** 120 差 18–25 %：差别在起点，不在终态。比较范围要避开探测器伪影。

## 手动验收

- [ ] 序列热图生成后出现阶段行与彩条；Auto / 手选阶段数即时切换；异常帧红箭头与原因一致；
- [ ] Batch Export 勾选跳过异常帧后少导出相应帧；Stages as Table 写出 CSV + JSON；
- [ ] Send to Compare 后切到 Compare，序列名可改，范围 / 只比形状 / 终态帧数改动后自动重算；
- [ ] Compare 的表格、三张图、Save 菜单、项目保存与重开；浅色 / 深色、中文。
