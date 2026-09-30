# Analyze feature

“打开即出结果”的统一工作区（重构阶段 2）。打开或拖入 CBF / NXS / TIFF（文件或文件夹）后：

1. `shared/detector_io` 读取图像与元数据（多模块 NXS 系列只列一次，按模块拼接）；
2. 按探测器名 + 帧尺寸匹配仪器配置（`InstrumentProfile`，标定结果会自动写入）；
   没有匹配时显示提示条：手动输入 / 标定一次（旧版 Cut & Fitting 保存过几何时另有“沿用旧几何”）；
3. 由探测器上最大散射角判断 GISAXS（≤ 20°）或 GIWAXS；可在顶栏强制模式；
4. GISAXS：在镜面反射杆两侧的列带里找 Yoneda 行，给出该行的水平 I(qy) 和束流中心列的
   竖直 I(qz)（只取样品水平线以上）；GIWAXS：I(q)、面内 / 面外扇区 I(q)、最强环的 I(χ)
   和规则网格上的 q∥–qz 图；
5. 结果曲线、叠加在探测器图上的切带都可拖动调整；状态行代替弹窗；一键导出 CSV + JSON；
6. Send to Fitting / Send Series to Fitting 写出 `_fit_input.dat`（原生列的 I、σ、像素数与
   observation）并交给 Fitting；可选 Sum N frames、gap guard、Refine x by Symmetry。

## 分层

| 层 | 内容 |
|---|---|
| `domain/` | 纯 numpy：有效像素、校正（背景、gap guard、帧相加）、原生列 / 分箱均值（Poisson 误差）、Yoneda 查找、切线、扇区、I(χ)、对称中心、分类 |
| `application/` | `AnalyzeFrame`（缓存 GIWAXS q 图）、`ResolveGeometry`、`ExportAnalysis`、`SaveInstrumentProfile`、ports |
| `infrastructure/adapters/` | `DetectorIoFrameSource`、`CsvCurveWriter`（含 `_fit_input.dat`）、`MatplotlibFigureWriter` |
| `presentation/` | `AnalyzeViewModel`（无 QWidget）、`AnalyzePage`、`GeometryDialog`、静态布局 `views/` |

## 约定

- 几何全部使用规范约定（`docs/architecture/geometry.md`）：像素 `(i, j)` 覆盖 `[j, j+1] × [i, i+1]`，
  第 0 行在顶部，束流中心是直射束；q 用精确掠入射公式，单位 Å⁻¹。
- χ = atan2(q∥, qz)：0° 沿表面法线（面外），±90° 在样品面内；样品水平线以下（αf < 0）的像素不参与。
- 导出的 CSV 以 `#` 注释行开头，列为 `x, I, sigma, pixels`；同名 `_analysis.json` 记录来源文件、
  几何、仪器配置、切带位置和提示信息。
- 大探测器（Lambda 9M 拼接后约 1.5×10⁷ 像素）的 q 图按行分块计算并以 float32 缓存，同一几何的
  后续帧直接复用。
