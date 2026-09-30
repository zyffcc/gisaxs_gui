# 探测器几何与 q 映射

> **Status**：Current（阶段 1：统一底座）
>
> **Scope**：像素 → q 的唯一实现、坐标约定、各页面历史约定与已知偏差
>
> **Last verified**：2026-09-28

像素到 q 的换算现在只有一份实现：`src/gimap/shared/geometry`。Analyze（GISAXS 与 GIWAXS，
包括原 WAXS 页面与原 Cut & Fitting 探测器部分的全部功能）直接使用下面的规范约定和精确公式；
Trainset 仍按自己的历史像素约定调用它，输出与迁移前逐像素一致（见“等价门槛”）。WAXS 页面与
Cut & Fitting 的探测器部分已下线：Fitting 只接收 Analyze 写出的曲线（q 已是精确公式的 Å⁻¹），
它们的中心约定与近似模型不再有调用方，相关代码已删除。

## 规范约定

`DetectorGeometry` 描述一块垂直于直射束的平板探测器和掠入射角：

| 量 | 约定 |
|---|---|
| 像素位置 | 像素 `(行 i, 列 j)` 覆盖连续区域 `[j, j+1] × [i, i+1]`，中心在 `(j+0.5, i+0.5)`；与 pyFAI 及 `imshow(extent=(0, W, H, 0))` 相同 |
| 行方向 | 第 0 行在图像顶部，y 向下增大 |
| 束流中心 | `beam_center_x_px / beam_center_y_px` 是**直射束**在上述坐标中的位置（pyFAI 的 PONI，以像素计） |
| 掠入射角 | `incidence_deg`，0 表示透射几何 |
| 单位 | 长度 m，波长 Å，q 为 Å⁻¹ |
| qy 符号 | 与探测器列方向同号：图像中直射束右侧的像素 qy > 0，左侧 qy < 0 |
| 带符号 q∥ | `√(qx² + qy²)` 取 qy 的符号（`signed_parallel`） |

`grazing_q_map(shape, geometry)` 用精确公式给出样品坐标系中的 `qx, qy, qz` 和带符号的
`q∥`：散射方向 `(D, X, Y)/R` 按样品倾角 αi 旋转后得到出射方向，`q = k_f − k_i`。
`transmission_map` 给出透射 SAXS/WAXS 的 `|q|`、2θ 和方位角 χ（0° 向右，90° 向上）。
`DetectorGeometry.horizon_row()`、`specular_row()`、`row_for_exit_angle()` 给出样品水平线、
镜面反射和任一出射角（如 Yoneda）在束流中心列上的行位置。

## 模块

| 模块 | 职责 |
|---|---|
| `shared/geometry/detector_geometry.py` | `DetectorGeometry`：校验、单位、裁剪、特征行位置 |
| `shared/geometry/q_mapping.py` | 位移 → q 的物理计算；`ExitAngleModel` 列出精确公式和三种历史近似；`grazing_q_region` / `region_displacements` 只计算一个像素矩形（大探测器按块计算） |
| `shared/geometry/legacy_conventions.py` | Trainset 的历史像素约定，物理部分调用上面的实现 |
| `shared/geometry/conventions.py` | 已存数据中的束流中心约定（numpy 下标；旧 Cut & Fitting 的“行从底部数”）↔ 规范约定 |
| `shared/geometry/instrument_profile.py` | 仪器配置：命名的几何 + 探测器名称/尺寸，用于加载文件时自动匹配 |
| `integrations/state/instrument_profiles.py` | 仪器配置的 JSON 存储（原子写入、带 schema 版本） |

## 各页面的历史约定

| 页面 | 像素位置 | 束流中心 | 出射角模型 | 单位 |
|---|---|---|---|---|
| Cut & Fitting（已下线） | 像素中心，但按端点网格 `j·W·p/(W−1)` 排列（间距拉伸 W/(W−1)，远端最多偏 1 像素） | 0 基像素中心下标，**行从图像底部数** | `HORIZON_SHIFT`：αf = atan((Y − D·tan αi)/D)，2θf = atan(X/D) | mm、nm、nm⁻¹ |
| Trainset | 像素中心 | 0 基像素中心下标，行从顶部数 | `SUBTRACT_INCIDENCE`：αf = atan(Y/D) − αi | mm、nm、nm⁻¹ |
| Calibration | 像素中心 | 0 基像素中心下标，行从顶部数 | 透射环 | m、Å |

## 等价门槛

`tests/geometry_legacy_reference.py` 冻结了迁移前 Trainset 实现的原样副本（Fitting 的副本随
其探测器部分一起删除）。
`tests/test_shared_geometry_equivalence.py` 用随机几何（含整数中心、0° 入射、单列/两行图像）
和完整 Pilatus 2M 帧比较新旧结果：差异只来自浮点结合顺序，上限为该图 max|q| 的 1e-12，
恰好为 0 的像素保持为 0，符号一致。

## 历史近似与精确公式的差距

同一几何下，用历史模型与精确模型计算整幅探测器 q，取最大偏差（未计入上表中的像素位置差异）：

| 场景 | 模型 | max \|Δqz\| | 约合像素 |
|---|---|---|---|
| GISAXS：Pilatus 2M，172 µm，D = 4.2 m，λ = 1.0332 Å，αi = 0.4° | `HORIZON_SHIFT`（Fitting） | 2.6e-4 Å⁻¹ | 1.0 |
| | `SUBTRACT_INCIDENCE`（Trainset） | 1.4e-4 Å⁻¹ | 0.6 |
| | `NO_INCIDENCE_OFFSET`（WAXS） | 4.3e-2 Å⁻¹ | 171 |
| GIWAXS：55 µm，D = 0.2 m，λ = 0.6888 Å，αi = 0.15° | `HORIZON_SHIFT` | 6.9e-2 Å⁻¹ | 29 |
| | `SUBTRACT_INCIDENCE` | 6.6e-2 Å⁻¹ | 28 |
| | `NO_INCIDENCE_OFFSET` | 2.4e-2 Å⁻¹ | 10 |

小角 GISAXS 下两种平板近似误差在 1 个像素以内；广角 GIWAXS 下探测器边缘误差可达数十个
像素，因此 GIWAXS 必须使用精确模型（Analyze 只用精确模型）。Fitting 现在拟合 Analyze 的曲线，
GISAXS 拟合因此也改用精确 q；与旧 `HORIZON_SHIFT` 相差不到 1 个像素，所以没有保留“旧 q 公式”开关。`NO_INCIDENCE_OFFSET` 行是已下线
WAXS 页面的模型：它把输入的中心当作样品水平线，而用户填的是直射束位置，qz 因此整体偏高 k·sin αi。

## 标定结果保存在哪里

应用标定结果时写入：仪器配置（见下文，Analyze 使用的唯一几何来源）、`detector.*`（最近一次
标定，numpy 下标约定，仅用作下次标定的距离初值）、`beam.wavelength / energy_kev` 与
`system.geometry_calibration` 记录。标定**不再**写旧 Cut & Fitting 的 `fitting.detector.*`
（该页已下线，内置默认值也已删除）。“结果与现有几何差别很大，是否覆盖？”只在会覆盖一个已保存
的同名仪器配置、且中心移动 > 10 px 或距离变化 > 5 % 时询问（`profile_change_is_significant`）。

旧版保存的 `fitting.detector.*`（行从底部数；`fitting.gisaxs_input.flip_ud` 打开时分析图像已上下
翻转、行号即 numpy 下标）只在 Analyze 的 “Use Previous Geometry…” 中读取一次并换算成仪器配置
（`geometry_from_fitting_settings`，`conventions.canonical_from_fitting_center`）。

## CBF 头中的几何值

CBF 头里的 `Wavelength`、`Detector_distance`、`Beam_xy` 只有在线站把这些值传给探测器服务器时
才会填写，可能是过期值。`shared/detector_io/metadata.py` 把它们放在单独的键里：

| 键 | 内容 |
|---|---|
| `header_wavelength_angstrom` | 波长，Å；头中必须带明确单位（A、Å、angstrom 或 nm），无单位或未知单位为 `None` |
| `header_distance_m` | 样品–探测器距离，m；单位 m 或 mm，其他为 `None` |
| `header_beam_xy_px` | `(x, y)`，按文件原样，文件自身的像素坐标 |
| `header_beam_center` | 换算到规范坐标的头中心（CBF：`Beam_xy` 原样，按像素角为原点；单模块 NeXus：按加载时的转置与翻转换算；多模块拼接：无） |

这些值**不覆盖**主字段（`energy_kev`、`wavelength_angstrom`、`distance_m`、
`beam_center_x_px/y_px`）：主字段决定伴随 NXS 能量查找是否触发，也是标定的中心种子。
Analyze 只在 设置 ▸ Analyze ▸ “Use the beam centre written in the file header” 打开时使用
`header_beam_center`（默认关）；否则头中心只作为束流中心菜单里的一个选项。

## 仪器配置（Instrument profile）

- 存储：用户数据目录中的 `instrument_profiles.json`（`%APPDATA%\GIMaP`，或 `GIMAP_HOME`；
  旧的 `config/instrument_profiles.json` 首次启动时导入一次；独立的旧 dialog 上下文只在内存中保存）。每个配置 = 名称 + 规范 `DetectorGeometry` + 探测器名 + 帧尺寸。
- 标定写入：应用标定结果时，`RecordInstrumentProfile` 以 “探测器名 行×列”（如
  `PILATUS 2M 1679×1475`）创建或更新配置；中心按 numpy 下标 +0.5 换到规范坐标，距离 mm → m。
  标定是透射几何，不知道 αi：更新已有配置时保留它原来的 αi，新配置为 0。探测器倾斜
  （`detector_rotation_deg`）不在 `DetectorGeometry` 中建模，超过 0.05° 时记在 `source` 里。
- 读取匹配：Analyze 按探测器名（前缀匹配，忽略大小写与多余空格）和帧尺寸挑选配置；尺寸不同的
  配置不会被自动选中。也可以在 Analyze 顶栏手动指定配置、输入几何；旧版 Cut & Fitting 保存过
  几何时，提示条的 Use Previous Geometry… 可把它（行号从底部数，“上下翻转”时不换算）保存为配置。

## 已决定（2026-09-28）

1. WAXS / GIWAXS 的中心是**直射束**；GIWAXS 只用精确模型（Analyze）。
2. Trainset 保留历史近似（GISAXS 下 1 像素内）。Fitting 的探测器部分已并入 Analyze，拟合曲线的
   q 来自精确模型；因差距在 1 像素内，不再提供“旧 q 公式”开关。
3. 文件头中的束流中心默认**不**使用；设置中可打开（NeXus 按加载变换换算）。
4. P03 的 CBF 头里没有 `Beam_xy`；其他线站的 `Beam_xy` 按 (列, 行)、像素角为原点读取（pyFAI 的读法），
   Analyze 在图上显示该中心，用户可以直接看出是否合理并一键改回配置中心。
