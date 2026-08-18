# WAXS / GIWAXS 渐进迁移地图

## 运行入口与 legacy 关系

- 主 GUI 入口是 feature-owned `src/gimap/features/waxs/presentation/page.py`，由 app composition root 嵌入 stacked widget。
- 历史 `WAXS/WAXS.py` 已从 3740 行收缩为薄兼容启动器；它和主 GUI 都装配同一个 feature page。
- `controllers/waxs_controller.py` 为空，不应重新发展成第二个 orchestration layer。目标入口是 View → ViewModel → use cases。

## 职责地图

| 顺序 | 职责 | 当前实现 | 输入/输出 | 当前外部依赖 | 目标位置 |
| --- | --- | --- | --- | --- | --- |
| 1 | 文件加载与 frame metadata | `ImageLoadWorker`、`detect_nxs_frame_count`、`load_image_matrix` | path/frame → float32 image/frame count | Qt thread、calibration loader、HDF5 | `LoadWaxsImage` + `WaxsImageRepository` adapter |
| 2 | 几何与 q map | `compute_q_maps`、`_cut_image_by_q_range` | shape + geometry → qr/qz/cut extent | NumPy | `domain.geometry` |
| 3 | mask/display 数值 | `prepare_display_array`、`estimate_display_limits`、`percentile_limits`、`_mask_limits` | image + thresholds/log → masked image/limits | NumPy | `domain.masking` |
| 4 | 1D integration/cuts | `integrate_image`、`line_cut_profile`、`circle_cut_profile`、angle/smoothing helpers | image + geometry/selection → x/y | NumPy | `domain.integration` + `IntegrateWaxsImage` |
| 5 | batch/in-situ | `BatchWorker.run`、`start_batch` | folder/pattern/settings → images/curves/matrices/progress | glob、filesystem、Qt thread、Matplotlib | `RunWaxsBatch` + catalog/export ports；JobRunner adapter |
| 6 | export | `export_curve_csv`、`write_matrix_csv`、`export_image_png` | arrays/settings/path → files | filesystem、Matplotlib | `WaxsExportRepository` adapter |
| 7 | presentation | `InSituProcessingWidget`、`ScatteringImageViewer`、Qt workers | typed state/commands ↔ widgets | PyQt、Matplotlib Qt canvas | `presentation.WaxsViewModel` + legacy widget bridge |

## 迁移前调用关系

```text
InSituProcessingWidget
  ├─ QFileDialog / QMessageBox
  ├─ ImageLoadWorker ── calibration.image_loader / HDF5
  ├─ pure geometry + mask + integration functions (same UI module)
  ├─ export helpers (same UI module) ── filesystem / Matplotlib
  └─ BatchWorker (same UI module)
       ├─ file discovery/loading
       ├─ integration
       └─ all export formats
```

## 本阶段完成后的调用关系

```text
InSituProcessingWidget / ScatteringImageViewer
  ├─ QFileDialog / QMessageBox（仅 presentation）
  └─ WaxsViewModel
       ├─ LoadWaxsImage ── WaxsImageRepository
       ├─ geometry / display / integration use cases ── pure domain
       ├─ ExportWaxsCurve / ExportWaxsImage ── WaxsExportPort
       └─ RunWaxsBatch ── WaxsBatchRunnerPort
            └─ JobRunnerWaxsBatchAdapter ── isolated worker process
                 ├─ CalibrationWaxsImageRepository
                 ├─ LocalWaxsFileCatalog
                 └─ LocalWaxsExportAdapter
```

- Qt 的 `ImageLoadWorker` 和 `BatchWorker` 仅保留线程/信号桥，不再实现文件发现、科学计算、导出或进程管理。
- `ui/waxs_page.py` 不再定义页面、domain、loader 或 batch 实现，只 re-export 历史名称。
- `WAXS/WAXS.py` 保留 `MainWindow`、直接执行和 loader 名称，但实现统一委托给 feature composition root/infrastructure。
- `load_image_matrix`/`load_tiff_matrix` 保留为 infrastructure-owned 兼容 API，不再被新的 WAXS 页面工作流调用。

## 必须保持的数值与行为

- NXS/TIFF 支持、P03 module stitching/orientation、1-based frame UI 与 0-based loader index。
- 当前 q 公式、Å⁻¹ 单位、beam center convention、pixel/distance 单位换算及 `-121.0` 未设置 sentinel。
- mask 在 linear intensity 空间定义；log display 不能重新解释 mask threshold。
- radial/azimuthal、pixel/2theta/q axis、bin edges、mean aggregation、line/circle selection orientation。
- batch 文件排序、每个 NXS frame 的命名 `_f0001`、首曲线背景、CSV headers、PNG colormap/limits。
- 页面布局、交互工具、现有独立 `WAXS/WAXS.py` 启动入口保持。

## Characterization tests

1. fake image repository：单 frame、多 frame、越界 frame、文件错误，无 QApplication。
2. 固定小数组的 q-map 数值、shape/orientation 与 q-range cut。
3. mask/log/percentile limits，包含 NaN、inf、阈值外、常数图。
4. radial/azimuthal、line/circle cut、wrap-around angle、smoothing 数值回归。
5. fake catalog/export repositories 的三文件 batch：frame 展开、progress、取消、单文件错误继续。
6. fake export adapter：curve/matrix/image command 的路径和设置保持。
7. ViewModel 状态转换，无 QApplication/真实文件。
8. legacy bridge 静态测试：UI 不再定义 scientific calculation、concrete loading/export/batch implementation。

## 迁移顺序

迁移已严格按文件加载 → geometry → mask → integration → batch → export → presentation 完成。每一步先引入 dependency seam 与测试，最后才将旧独立入口替换为同一 feature page 的兼容启动器。
