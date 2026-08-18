# Fitting Controller 渐进式拆分地图

- **Status**: Historical
- **Scope**: 第 12A 步的 `FittingController` 静态审计快照与迁移顺序
- **Related code**: [`controllers/fitting_controller.py`](../../controllers/fitting_controller.py)
- **Related tests**: [`tests/test_fitting_legacy_bridge.py`](../../tests/test_fitting_legacy_bridge.py)、
  [`tests/test_fitting_presentation.py`](../../tests/test_fitting_presentation.py)
- **Last verified**: 2026-08-18

## 当前迁移结果

第 12B–12G 的目标边界已经渐进落实：

- constraints、scoring、curve transformations、ROI/cut、detector settings 和 in-situ cut 的
  数学实现位于 domain，并有数值回归测试；
- 文件/曲线加载、导出、远程 cache、in-situ records、参数快照、AI artifacts/log、模型参数、
  AI catalog、dependency availability、模型构建和 q-space 均通过 application command 或 port；
- 具体文件系统、model registry 和 q-space adapters 均由 fitting feature 持有；旧路径只作
  import 兼容；
- `FittingViewModel` 已把 storage、in-situ 与 scientific command groups 拆为独立协作者，主
  ViewModel 保持在 300 行 review 阈值；
- AI fitting 使用 `Predictor`/model port 与 `JobRunner`，in-situ workflow 复用单文件 fitting
  use case，不复制算法；
- 生产运行时直接构造 feature-owned `FittingViewBinding`。顶层
  `controllers/fitting_controller.py` 与 feature `legacy_bridge.py` 仅 re-export；17,341 行
  binding 负责 Qt signal、widget、Matplotlib rendering 和动态 UI 契约，不再拥有上述计算或
  I/O 实现；它只能按有 characterization test 的 UI 状态组继续拆分，禁止机械重写。

当前验证基线见 [`remaining-work.md`](remaining-work.md)。以下内容保留为最初静态审计快照。

本文保留最初拆分依据，行号和“当前实现”描述不代表现在的完整架构。当前 Fitting
presentation ownership 与兼容边界以
[`docs/ui/workspaces/fitting.md`](../ui/workspaces/fitting.md) 为准。

- 审计日期：2026-08-17
- 审计对象：`controllers/fitting_controller.py`
- 文件规模：18,230 行；其中 `FittingController` 为 3,024–18,230 行，共 15,207 行。
- 本轮范围：第 12A 步，只做静态分析和拆分规划；未移动或修改任何 Python 源码、测试、配置或依赖。
- 行号说明：本文行号对应本次审计快照，后续修改文件后可能漂移；方法名和职责边界才是长期索引。

## 0. 拆分结论

`fitting_controller.py` 不是单一 controller，而是下列职责的集合：

1. 5 个可脱离 Qt 的数组处理函数；
2. 8 个 Qt widget、window、worker 或 display helper；
3. 远程文件缓存、目录扫描和异步图像读取；
4. `FittingController` 内的图像显示、ROI/cut、manual fitting、AI fitting、in-situ、绘图、导出和 session 编排；
5. 对 `global_params`、`user_settings`、`ModelParametersManager`、SciPy、Matplotlib、fabio、h5py 和外部 AI 子进程的直接访问。

最安全的拆分方向不是按文件行数平均切割，而是一次建立一个 dependency seam：先冻结纯数学行为，再抽文件端口/use case，再引入 ViewModel，最后迁移 AI 和 in-situ。Qt 控件、文件对话框和消息框应留在 presentation；数组运算、约束、评分和模型参数定义进入 domain；文件格式和外部进程进入 infrastructure adapter；工作流编排进入 application use case。

本文建议的总体迁移顺序与第 12B–12G 步一致：

```text
纯计算与参数结构
  → 文件加载/导出 ports + use cases
  → FittingState + ViewModel
  → AI candidate/refinement use cases
  → in-situ workflow
  → 删除已有测试保护的重复 legacy 实现
```

当前文件仍是有效的 legacy 入口。迁移期间应由它调用新实现，不能先移动方法再补测试，也不能同时重写 UI。

## 1. 文件级结构与共同依赖

### 1.1 顶层结构

| 行号 | 当前对象 | 当前职责 | 目标归属 |
|---|---|---|---|
| 65–166 | `apply_threshold_mask`、`apply_input_image_options`、`finite_mean_axis`、`finite_log_profiles`、`mirror_fill_detector_gaps` | threshold mask、翻转、有限值均值、log profile、detector gap 镜像填充 | fitting domain；先做数值回归测试 |
| 264–394 | `NoWheelDoubleSpinBox`、`CurrentPageHeightStackedWidget`、`ManualAutoRefineWorker`、`RefineUiBridge` | Qt widget 与 manual refine 线程/信号桥 | presentation；worker 后续改由 application `JobRunner` 调度 |
| 397–628 | `InsituBatchImageLoader`、`InsituCutWorker` | in-situ 文件读取与 cut 线程 | application workflow + file adapter + JobRunner；纯 cut 数学进入 domain |
| 631–2,491 | `IndependentMatplotlibWindow`、`IndependentFitWindow`、`UnifiedDisplayManager` | 2D/1D 独立窗口、交互选区、绘图和显示同步 | presentation |
| 2,507–2,686 | 远程缓存和路径函数 | 网络/云路径识别、缓存复制、容量限制 | infrastructure file/cache adapter |
| 2,689–3,021 | `FolderImageScanWorker`、`AsyncImageLoader` | 文件扫描、CBF/NXS/TIFF/stack 读取 | infrastructure adapters + application port；Qt signal wrapper 暂留 presentation |
| 3,024–18,230 | `FittingController` | 所有 fitting 工作流的 UI 编排、状态、计算与 I/O | 渐进式缩减为 legacy Qt 信号桥 |

### 1.2 共享状态与外部依赖

`FittingController` 同时维护 `input_image`、`raw_input_image`、`q`、`I`、`current_1d_data`、`current_cut_data`、`fitting_data`、当前文件/stack/NXS frame、ROI、selection、AI process、in-situ records/workers 和多个 window 引用。相同概念还可能存在于 Qt widget、`global_params`、`user_settings` 和 `ModelParametersManager` 中。

直接依赖包括：

- UI：PyQt5 widget、signal、`QThread`、`QProcess`、`QTimer`、`QFileDialog`、`QMessageBox`、graphics scene/view；
- 数值：NumPy、SciPy `ndimage` / `least_squares`；
- 绘图：Matplotlib Qt backend；
- 文件：fabio、h5py、`calibration.image_loader`、`utils.load_SAXS_data`、JSON/CSV/text；
- scientific/model：`utils.fitting`、q-space calculator、AI fitting profiles/pipeline/constraints/model registry；
- 状态：`core.global_params`、`core.user_settings`、`ModelParametersManager`；
- 外部执行：AI fitting worker 和权威脚本
  `utils/ML_Fitting_1D_GISAXS/Training/predict_topk.py`。

### 1.3 必须冻结的跨组行为

后续提取不得改变：

- 图像和 stack 的 array orientation、NXS frame 选择和求和范围；
- threshold mask 对 NaN、上下阈值和被屏蔽像素的处理；
- `flip_ud` 只对输入执行一次，不在显示阶段再次翻转；
- q 单位换算、source/display/model unit 的区别；
- ROI 在正/负 q 轴下的控制值映射、过滤顺序和边界；
- duplicate q 的稳定排序与合并方式、插值顺序和 points 数量；
- pixel cut 与 q-space cut 的轴方向、宽度/高度语义和均值规则；
- fitting 参数默认值、alias、bounds、单位、约束、模型分量和 residual 定义；
- AI 的 Fast/Balanced/Exhaustive profile、随机种子、candidate 排名和约束；
- session、parameter snapshot、AI 输出和 in-situ cache/export 的现有格式。

## 2. 文件和曲线加载

### 2.1 当前方法和类

| 子职责 | 当前类/方法 |
|---|---|
| 路径与远程缓存 | 顶层 `_default_remote_cache_dir`、`_app_root_dir`、`_resolve_remote_cache_dir`、`_display_remote_cache_dir`、`_is_mapped_network_drive`、`is_cloud_or_network_path`、`_cache_file_path_for_source`、`_is_remote_raw_cache_name`、`_copy_remote_file_to_cache`、`_enforce_remote_cache_limit`；controller `_load_remote_cache_settings`、`_save_remote_cache_settings`、`_configure_async_loader_remote_cache`、`_browse_remote_cache_folder`、`_clear_remote_file_cache` 和 remote 状态回调（2,507–2,686、3,367–3,530） |
| 文件夹扫描/导航 | `FolderImageScanWorker`；`_supported_folder_image_extensions`、`_logical_navigation_files`、`_ordinary_stack_sequence`、`_scan_folder_images_for_file`、`_apply_folder_image_scan_result`、`_show_previous_folder_image`、`_show_next_folder_image`、`_show_folder_image_at_offset`、`_select_folder_image`（2,689–2,740、3,533–3,857） |
| 图像/stack 读取 | `AsyncImageLoader` 的 `load_image`、`run`、`_load_detector_file`、`_load_multiple_nxs_frames`、`_load_multiple_detector_files`、`_load_single_cbf_file`、`_load_multiple_cbf_files`；controller `_import_gisaxs_file`、`_apply_imported_gisaxs_file`、`_validate_imported_file`、`_on_import_value_changed`、`_on_stack_value_changed`、`_update_stack_display`（2,743–3,021、5,989–6,239） |
| loader 回调/摄取 | `_on_image_loaded`、`_on_image_loading_progress`、`_on_image_loading_error`、`_ingest_workflow_image_without_preview`（8,774–8,832） |
| 1D 曲线读取 | `_import_1d_file`、`_on_1d_file_value_changed`、`_load_1d_data`（10,117–10,258） |
| 拖放 | `eventFilter`、`_detector_path_from_drop_event`（10,309–10,352） |

### 2.2 输入和输出

- 输入：用户选择/拖入的文件路径、load mode、stack count、当前 NXS frame、远程缓存设置；支持路径中实际读取的主要格式为 CBF、NXS、TIF/TIFF，1D 曲线为 DAT/TXT。
- 图像输出：二维 `float32` NumPy array、源路径、stack/NXS frame 导航状态、加载进度或错误字符串；结果继续写入 `raw_input_image` / `input_image` 并触发显示、q mesh 和 cut。
- 曲线输出：`q`、`I`、可选误差，以及 `current_1d_data = {q, I, err, file_path, q_source_unit}`；随后更新 ROI 和 1D plot。
- 失败目前通过异常、Qt error signal、`QMessageBox` 或 log/message 混合表达，没有统一的结构化错误类型。

### 2.3 UI 依赖

- `QFileDialog`、drag/drop event、line edit、stack spinbox、navigation button、progress bar、`QMessageBox`；
- `FolderImageScanWorker` 和 `AsyncImageLoader` 继承 `QThread`，把文件 I/O 与 Qt 生命周期绑定；
- controller 在回调中直接更新控件、图像和下游状态。

### 2.4 `global_params` 依赖

- `_import_1d_file` / `_on_1d_file_value_changed` 读写 `fitting.last_session.last_1d_directory`；
- `_on_load_mode_changed` 持久化 `fitting.gisaxs_input.load_mode`；
- 图像载入后的显示和 q-space 初始化还会间接读取 `fitting.gisaxs_input.*`、`fitting.detector.*` 和 beam 参数；
- 远程缓存设置主要来自 `user_settings`，不是 `global_params`。

### 2.5 可优先抽出的纯函数

- natural sort key、ordinary stack 序列计算、stack count clamp；
- NXS frame/window 选择规则；
- supported extension 判定和 drop path 解析中的非 Qt 部分；
- loader 输出到 typed scattering data / curve data 的规范化；
- 1D 列数据合法性检查和 source-unit metadata 构造。

文件打开、fabio/h5py 调用、远程缓存复制不是 domain 纯函数，应由 repository port 的 infrastructure adapter 承担。

### 2.6 建议迁移顺序

1. 用现有 stack 测试冻结 NXS/CBF/ordinary stack 行为和 array orientation；补 TIFF、错误和 1D 曲线 characterization tests。
2. 提取 typed `ScatteringImage` / `CurveData`、格式规则和结构化 `FileError`。
3. 定义 `ScatteringFileRepository`、`CurveRepository`、`FitResultRepository` ports。
4. 把现有 fabio/h5py/text/cache 逻辑包为 adapters，不改变格式分支。
5. 增加无 Qt 的 `LoadScatteringFile`、`LoadCurve`；Qt dialog 只提供路径。
6. controller 通过薄兼容调用 use case，待 ViewModel 接管状态后再删除重复加载代码。

### 2.7 现有测试覆盖

`tests/test_cut_fitting_stack.py` 已覆盖：独立 NXS 与 mosaic NXS 导航差异、NXS frame stack clamp 和求和方向、stack UI clamp、普通文件自然排序、CBF oversized stack clamp（有 fabio 时）。

缺口：真实 TIFF/NXS/HDF5 错误、损坏文件、无权限/不存在路径、远程 cache 命中与清理、1D DAT/TXT 单位和误差列、drag/drop、loader 取消，以及 use case 级无 GUI 文件测试。

## 3. 2D 图像显示

### 3.1 当前方法和类

| 子职责 | 当前类/方法 |
|---|---|
| 输入数组处理 | 顶层 `apply_threshold_mask`、`apply_input_image_options`、`finite_mean_axis`、`finite_log_profiles`、`mirror_fill_detector_gaps`（65–166） |
| 独立 2D window | `IndependentMatplotlibWindow`：display option widget、center picking、mouse selection、`update_image`、q-to-pixel、q extent、selection redraw（631–1,888） |
| 主图显示选项 | `_load_image_display_options`、`_save_image_display_options`、`_initialize_image_display_option_widgets`、`_sync_image_display_option_widgets`、`_apply_image_display_options`、threshold/mirror/flip/cmap 回调、`_reapply_input_image_options`、`_get_current_display_image`（4,911–5,332） |
| image/q 显示 | `_show_image`、`_display_image`、`_compute_q_meshgrids_and_store`、`_update_graphics_view`、`_prepare_image_data_for_display`、`_refresh_image_display`、`_show_independent_window`、`_on_detector_center_picked`（6,293–6,377、8,835–9,108） |
| preview/render | `_draw_selection_on_main_view`、`_downsample_for_preview`、`_preview_extent`、`_draw_preview_selection`、`_draw_detector_center_on_axis`、`_try_update_cached_preview`、`_update_graphics_view_with_selection`（9,538–9,838） |
| 色标 | `_calculate_vmin_vmax`、`_update_vmin_vmax_ui`、`_get_vmin_vmax_from_ui`、`_handle_color_scale` 和 auto/log/vmin/vmax 回调（9,842–10,080） |
| center/尺寸更新 | `_auto_find_center`、`_calculate_95_percent_width`、`_update_GUI_image`、`_update_outside_window`（10,452–10,578、10,644–10,982） |

### 3.2 输入和输出

- 输入：原始二维 image、threshold/flip/mirror/cmap/log/vmin/vmax 设置、detector geometry、q mesh、ROI selection、graphics view 尺寸。
- 输出：处理后的 `input_image`、display image、QPixmap/graphics item、Matplotlib image/overlay artists、cached preview、vmin/vmax、detector center 和 q-axis extent。
- 输入数组和显示数组不是同一个概念；`flip_ud` 当前在输入阶段应用一次，显示阶段不得重复。

### 3.3 UI 依赖

高度依赖 `QGraphicsScene/View`、QPixmap、checkbox/spinbox/combobox、mouse/key event、独立 QWidget 和 Matplotlib Qt canvas。`_prepare_image_data_for_display` 等少数数组方法本可独立，但当前由 controller/UI state 驱动。

### 3.4 `global_params` 依赖

- `fitting.gisaxs_input.show_cut_region`、`show_center`、`colormap`、`flip_ud`、threshold mask 开关/上下限、mirror gap fill/margin；
- `fitting.detector.beam_center_x/y`、pixel size、distance、`show_q_axis`；
- `beam.grazing_angle`、`beam.wavelength`；
- center pick 完成后直接保存 detector center。

### 3.5 可优先抽出的纯函数

- 顶层 5 个数组函数；
- log/color scale 的有限值范围计算；
- preview downsample 和 extent；
- 给定 image shape 与 detector center 的 mirror gap 变换；
- 给定显式 geometry 的 q mesh / q extent；
- 95% profile width 和 center profile 数学部分。

q mesh 提取时必须把 geometry 作为显式输入，不能在 domain 内读取 `global_params`。

### 3.6 建议迁移顺序

1. 为 5 个顶层函数和 color scale/q mesh 增加固定数组 fixture 回归测试。
2. 提取 `ImageDisplayOptions`、`DetectorGeometry` 值对象和纯 image transform service。
3. 由 application 接收 settings/session port 并构造显式输入。
4. controller/未来 ViewModel 只更新 state；presentation 将 state 渲染为 QPixmap/Matplotlib artists。
5. 最后再统一主窗口与独立窗口的显示 option，同步期间保留兼容桥。

### 3.7 现有测试覆盖

`tests/test_cut_fitting_stack.py` 已覆盖 threshold mask 的 NaN/阈值处理、masked pixel 在均值和 center log profile 中零权重、`flip_ud` 单次应用和 controller reapply，以及 graphics view 更新前 reset transform。

缺口：mirror gap fill、color scale、q-axis extent、center pick、主/独立窗口 option 同步、q mesh 数值、自动找中心、不同 image orientation 和离屏图像渲染。

## 4. ROI 和 cut

### 4.1 当前方法和类

| 子职责 | 当前类/方法 |
|---|---|
| ROI state/bounds | `_roi_active`、`_get_roi_active_arrays`、`_get_roi_domain_bounds`、`_current_q_has_negative_values`、`_roi_editing_should_be_enabled`、`_initialize_roi_from_current_q`（3,861–4,199） |
| 正负坐标/UI 映射 | `_roi_controls_use_abs_negative`、`_roi_data_to_control_range`、`_roi_data_to_control_values`、`_roi_control_to_data_values`、`_nearest_roi_control_value`、`_sync_roi_controls_to_current_display`（3,981–4,056） |
| ROI 应用/重采样 | `_apply_roi_to_data_and_refresh`、`_get_current_fit_axes`、`_compute_display_xmin_for_log`、`_adjust_roi_bounds_for_log_x`、`_resample_1d`、`_interpolate_series`、`_sort_filter_cut_pairs`、`_filter_cut_pairs_for_active_axis`（4,287–4,650） |
| selection | `IndependentMatplotlibWindow` 的 mouse/selection 方法；controller `_on_region_selected`、`_on_cutline_parameters_changed`、`_create_selection_from_parameters`、`_update_parameter_selection_display`、`_sync_independent_window_selection`、`_refresh_current_parameter_selection_from_ui`、`_clear_parameter_selection`（9,111–9,535） |
| cut 执行 | `_get_cut_center_coordinates`、`_resolve_cut_points`、`_perform_cut`、`_perform_cut_operation`、horizontal/vertical wrapper、`_extract_cut_q_mode`、`_extract_cut_pixel_mode`（10,985–11,338） |
| pixel/q 转换 | `_get_detector_for_pixel_conversion`、`_convert_pixel_coords_to_q`、`_convert_pixel_to_qy`、`_convert_pixel_to_qz`（11,341–11,446） |
| in-situ 重复实现 | `InsituCutWorker._sort_filter_pairs`、`_interpolate_series`、`run`（486–628） |

### 4.2 输入和输出

- 输入：image、`qy_grid` / `qz_grid`、cut orientation、center、width/height、ROI min/max、points 数、插值方式、positive/negative axis filter、detector geometry。
- 输出：有序且过滤后的 q/intensity，可能带 resampling；同步到 `q`、`I`、`current_cut_data`、selection metadata 和 plot。
- ROI/control 在 negative-only 模式下可能显示绝对值，但 domain 数据仍保留负 q；顺序不能颠倒。

### 4.3 UI 依赖

ROI slider/spinbox、cutline 控件、checkbox、graphics selection、Matplotlib mouse event、timer/debounce 和 plot refresh 紧密混合。`InsituCutWorker` 又把数学和 `QThread` signal 混在一起。

### 4.4 `global_params` 依赖

- `fitting.fit.points_num`；
- `fitting.gisaxs_input.center_x/center_y/cutline_*` 及 cut region 持久化；
- pixel cut 转 q 读取 `fitting.detector.pixel_size_x/y`、beam center、distance 和 beam grazing angle/wavelength；
- q-axis 显示也读取 detector/beam 参数。

### 4.5 可优先抽出的纯函数

- ROI finite mask、domain bounds 和 control/data 坐标变换；
- stable sort、非有限值过滤、duplicate q 分组平均；
- positive/negative 轴过滤，且必须固定“过滤在重采样之前”；
- linear/nearest/cubic 等插值的输入清洗与输出规则；
- 给定二维数组和显式 selection 的 horizontal/vertical pixel cut；
- 给定 q mesh 的 q-space rectangular cut；
- 给定 q mesh 的 fractional pixel→q 插值；
- selection rectangle 和 cut metadata 数据结构。

重要差异：普通 pixel cut 通过 `finite_mean_axis` 忽略非有限值；`InsituCutWorker` 的 pixel 模式目前使用普通 `np.mean`。在 characterization tests 证明等价或任务明确修改科学行为之前，不能把两者直接合成同一实现。

### 4.6 建议迁移顺序

1. 用固定小矩阵冻结 q/pixel 两种 cut、两种方向、negative-only、duplicate q 和 NaN 行为。
2. 提取 ROI 值对象、selection 数据结构、排序/过滤/插值函数。
3. 分别提取普通 cut 和 in-situ cut 的纯 kernel；先保留其均值差异。
4. controller 用 compatibility import 调用 domain，仍负责读取 widget 和画图。
5. 后续 in-situ workflow 复用经过明确选择的 cut API，而不是复制算法。

### 4.7 现有测试覆盖

`tests/test_cut_fitting_stack.py` 已覆盖 fractional pixel position 映射到不同 q、duplicate q 在插值前合并、negative-only 在 resampling 前过滤，以及有限值均值相关行为。

缺口：完整 horizontal/vertical cut、q-space selection、ROI 正负坐标 round-trip、各插值方法、points clamp、NaN/empty region、普通与 in-situ 差异、cut metadata 和固定 fixture 的迁移前后对比。

## 5. Manual fitting

### 5.1 当前方法和类

| 子职责 | 当前类/方法 |
|---|---|
| particle 参数/UI registry | `_collect_active_particles`、`_get_particle_sequence_flags`、shape/parameter alias 方法、dynamic particle widget 创建/删除、参数 range/connection、`_load_particle_parameters`、`get_all_particle_parameters`（12,140–13,435） |
| global model 参数 | `get_global_parameter`、`set_global_parameter`、`get_all_global_parameters`、`reset_global_parameters` 和 editing callbacks（15,595–15,958） |
| auto K | `_save_auto_k_enabled`、`_load_auto_k_enabled`、`_on_auto_k_button_clicked`、`_optimize_k_value`（15,639–15,874） |
| refine UI/worker | `ManualAutoRefineWorker`、`RefineUiBridge`；`_show_manual_auto_refine_dialog`（352–394、16,183–16,560） |
| refine setup/math | `_build_manual_refine_setup`、`_get_current_manual_param_values`、`_build_manual_refine_param_descriptors`、`_manual_refine_default_selected`、dialog state/bounds、`_run_manual_auto_refine`（16,563–16,901） |
| refine apply/preview | `_apply_manual_refine_result`、`_preview_manual_refine_curve`（16,904–16,990） |
| manual evaluation | `_perform_manual_fitting`、`_store_fitting_data`（16,994–17,177） |
| fitting spec lookup | `_get_particle_parameter`、`_get_ui_control_name`、`_get_last_fitting_spec_and_params`、`_validate_parameter_retrieval`（17,290–17,349、17,470–17,551、17,772–17,836） |

### 5.2 输入和输出

- 输入：当前 cut 或导入 1D 曲线、active particle shapes/sequence、particle 参数、global 参数 `background`、`sigma_res`、`nu_res`、`int_res`、`k`，以及 refine selection/bounds/target/stop 设置。
- manual evaluation 调用 `utils.fitting.make_mixed_model`、`params_template`、`mixed_model_components`；q 在送入模型前按现有规则转换为 `nm^-1`。
- auto refine 使用 SciPy `least_squares`，当前 residual 是 `log10` 空间差，epsilon 为 `1e-30`，并支持停止/进度。
- 输出：总拟合曲线、component curves、参数/meta、拟合结果字典、log/RMSE 等显示信息，以及更新后的参数控件。

### 5.3 UI 依赖

manual fitting 从大量动态 Qt widget 和 `ModelParametersManager` 读取参数；refine dialog、按钮状态、progress 和结果 apply 均在 controller；worker 继承 `QThread`。模型计算本身可脱离 Qt，但当前输入构造和结果展示未分离。

### 5.4 `global_params` 依赖

manual scientific evaluation 没有直接从 `core.global_params` 读取模型参数；主要状态来自 `ModelParametersManager` 和 Qt 控件。当前曲线、ROI、points 与 detector/q 单位则来自前序 controller 状态，并可能由 `global_params` 间接初始化。

### 5.5 可优先抽出的纯函数

- typed particle/global parameter structures，包含 alias、shape token、sequence 与默认值；
- 参数向量、bounds 和 selected-parameter descriptor 构造；
- mixed-model evaluation 的输入/输出数据结构；
- component/result 组装和统计量；
- `log10` residual、RMSE 和 auto-K objective；
- refine result 到参数值 mapping。

dynamic widget registry、label/style、dialog state 和 signal connection 不是 domain。

### 5.6 建议迁移顺序

1. 为每一种已支持 particle combination 建立固定 q/参数/curve fixture，记录 component 和总曲线。
2. 提取参数 dataclass、constraints/bounds、unit conversion、model evaluation 和 scoring；controller 用兼容导入。
3. 单独冻结 auto-K 和 least-squares refine 的初值、bounds、容差、随机/迭代设置和结果。
4. 创建 `RunManualFit` / `RefineManualFit` application use case 后，再让 ViewModel 持有 manual fitting state。
5. 最后移除 controller 内重复计算；dynamic particle widget 仍归 presentation。

### 5.7 现有测试覆盖

`tests/test_random_cylinder_forward_model.py` 只验证 `utils.fitting` 中 random cylinder 向量化结果与标量参考一致，以及 q=0 归一极限。这是底层模型覆盖，不是 controller manual fitting 回归。

缺口：manual mixed-model 总曲线/component、单位转换、参数 alias/default/bounds、auto-K、refine residual、停止行为、结果 apply、空/非法曲线和固定 seed/fixture 对比。

## 6. AI candidate generation

### 6.1 当前方法和类

| 子职责 | 当前类/方法 |
|---|---|
| profile/settings | `_ai_fitting_settings`、`_save_ai_fitting_settings`、`_default_ai_run_settings`、`_ai_run_settings`、`_restore_ai_session_settings`、`_current_ai_profile`、`_set_ai_profile`、`_mark_ai_profile_custom`（13,631–13,742） |
| workspace/model selection | `_connect_ai_fitting_settings_widgets`、workspace 打开/恢复/同步、model scan/refresh/browse/select、constraint combo 同步（13,745–14,455） |
| curve preparation | `_current_ai_curve_arrays`（14,458–14,560） |
| excluded points | `_filter_ai_excluded_points_for_display`、input data dialog、exclude/restore 方法（14,564–14,797） |
| request/output setup | `_ai_prediction_output_root`、`_ai_current_prediction_dir`、`_clear_ai_current_prediction_dir`、`_prepare_ai_prediction_io`、`_ai_exact_nonempty_arg`（14,800–14,857） |
| external generation process | `_start_ai_prediction`、running/log/stdout/stderr handlers、`_handle_ai_process_text`、finish/error/stop/budget callbacks（14,860–15,076） |
| constraints | `build_ai_constraints_json_from_ui`、`_write_ai_constraints_json`、`_show_advanced_constraints_dialog`（15,322–15,592） |

### 6.2 输入和输出

- 输入：Fast/Balanced/Exhaustive profile、模型路径、随机种子、candidate/refinement/sample-scale 参数、constraint mode/fixed geometry/advanced bounds、当前 q/I/误差和 excluded q points。
- `_current_ai_curve_arrays` 负责选择 cut 或导入曲线、清洗有限/正值、axis/ROI 过滤；没有误差时生成 `max(5% × I, 1e-30)`，并要求至少 16 点。
- application/adapter 把曲线与 constraints 写入任务目录，通过 `JobRunner` 运行
  `utils/ML_Fitting_1D_GISAXS/Training/predict_topk.py`。
- 输出：process log/progress、candidate/result 文件和 summary；失败以 process exit/error、日志和 UI 状态表达。

### 6.3 UI 依赖

workspace、profile/model/constraint widget、input data dialog、log view、按钮状态、`QProcess` 和 `QMessageBox` 均在 controller。candidate request 的参数收集直接读取 widget。

### 6.4 `global_params` 依赖

此组没有直接访问 `core.global_params`；AI UI 默认值/模型选择主要使用 `user_settings` 和 fitting session。曲线和 ROI 仍来自 controller state，后者可能由 global settings 初始化。

### 6.5 可优先抽出的纯函数

- AI curve source 选择、finite/positive 清洗、ROI/axis/exclusion mask；
- error/sigma fallback；
- profile + override 到 typed `CandidateGenerationRequest` 的映射；
- excluded-q normalization；
- constraint request/JSON 数据结构；
- process output message 到 progress/status 的解析。

模型发现、CLI 参数构造和 profile 定义已有部分实现位于 `utils.ai_fitting_*`，迁移时应复用并建立明确 ownership，不能在新 feature 再复制一份。

### 6.6 建议迁移顺序

1. 固定三种 profile 的完整参数和同一 seed 下的 request/fixture。
2. 提取纯 curve preparation 和 constraint mapping，并让 controller 兼容调用。
3. 定义 `Predictor` port 与 `GenerateCandidates` use case；外部进程通过 `JobRunner`，消息只传可序列化数据。
4. ViewModel 接收 progress/result/error，不管理 process。
5. 保留原输出文件格式，直到 candidate consumer 和 export 全部迁移并有回归测试。

### 6.7 现有测试覆盖

- `tests/test_ai_fitting_pipeline.py`：三种 profile 共用同一脚本、关键参数差异、`.keras`/SavedModel 路径规范化；
- `tests/test_ai_fitting_profiles.py`：profile 成本顺序、Balanced 默认、override 变为 Custom；
- `tests/test_ai_model_registry.py`：model discovery/contract 和 lazy load；
- `tests/test_ai_fitting_constraints.py`：geometry exclusion、D=0、hard-core 公式、sigma 语义；
- `tests/test_insitu_ai_compatibility.py`：旧 AI session 设置迁移。

缺口：controller 的完整 request 构造、curve source/ROI/exclusion/sigma fallback、固定 seed candidate 输出、progress/错误/取消/timeout、无 TensorFlow application 测试，以及三种 profile 的端到端结果比较。

## 7. Candidate verification、ranking 和 refinement

### 7.1 当前方法和类

| 子职责 | 当前类/方法 |
|---|---|
| candidate 文件/表格 | `_show_ai_candidate_table` 读取 `top20_candidates.json` 并构造表格；`_preview_ai_candidate_from_table`、`_load_selected_ai_candidate_from_table`（15,144–15,236） |
| candidate → 参数 | `_load_ai_candidate_params`（15,239–15,319） |
| physical constraints | `build_ai_constraints_json_from_ui`、advanced constraints dialog（15,322–15,592）；主要 verification/ranking 由权威 `Training/predict_topk.py` worker 完成 |
| manual candidate refinement | `_build_manual_refine_setup`、参数 descriptors/bounds、`_run_manual_auto_refine`、`_apply_manual_refine_result`、`_preview_manual_refine_curve`（16,563–16,990） |
| in-situ refine bridge | `_start_insitu_auto_refine`、`_insitu_auto_refine_selected_params`、refine progress/finish/apply/fail/cleanup（7,851–8,033） |

### 7.2 输入和输出

- 输入：candidate row 的 particle components、global parameters、metrics/rank，physical constraints、当前 curve、refine bounds/selected params 和 predictor/pipeline 输出。
- 输出：验证状态、ranked candidates、preview curve、加载到 UI 的参数、refined parameter/result 和 progress/error。
- 当前 controller 主要消费外部 pipeline 结果；generation、verification、ranking、refinement 的事实来源跨 controller 和脚本，边界尚未统一。

### 7.3 UI 依赖

candidate table/dialog、选中行、preview refresh、参数 widget 写入和 refine dialog/worker 都与 Qt 绑定。`_load_ai_candidate_params` 直接驱动 `ModelParametersManager`/UI，不是 application API。

### 7.4 `global_params` 依赖

没有直接使用 `core.global_params`。candidate row 中名为 `global_params` 的 JSON 字段是拟合模型的全局参数字典，不是应用全局单例；两者必须在新类型命名中区分。

### 7.5 可优先抽出的纯函数

- candidate JSON row schema/validation；
- candidate row → typed fitting parameters；
- physical verification service；
- deterministic score/rank key 和 tie-break；
- refinement parameter descriptors、bounds 投影、objective/scoring 和 result mapping；
- preview curve evaluation。

### 7.6 建议迁移顺序

1. 固定一组 candidate fixture，记录 verification、score、rank、参数映射和 preview curve。
2. 明确 pipeline 中 verification/ranking 的唯一 owner，先建立 port/use-case seam，不复制算法。
3. 将 predictor 置于 `Predictor` port 后，将 physical verification/scoring 放入纯 domain service。
4. refinement 经 `JobRunner` 执行；ViewModel 只接收 progress/result。
5. controller 保留 table selection 和旧 widget apply 桥，覆盖完毕后再删除旧逻辑。

### 7.7 现有测试覆盖

- `tests/test_insitu_ai_compatibility.py::test_candidate_row_preview_loads_parameters_and_requests_plot_refresh` 覆盖一条 legacy preview bridge；
- `tests/test_ai_curve_loader.py` 覆盖 candidate width 的 geometry 语义、refinement 使用用户 bounds、hybrid score 对窄区间 linear overshoot 的惩罚；
- `tests/test_ai_fitting_constraints.py` 覆盖部分 physical constraint 规则。

缺口：固定 candidate 集的完整 verification/ranking/tie-break、candidate JSON 错误、controller 参数映射、preview 数值、refinement 迁移前后结果、取消/错误，以及 Fast/Balanced/Exhaustive 下约束和 seed 行为。

## 8. In-situ workflow

### 8.1 当前方法和类

| 子职责 | 当前类/方法 |
|---|---|
| worker | `InsituBatchImageLoader`、`InsituCutWorker`（397–628） |
| dialog/mode/timer | `_setup_insitu_workflow_button`、dialog/canvas/layout/mode/visibility、`_start_insitu_timer`、`_stop_insitu_timer`、`_insitu_poll_latest`（6,406–6,953） |
| target/settings/status | `_resolve_insitu_target`、`_find_latest_cbf`、`_show_image_insitu`、`_insitu_workflow_settings`、状态/log/style/validation 方法（6,979–7,272） |
| workflow scheduling | `_start_insitu_workflow`、`_start_insitu_sequence_processing`、`_build_insitu_sequence_file_list`、pause/stop、watch list/poll/stability、`_process_next_insitu_workflow_file`、`_new_insitu_workflow_record`（7,275–7,563） |
| load/cut | `_load_insitu_workflow_batch_async`、image callbacks、`_start_insitu_cut_worker`、cut callbacks、deleted-point mask（7,566–7,809） |
| fitting/refine/AI | `_run_insitu_workflow_fit`、auto-refine start/progress/finish/fail/cleanup、AI full fit finish、`_complete_insitu_workflow_fit`、parameter dict、chi-square（7,812–8,128） |
| finalize/persistence/export | `_finalize_insitu_workflow_file`、cache path/reset/append/load、CSV/export/clear/open folder（8,131–8,274） |
| previews/monitoring | image/region/curve preview、heatmap state/window/refresh、trend keys/window/refresh（8,277–8,771） |

### 8.2 输入和输出

- 输入：单个最新文件或文件序列、folder/pattern、poll interval、file stability、batch/stack、cut 设置、fitting cadence、AI profile/refine 设置和取消/暂停动作。
- 每个文件经历 discover → stable → load → cut → optional fit/refine/AI → finalize；记录包含路径、load/fit 状态、参数、chi-square 和 error。
- 输出：内存 records、JSONL session cache、CSV、image/curve preview、heatmap、parameter trend 和 progress/status。

### 8.3 UI 依赖

整个 workflow 由 dialog widgets、`QTimer`、`QThread` worker、Matplotlib canvas 和 controller callbacks 驱动。调度状态与显示状态没有分离；presentation 同时管理进程细节。

### 8.4 `global_params` 依赖

workflow 方法本身很少直接访问 `core.global_params`，但 cut 的 detector geometry、image display、current fitting parameters 和下游 fitting 都依赖 controller/global state。settings 还混有 widget 值和 AI session/user settings。

### 8.5 可优先抽出的纯函数

- 文件自然排序、sequence 列表、watch target 和 batch 选择规则；
- serializable workflow state、record schema 和状态转换；
- file stability 判定的纯决策部分；
- fit cadence / continue-on-error 调度决策；
- record 聚合、chi-square、trend parameter keys、heatmap cut aggregation；
- cut kernel 应复用经过回归测试的 fitting domain API，不能复制。

### 8.6 建议迁移顺序

1. 建立 3 文件小 fixture，冻结顺序、单文件错误继续、取消、进度和 records/cache 格式。
2. 提取 serializable `InSituState` / `InSituRecord` 和纯调度决策。
3. 创建 application workflow；文件加载、cut 和单文件 fitting 分别调用已有 use cases。
4. 所有长任务经 `JobRunner`；workflow 只做序列、调度、聚合和恢复。
5. ViewModel 映射 workflow progress/state，Qt timer/dialog 只负责触发和显示。
6. 完成后删除重复 loader/cut/fitting orchestration，但保留旧入口桥直到动态调用确认完毕。

### 8.7 现有测试覆盖

`tests/test_insitu_ai_compatibility.py::test_simulated_insitu_settings_include_profile_without_acquisition` 只覆盖 `_insitu_workflow_settings` 保留 AI profile；stack 测试间接覆盖部分 loader 行为。

缺口：三文件 workflow、watch/poll/stability、顺序、暂停/恢复/取消、单文件错误继续、cache round-trip、同一 fitting use case 复用、进度、CSV、heatmap/trend 和 worker 崩溃隔离。

## 9. Plotting

### 9.1 当前方法和类

| 子职责 | 当前类/方法 |
|---|---|
| 2D 独立绘图 | `IndependentMatplotlibWindow`（631–1,888） |
| 1D/fit 独立绘图 | `IndependentFitWindow`，尤其 unit/filter/y-range controls 和 `update_plot`（1,891–2,332） |
| 1D display manager | `UnifiedDisplayManager.plot_1d_data`、GUI/independent 更新方法（2,335–2,491） |
| cut plot | `_plot_cut_data_with_log_handling`、`_plot_cut_result`、`_plot_cut_data_legacy`（9,237–9,278、11,449–11,580） |
| 单位/filter/limits | `_get_q_display_unit` 到 `_build_q_axis_label`、positive/negative filter 同步、`_filter_q_data_for_independent_display`、y-range、normalization、log/valid-data helpers（11,583–11,915） |
| fitting message/log UI | `_setup_fitting_text_browser` 到 `save_fitting_log`（11,921–12,134） |
| fitting plot | display mode switch/refresh、`_plot_fitting_result`、external window 显示、points-only 更新、log/norm/filter callbacks、`_get_current_data_for_display`、`_plot_data_points_only`（17,180–18,230） |
| in-situ plots | image/curve preview、heatmap 和 trend methods（8,277–8,771） |

### 9.2 输入和输出

- 输入：image/cut/imported curve/fitting result/component arrays、q source/display unit、log/norm/axis/y-range options、selection/center overlays。
- 输出：Matplotlib artists、Qt graphics items、axis label/limits、legend、text log 和 synchronized main/external window state。
- fitting plot 同时承担“准备可显示数据”和“执行渲染”，两者应分开。

### 9.3 UI 依赖

几乎全部依赖 Matplotlib Qt canvas、QGraphicsView、widget state 和 event callbacks。可测试的数组/axis preparation 被埋在 presentation 方法中。

### 9.4 `global_params` 依赖

2D window 和 q-axis plotting 读取 image display 与 detector/beam geometry；1D/fit plotting 主要读取 controller/UI state。text log/save 不直接读取 `global_params`。

### 9.5 可优先抽出的纯函数

- q source/model/display unit conversion；
- axis filter mask、normalization、valid y values/y limits；
- curve/component display series 组装；
- downsampling 和 finite/log-safe display preparation；
- plot labels/style 本身属于 presentation，可在纯 display-state mapper 中测试，但不应放进 scientific domain。

### 9.6 建议迁移顺序

1. 固定 q unit、positive/negative、log、normalization 和 y-limit 的数组输出。
2. 提取 `CurveDisplayState` mapper；保持 Matplotlib 调用在 presentation。
3. ViewModel 只提供当前曲线/fitting state 和 display commands。
4. 主窗口/独立窗口共同消费同一 display state，逐步删除重复数据准备代码。
5. 最后用离屏 GUI test 固定关键 artist/axis 状态，不做视觉重设计。

### 9.7 现有测试覆盖

`tests/test_cut_fitting_stack.py` 覆盖 graphics view fit/reset 和少量 axis/filter 数学的下游行为；其他 AI/manual 测试不验证 controller plotting。

缺口：q unit round-trip、normalization/log/y-range、cut/fitting/component plot、主/外部窗口同步、空/NaN/negative data、legend/checkbox、离屏渲染和 text log save。

## 10. Export

### 10.1 当前方法和类

| 子职责 | 当前类/方法 |
|---|---|
| fitting log | `save_fitting_log`（12,120–12,134） |
| particle/model parameters | `export_particle_parameters`、`import_particle_parameters`、`_build_fitting_parameter_snapshot`、`save_fitting_parameters_to_file`、`load_fitting_parameters_from_file`、save/load dialog（13,438–13,554） |
| AI output | `_export_ai_prediction_output`（15,085–15,141） |
| fitting curve | `_get_fitting_parameter_comment_lines`、`_build_export_header_lines`、`_export_fitting_data`（15,968–16,180） |
| in-situ | `_export_insitu_records_to_csv`、`_export_insitu_workflow_results`（8,209–8,254） |

### 10.2 输入和输出

- 输入：目标路径、TXT/CSV 选择、当前 q/I/fit/component/parameter state、AI output directory 或 in-situ records。
- 输出：fitting text/CSV（含注释/header 和 q display unit）、parameter JSON、AI output 文件夹复制、in-situ CSV、fitting log。
- 当前 success/failure 通过 dialog/message/exception 混合处理，格式化与路径交互未分离。

### 10.3 UI 依赖

`QFileDialog`、`QMessageBox`、clipboard/folder action 和当前 widget state。格式构造本可无 GUI，但现在与 dialog method 同体。

### 10.4 `global_params` 依赖

这些 export 方法没有直接读取 `core.global_params`；参数来自 `ModelParametersManager`、controller state 和 UI。`_get_fitting_parameter_comment_lines` 中的局部变量 `global_params` 只是模型 global parameter 名称列表，不是应用单例。

### 10.5 可优先抽出的纯函数

- typed export dataset、列对齐和 delimiter/header/comment 构造；
- q display unit 转换；
- parameter snapshot serialization；
- in-situ record → CSV rows；
- overwrite/不存在/权限/格式错误的结构化错误映射。

### 10.6 建议迁移顺序

1. 为每个现有格式保存 golden file，明确换行、delimiter、header、float precision、列顺序和 JSON keys。
2. 提取纯 serializer，定义 `FitResultRepository` / `ParameterSnapshotRepository` port。
3. 创建 `ExportFitResult` use case；路径来自 presentation，文件写入由 adapter 完成。
4. controller 暂时把旧 state 转成 request；待 ViewModel 状态稳定后移除旧格式化代码。
5. AI/in-situ export 分别随其 workflow 迁移，不在文件 use-case 任务中顺便重写。

### 10.7 现有测试覆盖

没有发现 fitting curve、parameter snapshot、AI output、in-situ CSV 或 log export 的直接测试。

优先补充：TXT/CSV/JSON golden files、单位/header/precision、空 result、非法路径、权限错误、已有文件、AI folder copy 错误、in-situ records 的稳定列顺序。

## 11. Session persistence

### 11.1 当前方法和类

| 子职责 | 当前类/方法 |
|---|---|
| controller state/session | `_set_default_parameters`、`get_parameters`、`set_parameters`、`get_imported_file`、`get_session_data`、`restore_session`（5,833–5,985） |
| fit checkbox | `_initialize_fit_checkboxes`、`_restore_fit_checkboxes`（5,484–5,535） |
| image/cut persistence | `_restore_gisaxs_input_parameters`、`_load_image_display_options`、`_save_image_display_options`、`_persist_cut_region_parameters`（4,878–5,010、5,346–5,372） |
| 1D last directory | `_import_1d_file`、`_on_1d_file_value_changed` 中的 `fitting.last_session` 更新（10,117–10,214） |
| particle snapshot | `_build_fitting_parameter_snapshot`、save/load file methods（13,464–13,534） |
| AI settings/migration | `_ai_fitting_settings` 到 `_restore_ai_session_settings`、workspace/model restore methods（13,631–14,245） |
| in-situ cache | `_insitu_cache_dir`、session cache path/reset/append/load/clear（8,157–8,266） |

### 11.2 输入和输出

- 输入：controller arrays/path/modes/stack/checkbox/ROI/AI state、用户参数 JSON、parameter snapshot JSON、in-situ JSONL records。
- 输出：`get_session_data()` dict、恢复后的 widget/controller state、`global_params` 持久化、`user_settings`、`ModelParametersManager` JSON 和 in-situ cache。
- restore 不只是反序列化，还会触发文件重载、widget 更新和后续显示，因此很难无 GUI 测试。

### 11.3 UI 依赖

`restore_session` 直接写大量 widget 并触发 loader/plot；checkbox/AI workspace/dialog state 也与 Qt 控件绑定。serialization、migration 和 effect orchestration 未分离。

### 11.4 `global_params` 依赖

- `fitting.last_session`；
- `fitting.gisaxs_input.*`；
- `fitting.fit.points_num`；
- `fitting.detector.*` 与 beam geometry；
- 多处方法直接 import singleton，另有多处重新构造 `GlobalParameterManager()` 获取同一 legacy instance。

并行状态源：AI/remote cache 使用 `user_settings`，particle/global model 参数使用 `ModelParametersManager`，in-situ 使用单独 JSONL cache。

### 11.5 可优先抽出的纯函数

- typed/versioned `FittingState` 和 session DTO；
- state → JSON-compatible dict、dict → state；
- legacy session key migration 和 default merge；
- parameter snapshot schema/validation；
- in-situ record JSONL encode/decode。

### 11.6 建议迁移顺序

1. 保存当前最小、完整、旧版本和损坏 session fixtures，固定 JSON keys/default/单位。
2. 将纯 serialization/migration 与“把状态应用到 widget/重新加载文件”的 effect 分开。
3. 通过现有 `SettingsRepository` / `SessionRepository` adapter 访问 legacy JSON，不改变格式。
4. FittingViewModel 持有 typed state；controller 只做 legacy widget/state 映射。
5. AI 与 in-situ state 随各自 use case 渐进迁移，不一次合并所有存储。

### 11.7 现有测试覆盖

`tests/test_insitu_ai_compatibility.py::test_ai_session_settings_migrate_without_breaking_old_sessions` 覆盖旧 AI session 设置迁移。未发现完整 fitting session round-trip 或恢复后 GUI 状态测试。

缺口：最小/完整/旧版 session、缺失/未知 key、损坏 JSON、文件已不存在、stack/NXS 恢复、ROI/checkbox/display option、parameter snapshot、in-situ cache 恢复，以及不启动 QApplication 的 serializer 测试。

## 12. Qt widget、window 和 worker 辅助类

### 12.1 当前方法和类

| 类/方法组 | 当前职责 |
|---|---|
| `NoWheelDoubleSpinBox`（264–268） | 禁止滚轮改变 spinbox |
| `CurrentPageHeightStackedWidget`（271–304） | 根据当前 page 调整高度 |
| `ManualAutoRefineWorker`（352–386） | QThread 包装 manual refine，转发 progress/result/error |
| `RefineUiBridge`（389–394） | refine UI signal bridge |
| `InsituBatchImageLoader`（397–483） | QThread 内准备远程文件并加载 batch image |
| `InsituCutWorker`（486–628） | QThread 内执行 cut 数学 |
| `IndependentMatplotlibWindow`（631–1,888） | 2D Matplotlib window、显示设置、center/selection mouse interaction |
| `IndependentFitWindow`（1,891–2,332） | 1D/fit Matplotlib window、unit/filter/y-range/point deletion controls |
| `UnifiedDisplayManager`（2,335–2,491） | 同步 GUI 和独立窗口的 1D plot |
| `FolderImageScanWorker`（2,689–2,740） | QThread 扫描目录 |
| `AsyncImageLoader`（2,743–3,021） | QThread 读取 CBF/NXS/TIFF/stack 和 remote cache |
| controller particle widget helpers（12,204–13,278） | dynamic page/widget/checkbox 创建、style、range、signal、context menu 和删除 |
| controller fitting text browser helpers（11,921–12,134） | log widget、context menu、detach、trim、save |

### 12.2 输入和输出

- widget/window 输入是 Qt parent、controller callbacks、NumPy arrays 和 display state；输出是 widget 状态、signals 和 rendered artists。
- worker 输入包含 callable/path/arrays/settings；输出通过 Qt signal 传 progress/result/error。
- 当前 worker payload 经常包含 controller callback 或 NumPy/Qt object，不等同于 Job 系统要求的可序列化消息。

### 12.3 UI 依赖

本组全部属于 presentation 或 legacy Qt bridge。即便 worker 中有纯计算/I/O，QThread wrapper 也不能进入 domain/application core。

### 12.4 `global_params` 依赖

`IndependentMatplotlibWindow` 直接读写 `fitting.gisaxs_input.show_cut_region/show_center/colormap`，并读取 detector center、q-axis geometry；其他 worker 大多不直接访问 singleton，但通过 controller settings 间接依赖。

### 12.5 可优先抽出的纯函数

- worker 内部的 I/O 决策和 cut 数学按前述 ports/domain 提取；
- window 的 display-data preparation、coordinate transform 和 hit-test geometry 可成为独立纯函数；
- widget 创建、style、event 和 signal bridge 保留 presentation，不为了行数强行拆成 domain service。

### 12.6 建议迁移顺序

1. 不先移动 helper class；先让它们调用已提取的 domain/use case/adapter。
2. 用 `JobRunner` 替代长计算/外部进程细节，但保留 Qt signal adapter 作为兼容桥。
3. Independent windows 消费 ViewModel display state，不直接读取 `global_params`。
4. particle widget 和 text browser 按明确 presentation 职责拆文件；controller 仅连接 legacy signals/navigation。
5. 最后确认没有动态 `getattr`/signal 调用后再删除旧 helper。

### 12.7 现有测试覆盖

现有覆盖主要是 `AsyncImageLoader` 的 stack 内部行为、graphics view reset，以及少量通过 `__new__` 构造 controller/helper 的轻量测试。没有系统覆盖 Qt helper 生命周期、signal、取消、worker 崩溃、window close、center/selection interaction 或主/独立窗口同步。

## 13. 推荐的 characterization tests

以下测试应在对应迁移步骤开始前建立；它们用于冻结现状，不表示本轮已经实现。

### P0：第 12B 前必须补齐

1. **Image transforms golden arrays**：threshold、NaN、flip、mirror gap、margin、空/全 mask，比较完整输出数组。
2. **ROI mapping table**：正 q、负 q、跨零、log-x、empty/NaN；固定 control↔data round-trip 和最终 mask。
3. **Cut numerical fixtures**：horizontal/vertical × pixel/q mode × finite/NaN；记录 q、I、shape、orientation、metadata。
4. **普通/in-situ cut 差异**：显式包含 NaN 的 fixture，分别锁定 `finite_mean_axis` 与 `np.mean` 的当前输出。
5. **Sort/filter/resample**：duplicate q、unsorted q、positive/negative-only、各 interpolation、points clamp；锁定处理顺序。
6. **Manual model fixtures**：每种当前支持的 particle/sequence 用固定 q 和参数记录总曲线与 component curves。
7. **Constraints/parameter fixtures**：alias、defaults、bounds、units、global/particle 参数顺序和 invalid cases。
8. **Scoring/refine fixtures**：linear/log score、auto-K、manual refine 初值/bounds/epsilon/停止；固定容差和 SciPy 版本范围。

### P1：第 12C–12D 前必须补齐

1. **File matrix**：CBF、NXS 2D/3D、TIF/TIFF、ordinary stack、1D DAT/TXT；固定 frame/stack/orientation/dtype/metadata。
2. **File failures**：不存在、损坏、unsupported、权限、空文件、缺少 dataset、remote cache copy 失败；期望结构化错误。
3. **Export golden files**：fitting TXT/CSV、parameter JSON、in-situ CSV；固定 header、delimiter、precision、q unit 和 key/column 顺序。
4. **Session round-trip**：最小、完整、旧 AI session、missing/unknown keys 和 lost source file。
5. **FittingViewModel transitions**：idle → loading → curve-ready → manual-fit-running → result/error；无需 QApplication 和真实配置。

### P2：第 12E–12G 前必须补齐

1. **AI request snapshot**：Fast/Balanced/Exhaustive 的 seed、constraints、CLI/port request 和 curve preprocessing。
2. **Fixed candidate fixture**：generation result、physical verification、score、rank/tie-break、candidate→params、preview 和 refinement。
3. **AI job behavior**：progress、success、structured error、cancel、timeout、worker crash；application test 使用 fake `Predictor`。
4. **Three-file in-situ workflow**：同一 fitting use case、顺序、进度、第二个文件失败继续、取消、序列化/恢复和聚合输出。
5. **Compatibility call audit**：旧 signal、dynamic `getattr`、session restore 和 prediction 对 `AsyncImageLoader` 的 legacy import 均有测试后再删方法。
6. **Offscreen GUI smoke**：主窗口启动、加载小图、cut、manual fit、AI error 不退出、关闭窗口；不比较视觉样式，只验证关键 widget/state/无崩溃。

## 14. 建议的实际拆分顺序

在不改变用户给定阶段顺序的前提下，文件内部推荐按以下依赖顺序迁移：

1. **第 12B-1**：顶层 image transforms 和 ROI sort/filter/interpolation；
2. **第 12B-2**：cut kernels 与 geometry 值对象，保留普通/in-situ 差异；
3. **第 12B-3**：particle/global parameter structures、constraints、unit conversion、manual evaluation/scoring/refine objective；
4. **第 12C**：typed scattering/curve/result、repository ports、现有格式 adapters、load/export use cases；
5. **第 12D**：`FittingState` 与第一版 ViewModel，只接管文件、当前曲线和 manual state；
6. **第 12E**：AI request、Predictor port、verification/ranking/refinement、JobRunner progress；
7. **第 12F**：in-situ serializable workflow，复用 file/cut/fitting use cases；
8. **第 12G**：以测试覆盖和调用审计为删除条件，清理重复 legacy 实现。

每一项都应遵循：先加 characterization test → 引入 seam → 新实现 → 旧 caller 兼容调用 → 对比结果 → 才删除旧实现。不得为了缩短文件而同时迁移不相关职责。

## 15. 当前测试覆盖结论

现有测试最强的区域是 stack 导航/求和、少量 image transforms、q filtering/interpolation、AI profile/constraint/model registry 和底层 random-cylinder forward model。最弱的区域是：

- manual fitting 与 auto-K/refinement 的 controller 级数值回归；
- 完整 horizontal/vertical、pixel/q cut；
- 文件错误和 1D loader；
- fitting/export golden files；
- 完整 session restore；
- AI generation → verification → ranking → refinement 固定 fixture；
- 三文件 in-situ、取消、错误继续和恢复；
- 独立窗口、主窗口和 worker 生命周期的离屏 GUI 行为。

因此第 12B 不应从 Qt 大类或 AI/in-situ 开始。首个安全 seam 应是已接近纯函数、已有部分测试、输入输出可用 NumPy 数组明确表达的 image/ROI/cut/scoring 代码。
