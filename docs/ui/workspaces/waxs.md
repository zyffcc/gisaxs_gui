# WAXS 界面与交互说明

- 状态：嵌入式 WAXS 页面由 WAXS feature 拥有；顶层 UI 路径仅保留薄 public alias。
- 当前调用链：`InSituProcessingWidget → WaxsViewModel → application use cases → ports/adapters`。
- Python 页面源：`src/gimap/features/waxs/presentation/views/page_view.py`。
- Python 子 View：同目录下的 `toolbar_view.py`、`preview_panel_view.py`、
  `configure_panel_view.py`、`roi_panel_view.py`、`integration_panel_view.py`、
  `advanced_panel_view.py` 和 `batch_panel_view.py`。
- 页面行为与表单绑定：`src/gimap/features/waxs/presentation/page.py`。
- ViewModel：`src/gimap/features/waxs/presentation/view_model.py`。
- 兼容入口：`ui/waxs_page.py`。
- 独立入口：`WAXS/WAXS.py`（薄兼容启动器）→ `src/gimap/features/waxs/standalone.py` → 同一 feature page。
- 页面 workflow layout：`src/gimap/features/waxs/presentation/workflow_layout.py`。
- 页面样式：`src/gimap/features/waxs/presentation/waxs_theme.qss`。
- 最近验证：2026-08-19。

## 当前现代化工作流

主画布继续固定在左侧，右侧控制栏从一条混合长表单拆为三个明确 workspace：

```text
Load data
    ↓
1 Cut + integrate | 2 Advanced | 3 Batch
    ↓
Preview / Results / Export
```

`Cut + integrate` 保留 ROI/Cut 与 1D Integration 子页签；`Advanced` 直接显示
Display、Mask、Geometry，不再增加一层折叠；`Batch` 独立承载 folder、pattern、output、
export selections 和 job status。单文件 integration 与文件夹 batch 仍调用原有命令。

`ui.waxs_page` re-export 页面类和 `load_image_matrix` public API，但不包含页面或
文件读取实现。图像读取兼容函数由 WAXS infrastructure 拥有；路径规范化、工作目录和目录
检查经 application-owned port 注入 ViewModel。Application shell 直接导入 feature page，并在
固定的第 5 个 workspace slot（index 4）原位替换启动 host；不再通过追加顺序猜测 WAXS index。

## 控件映射

| 功能/控件区域 | 当前位置 | 行为 |
| --- | --- | --- |
| Open File、Reload、NXS frame | `Load data` | Open detector file 为主操作；loader、extensions、frame indexing 和 drag/drop 不变 |
| toolbar auto/log/colormap | `Input` 快速显示控件 | 与 Advanced Display 原双向同步保留 |
| detector/curve viewer、metadata、2D/1D switch | `Preview / PlotPanel` | image orientation、q extent、overlay、curve rendering 不变 |
| ROI/Cut、1D Integration tabs | `1 Cut + integrate` | Q range、line/circle cut、binning、smoothing 和 axis mode 不变 |
| Display、Mask、Geometry tabs | `2 Advanced` | 独立 workspace；单位、默认值、mask threshold 和 q-map 几何不变 |
| Batch/In-situ input、export selection、start/pause/stop | `3 Batch` | 独立 workspace；仍调用 `WaxsViewModel` 与 JobRunner batch adapter |
| load/batch status 与 progress | shared `JobStatus` | 旧 `status_label`/`progress` 别名保留，百分比仍为 0–100 |
| latest integration status | `Results / ParameterSection` | curve point count 和 completion message 不变 |
| Export Image、Export 1D | `Export / ParameterSection` | 复用原按钮实例和 exporter adapters |

页面布局和 presentation ownership 不修改 WAXS 图像 orientation、q-map geometry、masking、cut、integration、
batch 或 export 行为。静态 widget hierarchy、控件默认值、tab 文本和 tab order 现在由上述
Python Views 维护；`page.py` 不再保留 `_build_ui`、toolbar 或各 tab 的第二套静态实现。
Matplotlib canvas/toolbar 仍由 Python 创建并注入 `viewerHost`，因为它们是运行时组件。原
`waxsControlTabs` 仍保留给 Cut/Integration 配置，Display/Mask/Geometry 位于独立 Advanced workspace；
objectName、signal connection、快捷键和错误提示保持不变。`WaxsViewModel` 不操作 QWidget
或具体文件系统。

`Ui_MainWindow` 只预留无业务控件的 `waxsPageHost`。`MainWindowComponents` 创建 feature page
后在同一 index 替换并释放 host，因此导航、Controller 和测试看到的 `waxsPageIndex` 稳定为 4。

独立窗口不维护第二套 loader、geometry、integration、batch、export 或 Qt 页面。
`python WAXS/WAXS.py` 和 `MainWindow` 名称仍可用，由独立 composition root 创建 AppContext
并托管与主 GUI 相同的 `InSituProcessingWidget`。

## 手动验收清单

- [ ] Open/Reload、TIFF/NXS、frame selector 和 detector drag/drop 正常；
- [ ] toolbar 与 Advanced Display 的 auto/log/colormap 双向同步；
- [ ] Cut + integrate、Advanced、Batch 切换不改变 vmin/vmax、mask 或 geometry 值；
- [ ] Preview 的 2D/1D switch、colorbar、metadata 和 orientation 正确；
- [ ] Q Range rectangle、Line Cut、Circle Cut、Pick Center 和 Clear ROI 正常；
- [ ] q-map extent、incidence、center、distance、pixel size、wavelength 单位不变；
- [ ] mask min/max、bad-pixel threshold、Apply/Reset Mask 正常；
- [ ] radial/azimuthal integration、bins、smoothing、q/pixel/2theta 输出一致；
- [ ] Results 显示正确 curve point count；
- [ ] Export Image 与 Export 1D 文件格式、坐标和 array orientation 不变；
- [ ] Batch folder/pattern/output 和三个 export selections 正常；
- [ ] Batch Start/Pause/Resume/Stop、continue/error behavior 和 JobStatus 正常；
- [ ] 单个坏文件不会导致 GUI 进程退出；
- [ ] calibration 写入的 geometry 在 WAXS 页面继续正确同步。
