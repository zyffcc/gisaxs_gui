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
- 最近验证：2026-08-30。

## 当前现代化工作流

主画布继续固定在左侧，右侧控制栏从一条混合长表单拆为三个明确 workspace：

```text
Load data
    ↓
1 Cut + integrate | 2 Advanced | 3 Batch
    ↓
Preview / Results / Export
```

`Cut + integrate` 保留 ROI/Cut 与 1D Integration 子页签，并在 ROI/Cut 顶部提供 Pixel / q-space
预览切换和只读 detector geometry 摘要（SDD、pixel size、beam center、wavelength、incidence）；
`Advanced` 直接显示 Display、Mask、Geometry，不再增加一层折叠；`Batch` 独立承载多数据源表格、
export root、四类导出、出版图外观、可选 q 显示范围和 job status。单文件 integration 与文件夹
batch 仍调用同一组 application commands。

q-space preview 与 2D q export 使用逐 detector cell 的 signed `qr` / `qz` 网格，不使用
`imshow extent` 把非规则 q 网格拉伸为规则矩形。默认 q export 显示 detector 可达的完整范围；
只有用户勾选 `Limit the exported q view` 后才应用 `qr/qz min/max`。出版图预设采用 300 dpi、
大号坐标轴和刻度、带单位的 colorbar，并允许直接配置 colormap、Linear/Log10、Auto limits 或
Vmin/Vmax。Pixel 与 q-space 预览都只对 detector 图像覆盖区域内部的 NaN 做显示插值；图像覆盖
区域之外保持为空，尤其保留 q-space 中 `qr≈0`、高 `qz` 的中央空洞。Display 中可为这类无数据
背景选择白色或黑色 `No-data color`。Signed `Qr` 的负、正分支分别绘制，不允许网格单元跨越
`Qr=0` 的不连续接缝，禁止在高 `Qz` 中央空洞形成伪插值条纹。
`Preview export style` 把这些选项投影到当前 2D preview；它只修改 display state。

Batch 的每一行表示一个输入文件夹，可分别编辑 glob pattern 和输出子目录。输出子目录默认使用
输入文件夹名，最终路径为 `<export root>/<output subfolder>/`，其下按 `2D_pixel/`、`2D_q/`
和 `1D/` 分类。输出子目录必须是简单目录名且在同一任务中唯一，避免不同数据源互相覆盖。

Batch `Preprocessing` 位于 q conversion 与运行命令之间，默认关闭：

- `Calibrate SDD`：默认参考峰 `q=2.132 Å⁻¹`、搜索半宽 `±0.035 Å⁻¹`。在当前 q-range/cut
  生效后使用未 log 的 1D 曲线寻找窗口内最高峰，按 `D_new = D_old × q_peak / q_target`
  更新 SDD，最多迭代三次后重新积分；该帧的 2D q export 使用校准后的 geometry；
- `Normalize peak`：默认参考峰 `q=2.132 Å⁻¹`、搜索半宽 `±0.035 Å⁻¹`、目标强度 `1`，使用
  未 log 峰值计算 `factor = I_target / I_peak`，并把整幅 AnalysisImage 与对应 1D curve 同乘
  该系数；
- `One factor per group (first frame)` 对 Batch data-source 表格中的每一行分别保存首个成功帧的
  系数并用于该组后续帧；`Independent factor per frame` 为每一帧独立求系数，不跨文件或 frame
  复用。Calibration 与 Normalization 仅支持 q integration axis；找不到窗口内有效正峰时该帧失败，
  并遵循现有 continue-on-error 行为。

`Preview preprocessing` 使用 Batch 表格当前选中的 data-source 行；`Preview item` 是该行按排序后
文件及 NXS 内部 frames 展开的 1-based 序号。预览在后台调用与正式 Batch 相同的 application
preprocessing use case，并用 Batch publication appearance、q-range 和当前 no-data color 投影到主
Preview，因此 colorbar、校准后 SDD 和 normalization factor 可以在运行前检查。Group-first 模式
预览第 N 张时会先从该组第一张求系数，再把同一系数应用到所选帧。

所有具名 WAXS checkbox、combo、数值输入、路径/模式输入及 Batch data-source 行自动保存到应用
settings JSON 的 `waxs` section，并在下次构造页面时恢复。`Load WAXS config...` / `Save WAXS
config...` 可显式读写版本化 JSON 快照；加载后同步更新依赖控件 enabled state 并写回自动记忆。
可移植配置文件通过 application-owned `WaxsConfigurationPort` 和 local JSON adapter 读写；当前
schema version 为 `1`，包含具名参数值与 Batch source rows。Batch worker payload 同步携带
`normalization_target_intensity`，因此 GUI 进程内与后台 JobRunner 使用相同目标强度。

出版图设计参考 TUM INSIGHT 的 reciprocal-space plotting 与 batch/export 工作流：
[Reus et al., J. Appl. Cryst. 57 (2024)](https://doi.org/10.1107/S1600576723011159)。

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
| ROI/Cut、1D Integration tabs | `1 Cut + integrate` | Pixel/q-space preview、geometry 摘要、Q range、line/circle cut、binning、smoothing 和 axis mode |
| Display、Mask、Geometry tabs | `2 Advanced` | 独立 workspace；可选 White/Black no-data 背景，单位、默认值、mask threshold 和 q-map 几何不变 |
| Batch data sources | `3 Batch / Data sources` | 多行 folder、pattern、output subfolder；新增行默认以输入文件夹名作为输出子目录 |
| Batch export selection | `3 Batch / Outputs` | 2D pixel PNG、2D q PNG、1D CSV、1D PNG 可独立选择 |
| Batch publication appearance | `3 Batch / Publication appearance` | colormap、Log10、Auto/Vmin/Vmax；可显式预览或从当前 preview 复制 |
| Batch q conversion | `3 Batch / q conversion` | 显示 SDD/pixel/center 等 geometry；可选 qr/qz 显示范围 |
| Batch preprocessing | `3 Batch / Preprocessing` | 可选 SDD peak calibration；可选按 data-source 首帧或逐帧的未 log peak normalization |
| Batch preprocessing preview | `3 Batch / Preprocessing` | 选择当前 data-source 行和展开后的 item 序号，以正式 Batch 算法预览图像、colorbar、SDD 与 factor |
| WAXS configuration | `3 Batch / Data sources` | 自动记忆到 settings JSON；显式 Save/Load 版本化 WAXS JSON 配置 |
| Batch start/pause/stop | `3 Batch` | 独立 workspace；仍调用 `WaxsViewModel` 与 JobRunner batch adapter |
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
- [ ] Pixel/q-space 切换使用真实 q 网格；incidence、center、distance、pixel size、wavelength 摘要与 Geometry 同步；
- [ ] Pixel/q-space 的图像覆盖区内部 NaN 可显示插值；覆盖区外及 `qr≈0`、高 `qz` 空洞保持背景，不出现伪插值条纹；White/Black 可切换；
- [ ] mask min/max、bad-pixel threshold、Apply/Reset Mask 正常；
- [ ] radial/azimuthal integration、bins、smoothing、q/pixel/2theta 输出一致；
- [ ] Results 显示正确 curve point count；
- [ ] Export Image 根据当前 Pixel/q-space 模式写出正确坐标、单位和 array orientation；
- [ ] Batch 多行 folder/pattern/output subfolder 可编辑，重复或非法 output subfolder 会就地报错；
- [ ] 四类 batch output 可独立选择，默认输出位于 export root 下以输入文件夹名命名的子目录；
- [ ] q export 未限制时显示完整范围，限制后使用指定 qr/qz 范围；
- [ ] Batch Calibration 默认 2.132±0.035 Å⁻¹，校准后的 SDD 用于该帧 q export 与最终积分；
- [ ] Batch Normalization 默认 2.132±0.035 Å⁻¹、目标强度 1；Group first 与 Per frame 模式分别复用或独立计算未 log 峰值系数，2D/1D 同比例缩放；
- [ ] 选中 Batch group 并指定 Preview item 后，可在主 Preview 检查 preprocessing 图像与 colorbar；Group-first 非首帧预览复用首帧系数；
- [ ] 所有 WAXS 参数和 Batch source rows 重启后从 settings JSON 恢复；显式 Save/Load JSON 往返保持参数、模式和默认 enabled state；
- [ ] 2D/1D PNG 的轴标题、刻度、colorbar 和 300 dpi 出版预设正确，外观 preview 不触发 scientific command；
- [ ] Batch Start/Pause/Resume/Stop、continue/error behavior 和 JobStatus 正常；
- [ ] 单个坏文件不会导致 GUI 进程退出；
- [ ] calibration 写入的 geometry 在 WAXS 页面继续正确同步。

## Interactive detector inspection

The detector toolbar's **Interactive** action opens the shared pixel viewer. Wheel zoom and drag pan
stay local to this window. **ROI** exposes a movable rectangle; **Apply ROI** uses the existing WAXS
Q-range selection workflow. Center and cut outlines track the workspace. Log intensity and histogram
limits update the workspace display controls; right-click gradient presets are inspection-only and
exports keep the workspace colormap. Cursor intensity uses full-resolution, unlogged source values.

For multi-frame NXS input, **Play/Pause** requests frames through the existing loader without queuing
multiple reads. Closing the window or leaving the pixel detector view stops playback. Q-space remains
available in the main detector view; switching to q-space hides stale data in the interactive window.
