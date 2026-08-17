# WAXS 布局迁移记录

## 控件映射

| 迁移前控件/区域 | 迁移后位置 | 行为 |
| --- | --- | --- |
| Open File、Reload、NXS frame | `Input / ParameterSection` | loader、supported extensions、frame indexing 和 drag/drop 不变 |
| toolbar auto/log/colormap | `Input` 快速显示控件 | 与 Advanced Display 原双向同步保留 |
| detector/curve viewer、metadata、2D/1D switch | `Preview / PlotPanel` | image orientation、q extent、overlay、curve rendering 不变 |
| ROI/Cut、1D Integration tabs | `Configure / ParameterSection` | Q range、line/circle cut、binning、smoothing 和 axis mode 不变 |
| Display、Mask、Geometry tabs | `Advanced display, mask and geometry / AdvancedSection` | 默认折叠；单位、默认值、mask threshold 和 q-map 几何不变 |
| Batch/In-situ input、export selection、start/pause/stop | `Run / ParameterSection` | 仍调用 `WaxsViewModel` 与 JobRunner batch adapter |
| load/batch status 与 progress | shared `JobStatus` | 旧 `status_label`/`progress` 别名保留，百分比仍为 0–100 |
| latest integration status | `Results / ParameterSection` | curve point count 和 completion message 不变 |
| Export Image、Export 1D | `Export / ParameterSection` | 复用原按钮实例和 exporter adapters |

本轮没有修改 WAXS image loading、q-map geometry、masking、cut、integration、batch、export
domain/application 代码，也没有修改 `WaxsViewModel`。原 `waxsControlTabs` 仍保留给 Basic
配置，Display/Mask/Geometry 移入单独的 Advanced tabs；所有业务控件实例和 signal connection
保持不变。

## 手动验收清单

- [ ] Open/Reload、TIFF/NXS、frame selector 和 detector drag/drop 正常；
- [ ] toolbar 与 Advanced Display 的 auto/log/colormap 双向同步；
- [ ] Advanced 折叠/展开不改变 vmin/vmax、mask 或 geometry 值；
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
