# Format Converter 界面与交互说明

- **Status**: Current
- **Scope**: Format Converter 的 PyQt presentation 所有权、控件映射与手动验收
- **Related code**:
  [`src/gimap/features/format_converter/presentation/`](../../../src/gimap/features/format_converter/presentation/)、
  [`format_converter_dialog_view.py`](../../../src/gimap/features/format_converter/presentation/views/format_converter_dialog_view.py)
- **Related tests**:
  [`tests/test_format_converter_presentation.py`](../../../tests/test_format_converter_presentation.py)、
  [`tests/test_format_converter_feature.py`](../../../tests/test_format_converter_feature.py)、
  [`tests/test_ui_workspace_layouts.py`](../../../tests/test_ui_workspace_layouts.py)
- **Last verified**: 2026-08-18

## 当前状态

Format Converter 主对话框的静态 widget hierarchy、布局、objectName、tab order 和默认视觉
属性以 feature-owned `presentation/views/format_converter_dialog_view.py` 为唯一来源。
`presentation/dialog.py` 继承 Python View，只注入 ViewModel、绑定信号、维护运行时状态并呈现
dialogs。调用链为：

```text
PyQt Dialog → FormatConverterViewModel → application use cases → ports
                                                       ↓
                                      infrastructure adapters
```

`QFileDialog`、`QMessageBox`、控件渲染和 worker signal 绑定留在 dialog。Frame selection
规则与输出格式可见性属于纯 domain 规则；路径规范化、目录扫描、preview 读取、输出估算
和转换经 ViewModel 调用 application use cases。ViewModel 不操作 QWidget。

`FolderImportDialog` 和 `ConversionProgressDialog` 的静态布局分别由
`views/folder_import_dialog_view.py` 与 `views/conversion_progress_dialog_view.py` 独立拥有，
与主对话框共用同一 behavior module，但不存在第二套布局实现。

旧的 `ui/format_converter_dialog.py` 与 `utils/format_converter.py` 兼容别名已删除；调用方直接
导入 `src.gimap.features.format_converter` 中的 owner 模块。

使用上的要点（2026-10-06）：帧选择模式按英文键保存在下拉项的数据里，中文界面下选择同样生效；
源列表各列按内容定宽、文件名列伸展，长路径中间省略（完整路径在提示里）；Open 的过滤器先列出所有
探测器格式（NXS、CBF、TIFF、EDF）。Esc 关闭窗口，转换运行中会先询问。

输出位置默认是第一个输入所在文件夹下的 `converted`（随输入变化，直到你自己选了位置）；上次的输出格式会恢复；
Add files / Add folder 从上次的文件夹开始。可以把文件或文件夹拖进窗口。预览：单帧的源只显示一张缩略图（多帧显示首 / 中 / 末），
统计（尺寸、类型、最小 / 最大、NaN、负值、可能饱和的像素）在缩略图上方；缩略图仅用于显示，取 log(1 + 值) 与 viridis 色图，
统计仍是存储的原始值。

## 控件映射

| 功能/控件区域 | Python View object / feature-owned 位置 | 行为 |
| --- | --- | --- |
| step 1 source actions、`input_tree`、dataset selector | `format_input_section` / Input | 原 signal、source model 和 dataset 选择不变 |
| step 2 tools、`selection_table` | `format_configure_section` / Configure | include、filter、sort、remove 不变 |
| frame mode/range/custom/Every N | `frame_advanced_section` | 默认折叠；值、默认 All、frame parsing 不变 |
| First/Middle/Last labels 与 statistics | `format_preview_panel` / Preview | 原 preview worker、orientation 和 statistics 不变 |
| output format、destination、naming | `format_output_section` | format 与命名规则不变 |
| dtype、metadata、container | `format_output_advanced` | 默认折叠；serialization 和默认值不变 |
| `output_summary`、Review & Convert | `format_run_section` | 原确认 dialog 和 conversion use case 不变 |
| conversion title/detail/progress/pause/cancel | shared `JobStatus` | 原 worker pause/cancel 信号不变 |
| succeeded/failed、Open folder、View report | conversion result area | 原 report path 和按钮不变 |

Python 控件属性对应 Python View 中清晰可见的 objectName。该 dialog 没有自定义快捷键。输入
过滤器、四种输出格式、metadata JSON、frame selection、dtype conversion、progress/pause/cancel
和错误文案是受测试保护的当前行为。

## 手动验收清单

- [ ] Add files、Add folder、Use current file 均可添加输入；
- [ ] NXS 多 dataset 选择仍能更新 frame/shape；
- [ ] Select all/none、filter、sort、remove 正常；
- [ ] 展开 Advanced 后 All、Current、Range、Custom、Every N 结果与原来一致；
- [ ] First/Middle/Last preview 的方向、dtype 和统计信息正确；
- [ ] TIFF、CBF、HDF5、NumPy 可选性与输入类型兼容；
- [ ] destination、命名样例、collision suffix 和 output estimate 正确；
- [ ] Advanced 折叠/展开不重置 dtype、metadata 或 container 值；
- [ ] Review & Convert 摘要的 input/output/frame count 正确；
- [ ] conversion 的 progress、Pause/Resume、Cancel 正常；
- [ ] 成功后 Open output folder、View report 可用；
- [ ] 关闭运行中的 progress dialog 仍受原安全限制。
