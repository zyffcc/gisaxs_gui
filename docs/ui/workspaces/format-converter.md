# Format Converter 布局迁移记录

- **Status**: Current
- **Scope**: Format Converter 的 PyQt presentation 所有权、控件映射与手动验收
- **Related code**:
  [`src/gimap/features/format_converter/presentation/`](../../../src/gimap/features/format_converter/presentation/)、
  [`format_converter_dialog_view.py`](../../../src/gimap/features/format_converter/presentation/views/format_converter_dialog_view.py)、
  [`ui/format_converter_dialog.py`](../../../ui/format_converter_dialog.py)
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

旧文件 `ui/format_converter_dialog.py` 只保留 9 行 import compatibility，直接 re-export
feature 的三个 dialog class，不包含第二套实现。生产 caller `ui/menu_manager.py` 已直接按需
导入 feature-owned dialog；旧路径仅服务尚未迁移的外部脚本和回归测试。

## 控件映射

| 旧实现中的控件/区域 | Python View object / feature-owned 位置 | 保持的行为 |
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

没有重命名现有 Python 控件属性；现有属性现在对应 Python View 中清晰可见的 objectName。旧实现
没有自定义快捷键，本次没有新增或改变。输入过滤器、四种输出格式、metadata JSON、frame selection、dtype
conversion、progress/pause/cancel 和错误文案保持原行为。

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
