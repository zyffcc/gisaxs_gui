# Python View UI source of truth

- **Status**: Current
- **Scope**: application shell、workspace pages、dialogs、windows、QSS
- **Related code**: `src/gimap/app/presentation/`、`src/gimap/features/*/presentation/`
- **Related tests**: `tests/test_ui_source_of_truth.py`、`tests/test_ui_workspace_layouts.py`
- **Last verified**: 2026-08-18

## 决策

GIMaP 使用 feature-owned、手工维护的 Python View 作为界面唯一事实来源，不再使用 Qt
Designer `.ui`、pyuic 生成模块或 UI 编译步骤。

```text
views/<page>_view.py
        │ widget hierarchy / layout / objectName
        ▼
page.py / dialog.py
        │ dependency injection / signal binding / rendering
        ▼
ViewModel → application use case → ports
```

应用外壳的 View 位于：

```text
src/gimap/app/presentation/views/
```

Feature View 位于：

```text
src/gimap/features/<feature>/presentation/views/
```

## 文件职责

`views/*_view.py` 只负责：

- PyQt widget hierarchy 与 layout；
- objectName、tab order、默认值和静态展示属性；
- 为 Matplotlib、动态图表或数据驱动控件提供命名明确的 host；
- 必要的 feature-local 视觉组件引用。

View 禁止导入 ViewModel、application、domain、infrastructure、controller、文件系统实现、
TensorFlow、BornAgain 或科学工作流。View 不显示 `QMessageBox`、不打开 `QFileDialog`、不执行
计算和文件读写。

`page.py`、`dialog.py` 或 `window_view.py` 负责：

- 实例化 View；
- 通过构造函数注入 ViewModel/use case；
- 将 Qt signals 映射为 commands；
- 将 state/result 转为控件展示；
- 安装 Matplotlib canvas、toolbar 和其他运行时组件；
- 在需要时保留薄兼容属性，但不得复制静态布局。

QSS 优先放在 owner 的 `presentation/styles/` 或应用级 design system 中。状态驱动的少量动态
样式可以留在 presentation behavior，但禁止在多个页面复制整套 stylesheet。

## 动态组件

以下内容可以由 `page.py`/`dialog.py` 或职责明确的 presentation component 动态创建：

- Matplotlib canvas 与 toolbar；
- 数量由模型、module.yaml 或 plugin 决定的参数编辑器；
- 大型结果表的 model/delegate；
- 当前工作流临时生成的结果 dialog；
- 运行时 plugin 提供的控件。

当动态组件的结构稳定、需要独立重做或持续增长时，应提取为单独的 Python View/component，
而不是塞回大型 ViewBinding。

## 当前所有权

| Owner | Python Views | 运行时注入边界 |
| --- | --- | --- |
| app | main shell、Display Settings | feature pages、navigation state |
| Format Converter | main、folder import、progress | preview、conversion worker state |
| Calibration | calibration dialog | Matplotlib figure、candidate overlays |
| Classification | page + dataset/preprocessing/experiment/results/inspection panels | 数据驱动表格与图 |
| WAXS | page + toolbar/preview/configure/ROI/integration/advanced/batch | scattering viewer、Matplotlib canvas |
| Trainset | shell + Dataset/Preview/Model/Run/Monitor 五步页面 | catalog/plugin 字段、交互画布、JobStatus |
| Prediction | controls、workspace、multi-file results、export/heatmap/trend dialogs | model-driven controls、Matplotlib canvas |
| Fitting | controls、workspace、detector dialog、两个独立绘图窗口 | 动态模型参数、Matplotlib canvas/toolbar |

当前共有 37 个显式 Python View。`tests/test_ui_source_of_truth.py` 维护完整 owner inventory，并
阻止 `.ui`、`_generated`、pyuic 标记和非法 runtime/workflow 依赖重新进入仓库。

## 修改页面的标准流程

1. 确认页面 owner 与对应 `views/*_view.py`；
2. 只在 View 中修改控件、布局和视觉默认值；
3. 在 `page.py`/`dialog.py` 中连接交互，不把业务逻辑放进 View；
4. 审计并明确记录 objectName、signals、快捷键、默认值和 tab order 的变化；
5. 为 View、ViewModel 状态转换与 public import aliases 运行 focused tests；
6. 运行离屏页面测试和 `python tools/check.py`；
7. 新增、删除或重命名 View 时同步显式 inventory 与 workspace 文档。

同一页面禁止同时存在 `.ui`、生成 Python 和手写 Python View。禁止重新建立包含所有 feature
页面的 monolithic 主窗口文件。
