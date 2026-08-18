# Main Window 兼容边界地图

- **Status**: Current
- **Scope**: 应用 shell、Python View 与 feature presentation binding 的组合边界
- **Related code**: [`main.py`](../../main.py)、
  [`src/gimap/app/window_view.py`](../../src/gimap/app/window_view.py)、
  [`src/gimap/app/main_window.py`](../../src/gimap/app/main_window.py)
- **Related tests**: [`tests/test_ui_workspace_layouts.py`](../../tests/test_ui_workspace_layouts.py)、
  [`tests/test_fitting_presentation.py`](../../tests/test_fitting_presentation.py)
- **Last verified**: 2026-08-18

## 启动与组合顺序

```text
main.MainWindow
  → ApplicationWindowView.setupUi
      → feature-owned Fitting and Prediction control factories
      → empty Trainset, Classification and WAXS compatibility hosts
  → app.MainWindowComponents composition boundary
      → feature-owned Fitting and Prediction workspaces
      → feature-owned WAXS page
      → shell, navigation and responsive layout
  → ApplicationRuntime
      → construct feature ViewModels with AppContext/ports
      → construct Trainset/Classification/Prediction/Fitting ViewBindings
      → delayed feature initialization
```

`ApplicationWindowView` 的实现现位于 app composition boundary；`ui/main_window.py` 仅保留
`Ui_MainWindow` 旧导入别名。
它仍是兼容属性命名空间，不应继续接收 feature workflow。Application shell
可以构造 feature presentation，但不得把 scientific calculation、I/O 或 workflow
orchestration 放回主窗口。

Application shell 的唯一界面源是
`src/gimap/app/presentation/views/main_window_view.py`，只定义 sidebar、menu/status bar 与五个
workspace hosts。`src/gimap/app/window_view.py` 是 composition wrapper，通过 feature-owned
factories 替换 Prediction 和 Fitting 空 host；仓库不再存在 `.ui` 或 pyuic 生成链。

## Workspace 所有权与剩余 seam

| Workspace | 当前 presentation owner | 主窗口兼容边界 | 下一条安全 seam |
| --- | --- | --- | --- |
| Format Converter | feature dialog | 菜单直连 feature；`ui/format_converter_dialog.py` 薄 re-export | 确认外部脚本不再使用后删除旧 import path |
| Calibration | feature dialog | 菜单直连 feature；`ui/geometry_calibration_dialog.py` 薄 re-export | 同上 |
| Fitting | feature controls、workspace、ViewBinding | 旧 controller 名称为薄别名 | 按稳定 UI 状态组拆分超大 binding |
| Prediction | feature controls、workspace、ViewBinding | 旧 controller 名称为薄别名 | 按预览/模块/batch 状态组拆分 binding |
| Trainset | feature page、ViewBinding | 旧 controller 名称为薄别名 | 按 design/local/remote 状态组拆分 binding |
| Classification | feature page、ViewBinding | 无主窗口 widget aliases；旧 controller 为薄别名 | 按 dataset/results/rendering 状态组拆分 binding |
| WAXS | feature page | shell 固定 slot；旧独立路径为薄启动器 | 继续按稳定 UI 区块缩小超大 presentation，不改变科学行为 |

## 本轮缩小的 violation

Fitting 和 Prediction 的控件构造分别由各自 feature 的
`presentation/views/` 与 `control_view_factory.py` 拥有。主窗口只调用两个 feature factories；
objectName、默认值、父子层级、binding 读取的属性名、tab 顺序和页面 index 均保持不变。
这些工厂是隔离 legacy 生成代码的过渡边界，不包含 workflow、科学计算、文件 I/O、
TensorFlow 或 BornAgain。

Prediction 的 44 行页面文本、默认显示值和 tab 标题也已迁入同一 feature module；
`retranslateUi` 只调用 `translate_prediction_controls`。Application shell 自己的侧边栏标题仍由
主窗口翻译，不属于 Prediction feature 页面。

Fitting 的 104 行页面文本、模型选项和默认显示值也已迁入其 feature module；
`retranslateUi` 只调用 `translate_fitting_controls`。Application shell 的 Cut & Fitting
导航标题继续由主窗口翻译。

Classification 启动时不再构造随后会被运行时删除的旧控件。主窗口仅创建保持 page
index 的空 host；composition root 安装 feature page，ViewBinding 复用同一实例并直接通过
page API 访问 table、preview 和 log。所有主窗口 Classification widget aliases 及其隐藏占位
控件均已删除。

Trainset 启动时也不再构造随后会被隐藏的 2,022 行旧控件及其 246 行翻译。主窗口生成代码
仅保留 page index 为 0 的空 host；composition root 安装 feature-owned 页面，并将同一个
实例注入 `TrainsetViewBinding`。

WAXS 不再依赖 `addWidget()` 的运行时追加顺序。Shell 预留 index 4，composition root 直接
导入 feature page 并原位替换 host，随后释放占位对象。

Tools 菜单按需直接导入 Format Converter 与 Calibration 的 feature-owned dialogs；两个
`ui/` dialog 文件只作为外部兼容入口，不再处于生产调用链。

`MainWindowComponents` 的实现由 `app/main_window.py` composition boundary 拥有，`main.py`
直接导入新路径。这样共享 `app/presentation` 继续禁止依赖 feature，而组合边界可以装配
feature presentation。`ui/components/main_window_components.py` 仅 re-export 旧名称，避免
已有外部脚本和测试立即失效；它不再定义 shell 或 workspace class。

侧边导航现由 `app/presentation/navigation.py` 单独拥有。Composition root 只实例化
`NavigationSidebar`，不再定义导航 widget；导航模块保持 feature-free。

页面索引 facade 与顶层 splitter 分别由 `app/presentation/shell.py` 中的 `ContentStack` 和
`MainShell` 拥有。`app/main_window.py` 不再定义这些纯 shell classes，只承担 feature page
创建、注入和顶层装配。

`ApplicationRuntime` 的延迟启动先设置默认页面，再对四个 feature ViewBinding 各调用一次
`initialize()`。跨 workspace 参数和 Fitting session 分别由独立 coordinator 负责。

显示设置 dialog 同样由 `app/presentation/settings_dialog.py` 与
`app/presentation/views/settings_dialog_view.py` 拥有，生产入口不再导入
`ui.settings_dialog`。旧 `ui/` 文件仅为薄 re-export；历史 `ui/main_window.ui` 已删除。

Trainset 与 Classification ViewBinding 强制接收 composition root 创建的 page，不再读取主窗口
host、创建 fallback page 或清理 legacy widgets。BornAgain adapter 由 `main.py` composition
root 创建，再以 simulation port 注入；presentation 不选择或构造具体 simulation adapter。

## 渐进迁移规则

每次只处理一个 workspace 的一个 seam：先增加 feature-owned factory/page，再迁移 shell
caller，运行离屏与 workspace 回归测试，最后删除同一份 legacy 实现。禁止一次性重写
应用主窗口，也禁止在迁移 controls 时顺便改变科学行为或页面视觉设计。
