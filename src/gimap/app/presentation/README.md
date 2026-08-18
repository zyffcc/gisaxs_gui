# App presentation

这里保存由 GIMaP application shell 所有、可被多个 feature presentation 使用的通用
PyQt5 视觉组件。组件只负责布局、展示状态和发出 UI signals，不包含科学计算、文件读取、
JobRunner/process 管理或 feature workflow。

`CollapsibleCardFrame` 是 Fitting 与 Prediction 已稳定共用的持久化折叠卡片；它只保存视觉
展开状态和可选高度，不持有任何 feature state。

`NavigationSidebar` 只管理顶层页面选择、折叠状态和图标显示；它不知道任何 feature
workflow，也不导入 feature modules。

`ContentStack` 和 `MainShell` 管理页面索引、splitter 尺寸、响应式最小宽度及侧栏持久化，
同样不依赖 feature。Composition root 只负责把这些 shell primitives 与 feature pages 组装。

`layout_primitives.py`、`responsive_layout.py`、`assets.py` 和 `style_loader.py` 拥有应用级
布局、屏幕适配、图标和 QSS 加载。历史 `ui/` 同名路径仅作薄 re-export；新代码不得再从
这些 legacy 路径导入。

Display Settings dialog 由 `settings_dialog.py` 拥有，静态布局来源是
`views/settings_dialog_view.py`；`ui/settings_dialog.py` 只保留旧 import path。应用 shell 的
Python View 位于 `views/main_window_view.py`，只包含 sidebar、menu/status bar 和 workspace
hosts。`app/window_view.py` 是 composition wrapper，负责装配 Fitting/Prediction feature
controls；它不包含科学或业务流程。

顶层菜单需要创建 feature dialogs，因此由 `app/menu_manager.py` composition boundary
拥有，而不属于本共享 presentation 目录。它只通过注入的 SettingsRepository 与设置边界
协作，不直接访问 `global_params`。

Feature-specific widgets 和 ViewModels 仍应留在对应 feature。该目录不是通用业务代码或
scientific shared kernel，也不得被 application/domain 导入。
