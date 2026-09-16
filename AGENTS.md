# GIMaP 项目说明

GIMaP 是基于 Python / PyQt5 的 GISAXS、WAXS 等散射数据分析桌面软件。

- `src/gimap/features/<feature>/` 拥有各功能的 UI、工作流、科学逻辑和适配器。
- `src/gimap/app/` 拥有应用壳和跨功能 UI 组件；`src/gimap/shared/` 放稳定复用能力。
- 旧顶层目录主要提供兼容入口，新增实现优先放在当前 feature owner。
- 显示优化应保持科学数值、单位、数组方向和数据谱系；相关契约见
  `docs/architecture/scientific-data-flow.md`。
- 保留用户已有的 tracked / untracked 修改；未经要求不 commit 或 push。
- 修改后运行相关 focused tests；UI 使用 `QT_QPA_PLATFORM=offscreen` 验证。
  完整检查入口为 `python tools/check.py`，环境限制应在交付中说明。
