# Presentation

WAXS PyQt/Matplotlib 页面、typed UI state 与 commands。Qt dialogs、interactive selection、
控件状态和结果渲染位于本层；ViewModel 调用 application use cases。文件读取、路径实现、
科学计算和 worker process 实现不得进入本层。

静态页面布局由 `views/` 下按 page、toolbar、preview、configure、ROI、integration、
advanced 和 batch 职责拆分的独立 Python View 维护。View 只创建控件、布局、objectName
与静态展示属性；`page.py` 负责注入 ViewModel/Matplotlib viewer、连接信号并保留交互行为。
