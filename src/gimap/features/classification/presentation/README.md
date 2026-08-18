# Presentation

本目录拥有 Classification 的 PyQt 页面、展示状态和 ViewModel。
`views/classification_page_view.py` 拥有页面 shell、workflow 导航、workspace sections 和日志区；
同目录五个独立 Python panel Views 分别拥有 dataset、inspection、preprocessing、experiment
和 results 静态控件。`page.py` 组合这些 Views。页面只负责控件、信号、拖放、响应式布局和状态渲染；
ViewModel 只调用 use cases，不直接训练、读写文件或管理进程。旧 controller 暂时作为 Qt
信号桥，不能在此之外继续承担业务编排。动态 class cards 是运行时模型集合，继续通过
`datasetCardsLayout` host 添加。
