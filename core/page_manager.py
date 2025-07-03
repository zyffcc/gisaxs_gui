"""
页面管理器 - 处理不同页面的大小和布局
"""

from PyQt5.QtWidgets import QSizePolicy
from PyQt5.QtCore import QTimer


class PageManager:
    """页面管理器，处理页面切换和大小调整"""
    
    def __init__(self, main_window):
        self.main_window = main_window
        self.ui = main_window
        self._setup_page_management()
    
    def _setup_page_management(self):
        """设置页面管理"""
        if hasattr(self.ui, 'mainWindowWidget'):
            # 连接页面切换信号
            self.ui.mainWindowWidget.currentChanged.connect(self._on_page_changed)
            
            # 设置stackedWidget的大小策略
            self.ui.mainWindowWidget.setSizePolicy(
                QSizePolicy.Expanding, 
                QSizePolicy.Expanding
            )
    
    def _on_page_changed(self, index):
        """页面切换时的处理"""
        # 延迟处理，确保页面完全加载
        QTimer.singleShot(10, lambda: self._adjust_page_layout(index))
    
    def _adjust_page_layout(self, index):
        """调整页面布局"""
        try:
            current_widget = self.ui.mainWindowWidget.widget(index)
            if not current_widget:
                return
            
            # 根据页面类型进行不同的处理
            if index == 0:  # 训练集构建页面
                self._setup_trainset_page(current_widget)
            elif index == 1:  # GISAXS预测页面
                self._setup_predict_page(current_widget)
            
            # 更新布局
            current_widget.updateGeometry()
            self.ui.mainWindowWidget.updateGeometry()
            
        except Exception as e:
            print(f"页面布局调整失败: {e}")
    
    def _setup_trainset_page(self, page_widget):
        """设置训练集构建页面"""
        # 设置训练集页面为可扩展
        page_widget.setSizePolicy(
            QSizePolicy.Expanding, 
            QSizePolicy.Expanding
        )
    
    def _setup_predict_page(self, page_widget):
        """设置GISAXS预测页面"""
        # 设置预测页面的大小策略
        page_widget.setSizePolicy(
            QSizePolicy.Preferred, 
            QSizePolicy.Preferred
        )
        
        # 如果有特定的控件需要调整，在这里处理
        self._adjust_predict_widgets(page_widget)
    
    def _adjust_predict_widgets(self, page_widget):
        """调整预测页面的控件"""
        try:
            # 查找并调整图像显示区域
            if hasattr(self.ui, 'gisaxsPredictImageShowTabWidget'):
                tab_widget = self.ui.gisaxsPredictImageShowTabWidget
                tab_widget.setSizePolicy(
                    QSizePolicy.Expanding, 
                    QSizePolicy.Expanding
                )
            
            # 调整参数区域的大小策略
            if hasattr(self.ui, 'widget') and hasattr(self.ui.widget, 'parent'):
                # 这里的widget是predict页面中的参数控制区域
                for child in page_widget.findChildren(object):
                    if hasattr(child, 'setSizePolicy'):
                        # 对于输入控件，设置为首选大小
                        if 'LineEdit' in child.__class__.__name__ or 'ComboBox' in child.__class__.__name__:
                            child.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        
        except Exception as e:
            print(f"预测页面控件调整失败: {e}")
    
    def switch_to_page(self, page_index):
        """切换到指定页面"""
        if hasattr(self.ui, 'mainWindowWidget'):
            self.ui.mainWindowWidget.setCurrentIndex(page_index)
    
    def get_current_page_index(self):
        """获取当前页面索引"""
        if hasattr(self.ui, 'mainWindowWidget'):
            return self.ui.mainWindowWidget.currentIndex()
        return 0


# 创建全局页面管理器实例
page_manager = None


def initialize_page_manager(main_window):
    """初始化页面管理器"""
    global page_manager
    page_manager = PageManager(main_window)
    return page_manager


def get_page_manager():
    """获取页面管理器实例"""
    return page_manager
