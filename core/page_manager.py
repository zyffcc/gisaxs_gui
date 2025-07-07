"""
页面管理器 - 处理不同页面的基本布局设置
每个页面现在都有自己的ScrollArea，不再需要动态调整主滚动区域大小
"""

from PyQt5.QtWidgets import QSizePolicy
from PyQt5.QtCore import QTimer


class PageManager:
    """页面管理器，处理页面切换和基本布局设置"""
    
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
        """调整页面布局 - 根据新的UI结构更新页面索引"""
        try:
            current_widget = self.ui.mainWindowWidget.widget(index)
            if not current_widget:
                return
            
            # 根据新的页面索引分配：
            # 0: Cut Fitting页面, 1: GISAXS Predict页面, 2: Trainset Build页面, 3: Classification页面
            if index == 0:  # Cut Fitting页面
                self._setup_fitting_page(current_widget)
            elif index == 1:  # GISAXS预测页面
                self._setup_predict_page(current_widget)
            elif index == 2:  # 训练集构建页面
                self._setup_trainset_page(current_widget)
            
            # 每个页面现在都有自己的ScrollArea，只需要基本的设置
            print(f"✓ 切换到页面 {index}，页面布局已优化")
            
        except Exception as e:
            print(f"页面布局调整失败: {e}")
    
    def _setup_fitting_page(self, page_widget):
        """设置Cut Fitting页面"""
        # Cut Fitting页面有自己的ScrollArea，只需基本设置
        page_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        print("✓ Cut Fitting页面设置完成")
    
    def _setup_predict_page(self, page_widget):
        """设置GISAXS预测页面"""
        # GISAXS预测页面保持原有布局，不需要ScrollArea调整
        page_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        print("✓ GISAXS预测页面设置完成")
        
    def _setup_trainset_page(self, page_widget):
        """设置训练集构建页面"""
        # 训练集页面有自己的ScrollArea，只需基本设置
        page_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        print("✓ 训练集构建页面设置完成")
        
    def _setup_classification_page(self, page_widget):
        """设置Classification页面"""
        # Classification页面有自己的ScrollArea，只需基本设置
        page_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        print("✓ Classification页面设置完成")

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
