"""
布局工具 - 为每个页面都有ScrollArea的新UI结构提供基础布局支持
每个页面现在都有独立的ScrollArea，不再需要动态调整主滚动区域大小
"""

from PyQt5.QtWidgets import QSizePolicy
from PyQt5.QtCore import QTimer


class LayoutUtils:
    """布局工具类 - 适配新的UI结构（每个页面都有ScrollArea）"""
    
    @staticmethod
    def setup_adaptive_stacked_widget(stacked_widget):
        """设置自适应的StackedWidget"""
        if not stacked_widget:
            return
        
        # 由于每个页面都有自己的ScrollArea，主要设置StackedWidget的基本属性
        stacked_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        # 连接页面切换信号，进行基本的页面优化
        stacked_widget.currentChanged.connect(
            lambda index: LayoutUtils._on_page_changed(stacked_widget, index)
        )
        
        print("✓ StackedWidget已配置为自适应模式（每页面独立ScrollArea）")
    
    @staticmethod
    def _on_page_changed(stacked_widget, index):
        """页面切换时的基本处理"""
        try:
            current_page = stacked_widget.widget(index)
            if current_page:
                # 确保页面使用合适的大小策略
                current_page.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
                current_page.updateGeometry()
                
                # 根据页面索引显示切换信息
                page_names = {
                    0: "Cut Fitting",
                    1: "GISAXS Predict", 
                    2: "Trainset Build",
                    3: "Classification"
                }
                
                page_name = page_names.get(index, f"Page {index}")
                print(f"✓ 切换到 {page_name} 页面 (索引: {index})")
                
        except Exception as e:
            print(f"页面切换处理失败: {e}")
    
    # 为了向后兼容，保留一些原有方法的简化版本
    @staticmethod
    def _restore_full_scroll_area(stacked_widget):
        """向后兼容：页面滚动区域已独立，无需特殊处理"""
        print("ℹ️  每个页面都有独立ScrollArea，无需调整主滚动区域")
    
    @staticmethod
    def _compress_scroll_area_for_predict(stacked_widget):
        """向后兼容：页面滚动区域已独立，无需特殊处理"""
        print("ℹ️  每个页面都有独立ScrollArea，无需压缩主滚动区域")
