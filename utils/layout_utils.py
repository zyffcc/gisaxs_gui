"""
布局工具 - 动态调整滚动区域高度，解决页面空白问题
"""

from PyQt5.QtWidgets import QSizePolicy
from PyQt5.QtCore import QTimer


class LayoutUtils:
    """布局工具类 - 专注解决滚动区域高度问题"""
    
    @staticmethod
    def setup_adaptive_stacked_widget(stacked_widget):
        """设置自适应的StackedWidget"""
        if not stacked_widget:
            return
        
        # 连接页面切换信号，动态调整滚动区域
        stacked_widget.currentChanged.connect(
            lambda index: LayoutUtils._on_page_changed(stacked_widget, index)
        )
    
    @staticmethod
    def _on_page_changed(stacked_widget, index):
        """页面切换时动态调整滚动区域高度"""
        QTimer.singleShot(50, lambda: LayoutUtils._adjust_scroll_area(stacked_widget, index))
    
    @staticmethod
    def _adjust_scroll_area(stacked_widget, index):
        """根据页面类型调整滚动区域高度"""
        try:
            if index == 0:  # 训练集页面 - 恢复完整滚动区域
                LayoutUtils._restore_full_scroll_area(stacked_widget)
                
            elif index == 1:  # 预测页面 - 压缩滚动区域
                LayoutUtils._compress_scroll_area_for_predict(stacked_widget)
            
        except Exception as e:
            print(f"调整滚动区域失败: {e}")
    
    @staticmethod
    def _restore_full_scroll_area(stacked_widget):
        """恢复完整滚动区域高度（训练集页面）"""
        try:
            scroll_content = LayoutUtils._find_scroll_content(stacked_widget)
            if scroll_content:
                scroll_content.setMinimumHeight(2500)
                scroll_content.resize(scroll_content.width(), 2500)
                print("✓ 训练集页面：恢复完整滚动区域")
        except Exception as e:
            print(f"恢复滚动区域失败: {e}")
    
    @staticmethod
    def _compress_scroll_area_for_predict(stacked_widget):
        """压缩滚动区域（预测页面）"""
        try:
            predict_page = stacked_widget.widget(1)  # 获取predict页面
            if predict_page:
                # 计算predict页面实际需要的高度
                actual_height = LayoutUtils._calculate_content_height(predict_page)
                target_height = max(600, actual_height + 100)  # 最小600，加100边距
                
                # 调整滚动区域
                scroll_content = LayoutUtils._find_scroll_content(stacked_widget)
                if scroll_content:
                    scroll_content.setMinimumHeight(target_height)
                    scroll_content.resize(scroll_content.width(), target_height)
                    print(f"✓ 预测页面：压缩滚动区域至 {target_height} 像素")
        except Exception as e:
            print(f"压缩滚动区域失败: {e}")
    
    @staticmethod
    def _find_scroll_content(stacked_widget):
        """查找滚动区域的内容widget"""
        try:
            parent = stacked_widget.parent()
            while parent:
                if hasattr(parent, 'objectName') and 'ScrollAreaWidgetContents' in str(parent.objectName()):
                    return parent
                for child in parent.findChildren(object):
                    if hasattr(child, 'objectName') and 'ScrollAreaWidgetContents' in str(child.objectName()):
                        return child
                parent = parent.parent()
            return None
        except Exception as e:
            print(f"查找滚动内容失败: {e}")
            return None
    
    @staticmethod
    def _calculate_content_height(page_widget):
        """计算页面内容实际高度"""
        try:
            page_widget.adjustSize()
            size_hint = page_widget.sizeHint()
            if size_hint.isValid():
                return size_hint.height()
            return 600  # 默认值
        except Exception as e:
            print(f"计算内容高度失败: {e}")
            return 600
