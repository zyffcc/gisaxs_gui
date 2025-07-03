# window_manager.py
"""
窗口管理器 - 处理窗口定位和自适应缩放
"""

from PyQt5.QtWidgets import QApplication, QDesktopWidget
from PyQt5.QtGui import QCursor
from PyQt5.QtCore import QRect
import sys
import os
# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.window_config import WindowConfig
from core.user_settings import user_settings


class WindowManager:
    """窗口管理器，负责窗口的定位和缩放"""
    
    def __init__(self):
        self.config = WindowConfig()
        self._desktop = None
    
    @property
    def desktop(self):
        """获取桌面对象（延迟初始化）"""
        if self._desktop is None:
            self._desktop = QApplication.desktop()
        return self._desktop
    
    def get_mouse_screen_info(self):
        """获取鼠标所在屏幕的信息"""
        cursor_pos = QCursor.pos()
        screen_number = self.desktop.screenNumber(cursor_pos)
        screen_geometry = self.desktop.screenGeometry(screen_number)
        
        return {
            'screen_number': screen_number,
            'geometry': screen_geometry,
            'cursor_pos': cursor_pos
        }
    
    def get_adaptive_window_size(self, base_width=None, base_height=None):
        """根据屏幕分辨率计算自适应窗口大小"""
        # 使用配置中的默认值
        if base_width is None:
            base_width = self.config.DEFAULT_WIDTH
        if base_height is None:
            base_height = self.config.DEFAULT_HEIGHT
            
        screen_info = self.get_mouse_screen_info()
        screen_geometry = screen_info['geometry']
        
        # 获取屏幕可用区域（排除任务栏等）
        available_geometry = self.desktop.availableGeometry(screen_info['screen_number'])
        
        # 计算缩放比例（基于屏幕宽度，以1920为基准）
        width_scale = screen_geometry.width() / 1920.0
        height_scale = screen_geometry.height() / 1080.0
        
        # 使用较小的缩放比例以确保窗口不会太大
        scale = min(width_scale, height_scale)
        
        # 限制缩放范围
        scale = max(self.config.MIN_SCALE, min(scale, self.config.MAX_SCALE))
        
        # 计算新的窗口大小
        new_width = int(base_width * scale)
        new_height = int(base_height * scale)
        
        # 确保窗口不超过可用屏幕区域的配置比例
        max_width = int(available_geometry.width() * self.config.MAX_SCREEN_RATIO)
        max_height = int(available_geometry.height() * self.config.MAX_SCREEN_RATIO)
        
        new_width = min(new_width, max_width)
        new_height = min(new_height, max_height)
        
        return new_width, new_height, scale
    
    def get_adaptive_minimum_size(self, base_min_width=None, base_min_height=None):
        """获取自适应的最小窗口大小"""
        # 使用配置中的默认值
        if base_min_width is None:
            base_min_width = self.config.MIN_WIDTH
        if base_min_height is None:
            base_min_height = self.config.MIN_HEIGHT
            
        _, _, scale = self.get_adaptive_window_size()
        
        min_width = int(base_min_width * scale)
        min_height = int(base_min_height * scale)
        
        # 确保最小尺寸不会太小
        min_width = max(min_width, 600)
        min_height = max(min_height, 450)
        
        return min_width, min_height
    
    def center_window_on_mouse_screen(self, window):
        """将窗口居中显示在鼠标所在的屏幕上"""
        screen_info = self.get_mouse_screen_info()
        screen_geometry = screen_info['geometry']
        window_geometry = window.geometry()
        
        # 计算居中位置
        x = screen_geometry.x() + (screen_geometry.width() - window_geometry.width()) // 2
        y = screen_geometry.y() + (screen_geometry.height() - window_geometry.height()) // 2
        
        window.move(x, y)
    
    def _center_window_on_primary_screen(self, window):
        """将窗口居中显示在主屏幕上"""
        primary_screen = self.desktop.screenGeometry(0)
        window_geometry = window.geometry()
        
        # 计算居中位置
        x = primary_screen.x() + (primary_screen.width() - window_geometry.width()) // 2
        y = primary_screen.y() + (primary_screen.height() - window_geometry.height()) // 2
        
        window.move(x, y)
    
    def setup_adaptive_window(self, window, base_width=None, base_height=None, 
                             base_min_width=None, base_min_height=None):
        """为窗口设置自适应大小和位置"""
        # 检查用户设置中是否启用了自适应
        if user_settings.is_adaptive_enabled():
            # 获取自适应尺寸
            adaptive_width, adaptive_height, scale = self.get_adaptive_window_size(base_width, base_height)
            min_width, min_height = self.get_adaptive_minimum_size(base_min_width, base_min_height)
            
            # 设置窗口大小
            window.setMinimumSize(min_width, min_height)
            window.resize(adaptive_width, adaptive_height)
        else:
            # 使用用户设置的固定大小
            width, height = user_settings.get_window_size()
            
            # 设置最小尺寸
            if base_min_width is None:
                base_min_width = self.config.MIN_WIDTH
            if base_min_height is None:
                base_min_height = self.config.MIN_HEIGHT
                
            window.setMinimumSize(base_min_width, base_min_height)
            window.resize(width, height)
            scale = 1.0
        
        # 根据配置决定是否在鼠标屏幕定位
        settings = self.config.get_window_settings()
        if settings['mouse_positioning']:
            self.center_window_on_mouse_screen(window)
        else:
            self._center_window_on_primary_screen(window)
        
        return scale
    
    def get_adaptive_font_size(self, base_font_size=None):
        """获取自适应字体大小"""
        if base_font_size is None:
            base_font_size = self.config.DEFAULT_FONT_SIZE
            
        _, _, scale = self.get_adaptive_window_size()
        
        # 字体缩放比例稍微保守一些
        font_scale = (scale - 1) * self.config.FONT_SCALE_FACTOR + 1
        font_scale = max(0.8, min(font_scale, 1.3))
        
        font_size = int(base_font_size * font_scale)
        return max(8, font_size)
    
    def apply_adaptive_font(self, window, base_font_size=None):
        """为窗口应用自适应字体"""
        # 检查是否启用自适应
        if user_settings.is_adaptive_enabled():
            # 使用自适应字体
            from PyQt5.QtGui import QFont
            
            font_size = self.get_adaptive_font_size(base_font_size)
            font = QFont()
            font.setPointSize(font_size)
            window.setFont(font)
        else:
            # 使用用户设置的字体调整
            from PyQt5.QtGui import QFont
            
            if base_font_size is None:
                base_font_size = self.config.DEFAULT_FONT_SIZE
            
            adjustment = user_settings.get_font_adjustment()
            font_size = max(8, base_font_size + adjustment)
            
            font = QFont()
            font.setPointSize(font_size)
            window.setFont(font)
    
    def get_screen_dpi_info(self):
        """获取屏幕DPI信息"""
        screen_info = self.get_mouse_screen_info()
        screen_number = screen_info['screen_number']
        
        # 获取物理DPI
        physical_dpi_x = self.desktop.physicalDpiX()
        physical_dpi_y = self.desktop.physicalDpiY()
        
        # 获取逻辑DPI
        logical_dpi_x = self.desktop.logicalDpiX()
        logical_dpi_y = self.desktop.logicalDpiY()
        
        return {
            'physical_dpi': (physical_dpi_x, physical_dpi_y),
            'logical_dpi': (logical_dpi_x, logical_dpi_y),
            'scale_factor': logical_dpi_x / 96.0  # Windows标准DPI是96
        }


# 创建全局实例
window_manager = WindowManager()
