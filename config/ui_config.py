# ui_config.py
"""
UI配置模块 - 处理不同分辨率和DPI设置下的界面适配
"""

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QSettings
from PyQt5.QtGui import QCursor
import json
import os


class UIConfig:
    """UI配置管理类"""
    
    def __init__(self):
        self.settings = QSettings("GISAXS Lab", "GISAXS Toolkit")
        self.screen_info = self.get_screen_info()
        
    def get_screen_info(self):
        """获取鼠标所在屏幕的信息"""
        # 获取鼠标当前位置
        cursor_pos = QCursor.pos()
        
        # 获取包含鼠标的屏幕
        desktop = QApplication.desktop()
        screen_num = desktop.screenNumber(cursor_pos)
        screen_rect = desktop.screenGeometry(screen_num)
        
        return {
            'width': screen_rect.width(),
            'height': screen_rect.height(),
            'dpi': desktop.physicalDpiX(),
            'screen_number': screen_num
        }
    
    def get_adaptive_font_size(self):
        """获取自适应字体大小"""
        width = self.screen_info['width']
        dpi = self.screen_info['dpi']
        
        # 基于屏幕宽度和DPI计算字体大小
        if width >= 3840:  # 4K
            base_size = 12
        elif width >= 2560:  # 2K
            base_size = 11
        elif width >= 1920:  # 1080p
            base_size = 10
        elif width >= 1366:  # 720p
            base_size = 9
        else:
            base_size = 8
        
        # DPI调整
        if dpi > 120:
            base_size += 1
        elif dpi < 96:
            base_size -= 1
            
        return max(8, min(16, base_size))  # 限制在8-16之间
    
    def get_adaptive_window_size(self):
        """获取自适应窗口大小"""
        width = self.screen_info['width']
        height = self.screen_info['height']
        
        # 计算窗口大小（屏幕的75-85%）
        if width >= 3840:  # 4K
            ratio = 0.75
        elif width >= 1920:  # 1080p及以上
            ratio = 0.8
        else:
            ratio = 0.85
        
        window_width = int(width * ratio)
        window_height = int(height * ratio)
        
        # 设置最大最小限制
        window_width = max(800, min(1600, window_width))
        window_height = max(600, min(1200, window_height))
        
        return window_width, window_height
    
    def get_adaptive_minimum_size(self):
        """获取自适应最小窗口大小"""
        width = self.screen_info['width']
        height = self.screen_info['height']
        
        min_width = max(600, int(width * 0.3))
        min_height = max(450, int(height * 0.3))
        
        return min_width, min_height
    
    def save_window_geometry(self, window):
        """保存窗口几何信息"""
        self.settings.setValue("window/geometry", window.saveGeometry())
        self.settings.setValue("window/state", window.saveState())
    
    def restore_window_geometry(self, window):
        """恢复窗口几何信息"""
        geometry = self.settings.value("window/geometry")
        state = self.settings.value("window/state")
        
        if geometry:
            window.restoreGeometry(geometry)
        if state:
            window.restoreState(state)
            
        return geometry is not None
    
    def get_scale_factor(self):
        """获取缩放因子"""
        dpi = self.screen_info['dpi']
        # 以96 DPI为基准
        return dpi / 96.0
    
    def scale_size(self, size):
        """按缩放因子调整尺寸"""
        factor = self.get_scale_factor()
        if isinstance(size, (list, tuple)):
            return [int(s * factor) for s in size]
        else:
            return int(size * factor)


# 全局配置实例
ui_config = None


def get_ui_config():
    """获取UI配置实例"""
    global ui_config
    if ui_config is None:
        ui_config = UIConfig()
    return ui_config
