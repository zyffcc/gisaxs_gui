# user_settings.py
"""
用户设置管理 - 保存和加载用户的显示设置
"""

import json
import os
import sys
# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.window_config import WindowConfig


class UserSettings:
    """用户设置管理类"""
    
    def __init__(self, settings_file="config/user_settings.json"):
        self.settings_file = settings_file
        self.config = WindowConfig()
        self._settings = self.load_settings()
    
    def load_settings(self):
        """从文件加载设置"""
        if os.path.exists(self.settings_file):
            try:
                with open(self.settings_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        
        # 返回默认设置
        return {
            "enable_adaptive_scaling": self.config.ENABLE_ADAPTIVE_SCALING,
            "window_width": self.config.DEFAULT_WIDTH,
            "window_height": self.config.DEFAULT_HEIGHT,
            "font_adjustment": 0
        }
    
    def save_settings(self):
        """保存设置到文件"""
        try:
            with open(self.settings_file, 'w', encoding='utf-8') as f:
                json.dump(self._settings, f, indent=2, ensure_ascii=False)
        except IOError:
            pass
    
    def get(self, key, default=None):
        """获取设置值"""
        return self._settings.get(key, default)
    
    def set(self, key, value):
        """设置值"""
        self._settings[key] = value
    
    def is_adaptive_enabled(self):
        """检查是否启用自适应"""
        return self.get("enable_adaptive_scaling", False)
    
    def get_window_size(self):
        """获取窗口大小"""
        width = self.get("window_width", self.config.DEFAULT_WIDTH)
        height = self.get("window_height", self.config.DEFAULT_HEIGHT)
        return width, height
    
    def set_window_size(self, width, height):
        """设置窗口大小"""
        self.set("window_width", width)
        self.set("window_height", height)
    
    def get_font_adjustment(self):
        """获取字体调整值"""
        return self.get("font_adjustment", 0)
    
    def set_font_adjustment(self, adjustment):
        """设置字体调整值"""
        self.set("font_adjustment", adjustment)
    
    def enable_adaptive_scaling(self, enabled):
        """启用/禁用自适应缩放"""
        self.set("enable_adaptive_scaling", enabled)


# 创建全局实例
user_settings = UserSettings()
