# display_config.py
"""
显示配置 - 用户可以选择的显示选项
"""

import json
import os
from PyQt5.QtCore import QSettings


class DisplayConfig:
    """显示配置管理"""
    
    def __init__(self):
        self.settings = QSettings("GISAXS Lab", "GISAXS Toolkit")
        self.config_file = "display_settings.json"
        self.load_config()
    
    def load_config(self):
        """加载配置"""
        default_config = {
            "enable_adaptive_scaling": False,  # 默认关闭自适应缩放
            "enable_adaptive_styles": False,   # 默认关闭自适应样式
            "enable_smart_positioning": True,  # 默认启用智能定位（鼠标屏幕）
            "preserve_designer_appearance": True,  # 默认保持Designer外观
            "font_size_adjustment": 0,  # 字体大小调整（-2到+2）
            "window_size_scale": 1.0    # 窗口大小缩放比例（0.8到1.2）
        }
        
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    self.config = json.load(f)
                    # 补充缺失的配置项
                    for key, value in default_config.items():
                        if key not in self.config:
                            self.config[key] = value
            except:
                self.config = default_config
        else:
            self.config = default_config
    
    def save_config(self):
        """保存配置"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
        except:
            pass
    
    def get(self, key, default=None):
        """获取配置项"""
        return self.config.get(key, default)
    
    def set(self, key, value):
        """设置配置项"""
        self.config[key] = value
        self.save_config()
    
    def is_adaptive_enabled(self):
        """是否启用自适应功能"""
        return self.get("enable_adaptive_scaling", False)
    
    def is_styles_enabled(self):
        """是否启用自适应样式"""
        return self.get("enable_adaptive_styles", False)
    
    def is_smart_positioning_enabled(self):
        """是否启用智能定位"""
        return self.get("enable_smart_positioning", True)
    
    def preserve_designer_appearance(self):
        """是否保持Designer外观"""
        return self.get("preserve_designer_appearance", True)


# 全局实例
_display_config = None


def get_display_config():
    """获取显示配置实例"""
    global _display_config
    if _display_config is None:
        _display_config = DisplayConfig()
    return _display_config
