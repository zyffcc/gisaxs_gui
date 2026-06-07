# user_settings.py
"""
用户设置管理 - 保存和加载用户的显示设置
"""

import json
import os
import shutil
import sys
# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.window_config import WindowConfig


class UserSettings:
    """用户设置管理类"""
    
    def __init__(self, settings_file="config/user_settings.json"):
        self.settings_file = settings_file
        self.default_settings_file = os.path.join("config", "default_user_settings.json")
        self.config = WindowConfig()
        self._ensure_default_settings_file()
        self._ensure_settings_file()
        self._settings = self.load_settings()

    def _build_default_settings(self):
        """构建默认设置。"""
        return {
            "enable_adaptive_scaling": self.config.ENABLE_ADAPTIVE_SCALING,
            "window_width": self.config.DEFAULT_WIDTH,
            "window_height": self.config.DEFAULT_HEIGHT,
            "font_adjustment": 0,
            "visual_font_scale": 100,
            "_auto_k_enabled": False,
            "fit.points_num": 50,
            "fit.interp_method": "Linear",
            "responsive_layout_mode": "auto",
            "responsive_resize_on_start": False,
            "responsive_font_enabled": False,
            "auto_detect_monitor_dpi": True,
            "adaptive_layout_enabled": True,
            "manual_screen_resolution": "auto",
            "layout_target_mode": "auto",
            "layout_target_custom": "1920x1080",
            "auto_fit_layout_target": True,
            "ai_fitting": {
                "model_base_dirs": [
                    "modules/Fitting_1D_Model",
                    "modules/Fitting_1D_model"
                ],
                "last_selected_model": "",
                "last_constraint_mode": "Free",
                "extra_model_paths": []
            }
        }

    def _load_default_settings(self):
        """从默认模板加载设置，并补齐代码内默认键。"""
        default_settings = self._build_default_settings()

        if not os.path.exists(self.default_settings_file):
            return default_settings

        try:
            with open(self.default_settings_file, 'r', encoding='utf-8') as f:
                template_settings = json.load(f)
            return {**default_settings, **template_settings}
        except (json.JSONDecodeError, IOError):
            return default_settings

    def _ensure_default_settings_file(self):
        """确保默认设置模板存在。"""
        if os.path.exists(self.default_settings_file):
            return

        try:
            with open(self.default_settings_file, 'w', encoding='utf-8') as f:
                json.dump(self._build_default_settings(), f, indent=2, ensure_ascii=False)
        except IOError:
            pass

    def _ensure_settings_file(self):
        """首次运行时从默认设置模板复制用户设置。"""
        if os.path.exists(self.settings_file):
            return

        try:
            shutil.copy(self.default_settings_file, self.settings_file)
        except IOError:
            pass
    
    def load_settings(self):
        """从文件加载设置"""
        default_settings = self._load_default_settings()

        if os.path.exists(self.settings_file):
            try:
                with open(self.settings_file, 'r', encoding='utf-8') as f:
                    loaded_settings = json.load(f)

                merged_settings = {**default_settings, **loaded_settings}
                if merged_settings != loaded_settings:
                    self._settings = merged_settings
                    self.save_settings()

                return merged_settings
            except (json.JSONDecodeError, IOError):
                pass
        
        return default_settings
    
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

    def get_visual_font_scale(self):
        """获取全局UI字体缩放百分比"""
        return int(self.get("visual_font_scale", 100))

    def set_visual_font_scale(self, scale):
        """设置全局UI字体缩放百分比"""
        self.set("visual_font_scale", int(scale))
    
    def enable_adaptive_scaling(self, enabled):
        """启用/禁用自适应缩放"""
        self.set("enable_adaptive_scaling", enabled)


# 创建全局实例
user_settings = UserSettings()
