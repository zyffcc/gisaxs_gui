# window_config.py
"""
窗口配置管理
"""

class WindowConfig:
    """窗口配置类"""
    
    # 默认窗口尺寸
    DEFAULT_WIDTH = 1400
    DEFAULT_HEIGHT = 900
    
    # 默认最小尺寸
    MIN_WIDTH = 800
    MIN_HEIGHT = 600
    
    # 默认字体大小
    DEFAULT_FONT_SIZE = 9
    
    # 缩放限制
    MIN_SCALE = 0.7
    MAX_SCALE = 1.5
    
    # 屏幕占用比例（最大）
    MAX_SCREEN_RATIO = 0.9
    
    # 字体缩放因子
    FONT_SCALE_FACTOR = 0.5
    
    # 是否启用自适应缩放
    ENABLE_ADAPTIVE_SCALING = True
    
    # 是否启用鼠标屏幕定位
    ENABLE_MOUSE_SCREEN_POSITIONING = True
    
    # 是否启用自适应字体
    ENABLE_ADAPTIVE_FONT = True
    
    @classmethod
    def get_window_settings(cls):
        """获取窗口设置"""
        return {
            'default_size': (cls.DEFAULT_WIDTH, cls.DEFAULT_HEIGHT),
            'min_size': (cls.MIN_WIDTH, cls.MIN_HEIGHT),
            'font_size': cls.DEFAULT_FONT_SIZE,
            'adaptive_scaling': cls.ENABLE_ADAPTIVE_SCALING,
            'mouse_positioning': cls.ENABLE_MOUSE_SCREEN_POSITIONING,
            'adaptive_font': cls.ENABLE_ADAPTIVE_FONT
        }
    
    @classmethod
    def get_scaling_settings(cls):
        """获取缩放设置"""
        return {
            'min_scale': cls.MIN_SCALE,
            'max_scale': cls.MAX_SCALE,
            'max_screen_ratio': cls.MAX_SCREEN_RATIO,
            'font_scale_factor': cls.FONT_SCALE_FACTOR
        }
