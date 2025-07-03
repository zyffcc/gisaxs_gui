# styles.py
"""
自适应样式模块 - 根据不同分辨率和DPI提供样式
"""

from ui_config import get_ui_config


class AdaptiveStyles:
    """自适应样式管理类"""
    
    def __init__(self):
        self.ui_config = get_ui_config()
        self.scale_factor = self.ui_config.get_scale_factor()
    
    def get_main_style(self):
        """获取主窗口样式"""
        font_size = self.ui_config.get_adaptive_font_size()
        
        return f"""
        QMainWindow {{
            background-color: #f0f0f0;
            font-size: {font_size}pt;
        }}
        
        QTabWidget::pane {{
            border: 1px solid #c0c0c0;
            background-color: white;
        }}
        
        QTabBar::tab {{
            background-color: #e1e1e1;
            padding: {self._scale(8)}px {self._scale(12)}px;
            margin-right: {self._scale(2)}px;
            border: 1px solid #c0c0c0;
            border-bottom: none;
        }}
        
        QTabBar::tab:selected {{
            background-color: white;
            border-bottom: 1px solid white;
        }}
        
        QPushButton {{
            background-color: #e1e1e1;
            border: 1px solid #a0a0a0;
            padding: {self._scale(6)}px {self._scale(12)}px;
            border-radius: {self._scale(3)}px;
            font-size: {font_size}pt;
        }}
        
        QPushButton:hover {{
            background-color: #d1d1d1;
        }}
        
        QPushButton:pressed {{
            background-color: #c1c1c1;
        }}
        
        QPushButton:disabled {{
            background-color: #f0f0f0;
            color: #a0a0a0;
        }}
        
        QLineEdit, QSpinBox, QDoubleSpinBox {{
            border: 1px solid #a0a0a0;
            padding: {self._scale(4)}px;
            border-radius: {self._scale(2)}px;
            font-size: {font_size}pt;
        }}
        
        QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
            border: 2px solid #0078d4;
        }}
        
        QComboBox {{
            border: 1px solid #a0a0a0;
            padding: {self._scale(4)}px;
            border-radius: {self._scale(2)}px;
            font-size: {font_size}pt;
        }}
        
        QComboBox::drop-down {{
            subcontrol-origin: padding;
            subcontrol-position: top right;
            width: {self._scale(20)}px;
            border-left: 1px solid #a0a0a0;
        }}
        
        QGroupBox {{
            font-weight: bold;
            border: 2px solid #a0a0a0;
            border-radius: {self._scale(5)}px;
            margin-top: {self._scale(10)}px;
            padding-top: {self._scale(10)}px;
            font-size: {font_size}pt;
        }}
        
        QGroupBox::title {{
            subcontrol-origin: margin;
            left: {self._scale(10)}px;
            padding: 0 {self._scale(5)}px 0 {self._scale(5)}px;
        }}
        
        QProgressBar {{
            border: 1px solid #a0a0a0;
            border-radius: {self._scale(5)}px;
            text-align: center;
            font-size: {font_size}pt;
        }}
        
        QProgressBar::chunk {{
            background-color: #0078d4;
            border-radius: {self._scale(3)}px;
        }}
        
        QScrollBar:vertical {{
            background: #f0f0f0;
            width: {self._scale(15)}px;
            border-radius: {self._scale(7)}px;
        }}
        
        QScrollBar::handle:vertical {{
            background: #c0c0c0;
            border-radius: {self._scale(7)}px;
            min-height: {self._scale(20)}px;
        }}
        
        QScrollBar::handle:vertical:hover {{
            background: #a0a0a0;
        }}
        """
    
    def _scale(self, value):
        """缩放数值"""
        return int(value * self.scale_factor)


# 全局样式实例
adaptive_styles = None


def get_adaptive_styles():
    """获取自适应样式实例"""
    global adaptive_styles
    if adaptive_styles is None:
        adaptive_styles = AdaptiveStyles()
    return adaptive_styles
