# settings_dialog.py
"""
显示设置对话框 - 简化版
"""

from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QCheckBox, 
                             QLabel, QSpinBox, QDoubleSpinBox, QPushButton,
                             QGroupBox, QFormLayout, QMessageBox, QComboBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
import sys
import os
# 添加父目录到路径以便导入
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.window_config import WindowConfig
from core.user_settings import user_settings


class SettingsDialog(QDialog):
    """显示设置对话框"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.config = WindowConfig()
        self.init_ui()
        self.load_settings()
    
    def init_ui(self):
        """初始化界面"""
        self.setWindowTitle("显示设置")
        self.setModal(True)
        self.resize(350, 250)
        
        layout = QVBoxLayout()
        
        # 分辨率设置组
        resolution_group = QGroupBox("窗口大小")
        resolution_layout = QFormLayout()
        
        # 窗口大小预设
        self.resolution_combo = QComboBox()
        self.resolution_combo.addItems([
            "800 x 600 (小)",
            "1000 x 700 (中等)",
            "1200 x 800 (大)",
            "1400 x 900 (超大)",
            "自定义"
        ])
        self.resolution_combo.currentTextChanged.connect(self.on_resolution_changed)
        resolution_layout.addRow("窗口大小:", self.resolution_combo)
        
        # 自定义分辨率
        custom_layout = QHBoxLayout()
        self.width_spin = QSpinBox()
        self.width_spin.setRange(600, 2400)
        self.width_spin.setValue(1000)
        self.width_spin.setSuffix(" px")
        
        self.times_label = QLabel("×")  # 保存引用以便控制可见性
        
        self.height_spin = QSpinBox()
        self.height_spin.setRange(400, 1600)
        self.height_spin.setValue(700)
        self.height_spin.setSuffix(" px")
        
        custom_layout.addWidget(self.width_spin)
        custom_layout.addWidget(self.times_label)
        custom_layout.addWidget(self.height_spin)
        
        resolution_layout.addRow("自定义大小:", custom_layout)
        
        # 初始时隐藏自定义设置
        self.width_spin.setVisible(False)
        self.height_spin.setVisible(False)
        self.times_label.setVisible(False)
        
        resolution_group.setLayout(resolution_layout)
        layout.addWidget(resolution_group)
        
        # 显示设置组
        display_group = QGroupBox("显示设置")
        display_layout = QFormLayout()
        
        # 启用自适应
        self.adaptive_cb = QCheckBox("启用自适应缩放")
        self.adaptive_cb.setToolTip("根据屏幕分辨率和DPI自动调整窗口和字体大小")
        display_layout.addRow(self.adaptive_cb)
        
        # 字体大小调整
        self.font_spin = QSpinBox()
        self.font_spin.setRange(-3, 3)
        self.font_spin.setSuffix(" 级")
        self.font_spin.setToolTip("调整字体大小（-3到+3级）")
        self.font_spin.valueChanged.connect(self.on_font_changed)
        display_layout.addRow("字体大小:", self.font_spin)
        
        display_group.setLayout(display_layout)
        layout.addWidget(display_group)
        
        # 按钮
        button_layout = QHBoxLayout()
        
        self.apply_button = QPushButton("应用")
        self.apply_button.clicked.connect(self.apply_settings)
        button_layout.addWidget(self.apply_button)
        
        button_layout.addStretch()
        
        self.cancel_button = QPushButton("取消")
        self.cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_button)
        
        self.ok_button = QPushButton("确定")
        self.ok_button.clicked.connect(self.accept_settings)
        self.ok_button.setDefault(True)
        button_layout.addWidget(self.ok_button)
        
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
    
    def on_resolution_changed(self, text):
        """分辨率选择改变时的处理"""
        is_custom = text == "自定义"
        self.width_spin.setVisible(is_custom)
        self.height_spin.setVisible(is_custom)
        self.times_label.setVisible(is_custom)
        
        if not is_custom:
            # 设置预设分辨率
            if "800 x 600" in text:
                self.width_spin.setValue(800)
                self.height_spin.setValue(600)
            elif "1000 x 700" in text:
                self.width_spin.setValue(1000)
                self.height_spin.setValue(700)
            elif "1200 x 800" in text:
                self.width_spin.setValue(1200)
                self.height_spin.setValue(800)
            elif "1400 x 900" in text:
                self.width_spin.setValue(1400)
                self.height_spin.setValue(900)
    
    def on_font_changed(self, value):
        """字体大小改变时立即应用（如果不是自适应模式）"""
        if not self.adaptive_cb.isChecked():
            self.apply_font_change(value)
    
    def apply_font_change(self, adjustment):
        """应用字体改变"""
        if self.parent_window:
            current_font = self.parent_window.font()
            base_size = 9  # 基础字体大小
            new_size = max(8, base_size + adjustment)
            new_font = QFont(current_font.family())
            new_font.setPointSize(new_size)
            self.parent_window.setFont(new_font)
    
    def apply_settings(self):
        """应用设置（不关闭对话框）"""
        if not self.adaptive_cb.isChecked():
            # 保存窗口大小设置
            new_width = self.width_spin.value()
            new_height = self.height_spin.value()
            user_settings.set_window_size(new_width, new_height)
            
            # 应用窗口大小
            if self.parent_window:
                self.parent_window.resize(new_width, new_height)
            
            # 保存并应用字体
            font_adjustment = self.font_spin.value()
            user_settings.set_font_adjustment(font_adjustment)
            self.apply_font_change(font_adjustment)
            
            # 保存设置到文件
            user_settings.save_settings()
            
            QMessageBox.information(self, "设置已应用", "窗口大小和字体设置已立即生效。")
        else:
            QMessageBox.information(self, "提示", "自适应模式下的设置需要重启应用程序后生效。")
    
    def load_settings(self):
        """加载当前设置"""
        # 加载自适应设置
        self.adaptive_cb.setChecked(user_settings.is_adaptive_enabled())
        
        # 加载字体设置
        self.font_spin.setValue(user_settings.get_font_adjustment())
        
        # 加载当前窗口大小
        if self.parent_window:
            current_size = self.parent_window.size()
            self.width_spin.setValue(current_size.width())
            self.height_spin.setValue(current_size.height())
        else:
            # 从设置中加载
            width, height = user_settings.get_window_size()
            self.width_spin.setValue(width)
            self.height_spin.setValue(height)
        
        # 根据当前大小选择对应的预设
        width, height = self.width_spin.value(), self.height_spin.value()
        if width == 800 and height == 600:
            self.resolution_combo.setCurrentText("800 x 600 (小)")
        elif width == 1000 and height == 700:
            self.resolution_combo.setCurrentText("1000 x 700 (中等)")
        elif width == 1200 and height == 800:
            self.resolution_combo.setCurrentText("1200 x 800 (大)")
        elif width == 1400 and height == 900:
            self.resolution_combo.setCurrentText("1400 x 900 (超大)")
        else:
            self.resolution_combo.setCurrentText("自定义")
            self.on_resolution_changed("自定义")
    
    def accept_settings(self):
        """确定并关闭对话框"""
        current_adaptive = user_settings.is_adaptive_enabled()
        new_adaptive = self.adaptive_cb.isChecked()
        
        if new_adaptive != current_adaptive:
            # 自适应设置发生改变，需要重启
            reply = QMessageBox.question(
                self,
                "需要重启",
                "自适应设置已更改，需要重启应用程序才能生效。是否现在重启？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )
            
            if reply == QMessageBox.Yes:
                # 保存设置并重启
                user_settings.enable_adaptive_scaling(new_adaptive)
                self.save_all_settings()
                user_settings.save_settings()
                self.restart_application()
            else:
                # 不重启，恢复原设置
                self.adaptive_cb.setChecked(current_adaptive)
        else:
            # 应用其他设置
            self.save_all_settings()
            self.apply_settings()
        
        self.accept()
    
    def save_all_settings(self):
        """保存所有设置"""
        # 保存窗口大小
        user_settings.set_window_size(self.width_spin.value(), self.height_spin.value())
        
        # 保存字体设置
        user_settings.set_font_adjustment(self.font_spin.value())
        
        # 保存自适应设置
        user_settings.enable_adaptive_scaling(self.adaptive_cb.isChecked())
        
        # 保存到文件
        user_settings.save_settings()
    
    def restart_application(self):
        """重启应用程序"""
        import os
        import sys
        os.execl(sys.executable, sys.executable, *sys.argv)
