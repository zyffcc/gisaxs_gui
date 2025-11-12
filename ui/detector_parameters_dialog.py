from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, 
                             QLabel, QDoubleSpinBox, QCheckBox, QPushButton, 
                             QGroupBox, QGridLayout)
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtGui import QFont
import json
import os
from core.global_params import global_params

# 导入探测器专用触发管理器
from utils.detector_parameter_trigger_manager import DetectorParameterTriggerManager


class DetectorParametersDialog(QDialog):
    """探测器参数对话框"""
    
    # 定义信号
    parameters_changed = pyqtSignal(dict)  # 参数改变时发出信号
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Detector Parameters")
        self.setModal(False)  # 改为非模态对话框，允许同时操作主界面
        self.resize(400, 350)
        
        # 参数字典
        self.parameters = {}
        
        # 初始化触发管理器
        self.param_trigger_manager = DetectorParameterTriggerManager(self)
        
    # 去抖由 meta 触发管理器统一处理
        
        # 初始化UI
        self._init_ui()
        
        # 加载当前参数值
        self._load_parameters()
        
        # 连接信号
        self._connect_signals()
        
    def _init_ui(self):
        """初始化用户界面"""
        layout = QVBoxLayout(self)
        
        # 创建主要参数组
        main_group = QGroupBox("Detector Parameters")
        main_layout = QFormLayout(main_group)
        
        # Distance (mm) - 步长0.1
        self.distance_spinbox = QDoubleSpinBox()
        self.distance_spinbox.setRange(100.0, 10000.0)
        self.distance_spinbox.setDecimals(1)
        self.distance_spinbox.setSingleStep(0.1)
        self.distance_spinbox.setValue(2000.0)
        main_layout.addRow("Distance (mm):", self.distance_spinbox)
        
        # Grazing Angle (deg) - 步长0.01
        self.angle_spinbox = QDoubleSpinBox()
        self.angle_spinbox.setRange(0.01, 10.0)
        self.angle_spinbox.setDecimals(3)
        self.angle_spinbox.setSingleStep(0.01)
        self.angle_spinbox.setValue(0.4)
        main_layout.addRow("Grazing Angle (deg):", self.angle_spinbox)
        
        # Wavelength (nm) - 步长0.001
        self.wavelength_spinbox = QDoubleSpinBox()
        self.wavelength_spinbox.setRange(0.001, 1.0)
        self.wavelength_spinbox.setDecimals(4)
        self.wavelength_spinbox.setSingleStep(0.001)
        self.wavelength_spinbox.setValue(0.015)
        main_layout.addRow("Wavelength (nm):", self.wavelength_spinbox)
        # Energy (keV) - 与波长互相换算: E(keV) = 1.239841984 / λ(nm)
        self.energy_spinbox = QDoubleSpinBox()
        self.energy_spinbox.setRange(0.1, 200.0)  # 覆盖常见能量范围
        self.energy_spinbox.setDecimals(4)
        self.energy_spinbox.setSingleStep(0.01)
        self.energy_spinbox.setValue(1.239841984 / 0.015)  # 与默认波长匹配
        main_layout.addRow("Energy (keV):", self.energy_spinbox)

        # 关系提示标签（可选）
        relation_label = QLabel("E (keV) = 1.239841984 / λ (nm)")
        font = relation_label.font()
        font.setPointSize(font.pointSize() - 1)
        relation_label.setFont(font)
        relation_label.setStyleSheet("color: #555;")
        main_layout.addRow("", relation_label)
        
        # Beam Center X (pixels) - 步长0.01，显示两位小数
        self.beam_center_x_spinbox = QDoubleSpinBox()
        self.beam_center_x_spinbox.setRange(0.0, 2000.0)
        self.beam_center_x_spinbox.setDecimals(2)
        self.beam_center_x_spinbox.setSingleStep(0.01)
        self.beam_center_x_spinbox.setValue(50.0)
        main_layout.addRow("Beam Center X (px):", self.beam_center_x_spinbox)
        
        # Beam Center Y (pixels) - 步长0.01，显示两位小数
        self.beam_center_y_spinbox = QDoubleSpinBox()
        self.beam_center_y_spinbox.setRange(0.0, 2000.0)
        self.beam_center_y_spinbox.setDecimals(2)
        self.beam_center_y_spinbox.setSingleStep(0.01)
        self.beam_center_y_spinbox.setValue(50.0)
        main_layout.addRow("Beam Center Y (px):", self.beam_center_y_spinbox)
        
        # Pixel Size X (μm) - 步长0.1
        self.pixel_size_x_spinbox = QDoubleSpinBox()
        self.pixel_size_x_spinbox.setRange(1.0, 1000.0)
        self.pixel_size_x_spinbox.setDecimals(1)
        self.pixel_size_x_spinbox.setSingleStep(0.1)
        self.pixel_size_x_spinbox.setValue(172.0)
        main_layout.addRow("Pixel Size X (μm):", self.pixel_size_x_spinbox)
        
        # Pixel Size Y (μm) - 步长0.1
        self.pixel_size_y_spinbox = QDoubleSpinBox()
        self.pixel_size_y_spinbox.setRange(1.0, 1000.0)
        self.pixel_size_y_spinbox.setDecimals(1)
        self.pixel_size_y_spinbox.setSingleStep(0.1)
        self.pixel_size_y_spinbox.setValue(172.0)
        main_layout.addRow("Pixel Size Y (μm):", self.pixel_size_y_spinbox)
        
        layout.addWidget(main_group)
        
        # 显示选项组
        display_group = QGroupBox("Display Options")
        display_layout = QVBoxLayout(display_group)
        
        # Show in q axis 复选框
        self.show_q_axis_checkbox = QCheckBox("Show in q axis")
        # 不在这里设置默认值，稍后从配置中读取
        display_layout.addWidget(self.show_q_axis_checkbox)
        
        layout.addWidget(display_group)
        
        # 按钮布局
        button_layout = QHBoxLayout()
        
        button_layout.addStretch()
        
        # 应用按钮
        self.apply_button = QPushButton("Apply")
        self.apply_button.clicked.connect(self._apply_parameters)
        button_layout.addWidget(self.apply_button)
        
        # 确定按钮
        self.ok_button = QPushButton("OK")
        self.ok_button.clicked.connect(self._ok_clicked)
        button_layout.addWidget(self.ok_button)
        
        # 取消按钮
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_button)
        
        layout.addLayout(button_layout)
        
    def _connect_signals(self):
        """连接信号（使用 meta 管理器自动连接，按 finished/changed 模式）"""
        detector_widgets = [
            (self.distance_spinbox, 'distance'),
            (self.angle_spinbox, 'angle'),
            (self.wavelength_spinbox, 'wavelength'),
            (self.beam_center_x_spinbox, 'beam_center_x'),
            (self.beam_center_y_spinbox, 'beam_center_y'),
            (self.pixel_size_x_spinbox, 'pixel_size_x'),
            (self.pixel_size_y_spinbox, 'pixel_size_y'),
        ]
        # 注册到 meta，并通过 connect_mode 选择连接模式
        for w, name in detector_widgets:
            mode = 'changed' if name in ('beam_center_x', 'beam_center_y') else 'finished'
            self.param_trigger_manager.register_detector_widget(w, name, connect_mode=mode)
        self.show_q_axis_checkbox.toggled.connect(self._on_parameter_changed)
        # 额外连接：波长/能量联动（不通过meta保存能量，只保存波长）
        self._updating_energy_pair = False
        self.wavelength_spinbox.valueChanged.connect(self._on_wavelength_changed_sync)
        self.energy_spinbox.valueChanged.connect(self._on_energy_changed_sync)
        
    def _disconnect_signals(self):
        """断开信号连接"""
        try:
            for sb in [self.distance_spinbox, self.angle_spinbox, self.wavelength_spinbox,
                       self.beam_center_x_spinbox, self.beam_center_y_spinbox,
                       self.pixel_size_x_spinbox, self.pixel_size_y_spinbox,
                       getattr(self, 'energy_spinbox', None)]:
                try:
                    if sb is not None:
                        sb.valueChanged.disconnect()
                except Exception:
                    pass
            self.show_q_axis_checkbox.toggled.disconnect()
        except Exception:
            pass  # 忽略断开连接的错误
        
    def _load_parameters(self):
        """从配置文件加载参数"""
        try:
            # 临时断开信号连接以避免在加载时触发自动保存
            self._disconnect_signals()
            
            # 设置探测器参数 - 从Fitting模块专用参数读取
            beam_x = global_params.get_parameter('fitting', 'detector.beam_center_x', 50.0)
            beam_y = global_params.get_parameter('fitting', 'detector.beam_center_y', 50.0)
            
            self.distance_spinbox.setValue(global_params.get_parameter('fitting', 'detector.distance', 2000.0))
            self.beam_center_x_spinbox.setValue(beam_x)
            self.beam_center_y_spinbox.setValue(beam_y)
            self.pixel_size_x_spinbox.setValue(global_params.get_parameter('fitting', 'detector.pixel_size_x', 172.0))
            self.pixel_size_y_spinbox.setValue(global_params.get_parameter('fitting', 'detector.pixel_size_y', 172.0))
            
            # 设置束流参数
            self.angle_spinbox.setValue(global_params.get_parameter('beam', 'grazing_angle', 0.4))
            self.wavelength_spinbox.setValue(global_params.get_parameter('beam', 'wavelength', 0.015))
            # 同步能量显示
            try:
                wl = self.wavelength_spinbox.value()
                if wl > 0:
                    self.energy_spinbox.blockSignals(True)
                    self.energy_spinbox.setValue(1.239841984 / wl)
                    self.energy_spinbox.blockSignals(False)
            except Exception:
                pass
            
            # 设置显示选项 - 从fitting.detector中读取
            self.show_q_axis_checkbox.setChecked(global_params.get_parameter('fitting', 'detector.show_q_axis', False))
            
            # 重新连接信号
            self._connect_signals()
            
        except Exception as e:
            print(f"加载探测器参数失败: {e}")
            # 确保重新连接信号
            self._connect_signals()
            
    def _on_parameter_changed(self):
        """参数改变时的处理（用于复选框等非数值控件）"""
        # 自动保存到global_params
        self._save_parameters()
        
        # 发出参数改变信号
        params = self._get_current_parameters()
        self.parameters_changed.emit(params)
    
    def _on_parameter_changed_internal(self):
        """内部参数变更处理（用于触发管理器）"""
        # 发射信号通知外部，但不保存（保存由触发管理器处理）
        params = self._get_current_parameters()
        self.parameters_changed.emit(params)
    
    def _save_parameters_immediately(self):
        """兼容旧接口：现在与延迟保存逻辑一致（已由去抖提交控制）"""
        self._save_parameters()
    
    def _save_parameters_delayed(self):
        """兼容旧接口：保留调用，实际逻辑由去抖调度"""
        self._save_parameters()
    
    # 去抖逻辑已迁移到 meta；保留占位以兼容旧调用
    def _on_detector_value_changed(self, *args, **kwargs):
        pass
    def _commit_detector_param(self, *args, **kwargs):
        pass
        
    def _get_current_parameters(self):
        """获取当前参数值"""
        return {
            'distance': self.distance_spinbox.value(),
            'grazing_angle': self.angle_spinbox.value(),
            'wavelength': self.wavelength_spinbox.value(),
            'energy': self.energy_spinbox.value(),  # 仅供外部查看，不单独持久化
            'beam_center_x': self.beam_center_x_spinbox.value(),
            'beam_center_y': self.beam_center_y_spinbox.value(),
            'pixel_size_x': self.pixel_size_x_spinbox.value(),
            'pixel_size_y': self.pixel_size_y_spinbox.value(),
            'show_q_axis': self.show_q_axis_checkbox.isChecked()
        }

    # ===================== 波长/能量联动 =====================
    def _on_wavelength_changed_sync(self):
        if getattr(self, '_updating_energy_pair', False):
            return
        wl = self.wavelength_spinbox.value()
        if wl <= 0:
            return
        try:
            self._updating_energy_pair = True
            energy = 1.239841984 / wl
            self.energy_spinbox.blockSignals(True)
            self.energy_spinbox.setValue(energy)
            self.energy_spinbox.blockSignals(False)
        finally:
            self._updating_energy_pair = False

    def _on_energy_changed_sync(self):
        if getattr(self, '_updating_energy_pair', False):
            return
        energy = self.energy_spinbox.value()
        if energy <= 0:
            return
        try:
            wl = 1.239841984 / energy
            # 限制在波长允许范围内
            wl = max(self.wavelength_spinbox.minimum(), min(self.wavelength_spinbox.maximum(), wl))
            self._updating_energy_pair = True
            self.wavelength_spinbox.blockSignals(True)
            self.wavelength_spinbox.setValue(wl)
            self.wavelength_spinbox.blockSignals(False)
            # 让 meta 去抖逻辑继续处理 wavelength 的持久化
        finally:
            self._updating_energy_pair = False
        
    def _save_parameters(self):
        """保存参数到配置文件"""
        try:
            params = self._get_current_parameters()
            
            print(f"Saving detector parameters:")
            print(f"  Beam center: ({params['beam_center_x']}, {params['beam_center_y']})")
            print(f"  Distance: {params['distance']}")
            print(f"  Show Q axis: {params['show_q_axis']}")
            
            # 保存到Fitting模块专用参数
            global_params.set_parameter('fitting', 'detector.distance', params['distance'])
            global_params.set_parameter('fitting', 'detector.beam_center_x', params['beam_center_x'])
            global_params.set_parameter('fitting', 'detector.beam_center_y', params['beam_center_y'])
            global_params.set_parameter('fitting', 'detector.pixel_size_x', params['pixel_size_x'])
            global_params.set_parameter('fitting', 'detector.pixel_size_y', params['pixel_size_y'])
            global_params.set_parameter('fitting', 'detector.show_q_axis', params['show_q_axis'])
            
            # 更新beam参数
            global_params.set_parameter('beam', 'grazing_angle', params['grazing_angle'])
            global_params.set_parameter('beam', 'wavelength', params['wavelength'])
                
            # 触发保存
            global_params.save_user_parameters()
            
            print("✓ Detector parameters saved successfully")
            
        except Exception as e:
            print(f"保存探测器参数失败: {e}")
            
    def _apply_parameters(self):
        """应用参数"""
        self._save_parameters()
        params = self._get_current_parameters()
        self.parameters_changed.emit(params)
        
    def _ok_clicked(self):
        """确定按钮点击"""
        self._apply_parameters()
        self.accept()
        
    def get_parameters(self):
        """获取参数（供外部调用）"""
        return self._get_current_parameters()
