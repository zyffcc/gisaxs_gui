"""
预处理控制器 - 管理GISAXS数据的预处理参数
"""

import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtWidgets import QMessageBox, QFileDialog


class PreprocessingController(QObject):
    """预处理参数控制器"""
    
    # 参数改变信号
    parameters_changed = pyqtSignal(str, dict)
    
    def __init__(self, ui, parent=None):
        super().__init__(parent)
        self.ui = ui
        self.parent = parent
        
        # 默认参数
        self.default_parameters = {
            'focus_region': {
                'type': 'q',
                'qr_min': 0.01,
                'qr_max': 3.0,
                'qz_min': 0.01,
                'qz_max': 3.0,
            },
            'noising': {
                'type': 'Gaussian',
                'snr_min': 80,
                'snr_max': 130,
            },
            'others': {
                'crop_edge': True,
                'add_mask': True,
                'normalize': True,
                'logarization': True,
                'crop_edge_params': {
                    'up_min': 0, 'up_max': 20,
                    'down_min': 0, 'down_max': 20,
                    'left_min': 0, 'left_max': 20,
                    'right_min': 0, 'right_max': 20,
                },
                'normalize_params': {
                    'minimum': 0,
                    'maximum': 20,
                }
            }
        }
        
        # 掩码文件路径
        self.mask_file_path = None
        
        # 设置信号连接
        self._setup_connections()
    
    def _setup_connections(self):
        """设置信号连接"""
        # 焦点区域参数
        self._setup_focus_region_connections()
        
        # 噪声参数
        self._setup_noising_connections()
        
        # 其他处理参数
        self._setup_others_connections()
    
    def _setup_focus_region_connections(self):
        """设置焦点区域参数连接"""
        # 焦点区域类型
        self.ui.FocusRegionTypeCombox.currentTextChanged.connect(self._on_focus_region_type_changed)
        
        # Q空间范围参数
        focus_region_inputs = [
            self.ui.FocusRegionQrMinValue,
            self.ui.FocusRegionQrMaxValue,
            self.ui.FocusRegionQzMinValue,
            self.ui.FocusRegionQzMaxValue,
        ]
        
        for input_widget in focus_region_inputs:
            if hasattr(input_widget, 'textChanged'):
                input_widget.textChanged.connect(self._emit_parameters_changed)
        
        # 显示按钮
        self.ui.FocusRegionQShowButton.clicked.connect(self._show_focus_region_parameters)
    
    def _setup_noising_connections(self):
        """设置噪声参数连接"""
        # 噪声类型
        self.ui.NoisingTypeCombox.currentTextChanged.connect(self._on_noising_type_changed)
        
        # 高斯噪声参数
        noising_inputs = [
            self.ui.NoisingGaussianSnrMinValue,
            self.ui.NoisingGaussianSnrMaxValue,
        ]
        
        for input_widget in noising_inputs:
            if hasattr(input_widget, 'textChanged'):
                input_widget.textChanged.connect(self._emit_parameters_changed)
        
        # 显示按钮
        self.ui.NoisingGaussianShowButton.clicked.connect(self._show_noising_parameters)
    
    def _setup_others_connections(self):
        """设置其他处理参数连接"""
        # 复选框
        checkboxes = [
            self.ui.OthersCropEdgeCheckBox,
            self.ui.OthersAddMaskCheckBox,
            self.ui.OthersNormalizeCheckBox,
            self.ui.OthersLogarizationCheckBox,
        ]
        
        for checkbox in checkboxes:
            if hasattr(checkbox, 'toggled'):
                checkbox.toggled.connect(self._on_checkbox_changed)
        
        # 边缘裁剪参数
        crop_edge_inputs = [
            self.ui.OthersCropEdgeUpMinValue,
            self.ui.OthersCropEdgeUpMaxValue,
            self.ui.OthersCropEdgeDownMinValue,
            self.ui.OthersCropEdgeDownMaxValue,
            self.ui.OthersCropEdgeLeftMinValue,
            self.ui.OthersCropEdgeLeftMaxValue,
            self.ui.OthersCropEdgeRightMinValue,
            self.ui.OthersCropEdgeRightMaxValue,
        ]
        
        for input_widget in crop_edge_inputs:
            if hasattr(input_widget, 'textChanged'):
                input_widget.textChanged.connect(self._emit_parameters_changed)
        
        # 归一化参数
        normalize_inputs = [
            self.ui.OthersNormalizeMinimumValue,
            self.ui.OthersNormalizeMaxmumValue,
        ]
        
        for input_widget in normalize_inputs:
            if hasattr(input_widget, 'textChanged'):
                input_widget.textChanged.connect(self._emit_parameters_changed)
        
        # 掩码文件选择
        self.ui.OthersAddMaskChooseFileButton.clicked.connect(self._choose_mask_file)
        self.ui.OthersAddMaskUserDefinedButton.clicked.connect(self._create_user_defined_mask)
        
        # 显示按钮
        self.ui.OthersShowButton.clicked.connect(self._show_others_parameters)
    
    def initialize(self):
        """初始化预处理参数"""
        self.set_parameters(self.default_parameters)
        self._update_focus_region_visibility()
        self._update_noising_visibility()
        self._update_others_visibility()
    
    def _on_focus_region_type_changed(self, region_type):
        """焦点区域类型改变处理"""
        self._update_focus_region_visibility()
        self._emit_parameters_changed()
    
    def _update_focus_region_visibility(self):
        """更新焦点区域相关界面的可见性"""
        region_type = self.ui.FocusRegionTypeCombox.currentText()
        
        if region_type == "q":
            self.ui.FocusRegionStackedWidget.setCurrentIndex(0)
        elif region_type == "Pixel":
            # 如果有像素页面的话
            pass
        elif region_type == "None":
            # 禁用焦点区域功能
            pass
    
    def _on_noising_type_changed(self, noising_type):
        """噪声类型改变处理"""
        self._update_noising_visibility()
        self._emit_parameters_changed()
    
    def _update_noising_visibility(self):
        """更新噪声相关界面的可见性"""
        noising_type = self.ui.NoisingTypeCombox.currentText()
        
        if noising_type == "Gaussian":
            self.ui.NoisingStackedWidget.setCurrentIndex(0)
        elif noising_type == "None":
            # 禁用噪声功能
            pass
    
    def _on_checkbox_changed(self):
        """复选框状态改变处理"""
        self._update_others_visibility()
        self._emit_parameters_changed()
    
    def _update_others_visibility(self):
        """更新其他处理选项的可见性"""
        # 根据复选框状态启用/禁用相关控件
        
        # 边缘裁剪
        crop_edge_enabled = self.ui.OthersCropEdgeCheckBox.isChecked()
        self.ui.OthersCropEdgeWidget.setEnabled(crop_edge_enabled)
        
        # 掩码添加
        add_mask_enabled = self.ui.OthersAddMaskCheckBox.isChecked()
        self.ui.OthersAddMaskWidget.setEnabled(add_mask_enabled)
        
        # 归一化
        normalize_enabled = self.ui.OthersNormalizeCheckBox.isChecked()
        self.ui.OthersNormalizeWidget.setEnabled(normalize_enabled)
    
    def _choose_mask_file(self):
        """选择掩码文件"""
        # 获取主窗口作为父窗口
        main_window = self.parent.parent if hasattr(self.parent, 'parent') else None
        
        file_path, _ = QFileDialog.getOpenFileName(
            main_window,
            "选择掩码文件",
            "",
            "图像文件 (*.png *.jpg *.jpeg *.tiff *.tif);;所有文件 (*.*)"
        )
        
        if file_path:
            self.mask_file_path = file_path
            self.ui.OthersAddMaskChooseFileButton.setText(f"已选择: {file_path.split('/')[-1]}")
            self._emit_parameters_changed()
    
    def _create_user_defined_mask(self):
        """创建用户自定义掩码"""
        # 获取主窗口作为父窗口
        main_window = self.parent.parent if hasattr(self.parent, 'parent') else None
        
        # 这里可以打开一个掩码编辑器或者提供掩码创建工具
        QMessageBox.information(
            main_window,
            "用户自定义掩码",
            "用户自定义掩码功能尚未实现。\n将来可以在这里添加掩码编辑器。"
        )
    
    def get_parameters(self):
        """获取当前预处理参数"""
        try:
            parameters = {
                'focus_region': self._get_focus_region_parameters(),
                'noising': self._get_noising_parameters(),
                'others': self._get_others_parameters(),
            }
            return parameters
        except Exception as e:
            print(f"获取预处理参数错误: {e}")
            return self.default_parameters.copy()
    
    def _get_focus_region_parameters(self):
        """获取焦点区域参数"""
        try:
            return {
                'type': self.ui.FocusRegionTypeCombox.currentText(),
                'qr_min': float(self.ui.FocusRegionQrMinValue.text()),
                'qr_max': float(self.ui.FocusRegionQrMaxValue.text()),
                'qz_min': float(self.ui.FocusRegionQzMinValue.text()),
                'qz_max': float(self.ui.FocusRegionQzMaxValue.text()),
            }
        except (ValueError, AttributeError):
            return self.default_parameters['focus_region'].copy()
    
    def _get_noising_parameters(self):
        """获取噪声参数"""
        try:
            return {
                'type': self.ui.NoisingTypeCombox.currentText(),
                'snr_min': float(self.ui.NoisingGaussianSnrMinValue.text()),
                'snr_max': float(self.ui.NoisingGaussianSnrMaxValue.text()),
            }
        except (ValueError, AttributeError):
            return self.default_parameters['noising'].copy()
    
    def _get_others_parameters(self):
        """获取其他处理参数"""
        try:
            return {
                'crop_edge': self.ui.OthersCropEdgeCheckBox.isChecked(),
                'add_mask': self.ui.OthersAddMaskCheckBox.isChecked(),
                'normalize': self.ui.OthersNormalizeCheckBox.isChecked(),
                'logarization': self.ui.OthersLogarizationCheckBox.isChecked(),
                'mask_file_path': self.mask_file_path,
                'crop_edge_params': {
                    'up_min': int(self.ui.OthersCropEdgeUpMinValue.text()),
                    'up_max': int(self.ui.OthersCropEdgeUpMaxValue.text()),
                    'down_min': int(self.ui.OthersCropEdgeDownMinValue.text()),
                    'down_max': int(self.ui.OthersCropEdgeDownMaxValue.text()),
                    'left_min': int(self.ui.OthersCropEdgeLeftMinValue.text()),
                    'left_max': int(self.ui.OthersCropEdgeLeftMaxValue.text()),
                    'right_min': int(self.ui.OthersCropEdgeRightMinValue.text()),
                    'right_max': int(self.ui.OthersCropEdgeRightMaxValue.text()),
                },
                'normalize_params': {
                    'minimum': float(self.ui.OthersNormalizeMinimumValue.text()),
                    'maximum': float(self.ui.OthersNormalizeMaxmumValue.text()),
                }
            }
        except (ValueError, AttributeError):
            return self.default_parameters['others'].copy()
    
    def set_parameters(self, parameters):
        """设置预处理参数"""
        if 'focus_region' in parameters:
            self._set_focus_region_parameters(parameters['focus_region'])
        
        if 'noising' in parameters:
            self._set_noising_parameters(parameters['noising'])
        
        if 'others' in parameters:
            self._set_others_parameters(parameters['others'])
    
    def _set_focus_region_parameters(self, focus_params):
        """设置焦点区域参数"""
        if 'type' in focus_params:
            type_index = self.ui.FocusRegionTypeCombox.findText(focus_params['type'])
            if type_index >= 0:
                self.ui.FocusRegionTypeCombox.setCurrentIndex(type_index)
        
        if 'qr_min' in focus_params:
            self.ui.FocusRegionQrMinValue.setText(str(focus_params['qr_min']))
        if 'qr_max' in focus_params:
            self.ui.FocusRegionQrMaxValue.setText(str(focus_params['qr_max']))
        if 'qz_min' in focus_params:
            self.ui.FocusRegionQzMinValue.setText(str(focus_params['qz_min']))
        if 'qz_max' in focus_params:
            self.ui.FocusRegionQzMaxValue.setText(str(focus_params['qz_max']))
    
    def _set_noising_parameters(self, noising_params):
        """设置噪声参数"""
        if 'type' in noising_params:
            type_index = self.ui.NoisingTypeCombox.findText(noising_params['type'])
            if type_index >= 0:
                self.ui.NoisingTypeCombox.setCurrentIndex(type_index)
        
        if 'snr_min' in noising_params:
            self.ui.NoisingGaussianSnrMinValue.setText(str(noising_params['snr_min']))
        if 'snr_max' in noising_params:
            self.ui.NoisingGaussianSnrMaxValue.setText(str(noising_params['snr_max']))
    
    def _set_others_parameters(self, others_params):
        """设置其他处理参数"""
        if 'crop_edge' in others_params:
            self.ui.OthersCropEdgeCheckBox.setChecked(others_params['crop_edge'])
        if 'add_mask' in others_params:
            self.ui.OthersAddMaskCheckBox.setChecked(others_params['add_mask'])
        if 'normalize' in others_params:
            self.ui.OthersNormalizeCheckBox.setChecked(others_params['normalize'])
        if 'logarization' in others_params:
            self.ui.OthersLogarizationCheckBox.setChecked(others_params['logarization'])
        
        # 设置边缘裁剪参数
        if 'crop_edge_params' in others_params:
            crop_params = others_params['crop_edge_params']
            self.ui.OthersCropEdgeUpMinValue.setText(str(crop_params.get('up_min', 0)))
            self.ui.OthersCropEdgeUpMaxValue.setText(str(crop_params.get('up_max', 20)))
            # 设置更多参数...
        
        # 设置归一化参数
        if 'normalize_params' in others_params:
            norm_params = others_params['normalize_params']
            self.ui.OthersNormalizeMinimumValue.setText(str(norm_params.get('minimum', 0)))
            self.ui.OthersNormalizeMaxmumValue.setText(str(norm_params.get('maximum', 20)))
    
    def validate_parameters(self):
        """验证预处理参数"""
        try:
            params = self.get_parameters()
            
            # 验证焦点区域
            focus_params = params.get('focus_region', {})
            qr_min = focus_params.get('qr_min', 0)
            qr_max = focus_params.get('qr_max', 0)
            qz_min = focus_params.get('qz_min', 0)
            qz_max = focus_params.get('qz_max', 0)
            
            if qr_min >= qr_max or qz_min >= qz_max:
                return False, "焦点区域范围无效"
            
            # 验证噪声参数
            noising_params = params.get('noising', {})
            snr_min = noising_params.get('snr_min', 0)
            snr_max = noising_params.get('snr_max', 0)
            
            if snr_min >= snr_max:
                return False, "信噪比范围无效"
            
            # 验证边缘裁剪参数
            others_params = params.get('others', {})
            if others_params.get('crop_edge', False):
                crop_params = others_params.get('crop_edge_params', {})
                for direction in ['up', 'down', 'left', 'right']:
                    min_val = crop_params.get(f'{direction}_min', 0)
                    max_val = crop_params.get(f'{direction}_max', 0)
                    if min_val >= max_val or min_val < 0:
                        return False, f"{direction}方向边缘裁剪参数无效"
            
            return True, "预处理参数有效"
            
        except Exception as e:
            return False, f"参数验证错误: {str(e)}"
    
    def reset_to_defaults(self):
        """重置为默认参数"""
        self.set_parameters(self.default_parameters)
    
    def _emit_parameters_changed(self):
        """发出参数改变信号"""
        parameters = self.get_parameters()
        self.parameters_changed.emit("预处理参数", parameters)
    
    def _show_focus_region_parameters(self):
        """显示焦点区域参数详情"""
        params = self._get_focus_region_parameters()
        message = f"""焦点区域参数:
类型: {params.get('type', 'N/A')}
qr 范围: {params.get('qr_min', 'N/A')} - {params.get('qr_max', 'N/A')} nm⁻¹
qz 范围: {params.get('qz_min', 'N/A')} - {params.get('qz_max', 'N/A')} nm⁻¹"""
        
        # 获取主窗口作为父窗口
        main_window = self.parent.parent if hasattr(self.parent, 'parent') else None
        QMessageBox.information(main_window, "焦点区域参数", message)
    
    def _show_noising_parameters(self):
        """显示噪声参数详情"""
        params = self._get_noising_parameters()
        message = f"""噪声参数:
类型: {params.get('type', 'N/A')}
信噪比范围: {params.get('snr_min', 'N/A')} - {params.get('snr_max', 'N/A')} dB"""
        
        # 获取主窗口作为父窗口
        main_window = self.parent.parent if hasattr(self.parent, 'parent') else None
        QMessageBox.information(main_window, "噪声参数", message)
    
    def _show_others_parameters(self):
        """显示其他处理参数详情"""
        params = self._get_others_parameters()
        
        enabled_features = []
        if params.get('crop_edge', False):
            enabled_features.append("边缘裁剪")
        if params.get('add_mask', False):
            enabled_features.append("添加掩码")
        if params.get('normalize', False):
            enabled_features.append("归一化")
        if params.get('logarization', False):
            enabled_features.append("对数化")
        
        message = f"""其他处理参数:
启用的功能: {', '.join(enabled_features) if enabled_features else '无'}
掩码文件: {params.get('mask_file_path', '未选择')}"""
        
        # 获取主窗口作为父窗口
        main_window = self.parent.parent if hasattr(self.parent, 'parent') else None
        QMessageBox.information(main_window, "其他处理参数", message)
