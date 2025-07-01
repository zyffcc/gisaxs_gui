"""
光束参数控制器 - 管理GISAXS实验的光束参数
"""

from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtWidgets import QMessageBox


class BeamController(QObject):
    """光束参数控制器"""
    
    # 参数改变信号
    parameters_changed = pyqtSignal(str, dict)
    
    def __init__(self, ui, parent=None):
        super().__init__(parent)
        self.ui = ui
        self.parent = parent
        
        # 默认参数
        self.default_parameters = {
            'wavelength': 0.1,  # nm
            'grazing_angle': 0.4,  # degrees
        }
        
        # 设置信号连接
        self._setup_connections()
    
    def _setup_connections(self):
        """设置信号连接"""
        # 波长变化
        self.ui.wavelengthValue.textChanged.connect(self._on_wavelength_changed)
        
        # 掠射角变化
        self.ui.angleValue.textChanged.connect(self._on_angle_changed)
    
    def initialize(self):
        """初始化光束参数"""
        self.set_parameters(self.default_parameters)
    
    def _on_wavelength_changed(self):
        """波长参数改变处理"""
        try:
            wavelength = float(self.ui.wavelengthValue.text())
            self._emit_parameters_changed()
        except ValueError:
            # 输入无效时不处理
            pass
    
    def _on_angle_changed(self):
        """掠射角参数改变处理"""
        try:
            angle = float(self.ui.angleValue.text())
            self._emit_parameters_changed()
        except ValueError:
            # 输入无效时不处理
            pass
    
    def get_parameters(self):
        """获取当前光束参数"""
        try:
            parameters = {
                'wavelength': float(self.ui.wavelengthValue.text()),
                'grazing_angle': float(self.ui.angleValue.text()),
            }
            return parameters
        except ValueError:
            return self.default_parameters.copy()
    
    def set_parameters(self, parameters):
        """设置光束参数"""
        if 'wavelength' in parameters:
            self.ui.wavelengthValue.setText(str(parameters['wavelength']))
        
        if 'grazing_angle' in parameters:
            self.ui.angleValue.setText(str(parameters['grazing_angle']))
        
        self._emit_parameters_changed()
    
    def validate_parameters(self):
        """验证光束参数"""
        try:
            params = self.get_parameters()
            
            # 验证波长
            wavelength = params.get('wavelength', 0)
            if wavelength <= 0 or wavelength > 10:
                return False, "波长必须在0-10 nm范围内"
            
            # 验证掠射角
            grazing_angle = params.get('grazing_angle', 0)
            if grazing_angle <= 0 or grazing_angle > 90:
                return False, "掠射角必须在0-90度范围内"
            
            return True, "光束参数有效"
            
        except Exception as e:
            return False, f"参数验证错误: {str(e)}"
    
    def reset_to_defaults(self):
        """重置为默认参数"""
        self.set_parameters(self.default_parameters)
    
    def _emit_parameters_changed(self):
        """发出参数改变信号"""
        parameters = self.get_parameters()
        self.parameters_changed.emit("光束参数", parameters)
