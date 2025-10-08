"""DEPRECATED.

本文件已被 meta 驱动的 `UniversalParameterTriggerManager` 完全取代。
保留一个最小空壳以避免历史 import 造成崩溃；请在后续清理周期中删除此文件。
"""

from utils.universal_parameter_trigger_manager import UniversalParameterTriggerManager  # noqa: F401


class FittingParameterTriggerManager(UniversalParameterTriggerManager):  # type: ignore
    """兼容占位：不再提供任何附加行为。"""

    def __init__(self, *args, **kwargs):  # pragma: no cover - legacy shim
        super().__init__(*args, **kwargs)
    
    def register_fitting_global_widget(self, widget, param: str):
        """注册fitting全局参数控件"""
        control_id = f"global_{param}"
        
        immediate_handler = lambda value: self._on_global_parameter_immediate(param, value)
        delayed_handler = lambda value: self._on_global_parameter_delayed(param, value)
        
        self.register_parameter_widget(
            widget=widget,
            widget_id=control_id,
            category='fitting_global',
            immediate_handler=immediate_handler,
                delayed_handler=delayed_handler,
                connect_signals=False
        )
    
    def register_gisaxs_input_widget(self, widget, param_name: str):
        """注册GISAXS输入参数控件"""
        control_id = f"gisaxs_{param_name}"
        
        handler = lambda value: self._on_gisaxs_input_parameter_changed(param_name, value)
        
        self.register_parameter_widget(
            widget=widget,
            widget_id=control_id,
            category='gisaxs_input',
            immediate_handler=handler,
                delayed_handler=handler,
                connect_signals=False  # 使用控制器自定义去抖
        )
    
    def register_detector_param_widget(self, widget, param_name: str):
        """注册探测器参数控件"""
        control_id = f"detector_{param_name}"
        
        handler = lambda value: self._on_detector_parameter_changed(param_name, value)
        
        self.register_parameter_widget(
            widget=widget,
            widget_id=control_id,
            category='detector_params',
            immediate_handler=handler,
                delayed_handler=handler  # 对话框目前保留内部连接，可后续改为自定义去抖
        )
    
    def _on_particle_parameter_immediate(self, widget_id: int, shape: str, param: str, value: float):
        """处理粒子参数立即触发"""
        try:
            particle_id = f"particle_{widget_id}"
            self.model_params_manager.set_particle_parameter('fitting', particle_id, shape, param, value)
            self._add_message(f"Parameter {param} for particle {widget_id} saved: {value}", "INFO")
        except Exception as e:
            self._add_message(f"Failed to save parameter {param} for particle {widget_id}: {e}", "ERROR")
    
    def _on_particle_parameter_delayed(self, widget_id: int, shape: str, param: str, value: float):
        """处理粒子参数延迟触发"""
        try:
            particle_id = f"particle_{widget_id}"
            self.model_params_manager.set_particle_parameter('fitting', particle_id, shape, param, value)
            self._add_message(f"Parameter {param} for particle {widget_id} updated: {value}", "INFO")
        except Exception as e:
            self._add_message(f"Failed to save wheel parameter {param} for particle {widget_id}: {e}", "ERROR")
    
    def _on_global_parameter_immediate(self, param: str, value: float):
        """处理全局参数立即触发"""
        try:
            self.model_params_manager.set_global_parameter('fitting', param, value)
            self._add_message(f"Global parameter {param} saved: {value}", "INFO")
        except Exception as e:
            self._add_message(f"Failed to save global parameter {param}: {e}", "ERROR")
    
    def _on_global_parameter_delayed(self, param: str, value: float):
        """处理全局参数延迟触发"""
        try:
            self.model_params_manager.set_global_parameter('fitting', param, value)
            self._add_message(f"Global parameter {param} updated: {value}", "INFO")
        except Exception as e:
            self._add_message(f"Failed to save wheel global parameter {param}: {e}", "ERROR")
    
    def _on_gisaxs_input_parameter_changed(self, param_name: str, value: float):
        """处理GISAXS输入参数变化"""
        try:
            # 调用原有的处理逻辑
            if hasattr(self.fitting_controller, '_on_parameter_display_changed'):
                self.fitting_controller._on_parameter_display_changed()
            self._add_message(f"GISAXS parameter {param_name} updated: {value}", "INFO")
        except Exception as e:
            self._add_message(f"Failed to handle GISAXS input parameter {param_name}: {e}", "ERROR")
    
    def _on_detector_parameter_changed(self, param_name: str, value: float):
        """处理探测器参数变化"""
        try:
            # 这里可以添加探测器参数的具体处理逻辑
            self._add_message(f"Detector parameter {param_name} updated: {value}", "INFO")
        except Exception as e:
            self._add_message(f"Failed to handle detector parameter {param_name}: {e}", "ERROR")
    
    def _trigger_immediate_save(self, category: str):
        """触发立即保存"""
        handler = self._save_handlers.get(category)
        if handler:
            handler(immediate=True)
    
    def _execute_delayed_save(self, category: str):
        """执行延迟保存"""
        handler = self._save_handlers.get(category)
        if handler:
            handler(immediate=False)
    
    def _save_fitting_particles(self, immediate: bool = False):
        """保存fitting粒子参数"""
        try:
            self.model_params_manager.save_parameters()
            if immediate:
                self._add_message("Fitting particle parameters saved immediately", "INFO")
            else:
                self._add_message("Particle parameters saved", "INFO")
        except Exception as e:
            self._add_message(f"Failed to save fitting particle parameters: {e}", "ERROR")
    
    def _save_fitting_global(self, immediate: bool = False):
        """保存fitting全局参数"""
        try:
            self.model_params_manager.save_parameters()
            if immediate:
                self._add_message("Fitting global parameters saved immediately", "INFO")
            else:
                self._add_message("Global parameters saved", "INFO")
        except Exception as e:
            self._add_message(f"Failed to save fitting global parameters: {e}", "ERROR")
    
    def _save_gisaxs_input(self, immediate: bool = False):
        """保存GISAXS输入参数"""
        try:
            # 调用原有的保存逻辑（如果有的话）
            if hasattr(self.fitting_controller, 'save_session_parameters'):
                self.fitting_controller.save_session_parameters()
            
            level = "INFO" if immediate else "DEBUG"
            self._add_message(f"GISAXS input parameters saved", level)
        except Exception as e:
            self._add_message(f"Failed to save GISAXS input parameters: {e}", "ERROR")
    
    def _save_detector_params(self, immediate: bool = False):
        """保存探测器参数"""
        try:
            # 这里可以添加探测器参数的具体保存逻辑
            level = "INFO" if immediate else "DEBUG"
            self._add_message(f"Detector parameters saved", level)
        except Exception as e:
            self._add_message(f"Failed to save detector parameters: {e}", "ERROR")
    
    def _add_message(self, message: str, level: str):
        """添加日志消息"""
        try:
            if level == "INFO":
                self.fitting_controller._add_fitting_success(message)
            elif level == "DEBUG":
                self.fitting_controller._add_fitting_message(message, "DEBUG")
            elif level == "ERROR":
                self.fitting_controller._add_fitting_error(message)
        except:
            print(f"[{level}] {message}")  # fallback to print