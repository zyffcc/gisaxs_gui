"""
DetectorParametersDialog专用的参数触发管理器
"""

from utils.universal_parameter_trigger_manager import UniversalParameterTriggerManager


class DetectorParameterTriggerManager(UniversalParameterTriggerManager):
    """DetectorParametersDialog专用的参数触发管理器"""
    
    def __init__(self, detector_dialog, parent=None):
        super().__init__(parent)
        
        self.detector_dialog = detector_dialog
    
    def register_detector_widget(self, widget, param_name: str, connect_mode: str = 'finished'):
        """注册探测器参数控件 (meta 去抖 + global_params 持久化)
        connect_mode: 'finished' | 'changed' | 'external'
        """
        def _after_commit(info, value, p=param_name):
            try:
                if hasattr(self.detector_dialog, '_on_parameter_changed_internal'):
                    self.detector_dialog._on_parameter_changed_internal()
                print(f"[META|Detector] {p} -> {value}")
            except Exception:
                pass
        meta = {
            'persist': 'global_params',
            'key_path': ('fitting', f'detector.{param_name}'),
            'debounce_ms': 250,
            'epsilon_abs': 1e-12,
            'epsilon_rel': 1e-10,
            'after_commit': _after_commit,
            'trigger_fit': False,
            'connect_mode': connect_mode,
        }
        self.register_parameter_widget(
            widget=widget,
            widget_id=f'meta_detector_{param_name}',
            category='detector_params',
            immediate_handler=lambda v: None,
            delayed_handler=None,
            connect_signals=True,
            meta=meta
        )
    
    # 旧 immediate/delayed 保存逻辑已由 meta 替代；保留空实现兼容潜在外部调用
    def _trigger_immediate_save(self, category: str):
        pass

    def _execute_delayed_save(self, category: str):
        pass