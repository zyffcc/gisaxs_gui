"""
通用参数触发管理器
用于管理所有参数控件的智能触发机制，避免输入过程中的频繁触发
"""

from PyQt5.QtCore import QObject, QTimer
from PyQt5.QtWidgets import QDoubleSpinBox, QSpinBox
from typing import Dict, Callable, Optional

from .parameter_trigger_compatibility import LegacyParameterTriggerMixin
from .parameter_trigger_diagnostics import ParameterTriggerDiagnosticsMixin


class UniversalParameterTriggerManager(
    ParameterTriggerDiagnosticsMixin,
    LegacyParameterTriggerMixin,
    QObject,
):
    """通用参数触发管理器"""

    def __init__(self, parent=None, *, settings_repository=None):
        super().__init__(parent)
        self.settings_repository = settings_repository

        # 旧机制数据结构（向后兼容）
        self._wheel_timers: Dict[str, QTimer] = {}
        self._save_timers: Dict[str, QTimer] = {}
        self._parameter_handlers: Dict[str, dict] = {}
        self.wheel_delay = 300
        self.save_delay = 500

        # 新：meta 驱动的统一去抖注册表 {widget_id: {widget, meta, last, pending, timer}}
        self._meta_registry: Dict[str, dict] = {}

    def register_parameter_widget(self,
                                  widget,
                                  widget_id: str,
                                  category: str,
                                  immediate_handler: Callable,
                                  delayed_handler: Optional[Callable] = None,
                                  custom_wheel_delay: Optional[int] = None,
                                  custom_save_delay: Optional[int] = None,
                                  connect_signals: bool = True,
                                  meta: Optional[dict] = None):
        """
        注册参数控件的触发处理

        Args:
            widget: QDoubleSpinBox或QSpinBox控件
            widget_id: 控件唯一标识符
            category: 控件分类（用于分组保存）
            immediate_handler: 立即触发处理函数 (回车/焦点丢失)
            delayed_handler: 延迟触发处理函数 (滚轮)，如果为None则使用immediate_handler
            custom_wheel_delay: 自定义滚轮延迟时间
            custom_save_delay: 自定义保存延迟时间
        """

        if not isinstance(widget, (QDoubleSpinBox, QSpinBox)):
            raise ValueError("Widget must be QDoubleSpinBox or QSpinBox")

        if meta:
            # 使用新 meta 模式（忽略旧 immediate/delayed 机制）
            self._register_meta_widget(widget, widget_id, category, meta)
        else:
            handler_info = {
                'widget': widget,
                'category': category,
                'immediate_handler': immediate_handler,
                'delayed_handler': delayed_handler or immediate_handler,
                'wheel_delay': custom_wheel_delay or self.wheel_delay,
                'save_delay': custom_save_delay or self.save_delay
            }
            self._parameter_handlers[widget_id] = handler_info
            if connect_signals:
                self._setup_widget_signals(widget, widget_id, handler_info)

    def _register_meta_widget(self, widget, widget_id: str, category: str, meta: dict):
        # 设定默认 meta 值
        meta = dict(meta)  # 复制
        meta.setdefault('debounce_ms', 280)
        meta.setdefault('epsilon_abs', 1e-12)
        meta.setdefault('epsilon_rel', 1e-10)
        meta.setdefault('persist', 'none')  # none | model_particle | model_global | settings | custom
        meta.setdefault('trigger_fit', False)
        meta.setdefault('after_commit', None)  # callable(info, value)
        meta.setdefault('custom_persist', None)  # callable(info, value)
        meta.setdefault('category', category)
        # 连接模式: 'changed' | 'finished' | 'external'（不自动连接）
        connect_mode = meta.setdefault('connect_mode', 'changed')

        timer = QTimer()
        timer.setSingleShot(True)
        timer.timeout.connect(lambda wid=widget_id: self._commit_meta_widget(wid))

        self._meta_registry[widget_id] = {
            'widget': widget,
            'meta': meta,
            'last_value': widget.value() if hasattr(widget, 'value') else None,
            'pending_value': None,
            'timer': timer
        }

        # 根据信号连接模式连接
        try:
            if connect_mode == 'finished':
                connected = False
                # 优先使用编辑完成
                if hasattr(widget, 'editingFinished'):
                    try:
                        widget.editingFinished.connect(lambda wid=widget_id: self._commit_meta_widget(wid))
                        connected = True
                    except Exception:
                        pass
                # 尝试回车提交（部分编辑类控件，如 QLineEdit）
                if not connected and hasattr(widget, 'returnPressed'):
                    try:
                        widget.returnPressed.connect(lambda wid=widget_id: self._commit_meta_widget(wid))
                        connected = True
                    except Exception:
                        pass
                # 兜底：若没有完成类信号，退回 changed 去抖
                if not connected and hasattr(widget, 'valueChanged'):
                    widget.valueChanged.connect(lambda _v, wid=widget_id: self._on_meta_value_changed(wid))
            elif connect_mode == 'external':
                # 不自动连接，由外部自行连接
                pass
            else:
                # 默认 changed 去抖
                if hasattr(widget, 'valueChanged'):
                    widget.valueChanged.connect(lambda _v, wid=widget_id: self._on_meta_value_changed(wid))
        except Exception:
            # 避免连接异常导致崩溃
            pass

    def _on_meta_value_changed(self, widget_id: str):
        info = self._meta_registry.get(widget_id)
        if not info:
            return
        w = info['widget']
        try:
            new_val = w.value()
        except Exception:
            return
        info['pending_value'] = new_val
        t = info['timer']
        if t.isActive():
            t.stop()
        t.start(info['meta']['debounce_ms'])

    def _commit_meta_widget(self, widget_id: str):
        info = self._meta_registry.get(widget_id)
        if not info:
            return
        # 停止并清理去抖计时器，避免残留触发
        try:
            t = info.get('timer')
            if t and t.isActive():
                t.stop()
        except Exception:
            pass

        meta = info['meta']
        pending = info.get('pending_value')
        # 在 finished 模式下，可能没有经过 valueChanged；此时直接读取控件当前值
        if pending is None:
            w = info.get('widget')
            if w is None:
                return
            try:
                pending = w.value() if hasattr(w, 'value') else None
            except Exception:
                pending = None
            if pending is None:
                return

        old = info.get('last_value')
        changed = True
        if old is not None:
            eps_abs = meta['epsilon_abs']
            eps_rel = meta['epsilon_rel']
            if abs(pending - old) <= (eps_abs + eps_rel * abs(old)):
                changed = False  # 未变化

        # 无论是否变化，都清空 pending，避免“只生效一次”的陈旧值干扰后续提交
        info['pending_value'] = None
        if not changed:
            return
        # 持久化
        persisted_ok = self._persist_meta(info, pending)
        if persisted_ok:
            info['last_value'] = pending
        # after_commit
        cb = meta.get('after_commit')
        if callable(cb):
            try:
                cb(info, pending)
            except Exception as e:
                print(f"after_commit error for {widget_id}: {e}")
        # trigger fit
        if meta.get('trigger_fit'):
            owner = self.parent()
            if owner and hasattr(owner, '_is_in_fitting_mode'):
                try:
                    # 仅在当前处于拟合模式时触发，避免无谓计算
                    if owner._is_in_fitting_mode():
                        owner._add_particle_message("🔄 Debounced meta trigger fitting")
                        owner._perform_manual_fitting()
                except Exception as e:
                    print(f"trigger_fit error: {e}")

    def _persist_meta(self, info: dict, value) -> bool:
        meta = info['meta']
        mode = meta.get('persist', 'none')
        try:
            if mode == 'none':
                return True
            elif mode == 'model_particle':
                # 尝试多层查找 model_params_manager
                mp = getattr(self, 'model_params_manager', None)
                if mp is None and self.parent() is not None:
                    mp = getattr(self.parent(), 'model_params_manager', None)
                if not mp:
                    return False
                pid = meta.get('particle_id')
                shape = meta.get('shape')
                param = meta.get('param')
                if not (pid and shape and param):
                    return False
                # 确保 particle_id 规范: 允许外部传入 'particle_1' 或 '1'
                if not str(pid).startswith('particle_'):
                    particle_key = f'particle_{pid}'
                else:
                    particle_key = pid
                if mp.set_particle_parameter('fitting', particle_key, shape, param, value):
                    mp.save_parameters()
                    return True
                return False
            elif mode == 'model_global':
                mp = getattr(self, 'model_params_manager', None)
                if mp is None and self.parent() is not None:
                    mp = getattr(self.parent(), 'model_params_manager', None)
                if not mp:
                    return False
                gparam = meta.get('param')
                if mp.set_global_parameter('fitting', gparam, value):
                    mp.save_parameters()
                    return True
                return False
            elif mode in {'settings', 'global_params'}:
                key_path = meta.get('key_path')  # e.g. ('fitting','detector.beam_center_x')
                if self.settings_repository is not None and key_path and len(key_path) == 2:
                    section, key = key_path
                    self.settings_repository.set(section, key, value)
                    self.settings_repository.save()
                    return True
                return False
            elif mode == 'custom':
                fn = meta.get('custom_persist')
                if callable(fn):
                    return bool(fn(info, value))
                return False
            else:
                return False
        except Exception as e:
            print(f"persist error ({mode}): {e}")
            return False

    def unregister_widget(self, widget_id: str):
        """取消注册控件"""
        if widget_id in self._meta_registry:
            timer = self._meta_registry[widget_id]['timer']
            if timer.isActive():
                timer.stop()
            timer.deleteLater()
            del self._meta_registry[widget_id]
        # 清理定时器
        if widget_id in self._wheel_timers:
            timer = self._wheel_timers[widget_id]
            if timer.isActive():
                timer.stop()
            timer.deleteLater()
            del self._wheel_timers[widget_id]

        # 清理处理函数映射
        if widget_id in self._parameter_handlers:
            del self._parameter_handlers[widget_id]

    def cleanup_all(self):
        """清理所有定时器和资源"""
        # 新 meta timers
        for wid, info in self._meta_registry.items():
            t = info['timer']
            if t.isActive():
                t.stop()
            t.deleteLater()
        self._meta_registry.clear()
        # 清理滚轮定时器
        for timer in self._wheel_timers.values():
            if timer.isActive():
                timer.stop()
            timer.deleteLater()
        self._wheel_timers.clear()

        # 清理保存定时器
        for timer in self._save_timers.values():
            if timer.isActive():
                timer.stop()
            timer.deleteLater()
        self._save_timers.clear()

        # 清理处理函数映射
        self._parameter_handlers.clear()

    def get_registered_widgets(self) -> Dict[str, dict]:
        """获取所有已注册的控件信息"""
        merged = {**self._parameter_handlers}
        for k, v in self._meta_registry.items():
            merged[k] = {'widget': v['widget'], 'meta': v['meta'], 'last_value': v['last_value']}
        return merged

    def __del__(self):
        """析构函数，确保资源清理"""
        try:
            self.cleanup_all()
        except:
            pass  # 忽略析构时的错误
