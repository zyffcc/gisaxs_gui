from PyQt5.QtWidgets import (
    QDialog,
)
from PyQt5.QtCore import pyqtSignal

from src.gimap.app.bootstrap import create_standalone_legacy_context
from src.gimap.features.fitting.domain import (
    DetectorSettings,
    energy_to_wavelength,
    wavelength_to_energy,
)

from .detector_parameter_trigger import DetectorParameterTriggerManager
from .views import DetectorParametersDialogView


class DetectorParametersDialog(QDialog, DetectorParametersDialogView):
    """探测器参数对话框"""

    # 定义信号
    parameters_changed = pyqtSignal(dict)  # 参数改变时发出信号

    def __init__(self, parent=None, *, view_model=None):
        super().__init__(parent)
        if view_model is None:
            view_model = getattr(parent, "fitting_view_model", None)
        if view_model is None:
            from ..bootstrap import create_fitting_view_model

            view_model = create_fitting_view_model(create_standalone_legacy_context())
        self.view_model = view_model
        self.setupUi(self)
        self.setModal(False)  # 改为非模态对话框，允许同时操作主界面

        # 参数字典
        self.parameters = {}

        # 初始化触发管理器
        self.param_trigger_manager = DetectorParameterTriggerManager(self)

    # 去抖由 meta 触发管理器统一处理

        self._bind_form()

        # 加载当前参数值
        self._load_parameters()

        # 连接信号
        self._connect_signals()

    def _bind_form(self):
        """Connect behaviour to the Python View widget tree."""
        font = self.energy_relation_label.font()
        font.setPointSize(font.pointSize() - 1)
        self.energy_relation_label.setFont(font)
        self.apply_button.clicked.connect(self._apply_parameters)
        self.ok_button.clicked.connect(self._ok_clicked)
        self.cancel_button.clicked.connect(self.reject)

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
            settings = self.view_model.load_detector_settings()
            beam_x = settings.beam_center_x
            beam_y = settings.beam_center_y

            self.distance_spinbox.setValue(settings.distance)
            self.beam_center_x_spinbox.setValue(beam_x)
            self.beam_center_y_spinbox.setValue(beam_y)
            self.pixel_size_x_spinbox.setValue(settings.pixel_size_x)
            self.pixel_size_y_spinbox.setValue(settings.pixel_size_y)

            # 设置束流参数
            self.angle_spinbox.setValue(settings.grazing_angle)
            self.wavelength_spinbox.setValue(settings.wavelength)
            # 同步能量显示
            try:
                wl = self.wavelength_spinbox.value()
                if wl > 0:
                    self.energy_spinbox.blockSignals(True)
                    self.energy_spinbox.setValue(wavelength_to_energy(wl))
                    self.energy_spinbox.blockSignals(False)
            except Exception:
                pass

            # 设置显示选项 - 从fitting.detector中读取
            self.show_q_axis_checkbox.setChecked(settings.show_q_axis)

            # 重新连接信号
            self._connect_signals()

        except Exception as e:
            print(f"Failed to load detector parameters: {e}")
            # 确保重新连接信号
            self._connect_signals()

    def _on_parameter_changed(self):
        """参数改变时的处理（用于复选框等非数值控件）"""
        # 通过 ViewModel 自动保存到注入的 settings repository。
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
        settings = self._current_settings()
        return {
            **settings.__dict__,
            'energy': self.energy_spinbox.value(),  # 仅供外部查看，不单独持久化
        }

    def _current_settings(self) -> DetectorSettings:
        return DetectorSettings(
            distance=self.distance_spinbox.value(),
            grazing_angle=self.angle_spinbox.value(),
            wavelength=self.wavelength_spinbox.value(),
            beam_center_x=self.beam_center_x_spinbox.value(),
            beam_center_y=self.beam_center_y_spinbox.value(),
            pixel_size_x=self.pixel_size_x_spinbox.value(),
            pixel_size_y=self.pixel_size_y_spinbox.value(),
            show_q_axis=self.show_q_axis_checkbox.isChecked(),
        )

    def _persist_detector_value(self, _name: str, _value: float) -> bool:
        self.view_model.save_detector_settings(self._current_settings())
        return True

    # ===================== 波长/能量联动 =====================
    def _on_wavelength_changed_sync(self):
        if getattr(self, '_updating_energy_pair', False):
            return
        wl = self.wavelength_spinbox.value()
        if wl <= 0:
            return
        try:
            self._updating_energy_pair = True
            energy = wavelength_to_energy(wl)
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
            wl = energy_to_wavelength(energy)
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

            self.view_model.save_detector_settings(self._current_settings())

            print("Detector parameters saved successfully")

        except Exception as e:
            print(f"Failed to save detector parameters: {e}")

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
