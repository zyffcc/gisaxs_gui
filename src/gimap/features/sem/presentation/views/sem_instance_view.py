"""Static controls for choosing a local SEM instance-detection method."""

from PyQt5.QtWidgets import (
    QComboBox, QDoubleSpinBox, QFormLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QVBoxLayout, QWidget,
)

from src.gimap.app.presentation.components import AdvancedSection, ParameterSection


def _number(parent, minimum, maximum, value, decimals=4):
    control = QDoubleSpinBox(parent)
    control.setRange(minimum, maximum)
    control.setDecimals(decimals)
    control.setValue(value)
    control.setKeyboardTracking(False)
    return control


def _path_row(parent, placeholder):
    row = QWidget(parent)
    layout = QHBoxLayout(row)
    layout.setContentsMargins(0, 0, 0, 0)
    edit = QLineEdit(row)
    edit.setPlaceholderText(placeholder)
    button = QPushButton("Browse…", row)
    layout.addWidget(edit, 1)
    layout.addWidget(button)
    return row, edit, button


class SemInstanceView:
    def _instance_controls(self, layout, parent):
        self.detection_section = ParameterSection("Particle detection", parent=parent)
        self.detection_method_combo = QComboBox(self.detection_section)
        for text, value in (
            ("StarDist (local AI)", "stardist"),
            ("Threshold / watershed", "threshold"),
            ("Existing instance labels", "labels"),
        ):
            self.detection_method_combo.addItem(text, value)
        self.detection_method_combo.setCurrentIndex(1)
        self.detection_section.add_widget(self.detection_method_combo)

        self.ai_resources_widget = QWidget(self.detection_section)
        resources = QVBoxLayout(self.ai_resources_widget)
        resources.setContentsMargins(0, 0, 0, 0)
        python_row, self.python_executable_edit, self.browse_python_button = _path_row(
            self.ai_resources_widget, "Select the local Python with StarDist installed",
        )
        model_row, self.model_directory_edit, self.browse_instance_model_button = _path_row(
            self.ai_resources_widget, "Select a local StarDist model directory",
        )
        resources.addWidget(QLabel("Local Python executable", self.ai_resources_widget))
        resources.addWidget(python_row)
        resources.addWidget(QLabel("Local instance model", self.ai_resources_widget))
        resources.addWidget(model_row)
        model_note = QLabel(
            "General pretrained nuclei models are not validated for SEM. Review particle boundaries before using size statistics.",
            self.ai_resources_widget,
        )
        model_note.setWordWrap(True)
        model_note.setProperty("gimapMeta", True)
        resources.addWidget(model_note)
        self.check_instance_resources_button = QPushButton("Check local resources", self.ai_resources_widget)
        resources.addWidget(self.check_instance_resources_button)
        self.detection_section.add_widget(self.ai_resources_widget)

        self.labels_widget = QWidget(self.detection_section)
        labels_layout = QVBoxLayout(self.labels_widget)
        labels_layout.setContentsMargins(0, 0, 0, 0)
        labels_row, self.label_path_edit, self.browse_instance_labels_button = _path_row(
            self.labels_widget, "Select an integer instance-label TIFF or PNG",
        )
        labels_layout.addWidget(labels_row)
        hint = QLabel(
            "Each particle needs a distinct positive integer label; 0 is background. "
            "Label image dimensions must match the SEM image or its selected ROI.", self.labels_widget,
        )
        hint.setWordWrap(True)
        hint.setProperty("gimapMeta", True)
        labels_layout.addWidget(hint)
        self.detection_section.add_widget(self.labels_widget)
        self.instance_status_label = QLabel("", self.detection_section)
        self.instance_status_label.setWordWrap(True)
        self.instance_status_label.setProperty("gimapMeta", True)
        self.detection_section.add_widget(self.instance_status_label)

        self.instance_ai_options = AdvancedSection(
            "Instance AI options", "Uses local files. No model downloads or automatic method fallback.",
            self.detection_section,
        )
        form = QFormLayout()
        self.ai_scale_spin = _number(parent, 0.25, 4.0, 1.0, decimals=2)
        self.ai_scale_spin.setSingleStep(0.25)
        self.probability_threshold_spin = _number(parent, 0.0, 1.0, 0.0)
        self.overlap_threshold_spin = _number(parent, 0.0, 1.0, 0.0)
        self.probability_threshold_spin.setSpecialValueText("Model default")
        self.overlap_threshold_spin.setSpecialValueText("Model default")
        self.normalize_low_spin = _number(parent, 0.0, 100.0, 1.0, decimals=2)
        self.normalize_high_spin = _number(parent, 0.0, 100.0, 99.8, decimals=2)
        for text, control in (
            ("AI image scale", self.ai_scale_spin),
            ("Probability threshold", self.probability_threshold_spin),
            ("NMS overlap threshold", self.overlap_threshold_spin),
            ("Normalize low percentile", self.normalize_low_spin),
            ("Normalize high percentile", self.normalize_high_spin),
        ):
            form.addRow(text, control)
        self.instance_ai_options.add_layout(form)
        self.detection_section.add_widget(self.instance_ai_options)
        layout.addWidget(self.detection_section)


__all__ = ["SemInstanceView"]
