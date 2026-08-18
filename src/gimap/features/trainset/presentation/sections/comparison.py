"""Comparison section for the Trainset page."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np


from PyQt5.QtCore import QTimer, Qt


from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QProgressBar,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)


from ..visualization_widgets import ArrayCanvas, ParameterCoverageWidget


class ComparisonMixin:
    """Own the comparison section."""

    @staticmethod
    def _append_parameter_tree(parent: QTreeWidgetItem, value: Any) -> None:
        if isinstance(value, dict):
            for key, child_value in value.items():
                child = QTreeWidgetItem(
                    (
                        str(key).replace("_", " "),
                        "" if isinstance(child_value, (dict, list)) else str(child_value),
                    )
                )
                parent.addChild(child)
                if isinstance(child_value, (dict, list)):
                    ComparisonMixin._append_parameter_tree(child, child_value)
        elif isinstance(value, list):
            for index, child_value in enumerate(value):
                child = QTreeWidgetItem(
                    (
                        f"item {index + 1}",
                        "" if isinstance(child_value, (dict, list)) else str(child_value),
                    )
                )
                parent.addChild(child)
                if isinstance(child_value, (dict, list)):
                    ComparisonMixin._append_parameter_tree(child, child_value)

    @staticmethod
    def _manual_config_entries(value: Any, prefix: str = ""):
        if isinstance(value, dict):
            for key, child in value.items():
                path = f"{prefix}.{key}" if prefix else str(key)
                yield from ComparisonMixin._manual_config_entries(child, path)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                path = f"{prefix}.{index}"
                yield from ComparisonMixin._manual_config_entries(child, path)
        elif isinstance(value, (bool, int, float, str)):
            yield prefix, value

    def show_comparison_parameters(self) -> None:
        if not self._comparison_details:
            return
        if self._parameter_dialog is not None and self._parameter_dialog.isVisible():
            self._parameter_dialog.raise_()
            self._parameter_dialog.activateWindow()
            return
        dialog = QDialog(self)
        self._parameter_dialog = dialog
        dialog.setAttribute(Qt.WA_DeleteOnClose, True)
        dialog.setWindowTitle("Parameters used and manual simulation")
        dialog.resize(1040, 720)
        dialog_layout = QVBoxLayout(dialog)
        note = QLabel(
            "Minimum, midpoint and maximum remain immutable audit snapshots. Manual simulation starts from one "
            "snapshot, lets you edit its physics, geometry, sample, mask and preprocessing settings, and renders "
            "an independent fourth image without replacing the saved three."
        )
        note.setWordWrap(True)
        dialog_layout.addWidget(note)
        tabs = QTabWidget()
        for key, heading in (
            ("minimum", "Minimum"),
            ("midpoint", "Midpoint"),
            ("maximum", "Maximum"),
        ):
            tree = QTreeWidget()
            tree.setHeaderLabels(("Parameter", "Value"))
            tree.header().setSectionResizeMode(0, QHeaderView.ResizeToContents)
            tree.header().setSectionResizeMode(1, QHeaderView.Stretch)
            root = QTreeWidgetItem((heading, ""))
            tree.addTopLevelItem(root)
            self._append_parameter_tree(root, self._comparison_details.get(key, {}))
            root.setExpanded(True)
            tabs.addTab(tree, heading)

        what_if_page = QWidget()
        what_if_layout = QVBoxLayout(what_if_page)
        source_row = QHBoxLayout()
        source_row.addWidget(QLabel("Start from"))
        source_combo = QComboBox()
        source_combo.addItem("Minimum snapshot", "minimum")
        source_combo.addItem("Midpoint snapshot", "midpoint")
        source_combo.addItem("Maximum snapshot", "maximum")
        source_combo.setCurrentIndex(1)
        source_row.addWidget(source_combo)
        source_row.addStretch(1)
        what_if_layout.addLayout(source_row)
        what_if_help = QLabel(
            "Auto-simulation is debounced. Geometry or physics edits rerun BornAgain; display-only changes do not. "
            "Mask/noise keep the same realization so image differences remain attributable to your edit."
        )
        what_if_help.setWordWrap(True)
        what_if_help.setProperty("infoPanel", True)
        what_if_layout.addWidget(what_if_help)
        editor_and_image = QSplitter(Qt.Horizontal)
        editor_tabs = QTabWidget()
        self._what_if_controls: Dict[str, QWidget] = {}
        physics_editor = QWidget()
        physics_form = QFormLayout(physics_editor)
        midpoint_values = self._comparison_details.get("midpoint", {}).get(
            "editable physics",
            self._comparison_details.get("midpoint", {}).get("physics values", {}),
        )
        scalar_values = {
            name: value
            for name, value in midpoint_values.items()
            if not str(name).startswith("__") and isinstance(value, (int, float, np.number))
        }
        ordered_names = list(self._comparison_parameter_specs) or list(scalar_values)
        for name in ordered_names:
            if name not in scalar_values:
                continue
            spec = self._comparison_parameter_specs.get(name, {})
            control = QDoubleSpinBox()
            control.setDecimals(6)
            control.setRange(-1e12, 1e12)
            control.setValue(float(scalar_values[name]))
            control.setKeyboardTracking(False)
            low = spec.get("minimum")
            high = spec.get("maximum")
            control.setToolTip(
                f"Configured training range: {low} to {high}. Manual values may go outside it for diagnosis."
                if low is not None and high is not None
                else "Editable physics value for the independent manual simulation."
            )
            physics_form.addRow(ParameterCoverageWidget._axis_label(name), control)
            self._what_if_controls[f"physics.{name}"] = control
        editor_tabs.addTab(self._scroll(physics_editor), "Physics values")

        config_roots = ("beam", "detector", "roi", "simulation", "sample", "mask", "preprocessing")
        grouped = {
            "Geometry": QWidget(),
            "Sample": QWidget(),
            "Mask & preprocessing": QWidget(),
        }
        forms = {name: QFormLayout(widget) for name, widget in grouped.items()}
        ignored_fragments = (
            ".parameters.",
            "simulation.grid_cache",
        )
        for path, value in self._manual_config_entries(self._comparison_config):
            if not path.startswith(config_roots) or any(
                fragment in path for fragment in ignored_fragments
            ):
                continue
            if path.endswith(".plugin") or path.endswith(".preset"):
                continue
            if path.startswith(("beam.", "detector.", "roi.", "simulation.")):
                group = "Geometry"
            elif path.startswith("sample."):
                group = "Sample"
            else:
                group = "Mask & preprocessing"
            if isinstance(value, bool):
                control = QCheckBox()
                control.setChecked(value)
            elif isinstance(value, int) and not isinstance(value, bool):
                control = QSpinBox()
                control.setRange(-1_000_000_000, 1_000_000_000)
                control.setValue(value)
            elif isinstance(value, float):
                control = QDoubleSpinBox()
                control.setDecimals(8)
                control.setRange(-1e12, 1e12)
                control.setValue(value)
                control.setKeyboardTracking(False)
            else:
                control = QLineEdit(str(value))
            control.setToolTip(
                f"Manual override for {path}. This changes only the fourth simulation; saved comparison snapshots stay unchanged."
            )
            forms[group].addRow(path.replace("_", " "), control)
            self._what_if_controls[f"config.{path}"] = control
        for group, widget in grouped.items():
            editor_tabs.addTab(self._scroll(widget), group)
        editor_and_image.addWidget(editor_tabs)
        result_panel = QWidget()
        result_layout = QVBoxLayout(result_panel)
        self._what_if_canvas = ArrayCanvas("Edit a value or press Simulate now")
        self._what_if_canvas.setMinimumSize(330, 290)
        self._what_if_status = QLabel(
            "Manual simulation is independent of the three saved comparison snapshots."
        )
        self._what_if_status.setWordWrap(True)
        self._what_if_progress = QProgressBar()
        self._what_if_progress.setRange(0, 0)
        self._what_if_progress.setVisible(False)
        result_layout.addWidget(self._what_if_canvas, 1)
        result_layout.addWidget(self._make_display_bar("manual"))
        preview_display = self._display_controls.get("preview", {})
        manual_display = self._display_controls.get("manual", {})
        if preview_display and manual_display:
            manual_display["colormap"].setCurrentText(preview_display["colormap"].currentText())
            manual_display["log"].setChecked(preview_display["log"].isChecked())
            manual_display["auto"].setChecked(preview_display["auto"].isChecked())
            manual_display["vmin"].setValue(preview_display["vmin"].value())
            manual_display["vmax"].setValue(preview_display["vmax"].value())
        result_layout.addWidget(self._what_if_progress)
        result_layout.addWidget(self._what_if_status)
        editor_and_image.addWidget(result_panel)
        editor_and_image.setSizes((500, 500))
        editor_and_image.setStretchFactor(0, 1)
        editor_and_image.setStretchFactor(1, 1)
        what_if_layout.addWidget(editor_and_image, 1)
        what_if_actions = QHBoxLayout()
        auto_simulate = QCheckBox("Auto-simulate after edits")
        auto_simulate.setChecked(True)
        simulate_now = QPushButton("Simulate now")
        simulate_now.setObjectName("primaryAction")
        what_if_actions.addWidget(auto_simulate)
        what_if_actions.addStretch(1)
        what_if_actions.addWidget(simulate_now)
        what_if_layout.addLayout(what_if_actions)

        update_timer = QTimer(dialog)
        update_timer.setSingleShot(True)
        update_timer.setInterval(700)

        def control_value(control: QWidget) -> Any:
            if isinstance(control, QCheckBox):
                return control.isChecked()
            if isinstance(control, (QSpinBox, QDoubleSpinBox)):
                return control.value()
            if isinstance(control, QComboBox):
                return control.currentText()
            return control.text() if isinstance(control, QLineEdit) else None

        def request_what_if() -> None:
            physics = {
                path.split(".", 1)[1]: control_value(control)
                for path, control in self._what_if_controls.items()
                if path.startswith("physics.")
            }
            overrides = {
                path.split(".", 1)[1]: control_value(control)
                for path, control in self._what_if_controls.items()
                if path.startswith("config.")
            }
            if physics:
                self.what_if_requested.emit({"physics": physics, "overrides": overrides})

        def schedule_what_if(*_args) -> None:
            if auto_simulate.isChecked():
                update_timer.start()

        def load_snapshot(*_args) -> None:
            key = str(source_combo.currentData())
            values = self._comparison_details.get(key, {}).get(
                "editable physics",
                self._comparison_details.get(key, {}).get("physics values", {}),
            )
            for path, control in self._what_if_controls.items():
                if not path.startswith("physics."):
                    continue
                name = path.split(".", 1)[1]
                if name in values and isinstance(values[name], (int, float, np.number)):
                    control.blockSignals(True)
                    control.setValue(float(values[name]))
                    control.blockSignals(False)
            schedule_what_if()

        update_timer.timeout.connect(request_what_if)
        source_combo.currentIndexChanged.connect(load_snapshot)
        simulate_now.clicked.connect(request_what_if)
        for control in self._what_if_controls.values():
            if isinstance(control, QCheckBox):
                control.toggled.connect(schedule_what_if)
            elif isinstance(control, (QSpinBox, QDoubleSpinBox)):
                control.valueChanged.connect(schedule_what_if)
            elif isinstance(control, QComboBox):
                control.currentTextChanged.connect(schedule_what_if)
            elif isinstance(control, QLineEdit):
                control.editingFinished.connect(schedule_what_if)
        tabs.addTab(what_if_page, "Manual simulation")
        dialog_layout.addWidget(tabs, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(dialog.close)
        dialog_layout.addWidget(buttons)
        dialog.destroyed.connect(lambda *_args: setattr(self, "_parameter_dialog", None))
        dialog.show()
        load_snapshot()

    def set_what_if_busy(self, busy: bool, message: str) -> None:
        if not hasattr(self, "_what_if_status"):
            return
        self._what_if_progress.setVisible(busy)
        self._what_if_status.setText(message)

    def set_what_if_result(self, image: np.ndarray, details: str) -> None:
        if self._parameter_dialog is None:
            return
        self._what_if_canvas.set_data(image)
        self._what_if_progress.setVisible(False)
        self._what_if_status.setText(details)

    def set_preview_stages(
        self,
        _reference,
        stages,
        stats: Dict[str, Any],
        spectrum_x: np.ndarray,
        spectrum_y: np.ndarray,
    ) -> None:
        """Compatibility adapter for older callers; all images are simulated."""
        final = np.asarray(stages[-1]["image"]) if stages else np.zeros((1, 1), dtype=np.float32)
        self.set_simulation_preview(
            {"minimum": final, "midpoint": final, "maximum": final},
            {"minimum": "Minimum", "midpoint": "Midpoint", "maximum": "Maximum"},
            stages,
            stats,
            spectrum_x,
            spectrum_y,
        )

    def set_parameter_samples(self, samples, parameter_names=None, parameter_specs=None) -> None:
        self.parameter_coverage.set_samples(samples, parameter_names, parameter_specs)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._apply_responsive_layout()
