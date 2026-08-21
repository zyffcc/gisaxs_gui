"""Particle Widget Editor for fitting presentation."""

from __future__ import annotations

from PyQt5.QtCore import Qt, QTimer, QPoint

from PyQt5.QtWidgets import (
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QMenu,
    QSizePolicy,
    QComboBox,
    QGridLayout,
    QLabel,
    QDoubleSpinBox,
    QPushButton,
)

from src.gimap.app.presentation import install_safe_wheel_behavior

from src.gimap.features.fitting.presentation.layout_primitives import (
    CurrentPageHeightStackedWidget,
    NoWheelDoubleSpinBox,
)

from ..binding_primitives import (
    COMPONENT_FORMULA_TOOLTIPS,
    COMPONENT_ORDER,
    COMPONENT_PARAMETER_SCHEMAS,
)


class ParticleWidgetEditorMixin:
    """Own particle widget editor behavior."""

    def _rebuild_particle_widget_editor(self, container: QWidget, widget_id: int) -> None:
        old_layout = container.layout()
        if old_layout is not None:
            while old_layout.count():
                item = old_layout.takeAt(0)
                child = item.widget()
                if child is not None:
                    child.setParent(None)
                    child.deleteLater()
            layout = old_layout
        else:
            layout = QVBoxLayout(container)
        container.setMinimumHeight(0)
        container.setMaximumHeight(16777215)
        layout.setContentsMargins(12, 10, 12, 12)
        layout.setSpacing(8)

        header = QWidget(container)
        header.setObjectName(f"fitParticleHeader_{widget_id}")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(10, 8, 10, 8)
        header_layout.setSpacing(10)
        title = QLabel(f"Component {widget_id}", header)
        title.setObjectName(f"fitParticleTitleLabel_{widget_id}")
        title.setMinimumWidth(88)
        title.setStyleSheet("font-weight: 600; color: #1f2937;")
        type_group = QWidget(header)
        type_group.setObjectName(f"fitParticleTypeGroup_{widget_id}")
        type_layout = QHBoxLayout(type_group)
        type_layout.setContentsMargins(0, 0, 0, 0)
        type_layout.setSpacing(4)
        type_label = QLabel("Type", type_group)
        type_label.setObjectName(f"fitParticleTypeLabel_{widget_id}")
        combo = QComboBox(type_group)
        combo.setObjectName(f"fitParticleShapeCombox_{widget_id}")
        combo.setMinimumWidth(158)
        combo.setMaximumWidth(236)
        for shape_name in COMPONENT_ORDER:
            combo.addItem(shape_name)
            combo.setItemData(
                combo.count() - 1, COMPONENT_FORMULA_TOOLTIPS[shape_name], Qt.ToolTipRole
            )
        combo.setToolTip(COMPONENT_FORMULA_TOOLTIPS["None"])
        type_layout.addWidget(type_label)
        type_layout.addWidget(combo)
        remove_button = QPushButton("Remove", header)
        remove_button.setObjectName(f"fitParticleRemoveButton_{widget_id}")
        remove_button.setToolTip("Remove this component")
        remove_button.setMinimumWidth(84)
        remove_button.setMaximumWidth(96)
        remove_button.clicked.connect(
            lambda _checked=False, wid=widget_id: self._remove_particle_widget(wid)
        )
        header_layout.addWidget(title)
        header_layout.addWidget(type_group)
        header_layout.addStretch(1)
        header_layout.addWidget(remove_button)
        layout.addWidget(header)

        stack = CurrentPageHeightStackedWidget(
            container,
            fitting_view_model=self.fitting_view_model,
        )
        stack.setObjectName(f"fitParticleStackWidget_{widget_id}")
        stack.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        none_page = QWidget(stack)
        none_layout = QVBoxLayout(none_page)
        none_layout.setContentsMargins(4, 6, 4, 6)
        none_label = QLabel("No component selected.", none_page)
        none_label.setToolTip(COMPONENT_FORMULA_TOOLTIPS["None"])
        none_layout.addWidget(none_label)
        none_page.setMaximumHeight(38)
        stack.addWidget(none_page)
        for shape_name in COMPONENT_ORDER[1:]:
            stack.addWidget(self._create_particle_parameter_page(stack, widget_id, shape_name))
        layout.addWidget(stack, 0)
        container.setMinimumSize(420, 0)
        container.setMaximumWidth(16777215)
        container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._register_ui_children(container)
        setattr(self.ui, f"fitParticleShapeCombox_{widget_id}", combo)
        setattr(self.ui, f"fitParticleStackWidget_{widget_id}", stack)
        install_safe_wheel_behavior(container)
        QTimer.singleShot(0, lambda widget=container: self._sync_particle_widget_height(widget))

    def _create_particle_parameter_page(
        self, parent: QWidget, widget_id: int, shape_name: str
    ) -> QWidget:
        page = QWidget(parent)
        page.setObjectName(f"fitParticle{self._shape_object_token(shape_name)}Page_{widget_id}")
        grid = QGridLayout(page)
        grid.setContentsMargins(2, 2, 2, 2)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(6)
        header_labels = (QLabel("Parameter", page), QLabel("Value", page), QLabel("Step", page))
        for col, header_label in enumerate(header_labels):
            header_label.setStyleSheet("font-size: 11px; font-weight: 600; color: #64748b;")
            grid.addWidget(header_label, 0, col)
        for row, (param_key, suffix, label_text, default_value, decimals, step) in enumerate(
            COMPONENT_PARAMETER_SCHEMAS[shape_name], 1
        ):
            label = QLabel(label_text, page)
            label.setMinimumHeight(24)
            value = QDoubleSpinBox(page)
            value.setObjectName(
                f"fitParticle{self._shape_object_token(shape_name)}{suffix}Value_{widget_id}"
            )
            value.setDecimals(decimals)
            value.setRange(-1e10, 1e10)
            value.setSingleStep(step)
            value.setValue(default_value)
            value.setMinimumHeight(26)
            value.setMaximumHeight(28)
            step_box = NoWheelDoubleSpinBox(page)
            step_box.setObjectName(
                f"fitParticle{self._shape_object_token(shape_name)}{suffix}Step_{widget_id}"
            )
            step_box.setDecimals(6)
            step_box.setRange(1e-9, 1e9)
            step_box.setSingleStep(step)
            step_box.setValue(step)
            step_box.setMinimumHeight(26)
            step_box.setMaximumHeight(28)
            step_box.setMaximumWidth(86)
            step_box.valueChanged.connect(
                lambda new_step, spin=value: spin.setSingleStep(float(new_step))
            )
            tooltip = COMPONENT_FORMULA_TOOLTIPS[shape_name]
            label.setToolTip(tooltip)
            value.setToolTip(tooltip)
            step_box.setToolTip(f"Single-step increment for {label_text}")
            grid.addWidget(label, row, 0)
            grid.addWidget(value, row, 1)
            grid.addWidget(step_box, row, 2)
        grid.setColumnStretch(1, 1)
        page.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        return page

    def _create_particle_widget(self, widget_id: int) -> QWidget:
        parent = getattr(
            self,
            "_particle_scroll_container",
            getattr(self.ui, "scrollAreaWidgetContents", self.ui),
        )
        container = QWidget(parent)
        container.setObjectName(f"fitParticleWidget_{widget_id}")
        self._rebuild_particle_widget_editor(container, widget_id)
        self._apply_particle_widget_style(container, widget_id)
        return container

    def _apply_particle_widget_style(self, widget: QWidget, widget_id: int):
        if widget is None:
            return
        widget.setStyleSheet(
            "QWidget {"
            "background-color: #ffffff;"
            "color: #172033;"
            "}"
            f"QWidget#{widget.objectName()} {{"
            "background-color: #ffffff;"
            "border: 1px solid #d6deea;"
            "border-radius: 12px;"
            "}"
            f"QWidget#fitParticleHeader_{widget_id} {{"
            "background-color: #f8fbff;"
            "border: 1px solid #e5edf6;"
            "border-radius: 10px;"
            "}"
            f"QWidget#fitParticleTypeGroup_{widget_id} {{"
            "background-color: #ffffff;"
            "border: 1px solid #dbe4f0;"
            "border-radius: 8px;"
            "}"
            f"QLabel#fitParticleTitleLabel_{widget_id} {{"
            "background-color: transparent;"
            "border: none;"
            "color: #1f2937;"
            "font-weight: 700;"
            "padding: 0 2px 0 0;"
            "}"
            f"QLabel#fitParticleTypeLabel_{widget_id} {{"
            "background-color: transparent;"
            "border: none;"
            "color: #526070;"
            "font-weight: 600;"
            "padding-left: 8px;"
            "padding-right: 2px;"
            "}"
            f"QComboBox#fitParticleShapeCombox_{widget_id} {{"
            "border: none;"
            "background-color: transparent;"
            "padding-left: 2px;"
            "padding-right: 24px;"
            "min-height: 28px;"
            "}"
            f"QComboBox#fitParticleShapeCombox_{widget_id}::drop-down {{"
            "border: none;"
            "background-color: transparent;"
            "width: 22px;"
            "subcontrol-origin: padding;"
            "subcontrol-position: top right;"
            "}"
            f"QComboBox#fitParticleShapeCombox_{widget_id}::down-arrow {{"
            "width: 10px;"
            "height: 10px;"
            "}"
            f"QPushButton#fitParticleRemoveButton_{widget_id} {{"
            "background-color: #f3f7fb;"
            "border: 1px solid #cfd9e6;"
            "border-radius: 8px;"
            "color: #334155;"
            "font-weight: 600;"
            "padding: 4px 10px;"
            "}"
            f"QPushButton#fitParticleRemoveButton_{widget_id}:hover {{"
            "background-color: #e8f0f8;"
            "border-color: #b8c7d9;"
            "}"
        )

    def _register_ui_children(self, widget: QWidget):
        if widget is None:
            return
        for child in widget.findChildren(QWidget):
            name = child.objectName()
            if name:
                setattr(self.ui, name, child)
        name = widget.objectName()
        if name:
            setattr(self.ui, name, widget)

    def _on_add_particle_clicked(self):
        try:
            widget_id = self._allocate_particle_id()
            particle_key = f"particle_{widget_id}"
            self.model_params_manager.ensure_particle_entry("fitting", particle_key, shape="Sphere")
            new_widget = self._create_particle_widget(widget_id)
            self._attach_particle_widget(new_widget, widget_id)
            self._add_fitting_success(f"Particle {widget_id} added")
        except Exception as e:
            self._add_fitting_error(f"Failed to add particle widget: {e}")

    def _allocate_particle_id(self) -> int:
        if self._recycled_particle_ids:
            self._recycled_particle_ids.sort()
            return self._recycled_particle_ids.pop(0)
        candidate = getattr(self, "_next_particle_candidate", 1)
        while candidate in self.particle_shape_configs:
            candidate += 1
        self._next_particle_candidate = candidate + 1
        return candidate

    def _attach_particle_widget(self, widget: QWidget, widget_id: int):
        if widget is None:
            return
        if self._particle_container_layout is not None and self._particle_add_button is not None:
            index = self._particle_container_layout.indexOf(self._particle_add_button)
            if index == -1:
                self._particle_container_layout.addWidget(widget)
            else:
                self._particle_container_layout.insertWidget(index, widget)
        elif self._particle_container_layout is not None:
            self._particle_container_layout.addWidget(widget)

        self._particle_widgets[widget_id] = widget
        self.particle_shape_configs[widget_id] = self._build_particle_config(widget_id)
        self._install_particle_context_menu(widget, widget_id)
        self._register_particle_show_checkbox(widget_id)

        self._setup_particle_connections([widget_id])
        self._setup_particle_parameter_connections([widget_id])
        self._setup_parameter_ranges([widget_id])
        self._initialize_particle_states([widget_id])
        self._schedule_model_parameters_region_refresh()

    def _sync_particle_widget_height(self, widget: QWidget):
        if widget is None:
            return
        for stack in widget.findChildren(CurrentPageHeightStackedWidget):
            stack.sync_current_height()
        layout = widget.layout()
        if layout is not None:
            layout.invalidate()
            layout.activate()
        widget.setMinimumHeight(0)
        widget.setMaximumHeight(16777215)
        widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        widget.updateGeometry()

    def _schedule_model_parameters_region_refresh(self):
        try:
            QTimer.singleShot(0, self._refresh_model_parameters_region_height)
        except Exception:
            self._refresh_model_parameters_region_height()

    def _refresh_model_parameters_region_height(self):
        model_card = (
            self.ui.gisaxsFittingPage.findChild(QWidget, "ModelParameterCard")
            if hasattr(self.ui, "gisaxsFittingPage")
            else None
        )
        fixed_controls_stack = (
            self.ui.gisaxsFittingPage.findChild(QWidget, "gisaxsFixedControlsStack")
            if hasattr(self.ui, "gisaxsFittingPage")
            else None
        )
        work_area_contents = (
            self.ui.gisaxsFittingPage.findChild(QWidget, "gisaxsWorkAreaContents")
            if hasattr(self.ui, "gisaxsFittingPage")
            else None
        )
        particle_container = getattr(self, "_particle_scroll_container", None)

        if self._particle_container_layout is not None:
            self._particle_container_layout.activate()
        if particle_container is not None:
            particle_container.updateGeometry()
            particle_container.adjustSize()
            for widget in particle_container.findChildren(QWidget):
                if widget.objectName().startswith("fitParticleWidget_"):
                    self._sync_particle_widget_height(widget)

        if model_card is not None:
            model_card.layout().activate() if model_card.layout() is not None else None
            model_card.setMinimumHeight(0)
            model_card.setMaximumHeight(16777215)
            model_card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
            model_card.updateGeometry()

        if fixed_controls_stack is not None:
            fixed_controls_stack.updateGeometry()
        if work_area_contents is not None:
            work_area_contents.updateGeometry()

    def _install_particle_context_menu(self, widget: QWidget, widget_id: int):
        if widget is None:
            return
        widget.setProperty("particle_id", widget_id)
        widget.setContextMenuPolicy(Qt.CustomContextMenu)
        try:
            widget.customContextMenuRequested.connect(self._handle_particle_context_menu_request)
        except Exception:
            pass

    def _handle_particle_context_menu_request(self, pos: QPoint):
        widget = self.sender()
        if widget is None:
            return
        widget_id = widget.property("particle_id")
        if widget_id is None:
            return
        global_pos = widget.mapToGlobal(pos)
        self._show_particle_context_menu(int(widget_id), global_pos)

    def _show_particle_context_menu(self, widget_id: int, global_pos: QPoint):
        menu = QMenu(self.ui)
        remove_action = menu.addAction("Remove Particle")
        if len(self._iter_particle_widget_ids()) <= 1:
            remove_action.setEnabled(False)
        action = menu.exec_(global_pos)
        if action == remove_action:
            self._remove_particle_widget(widget_id)

    def _remove_particle_widget(self, widget_id: int):
        if widget_id not in self.particle_shape_configs:
            return
        if len(self._iter_particle_widget_ids()) <= 1:
            self._add_fitting_warning("At least one particle widget must remain")
            return

        widget = self._particle_widgets.pop(widget_id, None)
        if widget is not None:
            try:
                widget.customContextMenuRequested.disconnect(
                    self._handle_particle_context_menu_request
                )
            except Exception:
                pass
            if self._particle_container_layout is not None:
                self._particle_container_layout.removeWidget(widget)
            widget.setParent(None)
            widget.deleteLater()

        checkbox = self._particle_show_checkboxes.pop(widget_id, None)
        if checkbox is not None:
            checkbox.setParent(None)
            checkbox.deleteLater()
            cb_name = checkbox.objectName()
            if cb_name and hasattr(self.ui, cb_name):
                delattr(self.ui, cb_name)

        meta_ids = self._particle_parameter_meta_ids.pop(widget_id, [])
        for meta_id in meta_ids:
            try:
                self.param_trigger_manager.unregister_widget(meta_id)
            except Exception:
                pass

        self._cleanup_particle_ui_attributes(widget_id)
        self.particle_shape_configs.pop(widget_id, None)
        self._recycled_particle_ids.append(widget_id)
        self.model_params_manager.remove_particle("fitting", f"particle_{widget_id}")
        self.model_params_manager.save_parameters()
        self._last_active_particle_ids = [
            wid for wid in self._last_active_particle_ids if wid != widget_id
        ]

        try:
            self._update_GUI_image("fitting" if self._is_in_fitting_mode() else "normal")
        except Exception:
            pass
        self._schedule_model_parameters_region_refresh()
        self._add_fitting_success(f"Particle {widget_id} removed")

    def _cleanup_particle_ui_attributes(self, widget_id: int):
        names = [
            f"fitParticleWidget_{widget_id}",
            f"fitParticleShapeCombox_{widget_id}",
            f"fitParticleStackWidget_{widget_id}",
        ]
        for shape in COMPONENT_PARAMETER_SCHEMAS:
            mapping = self._get_parameter_widget_mapping(widget_id, shape)
            names.extend(mapping.values())
            for widget_name in mapping.values():
                if widget_name.endswith(f"_{widget_id}"):
                    names.append(widget_name.replace("Value_", "Step_"))
        for name in names:
            if hasattr(self.ui, name):
                try:
                    attr = getattr(self.ui, name)
                    if hasattr(attr, "deleteLater"):
                        attr.deleteLater()
                except Exception:
                    pass
                try:
                    delattr(self.ui, name)
                except Exception:
                    pass
