"""Particle Registry for fitting presentation."""

from __future__ import annotations


from PyQt5.QtWidgets import (
    QVBoxLayout,
    QWidget,
    QCheckBox,
)


from ..binding_primitives import (
    COMPONENT_ORDER,
    COMPONENT_PARAMETER_SCHEMAS,
)


class ParticleRegistryMixin:
    """Own particle registry behavior."""

    def _setup_particle_shape_connector(self):
        """No description."""
        self._initialize_particle_ui_registry()
        if not getattr(self, "particle_shape_configs", None):
            self.particle_shape_configs = {}

        self.particle_control_types = {
            shape: [field[1] for field in schema]
            for shape, schema in COMPONENT_PARAMETER_SCHEMAS.items()
        }

        self._setup_particle_connections()

        self._setup_particle_parameter_connections()

        self._setup_global_parameter_connections()

        self._setup_parameter_ranges()

        self._initialize_particle_states()

        self._initialize_global_parameters()

        self._add_fitting_success("Particle Shape Connector initialized")

    def _iter_particle_widget_ids(self):
        """No description."""
        return (
            sorted(self.particle_shape_configs.keys())
            if getattr(self, "particle_shape_configs", None)
            else []
        )

    def _collect_active_particles(self):
        """No description."""
        active_shapes = []
        widget_order = []
        for widget_id in self._iter_particle_widget_ids():
            combo_name = f"fitParticleShapeCombox_{widget_id}"
            if not hasattr(self.ui, combo_name):
                continue
            combobox = getattr(self.ui, combo_name)
            current_text = combobox.currentText().strip() if combobox.currentText() else ""
            if current_text and current_text.lower() != "none":
                active_shapes.append(current_text.lower())
                widget_order.append(widget_id)
        return active_shapes, widget_order

    def _get_particle_sequence_flags(self):
        """No description."""
        flags = {}
        sequence = getattr(self, "_last_active_particle_ids", []) or []
        for idx, widget_id in enumerate(sequence, 1):
            checkbox_name = f"fitParticle{widget_id}ShowCheckBox"
            flags[idx] = self._get_checkbox_state(checkbox_name, False)
        return flags

    def _sequence_index_to_widget_id(self, seq_index: int):
        sequence = getattr(self, "_last_active_particle_ids", []) or []
        if 1 <= seq_index <= len(sequence):
            return sequence[seq_index - 1]
        return None

    def _initialize_particle_ui_registry(self):
        """No description."""
        try:
            if hasattr(self.ui, "scrollAreaWidgetContents"):
                self._particle_scroll_container = self.ui.scrollAreaWidgetContents
                self._particle_container_layout = self._particle_scroll_container.layout()
        except Exception:
            self._particle_scroll_container = None
            self._particle_container_layout = None

        add_button = None
        for name in ("addModelButton", "fitAddModelButton", "pushButton"):
            if hasattr(self.ui, name):
                add_button = getattr(self.ui, name)
                break
        if add_button:
            self._particle_add_button = add_button
            if not add_button.toolTip():
                add_button.setToolTip("Add particle model")
            if not getattr(add_button, "_particle_handler_connected", False):
                add_button.clicked.connect(self._on_add_particle_clicked)
                add_button._particle_handler_connected = True

        self._prepare_dynamic_show_checkbox_area()

        self.particle_shape_configs = {}
        idx = 1
        while hasattr(self.ui, f"fitParticleWidget_{idx}") and hasattr(
            self.ui, f"fitParticleShapeCombox_{idx}"
        ):
            self._register_existing_particle_widget(idx)
            idx += 1
        self._next_particle_candidate = idx
        self._schedule_model_parameters_region_refresh()

    def _prepare_dynamic_show_checkbox_area(self):
        if self._dynamic_show_layout is not None:
            return

        preferred_names = ("ParticlesNumWidget", "fitParticlesNumWidget")
        for name in preferred_names:
            host = getattr(self.ui, name, None)
            if isinstance(host, QWidget):
                layout = host.layout()
                if layout is None:
                    layout = QVBoxLayout(host)
                    layout.setContentsMargins(0, 0, 0, 0)
                    layout.setSpacing(4)
                self._dynamic_show_container = host
                self._dynamic_show_layout = layout
                self._particle_checkbox_host_name = name
                return

        host_widget = getattr(self.ui, "fitFittingShowWidget", None)
        if host_widget is None:
            return
        base_layout = host_widget.layout()
        if base_layout is None:
            from PyQt5.QtWidgets import QGridLayout

            base_layout = QGridLayout(host_widget)

        self._dynamic_show_container = QWidget(host_widget)
        layout = QVBoxLayout(self._dynamic_show_container)
        layout.setContentsMargins(0, 4, 0, 0)
        layout.setSpacing(4)
        self._dynamic_show_layout = layout
        self._particle_checkbox_host_name = "fitFittingShowWidget"

        if hasattr(base_layout, "addWidget"):
            if hasattr(base_layout, "rowCount"):
                row_index = base_layout.rowCount()
                try:
                    base_layout.addWidget(self._dynamic_show_container, row_index, 0, 1, 2)
                    return
                except Exception:
                    pass
            base_layout.addWidget(self._dynamic_show_container)

    def _register_existing_particle_widget(self, widget_id: int):
        widget = getattr(self.ui, f"fitParticleWidget_{widget_id}", None)
        if widget is None:
            return
        self._rebuild_particle_widget_editor(widget, widget_id)
        self._particle_widgets[widget_id] = widget
        if not self._particle_widget_style_template:
            self._particle_widget_style_template = widget.styleSheet()
            self._particle_widget_style_source_name = widget.objectName() or ""
        self._apply_particle_widget_style(widget, widget_id)
        self.particle_shape_configs[widget_id] = self._build_particle_config(widget_id)
        self._install_particle_context_menu(widget, widget_id)

        checkbox = getattr(self.ui, f"fitParticle{widget_id}ShowCheckBox", None)
        if checkbox is not None:
            self._register_particle_show_checkbox(widget_id, checkbox)

    def _build_particle_config(self, widget_id: int) -> dict:
        pages = {
            index: {"name": shape_name, "page_index": index}
            for index, shape_name in enumerate(COMPONENT_ORDER)
        }
        return {
            "combobox": f"fitParticleShapeCombox_{widget_id}",
            "stack_widget": f"fitParticleStackWidget_{widget_id}",
            "pages": pages,
        }

    def _register_particle_show_checkbox(self, widget_id: int, checkbox: QCheckBox = None):
        if checkbox is None:
            checkbox = self._create_particle_show_checkbox(widget_id)
        if checkbox is None:
            return None
        checkbox_name = checkbox.objectName() or f"fitParticle{widget_id}ShowCheckBox"
        checkbox.setObjectName(checkbox_name)
        checkbox.setProperty("particleCheckboxId", widget_id)
        if not hasattr(self.ui, checkbox_name):
            setattr(self.ui, checkbox_name, checkbox)
        if widget_id not in self._particle_show_checkboxes:
            checkbox.toggled.connect(self._on_component_checkbox_changed)
        checkbox.setText(checkbox.text() or f"Particle {widget_id}")
        self._particle_show_checkboxes[widget_id] = checkbox
        return checkbox

    def _create_particle_show_checkbox(self, widget_id: int):
        self._prepare_dynamic_show_checkbox_area()
        parent = self._dynamic_show_container or getattr(self.ui, "fitFittingShowWidget", None)
        if parent is None:
            return None
        checkbox = QCheckBox(f"Particle {widget_id}", parent)
        checkbox.setObjectName(f"fitParticle{widget_id}ShowCheckBox")
        checkbox.setProperty("particleCheckboxId", widget_id)
        if self._dynamic_show_layout is not None:
            self._insert_particle_checkbox_widget(checkbox, widget_id)
        elif hasattr(parent, "layout") and parent.layout() is not None:
            parent.layout().addWidget(checkbox)
        return checkbox

    def _insert_particle_checkbox_widget(self, checkbox: QCheckBox, widget_id: int):
        layout = self._dynamic_show_layout
        if layout is None:
            return
        can_insert = hasattr(layout, "insertWidget")
        inserted = False
        for pos in range(layout.count()):
            item = layout.itemAt(pos)
            if item is None:
                continue
            existing = item.widget()
            if existing is None:
                continue
            existing_id = existing.property("particleCheckboxId")
            if existing_id is None:
                continue
            if widget_id < existing_id and can_insert:
                layout.insertWidget(pos, checkbox)
                inserted = True
                break
        if not inserted:
            layout.addWidget(checkbox)

    def _shape_key(self, shape_name: str) -> str:
        return str(shape_name).strip().lower().replace("-", "_").replace(" ", "_")

    def _shape_object_token(self, shape_name: str) -> str:
        return "".join(part.capitalize() for part in self._shape_key(shape_name).split("_"))

    def _shape_display_name(self, shape_name: str) -> str:
        shape_key = self._shape_key(shape_name)
        for candidate in COMPONENT_ORDER:
            if self._shape_key(candidate) == shape_key:
                return candidate
        return str(shape_name)

    def _parameter_key_from_alias(self, shape_name: str, param_name: str) -> str:
        """Map fitting-template names (Int, sigma_R, h) to stored parameter keys."""
        alias = str(param_name)
        alias_map = {
            "Int": "intensity",
            "R": "radius",
            "sigma_R": "sigma_radius",
            "h": "height",
            "sigma_h": "sigma_height",
            "D": "diameter",
            "sigma_D": "sigma_diameter",
        }
        if alias in alias_map:
            return alias_map[alias]
        for schema_shape, schema in COMPONENT_PARAMETER_SCHEMAS.items():
            if self._shape_key(schema_shape) != self._shape_key(shape_name):
                continue
            for param_key, suffix, _label, _default, _decimals, _step in schema:
                if alias in (param_key, suffix):
                    return param_key
        return alias
