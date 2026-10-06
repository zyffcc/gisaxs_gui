"""Form Setup behavior for Format Converter."""

from __future__ import annotations


from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QButtonGroup,
    QHeaderView,
)

from src.gimap.app.presentation.section_bindings import (
    bind_advanced_section,
    bind_parameter_section,
)

# English keys of the frame-mode combo items, in View order. Logic reads these keys, never the
# displayed (possibly translated) item text.
FRAME_MODE_KEYS = ("All", "Current frame", "Frame range", "Custom", "Every Nth frame")


class FormSetupMixin:
    """Own form setup presentation behavior."""

    def _bind_form(self) -> None:
        """Attach behavior and compatibility names to the Designer-owned form."""
        # Set by a close that waits for a preview read; a new show clears it (ConversionMixin).
        self._close_when_idle = False
        bind_parameter_section(
            self.format_input_section,
            self.formatInputTitle,
            self.formatInputDescription,
            self.formatInputContent,
            self.formatInputContentLayout,
        )
        bind_parameter_section(
            self.format_configure_section,
            self.formatConfigureTitle,
            self.formatConfigureDescription,
            self.formatConfigureContent,
            self.formatConfigureContentLayout,
        )
        bind_parameter_section(
            self.format_preview_panel,
            self.formatPreviewTitle,
            self.formatPreviewDescription,
            self.formatPreviewContent,
            self.formatPreviewContentLayout,
        )
        bind_parameter_section(
            self.format_output_section,
            self.formatOutputTitle,
            self.formatOutputDescription,
            self.formatOutputContent,
            self.formatOutputContentLayout,
        )
        bind_parameter_section(
            self.format_run_section,
            self.formatRunTitle,
            self.formatRunDescription,
            self.formatRunContent,
            self.formatRunContentLayout,
        )
        bind_advanced_section(
            self.frame_advanced_section,
            self.frameAdvancedToggle,
            self.frameAdvancedDescription,
            self.frameAdvancedContent,
            self.frameAdvancedContentLayout,
        )
        bind_advanced_section(
            self.format_output_advanced,
            self.formatOutputAdvancedToggle,
            self.formatOutputAdvancedDescription,
            self.formatOutputAdvancedContent,
            self.formatOutputAdvancedContentLayout,
        )

        self.step_labels = [
            self.step_input_label,
            self.step_configure_label,
            self.step_output_label,
        ]
        self.preview_captions = [
            self.first_preview_caption,
            self.middle_preview_caption,
            self.last_preview_caption,
        ]
        self.preview_labels = [
            self.first_preview_label,
            self.middle_preview_label,
            self.last_preview_label,
        ]
        self.current_button.setEnabled(bool(self.current_file))
        # ``converted`` next to the data; it follows the first input until the user sets another.
        self._auto_destination = self.view_model.default_destination()
        self.destination_edit.setText(self._auto_destination)
        self.selection_splitter.setSizes((650, 340))
        self.input_tree.header().setSectionResizeMode(0, QHeaderView.Stretch)
        for column in range(1, 5):
            self.input_tree.header().setSectionResizeMode(
                column,
                QHeaderView.ResizeToContents,
            )
        selection_header = self.selection_table.horizontalHeader()
        for column in range(self.selection_table.columnCount()):
            selection_header.setSectionResizeMode(column, QHeaderView.ResizeToContents)
        selection_header.setSectionResizeMode(1, QHeaderView.Stretch)
        self.selection_table.verticalHeader().hide()
        # Series files differ only at the end (_00005, _00033 ...): elide in the middle.
        self.selection_table.setTextElideMode(Qt.ElideMiddle)
        self.input_tree.setTextElideMode(Qt.ElideMiddle)
        for index, mode in enumerate(FRAME_MODE_KEYS[: self.frame_mode.count()]):
            self.frame_mode.setItemData(index, mode)

        self.format_group = QButtonGroup(self.output_format_group)
        self.format_buttons = {
            "TIFF": self.tiff_format_button,
            "CBF": self.cbf_format_button,
            "HDF5": self.hdf5_format_button,
            "NumPy": self.numpy_format_button,
        }
        for format_name, button in self.format_buttons.items():
            button.setProperty("format_name", format_name)
            self.format_group.addButton(button)
        remembered = self.format_buttons.get(self.view_model.state.output_format)
        if remembered is not None:  # the format of the last conversion
            remembered.setChecked(True)
        for index, mode in enumerate(("original", "float32", "scale_uint16", "clip_uint16")):
            self.data_mode.setItemData(index, mode)

        self.add_files_button.clicked.connect(self._choose_files)
        self.add_folder_button.clicked.connect(self._choose_folder)
        self.current_button.clicked.connect(lambda: self.add_paths([self.current_file]))
        self.input_tree.currentItemChanged.connect(self._input_current_changed)
        self.dataset_combo.currentTextChanged.connect(self._dataset_changed)
        self.select_all_button.clicked.connect(lambda: self._set_all_included(True))
        self.select_none_button.clicked.connect(lambda: self._set_all_included(False))
        self.remove_selected_button.clicked.connect(self._remove_selected)
        self.sort_button.clicked.connect(self._sort_sources)
        self.filter_edit.textChanged.connect(self._filter_sources)
        self.selection_table.itemChanged.connect(self._include_changed)
        self.selection_table.itemSelectionChanged.connect(self._selection_current_changed)
        self.frame_mode.currentIndexChanged.connect(self._update_frame_editor)
        self.apply_frames.clicked.connect(self._apply_frame_selection)
        self.destination_button.clicked.connect(self._choose_destination)
        self.naming_combo.currentTextChanged.connect(self._update_output_preview)
        self.format_group.buttonClicked.connect(self._update_output_preview)
        self.destination_edit.textChanged.connect(self._update_output_preview)
        self.container_check.toggled.connect(self._container_toggled)
        self.write_sidecar.toggled.connect(self._sidecar_toggled)
        self.data_mode.currentIndexChanged.connect(self._update_output_preview)
        self.preserve_values.toggled.connect(self._preserve_values_toggled)
        self.back_button.clicked.connect(self._back)
        self.next_button.clicked.connect(self._next)
        self.cancel_button.clicked.connect(self.close)

        self._update_frame_editor()
        self._update_step_header()
