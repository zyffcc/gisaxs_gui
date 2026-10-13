"""Python View of the Settings dialog: a category list and one page per category.

Every page scrolls, so a long page (the Assistant's) keeps its controls at their own size in a
small dialog instead of squeezing them into each other.
"""

from PyQt5 import QtCore, QtWidgets


def _page(stack, title_text, description):
    """A scrolling page in ``stack``; returns its content widget and layout."""
    scroll = QtWidgets.QScrollArea()
    scroll.setObjectName("settingsPageScroll")
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
    scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
    page = QtWidgets.QWidget()
    page.setObjectName("settingsPageContent")
    layout = QtWidgets.QVBoxLayout(page)
    layout.setContentsMargins(16, 12, 16, 12)
    layout.setSpacing(10)
    title = QtWidgets.QLabel(title_text, page)
    title.setProperty("gimapRole", "heading")
    layout.addWidget(title)
    if description:
        note = QtWidgets.QLabel(description, page)
        note.setProperty("gimapRole", "muted")
        note.setWordWrap(True)
        layout.addWidget(note)
    scroll.setWidget(page)
    stack.addWidget(scroll)
    return page, layout


def _hint(parent, text):
    label = QtWidgets.QLabel(text, parent)
    label.setProperty("gimapRole", "muted")
    label.setWordWrap(True)
    return label


class SettingsDialogView(object):
    CATEGORIES = ("Appearance", "Analyze", "Data")

    def setupUi(self, SettingsDialog):
        SettingsDialog.setObjectName("SettingsDialog")
        SettingsDialog.setWindowTitle("Settings")
        SettingsDialog.setMinimumSize(QtCore.QSize(640, 420))
        root = QtWidgets.QVBoxLayout(SettingsDialog)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)
        body = QtWidgets.QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(4)
        root.addLayout(body, 1)

        self.category_list = QtWidgets.QListWidget(SettingsDialog)
        self.category_list.setObjectName("settingsCategoryList")
        self.category_list.setFixedWidth(150)
        self.category_list.addItems(self.CATEGORIES)
        body.addWidget(self.category_list)
        self.pages = QtWidgets.QStackedWidget(SettingsDialog)
        self.pages.setObjectName("settingsPages")
        body.addWidget(self.pages, 1)

        # Appearance -------------------------------------------------------
        page, layout = _page(self.pages, "Appearance", "Changes apply immediately.")
        form = QtWidgets.QFormLayout()
        form.setHorizontalSpacing(16)
        form.setVerticalSpacing(10)
        theme_row = QtWidgets.QHBoxLayout()
        theme_row.setSpacing(16)
        self.light_radio = QtWidgets.QRadioButton("Light", page)
        self.dark_radio = QtWidgets.QRadioButton("Dark", page)
        theme_row.addWidget(self.light_radio)
        theme_row.addWidget(self.dark_radio)
        theme_row.addStretch(1)
        form.addRow("Theme", theme_row)
        font_row = QtWidgets.QHBoxLayout()
        font_row.setSpacing(8)
        self.font_size_spin = QtWidgets.QDoubleSpinBox(page)
        self.font_size_spin.setObjectName("fontSizeSpin")
        self.font_size_spin.setDecimals(1)
        self.font_size_spin.setSingleStep(0.5)
        self.font_size_spin.setSuffix(" pt")
        self.font_size_spin.setKeyboardTracking(False)
        self.font_reset_button = QtWidgets.QPushButton("Default", page)
        font_row.addWidget(self.font_size_spin)
        font_row.addWidget(self.font_reset_button)
        font_row.addStretch(1)
        form.addRow("Font size", font_row)
        self.language_combo = QtWidgets.QComboBox(page)
        self.language_combo.setObjectName("languageCombo")
        self.language_combo.setToolTip(
            "The language of the interface. Values, units and file names are never translated; "
            "a few texts written while you work stay in English."
        )
        self.language_combo.setMinimumWidth(160)
        language_row = QtWidgets.QHBoxLayout()
        language_row.setSpacing(8)
        language_row.addWidget(self.language_combo)
        language_row.addStretch(1)
        form.addRow("Language", language_row)
        layout.addLayout(form)
        layout.addWidget(_hint(page, (
            "The interface follows the display scaling of the operating system "
            "(for example 125 % or 150 % in Windows), so it looks the same on "
            "every monitor. Use the font size for larger or smaller text; "
            "Ctrl + / Ctrl − / Ctrl 0 do the same from anywhere."
        )))
        layout.addStretch(1)

        # Analyze ----------------------------------------------------------
        page, layout = _page(self.pages, "Analyze", "Defaults for opening detector frames.")
        self.header_center_check = QtWidgets.QCheckBox(
            "Use the beam centre written in the file header", page
        )
        self.header_center_check.setObjectName("headerBeamCenterCheck")
        layout.addWidget(self.header_center_check)
        layout.addWidget(_hint(page, (
            "Off (recommended): every new frame keeps the beam centre of the "
            "instrument profile — the one you calibrated — even if the CBF "
            "(Beam_xy) or NeXus header says otherwise; many beamlines never "
            "update the header. On: a centre found in the header replaces the "
            "profile centre for that file. Either way the centre can be changed "
            "in Analyze at any time."
        )))
        fit_form = QtWidgets.QFormLayout()
        fit_form.setHorizontalSpacing(16)
        self.fit_side_combo = QtWidgets.QComboBox(page)
        self.fit_side_combo.setObjectName("fitSideCombo")
        fit_form.addRow("Send horizontal cut to Fitting", self.fit_side_combo)
        layout.addLayout(fit_form)
        layout.addWidget(_hint(page, (
            "Which half of the horizontal cut Analyze ▸ Send to Fitting passes on. The default keeps "
            "both halves as |qy| in two colours, so an asymmetry stays visible."
        )))
        layout.addStretch(1)

        # Data -------------------------------------------------------------
        page, layout = _page(self.pages, "Data", "")
        folder_row = QtWidgets.QHBoxLayout()
        folder_row.setSpacing(8)  # as the other rows: the button does not touch the field
        self.data_folder_edit = QtWidgets.QLineEdit(page)
        self.data_folder_edit.setObjectName("dataFolderEdit")
        self.data_folder_edit.setReadOnly(True)
        self.open_folder_button = QtWidgets.QPushButton("Open", page)
        folder_row.addWidget(self.data_folder_edit, 1)
        folder_row.addWidget(self.open_folder_button)
        data_form = QtWidgets.QFormLayout()
        data_form.setHorizontalSpacing(16)
        data_form.addRow("User data folder", folder_row)
        layout.addLayout(data_form)
        layout.addWidget(_hint(page, (
            "settings.json (all settings and preferences), instrument_profiles.json, "
            "session.json and model_parameters.json live here, outside the "
            "program folder, so updates never overwrite them. Set the GIMAP_HOME "
            "environment variable to use another folder."
        )))
        self.migration_label = _hint(page, "")
        layout.addWidget(self.migration_label)
        reset_row = QtWidgets.QHBoxLayout()
        self.reset_button = QtWidgets.QPushButton("Reset All Settings…", page)
        self.reset_button.setObjectName("resetSettingsButton")
        self.reset_button.setProperty("gimapRole", "danger")
        reset_row.addWidget(self.reset_button)
        reset_row.addStretch(1)
        layout.addLayout(reset_row)
        layout.addStretch(1)

        buttons = QtWidgets.QHBoxLayout()
        buttons.setContentsMargins(0, 0, 0, 0)
        buttons.addStretch(1)
        self.close_button = QtWidgets.QPushButton("Close", SettingsDialog)
        self.close_button.setDefault(True)
        buttons.addWidget(self.close_button)
        root.addLayout(buttons)

        self.category_list.currentRowChanged.connect(self.pages.setCurrentIndex)
        self.category_list.setCurrentRow(0)

    def add_category(self, title_text, description=""):
        """Append a category page supplied by a feature; returns ``(page, layout)``."""
        page, layout = _page(self.pages, title_text, description)
        self.category_list.addItem(title_text)
        return page, layout
