"""Option sections of Analyze (frames, corrections, GIWAXS cuts), placed in the process steps."""

from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from src.gimap.app.presentation.components import AdvancedSection, SegmentedControl

from ..frames_model import MAX_SUM

X_AXIS_ITEMS = (("q", "q"), ("2θ", "two_theta"))


def _spin(parent, low, high, value, decimals=3, step=0.01, suffix="") -> QDoubleSpinBox:
    spin = QDoubleSpinBox(parent)
    spin.setDecimals(decimals)
    spin.setRange(low, high)
    spin.setSingleStep(step)
    spin.setValue(value)
    spin.setKeyboardTracking(False)
    if suffix:
        spin.setSuffix(suffix)
    return spin


def _hint(text: str, parent) -> QLabel:
    label = QLabel(text, parent)
    label.setWordWrap(True)
    label.setProperty("gimapRole", "muted")
    return label


class OptionsPanelView:
    """Builds the option sections — frames, corrections, GIWAXS cuts — that the process steps show."""

    def build_option_sections(self, parent: QWidget) -> None:
        self._frames_section(parent)
        self._corrections_section(parent)
        self._intensity_section(parent)
        self._giwaxs_section(parent)

    def _frames_section(self, parent) -> AdvancedSection:
        section = AdvancedSection(
            "Frames",
            "Sum consecutive frames for better statistics: the frame shown plus the next "
            "ones (following files, or following frames of a NeXus file). Batch export and "
            "watching sum groups of this size.",
            parent,
            expanded=True,
        )
        self.frames_section = section
        form = QFormLayout()
        form.setHorizontalSpacing(8)
        self.sum_spin = QSpinBox(section)
        self.sum_spin.setObjectName("analyzeSumSpin")
        self.sum_spin.setRange(1, MAX_SUM)
        self.sum_spin.setSuffix(" frames")
        self.sum_spin.setSpecialValueText("1 frame (no sum)")
        self.sum_spin.setKeyboardTracking(False)
        form.addRow("Sum", self.sum_spin)
        section.add_layout(form)
        return section

    def _corrections_section(self, parent) -> AdvancedSection:
        section = AdvancedSection(
            "Corrections",
            "Applied to the frame before any cut, so curves, q map and exports all use them.",
            parent,
            expanded=True,
        )
        self.corrections_section = section
        background = QHBoxLayout()
        self.background_label = QLabel("No background", section)
        self.background_label.setProperty("gimapRole", "muted")
        self.background_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.background_button = QToolButton(section)
        self.background_button.setText("Choose…")
        self.background_button.setToolTip("Frame subtracted from every analysed frame")
        self.background_clear_button = QToolButton(section)
        self.background_clear_button.setText("×")
        self.background_clear_button.setAutoRaise(True)
        self.background_clear_button.setToolTip("Stop subtracting the background")
        background.addWidget(self.background_label, 1)
        background.addWidget(self.background_button)
        background.addWidget(self.background_clear_button)
        section.add_layout(background)
        form = QFormLayout()
        form.setHorizontalSpacing(8)
        self.background_scale_spin = _spin(section, -1e6, 1e6, 1.0, decimals=4, step=0.05)
        self.background_scale_spin.setToolTip("frame − scale × background (e.g. the exposure-time ratio)")
        self.background_frame_spin = QSpinBox(section)
        self.background_frame_spin.setRange(0, 100000)
        self.background_frame_spin.setToolTip("Frame of a NeXus background file")
        form.addRow("Scale", self.background_scale_spin)
        form.addRow("Frame", self.background_frame_spin)
        section.add_layout(form)

        limits = QGridLayout()
        limits.setHorizontalSpacing(6)
        self.minimum_check = QCheckBox("Min", section)
        self.minimum_spin = _spin(section, -1e12, 1e12, 0.0, decimals=2, step=1.0)
        self.maximum_check = QCheckBox("Max", section)
        self.maximum_spin = _spin(section, -1e12, 1e12, 1e6, decimals=2, step=100.0)
        limits.addWidget(self.minimum_check, 0, 0)
        limits.addWidget(self.minimum_spin, 0, 1)
        limits.addWidget(self.maximum_check, 1, 0)
        limits.addWidget(self.maximum_spin, 1, 1)
        section.add_widget(_hint("Valid raw intensity (pixels outside are ignored):", section))
        section.add_layout(limits)
        guard = QFormLayout()
        guard.setHorizontalSpacing(8)
        self.gap_guard_spin = QSpinBox(section)
        self.gap_guard_spin.setObjectName("analyzeGapGuardSpin")
        self.gap_guard_spin.setRange(0, 20)
        self.gap_guard_spin.setSuffix(" px")
        self.gap_guard_spin.setSpecialValueText("off")
        self.gap_guard_spin.setKeyboardTracking(False)
        self.gap_guard_spin.setToolTip(
            "Also ignore pixels this close to detector gaps and bad pixels, which often read "
            "wrong and show up as spikes in cuts. Remembered between sessions."
        )
        guard.addRow("Gap guard", self.gap_guard_spin)
        section.add_layout(guard)
        return section

    def _intensity_section(self, parent) -> AdvancedSection:
        section = AdvancedSection(
            "Intensity corrections (GIWAXS)",
            "Off by default. Peak positions do not change; intensities compared across χ or q do "
            "(orientation, pole figures, crystallinity ratios).",
            parent,
            expanded=False,
        )
        self.intensity_section = section
        self.solid_angle_check = QCheckBox("Solid angle of each pixel (cos³2θ)", section)
        self.solid_angle_check.setObjectName("analyzeSolidAngleCheck")
        self.solid_angle_check.setToolTip(
            "Pixels far from the beam see a smaller solid angle (flat detector): divide by cos³2θ, 1 at the beam"
        )
        section.add_widget(self.solid_angle_check)
        polarization = QHBoxLayout()
        self.polarization_check = QCheckBox("Polarisation, factor", section)
        self.polarization_check.setObjectName("analyzePolarizationCheck")
        self.polarization_spin = _spin(section, -1.0, 1.0, 0.99, decimals=3, step=0.01)
        self.polarization_spin.setObjectName("analyzePolarizationSpin")
        self.polarization_spin.setToolTip(
            "pyFAI's polarisation factor: 0.95–0.99 at a synchrotron (horizontal), 0 for an unpolarised "
            "laboratory source, −1 vertical"
        )
        polarization.addWidget(self.polarization_check)
        polarization.addWidget(self.polarization_spin)
        polarization.addStretch(1)
        section.add_layout(polarization)
        self.film_check = QCheckBox("Absorption in the film", section)
        self.film_check.setObjectName("analyzeFilmCheck")
        self.film_check.setToolTip(
            "The path of the beam in the film changes with the exit angle αf; relative to αf = αi. "
            "Refraction is not included: valid well above the critical angle."
        )
        section.add_widget(self.film_check)
        film = QFormLayout()
        film.setHorizontalSpacing(8)
        self.film_thickness_spin = _spin(section, 0.1, 1e6, 100.0, decimals=1, step=10.0, suffix=" nm")
        self.film_thickness_spin.setObjectName("analyzeFilmThickness")
        self.attenuation_spin = _spin(section, 0.001, 1e6, 100.0, decimals=3, step=1.0, suffix=" µm")
        self.attenuation_spin.setObjectName("analyzeAttenuationLength")
        self.attenuation_spin.setToolTip(
            "Attenuation length 1/µ of the film at this energy (tables such as CXRO or NIST)"
        )
        film.setRowWrapPolicy(QFormLayout.WrapLongRows)  # the panel is narrow: never wider than it
        for spin in (self.polarization_spin, self.film_thickness_spin, self.attenuation_spin):
            spin.setMaximumWidth(130)
        film.addRow("Thickness", self.film_thickness_spin)
        film.addRow("Attenuation length", self.attenuation_spin)
        section.add_layout(film)
        return section

    def _giwaxs_section(self, parent) -> AdvancedSection:
        section = AdvancedSection(
            "More cut settings",
            "χ = 0° is the surface normal, ±90° the sample plane.",
            parent,
            expanded=False,
        )
        self.giwaxs_section = section
        form = QFormLayout()
        form.setHorizontalSpacing(8)
        # The band widths are edited with the regions (views/regions_view.py places them there).
        self.in_plane_spin = _spin(parent, 0.5, 45.0, 10.0, decimals=1, step=1.0, suffix=" °")
        self.in_plane_spin.setToolTip("Half width of the in-plane sectors around χ = ±90°")
        self.out_of_plane_spin = _spin(parent, 0.5, 45.0, 10.0, decimals=1, step=1.0, suffix=" °")
        self.out_of_plane_spin.setToolTip("Half width of the out-of-plane sector around χ = 0°")
        self.bins_spin = QSpinBox(section)
        self.bins_spin.setRange(0, 20000)
        self.bins_spin.setSingleStep(50)
        self.bins_spin.setSpecialValueText("auto")
        self.bins_spin.setKeyboardTracking(False)
        self.x_axis_control = SegmentedControl(section)
        for text, value in X_AXIS_ITEMS:
            self.x_axis_control.addItem(text, value)
        form.addRow("Radial bins", self.bins_spin)
        form.addRow("I(q) axis", self.x_axis_control)
        section.add_layout(form)

        # The custom sector (set by the automation and the AI) is shown as a row of the regions;
        # its controls stay here, hidden, so both paths set the same widgets.
        self.sector_holder = QWidget(section)
        self.sector_holder.hide()
        holder = QVBoxLayout(self.sector_holder)
        self.sector_check = QCheckBox("Custom sector", self.sector_holder)
        self.sector_check.setToolTip("I(q) over a χ range, and I(χ) over its q range")
        holder.addWidget(self.sector_check)
        sector = QGridLayout()
        sector.setHorizontalSpacing(6)
        self.sector_chi_min = _spin(section, -90.0, 90.0, -30.0, decimals=1, step=1.0, suffix=" °")
        self.sector_chi_max = _spin(section, -90.0, 90.0, 30.0, decimals=1, step=1.0, suffix=" °")
        self.sector_q_min = _spin(section, 0.0, 100.0, 0.0, decimals=3)
        self.sector_q_min.setSpecialValueText("any")
        self.sector_q_max = _spin(section, 0.0, 100.0, 0.0, decimals=3)
        self.sector_q_max.setSpecialValueText("any")
        sector.addWidget(QLabel("χ", section), 0, 0)
        sector.addWidget(self.sector_chi_min, 0, 1)
        sector.addWidget(self.sector_chi_max, 0, 2)
        sector.addWidget(QLabel("q", section), 1, 0)
        sector.addWidget(self.sector_q_min, 1, 1)
        sector.addWidget(self.sector_q_max, 1, 2)
        self.sector_grid = QWidget(self.sector_holder)
        self.sector_grid.setLayout(sector)
        holder.addWidget(self.sector_grid)

        self.box_check = QCheckBox("q box", section)
        self.box_check.setToolTip(
            "I(qz) and I(q∥) inside a q∥–qz rectangle; drag it on the q map to move it"
        )
        section.add_widget(self.box_check)
        box = QGridLayout()
        box.setHorizontalSpacing(6)
        self.box_par_min = _spin(section, -100.0, 100.0, 0.2, decimals=3)
        self.box_par_max = _spin(section, -100.0, 100.0, 0.6, decimals=3)
        self.box_qz_min = _spin(section, -100.0, 100.0, 0.0, decimals=3)
        self.box_qz_max = _spin(section, -100.0, 100.0, 0.1, decimals=3)
        box.addWidget(QLabel("q∥", section), 0, 0)
        box.addWidget(self.box_par_min, 0, 1)
        box.addWidget(self.box_par_max, 0, 2)
        box.addWidget(QLabel("qz", section), 1, 0)
        box.addWidget(self.box_qz_min, 1, 1)
        box.addWidget(self.box_qz_max, 1, 2)
        self.box_grid = QWidget(section)
        self.box_grid.setLayout(box)
        section.add_widget(self.box_grid)
        section.add_widget(_hint("Values in Å⁻¹.", section))
        return section


__all__ = ["OptionsPanelView", "X_AXIS_ITEMS"]
