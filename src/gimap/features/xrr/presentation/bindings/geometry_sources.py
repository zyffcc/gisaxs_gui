"""Where each geometry value of the XRR window came from, and keeping what the user typed.

Each value has a source: ``default`` (built in), ``calibration`` (the last applied Geometry
Calibration, read when the window opens), ``file`` (the metadata of the loaded first frame),
``image center`` (no beam centre in the file and none from a calibration) or ``typed`` (entered or
picked by the user). Loading a frame never replaces a typed value: a different value in the file is
reported instead. Display only; qz is computed from the values as they are.
"""

from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path

from src.gimap.app.presentation.i18n import language_changed, tr, trf
from src.gimap.app.presentation.theme import set_role

# key, spin attribute, metadata key, metadata → window scale, calibration attribute
GEOMETRY_FIELDS = (
    ("distance", "distance_spin", "distance_m", 1000.0, "distance_mm"),
    ("energy", "energy_spin", "energy_kev", 1.0, "energy_kev"),
    ("pixel_x", "pixel_x_spin", "pixel_size_x_m", 1e6, "pixel_size_x_um"),
    ("pixel_y", "pixel_y_spin", "pixel_size_y_m", 1e6, "pixel_size_y_um"),
    ("center_x", "center_x_spin", "beam_center_x_px", 1.0, "beam_center_x_px"),
    ("center_y", "center_y_spin", "beam_center_y_px", 1.0, "beam_center_y_px"),
)
SPIN_OF = {key: spin_name for key, spin_name, *_rest in GEOMETRY_FIELDS}
GEOMETRY_KEY_PROPERTY = "gimapGeometryKey"
# How the sources read beside a field (empty: typed values need no mark) and in the summary.
SOURCE_TAGS = {
    "default": "built-in default",
    "calibration": "(from last calibration)",
    "file": "from file",
    "image center": "image center",
    "typed": "",
}
SOURCE_ROLES = {"default": "warning", "image center": "warning"}
SOURCE_TIPS = {
    "default": "A built-in default: check it before extracting.",
    "file": "Read from the metadata of the first frame.",
    "image center": "The middle of the image: the file has no beam center. Pick the direct beam on the preview.",
}
# Pairs named once when both values share a source.
PAIRS = (("pixel_x", "pixel_y", "pixel size"), ("center_x", "center_y", "beam center"))
FIELD_NAMES = {
    "distance": "distance",
    "energy": "energy",
    "pixel_x": "pixel size X",
    "pixel_y": "pixel size Y",
    "center_x": "beam center X",
    "center_y": "beam center Y",
}
UNITS = {"distance": "mm", "energy": "keV", "pixel_x": "µm", "pixel_y": "µm", "center_x": "px", "center_y": "px"}
# Values of the loaded file that were not used, and the calibration values the file replaced.
NOTE_LINES = (
    ("typed", "The file says: {values}"),
    ("calibration", "The last calibration says: {values}"),
)
SUMMARY_LINES = (
    ("file", "From the file: {fields}"),
    ("calibration", "From the last calibration: {fields}"),
    ("image center", "Image center (the file has no beam center): {fields}"),
    ("default", "Defaults, check them: {fields}"),
    ("typed", "Kept as typed: {fields}"),
)


def _number(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _when(timestamp: str) -> str:
    try:
        return datetime.fromisoformat(timestamp).astimezone().strftime("%Y-%m-%d %H:%M")
    except (TypeError, ValueError):
        return timestamp


class GeometrySourcesMixin:
    """Own the provenance of the geometry fields."""

    def _init_geometry_sources(self) -> None:
        self._filling_geometry = False
        self._geometry_sources: dict[str, str] = {}
        self._base_geometry: dict[str, tuple[float, str]] = {}
        self._calibration = None
        self._calibration_elsewhere = None  # the calibrated frame shape when the series differs
        self._inspection_summary = None
        getter = getattr(self.view_model, "last_calibration", None)
        calibration = getter() if callable(getter) else None
        self._defaults: dict[str, float] = {}
        self._calibration_values: dict[str, float | None] = {}
        for key, spin_name, _meta, _scale, attribute in GEOMETRY_FIELDS:
            spin = getattr(self, spin_name)
            self._defaults[key] = spin.value()
            self._calibration_values[key] = (
                _number(getattr(calibration, attribute, None)) if calibration is not None else None
            )
            base = self._base_for(key)
            self._base_geometry[key] = base
            self._set_geometry_value(key, *base)
            spin.setProperty(GEOMETRY_KEY_PROPERTY, key)
            # A bound method, not a lambda holding the window: no reference cycle through Qt.
            spin.valueChanged.connect(self._geometry_spin_edited)
        if any(source == "calibration" for _value, source in self._base_geometry.values()):
            self._calibration = calibration
        self._refresh_geometry_texts()
        language_changed().connect(self._refresh_geometry_texts)

    def _calibration_fits(self, shape) -> bool:
        """Whether the last calibration was made on frames of this shape (unknown: assumed)."""
        calibrated = getattr(self._calibration, "image_shape", None)
        return shape is None or calibrated is None or tuple(calibrated) == tuple(int(v) for v in shape[:2])

    def _base_for(self, key: str, shape=None) -> tuple[float, str]:
        """The last calibration's value, else the built-in default. A calibration of frames of
        another shape (another detector) gives only the energy."""
        value = self._calibration_values.get(key)
        if value is not None and (key == "energy" or self._calibration_fits(shape)):
            return value, "calibration"
        return self._defaults[key], "default"

    def geometry_sources(self) -> dict[str, str]:
        """The source of each geometry value now (``default``, ``calibration``, ``file`` …)."""
        return dict(self._geometry_sources)

    def _set_geometry_value(self, key: str, value: float, source: str) -> None:
        spin = getattr(self, SPIN_OF[key])
        self._filling_geometry = True
        try:
            spin.setValue(float(value))
        finally:
            self._filling_geometry = False
        self._geometry_sources[key] = source
        self._show_geometry_source(key)

    def _geometry_spin_edited(self, _value) -> None:
        sender = self.sender()
        key = sender.property(GEOMETRY_KEY_PROPERTY) if sender is not None else None
        if key in SPIN_OF:
            self._geometry_value_edited(str(key))

    def _geometry_value_edited(self, key: str) -> None:
        if self._filling_geometry:
            return
        self._geometry_sources[key] = "typed"
        self._show_geometry_source(key)

    def _show_geometry_source(self, key: str) -> None:
        label = self.geometry_source_labels.get(key)
        if label is None:
            return
        source = self._geometry_sources.get(key, "default")
        label.setText(tr(SOURCE_TAGS.get(source, "")))
        label.setVisible(bool(SOURCE_TAGS.get(source)))
        label.setToolTip(self._source_tip(source))
        set_role(label, SOURCE_ROLES.get(source, "muted"))
        self._align_geometry_tags()

    def _align_geometry_tags(self) -> None:
        """While any value shows a tag, a hidden tag keeps its place: the value fields line up."""
        labels = list(self.geometry_source_labels.values())
        tagged = any(not label.isHidden() for label in labels)
        for label in labels:
            policy = label.sizePolicy()
            if policy.retainSizeWhenHidden() != tagged:
                policy.setRetainSizeWhenHidden(tagged)
                label.setSizePolicy(policy)

    def _source_tip(self, source: str) -> str:
        if source == "calibration" and self._calibration is not None:
            return trf(
                "From the last applied geometry calibration ({image}, {time}).",
                image=Path(self._calibration.source_image).name or "?",
                time=_when(self._calibration.timestamp),
            )
        return tr(SOURCE_TIPS.get(source, "")) if SOURCE_TIPS.get(source) else ""

    def _apply_metadata(self, metadata: dict, shape) -> dict[str, list[tuple[str, float]]]:
        """Fill the geometry from a loaded frame. Return the values that were not used, as
        ``(key, value)``: ``typed`` — values of the file a typed value kept out; ``calibration`` —
        values of the last calibration a different value of the file replaced."""
        image_center = {"center_x": (shape[1] - 1) / 2.0, "center_y": (shape[0] - 1) / 2.0}
        fits = self._calibration_fits(shape)
        self._calibration_elsewhere = None if fits else tuple(self._calibration.image_shape)
        notes: dict[str, list[tuple[str, float]]] = {"typed": [], "calibration": []}
        for key, spin_name, meta_key, scale, _attribute in GEOMETRY_FIELDS:
            base = self._base_geometry[key] = self._base_for(key, shape)
            raw = _number(metadata.get(meta_key))
            from_file = None if raw is None else raw * scale
            spin = getattr(self, spin_name)
            tolerance = 0.5 * 10.0 ** -spin.decimals()
            if self._geometry_sources.get(key) == "typed":
                if from_file is not None and abs(spin.value() - from_file) > tolerance:
                    notes["typed"].append((key, from_file))
                continue
            if from_file is not None:
                if base[1] == "calibration" and abs(base[0] - from_file) > tolerance:
                    notes["calibration"].append((key, base[0]))
                self._set_geometry_value(key, from_file, "file")
            elif key in image_center and base[1] == "default":
                self._set_geometry_value(key, image_center[key], "image center")
            else:
                self._set_geometry_value(key, *base)
        return notes

    def _values_text(self, values: list[tuple[str, float]]) -> str:
        """``distance 2500.000 mm, energy 9.50000 keV`` (as many decimals as the field shows)."""
        parts = []
        for key, value in values:
            decimals = getattr(self, SPIN_OF[key]).decimals()
            parts.append(f"{tr(FIELD_NAMES[key])} {value:.{decimals}f} {UNITS[key]}")
        return ", ".join(parts)

    def _source_lines(self, sources: dict[str, str], notes: dict[str, list[tuple[str, float]]]) -> list[str]:
        named: dict[str, list[str]] = {}
        joined = {
            first: (second, pair_name)
            for first, second, pair_name in PAIRS
            if sources.get(first) == sources.get(second)
        }
        skipped = {second for second, _name in joined.values()}
        for key, *_rest in GEOMETRY_FIELDS:  # in the order of the form
            if key in skipped:
                continue
            name = joined[key][1] if key in joined else FIELD_NAMES[key]
            named.setdefault(sources.get(key, "default"), []).append(name)
        lines = [
            trf(template, fields=", ".join(tr(name) for name in named[source]))
            for source, template in SUMMARY_LINES
            if named.get(source)
        ]
        lines += [
            trf(template, values=self._values_text(notes[kind]))
            for kind, template in NOTE_LINES
            if notes.get(kind)
        ]
        if self._calibration_elsewhere is not None:
            rows, columns = self._calibration_elsewhere
            lines.append(
                trf(
                    "The last calibration is of {width} × {height} frames, not of this series: "
                    "only its energy is used.",
                    width=columns,
                    height=rows,
                )
            )
        return lines

    def _show_inspection_summary(self, inspection, notes: dict[str, list[tuple[str, float]]]) -> None:
        shape = inspection.first_frame.data.shape
        # Only what the text needs (not the frame), as it was when the frame was read: composed
        # again after a switch of the language.
        self._inspection_summary = (
            {
                "count": inspection.frame_count,
                "name": inspection.first_ref.label,
                "width": shape[1],
                "height": shape[0],
            },
            dict(self._geometry_sources),
            {kind: list(values) for kind, values in notes.items()},
        )
        self._compose_inspection_summary()

    def _compose_inspection_summary(self) -> None:
        header_values, sources, notes = self._inspection_summary
        header = trf("{count} frame(s) · first: {name} · shape {width} × {height}", **header_values)
        self.series_summary.setText("\n".join([header, *self._source_lines(sources, notes)]))

    def _refresh_geometry_texts(self, *_args) -> None:
        for key, *_rest in GEOMETRY_FIELDS:
            self._show_geometry_source(key)
        if self._calibration is not None:
            self.geometry_note.setText(
                trf(
                    "Last calibration: {image} · {time}",
                    image=Path(self._calibration.source_image).name or "?",
                    time=_when(self._calibration.timestamp),
                )
            )
        self.geometry_note.setVisible(self._calibration is not None)
        if self._inspection_summary is not None:
            self._compose_inspection_summary()


__all__ = ["GEOMETRY_FIELDS", "GeometrySourcesMixin"]
