"""Use case: the reflections of the phases a person gives, where they fall on the q map and the detector, and which
reflection each ring of I(q) is.

A phase is a unit cell with its symmetry (the operations of a CIF or of its space group, or a lattice centring) and
optionally atom sites, plus how the crystallites lie: at random (``powder``), with the plane (hkl) parallel to the substrate
(``plane``) or with the crystal axis [uvw] along the surface normal (``axis``). Fibre spots are placed at
(±q_∥, q_z) on the q map, in the transmitted channel and, with the refraction shift, also in the reflected one
(``domain/crystal/fibre.py``); powder rings at |q| on I(q) and as arcs on the q map. The detector positions come
from the inverse of the q map (``shared/geometry/q_inverse.positions_of_q``), tilt and refraction of the map
included. Everything is in Å⁻¹.

A CIF phase whose operations could not be found (``domain/crystal/cif.py``: ``symmetry`` ``none``) gives positions
only: every hkl of the cell, intensity 1, no extinctions; its warnings stay with the phase, and rings assigned to it
are ``tentative`` in the assignment.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field, replace
from functools import cached_property
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from src.gimap.shared.geometry import DetectorGeometry
from src.gimap.shared.geometry.q_inverse import exit_sine_of_qz, positions_of_q

from ..domain.crystal import (
    REFLECTED,
    TRANSMITTED,
    AtomSite,
    Candidate,
    Group,
    Phase,
    UnitCell,
    assign_all,
    centring_operations,
    fibre_spots,
    observed_qz,
    parse_operation,
    parse_operations,
    powder_rings,
    reachable,
    read_cif,
    reflections,
)
from ..domain.crystal.cif import FROM_SYMBOL, NO_SPACE_GROUP, POSITIONS_ONLY
from ..domain.crystal.composition import phase_composition
from ..domain.crystal.space_groups import CELL_MISFIT, DISAGREE, NO_NUMBER, ORIGIN_TAKEN, UNKNOWN, UNREADABLE

SYMMETRY_TEMPLATES = (FROM_SYMBOL, ORIGIN_TAKEN, UNREADABLE, UNKNOWN, CELL_MISFIT, DISAGREE, NO_NUMBER)
"""The notes and warnings of a CIF phase about its symmetry, with values (shown translated by presentation)."""
SYMMETRY_TEXTS = (NO_SPACE_GROUP, POSITIONS_ONLY)
"""The same, without values."""
POWDER, PLANE, AXIS = "powder", "plane", "axis"
ORIENTATIONS = (POWDER, PLANE, AXIS)
PHASE_COLORS = ("#f97316", "#22d3ee", "#a3e635", "#f472b6", "#facc15", "#c084fc", "#38bdf8", "#fb7185")
"""Marker colours of the phases in order (data colours, readable on the light and dark image backgrounds)."""
DEFAULT_Q_MAX = 3.0
CIF_COMPOSITION = "Expanded sites per cell: {sites}. CIF sum formula: {formula}. Z = {z}."
ARC_POINTS = 181


@dataclass(frozen=True)
class PhaseSpec:
    name: str
    cell: tuple[float, float, float, float, float, float]
    centring: str = "P"
    """Lattice centring, used when ``operations`` is empty (a cell typed in)."""
    operations: tuple[str, ...] = ()
    atoms: tuple[AtomSite, ...] = ()
    space_group: str = ""
    orientation: str = PLANE
    indices: tuple[int, int, int] = (0, 0, 1)
    """(hkl) of the plane parallel to the substrate, or [uvw] of the axis along the normal."""
    color: str = PHASE_COLORS[0]
    visible: bool = True
    source: str = ""
    """The CIF the phase was read from ("" when typed in)."""
    source_sha256: str = ""
    warnings: tuple[str, ...] = ()
    symmetry: str = ""
    """Where the operations of a CIF phase come from: ``file``, ``symbol`` (its space group) or ``none`` (positions
    only); ``""`` for a cell typed in (its ``centring``)."""
    notes: tuple[str, ...] = ()
    """How the operations were derived (``symbol``)."""
    formula: str = ""
    """The sum formula of the CIF (the name is its common or mineral name or its data block)."""
    formula_sum: str = ""
    """Only _chemical_formula_sum, for deriving bulk composition independently of the atom sites."""
    formula_units_z: Optional[float] = None
    space_group_number: int = 0
    cif_cell: Optional[tuple[float, ...]] = None
    cell_errors: tuple[float | None, ...] = ()

    @property
    def reference_cell(self) -> UnitCell:
        """Preserve the original strain reference when a trial cell is edited to index displaced rings."""
        return UnitCell(*(self.cif_cell or self.cell))

    @property
    def positions_only(self) -> bool:
        return self.symmetry == "none"

    @cached_property
    def composition(self) -> dict:
        return phase_composition(self.phase())

    def composition_text(self) -> str:
        if not self.source:
            return ""
        counts = self.composition["site_atoms_per_cell"]
        sites = ", ".join(f"{el} {n:g}" for el, n in sorted(counts.items())) or "unavailable"
        z = f"{self.formula_units_z:g}" if self.formula_units_z is not None else "unspecified"
        return CIF_COMPOSITION.format(sites=sites, formula=self.formula_sum or "unspecified", z=z)

    def phase(self) -> Phase:
        if self.positions_only:  # no symmetry known: every hkl, no intensities from the atoms as listed
            return Phase(self.name, UnitCell(*self.cell), (parse_operation("x,y,z"),), (),
                         self.formula_sum, self.formula_units_z)
        operations = parse_operations(self.operations) if self.operations else centring_operations(self.centring)
        return Phase(self.name, UnitCell(*self.cell), operations, tuple(self.atoms), self.formula_sum, self.formula_units_z)

    def symmetry_record(self) -> str:
        if self.positions_only:
            return "none: positions only (every hkl, no intensities, no extinctions)"
        if self.symmetry == "symbol":
            return f"{len(self.operations)} operations generated from the space group {self.space_group}"
        if self.operations:
            return f"{len(self.operations)} operations from the CIF"
        return f"lattice centring {self.centring}"

    def orientation_text(self) -> str:
        h, k, l = self.indices
        if self.orientation == PLANE:
            return f"({h} {k} {l}) parallel to the substrate"
        if self.orientation == AXIS:
            return f"[{h} {k} {l}] along the surface normal"
        return "powder (random orientation)"

    def record(self) -> dict:
        return {
            "name": self.name, "formula": self.formula, "cell": list(self.cell), "space_group": self.space_group,
            **self.composition, "composition_text": self.composition_text(),
            "symmetry": self.symmetry_record(), "symmetry_notes": list(self.notes),
            "positions_only": self.positions_only, "atoms": len(self.atoms),
            "intensities": "none (positions only)" if self.positions_only else
            "|F|² from the atom sites" if self.atoms else "1 for every allowed reflection",
            "orientation": self.orientation, "indices": list(self.indices), "orientation_text": self.orientation_text(),
            "source": self.source, "source_sha256": self.source_sha256, "warnings": list(self.warnings),
            "space_group_number": self.space_group_number,
            "reference_cell": list(self.reference_cell.as_tuple()),
            "reference_cell_errors": list(self.cell_errors),
        }


@dataclass(frozen=True)
class IndexingSettings:
    q_max: float = DEFAULT_Q_MAX
    refraction: bool = False
    """Place the fibre spots where refraction in the film puts them (both channels when ``reflected``)."""
    critical_angle_deg: float = 0.0
    """α_c of the film for the shift (0: the critical angle of Analyze's refraction correction, or none)."""
    reflected: bool = False
    """Also the reflected channel (with ``refraction``)."""
    label_fraction: float = 0.05
    """Spots and rings of at least this fraction of the phase's strongest get an hkl label."""

    def record(self) -> dict:
        return {"q_max": self.q_max, "refraction": self.refraction, "critical_angle_deg": self.critical_angle_deg,
                "reflected_channel": self.reflected, "label_fraction": self.label_fraction}


@dataclass(frozen=True)
class Marker:
    """One spot of one channel on one side: q map and detector positions (NaN where not seen)."""

    group: int
    channel: str
    q_parallel: float
    qz: float
    x_px: float = math.nan
    y_px: float = math.nan
    seen: bool = True
    """Outside the missing wedge and above the horizon."""


@dataclass(frozen=True)
class PhaseOverlay:
    spec: PhaseSpec
    groups: tuple[Group, ...]
    """Fibre spots (``plane``/``axis``) or powder rings (``powder``)."""
    markers: tuple[Marker, ...] = ()
    arcs: tuple[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], ...] = ()
    """Powder rings: (q∥, q_z, x, y) along each ring above the horizon."""
    rings: tuple[Group, ...] = ()
    """Powder rings of the phase (also for a fibre phase: for I(q) and the assignment)."""
    with_atoms: bool = False
    extinct_by_atoms: int = 0
    approximated: tuple[str, ...] = ()
    qz_shift: dict = field(default_factory=dict)
    """Channel → q_z on the map of every group (fibre phases), for the table."""

    @property
    def fibre(self) -> bool:
        return self.spec.orientation != POWDER

    def strongest(self) -> float:
        return max((group.intensity for group in self.groups), default=0.0)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def phase_from_cif(path, *, color: str = PHASE_COLORS[0], orientation: str = POWDER,
                   indices: tuple[int, int, int] = (0, 0, 1)) -> PhaseSpec:
    """A phase read from a CIF (cell, symmetry operations, atom sites)."""
    path = Path(path)
    structure = read_cif(path)
    spec = PhaseSpec(
        structure.name or path.stem, structure.cell.as_tuple(), "P", structure.operations, structure.atoms,
        structure.space_group, orientation, tuple(indices), color, True, str(path), _sha256(path), structure.warnings,
        structure.symmetry, structure.notes, structure.formula, structure.formula_sum, structure.formula_units_z,
        structure.space_group_number, structure.cell.as_tuple(), cell_errors=structure.cell_errors,
    )
    warning = spec.composition["composition_warning"]
    return replace(spec, warnings=(*spec.warnings, warning)) if warning else spec


def _normal(spec: PhaseSpec, cell: UnitCell) -> np.ndarray:
    if spec.orientation == AXIS:
        return cell.axis_direction(spec.indices)
    return cell.plane_normal(spec.indices)


def map_critical_angle(geometry: Optional[DetectorGeometry]) -> float:
    return float(geometry.critical_angle_deg) if geometry is not None else 0.0


def film_critical_angle(settings: IndexingSettings, geometry: Optional[DetectorGeometry]) -> float:
    """α_c of the shift: the one given, else that of Analyze's refraction correction."""
    if settings.critical_angle_deg > 0:
        return float(settings.critical_angle_deg)
    return map_critical_angle(geometry)


def channels_of(settings: IndexingSettings) -> tuple[str, ...]:
    return (TRANSMITTED, REFLECTED) if settings.refraction and settings.reflected else (TRANSMITTED,)


def q_y_of(geometry: DetectorGeometry, q_parallel, qz) -> np.ndarray:
    """q_y of the ray with in-plane magnitude |q_∥| (sign of q_∥) and map q_z: q_∥² = k²(cos²α_f + cos²α_i −
    2 cos α_f cos α_i cos 2θ_f), q_y = k cos α_f sin 2θ_f (NaN in the missing wedge)."""
    k = geometry.wavevector_inv_angstrom
    sine = exit_sine_of_qz(geometry, qz)
    cos_f = np.sqrt(1.0 - sine * sine)
    cos_i = math.cos(geometry.incidence_rad)
    q_parallel = np.asarray(q_parallel, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos_2theta = (cos_f ** 2 + cos_i ** 2 - (q_parallel / k) ** 2) / (2.0 * cos_f * cos_i)
        sin_2theta = np.sqrt(1.0 - cos_2theta ** 2)
    return np.where(np.abs(cos_2theta) <= 1.0, np.sign(q_parallel) * k * cos_f * sin_2theta, np.nan)


def detector_positions(geometry: DetectorGeometry, q_parallel, qz) -> tuple[np.ndarray, np.ndarray]:
    """Continuous detector coordinates (canonical pixels) of map points (q_∥ signed, q_z)."""
    return positions_of_q(geometry, q_y_of(geometry, q_parallel, qz), qz)


def _fibre_markers(spots, settings: IndexingSettings, geometry: Optional[DetectorGeometry]):
    """The markers of every spot, channel and side, and the map q_z of each spot per channel."""
    markers, shift = [], {}
    q_parallel = np.array([spot.q_parallel for spot in spots])
    qz = np.array([spot.qz for spot in spots])
    for channel in channels_of(settings):
        if settings.refraction and geometry is not None:
            shown = observed_qz(qz, geometry.wavevector_inv_angstrom, geometry.incidence_deg,
                                film_critical_angle(settings, geometry), map_critical_angle(geometry), channel)
        else:  # no shift: the true q_z (on a kinematic map, or one corrected with Analyze's αc)
            shown = qz
        shift[channel] = shown
        seen = np.isfinite(shown)
        if geometry is not None:
            sine = exit_sine_of_qz(geometry, shown)
            seen = seen & reachable(q_parallel, sine, geometry.wavevector_inv_angstrom, geometry.incidence_deg)
        for sign in (1.0, -1.0):
            signed = sign * q_parallel
            if geometry is not None:
                x, y = detector_positions(geometry, signed, shown)
            else:
                x = y = np.full(len(spots), np.nan)
            for index in range(len(spots)):
                if sign < 0 and q_parallel[index] < 1e-9:
                    continue  # a spot on the rod q_∥ = 0 has one position
                markers.append(Marker(index, channel, float(signed[index]), float(shown[index]),
                                      float(x[index]), float(y[index]), bool(seen[index])))
    return tuple(markers), shift


def _arcs(rings, geometry: Optional[DetectorGeometry]):
    arcs = []
    chi = np.radians(np.linspace(-90.0, 90.0, ARC_POINTS))
    for ring in rings:
        q_parallel, qz = ring.q * np.sin(chi), ring.q * np.cos(chi)
        if geometry is not None:
            x, y = detector_positions(geometry, q_parallel, qz)
        else:
            x = y = np.full(len(chi), np.nan)
        arcs.append((q_parallel, qz, x, y))
    return tuple(arcs)


def phase_overlay(spec: PhaseSpec, settings: IndexingSettings, geometry: Optional[DetectorGeometry]) -> PhaseOverlay:
    """The reflections of one phase and where they fall (see the module docstring)."""
    phase = spec.phase()
    found = reflections(phase, float(settings.q_max))
    rings = tuple(powder_rings(found, phase.operations))
    common = dict(rings=rings, with_atoms=found.with_atoms, extinct_by_atoms=found.extinct_by_atoms,
                  approximated=found.approximated)
    if spec.orientation == POWDER:
        return PhaseOverlay(spec, rings, arcs=_arcs(rings, geometry), **common)
    spots = tuple(fibre_spots(found, _normal(spec, phase.cell), phase.operations))
    markers, shift = _fibre_markers(spots, settings, geometry)
    return PhaseOverlay(spec, spots, markers, qz_shift=shift, **common)


def overlays(specs: Sequence[PhaseSpec], settings: IndexingSettings,
             geometry: Optional[DetectorGeometry]) -> tuple[PhaseOverlay, ...]:
    return tuple(phase_overlay(spec, settings, geometry) for spec in specs)


def table_rows(overlay: PhaseOverlay) -> list[dict]:
    """One row per spot or ring: hkl, d, |q|, q_∥, q_z, χ, multiplicity, mean |F|², intensity relative to the
    strongest of the phase (100), and the q_z of each channel on the map."""
    strongest = overlay.strongest() or 1.0
    rows = []
    for index, group in enumerate(overlay.groups):
        row = {
            "phase": overlay.spec.name, "hkl": group.label, "d_A": group.d, "q": group.q,
            "q_parallel": group.q_parallel if overlay.fibre else None, "qz": group.qz if overlay.fibre else None,
            "chi_deg": group.chi_deg if overlay.fibre else None, "multiplicity": group.multiplicity,
            "f2": group.mean_f2, "intensity": 100.0 * group.intensity / strongest,
        }
        for channel, values in overlay.qz_shift.items():
            row[f"qz_{channel}"] = float(values[index]) if np.isfinite(values[index]) else None
        rows.append(row)
    return rows


def candidates(overlays_: Sequence[PhaseOverlay]) -> list[Candidate]:
    """The powder ring positions of every visible phase (|q| does not depend on the orientation)."""
    found = []
    for overlay in overlays_:
        if not overlay.spec.visible:
            continue
        for ring in overlay.rings:
            found.append(Candidate(overlay.spec.name, ring.label, ring.q, ring.intensity, ring.multiplicity,
                                   ring.labels, ring.families))
    return found


def _assignment_axis(geometry: Optional[DetectorGeometry], transmission: bool) -> dict:
    """Trace the measured peak centres to the map that supplied I(q), independently of overlay settings."""
    if geometry is None:
        return {"kind": "unspecified"}
    if transmission:
        return {"kind": "transmission"}
    return {
        "kind": "refracted_transmitted" if geometry.refraction else "kinematic",
        "incidence_deg": geometry.incidence_deg,
        "sample_roll_deg": geometry.sample_roll_deg,
        "critical_angle_deg": geometry.critical_angle_deg if geometry.refraction else None,
        "k_sin_alpha_i_inv_A": geometry.wavevector_inv_angstrom * math.sin(geometry.incidence_rad),
    }


def assign_peaks(peaks: Sequence[dict], specs: Sequence[PhaseSpec], q_max: Optional[float] = None, *,
                 geometry: Optional[DetectorGeometry] = None, transmission: bool = False, q_offsets=None) -> dict:
    """``{"phases", "rule", "assignments"}``: each peak ``{"q", "q_err", "fwhm"}`` with its nearest reflection."""
    visible = [spec for spec in specs if spec.visible]
    if not visible:
        return {"phases": [], "rule": "", "assignments": []}
    from ..domain.crystal.assignment import tolerance

    largest = max((float(peak.get("q") or 0.) + tolerance(peak.get("q_err"), peak.get("fwhm"))
                   for peak in peaks), default=0.)
    window_limit = max((max(peak["q_window"]) for peak in peaks if "q_window" in peak), default=0.0)
    limit = max(float(q_max or 0.0), largest + 0.05, window_limit, 0.5)
    found = []
    for spec in visible:
        phase = spec.phase()
        offset = (q_offsets or {}).get(spec.name, 0.)
        rings = powder_rings(reflections(phase, max(.05, limit - offset)), phase.operations)
        found.extend(Candidate(spec.name, r.label, r.q + offset, r.intensity, r.multiplicity,
                               r.labels, r.families) for r in rings if r.q + offset > 0)
    observed = [peak for peak in peaks if peak.get("q")]
    triples = [(float(peak["q"]), float(peak.get("q_err") or math.nan), float(peak.get("fwhm") or math.nan))
               for peak in observed]
    loose = {spec.name for spec in visible if spec.positions_only}
    assignments = []
    for item, peak in zip(assign_all(triples, found), observed):
        record = item.record()
        for coordinate in ("q_parallel", "qz", "before_refraction", "sector", "chi_range", "tracked_phase", "tracked_hkl"):
            if coordinate in peak:
                record[coordinate] = peak[coordinate]
        record["tentative"] = bool(record["phase"] in loose)  # positions only: no extinctions, no intensities
        if "q_window" in peak:
            low, high = peak["q_window"]
            record["q_window"] = [low, high]
            record["overlapping_phases"] = sorted({line.phase for line in found
                                                   if low <= line.q <= high and line.phase != record["phase"]})
        assignments.append(record)
    return {
        "phases": [spec.record() for spec in visible],
        "rule": "nearest reflection within max(3 σ(q0), FWHM/2)",
        "q_axis": _assignment_axis(geometry, transmission),
        "assignments": assignments,
        "unassigned_q": [item["q"] for item in assignments if not item["assigned"]],
    }


def indexing_record(specs: Sequence[PhaseSpec], settings: IndexingSettings,
                    overlays_: Sequence[PhaseOverlay]) -> dict:
    """What the indexing used and found, for a record (full precision)."""
    return {
        "settings": settings.record(),
        "phases": [
            {**overlay.spec.record(), "reflections": table_rows(overlay), "with_atoms": overlay.with_atoms,
             "extinct_by_atoms": overlay.extinct_by_atoms, "approximated_f0": list(overlay.approximated)}
            for overlay in overlays_
        ],
        "conventions": {
            "q": "Å⁻¹; q_z along the surface normal, q_∥ the in-plane magnitude",
            "fibre": "crystallites at random about the surface normal; spots at (±q_∥, q_z), q_z ≥ 0",
            "channels": "T: k_z,f = q_z − k_z,i; R: k_z,f = q_z + k_z,i (Physics and Methods 7.10)",
        },
    }


def with_color(spec: PhaseSpec, index: int) -> PhaseSpec:
    return replace(spec, color=PHASE_COLORS[index % len(PHASE_COLORS)])


__all__ = [
    "AXIS", "DEFAULT_Q_MAX", "IndexingSettings", "Marker", "ORIENTATIONS", "PHASE_COLORS", "PLANE", "POWDER",
    "SYMMETRY_TEMPLATES", "SYMMETRY_TEXTS",
    "PhaseOverlay", "PhaseSpec", "assign_peaks", "candidates", "detector_positions", "film_critical_angle",
    "indexing_record", "overlays", "phase_from_cif", "phase_overlay", "q_y_of", "table_rows", "with_color",
]
