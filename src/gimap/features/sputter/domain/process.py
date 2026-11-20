"""The component separation and the process analysis of an in-situ result (pure Python and NumPy).

A newer analysis package splits the model image of every solution into its physical scattering terms (the
*components*: grains, large clusters, the film fluctuation or network, voids under the film, the substrate seen
through the metal, fluorescence, the beam-path background; on porous TiO₂ the clusters in the pores and on the cap
and the metal network in the pores). Each entry of ``main.json`` holds the share of every term in the model counts,
and the images of a window hold the share of every term per cell (``component_fractions``): a term's image is its
share times the model image. The terms add up to the model image.

From the solutions shown by default it also describes the growth process against the metal amount h (nm; it does
not depend on the X-ray wavelength): when the coverage reaches 0.5, when the film forms (half of the metal in the
continuous layer), the growth laws D = D₁ h^p and R = R₁ h^q before the film, the distance once the film has formed,
the coarsening after the deposition stopped, and the stage of every window (islands, coalescence, film, after the
deposition; on porous TiO₂ clusters, then film). It names the range of the same metal on the same substrate over the
scans the package was built from, and where this scan lies in it (``within``, ``below``, ``above``; only with at
least two scans in the range).

An older result has none of this: the components are empty and ``ClientResult.process`` is None.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

COMPONENTS = (("grains", "Grains and islands"), ("large_clusters", "Large clusters"),
              ("film", "Film fluctuation and network"), ("voids", "Voids under the film"),
              ("buried", "Buried clusters"), ("domains", "Metal-rich domains in the polymer"),
              ("base_roughness", "Rough top of the continuous layer"), ("clusters_in_pores", "Clusters in the pores"),
              ("clusters_on_cap", "Clusters on the cap layer"), ("pore_network", "Metal network in the pores"),
              ("substrate", "Substrate seen through the metal"), ("fluorescence", "Fluorescence"),
              ("background", "Beam-path background"))
"""(key, name) of every term the package can name, in its order."""
COMPONENT_NAMES = dict(COMPONENTS)
STAGES = (("islands", "Separate islands"), ("coalescence", "Coalescence"), ("film", "Continuous film"),
          ("after", "After the deposition"), ("tio2_clusters", "Clusters"), ("tio2_film", "Metal film"))
"""(key, name) of the stages of a scan."""
STAGE_NAMES = dict(STAGES)
DESCRIPTORS = (("h_cov50", "Metal amount at coverage 0.5", "nm"),
               ("h_base", "Metal amount at which the film forms", "nm"),
               ("D1", "Distance D at 1 nm of metal (D = D₁ h^p)", "nm"), ("p_D", "Growth exponent p of D", ""),
               ("R1", "Radius R at 1 nm of metal (R = R₁ h^q)", "nm"), ("p_R", "Growth exponent q of R", ""),
               ("D_film", "Distance D once the film has formed", "nm"),
               ("R_film", "Radius R once the film has formed", "nm"),
               ("h_stop", "Metal amount when the deposition stopped", "nm"),
               ("D_after_stop_per_decade", "Coarsening after the stop (change of ln D per decade of time)", ""),
               ("h_cap50", "Metal amount at half of the cap layer term", "nm"),
               ("h_film50", "Metal amount at half of the film term", "nm"),
               ("h_max", "Metal amount at the last window", "nm"),
               ("dev5_median", "Median fit deviation", ""),
               ("drops", "Steps back in R, D or coverage between windows", ""))
"""(key, name, unit) of the growth descriptors shown, in this order (a result shows those it has)."""
POSITIONS = ("within", "below", "above")


def _number(value) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


@dataclass(frozen=True)
class ProcessStage:
    stage: str
    t_start: float
    t_end: float
    h_start: Optional[float] = None
    h_end: Optional[float] = None

    @property
    def name(self) -> str:
        return STAGE_NAMES.get(self.stage, self.stage)


@dataclass(frozen=True)
class SystemRange:
    """The range of a descriptor over the scans of the same metal on the same substrate."""

    median: float
    low: float
    high: float
    scans: int = 1


@dataclass(frozen=True)
class ProcessAnalysis:
    source: str = ""
    """Which solutions the analysis was made on (the joint fit of the whole scan, or the smooth path)."""
    descriptors: dict = field(default_factory=dict)
    """Descriptor key -> value (a descriptor the scan has no value for is absent)."""
    stages: tuple = ()
    system: str = ""
    """The metal and substrate of the range (``Ag/Si``), "" when the package knows no scan of them."""
    system_scans: int = 0
    ranges: dict = field(default_factory=dict)
    """Descriptor key -> SystemRange."""
    positions: dict = field(default_factory=dict)
    """Descriptor key -> 'within', 'below' or 'above' the range."""

    def shown(self) -> list:
        """(key, name, unit, value) of the descriptors the scan has, in the order of ``DESCRIPTORS``."""
        return [(key, name, unit, self.descriptors[key]) for key, name, unit in DESCRIPTORS if key in self.descriptors]

    def stage_at(self, time_s: float) -> str:
        """The stage of the window at ``time_s`` ("" outside every stage)."""
        for stage in self.stages:
            if stage.t_start - 1e-9 <= time_s <= stage.t_end + 1e-9:
                return stage.stage
        return ""


def components_listed(listed) -> tuple:
    """(key, name) of the terms ``main.json`` lists (its own name when the package gives one)."""
    found = []
    for item in listed if isinstance(listed, list) else ():
        if isinstance(item, dict) and isinstance(item.get("key"), str) and item["key"]:
            found.append((item["key"], COMPONENT_NAMES.get(item["key"], str(item.get("label") or item["key"]))))
    return tuple(found)


def entry_components(record) -> dict:
    """Term key -> share of the model counts of one entry ({} when the entry has none)."""
    listed = record.get("components") if isinstance(record, dict) else None
    if not isinstance(listed, dict):
        return {}
    shares = {str(key): _number(value) for key, value in listed.items()}
    return {key: value for key, value in shares.items() if value is not None}


def process_analysis(record) -> Optional[ProcessAnalysis]:
    """The process analysis of ``main.json`` (None when it has none or the package could not make it)."""
    if not isinstance(record, dict) or not record.get("source"):
        return None
    listed = record.get("descriptors") if isinstance(record.get("descriptors"), dict) else {}
    descriptors = {key: _number(listed.get(key)) for key, _name, _unit in DESCRIPTORS}
    stages = []
    for item in record.get("stages") or ():
        if not isinstance(item, dict) or not isinstance(item.get("stage"), str):
            continue
        start, end = _number(item.get("t_start")), _number(item.get("t_end"))
        if start is None or end is None:
            continue
        stages.append(ProcessStage(item["stage"], start, end, _number(item.get("h_start")), _number(item.get("h_end"))))
    reference = record.get("reference") if isinstance(record.get("reference"), dict) else {}
    ranges = {}
    for key, _name, _unit in DESCRIPTORS:
        item = reference.get(key)
        if isinstance(item, dict):
            median, low, high = (_number(item.get(name)) for name in ("median", "min", "max"))
            scans = _number(item.get("n"))
            if None not in (median, low, high):
                ranges[key] = SystemRange(median, low, high, int(scans) if scans else 1)
    compared = record.get("compare") if isinstance(record.get("compare"), dict) else {}
    positions = {str(key): str(value) for key, value in compared.items() if value in POSITIONS}
    scans = _number(reference.get("n"))
    return ProcessAnalysis(str(record["source"]), {key: value for key, value in descriptors.items() if value is not None},
                           tuple(stages), str(reference.get("system") or ""), int(scans) if scans else 0, ranges,
                           positions)


__all__ = ["COMPONENTS", "COMPONENT_NAMES", "DESCRIPTORS", "POSITIONS", "STAGES", "STAGE_NAMES", "ProcessAnalysis",
           "ProcessStage", "SystemRange", "components_listed", "entry_components", "process_analysis"]
