"""The assistant's changes to Analyze as operations a person can preview, apply and undo.

Every settings tool the model runs (sector widths, a custom sector, a q box,
the frame, the mode, αi, the radial bins, the valid intensity range) is
recorded with the arguments that restore the state before it, so it can be
undone exactly.  ``propose_operations`` lets the model suggest changes without
making them: each suggestion is previewed (applied, captured, restored) and
waits in the panel for the person to apply it.  In the "preview first"
permission mode the changes the model made during a run are restored at the
end and offered the same way.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Optional

OPERATION_TOOLS = (
    "set_measurement_mode",
    "set_incidence_angle",
    "set_sector_widths",
    "set_radial_bins",
    "set_custom_sector",
    "set_cut_regions",
    "set_q_box",
    "set_frame",
    "set_valid_intensity_range",
    "set_beam_center",
    "set_gisaxs_cuts",
    "set_halves",
)
PROPOSED = "proposed"
APPLIED = "applied"
DISMISSED = "dismissed"
UNDONE = "undone"
FAILED = "failed"
SUPERSEDED = "superseded"
"""An exploration step of the run that a later change of the same setting replaced (not shown as a card)."""
FROM_RUN = "run"
"""The model ran it (recorded, undoable)."""
FROM_PROPOSAL = "proposal"
"""The model only suggested it (``propose_operations``)."""
_ids = itertools.count(1)


@dataclass
class Operation:
    tool: str
    arguments: dict
    title: str
    why: str = ""
    state: str = PROPOSED
    source: str = FROM_PROPOSAL
    inverse: Optional[dict] = None
    """Arguments of the same tool that restore the state before (``None``: cannot be undone)."""
    effect: str = ""
    """What the change did, in one line (the tool's summary)."""
    preview_png: Optional[bytes] = None
    before_state: object = None
    """The setting's effective state before the change (``setting_state``), to tell real changes from no-ops."""
    after_state: object = None
    no_effect: bool = False
    """Applied but changed nothing that shows (e.g. αi already had that value): never a card."""
    id: str = ""

    def __post_init__(self) -> None:
        if not self.id:
            self.id = f"op{next(_ids)}"

    def record(self) -> dict:
        """JSON-able description (without the picture)."""
        return {
            "id": self.id, "tool": self.tool, "arguments": dict(self.arguments), "title": self.title,
            "why": self.why, "state": self.state, "source": self.source, "undoable": self.inverse is not None,
            "effect": self.effect, "no_effect": self.no_effect,
        }


def inverse_arguments(tool: str, status: dict) -> Optional[dict]:
    """Arguments of ``tool`` that bring back the state ``status`` describes; ``None`` when unknown."""
    giwaxs = status.get("giwaxs") or {}
    gisaxs = status.get("gisaxs") or {}
    if tool == "set_beam_center":
        centre = (status.get("geometry") or {}).get("beam_center_px")
        return None if not centre else {"x_px": float(centre[0]), "y_px": float(centre[1])}
    if tool == "set_halves":
        return {"side": gisaxs["halves"]} if gisaxs.get("halves") else None
    if tool == "set_gisaxs_cuts":
        if not gisaxs:
            return None
        if gisaxs.get("horizontal_source") in ("yoneda", "horizon"):
            return {"automatic": True}
        (top, bottom), (left, right) = gisaxs["horizontal_rows"], gisaxs["vertical_columns"]
        return {
            "horizontal_row": 0.5 * (top + bottom), "horizontal_half_height": 0.5 * (bottom - top),
            "vertical_column": 0.5 * (left + right), "vertical_half_width": 0.5 * (right - left),
        }
    if tool == "set_measurement_mode":
        mode = status.get("mode")
        return {"mode": mode} if mode in ("auto", "gisaxs", "giwaxs") else None
    if tool == "set_incidence_angle":
        if "incidence_override" not in status:
            return None
        value = status.get("incidence_override")
        return {"degrees": None if value is None else float(value)}
    if tool == "set_frame":
        frame = status.get("frame")
        return None if frame is None else {"frame": int(frame), "sum": int(status.get("summed_frames") or 1)}
    if tool == "set_valid_intensity_range":
        corrections = status.get("corrections") or {}
        limits = corrections.get("valid_range")
        return None if limits is None else {"minimum": limits[0], "maximum": limits[1]}
    if not giwaxs:
        return None  # the sector settings exist only for a GIWAXS reduction
    if tool == "set_sector_widths":
        return {
            "in_plane_half_width_deg": float(giwaxs["in_plane_half_width_deg"]),
            "out_of_plane_half_width_deg": float(giwaxs["out_of_plane_half_width_deg"]),
        }
    if tool == "set_radial_bins":
        bins = giwaxs.get("radial_bins")
        return {"bins": 0 if bins in (None, "auto") else int(bins)}
    if tool == "set_cut_regions":
        return {"regions": list(giwaxs.get("regions") or [])}
    if tool == "set_custom_sector":
        sector = giwaxs.get("custom_sector")
        if sector is None:
            return {"enabled": False}
        low, high = sector["chi_deg"]
        q_low, q_high = sector.get("q") or (None, None)
        return {"enabled": True, "chi_min_deg": low, "chi_max_deg": high, "q_min": q_low, "q_max": q_high}
    if tool == "set_q_box":
        box = giwaxs.get("q_box")
        if box is None:
            return {"enabled": False}
        return {
            "enabled": True, "q_parallel_min": box["q_parallel"][0], "q_parallel_max": box["q_parallel"][1],
            "qz_min": box["qz"][0], "qz_max": box["qz"][1],
        }
    return None


def setting_state(tool: str, status: dict) -> object:
    """What ``tool`` changes, as it shows in ``status`` (effective values, comparable before and after)."""
    giwaxs = status.get("giwaxs") or {}
    geometry = status.get("geometry") or {}
    if tool == "set_measurement_mode":
        return status.get("measurement") or status.get("mode")
    if tool == "set_incidence_angle":
        return geometry.get("incidence_deg")
    if tool == "set_frame":
        return (status.get("frame"), status.get("summed_frames"))
    if tool == "set_valid_intensity_range":
        return tuple((status.get("corrections") or {}).get("valid_range") or ())
    if tool == "set_sector_widths":
        return (giwaxs.get("in_plane_half_width_deg"), giwaxs.get("out_of_plane_half_width_deg"))
    if tool == "set_radial_bins":
        return giwaxs.get("radial_bins")
    if tool == "set_custom_sector":
        return repr(giwaxs.get("custom_sector"))
    if tool == "set_cut_regions":
        return repr(giwaxs.get("regions"))
    if tool == "set_q_box":
        return repr(giwaxs.get("q_box"))
    gisaxs = status.get("gisaxs") or {}
    if tool == "set_beam_center":
        centre = geometry.get("beam_center_px")
        return None if not centre else tuple(round(float(value), 3) for value in centre)
    if tool == "set_halves":
        return gisaxs.get("halves")
    if tool == "set_gisaxs_cuts":
        return repr((gisaxs.get("horizontal_rows"), gisaxs.get("vertical_columns")))
    return None


def _number(value) -> str:
    return "auto" if value is None else f"{float(value):g}"


def describe(tool: str, arguments: dict, language: str = "English") -> str:
    """A title a person understands, in the report language."""
    chinese = language.startswith("中")
    a = arguments
    if tool == "set_beam_center":
        where = f"x = {_number(a.get('x_px'))}, y = {_number(a.get('y_px'))} px"
        return f"光束中心 {where}" if chinese else f"Beam centre {where}"
    if tool == "set_halves":
        names = {
            "mean": ("两半平均", "Mean of both halves"), "negative": ("只用 qy < 0 一半", "Only the qy < 0 half"),
            "positive": ("只用 qy > 0 一半", "Only the qy > 0 half"), "both_abs": ("两半都保留（|qy|）", "Both halves on |qy|"),
        }
        pair = names.get(str(a.get("side")), (str(a.get("side")), str(a.get("side"))))
        return pair[0] if chinese else pair[1]
    if tool == "set_gisaxs_cuts":
        if a.get("automatic"):
            return "自动切线（Yoneda、光束中心）" if chinese else "Automatic cuts (Yoneda, beam centre)"
        parts = []
        if a.get("horizontal_row") is not None:
            parts.append(("水平切线行 " if chinese else "horizontal cut at row ") + _number(a.get("horizontal_row")))
        if a.get("vertical_column") is not None:
            parts.append(("垂直切线列 " if chinese else "vertical cut at column ") + _number(a.get("vertical_column")))
        text = ", ".join(parts) or ("切线宽度" if chinese else "cut widths")
        return text[0].upper() + text[1:]
    if tool == "set_sector_widths":
        values = (_number(a.get("in_plane_half_width_deg")), _number(a.get("out_of_plane_half_width_deg")))
        return f"扇区半宽：面内 ±{values[0]}°，面外 ±{values[1]}°" if chinese else f"Sector half widths: in-plane ±{values[0]}°, out-of-plane ±{values[1]}°"
    if tool == "set_custom_sector":
        if not a.get("enabled"):
            return "移除自定义扇区" if chinese else "Remove the custom sector"
        span = f"{_number(a.get('chi_min_deg'))}…{_number(a.get('chi_max_deg'))}°"
        return f"添加自定义扇区 χ {span}" if chinese else f"Custom sector χ {span}"
    if tool == "set_cut_regions":
        names = [str(region.get("name") or "Region") for region in a.get("regions") or ()]
        if not names:
            return "移除所有切割区域" if chinese else "Remove the cut regions"
        return ("切割区域：" if chinese else "Cut regions: ") + ", ".join(names)
    if tool == "set_q_box":
        if not a.get("enabled"):
            return "移除 q 框" if chinese else "Remove the q box"
        box = (f"q∥ {_number(a.get('q_parallel_min'))}…{_number(a.get('q_parallel_max'))}, "
               f"qz {_number(a.get('qz_min'))}…{_number(a.get('qz_max'))} Å⁻¹")
        return f"q 框 {box}" if chinese else f"q box {box}"
    if tool == "set_frame":
        frame, count = int(a.get("frame") or 1), max(1, int(a.get("sum") or 1))
        if frame == -1 and count > 1:
            return f"最后 {count} 帧求和" if chinese else f"The last {count} frames summed"
        if frame < 0:
            where = f"倒数第 {-frame} 帧" if chinese else f"Frame {-frame} from the end"
        else:
            where = f"第 {frame} 帧" if chinese else f"Frame {frame}"
        if count > 1:
            return f"{where}起求和 {count} 帧" if chinese else f"{where}, {count} summed"
        return where
    if tool == "set_measurement_mode":
        return f"测量模式：{str(a.get('mode')).upper()}" if chinese else f"Measurement mode: {str(a.get('mode')).upper()}"
    if tool == "set_incidence_angle":
        value = a.get("degrees")
        if value is None:
            return "入射角回到仪器配置的值" if chinese else "Incidence angle from the instrument profile"
        return f"入射角 αi = {float(value):g}°" if chinese else f"Incidence angle αi = {float(value):g}°"
    if tool == "set_radial_bins":
        bins = int(a.get("bins") or 0)
        text = "auto" if bins <= 0 else str(bins)
        return f"径向 bin 数：{text}" if chinese else f"Radial bins: {text}"
    if tool == "set_valid_intensity_range":
        span = f"{_number(a.get('minimum'))}…{_number(a.get('maximum'))}"
        return f"有效强度范围 {span}" if chinese else f"Valid intensity range {span}"
    return tool


__all__ = [
    "APPLIED",
    "DISMISSED",
    "FAILED",
    "FROM_PROPOSAL",
    "FROM_RUN",
    "OPERATION_TOOLS",
    "Operation",
    "PROPOSED",
    "SUPERSEDED",
    "UNDONE",
    "describe",
    "inverse_arguments",
    "setting_state",
]
