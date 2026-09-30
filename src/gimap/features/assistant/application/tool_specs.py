"""The tools the model may call, as API tool definitions (name, when to use it, schema).

The order is fixed so the tool list stays byte-identical between turns (prompt
caching).  ``kind`` says how a call is gated: ``read`` tools only look or
change the view, ``write`` tools write files or change the corrections (asked
first in the confirm mode), ``final`` ends the run.
"""

from __future__ import annotations

from dataclasses import dataclass

from .models import GOALS

READ = "read"
WRITE = "write"
FINAL = "final"

Q_CURVES = ["radial", "in_plane", "out_of_plane", "sector", "box_q"]
ALL_CURVES = Q_CURVES + ["azimuthal", "sector_chi", "box_qz", "box_qpar", "horizontal", "vertical"]
HALVES = ["mean", "negative", "positive", "both_abs"]
NUMBER = {"type": "number"}
NUMBER_OR_NULL = {"type": ["number", "null"]}


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    schema: dict
    kind: str = READ
    strict: bool = False

    def definition(self) -> dict:
        tool = {"name": self.name, "description": self.description, "input_schema": self.schema}
        if self.strict:
            tool["strict"] = True
        return tool


def _object(properties: dict, required: tuple[str, ...] = ()) -> dict:
    return {
        "type": "object",
        "properties": properties,
        "required": list(required),
        "additionalProperties": False,
    }


REPORT_ITEM = _object(
    {
        "item": {"type": "string", "enum": [*GOALS, "other"]},
        "status": {"type": "string", "enum": ["done", "partial", "not_available"]},
        "findings": {"type": "string"},
        "evidence": {"type": "string"},
        "reason": {"type": "string"},
    },
    ("item", "status", "findings", "evidence", "reason"),
)


def tool_specs(*, allow_images: bool) -> tuple[ToolSpec, ...]:
    specs = [
        ToolSpec(
            "get_status",
            "Current frame, geometry, measurement type (GISAXS/GIWAXS), reduction settings and "
            "the available curves with their q ranges. The setters already return this; call it "
            "only when you need the state without changing anything.",
            _object({}),
        ),
        ToolSpec(
            "run_standard_pipeline",
            "Optional baseline in one call: GIMaP's standard procedure on this frame — geometry "
            "(keeps the instrument profile, or finds, fits and checks a calibration nearby), the last "
            "frames of a series summed, then for GIWAXS peaks, in-/out-of-plane, ring orientation and "
            "size of the strongest reliable peaks (and of the ring the user asked about); for GISAXS "
            "(technique gisaxs, or when Analyze reduces the frame as GISAXS) the horizontal cut at the "
            "Yoneda band, the beam centre on the symmetry axis, the halves, the in-plane spacing and a "
            "physical fit of I(qy) when a fitter is available. Returns every decision "
            "with its reason, needs_attention (values only the notes or the user know, with the option "
            "that supplies them), the peak table and the rings. Its choices are defaults, not limits: "
            "revisit any of them with the other tools when the question needs it (another frame, "
            "another ring, custom sectors, a skipped peak). Arguments override its defaults; writes "
            "(use_geometry) follow the permission mode.",
            _object(
                {
                    "calibration": {"type": ["string", "null"]},
                    "standard": {"type": ["string", "null"], "enum": ["agbh", "lab6", "ceo2", "lab6_ceo2", None]},
                    "energy_kev": NUMBER_OR_NULL,
                    "incidence_deg": NUMBER_OR_NULL,
                    "pixel_size_um": NUMBER_OR_NULL,
                    "frame": {"type": ["integer", "null"]},
                    "sum_frames": {"type": ["integer", "null"]},
                    "rings": {"type": ["integer", "null"]},
                    "recalibrate": {"type": ["boolean", "null"]},
                    "technique": {"type": ["string", "null"], "enum": ["gisaxs", "giwaxs", None]},
                    "fit": {"type": ["boolean", "null"]},
                }
            ),
        ),
        ToolSpec(
            "propose_operations",
            "Suggest changes to the Analyze settings instead of making them. Each one is previewed "
            "(applied, captured, restored, in the order given) and shown to the user as a card with a "
            "picture, to apply or dismiss; nothing stays changed. Use it when a change is the user's "
            "call (a mask for a shadow, a custom sector over the bright region, other frames, αi) or "
            "when the permission is 'preview first'. Tools: set_sector_widths, set_custom_sector, "
            "set_q_box, set_frame, set_measurement_mode, set_incidence_angle, set_radial_bins, "
            "set_valid_intensity_range, with their usual arguments. Give each a short title and why, in "
            "the report language. The user knows the cards by their titles: mention them by title, "
            "never by id.",
            _object(
                {
                    "operations": {"type": "array", "items": _object(
                        {
                            "tool": {"type": "string", "enum": [
                                "set_sector_widths", "set_custom_sector", "set_cut_regions", "set_q_box", "set_frame",
                                "set_measurement_mode", "set_incidence_angle", "set_radial_bins",
                                "set_valid_intensity_range", "set_beam_center", "set_gisaxs_cuts", "set_halves",
                            ]},
                            "arguments": {"type": "object"},
                            "title": {"type": "string"},
                            "why": {"type": "string"},
                        },
                        ("tool", "arguments"),
                    )},
                },
                ("operations",),
            ),
        ),
        ToolSpec(
            "set_measurement_mode",
            "Switch the reduction between automatic detection, GISAXS and GIWAXS. Use 'giwaxs' "
            "when the status shows the frame was reduced as GISAXS but the task is a GIWAXS analysis.",
            _object({"mode": {"type": "string", "enum": ["auto", "gisaxs", "giwaxs"]}}, ("mode",)),
        ),
        ToolSpec(
            "set_frame",
            "Choose the frame of a series (NeXus scans and in-situ runs have many: status shows "
            "'frames'): frame is 1-based, negative counts from the end (-1 = the last frame); sum "
            "sums that many consecutive frames from it for better statistics (1 = no summing), never "
            "past the last frame. Final state of an in-situ run: frame -1 with sum 10 (the last ten); "
            "start: frame 1. Sum when single frames are noisy.",
            _object({"frame": {"type": "integer"}, "sum": {"type": ["integer", "null"]}}, ("frame",)),
        ),
        ToolSpec(
            "set_incidence_angle",
            "Override the incidence angle αi (degrees) used for q; null returns to the instrument "
            "profile. Change it only when the user gave αi or the status shows an implausible value.",
            _object({"degrees": NUMBER_OR_NULL}, ("degrees",)),
        ),
        ToolSpec(
            "refine_beam_center_symmetry",
            "GISAXS: find the left–right symmetry axis of the horizontal cut (qy = 0) and move the "
            "beam-centre column there for this session (the direct beam is usually behind the beam stop). "
            "Returns the shift and the asymmetry before and after; no change below 0.05 px.",
            _object({}, ()),
        ),
        ToolSpec(
            "set_beam_center",
            "Beam centre (canonical pixels, row 0 at the top) for this session; every curve is recomputed.",
            _object({"x_px": NUMBER, "y_px": NUMBER}, ("x_px", "y_px")),
        ),
        ToolSpec(
            "set_gisaxs_cuts",
            "GISAXS: move the horizontal cut (centre row and half height in rows) and/or the vertical cut "
            "(centre column and half width); omitted values stay. automatic=true returns to the Yoneda "
            "band and the beam-centre column.",
            _object({
                "automatic": {"type": "boolean"}, "horizontal_row": NUMBER, "horizontal_half_height": NUMBER,
                "vertical_column": NUMBER, "vertical_half_width": NUMBER,
            }, ()),
        ),
        ToolSpec(
            "choose_halves",
            "GISAXS: compare the two halves of the horizontal cut I(qy) (coverage of |qy|, reach, agreement) "
            "and recommend mean / one half / both, with the reason. Run refine_beam_center_symmetry first.",
            _object({}, ()),
        ),
        ToolSpec(
            "set_halves",
            "GISAXS: which halves of I(qy) the curve for fitting uses: mean (averaged where both exist, the "
            "longer half beyond), negative, positive, or both_abs (both kept on |qy|, two colours).",
            _object({"side": {"type": "string", "enum": HALVES}}, ("side",)),
        ),
        ToolSpec(
            "in_plane_spacing",
            "GISAXS: peaks of the curve for fitting (|qy|, the halves chosen or 'side') and the dominant "
            "in-plane distance D = 2π/q* of the strongest side maximum — or of a shoulder, marked as such "
            "(a hint, not a measurement). Side maxima closer to qy = 0 than min_points bins are ignored.",
            _object({
                "side": {"type": "string", "enum": ["mean", "negative", "positive"]},
                "min_points": {"type": "integer"}, "background_window": NUMBER,
            }, ()),
        ),
        ToolSpec(
            "fit_horizontal_cut",
            "GISAXS: a numerical physical fit of the horizontal cut (the halves chosen, or 'side'): spheres, "
            "vertical and random cylinders with size dispersity and an interparticle distance D, or the "
            "composition given (1 = sphere, 2 = random cylinder, 3 = vertical cylinder). Returns the distinct "
            "solutions, best first, with R, D (nm), χ² and warnings. Also starts from the distance "
            "in_plane_spacing found (or distance_nm). Takes 10–60 s.",
            _object({
                "side": {"type": "string", "enum": ["mean", "negative", "positive"]},
                "components": {"type": "array", "items": {"type": "integer", "enum": [1, 2, 3]}, "maxItems": 4},
                "distance_nm": NUMBER,
            }, ()),
        ),
        ToolSpec(
            "set_sector_widths",
            "Half widths (0.5–45°, default 10°) of the in-plane sector (|χ| near 90°) and the "
            "out-of-plane sector (|χ| near 0°) behind the 'in_plane' and 'out_of_plane' I(q) curves. "
            "Narrower sectors separate orientations better but are noisier.",
            _object(
                {"in_plane_half_width_deg": NUMBER, "out_of_plane_half_width_deg": NUMBER},
                ("in_plane_half_width_deg", "out_of_plane_half_width_deg"),
            ),
        ),
        ToolSpec(
            "set_radial_bins",
            "Number of q bins of the I(q) curves (0 = automatic). Raise it only when find_peaks "
            "flags peaks as resolution_limited by the binning.",
            _object({"bins": {"type": "integer"}}, ("bins",)),
        ),
        ToolSpec(
            "set_custom_sector",
            "Add a custom χ sector with an optional q range, giving the curves 'sector' (I(q)) and "
            "'sector_chi' (I(χ)); enabled=false removes it. χ = 0° is the surface normal.",
            _object(
                {
                    "enabled": {"type": "boolean"},
                    "chi_min_deg": NUMBER,
                    "chi_max_deg": NUMBER,
                    "q_min": NUMBER_OR_NULL,
                    "q_max": NUMBER_OR_NULL,
                },
                ("enabled",),
            ),
        ),
        ToolSpec(
            "set_cut_regions",
            "GIWAXS: replace the list of cut regions the person sees in the Cuts step (an empty list removes "
            "them). Each region is a q range × a χ range: q_min/q_max in Å⁻¹ (both null: all q), "
            "chi_min_deg/chi_max_deg with χ = 0° the surface normal; both_sides=true (the usual case) takes "
            "|χ| on both halves and folds I(χ). Region N gives the curves 'regionN' (I(q)) and 'regionN_chi' "
            "(I(χ)). Use it for a ring or a band the question is about; keep the regions already there unless "
            "they are wrong.",
            _object({"regions": {"type": "array", "maxItems": 12, "items": _object({
                "name": {"type": "string"}, "q_min": NUMBER_OR_NULL, "q_max": NUMBER_OR_NULL,
                "chi_min_deg": NUMBER, "chi_max_deg": NUMBER, "both_sides": {"type": "boolean"},
            }, ("chi_min_deg", "chi_max_deg"))}}, ("regions",)),
        ),
        ToolSpec(
            "set_q_box",
            "Integrate a rectangle of the q∥–qz map (a Bragg rod or spot), giving the curves "
            "'box_q', 'box_qz' and 'box_qpar'; enabled=false removes it.",
            _object(
                {
                    "enabled": {"type": "boolean"},
                    "q_parallel_min": NUMBER,
                    "q_parallel_max": NUMBER,
                    "qz_min": NUMBER,
                    "qz_max": NUMBER,
                },
                ("enabled",),
            ),
        ),
        ToolSpec(
            "get_curve",
            "Downsampled values of one curve (x in Å⁻¹, or χ in degrees for 'azimuthal' and "
            "'sector_chi'; y in counts/pixel with its standard error). Use it to inspect a feature "
            "the metric tools do not cover; take numbers for the report from the metric tools.",
            _object(
                {
                    "curve": {"type": "string", "enum": ALL_CURVES},
                    "max_points": {"type": "integer"},
                    "x_min": NUMBER_OR_NULL,
                    "x_max": NUMBER_OR_NULL,
                },
                ("curve",),
            ),
        ),
        ToolSpec(
            "find_peaks",
            "Detect and fit the peaks of an I(q) curve: q, d = 2π/q (Å), FWHM, height above the "
            "background, area, significance (snr) and flags — weak (3–5σ, may be noise), overlap, "
            "at_edge, resolution_limited, fit_failed, spike (hot pixels or a module edge, not "
            "diffraction), broad (a halo, not a crystalline peak) — plus q-ratio series hints from the "
            "reliable peaks. When nothing "
            "qualifies it says why. Run it on 'radial' first, then on 'in_plane'/'out_of_plane' "
            "when sector-specific peaks matter. Raise background_window (Å⁻¹, default 0.15) when a "
            "broad peak (e.g. π–π stacking) is absorbed into the background.",
            _object(
                {
                    "curve": {"type": "string", "enum": Q_CURVES},
                    "q_min": NUMBER_OR_NULL,
                    "q_max": NUMBER_OR_NULL,
                    "min_snr": NUMBER,
                    "background_window": NUMBER,
                },
                ("curve",),
            ),
        ),
        ToolSpec(
            "compare_sectors",
            "Orientation per peak: the mean net intensity per pixel in the in-plane and "
            "out-of-plane sectors around each peak, their ratio (out/in) and which dominates. "
            "Default peaks: the last find_peaks result on 'radial'. Use it for the orientation result.",
            _object(
                {"q_values": {"type": "array", "items": NUMBER}, "window_fwhm": NUMBER},
            ),
        ),
        ToolSpec(
            "ring_orientation",
            "Orientation distribution of one ring: sets the I(χ) window of Analyze to "
            "q_center ± q_half_width (default from the fitted FWHM), then reports the measured χ "
            "coverage (missing wedge), maxima, anisotropy and Herman's orientation factor relative "
            "to the surface normal (1 along the normal, 0 random, −0.5 in plane). The GUI shows the I(χ).",
            _object({"q_center": NUMBER, "q_half_width": NUMBER_OR_NULL}, ("q_center",)),
        ),
        ToolSpec(
            "crystallite_size",
            "Scherrer coherence length (Å and nm) of the fitted peak nearest q_center, from the last "
            "find_peaks result of that curve. Without instrumental_fwhm (Å⁻¹) the value is a lower "
            "bound; shape_factor defaults to 0.9.",
            _object(
                {
                    "q_center": NUMBER,
                    "curve": {"type": "string", "enum": Q_CURVES},
                    "shape_factor": NUMBER,
                    "instrumental_fwhm": NUMBER,
                },
                ("q_center",),
            ),
        ),
        ToolSpec(
            "show_view",
            "Change what the GUI shows to the person watching (detector image or q map; the lower "
            "plot). It does not change the analysis.",
            _object(
                {
                    "view": {"type": "string", "enum": ["detector", "q_map"]},
                    "lower_plot": {"type": "string", "enum": ["azimuthal", "sector_chi", "box_qz", "box_qpar"]},
                }
            ),
        ),
    ]
    specs += [
        ToolSpec(
            "find_calibration_files",
            "Look for calibration material when the frame has no geometry (or to check the one it "
            "has). Searches the frame's folder and subfolders, the parent and all sibling folders, "
            "two folders up with their subfolders, and calibration / log folders three up. Returns "
            "ready calibration results (pyFAI .poni, GIMaP calibration .json), images of calibration "
            "standards (standard guessed from the file or folder name: AgBH, LaB6, CeO2, Si …; "
            "gimap_can_fit = GIMaP can fit it) and log or parameter files that may give the energy, "
            "distance, beam centre or incidence angle — each with its folder, file time and hours "
            "before (−) or after (+) the frame — plus the files and folders the user named.",
            _object({"max_results": {"type": "integer"}}),
        ),
        ToolSpec(
            "search_files",
            "Search one folder and its subfolders: any folder above the frame (up to the drive), a "
            "folder the user named, or one found before. Without name_contains it lists calibration "
            "material and logs as find_calibration_files does; with name_contains (e.g. 'lab6', "
            "'calib', '.poni', '0012') it lists every image, calibration or text file whose name "
            "contains it. max_depth: how many folder levels down (default 4).",
            _object(
                {"folder": {"type": "string"}, "name_contains": {"type": ["string", "null"]}, "max_depth": {"type": ["integer", "null"]}},
                ("folder",),
            ),
        ),
        ToolSpec(
            "inspect_file",
            "Read a file: the open frame, one found by a search, one the user named, or any other "
            "detector image, .poni, .json or text/log file by its path (outside the frame's folders "
            "the user may have to allow it). For a detector image: size, detector, header values "
            "(energy, distance, beam centre, pixel size, time) and whether it matches this frame's "
            "detector; for a .poni or GIMaP calibration file the geometry in it; for a text or log "
            "file the lines that mention distance, energy/wavelength, beam centre, pixel size, "
            "incidence angle or a calibrant.",
            _object({"path": {"type": "string"}}, ("path",)),
        ),
        ToolSpec(
            "calibrate_geometry",
            "Fit the geometry (beam centre, distance) to an image of a calibration standard with "
            "GIMaP's calibration: agbh, lab6, ceo2 or lab6_ceo2 (a mixed calibrant) when the name or "
            "the user says which; 'compare' "
            "when it is not known — fits every standard and ranks them by where each standard's "
            "lines land in q (line_q_error_percent; within 0.2 % is good), with a verdict; "
            "'auto' is a quick guess that can pick the wrong standard. energy_kev must come from the "
            "frame header, a log or the user; an approximate distance_mm (header or log) and "
            "pixel_size_um (when the image header has none) help. Returns each fit's assessment "
            "(judged by the line check; the rms residual in px can be several pixels on a tilted "
            "wide-angle detector even for a good fit) and its calibration_index; nothing is "
            "saved until use_geometry.",
            _object(
                {
                    "path": {"type": "string"},
                    "standard": {"type": "string", "enum": ["agbh", "lab6", "ceo2", "lab6_ceo2", "compare", "auto"]},
                    "energy_kev": NUMBER,
                    "distance_mm": NUMBER_OR_NULL,
                    "pixel_size_um": NUMBER_OR_NULL,
                },
                ("path", "standard", "energy_kev"),
            ),
        ),
        ToolSpec(
            "ask_user",
            "Ask the person at the screen to choose between options — e.g. calibration files, each "
            "with its time and folder in 'detail' — or, with allow_text, to type a missing value such "
            "as the X-ray energy. Use it when the files do not show which option is right or a "
            "needed number is nowhere to be found. Returns the chosen option and/or typed text, or "
            "that the person declined.",
            _object(
                {
                    "question": {"type": "string"},
                    "options": {"type": "array", "items": _object(
                        {"label": {"type": "string"}, "detail": {"type": "string"}}, ("label",),
                    )},
                    "allow_text": {"type": "boolean"},
                },
                ("question", "options"),
            ),
        ),
    ]
    if allow_images:
        specs.append(ToolSpec(
            "view_preview",
            "A small image of the q∥–qz map (log intensity) to judge qualitatively: rings versus "
            "arcs versus spots, streaks, detector artifacts. Never read numbers off it.",
            _object({}),
        ))
    specs += [
        ToolSpec(
            "note_missing_capability",
            "Record an analysis or tool GIMaP lacks for this task so developers can add it. Use it "
            "instead of guessing when no tool can do what is needed; then report the item as not_available.",
            _object({"capability": {"type": "string"}, "reason": {"type": "string"}}, ("capability", "reason")),
        ),
        ToolSpec(
            "set_valid_intensity_range",
            "Exclude pixels outside an intensity range (hot pixels, saturation) from every curve "
            "for this session; null removes a limit. The limits are compared with the image the "
            "curves come from: the sum of the summed frames (scale a per-frame count by their number), "
            "after any background subtraction. Use only for clear detector artifacts; the user may "
            "have to approve it.",
            _object({"minimum": NUMBER_OR_NULL, "maximum": NUMBER_OR_NULL}, ("minimum", "maximum")),
            kind=WRITE,
        ),
        ToolSpec(
            "use_geometry",
            "Use a geometry for this frame: saves it as the instrument profile of this detector and "
            "frame size (used for every such frame) and re-analyses. source 'calibration' takes a "
            "calibrate_geometry result (calibration_index, default the last), 'file' an inspected "
            ".poni or GIMaP calibration file (path), 'values' numbers read from a file or given by the "
            "user (distance_mm, beam_center_x_px, beam_center_y_px in pixels with the first pixel's "
            "centre at 0.5, energy_kev or wavelength_A; note says where they come from). "
            "incidence_deg sets αi when known. The user may have to approve it.",
            _object(
                {
                    "source": {"type": "string", "enum": ["calibration", "file", "values"]},
                    "calibration_index": {"type": ["integer", "null"]},
                    "path": {"type": ["string", "null"]},
                    "distance_mm": NUMBER_OR_NULL,
                    "beam_center_x_px": NUMBER_OR_NULL,
                    "beam_center_y_px": NUMBER_OR_NULL,
                    "energy_kev": NUMBER_OR_NULL,
                    "wavelength_A": NUMBER_OR_NULL,
                    "pixel_size_um": NUMBER_OR_NULL,
                    "incidence_deg": NUMBER_OR_NULL,
                    "profile_name": {"type": ["string", "null"]},
                    "note": {"type": ["string", "null"]},
                },
                ("source",),
            ),
            kind=WRITE,
        ),
        ToolSpec(
            "export_results",
            "Write the Analyze curves (CSV + JSON) and, with include_tables, the tables of this run "
            "(peaks, orientation, sizes) next to the data in gimap_analysis/. Only when the user asked "
            "for files; the user may have to approve it.",
            _object({"include_tables": {"type": "boolean"}}, ("include_tables",)),
            kind=WRITE,
        ),
        ToolSpec(
            "submit_report",
            "Finish the run: one item per requested result with status done, partial or "
            "not_available. findings quote tool numbers with units; evidence names the tools and "
            "curves; reason says exactly why something is partial or not available (e.g. 'no peak "
            "above 3σ between 0.2 and 2.0 Å⁻¹; strongest 1.8σ at 1.31 Å⁻¹'), empty when done. "
            "Call it exactly once, as the last step.",
            _object(
                {
                    "summary": {"type": "string"},
                    "items": {"type": "array", "items": REPORT_ITEM},
                    "caveats": {"type": "array", "items": {"type": "string"}},
                    "suggestions": {"type": "array", "items": {"type": "string"}},
                },
                ("summary", "items", "caveats", "suggestions"),
            ),
            kind=FINAL,
            strict=True,
        ),
    ]
    return tuple(specs)


__all__ = ["ALL_CURVES", "FINAL", "Q_CURVES", "READ", "WRITE", "ToolSpec", "tool_specs"]
