"""Record the Analyze setup before automatic analysis changes the frame, geometry or cuts."""

from dataclasses import asdict
from pathlib import Path

from ..application.record_values import plain
from ..application.settings_file import settings_from_record, settings_record
from .geometry_model import HEADER_CENTER_KEY, SECTION


def automatic_state(page, *, initial):
    model = page.view_model
    analysis = model.state.analysis
    state = {"inputs": {}, "summed_files": []}
    if analysis is not None:
        state["inputs"] = {**analysis.frame_files, **analysis.geometry_files, **analysis.correction_files}
        state["summed_files"] = list(dict.fromkeys(str(path) for path, _ in analysis.summed_frames
                                                  if path != analysis.path))
    if initial:
        state.update(settings=plain(settings_record(model.current_settings())),
                     frame_index=model.state.frame_index)
        if analysis and analysis.resolution.how == "file" and state["settings"]["profile"]:
            # A sibling PONI is read afresh on every run. Its transient profile timestamp is not an input.
            state["settings"]["profile"].pop("updated_at", None)
        state["settings"].update(use_header_center=model.use_header_center,
                                 forced_shapes=list(model.state.forced_shapes),
                                 held_cut=plain(model.held_cut(model.current_path, model.state.frame_index)))
        state["import_selections"] = model.export_import_selections()
    else:
        state["indexing"] = asdict(page.indexing.settings)
        state["phases"] = [
            {"name": phase.name, "cell": list(phase.cell), "centring": phase.centring,
             "orientation": phase.orientation, "indices": list(phase.indices),
             "cif": phase.source, "sha256": phase.source_sha256}
            for phase in page.indexing.phases if phase.visible
        ]
    return state


def restore_automatic_state(page, recipe):
    """Reinstate selected frames and the Analyze setup before the automatic procedure starts."""
    model = page.view_model
    settings = recipe["settings"]
    model.apply_settings(settings_from_record(settings))
    model.settings.set(SECTION, HEADER_CENTER_KEY, settings.get("use_header_center", False))
    model.state.forced_shapes = tuple(tuple(shape) for shape in settings.get("forced_shapes", ()))
    for path, selection in recipe.get("import_selections", {}).items():
        model.set_import_selection(path, selection)
    model.set_frame(recipe["frame_index"])
    model.show_with_cut(Path(recipe["frame"]), recipe["frame_index"], settings.get("held_cut"))
    page._sync_settings_widgets()
    page.run_analysis()
