"""Load ordered mask recipes and detector bitmaps, and export the current user mask."""

import json
from dataclasses import replace
from pathlib import Path

from PyQt5.QtWidgets import QFileDialog

from src.gimap.app.presentation.i18n import tr, trf
from src.gimap.shared.detector_io.mask_files import write_mask_export

from ...application import MaskImage, MaskRule, analysis_metadata, mask_edit_from_record, mask_edit_record
from ..texts import message_text

FORMATS = {
    ".json": "GIMaP masks (*.json)", ".edf": "EDF (*.edf)", ".tif": "TIFF (*.tif *.tiff)",
    ".npy": "NumPy (*.npy)", ".msk": "Fit2D (*.msk)",
}
ALL_MASKS = "Masks (*.json *.edf *.tif *.tiff *.npy *.msk)"
NOT_JSON = "{name} is not a GIMaP masks file (it is not valid JSON)."
NO_MASK_LIST = "{name} is not a GIMaP masks file (it has no masks list)."


class _NotMasks(ValueError):
    pass


def _shapes_of(path: Path) -> list:
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise _NotMasks(NOT_JSON) from exc
    try:
        items = record["masks"]
        if not isinstance(items, list):
            raise _NotMasks(NO_MASK_LIST)
        edits = [mask_edit_from_record(item) for item in items]
    except (KeyError, TypeError, IndexError, AttributeError) as exc:
        raise _NotMasks(NO_MASK_LIST) from exc
    imported = []
    for edit in edits:
        sources = tuple(dict.fromkeys((*edit.sources, str(path.resolve()))))
        if isinstance(edit, MaskImage):
            edit = replace(edit, path=str((path.parent / edit.path).resolve()))
        imported.append(replace(edit, sources=sources))
    return imported


class MaskFilesMixin:
    def save_masks(self, path: str | Path | None = None):
        state = self.view_model.state
        corrections = state.corrections
        edits = list(corrections.mask_shapes)
        if corrections.mask_path:
            edits.insert(0, MaskImage(str(Path(corrections.mask_path).resolve())))
        if not edits:
            return None
        if path is None:
            folder = self.view_model.default_export_dir() or Path(self._last_folder or ".")
            filters = ";;".join(tr(label) for label in FORMATS.values())
            path, selected = QFileDialog.getSaveFileName(self, tr("Save Masks"), str(folder / "masks"), filters)
            if not path:
                return None
            if not Path(path).suffix:
                suffix = next((ext for ext, label in FORMATS.items() if tr(label) == selected), ".json")
                path += suffix
        path = Path(path)
        analysis = state.analysis
        record = {
            "software": "GIMaP Analyze", "convention": "row 0 at top, pixel centres at half integers, non-zero = masked",
            "frame_shape": list(analysis.shape) if analysis is not None else None,
            "masks": [mask_edit_record(edit) for edit in edits],
        }
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            mask = None
            if path.suffix.lower() != ".json":
                if analysis is None or analysis.corrections != corrections or self._center_analysis_pending:
                    raise ValueError("Wait for the current frame analysis before saving a mask image.")
                record["analysis"] = analysis_metadata(analysis)
                sources = {Path(p).resolve() for edit in edits for p in edit.sources}
                sources.update(Path(edit.path).resolve() for edit in edits if isinstance(edit, MaskImage))
                sources.add(Path(analysis.path).resolve())
                sources.update(Path(p).resolve() for p in record["analysis"].get("source_files", ()))
                sources.update(Path(p).resolve() for p, _index in analysis.summed_frames or ())
                outputs = {path.resolve(), path.with_suffix(path.suffix + ".json").resolve()}
                if sources & outputs:
                    raise ValueError("Choose a different output file to preserve the mask and frame sources.")
                mask = analysis.drawn_mask
                if mask is None:
                    raise ValueError("The displayed frame has no user mask.")
            write_mask_export(path, record, mask)
        except (OSError, ValueError) as exc:
            self._status(trf("Could not save the mask: {error}", error=message_text(exc)), "error")
            return None
        self.notify_written(trf("Saved {name}", name=path.name), path.parent)
        return path

    def load_masks(self, path: str | Path | None = None) -> bool:
        if path is None:
            path, _ = QFileDialog.getOpenFileName(self, tr("Load Masks"), self._last_folder, tr(ALL_MASKS))
            if not path:
                return False
        path = Path(path)
        vm = self.view_model
        corrections = vm.state.corrections
        try:
            if path.suffix.lower() == ".json":
                edits = _shapes_of(path)
                for edit in edits:
                    if isinstance(edit, MaskImage):
                        vm.check_mask_image(edit.path)
                analysis = vm.state.analysis
                if analysis is not None:
                    if analysis.geometry is None and any(edit.kind == "q_annulus" for edit in edits):
                        raise ValueError("A beam-centred q annulus needs detector geometry.")
                    if any(isinstance(edit, MaskRule) and edit.relative for edit in edits):
                        # A relative threshold needs valid samples after all the preceding edits.
                        vm.preview_user_mask(analysis, replace(
                            corrections, mask_shapes=(*corrections.mask_shapes, *edits)))
                self._observe_setup()
                vm.set_mask_shapes((*corrections.mask_shapes, *edits))
            else:
                vm.check_mask_image(path)
                self._observe_setup()
                if not corrections.mask_path and not corrections.mask_shapes:
                    vm.set_mask_path(path.resolve())
                else:
                    vm.add_mask_shape(MaskImage(str(path.resolve())))
        except _NotMasks as exc:
            self._status(trf(str(exc), name=path.name), "error")
            return False
        except OSError as exc:
            self._status(trf("Could not read the masks: {error}", error=exc.strerror or exc), "error")
            return False
        except ValueError as exc:
            self._status(trf("Mask not loaded: {error}", error=message_text(exc)), "error")
            return False
        self._refresh_mask_list()
        self._status(trf("Masks loaded from {name}", name=path.name), "ok")
        self.run_analysis()
        return True
