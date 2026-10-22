# `sputter`

Workspace **Sputter** (sidebar). Particle size R, centre-to-centre spacing D and the metal thickness that can be
stated, window by window, for an in-situ sputter series on a polymer, and how far each value can be trusted.

- **Predict** (one background task, `SputterService.predict`): check PyTorch in GIMaP's own Python (a subprocess
  import) → install it with pip when it is missing (`PipInstaller`: `requirements-sputter.txt`, the installed
  NumPy pinned, `--no-user`, `ensurepip` when pip is missing, pip's cache in `.gimap` for a source folder, progress
  from pip's lines, failures classified with the command to run by hand; `GIMAP_SPUTTER_NO_INSTALL=1` switches it
  off) → run the toolkit bundled in `utils/sputter_toolkit` (`BundledToolkit`, `scripts/sputter_struct_v1.py
  analyze`, CPU, workers = max(1, min(8, cores − 1)); `GIMAP_SPUTTER_TOOLKIT` points at another copy for
  development) → read the result. Progress: `SPUTTER_PROGRESS`, `INTAKE_PLAN`, `INTAKE_CHUNK`, `INTAKE_EXCLUDED`
  (`ProgressTracker`). Stop ends the whole process tree (`process.run_streaming`: polled every 0.2 s, `taskkill /T
  /F` on Windows); closing the window stops pip even while it writes.
- **Results folder**: `<user data>/sputter/<scan>-<sha1 of the request and the model's version, 8>`
  (`results_folder`; the version is `BundledToolkit.model_id()`, the SHA-256 of `MANIFEST.json`): the same request
  to the same model reuses it (the toolkit goes on where it stopped), another geometry or another model gets a new
  one. `gimap_request.json` records the scan folder as chosen, the request, the model's version, the Python and
  PyTorch's version. A PyTorch that does not load after an installation cut off while pip wrote the packages is
  installed again over it on the next Predict (`--force-reinstall --no-deps`).
- **Form**: `CbfHeaderInspector` reads the first frame's text header (detector, serial number, pixel size,
  exposure and period, time stamp, flat field; an energy hint from a `Wavelength` line, a companion NeXus
  `beam_energy` or the `E<eV>` of the flat-field name — the last is the detector's calibration energy and is only
  shown beside the field). Geometry comes from the folder's own last values, else a matching user preset (serial
  number + flat field + date range), else (the energy) the header or NeXus hint, else the last values — not the
  sample roll (0, said as assumed) or the mask (`application/scan.py`: `prefill`, `remember`, `matching_preset`);
  each field says its source. Predict waits for the header of the folder in the picker and refuses a folder
  without frames. A new deposition rate redraws the result shown only when it is of the picker's scan. The beam centre is
  entered in GIMaP's pixel coordinates and converted to the toolkit's pixel-centre indices in `toolkit.py`.
- **Explanations** (`StructureFolderReader`): `physical` (default: `structure_physical.csv` — one stage path, network
  R/D in every island window, grade P, trends, t_eq, nominal thickness, film onset, rotation period), `window`,
  `selected`, `sequence`. A new deposition rate or t0 calls the toolkit's `physical` command only; **Add Refined
  Windows…** calls `table` then `physical`.
- Film windows: the thickness is not determined (the model's estimate equals its prior); the sketch draws a
  continuous film without particles. Cu is read with the Ag model and labelled “for reference only”.

Layers: `domain` (rows, series, reliability report from translatable templates, sketch), `application`
(`SputterService`, `AnalyzeRequest`, progress, environment and scan ports), `infrastructure` (`BundledToolkit`,
`PipInstaller` + `pip_output`, `process`, `CbfHeaderInspector`, `StructureFolderReader`), `presentation`
(`SputterPage` = `views/sputter_page_view.py` + `geometry.py`, `predict_flow.py`, `results_panel.py`,
`trend_plot.py`, `sketch_view.py`, `texts.py`, `preset_dialog.py`), `bootstrap.py`. Chinese texts:
`app/presentation/i18n/zh_sputter.py`, `zh_sputter_report.py`. Tests: `tests/test_sputter_feature.py`,
`tests/test_sputter_page.py` (fakes in `tests/sputter_fixtures.py`). Page: `docs/ui/workspaces/sputter.md`.
