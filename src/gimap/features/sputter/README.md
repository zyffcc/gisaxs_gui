# `sputter`

Workspace **Sputter** (sidebar). Particle size R, centre-to-centre spacing D and the metal thickness that can be
stated, window by window, for an in-situ sputter series on a polymer, and how far each value can be trusted.

- **Predict** (one background task, `SputterService.predict`): check PyTorch in GIMaP's own Python (a subprocess
  import) → install it with pip when it is missing (`PipInstaller`: `requirements-sputter.txt`, the installed
  NumPy pinned, `--no-user`, `ensurepip` when pip is missing, pip's cache in `.gimap` for a source folder, progress
  from pip's lines, failures classified with the command to run by hand; `GIMAP_SPUTTER_NO_INSTALL=1` switches it
  off) → run the toolkit bundled in `utils/sputter_toolkit` (`BundledToolkit`, `scripts/sputter_struct_v1.py
  analyze`, CPU, workers = max(1, min(8, cores − 1, (free memory − 1 GB) / 0.4 GB)); `GIMAP_SPUTTER_TOOLKIT` points at another copy for
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
  without frames. A new deposition rate, t0 or deposition end redraws the result shown only when it is of the
  picker's scan. The beam centre is entered in GIMaP's pixel coordinates and converted to the toolkit's pixel-centre
  indices in `toolkit.py`.
- **Explanations** (`StructureFolderReader`): `physical` (always the default: the network's method — one stage path,
  network R/D in every island window, grade P, trends, t_eq, nominal thickness, film onset, intensity modulation
  period — `modulation_period_s`, `rotation_period_s` in older results, called *intensity modulation period (sample
  stage translation?)*: the stage is translated back and forth during the deposition — and the windows where the beam
  may be off the sample, `beam_off_sample` / `beam_off_reason` from the toolkit's `sputter_offsample.py`: counted in
  the result line and the report (without a modulation: "or interrupted"), hollow grey points in Size Trend (R, D,
  L, t_eq), a column of the Windows table; a whole dip is flagged, its core below 50 % of the level and the sides
  below 80 %; they keep their values but are left out of the stage votes, the trends, the back-and-forth counts, the
  mass check at the film onset, the literature D trend and network checks, and the sketch's medians and storyboard), `literature` (second, when a deposition rate gives it values: a check of the network, never the default —
  R_lit/D_lit of `structure_physical.csv` as R/D, grade Lit, film after its percolation (D = 2π/q* goes on, R is not
  given), a percolation interval with the rate and t0 errors and a *not determined* state; `domain/literature.py`),
  `window`, `selected`, `sequence`. A new deposition rate, t0 or deposition end calls the toolkit's `physical` command
  only and keeps the explanation shown (without a rate the literature method gives way to the physical path); t0 and
  the end are compared also without a rate (`thickness_values`: the summary records them). **Add Refined Windows…**
  refuses the fields a change of them refuses, then calls `table` then `physical`.
- **Deposition end** (More options, optional, remembered per folder like the rate and t0; `AnalyzeRequest.t_end_s`,
  `--t-end-s`, not part of the results folder's code): the shutter is closed by hand, so a scan can go on after the
  deposition; the toolkit keeps the nominal thickness (and the literature method's δ) at rate·(t_end − t0) after it.
  Size Trend draws the plateau, marks the end (an end outside the windows only in the legend: the time axis keeps to
  the windows) and puts no δ tick beyond it; the summary records `t_end_s` and `t_end_given` (`deposition_end`). The
  toolkit's command line refuses a rate that is not positive or an end not after t0 before reading any frame.
- **Literature** tab (`presentation/literature_tab.py`): D, 2R/D, coverage and η = t_eq/δ over δ against the
  published values the toolkit copies into the summary, the checks with verdicts and references, the caveats.
  Without a rate it says how to give one. The toolkit's sentences are translated by matching them against
  `presentation/toolkit_texts.py` (`texts.note_text`; a test keeps the copy equal to the bundled toolkit).
- Film windows: the thickness is not determined (the model's estimate equals its prior); the sketch draws a
  continuous film without particles. Cu is read with the Ag model and labelled “for reference only”.

Layers: `domain` (rows, series, reliability report from translatable templates, literature method, sketch), `application`
(`SputterService`, `AnalyzeRequest`, progress, environment and scan ports), `infrastructure` (`BundledToolkit`,
`PipInstaller` + `pip_output`, `process`, `CbfHeaderInspector`, `StructureFolderReader`), `presentation`
(`SputterPage` = `views/sputter_page_view.py` + `geometry.py`, `predict_flow.py`, `results_panel.py`,
`trend_plot.py`, `literature_tab.py`, `sketch_view.py`, `texts.py`, `toolkit_texts.py`, `preset_dialog.py`),
`bootstrap.py`. Chinese texts: `app/presentation/i18n/zh_sputter.py`, `zh_sputter_report.py`,
`zh_sputter_literature.py`, `zh_sputter_toolkit.py`. Tests: `tests/test_sputter_feature.py`,
`tests/test_sputter_page.py` (fakes in `tests/sputter_fixtures.py`). Page: `docs/ui/workspaces/sputter.md`.
