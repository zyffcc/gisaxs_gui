# `sputter`

Workspace **Sputter** (sidebar): the structure of an in-situ sputter series, time window by time window. The page
has two views under one input bar (scan folder, metal, substrate, deposition rate, **Analyze**, the geometry in one
line with **Edit**, **Open Result…**):

- the **main screen** (shown first): the result of **Analyze**, which runs the client analysis package — for the
  window chosen, the measured image beside the model image of a solution, the solutions to choose from (the
  network's candidates and the refined one, each with its fit deviation), the values of the solution shown (grain
  radius R, height H, distance D, gap, size distribution width, coverage) and their evolution over the scan. Nothing
  on it grades a value: solutions the data cannot tell apart are simply all offered;
- **Details**: the geometry, the options of Analyze, and the earlier method, **Predict** with the networks bundled
  with GIMaP — particle size R, spacing D and the metal thickness that can be stated for a series on a polymer, and
  how far each value can be trusted — with its five tabs.

## Analyze (the main screen)

- **The package** is not part of GIMaP: a self-contained folder (`code`, `models`; NumPy / SciPy / Matplotlib only).
  `infrastructure/client.py` `locate_package`: the environment variable `GIMAP_SPUTTER_CLIENT`, else the folder the
  user chose (Details ▸ Analyze options ▸ **Choose Folder…**, kept in the preferences), else `utils/sputter_client`
  of the program folder. Without it the main screen says so, with **Choose Folder…**; **Open Result…** works anyway.
- **Analyze** (one background task, `ClientService.analyze`, TaskRunner key `sputter_client`): what the package's
  `run.cmd analyze` does, in GIMaP's own Python (`ClientPackageRunner`: `-u -m iampred.analyze --frames … --label
  gui_<scan>_<hash> --metal … --substrate … (--rate-nm-s | --qcm-nm-s) … --alpha-i … --out … --mode … --workers N
  --data … -- --sdd-mm … (--energy-ev | --wavelength-nm) … (--beam-col … --beam-row … | --auto-centre) --roll …
  [--mask …] [--t-dep …] [--t-sput …] [--poly-t …]`, cwd `<package>/code/iam`, `IAM_*`, `IAMPRED_MODELS`,
  `PYTHONPATH`, `TZ`, `OMP_NUM_THREADS` set). The beam centre is entered in GIMaP's pixel coordinates and handed
  over as pixel-centre indices (− 0.5), as for the bundled toolkit. Progress from the package's `[n/N] step ...`
  lines (`ClientProgress`); Cancel ends the process tree.
- **Labels and folders**: the package never imports a label twice, so the label's hash is of everything the import
  depends on (`ClientRequest.intake_values`). Output: `<the scan's results folder>/iampred-<hash of the request, the
  mode and the package's version>`; the same request to the same package shows the result it has. Data and work
  folders: `<user data>/sputter_client/<package version>/data/<pixel mask>` and `…/work`. An import that was cut off
  is removed before the next run (the package would refuse its label for ever); a finished one has the marker
  `<label>.gimap.json`. `gimap_request.json` in the output folder records the request, the package and the command.
- **Result** (`ClientFolderReader` → `domain/client.py`): `analysis.json`, `main/main.json` (windows with their
  entries, `pick`, `smooth`; a window may be `{t, error}`) and `main/w_<t>.npz` (the measured image and the model of
  every entry on cells of exit angle × q_y). The entry shown first is `smooth` (the smooth evolution over the scan),
  else `pick` (the lowest fit deviation). A choice of the user is kept per window in `main/choices.json` (only where
  it differs from the default; in the user data folder when the result cannot be written to).
- **Pictures** (Matplotlib, as the rest of the page): `pattern_view.PatternPair` draws both images on the result's
  own cells with shared axes and one logarithmic colour scale set by the measured image;
  `evolution_view.EvolutionPlot` draws R, H, D, the gap and the coverage of the chosen entries over the scan, marks
  the window shown and selects the window nearest to a click.
- **Real space** (the switch Patterns | Real Space beside the window selector): `real_space_view.RealSpacePicture`
  draws the chosen entry in place of the two images — a cut through a row of grains (substrate or polymer film,
  the continuous base layer `base_nm`, caps of radius r and height (H − base)·r/R) and the top view of a square of
  7 D (log-normal radii around R on a hexagonal arrangement of distance D, each shifted at random by about 0.12 D).
  The arrangement is `domain/real_space.py` (pure Python, one seed: another window or entry changes the sizes, not
  the dice). `base_nm` is not a shown quantity; the reader keeps it for the picture (`SKETCH_KEYS`). Grains that
  touch form a cluster (`clusters`) drawn as one piece; a cluster across the view has percolated (`spanning`,
  `arrangement`: islands, coalescing, network) and is drawn darker; a line above the picture says the arrangement,
  or the continuous layer and the grains on it. A porous TiO₂ entry is drawn from where its metal sits
  (`porous_space`: the package's `porous_geometry` keys): a cross-section of the layer to scale — particles, the pore
  metal as a band along its Gaussian in depth, the cap on a share of the tops — and the top of one particle enlarged
  (cap thickness, clusters). An entry without any of this says why.
- **Porous substrate** (`ClientResult.porous`): the tiles and the evolution show the island radius (relative), the
  width, the island, cap and film shares and the two correlation lengths of the metal in the pores.
- **Static frames** (Details ▸ Static Frames, `static_flow.py`, `views/sputter_static_view.py`): one sample measured
  after its deposition is analysed into a series (`StaticService` in `application/client_static.py`,
  `StaticPackageRunner` in `infrastructure/client_static.py`: `-u -m iampred.static --frames … --label …
  --nominal-nm … --out <series> …`, the package's own geometry for static frames unless a field is filled in,
  `IAM_STATIC_DATA` = `<user data>/sputter_client/static/<series id>`). The series is shown like a scan, one
  sample in the place of a time window, along the nominal thickness. `gimap_series.json` in the series folder
  records what each sample was analysed from; the same name with other inputs replaces the sample.

## Predict (Details: the earlier method)

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
- **Explanations** (`StructureFolderReader`): `literature` first — the default — when the summary's literature block
  has values (see below), then `physical` (the default without them: the network's method — one stage path,
  network R/D in every island window, grade P, trends, t_eq, nominal thickness, film onset, intensity modulation
  period — `modulation_period_s`, `rotation_period_s` in older results, called *intensity modulation period (sample
  stage translation?)*: the stage is translated back and forth during the deposition — and the windows where the beam
  may be off the sample, `beam_off_sample` / `beam_off_reason` from the toolkit's `sputter_offsample.py`: counted in
  the result line and the report (without a modulation: "or interrupted"), hollow grey points in Size Trend (R, D,
  L, t_eq), a column of the Windows table; a whole dip is flagged, its core below 50 % of the level and the sides
  below 80 %; they keep their values but are left out of the stage votes, the trends, the back-and-forth counts, the
  mass check at the film onset, the literature D trend and network checks, and the sketch's medians and storyboard), `literature` (first and the default when a deposition rate gives it values —
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
- **Show in Analyze** (`presentation/window_menu.py`; right-click a Windows row, the Structure Sketch or a point of
  Size Trend): `SputterService.window` → `BinnedWindowReader` (`infrastructure/windows.py`: the window's row of
  `binned/chunk_NNNNN.json`, its `raw_members` in the scan folder of `gimap_request.json`, the geometry of the request,
  the model's own beam-centre estimate of `intake/geometry.json`; problems from `domain/windows.py`) → the target the
  composition root sets (`app/sputter_handover.py`), which shows the frames summed in Analyze.
- **Island height H from the q_z fringes** (`domain/fringes.py`, `application/fringes.py` `FringeService`,
  `infrastructure/fringes.py` `FringeTool` + `FringeFolderStore`, `presentation/fringe_flow.py` + `fringe_panel.py`):
  after a result is shown, fresh outputs of `utils/sputter_fringes` (`fringe_heights.csv`, `fringe_summary.json`; in
  the result's `fringes_v1/`, else `<user data>/sputter_fringes/<result>/`; fresh = the same tool sources, `binned/
  run.json`, metal and off-sample windows) are read at once, else the tool runs in the background (a subprocess of
  GIMaP's Python, numpy/scipy only; `FRINGES_KEY`). Measured H adds a Size Trend panel (also in the simple view), two
  Windows columns (the nearest time group) and a Reliability paragraph; a failure goes to the log only.
- Film windows: the thickness is not determined (the model's estimate equals its prior); the sketch draws a
  continuous film without particles. Cu is read with the Ag model and labelled “for reference only”.

Layers: `domain` (rows, series, reliability report from translatable templates, literature method, sketch, the frames
of a window; `client.py`: the windows, entries and images of an Analyze result; `real_space.py`: where the grains of
an entry sit), `application` (`SputterService`,
`AnalyzeRequest`, progress, environment, scan and window ports; `client.py`: `ClientService`, `ClientRequest` and its
ports, `client_progress.py`, `client_static.py`), `infrastructure` (`BundledToolkit`, `PipInstaller` + `pip_output`, `process`,
`CbfHeaderInspector`, `StructureFolderReader`, `BinnedWindowReader`; `client.py`: `ClientPackageRunner`,
`ClientFolderReader`, `locate_package`; `client_static.py`: `StaticPackageRunner`), `presentation` (`SputterPage` = `views/sputter_page_view.py` +
`views/sputter_main_view.py` + `views/sputter_static_view.py` + `client_flow.py`, `client_panel.py`, `client_texts.py`,
`pattern_view.py`, `real_space_view.py`, `evolution_view.py`, `static_flow.py`, `geometry.py`, `predict_flow.py`, `results_panel.py`, `window_menu.py`, `trend_plot.py`,
`literature_tab.py`, `sketch_view.py`, `texts.py`, `toolkit_texts.py`, `preset_dialog.py`), `bootstrap.py`. Chinese
texts: `app/presentation/i18n/zh_sputter.py`, `zh_sputter_report.py`, `zh_sputter_literature.py`,
`zh_sputter_toolkit.py`, `zh_sputter_handover.py`, `zh_sputter_fringes.py`, `zh_sputter_client.py`. Tests:
`tests/test_sputter_feature.py`, `tests/test_sputter_page.py`, `tests/test_sputter_show_in_analyze.py` (fakes in
`tests/sputter_fixtures.py`), `tests/test_sputter_fringes.py`, `tests/test_sputter_fringes_page.py` (synthetic cuts in
`tests/sputter_fringe_data.py`), `tests/test_sputter_client.py`, `tests/test_sputter_main_screen.py` (a fake of the
package's process and invented results in `tests/sputter_client_fixtures.py`). Page: `docs/ui/workspaces/sputter.md`.
