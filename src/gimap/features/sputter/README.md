# `sputter`

Workspace **Sputter** (sidebar): the structure of an in-situ sputter series, time window by time window. The page
has two views under one input bar (scan folder, metal, substrate, deposition rate with its unit, **Analyze**, the
geometry in one line with **Edit**, **Open Result…**):

- the **main screen** (shown first): the result of **Analyze**, which runs the client analysis package — for the
  window chosen, the measured image beside the model image of a solution, the solutions to choose from (the
  network's candidates, the refined one and the joint fit of the whole scan, each with its fit deviation), the values
  of the solution shown (grain radius R, height H, distance D, gap, size distribution width, coverage) and their
  evolution over the scan. Nothing on it grades a value: solutions the data cannot tell apart are simply all offered;
- **Details**: at the left the geometry, the options of Analyze, Static Frames and More options; at the right the
  result the main screen shows, from the files of its folder: **Windows**, **Report** and **Logs**.

## Analyze (the main screen)

- **The package** comes with GIMaP: `utils/sputter_client`, a release build of the research code (only the files
  the commands below use, no names of the private data it was developed on; `MANIFEST.json` and
  `tests/test_sputter_client_bundle.py` check it). A self-contained folder (`code`, `models`; NumPy / SciPy /
  Matplotlib only).
  `infrastructure/client.py` `locate_package`: the environment variable `GIMAP_SPUTTER_CLIENT`, else the folder the
  user chose (Details ▸ Analyze options ▸ **Choose Folder…**, kept in the preferences), else `utils/sputter_client`
  of the program folder. A named folder (environment or chosen) that is not a package gives way to the program
  folder's package, with the reason in `ClientPackage.note` (the Analysis package line, marked, shows it); only
  without that package is it the problem. Without any the main screen says so, with **Choose Folder…**;
  **Open Result…** works anyway.
- **Analyze** (one background task, `ClientService.analyze`, TaskRunner key `sputter_client`): what the package's
  `run.cmd analyze` does, in GIMaP's own Python (`ClientPackageRunner`: `-u -m iampred.analyze --frames … --label
  gui_<scan>_<hash> --metal … --substrate … (--rate-nm-s | --qcm-nm-s) … --alpha-i … --out … --mode … --workers N
  --data … -- --sdd-mm … (--energy-ev | --wavelength-nm) … (--beam-col … --beam-row … | --auto-centre) --roll …
  [--mask …] [--t-dep …] [--t-sput …] [--poly-t …]`, cwd `<package>/code/iam`, `IAM_*`, `IAMPRED_MODELS`,
  `PYTHONPATH`, `TZ`, `OMP_NUM_THREADS` set; workers N = `ClientRequest.workers`: the number chosen in Details ▸
  Analyze options ▸ **Parallel processes** (1 … cores), or for **Automatic (N)** (the default) max(1, min(16, cores − 2,
  (free memory − 1.5 GB) / 0.3 GB), `infrastructure/process.py` `default_workers`, through
  `ClientService.automatic_workers`) worked out when the analysis starts (`presentation/workers_flow.py`; kept in the
  preferences as `workers`, a number or `auto`; Static Frames use the same number). Not part of the request's hash: the
  same request with another number shows the result it has; `ClientStarted.workers` hands it to the live view). The
  beam centre is entered in GIMaP's pixel coordinates and handed over as pixel-centre indices (− 0.5). Progress from
  the package's `[n/N] step ...` lines (`ClientProgress`); Cancel ends the process tree (`process.run_streaming`:
  polled every 0.2 s, `taskkill /T /F` on Windows).
- **The live view** (`live_panel.py`, events in `application/client_live.py`): while Analyze runs, the main screen
  shows the steps and what each found so far. The clock line adds *Processes: n* (`reading.workers`, else the number
  GIMaP passed). A newer package's `frames` events carry `frames`, `read_s`, `wall_s`, `mb`, and a `reading` event ends
  the import: `LiveRun` adds the chunks up (`disk_per_frame`, `disk_share`, the `reading` values first), the import's
  line says *Reading a of b frames · disk x s per frame (y % of the time)* (`reading_texts.py`), and when the disk takes
  ≥ 0.15 s per frame and ≥ 50 % of the time, or `reading.slow`, a warning banner (`live_disk_note`) says that reading
  from this disk is the slow part and that copying the scan folder to a local SSD helps. An older package: neither.
- **Deposition rate** (`presentation/rate_flow.py`): typed in the unit chosen beside it, **Å/s** (the default: what a
  quartz balance shows) or **nm/s**; the request carries nm/s (Å/s ÷ 10, to 12 significant digits). The unit is kept
  with the rate per scan folder and as the last used one; a folder or project an older page remembered has its rate in
  nm/s. **QCM reading** ticked: the number is the balance's reading and the package takes `QCM_FACTOR` (0.6) of it at
  the sample (`ClientRequest.rate_at_sample`). A rate at the sample above 1 nm/s or below 0.005 nm/s
  (`ClientRequest.unusual_rate`) asks once before Analyze runs whether the unit is right (**Analyze Anyway** /
  Cancel). The scan's line under the bar adds the nominal thickness at the last frame: rate at the sample × (frames ×
  period − onset), or × the deposition duration when that is shorter.
- **Labels and folders**: the package never imports a label twice, so the label's hash is of everything the import
  depends on (`ClientRequest.intake_values`). Output: `<the scan's results folder>/iampred-<hash of the request, the
  mode and the package's version>`; the same request to the same package shows the result it has. The scan's results
  folder is `<results root>/<scan>-<hash of the folder, metal and geometry>` (`application/use_cases.py`
  `scan_results_folder`; the hash keeps the values the page used to send to the earlier model, `FOLDER_VALUES`, so
  that results analysed then are found again); the results root is `<user data>/sputter` unless More options ▸ Save
  results in names another. Data and work folders: `<user data>/sputter_client/<package version>/data/<pixel mask>`
  and `…/work`. An import that was cut off is removed before the next run (the package would refuse its label for
  ever); a finished one has the marker `<label>.gimap.json`. `gimap_request.json` in the output folder records the
  request, the package and the command.
- **Result** (`ClientFolderReader` → `domain/client.py`): `analysis.json`, `main/main.json` (windows with their
  entries, `pick`, `smooth`; a window may be `{t, error}`) and `main/w_<t>.npz` (the measured image and the model of
  every entry on cells of exit angle × q_y). The entry shown first is `smooth` (the smooth evolution over the scan;
  the joint fit of the whole scan, label `joint`, when the package made one), else `pick` (the lowest fit deviation).
  Entries are named in `client_texts.entry_name`: *Candidate n*, *Refined*, *Joint fit (whole scan)*, *Model
  solution*; refined and joint entries have no network weight. A choice of the user is kept per window in
  `main/choices.json` (only where it differs from the default; in the user data folder when the result cannot be
  written to).
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
  `arrangement`: islands, coalescing, network) and is drawn darker; a line above the picture says the arrangement.
  An entry with a base layer (`RealSpace.continuous`: `base_nm` > 0) is a closed film: the top view is the metal of
  the base over the whole square, the grains on it are relief (slightly darker, thin outlines), and the line says
  *Continuous film x nm, grains on top: …*. A porous TiO₂ entry is drawn from where its metal sits
  (`porous_space`: the package's `porous_geometry` keys): a cross-section of the layer to scale — particles, the pore
  metal as a band along its Gaussian in depth, the cap on a share of the tops — and the top of one particle enlarged
  (cap thickness, clusters). An entry without any of this says why.
- **Porous substrate** (`ClientResult.porous`): the tiles and the evolution show the island radius (relative), the
  width, the island, cap and film shares and the two correlation lengths of the metal in the pores.
- **Static frames** (the bar's **After Deposition**, `kind_switch` → `input_stack` page 1: frames, name, nominal
  thickness, bare substrate; Details ▸ Static Frames keeps the series, mode and geometry; `static_flow.py`,
  `views/sputter_static_view.py`; Analyze dispatches by `analysis_kind()`; the live view's static half is
  `live_static.py`, `LiveRun(kind="static")`): one sample measured after its deposition is analysed into a series
  (`StaticService` in `application/client_static.py`, `StaticPackageRunner` in `infrastructure/client_static.py`:
  `-u -m iampred.static --frames … --label … --nominal-nm … --out <series> …`, the package's own geometry for static
  frames unless a field is filled in, `IAM_STATIC_DATA` = `<user data>/sputter_client/static/<series id>`). The series
  is shown like a scan, one sample in the place of a time window, along the nominal thickness. `gimap_series.json` in
  the series folder records what each sample was analysed from; the same name with other inputs replaces the sample.

## Details

- **Left** (`views/sputter_page_view.py`): Sample and geometry (X-rays, distance, incidence, beam centre; each field
  says where its value comes from, `application/scan.py`: `prefill`, `remember`, `matching_preset`; **Save Geometry
  as Preset…**); Analyze options (`views/sputter_main_view.py`: mode, parallel processes, onset, duration, polymer
  film thickness, the package and its folder, Open Report, Open Folder, the messages of the run); Static Frames (collapsed); More options
  (collapsed): the pixel mask with **Masks of the Package**, the sample roll, the folder results go to.
- **The form** (`geometry.py`): `CbfHeaderInspector` reads the first frame's text header (detector, serial number,
  pixel size, exposure and period, time stamp, flat field; an energy hint from a `Wavelength` line, a companion NeXus
  `beam_energy` or the `E<eV>` of the flat-field name — the last is the detector's calibration energy and is only
  shown beside the field). Geometry comes from the folder's own last values, else a matching user preset (serial
  number + flat field + date range), else (the energy) the header or NeXus hint, else the last values — not the
  sample roll (0, said as assumed) or the mask. Analyze waits for the header of the folder in the picker and refuses
  a folder without frames. The memory of an older page (its polymer, time window, rate per folder, the folder of the
  earlier method's result) is not read; the keys nothing reads (`DROPPED_KEYS`) go when the memory is written.
- **Right** (`result_tabs.py`, read with `ClientService.texts` → `ClientFolderReader.texts`): **Windows**, one row per
  window (a static series: per sample) of the solution the main screen shows — time or sample, nominal thickness, the
  tiles the result has, fit deviation, which solution; a window without one says why; the rows follow a choice on the
  main screen; copy with Ctrl+C. **Report**: the facts of `analysis.json` (the network used, the networks tried with
  their fit deviation, the mode, the pixel mask, the run time of the steps); **Time per step** (`steps_html`): every
  step of `steps` with its time (m:ss), the import row with `reading` (frames read of how many, the disk's seconds per
  frame and share of the time), the total and the processes (`workers`); a note when `reading` says the disk was the
  slow part (`reading_was_slow`); then `REPORT.md` rendered as Markdown (its images found in the result folder). An
  older result without `reading` / `workers` has the rows it can. **Logs**: the `analyze_*.log` of every step, in the order of the steps.
  Without a result each tab says *Analyze a scan or open a result*.

Layers: `domain` (`client.py`: the windows, entries and images of an Analyze result; `real_space.py`: where the grains
of an entry sit), `application` (`use_cases.py`: `SputterService`, the scan's results folder, `METALS`, `PROBLEMS`,
`Stopped`; `client.py`: `ClientService`, `ClientRequest`, `ResultTexts` and the ports; `client_progress.py`,
`client_static.py`; `client_live.py`: the live events of a running Analyze and `LiveRun`; `scan.py`: the scan header
and where each value comes from), `infrastructure` (`client.py`: `ClientPackageRunner`, `ClientFolderReader`,
`locate_package`; `client_static.py`: `StaticPackageRunner`; `process.py`; `scan_header.py`: `CbfHeaderInspector`),
`presentation` (`SputterPage` = `views/sputter_page_view.py` + `views/sputter_main_view.py` +
`views/sputter_live_view.py` + `views/sputter_static_view.py` + `client_flow.py`, `rate_flow.py`, `client_panel.py`,
`result_tabs.py`, `client_texts.py`, `reading_texts.py`, `workers_flow.py`, `live_panel.py`, `live_pictures.py`, `live_static.py`, `pattern_view.py`,
`real_space_view.py`, `evolution_view.py`, `static_flow.py`, `geometry.py`, `project_state.py`, `texts.py`,
`preset_dialog.py`), `bootstrap.py`. Chinese texts: `app/presentation/i18n/zh_sputter.py`, `zh_sputter_client.py`,
`zh_sputter_live.py`. Tests: `tests/test_sputter_feature.py`, `tests/test_sputter_page.py` (fakes in
`tests/sputter_fixtures.py`), `tests/test_sputter_client.py`, `tests/test_sputter_main_screen.py`,
`tests/test_sputter_details.py`, `tests/test_sputter_live.py`, `tests/test_sputter_reading.py` (a fake of the package's process and invented results in
`tests/sputter_client_fixtures.py`), `tests/test_sputter_client_bundle.py`. Page: `docs/ui/workspaces/sputter.md`.
