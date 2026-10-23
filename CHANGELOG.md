# Changelog

What changes for someone who uses GIMaP, version by version. Developer notes are in `docs/`.

## Unreleased

### User manual

- The manual is rewritten along the work: a quick start on the public data in the repository, then
  installation, the window, geometry and calibration, Analyze, Fitting, Compare, the tool windows
  (including SEM Particle Analysis), the AI assistant, Labs, a table of every output file, the known
  limitations in one chapter, and troubleshooting. It has screenshots and a Chinese edition,
  `docs/User_Manual_zh.md`; **Help ▸ User Manual** (`F1`) opens the edition of the interface language.

### SEM particle analysis

- **Tools → SEM Particle Analysis** measures calibrated image instances or imports particle-size
  CSVs independently of scattering prediction. **2D Prediction → SEM comparison** copies a radius
  prediction into the same tool. A local StarDist model can separate touching particles; existing
  integer labels and threshold/watershed remain available. StarDist runs in a cancellable local
  subprocess, with no image upload or automatic model download. Optional dependencies are listed
  in `requirements-sem-stardist.txt`; model weights stay outside the source repository.
- Exports include particle radii and diameters, comparison plots, reusable instance-label TIFFs
  and source/settings/model-hash records. Review outlines and the explicit image calibration
  before interpreting size statistics; general pretrained nuclei models are not SEM-validated.

### Sputter

- The **Sputter** workspace predicts an in-situ sputter series from its scan folder: choose the folder, check
  the geometry (each value says where it comes from: this folder's last values, a geometry preset of the
  beamtime, the file header, or the values used last — please check), press **Predict**. The model is part of
  GIMaP and runs in GIMaP's own Python; the first Predict installs PyTorch there (about 1.2 GB). The default
  explanation is a physical path of stages through time with the network's sizes in every island window, so
  values no longer flip between sources from window to window. The metal thickness is given where it can be
  stated: the island-volume equivalent t_eq and, from a deposition rate, the nominal thickness. The period of a
  periodic intensity modulation is found (see below). Cu is accepted and labelled as approximated with the Ag
  model.
- With a deposition rate, the default explanation is the **literature method (mass balance)** of the P03 in-situ
  GISAXS studies: D = 2π/q* of the direct peak, R from the nominal thickness δ = rate × (t − t0) by mass balance
  for hemispheres (grade Lit), and the percolation where 2R/D reaches 1. Its interval holds the scatter of the
  windows and the errors of the rate (±10%) and t0 (±2 s, ±0.5 s when given); when 2R/D may reach 1 within them
  but need not, the percolation is *not determined*. After it D = 2π/q* goes on and R is not given. Peaks that
  probably belong to the polymer, jumps and outliers are marked and left out; late peaks after a gap are compared
  with the trend of D. The new **Literature** tab plots D, 2R/D, coverage and the
  network's mass balance over δ against published values (Au/PS, Ag/PS, Ag/PMMA, Cu/graphene; thresholds by metal
  and host, each with its citation and DOI) and lists checks — consistent, inconsistent or not testable — with
  their references. The literature-method R agrees with the literature by construction; the page says so.
- The period of a periodic intensity modulation is called *intensity modulation period (sample rotation?)*: the
  data do not tell its source. The model's own notes and reasons are shown in the interface language.

### Analyze

- **Curves** has a **Log q** option alongside Log I. Plot selections keep their original
  data coordinates when the axis scale changes.
- **Geometry…** and **Enter Coordinates…** stay open while applying the beam centre.
  Coordinates show four decimal places; **Centre step** sets the remembered increment for
  arrow buttons, keys and the wheel. **Live Preview** refreshes the curves as the centre changes,
  combines rapid adjustments and preserves the curve zoom.
- **Make Left and Right Symmetric** finds the horizontal axis with a global coarse search,
  then refines it using the displayed column mean and log-distance weighting so low-q structure
  keeps its influence. Masked pairs stay fixed during refinement. Only x changes;
  0.0001 px is the numerical stopping tolerance, not a measured uncertainty.

### Fitting

- **Single analysis** now has **fit1d (network + physical fit)** in its Fit step. Add 1–4
  Sphere / Vertical cylinder particles in Model, then fit all their populations together.
  **General network** uses `cond_v2_ft4`; **Au nanoparticles on Si (adapted)** uses
  `cond_v2_ftr1`. Search defaults to **Auto**. Set GISAXS / SAXS and, for a Yoneda cut,
  enter qz in nm⁻¹; unset qz means zero with a warning. This route uses an isolated process,
  supports Stop and needs the installed kit, without V5/TensorFlow model assets.
  Results, plots and CSV/JSON keep the original v2 total curve, component curves, effective σ,
  parameter errors, states and warnings. Changed Model, curve selection or fit1d settings mark
  the saved fit stale and prevent exporting it as a current fit. Model values, ranges and fixed
  parameters are not passed to fit1d: only the composition is used; the mapped manual Model
  is a starting point for later refinement with its different physics.
- **1D Predict** has a fourth fit method, **fit1d (network + physical fit)**: an external kit
  (`gisaxs1d_gui_kit`, put in `utils/gisaxs1d_gui_kit` or named by the environment variable
  `GISAXS1D_KIT_HOME`; numpy and scipy only, no TensorFlow). It fits spheres and vertical cylinders
  of the composition you give — the GUI route does not expose random cylinders, and it does not choose the composition
  for you. **Parameters…** has its search (fast / auto / robust), curve type (GISAXS / SAXS),
  network, the qz of the cut and a systematic error. Its physics is its own (number-weighted
  sizes, Hosemann paracrystal, spheres at |q| = √(q∥² + qz²)): the curve shown is the kit's, and
  the row says which parameters have another equally good solution, sit at a bound or are not
  pinned down. Selecting a fit1d row keeps its original v2 curve and components visible while
  copying a manual starting model. Without the kit the method says where to put the folder.

### NeXus / HDF5 files

- Every NeXus/HDF5 suffix is read: `.nxs`, `.nx`, `.nx5`, `.h5`, `.hdf5`, `.hdf`, and every frame of
  a file. The content decides what a source is, a file name only proposes: one frame per file, a whole
  series in one file, one file per detector module (P03 Lambda 9M: `…_m01.nxs` … `…_m11.nxs` are a
  set only when they hold the same dataset and module shape at different module positions; they are
  stitched), or a master with its data files (Eiger). A folder lists each source once.
- Analyze's Data step says what a source is (“11 module files · 240 frames · stitched”), how its
  frames are oriented, and what is left out: a module file that is missing or holds fewer frames
  (the frames every module has are used), a data file that is missing, other image datasets in the
  file. Asking for a frame beyond the end is an error, never the last frame instead.
- **Orientation of non-Lambda files changed.** Files written by the Lambda server (P03) are read as
  before: the modules stitched by their translations, then transposed and flipped. Every other
  NeXus/HDF5 image is now read as stored, row 0 on top, like the CBF or TIFF of the same detector;
  before it was transposed and flipped as well. A geometry or a mask made on such a file before no
  longer fits it: check it (a square frame still matches its old instrument profile by size). GIMaP's
  own HDF5 frame exports read back unchanged; they now mark `/entry/data` as NXdata with `signal="data"`.
- Header values are read in their units: an exposure in ms (as the Lambda files store it) becomes
  seconds, energies in eV or keV, lengths in mm, cm or m; a value in a unit GIMaP does not know is
  left out rather than guessed. In unsigned integer data the largest value of the type means “no
  data” and is left out.
- Files compressed with bitshuffle or LZ4 need `hdf5plugin`, now in `requirements.txt`. A file
  still being written is read as far as it is and read again when it changes; a file another
  program holds is said so; a file left open by a writer that ended says how to read it
  (`h5clear -s` on a copy).

### Analyze: series

- A frame bar (“Frame [4] of 12”) above the steps for a source of several frames.
- **One cut for the whole series.** While the cut is automatic, every frame of a Series map, a Batch
  Export, a Send Series to Fitting and a watched folder's export is cut where the frame on screen is
  (GISAXS: the Yoneda rows and the beam-centre column; GIWAXS: the ring of I(χ)). *Find the cut again
  in every frame* (Series ▸ Options, and the Batch Export dialog) turns this off. The sum that Run
  Automatic Analysis sets is no longer used for series work.
- **Series map**: drawn at the true x, without bridging gaps (the exports keep the measured x); the
  frame axis counts real frame numbers from 1 (the frame in its file for one multi-frame file) or
  time (time stamps, or frame × exposure); a frame range under *Options*; for the GISAXS horizontal
  cut a choice of halves (± / + / − / |x| / mean); a title that says the curve, the frames and the
  cut; **Wide**. Exports are named `<series>_<curve>[_<halves>]_…` with a JSON record of the
  settings; the frame picked is saved under the name Batch Export gives that frame.
- **Watch** follows a source that grows: a file or a set of module files written frame by frame is
  listed at once, and each look (every 2 s) takes the frames it already held at the look before. A
  set of module files is exported once it has rested about 10 s. F5 and the watch read a frame again
  when its files changed; a master and its data files are one row; a source taken off the list comes
  back when it grows.

### Batch Export and Send Series to Fitting

- File names keep the 0-based frame index (`_frame0003`); headers, labels and titles say the frame
  from 1 (“frame 4 of 12”). A set of module files is named by its scan, without `_m01`. Every curve
  file says in `#` lines which frame, which sum and which files it comes from.
- The dialog checks the folder before anything is reduced, names the cut every frame gets (and waits
  while the frame on screen is read again without the automatic analysis' sum), and asks before it
  writes over an earlier export of the same data: **Replace**, **Keep Both in a New Subfolder** or
  **Cancel**. Enter answers Replace only when this export writes again every kind of file Replace
  would remove; otherwise Keep Both, and the question names what Replace would remove.
- GISAXS: **Halves of the cut** — the mean of both halves, the q > 0 half or the q < 0 half — for the
  model fit; the curve for Fitting still holds both halves and names the choice. The model table has
  the columns `halves` and `points`; a frame without counts or without points on the chosen half is
  listed with its reason, and a batch whose model fits failed no longer ends as a success.
- **Send Series to Fitting** writes into a folder of its own, `<name>_fitting/`, beside any Batch
  Export of the same data.
- Tables of all frames: frames whose points differ only slightly are matched row by row; grids that
  really differ are interpolated, never across a gap. `<name>_batch.json` lists every file written,
  the cut of every frame and every failed frame with its reason.

### Fitting: the halves of the cut

- **In-situ series** has its own **Halves of the cut**: the mean of both halves, the q > 0 half or
  the q < 0 half — never both together. It starts on what Analyze sent the curves with, else on the
  halves of Single analysis, else on the mean, and says where its value comes from. A frame without
  points on the chosen half fails with “No points on the q > 0 half” (or q < 0); it is never fitted on
  the other half instead.
- Its results table has one row per frame with **Points** and **Halves**; the saved table adds
  `points` and `halves_used`, and the JSON record (`gimap_series_fit_v2`) the halves and where they
  came from, every frame's state and reason, and the poor fits (χ²ᵣ above 10).
- A second Start, other curves or opening a project ask before results that are not saved are
  dropped (**Save**, **Discard**, **Cancel**).
- **Single analysis** keeps its four choices. *Find the particle shape*, *AI proposal* and *fit1d*
  cannot fit both halves on |q|: they switch to the mean of both halves and say so. A mean never
  interpolates across a gap of the other half. *Find the particle shape* and the model fit of Batch
  Export take a cut of any length (more than 1000 points are merged).
- A GIWAXS I(q) sent to Fitting gets a note: Fitting's models are particle form factors, not peaks.

### Command line and tool windows

- **Format Converter** reads every NeXus/HDF5 suffix and lists a set of module files once, stitched
  and named by its scan; the metadata in a CBF header is written as ASCII JSON.
- **XRR Series Extractor**: **Folder…** chooses the folder of a series; a series of NeXus files that
  hold one frame each is read; the JSON record lists every file read (`source_layout`).
- **Geometry Calibration**: a multi-frame file is calibrated on one frame or on the sum of all
  frames; importing a calibration made on summed frames reads that sum and selects its standard.
- `tools/gimap_agent.py auto`: a set of module files is analysed once, in a folder named by its scan;
  `--series` takes files that hold one frame each as one series; a source with a part missing is an
  open item of its report. The default sum of a series is now the last fifth of its frames, at most
  10, and the last frame alone below ten frames (was: the last 10 frames).

### Exported text

- Text files are UTF-8 without a byte-order mark. GIMaP's own symbols are written in ASCII (`1/A`,
  `deg`, `2theta`, `alpha_i`, `sigma`, `+/-`); file and frame names and paths are kept as they are.
  A script should read them with `encoding="utf-8"`.

### Fixed

- **Odd frames** (Analyze ▸ Series, Fitting ▸ In-situ series, Compare): a frame that differs from its
  neighbours in only a few points — such as a few hot detector columns in one frame — is now marked odd
  ("probably the detector"), even when it is too small to show over the whole curve. Before, such a frame
  could become a stage of its own and enter Compare's end state. Two or three such frames in a row, or two at
  either end of a series, are each found; real changes of the sample, however sudden, are still not odd.
- **Single analysis** and **In-situ series** read a curve file in the unit of q its header states, as 1D
  Predict does. The setting under **Curve ▸ File** is now **q unit of files without one in their header** and
  applies only to those files; In-situ series takes it at Start (before, it read every file in the unit of
  Single analysis' curve). The Curve step says the unit and where it came from, and the In-situ Curves step
  counts the curves by unit. A file whose header states both units is neither opened nor fitted. The fit's
  record (`curve.q_unit_from`), the series record (`per_frame.q_unit`, `q_unit_from`) and projects
  (`unit_setting`) name it. A project saved before opens with the unit it saved, unless the file's header
  states another one: the stated unit is then used, and the project's notes say so.
- **1D Predict** reads a curve file in the unit of q its header states. Analyze's `_fit_input.dat` (q in Å⁻¹)
  used to be read in the window's unit setting, nm⁻¹ by default, so q was ten times too small. The setting is
  now **q unit of files without one in their header**; the window lists each file's unit before the run and in
  the table, `request.json` and `fitting_run_summary.json` record it (`q_units`), and a file whose header
  states both units is not fitted.
- macOS: Batch Export with pictures ended the program (worker threads now have a 16 MB stack).
- Send to Compare opened a Compare page that stayed at “Comparing …”.
- Fitting ▸ In-situ series: refilling the results table of a long series froze the window.
- Peak fits with a Gaussian or Lorentzian profile no longer fit η (it is exactly 0 or 1): they
  converge, with tens of model evaluations instead of thousands.
- Saved pictures of square or tall frames no longer cut off the y-axis label; the colour bar is as
  tall as the frame.
- Trainset's `.h5` reference image is the detector image, not its pixel mask.

## 0.1.0 — 2026-10-13 (pre-release)

The first version since 0.0.2 (2026-07-16). Most of the program is new. The detector work, the
fitting and the comparison of series each have their own workspace, three pages of 0.0.2 are gone
(see *Removed and replaced*), and the settings moved out of the program folder (see *Updating from
0.0.2*).

### Analyze (new workspace)

- One workspace for GISAXS and GIWAXS. Open CBF, TIFF, EDF or NeXus frames or a whole folder
  (multi-module NeXus series are stitched) and the reduced curves appear without further clicks.
  The steps on the left — Data, Geometry, Mask & corrections, Cuts, Results, Export — each show
  their state and one line of what they found.
- **Geometry** is kept as an *instrument profile* of the detector (matched by detector name and
  frame size), so it is set once: **Find Calibration Automatically**, **Enter Geometry** or
  **Calibrate…**. q is exact for grazing incidence, in Å⁻¹; the beam centre is the direct beam.
  A frame whose detector reaches beyond 2θ = 20° is reduced as GIWAXS, otherwise as GISAXS; the
  mode can be forced.
- **Masks and corrections**: detector gaps, and isolated hot and dead pixels found in every frame,
  are left out automatically (the step and the export record say how many); rectangles and
  polygons drawn on the image, mask files, a gap guard, a valid intensity range, a background
  frame, and for GIWAXS filling gaps from the mirror side. The GIWAXS intensity corrections (solid
  angle, polarisation, film absorption) are off by default; when on, the errors are propagated and
  the record says what was applied.
- **Cuts**: GISAXS — the horizontal cut at the Yoneda band and the vertical cut at the beam-centre
  column. GIWAXS — I(q), in-plane and out-of-plane sectors, I(χ) of a ring, the q map and the cake;
  your own cuts by clicking a peak (**Ring**, **Sector**, **Spot**), by typing centre ± half width,
  or by drawing on the cake; cut sets can be saved and loaded.
- **Sources** shows on the image where every curve comes from. Curves with a signed x can be shown
  as both halves, one half, or folded onto |x|.
- **Series**: **Build Map** reduces every frame of a list, a folder or a multi-frame NeXus file and
  shows a curve as a map against frame and q, with the curve of a frame, the intensity or a peak
  (position, FWHM, area, height) in a q window against frame, and export of the map and the tables.
- **Batch Export**: one dialog from raw frames to a folder — tables per curve, per-frame curves,
  the input for Fitting, q maps and cakes (CSV, TXT or DAT), pictures (PNG, TIFF, SVG, PDF),
  converted detector frames (TIFF, EDF, NumPy, HDF5) and optional fitting of every frame. Several
  frames are reduced at once in low-priority processes (Gentle / Balanced / Fast), with progress,
  **Pause** and **Stop**; the folder gets a `README.txt` and a JSON record of the settings.
- **Export**: every curve file is written with a `<name>_analysis.json` record (source, geometry,
  corrections, pixels left out); the set-up of Analyze can be saved to a settings file and loaded
  for the next data (**Save Settings…** / **Load Settings…**).
- **Undo / Redo** of every change of the set-up, colour limits that stay where you put them
  (**Levels**), a switch for every kind of mark on the images (**Marks**), and box zoom on every
  image and plot.
- **Send to Fitting** opens the horizontal cut in Fitting; **Send Series to Fitting…** exports every
  frame and opens the series there.

### Fitting (new page)

- Four steps — **Curve → Model → Fit → Results** — laid out like Analyze, with the curve, the
  model, its terms and the residuals on the right.
- **Model**: one card per particle (sphere, random cylinder, vertical cylinder) with an optional
  distance D, a background and a resolution peak; every value can be fitted or fixed and has a
  range. Size spreads are relative everywhere (σR/R, σh/h, σD/D).
- **Fit**: one button, four methods — **Refine the current values**, **Search the ranges, then
  refine**, **Find the particle shape (no AI)** and **AI proposal (1D Predict)** — with progress
  and **Stop** (the best values so far are kept).
- **Results**: χ²ᵣ, every value ± its 1σ error, and warnings when a value stopped at a bound, when
  values are strongly correlated or when χ²ᵣ is well above 1. **Save** writes the data and fit
  (CSV with a JSON record), the plot and the model.
- Points can be left out of the fit by clicking them on the plot (**Exclude**); the model has
  **Undo / Redo**; the fitting range is dragged on the plot.
- **In-situ series**: every curve of a folder fitted with the model of Single analysis, each frame
  starting from the previous result or from the same model, with **Pause** and **Stop**, a table
  and a trend plot of every value with its error, and new curves picked up while a measurement
  runs (**Watch for new curves**).
- Fitting works in nm⁻¹ and says so; the curve files Analyze writes (`<name>_fit_input.dat`) are in
  Å⁻¹ and carry where the curve came from.

### Compare and stages (new, replaces Classification)

- **Stages**: once a Series map is built, Analyze marks where the curves change course (stages)
  and which frames do not belong (odd frames), says what grows or falls between stages, and can
  leave the odd frames out of Batch Export. Fitting's In-situ series uses the same stages.
- **Compare**: several series or samples side by side — from Analyze's Series map, a folder of
  curve files, or curve files — with how far each has changed, their paths through the main
  changes, their end states, groups of similar series and a table of how different they are.
  Tables and plots can be saved. No labels and no AI are needed.

### Automatic analysis and AI (new)

- **Run Automatic Analysis** needs no AI. GIWAXS: geometry, the last frames of a series summed,
  peaks with positions, widths and their errors, in-plane against out-of-plane, the orientation of
  the strongest reliable rings and crystallite sizes. GISAXS: the horizontal cut at the Yoneda
  band, the beam-centre column from the symmetry of the cut, the halves, the in-plane distance
  2π/q* and a numerical fit of I(qy). Every default decision is listed with its reason, questions
  only you can answer become fields, and the result can be saved as a report.
- **Process with AI…** / **Ask AI…** (optional): an AI works on the open frame with the same tools
  while you watch. Its changes to the set-up come back as cards to preview, apply or undo, and
  follow-up questions can be asked. It runs on your Claude plan through a local Claude Code (no
  API key), on the Claude API, or on another provider with an OpenAI-compatible API (DeepSeek,
  Qwen, OpenAI, a local model …).

### Tools (new windows in the Tools menu)

- **Geometry Calibration**: beam centre and detector distance from an image of a standard (NXS,
  CBF, TIFF or EDF), with the fitted rings drawn over the image, manual refinement, and export and
  import of the result. **Apply** saves the geometry as the instrument profile Analyze uses. The
  window shows and accepts the beam centre in the same pixel coordinates as Analyze (the centre of
  the first pixel is at 0.5), so a centre can be compared or copied between the two as it is. The
  exported record names the GIMaP version that wrote it.
- **Format Converter**: NXS, CBF and TIFF frames to TIFF, CBF, HDF5 or NumPy, with frame selection,
  a preview and a report; **Convert Current File…** starts it with the file shown.
- **XRR Series Extractor**: an XRR curve from a detector angle series (NXS modules or CBF files),
  with a CSV and a JSON record of every value that determines qz. The window shows and accepts
  the direct-beam centre in the pixel coordinates of Analyze and Geometry Calibration; the
  exported CSV and JSON record count pixel centres from 0 (0.5 px lower), and the record says so
  (`center_convention`).
- **1D Predict — Fit Many Curves…**: the experimental neural model for a list of curve files. It
  replaces the AI-assisted fitting of 0.0.2; each run writes its own folder
  `AI_Fitting_Output/v5_<date>_<time>/`. The model and its trained weights come with the portable
  version and with the Git repository, so 1D Predict also runs from a source checkout.
- **Settings**: appearance, defaults of Analyze, the user data folder and the AI set-up.

### Labs

- **2D Prediction** (was *GIMaP Predict*): a three-step page — import data, import model, predict.
  A prediction runs in the background and can be stopped; results are exported with a JSON record
  of the module, the model and the preprocessing. Input is CBF only.
- **Trainset Build** (was not implemented in 0.0.2): five steps from the design of a simulated
  GISAXS data set to a local run and its results. Submission to the Maxwell cluster is disabled.

### Interface

- A **Start** page: drop data or open it, pick a recent file, or choose what you want to know.
- **Projects** (`.gimap`): File ▸ Save Project keeps a sample as it was left — Analyze, Fitting,
  the In-situ series and Compare — and reopens it; data are referenced by path.
- Detector frames, folders and projects can be dropped anywhere on the window; curve files on the
  Fitting page.
- English or Chinese (View ▸ Language), a light and a dark theme, adjustable font size; the
  interface follows the display scaling of Windows.
- The recent data are remembered, and the set-up of the last session is offered (never applied
  without asking).
- Long jobs show progress and can be stopped; closing GIMaP while something runs asks first.
- An unexpected error no longer closes the program: a window says what happened and the error is
  written to `logs/errors.log` in the user data folder.
- Help ▸ User Manual (`F1`) opens the manual in your browser.

### Command line and agents (new)

- `tools/gimap_agent.py auto <frames> --notes "…" --out <folder>` runs the automatic analysis
  (GIWAXS or GISAXS) without the window and writes a report with every decision and its reason;
  the exit code says whether something needs an answer (`2`) or a frame failed (`1`).
- `status`, `find-calibration`, `tools` and `call` run single tools; `mcp` serves them to agents
  such as Codex or Claude Code. Nothing is written next to the data.
- `docs/agents/giwaxs-playbook.md` and `gisaxs-playbook.md` say what the automatic procedure
  decides by default and where that may not fit a question.
- `tools/eval_giwaxs_agent.py` checks the automatic procedure (`baseline`) or an agent's written
  report (`report`) against a GIWAXS case: a JSON file you write for a measurement you know well.
  Cases are read from `docs/agents/eval/local/`, a folder Git ignores, so a case on your own data
  stays on your computer; the repository ships none. Reports saved by Windows tools are read
  (UTF-8 with a byte order mark, UTF-16). When there is no case, or a case or a report cannot be
  used — another encoding, an empty report, a malformed case — the tool says why in one sentence
  and ends with exit code `2`.

### Removed and replaced

| In 0.0.2 | In 0.1.0 |
|---|---|
| Cut & Fitting page | **Analyze** for the detector image, the geometry and the cut; **Fitting** for the curve |
| WAXS / GIWAXS page | Analyze, GIWAXS mode |
| Classification page | Stages in Analyze ▸ Series, and **Compare** |
| AI-assisted fitting, results in `AI_Fitting_Output/current_prediction` | 1D Predict, results in `AI_Fitting_Output/v5_<date>_<time>/` |
| GIMaP Predict page | Labs ▸ 2D Prediction |

The packages `umap-learn`, `scikit-learn` and `tqdm` are no longer needed. `pyqtgraph` is new and
required; `anthropic` and `openai` (Process with AI) and `psutil` (Batch Export) are new and optional.

### Fixed before the release

These were found while 0.1.0 was checked; they matter to those who ran a development copy.

- A geometry saved after the beam centre had been moved — typed in the geometry dialog,
  calibrated, or found automatically — is now the one that is used. Before, the centre set
  earlier in the session silently stayed in force.
- Opening a project replaces what Fitting holds as well: a project without a Fitting curve or an
  In-situ series no longer keeps those of the previous sample, and Save Project no longer writes
  them into the opened project.
- Run Automatic Analysis with no frame open now says “Open a frame first.” instead of leaving a
  warning on the first frame opened afterwards.
- A plain confirmation note of one page no longer stays over the buttons of the page you switch
  to; warnings and notes with a button stay.
- Geometry Calibration showed the beam centre half a pixel lower in x and y than Analyze showed
  for the same profile. Both now use the same pixel coordinates. The geometry that is saved did
  not change: profiles saved before are still right. An exported calibration file keeps its
  numbers too (its centres count pixel centres from 0) and now says which convention they use.
- Geometry Calibration no longer fills **Estimated distance** with a distance nobody measured.
  Before, a new installation put the built-in 2000 mm into the field when an image was opened,
  and the search prefers solutions near that value: on a silver behenate image taken at about
  1630 mm, with silver behenate chosen, a solution near 1830 mm came first. Now the field is
  filled only with the distance the image file states, or else with the distance of a
  calibration applied before; with neither it shows *Optional* and the search gets no estimate.
  A distance the field cannot show (more than 100 m) is no longer turned into 100000.00 mm.
  Settings carried over from an earlier copy can lack the record of their last **Apply**: the
  field then shows *Optional* until **Apply** is pressed once in this version.
- XRR Series Extractor showed and took the direct-beam centre half a pixel lower in x and y than
  Analyze and Geometry Calibration (it counted pixel centres from 0). Its centre fields, **Pick
  direct-beam center**, the marks on the preview and the ROI x / ROI y columns now use the pixel
  coordinates of the other two, so a centre is copied between the three as it is. A centre you
  typed by hand in an earlier copy is typed 0.5 px larger in x and y now to name the same point;
  the centre taken from the last calibration needs nothing. A centre picked with the mouse is
  now the point clicked: before, the extraction used a point half a pixel to the right of and
  below it, which mattered for a ROI of one or a few pixels. The exported CSV and JSON record
  keep their numbers (pixel centres counted from 0); the record now says which convention they
  use.
- Fitting starts a new user with a neutral model. Before, a source checkout started with the
  values left by an earlier fit.
- A vertical cylinder taken over from a solution selected in the 1D Predict window is saved with
  its own values. Before, `model_parameters.json` in the user data folder — and a file written
  with **File ▸ Labs Parameters ▸ Save As…** — kept the radius and distance that were there and
  gained a second block named `vertical cylinder`. The Fitting pages showed the right values,
  and spheres and random cylinders were saved correctly. A file with the extra block still
  loads: `vertical_cylinder` is the block that is used, and the extra one leaves the file the
  next time a value of a vertical cylinder is saved.
- 1D Predict runs from a source checkout: the trained weights are in the Git repository now.
- Automatic analysis and the command line use a calibration the notes name by its file name
  alone (`calibration LaB6_take2.poni`): the file of that name in the folder of the frame, or in
  a folder the notes name. Before, only a full path counted as named, and such a file was left
  to the automatic search, which could prefer the calibration of a neighbouring data set.

### Known limitations

- **Fitting does not read `.csv` curves**, although Analyze exports its curves as CSV. Use
  **Send to Fitting**, or open the `<name>_fit_input.dat` that Analyze writes; `.dat` and `.txt`
  files with columns `q I [σ]` open too.
- **Geometry Calibration can put a wrong solution first.** Seen on a silver behenate image whose
  path does not name the standard. With the standard left on “Auto Detect” and no estimated
  distance, a LaB₆ solution at a much shorter distance came first, and the silver behenate
  solution was last in the list of candidates. A wrong value in **Estimated distance** misleads
  too, also with the right standard: with silver behenate chosen and 2000 mm typed in, a
  solution near 1830 mm came first for a real distance of about 1630 mm. The field is no longer
  filled with a built-in value, but a number can stand in it without your typing: each time an
  image is opened it takes the distance the image file states, or else the distance of the
  calibration applied last, which may belong to another set-up. Choose the standard yourself
  when you know it. Leave **Estimated distance** on *Optional* (0) when you do not know the
  distance, enter the approximate distance of your set-up when you do, and set a number you did
  not type back to 0 unless it is that distance. Before you press **Apply**, check that the
  distance is the one of your set-up. This will be corrected in a later version.
- **Trainset Build (Labs) is not fully translated**: in the Chinese interface many texts of its
  steps are still English. 2D Prediction has a few such texts too.
- After the interface language is switched in the middle of a session, a few texts keep the old
  language until the next run or restart (for example the outcome line of a finished automatic
  analysis).
- **A calibration found by the automatic search may belong to another experiment.** The search
  looks in the folders around the frame, up to three levels up, and ranks ready calibration
  files by how close in time they were written to the frame. In a folder that was copied or
  downloaded in one go the file times say nothing. The report names the file it used and where
  it was found; check that it is the calibration of your measurement, or name it in the notes or
  give it yourself.
- **XRR Series Extractor can place a beam centre from a NeXus file header wrongly.** It takes the
  two numbers of the header as column and row of the frame it shows, without the turn (transpose
  and flip) the frame gets when it is loaded; Analyze applies that turn to the same header. Such a
  centre is marked *from file*, and it replaces the centre of the last calibration. Check on the
  preview that the cross lies on the direct beam, or use **Pick direct-beam center**. CBF series
  are not affected: their headers are not used for the centre.
- Quitting GIMaP during a Batch Export keeps the files of the frames already done, but the batch
  tables, `README.txt` and the JSON record of the batch are not written. **Stop** writes them.
- 1D Predict writes its run folders into the folder GIMaP was started from (for the portable
  version, the program folder); they are not removed automatically.
- Windows is the platform GIMaP is developed and tested on. TensorFlow runs on the CPU; a GPU
  set-up is not configured.
- **Sputter does not determine the thickness of a continuous film**, and a film with particles on it
  is not modelled. Read the island-volume equivalent t_eq (island windows, conditional) and the nominal
  thickness from a deposition rate; compare with XRR, a quartz balance or a cross-section.
- **Sputter reads Cu with the Ag model** (the networks were trained for Ag and Au): Cu results are for
  reference only.
- **Sputter's first Predict downloads PyTorch** (about 1.2 GB, needs the internet); the model runs on
  the CPU. Without internet, install it by hand (manual, section 13.7).
- **Sputter's literature method rests on the deposition rate and t0 you give** and on D = 2π/q*. The
  percolation interval assumes the rate within ±10% and t0 within ±2 s (±0.5 s when t0 is given), engineering
  choices the page does not change (the toolkit's `physical` command does), and leaves out the systematic error
  of 2π/q*. PNO, PSBMA and PEO have no published percolation data: the band of all polymers of the metal is used.

### Updating from 0.0.2

- From 0.1.0 on, the settings, the instrument profiles, the recent files and the fitting model
  parameters are kept in one user data folder outside the program (`%APPDATA%\GIMaP` on Windows),
  so later updates keep them.
- Settings found in the program's own `config/` folder (a source checkout updated in place) are
  imported once and left untouched. A new portable folder starts with the defaults: the settings
  of a portable 0.0.2 are not carried over.
- Geometry is now an instrument profile in Analyze: calibrate or enter it once for each detector.
- When you run from source, install the requirements again (`python -m pip install -r
  requirements.txt`): the list of packages changed.

**Updating a source checkout.** The history of `main` on GitHub was rewritten on 2026-10-13. A
clone last updated before that day can hold commits that are no longer on GitHub; `git pull`
then stops with an error or tries to merge the old history with the new one. Do not merge, and do not
push from such a clone. Move it to the new `main` instead:

- Once, for a checkout that does not have the script yet: `git fetch origin`, then
  `git reset --hard origin/main` (with `main` checked out). This discards your edits of tracked
  files, and commits you made yourself are no longer on the branch: run `git stash` (edits) or
  `git branch my-work` (commits) first if you want to keep them, and do not merge or push that
  branch afterwards. Files Git does not track (data, results) and your settings in
  `%APPDATA%\GIMaP` stay; only a file of your own at a place where the new version has one — such
  as weight files of 1D Predict copied in by hand — is replaced by the file of the new version.
  One edit you did not make yourself is `config/model_parameters.json`, where 0.0.2 kept the
  model of Fitting: do not bring it back with `git stash pop`, which would put conflict markers
  into the file (`README.md` says what to do instead).
- Then, and from now on: `powershell -ExecutionPolicy Bypass -File tools\update_source.ps1`, run
  in the folder of the checkout. Run it once right after the two commands: on Windows they leave
  some model files of 1D Predict with the line endings of the older checkout, and 1D Predict
  does not start until the script has written them again. For later updates the script keeps the
  state you had as a branch `backup/…` when the history was rewritten, puts your edits of
  tracked files aside and back on top of the new version, leaves files Git does not track alone,
  and says what it did and how to go back; `-DryRun` only reports. `README.md` has the details.
- This update downloads more than earlier ones: the weights of 1D Predict (about 56 MB) are part
  of the repository now.
