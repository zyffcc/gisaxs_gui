# GIMaP User Manual

## 1. Introduction

GIMaP is a PyQt-based desktop GUI for GISAXS/GIWAXS data visualization, analysis, fitting, and machine-learning-assisted workflows. It is intended for users who work with grazing-incidence scattering data and want a graphical workflow for inspecting detector images, extracting 1D curves, fitting physical models, running trained-model predictions, and organizing batch results.

The current software should be treated as early pre-release / beta scientific software. Some workflows are stable enough for daily testing, while others appear experimental or under development.

## 2. Installation

### Windows Installer

* Visit the [GitHub Releases](https://github.com) page.
* Look for the latest version marked with the **Latest** tag.
* Download the `*-setup.exe` (or the provided `.zip` archive).


### Running from Source

Install Python 3.10 or 3.11, then run:

```powershell
cd gisaxs_gui

python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt

python main.py
```

If using Conda:

```powershell
conda create -n GUI python=3.11 -y
conda activate GUI
pip install -r requirements.txt
python main.py
```

The source dependency file is `requirements.txt`. No `environment.yml` file was found.

## 3. Main Interface

The main window has a sidebar on the left, the active workspace in the middle, a menu bar
(**File, View, Tools, Help**); messages appear as short notes at the bottom right.

The sidebar groups the pages into **Workspaces** (Start, Analyze, Fitting, Compare) and **Labs**
(2D Prediction, Trainset Build — the machine-learning tools). `« Collapse`
(or **View ▸ Collapse Sidebar**, `Ctrl+B`) reduces it to icons; `Ctrl+1` … switch pages.

- **Start** (the first page): drop detector images or a folder, or open them — Analyze opens at
  once and shows the file, its detector and frames while the image loads. Or choose what you want
  to know: *Crystals and orientation* (GIWAXS), *Nanostructure* (GISAXS), *In-situ or batch series*,
  *Calibrate the detector* — or describe your data and question to the AI (**Ask the AI…**).
- **Analyze**: one workspace for GISAXS and GIWAXS. The command bar holds the file, the mode
  (Auto / GISAXS / GIWAXS), αi, **Run Automatic Analysis**, **Ask AI…**, **Export** and **Send to
  Fitting**. On the left the steps — ① Data ② Geometry ③ Mask & corrections ④ Cuts ⑤ Results
  ⑥ Export — each with its state and one line of what it found; the image in the middle
  (Detector / q map, **Sources** shows where every curve comes from — click a curve to see only
  its region); on the right **Curves**, **Results** and **Series**.
  - *Run Automatic Analysis* needs no AI. GIWAXS: geometry, the last frames of a series summed,
    peaks, in-plane / out-of-plane, ring orientation and crystallite sizes. GISAXS: the horizontal
    cut at the Yoneda band, the beam-centre column moved to the symmetry axis, the halves (averaged
    when both agree, otherwise the usable one — with the reason), the in-plane distance 2π/q*
    (a shoulder is reported as a hint) and a physical fit of I(qy) (spheres, vertical and random
    cylinders with size spread and distance D). When several models fit equally well, Results asks
    which one you expect and names the one that agrees with the measured spacing. Save the fitted
    curve, the fit table or the report; **Refine in Fitting** opens the prepared curve in Fitting.
    **Fit details** (under the peak table or the fit table, closed until you open it) follow the
    selected row: for a peak, the points it was fitted to with the Gaussian and its local background,
    q, d and FWHM with their errors, height, area, background and slope, S/N, χ²ᵣ, the fit window,
    the Scherrer size, how peaks are found, and where the ring lies on the q map; for a model
    solution, every parameter (R, σR/R, h, D, σD/D, weight, amplitude, background, resolution), χ²,
    convergence, evaluations and time, the q range, the start of D and the algorithm, its warnings,
    and **Show in Fitting**, which opens the curve in Fitting with that solution drawn (Export Data…
    there saves it with its parameters).
    Questions only you can answer (αi, the energy, the calibration) become fields: fill them in and
    run again.
  - *Mask & corrections*: detector gaps, and isolated hot and dead pixels found in every frame, are
    left out automatically (the step says how many; **Show Masked Pixels** circles them; the option
    can be turned off). Negative values of floating-point (dark-subtracted) frames are kept as data.
  - *Series*: with several files, a folder or a multi-frame NeXus file, **Build Map** reduces every
    frame with the current settings (with progress and Cancel) and shows the chosen curve as an
    intensity map against frame and q, with adjustable colour scale. Drag the horizontal band to
    pick a frame (its curve is drawn below, **Open** shows it in Analyze) and the vertical band to
    pick a q window (its intensity against frame is drawn below). Export the map as a CSV table or
    a figure. The map's stages and odd frames are found at once (section 7); **Send to Compare** puts the
    series beside others.
  - The geometry comes from an *instrument profile* matched by detector name and frame size; the
    first time, **Find Calibration Automatically**, enter it, or run Geometry Calibration once.
    See `docs/ui/workspaces/analyze.md`.
- **Fitting**: fit 1D curves — one curve, or an in-situ series of curves. See section 4.
- **Compare**: runs or samples side by side, with their stages and odd frames. See section 7.
- **2D Prediction**, **Trainset Build**: Labs (sections 5–6).

The former WAXS page has been retired: every WAXS/GIWAXS function (background, masks, sectors,
q-range and circle cuts, integration axis, image export, batch with calibration/normalization)
is in Analyze's GIWAXS mode. The detector half of the former Cut & Fitting page is in Analyze too.

### Menus

| Menu | Entries |
|---|---|
| File | Open Data… (`Ctrl+O`), Open Folder… (`Ctrl+Shift+O`), Open Recent, Open Project… (`Ctrl+Shift+P`), Save Project (`Ctrl+S`), Save Project As… (`Ctrl+Shift+S`), Labs Parameters ▸ Load… / Save As…, Exit |
| View | the workspaces (`Ctrl+1`…), Collapse Sidebar (`Ctrl+B`), Full Screen, Theme ▸ Light / Dark, Font Size ▸ Larger (`Ctrl++`) / Smaller (`Ctrl+-`) / Reset (`Ctrl+0`) |
| Tools | Geometry Calibration… (`Ctrl+Shift+G`), Format Converter… (`Ctrl+Shift+C`), Convert Current File…, XRR Series Extractor… (`Ctrl+Shift+R`), Fit Settings & Batch…, Process with AI… (`Ctrl+Shift+L`), Settings… (`Ctrl+,`) |
| Help | User Manual (`F1`), GitHub Repository, Open User Data Folder, About GIMaP |

*Labs parameters* are the settings of 2D Prediction and Trainset Build; Analyze keeps
its own set-up in settings files (**Export ▸ Save Settings… / Load Settings…**, also used by Batch
Export), and a project keeps everything of one sample (below).

### Projects

A project (`.gimap`, a JSON file) reopens a sample as it was left: the frames and the whole set-up of
Analyze (mode, profile, masks and drawn regions, corrections, cuts), the curve, halves, fitting range,
left-out points and model of Fitting, and the folder and choices of the In-situ series. **File ▸ Save
Project** (`Ctrl+S`) writes it (the first time it asks where, next to the first frame by default);
**Open Project…**, the Start page's **Open Project…**, **Open Recent** (projects first) or dropping a
`.gimap` file on the window opens it. The window title shows the project's name. Data are referenced
by path, not copied: when a file has moved, the project still opens and a message lists what is
missing.

### Memory between sessions

- **Recent data**: every file or folder opened (dialog, drag and drop, Start page, Batch Export) is
  listed in **File ▸ Open Recent** and on the Start page, newest first (at most 8; files that no
  longer exist are left out; *Clear the List* empties it).
- **Last set-up**: when GIMaP closes after a frame was analysed, the Analyze set-up (mode, profile,
  masks, corrections, cut regions …) is kept in `last_analyze_setup.json` in the user data folder.
  With the first frame of the next session a message offers **Use It** — it is never applied without
  asking, since a new beamtime may need other settings.

### When something goes wrong

An unexpected error does not close GIMaP: a window says what happened (the traceback behind
**Show Details**, **Copy Details** for a report) and the work goes on. Every error is written to
`logs/errors.log` in the user data folder (**Open Log Folder**). Files that cannot be read say why in
words — empty (still being written?), not a detector image, a damaged or incomplete NeXus file, no
permission — instead of the reader library's message.

### Settings and user data

**Tools ▸ Settings…** has four pages; every change applies immediately.

- **Appearance**: light or dark theme, the font size and the interface language (English by
  default, or 中文; values, units and file names are never translated). The interface follows the
  display scaling of the operating system (for example 125 % or 150 % in Windows), so it looks the
  same on every monitor, including low-resolution screens.
- **Analyze**: whether a beam centre written in the file header (CBF `Beam_xy`, NeXus
  `beam_center_x/y`) replaces the calibrated profile centre — off by default, because many
  beamlines never update it — and which half of the horizontal cut *Send to Fitting* uses.
- **Data**: the user data folder and *Reset All Settings…*.
- **Assistant**: the brain (Claude Code on your Claude plan, the Claude API, or another AI
  provider: DeepSeek, Qwen, OpenAI, Kimi, GLM, Gemini, OpenRouter, SiliconFlow, Azure, Ollama …),
  standing instructions added to every run (e.g. where calibrations are kept, the beamline energy),
  Claude Code's program, model and sign-in, the API models and keys, effort, turn limit and default
  permission (see *Process with AI* in section 8).

All settings, preferences, instrument profiles, the session and the fitting model parameters are
kept in one user data folder outside the program: `%APPDATA%\GIMaP` on Windows (`~/.config/gimap`
elsewhere, or the folder in the `GIMAP_HOME` environment variable). Updating the program never
overwrites them. Files from older versions (`config/user_parameters.json`,
`config/user_settings.json`, `config/instrument_profiles.json`) are imported once and left
untouched.

## 4. Fitting Page

The Fitting page fits 1D curves with particle models: one curve at a time (**Single analysis**)
or a whole series of curves from an in-situ experiment (**In-situ series**). Switching between the
two keeps each side's curve, parameters, Recipe and results.

The detector work of the former Cut & Fitting page — opening images, the beam centre and Yoneda
band, the cut region, masks around detector gaps, summing frames — now happens in **Analyze**,
which writes the curve that Fitting fits:

| Former Cut & Fitting control | Now |
|---|---|
| Import image, Previous / Next | Analyze file list |
| Detector parameters | Analyze instrument profile (Geometry…, Calibrate…) |
| Find Yoneda & Set Cut, cut thickness | Analyze's automatic horizontal cut (drag the band or enter rows) |
| Optimize Center X | Analyze ▸ Beam centre ▸ Refine x by Symmetry |
| Mask negative pixels + guard | Analyze ▸ Options ▸ Gap guard (3 px by default) |
| Stack | Analyze ▸ Options ▸ Sum N frames |
| Cut region width | Fitting's fitting range (below) |
| In-situ CBF / NXS series | Analyze ▸ Send to Fitting ▸ Send Series to Fitting… |

### Single analysis: one curve in four steps

Laid out like Analyze: the steps on the left, each saying where it stands; the plot on the right —
the points (paler outside the fitting range), the model over the whole curve, with **Terms** each of
its parts (every particle, the background, the resolution peak), the **orange band** that is the
fitting range (drag it) — and below it the residuals, (I − model)/σ (ln(I/model) when the file has no σ).
The command bar has **Open Curve…** (its menu: **Load Model…**), **Undo / Redo** of the model (every fit
is one step), **Fit** (Ctrl+Return) and **Save** (data and fit, plot, model).

1. **Curve** — the file, its points and q range in nm⁻¹ and Å⁻¹, and whether σ is in the file (the fit
   weights each point by 1/σ; without σ every point gets the same relative weight). For a cut through
   the beam: **Halves of the cut** — their mean (where both exist; beyond, the longer half alone), both
   on |q|, or one half. **Fitting range** in nm⁻¹ (or drag the band; **Whole Curve**). Under *File*: the
   q unit of the file (Å⁻¹ for Analyze's curves). **Leaving points out**: switch on **Exclude** above the
   plot, then click a point to leave it out of the fit (click again to take it back) or drag a box around
   several; they are drawn as red ×, the Curve step says how many, and **Include All** takes them back.
   The left-out points are kept with the curve (and in a project) and written in the fit's record.
2. **Model** — one card per particle: its family (sphere, random cylinder, vertical cylinder), **Distance D**
   on or off (the paracrystal interference), and its values — scale, R, **σR/R**, h and **σh/h** for
   cylinders, D and **σD/D**. Spreads are relative everywhere (as in Analyze's results and 1D Predict).
   **fit** ticked: the fit may change the value; unticked: it stays fixed. **Ranges** shows each value's
   min – max (the fit keeps it inside, *Search the ranges* searches across it; a value typed outside widens
   its range). A card for the background, the resolution peak A/(1 + (|q|/w)^ν) and the factor k (fixed
   at 1 by default). After a fit every value shows its ± error, or *at a bound*.
3. **Fit** — one button, four methods:
   - **Refine the current values** — bounded least squares from the values in Model; stays near them.
   - **Search the ranges, then refine** — differential evolution across the ranges of the free values
     (log scale when a range spans 10× or more), then refinement from the three best starts.
   - **Find the particle shape (no AI)** — the numerical fit of sphere, random and vertical cylinder from
     several starts (as in Analyze's automatic analysis); the solutions go to Results, the best into Model.
     A first curve opens here.
   - **AI proposal (1D Predict)** — the V5 model with its numerical correction; its solutions likewise.
   Progress and **Stop** (the best values so far are kept). The scales (every particle's scale, the
   background, the peak A) are solved exactly at every step, so the search only moves sizes, spreads,
   distances and the peak's shape. *Advanced*: the number of evaluations; **Fit Many Curves…** (1D Predict
   for a list of files).
4. **Results** — χ²ᵣ (with σ) or the variance of ln I, points, free values, converged or not; warnings when a
   value stopped at a bound of its range, when values are strongly correlated (|ρ| ≥ 0.95: the data do not
   separate them) or when χ²ᵣ is well above 1; every value ± its 1σ error (covariance at the solution,
   scaled by χ²ᵣ); the **Solutions** of *Find the particle shape* or 1D Predict with their χ²ᵣ on these
   points (**Use This Solution** puts one into Model; after *Find the particle shape* the best is already
   there, and **Fit** refines it and gives its errors). **Save Data and Fit…** writes q (nm⁻¹ and Å⁻¹), I, σ,
   the model, the residuals and every term as CSV, with a JSON record of the model, the fit (errors,
   bounds, correlations) and the points next to it; **Save Plot…** (PNG or SVG) and **Save Model…** (JSON,
   to load for another curve). The log is folded at the bottom.

The model and the choices are kept for the next session, and the last curve opens again.

**From Analyze.** *Send to Fitting* opens the cut here with the halves chosen in Analyze;
*Results ▸ Fit details ▸ Show in Fitting* puts that solution into Model — sizes and spreads converted,
the scales solved on the solution's own curve — and says how closely this model reproduces it (the same
curve for spheres and random cylinders; Fitting's vertical cylinder weights radii by R⁴, so it is loaded
as a start to refine).

Analyze's curve files, `<name>_fit_input.dat`, have four columns — q (Å⁻¹), intensity, σ and the number
of detector pixels averaged — and a header recording where the curve came from; any `.dat` / `.txt` /
`.csv` file with columns `q I [σ]` opens too.

**In-situ series** fits every curve of a folder with the model of Single analysis (below).

Typical AI fitting outputs are written to:

```text
AI_Fitting_Output/current_prediction
```

The output may include:

- `top20_candidates.json`
- `top20_candidates.csv`
- `best_fit_curves.npz`
- `residuals_top5.npz`
- PNG plots for top candidates and residuals

### In-situ series

Every curve of a folder, fitted with the model of Single analysis — the same steps, command bar and
plots as there.

1. **Curves** — **Choose Folder…**, or in Analyze list the frames and choose **Send to Fitting ▸ Send
   Series to Fitting…** (every frame is exported, optionally summed N at a time, and the series opens
   here with its curves listed). **Files** is the pattern (`*_fit_input.dat` by default; any `.dat` /
   `.txt` curves work), optionally **Also in subfolders**; **Frames** first – last and every n-th (the
   number is the last one in the file name, natural order). **Watch for new curves** keeps fitting
   curves written into the folder while the series runs (an in-situ measurement).
2. **Start** — what every frame uses: the model, the halves, the fitting range and the left-out points of
   Single analysis (**Edit in Single Analysis** to change them). Each frame starts from **the previous
   frame's result** (a slowly changing sample; if a frame did not converge the next starts from Single's
   model again) or **the model in Single analysis**. Method: **Refine (fast)** or **Search the ranges,
   then refine (slow)**. Then **Start** in the command bar; **Pause** and **Stop** while it runs.
3. **Results** — how many frames were fitted, failed or did not converge; a table with every frame's χ²ᵣ
   and each free value ± 1σ (click a row to see that frame's curve and model); a plot of one value
   against the frame, with its ±1σ error bars (or χ²ᵣ); **Open Frame in Single Analysis** takes a frame
   and its fitted model back to Single analysis. **Save ▸ Table of Every Frame…** writes a CSV (frame,
   file, χ²ᵣ, log RMSE, converged, every value and its error) with a JSON record next to it (the start
   model, halves, range, left-out points, method, and why a frame failed); **Trend Plot…** and **Selected
   Frame's Plot…** save the figures.

A curve that cannot be read or fitted is marked ✗ in the list and the series goes on. The folder and
the choices are kept for the next session and in a project.

## 5. GIMaP Predict Page

The GIMaP Predict page runs trained prediction modules on GISAXS data.

### Choose Single File or Multi Files

Use **Single File** mode for one input file and **Multi Files** mode for folder or batch prediction workflows.

### Choose GISAXS File or Folder

Select a GISAXS detector file in single-file mode. In multi-file mode, select a folder or file collection according to the available controls.

### Set Stack / Range / Every

For stacked data or batch processing, configure the stack index, range, and every/step controls to decide which frames or files are processed.

### Select Module

Prediction modules are discovered from the `modules/` directory. Select a module that matches the input file type and intended prediction output.

### Edit Module Configuration

Module configuration is stored in `module.yaml`. The GUI includes controls for viewing or editing module configuration where connected. This feature appears to be experimental or under development in some workflows.

### Import Model

Import or select the trained model associated with the module. The module configuration contains model path information.

### Framework Selection

Module configurations include a `framework` field. Existing repository modules use TensorFlow/Keras-style model loading. Other frameworks should be considered unsupported unless a working module and loader are present.

### Model Loaded Status Indicator

The page shows whether a model is loaded or ready. If prediction is unavailable, first confirm that the model path is valid and the required framework dependency is installed.

### Run Prediction

After selecting input data and loading a model, run prediction from the page controls. Prediction output depends on the selected module.

### View Prediction Output

Outputs may include scalar values, parameter vectors, structured prediction results, or 2D prediction displays depending on the module configuration.

### GISAXS Preview Tab

The preview tab shows the selected GISAXS input data before or during prediction.

### Predict-2D Tab

The Predict-2D tab is used for 2D output display when the selected module provides compatible output.

### Export Current Result

Export the current prediction result from the page when a result is available.

### Multi-file Results External Window

Multi-file prediction results can be opened in an external results window for review.

### Export All Multi-file Results

Use the multi-file results window or export controls to save all batch prediction results.

## 6. Trainset Build Page

The Trainset Build page provides controls for generating training data. The controller includes beam parameters, detector parameters, sample parameters, preprocessing parameters, generation settings, output folder/name settings, run, and stop controls.

**This feature is under development.**

## 7. Compare Page and Stages

*Classification* was replaced (2026-10-01) by what a series of scattering frames actually needs — no labels and
no AI: which frames do not belong, where the series changes course, how fast, and which runs or samples are
alike. Everything uses Analyze's curves (the same geometry, masks and cuts).

**Stages in Analyze ▸ Series.** Once a map is built, the stages are found in the background: a colour strip at
the right of the map, dashed lines where a stage begins, red arrows at odd frames (Marks ▸ Stages hides them).
Under the controls: **Stages** Auto (n) or a number you choose, and the frames of every stage. Folded below, *What
changes, and the odd frames*: the typical frame of each stage, where the curve grows or falls most between stages
(relative to the rest of it), by which frame half and 90 % of the change had happened, and why each odd frame is
odd — a difference near one q in a few points is the detector (mask it in the Mask step). **Leave the odd frames
out of Batch Export** does what it says. The lower-right plot can show *Change along the series* (the main
component, coloured by stage). **Export ▸ Stages as Table…** writes every frame's stage, odd or not and why, with a
JSON record of the method.

How they are found: log I is compared in shape (each frame's mean level removed), odd frames first (a frame that
matches neither the frames before nor after it), then the main components of the other frames, cut into stages by
straight lines in frame order; a stage is added while it explains at least 5 % of the change and more than noise
would. A stage describes the series; it is not a phase by itself — a smooth change is also cut where its pace
bends. What grows or falls between stages says whether the structure changed.

**Stages in Fitting ▸ In-situ series.** The listed curves are compared as they will be fitted: the Curves step says
how many stages and odd frames there are, the frame list is coloured by stage, **Leave out the odd frames** is on
by default, and *Each frame starts from* offers **The previous result; the Single model at each new stage**. The
trend is drawn in the colours of the stages.

**The Compare page** (sidebar ▸ Compare). Add series with **Add Series**: the Series map of Analyze (or **Send to
Compare** in its Series tab), a folder of curve files, or curve files (q and I columns; nm⁻¹ is converted). Typical
use: open one sample in Analyze, build its map, Send to Compare; repeat for every sample.

1. **Series** — the list (double-click a name to rename it), Remove Selected, Remove All.
2. **Compare** — the q range compared (the range every series covers; narrow it to leave out a noisy edge or a
   detector artefact, **Whole Range** to go back), **Compare the shape only** (on by default), and how many of the
   last frames make a series' end state. Every change compares again within a second or two.
3. **Results** — groups (three or more series, split where the end states clearly separate), the series that
   differs most at the end and at the start, and per series: frames, odd frames, stages, the frame by which half
   and 90 % of its change had happened; a table of how different the series are, at the end or at the start (in
   percent of intensity).

On the right: how far each series has changed (against frame, or the share of each series for runs recorded at
different rates), their paths through the two main changes, and their end states. **Save** writes a table of every
series or of every frame (CSV with a JSON record) and the plots. A project keeps the series (curve files by path,
Analyze maps in `<project>.compare.npz`).

## 8. WAXS / GIWAXS

Open WAXS or GIWAXS frames in **Analyze** (frames with scattering angles above 20° are reduced as GIWAXS
automatically, or choose **GIWAXS**). The steps follow the usual processing:

1. **Import and preprocess** (Mask & corrections): hot and dead pixels are left out automatically; draw
   rectangles or polygons on the detector image, or load a mask file (JSON from GIMaP, or an EDF/TIFF image
   of the same size where non-zero = masked); **Fill gaps from the mirror side** takes pixels without data
   from their mirror position about the beam-centre column (GIWAXS is symmetric in ±q∥). Background,
   gap guard and the valid intensity range are under Corrections. **Intensity corrections (GIWAXS)** —
   off by default — divide by the solid angle of each pixel (cos³2θ), the polarisation (factor 0.95–0.99
   at a synchrotron, 0 for a laboratory source) and the absorption in the film (thickness and attenuation
   length; relative to αf = αi, without refraction). Peak positions do not change; intensities compared
   across χ or q do (orientation, pole figures, crystallinity ratios). For photon counts the errors are
   propagated (a corrected pixel's variance is its value divided by the factor); the JSON record says
   what was corrected and how.
2. **Calibration** when the frame has no geometry (the banner: Find Calibration Automatically, Enter
   Geometry, Calibrate).
3. **q map**: the view **q map** shows the intensity against q∥ (qr) and qz; **Cake** unwraps it onto χ and q.
   **Save ▾** above the image saves the view as a figure or its data as a CSV table with the axes.
4. **Cut regions** (Cuts step): the full ring, the in-plane and out-of-plane bands and the ring of I(χ) are
   there from the start. To add your own cut, press **Ring**, **Sector** or **Spot** and click on the image
   (detector, q map or cake; for a ring also on the I(q) plot): the region snaps to the peak there — centre and
   FWHM fitted from the pixels, window = centre ± FWHM — and a note gives q, d and the FWHM (Undo is on the
   note). **Add ▾** also draws a rectangle on the Cake view, adds presets, and saves or loads a set of cuts
   (JSON) for the next data set. A region is a q range × a χ range (optionally both sides ±χ). Move and
   resize it on the Cake view, or type its **centre ± half width** under the list; **Snap to Peak** re-centres
   it on the nearest peak. Each region gives I(q) (upper plot) and I(χ) (lower plot) in its own colour, and is
   outlined on the q map; **Sources** shows its pixels on the detector.
5. **Results**: each plot has **Save ▾** (figure, or its curves as CSV); **Export** writes every curve with a
   JSON record. **Run Automatic Analysis** adds peaks, orientation and sizes; while it runs, the Results tab
   shows what it is doing, the phases done and the time, and **Stop** ends it before its next step (a slow step
   such as the geometry or model fit finishes first — the panel says so). What it found is kept: **Save
   Report…** or **Discard**.
   Every image and plot has **Zoom** (drag a rectangle; Shift + drag works everywhere; Reset shows all again),
   and a curve with a signed x (qy, χ) can be shown as both halves, one half, or both folded onto |x|.
6. **In-situ and batch**: list the files (or open a multi-frame NeXus file) and use the **Series** tab: any
   curve of any region as an intensity map against frame and q, a frame's curve, the intensity or the peak
   (position, FWHM, area, height) in a q window against frame, and CSV export.
7. **Batch Export…** — shown in the command bar, under the file list and in the Series tab as soon as more than
   one frame is listed (Ctrl+Shift+E; also in the Export step, Export ▾, and on the Start page with a folder of raw frames):
   one dialog in four groups — **Data** (a table per curve with every frame as a column; per-frame curves with σ;
   the Fitting input; q maps; cakes — as CSV, tab-separated TXT or space-separated DAT), **Pictures** (the
   detector image and/or the q map as PNG, TIFF, SVG or PDF), **Converted detector frames** (the data as TIFF,
   EDF, NumPy or HDF5: a format conversion) and **Fitting** (optional: the peaks of the ring regions for GIWAXS,
   a particle model of the horizontal cut for GISAXS; start values from the previous frame, the first frame, or
   found anew; **Try on This Frame** shows the fit before the batch). Every option shows the file it writes;
   the files of one kind go into their own subfolder, and `README.txt` says what each file is.
   The choices and the folder are remembered. **Speed**: *Gentle* reduces one frame at a time (the
   computer stays free), *Balanced* (default) and *Fast* several at once in separate processes at low
   priority — as many as the cores and half of the free memory allow (a Lambda 9M frame needs about 1.4 GB
   while it is reduced); a short batch stays one at a time. While it runs, a panel above the Results /
   Series tabs shows the frames being reduced, how many are done, the time per frame and the time left, the
   latest fit, **Pause** (no new frames) and **Stop** (the frames being reduced finish first — the panel
   says so; everything written stays and the tables are written for the frames done); how many frames at
   once can be lowered while running. The **Series** tab shows the map of every frame done so far, growing
   as they come, with the newest frame's curve below it (click a row to look at another one); after the
   batch the map stays there to explore and export. **Save Settings… / Load Settings…** keep the whole
   set-up (geometry, masks, corrections, cut regions, export choices) as a file, so the next raw data
   needs only: Batch Export… → folder → Export.

   Example (Lambda 9M, 4727 × 3142 px, peak fitting on): one frame at a time about 2.5 s per frame; with
   *Balanced* (4 at once) 41 frames took 31 s.
8. **Colour limits**: next to every image is a histogram of the displayed values with two handles: drag them
   (or the band between them) for a quick change — the mouse wheel over the bar zooms it for finer moves. **Levels
   ▾** has *Auto for every frame* (each new frame gets its own limits) and the rule (1–99.7 %, 0.1–99.9 %, 5–95 %,
   min–max, mean ± 3σ), **Min** / **Max** typed in intensity units, and **Auto Once**; a click on **Levels**
   itself is Auto Once. Limits you drag or type stay — for other frames, re-analysis, the live Series map — until
   Auto (the button then reads *Levels (fixed)*); they mean the same in log and linear display, and the detector,
   q map, cake and Series map each keep their own — the Series map one per curve (I(q), I(χ), a region …), so
   switching the curve does not carry one curve's limits to another. In **Batch Export ▸ Pictures**, *Colour scale*: each frame its
   own limits (1–99.7 %), or the limits on screen for every frame, so the pictures can be compared (recorded in
   the batch's JSON). This follows napari (auto-contrast once / continuous), Mantid (the Autoscale check box) and
   silx / pyFAI (autoscale modes, typed limits).
9. **Marks**: every image (detector, q map, cake, the Series map) has **Marks ▾**, every curve plot an eye
   button: one switch per kind of mark drawn there — beam centre, sample horizon, cut bands, masks and cut
   regions, the pixel overlay (masked pixels, curve sources), marked hot / dead pixels, the q box, the colour
   bar, the window band of a plot — plus Show All / Hide All. Hidden marks stay hidden when the frame changes
   and next session; pressing **Sources** or **Show Masked Pixels** shows the pixel overlay again. The marks you
   made can be removed from the same menu: **Remove the Drawn Masks**, **Remove All Cut Regions**, **Remove
   the q Box** (Undo brings them back). Single regions are shown or hidden with their check box in the list.
   The calibration preview has its own switches (masked pixels, rings, beam centre lines, Clean image).
   On the **q map** the beam centre is the direct beam, q∥ (qy) = 0, qz = 0 — just below the map, which starts at
   the sample horizon (drawn dashed at qz = k sin αi); Fit view keeps it in sight. It is moved on the detector
   image.
10. **Undo / Redo** (↶ ↷ in the command bar, `Ctrl+Z`, `Ctrl+Shift+Z` or `Ctrl+Y`): every change of the
   set-up — a mask drawn, a region added or moved, a band dragged, the profile, αi or the mode, the
   corrections, a settings file loaded, an AI card applied — is one step; the tooltip says what the next
   Undo or Redo changes (“Undo: cut regions”). A drag or a quick series of the same edits is one step. The
   instrument profiles themselves (a calibration saved) are not undone this way.

### Process with AI (GIWAXS assistant)

**Process with AI…** (Analyze toolbar, or **Tools ▸ Process with AI…**, `Ctrl+Shift+L`)
lets Claude analyse the GIWAXS frame shown in Analyze with the same tools you use: it switches the
mode, adjusts sectors, sets the I(χ) window, fits peaks and exports files, and the page follows
every step on screen. The **Claude** panel on the right shows a summary of Claude's reasoning,
each tool call (hover for its arguments and result) and, at the end, the report.

The AI can think with one of three **brains** (Settings ▸ Assistant, or *Brain* in the start
dialog):

- **Claude Code — your Claude plan (default)**: GIMaP runs your local Claude Code in the
  background, so the run counts against your Claude Pro or Max plan and needs no API key. Claude
  Code must be installed (the Claude desktop app includes it) and signed in once: **Settings ▸
  Assistant ▸ Sign In…** opens a window running `claude auth login`; choose your Claude account,
  then press **Check**. Claude Code gets only the GIMaP tools — no shell, files or web — and no
  API key is passed to it. Model and effort are set in Settings (empty model = Claude Code's
  default).
- **Claude API — API key**: billed per token. Install the SDK once (`pip install anthropic`) and
  add a key in **Settings ▸ Assistant** (or set `ANTHROPIC_API_KEY`); **Test Connection** checks
  the key and model.
- **Other AI provider — OpenAI-compatible API**: DeepSeek, Qwen (Alibaba Cloud Model Studio),
  OpenAI, Kimi, Zhipu GLM, Google Gemini (its compatible endpoint), OpenRouter, SiliconFlow, Azure
  OpenAI, or a model on your own computer or lab server (Ollama, LM Studio, vLLM, any compatible
  service). In **Settings ▸ Assistant ▸ Other AI providers** pick the provider (the address is filled
  in; change it for a local server or Azure), save its key (or set the provider's environment
  variable, e.g. `DEEPSEEK_API_KEY`, `DASHSCOPE_API_KEY`), choose a model (typed, suggested, or
  **Get List** from the provider) and **Test Connection** — it sends one tiny request with a tool,
  because the model must support tool calling. Keys stay in the user data folder, one per provider.

1. Open a GIWAXS frame with its instrument profile, then click **Process with AI…** and choose:
   - the results: *Peak table* (q, d, FWHM, intensity, significance), *Orientation* (in-plane
     versus out-of-plane intensity of each peak), *Orientation distribution of one ring* (I(χ),
     maxima, Herman's orientation factor; pick the ring or let Claude choose), *Crystallite size*
     (Scherrer coherence length);
   - optional notes on the sample and what you want to know;
   - **Permissions**: *Preview first* (recommended for new users: every settings change the AI
     makes is undone when it finishes and comes back as a card to apply), *Ask me before writing
     files or changing corrections* (a dialog appears for
     each such action; a refusal is respected) or *Fully automatic* (everything is still logged);
   - whether Claude may see a small image of the q map (qualitative only), and the report language.
2. **Stop** ends a run at any time. The report lists each requested result as *done*, *partial* or
   *not available* with the reason (for example "no peak above 3σ between 0.2 and 2.0 Å⁻¹"), and
   the tables below it come from the tool results, not from Claude's text. **Save Report…**
   writes an HTML report and a JSON record next to the data (`gimap_analysis/`).

**Baseline first, then questions.** Claude may start with GIMaP's standard procedure in one step
(*run_standard_pipeline*: geometry, the last frames of a series, peaks, in-/out-of-plane, ring
orientation and sizes of the strongest reliable peaks and of the ring you asked about, each
decision with its reason). Its choices are defaults, not limits: Claude then follows what your
question needs — the start of an in-situ series, lines every sample shares, peak ratios, a peak
the baseline skipped — and marks interpretations as hypotheses, with the numbers behind them.

**Changes you can preview, apply and undo.** The AI's answer is not only text. Every change it
makes to Analyze (sector widths, a custom sector, a q box, the frame, the mode, αi, the radial bins,
the valid intensity range) is recorded, and it can suggest changes without making them. They appear
as cards under **Changes** in the panel: a picture of the q map with the change, the setting in
words, why the AI suggests it, and **Apply**, **Dismiss** or **Undo**; **Undo All** takes back every
applied change. In *Preview first* the AI's own exploration is restored at the end: a setting the AI
proposed a value for shows only that proposal; otherwise the net change of each setting remains as
a card, and changes that ended where they started or changed nothing visible are left out.

**Follow-up questions.** When a run has finished, type a question under the report (*Ask a
follow-up…*, e.g. “Why is there almost no signal in the in-plane sector?”) and press **Ask**: a
new run on the same frame and settings answers it, knowing what the previous run found, and can
bring new cards.

**A frame without geometry.** When no instrument profile matches the frame, the AI does not stop.
Files or folders you name in the notes (or in the standing instructions) are read and searched
first — e.g. *“the calibration is D:\beamtime\calib\img_0005.cbf, 11.8 keV”*. It also searches the
frame's folder and subfolders, the parent and all sibling folders, two folders up with their
subfolders and calibration / log folders three up, and can search any folder further up or down
(read-only) for

- calibration results — pyFAI `.poni` files and calibrations saved by GIMaP's Geometry
  Calibration (`.json`);
- images of a calibration standard — the standard is taken from the file or folder name (AgBH /
  silver behenate, LaB6, CeO2, a LaB6 + CeO2 mixture; Si, Al2O3 and Cr2O3 are recognised but
  cannot be fitted yet); a multi-module NeXus series (`*_m01.nxs` … `*_m11.nxs`) counts as one image;
- logs and parameter files (`.log`, `.fio`, `.txt` …) that state the energy, the distance, the beam
  centre or the incidence angle.

It prefers a calibration of the same kind of detector file taken before the frame, with
*giwaxs*/*waxs* and *final*/*redone* in its name, fits standard images with GIMaP's calibration and
judges a fit by where the standard's lines land in q: within 0.2 % on average is good, within 0.5 %
usable. (The ring-fit residual in pixels is not used for this: on a tilted wide-angle detector a
good calibration can leave several pixels.) When the name does not say which standard an image
shows, it fits every standard and compares the same way.
Only when the numbers cannot decide, or the energy is nowhere to be found, a **Claude asks** dialog
lists the options with their file times and folders (or asks you to type the value). Files within
two folders above the frame and files you named are read without asking; other files are read
after you agree (confirm mode) or with a note in the panel (automatic mode). Hidden and system
folders (`.ssh`, `AppData`, `Windows` …) are never read. The chosen geometry is saved as the instrument profile of the detector — in
the confirm mode after you approve it — and the report states which file, standard and values it
came from.

What is sent: the frame's status (file name, geometry, settings) and the reduced curves as
numbers; the detector file itself is never sent, and the q-map image only when allowed. The panel
shows the tokens of each run: with Claude Code also the plan's usage-limit warnings and the cost at
API prices (for comparison — a plan is not billed per token); with the API the estimated cost.
The API brain's default model is `claude-opus-5` (effort *high*); its requests use Anthropic's
server-side fallback for declined requests. Every
run is recorded in the user data folder (`assistant_runs/`), and analyses Claude found missing are
collected in `assistant_feature_requests.jsonl` so they can be added later.

Limits: Scherrer sizes are lower bounds of the coherence length unless the instrumental width is
known, and broad halos (FWHM above 15 % of q) get none; Herman's factor assumes a film that is
isotropic in its plane and weighs each |χ| by sin χ, so it is given only when at least 80 % of that
weight is measured (the part near the sample plane counts most); with a partly measured ring the
report also gives f of a random ring over the same |χ|, the value to compare with. Regions far
below the diffuse background at the same q (under 20 %: a shadow, an absorber or an insensitive
detector area) count as unmeasured, and a sector lying in one is not compared. Intensity next to
the qz axis may be unmeasured (missing wedge). Peaks one or two bins wide far above the
background are marked as spikes (hot pixels, module edges), not diffraction. Claude uses
crystallographic labels such as (100) or edge-on only when the sample notes or the data make the
assignment clear.

### Without the window: command line, Codex and other agents

`tools/gimap_agent.py` runs the same tools without the GIMaP window, for scripts and for other
agents (Codex reads `AGENTS.md`, which points to `docs/agents/giwaxs-playbook.md`):

```bash
python tools/gimap_agent.py auto sample_A_00001_m01.nxs sample_B_00001_m01.nxs --notes "alpha_i = 0.4 deg"
```

`auto` performs the standard procedure — geometry (a saved instrument profile, or the best
calibration found around the frame, checked by its lines), the last ten frames of a series summed,
peaks, in-/out-of-plane, ring orientation and sizes of the strongest reliable peaks — and writes
for every frame a `report.md` with each decision and its reason, the curves as CSV, the q map, and a
`summary.md` for the batch (later frames of the same detector reuse the first calibration). What
only you know goes under *Needs attention* with the option that supplies it (`--incidence-deg`,
`--energy-kev`, `--pixel-size-um`, `--calibration`, `--standard`); αi, the energy and the pixel
size are also read from `--notes` when it states exactly one value. Results go to the user data
folder (`assistant_runs/cli/`) or `--out`, never next to the data; saved instrument profiles are
read but not changed. `auto` is a baseline, not the whole analysis: `status`, `find-calibration`,
`tools` and `call` (a JSON list of tool calls) give finer control, and `mcp` serves every tool over
MCP stdio for Codex or Claude Code (`open_frame`, then `run_standard_pipeline` and the others on the
same frame). The playbook describes both levels — the baseline every agent runs, and the questions
a capable agent follows up — and `tools/eval_giwaxs_agent.py` evaluates them on a real case
(`baseline`: what the procedure must get right; `report`: mistakes and findings in an agent's
written report).

## 9. Geometry Calibration Tool

Open **Tools > Geometry Calibration...** (`Ctrl+Shift+G`) to calibrate a SAXS, GISAXS, or GIWAXS detector geometry without leaving the current page.

1. Open a calibration-standard `.nxs` or `.cbf` image, or paste its path into the image field and press Enter.
2. Confirm the detected energy and pixel size. For a CBF with matching scan NXS metadata, the energy is filled automatically. Enter the energy in keV if it is still missing.
3. Confirm the detector model. If it cannot be identified, choose a known detector from **Detector model**, or select **Custom pixel size** and enter the values in **Advanced Settings**.
4. Choose a standard, or leave **Auto Detect** selected. An approximate detector distance is optional but helps resolve harmonic alternatives.
5. Click **Auto Calibration**. The calculation runs in the background and can be cancelled.
6. Review the center, distance, residual, confidence, high-contrast overlay legend, and alternative candidates. **Clean image** temporarily hides all calibration overlays, **Reset view** restores the complete detector mosaic after zooming or panning, and **Focus image** hides the result panels to give tall WAXS mosaics more room. The horizontal divider can also be dragged.
7. After calibration, **Manual refine** opens automatically. Drag the center marker, edit the center/distance values, or pair a detected ring with a theoretical peak. Use **Finish manual** to collapse the panel when more image space is needed.
8. Click **Apply** to update the shared application geometry. Calibration results can also be exported to or imported from JSON.

Solid yellow overlays are matched theoretical rings, dashed orange overlays are unused theoretical rings, and dotted white overlays are detected experimental radii; the preview legend identifies each style. Partial WAXS arcs and centers outside the active detector area are supported. A low-confidence result or a one-ring result should be treated as ambiguous and reviewed manually.

## 10. XRR Series Extractor

Open **Tools > XRR Series Extractor...** (`Ctrl+Shift+R`) to obtain an XRR curve from a
GIWAXS/GISAXS detector angle series without leaving the current workspace.

1. Select one NXS detector-module file, or select a CBF file/folder and glob pattern. For NXS, the
   sibling module files are stitched for each detector frame; for CBF, each naturally sorted file is
   one point.
2. Choose a linear sample-angle sequence (`theta start` and `theta step`) or enter an NXS motor
   dataset containing one theta value per frame.
3. Enter detector distance, beam energy, pixel sizes and the direct-beam center. Use **Load first
   frame** and **Pick direct-beam center** to choose the center with the mouse. Select whether the
   specular reflection moves up or down in detector-y.
4. Choose an ROI radius and `Sum` or `Mean`. Radius 0 reads one pixel; larger values integrate a
   circular neighborhood around the calculated specular beam.
5. Click **Run extraction**. The worker loads one full frame at a time, displays the current detector
   and ROI, records one scalar point, and then releases that frame. It does not load the full series
   into memory.
6. Switch freely between **Live frame** and **XRR points**. Processing updates both but never changes
   the selected tab. Use **Export CSV** to save theta, qz, intensity, ROI position and valid-pixel
   count.

The current extractor reports the raw ROI sum or mean. It does not apply monitor/flux normalization,
footprint correction, background subtraction or resolution correction.

## 11. Model Configuration

Prediction modules are configured with YAML files under `modules/`. Existing module files contain fields such as:

- `id`: internal module identifier
- `name`: user-visible module name
- `framework`: model framework, for example TensorFlow/Keras
- `version`: module version
- `model.model_path`: path to the trained model
- `preprocess.entry`: preprocessing entry point
- `preprocess.steps`: preprocessing steps
- `preprocess.params`: preprocessing parameters
- `io.input_type`: expected input type, for example `cbf`
- `io.stack_axis`: stack axis when applicable
- `io.input_shape`: expected model input shape
- `outputs`: output definition, parameter names, ranges, or output type

When adding a new module, make sure the model path, input type, preprocessing code, and output definition match the trained model.

AI fitting model discovery is handled separately and searches fitting-model folders under `modules/`, including `modules/Fitting_1D_Model`.

## 12. Troubleshooting

### File Cannot Be Loaded

The message under the image says why in words (empty file, not a detector image, damaged or
incomplete NeXus, no permission). Otherwise, check that the file format is supported by the active loader. For detector formats such as `.cbf`, confirm that `fabio` is installed. For HDF5/Nexus-like files, confirm that `h5py` is installed and that the internal dataset path matches what the loader expects. If support for a custom file format is required, please contact yufeng.zhai@desy.de.

### Model Not Imported

Check the selected module configuration and model path. If the path points to a local machine-specific location, update it for the current computer.

### Predict Button Disabled or Prediction Fails

Confirm that input data is selected, a module is selected, the model is loaded, and required dependencies are installed.

### TensorFlow / PyTorch Not Available

`requirements.txt` lists TensorFlow 2.15.x. PyTorch is not listed as a project dependency. If a module requires another framework, install and configure it separately and confirm that loader code exists.

### Windows SmartScreen Warning

Unsigned pre-release executables can trigger SmartScreen. Only run files downloaded from a trusted release source. If you do not want to run an unsigned executable, you can run GIMaP from source in a local Python environment instead. See the “Running from Source” section in README file.

### Missing Dependency

Activate the correct Python environment and reinstall dependencies:

```powershell
pip install -r requirements.txt
```

### Empty Output

Check the input range, stack selection, model compatibility, and whether the selected frame contains valid data. For fitting workflows, also inspect masks, cut ranges, parameter bounds, and scale/background settings.

### Export Fails

Confirm that the output folder exists and is writable. Avoid exporting into protected system folders.

### GUI Layout Too Large for Small Screen

GIMaP uses the display scaling of the operating system and a compact layout that fits a
1280 × 720 screen. If text is too large or too small, use **View ▸ Font Size** or
**Tools ▸ Settings ▸ Appearance**; dense panels scroll instead of being clipped. The window
size and position are restored at the next start.

## 12. FAQ

### Is GIMaP production-ready?

No. The current repository should be treated as early pre-release / beta scientific software.

### Can I use my own trained model?

Yes, if it can be represented by a compatible module configuration and supported loader. Add or edit a `module.yaml` under `modules/` and make sure preprocessing and output definitions match the model.

### Where are AI fitting results saved?

AI fitting results are saved under `AI_Fitting_Output/current_prediction`.

## 13. Version and Contact

This documentation describes the current source repository state and may change as the GUI evolves.

Contact:

[yufeng.zhai@desy.de](mailto:yufeng.zhai@desy.de)
