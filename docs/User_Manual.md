# GIMaP User Manual

## 1. Introduction

GIMaP is a PyQt-based desktop GUI for GISAXS/GIWAXS data visualization, analysis, fitting, and machine-learning-assisted workflows. It is intended for users who work with grazing-incidence scattering data and want a graphical workflow for inspecting detector images, extracting 1D curves, fitting physical models, running trained-model predictions, and organizing batch results.

This manual describes version 0.1.0, a pre-release: treat it as beta scientific software. The workspaces (Analyze, Fitting, Compare) are meant for daily work; the Labs pages and 1D Predict are experimental. What changed since 0.0.2, and the known limitations, are in `CHANGELOG.md` (section 15).

## 2. Installation

### Windows: the portable version

No Python is needed; the package brings its own.

1. Open the [GitHub Releases](https://github.com/zyffcc/gisaxs_gui/releases) page and choose the latest release.
2. Download `00_Download_and_Install_GIMaP-<version>.bat` into an empty folder with a short path that you
   may write to (for example `C:\GIMaP`).
3. Double-click it. It downloads the package parts, verifies each one, joins and extracts them into the
   folder `GIMaP-<version>-windows-x64-portable` beside the installer, and starts GIMaP once.

**Starting GIMaP later.** Double-click `Start_GIMaP.bat` in the folder
`GIMaP-<version>-windows-x64-portable` (a desktop shortcut to that file is convenient). The first start
prepares the bundled Python environment for the folder, which takes a moment.

**Updating.** Run the installer of the new version in the same way. It extracts into its own folder; start
the new `Start_GIMaP.bat` and delete the old folder when you no longer need it. Your settings, instrument
profiles and recent files are kept outside the program folder (*Settings and user data* in section 3), so
an update keeps them.

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

The dependencies are listed in `requirements.txt`; on macOS, BornAgain is installed separately (see
`README.md`). The trained weights of 1D Predict are in the Git repository (about 56 MB), so a checkout runs
everything, including 1D Predict (section 4).

### Updating a source checkout

In the folder of the checkout, run:

```powershell
powershell -ExecutionPolicy Bypass -File tools\update_source.ps1
```

`tools/update_source.ps1` updates the branch that is checked out from GitHub (`origin`) and says what it
found and what it did:

- Nothing new: nothing is changed. New commits only: the branch moves forward to them.
- After the history was rewritten (see below): the state you have is first kept as a branch
  `backup/<date>-<time>-<commit>`, then the branch is moved to the new history.
- Your edits of tracked files are put aside (`git stash`) and put back on top of the new version; a copy
  stays in the stash. When they do not fit the new version, the new version is left complete and the
  script says how to get them back.
- Files Git does not track or ignores — your data, results, `AI_Fitting_Output/` — are not touched, and
  your settings and instrument profiles are outside the checkout anyway (*Settings and user data* in
  section 3). When the new version brings a file to a place where you have a different file of your own,
  the script stops before it changes anything.
- Model files of 1D Predict that an older checkout holds with converted line endings are written again as
  the repository has them, also when there is nothing new to download. 1D Predict checks these files byte
  for byte, and Git does not notice the difference on its own.
- When Git cannot replace a file because another program has it open (GIMaP itself, an editor), the
  version you had is put back whole, with your edits where they were: close the program and run the script
  again.
- When the list of Python packages changed, it says so: run `pip install -r requirements.txt` then.

Add `-DryRun` to see what an update would do without changing anything. At the end the script prints how to
go back to what you had, and how to delete the backup once things look right. The script needs Git 2.16 or
newer and says so when it finds an older one.

A checkout that does not have the script yet (one last updated before 2026-10-13) is first moved to the new
`main` with two plain commands, with `main` checked out:

```powershell
git fetch origin
git reset --hard origin/main
```

Then run the script once, with the command at the top of this section. On Windows the two commands leave
some model files of 1D Predict with the line endings the older checkout had given them, and 1D Predict
(*General V5*) does not start until the script has written them again. From then on, update with the script.
(On Linux or macOS, where Windows PowerShell is missing, nothing was converted, and `git pull` works again
after the two commands.)

`git reset --hard` discards your edits of tracked files, and commits you made yourself are no longer on the
branch. To keep them, run `git stash` (edits) or `git branch my-work` (commits) first; afterwards
`git stash pop` puts the edits back, and `git cherry-pick <commit>` brings a commit over. Do not merge or
push `my-work` itself: it carries the old history. Files Git does not track stay, with one exception: where
the new version has a file at the same place — for example weight files of 1D Predict that you had copied
into `modules/Fitting_1D_Model/Workflow_v5/` by hand — the file of the new version replaces yours without
asking.

To keep all of that instead, let the script do the first update: `git fetch origin`, then
`git checkout origin/main -- tools/update_source.ps1`, then the script. It keeps the state you have as a
backup branch, puts your edits back, stops before it would replace a file of yours, and writes the model
files again in the same run.

A checkout of 0.0.2 that was used for fitting has one edit you did not make yourself:
`config/model_parameters.json`, where 0.0.2 kept the model of Fitting. Its values are not carried over:
0.1.0 keeps the model in the user data folder and starts with a neutral one. The script reports the file as
an edit that does not fit the new version and ends with exit code 2: the new version is complete, and what
you had changed waits in the stash. When this file was your only edit, delete that copy with
`git stash drop`. Do not run `git stash pop` for it, although the script names that command for edits you
want back: it would write conflict markers into the file, which is the model a new user of Fitting starts
with. (With other edits in the same stash: `git stash pop`, then
`git checkout HEAD -- config/model_parameters.json` to put the file of the new version back, then
`git stash drop`.)

The history of `main` was rewritten on 2026-10-13. A clone last updated before that day can hold commits
that are no longer on GitHub: `git pull` then stops with an error, or tries to merge the old history with the
new one.
Do not merge, and do not push from such a clone; move it to the new `main` as described above (or clone
again).

## 3. Main Interface

The main window has a sidebar on the left, the active workspace in the middle, a menu bar
(**File, View, Tools, Help**); messages appear as short notes at the bottom right.

The sidebar groups the pages into **Workspaces** (Start, Analyze, Fitting, Compare) and **Labs**
(2D Prediction, Trainset Build — the machine-learning tools). `« Collapse`
(or **View ▸ Collapse Sidebar**, `Ctrl+B`) reduces it to icons; `Ctrl+1` … `Ctrl+6` switch pages.

Detector frames, folders and `.gimap` projects can be dropped anywhere on the window; a curve file
(`.dat`, `.txt`) dropped while Fitting is shown opens there. The window title names
the frame shown (and the project, when one is open). Closing GIMaP while something still runs —
Batch Export, a Series map, the automatic analysis, a fit, an In-situ series, a 2D prediction — asks
first and lists what runs; **Yes** stops them and quits.

Every curve plot has a menu on the right mouse button (**Reset View**, the log axes, **Plot as
Figure…**, **Curves as Data…**, **Copy Image**, **Copy Data**); dragging with the right button
zooms. When a plot is narrow, its title wraps above it and its options move into a **⋯** menu.

The result tables — the peaks and solutions of the automatic analysis, the tables of Fitting and of
Compare — can be copied: select rows and press `Ctrl+C`, or use **Copy Rows** / **Copy Table** on the
right mouse button. The rows are copied with their column names, ready to paste into a spreadsheet.

- **Start** (the first page): drop detector images or a folder on the drop box, or open them —
  Analyze opens at once and shows the file, its detector and frames while the image loads. Recent
  files are listed with their folder; the page scrolls instead of squeezing on a small screen. Or choose what you want
  to know: *Crystals and orientation* (GIWAXS), *Nanostructure* (GISAXS), *In-situ or batch series*,
  *Calibrate the detector* — or describe your data and question to the AI (**Ask the AI…**).
- **Analyze**: one workspace for GISAXS and GIWAXS. The command bar holds the file, the mode
  (Auto / GISAXS / GIWAXS), αi, **Run Automatic Analysis**, **Ask AI…**, **Export** and **Send to
  Fitting**. On the left the steps — ① Data ② Geometry ③ Mask & corrections ④ Cuts ⑤ Results
  ⑥ Export — each with its state and one line of what it found; the image in the middle
  (Detector / q map, **Sources** shows where every curve comes from — click a curve to see only
  its region); on the right **Curves**, **Results** and **Series**.
  - *The file list* (Data step): click a file to show it; `Page Up` / `Page Down` (or ‹ › in the
    command bar) step through the list. To take one file off the list, right-click it —
    **Remove from List**, **Show in Folder**, **Copy Path** — or select it and press `Delete`. The
    file itself is not touched, and the rest of the list, the Series map and Batch Export go on
    without it. **Clear** empties the list.
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
    and **Show in Fitting**, which opens the curve in Fitting with that solution drawn (**Save ▸ Data
    and Fit…** there saves it with its parameters).
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

The tool windows are in the **Tools** menu: Geometry Calibration (section 9), Format Converter
(section 10), XRR Series Extractor (section 11) and 1D Predict (section 4).

The former WAXS page has been retired: every WAXS/GIWAXS function (background, masks, sectors,
q-range and circle cuts, integration axis, image export, batch with calibration/normalization)
is in Analyze's GIWAXS mode. The detector half of the former Cut & Fitting page is in Analyze too.

### Menus

| Menu | Entries |
|---|---|
| File | Open Data… (`Ctrl+O`), Open Folder… (`Ctrl+Shift+O`), Open Recent, Open Project… (`Ctrl+Shift+P`), Save Project (`Ctrl+S`), Save Project As… (`Ctrl+Shift+S`), Labs Parameters ▸ Load… / Save As…, Exit |
| View | the workspaces (`Ctrl+1`…`Ctrl+6`), Collapse Sidebar (`Ctrl+B`), Full Screen (`F11`), Theme ▸ Light / Dark, Font Size ▸ Larger (`Ctrl++`) / Smaller (`Ctrl+-`) / Reset (`Ctrl+0`), Language ▸ English / 中文 |
| Tools | Geometry Calibration… (`Ctrl+Shift+G`), Format Converter… (`Ctrl+Shift+C`), Convert Current File…, XRR Series Extractor… (`Ctrl+Shift+R`), 1D Predict — Fit Many Curves…, Process with AI… (`Ctrl+Shift+L`), Settings… (`Ctrl+,`) |
| Help | User Manual (`F1`), GitHub Repository, Open User Data Folder, About GIMaP |

*Labs parameters* are the settings of 2D Prediction and Trainset Build; Analyze keeps
its own set-up in settings files (**Export ▸ Save Settings… / Load Settings…**, also used by Batch
Export), and a project keeps everything of one sample (below).

### Projects

A project (`.gimap`, a JSON file) reopens a sample as it was left: the frames and the whole set-up of
Analyze (mode, profile, masks and drawn regions, corrections, cuts), the curve, halves, fitting range,
left-out points and model of Fitting, the folder and choices of the In-situ series, and the series of
Compare. **File ▸ Save
Project** (`Ctrl+S`) writes it (the first time it asks where, next to the first frame by default);
**Open Project…**, the Start page's **Open Project…**, **Open Recent** (projects first) or dropping a
`.gimap` file on the window opens it. The window title shows the project's name and the frame shown.
After **Clear** in Analyze, what you open next is no longer that project: **Save Project** asks where
to save instead of overwriting it. A project cannot be opened while a job runs (an In-situ series,
Batch Export …): stop it first. Opening a project replaces what the window holds: the curve and the
In-situ series of Fitting and the series of Compare of the sample before are closed (save them first
if you need them). Data are referenced by path, not copied: when a file has moved, the project still
opens; a note that stays until you close it says how many files are missing and names the first ones,
and **Details…** lists them all.

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
overwrites them. Settings files of older versions found in the program's own `config/` folder
(`user_parameters.json`, `user_settings.json`, `instrument_profiles.json`: a source checkout updated
in place) are imported once and left untouched; a new portable folder has none and starts with the
defaults.

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
| Mask negative pixels + guard | Analyze ▸ Mask & corrections step ▸ **Corrections** (folded) ▸ Gap guard (3 px by default) |
| Stack | Analyze ▸ Data step ▸ **Frames** (folded) ▸ Sum |
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
   at 1 by default). After a manual fit every value shows its ± error, or *at a bound*;
   fit1d keeps its own parameter errors and states in Results.
3. **Fit** — one button, five methods:
   - **Refine the current values** — bounded least squares from the values in Model; stays near them.
   - **Search the ranges, then refine** — differential evolution across the ranges of the free values
     (log scale when a range spans 10× or more), then refinement from the three best starts.
   - **Find the particle shape (no AI)** — the numerical fit of sphere, random and vertical cylinder from
     several starts (as in Analyze's automatic analysis); the solutions go to Results, the best into Model.
     A first curve opens here.
   - **AI proposal (1D Predict)** — the V5 model with its numerical correction; its solutions likewise.
   - **fit1d (network + physical fit)** — the current v2 networks and physical fit, for the 1–4
     Sphere / Vertical cylinder populations you put in Model. See *fit1d* below.
   Progress and **Stop** (Refine / Search keeps the best values; cancelling a search for solutions or
   fit1d produces no new result). The scales (every particle's scale, the
   background, the peak A) are solved exactly at every step, so the search only moves sizes, spreads,
   distances and the peak's shape. *Advanced*: the number of evaluations; **Fit Many Curves…** (the 1D
   Predict window for a list of files, see *1D Predict* below).
4. **Results** — χ²ᵣ (with σ) or the variance of ln I, points, free values, converged or not; warnings when a
   value stopped at a bound of its range, when values are strongly correlated (|ρ| ≥ 0.95: the data do not
   separate them) or when χ²ᵣ is well above 1; every value ± its 1σ error (covariance at the solution,
   scaled by χ²ᵣ); the **Solutions** of *Find the particle shape* or 1D Predict with their χ²ᵣ on these
   points (**Use This Solution** puts one into Model; after *Find the particle shape* the best is already
   there, and **Fit** refines it and gives its errors). **Save Data and Fit…** writes q (nm⁻¹ and Å⁻¹), I, σ,
   the model, the residuals and every term as CSV, with a JSON record of the model, the fit (errors,
   bounds, correlations) and the points next to it; **Save Plot…** (PNG or SVG) and **Save Model…** (JSON,
   to load for another curve). The log is folded at the bottom.

The model and the choices are kept for the next session, and the last curve opens again. A new installation
starts with a neutral model, not with the values of an earlier fit: set the particle and its sizes for your
sample, or let **Find the particle shape (no AI)** do it.

**fit1d.** In **Model**, add 1–4 **Sphere** / **Vertical cylinder** particles; repeated families are
allowed. In **Fit**, select **fit1d (network + physical fit)**: all populations in that one curve are
fitted together. Their composition comes from you; a good fit does not identify the true composition.
Choose **General network** (`cond_v2_ft4`) or **Au nanoparticles on Si (adapted)** (`cond_v2_ftr1`),
and **Auto** (default), Fast or Robust search. Set GISAXS / SAXS. For a Yoneda cut, enter the vacuum
**Cut qz** in nm⁻¹: it is not filled from Analyze automatically; unset means zero and adds a warning
when a sphere is present. SAXS ignores qz and has no structure factor or low-q term.

Only the particle families are taken from Model. Its parameter values, ranges, fixed parameters,
Distance D switches and global terms are not sent as fit1d constraints. The v2 convention uses
number-weighted truncated Gaussian sizes (relative widths), Hosemann spread w = w_rel·D, q in nm⁻¹
and lengths in nm; spheres on a Yoneda cut use |q| = √(q∥² + qz²). Its widths and forward curve differ
from the manual model.

The plotted total and **Terms**, residuals and **Results** use the original v2 result. Results show
χ²/point, native parameter values, errors, states and equivalent-solution ranges. CSV/JSON saved
through **Save ▸ Data and Fit…** preserves the native total, components, effective σ and warnings.
The selected systematic error floor is added to measurement σ in quadrature; without σ, fit1d
assumes Poisson errors and says so. Values copied into Model are only a manual starting point;
**Refine** subsequently uses the manual physics. Editing Model, selected points or fit1d settings
marks the saved fit stale; its table and χ² are labelled as previous results, the plot shows the manual
starting model, and exporting the stale fit is refused. Refit, or restore its conditions.

Install `gisaxs1d_gui_kit` under `utils/gisaxs1d_gui_kit`, or set `GISAXS1D_KIT_HOME` to its folder.
This single-curve method runs in a cancellable child process and needs no V5/TensorFlow assets.

**From Analyze.** *Send to Fitting* opens the cut here with the halves chosen in Analyze;
*Results ▸ Fit details ▸ Show in Fitting* puts that solution into Model — sizes and spreads converted,
the scales solved on the solution's own curve — and says how closely this model reproduces it (the same
curve for spheres and random cylinders; Fitting's vertical cylinder weights radii by R⁴, so it is loaded
as a start to refine).

Analyze's curve files, `<name>_fit_input.dat`, have four columns — q (Å⁻¹), intensity, σ and the number
of detector pixels averaged — and a header recording where the curve came from; any `.dat` / `.txt`
file with columns `q I [σ]` opens too. Fitting does not read `.csv` files, also not the CSV curves that
Analyze exports: use **Send to Fitting**, or open the `_fit_input.dat` written next to them (a `.csv`
dropped on Fitting is refused with that reason).

**In-situ series** fits every curve of a folder with the model of Single analysis (below).

### 1D Predict

**1D Predict** is the experimental neural model (Workflow V5): it proposes compositions and parameters for a
curve and corrects each proposal numerically against the measured points. Take its solutions as starts to
check and refine, not as results.

- **For the curve open in Fitting**: choose **AI proposal (1D Predict)** in the Fit step and press **Fit**;
  the solutions are listed in Results.
- **For a list of curve files**: **Tools ▸ 1D Predict — Fit Many Curves…** (or **Fit Many Curves…** under
  *Advanced* in the Fit step) opens the 1D Predict window. **Add curves…** lists text curve files (**Use
  current cut** takes the curve open in Fitting instead). **Parameters…** sets the fit method — *General V5
  (experimental)*, *Single RC specialist (experimental)*, *Physical fit (numerical)*, which uses no
  network, or *fit1d (network + physical fit)*, an external kit for spheres and vertical cylinders that
  needs the composition from you and the folder `utils/gisaxs1d_gui_kit` — the particle families when you know them, the q unit of the files and which half of q is fitted.
  **Fit curve** starts, **Cancel** stops, **Open results** opens the folder of the run. The table lists the
  candidates of every curve with their log RMSE and the stage that produced them; select one to see its
  fit, its parameters and units.

Each run writes a new folder `AI_Fitting_Output/v5_<date>_<time>/` in the folder GIMaP was started from
(for the portable version: the program folder): `request.json`, `batch_summary.json`,
`top20_candidates.json`, `fitting_run_summary.json` and, for every curve and half, `solutions.json`
(parameters, units, the stage that ran and why), `display.npz` and `fitting_curves.csv` (*General V5* also
writes `candidates.npz`).
The folders are not removed automatically. Fits you save yourself (**Save**) go where you choose.

Selecting a fit1d row in the batch window keeps its original v2 curve and components visible; a
manual model is copied only as a starting point. The batch window's fit1d search defaults to Fast,
while the main single-curve page defaults to Auto.

The model and its trained weights are in `modules/Fitting_1D_Model/Workflow_v5/`, both in the portable version
and in the Git repository, so 1D Predict runs from a source checkout too. Each network checks its files
against their checksums before it is used: with a changed or missing file *General V5* does not start, and
*Single RC specialist* goes on with its numerical fit and says why (section 13).
fit1d uses its separately installed kit and the two networks described above.

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

## 5. 2D Prediction (Labs)

**2D Prediction** (sidebar ▸ Labs) runs a trained prediction module on GISAXS detector frames. The page has
three steps on the left and the canvas on the right.

1. **Import data.** **Single file**: **Choose CBF frame…** (or drop a CBF frame on the canvas); **Stack**
   is how many consecutive detector files make one input. **Folder batch**: a folder of numbered frames,
   the **Range** of frame numbers and **Files per prediction**; a line under the fields says how many
   files and predictions that makes. Prediction reads CBF frames (`.cbf`) only.
2. **Import model.** Choose the **Prediction setup** — a module from `modules/` (section 12) — and press
   **Import model...**; the line below says whether the model is loaded. Folded: **Advanced model
   configuration** (the framework check, **Reload Config**, **Edit Config**) and **Advanced model sources**
   (the model library).
3. **Predict.** Three states — *Input*, *Model*, *Framework* — say what is still missing; **Predict** is
   enabled when all three are ready. A prediction runs in the background; **Stop** ends it.

The canvas shows **Input preview** once a frame is loaded and **Prediction result** after a prediction.
After a prediction, *Preprocessed* shows the steps of the module in order with the image each one produced
(click one to see it large). **Export input...** saves the frame as displayed; **Export result...** saves
the result as a picture, its data beside it, and a JSON record of the module, the model, the preprocessing
steps and the input file. A folder batch lists its results under **Batch results**, with filter, sorting
and export. The **Activity log** is folded at the bottom.

The settings of this page and of Trainset Build are the *Labs parameters* (**File ▸ Labs Parameters ▸
Load… / Save As…**).

## 6. Trainset Build (Labs)

**This feature is under development.**

**Trainset Build** (sidebar ▸ Labs) designs a simulated GISAXS training set (BornAgain), checks it locally
and prepares the training. Five steps on the left, each with its state:

1. **Dataset Design** — optionally load a measured image as a reference for the geometry (**Load file...**;
   it guides the beam centre, the region of interest and the masks and is never used as a training image);
   then **Geometry + ROI**, **Particle population** and **Sampling + files**. The detector, the region of
   interest and the mask are previewed on the right. **Validate design** checks the design.
2. **Local Preview** — simulated examples over the chosen parameter ranges, with the preprocessing
   (background, noise, masks and transforms under **Advanced preprocessing**).
3. **Model Design** — the training settings and the layers of the network.
4. **Local Run** — the job package, a local generation and training run with progress and **Stop safely**.
   Submission to the Maxwell cluster is disabled in this version.
5. **Monitor & Results** — the log, the metrics and the registration of the best model.

**Remember changes** keeps the design for the next session; **Reset defaults** restores the built-in
values (the note that follows offers Undo); **Load** and **Save** read and write a project file. In the
Chinese interface many texts of this page are still English.

## 7. Compare Page and Stages

Stages and Compare answer what a series of scattering frames needs — without labels and without AI: which
frames do not belong, where the series changes course, how fast, and which runs or samples are alike.
Everything uses Analyze's curves (the same geometry, masks and cuts). They replace the Classification page
of version 0.0.2.

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

1. **Series** — the list (double-click a name to rename it), Remove Selected, Remove All. A folder (one
   series) or curve files (together one series) can also be dropped on the page.
2. **Compare** — the q range compared (the range every series covers; narrow it to leave out a noisy edge or a
   detector artefact, **Whole Range** to go back), **Compare the shape only** (on by default), and how many of the
   last frames make a series' end state. Every change compares again within a second or two.
3. **Results** — groups (three or more series, split where the end states clearly separate), the series that
   differs most at the end and at the start, and per series: frames, odd frames, stages, the frame by which half
   and 90 % of its change had happened; a table of how different the series are, at the end or at the start (in
   percent of intensity).

On the right: how far each series has changed (against frame, or the share of each series for runs recorded at
different rates), their paths through the two main changes, and their end states. **Save** writes a table of every
series or of every frame (CSV with a JSON record) and the plots; the tables on the page can be copied
(`Ctrl+C`). A project keeps the series (curve files by path, Analyze maps in `<project>.compare.npz`).

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

   Example (frames of a 9-megapixel Lambda detector, peak fitting on): about 2.5 s per frame one at a time,
   and less than 1 s per frame with *Balanced* (4 at once).
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

### Process with AI

**Ask AI…** in Analyze's command bar (or **Tools ▸ Process with AI…**, `Ctrl+Shift+L`)
lets an AI analyse the frame shown in Analyze — GIWAXS or GISAXS — with the same tools you use: it
switches the mode, adjusts sectors and cuts, sets the I(χ) window, fits peaks or the horizontal cut and
exports files, and the page follows every step on screen. The panel on the right shows a summary of the
AI's reasoning, each tool call (hover for its arguments and result) and, at the end, the report. (The
text below says Claude; the other providers work the same way.)

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

1. Open a frame, then click **Ask AI…** and choose:
   - the results. For a GIWAXS frame: *Peak table* (q, d, FWHM, intensity, significance), *Orientation*
     (in-plane versus out-of-plane intensity of each peak), *Orientation distribution of one ring* (I(χ),
     maxima, Herman's orientation factor; pick the ring or let Claude choose), *Crystallite size*
     (Scherrer coherence length). For a GISAXS frame: *Horizontal cut at Yoneda, centred, halves
     chosen*, *In-plane distance 2π/q\**, *Fit of I(qy): shape, radius, distance*;
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
(*run_standard_pipeline*, the same as Run Automatic Analysis. GIWAXS: geometry, the last frames of a
series, peaks, in-/out-of-plane, ring orientation and sizes of the strongest reliable peaks and of the
ring you asked about. GISAXS: the Yoneda cut, the symmetry axis, the halves, the spacing and the fit.
Each decision comes with its reason). Its choices are defaults, not limits: Claude then follows what your
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
first — e.g. *“the calibration is D:\beamtime\calib\img_0005.cbf, 12.4 keV”*. It also searches the
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
agents (Codex reads `AGENTS.md`, which points to `docs/agents/giwaxs-playbook.md` and
`docs/agents/gisaxs-playbook.md`):

```bash
python tools/gimap_agent.py auto sample_A_00001_m01.nxs sample_B_00001_m01.nxs --notes "alpha_i = 0.2 deg" --out results
```

`auto` performs the standard procedure of Run Automatic Analysis for every frame. The technique is
detected once the geometry is applied — a detector that reaches beyond 2θ = 20° is GIWAXS, otherwise
GISAXS — and `--technique giwaxs|gisaxs` forces one.

- **GIWAXS**: geometry, the last ten frames of a series summed, peaks, in-/out-of-plane, ring orientation
  and sizes of the strongest reliable peaks.
- **GISAXS**: geometry, the horizontal cut at the Yoneda band, the beam-centre column from the symmetry of
  the cut, the halves, the in-plane distance 2π/q* and the numerical fit of I(qy) (`--no-fit` prepares the
  cut only).
- **Geometry** is taken, in this order, from `--calibration` (a `.poni`, a GIMaP calibration `.json` or an
  image of a standard), a calibration file the notes name, a saved instrument profile, and a search around
  the frame. `--no-saved-profiles` ignores the saved profiles and `--recalibrate` calibrates even when a
  saved profile exists. A saved profile is matched by detector name and frame size only, and the search
  can pick the calibration of another experiment: the report says which file was used and, for a file
  the search found, its full path and where it lies seen from the frame (the decision *calibration found
  by the search*) — check it.

For every frame it writes a `report.md` with each decision and its reason (**Decisions**) and what only
you can answer (**Needs attention**), the curves as CSV and the q map, and a `summary.md` for the batch
(later frames of the same size reuse the first calibration). Results go to `--out`; without it they go
to the user data folder (`assistant_runs/cli/`). Nothing is written next to the data, and saved
instrument profiles are read but not changed.

| Option | What it sets |
|---|---|
| `--notes "…"`, `--notes-file FILE` | your own notes: αi, the energy (keV) and the pixel size (µm) are read when the notes give exactly one value; a calibration file they name is tried before the search, and the calibrant when they name exactly one |
| `--incidence-deg`, `--energy-kev`, `--pixel-size-um` | αi, the energy and the pixel size; an option takes precedence over the notes, the file header and the profile |
| `--calibration FILE`, `--standard NAME` | the calibration, and the standard shown in calibration images (`agbh`, `lab6`, `ceo2` or `lab6_ceo2`) |
| `--frame N`, `--sum K` | the frame of a series (1 = first, −1 = last) and how many frames are summed (default: the last 10) |
| `--rings N` | how many of the strongest reliable peaks get orientation and size (GIWAXS) |
| `--technique`, `--no-fit`, `--no-saved-profiles`, `--recalibrate` | as described above |
| `--out FOLDER` | where the results go |

Exit codes of `auto`: **0** — every frame was analysed and nothing is open; **2** — at least one frame
has a question under *Needs attention* (a missing value with the option that supplies it, a judgement
such as the choice of the GISAXS model, or no geometry found so that nothing was analysed); **1** — at
least one frame failed, for example a file that cannot be read (the other frames still have their
reports; 1 wins over 2).

`auto` is a baseline, not the whole analysis: `status`, `find-calibration`, `tools` and `call` (a JSON
list of tool calls) give finer control, and `mcp` serves every tool over MCP stdio for Codex or Claude
Code (`open_frame`, then `run_standard_pipeline` and the others on the same frame). The playbooks
describe both levels — the baseline every agent runs, and the questions a capable agent follows up —
and `tools/eval_giwaxs_agent.py` evaluates them on a GIWAXS case (`baseline`: what the procedure must
get right; `report`: mistakes and findings in an agent's written report). A case is a JSON file you write
for a measurement you know well and put in `docs/agents/eval/local/`, a folder Git ignores, so a case on
your own data stays on your computer (`docs/agents/eval/README.md` describes the format); the repository
ships none, and the tool says so when it finds no case.

## 9. Geometry Calibration Tool

Open **Tools > Geometry Calibration...** (`Ctrl+Shift+G`) to calibrate a SAXS, GISAXS, or GIWAXS detector geometry without leaving the current page.

1. Open an image of a calibration standard (`.nxs`, `.cbf`, `.tif` / `.tiff` or `.edf`), or paste its path into the image field and press Enter.
2. Confirm the detected energy and pixel size. For a CBF with matching scan NXS metadata, the energy is filled automatically. Enter the energy in keV if it is still missing.
3. Confirm the detector model. If it cannot be identified, choose a known detector from **Detector model**, or select **Custom pixel size** and enter the values in **Advanced configuration**.
4. Choose the standard. When the path of the image — its file name or a folder above it — names one standard (for example `AgBh`, `LaB6` or `CeO2`), it is selected for you; when it names more than one, the field is set to **Auto Detect**; when it names none, the field is left as it was (**Auto Detect** when the window opens). Auto Detect tries every known standard. **Estimated distance** is optional and steers the search: leave it on *Optional* when you do not know the distance, type the approximate distance when you do. When an image is opened the field is filled in only if the image file states a distance (a NeXus file can; the distance in a CBF or EDF header is not used) or, failing that, if a calibration was applied before, with the distance of that calibration: check that such a number is the distance of this set-up. An approximate distance helps to resolve harmonic alternatives, a wrong one can mislead (see the limitation below).
5. Click **Auto Calibration**. The calculation runs in the background and can be cancelled.
6. Review the center, distance, residual, confidence, high-contrast overlay legend, and alternative candidates. **Clean image** temporarily hides all calibration overlays, **Reset view** restores the complete detector mosaic after zooming or panning, and **Focus image** hides the result panels to give tall WAXS mosaics more room. The horizontal divider can also be dragged.
7. To adjust the result by hand, press **Manual refine**: drag the center marker, edit the center/distance values, or pair a detected ring with a theoretical peak; **Reset to fitted** goes back to the fitted solution and **Finish manual** closes the panel. The fitted values stay as they are until you press **Manual refine**.
8. Click **Apply** to save the geometry as the instrument profile of the detector; one note says what was
   applied and which profile Analyze will use. Calibration results can also be exported to or imported from
   JSON (the note after Export opens the folder). An image or an exported `.json` can also be dropped
   on the window; the file dialogs start in the last folder used.

Solid yellow overlays are matched theoretical rings, dashed orange overlays are unused theoretical rings, and dotted cyan overlays are detected experimental radii; the preview legend identifies each style. Partial WAXS arcs and centers outside the active detector area are supported. A low-confidence result or a one-ring result should be treated as ambiguous and reviewed manually.

The beam centre is shown and entered in the same pixel coordinates as in Analyze: pixel corners lie on
whole numbers, so the centre of the first pixel is (0.5, 0.5); x counts from the left edge and y from the
top edge. A centre can be compared or copied between this window and Analyze as it is. An exported
calibration file keeps its numbers as in earlier versions: there the beam centres count pixel centres from
0, half a pixel lower than the window shows, and the file says so (`metadata.center_convention`). Importing
the file shows the same centre again.

**Known limitation: a wrong solution can come first.** Both cases below were seen on a silver behenate image
whose path does not name the standard.

- With the standard left on **Auto Detect** and no estimated distance, a solution of another standard can be
  ranked first: a LaB₆ solution at a much shorter distance came first, and the silver behenate solution was
  last in the list of candidates. Choose the standard yourself when you know it. An approximate distance
  helps Auto Detect too: with 1500 to 1800 mm typed in, the silver behenate solution came first on this
  image.
- A wrong estimated distance can mislead, also with the right standard: the search prefers solutions near
  the value in **Estimated distance**. With silver behenate chosen and 2000 mm typed in, a solution near
  1830 mm came first, although the real distance is about 1630 mm; with the field left on *Optional*, the
  right solution came first.

**Estimated distance** can hold a number you did not type. It is never filled with a built-in value: on a
new installation it stays on *Optional*. But each time an image is opened it takes the distance the image
file states, or else the distance of the calibration applied last, also when you had typed another value or
cleared the field; with neither, a value you typed stays. The calibration applied last may be the one of
another set-up or detector. Enter the approximate distance of your set-up, or set the field to 0 (it then
shows *Optional*) when you do not know it.

Before you press **Apply**, check that the distance is the one of your set-up and that the drawn rings lie on
the rings of the image. This will be corrected in a later version.

## 10. Format Converter

Open **Tools ▸ Format Converter…** (`Ctrl+Shift+C`) to write detector frames in another format;
**Tools ▸ Convert Current File…** opens it with the file shown in the active workspace. The sources are
only read, never changed. The window has three steps; **Next** and **Back** move between them.

1. **Input** — **Add files**, **Add folder** or **Use currently opened file** (files and folders can also
   be dropped on the window). The converter reads NXS, CBF and TIFF; single images and multi-frame files
   are recognised, and for a NeXus file with several datasets you choose the dataset.
2. **Configure & Preview** — tick the sources to convert (**Select all**, **Select none**, filter by
   file name, **Sort by filename**, **Remove selected**). For multi-frame NXS files, **Advanced frame
   selection** offers *All*, *Current frame*, *Frame range*, *Custom* (for example `1, 5, 8–20`) and
   *Every Nth frame*. The preview shows the first, middle and last frame with their size, data type and
   value range.
3. **Run, Results & Export** — the **Output format**: TIFF, CBF, HDF5 or NumPy (`.npy`); the
   **Destination** (by default a folder `converted` next to the first input) and the file-name template
   for multi-frame sources. **Advanced output options** hold the data type (the original values and type
   are kept by default), the metadata JSON written beside the images, and one HDF5 container for all
   images. **Review & Convert** shows what will be written and its size before it starts.

While it runs, a window shows the progress with **Pause** and **Cancel** (the image being written is
finished first); afterwards **Open output folder** and **View report**.

EDF frames are not an input of the converter: with an EDF frame shown, **Convert Current File…** opens the
window without it and says why under the input list. To convert EDF frames (or any frame Analyze opens, as
Analyze reads it: modules stitched, frames summed), use **Analyze ▸ Batch Export ▸ Converted detector
frames** (TIFF, EDF, NumPy or HDF5).

## 11. XRR Series Extractor

Open **Tools > XRR Series Extractor...** (`Ctrl+Shift+R`) to obtain an XRR curve from a
GIWAXS/GISAXS detector angle series without leaving the current workspace.

1. Select one NXS detector-module file, or select a CBF file/folder and glob pattern. For NXS, the
   sibling module files are stitched for each detector frame; for CBF, each naturally sorted file is
   one point.
2. Choose a linear sample-angle sequence (**Theta start** and **Theta step**) or enter an NXS motor
   dataset containing one theta value per frame.
3. Enter detector distance, beam energy, pixel sizes and the direct-beam center. The window starts
   from the last applied Geometry Calibration (marked *from last calibration*; only its energy when that
   calibration was of frames of another size). **Load first frame** never overwrites a value you typed;
   a tag beside each field and the summary say which values come from the file, the last calibration,
   the image centre or the built-in defaults. Use **Pick direct-beam center** to choose the center with
   the mouse. Select whether the specular reflection moves up or down in detector-y. A file or folder
   can also be dropped on the window.
4. Choose an ROI radius and `Sum` or `Mean`. Radius 0 reads one pixel; larger values integrate a
   circular neighborhood around the calculated specular beam.
5. Click **Run extraction**. The worker loads one full frame at a time, displays the current detector
   and ROI, records one scalar point, and then releases that frame. It does not load the full series
   into memory.
6. Switch freely between **Live frame** and **XRR points**. Processing updates both but never changes
   the selected tab. Use **Export CSV…** to save theta, qz, intensity, ROI position and valid-pixel
   count (by default `<source folder>/<source name>_xrr.csv`). A `.json` record beside it keeps every
   value that determines qz (series, θ, distance, energy, pixels, centre, direction, ROI) and where
   each came from.

The current extractor reports the raw ROI sum or mean. It does not apply monitor/flux normalization,
footprint correction, background subtraction or resolution correction.

The direct-beam centre is shown and entered in the same pixel coordinates as in Analyze and Geometry
Calibration: pixel corners lie on whole numbers, so the centre of the first pixel is (0.5, 0.5); x counts
from the left edge and y from the top edge (hover over the centre fields or the *ROI x* / *ROI y* headers
to read this in the window). A centre can be copied between the three windows as it is. The preview, its
cross and ROI circle, and the *ROI x* / *ROI y* columns use the same pixels, and a centre picked with the
mouse is the point clicked: with radius 0, the pixel you click is the pixel that is read. The image centre
the window falls back to (step 3) is the middle of the frame, (columns / 2, rows / 2).

The exported files count differently: `roi_center_x_px` / `roi_center_y_px` of the CSV and
`geometry.beam_center_x_px` / `beam_center_y_px` of the JSON record count pixel centres from 0, so they are
0.5 px lower than the window shows, and the record says so (`center_convention`). If you noted a centre
from this window in a copy of GIMaP from before 2026-10-13, add 0.5 px to x and y: the window counted
pixel centres from 0 then.

**Known limitation: a beam centre from a NeXus header.** When a NeXus file states a beam centre, the window
takes the two numbers as column and row of the frame it shows, without the turn (transpose and flip) the
frame gets when it is loaded, so the centre can lie away from the direct beam (Analyze applies the turn to
the same header). The fields are then marked *from file*, and the value replaces the centre of the last
calibration. Before you run the extraction, check on the preview that the cross lies on the direct beam, or
use **Pick direct-beam center**. CBF series are not affected.

## 12. Model Configuration

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

The V5 methods of 1D Predict (section 4) use their bundled model in
`modules/Fitting_1D_Model/Workflow_v5`; fit1d uses its installed `gisaxs1d_gui_kit` instead.

## 13. Troubleshooting

### File Cannot Be Loaded

The message under the image says why in words (empty file, not a detector image, damaged or
incomplete NeXus, no permission). Otherwise, check that the file format is supported by the active loader. For detector formats such as `.cbf`, confirm that `fabio` is installed. For HDF5/Nexus-like files, confirm that `h5py` is installed and that the internal dataset path matches what the loader expects. If support for a custom file format is required, please contact yufeng.zhai@desy.de.

### Model Not Imported

Check the selected module configuration and model path. If the path points to a local machine-specific location, update it for the current computer.

### Predict Button Disabled or Prediction Fails

In 2D Prediction the three states beside **Predict** say what is missing: *Input* (a CBF frame or a folder), *Model* (press **Import model...**) and *Framework* (the framework the module needs is installed).

### 1D Predict Does Not Start

*General V5* checks its model files (`modules/Fitting_1D_Model/Workflow_v5/`) against their checksums before it runs, and does not start when one is missing or was changed: the message names the file. *Single RC specialist* does not stop in that case; it goes on with its numerical fit and gives *Fast model unavailable* as the reason of the solution. The portable version and the Git repository both contain the files. In a source checkout:

- A checkout last updated before the weights were added does not have them yet: update it (section 2, *Updating a source checkout*).
- A checkout made on Windows before 2026-10-13 holds some of these files with converted line endings, also after it was moved to the new `main` with the two plain commands, and `git status` does not show it. Run `tools\update_source.ps1`: it writes these files again, also when there is nothing new to download.
- A model file that you changed yourself is left alone by the script; `git status` lists it as modified. `git checkout -- modules/Fitting_1D_Model/Workflow_v5` puts the files of the repository back (and discards your changes in that folder).

The numerical methods of Fitting (*Refine*, *Search*, *Find the particle shape*) use no network: the check does not concern them.

fit1d needs its own kit under `utils/gisaxs1d_gui_kit` (or `GISAXS1D_KIT_HOME`); its main single-curve
route does not require V5/TensorFlow weights.

### A `.csv` Curve Does Not Open in Fitting

Fitting reads `.dat` and `.txt` curves. For a curve from Analyze use **Send to Fitting**, or open the `_fit_input.dat` written next to the CSV.

### TensorFlow / PyTorch Not Available

`requirements.txt` asks for TensorFlow 2.15–2.16 (`tensorflow-intel` on Windows); 2.15.1 is the tested version. PyTorch is not a dependency. If a module requires another framework, install and configure it separately and confirm that loader code exists.

### Windows Warns About the Installer

The installer and the launcher are batch files (`.bat`) and are not signed, so Windows may ask for confirmation before it runs a downloaded one, and an institution's antivirus or application control may block it. Only run files downloaded from the GitHub Releases page of GIMaP. If you do not want to run them, run GIMaP from source in a local Python environment instead (section 2).

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

## 14. FAQ

### Is GIMaP production-ready?

No. Version 0.1.0 is a pre-release: treat it as beta scientific software and check results that matter.

### Can I use my own trained model?

For 2D Prediction, yes, if it can be represented by a compatible module configuration and supported loader.
Add or edit a `module.yaml` under `modules/` (section 12) and make sure preprocessing and output definitions
match the model. V5 uses its bundled model; fit1d offers General and Au nanoparticles on Si networks in its settings.

### Where are the results of 1D Predict saved?

In `AI_Fitting_Output/v5_<date>_<time>/` in the folder GIMaP was started from (for the portable version: the program folder); see *1D Predict* in section 4. Fits you save yourself (**Save**) go where you choose.

## 15. Version and Contact

This manual describes **GIMaP 0.1.0** (pre-release). **Help ▸ About GIMaP** shows the version of the
program you are running. `CHANGELOG.md` in the program folder lists what changed since 0.0.2 — the pages
that were replaced, where the settings are kept now — and the known limitations of this version.

Contact:

[yufeng.zhai@desy.de](mailto:yufeng.zhai@desy.de)
