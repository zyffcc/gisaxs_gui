# GIMaP

GIMaP (Grazing-Incidence Mapping Package) is a desktop application for GISAXS/GIWAXS data visualization, analysis, fitting, and machine-learning-assisted workflows. It is built with PyQt. The current version is 0.1.0, a pre-release: see [CHANGELOG.md](CHANGELOG.md) for what changed since 0.0.2 and for the known limitations.

## Overview

GIMaP provides a single desktop interface for working with grazing-incidence scattering data. **Analyze** reduces GISAXS and GIWAXS detector frames to curves (exact q, instrument profiles, in-situ series); **Fitting** fits those curves with particle models, one curve or a whole series; **Compare** sets runs or samples side by side; the **Labs** pages hold trained-model prediction and trainset generation.

The application entry point is `main.py`. The user manual is [docs/User_Manual.md](docs/User_Manual.md), in Chinese [docs/User_Manual_zh.md](docs/User_Manual_zh.md); **Help ▸ User Manual** (`F1`) opens the edition of the interface language from the program, in your browser.

## Key Features

- **Analyze** (GISAXS and GIWAXS): open CBF, TIFF, EDF or NeXus/HDF5 frames (`.nxs`, `.nx`, `.nx5`,
  `.h5`, `.hdf5`, `.hdf`) or a whole folder and get the reduced curves without further clicks. Files
  that belong together are one source — the module files of a P03 Lambda detector are stitched, a
  master is read with its data files — and every frame of a multi-frame file is a frame of the series.
  A step rail leads through Data → Geometry → Mask & corrections → Cuts → Results → Export; every
  step shows its state.
  - Geometry from a calibrant (AgBh, LaB₆, CeO₂ …) or entered once and kept as the instrument profile
    of the detector; exact q (Å⁻¹), beam centre = direct beam.
  - Masks (detector gaps, mask files, drawn shapes, value limits, isolated hot/dead pixels removed per
    frame), background subtraction, optional mirror filling; optional GIWAXS intensity corrections
    (solid angle, polarisation, film absorption, with propagated errors); series alignment and
    normalisation to a reference peak. Undo / Redo of every set-up change (Ctrl+Z).
  - GISAXS: Yoneda and beam-centre cuts; GIWAXS: I(q), in-plane / out-of-plane sectors, q boxes and
    I(χ). Custom cuts: click a peak (ring, sector or spot), type centre ± width, or draw on the cake;
    Save / Load Cuts.
  - Curves can be shown as +q, −q, both halves or |q|; every curve shows its sources. Box zoom
    (Shift+drag) on every image and plot.
  - **Run Automatic Analysis** (no AI) — with progress, Stop, and the finished part kept:
    - GIWAXS: peak table with fitted positions, widths and uncertainties, orientation, crystallite size;
    - GISAXS: the horizontal cut at the Yoneda band, the beam-centre column from the symmetry of the
      cut, the halves (averaged when they agree, otherwise the usable one, with the reason), the
      in-plane distance 2π/q* and a numerical model fit (spheres, vertical and random cylinders with
      size spread and distance).
  - **Series**: heat maps and peak tables of in-situ series, drawn at the true q (or χ), with the frame axis
    in frame numbers (or time); every frame is cut where the frame on screen is, unless you have the
    cut found again in every frame.
  - **Batch Export**: from raw frames to a folder in a few clicks — curves, fit input, q maps, cake and
    detector images, converted frames (TIFF, EDF, NPY, HDF5), optional peak fitting of every frame
    (start from the previous frame, the first frame or fresh), CSV/TXT/DAT and PNG/TIFF/SVG/PDF, a
    README of the folder and a JSON record of the settings. Several frames are reduced at once in
    low-priority processes (sized to the cores and free memory; Gentle / Balanced / Fast), with Pause,
    Stop and the frames appearing live in the Series map.
  - Remembers the recent data (File ▸ Open Recent, Start page) and offers the set-up of the last session.
- **Fitting**: a step rail (Curve → Model → Fit → Results) over the controls; fit 1D curves from
  Analyze (or a `.dat` / `.txt` curve file) with sphere, random-cylinder and vertical-cylinder
  models. One **Fit** button runs the chosen method: refine the current values, search the ranges,
  find the particle shape numerically (no AI), or ask the experimental neural model (1D Predict).
  Results give every value ± 1σ and warnings. **In-situ series** fits every curve of a folder.
- **Stages and Compare** (no AI): a series map in Analyze shows where the curves change course
  (stages), which frames do not belong (odd frames: a detector glitch, a shutter) and what changes between
  stages; Compare puts several runs or samples side by side — how fast each changed, where each ended,
  which are alike.
- **Projects** (`.gimap`): **File ▸ Save Project** keeps a sample as it was left — the frames and
  set-up of Analyze, the curve and model of Fitting, the In-situ series and the Compare series (data
  are referenced by path, not copied); **Open Project…**, **Open Recent** or a drop reopens it.
- **Drag and drop**: detector frames, folders and `.gimap` projects can be dropped anywhere on the
  window; curve files on the Fitting page; folders or files of curves on the Compare page.
- **Tools** (Tools menu): **Geometry Calibration** (beam centre and detector distance from an image
  of a standard), **Format Converter** (NeXus/HDF5, CBF and TIFF frames to TIFF, CBF, HDF5 or NumPy;
  **Convert Current File…**), **XRR Series Extractor** (an XRR curve from a detector angle series),
  **1D Predict — Fit Many Curves…** and **Settings**.
- **Process with AI** (optional): an AI (Claude, DeepSeek, Qwen, OpenAI or a local model) runs the
  Analyze tools on the open frame while you watch; its changes come back as cards you can preview,
  apply or undo, and it says why anything could not be determined. Claude runs on a Claude Pro/Max
  plan through a local Claude Code (no API key) or on the API with a key.
- **Without the window**: `tools/gimap_agent.py` runs the automatic analysis and the single analysis
  tools from the command line and serves them over MCP to agents such as Codex or Claude Code (see
  [Command Line and MCP](#command-line-and-mcp)).
- **Labs**: 2D Prediction with trained models (single files or folder batches) and Trainset Build
  (simulated GISAXS training sets with BornAgain).
- **Interface**: English or Chinese (**View ▸ Language**), light or dark theme, adjustable font size;
  **Help ▸ User Manual** (`F1`) opens the manual in your browser; unexpected errors are logged
  (`<user data>/logs/errors.log`) and shown without closing the application.

## Installation

### Windows: the portable version

No Python is needed; the package brings its own.

1. Open https://github.com/zyffcc/gisaxs_gui/releases and choose the latest release.
2. Download `00_Download_and_Install_GIMaP-<version>.bat` into an empty folder with a short path
   that you may write to (for example `C:\GIMaP`).
3. Double-click it. It downloads the package parts, verifies each one (SHA-256), joins and extracts
   them into the folder `GIMaP-<version>-windows-x64-portable` beside the installer, and starts
   GIMaP once. Windows may ask for confirmation before it runs a downloaded `.bat` file.

**Starting GIMaP later**: double-click `Start_GIMaP.bat` in the folder
`GIMaP-<version>-windows-x64-portable` (a desktop shortcut to that file is convenient). The first
start prepares the bundled Python environment for the folder, which takes a moment.

**Updating**: run the installer of the new version in the same way. It extracts into its own
`GIMaP-<new version>-…` folder; start the new `Start_GIMaP.bat` and delete the old folder when you
no longer need it. From 0.1.0 on, the settings, the instrument profiles, the recent files and the
fitting model parameters are kept outside the program folder, in `%APPDATA%\GIMaP`, so an update
keeps them. Version 0.0.2 kept its settings inside its own program folder: they are not carried
over, and 0.1.0 starts with the defaults.

### Windows: a source folder with one double-click (no Python needed)

For a clone of the repository (`git clone https://github.com/zyffcc/gisaxs_gui.git`) or a ZIP of it
(**Code ▸ Download ZIP**, extracted to a short path that you may write to):

1. Double-click `Start_GIMaP.bat` in the folder. The first start sets up a private environment in
   `.gimap\` inside the folder: it downloads uv (a small Python installer, pinned version, checked
   against its SHA-256), Python 3.10 and the packages of `requirements.txt` - a few GB, 5-20
   minutes. Then GIMaP opens. Later starts are immediate.
2. To update a clone, double-click `Update_GIMaP.bat` (it runs `tools/update_source.ps1`, which needs
   Git); the next `Start_GIMaP.bat` adds new packages first when the update needs any.

Nothing outside the folder is changed; delete `.gimap\` to remove the environment. The Sputter
workspace needs its model toolkit, research code that is not part of the repository: copy the toolkit
folder you were given to `utils\sputter_toolkit`, and the next `Start_GIMaP.bat` adds PyTorch for it
(`requirements-sputter.txt`); nothing else needs setting up. In a portable folder (with `runtime\`)
`Start_GIMaP.bat` behaves as before.

### Running from Source

Recommended source environment: Windows with Python 3.10 or 3.11.

```powershell
cd gisaxs_gui

python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python main.py
```

If PowerShell blocks activation in the current terminal session:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Conda can also be used if preferred:

```powershell
conda create -n GUI python=3.11 -y
conda activate GUI
python -m pip install -r requirements.txt
python main.py
```

The trained weights of 1D Predict are in the Git repository (about 56 MB), so a checkout runs
everything, including [1D Predict](#1d-predict-experimental-neural-fitting).

#### Updating a source checkout

In the folder of the checkout, run:

```powershell
powershell -ExecutionPolicy Bypass -File tools\update_source.ps1
```

`tools/update_source.ps1` updates the branch that is checked out from GitHub (`origin`) and says
what it found and what it did:

- Nothing new: nothing is changed. New commits only: the branch moves forward to them.
- After the history was rewritten (see below): the state you have is first kept as a branch
  `backup/<date>-<time>-<commit>`, then the branch is moved to the new history.
- Your edits of tracked files are put aside (`git stash`) and put back on top of the new version;
  a copy stays in the stash. When they do not fit the new version, the new version is left
  complete and the script says how to get them back.
- Files Git does not track or ignores — your data, results, `AI_Fitting_Output/` — are not
  touched, and the settings and instrument profiles are outside the checkout anyway
  (`%APPDATA%\GIMaP`). When the new version brings a file to a place where you have a different
  file of your own, the script stops before it changes anything.
- Model files of 1D Predict that an older checkout holds with converted line endings are written
  again as the repository has them, also when there is nothing new to download. 1D Predict
  checks these files byte for byte, and Git does not notice the difference on its own.
- When Git cannot replace a file because another program has it open (GIMaP itself, an editor),
  the version you had is put back whole, with your edits where they were: close the program and
  run the script again.
- When the list of Python packages changed, it says so: run `python -m pip install -r
  requirements.txt` then.

Add `-DryRun` to see what an update would do without changing anything. At the end the script
prints how to go back to what you had, and how to delete the backup once things look right. The
script needs Git 2.16 or newer and says so when it finds an older one.

A checkout that does not have the script yet (one last updated before 2026-10-13) is first moved
to the new `main` with two plain commands, with `main` checked out:

```powershell
git fetch origin
git reset --hard origin/main
```

Then run the script once, with the command at the top of this section. On Windows the two
commands leave some model files of 1D Predict with the line endings the older checkout had given
them, and 1D Predict (*General V5*) does not start until the script has written them again. From
then on, update with the script. (On Linux or macOS, where Windows PowerShell is missing, nothing
was converted, and `git pull` works again after the two commands.)

`git reset --hard` discards your edits of tracked files, and commits you made yourself are no
longer on the branch. To keep them, run `git stash` (edits) or `git branch my-work` (commits)
first; afterwards `git stash pop` puts the edits back, and `git cherry-pick <commit>` brings a
commit over. Do not merge or push `my-work` itself: it carries the old history. Files Git does
not track stay, with one exception: where the new version has a file at the same place — for
example weight files of 1D Predict that you had copied into
`modules/Fitting_1D_Model/Workflow_v5/` by hand — the file of the new version replaces yours
without asking.

To keep all of that instead, let the script do the first update: `git fetch origin`, then
`git checkout origin/main -- tools/update_source.ps1`, then the script. It keeps the state you
have as a backup branch, puts your edits back, stops before it would replace a file of yours, and
writes the model files again in the same run.

A checkout of 0.0.2 that was used for fitting has one edit you did not make yourself:
`config/model_parameters.json`, where 0.0.2 kept the model of Fitting. Its values are not carried
over: 0.1.0 keeps the model in the user data folder and starts with a neutral one. The script
reports the file as an edit that does not fit the new version and ends with exit code 2: the
new version is complete, and what you had changed waits in the stash. When this file was your
only edit, delete that copy with `git stash drop`. Do not run `git stash pop` for it, although
the script names that command for edits you want back: it would write conflict markers into the
file, which is the model a new user of Fitting starts with. (With other edits in the same
stash: `git stash pop`, then `git checkout HEAD -- config/model_parameters.json` to put the
file of the new version back, then `git stash drop`.)

The history of `main` was rewritten on 2026-10-13 (and once more on 2026-10-21, after the removed commits had been merged back from an old clone; `tests/test_release_history.py` now fails on a branch that holds them). A clone last updated before that day can hold
commits that are no longer on GitHub: `git pull` then stops with an error, or tries to merge the
old history with the new one. Do not merge, and do not push from such a clone; move it to the new
`main` as described above (or clone again).

#### BornAgain on Windows and macOS

GIMaP requires BornAgain 24.1. On Windows (and Linux), BornAgain provides a PyPI
wheel for supported Python versions, so `requirements.txt` installs it directly.

BornAgain does not publish a prebuilt macOS wheel on PyPI. On macOS, first install
the official Homebrew formula:

```bash
brew tap mlz/homebrew https://jugit.fz-juelich.de/mlz/homebrew/
brew install mlz/homebrew/bornagain@24.1
bornagain_info
```

The Homebrew wheel must have the same CPython ABI tag as the GIMaP environment:
for example, a `cp314` wheel cannot be installed into Python 3.10. Do not force an
ABI-mismatched wheel; use a matching project wheel or build BornAgain 24.1 against
the same Python interpreter. See [the development guide](docs/development.md) for
the complete platform-specific setup and verification steps.


### Screen size

GIMaP follows the display scaling of the operating system and its layout fits a 1280 × 720 screen
(the smallest window is 1024 × 640); dense panels scroll instead of being clipped. A 1080p or
larger screen is more comfortable for the detector image beside the curves. If the text is too
large or too small, use **View ▸ Font Size**.

## Basic Workflow

1. Start the application (`Start_GIMaP.bat`, or `python main.py` from source).
2. On the **Start** page, drop detector images or a folder, or pick one of the recent ones; or use
   **File ▸ Open Data…** / **Open Folder…**.
3. **Analyze** shows the frame and its curves at once. The first time, calibrate or enter the
   geometry (it is saved as the instrument profile of the detector); check the mask and the cuts.
4. **Run Automatic Analysis** for the peak table and orientation (GIWAXS) or the Yoneda cut, the
   spacing and a model fit (GISAXS), or **Batch Export** to reduce a whole folder to the files you
   choose.
5. **File ▸ Save Project** keeps the sample as you leave it.

## Analyze → Fitting Workflow

1. In **Analyze**, **Send to Fitting** opens the horizontal cut in **Fitting** (its menu chooses the
   halves: both on |qy|, their mean, or one side); **Send Series to Fitting…** exports every frame
   into a folder `<name>_fitting/` and opens **Fitting ▸ In-situ series**, which fits every frame on
   the mean of both halves or on one half (**Halves of the cut**), never on both together.
2. Fitting has four steps: **Curve → Model → Fit → Results**. A first curve opens on **Fit** with
   **Find the particle shape (no AI)**: sphere, random cylinder and vertical cylinder are fitted
   numerically from several starts and the solutions are listed.
3. For full control, edit the particle cards in **Model** (tick **fit** for the free values,
   **Ranges** for their bounds) and choose **Refine the current values** or **Search the ranges,
   then refine**; **AI proposal (1D Predict)** asks the experimental neural model. One **Fit**
   button (Ctrl+Return) runs the chosen method; **Stop** keeps the best values so far.
4. **Results** gives χ²ᵣ, every value ± 1σ and warnings (a value at a bound, strongly correlated
   values). **Save ▾** writes the data and fit (CSV with a JSON record), the plot and the model.

Fitting shows q in nm⁻¹ (Analyze uses Å⁻¹; the Curve step shows both). Fitting opens `.dat` and
`.txt` curves; the `.csv` curves that Analyze exports are not read by Fitting — use **Send to
Fitting**, or the `<name>_fit_input.dat` file written next to them.

## 1D Predict (experimental neural fitting)

1D Predict proposes particle models for a 1D curve with the bundled Workflow V5 model
(`modules/Fitting_1D_Model/Workflow_v5`) and corrects each proposal numerically against the
measured points. It is experimental: take its solutions as starts to refine and check in Fitting.

- **One curve**: Fitting ▸ Fit ▸ **AI proposal (1D Predict)**, then **Fit**. The solutions are
  listed in Results; **Use This Solution** puts one into Model.
- **Many curves**: **Tools ▸ 1D Predict — Fit Many Curves…** (also Fitting ▸ Fit ▸ Advanced ▸
  **Fit Many Curves…**): **Add curves…** or **Use current cut**, **Parameters…** (the fit method —
  *General V5 (experimental)*, *Single RC specialist (experimental)* or *Physical fit (numerical)*,
  which uses no network — the particle families, the q unit of files that state none in their header and the halves), **Fit
  curve**, **Cancel**, **Open results**.
- **Output**: every run writes a new folder `AI_Fitting_Output/v5_<date>_<time>/` in the folder
  GIMaP was started from (for the portable version: the program folder): `request.json`,
  `batch_summary.json`, `top20_candidates.json`, `fitting_run_summary.json` and, for every curve
  and half, `solutions.json` (parameters, units, the stage that ran and why), `display.npz` and
  `fitting_curves.csv` (*General V5* also writes `candidates.npz`). The folders are not removed
  automatically.
- **Model files**: the model and its trained weights (`*.h5`, `*.npz`) are in
  `modules/Fitting_1D_Model/Workflow_v5/`, both in the Git repository and in the portable Windows
  version. Each network checks its files against their checksums before it is used. With a
  changed or missing file *General V5* does not start, and the message names the file; *Single
  RC specialist* goes on with its numerical fit and gives *Fast model unavailable* as the reason.
  In a source checkout, see *Updating a source checkout* above when this happens. The numerical
  methods of Fitting (*Refine*, *Search*, *Find the particle shape*) use no network: the check
  does not concern them.

## 2D Prediction Workflow (Labs)

1. Open **2D Prediction** in the sidebar (Labs).
2. **1. Import data**: **Single file** (**Choose CBF frame…**, and **Stack** for how many
   consecutive files make one input) or **Folder batch** (a folder of numbered frames, **Range**
   and **Files per prediction**). Prediction reads CBF frames only.
3. **2. Import model**: choose the **Prediction setup** (a module under `modules/`) and **Import
   model...**. **Advanced model configuration** holds the framework check, **Reload Config** and
   **Edit Config**.
4. **3. Predict**: the button is enabled when *Input*, *Model* and *Framework* are ready; **Stop**
   ends a run.
5. The canvas on the right shows **Input preview** and **Prediction result**; **Export input...**
   and **Export result...** save what is shown with a JSON record. A folder batch lists its
   results under **Batch results**.

## Model Import

Prediction modules are configured through `module.yaml` files under `modules/`. Existing module configurations include fields such as:

- `id`
- `name`
- `framework`
- `version`
- `model.model_path`
- `preprocess.entry`
- `preprocess.steps`
- `preprocess.params`
- `io.input_type`
- `io.input_shape`
- `outputs`

**For normal users, only the model path should be changed when importing or replacing a trained model. Different models may require different preprocessing settings, so please make sure the selected model and preprocessing workflow match.**

1D Predict does not use these modules: it always runs the bundled Workflow V5 model.

## Command Line and MCP

`tools/gimap_agent.py` runs GIMaP's analysis without the window, for scripts and for agents
(Codex, Claude Code):

```bash
python tools/gimap_agent.py auto <frame> [<frame> ...] --notes "<your notes>" --out <folder>
```

- `auto` detects GIWAXS or GISAXS for each frame once its geometry is applied
  (`--technique giwaxs|gisaxs` forces one) and writes, for every frame, a `report.md` with each
  decision and its reason, the curves as CSV and the q map, plus a `summary.md` for the batch.
- Nothing is written next to the data: results go to `--out` (without it, to
  `assistant_runs/cli/` in the user data folder).
- A series is analysed in its final state: the last fifth of its frames summed, at most 10 (below
  ten frames the last frame alone); `--frame` and `--sum` choose other frames. The module files of
  one detector are analysed once, however many of them are given; `--series` takes files that hold
  one frame each as one series.
- Exit codes: `0` everything analysed and nothing open; `2` at least one frame has a question
  under *Needs attention* (a missing value, a judgement such as the GISAXS model, or no geometry);
  `1` at least one frame failed (`1` wins over `2`).
- `status`, `find-calibration`, `tools` and `call` give finer control; `mcp` serves the same tools
  as an MCP server on stdio.

Read [docs/agents/giwaxs-playbook.md](docs/agents/giwaxs-playbook.md) or
[docs/agents/gisaxs-playbook.md](docs/agents/gisaxs-playbook.md) before relying on the numbers:
they say which default decisions may not fit your question. `python tools/gimap_agent.py auto --help`
lists every option.

## Documentation

- [docs/User_Manual.md](docs/User_Manual.md) — the user manual; [docs/User_Manual_zh.md](docs/User_Manual_zh.md) — the same in Chinese (**Help ▸ User Manual**, `F1`, opens the edition of the interface language in your browser).
- [CHANGELOG.md](CHANGELOG.md) — what changed in each version, and the known limitations.
- [docs/agents/](docs/agents/) — the playbooks for the command line and for agents.
- [docs/development.md](docs/development.md), [docs/architecture/](docs/architecture/) and
  [docs/ui/workspaces/](docs/ui/workspaces/) — for developers.

## System Requirements

- Windows is the primary development and usage target.
- Python 3.10 or 3.11 is recommended for source execution.
- A 1280 × 720 screen or larger.
- CPU execution is supported. GPU TensorFlow setups are not configured by this repository and must be installed separately if required.

## Development

Install the development dependencies and run the unified verification command:

```bash
python -m pip install -r requirements-dev.txt
python tools/check.py
```

This runs the test suite (about 14 minutes), an offscreen start-and-close check of the real main
window with its seven pages, and the deliberately narrow Ruff baseline, without formatting unrelated
code. Configuration is in `pyproject.toml`; details are in
[docs/development.md](docs/development.md).

## Dependencies

Dependencies are listed in `requirements.txt`:

- numpy>=1.24,<2.0
- scipy>=1.10
- threadpoolctl>=3.1 (limits BLAS threads in fit1d workers)
- matplotlib>=3.7
- pyqtgraph>=0.13.7,<0.15
- BornAgain==24.1 on Windows and Linux; installed separately on macOS
- PyQt5>=5.15
- opencv-python>=4.8
- h5py>=3.9
- hdf5plugin>=4 (reads NeXus/HDF5 frames compressed with bitshuffle or LZ4, as Eiger detectors write them)
- fabio>=2023.4.0
- tensorflow-intel>=2.15,<=2.16; platform_system=="Windows"
- tensorflow>=2.15,<=2.16; platform_system!="Windows"
- PyYAML>=6.0
- anthropic>=1.8 and openai>=1.40 (optional: Process with AI; GIMaP runs without them)
- psutil>=5.9 (optional: Batch Export reads the physical cores and the free memory with it to
  choose how many frames to reduce at once; without it GIMaP asks the operating system)


## Feedback and Contact

For feedback, bug reports, or collaboration questions, contact:

[yufeng.zhai@desy.de](mailto:yufeng.zhai@desy.de)


## License

This project is released under the MIT License. See [LICENSE](LICENSE) for the full license text.
