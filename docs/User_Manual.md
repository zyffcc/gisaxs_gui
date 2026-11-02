# GIMaP User Manual

GIMaP is a desktop program for grazing-incidence X-ray scattering: GISAXS and GIWAXS. It turns
detector frames into calibrated curves (I(q), I(qy), I(χ)), finds peaks, orientation and sizes,
fits particle models to 1D curves, follows in-situ series and compares runs. Every curve keeps a
record of where it came from and with which settings.

This manual describes version 0.2.0, a pre-release: treat it as beta scientific software and check
results that matter. The workspaces (Analyze, Fitting, Compare, Sputter) are meant for daily work;
Sputter's model covers Ag and Au sputtered onto polymers (chapter 8). The Labs pages and 1D Predict are
experimental. **Help ▸ About GIMaP** shows the version you are running.
What changed since 0.0.2, and the known limitations of this version, are in `CHANGELOG.md` in the
program folder (chapter 13 collects the limitations).

This manual says what to do and what you get. How GIMaP computes what it shows — the formulas,
thresholds, error models and their limits — is in a separate reference, Physics and Methods
([`docs/Physics_Manual.md`](Physics_Manual.md), **Help ▸ Physics and Methods**); where a method matters for
what you decide, this manual names its section there.

New to GIMaP? Start with chapter 1: two walk-throughs of about ten minutes each, on public data.

## Contents

1. Quick start
2. Installation and updating
3. The window
4. Geometry and calibration
5. Analyze
6. Fitting
7. Compare and stages
8. Sputter
9. Tool windows
10. AI assistant
11. Labs
12. Output files
13. Known limitations
14. Troubleshooting, FAQ, version and contact

## 1. Quick start

Two walk-throughs, each about ten minutes: one GIWAXS frame and one GISAXS frame, both public data.
Each goes the same way: open the frame, give it a geometry, run the automatic analysis, read the
results, send the curve to Fitting. No AI is used.

### 1.1 Before you start

The two data sets come with GIMaP, in the portable version as in a source checkout: they are in the
folder `tests/data/external/` of the program folder:

| Folder | Files | What it is |
|---|---|---|
| `giwaxs_p08_mapi/` | `S121_MAI_A2_00841.tif`, `LaB6_2021_12_DESY_P08.poni` | GIWAXS of a MAPbI₃ perovskite film, DESY P08, PerkinElmer 2048 × 2048 detector, αi = 0.075° |
| `gisaxs_galaxi/` | `galaxi_data.tif`, `galaxi_bornagain.poni` | GISAXS of Ag nanoparticles, GALAXI (Jülich), Pilatus 1M, λ = 1.34 Å, αi = 0.463° |

`tests/data/external/README.md` gives their sources, licences and documented geometry.

1. Start GIMaP (chapter 2). The **Start** page opens.
2. **Help ▸ Example Data ▸ GIWAXS — MAPbI₃ Film (DESY P08)…** copies the GIWAXS example (the frame
   and its `.poni`) to a folder of your choice, by default `Documents\GIMaP examples\giwaxs_p08_mapi`,
   and opens the frame in Analyze: step 1 of section 1.2 is then done. GIMaP writes its results next
   to the data, in a folder `gimap_analysis/`; working on a copy keeps the program folder clean.
   (You can also copy the folder yourself and open the frame from the Start page, as section 1.2 says.)
3. For section 1.3, take **Help ▸ Example Data ▸ GISAXS — Ag Nanoparticles (GALAXI)…** the same way.
   Keep each example in a folder of its own, as the default folders do. **Find Calibration
   Automatically** also looks in the folders beside and above the frame, but it takes a calibration
   file in the frame's own folder first (section 4.2): each frame gets the `.poni` of its own example,
   not the one of the example next to it.

**Help ▸ Example Data** also has a silver behenate calibrant (pyFAI) and a GIWAXS frame taken with a
tilted detector (Xeuss). The tooltip of each entry names its source and licence, and the copy comes
with `SOURCE_AND_LICENCE.md`.

![The Start page](manual_images/en/01_start.png)
*The Start page: drop data or open it, choose what you want to know (the series card has its own
Open Folder… and Open Files… links), or ask the AI. Files opened before are listed under Recent.*

### 1.2 GIWAXS: a perovskite film

**1. Open the frame.** On the Start page press **Open Files…** and choose `S121_MAI_A2_00841.tif`
(or drag the file onto the window). **Analyze** opens. Check:

- the command bar names the file, and the step **Data** its size, `2048×2048 px`;
- the step **Data** is ✓ and **Mask & corrections** says how many pixels are left out
  (`0.0058 % of the pixels left out`: isolated hot and dead pixels);
- **Geometry** shows **!** *No geometry yet*;
- a yellow banner says *No instrument profile for this detector (2048×2048)* and offers
  **Find Calibration Automatically**, **Enter Geometry…** and **Calibrate…**.

The detector image is shown in pixels. Without a geometry there are no q curves yet.

**2. Give it a geometry.** Press **Find Calibration Automatically** in the banner. GIMaP searches
the folders around the frame and finds `LaB6_2021_12_DESY_P08.poni`. Check:

- **Geometry** is ✓ with `811.8 mm · λ 0.6888 Å · αi 0°`;
- **Cuts** is ✓ (*Ring and bands*), and the curves appear;
- the **Results** step says where the geometry came from: *a saved calibration file (not re-checked
  against a standard image); found by the search: LaB6_2021_12_DESY_P08.poni*, and *1 question(s)*.

The geometry is now saved as the instrument profile *Detector 2048×2048*. Analyze uses it for every
frame of this size from now on (section 4.1). With your own data, make sure the file found is the
calibration of your measurement.

**3. Run the automatic analysis.** Press **Run Automatic Analysis** in the command bar. It takes a
few seconds; the **Results** tab shows each step as it is done. When it ends, the Results step
asks one question: the incidence angle. Type `0.075` in **Incidence angle αi (°)** and press
**Run Again with These Answers**. **Geometry** now shows `αi 0.075°`, and the αi field in the command
bar shows `αi 0.075 °` in orange: a value you set, used for every frame you show, also the next one you
open, until you go back to the profile's αi (section 4.1).

**4. Read the results.** The Results tab begins with a line such as
*GIWAXS — 12 peaks (12 reliable), 3 rings analysed*. Below it:

- **Peaks**: q and FWHM in Å⁻¹, d in Å, L (Scherrer size, a lower bound) in nm, and whether the peak
  is out-of-plane (OOP) or in-plane (IP). Here you find peaks near q = 0.90, 1.00, 1.41, 1.99 and
  2.22 Å⁻¹. The data's README names 0.899 Å⁻¹ as PbI₂ (001) and 0.999 Å⁻¹ as MAPbI₃ (110).
  GIMaP does not assign phases: compare q with the lines of your material.
- The three rings with the largest peak area (and no warning), with their orientation (Hermans f)
  and where their maximum lies.
- **Checks**: shadowed and unmeasured χ ranges (the missing wedge next to qz) that bias f.
- **Geometry**: the profile, distance, beam centre, wavelength and αi used.

Select a peak and open **Fit details** to see the points it was fitted to, the Gaussian and its
background, and the errors. **Save Report…** keeps everything as a web page or Markdown. Switch the
image to **q map** or **Cake** to see the rings in q; **Sources** shows which pixels each curve uses.

![GIWAXS results](manual_images/en/07_giwaxs_results.png)
*The Results tab of a GIWAXS frame: the summary line, the peak table, Fit details and the ring
orientation; the checks follow further down.*

**5. Send to Fitting.** Press **Send to Fitting**. GIMaP writes
`gimap_analysis/S121_MAI_A2_00841_fit_input.dat` next to the frame (columns q, I, σ, pixels; q in
Å⁻¹; the `#` header lines say where the points come from) and opens **Fitting ▸ Single analysis**
with it. The step **Curve** says `2050 points fitted` (the file's name is in the command bar).

For GIWAXS, I(q) is what is sent. Fitting's models are particle models (chapter 6), and a note at the
top of the Fitting page says so: they are particle form factors, not peaks, so for crystalline peaks the
peak table above is usually the result you want (peak fits of many frames: **Batch Export ▸ Peaks of
the ring regions**, section 5.7). The `_fit_input.dat` file is also the curve with its record, ready for
other programs.

### 1.3 GISAXS: silver nanoparticles

The source of these data describes the sample as Ag spheres with a radial paracrystal of peak
distance 53.6 nm (BornAgain's example *experiment_at_galaxi*).

**1. Open the frame.** First clear what section 1.2 left in Analyze: press **Clear** in the step Data
(this also forgets the answers given there), then right-click the αi field in the command bar ▸
**Back to Profile αi**. Otherwise the αi of the perovskite film, 0.075°, would be used for this frame too, and the run
would not ask for αi. Then take the GISAXS example (**Help ▸ Example Data ▸ GISAXS — Ag Nanoparticles
(GALAXI)…**, section 1.1), or **File ▸ Open Data…** (`Ctrl+O`) and choose `galaxi_data.tif`. Analyze
shows `1043×981 px` (rows × columns). **Mask & corrections** leaves out `1.4 %` of the pixels: the gaps
between the Pilatus modules. The banner asks for a geometry, as before.

**2. Give it a geometry.** Press **Find Calibration Automatically**. The search finds
`galaxi_bornagain.poni` in the frame's folder and takes it before the GIWAXS calibration of section 1.2
in the folder beside it: a calibration file in the frame's own folder comes first (section 4.2). Check:

- **Geometry** is ✓ with `1730.0 mm · λ 1.34 Å · αi 0°`;
- **Cuts** shows *Yoneda cut*. The mode stays **Auto**: frames whose largest scattering angle is at
  most 20° are treated as GISAXS;
- the Results step says *a saved calibration file (not re-checked against a standard image); found
  by the search: galaxi_bornagain.poni*, and *1 question(s)*.

You can also name the calibration before you press the button: type `galaxi_bornagain.poni` in
**Beamtime notes** of the step **Results** (the file lies in the frame's folder, so its name is
enough). A calibration named there is tried before the instrument profile and the search, in every
run.

**3. Run the automatic analysis.** Press **Run Automatic Analysis**. This takes about half a minute
(the model fits are the slow part). The Results step then asks for αi. Type `0.463` in
**Incidence angle αi (°)** and press **Run Again with These Answers**.

The answer matters: with αi = 0° the Yoneda band was quoted at αf = 0.639°, with the right αi at
αf = 0.176°. **Geometry** now shows `1730.0 mm · λ 1.34 Å · αi 0.463°`, and the αi field shows
`αi 0.463 °` in orange, as in section 1.2: the run kept the instrument profile saved in step 2 and
used your αi with it. Had you named the calibration in the notes, the run would have applied that file
again and saved your αi with it in the instrument profile of this detector, and the field would not be
orange (section 4.1).

**4. Read the results.**

| Part | What you see here | What to check |
|---|---|---|
| **Horizontal cut** | At the Yoneda band, rows 605–609 | The orange band on the image lies on the Yoneda band |
| Beam centre | x moved to the symmetry axis, x = 597.26 px | The documented direct beam is x = 597.1 px |
| Halves | *Halves: Mean of both halves*, the reason in its tooltip | Both halves agree; otherwise the usable one is taken |
| **Length scale 2π/q\*** | *2π/q\* = 55.6 nm at a shoulder of the cut (\|qy\| = 0.0113 Å⁻¹, no resolved peak): not necessarily the particle spacing.* | It is not the particle spacing: near the documented 53.6 nm by coincidence (a BornAgain simulation of the same particles without interference shows a shoulder too) |
| **Fit of I(qy)** | A table of solutions: model, R, h, D (nm), χ²/N; a value at a search bound is marked ⚠ (*500 ⚠*) and named under the table in a warning (*⚠ Not determined by the curve (at a search bound) — 1: D, σh/h; …*; which bound: its tooltip) | Several models may fit about equally well |

Here the Results tab also shows *1 point(s) to check*: **Model choice**. The first-ranked solution, a
random cylinder (χ²/N 1.26), has D at its upper bound (500 nm), and the second also has a value at a
bound. The run reports rank 3, a sphere (R = 9.44 nm, D = 48.3 nm, χ²/N 1.36): the best-ranked
solution within 10 % of the first's χ²/N with no non-linear value at a search bound. The table selects it, and the
line under it says so. Several families fit within those 10 %, so the curve alone does not decide the
model: GIMaP names the solutions whose D is within 15 % of 2π/q\* (a coincidence of two lengths, not
evidence) and leaves the choice to you. For this sample you know the particles are spheres. The quick
fit is a screening step: confirm the model in Fitting (step 5).

![GISAXS results](manual_images/en/08_gisaxs_results.png)
*The Results tab of a GISAXS frame: the cut, the length scale 2π/q\*, the fitted curve and the table of solutions.*

**5. Send to Fitting.** Press **Send to Fitting**. GIMaP writes
`gimap_analysis/galaxi_data_fit_input.dat` (the halves as chosen in the Cuts step, here the mean;
q in Å⁻¹) and opens it in **Fitting ▸ Single analysis**: **Curve** ✓ `599 points fitted`, **Model**
`Sphere · 9 free`, **Fit** *Choose a method, then Fit*. Fitting shows q in nm⁻¹.

The Fit step has chosen **Find the particle shape (no AI)** for this curve, and the line under
**Method** says why: *Recommended for this curve: a GISAXS cut not fitted yet (Predict parameters needs
the optional fit1d add-on, which is not installed).* With the fit1d kit installed (section 2.3) it
chooses **Predict parameters (fit1d)** instead (*Recommended for this curve: a GISAXS cut not fitted
yet.*); to follow this walkthrough, choose **Find the particle shape (no AI)** in **Method** then. Press
**Find the particle shape**. After about 15 s the status line says *5 solutions; the best, Sphere (χ²ᵣ
1.39), is in Model*. Results lists the other
solutions under **Other solutions**; press **Refine These Values** to refine the sphere and get its
errors. The same curve opens with **Refine in Fitting** in Analyze's Results tab.

Sizes depend on the model and on its size distribution. BornAgain's model uses a log-normal
distribution of radii, GIMaP's models another one: compare radii with care.

![Fitting with the curve sent from Analyze](manual_images/en/11_fitting_single.png)
*Fitting ▸ Single analysis after a refining fit: the curve, the model over it, the fitting range
(orange) and the residuals.*

### 1.4 Where to go next

| You want to … | See |
|---|---|
| Set up the geometry of your own detector | Chapter 4 |
| Draw masks, choose cut regions, follow a series, export many frames | Chapter 5 |
| Fit a model properly, or a whole in-situ series | Chapter 6 |
| Compare runs or samples | Chapter 7 |
| Follow the islands of an in-situ sputter series | Chapter 8 |
| Let an AI do the analysis while you watch | Chapter 10 |
| Know what each exported file contains | Chapter 12 |

## 2. Installation and updating

### 2.1 Windows: the portable version

No Python is needed; the package brings its own.

1. Open the [GitHub Releases](https://github.com/zyffcc/gisaxs_gui/releases) page and choose the
   latest release. It offers the package in one of two forms: one file,
   `GIMaP-<version>-windows-x64-portable.zip`, or, when that file would be larger than GitHub allows
   (1900 MiB), its parts `GIMaP-<version>-windows-x64-portable.zip.part01`, `.part02` … with the
   installer `00_Download_and_Install_GIMaP-<version>.bat`.
2. Choose a folder with a short path that uses English letters only and that you may write to (for
   example `C:\GIMaP`).
   - **One ZIP:** download it, extract the whole ZIP into that folder, and double-click
     `Start_GIMaP.bat` in the folder `GIMaP-<version>-windows-x64-portable`.
   - **Parts:** download only `00_Download_and_Install_GIMaP-<version>.bat` into that folder (an empty
     one) and double-click it. It downloads the parts, verifies each one (SHA-256), joins and extracts
     them into the folder `GIMaP-<version>-windows-x64-portable` beside the installer, and starts
     GIMaP once. (`Install_GIMaP_parts.bat` joins and extracts parts you downloaded yourself, without
     checking them.)
3. To check a download yourself, compare its SHA-256 with the line for it in `SHA256SUMS.txt` on the
   release page. PowerShell prints it with `Get-FileHash -Algorithm SHA256` followed by the file name.

**What is in the folder.** `runtime\` (the bundled Python with all the packages GIMaP needs), the
program, `Start_GIMaP.bat`, the example data of section 1.1 (`tests\data\external\`), and `licenses\`:
the licence texts of everything in `runtime\`, with GIMaP's own `LICENSE` and `THIRD_PARTY_NOTICES.md`
(also **Help ▸ Third-party Notices**). PyTorch is not in the package: the Sputter workspace installs it
at its first **Predict** (section 2.3). `Update_GIMaP.bat` is there too, but it updates only git
clones (section 2.4).

**Starting GIMaP later.** Double-click `Start_GIMaP.bat` in the folder
`GIMaP-<version>-windows-x64-portable`; a desktop shortcut to that file is convenient. The first
start prepares the bundled Python environment for the folder, which takes a few minutes.

**Path and memory.** The folder path must use English letters only (for example `D:\GIMaP`, not a
folder with a Chinese name such as a localised "Downloads"): Qt cannot load its plugins from such a
path, and `Start_GIMaP.bat` says so and stops. Move the whole folder to such a path and start it again.
The preparation at the first start (`tools\portable_unpack.py`) needs little memory; when it does not
succeed, GIMaP is not started and the window gives the reason. If it mentions memory ("Not enough
memory resources are available to process this command"): restart the computer, close other programs,
check that virtual memory is on (System ▸ About ▸ Advanced system settings ▸ Performance Settings ▸
Advanced ▸ Virtual memory: automatic), keep a few GB free on that drive, and double-click
`Start_GIMaP.bat` again. If it still fails, extract the release ZIP again into an empty folder.

The installer and the launcher are batch files and are not signed. Windows may ask for confirmation
before it runs one, and an institution's antivirus may block it. Only run files from the GitHub
Releases page of GIMaP; otherwise run GIMaP from source (section 2.2).

### 2.2 Running from source

**The simple way on Windows (no Python needed).** Double-click `Start_GIMaP.bat` in the folder that
holds `main.py`: a clone of the repository, or a ZIP of it (**Code ▸ Download ZIP** on GitHub; Windows
may extract it into a folder `gisaxs_gui-main` that holds another `gisaxs_gui-main`, the one with
`main.py`). As for the portable version, the folder path must use English letters only, and it
should be short and writable (section 2.1). The first start sets up a private environment in the
folder `.gimap`: it downloads uv (a small Python installer, pinned version, checked against its
SHA-256), Python 3.10, the packages of `requirements.txt` (about 2 GB) and PyTorch for the Sputter
workspace (`requirements-sputter.txt`, about 1.2 GB more), which takes 3-20 minutes depending on the
connection, then opens GIMaP. Later starts are immediate. When installing the packages fails (for
example for lack of free memory), it tries again one package at a time, which needs much less memory
and takes longer. When an update changes `requirements.txt` or `requirements-sputter.txt`, the next
start adds the missing packages first. Nothing outside the folder is changed; deleting `.gimap`
removes the environment.

**Moving the folder.** The environment in `.gimap` belongs to the place where it was set up. After
you move or rename the folder, delete its `.gimap` folder and double-click `Start_GIMaP.bat`: the
environment is set up again for the new place. To build it again in place (for example when it was
damaged), run `powershell -NoProfile -ExecutionPolicy Bypass -File tools\setup_source_env.ps1 -Reinstall`.

**By hand.** Install Python 3.10 or 3.11, then in the folder of the checkout:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python main.py
```

If PowerShell refuses to activate the environment, run
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first. With Conda:

```powershell
conda create -n GUI python=3.11 -y
conda activate GUI
python -m pip install -r requirements.txt
python main.py
```

Start GIMaP from such an environment with `python main.py`. `Start_GIMaP.bat` never uses it: it always
uses its own `.gimap` environment and would set one up beside yours.

The Sputter workspace's model is in the repository (`utils/sputter_toolkit`) and runs on PyTorch (CPU): its
first **Predict** installs PyTorch into the Python GIMaP runs in. To install it beforehand, run
`python -m pip install -r requirements-sputter.txt` in the same environment.

The trained weights of 1D Predict are in the Git repository (about 56 MB), so a checkout runs
everything, including 1D Predict (chapter 6). On macOS, BornAgain is installed separately: `README.md`
says how.

### 2.3 Optional parts

GIMaP runs without these. Each is needed only for the function named.

| Part | Needed for | How |
|---|---|---|
| An AI account or a local model | **Ask AI…**, **Tools ▸ Process with AI…** | The Python packages come with `requirements.txt`; set up the AI in **Tools ▸ Settings… ▸ Assistant** (chapter 10) |
| The fit1d kit (`gisaxs1d_gui_kit`) | **Predict parameters (fit1d)** in Fitting (also for every frame of an In-situ series) and the method *fit1d* in 1D Predict | Put it in `utils/gisaxs1d_gui_kit`, or name its folder in the environment variable `GISAXS1D_KIT_HOME` (chapter 6) |
| StarDist | Separating touching particles in **Tools ▸ SEM Particle Analysis…** | A separate Python environment with `requirements-sem-stardist.txt`, and a local model folder (chapter 9) |
| PyTorch (CPU) | **Predict** in the Sputter workspace | Not in the portable package: installed into GIMaP's Python on the first Predict (about 1.2 GB, needs the internet). In a source folder, the first start of `Start_GIMaP.bat` installs it (section 2.2). By hand: `python -m pip install -r requirements-sputter.txt` (section 8.1) |

TensorFlow runs on the CPU; a GPU set-up is not configured.

### 2.4 Updating

**Portable version.** Get the new version as in section 2.1, the installer or the ZIP. It goes into a
folder of its own; start the new `Start_GIMaP.bat` and delete the old folder when you no longer need
it. `Update_GIMaP.bat` cannot update a portable folder: it has no `.git`, and the script says so and
changes nothing.

**Source checkout (a git clone).** Double-click `Update_GIMaP.bat` in the folder (it needs Git 2.16 or
newer), or run:

```powershell
powershell -ExecutionPolicy Bypass -File tools\update_source.ps1
```

The script updates the branch from GitHub and says what it found and what it did. It puts your
uncommitted edits aside and back on top of the new version, leaves your data and results alone, and
stops before it would overwrite a file of yours. An edit that does not fit the new version stays put
aside, and the script says so. Commits of your own that are not on GitHub are kept on a backup branch
when the branch has to be moved to a rewritten history; `README.md` explains how to get them back. Add `-DryRun` to see what it would do without changing
anything. When it says a list of Python packages changed (`requirements.txt`, or
`requirements-sputter.txt`: PyTorch, for Sputter), there is nothing to do if you start GIMaP with
`Start_GIMaP.bat`: its next start adds the new packages itself. In an environment you set up
yourself (conda, venv), run the command the script prints for each changed list, such as
`python -m pip install -r requirements.txt`.

A folder from **Code ▸ Download ZIP** has no `.git` and is not a checkout: the script says so and
changes nothing. Download the ZIP again, or clone the repository with Git
(`git clone https://github.com/zyffcc/gisaxs_gui.git`), which the script updates from then on.
Extract the new ZIP into a new folder (moving `.gimap` over does not work, section 2.2): its first
`Start_GIMaP.bat` sets up the environment again. Your settings stay in the user data folder.

A checkout last updated before 2026-10-13 does not have the script yet. The history of `main` was
rewritten on that day: do not run `git pull`, merge or push from such a checkout. Move it first with
the steps in `README.md`, section *Updating a source checkout*; from then on use the script.

**What an update keeps.** Settings, instrument profiles, recent files and the fitting model are kept
in the user data folder, outside the program (section 3.11), so an update keeps them. Version 0.0.2
kept its settings inside its own program folder. A new portable folder does not carry them over and
starts with the defaults. A source checkout updated in place imports them once, at the first start,
while the user data folder holds no settings yet.

### 2.5 System and screen

Windows is the platform GIMaP is developed and tested on. A 1280 × 720 screen is enough (the
smallest window is 1024 × 640); a 1080p or larger screen is more comfortable for the detector image
beside the curves. GIMaP follows the display scaling of the operating system (for example 125 % or
150 %).

## 3. The window

### 3.1 The parts of the window

![The Analyze workspace](manual_images/en/02_analyze_overview.png)
*Analyze: the sidebar, the command bar, the steps on the left with the frame list of the Data step (Choose
Frames… below it), the image in the middle, the curves on the right.*

- **The sidebar** (left) holds the pages. *Workspaces*: **Start**, **Analyze**, **Fitting**,
  **Compare**, **Sputter**. *Labs*: **2D Prediction**, **Trainset Build**, the experimental machine-learning
  tools. **« Collapse** at its foot (or `Ctrl+B`) reduces it to icons. `Ctrl+1` … `Ctrl+7` switch pages
  in this order: Start, Analyze, Fitting, Compare, Sputter, 2D Prediction, Trainset Build (so Sputter
  is `Ctrl+5`).
- **The menu bar**: **File, Edit, View, Analyze, Fitting, Tools, Help** (section 3.6). The tool windows
  open from **Tools**; **Analyze** and **Fitting** hold the commands of those two pages.
- **The page.** Analyze and Fitting share one layout: a command bar at the top, the steps of the work
  on the left, the plots on the right.
- **Messages** appear as short notes at the bottom right. A note about a written file has an
  **Open Folder** button. Each page also has a status line at its foot.

The window title names the frame shown, and the project when one is open.

**Dropping files.** Detector frames, folders and `.gimap` projects can be dropped anywhere on the
window. A curve file (`.dat`, `.txt`) dropped while Fitting is shown opens there.

**Closing.** Closing GIMaP while something still runs — Batch Export, a Series map, the automatic
analysis, an AI analysis, a fit, an In-situ series, a 2D prediction, an SEM Particle Analysis, a
Sputter prediction — asks first and lists what runs. **Yes** stops them and quits; **Cancel** keeps working. An SEM analysis
that is still finishing may ask you to close the window again once it has stopped.

### 3.2 The Start page

- **Drop box**: drag detector images or a folder onto it, or press **Open Files…**, **Open Folder…**,
  **Batch Export…** (a folder of raw frames straight to curves, with the settings of last time) or
  **Open Project…**. Analyze opens at once and shows the file, its detector and frames while the
  image loads.
- **Recent**: the five files, folders and projects opened last, with their folder (the full list is
  in **File ▸ Open Recent**).
- **What do you want to know?**: *Crystals and orientation* (GIWAXS), *Nanostructure* (GISAXS),
  *In-situ or batch series*, *Calibrate the detector*. Each opens the right place, with the mode
  chosen. The series card has two links of its own: **Open Folder…** (a folder of frames: one file per
  frame, or the module files of one scan) and **Open Files…** (chosen frames, or one NeXus/HDF5 file that
  holds the whole series); both end on Analyze's Series tab.
- **Or ask the AI**: describe your data and question, then **Ask the AI…** (chapter 10).

### 3.3 Steps and command bars

In Analyze and Fitting each step on the left says where it stands, with one line of what it found:

| Mark | Meaning |
|---|---|
| ✓ | Done, nothing to check |
| ! | Done, but look at it (a question, a warning) |
| × | Failed: the line says why |
| … | Running |
| a number | Not done yet |

A line longer than the step list is wide is cut with “…”; rest the pointer on the step for the whole line.

Click a step to see its controls. The controls used most are shown; rare options are folded under a
heading you can open.

The screen keeps what you need each time you look: states, results, failures, missing or guessed values and
where a value comes from. How to use a control and what a section is for are in tooltips: rest the pointer on
the control, or on the title of the section (the pointer shows a question mark where a title has one).

When Analyze's command bar is narrow, the long button texts are shortened (*Run Analysis*, *Batch…*,
*Fitting*); the full text stays in the tooltip.

### 3.4 Plots and images

**Curve plots.**

- The right mouse button opens a menu: **Reset View**, the log axes, **Copy Image**, **Copy Data**,
  and on plots with a **Save** button also **Plot as Figure…** and **Curves as Data…**.
- Dragging with the right button zooms; the mouse wheel zooms too.
- **Zoom** (the magnifier): drag a rectangle. Shift + drag works on every plot and image without the
  button. **Reset** shows everything again.
- **Log I** and **Log q** switch the axes. The eye button shows or hides each kind of mark.
- When a plot is narrow, its title wraps above it and its options move into a **⋯** menu.

**Detector images** (also the q map, the cake and the Series map) have **Zoom**, **Fit view**,
**Marks**, a colour map, **Log** and **Levels**. Next to each image is a histogram with two handles
for the colour limits. Section 5.4 describes the colour limits and marks in detail.

Saved figures always use the light colours, also when the screen shows the dark theme.

### 3.5 Tables

The result tables — the peaks and solutions of the automatic analysis, the tables of Fitting and of
Compare — can be copied. Select rows and press `Ctrl+C`, or use **Copy Rows** / **Copy Table** on
the right mouse button. The rows are copied with their column names, ready to paste into a
spreadsheet.

### 3.6 Menus

| Menu | Entries |
|---|---|
| File | Open Data… (`Ctrl+O`), Open Folder… (`Ctrl+Shift+O`), Open Recent, Open Project… (`Ctrl+Shift+P`), Save Project (`Ctrl+S`), Save Project As… (`Ctrl+Shift+S`), Labs Parameters ▸ Load… / Save As…, Exit |
| Edit | Undo (`Ctrl+Z`), Redo (`Ctrl+Y`; `Ctrl+Shift+Z` also works): the set-up in Analyze, the model in Fitting's Single analysis |
| View | *Workspaces* and *Labs*: the pages (`Ctrl+1`…`Ctrl+7`), Collapse Sidebar (`Ctrl+B`), Full Screen (`F11`), Theme ▸ Light / Dark, Font Size ▸ Larger (`Ctrl++`) / Smaller (`Ctrl+-`) / Reset (`Ctrl+0`), Language ▸ English / 中文 |
| Analyze | Run Automatic Analysis, Stop Automatic Analysis, Repeat Analysis (`F5`), Export Curves… (`Ctrl+E`), Batch Export… (`Ctrl+Shift+E`), Batch Export a Folder…, Send to Fitting, Send Series to Fitting…, Send Series to Compare, Set-up Files ▸ Save Settings… / Load Settings… / Save Cuts… / Load Cuts… / Save Masks… / Load Masks…, Figures ▸ Save Image… / Save q-Map Data… / Save View Data…, Clear Frame List |
| Fitting | Open Curve…, Fit a Folder of Curves…, Fit (`Ctrl+Return`), Stop Fit, Load Model…, Save Model…, Save Data and Fit…, Save Plot…, In-situ Series ▸ Start / Stop, 1D Predict — Fit Many Curves… |
| Tools | Geometry Calibration… (`Ctrl+Shift+G`), Format Converter… (`Ctrl+Shift+C`), Convert Current File…, XRR Series Extractor… (`Ctrl+Shift+R`), SEM Particle Analysis…, 1D Predict — Fit Many Curves…, Process with AI… (`Ctrl+Shift+L`), Settings… (`Ctrl+,`) |
| Help | User Manual (`F1`), Physics and Methods, Keyboard Shortcuts…, Example Data ▸ GISAXS — Ag Nanoparticles (GALAXI)… / SAXS — AgBh Calibrant (pyFAI)… / GIWAXS — MAPbI₃ Film (DESY P08)… / GIWAXS — PbTe, Tilted Detector (Xeuss)…, Report a Problem…, Open Error Log Folder, Open User Data Folder, GitHub Repository, Check for Updates…, How to Cite GIMaP…, Third-party Notices, About GIMaP |

![The Analyze menu open over the Analyze workspace](manual_images/en/23_analyze_menu.png)
*The Analyze menu with one frame open in Analyze. Entries that cannot run now are greyed out; their tooltip says
why.*

![The Help menu](manual_images/en/24_help_menu.png)
*The Help menu: the two manuals, the keyboard shortcuts, the example data (a submenu), where to report a problem,
the folders GIMaP keeps, and the links to GitHub, the citation, the notices and the version.*

- While Fitting's Single analysis is shown, **File ▸ Open Data…** opens a curve and says so
  (*Open Curve…*).
- **Edit ▸ Undo** and **Redo** act on the page shown: Analyze, or Fitting's Single analysis. On
  Analyze the entry names what it would change (for example *Undo: cut regions*). On the other pages
  they are greyed out.
- Every entry of **Analyze** and **Fitting** first shows its page (in Fitting the Single analysis, or
  the In-situ series for Fit a Folder of Curves… and In-situ Series ▸ Start / Stop), then does what
  the page's own button does. An entry that cannot run now is greyed out, and its tooltip says why
  (for example *Open a frame first.*).
- The keys shown next to Undo, Redo, Repeat Analysis, Export Curves…, Batch Export… and Fit belong to
  the page: they act while that page has the keyboard focus (section 3.7). The menu entries work from
  any page.
- *Labs Parameters* are the settings of 2D Prediction and Trainset Build. The file also holds Fitting's
  model (the particles and global parameters of the Single analysis): **Load…** replaces Fitting's model
  with the one in the file and keeps it as the stored model, without asking. Save a project first if you
  need the model you have. Analyze keeps its own set-up in settings files (**Analyze ▸ Set-up Files ▸ Save Settings… / Load Settings…**, also in the
  page's **Export** menu and used by Batch Export). A project keeps everything of one sample (section 3.8).
- **Tools ▸ 1D Predict — Fit Many Curves…** fits a list of curve files with the experimental neural
  model (chapter 6). The other tool windows are described in chapters 4 and 9.
- **Help ▸ User Manual** opens this manual in your web browser; in the Chinese interface it opens the
  Chinese edition. **Physics and Methods** (how GIMaP computes what it shows) and **Third-party
  Notices** open the same way; a link from one of these documents to another opens the other one at
  that section. **Keyboard Shortcuts…** lists every key of the menus and of the pages
  in a table you can copy (`Ctrl+C`).
- **Help ▸ Example Data** copies one of the public examples that come with GIMaP to a folder you
  choose (by default `Documents\GIMaP examples\<example>`) and opens its frame in Analyze (section 1.1).
- **Help ▸ Report a Problem…** copies the versions of GIMaP, the operating system, Python, Qt, PyQt
  and numpy (no paths, no data) to the clipboard and opens a new issue on GitHub in your browser:
  paste them there. **Check for Updates…** opens the releases page on GitHub in your browser (newest
  first, pre-releases included); GIMaP itself makes no network request.

### 3.7 Keyboard shortcuts

The keys of the File, View and Tools menus work on every page. The keys below belong to a page or a
window: they act while it has the keyboard focus. After switching pages with `Ctrl+1` … `Ctrl+7` or the
sidebar, click into the page first. The page keys marked *menu* are also shown in the Edit, Analyze and
Fitting menus (section 3.6); those entries work from any page. **Help ▸ Keyboard Shortcuts…** shows
the keys of the menus and this table in the program.

| Where | Keys | What it does |
|---|---|---|
| Analyze | `Page Up` / `Page Down` | Previous / next imported frame (through the frames of a source, then on to the next source) |
| Analyze | `Ctrl+Z`, `Ctrl+Shift+Z` or `Ctrl+Y` | Undo / redo a change of the set-up (*menu*: Edit ▸ Undo / Redo) |
| Analyze | `Ctrl+E` | Export the curves of this frame (*menu*: Analyze ▸ Export Curves…) |
| Analyze | `Ctrl+Shift+E` | Batch Export… (*menu*: Analyze ▸ Batch Export…) |
| Analyze | `F5` | Analyse the frame again (*menu*: Analyze ▸ Repeat Analysis) |
| Analyze | `Delete` or `Backspace` | Remove the selected source (all its frames), mask or cut region from its list (the files themselves are not touched) |
| Analyze | `Esc` | Stop picking the beam centre on the image; cancel drawing a mask |
| Analyze, drawing a polygon mask | `Return` (or `Enter`) / `Backspace` | Close the polygon (from three points on) / take back the last point |
| Fitting | `Ctrl+Return` | Fit (*menu*: Fitting ▸ Fit) |
| Fitting | `Ctrl+Z`, `Ctrl+Shift+Z` or `Ctrl+Y` | Undo / redo a change of the model (*menu*: Edit ▸ Undo / Redo) |
| Compare | `Delete` | Remove the selected series from the table |
| Tables | `Ctrl+C` | Copy the selected rows with their column names |
| Geometry Calibration | `Esc` | Close the window (it asks first while a calibration runs) |
| XRR Series Extractor | `Esc` | Stop picking the direct-beam centre on the preview; when not picking, close the window |
| Trainset Build (the image) | `Esc` | Leave the drawing mode: picking the direct beam, the ROI or a mask shape |

### 3.8 Projects

A project (`.gimap`, a JSON file) reopens a sample as it was left:

- the frames and the whole set-up of Analyze (mode, profile, masks and drawn regions, corrections,
  cuts), the frames chosen for import (section 5.1) and whether one cut is held for a series;
- the curve, halves, fitting range, left-out points and model of Fitting;
- the folder and choices of the In-situ series, and the series of Compare.

Sputter and the Labs pages are not part of a project. Sputter keeps the values of each scan folder
itself (section 3.9); the settings of the Labs pages go, together with Fitting's model, into their own
files with **File ▸ Labs Parameters ▸ Save As…**.

**File ▸ Save Project** (`Ctrl+S`) writes it; the first time it asks where (next to the first frame
by default). **Open Project…**, the Start page's **Open Project…**, **Open Recent** (projects first)
or dropping a `.gimap` file on the window opens it.

- Data are referenced by path, not copied. Series that Compare took from Analyze are kept in a second
  file, `<name>.compare.npz`, next to the project: move or copy the two files together.
- When a file has moved, the project still opens; a note
  says how many files are missing and names the first ones, and **Details…** lists them all.
- Opening a project replaces what the window holds: the frames of Analyze, the curve and the In-situ
  series of Fitting and the series of Compare of the sample before are closed. Save them first if you need them.
- After **Clear** in Analyze, what you open next is no longer that project: **Save Project** asks
  where to save instead of overwriting it.
- A project cannot be opened while a job runs (an In-situ series, Batch Export …): stop it first.
- A project keeps no fits of the In-situ series. When frames fitted there are not saved yet, opening a
  project asks first: **Save** (the table of every frame), **Discard** or **Cancel** (nothing changes).

### 3.9 Memory between sessions

- **Recent data**: every file or folder opened (dialog, drag and drop, Start page, Batch Export) is
  listed in **File ▸ Open Recent** and on the Start page, newest first. At most 8 are kept; files
  that no longer exist are left out; **Clear the List** empties it. Recent projects are kept apart
  (the last 6), listed first, and stay after **Clear the List**.
- **Last set-up**: when GIMaP closes after a frame was analysed, the Analyze set-up (mode, profile,
  masks, corrections, cut regions …) is kept in `last_analyze_setup.json` in the user data folder.
  With the first frame of the next session a note offers **Use It**. It is never applied without
  asking, since a new beamtime may need other settings.
- **Sputter**: the values of each scan folder's last Predict (geometry, metal, polymer, mask, time window,
  deposition rate, t0 and deposition end), the values used last (another folder does not take over the mask or the sample roll),
  the geometry presets and the result shown last (opened again at the next start); PyTorch is looked for afresh
  at every start.
- The window size and position, the theme, the font size and the language are restored at the next
  start.

### 3.10 When something goes wrong

An unexpected error does not close GIMaP. A window says what happened: **Show Details** shows the
technical report, **Copy Details** copies it for a bug report, and the work goes on. Every error is
written to `logs/errors.log` in the user data folder. **Open Log Folder** in that window, or
**Help ▸ Open Error Log Folder** at any time, opens the folder; when no error has been logged yet,
the menu entry says so. To report a problem, **Help ▸ Report a Problem…** (section 3.6) copies the
versions to paste into the report.

Files that cannot be read say why in words: empty (still being written?), not a detector image, a
damaged or incomplete NeXus file, a NeXus/HDF5 file another program is still writing, no permission.
Chapter 14 has more.

### 3.11 Settings and the user data folder

**Tools ▸ Settings…** (`Ctrl+,`) has four pages. Changes apply immediately; only the keys of the AI
are stored with their own **Save** button.

| Page | What it sets |
|---|---|
| **Appearance** | Light or dark theme, the font size, the interface language (English by default, or 中文). Values, units and file names are never translated. |
| **Analyze** | **Use the beam centre written in the file header** (CBF `Beam_xy`, NeXus `beam_center_x/y`): off by default, because many beamlines never update it, so the calibrated profile centre is kept. **Send horizontal cut to Fitting**: which half of the GISAXS horizontal cut Send to Fitting passes on (by default both halves on \|qy\| in two colours). |
| **Data** | Where the user data folder is (**Open**), and **Reset All Settings…** (instrument profiles are kept; restart GIMaP afterwards) |
| **Assistant** | The AI: which one, its standing instructions, models, keys and permissions (chapter 10) |

![Settings](manual_images/en/19_settings.png)
*Tools ▸ Settings…: the Appearance page.*

All settings, preferences, instrument profiles, the session and the fitting model are kept in one
user data folder outside the program, together with the error log (section 3.10), the result folders
of the Sputter workspace (chapter 8) and the records of the AI assistant's runs (chapter 10):

- Windows: `%APPDATA%\GIMaP`;
- elsewhere: `~/.config/gimap` (`$XDG_CONFIG_HOME/gimap` when that variable is set);
- or the folder named in the environment variable `GIMAP_HOME`.

**Help ▸ Open User Data Folder** opens it. Updating the program never overwrites it, and deleting the
program folder does not remove it.

## 4. Geometry and calibration

Without a geometry GIMaP shows the detector in pixels and nothing in q. This chapter explains where
the geometry comes from, the three ways to get one, the Geometry Calibration window, the pixel
coordinates of the beam centre, and how to adjust the centre in Analyze.

### 4.1 Instrument profiles

An *instrument profile* is a saved geometry for one detector: sample–detector distance, pixel size,
wavelength, beam centre and incidence angle αi. Profiles are kept in the user data folder
(`instrument_profiles.json`, section 3.11).

When a frame is opened, Analyze picks the profile that matches it:

- **Frame size.** A profile for another frame size is never used.
- **Detector name**, when the file states one (CBF and NeXus usually do, TIFF does not). A matching
  name counts more than the size.
- When several profiles match equally, the newest is used.

A different detector name does not rule a profile out: when nothing better matches, a profile of
the same frame size is used. So a profile made for a TIFF without a detector name is used for every
TIFF of that size. When you change set-up (distance, energy) but keep the detector, make or choose
a new profile.

The step **Geometry** shows the geometry in use (`811.8 mm · λ 0.6888 Å · αi 0°`). Its field
**Instrument profile** is set to **Automatic (match detector)**; choose a profile there to use it
instead.

**αi** is part of the profile, but the command bar has its own field. It shows the profile's αi
(*αi from profile* while no frame is shown). A value you type is shown in orange and is used for
every frame; it is also remembered for the next session, and a note at the first frame then says
so. Right-click the field ▸ **Back to Profile αi** to drop it; **Clear** in the step Data does not.

- An αi answered in the Results step (**Run Again with These Answers**) is set the same way, in orange,
  when the run keeps the profile the detector has (section 1.2).
- When the run applies a calibration instead — one named in the beamtime notes, or one it found for a
  detector without a profile — the answered αi is saved with it in the instrument profile (section 1.3).
- Sputter ▸ **Show in Analyze** sets αi as if you had typed it, and may set the beam centre and choose
  a profile (section 8.6).

How a profile is chosen and what each override replaces is in
[Physics and Methods, section 1.4](Physics_Manual.md#14-where-the-geometry-of-a-frame-comes-from).

**The beam centre in file headers** is not used by default: the calibrated profile centre is kept,
because many beamlines never update the header. **Tools ▸ Settings… ▸ Analyze ▸ Use the beam centre
written in the file header** changes that (section 3.11). **Beam centre ▸ Use File Header Centre**
takes the header centre as the centre of this session (section 4.5).

### 4.2 Three ways to a geometry

When no profile matches, a banner over the image says so and offers three buttons. The step
**Geometry** has **Find Calibration Automatically** and **Calibrate…** too, and **Geometry…** in
place of **Enter Geometry…**.

**Find Calibration Automatically.** GIMaP takes the first of these that fits the frame:

1. a calibration you named in the beamtime notes of the **Results** step: by path, or by file name
   alone when the file lies in the frame's folder or in a folder the notes name;
2. the instrument profile this detector already has;
3. ready calibration files (`.poni`, an exported `.json`) in the frame's folder and the folders around
   it (beside it, and up to three levels up). They are ranked by folder first: the frame's own
   folder, then its subfolders, the parent folder, the folders beside the frame's, and further up.
   Within one of these, the file written closest in time to the frame comes first, one written before
   the frame is preferred;
4. images of a standard there (AgBh, LaB₆, CeO₂), fitted and accepted when the standard's lines land
   within 0.5 % of their q.

The step **Results** says which calibration was used and where it was found. A `.poni` or `.json`
file is taken as it is: it is not checked against a standard image. It is skipped when it states a
pixel size that differs from the frame's by more than 1 %, or a frame size that differs. But a TIFF
often states no pixel size and a `.poni` often no frame size, and then nothing tells two detectors
apart: the calibration of another detector in a neighbouring folder can be applied when the frame's
own folder holds none. The GIWAXS `.poni` of section 1.1, for example, fits the GALAXI TIFF as far
as the search can tell; only its folder puts it after `galaxi_bornagain.poni`. The result is saved
as the instrument profile of the detector.

**A tilted detector.** GIMaP's geometry is a flat detector perpendicular to the beam. A `.poni` with a
tilt (pyFAI's `Rot1`, `Rot2`) is converted to the direct beam and its distance, and the tilt is left out;
the instrument profile keeps it, and for every frame GIMaP compares |q| of the tilted geometry with that of the
flat geometry made from the same `.poni` on the valid pixels ([Physics and Methods, section 3.7](Physics_Manual.md#37-what-a-poni-tilt-left-out-costs-in-q)).
The estimate is what the tilt alone costs: a beam centre moved afterwards (by the GISAXS symmetry axis of the
automatic analysis, or by hand) does not change it.
When the tilt exceeds 1° or |q| is off by more than 0.5 %, Analyze's status line and the step **Geometry**
(marked with a warning) say so (*tilt 6.06° left out: |q| up to 6.3 % off*), and the automatic analysis lists **Detector
tilt** among the points to check (`tools/gimap_agent.py`: under *Needs attention*, exit code 2). Below both
limits the decision **Detector tilt** says how much it changes |q|. The tilt and the error are written into
`<stem>_analysis.json` (chapter 12). The Xeuss example of **Help ▸ Example Data** is such a case: its 6.06°
tilt puts |q| off by up to 6.3 % at the corner farthest from the beam. For exact q of a tilted detector,
reduce the frame with the full `.poni` geometry (pyFAI, for example).

The search may pick the calibration of another experiment: one in a folder beside or above the
frame's when the frame's own folder holds none, or, among several in one folder that was copied in
one go, the wrong one (the file times then say nothing). Check the file named and the distance shown in
**Geometry**, or name the right one in the notes (a calibration named there is tried before the
profile and the search), or use one of the two other ways. A wrong profile is removed with
**Geometry…** ▸ **Delete profile**.

**Enter Geometry…** opens the dialog **Detector geometry**: **Profile name**,
**Sample–detector distance**, **Pixel size x** / **y**, **Wavelength**, **Beam centre x** / **y**
(section 4.4), **Centre step**, **Incidence angle αi**.

- **Save Profile** saves every value as an instrument profile for this detector and frame size. The
  saved geometry is used at once: a centre set earlier in this session for this detector is dropped,
  and **Instrument profile** goes back to **Automatic (match detector)**.
- **Apply Centre** uses the beam centre for this session only (section 4.5); the other values take
  effect only with **Save Profile**. With no profile in use, save one first.
- In the step Geometry the same dialog opens with **Geometry…**, filled with the geometry in use. Save
  it under the same name to change the profile, under a new name to keep both. **Delete profile**
  removes it.

**Calibrate…** opens the Geometry Calibration window (section 4.3) with the frame shown; Analyze
waits until you close it. Use it when the frame is an image of a standard, or open the standard's
image there. **Apply** in that window (also when opened from the **Tools** menu) saves the
profile, and Analyze analyses the frame again with it; a beam centre set earlier in this
session is dropped in favour of the calibrated one.

A fourth button, **Use Previous Geometry…**, appears only after an update from 0.0.2: it saves the
geometry of the former Cut & Fitting page as a profile.

![The Detector geometry dialog](manual_images/en/04_geometry_dialog.png)
*Geometry…: the geometry of the profile in use, with Live Preview, Apply Centre and Save Profile;
the Beam centre button on the left shows a centre set for this session.*

### 4.3 The Geometry Calibration window

**Tools ▸ Geometry Calibration…** (`Ctrl+Shift+G`) calibrates a SAXS, GISAXS or GIWAXS detector
geometry without leaving the current page. It also opens from **Calibrate…** in Analyze and from the
Start page (*Calibrate the detector*).

1. Open an image of a calibration standard (NeXus/HDF5 such as `.nxs` or `.h5`, `.cbf`, `.tif` /
   `.tiff` or `.edf`) with **Open…**, or paste its path into the image field and press Enter. An image
   can also be dropped on the window. NeXus/HDF5 files are read as Analyze reads them (module files
   stitched, the same orientation; section 5.1). For a file of several frames, a row **Frames:** chooses
   **A single frame** (its number, counted from 1) or **Sum of all frames** (summed while you wait, with
   progress; it can be cancelled). A file with several equally likely image datasets asks which one.
2. Confirm the detected energy and pixel size. For a CBF with matching scan NXS metadata, the energy
   is filled in automatically. Enter the energy in keV if it is still missing.
3. Confirm the detector model. If it cannot be identified, choose a known detector from
   **Detector model**, or select **Custom pixel size** and enter the values in
   **Advanced configuration**. *Auto detected* keeps the pixel size of the file.
4. Choose the standard. When the path of the image — its file name or a folder above it — names one
   standard (for example `AgBh`, `LaB6` or `CeO2`), it is selected for you; when it names more than
   one, the field is set to **Auto Detect**; when it names none, the field is left as it was
   (**Auto Detect** when the window opens). Auto Detect tries every known standard.
5. **Estimated distance** is optional and steers the search: leave it on *Optional* when you do not
   know the distance, type the approximate distance when you do. When an image is opened the field is
   filled in only if the image file states a distance (a NeXus file can; the distance in a CBF or EDF
   header is not used) or, failing that, if a calibration was applied before, with the distance of
   that calibration: check that such a number is the distance of this set-up. **Distance range**
   limits the search (*Auto · 30–10000 mm*, *SAXS · 500–10000 mm*, *WAXS · 30–1500 mm*, or *Custom*
   with the bounds in **Advanced configuration**).
6. Click **Auto Calibration**. The calculation runs in the background; **Cancel** stops it after the
   current step.
7. Review the result: beam centre, distance, matched rings, RMS residual, confidence and warning of
   the **Selected solution** (a warning is marked ⚠; its tooltip says why it matters), and the
   **Candidate solutions** below. The line above the preview names only the image and its size. Selecting a candidate draws its
   rings. **Clean image** temporarily hides all overlays, **Reset view** restores the complete
   detector mosaic after zooming or panning, and **Focus image** hides the result panels to give tall
   WAXS mosaics more room. The horizontal divider can be dragged.
8. To adjust the result by hand, press **Manual refine**: drag the centre marker, edit the
   centre/distance values, or pair a detected ring with a theoretical peak (**Fit selected ring**).
   **Reset to fitted** goes back to the fitted solution and **Finish manual** closes the panel. The
   fitted values stay as they are until you press **Manual refine**.
9. Click **Apply** to save the geometry as the instrument profile of the detector. One note says what
   was applied and which profile Analyze will use. When the result moves the beam centre of a saved
   profile by more than 10 px or its distance by more than 5 %, the window asks before it overwrites
   that profile.

**Export Calibration…** writes the result as a `.json` file. It proposes
`<image name>.gimap-calibration.json` next to the calibration image; choose another folder when the
data folder should stay untouched. The record names the frame used, or that frames were summed. The
note after it opens the folder. **Import Calibration…** reads such a file back and selects its standard;
for a calibration made on summed frames it shows the first frame at once, then reads the same sum. An
exported `.json` can also be dropped on the window. **Open…** and **Import
Calibration…** start in the folder of the image shown, else in the folder used last.

**The overlays.** Solid yellow lines are matched theoretical rings, dashed orange lines are unused
theoretical rings, dotted cyan lines are detected experimental radii, and pink lines cross at the
beam centre; the legend above the image names each style. Partial WAXS arcs and centres outside the active detector area are
supported. A low-confidence result or a one-ring result should be treated as ambiguous and reviewed
by hand. How the rings are fitted, and what the RMS residual and the confidence measure, is in
[Physics and Methods, chapter 3](Physics_Manual.md#3-calibration).

![The Geometry Calibration window](manual_images/en/03_calibration.png)
*Geometry Calibration: the fitted rings over the standard, the selected solution and the candidates.*

**The beam centre in this window** is shown and entered in the same pixel coordinates as in Analyze
(section 4.4). A centre can be compared or copied between this window and Analyze as it is. An
exported calibration file keeps its numbers as in earlier versions: there the beam centres count
pixel centres from 0, half a pixel lower than the window shows, and the file says so
(`metadata.center_convention`). Importing the file shows the same centre again.

**Known limitation: a wrong solution can come first.** Both cases below were seen on a silver
behenate image whose path does not name the standard.

- With the standard left on **Auto Detect** and no estimated distance, a solution of another standard
  can be ranked first: a LaB₆ solution at a much shorter distance came first, and the silver behenate
  solution was last in the list of candidates. Choose the standard yourself when you know it. An
  approximate distance helps Auto Detect too: with 1500 to 1800 mm typed in, the silver behenate
  solution came first on this image.
- A wrong estimated distance can mislead, also with the right standard: the search prefers solutions
  near the value in **Estimated distance**. With silver behenate chosen and 2000 mm typed in, a
  solution near 1830 mm came first, although the real distance is about 1630 mm; with the field left
  on *Optional*, the right solution came first.

**Estimated distance** can hold a number you did not type. It is never filled with a built-in value:
on a new installation it stays on *Optional*. But each time an image is opened it takes the distance
the image file states, or else the distance of the calibration applied last, also when you had typed
another value or cleared the field; with neither, a value you typed stays. The calibration applied
last may be the one of another set-up or detector. Enter the approximate distance of your set-up, or
set the field to 0 (it then shows *Optional*) when you do not know it.

Before you press **Apply**, check that the distance is the one of your set-up and that the drawn rings
lie on the rings of the image. This will be corrected in a later version.

### 4.4 Pixel coordinates

Analyze, Geometry Calibration and the XRR Series Extractor show and accept the beam centre in the
same pixel coordinates:

- pixel corners lie on whole numbers, so the centre of the first pixel is (0.5, 0.5);
- x counts from the left edge and y from the top edge of the image as shown (row 0 at the top);
- the beam centre is the position of the direct beam.

The same profile therefore shows the same numbers in all three windows, and a centre can be copied
from one to another as it is.

Other programs may count differently. The GALAXI frame of section 1.3, for example, documents its
direct beam at (597.1, 323.4) px counted from the bottom-left corner; in GIMaP that is
(597.1, 719.6) px.

**Exported files count from 0.** The calibration `.json` of Geometry Calibration and the CSV and JSON
records of the XRR Series Extractor count pixel centres from 0, half a pixel lower than the windows
show, as earlier versions did. Each file says so (`metadata.center_convention` in the calibration
file, `center_convention` in the XRR record). Importing such a file shows the window's numbers again.

Trainset Build (Labs) keeps the conventions of the training data it was made for (chapter 11). The
conventions and the conversion between them are defined in
[Physics and Methods, sections 1.2 and 1.3](Physics_Manual.md#12-canonical-pixel-coordinates).

### 4.5 The beam centre in Analyze

The profile's beam centre is used unless you change it. The step **Geometry** shows it on the button
**Beam centre**, to four decimal places, with where it comes from: `Beam x, y px · profile`,
`· this session` or `· file header`. Its menu has:

| Entry | What it does |
|---|---|
| **Pick on Image** | Click the direct beam on the detector image (view **Detector**, not the q map; `Esc` cancels) |
| **Enter Coordinates…** | Type the centre in a small window (below) |
| **Use File Header Centre** | Take the centre written in the file, when it has one (the entry shows it) |
| **Refine x by Symmetry** | GISAXS: the same search as **Make Left and Right Symmetric** (section 4.6) |
| **Back to Profile Centre** | Forget the centre set in this session |
| **Save to Profile** | Write the session centre into the instrument profile |

You can also drag the cyan cross on the image. A centre set in any of these ways holds for this
session and for every frame of the same detector size (rows × columns); frames of another size go
back to their profile's centre (or the header centre, when the setting above is on), and a note says
so. It is also kept in settings files and projects. **Save to Profile**
keeps it for good. Sputter ▸ **Show in Analyze** sets a centre for this session in the same way when
one was given for its Predict (section 8.6).

**Geometry…** and **Enter Coordinates…** open small windows that stay open while you work, so you
can adjust the centre and watch the curves:

- the coordinates show four decimal places;
- **Centre step** sets how far the arrow buttons, the arrow keys and the mouse wheel move the centre
  (down to 0.0001 px); both windows share it and it is remembered;
- **Apply Centre** uses the centre for this session and redraws the curves; adjust and apply as often
  as you like;
- with **Live Preview** on, every change redraws the curves by itself. Quick changes are combined,
  and a zoom you set on the curves is kept;
- **Save Profile** (in Geometry…) saves all geometry values and leaves the window open;
- **Close** keeps the centre applied last and drops values not applied.

The windows close when you change the file or frame, or clear the list, so that a centre typed for
one detector is never applied to another.

### 4.6 Make Left and Right Symmetric

In GISAXS the horizontal cut is symmetric about qy = 0. When the beam centre column is a little off,
the two halves of I(qy) do not match. **Make Left and Right Symmetric** (step **Cuts**, GISAXS) moves
the centre column onto the symmetry axis of the cut:

1. a coarse search over the whole horizontal cut finds the axis;
2. a fine search, a few pixels around it, compares the column means of the valid pixels left and
   right of the trial axis. Only columns measured on both sides count (gaps and masked columns stay
   out, nothing is bridged across a gap). Each pair is weighted by 1 / distance from the axis, so
   the many far columns do not outweigh the low-q structure.

The scores, weights and acceptance rules are in
[Physics and Methods, section 8.2](Physics_Manual.md#82-beam-centre-column-from-mirror-symmetry).
Only x changes; y is kept. The result is shown in a note (*coarse … → refined … px*). 0.0001 px is
the numerical stopping tolerance, not a measured uncertainty. When the data do not cover enough, or
several axes fit equally well, the centre stays as it was and the note says why.

The search runs in the background; press the button again (*Cancel Centre Search*) to stop it. The
new centre holds for this session (section 4.5); **Beam centre ▸ Save to Profile** keeps it. The
automatic analysis of a GISAXS frame does the same step by itself.

## 5. Analyze

Analyze is the one workspace for detector frames, GISAXS and GIWAXS alike. It turns a frame into
curves in q (Å⁻¹), shows where every point of a curve comes from, and writes the curves with a
record of how they were made. Fitting only fits curves; everything that touches the detector image
happens here.

The page has five parts (the picture in section 3.1 shows them with a GIWAXS frame):

| Part | What it holds |
|---|---|
| Command bar | **Open…** (its arrow: **Open Folder…**), ↶ ↷ (Undo / Redo), ‹ › (previous / next frame, when more than one frame is listed), the mode **Auto / GISAXS / GIWAXS**, αi, **Run Automatic Analysis** (with **Stop** while it runs), **Ask AI…**, **Export** ▾, **Batch Export…** (when more than one frame is listed), **Send to Fitting** ▾ |
| Steps (left) | **Data**, **Geometry**, **Mask & corrections**, **Cuts**, **Results**, **Export**. Each shows its state (✓, !, × or …, as in 3.3) and one line of what it found; click a step for its controls. Rarely used options are folded away. |
| Image (middle) | **Detector**, **q map** or **Cake** (GIWAXS), **Log**, the colour map, **Levels**, **Marks**, Zoom, **Fit view**, **Sources**, **Save** ▾; the values under the cursor below the image |
| Tabs (right) | **Curves** (an upper and a lower plot), **Results** (what the automatic analysis or the AI found, with its evidence), **Series** (every frame as intensity against frame and q) |
| Status line | What is happening, a progress bar and **Cancel**. A note at the bottom right says when a file was written, with **Open Folder**. |

On a narrow window the command bar shortens its long buttons (section 3.3) and, if needed, moves the
actions to a second row.

The mode: **Auto** reduces a frame as GIWAXS when the detector reaches beyond a scattering angle
2θ = 20°, otherwise as GISAXS. Auto decides again for every frame. Choose **GISAXS** or **GIWAXS**
to force it; the status line warns when a forced mode does not look right for the frame.

The αi field shows the profile's αi until you type a value; a typed value is shown in orange and
kept until you right-click it ▸ **Back to Profile αi** (section 4.1).

Geometry — instrument profiles, finding a calibration, the beam centre and its dialogs — is
described in chapter 4. This chapter starts from a frame that has a geometry.

### 5.1 Opening data and the frame list

GIMaP reads CBF, TIFF and EDF frames, and NeXus/HDF5 files (`.nxs`, `.nx`, `.nx5`, `.h5`, `.hdf5`,
`.hdf`). Files that belong together are one *source*: the module files of one detector are stitched,
a master is read with its data files, and every frame of a multi-frame file is a frame of the series
(*NeXus and HDF5 files* below).

1. Open frames in one of these ways:
   - **Open…** in the command bar or **File ▸ Open Data…** (`Ctrl+O`), or the arrow of **Open…** ▸
     **Open Folder…** or **File ▸ Open Folder…** (`Ctrl+Shift+O`);
   - drop files or a folder anywhere on the window;
   - the Start page or **File ▸ Open Recent**.
2. Analyze opens at once and looks at the sources in the background (**Cancel Import** stops a slow
   import). A folder that holds no frames itself but has them one level down (one subfolder per
   detector) is opened one level down, and the status line says which subfolders.
3. For a source of several frames, the window **Import Frames** asks which frames to list, for each
   source on its own: **All Frames**, **Frame Range** (first **to** last, **every** n-th) or **Specific
   Frames** (frame numbers separated by commas, counted from 1, for example `1, 5, 20`). Only references
   are listed: a frame is read when you show it, so a stack of thousands of frames opens quickly.
4. The **Data** step names the source while the image loads, then its detector, frame size and
   frames. The curves appear without further clicks once the frame has a geometry. Without one, a
   banner above the image offers **Find Calibration Automatically**, **Enter Geometry…** and
   **Calibrate…** (section 4.2).

**The frame list** (Data step) has one row per imported frame, whether it comes from a file of its own
or from a NeXus stack:

- Click a row to read and show that frame. `Page Up` / `Page Down`, or ‹ › in the command bar, step
  through the imported frames only.
- For a source of several frames a bar above the steps, **Frame [4] of 12**, chooses the frame
  (counted from 1). A frame that was not imported must first be added with **Choose Frames…**.
- **Choose Frames…** (under the list) changes the import selection later (**Keep Current Selection**
  leaves a source as it is). Build Map, Batch Export and Send Series to Fitting take the imported frames,
  then apply their own range, interval and sum. A sum never bridges frames that were left out; frame
  numbers, times and the files named in the records keep their original meaning. Changing the selection
  clears the Series map and the batch-fit preview of the old one. A project keeps the selection.
- To take a source off the list, right-click one of its rows — **Remove Source from List**, **Show in
  Folder**, **Copy Path** — or select it and press `Delete` or `Backspace`. All its frames go; the files
  themselves are not touched, and the rest of the list, the Series map and Batch Export go on without it.
- **Clear** (or **Analyze ▸ Clear Frame List**) empties the list, and the Series map with it.
- **Frames ▸ Sum** (folded under the list) adds consecutive frames for better statistics: the
  frame shown plus the next ones (the following files, or the following frames of a NeXus file).
  Batch Export and watching sum groups of the same size.
- Above the list the Data step says what the source shown consists of (for example *11 module files ·
  240 frames · stitched*), how its frames are oriented, and — in the warning colour — what is missing
  from it. **File header** gives the energy, distance and exposure (in s) the file states.
- When the files of a frame change (name, time or size), **F5** (**Analyze ▸ Repeat Analysis**), showing
  the frame again and watching read it again; otherwise it is not read again.

**Watch** (next to **Clear**) watches a folder during a measurement: choose the folder, and it is
looked at every 2 s. New frames are listed and the newest is shown as soon as it is completely
written. Press **Watch** again to stop.

- A file, or a set of module files, that grows frame by frame while the detector writes is followed
  while it grows: each look takes the frames it already held at the look before, so the newest frame,
  which may still be being written, waits for the next look. The Data step says *Still growing: frame N
  is the newest complete one*. A Series map or Batch Export started meanwhile takes only these frames.
  A program on the same Windows PC that writes the file with HDF5's file locking locks it against
  reading: such a file is listed as being written and read once it is closed.
- With **Export ▾ ▸ Export New Frames While Watching** ticked, each complete new frame is exported
  once (next to the data, in `gimap_analysis/`; with **Sum**, only complete groups). A set of module
  files is exported once it is whole: at once when it has as many module files as another set of the
  folder, otherwise after its files have stayed unchanged for about 10 s. A set with fewer module files
  than another set waits about 30 s and is then exported as it is (a detector may have had a module
  switched off). A module file that is missing or cannot be read is named in the status line, and the
  set waits until it is there.
- A Series map that is built already grows with the new frames, without a new **Build Map** (when
  export is on and there is no map yet, one is built with the chosen curve). Picking a row stops the map
  from following the newest frame; **Show the newest frame** under the Series options turns it back on.
- A file another program holds open is tried again at every look and said once in the status line; a
  source taken off the list comes back when it grows.

#### NeXus and HDF5 files

What a source is follows from what the files hold; a file name only proposes:

- **one frame per file** — each file is a frame; a folder of them is a series;
- **a series in one file** — a stack of frames (a grid of scan points × frames is read in storage
  order, and the Data step says so);
- **one file per detector module** (P03 Lambda 9M: `…_m01.nxs` … `…_m11.nxs`, each holding every
  frame) — one stitched frame per frame, named by the scan without `_m01`. The files are a set only
  when they hold the same dataset and module size at different module positions; files merely named
  alike stay separate, and the Data step says why;
- **a master with its data files** (Eiger: `…_master.h5` with `…_data_000001.h5` …) — one source whose
  frames run through the data files.

HDF5 files without a detector image (model weights, result tables) are not listed from a folder.

**The image** is the dataset the file marks as its default plot, else the `data` of the detector, else
the best image-shaped dataset; masks, flat fields and dark frames are never taken for the image. A file
with several images shows one and names the others (the Format Converter and Geometry Calibration let
you choose).

**Orientation.** Files written by the Lambda server (P03) are shown as GIMaP always showed them: the
modules stitched by their positions, then transposed and flipped (rotated by 90°); the Data step says
*P03 Lambda: transposed and flipped*. Every other NeXus/HDF5 image is read as stored, row 0 on top, like
a CBF or TIFF of the same detector (*Image as stored in the file*). Earlier versions turned these other
files as well: a geometry or mask made on such a file with an earlier version no longer fits it. Check
it (a square frame still matches its old instrument profile by size).

**Values.** Header values are read in their units — an exposure written in ms is shown in s, the energy
in eV or keV, the distance in mm or m; a value in a unit GIMaP does not know is left out, not guessed.
The detector's `pixel_mask` is applied, and in unsigned integer data the largest value of the type means
“no data”.

**What is missing is said, never filled in**: a module file that is missing (that part of the detector
stays empty), module files with different numbers of frames (the frames every module has are used), a
data file that is missing (the frames after it cannot be read), a frame beyond the end (an error, never
the last frame instead). A file still being written is read as far as it is and read again when it
changes; a file another program holds cannot be read until it is closed; a file left open by a program
that ended says so (`h5clear -s` on a copy of it makes it readable). Files compressed with bitshuffle or
LZ4 need the Python package `hdf5plugin` (in `requirements.txt`).

### 5.2 Mask and corrections

The **Mask & corrections** step says how many pixels are left out, and why: detector gaps and
marked pixels, hot and dead pixels found in this frame, your masks, the gap guard, the intensity
range, the background, and how many pixels were filled from the mirror side.

![The Mask & corrections step](manual_images/en/05_mask_step.png)

*The summary at the top counts each kind of pixel left out; on the image, amber circles mark the hot pixels found in this frame and the red rectangle is a drawn mask.*

**Show Masked Pixels on the Image** marks them on the detector image: red = left out, amber
circles = hot or dead pixels, cyan = filled from the mirror side (used in the curves).

**Which pixels count as data.** On an integer counting detector, negative values are gap codes
and are left out. On a floating-point frame (dark- or background-subtracted) negative values are
data and are kept.

**Bad pixels** — **Leave out hot and dead pixels** (on by default, remembered). In every frame
GIMaP leaves out isolated hot pixels, dead pixels (counting detectors only) and scattered bright
pixels along a defective detector row or column; the criteria are in
[Physics and Methods, section 4.4](Physics_Manual.md#44-bad-pixels-found-in-the-frame). Anything wider
than one pixel — a Bragg spot, a streak, the beam-stop edge, a Yoneda band — is kept, and whole rows
or columns are never removed. The numbers go into the export record (`bad_pixels_left_out`). Turn the
option off to put these pixels back into the curves.

**Masks you draw:**

1. Press **Rectangle** and click two opposite corners on the detector image, or press **Polygon**
   and click the corners (double-click or `Enter` closes it, `Backspace` removes the last corner,
   `Esc` cancels).
2. The mask is outlined in red and left out of every curve at once.
3. **Remove Selected** (or `Delete` in the list) and **Clear Masks** remove masks; the detector
   gaps stay masked.
4. **Save…** writes the drawn masks as `.json` (in the pixel coordinates of section 4.4). **Load…**
   reads such a file, or a mask image of the same size (EDF or TIFF, non-zero = masked, as in pyFAI).

**Fill gaps from the mirror side (GIWAXS)** (off by default). A GIWAXS pattern is symmetric in
±q∥. A pixel without data (module gap, mask, hot pixel) takes the value at its mirror position
about the beam-centre column, interpolated between two columns that both have data. Where the
mirror has no data either, the gap stays. A filled pixel repeats photons measured on the other side,
so while this is on the error bars of every GIWAXS curve come from the scatter of the pixels, not
from counting statistics (section 5.4). The record counts the filled pixels, and Fitting is told
that the counting statistics no longer hold.

**Corrections** (folded): a background frame (**Choose…**, with **Scale** and, for NeXus, **Frame**),
a valid raw intensity range (**Min** / **Max**), and the **Gap guard** (default 3 px; 0–20 px,
remembered), which also leaves out pixels this close to gaps and bad pixels.

The order is fixed: invalid detector pixels → hot and dead pixels → masks → background → gap guard
→ intensity range → mirror fill → intensity corrections. Curves, q map, cake and exports all use
the corrected frame.

**Intensity corrections (GIWAXS)** (folded, off by default). They divide each pixel by a geometric
factor; q does not change:

| Correction | Enter |
|---|---|
| **Solid angle of each pixel (cos³2θ)** | — |
| **Polarisation, factor** | f (default 0.99): 0.95–0.99 at a synchrotron (horizontal), 0 for an unpolarised laboratory source, −1 vertical |
| **Absorption in the film** | **Thickness** (nm), **Attenuation length** 1/µ (µm, from tables such as CXRO or NIST) |

Peak positions do not change; intensities compared across χ or q do — orientation, pole figures,
crystallinity ratios. The film absorption leaves out refraction, so it holds well above the critical
angle; with it on, pixels at or below the sample horizon are left out. The factors, and how the error
bars follow them, are in [Physics and Methods, chapter 5](Physics_Manual.md#5-intensity-corrections-for-giwaxs);
the JSON record (`intensity_corrections`) gives the formulas and parameters.

### 5.3 Cuts

The **Cuts** step holds the cut positions. **Auto Cuts** returns to the automatic positions.

#### GISAXS: horizontal and vertical cut

GIMaP finds the Yoneda band (the brightest row between the sample horizon and an exit angle of
1.5°, measured beside the specular rod) and takes:

- the **horizontal cut** I(qy) in a band 5 px high at the Yoneda row (orange on the image);
- the **vertical cut** I(qz) in a band 10 px wide at the beam-centre column.

If no Yoneda band is found, the horizontal cut sits just above the horizon and the step says so —
check αi. To move a cut, drag the orange band on the image (its edges change its height), or
double-click a column to move the vertical cut. The step says whether the cut is at the Yoneda
band, set by hand, or above the horizon.

**Halves of the horizontal cut.** qy is signed: the two sides of the beam are two measurements of
the same structure. Choose which the curve for Fitting uses:

| Choice | Curve for Fitting |
|---|---|
| **Both halves on \|qy\| (two colours)** | both halves, folded onto \|qy\| |
| **Mean of both halves** | the average of the two |
| **qy < 0 half** / **qy > 0 half** | one side only |

The same choice is in **Send to Fitting ▾** and in **Tools ▸ Settings… ▸ Analyze**. **Make Left and Right
Symmetric** moves the beam-centre column onto the symmetry axis of the cut (only x changes;
section 4.6 describes how).

#### GIWAXS: regions, Ring / Sector / Spot

A GIWAXS frame gives I(q) in the upper plot and I(χ) in the lower plot. χ = 0° is the surface
normal and ±90° the sample plane.

![GIWAXS cuts on the Cake view](manual_images/en/06_cuts_cake.png)

*Each region has its own colour, the same in the list, on the q map, on the Cake view and in the plots.*

Three views above the image:

- **Detector** — the frame as measured, in pixels;
- **q map** — the intensity regridded onto q∥ (qr) and qz;
- **Cake** — unwrapped onto χ against q: a ring is a vertical line, a region a rectangle.

**Save ▾** above the image: **View as Figure…** (the view with its colour bar) or **View Data as
CSV…** (the q map or the cake as a table with its axes).

**Regions.** A region is a q range × a χ range, optionally on both sides (±χ, folded onto |χ|).
Each region gives two curves in its own colour: I(q) (upper plot) and I(χ) (lower plot). The list
starts with the standard cuts — the full ring, the in-plane band, the out-of-plane band and the
ring of I(χ) — which the automatic analysis and the AI also use. Tick a region to show it.

**Add a cut with one click.** Press **Ring**, **Sector** or **Spot** under *Click on the image to
add a cut*, then click on the detector image, the q map or the Cake view:

| Button | Click | You get |
|---|---|---|
| **Ring** | a peak (also on the upper I(q) plot) | the peak's q window × every χ: I(χ) for orientation |
| **Sector** | a direction | every q within a few degrees of that χ (±5°, or the spot's width): I(q) along it |
| **Spot** | a spot | the spot's q window × its χ window: both profiles of the spot |

The region snaps to the peak: centre and FWHM are fitted from the pixels, and the window is
centre ± FWHM, without reaching into a neighbouring peak. A note gives q, d and the FWHM, with
Undo. If there is no clear peak near the click, a fixed width is used and the note says so. New
regions are named by position, for example `Ring q 0.998` or `Sector |χ| 85°`.

**Change a region exactly.** Select it and type its centre ± **Half width** (q to four decimals,
χ to 0.1°), or tick **All q**, **All χ**, **Both sides (±χ)**. The line below shows the range and
d = 2π/q. **Snap to Peak** re-centres the window on the nearest peak. **Remove Region** deletes it.
Regions can also be moved and resized on the Cake view.

**Add ▾** also has **Draw on the Cake View** (two corners), **Ring at the Strongest Peak**,
**In-plane Band (|χ| 70–90°)**, **Out-of-plane Band (|χ| 0–20°)**, **Whole Pattern**, and
**Save Cuts…** / **Load Cuts…** (a set of regions as JSON for the next data set; loading replaces
your regions, the standard cuts stay). **Strongest Ring** puts the I(χ) window back on the most
prominent ring.

**More cut settings** (folded): **Radial bins** (auto by default), the **I(q) axis** (q or 2θ),
and a **q box** (a q∥ × qz rectangle, values in Å⁻¹, draggable on the q map), which gives I(qz)
and I(q∥). The lower plot's drop-down chooses **I(χ) of the regions**, **q box: I(qz)** or
**q box: I(q∥)**.

### 5.4 Curves and results

**Where the points come from.** Each point is the mean intensity of the valid pixels in one bin,
never finer than one pixel. Bins with very few pixels (near the beam, at region edges) are merged
with their neighbours along q, never across a gap; I(χ) is not merged. Bins with no pixel — gaps,
masked χ ranges, the missing wedge — have no point, and the curve is broken there. The error bar σ is
the counting error for raw photon counts (a bin of zero counts gets the error of one count, so σ is never
0), and the scatter of the pixels for floating-point frames,
after background subtraction or with mirror fill ([Physics and Methods, sections 6.4 and
6.5](Physics_Manual.md#64-the-three-error-models) define both and the merging). GIMaP never smooths a
curve: for smoother curves widen the region's χ range or lower **Radial bins**.

**Sources.** Press **Sources** to see on the detector image which pixels each curve comes from, in
the colours of the plots. Click a curve to see only its region.

**The plots.** Each curve plot has:

- **Log I** and **Log q** (log axes; for an I(χ) or 2θ plot the second reads **Log χ** / **Log 2θ**).
  With a log x axis, points at x ≤ 0 are not drawn: to see both halves of a GISAXS cut on a log q
  axis, choose |qy| (below). A window band you placed keeps its data range when you switch the axis.
- For a signed x (qy of the horizontal cut, χ of a ring): **±** (both halves), **+**, **−**, or
  **|qy|** / **|χ|** (both halves on |x|, the negative one dashed in a second shade). This changes
  only the display and **Plot as Figure…**, not the exported data or the curve for Fitting.
- Zoom (drag a rectangle; `Shift` + drag works in every view), Reset, and the Marks (eye) button.
- **Save ▾**: **Plot as Figure…** (a publication figure) or **Curves as Data…** (one CSV per curve
  plus a JSON record).
- The right-button menu (**Copy Image**, **Copy Data** …), zooming with the right button and the
  **⋯** menu of a narrow plot work as in every plot (section 3.4).

**Colour limits.** Next to every image is a histogram of the displayed values with two handles.
Drag a handle, or the band between them, for a quick change; the mouse wheel over the bar zooms it
for finer moves. **Levels ▾** has:

- **Auto for every frame** and its **Rule**: 1–99.7 %, 0.1–99.9 %, 5–95 %, Min–max, Mean ± 3σ;
- **Min** / **Max** typed in intensity units;
- **Auto Once** (a click on **Levels** itself does the same).

Limits you drag or type stay — for other frames, re-analysis and the live Series map — until you
choose Auto; the button then reads **Levels (fixed)**. They mean the same in log and linear
display. The detector, q map, cake and Series map each keep their own limits.

**Marks.** Every image has **Marks ▾**, every curve plot an eye button: one switch per kind of
mark — beam centre, sample horizon, cut bands, masks and cut regions, the pixel overlay, marked hot
and dead pixels, the q box, the colour bar, the window band — and **Show All Marks** / **Hide All
Marks**. Hidden marks stay hidden for other frames and the next session. The same menu removes
what you made: **Remove the Drawn Masks**, **Remove All Cut Regions**, **Remove the q Box** (Undo
brings them back). On the q map the beam centre is the direct beam (q∥ = 0, qz = 0), just below
the map; the sample horizon is drawn dashed at qz = k sin αi.

**Undo / Redo** (↶ ↷, `Ctrl+Z`, `Ctrl+Shift+Z` or `Ctrl+Y`). Every change of the set-up is one
step: a mask, a region added or moved, a band dragged, the profile, αi, the beam centre, the mode,
the corrections, the frames summed, a settings file loaded, an AI card applied. A drag, or a quick series of the same edit, is one
step. The tooltip says what the next Undo changes (for example *Undo: cut regions*). Saved
instrument profiles are not undone this way.

### 5.5 Run Automatic Analysis

**Run Automatic Analysis** runs GIMaP's standard procedure on the frame shown. It needs no AI and
works for GIWAXS and GISAXS.

Before running, you can type **Beamtime notes** in the **Results** step: αi, the energy, the
pixel size, where the calibration is — whatever you know. An αi you set in the command bar is the αi the run
uses: it is not asked for again, and the report's *Decisions* say it came from the command bar. A value from the
notes or an answer that differs from it replaces it there, so the αi reported is always the one used.

**GIWAXS.** The run:

1. finds the geometry (the instrument profile, a calibration file near the data, or an image of a
   standard);
2. sums the last frames of a series: the last fifth of its frames, at most 10 (of fewer than ten
   frames, the last one alone);
3. finds the peaks and fits them;
4. compares in-plane and out-of-plane;
5. gives the orientation of the strongest reliable ring and Scherrer sizes.

The Results tab shows **Peaks**, **Checks** (shadows, artefacts, unmeasured ranges), **Geometry**
(where it came from: distance, beam centre, wavelength, αi), **Series** (with **Compare with the
Start of the Series**) and **Report**. Sizes are Scherrer lower bounds. Phases are not assigned
automatically.

What the numbers are is said in the report's *Decisions* (**Show the Full Text**): q is kinematic, without a
refraction correction, and the line gives how far refraction can shift q_z near or below the film's critical
angle at your αi (up to about k·sin αi); the uncertainties are statistical only, from Poisson counting statistics
or, for a floating-point, background-subtracted or mirror-filled frame, from the pixel scatter, without the
geometry. At most 12 candidate peaks, the most prominent, are fitted: when more qualify, *Decisions* says how many,
and the peak table, the rings and the series hints are those of the 12 (the command line's `--max-peaks` fits
more). The orientation sentence compares f with f_iso, the value of a random ring over the same measured |χ|: it
says *along the surface normal* or *in the sample plane* only when the I(χ) maximum lies below 20° or above 70°
and |f − f_iso| reaches max(0.1, 3σ_f); a ring whose maximum lies between 20° and 70° is called *tilted* at that
|χ| from its maximum alone, without that margin. Every ring line gives f ± σ_f, f_iso and f − f_iso with the
decimals of σ_f. When f changes by more than that statistical uncertainty once the bins next to the sample horizon
(|χ| ≥ 85°) are left out, the ring's notes and *Decisions* say so: there the exit angle is smallest, and absorption
or a shadow can lower the intensity, which GIMaP does not correct. A peak within 1.5 widths of the end of the data
(*at the end of the data: check*) is not used for the rings, the sizes or the series hints: its shape may be cut
off.

![Results of the automatic analysis of a GIWAXS frame](manual_images/en/07_giwaxs_results.png)

*The peak table (q, d, FWHM, trust, Scherrer size L, OOP/IP); **Fit details** below it opens the fit of the selected peak.*

**GISAXS.** The run:

1. finds the geometry and, as for GIWAXS, sums the last frames of a series;
2. takes the horizontal cut at the Yoneda band (if none is found, it asks you to check αi);
3. moves the beam-centre column to the symmetry axis;
4. chooses the halves: their average when both are usable and agree, otherwise the usable one,
   with the reason (a half shadowed, cut by a gap, too short);
5. gives the length scale 2π/q\* of a side maximum, or else of a shoulder, of the cut: *a length
   scale, not necessarily the particle spacing; check it with a model fit*. A form factor or the tail
   of the direct beam makes a shoulder too, so 2π/q\* is never reported as the spacing (the report
   section is **Length scale 2π/q\***; the decision **length scale** says where, and once what it is);
6. fits I(qy) with physical models: spheres, vertical cylinders and random cylinders, with size
   spread and a distance D. This quick fit is a screening step (chapter 13): confirm the model in
   Fitting.

The solutions are ranked by a robust misfit, the mean Huber loss of the σ-normalised residuals r, so
that a few outlying points do not decide the order. The table's χ²/N is the mean of r² over
the points, with no correction for the number of free values (unlike Fitting's χ²ᵣ). A non-linear value
that ended on a bound of its search range (for example D at 500 nm) is not determined by the curve: the
table marks it ⚠ and a warning under the table names it by rank (*⚠ Not determined by the curve (at a search
bound) — 1: D, …*; which bound, *D at its upper bound*, is in its tooltip), as do **Fit details** and the report's
note column. Among the solutions within 10 % of the first solution's χ²/N, the run reports the
best-ranked one that converged with no non-linear value at a bound; the table selects it, and the line under the
table says so when it is not the first. When none of them is free of a bound, the first is reported
and a point to check (**Fit**) says that it sits at a bound. The amplitudes, the background and the
beam-tail amplitude are fitted non-negative and are not counted as values at a bound; one that ends at
0 is named among the report's *Decisions* (*fit caveat: background BG at 0, the non-negativity limit*).
A reported solution with χ²/N above 10 — the limit of Fitting's poor fits (section 6.6) — misses the
curve by far more than its noise: a point to check (**Fit**) says that its R and D are not measurements
(the AgBh calibrant of **Help ▸ Example Data** gives χ²/N = 472: Bragg rings, no particles). When several models fit within those
10 %, the curve alone does not decide the model: the Results tab asks which one you expect and names
the solutions whose D is within 15 % of 2π/q\*. **Save Fitted Curve…** (q in Å⁻¹, I, σ and the fit
the run reports), **Save Fit Table…** (every solution with its parameters in nm, χ²/N, the values at a
bound in `at_bounds` and the reported one in `chosen`) and **Save Plot…** save the results; **Refine
in Fitting** opens the prepared curve (the halves chosen) in Fitting.

As for GIWAXS, the report's *Decisions* (**Show the Full Text**) say what the numbers are: q is kinematic,
and near the critical angle refraction mostly shifts q_z — that of the Yoneda band most — by an amount of
the order of k·sin αi, which the line gives for your frame (q_y and 2π/q\* barely change); the σ of the
cuts, by which χ²/N is normalised, are statistical only, from Poisson counting statistics or the pixel
scatter, without the geometry, and the quick fit gives no ± for R, h or D.

![Results of the automatic analysis of a GISAXS frame](manual_images/en/08_gisaxs_results.png)

*The cut with its halves, the length scale 2π/q\*, the fit (Log q / Log I) and the table of solutions.*

**Fit details** (closed under the peak table or the fit table until you open it; it then stays
open for later reports) follows the selected row:

- a peak: the points it was fitted to, with the Gaussian and its local linear background; q, d and
  FWHM with errors; height, area, background and slope; S/N, χ²ᵣ, the fit window, the Scherrer
  size, and where the ring is measured, shadowed or missing on the q map (how the peak was fitted and how
  peaks are found, with the thresholds of the run, is the tooltip of **Fit details**);
- a model solution: every parameter (R, σR/R, h, D, σD/D, weight, amplitude, background,
  resolution), χ²/N, convergence, evaluations and time, the q range, the start of D, the algorithm,
  its warnings, and **Show in Fitting**, which opens the curve in Fitting with that solution drawn.

**Questions only you can answer** (αi, the energy, the pixel size, the calibration file and its
standard, the GISAXS model) become fields in the Results step. Fill them in and press **Run Again with These Answers** (or `Enter` in a field).

**While it runs** the right side switches to Results. A panel shows the step in progress and the
time, and ticks off the phases. **Stop** (in the panel and the command bar) ends the run before
its next step. A step already running finishes first, and the panel says what it waits for (a
geometry fit or model fit can take a minute). What was found is kept: **Save Report…** or
**Discard**.

**Whose results.** The results belong to the frame they were run on. While a run lasts, the file
list is locked. When you show another frame, the Results step says which frame the results are
from; they come back with that frame. **Save Report…** writes a web page with the pictures (`.html`,
opens in any browser) or the text as Markdown (`.md`); the save dialog starts next to the data.

The sum the automatic analysis sets is for the frame it analyses. A Series map, Batch Export and Send
Series to Fitting do not use it (their dialog and the Series tab say so); to sum there, set **Data ▸
Frames ▸ Sum** yourself.

**Ask AI…** lets an AI do the same with the same tools while you watch (chapter 10).

### 5.6 Series map and stages

With several frames listed — files, a folder, or a multi-frame NeXus/HDF5 file — the **Series** tab
shows one curve of every frame as an intensity map against frame and q.

![The Series map of an in-situ run](manual_images/en/09_series_map.png)

*Drag the horizontal band to pick a frame and the vertical band to pick a q window. The line above the map
names the curve, the frames and the cut held for every frame; the box beside it (±) chooses the halves of the
cut, and the frame axis counts from 1. Here a synthetic run of 40 frames: “3 stages: frames 1–17, 18–22, 23–40 ·
1 odd frame”. The odd frame is frame 34, where a few detector columns are hot (the short bright mark near
qy 0.05 Å⁻¹); the red arrow beside the map marks it, and it does not make a stage of its own.*

1. Choose the **Curve** (GISAXS: the horizontal cut by default; GIWAXS: I(q); every region's I(q)
   and I(χ) are in the list). For a first look at a long series, set the box beside **Curve** to
   *every n frame* (for example *every 5 frame*).
2. Press **Build Map**. Every imported frame (or group of summed frames) is reduced with the current
   settings; the map grows as frames are done. Long series use several low-priority processes;
   progress, time left, **Pause** and **Stop** are shown, and rows already done stay.
3. Drag the horizontal band to pick a frame: its curve is drawn below; **Open** shows that frame in
   Analyze, with the cut of the series.
4. Drag the vertical band to pick a q window: further below, against frame, the **Mean intensity** or
   the peak in the window — **Peak position**, **Peak width (2.355 σ)**, **Peak area**, **Peak height**
   (centroid, 2.355 σ, area and height after a straight background). A window in a detector gap, or
   outside the data, has no value: the plot says so instead of drawing one.
5. **Export ▾** (once a map is built): **Map as CSV Table…**, **Map as Figure…**, **Stages as
   Table…**, **Selected Frame's Curve as Figure… / as Table…**, **Trace as Figure… / Trace as
   Table…** (the value of the lower plot for every frame), **Peak Table of Every Frame…**.

**One cut for the series.** While the cut is automatic, every frame of the map — and of a Batch
Export, a Send Series to Fitting and a watched folder's export — is cut where the frame on screen is:
for GISAXS the rows of the Yoneda band and the beam-centre column, for GIWAXS the ring of I(χ). A row
of the map then differs from the next by the sample, not by where it was cut. The line under the
buttons names the cut (*Cut of the series — horizontal cut: rows 202–206 for every frame*). A cut you
set yourself is the same for every frame anyway.

**What the map shows.**

- Every point is drawn at the x where it was measured: a line sits at its q, and a detector gap stays a
  gap (the exports keep the measured x).
- The frame axis counts the frames from 1, in whole numbers — for one multi-frame file the frame in the
  file, for several files the place in the list.
- For the GISAXS horizontal cut, the box beside the map chooses the halves: **±** both as measured,
  **+** or **−** one half, **|x|** both on |x|, or **mean**. The stages, the traces and the exports
  follow the choice.
- The line above the map says what it holds: the curve, its halves, the frames, the cut and any sum.
  When the settings change after the map was built it says *Built with other settings — Build Map
  again.*
- **Wide** gives the Series tab more room; the frame's curve and the trace below the map grow with it.

**Options: frames, cut, axes** (folded under the plots):

- **Frames** first **to** last (*the last*: the last frame listed): the frames a map takes; the line under
  the buttons says so when they are not all. To take every frame again, set the first frame to 1 and the
  last to *the last*.
- **Find the cut again in every frame**: each frame is cut anew instead of where the frame on screen is.
- **Frame axis**: **Frame number**, or **Time (s)** when the frames carry a time (their time stamps, or
  frame × exposure). In a saved picture every row has the same height, so the picture of skipped or
  unevenly timed frames does not show real time intervals; the CSV keeps every row's frame and time.
- **Stages compare**: **The whole x axis**, or **The x range shown in the map** (zoom the map and the
  stages follow).
- **Show the picked frame's image**: picking a row also shows that frame in Analyze.

The map is an image like the detector: Log, colour map and Levels work there too. The readout shows
frame, file, x and intensity. The map belongs to the list it was built from: list more files and it
stays, marked as the map of the earlier list; **Clear** under the frame list removes it.

**Names of the exports.** Every file is named after the series, its curve and halves —
`<series>_<curve>[_<halves>]_map.csv`, `…_map.png`, `…_trace_<value>.csv`, `…_peaks.csv`, `…_stages.csv`
— and the frame picked after that frame, as Batch Export names it (`<frame>_<curve>[_<halves>].csv`).
Every table says in `#` lines which curve, halves, frames and cut it holds, with a JSON record of the
settings next to it.

**Results: batch fitting.** When a Batch Export fits every frame (section 5.7), a panel *Results: batch
fitting* appears below the map. Each frame done adds a row with its parameters, state and any reason it
failed. Select a row to see the points actually fitted with the model over them, or open **Parameter
trend** to follow one parameter against the frame number. It works also when the fitted curves are not
exported; q is in Å⁻¹ and lengths in nm. The results stay after Pause, Stop or the end, until **Clear**,
a new batch fit or another import selection.

**Stages.** As soon as the map is built, GIMaP cuts the series into stages and marks odd frames
(coloured stage bar, dashed boundaries, red arrows; the **Stages** row; folded *What changes, and
the odd frames*; **Change along the series** in the drop-down of the plot under the map). **Leave the odd frames out of
Batch Export** skips them in the next Batch Export of this list. **Send to Compare** (or **Analyze ▸
Send Series to Compare**) puts the series beside others. How stages and odd frames are found, and how to
read them, is in chapter 7.

### 5.7 Batch Export

Batch Export reduces every imported frame with the current settings and writes what you choose.
Set up one frame (geometry, mask, cuts), then export all of them the same way.

![The Batch Export dialog](manual_images/en/10_batch_export.png)

*Every option shows the file it writes. The Cuts row names the cut every frame gets; the Summary says how many
frames, which curves and about how big and how long the export is.*

**Where to start it:** **Batch Export…** in the command bar or **Analyze ▸ Batch Export…**
(`Ctrl+Shift+E`), **Batch Export N Frames…** under the file list, **Batch Export…** in the Series tab, the
Export step or **Export ▾**. For a folder of raw frames, **Analyze ▸ Batch Export a Folder…** or the Start
page opens the folder and then the dialog, once its first frame is analysed.

**The dialog:**

- Top: **Frames** (**Every** n-th listed frame), **Settings** (a summary, with **Load Settings…** /
  **Save Settings…**), **Cuts** (the cut every frame gets, with the box **Find the cut again in every
  frame**, section 5.6), **Speed**.
- Middle, four groups:

| Group | Options | Files |
|---|---|---|
| **Data (numbers)** — CSV (comma), TXT (tab) or DAT (space) | **All frames in one table per curve** (x first, one column per frame); **Each frame: its curves, with error bars and pixel counts**; **Each frame: the curve for Fitting**; **Each frame: the q map as numbers**; **Each frame: the unwrapped χ–q map as numbers** | `<name>_<curve>_frames.csv`, `curves/<frame>_<curve>.csv` (+ `_analysis.json`), `fit_input/<frame>_fit_input.dat`, `maps/<frame>_qmap.csv`, `maps/<frame>_cake.csv` |
| **Pictures** — PNG, TIFF, SVG or PDF | **Each frame: picture of the detector image**; **Each frame: picture of the q map**; **Colour scale**: each frame its own limits (1–99.7 %), or the limits on screen for every frame (comparable pictures) | `images/<frame>_detector.png`, `images/<frame>_qmap.png` |
| **Converted detector frames** — TIFF, EDF, NumPy or HDF5 | **Each frame: the detector data in another format** (values as read: modules stitched, frames summed; in the type the file stores, with the gap codes and the pixels without data marked as the Format Converter writes them, section 9.1; pixels left out only by the hot-pixel test or a drawn mask keep their values) | `frames/<frame>.tif` |
| **Curve for Fitting** (GISAXS) | **Halves of the cut**: **Mean of both halves**, **q > 0 half** or **q < 0 half** — what the model fit takes in every frame | (named in the curves for Fitting) |
| **Fitting (optional)** | see below | `<name>_peak_fits.csv` or `<name>_model_fits.csv`, with a `.png` |

- Bottom: **Save to** (**Browse…**; the first time `gimap_analysis/` next to the data, then the last
  folder used; **Next to the Data** goes back to the folder beside the files, and a folder used last for
  other data is marked so) and **In a subfolder named “`<name>`”** (on by default). The line below says
  where the files go; **Export N Frames** starts the batch.

**Before it starts**, the **Summary** says how many frames, which curves and halves, and about
how big and how long the export will be (the cut is said by the Cuts row; what each file is, by `README.txt` in the
folder). While the frame on screen is read again without the sum of the
automatic analysis, the Cuts row says so and Export waits. A folder that cannot be written is said at
once, with Export off. When the folder holds an earlier export of the same data (or curves for Fitting),
Export asks **Replace**, **Keep Both in a New Subfolder** or **Cancel**; the question names what Replace
would remove, and `Enter` answers Replace only when this export writes every such kind of file again —
otherwise Keep Both.

For tables and per-frame curves, tick the curves you want (**All** / **None**). In a table of all
frames, frames with the same points are put side by side; frames whose points differ only slightly
(rounding, a few missing columns) are matched row by row; only grids that really differ are interpolated
onto the first frame's x, never across a gap and never extrapolated. The table header says which. The
folder always gets `README.txt` (what each file is, the curve names, the units, the cut, the sum, the
frames that failed) and `<name>_batch.json` (the choices, the frames, every failed frame with its reason,
every file written, the cut of every frame, and the first frame's full settings).

**Names.** File names keep the frame index counted from 0 (`scan_00007_frame0003_sum3_…`; a file of one
frame has no `_frame`); headers, labels and picture titles count from 1 (*frame 4 of 12*), and every curve
file says in `#` lines which frame, sum and files it comes from. A set of module files is named by its
scan, without `_m01`.

**Fitting (optional)** — **No fitting** by default:

- **Peaks of the ring regions** (GIWAXS): every region with a q window (Ring or Spot) is one peak.
  It is fitted in the region's I(q), over the window plus one window width on each side, with a
  **Peak shape** (Pseudo-Voigt, Gaussian, Lorentzian) on a straight background. One row per frame
  and peak: q, d, FWHM and area with errors, height, η, χ²ᵣ, and a note when a fit is not
  significant, too wide or has too few points.
- **Particle model of the horizontal cut** (GISAXS): the half or the mean chosen in **Halves of the
  cut** fitted with spheres, random or vertical cylinders, with size spread and distance D (**Model**:
  automatic comparison or one; a cut of more than 1000 points is merged first). The two halves are
  never fitted together. One row per frame: `halves`, `points`, model, `chi2` (χ²/N, as in section 5.5) and the parameters in nm.
  About half a minute per frame. The choice starts on the Cuts step's halves when they are the mean or
  one half, else on the last batch's. A frame without counts, or without points on the chosen half
  (*No points on the q > 0 half*), is not fitted and its reason is listed. The curves for Fitting still
  hold both halves with signed qy and name the choice.
- **Start values**: **The previous frame's result** (a slowly changing series), **The first frame's
  result, for every frame** (frames independent), or **Found anew in each frame**.
- **Try on This Frame** fits the frame on screen with these choices before the batch.
  **Also save the fitted curve of each frame** writes `fits/<frame>_<curve>_fit.csv` (peaks) or
  `fits/<frame>_model_fit.csv` (the particle model: the points fitted, in nm⁻¹, with the fit).
- A Gaussian or Lorentzian peak keeps η at exactly 0 or 1; only the Pseudo-Voigt fits it.
- The fits of every frame are shown as they come in the Series tab, under *Results: batch fitting*
  (section 5.6).

**In-situ corrections (reference peak)** (folded, GIWAXS): a peak that does not change during the
run (substrate, internal standard) corrects drifts. Give **Reference peak q** and **Search ±**
(Å⁻¹); the strongest point in that window is the peak. **Align the detector distance so the peak
sits at this q** scales each frame's distance; **Normalise the intensity to the peak height**
scales the intensities so that the reference peak has the height in **Peak height after** (1 by
default), with **One factor for the whole series (from the first frame)** (the default) or **Every
frame to its own peak**. Both are off by default; `<name>_batch.json` keeps these settings
(`series_correction`). The formulas are in
[Physics and Methods, section 6.10](Physics_Manual.md#610-reference-peak-corrections-of-a-series).

**Speed:**

| Speed | What it does |
|---|---|
| **Gentle** | one frame at a time; the computer stays free |
| **Balanced** (recommended) | several frames at once in low-priority processes (about a quarter of the cores, at most 4) |
| **Fast** | more at once (all cores but two, at most 8); other programs get slower |

The number at once never exceeds what half of the free memory allows (a Lambda 9M frame, 14.9
megapixels, needs about 1.4 GB while reduced). A short batch stays one frame at a time, and the
**Speed** box then says so. As an example, Lambda 9M frames with peak fitting took about 2.5 s each
one at a time, and less than 1 s each with *Balanced* (4 at once).

**After it ends** the status line says how many frames were exported, into which folder and in how
long, how many model fits succeeded and why the others did not; failed frames are listed with their
reasons (in the panel, in `README.txt` and in `<name>_batch.json`). A batch with failed frames or failed
fits does not end as a success.

**While it runs** a panel above the right tabs shows the frames being reduced, how many are done,
the time per frame and the time left, and the latest fit. **Pause** starts no new frames;
**Stop** ends the batch after the frames in progress — everything written stays, and the tables
are written for the frames done. The number of frames at once can be lowered while running. The
Series tab shows the map of the frames done so far, growing as they come; **Show the newest frame**
makes the curve below follow the newest frame (click a row to look at another one).

The choices and the folder are remembered for next time. **Save Settings…** / **Load Settings…** (in the
dialog, in **Export ▾** and in **Analyze ▸ Set-up Files**) keep the whole set-up (mode, instrument profile with its geometry, αi, centre, sum, masks,
corrections, cut regions, export choices) as a file: the next raw data need only **Batch Export…**
→ folder → **Export N Frames**. A mask or background file named in it that cannot be found is
listed and ignored.

### 5.8 Export and Send to Fitting

**Export** (command bar, `Ctrl+E`; **Export Curves…** in the Export step and in the **Analyze** menu)
writes, next to the data in `gimap_analysis/`:

- one CSV per curve (`x, I, sigma, pixels`), named `<stem>_<curve>.csv`;
- `<stem>_analysis.json`: the source, geometry, where the beam centre came from, corrections, the
  number of bad pixels left out, the cuts;
- `<stem>_fit_input.dat`, the curve for Fitting.

`<stem>` is the file name; a frame of a multi-frame file adds `_frame` and its number, a sum adds
`_sum` and the number of frames summed.

If that folder cannot be written, GIMaP asks for another one. A note at the bottom right names
what was written, with **Open Folder**.

Figures, map data and settings files:

| Entry | Where | Writes |
|---|---|---|
| **Save Image…** | Export step, **Export ▾**, **Analyze ▸ Figures** | the detector image or q map as shown, with a colour bar (PNG, TIFF, SVG, PDF) |
| **Save q-Map Data…** | Export step, **Export ▾**, **Analyze ▸ Figures** | the intensity on a regular q grid (qy–qz or q∥–qz) as a CSV table with its axes |
| **Save Plots…** | Export step | the upper and lower plots as figures, one column wide |
| **Save Upper Plot…** / **Save Lower Plot…** | **Export ▾** | one of the two plots |
| **Save Settings…** / **Load Settings…** | **Export ▾**, **Analyze ▸ Set-up Files**, the Batch Export dialog | the set-up as a file (section 5.7) |

The image's own **Save ▾** (section 5.3) and **Analyze ▸ Figures ▸ Save View Data…** save the q map or
the cake as a table with its axes.

Saved figures always use the light colours, whatever the theme on screen.

**Send to Fitting** writes `<stem>_fit_input.dat` in `gimap_analysis/` and opens it in Fitting.
The file has columns `q I sigma pixels`, q in Å⁻¹, and `#` header lines with the source, the gap
guard, the frame (*frame 4 of 12*) and the frames summed. GISAXS sends the whole horizontal cut
(signed qy), and Fitting uses the halves chosen (its arrow ▾ chooses them, as in the Cuts step; the
mean or one half is also written in the file); GIWAXS sends I(q), and Fitting opens it with a note
that its models are particle form factors, not peaks. Fitting shows q in nm⁻¹ and says so. When the
folder next to the data cannot be written, GIMaP asks for another one and keeps it for the session.

**Send Series to Fitting…** (in the arrow ▾, the Export step and the **Analyze** menu) opens the Batch Export dialog with
only the curve for Fitting of each frame: choose **Every** n-th frame, the folder and, for GISAXS, the
**Halves of the cut**, then **Export and Open in Fitting** writes `fit_input/<frame>_fit_input.dat` for
every imported frame (or sum of frames) into a folder of its own, `<name>_fitting/` (next to any Batch
Export of the same data, which it leaves alone), and opens its `fit_input/` in **Fitting ▸ In-situ
series** (chapter 6).

## 6. Fitting

The **Fitting** page fits 1D curves with particle models. It has two modes, switched at the top of the page:

- **Single analysis** — one curve, in four steps (section 6.2).
- **In-situ series** — every curve of a folder, or every frame of a detector series, fitted with the model
  of Single analysis (section 6.6).

Switching between the two keeps each side's curve, model and results. With no curve open, the Curve step
offers both ways in side by side: **Open Curve…** and **Fit a Folder of Curves…**. For a series or a batch
of samples the usual way is: fit one representative curve here (find its particle shape, then refine), then
press **Fit Every Curve in This Folder…** in Results.

Fitting works in nm⁻¹ and labels every axis and field with its unit. Analyze works in Å⁻¹; the curve files it
writes for Fitting are in Å⁻¹ and are converted when they are opened.

Coming from version 0.0.2: the detector work of the former Cut & Fitting page now happens in **Analyze**,
which writes the curve that Fitting fits.

| Former Cut & Fitting control | Now |
|---|---|
| Import image, Previous / Next | Analyze file list |
| Detector parameters | Analyze instrument profile (**Geometry…**, **Calibrate…**) |
| Find Yoneda & Set Cut, cut thickness | Analyze's automatic horizontal cut (drag the band or enter rows) |
| Optimize Center X | **Analyze ▸ Beam centre ▸ Refine x by Symmetry** |
| Mask negative pixels + guard | **Mask & corrections** step ▸ **Corrections** (folded) ▸ **Gap guard** (3 px by default) |
| Stack | **Data** step ▸ **Frames** (folded) ▸ **Sum** |
| Cut region width | Fitting's **Fitting range** |
| In-situ CBF / NXS series | Analyze ▸ **Send Series to Fitting…**, or the frames themselves in **In-situ series** (section 6.6) |

### 6.1 Curves Fitting reads

Fitting reads `.dat` and `.txt` text curves:

- Analyze's curve files, `<name>_fit_input.dat`, have four columns — q (Å⁻¹), intensity, σ and the number of
  detector pixels averaged — and a header recording where the curve came from.
- Any other `.dat` / `.txt` file with columns `q I [σ]` opens too. A file whose header states the unit of q is read
  in that unit — `q in <unit>` in a `#` line, or the name of the first column (`q (1/A)`, `q_nm^-1`,
  `# columns: qy (1/A) …`, `# q [1/nm]  I`); Analyze's curve files state Å⁻¹. For a file that states none, set
  **Curve ▸ File ▸ q unit of files without one in their header** (Å⁻¹ by default). A file whose header states both
  Å⁻¹ and nm⁻¹ is not opened: *The header states q both in 1/A and in 1/nm; correct it to one unit.*

Fitting does not read `.csv` files, also not the CSV curves that Analyze exports. Use **Send to Fitting** in
Analyze, or open the `_fit_input.dat` written next to the CSV. A `.csv` dropped on Fitting is refused with
that reason.

Three ways to get a curve in:

1. **Open Curve…** in the command bar (`Ctrl+O` while Fitting is shown), or drop a curve file on the page.
2. **Analyze ▸ Send to Fitting** — writes `<name>_fit_input.dat` and opens it here. The halves chosen in
   Analyze become the **Halves of the cut** in Fitting:

   | In Analyze | In Fitting |
   |---|---|
   | Both halves on \|qy\| (two colours) — default | Both halves on \|q\| |
   | Mean of both halves | Mean of both halves |
   | qy < 0 half | q < 0 half |
   | qy > 0 half | q > 0 half |

   When the folder next to the data cannot be written, Analyze asks for another folder and keeps it for the
   session. A GIWAXS I(q) opens with a note at the top of the page: Fitting's models are particle form factors
   for the horizontal cut of a GISAXS frame, not diffraction peaks, so no particle-shape search is set up for it
   (for peaks use **Analyze ▸ Batch Export ▸ Peaks of the ring regions**).

3. **Analyze ▸ Results ▸ Fit details ▸ Show in Fitting** (after a GISAXS automatic analysis, section 5.5) —
   opens the same curve and puts the selected solution into **Model**. Sizes and spreads are converted and the
   scales are solved on the solution's own curve. The status line says how closely Fitting's model reproduces
   it: the same curve for spheres and random cylinders; Fitting's vertical cylinder weights radii by R⁴, so it
   is loaded as a start to refine. **Refine in Fitting** opens the curve without a solution.

### 6.2 Single analysis: one curve in four steps

![Fitting, Single analysis](manual_images/en/11_fitting_single.png)

*The step rail on the left, here with the Results step open: the quality, the next steps (Fit Again, Fit Every
Curve in This Folder…), what was done automatically at the edges of the ranges and the warnings (when there are any,
none for this fit), and every value ± its error; on the right the curve with the model, the orange fitting range and
the residuals below.*

The page is laid out like Analyze: the steps on the left, each saying where it stands, and the plots on the right.

- **The curve plot**: the points (paler outside the fitting range), the model over the whole curve and the
  **orange band** of the fitting range (drag it). **Terms** also draws each part of the model: every particle,
  the background and the resolution peak.
- **The residual plot** below: (I − model)/σ, or ln(I/model) when the file has no σ.
- **The command bar**: **Open Curve…** (its menu: **Load Model…**), **Undo / Redo** (`Ctrl+Z` /
  `Ctrl+Shift+Z`: every change of the model, the fitting range or the left-out points is one step, and so is every
  fit, solution used and model loaded), **Fit** (`Ctrl+Return`), **Stop** and **Save** (**Data and Fit…**,
  **Plot…**, **Model…**).

**1. Curve**

- The card: its points, its q range in nm⁻¹ (in Å⁻¹ in the card's tooltip), *σ from the file*, and the unit its q
  was read in and where that came from (*q read in Å⁻¹: stated in the file*, or *… not stated in the file: the
  setting for files without one*). The fit weights each point by 1/σ; without σ every point gets the same relative
  weight, and a warning says so in the card and again in the Fit step (*⚠ No σ in the file: every point weighted
  alike*; hover for what that means for each method).
- How to set the fitting range, what Add Particle offers, what each method does and how to fix values are the
  tooltips of those controls; the steps themselves only say where things stand.
- **Halves of the cut**, for a cut through the beam (q < 0 and q > 0): **Mean of both halves** (averaged where
  both halves were measured; across a gap of one half, and beyond its end, the other half alone — never
  interpolated), **Both halves on |q|**, **q > 0 half** or **q < 0 half**. *Predict parameters*, *Find the
  particle shape* and the *AI proposal* take one curve: on **Both halves on |q|** they switch the box to the mean
  and say so; *Refine* and *Search* fit both halves.
- **Fitting range** in nm⁻¹, or drag the orange band; **Whole Curve** fits every point.
- Under *File* (folded): **q unit of files without one in their header** — Å⁻¹ by default; the In-situ series uses
  it too. A file whose header states its unit (Analyze's curves: Å⁻¹) is read in that unit, whatever is set here.
  Changing the setting opens a curve without a unit again in the new unit; for a curve that states one, the status
  line says why nothing changes. A project and the last session keep the unit a curve was read in and this
  setting. A curve saved before opens as it did, unless its header states another unit than the one saved: it is
  then read in the stated unit, and the status line (for a project, its notes) says so.
- **Leaving points out**: switch on **Exclude** above the plot, then click a point to leave it out of the fit
  (click again to take it back), or drag a box around several. They are drawn as red ×, the Curve step says how
  many, and **Include All** takes them back. The left-out points are kept with the curve (and in a project) and
  written in the fit's record.

**2. Model**

- **Add Particle** adds one card per particle: **Sphere**, **Random cylinder** or **Vertical cylinder**.
- Each card has **Distance D** on or off (the paracrystal interference between neighbours) and its values:
  scale, R and **σR/R**, h and **σh/h** for random cylinders, D and **σD/D**. Spreads are relative everywhere
  (as in Analyze's results and in 1D Predict). The scale multiplies the form factor, which is 1 at q = 0 for
  spheres and random cylinders but 10⁻⁶·⟨R⁴⟩/4 (R in nm) for vertical cylinders: compare scales only within
  one family.
- **fit** ticked: the fit may change the value. Unticked: the value stays fixed.
- **Ranges** shows each value's min – max. The fit keeps the value inside it; *Search the ranges* searches
  across it. A value typed outside its range widens the range.
- **Background and resolution peak**: a constant background, and a peak at q = 0 for the tail of the direct and
  reflected beam and the resolution there; an overall factor k, fixed at 1 by default, multiplies the particles and
  the peak. The model as a formula, and how **σD/D** relates to the usual paracrystal spread, are in Physics and
  Methods, [section 9.2](Physics_Manual.md#92-the-intensity-model) and
  [section 9.5](Physics_Manual.md#95-structure-factor-one-dimensional-paracrystal).
- After a fit every value shows its ± error, or *⚠ at a bound* (orange) when it stopped at the edge of its range and
  was not resolved (section 6.3, *Values at the edge of their range*).

A new installation starts with a neutral model, not with the values of an earlier fit. Set the particle and its
sizes for your sample, or let the Fit step find them.

**3. Fit** — one **Method** drop-down and one button:

- When a curve is opened the page chooses the method for it. Another curve with a model not yet edited gets
  **Find the particle shape (no AI)**, which needs nothing beyond GIMaP; otherwise the choice stays as it was (a
  GIWAXS I(q) never gets a particle search). After a prediction or a shape search the method turns to **Refine the
  values in Model**.
- An unfitted GISAXS horizontal cut gets a recommended method, and the line under **Method** says why. Which one
  depends on the fit1d kit, an optional add-on that does not come with GIMaP (section 6.4):
  - without the kit (as GIMaP is distributed), **Find the particle shape (no AI)**, also when you have edited the
    model: *Recommended for this curve: a GISAXS cut not fitted yet (Predict parameters needs the optional fit1d
    add-on, which is not installed).* It fits the cut with Fitting's own models. The tooltip of **Predict
    parameters (fit1d)** then says that it is an optional add-on and where to put it;
  - with the kit installed, **Predict parameters (fit1d)**: *Recommended for this curve: a GISAXS cut not fitted
    yet.*

  GIMaP looks for the kit again whenever a curve is opened.
- Under the method, only what it needs: **Particles** for Predict (section 6.4), **Families** for the shape search
  (*Try every family* or *The families in Model*).
- Press **Fit** in the command bar (`Ctrl+Return`) or the button below, which carries the method's name (for
  example **Refine**). A progress bar and **Stop** appear while it runs.
- **Resolve values at the edge of their range automatically** (on by default, remembered; the In-situ series
  follows it): see section 6.3.
- **Options** (folded): the settings of Predict (fit1d) (network, search, curve type, qz, systematic error;
  section 6.4),
  **Evaluations at most (Refine, Search)** (automatic by default) and **1D Predict Batch Window…** (section 6.5).

**4. Results**, from the top:

- The quality: χ²ᵣ (with σ) or the variance of ln I (without σ), the points, the free values, converged or not.
  After Predict a second line says that Model holds only a starting point.
- What to do next: **Refine These Values** (after a prediction or a shape search), **Fit Again** (the same
  method, with the values, fixed values and ranges now in Model) and **Fit Every Curve in This Folder…** (the
  In-situ series on this curve's folder, each curve starting from this model).
- *Done automatically* (green): values at the edge of their range that the fit widened and fitted again, with
  χ² before and after.
- Warnings (orange), only what you need to know: a value still at an edge, the kit's notes, strongly correlated
  pairs (|ρ| ≥ 0.95: the data do not separate them; fix one — the first two pairs are named, hover for the rest)
  or χ²ᵣ well above 1.
- The values, each ± its 1σ error (the covariance at the solution, scaled by χ²ᵣ). Select values and press
  **Fix**, **Let It Fit** or **Range…** (or right-click): the next **Fit Again** uses them. **Range…** shows the
  value's min – max in Model.
- **Other solutions** (folded): the solutions of *Find the particle shape* or of the AI proposal, each with its
  χ²ᵣ on these points; a solution from **Show in Fitting** shows Analyze's χ²/N (section 5.5) until you fit it.
  **Use This Solution** puts one into Model (Undo brings the previous model back). After
  Predict with several compositions, **Other compositions** instead (section 6.4).
- **Constraints and ranges** (folded, after Predict): every constraint passed to fit1d and where it came from.
- **Save Data and Fit…** writes a CSV — q (nm⁻¹ and Å⁻¹), I, σ, the model, the residuals and every term — with a
  JSON record next to it: the model, the fit (errors, bounds, correlations, what was done at the edges) and the
  points used. **Save Plot…** writes PNG or SVG; **Save Model…** writes the model as JSON, to load for another
  curve with **Load Model…**.
- **Run Log** (folded) at the bottom.

What to check:

- A fit belongs to the curve and points it was made on. When you change the halves, the fitting range or the
  left-out points afterwards, the Fit and Results steps say *Changed after the fit: Fit again*.
- Opening another curve stops a fit that is running; a fit that ends after that is not kept, and the status line
  says so.
- A save that fails (for example on a read-only file) leaves an earlier file as it was and says why.

The model, the method and the choices are kept for the next session, and the last curve opens again.

### 6.3 Fit methods

One **Fit** button runs the method chosen in **Method**:

| Method | What it does | When to use it |
|---|---|---|
| **Predict parameters (fit1d)** | A separate kit: network proposals and its own physical fit, for spheres and vertical cylinders, with errors and equivalent solutions (section 6.4). | Only with the fit1d kit, an optional add-on (section 6.4): a start for a GISAXS cut, with its errors and equivalent solutions. Chosen for an unfitted GISAXS cut when the kit is installed. |
| **Refine the values in Model** | Bounded least squares from the values in Model, with Fitting's own model; stays near them. | The model is close: after a prediction or a solution, after Analyze, or to get errors. |
| **Search the ranges, then refine** | Differential evolution across the ranges of the free values (log scale when a range spans 10× or more), then refinement from the three best starts. | The start is poor but you know the family and sensible ranges. Slower. |
| **Find the particle shape (no AI)** | The numerical fit of sphere, random cylinder and vertical cylinder from several starts, as in Analyze's automatic analysis. The solutions go to Results, the best into Model. | The first look at a new curve, a GISAXS cut included: chosen for an unfitted GISAXS cut when the fit1d kit is not installed. Needs no add-on and uses no network. |
| **AI proposal (1D Predict)** | The experimental V5 network proposes compositions and parameters, then corrects them numerically. Its solutions go to Results. | A second opinion on the family. Check and refine its solutions; do not report them as results. |

- The scales (every particle's scale, the background, the peak A) are solved exactly (non-negative least
  squares) at every step. The search therefore moves only sizes, spreads, distances and the peak's shape.
- **Stop**: *Refine* and *Search* keep the best values so far. Stopping *Find the particle shape*, the AI
  proposal or Predict produces no new result.
- The AI proposal also writes a run folder in `AI_Fitting_Output/` (section 6.5).

**Values at the edge of their range.** With **Resolve values at the edge of their range automatically** on,
a value that stops at the edge of its range after Predict, Refine or Search is dealt with by the fit itself, and
Results lists what was done under *Done automatically*: the range is widened on that side and the curve fitted
again (kept only when χ² is at most 2 % worse; values you fixed are never touched); a value still at the edge, or
one that cannot be widened (a spread at its lower limit, D = 2R), gives only a bound and is reported as one-sided
(for example *≥ 0.549*). When even that is clearly worse at a limit of the model itself (σR/R at 0.6), the curve
wants a value beyond the model: the model may lack something. The steps and tolerances are in
[Physics and Methods, section 9.8](Physics_Manual.md#98-refine-search-goodness-of-fit-and-the-errors). Turn the
box off to leave such values at their edge, with a warning.

### 6.4 fit1d

fit1d is an optional add-on: a separate kit with its own networks and physics. It is not distributed with
GIMaP and is not publicly available, so this section is for those who have a copy. Everything else in Fitting
works without it; for a first fit of a GISAXS cut use **Find the particle shape (no AI)** (section 6.3).

**Where the kit goes.** Put the folder `gisaxs1d_gui_kit` (it holds `gisaxs1d_kit/`, `app/`, `models/` and
`tables_packed/`) in `utils/gisaxs1d_gui_kit` in the program folder, or set the environment variable
`GISAXS1D_KIT_HOME` to its folder, then start GIMaP again. It needs numpy and scipy only, no TensorFlow and no
V5 files. Without the kit, an unfitted GISAXS cut gets **Find the particle shape (no AI)** instead of Predict
(section 6.2), the tooltip of **Predict parameters (fit1d)** says that it is an optional add-on and where it goes,
and choosing it anyway stops with a message that names the folder where GIMaP looked.

In the Fit step the kit is the method **Predict parameters (fit1d)**. It fits all the populations of one curve
together. A good fit does not prove that the composition is the true one.

**What to enter**

1. In **Fit**, select **Predict parameters (fit1d)** (preselected for an unfitted GISAXS cut when the kit is
   installed). Under it,
   **Particles** chooses what it fits — **The particles in Model**, **Try 1–3 vertical cylinders** or **Try
   spheres and vertical cylinders (1–3)**:
   - **Try spheres and vertical cylinders (1–3)** or **Try 1–3 vertical cylinders**: the kit fits every
     composition of these shapes and ranks them (by BIC); the best is shown. For a GISAXS cut whose Model is not
     yet edited, **Particles** starts on **Try spheres and vertical cylinders (1–3)**. The Model's values and
     ranges are not passed, since the populations differ between compositions.
   - **The particles in Model**: the 1–4 **Sphere** / **Vertical cylinder** particles you put in Model (repeated
     families are allowed), with Model's constraints (below). Random cylinders are not offered by fit1d.
2. Its settings are folded under **Options** in the Fit step:
   - **fit1d network**: **General network** (`cond_v2_ft4`) or **Au nanoparticles on Si (adapted)**
     (`cond_v2_ftr1`).
   - **fit1d search**: **Auto (more starts when needed)** (default here), **Fast (one search)** or
     **Robust (always many starts)**.
   - **fit1d curve type**: **GISAXS (Yoneda cut)** or **SAXS (dilute, no structure factor)**. SAXS ignores qz and
     has no structure factor or low-q term.
   - **Cut qz (nm⁻¹)**: for a Yoneda cut, tick it and enter the vacuum qz of the cut. It is not filled from
     Analyze. Unticked means qz = 0, with a warning when a sphere is present.
   - **fit1d systematic error (%)**: a relative error added to σ in quadrature (0 by default; 2–5 % suits
     bright measured curves). Without σ in the file, fit1d assumes Poisson errors and says so.
3. Press **Fit**. It runs in a separate process; **Stop** ends it without a result. Values at the edge of their
   range are resolved as described in section 6.3.

**Constraints from Model** (with **The particles in Model**): values you fixed and ranges you changed (different
from the default ones) are passed to the kit — R and D in nm, σR/R as it is, σD/D converted to the kit's w/D, and
a fixed background. Default ranges are left to the kit, which sets them from the q range of the curve. A range
beyond what this q range can measure is said before the fit. **Results ▸ Constraints and ranges** and the JSON
record list every constraint used and where it came from (Model, widened here, a bound).

**Other compositions.** After **Try …**, Results has a folded **Other compositions** list (up to six: χ²/point,
ΔBIC, which ones cannot be told apart, the kit's flags). Compositions within ΔBIC ≤ 10 of the best cannot be told
apart on this curve: choose from what you know of the sample. Select one to see its values ± errors,
equivalent ranges and states; **Use This Composition** predicts again with it, its values as the start (Undo
brings the previous model back).

**Reading the result**

- The plotted total, the **Terms**, the residuals and **Results** show fit1d's own result: χ²/point, its native
  parameter values, errors, states and equivalent-solution ranges. Parameter warnings are in the *state* column;
  hover for ranges.
- **Save ▸ Data and Fit…** keeps the native total, the components, the effective σ and the warnings in the CSV
  and its JSON record.
- The values copied into Model are only a manual starting point. **Refine These Values** afterwards uses the
  manual model's physics.
- Editing Model, the selected points or the fit1d settings marks the fit stale: its table and χ² are labelled as
  the previous result, the plot shows the manual starting model, and exporting the stale fit is refused. Fit
  again, or restore its conditions.

fit1d has its own physics (number-weighted truncated Gaussian sizes, a Hosemann paracrystal with w = w_rel·D,
spheres on a Yoneda cut evaluated at |q| = √(q∥² + qz²) and vertical cylinders at q∥), so its widths and forward curve are not those of
Fitting's model, which [Physics and Methods, chapter 9](Physics_Manual.md#9-model-fitting-of-1d-curves-fitting)
describes.

### 6.5 1D Predict

![The 1D Predict window](manual_images/en/13_1d_predict.png)

*The candidates of every curve on the left (with their Stage); the selected candidate's fit and parameters on the right.
The second line under the buttons, “q in Å⁻¹, as their header states”, says how the three Analyze curve files
(`_fit_input.dat`) are read, with no unit set by hand; **Curve / side** repeats it after each name (“· Å⁻¹”).*

**1D Predict** is the experimental neural model (Workflow V5). It proposes compositions and parameters for a curve
and corrects each proposal numerically against the measured points. Take its solutions as starts to check and
refine, not as results.

- **For the curve open in Fitting**: choose **AI proposal (1D Predict)** in the Fit step and press **Fit**; the
  solutions are listed in Results.
- **For a list of curve files**: **Tools ▸ 1D Predict — Fit Many Curves…** (or **1D Predict Batch Window…**
  under *Options* in the Fit step) opens the 1D Predict window. To fit every curve of a folder with your model,
  use the In-situ series instead (section 6.6).

**In the window**

1. **Add curves…** lists text curve files (`q I [σ]`; `.dat`, `.txt`, also comma-separated `.csv`). **Use current
   cut** takes the curve open in Fitting instead. The line under the buttons says in which unit of q each file is
   read: the unit its header states or, for a file whose header states none, the setting **q unit of files without
   one in their header**. A file whose header states both Å⁻¹ and nm⁻¹ is not fitted and is named there.
2. **Parameters…** sets:
   - **Fit method**: *General V5 (experimental)* (default), *Single RC specialist (experimental)*,
     *Physical fit (numerical)*, which uses no network, or *fit1d (network + physical fit)*, only with the optional
     fit1d kit (section 6.4); fit1d needs the **Complete composition** (spheres and vertical cylinders only).
   - **Complete composition**: the particle families when you know them.
   - **q unit of files without one in their header**: nm⁻¹ by default. A file whose header states its unit of q is
     read in that unit, whatever is set here; Analyze's `_fit_input.dat` files state Å⁻¹ and need no setting. A
     header states the unit with `q in <unit>` in a `#` line or in the name of the first column (`q (1/A)`,
     `q_nm^-1`, `# columns: qy (1/A) …`, `# q [1/nm]  I`). **Use current cut** is converted automatically.
     *Note:* this default differs from Fitting's (Å⁻¹, section 6.1), so a curve without a unit in its header is read
     10× apart in q in the two places. Check the line under the buttons, or give the file a header with its unit.
   - **q sides**: *Each half separately*, *q > 0 half* or *q < 0 half*.
   - For fit1d: its search (**Fast** by default in this window), curve type, network, qz and systematic error.

   **Save Settings** keeps them.
3. **Fit curve** (**Fit N files** when several are listed) starts, **Cancel** stops (finished files stay saved),
   **Open results** opens the run folder.
4. The table lists the candidates of every curve and half: **Curve / side** (the file and + or −), **#**,
   **Components** (S sphere, RC random cylinder, VC vertical cylinder), **logRMSE**, **RMS/σ** and **Stage**; for a
   file, **Curve / side** also gives the unit its q was read in (hover to see where it comes from).
   Select one to see, in the tabs on the right, its **Fit plot**, its **Parameters and units** and the **Batch log**.

The **Stage** says what produced a candidate. For *Single RC specialist* it is *Neural model*, *Model + amplitude*
or *Numerical fallback*; fit1d rows say *fit1d*; *General V5* and *Physical fit* show the name of the step that
produced it (for example `discovery_neural`). Hover for the details and, for a fallback, the reason. The scores are
not probabilities.

*Single RC specialist* uses its network only for one **Random cylinder** in **Complete composition** and a curve
with native detector counts (Analyze's horizontal cut). Otherwise it falls back to its numerical fit and says why.

Selecting a fit1d row keeps its original curve and components visible; a manual model is copied only as a starting
point.

**Where the results go.** Each run writes a new folder `AI_Fitting_Output/v5_<date>_<time>_<µs>/` in the folder GIMaP
was started from (for the portable version: the program folder). It holds `request.json`, `batch_summary.json`,
`top20_candidates.json`, `fitting_run_summary.json` and, for every curve and half, `solutions.json` (parameters,
units, the stage that ran and why), `display.npz` and `fitting_curves.csv` (*General V5* also writes
`candidates.npz`). `request.json` and `fitting_run_summary.json` list the unit each curve's q was read in
(`q_units`); each file's entry in `batch_summary.json` and its candidates in `top20_candidates.json` carry it as
`q_unit_read` and `q_unit_from`. The folders are not removed automatically. Fits you save yourself (**Save**) go
where you choose.

**Model files.** The model and its trained weights are in `modules/Fitting_1D_Model/Workflow_v5/`, both in the
portable version and in the Git repository, so 1D Predict runs from a source checkout too. Each network checks its
files against their checksums before it is used. With a changed or missing file *General V5* does not start, and
the message names the file. *Single RC specialist* goes on with its numerical fit and gives *Fast model
unavailable* as the reason (section 14.3). fit1d does not use these files: it uses the optional kit (section 6.4).

### 6.6 In-situ series

![Fitting, In-situ series](manual_images/en/12_fitting_insitu.png)

*The Results step on the left (the halves and points per frame, then every frame's χ²ᵣ and values ± 1σ); on the
right the selected frame's fit and, below it, the trend of R through the series, coloured by stage.*

Every curve of a folder, fitted with the model of Single analysis. The command bar has **Choose Folder…** and
**Start**; **Pause** and **Stop** appear while the series runs, **Save** once frames are fitted.

**1. Curves**

- **Choose Folder…** (or drop a folder on the page), **Fit Every Curve in This Folder…** in Single analysis'
  Results, or in Analyze list the frames and choose **Send Series to Fitting…**: every frame is exported
  (optionally summed N at a time) into `<name>_fitting/`, and the series opens here with its curves listed.
- **Files** is the pattern: `*_fit_input.dat` by default; any `.dat` / `.txt` curves work. **Also in
  subfolders** looks deeper.
- **Detector frames.** **Files** can also name detector files (`*.cbf`, `*.tif`, `*.edf`, `*.nxs` …), and a folder
  that holds frames but no curves switches to its frames by itself. Every frame (every frame of a NeXus/HDF5 file;
  a set of module files is one source; Analyze's **Sum** groups neighbouring frames) is cut as Analyze cuts the
  frame it has set up — instrument profile, αi, beam centre, masks and corrections, bad pixels, the cut of the
  series and the halves — giving the same curve as `<stem>_fit_input.dat` (q in Å⁻¹), in memory only: nothing is
  written next to the data. The step says whose set-up is used; while Analyze has no frame set up it says why and
  Start does not begin. **Take Analyze's Set-up** takes Analyze's set-up as it is now and cuts the frames again.
  The JSON record keeps Analyze's settings.
- The curves are listed in counting order of their names (`frame_2` before `frame_10`) and numbered 1, 2, 3 … in
  that order. **Frames**: first – last of these numbers, and **Every** n-th.
- **The unit of q**: every curve file is read in the unit its header states, else in Single analysis' **q unit of
  files without one in their header** (its value at Start, kept for the run). The step counts the listed curves by
  unit and where it came from — *2 curves: q in Å⁻¹, as their header states*; *1 curves: q in nm⁻¹, no unit in
  their header (Single analysis ▸ Curve ▸ File)* — and names a file whose header states both units: it is not
  fitted, and its frame says why. Detector frames are cut in Å⁻¹.
- **Watch for new curves**: while the series runs, the folder is looked at every 2 s and curves written into it
  (for example by Analyze during a measurement) are fitted too. A new file is taken once its size and time are the
  same at two looks; a file changed after it was read is fitted again; a folder that cannot be read is waited for.
- **Halves of the cut**: every frame is fitted on one curve — **Mean of both halves**, **q > 0 half** or **q < 0
  half**, never on both halves together. Until you choose, it takes the halves Analyze sent the curves with, else
  those of Single analysis unless they are both on |q|, else the mean; the line under it says where the value comes
  from. Curves with q on one side only have no halves: the box is switched off, and its tooltip says why. A frame
  without points on the chosen half fails with *No points on the q > 0 half* (or *q < 0*); it is never fitted on the
  other half instead.
- Once the curves are listed, they are compared in the background (section 7.1). The step says how many stages
  and odd frames there are, and the list is coloured by stage. **Leave out the odd frames** is on by default.

**2. Start**

- Every frame uses the model, the fitting range and the left-out points of Single analysis (**Edit in Single
  Analysis** changes them) and the halves of the Curves step. Values at the edge of their range are resolved as
  **Resolve values at the edge of their range automatically** in Single analysis says (section 6.3).
- **Each frame starts from**:
  - **The previous frame's result** (default) — for a slowly changing sample. A frame that did not converge
    makes the next one start from Single's model again.
  - **The model in Single analysis** — every frame independent of the others.
  - **The previous result; the Single model at each new stage** — follows the sample within a stage and starts
    afresh where the curves change course.
- **Method**: **Refine (fast)**, **Search the ranges, then refine (slow)** or **Predict each frame (fit1d)**.
- **Predict each frame (fit1d)** runs Predict on every frame, with the particles, fixed values and ranges of
  Single's model and the settings of Single's Fit step (only with the optional fit1d kit, section 6.4). The table
  holds fit1d's values and
  errors, and the frame plot fit1d's own curve. **Prefer the solution that continues the previous frame** (on by
  default): each frame is fitted once more with every value held near the previous frame's; when that fit is as
  good as the free one within the kit's tolerance, it is taken (the smoothest of equally good solutions), otherwise
  the free fit is taken and the record says the sample changed there. It does not apply when every frame starts from
  Single's model, nor at the first frame of a new stage.
- Then **Start** in the command bar. **Pause** and **Stop** act after the frame being fitted; the frames done are
  kept. A project cannot be opened while a series runs (Stop first). A run started before the stages were found
  uses none (it says so).

**3. Results**

- How many frames were fitted, failed or did not converge, with the halves and points per frame.
- A table with a row per frame: χ²ᵣ, each free value ± 1σ, the **Points** fitted and the **Halves**. Click a row
  to see that frame's curve and model; a failed frame shows its reason.
- Below the frame plot: one value against the frame (or χ²ᵣ), with ±1σ error bars, in the colours of the stages.
  The plot is titled with the name of the series.
- **Open Frame in Single Analysis** takes a frame and its fitted model back to Single analysis to look closer.
- **Save ▸ Table of Every Frame…** writes a CSV (frame, file, χ²ᵣ, log RMSE, converged, points, halves used, every
  value and its error) with a JSON record next to it: the start model, the halves and where they came from, range,
  left-out points, method, every frame's state, reason and the unit its q was read in (`q_unit`, `q_unit_from`;
  `q_unit_of_files` is the setting for files without one), the poor fits (χ²ᵣ above 10) and which curves or frames
  were listed. **Trend Plot…** and **Selected Frame's Plot…** save the figures.
- Before results that are not saved are dropped — a second Start, other curves, a project opened — GIMaP asks:
  **Save**, **Discard** or **Cancel**.
- A series of GIWAXS I(q) curves gets the note of section 6.1 again after the run, with how many fits are poor.
- **Run Log** (folded) at the bottom.

A curve that cannot be read or fitted is marked ✗ in the list, and the series goes on. The folder and the choices
are kept for the next session and in a project.

## 7. Compare and stages

Stages and Compare answer what a series of scattering frames needs, without labels and without AI:

- which frames do not belong (odd frames);
- where the series changes course (stages), and how fast;
- which runs or samples are alike.

Everything uses Analyze's curves, with the same geometry, masks and cuts. They replace the Classification page of
version 0.0.2. Stages here are segments of a series found from its curves alone; the growth stages of a sputter
series (islands, coalescence, film) are a different thing, described in chapter 8.

### 7.1 Stages and odd frames

**How they are found**

1. The curves are compared as log I, on the q points (almost) every frame has. By default each frame's mean level
   is removed, so only the shape counts.
2. Odd frames first: a frame that matches neither the 3 frames before nor the 3 after it, by far more than the
   change from frame to frame — over the whole curve, or only in a few points (about 1 %) where it differs from
   both sides in the same direction, such as a few hot detector columns in one frame. A real change of the sample
   is not odd: after a sudden change a frame matches the frames that follow, and in a steady change it lies
   between the frames before and after. Two or three such frames in a row (two at the start or end of the series)
   are each found. Left in, one such frame would become a component of its own and a false stage, and would enter
   the end state in Compare.
3. The main components of the other frames.
4. Stages: the components are cut into straight pieces in frame order (at least 5 frames per stage, at most 8
   stages). A stage is added only while it explains at least 5 % of the remaining change and more than noise
   would. A series of more than 1,500 frames (odd frames left out) is first averaged in blocks of consecutive
   frames, so a stage boundary is then known only to within one block.
5. Progress: the frame by which half and 90 % of the change along the main component had happened.

The criteria and the stopping rule are in [Physics and Methods, chapter 11](Physics_Manual.md#11-series).

**How to read them**

- A stage describes the series; it is not a phase by itself. A smooth change is also cut where its pace bends.
  What grows or falls between stages (relative to the rest of the curve) says whether the structure changed.
- Each odd frame comes with its reason. A difference near one q in a few points is most likely the detector:
  mask it in Analyze's **Mask & corrections** step and build the map again. A real event of the sample that lasts
  only one to three frames (a crystallite that flashes in one frame) is also marked odd; its reason says at which
  q and by how much. A change that persists is not odd.
- At the start or end of a series there are frames on one side only, so a change in the first or last one or two
  frames cannot be told from an artefact: it is marked odd with its reason. A watched series whose newest frame
  shows something new is judged again as more frames arrive.

**Where they appear**

- **Analyze ▸ Series** (section 5.6): a colour strip beside the map, dashed lines where a stage begins, red arrows
  at odd frames (**Marks ▸ Stages** hides them). **Stages** is *Auto (n)* or a number you choose. The folded
  *What changes, and the odd frames* lists the typical frame of each stage, what grows or falls most between
  stages, the half / 90 % frames and the reason of each odd frame. **Leave the odd frames out of Batch Export**
  does what it says. The plot under the map can show *Change along the series*. **Export ▸ Stages as Table…**
  writes every frame's stage, odd or not and why, with a JSON record of the method. The reasons are given on the
  map's own axis (χ, qy or q). **Options ▸ Stages compare** chooses the whole x axis or the range the map shows.
- **Fitting ▸ In-situ series** (section 6.6): the listed curves are compared as they will be fitted (with the
  halves, range and left-out points of Single analysis). The list is coloured by stage, **Leave out the odd
  frames** is on by default, and **Each frame starts from** offers **The previous result; the Single model at
  each new stage**.

### 7.2 The Compare page

![The Compare page](manual_images/en/14_compare.png)

*Three steps on the left, here with the Results step open; on the right how far each series has changed, their paths
and their end states. Two synthetic runs of 40 frames, 3 stages each: A has one odd frame (frame 34, × on the change
plot), left out of its end state, so A's end-state curve has no spike from its hot columns near qy 0.05 Å⁻¹. A and B:
end states 8 % apart (5 % at the start).*

Typical use: open one sample in Analyze, build its map in the Series tab, press **Send to Compare**; repeat for
every sample. Two or more series are compared with each other; one alone is described.

**1. Series**

- **Add Series** adds **The Series Map of Analyze**, a **Folder of Curves…** (one series) or **Curve Files…**
  (together one series). Curve files need q and I columns (`.dat`, `.txt`, `.csv`, `.xy`, `.chi`; at least two
  per series, in counting order of their names: `frame_2` before `frame_10`). q in nm⁻¹ is converted to Å⁻¹
  when the header says so.
- A folder written by Analyze (a Batch Export folder, its `curves/`, or `gimap_analysis/`) holds several kinds of
  curves. One kind is one series: when there are several, a question asks which one (the curves for Fitting are
  named *the curve for Fitting*, files that are not Analyze's *other files*), and the series is named
  *`<sample or batch>` · `<curve>`*. A Batch Export table of all frames (`<name>_<curve>_frames.csv`) can be added as
  a series too.
- A folder or curve files can also be dropped on the page. The same map or folder is not added twice.
- Double-click a name to rename the series. **Remove Selected** (or `Delete`) and **Remove All** (the note that
  follows offers Undo).
- Long names are shortened on the plots and in the results to the part that differs between series; tables,
  CSV and JSON keep the full names.

**2. Compare**

- The **q range** compared (Å⁻¹; named after the axis of the series, for example **qy range**): the range every
  series covers. Narrow it to leave out a noisy edge or a detector artefact; **Whole Range** goes back.
- **Compare the shape only** (on by default): each frame's mean level is removed, so a brighter beam, a thicker
  film or a longer exposure is not a new structure. Off: the overall intensity counts too.
- **End state**: the mean of the last N frames of a series (10 by default; odd frames left out).
- Every change compares again after a moment, in the background. When a comparison fails (for example no common
  points in the range), Results says why and what to change.
- The series must share their x axis (qy, q or χ). Series on different axes — say a GISAXS qy map and a GIWAXS
  q map — are not compared, and Results names the axes (*The series have different x axes: …*).

**3. Results**

- Groups (three or more series): the series are joined step by step by how alike their end states are, and the
  groups are cut at the largest jump in that joining distance. There are no groups when that jump is less than
  twice the step before it, or when no two series differ by more than 2 %
  ([Physics and Methods, section 11.10](Physics_Manual.md#1110-compare)).
- The series that differs most at the end and at the start.
- Per series: frames, odd frames, stages, the frame by which half and 90 % of its change had happened.
- A table of how different the series are, **At the end** or **At the start**, in percent of intensity.
- **Odd frames and stages of every series** (folded): each series' stages and its odd frames with their reasons;
  where each stage changed most from the one before (the q that grew and fell most, in %) is its tooltip and is in
  the JSON record.

**The plots on the right**

- **How far each series has changed** — along the main change chosen above the plot (**Component 1** explains the
  most), against **Frame** or **Share of the series (0–1)** for runs recorded at different rates; × marks an odd
  frame.
- **Paths through the two main changes** — ○ the first kept frame, ■ the last. With only one main change there is
  a note instead of empty axes.
- **End states** — the end-state curve of every series.

**Saving**

- **Save** appears once there is a comparison (it waits while the series are being compared again).
  **Save ▸ Table of Every Series…** or **Table of Every Frame…** writes a CSV with a JSON record: the method and
  settings, and per series its source, odd frames with their reasons, stages, half / 90 % of the change, the
  differences at the end and at the start and `stage_details` — for every boundary between stages the x where the
  curve grew and fell most (`grows_most`, `falls_most`: `x` on the record's `x_label`, `percent` relative to the rest
  of the curve) and the same as a sentence.
- **Change Plot…**, **Paths Plot…**, **End States Plot…** save the figures, always in the light colours.
- The tables on the page can be copied (`Ctrl+C`, or right-click), with their column names.
- A project keeps the settings and the series: curve files by path, Analyze maps in `<project>.compare.npz`.

### 7.3 Before you conclude

- **A detector artefact that comes and goes makes odd frames.** A narrow spike at one q that appears in only some
  frames marks those frames odd, with the reason *probably the detector, not the sample*. Mask it in Analyze and rebuild the
  map; do not read it as a change of the sample.
- **Keep the compared range clear of artefacts and shadowed regions.** Over the full q range a series can look
  like an outlier because of a spike or a shadow. After narrowing the **q range** to the clean part, the
  difference at the end may be much smaller.
- **Look at both tables.** Check the differences **At the start** as well as **At the end**: series may differ
  where they began rather than where they ended.

## 8. Sputter

The **Sputter** workspace (sidebar, `Ctrl+5`) follows an in-situ sputter deposition: the PILATUS CBF frames of
one scan, taken while Ag or Au is sputtered onto a polymer (Cu is read with the Ag model, for reference only).
For every time window it gives the particle size R (the base radius of a flat-based hemisphere, h = R), the
spacing D, the island height H, the growth stage, the metal thickness where it can be stated, and how far each
value can be trusted. How each value is computed is in [Physics and Methods, chapter 12](Physics_Manual.md#12-sputter).

### 8.1 Predict in three steps

1. **Data ▸ Choose…**: the folder of one scan (its CBF frames, or a scan folder with them in one subfolder such
   as `p2m/`). One line then sums up the frames, the detector, the exposure and the time (its tooltip adds the
   serial number, the pixel size, the first frame and the flat field).
2. **Sample and geometry**: **Metal** (Ag, Au, or Cu, approximated with the Ag model), **X-rays** (**Energy
   (keV)** or **Wavelength (nm)**), **Distance (mm)**, **Incidence αi (°)** and **Beam centre (px)** (col, row).
   Check every value. Beside each, the page says where it comes from — this folder's last Predict, a preset, for
   the energy a `Wavelength` line of the header or the scan's NeXus file, or the last Predict of any folder
   (*please check*) — or asks for it: a PILATUS header holds neither the distance nor the incidence angle. The
   `E<eV>` in a flat-field file name is the detector's calibration energy, not necessarily the beam's: it is shown
   beside the field and never filled in. A value to check and an energy of the files that differs from the one used
   are shown as a warning (⚠); the tooltip of the line says why. An empty **Beam centre (px)** is estimated from
   the frames; a centre you give is in Analyze's pixel coordinates (section 4.4).
3. **Predict**. The first time, PyTorch is installed into GIMaP's Python (*Step 1 of 2*, about 1.2 GB, needs the
   internet). Then the frames are read and summed into time windows, and the model runs (*Step 2 of 2*).
   **Cancel** stops at once; **Details** shows the messages; the tooltip of the line under the job status names the
   Python used.

![The Sputter workspace with a scan folder chosen, before Predict](manual_images/en/22_sputter.png)
*Sputter with a folder chosen, before Predict. Here three PILATUS CBF frames from the repository's `TestSAXSdata/`
folder stand in for a scan (they are not a sputter series, so no result is shown): the line under **Data** sums up the frames, the detector, the exposure and
the time. The energy, distance and incidence angle are not in the files, so the page asks for them (the flat-field
energy is shown beside the energy, not filled in); an empty beam centre is estimated from the frames. The line under
the job status says that PyTorch is installed on the first Predict.*

**Save Geometry as Preset…** fills in later scans of this detector and flat field within one beamtime (a week either
side of the scan, to start with). Section 3.9 says what else is remembered.

### 8.2 More options

| Field | What it is for |
|---|---|
| **Polymer** | The host: **PSBMA**, **PS**, **PEO**, or *polymer* when it is none of these (you can also type a name). It chooses the published values of the Literature tab. |
| **Sample roll (°)** | 0 when neither this folder nor a preset gives one (the page says so). |
| **Mask** | A detector mask (`.npy`, optional), remembered for this folder only. |
| **Time window (s)** | 0.5 s by default; the first 5 s are always cut into 0.1 s. |
| **Deposition rate (nm/s)** | Optional, for example from a quartz balance: gives the nominal thickness and the literature method. |
| **Start of deposition t0 (s)** | Empty: the first frame, an assumption the notes state; δ is then too large if the deposition began later. |
| **Deposition end (s)** | When the shutter was closed. Empty: the end of the scan. |
| **Save results in** | Where the result folders go (default: the folder `sputter` in the user data folder). |
| **Open Results Folder…** | Shows an earlier result. |
| **Add Refined Windows…** | Once a result is shown: merges physical-refinement windows computed elsewhere (one JSON each). |
| **Explanation** | Which explanation is shown (section 8.3); it can be changed once a result offers more than one. |

The rate, t0 and end are remembered for this scan folder only; the polymer and the time window are taken over
from the last Predict.

Each request gets its own folder, `sputter/<scan>-<code>`, named from the scan and every value the result depends
on: the same request reuses it (a stopped run goes on), another geometry or model gets a new one.

### 8.3 Two explanations, and the grades

**More options ▸ Explanation** chooses how the windows are explained:

- **Literature method (mass balance)**, the default when a deposition rate is given and at least 10 windows show
  the metal's correlation peak. As in the in-situ GISAXS studies at P03, D = 2π/q\* of the peak, δ comes from the
  rate, and R from mass balance for hemispheres; this R agrees with the literature by construction, so the
  **Literature** tab holds the independent tests. After the percolation D goes on, R is not given.
- **Physical path (network)**, the default without a rate: one path of stages through time (pre-nucleation,
  islands, coalescence, film) that never turns back from film to islands, with the network's R and D in every
  island window.

**Explanation** also offers **Per window (network / refined)**, **Learned selector** and **Time-consistent path
(exploratory)** when the folder holds them. Each R and D carries a grade:

| Grade | Source |
|---|---|
| **Lit** | literature method |
| **P** | network, on the physical path |
| **A** | physical refinement, converged, unique (from **Add Refined Windows…**) |
| **B** | as A, but an equally good alternative exists |
| **C** | network only, window by window |
| **L** | learned selector (exploratory) |
| **S** | time-consistent path, supported by the data (exploratory) |
| **T** | time-consistent path, driven by its constraints (exploratory) |

Terms on the page:

- **q\***: the position of the correlation peak; 2π/q\* is the spacing it implies.
- **δ**: the nominal thickness, rate × (t − t0).
- **t_eq**: the metal in the islands the network sees, spread as a layer (uncertain by a factor of about 2–3).
- **η = t_eq/δ**: the share of the deposited metal found in those islands.
- **Percolation**: where the islands touch (2R = D), given as δ with its 68% interval, or *not determined*.
- **Film onset**: the first window the physical path gives to the film network, with its interval (*not
  established* when a path without film fits about as well).
- **TS**: the network of the late, film-like windows, named after the Teubner–Strey-type scattering it was trained
  on; its **TS length L** is a lateral texture length, not an island spacing.

Physics and Methods defines [the physical path](Physics_Manual.md#126-the-physical-path),
[the literature method](Physics_Manual.md#1210-the-literature-method-mass-balance-for-hemispheres) and
[the grades](Physics_Manual.md#1213-grades).

### 8.4 The tabs

The tabs on the right are **Size Trend**, **Literature**, **Structure Sketch**, **Windows** and **Reliability**.

**Size Trend** opens in a simple view: R and D of the explanation shown (a faint point per window), their smoothed
trends, and the percolation or film onset with its interval; film windows are shaded grey on every panel. With the
physical path the D panel also marks the direct 2π/q\* (grey ×). Before there is a result, the tab says how to get
one. Tick **Show details** to add:

- the **Metal thickness** panel: t_eq as a band, the nominal thickness and the deposition end (with the literature
  method only the nominal thickness δ, which the islands hold by construction);
- the panel **TS length L (film windows)**;
- the direct 2π/q\* with its check (red circle: a D outside the simulated 90 % range of 2π/q\*);
- the windows where the beam may be off the sample (hollow grey points, section 8.5).

**Logarithmic size axis** (on by default) puts the sizes and thicknesses on a log axis; H always stays linear.
With a rate, the top axis also gives δ.

The panel **H, island height from q_z fringes (nm)** appears in both views once GIMaP has measured H in the
background, seconds to half a minute after a result is shown: minima in the vertical cuts that move to lower q_z
give H per 1 s group with its 68% interval (hollow: seen on one side of the beam only; before the first minimum is
in range, only a bound). H is the vertical size of the islands, the film thickness only once a continuous film has
formed ([Physics and Methods, section 12.12](Physics_Manual.md#1212-island-height-ht-from-moving-q_z-fringes)).

**Literature** compares the series with published growth, over δ. It opens with two panels, the literature
method's D = 2π/q\* with its trend and 2R/D with the line 2R = D, both against the published percolation band of
the metal on its **Polymer**. **Show details** adds the peaks
not used, the network's values, the published D(δ) curves, the coverage and η. Below, each check says *consistent*,
*inconsistent* or *not testable*, with references and DOI; without a rate only whether D decreases is tested
([Physics and Methods, section 12.11](Physics_Manual.md#1211-checks-against-published-growth)). The tab needs
the explanation **Physical path (network)** or **Literature method (mass balance)**; with another one it says
so.

**Structure Sketch** draws the fitted model of one window (side and top view, one ruler for the series). **Time**
chooses the window. **Median of ±5 windows** (on by default) draws the median of the neighbouring windows (about
±2.5 s); untick it for each window's own values. **Show model burial depth (unreliable)** adds the model's burial
depth. **Save Image…** saves the picture, **Save Storyboard…** six windows across the series, drawn with the same
two choices (PNG, PDF or SVG; the save dialog starts in the result folder). Film windows are drawn without
particles.

**Windows** lists every window with its stage, values, grades and typical errors, thicknesses, *Beam may be off the
sample*, whether the literature method used its peak (not a *host peak*, *outlier* or *late peak*), notes and, once
measured, **H fringes (nm)**. `Ctrl+C` copies rows.

**Reliability** says in words what the grades and the typical errors mean, what the path assumes and what the
thickness values are; with the literature method also how many checks agree.

### 8.5 The result line and its flags

The line under **Predict** (shown once there is a result) names the windows, how many have R and D, and the film
onset (with the literature method: the percolation). Two flags may follow:

- *intensity modulation period ≈ … s (sample stage translation?)*: the intensity dips periodically, most likely
  because the sample stage moves back and forth and the long grazing-incidence footprint reaches the sample edge.
  The data do not prove it, so the page asks.
- *beam may be off the sample in … windows (stage translation)*, or *beam may be off the sample or interrupted in
  … windows* when there is no modulation: windows far below the intensity of their neighbours. They keep their values in **Windows** but are left out of the stage path, the
  trends, the checks and H ([Physics and Methods, section 12.9](Physics_Manual.md#129-windows-where-the-beam-may-be-off-the-sample)).

### 8.6 Show in Analyze

Right-click a row of **Windows**, the **Structure Sketch** or a point of **Size Trend** and choose **Show in
Analyze**: Analyze's frame list is replaced by the raw frames of that window, summed (**Sum**), so its image and
cuts show what the model read. αi and a beam centre you gave are taken over; distance and wavelength come from an
instrument profile within 0.5 mm and 0.1% of the Sputter values, or the notice says what to enter. Nothing changes
while Analyze is busy.

### 8.7 Changing the rate, t0 or end after a run

A new **Deposition rate**, **t0** or **Deposition end** rewrites the files of the result shown without reading the
frames again (a rate must be positive, an end after t0). For a result of another scan than the one under **Data**,
the values are kept for the next **Predict** of the folder under **Data**. Without a rate, the literature method gives way to the physical
path.

### 8.8 Before you conclude

- The thickness of a continuous film is not determined (in simulation the model returns its prior), and a film with
  particles on it is not modelled. Use t_eq and the nominal thickness; compare with XRR, a quartz balance or a
  cross-section.
- Absolute R rests on the hemisphere shape (h = R): trust trends more than single values.
- The literature comparison rests on the rate (scale a quartz-balance rate to the sample position), t0 and the end
  you give, on all the metal sticking, and on D = 2π/q\*, which can be off by more than 40% for D > 10 nm once
  coalescence starts. A *not determined* percolation is not determined.
- Published thresholds come from other chambers and rates: a band is a comparison, not a prediction.

Every stated limit: [Physics and Methods, section 12.14](Physics_Manual.md#1214-stated-limits).

### 8.9 Output files

Each result folder holds the model's tables (`structure_physical.csv`, `structure.csv`) with their summaries, the
binned frames, `gimap_request.json`, the island heights (`fringes_v1/`) and the pictures you save; chapter 12 lists
them all.

## 9. Tool windows

This chapter describes three windows of the **Tools** menu: Format Converter, XRR Series Extractor and SEM
Particle Analysis. The other entries of that menu are described with their work: Geometry Calibration
(section 4.3), 1D Predict (section 6.5) and Process with AI (chapter 10). The windows work beside the
workspaces: they open over the page you are on and leave it as it is. Each window remembers its size and
position. What each tool computes, in which units and pixel coordinates, and what it does not correct is in
[Physics and Methods, chapter 13](Physics_Manual.md#13-tool-windows-and-labs).

### 9.1 Format Converter

Open **Tools ▸ Format Converter…** (`Ctrl+Shift+C`) to write detector frames in another format.
**Tools ▸ Convert Current File…** opens it with the file shown in the active workspace. The source files
are only read, never changed.

![The Format Converter at its second step](manual_images/en/15_format_converter.png)
*Step 2: the sources to convert on the left; on the right the preview with the frame's size, data type, value range and counts of invalid, negative and saturated pixels (a multi-frame source shows its first, middle and last frame).*

The window has three steps; **Next** and **Back** move between them.

1. **Input** — **Add files**, **Add folder** or **Use currently opened file**. Files and folders can also
   be dropped on the window. The converter reads NeXus/HDF5 (`.nxs`, `.nx`, `.nx5`, `.h5`, `.hdf5`, `.hdf`), CBF
   and TIFF, with the same reader and orientation as Analyze (section 5.1). Single images and multi-frame files
   are recognised; a set of module files is one input (stitched, named by its scan without `_m01`), a master
   with its data files too; for a NeXus file with several datasets you choose the dataset. The list says what an
   input consists of and what the reader had to say about it (⚠ a module file missing, unequal frame counts). An
   HDF5 file without a detector image is refused, with the reason.
2. **Configure & Preview** — tick the sources to convert (**Select all**, **Select none**, a filter by
   file name (**Filter by filename…**), **Sort by filename**, **Remove selected**). For multi-frame NXS files, **Advanced frame
   selection** offers *All*, *Current frame*, *Frame range*, *Custom* (for example `1, 5, 8–20`) and
   *Every Nth frame*. The preview shows the first, middle and last frame.
3. **Run, Results & Export** — choose:

   | Setting | Choices |
   |---|---|
   | **Output format** | TIFF, CBF, HDF5 or NumPy (`.npy`). A format that every input already has is not offered (HDF5 is, for the container) |
   | **Destination** | by default a folder `converted` next to the first input |
   | File names | a frame of a single-frame source keeps the source's name (`<source>.tif`). For multi-frame sources the **Multi-frame template**, by default `{source}_{frame:06d}` (or `{source}_img_{frame:06d}`), for example `scan_001_000123.tif`. The frame number counts from 1, unlike Batch Export's names, where `_frame0003` counts from 0. **Add suffix automatically when names collide** (on by default) adds `_001`, `_002` … instead of stopping |
   | **Advanced output options** | the data type (the stored values and type are kept by default, see below; CBF holds integers: it keeps neither NaN nor fractional or negative floating-point values); **Write metadata sidecar JSON** (on by default): **One metadata file for the whole conversion** (`conversion_metadata.json`, the default) or **One JSON file beside each output image**; **Export as one NeXus/HDF5 container** (`converted_images.h5`, or `converted_images_001.h5` … when that file exists) |

   In a CBF or TIFF header the metadata is written as ASCII JSON (`json.loads` reads it back exactly, paths
   outside ASCII included). On the third step the button reads **Review & Convert**: it shows what will be
   written and its size, and **Convert** there starts the conversion.

**Pixels without data.** A converted frame reads back in GIMaP as its source does: the same pixels without data
(module gaps, bad pixels, masked or not covered by a module), the same values elsewhere, and integer counts stay
counts (counting errors, the dead-pixel test). The exception is a floating-point frame written as CBF, which
holds integers: fractional parts are cut, and a negative value (−1 or less) reads back as a gap code, a pixel
without data. Use TIFF or HDF5 for such frames. By default (**Keep the stored data type**) a frame is written
in the type its file stores — the preview's **Data type** — not as the reader's 32-bit float:

| Source | Written as |
|---|---|
| Integer counts (Pilatus CBF or TIFF, NeXus) | the stored type; gap codes as stored (Pilatus −1, −2). A pixel the reader had set aside (a NeXus `pixel_mask` pixel, a gap between stitched modules) is −1. An unsigned frame that needs −1 is written in a signed type that holds every value (uint32 → int32); in HDF5 it keeps its type and the largest value of the type marks the pixel |
| Floating-point frames | as stored, NaN on every pixel without data (CBF holds integers: −1 there; values truncated, negative values read back as gap codes) |
| **Save as 32-bit float** | 32-bit float, NaN on every pixel without data (CBF: −1) |
| **Convert to uint16 …** / **Clip to the uint16 range** | uint16 from the pixels with data (scaled onto 0–65534, or cut at 65535); 65535 on the pixels without data |

An HDF5 output (and the container) also marks every pixel without data in the NeXus
`/entry/instrument/detector/pixel_mask` (bit 0), which GIMaP and other NeXus readers honour. Each frame's
metadata says what was done under `pixel_values`: the stored and written types, how many pixels have no data and
how they are written; for a floating-point frame in CBF also how many pixels read back without data
(`pixels_read_back_without_data`), how many negative values become gap codes (`negative_data_read_as_gap_codes`)
and how many values were truncated (`values_truncated`). GIMaP does not open `.npy` files as frames: there, −1 or NaN is the mark. The uint16
modes keep the mark only in HDF5: a TIFF or CBF carries no mask, so a pixel without data reads back from it as a
valid pixel of 65535 (chapter 13). The rules are in
[Physics and Methods, section 13.3](Physics_Manual.md#133-format-converter).

While it runs, a window shows the progress with **Pause** and **Cancel** (the image being written is
finished first). Afterwards use **Open output folder** and **View report**. The report,
`conversion_report.json` in the destination folder, lists every frame written or failed; a later run into
the same folder replaces it.

EDF frames are not an input of the converter. With an EDF frame shown, **Convert Current File…** opens
the window without it and says why under the input list. To convert EDF frames — or any frame as
Analyze reads it, with modules stitched and frames summed — use **Analyze ▸ Batch Export ▸ Converted
detector frames** (TIFF, EDF, NumPy or HDF5; section 5.7).

### 9.2 XRR Series Extractor

Open **Tools ▸ XRR Series Extractor…** (`Ctrl+Shift+R`) to obtain an XRR curve from a GIWAXS/GISAXS
detector angle series. One point of the curve comes from each detector frame.

![The XRR Series Extractor with the first frame of a CBF series loaded](manual_images/en/16_xrr.png)
*Left: the series (**Browse…** for a file, **Folder…** for the folder of the series; below, what the series consists of and which values are not in the file), θ (the geometry is further down) and the ROI. Right: the frame with the direct-beam cross and the ROI circle on the specular beam; the **XRR points** tab holds the curve.*

1. **Choose the series.** **Browse…** selects one NeXus/HDF5 file or a CBF file, **Folder…** the folder of the
   series; a file or folder can also be dropped on the window.
   - The module files of a set are stitched for each detector frame; a file of several frames gives one point
     per frame (when modules hold different numbers of frames, only the frames every module has are used).
   - A NeXus file that holds one frame stands for the series of files numbered like it; files numbered alike
     that do not fit (another frame size, more than one frame) are named and left out. A folder of NeXus files is
     read file after file.
   - For CBF, a glob pattern (**CBF pattern**, e.g. `*.cbf`): each naturally sorted file is one point.
2. **Set θ.** Either a linear sequence (**Theta start (°)** and **Theta step (°)**) or an NXS motor dataset with
   one θ value per frame.
3. **Set the geometry.** Detector distance, beam energy, pixel sizes and the direct-beam centre. Select
   in **Reflection direction** whether the specular reflection moves up or down in detector y.
   - The window starts from the last applied Geometry Calibration, marked *from last calibration*. When
     that calibration was of frames of another size, only its energy is taken.
   - **Load first frame** never overwrites a value you typed. A tag beside each field says whether a value
     comes from the file, the last calibration, the image centre or the built-in defaults; the summary under
     **Load first frame** warns only of the values that are not in the file and need checking
     (*⚠ Not in the file — check: …*).
   - **Pick direct-beam center** lets you click the centre on the frame.
4. **Set the ROI.** An ROI radius and `Sum` or `Mean`. Radius 0 reads one pixel; a larger radius
   integrates a circle around the calculated specular beam.
5. **Run extraction.** One full frame is loaded at a time: the window shows it with the ROI, records one
   point, and releases the frame. The whole series is never held in memory.
6. **Look and export.** Switch freely between **Live frame** and **XRR points**; processing updates both
   but never changes the tab. **Export CSV…** saves θ (°), qz (Å⁻¹), intensity, ROI position and
   valid-pixel count.

The CSV goes by default to `<source folder>/<source name>_xrr.csv`; for a folder chosen with **Folder…** (CBF
or NeXus files) it goes into that folder, named after it. A `.json` record beside it keeps every value that determines qz (series, θ,
distance, energy, pixels, centre, direction, ROI) and where each came from, and under `source_layout` what the
series consists of, every file read and the reader's warnings (the status line after a run repeats them).

**What is not corrected.** The extractor reports the raw ROI sum or mean. It does not apply monitor/flux
normalisation, footprint correction, background subtraction or resolution correction, and θ gets no offset.
A θ motor dataset is converted only when its unit is `mrad` or `rad`; any other unit, µrad included, is read as
degrees. The ROI uses the pixels the reader marks as valid: Analyze's bad-pixel search, masks and value limits
are not applied, and negative values of a signed-integer NeXus/HDF5 frame enter the sum. The formulas are in
[Physics and Methods, section 13.1](Physics_Manual.md#131-xrr-series-extractor).

**Pixel coordinates.** The direct-beam centre is shown and entered in the same pixel coordinates as in
Analyze and Geometry Calibration: pixel corners lie on whole numbers, so the centre of the
first pixel is (0.5, 0.5); x counts from the left edge and y from the top edge (section 4.4). Hover over the centre
fields or the *ROI x* / *ROI y* headers to read this in the window. A centre can be copied between the
three windows as it is.

- The preview, its cross and ROI circle, and the *ROI x* / *ROI y* columns use the same pixels.
- A centre picked with the mouse is the point clicked: with radius 0, the pixel you click is the pixel
  that is read.
- The image centre the window falls back to (step 3) is the middle of the frame, (columns / 2, rows / 2).

The exported files count differently: `roi_center_x_px` / `roi_center_y_px` of the CSV and
`geometry.beam_center_x_px` / `beam_center_y_px` of the JSON record count pixel centres from 0, so they
are 0.5 px lower than the window shows, and the record says so (`center_convention`). If you noted a
centre from this window in a copy of GIMaP from before 2026-10-13, add 0.5 px to x and y: the window
counted pixel centres from 0 then.

**Known limitation: a beam centre from a NeXus header.** When a NeXus file written by the Lambda server
(P03) states a beam centre, the window takes the two numbers as column and row of the frame it shows,
without the turn (transpose and flip) the frame gets when it is loaded. The centre can then lie away from
the direct beam (Analyze applies the turn to the same header). The fields are marked *from file*, and the
value replaces the centre of the last calibration. Other NeXus/HDF5 files are read as stored, without a
turn, but the window still counts the header numbers half a pixel differently from Analyze. Before you run
the extraction, check on the preview that the cross lies on the direct beam, or use **Pick direct-beam
center**. CBF series are not affected.

### 9.3 SEM Particle Analysis

Open **Tools ▸ SEM Particle Analysis…** to measure particle sizes in an SEM image, or to load sizes
measured elsewhere from a CSV file. The window works on its own: it needs no prediction model, no
TensorFlow and no scattering data. A radius distribution predicted in **Labs ▸ 2D Prediction** can be
compared with it: **SEM comparison…** there opens this window with a copy of the prediction attached
(section 11.1).

![The SEM Particle Analysis window after an extraction](manual_images/en/17_sem.png)
*Check the outlines first: cyan particles are kept, orange ones were rejected (the table says why), the yellow box is the ROI.*

What is measured is the **equivalent-circle radius of the projected area** of each particle:

```text
A_nm² = A_px × nm_per_pixel²
R_nm  = sqrt(A_px / pi) × nm_per_pixel
D_nm  = 2 × R_nm
```

`A_px` is the number of pixels of the particle. A particle that is not round is also given the radius
of the circle of the same area. The distribution is drawn against R; the table gives R and D. The
segmentation, the filters and the comparison with a prediction are defined in
[Physics and Methods, section 13.2](Physics_Manual.md#132-sem-particle-analysis).

#### Measuring an image

1. **Open the image.** Under **SEM source**, press **Open image…** and choose a TIFF, PNG, JPEG or BMP.
2. **Enter the calibration.** Under **Image calibration**, choose the **Calibration method**:
   - **nm per pixel** — type the value;
   - **Scale-bar length / pixels** — type the length of a scale bar in nm and its length in pixels;
     `nm_per_pixel = length_nm / length_px`.

   The calibration is **never guessed**. GIMaP does not read the scale-bar text, the magnification or
   the DPI of the file. Opening another file clears the value, and nothing can be extracted until a
   positive value is set for that file.
3. **Exclude the annotations.** In the folded section **Segmentation and image ROI**, tick **Use image
   ROI (pixel edges)** and type **ROI left x**, **ROI top y**, **ROI right x**, **ROI bottom y** so the
   scale bar, text and frame are outside. The ROI uses the pixel edges of the original image: row 0 is
   at the top, the centre of pixel (row i, column j) is at (j + 0.5, i + 0.5).
4. **Choose the detection** under **Particle detection** (see below), and its settings.
5. **Run.** The button under the source reads **Extract particles**, **Detect particles (local AI)** or
   **Measure labeled particles**, after the method.
6. **Review.** Look at **SEM image / outlines** and **Particle measurements**: which regions were kept,
   which were rejected and why. Only then use the distribution.

When no particle passes the filters, the candidates, their reasons and the outlines are still shown and
can be exported for review; there is then no distribution.

#### Detection methods

Changing the method makes the previous measurement invalid. GIMaP never switches to another method by
itself when one fails.

| Method | When to use it | What it needs |
|---|---|---|
| **Threshold / watershed** (default) | well-separated particles with clear contrast | nothing more |
| **StarDist (local AI)** | touching particles that a threshold cannot separate | a local Python with StarDist, and a local model folder |
| **Existing instance labels** | labels made elsewhere (StarDist outside GIMaP, corrected by hand …) | an integer TIFF/PNG label image |

**Threshold / watershed.** Settings in **Segmentation and image ROI**:

| Setting | What it does — and what to check |
|---|---|
| **Particle contrast**: Bright / Dark particles | bright: pixels above the threshold; dark: pixels at or below it |
| **Threshold**: Otsu / Manual | Otsu is computed from the finite values inside the ROI; Manual uses the value in **Stored intensity** (the stored intensity units of the image). Check that background and particles are separated. |
| **Split touching particles (watershed)**, **Watershed peak distance (px)** | separates touching particles; the distance decides into how many pieces a cluster is cut — look for over- and under-splitting |
| **Minimum area (px²)**, **Maximum area (px²)** | removes noise or large background regions — and possibly real particles; the maximum defaults to *No upper limit* |
| **Minimum circularity** | `4 pi A / perimeter²`; the value is kept in the table |
| **Exclude particles touching the ROI boundary** | on by default; particles cut by the ROI are rejected, with the reason |

Grey images are segmented on their stored intensities. Colour images are turned into grey with the
recorded luminance formula (0.2126 R + 0.7152 G + 0.0722 B). The contrast stretch on screen only helps
you see; it does not change the threshold, the areas or any measured value.

**StarDist (local AI)** — optional. It separates touching particles with a trained instance-segmentation
model, run on your computer.

- Set **Local Python executable** (a Python environment with StarDist and TensorFlow) and **Local
  instance model** (a folder with `config.json` and `weights_best.h5` or `weights_last.h5`); **Browse…**
  chooses them, **Check local resources** checks that the files are there.
- By default GIMaP looks in `~/.cache/gimap/sem/stardist/` (or the folder in `GIMAP_SEM_STARDIST_HOME`):
  the Python in `runtime/Scripts/python.exe` (Windows) or `runtime/bin/python`, the model in
  `models/2D_versatile_fluo/`.
- To set up that environment, use its own Python:
  `python -m pip install -r requirements-sem-stardist.txt` (StarDist 0.9.2, CSBDeep 0.8.2, TensorFlow
  2.15.1). The list installs the packages only, not the model weights; GIMaP's own environment does not
  need them.
- The model runs in a separate process on the CPU. The image is not uploaded, and no weights are
  downloaded. **Cancel** ends the process. A run is stopped after 15 minutes.
- **Instance AI options** (folded): **AI image scale**, **Probability threshold**, **NMS overlap
  threshold** (*Model default* uses the thresholds saved with the model), **Normalize low / high
  percentile** (computed inside the ROI; dark particles are inverted first).
- The outlines are put back on the original pixel grid before anything is measured: the AI image scale
  never changes the areas. The calibration, ROI and the area, circularity and boundary filters are the
  same for all three methods.

The official `2D_versatile_fluo` weights were trained on fluorescence images of cell nuclei: **they are
not validated for SEM**. Judge them on the outlines of your own images; more particles found does not
mean more accurate. A model trained on SEM images can be loaded from a local folder.

**Existing instance labels.** Open the SEM image first, then choose the label file. 0 is background and
every particle has its own positive integer. RGB colour maps, floating-point labels and multi-page files
are refused. The label image has the size of the SEM image, or exactly the size of the ROI you set. GIMaP
does not check that the labels were made from this image.

#### Sizes from a CSV file

Press **Open CSV…**, then in **CSV measurement settings** choose:

- **Value column**, and whether it holds a **Radius** or a **Diameter** (**Stored size quantity**);
- **Stored unit**: `nm`, `µm` or `pixels` (pixels need the **Image calibration** of that source);
- **Delimiter** (`\t` for a TSV) and **Rows before header**; **Read CSV columns** reads the column names
  again.

No column is chosen for you. Press **Import particle sizes**: diameters are halved, µm converted to nm.
Rows that are empty, not numbers, not finite or not positive are rejected, with the reason in the table.

#### Distribution and comparison

| Setting | Meaning |
|---|---|
| **Histogram bins**, **Minimum radius (nm)**, **Maximum radius (nm)** | the bins; the maximum defaults to *From data and prediction* |
| **Display** ▸ **Probability (same bins)** (default) | SEM and prediction rebinned to the same edges, each normalised to a sum of 1 within the range; what falls outside is reported |
| **Display** ▸ **Paper display (independent peak scaling)** | 60 SEM bins; the prediction on its own radius axis, minimum subtracted, scaled to the SEM peak — the heights are then no comparable probabilities |
| **SEM radius scale** | 1 by default; scales the SEM radii for the comparison only — the measured radii stay in the particle table. Any other value is shown as a warning under the tabs (*⚠ SEM radius scale: …*) |

After a change, press **Update distribution**; an export needs an up-to-date distribution. **Remove
prediction** keeps the SEM measurement and shows it alone. Opening the window again from the **Tools**
menu also removes a prediction attached before.

#### Export

**Export SEM results…** asks for a folder and writes a new subfolder `SEM_analysis_<date>_<time>/` in it.
It refuses to overwrite result files that are already there.

| File | Content |
|---|---|
| `sem_particles.csv` | every object: kept or not, R, D; for an image also area, centre, circularity and the reason of a rejection |
| `sem_distribution.csv` | the bin edges, the SEM probability and, if attached, the prediction |
| `sem_distribution.png` / `.pdf` / `.svg` | the distribution figure (light colours), with the prediction if attached |
| `sem_display_arrays.npz` | the arrays drawn, and the measured radii |
| `sem_overlay.png` / `.pdf` / `.svg` | outlines, kept and rejected objects and the ROI, for review (image only) |
| `sem_segmentation.npz` | the grey image at full resolution (no display stretch), the labels, the outline overlay (image only) |
| `sem_instance_labels.tif` | the labels as int32 at full resolution, original IDs — reusable with **Existing instance labels** (image only) |
| `sem_analysis.json` | sources with SHA-256, reader, calibration, ROI, method and its settings, filters, counts, comparison settings, the prediction's source, and the list of files |

The CSV files carry the source and settings in a `#` header line too. A StarDist run also records the
model's configuration, thresholds and the SHA-256 of its weights; the weights themselves are not copied.

Reading, extracting, updating the distribution and exporting run in the background. **Cancel** discards
the pending result; a file already being written may still be completed, and nothing already written is
deleted.

## 10. AI assistant

**Ask AI…** in Analyze's command bar, or **Tools ▸ Process with AI…** (`Ctrl+Shift+L`), lets an AI
analyse the frame shown in Analyze — GIWAXS or GISAXS — with the same tools you use. It switches the
mode, adjusts sectors and cuts, sets the I(χ) window, fits peaks or the horizontal cut and exports
files, and the page follows every step on screen. The AI is optional: **Run Automatic Analysis** gives
the standard results without it (section 5.5).

During a run a panel shows a summary of the AI's reasoning, each tool call (hover for its arguments and
result), the change cards and, at the end, the report.

### 10.1 The three brains and their set-up

Choose the brain in **Tools ▸ Settings… ▸ Assistant**, or under *Brain* in the start dialog (**Set Up
AI…** opens the settings).

| Brain | Billing | Set-up |
|---|---|---|
| **Claude Code — your Claude plan** (default) | counts against your Claude Pro or Max plan; no API key | Claude Code must be installed (the Claude desktop app includes it: `claude.exe` under `%LOCALAPPDATA%\Packages\Claude_…\LocalCache\Roaming\Claude\claude-code`, found by itself or chosen with **Browse…** beside **Program**) and signed in once: **Sign In…** opens a window running `claude auth login`; choose your account, then press **Check**. Model and effort are set in Settings (empty model = Claude Code's default). |
| **Claude API — API key** | per token | `pip install anthropic` once; type the key under **API key** in Settings and press **Save** (or set `ANTHROPIC_API_KEY`); **Test Connection** checks key and model. Default model `claude-opus-5`, effort *high*. |
| **Other AI providers — OpenAI-compatible API** | as the provider bills | `pip install openai` once. Under **Other AI providers** pick the provider (the address is filled in; change it for a local server or Azure), save its key or set its environment variable (e.g. `DEEPSEEK_API_KEY`, `DASHSCOPE_API_KEY`), choose a model (typed, suggested, or **Get List**) and press **Test Connection**. |

The other providers include DeepSeek, Qwen (Alibaba Cloud Model Studio), OpenAI, Kimi, Zhipu GLM, Google
Gemini, OpenRouter, SiliconFlow, Azure OpenAI and a model on your own computer or lab server (Ollama, LM
Studio, vLLM …). The model must support tool calling: **Test Connection** sends one tiny request with a
tool.

**Where the keys are kept.** A key saved with **Save** is written **in plain text**, not encrypted, into the
user data folder: `claude_api_key.txt` for the Claude API, `assistant_provider_keys.json` for the other
providers (one entry each). The files are meant for your user account only, but anyone who can read that folder
— an administrator, a backup, a `GIMAP_HOME` on a shared drive — can read the keys. A saved key takes precedence
over the environment variable; **Remove** deletes it. On a shared computer, set the environment variable
instead, or remove the key when you are done.

Claude Code gets only the GIMaP tools — no shell, files or web — and no API key is passed to it.

**Standing instructions for the AI** (**Tools ▸ Settings… ▸ Assistant**) are added to every run, for example where
your calibrations are kept or the energy of your beamline.

### 10.2 Starting a run

![The start dialog of Process with AI](manual_images/en/18_ask_ai.png)
*The start dialog: results to report, notes, brain, permissions, report language. The line above the buttons says whether the brain is ready and what is sent. Here Claude Code is not installed on this computer, so the line says where to get it (or to choose `claude.exe` in Settings ▸ Assistant) and **Start** stays greyed out; with Claude Code signed in, the line names the plan and the model instead.*

1. Open a frame in Analyze and press **Ask AI…**.
2. Tick the results you want:
   - GIWAXS: *Peak table (q, d, FWHM, intensity)*, *Orientation: in-plane vs out-of-plane*,
     *Orientation distribution of one ring* (I(χ), maxima, Hermans orientation factor; pick the ring
     or let the AI choose), *Crystallite size (Scherrer)*;
   - GISAXS: *Horizontal cut at Yoneda, centred, halves chosen*, *Length scale 2π/q\* of the cut*, *Fit of
     I(qy): shape, radius, distance*.
3. Optionally write notes in **Sample and instructions (optional)**: the sample, what you want to know,
   where the calibration is.
4. Choose the **Permissions**:

   | Permission | What happens |
   |---|---|
   | **Preview first** (recommended for new users) | every settings change the AI makes is undone when it finishes and comes back as a card to apply |
   | **Ask before writing files or changing corrections** | a dialog asks for each such action; a refusal is respected |
   | **Fully automatic** | everything is done, logged and can be undone |

5. Choose whether to **Send a small q-map image (qualitative; Claude only)** — the AI then judges the map by
   eye, every number still comes from the tools — and the **Report language**, then press **Start**. What a
   run does is the tooltip of **Start**.

**Stop** ends a run at any time. The report lists each requested result as *done*, *partial* or *not
available* with the reason (for example “no peak above 3σ between 0.2 and 2.0 Å⁻¹”). The tables below it
come from the tool results, not from the AI's text. **Save Report…** writes an HTML report and a JSON
record (`<frame>_ai_report.html` / `.json`; proposed in `gimap_analysis/` next to the data when that
folder exists, otherwise next to the frame).

**Baseline first, then questions.** The AI may start with GIMaP's standard procedure in one step (the
same as Run Automatic Analysis, with each decision and its reason). Its choices are defaults, not limits:
the AI then follows what your question needs — the start of an in-situ series, lines every sample shares,
peak ratios, a peak the baseline skipped — and marks interpretations as hypotheses, with the numbers
behind them.

### 10.3 Change cards: preview, apply, undo

Every change the AI makes to Analyze (sector widths, a custom sector, a q box, the frame, the mode, αi,
the radial bins, the valid intensity range) is recorded; the AI can also suggest changes without making
them. They appear as cards under **Changes**: a picture of the q map with the change, the setting in
words, why the AI suggests it, and **Apply**, **Dismiss** or **Undo**. **Undo All** takes back every
applied change.

In *Preview first* the AI's own exploration is restored at the end. A setting the AI proposed a value
for shows only that proposal; otherwise the net change of each setting remains as a card. Changes that
ended where they started, or changed nothing visible, are left out.

### 10.4 Follow-up questions

When a run has finished, type a question in the field under the report (e.g. “Why is there almost no
signal in the in-plane sector?”) and press **Ask**. A new run on the same frame and settings answers it,
knowing what the previous run found, and can bring new cards.

### 10.5 A frame without geometry

When no instrument profile matches the frame, the AI does not stop; it looks for the geometry.

1. **What you named.** Files or folders named in the notes or in the standing instructions are read
   first — e.g. *“the calibration is D:\beamtime\calib\img_0005.cbf, 12.4 keV”*. A calibration named by
   its file name alone (*“calibration LaB6_take2.poni”*) is used too: the file of that name in the folder
   of the frame, or in a folder the notes name.
2. **A search.** The frame's folder and subfolders, the parent and its other subfolders, two folders up
   with their subfolders, and calibration / log folders three up; further folders up or down can be
   searched read-only. It looks for:
   - calibration results — pyFAI `.poni` files and calibrations exported by Geometry Calibration
     (`.json`);
   - images of a calibration standard, the standard taken from the file or folder name (AgBH / silver
     behenate, LaB6, CeO2, a LaB6 + CeO2 mixture; Si, Al2O3 and Cr2O3 are recognised but cannot be
     fitted yet); a multi-module NeXus series (`*_m01.nxs` … `*_m11.nxs`) counts as one image;
   - logs and parameter files (`.log`, `.fio`, `.txt` …) that state the energy, distance, beam centre or
     incidence angle.
3. **The choice.** Ready calibration files are listed by folder first (the frame's own folder, then
   its subfolders, the parent folder, the folders beside it, further up), then by file time. Of the
   standard images it prefers one of the same kind of detector file taken before the frame, with
   *giwaxs*/*waxs* and *final*/*redone* in its name. Standard images are fitted with GIMaP's
   calibration and judged by where the standard's lines land in q: within 0.2 % on average is good,
   within 0.5 % usable ([Physics and Methods, section 3.4](Physics_Manual.md#34-quality-measures)). When the name does not say which standard an image shows, every standard is
   fitted and compared the same way.
4. **Asking you.** Only when the numbers cannot decide, or the energy is nowhere to be found, a dialog
   **The AI asks** lists the options with their file times and folders, or asks you to type the value.

Files within two folders above the frame and files you named are read without asking; other files are
read after you agree (*Ask before…*) or with a note in the panel (the other two permissions). Hidden and system
folders (`.ssh`, `AppData`, `Windows` …) are never read. The chosen geometry is saved as the instrument
profile of the detector — with *Ask before…* only after you approve it — and the report states which file,
standard and values it came from. **Check it**: a calibration found by the search can belong to another
experiment (chapter 13).

### 10.6 What is sent, and what is kept

- **Sent to the AI**: the frame's status (file name, geometry, settings) and the reduced curves as
  numbers. The detector file itself is never sent; the small q-map image only when you allowed it. With
  a local model (Ollama, LM Studio …) nothing leaves your computer or lab network.
- **Shown**: the tokens of each run; with Claude Code also the plan's usage-limit warnings and the cost
  at API prices (for comparison only — a plan is not billed per token); with an API the estimated cost.
- **Kept on your computer**: every run in `assistant_runs/` of the user data folder; analyses the AI
  found missing in `assistant_feature_requests.jsonl` there; a key you saved, in plain text (section 10.1).

**How to read the numbers.** The rules behind them are in Physics and Methods:
[section 7.7](Physics_Manual.md#77-ring-orientation-and-the-hermans-orientation-factor) (Hermans),
[section 7.8](Physics_Manual.md#78-crystallite-size-scherrer) (Scherrer) and
[chapter 8](Physics_Manual.md#8-gisaxs-automatic-analysis) (GISAXS).

- Scherrer sizes are lower bounds of the coherence length unless the instrumental width is known; broad
  halos (FWHM above 15 % of q) get none.
- The Hermans factor assumes a film that is isotropic in its plane and weighs each |χ| by sin χ. It is
  given only when at least 80 % of that weight is measured. The report also gives f_iso, the f of a random
  ring over the same |χ|, and f − f_iso: *along the surface normal* or *in the sample plane* is named only when
  |f − f_iso| reaches max(0.1, 3σ_f), while a ring whose I(χ) maximum lies between 20° and 70° is called *tilted*
  from that maximum alone. f weighs the bins next to the sample horizon most; when it depends on them beyond its
  statistical uncertainty, the ring's notes say so.
- Regions far below the diffuse background at the same q (under 20 %: a shadow, an absorber, an
  insensitive detector area) count as unmeasured, and a sector lying in one is not compared. Intensity
  next to the qz axis may be unmeasured (missing wedge).
- Peaks one or two bins wide far above the background are marked as spikes (hot pixels, module edges),
  not diffraction.
- Crystallographic labels such as (100) or edge-on are used only when the sample notes or the data make
  the assignment clear.

### 10.7 Without the window

`tools/gimap_agent.py` runs the same standard procedure without the window, for scripts and other
agents: `python tools/gimap_agent.py auto <frames> --notes "…" --out <folder>` writes a `report.md` per
frame with every decision and what needs your answer. A series is analysed in its final state: the last
fifth of its frames summed, at most 10 (below ten frames the last frame alone); `--frame N` and `--sum K`
choose other frames; `--max-peaks N` fits more than the 12 most prominent candidate peaks. The module files of one detector are analysed once, however many of them are given,
in a folder named by the scan (a part missing is an open item of its report); `--series` takes files that
hold one frame each as one series. Its options, exit codes, where its defaults may
not fit your question, and how to check it on a GIWAXS measurement you know (`tools/eval_giwaxs_agent.py`;
a case built on your own data goes in `docs/agents/eval/local/`, which Git ignores) are described in
`docs/agents/giwaxs-playbook.md` and `docs/agents/gisaxs-playbook.md`.

## 11. Labs

The **Labs** pages (sidebar ▸ Labs) are experimental. Their settings are the *Labs parameters*: **File ▸
Labs Parameters ▸ Load… / Save As…** reads and writes them, together with Fitting's model.

### 11.1 2D Prediction

**2D Prediction** runs a trained prediction module on GISAXS detector frames. Three steps on the left,
the canvas on the right.

![2D Prediction with a frame loaded](manual_images/en/20_prediction.png)
*A CBF frame is loaded and shown under Input preview; no model is imported yet. The three states under 3. Predict say what is still missing, and Predict stays disabled until all three are ready.*

1. **Import data.**
   - **Single file**: **Choose CBF frame…**, or drop a CBF frame on the canvas. **Stack** is how many
     consecutive detector files make one input.
   - **Folder batch**: a folder of numbered frames, the **Range** of frame numbers and **Files per
     prediction**; a line under the fields says how many files and predictions that makes.

   Prediction reads CBF frames (`.cbf`) only.
2. **Import model.** Choose the **Prediction setup** — a module from `modules/` (section 11.3) — and
   press **Import model...**; the *Model* state under 3. Predict says whether it is loaded. Folded: **Advanced model
   configuration** (the framework check, **Reload Config**, **Edit Config**) and **Advanced model
   sources** (the model library).
   - The shipped modules come without a model. The first **Import model...** asks for the folder of the
     trained model (a TensorFlow SavedModel folder). GIMaP remembers that folder for you in the user data
     folder (section 3.11), not in the module's `module.yaml`, so the program files stay unchanged; the
     next **Import model...** loads it without asking.
   - Press **Import model...** again once the model is loaded to choose another folder.
   - The Activity log says when a module has no model yet, and names a remembered folder that no longer
     exists. **Reset All Settings…** also forgets the remembered folders.
3. **Predict.** Three states — *Input*, *Model*, *Framework* — say what is still missing; **Predict** is
   enabled when all three are ready. A prediction runs in the background; **Stop** ends it.

The canvas shows **Input preview** once a frame is loaded and **Prediction result** after a prediction.
*Preprocessed* shows the steps of the module in order with the image each one produced (click one to see
it large).

- **Export input...** saves the frame as displayed, with a JSON record.
- **Export result...** saves the result as a picture, its data beside it, and a JSON record of the
  module, the model, the preprocessing steps and the input file.
- A folder batch lists its results under **Batch results**, with filter, sorting and export.
- The **Activity log** is folded at the bottom.

**SEM comparison…** (above the canvas) opens **SEM Particle Analysis** (section 9.3) with the current
prediction attached, when the module declares a radius axis in nm in its `module.yaml` (section 11.3);
otherwise the tool opens without a prediction and the Activity log says why. Of the shipped modules
only `Au_Silicon_15nm` declares its axes. Then open the SEM image or the sizes CSV there. What is attached is a copy: the radius distribution with its source (input files, module, model,
preprocessing, runtime). Changing the input or model on the prediction page afterwards does not change
an open comparison, and an outdated result is not attached again. Opening the SEM tool from the Tools
menu removes a prediction attached earlier.

The two distributions are not the same quantity. SEM gives the projected-area radius of the particles in
a small field of view, counted per particle. The prediction gives the radius distribution of the module's
model, from the scattering of the illuminated area. Read a difference as a result to discuss, not as an
error of either.

### 11.2 Trainset Build

**This feature is under development.**

**Trainset Build** designs a simulated GISAXS training set (BornAgain), checks it locally and prepares
the training. Five steps on the left; a step shows its state once it has one:

![Trainset Build](manual_images/en/21_trainset.png)
*Step 1, Dataset Design. The five steps and their states are on the left; the preview on the right shows the detector, region of interest and masks once a reference file is loaded.*

1. **Dataset Design** — optionally load a measured image as a reference for the geometry (**Load
   file...**: it guides the beam centre, the region of interest and the masks and is never used as a
   training image); then **Geometry + ROI**, **Particle population** and **Sampling + files**. **Validate
   design** checks the design.
2. **Local Preview** — simulated examples over the chosen parameter ranges, with the preprocessing
   (background, noise, masks and transforms under **Advanced preprocessing**).
3. **Model Design** — the training settings and the layers of the network.
4. **Local Run** — the job package, a local generation and training run with progress and **Stop
   safely**. Submission to the Maxwell cluster is disabled in this version.
5. **Monitor & Results** — the log, the metrics and **Register best model in 2D Prediction**. It writes a
   module `generated_<project>/` (`module.yaml` and `preprocessing.py`) into `modules/` of the program folder
   and opens 2D Prediction with it chosen as **Prediction setup**; its `module.yaml` names the trained model by
   its full path on this computer. In a git clone, git ignores these `modules/generated_*/` folders, so a
   registered model is never committed.

**Remember changes** keeps the design for the next session; **Reset defaults** restores the built-in
values (the note that follows offers Undo); **Load** and **Save** read and write a project file. In the
Chinese interface many texts of this page are still English.

### 11.3 Prediction modules

A prediction module is a folder under `modules/` with a `module.yaml`. Its main fields:

| Field | Meaning |
|---|---|
| `id`, `name`, `version` | internal identifier, the name shown, the version |
| `framework` | the model framework, for example TensorFlow/Keras |
| `model.model_path` | the default model: empty, or a path relative to the module folder. A folder chosen with **Import model...** is kept in your user data folder and takes precedence |
| `preprocess.entry`, `preprocess.steps`, `preprocess.params` | the preprocessing entry point, steps and parameters |
| `io.input_type`, `io.stack_axis`, `io.input_shape` | the input type (for example `cbf`), the stack axis, the model's input shape |
| `outputs` | the outputs: parameter names, ranges or output type; for a distribution, `axes: row` and `axes: column` with `key`, `label`, `unit`, `min`, `max` |

When you add a module, make sure the input type, the preprocessing code and the output definition match
the trained model. Leave `model_path` empty rather than writing a folder of your computer: each user
chooses the model with **Import model...**, which never writes into `module.yaml`. A path that a
`module.yaml` already names is still used when no folder was chosen. The one exception is a module that
Trainset Build registers (section 11.2): its `module.yaml` names the model file of this computer. What 2D
Prediction does with a frame, and what it does not correct, is in
[Physics and Methods, section 13.4](Physics_Manual.md#134-2d-prediction-labs).

For the SEM comparison, the module must declare the radius axis: `outputs: axes: row` with `key: r`,
`unit: nm`, `min` and `max`, as in `Au_Silicon_15nm`. A distribution whose module declares no axes is
drawn on a placeholder axis (0.05–15 nm on both axes), which is not a physical scale, and is not attached
to the SEM comparison.

The V5 methods of 1D Predict use their bundled model in `modules/Fitting_1D_Model/Workflow_v5`; fit1d
uses its installed `gisaxs1d_gui_kit` instead (section 6.4).

## 12. Output files

GIMaP never changes your data files. Exports from Analyze go by default into a folder
**`gimap_analysis/` next to the data**; most other exports ask for a place and propose one. Some propose a place
beside the data too (the XRR CSV, the Format Converter's `converted/`, a calibration, a project): on a read-only
data share, choose another folder. The user data folder is `%APPDATA%\GIMaP` on Windows and `~/.config/gimap`
elsewhere, or the folder in `GIMAP_HOME` (section 3.11; **Help ▸ Open User Data Folder**). The table lists every
file GIMaP writes; the paragraphs below it, what the user data folder and the Windows package hold.

| Action | Files written | Default location | Record that comes with it |
|---|---|---|---|
| Analyze ▸ **Export** (`Ctrl+E`), **Export Curves…** | `<stem>_<curve>.csv` per curve, `<stem>_fit_input.dat`, `<stem>_analysis.json` (`_frameNNNN` added for a frame of a multi-frame file, `_sumN` when frames are summed) | `gimap_analysis/` next to the data, without a dialog; a folder dialog if it cannot be written | `<stem>_analysis.json` (settings, geometry and a `.poni` tilt left out, corrections, pixels left out, error model, binning, UTC time of writing) and `#` header lines |
| Analyze plots ▸ **Save ▸ Curves as Data…** | the CSV files of that plot and `_analysis.json` | folder dialog in `gimap_analysis/` | as Export |
| Analyze plots and images ▸ figures (**Plot as Figure…**, **View as Figure…**, **Analyze ▸ Figures ▸ Save Image…** …) | PNG, TIFF, SVG or PDF, light colours | dialog in `gimap_analysis/` | none |
| Analyze ▸ **Save q-Map Data…**, **View Data as CSV…** (also **Analyze ▸ Figures**) | `<stem>_qmap.csv`, `<stem>_cake.csv` | dialog in `gimap_analysis/` | `#` header lines (source, frame, axes) |
| Analyze ▸ Series ▸ **Export** (**Map as CSV Table…**, **Map as Figure…**, **Stages as Table…**, **Selected Frame's Curve as Figure…** / **as Table…**, **Trace as Figure…** / **Trace as Table…**, **Peak Table of Every Frame…**) | `<series>_<curve>[_<halves>]_map.csv`, `…_map.png`, `…_stages.csv`, `…_trace_<value>.csv` and `….png`, `…_peaks.csv`; the frame picked as `<frame>_<curve>[_<halves>].csv` and `.png` (figures also as TIFF, SVG or PDF) | dialog in `gimap_analysis/` | `#` header lines (curve, halves, frames, cut) and a `.json` of the settings next to each table; none for the figures |
| Analyze ▸ **Send to Fitting** | `<stem>_fit_input.dat` (q in Å⁻¹) | `gimap_analysis/` next to the data; a folder you choose (kept for the session) when that cannot be written | `#` header lines (source, frame, sum, curve, observation with the halves) |
| Analyze ▸ **Batch Export…** | per-curve tables `<batch>_<curve>_frames.csv` (or `.txt`, `.dat`); if you chose fitting, a fit table `<batch>_peak_fits.csv` (or `<batch>_model_fits.csv`) with its trend plot `<batch>_peak_fits.png` (`_model_fits.png`) and, if chosen, the fitted curves `fits/<frame>_<peak or model>_fit.csv`; subfolders `curves/`, `fit_input/`, `maps/`, `images/`, `frames/` after your choices; frame names keep the 0-based index (`…_frame0003_sum3_…`) | the last batch folder, else `gimap_analysis/` next to the data; by default in a subfolder named after the frames' folder (one file: its name; several folders: `batch`); **Keep Both** writes into a new subfolder | `README.txt` and `<batch>_batch.json` (every file written, the cut of every frame, failed frames and why); `curves/` also gets an `_analysis.json` per frame |
| Analyze ▸ **Send Series to Fitting…** | `fit_input/<frame>_fit_input.dat`, `README.txt`, `<batch>_batch.json` | a folder of its own, `<name>_fitting/`, next to any Batch Export of the same data | `README.txt`, batch JSON |
| Analyze ▸ **Export New Frames While Watching** | as Export, for each new frame | `gimap_analysis/` next to the frame; when it cannot be written, a folder you choose once (kept for the session); with none, the export is switched off and the status line says why | as Export |
| Analyze ▸ **Save Settings…**; Mask step ▸ **Save…**; Regions ▸ **Add ▸ Save Cuts…** | `analyze_settings.json`; `masks.json`; `cuts.json` | dialog in `gimap_analysis/` | the file is the record |
| Run Automatic Analysis ▸ **Save Report…** | `<stem>_report.html` (with pictures) or `.md` | the folder you chose last in this session for frames of the same folder, else `gimap_analysis/` if it exists, else next to the frame | the report itself (no JSON) |
| Process with AI ▸ **Save Report…** | `<stem>_ai_report.html` + `<stem>_ai_report.json` | as above | the JSON |
| Process with AI (automatic) | a record of every run; missing analyses | user data folder: `assistant_runs/`, `assistant_feature_requests.jsonl` | — |
| Process with AI, when the AI exports | the curves as Export, and `<stem>_assistant.json` | `gimap_analysis/` next to the data | `<stem>_analysis.json`, `<stem>_assistant.json` |
| Fitting ▸ Single analysis ▸ **Save ▸ Data and Fit…** | `<curve>_fit.csv` + `<curve>_fit.json` (written as a pair) | the folder saved to last, else the folder a curve was last opened from, else the curve's folder | the JSON (model, method, range, left-out points; fit1d keeps its own curves and warnings) |
| Fitting ▸ **Save ▸ Plot…**, **Save ▸ Model…** | `<curve>_fit.png` / `.svg`; `<curve>_model.json` | as above | none; the model file |
| Fitting ▸ In-situ series ▸ **Save ▸ Table of Every Frame…** | `series_fit.csv` + `series_fit.json` | the series folder | the JSON (`gimap_series_fit_v2`: start model, halves and where they came from, every frame's state and reason; for detector frames Analyze's settings) |
| Fitting ▸ In-situ series ▸ **Trend Plot…**, **Selected Frame's Plot…** | `series_trend_<value>.png` (the table column of the value, for example `series_trend_chi2_reduced.png`) and `series_frame_<NNNNN>.png` (the frame counted from 1, five digits), or `.svg`; `series_trend.png`, `series_frame.png` while nothing is fitted | the series folder | none (the title in the picture names the value or frame) |
| **Tools ▸ 1D Predict — Fit Many Curves…**, and Fitting's *AI proposal (1D Predict)* | a run folder: `request.json`, `batch_summary.json` …, a subfolder per curve with `solutions.json`, `fitting_curves.csv` … | `AI_Fitting_Output/v5_<date>_<time>_<µs>/` in the folder GIMaP was started from (portable version: the program folder); never removed automatically | `request.json` |
| Compare ▸ **Save ▸ Table of Every Series… / Table of Every Frame…** | `compare_series.csv` / `compare_frames.csv` + `.json` | the folder of the first series added from files, else the folder a series was last added from (while it is there), else your home folder | `#` header lines and the JSON |
| Compare ▸ **Change Plot… / Paths Plot… / End States Plot…** | `compare_*.png` or `.svg` | as above | none |
| Sputter ▸ **Predict** | a folder per scan and request: `structure_physical.csv` + `structure_physical_summary.json` (the physical path), `structure.csv` + `structure_summary.json` (per window) with the model's own trend figure `structure_trend.png`, and the frames as the model read them (`intake/`) and summed into windows (`binned/`). `structure_selected.csv` and `structure_sequence.csv` (each with its summary and `_trend.png`) are there only when made with the toolkit's `select` and `sequence` commands | user data folder: `sputter/<scan>-<code>/` (or **More options ▸ Save results in**); the same request reuses its folder | `gimap_request.json` (the request, the Python, PyTorch's version) |
| Sputter ▸ a new **Deposition rate**, **t0** or **Deposition end** after a run (section 8.7) | `structure_physical.csv` + `structure_physical_summary.json` written again (no frame is read) | the result folder | the summary |
| Sputter ▸ **More options ▸ Add Refined Windows…** | `structure.csv`, `structure_summary.json`, `structure_trend.png` and the physical files written again with the refined windows | the result folder | the summaries |
| Sputter ▸ Structure Sketch ▸ **Save Image… / Save Storyboard…** | `sketch_<time>s.png` / `storyboard.png`, or PDF, SVG | the result folder | none |
| Sputter, in the background after a result is shown | `fringe_heights.csv` (H from the q_z fringes per time group) + `fringe_summary.json` | the result folder's `fringes_v1/`; when it cannot be written, the user data folder: `sputter_fringes/<result folder>/` | the summary: SHA-256 of `binned/run.json`, the chunks and `structure_physical.csv`, the tool's version, αc and its source, the settings |
| XRR Series Extractor ▸ **Export CSV…** | `<source>_xrr.csv` + `.json` | next to the source file; for a folder chosen with **Folder…**, inside it, named after it | the JSON (every value that determines qz and its origin; `center_convention`; `source_layout` with every file read) |
| Format Converter | converted frames: `<source>.<ext>` for a single-frame source; for a multi-frame source by default `<source>_<frame counted from 1, 6 digits>.<ext>`; `_001` … added on a name collision; or one `converted_images.h5` (`converted_images_001.h5` … when it exists) | `converted/` next to the first input | `conversion_report.json`, `conversion_metadata.json` (or one JSON per frame) |
| SEM Particle Analysis ▸ **Export SEM results…** | the files of section 9.3 | a new `SEM_analysis_<date>_<time>/` in the folder you choose; never overwrites | `sem_analysis.json`; `#` header line in the CSVs |
| 2D Prediction ▸ **Export input...** / **Export result...** | `<input>_input.png` + `.json`; `<input>_prediction.png` + `.json` (+ `.txt` when the result has a curve or map) | the export folder used last, else the input's folder | the JSON |
| 2D Prediction ▸ **Export result...** with several files | your choice of `prediction_results_<time>.jsonl`; `prediction_curves_<time>.txt` (or `prediction_parameters_<time>.txt`); images in `prediction_images_<time>/` | a folder you choose (starts at the export folder used last) | `#` header lines in the `.txt` |
| Trainset Build ▸ **Prepare job package** | a job folder (`config.yaml`, scripts, `manifest.json` …) | `trainset_jobs/<project>/` in the program folder (the workspace can be changed) | `manifest.json` (checksums) |
| Trainset Build ▸ **Save** | `<project>.yaml` | the project folder used last, else the program folder | the file |
| Trainset Build ▸ Monitor & Results ▸ **Register best model in 2D Prediction** | `generated_<project>/module.yaml` + `preprocessing.py` | `modules/` in the program folder | the `module.yaml` (it names the model file by its full path) |
| Geometry Calibration ▸ **Export Calibration…** | `<image>.gimap-calibration.json` | next to the calibrant image | the file (beam centre counted from pixel centres at 0, 0.5 px lower than the windows show; `metadata.center_convention`) |
| Geometry Calibration ▸ **Apply** | the instrument profile | user data folder: `instrument_profiles.json` | — |
| **Tools ▸ Settings… ▸ Assistant** ▸ **API key** ▸ **Save** | `claude_api_key.txt` (Claude API), `assistant_provider_keys.json` (other providers), in plain text; **Remove** deletes the key | user data folder | — |
| **File ▸ Save Project** | `<name>.gimap`; `<name>.compare.npz` when Compare holds maps from Analyze | next to the first frame | the project (data referenced by path). Sputter results are not part of it: they stay in their result folder |
| **File ▸ Labs Parameters ▸ Save As…** | `gimap_parameters.json` | dialog in the folder GIMaP was started from | — |
| Errors (automatic) | `logs/errors.log` (moved to `errors.old.log` above 2 MB) | user data folder | — |
| **Help ▸ Example Data** | the example's frame with its `.poni` (and mask), and `SOURCE_AND_LICENCE.md` | a folder you choose, by default `Documents/GIMaP examples/<example>/`; files already there are kept | `SOURCE_AND_LICENCE.md` (source and licence of the data) |
| `tools/gimap_agent.py` | a subfolder per frame (a set of module files: per scan; `--series`: `<name>_series/`) with `report.md`, `report.json`, curves, q map; `summary.md` + `summary.json` | `--out`; without it `assistant_runs/cli/<time>/` in the user data folder; never next to the data | `report.json` |

**Which release wrote a file.** The records of Analyze (`_analysis.json`, the settings file, the series and
stages tables' JSON), of Batch Export and Send Series to Fitting (`<batch>_batch.json`; `README.txt` says *Written by
GIMaP v…*), of Fitting (`<curve>_fit.json`, `series_fit.json`), Compare, Geometry Calibration, the XRR Series
Extractor, SEM Particle Analysis, 2D Prediction and the Format Converter name the GIMaP release that wrote them:
`software_version`, for example `GIMaP v0.2.0`. In Analyze's records `software` stays `GIMaP Analyze 1`, the name
of the record format. The records of Sputter, the AI assistant, `tools/gimap_agent.py` and 1D Predict's run
folders do not name the release yet ([Physics and Methods, section 14.5](Physics_Manual.md#145-version-of-the-software)).

**What `<stem>_analysis.json` says besides the settings.** `created` is the UTC time the record was written
(ISO 8601). `error_model` says how the σ of the points were computed: `poisson` (photon counts),
`pixel_scatter` (values that are not counts as measured: a dark- or background-subtracted frame, pixels
filled from the mirror side) or `scaled_poisson` (GIWAXS intensity corrections on counts). `binning` gives,
for GIWAXS, the number, q range and width of the radial bins, how many points the I(q) kept after sparse
bins were joined and the rule for joining them; for GISAXS, one point per detector column or row. In
`geometry`, `dropped_tilt_deg`, `dropped_rotations_deg` and `flat_detector_q_error_max` / `_rms` give the
tilt of a `.poni` that the flat detector leaves out and the largest and RMS relative error in |q| it causes
(`null` when no tilt is known; section 4.2). The instrument profile's `source` says who chose the geometry:
`automatic analysis: <file>` for Run Automatic Analysis and `tools/gimap_agent.py`, `assistant: <file>` for
Process with AI; profiles saved before 0.2.0 say `assistant:` for both ([Physics and Methods, section 14.2](Physics_Manual.md#142-the-record-stem_analysisjson)).

A save dialog for one file asks before it replaces that file; the record written beside it (for
example a `.json`) is replaced without a question. Exports without a file dialog — Export, **Curves as
Data…** (a folder dialog), Send to Fitting, watching, Batch Export, Send Series to Fitting, the AI's
exports, the converter's report and metadata JSON — replace files of the same name in their folder. Batch
Export and Send Series to Fitting first ask about an earlier export of the same data (**Replace**, **Keep Both
in a New Subfolder**, **Cancel**; section 5.7). The Format Converter never overwrites converted images (it adds
`_001` … or stops), and SEM Particle Analysis never overwrites its results. Sputter writes the tables of a
result folder again when you change the rate, t0 or end or add refined windows, and **Register best model in 2D
Prediction** replaces the module of the same project.

**Text files.** Every text file GIMaP writes (CSV, TXT, DAT, the curves for Fitting, `README.txt`) is UTF-8
without a byte-order mark. GIMaP's own symbols are written in ASCII — `1/A`, `deg`, `2theta`, `alpha_i`, `chi`,
`sigma`, `+/-` — and file names, frame names and paths as they are; the JSON records keep every name exactly. A
script should read the files with `encoding="utf-8"` (for example `numpy.loadtxt(path, encoding="utf-8")`). A
name outside ASCII may be shown wrongly when a CSV is opened in Excel by a double-click; the numbers are not
affected.

**In the user data folder.** Besides the files of the table, GIMaP keeps there `settings.json` (the settings
and what the pages remember, section 3.9), `instrument_profiles.json`, `session.json`, `model_parameters.json`
(Fitting's model), `last_analyze_setup.json` and `logs/`; the AI assistant's `assistant_runs/`,
`assistant_feature_requests.jsonl` and saved keys (`claude_api_key.txt`, `assistant_provider_keys.json`, plain
text); and Sputter's result folders in `sputter/` (and `sputter_fringes/`), which are never removed. Delete old
result folders yourself, or choose **More options ▸ Save results in** on a data disk for long scans. Outside
that folder, `Start_GIMaP.bat` keeps Matplotlib's font cache in `%LOCALAPPDATA%\GIMaP\matplotlib`, and on
Windows whether a few cards of Fitting and 2D Prediction are folded is kept in the registry, under
`HKEY_CURRENT_USER\Software\GIMaP`.

**The Windows package.** Besides the program, the ZIP holds `licenses\`: the licence files of every component of
the package, listed in `licenses\README.txt`, with `THIRD_PARTY_NOTICES.md` (**Help ▸ Third-party Notices**)
and `LICENSE`. The release page lists the SHA-256 of every file to download in `SHA256SUMS.txt`: check a
download with `Get-FileHash -Algorithm SHA256 <file>` in PowerShell, or `sha256sum -c --ignore-missing
SHA256SUMS.txt` (it checks the files you downloaded).

## 13. Known limitations

Version 0.2.0 is a pre-release. These limitations are known; each comes with what to do. This chapter is the complete list for this version. What the methods themselves leave out (scattering physics, instrument, statistics) is in [Physics and Methods, chapter 15](Physics_Manual.md#15-what-gimap-does-not-model).

| Limitation | What to do |
|---|---|
| **Fitting does not read `.csv` curves**, although Analyze exports its curves as CSV. | Use **Send to Fitting**, or open the `<name>_fit_input.dat` that Analyze writes; `.dat` and `.txt` files with columns `q I [σ]` open too. |
| **Geometry Calibration can put a wrong solution first.** With the standard on **Auto Detect** and no estimated distance, another standard can come first; a wrong **Estimated distance** misleads also with the right standard, and the field can hold a distance you did not type (section 4.3). | Choose the standard yourself when you know it. Leave **Estimated distance** on *Optional* or type the approximate distance of your set-up; set a number you did not type back to 0 unless it is that distance. Before **Apply**, check the distance and that the drawn rings lie on the rings of the image. |
| **Geometry Calibration fits silver behenate, LaB₆, CeO₂ and a LaB₆ + CeO₂ mixture only.** The AI's geometry search recognises images of Si, Al₂O₃ and Cr₂O₃ by their names but cannot fit them (section 10.5). | Calibrate such an image with pyFAI and give GIMaP the `.poni`: name it in the notes, or put it beside the frames (section 4.2). |
| **An instrument profile is matched by detector name and frame size only**, and **a calibration found by the automatic search may belong to another experiment** (one beside or above the frame's folder is taken when that folder holds none; file times say nothing in a folder copied in one go) — or even to another detector: a `.poni` or `.json` is applied without checking its pixel size or frame size (section 4.2). | Read which file the report names and check the distance in **Geometry**. Name the calibration in the notes, or give it yourself; remove a wrong profile with **Geometry…** ▸ **Delete profile**. |
| **GIMaP's detector is flat and perpendicular to the beam**: the tilt of a `.poni` (`Rot1`, `Rot2`) is left out, and so are `Rot3` and its `orientation` entry; Geometry Calibration fits no tilt. On a tilted detector \|q\| is off away from the beam: by up to 6.3 % on the Xeuss example (6.06°). Above 1° of tilt or 0.5 % in \|q\| the automatic analysis says so (section 4.2). | Read the error in the report or in `<stem>_analysis.json` (`flat_detector_q_error_max`). For exact q, reduce such frames with the full `.poni` geometry (pyFAI), or calibrate with the detector perpendicular to the beam. |
| **Run Automatic Analysis (GIWAXS) fits at most 12 candidate peaks**, the most prominent; the window has no setting for more. The others are left out of the peak table, the sector comparison, the rings and the series hints; *Decisions* says how many candidates there were (section 5.5). | Run `tools/gimap_agent.py auto … --max-peaks N` (section 10.7), or ask the AI to run `find_peaks` with `max_peaks`, when you need the full list. |
| **The GIWAXS and GISAXS q are kinematic: no refraction correction.** Near or below the film's critical angle, the q_z of a peak can be off by up to about k·sin αi, and in GISAXS that of the Yoneda band most, by an amount of that order (the report's *Decisions* give the number for your frame); the ± and σ of the report contain neither that nor the geometry's uncertainty ([Physics and Methods, sections 7.9 and 8.6](Physics_Manual.md#79-what-the-report-states)). | Measure above the critical angle for positions, or correct q_z yourself for the material; quote the systematic error separately from the statistical ±. |
| **Hermans f and the texture sentence are not corrected for the exit angle.** f weighs the bins next to the sample horizon most, where absorption in the film or a shadow can lower the intensity; on the public P08 frame leaving out the bins within 5° of the sample plane moves f − f_iso by 17–27 σ_f ([Physics and Methods, section 7.7](Physics_Manual.md#77-ring-orientation-and-the-hermans-orientation-factor)). | Read the ring's notes and the *ring orientation* line of *Decisions*; when they name the ring, check I(χ) near χ = ±90° before reading f as texture. |
| **XRR Series Extractor can place a beam centre from a NeXus header wrongly** (files of the Lambda server, section 9.2). | Check on the preview that the cross lies on the direct beam, or use **Pick direct-beam center**. CBF series are not affected. |
| **NeXus/HDF5 files not written by the Lambda server are now read as stored** (earlier versions transposed and flipped them). A geometry or mask made on such a file with an earlier version no longer fits it (section 5.1). | Check the beam centre and masks of such data; calibrate again if needed. |
| **XRR intensities are raw**: no monitor normalisation, footprint, background or resolution correction, and no θ offset. A θ motor dataset in µrad is read as degrees (only `mrad` and `rad` are converted). Analyze's bad-pixel search and masks are not applied, and negative values of a signed-integer NeXus/HDF5 frame enter the sum (section 9.2). | Apply these corrections yourself to the exported CSV. Give θ in degrees, mrad or rad, and keep the ROI off module gaps and hot pixels; check the valid-pixel count of each point. |
| **Format Converter does not read EDF.** | Use **Analyze ▸ Batch Export ▸ Converted detector frames**. |
| **The uint16 modes of the Format Converter mark pixels without data only in HDF5**: in a TIFF or CBF such a pixel holds 65535 and reads back as a valid value (section 9.1). | Keep the stored data type (the default), or write HDF5, when the gaps and bad pixels matter. |
| **SEM sizes depend on the segmentation**, and the default StarDist weights are trained on cell nuclei, not SEM. The scale is never read from the image. | Enter the calibration from the scale bar; check every outline; prefer a model trained on SEM images. |
| **fit1d** (Predict parameters) offers spheres and vertical cylinders only, not random cylinders; with **Try …** compositions the Model's values and ranges are not passed. It needs the separately installed kit. | Use **The particles in Model** to pass fixed values and ranges; treat the result as a start for **Refine These Values**. Without the kit, use *Find the particle shape (no AI)*. |
| **2D Prediction reads CBF frames only.** | Convert other frames to CBF with the Format Converter. |
| **2D Prediction applies no geometry and no correction to a frame**: the module's own preprocessing is all that runs (section 11.3). | Use a module only for data of the set-up it was trained for. |
| **The Ag and Cu distribution modules of 2D Prediction declare no axes**: their maps and curves are drawn on a placeholder axis (0.05–15 nm), which is not a physical scale, and they are not attached to the SEM comparison (section 11.3). | Read only the shape of these distributions. When you know the axes of the trained model, add them to the module's `module.yaml`. |
| **Trainset Build** is under development; submission to the Maxwell cluster is disabled. Its BornAgain grid cache stores images as float16, which overflows above 65504: with the default design, part of the pixels of a training image can come out not finite ([Physics and Methods, section 13.5](Physics_Manual.md#135-trainset-build-labs)). | Use the local run; check the local preview for pixels that are not finite. |
| **Trainset Build (Labs) is not fully translated**: in the Chinese interface many texts are still English; 2D Prediction has a few such texts too. | — |
| After the interface language is switched in a session, a few texts keep the old language (e.g. the outcome line of a finished automatic analysis). | Run the analysis again, or restart. |
| Quitting GIMaP during a Batch Export keeps the files of the frames done, but the batch tables, `README.txt` and the batch's JSON record are not written. | End a batch with **Stop**: it writes them. |
| 1D Predict writes its run folders into the folder GIMaP was started from; they are not removed. | Delete old `AI_Fitting_Output/v5_…` folders yourself. |
| **The quick model fit of a GISAXS cut** (Run Automatic Analysis; *Find the particle shape (no AI)* in Fitting) **is a screening step, not a size measurement.** On 14 BornAgain simulations of cuts of Ag spheres with known radius, distance and disorder (the GALAXI model, Poisson noise at the data's count level), the fit ranked a random cylinder first in 12 and a sphere in 1, 10 first-ranked solutions had a value at a search bound, and the best sphere solution's R was a median 11 % above the R⁶-weighted mean radius of the simulation (from −29 % to +26 %); D and its disorder were not recovered. 2π/q\* of a peak or shoulder was within 10 % of the true distance in 1 of 10 simulations with interference, and a peak or shoulder appeared in all 4 without (sections 1.3 and 5.5; [Physics and Methods, section 8.5](Physics_Manual.md#85-the-quick-model-fit-and-how-the-solution-is-chosen)). | Choose the model from what you know of the sample and confirm it in **Fitting** (refine it, judge the residuals). Do not quote 2π/q\* or the quick fit's D as the particle spacing. |
| **1D Predict (Workflow V5) is validated on simulated curves only**, with the composition and resolution given; finding the composition (none given in **Complete composition**), an unknown resolution and measured data are not validated ([Physics and Methods, section 10.8](Physics_Manual.md#108-validation-status)). | Treat its solutions as a second opinion: refine them in Fitting and judge them by the residuals. |
| **No record holds the random seeds, and pixels left out are counted, not located.** The records of Sputter, the AI assistant, `tools/gimap_agent.py` and 1D Predict's run folders do not name the GIMaP version; the other export records do (`software_version`, chapter 12) ([Physics and Methods, section 14.9](Physics_Manual.md#149-what-the-records-do-not-yet-contain)). | Note the version (**Help ▸ About GIMaP**) with results from those, and name it when you cite GIMaP (section 14.12). |
| **API keys saved in GIMaP are kept in plain text** in the user data folder (`claude_api_key.txt`, `assistant_provider_keys.json`; section 10.1): whoever can read that folder can read them. | On a shared computer, or with `GIMAP_HOME` on a shared drive, set the key as an environment variable instead, or **Remove** it after use. |
| **Sputter does not determine the thickness of a continuous film**, and a film with particles on it is not modelled (section 8.8). | Read the island-volume equivalent t_eq (island windows, conditional) and the nominal thickness from a deposition rate; compare with XRR, a quartz balance or a cross-section. |
| **Sputter reads Cu with the Ag model** (the networks were trained for Ag and Au). | Use Cu results for reference only. |
| **Sputter reads folders of CBF frames only**, and its networks were trained for Ag and Au on polymer substrates (chapter 8). | For NeXus series or another substrate use Analyze's series map (section 5.6) and Fitting's in-situ series (section 6.6); converting frames to CBF does not make another substrate valid. |
| **Sputter keeps every result folder**, with the frames as the model read them (`intake/`, `binned/`), in the user data folder unless you chose another place; nothing removes them, and **File ▸ Save Project** does not hold Sputter results. | For long scans choose **More options ▸ Save results in** on a data disk; delete old folders yourself; open a result again with **More options ▸ Open Results Folder…**. |
| **Sputter's island height H from the q_z fringes is the height of a uniform layer** (minima at H Q = 2π n): for islands of another shape and with the refraction model it may be off by a constant of about 1 nm; with one fringe period or less in the range it carries a systematic error of up to 8% (inside its interval). It is not given while no minimum is in the detector's q_z range, while a minimum crosses a module gap, or when the order cannot be told. | Read the trend; compare with XRR, AFM or a cross-section; the summary names αc and where it comes from. |
| **Sputter flags windows where the beam may be off the sample from the intensity alone** (a dip below half of the level of the neighbouring windows within one modulation period, or below 20% within 10 s without a modulation, with its sides down to 80%; section 8.5). The stage position and the range of its translation are not read: a dip that never falls below half of the level is not flagged, however long, nor is a slower drift off the sample; a dip with another cause (a drop of the beam, the shutter) is flagged alike, and the first and last windows are not judged. When the footprint reaches the sample edge, edge scattering or streaks can raise the intensity in the analysed region instead of lowering it: such windows are not flagged. | Compare the flagged windows with the stage log of the beamtime when you have it; their values stay in the Windows table. |
| **Sputter's literature method rests on the deposition rate, t0 and deposition end you give** (GISAXS measures none of them) and on D = 2π/q*. Its percolation interval assumes the rate within ±10% and t0 within ±2 s (±0.5 s when t0 is given) — engineering choices the page does not change — takes the deposition end as exact, and leaves out the systematic error of 2π/q* (more than 40% for D > 10 nm once coalescence starts). Without a deposition end the deposition is taken to last to the end of the scan. PNO, PSBMA and PEO have no published percolation data: the band of all polymers of the metal is used (section 8.4). | Give the rate at the sample position, and t0 and the deposition end when you know them; for other errors run the toolkit's `physical` command with `--rate-error` and `--t0-error`. Read a *not determined* percolation as not determined, and compare with other chambers' thresholds as a comparison, not a prediction. |
| **Sputter's first Predict downloads PyTorch** (about 1.2 GB, needs the internet); the model runs on the CPU. | Without internet, install PyTorch by hand (section 14.7). |
| Windows is the platform GIMaP is developed and tested on. TensorFlow runs on the CPU; a GPU set-up is not configured. | — |

## 14. Troubleshooting, FAQ, version and contact

### 14.1 A file cannot be loaded

The message under the image says why in words: an empty file (still being written?), not a detector
image, a damaged or incomplete NeXus file, no permission. A NeXus/HDF5 file that another program is still
writing cannot be read until it is closed; one left open by a program that ended can be read after
`h5clear -s` on a copy; one compressed with a filter this installation cannot read needs the Python package
`hdf5plugin` (section 5.1, *NeXus and HDF5 files*). Otherwise check that the format is supported:
`.cbf` needs `fabio`, HDF5/NeXus files need `h5py` and the dataset where the reader expects it. To ask
for another file format, use **Help ▸ Report a Problem…** or write to the author (section 14.12).

### 14.2 A `.csv` curve does not open in Fitting

Fitting reads `.dat` and `.txt` curves. For a curve from Analyze use **Send to Fitting**, or open the
`_fit_input.dat` written next to the CSV.

### 14.3 1D Predict Does Not Start

*General V5* checks its model files (`modules/Fitting_1D_Model/Workflow_v5/`) against their checksums
before it runs, and does not start when one is missing or was changed: the message names the file.
*Single RC specialist* does not stop in that case; it goes on with its numerical fit and gives *Fast
model unavailable* as the reason of the solution. The portable version and the Git repository both
contain the files. In a source checkout:

- A checkout last updated before the weights were added does not have them yet: update it (section 2.4).
- A checkout made on Windows before 2026-10-13 holds some of these files with converted line endings,
  also after it was moved to the new `main` with the two plain commands, and `git status` does not show
  it. Run `tools\update_source.ps1`: it writes these files again, also when there is nothing new to
  download.
- A model file that you changed yourself is left alone by the script; `git status` lists it as modified.
  `git checkout -- modules/Fitting_1D_Model/Workflow_v5` puts the files of the repository back (and
  discards your changes in that folder).

The numerical methods of Fitting (*Refine the values in Model*, *Search the ranges, then refine*, *Find the
particle shape (no AI)*) use no model files: the check does not concern them. *Predict parameters (fit1d)*
needs its own kit under `utils/gisaxs1d_gui_kit` (or `GISAXS1D_KIT_HOME`); without it the method says where to
put the folder.

### 14.4 Predict is disabled, or the model is not imported (2D Prediction)

The three states above **Predict** say what is missing: *Input* (a CBF frame or a folder), *Model*
(press **Import model...**) and *Framework* (the framework the module needs is installed). When the model
does not load, read the Activity log: it names the folder that was tried. A folder that is gone, or one
named in `module.yaml` that belongs to another computer, is asked for again by **Import model...**; when
the model is loaded, press **Import model...** again to choose another folder.

### 14.5 SEM Particle Analysis does not extract

- The run button stays disabled until a positive **Image calibration** is set for this file.
- **StarDist (local AI)**: press **Check local resources**. It needs a Python with StarDist and
  TensorFlow (`requirements-sem-stardist.txt`) and a model folder with `config.json` and the weights.
  Nothing is downloaded for you.
- **Existing instance labels**: the labels must be an integer image of the size of the SEM image or of
  the ROI.
- No particle passes the filters: look at the rejected outlines and their reasons, then change the
  threshold, contrast, area limits or ROI.

### 14.6 The AI assistant does not start

The start dialog says whether the chosen brain is ready. Claude Code: install it (or choose the `claude.exe` of
the Claude desktop app, under `%LOCALAPPDATA%\Packages\Claude_…\LocalCache\Roaming\Claude\claude-code`, with
**Browse…** beside **Program**), then **Sign In…** and **Check** in **Tools ▸ Settings… ▸ Assistant**. Claude API: `pip install anthropic`, a key, **Test Connection**.
Other providers: `pip install openai`, a key, a model that supports tool calling, **Test Connection**.
A key saved in GIMaP takes precedence over the environment variable: when the saved one is wrong or
expired, replace it, or **Remove** it to use the variable.

### 14.7 Sputter does not predict, or TensorFlow / PyTorch is missing

**TensorFlow and other packages.** `requirements.txt` asks for TensorFlow `>=2.15,<=2.16` (`tensorflow-intel`
on Windows), which in practice installs a 2.15 release; 2.15.1 is the tested version. For a missing package,
activate the right Python environment and run `pip install -r requirements.txt`.

**Sputter.** The job status under **Predict** says what stopped it:

- *The Sputter model is not in this GIMaP folder: … is missing or incomplete.* The model is part of GIMaP, in
  `utils/sputter_toolkit/` of the program folder, and the message says what each kind of installation does
  (section 2.4). The portable version comes with the model, so its folder is incomplete: install or extract
  the release again. Update a git clone with `Update_GIMaP.bat`; download a folder from **Code ▸ Download
  ZIP** again.
- *No CBF frames in this folder.* Choose the folder that holds the scan's CBF frames, or the scan folder with
  them in one subfolder (section 8.1). Sputter reads CBF frames only (chapter 13).
- *Give the photon energy or the wavelength*, *Give the sample-detector distance and the incidence angle (both
  positive)* and the like: a value of **Sample and geometry** or **More options** is missing or out of range —
  both beam-centre coordinates or neither, a positive time window and rate, an end after t0.
- *The Sputter model stopped with an error (exit code …)*: press **Details**, whose last lines say why. Steps
  that were finished stay in the result folder and are reused by the next **Predict**. When the lines do not
  help, send them with **Help ▸ Report a Problem…**.
- *… has no `structure_physical.csv` (press Predict, or choose a results folder)*: the folder chosen with
  **Open Results Folder…** holds no finished result.
- **Show in Analyze** says *No instrument profile has the Sputter geometry*: enter the distance and λ it names in
  Analyze (Geometry step ▸ **Geometry…**), then show the window again (section 8.6).

**PyTorch.** It is needed by the Sputter workspace only: its first **Predict** downloads it from pypi.org and
installs it into the Python GIMaP runs in (`requirements-sputter.txt`, keeping the installed NumPy). When that
fails, the job status says why and gives the command to run by hand:

- no connection to pypi.org (proxy, SSL, firewall): connect, or set the proxy (`HTTPS_PROXY`);
- no permission to write into the environment: start GIMaP from a folder you can write to (`Start_GIMaP.bat`), or
  install as an administrator. GIMaP never installs into the user's own site-packages: the model does not use
  them;
- the disk is full: about 1.5 GB is needed;
- no PyTorch for this Python: start GIMaP with `Start_GIMaP.bat`, which sets up Python 3.10;
- no pip: GIMaP first adds it with `ensurepip`;
- PyTorch is installed but does not load: on Windows install the Microsoft Visual C++ Redistributable, or install
  PyTorch again over it (`python -m pip install --force-reinstall --no-deps -r requirements-sputter.txt`). When
  GIMaP was closed while pip wrote the packages, the next **Predict** does that itself.

By hand: `python -m pip install -r requirements-sputter.txt` with the Python named under **Predict**. The
environment variable `GIMAP_SPUTTER_NO_INSTALL=1` switches the automatic installation off.

### 14.8 Windows warns about the installer

The installer and the launcher are batch files (`.bat`) and are not signed, so Windows may ask before it
runs a downloaded one, and an institution's antivirus or application control may block it. Only run
files downloaded from the GitHub Releases page of GIMaP, and check them against `SHA256SUMS.txt` of the
release (chapter 12, *The Windows package*). Otherwise run GIMaP from source in a local Python environment
(section 2.2).

### 14.9 An export is empty or fails

- **Empty**: check the q or frame range, the masks, the cut, and that the frame has valid data. In
  Fitting, also the fitting range, the parameter bounds and the scale and background. In 2D Prediction,
  the stack and whether the module fits the data.
- **Fails**: check that the folder exists and can be written; do not export into protected system
  folders. When `gimap_analysis/` next to the data cannot be written (a read-only data share), Export
  asks for another folder. The tools that propose a place beside the data (the XRR CSV, the Format
  Converter, Geometry Calibration's export, a project) need another folder there too.

### 14.10 The window is too large for the screen

GIMaP follows the display scaling of the operating system and fits a 1280 × 720 screen. If text is too
large or too small, use **View ▸ Font Size** or **Tools ▸ Settings… ▸ Appearance**; dense panels scroll
instead of being clipped. The window size and position are restored at the next start.

### 14.11 FAQ

**Is GIMaP production-ready?** No. Version 0.2.0 is a pre-release: treat it as beta scientific software
and check results that matter.

**Can I use my own trained model?** For 2D Prediction, yes, if it can be described by a module
configuration with a supported loader (section 11.3). For SEM Particle Analysis, any local StarDist model
folder. 1D Predict's V5 methods use their bundled model; fit1d offers the *General network* and *Au
nanoparticles on Si (adapted)* networks in its settings. Sputter uses the model that comes with GIMaP in
`utils/sputter_toolkit/`; a new model comes with a GIMaP update.

**Where are the results of 1D Predict saved?** In `AI_Fitting_Output/v5_<date>_<time>_<µs>/` in the
folder GIMaP was started from (for the portable version: the program folder). Fits you save yourself
(**Save**) go where you choose. Chapter 12 lists every output.

**Where are Sputter's results saved?** In `sputter/<scan>-<code>/` of the user data folder, or in the folder
set with **More options ▸ Save results in** (section 8.2; chapter 12). They are kept until you delete them.

**Does anything leave my computer?** Your data leave it only when you use the AI assistant with a remote
brain (Claude Code, the Claude API or another provider's server): then the frame's status and the reduced
curves are sent as numbers, never the detector file, and the small q-map image only when you allow it
(section 10.6). With a model on your own computer or lab server, nothing leaves that network. GIMaP itself
goes online in one case: Sputter's first **Predict** downloads PyTorch from pypi.org with pip (section 14.7);
nothing of your data is sent. **Help ▸ GitHub Repository**, **Check for Updates…**, **Report a Problem…** and
the model library of 2D Prediction open a page in your browser. **Report a Problem…** puts the versions of
GIMaP and of your system (no paths, no data) on the clipboard and opens a new issue on GitHub: nothing is sent
until you paste it there and submit the issue yourself. StarDist, the V5 methods of 1D Predict and 2D
Prediction run on your computer and download nothing.

### 14.12 Version and Contact

This manual describes **GIMaP 0.2.0** (pre-release). **Help ▸ About GIMaP** shows the version of the
program you are running. `CHANGELOG.md` in the program folder lists what changed since 0.0.2 — the pages
that were replaced, where the settings are kept now — the changes not yet released, and the known
limitations of this version.

- **Help ▸ Check for Updates…** opens the releases page on GitHub in your browser (newest first,
  pre-releases included): compare the newest number with the one About shows. The check itself sends no request from GIMaP. Section 2.4 says how
  to update.
- **Help ▸ How to Cite GIMaP…** gives the reference to cite when you publish results obtained with
  GIMaP, read from `CITATION.cff` in the program folder, with **Copy** and **Copy BibTeX**. Name the
  version you used: the export records write it as `software_version` (chapter 12); for results from a
  record without it (one written before 0.2.0, or one of those chapter 13 names) note it from **Help ▸
  About GIMaP**. The reference names a DOI only when `CITATION.cff` gives one.
- **Licence.** GIMaP is released under the MIT licence (`LICENSE` in the program folder). The Windows
  package also contains components under other licences (PyQt5, for one, under the GPL-3.0), and the
  example data keep their own licences: **Help ▸ Third-party Notices** lists them all.
  **Help ▸ About GIMaP** names the licence and the versions of Python, Qt, PyQt and numpy, with **Copy**.

Contact: [yufeng.zhai@desy.de](mailto:yufeng.zhai@desy.de)
