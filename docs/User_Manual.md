# GIMaP User Manual

GIMaP is a desktop program for grazing-incidence X-ray scattering: GISAXS and GIWAXS. It turns
detector frames into calibrated curves (I(q), I(qy), I(χ)), finds peaks, orientation and sizes,
fits particle models to 1D curves, follows in-situ series and compares runs. Every curve keeps a
record of where it came from and with which settings.

This manual describes version 0.2.0, a pre-release: treat it as beta scientific software and check
results that matter. The workspaces (Analyze, Fitting, Compare, Sputter) are meant for daily work;
Sputter runs an analysis package that comes with GIMaP (chapter 8). The Labs pages and 1D Predict are
experimental. **Help ▸ About GIMaP** shows the version you are running.
What changed since 0.0.2, and the known limitations of this version, are in `CHANGELOG.md` in the
program folder (chapter 13 collects the limitations).

This manual says what to do and what you get. How GIMaP computes what it shows — the formulas,
thresholds, error models and their limits — is in a separate reference, Physics and Methods
([`docs/Physics_Manual.md`](Physics_Manual.md), **Help ▸ Physics and Methods**); where a method matters for
what you decide, this manual names its section there.

New to GIMaP? Start with chapter 1: two walk-throughs of about ten minutes each, on public data.

Open **Help ▸ User Manual** (`F1`, which opens it at the chapter of the page shown) for the offline reading page: a fixed, expandable chapter outline on
the left and **Search this document**, which searches both headings and text. Click a chapter or a
search result to jump to it; the outline marks your current section. Press `/` to focus search, `Enter`
to open the first result and `Esc` to clear it. In a narrow window, **Contents** opens the navigation.
The page follows your browser/system light or dark appearance; printing uses light colours.
On Windows and macOS, formulas and commands keep their source spacing and line breaks in fixed-width blocks; long lines scroll within
the block. Inline code keeps its fixed-width formatting too.

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

**3. Run the automatic analysis.** Press **Run Automatic Analysis**. This takes from half a minute to a
few minutes (the model fit is the slow part). The Results step then asks for αi. Type `0.463` in
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
| **Fit of I(qy)** | A table of solutions, most probable first: model, p (the probability), R, h, D (nm) of every population in the order of the radii (`3.17 / 6.15`; `44 / –`: the second population has no paracrystal) and χ²ᵣ; a value at a bound of its range is marked ⚠ and named under the table in a warning (*⚠ Not determined by the curve (at a search bound) — 2: Tail ν; 3: Tail ν*; which bound: its tooltip) | Whether another solution is about as probable; what is at a bound |

The fit is **Predict parameters** (section 6.4): GIMaP's network proposes one or two populations of spheres and
upright cylinders, with or without a paracrystal, each solution is refined by least squares and weighted by its
probability, and the most probable solution is reported. Here it is an upright cylinder with a paracrystal next
to a second, larger upright cylinder (R = 3.17 / 6.15 nm, D = 44 / – nm, χ²ᵣ 1.07, probability 0.39). Two more
solutions are about as probable (0.32 and 0.30), so the Results tab shows a point to check, **Model choice**:
*3 solutions have a probability of at least 0.2 (…): the curve alone does not decide the model.* The probabilities
compare only the compositions that were tried. The cut's broad log-normal sizes and its rise at low q are described
here by two populations. For this sample you know the particles are spheres: fit that model in Fitting (step 5).

![GISAXS results](manual_images/en/08_gisaxs_results.png)
*The Results tab of a GISAXS frame: the cut, the length scale 2π/q\*, the fitted curve and the table of solutions.*

**5. Send to Fitting.** Press **Send to Fitting**. GIMaP writes
`gimap_analysis/galaxi_data_fit_input.dat` (the halves as chosen in the Cuts step, here the mean;
q in Å⁻¹) and opens it in **Fitting ▸ Single analysis**: **Curve** ✓ `598 points fitted`, **Model**
`Sphere · 9 free`, **Fit** *Choose a method, then Fit*. Fitting shows q in nm⁻¹.

The Fit step has chosen **Predict parameters** for this curve, and the line under **Method** says why:
*Recommended for this curve: a GISAXS cut not fitted yet.* Model holds a sphere, the particle you know this sample
has: under **Method**, set **Particles** to **The particles in Model**, then press **Predict parameters**. When it
ends, Model holds the solution and the status line reads *1 solution; the best, Sphere (χ²ᵣ 1.82 over 596 of 598
points; 2 below the network's q window), is in Model.* **Results ▸ Other solutions** lists it (R 4.31 nm, D 39.2 nm;
p shows —, since only this composition was tried), and **Fit details…** gives each value in Model with its 68 %
interval (*R 4.31 [4.31, 6.71] nm*) and what to keep in mind: here R and D lie at the edge of their intervals and
σR/R stopped at a bound, so one Gaussian population of spheres describes this cut only roughly. Press **Refine
These Values** to refine it with least squares and get its errors (χ²ᵣ 1.85 on all 598 points; Fit details says
that the two χ²ᵣ are not of the same points); judge the residuals (section 6.2). The same curve opens with
**Refine in Fitting** in Analyze's Results tab.

Sizes depend on the model and on its size distribution. BornAgain's model uses a log-normal
distribution of radii; Fitting's default and the automatic analysis use a Gaussian by number (choose *Log-normal, by
number* under **Sizes** to compare like with like): compare radii with care.

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
(also **Help ▸ Third-party Notices**). `Update_GIMaP.bat` is there too, but it updates only git
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
SHA-256), Python 3.10 and the packages of `requirements-full.txt` (the core and every optional part, about 2 GB), which takes 3-20 minutes
depending on the connection, then opens GIMaP. Later starts are immediate. When installing the packages fails (for
example for lack of free memory), it tries again one package at a time, which needs much less memory
and takes longer. When an update changes `requirements.txt`, the next
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

`requirements.txt` is the core: Analyze, Geometry Calibration, Fitting with Predict parameters, Series, Batch Export, the
automatic analysis, XRR, Sputter and SEM. The optional parts have their own files: `requirements-tensorflow.txt`
(2D Prediction, Trainset training), `requirements-bornagain.txt` (BornAgain simulations) and
`requirements-ai.txt` (Process with AI); `requirements-full.txt` installs everything. A function whose package is
missing says which file to install.

Start GIMaP from such an environment with `python main.py`. `Start_GIMaP.bat` never uses it: it always
uses its own `.gimap` environment and would set one up beside yours.

**As a package, with a `gimap` command.** In the folder of the checkout, `python -m pip install -e .`
installs GIMaP with the packages of `requirements.txt` and adds the command `gimap`. `gimap` starts GIMaP;
`gimap frame_0001.cbf scan_folder sample.gimap` also opens what it names (frames and folders in Analyze, a
project as a project); `gimap --version` prints the version and `gimap --help` the options. `python main.py`
takes the same arguments. Install it editable (`-e`), as shown: the manuals, icons and example data stay in the
checkout, which a plain `pip install .` does not copy.

The trained network of **Predict parameters** and 1D Predict is in the Git repository (about 13 MB, section 6.4),
so a checkout runs everything, including Predict parameters (chapter 6). On macOS, BornAgain is installed
separately: `README.md` says how.

### 2.3 Optional parts

GIMaP runs without these. Each is needed only for the function named.

| Part | Needed for | How |
|---|---|---|
| An AI account or a local model | **Ask AI…**, **Tools ▸ Process with AI…** | The Python packages come with `requirements-ai.txt` (in `requirements-full.txt` and `Start_GIMaP.bat`); set up the AI in **Tools ▸ Settings… ▸ Assistant** (chapter 10) |
| StarDist | Separating touching particles in **Tools ▸ SEM Particle Analysis…** | A separate Python environment with `requirements-sem-stardist.txt`, and a local model folder (chapter 9) |
| The analysis package | **Analyze** and **Static Frames** of the Sputter workspace | Comes with GIMaP (`utils/sputter_client`); it needs no further Python package. **Choose Folder…** or `GIMAP_SPUTTER_CLIENT` use another one (section 8.1) |

TensorFlow runs on the CPU; a GPU set-up is not configured.

### 2.4 Updating

**Windows portable version.** Get the new version as in section 2.1, the installer or the ZIP. It goes into a
folder of its own; start the new `Start_GIMaP.bat` and delete the old folder when you no longer need
it. `Update_GIMaP.bat` cannot update a portable folder: it has no `.git`, and the script says so and
changes nothing.

**macOS portable version.** Extract the new package into a new folder and open its `Start_GIMaP.command`.
Its first launch prepares the runtime for that package. Settings remain in the user data folder.

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
anything. When it says the list of Python packages changed (`requirements.txt`), there is nothing to do if you start GIMaP with
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

Release targets are Windows x64 and macOS 15 or later, with separate Apple Silicon and Intel packages.
The release page identifies the systems verified for each package. A 1280 × 720 screen is enough (the
smallest window is 1024 × 640); a 1080p or larger screen is more comfortable for the detector image
beside the curves. GIMaP follows the display scaling of the operating system (for example 125 % or
150 %).

### macOS portable package

Choose the package for **Apple Silicon (arm64)** or **Intel (x86_64)**. Extract the ZIP, then double-click
`Start_GIMaP.command`. The first launch verifies and unpacks its Python runtime into
`~/Library/Application Support/GIMaP/runtimes/`; the Terminal window shows preparation and launch errors.
Later launches reuse that runtime. No Python installation is needed. Keep the launcher, `app/` and
`runtime.tar.gz` together. The package is currently unsigned: use macOS's Open action for a download you
trust; do not disable system protection. Target systems are macOS 15 or later and Windows x64; the release
page lists the exact systems and architectures verified for each package. A source checkout running on a
developer's computer does not replace package acceptance testing.

For source installations on a Mac, use Python 3.10 and `python -m pip install -r requirements-release.txt`. It selects the compatible official SciPy build; version numbers alone do not identify the binary build.

## 3. The window

### 3.1 The parts of the window

![The Analyze workspace](manual_images/en/02_analyze_overview.png)
*Analyze: the sidebar, the command bar, the steps on the left with the frame list of the Data step (Choose
Frames… below it), the image in the middle, the curves on the right.*

- **The sidebar** (left) holds the pages. *Workspaces*: **Start**, **Analyze**, **Fitting**,
  **Compare**, **Sputter**. *Labs*: **2D Prediction**, the experimental machine-learning
  tools. **« Collapse** at its foot (or `Ctrl+B`) reduces it to icons. `Ctrl+1` … `Ctrl+6` switch pages
  in this order: Start, Analyze, Fitting, Compare, Sputter, 2D Prediction (so Sputter
  is `Ctrl+5`).
- **The menu bar**: **File, Edit, View, Analyze, Fitting, Tools, Help** (section 3.6). The tool windows
  open from **Tools**; **Analyze** and **Fitting** hold the commands of those two pages.
- **The page.** Analyze and Fitting share one layout: a command bar at the top, the steps of the work
  on the left, the plots on the right.
- **Messages** appear as short notes at the bottom right. A note about a written file has an
  **Open Folder** button. Each page also has a status line at its foot.

The window title names the frame shown, and the project when one is open. A `*` after the title says that
the work has changes that are not saved (section 3.8).

**Starting.** While GIMaP starts, a small window shows its version and what it is loading; the main window
follows a few seconds later. Compare and Sputter are built when you first open them. A second start while
GIMaP is running asks first: **Switch to It** brings the running window to the front (and opens there what the
second start was given), **Start Anyway (Read-only Settings)** opens a second window that saves neither
settings nor session, so the two never overwrite each other's files.

**Small screens.** When the window is smaller than a page needs (below about 1200 × 690 for Analyze), the page
keeps its size and scroll bars appear, instead of plots and tables being cut off.

**Dropping files.** Detector frames, folders and `.gimap` projects can be dropped anywhere on the
window. A curve file (`.dat`, `.txt`) dropped while Fitting is shown opens there.

**Closing.** When the work has unsaved changes, closing asks first: **Save** (the project, or a new one),
**Don't Save** or **Cancel**. Closing GIMaP while something still runs — Batch Export, a Series map, the automatic
analysis, an AI analysis, a fit, an In-situ series, a 2D prediction, an SEM Particle Analysis, a
Sputter analysis — asks first and lists what runs. **Yes** stops them and quits; **Cancel** keeps working. An SEM analysis
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
| View | *Workspaces* and *Labs*: the pages (`Ctrl+1`…`Ctrl+6`), Collapse Sidebar (`Ctrl+B`), Full Screen (`F11`), Theme ▸ Light / Dark, Font Size ▸ Larger (`Ctrl++`) / Smaller (`Ctrl+-`) / Reset (`Ctrl+0`), Language ▸ English / 中文 |
| Analyze | Run Automatic Analysis, Stop Automatic Analysis, Repeat Analysis (`F5`), Absolute Scale from a Standard…, Export Curves… (`Ctrl+E`), Batch Export… (`Ctrl+Shift+E`), Batch Export a Folder…, Send to Fitting, Send Both Cuts to Fitting, Send Series to Fitting…, Send Series to Compare, Set-up Files ▸ Save Settings… / Load Settings… / Save Cuts… / Load Cuts… / Save Masks… / Load Masks…, Figures ▸ Save Image… / Save q-Map Data… / Save View Data…, Clear Frame List |
| Fitting | Open Curve…, Import Curves…, Fit a Folder of Curves…, Fit (`Ctrl+Return`), Stop Fit, Fit Several Curves Together…, Load Model…, Save Model…, Save Data and Fit…, Save Plot…, Batch / In-situ Series ▸ Start / Stop, 1D Predict — Fit Many Curves… |
| Tools | Geometry Calibration… (`Ctrl+Shift+G`), Format Converter… (`Ctrl+Shift+C`), Convert Current File…, XRR Series Extractor… (`Ctrl+Shift+R`), SEM Particle Analysis…, 1D Predict — Fit Many Curves…, Process with AI… (`Ctrl+Shift+L`), Settings… (`Ctrl+,`) |
| Help | User Manual (`F1`), Physics and Methods, Keyboard Shortcuts…, Example Data ▸ GISAXS — Ag Nanoparticles (GALAXI)… / SAXS — AgBh Calibrant (pyFAI)… / GIWAXS — MAPbI₃ Film (DESY P08)… / GIWAXS — PbTe, Tilted Detector (Xeuss)…, Report a Problem…, Open Log Folder, Open User Data Folder, GitHub Repository, Check for Updates…, How to Cite GIMaP…, Third-party Notices, About GIMaP |

![The Analyze menu open over the Analyze workspace](manual_images/en/23_analyze_menu.png)
*The Analyze menu with one frame open in Analyze. Entries that cannot run now are greyed out; their tooltip says
why.*

![The Help menu](manual_images/en/24_help_menu.png)
*The Help menu: the two manuals, the keyboard shortcuts, the example data (a submenu), where to report a problem,
the folders GIMaP keeps, and the links to GitHub, the citation, the notices and the version.*

- While Fitting's Single analysis is shown, **File ▸ Open Data…** opens a curve and says so
  (*Open Curve…*).
- **Edit ▸ Undo** and **Redo** act on the page shown: Analyze, or Fitting's Single analysis. The
  entry names what it would change (for example *Undo: cut regions*, *Undo: the fit (Refine)*). On the other pages
  they are greyed out.
- Every entry of **Analyze** and **Fitting** first shows its page (in Fitting the Single analysis, or
  the In-situ series for Fit a Folder of Curves… and Batch / In-situ Series ▸ Start / Stop), then does what
  the page's own button does. An entry that cannot run now is greyed out, and its tooltip says why
  (for example *Open a frame first.*).
- The keys shown next to Undo, Redo, Repeat Analysis, Export Curves…, Batch Export… and Fit belong to
  the page: they act while that page has the keyboard focus (section 3.7). The menu entries work from
  any page.
- *Labs Parameters* are the settings of 2D Prediction. The file also holds Fitting's
  model (the particles and global parameters of the Single analysis): **Load…** replaces Fitting's model
  with the one in the file and keeps it as the stored model, without asking. Save a project first if you
  need the model you have. Analyze keeps its own set-up in settings files (**Analyze ▸ Set-up Files ▸ Save Settings… / Load Settings…**, also in the
  page's **Export** menu and used by Batch Export). A project keeps everything of one sample (section 3.8).
- **Tools ▸ 1D Predict — Fit Many Curves…** fits a list of curve files with the experimental neural
  model (chapter 6). The other tool windows are described in chapters 4 and 9.
- **Help ▸ User Manual** opens this manual in your web browser; in the Chinese interface it opens the
  Chinese edition. Its fixed chapter outline and text search work offline. **Physics and Methods**
  (how GIMaP computes what it shows) and **Third-party
  Notices** open the same way; a link from one of these documents to another opens the other one at
  that section. **Keyboard Shortcuts…** lists every key of the menus and of the pages
  in a table you can copy (`Ctrl+C`).
- **Help ▸ Example Data** copies one of the public examples that come with GIMaP to a folder you
  choose (by default `Documents\GIMaP examples\<example>`) and opens its frame in Analyze (section 1.1).
- **Help ▸ Report a Problem…** copies the versions of GIMaP, the operating system, Python, Qt, PyQt
  and numpy (no paths, no data) to the clipboard and opens a new issue on GitHub in your browser:
  paste them there. **Check for Updates…** opens the releases page on GitHub in your browser (newest
  first, pre-releases included); GIMaP itself makes no network request.

![The offline help reader](manual_images/en/25_help_reader.png)
*The help reader: expandable chapters and text search stay beside the document while you read.*

### 3.7 Keyboard shortcuts

The keys of the File, View and Tools menus work on every page. The keys below belong to a page or a
window: they act while it has the keyboard focus. After switching pages with `Ctrl+1` … `Ctrl+6` or the
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

### 3.8 Projects

A project (`.gimap`, a JSON file) reopens a sample as it was left:

- the frames and the whole set-up of Analyze (mode, profile, masks and drawn regions, corrections,
  cuts), the frames chosen for import (section 5.1) and whether one cut is held for a series;
- the curve, halves, fitting range, left-out points, model and selected method of Fitting, and its results: the
  last fit (values, errors, χ²ᵣ, values at a bound, correlations), every solution of Predict parameters (probability,
  intervals, flags) and where a fit started from. They are shown again as they were; when the curve's file changed
  since the project was saved (its SHA-256), they are shown as stale (*Changed after the fit: Fit again*) and the
  project's notes say so;
- the folder and choices of the In-situ series with its fitted frames (each frame's values, errors, χ²ᵣ and points,
  shown again once the curves are listed; a frame whose file is no longer listed is left out, and the status line
  says how many), and the series of Compare;
- the Sputter form (the input bar with the rate's unit, the geometry, the options of Analyze and of Static
  Frames), scan/mask/output paths and the reference to the result folder shown (the result is not copied). A project
  of an older version keeps its geometry; its result of the earlier method (Predict) is not opened.

The 2D Prediction settings go, together with Fitting's model, into their own files with
**File ▸ Labs Parameters ▸ Save As…**. Older projects without Sputter reset that workspace to its empty form.

**File ▸ Save Project** (`Ctrl+S`) writes it; the first time it asks where (next to the first frame
by default). **Open Project…**, the Start page's **Open Project…**, **Open Recent** (projects first)
or dropping a `.gimap` file on the window opens it.

- Data are referenced by path, not copied. Compare's embedded arrays are kept in a second file,
  `<name>.<generation>.compare.npz`, next to the project. Move or copy the `.gimap` and the file named by its
  `compare.arrays_file` together. Older `<name>.compare.npz` projects still open.
- Saving writes new arrays and a temporary project file before atomically replacing the `.gimap`. If writing
  or replacement fails, the previous project and its arrays remain usable. An old array file may remain if
  another program holds it open; the project names the array file it actually uses.
- While Compare is importing or restoring series, **Save Project** asks you to wait or cancel that import
  first; it does not save an incomplete list. Restored Compare data are read in the background, in their saved
  order. Any read errors appear in Compare's status line and notifications.
- When a file has moved, the project still opens; a note
  says how many files are missing and names the first ones, and **Details…** lists them all.
- Analyze restores the saved source order, displayed source and original frame number. The source order
  also determines which adjacent single-frame files contribute to **Sum** and their order in a series.
  If that source is missing, a note explains the fallback to the first imported frame of another source.
  If a NeXus chunk is now represented by its master, its frame number and import selection are mapped into that master.
- A project saved with a method removed in 0.2.0 (*Predict parameters (fit1d)*, *Find the particle shape*) opens
  with the method the page chooses for its curve (section 6.2); the options of those methods are not read.
- Opening a project replaces what the window holds: the frames of Analyze, the curve and the In-situ
  series of Fitting and the series of Compare of the sample before are closed. Save them first if you need them.
- After **Clear** in Analyze, what you open next is no longer that project: **Save Project** asks
  where to save instead of overwriting it.
- Work that differs from what was last saved or opened puts a `*` after the window title. Opening another
  project or closing the window then asks first: **Save**, **Don't Save** or **Cancel**. Changing back what you
  changed removes the `*` again; looking at another frame of the list is no change.
- A project that cannot be opened is taken off **Open Recent**, and the message says so.
- A project cannot be opened while a job runs (an In-situ series, Batch Export …): stop it first.
- Opening a project replaces the fits of the In-situ series. When frames fitted there are not saved yet, opening a
  project asks first: **Save** (the table of every frame), **Discard** or **Cancel** (nothing changes).

### 3.9 Memory between sessions

- **Recent data**: every file or folder opened (dialog, drag and drop, Start page, Batch Export) is
  listed in **File ▸ Open Recent** and on the Start page, newest first. At most 8 are kept; files
  that no longer exist are left out; **Clear the List** empties it. Recent projects are kept apart
  (the last 6), listed first, and stay after **Clear the List**.
- **Last set-up**: when GIMaP closes after a frame was analysed, the Analyze set-up (mode, profile,
  masks, corrections, cut regions …) is kept in `last_analyze_setup.json` in the user data folder.
  With the first frame of the next session a note offers **Use It**. It is never applied without
  asking, since a new beamtime may need other settings, and it is not offered when you open a project:
  the project's own set-up wins. A file that could not be opened is taken off Recent (a note says so).
- **Autosave and recovery**: changed work is saved every 2 minutes and after each long job (Batch Export, the
  automatic analysis, a fit, an In-situ series …) to `autosave/session.gimap` in the user data folder, never
  while a job runs. When GIMaP was not closed properly (a crash, a power cut, a window ended in the Task
  Manager), the next start asks **GIMaP was not closed properly. Restore the work autosaved at …?**:
  **Restore** opens that work (with a `*`: save it to keep it; **Save Project** writes to the project it came
  from), **Discard** deletes it. A normal close deletes the autosave. The Series map comes back with the autosave and
  with a project (also one built online while watching), up to 4 million values (x and intensity of every row); a
  larger map is not kept, and a note says to build it again.
- **Sputter**: the values of each scan folder's last Analyze (geometry, metal, mask, the substrate, the rate with
  its unit and whether it is a QCM reading, the mode, the start and duration of the deposition and the film
  thickness), the values used last (another folder does not take over the mask, the sample roll or a rate), the
  geometry presets, the folder of the analysis package and the result shown last (opened again at the next start).
  What an older version remembered for its earlier method (Predict) is not read.
- The window size and position, the positions of the dividers between panels (splitters), the theme, the
  font size and the language are restored at the next start.

### 3.10 When something goes wrong

An unexpected error does not close GIMaP. A window says what happened: **Show Details** shows the
technical report, **Copy Details** copies it for a bug report, and the work goes on. Every error is
written to `logs/errors.log` in the user data folder. The same folder keeps `gimap.log`, what each session did
(its version, warnings, failed saves; also when GIMaP runs without a console), in files of at most 1 MB with the
three before, and `crash.log`, where the Python stack is written when GIMaP is ended by a crash of the system
libraries. **Open Log Folder** in that window, or **Help ▸ Open Log Folder** at any time, opens the folder; when
nothing has been logged yet, the menu entry says so. To report a problem, **Help ▸ Report a Problem…** (section 3.6) copies the
versions to paste into the report.

Files that cannot be read say why in words: empty (still being written?), not a detector image, a
damaged or incomplete NeXus file, a NeXus/HDF5 file another program is still writing, no permission.
Chapter 14 has more.

### 3.11 Settings and the user data folder

**Tools ▸ Settings…** (`Ctrl+,`) has four pages. Changes apply immediately (the font size a moment after the
last arrow click, or at once with Enter); only the keys of the AI are stored with their own **Save** button.

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

- **One GIMaP per folder.** A running GIMaP holds `gimap.lock` there; a second start asks first (section 3.1).
  A lock left by a GIMaP that crashed is taken over without a question.
- **Settings of older versions.** The first time the default folder is opened, GIMaP imports once the settings
  an older version kept in the program folder (`config/user_parameters.json`, `config/user_settings.json`,
  `config/instrument_profiles.json`): settings only, never a session (the last Fitting curve, the recent and
  last folders stay behind). A folder named by `GIMAP_HOME` imports nothing.

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

**A `.poni` beside the frames comes first.** When the frame's own folder holds a pyFAI `.poni` named
like the frame (or only one `.poni`), and the file fits the frame (its frame size and pixel size, when
it states them, and a wavelength), its geometry is used ahead of a profile matched by name and size:
such a profile may come from another experiment with the same detector. **Geometry** then says
*galaxi_bornagain.poni beside the frame*. A beam centre or geometry you save while that file is in use
names it, and from then on your saved profile is used instead.

**A profile picked for another detector is not used.** A profile chosen in **Instrument profile** (or
kept from a project) that is for another frame size is set aside for frames of another size: they take
the `.poni` beside them or the profile that matches them, and the banner says which profile was not used
and why. **Use “…” Anyway** in the banner uses it for frames of that size all the same.

The step **Geometry** shows the geometry in use (`811.8 mm · λ 0.6888 Å · αi 0°`). Its field
**Instrument profile** is set to **Automatic (match detector)**; choose a profile there to use it
instead.

**αi** is part of the profile, but the command bar has its own field. It shows the profile's αi
(*αi from profile* while no frame is shown). A value you type is shown in orange and is used for
every frame; it is also remembered for the next session, and a note at the first frame then says
so. Right-click the field ▸ **Back to Profile αi** to drop it; **Clear** in the step Data does not.
The remembered value belongs to the instrument profile it was set with. When the first frame of the
next session has another profile, or its detector has none yet, the value is not used: the field
shows the profile's αi, a note says so and offers the old value in one step (**Use 0.075°**), and
Run Automatic Analysis asks for αi and names the old value as a suggestion. A value from the command
bar is never written into an instrument profile, so **Back to Profile αi** always returns the
profile's own value.

- An αi answered in the Results step (**Run Again with These Answers**) is set the same way, in orange,
  when the run keeps the profile the detector has (section 1.2).
- When the run applies a calibration instead — one named in the beamtime notes, or one it found for a
  detector without a profile — the answered αi is saved with it in the instrument profile (section 1.3).

How a profile is chosen and what each override replaces is in
[Physics and Methods, section 1.4](Physics_Manual.md#14-where-the-geometry-of-a-frame-comes-from).

**The beam centre in file headers** is not used by default: the calibrated profile centre is kept,
because many beamlines never update the header. **Tools ▸ Settings… ▸ Analyze ▸ Use the beam centre
written in the file header** changes that (section 3.11). **Beam centre ▸ Use File Header Centre**
takes the header centre as the centre of this session (section 4.5).

### 4.2 Three ways to a geometry

When no profile matches, a banner over the image says so and offers **Find Calibration Automatically**,
**Load .poni…**, **Enter Geometry…** and **Calibrate…**. The step **Geometry** has **Find Calibration
Automatically**, **Calibrate…**, **Load .poni…** and **Export .poni…** too, and **Geometry…** in place of
**Enter Geometry…**.

**Load .poni…** reads a pyFAI `.poni` you choose. A file that states another frame size, or a pixel size
more than 1 % from the frame header's, is named with the reason and you are asked before it is used. A file
with a wavelength becomes the instrument profile of this detector (with its tilt), is picked in
**Instrument profile** and the frame is analysed again. A file without a wavelength opens **Detector
geometry** filled with it, the wavelength *Required*. **Export .poni…** writes the geometry in use (or, in
the dialog, its fields) as a pyFAI `.poni` (version 2.1: pixel sizes, frame size, distance, PONI, rot1–rot3,
wavelength) for pyFAI, GIXSGUI and other programs; αi is no part of a `.poni`.

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
as the search can tell; only its folder puts it after `galaxi_bornagain.poni`. A calibration file the
search found in another folder than the frame's (or one below it) is listed with that folder
(*galaxi_bornagain.poni (sibling folder gisaxs_galaxi)*). It is skipped, with the reason among the
decisions, when the detector shape it states (`max_shape` of a `.poni`) is not the frame's. When it was
made at another energy than the frame's (more than 0.2 % apart), its distance and beam centre are used
with the frame's energy, as for any calibration, and a decision of its own says so (**Calibration**:
*… made at another energy than the frame's*), so that it is never adopted without a word. A file you
named, or one beside the frame, is not second-guessed this way. The result is saved as the instrument profile of the detector.

When the frame's own header states a detector distance that differs from the calibration in use by
more than 5 %, the Data step says so (*⚠ The file header gives a detector distance of 370 mm, the
calibration in use 147.6 mm (151 % apart).*) and the automatic analysis lists it under
*Needs attention* as **detector distance**. A header often holds a nominal distance, and then the
calibration stands; but this is also what the calibration of another experiment looks like.

**A tilted detector.** A `.poni` with a tilt (pyFAI's `Rot1`, `Rot2`, `Rot3`) gives a tilted geometry:
every pixel is placed where pyFAI places it, so q, χ, the solid angle and the polarisation factor are those of the
tilted detector ([Physics and Methods, section 3.7](Physics_Manual.md#37-the-detector-tilt-of-a-poni)). The step
**Geometry** and the summary say *detector tilt 6.06°*, and the automatic analysis lists it under *Decisions*. A
beam centre moved later (the GISAXS symmetry axis, or by hand) keeps the tilt. The Xeuss example of
**Help ▸ Example Data** is such a case: leaving its 6.06° tilt out, as GIMaP 0.1 did, put |q| off by up to 6.3 %.
Geometry Calibration fits the tilt (rot1, rot2; section 4.3), and the **Geometry** dialog has fields for
rot1, rot2 and rot3.

**Check rot3 on a GISAXS frame.** Powder rings constrain rot1 and rot2. To check the rotation about the beam, open **Geometry ▸ Check rot3 ▸ Measure Specular Rod**. Click two widely separated points on the ridge. Both qy values should be close to zero within the precision of the picks. To use the angle shown, enter it under **Geometry… ▸ Rotation about the beam rot3** and choose **Save Profile**. Repeat the check. If the direction agrees but both qy values remain offset, check the direct-beam centre. Under **Measure Horizon or Yoneda Band**, pick two points on the same feature. Their exit angles should agree. On the horizon both should be zero.

A `.poni` can specify detector orientation 1–4. GIMaP uses the stored row and column directions in the geometry. Load the frame in the same storage order as the calibration.

The search may pick the calibration of another experiment: one in a folder beside or above the
frame's when the frame's own folder holds none, or, among several in one folder that was copied in
one go, the wrong one (the file times then say nothing). Check the file named and the distance shown in
**Geometry**, or name the right one in the notes (a calibration named there is tried before the
profile and the search), or use one of the two other ways. A wrong profile is removed with
**Geometry…** ▸ **Delete profile**.

**Enter Geometry…** opens the dialog **Detector geometry**: **Profile name**,
**Sample–detector distance**, **Pixel size x** / **y**, **Wavelength** and **Energy** (one value in two
units: typing one sets the other), **Detector tilt rot1**, **Detector tilt rot2**, **Rotation about the beam
rot3** (pyFAI's rotations in degrees; 0 is a detector perpendicular to the beam), **Beam centre x** / **y**
(section 4.4), **Centre step**, **Incidence angle αi**, and **Load .poni…** / **Export .poni…**.

- The line under **Energy** says where the wavelength comes from: *Wavelength from the file header.*,
  *Wavelength from LaB6.poni.*, *Wavelength of profile “…”.* or *Wavelength entered by you.* When the file
  states no wavelength or energy, both fields read *Required* and the line says so in orange: no value is
  assumed, and **Save Profile** and **Export .poni…** wait until you type one.

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
*Geometry…: the geometry in use (here from the LaB₆ `.poni` beside the P08 frame), with the energy, the line saying
where the wavelength comes from, the detector tilt rot1–rot3, Load .poni… / Export .poni…, Live Preview, Apply
Centre and Save Profile.*

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
   is filled in automatically. The line under **Energy** says where it came from (*From the file header.*,
   *From the companion NXS file.*, *Entered by you.*). When the file states none, the field reads
   *Required*, the line says *Not in the file: enter the X-ray energy* in orange, and **Auto Calibration**
   waits until you type it — no energy is assumed (an earlier version silently used 12 keV). For the next
   image without an energy, the value you typed is kept and the line asks you to check it. The calibration
   record and the instrument profile say where the energy came from.
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
   with the bounds in **Advanced configuration**). The range is a limit, not a starting point: no solution
   outside it is shown, also not after the fit of centre and distance. When none lies in it, the calibration
   stops with *No solution lies within the distance range* — widen the range, or check the energy and the
   pixel size.
   An image of the standard taken at grazing incidence (AgBH on a wafer: arcs above the sample horizon, the
   specular rod through the beam, the Yoneda band) is recognised. The rod is left out of the ring search
   and the fit, a horizontal band above the beam is not taken for a ring, and for AgBH every order between the
   matched ones must be there — so the Yoneda band is no longer taken for the first ring of a solution at
   half the distance. The warnings of the solution say when this applied.
6. Under **Advanced configuration**, enter a measured roll in **Fixed rot3** (degrees, default 0). The ring fit keeps that value.
   **Detector tilt: Refine rot1, rot2** (on by default) fits the tilt of the detector with the centre and the
   distance when three or more rings are matched ([Physics and Methods, section 3.8](Physics_Manual.md#38-detector-tilt-from-the-rings-of-the-standard)).
   Click **Auto Calibration**. The calculation runs in the background; **Cancel** stops it after the
   current step.
7. Review the result: beam centre, distance, **Detector tilt** (rot1 and rot2, *0.372°, 0.433°*, or *not
   refined*; its tooltip gives the 1σ and the RMS of the ring points with and without the tilt), matched rings, RMS residual,
   confidence and warning of the **Selected solution** (a warning is marked ⚠; its tooltip says why it matters), and the
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

**Export Calibration…** writes the result as a `.json` file, or as a pyFAI `.poni` (choose *pyFAI geometry
(*.poni)* in the file dialog; the tilt is written as Rot1, Rot2, Rot3). It proposes
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

**Known limitation: a wrong estimated distance can put a wrong solution first.** Seen on a silver
behenate image whose path does not name the standard.

- A wrong estimated distance can mislead, also with the right standard: the search prefers solutions
  near the value in **Estimated distance**. With silver behenate chosen and 2000 mm typed in, a
  solution near 1790 mm came first, although the real distance is about 1630 mm; with **Auto Detect**
  and 2000 mm, a LaB₆ solution at 120 mm came first. With the field left on *Optional*, the right
  solution came first, with the standard chosen and with **Auto Detect** alike. Choose the standard
  yourself when you know it. An approximate distance helps: with 1500 to 1800 mm typed in, the silver
  behenate solution came first on this image.
- Before this version, **Auto Detect** without an estimated distance put a LaB₆ solution at a much
  shorter distance first on this image, and the silver behenate solution last in the list of
  candidates. A harmonic standard is now ranked by how complete its orders are, and silver behenate
  comes first.

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

The retired Trainset research code keeps its legacy coordinate convention; it is not a GIMaP workspace. The
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
keeps it for good.

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
| Command bar | **Open…** (its arrow: **Open Folder…**), ↶ ↷ (Undo / Redo), ‹ › (previous / next frame, when more than one frame is listed), the mode **Auto / GISAXS / GIWAXS / SAXS**, αi, **Run Automatic Analysis** (with **Stop** while it runs), **Ask AI…**, **Export** ▾, **Batch Export…** (when more than one frame is listed), **Send to Fitting** ▾ |
| Steps (left) | **Data**, **Geometry**, **Mask & corrections**, **Cuts**, **Results**, **Export**. Each shows its state (✓, !, × or …, as in 3.3) and one line of what it found; click a step for its controls. Rarely used options are folded away. |
| Image (middle) | **Detector**, **q map** or **Cake** (GIWAXS), **Log**, the colour map, **Levels**, **Marks**, Zoom, **Fit view**, **Sources**, **Save** ▾, **Panels** ▾; the values under the cursor below the image, on two lines: the pixel and its value, q with q_y or q_∥ and q_z, then 2θ, χ, αf (grazing incidence) and d = 2π/q (on the q map and the cake the point is given in q). It is the widest of the three panels; **Panels ▾** hides the steps or the curves and results, so the image has the whole room, and shows them again |
| Tabs (right) | **Curves** (an upper and a lower plot), **Results** (what the automatic analysis or the AI found, with its evidence), **Series** (every frame as intensity against frame and q) |
| Status line | What is happening, a progress bar and **Cancel**. A note at the bottom right says when a file was written, with **Open Folder**. |

On a narrow window the command bar shortens its long buttons (section 3.3) and, if needed, moves the
actions to a second row.

The mode: **Auto** reduces a frame as GIWAXS when the detector reaches beyond a scattering angle
2θ = 20°, otherwise as GISAXS. Auto decides again for every frame. Choose **GISAXS** or **GIWAXS**
to force it; the status line warns when a forced mode does not look right for the frame.
**SAXS** (SAXS/WAXS in transmission) reduces a frame taken in transmission: I(q) over the whole ring and I(χ) from −180° to 180°
(χ 0° to the right of the beam, 90° up), the cut regions, the q map in q_y–q_z and the cake; there is no
horizon, no Yoneda cut and no missing wedge. Auto takes it when no αi is known and the file name names a
standard (AgBh, LaB₆, CeO₂), or the frame is small-angle and shows at least three sharp rings about the beam
centre, each through at least 80 % of the directions the detector covers; the status line says so. An αi
above 0 always means grazing incidence. Run Automatic Analysis treats such a frame as before: it writes the
cuts, fits no particle model, and says so under **technique** (give αi or the technique if the frame is a
grazing-incidence one after all).

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
   banner above the image offers **Find Calibration Automatically**, **Load .poni…**, **Enter Geometry…**
   and **Calibrate…** (section 4.2).

**Frames listed already.** When you open or drop files or a folder while frames are listed, GIMaP asks
whether to **Replace** them or **Add to List**. Frames added must come from the same detector: when the
first new frame has another size than the frame shown, nothing is added and the status line says why
(*These frames are 1043×981 …; the frames listed are 2048×2048 …*). A list never mixes two detectors, so
Batch Export never mixes two geometries.

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
  even when adding frames does not change the file size or timestamp. Recently written files and files
  held by their writer are inspected again; stable files keep their cached layout. Each look takes the frames it already held at the look before, so the newest frame,
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

**Online mode** — the arrow of **Watch**:

- **Pause Watching** stops the looks; when you untick it, everything written meanwhile is listed and reduced.
  Stopping the watch also ends a pause.
- **Settle Time**: **Two Looks (2 s)** (the default: a new file is read once it is the same at two looks), **5 s**,
  **10 s** or **30 s** — for a detector that pauses while it writes a file, so that a frame is never read
  half-written.
- **Add New Frames to the Series Map** (on by default): every complete new frame is reduced with the set-up on
  screen and its curve appended to the Series map (section 5.6), its traces and its timeline — also when no map was
  built and nothing is exported. A map that the watch begins starts with the frame on screen when watching began. Only the curves are kept, never the frames: 1000 frames of 200 points take about
  3 MB of curves.
- **Record Every Frame in the Results Database**: the curve of every frame appended to the map also goes into
  `gimap_analysis/gimap_series.sqlite` next to the data (about once a second), with its frame, file, time and the
  settings of the map, and so does every skipped file. Compare reads it (section 7.2), so a run can be compared
  again later without its frames.
- **Skipped Files…** opens the log of the files the watch could not take, with the time and the reason: held by
  the program writing it, a failed reduction, a frame without the curve of the map.
- **Fit New Frames in Fitting…** exports the frame on screen, switches **Export New Frames While Watching** on
  and opens `gimap_analysis/` in **Fitting ▸ In-situ series** with **Watch for new curves** ticked, and starts
  there once the curves are listed: every curve written so far and every new one is fitted from the result of the
  one before (section 6.5). While the model in Single analysis is still the starting guess, it does not start: the
  status line says to set the model up on a frame and press **Start**.

These choices are remembered between sessions. On a desktop computer, a watched run of 1000 frames of 20,000
pixels took about 60 ms per frame on average to be listed, reduced, appended and recorded
(`tools/benchmarks/insitu_online.py`).

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

*On this synthetic frame, the red ring previews a four-pixel expansion of the drawn annulus. The pixel count beside Apply mask edit gives the size of the proposed user mask.*

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

**User mask**

Choose a tool under **User mask**. **Exclude pixels** adds the selected area to the mask.
**Include pixels** restores that area wherever the detector and other corrections allow it.

| To select | What to do |
|---|---|
| Rectangle or ellipse | Press **Draw** and click opposite corners of its bounding box. |
| Polygon | Press **Draw** and click the vertices. Double-click or press `Enter` to close, `Backspace` to remove a vertex, or `Esc` to cancel. |
| Circle or annulus | Press **Draw** and click the centre. For a circle, click the rim. For an annulus, click the inner rim, then the outer rim. |
| Freehand stroke | Choose **Brush** or **Eraser**, set **Radius** in pixels, press **Draw** and drag. The eraser restores pixels from earlier edits. |
| Ring about the direct beam | Choose **Beam-centred q annulus** and enter **Inner q** and **Outer q** in Å⁻¹. This uses the current detector geometry. |

Drawing previews the total number of pixels in the user mask. Press **Apply mask edit** to use it
in all curves. Excluding a rectangle or polygon applies it as soon as you finish drawing.

For a q annulus or a value rule, press **Preview mask**, check the total and press **Apply mask edit**.
**Above threshold** and **Below threshold** act on raw values. For a relative threshold, tick
**Multiple of the median** and enter the multiplier. The median uses the remaining valid pixels.
**Saturated pixels** includes the threshold itself: enter the saturation level for the exposure.
**Negative pixels** selects values below zero. With frame summing on, these rules compare the sum.

**Grow mask** and **Shrink mask** expand and erode the current user mask by a circular radius in pixels.
**Invert mask** swaps masked and unmasked pixels. Edits run in list order. **Remove Selected** or
`Delete` removes one edit. **Clear Masks** leaves zero pixels in the user mask. Edit ▸ **Undo** and
**Redo** work for drawing, rules, loading, removal and clearing.

**Save…** can write a GIMaP `.json` recipe that repeats the edits on each frame. Keep any mask images
the recipe references. To save the resulting user mask for this frame, choose EDF, TIFF, NumPy
(`.npy`) or Fit2D (`.msk`). The image stores 0 for unmasked and 1 for masked. Detector gaps and
automatic bad-pixel exclusions are tracked separately. A record beside it (for example
`masks.edf.json`) gives the frame, geometry, settings and mask sources. **Load…** accepts these
recipes, older rectangle/polygon JSON files and mask images of the same shape as the frame.
As in pyFAI, any non-zero image pixel is masked. See Physics and Methods, section 4.5.

**Fill gaps from the mirror side (GIWAXS)** is off by default. Use it for a sample whose scattering
is symmetric in ±q∥. Each gap pixel takes its intensity from the opposite in-plane angle at the same
exit angle. On a tilted detector this can be in another row. All pixels contributing to the
interpolation must have measured values. The missing wedge stays empty. The export records the
number of filled pixels and uses pixel scatter for the error bars (section 5.4).

**Corrections** (folded): a **dark frame** (**Choose…** next to *No dark frame*, with **Dark scale** and, for NeXus,
**Dark frame**; off by default), subtracted from every frame before the background; with it the error bars come
from the scatter of the pixels ([Physics and Methods 4.11](Physics_Manual.md#411-dark-frame)). Then a background
frame (**Choose…**, with **Scale** and, for NeXus, **Frame**),
a valid raw intensity range (**Min** / **Max**), and the **Gap guard** (default 3 px; 0–20 px,
remembered), which also leaves out pixels this close to gaps and bad pixels. **Refraction in the film, αc**
makes q_z the one inside a film of that critical angle (q∥ does not change; pixels below αc are left out of the q
maps and of the cuts in q_z). Off by default; Physics and Methods 2.6.

**Flat field** (detector response; off by default; GISAXS and GIWAXS): **Choose…** next to *No flat field* takes a
detector image of the frame's size (CBF, TIFF, EDF, NeXus/HDF5: its first frame) or a NumPy `.npy` array, and **×**
stops it. Every frame is divided by the flat field, normalised to mean 1 over the pixels it keeps; pixels where it
is at or below 0.05 (dead, shadowed, module gaps) are left out. A flat field of another size is refused with the
reason, never resampled. The line under it says the mean it was normalised by and how many pixels it leaves out.
The same flat field serves every frame of a Batch Export, a series and the folder watch, and is saved with the
settings. With photon counts the error bars follow the division (scaled Poisson); the JSON record
(`corrections.flat_field`) names the file with its SHA-256, the mean and the pixels left out
([Physics and Methods 4.10](Physics_Manual.md#410-flat-field-detector-response)).

The order is fixed: invalid detector pixels → hot and dead pixels → masks → dark frame → background → pixels the
flat field leaves out → gap guard → intensity range (on the values as measured) → division by the flat field →
mirror fill → intensity corrections → normalisation. On a rotated detector the angle-dependent corrections are applied to the measured mirror partners before interpolation. Curves, q map, cake and exports all use the corrected frame.

**Normalisation** (folded, every term off by default; GISAXS and GIWAXS). Its first line always says which header
values the frame shown offers (*In the file header: exposure 20 s; Monitor = 5000*). After every other correction
each frame is multiplied by **A / (t · M · T · d)**:

| Term | Enter |
|---|---|
| **Exposure t** | *From the file header* (the exposure time the file states; of a sum of frames, their sum) or *Typed* (s) |
| **Monitor M** | a monitor of the header (*Header: I0 = …*: EDF keywords, the `Flux` of a Pilatus CBF, a NeXus `NXmonitor`; of a sum of frames, their sum) or *Typed* |
| **Transmission T** | 0 < T ≤ 1 |
| **Thickness d** | mm |
| **Absolute scale A** | a factor from a glassy-carbon or water measurement in the same set-up |

The line under the fields says by what the frame was multiplied, each term with its source, and the unit of the
intensity, which the plots, the CSV column and the fit input then carry (*counts/(pixel·s·monitor)*). The error
bars are multiplied by the same factor; photon counts keep their Poisson errors. A frame whose header lacks the
value chosen is not analysed, and the status line says why. The JSON record (`normalisation`) and every curve file
(`# intensity: …` line) state the factor and each term with its source; the settings file keeps the choice
([Physics and Methods 4.12](Physics_Manual.md#412-normalisation-exposure-monitor-transmission-thickness-absolute-scale)).

**Calibrating with a secondary standard**

1. Measure the standard in transmission with the same instrument set-up as the sample. Choose the instrument profile
   in Analyze or open a frame with geometry. Open **Analyze ▸ Absolute Scale from a Standard…**.
2. Choose the standard's detector file and frame number, or a reduced radial curve containing q, I and optionally σ.
   For a frame, check the current mask, dark, flat field and background first. Use the appropriate empty-cell or
   empty-beam subtraction. For a curve, supply corrected intensity before frame normalisation. Enter the standard's
   exposure, monitor, transmission and thickness in mm. Use 1 for an unused exposure, monitor or transmission term.
3. Choose **Water** and its measured temperature, or **Reference curve file** and your reference table (for example,
   your SRM 3600 certificate). Supply q, differential scattering cross section in cm⁻¹ and pointwise standard
   uncertainty in cm⁻¹. Select the q unit for each file independently. Enter any shared **one-standard-deviation**
   scale uncertainty once as a percentage. Use zero for a component with no supplied uncertainty.
4. Leave both q limits blank for the overlap, or enter a range in Å⁻¹. For water, select its flat SAXS plateau.
   Click **Fit Scale**. Inspect the overlay and the residual (scaled measurement minus reference). A trend in the
   residual can indicate a background mismatch or an unsuitable q range.
5. Enter the **sample** thickness in mm and click **Apply Scale**. GIMaP converts both thicknesses to cm and saves
   A, its standard uncertainty, source and UTC date in the instrument profile. Set the sample's own exposure,
   monitor and transmission under **Mask & Corrections ▸ Normalisation**, using the same monitor convention as for
   the standard. For a later sample, **Use Saved Scale** reuses the profile calibration with its new thickness.
   A change in detector distance, pixel size, tilt or wavelength requires a calibration for the new set-up.

The curve CSV labels intensity in `cm^-1`. In the frame JSON, `normalisation.standard_calibration` identifies the
standard measurement, the reference, the fitted q range and A ± σ_A. σ_A is common to the whole curve. The
pointwise σ column gives the statistical uncertainty of each reduced point. For a combined batch table,
each entry in `frame_normalisations` links its sample terms to the standard in the batch JSON.
A subsequent series normalisation to a target peak height gives relative intensities.

![Fit and residuals for a synthetic standard](manual_images/en/absolute_scale.png)

**Intensity corrections (GIWAXS)** (folded, off by default). They divide each pixel by a geometric
factor; q does not change:

| Correction | Enter |
|---|---|
| **Solid angle of each pixel (cos³2θ)** | — |
| **Polarisation, factor** | f (default 0.99): 0.95–0.99 at a synchrotron (horizontal), 0 for an unpolarised laboratory source, −1 vertical |
| **Absorption in the film** | **Thickness** (nm), **Attenuation length** 1/µ (µm, from tables such as CXRO or NIST) |
| **Lorentz factor (fibre texture)** | — (L = 1/(cos αi cos αf \|sin 2θf\|), as in GIXSGUI) |

Peak positions do not change; intensities compared across χ or q do — orientation, pole figures,
crystallinity ratios. The film absorption leaves out refraction, so it holds well above the critical
angle; with it on, pixels at or below the sample horizon are left out. The factors, and how the error
bars follow them, are in [Physics and Methods, chapter 5](Physics_Manual.md#5-intensity-corrections-for-giwaxs);
the JSON record (`intensity_corrections`) gives the formulas and parameters.

The **Lorentz factor** is for orientation and pole analysis of crystallites oriented at random about the surface
normal: after it, I(χ) counts the crystallites per degree of χ, and the Hermans factor of Run Automatic Analysis is
then not weighted by sin χ a second time. It is meant for crystalline peaks only: it also changes diffuse
scattering, the amorphous halo and the I(q) of a whole ring, and pixels on the specular rod are left out
([Physics and Methods 5.6](Physics_Manual.md#56-lorentz-factor-of-a-fibre-texture)).

### 5.3 Cuts

The **Cuts** step holds the cut positions. **Auto Cuts** returns to the automatic positions.

#### GISAXS: horizontal and vertical cut

GIMaP finds the Yoneda band (the brightest row between the sample horizon and an exit angle of
1.5°, measured beside the specular rod) and takes:

- the **horizontal cut** I(qy) in a band 5 px high at the Yoneda row (orange on the image);
- the **vertical cut** I(qz) in a band nominally 10 px wide. Its inner edge clears the projected specular rod by 15 columns when possible. Analyze tries the right side first, then the left. If neither side fits, it uses the side with more room and warns. This is a geometric clearance. A broader rod may need a larger offset.

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

#### GISAXS: cuts of your own

Under the halves is the **List of cuts**. Its first two rows are the automatic cuts — **Yoneda cut**, the one Fitting
takes, and **Vertical cut** — and you can add as many named cuts as you need:

- **Add ▾** — a horizontal band (I(qy)), a vertical band (I(qz)), a box (I(qy) or I(qz)), a sector (I(qyz) in one
  direction from the direct beam) or a line in any direction (I(s) along it). A new cut starts where it is easy to
  see and is selected, so you can type its position at once.
- **Draw on the image or the q map** — Horizontal, Vertical and Sector take one click, Box two opposite corners, Line
  its two ends. Esc cancels.
- The editor under the table sets the position: centre ± half width of a band, from–to of a box (and whether it is
  cut along qy or qz), χ ± half width and the q range of a sector, the start and end points and the half width of a
  line. The list beside **Add** chooses the unit: **q (Å⁻¹)**, **Angles (°)** (the exit angles: αf of qz, 2θf of
  qy on the sample horizon) or **Pixels (px)** (rows on the beam-centre column, columns along the calculated horizon). The line
  under the editor gives the position in the other two units.
- Tick a row to show or hide its curve and outline. Double-click a name to rename it. **Duplicate** copies the
  selected cut (of an automatic cut: a band of your own at the same q, to move elsewhere). **Remove** deletes it;
  Ctrl+Z brings it back. Typing the Yoneda cut sets its rows, as dragging the orange band does. **Auto Cuts**
  returns the automatic cuts to their places and keeps yours.

![GISAXS cuts on the q map](manual_images/en/06b_gisaxs_cuts.png)

*The Yoneda cut, a band at the specular reflection, a vertical band at a Bragg rod, a sector and a line, each in its
own colour in the table, on the q map and in the plots; the band selected is typed in exit angles.*

A cut has one colour everywhere: its row, its curve, its outline on the detector and on the q map, and its pixels
under **Sources**. Horizontal bands and boxes along qy are in the upper plot with the Yoneda cut, vertical bands and
boxes along qz in the lower plot; when there are sectors or lines, the list above the lower plot shows them. On the
q map you can drag and resize bands and boxes. A pixel belongs to a cut when the q of its centre lies inside (whole
pixels; only those above the horizon), so on the detector a band of constant qz curves slightly.

Cuts are kept in q: they hold for every frame of the same set-up, also in Batch Export and series maps, and **Save
Settings…** saves them. When a frame of another detector or geometry comes (detector, frame size, pixel size,
distance, wavelength, beam centre or tilt), the Cuts step asks: **Keep Cuts** or **Remove Cuts**; until you choose,
the cuts are applied at the same q. Each exported cut states its definition in q — and in αf or 2θf — in the
`# region:` line of its CSV and in `_analysis.json`; the automatic cuts add the qz and αf (qy and 2θf) of their band
edges.

#### GIWAXS: regions, Ring / Sector / Spot

A GIWAXS frame gives I(q) in the upper plot and I(χ) in the lower plot. χ = 0° is the surface
normal and ±90° the sample plane.

![GIWAXS cuts on the Cake view](manual_images/en/06_cuts_cake.png)

*Each region has its own colour, the same in the list, on the q map, on the Cake view and in the plots.*

Three views above the image:

- **Detector** — the frame as measured, in pixels;
- **q map** — the intensity regridded onto q∥ (qr) and qz. The grid is never finer than the detector
  samples q (at most 800 × 800 cells for GIWAXS, 600 × 600 for GISAXS), so the cells of the measured
  region hold a pixel; what stays blank was not measured (gaps, the mask, the missing wedge). The view
  opens on the measured cells;
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

In **Log** display, pixels at or below 0 — the noise of a dark- or background-subtracted frame — are drawn in the
lowest colour of the map (at the 0.1 % percentile of the positive values), not left without colour, and the
automatic limits come from the positive values only; masked pixels stay without colour. In linear display the
limits are the percentiles of every value, and the bar is drawn around them, so a frame whose brightest pixels are
a thousand times its typical ones keeps a readable bar. Only the display changes, never the data.

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
Start of the Series**) and **Report**. Sizes are Scherrer lower bounds. Phases are assigned only to
the phases you give (**Phases**, section 5.9).

What the numbers are is said in the report's *Decisions* (**Show the Full Text**): q is kinematic, without a
refraction correction, and the line gives how far refraction can shift q_z near or below the film's critical
angle at your αi (up to about k·sin αi); the uncertainties are statistical only, from Poisson counting statistics
or, for a floating-point, background-subtracted or mirror-filled frame, from the pixel scatter, without the
geometry. At most 12 candidate peaks, the most prominent, are fitted: when more qualify, *Decisions* and the line
under the peak table say how many candidates there were and how many of the 12 gave a peak (a fit can merge two
candidates or reject one), and the peak table, the rings and the series hints are those of the 12 (the command
line's `--max-peaks` fits more). The saved report gives q, d and the FWHM of each peak with their ±. The ± of a fit
whose reduced χ² is above 1 are multiplied by √χ²_r; a peak whose fit has χ²_r above 10 is marked *poor fit* in the
caveat column: its shape is not one Gaussian on a line, so read its width and size with care. A *lamellar
(1:2:3:4)* hint needs its orders one after the other: a peak near five times the first q without the third and
fourth order is not listed. The orientation sentence compares f with f_iso, the value of a random ring over the same measured |χ|: it
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
6. fits I(qy) with **Predict parameters**, Fitting's network method (section 6.4): GIMaP's network
   proposes one or two populations of spheres and upright cylinders, with or without a paracrystal
   distance D, and their values; each solution is refined by least squares and weighted by its posterior
   probability. The cut is taken at the effective interface (q_z = 0); 2π/q\* stays a result of its own
   and does not start the fit.
   Confirm the model in Fitting.

The solutions are listed most probable first, with their probability **p**. Probabilities are relative to the
compositions that were tried; the single solution of a composition you gave has none (*only this composition was
tried*). R, h and D show every population, in the order of the radii (`3.06 / 6.02`, `49.4 / –`: – for a population
without a paracrystal). The table's χ²ᵣ is Σ((I − model)²/σ²)/(N − number of free values) over the fitted points,
with the curve's σ: the number Fitting shows for the same solution. Predict parameters leaves out the points below
the network's q window (q < q_max/300); the report says how many (*fitted: 596 of 598 points; 2 below the network's
q window*). The run reports the most probable solution; the table selects it. A non-linear value that ended on a bound
of its range is not determined by the curve: the table marks it ⚠ and a warning under the table names it by rank
(*⚠ Not determined by the curve (at a search bound) — 1: D, …*; which bound, *D at its upper bound*, is in its
tooltip), as do **Fit details** and the report's note column; when the reported solution has one, a point to check
(**Fit**) says so. The amplitudes, the background and the beam-tail amplitude are fitted non-negative and are not
counted as values at a bound; one that ends at 0 is named among the report's *Decisions* (*fit caveat: Background
at 0, the non-negativity limit*). A reported solution with χ²ᵣ above 10 — the limit of Fitting's poor fits
(section 6.5) — misses the curve by far more than its noise: a point to check (**Fit**) says that none of the
compositions tried describes the cut and that its R and D are not measurements (the Bragg rings of a powder or a
calibrant give such a misfit). The particles are evaluated at q_z = 0, which puts a sphere's radius more than about
5 % off once q_z·R is above 1.7 at the cut's q_z = k(sin αi + sin αf) (αi of the geometry, αf of the Yoneda band the
cut is at): such a sphere or random cylinder is flagged (*R: q_z·R is above 1.7 at the cut's q_z, where the form
factor at q_z = 0 … biases the radius, usually too small; give both angles of the cut …*), and a point to check
(**Fit**) gives its q_z·R and says what to do: **Send to Fitting**, enter αi, αf and λ under Model ▸ **Spheres in
grazing incidence** and run Predict parameters again, which then fits spheres with the exit-reflected path (random
cylinders stay at q_z = 0). Without αi, or with the cut away from the Yoneda band, nothing is flagged. When another solution has a probability of 0.2 or more, the curve alone does not
decide the model: a line under the table says so, and the Results tab asks which one you expect and names the
solutions whose D is within 15 % of 2π/q\*. The report gives the refined R, h and D of the reported
solution with their 68 % intervals (*fit intervals: R (1) 3.17 [3.04, 3.26] nm …*: the refined value, then its
interval; from the least-squares errors when the posterior's sampling did not converge). Every value has one name,
that of Fitting's Model table, in the table, the notes, Fit details and the report: Tail A, Tail w and Tail ν for the
beam tail, σ_q, Scale, and with several populations the population's number in brackets (R (2), σR/R (1)); the
report lists the keys of its CSV and JSON files with these names. The notes about the background, the beam tail and
the resolution come after those about the particles, in one paragraph, and a term that is absent (Tail A = 0) is
not flagged. **Save Fitted Curve…** (q in Å⁻¹, I, σ and the fit the run reports), **Save Fit
Table…** (every solution with its probability, parameters in nm, the 68 % intervals of R, h and D, χ²ᵣ and the number of fitted points, the
values at a bound in `at_bounds` and the reported one in `chosen`) and **Save Plot…** save the results; **Refine
in Fitting** opens the prepared curve (the halves chosen) in Fitting.

As for GIWAXS, the report's *Decisions* (**Show the Full Text**) say what the numbers are: q is kinematic,
and near the critical angle refraction mostly shifts q_z — that of the Yoneda band most — by an amount of
the order of k·sin αi, which the line gives for your frame (q_y and 2π/q\* barely change); the σ of the
cuts, by which χ²ᵣ is normalised, are statistical only, from Poisson counting statistics or the pixel
scatter, without the geometry; the intervals of R, h and D come from the posterior given those σ, so they do not
include the geometry or the choice of model.

![Results of the automatic analysis of a GISAXS frame](manual_images/en/08_gisaxs_results.png)

*The cut with its halves, the length scale 2π/q\*, the fit (Log q / Log I) and the table of solutions.*

**Fit details** (closed under the peak table or the fit table until you open it; it then stays
open for later reports) follows the selected row:

- a peak: the points it was fitted to, with the Gaussian and its local linear background; q, d and
  FWHM with errors; height, area, background and slope; S/N, χ²ᵣ, the fit window, the Scherrer
  size, and where the ring is measured, shadowed or missing on the q map (how the peak was fitted and how
  peaks are found, with the thresholds of the run, is the tooltip of **Fit details**);
- a model solution: every parameter (R, σR/R, h, D, σD/D, weight, Scale, Background, Tail A, Tail w, Tail ν, σ_q),
  with the 68 % interval of R, h and D, its probability, χ²ᵣ with the points it is of, log RMSE, convergence and time, the points and q range, whether
  you gave the composition or the network chose it, its warnings and values at a bound, and **Show in Fitting**,
  which opens the curve in Fitting with that solution drawn.

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
Both carry a section **Mask** and a section **Record**: the GIMaP version, the input file with its
SHA-256, the calibration file with its path and SHA-256 when the run took the geometry from one, the
instrument profile with its source (a profile made from a file names that file's path), and the
corrections (background, gap guard, bad pixels, intensity corrections).

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
   Table…** (the value of the lower plot for every frame), **Peak Table of Every Frame…**, **Save Results
   Database…** (every frame's curve into `gimap_series.sqlite`, which Compare reads, with the fits of *Results: batch
   fitting* when a Batch Export fitted the frames; added to a database that is there, a frame recorded before
   replaced) and **Timeline as Table…**.

![The timeline under the Series map](manual_images/en/09b_series_timeline.png)

*The timeline of 40 frames made of the public GALAXI frame with a side peak that moves at once at frame 25: χ² ≈ 1 between frames that differ by noise alone, ▲ at the jump, dashed lines at the stage boundaries.*

**The timeline.** The drop-down of the plot under the map also has **χ² to the frame before (timeline)** and
**χ² to the reference frame (timeline)**: for every frame, how far its curve is from the one before it, or from
the reference frame (**Options ▸ Reference frame**, the first by default), in units of the noise of each x
column, which is taken from the series itself. Two frames that differ by noise alone give about 1; a transition
stands out as a peak. Dashed lines mark the stage boundaries, ▲ the jumps (the frames where the series changes at
once) and × the odd frames. **Click a point** of the plot under the map — of the timeline or of any other trace —
to pick that frame and show it in Analyze. **Timeline as Table…** writes per frame both χ², whether it is a jump,
its stage and whether it is odd. How it is computed: [Physics and Methods, section 11.12](Physics_Manual.md#1112-the-timeline-how-far-each-frame-is-from-its-neighbour-and-from-a-reference-frame).

**One cut for the series.** While the cut is automatic, every frame of the map — and of a Batch
Export, a Send Series to Fitting and a watched folder's export — is cut where the frame on screen is:
for GISAXS the selected horizontal rows and vertical columns, for GIWAXS the ring of I(χ). A row
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

The dialog says where each folder of frames gets its geometry: *geometry from LaB6.poni beside them*, *profile
“…”, as the frame on screen*, or *the profile picked or the one that fits their detector and size, read frame by
frame; a frame without one gives no q curves*. The summary line names the geometry of the frame on screen. When
no frame can have a geometry (none on screen, no `.poni` beside any frame, no instrument profile), the dialog
says *No geometry: q curves cannot be made* and **Export** stays off.

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
| **Data (numbers)** — CSV (comma), TXT (tab) or DAT (space) | **All frames in one table per curve** (x first, one column per frame); **Each frame: its curves, with error bars and pixel counts**; **Each frame: its q curves as NXcanSAS (.h5)** (section 5.8; always HDF5, whatever the format); **Each frame: the curve for Fitting**; **Each frame: the q map as numbers**; **Each frame: the unwrapped χ–q map as numbers** | `<name>_<curve>_frames.csv`, `curves/<frame>_<curve>.csv` (+ `_analysis.json`), `nxcansas/<frame>_nxcansas.h5`, `fit_input/<frame>_fit_input.dat`, `maps/<frame>_qmap.csv`, `maps/<frame>_cake.csv` |
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

For tables, per-frame curves and NXcanSAS files, tick the curves you want (**All** / **None**). In a table of all
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
  cut** fitted with **Predict parameters**, as in section 5.5 (**Model**: **Automatic (the network chooses the
  composition)** — one or two populations of spheres and upright cylinders, with or without a paracrystal — or
  one sphere, random cylinder or vertical cylinder with a paracrystal; a cut of more than 1000 points is merged
  first). Every frame is predicted from its own data, without start values. The two halves are never fitted
  together. One row per frame, the most probable solution: `halves`, `points`, model, `chi2` (χ²ᵣ, as in section
  5.5), `probability` and the parameters in nm. From some seconds to a few minutes per frame, depending on the number of points and the computer. The choice starts on the Cuts step's halves
  when they are the mean or one half, else on the last batch's. A frame without counts, or without points on the
  chosen half (*No points on the q > 0 half*), is not fitted and its reason is listed. The curves for Fitting still
  hold both halves with signed qy and name the choice.
- **Start values** (peaks only): **The previous frame's result** (a slowly changing series), **The first frame's
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

**Export** (command bar, `Ctrl+E`, and the **Analyze** menu) writes, next to the data in `gimap_analysis/`;
**Export Curves…** in the Export step asks for the folder first — it starts in the folder chosen last (remembered
between sessions), else next to the data — and writes the same files there:

- one CSV per curve (`x, I, sigma, pixels`), named `<stem>_<curve>.csv`; a cut region's curves carry the region's
  name (`<stem>_ring_q1_0.csv` and `<stem>_ring_q1_0_chi.csv` for a region *ring q1.0*), with its key added when
  two regions share a name or a name is that of another curve (`<stem>_ring_q1_0_region3.csv`); Batch Export
  keeps the keys (`region1`), as its tables do;
- `<stem>_analysis.json`: the source, geometry, where the beam centre came from, corrections, the
  number of bad pixels left out, the cuts;
- `<stem>_fit_input.dat`, the curve for Fitting (**Analyze ▸ Send Both Cuts to Fitting** also writes the vertical
  cut beside it as `<stem>_fit_input_qz.dat`, section 6.6);
- with **Also as NXcanSAS (.h5)** ticked (below **Export Curves…** in the Export step; remembered between sessions),
  also `<stem>_nxcansas.h5`: the curves against q as **NXcanSAS** (HDF5, NeXus definition NXcanSAS 1.1), which
  SasView, SASfit and other NeXus programs open, and Fitting too (section 6.1). Each curve is a SASentry of its own,
  titled with the frame, the curve and which q its Q is: a horizontal cut keeps its signed qy (both halves; a
  program that fits |Q| sees each |q| twice, so fit one half there), a vertical cut has qz. Each holds Q in
  1/angstrom as Analyze computed it, I (1/cm after standard calibration, arbitrary units otherwise), Idev = σ, a mask of the points
  without a finite value, the wavelength, the distance, the pixel size and αi (as the sample's pitch), and the same
  record as `_analysis.json` (as JSON in a note of SASprocess). I(χ) and curves shown against 2θ are not in it —
  NXcanSAS has no angle axis — but in their CSVs. The box applies to **Save the Curves of the Plot** and **Export
  New Frames While Watching** too.

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

### 5.9 Phases: indexing and pole figures

**Phases** in the image toolbar opens the **Phases and Indexing** window (GIWAXS frames). It has three tabs.

**Phases.** **Add from CIF…** reads a CIF: the cell, the symmetry operations and the atom sites (intensities |F|²
from the atoms). **Add Cell** adds a cell you type in, with its lattice **Centring** (P, A, B, C, I, F, R): its
allowed reflections all get intensity 1. Under **Selected Phase** you change the name, the cell and the
**Orientation**: **Powder (random orientation)**, **Plane (hkl) parallel to the substrate** or **Axis [uvw] along the
surface normal**, with the three indices. **Show** hides or shows a phase. Each phase has its colour; the names are
written in the corner of the image. **Overlay** sets **q max** (3 Å⁻¹), the **hkl labels**, and **Shift the fibre spots
for refraction in the film**, with **αc of the film** (0: the αc of the refraction correction, section 5.2) and
**Also the reflected channel (DWBA)**.

What is drawn:

- a fibre phase: a dot at (±q∥, qz) of every reflection on the **q map**, at its pixel on the **Detector** and at
  (q, χ) on the **Cake**; the dot grows with m·|F|², hollow dots are in the missing wedge (not seen by the detector)
  or in the reflected channel; hover a dot for its hkl, d, q∥, qz and χ;
- a powder phase: arcs of its rings on the q map and the detector, lines on the cake;
- every phase: a dashed line at each ring on **I(q)**, labelled for the strongest.

![Phases on the q map and I(q): tetragonal MAPbI₃ with (110) parallel to the substrate, and PbI₂ (public P08 frame)](manual_images/en/07a_phases_overlay.png)

**Reflections.** The table lists, for every phase shown, the fibre spots or rings: hkl, d, q, q∥, qz, χ, the
multiplicity m, |F|² and the intensity relative to the strongest (100). Copy it with Ctrl+C. Select a row and press
**Pole Figure of This Ring**: the I(χ) ring moves onto that reflection and its pole figure is fitted with χ0 of the
reflection.

**Pole Figure.** The ring is the one of I(χ) (the purple band over I(q)). **Fit Pole Figure** folds its I(χ), takes
off the background under the ring (**Automatic**: from I(q) at the ring; or type a value), applies the Lorentz factor
(sin χ, or none when the frame is already Lorentz-corrected) and fits a **Model**: **Gaussian + isotropic** or
**Maier–Saupe + isotropic**, about a **Direction**: **Along the normal (χ0 = 0°)**, **In the plane (χ0 = 90°)**,
**Fitted direction** or **Set χ0**. **Fit up to χ** (85°) leaves out the bins next to the sample horizon; lower it when I(χ)
falls towards the horizon. The plot shows the points fitted, the points left out (grey) and the fitted curve. The
result gives the **oriented fraction**, the **FWHM** (mosaicity) and χ0 with their ±, the Hermans f of the fitted
distribution (the missing wedge filled by the model) next to f from the measured bins alone, the 2D order
2<cos²χ> − 1 and χ²_red. A χ²_red far above 1 means that one Gaussian and an isotropic part do not describe the ring:
read the numbers with care.

![The pole figure of the MAPbI₃ ring at 0.997 Å⁻¹ (public P08 frame)](manual_images/en/07b_pole_figure.png)

**Export…** writes the pole figure as CSV (`<frame>_pole_figure.csv`, in
`gimap_analysis/` next to the data by default) and `<frame>_indexing.json` beside it: the phases (with the CIF and its
SHA-256), the reflection table, the settings and the fit.

**In the automatic analysis.** With phases shown (or `--phase FILE.cif` on the command line, repeated for several),
**Run Automatic Analysis** assigns each peak to the nearest reflection within max(3σ, FWHM/2): the *Decisions* line
*phase assignment* gives the phase, hkl and Δq of every peak; the report's **Phase assignment** section adds the other
reflections within the window. Each analysed ring also gets a pole-figure fit next to its Hermans f (a line under the
ring in the report). The nearest reflection is an assignment, not a proof: check the alternatives and Δq. The
formulas are in [Physics and Methods 7.10 and 7.11](Physics_Manual.md#710-reflection-overlay-and-phase-assignment).

## 6. Fitting

The **Fitting** page fits 1D curves with particle models. It has two modes, switched at the top of the page:

- **Single analysis** — one curve, in four steps (section 6.2).
- **Batch / In-situ series** — selected text curves (including several intensity columns in one file), every
  curve of a folder, or every frame of a detector series, fitted with the model of Single analysis (section 6.5).

Switching between the two keeps each side's curve, model and results. With no curve open, the Curve step
offers both ways in side by side: **Open Curve…** and **Fit a Folder of Curves…**. For a series or a batch
of samples the usual way is: fit one representative curve here (predict its parameters, then refine), then
press **Fit Every Curve in This Folder…** in Results.
For selected TXT/DAT files or a shared-q table, use **Import Curves…** in either page's command bar or the
**Fitting** menu. Dropping several curve files on Single analysis opens this same import preview.

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
| In-situ CBF / NXS series | Analyze ▸ **Send Series to Fitting…**, or the frames themselves in **Batch / In-situ series** (section 6.5) |

### 6.1 Curves Fitting reads

Fitting reads `.dat`, `.txt` and `.csv` text curves and NXcanSAS files (`.h5`):

- Analyze's curve files, `<name>_fit_input.dat`, have four columns — q (Å⁻¹), intensity, σ and the number of
  detector pixels averaged — and a header recording where the curve came from.
- Any other `.dat` / `.txt` file with columns `q I [σ]` opens too. A file whose header states the unit of q is read
  in that unit — `q in <unit>` in a `#` line, or the name of the first column (`q (1/A)`, `q_nm^-1`,
  `# columns: qy (1/A) …`, `# q [1/nm]  I`); Analyze's curve files state Å⁻¹. For a file that states none, set
  **Curve ▸ File ▸ q unit of files without one in their header** (Å⁻¹ by default, in the 1D Predict window too).
  A file whose header states both Å⁻¹ and nm⁻¹ is not opened: *The header states q both in 1/A and in 1/nm; correct
  it to one unit.* A file that states no unit is always said so — *Unit of q assumed: Å⁻¹ (not stated in the file).
  If its q is in nm⁻¹, every length found (R, D) is wrong by a factor of 10 …* with how to change it — in the Curve
  card, in the Fit step, in the status line when the curve opens and in Results next to the values. When the q
  range in the assumed unit is also implausible for a GISAXS or SAXS cut (|q| beyond 10 nm⁻¹ = 1 Å⁻¹, or a whole
  curve below 0.02 nm⁻¹), the warning says that instead, and what it does to every length (R and D come out 10×
  too small or too large). A title line above the line of column names is a header like any other; a row that is
  no numbers, or that holds NaN or inf, is left out, and the card and the status line say how many (*3 rows of the
  file could not be read as numbers and were left out.*).
- NXcanSAS files (`.h5`, HDF5 written by Analyze, SasView, Mantid and other programs; `.hdf5` and `.nxs` through
  **All files**): each SASdata group of the file is a curve — Q, I, Idev as σ, and Gaussian Qdev when present. q is read in the unit the file
  states on Q: 1/angstrom and 1/nm as they are, 1/m converted to nm⁻¹; the Curve card says *q read in Å⁻¹: stated in
  the file*. A Q whose unit Fitting does not know is read like a text file that states none. Points the file's mask
  leaves out are not fitted. A file of several curves asks which one to open (**Open Curve…**, a dropped file) and
  is then named *file.h5 · curve 2*; **Import Curves…** takes every curve of it. A file of fits (Save ▸ Data and
  Fit, an In-situ series' **Fits as NXcanSAS…**) holds the points fitted and the model of each: only its measured
  curves are offered (the first chosen; one alone opens at once), and **Import Curves…** takes those, so the fits of
  a series open again as a series. An HDF5 file of detector frames
  is no curve: dropped on the window it goes to Analyze.

A `.csv` file is read by the names of its header, not by the place of its columns: q is the column named q (`q`,
`qy (A^-1)`, `q_nm^-1` …), I the one named I or intensity, σ the one named sigma, σ, err or dI, and the Gaussian q width the one named dQ or sigma_q. Other
columns (a model, residuals, terms, pixel counts) are not read. So the cuts of the command line (`horizontal.csv`,
the names in its last `#` line), Analyze's curve tables and Fitting's own **Save ▸ Data and Fit…** table — which has
q twice, in nm⁻¹ and then in Å⁻¹: the first is taken, in nm⁻¹ — open as they are. q is read in the unit the name of
its column states. A CSV without a header of two or three columns is read as `q I [σ]`, like a `.dat` file. When the
header does not decide (two intensities, q of two axes, or no header and more columns), **Choose Columns** shows the
header and the first ten rows, as **Import Curves…** does, and asks which column is q, I and σ (or none), and the
unit of q when the q column's name states none. The record of a fit names the columns that were read
(`curve.columns_read`).

Single-curve text files without column names use `q I [σ]`. To supply Gaussian resolution, name the columns,
for example `# q_nm^-1 I sigma dQ`. dQ is one standard deviation and defaults to the q column's unit.
The pixel count in Analyze's fourth column is a count. To split `q I1 I2 …`, use **Import Curves…** explicitly.

**Import selected files or several curves from one file**

![Previewing shared-q curves before importing](manual_images/en/11a_fitting_import.png)

1. Click **Import Curves…** and select one or more `.txt` / `.dat` files, or NXcanSAS `.h5` files. The cancellable
   preview shows the first ten data rows, column roles, q unit and total number of curves. It does not rewrite or
   split files. An NXcanSAS file gives every curve in it (one per SASdata group, in its order; the rows shown are
   those of the first): the column layout does not apply to it.
2. For each file, check **Column layout**:

   | Layout | Interpretation |
   |---|---|
   | **Single curve: q, I, optional σ** | Column 1 is q, column 2 is I, column 3 (when present) is σ. |
   | **Shared q: q, I₁, I₂, …** | Column 1 is shared q; each remaining column is a separate intensity curve with no supplied σ. |

   A clear header such as `q I sigma` or `q I1 I2` suggests the corresponding layout. Without an unambiguous
   header the default remains **Single curve**: numbers alone cannot distinguish σ from I₂. Select **Shared q**
   yourself when appropriate. **Apply layout to all** is available for files with matching columns.
   Clearly labelled error columns (such as `sigma`, `sigma1`) are protected from shared-q import.
   Clearly labelled intensity columns cannot be used as σ; mixed layouts with an error column elsewhere
   must be rearranged into one of the supported layouts.
   GIMaP Analyze's marked `q I sigma pixels` exports always remain one curve; sigma and pixel counts cannot be
   imported as intensities. Paired multi-curve errors (`q I1 sigma1 I2 sigma2`) are not supported by this importer.
3. Choose **Use curves as**: **Independent samples (fit each curve)** is the default. Each curve starts from the
   same Single analysis model; stage detection, odd-frame exclusion and continuity between curves are disabled.
   **Time series (file and column order)** uses the displayed file order, then columns from left to right; check
   that this is the intended experimental order. It enables the usual series controls (section 6.5).
4. Set **q unit when absent from the header**. Its initial value comes from Single analysis; a unit in each
   file's header takes priority. Click **Import N curves** to open the fixed selection in **Batch / In-situ series**.
   Select a representative curve and **Open Frame in Single Analysis** to set up or refine the model, then return
   and press **Start**. The fit range and left-out points also come from Single analysis: check them before Start.

The list and results identify each curve by its file and intensity column. Projects preserve that exact
selection and column mapping, including header names; reopening reports missing files or changed columns.
The CSV and JSON exports retain the source file and selected q/I/σ columns. A mapped curve opened in Single
analysis also keeps this source in its fit record; **Import Curves from This File…** returns to the preview.
The original files must remain available. **Watch for new curves** is unavailable for a fixed selection;
import again to change its files/columns, or use folder mode for watching ordinary single-curve files.

Missing σ stays missing: Refine and Search weight ln I equally, without inventing counting errors; Predict
parameters takes a relative σ from the scatter of ln I about its trend (section 6.4). Each imported curve
keeps q, I and its supplied σ aligned by row; invalid values in one intensity column do not remove valid points
from another curve.

Other ways to get a single curve in:

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
   for the horizontal cut of a GISAXS frame, not diffraction peaks, so no method is recommended for it (for peaks
   use **Analyze ▸ Batch Export ▸ Peaks of the ring regions**).

3. **Analyze ▸ Results ▸ Fit details ▸ Show in Fitting** (after a GISAXS automatic analysis, section 5.5) —
   opens the same curve and puts the selected solution into **Model**. The automatic analysis fits with
   Fitting's own model (Predict parameters, sizes *Gaussian, by number*), so the solution comes in as it is; the
   status line says how closely Model reproduces the solution's curve. **Refine in Fitting** opens the curve
   without a solution.

### 6.2 Single analysis: one curve in four steps

![Fitting, Single analysis](manual_images/en/11_fitting_single.png)

*The step rail on the left, here with the Results step open: the quality, the next steps (Fit Again, Fit Every
Curve in This Folder…), what was done automatically at the edges of the ranges and the warnings (when there are any,
none for this fit), and every value ± its error; on the right the curve with the model, the orange fitting range and
the residuals below.*

The page is laid out like Analyze: the steps on the left, each saying where it stands, and the plots on the right.

- **The curve plot**: the points (paler outside the fitting range), the model over the whole curve and the
  **orange band** of the fitting range (drag it). **Components** (off initially) shows the additive intensity
  contributions: particles, the constant background and any beam-tail/low-q term. The total model is a stronger
  solid line; contributions use thinner dashed or dotted lines. Use the arrow beside **Components** to choose
  individual contributions.
- **The residual plot** below uses (I − model)/σ for the selected intensity weights or ln(I/model) for a log fit.
- **The command bar**: **Open Curve…** (its menu: **Load Model…**), **Undo / Redo** (`Ctrl+Z` /
  `Ctrl+Shift+Z`: every change of the model, the fitting range or the left-out points is one step, and so is every
  fit, solution used and model loaded), **Fit** (`Ctrl+Return`), **Stop** and **Save** (**Data and Fit…**,
  **Plot…**, **Model…**, **Solutions…**). After Undo or Redo the status line names the step (*Undone: the fit
  (Refine).*), and the buttons' tooltips name the step they would take (*Undo: solution 2 in Model (Ctrl+Z)*).

Showing or hiding contributions keeps the current x/y limits, including a manual zoom. By default, automatic
limits and **Reset** use only the data and total model; small component tails can lie below the visible range.
To inspect those tails, select **Include components in auto range** in the component menu. Changing this option
resets the view immediately, and subsequent **Reset** uses the same choice. On log axes, zero or negative values
cannot be drawn; they are not shifted, clipped to a positive floor or normalised. These choices only affect the
display: fits, residuals and **Save ▸ Data and Fit…** retain the complete model and all contributions. Contributions
are intensities that add to the model, not particle fractions. For a batch result, choose **Open Frame in Single
Analysis** to inspect its contributions here.

**1. Curve**

- The card: its points, its q range in nm⁻¹ (in Å⁻¹ in the card's tooltip), *σ from the file*, and the unit its q
  was read in and where that came from (*q read in Å⁻¹: stated in the file*, or *… not stated in the file: the
  setting for files without one*). The fit weights each point by 1/σ. Points whose σ is zero, negative or missing
  are left out of the fit, and the card says how many (*4 points with a zero, negative or missing σ were left out:
  they cannot be weighted …*); the card says *σ from the file* only when the fitted points are weighted by it.
  Without σ a warning in the card says what
  each method does — *⚠ No σ in the file. Predict parameters: a relative σ of 3.4 %, estimated from the scatter of
  ln I about its trend. Refine and Search: ln I is fitted, every point weighted alike.* — and the Fit step says it
  for the method chosen. Points with a negative or zero intensity are counted: with σ they are fitted like any
  other point (*12 of the 400 fitted points have a negative or zero intensity (9 negative, 3 zero): they are kept
  and weighted by their σ …*, in the card and in the status line when the curve opens); without σ they are left
  out, because ln I has no value there, and a warning says how many.
- How to set the fitting range, what Add Particle offers, what each method does and how to fix values are the
  tooltips of those controls; the steps themselves only say where things stand.
- **Halves of the cut**, for a cut through the beam (q < 0 and q > 0): **Mean of both halves** (averaged where
  both halves were measured; across a gap of one half, and beyond its end, the other half alone — never
  interpolated), **Both halves on |q|**, **q > 0 half** or **q < 0 half**. *Predict parameters* takes one curve:
  on **Both halves on |q|** it switches the box to the mean and says so; *Refine* and *Search* fit both halves.
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

- **Add Particle** adds one card per particle: **Sphere**, **Random cylinder**, **Vertical cylinder** or
  **Core–shell sphere** (a core of radius R in a shell of thickness **t**, the core's contrast **ρc/ρs** times the
  shell's; for Refine and Search: Predict parameters does not propose it).
- Each card has **S(q)** on or off (the interference between neighbours). With it, **Interference** chooses the
  function — **1D paracrystal** (the default: D and **σnn/D**, the relative standard deviation of the distance between
  neighbours), **2D hexagonal paracrystal** (a disordered hexagonal lattice in the plane, averaged over its orientation:
  D and σnn/D) or **Hard spheres (Percus–Yevick)** (radius **R_HS** and volume fraction **φ**); the last two are for
  Refine and Search — and **Coupling** LMA (P·S, the default) or DA (sizes independent of positions;
  [Physics and Methods, sections 9.5](Physics_Manual.md#95-structure-factor-one-dimensional-paracrystal) and
  [9.14](Physics_Manual.md#914-models-for-manual-fitting)). The values: scale, R and **σR/R**, h and **σh/h** for random
  cylinders, t and ρc/ρs for core–shell spheres, and those of the interference function. Spreads are relative
  everywhere (as in Analyze's results and in 1D Predict). The scale is the intensity of the particles at q = 0: every
  form factor is 1 there.
- Each value is one line: its name (the tooltip gives its name in links, such as `R_1`), the value, ± its error after a
  fit, **min** and **max** (with **Ranges**), **fit** and **=** (link). Two populations fit on one screen.
- For **Refine** and **Search**, choose **Sizes** on each population. **Gaussian, by number** uses centre R and
  width σ = R·(σR/R), restricted to positive radii within four σ. **Log-normal, by number (median)** uses median
  M and s = SD(ln R). Its number mean is M exp(s²/2) and its relative standard deviation is √(exp(s²) − 1).
  **Schulz (Gamma), by number** uses number mean R and relative standard deviation w, with shape 1/w² and scale Rw².
  The results give the number mean and relative standard deviation after normalization. Saved models retain their
  original laws. Models using the new laws, weights or file resolution use the `gimap_fit_model_v3` JSON format.
- Under **Resolution**, choose **Constant σ_q** (0 gives an unsmeared curve) or **Resolution from the file**.
  The latter averages each particle term at |qᵢ + dQᵢ Z| with Z a standard normal variable. dQ must be finite and
  nonnegative at each fitted point. A width of zero evaluates that point directly.
- Under **Weights**, **σ from the file** uses residual (I − model)/σ, **Relative (σ = I)** uses (I − model)/I
  and **Poisson (σ = √I)** uses (I − model)/√I. **None (fit ln I)** uses ln I − ln model, as does the default
  when no usable file σ is available. Relative, Poisson and log fits use positive intensities. **Save Data and Fit**
  records these choices and the fit report lists them. [Physics and Methods, chapter 9](Physics_Manual.md#93-size-distributions)

![Size law, resolution and weights for a manual fit](manual_images/en/11b_manual_fit.png)

- **fit** ticked: the fit may change the value. Unticked: the value stays fixed.
- **=** links the value to another (**Same as**: e.g. one σR/R for two populations) or to an expression of others
  (**Expression…**: `2*R_1`, `R_1 + t_shell_1`; numbers, + − * / **, sqrt, exp, log, min, max …). A linked value is
  greyed: it follows the others while you edit and in every fit, is not fitted itself, and gets its error from theirs.
  The card **Links and constraints** lists every link (× unlinks), **Particles do not overlap** (D ≥ 2R, R_HS ≥ R
  while both are fitted; on by default) and the inequality constraints: **Add Constraint…** takes, for example,
  `D_2 >= 2*R_2 + 3`. Links and constraints are saved with the model and in the fit's record; removing a particle
  removes those that read it ([Physics and Methods, section 9.15](Physics_Manual.md#915-links-and-constraints-between-parameters)).
- **Ranges** shows each value's min – max. The fit keeps the value inside it; *Search the ranges* searches
  across it. A value typed outside its range widens the range. For linear terms such as intensity and background,
  **∞** means no upper limit; untick it to enter a finite maximum.
- **Background and beam tail**: a constant background, the beam tail A / (1 + (|q|/w)^ν) at q = 0 (**Tail A**,
  **Tail w**, **Tail ν**: the empirical tail of the direct and reflected beam), and the q resolution **σ_q** (0, off,
  by default); an overall factor k, fixed at 1 by default, multiplies the particles and the tail. The tail's ranges
  follow the curve: A up to 10× its largest intensity, w from 0.3 q_min to min(30 q_min, q_max), ν from 1 to 6 — set
  when a curve opens with the starting model and when Refine or Search starts from the generic ranges (the status
  line says so); ranges you type are kept. A tail at an edge of its range, or with no finite error, is flagged after
  the fit ([Physics and Methods, section 9.6](Physics_Manual.md#96-parameters-and-defaults)). The model as a formula, and how **σnn/D** relates to the σD/D of earlier versions and of 1D Predict, are in Physics and
  Methods, [section 9.2](Physics_Manual.md#92-the-intensity-model) and
  [section 9.5](Physics_Manual.md#95-structure-factor-one-dimensional-paracrystal).
- **Spheres in grazing incidence**: αi, αf, λ and the effective reflection coefficient |r|, arg r. With αi = αf = 0
  (the default) spheres are evaluated at q_z = 0, the effective interface approximation, which needs nothing about the
  substrate. Enter αi, αf (the row of the cut) and λ to evaluate spheres as the direct plus the exit-reflected path,
  and free |r| and arg r to fit the reflection; cylinders are not affected. After a fit with the angles entered, the
  results show q₊R of every sphere. **Substrate** (under the values): *Unknown* (the default, the levels above), a
  material of the table (**Si**, **SiO₂**, **Glass (plate)**, **Au**: δ and β at the cut's λ, 4.9–30 keV) or **Typed δ,
  β**. With a substrate and both angles the spheres are in the four-term DWBA with the substrate's Fresnel coefficients
  (|r| and arg r are then not used); the line below says which level is in use (*Level: Four-term DWBA, substrate known
  on Si (δ = …, β = …)*). Predict parameters does not use the substrate; its solutions come back without it
  ([Physics and Methods, section 9.17](Physics_Manual.md#917-the-four-term-dwba-with-a-known-substrate)). When to use which, with examples on Si and on a polymer film:
  [Physics and Methods, section 9.10](Physics_Manual.md#910-spheres-and-cylinders-in-a-horizontal-cut-the-effective-interface-approximation);
  how to read a fit with three populations: [section 9.11](Physics_Manual.md#911-the-three-population-model).
- After a fit every value shows its ± error, or *⚠ at a bound* (orange) when it stopped at the edge of its range
  (section 6.3, *Values at the edge of their range*).

A new installation starts with a neutral model, not with the values of an earlier fit. Set the particle and its
sizes for your sample, or let the Fit step find them.

**3. Fit** — one **Method** drop-down and one button. **Method** has three entries: **Predict parameters**
(GIMaP's network, then least squares), **Refine the values in Model** and **Search the ranges, then refine**
(section 6.3).

- When a curve is opened the page chooses the method for it. An unfitted GISAXS horizontal cut gets **Predict
  parameters**, and the line under **Method** says why: *Recommended for this curve: a GISAXS cut not fitted yet.*
  Every other curve keeps the method chosen before (a GIWAXS I(q) never gets a recommendation). After a prediction
  the method turns to **Refine the values in Model**. Should GIMaP's network be missing, the line under the method
  says so and names its file (section 14.3).
- Under the method, only what it needs: **Particles** for Predict parameters (section 6.4).
- Press **Fit** in the command bar (`Ctrl+Return`) or the button below, which carries the method's name (for
  example **Refine**). A progress bar and **Stop** appear while it runs. **Refine** and **Search** go on from their
  own result while χ²ᵣ falls by more than 0.1 % (at most 5 more runs; the Run Log and the report say how often), so a
  fit no longer stops above a minimum that a second Refine would reach.
- **Options** (folded): **Evaluations at most (Refine, Search)** (automatic by default) and **1D Predict Batch
  Window…** (section 6.4).

**4. Results**, from the top:

- The quality: χ²ᵣ (with σ) or the variance of ln I (without σ), the points, the free values, converged or not —
  or *did not improve on the start values* when no curve Refine or Search tried was better than the one it started
  from: the start values are kept, the fit is not called converged and no errors are given (without σ this happens
  when the start already describes ln I better than any curve whose scales the fit solves; change a value, or use
  Search). Refine and Search refuse a curve with fewer points than free values plus one (*Only 5 points in the
  fitting range for 9 free values: a fit needs at least 10 …*) and a curve without an intensity above zero.
  After Predict parameters it says how many solutions there are and that the most probable is in Model: **Refine**
  gives its quality and the error of every value.
- What to do next: **Refine These Values** (after a prediction), **Fit Again** (the same
  method, with the values, fixed values and ranges now in Model) and **Fit Every Curve in This Folder…** (the
  In-situ series on this curve's folder, each curve starting from this model).
- **Fit details…** opens a separate, nonmodal window for boundary diagnostics, correlations and assumptions;
  after Predict parameters it gives the posterior of the solution in Model (section 6.4). Routine diagnostic notes
  do not turn a completed prediction into a yellow warning.
- **Uncertainty and Report…** (after Refine or Search): **Errors and correlations** lists every free and linked value
  with its 1σ error and the correlation matrix as a heat map with its values, and names the pairs with |r| > 0.9;
  **χ² profile** fixes one value at a grid of values, fits the others again and gives the Δχ² = 1 interval (**Run
  Profile**, **Stop**); **Fit report** shows the report — data file and its SHA-256, q range, model, every value
  ± error with its status, χ²ᵣ, the correlated pairs, the settings — as Markdown, HTML or text, to **Copy** or **Save
  Report…** ([Physics and Methods, section 9.16](Physics_Manual.md#916-uncertainty-after-refine-and-search)). A value
  the curve does not determine (its error not finite, or above 10× the value) is marked ⚠ in the table and the report.
  **Posterior check (MCMC)** runs a short Markov chain Monte Carlo on the posterior of the solution in Model — a
  solution of Predict parameters (its own posterior, with the model error) or the last Refine or Search (its
  likelihood) — in the background (**Run Check**, **Stop**; **Steps**, by default 3000, **Walkers**, **Seed**). The
  table sets the run's median and 68 % / 95 % intervals beside the intervals checked (Predict's, or the fit's ± 1σ),
  with each value's autocorrelation time τ and R̂; **Trace of** draws one value step by step; the line under the table
  says whether the run is long enough (*Converged: … ≥ 50 × the longest autocorrelation time …, R̂ ≤ …*) and marks ⚠ a
  value whose interval does not overlap the one checked; **Save Check…** writes it all as JSON. It is a check only:
  nothing in Model, Results or Predict's values changes
  ([Physics and Methods, section 9.19](Physics_Manual.md#919-posterior-check-by-mcmc)).

  ![Posterior check](manual_images/en/27_fitting_mcmc_check.png)

  *The Posterior check tab after a Refine: the run's intervals beside the fit's ± 1σ, τ and R̂, the verdict, and the trace of R (three walkers, the 16–84 % band and the median).*

- When there are several solutions (of Predict parameters, or of Analyze), the drop-down above the parameter table
  switches the entire solution: parameters, errors, total curve, components and residuals together. It does not
  combine individual parameter endpoints from unrelated solutions. The exported record identifies the selected
  solution. Each entry gives the solution's number and name — numbered (*Sphere #1*, *Sphere #2*) when two have the
  same composition — its χ²ᵣ, p and flags in short (*⚠ at bound: Tail ν*; their sentences on hover), and
  *incomplete* for a solution kept when Predict was stopped. After **Refine These Values** the first entry is that
  fit (*Refined from solution 1 (Refine) · χ²ᵣ 1.04*), and the quality box says that the values come from a new
  refinement of solution 1, not from Predict; the solution's own values and intervals stay with it below.
- Right-click the curve: **Show components** and **Components** control each population, constant background
  and low-q term separately. This only changes visibility; it does not change the fit or its scientific arrays.
- The values, each ± its 1σ error (the covariance at the solution, scaled by χ²ᵣ). With a solution of Predict
  parameters in Model and no fit since, ± is half the width of the solution's 68 % posterior interval (the tooltip
  gives the interval, the column header says which ± it is); the scales and the background show **–**: Predict solves
  them for every sample, they have no interval. Select values and press
  **Fix**, **Let It Fit** or **Range…** (or right-click): the next **Fit Again** uses them. **Range…** shows the
  value's min – max in Model. A free value the fit gives no error for shows **–**; its tooltip says why (at a bound
  of its range, the Scale of its population or Tail A at 0, or not constrained by the curve).
- **Other solutions** (folded): the solutions of *Predict parameters*, each with its χ²ᵣ and its probability **p**
  (the column names' tooltips define both; the row's tooltip says how many points χ²ᵣ is of). R and D show every
  population, in the order of the radii (`3.06 / 6.02`, `49.4 / –`); two solutions of the same composition say in
  their tooltip in which values they differ (σnn/D, σ_q …). A solution from **Show in Fitting** shows the χ²ᵣ of
  Analyze's fit (section 5.5) until you fit it.
  **Use This Solution** puts one into Model (Undo brings the previous model back). **Export Solutions…** (also
  **Save ▸ Solutions…**) writes every solution: `<curve>_solutions.csv` (a row per solution and parameter: the
  composition, probability, χ²ᵣ and its points, flags, whether it is incomplete, and the value, posterior median,
  68 % and 95 % intervals and least-squares error), `<curve>_solutions.json` (the record of Save Data and Fit with
  every solution under `predict`) and `<curve>_solutions_curves.csv` (the points and every solution's model,
  residual and terms).
- **Save Data and Fit…** writes a CSV with q (nm⁻¹ and Å⁻¹), I, the σ used in the objective, the model,
  the residuals, dQ in nm⁻¹ when file resolution was used, and each model term. It writes a
  JSON record next to it: the model, the fit (errors, values at a bound, correlations) and the
  points used; after Predict parameters also every solution (`predict`: composition, probability, χ²ᵣ, flags and
  every value with its median and intervals). Choose **NXcanSAS (*.h5)** as the type for an HDF5 file with two SASentry groups, the points fitted
  (Idev = the σ used for fitting, Qdev = dQ when used) and the model, Q = |q| in 1/nm. The record is next to the
  file and inside it. **Save Plot…** writes the plot and residuals as a PNG, SVG or PDF on white, including
  measured error bars with file-σ weighting, the visible terms and the shaded fitting range. **Save Model…** writes the model as JSON, to load for another
  curve with **Load Model…**.
- **Run Log** (folded) at the bottom.

What to check:

- A fit belongs to the curve and points it was made on. When you change the halves, the fitting range or the
  left-out points afterwards, the Fit and Results steps say *Changed after the fit: Fit again*.
- Opening another curve stops a fit that is running; a fit that ends after that is not kept, and the status line
  says so.
- A save that fails (for example on a read-only file) leaves an earlier file as it was and says why.

The model, the method and the choices (**Particles** too) are kept for the next session, and the last curve opens
again. A different curve arriving during startup takes precedence.

### 6.3 Fit methods

One **Fit** button runs the method chosen in **Method**. Fitting has one network method, **Predict parameters**,
and two least-squares methods that work on Model as it is:

| Method | What it does | When to use it |
|---|---|---|
| **Predict parameters** | GIMaP's network proposes compositions and values; Fitting's model weighs them against the curve (a posterior) and each solution is refined by least squares. Every solution has its probability and the 68 % interval of each value (section 6.4). | A GISAXS cut whose particles or sizes are unknown, with how probable each solution is and how well the curve fixes every value. Chosen for an unfitted GISAXS cut. |
| **Refine the values in Model** | Bounded least squares from the values in Model, with Fitting's own model; stays near them. | The model is close: after a prediction or a solution, after Analyze, or to get errors. |
| **Search the ranges, then refine** | Differential evolution across the ranges of the free values (log scale when a range spans 10× or more), then refinement from the three best starts. | The start is poor but you know the family and sensible ranges. Slower. |

- The scales (every particle's scale, the background, the peak A) are solved within their limits (bounded weighted
  least squares) at every step. The search therefore moves only sizes, spreads, distances and the peak's shape.
- **Stop**: *Refine* and *Search* keep the best values so far. A stop while results wait to be displayed remains
  a stopped outcome; it is not reported as convergence. Stopping *Predict parameters* produces no new result.

**Values at the edge of their range.** Model's fixed values and min–max ranges are hard limits for Refine and
Search, including population scales and global terms: they never widen them or fix a free value automatically, and
there is no option to do so. A value that stopped at an edge of its range is named under **Fit details…** (*…
stopped at a bound of its range: widen the range or fix it.*): change its range in Model (**Range…**) or fix it,
then **Fit Again**. Predict parameters does not use Model's fixed values and ranges: its network looks at the whole
range it knows, and a value at a bound of that range is flagged (section 6.4).

### 6.4 Predict parameters and 1D Predict

**Predict parameters** finds the particles of a cut and their values from the curve. It is Fitting's one network
method. GIMaP's network, trained on simulated cuts of Fitting's own model, proposes compositions and values; the
model weighs them against the measured points (a posterior), and every solution is refined by least squares. Each
solution comes with how probable it is and with the range in which each value lies (its 68 % interval). The
intervals and probabilities allow the model an error of 5 % of the intensity at every point, since Fitting's model
describes a measured curve only to a few per cent; the fit itself and its χ²ᵣ use the curve's own σ. The network
(about 13 MB) comes with GIMaP and runs with numpy: nothing has to be installed.
[Physics and Methods, section 9.13](Physics_Manual.md#913-predict-parameters) describes the method and its
benchmarks.

- **For the curve open in Fitting**: the method **Predict parameters** of the Fit step (below).
- **For a list of curve files**: **Tools ▸ 1D Predict — Fit Many Curves…** (or **1D Predict Batch Window…**
  under *Options* in the Fit step) opens the 1D Predict window (further below). To fit every curve of a folder with
  your model, use the In-situ series instead (section 6.5); its **Predict each frame** runs Predict parameters on
  every frame.
- Analyze's **Run Automatic Analysis** (section 5.5), Batch Export's **Particle model of the horizontal cut**
  (section 5.7), **Process with AI** (chapter 10) and the command line fit a GISAXS cut with Predict parameters too.

**What to enter in the Fit step**

1. In **Fit**, select **Predict parameters** (preselected for an unfitted GISAXS cut). Under it, **Particles**
   chooses what it tries:
   - **The particles in Model**: the 1–3 particles of Model — their families, and whether each has the paracrystal
     (**Distance D**). The populations are taken in the order of their radii. With no particle in Model, Predict
     takes **Spheres and upright cylinders (1–2)**.
   - **Spheres and upright cylinders (1–2)** or **Any particles (1–3)**: the network proposes the four
     compositions it finds most probable among these, and its most probable of every number of populations; their
     probabilities decide.
2. The cut's geometry comes from Model's angles (αi, αf and λ under *Grazing incidence*): with both, spheres have the
   direct and the exit-reflected path, and Predict fits the effective reflection coefficient r with the rest (it is
   then in Model, *Spheres in grazing incidence*); with the angles 0, q_z = 0 (the effective interface, Physics and
   Methods, sections 9.10 and 9.13). Enter the angles for spheres of radius above about 3 nm (q₊R above 1.7):
   without them the radius comes out 10–40 % off on a real substrate, with them about 6 % (median, on simulated
   curves at the Yoneda angle). On a cut well above the critical angle the reflected wave is weak, and a solution
   with a large |r| and a sphere about half the size can describe the curve as well: there fix |r| (`refl_amp`) to
   the substrate's value, or to 0, and refine. With both angles Predict takes about 2.4 times as long.
   Predict uses its Gaussian size model for the three manual number distributions, its own q resolution and the
   curve's measured σ. Earlier models retain their existing Predict size-law and coupling conventions.
3. Without σ in the file, Predict takes a relative σ from the scatter of ln I about its trend (at least 1 %); the
   Curve card names its value, and the 1D Predict window uses the same estimate.
4. Press **Fit**. It runs in a separate process. The line under the bar says the step — *Step 1 of 3 · Network
   proposals: composition 2 of 4*, *Step 2 of 3 · Refining the modes …*, *Step 3 of 3 · Sampling the posterior:
   solution 2 of 3* — the time since the start and, once an item of the step is done, about how long the step still
   takes. **Stop** ends it at once: the solutions whose posterior was sampled before Stop are kept, marked
   *incomplete*, their probabilities over those alone (none sampled: nothing changes). For a curve of a few hundred
   points on one core it takes some seconds with **The particles in Model**, from some tens of seconds to a few
   minutes with **Spheres and upright cylinders (1–2)** and a few minutes or more with **Any particles (1–3)**,
   whose random cylinders are slow (20–30 s per least-squares fit). The time grows with the number of points and
   of compositions tried, and depends on the computer.

There is one way through, and nothing else to set: the network's 1000 samples of each composition are weighted
against the curve; the most probable values of each composition are refined by least squares; when none of them
refines below χ²ᵣ 2, the ranges are searched for the two best compositions; then the posterior is sampled around
the solutions within a BIC of 20 of the best (at most four), which gives their probabilities and intervals.

Model's fixed values and ranges are not used by Predict: the network looks at the whole range of every value. The
solution that comes into Model keeps Model's geometry; **Refine** afterwards keeps the values you fix.

**Reading the result**

- The solutions go to **Results ▸ Other solutions**, most probable first, with χ²ᵣ and their probability **p**; the
  most probable is in Model, and **Method** turns to **Refine the values in Model**. The probabilities compare the
  compositions tried and the distinct solutions of each: a solution with p 0.6 next to one with 0.4 means that the
  curve does not decide between them. With two or three populations this is common: another R of one population at
  the same D, or the paracrystals of two populations exchanged, can describe the curve as well; Predict tries these
  too and lists them. On simulated curves the true solution was among those listed in 95 % (one or two populations)
  and 87 % (one to three) of the cases: look at the others before you report one. Probabilities are relative to the
  compositions that were tried: with **The particles in Model** only one is tried, and its single solution shows no
  probability (*Only this composition was tried*).
- χ²ᵣ = Σ((I − model)²/σ²)/(N − number of free values), with the curve's σ, over the points Predict fitted. It
  leaves out the points below the network's q window, and the status line, the table's tooltip and Fit details say
  how many (*χ²ᵣ 1.04 over 596 of 598 points; 2 below the network's q window*). The same number is shown wherever
  the solution appears. A **Refine** afterwards fits every point of the fitting range and says so when its χ²ᵣ is
  not of the same points.
- **Fit details…** lists the values in Model with their 68 % intervals: the refined value first, then its
  interval, e.g. `R 4.12 [4.08, 4.16] nm`, and the probability. When a refined value lies at an end of its interval,
  a line says so (*the refined value lies at the edge of the posterior: the model describes this curve only within
  the model error*). Below it, what to keep in mind, one line per kind of remark: a value the curve says little
  about (its interval spans most of its allowed range), a value at a bound, strongly correlated values, few
  effective samples (the intervals are then rough), that the values are those of a fit with the model error added to
  σ, or that the solutions were compared by χ² and their number of values (BIC) because the posterior's sampling
  did not converge — usually a curve the model does not describe within its errors; the intervals are then the
  least-squares errors. The remarks about the background, the beam tail and the resolution follow in one paragraph
  (*Background, beam tail and resolution: …*), and a term that is absent (Tail A = 0) is not flagged. The values
  are named as in Model (Tail ν, σ_q, σR/R (1): the population's number in brackets). A population whose size
  the curve does not show is named too: one whose Guinier region lies at or below the lowest q (the curve shows only
  its decay, which roughness or a rise at low q produce as well: check it before you report particles of that size),
  or one smaller than the curve resolves (a smooth background). A sphere or random cylinder fitted at q_z = 0 (no
  angles in Model) whose q_z·R at the cut is above 1.7 is named as well: its radius is then biased, usually too
  small; give both angles. GIMaP can say this only where it knows the cut's angles, as in the automatic analysis
  (section 5.5): a curve file does not carry them, so a Predict here without angles in Model does not flag it.
- **Refine These Values** (or **Refine the values in Model**) then gives the least-squares errors under Model's own
  limits, as for any model.

The network knows cuts spanning a factor 10–300 in q (points below q_max/300 are left out, and said), one to three
populations, and radii from 0.5/q_max to 5/q_min.

**The 1D Predict window**

![The 1D Predict window](manual_images/en/13_1d_predict.png)

*The candidates of every curve on the left (with their Stage); the selected candidate's fit and parameters on the right.
The second line under the buttons, “q in Å⁻¹, as their header states”, says how the three Analyze curve files
(`_fit_input.dat`) are read, with no unit set by hand; **Curve / side** repeats it after each name (“· Å⁻¹”).*

The window runs Predict parameters on every curve listed; it has no other method.

1. **Add curves…** lists text curve files (`q I [σ]`; `.dat`, `.txt`, also comma-separated `.csv`). **Use current
   cut** takes the curve open in Fitting instead. The line under the buttons says in which unit of q each file is
   read: the unit its header states or, for a file whose header states none, the setting **q unit of files without
   one in their header**. A file whose header states both Å⁻¹ and nm⁻¹ is not fitted and is named there.
2. **Parameters…** sets:
   - **Complete composition**: up to three boxes, the particle families when you know them (each population with
     the paracrystal); with every box on *Auto / unused* the network chooses among all compositions.
   - **q unit of files without one in their header**: Å⁻¹ by default, as on the Fit page (section 6.1; settings
     saved with nm⁻¹ keep it). A file whose header states its unit of q is read in that unit, whatever is set here;
     Analyze's `_fit_input.dat` files state Å⁻¹ and need no setting. A header states the unit with `q in <unit>` in
     a `#` line or in the name of the first column (`q (1/A)`, `q_nm^-1`, `# columns: qy (1/A) …`,
     `# q [1/nm]  I`). **Use current cut** is converted automatically. When a file states no unit (section 6.1), a warning under the status line names the file, the assumed
     unit and how to change it; the candidate's tooltip and its *Parameters and units* text repeat it, and the
     records keep it (`curve_notes`).
   - **q sides**: *Each half separately*, *q > 0 half* or *q < 0 half*.
   - **Cut qz (nm⁻¹)**: the cut's q_z, at whose Born term spheres are evaluated; unticked (the default), q_z = 0,
     the effective interface. The exit-reflected path needs both angles: fit such a curve on the Fit page with the
     angles in Model.
   - **Relative σ added to counting σ** (10 % by default) and **Absolute σ added to counting σ**: only for **Use
     current cut** on a curve Analyze wrote with the pixel counts of its detector columns; they are added in
     quadrature to the counting σ. A curve without σ gets the relative σ of Predict parameters, estimated from the
     scatter of ln I about its trend (the same value as on the Fit page; its points with a negative or zero
     intensity are left out); each candidate records it (`sigma_source`, `sigma_relative`, `curve_notes`).

   The files are read by the reader of the Fit page: any header lines (also a title line above the column names),
   commas, tabs, semicolons or spaces. Negative and zero intensities are counted in each candidate's `curve_notes`.

   **Save Settings** keeps them. Settings saved by an earlier version with *General V5*, the *Single RC
   specialist*, the *Physical fit* or *fit1d* open as Predict parameters: their own options are dropped, and fit1d's
   cut qz is kept as **Cut qz**.
3. **Fit curve** (**Fit N files** when several are listed) starts, **Cancel** stops (finished files stay saved),
   **Open results** opens the run folder.
4. The table lists the candidates of every curve and half: **Curve / side** (the file and + or −), **#**,
   **Components** (S sphere, RC random cylinder, VC vertical cylinder), **logRMSE**, **RMS/σ** and **Stage**; for a
   file, **Curve / side** also gives the unit its q was read in (hover to see where it comes from).
   Select one to see, in the tabs on the right, its **Fit plot**, its **Parameters and units** and the **Batch log**.

The **Stage** of every candidate is *Network + posterior* (Predict parameters); hover for its probability and
flags.

**Where the results go.** Each run of the window writes a new folder `AI_Fitting_Output/v5_<date>_<time>_<µs>/` in the GIMaP user data folder (also for the portable version). **Help ▸ Open User Data Folder** opens it. It holds `request.json`, `batch_summary.json`,
`top20_candidates.json`, `fitting_run_summary.json` and, for every curve and half, `solutions.json` (parameters,
units, the stage that ran, the probabilities and intervals), `display.npz` and
`fitting_curves.csv`. `request.json` and `fitting_run_summary.json` list the unit each curve's q was read in
(`q_units`); each file's entry in `batch_summary.json` and its candidates in `top20_candidates.json` carry it as
`q_unit_read` and `q_unit_from`. The folders are not removed automatically. Fits you save yourself (**Save**) go
where you choose.

**The network file.** The weights of Predict parameters are in `src/gimap/features/fitting/domain/posterior/data/`
(`flow_v1.npz`, with its record `flow_v1.json` and model card `MODEL_CARD.md`), both in the portable version and in
the Git repository. Predict checks them against the SHA-256 in the record before it uses them; with a changed or
missing file it does not start, and the message names the file (section 14.3). The earlier networks of 1D Predict,
*General V5* and *Single RC specialist*, were removed in 0.2.0 (Physics and Methods, chapter 10, gives why). So were
the fit1d kit (*Predict parameters (fit1d)*) and the quick physical fit (*Find the particle shape (no AI)*, the
*Physical fit* of this window): Fitting has one network method, so there is one choice to make.

### 6.5 In-situ series

![Fitting, In-situ series](manual_images/en/12_fitting_insitu.png)

*The Results step on the left (the halves and points per frame, then every frame's χ²ᵣ and values ± 1σ); on the
right the selected frame's fit and, below it, the trend of R through the series, coloured by stage.*

The **Batch / In-situ series** tab fits curves with the model of Single analysis. The command bar has
**Import Curves…**, **Choose Folder…** and **Start**; **Pause** and **Stop** appear while fitting runs,
**Save** once frames are fitted. **Import Curves…** supports selected files and shared-q tables (section 6.1).
The folder and time-series controls below apply to folder mode and ordered sequences; independent imported
samples use the same initial model and have no stage detection, odd-frame exclusion or continuity.

Folder discovery, unit-header inspection, previews, detector-frame cuts and watching run in the background.
While the initial list and preview are prepared, **Start** is unavailable and **Stop** cancels preparation.
If **Stop** is pressed while a curve or frame is being read, the request is recorded immediately; when that
read returns, fitting is skipped and the cancelled frame is not counted as a failed fit. **Pause** lets the
current frame finish before pausing. Completed results stay available.

**1. Curves**

- **Choose Folder…** (or drop a folder on the page), **Fit Every Curve in This Folder…** in Single analysis'
  Results, or in Analyze list the frames and choose **Send Series to Fitting…**: every frame is exported
  (optionally summed N at a time) into `<name>_fitting/`, and the series opens here with its curves listed.
- The frame range and halves controls wait for listing to finish. The start strategy, method and follow choice
  can still be changed; saving a project during preparation keeps those choices.
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
- Folder curves are listed in counting order of their names (`frame_2` before `frame_10`) and numbered 1, 2, 3 … in
  that order. **Frames**: first – last of these numbers, and **Every** n-th.
- **The unit of q**: every curve file is read in the unit its header states, else in Single analysis' **q unit of
  files without one in their header** (its value at Start, kept for the run). The step counts the listed curves by
  unit and where it came from — *2 curves: q in Å⁻¹, as their header states*; *1 curves: q in nm⁻¹, no unit in
  their header (Single analysis ▸ Curve ▸ File)* — and names a file whose header states both units: it is not
  fitted, and its frame says why. Detector frames are cut in Å⁻¹.
  Imported selections instead keep the fallback unit chosen in the import preview; each file's header still wins.
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
  Analysis** changes them) and the halves of the Curves step. Every method preserves Model's fixed values and
  limits; values at a boundary are reported without widening the limits (section 6.3).
- **Each frame starts from**:
  - **The previous frame's result** (default) — for a slowly changing sample. A frame that did not converge
    makes the next one start from Single's model again.
  - **The model in Single analysis** — every frame independent of the others.
  - **The previous result; the Single model at each new stage** — follows the sample within a stage and starts
    afresh where the curves change course.
  - **The previous result; Predict parameters where the chain breaks** — each frame from the result of the frame
    before; where that chain breaks (the fit failed or did not converge, a new stage begins, or χ²ᵣ is more than 3
    times the frame before's and above 2), Predict parameters (section 6.4, its method unchanged) fits the frame
    again with the particles of Single's model, from its solution closest to the frame before among those at least
    a tenth as probable as the most probable one, and the better fit is kept. If the manual size law, weights or resolution differ, Refine uses the network values as a
    starting point and applies the manual settings before comparing the two fits. A value at a bound (the beam tail, the
    background or any other), a value the curve does not determine and a poor fit are flagged but do not break the
    chain, so a series whose curves change smoothly stays smooth. A model with
    no particle or more than three is refined only (the Run Log says so).
- **Then a backward pass**: once every frame is done, from the last but one to the first, each frame is refined from
  the result of the frame after it, and the better of its two fits is kept — a sample whose first frames are hard to
  fit is then followed back from where the fit is sure. Not while **Watch for new curves** waits for curves.
- **Method**: **Refine (fast)**, **Search the ranges, then refine (slow)** or **Predict each frame**.
- **Predict each frame** runs Predict parameters (section 6.4) on every frame, with the particles of the frame's
  start model (1–3 particles: their families and paracrystals), then refines the solution kept with least squares
  under the start model's fixed values and ranges; the table holds those values and their errors. **Prefer the
  solution that continues the previous frame** (on by default): of the solutions at least 0.2 times as probable as
  the most probable one, the frame keeps the one closest to the previous frame's sizes and distances (the smoothest
  series); a clearly more probable solution is still taken. It does not apply when every frame starts from Single's
  model, nor at the first frame of a new stage.
- Then **Start** in the command bar. **Pause** and **Stop** act after the frame being fitted; Stop during
  reading skips that frame's fit as described above. The frames done are kept. A project cannot be opened while a series runs (Stop first). A run started before the stages were found
  uses none (it says so).

**3. Results**

- How many frames were fitted, failed or did not converge, with the halves and points per frame. A frame whose beam
  tail stopped at an edge of its range, or with a value without a finite error, is **flagged** (marked in the list,
  counted at the end of the run and listed under `flagged` in the record).
- A table with a row per frame: χ²ᵣ, each free value ± 1σ, **Started from**, the **Flags**, the **Points** fitted
  and the **Halves**. Click a row to see that frame's curve and model; a failed frame shows its reason. **Started
  from** says where the frame's kept fit started: *continued from frame n* (the frame before; *backward pass*: the
  frame after), *restarted from Predict (why the chain broke)* — the rule that chose Predict's solution is on hover —,
  *… (the chain broke: why)* when the continued fit was still the better one, or *from the model in Single
  analysis*. **Flags** says what to know before trusting the row: failed, not converged, a poor fit, values at a
  bound of their range, the beam tail at an edge, a value the curve does not determine.
- Below the frame plot: one value against the frame (or χ²ᵣ), with ±1σ error bars, in the colours of the stages.
  The plot is titled with the name of the series. When the curves say when their frames were measured (curves for
  Fitting that Analyze wrote of frames with a time stamp), the box beside the value draws it against **Time (s)**
  since the first fitted frame.
- **Open Frame in Single Analysis** takes a frame and its fitted model back to Single analysis to look closer.
- **Save ▸ Table of Every Frame…** writes a CSV (frame, file, χ²ᵣ, log RMSE, converged, points, halves used, every
  value and its error) with a JSON record next to it: the start model, the halves and where they came from, range,
  left-out points, method, every frame's state, reason and the unit its q was read in (`q_unit`, `q_unit_from`;
  `q_unit_of_files` is the setting for files without one), the poor fits (χ²ᵣ above 10) and which curves or frames
  were listed. The table ends with `flags`, `start_from` (previous, next, single or predict), `start_frame`,
  `chain_break`, `predict_choice` and `fit_pass`, and the record says the start and the backward pass. **Fits as NXcanSAS…** writes, for every fitted
  frame, the points fitted and the model at their q (one `.h5`, two entries per frame, each with the frame's
  record and its role, data or model; Q in 1/nm). **Results Database…** writes every frame's fit — state, χ²ᵣ, values and errors, flags, where
  it started — into `gimap_series.sqlite`; one in the folder of the curves or up to two folders above it (Analyze's,
  next to its curves) is offered first. **Trend Plot…** and **Selected Frame's Plot…** save the figures.
- Before results that are not saved are dropped — a second Start, other curves, a project opened — GIMaP asks:
  **Save**, **Discard** or **Cancel**.
- A series of GIWAXS I(q) curves gets the note of section 6.1 again after the run, with how many fits are poor.
- **Run Log** (folded) at the bottom.

A curve that cannot be read or fitted is marked ✗ in the list, and the series goes on. The folder or exact imported
selection, its units and choices are kept for the next session and in a project. Mapped sources keep their column
indices and names in result records; indices in JSON are zero-based, while the preview labels columns from 1.

### 6.6 Several curves fitted together

![Fit Several Curves Together](manual_images/en/26_fitting_joint.png)

*Two cuts of one frame fitted together: the vertical cut (right, its Yoneda peak and the height's minimum) holds the cylinders' height h, shared with the horizontal cut; every curve its own scale and background.*

**Fitting ▸ Fit Several Curves Together…** fits a list of curves with one χ²: typically the horizontal and the vertical
cut of one GISAXS frame, whose vertical cut sees what the horizontal cut does not — the height of upright cylinders and
the vertical extent of the particles. The model is the one in Model of Single analysis when the fit starts.

- **Curves**: **Add Files…** (curve files), **Add Curve of the Page** (the curve in Single analysis, with the angles of
  its Model) and **Remove**; **Analyze ▸ Send Both Cuts to Fitting** writes the frame's horizontal cut
  (`<stem>_fit_input.dat`) and its vertical cut (`<stem>_fit_input_qz.dat`, next to the data in `gimap_analysis/`, or
  where you choose when that folder cannot be written) and lists both here with their geometry. Under the list, the
  selected curve's **Kind** — *Horizontal cut* (αi, αf, λ), *Vertical cut* (αi, λ, **q∥** of its columns and, when
  Analyze refracted its q_z, **αc of q_z**) or *Other curve (q_z = 0)* — and its values.
- **Substrate** (all curves): a vertical cut is computed in the four-term DWBA and needs it (Si, SiO₂, Glass (plate),
  Au or **Typed δ, β**); a horizontal cut with both angles is then in the four-term DWBA as well.
- **Parameters**: every free value of Model with **Shared** (fitted once; by default every particle value) or not (each
  curve its own value). The scales — every population's Int, the background, Tail A — are always each curve's own, so
  every curve has its own scale and background. When a vertical cut is listed, upright cylinders get their height h and
  σh/h, shared.
- **Fit Together** with *Refine* (from the values in Model) or *Search the ranges, then refine*; a progress bar and
  **Stop** (the best values so far are kept). The summary gives χ²ᵣ over all points and each curve's Σr²/N; **Show**
  picks the curve drawn with its model. **Use in Model** puts the values of the first horizontal cut (shared values
  included; a height fixed at the joint value) into Model; **Save Record…** writes every curve with its model, errors,
  the shared values, χ² and the covariance (JSON).

**Fit Several Curves Together** warns when the exported vertical band crosses qy = 0 or its centre has |qy| below 0.02 nm⁻¹. The particle model excludes the specular reflection. If the band includes it, move the cut beside the ridge and send both curves again. The vertical-cut model and its check against BornAgain are described in
[Physics and Methods, section 9.18](Physics_Manual.md#918-several-curves-fitted-together-the-vertical-cut).

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
- **Fitting ▸ Batch / In-situ series** (section 6.5): the listed curves are compared as they will be fitted (with the
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

- **Add Series** adds **The Series Map of Analyze**, a **Folder of Curves…** (one series), **Curve Files…**
  (together one series) or a **Results Database…** (`gimap_series.sqlite`, written by Analyze's Series tab or a
  watched folder: one curve of one series, without its frames; a question asks which when it holds several). Curve files need q and I columns (`.dat`, `.txt`, `.csv`, `.xy`, `.chi`; at least two
  per series, in counting order of their names: `frame_2` before `frame_10`). q in nm⁻¹ is converted to Å⁻¹
  when the header says so.
- A folder written by Analyze (a Batch Export folder, its `curves/`, or `gimap_analysis/`) holds several kinds of
  curves. One kind is one series: when there are several, a question asks which one (the curves for Fitting are
  named *the curve for Fitting*, files that are not Analyze's *other files*), and the series is named
  *`<sample or batch>` · `<curve>`*. A Batch Export table of all frames (`<name>_<curve>_frames.csv`) can be added as
  a series too.
- A folder or curve files can also be dropped on the page. The same map or folder is not added twice.
- Folder discovery and curve reading run in a background queue; folders are added in the order requested.
  The job panel shows progress and **Cancel**. Cancelling removes unfinished series and queued imports;
  fully read series already added stay. A cancellation is registered immediately. An active directory scan
  or single-table read must return before its result is discarded; reading separate curves stops between files.
  Opening a project uses the same queue for its curves and sidecar arrays, with read errors shown here.
- Double-click a name to rename the series. **Remove Selected** (or `Delete`) and **Remove All** (the note that
  follows offers Undo).
- Long names are shortened on the plots and in the results to the part that differs between series; tables,
  CSV and JSON keep the full names.

![Compare import progress](manual_images/en/14a_compare_import.png)

*Importing 60 synthetic curves: the window stays responsive and Cancel stops unfinished imports.*

**2. Compare**

- The **q range** compared (Å⁻¹; named after the axis of the series, for example **qy range**): the range every
  series covers. Narrow it to leave out a noisy edge or a detector artefact; **Whole Range** goes back.
- **Compare the shape only** (on by default): each frame's mean level is removed, so a brighter beam, a thicker
  film or a longer exposure is not a new structure. Off: the overall intensity counts too.
- **End state**: the mean of the last N frames of a series (10 by default; odd frames left out).
- Every change compares again after a moment, in the background. When a comparison fails (for example no common
  points in the range), Results says why and what to change.
- Changing the series, q range, shape-only choice or end-frame count immediately hides the old results and
  disables table and plot exports. They return when the current settings have been compared. Returning to the
  settings of an existing result restores it immediately if the series have not changed. An import that adds
  no series keeps the existing comparison; new data are compared after the queue finishes.
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
- A project keeps the settings and the series: curve files by path, embedded arrays in
  `<project>.<generation>.compare.npz` (section 3.8). Finish or cancel pending imports before saving the project.

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

**Project files.** File ▸ Save Project saves this workspace's form (the input bar, the geometry and the options) and the references to the result folders shown. Opening a project restores them; an older project without Sputter clears this workspace. Keep result folders beside their inputs when moving a project; missing folders are reported.


The **Sputter** workspace (sidebar, `Ctrl+5`) follows an in-situ sputter deposition: the PILATUS CBF frames of
one scan, taken while a metal is sputtered onto a substrate. Under its input bar it has two views, chosen with
**Main Screen | Details** at the right of the bar:

- the **main screen** (shown first, section 8.1) shows the result of **Analyze**: for each time window the measured
  image beside the image of a model, and the grains that model stands for — radius R, height H, distance D, gap,
  size distribution width, coverage and, when there is one, the thickness of the continuous metal layer under the
  grains — with their evolution over the scan. Analyze runs a separate analysis package;
- **Details** (section 8.2) holds the geometry, the options of Analyze and of samples measured after their
  deposition, the room the frames Analyze imports take (**Storage**), and five tabs about the result the main screen
  shows: **Windows**, **Growth Process**, **Report**, **Logs** and **Steps**.

What the main screen shows is in [Physics and Methods, section 12.1](Physics_Manual.md#121-analyze-what-the-main-screen-shows).

### 8.1 The main screen: Analyze

**What you need.** Nothing more: the analysis package comes with GIMaP (`utils/sputter_client` of the program
folder; a folder with `code` and `models` that runs in GIMaP's own Python, with NumPy, SciPy and Matplotlib, no
PyTorch). To use another package, choose its folder with **Choose Folder…** (**Details ▸ Analyze options ▸
Analysis package**) or set the environment variable `GIMAP_SPUTTER_CLIENT` to it; both take precedence over the
one that comes with GIMaP. A folder named so that is not an analysis package (moved since, or the wrong folder) is
passed over: the package that comes with GIMaP runs, and the **Analysis package** line, marked, says why in its
tooltip. Without any package the main screen says so; **Open Result…** shows a folder analysed before all the
same.

**The input bar**, from the left (with **In Situ** chosen at its left end; **After Deposition** is for samples
measured after their deposition, see *Samples measured after their deposition* below):

1. **Choose…**: the folder of one scan (its CBF frames, or a scan folder with them in one subfolder such as
   `p2m/`). A line under the bar then sums up the frames, the detector, the exposure and the time.
2. **Metal** (Ag, Au, Cu) and **Substrate** (Si, TiO₂, PS, PET, PSBMA, PNO, PEO, Polymer). **Polymer** is a
   polymer film of another or unknown kind, a block copolymer for instance: the model describes every polymer as one
   film layer, and which polymer it is changes the metal's sizes by a few percent; Ag and Cu run on every polymer. The
   package has a network for some pairs only: for another pair Analyze stops at once, and **Details** of the job
   status shows the package's sentence that names the pairs it has.
3. **Rate**: the deposition rate, which is needed, in the unit chosen beside it: **Å/s** (the default — what a
   quartz balance shows) or **nm/s** (1 nm/s = 10 Å/s). Tick **QCM reading** when the number is the reading of the
   quartz balance: the package takes 0.6 of it as the rate at the sample. Unticked, it is the rate at the sample
   itself. The unit is remembered with the rate for each scan folder. Once the header of the scan is read, the line
   under the bar adds the nominal thickness the rate reaches at the last frame (rate at the sample × the time from the
   start of the deposition to the last frame, or to its end when a duration is given). A rate at the sample above
   1 nm/s or below 0.005 nm/s is unusual for sputtering: before Analyze runs it asks whether the unit is right
   (**Analyze Anyway** or **Cancel**).
4. The line below says the geometry that will be used — energy or wavelength, distance, incidence angle, beam
   centre, sample roll — or which of them are still to enter. **Edit** opens Details at these fields (section 8.2
   says where their values come from). An empty beam centre is estimated from the frames; the sample
   roll is 0 unless you enter one, and without a **Mask** the package uses the pixel mask its network for this metal
   and substrate was trained with - the P03 mask of 2021 for Cu on PEO, of 2025 for the others (both are under
   **More options**, section 8.2). On top of that mask, or of the one you give, the package masks what the frames
   themselves show to be unusable: dead and hot pixels, the beamstop and its holder (the automatic mask; a mask of
   another detector size is not used, the automatic mask then stands alone). Hover over the name of the result to
   see the mask it was read with.
5. **Analyze**. The job status at the right says in one line what the analysis is doing now, and its bar shows the
   share that is done. **Cancel** ends it at once; frames that were read completely are kept for the next run.
   **Details** shows the package's messages. While the analysis runs, the main screen shows what it is doing and what
   it has found so far (the live view):

   - At the top, the same line across the view, with counts and, where the pace is known, the time still to go — for
     example *Reading frames: 400 of 1,200 · about 4 min left*, *Network cu_v4: drawing the candidates with the
     forward model, window 7 of 24*, *Comparing the networks: cu_p5 21.5, cu_v4 12.5 → cu_v4*, *Joint fit: iteration 2
     of 6 · median fit deviation 11.3 · 5:40 so far, about 11 min left* (seconds per iteration so far times the
     iterations left: the time if the fit takes all of them), *Drawing the main screen: window 5 of 20*. Between the
     events of the package it names the step (*Step 3 of 4: Writing the report…*). A line without a count of its own —
     a network starting, the network proposing candidates for every window, the joint fit preparing, a step — adds how
     long the run has been at it once that is 10 s: *Starting network cu_p5 (the first of 2 to compare) · 0:45 so
     far*. With a newer analysis package the
     joint fit also says how far it is inside an iteration, several times a minute: *Joint fit: iteration 3 of 6 ·
     derivatives: 12 of 20 windows · 6:12 so far, about 9 min left* (the derivatives are the long part of an
     iteration; the time left is seconds per window × the windows left in this iteration + seconds per iteration ×
     the iterations after it).
   - At the left, the four steps — **Read the frames**, **Find the solutions**, **Write the report**, **Draw the
     images** — each with its state and one line of what it found (frames read so far, time windows solved, the
     network that fits the data better, windows drawn), and the time since the start with the number of processes
     and how many of them are at work now (*Running for 03:12 · Processes: 16 · 14 at work now*; a process at work
     computes or reads from disk). While the solutions are
     found the line says what the network is doing and how far it has come: the network inputs prepared, the
     network proposing candidates, the windows solved and the model images computed, the windows refined (Refine:
     every window gets its candidates first, then each is refined), the iterations of the whole-scan refinement
     (Full). The bar of the job status moves on inside each step as well.
   - **The steps as a tree** (a newer analysis package): the four steps become a checklist of the steps and their
     sub-steps — reading the frames (beam centre, pixel mask, reading, start of the deposition), the time windows,
     each network, the comparison of the networks, the joint fit (preparing, iterations) and writing the result. ○ is
     still to come, ● at work — with the line of what happens now under it —, ✓ done, with what it found at the
     right (cut when long; hover for all of it), ✗ failed, – skipped. A step folds its sub-steps away once it is done;
     ▸ opens them again. A step that saved what it found ends in ›: click it to see it at the right — the beam centre
     and the pixel mask on the detector summed (masked pixels in red, the beam +, the sample horizon dashed), the
     frames read, the count rate with its baseline and the start of the deposition, the time windows, the fit
     deviation of each network, the joint fit's median fit deviation per iteration with its current solution (R, H, D,
     the continuous layer t_b and the coverage; it follows every iteration while it is open). **Back to the
     Analysis** closes it.
   - At the right, the step at work in one paragraph, with a picture and a plot. Reading the frames: the detector
     summed over sampled frames, with the pixels the mask leaves out in red, the direct beam (+) and the sample
     horizon (dashed), beside the frames read last on the grid of q_y and exit angle; below, the count rate of every
     frame read so far, which rises as the metal is deposited. Finding the solutions (and the two steps after it):
     the measured image of the window solved last beside the model image of its solution, its values in the line
     under them, and the radius, height, distance and coverage of every window solved so far.
   - **Reading from disk.** A newer analysis package also says how long its processes spend reading the frame files
     from disk: the line of **Read the frames** becomes, for example, *Reading 312 of 1,241 frames · disk 0.82 s per
     frame (85 % of the time)*, and keeps the disk part once the frames are read. When the disk takes at least
     0.15 s per frame and at least half of the time (or the package says so), a note above the pictures says that
     reading the frames from this disk is the slow part — a USB or network drive, or an on-access virus scan of every
     file — and that copying the scan folder to a local SSD first makes the import faster. More processes do not
     help then. With a fast disk, or an older package, nothing more is shown.
   - **When nothing happens.** When none of the analysis's processes has computed or read for a minute and the
     package has said nothing new, a note across the live view says so — the package has most likely stopped on an
     error inside one of its processes and will not go on by itself — with the last line of its log, and **Stop
     Analysis** ends it. The line at the top and the job status then say *No progress for 1:10: the analysis
     package's processes are all waiting*. GIMaP does not stop it for you: a disk that does not answer looks the same
     for a while.
   - A click on a step that has started shows what it found; a click on the step at work follows the analysis
     again. When the analysis ends, its result replaces the live view (after a failure or **Cancel**, what was shown
     before comes back). What the live view shows is not the result: the result is what the package writes at the
     end, with every candidate of every window.
   - **Preliminary result.** In mode Refine the network's result (its candidates and their time path) is shown on the
     main screen within minutes, its name followed by a note that the joint fit of the whole scan is running; the
     analysis goes on. A line under the window selector says what it does now (the same line as above); **Steps**
     beside it shows the live view with its steps again, and **Show the Preliminary Result** there comes back. After every
     iteration of the joint fit, its current solution is drawn dashed over the evolution — radius, height, distance,
     gap and coverage in their colours, named *Joint fit, iteration k of n* in the legend — and the next iteration
     replaces it. The tiles keep the values of the solution shown. When the analysis ends, the final result (which
     holds the joint fit) replaces the preliminary one and the dashed lines go.

Analysing the same request again shows the result it has. **Details ▸ Analyze options** has the rest:

| Field | What it is for |
|---|---|
| **Mode** | **Auto**: Quick for Si and TiO₂, Refine for polymer substrates. **Quick**: the network's candidates, each drawn by the physical model (about a second per time window). **Refine**: also refines the solution of every window (minutes per window on a polymer). **Full**: a longer refinement of every window and of the whole scan together (hours). |
| **Growth-law prior** | Whether the joint fit of the whole scan keeps the grain distance D and radius R near the growth laws D = D₁ h^p and R = R₁ h^q of the metal on this substrate before the film forms (h: the metal amount; *Physics and Methods*, section 12.1). **Auto** (the default): on for Ag on PS, where it passed a held-out test whose criteria were fixed before the runs; off for the other systems. **On**: for every system with a law in the package; for a system without one it changes nothing. **Off**: never. Only the joint fit uses it (Refine and Full; Auto picks Refine for polymers). For a sample that grows differently from the package's reference scans, choose **Off**. **Details ▸ Report** says whether it was used and whose laws; the export's `METHODS.md` gives the laws. |
| **Parallel processes** | How many processes the analysis runs at once, for Analyze and Static Frames. **Automatic (N)** (the default): one per core, two cores left free and at most 16, fewer when the free memory is short; N is worked out again whenever the box is shown and when the analysis starts. Or a number from 1 to the computer's cores. Kept for the next session. The live view and **Details ▸ Report** say how many ran. |
| **Start of deposition (s)** | The time of the scan at which the deposition started. Empty: found in the data. |
| **Deposition duration (s)** | How long the deposition lasted after its start; the metal amount stays constant afterwards. Empty: to the end of the scan. |
| **Polymer film thickness (nm)** | Shown for a polymer substrate. Empty: the package's own value for that polymer. |
| **Analysis package** | The folder in use, and **Choose Folder…**. |
| **Open Report**, **Open Folder** | The package's report of the result shown (every quantity with its range and where it comes from) in the program that opens Markdown files, and the result's folder in the file manager. The report is also in **Details ▸ Report**. |
| **Export…** | Also in the top bar beside **Open Result…**. Asks for a folder and writes `<scan>_export` into it: `evolution` and `patterns` as PDF and SVG (fonts embedded) and PNG at 300 dpi, `solution.csv` (the solution shown in each window — the joint fit, or the one you chose — with its fit deviation; units in the column names), `all_solutions.csv` (every solution of every window), `METHODS.md` (geometry, frames, mask, network, the joint fit's κ and Δt, the fit deviation's definition and median, the package — all taken from this result's records) and a README. Nothing is computed again; a package older than the export says so. |

**Reading the result.** The main screen shows one time window:

- **The window.** The slider, the arrows beside it or a click on the evolution plots choose it; beside the slider
  are its time in the scan and the nominal deposit at that time.
- **Measured image** and **Model image**: the count rate over q_y and the exit angle, on the same axes and with
  one logarithmic colour scale. The scale is set by the measured image, so it does not change with the solution;
  cells without data are empty in both.
- **Solution shown**: the solutions of this window — **Candidate 1** to **3** of the network and, when the
  analysis refined, **Refined** and **Joint fit (whole scan)** (every window refined together over the scan, smooth in
  time) — each with its *fit deviation*: how far its model image is from the measured one (lower is better; it
  compares the solutions of one window). The solution shown first is the one that a smooth evolution over the whole
  scan selects — the joint fit when there is one —, which removes most of the ambiguity of a single window; it need
  not be the one with the lowest fit deviation. Choose another one when its image matches better by eye: the values, the
  model image and the evolution follow, and the choice is kept with the result. **Back to Default** takes it back.
- **The values**: **Radius R**, **Height H**, **Continuous layer t_b** (the thickness of the continuous metal layer
  under the grains; 0 while they are separate islands), **Distance D** (centre to centre), **Gap** (D − 2R), **Size
  distribution width** (σ of a log-normal distribution of the radius) and **Coverage** of the solution shown. A
  quantity that the model of this sample does not have is not shown; t_b only for a result of a newer analysis
  package, which lists it.
- **The evolution**: R, H, t_b (dark gold), D and the gap (nm) and the coverage over the time in the scan, for the
  solution shown in each window; the dashed line marks the window shown. With a newer analysis package faint bands
  behind both plots mark the stages of the growth — **Separate islands**, **Coalescence** (coverage above 0.5),
  **Continuous film** (half of the metal in the continuous layer) and **After the deposition** — named where the band
  is wide enough (see **Details ▸ Growth Process**).
- **Patterns | Real Space | Components**, at the right of the window line: **Real Space** draws the solution shown in place of
  the two images, in nm. Left a cut through a row of grains: the substrate (a polymer film is drawn, its
  thickness not to scale), the continuous layer under the grains when there is one, and the grains with their
  radius and height. Right the view from above: grains with the radius and size distribution of the solution. The
  whole scan is drawn as one growing film: the view has the same size in nm in every window, and the grains of a
  window are those of the window before. While the film is open, new nuclei appear in free places where the
  distance D shrinks (nucleation) and neighbouring islands merge into one round island where it grows
  (coalescence). From about half coverage on, nothing disappears or moves any more: the islands grow together into
  worm-like clusters and then a connected film, and what stays open are channels down to the substrate. The metal
  drawn covers the share of the view the solution's coverage gives and never less than before (with a continuous
  layer at least 85 %: the model's layer has no lateral structure of its own, so the channels of the picture are an
  illustration). Samples measured after deposition are drawn the same way, ordered by their nominal thickness. The
  positions are an illustration; the sizes are those of the solution. The cut on the left runs through the middle
  of the view from above. It follows the window and the solution you choose. Grains that touch are drawn as one piece; a
  piece that runs across the whole view is darker, and the line above the picture says whether the grains are
  separate, coalescing or a connected network (percolated). Once the film is joined - a continuous layer under the
  grains, or grains joined into a network across the view - it is drawn as the film itself, not as grains: the view
  from above shows the height of the metal (lighter where it is low, darker where the relief rises) with the holes of
  a network in the colour of the substrate, the cut is the film's profile along the middle of that view, and the line
  says *Continuous film … nm, grains on top: …* or that the network crosses the view. The tiles then speak of the film
  too: **Film height H**, **Relief radius R** and **Relief spacing D** (the lateral correlation length of the film's
  surface, not the distance of separate particles); **Gap** is not shown.
- **Components** splits the model image of the solution shown into its physical terms — **Grains and islands**,
  **Large clusters**, **Film fluctuation and network**, **Voids under the film**, **Substrate seen through the metal**,
  **Fluorescence**, **Beam-path background** …; on porous TiO₂ **Clusters in the pores**, **Clusters on the cap layer**
  and **Metal network in the pores**. Each term with at least 0.1 % of the model counts gets a small image (at most
  six, the largest first), on the colour scale of the measured image, so a term that contributes little looks dark;
  its title gives its share of the counts. The terms add up to the model image. At the right, **Share of the model
  counts**: every term over the scan for the solution shown in each window, the window shown marked. A result of an
  older analysis package says that it does not split its model.

When the data cannot tell two solutions apart, both are there, as candidates with similar fit deviations: compare
their images. A window the package could not use (no frames in it, for example) says why in place of its images.

**Metal in a porous TiO₂ layer.** The model has no lattice of grains there, so there is no height, distance, gap or
coverage. The values are the **Island radius (relative)** — follow how it changes, not its number — the size
distribution width, the **Island share** (of the metal in islands that can be resolved; it can exceed 1 early on),
the **Cap share** (of the metal in the cap on the TiO₂ particles), the **Vertical** and **Lateral correlation**
lengths of the metal network in the pores and the **Film share**. The evolution shows the three lengths at the left
and the three shares at the right. **Real Space** draws the layer: left a cross-section to scale, with the TiO₂
particles, the metal in the pores as a band at the depth where the model puts it, and the cap on the share of the
particle tops it covers; right the top of one particle enlarged, with the cap at its thickness and the clusters at
their size. The line below says how much metal has been deposited and how much of it is in islands, in the cap and
in the pores.

**Samples measured after their deposition.** A sample measured after its deposition — one of a thickness series
of Au films, for example — has no deposition rate and no frames from before the deposition. Choose **After
Deposition** at the left of the input bar (**In Situ** is a scan recorded during the deposition). The bar then holds
**Add Samples…** and a line that sums up the samples, and the **Sample List** stands at the left of the main screen
and of Details (the switch **Sample List** at the right of the bar hides or shows it):

1. **Add Samples… ▸ Folder…**: a folder of samples — each folder in it with CBF frames is one sample, however deep
   the frames lie (measurement folders, `run1` …; frames whose file name says GIWAXS are left out) — or one sample's
   folder; **CBF File…** takes one frame as a sample. A folder that holds nothing but one folder (an archive unpacked
   into a folder of the same name) is looked into. Zip archives are not opened: unpack them first. Samples in the
   list already are not added again.
2. GIMaP reads each folder's name. The **nominal thickness** is a number followed by `nm` or by the metal (`10Au`,
   `6.5nm`, `6p5_Au`); `H` or `HiPIMS` right before it marks a HiPIMS deposition. What stands before it is the
   substrate (`850` of `850_H10Au`): a folder beside the samples named just that (`850`) is their **bare substrate**
   — another piece of the same substrate without metal, measured the same way, which gives the substrate part of the
   images; without one the film is taken to lie directly on Si. The samples of one substrate and one deposition
   method are a **series** (`850 · DC`, `850 · HiPIMS`; DC is named only where HiPIMS samples of the same substrate
   are listed too), analysed into a folder of its own. A sample whose name gives no thickness is listed unticked
   with *Thickness?*: double-click its **nm** to enter it. Only Au films can be analysed: a folder named with another
   metal is listed unticked (*Not Au*).
3. **The list** shows each series — its bare substrate and how many of its samples are analysed — with its samples:
   a box (ticked: Analyze runs it; the box of a series ticks all of its samples), the name and the nominal thickness
   (double-click either to change it) and the state: *Waiting*, *At work…*, *Fit 2.91* once analysed (its fit
   deviation), *Failed* (the reason in the tooltip). A right-click offers **Analyze This Sample**, **Analyze This
   Series**, **Show Result**, **Bare Substrate…** or **No Bare Substrate (on Si)** for the whole series, **Open
   Folder** and **Remove**; **Edit** selects all or none and removes samples.
4. **Analyze** runs the ticked samples one after another, in the order of the list. The job status says *Sample 3 of
   12: 850_6Au* and its step, and the bar covers the whole list. While a sample runs the main screen shows three
   steps: the frames of the sample read (the detector summed, the usable frames summed, which frames are left out and
   why), the solution of each frame as it is found (measured and model image of the frame solved last; below, the
   series with every frame solved so far at the sample's nominal thickness), and the sample added to its series. A
   sample that fails is marked and the next one starts; **Cancel** stops the sample at work and the ones after it
   (those analysed stay in their series). A sample analysed before with the same frames and values is not run
   again; samples analysed before with other frames or values are replaced after one question that names them all.
   When the samples are done, the samples of each series analysed are fitted together as one growing film (*Fitting
   the samples of … together as one growing film*): ordered by nominal thickness, a thicker sample is built on the
   thinner one - radius, spacing, continuous layer, film height, metal amount and coverage (below closure) may not
   decrease from one sample to the next. About 5-10 minutes per series on a desktop; the samples keep their own
   solutions if it fails.
   A series of one sample, or an older analysis package, skips this step.
5. Afterwards the main screen shows the series of the last sample analysed: one sample in place of each time window
   — its value is the median over its frames — and the evolution along the nominal thickness. Choose a sample in the
   list to see its series at that sample; moving along the series on the main screen chooses its row.
6. **Export…** (beside **Open Result…**, and in Details) writes `samples_report` into a folder you choose:
   `report.html`, one file to send, in the interface language, with every series of the list in two parts. **Separate
   particles** — the grains stand apart or touch without joining across the picture: radius R, height H, spacing D,
   the gap D − 2R and the coverage. **Connected film** — a continuous layer under the grains, or grains joined into a
   network: there are no separate particles any more, so the film's thickness and density, its height and the relief
   of its surface (height, radius, spacing) are given. Each part has its table (with the nominal thickness, the
   stage, the metal amount h_m and the fit deviation) and its panel of the figure against the nominal thickness; the
   report says from which thickness on the film is connected, and shows each sample's measured and model images with
   two cuts. `samples.csv` lists every sample with its part. Parts and stages follow the **Real Space** picture of the
   main screen and the captions of its tiles: *separate islands*, *grains in contact* (touching, no cluster across
   the picture), *grains joined into a network*; with a continuous layer, *grains on a thin layer* while it is
   thinner than the grains stand above it, else a *continuous film*, or a *film with a density below 1* below 0.95 of
   the bulk density.

Under **Details ▸ Static Frames (after deposition)**: the **Results folder** (each series goes into a folder of its
own in it, named after the series; a series already there is added to), the **Mode** (Quick, Refine or Full), the
geometry of these frames (empty fields keep the package's own values, which **Package Values** fills in; **Masks of
the Package** lists the pixel masks it brings) and the log. One analysis runs at a time. The list is kept with the
project and for the next session.

The package supports Au films at an incidence angle near 0.40°; what it says about other cases appears in the
section's log.

![After Deposition: the sample list beside the main screen](manual_images/en/22b_sputter_samples.png)
*After Deposition with a folder of samples added — here invented folders and the test suite's made-up results
(synthetic images, invented values). Left the Sample List: three series with their bare substrate, each sample with
its nominal thickness read from the folder name and its fit deviation; a sample whose name gives no thickness is
left out (*Thickness?*) and its series listed last; one sample is not analysed yet. Right the series of the sample
chosen, at that sample.*

![The main screen of Sputter with a result](manual_images/en/22_sputter.png)
*The main screen with a result — here a made-up one: synthetic images and invented values, and three PILATUS CBF
frames from the repository's `TestSAXSdata/` folder standing in for the scan folder. Top: the input bar, the
geometry in one line and the job status of Analyze. Below: the window chosen with the switch Patterns | Real
Space, the measured image beside the model image of the solution shown, the solutions with their fit deviations, the values, and the evolution over the scan.*

**Before you conclude.** The values are those of a physical model whose image reproduces the measured one, on an
evolution that is smooth over the scan. They were not compared with an independent measurement of your sample
(SEM, AFM, XRR), and the nominal deposit and the metal amount scale with the rate you give. The package's report
(**Details ▸ Report**) says for every quantity whether it is read from the counts, a model value or an estimate.

**Open Result…** shows a result analysed before: choose its folder, or the scan's results folder to get its newest
result. A folder without a result of Analyze (one of the earlier method, Predict, which is no longer part of GIMaP)
is not opened. Chapter 12 says where the results are.

### 8.2 Details

**Main Screen | Details** at the right of the bar switches to Details. At the left, from the top:

- **Sample and geometry**: **X-rays** (**Energy (keV)** or **Wavelength (nm)**), **Distance (mm)**, **Incidence αi
  (°)** and **Beam centre (px)** (col, row). Check every value. Beside each, the page says where it comes from —
  this folder's last Analyze, a preset, for the energy a `Wavelength` line of the header or the scan's NeXus file,
  or the last run of any folder (*please check*) — or asks for it: a PILATUS header holds neither the distance nor
  the incidence angle. The `E<eV>` in a flat-field file name is the detector's calibration energy, not necessarily
  the beam's: it is shown beside the field and never filled in. A value to check and an energy of the files that
  differs from the one used are shown as a warning (⚠); the tooltip of the line says why. An empty **Beam centre
  (px)** is estimated from the frames; a centre you give is in Analyze's pixel coordinates (section 4.4). **Save
  Geometry as Preset…** fills in later scans of this detector and flat field within one beamtime (a week either
  side of the scan, to start with). Section 3.9 says what else is remembered.
- **Analyze options** (section 8.1).
- **Static Frames (after deposition)**, collapsed (section 8.1, *Samples measured after their deposition*).
- **More options**, collapsed:

| Field | What it is for |
|---|---|
| **Mask** | A detector mask (`.npy`, optional), remembered for this folder only. **Masks of the Package** offers the pixel masks the analysis package brings. |
| **Sample roll (°)** | 0 when neither this folder nor a preset gives one (the page says so). |
| **Save results in** | Where the result folders go (default: the folder `sputter` in the user data folder). |

**Storage** (folded, under More options): the frames Analyze imports are kept so that the same scan is not read
again — often a few gigabytes per scan. They are in GIMaP's local folder (on Windows
`%LOCALAPPDATA%\GIMaP\sputter_client`, on macOS `~/Library/Caches/GIMaP/sputter_client`), not in the user data folder,
which on Windows roams with the profile. Opened, the section lists each import (its size, when an analysis last used it,
where it is), their total and the size of the results. **Remove Selected** and **Remove Unused For** *n* **days** remove
imports after asking; the one an analysis is using stays. **Limit of the imports** (default 5 GB, 0 for no limit): when
an analysis starts and when it ends, the imports not used for the longest time are removed until the rest fit — never
the one in use, never a result. Imports an older GIMaP kept in the user data folder are still used and are listed as
*user data folder (older GIMaP)*; nothing moves them.

At the right, five tabs about the result the main screen shows (before there is one, each says *Analyze a scan or
open a result*):

- **Windows**: one row per time window (for a series of samples measured after their deposition, per sample) with
  the solution shown on the main screen — the time, the nominal thickness, its values, its fit deviation and which
  solution it is. Choosing another solution on the main screen changes its row. `Ctrl+C` copies the selected rows.
- **Growth Process** (an in-situ scan analysed with a newer package): which solutions it is made from (the joint fit
  of the whole scan, or the smooth path through the windows); **Stages of the growth** with their time range and metal
  amount; **Growth laws** against the metal amount h — the amount at coverage 0.5 and at which the film forms, D = D₁ h^p
  and R = R₁ h^q before the film, D and R once the film has formed, the coarsening after the deposition stopped, the
  median fit deviation and the steps back between windows — each beside the median and range of the same metal on
  the same substrate over the scans the package was built from, and whether this scan lies within, below or above
  it; and **Physical terms of the model**: per window the share of each term for the solution shown (it follows your
  choices). Metal amounts are equivalent thicknesses: they do not depend on the X-ray wavelength.
- **Report**: the network the analysis used and the networks it tried with their fit deviation, the mode, the pixel
  mask and the run time; then **Time per step**, a small table of every step with its time (m:ss) — the import row
  adds the frames read of how many and the disk's seconds per frame with their share of the time — the total of all
  steps and the number of processes; a note when reading the frames from disk was the slow part (as in the live
  view); and the package's report (`REPORT.md`). A result of an older package has the rows it can (no reading facts,
  no processes).
- **Logs**: the log of each step of the analysis (`analyze_*.log`), one at a time.
- **Steps**: the steps of the analysis as the live view showed them, each with what it found; click a step marked ›
  to see its picture at the right. A result of an older analysis package says that it has no record of its steps.

![Details of the Sputter workspace with a scan folder chosen](manual_images/en/22a_sputter_details.png)
*Details with a folder chosen. Here three PILATUS CBF frames from the repository's `TestSAXSdata/` folder stand in
for a scan (they are not a sputter series, so no result is shown): the line under the input bar sums up the frames,
the detector, the exposure and the time. The energy, distance and incidence angle are not in the files, so the page
asks for them (the flat-field energy is shown beside the energy, not filled in); an empty beam centre is estimated
from the frames.*

### 8.3 Output files

A result of Analyze is in a subfolder of the scan's folder, `sputter/<scan>-<code>/iampred-<code>`: the package's
report (`REPORT.md`, `report.png`, `report.csv`) and figures, `windows.csv`, `analysis.json`, the logs of its steps
(`analyze_*.log`), the data of the main screen (`main/main.json` and one `w_<t>.npz` and `.png` per time window),
your choices of a solution (`main/choices.json`) and `gimap_request.json`. Chapter 12 lists them all.

## 9. Tool windows

This chapter describes three windows of the **Tools** menu: Format Converter, XRR Series Extractor and SEM
Particle Analysis. The other entries of that menu are described with their work: Geometry Calibration
(section 4.3), 1D Predict (section 6.4) and Process with AI (chapter 10). The windows work beside the
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
3. **Set the geometry.** Enter the distance, energy, pixel sizes and direct-beam centre. In **Detector rotations**, check pyFAI rot1, rot2 and rot3 in degrees and **pyFAI orientation** (1–4). Missing rotations default to zero and orientation defaults to 3. **Reflection direction** chooses the upward or downward ray in the laboratory. With detector roll the spot can move across columns.
   - The window starts from the last applied Geometry Calibration, marked *from last calibration*. When
     that calibration was of frames of another size, only its energy is taken.
   - **Load first frame** never overwrites a value you typed. A tag beside each field says whether a value
     comes from the file, the last calibration, the image centre or the built-in defaults; the summary under
     **Load first frame** warns only of the values that are not in the file and need checking
     (*⚠ Not in the file — check: …*).
   - If a file gives an unsupported orientation or tilt, the inspection names the field and its value. Check the source beside the value used for extraction. Full turns of rot3 are equivalent.
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
A θ motor dataset accepts degrees, rad, mrad and urad (including µrad/μrad). Unknown units and non-finite angles stop extraction. Integer negative gap codes and the reader mask are excluded; floating negative data are retained. Analyze's interactive masks and bad-pixel search are not applied. Header beam centres use the displayed frame's transformed coordinates; stitched module sets without a unique centre require a calibrated or picked centre.


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

**NeXus header centre.** XRR and Analyze now use the same displayed centre, including the P03 transpose and vertical flip. A stitched module set with no unique header centre uses calibration or an explicitly picked centre. Check the cross on the preview before extraction.

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
hold one frame each as one series.

- **Frames.** An argument may be a file, a pattern such as `*.tif` or a folder. A pattern the shell did not
  expand (PowerShell does not) is expanded by GIMaP, in natural order (`f2` before `f10`); a folder stands for
  every detector image in it (`.cbf`, `.tif`, `.tiff`, `.edf`, NeXus/HDF5; subfolders are not searched; the
  files given as `--calibration`, `--mask` or `--flat-field` are left out). `summary.md` says how many files each pattern or
  folder gave; one that gives none is an error (exit code 1).
- **Values that cannot be used.** `--incidence-deg` must be above 0° and below 90°, and a `--calibration`,
  `--mask` or `--flat-field` file must exist: otherwise the command ends with the reason and exit code 1, before anything is read
  (it never goes on with a calibration it found itself in place of the one you named).
- **A calibration that is not the frame's.** A calibration nobody named, which the search took from another folder
  than the frame's (or one below it), is an open item of the report with its full path (*calibration found by the
  search*; exit code 2): give it as `--calibration` when it is the right one. A `.poni` that states another
  detector shape than the frame's — in `max_shape`, or by the detector it names (a Pilatus 300K file on a Pilatus
  1M frame) — is not used: *the calibration does not match this frame*. In a GISAXS run, a symmetry axis more than
  100 px from the calibrated beam centre is not applied (the report says *not applied* and asks for the
  calibration of this measurement), and a symmetry axis that could not be found is an open item too.
- **Folders.** Each frame has its own folder under `--out`, named by the frame, and `summary.md` links every
  row to its `report.md`. Frames of the same name in different folders get the nearest folder name that tells
  them apart (`<name>_<folder>`), else a number. An `--out` folder that already holds the summary of an earlier
  run is refused (exit code 1); `--overwrite` replaces that run: the frame folders its `summary.json` lists
  are removed first, nothing else.
- **Mask.** `--mask FILE` applies one mask to every frame, read as **Load Masks** in Analyze reads it: a GIMaP
  masks `.json` recipe (ordered shapes, rules and image sources) or a mask image of the frame's size whose non-zero pixels are left out. A frame
  the mask does not fit fails with the reason. `report.md` (section **Mask**) and `report.json` (`mask`) always
  say which mask was applied, or that none was; detector gaps and the bad pixels found automatically are
  named apart from it.
- **Flat field.** `--flat-field PATH` divides every frame by a flat field (detector response) as the **Flat
  field** of Analyze's Corrections does: a detector image of the frame's size or a `.npy` array, normalised to
  mean 1, pixels at or below 0.05 of it left out. A frame it does not fit fails with the reason. `report.md`
  names it among the **Decisions** (*flat field: divided by flat.npy (120 pixels left out)*) and in **Record**
  (the file with its SHA-256); `report.json` has it in `inputs` and in `analysis_record.corrections.flat_field`.
- **Record.** `report.json` names the release, the UTC time, the options as given (the notes too), and every
  input file (the frame's files, the calibration, a mask, a flat field) by path, size and SHA-256; for a model fit also the
  network, the SHA-256 of its weights and the seed. `summary.json` is one object with the release, the time, the
  options and the frame reports (`frames`). Every CSV starts with `#` lines that name the release, the inputs
  with their SHA-256 and the geometry, and names its columns in plain ASCII with one spelling of the units
  (`qy (A^-1), I, sigma`; `chi (deg)`), so that a program reading in the system's code page reads them; the curve, q-map, cake and fit-input files Analyze exports in the window
  carry the same lines. `report.md` has the section **Record** of the window's saved report (section 5.5). In
  `report.json`, `geometry` holds the values q was computed with (15 significant digits; the text rounds them) —
  for GISAXS its beam-centre x is the refined symmetry axis — while `calibration` and `tables.geometry_used` hold the
  calibration as it was read, before that refinement (`gisaxs.symmetry` has both columns); `critical_angle_deg` is
  `null` when none was given. `report.md` states the pixel size and the energy with the geometry, defines the
  *asymmetry* it prints for the symmetry axis and names the disorder of the distance σnn/D, as Fitting's Model
  table does. And every crystallite size names its unit (`size_A`, `size_nm`). A run repeated on the same input gives the same files except for the
  time and the durations ([Physics and Methods, section 14.10](Physics_Manual.md#1410-the-record-of-a-run-without-the-window)).
- **Curves as NXcanSAS.** `--curve-format nxcansas` writes the curves against q of each frame as one NXcanSAS file,
  `curves_nxcansas.h5`, instead of their CSVs (as Analyze's **Also as NXcanSAS (.h5)** writes it, section 5.8; the
  I(χ) of the last ring stays `azimuthal_last_ring.csv`, the fit stays `fit_curve.csv`); `--curve-format both`
  writes both; `csv`, the default, writes the CSVs only. `call … --out` takes the option too.
- **Language and time.** The headings of `report.md` and `summary.md` are English; `--language zh` writes the
  section headings (the `##` lines) in Chinese and nothing else: the titles, the tables and their column names,
  the decisions, the questions and every other sentence stay English, as do `report.json` and the CSVs (one
  language per report; in the window the headings follow the program's language). The GISAXS model
  fit takes from some seconds to a few minutes per frame, depending on the number of points and the computer;
  `--no-fit` leaves it out.

Its options, exit codes, where its defaults may
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

Trainset Build has been removed from GIMaP. There is no training-set page, menu entry or startup task. Existing training files are not deleted. Train models outside GIMaP, then import a compatible model into 2D Prediction (section 11.3).

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
external training tools register: its `module.yaml` names the model file of this computer. What 2D
Prediction does with a frame, and what it does not correct, is in
[Physics and Methods, section 13.4](Physics_Manual.md#134-2d-prediction-labs).

For the SEM comparison, the module must declare the radius axis: `outputs: axes: row` with `key: r`,
`unit: nm`, `min` and `max`, as in `Au_Silicon_15nm`. A distribution whose module declares no axes is
drawn on a placeholder axis (0.05–15 nm on both axes), which is not a physical scale, and is not attached
to the SEM comparison.

Predict parameters uses its network in `src/gimap/features/fitting/domain/posterior/data/` (section 6.4), not a
prediction module.

The 1D Predict `request.json` and `fitting_run_summary.json` identify the network's record by SHA-256 (the record
names the SHA-256 of the weights, checked when they are loaded). The request records the normalized options and the
random-seed policy: Predict parameters uses a fixed seed.

## 12. Output files

GIMaP never changes your data files. Exports from Analyze go by default into a folder
**`gimap_analysis/` next to the data**; most other exports ask for a place and propose one. Some propose a place
beside the data too (the XRR CSV, the Format Converter's `converted/`, a calibration, a project): on a read-only
data share, choose another folder. The user data folder is `%APPDATA%\GIMaP` on Windows and `~/.config/gimap`
elsewhere, or the folder in `GIMAP_HOME` (section 3.11; **Help ▸ Open User Data Folder**). The table lists every
file GIMaP writes; the paragraphs below it, what the user data folder and the Windows package hold.

| Action | Files written | Default location | Record that comes with it |
|---|---|---|---|
| Analyze ▸ **Export** (`Ctrl+E`), **Export Curves…** | `<stem>_<curve>.csv` per curve, `<stem>_fit_input.dat`, `<stem>_analysis.json` (`_frameNNNN` added for a frame of a multi-frame file, `_sumN` when frames are summed); with **Also as NXcanSAS (.h5)** also `<stem>_nxcansas.h5` | `gimap_analysis/` next to the data, without a dialog; a folder dialog if it cannot be written | `<stem>_analysis.json` (settings, geometry with the tilt of a `.poni`, corrections, pixels left out, error model, binning, UTC time of writing) and `#` header lines |
| Analyze plots ▸ **Save ▸ Curves as Data…** | the CSV files of that plot and `_analysis.json` | folder dialog in `gimap_analysis/` | as Export |
| Analyze plots and images ▸ figures (**Plot as Figure…**, **View as Figure…**, **Analyze ▸ Figures ▸ Save Image…** …) | PNG, TIFF, SVG or PDF, light colours | dialog in `gimap_analysis/` | none |
| Analyze ▸ **Save q-Map Data…**, **View Data as CSV…** (also **Analyze ▸ Figures**) | `<stem>_qmap.csv`, `<stem>_cake.csv` | dialog in `gimap_analysis/` | `#` header lines (source, frame, axes) |
| Analyze ▸ Series ▸ **Export** (**Map as CSV Table…**, **Map as Figure…**, **Stages as Table…**, **Selected Frame's Curve as Figure…** / **as Table…**, **Trace as Figure…** / **Trace as Table…**, **Peak Table of Every Frame…**) | `<series>_<curve>[_<halves>]_map.csv`, `…_map.png`, `…_stages.csv`, `…_trace_<value>.csv` and `….png`, `…_peaks.csv`; the frame picked as `<frame>_<curve>[_<halves>].csv` and `.png` (figures also as TIFF, SVG or PDF) | dialog in `gimap_analysis/` | `#` header lines (curve, halves, frames, cut) and a `.json` of the settings next to each table; none for the figures |
| Analyze ▸ Series ▸ **Export ▸ Save Results Database…**; Watch ▾ ▸ **Record Every Frame in the Results Database** | `gimap_series.sqlite` (SQLite: every frame's curve as reduced, with its frame, file and time; the skipped files) | dialog in `gimap_analysis/`; while watching, `gimap_analysis/` next to the data | the settings of the map, in the database |
| Analyze ▸ Series ▸ **Export ▸ Timeline as Table…** | `<series>_<curve>[_<halves>]_timeline.csv` | dialog in `gimap_analysis/` | `#` header lines and a `.json` (method, reference frame, jumps) |
| Fitting ▸ In-situ series ▸ **Save ▸ Fits as NXcanSAS…** / **Results Database…** | `series_fits.h5` (points fitted and model of every fitted frame); `gimap_series.sqlite` (every frame's fit) | the folder of the curves; a database found there or up to two folders above | each entry holds the frame's record; the database the fit settings |
| Analyze ▸ **Send to Fitting** | `<stem>_fit_input.dat` (q in Å⁻¹) | `gimap_analysis/` next to the data; a folder you choose (kept for the session) when that cannot be written | `#` header lines (source, frame, sum, curve, observation with the halves) |
| Analyze ▸ **Batch Export…** | per-curve tables `<batch>_<curve>_frames.csv` (or `.txt`, `.dat`); if you chose fitting, a fit table `<batch>_peak_fits.csv` (or `<batch>_model_fits.csv`) with its trend plot `<batch>_peak_fits.png` (`_model_fits.png`) and, if chosen, the fitted curves `fits/<frame>_<peak or model>_fit.csv`; subfolders `curves/`, `nxcansas/`, `fit_input/`, `maps/`, `images/`, `frames/` after your choices; frame names keep the 0-based index (`…_frame0003_sum3_…`) | the last batch folder, else `gimap_analysis/` next to the data; by default in a subfolder named after the frames' folder (one file: its name; several folders: `batch`); **Keep Both** writes into a new subfolder | `README.txt` and `<batch>_batch.json` (every file written, the cut of every frame, failed frames and why); `curves/` also gets an `_analysis.json` per frame |
| Analyze ▸ **Send Series to Fitting…** | `fit_input/<frame>_fit_input.dat`, `README.txt`, `<batch>_batch.json` | a folder of its own, `<name>_fitting/`, next to any Batch Export of the same data | `README.txt`, batch JSON |
| Analyze ▸ **Export New Frames While Watching** | as Export, for each new frame | `gimap_analysis/` next to the frame; when it cannot be written, a folder you choose once (kept for the session); with none, the export is switched off and the status line says why | as Export |
| Analyze ▸ **Save Settings…**; Mask step ▸ **Save…**; Regions ▸ **Add ▸ Save Cuts…** | `analyze_settings.json`; `masks.json` or a mask image (section 5.2); `cuts.json` | dialog in `gimap_analysis/` | JSON recipes are self-describing. Mask images have a companion `<filename>.json`. |
| Run Automatic Analysis ▸ **Save Report…** | `<stem>_report.html` (with pictures) or `.md` | the folder you chose last in this session for frames of the same folder, else `gimap_analysis/` if it exists, else next to the frame | the report itself (no JSON) |
| Process with AI ▸ **Save Report…** | `<stem>_ai_report.html` + `<stem>_ai_report.json` | as above | the JSON |
| Process with AI (automatic) | a record of every run; missing analyses | user data folder: `assistant_runs/`, `assistant_feature_requests.jsonl` | — |
| Process with AI, when the AI exports | the curves as Export, and `<stem>_assistant.json` | `gimap_analysis/` next to the data | `<stem>_analysis.json`, `<stem>_assistant.json` |
| Fitting ▸ Single analysis ▸ **Save ▸ Data and Fit…** | `<curve>_fit.csv` (or NXcanSAS `<curve>_fit.h5`) + `<curve>_fit.json` (written as a pair) | the folder saved to last, else the folder a curve was last opened from, else the curve's folder | the JSON (model, method, range, left-out points; every solution of Predict parameters) |
| Fitting ▸ Single analysis ▸ **Save ▸ Solutions…** (**Export Solutions…**) | `<curve>_solutions.csv` + `<curve>_solutions.json` (written as a pair) + `<curve>_solutions_curves.csv` | as Data and Fit | the JSON (the Data and Fit record with every solution, its values and intervals) |
| Fitting ▸ **Save ▸ Plot…**, **Save ▸ Model…** | `<curve>_fit.png` / `.svg`; `<curve>_model.json` | as above | none; the model file |
| Fitting ▸ Batch / In-situ series ▸ **Save ▸ Table of Every Frame…** | `series_fit.csv` + `series_fit.json` | the series folder | the JSON (`gimap_series_fit_v2`: start model, halves and where they came from, every frame's state and reason; for detector frames Analyze's settings) |
| Fitting ▸ Batch / In-situ series ▸ **Trend Plot…**, **Selected Frame's Plot…** | `series_trend_<value>.png` (the table column of the value, for example `series_trend_chi2_reduced.png`) and `series_frame_<NNNNN>.png` (the frame counted from 1, five digits), or `.svg`; `series_trend.png`, `series_frame.png` while nothing is fitted | the series folder | none (the title in the picture names the value or frame) |
| **Tools ▸ 1D Predict — Fit Many Curves…** | a run folder: `request.json`, `batch_summary.json` …, a subfolder per curve with `solutions.json`, `fitting_curves.csv` … | `AI_Fitting_Output/v5_<date>_<time>_<µs>/` in the GIMaP user data folder; never removed automatically | `request.json` |
| Compare ▸ **Save ▸ Table of Every Series… / Table of Every Frame…** | `compare_series.csv` / `compare_frames.csv` + `.json` | the folder of the first series added from files, else the folder a series was last added from (while it is there), else your home folder | `#` header lines and the JSON |
| Compare ▸ **Change Plot… / Paths Plot… / End States Plot…** | `compare_*.png` or `.svg` | as above | none |
| Sputter ▸ **Analyze** (section 8.1) | a folder per request: `analysis.json` (status, model family, steps, the files), the package's `REPORT.md`, `report.png`, `report.csv`, `windows.csv`, `summary.json`, `evolution.png`, `patterns.png` and step logs, and `main/` with `main.json` and one `w_<t>.npz` + `w_<t>.png` per time window (the measured image and the model image of every solution) | the scan's folder in the user data folder (or **More options ▸ Save results in**): `sputter/<scan>-<code>/iampred-<code>/`; the same request to the same package shows the result it has. The frames as the package read them are kept in GIMaP's local folder (`%LOCALAPPDATA%\GIMaP\sputter_client\` on Windows, `~/Library/Caches/GIMaP/sputter_client/` on macOS), at most 5 GB by default (**Details ▸ Storage**); the steps' details are in the result's `trace/` | `gimap_request.json` (the request, the label of the imported scan, the package and its version, the Python, the command) |
| Sputter ▸ another **Solution shown** on the main screen | `choices.json`: the solution chosen per time window, where it is not the default | the result's `main/`; when it cannot be written, the user data folder: `sputter_client/choices/` | the file itself (result folder, GIMaP version, time) |
| XRR Series Extractor ▸ **Export CSV…** | `<source>_xrr.csv` + `.json` | next to the source file; for a folder chosen with **Folder…**, inside it, named after it | the JSON (every value that determines qz and its origin; `center_convention`; `source_layout` with every file read) |
| Format Converter | converted frames: `<source>.<ext>` for a single-frame source; for a multi-frame source by default `<source>_<frame counted from 1, 6 digits>.<ext>`; `_001` … added on a name collision; or one `converted_images.h5` (`converted_images_001.h5` … when it exists) | `converted/` next to the first input | `conversion_report.json`, `conversion_metadata.json` (or one JSON per frame) |
| SEM Particle Analysis ▸ **Export SEM results…** | the files of section 9.3 | a new `SEM_analysis_<date>_<time>/` in the folder you choose; never overwrites | `sem_analysis.json`; `#` header line in the CSVs |
| 2D Prediction ▸ **Export input...** / **Export result...** | `<input>_input.png` + `.json`; `<input>_prediction.png` + `.json` (+ `.txt` when the result has a curve or map) | the export folder used last, else the input's folder | the JSON |
| 2D Prediction ▸ **Export result...** with several files | your choice of `prediction_results_<time>.jsonl`; `prediction_curves_<time>.txt` (or `prediction_parameters_<time>.txt`); images in `prediction_images_<time>/` | a folder you choose (starts at the export folder used last) | `#` header lines in the `.txt` |
| Geometry Calibration ▸ **Export Calibration…** | `<image>.gimap-calibration.json` | next to the calibrant image | the file (beam centre counted from pixel centres at 0, 0.5 px lower than the windows show; `metadata.center_convention`) |
| Geometry Calibration ▸ **Apply** | the instrument profile | user data folder: `instrument_profiles.json` | — |
| **Tools ▸ Settings… ▸ Assistant** ▸ **API key** ▸ **Save** | `claude_api_key.txt` (Claude API), `assistant_provider_keys.json` (other providers), in plain text; **Remove** deletes the key | user data folder | — |
| **File ▸ Save Project** | `<name>.gimap`; `<name>.<generation>.compare.npz` when Compare holds embedded arrays | next to the first frame | the project (data referenced by path); move the current `compare.arrays_file` with it (section 3.8). Sputter results stay in their result folder |
| **File ▸ Labs Parameters ▸ Save As…** | `gimap_parameters.json` | dialog in the GIMaP user data folder | — |
| Errors (automatic) | `logs/errors.log` (moved to `errors.old.log` above 2 MB) | user data folder | — |
| **Help ▸ Example Data** | the example's frame with its `.poni` (and mask), and `SOURCE_AND_LICENCE.md` | a folder you choose, by default `Documents/GIMaP examples/<example>/`; files already there are kept | `SOURCE_AND_LICENCE.md` (source and licence of the data) |
| `tools/gimap_agent.py` | a subfolder per frame (a set of module files: per scan; `--series`: `<name>_series/`; frames of one name in different folders: `<name>_<folder>/`) with `report.md`, `report.json`, curves and fit tables (CSV whose `#` lines give the release, each input by SHA-256 and the geometry), q map; `summary.md` (links every frame's report) + `summary.json` | `--out`; without it `assistant_runs/cli/<time>/` in the user data folder; never next to the data. An `--out` folder with the summary of an earlier run is refused unless `--overwrite` | `report.json` (release, UTC time, the options as given, every input file by path, size and SHA-256, the mask applied or none, Analyze's record of the frame; for a fit the network, the SHA-256 of its weights and the seed) |

**Which release wrote a file.** The records of Analyze (`_analysis.json`, the settings file, the series and
stages tables' JSON), of Batch Export and Send Series to Fitting (`<batch>_batch.json`; `README.txt` says *Written by
GIMaP v…*), of Fitting (`<curve>_fit.json`, `series_fit.json`), Compare, Geometry Calibration, the XRR Series
Extractor, SEM Particle Analysis, 2D Prediction and the Format Converter name the GIMaP release that wrote them:
`software_version`, for example `GIMaP v0.2.0`. In Analyze's records `software` stays `GIMaP Analyze 1`, the name
of the record format. The records of Sputter, the AI assistant, `tools/gimap_agent.py` (`report.json`,
`summary.json` and the `#` lines of its CSV tables) and 1D Predict's run folders name it too; records written by
earlier versions may not ([Physics and Methods, section 14.5](Physics_Manual.md#145-version-of-the-software)).

**What `<stem>_analysis.json` says besides the settings.** `created` is the UTC time the record was written
(ISO 8601). `error_model` says how the σ of the points were computed: `poisson` (photon counts),
`pixel_scatter` (values that are not counts as measured: a dark- or background-subtracted frame, pixels
filled from the mirror side) or `scaled_poisson` (GIWAXS intensity corrections on counts). `binning` gives,
for GIWAXS, the number, q range and width of the radial bins, how many points the I(q) kept after sparse
bins were joined and the rule for joining them; for GISAXS, one point per detector column or row. In
`geometry`, `rot1_rad`, `rot2_rad` and `rot3_rad` are the detector's rotations (pyFAI's, all 0 for a detector
perpendicular to the beam; section 4.2). The fields `dropped_tilt_deg`, `flat_detector_q_error_max` and the other
`flat_detector_…` and `dropped_…` fields described what a flat detector left out up to 0.1; they are `null` now
that the tilt is part of the geometry. The instrument profile's `source` says who chose the geometry:
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
text); and Sputter's result folders in `sputter/`, which are never removed. Delete old
result folders yourself, or choose **More options ▸ Save results in** on a data disk for long scans. The frames Sputter ▸
Analyze imports are in the local folder `%LOCALAPPDATA%\GIMaP\sputter_client` (macOS
`~/Library/Caches/GIMaP/sputter_client`, Linux `~/.cache/gimap/sputter_client`), kept under a limit (**Details ▸
Storage**, default 5 GB); those an older GIMaP kept in `sputter_client/` of the user data folder are still used, and
Storage lists and removes them. Outside
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
| **The rows of a Series map carry no σ.** The timeline's χ² takes the noise of each x column from the series itself, and a Series map recorded in the results database holds x and I only (section 5.6). | Read the timeline as a distance in units of the frame-to-frame noise, not as a goodness of fit; for σ of every point, use Batch Export ▸ Each frame: its curves, with error bars. |
| **Cancellation does not forcibly interrupt low-level file reading.** A Compare directory scan or single-table read, or the current Fitting read, may still be returning. | Press Cancel or Stop once; the request is registered, and the unfinished import is discarded or the frame fit skipped when reading returns. Completed series and results stay (sections 6.5, 7.2). |
| **Curve import supports `q I [sigma]` and `q I1 I2 …`, not interleaved `q I1 sigma1 I2 sigma2`.** A headerless three-column table cannot identify its meaning by numbers alone. | Choose the layout explicitly in **Import Curves…** (section 6.1). For separate errors per curve, provide one `q I sigma` file per curve. Fixed imported selections cannot watch for new columns/files. |
| **Geometry Calibration can put a wrong solution first.** A wrong **Estimated distance** misleads, also with the right standard and with **Auto Detect**, and the field can hold a distance you did not type (section 4.3). | Choose the standard yourself when you know it. Leave **Estimated distance** on *Optional* or type the approximate distance of your set-up; set a number you did not type back to 0 unless it is that distance. Before **Apply**, check the distance and that the drawn rings lie on the rings of the image. |
| **Geometry Calibration fits silver behenate, LaB₆, CeO₂ and a LaB₆ + CeO₂ mixture only.** The AI's geometry search recognises images of Si, Al₂O₃ and Cr₂O₃ by their names but cannot fit them (section 10.5). | Calibrate such an image with pyFAI and give GIMaP the `.poni`: name it in the notes, or put it beside the frames (section 4.2). |
| **An instrument profile is matched by detector name and frame size only. A calibration found by the automatic search may belong to another experiment.** Even matching detector dimensions cannot establish which run produced a calibration (section 4.2). | Check the file named in the report and the distance in **Geometry**. Supply the calibration or name it in the notes. Remove a wrong profile with **Geometry…** ▸ **Delete profile**. |
| **Powder rings leave rot3 undetermined.** They also constrain tilt weakly at SAXS distances (±0.2° on the public silver behenate frame). | Use a wide-angle standard for rot1 and rot2. Check rot3 on a GISAXS ridge and the horizon in **Geometry ▸ Check rot3**. GIMaP reads flat-detector `.poni` orientations 1–4 (section 4.2). |
| **The automatic Yoneda search and horizontal cut average detector rows.** Analyze warns when the horizon varies by more than half a row across the frame. | Add a horizontal band in q for a constant exit angle. Mirror filling needs a measured partner on the other side. It cannot recover the missing wedge. |
| **Run Automatic Analysis (GIWAXS) fits at most 12 candidate peaks**, the most prominent; the window has no setting for more. The others are left out of the peak table, the sector comparison, the rings and the series hints; *Decisions* says how many candidates there were (section 5.5). | Run `tools/gimap_agent.py auto … --max-peaks N` (section 10.7), or ask the AI to run `find_peaks` with `max_peaks`, when you need the full list. |
| **The GIWAXS and GISAXS q are kinematic: no refraction correction.** Near or below the film's critical angle, the q_z of a peak can be off by up to about k·sin αi, and in GISAXS that of the Yoneda band most, by an amount of that order (the report's *Decisions* give the number for your frame); the ± and σ of the report contain neither that nor the geometry's uncertainty ([Physics and Methods, sections 7.9 and 8.6](Physics_Manual.md#79-what-the-report-states)). | Measure above the critical angle for positions, or correct q_z yourself for the material; quote the systematic error separately from the statistical ±. |
| **Hermans f and the texture sentence are not corrected for the exit angle.** f weighs the bins next to the sample horizon most, where absorption in the film or a shadow can lower the intensity; on the public P08 frame leaving out the bins within 5° of the sample plane moves f − f_iso by 17–27 σ_f ([Physics and Methods, section 7.7](Physics_Manual.md#77-ring-orientation-and-the-hermans-orientation-factor)). | Read the ring's notes and the *ring orientation* line of *Decisions*; when they name the ring, check I(χ) near χ = ±90° before reading f as texture. |
| **NeXus/HDF5 files not written by the Lambda server are now read as stored** (earlier versions transposed and flipped them). A geometry or mask made on such a file with an earlier version no longer fits it (section 5.1). | Check the beam centre and masks of such data; calibrate again if needed. |
| **The normalisation divides the frame after the background and dark frame are subtracted** in the frame's own units. Transmission and thickness are entered by the user (section 5.2). | Set the background's **Scale** (and the **Dark scale**) to the ratio of exposure times or monitors of the frame and the background. |
| **XRR intensities are raw**: no monitor normalisation, footprint, background, resolution correction or θ offset. Analyze masks and bad-pixel search are not applied (section 9.2). | Apply the required corrections to the exported curve and check the ROI and valid-pixel counts. |
| **Format Converter does not read EDF.** | Use **Analyze ▸ Batch Export ▸ Converted detector frames**. |
| **The uint16 modes of the Format Converter mark pixels without data only in HDF5**: in a TIFF or CBF such a pixel holds 65535 and reads back as a valid value (section 9.1). | Keep the stored data type (the default), or write HDF5, when the gaps and bad pixels matter. |
| **SEM sizes depend on the segmentation**, and the default StarDist weights are trained on cell nuclei, not SEM. The scale is never read from the image. | Enter the calibration from the scale bar; check every outline; prefer a model trained on SEM images. |
| **2D Prediction reads CBF frames only.** | Convert other frames to CBF with the Format Converter. |
| **2D Prediction applies no geometry and no correction to a frame**: the module's own preprocessing is all that runs (section 11.3). | Use a module only for data of the set-up it was trained for. |
| **The Ag and Cu distribution modules of 2D Prediction declare no axes**: their maps and curves are drawn on a placeholder axis (0.05–15 nm), which is not a physical scale, and they are not attached to the SEM comparison (section 11.3). | Read only the shape of these distributions. When you know the axes of the trained model, add them to the module's `module.yaml`. |
| After the interface language is switched in a session, a few texts keep the old language (e.g. the outcome line of a finished automatic analysis). | Run the analysis again, or restart. |
| Quitting GIMaP during a Batch Export keeps the files of the frames done, but the batch tables, `README.txt` and the batch's JSON record are not written. | End a batch with **Stop**: it writes them. |
| **The model fit of the automatic analysis chooses the particles from the curve alone.** Run Automatic Analysis (and Batch Export's particle model) fits the cut with Predict parameters among one or two populations of spheres and upright cylinders, at the effective interface (a sphere too large for it, q_z·R above 1.7 at the cut, is flagged in the automatic analysis when αi is known; section 5.5); random cylinders only when you give them as the composition. On the public GALAXI frame of Ag spheres it reports two upright cylinders (section 1.3): broad sizes and a rise at low q can be described by another composition than the true one. 2π/q\* is a length scale, not the particle spacing: on BornAgain simulations of the GALAXI model it was within 10 % of the true distance in 1 of 10 simulations with interference, and a peak or shoulder appeared in all 4 without (sections 1.3 and 5.5; [Physics and Methods, section 8.5](Physics_Manual.md#85-the-model-fit-and-how-the-solution-is-chosen)). | Choose the model from what you know of the sample and confirm it in **Fitting**: Predict parameters with **The particles in Model**, then refine it and judge the residuals. Do not quote 2π/q\* as the particle spacing. |
| **The models for manual fitting are not proposed by Predict parameters.** Core–shell spheres, the 2D hexagonal paracrystal (infinite domains) and Percus–Yevick hard spheres (a 3D liquid) are fitted by Refine and Search only; core–shell spheres at q_z = 0. | Start them from values you know, or from a Predict solution with spheres, then change the family or the interference in Model and Refine ([Physics and Methods, section 9.14](Physics_Manual.md#914-models-for-manual-fitting)). |
| **Fitting is slower with sizes by intensity and with random cylinders.** With a sphere or vertical cylinder and sizes by number or log-normal, **Search the ranges** takes a few seconds as before; with *Gaussian, by intensity* (models of earlier versions) tens of seconds, and with a random cylinder (15–800 ms per evaluation) up to minutes. | Set **Sizes** to *Gaussian, by number*, narrow the ranges of R, h and σR/R before a search, or start **Refine** from values close to the curve. |
| **A vertical cut is modelled for spheres and upright cylinders only**, at one in-plane q (q∥ of its columns), without a q resolution or the specular rod, and only in Fitting ▸ Fit Several Curves Together. Its scales are its own: with several populations their ratio is not tied to the horizontal cut's (section 6.6). | Take the vertical cut beside the specular rod; fit random cylinders and core–shell spheres on horizontal cuts. |
| **The substrate table holds bare Si, SiO₂, plate glass and Au from 4.9 to 30 keV.** An oxide, a polymer film, roughness and the particle layer's own index are not part of it (section 6.2). | Type δ and β of an effective substrate, or use the two paths with a free r. |
| **The posterior check (MCMC) is slow for random cylinders and samples one mode.** A random cylinder takes 15–200 ms per evaluation (hours for a check); the run explores the mode it starts at (section 6.2). | Check spheres and upright cylinders; compare other modes through Predict's other solutions. |
| **Predict parameters knows Fitting's model only** (sizes Gaussian by number, the LMA; spheres with two paths and a free reflection coefficient r when Model has both angles, else at the Born term of the cut's q_z or the effective interface). It was checked on simulated curves of an independent model and on BornAgain simulations, not on measured curves of known particles; physics outside the model (roughness, other shapes) can give a composition a high probability that it does not deserve. The waves the surface reflects (the four-term DWBA) matter for spheres beyond q₊R ≈ 1.7 (R ≈ 3 nm at α_i = 0.4°, 12 keV): with both angles R is off by about 6 % (median; P90 19–40 %) and the composition right in about 60 % of such curves; with the q_z alone R is off by about 10 % (q₊R 2.3–3.5) to 40 % (above) and a sphere is usually taken for upright cylinders; upright cylinders are not affected. With both angles on a curve whose reflected wave is weak (far above the critical angle), a smaller sphere with a large \|r\| can fit as well, and on BornAgain's model of the GALAXI frame (broad log-normal sizes, αf at the critical angle) the free r let wrong sizes fit within the errors (with one sphere population R 16–500 nm, once at the 500 nm bound, on the three simulated draws; with the composition chosen one 16 nm sphere without a paracrystal on two); and on two-sphere curves without reflection this network misses the true sizes somewhat more often than the previous one ([Physics and Methods, section 9.13](Physics_Manual.md#913-predict-parameters)). | Judge every solution by its residuals and by what you know of the sample; give both angles for spheres beyond q₊R 1.7; far above the critical angle check the fitted \|r\| (`refl_amp` in Model) against the substrate's and fix it if it is far off, then refine. |
| **Run records do not hold everything needed to repeat a run.** They identify `software_version`, the network's record and Predict parameters' fixed seed; masks are counted, not stored as pixel lists. | Keep the input, settings and model files with the exported records. |
| **API keys saved in GIMaP are kept in plain text** in the user data folder (`claude_api_key.txt`, `assistant_provider_keys.json`; section 10.1): whoever can read that folder can read them. | On a shared computer, or with `GIMAP_HOME` on a shared drive, set the key as an environment variable instead, or **Remove** it after use. |
| **Sputter ▸ Analyze reads Cu with networks trained for Cu on two polymer films (a homopolymer and a block copolymer)**. | For Cu on another substrate, choose **Polymer** for a polymer film and compare the model image with the measured one before using the values. |
| **Sputter ▸ Analyze away from the networks' energies rests on the joint fit.** The network for Au and Ag on TiO₂ was trained at 8–15 keV and αi 0.25–0.55°: in that range a TiO₂ scan is analysed as usual at its own energy. The other networks were trained at one wavelength (0.105 nm for Cu, 12 keV for the others) and αi 0.40°; within 5 % of that wavelength (for Au: at 12 keV only) a scan is analysed as usual. Farther away, and on TiO₂ outside 8–15 keV, the package computes the optical constants at the scan's own energy from its table (4–30 keV), the networks give only the starting solutions, and **Auto** picks **Refine**; the result is the joint fit's. A photon energy outside 4–30 keV is refused when the scan is imported, with the reason. A TiO₂ scan at an αi outside 0.25–0.55° is analysed, with a warning in the log of the network step under **Details**. | Away from the networks' energies use **Refine** or **Full** and judge the result by its fit deviation. Keep the photon energy that was measured: changing it to match a network scales every q. |
| **Sputter ▸ Analyze runs the analysis package that comes with GIMaP**, which analyses only the pairs of metal and substrate the package has a network for; its values are model values that were not compared with an independent measurement of your sample (section 8.1). The frames it reads are kept, one copy per request that changes the import (rate, geometry, mask, substrate …), in GIMaP's local folder, at most 5 GB together by default: the ones not used for the longest time are removed first. | Judge a solution by its model image beside the measured one, and compare with SEM, AFM or XRR where you can. **Details ▸ Storage** lists the imports, removes them and sets the limit. |
| **Sputter reads folders of CBF frames only** (chapter 8). | For NeXus series, use Analyze's series map (section 5.6) and Fitting's in-situ series (section 6.5). |
| **Sputter keeps every result folder** in the user data folder unless you chose another place; nothing removes them, and **File ▸ Save Project** holds the Sputter form and references its result folder. | For long scans choose **More options ▸ Save results in** on a data disk; delete old folders yourself; open a result again with **Open Result…**. |
| **TensorFlow runs on the CPU on Windows and macOS**; a GPU set-up is not configured. | Use the bundled CPU runtime. |

## 14. Troubleshooting, FAQ, version and contact

### 14.1 A file cannot be loaded

The message under the image says why in words: an empty file (still being written?), not a detector
image, a damaged or incomplete NeXus file, no permission. A NeXus/HDF5 file that another program is still
writing cannot be read until it is closed; one left open by a program that ended can be read after
`h5clear -s` on a copy; one compressed with a filter this installation cannot read needs the Python package
`hdf5plugin` (section 5.1, *NeXus and HDF5 files*). Otherwise check that the format is supported:
`.cbf` needs `fabio`, HDF5/NeXus files need `h5py` and the dataset where the reader expects it. To ask
for another file format, use **Help ▸ Report a Problem…** or write to the author (section 14.12).

### 14.2 A `.csv` curve asks for its columns

Fitting reads a CSV by the names of its header (section 6.1). When they do not say which column is q, I and σ — two
intensity columns, q of two axes, a table without a header of more than three columns — **Choose Columns** asks; pick
the columns there (q, I and σ must be three different columns), or rename the header to `q, I, sigma`. A file
that is neither `.dat`, `.txt`, `.csv` nor `.h5` is refused with what Fitting reads.

### 14.3 1D Predict Does Not Start

*Predict parameters* (in Fitting and in the 1D Predict window) checks its network
(`src/gimap/features/fitting/domain/posterior/data/flow_v1.npz`) against the SHA-256 in its record `flow_v1.json`
before it runs, and does not start when one of the two is missing or the weights were changed: the message names
the file. The portable version and the Git repository both contain them. In a source checkout:

- A checkout last updated before the network was added does not have it yet: update it (section 2.4).
- A network file that you changed yourself is left alone by the update script; `git status` lists it as modified.
  `git checkout -- src/gimap/features/fitting/domain/posterior/data` puts the files of the repository back (and
  discards your changes in that folder).
- Git hands the weights out as binary (`.gitattributes`), so a checkout does not convert them.

The least-squares methods of Fitting (*Refine the values in Model*, *Search the ranges, then refine*) use no
model files: the check does not concern them. Without the network, Fitting does not recommend Predict parameters
for a GISAXS cut, and the line under **Method** names the missing file when you choose it (section 6.2).

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

### 14.7 Sputter does not analyse, or TensorFlow is missing

**TensorFlow and other packages.** `requirements-tensorflow.txt` asks for TensorFlow `>=2.15,<=2.16`
(`tensorflow-intel` on Windows), which in practice installs a 2.15 release; 2.15.1 is the tested version. For a
missing package, activate the right Python environment and run `pip install -r` with the file the message names
(`requirements-full.txt` for everything).

**Sputter ▸ Analyze.** The job status in the input bar says what stopped it:

- *The analysis package was not found*, or *… is not an analysis package: … is missing*: the program folder has no
  complete `utils/sputter_client` (the package that comes with GIMaP), and no package was chosen. Choose
  the package's folder with **Choose Folder…** (the folder that holds `code` and `models`), or set the environment
  variable `GIMAP_SPUTTER_CLIENT` to it (section 8.1).
- *Give the deposition rate (Å/s or nm/s)*, *Give the photon energy or the wavelength* and the like: a value of the
  input bar or of the geometry is missing; the line under the bar names the geometry still to enter, and **Edit**
  shows its fields.
- *The analysis stopped with an error (failed at …)* or *(exit code …)*: press **Details**; the log under
  **Analyze options** ends with the package's last lines. A last line *no network for … on …* means that the
  package has no network for this metal on this substrate; the line names the pairs it has.
- *No progress for …: the analysis package's processes are all waiting*: the package has stopped working without
  ending. The note in the live view shows the last line of its log, and **Stop Analysis** ends it (section 8.1,
  *When nothing happens*).
- An import that stops with *this scan is at … keV …; the optics are tabulated from 4 to 30 keV*: the photon energy is
  outside the package's table of optical constants; check the energy or wavelength under **Details ▸ Sample and
  geometry** (chapter 13).
- *The analysis finished, but this package does not write the data of the main screen*: the package is an older
  one; its report is in the folder named. Use a newer package.
- **Open Result…** says *… holds no result of the analysis (main/main.json)*: the folder holds no finished
  result of Analyze.

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
folder. Predict parameters uses the network that comes with GIMaP (`tools/ml1d` trains another one). Sputter uses
the analysis package that comes with GIMaP in
`utils/sputter_client/`, or another one you choose (section 8.1); a new package comes with a GIMaP update.

**Where are the results of 1D Predict saved?** In `AI_Fitting_Output/v5_<date>_<time>_<µs>/` in the
GIMaP user data folder. Fits you save yourself
(**Save**) go where you choose. Chapter 12 lists every output.

**Which fitting method should I choose?** For a GISAXS cut whose particles you do not know, **Predict parameters**:
it is Fitting's one network method and gives every solution with its probability. When Model is close to the
curve, **Refine the values in Model**; when the start is poor but you know the family and its ranges, **Search the
ranges, then refine** (section 6.3). *Predict parameters (fit1d)* and *Find the particle shape (no AI)* were
removed in 0.2.0, so that there is one network method to choose.

**Where are Sputter's results saved?** In `sputter/<scan>-<code>/` of the user data folder, or in the folder
set with **More options ▸ Save results in** (section 8.2; chapter 12); a result of Analyze is in its subfolder
`iampred-<code>/`. They are kept until you delete them.

**Does anything leave my computer?** Your data leave it only when you use the AI assistant with a remote
brain (Claude Code, the Claude API or another provider's server): then the frame's status and the reduced
curves are sent as numbers, never the detector file, and the small q-map image only when you allow it
(section 10.6). With a model on your own computer or lab server, nothing leaves that network. GIMaP itself
does not go online by itself. **Help ▸ GitHub Repository**, **Check for Updates…**, **Report a Problem…** and
the model library of 2D Prediction open a page in your browser. **Report a Problem…** puts the versions of
GIMaP and of your system (no paths, no data) on the clipboard and opens a new issue on GitHub: nothing is sent
until you paste it there and submit the issue yourself. StarDist, Predict parameters and 2D Prediction run on
your computer and download nothing.

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
