# Changelog

What changes for someone who uses GIMaP, version by version. Developer notes are in `docs/`.

## 0.2.0 — unreleased (pre-release)

What changed since 0.1.0 (2026-10-13). New are the **Sputter** workspace, **SEM Particle Analysis**, the
**Edit**, **Analyze** and **Fitting** menus with a fuller **Help** menu, NeXus/HDF5 files read by what they
hold, a source folder that starts with one double-click, and a reference of the methods, *Physics and
Methods*. Some results differ from those of 0.1.0, and a model folder imported in 2D Prediction has to be
chosen once more: read *Updating from 0.1.0* at the end of this version before you update.

### Sputter (new workspace)

- **Sputter** (sidebar, `Ctrl+5`) follows an in-situ sputter deposition from the PILATUS CBF frames of one
  scan, taken while Ag or Au is sputtered onto a polymer. For every time window it gives the particle size R,
  the spacing D, the island height H, the growth stage, the metal thickness where it can be stated, and how far
  each value can be trusted (manual chapter 8; Physics and Methods chapter 12). Cu is accepted and labelled as
  approximated with the Ag model, also on saved pictures.
- **Predict in three steps**: choose the scan folder, check the geometry, press **Predict**. Each value of the
  geometry says where it comes from: this folder's last values, a geometry preset of the beamtime (**Save
  Geometry as Preset…**), the file header, or the values used last (*please check*). The model is part of GIMaP
  (`utils/sputter_toolkit`) and runs in GIMaP's own Python, in a process of its own; the first Predict installs
  PyTorch there (about 1.2 GB). **Cancel** ends the whole process tree. Frames are read by no more processes than
  the free memory holds.
- Each request gets its own result folder, `sputter/<scan>-<code>` in the user data folder (or **More options ▸
  Save results in**): the same request reuses it, another geometry or model gets a new one. **More options ▸ Open
  Results Folder…** shows an earlier result.
- **Two explanations.** With a deposition rate (and at least 10 windows with the metal's peak), the default is
  the **literature method (mass balance)** of the P03 in-situ GISAXS studies: D = 2π/q* of the direct peak, R
  from the nominal thickness δ = rate × (t − t0) by mass balance for hemispheres (grade Lit), and the
  percolation where 2R/D reaches 1. Its interval holds the scatter of the windows and the errors of the rate
  (±10%) and t0 (±2 s, ±0.5 s when given); when 2R/D may reach 1 within them but need not, the percolation is
  *not determined*. After it D = 2π/q* goes on and R is not given. Peaks that probably belong to the polymer,
  jumps and outliers are marked and left out; late peaks after a gap are compared with the trend of D. Without a
  rate the default is the model's own **physical path**: one path of stages through time with the network's
  sizes in every island window (one source of R and D per window), and the film onset with its interval; the
  legend says whether the onset is established. A new deposition rate keeps the explanation shown; taking it
  away returns from the literature method to the physical path.
- **Tabs**: Size Trend, Literature, Structure Sketch, Windows and Reliability. **Size Trend** and **Literature**
  open in a simple view (R and D with their trends and the percolation or film onset; D and 2R/D against the
  published percolation band); **Show details** adds the metal thickness, the TS length L, the direct 2π/q*
  with its checks, the published D(δ) curves, the coverage and the network's mass balance. The **Literature**
  tab compares with published values (Au/PS, Ag/PS, Ag/PMMA, Cu/graphene; thresholds by metal and host, each
  with its citation and DOI) and lists checks — consistent, inconsistent or not testable — with their
  references. The literature-method R agrees with the literature by construction; the page says so.
- **Metal thickness** where it can be stated: the island-volume equivalent t_eq and, from a deposition rate, the
  nominal thickness. Film windows say *thickness not determined*.
- **Intensity modulation period (sample stage translation?)**: the sample stage is translated back and forth
  during the deposition (against beam damage), and near the ends of its travel the beam can reach the sample
  edge or leave the sample; the data alone do not prove the source. The toolkit's summary names it
  `modulation_period_s` (`rotation_period_s` stays for older readers).
- **Windows where the beam may be off the sample** are flagged from the intensity alone, a whole dip at a time:
  a dip whose intensity falls below half of the level within one modulation period before and after (without a
  modulation only a dip below 20% within 10 s on each side), with the windows next to it that stay below 80% of
  their level. They keep their values in the Windows table (column *Beam may be off the sample*, with the
  reason), are drawn as hollow grey points in Size Trend and counted in the result line and the Reliability
  report, but they cast no stage vote and are left out of the trend lines, the back-and-forth counts, the mass
  check at the film onset and the literature method's trend of D and checks of the network. Without a
  modulation the page says the beam may also have been interrupted.
- **More options ▸ Deposition end (s)** (optional, remembered per scan folder like the rate and t0): the
  shutter is closed by hand and a scan can go on after the deposition. After the end the nominal thickness stays
  at rate × (t_end − t0), and the literature method uses that constant δ in the later windows. Left empty, the
  deposition lasts to the end of the scan, and the notes say that no end was given. Changing the rate, t0 or the
  end rewrites only the physical files. A rate that is not positive or an end not after t0 is refused, also by
  **Add Refined Windows…** and by the toolkit's command line.
- **Show in Analyze** (right-click a row of Windows, the Structure Sketch or a point of Size Trend): the raw
  detector frames of that time window open in Analyze in place of the frames listed there and are shown as one
  sum, with αi — and a beam centre given for Predict — as in Sputter. Distance and wavelength come from an
  instrument profile within 0.5 mm and 0.1% of the Sputter values; without one the notice says which values to
  enter (no profile is created).
- **Island height H from the q_z fringes**: after a result is shown, the minima that move to lower q_z in the
  vertical cuts at low q_y are measured in the background (`utils/sputter_fringes`): Size Trend gets the panel
  **H, island height from q_z fringes (nm)** (also in the simple view) with 68% intervals and the nominal
  thickness for comparison, Windows the columns **H fringes (nm)** and **H fringes 68% (nm)**, Reliability a
  paragraph. H is the vertical size of the islands or the layer (refraction-corrected); it is the film thickness
  only once a continuous film has formed, and it is given only while a minimum is in the detector's q_z range and
  its order is known.
- The model's notes and reasons are shown in the interface language.

### SEM Particle Analysis (new tool window)

- **Tools ▸ SEM Particle Analysis…** measures particle sizes in calibrated SEM images, or imports particle-size
  CSVs, independently of scattering prediction. The nm per pixel comes only from a value you enter or a known
  scale-bar length, never from text in the image. A local StarDist model can separate touching particles;
  existing integer labels and threshold/watershed remain available. StarDist runs in a cancellable local
  subprocess of its own Python environment (`requirements-sem-stardist.txt`), with no image upload and no
  automatic model download; model weights stay outside the source repository.
- Exports include particle radii and diameters, comparison plots, reusable instance-label TIFFs and records of
  the source, the scale, the ROI, the settings and the SHA-256 of the model files. Review the outlines and the
  image calibration before interpreting size statistics; general pretrained nuclei models are not SEM-validated.
- **2D Prediction ▸ SEM comparison…** copies a radius prediction into the same window, when the module declares
  its radius axis (`outputs: axes: row` with key `r`, unit `nm`, `min` and `max`). Otherwise nothing is attached
  and the Activity log says why; the axis records say whether the module declared them (`declared`).

### Menus and Help

- The menu bar is **File, Edit, View, Analyze, Fitting, Tools, Help** (manual section 3.6):
  - **Edit ▸ Undo / Redo** (`Ctrl+Z`, `Ctrl+Y`) act on the set-up in Analyze or the model in Fitting's Single
    analysis; in Analyze the entry names what it would change (*Undo: cut regions*).
  - **Analyze**: Run / Stop Automatic Analysis, Repeat Analysis (`F5`), Export Curves… (`Ctrl+E`), Batch
    Export… (`Ctrl+Shift+E`), Batch Export a Folder…, Send to Fitting, Send Series to Fitting…, Send Series to
    Compare, **Set-up Files** (settings, cuts and masks), **Figures** (Save Image…, Save q-Map Data…, Save View
    Data…) and Clear Frame List.
  - **Fitting**: Open Curve…, Fit a Folder of Curves…, Fit (`Ctrl+Return`), Stop Fit, Load / Save Model…, Save
    Data and Fit…, Save Plot…, In-situ Series ▸ Start / Stop, and 1D Predict — Fit Many Curves….
  - Every entry first shows its page, then does what the page's own button does. An entry that cannot run now is
    greyed out, and its tooltip says why (*Open a frame first.*). The keys belong to the page and act while it
    has the keyboard focus; the menu entries work from any page.
- **Help** has new entries:
  - **Physics and Methods** (the new reference, see *Documentation*);
  - **Keyboard Shortcuts…**, a table of every key of the menus and pages that you can copy;
  - **Example Data**, which copies one of the four public examples that come with GIMaP to a folder you choose
    (by default `Documents\GIMaP examples\…`) and opens its frame in Analyze;
  - **Report a Problem…**: it puts the versions of GIMaP, the system, Python, Qt, PyQt and numpy (no paths, no
    data) on the clipboard and opens a new issue on GitHub in your browser; nothing is sent until you submit it;
  - **Open Error Log Folder**;
  - **Check for Updates…**, which opens the releases page on GitHub, newest first and pre-releases included
    (GIMaP itself makes no network request);
  - **How to Cite GIMaP…** with **Copy** and **Copy BibTeX**, read from `CITATION.cff`;
  - **Third-party Notices**.
- **Help ▸ About GIMaP** also shows the licence and the versions of Python, Qt, PyQt and numpy, with **Copy**.
- **Help ▸ User Manual** and **Physics and Methods** open as a page in your browser, in the edition of the
  interface language (Third-party Notices opens the same way, in English). The Contents links now jump to
  their chapter.

### Analyze

- **Opening data**: for a source of several frames, the new window **Import Frames** asks which frames to list
  (all, a range with every n-th, or frame numbers counted from 1); **Choose Frames…** changes it later, and only
  references are listed, so a stack of thousands of frames opens quickly. **Cancel Import** stops a slow import;
  a folder whose frames lie one level down (one subfolder per detector) is opened there. A source is taken off
  the list with a right click (**Remove Source from List**, **Show in Folder**, **Copy Path**) or `Delete`.
- **Watch** (beside **Clear**) watches a folder during a measurement: new frames are listed and the newest is
  shown once it is completely written; with **Export ▾ ▸ Export New Frames While Watching** each new frame is
  exported once, into `gimap_analysis/` beside the data.
- **Curves** has a **Log q** option alongside Log I. Plot selections keep their original data coordinates when
  the axis scale changes.
- **Geometry…** and **Enter Coordinates…** stay open while applying the beam centre. Coordinates show four
  decimal places; **Centre step** sets the remembered increment for arrow buttons, keys and the wheel. **Live
  Preview** refreshes the curves as the centre changes, combines rapid adjustments and preserves the curve zoom.
- **Make Left and Right Symmetric** finds the horizontal axis with a global coarse search, then refines it using
  the displayed column mean and log-distance weighting so low-q structure keeps its influence. Masked pairs stay
  fixed during refinement. Only x changes; 0.0001 px is the numerical stopping tolerance, not a measured
  uncertainty.

### NeXus / HDF5 files

- Every NeXus/HDF5 suffix is read: `.nxs`, `.nx`, `.nx5`, `.h5`, `.hdf5`, `.hdf`, and every frame of a file. The
  content decides what a source is, a file name only proposes: one frame per file, a whole series in one file,
  one file per detector module (P03 Lambda 9M: `…_m01.nxs` … `…_m11.nxs` are a set only when they hold the same
  dataset and module shape at different module positions; they are stitched), or a master with its data files
  (Eiger). A folder lists each source once.
- Analyze's Data step says what a source is (“11 module files · 240 frames · stitched”), how its frames are
  oriented, and what is left out: a module file that is missing or holds fewer frames (the frames every module
  has are used), a data file that is missing, other image datasets in the file. Asking for a frame beyond the end
  is an error, never the last frame instead.
- **Orientation of non-Lambda files changed.** Files written by the Lambda server (P03) are read as before: the
  modules stitched by their translations, then transposed and flipped. Every other NeXus/HDF5 image is now read
  as stored, row 0 on top, like the CBF or TIFF of the same detector; before it was transposed and flipped as
  well. A geometry or a mask made on such a file before no longer fits it: check it (a square frame still matches
  its old instrument profile by size). GIMaP's own HDF5 frame exports read back unchanged; they now mark
  `/entry/data` as NXdata with `signal="data"`.
- Header values are read in their units: an exposure in ms (as the Lambda files store it) becomes seconds,
  energies in eV or keV, lengths in mm, cm or m; a value in a unit GIMaP does not know is left out rather than
  guessed. In unsigned integer data the largest value of the type means “no data” and is left out.
- Files compressed with bitshuffle or LZ4 need `hdf5plugin`, now in `requirements.txt`. A file still being
  written is read as far as it is and read again when it changes; a file another program holds is said so, and
  one that its writer on the same Windows PC holds locked is read once it is closed; a file left open by a writer
  that ended says how to read it (`h5clear -s` on a copy).

### Analyze: series

- A frame bar (“Frame [4] of 12”) above the steps for a source of several frames.
- **One cut for the whole series.** While the cut is automatic, every frame of a Series map, a Batch Export, a
  Send Series to Fitting and a watched folder's export is cut where the frame on screen is (GISAXS: the Yoneda
  rows and the beam-centre column; GIWAXS: the ring of I(χ)). *Find the cut again in every frame* (Series ▸
  Options, and the Batch Export dialog) turns this off. The sum that Run Automatic Analysis sets is no longer used
  for series work.
- **Series map**: drawn at the true x, without bridging gaps (the exports keep the measured x); the frame axis
  counts real frame numbers from 1 (the frame in its file for one multi-frame file) or time (time stamps, or
  frame × exposure); a frame range under *Options*; for the GISAXS horizontal cut a choice of halves (± / + / − /
  |x| / mean); a title that says the curve, the frames and the cut; **Wide**. Exports are named
  `<series>_<curve>[_<halves>]_…` with a JSON record of the settings; the frame picked is saved under the name
  Batch Export gives that frame.
- **Watch** follows a source that grows: a file or a set of module files written frame by frame is listed at
  once, and each look (every 2 s) takes the frames it already held at the look before. A set of module files is
  exported once it is whole: at once when it has as many module files as another set of the folder, otherwise
  after about 10 s unchanged; a set with fewer module files is exported as it is after about 30 s. F5 and the
  watch read a frame again when its files changed; a master and its data files are one row; a source taken off
  the list comes back when it grows.

### Batch Export and Send Series to Fitting

- File names keep the 0-based frame index (`_frame0003`); headers, labels and titles say the frame from 1
  (“frame 4 of 12”). A set of module files is named by its scan, without `_m01`. Every curve file says in `#`
  lines which frame, which sum and which files it comes from.
- The dialog checks the folder before anything is reduced, names the cut every frame gets (and waits while the
  frame on screen is read again without the automatic analysis' sum), and asks before it writes over an earlier
  export of the same data: **Replace**, **Keep Both in a New Subfolder** or **Cancel**. Enter answers Replace
  only when this export writes again every kind of file Replace would remove; otherwise Keep Both, and the
  question names what Replace would remove.
- GISAXS: **Halves of the cut** — the mean of both halves, the q > 0 half or the q < 0 half — for the model fit;
  the curve for Fitting still holds both halves and names the choice. The model table has the columns `halves`
  and `points`; a frame without counts or without points on the chosen half is listed with its reason, and a
  batch whose model fits failed no longer ends as a success.
- **Send Series to Fitting** writes into a folder of its own, `<name>_fitting/`, beside any Batch Export of the
  same data.
- Tables of all frames: frames whose points differ only slightly are matched row by row; grids that really
  differ are interpolated, never across a gap. `<name>_batch.json` lists every file written, the cut of every
  frame and every failed frame with its reason.

### Fitting

- **Single analysis** has one **Method** drop-down, and the button carries the method's name. When a curve is
  opened the page chooses the method: a curve whose model is not yet edited gets **Find the particle shape (no
  AI)**, which needs nothing beyond GIMaP; an unfitted GISAXS cut gets **Predict parameters (fit1d)**, or
  **Find the particle shape (no AI)** when the fit1d kit is missing, with a line that says why.
- **Results** says what to do next (**Refine These Values**, **Fit Again**, **Fit Every Curve in This
  Folder…**). With **Resolve values at the edge of their range automatically** (on by default) a value that
  stops at the edge of its range is fitted again with the range widened, listed under *Done automatically*; a
  value still at the edge is reported as a one-sided bound (for example *≥ 0.549*). Select values and press
  **Fix**, **Let It Fit** or **Range…**: the next **Fit Again** uses them.
- **Predict parameters (fit1d)** fits spheres and vertical cylinders with fit1d, a kit with its own networks
  and physics (number-weighted sizes, Hosemann paracrystal, spheres at |q| = √(q∥² + qz²)). It comes with GIMaP
  in `utils/gisaxs1d_gui_kit` (about 62 MB: two numpy networks and packed form-factor tables; numpy, scipy and
  threadpoolctl only, no TensorFlow); the environment variable `GISAXS1D_KIT_HOME` points to another copy. When
  the kit is missing the method says where GIMaP looked, and everything else in Fitting works.
  **Particles** chooses what it fits:
  *The particles in Model* (1–4, with the values you fixed and the ranges you changed passed as constraints,
  listed under **Constraints and ranges**), or *Try 1–3 vertical cylinders* / *Try spheres and vertical
  cylinders (1–3)*, which ranks every composition by BIC and lists the others under **Other compositions**.
  **General network** uses `cond_v2_ft4`, **Au nanoparticles on Si (adapted)** `cond_v2_ftr1`; set GISAXS / SAXS
  and, for a Yoneda cut, the qz of the cut in nm⁻¹ (unset means zero, with a warning). It runs in a separate
  process and supports Stop. Results, plots and CSV/JSON keep the kit's own total curve, component curves,
  effective σ, parameter errors, states and warnings; the values copied into Model are only a starting point for
  later refinement with Fitting's own physics. A changed Model, curve selection or fit1d setting marks the fit
  stale and prevents exporting it as a current fit.
- **1D Predict** has a fourth fit method, **fit1d (network + physical fit)**, with the same kit and the
  composition you give (**Parameters…**: search, curve type, network, the qz of the cut and a systematic error).
  The row says which parameters have another equally good solution, sit at a bound or are not pinned down.
  Selecting a fit1d row keeps its own curve and components visible while copying a manual starting model.
- **In-situ series**: **Predict each frame (fit1d)** as a method, preferring the solution that
  continues the previous frame; **Fit Every Curve in This Folder…** in Single analysis starts it on the curve's
  folder. **Files** may also name detector frames (CBF, TIFF, EDF, NeXus), which are cut as Analyze cuts the
  frame it has set up, in memory only (**Take Analyze's Set-up**).

### Fitting: the halves of the cut

- **In-situ series** has its own **Halves of the cut**: the mean of both halves, the q > 0 half or the q < 0
  half — never both together. It starts on what Analyze sent the curves with, else on the halves of Single
  analysis, else on the mean, and says where its value comes from. A frame without points on the chosen half
  fails with “No points on the q > 0 half” (or q < 0); it is never fitted on the other half instead.
- Its results table has one row per frame with **Points** and **Halves**; the saved table adds `points` and
  `halves_used`, and the JSON record (`gimap_series_fit_v2`) the halves and where they came from, every frame's
  state and reason, and the poor fits (χ²ᵣ above 10).
- A second Start, other curves or opening a project ask before results that are not saved are dropped
  (**Save**, **Discard**, **Cancel**).
- **Single analysis** keeps its four choices. *Find the particle shape*, *AI proposal* and *fit1d* cannot fit
  both halves on |q|: they switch to the mean of both halves and say so. A mean never interpolates across a gap
  of the other half. *Find the particle shape* and the model fit of Batch Export take a cut of any length (more
  than 1000 points are merged).
- A GIWAXS I(q) sent to Fitting gets a note: Fitting's models are particle form factors, not peaks.

### Command line and tool windows

- **Format Converter** reads every NeXus/HDF5 suffix and lists a set of module files once, stitched and named by
  its scan; the metadata in a CBF header is written as ASCII JSON. Its default data type is now called **Keep
  the stored data type** (it was *Preserve / use loader data type*); see *Fixed* for what it writes.
- **XRR Series Extractor**: **Folder…** chooses the folder of a series; a series of NeXus files that hold one
  frame each is read; the JSON record lists every file read (`source_layout`).
- **Geometry Calibration**: a multi-frame file is calibrated on one frame or on the sum of all frames; importing
  a calibration made on summed frames reads that sum and selects its standard.
- `tools/gimap_agent.py auto`: a set of module files is analysed once, in a folder named by its scan; `--series`
  takes files that hold one frame each as one series; a source with a part missing is an open item of its
  report. The default sum of a series is now the last fifth of its frames, at most 10, and the last frame alone
  below ten frames (was: the last 10 frames). Run Automatic Analysis in the window sums the same frames.

### Exported text

- Text files are UTF-8 without a byte-order mark. GIMaP's own symbols are written in ASCII (`1/A`, `deg`,
  `2theta`, `alpha_i`, `sigma`, `+/-`); file and frame names and paths are kept as they are. A script should read
  them with `encoding="utf-8"`.

### Installation and updating

- **The fit1d kit comes with GIMaP** (`utils/gisaxs1d_gui_kit`, about 62 MB), so **Predict parameters (fit1d)**
  and the method *fit1d* of 1D Predict work after a plain download; before, the kit had to be put there by hand.
  Only its runtime files are bundled: `SOURCE.json` lists what was left out (example measurements, results on
  measured data, research notes) and the text changes, `FILES_SHA256.json` the checksum of every file. The first
  fit unpacks the form-factor tables once into `~/.cache/gisaxs1d_tables` (about 74 MB).
- **A source folder starts with one double-click.** `Start_GIMaP.bat` in a clone or a Download ZIP of the
  repository sets up a private environment in `.gimap` of the folder at its first start: uv (pinned version,
  checked against its SHA-256), Python 3.10, the packages of `requirements.txt` (about 2 GB) and PyTorch for
  Sputter (`requirements-sputter.txt`), with a progress bar over the steps; 3–20 minutes depending on the
  connection. Later starts are immediate; when an update changes a list of packages, the next start adds what
  is missing. When installing the packages fails (for example for lack of memory), it tries again one package at
  a time. Nothing outside the folder is changed. After the folder is moved, delete `.gimap` (or run
  `tools\setup_source_env.ps1 -Reinstall`). A venv or conda environment you made yourself is never used by it;
  start GIMaP from one with `python main.py`.
- **`Update_GIMaP.bat`** runs `tools\update_source.ps1` (git clones only). The script now says which list of
  packages changed (`requirements.txt`, `requirements-sputter.txt`): nothing to do with `Start_GIMaP.bat`, and
  for an environment you set up yourself it prints the `python -m pip install -r` line for each. In a folder
  without `.git` (a portable version, a Download ZIP) it says what to do instead and changes nothing.
- **The Windows package** comes as one ZIP, or, above 1900 MiB, as parts with an installer,
  `00_Download_and_Install_GIMaP-<version>.bat`, that downloads them, checks each one's SHA-256, joins and
  extracts them (`Install_GIMaP_parts.bat` joins parts you downloaded yourself, without checking them). The release page has
  `SHA256SUMS.txt` with the checksum of every file to download. The package holds `licenses\`: the licence files of
  everything in `runtime\`, with GIMaP's `LICENSE` and `THIRD_PARTY_NOTICES.md`. Its Python environment holds
  Python 3.10 and only the packages of `requirements.txt`; PyTorch is not in it (Sputter's first Predict
  installs it).
- **The first start needs little memory.** The preparation of the portable folder (`tools\portable_unpack.py`)
  reads only the files that need it; when it does not succeed, the folder is no longer marked as prepared, GIMaP
  is not started and the window says what to do.
- **A folder path with non-English letters** (such as a localised “Downloads”) stops `Start_GIMaP.bat` with a
  message: Qt cannot load its plugins from such a path. `GIMAP_ALLOW_NON_ASCII=1` skips the check.
- **“No Qt platform plugin could be initialized”** no longer stops GIMaP when `QT_PLUGIN_PATH` points at another
  program's Qt or a Qt plugin path is missing: `main.py` points Qt at the plugins of the PyQt5 it runs with. An
  Anaconda Python that was not activated works too.
- `.bat` and `.cmd` files end their lines with CRLF in every checkout.

### Documentation

- Help documents open as offline reading pages with a fixed, expandable chapter outline, full-text
  section search and a current-section marker. The reader supports English and Chinese, narrow windows,
  light/dark browser appearance and light-colour printing; links between documents still open the named section.
- Formula and command blocks retain their line breaks, indentation and fixed-width alignment, including
  equation numbers. Long lines scroll inside the block. Inline code is preserved in paragraphs, tables and
  links. Previously Qt's HTML export could flatten these blocks into ordinary paragraphs and drop code fonts.
- The **User Manual** is rewritten along the work: a quick start on the public example data, then installation,
  the window, geometry and calibration, Analyze, Fitting, Compare, **Sputter** (chapter 8, a chapter of its own,
  in the order of the page), the tool windows (including SEM Particle Analysis), the AI assistant, Labs, a table
  of every output file, the known limitations in one chapter (13) and troubleshooting (14). The chapters after
  Compare moved up by one. It has screenshots and a Chinese edition, `docs/User_Manual_zh.md`; **Help ▸ User
  Manual** (`F1`) opens the edition of the interface language.
- How GIMaP computes what it shows (formulas, thresholds, error models, their limits) is in a separate
  reference, `docs/Physics_Manual.md` (Chinese: `docs/Physics_Manual_zh.md`), opened by **Help ▸ Physics and
  Methods**. The manual says what to do and what you get, and names the section of the reference where a method
  matters; its last chapter says what GIMaP does not model.
- `THIRD_PARTY_NOTICES.md` (**Help ▸ Third-party Notices**) lists what GIMaP and its Windows package contain
  that is not GIMaP's own work, with the licences. `CITATION.cff` gives the reference to cite (**Help ▸ How to
  Cite GIMaP…**).
- `README.md` is shorter: what GIMaP does, how to install it, a quick start, and where the documents are.

### Fewer small texts on screen

- The grey explanatory lines were reviewed screen by screen. What is needed every time stays on screen: the state
  of each step, results, failures, missing or guessed values, where a value comes from, and what an AI run sends.
  How to use a control, why a default is what it is, and the description of a section are now the tooltip of
  that control or of the section's title (the pointer shows a question mark there). A result is written once per
  screen, not in the step list, the status line and the results box at once.
- A step in the step list keeps its line to one row, cut with “…” when longer; the whole line is its tooltip.
- The value readout under the image is empty while the cursor is not over the image. An idle job shows only
  **IDLE**, without “Ready”.
- Texts that warn are shown as warnings (⚠, warning colour): a fit value at a bound of its range, a curve file
  without σ, the warnings of a solution in Geometry Calibration, guessed values in XRR Series Extractor, the
  flat-field energy in Sputter, and 1D Predict's “the candidates are a starting point”.
- Compare's saved JSON records now hold, for every series, the stages' changes shown in the tooltip of *Odd
  frames and stages of every series* (`stage_details`: from which stage to which, the q that grew and fell most,
  in percent).

### Names and labels

- Quick fits show **χ²/N** where they showed χ²: Run Automatic Analysis (GISAXS) and its saved report and fit
  table, the model rows of Batch Export, the fit results of Analyze's series, and a solution that Analyze sent to
  Fitting (**Show in Fitting**). χ²/N is the mean squared σ-normalised residual without a degrees-of-freedom
  correction; Fitting's own χ²ᵣ is unchanged. The `chi2` column of the model-table CSV keeps its name, and a `#`
  line says what it holds.
- Run Automatic Analysis (GISAXS) said it chose the fit with the lowest χ². It reports the solution ranked first
  by a robust misfit (the mean Huber loss of the σ-normalised residuals); `report.md`, the fit table's tooltip,
  `fit_solutions.csv` and the manual now say so.
- The orientation factor is spelled **Hermans** f (after P. H. Hermans), not “Herman's”, in the automatic
  analysis, its reports and the manual. The keys of exported records (`herman`, `herman_err`) are unchanged.
- A saved AI report is titled *AI report – `<frame>`*, whichever provider ran it (the footer names the model).
- In the Chinese interface, *paracrystal* is 次晶 (it was 准晶, which means quasicrystal).
- Labs texts and identifiers no longer name where a model or method came from by a person's name: the tooltip of
  Trainset's physical background, the ids of the two Cu prediction modules (now `CuPolymer_FF_256` and
  `CuPolymer_SF_4Para`, as written into export records) and the Trainset simulation engine (now `bornagain`; a
  project saved with the former name still loads and is saved under the new one, and form-factor grids cached
  under the former name are still found).

### Fixed

- Analyze said “1 frames imported”; the tooltip of the file name in Analyze's command bar stayed at “reading…”
  after the file was read; the Export step said that everything is written next to the data, although Batch
  Export and saved images go where you choose (each button's tooltip now says where it writes). Sputter asked to
  enter a Sample roll, although an empty roll is used as 0: it now says *0 assumed — please check*.
- **Odd frames** (Analyze ▸ Series, Fitting ▸ In-situ series, Compare): a frame that differs from its neighbours
  in only a few points — such as a few hot detector columns in one frame — is now marked odd ("probably the
  detector"), even when it is too small to show over the whole curve. Before, such a frame could become a stage
  of its own and enter Compare's end state. Two or three such frames in a row, or two at either end of a series,
  are each found; real changes of the sample, however sudden, are still not odd.
- **The unit of q of a curve file.** Single analysis, In-situ series and 1D Predict read a curve file in the unit
  of q its header states. Analyze's `_fit_input.dat` (q in Å⁻¹) used to be read by 1D Predict in its unit
  setting, nm⁻¹ by default, so q was ten times too small; In-situ series read every file in the unit of Single
  analysis' curve. The setting is now **q unit of files without one in their header** and applies only to those
  files (In-situ series takes it at Start). The windows say each file's unit and where it came from; a file whose
  header states both units is neither opened nor fitted. The records (`curve.q_unit_from`, `per_frame.q_unit`,
  `q_unit_from`, `q_units` in `request.json` and `fitting_run_summary.json`) and projects (`unit_setting`) name
  it. A project saved before opens with the unit it saved, unless the file's header states another one: the
  stated unit is then used, and the project's notes say so.
- Analyze, GIWAXS with **Fill gaps from the mirror side**: the error bars came from counting statistics,
  although a filled pixel repeats photons measured on the other side, so σ was too small. They now come from the
  scatter of the pixels, as the manual said — in the curves, the peak fits of Batch Export and the fit-input
  record (`counting_model_valid`). A GISAXS frame, which mirror filling does not change, keeps Poisson σ and
  `counting_model_valid: true`.
- Analyze, GIWAXS: a bin whose pixels all hold 0 counts got σ = 0, which makes a weighted fit (Fitting, the peak
  fits) divide by zero or give that point infinite weight. Such a bin now gets the error of one count,
  σ = √max(Σ, 1)/n, as a line of a GISAXS cut always did (after the intensity corrections: of one count at the
  bin's mean factor). Without intensity corrections every other bin keeps its σ.
- **Export records name the release that wrote them**: `software_version` (for example `GIMaP v0.2.0`) in the
  records of Analyze (`_analysis.json`, the settings file, the series and stages records), Batch Export and Send
  Series to Fitting (`<batch>_batch.json`, and `README.txt` says *Written by GIMaP v…*), Fitting
  (`<curve>_fit.json`, `series_fit.json`), Compare, the XRR Series Extractor, SEM Particle Analysis, 2D Prediction
  and the Format Converter, as Geometry Calibration's export already did. Analyze's records gave only
  `GIMaP Analyze 1`, the name of their format, which they keep as `software`; no other key changed.
- **Format Converter** and **Batch Export ▸ Converted detector frames** wrote every frame as the reader's 32-bit
  float without the reader's mask. A Pilatus CBF (or TIFF) converted to HDF5, TIFF, EDF or NumPy then read back as
  a floating-point frame: its gap codes (−1, −2; about 2·10⁵ pixels of a Pilatus 1M) counted as data in Analyze,
  and the frame was no longer counts; a NeXus frame converted to CBF held 0, a valid count, in the gaps between
  its modules. A frame is now written in the type its file stores, with the gap codes as stored; pixels the
  reader set aside are −1 (NaN in float frames, the largest value in unsigned HDF5 frames), and HDF5 files mark
  every pixel without data in the NeXus `pixel_mask`. A converted frame reads back with the same pixels without
  data, the same values and the same type — except a floating-point frame written as CBF, which holds integers:
  its fractional parts are cut and its negative values read back as gap codes, and the record counts them
  (`negative_data_read_as_gap_codes`, `values_truncated`, `pixels_read_back_without_data`; the Format
  Converter's CBF warning says so). The uint16 choices take their range from the pixels with data and
  write 65535 on the others. Each frame's metadata says what was done (`pixel_values`); the CBF reader now
  records `stored_dtype`.
- Peak fits with a Gaussian or Lorentzian profile no longer fit η (it is exactly 0 or 1): they converge, with
  tens of model evaluations instead of thousands.
- **2D Prediction ▸ Import model...** wrote the chosen model folder into the module's `module.yaml`, a program
  file shared by everyone who uses the installation (and tracked in git). The folder is now remembered in the user
  data folder and takes precedence over the module's `model_path`; `module.yaml` is never written. A path that a
  `module.yaml` already names still loads. The shipped modules name no model folder any more (they named four
  folders of one computer). Once a model is loaded, **Import model...** asks for another folder; the Activity log
  says when a module has no model yet or a remembered folder is gone.
- Fitting's AI proposal and 1D Predict found no model when GIMaP was started from another folder than its own
  (`main.py` double-clicked, an IDE, a shortcut): the models, and relative folders in the settings, are now
  looked up in the program folder.
- A Trainset Build simulation with BornAgain 24.1 ended with an access violation (exit code 0xC0000005) in an
  environment installed with pip or uv, such as the one `Start_GIMaP.bat` sets up: the job worker no longer loads
  PyQt5.
- GIMaP could crash after a long session, when memory was cleaned up in the middle of drawing a window.
- macOS: Batch Export with pictures ended the program (worker threads now have a 16 MB stack).
- Send to Compare opened a Compare page that stayed at “Comparing …”.
- Fitting ▸ In-situ series: refilling the results table of a long series froze the window.
- Saved pictures of square or tall frames no longer cut off the y-axis label; the colour bar is as tall as the
  frame.
- Trainset's `.h5` reference image is the detector image, not its pixel mask.
- Fitting: the tooltip of a particle's scale said the form factor is 1 at q = 0; that holds for spheres and
  random cylinders, while a vertical cylinder's is 10⁻⁶·⟨R⁴⟩/4 (R in nm). The tooltip and the manual say so.
- Fitting: a hidden window-wide **Ctrl+Alt+M** debug key is gone; on German and Polish keyboards it caught
  AltGr+M (µ).
- **The automatic calibration search** (Find Calibration Automatically, Run Automatic Analysis,
  `tools/gimap_agent.py`, the AI's `find_calibration_files`) ranked the ready calibration files (`.poni`, GIMaP
  `.json`) by file time alone. In folders copied side by side — the examples of **Help ▸ Example Data**, a
  download, a clone — the times say nothing: the GALAXI GISAXS frame took the LaB₆ `.poni` of the P08 example
  beside it and was analysed as GIWAXS. Calibration files are now ranked by folder first (the frame's folder, its
  subfolders, the parent folder, sibling folders, then further up), then by time. A calibration named in the
  notes still comes first, and the pixel-size and frame-size checks are unchanged; section 1.3 of the manual no
  longer needs the calibration named in the notes.
- Run Automatic Analysis (GISAXS): the decision *beam centre* gave the new column rounded to four significant
  digits but written to 0.01 px (*x 597.10 → 597.30 px (+0.16 px)* while 597.26 px was applied), and
  `report.md`'s symmetry axis said 597.3 px. Both now give the centre applied, to 0.01 px, as the Results tab.
- Compare's save dialogs proposed a bare file name when no series came from a folder (only series sent from
  Analyze), so a table or a picture went to the folder GIMaP was started from. They now start in the home folder;
  a remembered folder that is gone is passed over.
- In a git clone, a model registered by Trainset Build (`modules/generated_<project>/`, whose `module.yaml` holds
  the full path of the weights on this computer) showed as a new file to commit; git now ignores these folders.
- The manual, the README and this changelog named 1D Predict's run folder `AI_Fitting_Output/v5_<date>_<time>/`;
  it is `v5_<date>_<time>_<µs>/` (the microseconds keep two runs started in the same second apart).

Found in development copies of 0.2.0, and fixed before this release (for those who ran one):

- `Start_GIMaP.bat` closed its window at once in a folder whose path holds a “)”, such as
  `gisaxs_gui-main (1)\gisaxs_gui-main` (a ZIP downloaded twice); and it closed silently when PowerShell could not
  start for lack of memory. Both now keep the window open with what to do.
- **Help ▸ User Manual** showed nothing after the first placeholder in angle brackets (from section 5.8 on); every
  such text is now shown as written.
- A link from one Help document to another (the User Manual to a section of Physics and Methods, and back) opened
  the raw Markdown file. It now opens the other document's page at that section.
- When the Sputter model was missing, the message told every installation to run `Update_GIMaP.bat`, which updates
  only git clones. It now says what to do for the portable version (install or extract the release again), a git
  clone and a folder from Download ZIP.
- In the dark theme, **Structure Sketch ▸ Save Image…** wrote the titles and the scale bar of the side view in
  the light text colour of the screen onto a white background. The picture is now saved in the light colours,
  as every exported figure.
- The Chinese tooltip of the **Sputter** entry in the sidebar was in English.
- **2D Prediction ▸ SEM comparison…** attached the results of the Ag and Cu distribution modules, which declare
  no axes, with a placeholder 0.05–15 nm axis that is not a physical scale.
- Fitting chose **Predict parameters (fit1d)** for an unfitted GISAXS cut and called it recommended also without
  the fit1d kit, and its tooltip called it *the first step for a GISAXS cut*. Without the kit such a cut now
  gets **Find the particle shape (no AI)** (*Recommended for this curve: … (the kit of Predict parameters (fit1d)
  is missing here)*), and the tooltip of Predict says where the kit belongs.
- The screenshots of both manuals still showed 0.1.0 (four menus, no Sputter in the sidebar). They are taken again
  with this version, in English and Chinese, on the public example data, the example frames in `TestSAXSdata/` or
  synthetic data, and three are new: the Sputter workspace (chapter 8) and the Analyze and Help menus (section 3.6).
- Frames stored as big-endian floating point (`>f4`, as h5py and fabio report it for HDF5/NeXus and EDF) were read
  like integer counts: their negative values were dropped as detector gap codes, and Analyze gave them Poisson
  errors and the counting-detector dead-pixel test. They now keep their negative values as data, in Analyze, the
  Format Converter and Batch Export.
- Batch Export: the README of a model-fit batch called the chi2 column of `<batch>_model_fits.csv` a reduced chi2.
  It is χ²/N (the mean squared σ-normalised residual, no degrees-of-freedom correction), as the table's header
  and the manual say; the README now says so and no longer promises error columns.
- A Format Converter container read one numbered frame on its own (`/entry/data/image_000003`) with the
  `pixel_mask` row of the first frame; it now takes the frame's own row.
- **File ▸ Labs Parameters ▸ Load…** also replaces Fitting's model (Single analysis) with the one in the file, which
  the tooltips and section 3.6 of the manual did not say. They now do.
- The README described only the installer of a split package (a package under 1900 MiB is one ZIP), left PyTorch
  out of the first start of a source folder, and said that **Stop** keeps the best values for every Fit method (only
  *Refine* and *Search* do). The agent playbook's MCP example used a developer computer's paths; it now uses
  placeholders.
- The release build packed a folder without `CITATION.cff`, the Physics manuals or new screenshots when git did
  not track them (Help then said "not found" or cited without authors). It now stops, as for program files, and
  also when a conda package of the environment is not from conda-forge (the notices say they all are).
  `SHA256SUMS.txt` of a split package lists the parts and the installers, not the joined ZIP, which is not
  uploaded. The release documents say which licences the licence collector checks (GPL-3.0, LGPL-3.0; other GPL
  and LGPL versions are listed for a check by hand).
- **A tilted detector is no longer passed off as "Status: OK".** GIMaP's geometry is a flat detector perpendicular
  to the beam; the tilt of a `.poni` was folded into the beam centre and the distance and then dropped with only a
  note, so the public Xeuss frame (6.06° tilt), whose |q| is off by up to 6.3 % far from the beam, got "Status:
  OK" and exit code 0. The instrument profile now keeps the tilt, and Analyze compares |q| of the tilted geometry
  with that of the flat geometry made from the same `.poni` on each frame's valid pixels, so the estimate is what
  the tilt alone costs: a beam centre moved afterwards (the GISAXS symmetry axis, a centre set by hand) is no part
  of it (AgBh example: 0.08 %, where the moved centre would read 4.1 %), and the report's lines and `report.json`
  give the same number. Above 1° of tilt or 0.5 % in |q| the automatic analysis (Results
  and `report.md`) lists **Detector tilt** under *Needs attention* with the tilt and the error
  (`tools/gimap_agent.py` exits with 2), and Analyze's status line and Geometry step say it too; below that a
  *Decisions* line gives the error (P08: 0.31°, 0.2 %). `<stem>_analysis.json` records `dropped_tilt_deg`,
  `dropped_rotations_deg` and `flat_detector_q_error_max` / `_rms` in `geometry`, `report.json` in
  `geometry.flat_detector` (Physics and Methods, section 3.7). Its `geometry_convention` now says "kinematic
  (vacuum) grazing-incidence q of a flat detector perpendicular to the beam" instead of "exact".
- `<stem>_analysis.json` did not say when it was written, how σ was computed or how the curves were binned. It now
  has `created` (UTC, ISO 8601), `error_model` (`poisson`, `pixel_scatter` or `scaled_poisson`, as the reduction
  decided) and `binning` (GIWAXS: number, q range and width of the radial bins, the points kept and the merge rule;
  GISAXS: one point per detector column or row); every earlier key is kept.
- A geometry chosen by Run Automatic Analysis or `tools/gimap_agent.py` was saved with the source
  `assistant: <file>`, as if the AI had chosen it; it is now `automatic analysis: <file>` (`assistant:` stays for
  Process with AI, and profiles saved earlier keep their label).
- Run Automatic Analysis, `tools/gimap_agent.py` and the tool `run_standard_pipeline`: an αi typed into Analyze's
  command bar (or set by `set_incidence_angle`) before the run was used by the analysis, yet *Needs attention* said
  "Neither the options, the notes nor an instrument profile give αi, so 0° is used" and the command line exited with
  2. The run now takes that αi (*Decisions*: "from Analyze's command bar") and asks nothing; an αi from the options
  or the notes that differs from it replaces it there, so the αi reported is the αi used.
- GIWAXS automatic analysis: the peak search fits at most 12 candidates, the most prominent, and the report said
  only "12 peak(s)" although more had qualified (on the public P08 frame 115 candidates; fitting all gives 32 peaks,
  among them q = 3.585 Å⁻¹ at SNR 33). *Decisions* now says "N candidate peaks found; the 12 most prominent were
  fitted", the peak search record gives `candidates`, `max_peaks` and `truncated`, and the series hints are titled as
  taken from the fitted peaks only. `tools/gimap_agent.py auto --max-peaks N` and `max_peaks` of `find_peaks` and
  `run_standard_pipeline` fit more (default 12).
- GIWAXS ring orientation: the texture sentence compared |f| with 0.1, although a partly measured random ring does
  not give f = 0. It now compares f with f_iso: "along the surface normal" or "in the sample plane" is named only when
  |f − f_iso| ≥ max(0.1, 3σ_f) (a ring whose I(χ) maximum lies between 20° and 70° is called tilted from that maximum
  alone, without the margin), and a maximum that f contradicts is no longer called an orientation. The sentence "weak
  or no preferred orientation (f = …)" is now "no clear preferred orientation (f − f_iso = …, within ±…)". Every ring
  gives f ± σ_f, f_iso and f − f_iso, and says when leaving out the bins next to the sample horizon (|χ| ≥ 85°)
  changes f − f_iso by more than 0.01 and 3σ_f, and whether the verdict would change; *Decisions* names the texture
  rule and those rings. On the public P08 frame the ring at 1.989 Å⁻¹ is now "oriented along the surface normal"
  (f − f_iso = +0.105) and all three rings are named, with shifts of 17–27 σ_f.
- GIWAXS automatic analysis: a peak within 1.5 widths of the end of the data (flag `at_edge`, shape possibly cut off)
  counted as reliable. It could be one of "the strongest reliable peaks" analysed as a ring, get a Scherrer size and
  complete a series hint, while the Results tab called it "at the end of the data: check" and `report.md` printed the
  raw flag. It is now left out of the rings, the sizes and the series hints (*Decisions*: "not analysed: … (at the
  end of the data)"), the peak table says "at the end of the data: check" (and "overlaps a neighbouring peak",
  "narrower than 3 bins …" for the other flags), the summary marks it `*` and the Results tab does not count it as
  reliable. A ring with fewer than 10 χ bins gave coverage 0 and the unmeasured range 0–90°; they are now those of
  the bins it has.
- GIWAXS and GISAXS reports: numbers were rounded twice or to a precision of their own. A Scherrer size of 29.348 nm
  was 29.35 in the record and "≥ 29.4" in the table, next to "L ≥ 29.3 nm" in the steps: the record now keeps the
  printed values unrounded and each is rounded once. f had fewer decimals than its σ_f ("f = 0.139 ± 0.0027"), the
  note on f_iso read "would give f = -0.00", and a ring line left f_iso out when it was within 0.005 of 0: f, f_iso
  and f − f_iso now have the decimals of σ_f and a Unicode minus, and every ring line gives f_iso. The beam centre was shown as
  "(545.458310126973, 1826.94747674757) px" in the Results tab and "(545.5, 1827)" in `report.md`; both now give it
  to 0.01 px, like the cuts (the `report.json` blocks keep four significant figures). Table columns had varying decimals ("1.9" next to "1.900", "R 8.6 nm" next to
  "R 9.44"); each column now has one. The peak search record said `bins` for the number of I(q) points searched: it
  is now `points`, with the reduction's `radial_bins`.
- Analyze Export: the frame's `<stem>_analysis.json` listed the curves in `files` but not the
  `<stem>_fit_input.dat` written by the same export; it lists every file now.
- The GIWAXS report did not say what its numbers are. *Decisions* now always says that q is kinematic (no refraction
  correction), with the q_z shift of up to about k·sin αi that refraction can cause near or below the film's critical
  angle (0.0119 Å⁻¹ on the public P08 frame), and that the uncertainties are statistical only, from Poisson counting
  statistics or — for floating-point, background-subtracted and mirror-filled frames — the pixel scatter, without
  the geometry. Fit details said the peaks were "weighted by the Poisson error of each bin" also when the errors were
  the pixel scatter; it now names the error model, which `get_status` gives as `error_model`. The GISAXS report
  said nothing of the kind either: its *Decisions* now say the same in their own words — refraction mostly shifts
  q_z, that of the Yoneda band most, by an amount of the order of k·sin αi (0.0379 Å⁻¹ on the public GALAXI frame),
  while q_y and 2π/q* barely change — and that the σ of the cuts, by which χ²/N is normalised, are statistical only,
  with no ± for the quick fit's R, h or D.
- GISAXS automatic analysis: 2π/q* of a peak or shoulder of the horizontal cut was reported as the in-plane
  spacing D, although a form factor or the beam tail makes a shoulder too (on BornAgain simulations of the GALAXI
  model, 73.9 nm for a true 53.6 nm, and a shoulder at 60.6 nm without any interference). Every report, table
  header, *Decisions* line, command-line summary and AI tool description now calls it *a length scale, not
  necessarily the particle spacing; check it with a model fit* (report section **Length scale 2π/q\***, decision
  `length scale`, batch column `2π/q* (nm)`); the numbers and the record keys (`spacing`, `distance_nm`) are
  unchanged. The model question names the solutions whose D is within 15 % of 2π/q*, no longer "the observed
  spacing".
- GISAXS automatic analysis: the solution reported was the first-ranked one even with a value at a search bound
  (on the public GALAXI frame a random cylinder with D at its 500 nm bound). The quick fit now names each value that
  ended on a bound (`at_bounds`), and among the solutions within 10 % of the first's χ²/N the run reports the
  best-ranked one that converged with no non-linear value at a bound (GALAXI: the sphere, rank 3, R = 9.44 nm); when
  there is none it reports the first and *Needs attention* says it sits at a bound. The amplitudes, the background
  and the beam-tail amplitude are solved non-negative and are not counted: one at 0 is a *fit caveat* of the
  *Decisions* (the GALAXI sphere: *background BG at 0, the non-negativity limit*). The fit table of the Results tab marks a
  value at a bound ⚠, names them by rank under the table (*1: D at its upper bound, …*) and selects the reported
  solution; `report.md`
  marks it *(chosen)* and gives the values at a bound in the note column, `fit_solutions.csv` has `at_bounds` and
  `chosen` columns, the fit record `chosen`, and `fit_curve.csv` the reported solution's curve (Physics and Methods,
  section 8.5).
- GISAXS automatic analysis: *Needs attention* gave D = 48.4 nm where the fit table said 48.3 nm, from the tool's
  answer rounded to four significant digits and rounded again. Every number is now formatted once from the fit
  record.
- GISAXS automatic analysis: a reported fit far from its curve passed without a word (the public AgBh SAXS frame,
  Bragg rings of a calibrant: a sphere with χ²/N = 472). A converged reported solution with χ²/N above 10 — the
  limit of Fitting's poor fits — is now a point to check (**Fit**, exit code 2): its R and D are not measurements.

### New known limitations

Chapter 13 of the manual is the complete list for this version. These are the ones added since 0.1.0: the
limitations of the new features, and ones found or written down since. The limitations listed for 0.1.0 still
hold.

- **Sputter does not determine the thickness of a continuous film**, and a film with particles on it is not
  modelled. Read the island-volume equivalent t_eq (island windows, conditional) and the nominal thickness from
  a deposition rate; compare with XRR, a quartz balance or a cross-section.
- **Sputter reads folders of CBF frames only**, and its networks were trained for Ag and Au on polymer
  substrates; **Cu is read with the Ag model**: Cu results are for reference only. For NeXus series or another
  substrate use Analyze's Series map and Fitting's In-situ series.
- **Sputter's first Predict downloads PyTorch** (about 1.2 GB, needs the internet); the model runs on the CPU.
  Without internet, install it by hand (manual, section 14.7).
- **Sputter keeps every result folder**, with the frames as the model read them (`intake/`, `binned/`), in the
  user data folder unless you chose another place; nothing removes them, and **File ▸ Save Project** does not
  hold Sputter results.
- **Sputter's literature method rests on the deposition rate, t0 and deposition end you give** and on
  D = 2π/q*. The percolation interval assumes the rate within ±10% and t0 within ±2 s (±0.5 s when t0 is given),
  engineering choices the page does not change (the toolkit's `physical` command does), takes the deposition end
  as exact, and leaves out the systematic error of 2π/q* (more than 40% for D > 10 nm once coalescence starts).
  PNO, PSBMA and PEO have no published percolation data: the band of all polymers of the metal is used.
- **Sputter flags windows where the beam may be off the sample from the intensity alone** (a dip below half of
  the neighbouring level within one modulation period, or below 20% within 10 s without a modulation, with its
  sides down to 80%). The stage position and the range of its translation are not read: a dip that never falls
  below half of the level is not flagged, however long, nor is a slower drift off the sample; a dip with another
  cause is flagged alike, and the first and last windows are not judged. When the footprint reaches the sample
  edge, edge scattering or streaks can raise the intensity in the analysed region instead of lowering it: such
  windows are not flagged.
- **Sputter's island height H from the q_z fringes is the height of a uniform layer**: for islands of another
  shape it may be off by a constant of about 1 nm, and with one fringe period or less in the range by up to 8%
  (inside its interval). It is not given while no minimum is in the detector's q_z range, while a minimum crosses
  a module gap, or when the order cannot be told.
- **NeXus/HDF5 files not written by the Lambda server are now read as stored**: a geometry or mask made on such a
  file with 0.1.0 no longer fits it. Check the beam centre and masks; calibrate again if needed.
- **An instrument profile is matched by detector name and frame size only**, and a `.poni` or `.json`
  calibration is applied without checking its pixel size or frame size; this adds to 0.1.0's limitation of the
  automatic search, which can find the calibration of another experiment. Check the distance in **Geometry**.
- **Geometry Calibration fits silver behenate, LaB₆, CeO₂ and a LaB₆ + CeO₂ mixture only.** The AI's geometry
  search recognises images of Si, Al₂O₃ and Cr₂O₃ by their names but cannot fit them: calibrate such an image
  with pyFAI and give GIMaP the `.poni`.
- **XRR intensities are raw**: no monitor normalisation, footprint, background or resolution correction, and no
  θ offset. A θ motor dataset in µrad is read as degrees (only `mrad` and `rad` are converted). Analyze's
  bad-pixel search and masks are not applied, and negative values of a signed-integer NeXus/HDF5 frame enter
  the sum.
- **Format Converter does not read EDF** (use Batch Export ▸ Converted detector frames), and its uint16 modes
  mark pixels without data only in HDF5: in a TIFF or CBF such a pixel holds 65535 and reads back as a valid
  value.
- **SEM sizes depend on the segmentation**, and the default StarDist weights are trained on cell nuclei, not
  SEM. The scale is never read from the image.
- **fit1d** offers spheres and vertical cylinders only; with *Try …* compositions the Model's values and ranges
  are not passed.
- **2D Prediction reads CBF frames only and applies no geometry and no correction** to a frame: the module's
  own preprocessing is all that runs. **The Ag and Cu distribution modules declare no axes**: their maps and
  curves are drawn on a placeholder axis (0.05–15 nm), which is not a physical scale.
- **Trainset Build** stores the images of its BornAgain grid cache as float16, which overflows above 65504: with
  the default design, part of the pixels of a training image can come out not finite.
- **1D Predict (Workflow V5) is validated on simulated curves only**, with the composition and resolution given;
  finding the composition, an unknown resolution and measured data are not validated.
- **The quick model fit of a GISAXS cut** (Run Automatic Analysis; *Find the particle shape (no AI)*) **is a
  screening step, not a size measurement.** On 14 BornAgain simulations of Ag-sphere cuts with known parameters it
  ranked a random cylinder first in 12 and a sphere in 1, and the best sphere's R was a median 11 % above the
  R⁶-weighted mean radius (−29 % to +26 %); D and its disorder were not recovered, and 2π/q* was within 10 % of the
  true distance in 1 of 10 cuts with interference. Choose the model from what you know of the sample and confirm it
  in Fitting.
- **No record holds the random seeds, and pixels left out are counted, not located.** The records of Sputter,
  the AI assistant, `tools/gimap_agent.py` and 1D Predict's run folders do not name the GIMaP version; note it
  (**Help ▸ About GIMaP**) with results from those.
- **API keys saved in GIMaP are kept in plain text** in the user data folder (`claude_api_key.txt`,
  `assistant_provider_keys.json`). On a shared computer, set the key as an environment variable instead, or
  **Remove** it after use.
- **GIMaP's detector is flat and perpendicular to the beam**: a `.poni` tilt (and `Rot3`, and its `orientation`
  entry) is left out, and Geometry Calibration fits no tilt. On a tilted detector |q| is off away from the beam (up
  to 6.3 % on the Xeuss example); the error is now computed and reported, not corrected. For exact q reduce such
  frames with the full `.poni` geometry (pyFAI).
- **Run Automatic Analysis (GIWAXS) fits at most 12 candidate peaks**; the window has no setting for more. Use
  `tools/gimap_agent.py auto --max-peaks N`, or `find_peaks` with `max_peaks`, for the full list.
- **The GIWAXS and GISAXS q are kinematic** (no refraction correction): near or below the film's critical angle a
  peak's q_z can be off by up to about k·sin αi (in GISAXS the Yoneda band's most, by an amount of that order), which
  the report states but does not correct, and no ± or σ of the report contains it or the uncertainty of the geometry.
- **Hermans f and the texture sentence are not corrected for the exit angle**: f weighs the bins next to the sample
  horizon most, where absorption or a shadow can lower the intensity; the report says when f depends on them.

### Updating from 0.1.0

- **What is kept.** Settings, instrument profiles, recent files, the model of Fitting and saved API keys are in
  the user data folder (`%APPDATA%\GIMaP` on Windows, or the folder `GIMAP_HOME` names), as in 0.1.0, so the new
  version finds them. Projects (`.gimap`) of 0.1.0 open.
- **Portable version:** extract the new package into a new folder (manual, section 2.4) and delete the old one
  when you no longer need it; `Update_GIMaP.bat` cannot update a portable folder.
- **Source checkout:** double-click `Update_GIMaP.bat`, or run `powershell -ExecutionPolicy Bypass -File
  tools\update_source.ps1`. `requirements.txt` gained `hdf5plugin` and `threadpoolctl`, and
  `requirements-sputter.txt` (PyTorch) is new. With `Start_GIMaP.bat` there is nothing to do; in an environment
  you set up yourself, run `python -m pip install -r requirements.txt` (Sputter installs PyTorch at its first
  Predict, or run the same line with `requirements-sputter.txt`).
- **A model folder imported in 2D Prediction** with 0.1.0 was written into the module's `module.yaml`, which no
  longer names one. Choose it once more with **Import model...**; from then on it is kept in the user data
  folder. In a portable version the new folder has its own `module.yaml`. In a source checkout the update script
  reports such a `module.yaml` as an edit of yours: when it does not fit the new version, it waits in the stash
  and the script ends with exit code 2. When it was your only edit, delete that copy with `git stash drop` after
  importing the folder again.
- **The sidebar has a page more**: Sputter is `Ctrl+5`; 2D Prediction moved to `Ctrl+6` and Trainset Build to
  `Ctrl+7`.
- **Results that can differ from those of 0.1.0** (each is described above):
  - 1D Predict on Analyze's `_fit_input.dat`: 0.1.0 read q ten times too small. Run such fits again.
  - GIWAXS curves with **Fill gaps from the mirror side**: the error bars come from the scatter of the pixels
    (those of 0.1.0 were too small), and so do the errors of fits made on them.
  - NeXus/HDF5 frames not written by the Lambda server are no longer transposed and flipped.
  - A series is cut where the frame on screen is, unless *Find the cut again in every frame* is on.
  - The automatic analysis of a series sums the last fifth of its frames (at most 10) instead of the last 10.
  - Stages and odd frames: a frame that differs in only a few points is odd, not a stage.
  - Frames converted by the Format Converter or Batch Export keep their stored type and gap codes.
- **Scripts that read GIMaP's exports**: text files are UTF-8 without a byte-order mark and use ASCII symbols
  (*Exported text*); Send Series to Fitting writes into `<name>_fitting/`; the two Cu prediction modules are
  named `CuPolymer_FF_256` and `CuPolymer_SF_4Para` in the records of 2D Prediction.

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
  `AI_Fitting_Output/v5_<date>_<time>_<µs>/`. The model and its trained weights come with the portable
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
| AI-assisted fitting, results in `AI_Fitting_Output/current_prediction` | 1D Predict, results in `AI_Fitting_Output/v5_<date>_<time>_<µs>/` |
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
