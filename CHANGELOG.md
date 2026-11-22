# Changelog

What changes for someone who uses GIMaP, version by version. Developer notes are in `docs/`.

## 0.2.0 — unreleased (pre-release)

- **GIWAXS film absorption and crystal orientation** use refracted incident and exit waves near the sample horizon, including evanescent attenuation below the critical angle. For a frame correction, enter the optical constants at the measurement wavelength or select a film CIF. The attenuation-length option retains the large-angle geometrical correction.

- **Measure from frame** now measures incidence αi alongside roll ψ. Analyze records the shadow, specular and available GISAXS Yoneda estimates with their uncertainties. Automatic analysis uses a resolved measurement when the nominal angle is missing or differs by more than 3σ. **Keep entered αi** retains the nominal value.

- Analyze can measure sample roll ψ from the shadow or Yoneda band and the two sides of a GIWAXS ring or a GISAXS specular rod. The applied angle sets the sample axes for cuts, pole figures and reflection overlays. Automatic analysis reports the estimates and applies a resolved roll at three standard errors.

What changed since 0.1.0 (2026-10-13). New are the **Sputter** workspace, **SEM Particle Analysis**, the
**Edit**, **Analyze** and **Fitting** menus with a fuller **Help** menu, NeXus/HDF5 files read by what they
hold, a source folder that starts with one double-click, and a reference of the methods, *Physics and
Methods*. Some results differ from those of 0.1.0, and a model folder imported in 2D Prediction has to be
chosen once more: read *Updating from 0.1.0* at the end of this version before you update.

### Release preparation — Windows and macOS

- Analyze saves indexed q maps with their visible phase and region overlays, legend and publication settings.
  The GIWAXS report opens with phase matches, ring orientations and warnings. Replacing a sample offers to
  reset its αi and visibility settings. In situ fits can export radius and spacing together against elapsed
  time and point to the first fit needing review. Both GISAXS cuts open together for a joint fit. A saved
  fit can be reopened, compared after a change and reproduced using a relocated copy of its verified input.
- **First-use workflows.** Geometry Calibration accepts wavelength, identifies pixel pitch inferred from frame size,
  shows ring-fit warnings beside the job status and saves a PONI directly. Fitting explains which halves Predict
  uses, marks bound-limited values and saves the curve and parameters together from Results. The GIWAXS examples
  include candidate CIFs. Analyze can load several at once, identifies the strongest reliable ring by fitted area
  and exports the measured peak table with its indexing and geometry.
- The manuals document the precision of an applied beam centre in Analyze, Geometry Calibration and XRR,
  and the differences observed when reproducing archived Predict parameters fits on another platform.
- **Predict parameters** flags upright cylinders that could represent spheres above q_z·R = 2 at levels 2 and 3,
  and uncertain radius intervals above 1.7 at level 3. Fit details compare the point spacing at the intensity
  maximum with the linear validation grid and flag cuts with fewer than 80 or more than 400 points.
  Joint vertical fits compare heights from two exit-angle ranges and report the dilute-layer and independent-height assumptions.
- **GIWAXS orientation** reports measured Hermans f beside its same-coverage isotropic reference f_iso. The systematic range includes empty and continued edge gaps and tests attenuation near the horizon. Reports name the scenarios at both limits and flag rings whose range is wider than 0.05. The Pole Figure tool, CSV headers and JSON records include the reference and limits.

- **Fit Several Curves Together** starts vertical cuts at twice the substrate critical angle. Set the limit to zero
  to use the full cut. The report flags likely height underestimation in dense or high-Z layers when the vertical
  reduced χ² exceeds 3. The cylinder DA now averages coherent amplitudes over the height distribution. Heights and
  radii are assumed independent.

- **Automatic GISAXS analysis** fits the Born term at the Yoneda cut's vacuum q_z when the incidence and exit
  angles are known (level 2), and uses the effective interface at q_z = 0 otherwise (level 1). The report and
  JSON record identify the level used. At level 2, Predict flags sphere and random-cylinder radii for uncertain
  interval coverage above q_z·R = 1.2 and strong model error above 2.

- **Reproduce from Record…** in Analyze ▸ Tools and `gimap reproduce RECORD.json` rerun recorded reductions and fits.
  Records preserve mask bitmaps, input digests, geometry, fit settings and full-precision comparison values.
  The comparison reports bitwise agreement and the maximum relative difference per column (User Manual §12).
  Maps built while watching a folder can also be replayed after reopening a project that kept their row records.

- **Geometry Calibration** searches along predicted rings for weak reflections and ranks geometries by the rings present, strong peaks left unexplained and the fitted point residual. It measures each ring position from detector counts on both sides of the peak. The engine found all 300 simulated AgBh geometries (previously 213) and matched exactly three orders on all 150 three-order frames. Three-ring uncertainty accounts for centre–tilt coupling when all three pairs constrain the fit. For four to six matched rings, distance and rot2 each fell within 1σ in 56.5 % of frames. See *Physics and Methods* 3.3 for coverage of all five parameters.
- **Geometry Calibration** refines arc-derived beam centres before ring detection on transmission frames as well as grazing-incidence frames.
- **Geometry Calibration and GIWAXS orientation.** Calibration now refines the beam centre and distance with tilt fixed and reports ring-point RMS before and after refinement. Orientation reports identify weak bins excluded next to the exit-angle cut, vary that threshold in the Hermans f range, and omit model fills from undetermined pole fits. Sparse bins with measured scatter retain an uncertainty when the pooled variance fit crosses zero.
- **Analyze ▸ Mask & corrections.** Raw background and flat-field frames can use an exposure-scaled dark.
  Set the flat-field threshold, save detector gain, offset and error model in an instrument profile, and choose
  pooled scatter errors or smoothed counting errors for sparse curves. Propagated errors include flat-field
  noise and treat the measured flat mean as fixed. GIWAXS exports identify the error model for the measured
  orientation curve when it differs from the model for the radial curve.

- The README and User Manual give the command that reproduces the paper's GALAXI Yoneda cut.
  Desktop CI checks Python 3.10 and 3.11 and reads NXcanSAS exports with SasView's loader.
  Two Predict fits are checked against the archived numerical results. The AI assistant asks before
  sending data to a service, records the provider and model, and uses the OS credential store when available.
- **GIWAXS ring orientation** uses polarisation-corrected pixel means and an exit-angle cut. Reports give measured Hermans f with a statistical error and systematic range, fitted f and the background-dependent oriented fraction. Poor pole fits are flagged. Tilted-axis models average over azimuth. Lorentz weighting is available for peak areas, with an exclusion band around the plane of incidence. Phase assignments identify the q convention, Δq, Δd/d, ambiguous matches and unassigned peaks.

- **Fitting ▸ In-situ series.** Inspect each Predict choice, its rank and its probability in the results table,
  CSV, JSON and series report. The closest-solution rule now includes the relative radius width σ_R/R.
  An optional hidden Markov model selects a path through the series and gives each solution a smoothed probability.
  The trend shows the chosen mode's posterior median and 68% interval, with an optional least-squares marker.
- **Analyze ▸ Absolute Scale from a Standard…** fits a detector frame or reduced radial curve against water at the
  measured temperature or a user-supplied reference such as glassy carbon. Calibrated exports carry intensity in
  cm⁻¹ and record the standard, fitted range, scale and uncertainty. The instrument profile keeps the calibration.
- **Analyze ▸ Mask & corrections.** Draw circles, annuli, ellipses and freehand strokes, erase pixels, or select a
  ring about the direct beam in Å⁻¹. Preview intensity thresholds, saturation, negative pixels, mask growth,
  shrinkage and inversion. Each mask edit supports Undo and Redo. Save an ordered JSON recipe or the current
  frame's user mask as EDF, TIFF, NumPy or Fit2D with its source record. See User Manual 5.2 and Physics and
  Methods 4.5.
- **Fitting, Refine and Search.** Choose a number-weighted Gaussian, log-normal or Schulz distribution for each
  population, use per-point Gaussian resolution from text curves or NXcanSAS, and select file, relative, Poisson
  or logarithmic residuals. The fit record and report identify the settings used.
- **Detector geometry in grazing incidence.** Analyze reads and writes pyFAI detector orientations 1–4,
  displays the tilted horizon, and projects GIWAXS mirror partners onto the detector. XRR projects the
  specular spot using the detector rotations. GISAXS users can check rot3 on a measured ridge or band.
  The automatic vertical cut clears the specular rod by 15 columns where possible. Joint fitting warns
  about a vertical band on the rod, and automatic horizontal cuts warn when the horizon crosses more than half a row.

- **In-situ series after a user test.** The chained fit no longer breaks where a value reaches a bound of its range (the beam tail, the background or any other), at an undetermined value or at a poor fit — those are flagged; it breaks only at a failed or unconverged fit, a new stage or a jump of χ²ᵣ, and there Predict parameters (unchanged) restarts from its solution closest to the frame before among those at least a tenth as probable as the most probable one. A smooth series no longer gets steps from restarts (a linked two-sphere model jumped by a factor 2 in R₁ where only the beam tail's exponent had reached its bound). The table's new **Started from** column says “continued from frame n” or “restarted from Predict (why)”, with the rule that chose Predict's solution on hover; the saved table and record add `start_frame` and `predict_choice`. **Refine** and **Search** go on from their own result while χ²ᵣ falls by more than 0.1 % (at most 5 more runs, counted in the Run Log, the record and the report). **Fit New Frames in Fitting…** ticks **Watch for new curves** and starts once the model is set up; a map the watch begins includes the frame on screen when watching began; the Series map comes back with a project and after a crash; NXcanSAS files of fits mark their entries as data or model, and opened again offer the measured curves. Manual 3.9, 5.1, 6.1, 6.2, 6.5; *Physics and Methods* 9.8, 11.13, 14.2.
- **Third user test: Analyze and GIWAXS.** Opening frames while others are listed no longer ends the program on a
  computer without a display (the Replace / Add to List question, and the other questions of Analyze, Batch
  Export, Sputter, Format Converter and the session, are now GIMaP's own question window). A CIF that names its
  space group without listing the operations gets them from the symbol (all 230 groups and the common settings);
  without that the phase gives positions only, says so in the Symmetry column and the report, and its assignments
  are tentative — it no longer lists forbidden reflections with wrong intensities. A phase is named after its common
  or mineral name or its data block, not its sum formula. The Results peak table has the phase, hkl and Δq of each
  peak. A pole figure whose fit ends at a bound leads with *not determined* and the Hermans f of the measured bins.
  The Normalisation, Corrections and intensity-correction sections head the Mask & corrections step, and the step
  says when the intensities are scaled. Geometry Calibration shows one RMS residual and names it (ring points with the
  tilt, ring radii without). Manual 5.2, 5.9, 4.3; *Physics and Methods* 3.4, 7.10, 7.11.
- **Predict parameters** reports least-squares values, sampled medians, interval sources, BIC weights and Monte
  Carlo errors. It flags high-q_z·R limits and departures from the validation point density. The MCMC check
  compares independently initialized ensembles and reports autocorrelation time and split R̂.

- **Calibration in the automatic analysis:** when the line positions choose another solution than the ring fit, the
  record gives the 1σ of that solution (before, the 1σ of the ring fit's first choice stood beside the chosen
  geometry), and the first choice keeps its own geometry and 1σ among the alternatives (before, its entry held the
  chosen geometry).
- **GIWAXS indexing and pole figures (Analyze ▸ Phases).** Give phases as a CIF (cell, symmetry operations, atom
  sites; |F|² from Cromer–Mann scattering factors) or a cell with its centring, at random or as a fibre texture
  ((hkl) parallel to the substrate or [uvw] along the normal): their reflections are drawn with hkl labels on the
  q map, the detector, the cake and I(q), several phases in their colours, optionally shifted for refraction in the
  film with the transmitted and reflected DWBA channels; a table lists hkl, d, q∥, qz, χ, m and |F|². Run Automatic
  Analysis assigns each ring to the nearest reflection within max(3σ, FWHM/2) (a *phase assignment* decision, the
  peak rows, a report section; `--phase FILE.cif` on the command line) and fits the pole figure of each analysed
  ring — Gaussian or Maier–Saupe about a direction plus an isotropic fraction, the missing wedge
  filled by the model — giving the oriented fraction, the FWHM and Hermans f with their errors next to the Hermans
  factor of the measured bins. The window exports the pole figure as CSV with a JSON record. Manual 5.9;
  *Physics and Methods* 7.10, 7.11.
- **In-situ series: online watching, a timeline, chained fits and a results database.** The arrow of **Watch** in
  Analyze pauses and resumes the watch, sets how long a new file must rest before it is read, adds every new frame to
  the Series map (also without Build Map or an export; only curves are kept), records every frame in
  `gimap_analysis/gimap_series.sqlite` and keeps a log of the files it skipped; **Fit New Frames in Fitting…** hands
  the exports to In-situ series. The plot under the Series map has a timeline — the χ² of every frame to the one
  before and to a reference frame, with stage boundaries, jumps and odd frames — and a click on it shows that frame.
  Fitting ▸ In-situ series can start every frame from the previous result and ask Predict parameters only where that
  chain breaks (a failed or unconverged fit, a jump of χ²ᵣ, a new stage; Predict's method is unchanged), add a backward
  pass, show a Flags column, draw the trend against time, and save the fits as NXcanSAS or into the results database.
  Compare ▸ Add Series ▸ **Results Database…** reads a recorded series without its frames. Manual sections 5.1, 5.6,
  6.5, 7.2, 12 and 13; *Physics and Methods* 11.12–11.14.
- **Fitting: several curves fitted together, the four-term DWBA with a known substrate, and an MCMC check.** **Fitting ▸ Fit Several Curves Together…** fits a list of curves with one χ², values shared or each curve's own, every curve with its own scale and background; **Analyze ▸ Send Both Cuts to Fitting** writes a frame's horizontal cut and its vertical cut (`<stem>_fit_input_qz.dat`, with its geometry) and lists both there. The vertical cut is computed in the four-term DWBA of spheres or upright cylinders on the substrate, point by point along αf, and holds the cylinders' height that a horizontal cut cannot see. Model's card *Spheres in grazing incidence* has a **Substrate** (Si, SiO₂, plate glass, Au from Henke's tables at 4.9–30 keV, or δ and β typed): with αi and αf the spheres of a manual fit are then in the four-term DWBA, and the line below names the level in use. Results ▸ Uncertainty and Report… has a **Posterior check (MCMC)** tab: a short ensemble MCMC run on the posterior of the solution in Model (Predict's own, or the last Refine or Search) beside the intervals it checks, with τ, R̂ and a verdict; a check only, it changes no value. Predict parameters is unchanged (its results on the benchmark cases are identical). Checked against BornAgain 24.1 to 4 · 10⁻⁴ (horizontal cut) and 2 · 10⁻⁴ (vertical cut) with size distributions. Manual 6.2, 6.6, chapter 13; *Physics and Methods* 9.17–9.19.
- **Sputter ▸ Analyze: Au and Ag on TiO₂ at any energy from 8 to 15 keV.** The TiO₂ network was retrained on images
  simulated at 8–15 keV and αi 0.25–0.55°, with optical constants that follow the energy (the earlier ones exactly at
  12 keV) and with parts of the detector cut away, so a TiO₂ scan in that range is analysed as usual at its own energy,
  Au included; before, only Au at 12 keV and Ag within 5 % of it were. It passed the tests set before it was trained,
  on the reference scans at 12 keV and on simulated scans at other energies, angles and detector coverage. The earlier
  network stays the alternative at 12 keV, and the better of the two is shown. A TiO₂ scan at an αi outside
  0.25–0.55° gets a warning in the log. The networks for Si, polymers and Cu are unchanged: their retrained versions
  did not pass those tests. The package's report under **Details** now lists the model values for Si, polymer and Cu
  scans as well (before, only the direct readings). Manual chapter 13; *Physics and Methods* 12.1.
- **GISAXS cuts of your own, in q: bands, boxes, sectors and lines in any direction.** The Cuts step of a GISAXS
  frame has a **Cuts** table: the automatic Yoneda cut (the one Fitting takes) and vertical cut first, then as many
  named cuts as you add — horizontal bands (I(qy)), vertical bands (I(qz)), boxes (along qy or qz), sectors (I(qyz)
  in a direction from the direct beam) and lines in any direction with a width (I(s) along them). Add them from
  **Add ▾** or draw them on the detector image or the q map; type their position and width in q (Å⁻¹), exit angles
  (αf, 2θf) or detector pixels; rename, duplicate, remove, show or hide them. Each cut has one colour in the table,
  the plots, the detector view and the q map (where the automatic bands are now drawn too, and bands and boxes can be
  dragged). A cut averages the whole pixels whose centre q lies inside it, with the frame's error model; sectors and
  lines are binned one pixel wide. Cuts are kept in q (also in settings files, Batch Export and series), and a frame
  of another detector or geometry asks whether to **Keep Cuts** or **Remove Cuts** instead of using them silently.
  The CSV and `_analysis.json` of every cut state it in q (and αf or 2θf); those of the automatic cuts add the qz/αf
  and qy/2θf of their band edges. The automatic cuts and their numbers are unchanged; **Auto Cuts** now keeps the
  cuts you added (*Physics and Methods* 6.7).

- **Fitting keeps what it finds.** *Open Curve…* reads `.csv` curves — the command line's cuts, Analyze's curve
  tables and Fitting's own *Save ▸ Data and Fit* table — by the names of the header (q, I, σ; the q in nm⁻¹ of
  Fitting's table, not the second q in Å⁻¹), and asks for the columns (*Choose Columns*) when the header does not
  say. *Save ▸ Solutions…* (*Export Solutions…* in Results) writes every solution of Predict parameters — composition,
  probability, χ²ᵣ, flags and every value with its median and 68 %/95 % intervals — as CSV and JSON, with their
  curves and terms; *Save ▸ Data and Fit* adds them to its record, and the ± column shows half the 68 % interval of a
  Predict solution in Model. A project keeps the last fit, the solutions and the series' fitted frames (stale when
  the curve file changed). Predict shows its step (proposals, refinement, sampling of each solution), the time and
  an estimate; **Stop** keeps the solutions already sampled, marked incomplete (the numbers of a run are unchanged).
  After *Refine These Values* the results and the selector say *Refined from solution 1 (Refine)*; solutions of one
  composition are numbered (*#1*, *#2*) and show their flags. *Save Plot…* writes the plot with its residuals and
  error bars as PNG, SVG or PDF; the status line is one shortened line; Undo and Redo say what they took back.
- **Normalisation, a dark frame and the Lorentz factor in Analyze.** Mask & corrections ▸ **Normalisation** (every
  term off by default) multiplies each frame, after every other correction, by A / (t · M · T · d): the exposure time
  from the file header or typed, a monitor of the header (EDF keywords, the `Flux` of a Pilatus CBF, a NeXus
  `NXmonitor`, now read per frame) or typed, the transmission, the thickness (mm) and an absolute-scale factor; a sum
  of frames is divided by the sums of their exposure times and monitors. The section always lists the header values
  the frame offers and says by what the frame was multiplied. σ is multiplied by the same factor (photon counts stay
  Poisson, scaled); the unit of the intensity goes into the plots, the CSV column, an `# intensity:` line of every
  curve file, the fit input and the record (`normalisation`: the factor and each term with its source). A **dark
  frame** with its own scale is subtracted before the background (σ then from the pixel scatter). **Lorentz factor
  for peak areas** among the GIWAXS intensity corrections divides by GIXSGUI's L = 1/(cos αi cos αf |sin 2θf|);
  the radial curve then sums the weighted pixels. Ring orientation uses the mean intensities before this weighting. *Physics and Methods* 4.11, 4.12, 5.6 and 7.7.
- **The Analyze display.** In log display, pixels at or below 0 (a dark-subtracted frame) take the lowest colour
  instead of none, and the automatic limits come from the positive values: such frames no longer look black. In
  linear display the colour bar is drawn around the limits. The image is the widest panel, and **Panels ▾** hides
  the steps or the curves and results. Find Calibration Automatically no longer switches the right panel to Results,
  and the Results geometry says the αi in use (and the run's, when it differs). **Export Curves…** asks for the folder
  (starting in the last one chosen) and names the curves of a cut region by the region's name.
- **Fitting: links and constraints, the uncertainty of a fit, a report, a tail that stays within the curve, and more
  models for manual fits.** In the Model step every value is one line (name, value, ±, min, max, fit and **=**), so two
  populations fit on one screen. **=** links a value to another (one σR/R or one D for two populations) or to an
  expression of others (`2*R_1`; parsed, never `eval`-ed); the card **Links and constraints** lists them, switches
  **Particles do not overlap** (D ≥ 2R) and takes inequality constraints (`D_2 >= 2*R_2 + 3`); linked values follow
  the others in every fit and get their errors by propagation; all of it is saved with the model and in the record.
  **Results ▸ Uncertainty and Report…** shows, after Refine or Search, the correlation matrix as a heat map with its
  values and the covariance errors, names the pairs with |r| > 0.9, draws the χ² profile of one value with its
  Δχ² = 1 interval, and writes a fit report (Markdown, HTML or text); the record of Save Data and Fit… gains the
  correlation matrix and covariance. The beam tail's ranges now follow the curve in Refine and Search (A ≤ 10× the
  largest intensity, w within 0.3 q_min–min(30 q_min, q_max), ν ≤ 6) — with the old ranges A ran to 10¹⁷ and more —
  and a tail at an edge or without a finite error is flagged; a series counts such frames as flagged. New for Refine
  and Search (Predict parameters is unchanged): the **Core–shell sphere** and, as **Interference**, the **2D hexagonal
  paracrystal** and **Hard spheres (Percus–Yevick)**, each checked against SasView's formulas or BornAgain
  (*Physics and Methods* 9.6, 9.14–9.16).

- **Geometry: the detector tilt is calibrated, no energy is assumed, `.poni` in and out, transmission frames.**
  Geometry Calibration fits pyFAI's rot1 and rot2 with the beam centre and the distance on points of the standard's
  rings (**Detector tilt: Refine rot1, rot2**, on with three or more rings) and saves their 1σ estimates in the
  instrument profile. On synthetic rings the tilt is found within 0.002°,
  and an untilted geometry keeps every q to 10⁻¹² (*Physics and Methods* 3.8). The calibration energy is no longer
  12 keV when the file states none: the field reads *Required*, Auto Calibration waits, and the record and profile say
  where the energy came from. Analyze's **Detector geometry** dialog has **Energy** beside the wavelength, a line saying
  where it came from (or that none is known: *Required*), and rot1, rot2, rot3. **Load .poni…** and **Export .poni…**
  (Geometry step, dialog, banner; Export Calibration… as `.poni`) read and write pyFAI PONI 2.1. A `.poni` beside the
  frames is used and named; a profile picked for frames of another size is no longer applied to another detector's
  frames (**Use “…” Anyway** in the banner); opening frames while others are listed asks **Replace** or **Add to List**,
  and frames of another detector are not added. Batch Export says where each folder's geometry comes from and stays
  off when no frame can have one. The new mode **SAXS** (SAXS/WAXS in transmission; Auto takes it for a calibrant without αi)
  gives I(q) and I(χ) round the whole ring with no horizon or Yoneda cut. The cursor readout shows q, q_y or q_∥, q_z,
  2θ, χ, αf, d and the pixel value on two lines.
- **The application around the pages: start-up, unsaved work, crashes, theme, log, command line.** A splash
  screen with the version and what is loading appears within a second (the window used to appear after 13–26 s
  with nothing on screen); Compare and Sputter are built when first opened. One GIMaP per user data folder
  (`gimap.lock`): a second start offers **Switch to It** (its files open in the running window) or **Start Anyway
  (Read-only Settings)**. Unsaved work puts `*` in the title; closing or opening another project asks **Save /
  Don't Save / Cancel**. Changed work is autosaved every 2 minutes and after each long job to `autosave/` in the
  user data folder, and after a crash the next start offers it back. A theme or font-size switch restyles the
  windows, not every widget once per ancestor: from 20 s with the window frozen to 2–3 s;
  the font-size field applies after the last click. `logs/gimap.log` (rotating; also under `pythonw.exe`) and
  `logs/crash.log` (native crashes) join `errors.log`: **Help ▸ Open Log Folder**. `python main.py FILE…`, the
  `gimap` command of `pip install -e .` and `--version`. The settings of older versions are imported once, into
  the default folder only and never a session (a `GIMAP_HOME` folder imports nothing). Also: the “Use the set-up
  of your last session?” note is not offered after a project was opened; an old note never covers a newer status
  line; a file or project that could not be opened leaves Recent; splitter positions are remembered; a window
  smaller than a page scrolls it instead of cutting it off; `F1` opens the manual at the chapter of the page shown.
- **A sphere too large for the form factor at q_z = 0 is flagged.** Predict parameters fits a cut at q_z = 0 (the
  effective interface) unless it is given the cut's angles; for spheres that holds within about 5 % of R only up to
  q_z·R ≈ 1.7 (on the four-term DWBA benchmark the median error of R is 15–29 % above it, with both angles below
  10 %). A solution fitted at q_z = 0 whose sphere or random cylinder has q_z·R above 1.7 at the cut's
  q_z = k(sin αi + sin αf) now has the flag *qzR* (*R: q_z·R is above 1.7 at the cut's q_z, … biases the radius,
  usually too small; give both angles of the cut …*), in English and Chinese. This flag applies to first-level fits
  with a measured q_z supplied for flags (*Physics and Methods* 9.13).
- **Curves as NXcanSAS (HDF5), for SasView and other NeXus programs, and back into Fitting.** Analyze's Export
  step has **Also as NXcanSAS (.h5)** (remembered; it applies to Export Curves…, Save the Curves of the Plot and
  Export New Frames While Watching): `<stem>_nxcansas.h5` holds the curves against q, one SASentry each (NeXus
  definition NXcanSAS 1.1), titled with the frame, the curve and which q its Q is — a horizontal cut keeps its
  signed qy —, with Q in 1/angstrom, I, Idev = σ, the wavelength, distance, pixel size and αi, and the record of
  `_analysis.json` as a note of SASprocess; I(χ) and 2θ curves stay CSV only. Batch Export has **Each frame: its q
  curves as NXcanSAS (.h5)** (`nxcansas/<frame>_nxcansas.h5`), the command line `--curve-format csv|nxcansas|both`
  (default csv; `curves_nxcansas.h5`), and Fitting's **Save Data and Fit…** the type NXcanSAS (the points fitted and
  the model, Q in 1/nm). Fitting opens NXcanSAS `.h5` files of any program: Q in the unit the file states (1/m is
  converted), Idev as σ, the points its mask leaves out left out; a file of several curves asks which one, and
  **Import Curves…** takes all of them. SasView's reader (sasdata 0.11) opens the files GIMaP writes, with their
  titles, units, σ, wavelength, distance and αi.
- **The benchmark files of Predict parameters are the ones the paper names.** `tools/benchmarks/` now holds the
  final runs under the names of the paper's table (`posterior_recovery`, `posterior_extended`, `posterior_dwba_tp`
  / `_born` / `_eia` with `posterior_dwba_pairs.json`, `posterior_own`, `posterior_floor03`, `posterior_floor08`,
  `posterior_amplitude100`, `posterior_timing`): the summaries beside the scripts and every cut of every run in
  `tools/benchmarks/per_cut/*.json.gz`. All were made with one version of the fitting code (git tree 7e68831a,
  commits 2eb0243 and 7179b85). Results of earlier code and of the removed fit1d and V5 methods moved to
  `tools/benchmarks/history/`. *Physics and Methods* 9.13 and 10.3 (English and Chinese), the network's
  `MODEL_CARD.md` and the README of the folder quote these runs, their tables printed from the files by the new
  `tools/benchmarks/summary_tables.py`: on the 200 held-out curves the P90 of R is 10 % and 14 % (the documents
  said 17 % and 23 %, from an earlier run), the true composition is the most probable solution's in 88 % and 65 %,
  and the extended set has 600 curves. "True composition most probable" is judged on the most probable solution,
  and the documents now say so. Predict parameters itself is unchanged.
- **Fixes from a user test (fitting, command line, records).** *Fitting:* a Refine or Search that could not
  beat its start now says *did not improve on the start values* instead of *converged* and gives no errors; points
  whose σ is zero, negative or missing are left out with their number (before, one such point silently dropped
  the whole σ column while the card still said *σ from the file*); a curve with fewer points than free values
  plus one, or without an intensity above zero, is refused with the reason; a file that states no unit of q
  always says *Unit of q assumed: … (not stated in the file)*, in the card, the status line and Results; rows
  with NaN or inf are counted and said like rows that are no numbers. *Command line (`tools/gimap_agent.py
  auto`):* `--incidence-deg` outside (0°, 90°) and a `--calibration` file that does not exist end with exit code
  1 before anything is read; a calibration the search took from another folder than the frame's is an open item
  with its path (exit code 2, before: Status OK); a `.poni` that names a detector of another shape than the frame
  (a Pilatus 300K file on a Pilatus 1M frame) is not used — *the calibration does not match this frame*; a
  GISAXS symmetry axis more than 100 px from the calibrated centre is not applied, and one that cannot be found
  is an open item. *Records:* the headers of the run's CSVs are ASCII with one spelling of the units
  (`qy (A^-1), I, sigma`; before `qy (Å⁻¹)`, which numpy on a Chinese Windows could not read) — scripts that
  matched the old header need a look; `report.md` states pixel size and energy, defines the *asymmetry* of the
  symmetry axis and names the disorder σnn/D as Fitting does (before σD/D = σnn/D / √(2π), another number);
  `geometry.critical_angle_deg` is `null` when none was given; the manual says that `geometry` is what q was
  computed with and `calibration` the file as read. `--language zh` translates the section headings only: its
  help and the manual now say so.
- **Predict parameters: one χ²ᵣ per solution, every population shown, plainer notes.** A solution now has the same
  χ²ᵣ wherever it appears — the Fit page's status line, table and Fit details, the automatic analysis's table,
  report, CSV and JSON, Batch Export's table — computed over the points Predict fitted, and each place says how
  many those are (*596 of 598 points; 2 below the network's q window*); a Refine afterwards says when its χ²ᵣ is
  of other points. The reports' former χ²/N (a mean over all points) is gone: `chi2` in the saved tables and
  records is this χ²ᵣ, and its definition is in the tooltip of the symbol. With two or three populations the
  headline, the tables and the model question give every R and D in the order of the radii
  (`R = 3.06 / 6.02 nm, D = 49.4 / – nm`). Fit details and the report give the value that is in Model with its 68 %
  interval (`R 4.56 [4.56, 7.52] nm`, enough digits to tell them apart; never a median) and say when the value lies
  at the edge of its interval; probabilities are stated as relative to the compositions tried, and one composition
  tried shows none. Every quantity has one name, that of the Model table (Tail A, Tail w, Tail ν, σ_q, Scale,
  σR/R (1)), in flags, tables and reports; a term that is absent (Tail A = 0) is no longer flagged, the notes about
  background, tail and resolution follow the others in one paragraph, a free value without an error shows “–” with
  the reason, and “1 solutions” is “1 solution”. Automatic-analysis records gain `points`, `flags` and
  `more_warnings` per solution; scripts that read the old `warnings` texts or `ν_Res` names need a look.
- **Fitting reads curves the same way everywhere.** Files whose header states no unit of q are read in Å⁻¹ on the
  Fit page and in the 1D Predict window alike, and a warning next to the result names the assumed unit and how to
  change it when the q range is implausible. A curve without σ gets the same estimated relative σ in both places,
  files with a title line open in both, and negative or zero intensities are counted and reported.
- **Automatic analysis and records.** An incidence angle remembered from the last session is used only with the
  instrument profile it was set with and is never saved into a profile; for another detector GIMaP asks again and
  offers the old value. Saved reports and the CSV files exported in the window name the GIMaP version, the input
  file's SHA-256, the calibration file and instrument profile, mask and corrections; the peak table gives ± and
  marks poor fits; sizes in `report.json` carry their unit, and `.edf` standard images can be calibrated from the
  command line. The q map is no longer finer than the detector (no holes on tilted or small frames) and opens on
  the measured area; a file-header distance more than 5 % from the calibration is a point to check; a calibrant
  frame in transmission is recognised and no particles are fitted to it; a neighbouring folder's calibration made
  for another detector is not adopted.

- **Command line (`tools/gimap_agent.py auto`): frames no longer overwrite each other, and a run can be traced
  to its inputs.** Each frame has its own folder (frames of one name in different folders used to share one;
  they now get `<name>_<folder>`), linked from `summary.md`, whose open questions keep every frame's items and
  name the frames. A frame argument may be a pattern (expanded by GIMaP when the shell does not, as in
  PowerShell) or a folder (every detector image in it); an `--out` folder that holds the summary of an earlier
  run is refused unless `--overwrite` (which removes only the frame folders that summary lists). `--mask FILE`
  applies a mask as Load Masks reads it, and every report says which mask was applied or that none was.
  `report.json` now holds the UTC time, the options as given (the notes too), every input file (frame,
  calibration, mask, background) by path, size and SHA-256, Analyze's record of the frame and, for a model
  fit, the network, the SHA-256 of its weights and the seed; `summary.json` is an object (`software_version`,
  `created`, `options`, `arguments`, and the former list as `frames`); `fit_solutions.csv` starts with the `#`
  lines of the curve tables. Reports have one language (English; `--language zh`, and the program's language in
  the window) instead of headings in both. The fit is announced as taking "some seconds to a few minutes", not
  "about 25 s". Physics and Methods 14.10 describes the record; sections 2.1 and 5.2 state the solid-angle
  factor as coded, (L/|t|)³, which is cos³ 2θ only for an untilted detector. `tools/benchmarks/cut_model_checks.py`
  has `--out` and refuses to overwrite its stored summary when an optional comparison was skipped (`--force`).
- **Fitting has one network method, Predict parameters, so the choice is simple.** The Fit step offers **Predict
  parameters** (GIMaP's network, its posterior and least squares), **Refine the values in Model** and **Search the
  ranges, then refine**; an unfitted GISAXS cut gets Predict parameters. Under Predict, **Particles** is *The
  particles in Model*, *Spheres and upright cylinders (1–2)* or *Any particles (1–3)*; there is one way through
  (no *Options ▸ Predict: the solutions* any more), and without σ in the file Predict takes a relative σ from the
  scatter of the curve about its trend. On one core it takes about 5 s per curve with the composition given, about
  25 s choosing among one or two populations, 30–70 s among one to three of spheres and upright cylinders, and a few
  minutes when random cylinders are among the choice (*Any particles*: 20–30 s per least-squares fit). **The fit1d kit and the quick
  physical fit are removed:** *Predict parameters (fit1d)* with `utils/gisaxs1d_gui_kit` (about 62 MB) and the
  dependency `threadpoolctl`, *Find the particle shape (no AI)*, and the *Physical fit (numerical)* and *fit1d*
  methods of the 1D Predict window, which now has the one method (its settings: composition, q unit and sides, the
  cut's q_z, the σ of files without one). Settings and projects saved with a removed method open with Predict
  parameters (fit1d's cut qz becomes the window's **Cut qz**). The benchmark scripts that compare against fit1d run
  with a copy of the kit you give them (`tools/benchmarks/README.md`).
- **Predict parameters: intervals that hold the refined value, and a posterior that converges on measured
  cuts.** The posterior around a solution is now sampled in up to eight rounds, the later ones over all values
  together, so a spread at its bound that trades against the radius, or a bright cut the model describes only to a
  few per cent, no longer leaves the weight on a handful of samples (the *few effective samples* note and the
  comparison by BIC become rare). Every 68 %/95 % interval holds the refined value and stays inside the value's
  range: a relative spread is never given a negative limit, and a length no longer gets limits orders of magnitude
  apart. A value the curve does not see (the shape of a population whose amplitude is 0) is shown without an error
  where it had an error of 0, in Fitting's own fits as well.
- **The automatic analysis fits with Predict parameters.** Run Automatic Analysis, Batch Export's *Particle model
  of the horizontal cut*, Process with AI, `tools/gimap_agent.py` and the MCP tool `fit_horizontal_cut` fit the
  horizontal cut with Predict parameters: without a composition the network chooses among one or two populations
  of spheres and upright cylinders, with or without a paracrystal (random cylinders only when given as the
  composition). The run reports the most probable solution with its
  probability and the 68 % intervals of R, h and D (*fit intervals*); the fit table has a probability column, and
  **Save Fit Table…** writes the probability and the intervals. When another solution has a probability of 0.2 or
  more, the run asks which model you expect (**model**); a reported value at a bound of its range is a point to
  check, and χ²/N above 10 a poor fit, as before. 2π/q\* no longer starts the fit (it stays a result of its own),
  and the solutions are no longer ranked by the Huber misfit with the rule of 10 % of χ²/N. Batch Export takes
  from some seconds to a few minutes per frame and needs no start values for the particle model. On the public GALAXI example the earlier first-level fit
  reported an upright cylinder with a paracrystal next to a second upright cylinder (R = 3.17 / 6.15 nm, D = 44 / –
  nm, χ²ᵣ 1.07, probability 0.39) and asked for the model, two more solutions being about as probable: for these
  spheres, fit a sphere in Fitting (manual, section 1.3).
- **Fitting ▸ Predict parameters (new):** GIMaP's own network finds the particles of a 1D GISAXS cut and their
  values from the curve. It proposes compositions (one to three spheres, upright or random cylinders, with or
  without the paracrystal) and values; Fitting's model weighs them against the measured points, and every solution
  is refined by least squares. Each solution comes with its probability and the 68 % interval of every value
  (**Results ▸ Fit details… ▸ Posterior of this solution**, and the column **p** of the solutions). The network
  (about 13 MB, in the repository and the packages) was trained on simulated cuts of Fitting's model and runs with
  numpy; GIMaP checks it against the SHA-256 of its record. It is also the one method of the 1D Predict window and of
  the In-situ series' **Predict each frame**. With both angles of the cut in Model, spheres have the direct and the
  exit-reflected path with a free effective reflection coefficient r (the network was trained on such cuts too), so
  the waves the surface reflects are taken into account. Physics and Methods 9.13 describes it and its benchmarks: on
  fit1d's 200 held-out curves its most probable solution has the true composition in 88 % / 65 % of them (fit1d's
  own physics / with beam tail and roughness; fit1d 69 % / 30 %) and a listed solution has it in 96 % / 84 %, with
  median errors of 0.5–0.7 % in R (fit1d 0.5–1.0 %) and a 90th percentile of R of 10 % / 14 % (fit1d 7.1 % / 19 %);
  on 600 curves of one to three populations 69 % / 58 %, 90th percentile of R 28 % / 45 %. Its intervals allow
  Fitting's model an error of 5 % of the intensity at every point (the fit and χ²ᵣ keep the curve's σ), and its 68 %
  intervals hold the truth in 58–91 % of the cases on the 200 curves (fit1d's errors 26–70 %). (These are the final
  runs of `tools/benchmarks/`, printed by `summary_tables.py`.) Besides the network's most probable modes it refines three modes of its tempered weights and, for two or
  three populations, starts at their degeneracies (another R at the same D, the paracrystals of two populations
  exchanged); choosing the composition it tries the network's four most probable and the most probable of every
  number of populations. The true solution is among those listed in 93 % and 86 % of these curves. Drawn with the
  four-term DWBA on Si, spheres beyond q₊R ≈ 1.7 (R ≈ 3 nm at α_i 0.4°, 12 keV) come out 4–7 % off in R (median)
  with both angles, and the true composition of such curves is the most probable in 50–70 %; with the q_z alone
  11 % (q₊R 2.3–3.5) to 40 % off (most probable in 3–8 %). With both angles
  Predict takes about 2.4 times as long, and on a curve without a reflected wave a smaller sphere with a large |r|
  can fit as well (`posterior_recovery.py --scattering dwba --cut two_path`, Physics and Methods 9.13).
- **1D Predict: General V5 and Single RC specialist removed.** On fit1d's 200 held-out curves V5 recovered R with a
  median error of 32 % and D of 22 % (20 s per curve); Predict parameters 0.6 % and 0.0–0.1 %. Settings saved with
  a V5 method open as Predict parameters. Fitting's *AI proposal (1D Predict)* is replaced by **Predict
  parameters**. 1D Predict no longer needs TensorFlow, and `modules/Fitting_1D_Model/Workflow_v5` (about 56 MB) is
  gone from the repository and the packages.
- **Sputter: the earlier method is removed.** Predict with its PyTorch networks, the Size Trend, Literature,
  Structure Sketch and Reliability tabs, the island height from the q_z fringes and Show in Analyze are gone; GIMaP
  no longer installs PyTorch. Sputter is Analyze on the main screen; a result folder of Predict no longer opens.
- **Sputter ▸ Details** shows only what Analyze uses: Sample and geometry, Analyze options, Static Frames and More
  options (mask, sample roll, where results go), and at the right **Windows** (the solution shown, one row per time
  window, copyable), **Report** (the networks used and tried, the mode, the mask, the run time, and `REPORT.md`) and
  **Logs** (the log of each step) for the result on the main screen.
- **Sputter, deposition rate:** a unit beside the rate, **Å/s** (the default, as a quartz balance shows it) or
  **nm/s**; a rate at the sample above 1 nm/s or below 0.005 nm/s asks whether the unit is right before Analyze
  runs, and the line under the bar gives the nominal thickness reached at the last frame. A rate remembered before
  is in nm/s.
- **Sputter:** the joint fit of the whole scan (every window refined together, smooth in time) is a solution of its
  own, *Joint fit (whole scan)*, shown first when the package made one.
- **Sputter ▸ Analyze, polymer and Cu samples (mode auto / refine):** the evolution no longer jumps from window to
  window. The two networks of a kind are compared on their candidates, and the better one is fitted over the whole
  scan at once: one metal-amount factor for the scan, sizes, coverage and amount that do not decrease, smooth in
  time. Windows are no longer refined one by one, and only one network is fitted, so a 60 s scan takes about a
  quarter of the time it took before (about 20 min instead of about an hour on a small office PC).
- **Sputter ▸ Analyze:** when the joint fit of the whole scan could not finish (it is tried once more with fewer
  processes when memory runs short), the result says so in the line above the window selector, in the warning
  colour: the solutions shown are the network's, and the evolution can jump between windows.
- **Sputter ▸ Analyze, faster import and a result first:** every frame is read from disk once (three times before),
  and only the frames the analysis uses are read: the first 30 s, the last seconds and the time windows with a
  margin — on a long scan a fraction of the frames, with exactly the same values in every window. For polymer and Cu
  samples the network's result (its solutions and their time path) is on the main screen within minutes, marked as
  preliminary, while the joint fit of the whole scan runs; the joint fit replaces it when it is done.
- **Sputter, speed of an analysis:** **Details ▸ Analyze options ▸ Parallel processes** — *Automatic (N)* (the
  default, from the cores and the free memory) or a number you choose, kept for the next session and used by Static
  Frames too. The live view says how many processes run and, with a newer analysis package, how long reading the
  frames from disk takes per frame; when the disk is the slow part (a USB or network drive, or a virus scan of every
  file) it says so and suggests copying the scan folder to a local SSD. **Details ▸ Report** adds *Time per step*.
- **Sputter ▸ Analyze says what it is doing now:** one line at the top of the live view, in the job status and,
  under the preliminary result, on the main screen — for example *Reading frames: 400 of 1,200 · about 4 min left*,
  *Network cu_v4: drawing the candidates with the forward model, window 7 of 24*, *Joint fit: iteration 2 of 6 ·
  median fit deviation 11.3 · 5:40 so far, about 11 min left*. While the joint fit runs, its current solution is
  drawn dashed over the preliminary result's evolution after every iteration, until the final result replaces it.
- **Sputter: the growth laws as a prior of the joint fit.** **Details ▸ Analyze options ▸ Growth-law prior**: **Auto** (the default) keeps D and R near the growth laws of the metal on its substrate before the film forms for Ag on PS, where this passed a held-out test fixed before the runs, and not for the other systems; **On** and **Off** decide for every system. Only the joint fit uses it (*Physics and Methods* 12.1 gives the terms). **Details ▸ Report** and the export's `METHODS.md` say whether it was used. An older analysis package gets **On** or **Off** as an option it does not know: keep **Auto** with it.
- **Sputter: a damaged frame no longer stops the import.** A frame file that cannot be decoded (cut short on disk, for example) is left out, and **Details ▸ Report** names it in the row of the import; before, the whole scan could not be analysed.
- **Sputter: the model split into its physical terms, and the growth process.** With a newer analysis package the
  switch beside the window selector has **Components**: the model image of the solution shown split into its terms
  (grains and islands, large clusters, film fluctuation and network, voids under the film, substrate seen through the
  metal, fluorescence, beam-path background; on porous TiO₂ the clusters in the pores and on the cap and the metal
  network in the pores), each with its share of the counts, and the shares over the scan. The evolution shows the
  stages of the growth (separate islands, coalescence, continuous film, after the deposition) as bands, and the new
  tab **Details ▸ Growth Process** lists the stages, the growth laws against the metal amount (coverage 0.5 and film
  formation at, D = D₁ h^p and R = R₁ h^q before the film, D after it, coarsening after the stop) beside the range of
  the same metal on the same substrate, and the shares of the terms per window. Results of an older package say that
  they have none of this.
- **Sputter ▸ Real Space:** a solution with a continuous base layer is drawn as a closed film: the top view is the
  metal of the layer over the whole square, with the grains as relief on it, and the line says *Continuous film …
  nm, grains on top: …*.
- **Sputter ▸ Analyze** starts up to 16 workers (two cores left free) and no more than the free memory holds after
  1.5 GB, 0.3 GB each.
- **Sputter: the continuous metal layer t_b.** A result of a newer analysis package shows the thickness of the
  continuous metal layer under the grains (0 while they are separate islands): a tile *Continuous layer t_b*, a
  dark-gold curve in the evolution (and in the joint fit's dashed overlay) and a column of **Details ▸ Windows**.
  Older results, which do not list it, show none of these.
- **Sputter ▸ Analyze, the steps as a tree:** with a newer analysis package the live view lists every step and
  sub-step like a task list (○ to come, ● at work with what happens now under it, ✓ done with what it found, ✗ failed,
  – skipped; done steps fold their sub-steps). A step marked › shows what it found: the beam centre and the pixel mask
  on the detector, the frames read, the start of the deposition, the time windows, the networks compared, the joint
  fit's iterations and solution. **Steps** beside the preliminary result's note opens them again; after the run
  **Details ▸ Steps** shows them for the result.
- **Sputter ▸ Analyze, the joint fit at work:** with a newer analysis package the line says how far an iteration
  has come, several times a minute — *Joint fit: iteration 3 of 6 · derivatives: 12 of 20 windows · 6:12 so far,
  about 9 min left* — with the time left from the pace of its windows.
- **Sputter ▸ Analyze, storage:** the frames Analyze imports go to a local folder (`%LOCALAPPDATA%\GIMaP` on Windows,
  `~/Library/Caches/GIMaP` on macOS), no longer the roaming user data folder; imports kept there by an older GIMaP are
  still used. They are limited to 5 GB together by default (the least recently used are removed first, never the one
  in use, never a result); **Details ▸ Storage** lists them with their size and last use, removes chosen ones or those
  unused for some days, and sets the limit.
- **Sputter ▸ Analyze, when nothing happens:** the clock line of the live view says how many of the analysis's
  processes are at work now (*Running for 03:12 · Processes: 16 · 14 at work now*), and a line without a count of its
  own says how long the run has been at it (*Starting network cu_p5 (the first of 2 to compare) · 0:45 so far*). When
  none of the processes has computed or read for a minute and the package has said nothing, a note across the live
  view says that it has most likely stopped on an error and will not go on, with the last line of its log, and
  **Stop Analysis**; the line at the top and the job status say *No progress for 1:10: the analysis package's
  processes are all waiting*. Before, such a run showed the same line for ever.
- **Sputter ▸ Analyze away from the networks' wavelength runs to the end.** A Cu scan away from 0.105 nm (or any scan
  away from its networks' wavelength by more than 5 %; Au away from 12 keV) is analysed with optical constants
  computed at its own energy, the networks giving only the starting solutions and **Auto** picking **Refine**; a photon
  energy outside 4–30 keV is refused when the scan is imported, with the reason. Before, such a Cu analysis waited
  without ending (manual chapter 13 says what this means for the result).

- **Checked against pyFAI:** on the four public frames, every pixel's q, 2θ, q∥ and q_z agree with pyFAI to
  10⁻¹³, and the azimuthal I(q) to 10⁻⁷ on the same pixels and bins; pyFAI's Σ I / Σ F differs from GIMaP's mean of
  I / F by up to 1.3 % on a frame close to the sample (Physics and Methods 3.7, 6.1;
  `tools/benchmarks/pyfai_comparison.py`).
- **Analyze, refraction (optional):** with the film's critical angle (**Corrections ▸ Refraction in the film,
  αc**, or `--critical-angle-deg`), q_z is the one inside the film, k(√(sin²αf − sin²αc) + √(sin²αi − sin²αc)); q∥
  does not change, pixels below αc are left out of what uses q_z, and the record, CSV headers and report say so
  (Physics and Methods 2.6). Off by default.
- **Fitting, D ≥ 2R:** a fit keeps the paracrystal's distance at or above twice the mean radius, so the particles
  do not overlap (D could fall to 1 nm before); a D at 2R is reported at its lower bound (Physics and Methods 9.6).
- **Lighter install:** `requirements.txt` is the core (Analyze, Calibration, Fitting with Predict parameters, Series, Batch
  Export, the automatic analysis, XRR, Sputter, SEM). TensorFlow (2D Prediction, Trainset training),
  BornAgain and the AI clients are optional: `requirements-tensorflow.txt`, `requirements-bornagain.txt`,
  `requirements-ai.txt`, all of them `requirements-full.txt`. GIMaP starts and opens every workspace without them,
  and a feature that needs one says which file to install. `Start_GIMaP.bat` and the portable packages still set up
  everything.
- **Earlier 1D models removed:** the Keras models K1–K4 of 1D Predict (`k1_k2_k3_k4_phys`, `saved_model_K3K4`,
  22 MB) and their research code `utils/ML_Fitting_1D_GISAXS` (490 files). 1D Predict runs GIMaP's own
  network (Predict parameters). A folder of an earlier model is refused with what to use instead.
- **1D Predict, Use current curve:** the curve of Single analysis, as the page selects its points: the halves chosen,
  the fitting range (|q|) and the points left out. Before, it took every point of the halves chosen. The former
  Fitting pages, built but no longer shown, are removed.
- **Fitting, beam tail:** the term A / (1 + (|q|/w)^ν) is labelled what it is, the empirical tail of the direct
  and reflected beam (**Background and beam tail**, **Tail A / w / ν**; it was the *resolution peak*). The q
  resolution is σ_q. Keys and records are unchanged (`res_amplitude`, `res_width`, `res_exponent`).
- **Tests against pyFAI and pygid:** q, 2θ, q∥ and q_z at fixed pixels of the four public frames are checked
  against stored pyFAI and pygid values to 10⁻¹¹ Å⁻¹ in every test run, without either program installed; the P08
  ring positions are now held to 0.0025 Å⁻¹ (was 0.01).
- **Checked against pygid and BornAgain:** every pixel's q∥ and q_z agree with pygid to 10⁻¹⁵ once its pixel
  convention (pixel i at i·p, half a pixel from pyFAI's centre) is taken into account. On the public GALAXI frame,
  Fitting finds the same sphere radius and spread on the measured cut as on BornAgain's model of it simulated on that
  cut; the cut does not determine the distance D (Physics and Methods 3.7, 9.10).
- **Fitting, q resolution:** σ_q in *Background and resolution peak* (0, off, by default) convolves the particle
  terms with a Gaussian in q, so an ordered film's peak width is not read as disorder (Physics and Methods 9.2).
- **Fitting, Coupling:** a particle with Distance D has *Coupling* LMA (P·S, as before) or DA (P + |⟨F⟩|²(S − 1):
  sizes independent of positions), so the part of a size spread that the spacing does not modulate is described by
  the particles themselves instead of a further population (Physics and Methods 9.5, 9.11).
- **Command line:** every CSV of `tools/gimap_agent.py` begins with `#` lines that give the release, each input
  file with its SHA-256 and the geometry, so a curve says where it came from without `report.json` beside it.
- **Fitting, mean of both halves:** the q > 0 half gives the |q| points, as in Analyze and the Series map, so one
  cut gives one mean curve in every page (Fitting took the half with more points: a point more or less).
- **Fitting, results:** each particle's radius by number (microscopy counts particles; Fitting's spheres
  and random cylinders are weighted by intensity) and σ_nn/D, also in the exported record (`derived`). For a broad
  sphere distribution (σ_R/R above about 0.2) the radius by number is *not defined* (Physics and Methods 9.3).
- **Geometry Calibration:** the beam centre and distance have 1σ estimates in the calibration record,
  the result tooltips and the instrument profile's source. Physics and Methods 3.3 gives the estimator for each ring count.
- **Analyze, a tilted detector:** the rotations of a `.poni` (pyFAI's rot1, rot2, rot3) are part of the
  geometry. Every pixel is placed where pyFAI places it, so q, χ, the exit angle of a row, the solid angle and the
  polarisation factor are those of the tilted detector (on the Xeuss example |q| changes by up to 6.3 %; it now
  equals pyFAI's to 10⁻⁹). The tilt is a decision of the automatic analysis instead of a point to check, and the
  record has `rot1_rad` … `rot3_rad` in `geometry`. Physics and Methods, section 3.7, is rewritten.
- **Analyze, records:** `<stem>_analysis.json` lists every input file by name with its SHA-256
  (`input_sha256`: the source, its module or data files, and the frames summed with it), so a curve can be traced
  to its frames on another computer, where the folders differ.
- **Fitting, records:** the source of a curve is recorded with its file name and SHA-256.
- **Physics and Methods, section 9.12:** a record of fit1d (removed in this version) — its model and how well it
  recovered known parameters on 200 curves it had never seen, against which section 9.13 compares Predict parameters
  (the benchmark is `tools/benchmarks/fit1d_recovery.py`, with an independent forward model).
- **Fitting, spheres in grazing incidence:** a new card in Model, *Spheres in grazing incidence*. By default
  nothing changes: spheres are evaluated at q_z = 0 (the effective interface approximation, no substrate
  needed). Entering αi, αf and λ of the cut evaluates spheres as the direct plus the exit-reflected path with a
  free effective reflection coefficient |r|·e^(i·arg r); the results then show q₊R of each sphere. Physics and
  Methods has new sections 9.10 (why upright cylinders are
  exact in this approximation and spheres are not, with examples on Si and on a polymer film) and 9.11 (what each
  population of the usual three-population fit takes up).
- **Fitting:** parameter limits and fixed values are enforced by Refine and Search, including linear
  intensity/background terms; a value at the edge of its range is named, never widened by itself. Complete
  solutions can be switched as a whole from the result dropdown; components are available in the curve's
  right-click menu. Detailed diagnostics move to **Fit details…**.

- The **Sputter** workspace opens on a new **main screen**. **Analyze** (scan folder, metal, substrate and
  deposition rate in one input bar) runs a separate analysis package and shows, window by window, the measured
  image beside the model image of a solution, the solutions to choose from (the network's candidates and the
  refined one, each with its fit deviation), and the radius, height, distance, gap, size distribution width and
  coverage of the solution shown with their evolution over the scan. The solution shown first is the one a smooth
  evolution over the whole scan selects; a choice of another one is kept with the result. **Open Result…** shows a
  folder analysed before. Everything the page held before — the geometry, **Predict** with the networks that
  come with GIMaP, **More options** and the five tabs — is in the second view, **Details**, and works as before;
  the scan folder moved into the input bar, and the metal is one choice for both methods.
- Sputter main screen: **Patterns | Real Space** beside the window selector draws the solution shown in real space —
  a cut through a row of grains with the base layer under them, and the view from above — at the radius,
  distance, size distribution and height of the solution, in nm; touching grains are drawn as one piece and the
  picture says whether the grains are separate, coalescing or percolated. Results on porous TiO₂ show their own
  values (island radius, relative; island, cap and film shares; the correlation lengths of the metal in the pores),
  and Real Space draws the TiO₂ layer with the metal in its pores, the cap on the particle tops and the clusters.
  **After Deposition** (the input bar) analyses samples measured after their deposition, a whole folder of them at
  once, into series the main screen shows along the nominal thickness.
- The analysis package of Sputter ▸ Analyze and Static Frames comes with GIMaP (`utils/sputter_client`): nothing
  has to be chosen. **Choose Folder…** or `GIMAP_SPUTTER_CLIENT` still use another package; a folder named so that is
  not an analysis package (chosen once and moved since, or the wrong folder) is passed over for the one that comes
  with GIMaP, and the Analysis package line says why. It is under the MIT License (`utils/sputter_client/LICENSE`).
  Without a mask file it uses the pixel mask its network was trained with (Cu on PEO: the P03 mask of 2021; before,
  Cu on PEO failed at predict with "cache mask …, IAM_PIXMASK requires …"), and a mask file you choose is accepted
  by every network. The package also masks what the frames themselves show to be unusable - dead and hot pixels, the
  beamstop and its holder - on top of the mask (an automatic mask; the tooltip of the result's name says which mask
  the frames were read with).
- Sputter ▸ Analyze: the networks for Ag on polymers and for Cu on PEO were trained again with random extra masks
  (blocks like beamstops, lines like module gaps), so their answer hardly depends on which pixels are masked - another
  beamtime's mask or the automatic one. With 10 % more of the pattern masked, the first radius they give moves by
  3 % (polymer) and 1 % (Cu) instead of 7 % and 19 % (median over real windows). On the reference scans the best
  candidate fits as well as before or better (fit deviation 0.82 of before on average for the polymer scans, 0.94
  for the Cu scan; the first Cu candidate 1.07). The TiO2 network and the two networks of Static Frames were trained
  the same way: their candidates fit the reference frames clearly better than before (fit deviation 0.61 of before
  on 13 TiO2 scans, 0.78 on 211 static frames). The Si network is the one before (two retrainings did not fit the
  reference scans as well). Static Frames use the automatic mask too, so frames of another detector size are read
  instead of failing on the mask.
- Sputter: Cu is Cu. The metal list says Cu (no more "approximated with the Ag model"); Analyze reads Cu with its own
  networks, and the earlier method (Predict), whose networks are for Ag and Au, does not run for Cu and says so.
- Sputter ▸ Analyze reads scans whose first frames are empty (the X-ray shutter opened after the detector started):
  the automatic beam centre is taken from frames with counts only, where before one empty frame among those it looked
  at made it fail ("No supported specular-centre candidate"). Frames on a network drive that its sync client holds
  for a moment are read again (a few tries) instead of failing the import.
- Sputter ▸ Analyze: a new **Substrate** choice, **Polymer**, for a polymer film of another or unknown kind (a block
  copolymer, for instance): the model describes every polymer as one film layer, and which polymer it is changes the
  metal's sizes by a few percent. Cu runs on every polymer and Ag on PEO too. The Cu network was trained again with a
  second scan (Cu on a block copolymer, grown to a closed film): on that scan the network alone fits clearly better than
  before, and Analyze still runs the earlier Cu network next to it and keeps the better of the two for each scan.
- Sputter ▸ Analyze, Cu: a closed Cu film (from about 10 nm) is fitted much better. Its exit-angle fringes come from the
  whole film thickness; the model now describes the film as a dense layer under small grains, with lateral fluctuations
  through the whole film whose size spectrum is that of a continuous film, and the Cu network was trained again on
  that. On the closed-film part of the reference scan the fit deviation after refinement drops by about a quarter; the
  earlier Cu network still runs next to it, and Analyze keeps the better of the two for each scan. The other networks
  are unchanged: the new term is off for them and their results are the same as before.
- Sputter ▸ Analyze: a scan whose first frames are dark (the X-ray shutter opened after the detector started) takes its
  bare substrate and the start of the deposition from the frames after the shutter opened. Before, the dark frames
  went into the bare substrate, which made it several times too weak in every time window of such a scan.
- Sputter ▸ Analyze shows what it is doing while it runs: the main screen becomes a live view with the four steps
  (read the frames, find the solutions, write the report, draw the images), each with its state and one line of what
  it found, and, for the step at work, the detector summed with its mask and the frames read last, the count rate of
  every frame read so far, then the measured and model image of each time window as it is solved and the sizes found
  so far. A click on a step that has started shows what it found. The analysis package that comes with GIMaP sends
  what the view shows (an older package chosen in Details is followed by its steps alone); the results are the same
  as before. Between the two networks of an analysis the view keeps the last solved window instead of going blank.
  While the solutions are found, the step says what the network is doing and how far it has come — network inputs
  prepared, candidates proposed, windows solved and model images computed, windows refined, iterations of the
  whole-scan refinement — and the bar of the job status moves on inside each step. In Refine mode every window gets
  its candidates before the refinements start, so the evolution fills early (the results are the same).
- Sputter: **In Situ | After Deposition** at the left of the input bar. **After Deposition** is for samples
  measured after their deposition, which have no deposition rate and no frames from before it. **Add Samples…** takes
  a folder of samples — each folder in it with CBF frames is one sample, however deep the frames lie — or one
  sample's folder or one CBF file (zip archives are no longer opened: unpack them first). The **nominal thickness**
  (`10Au`, `6.5nm`), a HiPIMS deposition (`H10Au`) and the **bare substrate** (a folder named like the part before the
  thickness, `850` for `850_10Au`; without one the film is taken to lie on Si) are read from the folder names; what a
  name does not say is asked for in the list, not guessed. The **Sample List** at the left of the main screen groups
  the samples into series (substrate and deposition method), with a box to tick each, its thickness (double-click to
  change it) and its state; **Analyze** runs the ticked samples one after another (*Sample 3 of 12*), a failed one does
  not stop the others, and a right-click analyses one sample or one series. While a sample runs the main screen shows
  three steps live: the frames read (which are usable, which are left out and why), the solution of each frame as it
  is found with its measured and model image, and the series with every frame solved so far. Choosing a sample shows
  its series at that sample. **Export…** writes one HTML report of every series of the list, each in two parts —
  separate particles (radius, height, spacing, gap, coverage) and the connected film (thickness, density, height and
  the relief of its surface: a connected film has no separate particles) — with their tables, the stage of each
  sample as the Real Space picture names it, a figure of both parts and the fits, and a CSV of every sample. **Details ▸ Static Frames** keeps
  the results folder (one folder per series in it), the mode and the geometry; **Analyze Sample** there is gone.
- **Geometry Calibration**: the **Distance range** is a limit, not a starting point. No solution outside it is shown,
  also not after the fit of centre and distance (before, the fit could leave a custom range); when none lies in it,
  the calibration says so. An image of the standard taken at grazing incidence (AgBH on a wafer: arcs above the
  sample horizon, the specular rod, the Yoneda band) is recognised: the rod is left out of the ring search and the
  fit, a horizontal band above the beam is not taken for a ring, the rings are looked for in the half of the image the
  sample does not shadow, and the coarse beam centre is polished to the pixel first. For silver behenate the orders
  between the matched ones must all be there, so the Yoneda band is no longer taken for the first ring of a solution
  at half the distance; the best solution near a stated estimated distance is always refined. **Auto Detect**
  without an estimated distance now ranks silver behenate first on the public silver behenate image (before, a LaB₆
  solution at a much shorter distance).
  A matched ring counts in the ranking by how clearly it stands out: four orders paired with peaks of a few percent
  contrast no longer outweigh three clear rings (with NumPy 1.26 / SciPy 1.15 such a chance sequence gave 1683 mm for
  a 2200 mm grazing image); the real test frames give the same solutions as before.
- Sputter ▸ Details: a long folder of the analysis package wraps inside its column with any font (on macOS a long
  folder name widened Details).
- Sputter ▸ Analyze, limitations: the analysis package analyses only the pairs of metal and substrate it has a
  network for; its values are
  model values that were not compared with an independent measurement of the sample; with an older package a step of
  the analysis does not report progress while it works.
- Fitting's **Components** control shows additive particle, background and beam-tail/low-q contributions with
  separate visibility choices and thinner dashed/dotted lines below the total model. Toggling contributions
  preserves the current view. Automatic limits use data and total model by default; **Include components in
  auto range** explicitly resets the view to include visible contributions. Fits, residuals and numerical exports
  are unchanged; zero and negative contributions remain absent from logarithmic display.
- Fitting adds **Import Curves…** for selected TXT/DAT files and shared-q tables (`q I1 I2 …`), with column
  preview, explicit layout and q-unit choices. Existing `q I [sigma]` inputs and Analyze's `q I sigma pixels`
  exports keep their roles. Independent samples start from the same model without time continuity or odd-frame
  exclusions; ordered sequences retain the selected file/column order. Projects and exports preserve every
  curve's physical source and column mapping. Multi-file drops open the preview.
- Curve rows with an unreadable sigma now skip the whole row together, preventing later errors from shifting
  onto earlier q/I values; valid three-column data are unchanged. Missing sigma stays absent.
- Curve import limitations: interleaved `q I1 sigma1 I2 sigma2` tables are not supported; use separate
  `q I sigma` files. Headerless three-column data require an explicit layout choice. Fixed imported selections
  do not watch for new files or columns; folder mode continues to watch ordinary single-curve files.
- Project saves write Compare arrays to a unique `<name>.<generation>.compare.npz` and replace the project
  atomically only after the new arrays and temporary JSON are complete. Failed saves keep the previous pair;
  older sidecar names remain readable. Move the project with its current `compare.arrays_file`; an old array
  file can remain if another program prevents its removal.
- Compare immediately hides stale results and disables table/plot exports after a series or setting changes.
  Results return for the current settings, including immediately when the original settings are restored.
- Compare discovers folders, imports curves and restores project arrays in a background queue with progress
  and Cancel. Completed series stay; unfinished and queued series are discarded. Saving a project waits until
  imports finish or are cancelled. Read failures are shown in Compare, and imports that add nothing keep the
  existing comparison. Cancellation cannot interrupt an active directory scan or single-table read; its result
  is discarded when it returns, while separate curves stop between files.
- Fitting scans folders, checks unit headers, previews curves, reads detector cuts and watches folders in the
  background. Start waits for the initial listing; Stop cancels preparation. Stop during reading skips that
  frame's fit without recording a false failure; Pause lets the current frame finish. Completed fits stay.
- An omitted `.poni` Rot3 is reported separately from the |q| error in Analyze, automatic reports and exports.
  Every nonzero rotation is explained; above 1° it is an independent review item. This is a reminder, not an
  accuracy guarantee. Rot3 can leave |q| unchanged while affecting χ, qy/qz and in-plane/out-of-plane cuts;
  full rotation geometry remains unsupported. Original rotations and separate magnitude/direction flags are saved.
- Project reopening restores Analyze's source order, identity and original frame after missing files,
  preserving the order used for adjacent single-frame sums and series,
  including chunk-to-master frame and import-selection mapping.
- Fitting projects and sessions restore the model and the chosen method with the curve.
- Stop remains a stopped outcome during boundary refinement and queued result delivery, while retaining
  the completed numerical result; an interrupted boundary check does not claim convergence or a settled bound.
- Removed Trainset Build from navigation, menus and application startup. Existing research files remain untouched.
- Windows help preserves the original formula spacing and inline code across Qt versions. NeXus monitoring
  reinspects recent writes and pending writers so appended frames are detected even when the file size
  and timestamp are unchanged; settled archived files retain their cached layouts.
- Windows build and source setup scripts use their own PowerShell modules when launched through PowerShell 7.
- Windows portable packages preserve empty runtime files when adding the application, so standard Windows
  extraction does not turn Python package initializers into invalid files that prevent startup.
- XRR excludes integer negative gap codes while keeping floating negative data, reads urad/µrad/μrad correctly,
  rejects unknown angle units and uses transformed NeXus header beam centres.
- Save Project restores the Sputter form and references its result folder; older projects clear that workspace.
- Fitting keeps physical **Fit a Folder of Curves…** separate from the experimental **1D Predict — Fit Many Curves…** entry; the manual explains which workflow to use.
- 1D Predict writes `AI_Fitting_Output/v5_<date>_<time>_<µs>/` in the user data folder, independent of the launch folder.
  Sputter, 1D Predict, AI assistant and command-line reports identify the GIMaP release.
- Added a macOS portable builder and checks for Windows x64, macOS Apple Silicon and Intel. Package acceptance
  remains separate from source checks; release assets state the tested systems.
- Research code and bundled network licences confirmed by their author Yufeng Zhai as MIT; dependency licences unchanged.


### Sputter (new workspace)

- **Sputter** (sidebar, `Ctrl+5`) follows an in-situ sputter deposition from the PILATUS CBF frames of one scan,
  or a series of samples measured after their deposition. **Analyze** runs the analysis package that comes with
  GIMaP and shows, window by window, the measured image beside the model image of each solution and the grain
  radius, height, distance, gap, size distribution width and coverage with their evolution (manual chapter 8;
  Physics and Methods chapter 12). The lines above under *Release preparation* say what was added to it since.
- Each value of the geometry says where it comes from: this folder's last values, a geometry preset of the
  beamtime (**Save Geometry as Preset…**), the file header, or the values used last (*please check*).

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

- **Crystal orientation** reports the mosaic width and fraction of each of up to two fibre components, including two widths about the same plane normal. It models exit transmission from the CIF composition and reports per-ring residuals. The public MAPbI₃ frame has an asymmetric horizon loss that prevents a quantitative texture assignment.

- With phases loaded, Analyze and Run Automatic Analysis fit a shared crystal fibre texture to each phase's assigned GIWAXS rings. The Pole Figure tab shows how each reflection family contributes, with mosaic width, oriented fraction, competing axes and per-ring residuals.

- **Flat field (detector response).** Mask & corrections ▸ Corrections has a **Flat field** (**Choose…**, **×**;
  off by default; GISAXS and GIWAXS): every frame is divided by a detector image of its size (CBF, TIFF, EDF,
  NeXus/HDF5) or a `.npy` array, normalised to mean 1 over the pixels it keeps; pixels where it is at or below
  0.05 are left out and counted, and one of another size is refused, never resampled. It comes right after the
  background, and with the GIWAXS intensity corrections the factor is `F = F_Ω F_P F_A F_ff`. For photon counts
  the error bars follow it (`scaled_poisson`: Var = I / F²). Batch Export, series and the folder watch use the same
  flat field; settings files keep it; the JSON record has `corrections.flat_field` (file, SHA-256, normalisation
  mean, pixels left out). `tools/gimap_agent.py auto --flat-field PATH` does the same (a missing file: exit code
  1), and every report names the flat field among its Decisions. *Physics and Methods* 4.10.
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

- **One forward model.** Fitting computes its curves with one implementation of a 1D cut (Physics and Methods 9.2),
  checked against fit1d's, 1D Predict's, an independent model and BornAgain. What you see:
  - Each particle card has **Sizes**: *Gaussian, by number* (the default for a new particle: the mean and spread of
    the particle count, as a microscope counts), *Log-normal, by number* (new, for broad distributions), or
    *Gaussian, by intensity* (what Fitting used before). Results give every particle's sizes by number and by
    intensity.
  - The spacing disorder is **σnn/D**, the relative standard deviation of the distance between neighbours,
    √(2π) ≈ 2.51 times the former σD/D. Default 0.25.
  - Every family's form factor is 1 at q = 0, so a vertical cylinder's **Scale** is in the same units as a sphere's.
  - The size and orientation averages converge at large q: the former fixed rules were off by up to 67 % (spheres
    of 20 nm with σR/R = 0.4 at q = 3 nm⁻¹) and 55 % (random cylinders) at high q.
  - A model saved before opens with the sizes it had, its σD/D converted to σnn/D and a vertical cylinder's scale
    converted; its curve changes only where the former rules were off.
  - Sizes by number come from small tables of the exact averages (about 1 ms per evaluation, as fast as before).
    Where no table applies the converged averages cost time: **Search the ranges** with *Gaussian, by intensity* takes
    tens of seconds, with a random cylinder up to minutes (*Known limitations* in the manual).

- **Single analysis** has one **Method** drop-down — **Predict parameters**, **Refine the values in Model**, **Search
  the ranges, then refine** — and the button carries the method's name. An unfitted GISAXS cut gets Predict
  parameters, with a line that says why.
- **Results** says what to do next (**Refine These Values**, **Fit Again**, **Fit Every Curve in This
  Folder…**). A value that stops at the edge of a Model range stays within that limit and is named in *Fit
  details*. Select values and press **Fix**, **Let It Fit** or **Range…**: the next **Fit Again** uses them.
- **In-situ series**: **Predict each frame** as a method (Predict parameters on every frame, then refined),
  preferring, of solutions about as probable, the one that continues the previous frame; **Fit Every Curve in This Folder…** in Single analysis starts it on the curve's
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
- **Single analysis** keeps its four choices. *Predict parameters* cannot fit both halves on |q|: it switches to
  the mean of both halves and says so. A mean never interpolates across a gap of the other half. The model fit of
  Batch Export and of the automatic analysis takes a cut of any length (more than 1000 points are merged).
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

- **A source folder starts with one double-click.** `Start_GIMaP.bat` in a clone or a Download ZIP of the
  repository sets up a private environment in `.gimap` of the folder at its first start: uv (pinned version,
  checked against its SHA-256), Python 3.10 and the packages of `requirements.txt` (about 2 GB), with a progress
  bar over the steps; 3–20 minutes depending on the
  connection. Later starts are immediate; when an update changes a list of packages, the next start adds what
  is missing. When installing the packages fails (for example for lack of memory), it tries again one package at
  a time. Nothing outside the folder is changed. After the folder is moved, delete `.gimap` (or run
  `tools\setup_source_env.ps1 -Reinstall`). A venv or conda environment you made yourself is never used by it;
  start GIMaP from one with `python main.py`.
- **`Update_GIMaP.bat`** runs `tools\update_source.ps1` (git clones only). The script now says when the list of
  packages changed (`requirements.txt`): nothing to do with `Start_GIMaP.bat`, and
  for an environment you set up yourself it prints the `python -m pip install -r` line for each. In a folder
  without `.git` (a portable version, a Download ZIP) it says what to do instead and changes nothing.
- **The Windows package** comes as one ZIP, or, above 1900 MiB, as parts with an installer,
  `00_Download_and_Install_GIMaP-<version>.bat`, that downloads them, checks each one's SHA-256, joins and
  extracts them (`Install_GIMaP_parts.bat` joins parts you downloaded yourself, without checking them). The release page has
  `SHA256SUMS.txt` with the checksum of every file to download. The package holds `licenses\`: the licence files of
  everything in `runtime\`, with GIMaP's `LICENSE` and `THIRD_PARTY_NOTICES.md`. Its Python environment holds
  Python 3.10 and only the packages of `requirements.txt`.
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

- The model fits of Analyze show **χ²/N** where they showed χ²: Run Automatic Analysis (GISAXS) and its saved report and fit
  table, the model rows of Batch Export, the fit results of Analyze's series, and a solution that Analyze sent to
  Fitting (**Show in Fitting**). χ²/N is the mean squared σ-normalised residual without a degrees-of-freedom
  correction; Fitting's own χ²ᵣ is unchanged. The `chi2` column of the model-table CSV keeps its name, and a `#`
  line says what it holds.
- Run Automatic Analysis (GISAXS) said it chose the fit with the lowest χ². It reports the most probable solution
  of Predict parameters; `report.md`, the fit table's tooltip, `fit_solutions.csv` and the manual say so.
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

- **Linux (from source):** the interface uses Ubuntu's font, Ubuntu Sans (else Liberation Sans or Arimo), where it
  is installed, instead of DejaVu Sans, which is about 13 % wider: the panels fit as on Windows and macOS. A
  calibration file named in the notes in other letter cases (`GEOMETRY.PONI` for `geometry.poni`) is found there
  too, as on Windows and macOS. Seen first on Linux, fixed everywhere: a wrapped line of Fitting ▸ In-situ series's
  list of frames could lose its last letters at the right edge; Analyze's command bar could misjudge by a pixel or two
  when to use its shorter texts; the value fields of XRR's geometry did not line up when a tag's longest word was
  wider than half the row.
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
- The Chinese tooltip of the **Sputter** entry in the sidebar was in English.
- **2D Prediction ▸ SEM comparison…** attached the results of the Ag and Cu distribution modules, which declare
  no axes, with a placeholder 0.05–15 nm axis that is not a physical scale.
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
  while q_y and 2π/q* barely change — and that the σ of the cuts, by which χ²/N is normalised, are statistical only;
  the intervals of R, h and D are the posterior of Predict parameters given those σ.
- GISAXS automatic analysis: 2π/q* of a peak or shoulder of the horizontal cut was reported as the in-plane
  spacing D, although a form factor or the beam tail makes a shoulder too (on BornAgain simulations of the GALAXI
  model, 73.9 nm for a true 53.6 nm, and a shoulder at 60.6 nm without any interference). Every report, table
  header, *Decisions* line, command-line summary and AI tool description now calls it *a length scale, not
  necessarily the particle spacing; check it with a model fit* (report section **Length scale 2π/q\***, decision
  `length scale`, batch column `2π/q* (nm)`); the numbers and the record keys (`spacing`, `distance_nm`) are
  unchanged. The model question names the solutions whose D is within 15 % of 2π/q*, no longer "the observed
  spacing".
- GISAXS automatic analysis: the solution reported was the first-ranked one even with a value at a search bound
  (on the public GALAXI frame a random cylinder with D at its 500 nm bound), and nothing said so. The fit now names
  each value that ended on a bound (`at_bounds`), and when the reported (most probable) solution has one, *Needs
  attention* says so. The amplitudes, the background and the beam-tail amplitude are solved non-negative and are not
  counted: one at 0 is a *fit caveat* of the *Decisions* (*background BG at 0, the non-negativity limit*). The fit
  table of the Results tab marks a value at a bound ⚠, names them by rank under the table (*1: D at its upper bound,
  …*) and selects the reported solution; `report.md`
  marks it *(chosen)* and gives the values at a bound in the note column, `fit_solutions.csv` has `at_bounds` and
  `chosen` columns, the fit record `chosen`, and `fit_curve.csv` the reported solution's curve (Physics and Methods,
  section 8.5).
- GISAXS automatic analysis: *Needs attention* gave D = 48.4 nm where the fit table said 48.3 nm, from the tool's
  answer rounded to four significant digits and rounded again. Every number is now formatted once from the fit
  record.
- GISAXS automatic analysis: a reported fit far from its curve passed without a word (the public AgBh SAXS frame,
  Bragg rings of a calibrant, gave a sphere with χ²/N = 472). A converged reported solution with χ²/N above 10 — the
  limit of Fitting's poor fits — is now a point to check (**Fit**, exit code 2): its R and D are not measurements.

### New known limitations

Chapter 13 of the manual is the complete list for this version. These are the ones added since 0.1.0: the
limitations of the new features, and ones found or written down since. The limitations listed for 0.1.0 still
hold.

- **Sputter reads folders of CBF frames only.** For NeXus series use Analyze's Series map and Fitting's In-situ
  series.
- **Sputter keeps every result folder**, and the frames its analysis package read (`sputter_client/`), in the user
  data folder unless you chose another place; nothing removes them, and **File ▸ Save Project** holds its parameters
  and result-folder reference.
- **NeXus/HDF5 files not written by the Lambda server are now read as stored**: a geometry or mask made on such a
  file with 0.1.0 no longer fits it. Check the beam centre and masks; calibrate again if needed.
- **An instrument profile is matched by detector name and frame size only**, and a `.poni` or `.json`
  calibration is applied without checking its pixel size or frame size; this adds to 0.1.0's limitation of the
  automatic search, which can find the calibration of another experiment. Check the distance in **Geometry**.
- **Geometry Calibration fits silver behenate, LaB₆, CeO₂ and a LaB₆ + CeO₂ mixture only.** The AI's geometry
  search recognises images of Si, Al₂O₃ and Cr₂O₃ by their names but cannot fit them: calibrate such an image
  with pyFAI and give GIMaP the `.poni`.
- **Analyze's normalisation divides the frame after the background and dark frame are subtracted** in the frame's
  own units, with the scales you type: the background's own exposure or monitor is not used. Set its **Scale** to
  the ratio. Transmission and thickness are typed, not measured.
- **XRR intensities are raw**: no monitor normalisation, footprint, background or resolution correction, and no
  θ offset. Angle units and signed-integer gap codes are checked before extraction.
- **Format Converter does not read EDF** (use Batch Export ▸ Converted detector frames), and its uint16 modes
  mark pixels without data only in HDF5: in a TIFF or CBF such a pixel holds 65535 and reads back as a valid
  value.
- **SEM sizes depend on the segmentation**, and the default StarDist weights are trained on cell nuclei, not
  SEM. The scale is never read from the image.
- **2D Prediction reads CBF frames only and applies no geometry and no correction** to a frame: the module's
  own preprocessing is all that runs. **The Ag and Cu distribution modules declare no axes**: their maps and
  curves are drawn on a placeholder axis (0.05–15 nm), which is not a physical scale.

- **Predict parameters knows Fitting's model only** (sizes Gaussian by number, the LMA, spheres at the Born term of
  the cut's q_z or the effective interface). It was checked on simulated curves of an independent model and on
  BornAgain simulations, not on measured curves of known particles; physics outside the model can give a
  composition a probability it does not deserve. Judge every solution by its residuals.
- **The model fit of the automatic analysis chooses the particles from the curve alone** (Predict parameters among
  one or two populations of spheres and upright cylinders). On the public GALAXI frame of Ag spheres the earlier first-level fit reported two
  upright cylinders. 2π/q* is a length scale, not the particle spacing: on BornAgain simulations it was within 10 % of
  the true distance in 1 of 10 cuts with interference. Choose the model from what you know of the sample and confirm
  it in Fitting.
- Cancel/Stop records a request immediately but cannot forcibly interrupt a directory scan or single-file read.
  Unfinished Compare imports are discarded, or a Fitting frame is not fitted, when the active read returns.
- Older Analyze and Fitting records may lack a reproduction recipe.
  Sputter, 1D Predict, AI and CLI records now identify the software version; Predict parameters uses a fixed seed.

- **API keys saved without an OS credential store** are kept in plain text in the user data folder
  (`claude_api_key.txt`, `assistant_provider_keys.json`). On a shared computer, use the provider's
  environment variable or **Remove** the saved key after use.
- **GIMaP's detector is flat and perpendicular to the beam**: a `.poni` tilt (and `Rot3`, and its `orientation`
  entry) is left out, and Geometry Calibration fits no tilt. On a tilted detector |q| is off away from the beam (up
  to 6.3 % on the Xeuss example); the error is now computed and reported, not corrected. For exact q reduce such
  frames with the full `.poni` geometry (pyFAI). Rot3 is separately reported for direction: a small |q| error does not guarantee accurate χ, qy/qz or directional cuts.
- **Run Automatic Analysis (GIWAXS) fits at most 12 candidate peaks**; the window has no setting for more. Use
  `tools/gimap_agent.py auto --max-peaks N`, or `find_peaks` with `max_peaks`, for the full list.
- **The GIWAXS and GISAXS q are kinematic** (no refraction correction): near or below the film's critical angle a
  peak's q_z can be off by up to about k·sin αi (in GISAXS the Yoneda band's most, by an amount of that order), which
  the report states but does not correct, and no ± or σ of the report contains it or the uncertainty of the geometry.
- **Hermans f and the texture sentence are not corrected for the exit angle**: f weighs the bins next to the sample
  horizon most, where absorption or a shadow can lower the intensity; the report says when f depends on them.

### Updating from 0.1.0

- **Instrument profiles made from a tilted `.poni`** are read with the tilt in their geometry: results of such
  frames change (|q| by up to 6.3 % on the Xeuss example, less near the beam).
- **What is kept.** Settings, instrument profiles, recent files, the model of Fitting and saved API keys are in
  the user data folder (`%APPDATA%\GIMaP` on Windows, or the folder `GIMAP_HOME` names), as in 0.1.0, so the new
  version finds them. Projects (`.gimap`) of 0.1.0 open.
- **Portable version:** extract the new package into a new folder (manual, section 2.4) and delete the old one
  when you no longer need it; `Update_GIMaP.bat` cannot update a portable folder.
- **Source checkout:** double-click `Update_GIMaP.bat`, or run `powershell -ExecutionPolicy Bypass -File
  tools\update_source.ps1`. `requirements.txt` gained `hdf5plugin`. With `Start_GIMaP.bat` there is nothing to do;
  in an environment you set up yourself, run `python -m pip install -r requirements.txt`.
- **A model folder imported in 2D Prediction** with 0.1.0 was written into the module's `module.yaml`, which no
  longer names one. Choose it once more with **Import model...**; from then on it is kept in the user data
  folder. In a portable version the new folder has its own `module.yaml`. In a source checkout the update script
  reports such a `module.yaml` as an edit of yours: when it does not fit the new version, it waits in the stash
  and the script ends with exit code 2. When it was your only edit, delete that copy with `git stash drop` after
  importing the folder again.
- **The sidebar has a page more**: Sputter is `Ctrl+5`; 2D Prediction moved to `Ctrl+6` and Trainset Build to
  `Ctrl+7`.
- **Results that can differ from those of 0.1.0** (each is described above):
  - Fitting's curves of large or broadly distributed particles at high q (the size and orientation averages now
    converge), and its spacing disorder, now σnn/D = √(2π)·σD/D. New particles have sizes by number.
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

- **Fitting did not read `.csv` curves** (0.2.0 reads them), although Analyze exports its curves as CSV. Use
  **Send to Fitting**, or open the `<name>_fit_input.dat` that Analyze writes; `.dat` and `.txt`
  files with columns `q I [σ]` open too.
- **Geometry Calibration** ranked a wrong solution first on the public silver behenate frame. This affected
  **Auto Detect** with an empty distance field and a run with the standard selected and a 2000 mm estimate.
  Version 0.2.0 resolves both cases on this frame (User Manual 4.3).
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
