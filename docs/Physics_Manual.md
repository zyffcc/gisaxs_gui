# GIMaP Physics and Methods Reference

This reference states what GIMaP computes, from a detector frame to a reported number: every equation as it is implemented, with its symbols, units, defaults and thresholds, the assumptions behind it, what it does not model, and how its uncertainties are estimated. It is written for the scientists who measure GISAXS and GIWAXS with GIMaP, so that they can judge and report its results, and for referees, who should be able to reproduce a number from the equations and the settings in the export records. It does not explain the windows and buttons: each method names the place in the program where it is used in one short clause.

Applies to GIMaP 0.2.0.

Opened from **Help ▸ Physics and Methods**, this reference has a fixed, expandable chapter outline and
**Search this document** for headings and text. Both work offline; click a result to jump to its section.
Equations retain their original symbols, spacing and line breaks in fixed-width blocks. Scroll horizontally
inside a wide block to read the complete equation and its number; the text remains selectable and copyable.

The [User Manual](User_Manual.md) (`docs/User_Manual.md`, Help ▸ User Manual) explains how to use the program; this reference explains what the program does with the data. Both describe the same version. Where they seem to disagree, or where this text disagrees with the program, please report it.

To cite GIMaP, use the metadata in [`CITATION.cff`](../CITATION.cff) at the root of the repository, or Help ▸ How to Cite GIMaP…, and name the version you used. Methods taken from other work are cited where they are used; the full list is under References at the end.

## Contents

- [0. Notation and conventions](#0-notation-and-conventions)
  - [0.1 Symbols](#01-symbols)
  - [0.2 Units](#02-units)
  - [0.3 Conventions of this reference](#03-conventions-of-this-reference)
- [1. Detector geometry and pixel coordinates](#1-detector-geometry-and-pixel-coordinates)
  - [1.1 The detector model](#11-the-detector-model)
  - [1.2 Canonical pixel coordinates](#12-canonical-pixel-coordinates)
  - [1.3 Conventions that count pixel centres from 0](#13-conventions-that-count-pixel-centres-from-0)
  - [1.4 Where the geometry of a frame comes from](#14-where-the-geometry-of-a-frame-comes-from)
  - [1.5 Exception: Trainset's legacy convention](#15-exception-trainsets-legacy-convention)
- [2. The scattering vector in grazing incidence](#2-the-scattering-vector-in-grazing-incidence)
  - [2.1 Frames and assumptions](#21-frames-and-assumptions)
  - [2.2 Derived quantities](#22-derived-quantities)
  - [2.3 Horizon, specular and Yoneda rows](#23-horizon-specular-and-yoneda-rows)
  - [2.4 GISAXS or GIWAXS](#24-gisaxs-or-giwaxs)
  - [2.5 What the q map does not model](#25-what-the-q-map-does-not-model)
  - [2.6 Refraction in the film (optional)](#26-refraction-in-the-film-optional)
- [3. Calibration](#3-calibration)
  - [3.1 Calibrants](#31-calibrants)
  - [3.2 Ring model](#32-ring-model)
  - [3.3 What the engine refines and how](#33-what-the-engine-refines-and-how)
  - [3.4 Quality measures](#34-quality-measures)
  - [3.5 From a calibration to an instrument profile](#35-from-a-calibration-to-an-instrument-profile)
  - [3.6 pyFAI `.poni` files](#36-pyfai-poni-files)
  - [3.7 The detector tilt of a `.poni`](#37-the-detector-tilt-of-a-poni)
- [4. Reading and preparing frames](#4-reading-and-preparing-frames)
  - [4.1 Stored values and valid pixels](#41-stored-values-and-valid-pixels)
  - [4.2 NeXus and HDF5 sources](#42-nexus-and-hdf5-sources)
  - [4.3 Summing frames](#43-summing-frames)
  - [4.4 Bad pixels found in the frame](#44-bad-pixels-found-in-the-frame)
  - [4.5 Masks and value limits](#45-masks-and-value-limits)
  - [4.6 Background subtraction](#46-background-subtraction)
  - [4.7 Gap guard](#47-gap-guard)
  - [4.8 Mirror filling (GIWAXS)](#48-mirror-filling-giwaxs)
  - [4.9 The order of the steps](#49-the-order-of-the-steps)
- [5. Intensity corrections for GIWAXS](#5-intensity-corrections-for-giwaxs)
  - [5.1 Definition](#51-definition)
  - [5.2 Solid angle](#52-solid-angle)
  - [5.3 Polarisation](#53-polarisation)
  - [5.4 Absorption in the film](#54-absorption-in-the-film)
  - [5.5 Uncertainty of corrected photon counts](#55-uncertainty-of-corrected-photon-counts)
- [6. Reduction to curves and uncertainties](#6-reduction-to-curves-and-uncertainties)
  - [6.1 Principles](#61-principles)
  - [6.2 Fixed-edge bins](#62-fixed-edge-bins)
  - [6.3 Native detector lines (GISAXS)](#63-native-detector-lines-gisaxs)
  - [6.4 The three error models](#64-the-three-error-models)
  - [6.5 Merging sparse bins along q](#65-merging-sparse-bins-along-q)
  - [6.6 GIWAXS curves](#66-giwaxs-curves)
  - [6.7 GISAXS cuts](#67-gisaxs-cuts)
  - [6.8 The q map and the cake](#68-the-q-map-and-the-cake)
  - [6.9 The two halves of a signed cut](#69-the-two-halves-of-a-signed-cut)
  - [6.10 Reference-peak corrections of a series](#610-reference-peak-corrections-of-a-series)
- [7. GIWAXS automatic analysis](#7-giwaxs-automatic-analysis)
  - [7.1 Input: technique, frames and the reduced curves](#71-input-technique-frames-and-the-reduced-curves)
  - [7.2 Background: SNIP on LLS-transformed data](#72-background-snip-on-lls-transformed-data)
  - [7.3 Peak finding](#73-peak-finding)
  - [7.4 Peak profile fit and its uncertainties](#74-peak-profile-fit-and-its-uncertainties)
  - [7.5 Which rings are analysed](#75-which-rings-are-analysed)
  - [7.6 In-plane and out-of-plane sectors](#76-in-plane-and-out-of-plane-sectors)
  - [7.7 Ring orientation and the Hermans orientation factor](#77-ring-orientation-and-the-hermans-orientation-factor)
  - [7.8 Crystallite size (Scherrer)](#78-crystallite-size-scherrer)
  - [7.9 What the report states](#79-what-the-report-states)
- [8. GISAXS automatic analysis](#8-gisaxs-automatic-analysis)
  - [8.1 Yoneda band and the cuts](#81-yoneda-band-and-the-cuts)
  - [8.2 Beam-centre column from mirror symmetry](#82-beam-centre-column-from-mirror-symmetry)
  - [8.3 The halves of the cut](#83-the-halves-of-the-cut)
  - [8.4 Length scale 2π/q* of the cut](#84-length-scale-2πq-of-the-cut)
  - [8.5 The model fit and how the solution is chosen](#85-the-model-fit-and-how-the-solution-is-chosen)
  - [8.6 What the GISAXS analysis does not model](#86-what-the-gisaxs-analysis-does-not-model)
- [9. Model fitting of 1D curves (Fitting)](#9-model-fitting-of-1d-curves-fitting)
  - [9.1 Points that are fitted](#91-points-that-are-fitted)
  - [9.2 The intensity model](#92-the-intensity-model)
  - [9.3 Size distributions](#93-size-distributions)
  - [9.4 Form factors as coded](#94-form-factors-as-coded)
  - [9.5 Structure factor: one-dimensional paracrystal](#95-structure-factor-one-dimensional-paracrystal)
  - [9.6 Parameters and defaults](#96-parameters-and-defaults)
  - [9.7 Objective and the scales](#97-objective-and-the-scales)
  - [9.8 Refine, Search, goodness of fit and the errors](#98-refine-search-goodness-of-fit-and-the-errors)
  - [9.9 Find the particle shape (removed in 0.2.0)](#99-find-the-particle-shape-removed-in-020)
  - [9.10 Spheres and cylinders in a horizontal cut: the effective interface approximation](#910-spheres-and-cylinders-in-a-horizontal-cut-the-effective-interface-approximation)
  - [9.11 The three-population model](#911-the-three-population-model)
  - [9.12 fit1d (removed in 0.2.0)](#912-fit1d-removed-in-020)
  - [9.13 Predict parameters](#913-predict-parameters)
- [10. 1D Predict (many curves)](#10-1d-predict-many-curves)
  - [10.1 Input](#101-input)
  - [10.2 The method](#102-the-method)
  - [10.3 Workflow V5, removed in 0.2.0](#103-workflow-v5-removed-in-020)
- [11. Series](#11-series)
  - [11.1 The series map](#111-the-series-map)
  - [11.2 Traces and the peak in a window](#112-traces-and-the-peak-in-a-window)
  - [11.3 Batch fitting of the ring peaks of every frame](#113-batch-fitting-of-the-ring-peaks-of-every-frame)
  - [11.4 Curves made comparable](#114-curves-made-comparable)
  - [11.5 Odd frames](#115-odd-frames)
  - [11.6 Main components](#116-main-components)
  - [11.7 Stages: segmentation and its stopping rule](#117-stages-segmentation-and-its-stopping-rule)
  - [11.8 How fast: the half and 90 % frames](#118-how-fast-the-half-and-90--frames)
  - [11.9 Where the stages are used](#119-where-the-stages-are-used)
  - [11.10 Compare](#1110-compare)
  - [11.11 Limits of the series methods](#1111-limits-of-the-series-methods)
- [12. Sputter](#12-sputter)
  - [12.1 Analyze: what the main screen shows](#121-analyze-what-the-main-screen-shows)
  - [12.2 The deposition rate](#122-the-deposition-rate)
  - [12.3 The real-space picture](#123-the-real-space-picture)
- [13. Tool windows and Labs](#13-tool-windows-and-labs)
  - [13.1 XRR Series Extractor](#131-xrr-series-extractor)
  - [13.2 SEM Particle Analysis](#132-sem-particle-analysis)
  - [13.3 Format Converter](#133-format-converter)
  - [13.4 2D Prediction (Labs)](#134-2d-prediction-labs)
  - [13.5 Trainset Build (Labs)](#135-trainset-build-labs)
- [14. Provenance and reproducibility](#14-provenance-and-reproducibility)
  - [14.1 Sources of a curve on the detector](#141-sources-of-a-curve-on-the-detector)
  - [14.2 The record `<stem>_analysis.json`](#142-the-record-stem_analysisjson)
  - [14.3 Header lines of the text tables](#143-header-lines-of-the-text-tables)
  - [14.4 Pixels left out](#144-pixels-left-out)
  - [14.5 Version of the software](#145-version-of-the-software)
  - [14.6 Records of the tool windows and Labs](#146-records-of-the-tool-windows-and-labs)
  - [14.7 Checksums](#147-checksums)
  - [14.8 Seeds and deterministic runs](#148-seeds-and-deterministic-runs)
  - [14.9 What the records do not yet contain](#149-what-the-records-do-not-yet-contain)
  - [14.10 The record of a run without the window](#1410-the-record-of-a-run-without-the-window)
- [15. What GIMaP does not model](#15-what-gimap-does-not-model)
  - [15.1 Scattering physics](#151-scattering-physics)
  - [15.2 Instrument and geometry](#152-instrument-and-geometry)
  - [15.3 Statistics and uncertainties](#153-statistics-and-uncertainties)
  - [15.4 Data preparation](#154-data-preparation)
  - [15.5 Series and Sputter](#155-series-and-sputter)
  - [15.6 Tool windows and Labs](#156-tool-windows-and-labs)
  - [15.7 Records](#157-records)
- [References](#references)

## 0. Notation and conventions

### 0.1 Symbols

The table lists the symbols that occur in more than one chapter. Every other symbol is defined where it is used and holds only in that section. The same quantity has the same symbol throughout; where the program labels a quantity differently, the label is given too.

| Symbol | Unit | Meaning | Defined in |
|---|---|---|---|
| E | keV (eV in the Sputter tools) | photon energy | (1.1) |
| k | Å⁻¹ | wavenumber 2π/λ, the length of k_i and k_f | (1.2) |
| k_i, k_f | Å⁻¹ | incident and scattered wavevectors | (2.4), section 2.1 |
| α_i | ° | grazing angle of incidence between the beam and the sample surface | section 1.1 |
| α_f | ° | exit angle above the sample surface, negative below the horizon | (2.8) |
| 2θ_f | ° | in-plane exit angle | (2.9) |
| 2θ | ° | scattering angle between k_i and k_f; θ is half of it (in XRR, θ is the sample angle) | (2.12) |
| q_x, q_y, q_z | Å⁻¹ | components of the scattering vector q = k_f − k_i in the sample frame: q_x along the projection of the beam onto the surface, q_y in the surface across the beam, q_z along the surface normal | (2.5)–(2.7) |
| q | Å⁻¹ | magnitude of the scattering vector (written with bars or abs() where the sign of a signed coordinate is dropped) | (2.10) |
| χ | ° | polar angle of q from the surface normal: 0° out of plane, ±90° in the plane, sign of q_y | (2.13) |
| L | m (entered in mm) | sample–detector distance, measured along the direct beam | section 1.1 |
| L_p | m | distance from the sample to the centre of pixel p | (2.2) |
| X, Y | m | position of a pixel centre on the detector relative to the direct beam, X to the right, Y up | (1.3), (1.4) |
| x, y | px | canonical pixel coordinates: pixel corners on integers, row 0 at the top | section 1.2 |
| i, j | – | row and column index of a pixel | section 1.2 |
| x_c, y_c | px | position of the direct beam in canonical coordinates | section 1.1 |
| p_x, p_y | m (entered in µm) | pixel size along the columns (x) and the rows (y) | section 1.1 |
| I | counts per pixel | intensity: the mean of the valid pixel values of a bin or a detector line | section 6.1 |
| n | – | number of valid pixels in a bin or a detector line | section 6.1 |
| N | – | number of points or bins of a curve | where used |
| σ | unit of the quantity | standard error (one standard deviation) of a point or a parameter | section 6.4 |
| χ²_red | – | reduced χ²: the sum of squared σ-normalised residuals divided by the number of degrees of freedom | sections 7.4, 9.8, 11.3 |
| FWHM | unit of the abscissa | full width at half maximum of a peak | section 7.4 |
| d | Å | lattice spacing of a reflection, d = 2π/q | section 7.4 |
| τ | Å, reported in nm | Scherrer crystallite size (`L` in the code) | section 7.8 |
| f | – | Hermans orientation factor (the program writes "Herman's f") | section 7.7 |
| q* | Å⁻¹ (nm⁻¹ in Sputter) | position of the in-plane correlation peak of a GISAXS cut | section 8.4 |
| R | nm | radius of a particle or an island | section 9.4 |
| σ_R/R, σ_h/h, σnn/D | – | relative spreads of R and h (by the size distribution chosen, section 9.3) and relative standard deviation of the neighbour distance D (Hosemann; the w/D of fit1d, removed in 0.2.0) | sections 9.3, 9.5 |
| σ_D/D | – | the paracrystal spread of earlier Fitting, and of the quick fit and 1D Predict V5 (both removed in 0.2.0): (σnn/D)/√(2π) | (9.13) |
| P(q), S(q) | – | form factor and structure factor | sections 9.4, 9.5 |
| p | – | paracrystal factor exp(−q² σ_nn² / 2) | (9.12) |
| δ, β | – | refractive-index decrement and absorption index, n = 1 − δ + iβ | section 13.5 |
| α_c | ° | critical angle of total external reflection | section 8.6 |
| t | s | time; t_exp is an exposure time | chapters 11, 12 |

### 0.2 Units

- **Analyze, Geometry Calibration and XRR Series Extractor** use q in Å⁻¹ and d in Å. The curves Analyze exports, `<stem>_fit_input.dat` included, are in Å⁻¹.
- **Fitting works in nm⁻¹ internally.** A curve read in Å⁻¹ is multiplied by 10 (1 Å⁻¹ = 10 nm⁻¹; 1 nm = 10 Å). Sizes and distances of the models are in nm. The model fit of the GISAXS automatic analysis (Predict parameters) converts in the same way (section 8.5).
- **Angles** are in degrees in the windows and the records. Formulas use radians wherever an angle enters a trigonometric function or is a width in radians (for example β in (7.30)).
- **Detector lengths.** Pixel coordinates are in px. L, X, Y and the pixel sizes are in metres in the formulas; the windows take L in mm and pixel sizes in µm.
- **Intensities** are mean counts per pixel (`I (counts/pixel)`), not absolute intensities (section 2.5).

### 0.3 Conventions of this reference

- Equations are numbered per chapter: (5.3) is the third numbered equation of chapter 5, while "section 5.3" is a section.
- A "valid pixel" is defined in section 4.1. A "usable pixel" is valid and above the sample horizon (section 2.3).
- Each method ends with a line "Implementation:" that names the modules, with paths relative to the root of the repository. Paths under `utils/` and `modules/` are research code and prediction modules that GIMaP runs in separate processes.
- Defaults and thresholds are those of GIMaP 0.2.0.
- Numbers in the examples come from the public data in `tests/data/external` (with the geometry in its README), from the example frames in `TestSAXSdata` (the authors' own measurements), or they are invented and marked as such.
- Numerical routines named `scipy.…` are those of SciPy (Virtanen et al., 2020).

## 1. Detector geometry and pixel coordinates

Every q value in Analyze comes from one geometry object and one pixel convention. This chapter defines both. The equations of the scattering vector follow in chapter 2.

### 1.1 The detector model

GIMaP describes a measurement with seven numbers, and three more for a tilted detector. The detector is flat; without rotations it is perpendicular to the direct beam (a `.poni` tilt, section 3.7). The sample surface is tilted by the grazing angle α_i.

| Symbol | Stored field | Unit | Meaning |
|---|---|---|---|
| `p_x`, `p_y` | `pixel_size_x_m`, `pixel_size_y_m` | m | pixel pitch along the columns (x) and the rows (y) |
| `L` | `distance_m` | m | distance from the sample to the point where the direct beam meets the detector, measured along the beam |
| `x_c`, `y_c` | `beam_center_x_px`, `beam_center_y_px` | px | position of the direct (transmitted) beam, in the canonical coordinates of section 1.2 |
| `λ` | `wavelength_angstrom` | Å | X-ray wavelength |
| `α_i` | `incidence_deg` | ° | grazing angle between the beam and the sample surface; 0 means transmission |

The pixel sizes, `L` and `λ` must be positive and finite. The beam centre must be finite; it may lie outside the frame. The grazing angle must satisfy `−90° < α_i < 90°`. When a wavelength is derived from a photon energy, GIMaP uses

```text
λ [Å] = 12.398419843320026 / E [keV]                                (1.1)
k = |k_i| = |k_f| = 2π / λ          [Å⁻¹]                            (1.2)
```

The constant is hc in keV·Å (CODATA 2018; Tiesinga et al., 2021). Because lengths are in metres and λ in Å, every q in Analyze is in Å⁻¹.

Implementation: `src/gimap/shared/geometry/detector_geometry.py`, `src/gimap/shared/detector_io/radiation.py`

### 1.2 Canonical pixel coordinates

All positions on the detector use one continuous coordinate system, called canonical in the code:

- The origin is the outer corner of the first stored pixel. x grows to the right, y grows downwards, both in pixels.
- Pixel corners lie on integers. The pixel in row `i` and column `j` covers `[j, j+1] × [i, i+1]`.
- The centre of pixel `(i, j)` is therefore `(x, y) = (j + 0.5, i + 0.5)`.
- Row 0 is the top row of the displayed image. For most formats this is the first stored row; section 4.2 lists the one reader that transforms the array.
- `(x_c, y_c)` is the position of the direct beam, not the point of normal incidence (PONI). The two coincide only for an untilted detector; section 3.6 gives the conversion.

This is the pixel-corner convention of pyFAI (Ashiotis et al., 2015) and of a Matplotlib image drawn with `extent = (0, W, H, 0)`. Analyze, Geometry Calibration and XRR Series Extractor show and accept beam centres in these coordinates, so the same beam centre reads the same in the three windows. Drawn masks use the same coordinates (section 4.5). The SEM ROI uses the same corner convention, as `(left, top, right, bottom)` in pixels of the original image.

The displacement of a pixel centre from the direct beam, in metres on the detector, is

```text
X_j = (j + 0.5 − x_c) · p_x                                          (1.3)
Y_i = (y_c − (i + 0.5)) · p_y                                        (1.4)
```

X points to the right of the image and Y points up: the sign of Y is opposite to the row index. `X`, `Y` and `L` share one length unit. A test pins the convention: with `(x_c, y_c) = (3.5, 2.5)` the centre of pixel (row 2, column 3) sits at `X = Y = 0`, and the pixel one row up has `Y = +p_y`.

For a sub-image whose top-left corner is pixel `(top, left)`, the beam centre becomes `(x_c − left, y_c − top)`. Every physical pixel keeps its q.

Implementation: `src/gimap/shared/geometry/q_mapping.py`, `src/gimap/shared/geometry/detector_geometry.py`

### 1.3 Conventions that count pixel centres from 0

Some stored numbers count pixel centres from 0 (the "index" convention: the centre of the first pixel is (0, 0), row 0 at the top). They are 0.5 px smaller on both axes:

```text
x_canonical = x_index + 0.5,      y_canonical = y_index + 0.5        (1.5)
```

The conversion is done only in `src/gimap/shared/geometry/conventions.py`. The index convention is used in these places:

- **The calibration engine and the exported calibration `.json`.** The candidates' `center_x_px` and `center_y_px` are index coordinates. The file says so in `metadata.center_convention` (`"name": "index"` and a sentence that explains the 0.5 px difference). A file without that key is read as index coordinates; a file that declares `"canonical"` is converted on reading. The window converts for display.
- **XRR Series Extractor.** The extraction, the exported CSV (`roi_center_x_px`, `roi_center_y_px`) and its JSON record (`beam_center_x_px`, `beam_center_y_px`) use index coordinates, and the record names them with `center_convention`. The window converts for display.
- **Settings.** `detector.beam_center_x` and `detector.beam_center_y`, written when a calibration is applied, are index coordinates.

A third, legacy convention counts rows from the bottom: `x_canonical = x + 0.5`, `y_canonical = rows − y − 0.5`. It is read once, when a geometry stored by the former Cut & Fitting page is converted into an instrument profile.

Example (public GALAXI frame, section 1.4): the beam centre (597.1, 719.6) px shown in Analyze is written as (596.6, 719.1) in a calibration `.json`.

Implementation: `src/gimap/shared/geometry/conventions.py`, `src/gimap/features/calibration/domain/models.py`, `src/gimap/features/calibration/presentation/pixel_coordinates.py`, `src/gimap/features/xrr/application/export_record.py`, `src/gimap/features/xrr/presentation/pixel_coordinates.py`

### 1.4 Where the geometry of a frame comes from

Analyze takes the geometry of a frame from an instrument profile: a stored geometry with a detector name and a frame shape. A profile is made from a calibration (chapter 3) or entered by hand (Analyze, Geometry step).

- **Profile choice.** A profile named by the user is used as it is. Otherwise every profile gets a score: a profile with a stored frame shape that differs from the frame scores 0 and is never used; an equal shape adds 1; a detector name that equals or prefixes the other (case-insensitive, runs of spaces ignored) adds 2. The highest score wins; ties go to the most recently updated profile.
- **Overrides.** Each override replaces one field and nothing else:
  - a beam centre set in the session (only on frames of the shape it was set on);
  - the beam centre written in the file header (CBF `Beam_xy`, NeXus `beam_center_x/y`), only when the user opts in; the default is off;
  - the grazing angle α_i;
  - the distance (used by the series distance alignment, section 6.10).
- **α_i and calibration.** A new profile made from a calibration starts with `α_i = 0`, because the calibration image is a transmission pattern. Updating an existing profile from a calibration keeps its α_i.
- **Symmetry refinement.** The GISAXS action Make Left and Right Symmetric changes `x_c` only (section 6.7).

Example used in this reference. The public GISAXS frame `tests/data/external/gisaxs_galaxi/galaxi_data.tif` (GALAXI, Pilatus 1M, 981 × 1043 px) has, as documented by BornAgain: `λ = 1.34 Å`, `α_i = 0.463°`, `L = 1.730 m`, `p_x = p_y = 172 µm`, and the direct beam at (597.1, 323.4) px counted from the bottom-left. In canonical coordinates that is `(x_c, y_c) = (597.1, 1043 − 323.4) = (597.1, 719.6)`.

Implementation: `src/gimap/features/analyze/application/geometry_resolution.py`, `src/gimap/shared/geometry/instrument_profile.py`, `src/gimap/features/calibration/domain/instrument_profile.py`

### 1.5 Exception: Trainset's legacy convention

The retired Trainset research code keeps the convention with which its training data were generated. It is not an application workspace. Its detector grid is

```text
X_j = (j − x_c,index) · p_x,      Y_i = (y_c,index − i) · p_y          (1.6)
α_f = atan2(Y, L) − α_i,          2θ_f = atan2(X, L)                   (1.7)
k̂_f = (cos α_f cos 2θ_f,  cos α_f sin 2θ_f,  sin α_f)                  (1.8)
```

with the beam centre in index coordinates, pixel sizes and `L` in mm and `λ` in nm, so q comes out in nm⁻¹. The scattering vector then follows equations (2.5) to (2.7) and (2.11).

- The pixel positions (1.6) are the canonical ones of (1.3) and (1.4) after the shift (1.5).
- Only the exit-angle model (1.7) differs from the exact model of chapter 2. In the plane of incidence (`X = 0`) the two agree to 10⁻¹⁵; away from it they differ.

Implementation: `src/gimap/shared/geometry/legacy_conventions.py`

## 2. The scattering vector in grazing incidence

### 2.1 Frames and assumptions

The q map of Analyze rests on these assumptions:

1. The detector is flat. `L` is the distance from the sample to the direct beam on the detector, measured along the beam. The detector may be tilted: the rotations rot_1, rot_2, rot_3 of a `.poni` are part of the geometry (section 3.7). The formulas of this section are written for the untilted detector, perpendicular to the direct beam (all three rotations 0); for a tilted one the ray `(L, X, Y)` of a pixel is replaced by the vector (3.15), and everything after it is unchanged.
2. The sample is tilted by α_i about the horizontal axis perpendicular to the beam (the detector's X axis). The plane of incidence is vertical. It contains the beam and the detector column through the direct beam.
3. The scattering is elastic and comes from a point at the origin. There is no refraction (the vacuum, kinematic q).
4. Each pixel is represented by its centre.

Two frames are used. In the laboratory frame, `x_L` points along the beam, `y_L` along the detector X axis and `z_L` along the detector Y axis. In the sample frame, `x_s` is the projection of the beam onto the surface, `y_s` lies in the surface perpendicular to `x_s`, and `z_s` is the surface normal. The sample frame is the laboratory frame rotated by α_i about `y_L`:

```text
x_s = x_L cos α_i + z_L sin α_i,    y_s = y_L,    z_s = z_L cos α_i − x_L sin α_i     (2.1)
```

The ray from the sample to a pixel centre has the laboratory direction `(L, X, Y)/L_p`. Rotated into the sample frame:

```text
L_p = √(L² + X² + Y²)                                                   (2.2)
k̂_f = ( (L cos α_i + Y sin α_i)/L_p ,  X/L_p ,  (Y cos α_i − L sin α_i)/L_p )      (2.3)
k_i = k (cos α_i, 0, −sin α_i)                                        (2.4)
```

The scattering vector is `q = k_f − k_i = k k̂_f − k_i`:

```text
q_x = k (k̂_f,x − cos α_i)                                             (2.5)
q_y = k k̂_f,y = k X / L_p                                               (2.6)
q_z = k (k̂_f,z + sin α_i)                                             (2.7)
```

The exit angles are defined by `k̂_f = (cos α_f cos 2θ_f, cos α_f sin 2θ_f, sin α_f)`:

```text
sin α_f = (Y cos α_i − L sin α_i) / L_p                                 (2.8)
tan 2θ_f = X / (L cos α_i + Y sin α_i)                                (2.9)
```

The code computes the three components of (2.3) directly; it never forms 2θ_f.

Signs: `q_y > 0` to the right of the direct beam on the image, `q_z > 0` above the horizon. At the direct beam `q = 0` and `α_f = −α_i`. The tests pin three points: the direct beam gives `q = (0, 0, 0)`; the horizon (`X = 0`, `Y = L tan α_i`) gives `q = k (1 − cos α_i, 0, sin α_i)`; the specular beam (`X = 0`, `Y = L tan 2α_i`) gives `q = (0, 0, 2k sin α_i)`. For every pixel `|k_f| = k` to 10⁻¹³ Å⁻¹. With `α_i = 0` the map reduces to transmission rings.

This exact flat-detector model is the only q mapping that Analyze uses. Its derivation is standard (Renaud et al., 2009; Hexemer & Müller-Buschbaum, 2015).

Implementation: `src/gimap/shared/geometry/q_mapping.py`

### 2.2 Derived quantities

```text
|q| = √(q_x² + q_y² + q_z²)                                            (2.10)
q_∥ = sign(q_y) · √(q_x² + q_y²)                                       (2.11)
2θ = 2 arcsin( |q| λ / 4π ),    cos 2θ = L / L_p                         (2.12)
χ = atan2(q_∥, q_z)                       (GIWAXS, in degrees)         (2.13)
```

- **q_∥** (labelled q∥ or qr) is the full in-plane magnitude, including `q_x`, with the sign of `q_y`. The sign is applied with `copysign`, so a pixel with `q_y = +0.0` gets a positive q_∥.
- **|q| and 2θ do not depend on α_i.** A rotation keeps the angle between `k_i` and `k_f`, so 2θ from (2.12) is the true scattering angle. In the laboratory frame `cos 2θ = k̂_i · k̂_f = L/L_p`. The optional 2θ axis of the GIWAXS I(q) curves converts the q of each point with (2.12); the bins stay uniform in q.
- **χ** (GIWAXS polar angle): 0° along the surface normal (out of plane), ±90° in the sample plane, with the sign of `q_y`. Above the horizon `q_z > 0`, so χ lies between −90° and +90°. Because `q_∥² + q_z² = |q|²`, a point at (|q|, χ) has `q_∥ = |q| sin χ` and `q_z = |q| cos χ` exactly.
- **Missing wedge.** On the detector column through the beam (`X = 0`) `q_y = 0` but `q_x ≠ 0`, so χ is not zero there. With the public P08 geometry (section 2.4), the beam column reaches χ ≈ 2.2° at |q| = 0.73 Å⁻¹, 5.7° at 1.83 Å⁻¹ and 9.0° at 2.87 Å⁻¹. Smaller χ at these |q| is not measured by a flat detector (Baker et al., 2010).
- A second azimuth exists in the code for transmission data, `atan2(Y, X)` (0° to the right, 90° up). No part of the program uses it; every χ shown or exported by Analyze is the χ of (2.13).

Implementation: `src/gimap/shared/geometry/q_mapping.py`, `src/gimap/features/analyze/domain/giwaxs.py`

### 2.3 Horizon, specular and Yoneda rows

In the plane of incidence (`X = 0`), equation (2.8) reduces to `Y = L tan(α_f + α_i)`. The row (canonical, continuous) of a given exit angle on the beam column is therefore

```text
row(α_f) = y_c − L tan(α_f + α_i) / p_y                               (2.14)
row_horizon  = y_c − L tan α_i / p_y          (α_f = 0)               (2.15)
row_specular = y_c − L tan 2α_i / p_y         (α_f = α_i)             (2.16)
```

and the exit angle at a row on the beam column is `α_f = atan2((y_c − row) p_y, L) − α_i`, which is (2.8) at `X = 0`.

- **The horizon is a straight row.** From (2.8), `sin α_f = 0` exactly when `Y = L tan α_i`, whatever X. On a flat detector perpendicular to the beam the horizon is therefore one detector row.
- **Above the horizon.** A pixel counts as above the horizon when `q_z ≥ k sin α_i` (evaluated in single precision). This is the same as `α_f ≥ 0` and as `Y ≥ L tan α_i`. Every GIWAXS curve, the GIWAXS and GISAXS q maps and the cake use only these pixels. The GISAXS vertical cut uses only rows entirely above the horizon; the horizontal cut uses the rows of its band, wherever they are (section 6.7).
- **α_i = 0.** The horizon then equals the beam-centre row. The GISAXS reduction warns: "Incidence angle is 0°: the sample horizon equals the beam centre row."

Example (GALAXI geometry of section 1.4): `row_horizon` = 638.3 and `row_specular` = 557.0. The Yoneda band of this frame lies at row 607.5, which (2.14) turns into `α_f` = 0.18°.

Implementation: `src/gimap/shared/geometry/detector_geometry.py`, `src/gimap/features/analyze/domain/gisaxs.py`, `src/gimap/features/analyze/domain/giwaxs.py`

### 2.4 GISAXS or GIWAXS

When a frame has a geometry, Analyze proposes the technique from the largest scattering angle on the detector. For a flat detector that angle is reached at a corner, so the four pixel corners of the frame are tested:

```text
2θ_max = max over (x, y) ∈ {0, W} × {0, H} of atan2( √( ((x − x_c) p_x)² + ((y − y_c) p_y)² ), L )      (2.17)
```

where `W` and `H` are the numbers of columns and rows. The frame is GIWAXS when `2θ_max > 20°` and GISAXS otherwise. The user can force either mode; Analyze then points out when the forced mode disagrees with this rule. Without a geometry nothing is reduced. `tools/gimap_agent.py` uses the same rule (`--technique` forces it).

A powder calibrant measured in transmission at a SAXS distance is GISAXS by this rule. The automatic analysis (not Analyze's mode) therefore looks for rings when the rule said GISAXS and no incidence angle is known (none given, noted or set, and 0° in the geometry): the valid pixels are averaged in 1-pixel rings about the beam centre and in 24 sectors of 15°; a maximum of the ring average that exceeds its 61-pixel running median by 6 robust σ and by 30 % of that median, and is at most 25 pixels wide at half height, is a ring; it is complete when the detector covers at least 90° of azimuth at its radius and at least 80 % of the covered sectors show 30 % of its height above their own background. With three or more complete rings the frame is reported as a transmission frame of a calibrant or powder: the cuts are written and no particle model is fitted (`analyze/domain/ring_frame.py`, `assistant/application/ring_frames.py`). The public silver behenate frame has five such rings, the public GALAXI GISAXS frame none.

Examples from the public data: the GALAXI frame reaches `2θ_max` = 5.3° (GISAXS). The P08 GIWAXS frame (`giwaxs_p08_mapi`, PerkinElmer 2048 × 2048 px, `p` = 200 µm, `λ` = 0.6888 Å, `L` = 811.8 mm, direct beam (545.5, 1826.9) px from its `.poni`, `α_i` = 0.075°) reaches 30.2° (GIWAXS).

The choice decides the reduction (chapter 6). Mirror filling and the intensity corrections are applied only to GIWAXS frames.

Implementation: `src/gimap/features/analyze/domain/classification.py`

### 2.5 What the q map does not model

The following effects are not in the q map, nor anywhere else in Analyze:

- **Refraction and the distorted-wave Born approximation (DWBA).** `q_z` is the kinematic vacuum value unless the film's critical angle is given (section 2.6). No reflected-wave terms of the DWBA are computed.
- **Detector tilt and rotation.** A `.poni` tilt (rot1, rot2, rot3) is part of the geometry and enters q, χ, the solid angle and the polarisation factor (section 3.7). The calibration refines no tilt (section 3.3). The straight horizon row below assumes an untilted detector.
- **Sample azimuth and roll.** The sample is assumed to be tilted only about the horizontal axis perpendicular to the beam. A rotation about the surface normal or about the beam is not described.
- **Parallax and sensor thickness.** A pixel is its centre; there is no depth of interaction and no pixel splitting.
- **Flat field and efficiency.** No flat-field, detector-efficiency, air or window absorption correction is applied.
- **Absolute intensity.** Intensities are mean counts per pixel. They are not scaled by a monitor, the transmission, the exposure time or the sample volume. They are divided by a geometric factor only when the GIWAXS intensity corrections are on (chapter 5); a series can be normalised to a reference peak (section 6.10).
- **Beam properties.** Divergence, energy bandwidth and footprint are not modelled.

The flat, perpendicular detector is also assumed by the straight horizon (section 2.3), by mirror filling (section 4.8) and by the solid-angle and polarisation factors (chapter 5).

Implementation: `src/gimap/shared/geometry/q_mapping.py` (the only q mapping Analyze uses)

### 2.6 Refraction in the film (optional)

With **Refraction in the film, αc** ticked (**Corrections**, folded; `--critical-angle-deg` of
`tools/gimap_agent.py auto`), q_z is the one inside a film of critical angle α_c:

```text
q_z = k ( Re√(sin²α_f − sin²α_c) + Re√(sin²α_i − sin²α_c) )                                       (2.18)
```

α_f is the exit angle of the ray in vacuum (2.8). The in-plane part of the wavevector is kept at the surface, so q_x,
q_y and q_∥ do not change; |q| of (2.10) and χ of (2.13) use the q_z of (2.18). Off (the default), q_z is (2.7).

- Below α_c a wave in the film is evanescent and its term is 0. q_z then no longer tells α_f, so pixels with
  α_f < α_c are left out of the q maps, of I(q), I(χ), the sectors and boxes of GIWAXS, and the vertical GISAXS cut
  stops at the row of α_c. The horizontal cut I(q_y) does not change. With α_i < α_c the incident term is 0.
- Hand example (tested): λ = 1 Å, α_i = 0.2°, α_c = 0.15°, α_f = 1.0°: q_z = 2π (0.0172549 + 0.0023088) =
  0.122923 Å⁻¹, against the kinematic 2π (sin 1° + sin 0.2°) = 0.131583 Å⁻¹.
- It describes one homogeneous film and the transmitted waves only. The reflected-wave terms of the DWBA, a
  substrate and a depth profile are not included, and α_c is taken as given (for example from the film's density, or
  from the Yoneda band, section 2.3). An α_c too large or too small moves q_z most near α_f ≈ α_c.
- The value is a setting of the measurement, kept with the Analyze settings, not in the instrument profile. The record
  gives it as `geometry.critical_angle_deg` (0: kinematic), `geometry_convention` says what q_z is, the CSV headers of
  the command line name it, and the automatic report's *q* decision reads *q_z corrected for refraction in the film
  (αc = …)* (sections 7.9, 8.6).

Implementation: `src/gimap/shared/geometry/q_mapping.py` (`refracted_qz`), `src/gimap/features/analyze/domain/giwaxs.py`,
`src/gimap/features/analyze/domain/gisaxs.py`

## 3. Calibration

Geometry Calibration (Tools menu) finds the beam centre and the distance from a powder pattern of a standard. The energy and the pixel size are inputs and are not refined. The same engine runs without the window in Run Automatic Analysis and in `tools/gimap_agent.py`.

### 3.1 Calibrants

| Key | Standard | Values in the code | Lines used |
|---|---|---|---|
| `agbh` | silver behenate | lamellar `d_001` = 58.38 Å | orders n = 1 … 20 |
| `lab6` | lanthanum hexaboride | cubic, `a` = 4.1569 Å | 20 lines, `N = h² + k² + l²` ∈ {1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22} |
| `ceo2` | cerium dioxide | fluorite, `a` = 5.41165 Å | 17 lines with h, k, l all odd or all even, `N` ∈ {3, 4, 8, 11, 12, 16, 19, 20, 24, 27, 32, 35, 36, 40, 43, 44, 48} |
| `lab6_ceo2` | mixture | both sets | the union of both line lists |

```text
q_n = 2π n / d_001                     (AgBH, n = 1 … 20)              (3.1)
q_hkl = 2π √(h² + k² + l²) / a          (cubic standards)               (3.2)
d_hkl = a / √(h² + k² + l²)                                            (3.3)
```

- For LaB6, N = 7 and 15 are absent because they are not sums of three squares. The first spacings are 4.1569 Å (100), 2.9394 Å (110), 2.4000 Å (111) and 2.0785 Å (200); q runs from 1.5115 to 7.0896 Å⁻¹.
- For CeO2 the first spacings are 3.1244 Å (111), 2.7058 Å (200), 1.9133 Å (220) and 1.6317 Å (311); q runs from 2.0110 to 8.0440 Å⁻¹.
- For AgBH, q runs from 0.10763 Å⁻¹ (n = 1) to 2.1525 Å⁻¹ (n = 20). The code stores relative intensities 1/n, but the engine does not use them.
- With the standard set to Auto, all four line sets are tried.
- The values are fixed; temperature is not taken into account. The code cites no source. The AgBH spacing equals the 58.380(3) Å of Huang et al. (1993). The LaB6 value differs from the certified lattice parameter of NIST SRM 660c (4.156826 Å at 22.5 °C; Black et al., 2020) by 1.8 × 10⁻⁵ relative. The CeO₂ value differs from the certified lattice parameter of NIST SRM 674b (5.41153 Å at 22.5 °C; NIST, 2017) by 2.2 × 10⁻⁵ relative; the code does not state its source.

Implementation: `src/gimap/features/calibration/domain/standards.py`

### 3.2 Ring model

The rings of a powder standard are circles on the flat detector perpendicular to the beam, centred on the direct beam:

```text
2θ = 2 arcsin( q λ / 4π )                                              (3.4)
r = L tan 2θ                    [m]                                     (3.5)
r_px = L tan 2θ / p̄,      p̄ = (p_x + p_y) / 2                           (3.6)
```

The search for candidates (ring detection and matching) uses radii in pixels with the mean pixel size `p̄`, i.e. square pixels. The joint refinement (section 3.3, step 6) uses `p_x` and `p_y` separately. The overlay in the window draws each ring as an ellipse of width `2r/p_x` and height `2r/p_y`. A line with `qλ/4π > 1` cannot be reached and is refused.

Implementation: `src/gimap/features/calibration/domain/geometry_model.py`, `src/gimap/features/calibration/domain/ring_geometry.py`

### 3.3 What the engine refines and how

Inputs: the image, the energy (λ from (1.1)), the pixel size (from the header or entered), the standard or Auto, an optional estimated distance, and a distance range of 30 to 10 000 mm. Internally all positions are index coordinates (section 1.3), and so are the candidates of the result and of the exported `.json`; the window, Apply (section 3.5) and the summary of the automatic analysis shift them by +0.5 px with (1.5).

1. **Preprocessing.** Valid pixels are finite, not masked by the reader and above 0, so zero-filled gaps and apertures are left out. Percentiles and the maximum are taken on a sample of about 2.5 × 10⁵ valid pixels. When the maximum exceeds `max(20 P_99.95, 10⁶)`, the pixels at that value are dropped. Values are clipped to `[0, P_99.9]` and transformed with `ln(1 + I)`. By default a smooth background is subtracted: a Gaussian of σ = `max(6, min(H, W)/90)` px, normalised by the smoothed valid mask so that gaps do not bleed in; negative results are set to 0.
2. **Beam-centre proposals.** Up to 14 proposals, at least 3 px apart, each with a support between 0 and 1. They come from the direct beam and beam stop along the bright beam column, from voting of intensity-gradient lines of partial arcs (which can place a centre outside the detector), from the image centre, from the header centre (support 0.70), from 180° symmetry by phase correlation (support 0.20), and from a coarse-to-fine search for the sharpest radial profile.
3. **Ring detection for each proposal.** The image is resampled onto 360 azimuths × 1 px radii. The radial profile is the 60th percentile over azimuth, which is robust against streaks and gaps (the plain radial mean when OpenCV is missing). The profile is smoothed with a Gaussian of σ = 1.2 px. Peaks need a prominence of at least `max(2.3 MAD, 0.08 (P_85 − median), 10⁻⁵)`, an azimuthal coverage of at least 1.5 % and a radius of at least 40 px. Peak radii stay on the 1 px grid; they are not interpolated.
4. **Distance hypotheses and matching.** Each of the first 35 peaks paired with each line of the standard gives a distance by inverting (3.6); distances inside the range are clustered within 0.8 %. At each distance, predicted and observed radii are paired greedily within a tolerance of `max(3 px, 0.018 r, 0.6 w)` (w: peak width). With the pairs fixed, L is solved by one-parameter linear least squares, equation (3.7).
5. **Ranking.** A weighted score rewards matched rings (each counted in proportion to its prominence up to 0.1 in the log-scaled profile, about 10 % contrast, and whole above it: chance coincidences with noise-level peaks do not outweigh fewer clear rings), azimuthal coverage, valid area, centre support, an unbroken run of orders and prominence, and a Gaussian prior of 20 % width around an estimated distance. It penalises the residual, inconsistent per-ring distances, unexplained inner peaks and (for AgBH) a missing first order.
6. **Joint refinement.** For up to 6 groups of proposals with up to 3 distinct distances each, the beam centre and L are refined together. Ring points are the valid pixels within `h = clip(0.012 r, 3, 10)` px of each predicted ring whose preprocessed signal reaches the 70th percentile of the annulus, at most 800 per ring. The residual (3.8) is minimised with `scipy.optimize.least_squares` (soft-L1 loss, f_scale 2 px, at most 80 evaluations). The centre may move by 5 px (direct-beam proposal), by 40 to 140 px (arc voting) or by 12 to 60 px (other proposals); L may move by ±15 %. Only `x_c`, `y_c` and `L` are free.
7. **Re-matching.** At the refined centre the rings are detected again and matched at the refined L, and L is solved again with (3.7). **The distance and the residual that are reported come from this step**, not from the joint fit. On a synthetic AgBH frame (true L = 1600.0 mm, 4 rings) the joint fit gave 1599.96 mm and the reported value was 1599.85 mm; the centre was within 0.03 px of the true one.
8. **Result.** The best three candidates are kept, plus the best candidate of each other standard tried, up to five. The record holds the SHA-256, size and modification time of the image, the energy, λ, the pixel sizes, every candidate and the image shape. With fewer than two matched rings a warning says that the distance is ambiguous.

```text
L = Σ_m u_m r_m / Σ_m u_m²,     u_m = tan 2θ_m / p̄     (r_m: observed radii in px)     (3.7)
ρ_p = [ √( ((x_p − x_c) p_x)² + ((y_p − y_c) p_y)² ) − L tan 2θ(q_p) ] / p̄             (3.8)
```

In (3.8), `(x_p, y_p)` are the integer index coordinates of ring pixel p, `q_p` the line it was assigned to, and ρ_p is in pixels.

What is not refined: the wavelength, the pixel sizes, and any detector tilt or rotation. The field `detector_rotation_deg` of a candidate stays 0.

**Uncertainty.** The points along one ring are not independent: when a ring is slightly off, all of them move
together. The 1σ of x_c, y_c and L is therefore the scatter between the rings: with three or more matched rings the
fit is repeated with each ring left out, and σ² = (n − 1)/n · Σ (θ_i − θ̄)² (jackknife). With two rings only the
least-squares covariance s² (JᵀJ)⁻¹ is left, a lower bound. On a synthetic AgBh image with five rings σ is 0.02 px
for the centre and 0.04 mm at 1 m. A ring matched wrongly, or a bias common to all rings, is not in σ. The values are
in the calibration record (`center_x_sigma_px`, `center_y_sigma_px`, `distance_sigma_mm`, `uncertainty_method`), in
the tooltips of the result, and in the `source` of the instrument profile, so they reach every `<stem>_analysis.json`.

Implementation: `src/gimap/features/calibration/domain/engine.py`, `src/gimap/features/calibration/domain/preprocessing.py`, `src/gimap/features/calibration/domain/center_estimator.py`, `src/gimap/features/calibration/domain/radial_profile.py`, `src/gimap/features/calibration/domain/peak_detector.py`, `src/gimap/features/calibration/domain/peak_matcher.py`, `src/gimap/features/calibration/domain/candidate_ranker.py`, `src/gimap/features/calibration/domain/optimizer.py`

### 3.4 Quality measures

GIMaP uses two measures of a calibration.

**Ring residual (the Geometry Calibration window).** `rms_residual_px` is the RMS of observed minus predicted ring radius, in pixels, at the reported centre and distance (step 7). `distance_consistency` is the standard deviation divided by the mean of the per-ring distances. `azimuthal_coverage` is the mean coverage of the matched rings. The window shows a confidence:

| Confidence | Condition |
|---|---|
| High | at least 4 rings, RMS ≤ 2 px and coverage ≥ 0.18 |
| Medium | at least 2 rings and RMS ≤ 6 px; or at least 4 rings, centre support ≥ 0.8, RMS ≤ 8 px and coverage ≥ 0.05; or at least 5 rings, centre support ≥ 0.8, RMS ≤ 10 px and coverage ≥ 0.015 |
| Low | otherwise |

**Line positions (Run Automatic Analysis, `tools/gimap_agent.py`).** The ring residual depends on how many rings are fitted and on what the flat model cannot describe, such as a tilt. The automatic analysis therefore also checks where the standard's lines land in q:

1. The calibration image is integrated with the candidate geometry, canonical centre and flat detector: `q = (4π/λ) sin(½ atan2(r, L))`, with r from the pixel centres. Bins are 0.0015 Å⁻¹ wide; bins with at least 20 pixels count.
2. For every line with at least 12 such bins within ±0.025 Å⁻¹, a Gaussian on a linear background is fitted.
3. A line is accepted when the amplitude exceeds 5 times the noise (the standard deviation of the points more than 3 widths from the centre), the fitted centre lies within 0.02 Å⁻¹ of the line, and the Gaussian σ lies between 0.0008 and 0.015 Å⁻¹.

```text
ε = (1/M) Σ_m |q_m,fit − q_m,line| / q_m,line          (M accepted lines)          (3.9)
```

| Verdict | Condition |
|---|---|
| good | ε ≤ 0.2 % with at least 3 lines |
| usable | ε ≤ 0.5 % with at least 3 lines |
| doubtful | ε larger |
| good (no line check possible) | at least 3 rings, RMS ≤ 1.5 px and confidence High or Medium |
| unreliable (no line check possible) | fewer than 2 rings |

Among the candidates with at least 3 accepted lines, the one with the smallest ε is chosen, even when the engine ranked another first; the record says which rule chose (`chosen_by`).

"Good" means that the geometry agrees with its own calibration image. It does not show that the calibration belongs to the sample data. A known limit is recorded in the tests: on the partial arcs of the public pyFAI AgBH frame the fitted centre lies about 3 px off and the distance 0.45 % off.

Implementation: `src/gimap/features/calibration/domain/candidate_ranker.py`, `src/gimap/features/calibration/domain/line_check.py`, `src/gimap/features/calibration/application/headless.py`, `src/gimap/features/assistant/domain/calibration_quality.py`

### 3.5 From a calibration to an instrument profile

Apply in Geometry Calibration turns the selected candidate into an instrument profile:

- the beam centre from index to canonical coordinates (1.5), the distance from mm to m, λ from the energy, and the two pixel sizes;
- `α_i = 0` for a new profile; an updated profile keeps its α_i;
- GIMaP asks before an update moves the beam centre by more than 10 px or the distance by more than 5 %;
- the profile's `source` names the standard, the time and the image. A rotation above 0.05° would be noted there as not modelled; since the engine never sets a rotation, this happens only for an imported record that carries one.

Implementation: `src/gimap/features/calibration/domain/instrument_profile.py`, `src/gimap/features/calibration/domain/manual_refinement.py`

### 3.6 pyFAI `.poni` files

Run Automatic Analysis and `tools/gimap_agent.py` accept a pyFAI `.poni` file as the geometry. Its PONI is converted to the direct beam with the relations of pyFAI's `getFit2D` (the Fit2D geometry; Hammersley, 2016):

```text
x_c = (poni_2 − L_poni tan rot_1) / p_2                                     (3.10)
y_c = (poni_1 + L_poni tan rot_2 / cos rot_1) / p_1                          (3.11)
L = L_poni / (cos rot_1 cos rot_2)                                           (3.12)
tilt = arccos(cos rot_1 cos rot_2)                                      (3.13)
```

- `L_poni` is the `.poni` Distance (m); `poni_1`, `poni_2` (m) and `p_1`, `p_2` (pixel sizes) refer to the slow axis (rows, so `p_y = p_1`) and the fast axis (columns, `p_x = p_2`). They are taken as measured from the corner of pixel (0, 0) of the stored image; an `orientation` entry of newer `.poni` files is not read.
- The pixel sizes come from `PixelSize1/2`, from `detector_config`, or from a known detector name (Pilatus 172 µm, Eiger 75 µm, PerkinElmer 200 µm, Lambda 55 µm, MAR345 100 µm, Jungfrau 75 µm). The wavelength is converted from m to Å.
- The rotations go into the geometry with the direct beam and the distance (section 3.7); the instrument profile also keeps the file's distance, PONI, rotations and pixel sizes (`tilt` in `instrument_profiles.json`; `dropped_tilt` up to 0.2.0, still read). Above 0.5° the file's inspection notes the tilt.
- A `.poni` is applied as it is: its pixel size and frame size are not checked against the data.

Examples from the public data: the P08 file gives (545.46, 1826.95) px, `L` = 811.77 mm, tilt 0.31°; the pyFAI AgBh file gives (179.98, 263.86) px, `L` = 1583.31 mm, tilt 0.57° (note); the Xeuss file gives (7.05, 589.49) px, `L` = 147.58 mm, tilt 6.06° (note).

Implementation: `src/gimap/features/assistant/domain/calibration_files.py`

### 3.7 The detector tilt of a `.poni`

A `.poni` tilt is part of the geometry: the rotations rot_1, rot_2, rot_3 are stored with it (`rot1_rad`,
`rot2_rad`, `rot3_rad`) and every pixel is placed where pyFAI places it. pyFAI's vector from the sample to the
centre of pixel (row i, column j), with `p_1 = (i + ½) p_1 − poni_1`, `p_2 = (j + ½) p_2 − poni_2` (the pixel sizes
`p_1`, `p_2` of the file), `c_k = cos rot_k` and `s_k = sin rot_k`, is

```text
t_1 = p_1 c_2 c_3 + p_2 (c_3 s_1 s_2 − c_1 s_3) − L_poni (c_1 c_3 s_2 + s_1 s_3)
t_2 = p_1 c_2 s_3 + p_2 (c_1 c_3 + s_1 s_2 s_3) − L_poni (−c_3 s_1 + c_1 s_2 s_3)            (3.14)
t_3 = p_1 s_2 − p_2 c_2 s_1 + L_poni c_1 c_2
```

with `t_3` along the beam. The vector is linear in the pixel position and vanishes across the beam at the direct
beam (3.10)–(3.11), at distance L (3.12). GIMaP therefore writes it relative to the direct beam:

```text
(along the beam, right, up) = L ê_beam + X ĉ + Y r̂                                          (3.15)
ĉ = M ê_2,   r̂ = −M ê_1,   M = R_3 R_2 R_1,   laboratory axes (t_3, t_2, −t_1)
```

X and Y are the displacements (1.3), (1.4) from the direct beam, ĉ the direction of the detector columns and r̂
that of its rows upward. Untilted, (3.15) is `(L, X, Y)`. Every quantity of a pixel comes from this vector: q of
(2.8), 2θ and χ, the exit angle of a row, the solid angle `(L / |t|)³` and the polarisation factor (section 5).

- A beam centre moved later (the GISAXS symmetry axis, a centre set by hand) moves the detector in its own plane;
  the rotations stay.
- A test compares (3.15) with (3.14) for several sets of rotations (equal to 10⁻¹⁷ m) and, on the public Xeuss
  frame (tilt 6.06°), |q| of every pixel with pyFAI's `4π sin θ / λ` (equal to 10⁻⁹).
- `tools/benchmarks/pyfai_comparison.py` compares with pyFAI itself (2026.9) on the four public frames, tilts 0 to
  6.06°: |q|, 2θ, and q∥ and q_z against pyFAI's fiber units `qip` and `qoop` (sample orientation 4) agree to
  10⁻¹³; the solid angle agrees to 10⁻⁷ up to one constant (pyFAI normalises to the PONI, GIMaP to the direct
  beam), the polarisation factor exactly.
- `tools/benchmarks/pygid_comparison.py` compares with pygid (0.2.20) on the two pygid example frames (P08, Xeuss):
  q∥ and q_z of every pixel agree to 10⁻¹⁵ once GIMaP's direct beam is moved by half a pixel along both axes. pygid
  places pixel i at i·p from the PONI, pyFAI and GIMaP at its centre, (i + ½)·p; as given, the two differ by up to
  1.1 × 10⁻³ Å⁻¹ (P08) and 2.4 × 10⁻³ Å⁻¹ (Xeuss). The ring centres of pygid's polar map agree with GIMaP's I(q) to
  7 × 10⁻⁴ Å⁻¹ on P08, a third of the map's q step. The polar maps published with the examples (pygid 0.1.16) put
  the rings 1.6 to 1.7 × 10⁻³ Å⁻¹ lower; their q grid differs from that of pygid 0.2.20 by up to 1.4 × 10⁻³ Å⁻¹.
- What the tilt changes: leaving it out puts |q| off by up to 0.20 % on the P08 file (tilt 0.31°), 0.08 % on the
  AgBh file (0.57°) and 6.3 % on the Xeuss file (6.06°, at the corner farthest from the beam). Rot3 alone leaves |q|
  unchanged and turns χ by rot_3.
- The automatic analysis lists the tilt under *Decisions* (`detector tilt: 6.06° in the geometry`), and the
  Geometry step and its summary say `detector tilt 6.06°`. The record has the rotations in `geometry`.
- Profiles saved by GIMaP 0.1 kept the tilt beside a flat geometry (`dropped_tilt`). They are read with the tilt
  in the geometry. Geometry Calibration refines no tilt (section 3.3): a profile it updates is untilted.
- What still assumes an untilted detector: the straight horizon row of section 2.5, the mirror filling of
  section 4.8, the line check of the calibration and the XRR geometry (section 13.1). For tilts of a few degrees
  these are small near the beam and grow towards the edges.

Implementation: `src/gimap/shared/geometry/detector_geometry.py` (`in_plane_axes`),
`src/gimap/shared/geometry/q_mapping.py` (`lab_rays`), `src/gimap/shared/geometry/instrument_profile.py`,
`src/gimap/shared/geometry/detector_tilt.py`

## 4. Reading and preparing frames

### 4.1 Stored values and valid pixels

The readers convert each frame to single precision for computation but do not rescale or offset it. They record the stored data type in `metadata["stored_dtype"]` (TIFF, EDF, CBF and NeXus/HDF5; a frame without this entry is treated as integer counts). Each reader also returns a mask of pixels that carry no data:

- **CBF:** non-finite or negative values.
- **EDF:** non-finite values, and `|I − Dummy| ≤ DDummy` when the header gives a non-zero `Dummy`.
- **TIFF:** non-finite values only; negative codes are left to Analyze.
- **NeXus/HDF5:** the `pixel_mask` bits with a defined invalid meaning, bits 0, 1, 2, 3, 4, 6 and 8 (`0x15F`: gap, dead, under-responding, over-responding, noisy, problematic cluster, user mask); undefined bits such as 10 and the virtual-pixel bit 31 are kept. Also the largest value of an unsigned integer type (the detector's "no data" value), non-finite values, and parts of a stitched canvas that no module covers. These pixels are NaN in the data.

Analyze then decides which pixels are valid:

```text
valid(p) = finite(I_p) ∧ ¬mask_reader(p) ∧ (I_p ≥ 0 unless the stored type is floating point)     (4.1)
```

- On an integer counting detector, negative values are gap or defect codes (Pilatus writes −1 and −2) and are left out.
- On a floating-point frame (dark- or background-subtracted), negative values are data and are kept.
- The rule follows the stored type, not the physical meaning of the values (section 6.4). A floating-point type counts as such in either byte order (`float32`, or `>f4` as h5py and fabio report a big-endian one).

Examples: the GALAXI frame is stored as int32 and has between 5 000 and 10 000 gap pixels at −1. The P08 frame is stored as float32 (dark-subtracted); about 9.8 × 10⁵ of its pixels (23 %) are negative and are kept as data.

Implementation: `src/gimap/features/analyze/domain/validity.py`, `src/gimap/shared/detector_io/loading.py`, `src/gimap/shared/detector_io/nexus_reading.py`

### 4.2 NeXus and HDF5 sources

NeXus files follow the NeXus format (Könnecke et al., 2015). The content of the files decides what a source is; a file name only proposes. A source is one of:

- `single_frame`: one file, one frame;
- `stack`: one file holding the whole series;
- `module_set`: one file per detector module, each with all frames (P03 Lambda 9M). Files named `…_mNN` form a set only when all hold the same dataset and module shape and carry pairwise distinct module positions (`<detector group>/translation/distance`);
- `chunked_master`: a master file whose frames are in data files (Dectris `data_NNNNNN` links, a virtual dataset, an external link);
- `no_image`: no detector image.

Every frame of a source is read by its index.

**Orientation.** A file written by the Lambda server (it has a module position, a `layout` such as `1556X516`, or "lambda" in its description, type or local name) gets the transform GIMaP has always applied to P03 Lambda data:

1. a module stored the other way round from its declared layout (columns × rows) is transposed first;
2. the modules are placed at their translations on a canvas whose origin is translation 0 (the canvas is shifted only when a translation is negative, so that the smallest one becomes 0); uncovered parts are invalid;
3. the canvas is transposed;
4. the canvas is flipped vertically.

Every other NeXus/HDF5 image is read as stored, row 0 at the top. `metadata["transformations"]` lists what was done, for example `stitched using P03 translations`, `canvas transpose`, `vertical flip`. A header beam centre `(fast, slow)` of a single Lambda module becomes `(slow, rows − fast)` after the transform; a stitched set has no single header centre.

Implementation: `src/gimap/shared/detector_io/nexus_layout.py`, `src/gimap/shared/detector_io/nexus_modules.py`, `src/gimap/shared/detector_io/nexus_reading.py`

### 4.3 Summing frames

Frames listed with Sum N are added pixel by pixel:

```text
I_sum(p) = Σ_k I_k(p)    if p is valid in every frame k;   otherwise NaN and invalid      (4.2)
```

The running total is kept in double precision and stored in single precision. A sum of photon counts is still photon counts, so the Poisson model of section 6.4 applies to it.

Implementation: `src/gimap/features/analyze/application/frame_preprocessing.py`, `src/gimap/features/analyze/domain/corrections.py`

### 4.4 Bad pixels found in the frame

Leave out hot and dead pixels (Mask step; setting `analyze.bad_pixels`) is on by default. The search runs on the frame as loaded (the sum, for summed frames), among its valid pixels, before masks and background. Let `b` be the largest value among the valid pixels of the 8-neighbourhood.

```text
hot:   I > 5 · max(b, 0) + σ    and    I − b > 10 σ                                   (4.3)
       σ = √(max(b, 0) + 1)                       (integer frames)
       σ = 1.4826 · median|Δ − median(Δ)|,   Δ = I − S_8 / 8      (floating frames)     (4.4)
dead:  I ≤ 0    and    every valid neighbour ≥ 20 counts         (integer frames only)  (4.5)
```

- A pixel needs at least one valid neighbour to be judged. In (4.4), `S_8` is the sum of the valid neighbours (invalid ones count as 0), and the robust noise is taken over the whole frame (on a sample of about 2 × 10⁵ pixels). On a floating frame with fewer than 16 valid pixels, or with a zero spread, no hot pixel is flagged.
- For the dead test, a true zero next to neighbours of at least 20 counts has a Poisson probability of at most e⁻²⁰.
- A separable 3 × 3 sum selects candidates first; only these are compared with their eight neighbours.
- **Defective rows and columns.** A pixel is a line candidate when it passes (4.3) against the two pixels across the line (the pixels above and below for a row, left and right for a column), with `b` the larger of the two. A row or column is defective when it has at least 10 candidates, spread over at least 50 px, with no run longer than 10 consecutive pixels. Only the candidate pixels are left out, never a whole line. Pixels already flagged as hot are not counted again.

Anything wider than one pixel, such as a Bragg spot, a streak, a rod, the beam-stop edge or the Yoneda band, is kept. The numbers go into the record as `bad_pixels_left_out` (`hot`, `dead`, `defective_line_pixels`). Clusters of bad pixels and bad pixels of the background frame are not searched for.

Implementation: `src/gimap/features/analyze/domain/bad_pixels.py`

### 4.5 Masks and value limits

- **Mask file.** An image of the same shape; every non-zero pixel is masked (the pyFAI and Xeuss convention). A file of another shape is refused.
- **Drawn masks.** Rectangles (two corners, edges included) and polygons (three or more vertices) in canonical coordinates. A pixel is masked when its centre `(j + 0.5, i + 0.5)` lies inside a shape; polygons use Matplotlib's point-in-path test. The record counts the masked pixels of drawn shapes and mask file together (`drawn_mask_pixels`).
- **Valid value range.** Optional limits `[I_min, I_max]`, both inclusive, applied to the pixel value at that step of the order (section 4.9), that is after background subtraction.

Implementation: `src/gimap/features/analyze/domain/preprocess.py`, `src/gimap/features/analyze/domain/corrections.py`, `src/gimap/features/analyze/application/frame_preprocessing.py`

### 4.6 Background subtraction

```text
I′ = I − s · B                                                          (4.6)
```

`B` is a frame of the same detector and shape (for NeXus a chosen frame of the file) and `s` a scale, for example the ratio of exposure times. A pixel is valid only where both frames are valid; the background's validity follows (4.1) with its own stored type. Negative results stay valid. A subtracted frame is no longer photon counts, so its errors come from the pixel scatter (section 6.4).

Implementation: `src/gimap/features/analyze/domain/corrections.py`

### 4.7 Gap guard

Pixels next to module gaps and bad pixels often read wrong (edge pixels, charge sharing). The gap guard removes every pixel within N pixels of an invalid pixel:

```text
valid′(p) = valid(p) ∧ no pixel of the (2N + 1) × (2N + 1) square centred on p is invalid      (4.7)
```

- `0 ≤ N ≤ 20`. The default is 3 px in the Analyze window (Corrections; remembered as `analyze.gap_guard_px`). The domain object and settings files without the key default to 0.
- It grows around every pixel that is invalid at this point: reader-masked pixels and gap codes, pixels invalid in any summed frame, bad pixels, drawn and file masks, and pixels invalid in the background frame.
- It does not grow around pixels excluded later by the value range, nor around the horizon. The border of the frame does not count as invalid.

Implementation: `src/gimap/features/analyze/domain/corrections.py`

### 4.8 Mirror filling (GIWAXS)

Fill gaps from the mirror side (Mask step) is off by default and acts only on GIWAXS frames with a geometry. On a flat detector perpendicular to the beam, the reflection `X → −X` maps `q_y → −q_y` and leaves `q_x`, `q_z`, |q|, α_f and 2θ unchanged; it changes the signs of q_∥ and χ. A pixel without data is given the value at its mirror position about the beam-centre column, in the same row:

```text
m_j = 2 x_c − (j + 0.5) − 0.5 = 2 x_c − j − 1        (fractional column index)       (4.8)
j_0 = ⌊m_j⌋,    w = m_j − j_0                                                          (4.9)
I_fill(i, j) = (1 − w) I(i, j_0) + w I(i, j_0 + 1)                                     (4.10)
```

- Only pixels that are invalid at this point are filled: gaps, masks, bad pixels, gap-guard and value-range pixels.
- Both source pixels must be valid and inside the detector (only `j_0` when `w = 0`). Where the mirror has no data either, the gap stays.
- The record counts the filled pixels (`pixels_filled_from_mirror`).
- The method assumes a pattern symmetric in ±q_∥ (no in-plane anisotropy) and the flat, perpendicular detector. It copies measured photons, so the counting statistics no longer hold: the error model switches to pixel scatter (section 6.4) and the fit input says `counting_model_valid: false`. In I(χ), a filled region is the mirror image of the measured side by construction.

Implementation: `src/gimap/features/analyze/domain/preprocess.py`

### 4.9 The order of the steps

1. Load the frame or the sum of frames; validity (4.1).
2. Bad pixels found in the raw frame (section 4.4).
3. Drawn masks and mask file (section 4.5).
4. Background subtraction (section 4.6).
5. Gap guard (section 4.7).
6. Valid value range (section 4.5).
7. Mirror filling, GIWAXS only (section 4.8).
8. Intensity corrections, GIWAXS only (chapter 5).

Steps 4 to 7 run only when at least one of them is active. The raw frame is kept: changing any setting recomputes from the raw frame, so corrections never stack. Curves, q maps, the cake and every export use the result.

Implementation: `src/gimap/features/analyze/application/frame_preprocessing.py`, `src/gimap/features/analyze/application/use_cases.py`

## 5. Intensity corrections for GIWAXS

### 5.1 Definition

The intensity corrections (Mask step, folded; GIWAXS only) are off by default. When at least one is on, every pixel is divided by a factor:

```text
I′_p = I_p / F_p,      F_p = F_Ω · F_P · F_A                                     (5.1)
```

- Each factor is optional; a factor that is off equals 1.
- q never changes. Peak positions therefore do not change; intensities compared across χ or q do.
- `F_Ω` and `F_P` are 1 at the direct beam and `F_A` is 1 at `α_f = α_i`, so corrected values stay close to counts per pixel.
- Pixels where F is not finite or not positive become invalid.
- They are applied after all other preparation steps (section 4.9) and need the geometry. X, Y and L_p are those of (1.3), (1.4) and (2.2), and `cos 2θ = L/L_p` (2.12). On a tilted detector the vector t of a pixel is (3.15), its length |t| takes the place of L_p, and cos 2θ is its component along the beam over |t|.
- Comparable geometric corrections are part of other grazing-incidence packages, for example GIXSGUI (Jiang, 2015).

Implementation: `src/gimap/features/analyze/domain/intensity.py`, `src/gimap/features/analyze/application/frame_preprocessing.py`

### 5.2 Solid angle

A pixel of a flat detector, with its centre at the vector t from the sample (3.15), subtends
`Ω_p = p_x p_y L_n / |t|³`, where L_n is the perpendicular distance from the sample to the detector plane. The
correction is this solid angle relative to that of the pixel at the direct beam, where `|t| = L`:

```text
F_Ω = Ω_p / Ω_beam = (L / |t|)³                                                  (5.2)
```

This is what the code computes for every geometry, tilted or not. L_n cancels, and F_Ω is 1 at the direct beam
(pyFAI normalises to the point of normal incidence instead; the two differ by one constant, section 3.7). For an
untilted detector (all rotations 0: perpendicular to the direct beam) `|t| = L_p` (2.2) and `L/L_p = cos 2θ`, so

```text
Ω_p = p_x p_y L / L_p³ = (p_x p_y / L²) cos³ 2θ,      F_Ω = cos³ 2θ      (untilted)      (5.3)
```

On a tilted detector `L/|t|` is not cos 2θ (the component of t along the beam changes across the detector), so
(5.3) does not hold there and (5.2) is used.

Implementation: `src/gimap/features/analyze/domain/intensity.py`

### 5.3 Polarisation

```text
F_P = ½ [ 1 + cos² 2θ − f_pol · cos 2φ · sin² 2θ ]                                  (5.4)
cos 2φ = (X² − Y²) / (X² + Y²)          (taken as 0 at the direct beam)          (5.5)
```

φ is the azimuth on the detector measured from the horizontal X axis. The formula and the sign of f_pol follow pyFAI. Typical values of f_pol are 0.95 to 0.99 for a horizontally polarised synchrotron beam, 0 for an unpolarised laboratory source and −1 for vertical polarisation. The window proposes `f_pol` = 0.99 when the box is ticked; the domain default is off. With `f_pol = 1`, `F_P = cos² 2θ` along the horizontal through the beam and 1 along the vertical; with `f_pol = 0`, `F_P = (1 + cos² 2θ)/2` everywhere. For the physics of the correction see Kahn et al. (1982).

Implementation: `src/gimap/features/analyze/domain/intensity.py`

### 5.4 Absorption in the film

A uniform film of thickness t_f and attenuation length Λ_att = 1/µ is entered with t_f in nm and Λ_att in µm (from tables such as CXRO or NIST at the beam energy). Scattering from depth z travels `z/sin α_i` in and `z/sin α_f` out:

```text
s(α_f) = 1 / sin α_i + 1 / sin α_f                                               (5.6)
A(α_f) = ∫ from 0 to t_f of exp(−z s / Λ_att) dz = (Λ_att / s) (1 − exp(−t_f s / Λ_att))     (5.7)
F_A = A(α_f) / A(α_i)                                                            (5.8)
```

- The code evaluates `(1 − exp(−t_f s/Λ_att))/s`; the constant Λ_att cancels in the ratio.
- `sin α_f` is the exact value (2.8) for each pixel.
- Pixels with `sin α_f ≤ 0` (at or below the horizon) have no factor and are left out. The correction needs `α_i > 0`.
- Limits: for `t_f ≪ Λ_att`, `F_A → 1`; for `t_f ≫ Λ_att`, `F_A → s(α_i)/s(α_f)`.
- Not included: refraction (so the correction is valid only well above the critical angle), the reflected-wave paths of the DWBA, the substrate, and a varying composition with depth.

Implementation: `src/gimap/features/analyze/domain/intensity.py`

### 5.5 Uncertainty of corrected photon counts

When the frame holds photon counts (section 6.4), each corrected pixel keeps its Poisson variance, scaled:

```text
Var(I′_p) = Var(I_p) / F_p² = I_p / F_p² = I′_p / F_p                           (5.9)
σ_k = √max( Σ_{p∈k} I′_p / F_p ,  f̄_k² ) / n_k,      f̄_k = (1/n_k) Σ_{p∈k} 1/F_p      (5.10)
```

where the sums run over the n_k pixels of bin k and `I_k = Σ I′_p / n_k`. `f̄_k²` is the variance of one count at the bin's mean factor: as in (6.5), a bin of zero counts gets the error of one count, not `σ = 0`; with every F = 1, (5.10) is (6.5). When the frame is not photon counts, the corrected values get the pixel-scatter σ of section 6.4. Values corrected in this way are no longer counts: the fit input records `counting_model_valid: false` whenever a GIWAXS correction is on. The JSON record `<stem>_analysis.json` holds `intensity_corrections` with the formulas and their parameters. The factor is computed in blocks of 512 rows and kept while shape, geometry and options stay the same.

Implementation: `src/gimap/features/analyze/domain/intensity.py`, `src/gimap/features/analyze/domain/binning.py`, `src/gimap/features/analyze/application/export_record.py`

## 6. Reduction to curves and uncertainties

### 6.1 Principles

- Every point of a curve is the mean of the valid pixels assigned to it. The intensity unit is counts per pixel (`I (counts/pixel)`), divided by F when the GIWAXS corrections are on.
- **No pixel splitting.** Each pixel goes whole to the one bin or detector line that holds its centre's coordinate; pyFAI's split-pixel integrators distribute it instead.
- **Checked against pyFAI.** On the four public frames, GIMaP's I(q) over the whole azimuth and pyFAI's `integrate1d` (2026.9, histogram, no splitting) on the same pixels, q range and bins agree to 10⁻⁷ (95 % of the bins; the largest difference 0.2 %, where pixels on a bin edge fell the other way) on every bin that holds the same pixels in both. With the solid angle and polarisation, pyFAI divides the sum of the counts by the sum of the factors (`Σ I / Σ F`), GIMaP takes the mean of `I / F`; the two differ where F and I both vary across a bin: by a median of 0.26 % (at most 1.3 %) on the Xeuss frame, close to the sample, and 0.04 % on the P08 frame (`tools/benchmarks/pyfai_comparison.py`).
- No smoothing and no interpolation across gaps. A bin without pixels gives no point, and the plotted curve is broken there.
- Each curve carries, per point, x, I, σ and n (the number of pixels).
- Two samplings are used: one point per detector line (GISAXS cuts, section 6.3) and fixed-edge bins (GIWAXS, section 6.2).

### 6.2 Fixed-edge bins

For a coordinate x (q, q_z, q_∥ or χ) and N bins over `[x_lo, x_hi]`:

```text
e_k = x_lo + k (x_hi − x_lo) / N,            k = 0 … N                              (6.1)
pixel p goes to bin k  when  e_k ≤ x_p < e_{k+1}   (x_p = x_hi: the last bin)        (6.2)
x̄_k = (e_k + e_{k+1}) / 2,     S_k = Σ_{p∈k} I_p,     I_k = S_k / n_k                 (6.3)
```

Pixels outside the range or non-finite are ignored. Empty bins are left out. Before merging (section 6.5), the x of a point is the bin centre, not the mean x of its pixels.

Implementation: `src/gimap/features/analyze/domain/binning.py`

### 6.3 Native detector lines (GISAXS)

A GISAXS cut has one point per detector column (horizontal cut) or per detector row (vertical cut). For line l with n_l valid pixels in the band:

```text
x_l = (1/n_l) Σ_{p∈l} x_p,       I_l = S_l / n_l,       S_l = Σ_{p∈l} I_p              (6.4)
```

x is the mean of the exact `q_y` (or `q_z`) of the line's valid pixels, not the value at the band centre. Lines without valid pixels give no point; the points are sorted by x.

Implementation: `src/gimap/features/analyze/domain/binning.py`

### 6.4 The three error models

**(a) Poisson, for photon counts.**

```text
σ_k = √max(S_k, 1) / n_k          (fixed-edge bins, GIWAXS)                         (6.5)
σ_l = √max(S_l, 1) / n_l          (detector lines, GISAXS)                          (6.6)
```

Both clamp the sum to at least one count, so a bin or line of zeros gets `σ = 1/n`, never `σ = 0`, which a weighted fit could not use (a division by zero, or a point of infinite weight). For integer counts the clamp changes only such bins and lines: any other sum is at least 1.

**(b) Pixel scatter, for values that are not photon counts.**

```text
s_k² = ( Σ_{p∈k} I_p² − S_k² / n_k ) / (n_k − 1)                                     (6.7)
σ_k = √( s_k² / n_k )                                                               (6.8)
```

Negative `s_k²` (rounding) is set to 0. A bin or line with a single pixel takes the median `s²` of the bins with at least two pixels (0 if there are none).

**(c) Scaled Poisson, after the GIWAXS intensity corrections:** equation (5.10).

**Which model.** The choice is made once per frame:

| Frame | σ |
|---|---|
| photon counts: stored type not floating point, no background subtracted, no mirror filling | (a); (c) when GIWAXS intensity corrections are on |
| floating-point frame (dark- or background-subtracted at the beamline) | (b) |
| background subtracted in GIMaP | (b) |
| mirror filling applied (GIWAXS with a geometry, even if no pixel needed filling) | (b) |

**Assumptions and limits.**

- Poisson σ assumes that the stored values are raw photon counts. The choice follows the stored type: a raw-count frame stored as floating point gets (b), and a processed frame stored as integers gets (a).
- Pixel scatter assumes that the pixels of a bin are independent samples of one mean. Real structure inside a bin inflates σ: texture across χ in a full-ring q bin, the Yoneda profile across the rows of a GISAXS band, or the q gradient across a bin.
- Under mirror filling, the filled pixels are counted in n_k like measured ones. Where a bin holds both a pixel and its mirror copy, (6.8) therefore still underestimates σ somewhat (this is an inference from the code, not a measured number).
- Pixels are taken as independent; charge sharing and other correlated noise are not modelled. Only the standard error of each point is given; no covariance between points is computed.

Implementation: `src/gimap/features/analyze/domain/binning.py`, `src/gimap/features/analyze/application/frame_preprocessing.py`, `src/gimap/features/analyze/application/use_cases.py`

### 6.5 Merging sparse bins along q

Near the beam, at the edge of a sector, next to a gap or in the missing wedge, a bin catches few pixels. The I(q)-type curves of GIWAXS join such bins with their neighbours along x. This applies to I(q), the in-plane and out-of-plane sectors, a sector's I(q), the q-box curves I(q), I(q_z) and I(q_∥), and a region's I(q). It never applies to I(χ), to the GISAXS cuts or to the peak-fit windows.

- Threshold: `n_min = max(8, round(0.1 · median n))`. A curve whose bins all reach `n_min` is returned unchanged.
- The filled bins are walked in order of x and added to a group. A group closes when its pixel count reaches `n_min`.
- A step to the next filled bin of more than 2.5 median bin steps also closes the group, even when it is still sparse: a gap stays a gap.
- A sparse group at the end joins the previous point if it lies within 2.5 steps of it.

A merged point is

```text
x = Σ n_k x_k / Σ n_k,      I = Σ n_k I_k / Σ n_k,      n = Σ n_k                   (6.9)
σ = √( Σ n_k² σ_k² ) / Σ n_k                                                       (6.10)
```

For Poisson bins (6.10) equals `√(Σ max(S_k, 1))/Σ n_k`: the Poisson error of all the pixels together, `√(Σ S_k)/Σ n_k`, when no bin of the group is all zeros. For scatter bins it is `√(Σ n_k s_k²)/Σ n_k`. The x of a merged point is the pixel-weighted mean of the bin centres.

Implementation: `src/gimap/features/analyze/domain/binning.py`

### 6.6 GIWAXS curves

All GIWAXS curves use the usable pixels: valid (chapter 4) and above the horizon (section 2.3). Their q range `[q_lo, q_hi]` is the smallest and largest q of the usable pixels.

**Bin count.** The pixel q step δq is the 99th percentile of `|∇q|` on every fourth pixel, divided by 4. It is close to the largest q change between neighbouring pixels, which occurs near the beam. The radial bin count is

```text
N = clip( ⌈(q_hi − q_lo) / δq⌉ , 100, 3000 )                                      (6.11)
```

unless Radial bins (More cut settings) sets N. The bin width `(q_hi − q_lo)/N` then equals δq to within one part in N, so no bin is narrower than the coarsest pixel step; farther from the beam a bin spans several pixel steps. The clip can make bins narrower (fewer than 100 pixel steps in the range) or wider (more than 3000). Example (P08 geometry): δq = 0.00224 Å⁻¹, q from 0.012 to 4.757 Å⁻¹, N = 2114.

**Curves.**

| Curve | Pixels | x and bins | Merged |
|---|---|---|---|
| I(q) | all usable | q, N bins over `[q_lo, q_hi]` | yes |
| In-plane sector I(q) | `abs(χ) ≥ 90° − w_ip` | as I(q) | yes |
| Out-of-plane sector I(q) | `abs(χ) ≤ w_oop` | as I(q) | yes |
| I(χ) | q inside the I(χ) window | χ, 180 bins of 1° over [−90°, 90°] | no |
| Sector I(q) | `χ_1 ≤ χ ≤ χ_2` (signed), optional q range | q, N bins over the sector's q range | yes |
| Sector I(χ) (with a q range) | the sector | χ, `round((χ_2 − χ_1)/1°)` bins | no |
| q box I(q), I(q_z), I(q_∥) | the box in q_∥ × q_z | `max(20, min(3000, ⌊√n_box⌋))` bins over the box's q range (for I(q)) or its sides | yes |
| Region I(q) | q range × χ range | `max(20, round(N Δq_region / (q_hi − q_lo)))` bins over the region's q range | yes |
| Region I(χ) | the region | χ, bin width `max(1°, 2 δq / q̄)` (in degrees, q̄ the middle of the q range), at least 10 bins | no |

- The sector half-widths `w_ip` and `w_oop` default to 10° (window range 0.5° to 45°).
- A region's χ range is taken on |χ| when Both sides is ticked (the default), within [0°, 90°]; otherwise on signed χ within [−90°, 90°]. A new ring region spans ±max(0.01 Å⁻¹, 1.5 % of q) over all χ.
- A sector with a q range keeps N bins over that narrower range, so its bins can be narrower than δq; sparse ones are merged.
- **The I(χ) window** is the user's, or the strongest ring of I(q). That ring is found on ln I after skipping the first 5 % of the points. A 3-point median removes spikes, a running median of `max(5, n/10)` points (odd) is the baseline, and the peak with the largest prominence of at least 0.05 and a width of at least 3 points wins. The window is ±max(3 point steps, 1.5 % of q).
- **The peak-fit window** of a region is binned at the pixel step δq (at least 10 bins) and not merged.
- **2θ axis.** With the I(q) axis set to 2θ, the q of every point of the I(q)-type curves is converted with (2.12).

Implementation: `src/gimap/features/analyze/domain/giwaxs.py`, `src/gimap/features/analyze/domain/regions.py`, `src/gimap/features/analyze/domain/binning.py`

### 6.7 GISAXS cuts

**Yoneda band.** The search window runs from the row of `α_f = 1.5°` down to the row of `α_f = −0.2°` (equation (2.14)). The 0.2° below the computed horizon absorbs small errors of α_i or of the beam centre. The columns are two bands of 100 px on both sides of a 31-column strip around the beam column `c_0 = ⌊x_c⌋`, namely `[c_0 − 115, c_0 − 15)` and `[c_0 + 16, c_0 + 116)`. This keeps the direct beam, the specular rod and a rod-shaped beam stop out. The profile is the median of the valid pixels in each row, smoothed with a 3-row moving average that ignores missing rows. The brightest row gives `r_Y = row + 0.5`, and `α_f` follows from (2.14). The window needs at least 3 rows. For the cited phenomenon see Yoneda (1963).

**Horizontal cut I(q_y).** The band is centred on `r_c`: the Yoneda row, a row set by hand, or, when no Yoneda band is found, the horizon row minus 10 px (clamped to the frame).

```text
rows i = ⌊r_c − h⌋ … ⌈r_c + h⌉ − 1,       h = 2.5 px by default                    (6.12)
```

These are all rows whose area overlaps `[r_c − h, r_c + h]`: 5 rows when `r_c` is a pixel centre, 6 otherwise. Every valid pixel of these rows is used, including pixels below the horizon if the band is moved there. One point per column (6.4), `x = mean q_y`, signed.

**Vertical cut I(q_z).**

```text
columns j = ⌊c − w⌋ … ⌈c + w⌉ − 1,        w = 5 px by default, c = x_c by default    (6.13)
rows i = 0 … ⌊row_horizon⌋ − 1                                                       (6.14)
```

That gives 10 columns when c is an integer and 11 otherwise, and only rows entirely above the horizon. One point per row, `x = mean q_z`.

The nominal band sizes are therefore "5 px high" and "10 px wide". The error model is that of section 6.4.

**Make Left and Right Symmetric** (Cuts step) moves `x_c` onto the mirror axis of the horizontal band and changes nothing else. The Cuts step and Run Automatic Analysis use the same method; section 8.2 gives it.

**GISAXS q map.** A grid of at most 600 × 600 cells in `q_y`–`q_z` over the range of the usable pixels (section 6.8).

Implementation: `src/gimap/features/analyze/domain/gisaxs.py`, `src/gimap/features/analyze/domain/binning.py`, `src/gimap/features/analyze/domain/symmetry.py`, `src/gimap/features/analyze/domain/symmetry_coarse.py`

### 6.8 The q map and the cake

Both are means of pixel values on a regular grid:

```text
cell(p) = ( ⌊N_y (y_p − y_0)/(y_1 − y_0)⌋ , ⌊N_x (x_p − x_0)/(x_1 − x_0)⌋ ),    value = mean of I_p in the cell      (6.15)
```

| Map | x | y | Grid | Range |
|---|---|---|---|---|
| GISAXS q map | `q_y` | `q_z` | at most 600 × 600 | min to max of the usable pixels |
| GIWAXS q map | `q_∥` | `q_z` | at most 800 × 800 | min to max of the usable pixels |
| Cake | q | χ | 800 (q) × 360 (χ, 0.5° each) | q from 0 to the largest q; χ from −90° to +90° |

- Only usable pixels (valid and above the horizon) enter.
- A q map is never finer than the detector samples q. Along each axis the cell is at least the step of that coordinate between a usable pixel and its right and lower neighbours (their root sum of squares; the 95th percentile over the usable pixels), so `N = min(N_max, ⌊range / step⌋)`, at least 32. A 619 × 487 frame then gets about 490 × 580 cells instead of 800 × 800, on which nearly half the cells between measured ones stayed empty.
- Empty cells are NaN; in the cake, the missing wedge next to χ = 0° stays empty.
- The q maps are stored with row 0 at the largest `q_z`; the cake with row 0 at χ = −90°.
- The maps carry no σ. They are for display and for export as tables (Save ▸ View Data as CSV…); no curve is computed from them.

Implementation: `src/gimap/features/analyze/domain/binning.py`, `src/gimap/features/analyze/domain/cake.py`, `src/gimap/features/analyze/domain/gisaxs.py`, `src/gimap/features/analyze/domain/giwaxs.py`

### 6.9 The two halves of a signed cut

The horizontal GISAXS cut has signed `q_y`, and I(χ) has signed χ. The plots show both halves as measured (±), one half (+ or −), or both folded onto |x|. This choice changes only the display.

A model is a function of |q|, so a fit takes one curve: one half, or the mean of the halves on |q|:

```text
I(|q|) = ½ [ I_+(|q|) + Ĩ_−(|q|) ]                                                  (6.16)
σ(|q|) = ½ √( σ_+²(|q|) + σ̃_−²(|q|) )                                               (6.17)
```

- One half gives the |q| points of the mean: the `q_y > 0` half, in the prepared cut of Run Automatic Analysis and of its model fit, in the Series map and in Fitting alike, so one cut gives one mean curve. A half with fewer than two points never gives the grid.
- The tilde marks linear interpolation of the other half, and of its σ, at that |q|.
- Points are averaged only where the other half has measured points on both sides without a hole, or a point at the same |q| (within 1 % of a step). A hole is an interval wider than 2.5 times the local step, the median of the 9 surrounding intervals.
- Elsewhere the measured half stands alone with its own σ. Points of the other half that lie in a hole of this half, or beyond its ends, are added as measured. Nothing is interpolated across a hole.

**Automatic choice** (Run Automatic Analysis and its model fit). Only points with finite values, `I > 0` and at least one pixel are used. For each half GIMaP takes:

- the **reach**: its largest |q_y|;
- the **coverage**: the share of bins of width 1.5 × the median |q_y| step that hold at least one point. The bins run from the larger of the two halves' fourth |q_y| values to the half's reach, so the three points next to `q_y = 0` (specular rod, beam stop) are skipped.

The decision, in this order:

1. Only one half is on the detector: that half is fitted.
2. A half is not used when its reach is below 0.3 times the other's, or when its coverage is below 0.6 and more than 0.1 below the other's. The other half is fitted.
3. Otherwise both halves are interpolated linearly onto the |q_y| of the `q_y > 0` half within the common range (from the skipped points to the shorter reach; at least 5 points are needed). The mismatch is `m = median |ln(I_−/I_+)|`. When `m > 0.15` (about 16 %) the halves still disagree after the centre correction (a shadow, an absorber or real in-plane anisotropy): both are kept for display, and the better-covered half is fitted.
4. Otherwise the mean (6.16) is fitted.

**Prepared curve and files.** In the prepared cut a point without a usable σ gets the median relative σ of the other points; a cut without any σ gets 10 % of I. `<stem>_fit_input.dat` always holds both halves with signed q in Å⁻¹, and its record names the halves a fit should take. Fitting works in nm⁻¹ internally (1 Å⁻¹ = 10 nm⁻¹). A fit given more than 1000 points merges consecutive groups of `g = ⌈n/1000⌉` points: I is the 1/σ²-weighted mean, `σ = (Σ σ⁻²)^(−1/2)`, and q the plain mean (without σ, both are plain means). The last points that do not fill a group are dropped. The note of the prepared curve says what was averaged and merged.

Implementation: `src/gimap/shared/signed_cut.py`, `src/gimap/features/analyze/application/export_record.py`

### 6.10 Reference-peak corrections of a series

The Series map, its traces and the peak followed in a window are described in sections 11.1 and 11.2. This section gives the two corrections that Batch Export can apply to each frame of a GIWAXS series (In-situ corrections; both off by default). A peak that does not change during the run, such as a substrate reflection or an internal standard, is located as the highest I(q) point within `q_ref ± Δ` (defaults 2.132 and 0.035 Å⁻¹). Its position is therefore limited by the bin width; it is not a fitted centre.

```text
L′ = L · q_obs / q_ref           (repeated up to 3 times, until |ΔL|/L < 10⁻⁶)        (6.18)
g = I_target / I_peak            (I and σ multiplied by g; q unchanged)             (6.19)
```

Equation (6.18) is a small-angle scaling. Re-reducing the frame with L′ and repeating makes it converge at wide angles too. The target `I_target` is 1 by default. The factor g is taken from the first frame for the whole series, or from each frame.

Implementation: `src/gimap/features/analyze/domain/series.py`, `src/gimap/features/analyze/application/series.py`

## 7. GIWAXS automatic analysis

The automatic GIWAXS analysis takes the curves that Analyze has reduced from one frame (or a sum of frames) and
derives peak positions, d-spacings, widths, a coarse in-plane/out-of-plane comparison, the orientation
distribution and Hermans orientation factor of the strongest rings, and Scherrer sizes. It needs no AI. It runs
from Analyze's **Run Automatic Analysis**, and the same tools serve the command line (`tools/gimap_agent.py`)
and **Process with AI**. Every step is a fixed rule, so the same input and the same saved instrument profile give
the same numbers. Each rule and its reason is written into the report's *Decisions*.

The order of the steps is fixed: peak search on the full I(q) → comparison of the in-plane and out-of-plane
sectors at every peak → choice of the rings → for each ring, the orientation distribution and Hermans factor, and
the Scherrer size when the ring is a reliable fitted peak.

Units: q in Å⁻¹, d in Å, sizes reported in nm (τ in Å divided by 10), χ in degrees.

Implementation: `src/gimap/features/assistant/application/pipeline.py`

### 7.1 Input: technique, frames and the reduced curves

**Technique.** The frame is classified by the rule of section 2.4 (GIWAXS when `2θ_max > 20°`). Without a geometry nothing is classified and nothing is analysed. A technique that the user gives, or the mode chosen in Analyze, takes precedence over the classification.

**Frames of a series.** By default the analysis sums the last frames of a series, counted back from the last frame:

```text
n_sum = max( 1, min( 10, ⌊N_frames / 5⌋ ) )                                               (7.1)
```

A series of fewer than 10 frames is therefore represented by its last frame alone. A frame and a sum given by the
user replace this rule.

**Curves.** The analysis uses four curves of the GIWAXS reduction (section 6.6): the radial I(q), the in-plane and out-of-plane sectors (half widths 10° by default), and I(χ) in 1° bins over the ring window of section 7.7. All use the valid pixels above the sample horizon (section 2.3), with the binning, the merging of sparse bins and the standard errors of sections 6.2 to 6.6. χ is that of (2.13): 0° along the surface normal, ±90° in the sample plane, with the sign of `q_y`. Intensity corrections of Analyze (off by default, chapter 5) are applied before these curves are made.

Implementation: `src/gimap/features/assistant/application/pipeline.py`, `src/gimap/features/analyze/domain/giwaxs.py`

### 7.2 Background: SNIP on LLS-transformed data

The background under the peaks is estimated with the SNIP clipping algorithm (Ryan et al., 1988; Morháč et al.,
1997; Morháč & Matoušek, 2008) on log-log-square-root (LLS) transformed values, with a decreasing clipping window:

```text
v_i = ln( ln( √(y_i + 1) + 1 ) + 1 ),   y_i clipped to ≥ 0                                  (7.2)
v_i ← min( v_i , ½ (v_{i−p} + v_{i+p}) )   for p = P, P−1, …, 1                             (7.3)
b_i = ( exp( exp(v_i) − 1 ) − 1 )² − 1                                                      (7.4)
```

P is the half width, in points, of the widest clipping step. It is derived from a width W in q and the median
bin spacing Δq of the curve:

```text
P = clip( round(W / Δq), 3, ⌊N / 3⌋ ),   W = 0.15 Å⁻¹ for GIWAXS (0.02 Å⁻¹ for GISAXS)        (7.5)
```

Features broader than about W count as background. Because SNIP follows the lower edge of the noise, GIMaP applies
it to a copy of the curve smoothed with a Gaussian of σ = 2 bins, and then lifts the baseline onto the noise:

```text
b_i ← b_i + median{ y_j − b_j : abs(y_j − b_j) < 3 σ_j }                                    (7.6)
```

The shift (7.6) is applied only when at least `max(5, ⌊N/5⌋)` bins satisfy the condition. Here y is the
unsmoothed curve and σ its bin error. The baseline is used to find candidates (section 7.3), as the start of each
peak fit, for the sector comparison (section 7.6) and as the ring background of the orientation analysis
(section 7.7). It is not the background of the reported peak heights: those refer to the local linear background of
the profile fit (section 7.4).

Implementation: `src/gimap/features/assistant/domain/profiles.py`

### 7.3 Peak finding

The search runs on the measured bins of the radial curve (finite values, at least one pixel, `σ > 0`), over its whole
q range. At least 12 bins are needed.

1. The residual `r = y − b` is smoothed with a Gaussian of σ = 1 bin. Local maxima are found with
   `scipy.signal.find_peaks` (any prominence, width ≥ 1 bin). The width w (in bins) is the width at half
   prominence.
2. For each maximum at bin i, the local significance and the relative height are

   ```text
   SNR_loc = mean_k( r_k ) / ( √(Σ_k σ_k²) / m ),   k = i − round(w/2) … i + round(w/2), m bins       (7.7)
   h_rel = r̃_i / abs(b_i)                                                                          (7.8)
   ```

   with r̃ the smoothed residual.
3. A maximum becomes a candidate when `SNR_loc ≥ 3`, `h_rel ≥ 0.02` and `w ≥ 2` bins. The relative-height test is
   needed because bins of many pixels make tiny ripples statistically significant.
4. At most 12 candidates, those of largest prominence, are fitted (section 7.4), in order of prominence. A fit that
   converges closer than half the larger FWHM to a peak already accepted is dropped, unless the widths differ by more
   than a factor 3 and both have SNR ≥ 5 (a sharp reflection on a halo).

**The cap.** The limit of 12 keeps a run short: fitting candidates that fail or merge takes most of the time. It is
the default, not a judgement of the other candidates. When more candidates qualify than are fitted, the report says
so: the *Decisions* line *peak search* gives "N candidate peaks found; the 12 most prominent were fitted", the peak
search record gives `candidates`, `max_peaks` and `truncated`, and the series hints are titled as taken from the
fitted peaks only. The peak table, the sector comparison, the rings analysed and the series hints all use the fitted
peaks. A candidate is a maximum before fitting; fitting merges or rejects many of them, so fitting all gives fewer
peaks than candidates. `--max-peaks N` (command line) or `max_peaks` (the tools `find_peaks` and
`run_standard_pipeline`) fits more; Run Automatic Analysis in the window always uses 12. On the public P08 frame
115 candidates qualify; fitting all of them gives 32 peaks instead of 12, among them clean ones such as
q = 3.585 Å⁻¹ (SNR 33), in about 8 s instead of under 1 s.

When no peak qualifies, the report gives the reason and the strongest feature seen (its q, SNR_loc and h_rel).

Implementation: `src/gimap/features/assistant/domain/peaks.py`

### 7.4 Peak profile fit and its uncertainties

Each candidate is fitted with a Gaussian on a local linear background:

```text
I(q) = h_pk exp( −½ ((q − q_0)/s)² ) + b_0 + b_1 (q − q_0)                                        (7.9)
s = FWHM / ( 2 √(2 ln 2) )                                                                  (7.10)
```

**Window and start.** With `FWHM_guess = max(w Δq, 2 Δq)`, the fit uses the bins within
`h = max(3 FWHM_guess, 5 Δq)` of the candidate, and at least 7 of them. The start is h_pk = r at the candidate (at
least 10⁻¹²), q_0 at the candidate, FWHM_guess, b_0 = the SNIP baseline there and b_1 = 0. Bounds: h_pk ≥ 0, q_0 within the
window, 0.5 Δq ≤ FWHM ≤ 2h; b_0 and b_1 are free.

**Fit.** Weighted least squares with the bin errors as absolute σ — Poisson counting errors for photon counts as
measured, the pixel scatter of each bin otherwise (section 6.4); the report's *Decisions* line *uncertainties* names
which (`scipy.optimize.curve_fit`, bounded trust region,
at most 5000 evaluations). The fit is rejected when q_0 lies on the window edge, when FWHM reaches its upper bound,
or when h_pk ≤ 0. If the fitted centre leaves the candidate's half-maximum span (a shoulder or neighbour widened the
candidate), the peak is fitted again with twice the width of its cleaner flank; the refit is kept when it stays
inside the span and is significant.

**Goodness of fit and errors.** With n points in the window and C the covariance matrix returned for absolute σ:

```text
χ²_red = Σ_i ((y_i − I(q_i)) / σ_i)² / (n − 5)                                                 (7.11)
σ_p = √(C_pp) · max(1, √χ²_red)                                                                (7.12)
```

The errors are inflated when `χ²_red > 1` and never deflated. After the fit, a peak is kept only if

```text
SNR = h_pk / σ(h_pk) ≥ 3   and   h_pk ≥ 0.02 abs(b_0)                                                  (7.13)
```

A candidate whose profile cannot be fitted is kept, with the flag `fit_failed` and the width from step 1 of section 7.3,
only if `SNR_loc ≥ 6`.

**Derived quantities.**

```text
d = 2π / q_0                                                                                 (7.14)
σ_d = 2π σ(q_0) / q_0²                                                                         (7.15)
A = h_pk · FWHM · √( π / (4 ln 2) ) ≈ 1.0645 h_pk · FWHM                                          (7.16)
```

h_pk is in counts per pixel above the local linear background; A is in counts per pixel × Å⁻¹.

**Flags.** Δq is the median bin spacing.

| Flag | Condition | Consequence |
|---|---|---|
| weak | SNR below 5 | caveat |
| at_edge | q_0 ± 1.5 FWHM outside the searched range | not reliable: "at the end of the data: check" |
| resolution_limited | FWHM below 3 Δq | reported; a size is a lower bound |
| spike | resolution-limited and h_pk above 5 × abs(b_0), the local background | caveat |
| step_edges (with spike) | a flat-topped box: its 20 %→80 % rise or fall within 3 bins of the 3-bin average, half-maximum width ≥ 15 bins and ≥ 3 × its edges, top ≥ 10 σ | caveat |
| broad | FWHM above 0.15 q_0 (a halo) | caveat; no size |
| overlap | another peak closer than 0.75 (FWHM_1 + FWHM_2) | reported only |
| poor_fit | χ²_red of the peak fit above 10 | reported ("poor fit") in the peak table; the errors are those of equation 7.12 |
| fit_failed | no usable profile fit | caveat |

A peak with a caveat (step_edges, spike, fit_failed, weak or broad, in this order of precedence) is listed in the
report with the reason, but it is not used for ring selection, sizes or series hints. Neither is a peak flagged
at_edge, whose shape, position and width the end of the searched range may cut off: the report lists it as "at the
end of the data: check". The peaks left, without a caveat and not at the edge, are the *reliable* peaks.

**Series hints.** From each of the three lowest-q reliable peaks, GIMaP looks for the orders of a lamellar
series (1 : 2 : 3 : 4) and of a hexagonal series (1 : √3 : 2 : √7 : 3), each order matched within 2 % of its
expected q. A series is reported when at least three orders (the first included) match. The orders of a lamellar
series must follow one another: the search ends at the first missing order, so a peak at 4.92 times the first q with
the third and fourth order missing is no lamellar series. A hexagonal series must include √3 or √7, the orders a
lamellar series does not have. This is a hint; no phase is assigned.

Implementation: `src/gimap/features/assistant/domain/peaks.py`

### 7.5 Which rings are analysed

Among the reliable peaks (no caveat, not at_edge), the analysis takes the **three of largest area A** (7.16); the
number is an option (`--rings N`). A ring the user names (`ring_q`) is added when no chosen peak lies within half its
FWHM. Each chosen ring gets an orientation analysis (section 7.7). A Scherrer size (section 7.8) is computed only when
a reliable peak lies within half a FWHM of the ring. The rings left out, and why, are listed in the *Decisions*.

Implementation: `src/gimap/features/assistant/application/pipeline.py`

### 7.6 In-plane and out-of-plane sectors

For every radial peak (q_0, FWHM), the in-plane and out-of-plane sector curves are compared. Each sector gets its own
SNIP baseline b_s (7.2–7.6; P from W = 0.15 Å⁻¹ and the coarser of the two sector bin spacings, at least 3 points).
The net intensity per pixel and its error are averaged over the m bins with `abs(q − q_0) ≤ h`:

```text
h = max( 0.5 FWHM, 1.5 Δq )                                                                 (7.17)
N_s = mean( I_s − b_s ),   σ_N = √(Σ σ²) / m                                                 (7.18)
```

A sector "sees" the peak when `N_s / σ_N ≥ 3`. When both do,

```text
ρ_oi = N_oop / N_ip                                                                            (7.19)
σ_ρ = abs(ρ_oi) · √( (σ_oop / N_oop)² + (σ_ip / N_ip)² )                                        (7.20)
```

ρ_oi ≥ 2 is reported as *mainly out-of-plane*, ρ_oi ≤ ½ as *mainly in-plane*, anything between as *both sectors*.
A sector whose own background is below 0.2 times the background of the full ring at that q (the radial baseline)
is taken as shadowed (beam stop, absorber, insensitive area) and is not compared. A sector without pixels at q_0 is
named as not measured.

This is a coarse comparison of two 20°-wide sectors. It is not a pole figure.

Implementation: `src/gimap/features/assistant/domain/orientation.py`

### 7.7 Ring orientation and the Hermans orientation factor

**Ring window.** The ring is centred on the fitted q_0 of the nearest radial peak (when one lies within
`max(2 FWHM, 3 Δq)` of the requested q), with the half width

```text
h_χ = max( 0.75 FWHM, 2 Δq ),   limited to 0.04 q_0 for a broad peak                          (7.21)
```

Without a fitted peak the window is ±1.5 % of the requested q. The pixels with q in this window give I(χ) in 1° bins.

**Background.** The SNIP baseline b of the radial I(q) at the ring centre is subtracted. It is one number for the
whole ring, independent of χ.

**Folding.** The two halves ±χ are combined into 90 bins of `abs(χ)`, bin k covering [k°, k+1°), with centre
χ_k = k + 0.5°, by inverse-variance weighting:

```text
Ī_k = Σ_j w_j (I_j − b) / Σ_j w_j,   w_j = 1/σ_j²                                            (7.22)
σ_k = 1 / √( Σ_j w_j )                                                                       (7.23)
```

Bins without pixels are unmeasured. When `b > 0`, bins whose intensity is below 0.2 b (a shadow), and their
neighbours, count as unmeasured too.

**Requirements.** No factor is computed when fewer than 10 χ bins are measured, when no folded bin reaches
`Ī_k / σ_k ≥ 3`, or when the measured share of the sin χ weight is below 80 %:

```text
C_w = Σ_{k measured} sin χ_k / Σ_{k = 0…89} sin χ_k ≥ 0.8                                    (7.24)
```

The sin χ weight makes the range near the sample plane count most. A missing wedge of `abs(χ)` below 10° costs only
1.5 % of the weight (C_w = 0.985); an isotropic ring seen only for `abs(χ)` below 60° gives C_w = 0.50 and is refused.

**Hermans factor.** For a fibre texture about the surface normal, with negative net values clipped to zero:

```text
⟨cos²χ⟩ = Σ_k max(Ī_k, 0) sin χ_k cos²χ_k / Σ_k max(Ī_k, 0) sin χ_k                         (7.25)
f = ( 3 ⟨cos²χ⟩ − 1 ) / 2                                                                    (7.26)
```

The sums run over the measured bins. f = 1 means all orientations along the normal, f = 0 random, f = −½ all in the
plane (Hermans & Platzek, 1939; for GIWAXS see Baker et al., 2010). The code and the current report write this as
"Herman's f"; the standard name is the Hermans orientation function.

**Isotropic reference.** Because unmeasured bins bias f, the report gives the value an isotropic ring would have on
the same measured bins:

```text
f_iso = f  evaluated with Ī_k = 1 on every measured bin                                      (7.27)
```

f must be compared with f_iso, not with 0. With the code's 1° bins, an isotropic ring with `abs(χ)` below 10° missing
gives f_iso = −0.015; with `abs(χ)` below 20° missing, f_iso = −0.058 (C_w = 0.94).

**Uncertainty.** Monte Carlo with 200 draws and a fixed seed (0):

```text
f^(m) = f( Ī_k + ε_k^(m) ),   ε_k^(m) ~ N(0, σ_k²),   m = 1 … 200                             (7.28)
σ_f = standard deviation of { f^(m) }                                                        (7.29)
```

The clipping of (7.25) is applied in each draw. σ_f is a statistical uncertainty: it propagates the bin errors σ_k,
which are Poisson counting errors for photon counts as measured and the pixel scatter otherwise (section 6.4; the
public P08 frame is a floating-point frame, so there σ_f is a pixel-scatter uncertainty, not counting noise). It does
not contain the uncertainty of the background b, of the window or of the geometry, nor the dependence on the bins
next to the sample horizon described below.

**Maxima and texture text.** The folded profile is interpolated across unmeasured bins and smoothed with a Gaussian
of σ = 1.5 bins. Maxima need a prominence of at least 3 times the median σ_k, and are reported only in measured bins
(at most three, by height). Their FWHM is the width at half height above the lowest measured value; a maximum at the
fold edge (0° or 90°) has its width doubled. The texture sentence follows these rules, in order:

| Condition | Text |
|---|---|
| max − min of the smoothed measured profile below 3 × median σ_k | isotropic within the noise |
| a second maximum ≥ 0.85 × the highest and ≥ 30° away | no single preferred orientation |
| abs(f − f_iso) below max(0.1, 3σ_f) and the highest maximum within 20° of 0° or 90° | no clear preferred orientation (f − f_iso and the margin are given) |
| highest maximum below 20° but f − f_iso < 0, or above 70° but f − f_iso > 0 | no single preferred orientation (the maximum and f disagree) |
| highest maximum below 20° | oriented along the surface normal |
| highest maximum above 70° | oriented in the sample plane |
| otherwise | tilted, maximum at ±χ |

The rule compares f with f_iso, not with 0 (7.27): for a partly measured ring a random orientation does not give
f = 0, and a ring measured only up to `abs(χ)` = 80° has f_iso = +0.10 already. The margin is 0.1, or three
statistical uncertainties σ_f when they are larger. A maximum next to an unmeasured range is flagged: the true
maximum may lie inside that range. The report's *Decisions* line *texture rule* states the rule, and every ring
row gives f ± σ_f, f_iso and f − f_iso.

**The bins next to the sample horizon.** The sin χ weight makes the bins near `abs(χ)` = 90° count most in f, and
there the exit angle α_f is smallest: absorption in the film, the transmission function or a shadow of the sample
can lower the intensity, and none of them is corrected. f − f_iso is therefore also computed with the bins
`abs(χ)` ≥ 85° left out (f and f_iso on the same remaining bins):

```text
Δ_h = (f − f_iso)  −  (f − f_iso) over the measured bins with χ_k < 85°
```

When `abs(Δ_h)` exceeds both 0.01 and 3σ_f, the ring's notes give both values and Δ_h in units of σ_f, a second
note says when the texture rule would give another verdict without those bins, and the *Decisions* line *ring
orientation* names the rings. On the public P08 frame, where I(χ) falls to the radial background within about 5° of
the sample plane, leaving those bins out moves f − f_iso of the three analysed rings by 17–27 σ_f (for example
+0.141 → +0.094 at q = 0.997 Å⁻¹), and the verdicts of the rings at 0.997 and 1.989 Å⁻¹ would change. f, f − f_iso
and the texture sentence of such a ring describe the measured intensity, including whatever lowers it next to the
horizon; whether that is texture or an exit-angle effect needs a judgement the analysis does not make.

**Assumptions and what is not modelled.** The factor assumes in-plane isotropy (fibre texture). The missing wedge
near χ = 0 is not reconstructed; it only enters through C_w and f_iso. I(χ) is the intensity on the detector:
no Lorentz or Ewald-sphere correction is applied to it, and no refraction correction of q_z. The background is a
χ-independent constant; an anisotropic diffuse background biases f.

Implementation: `src/gimap/features/assistant/domain/orientation.py`, with the window in
`src/gimap/features/assistant/application/tools.py`

### 7.8 Crystallite size (Scherrer)

The Scherrer equation (Scherrer, 1918) written in q gives the crystallite size τ (the code's `L`) from the FWHM of the Gaussian fit (7.9) in Å⁻¹:

```text
τ = 2π K / β_q                                                                               (7.30)
β_q = √( FWHM² − FWHM_instr² )                                                               (7.31)
σ_τ = τ · FWHM · σ_FWHM / β_q²                                                                (7.32)
```

(7.30) is the usual `τ = K λ / (β cos θ)` with `β = Δ(2θ)` in radians, since `β_q = (2π/λ) cos θ · β`.
(7.31) removes an instrumental width by Gaussian quadrature; (7.32) is the exact derivative of (7.30–7.31) with
respect to FWHM. Defaults: K = 0.9 (Patterson, 1939; Langford & Wilson, 1978). **Run Automatic Analysis passes no
instrumental width (FWHM_instr = 0).** Every automatic size is therefore a lower bound of the coherence length, and is
reported with "≥". The beam footprint and divergence, the sample–detector distance and pixel size, the q binning and
the mixing of directions over all χ all broaden the radial peak (Smilgies, 2009), and so do strain and
paracrystalline disorder; none of these is subtracted. A size is also marked a lower bound when FWHM is below 3 bins
(binning-limited). The tool accepts K and FWHM_instr when called from a script or by Process with AI.

No size is given when FWHM is above 0.15 q_0 (a halo: the report gives 2π/FWHM as a correlation length, not a crystal
size), or when FWHM ≤ FWHM_instr.

Example (invented values): q_0 = 1.50 Å⁻¹, FWHM = 0.030 ± 0.0006 Å⁻¹ give d = 4.189 Å and
τ ≥ 188.5 ± 3.8 Å, reported as τ ≥ 18.85 nm.

Implementation: `src/gimap/features/assistant/domain/size.py`

### 7.9 What the report states

For each peak: q_0 ± σ, d ± σ_d, FWHM ± σ, height, local background, area, SNR, χ²_red, flags and caveat, the sector
preference and ρ_oi, and the size with its lower-bound mark and note. For each ring: the q window, coverage (unweighted
share of measured 1° bins) and C_w, f ± σ_f, f_iso, f − f_iso with its margin and its value without the bins
next to the horizon, the maxima, the unmeasured and shadowed `abs(χ)` ranges, the texture text and the reason when
no f was computed. The search settings (q range, the I(q) points searched and the radial bins of the reduction,
bin width, W, thresholds, the candidates that qualified and the cap), the series hints, every default decision with
its reason (*Decisions*) and the open questions (*Needs attention*) complete it. The record keeps the values
`report.md` prints (q, d, FWHM, SNR and size of a peak; f, σ_f, f_iso and f − f_iso of a ring) unrounded, and
`report.md` rounds each once: one number of decimals per column of the peak table (q 4, d 3, FWHM 4, SNR 1, size 1),
f, f_iso and f − f_iso with the decimals that show σ_f to two significant digits, pixel positions to 0.01 px. Besides the choices of the run, the GIWAXS *Decisions* always say what the numbers are:

- *q*: kinematic, no refraction correction. Near or below the film's critical angle, refraction shifts q_z by up to
  about k sin α_i, and the line gives that number for the frame (`2π/λ · sin α_i`; 0.0119 Å⁻¹ for the public P08
  frame at α_i = 0.075°, λ = 0.6888 Å), or says that it is unknown while α_i is 0° for want of a value. With the
  film's critical angle given (section 2.6) it reads *q_z corrected for refraction in the film (αc = …)*;
- *uncertainties*: statistical only, with the error model of the frame (Poisson counting statistics, or the pixel
  scatter of each bin); the geometry, the background under a ring and refraction are not included;
- *peak search* when the list was cut at the cap (section 7.3), *texture rule*, and *ring orientation* for rings
  whose f depends on the bins next to the horizon (section 7.7).

*Incidence angle* names where α_i came from, in this order: the options, the notes, the value set in Analyze's
command bar (by hand or by the tool `set_incidence_angle`; Analyze uses it instead of the instrument profile's),
the instrument profile. A value from the options or the notes that differs from the one set in the command bar
replaces it there, so the α_i reported is the α_i used. Only when none of them gives α_i is 0° used and asked for. Numbers in the JSON report are rounded to four significant digits; the beam centre keeps full precision.

Implementation: `src/gimap/features/assistant/application/pipeline_rows.py`

## 8. GISAXS automatic analysis

For a GISAXS frame the automatic analysis places the horizontal cut at the Yoneda band, moves the beam-centre column
to the mirror axis of the cut, decides which halves of the cut to use, gives the length scale 2π/q* of a side maximum
or shoulder (a length the curve shows, not necessarily the particle spacing), and fits the prepared cut with
Predict parameters (section 8.5). It runs from **Run Automatic Analysis** in
Analyze and from the command line. The technique and the frames are chosen as in section 7.1.

Units: q_y and q_z in Å⁻¹ in Analyze; the fit converts to nm⁻¹ (q_nm = 10 q_Å); lengths in nm.

Implementation: `src/gimap/features/assistant/application/gisaxs_procedure.py`

### 8.1 Yoneda band and the cuts

The cuts are those of section 6.7: the horizontal cut I(q_y) on a band of 5 rows centred on the Yoneda row r_Y, and the vertical cut I(q_z) on 10 columns around x_c (11 when x_c is not an integer, as after the symmetry refinement), above the horizon. Each has one point per detector line with the standard errors of section 6.4. Nothing is interpolated or rebinned, so gaps stay gaps.

The Yoneda search of section 6.7 is purely empirical: **no critical angle of any material is used**, and the brightest diffuse row between `α_f = −0.2°` and `α_f = 1.5°` is taken, whatever its origin. The report gives r_Y and its exit angle α_f(r_Y) from (2.14). When the window holds no valid pixels, the cut is placed 10 px above the horizon, with a warning and a *Needs attention* item (check α_i).

Implementation: `src/gimap/features/analyze/domain/gisaxs.py`

### 8.2 Beam-centre column from mirror symmetry

The direct beam is usually hidden by the beam stop, but the diffuse scattering is symmetric in q_y. The column of
q_y = 0 is found by mirroring the band of the horizontal cut. Only x is refined; the beam-centre row is kept.

**Profile.** For every column j, Ī_j is the mean of the valid pixels in the cut's rows. It is compared on a
compressed scale:

```text
T_j = asinh( Ī_j / s ),   s = 25th percentile of the positive Ī_j                             (8.1)
Huber_δ(u) = ½ u²  for u ≤ δ,   δ (u − ½ δ)  for u > δ                                         (8.2)
```

**Coarse scan (global).** Every integer and half-integer axis c across the detector is tried (with a stride on very
wide frames). For each, the columns c − d and c + d (d ≥ 1 px) are paired only where both are measured; no gap is
bridged. An axis is a candidate when at least 60 % of the possible pairs are measured (and at least 40), when the
pairs carry at least half of the profile's prominent contrast, and when the paired values keep at least 10 % of the
profile variance. Its score is

```text
E_0(c) = mean_d Huber_1( abs(T_{c−d} − T_{c+d}) ) / Var( paired T )                              (8.3)
```

The best score must not exceed 0.25. Several similarly supported minima (a repeated scattering lobe) stop the search
with a message instead of a guess.

**Fine search (subpixel).** Around the coarse axis, within ±min(4 px, 10 % of the distance to the nearer frame edge),
T is interpolated linearly and

```text
E(c) = Σ_d w_d Huber_1( abs(T(c − d) − T(c + d)) ) / Σ_d w_d,   w_d = 1/d,   d = 1, 1.5, 2, … px     (8.4)
```

is minimised: first on a grid of about 1 px spacing, then by bounded scalar minimisation between the neighbours of the
best grid point, with a stopping tolerance of 10⁻⁴ px. The offsets d stop a margin (at most 80 px) before the nearer
frame edge, and only offsets whose interpolation support is measured over the whole search interval are used (at
least 40 and 60 % of them). The weights 1/d give equal logarithmic ranges of distance similar weight, so the many far
pixels do not outweigh the low-q structure; the distance is in pixels, approximately proportional to `abs(q_y)`. The
coarse axis, the best grid point, the minimiser and the initial centre (when inside the interval) compete, and the
lowest E wins. The result is given in canonical pixel coordinates (centre of column j at j + 0.5).

**No uncertainty is reported for the centre.** The 10⁻⁴ px tolerance is a numerical stopping rule, and E is not a
probability. A shift of more than 20 px from the calibrated centre is put under *Needs attention*. In the automatic
analysis a shift of more than 100 px is not applied at all: a calibration of the measurement is off by a few pixels,
so the run keeps the calibrated centre and reports that the calibration does not match the frame; an axis that
could not be found is an open item as well. The report's *asymmetry a → b* is E at the calibrated and at the
refined axis.

Implementation: `src/gimap/features/analyze/domain/symmetry.py`, `src/gimap/features/analyze/domain/symmetry_coarse.py`

### 8.3 The halves of the cut

After the centre correction the two halves, `q_y < 0` and `q_y > 0`, should agree. The analysis decides which halves to fit by the automatic choice of section 6.9 (reach, coverage and the mismatch m, with the limits 0.3, 0.6 and 0.15). It prepares the fitted curve as described there: the mean of the halves (6.16)–(6.17) on the |q_y| of the `q_y > 0` half, or one half, with at most 1000 points. The *Decisions* of the report give the reason of the choice, with the coverage of each half and the mismatch.

Implementation: `src/gimap/shared/signed_cut.py`

### 8.4 Length scale 2π/q* of the cut

The peak search of sections 7.3–7.4 runs on the prepared `abs(q_y)` curve with W = 0.02 Å⁻¹. Maxima closer to q_y = 0 than
10 bins belong to the central peak and are ignored. The strongest (highest SNR) remaining peak without the flags
spike, broad or fit_failed is a **maximum** (a weak peak, SNR 3–5, qualifies), and

```text
D* = 2π / q*                                                                                  (8.5)
```

is reported as a **length scale of the cut** (in nm: 2π/q*/10): the decision *2π/q\* = … nm at the peak of the cut
(|qy| = … Å⁻¹)*, with the reason *a length scale, not necessarily the particle spacing; check it with a model fit*.
If there is none, the strongest broad or fit_failed feature with
SNR ≥ 5 is reported as a **shoulder** (a change of slope, not a resolved peak): *… at a shoulder of the cut (…), no
resolved peak*, with the same reason and *a form factor or the beam tail makes a shoulder too*. The 0.15 q_0 "broad" limit of GIWAXS is reused, so a wide correlation peak is often classed a shoulder.
Example (invented): q* = 0.020 Å⁻¹ gives D* = 314 Å = 31.4 nm.

**Why D\* is not called a spacing.** A side maximum of I(q_y) can come from the interparticle correlation, but the
knee of a form factor and the tail of the direct beam make a shoulder too, and with the paracrystal model of chapter 9
the fitted D differs from 2π/q* when the disorder is large. On BornAgain 24.1 simulations of BornAgain's documented
model of the GALAXI frame, placed on the measured q grid, the search found a shoulder at 2π/q* = 73.9 nm for a peak
distance of 53.6 nm, and a shoulder at 60.6 nm for the same particles without interference. On 14 simulated cuts with
known parameters, 2π/q* was within 10 % of the true distance in 1 of 10 cuts with interference, and a maximum or
shoulder was found on all 4 cuts without interference. D* is therefore a result of its own: the fit of section 8.5
needs no start and does not take one from it. The records keep its value under the earlier key (`spacing.distance_nm`).

Implementation: `src/gimap/features/assistant/domain/gisaxs_cut.py`, `src/gimap/features/assistant/application/gisaxs_tools.py`

### 8.5 The model fit and how the solution is chosen

The prepared curve (section 8.3; q = 10 `abs(q_y)` in nm⁻¹) is fitted with **Predict parameters**, the network method
of Fitting (section 9.13): GIMaP's network proposes compositions and values, every solution is refined by bounded
least squares in Fitting's model (9.1), its posterior is sampled around it, and the solutions are weighted by their
posterior probability. The same function fits the cut in Batch Export's *Particle model of the horizontal cut*, in
Process with AI and the MCP tool `fit_horizontal_cut`, and on the command line (`tools/gimap_agent.py`). It takes about
25 s per cut on one core.

**Model and compositions.** The cut is taken at the effective interface (q_z = 0, section 9.10), with Fitting's
defaults: sizes Gaussian by number, the paracrystal of section 9.5 in the LMA, the beam tail A / (1 + (q/w)^ν) and a
constant background (9.1). Without a given composition (as in Run Automatic Analysis) the network chooses among one
or two populations of spheres and upright cylinders, each with or without a paracrystal (the head's four most
probable of these, and its most probable of one and of two populations). A given composition (1 sphere, 2 random cylinder, 3 vertical cylinder, each population with a
paracrystal: Batch Export's **Model**, the `components` of the tool) is fitted whole; random cylinders are fitted only
then, since they take far longer. The network needs no start: nothing is taken from D* of section 8.4, which stays a
result of its own. The fit needs at least 8 points; a cut of more than 1000 points is merged first (section 6.9).

**Scores.** For each solution, with r_i = (I_model(q_i) − I_i)/σ_i:

```text
χ²ᵣ = Σ_i r_i² / (N − k)                                                                      (8.6)
logRMSE = √( mean over I_i > 0 of ( ln(I_model,i / I_i) )² )                                      (8.7)
```

N counts the points Predict parameters fitted and k its free values (the scales included); σ is the curve's own.
Predict leaves out the points below the network's q window (q < q_max/300, section 9.13) and those without a usable
value, and every report says how many (*596 of 598 points; 2 below the network's q window*; `points` of a solution:
`fitted`, `given`, `below_window`, `unusable`). It is Fitting's χ²_red (9.23) of the same solution on those points,
and the one number shown for a solution on the Fit page, in the automatic analysis's table, report, CSV and JSON
and in Batch Export's table (until 0.2.0's release preparation the reports gave a mean over all points handed
over, χ²/N, which differed from the Fit page's value). The probability p of a solution is its
composition's posterior probability times the solution's share of it (section 9.13); the solutions are sorted by p,
most probable first, and the five most probable are kept. Probabilities are relative to the compositions that
were tried: the single solution of a composition that was given shows none. A solution's "weight" `a_j / Σ a` is a normalised
amplitude, not a probability or a volume fraction.

**Values at a bound.** The least squares of Predict parameters keep every population inside the box of the network's
prior (section 9.13: R from 0.5/q_max to 5/q_min, σR/R, σh/h and σnn/D 0.02–0.5, D and h within the prior's D/2R and
h/2R at radii up to 1.5 times the start's, the tail's ν 1.8–2.2 and w at most 30 q_min). A free shape or tail value
within 10⁻⁴ of its range of a bound has *ended on the bound*; the solution names it in `at_bounds` (parameter, lower
or upper bound, component; the keys `sigma_Res` and `nu_Res` are the tail's w and ν, which every text calls
Tail w and Tail ν, the names of Fitting's Model table: `shared/fit_names.py` holds the one name of every quantity
and the sentence of every flag). With Tail A = 0 the tail is absent and its w and ν are not listed, nor flagged; the
same holds for the values of a population whose Scale is 0. A value on a bound is not determined by the
curve: the bound, not the data, holds it. These are the non-linear values only. The linear terms a_j, BG and A are
solved by non-negative least squares and are not in `at_bounds`: one at 0 is held by that limit (a background the
curve would want below zero, a beam tail that is absent, so that w and ν no longer change the curve), and the report
names it for the reported solution among the *Decisions* (*fit caveat*: *Background at 0, the non-negativity
limit*).

**The solution the automatic analysis reports** is the most probable one, the first. The fit record keeps its index
(`chosen`), the Results tab selects it, and `fit_curve.csv` holds its curve. The report's *Decisions* give it
(*fit*: model, the R of every population and the D of those with a paracrystal in the order of the radii,
`R = 3.06 / 6.02 nm, D = 49.4 / – nm`, χ²ᵣ and probability), the refined R, h and D of every population with their
68 % intervals (*fit intervals*: `R (1) 3.17 [3.04, 3.26] nm`, the refined value and then its interval — the
posterior given the σ of the points, sampled around the solution, or the least-squares errors when the sampling did
not converge; the geometry and the choice of model are not in it), the notes of the fit (the flags of section 9.13
as sentences, one per kind; those about the background, the beam tail and the resolution in one line after those
about the particles) and its values at a bound (*fit caveat*).
Further rules of the report (`gisaxs_choice.py`):

- Another solution with p ≥ 0.2: the curve alone does not decide the model (*Needs attention*, **model**). The point
  lists the probable solutions with their rank and probability and names those whose D is within 15 % of D* (two
  lengths that happen to agree, not evidence for the model, since D* is not necessarily a spacing, section 8.4).
- A value of the reported solution at a bound: *Needs attention* (**fit**) says that those values are not determined
  by the curve.
- D ≥ 250 nm in the reported solution: it has no interparticle correlation in the q range, and D is not determined by
  the curve.
- A reported solution with χ²ᵣ > 10 is a **poor fit** (*Needs attention*, **fit**): it misses the curve by far more
  than the σ of its points, so none of the compositions tried describes the cut and its R and D are not measurements.
  The limit is that of Fitting's poor fits of a series (χ²_red > 10; User Manual, section 6.5).

Every number of these texts is formatted once from the fit record (R, h, D to 4 decimals of nm), so a value reads the
same in the table and in *Needs attention* (48.3484 nm is 48.3 nm in both). **Save Fit Table…** writes one line per
population of each solution: rank, model, χ²ᵣ (`chi2`; a `#` line gives the fitted points), probability, log RMSE, the parameters (nm; the spreads relative), the
68 % intervals of R, h and D, `at_bounds` (`name:upper` or `name:lower`) and `chosen`.

**GALAXI.** On the public GALAXI frame (α_i = 0.463°, `galaxi_bornagain.poni`) the run reports an upright cylinder
with a paracrystal next to a second upright cylinder, R = 3.17 / 6.15 nm, D = 44 / – nm, χ²ᵣ = 1.07 over 596 of 598
points, probability 0.39; two more solutions have probabilities of 0.32 and 0.30, so *Needs attention* asks for the
model (**model**). The cut's broad log-normal sizes and its low-q rise are described by two populations; the sample's
particles are spheres (BornAgain's model, section 9.10), which a fit with a sphere in Model and the particles of
Model as the composition describes (section 9.13). The choice of model is the user's: the automatic analysis
describes the cut and gives Fitting a start.

**Until 0.2.0** the analysis fitted the cut with a quick numerical fit (no network; the same fit was Fitting's *Find
the particle shape (no AI)*, section 9.9): single-component spheres, vertical and random cylinders from fixed starts
(one with D = D*), ranked by a robust misfit (the mean Huber loss of r, Huber 1964), the reported solution the
best-ranked one free of bounds within 10 % of the first's χ²/N, and no ± for R, h or D. On 14 BornAgain simulations
of cuts of Ag spheres with known parameters (the GALAXI model with R, scale, D and disorder varied, Poisson noise at
the data's count level) it ranked a random cylinder first in 12 and a sphere in 1, 10 first-ranked solutions had a
value at a bound, the best sphere's R was a median 11 % above the R⁶-weighted mean radius (−29 % to +26 %), and D and
the disorder were not recovered. On the GALAXI frame it reported a random cylinder with χ²/N = 1.05 and D at its lower
bound.

Implementation: `create_curve_fit()` in `src/gimap/features/fitting/bootstrap.py` (Predict parameters as a function:
`domain/posterior/refine.py`, the rows of `infrastructure/adapters/posterior_job.py`), the choice and its texts in
`src/gimap/features/assistant/application/gisaxs_choice.py` and `gisaxs_procedure.py`, forward model `cut_model`
(`src/gimap/features/fitting/domain/cut_model/`)

### 8.6 What the GISAXS analysis does not model

- The cut is treated as a function of `abs(q_y)` only. q_z at the Yoneda row, reflection and refraction (DWBA) terms
  and the Yoneda enhancement itself are not modelled.
- The term `A / (1 + (q/w)^ν)` is an additive empirical tail of the direct and reflected beam; no resolution
  convolution is applied.
- The Yoneda row is found without αc; the beam-centre row is not refined; the centre has no uncertainty.
- Without a given composition the network chooses among one or two populations of spheres and upright cylinders;
  random cylinders and three populations are fitted only when given as the composition (section 8.5).

Like the GIWAXS report (section 7.9), the GISAXS report's *Decisions* always say what its numbers are:

- *q*: kinematic, no refraction correction. Near the critical angle refraction mostly shifts q_z, that of the Yoneda
  band (α_f ≈ α_c) most, by an amount of the order of k sin α_i, and the line gives that number for the frame
  (`2π/λ · sin α_i`; 0.0379 Å⁻¹ for the public GALAXI frame at α_i = 0.463°, λ = 1.34 Å); q_y and 2π/q* barely change.
  With the film's critical angle given (section 2.6) it reads *q_z corrected for refraction in the film (αc = …)*;
- *uncertainties*: statistical only — the σ of the cuts' points, by which χ²ᵣ is normalised, with the error model of
  the frame (Poisson counting statistics, or the pixel scatter of each bin); the geometry and refraction are not
  included. The 68 % intervals of R, h and D (*fit intervals*, section 8.5) are those of the posterior within the
  model given these σ: the choice of model is not in them.

## 9. Model fitting of 1D curves (Fitting)

Fitting fits one curve, or every frame of a series, with its own model. The Fit step has three methods: the
numerical *Refine the values in Model* (Refine) and *Search the ranges, then refine* (Search), described in this
chapter, and *Predict parameters*, GIMaP's network and the posterior of this model (section 9.13), which also fits
the cut of the GISAXS automatic analysis (section 8.5); the 1D Predict window is chapter 10. *Find the particle shape
(no AI)* and *Predict parameters (fit1d)* were removed in 0.2.0, so that Fitting has one network method; sections 9.9
and 9.12 keep what they were.

**Units: Fitting works in nm⁻¹ internally.** A curve read in Å⁻¹ (an Analyze `_fit_input.dat`) is multiplied by 10.
Sizes and distances are in nm, w in nm⁻¹, the spreads σ_R/R, σ_h/h, σnn/D are relative.

**The cut's q_z.** Predict parameters takes the cut's geometry from Model's grazing-incidence values (αi, αf and λ;
section 9.10): with both angles, spheres have the direct and the exit-reflected path, q₊ = k_i + k_f and
q₋ = k_i − k_f, with a free effective reflection coefficient r (the two paths of section 9.10, section 9.13). The 1D
Predict window takes its **Cut qz (nm⁻¹)** (chapter 10): spheres at the Born term of that q_z alone. Without either,
q_z = 0, the effective interface. Analyze's finite-width horizontal band is not automatically replaced by a single inferred q_z.

### 9.1 Points that are fitted

**Text-table input.** The import preview shows the header and the first ten rows of a `.dat` or `.txt` file.
The confirmed single-curve layout reads `q I [sigma]`; the shared-q layout reads `q I1 I2 …` as independent
curves, each paired with the first q column. Clear column names such as `q I1 I2` can suggest the second layout,
but an unlabelled three-column table is ambiguous and keeps the single-curve default until the user chooses
otherwise. Analyze's marked exports and its `q I sigma pixels` schema remain one curve; the sigma and pixel-count
columns cannot be imported as separate intensities. Named uncertainty columns (including `sigma1`, `sigma2`)
also disable shared-q splitting. Interleaved `q I1 sigma1 I2 sigma2 …` tables are not supported as multiple
curves; single-curve import can read their first q/I/sigma triple only. The unit stated in the file takes precedence over the
setting for files without a unit; contradictory unit declarations remain an error.
Explicit import refuses to use a named intensity column as sigma; a table such as `q I1 I2 sigma` must be
rearranged into a supported layout instead of silently using I2 as the uncertainty.

Reading keeps each selected q, I and optional sigma from the same row, then removes rows whose selected values
are missing, invalid or non-finite. A missing I1 does not remove a valid I2 at that q when the table retains its
column positions (for example, an empty comma-separated cell or `nan`). A shortened whitespace-separated row
cannot identify the missing column and is omitted instead of shifting the remaining values. Missing sigma
stays absent: importing a shared-q table does not synthesize counting errors. The legacy single-curve reader
also rejects a malformed sigma row as a whole; a later sigma can no longer become paired with an earlier q/I.
Valid legacy q/I/sigma values are unchanged.

Each imported curve keeps its physical source file, selected column indexes, original column names when present,
table width and whether sigma came from a column. Projects and fit records retain this mapping. A changed header
or missing selected column is reported and requires importing the table again; it is not silently remapped.

A curve with signed q (a horizontal GISAXS cut) is fitted on one of: the mean of both halves (6.16)–(6.17), both halves on
`abs(q)`, or one half. Then the fitting range and points left out apply. Of these points only `abs(q) > 0` with finite
values are kept. σ is used only when every kept point has a finite `σ > 0`; otherwise σ is dropped for the whole
curve, and `I > 0` is required.

Implementation: `src/gimap/features/fitting/application/curve_sources.py`, `src/gimap/features/fitting/infrastructure/adapters/curve_tables.py`, `src/gimap/features/fitting/infrastructure/adapters/scattering_curve_files.py`, `src/gimap/features/fitting/application/single_fit.py`, `src/gimap/features/fitting/domain/fit_engine.py`

### 9.2 The intensity model

```text
I(q) = BG + k · [ Σ_i Int_i · T_i(q) + A / ( 1 + (q / w)^ν ) ],   q = abs(q) in nm⁻¹        (9.1)
T_i = P_i · S_i  (LMA)   or   P_i + C_i (S_i − 1)  (DA),   P_i(0) = 1
```

Int_i is the scale of component i. Every family is normalised to P(0) = 1 (section 9.4), so Int_i is the intensity
of component i at q = 0 (times S_i(0)). BG is a constant background (not multiplied by k), A, w and ν the amplitude,
width and exponent of the beam-tail term, and k an overall factor fixed at 1 by default. By default the model depends on
`abs(q)` only: spheres are evaluated at q_z = 0, the effective interface approximation. Given the angles of the cut,
spheres can instead use the direct and the exit-reflected path of the DWBA (section 9.10). The beam-tail term is additive, not a resolution convolution. With w ≤ 0 or ν ≤ 0 the term is 1.

**q resolution (σ_q, off by default).** With σ_q > 0 each particle term `Int_i·T_i` is convolved with a Gaussian
of standard deviation σ_q in q; the beam-tail term and BG are not. The terms are made on a uniform grid from
max(0, q_min − 5σ_q) to q_max + 5σ_q, at most σ_q/5 apart, folded at q = 0 (the cut is symmetric: I(−q) = I(q)),
convolved with the Gaussian cut at ±5σ_q and interpolated back to the points. A test compares the result with a
direct convolution of the symmetric cut (within 10⁻³). Smearing matters for well-ordered films: when σ_q is above
about 0.3 of the width of the structure peak, the fitted σnn/D otherwise takes up the resolution as disorder.

The plot's **Components** are these additive intensities: `k·Int_i·T_i`, the scaled beam-tail term and
`BG`; their sum is the total model. They are not normalised population fractions. Component visibility and axis limits do not change the
fit, residuals or numerical exports. Zero and negative values are omitted only from logarithmic display.

**One forward model.** Fitting now computes (9.1) with `cut_model`, GIMaP's one implementation of a 1D cut,
which the proposal network's training and refinement are to use as well. Before, the same physics existed four times
with different conventions: Fitting's model, the quick fit of the automatic analysis (which used 1D Predict's
forward model; removed in 0.2.0), the fit1d kit's tables (removed in 0.2.0) and 1D Predict V5's own forward model. `cut_model` was checked against each of them, against an
independent implementation and against BornAgain (`tools/benchmarks/cut_model_checks.py` and `cut_model_bornagain.py`,
q = 0.02–3 nm⁻¹, the largest relative deviation over the curve):

| Reference | Cases | Largest deviation |
|---|---|---|
| `tools/benchmarks/yoneda_truth.py` (no shared code; its size rule refined to 1500 nodes) | spheres R 4–6 nm and upright cylinders, σR/R 0.2–0.3, paracrystal; Born at the cut's q_z and four-term DWBA on Si; LMA and DA | 2.7 · 10⁻⁵ (the tables of section 9.3; the direct quadrature 1.4 · 10⁻⁶) |
| fit1d's tables (sizes by number, Hosemann paracrystal) | spheres, upright and random cylinders, R 2–10 nm, Born at q_z; spheres in the four-term DWBA | 3.5 · 10⁻⁵ |
| 1D Predict V5's forward model | one size with a paracrystal | 6 · 10⁻¹⁴ |
| Fitting's earlier formulas on 20001 nodes | spheres R 2–20 nm, σR/R 0.05–0.4; DA; two paths | 2 · 10⁻⁶ to 1.8 · 10⁻³ (the fine rectangle rule's own error) |
| BornAgain 24, average layer materials off | spheres and upright cylinders on Si in the four-term DWBA, σR/R 0.2 (BornAgain gets the same radii and number weights) | 3.9 · 10⁻⁴ (sphere), 2.8 · 10⁻⁴ (cylinder) |

For one size BornAgain and `cut_model` agree to 10⁻³ (median), more only at the zeros of the form factor, where both
curves go to zero. BornAgain's default gives the layer holding the particles their volume-averaged refractive index,
which changes the reflection the DWBA terms carry: for spheres R = 4 nm on a paracrystal of D = 14 nm (about 17 % of
the layer) the curve then differs by 32 % (median). `cut_model`, like `yoneda_truth.py` (and fit1d), uses the reflection
of the bare substrate; the free effective r of the two-path model can take up such a difference (section 9.10). One
core computes 1000 curves of 500 points (R 1–15 nm, σR/R 0.05–0.4) in 1.4 s (spheres), 1.2 s (upright cylinders)
and 11 s (random cylinders).

**Conventions of the earlier models** (`cut_model/conventions.py` is the one place that converts them):

| Quantity | Fitting now | Fitting earlier | fit1d (until 0.2.0) | 1D Predict V5 and the quick fit (until 0.2.0) |
|---|---|---|---|---|
| sizes | Sizes of each component (section 9.3) | Gaussian by intensity (spheres, random cylinders), by number ±3σ (vertical cylinders) | Gaussian by number | Gaussian by intensity (all three) |
| spacing disorder | σnn/D | σD/D = (σnn/D)/√(2π) | w = σnn (nm) | σD/D, as earlier Fitting |
| vertical cylinder at q = 0 | P = 1 | 10⁻⁶ ⟨R⁴⟩/4 (R in nm) | P = 1 | P = 1 |
| low-q term | A / (1 + (q/w)^ν) | the same | C / (1 + (q/w)^ν), ν = 2 by default | relative to the particle curve (10.5) |

A model saved by an earlier version (`gimap_fit_model_v1`) is converted when it is read: σnn/D = √(2π)·σD/D (with its limits),
a vertical cylinder's Int times 10⁻⁶ ⟨R⁴⟩/4, and the sizes it had (Gaussian by intensity for spheres and random
cylinders, by number for vertical cylinders). Its curve then changes only by the size rule (section 9.3).

Implementation: `src/gimap/features/fitting/domain/cut_model/`, `src/gimap/features/fitting/domain/fit_model.py`

### 9.3 Size distributions

Each component's **Sizes** say what R and σR/R are (number weights n, intensity weights g; σ = R · σR/R):

```text
Gaussian, by number (default)    n(x) ∝ exp( −½ ((x − R)/σ)² ),   max(0, R − 4σ) ≤ x ≤ R + 4σ                 (9.2)
Log-normal, by number             n(x) ∝ exp( −½ ((ln x − μ)/s)² ) / x,   s² = ln(1 + (σR/R)²),  μ = ln R − s²/2  (9.3)
Gaussian, by intensity            g(x) ∝ exp( −½ ((x − R)/σ)² ),   n(x) = g(x)/V(x)²,   same range as (9.2)
P(q) = Σ_j n_j V_j² |A_j(q)|² / Σ_j n_j V_j²,     C(q) = | Σ_j n_j V_j A_j(q) |² / Σ_j n_j V_j²                  (9.4)
```

A_j is the amplitude of a particle of size x_j normalised to A(0) = 1 (section 9.4) and V_j its volume, so P(0) = 1
whatever the law. The Gaussian by number is the size distribution of `tools/benchmarks/yoneda_truth.py` (and of fit1d),
cut at ±4σ; the cut is part of the definition: weighted by R⁶ the upper cut removes about 1.5 % of a sphere's
intensity at σR/R = 0.35, more above. In the log-normal R and σR/R are again the number mean and relative standard
deviation; it is integrated from 4s below its median to 4s above the median of its V²-weighted form (a log-normal of
the same s with the median times exp(p s²), V² ∝ R^p), so its largest particles, which dominate the intensity of a
broad distribution, are not cut off. The Gaussian by intensity is Fitting's earlier distribution for spheres and
random cylinders: every size scatters the same at q = 0.

**Quadrature.** Composite Gauss–Legendre with 8 nodes per panel. A form factor squared oscillates in x with
period π/q, so the panels are the number of such periods across the distribution at the largest q of the curve,
rounded up to a power of 2, at least 2 (at most 4096).

**Large q·σ: the envelope.** Averaged over sizes that vary on a scale σ (σ = R·σR/R; for the log-normal the lower
end of its intensity-weighted part), the oscillating part of |F|² is damped by about e^{−2(qσ)²}. From
q_cut = 6/σ on it is negligible, and the average is that of the envelope of |F|², its average over one oscillation:
9(1 + x²)/(2x⁶) for a sphere (for two DWBA paths (9/(2ab)) Re[h₁(a)h₁(b)*] with the spherical Hankel function h₁),
2(J₁² + Y₁²)/x² for a disc and 2/x² for the height factor of a cylinder. Between q_cut and 2·q_cut the average goes over
smoothly (a smoothstep in q) from the resolved one to the envelope's, so the model stays a smooth function of R and σ,
and the rule only has to resolve 2·q_cut; the coherent term C fades to 0 with the same step (it is damped by
e^{−(qσ)²/2}). What the envelope leaves out is the ringing of the hard cut at R + 4σ of the Gaussian by number, an
artefact of the cut: compared with the fully resolved average the curve differs by at most 3 · 10⁻⁵ for σR/R ≤ 0.2,
10⁻³ at 0.35 and 10⁻² at 0.5 (tests). A Gaussian by intensity keeps much weight at small R; it is always fully resolved.

**Tables (the default path).** For sizes by number (a Gaussian or a log-normal with 0.02 ≤ σR/R ≤ 1) and one scattering
path (the effective interface, or Born at the cut's q_z), the averages come from small tables (`cut_model/tables.py`,
3.4 MB, built by `tools/ml1d/build_tables.py` and checked against a fresh build). Each form factor splits exactly into
a smooth part and a smooth amplitude times an oscillation, f(x)² = env(x) + Re[G(x) e^{iωx}] (ω = 2 for the sphere and
the disc, 1 for the height factor); averaged over sizes, ⟨env⟩/env(x̄) and ⟨G e^{iωx̄(t−1)}⟩/G(x̄) (t = R/R̄) are smooth in
ln x̄ (respectively u = q·σ) and ln(σR/R) and are interpolated with four-point Lagrange polynomials; the width grid
has a node at σR/R = 0.25, where the Gaussian starts to be cut at R = 0. The tables carry the definition above (resolved
below q_cut, the envelope beyond) and agree with the direct quadrature to 10⁻⁴ (typically 2 · 10⁻⁵); for broad
distributions (σR/R ≥ 0.45) they are closer to a fully resolved reference than the direct quadrature. One evaluation of
a sphere or an upright cylinder then takes about 1 ms at 750 points, whatever R and σ. Spheres with two paths
(A = c₁ f(aR) e^{−iQ₁R} + c₂ f(bR) e^{−iQ₂R}, a = √(q² + Q₁²), b = √(q² + Q₂²)) are the two Born terms from the
tables and a cross term X = ⟨f(aR) f(bR) e^{−i(Q₁−Q₂)R}⟩, P̂ = |c₁|²⟨f(aR)²⟩ + |c₂|²⟨f(bR)²⟩ + 2 Re(c₁c₂* X). With
f(x) = Re[A(x) e^{ix}], A(x) = −3/x² − 3i/x³ exactly, X is a sum of four terms e^{iωR̄} times exact powers of aR̄ and bR̄
times the moments m_k(ν) = ⟨n t^{6−k} e^{iν(t−1)}⟩/⟨n t⁶⟩ (k = 4, 5, 6, ν = ωR̄, ω = ±a ± b − (Q₁ − Q₂)), which depend
on ν and σR/R only; they are tabulated on the nodes of the direct quadrature (3.6 MB more) and interpolated in the
same way. As in the direct quadrature, the terms the envelope of the pair leaves out (ω ≈ ±(a + b) − (Q₁ − Q₂)) fade
with the smoothstep of the in-plane q, and below bR̄ = 2 X is taken on the nodes. A row comes from these tables
when the direct quadrature would need 16 panels (128 nodes) or more; on fewer the direct quadrature is as fast and
is kept. On random rows (R from 0.3 to 400 nm, σR/R 0.02–0.8, both laws, k_i above and below k_f; 800 rows) the
result differs from the direct quadrature by at most 1.4 · 10⁻³ of the curve's largest value (99 % within
4.2 · 10⁻⁵; the rows left to the direct quadrature not at all); with the tables for every row, by 3.5 · 10⁻⁶
(median), 99 % within 4.4 · 10⁻⁴ and up to 6 · 10⁻³ where |r| ≈ 1 with a phase near π and the two paths cancel to
10⁻³ of either (there 10⁻⁵ of each path). Two paths are 23 times faster on these rows, 1.6–2 times on small
spheres of moderate width (R 2–10 nm, σR/R 0.1–0.35) and up to 130 times on broad log-normal distributions. Narrower distributions
(below x̄ = qR̄ = 2 too), sizes by intensity, the DWBA, two paths in the decoupling approximation and the coherent term
at q_z ≠ 0 use the direct quadrature.
Earlier, Fitting used rectangle rules of fixed size (25 radii of a sphere, 13 radii and 13 heights
of a random cylinder, 26 radii over ±3σ of a vertical cylinder). They alias at large q·R·σ: on q ≤ 3 nm⁻¹ a sphere
of R = 5 nm and σR/R = 0.2 is off by up to 1.8 %, R = 10 nm and 0.3 by 26 %, R = 20 nm and 0.4 by 67 %
(`tools/benchmarks/cut_model_checks.json`). Fits of large or broad particles to high q therefore change with this version.

**By number and by intensity.** After a fit the results give each particle's mean radius and relative spread both
ways (*Sizes by number | by intensity*; `derived` in the exported record: `R_number_mean_nm`, `sigma_R_number_rel`,
`R_intensity_mean_nm`, `sigma_R_intensity_rel`), weighted by V² ∝ R⁶ for spheres and R⁴ for cylinders. Microscopy
counts particles; the curve is dominated by the intensity. For the log-normal the moments are exact: by intensity it
is a log-normal with the same width and the mean R (1 + (σR/R)²)^p.

The number distribution of a Gaussian by intensity is G(R)/R⁶ for spheres. It has a maximum only for
σR/R < 1/√24 ≈ 0.20 (1/√16 = 0.25 for cylinders); above that it rises towards small R, and its mean would be set by
where the distribution is cut. GIMaP then writes *not defined*: the condition is that the largest number weight lies
inside the range and the smallest node holds less than 10 % of it. A broad intensity-weighted sphere distribution
therefore does not say how many small particles there are.

New components have sizes by number; models saved in earlier versions keep the sizes they had (section 9.2).

Implementation: `src/gimap/features/fitting/domain/cut_model/sizes.py`, `src/gimap/features/fitting/domain/fit_model.py`
(`number_size`, `intensity_size`)

### 9.4 Form factors as coded

**Sphere** (amplitude normalised to F(0) = 1; series `1 − x²/10 + x⁴/280 − x⁶/15120` for x below 0.05; V ∝ R³):

```text
F_s(x) = 3 ( sin x − x cos x ) / x³,   x = q R                                                      (9.5)
```

**Random cylinder** (homogeneous cylinder of radius R and height h, isotropic orientation, V ∝ R² h; R and h have
independent distributions):

```text
F_c(q, R, h, α) = [ 2 J_1(q R sin α) / (q R sin α) ] · sinc( q h cos α / 2 ),   sinc(u) = sin u / u     (9.7)
P_cyl(q) = ∫₀¹ du  ⟨R⁴ (2 J_1(q R √(1−u²)) / (q R √(1−u²)))²⟩ / ⟨R⁴⟩  ·  ⟨h² sinc²(q h u / 2)⟩ / ⟨h²⟩,   u = cos α   (9.9)
```

⟨ ⟩ are averages by number (9.4). The orientation average is taken in α on [0, π/2] with weights sin α (u = cos α
is uniform for an isotropic orientation; q sin α and q cos α are smooth in α, whereas in u the radius argument
q√(1 − u²) oscillates ever faster towards u = 1), by composite Gauss–Legendre with one panel per period
π/(q · max(R, h/2)). The radius average depends on q sin α only and the height average on q cos α only: each is made once
on a uniform 1D grid with 64 points per period of the largest size and taken at every (α, q) by four-point
interpolation; with the envelope of section 9.3 the grids and the orientation rule end at 2·q_cut, and beyond it the
envelope averages are taken directly. For one size the result agrees with adaptive quadrature to 10⁻⁵. With sizes by number the radius and height averages on the grids come from the tables of
section 9.3, the orientation rule then follows the characteristic size of the intensity-weighted distribution; a
random cylinder takes 15–200 ms to evaluate once (a broad log-normal up to 0.8 s), much more than a sphere. Earlier the orientation average was a 24-point rectangle rule in α with sin α weights; it is off
by up to 18 % (R = 3 nm, h = 10 nm) and 55 % (R = 5 nm, h = 30 nm) at q ≤ 3 nm⁻¹.

**Vertical cylinder** at q_z = 0 (cylinders standing on the substrate, q in the plane; V ∝ R² at a fixed height):

```text
P_vc(q) = ⟨R⁴ (2 J_1(q R)/(q R))²⟩ / ⟨R⁴⟩,   P_vc(0) = 1                                                (9.10)
```

The height drops out: the q_z-dependent factor of a cylinder is the same for every radius and every DWBA path
(section 9.10). Earlier, Fitting's vertical cylinder was not normalised, `P_vc = 10⁻⁶ ⟨R⁴ (2 J_1(qR)/(qR))²⟩/4`
(R in nm; 1.56 · 10⁻⁴ at R = 5 nm, 1.94 · 10⁻⁴ with σR/R = 0.2), so its Int was in other units than a sphere's; a model
saved then is converted (section 9.2).

In the two-path or DWBA geometry the sphere's amplitude is the sum of its paths (9.29, 9.27). Random cylinders are
averaged at |q| = √(q² + q_z²): with q_z = 0 in the effective interface and the two-path model (as in earlier versions), with the
cut's vacuum q_z in the Born and DWBA geometries of `cut_model` (as fit1d did).

Implementation: `src/gimap/features/fitting/domain/cut_model/shapes.py`

### 9.5 Structure factor: one-dimensional paracrystal

For a component whose structure factor is switched on, with `D > 0` and `σnn/D > 0` (Hosemann & Bagchi, 1962; for
GISAXS see Renaud et al., 2009):

```text
S(q) = ( 1 − p² ) / ( 1 − 2 p cos(q D) + p² ),   p = exp( −q² σ_nn² / 2 ),   σ_nn = (σnn/D) · D             (9.12)
```

σnn/D is the relative standard deviation of the distance between neighbours, a Gaussian of mean D; fit1d (section 9.12) called it w/D.
S is evaluated as `−expm1(2 ln p) / (expm1(ln p)² + 4p sin²(qD/2))`, without cancellation at small q, and
`S(0) = (σnn/D)²`. Otherwise S = 1.

In **earlier versions** Fitting's parameter was σD/D with `p = exp(−π q² σ_D²)`, σ_D = (σD/D)·D: the same function for

```text
σ_nn = √(2π) · σ_D,   i.e.   σnn/D = √(2π) · σD/D ≈ 2.507 · σD/D                                       (9.13)
```

(identical within 10⁻¹⁴). A model saved by an earlier version is converted when read, its limits too. Example (invented):
σD/D = 0.2 of an old model is σnn/D = 0.50.

D is kept at or above 2R by the fits (section 9.6).

**Coupling: LMA or DA.** With **Coupling** of a card (shown when Distance D is on) the paracrystal enters in one of
two ways:

```text
LMA:  I = P(q) · S(q)                                                                              (9.14a)
DA:   I = P(q) + C(q) (S(q) − 1)                                                                   (9.14b)
```

- LMA (the default, and the only one before 0.2) takes every size among neighbours of its own size. DA takes sizes
  and positions as independent.
- C = |⟨VA⟩|²/⟨V²⟩ is the coherent term of (9.4), the amplitude averaged by number (random cylinders also over
  orientation). A test checks the sphere against a direct average by number (10⁻⁶).
- With one size, DA and LMA are the same curve for spheres and vertical cylinders; randomly oriented cylinders differ,
  their orientation being a disorder of its own. With a size spread, DA keeps the part `P − C` free of the spacing:
  more intensity below the structure peak, a weaker peak. At σR/R = 0.2 and σnn/D = 0.75, the low-q intensity of
  spheres is about 1.5 times that of the LMA with a Gaussian by intensity, 1.2 times with a Gaussian by number.
- This is the part that a same-size population without a structure factor takes up in a three-population fit
  (section 9.11). On the simulated films of 9.11 (spheres R = 4 nm, four-term DWBA, resolution, beam tail and
  roughness; main population and a large one, low-q term free; fitted with Fitting's earlier model), a film made in
  the DA is fitted with χ²ᵣ ≈ 2.5 in DA and 8.3 in LMA; a film made in the LMA with 11 in LMA and 32 in DA. The curve
  therefore tells the two apart. A third, same-size population brings the DA film to 1.1–1.4 in LMA as well, because
  it also takes up the difference of the sphere's shape (section 9.10). D and σnn/D stay within 0.3 % and 0.005 in
  every case.
- fit1d (section 9.12) used the LMA.

Implementation: `src/gimap/features/fitting/domain/cut_model/structure.py`, `src/gimap/features/fitting/domain/cut_model/model.py`

### 9.6 Parameters and defaults

| Key | Label | Unit | Default | Range | Free |
|---|---|---|---|---|---|
| Int | Scale | – | 1 | ≥ 0 (solved, section 9.7) | yes |
| R | R | nm | 5 | 0.3–500 | yes |
| sigma_R | σR/R | – | 0.2 | 0.01–1.0 | yes |
| h | h | nm | 10 | 0.5–1000 | yes |
| sigma_h | σh/h | – | 0.2 | 0.01–1.0 | yes |
| D | D | nm | 30 (a new component: max(3R, 1)) | 1–3000 | yes |
| sigma_nn | σnn/D | – | 0.25 | 0.01–1.0 | yes |
| background | Background (BG) | – | 0 | ≥ 0 (solved) | yes |
| res_amplitude | Tail A | – | 0 | ≥ 0 (solved) | yes |
| res_width | Tail w | nm⁻¹ | 0.02 | 0.001–0.1 | yes |
| res_exponent | Tail ν | – | 3 | 1–20 | yes |
| k | Factor k | – | 1 | 10⁻⁶–10⁶ | no |
| q_resolution | σ_q | nm⁻¹ | 0 | 0–0.1 | no |
| alpha_i | αi | ° | 0 | 0–5 | no |
| alpha_f | αf | ° | 0 | 0–5 | no |
| wavelength | λ | nm | 0.1 | 0.005–1 | no |
| refl_amp | \|r\| | – | 0 | 0–1 | no |
| refl_phase | arg r | rad | 0 | −π–π | no |

Each component has Int, R, σR/R (h and σh/h for random cylinders) and, with its structure factor, D and σnn/D; its
**Sizes** (Gaussian by number, the default; log-normal by number; Gaussian by intensity) and, with D, its **Coupling**.

**D is at least 2R.** A fit keeps the distance of a component's paracrystal at or above twice its mean radius, so
particles of the mean radius do not overlap. With R and D both free, D is searched between max(D_min, 2R) and
D_max at every R; with R fixed, D's minimum is raised to 2R; with D fixed, R's maximum is lowered to D/2. A D that
ends at 2R is reported at its lower bound (section 9.8): the particles touch, and the curve does not hold D there.
Values that are not fitted are left as they are. For a random cylinder 2R is the smallest in-plane size, so the
limit is a minimum, not its contact distance. Before 0.2, D could fall below 2R (on the GALAXI cut of 9.10 to 1 nm).

Implementation: `src/gimap/features/fitting/domain/fit_model.py`

### 9.7 Objective and the scales

**Residuals.**

```text
r_i = ( I_i − m_i ) / σ_i                         with σ                                          (9.15)
r_i = ln I_i − ln max( m_i, 10⁻³⁰⁰ )               without σ (each point the same relative weight)    (9.16)
```

m_i is the model (9.1). The fit minimises Σ r_i².

**Scales by bounded least squares (variable projection).** The intensity is linear in Int_i, BG and A. Those of
them that are free are eliminated at every evaluation of the model (Golub & Pereyra, 1973). With the non-linear values
set, the columns `k P_i S_i`, `1` and `k / (1 + (q/w)^ν)` form the matrix X, the fixed terms the vector m_fixed, and

```text
c = argmin_{l ≤ c ≤ u} ‖ W ( X c − (I − m_fixed) ) ‖²,   W = diag(1/σ_i), or diag(1/I_i) without σ        (9.17)
```

Solved with bounded linear least squares (`scipy.optimize.lsq_linear`, BVLS, `tol=1e-10`, at most 100 times the number of free linear terms iterations). Columns are normalised by their norms to reduce sensitivity to scale differences. l and u are the Model limits, default [0, ∞). Without σ, (9.17) still minimises relative residuals while the outer objective is logarithmic (9.16): this remains an approximate profile, not exact elimination under the log likelihood.

Implementation: `src/gimap/features/fitting/domain/fit_engine.py`

### 9.8 Refine, Search, goodness of fit and the errors

**Coordinates.** Every free non-linear parameter needs a finite range [l, u] and is mapped to t ∈ [0, 1]:

```text
x = l + t (u − l)                                                     (linear)                     (9.18)
x = exp( ln l + t (ln u − ln l) )            when l > 0 and u ≥ 10 l   (logarithmic)                 (9.19)
```

With the default ranges of 9.6 every non-linear parameter is searched on a log scale.

**Refine** (`method = local`). Bounded least squares, trust-region reflective (Branch et al., 1999;
`scipy.optimize.least_squares`), on offsets from the start in the unit cube, bounds [−t_0, 1 − t_0], `x_scale = 0.02`
(the first steps are about 2 % of each range), `ftol = xtol = 10⁻⁶`, `gtol = 10⁻⁸`. With p free parameters and n_nl of
them non-linear, the budget is B = 150 (p + 1) evaluations, unless *Evaluations at most* is set, and

```text
max_nfev = max( 5, ⌊ max(20, B) / (n_nl + 1) ⌋ )                                                   (9.20)
```

so that with the finite-difference Jacobian a start uses about B evaluations. The model returned is the best one
evaluated during the whole run, not the last iterate. "Converged" means the solver ended on a convergence criterion.
The starting curve competes too, by the reported objective. When it beats every curve the fit tried by more than a
fraction 10⁻⁶ of its Σr² — without σ the scales are solved with relative weights while the objective is ln I, so a
start whose scales suit ln I can stay ahead — the start values are returned unchanged and the result says so
(`start_kept`, *did not improve on the start values*): it is not called converged, and Refine and Search give no
errors for it, since no minimum was found. Refine and Search refuse a curve with no more points than free values
and one without an intensity above zero. With σ in the file, points whose σ is zero, negative or not a number are
left out and counted (`sigma_invalid` among the curve's notes); σ is dropped only when no point has one.

**Search** (`method = global`). Differential evolution (Storn & Price, 1997; `scipy.optimize.differential_evolution`)
over the unit cube of the non-linear parameters, then Refine from the best three distinct members. With the default
budget B = 4000:

```text
population = max( 8, 5 n_nl ),   initialised by a Latin hypercube with seed 1729, member 0 = Model   (9.21)
generations = max( 1, ⌊ (3B/4) / population ⌋ − 1 )                                                (9.22)
```

`polish = False`, `tol = atol = 0` (all generations run), `updating = immediate`, seed 1729, and SciPy's default
strategy and mutation. The three starts are the best members whose ∞-norm distance from each other in the cube
exceeds 0.05 (fewer when fewer are distinct); the last quarter of the budget is shared among them, each start getting
`max(20, ⌊(B/4)/n_starts⌋)` evaluations through (9.20). With fixed seed and budget, Search
is deterministic.

**Reduced χ².** With N points and p free parameters, including the scales solved by bounded least squares:

```text
χ²_red = Σ_i r_i² / ( N − p )                                                                          (9.23)
```

With σ this is the usual reduced χ². Without σ it is the variance of ln I about the model, corrected for degrees of
freedom. The logarithmic error `logRMSE = √( mean( ln(I_i/m_i)² ) )` over the positive points is reported too.

**Parameter errors.** At the solution, the Jacobian J of the residuals with respect to *all* free parameters
(non-linear and scales, the scales held at their solved values) is taken by central differences with step 10⁻⁴ × the
parameter's size (its absolute value, or its range width when the value is 0). The columns are scaled by the same size
before the inversion:

```text
G = diag( size_1 … size_p ),   s² = Σ_i r_i² / ( N − p )                                            (9.24)
C = G [ (J G)ᵀ (J G) ]⁺ G · s²      ( = (JᵀJ)⁻¹ s² when JᵀJ is invertible )                          (9.25)
σ_k = √( C_kk ),   ρ_kl = C_kl / √( C_kk C_ll )                                                       (9.26)
```

⁺ is the pseudo-inverse. The errors are linearised (Wald) 1σ errors, not profile intervals. They are **always scaled
by s² = χ²_red**, so they also shrink when χ²_red is below 1, unlike the peak fits of chapter 7, which only inflate (7.12).
Pairs with `abs(ρ) ≥ 0.95` are listed as strongly correlated. A non-linear parameter within 10⁻³ of either end of its
(log-)unit range, and a scale at either of its limits, is listed as at a bound; its error is not meaningful.

Implementation: `src/gimap/features/fitting/domain/fit_engine.py`

**Values at the edge of their range.** Model's fixed values and limits are hard constraints. Refine and Search report boundary diagnostics without widening ranges or fixing free parameters automatically. The returned curve has the smallest residual sum of squares among the feasible starting curve and the evaluated curves, so refinement cannot replace a better feasible start. The values still at an edge are named under **Fit details…**
(*… is still at the … edge of its range: the curve does not determine it beyond there*); there is no option to widen
the ranges automatically.

Implementation: `src/gimap/features/fitting/domain/fit_engine.py`, `src/gimap/features/fitting/domain/fit_edges.py`,
`src/gimap/features/fitting/application/single_fit.py` (`fit_resolving`)

### 9.9 Find the particle shape (removed in 0.2.0)

Until 0.2.0 the Fit step had a fourth method, *Find the particle shape (no AI)*: the quick numerical fit that the
GISAXS automatic analysis used then (a variable-projection fit of sphere, vertical cylinder and random cylinder from
the starts R = 1.5 nm and 4.0 nm, and R = 4.0 nm with Model's D; at most 5 solutions, sorted by Fitting's χ²_red), its
solutions converted into Fitting's model (Int solved again against the solution's own curve, the largest relative
deviation reported). It and *Predict parameters (fit1d)* (section 9.12) were removed in 0.2.0, so that Fitting has
one network method, *Predict parameters* (section 9.13), beside Refine and Search. The automatic analysis fits its cut
with Predict parameters as well (section 8.5); a solution of it sent with **Show in Fitting** is taken over as it is
(its values are in Fitting's own conventions, its amplitudes become the Int). `native_solution.py` still converts the
quick fit's solutions of records made before 0.2.0.

Implementation (now): `src/gimap/features/fitting/application/single_fit.py` (`model_from_solution`,
`cut_model_solution`), `src/gimap/features/fitting/domain/native_solution.py`

### 9.10 Spheres and cylinders in a horizontal cut: the effective interface approximation

**The full expression needs the substrate.** For particles sitting on a surface, the distorted-wave Born
approximation (DWBA) adds the waves reflected by the surface before and after the scattering:

```text
A(q_y) = F(q_y, k_i+k_f) + r_i F(q_y, k_f−k_i) + r_f F(q_y, k_i−k_f) + r_i r_f F(q_y, −k_i−k_f)      (9.27)
k_i = k sin α_i,   k_f = k sin α_f,   F(q_∥, Q) = ∫ V e^{−i q_∥·r_∥} e^{−iQz} dr
```

r_i and r_f are the reflection coefficients of the surface at α_i and α_f. They depend on the whole layer stack
under the particles (polymer film, oxide, roughness, the particle layer itself), which is usually not known.
Example: at the Si critical angle, r_f = 0.82 − 0.15i on bare Si and −0.64 − 0.29i on Si covered with 100 nm
polystyrene (λ = 0.1045 nm).

**What GIMaP does by default.** Fitting (and Predict parameters without the angles in Model or *Cut qz*) uses the effective interface approximation
(EIA) of the Müller-Buschbaum group: the transmission factors are constant along a horizontal cut, so they go into
the scale, and the form factor is taken at q_z = 0. No substrate information is needed.

**Why upright cylinders are exact.** For a cylinder standing on the surface the amplitude separates:

```text
F(q_∥, Q) = πR²H · [2J₁(q_∥R)/(q_∥R)] · sinc(QH/2) e^{−iQH/2}                                    (9.28)
```

All four terms of (9.27) share the in-plane factor 2J₁(q_∥R)/(q_∥R). The Q-dependent factor is the same at every
point of the cut, so the cut has exactly the shape of the q_z = 0 form factor, on any substrate. This holds when
the heights do not vary with the radius; a height that grows with R would weight the radii differently.

**Why spheres are an approximation.** A sphere has no such separation: its form factor depends on
|q| = √(q_y² + Q²), and the four terms have different Q. The two that matter at the Yoneda angle (|r_f| ≈ 0.8,
|r_i| ≈ 0.04 at α_i = 0.4°) are q₊ = k_i + k_f and q₋ = k_i − k_f. Setting Q = 0 changes the positions of the
minima, and the radius moves to compensate. The error grows with q₊R. A hemisphere at q_z = 0 has exactly half
the amplitude of the full sphere, so the EIA cannot tell them apart; R is the in-plane radius.

Simulated cuts (four-term DWBA, checked against BornAgain 24.1), λ = 0.1045 nm, α_i = 0.40°, cut at the Si Yoneda
angle α_f = 0.150° (q₊ = 0.577 nm⁻¹), σ_R/R = 0.15, D = 3.2R, σ_nn/D = 0.2, 10⁵ counts at the peak, fitted in the EIA
with the right family:

| Particles | R (nm) | q₊R | R error, bare Si | R error, 100 nm PS on Si |
|---|---|---|---|---|
| upright cylinder, H = 6 nm | 4 | – | 0.0 % | 0.0 % |
| sphere | 2 | 1.15 | −0.4 % | +1.9 % |
| sphere | 3 | 1.73 | +1.4 % | +5.0 % |
| sphere | 4 | 2.31 | −4.0 % (χ²ᵣ 31) | −1.6 % (χ²ᵣ 52) |
| sphere | 6 | 3.46 | −39 % | −33 % |
| hemisphere | 2 | 1.15 | −1.1 % | −4.1 % (χ²ᵣ 1.0) |
| hemisphere | 3 | 1.73 | −1.0 % | −7.3 % (χ²ᵣ 1.5) |
| hemisphere | 4 | 2.31 | −5.1 % | −16 % |
| hemisphere | 6 | 3.46 | −24 % | −43 % |

- Up to q₊R ≈ 1.7 for spheres and ≈ 1.2 for hemispheres the EIA radius is within about 5 %.
- Beyond that the error depends on the unknown substrate, sign included, so it cannot be corrected afterwards.
  χ²ᵣ usually shows it, but not always (hemispheres of 2 nm on PS: −4 % at χ²ᵣ = 1.0; of 3 nm: −7 % at 1.5). For
  hemispheres the polymer film sets the limit: on bare Si they stay within 3 % up to q₊R ≈ 2.0.
- D and σ_nn/D stay within 1 % in every case: the spacing comes from the structure factor, not from the form factor.
- With α_i and α_f entered, Fitting shows q₊R of every sphere under the result, as information.

**Two paths with a free reflection coefficient.** In *Spheres in grazing incidence* (Model step), α_i, α_f and λ
of the cut switch spheres to the two terms that dominate:

```text
A = f(√(q²+q₊²)R) e^{−iq₊R} + r f(√(q²+q₋²)R) e^{−iq₋R},   r = |r| e^{i arg r}                       (9.29)
|A|² = f₊² + |r|² f₋² + 2|r| f₊ f₋ cos(arg r + 2k_f R)
```

r is a free effective coefficient, so the substrate is still not needed; only the measured angles are. Free |r|
and arg r to fit them. On the simulated cuts above it brings the radius of spheres and hemispheres with q₊R ≥ 2
back to within a few percent on both substrates, and also on a dense particle film whose own layer changes the
reflection. Its limits:

- Below q₊R ≈ 1.5 the two paths differ too little: r and R trade against each other (R off by +15 % to +38 % at
  R = 2 nm). Use the EIA there.
- Two settings of r can describe a cut equally well with radii about 5 % apart. Refine from the EIA radius, and
  compare a second start with arg r shifted by π.
- r is an effective number: it also takes up r_i, the particle layer and the phase convention. It is not the
  reflectivity of the substrate (a dense film needed |r| > 1; the default range stops at 1).
- With the angles entered and |r| fixed at 0 the sphere is the Born term at the vacuum q_z. That is not better than
  q_z = 0: on Si it is worse from R = 4 nm, because the exit-reflected path pulls the effective q_z below q₊.
- Upright cylinders do not change (9.28); random cylinders stay at |q|.

Implementation: `src/gimap/features/fitting/domain/fit_model.py` (`grazing`, `qz_plus_r`),
`src/gimap/features/fitting/domain/scattering_model.py` (`sphere_two_path_pd`)

**A measured cut: GALAXI against BornAgain's model of it.** BornAgain's example of the public GALAXI frame (r23,
`hardcode_galaxi.py`) describes it by Ag spheres 18.5 nm deep in HMDSO on PTFE on Si, log-normal radii (median
5.75 nm, σ = 0.4), a radial paracrystal (D = 53.6 nm, disorder 10.5 nm) with size–spacing coupling κ = 17.5, in
the DWBA of the layer stack. In r23 the size distribution is commented out; with it, BornAgain 24.1 describes the
Yoneda cut of Analyze with one scale and one background at χ²_red = 2.1 above q = 0.3 nm⁻¹, and without it at 22
(its monodisperse spheres have a minimum at 0.78 nm⁻¹ the cut does not have).

- Fitting describes the measured cut with a sphere and a paracrystal at χ²_red ≈ 1.1 (EIA or two paths, LMA or
  DA): R = 9.4 to 11.5 nm, σ_R/R = 0.17 to 0.35. On BornAgain's model of the frame, simulated on the same cut
  with the counts of the measurement, it finds the same: R = 8.5 to 10.0 nm, σ_R/R = 0.18 to 0.35 (one of 12 fits,
  with two paths, at 23 nm). Fitting reads the measured cut as it reads the model of it.
- These R are not a mean of the log-normal: by number it is 6.2 nm, weighted by R⁶ 16.3 nm. A Gaussian on
  intensity weights fitted to a broad log-normal lies between them; σ_R/R is close to the log-normal's relative
  spread, 0.42.
- D is not determined by this cut, measured or simulated: it shows no correlation peak, and the fitted D ends
  anywhere from about 200 to 3000 nm, on the simulated cut from 74 to 2500 nm (before D ≥ 2R of section 9.6, also
  at 1 to 7 nm, overlapping particles). With D and the disorder held at BornAgain's values, χ²_red rises to
  6 to 22: the paracrystal of (9.12) has no size–spacing coupling, which damps BornAgain's S strongly at κ = 17.5.

The scripts are `physics/galaxi_bornagain.py` and `physics/galaxi_ba_model.py` of the GIMaP paper's materials.

### 9.11 The three-population model

**The habit.** Yoneda cuts of nanoparticle films are often fitted with three populations of upright cylinders,
a large, a medium and a small one, usually with a paracrystal on the main population.

**Why three.** The cut spans about two decades in q, and each part is dominated by something else:

- low q: the tail of the direct and reflected beam, roughness, aggregates;
- around the structure peak: the particles and their spacing;
- high q: the decay of the form factor.

One population per window lets each window be described without pulling the others. The fit converges, and the
residuals become flat. Some films do contain several sizes (primary particles and clusters in sputtered or sprayed
films), but a good three-population fit does not show that by itself.

**What each population takes up.** Simulated film: spheres R = 4 nm on Si (σ_R/R = 0.15, D = 12.8 nm,
σ_nn/D = 0.2), four-term DWBA, resolution 0.005 nm⁻¹; fitted with three upright cylinders in the EIA (Search,
then Refine):

| Population | What it became | What it means |
|---|---|---|
| main, with structure factor | D within 0.2 %, σ_nn/D within 0.01; R 4–9 % low | spacing and disorder are measurements; R carries the shape difference (cylinder for sphere, 9.10) |
| same size, no structure factor | appears when the true film follows the decoupling approximation or has truncated particles; takes 40–60 % of the intensity below 0.15 nm⁻¹ | the part of the same particles' scattering that the spacing does not modulate (⟨\|F\|²⟩ − \|⟨F⟩\|², from size spread and imperfect order); not a second kind of particle. *Coupling* DA of the main population describes it directly (section 9.5) |
| large, no structure factor | R of tens to hundreds of nm or σ_R/R at its limit of 1; takes the low-q upturn | roughness, beam tail, aggregates; its R has no meaning; with the low-q term free it is mostly redundant |
| small, narrow | R ≈ 2 nm with a few % of the intensity even though the film has no small particles | compensates the form-factor shape at high q (q_z, DWBA, particle shape); real small clusters would look the same |

**How to read a three-population fit.**

- Take D and σ_nn/D of the population with the structure factor as results.
- Take its R with the shape caveat of section 9.10.
- Take the other two populations as descriptions of their q windows. Read them as particles only with independent
  support: SEM or AFM, the same population at another α_f, or a size that evolves smoothly in a series.

### 9.12 fit1d (removed in 0.2.0)

Until 0.2.0 the Fit step and the 1D Predict window offered *Predict parameters (fit1d)*, a separate kit
(`utils/gisaxs1d_gui_kit`, about 62 MB) with a model of its own: a network proposed starting values, a least-squares
fit refined them. It was removed in 0.2.0, with *Find the particle shape* (section 9.9), so that Fitting has one
network method, *Predict parameters* (section 9.13). Its record stays here because section 9.13 is compared with it;
the kit remains in the history of the repository.

**The model.**

```text
I(q) = Σ_j A_j F_j(q) S_j(q) + B + C / (1 + (q/w)^ν),      A_j, B, C ≥ 0                            (9.30)
```

Up to four populations, each a sphere evaluated at |q| = √(q_∥² + q_z²) with the q_z of the cut (*Cut qz*; q_z = 0:
the EIA of section 9.10), optionally in a four-term DWBA on Si, or an upright cylinder evaluated at q_∥; sizes a
number-weighted Gaussian cut at ±4σ; S_j the Hosemann paracrystal in the LMA with spread w = (σ_nn/D)·D; the amplitudes
by non-negative least squares (variable projection); ν fixed at 2 unless freed. Its network (eight heads of starting
values, trained on 2.05 million curves of (9.30); `general` and `au_np_si`) ran in numpy. The best heads were refined by
Levenberg–Marquardt, with a robust search when a check fired; the errors were the local covariance scaled by
max(1, χ²/dof). With *auto*, every candidate composition was fitted and ranked by

```text
BIC = χ² + k ln N                                                                                    (9.31)
```

(k free parameters, N points); compositions within ΔBIC ≤ 10 of the lowest were reported as indistinguishable.

**Benchmark.** 200 curves fit1d had never seen, made with an independent forward model (`yoneda_truth.py`, no shared
code): one sphere, one upright cylinder, two spheres, or a sphere and a cylinder; R 1.2–10 nm, σ_R/R 0.1–0.35, D/2R
1.3–2.5, σ_nn/D 0.1–0.3, 10³–10⁵ counts at the peak, q 0.02–2 nm⁻¹; half with fit1d's own physics, half with a
resolution of 0.005 nm⁻¹, a beam tail and roughness. The true composition given, median / 90th percentile over all
populations:

| | network alone | after the fit |
|---|---|---|
| R (relative) | 14 % / 48 % | 0.7 % / 13 % |
| D (relative) | 3 % / 33 % | 0.1 % / 0.3 % |
| σ_R/R (absolute) | 0.06 / 0.18 | 0.006 / 0.08 |
| σ_nn/D (absolute) | 0.011 / 0.049 | 0.001 / 0.005 |
| time per curve (one core) | 0.12 s | 3.5 s |

On the cuts with fit1d's own physics 61–70 % of the values lay within their 1σ, with the beam tail and roughness
26–43 %; *auto* (up to two populations) found the true composition in 69 % and 30 % of the cuts.

**GALAXI.** On the Yoneda cut of the public GALAXI frame and on BornAgain's model of it (section 9.10: log-normal
radii, number mean 6.2 nm, relative spread 0.42; radial paracrystal D = 53.6 nm, σ_nn/D = 0.196), simulated with the
counts of the measurement (three noise draws; `tools/benchmarks/galaxi_fit1d.py`):

| | one sphere, Born at the cut's q_z | one sphere, DWBA (Si substrate) |
|---|---|---|
| BornAgain's model: χ²/N | 6.1–6.5 | 1.7–1.8 |
| R by number, σ_R/R | 4.9 nm, 0.15–0.16 | 4.2 nm, 0.50 (at its upper limit) |
| D ± 1σ, σ_nn/D | 56.9–57.6 nm, 0.41–0.42 | 53.8–55.1 ± 1.3 nm, 0.50 (at its upper limit) |
| measured cut: χ²/N, R, D | 14.4, 5.2 nm, 56.4 nm | 2.0, 4.8 nm, 43.1 ± 0.9 nm |

With the DWBA, D of BornAgain's model came back within about 1σ, while the widths stopped at the upper limit of
fit1d's range. *auto* (up to three spheres) added a sphere of R ≈ 0.18 nm, a background term, on two of the three
simulated cuts.

The benchmarks still run with a copy of the kit: `tools/benchmarks/fit1d_recovery.py` (summary
`fit1d_recovery_summary.json`) takes it from the environment variable `GISAXS1D_KIT_HOME`, and
`tools/benchmarks/galaxi_fit1d.py` from `utils/gisaxs1d_gui_kit` of a checkout that still has it;
`tools/benchmarks/recovery_cases.py` holds the held-out curves that fit1d and Predict parameters are both scored on.

### 9.13 Predict parameters

*Predict parameters* finds the particles of a cut and their values from the curve, says how probable each solution
is and how well the curve fixes every value. A network trained on simulated cuts proposes compositions and values;
the proposals are weighted against Fitting's own model on the measured points; every solution is refined by least
squares and its posterior is sampled around it. The network only decides where the model is looked at: the values,
their intervals and the probabilities come from the model and the curve.

**Compositions and values.** A composition is one to three populations, listed by increasing mean radius, each a
sphere, an upright cylinder or a random cylinder, with or without the paracrystal of section 9.5 — 258 compositions.
Sizes are Gaussian by number (section 9.3), the coupling is the LMA. A population has R, σR/R, h/2R and σh/h (random
cylinders), D/2R and σnn/D (with a paracrystal); the cut adds the beam tail's width w and exponent ν (section 9.2), the
Gaussian resolution σ_q of the particle terms and the amplitude τ of a smooth multiplicative model error
exp(Σ_k b_k B_k(x)), b_k ~ N(0, τ²), over 8 cubic B-splines in x = ln(q/q_min)/ln(q_max/q_min). The scales (every
population's Int, the tail's A, the background) are not proposed: they are solved for every set of the others.

**What the network sees.** Everything is dimensionless: q̃ = q/q_max, lengths × q_max. The model depends on q·R only
(given q_z/q_max and σ_q/q_max), so one network serves cuts of any distance and energy. The window is the measured q
range, at most ρ = q_max/q_min = 300 (lower points are left out and counted) and at least 10. The curve is binned on
128 logarithmic points (the σ-weighted mean of the measured points in each bin, a bin without points masked); the
network reads ln I relative to its median, ln(σ/I), the mask and the position, and is told ln ρ, q₊/q_max and q₋/q_max
and whether the cut has two paths: with both angles known (q₊ = k_i + k_f and q₋ = k_i − k_f, vacuum), spheres have
the direct and the exit-reflected path with an effective reflection coefficient r (section 9.10); with only q₊, the
Born term at q₊; with neither, the effective interface (encoding 4).

**The prior.** Uniform in a box: ln R̃ from 0.5 to 5ρ (every population above the previous one, by stick-breaking),
σR/R 0.02–0.5, h/2R 0.1–10 with h̃ ≤ 200 and R̃ ≤ 50 for random cylinders, D/2R 1–20, σnn/D 0.02–0.5, w from 0.3 to
30 q_min (log), ν 1.8–2.2, σ_q/q_max 10⁻⁴–10⁻² (log), τ 0.003–0.15 (log), the real and imaginary part of r from −1 to
1 (they change nothing on a cut without two paths, where their posterior is the prior). The scales: uniform from 0 to
10 times the curve's maximum. The tail is held to the Lorentzian wings of a beam profile (∝ q⁻², as fit1d's ν = 2): with ν free in
1–6 a wide tail takes over the particles' own scattering at low q (on BornAgain's GALAXI model D fell to 14–22 nm), and
a tail capped at 3 q_min (encoding 2) cannot describe a low-q rise (D rose to about 105 nm); with ν near 2 least
squares from different starts reaches one solution there (D 51 nm, true 53.6 nm). GALAXI was used to find this.

**The network** (`src/gimap/features/fitting/domain/posterior/flow.py`):

- a 1D CNN (four stages of two convolutions and a pooling, 32–128 channels) and two dense layers give a summary h of
  the curve and its conditions;
- a head gives the probabilities of the 258 compositions from h;
- for a composition, a conditional normalising flow — ten rational-quadratic spline couplings (Durkan et al. 2019),
  10 bins on [−6, 6], conditioners of two layers of 256 — maps the values (the logit of their position in the box)
  to a standard normal. Values a composition does not have are set to 0 before every conditioner and never moved, so
  the density of the others is exact; sampling runs the couplings backwards.

It was trained by maximum likelihood (neural posterior estimation) with the cross-entropy of the head on 5 million
simulated cuts of `cut_model` (`tools/ml1d`, `prior_v2.json`): 40 % at the effective interface, 18 % at the Born term
of a q₊/q_max from 0 to 1 and 42 % with two paths (q₋ = q₊ × U(−0.5, 0.9), r from the prior); every population's
term on 8 points per bin (then averaged, as a
measured bin is), convolved with the cut's resolution; for every batch anew the populations' shares (2.5 decades),
the tail, the background, the model error, 100–1000 measured points (linear or logarithmic in q), Poisson counts
(10²–10⁷ at the highest point) and gaps. 30 epochs (2 h) on one GPU; the weights (13 MB, float32) come with GIMaP with
their SHA-256 and model card, and GIMaP evaluates them with numpy: neither PyTorch nor TensorFlow is needed.

**From the proposals to the solutions** (`posterior/infer.py`, `laplace.py`, `refine.py` `predict`). The Fit step
has one path; **Particles** chooses the compositions: *The particles in Model* (their families and whether each has
the paracrystal, in the order of their radii; with an empty Model it takes the next choice), *Spheres and upright
cylinders (1–2)* or *Any particles (1–3)*.

1. The compositions: that of Model, or the head's four most probable among the choice and its most probable of every
   number of populations among them (`infer.PER_COUNT`): the head often prefers one number of populations, and a
   composition that is not tried cannot be chosen (on the development curves this raised the share of curves whose
   composition is chosen right from 57 %/47 % to 63 %/62 % with one to three populations, *model*/*measured*).
2. For each, N = 1000 samples of the flow (300 for a composition with a random cylinder). Per sample, the terms of
   cut_model on the measured points (with the sample's resolution), the scales by non-negative least squares, and the
   likelihood (on a cut with two paths every sphere sample with its own r): Gaussian in σ, the model error integrated
   out exactly (Woodbury identity), the scales integrated out under their uniform priors (Laplace for the positive ones, a half Gaussian for those at 0). Weights L·π/q. For a
   bright curve the likelihood is far narrower than any proposal amortised over all curves: the weights sit on a few
   samples (effective sample size of order 1), which therefore only point at the modes. Without σ in the file, σ is a
   relative error from the scatter of ln I about its local trend (the robust second difference; at least 1 %).
3. The samples fall into modes (radii and distances 10 % apart, spreads 0.05 apart start a new one); the up to three
   heaviest modes of each composition are refined by least squares from their heaviest sample (section 9.8; σ_q free;
   every population held to the prior's box: R from 0.5/q_max to 5/q_min, the spreads 0.02–0.5, D/2R and h/2R in the
   prior's ranges at radii up to 1.5 times the start's; the tail within the prior's ranges). So are the up to three
   modes of the weights tempered to 50 effective samples (w^β, the largest β ≤ 1 that gives them) whose heaviest sample
   is another: on a bright curve the weights sit on one or two samples, and a mode the network proposed with a
   slightly lower likelihood would otherwise never be refined. On a cut with two paths, r is free in least squares
   (|r| ≤ 1, its phase within ±π of the start's) and goes with the solution's model into Model; it is not one of the
   reported values. For the two best compositions (by BIC) of two or three
   populations, two more starts come from each of their two best solutions (`posterior/moves.py`): a population's paracrystal
   peak fixes its D to a fraction of a per cent while its form factor fixes R only loosely, so another R at the same
   D, or the paracrystals of two populations exchanged, often describe the curve as well, and the network seldom
   proposes them. Every population's R × 0.5 and × 2 at its D and every pair's (D, σnn/D) exchanged (where the order
   of the radii and D ≥ 2R still hold) are scored by χ² with the scales solved; the best two are refined. A
   composition with a random cylinder gets neither the tempered modes nor the moves: its least-squares fits take
   20–30 s each, and the benchmarks below have no random cylinders. When no
   solution of one of the two best compositions
   (by BIC) refines below χ²/N = 2 (the network's proposals miss the curve, as on GALAXI), one search of the ranges
   (section 9.8) started from its 16 heaviest samples is added. Steps 1–3 use the σ of the counts, so the solutions
   and their χ²/N are those of the curve.
4. From here on the σ of every point includes a model error of 5 % of its intensity, σ_m² = σ² + (0.05 I)²
   (`refine.MODEL_ERROR_FLOOR`; *The model error* below). The refined solutions are ranked by BIC = χ²_m + k ln N (k
   the free values, the scales included); those more than 20
   above the best are dropped, and around each of the others (at most four) the posterior is sampled: a Student t (5
   degrees of freedom) over the values the curve constrains, its scale from the Gauss–Newton curvature of the
   residuals plus the prior's, inflated by 1.4 (and by √(χ²/N) on a curve the model does not describe within σ); the
   others (an absent tail, a resolution at its bound, τ) from piecewise densities that a scan of the likelihood along
   each sets. That is the first of up to eight rounds of 400 samples (population Monte Carlo; three for a composition
   with a random cylinder), run until 200 samples are effective. Each later round is a Student t over all values
   together, the scanned ones and τ included, with the mean and the covariance (inflated by 1.4) of all samples so
   far, their weights tempered to two effective samples per value: it follows a posterior that lies away from the
   least-squares optimum, or along a direction the first round has no width in (a spread at its bound that trades
   against the radius). All samples are weighted against the mixture of the rounds' proposals. The weighted
   samples give every value's median and 68 %/95 % interval, and the evidence Z of the solution.
5. A composition's evidence is the sum of those of its distinct solutions; the probabilities of the compositions are
   ∝ Z (equally likely beforehand), a solution's is its composition's times its share. When the sampling of any
   solution did not converge (fewer than 10 effective samples, or a refined solution whose radii left the
   composition's order and has no local posterior), all are compared by BIC instead, Z ∝ exp(−BIC/2), and their
   intervals are the least-squares errors of section 9.8 with σ_m (scaled by χ²_m/N when it is above 1; on a log
   scale for the lengths; never beyond the value's range); the
   solutions carry the flag *bic* (*compared by χ² and the number of values (BIC): the posterior's sampling did not
   converge*). This happens on curves the model does not describe within σ: with the model error near its bound the
   posterior lies away from the least-squares optimum, and the evidences cannot be estimated (on BornAgain's GALAXI
   model they ranked a solution with χ²/N = 11.5 above one with 4.4).

Every solution holding at least 1 % of the posterior is returned, most probable first. Each carries the refined
values with their least-squares errors (9.8), the posterior median and intervals, χ²ᵣ, its probability and flags. An
interval always holds the refined value. The refined value is the least-squares optimum with the counts' σ, and the
posterior has the model error and the prior's ranges, so on a curve the model does not describe within σ, or for a
value at a bound, the refined value can lie beside the central interval, which is then extended to it. A value the
curve does not see at all (the shape of a population whose amplitude is 0) has no least-squares error. The flags: a
value at a bound of its range; *unconstrained* when its 68 % interval covers more than 0.7 of the prior's range;
strongly correlated pairs (beyond ±0.95); few effective samples (below 50: the intervals are then taken with the
weights raised to the largest power β ≤ 1 that gives 50 effective samples, which moves them towards the wider
proposal); *bic*; *large* and *small* for a population whose size the measured q range does not show (q_min·R ≥ 0.7:
its Guinier region lies at or below the lowest q, and the decay the curve shows is also what roughness or a rise at low
q produce — on the *measured* benchmark curves an extra population of R 35–55 nm took up the roughness on 12 of 100
and 54 of 600 curves; q_max·R < 1: it acts as a smooth background, as fit1d named such a population). The most
probable goes into Model, the others to **Other solutions**. Model's fixed values and
ranges are not used; **Refine the values in Model** afterwards keeps them. Time per curve on one core (development
curves): about 5 s with the composition given, about 25 s choosing among one or two populations, 30–70 s among one
to three of spheres and upright cylinders, a few minutes when random cylinders are among the choice; with both angles
(two paths) about 2.4 times as long.

Model's coupling (DA) and size law (log-normal) are used in steps 2–4: the network, trained on the LMA and Gaussian
sizes, is then only a proposal, which the weights correct. Fitting has no four-term DWBA: with both angles Predict
uses the two paths of spheres with a free r (the terms that matter at the Yoneda angle, section 9.10), with q_z alone
the Born term at it, else the effective interface.

**The model error.** Without it (steps 4–5 with the counts' σ), the 68 % intervals held the truth in 60–64 % of the
curves of Fitting's own model (fit1d's 200 development curves drawn in `cut_model`, the composition given:
`posterior_recovery.py --truth own`) and in 49–57 % with two or three populations, where the sampling around a
solution sees only part of a curved posterior. On the independent truth model with a resolution, a beam tail and
roughness (*measured*, section 9.12) they held it in 41–46 %: there Fitting's model is close to the curve but not
exact, and the values that make up the difference — above all D and σnn/D, which the peak fixes to 0.1 % — move by
more than the counts' σ allows (their errors were 1.6–2.4 times the half-widths of the intervals). A model error per
point says how well the model can describe a measured cut. On the development curves (seeds 20260601: 200 of fit1d's
kind and 300 of the extended set; never the held-out ones), floors of 0, 3, 5 and 8 % gave 68 % intervals holding the
truth in 61/46, 78/63, 82/67 and 87/77 % (*model*/*measured*, fit1d's kind) and 51/45, 68/59, 71/67 and 78/70 %
(extended), with the same fits; 5 % is the smallest that brings both levels to 68 % or above. The composition was chosen
right more often on the *measured* curves (56 % instead of 41 %, 43 % instead of 36 %) and as often on the others
(88 % and 57 %), at about 5 % more time. Applied to every step (the network's weights and the least-squares fits too),
the same floor gave similar intervals but led the fits of some curves into other modes (χ²/N 4–7 where 1–2 was
reachable), so steps 1–3 keep the counts' σ. On curves of Fitting's own model the 68 % intervals are then
conservative (77–94 % for one or two populations, 75–88 % for one to three; `posterior_own_summary.json`,
`posterior_own_extended_summary.json`).

**Calibration in the model.** On 2000 simulated cuts not trained on (`tools/ml1d/evaluate_flow.py`, the true
composition given, 1000 samples each), the network's own 68 % and 95 % intervals hold the truth in 66–70 % and
92–95 % of the cases, for every value; the ranks of the truth among the samples are uniform within the statistics
(χ² of 20 rank bins 9–29, 19 degrees of freedom). The head ranks the true composition first in 20 % of the cuts:
many of the 258 compositions describe a cut about equally well (a paracrystal on a weak population, two populations
of similar size), and the likelihood decides among the head's most probable (four in GIMaP's Predict parameters, and
the best of every number of populations).

**Benchmark.** The 200 held-out curves of fit1d's benchmark (the independent truth model `yoneda_truth.py`; section
9.12), with the same scoring (`tools/benchmarks/posterior_recovery.py`, `--cut born`: Predict is given the cut's q_z,
the truth's Born term): half of them in fit1d's own physics
(*model*), half with a resolution, a beam tail and roughness (*measured*). Medians and 90th percentiles of the errors
over all populations, R and D relative, the spreads absolute; the true composition given. The "68 %" columns give how
often the 68 % interval of the solution (Predict) or ±1 least-squares error (fit1d) holds the truth; the composition
chosen among spheres and upright cylinders, one or two; time per curve on one core (`*_summary.json` beside the
scripts):

| level | value | Predict: median / P90 | Predict: 68 % | fit1d: median / P90 | fit1d: 68 % |
|---|---|---|---|---|---|
| model | R | 0.6 % / 17 % | 73 % | 0.5 % / 7.1 % | 61 % |
| model | σR/R | 0.005 / 0.083 | 73 % | 0.005 / 0.062 | 69 % |
| model | D | 0.0 % / 0.4 % | 88 % | 0.0 % / 0.2 % | 70 % |
| model | σnn/D | 0.001 / 0.005 | 87 % | 0.000 / 0.002 | 68 % |
| measured | R | 0.7 % / 23 % | 62 % | 1.0 % / 19 % | 34 % |
| measured | σR/R | 0.008 / 0.10 | 65 % | 0.010 / 0.10 | 36 % |
| measured | D | 0.1 % / 0.5 % | 78 % | 0.1 % / 0.3 % | 43 % |
| measured | σnn/D | 0.002 / 0.006 | 63 % | 0.001 / 0.006 | 26 % |
| | composition right (model / measured) | 88 % / 60 % | | 69 % / 30 % | |
| | seconds: composition given / chosen | 4.1 / ≈ 15 | | 3.5 / 6.3 | |

On 1200 curves drawn the same way on a seed range never used while the method was developed (`heldout_cases.py`):
one to three populations, the LMA or the DA (given to both methods), the composition chosen among one to three
populations:

| level | value | Predict: median / P90 | Predict: 68 % | fit1d: median / P90 | fit1d: 68 % |
|---|---|---|---|---|---|
| model | R | 1.3 % / 29 % | 84 % | 4.5 % / 40 % | 47 % |
| model | D | 0.1 % / 1.1 % | 88 % | 0.1 % / 4.5 % | 48 % |
| measured | R | 2.1 % / 46 % | 75 % | 7.2 % / 49 % | 30 % |
| measured | D | 0.1 % / 31 % | 80 % | 0.3 % / 12 % | 28 % |
| | composition right (model / measured) | 69 % / 56 % | | 32 % / 16 % | |
| | seconds: composition given / chosen | ≈ 7 / ≈ 40 | | 8.9 / 26 | |

What this says:

- Predict chooses the composition right more often than fit1d (88 % and 60 % against 69 % and 30 %; 69 % and 56 %
  against 32 % and 16 %), and lists the true composition among its solutions in 97 % and 82 %, and 80 % and 71 %
  (`true_in_equivalent_set`); its median errors are as small or smaller (R 0.6–0.7 % against 0.5–1.0 %, 1.3–2.1 %
  against 4.5–7.2 %).
- Its large errors are larger than fit1d's on fit1d's set (P90 of R 17 % and 23 % against 7.1 % and 19 %) and smaller
  on the extended set (29 % and 46 % against 40 % and 49 %); D's on the measured curves of the extended set are larger
  (31 % against 12 %). Nearly all are curves of two or three populations (a population off by more than 20 % in R:
  25 of the 200 curves, 24 of them with two; 246 of the 1200, 239 with two or three). For most of these the solution
  describes the curve within its errors with other sizes (χ²/N ≤ 1.5 in 18 of the 25 and 193 of the 246): the curve
  does not separate the populations.
- **Against the previous network** (flow_v3: the Born term only, `prior_v1`; the same curves): on fit1d's
  set the P90 of R was 7.2 % and 14 %, now 17 % and 23 % — 30 of the 300 populations off by more than 20 %
  instead of 21. Eleven curves are newly off, all of two populations: on six the truth is among the listed solutions
  at the same χ²/N (the curve does not fix the second population's size), on five no solution near the truth was
  found (χ²/N 0.98–2.1, one 17.7); three curves are newly right. The P90 lies at that edge (29 and 40 populations off
  by more than 10 %), so it doubles: resampling the curves puts the ratio at 1.04–3.6 (90 %). On the extended set the
  P90 fell (33 % and 54 % before), 73 curves newly off and 77 newly right; the composition was chosen right as often
  (88 %/64 % and 68 %/57 % before). The development curves (seed 20260601) had shown no such loss (P90 9.4 % before,
  8.7 % now): the network that also learnt cuts with two paths proposes the true mode of some two-sphere Born curves
  less often.
- **Several solutions.** The truth's mode is the most probable solution in 88 % of the curves of fit1d's set and 79 %
  of the extended set, and among the solutions listed (probability ≥ 0.05) in 92 % and 87 % (90 % and 85 % for the
  *model* and *measured* curves of the extended set; `truth_mode` in the summaries). One solution is listed for 80 %
  and 63 % of the curves, two to four for the others. On the development curves, every curve whose list missed the truth's mode had not
  found that mode at all — none was dropped by its probability or by the cut at ΔBIC 20 — and least squares started at
  the true values reached a χ²_m lower by more than 10 for 4 of 12 (one or two populations) and 29 of 66 (extended)
  of them: the search missed it; for nearly all others it fitted as well, a mode the curve cannot tell apart that the
  search did not reach. Step 3's tempered modes and moves were added for these: on the development curves the truth's mode is
  listed in 96 % and 86 % (94 % and 78 % without the moves). Half of the network's samples drawn from a base normal of
  width 2 (`infer.PROPOSAL_WIDTHS`) reached other missing modes but lost others (81 % listed on the extended
  development curves against 84 %, larger P90 errors there), so it is not used.
- Its 68 % intervals hold the truth in 73–88 % (*model*) and 62–78 % (*measured*) of the cases on fit1d's set, and in
  84–88 % and 75–80 % on the extended set; its 95 % intervals in 87–94 % and 79–95 %, and 92–96 % and 85–91 %. fit1d's
  ±1 errors hold it in 18–70 %. Read the intervals as including a model error of 5 % per point (*The model error*,
  above): curves of three populations whose solution is in another mode (above) are not covered by any interval.
- Time: 4.1 s and about 15 s on fit1d's set (fit1d 3.5 s and 6.3 s), about 7 s and 40 s on the extended set (fit1d
  8.9 s and 26 s). The runs of this table shared slower nodes, so these are the previous versions' times on a normal
  node scaled by the change measured on one node: against commit 96f000c, unchanged and +24 % on fit1d's set, −18 %
  and +16 % on the extended set (a faster sphere amplitude in `cut_model.shapes`, with the same values, pays for most
  of the moves from two solutions and the longer shortlist). Against the version without the model error, the
  tempered modes, the moves and the longer shortlist (2.7/9.4 s and 4.3/20.5 s), Predict takes 1.5–2 times as long.
  The network with two paths takes the same time on these cuts (one node, 48 development curves of each set: −11 % and
  −1 % on fit1d's kind, +8 % and −7 % on the extended kind). Given both angles (two paths, the DWBA curves below)
  Predict takes 2.4 times as long, with the composition given and to choose (one node, 48 development curves of each
  set): about 10 s and 36 s on fit1d's set, 17 s and 95 s on the extended set. Before the two paths were tabulated
  (section 9.3) it took 2.8–3.6 times as long; the tables save 18–33 % on fit1d's kind and 16–22 % on the extended kind.
- The measured curves have roughness falling as q⁻³ at low q, which the tail (ν near 2) cannot take up: there Predict
  chooses the composition right less often (60 % against 88 % in fit1d's own physics); on 13 of the 100 *measured*
  curves of fit1d's set it chose a composition with one more population that describes the curve better than the
  true one, the extra population taking up the roughness.
- **The waves the surface reflects.** Both sets draw their curves with the Born term at the cut's q_z, so they do not
  test the other three DWBA terms of section 9.10. The same cases, counts and noise drawn with all four terms
  (`--scattering dwba`, on bare Si and on 100 nm polystyrene on Si, `--substrate ps100`; the cut at the Si Yoneda
  angle, q₊ = 0.577 nm⁻¹, so q₊R = 1.7 at R = 2.9 nm) pair curve by curve with the table above; Predict is given both
  angles (`--cut two_path`: the two paths with a free r), q_z alone (`--cut born`) or neither (`--cut eia`)
  (`posterior_dwba_*_summary.json`; `posterior_dwba_pairs_*.json` from `tools/benchmarks/posterior_pairs.py` also
  holds the previous network's runs). Errors of R of the spheres, the true composition given, median (bias),
  Born-term truth → DWBA on Si given q_z alone → DWBA on Si given both angles → DWBA on PS given both angles:

  | q₊R of the sphere | fit1d's set | extended set |
  |---|---|---|
  | up to 1.7 | 0.4–0.6 % → 1.6–2.4 % → 2.3–3.8 % → 2.8–3.5 % | 0.8–5.5 % → 2.8–6.9 % → 2.6–6.7 % → 3.5–8.5 % |
  | 1.7–2.3 | 0.5 % → 4 % (−2 %) → 4 % (−3 %) → 3 % (+2 %) | 0.9 % → 4 % (−1 %) → 4 % (0 %) → 4 % (+1 %) |
  | 2.3–3.5 | 0.8 % → 11 % (−11 %) → 8 % (+1 %) → 4 % (+3 %) | 0.7 % → 12 % (−10 %) → 6 % (0 %) → 4 % (+3 %) |
  | above 3.5 | 5 % → 40 % (−30 %) → 8 % (0 %) → 3 % (+2 %) | 3 % → 38 % (−24 %) → 7 % (+3 %) → 5 % (+2 %) |

  - Upright cylinders alone give the same results with the four terms (all 50 and 200 such curves): their cut has
    the in-plane shape exactly (section 9.10).
  - Spheres above q₊R 1.7: given both angles, R is off by 6.5 % and 5.9 % (median, Si; P90 19 % and 40 %) and by
    3.3 % and 4.1 % on PS, and their 68 % intervals hold the truth in 54 % and 65 % (PS 62 %). With the direct
    quadrature of the two paths instead of the tables (section 9.3; the same network and curves) R was off by 5.7 % and
    5.8 %: values that differ by at most 10⁻³ change the solution of a few curves, and a median over 145 spheres moves
    by that much. Given q_z alone
    they are off by 12 % and 17 % (P90 48 % and 54 %), mostly too small, and the intervals hold the truth in 6–11 %;
    the previous network gave the same (12 % and 17 %).
  - The composition. Of the curves whose largest sphere has q₊R above 1.7, Predict given both angles chooses the right
    one in 62 % on Si and 68 % on PS (fit1d's set) and 57 % and 62 % (extended), against 4 % and 18 % on Si given q_z
    alone (the previous network: 6–20 % on Si and PS; 72 % and 68 % with the Born-term truth) — there a single sphere is mostly taken for one or
    two upright cylinders on Si and for a sphere with a cylinder on PS. The truth's mode is listed for 96 % and 84 % of
    the curves with a sphere above q₊R 2.3 (given q_z alone 50 % and 36 %; Born-term truth 85 % and 84 %).
  - Over all curves, on Si given both angles: composition right in 74 %/56 % (*model*/*measured*) on fit1d's set and
    60 %/51 % on the extended set (on PS 78 %/61 % and 63 %/53 %), against 39 %/25 % and 34 %/32 % given q_z alone and
    88 %/60 % and 69 %/56 % with the Born-term truth; P90 of R 16 %/21 % and 34 %/42 % (given q_z alone 48 %/40 % and
    52 %/52 %).
  - Below q₊R 1.7 the two paths are a little worse than q_z alone: on fit1d's set 6 of 45 spheres at q₊R 1.2–1.7
    off by more than 20 % (none given q_z alone), and the composition of the curves without a larger sphere right in
    70 % and 53 % against 80 % and 54 %.
  - χ²/N > 2 marks only 3 % and 8 % of the curves where a sphere's R is off by more than 10 % (61 % and 49 % given q_z
    alone): what is left fits the curve within its errors, so χ²/N does not point at it.
  - Predict without the cut's q_z (the effective interface, `--cut eia`) on the same Si curves, as before: R of
    spheres at q₊R 1.7–3.5 low by 16 % (fit1d's set), D and σnn/D within about 1 % as in section 9.10, χ²/N > 2 for
    only 18–30 % of the curves with a sphere off by more than 10 %, composition right in 46 %/29 % and 44 %/35 %.
  - **Two paths where the curve has none.** Given both angles on curves without a reflected wave (the Born-term
    truth; development seed 20260601, `--scattering born_qz --cut two_path`), spheres above q₊R 1.7 are off by 7.9 %
    and 10.5 % (median; P90 60 % and 63 %) against 0.5 % and 1.5 % given q_z alone (P90 19 % and 53 %), and the
    truth's mode is listed for 76 % and 65 % of these curves against 92 % and 82 %: a sphere of about half the radius
    with a strong r describes such a curve as well. At the Yoneda angle of the substrate |r_f| is near 1 and the
    reflected wave strong; on a cut well above the critical angle it is weak, and a solution with a large |r| and a
    small sphere there is suspect (fix `refl_amp` to the substrate's |r_f|, or to 0, and refine).
  - So, on a real substrate at the Yoneda angle, Predict given both angles holds the sizes of spheres to 2–9 %
    (median, by q₊R) and chooses their composition right in about 60 % of the curves; given q_z alone the error
    of R grows to about 10 % at q₊R 2.3–3.5 and 40 % above (too small), as before.
  - The stop rule set before training (development seed: composition of the curves with a sphere above q₊R 1.7 right
    in ≥ 50 %, R of these spheres ≤ 6 % median, on the Born-term truth given q_z alone the composition no more than 3
    points and the P90 of R no more than 1.2 times worse) was missed narrowly there (R 6.5 % and 6.3 %; composition
    −3.0 and −4.7 points). The network was kept: these curves had 5–20 % and 14.6–20.6 % before. On the held-out
    curves the composition is within the rule (−2.0 and +0.4 points) and R on the extended set (5.9 %); R on fit1d's
    set is just outside it (6.5 %; 5.7 % with the direct quadrature, above), and the P90 of R on fit1d's set is 1.9
    times worse (above, *Against the previous network*).

Earlier runs on the same curves: before the search of the ranges (step 3), the comparison by BIC (step 5), the
prior's box for least squares and the faster settings (2000 samples, four compositions, the posterior of every
solution), Predict had the same medians, the composition right more often on the measured curves (65 % and 40 %),
intervals that held the truth less often (43–62 % and 30–44 %), and took 40 s and 86 s to choose
(`posterior_*_first_summary.json`). With those rules but without the model error and the tempered modes (the
summaries at commit a0c597c): the same medians, P90 of R 36 % and 24 % (fit1d's set) and 58 % and 66 % (extended),
the composition right in 87 %/48 % and 58 %/37 %, intervals holding the truth in 39–62 % and 40–60 %, 2.7/9.4 s and
4.3/20.5 s. With the model error and the tempered modes but without the moves (commit 98a7cc8): P90 of R 9.5 %/18 %
and 43 %/55 %, the truth's mode listed in 91 % and 79 %, the composition right in 87 %/68 % and 58 %/45 %, intervals
holding the truth in 60–91 % and 70–85 %, 3.0/10.1 s and 6.8/28.5 s. With the moves from one solution and the
head's three most probable compositions (commit 96f000c): the truth's mode listed in 93 % and 85 %, the
composition right in 87 %/67 % and 57 %/44 %, P90 of R 9.7 %/14 % and 34 %/53 %. With the previous network
(flow_v3, the Born term only, `prior_v1`): P90 of R 7.2 %/14 % and 33 %/54 %, the truth's mode listed in 95 % and
87 %, the composition right in 88 %/64 % and 68 %/57 %, intervals holding the truth in 61–93 % and 70–88 %.

fit1d was nevertheless removed from GIMaP in 0.2.0, so that Fitting has one network method: its benchmark stays the
reference (section 9.12). 1D Predict V5, on the same 200 curves, had median errors of 31 % (R) and 22 % (D)
(chapter 10).

**GALAXI.** The cut of the public GALAXI frame and three Poisson draws of BornAgain's model of it (log-normal radii,
number mean 6.24 nm, radial paracrystal of 53.6 nm; four-term DWBA; α_i = 0.463°, α_f = 0.18°, near the critical
angle of Si), as in section 9.12 (`tools/benchmarks/galaxi_posterior.py`, `galaxi_posterior.json`). One sphere
population with a paracrystal, Born at the cut's q_z:

| source | Predict: R (nm) | Predict: D (nm), 68 % | χ²/N | fit1d: R, D (nm) | χ²/N |
|---|---|---|---|---|---|
| BornAgain's model, three draws | 4.84–4.88 | 49.2 [39.7, 61.0] on one draw; 676 and 1350 (*unconstrained*) on the others | 4.4–5.4 | 4.88–4.93, 56.9–57.6 | 6.1–6.5 |
| the same, log-normal sizes | 4.84–4.90 | 49.3–51.3, intervals [41–43, 59–62], each holds 53.6 | 4.2–4.4 | | |
| measured cut | 5.16 | 46.4 [40.0, 53.9] | 8.8 | 5.20, 56.4 | 14.4 |

With the network that came before this one (Born term only) the Gaussian sizes gave D 49.2–52.1 nm on all three draws
(intervals [41–44, 59–63], each holding 53.6). With the composition chosen among one to three sphere populations,
Predict now takes one sphere with a paracrystal on the measured cut (R 5.16 nm, D 46.4 nm), and on the draws a sphere
with a second one whose paracrystal is at a bound or unconstrained (twice) or three spheres, one of R 0.21 nm flagged
*small* (D of the second 51.2 nm); before: two spheres on two draws and one on the third, D 42–53 nm. fit1d chose a
sphere of R 0.18 nm on one of the draws. All these solutions were compared by BIC (the model does not describe the cut
within σ: χ²/N 4–9), and their intervals are least-squares errors with the model error: the cut has a shoulder, not a
resolved peak, so D is fixed to ±7–10 nm (±3 nm without the model error). fit1d's four-term DWBA, which Fitting does
not have, describes the simulated cuts better (χ²/N 1.75, D 53.8–55.1 nm).

*Both angles* (two paths, r free). The two paths describe these cuts within their errors (χ²/N 0.96–1.55 against
4.2–8.8), but they do not recover the particles: with one sphere population, R 8.4 nm on the measured cut and 500 nm
(at the bound, flagged *large*), 110 nm and 16 nm on the draws, D 460–6700 nm instead of 53.6 nm; with the composition chosen
among one to three spheres, two spheres on the measured cut (R 6.06 nm with D 43.8 nm [31.1, 61.7], and 8.8 nm), on
one draw R 5.0 nm with D 104 nm [68, 159] and a sphere of 525 nm at the bound, on the other two one sphere of
15.5–16.1 nm without a paracrystal; the fitted |r| 0.31–1.0 at phases spread over the whole circle. On this cut
(broad log-normal sizes, α_f at the critical angle) the free r takes up what the sizes and the paracrystal should
describe; the solutions it gives carry flags (*at bound*, *large*, *unconstrained*) and should not be taken without
them being checked against what is known of the sample. Time (this PC): 12–14 s per variant with one sphere and 46–63 s
for the choice at the Born term; 12–26 s and 97–139 s with two paths (56 min for the choice on the measured cut
before the cross term was tabulated, section 9.3). GALAXI was used to set the tail's prior and the rules of steps 3
and 5, so this is no independent test.

Implementation: `src/gimap/features/fitting/domain/posterior/`, `application/single_predict.py`,
`infrastructure/adapters/posterior_job.py`; training and checks: `tools/ml1d/`; benchmarks:
`tools/benchmarks/posterior_recovery.py`, `v5_recovery.py`, `galaxi_posterior.py`.

## 10. 1D Predict (many curves)

The 1D Predict window (**Tools ▸ 1D Predict — Fit Many Curves…**) fits a list of curve files, or the curve open in
Fitting, one half at a time, with *Predict parameters* (section 9.13), its one method. It runs in a separate,
cancellable process and needs no TensorFlow. Settings saved with the methods of earlier versions (*General V5*,
*Single RC specialist*, the physical fit, fit1d) open as Predict parameters: their own options are dropped, and
fit1d's *Cut qz* is kept as the cut's q_z.

Implementation: `src/gimap/features/fitting/infrastructure/adapters/workflow_v5.py` (the job, its files),
`posterior_job.py`, `src/gimap/features/fitting/application/workflow_v5.py` (the options).

### 10.1 Input

Each side of a signed curve is fitted separately (no folding). A side needs 8–1000 points with strictly increasing
`q > 0`; q = 0 is dropped, nothing is interpolated or invented. q in Å⁻¹ is multiplied by 10; a file whose header
states no unit is read in the setting (Å⁻¹ by default, the default of the Fit page too). Without σ the curve gets
the estimate of Predict parameters on the Fit page (section 9.13, step 2): the
points with I ≤ 0 are left out, since ln I has no value there, and

```text
σ_i = r I_i,   r = max( 0.01, median |ln I_(i+1) − 2 ln I_i + ln I_(i−1)| / (0.6745 √6) )              (10.1)
```

(the robust scatter of ln I about its local trend; r is recorded as `sigma_relative`). For **Use current cut** on a
curve with the pixel counts of its detector columns, **Relative σ added to counting σ** (0.1) and **Absolute σ added
to counting σ** (0) in **Parameters…** are added in quadrature to the counting σ. The side's largest intensity is recorded
as its normalizer; the method sees the curve as measured. The scores of a candidate are those of the measured points:
logRMSE (natural logarithm, positive points), RMS/σ and the mean of the squared normalised residuals.

### 10.2 The method

Section 9.13 on each half, with **Complete composition** as the composition (1–3 populations, each with the
paracrystal) or, with *Auto*, the head's four most probable among all compositions and its most probable of every
number of populations; q_z of the cut from **Cut qz
(nm⁻¹)** (unticked: the effective interface). The eight most probable solutions of a half are its candidates, with
the stage *Network + posterior*; each carries its probability, its posterior intervals and flags (`solutions.json`:
`probability`, `posterior`, `warnings`, `ess`). The run writes the files it wrote before 0.2.0 (`request.json`,
`batch_summary.json`, `top20_candidates.json`, `fitting_run_summary.json`, and per half `solutions.json`,
`display.npz`, `fitting_curves.csv`).

### 10.3 Workflow V5, removed in 0.2.0

Until 0.2.0 the window's default was *General V5*, a TensorFlow workflow (`modules/Fitting_1D_Model/Workflow_v5/
conditional_v2`) whose networks proposed one of 34 compositions of 1–4 components and their values, which four damped
Gauss–Newton steps then corrected; *Single RC specialist* (`stable_blue_rc_v1`) did the same for one random cylinder
and native detector counts. Its validation (its model card) covered simulated curves with the composition and the
resolution given, and measured a curve-fit rate, not parameters: 114 of 128 curves within logRMSE 0.05.

Both were removed when *Predict parameters* recovered the parameters better on every benchmark. On the 200 held-out
curves of fit1d's benchmark (section 9.12; `tools/benchmarks/v5_recovery.py`, the true composition given, V5's
default options, the q > 0 half, sizes compared in V5's convention):

| level | value | Predict: median / P90 | V5: median / P90 |
|---|---|---|---|
| model | R | 0.7 % / 36 % | 31 % / 141 % |
| model | σR/R | 0.007 / 0.13 | 0.14 / 0.36 |
| model | D | 0.0 % / 0.6 % | 39 % / 311 % |
| measured | R | 0.9 % / 24 % | 34 % / 87 % |
| measured | σR/R | 0.009 / 0.12 | 0.16 / 0.42 |
| measured | D | 0.1 % / 0.9 % | 15 % / 274 % |
| | seconds per curve | 2.7 | 19.8 |

V5 also gave no probability and no interval. The code, the weights and their records remain in the repository's
history (the last commit that has them: 06e7395), and `v5_recovery.py` runs there.

## 11. Series

A series is an ordered list of frames of one measurement: files, a folder, or a multi-frame NeXus/HDF5 file. GIMaP reduces every frame (or every group of `N` summed frames) with the same geometry, masks, corrections and cut, and stacks one curve per frame into a map. This chapter states what is computed on such a stack: the map itself, a peak followed in a fixed window, the fit of every frame's ring peaks, the odd frames and stages of a series, and the comparison of several series.

Units follow the source of the curves. In Analyze the x of a map is q (Å⁻¹), q_y (Å⁻¹) or χ (°). Fitting ▸ In-situ series compares its curves on the q (nm⁻¹) of the first usable curve. Compare uses the x of its series, which must all carry the same x label.

### 11.1 The series map

Each row of the map is one frame, or one group of summed frames, in list order. Each column is one x value. The cut found on the chosen frame is held for every frame. The map keeps measured values; it does not resample them when it can avoid it.

1. When all curves have the same x, the rows are stacked as they are.
2. When the curves were measured on the same points but some points are missing in some frames (valid detector columns that differ between frames), the union of all x is the grid, and every value goes to the column of its own x. A value that occurs twice at one x (both halves of a signed cut on |x|) gets two columns. This is used only when the union holds at most 1.5 times as many points as the longest curve.
3. Otherwise the curves are put on one even grid from the lowest to the highest x of all curves, with the median of the curves' usual steps. Each curve is linearly interpolated onto it, never beyond its own ends and never across a hole of its own: an interval wider than 2.5 times the median of the 9 steps around it (a detector gap, the beam stop). Cells outside or inside a hole stay empty (NaN).

A signed cut can be stacked as measured, as either half, folded onto |x|, or as the mean of the halves (6.16) on the |x| of the `x > 0` half. The map carries no σ.

The picture of the map is drawn on an even grid with the usual step, at most 8192 columns (each pixel column shows the nearest measured x, a hole stays empty). This is for display only; the exports keep the measured x.

The frame axis is the frame number counted from 1. It can be time when the frames carry one, in this order of preference:

```text
t_r = T_r − T_1                                   (time stamp or per-frame time of row r)      (11.1)
t_r = (j_r − j_1) · t_exp                         (one multi-frame source, exposure t_exp)     (11.2)
t_r = (T_file(r) + j_r · t_exp) − (T_file(1) + j_1 · t_exp)   (several multi-frame sources)      (11.3)
```

Here `T` is a time stamp, `T_file(r)` the start time of the file of row `r`, `j_r` the frame index in its file and t_exp the single exposure time of all frames. Times from (11.2) and (11.3) are lower bounds: the read-out time between frames is not in the file. The time axis changes only the drawing and the exports. Stages (section 11.7) are always cut in row order.

Implementation: `src/gimap/features/analyze/domain/series_map.py`, `src/gimap/features/analyze/application/series_map.py`

### 11.2 Traces and the peak in a window

Under the map (Analyze ▸ Series), a vertical band selects an x window `[x_lo, x_hi]`. The trace is either the mean intensity in the window or a property of the peak in it.

**Mean intensity.** For every row, the mean of the finite values in the columns with `|x − x_0| ≤ h`, where `x_0` is the window centre and `h` its half width (at least half the median grid step; one grid step by default). When no column lies inside, the nearest point is used, unless the window lies in a hole or beyond the ends of the map; then every row has no value. No value is made up.

**Default position.** The first window is put at the x where the last quarter of the rows differs most from the first quarter, in units of the noise of that column:

```text
score(x) = | mean_last(I) − mean_first(I) | / max( ν(x), median_x ν )                          (11.4)
ν(x) = 1.4826 · median_r | Δ²_r I(x) | / √6        (4 rows or more)                             (11.5)
ν(x) = √| mean_r I(x) |                             (fewer than 4 rows)                          (11.6)
```

`Δ²_r` is the second difference along the rows. A steady rise or fall cancels in it, so a trend is not counted as noise.

**Peak in the window.** For every row, with the points `x_1 < … < x_N` of the window (`N ≥ 5`) and their finite values (at least 5):

```text
n_e  = max(1, min(3, ⌊N/5⌋))                                                                  (11.7)
b(x) = ȳ_L + (ȳ_R − ȳ_L) · (x − x̄_L) / (x̄_R − x̄_L)                                           (11.8)
z_j  = y_j − b(x_j),   w_j = max(z_j, 0)                                                        (11.9)
x_pk = ∑ x_j w_j / ∑ w_j                                                                       (11.10)
σ_m² = ∑ (x_j − x_pk)² w_j / ∑ w_j                                                             (11.11)
FWHM_m = 2.3548 · σ_m                                                                          (11.12)
A    = ∑ ½ (z_j + z_{j+1}) (x_{j+1} − x_j)                                                     (11.13)
h_pk = max_j z_j                                                                               (11.14)
```

`(x̄_L, ȳ_L)` and `(x̄_R, ȳ_R)` are the means of the first and of the last `n_e` finite points. So the background line uses 1 point at each end for windows of 5–9 points, 2 for 10–14 points and 3 from 15 points. `z_j` is the net intensity and `w_j` its positive part. A row without anything above the background gives no value. The area uses the unclipped net intensity on the row's own, unevenly spaced x. Position and width are in the unit of the map's x; the area is intensity × that unit.

The window is the same for every row. It does not follow the peak, and no position is predicted from earlier rows. `FWHM_m` is a moment width: the FWHM of a Gaussian that lies well inside the window. It grows with the window (tails, a sloping wing) and has no error. A fitted FWHM with its error comes from the batch fit (section 11.3). The CSV export says this in its header.

Implementation: `src/gimap/features/analyze/domain/series_map.py`, `src/gimap/features/analyze/application/series_map.py`

### 11.3 Batch fitting of the ring peaks of every frame

Batch Export ▸ Fit ▸ Peaks of the ring regions fits one peak per cut region in every GIWAXS frame. Each cut region with a q window `[q_lo, q_hi]`, a χ range (default 0–90°) and a both-sides flag is one target. GISAXS frames have no peak targets.

**Fit window and data.** The fit window is the region's window widened by its own width on each side:

```text
W = [ max(0, 2 q_lo − q_hi), 2 q_hi − q_lo ]                                                   (11.15)
```

The reason in the code: a window of only ± one FWHM leaves too little background to tell it from the tails of a pseudo-Voigt. With the frame's q maps, I(q) is binned straight from the valid pixels above the sample horizon in the region's χ range, at the detector's own q step (at least 10 bins; the width/100 when the step is unknown). The uncertainty is Poisson when the frame holds counts (an integer frame without a background file and without mirror filling, section 6.4); otherwise it is the pixel scatter. With an active intensity correction the variance is scaled as in section 5.5. Without q maps, the region's own I(q) curve is used (2θ converted to q).

**Model.** The single-peak fit of Analyze's cut regions:

```text
I(q) = A · P_pV(q; q_0, Γ, η) + b_0 + b_1 (q − q_0,start)                                      (11.16)
P_G(q) = √(4 ln 2 / π) / Γ · exp(−4 ln 2 (q − q_0)² / Γ²)                                      (11.17)
P_L(q) = (2 / (π Γ)) / (1 + 4 (q − q_0)² / Γ²)                                                 (11.18)
P_pV = η P_L + (1 − η) P_G                                                                     (11.19)
```

Both shapes have unit area and the FWHM `Γ`, so `A` is the integrated intensity above the background; (11.19) is a pseudo-Voigt profile with a common FWHM and a free mixing η (Thompson et al., 1987, for the profile family). `q_0,start` is the starting centre. The parameters are found by weighted least squares (`scipy.optimize.curve_fit`, σ as weights, `absolute_sigma=True`, at most 20000 evaluations) with the bounds `q_0 ∈ [q_min, q_max]` of the points, `Γ ∈ [0.5 δq, 2 S]` (δq the median step, S the span of the points), `A ≥ 0`, `η ∈ [0, 1]`. A Gaussian or Lorentzian profile is fitted with η fixed (five parameters).

```text
χ²_red = ∑ ((y − f)/σ)² / (N − p)                                                              (11.20)
err(p_k) = √C_kk · √max(1, χ²_red)                                                              (11.21)
h_pk = A · P_pV(q_0; q_0, Γ, η),    d = 2π / q_0                                               (11.22)
```

A fit is not reported (its row keeps the reason) when there are fewer than 7 points in the window, when `A ≤ 0` ("no peak above the background"), when `A < 3 · err(A)` ("no significant peak"), or when `Γ ≥ 0.9 S` ("the peak is as wide as the window").

**Starting values.** The person chooses one rule for the whole batch:

- *fresh*: every frame from its own data. The background line goes through the means of the first and last `max(1, min(3, ⌊N/6⌋))` points; the start centre is the highest point above that line, the start width Γ_0 its half-maximum width (at least 2 steps), `A_0 = 1.064 · h_pk · Γ_0`, `η_0 = 0.5`.
- *previous* (default): centre, FWHM, area and η of the last successful fit of the same peak. The background start is always taken from the frame's own window.
- *first*: the values of the first successful fit of that peak.

The start centre is clipped into the window. A failed fit is not remembered, so the next frame starts from the last good result. There is no extrapolation from earlier frames.

The table gives per frame and peak: q (Å⁻¹) and its error, d (Å), FWHM (Å⁻¹) and its error, area and its error, height, η, χ²_red and the note.

For GISAXS frames, the batch fit instead sends each frame's horizontal cut (the halves chosen) to Predict parameters, as the automatic analysis does (section 8.5): the network chooses among one or two populations of spheres and upright cylinders, with or without a paracrystal, or **Model** gives the composition (a sphere, a random cylinder or a vertical cylinder, with a paracrystal). It needs no starting values, so the starting-value rule applies to the peaks only and every frame is predicted from its own data; nothing is carried from frame to frame. The table has the most probable solution of each frame with its probability, χ²ᵣ (8.6) and log RMSE; lengths are in nm, q in nm⁻¹. From some seconds to a few minutes per frame, depending on the number of points and the computer.

Implementation: `src/gimap/features/analyze/application/batch_fit.py`, `src/gimap/features/analyze/domain/peak_fit.py`, `src/gimap/features/analyze/domain/regions.py`

### 11.4 Curves made comparable

Stages, odd frames and Compare work on one representation of the curves. With the image `I_{f,b}` of frames `f` and bins `b` (NaN where a frame has no data):

```text
y_{f,b} = log10 I_{f,b}            if I_{f,b} > 0,   else missing                              (11.23)
bin b kept  ⇔  q_b finite, q_b in the q range, y finite in ≥ 95 % of the frames                (11.24)
y_{f,b} ← median_f y_{f,b}         where missing (kept bins only)                              (11.25)
ȳ_f = mean_b y_{f,b}                                                                            (11.26)
x_{f,b} = y_{f,b} − ȳ_f            (shape only, default);   x_{f,b} = y_{f,b} otherwise         (11.27)
```

Fewer than 3 kept bins is an error ("widen the q range"). Non-positive values (gap codes, zero counts) are missing, not zero. A missing value filled with the median of its bin cannot make a frame look odd. With *shape only* an overall change of intensity (beam current, thickness, exposure) is not a change of structure; the level `ȳ_f` is kept and exported separately.

Implementation: `src/gimap/shared/series_stages/comparable.py`

### 11.5 Odd frames

An odd frame matches neither the frames before it nor those after it, by far more than the series moves from frame to frame there. It is found before any principal component, because one odd frame would otherwise become a component of its own. The search needs at least 4 frames. Below, `x` holds only the frames still in the search, and `f` and `j` are positions in that list.

```text
Δ_{j,b} = | x_{j+1,b} − x_{j,b} |                                                              (11.28)
s_j     = RMS_b Δ_{j,b}                     (step j → j+1)                                     (11.29)
s_typ   = max( median_j s_j, 10⁻¹² )            (typical step)                                     (11.30)
u_b     = 1.4826 · median_j Δ_{j,b}         (usual step of bin b)                              (11.31)
S⁻_f    = median of x over rows f−3 … f−1,   S⁺_f = median over rows f+1 … f+3                 (11.32)
d^±_f   = x_f − S^±_f,   d_f = the one of d⁻_f, d⁺_f with the smaller RMS                     (11.33)
m_f     = max( median_{j ∈ J_f} s_j, s_typ ),    J_f = {f−4 … f+3} without f−1 and f           (11.34)
```

Only the sides that exist are used; at an end of the series there is one side. The distance is to the *closer* side: a frame right after a sudden change matches the frames that follow it, so it is not odd.

**Whole-curve test.**

```text
odd  ⇔  RMS(d_f) / m_f > 5   and   RMS(d_f) > 5 s_typ,    z = RMS(d_f) / m_f                       (11.35)
```

**Spike test** (only when the whole-curve test fails). The part of the difference that has the same sign against both sides:

```text
c_{f,b} = sign(d⁻_b) · min(|d⁻_b|, |d⁺_b|)   if d⁻_b · d⁺_b > 0,   else 0                      (11.36)
T_f     = the n_top bins of largest c², n_top = max(3, round(0.01 N_b))                          (11.37)
μ_f     = max( RMS_{b∈T_f} u_b,  median_{j∈J_f} RMS_{b∈T_f} Δ_{j,b} )                          (11.38)
odd  ⇔  RMS_{b∈T_f} c_{f,b} / μ_f > 8,        z = that ratio                                    (11.39)
```

`N_b` is the number of kept bins. At an end of the series `c = d` of the one side. A steady change lies between its neighbours and gives `c = 0`; a step matches one side. The threshold 8 is higher than 5 because these bins are the largest of the frame: in noise alone they reach about 3.5 times their usual step (stated in the code as measured on Gaussian and few-count Poisson series of 5–3200 frames). Over the whole curve a spike is diluted: a hot column 25 times too bright in 1 % of the bins is 1.4 in log10 there but only about 0.14 RMS.

**Detector or sample.** The difference is called *narrow* ("probably the detector, not the sample") when its `n_top` largest bins hold more than 80 % of its squared sum. The reported q is the median q of those bins.

**Repeated search.** An odd frame does not vote for its neighbours. After a pass, the odd frames are removed and the search is repeated, at most 3 passes. In a later pass only frames that were within 3 positions of a frame newly removed in the previous pass can be added. The search stops when nothing new is found or fewer than 4 frames remain. In the stage search, if fewer than 3 frames would be left, no frame is called odd. So two or three glitched frames in a row are all found; four or more alike frames are a state of their own (at the ends, three). A change in the first or last one or two frames cannot be told from an artefact, because only one side exists; it is reported as odd with its reason.

Odd frames keep their row: they are assigned to the stage they lie in, but they are left out of the components, the cut, the stage means, the end states of Compare and (on request) the next Batch Export.

Implementation: `src/gimap/shared/series_stages/comparable.py`

### 11.6 Main components

With the kept (not odd) rows `K` and their mean `x̄`:

```text
X_K − x̄ = U S Vᵀ,    γ_i = S_ii² / ∑_l S_ll²                                                          (11.40)
n_pc = min( 10, first n with ∑_{i≤n} γ_i ≥ 0.95, rank )                                            (11.41)
scores = (x − x̄) V_{n_pc}ᵀ          (every row, odd rows projected too)                              (11.42)
```

The sign of a component is arbitrary. It is turned so that the mean score of the last `n_end` kept rows minus that of the first `n_end` is not negative, `n_end = max(1, min(10, ⌊|K|/4⌋))`. The first component is the "Change along the series" trace of Analyze.

Implementation: `src/gimap/shared/series_stages/comparable.py`, `src/gimap/shared/series_stages/stages.py`

### 11.7 Stages: segmentation and its stopping rule

The scores of the kept rows are cut into at most 8 stages of at least 5 frames. Within each stage every component follows its own straight line in the frame index.

**Long series.** With more than 1500 kept rows, the kept rows are averaged in blocks of `n_blk = ⌈|K|/1500⌉` consecutive kept rows. A block's position `t` is the frame index of its first row, so the gaps of odd frames stay. The shortest stage is then `max(1, ⌈5/n_blk⌉)` blocks. Otherwise `n_blk = 1` and `t` is the frame index.

**Cost of a stage.** For the rows `i … j−1` with positions `t` and scores `s` (vectors of length `n_pc`):

```text
C(i,j) = ∑ ‖s_f − s̄‖² − ‖ ∑ (t_f − t̄)(s_f − s̄) ‖² / ∑ (t_f − t̄)²,      clipped at 0             (11.43)
C(i,j) = ∞                                    when j − i < shortest stage                       (11.44)
```

This is the summed squared residual of an independent least-squares line per component.

**Optimal cut.** Dynamic programming gives the exact optimum for every number of stages `M = 1 … 8` (optimal partitioning; Jackson et al., 2005):

```text
V_1(j) = C(0, j),    V_M(j) = min_{i<j} [ V_{M−1}(i) + C(i, j) ],    C_M = V_M(n)                (11.45)
```

**What noise would explain.** A penalty in the style of the Bayesian information criterion (BIC; Schwarz, 1978), with the noise taken from second differences of the scores (a straight drift has none):

```text
σ²_c = median_f ‖ s_{f+1} − 2 s_f + s_{f−1} ‖² / (6 n_pc)                                          (11.46)
P    = σ²_c · (2 n_pc + 1) · ln n                     (n rows or blocks cut, n ≥ 4; else P = 0)      (11.47)
```

The factor 6 is the variance of a second difference of white noise.

**Stopping rule.** Start with one stage. Add stage `M` while

```text
C_{M−1} − C_M ≥ max( 0.05 · C_1,  P )                                                           (11.48)
```

and stop at the first `M` that fails; there is no look-ahead. The 5 % is of the *one-stage* residual `C_1`, not of the residual still left. The reason in the code: real curves are not exactly piecewise straight, so a noise criterion alone would keep splitting. Invented example: with `C_1 = 100`, `C_2 = 17`, `C_3 = 1.0`, `C_4 = 0.45` and `P = 0.3`, the gains are 83, 16 and 0.55 against the bar 5, so 3 stages are suggested. The share `C_M / C_1` is recorded for every `M` (`unexplained_by_stages`), and the person may choose another number of stages.

**What changes between stages.** For each stage, the mean of `log10 I` over its kept rows on the compared bins (unfilled values). Between stage `s−1` and `s`:

```text
Δ_b = m_{s,b} − m_{s−1,b},    Δ_b ← Δ_b − mean_b Δ_b                                            (11.49)
change in percent = 100 · (10^{Δ_b} − 1)                                                        (11.50)
```

Up to 3 q of the largest positive `Δ` ("grows most") and of the most negative ("falls most") are listed, each more than 2 % of the compared q span from the others. The change is relative to the rest of the curve. Compare shows them in the tooltip of its folded *Odd frames and stages of every series* and writes them into its JSON record (`stage_details`: each `x` with its `percent` of (11.50)). The representative frame of a stage is its kept frame nearest (least squares) to the stage's mean `x`.

A stage is a description, not a phase: a smooth change is also cut where its pace bends. What grows or falls between stages says whether a boundary is structural. Boundaries have no uncertainty attached.

Implementation: `src/gimap/shared/series_stages/stages.py`, `src/gimap/features/analyze/application/series_stages.py`

### 11.8 How fast: the half and 90 % frames

On the first-component scores `v` of the kept rows, in row order (at least 6 values):

```text
m_i   = median( v_{i−2} … v_{i+2} )          (window cut at the ends)                           (11.51)
start = median( m_1 … m_5 ),   end = median of the last 10 m_i                                 (11.52)
share_i = (m_i − start) / (end − start)                                                         (11.53)
```

The half frame is the first kept row with `share ≥ 0.5`, the 90 % frame the first with `share ≥ 0.9`. Nothing is given when `|end − start| < 10⁻¹²`. In Analyze and Fitting they are given only when more than one stage is suggested; Compare always gives them.

Implementation: `src/gimap/shared/series_stages/stages.py`

### 11.9 Where the stages are used

- **Analyze ▸ Series**: on the map's own x and halves, over the whole x axis or the x range shown. The record (`<series>_stages.csv` with a JSON) gives every frame's stage, odd flag, reason, level and component scores, and the method in words.
- **Fitting ▸ In-situ series**: on the curves as they will be fitted (halves, range and left-out points of Single analysis), on the q (nm⁻¹) grid of the first usable curve; the others are linearly interpolated, never extrapolated. Curves that cannot be used are skipped and named. The start rule *stages* begins every new stage from the model of Single analysis and otherwise from the previous converged frame.

Implementation: `src/gimap/features/analyze/application/series_stages.py`, `src/gimap/features/fitting/application/series_fit.py`

### 11.10 Compare

The Compare page puts several series side by side. All series must have the same x label.

**Common grid.**

```text
x_lo = max_series ( lowest x with data ),   x_hi = min_series ( highest x with data )           (11.54)
grid = the first series' x in [x_lo, x_hi]          (at least 3 points)                          (11.55)
```

Every frame is interpolated linearly in I onto this grid, only between its own first and last finite points (never extrapolated). Inside that span the interpolation bridges missing values, so a detector gap inside a frame is filled linearly here, unlike in the Series map.

**One representation, one space.**

1. The curves of all frames of all series are stacked and made comparable together (section 11.4): the 95 % coverage is counted over the stack, with the chosen q range and *shape only* (default on).
2. Odd frames are found per series, on that series' block of the common representation (section 11.5). They can differ from the odd frames Analyze shows for the same series, because the grid and range differ.
3. The main components (section 11.6) are computed once from the kept frames of all series. Each series' scores are its rows in that common space. Each component is turned so that, summed over all series, the mean of the last `n_end` kept frames minus the first `n_end` is not negative.
4. Per series: the half and 90 % frames (section 11.8) along the common first component; its own stages (section 11.7) on its block, with its own components (none when it is too short to cut).

**End and start states.** With `N_e` = 10 frames by default (odd frames left out):

```text
e_i = mean of x over the last N_e kept frames of series i                                       (11.56)
a_i = mean of x over the first N_e kept frames of series i                                      (11.57)
δ_ij = 100 · ( 10^{ RMS_b( e_i,b − e_j,b ) } − 1 )   [%]                                        (11.58)
```

`δ_ij` is the typical ratio of the two end-state curves expressed as a percentage of intensity (of shape, when *shape only* is on). The same formula with `a` gives the table at the start. For example, an RMS log10 difference of 0.01 is 2.3 %.

**Groups** (three or more series). Ward linkage (Ward, 1963; SciPy, Euclidean distance) is run on the end-state vectors `e_i` themselves, not on the percentage matrix. With the sorted merge heights `g_1 ≤ … ≤ g_{n−1}`:

```text
r_m = g_{m+1} / max(g_m, 10⁻¹²),    m* = argmax_m r_m                                           (11.59)
no split  ⇔  r_{m*} < 2   or   max_ij δ_ij < 2 %                                                (11.60)
otherwise cut the tree at (g_{m*} + g_{m*+1}) / 2                                               (11.61)
```

Groups are numbered in series order. So "alike" and "different" are read from a jump of the merge heights, not from a fixed threshold; the 2 % floor keeps series that differ by less than about 2 % of intensity in one group (an RMS log10 difference of 0.0086). Below three series every series is in group 1.

**What the results say.** For two series: their end-state and start-state distances. For three or more: the groups and the series that differs most, the one with the largest mean of its row of `δ`. The share shown with the first main change, "Component 1 (… %)" above the change plot, is `γ_1` of (11.40), the share of the variance of all kept frames about their common mean that the first component explains.

Implementation: `src/gimap/features/compare/domain/comparison.py`, `src/gimap/features/compare/application/use_cases.py`

### 11.11 Limits of the series methods

- Stages and odd frames are found in frame order. With uneven time steps (skipped frames, a paused acquisition), a change that is steady in time is not straight in frame order and may be cut.
- Only positive intensities enter (log scale). A q bin with data in fewer than 95 % of the frames is not compared; a long gap in some frames can remove a q region from the comparison.
- *Shape only* removes the overall level. A real change of the total scattered amount is then only in the level column of the exports.
- The thresholds (5, 8, 0.8, 5 %, 2×, 2 %) are working choices stated in the code, not derived from a noise model of the user's detector. Stage boundaries, odd-frame flags and groups carry no confidence values.
- The peak in a window (section 11.2) is a moment estimate with a fixed straight background; its width depends on the window.
- Compare interpolates linearly in I onto the first series' points. A series measured on a much finer grid is compared only at those points.

## 12. Sputter

The Sputter workspace follows an in-situ sputter deposition series. Its **Analyze** runs a separate analysis package (`iampred`, bundled with GIMaP as `utils/sputter_client`) as a subprocess and shows what the package writes. The networks, the physical forward model and the refinement are the package's and are documented with it; this chapter states what GIMaP does with the package's input and output (12.1), with the deposition rate (12.2), and how it draws a solution in real space (12.3).

### 12.1 Analyze: what the main screen shows

**Inputs handed to the package.** The folder of CBF frames; the metal and the substrate; the deposition rate in nm/s, either as the rate at the sample or as the reading of the quartz balance (section 12.2); α_i; the sample–detector distance; the energy or the wavelength; the direct-beam position, entered in GIMaP's pixel coordinates and handed over minus 0.5 px as 0-based pixel centres (section 1.3), or, when it is empty, the request to estimate it from the frames; the sample roll (0° when empty); optionally a detector mask (without one the package uses the pixel mask it was built with), the time of the scan at which the deposition started, its duration, and the thickness of a polymer film. GIMaP changes none of these values. The package keeps an imported scan under a label; GIMaP derives the label from all these inputs, so a changed input is imported anew and never read from an earlier import.

**Quantities.** For every time window the package gives several *entries*: up to three candidates of its network, ranked, and, when it refined, the refined solution and the joint fit of the whole scan (every window refined together over the scan, smooth in time; the package then marks it as lying on its smooth evolution). An entry holds the grain radius R, the grain height H, the centre-to-centre distance D, the gap D − 2R, the width σ of the log-normal distribution of the radius, the coverage (0 to 1) and the metal amount (nm), each as the package defines it for the model family of the sample; a quantity a family does not have is absent and is not shown. Every entry of a grain family also holds the thickness t_b of the continuous metal layer under the grains (0 for separate islands; the grains stand on it, their caps H − t_b high, section 12.3); it is shown as a value only when the result lists it among its quantities (a newer package), otherwise it is used for the real-space picture alone. Lengths are in nm, the time t in s from the first frame; the nominal deposit of a window is the package's value. GIMaP shows the values with three significant digits (two for σ and the coverage) and computes none of them.

**Fit deviation.** The package's measure of how far the model image of an entry is from the measured image (its deviance per cell with a floor for the model error; lower is better). GIMaP shows the number and does not turn it into a grade, a probability or a threshold.

**Which entry is shown.** In this order: the entry the user chose for the window; the entry the package marks as lying on its smooth evolution over the whole scan; the entry it marks as having the lowest fit deviation; the entry with the lowest fit deviation; the first entry. The evolution plots show the entries so chosen, one per window, against t; a window without a solution leaves a gap in the lines.

**Images.** The measured count rate and the model of each entry are arrays over the exit angle α_f (°) and q_y (nm⁻¹) on the package's cells, which need not be evenly spaced. Both are drawn on those cells (cell edges halfway between neighbouring centres), with q_y on a logarithmic axis when every q_y is positive, on identical axes and with one logarithmic colour scale. Its limits are the 1 % and 99.7 % percentiles of the positive measured values (of the model's when nothing was measured), so they do not depend on the entry shown. Cells without data are left empty; measured values below the lower limit, zero counts included, are drawn in the lowest colour. The arrays are not changed.

**Components.** A newer package writes, for every entry, the model image split into the physical terms of its forward model, each alpha_f-smeared and scaled like the model so that the terms add up to the model image (the package checks this to about 10⁻¹⁵): grains and islands with their interference; large clusters; the film term (density fluctuation, connected network or roughness of a continuous film); voids under the continuous layer; buried clusters; metal-rich domains in a polymer; the rough top of the continuous layer; fluorescence (flat in q_y); the substrate scattering seen through the metal; and the beam-path background (the flat share of the pre-state and, where the family has it, the difference term of the beam path, which can be negative). On porous TiO₂ the terms are the clusters in the pores and on the cap layer, the metal network in the pores (the difference with the network strength set to zero) and the substrate seen through the metal. For each entry the package gives each term's share of the model counts on the fitted cells, Σ E c_k / Σ E μ (E the exposure, c_k the term, μ the model), and per cell the term's share of the model image (float16, about three significant digits). GIMaP draws a term's image as that share times the model image, on the colour scale of the measured image (12.1, Images); it draws terms with a share of at least 0.1 % and computes nothing else.

**Growth process.** From the entries shown by default (the joint fit when every window has one, else the smooth path) the package describes the growth against the metal amount h (nm; the equivalent thickness of the joint fit — mass factor × rate × time — so it does not depend on the X-ray wavelength): the amounts at which the coverage first reaches 0.5 and at which half of the metal is in the continuous layer (the film forms; log-linear interpolation between windows); the power laws D = D₁ h^p and R = R₁ h^q fitted by least squares in ln h over the windows before the film forms (at least three windows spanning at least a factor e^0.3 in h); the median D and R once the film has formed; for a scan whose deposition stopped, the amount then and the change of ln D per decade of time after it; the median fit deviation; and the number of steps back between neighbouring windows (R by more than 10 %, D by more than 5 %, the coverage by more than 0.03). The stage of a window is *islands* below coverage 0.5, *coalescence* above it, *continuous film* once half of the metal is in the continuous layer and *after the deposition* once the amount stays constant, taken on the running maximum of the coverage and of the film share, because sputtering only adds metal; on porous TiO₂ the stages are *clusters* and *metal film* (the film term above half of its largest value). The package lists the median and range of each descriptor over the scans of the same metal on the same substrate it was built from (development reference, without independent characterisation) and whether this scan lies within, below or above that range (only with at least two scans). GIMaP shows these values and draws the stages as bands behind the evolution plots (a boundary halfway between the windows of neighbouring stages); it computes none of them.

**Growth-law prior.** With the prior on (*Analyze options*), the joint fit of the whole scan gets, for every window before the deposition stopped and for each law X ∈ {D, R}, the residual r = √(w g(h)) (ln X − ln X₁ − p ln h) / σ(h), σ(h)² = s_a² + (ln h)² s_p², g(h) = 1 / (1 + (h / h_base)⁶). ln X₁ and p are the medians of the power laws above over the package's reference scans of the same metal on the same substrate, s_a = max(half their range of ln X₁, 0.15), s_p = max(half their range of p, 0.10), h_base the median amount at which the film forms, and w = 1, so a departure of one width costs as much as one unit of fit deviation in a window. A law of fewer than two reference scans is not used. The same terms also weigh the candidates when the start path through the scan is chosen. *Auto* switches the prior on only for the systems where it passed a leave-one-out test whose criteria were fixed before the runs (each reference scan fitted with the laws of the other scans of its system and compared with its joint fit without the prior: median and sum of the fit deviation at most 1.15 times, no more steps back, and on most scans no larger jumps Σ(|Δ ln D| + |Δ ln R|) between neighbouring windows before the film): at present Ag on PS. The laws are model values of the reference scans (development reference, without independent characterisation).

### 12.2 The deposition rate

The rate is typed in Å/s (what a quartz balance shows) or in nm/s and handed to the package in nm/s: r [nm/s] = r [Å/s] / 10, rounded to 12 significant digits so that a typed 2.2 Å/s is 0.22 nm/s exactly. With *QCM reading* the number is the balance's reading r_QCM and the package takes the rate at the sample as

r_sample = 0.6 r_QCM                                                     (12.1)

(the factor is the package's, `iampred/intake.py`); otherwise r_sample = r. A rate at the sample outside 0.005 nm/s ≤ r_sample ≤ 1 nm/s makes the page ask whether the unit is right before the analysis starts. The range is a check of the unit (a QCM reading taken as nm/s is ten times too fast), not a physical limit: the analysis runs with any positive rate once confirmed.

The line under the input bar states the nominal thickness reached at the last frame,

δ_end = r_sample (min(t_N, t_0 + Δt) − t_0),   t_N = N τ                 (12.2)

with N the number of frames, τ the exposure period of the first frame's header (its exposure time when there is no period), t_0 the start of the deposition entered (0 when empty) and Δt its duration entered (none when empty). It is shown only; the nominal deposit of each window is the package's value.

Implementation: `src/gimap/features/sputter/application/client.py` (`QCM_FACTOR`, `RATE_LOW_NM_S`, `RATE_HIGH_NM_S`), `src/gimap/features/sputter/presentation/rate_flow.py`

### 12.3 The real-space picture

**Real Space** draws the solution shown at its size; the arrangement is an illustration. The top view is a square of 7 D. Grains sit on a hexagonal lattice of spacing D, each shifted by 0.12 D times a standard normal number in x and in y; the radius of each is r = R exp(σ z), with z standard normal clipped to ±2.5, and the height of its cap above the base layer is (H − b) r/R, b being the thickness of the continuous base layer (`base_nm`; a cap of height R without H). The side view is a cut 4 D long through one row of grains. The random numbers come from one fixed seed and are drawn in the same order for every solution, so another window or solution changes the sizes and not the positions.

Two grains touch when their centres are no farther apart than the sum of their radii; touching grains form a cluster. The arrangement is *islands* when no two grains of the view touch, *network* when a cluster connects two opposite edges of the square (percolated), else *coalescing*. A solution with b > 0 is a closed film: the top view is the metal of the base layer over the whole square, with the grains as relief on it.

Implementation: `src/gimap/features/sputter/domain/real_space.py`, `src/gimap/features/sputter/presentation/real_space_view.py`

## 13. Tool windows and Labs

The tools of this chapter work beside the Analyze → Fitting pipeline. Each keeps its own conventions, and some
of them differ from Analyze on purpose. This chapter states, for each tool, what is computed, in which units and
pixel convention, what is assumed and what is not corrected. Lengths on the detector are in pixels unless a unit
is given.

### 13.1 XRR Series Extractor

Integer negative gap codes, non-finite pixels and the detector reader mask are excluded from the ROI; floating negative data remain valid. Motor units support degrees, rad, mrad and urad (µrad/μrad); unknown units and non-finite angles are rejected. The transformed NeXus header centre follows the same displayed coordinates as Analyze.


**Purpose and assumptions.** The extractor turns an angle series of area-detector frames into a specular
reflectivity curve, one point per frame (Tools ▸ XRR Series Extractor). It assumes one geometry:

- the beam and the detector are fixed; the sample turns by θ about a horizontal axis through the beam;
- the detector is flat and perpendicular to the direct beam, with no tilt;
- the scattering plane is vertical, so the specular beam moves along a detector column only;
- θ is the sample angle as read or computed, without any offset.

**Wavelength and q_z.** λ follows from the photon energy E in keV by (1.1), and

```text
q_z = 4π sin θ / λ                                                               (13.1)
```

This is the specular wavevector transfer (Als-Nielsen & McMorrow, 2011). q_z is in Å⁻¹. It is the vacuum value: refraction in the film is not taken into account.

**Position of the specular beam.** A sample turned by θ sends the specular beam to a scattering angle 2θ from the
direct beam. On a detector at distance L:

```text
x_s = x_b                                                                        (13.2)
y_s = y_b + s_dir · L · tan(2θ) / p_y                                                (13.3)
```

| Symbol | Meaning | Unit, default |
|---|---|---|
| (x_b, y_b) | direct-beam position at θ = 0 (column, row) | px |
| L | sample–detector distance | m (entered in mm; default 5000 mm) |
| p_y | vertical pixel size | m (entered in µm; default 172 µm) |
| s_dir | direction of the specular beam: −1 "Specular moves up (−y)" (default), +1 "Specular moves down (+y)" | — |
| E | photon energy | keV (default 12 keV) |

The horizontal pixel size p_x is recorded but not used: x_s does not change with θ.

**Integration window.** The intensity of a point is read in a circular region of interest (ROI) of radius r
around (x_s, y_s). Pixel (row i, column j) belongs to the ROI when

```text
r = 0:   j = ⌊x_s + ½⌋ and i = ⌊y_s + ½⌋   (the single nearest pixel)              (13.4)
r > 0:   (j − x_s)² + (i − y_s)² ≤ r²                                             (13.5)
```

Here x_s and y_s are 0-based pixel-centre indices (see below). A pixel is used only when it is finite and not
in the reader's invalid mask. The point's intensity is then

```text
I_sum  = Σ_{valid ROI pixels} I_ij                                               (13.6)
I_mean = I_sum / N_valid                                                         (13.7)
```

`Sum` is the default; r has a default of 2 px (range 0–500 px). N_valid is written with each point. A ROI with no
valid pixel gives no intensity (an empty CSV cell). A circle of r = 2 centred on a pixel holds 13 pixels; one of
r = 1 holds 5. The ROI is clipped at the frame edges.

Which pixels are invalid is the reader's decision (section 4.1): in a CBF frame, negative values
and non-finite values; in a NeXus/HDF5 frame, the rejected `pixel_mask` bits, the maximum of an unsigned type,
and the parts of a stitched canvas that no module covers. The extractor does not apply Analyze's bad-pixel
search, drawn masks or thresholds. The XRR adapter additionally excludes negative codes in signed-integer frames; floating negative values remain data.

**θ of each frame.** Either a linear sequence or a motor dataset in the NeXus file:

```text
θ_n = θ_start + n · Δθ,    n = 0, 1, … (position in the whole series)            (13.8)
θ_n = v_n · c_unit,         c_unit = 180/(1000π) for "mrad", 180/π for "rad", 1 otherwise   (13.9)
```

Defaults: θ_start = 0°, Δθ = 0.01°. In (13.9) the `units` attribute of the dataset decides; `deg`/`degree`/`degrees`, `rad`, `mrad` and `urad` (also µrad/μrad) are supported. Missing units retain the documented degree default; unknown units are rejected.

**Pixel coordinates.** The extraction works internally in 0-based pixel-centre indices (the index convention of
section 1.3: the centre of the first pixel is (0, 0), x is the column, y the row, row 0 at the top). The window
shows and accepts canonical pixel coordinates, as Analyze and Geometry Calibration do, and converts with (1.5). The exported CSV (`roi_center_x_px`, `roi_center_y_px`) and the JSON record (`geometry.beam_center_x_px`,
`beam_center_y_px`) keep the index values, 0.5 px lower than the window shows; the record states this in
`center_convention`.

**Where the geometry comes from.** Each value is tagged with its source. A value typed by the user is never
replaced. Otherwise the metadata of the first frame (`file`) replaces the last applied Geometry Calibration
(`calibration`), which replaces the built-in default (`default`). A calibration made on frames of another shape
contributes only its energy. Without any beam centre, the frame centre is used (`image center`).

**Worked example (invented geometry).** E = 10 keV gives λ = 1.23984 Å by (1.1). For θ = 0.5°, (13.1) gives
q_z = 0.0884475 Å⁻¹. With L = 2 m, p_y = 172 µm and s_dir = −1, (13.3) moves the specular beam by −202.97 px from the
direct beam.

**What is not corrected or modelled.** The intensities are raw ROI sums or means of the frame values. The
extractor applies none of the following:

- background subtraction (the diffuse scattering inside the ROI is part of I);
- footprint correction;
- monitor, flux or exposure-time normalisation;
- polarisation, absorption or resolution corrections;
- refraction (q_z of (13.1) is the vacuum value);
- a θ offset (sample misalignment) or detector tilt.

No uncertainty is written. For photon counts and `Sum`, √I_sum is the Poisson estimate; GIMaP does not write it. No reflectivity model (for example Parratt, 1954) is fitted: the tool only produces the curve.

Two limits of the inputs are known. The beam centre stored in the header of a NeXus file written by the Lambda
server is taken without the transpose and flip applied to the frame, so it can be misplaced (User Manual,
known limitations). A CBF frame without an energy in its header takes the energy of a NeXus file found in a
sibling folder of the scan; the record then marks the energy as `file` without naming that file.

Implementation: `src/gimap/features/xrr/domain/reflectometry.py` (equations), `src/gimap/features/xrr/application/use_cases.py`
(θ, one frame at a time), `src/gimap/features/xrr/infrastructure/adapters/detector_series.py` (series, angle units),
`src/gimap/features/xrr/presentation/pixel_coordinates.py` (coordinates).

### 13.2 SEM Particle Analysis

**Purpose.** The tool measures projected particle sizes in an electron micrograph, or reads sizes measured
elsewhere, and can compare their radius distribution with a distribution predicted by 2D Prediction (Tools ▸ SEM
Particle Analysis). Measurements are made on the pixel grid of the original image.

**Image input.** Pillow reads the image. A palette image is first converted to RGB. A colour image becomes grey
by Rec. 709 luminance, alpha ignored:

```text
I_grey = 0.2126 I_R + 0.7152 I_G + 0.0722 I_B                                               (13.10)
```

A grey image keeps its stored values. The SHA-256 and size of the file, the stored shape and type, the image
mode and the grey conversion are recorded.

**Scale.** The scale s (nm per pixel) comes only from the user, either typed or from a scale bar of known length:

```text
s = b_nm / b_px                                                                  (13.11)
```

with b_nm the bar length in nm and b_px its length in pixels, both `> 0`. GIMaP does not read the scale-bar text,
the magnification or the DPI of the file. Without a positive scale nothing is extracted.

**ROI.** `(left, top, right, bottom)` in integer pixel edges of the original image, row 0 at the top, with
`0 ≤ left < right ≤ W` and `0 ≤ top < bottom ≤ H`. The default is the whole image.

**Threshold segmentation.**

With the threshold method, the threshold t is either typed (in stored intensity units) or found by Otsu's method (Otsu, 1979)
on the finite ROI values. The histogram has 256 bins between the minimum and maximum value, with bin centres
c_k and counts n_k:

```text
W(k) = Σ_{j≤k} n_j,   M(k) = Σ_{j≤k} n_j c_j,   W_T = W(255),   M_T = M(255)       (13.12)
S(k) = (M_T · W(k) − M(k) · W_T)² / [W(k) · (W_T − W(k))]                         (13.13)
t = c_{k*},   k* = argmax_k S(k)                                                  (13.14)
```

S(k) is proportional to Otsu's between-class variance. The threshold is quantised to a bin centre; when all
values are equal, t is that value. The foreground is

```text
bright particles:  F = { I > t and I finite }                                     (13.15)
dark particles:    F = { I ≤ t and I finite }
```

Without splitting, each 8-connected component of F is one object. With **Split touching particles (watershed)**,
a marker-controlled watershed (Meyer, 1994) on the Euclidean distance transform E_d of F is used:

1. Markers are the plateaus of local maxima of E_d in a (2w+1) × (2w+1) window, with w the watershed peak
   distance (default 3 px). Each plateau keeps one marker, its pixel nearest to the plateau centroid.
2. A priority flood assigns the foreground pixels to the markers in order of decreasing E_d, through
   8-neighbours, without leaving F.

**StarDist instances (optional).**

A StarDist 2D model (Schmidt et al., 2018) can find the objects instead. It runs out of process in a separately installed Python, on
the CPU only, with a time limit of 900 s. The ROI is normalised first:

```text
x' = (σ_p · x − P_lo) / (P_hi − P_lo)                                             (13.16)
```

Here σ_p = +1 for bright and −1 for dark particles. P_lo and P_hi are the percentiles p_lo and p_hi (defaults 1 and
99.8) of σ_p·x over the finite ROI pixels. Non-finite pixels are filled with the median of the finite ones, and
there is no clipping. The model is read from a local folder (`config.json`, optional `thresholds.json`, and
`weights_best.h5` or else `weights_last.h5`); nothing is downloaded. Inference uses

```text
predict_instances(x', axes = "YX", scale = (a, a), n_tiles = max(1, ⌈n_ROI · a / 768⌉) per axis,
                  prob_thresh, nms_thresh)                                        (13.17)
```

with the AI image scale a (default 1; the window offers 0.25–4), n_ROI the ROI length along each axis in pixels, and
the model's own thresholds unless others are given. StarDist returns labels on the original ROI grid; labels on non-finite pixels are set to 0. The scale a
changes only the model input, never the measurement grid. The default model, `2D_versatile_fluo`, was trained on
fluorescence images of nuclei and is not validated for SEM.

Before and after the run GIMaP computes the SHA-256 of every model file it reads (and of a
`model_provenance.json` when the folder has one). If any file changed during the run, the result is refused.

**Existing labels.** A label image made elsewhere can be measured. It must be a single page of non-negative
integers within the int32 range, of the size of the image or of the ROI. The IDs are kept; labels outside the
ROI are set to 0, and the IDs clipped by the ROI are recorded. GIMaP does not check that the labels belong to the
image.

**Measurements.**

The same measurements apply to the three methods. For an object of N_px pixels:

```text
A = N_px                     (px)                                                (13.18)
A_nm² = A · s²               (nm²)                                               (13.19)
R = √(A / π) · s             (nm, the radius of the circle of equal projected area)   (13.20)
diameter = 2R                (nm)                                                (13.21)
C = min(1, 4π A / P²)                                                            (13.22)
x_cen = mean(j) + 0.5,   y_cen = mean(i) + 0.5     (canonical pixel coordinates)     (13.23)
```

P is a marching-squares perimeter (the two-dimensional case of marching cubes; Lorensen & Cline, 1987) through the half-level contour of the object mask. Each 2 × 2 cell
contributes 0 (no or four object pixels), √½ (one or three), 1 (two adjacent) or √2 (two diagonal). Holes add
their own contour to P.

**Digitisation bias of C.** C is 1 only for an ideal circle. The marching-squares contour of a straight edge at
angle φ to the pixel rows (0 ≤ φ ≤ 45°) has length cos φ + (√2 − 1) sin φ per unit of true length. Averaged over
φ this gives

```text
P / P_true → (4/π)(2√2 − 2) ≈ 1.0548,    C_disc → 1 / 1.0548² ≈ 0.899             (13.24)
```

for large digitised discs. A synthetic check with discs of radius 5–100 px at random sub-pixel centres gave a
median C of 0.90 at every radius, with a spread of 0.84–0.95 at 5 px, 0.89–0.92 at 20 px and ±0.002 at 100 px.
A digitised square gives about 0.81 (ideal π/4 = 0.785). A minimum circularity above about 0.85 therefore
rejects digitised discs as well.

**Rejection.** An object can be rejected for one or more reasons, all recorded:

| Reason | Default |
|---|---|
| `touches_roi_border` (its bounding box touches the ROI edge) | on |
| `area_below_minimum` | `A < 9` px |
| `area_above_maximum` | no maximum |
| `circularity_below_minimum` | `C < 0` (off) |

Only accepted objects enter the radius distribution. Every result is marked `review_required`.

**Sizes from a CSV file.** A column chosen by the user, with its quantity and unit:

```text
R = v · f_u / m,    f_u = 1 (nm), 1000 (µm) or s (px);   m = 2 for a diameter, 1 for a radius   (13.25)
```

Non-numeric, non-finite and non-positive values are rejected. A `sem_particles.csv` written by GIMaP keeps its
`accepted` flags and its upstream `#` provenance.

**Distribution and comparison with a prediction.**

The SEM distribution is number-weighted: each accepted particle counts once. With N bins (default 30; 2–500 in the
window), common edges and an optional radius scale κ (default 1, applied for the comparison only):

```text
e_k = r_min + k (r_max − r_min) / N,    k = 0 … N                                (13.26)
p_k = n_k / Σ_k n_k,    n_k = number of accepted κR in [e_k, e_{k+1})            (13.27)
```

The last bin includes its right edge. By default r_min = 0 and r_max = max(κR), or the upper edge of the
prediction when it is larger. Particles outside the range are counted and recorded, not normalised in.

A predicted distribution w_l on the axis points a_l (l = 0 … M − 1) is rebinned as a piecewise-uniform density:

```text
b_0 = max(0, a_0 − (a_1 − a_0)/2),   b_l = (a_{l−1} + a_l)/2,   b_M = a_{M−1} + (a_{M−1} − a_{M−2})/2   (13.28)
Φ(b_l) = Σ_{m<l} w_m / Σ_m w_m,   Φ linear between the b_l, 0 below b_0, 1 above b_M      (13.29)
p̂_k = max(0, Φ(e_{k+1}) − Φ(e_k)) / Σ_k max(0, Φ(e_{k+1}) − Φ(e_k))                     (13.30)
```

The mass outside [r_min, r_max] is recorded. In the *Paper display* the SEM histogram has 60 bins and the
prediction is drawn on its own axis with its minimum subtracted and its peak scaled to the largest p_k of (13.27);
the record then says `display_model_is_probability: false`.

**The prediction handed over by 2D Prediction.** It is accepted only when the module's row axis has the key `r`
and the unit `nm`. The radius distribution is the row marginal of the predicted matrix (equation (13.38)), on the
bin centres of the module's axis (equation (13.37)). The comparison holds a frozen copy with the module, model
path, preprocessing steps and settings, runtime and input files.

**What is and is not computed.** GIMaP computes the counts (detected, accepted, rejected, invalid pixels), the
per-particle values of (13.18)–(13.23) and the binned probabilities of (13.27) and (13.30). It computes no
summary statistic: no mean, median, standard deviation, polydispersity index, distribution fit or confidence
interval, and no test of agreement between SEM and prediction. Such numbers must be computed from
`sem_particles.csv` and their method stated. The two distributions are also different quantities: SEM counts
particles in a small field of view, the scattering prediction describes the illuminated area as the module's
model defines it.

Implementation: `src/gimap/features/sem/domain/segmentation.py` (Otsu, watershed, measurements, perimeter),
`src/gimap/features/sem/domain/label_measurements.py` (StarDist and existing labels),
`src/gimap/features/sem/domain/comparison.py` (distribution, rebinning),
`src/gimap/features/sem/infrastructure/stardist_worker.py` (normalisation, inference),
`src/gimap/features/sem/infrastructure/instance_runtime.py` (model files, labels),
`src/gimap/features/sem/infrastructure/readers.py` (image and CSV input).

### 13.3 Format Converter

**Purpose.** The converter writes detector frames in another container format (Tools ▸ Format Converter). It reads
NeXus/HDF5, CBF and TIFF with the shared reader that Analyze uses, so a converted frame has Analyze's orientation:
module sets are stitched, Lambda-server files get the P03 transform, other NeXus/HDF5 images are read as stored.
EDF is not an input. Outputs are TIFF, CBF, HDF5 (`/entry/data/data`, NXdata, gzip) and NumPy `.npy`, or one
NeXus/HDF5 container with one dataset per frame.

**Pixels without data.** For each frame the converter first decides which pixels hold no data, with the validity
rule (4.1) of Analyze: a pixel holds no data when its value is not finite, when the reader masks it, or when it is
negative and the stored type is not floating point. The reader's mask holds the rejected NeXus `pixel_mask` bits, the maximum of an unsigned type, the parts of a
stitched canvas that no module covers. Negative values in integer data are the gap and
bad-pixel codes of counting detectors (Pilatus −1, −2).

**Values and type.** The reader returns float32 values together with the stored type (`stored_dtype`). The data mode
decides what is written:

| Data mode (window text) | Written values |
|---|---|
| Keep the stored data type (default) | the stored type and every stored value |
| Save as 32-bit float | float32 |
| Convert to uint16 using the original data range | uint16, equation (13.31) |
| Clip to the uint16 range | uint16, equation (13.32) |

In the default mode an integer frame keeps its stored type and values, its stored negative codes included. A pixel
without data that has no stored code (NaN from the reader) is written as −1. An unsigned frame that needs such a
code is written in the smallest signed type that holds its values, except in HDF5, where it keeps its type and the
largest value of the type marks the pixel (the NeXus convention). A floating-point frame, or a frame whose values
are no longer whole numbers, is written as float with NaN on every pixel without data. CBF holds integers only:
there float values are truncated to int32 and a pixel without data is −1; a negative value (−1 or less) then reads
back as a gap code under rule (4.1), so CBF suits counts only. `pixel_values` counts those pixels
(`negative_data_read_as_gap_codes`), the values truncated (`values_truncated`) and every pixel that reads back
without data (`pixels_read_back_without_data`).

```text
I_u16 = round(65534 · clip((I − I_min) / (I_max − I_min), 0, 1)),   65535 on pixels without data   (13.31)
I_u16 = trunc(clip(I, 0, 65535)),                                     65535 on pixels without data   (13.32)
```

I_min and I_max are taken over the pixels with data; a constant frame gives zeros in (13.31). In (13.32) the
fractional part is cut off, not rounded, and a value at or above 65535 becomes 65535. The value 65535 marks a pixel
without data, as the largest value of an unsigned type does in NeXus.

An HDF5 output, and the one-file container while all frames have one size, also marks every pixel without data in
the NeXus `pixel_mask` (bit 0, "gap") at `/entry/instrument/detector/pixel_mask`.

**What reads back as it was.** On the repository's frame `TestSAXSdata/…_00005.cbf` (int32, 197 849 pixels with gap
codes) a round trip through every output format gave:

- default mode: int32 with the codes −1 and −2 in HDF5, TIFF, CBF and NumPy; GIMaP reads the same 197 849 pixels as
  without data;
- 32-bit float: NaN on those pixels (CBF: int32 with −1); the same pixels read back as without data;
- the uint16 modes: the pixels without data hold 65535. HDF5 reads them back as without data (its `pixel_mask` and
  the NeXus convention): the same 197 849 pixels after (13.31); after (13.32) also the 334 pixels whose values were
  clipped to 65535. TIFF, CBF and NumPy files carry no mask, and their readers take 65535 as a value: there
  the pixels without data read back as valid pixels of 65535.

**Records.** The reader's metadata, with the source, the 1-based frame number, the output name, shape and type, goes
into the header (`GIMaP_metadata`, ASCII JSON for CBF, EDF and TIFF) or the HDF5 attribute `metadata_json`, and into
`conversion_metadata.json`. Its `pixel_values` entry gives the stored type, the written type, the number of pixels
without data, how they were written and the rule (4.1). `conversion_report.json` lists every frame written or
failed.

Implementation: `src/gimap/features/format_converter/infrastructure/adapters/local_files.py` (reading, writing,
records), `src/gimap/features/format_converter/domain/rules.py` ((13.31), (13.32)),
`src/gimap/shared/detector_io/output_values.py` (the rule (4.1), written type and codes), `src/gimap/shared/detector_io/writing.py`
(containers, `pixel_mask`).

### 13.4 2D Prediction (Labs)

**Purpose.** 2D Prediction runs a trained neural network ("prediction module") on GISAXS detector frames and
shows what it predicts: a joint size distribution or a set of sample parameters (Labs ▸ 2D Prediction). It applies
no geometry: no beam centre, no q map and no physical model enter the computation. The result is what the module's
network was trained to output, on the axes the module declares.

**Input.** Only CBF frames are read, directly with fabio (Knudsen et al., 2013), as float32 of the stored values. The shared reader is not
used: there is no P03 transform, no header mask and no gap-code mask, and negative gap codes stay in the image. A
stack of N consecutive files (sorted by name; in a folder batch, each group of *Files per prediction*) is summed:

```text
I = Σ_{f=1…N} I_f                                                                (13.33)
```

**Module.** A module is a folder under `modules/` with a `module.yaml` and its own `preprocess.py`. GIMaP calls the
entry named in `preprocess.entry` with the image, the configured steps and parameters, and the module folder. The
steps are therefore the module's code, not GIMaP's. The shipped modules use these steps:

| Module folder | Steps (in order) | Model input | Output |
|---|---|---|---|
| `Au_Silicon_15nm` | crop (left 800, up 800, down 450, right 250) → resize 256 × 256 nearest → set_invalid (NaN and `I < 0` → −1) → mask → cut_columns (0–20 → −1) → log_and_normalize (maximum without columns 0–10 and rows 240–256) → mask → cut_columns → cut_rows (240–256 → −1) | 1 × 256 × 256 × 1 | joint distribution; rows R 0.05–15 nm, columns H 0.1–30 nm |
| `Ag_Polymer_Silicon_Ale`, `Ag_Polymer_Silicon_Geo` (same code and steps) | crop (200, 800, 300, 700) → resize nearest → experiment_mask (g = 0) → log_and_normalize → mask | 1 × 256 × 256 × 1 | 30 × 30 distribution, no axes declared |
| `Cu_Polymer_Silicon_Sputter` | crop (530, 1100, 300, 670) → resize nearest → experiment_mask (g = −1e-5) → log_and_normalize → mask | 1 × 256 × 256 × 1 | 30 × 30 distribution, no axes declared |
| `Cu_Polymer_Silicon_Sputter_SF` | crop (530, 1100, 300, 670) → resize nearest → detector_cleanup → log_and_normalize → fixed_mask → add_mask_channel | 1 × 256 × 256 × 2 | 4 parameters `t_Cu`, `t_polymer`, `D`, `sigma` |

Crop values are (left, up, down, right) in stored pixels.

**Preprocessing steps as implemented.** For an image of H × W pixels, cropped to H_c × W_c:

```text
crop:     I_c = I[up : H − down, left : W − right]                               (13.34)
resize:   scipy.ndimage.zoom(I_c, (256/H_c, 256/W_c), order = 0)   (nearest neighbour)
```

The logarithmic scaling of `Au_Silicon_15nm` is

```text
I_log = ln(I · e / (I_max + ε) + ε)   where the argument is > 0;   I_log = −1 elsewhere;   ε = 1e-8   (13.35)
```

I_max is the maximum outside the excluded bands. The brightest pixel maps to 1, a masked pixel (−1) maps to −1,
and a valid zero maps to ln ε ≈ −18.42. The Ag, Cu and SF modules use the same expression with I_max over the
whole image; non-finite results become −1.

The `experiment_mask` step of the Ag and Cu modules builds a mask on the resized image:

1. invalid = not finite, or I ≤ g (g = `gap_value_threshold`: 0 for Ag, so zeros are masked; −1e-5 for Cu);
2. a row or column with more than 80% invalid pixels is masked entirely;
3. pixels with `|I − median_5(I)| > b · 1.4826 · MAD` are masked; with the shipped b = 1e6 this test is in effect
   off;
4. two discs are masked: centre (row 0, column 270), radius 30 px, and centre (row 246, column 256), radius 20 px;
5. optional edge bands (Cu module only; zero in the shipped YAML), one binary dilation, and filling of holes.

Before the logarithm, masked and negative pixels are filled with the maximum of the valid pixels; after it, masked
pixels are set to −1. The `mask` step sets the pixels of the module's `assets/mask.npy` to −1. In the SF module,
`detector_cleanup` replaces non-finite values and values ≥ 10⁶ by the maximum of the valid pixels; negative values
are not replaced and become −1 in the logarithm. `add_mask_channel` appends the fixed mask (1 = masked) as a second
channel. The modules differ in how they fail: `Au_Silicon_15nm` stops on an unknown step or a missing mask file,
while the Ag, Cu and SF modules skip both without a message.

**Shape contract.** The output of the module is brought to NHWC. If the model's height and width still differ, a
nearest-index resize is applied, and channels are repeated or cut to the model's number.

**Model.** The network runs with TensorFlow/Keras in a separate worker process. An optional `manifest.json` beside
a `.keras` file may give its SHA-256, which is then checked before loading.

**What is predicted, and the axis contract.**

*Parameter outputs* (`outputs.type` = `parameters`, `sf_parameters` or `sf_4_parameters`): the network output
y ∈ [0, 1] of the first sample is scaled per parameter,

```text
p = y · (p_max − p_min) + p_min                                                  (13.36)
```

with the names and ranges of the YAML. Without them GIMaP falls back to `t_Cu, t_polymer, D, sigma` with
p_min = (0, 10, 4, 0.2) and p_max = (25, 50, 20, 4). No unit is declared or applied.

*Distribution outputs*: a matrix M of n_r rows and n_c columns. Each axis has a key, label, unit, minimum and
maximum from `outputs.axes.row` and `outputs.axes.column`, and its points are bin centres:

```text
a_i = min + (i + ½)(max − min)/n,    i = 0 … n − 1                               (13.37)
P_row(i) = Σ_j M_ij,    P_col(j) = Σ_i M_ij                                       (13.38)
```

The marginals of (13.38) are not normalised. When a module declares no axes, GIMaP uses the key `r`, unit `nm` and
0.05–15 nm for the rows, and the key `h` with 0.05–15 nm for the columns. For `Au_Silicon_15nm`, the YAML comment
says the network was trained with rows = R and columns = H/2, and the column axis shows the full height H
(0.1–30 nm).

**Points to confirm with the model authors.**

- The Ag and Cu distribution modules declare no axes, so they are shown, and handed to the SEM comparison, on the
  default 0.05–15 nm axes that the module never stated.
- The four SF parameters have no units in the YAML; nm for `t_Cu`, `t_polymer`, `D` and `sigma` is likely but not
  stated.

**What is not done.** No uncertainty is given for a prediction. The input is not corrected (no flat field, no
background, no transform), and the preprocessing differs between modules; a module is valid only for frames like
those it was trained on.

Implementation: `src/gimap/features/prediction/infrastructure/adapters/fabio_images.py` (input, stack sum),
`src/gimap/features/prediction/infrastructure/adapters/module_preprocessing.py` (module entry, shape),
`src/gimap/features/prediction/domain/array_contracts.py` (shape, (13.36)–(13.38)),
`modules/Au_Silicon_15nm/preprocess.py`, `modules/Cu_Polymer_Silicon_Sputter/preprocess.py`,
`modules/Cu_Polymer_Silicon_Sputter_SF/preprocess.py` (steps), `src/gimap/integrations/tensorflow/predictor.py` (model run).

### 13.5 Trainset Build (Labs)

This workspace has been removed from GIMaP. Its simulation and training methods are not part of the released application. Existing research files are not changed.

## 14. Provenance and reproducibility

A number in a paper should be traceable to the file, frame, settings and code that produced it, in the spirit of the FAIR principles (Wilkinson et al., 2016). GIMaP writes this
information next to its exports: in JSON records, in `#` header lines of text tables, and on screen. This chapter
states what each record holds, which checksums and seeds make a run repeatable, and what the records do not yet
contain.

### 14.1 Sources of a curve on the detector

In Analyze, the **Sources** button of the detector view draws, in the colours of the plots, the pixels each
reduced curve comes from; clicking a curve shows only its pixels. The mask is rebuilt from the region the curve
records (bands of rows and columns for GISAXS cuts; χ sectors, q windows and q_∥–q_z boxes above the horizon for
GIWAXS) and the valid-pixel mask of the frame. Its pixel count equals the number of pixels counted in the curve's
bins.

Implementation: `src/gimap/features/analyze/domain/sources.py`.

### 14.2 The record `<stem>_analysis.json`

Every curve export of Analyze writes `<stem>_analysis.json` beside the curve files. It holds:

| Group | Keys |
|---|---|
| Program and source | `software` (the record name `GIMaP Analyze 1`), `software_version` (the release, for example `GIMaP v0.2.0`; section 14.5), `source_file`, `frame_index`, `frame_count`, `frame` (1-based label), `summed_frames` (file and index of each frame added) |
| Detector | `detector_name`, `frame_shape`, `measurement` (GISAXS or GIWAXS) |
| Geometry | `geometry` (pixel sizes, distance, beam centre, wavelength, α_i; `rot1_rad`, `rot2_rad`, `rot3_rad`: the detector's rotations of section 3.7; the `dropped_…` and `flat_detector_…` fields of 0.1 are `null`), `geometry_convention` (canonical pixel-corner frame, row 0 at the top, kinematic (vacuum) grazing-incidence q of a flat detector, tilted by the rotations, in Å⁻¹), `instrument_profile` (name, source, time of update; the source begins with `automatic analysis:` when Run Automatic Analysis or `tools/gimap_agent.py` chose the geometry and with `assistant:` when the AI did, as it did for both before 0.2.0) |
| Statistics and binning | `error_model` (`poisson`, `pixel_scatter` or `scaled_poisson`: the three error models of section 6.4, as `photon_counts` and the GIWAXS intensity corrections decided; `null` without a reduction), `binning` (GIWAXS: `radial_bins`, `q_min`, `q_max`, `bin_width` and `unit` of the fixed-edge bins of every I(q), `bins_chosen`, `radial_points` (the points of I(q) after merging), `merge` (the rule of section 6.5), `chi_bin_deg` and `chi_merge`; GISAXS: one point per detector column or row, no merging) |
| Time | `created`: the UTC time the record was written (ISO 8601, for example `2026-11-01T05:52:31+00:00`) |
| Corrections | `corrections` (every setting), `intensity_corrections` (GIWAXS: how applied, formulas, parameters), `bad_pixels_left_out`, `drawn_mask_pixels`, `pixels_filled_from_mirror` |
| Results | `markers` (beam centre, horizon row, cut bands, Yoneda estimate, I(χ) window, peaks …), `curves` (key, title, x label, region of each curve), `messages` |
| Files | `source_files` (every file a frame was read from: module files, master and data files), `input_sha256` (every input file by name with its SHA-256: the source, its module or data files and the frames summed with it), `source_layout` for NeXus/HDF5 (kind, summary, dataset path, `transformations`), `files` (the files written) |

The numbers are those of the reduction; `geometry` is the geometry that was applied.

Implementation: `src/gimap/features/analyze/application/export_record.py`, `src/gimap/features/analyze/infrastructure/adapters/csv_export.py`.

### 14.3 Header lines of the text tables

Text tables begin with `#` lines, so that `numpy.loadtxt`, Origin and spreadsheet programs skip them. They are
UTF-8 without a byte-order mark; GIMaP's own symbols are written in ASCII (`1/A`, `deg`).

- Curve CSV `<stem>_<curve>.csv`: a title line `# GIMaP Analyze export (q in 1/A, angles in deg)`, then
  `# source:` (the path as it is), `# frame:`, `# sum of …` when frames were summed, `# files:` when the frame was read
  from several files, the record lines of section 14.10 (`# GIMaP v… (Analyze); settings and decisions in
  <stem>_analysis.json`, one `# input: <file> sha256 <digest>` per input file, `# geometry:`), `# curve:` and
  `# region:` (JSON). The fit input, the q map and the cake carry the same record lines.
- Fit input `<stem>_fit_input.dat` (q in Å⁻¹, the curve Fitting reads): the title
  `# GIMaP Analyze fit input (q in 1/A)`, the source, frame and curve lines, `# observation:` (JSON) and
  `# columns:`. The observation says how the points were measured: `source` (`native_detector_columns` for
  GISAXS, `radial_bins` for GIWAXS), `file_format`, `intensity_unit` (`counts_per_pixel`), `counting_model_valid`
  (true only for photon counts in the sense of section 6.4, without GIWAXS intensity corrections and without a
  reference-peak normalisation; whether σ may be taken from Poisson statistics), `threshold_enabled` (a value range
  was set), `gap_guard_px`, `summed_frames`, for GISAXS the halves a fit should take when they were chosen, and for a
  frame of a multi-frame source `frame_index` (0-based) and `frame_count`. The file is written under a temporary name
  and renamed when complete.
- q-map and cake CSV: a title line saying what the cells hold (mean intensity per cell, blank = no pixel), the
  source and frame lines and a line naming the row and column axes.

An example with an invented file name:

```text
# GIMaP Analyze fit input (q in 1/A)
# source: D:/example/film_A_00042.cbf
# frame: 1 of 1 (index 0)
# curve: Horizontal cut I(qy)
# observation: {"counting_model_valid": true, "file_format": "cbf", "gap_guard_px": 1, "intensity_unit": "counts_per_pixel", "source": "native_detector_columns", "summed_frames": 1, "threshold_enabled": false}
# columns: qy (1/A)  I  sigma  pixels
```

Frames are counted from 1 where a person reads them and by their 0-based index in file names; the `# frame:` line
gives both. A header line names at most 12 files; the JSON record names all.

Implementation: `src/gimap/features/analyze/infrastructure/adapters/csv_export.py`.

### 14.4 Pixels left out

`bad_pixels_left_out` gives, per frame, the numbers of hot pixels, dead pixels (counting detectors only) and pixels
on defective detector lines that were excluded automatically (the rules are in section 4.4). `drawn_mask_pixels` is the number of pixels of the user's mask, and `pixels_filled_from_mirror` the
number of pixels filled from the mirror image. A value `null` means that the step was off. The records give counts,
not the pixel positions; the positions follow again from the same frame and settings.

### 14.5 Version of the software

The release version is written in one place, `src/gimap/shared/version.py`; `SOFTWARE` is `GIMaP v` followed by that
version (`GIMaP v0.2.0` for this release). These records name the release that wrote them, under the key
`software_version` with the value `SOFTWARE` (section 14.9 lists what still does not):

| Record | Where the release is | Other key that names the program |
|---|---|---|
| Analyze `<stem>_analysis.json` (also `curves/` of a Batch Export, the AI's exports), settings file `analyze_settings.json`, series map and its tables' `.json` | `software_version` | `software`: `GIMaP Analyze 1`, the name of the record format, not the release |
| Analyze stages table `.json`; Compare `compare_series.json` / `compare_frames.json` | `software_version` | `software`: `GIMaP` |
| Batch Export and Send Series to Fitting: `<batch>_batch.json`; `README.txt` | `software_version`; the line `Written by GIMaP v…` | — |
| Analyze series exports (`#` header lines of the tables) | the line `# software_version: GIMaP v…` | `# software: GIMaP Analyze 1` |
| Fitting `<curve>_fit.json` (also after Predict), series table `series_fit.json` | `software_version` | `series_fit.json`: `software`, the same value |
| Geometry Calibration export `<image>.gimap-calibration.json` | `software_version` | — |
| XRR record, SEM `sem_analysis.json`, 2D Prediction `<input>_*.json` and every line of `prediction_results_<time>.jsonl` | `software_version` | — |
| 2D Prediction `prediction_curves_<time>.txt`, `prediction_parameters_<time>.txt` | the line `# Software: GIMaP v…` | — |
| Format Converter: the record of every frame (`conversion_metadata.json`, the JSON per frame, the header of a converted CBF, EDF or HDF5 frame), `conversion_report.json` | `software_version` | — |
| `tools/gimap_agent.py`: `report.json` of every frame, `summary.json`; its CSV tables (section 14.10) | `software_version`; the first `#` line `# GIMaP v… (command line)` | — |

Records written before this release have no `software_version`; their other keys are unchanged. The curve tables
of Analyze and Batch Export name only `GIMaP Analyze 1` in their `#` lines: the `_analysis.json` or batch record
beside them names the release.

### 14.6 Records of the tool windows and Labs

| Tool | Record | What it holds |
|---|---|---|
| XRR Series Extractor | `<name>.json` beside `<name>_xrr.csv` (`gimap-xrr-curve-record`, version 1) | `software_version`; `center_convention`; the column definitions with units; θ first and last; `source_layout` (every file read, the reader's warnings); `series` (source, kind, pattern, angle mode, start and step or dataset); `geometry` (distance, E, λ, pixel sizes, beam centre, direction, pixel convention, the source of each value, and the calibration used); `extraction` (radius, aggregation); `formulas` (13.1), (13.3) and (13.6) as text |
| SEM Particle Analysis | `sem_analysis.json` (schema 1); the same source, settings and comparison as one JSON `#` line at the top of `sem_particles.csv` and `sem_distribution.csv` | `software_version`; source path, SHA-256 and size; grey conversion; scale; ROI; method and every setting; threshold and its method; radius, perimeter and centroid definitions; counts; for StarDist the library versions, Python, normalisation (13.16), scale, tiles, thresholds, model configuration, score range and the SHA-256 and size of every model file; for existing labels the SHA-256 of the label file; the comparison settings, normalisation and rebinning rules, the prediction's source and arrays; plot settings; the list of files |
| 2D Prediction | `<input>_prediction.json` (`gimap.prediction-export/1`) | `software_version`; input files, mode and stack; module name, id, version and YAML file; model path, framework, runtime and its version; the configured steps and the steps applied (names, and the shape of each intermediate image); display settings |
| Format Converter | `conversion_metadata.json` (or one JSON per frame), `conversion_report.json`, header `GIMaP_metadata` | `software_version` (in each frame's record and in the report); the reader's metadata of each frame, source, frame number, output name, shape and type; `pixel_values` (stored and written type, number of pixels without data, how they were written, the rule (4.1)); frames written and failed; in HDF5 the `pixel_mask` |

Implementation: `src/gimap/features/xrr/application/export_record.py`, `src/gimap/features/sem/infrastructure/exporter.py`,
`src/gimap/features/prediction/application/export_record.py`,
`src/gimap/features/format_converter/infrastructure/adapters/local_files.py`.

### 14.7 Checksums

SHA-256 digests identify files whose content matters:

- SEM: the source image, a label image and a CSV source; every StarDist model file, before and after the run
  (a change refuses the result).
- Geometry Calibration: the calibrant image (`image_sha256`).
- Analyze: every input file of an export, by file name (`input_sha256` of `<stem>_analysis.json`: the source, its
  module or data files and the frames summed with it).
- Command line (`tools/gimap_agent.py`): the frame's files, the calibration file and a mask or background file
  when one was used, each with its path, size and SHA-256 (`inputs` of `report.json`); the same digests of the
  frame's files in the `#` lines of every CSV; and for a model fit the SHA-256 of the network's weights
  (`gisaxs.fit.method.network.sha256`; section 14.10).
- 2D Prediction: an optional `manifest.json` beside a `.keras` model may give a SHA-256, which is checked before
  the model is loaded. The digest is not written to the export record.

### 14.8 Seeds and deterministic runs

Where an algorithm draws random numbers, the seed is fixed, so the same input and settings give the same result:

| Where | Seed | Use |
|---|---|---|
| Fitting, global search | 1729 | Latin-hypercube start population and differential evolution |
| Predict parameters (Fit step, 1D Predict window, the GISAXS automatic analysis) | 0 | the network's samples, the local posterior and the search of the ranges (numpy generators) |
| Fitting, manual refinement | `random_seed`, default 1729 | Latin-hypercube population and differential evolution, or Sobol candidates |
| Fitting, AI fitting profile | `random_seed`, default 123 (can be set in the AI settings) | passed to the research code in its subprocess |
| Geometry Calibration, centre estimate | 0 | random pairs of gradient pixels |
| Run Automatic Analysis, Hermans orientation factor | 0 (default) | Monte Carlo error of f |

StarDist inference and the 2D Prediction networks draw no random numbers in GIMaP. The seeds are part of the code
or of the saved settings. Research run records state their seed policy; not every runtime exposes a controllable seed.

The 1D Predict request and summary also record the SHA-256 of the network's record, which names the SHA-256 of the weights, checked when they are loaded. Requests preserve normalized options and state the random-seed policy: Predict parameters uses a fixed seed (0), so a run repeats with the same curve, settings and network. (The physical fit and fit1d, which used their runtime defaults without a fixed seed, were removed in 0.2.0.)

### 14.9 What the records do not yet contain

These gaps are known at the time of writing. A reader who needs the information must keep it separately.

- Current Sputter, AI assistant, CLI and 1D Predict run records include `software_version`. Older records may lack it. Fitting model files and masks/cuts settings still do not identify a release.
- The 2D Prediction record has no hash of the model weights, the YAML file or `preprocess.py`, and it names the
  steps but does not copy their parameters (crop margins, thresholds); these stay in the module's YAML file, which
  can change later. Axes filled in by default (section 13.4) are not marked as such. The copy handed to the SEM
  comparison does carry the parameters.
- The XRR record does not name the companion NeXus file from which a CBF series took its energy.
- The SEM record holds no summary statistics, because none are computed (section 13.2).
- The uint16 modes of the Format Converter write pixels without data as 65535; outside HDF5, which marks them in
  its `pixel_mask`, they read back as valid values (section 13.3).
- Seeds are not written into the records (section 14.8), except the seed of Predict parameters in the command
  line's `report.json` (section 14.10).
- Bad pixels, drawn masks and mirror-filled pixels are recorded as counts, not as positions (section 14.4).
- `<stem>_analysis.json` gives its time of creation in UTC (`created`), as the XRR and SEM records do; the 2D
  Prediction record gives local time without a time zone.

### 14.10 The record of a run without the window

`tools/gimap_agent.py auto` (and `run_standard_pipeline` of its MCP server, when it is given a folder) writes one
folder per frame and a summary of the batch. The record of a frame is `report.json`; it uses the keys of the
window's records:

| Group | Keys of `report.json` |
|---|---|
| Program and time | `software_version` (the release, section 14.5); `created` (UTC, ISO 8601, to the second; one time for every report of a batch and its `summary.json`) |
| Options | `options`: every option of the command line under its own name, as it was given (`frames`: the arguments before patterns and folders were expanded; `calibration`, `mask`, `incidence_deg`, `technique`, `no_fit` …; `notes` and `notes_file`), and `notes_text`: the notes the analysis read. `null` when the files were written through MCP or `call` |
| Inputs | `inputs.frame`: every file the frame was read from (module files, a master with its data files) and every frame summed with it; `inputs.calibration`: the `.poni`, calibration `.json` or image of a standard the geometry was taken from (`null` for a saved instrument profile); `inputs.mask` and `inputs.background`: the mask and the background file when one was used, else `null`. Each file is `{"file": name, "path": the path as given, "bytes": size, "sha256": SHA-256}` |
| Mask | `mask`: `applied` (true or false), `file` (as above), `kind` (a mask image whose non-zero pixels are left out, or a GIMaP masks file with its number of drawn shapes), `masked_pixels`, and `bad_pixels_left_out` (section 14.4): the pixels found automatically are counted apart from the mask. `report.md` says the same under **Mask**, also when no mask was applied |
| Analyze's record | `analysis_record`: the record of section 14.2 for the frame as it was reduced (geometry, corrections, error model, binning, markers, `input_sha256`), without `files` |
| Results | the decisions with their reasons (`decisions`), the open questions (`needs_attention`), the steps, the peaks and rings (GIWAXS) or `gisaxs` (cuts, symmetry axis, halves, length scale, fit), `tables` and `folder` (the frame's folder, relative to the summary); in `summary.json` each frame also has `outputs` (the files written) |
| Model fit | `gisaxs.fit.method`: `name` (`Predict parameters`), `network` (`name`, for example `flow_v1`, and `sha256`: the SHA-256 of the weights file, the value the network's record names and the loader checks) and `seed` (0, section 14.8). `gisaxs.fit.solutions` holds every solution with its model, parameters, probability, χ²ᵣ (`chi2`, with `points`: `fitted`, `given`, `below_window`, `unusable`), `flags`, `warnings` and `more_warnings` (the flags as sentences: of the particles and the solution, and of background, beam tail and resolution) and `intervals` (median, 68 % and 95 % limits; the texts print the refined value with the 68 % limits); `fitted_points` repeats the points of the fit; `curve`, `points` and `q_range_inv_angstrom` say what was fitted, `data` holds the fitted points |

- `summary.json` is one object: `software_version`, `created`, `options`, `arguments` (one sentence per pattern or
  folder among the arguments: how many files it stood for) and `frames`, the list of the frame reports without
  `tables`, `analysis_record` and `options`. Before 0.2.0 the file was that list alone;
  `tools/eval_giwaxs_agent.py --from` reads both.
- Every CSV (`radial.csv`, `horizontal.csv` …, `fit_curve.csv`, `fit_solutions.csv`) is ASCII in its header (one
  spelling of the units in the column names: `qy (A^-1), I, sigma`, `q (A^-1), I, sigma, I_fit`, `chi (deg)`) and
  begins with the same `#` lines: `# GIMaP v… (command line); settings and decisions in report.json`, one
  `# input: <file> sha256 <digest>` line per input file, `# frame:` and `# geometry:` (distance, direct beam,
  wavelength, α_i, detector tilt, and α_c when q_z is refracted). `fit_solutions.csv` then says what the table is
  and the units of its columns. One function writes these lines for the command line and for the files Analyze
  exports in the window (`export_record.provenance_lines`; there the first line reads `(Analyze)`).
- `geometry` holds the values q was computed with, to 15 significant digits (the distance q was computed
  with, not its four-digit rounding); `report.md` rounds them for reading. For GISAXS its `beam_center_px[0]` is
  the symmetry axis the run moved the centre to. `calibration` and `tables.geometry_used` hold the calibration as
  it was read and first applied, before that refinement (`gisaxs.symmetry` has both: `initial_x_px`, `x_px`, and
  `not_applied` when the axis was too far to be used). `geometry.critical_angle_deg` is `null` when no critical
  angle was given (q is kinematic). `tables` keeps the tools' answers as the
  AI saw them (four significant digits), except `geometry_used`. Every crystallite size names its unit:
  `tables.crystallite_sizes[].size_A`, `size_err_A`, `size_nm`, `size_err_nm`, beside `peaks[].size_nm`.
- `report.md` has a section **Record**: the release, each input file with its SHA-256, the calibration file with
  its path and SHA-256, the instrument profile with its source (a profile a tool made from a file names the file's
  full path in brackets), the corrections and the error model. The window's **Save Report…** attaches the same keys
  (`software_version`, `created`, `inputs`, `mask`, `analysis_record`; `options` is `null` there) and writes the
  same section into its web page and Markdown.
- Each frame has its own folder, named by the frame. Frames of the same name in different folders get the nearest
  folder name that tells them apart (`<name>_<folder>`), else a number. An output folder that already holds a
  summary is refused; `--overwrite` first removes the frame folders the earlier `summary.json` lists, nothing else.
- Two runs on the same input with the same options give the same files byte for byte, except `created` and the
  durations (`seconds`) in the JSON records. For that reason an instrument profile the run made itself is recorded
  without its time of update.

Implementation: `src/gimap/app/headless_record.py`, `src/gimap/features/assistant/application/run_record.py`,
`src/gimap/app/headless_inputs.py`,
`src/gimap/shared/file_digest.py`, `src/gimap/features/assistant/application/gisaxs_export.py`.

## 15. What GIMaP does not model

This chapter collects the limits stated in chapters 1 to 14, so that they can be checked in one place. Each item names the section that states it in full. An item is listed once, under the first heading it fits.

### 15.1 Scattering physics

- **Refraction and the DWBA.** Every q of Analyze, Fitting and 1D Predict is the kinematic vacuum value unless Analyze is given the film's critical angle (q_z of one film, section 2.6), and no reflected-wave terms of the distorted-wave Born approximation are computed (sections 2.5, 8.6, 9.2), except
Fitting's optional two-path sphere with a free effective reflection coefficient (section 9.10). The Yoneda band is located empirically, without any critical angle (sections 6.7, 8.1). The film absorption correction and the q_z of the XRR extractor also ignore refraction (sections 5.4, 13.1). The only refraction-corrected quantity is the Sputter island height H(t), which uses the refracted wavevector transfer for the positions of fringe minima, not a DWBA intensity (section 12.12). The GIWAXS report says that q is kinematic in its *Decisions*, with the q_z shift of up to about k sin α_i that refraction can cause near or below the film's critical angle at the frame's α_i (section 7.9); the GISAXS report says so too, with the Yoneda band's q_z shifted most (section 8.6).
- **Absolute intensity.** Intensities are mean counts per pixel. No monitor, transmission, exposure-time, sample-volume, flat-field, detector-efficiency, air- or window-absorption correction is applied (section 2.5). Only the optional GIWAXS geometric factors divide the intensities (chapter 5), and a series can be normalised to a reference peak (section 6.10). The scales of the Fitting models are therefore relative. Every family's form factor is normalised to P(0) = 1, so Int is not number density × ⟨V²⟩ × Δρ² (section 9.4).
- **Lorentz and Ewald-sphere corrections.** I(χ) and the ring intensities are intensities on the detector. No Lorentz factor is applied, and the missing wedge near χ = 0 is not reconstructed (sections 2.2, 7.7).
- **Texture.** The Hermans factor assumes a fibre texture about the surface normal; the missing wedge enters only through the coverage C_w and the isotropic reference f_iso, and the background under a ring is one χ-independent constant (section 7.7). No exit-angle correction (film absorption, transmission function, shadowing) is applied to I(χ), and the sin χ weight makes the bins next to the sample horizon count most: f can depend on them by many σ_f, which the report states but does not correct (section 7.7). The in-plane and out-of-plane comparison uses two sectors of 20°; it is not a pole figure (section 7.6).
- **Phases.** No phase is assigned. Lamellar and hexagonal series are reported as hints (section 7.4).
- **GISAXS models.** The 1D models depend on the magnitude of q_y only; q_z at the Yoneda row and the Yoneda enhancement are not modelled (sections 8.6, 9.2). The beam-tail term is an additive empirical term; no resolution convolution or instrumental smearing is applied (sections 8.6, 9.2). Fitting and Predict parameters use sizes by number unless a component's Sizes says otherwise (sections 8.5, 9.3). Without a given composition the automatic analysis lets the network choose among one or two populations of spheres and upright cylinders; random cylinders and three populations only when given (section 8.6).
- **Paracrystal spread.** Until 0.2.0 the quick fit and 1D Predict reported σ_D/D, √(2π) ≈ 2.51 times smaller than the relative standard deviation of the neighbour distance σnn/D that Fitting and Predict parameters now use (sections 9.5, 13.5); the 1D Predict window gives both.

### 15.2 Instrument and geometry

- **Detector tilt and rotation.** The calibration refines only x_c, y_c and L; it fits no tilt. A tilt comes from a `.poni` file and is applied as pyFAI applies it (section 3.7). The straight horizon, mirror filling, the calibration's line check and the XRR geometry still assume an untilted detector (sections 2.5, 4.8, 3.3, 13.1).
- **Sample orientation.** The sample is tilted by α_i about the horizontal axis perpendicular to the beam and by nothing else; a rotation about the surface normal or about the beam is not described in Analyze (section 2.5). Sputter alone hands a sample roll to its analysis package (section 12.1).
- **Pixels.** A pixel is represented by its centre: there is no parallax, no sensor thickness and no pixel splitting (sections 2.5, 6.1).
- **Beam.** Divergence, energy bandwidth and footprint are not modelled (section 2.5). The XRR extractor applies no footprint correction (section 13.1).
- **Calibration.** The wavelength and the pixel sizes are inputs, not refined. The calibrant constants are fixed and not corrected for temperature. Ring radii stay on a 1 px grid. A "good" calibration agrees with its own image; this does not show that it belongs to the sample data (sections 3.1 to 3.4). The `orientation` entry of newer `.poni` files is not read (section 3.6).
- **Instrumental broadening.** Run Automatic Analysis subtracts no instrumental width, so every automatic Scherrer size is a lower bound. Footprint, divergence, distance, pixel size, binning, strain and paracrystalline disorder all broaden a peak and none is subtracted (section 7.8).

### 15.3 Statistics and uncertainties

- **Choice of the error model.** Poisson errors (scaled when the GIWAXS intensity corrections are on) are used only for frames stored as integers, with no background subtracted and no mirror filling. The choice follows the stored type, not the physical meaning of the values (sections 4.1, 6.4).
- **Independence.** Pixels are taken as independent: charge sharing and correlated noise are not modelled, and no covariance between the points of a curve is computed (section 6.4).
- **Pixel scatter.** Real structure inside a bin inflates the pixel-scatter error; after mirror filling it is somewhat too small (section 6.4).
- **Maps.** q maps, the cake and the series maps carry no σ (sections 6.8, 11.1).
- **Fitted parameters.** All parameter errors are linearised (Wald) errors, not profile intervals. The GIWAXS peak errors are only inflated by χ²_red, the Fitting errors are scaled by it in both directions (sections 7.4, 9.8). σ_f of the Hermans factor is statistical only: it propagates the bin errors (Poisson counting errors or the pixel scatter), not the background, the window, the geometry or the bins next to the sample horizon (section 7.7).
- **Geometry in the uncertainties.** No reported ± of a peak position, a d-spacing, a size or f contains the uncertainty of the beam centre, the distance or α_i, or the q_z shift of refraction; the GIWAXS report says so in *Decisions* (section 7.9).
- **Quantities without an uncertainty.** The symmetry centre (section 8.2), the reference-peak position (section 6.10), the moment width of a peak in a window (section 11.2), stage boundaries, odd-frame flags and Compare groups (section 11.11), XRR points (section 13.1), 2D Prediction outputs (section 13.4).
- **Scores.** The scores of the automatic analysis and the 1D Predict window (χ²ᵣ, RMS/σ, logRMSE) are not probabilities, and mixture weights are not volume or mass fractions. The probabilities and intervals of Predict parameters are those of Fitting's model and its prior (section 9.13): physics outside the model is not in them, and its benchmarks are simulated curves (an independent model, BornAgain). The Sputter reliability numbers come from simulation within the hemisphere model and are exploratory (sections 12.5, 12.14).
- **SEM.** No summary statistic, distribution fit or test of agreement is computed (section 13.2).

### 15.4 Data preparation

- **Bad pixels.** Only single hot and dead pixels and scattered pixels of defective lines are found; clusters and the bad pixels of a background frame are not searched for (section 4.4).
- **Mirror filling** assumes a pattern symmetric in ±q_∥ (section 4.8).
- **Gaps.** Analyze never interpolates across gaps (sections 6.1, 11.1). Compare does interpolate across a gap inside a frame (section 11.10).
- **Reference-peak corrections.** The peak position is limited by the bin width, and the distance scaling is a small-angle relation made exact by iteration (section 6.10).

### 15.5 Series and Sputter

- **Series.** Stages and odd frames are found in frame order, not in time; only positive intensities enter; a q bin with data in fewer than 95 % of the frames is not compared; *shape only* removes the overall level; the thresholds are working choices, not derived from a noise model of the detector (section 11.11).
- **Sputter.** The values are the analysis package's (section 12.1): GIMaP neither computes nor grades them; the rate and t_0 are not measured by GISAXS, and the nominal deposit scales with the rate given; the range of usual rates is a check of the unit, not a limit (section 12.2); the real-space picture places the grains at random (section 12.3).

### 15.6 Tool windows and Labs

- **XRR Series Extractor.** No background subtraction, footprint, monitor, polarisation, absorption, resolution or refraction correction, and no θ offset. Angle units and transformed header coordinates are checked as described in section 13.1.
- **SEM Particle Analysis.** The scale comes only from the user. The default StarDist model is not validated for SEM. The circularity of a digitised disc is about 0.90, not 1. A SEM distribution and a scattering prediction are different quantities (section 13.2).
- **Format Converter.** The uint16 modes lose the marking of pixels without data outside HDF5 (section 13.3).
- **2D Prediction.** No geometry and no correction enter; the preprocessing is the module's own; axes that a module does not declare are filled in by default; the units of the SF parameters are not declared (section 13.4).

### 15.7 Records

The records give counts, not positions, of excluded and filled pixels, do not all name the release version, do not hold the seeds, and the 2D Prediction record holds neither the step parameters nor checksums of the model files. Section 14.9 lists these gaps.

## References

The list follows the style of the IUCr journals: titles are given for books, proceedings and certificates, not for journal articles. An entry without a DOI has none (older books and journals) or is given with its URL. The 25 references built into the literature method of the Sputter workspace are listed in the table of section 12.15; those cited in the text are repeated here.

- Abraham, J. W., Strunskus, T., Faupel, F. & Bonitz, M. (2016). *J. Appl. Phys.* **119**, 185301. https://doi.org/10.1063/1.4948375
- Akaike, H. (1974). *IEEE Trans. Autom. Control*, **19**, 716–723. https://doi.org/10.1109/TAC.1974.1100705
- Als-Nielsen, J. & McMorrow, D. (2011). *Elements of Modern X-ray Physics*, 2nd ed. Chichester: Wiley. https://doi.org/10.1002/9781119998365
- Ashiotis, G., Deschildre, A., Nawaz, Z., Wright, J. P., Karkoulis, D., Picca, F. E. & Kieffer, J. (2015). *J. Appl. Cryst.* **48**, 510–519. https://doi.org/10.1107/S1600576715004306
- Ayer, M., Brunk, H. D., Ewing, G. M., Reid, W. T. & Silverman, E. (1955). *Ann. Math. Stat.* **26**, 641–647. https://doi.org/10.1214/aoms/1177728423
- Baker, J. L., Jimison, L. H., Mannsfeld, S., Volkman, S., Yin, S., Subramanian, V., Salleo, A., Alivisatos, A. P. & Toney, M. F. (2010). *Langmuir*, **26**, 9146–9151. https://doi.org/10.1021/la904840q
- Black, D. R., Mendenhall, M. H., Brown, C. M., Henins, A., Filliben, J. & Cline, J. P. (2020). *Powder Diffr.* **35**, 17–22. https://doi.org/10.1017/S0885715620000068
- Branch, M. A., Coleman, T. F. & Li, Y. (1999). *SIAM J. Sci. Comput.* **21**, 1–23. https://doi.org/10.1137/S1064827595289108
- Bulut, Y. et al. (2024). *Langmuir*, **40**, 22591. https://doi.org/10.1021/acs.langmuir.4c02344
- Gensch, M. et al. (2019). *ACS Appl. Mater. Interfaces*, **11**, 29416. https://doi.org/10.1021/acsami.9b08594
- Gensch, M. et al. (2021). *ACS Appl. Nano Mater.* **4**, 4245. https://doi.org/10.1021/acsanm.1c00829
- Golub, G. H. & Pereyra, V. (1973). *SIAM J. Numer. Anal.* **10**, 413–432. https://doi.org/10.1137/0710036
- Hammersley, A. P. (2016). *J. Appl. Cryst.* **49**, 646–652. https://doi.org/10.1107/S1600576716000455
- Henke, B. L., Gullikson, E. M. & Davis, J. C. (1993). *At. Data Nucl. Data Tables*, **54**, 181–342. https://doi.org/10.1006/adnd.1993.1013
- Hermans, P. H. & Platzek, P. (1939). *Kolloid-Z.* **88**, 68–72. https://doi.org/10.1007/BF01518890
- Hexemer, A. & Müller-Buschbaum, P. (2015). *IUCrJ*, **2**, 106–125. https://doi.org/10.1107/S2052252514024178
- Hodas, M. et al. (2017). *Phys. Rev. B*, **95**, 035424. https://doi.org/10.1103/PhysRevB.95.035424
- Hosemann, R. & Bagchi, S. N. (1962). *Direct Analysis of Diffraction by Matter*. Amsterdam: North-Holland.
- Huang, T. C., Toraya, H., Blanton, T. N. & Wu, Y. (1993). *J. Appl. Cryst.* **26**, 180–184. https://doi.org/10.1107/S0021889892009762
- Huber, P. J. (1964). *Ann. Math. Stat.* **35**, 73–101. https://doi.org/10.1214/aoms/1177703732
- Jackson, B., Scargle, J. D., Barnes, D., Arabhi, S., Alt, A., Gioumousis, P., Gwin, E., Sangtrakulcharoen, P., Tan, L. & Tsai, T. T. (2005). *IEEE Signal Process. Lett.* **12**, 105–108. https://doi.org/10.1109/LSP.2001.838216
- Jiang, Z. (2015). *J. Appl. Cryst.* **48**, 917–926. https://doi.org/10.1107/S1600576715004434
- Kahn, R., Fourme, R., Gadet, A., Janin, J., Dumas, C. & André, D. (1982). *J. Appl. Cryst.* **15**, 330–337. https://doi.org/10.1107/S0021889882012060
- Knudsen, E. B., Sørensen, H. O., Wright, J. P., Goret, G. & Kieffer, J. (2013). *J. Appl. Cryst.* **46**, 537–539. https://doi.org/10.1107/S0021889813000150
- Könnecke, M. et al. (2015). *J. Appl. Cryst.* **48**, 301–305. https://doi.org/10.1107/S1600576714027575
- Künsch, H. R. (1989). *Ann. Stat.* **17**, 1217–1241. https://doi.org/10.1214/aos/1176347265
- Langford, J. I. & Wilson, A. J. C. (1978). *J. Appl. Cryst.* **11**, 102–113. https://doi.org/10.1107/S0021889878012844
- Lawson, C. L. & Hanson, R. J. (1974). *Solving Least Squares Problems*. Englewood Cliffs: Prentice-Hall.
- Levenberg, K. (1944). *Q. Appl. Math.* **2**, 164–168. https://doi.org/10.1090/qam/10666
- Lorensen, W. E. & Cline, H. E. (1987). *ACM SIGGRAPH Comput. Graph.* **21**(4), 163–169. https://doi.org/10.1145/37402.37422
- Marquardt, D. W. (1963). *J. Soc. Ind. Appl. Math.* **11**, 431–441. https://doi.org/10.1137/0111030
- McKay, M. D., Beckman, R. J. & Conover, W. J. (1979). *Technometrics*, **21**, 239–245. https://doi.org/10.1080/00401706.1979.10489755
- Meyer, F. (1994). *Signal Process.* **38**, 113–125. https://doi.org/10.1016/0165-1684(94)90060-4
- Morháč, M., Kliman, J., Matoušek, V., Veselský, M. & Turzo, I. (1997). *Nucl. Instrum. Methods Phys. Res. A*, **401**, 113–132. https://doi.org/10.1016/S0168-9002(97)01023-1
- Morháč, M. & Matoušek, V. (2008). *Appl. Spectrosc.* **62**, 91–106. https://doi.org/10.1366/000370208783412762
- NIST (2017). *Certificate of Analysis, Standard Reference Material 674b, X-Ray Powder Diffraction Intensity Set*. Gaithersburg: National Institute of Standards and Technology. https://tsapps.nist.gov/srmext/certificates/674b.pdf
- Otsu, N. (1979). *IEEE Trans. Syst. Man Cybern.* **9**, 62–66. https://doi.org/10.1109/TSMC.1979.4310076
- Parratt, L. G. (1954). *Phys. Rev.* **95**, 359–369. https://doi.org/10.1103/PhysRev.95.359
- Patterson, A. L. (1939). *Phys. Rev.* **56**, 978–982. https://doi.org/10.1103/PhysRev.56.978
- Pospelov, G., Van Herck, W., Burle, J., Carmona Loaiza, J. M., Durniak, C., Fisher, J. M., Ganeva, M., Yurov, D. & Wuttke, J. (2020). *J. Appl. Cryst.* **53**, 262–276. https://doi.org/10.1107/S1600576719016789
- Renaud, G., Lazzari, R. & Leroy, F. (2009). *Surf. Sci. Rep.* **64**, 255–380. https://doi.org/10.1016/j.surfrep.2009.07.002
- Robertson, T., Wright, F. T. & Dykstra, R. L. (1988). *Order Restricted Statistical Inference*. Chichester: Wiley.
- Ryan, C. G., Clayton, E., Griffin, W. L., Sie, S. H. & Cousens, D. R. (1988). *Nucl. Instrum. Methods Phys. Res. B*, **34**, 396–402. https://doi.org/10.1016/0168-583X(88)90063-8
- Scherrer, P. (1918). *Nachr. Ges. Wiss. Göttingen Math.-Phys. Kl.* **1918**, 98–100.
- Schmidt, U., Weigert, M., Broaddus, C. & Myers, G. (2018). *Medical Image Computing and Computer Assisted Intervention – MICCAI 2018*, Lecture Notes in Computer Science, Vol. 11071, pp. 265–273. Cham: Springer. https://doi.org/10.1007/978-3-030-00934-2_30
- Schwartzkopf, M. et al. (2015). *ACS Appl. Mater. Interfaces*, **7**, 13547. https://doi.org/10.1021/acsami.5b02901
- Schwarz, G. (1978). *Ann. Stat.* **6**, 461–464. https://doi.org/10.1214/aos/1176344136
- Sen, P. K. (1968). *J. Am. Stat. Assoc.* **63**, 1379–1389. https://doi.org/10.1080/01621459.1968.10480934
- Smilgies, D.-M. (2009). *J. Appl. Cryst.* **42**, 1030–1034. https://doi.org/10.1107/S0021889809040126
- Storn, R. & Price, K. (1997). *J. Glob. Optim.* **11**, 341–359. https://doi.org/10.1023/A:1008202821328
- Theil, H. (1950). *Proc. K. Ned. Akad. Wet.* **53**, 386–392, 521–525, 1397–1412.
- Thompson, P., Cox, D. E. & Hastings, J. B. (1987). *J. Appl. Cryst.* **20**, 79–83. https://doi.org/10.1107/S0021889887087090
- Tiesinga, E., Mohr, P. J., Newell, D. B. & Taylor, B. N. (2021). *Rev. Mod. Phys.* **93**, 025010. https://doi.org/10.1103/RevModPhys.93.025010
- Virtanen, P. et al. (2020). *Nat. Methods*, **17**, 261–272. https://doi.org/10.1038/s41592-019-0686-2
- Ward, J. H. (1963). *J. Am. Stat. Assoc.* **58**, 236–244. https://doi.org/10.1080/01621459.1963.10500845
- Wilkinson, M. D. et al. (2016). *Sci. Data*, **3**, 160018. https://doi.org/10.1038/sdata.2016.18
- Yoneda, Y. (1963). *Phys. Rev.* **131**, 2010–2013. https://doi.org/10.1103/PhysRev.131.2010
