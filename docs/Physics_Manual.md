# GIMaP Physics and Methods Reference

This reference states what GIMaP computes, from a detector frame to a reported number. It gives each equation as implemented, with its symbols, units, defaults and thresholds, its assumptions, what it leaves out, and how its uncertainties are estimated. With the equations and the settings in the export records, a reader can reproduce a reported number. Each method names in one short clause where the program uses it; the windows and buttons are described in the User Manual.

Applies to GIMaP 0.2.0.

In **Help ▸ Physics and Methods** this reference has an expandable chapter outline and **Search this document**
for headings and text. Both work offline; click a result to jump to its section. Equations are in fixed-width
blocks; scroll a wide block sideways to see the whole equation and its number.

The [User Manual](User_Manual.md) (`docs/User_Manual.md`, Help ▸ User Manual) explains how to use the program; this reference explains what the program does with the data. Both describe the same version. Please report any place where they disagree, or where this text disagrees with the program.

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
  - [3.8 Detector tilt from the rings of the standard](#38-detector-tilt-from-the-rings-of-the-standard)
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
  - [4.10 Flat field (detector response)](#410-flat-field-detector-response)
  - [4.11 Dark frame](#411-dark-frame)
  - [4.12 Normalisation: exposure, monitor, transmission, thickness, absolute scale](#412-normalisation-exposure-monitor-transmission-thickness-absolute-scale)
- [5. Intensity corrections for GIWAXS](#5-intensity-corrections-for-giwaxs)
  - [5.1 Definition](#51-definition)
  - [5.2 Solid angle](#52-solid-angle)
  - [5.3 Polarisation](#53-polarisation)
  - [5.4 Absorption in the film](#54-absorption-in-the-film)
  - [5.5 Uncertainty of corrected photon counts](#55-uncertainty-of-corrected-photon-counts)
  - [5.6 Lorentz factor of a fibre texture](#56-lorentz-factor-of-a-fibre-texture)
- [6. Reduction to curves and uncertainties](#6-reduction-to-curves-and-uncertainties)
  - [6.1 Principles](#61-principles)
  - [6.2 Fixed-edge bins](#62-fixed-edge-bins)
  - [6.3 Native detector lines (GISAXS)](#63-native-detector-lines-gisaxs)
  - [6.4 Curve error models](#64-curve-error-models)
  - [6.5 Merging sparse bins along q](#65-merging-sparse-bins-along-q)
  - [6.6 GIWAXS curves](#66-giwaxs-curves)
  - [6.7 GISAXS cuts](#67-gisaxs-cuts)
  - [6.8 The q map and the cake](#68-the-q-map-and-the-cake)
  - [6.9 The two halves of a signed cut](#69-the-two-halves-of-a-signed-cut)
  - [6.10 Reference-peak corrections of a series](#610-reference-peak-corrections-of-a-series)
  - [6.11 SAXS/WAXS in transmission](#611-saxswaxs-in-transmission)
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
  - [7.10 Reflection overlay and phase assignment](#710-reflection-overlay-and-phase-assignment)
  - [7.11 Pole figure of a ring and its orientation distribution](#711-pole-figure-of-a-ring-and-its-orientation-distribution)
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
  - [9.14 Models for manual fitting](#914-models-for-manual-fitting)
  - [9.15 Links and constraints between parameters](#915-links-and-constraints-between-parameters)
  - [9.16 Uncertainty after Refine and Search](#916-uncertainty-after-refine-and-search)
  - [9.17 The four-term DWBA with a known substrate](#917-the-four-term-dwba-with-a-known-substrate)
  - [9.18 Several curves fitted together; the vertical cut](#918-several-curves-fitted-together-the-vertical-cut)
  - [9.19 Posterior check by MCMC](#919-posterior-check-by-mcmc)
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
  - [11.12 Neighbour-frame and reference comparisons](#1112-the-timeline-how-far-each-frame-is-from-its-neighbour-and-from-a-reference-frame)
  - [11.13 Chained fits, Predict choices and the continuity prior](#1113-chained-fit-of-an-in-situ-series)
  - [11.14 The results database](#1114-the-results-database)
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
| K, B, D_ff | counts (after ADU conversion) | dark, background and uniform flat exposures, respectively | sections 4.6, 4.10 |
| F_ff | – | flat-field response relative to the mean over retained pixels | (4.11) |
| t_exp | s | exposure time | sections 4.11, 4.12 |
| d | as specified | sample thickness for normalisation (entered in mm) or film thickness for absorption (entered in nm) | sections 4.12, 5.4 |
| A_abs | depends on intensity units | absolute-scale factor | (4.15) |
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
- **Intensities** default to mean counts per pixel. Calibration against a secondary standard gives a differential scattering cross section per unit volume in cm⁻¹ (section 4.12).

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

GIMaP describes the beam, flat detector and sample by the quantities below. Three additional rotations set the detector orientation relative to the beam (section 3.7).

| Symbol | Stored field | Unit | Meaning |
|---|---|---|---|
| `p_x`, `p_y` | `pixel_size_x_m`, `pixel_size_y_m` | m | pixel pitch along the columns (x) and the rows (y) |
| `L` | `distance_m` | m | distance from the sample to the point where the direct beam meets the detector, measured along the beam |
| `x_c`, `y_c` | `beam_center_x_px`, `beam_center_y_px` | px | position of the direct (transmitted) beam, in the canonical coordinates of section 1.2 |
| `λ` | `wavelength_angstrom` | Å | X-ray wavelength |
| `α_i` | `incidence_deg` | ° | grazing angle between the beam and the sample surface; 0 means transmission |
| `ψ` | `sample_roll_deg` | ° | sample roll about the beam, positive toward lab right from lab up, default 0 |

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

This is the pixel-corner convention of pyFAI (Ashiotis et al., 2015) and of a Matplotlib image drawn with `extent = (0, W, H, 0)`. Analyze, Geometry Calibration and XRR Series Extractor show and accept beam centres in these coordinates. They display an applied calibration centre to four, three and two decimal places, respectively. Drawn masks use the same coordinates (section 4.5). The SEM ROI uses the same corner convention, as `(left, top, right, bottom)` in pixels of the original image.

The displacement of a pixel centre from the direct beam, in metres on the detector, is

```text
X_j = (j + 0.5 − x_c) · p_x                                          (1.3)
Y_i = (y_c − (i + 0.5)) · p_y                                        (1.4)
```

X points to the right of the image and Y points up: Y decreases as the row index grows. `X`, `Y` and `L` share one length unit. Example: with `(x_c, y_c) = (3.5, 2.5)` the centre of pixel (row 2, column 3) sits at `X = Y = 0`, and the pixel one row up has `Y = +p_y`.

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

Analyze takes the geometry of a frame from an instrument profile: a stored geometry with a detector name and a frame shape. A profile is made from a calibration (chapter 3), from a pyFAI `.poni` (section 3.6) or entered by hand (Analyze, Geometry step).

- **Order.** (1) The profile picked by the user, when it is for frames of this size or has no stored size. (2) The `.poni` beside the frame: in the frame's own folder, the `.poni` whose name begins like the frame's (or the frame's like its), else the only `.poni` there; it is used when it states a wavelength, its `max_shape` (if stated) equals the frame shape and its pixel size agrees within 1 % with the header's (if stated). (3) The profile matched by score (below). (4) None: nothing is reduced.
- **Another detector.** A profile picked by the user for frames of another size is not used on this frame: the frame then takes (2) or (3), and the message says which profile was set aside and why. Only an explicit **Use Anyway** for that frame size uses it.
- **Score.** Every profile gets a score: a profile with a stored frame shape that differs from the frame scores 0 and is never used; an equal shape adds 1; a detector name that equals or prefixes the other (case-insensitive, runs of spaces ignored) adds 2. The highest score wins; ties go to the most recently updated profile.
- **A `.poni` beside the frame and a matched profile.** The file wins over a profile matched by score, because a profile of the same detector may come from another experiment. A profile whose `source` names that file or the frame's folder wins over the file: it holds a beam centre or geometry saved in Analyze while the file was in use, so the later values are used.
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

The retired Trainset research code, outside the application, keeps the convention of its training data. Its detector grid is

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
2. The incident beam makes a grazing angle α_i with the surface. The sample roll ψ sets the direction of the surface normal about the beam. Both angles are specified independently of the detector rotations.
3. The scattering is elastic and comes from a point at the origin. There is no refraction (the vacuum, kinematic q).
4. Each pixel is represented by its centre.

In the laboratory frame, the axes point along the incident beam, to the right and up. Apply the PONI detector rotations and orientation first to obtain the pixel ray `t = (L, X, Y)` in those axes. A positive ψ tips the sample normal toward lab right, the convention of Sputter’s `sample_roll_deg`. Express the transverse ray in axes aligned with the rolled sample:

```text
X′ = X cos ψ − Y sin ψ,        Y′ = X sin ψ + Y cos ψ
n  = (−sin α_i, cos α_i sin ψ, cos α_i cos ψ)
```

Use `X′, Y′` in place of `X, Y` in (2.1)–(2.9). The incident ray lies on the roll axis. The norm of the ray and the vacuum |q| are invariant. χ, signed q∥, q_z and the exit angle use the rolled components. The inverse projection uses the opposite rotation to place reflections on the detector. The beam polarization stays in the laboratory frame. At ψ = 0 the laboratory-to-sample conversion is

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

Signs at ψ = 0 for a perpendicular detector in orientation 3: `q_y > 0` to the right of the direct beam on the image, `q_z > 0` above the horizon. At the direct beam `q = 0` and `α_f = −α_i`. Three reference points: the direct beam gives `q = (0, 0, 0)`; the horizon (`X = 0`, `Y = L tan α_i`) gives `q = k (1 − cos α_i, 0, sin α_i)`; the specular beam (`X = 0`, `Y = L tan 2α_i`) gives `q = (0, 0, 2k sin α_i)`. For every pixel `|k_f| = k` to 10⁻¹³ Å⁻¹. With `α_i = 0` the map reduces to transmission rings.

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
- **|q| and 2θ are independent of α_i and ψ.** A rotation keeps the angle between `k_i` and `k_f`, so 2θ from (2.12) is the true scattering angle. In the laboratory frame `cos 2θ = k̂_i · k̂_f = L/L_p`. The optional 2θ axis of the GIWAXS I(q) curves converts the q of each point with (2.12); the bins stay uniform in q.
- **χ** (GIWAXS polar angle): 0° along the surface normal (out of plane), ±90° in the sample plane, with the sign of `q_y`. Above the horizon `q_z > 0`, so χ lies between −90° and +90°. Because `q_∥² + q_z² = |q|²`, a point at (|q|, χ) has `q_∥ = |q| sin χ` and `q_z = |q| cos χ` exactly.
- **Missing wedge.** In the plane of incidence (`X′ = 0`, the beam column for an unrotated detector and ψ = 0) `q_y = 0` but `q_x ≠ 0`, so χ is not zero there. With the public P08 geometry (section 2.4), the beam column reaches χ ≈ 2.2° at |q| = 0.73 Å⁻¹, 5.7° at 1.83 Å⁻¹ and 9.0° at 2.87 Å⁻¹. Smaller χ at these |q| is not measured by a flat detector (Baker et al., 2010).
- For grazing-incidence frames, every χ shown or exported by Analyze is the χ of (2.13). Transmission frames use the detector azimuth (section 6.11).

Implementation: `src/gimap/shared/geometry/q_mapping.py`, `src/gimap/features/analyze/domain/giwaxs.py`

### 2.3 Horizon, specular and Yoneda rows

At ψ = 0 on a perpendicular detector in orientation 3, equation (2.8) in the plane of incidence gives the following continuous row coordinates on the beam column.

```text
row(α_f) = y_c − L tan(α_f + α_i) / p_y                               (2.14)
row_horizon  = y_c − L tan α_i / p_y          (α_f = 0)               (2.15)
row_specular = y_c − L tan 2α_i / p_y         (α_f = α_i)             (2.16)
```

and the exit angle at a row on the beam column is `α_f = atan2((y_c − row) p_y, L) − α_i`, which is (2.8) at `X = 0`.

- **Tilted horizon.** For the full laboratory ray of section 3.7, the horizon satisfies `t · n = 0`. This is a line on any flat detector. At α_i = 0 it passes through the direct beam. For a perpendicular detector in orientation 3 its slope in image coordinates is `d(row)/d(column) = (p_x/p_y) tan ψ`.
- **Above the horizon.** The vacuum exit angle satisfies `sin α_f = t · n / |t|`. Tilted GISAXS vertical cuts select pixels individually at α_f ≥ 0 (α_f ≥ α_c for refracted q_z). At ψ = 0 on a perpendicular detector in orientation 3, the whole-row selection in (6.14) applies.

Example (GALAXI geometry of section 1.4): `row_horizon` = 638.3 and `row_specular` = 557.0. The Yoneda band of this frame lies at row 607.5, which (2.14) turns into `α_f` = 0.18°.

**Measuring ψ on one frame.** Two estimators use the calibrated rays and the valid measured pixels, without film absorption, Lorentz weighting or mirror filling. Each searches ψ from −8° to +8° in steps of 0.1°. The first uses the horizon, a straight line on the detector: it scores the intensity drop across that line in strips of columns near the horizon. A free offset lets the observed edge lie above or below the geometric horizon (for example at a Yoneda band). Leaving out contiguous groups of strips shows how much the slope depends on the part of the edge used.

At fixed vacuum |q|, rays on opposite sides of the incidence plane have equal |χ|. Write their laboratory azimuths from vertical as β = atan2(X,Y). For each strong ring, the second estimator minimises the mean of `[I(ψ+β) − I(ψ−β)]²` over the same measured pairs of angular bins at every trial ψ. A ring with a flat profile, an ambiguous minimum or too little coverage on both sides gives no estimate. The uncertainty includes angular sampling, correlated sector residuals and differences between rings. In GISAXS the second estimator fits the direction of a narrow central specular rod in laboratory coordinates.

A shadow edge, Yoneda band or specular rod alone can give the roll. A tilted fibre axis gives the same ring symmetry as a rolled substrate. A GIWAXS ring estimate is therefore applied only when the edge also lies three standard errors from zero and agrees with the rings, or when the user states that the fibre axis is the surface normal. Conflicting estimates give no roll. Two estimates with standard errors s₁ and s₂ agree when `|ψ₁ − ψ₂| ≤ 2√(s₁²+s₂²)`; they are then averaged with inverse-variance weights. Both come from the same frame, so the error of the mean is the larger of the smaller single error and half their difference. A disagreement is recorded with both estimates. Automatic analysis uses a roll supplied for the run or set in Analyze (zero included) and keeps a nonzero roll of the resolved geometry. Otherwise it applies the frame estimate when `|ψ| ≥ 3s`. The errors assume a correct detector calibration.

Implementation: `src/gimap/shared/geometry/detector_geometry.py`, `src/gimap/features/analyze/domain/gisaxs.py`, `src/gimap/features/analyze/domain/giwaxs.py`

**Measuring the external incidence angle.** The direct beam, pixel pitches and detector rotations define a ray for each pixel. Write its components as (b, x, z) along the beam, across it and up in the plane of the rolled sample normal. On the horizon the exit angle is zero, giving αᵢ = atan2(z,b). For an on-axis specular reflection the ray has turned through 2αᵢ, giving αᵢ = ½ atan2(z,b). A GISAXS band at a known material critical angle α_c satisfies (z cos αᵢ − b sin αᵢ)/|r| = sin α_c. On the incidence plane this gives αᵢ = atan2(z,b) − α_c. For a band from a multilayer, choose the interface. For a selected Si, SiO₂, plate glass or Au layer, the program computes α_c ≃ √(2δ) in radians from the density and the wavelength-dependent forward scattering factors of §9.17. Outside the table's 4.895–30 keV the band is left out, the report says why, and the other estimators still run. A known α_c can also be entered directly.

The shadow fit finds its intercept with the input αᵢ set to zero. Its error includes sampling, leaving out contiguous detector strips, and the shift when only strips near the beam are used. The specular search tests incidence angles of 0–3° and needs a localized peak on the calculated ray, above both flanks. To reject Bragg arcs it follows the cone of constant |q|: after subtraction of a local radial background, the peak is rejected when either side keeps at least half the axial signal at 1.5–3 radial widths. A compact diffraction spot exactly on that ray can still pass as a specular reflection. The specular error is half the larger of one sampling interval and FWHM/2.355 of the peak, in scattering angle. The GISAXS band uses the off-axis Yoneda estimate; its width and row sampling give its error, and its illuminated flank does not also count as a shadow.

The program takes the largest set of estimates in which every pair agrees within three combined standard errors. When that set is unique, it reports their inverse-variance weighted mean. The error of the mean is at least the smallest single error and grows with excess scatter between the methods. A selected angle must exceed zero by more than three standard errors. The errors assume a correct calibration, feature identification and supplied critical angle. The weight of estimator j is (1/sⱼ²)/Σ(1/sₖ²) over the selected estimates. The report names an estimator with a weight of 90 % or more, for example a single specular-like spot. The near-beam intercept checks the full shadow trace. The JSON keeps the individual estimates (also when none is selected), the calibrated direct beam, the unsmoothed spot and flank intensities, and the strongest edge position of each strip used.

### 2.4 GISAXS or GIWAXS

When a frame has a geometry, Analyze proposes the technique from the largest scattering angle on the detector. For a flat detector that angle is reached at a corner, so the four pixel corners of the frame are tested:

```text
2θ_max = max over (x, y) ∈ {0, W} × {0, H} of atan2( √( ((x − x_c) p_x)² + ((y − y_c) p_y)² ), L )      (2.17)
```

where `W` and `H` are the numbers of columns and rows. The frame is GIWAXS when `2θ_max > 20°` and GISAXS otherwise. The user can force either mode; Analyze then points out when the forced mode disagrees with this rule. Without a geometry nothing is reduced. `tools/gimap_agent.py` uses the same rule (`--technique` forces it).

**SAXS/WAXS in transmission.** A powder calibrant measured in transmission at a SAXS distance would be GISAXS by this rule. Analyze's Auto therefore takes a frame as transmission (mode **SAXS**, SAXS/WAXS in transmission, section 6.11) when no incidence angle is known (none set, and 0° in the geometry) and either the file name names a standard (the words `agbh`, `agbe`, `behenate`, `lab6`, `ceo2`) or the frame shows three or more complete rings by the test below. An incidence angle above 0 always means grazing incidence. The mode can also be chosen by hand. The ring test runs once per frame and beam centre (`analyze/application/technique.py`).

The ring test averages the valid pixels in 1-pixel rings about the beam centre and in 24 sectors of 15°. A ring is a maximum of the ring average that exceeds its 61-pixel running median by 6 robust σ and by 30 % of that median, and is at most 25 pixels wide at half height. It is complete when the detector covers at least 90° of azimuth at its radius and at least 80 % of the covered sectors show 30 % of its height above their own background. With three or more complete rings Analyze reduces the frame in transmission. The automatic analysis reports it as a transmission frame of a calibrant or powder, runs its GISAXS procedure on it (the cuts are written) and fits no particle model (`analyze/domain/ring_frame.py`, `assistant/application/ring_frames.py`). The public silver behenate frame has five such rings, the public GALAXI GISAXS frame none.

Examples from the public data: the GALAXI frame reaches `2θ_max` = 5.3° (GISAXS). The P08 GIWAXS frame (`giwaxs_p08_mapi`, PerkinElmer 2048 × 2048 px, `p` = 200 µm, `λ` = 0.6888 Å, `L` = 811.8 mm, direct beam (545.5, 1826.9) px from its `.poni`, `α_i` = 0.075°) reaches 30.2° (GIWAXS).

The choice decides the reduction (chapter 6). Mirror filling and the intensity corrections are applied only to GIWAXS frames.

Implementation: `src/gimap/features/analyze/domain/classification.py`

### 2.5 What the q map does not model

Analyze, the q map included, leaves out these effects or applies them only when chosen:

- **Refraction and the distorted-wave Born approximation (DWBA).** `q_z` is the kinematic vacuum value unless the film's critical angle is given (section 2.6). No reflected-wave terms of the DWBA are computed.
- **Detector tilt and rotation** are included: all three rotations enter the pixel rays, q, χ, the solid angle and the polarisation factor (section 3.7). Geometry Calibration fits rot1 and rot2 (section 3.8).
- **Sample azimuth.** The sample axes are fixed by the surface normal and the incident beam. A crystallographic azimuth about the surface normal needs separate information.
- **Parallax and sensor thickness.** A pixel is its centre; there is no depth of interaction and no pixel splitting.
- **Flat field and efficiency.** A flat field (detector response) divides the frames only when one is chosen (section 4.10; off by default). No detector-efficiency, air or window absorption correction is applied.
- **Absolute intensity.** Normalisation by exposure time, monitor, transmission and thickness, and calibration in cm⁻¹, are optional (section 4.12); they act after the flat-field and geometric corrections (section 4.10 and chapter 5). Normalising a series to a reference peak gives a relative intensity (section 6.10).
- **Beam properties.** Divergence, energy bandwidth and footprint are not modelled.

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
- Example: λ = 1 Å, α_i = 0.2°, α_c = 0.15°, α_f = 1.0°: q_z = 2π (0.0172549 + 0.0023088) =
  0.122923 Å⁻¹, against the kinematic 2π (sin 1° + sin 0.2°) = 0.131583 Å⁻¹.
- The model is one homogeneous film with the transmitted waves only, without the reflected-wave terms of the DWBA, a
  substrate or a depth profile. α_c is taken as given (for example from the film's density, or from the Yoneda band,
  section 2.3). A wrong α_c moves q_z most near α_f ≈ α_c.
- The value is a setting of the measurement, kept with the Analyze settings and not in the instrument profile. The
  record gives it as `geometry.critical_angle_deg` (0: kinematic), `geometry_convention` says what q_z is, the CSV headers of
  the command line name it, and the automatic report's *q* decision reads *q_z corrected for refraction in the film
  (αc = …)* (sections 7.9, 8.6).

Implementation: `src/gimap/shared/geometry/q_mapping.py` (`refracted_qz`), `src/gimap/features/analyze/domain/giwaxs.py`,
`src/gimap/features/analyze/domain/gisaxs.py`

## 3. Calibration

Geometry Calibration (Tools menu) finds the beam centre, the distance and the detector tilt (pyFAI's rot1 and rot2, section 3.8) from a powder pattern of a standard. The energy and the pixel size are fixed inputs. The energy comes from the file header (or the companion NXS of a CBF) or is entered, and the record says which (`metadata.energy_source`: `file header`, `companion NXS`, `entered by user`, `imported calibration`). Without an energy, Auto Calibration does not start. The same engine runs without the window in Run Automatic Analysis and in `tools/gimap_agent.py`.

### 3.1 Calibrants

Geometry Calibration accepts the measured beam energy in keV or wavelength in Å.
For a wavelength entry it uses equation (1.1) to fill the energy field. The ring fit uses
the displayed energy in keV (five decimal places). For example, 1.0 Å gives 12.39842 keV.

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
- For AgBH, q runs from 0.10763 Å⁻¹ (n = 1) to 2.1525 Å⁻¹ (n = 20). The relative intensities 1/n provide a weak ordering prior in the final comparison of candidate geometries (section 3.3).
- With the standard set to Auto, all four line sets are tried.
- The values are fixed and do not depend on temperature. The AgBH spacing equals the 58.380(3) Å of Huang et al. (1993). The LaB6 value differs from the certified lattice parameter of NIST SRM 660c (4.156826 Å at 22.5 °C; Black et al., 2020) by 1.8 × 10⁻⁵ relative, the CeO₂ value from that of NIST SRM 674b (5.41153 Å at 22.5 °C; NIST, 2017) by 2.2 × 10⁻⁵ relative.

Implementation: `src/gimap/features/calibration/domain/standards.py`

### 3.2 Ring model

The rings of a powder standard are circles on the flat detector perpendicular to the beam, centred on the direct beam:

```text
2θ = 2 arcsin( q λ / 4π )                                              (3.4)
r = L tan 2θ                    [m]                                     (3.5)
r_px = L tan 2θ / p̄,      p̄ = (p_x + p_y) / 2                           (3.6)
```

The initial radial scan uses the mean pixel size `p̄`. The guided ring search and joint refinement (section 3.3) use `p_x` and `p_y` separately. For a perpendicular detector, the overlay has width `2r/p_x` and height `2r/p_y`. For a tilted detector it follows contours of constant 2θ. A line with `qλ/4π > 1` cannot be reached and is refused.

Implementation: `src/gimap/features/calibration/domain/geometry_model.py`, `src/gimap/features/calibration/domain/ring_geometry.py`

### 3.3 What the engine refines and how

Inputs: the image, the energy (λ from (1.1)), the pixel size (from the header or entered), the standard or Auto, an optional estimated distance, and a distance range of 30 to 10 000 mm. Internally all positions are index coordinates (section 1.3), and so are the candidates of the result and of the exported `.json`; the window, Apply (section 3.5) and the summary of the automatic analysis shift them by +0.5 px with (1.5).

1. **Preprocessing.** Valid pixels are finite, not masked by the reader and above 0, so zero-filled gaps and apertures are left out. Percentiles and the maximum are taken on a sample of about 2.5 × 10⁵ valid pixels. When the maximum exceeds `max(20 P_99.95, 10⁶)`, the pixels at that value are dropped. Values are clipped to `[0, P_99.9]` and transformed with `ln(1 + I)`. By default a smooth background is subtracted: a Gaussian of σ = `max(6, min(H, W)/90)` px, normalised by the smoothed valid mask so that gaps do not bleed in; negative results are set to 0.
2. **Beam-centre proposals.** Up to 14 initial proposals, at least 3 px apart, each with a support between 0 and 1. They come from the direct beam and beam stop along the bright beam column, from voting of intensity-gradient lines of partial arcs (which can place a centre outside the detector), from the image centre, from the header centre (support 0.70), from 180° symmetry by phase correlation (support 0.20), and from a coarse-to-fine search for the sharpest radial profile.
   For transmission and grazing-incidence images alike, the engine also tests refined copies of up to four arc-voting proposals. It centres a radial profile on each trial position and maximises the sharpness of the rings using the brightest quarter of valid pixels. The grid step decreases from 4 to 2, 1 and 0.5 px. A refined centre is added if it lies at least 1 px from every existing proposal.
3. **Ring detection for each proposal.** The image is resampled onto 360 azimuths × 1 px radii. The radial profile is the 60th percentile over azimuth, which is robust against streaks and gaps (the plain radial mean when OpenCV is missing). The profile is smoothed with a Gaussian of σ = 1.2 px. Peaks need a prominence of at least `max(2.3 MAD, 0.08 (P_85 − median), 10⁻⁵)`, an azimuthal coverage of at least 1.5 % and a radius of at least 40 px. Peak radii stay on the 1 px grid.
   For the polar profile, let e(r,φ) be the signal above the local background along the ray at φ.
   That background is a radial Gaussian average (σ = 8 px) over valid samples. At radius r, coverage is the
   fraction of all 360 rays that are valid and satisfy e(r,φ) > max(1.5 MAD_φ(e), 0.005) in the preprocessed
   signal. An arc with similar intensity at every illuminated azimuth can therefore support a peak.
4. **Distance hypotheses and matching.** Each of the first 35 peaks paired with each standard line gives a distance by inverting (3.6). Distances inside the range are clustered within 0.8 %. At each distance, predicted and observed radii are paired in order of increasing absolute difference. For AgBH the tolerance is h = max(2.5 px, 0.012 r). For other standards it is max(3 px, 0.018 r, 0.6 w), where w is the peak width. The paired radii give L by (3.7). If any AgBH pair exceeds its tolerance after this fit, the engine removes the largest |Δr|/h and refits until the remaining pairs agree.
5. **Preliminary ranking.** A weighted score rewards matched rings, azimuthal coverage, valid area, centre support, an unbroken run of orders and prominence, and a Gaussian prior of 20 % width around an estimated distance. A matched ring counts in proportion to its prominence up to 0.1 in the log-scaled profile (about 10 % contrast) and fully above it, so that noise-level peaks do not outweigh fewer clear rings. The score penalises the residual, inconsistent per-ring distances, unexplained inner peaks and (for AgBH) a missing first order.
6. **Joint refinement.** For up to 6 groups of proposals with up to 3 distinct distances each, the beam centre and L are refined together. Ring points are the valid pixels within `h = clip(0.012 r, 3, 10)` px of each predicted ring whose preprocessed signal reaches the 70th percentile of the annulus, at most 800 per ring. The residual (3.8) is minimised with `scipy.optimize.least_squares` (soft-L1 loss, f_scale 2 px, at most 80 evaluations). The centre may move by 5 px (direct-beam proposal), by 40 to 140 px (arc voting) or by 12 to 60 px (other proposals); L may move by ±15 %. Only `x_c`, `y_c` and `L` are free.
   Within 25 % of a supplied distance estimate, hypotheses differing by at least 1.5 % in distance may
   enter the same group's refinement. Farther from the estimate, or without one, the separation is 25 %.
7. **Re-matching.** At the refined centre the rings are detected again and matched at the refined L, and L is solved again with (3.7). The ring-radius RMS from this step enters the preliminary ranking.
8. **Candidate selection.** The guided ring search (below) checks up to 20 distinct geometries. When it confirms at least two lines, these lines enter the point fit. A radial peak resolved on its own also enters when the line spacing leaves less than 2 px of search half-width after its flanks and its angular support is at least 4/72 of a full circle; it takes the untested half-weight in S below. A short arc without enough supporting sectors adds no order. With fewer than two confirmed lines the radial assignments are kept. Up to five geometries go to the point fit: candidates of the guided and radial rankings and, in Auto mode, of competing standards. The search is repeated on the fitted ellipses, and a change of assigned lines starts another point fit. The record holds the SHA-256, size and modification time of the image, the energy, λ, pixel sizes, candidates and image shape. A single matched ring carries a distance-ambiguity warning.

```text
L = Σ_m u_m r_m / Σ_m u_m²,     u_m = tan 2θ_m / p̄     (r_m: observed radii in px)     (3.7)
ρ_p = [ √( ((x_p − x_c) p_x)² + ((y_p − y_c) p_y)² ) − L tan 2θ(q_p) ] / p̄             (3.8)
```

In (3.8), `(x_p, y_p)` are the integer index coordinates of ring pixel p, `q_p` the line it was assigned to, and ρ_p is in pixels.

**Guided ring search.** At a trial geometry, let (a,b,c) be the displacement in metres for one pixel step along
a detector ray, expressed along the incident beam, to its right and upwards. For a standard line q and
t = tan[2 arcsin(qλ/4π)], the predicted intersection lies at r = L t / (√(b² + c²) − a t) pixels from the beam
when the denominator is positive. This follows the ellipse on a tilted detector with either pixel aspect ratio.
The engine samples 72 sectors of 12 rays each, averaging the valid preprocessed signals at integer offsets
from each predicted r. A radial bin needs at least four valid rays. On a recognised grazing-incidence frame,
the search leaves out the specular rod and the half-plane shadowed by the sample.

The search half-width w is ⌈0.012 L t / p̄⌉, clipped to 4–12 px and narrowed where lines are close. The search
and its two background flanks together extend at most 45 % of the nearest predicted line spacing on either side.
Lines below 40 px in equivalent radius, or with less than 2 px available for w, are skipped. Each flank begins
at w + 3 px and ends at most w + 10 px from the prediction. A sector needs four populated bins on each flank
and w + 1 in the search window. Let B₋, B₊ and v₋, v₊ be the sample means and variances on the two flanks.
The peak P is the maximum of the search profile after smoothing with weights (1,2,1)/4. A sector supports
the ring when the peak lies inside both endpoints, P − max(B₋,B₊) > max(4√[(v₋ + v₊)/2], 10⁻⁵), and its
offset lies within 3 px of the median offset of the significant sectors. Acceptance needs at least four
supporting sectors and at least 20 % of the observable sectors.

**Final geometry and ranking.** With at least two matches and enough points, the engine refines the centre
and distance on the sector centroids in section 3.8. Automatic tilt refinement frees rot1 and rot2 with
at least three orders contributing centroids. The comparison of ring assignments uses the following dimensionless score.

```text
S = ½ Σ_confirmed (1 + s/o) + ½ Σ_untested min(1, p/0.1) − ½ Σ_absent o/72
    − U − ln(1 + R²) + 0.2 exp[−½ ((L − L_est)/(0.2 L_est))²] + 0.1 K
```

Here s and o are the supporting and observable sector counts, and p is the prominence of an independently
matched radial peak whose predicted ring has fewer than four observable sectors. An absent line has at least
four observable sectors and fails the acceptance test. For each supporting sector, use the interval between
the smallest and largest detector-pixel radius r(φ) + δₛ on its valid rays at the measured peak offset δₛ.
A match from an untestable arc contributes its observed radial-scan radius. U counts independent radial
peaks of at least 30 % of the strongest prominence whose radius r lies farther from every such interval
or observed radius than max(3 px, 0.018 r) plus the beam-centre displacement since that scan. R is the final ring-point RMS
in pixels (2 when unavailable). The distance term is zero without an estimate. K is the mean sign of
(Iᵢ − Iⱼ)(pᵢ − pⱼ) over pairs of confirmed lines with tabulated intensities I and measured guided peak heights
p. K is zero when a comparison is unavailable. The engine evaluates S again on the final geometry, after any
tilt refinement and refit with newly confirmed orders. The largest S ranks first.

Wavelength, pixel size and rot3 are fixed. Powder rings measure angles to the incident beam and so cannot give a rotation about it; rot3 keeps its supplied value (default 0°). The physical rotations of a candidate are `rot1_deg`, `rot2_deg` and `rot3_deg`.

**Uncertainty.** The sector centroids from one ring can shift together. With n ≥ 3 rings contributing points,
the engine omits each whole ring in turn and refits the remaining centroids. For a parameter θ, the jackknife
variance is σ²_J = (n − 1)/n · Σ (θ_i − θ̄)², where θ_i is its estimate after omitting ring i and θ̄ is the mean
of the n estimates.

- With four or more rings, each refit varies the same parameters as the full fit. With three rings and tilt refinement disabled, each refit varies x_c, y_c and L.
- With three rings and fitted tilts, each pair of rings is first fitted with all five parameters free. All three refits must converge to finite values, have |rot1| and |rot2| below 30° and give 0.75 L_full < L < 1.25 L_full. For each refit, the solver's final Jacobian with columns normalised to unit length must satisfy s_min/s_max > √ε, where s_min and s_max are its extreme singular values and ε is double-precision machine epsilon. When these checks pass, all five parameters receive the joint ring-jackknife σ. This includes the coupling between beam centre, distance and tilt.
- If any two-ring refit fails those checks, all three pairs are refitted with both tilts fixed at their three-ring values. The full five-parameter fit gives C = s² (JᵀJ)⁻¹, where J is the Jacobian of the pixel residuals (3.17) and s² = Σ ρ_p²/(N − 5) for N centroids. Each tilt has σ = √C_ii. For x_c, y_c and L, the reported σ is max(σ_J, √C_ii). If JᵀJ is singular, only the conditional jackknife uncertainties for x_c, y_c and L are available.
- With two rings, uncertainties come from the covariance alone, with denominator N − p for p free parameters. The default fit keeps the tilts fixed, so p = 3. This is a lower bound, because the point residuals within a ring can be correlated.

The estimates assume the ring assignments are correct and omit systematic errors shared by all rings. The calibration
record gives `center_x_sigma_px`, `center_y_sigma_px`, `distance_sigma_mm` and `uncertainty_method`. The result tooltips
show these uncertainties and the instrument profile carries them into each `<stem>_analysis.json`.

**Measured coverage of calibration standard errors.** The benchmark calibrates simulated Pilatus 1M AgBh frames
with three to nine visible rings, rotations within ±3° and a distance estimate within 15 % of the truth. Every
geometry returned by the line-position check (section 3.4) was recovered: direct beam within 5 px, distance within
5 %. For x_c, y_c, L, rot1 and rot2, the interval θ ± σ held the true value in 53–69 % of the fits with four or more
matched rings, near the 62–65 % a Student t with n − 1 degrees of freedom gives for so few rings. The 95th
percentile of the tilt error √[(rot1_fit − rot1_true)² + (rot2_fit − rot2_true)²] was 0.036°.
[The calibration benchmark](../tools/benchmarks/README.md#geometry-calibration-against-pyfai-and-the-coverage-of-its-1σ)
gives the conditions and the results per ring count.

Implementation: `src/gimap/features/calibration/domain/engine.py`, `src/gimap/features/calibration/domain/preprocessing.py`, `src/gimap/features/calibration/domain/center_estimator.py`, `src/gimap/features/calibration/domain/radial_profile.py`, `src/gimap/features/calibration/domain/peak_detector.py`, `src/gimap/features/calibration/domain/peak_matcher.py`, `src/gimap/features/calibration/domain/guided_rings.py`, `src/gimap/features/calibration/domain/candidate_ranker.py`, `src/gimap/features/calibration/domain/optimizer.py`

### 3.4 Quality measures

GIMaP uses two measures of a calibration.

For a guided match, the recorded observed radius is r₀ + medianₛ(δₛ r₀/rₛ). Here r₀ = L tan(2θ)/p̄
is the equivalent radius on a perpendicular detector, rₛ is the mean predicted radius of the rays in
supporting sector s, and δₛ is the measured radial peak offset sampled in integer pixel steps.
Thus the record gives one radius per matched order in units of the mean pixel size. The ring-point RMS uses
individual subpixel centroids on the detector.

**Ring residual (the Geometry Calibration window).** *Ring-point RMS* gives the RMS of (3.17) for the final geometry with fixed or fitted tilt. Its tooltip gives the starting geometry on the same points. After fitting tilt, it also gives the RMS for centre and distance refitted at zero tilt on those points. *Ring-radius RMS* is shown when no point fit is recorded. The ring-radius value `rms_residual_px` is the RMS of observed minus predicted equivalent radii for the final assigned orders. It enters the confidence rating. `distance_consistency` is the standard deviation divided by the mean of the per-ring distances. `azimuthal_coverage` is the mean coverage of the matched rings. The window shows a confidence:

For the confidence thresholds below, RMS is the larger of the ring-radius RMS and the available ring-point RMS.

| Confidence | Condition |
|---|---|
| High | at least 4 rings, RMS ≤ 2 px and coverage ≥ 0.18 |
| Medium | at least 2 rings and RMS ≤ 6 px; or at least 4 rings, centre support ≥ 0.8, RMS ≤ 8 px and coverage ≥ 0.05; or at least 5 rings, centre support ≥ 0.8, RMS ≤ 10 px and coverage ≥ 0.015 |
| Low | otherwise |

**Line positions (Run Automatic Analysis, `tools/gimap_agent.py`).** The ring residual depends on how many rings are fitted and on what the model cannot describe. The automatic analysis therefore also checks where the standard's lines land in q:

1. The calibration image is integrated with the candidate geometry, canonical centre and flat detector: `q = (4π/λ) sin(½ atan2(r, L))`, with r from the pixel centres; a tilted candidate is integrated with its rotations (GIMaP's pixel→q mapping, section 3.7). Bins are 0.0015 Å⁻¹ wide; bins with at least 20 pixels count.
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

When AgBh assignments ranked by guided ring evidence are compared, the accepted lines of the first-ranked AgBh
solution form a reference set if there are at least three. An alternative AgBh assignment must measure every line
of that set, and ε is compared on exactly those q values, so that another harmonic assignment cannot win by fitting
a different subset. For other standards, with fewer than three reference lines, or for older records, each
candidate's ε is taken on its own set of at least three accepted lines. The record names the rule in `chosen_by`.

"Good" means agreement with the calibration image. Check in the acquisition records that the calibration belongs to the sample data. Section 3.8 compares the GIMaP and pyFAI calibrations of the public AgBh frame.

Implementation: `src/gimap/features/calibration/domain/candidate_ranker.py`, `src/gimap/features/calibration/domain/line_check.py`, `src/gimap/features/calibration/application/headless.py`, `src/gimap/features/assistant/domain/calibration_quality.py`

### 3.5 From a calibration to an instrument profile

Apply in Geometry Calibration turns the selected candidate into an instrument profile:

- the beam centre from index to canonical coordinates (1.5), the distance from mm to m, λ from the energy, and the two pixel sizes;
- `α_i = 0` for a new profile; an updated profile keeps its α_i;
- GIMaP asks before an update moves the beam centre by more than 10 px or the distance by more than 5 %;
- the detector tilt: rot1, rot2, rot3 of the candidate become the rotations of the geometry (centre and distance stay those of the direct beam);
- the profile's `source` names the standard, the time, the image, the 1σ of centre and distance, the fitted tilt with its 1σ and where the energy came from.

Implementation: `src/gimap/features/calibration/domain/instrument_profile.py`, `src/gimap/features/calibration/domain/manual_refinement.py`

### 3.6 pyFAI `.poni` files

Run Automatic Analysis, `tools/gimap_agent.py` and Analyze (**Load .poni…**, and the `.poni` beside a frame, section 1.4) accept a pyFAI `.poni` file as the geometry. Its PONI is converted to the direct beam with the relations of pyFAI's `getFit2D` (the Fit2D geometry; Hammersley, 2016):

```text
x_c = (poni_2 − L_poni tan rot_1) / p_2                                     (3.10)
y_c = (poni_1 + L_poni tan rot_2 / cos rot_1) / p_1                          (3.11)
L = L_poni / (cos rot_1 cos rot_2)                                           (3.12)
tilt = arccos(cos rot_1 cos rot_2)                                      (3.13)
```

- `L_poni` is the normal distance in metres. Equations (3.10)–(3.11) give the beam centre in orientation 3. For a detector of width W and height H, orientation 1 uses `(W − x, H − y)`, 2 uses `(x, H − y)`, 3 uses `(x, y)` and 4 uses `(W − x, y)`. These transforms act on continuous corner coordinates. A reversed array index uses `N − 1 − i`. GIMaP applies the same signs to pixel displacements before the rotations. If an axis is reversed, its dimension must be available from the detector configuration, a known detector shape or the frame. Unsupported orientation values are rejected.
- The pixel sizes come from `PixelSize1/2`, from `detector_config`, or from a known detector name (Pilatus 172 µm, Eiger 75 µm, PerkinElmer 200 µm, Lambda 55 µm, MAR345 100 µm, Jungfrau 75 µm). The wavelength is converted from m to Å.
- The rotations go into the geometry with the direct beam and the distance (section 3.7); the instrument profile also keeps the file's distance, PONI, rotations and pixel sizes (`tilt` in `instrument_profiles.json`; `dropped_tilt` up to 0.2.0, still read). Above 0.5° the file's inspection notes the tilt.
- Analyze compares the stated detector dimensions and pixel sizes against the frame. **Load .poni…** reports a shape mismatch or a pixel size differing by more than 1 % from the header. Automatic loading skips incompatible files. A file without a wavelength needs one typed. Run Automatic Analysis requires matching ordered dimensions for flipped detectors. The signed ray coordinates of all four orientations agree with pyFAI 2026.9.0.
- **Writing.** Analyze and Geometry Calibration export PONI 2.1 with the pixel sizes, frame shape, detector orientation, normal distance, PONI coordinates, rotations and wavelength. Undo the orientation flip before applying the inverse of (3.10)–(3.12). The default orientation is 3.

```text
L_poni = L cos rot_1 cos rot_2,   poni_2 = x_c p_2 + L_poni tan rot_1,   poni_1 = y_c p_1 − L_poni tan rot_2 / cos rot_1      (3.16)
```

  Reading a written file gives the geometry back to rounding (the q of every pixel to 10⁻¹²). GIMaP stores ψ and available measurement details in `# GIMaP sample_roll_deg` and `# GIMaP sample_roll_measurement` comments. pyFAI’s detector angles keep their usual values. α_i is no part of a `.poni`.

Examples from the public data: the P08 file gives (545.46, 1826.95) px, `L` = 811.77 mm, tilt 0.31°; the pyFAI AgBh file gives (179.98, 263.86) px, `L` = 1583.31 mm, tilt 0.57° (note); the Xeuss file gives (7.05, 589.49) px, `L` = 147.58 mm, tilt 6.06° (note).

Implementation: `src/gimap/shared/geometry/poni.py`, `src/gimap/features/assistant/domain/calibration_files.py`, `src/gimap/features/analyze/application/geometry_files.py`

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

Equations (3.14)–(3.15) show orientation 3. For orientations 1, 2 and 4, apply the row and column signs of section 3.6.
X and Y are the displacements (1.3), (1.4) from the direct beam, ĉ the direction of the detector columns and r̂
that of its rows upward. Without rotations in orientation 3, (3.15) is `(L, X, Y)`. Every quantity of a pixel comes from this vector: q of
(2.8), 2θ and χ, the exit angle of a row, the solid angle `(L / |t|)³` and the polarisation factor (section 5).

- **Ray intersection.** Let `n = ĉ × r̂` and `b = (L, 0, 0)`. A laboratory ray `u` reaches the detector at `s u`, where `s = (n·b)/(n·u)`. A finite positive s is required. Project `s u − b` onto the two signed detector axes to obtain the pixel displacements. Mirror filling and XRR use this same intersection.
- A beam centre moved later (the GISAXS symmetry axis, a centre set by hand) moves the detector in its own plane;
  the rotations stay.
- `tools/benchmarks/pyfai_comparison.py` compares with pyFAI 2026.9 on the four public frames, tilts 0 to 6.06°:
  |q|, 2θ, and q∥ and q_z against pyFAI's fiber units `qip` and `qoop` (sample orientation 4) agree to 10⁻¹³. The
  solid angle agrees to 10⁻⁷ up to one constant (pyFAI normalises to the PONI, GIMaP to the direct beam), the
  polarisation factor exactly.
- pygid (0.2.20) places pixel i at i·p from the PONI; pyFAI and GIMaP place it at its centre, (i + ½)·p. With
  GIMaP's direct beam moved by half a pixel along both axes, q∥ and q_z of every pixel agree with pygid's to 10⁻¹⁵;
  without the shift they differ by up to 2.4 × 10⁻³ Å⁻¹ (`tools/benchmarks/pygid_comparison.py`).
- Leaving the tilt out would put |q| off by up to 0.20 % on the P08 file (tilt 0.31°) and 6.3 % on the Xeuss file
  (6.06°, at the corner farthest from the beam). Rot3 alone leaves |q| unchanged and turns χ by rot_3.
- The automatic analysis lists the tilt under *Decisions* (`detector tilt: 6.06° in the geometry`), and the
  Geometry step and its summary say `detector tilt 6.06°`. The record has the rotations in `geometry`.
- Profiles saved by GIMaP 0.1 kept the tilt beside a flat geometry (`dropped_tilt`); they are read with the tilt
  in the geometry. The Detector geometry dialog of Analyze has fields for rot1, rot2 and rot3 (degrees).
- **Checking rot3 with a GISAXS frame.** In **Geometry ▸ Check rot3**, pick two distant points on the observed rod. On the plane of incidence the exact condition is `q_y = k X′/|t| = 0`. The signed magnitude q∥ includes q_x and can be nonzero along that plane. For the two rays, the tool holds ψ fixed and solves `ΔX′ = 0` as rot3 varies, taking the solution nearest the current angle modulo 180°. Both endpoint q_y values test the beam centre as well. Enter the angle in the geometry dialog. A second pair on the horizon or the same Yoneda band tests Δα_f = 0 using (2.8).
- The initial ring search before tilt refinement still assumes a perpendicular detector. A horizontal cut defined by detector rows can also mix exit angles (section 6.7).

Implementation: `src/gimap/shared/geometry/detector_geometry.py` (`in_plane_axes`),
`src/gimap/shared/geometry/q_mapping.py` (`lab_rays`), `src/gimap/shared/geometry/instrument_profile.py`,
`src/gimap/shared/geometry/detector_tilt.py`

### 3.8 Detector tilt from the rings of the standard

A detector that is not perpendicular to the beam turns the rings into conic sections. After the centre-and-distance
fit (section 3.3, steps 1–7), the sector centroids determine the direct beam `(x_c, y_c)` and the distance `L`. When tilt refinement is enabled they also determine rot_1 and rot_2. Rot_3 stays at its supplied value (default 0°). The geometry is GIMaP's (1.3)–(1.4) with (3.15), the one
Analyze uses, so a calibrated geometry and the q of Analyze agree by construction.

**Ring points.** For each matched ring of q_m, the valid pixels within `h = 2 clip(0.012 r, 3, 10)` px (first pass)
or `clip(0.012 r, 3, 10)` px (later passes) of the ring of the current geometry are split into azimuthal sectors
about 2 px of arc long (36 to 720 per ring). Each pixel has weight w = max(0, I − I₂₅), where I is the measured
detector intensity and I₂₅ is the sector's lower quartile. A sector needs at least three valid pixels,
including one at an offset ≤ −h + 1 px and one at ≥ h − 1 px in the residual coordinate (3.17). These
conditions exclude profiles cut short by detector edges and gaps. Among sectors that span both flanks,
the peak weight must be positive and at least 0.3 of the median peak weight for the ring. Each qualifying
sector gives one subpixel centroid (Σ w x / Σ w, Σ w y / Σ w).

**Residual and fit.** With `t = (along, right, up)` the laboratory vector of a point (3.15) and
`tan 2θ_p = √(right² + up²) / along`,

```text
ρ_p = L ( tan 2θ_p − tan 2θ(q_m) ) / p̄                                            (3.17)
```

in pixels; for an untilted detector (3.17) is exactly (3.8). It is minimised with `scipy.optimize.least_squares`
(soft-L1 loss, f_scale 2 px, at most 200 evaluations); the centre may move by `clip(0.04 max(W, H), 12, 60)` px,
L by ±25 % within the distance range, rot_1 and rot_2 within ±30°. The points are taken again around the rings of
the new geometry and the fit repeated, four passes at most (fewer when the centre moves less than 0.01 px and the
rotations less than 10⁻⁶ rad). On the final set of points the record gives the RMS of the starting geometry (`initial_point_rms_px`) and the fitted geometry (`point_rms_px`). A tilted fit also reports `untilted_point_rms_px` after fitting centre and distance on these points with rot_1 = rot_2 = 0.

**When.** The point fit needs at least 30 accepted centroids on at least two distinct orders. With the
window's **Refine rot1, rot2** option enabled, tilt is fitted when three or more orders contribute centroids
after the sector checks. With two contributing orders, or with the option cleared, the centre and distance
are fitted at fixed tilt. An explicit engine request for `tilt="on"` allows two contributing orders.

In a calibration JSON, `metadata.ring_point_fits` stores the before and after values for each geometry and its matched lines. Headless summaries give both values directly on each solution.

**Uncertainty.** Section 3.3 gives the uncertainty for each parameter from the final set of centroids. For a fitted tilt, the record includes `rot1_sigma_deg` and `rot2_sigma_deg`.

Applying or exporting a manually changed geometry clears the point-fit RMS values and uncertainties. The retained ring-radius RMS describes the original automatic ring matches.

**Validation.** From an untilted start 3 % off in distance, ring points of the public Xeuss `.poni` geometry
(tilt 6.06°) give rot_1 and rot_2 within 0.012° of the file's values. On the public AgBh frame at λ = 1.0 Å, GIMaP
matches orders 1–5 with a ring-point RMS of 0.142 px. Its direct beam lies 0.193 px from that of the independent
pyFAI calibration, and the two distances to the direct beam are 1633.473 and 1634.053 mm.

Implementation: `src/gimap/features/calibration/domain/tilt_fit.py`, `src/gimap/features/calibration/domain/optimizer.py`
(`estimate_uncertainty`), `src/gimap/features/calibration/domain/engine.py`

## 4. Reading and preparing frames

### 4.1 Stored values and valid pixels

The readers convert each frame to single precision, without rescaling or offset. They record the stored data type in `metadata["stored_dtype"]` (TIFF, EDF, CBF and NeXus/HDF5; a frame without this entry is treated as integer counts). Each reader also returns a mask of pixels that carry no data:

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

**Orientation.** A file written by the Lambda server (it has a module position, a `layout` such as `1556X516`, or "lambda" in its description, type or local name) gets the P03 Lambda transform:

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

- **Mask order.** The editable user mask starts empty or with a loaded mask image. Edits are replayed in order for each frame. An image adds every non-zero pixel (the pyFAI convention). The shape must match the detector array, with row 0 at the top. An exclude operation adds a region and an include operation removes it. Detector validity and the automatic bad-pixel exclusions apply independently. [pyFAI mask convention](https://pyfai.readthedocs.io/en/stable/conventions.html)
- **Pixel shapes.** Test the centre of each pixel, `(j + 0.5, i + 0.5)`. Rectangle edges are included. Polygons use Matplotlib's point-in-path test. For a centre `(cx, cy)`, write `u = x − cx`, `v = y − cy`. A disk has `u² + v² ≤ R²` and an annulus has `r² ≤ u² + v² ≤ R²`. An axis-aligned ellipse with semiaxes `a, b` has `(u/a)² + (v/b)² ≤ 1`. A brush or eraser selects centres within the chosen radius of any segment of the stroke, including the endpoints. A single click gives a disk.
- **Beam-centred q annulus.** Select `q_min ≤ |q| ≤ q_max`, in Å⁻¹, where `|q| = (4π/λ) sin(2θ/2)`. The laboratory scattering angle 2θ is measured from the direct beam to the ray through the pixel centre. It includes detector tilt and both pixel pitches (section 2). The limits refer to vacuum scattering angles for any film refraction setting.
- **Value rules.** Write `I` for the finite raw value in Analyze’s single-precision frame (section 4.1), before subtracting dark or background frames. During frame summing, `I` is the sum (section 4.3). Above and below select `I > T` and `I < T`. An absolute threshold is in the frame's units. A relative threshold is `T = k × median(I)`, computed over detector-valid pixels remaining after automatic bad-pixel removal and preceding user edits. The saturation rule uses `I ≥ S` with an explicitly supplied saturation level. The negative rule uses `I < 0`.
- **Morphology.** For an integer radius n in pixels, use the disk of integer offsets satisfying `dx² + dy² ≤ n²`. Dilation masks a centre if any pixel in that disk was masked. Erosion keeps it masked if all pixels in the disk were masked. Pixels outside the frame count as unmasked. Inversion complements the user mask within the detector array. Clearing returns it to zero everywhere.
- **Record.** `drawn_mask_pixels` counts the final user mask. `corrections.mask_path` names an initial image if present, `corrections.mask_shapes` records each subsequent edit and its parameters, and `corrections.mask_sources` lists the source paths. Their file digests are in `input_sha256`. Exporting a mask image writes 0 or 1 for this frame's user mask.
- **Valid value range.** Optional limits `[I_min, I_max]`, both inclusive, applied to the pixel value at that step of the order (section 4.9), that is after background subtraction.

Implementation: `src/gimap/features/analyze/domain/mask_raster.py`, `src/gimap/features/analyze/domain/corrections.py`, `src/gimap/features/analyze/application/frame_preprocessing.py`

### 4.6 Background subtraction

Let `I` be the sample frame, `B` the background, `K` the dark and `s_B` the background scale. Subtract the dark from each exposure before scaling the background:

```text
J = (I − s_D K) − s_B (B − s′_D K)                                         (4.6)
```

The background must have the sample's detector shape and orientation. Each input has its own validity mask. Negative differences remain valid. With no dark, `J = I − s_B B`. Section 4.11 gives the source of the background dark scale.

### 4.7 Gap guard

Pixels next to module gaps and bad pixels often read wrong (edge pixels, charge sharing). The gap guard removes every pixel within N pixels of an invalid pixel:

```text
valid′(p) = valid(p) ∧ no pixel of the (2N + 1) × (2N + 1) square centred on p is invalid      (4.7)
```

- `0 ≤ N ≤ 20`. The default is 3 px in the Analyze window (Corrections; remembered as `analyze.gap_guard_px`). The domain object and settings files without the key default to 0.
- It grows around every pixel that is invalid at this point: reader-masked pixels and gap codes, pixels invalid in any summed frame, bad pixels, drawn and file masks, pixels invalid in the background frame, and pixels the flat field leaves out (section 4.10).
- It does not grow around pixels excluded later by the value range, nor around the horizon. The border of the frame does not count as invalid.

Implementation: `src/gimap/features/analyze/domain/corrections.py`

### 4.8 Mirror filling (GIWAXS)

Fill gaps from the mirror side (Mask step) is off by default and acts only on GIWAXS frames with a geometry. At zero roll on a flat detector perpendicular to the beam, the reflection `X → −X` maps `q_y → −q_y` and leaves `q_x`, `q_z`, |q|, α_f and 2θ unchanged; it changes the signs of q_∥ and χ. A pixel without data is given the value at its mirror position about the beam-centre column, in the same row:

```text
m_j = 2 x_c − (j + 0.5) − 0.5 = 2 x_c − j − 1        (fractional column index)       (4.8)
j_0 = ⌊m_j⌋,    w = m_j − j_0                                                          (4.9)
I_fill(i, j) = (1 − w) I(i, j_0) + w I(i, j_0 + 1)                                     (4.10)
```

- Only pixels that are invalid at this point are filled: gaps, masks, bad pixels, gap-guard and value-range pixels.
- For a rolled sample or tilted detector, reflect the outgoing ray in the sample plane of incidence. In the transverse coordinates of section 2.1 this changes the sign of X′ and preserves Y′. Intersect the reflected ray with the detector plane to find the opposite 2θ_f at the same α_f. Bilinear interpolation uses only originally valid pixels with nonzero weight. At ψ = 0 on a perpendicular detector in orientation 3 the calculation reduces to (4.8)–(4.10).
- A partner outside the measured image, or one with a missing interpolation neighbour, leaves the gap empty. The reciprocal-space missing wedge has no measured partner. The record counts filled detector pixels in `pixels_filled_from_mirror`.
- Mirror filling assumes symmetry in ±q_∥. It reuses measured intensities, so the error model switches to pixel scatter (section 6.4) and the fit input records `counting_model_valid: false`.

Implementation: `src/gimap/features/analyze/domain/preprocess.py`

### 4.9 The order of the steps

1. Load the frame or the sum of frames; validity (4.1).
2. Bad pixels found in the raw frame (section 4.4).
3. User mask from ordered image, shape and rule edits (section 4.5).
4. Dark frame (section 4.11), then background subtraction (section 4.6).
5. Flat field, first part: the pixels it does not keep are left out (section 4.10).
6. Gap guard (section 4.7).
7. Valid value range (section 4.5), on the values before the flat-field division.
8. Flat field, second part: the division `I / F_ff` (section 4.10).
9. Mirror filling, GIWAXS only (section 4.8), uses intensities after flat-field division. For a rotated detector with angle-dependent intensity corrections, step 10 precedes interpolation so each donor has its own correction.
10. Intensity corrections, GIWAXS only (chapter 5), the Lorentz factor among them (section 5.6).
11. Normalisation (section 4.12): one constant per frame, last.

Steps 4 to 9 run only when at least one of them is active. The raw frame is kept: changing any setting recomputes from the raw frame, so corrections never stack. Curves, q maps, the cake and every export use the result.

Implementation: `src/gimap/features/analyze/application/frame_preprocessing.py`, `src/gimap/features/analyze/application/use_cases.py`

### 4.10 Flat field (detector response)

Choose a uniform exposure `D_ff` from the same detector, either a detector image (first frame) or a NumPy `.npy` array. The flat field is off by default. For a raw flat, `r = t_exp,ff / t_exp,K` when both exposures are available, otherwise `r = s_D`. Selecting **Flat is already dark-subtracted**, or omitting the dark, sets `r = 0`.

```text
U_p = D_ff,p − r K_p,     m = mean(U_p over kept pixels)
F_ff,p = U_p / m                                                               (4.11)
kept(p) = valid(p) ∧ F_ff,p > h                                                (4.12)
I′_p = J_p / F_ff,p                                                            (4.13)
```

Start with valid positive responses. Remove pixels with `U_p ≤ h m` and recompute the mean until the kept set is stable. The flat-field threshold `h` defaults to 0.05 and accepts `0 ≤ h < 1`. The gap guard grows around excluded pixels. The value range applies before division. Mirror filling and GIWAXS intensity corrections follow it. The flat must match the detector shape. `corrections.flat_field` records its file, SHA-256, mean, threshold, dark scale and source, masked count and shape.

For Poisson or supplied σ errors, let `V_X` be the pixel variance of frame `X` in count units. First-order propagation treats the flat normalisation mean `m` as fixed:

```text
Var(F) = (V_Dff + r² V_K) / m²                         (F = F_ff,p)
Var(J/F) = Var(J)/F² + J² Var(F)/F⁴                    (independent J and F)
c_K = s_B s′_D − s_D,   J = I − s_B B + c_K K
∂(J/F)/∂K = c_K/F + Jr/(mF²)
Var(J/F) = (V_I + s_B² V_B)/F² + J² V_Dff/(m² F⁴)
           + (c_K/F + J r/(m F²))² V_K                 (shared K)
```

The same dark pixel changes the numerator by `c_K` and the flat response by `−r/m` per count. Adding the two contributions to the derivative before squaring retains the cross term `2 c_K J r V_K/(m F³)`. With automatic errors, subtracting a dark or background selects pixel scatter. With forced Poisson or supplied σ errors, raw reference variances use nonnegative counts as Poisson estimates. A reference supplied after dark subtraction holds only net counts, so its original dark and read noise are lost. A supplied σ frame provides `V_I`. Because `m` is held fixed, the propagation omits the interpixel covariance it causes and the correlations across frames that reuse a flat.

The flat exposure must retain its count or ADU scale. A unit-mean response map alone cannot supply its counting variance.

### 4.11 Dark frame

The dark `K` is a beam-off exposure of the same detector. Enter the sample dark scale `s_D`, usually the sample-to-dark exposure ratio. For a raw background, `s′_D` defaults to the background-to-dark exposure ratio when both headers provide it and to `s_D` otherwise. Entering a background dark scale overrides the ratio. **Background is already dark-subtracted** sets it to zero. Reduction requires valid pixels in the sample and the reference frames. Collecting the dark terms shows when they cancel:

```text
J = I − s_B B + (s_B s′_D − s_D) K                                           (4.14)
```

`corrections.reference_dark` records the background dark scale, its source and the already-subtracted flag. `dark_path`, `dark_frame` and `dark_scale` identify the sample's dark exposure. Automatic mode uses pixel scatter for a dark- or background-subtracted sample (section 6.4).

### 4.12 Normalisation: exposure, monitor, transmission, thickness, absolute scale

After every other correction (section 4.9) each pixel is multiplied by one constant `c` of the frame (Mask &
corrections ▸ Normalisation; every term off by default):

```text
I_norm,p = c · I′_p,      c = A_abs / (t_exp · M · T · d)                              (4.15)
```

| Term | Meaning | Source |
| --- | --- | --- |
| `t_exp` | exposure time (s) | the file header (`exposure_time_s` of the reader: CBF `Exposure_time`, EDF `ExposureTime` / `count_time`, NeXus `count_time` / `frame_time`) or typed |
| `M` | monitor (ion chamber, diode, `Flux` of a Pilatus CBF, ring current) | a header value chosen by its name (EDF keywords such as `Monitor`, `I0`, `Intensity0`; NeXus `NXmonitor` `data` / `integral` or `control/integral` of the frame) or typed |
| `T` | transmission of the sample, `0 < T ≤ 1` | typed |
| `d` | thickness entered in mm, converted to cm for a standard-derived scale | typed |
| `A_abs` | absolute-scale factor | typed or fitted to a standard measured in the same set-up |

- A term that is off is 1. The Normalisation section lists the header values of the frame shown (exposure time
  and monitors). A frame whose header lacks the chosen value is not analysed, and the message says why. For a sum
  of frames (section 4.3) `t_exp` and `M` are the sums over the frames summed.
- **What it changes.** Only the intensity scale; q, the shape of every curve, the cuts and the peak positions stay.
  The standard-derived scale gives cm⁻¹. Manual normalisation gives `counts/(pixel·s·monitor·mm) × A_abs` with the terms in use.
- **Uncertainty.** Normalisation scales the sample and reference variances together. Conditional on the scale,

```text
Var(I_norm,p) = c² · Var(I′_p)                                                (4.16)
σ_norm,k = c · σ_k                                                            (4.17)
```

  The same factor `c` scales point errors estimated from pixel scatter or a supplied σ frame.
- **Record.** `normalisation` of `<stem>_analysis.json` gives the factor `c`, the unit and every term with its
  value and source (`typed`, `header (exposure_time_s)`, `header (<name>)`); `header_values` lists what the header
  offered; `corrections.normalisation` keeps the settings. Each curve file carries a header line
  `intensity: <unit>; normalised I = A_abs*I/(t_exp*M*T*d), factor c (…terms…); sigma scaled by the same factor`.
- **Background.** GIMaP subtracts the background and dark frame in detector units before normalisation. Scale a background measured at a different exposure or transmission accordingly. Supply the measured transmission and thickness for each sample.

#### Absolute scale from a secondary standard

Measure the standard in transmission with the same beam, detector geometry and corrections used for the sample. Use a background-subtracted radial mean. For water, subtract the empty cell and select the flat small-angle region. The forward cross section of liquid water at 20 °C is 0.01632 cm⁻¹. It varies with temperature as the product of squared scattering length density, absolute temperature and isothermal compressibility. GIMaP linearly interpolates the Graz water table at 1 °C intervals from 0 to 100 °C at ambient pressure. The table gives a nominal value without an uncertainty. [Orthaber, Bergmann & Glatter (2000)](https://doi.org/10.1107/S0021889899015216). [Graz temperature table reproduced in BioXTAS RAW](https://sources.debian.org/src/bioxtasraw/2.3.0-2.1/bioxtasraw/RAWSettings.py/).

For glassy carbon, supply the reference for the material measured. Write independent, one-standard-deviation point uncertainties in its third column. Enter any common fractional scale uncertainty separately. For example, the published SRM 3600 calibration gives a 2.58% standard uncertainty (6.25% expanded uncertainty, coverage factor 2.423). When treating this as a common scale uncertainty, enter 2.58% once. Put any additional independent point uncertainties in the third column, or use zero if none are supplied. [Allen et al. (2017), §3.1.1](https://journals.iucr.org/j/issues/2017/02/00/vg5064/index.html).

Let `x_i = I_i/(t_exp M T d_cm)` be the corrected standard intensity and `u_i` its statistical uncertainty after the same division. Within the chosen overlap in q (Å⁻¹), the reference `R_i` is the water level or the linear interpolant of file values `r_j`. With interpolation coefficients `L_ij` and independent file uncertainties `s_j`,

```text
R_i = Σ_j L_ij r_j,        v_i² = Σ_j L_ij² s_j²
w_i = 1/(u_i² + b² v_i²),  χ²(b) = Σ_i w_i (x_i − b R_i)²,     A_abs = 1/b       (4.18)
```

The iterative least-squares fit estimates one positive slope through the origin. The weights in (4.18) use the diagonal elements of the covariance matrix. The plotted residual is `A_abs x_i − R_i` in cm⁻¹. Water has `v_i = 0`. Bins with zero combined variance are omitted when other bins have positive variance. If no point uncertainties are available, all weights are one. At least two distinct measured q values are required.

For the first-order uncertainty, define `z_i = R_i + b v_i² w_i (x_i − b R_i)`, `D = Σ_i w_i z_i²`, and `h_i = w_i z_i/D`. For `n` fitted bins and common fractional standard uncertainty `f`, the calculation is

```text
V_b = Σ_i (h_i u_i)² + b² Σ_j (s_j Σ_i h_i L_ij)² + max(0, χ²/(n−1) − 1)/D
σ_A_abs² = V_b/b⁴ + (A_abs f)²                                                       (4.19)
```

Accumulating the interpolated contributions at each original reference point accounts for shared reference values. Excess residual scatter increases the uncertainty. In the equal-weight case with no point uncertainties, `V_b = Σ_i (x_i − b R_i)²/[(n−1)D]`. Uncertainties in the entered normalisation terms are outside the fit. The JSON record identifies the standard and measurement, fit range, date and `A_abs ± σ_A_abs`. This calibration uncertainty is shared by all points and frames using that scale. Its contribution to a sample intensity is `|I_norm| σ_A_abs/A_abs`.

Implementation: `src/gimap/features/analyze/domain/absolute_scale.py`, `src/gimap/features/analyze/domain/normalisation.py`, `src/gimap/features/analyze/application/frame_preprocessing.py`, `src/gimap/features/analyze/application/export_record.py`.

## 5. Intensity corrections for GIWAXS

### 5.1 Definition

Frame-wide intensity corrections are off by default. Ring orientation has its own default polarisation correction (§7.7). When at least one is on, every pixel is divided by a factor:

```text
I′_p = I_p / F_p,      F_p = F_Ω · F_P · F_A · F_L                               (5.1)
```

- Each factor is optional; a factor that is off equals 1.
- q never changes. Peak positions therefore do not change; intensities compared across χ or q do.
- `F_Ω` and `F_P` are 1 at the direct beam and `F_A` is 1 at `α_f = α_i`, so corrected values stay close to counts per pixel. The Lorentz factor `F_L` (section 5.6) is not normalised.
- Pixels where F is not finite or not positive become invalid.
- With a flat field (section 4.10), `F_p = F_Ω · F_P · F_A · F_L · F_ff`: the flat field is divided earlier in the order (section 4.9), the factors here last, and the counts' variance (5.9) uses the whole product.
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

The code uses (5.2) for every geometry, tilted or not. L_n cancels, and F_Ω is 1 at the direct beam. pyFAI
normalises to the point of normal incidence; the two differ by one constant (section 3.7). For an untilted
detector (all rotations 0: perpendicular to the direct beam) `|t| = L_p` (2.2) and `L/L_p = cos 2θ`, so

```text
Ω_p = p_x p_y L / L_p³ = (p_x p_y / L²) cos³ 2θ,      F_Ω = cos³ 2θ      (untilted)      (5.3)
```

On a tilted detector `L/|t|` differs from cos 2θ, because the component of t along the beam changes across the
detector; there only (5.2) applies.

Implementation: `src/gimap/features/analyze/domain/intensity.py`

### 5.3 Polarisation

```text
F_P = ½ [ 1 + cos² 2θ − f_pol · cos 2φ · sin² 2θ ]                                  (5.4)
cos 2φ = (X² − Y²) / (X² + Y²)          (taken as 0 at the direct beam)          (5.5)
```

φ is the azimuth on the detector measured from the horizontal X axis. The formula and the sign of f_pol follow pyFAI. Typical values of f_pol are 0.95 to 0.99 for a horizontally polarised synchrotron beam, 0 for an unpolarised laboratory source and −1 for vertical polarisation. The window proposes `f_pol` = 0.99 when the box is ticked; the domain default is off. With `f_pol = 1`, `F_P = cos² 2θ` along the horizontal through the beam and 1 along the vertical; with `f_pol = 0`, `F_P = (1 + cos² 2θ)/2` everywhere. For the physics of the correction see Kahn et al. (1982).

Implementation: `src/gimap/features/analyze/domain/intensity.py`

### 5.4 Absorption in the film

For a homogeneous film with optical constants δ and β at wavelength λ, let s = sin α in vacuum and k = 2π/λ. At the vacuum–film interface the transmitted wave has

```text
v(α) = √(s² − 2δ + 2iβ),      Im v ≥ 0
k_z(α) = k v(α),              t(α) = 2s / (s + v)
g = 2 Im[k_z(α_i) + k_z(α_f)]
A(α_f,d) = |t_i|² |t_f|² ∫₀ᵈ exp(−gz) dz = |t_i|² |t_f|² [1 − exp(−gd)] / g
F_film = A(α_f,d) / A(α_i,d)
```

The reciprocal exit wave enters with the same decaying root as the incident wave. Below α_c ≈ √(2δ), its imaginary normal wavevector limits the depth contributing to diffraction even for very weak absorption. The model includes transmission at the top surface and omits reflection at the substrate. The depth integral tends to d for gd ≪ 1 and to 1/g for gd ≫ 1. Normalisation to specular exit cancels the incident transmission. The correction divides each pixel by F_film before binning, using its external exit angle (2.8). It requires α_i > 0 and excludes pixels at or below the horizon. Wavelength, depth and wavevectors in these formulas must use consistent length units.

**Large-angle limit (Table 2).** When both sin²α ≫ 2δ, 2β, the fields approach unit amplitude and Im k_z ≈ kβ/sin α. With μ = 4πβ/λ and attenuation length Λ_att = 1/μ this gives

```text
S(α_f) = 1 / sin α_i + 1 / sin α_f                                               (5.6)
A(α_f,d) ≈ (1 − exp(−μ d S)) / (μ S)                                             (5.7)
F_A = A(α_f,d) / A(α_i,d)                                                        (5.8)
```

When only Λ_att is supplied, GIMaP uses (5.6–5.8) directly. F_A tends to 1 in the thin limit and to S(α_i)/S(α_f) in the thick limit. Film thickness is entered in nm and the tabulated attenuation length in µm. [Gasser et al., §§2.5–2.6 and Table 2](https://journals.iucr.org/j/issues/2025/01/00/xx5064/)

Implementation: `src/gimap/features/analyze/domain/refracted_film.py`, `src/gimap/features/analyze/domain/intensity.py`

### 5.5 Uncertainty of corrected photon counts

For observed Poisson errors (section 6.4), let `I_p` be the sample count before reference subtraction and `I′_p = I_p / F_p` its contribution after correction. The sample shot variance per pixel and its contribution to the error of a bin mean are

```text
Var(I′_p) = Var(I_p) / F_p² = I_p / F_p² = I′_p / F_p                           (5.9)
σ_k = √max( Σ_{p∈k} I′_p / F_p ,  f̄_k² ) / n_k,      f̄_k = (1/n_k) Σ_{p∈k} 1/F_p      (5.10)
```

The sums run over the n_k pixels of bin k. The factor F_p includes the flat response F_ff when used (section 4.10).
With a flat field alone, a GISAXS detector line uses (5.10) with F_p = F_ff. The one-count floor f̄_k² gives a
bin of zero counts the error of one count at its mean scale. Setting every F_p to one gives (6.5).

For measured dark, background or flat frames, section 4.10 gives the reference variance, including the covariance
from a shared dark. Divide that variance by the square of the remaining GIWAXS factors before adding it to the
sample contribution. For a Lorentz-weighted radial sum, multiply the standard error of the mean by n_k (section 5.6).
Section 6.4 gives the corresponding errors for the smoothed Poisson, pixel-scatter and supplied σ options.

The fit input records `counting_model_valid: false` whenever a GIWAXS correction or flat field is on.
The JSON record `<stem>_analysis.json` holds `intensity_corrections` with the formulas and parameters.
The factor is computed in blocks of 512 rows and kept while shape, geometry and options stay the same.

Implementation: `src/gimap/features/analyze/domain/intensity.py`, `src/gimap/features/analyze/domain/binning.py`, `src/gimap/features/analyze/application/export_record.py`

### 5.6 Lorentz factor of a fibre texture

**Lorentz factor for peak areas** applies the fibre-texture Jacobian of Jiang (2015, Eq. 5) to detector intensities from crystalline reflections. The radial curve sums the weighted pixels in each q bin for peak-area integration. The default radial curve averages its pixels. The switch is off by default.

~~~text
1/L = cos α_i cos α_f |sin 2θ_f| = cos α_i |q_y|/k                              (5.11)
sin χ = q_xy/q,   q_xy = √(q_x² + q_y²)                                        (5.12)
~~~

Equation (5.11) converts between detector solid angle and the fibre-averaged (q_xy, q_z) plane. Its zero extends across the plane of incidence. On corrected maps and sums, GIMaP excludes |q_y|/k below the selected half-width in horizontal pixels times pixel pitch / detector distance (default 2 px). The correction and its mask are recorded. The variance of a weighted sum is the sum of its pixel variances (5.9–5.10). A peak area fitted against q integrates the bin sums over q. On a uniform grid, the fitted area divided by Δq gives the corresponding total weighted counts under the peak.

A ring bin averages the intensities of its pixels. For a fibre texture this mean samples pole density per steradian, so crystallites per unit polar angle have density proportional to (I − b) sin χ. Applying (5.11) before this average would introduce the angular factor |q_y|/q_xy. At zero incidence on a kinematic ring it equals √(1 − sin²θ/sin²χ), reaching zero at the missing-wedge edge χ = θ. At finite incidence the detector geometry gives q_x and q_y directly. The ring profile, its background, and its error bars are taken before this weighting or its mask (§7.7).

Implementation: `src/gimap/features/analyze/domain/intensity.py`, `src/gimap/features/assistant/domain/orientation.py`

## 6. Reduction to curves and uncertainties

### 6.1 Principles

- Each point normally gives the mean of its valid pixels. The GIWAXS Lorentz-weighted radial curve gives the sum of corrected pixel intensities in each q bin (§5.6). For independent pixels its variance is the sum of their corrected variances. For a mean both the intensity and standard error are divided by the number of contributing pixels. The intensity unit of a mean is counts per pixel (`I (counts/pixel)`) before normalisation.
- **No pixel splitting.** Each pixel goes whole to the one bin or detector line that holds its centre's coordinate; pyFAI's split-pixel integrators distribute it instead.
- **Checked against pyFAI.** On the four public frames, GIMaP's I(q) and pyFAI's `integrate1d` (2026.9, histogram, no splitting) agree to 10⁻⁷ on every bin that holds the same pixels in both. With the solid angle and polarisation, pyFAI divides the sum of the counts by the sum of the factors (`Σ I / Σ F`) and GIMaP takes the mean of `I / F`. The two differ where F and I both vary across a bin, by a median of 0.26 % on the Xeuss frame, close to the sample (`tools/benchmarks/pyfai_comparison.py`).
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

### 6.4 Curve error models

Automatic mode chooses Poisson errors for integer storage and scatter for floating-point, dark-subtracted or background-subtracted data. Integer storage alone gives no detector noise calibration. A CCD or flat panel can store offset ADU, and an Eiger can store counts after detector corrections. Choose the detector model under **Detector and curve errors**. The instrument profile stores gain `g` (counts per ADU), offset `o` (ADU), model and optional σ frame. The JSON record gives both the request and the model used. Mirror filling uses scatter because it duplicates measurements.

**(a) Poisson.** For raw sample and reference frames use `C = g (raw − o)`, with `g = 1` and `o = 0` by default. For a sum, subtract one offset per exposure. Already dark-subtracted references use `g raw`. Negative corrected intensities remain valid. The shot variance estimate in counts is `max(C, 0)`. For an unscaled bin of `n` pixels with total counts `S` the default error of its mean is

```text
σ² = max(S, 1) / n²       (radial or azimuthal bins)                             (6.5)
σ² = max(S, 1) / n²       (detector rows or columns)                             (6.6)
```

At about 1–10 counts per point an observed downward fluctuation also shrinks its own error, giving it greater weight in a fit. **Smoothed intensity** estimates the shot variance from up to two occupied bins on each side, excluding the target and stopping at gaps. It uses the observed estimate for an isolated bin. With corrections, write `S_k` for the sum of sample shot variances and `W_k` for the sum of one-count scale factors. The smoothed prediction is `W_k Σ_neighbors S / Σ_neighbors W`. Floor the shot variance at `(W_k/n_k)²`, add the summed reference variance (4.10), then divide by `n_k²`. This changes only the exported σ; Fitting keeps its own weighting. Near a sharp feature the neighbours can have a different mean.

**(b) Scatter of the pixels.** For corrected pixel values `X` in bin `k`,

```text
s_k² = [Σ X² − (Σ X)² / n_k] / (n_k − 1)                                      (6.7)
σ_k² = s_k² / n_k                                                              (6.8)
```

For `n_k < 10`, the default estimates pixel variance from a line `v_k = a + b I_k`. The fit uses the unbiased variances of bins with at least two pixels on this curve, weighted by `n−1` and downweighted for large residuals using Huber weights. With sparsely populated bins, groups of nearby intensity are first averaged by degrees of freedom (target up to 24 per group). Evaluation is restricted to the fitted intensity range. For one distinct intensity the fit gives a constant.

If the line predicts `v_j ≤ 0` for bins with `n_j ≥ 2`, their measured scatter sets a floor `V = Σ (n_j−1)s_j² / Σ (n_j−1)`, summed over those bins. Otherwise `V = 0`. The variance of a sparse bin's mean is `σ_k² = max(v_k, V) / n_k`. This retains measured χ bins whose scatter is positive where the line crosses zero. With pooling off, a singleton takes the median measured variance. If every bin is a singleton the scatter is unmeasurable and the reported σ is zero. Texture and gradients within a bin contribute to scatter.

**(c) Propagated errors.** Scaled Poisson uses sections 4.10 and 5.5. **Given σ frame** takes an independent standard deviation in sample input units for each pixel (for a sum, supply σ of the sum). Convert it by `g`, propagate through all corrections, add reference variance and use `σ_k² = Σ Var(X_p) / n_k²` without a one-count floor. Charge sharing, correlations between bins and covariance between frames using the same references are outside these point uncertainties.

### 6.5 Merging sparse bins along q

Near the beam, at the edge of a sector, next to a gap or in the missing wedge, a bin can contain few pixels. Analyze joins sparse bins along x in GIWAXS radial, sector, q-box and region profiles, transmission radial profiles, and user-defined GISAXS sector and line cuts. I(χ), native GISAXS detector-line profiles and peak-fit windows retain their original sampling.

- Threshold: `n_min = max(8, round(0.1 · median n))`. A curve whose bins all reach `n_min` is returned unchanged.
- The filled bins are walked in order of x and added to a group. A group closes when its pixel count reaches `n_min`.
- A step to the next filled bin of more than 2.5 median bin steps also closes the group, even when it is still sparse: a gap stays a gap.
- A sparse group at the end joins the previous point if it lies within 2.5 steps of it.

A merged point uses the pixel counts as weights for its coordinate and intensity. For independent constituent means with standard errors `σ_k`, propagation gives

```text
x = Σ n_k x_k / Σ n_k,      I = Σ n_k I_k / Σ n_k,      n = Σ n_k                   (6.9)
σ = √(Σ n_k² σ_k²) / n                                                              (6.10)
```

Observed Poisson errors use (6.10), including the one-count floor in each constituent bin. For scatter errors, (6.10) carries the within-bin uncertainty to the merged point. With pooling enabled, final points with fewer than ten pixels instead receive the fitted pixel variance from §6.4(b), divided by their pixel count. This fit uses the pixel moments of the final points. For smoothed Poisson errors, each group accumulates sample and reference variances and one-count scales. The neighbouring final points supply the smoothed estimate in §6.4(a).

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
| I(q) | valid pixels, averaged by default or summed with Lorentz weighting | q, N bins over `[q_lo, q_hi]` | means only |
| In-plane sector I(q) | `abs(χ) ≥ 90° − w_ip` | as I(q) | yes |
| Out-of-plane sector I(q) | `abs(χ) ≤ w_oop` | as I(q) | yes |
| I(χ) | ring q window, measured pixels above the orientation exit threshold, before Lorentz weighting | χ, 180 bins of 1° over [−90°, 90°] | no |
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

**Yoneda band.** The search window runs from the row of `α_f = 1.5°` down to the row of `α_f = −0.2°` (equation (2.14)). The 0.2° below the computed horizon absorbs small errors of α_i or of the beam centre. The columns are two bands of 100 px on both sides of a 31-column strip around the beam column `c_0 = ⌊x_c⌋`, namely `[c_0 − 115, c_0 − 15)` and `[c_0 + 16, c_0 + 116)`. This keeps the direct beam, the specular rod and a rod-shaped beam stop out. The profile is the median of the valid pixels in each row, smoothed with a 3-row moving average that ignores missing rows. The brightest row gives `r_Y = row + 0.5`, and `α_f` follows from (2.14). The window needs at least 3 rows. The band is the one described by Yoneda (1963).

With nonzero sample roll the Yoneda search follows constant exit angle across the side bands. The horizontal and vertical cut widths define q_z limits on the beam column and q_y limits at the horizon, respectively. The mean and error use the original measured pixels inside these limits. The detector displays the projected bands. The row and column rules below describe ψ = 0, where a horizon-height change of more than half a row across the detector triggers a warning.

**Horizontal cut I(q_y).** The centre `r_c` is the Yoneda row or the row set by hand. If the search finds no band, move 10 rows from the horizon on the beam column towards positive exit angles and clamp to the frame. If that column has no unique horizon crossing, use the middle row and warn.

```text
rows i = ⌊r_c − h⌋ … ⌈r_c + h⌉ − 1,       h = 2.5 px by default                    (6.12)
```

These are all rows whose area overlaps `[r_c − h, r_c + h]`: 5 rows when `r_c` is a pixel centre, 6 otherwise. Every valid pixel of these rows is used, including pixels below the horizon if the band is moved there. One point per column (6.4), `x = mean q_y`, signed.

**Vertical cut I(q_z).**

```text
columns j = ⌊c − w⌋ … ⌈c + w⌉ − 1,        w = 5 px by default                       (6.13)
rows i = 0 … ⌊row_horizon⌋ − 1    (perpendicular detector, orientation 3, vacuum)    (6.14)
```

The default band clears the full visible q_y = 0 rod by 15 columns at its inner edge, using the exclusion already used for the Yoneda search. It tries the right side, then the left, and warns if only a smaller clearance fits. A user-specified centre c is kept. For a tilted plane, use pixel-centre exit angles (section 2.3). Each occupied row gives one point, `x = mean q_z`. Add ▸ Vertical Band uses the nominal q step `k p_x/L` for its initial clearance (15 steps) and width (10 steps).

The nominal band sizes are therefore "5 px high" and "10 px wide". The error model is that of section 6.4.

**Where the automatic bands lie in q.** Their curves' records (the `# region:` line of the CSV and `_analysis.json`)
also give `qz_edges` and `alpha_f_edges_deg` (the q_z and α_f of the horizontal band's row edges `⌊r_c − h⌋` and
`⌈r_c + h⌉` on the beam-centre column) and `qy_edges` and `two_theta_f_edges_deg` (the q_y and 2θ_f of the vertical
band's column edges where each meets the horizon). The band limits are pixel edges, (6.12)–(6.14).

**Your own cuts, defined in q.** The Cuts step holds any number of named cuts besides the two automatic ones.
Each is defined in the `q_y`–`q_z` plane (Å⁻¹, `q_y` signed), so it means the same on every frame of the geometry and
is drawn alike on the detector (mapped back to pixels) and on the q map. A pixel belongs to a cut when the q of its
centre lies inside it — whole pixels, never split — and only usable pixels count (valid and above the horizon, as in
6.8):

```text
horizontal band   z_0 ≤ q_z,p ≤ z_1                  one point per column, x = mean q_y   (6.4)
vertical band     y_0 ≤ q_y,p ≤ y_1                  one point per row, x = mean q_z      (6.4)
box               both, one point per column (along q_y) or per row (along q_z)
sector            c_0 ≤ atan2(q_y,p, q_z,p) ≤ c_1,   r_0 ≤ q_yz,p = √(q_y,p² + q_z,p²) ≤ r_1        (6.14a)
line              0 ≤ s_p ≤ L,  |t_p| ≤ w,   s_p = (q_p − q_A)·û,  t_p = (q_p − q_A)·n̂                (6.14b)
```

χ of a sector is measured from the `q_z` axis (0°, the surface normal) with the sign of `q_y`; its origin is the direct
beam. A line runs from `q_A` to `q_B` (`û` along it, `n̂` across, `L = |q_B − q_A|`, half width `w`). Sector and line
curves are fixed-edge bins (6.2) of width `Δq = k p/D` (`p` the smaller pixel size: the q of one pixel near the direct
beam) over `[r_0, r_1]` (a sector without a q range: over the range of its pixels) or `[0, L]`, sparse bins joined as
in 6.5; their x is `q_yz` or the distance `s` along the line. The errors follow the frame's model (6.4), as for every
curve. A horizontal band at the q_z of the Yoneda row does not reproduce the automatic cut exactly: a row of constant
q_z curves a little across a flat detector, so off the beam column it takes slightly different pixels.

**Typing a cut in angles or pixels.** Each axis converts on its own, exactly and invertibly:

```text
q_z = k (sin α_f + sin α_i)       (with refraction: k (√(sin² α_f − sin² α_c) + √(sin² α_i − sin² α_c)))   (6.14c)
q_y = k sin 2θ_f                  (2θ_f on the sample horizon, α_f = 0)
row y of q_z:  q_z(x_c, y) = q_z on the beam-centre column;   column x of q_y:  q_y(x, y_h(x)) = q_y along the horizon   (6.14d)
```

The rows and columns are found by mapping q back to the detector (the exact model of section 2, tilted detectors
included). A line's half width typed in pixels is `w/(k p/D)`, in degrees `w/k` (small angles). The values shown in
the two other units are these conversions; the cut is the q region.

**Cuts on another set-up.** The cuts are kept with the set-up they were made on: detector name, frame size, pixel
size, distance (±0.5 mm), wavelength (relative 10⁻⁴), beam centre (±1 px) and detector tilt (±0.01°). A frame of
another file that differs in any of these asks whether to keep or remove the cuts; until then they are applied (they
are in q). α_i is not compared: in q the cuts mean the same at another α_i.

**Make Left and Right Symmetric** (Cuts step) moves `x_c` onto the mirror axis of the horizontal band and changes nothing else. The Cuts step and Run Automatic Analysis use the same method; section 8.2 gives it.

**GISAXS q map.** A grid of at most 600 × 600 cells in `q_y`–`q_z` over the range of the usable pixels (section 6.8).

Implementation: `src/gimap/features/analyze/domain/gisaxs.py`, `src/gimap/features/analyze/domain/gisaxs_cuts.py`, `src/gimap/features/analyze/application/gisaxs_cuts.py`, `src/gimap/shared/geometry/q_inverse.py`, `src/gimap/features/analyze/domain/binning.py`, `src/gimap/features/analyze/domain/symmetry.py`, `src/gimap/features/analyze/domain/symmetry_coarse.py`

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
- A q map is never finer than the detector samples q. Along each axis the cell is at least the step of that coordinate between a usable pixel and its right and lower neighbours (their root sum of squares; the 95th percentile over the usable pixels), so `N = min(N_max, ⌊range / step⌋)`, at least 32. A 619 × 487 frame then gets about 490 × 580 cells.
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

- The |q| points of the mean are those of the `q_y > 0` half, in the prepared cut of Run Automatic Analysis and its model fit, in the Series map and in Fitting, so one cut gives one mean curve. A half with fewer than two points never sets the grid.
- Between two independent bins on the other half, at fractional distance `t`, the interpolated intensity is `(1−t) I₁ + t I₂` and its variance is `σ̃² = (1−t)² σ₁² + t² σ₂²`. Equation (6.17) gives the marginal standard error of the average.
- Points are averaged only where the other half has measured points on both sides without a hole, or a point at the same |q| (within 1 % of a step). A hole is an interval wider than 2.5 times the local step, the median of the 9 surrounding intervals.
- Elsewhere the measured half stands alone with its own σ. Points of the other half that lie in a hole of this half, or beyond its ends, are added as measured. Nothing is interpolated across a hole.

An endpoint used in two averaged points correlates their errors. Let `Wᵢₖ` be the weight of original bin k in output i, including any merging. Then `Cᵢⱼ = Σₖ Wᵢₖ Wⱼₖ σₖ²`. Two adjacent interpolated midpoints each use one quarter of their shared bin after averaging the halves, giving a covariance of `σ_shared²/16`. The pointwise error is `√Cᵢᵢ`. For fitting with diagonal weights GIMaP uses `fit_sigmaᵢ = √(Σⱼ Cᵢⱼ)`. This accounts for all reused bins and gives a conservative variance for any linear combination because each off-diagonal covariance is nonnegative.

**Automatic choice** (Run Automatic Analysis and its model fit). Only points with finite values, `I > 0` and at least one pixel are used. For each half GIMaP takes:

- the **reach**: its largest |q_y|;
- the **coverage**: the share of bins of width 1.5 × the median |q_y| step that hold at least one point. The bins run from the larger of the two halves' fourth |q_y| values to the half's reach, so the three points next to `q_y = 0` (specular rod, beam stop) are skipped.

The decision, in this order:

1. Only one half is on the detector: that half is fitted.
2. A half is not used when its reach is below 0.3 times the other's, or when its coverage is below 0.6 and more than 0.1 below the other's. The other half is fitted.
3. Otherwise both halves are interpolated linearly onto the |q_y| of the `q_y > 0` half within the common range (from the skipped points to the shorter reach; at least 5 points are needed). The mismatch is `m = median |ln(I_−/I_+)|`. When `m > 0.15` (about 16 %) the halves still disagree after the centre correction (a shadow, an absorber or real in-plane anisotropy): both are kept for display, and the better-covered half is fitted.
4. Otherwise the mean (6.16) is fitted.

**Prepared curve and files.** In the prepared cut a point without a usable σ gets the median relative σ of the other points. A cut without any σ gets 10 % of I in automatic analysis. `<stem>_fit_input.dat` holds both halves with signed q in Å⁻¹. Its record names the halves to fit. Fitting works in nm⁻¹ internally (1 Å⁻¹ = 10 nm⁻¹). Above 1000 points, consecutive groups of `g = ⌈n/1000⌉` points are merged. Intensity uses inverse marginal variance weights, q uses the arithmetic mean, and the weights enter W above. With independent bins the merged error reduces to `(Σ σ⁻²)^(−1/2)`. Without σ both means use equal weights. The last incomplete group is dropped. For an averaged cut the automatic fit table gives the pointwise error in `sigma` and the error used for weighting in `fit_sigma`.

Implementation: `src/gimap/shared/signed_cut.py`, `src/gimap/features/analyze/application/export_record.py`

### 6.10 Reference-peak corrections of a series

The Series map, its traces and the peak followed in a window are described in sections 11.1 and 11.2. This section gives the two corrections that Batch Export can apply to each frame of a GIWAXS series (In-situ corrections; both off by default). A peak that does not change during the run, such as a substrate reflection or an internal standard, is located as the highest I(q) point within `q_ref ± Δ` (defaults 2.132 and 0.035 Å⁻¹). Its position is therefore limited by the bin width.

```text
L′ = L · q_obs / q_ref           (repeated up to 3 times, until |ΔL|/L < 10⁻⁶)        (6.18)
g = I_target / I_peak            (I and σ multiplied by g; q unchanged)             (6.19)
```

Equation (6.18) is a small-angle scaling. Re-reducing the frame with L′ and repeating makes it converge at wide angles too. The target `I_target` is 1 by default. The factor g is taken from the first frame for the whole series, or from each frame.

Implementation: `src/gimap/features/analyze/domain/series.py`, `src/gimap/features/analyze/application/series.py`

### 6.11 SAXS/WAXS in transmission

A frame taken in transmission (a calibrant, or a sample in transmission) has no sample surface: no horizon, no
Yoneda band, no missing wedge. In the mode **SAXS** (SAXS/WAXS in transmission; chosen by hand, or by Auto, section 2.4) every valid pixel
is used:

- q = 4π sin θ / λ of every pixel, and q_y (to the right), q_z (up) from (2.8) with α_i = 0, which are the
  components of q across the beam;
- χ = atan2(q_z, q_y): 0° to the right of the beam, 90° up, ±180° to the left (the detector azimuth); it is not the
  GIWAXS χ of section 6.6;
- `radial`: I(q) over the whole ring, with the bins and errors of section 6.6;
- `azimuthal`: I(χ) from −180° to 180° in 1° bins, in the q window of the strongest ring (or the one set);
- the cut regions (rings, sectors, spots) in this χ;
- the q map in q_y–q_z and the cake over χ from −180° to 180° (0.5° rows).

Mirror filling and the GIWAXS intensity corrections are not applied. A pinned GISAXS or GIWAXS mode on such a frame
is pointed out like any mode that disagrees with Auto. On the public silver behenate frame (five complete rings)
Auto takes this mode; I(χ) covers −180° to 180°, where GIWAXS kept only the half above the horizon.

Implementation: `src/gimap/features/analyze/domain/transmission.py`, `src/gimap/features/analyze/application/technique.py`

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

**Curves.** The analysis uses four curves of the GIWAXS reduction (section 6.6): the radial I(q), the in-plane and out-of-plane sectors (half widths 10° by default), and I(χ) in 1° bins over the ring window of section 7.7. All use the valid pixels above the sample horizon (section 2.3), with the binning, the merging of sparse bins and the standard errors of sections 6.2 to 6.6. χ is that of (2.13): 0° along the surface normal, ±90° in the sample plane, with the sign of `q_y`. Orientation uses the exit-angle selection and default polarisation in §7.7. The other curves use the configured frame corrections (chapter 5).

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

Here y is the unsmoothed curve and σ its bin error. The shift (7.6) is applied when at least `max(5, ⌊N/5⌋)` bins
satisfy the condition. The baseline serves to find candidates (section 7.3), to start each peak fit and to compare the
sectors (section 7.6). Reported peak heights refer to the local linear background of the profile fit (section 7.4).
The ring orientation fits its own background (section 7.7).

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

**The cap.** The limit of 12 keeps a run short: candidates that fail or merge take most of the time. When more
candidates qualify, the report says so. The *Decisions* line *peak search* gives "N candidate peaks
found; the 12 most prominent were fitted", the peak search record gives `candidates`, `max_peaks` and `truncated`, and
the title of the series hints says that they come from the fitted peaks only. The peak table, the sector comparison,
the rings analysed and the series hints all use the fitted peaks. Fitting merges or rejects many candidates, so
fitting all of them gives fewer peaks than candidates. `--max-peaks N` (command line) or `max_peaks` (the tools
`find_peaks` and `run_standard_pipeline`) fits more; Run Automatic Analysis in the window always uses 12. On the
public P08 frame 115 candidates qualify; fitting all of them gives 32 peaks in about 8 s, against 12 peaks in under 1 s.

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

**Fit.** Weighted least squares (`scipy.optimize.curve_fit`, bounded trust region, at most 5000 evaluations) with the
bin errors as absolute σ: Poisson counting errors for photon counts as measured, otherwise the pixel scatter of each
bin (section 6.4). The report's *Decisions* line *uncertainties* names which. The fit is rejected when q_0 lies on the window edge, when FWHM reaches its upper bound,
or when h_pk ≤ 0. If the fitted centre leaves the candidate's half-maximum span (a shoulder or neighbour widened the
candidate), the peak is fitted again with twice the width of its cleaner flank; the refit is kept when it stays
inside the span and is significant.

**Goodness of fit and errors.** With n points in the window and C the covariance matrix returned for absolute σ:

```text
χ²_red = Σ_i ((y_i − I(q_i)) / σ_i)² / (n − 5)                                                 (7.11)
σ_p = √(C_pp) · max(1, √χ²_red)                                                                (7.12)
```

The errors are inflated when `χ²_red > 1` and never deflated. After the fit, a peak is kept if

```text
SNR = h_pk / σ(h_pk) ≥ 3   and   h_pk ≥ 0.02 abs(b_0)                                                  (7.13)
```

A candidate whose profile cannot be fitted is kept when `SNR_loc ≥ 6`, with the flag `fit_failed` and the width from
step 1 of section 7.3.

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
report with the reason but is not used for ring selection, sizes or series hints. The same holds for a peak flagged
at_edge, since the end of the searched range may cut off its shape, position and width; the report lists it as "at the
end of the data: check". The peaks left, without a caveat and not at the edge, are the *reliable* peaks.

**Series hints.** From each of the three lowest-q reliable peaks, GIMaP looks for the orders of a lamellar
series (1 : 2 : 3 : 4) and of a hexagonal series (1 : √3 : 2 : √7 : 3), each order matched within 2 % of its
expected q. A series is reported when at least three orders (the first included) match. The orders of a lamellar
series must follow one another: the search ends at the first missing order. A hexagonal series must include √3 or √7, the orders a
lamellar series does not have. This is a hint; no phase is assigned.

Implementation: `src/gimap/features/assistant/domain/peaks.py`

### 7.5 Which rings are analysed

Among the reliable peaks (no caveat, not at_edge), the analysis takes the **three of largest area A** (7.16); the
number is an option (`--rings N`). A ring the user names (`ring_q`) is added when no chosen peak lies within half its
FWHM. Each chosen ring gets an orientation analysis (section 7.7). A Scherrer size (section 7.8) is computed when a
reliable peak lies within half a FWHM of the ring. The rings left out, and why, are listed in the *Decisions*.

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

This is a coarse comparison of two 20°-wide sectors; sections 7.7 and 7.11 give the orientation distribution.

Implementation: `src/gimap/features/assistant/domain/orientation.py`

### 7.7 Ring orientation and the Hermans orientation factor

With a phase assignment, the joint crystal orientation fit of §7.11 uses the same radial windows, detector errors and folding. When the CIF supplies film optical constants, that fit models refraction and attenuation of both film waves and uses measured pixels down to the horizon of the active q map; an exit-angle or polar-angle limit set by the user still applies. The single-ring Hermans factor and pole figure use the pixel selection described below.

**Ring window.** The ring is centred on the fitted q_0 of the nearest radial peak (when one lies within
`max(2 FWHM, 3 Δq)` of the requested q), with the half width

```text
h_χ = max( 0.75 FWHM, 2 Δq ),   limited to 0.04 q_0 for a broad peak                          (7.21)
```

Without a fitted peak the window is ±1.5 % of the requested q. The pixels with q in this window give I(χ) in 1° bins.

**Pixel selection and background.** The orientation profile uses measured pixels, with polarisation correction on by default (factor 0.99 when no frame value is selected, §5.3) and the selected solid-angle, absorption and flat-field corrections. Pixels with α_f below the selected minimum are removed before binning. The automatic minimum is twice the film critical angle when supplied, otherwise twice the incidence angle. A Gaussian on a linear baseline is fitted to the radial pixel means, in the same intensity units, over three ring-window widths. The baseline at the window centre is the background b, constant in χ. Without a peak fit, the adjacent radial bins give b.

Each measured I(χ) bin has the σ of the detector error model of §6.4. Mirror filling of the radial curve leaves the
ring pixels and their errors unchanged. The saved ring records the model in `orientation_settings.error_model`.

**Folding.** The two halves ±χ are combined into 90 bins of `abs(χ)`, bin k covering [k°, k+1°), with centre
χ_k = k + 0.5°, by inverse-variance weighting:

```text
Ī_k = Σ_j w_j (I_j − b) / Σ_j w_j,   w_j = 1/σ_j²                                            (7.22)
σ_k = 1 / √( Σ_j w_j )                                                                       (7.23)
```

Bins without pixels are unmeasured. When `b > 0`, bins whose intensity is below 0.2 b (a shadow), and their
neighbours, count as unmeasured too.

At the high-|χ| edge set by the exit-angle cut, GIMaP also excludes a contiguous run of weak bins.
Starting in the last measured bin at that edge, it moves inward while Ī_k ≤ 0.05 M, where M is the
median net intensity of measured, unshadowed folded bins. A gap or a bin above the threshold ends the
run. The test requires M > 0. An isolated weak bin inside the ring stays in the sum. The report gives
the edges of the excluded 1° bins. The pole fit uses the same selection.

**Requirements.** No factor is computed when fewer than 10 χ bins are measured, when no folded bin reaches
`Ī_k / σ_k ≥ 3`, or when the measured share of the sin χ weight is below 80 %:

```text
C_w = Σ_{k measured} sin χ_k / Σ_{k = 0…89} sin χ_k ≥ 0.8                                    (7.24)
```

The sin χ weight makes the range near the sample plane count most. A missing wedge of `abs(χ)` below 10° costs only
1.5 % of the weight (C_w = 0.985); an isotropic ring seen only for `abs(χ)` below 60° gives C_w = 0.50 and is refused.

**Hermans factor.** For a fibre texture about the surface normal, with signed background-subtracted bin means and positive total weight:

```text
⟨cos²χ⟩ = Σ_k Ī_k sin χ_k cos²χ_k / Σ_k Ī_k sin χ_k                         (7.25)
f = ( 3 ⟨cos²χ⟩ − 1 ) / 2                                                                    (7.26)
```

The sums run over the measured bins. f = 1 means all orientations along the normal, f = 0 random, f = −½ all in the
plane (Hermans & Platzek, 1939; for GIWAXS see Baker et al., 2010).

Both sums use the solid-angle measure for pole density, sin χ (5.12), for either map Lorentz setting. Signed net intensities allow upward and downward background fluctuations to cancel. Near zero total signal the ratio is unstable.

**Isotropic reference.** Because unmeasured bins bias f, the report gives the value an isotropic ring would have on
the same measured bins:

```text
f_iso = f  evaluated with Ī_k = 1 on every measured bin                                      (7.27)
```

Compare f with f_iso: on a partly measured ring a random orientation does not give f = 0. With the code's 1° bins, an
isotropic ring with `abs(χ)` below 10° missing gives f_iso = −0.015; with `abs(χ)` below 20° missing, f_iso = −0.058
(C_w = 0.94); a ring measured only up to `abs(χ)` = 80° has f_iso = +0.10.

**Uncertainty.** Monte Carlo with 200 draws and a fixed seed (0):

```text
f^(m) = f( Ī_k + ε_k^(m) ),   ε_k^(m) ~ N(0, σ_k²),   m = 1 … 200                             (7.28)
σ_f = standard deviation of { f^(m) }                                                        (7.29)
```

σ_f is a statistical uncertainty: it propagates the bin errors σ_k of section 6.4 (on the public P08 frame, a
floating-point frame, these are the pixel scatter). The uncertainty of b, of the window and of the geometry is not in
it; the systematic range below covers b and the bins next to the sample horizon.

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

A maximum next to an unmeasured range is flagged, since the true maximum may lie inside that range. The report's
*Decisions* line *texture rule* states the rule, and every ring row gives the measured f, its statistical error and
its systematic range. f − f_iso is a coverage diagnostic for the texture rule; it does not restore the intensity
missing from f.

**Systematic range.** f (7.25–7.26) is recomputed for each combination of exit-angle minimum × {0.5, 1, 2}, horizon intensity threshold {off, 0.05 M, 0.10 M} and background shift {−σ_b, 0, +σ_b}. Without a fitted standard error of the background, σ_b is 20% of |b|. Each resulting folded profile is used in these ways:

- The measured bins, with the gaps left empty. The nominal selection gives the reported central f and f_iso.
- The unmeasured high-angle edge continued to 90° at the mean net pole density of a 10° band. The band lies before an attenuated tail: among contiguous tails in the last 10° of measured coverage, from the longest down, the first whose bins are all measured and below 75% of the mean net density of the 10° before it. Without such a tail the band is the final 10° of coverage.
- With an attenuated tail, also the tail dropped with the horizon gap left empty, and tail and gap both filled at the band density.
- Separately, the wedge from 0° to the first measured bin filled at the mean net density of the first 10° of coverage. When the nominal pole fit and the fit for that selection are both determined, their model fill of this wedge also enters, with one propagated standard error on either side (§7.11).

Band means are arithmetic means of the measured folded Ī_k. A continued density is bounded below by zero and multiplied by sin χ_k **at the target bin** in both sums. Elsewhere the signed measured densities are kept. Internal gaps stay empty. The bracket gives the smallest and largest f across these alternatives and names the settings and scenario of each endpoint. When its width exceeds 0.05 the report flags that the measured χ range does not determine the orientation factor. An unmeasured peak stronger than the neighbouring band can put f outside this range; errors of the geometry and a background that varies with χ also shift f.

For the public P08 MAPbI₃ ring at 0.997 Å⁻¹ and α_i = 0.075°, the central estimate is f = 0.142 ± 0.00275 (stat), with f_iso = 0.0563. The unmeasured 84–90° interval accounts for 10.5% of the full sin χ weight (7.24). Continuing the 71–81° band through both 81–84° and the gap gives the lower limit of 0.0711 at b − σ_b. Omitting 81–84° and leaving the gap empty gives the upper limit of 0.157 at b + σ_b.

Implementation: `src/gimap/features/assistant/domain/orientation.py` and `src/gimap/features/analyze/domain/crystal/pole_systematics.py`, with the window in
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
instrumental width (FWHM_instr = 0)**, so every automatic size is a lower bound of the coherence length and is
reported with "≥". The beam footprint and divergence, the sample–detector distance and pixel size, the q binning, the
mixing of directions over all χ (Smilgies, 2009), strain and paracrystalline disorder all broaden the radial peak, and
none of them is subtracted. A size is also a lower bound when FWHM is below 3 bins (binning-limited). A script or
Process with AI can pass K and FWHM_instr to the tool.

No size is given when FWHM ≤ FWHM_instr or FWHM > 0.15 q_0. For such a halo the report gives 2π/FWHM as a correlation
length.

For example, q_0 = 1.50 Å⁻¹ and FWHM = 0.03000 ± 0.00060 Å⁻¹ give d = 4.19 Å and
τ ≥ 188.5 ± 3.8 Å. In nm the report gives τ ≥ 18.85 ± 0.38 nm.

Implementation: `src/gimap/features/assistant/domain/size.py`

### 7.9 What the report states

For each peak: q_0 ± σ, d ± σ_d, FWHM ± σ, height, local background, area, SNR, χ²_red, flags and caveat, the sector
preference and ρ_oi, and the size with its lower-bound mark and note. For each ring: the q window, coverage (unweighted
share of measured 1° bins), C_w, measured f ± σ_f [systematic range], isotropic reference, peak directions, missing and shadowed ranges, fitted f and oriented fraction. The record keeps the background, polarisation, exit threshold, scenario results and fit-quality reasons. Reports round each positive statistical uncertainty to two significant digits and the estimate to the same decimal place. Decimal ties round to even. The systematic range is shown separately. In compact tables, parentheses give the uncertainty in units of the last displayed digit. Search limits, settings and open questions accompany the results. The *Decisions* say what the numbers are:

- *q*: kinematic, no refraction correction. Near or below the film's critical angle, refraction shifts q_z by up to
  about k sin α_i, and the line gives that number for the frame (`2π/λ · sin α_i`; 0.0119 Å⁻¹ for the public P08
  frame at α_i = 0.075°, λ = 0.6888 Å), or says that it is unknown when α_i is missing (0°). With the
  film's critical angle given (section 2.6) it reads *q_z corrected for refraction in the film (αc = …)*;
- *uncertainties* names the statistical error model and the separate sensitivity range for f.
- *peak search* records the candidate cap, *texture rule* gives the isotropic comparison, and *ring orientation* states the exit cut and polarisation.
- *phase assignment* when phases were given, and *pole figure* when rings were analysed (sections 7.10, 7.11).

*Incidence angle* records the nominal value from the run options, notes, Analyze command bar, a unit-labelled sample incidence in the selected frame header, or instrument profile, in that order. A frame sum uses a header value only when its members agree. The frame measurement in §2.3 supplies a missing nominal angle and replaces a nominal angle that differs by more than three measured standard errors. **Keep entered αi** or `--keep-incidence` retains the nominal angle and still reports the comparison. When both sources are missing, the run uses 0° and asks for αi. The JSON record keeps geometry and peak-fit values to 15 significant digits. The tool-response snapshots use the precision specified in section 14.10.

Implementation: `src/gimap/features/assistant/application/pipeline_rows.py`

### 7.10 Reflection overlay and phase assignment

Analyze ▸ **Phases** (the image toolbar) takes known phases and shows where their reflections fall; the automatic
analysis assigns its rings to them. A phase is a unit cell `(a, b, c, α, β, γ)` with its symmetry and, optionally,
its atom sites. It is read from a CIF (`_cell_length_*`, `_cell_angle_*`, the operations of
`_space_group_symop_operation_xyz` or `_symmetry_equiv_pos_as_xyz`, `_atom_site_label`, `_type_symbol`,
`_fract_x/y/z`, `_occupancy`, `_U_iso_or_equiv` or `_B_iso_or_equiv`; parenthesized standard uncertainties on the cell parameters are retained for strain)
or typed in with a lattice centring (P, A, B, C, I, F, R in the hexagonal setting).

**A CIF without operations.** The operations are then generated from the space group the file names: its Hall
symbol (`_space_group_name_Hall`), else its Hermann–Mauguin symbol (`_space_group_name_H-M_alt`,
`_symmetry_space_group_name_H-M`), else its number (`_space_group_IT_number`, `_symmetry_Int_Tables_number`). A table
gives the Hall symbol of each of the 230 types in the settings of *International Tables* A (both origin choices where
there are two; hexagonal and rhombohedral axes of the R groups; the monoclinic groups with unique axis b, also as
`P 21/n`, `P 21/a`, `I 2/a`) and the other five axis settings of every orthorhombic P group (`P b n m` for P n m a);
the group is the closure of the Hall generators with the lattice translations (Hall, 1981). The symbol is read with
or without spaces, with `2_1` or `2(1)` for 2₁, the old cubic `Fm3m` and `Cmca` for Fm3̄m and Cmce. Where the symbol
leaves a choice open it is made and said: the origin choice that puts the atoms on fewer sites (atoms sit on special
positions at the right origin and spread over general ones at the wrong one; choice 2 when that is a tie), and the
hexagonal or rhombohedral axes from the cell. The cell must have the shape of the system and setting (a = b and 90°
for a tetragonal group, γ = 120° on hexagonal axes, α = γ = 90° for unique axis b; lengths within 0.2 %, angles
within 0.2°). Tests check the general positions, glide planes and screw axes of every entry and the extinctions of
16 groups (among them I4/mcm, Pnma, Fm3̄m and R3̄m); a MAPbI₃ CIF that names only I 4/m c m gives the reflections and
intensities of the CIF with all 32 operations.

When the operations cannot be found — a symbol that cannot be read or is not in the table, a cell of another shape,
no space group named — the phase gives **positions only**: every (h, k, l) ≠ 0 of the cell, |F|² = 1, no
extinctions. Its warnings stay with the phase (the Symmetry column, the record and the report), and a ring assigned
to it is `tentative` (7.42).

**Reciprocal lattice.** With the columns of A the vectors **a**, **b**, **c** in the Cartesian frame of the cell
(**a** along x, **b** in the x–y plane):

```text
B = 2π (A⁻¹)ᵀ,    q_hkl = B · (h, k, l)ᵀ,    d = 2π / |q_hkl|                                         (7.33)
```

Every (h, k, l) ≠ 0 with `|q| ≤ q_max` (3 Å⁻¹ by default) is tested. A reflection **h** is **systematically
absent** when an operation `x′ = R x + t` of the space group leaves it unchanged while its phase shift is not an
integer:

```text
h R = h   and   h · t ∉ ℤ                                                                          (7.34)
```

The contributions of an atom and its image then cancel for every position, so the rule needs no atoms.

**Intensity.** With atom sites (expanded by the operations; images closer than 10⁻³ in each fractional coordinate
count once), the squared structure factor is

```text
F(h) = Σ_j o_j f0_j(s) exp(−B_j s²) exp(2πi h · x_j),    s = sin θ / λ = |q| / 4π                     (7.35)
f0(s) = Σ_{i=1..4} a_i exp(−b_i s²) + c                                                              (7.36)
```

o is the occupancy, B = 8π² U the isotropic displacement (0 when the CIF gives none) and (7.36) the Cromer–Mann form
of the neutral-atom scattering factor with the coefficients of *International Tables* Vol. C, Table 6.1.1.4
(Cromer & Mann, 1968; Prince, 2004), kept in `scattering.py` for 66 elements; each row gives f0(0) = Z to within
0.06 electrons (tested). An element not in the table takes the f0 of the listed element nearest in Z, scaled by
Z/Z_listed (the record names it). Ions are taken as neutral atoms; anomalous terms f′, f″ are left out. Reflections
with |F|² below 10⁻⁶ of the strongest are not listed. Without atoms every allowed reflection has |F|² = 1.

**Powder rings.** Reflections at the same |q| (within 10⁻⁶ Å⁻¹) form a ring with the multiplicity m and the
intensity `I = Σ |F|²` (m |F|² for equivalent ones); the label names one reflection per family (related by a
rotation of the space group or by Friedel's law; for a typed cell, the same indices up to order and sign). Rings are
drawn as dashed lines on I(q) for every phase shown, and as arcs on the q map and the detector for a powder phase.

**Fibre spots.** For crystallites at random about the surface normal with the plane (h_p k_p l_p) parallel to the
substrate, or the axis [u v w] along the normal,

```text
n = B h_p / |B h_p|   or   n = A u / |A u|,     q_z = q · n,     q_∥ = √( |q|² − q_z² )                  (7.37)
```

Reflections at the same (q_∥, q_z) are one spot with their multiplicity and Σ|F|². Only q_z ≥ 0 is listed: the
crystallites turned upside down put the spot of −h there. Each spot is drawn at +q_∥ and −q_∥, its size grows as
√(I / I_max) of the phase, and its χ = atan2(q_∥, q_z) is the polar angle of the plane normal (the χ of a pole
figure, 7.11).

**Refraction and the two channels.** Optionally (*Shift the fibre spots for refraction in the film*) a spot is
placed where refraction in a film of critical angle α_c puts it. The incident wave inside the film has

```text
k_z,i = k Re√( sin²α_i − sin²α_c )                                                                  (7.38)
```

and a reflection at the true (q_∥, q_z) leaves the film upwards in two channels of the distorted-wave Born
approximation (Jiang, 2015; Toney & Brennan, 1989; Savikhin et al., 2020):

```text
transmitted (T):  k_z,f = q_z − k_z,i          reflected (R):  k_z,f = q_z + k_z,i                     (7.39)
sin α_f = √( (k_z,f / k)² + sin²α_c ),     k_z,f ≥ 0                                                  (7.40)
```

The reflected channel is the wave reflected by the substrate first; the two terms with a reflected exit wave leave
along the same two directions. The spot is drawn at the q_z of its exit angle on the map shown: `k (sin α_f +
sin α_i)` on the kinematic map (2.7), or (2.18) when Analyze's refraction correction is on, so with the same α_c the
T spot falls on its true q_z. A T spot with q_z < k_z,i is scattered into the substrate and not drawn. α_c of the
shift is the one typed in the window, or else Analyze's. Without the shift every spot is drawn at its true q_z, and
q_∥ does not change at the surface. Hand check (tested): λ = 1 Å, α_i = 0.2°, α_c = 0.15°; a spot at the true
q_z = 0.122923 Å⁻¹ appears at 0.131583 Å⁻¹ on the kinematic map (α_f = 1.0°, the example of 2.6), and 2 k_z,i higher
in the reflected channel on a corrected map.

**Detector position and missing wedge.** The exit angle follows from the map q_z (`q_inverse.exit_sine_of_qz`), the
in-plane angle from q_∥:

```text
cos 2θ_f = ( cos²α_f + cos²α_i − (q_∥ / k)² ) / ( 2 cos α_f cos α_i ),   q_y = sign(q_∥) k cos α_f sin 2θ_f     (7.41)
```

and the ray is met with the detector plane, tilted or not (the inverse of (2.3), `q_inverse.positions_of_q`). A
spot with |cos 2θ_f| > 1, i.e. `q_∥ < k |cos α_f − cos α_i|`, lies in the missing wedge: it is drawn hollow on the q
map and not on the detector. Tested: the detector position of a spot gives back its q_∥ and q_z to 10⁻⁹ Å⁻¹.

**Assignment of the rings.** A peak of I(q) at q_0 ± σ(q_0) with width FWHM (7.4) is assigned to the nearest powder
ring position |q_hkl| of the phases shown when

```text
|q_0 − q_hkl| ≤ δ,    δ = max( 3 σ(q_0), FWHM / 2 )                                                 (7.42)
```

σ(q_0) of a strong ring is often 10⁻⁴ Å⁻¹, below the accuracy of the cell constants and the geometry, while a
reflection within the half width still lies under the peak; the window therefore takes the larger of the two. For each candidate the differences are `Δq = q_0 − q_hkl` and
`Δd/d = (d_0 − d_hkl)/d_hkl = q_hkl/q_0 − 1`. The latter is the exact fractional difference relative to the reference
spacing. Two or more reflection families in the window make the assignment ambiguous, even if all belong to one phase.

The *Phase assignment* report identifies the q used for I(q). With no film refraction correction it is the kinematic
|q|. With a film α_c applied to the reduction it is the transmitted-channel |q| using the internal q_z of (2.6).
Near the horizon q_z can shift by about k sin α_i; the report gives this value for the frame. Above α_c one reflection
can give two rings, one per channel; the assignment compares radial peak centres with powder positions and does not
tell the channels apart.

Provide candidate phases in the Phases window or with `--phase CIF` of `tools/gimap_agent.py auto`. The report
lists the nearest and alternative reflections, marks ambiguous assignments and lists the measured q of unassigned
peaks. The JSON records the q axis, match tolerance, Δq, Δq/σ(q_0), Δd/d and alternatives. A match to a phase
known only by its reflection positions is also marked tentative.

Validation (tested). The first nine LaB₆ rings (Pm3̄m, a = 4.1569 Å) fall on their d-spacings to 10⁻⁴ Å and on
GIMaP's LaB₆ calibrant lines (3.1) to 10⁻⁶ Å⁻¹. In MAPbI₃ (I4/mcm, from a CIF with its 32 operations) 111, 101, 100,
001, 113 and 301 are absent by (7.34). On the public P08 frame the peaks at 0.8989, 0.9972, 1.4091, 1.9902 and
2.2255 Å⁻¹ are assigned to PbI₂ 001 and MAPbI₃ 110, 112, 004 and 114 (MAPbI₃ 002 also lies in the window of the
0.9972 Å⁻¹ peak); the peak at 0.452 Å⁻¹ has no reflection within ±0.026 Å⁻¹ and stays unassigned.

Not modelled: the Lorentz–polarisation and absorption factors of the ring and spot intensities (the sizes and the
I column are m|F|²), peak widths, the DWBA amplitudes of the channels (the reflected spots are drawn without a
weight), more than one film layer, and anomalous scattering.

Implementation: `src/gimap/features/analyze/domain/crystal/` (`cell.py`, `symmetry.py`, `cif.py`,
`scattering.py`, `reflections.py`, `fibre.py`, `assignment.py`), `src/gimap/features/analyze/application/indexing.py`,
`src/gimap/features/assistant/application/pipeline_phases.py`

### 7.11 Pole figure of a ring and its orientation distribution

**Profile.** The pole fit uses the same signed I(χ) means, background, exit-angle threshold, polarisation, folding and shadow mask as the Hermans calculation (§7.7). The dialog can also take a user-supplied b.

~~~text
N_k = Ī_k sin χ_k,   σ(N_k) = σ_k sin χ_k                                    (7.43)
~~~

**Models.** χ is the polar angle of the plane normal from the surface normal; ρ(χ) is the pole density per
steradian, normalised over the hemisphere, and N(χ) the fraction of crystallites per radian of χ:

```text
∫₀^{π/2} ρ(χ) 2π sin χ dχ = 1,        N(χ) = 2π ρ(χ) sin χ                                         (7.44)
ρ(χ) = (1 − φ) g(χ) + φ / 2π                                                                        (7.45)
Gaussian:      g ∝ Σ_{c = χ₀, −χ₀, 180°−χ₀} exp( −(χ − c)² / 2s² ),   s = FWHM / 2√(2 ln 2)             (7.46)
Maier–Saupe:   g ∝ (1/2π) ∫₀²π exp(m u²) dψ                                        (7.47)
u = cos χ cos χ₀ + sin χ sin χ₀ cos ψ
```

The Gaussian in (7.46) is a cone with a Gaussian spread in polar angle and image terms at the folds. **Gaussian mosaic** uses the same azimuthal average as (7.47) with the antipodal Gaussian kernels exp[−arccos(±u)²/(2s²)] summed in the integrand. Its width describes the spread about the tilted director. For Maier–Saupe the intrinsic FWHM is 2 arcsin √(ln 2 / |m|) when |m| > ln 2. Fibre averaging integrates over all director azimuths ψ.

**Fit.** `N_k = A · ⟨N(χ)⟩_k · 1°`, the model averaged over each 1° bin (12 points), is fitted by weighted least
squares (`scipy.optimize.least_squares`, bounded: 0 ≤ φ ≤ 1, 0.3° ≤ FWHM ≤ 150°, −60 ≤ m ≤ 60000, 0° ≤ χ₀ ≤ 90°) to
the measured bins after the exit-angle selection. An optional additional upper χ limit defaults to 90°. The direction is fixed at 0°
(along the normal, the default of the window), at 90°, at a value typed in or at the χ of a reflection chosen in the
table (*Pole Figure of This Ring*), or fitted; a fitted direction that ends within 1° of 0° or 90° is fixed there,
where its error is not defined. Starting values come from the data (the scale from the sum, φ from the lowest pole
densities, χ₀ from the highest), and the fit also starts from FWHM = 4°, 15° and 45° and, for a free direction, χ₀ = 20°, 45° and 70°. The polar-angle Gaussian also starts at φ = 0.5 and Maier–Saupe at m = −20 for a girdle. The converged solution with the lowest χ² is kept. Errors are `√diag((JᵀJ)⁻¹) · max(1, √χ²_red)` as (7.12);
those of derived quantities by linear propagation through the numerical Jacobian. They are statistical: b, the
window and the model are taken as exact.

**Results.** The oriented fraction 1 − φ, the FWHM, χ₀, and the Hermans factor of the fitted distribution

```text
⟨cos²χ⟩ = ∫ ρ cos²χ sin χ dχ / ∫ ρ sin χ dχ,     f = (3⟨cos²χ⟩ − 1)/2 = (1 − φ) f_oriented             (7.48)
```

which includes the missing wedge as the model fills it, next to f from the measured bins alone, its same-bin isotropic reference f_iso (7.27),
and the 2D order `2⟨cos²χ⟩ − 1` of ρ along the ring without the sin χ weight (the orientation parameter
SciAnalysis reports). Statistical errors of the fit and of its derived moments use the fitted covariance. For the filled-wedge sensitivity, the measured bins are held fixed and the Jacobian of the hybrid moment propagates covariance of all fitted parameters through the extrapolation.

**Automatic analysis.** A Gaussian cone plus an isotropic component is fitted to each selected ring. The highest I(χ) maximum fixes χ₀ to 0° below 20° or to 90° above 70°. Other directions are fitted. The report gives the measured f, the model f and the oriented fraction together. **1 − φ depends on the background b.** A fit is *not determined* when it does not converge, reaches a bound of a free parameter, gives χ²_red > 5 or FWHM > 100°, or has parameters that cannot be identified. *Needs attention* gives the reason; the values of such a fit are diagnostic only, and its extrapolation is left out of the systematic range of §7.7.

**Export.** *Export…* writes the profile as CSV (`chi_deg N sigma measured model`; the source, geometry and fit in
`#` lines) and `<stem>_indexing.json` beside it: the record of the frame (14.2), the phases with their source files
and SHA-256, the reflection table, the settings and the pole-figure fit at full precision.

Assumptions: a fibre texture (isotropic about the surface normal), one oriented population plus an isotropic
fraction, a background constant in χ and the sin χ measure on the hemisphere. The low-angle wedge is extrapolated from the chosen model.

Implementation: `src/gimap/features/analyze/domain/crystal/pole_figure.py`, `pole_models.py`,
`src/gimap/features/analyze/application/pole_figure.py`

#### Crystal orientation from assigned rings

All usable rings assigned to a phase constrain its fibre texture. A ring keeps its radial peak window and the local background from §7.7. The fit excludes windows shared by visible phases and assignments with unknown symmetry. In a unit cell without atom sites, each allowed reflection has unit intensity.

An oriented population a has a reciprocal-lattice plane normal **n**ₐ preferentially along the substrate normal, so planes (hkl) lie preferentially parallel to the substrate. Its mosaic FWHM is wₐ and sₐ = wₐ / √(8 ln 2) in radians. For reflection direction **ĝ**, cos ψ₍g,a₎ = |**ĝ**·**n**ₐ|. Averaging over rotation of the crystal about **n**ₐ gives the pole density Gₐ. With absolute population fractions pₐ and isotropic fraction φ,

```text
Kₐ(γ) = Cₐ {exp[−γ²/(2sₐ²)] + exp[−(π−γ)²/(2sₐ²)]},   ∫₀^{π/2} Kₐ(γ) sin γ dγ = 1
Gₐ(χ,ψ) = (1/2π) ∫₀^{2π} Kₐ(arccos[cos χ cos ψ + sin χ sin ψ cos α]) dα
Nᵣₖ = Aᵣ ⟨F_film⟩ᵣₖ ∫bin k sin χ [φ + Σₐ pₐ Σg Wᵣg Gₐ(χ,ψ₍g,a₎)] dχ
φ + Σₐ pₐ = 1,    Σg Wᵣg = 1
```

Each ring has its own scale Aᵣ. Every equivalent reflection in its window contributes |F|² (§7.10). An alternative fit estimates nonnegative family weights per ring while preserving calculated ratios within each family. The family curves include their shares of the isotropic signal and sum to the plotted prediction. Even Legendre moments evaluate the spherical average over the 1° bin.

**Film transmission.** Let nⱼ be the occupied number of atoms of element j in the unit cell of volume V. The CIF formula sum multiplied by the number of formula units Z supplies these populations when available. Otherwise the fit expands the occupied sites by symmetry. The phase record lists both occupied sites and formula × Z. A mismatch is flagged when elemental counts differ by more than the larger of 0.001 of the count and 0.0001 atom per cell. With a formula but no Z, only the relative elemental populations can be compared. At the frame wavelength λ, the neutral electron density, film index and reciprocal exit-wave field are

```text
ρₑ = Σⱼ nⱼ Zⱼ / V,    n = 1 − δ + iβ
(δ, β) = [rₑ λ² / (2πV)] Σⱼ nⱼ (f₁ⱼ, f₂ⱼ)
β = μλ/(4π)                                                (μ is the linear absorption coefficient)
u = sin αf = q cos χ / (2π/λ) − sin αi                    (external q)
|T_f|² = |2u / [u + √(u² − 2δ + 2iβ)]|²,    u > 0
```

Zⱼ is the atomic number. The forward factors for H, C, N, O, Na, Si, Ca, I, Au and Pb are interpolated over 4.895–30 keV. When factors are unavailable, δ is estimated from rₑλ²ρₑ/(2π), β is unknown and the fit retains the §7.7 exit cut. With film optics, each signed bin averages the full depth-integrated factor F_film (§5.4) over its measured pixels at their external exit angles, then combines ±χ with the intensity profile's weights. The fit takes the external exit angle from each measured detector ray, including when the q map uses refraction. For display bins without pixels it uses the ray at the centre of the ring window when that ray exists. Within the radial window of the active q map, each measured pixel is binned by its internal pole angle χ = atan2(|q∥|, qzᶠ), where qzᶠ = k Re[√(sin²αi − 2δ + 2iβ) + √(sin²αf − 2δ + 2iβ)]. Both square roots use the branch with nonnegative imaginary part. The in-plane momentum transfer is conserved. The single interface describes the enhancement near the critical angle; it does not describe a beam stop, an asymmetric sample shadow, a diffuse background that varies with χ, or a peak leaving the extraction window. [Henke scattering factors](https://henke.lbl.gov/optical_constants/asf.html) · [Gasser et al., 2025, §2.6](https://doi.org/10.1107/S1600576724010628)

The default orientation model uses the thick-film limit when the thickness is unknown. For a frame corrected with the geometrical absorption factor, the model uses the supplied thickness and divides F_film by the factor already applied to each pixel. A frame corrected with the refracted-wave model needs no further film multiplier. The P08 benchmark also fits one thickness shared by all rings of each phase, including that parameter in BIC and the degrees of freedom. It reports the optical depth τ = μd for a finite thickness. At the thick limit only the angular dependence can be inferred.

The search compares isotropic material with one or two oriented populations. Each has its own width (2°–120°) and axis, including two widths about the same axis. Primitive (hkl) with |h|, |k|, |l| ≤ 2 are grouped by symmetry and inversion. For N measured bins, BIC = χ² + k ln N. With m oriented populations, k includes 2m shared continuous parameters and one scale per ring, or one amplitude per family per ring when its weights are free. A pair must improve on each constituent's single-component BIC. Two distinct axes require noncollinear reflection poles. Against each constituent alone and its supported two-width fit, the two-axis prediction must change at least one measured ring by √(Σₖ(ΔNₖ/σₖ)²) > 1 after refitting scales and weights.

An orientation is **determined** when the joint and every ring’s reduced χ² are ≤ 5, the next candidate is at least 6 BIC units worse, the parameters are identifiable, widths lie away from their bounds and φ is below the isotropic limit. The per-ring degrees of freedom subtract that ring’s share of the weighted Jacobian leverage. The full covariance, scaled by max(1, χ²_red), gives conditional 1σ errors for each width and each absolute fraction including their correlations. These local errors exclude uncertainty in the background, optical constants, phase assignment and model choice. A parameter on a bound can have an asymmetric interval.

Each population is tested against an isotropic replacement on the measured bins, with the other populations and the ring amplitudes refitted. A BIC increase below 6 is weak evidence for that population. If either of two components can be removed, the better single-component fit is kept (the closer width for a tie). A second test uses the joint covariance. If pₐ < 3σ(pₐ) and corr(pₐ, φ) < −1/√2, more than half the local variance of pₐ is exchange with random material; the component is **near-random** when its replacement also raises χ² by at most one per measured bin on average. In a two-component fit the broader fraction is also pooled when |corr(p_broad, φ)| > 0.99 and p_broad < 3σ(p_broad), even when it improves the profile shape. For a single fibre component φ = 1 − p forces correlation −1, so this test does not apply. Near-random components and components with weak BIC support are pooled with φ, and their axes are omitted. The pooled error follows from the covariance of 1 − Σ pₐ over the retained components. An oriented fraction below three standard errors among the remaining components leaves the axis undetermined. The saved record gives the replacement ΔBIC, the fraction correlation and the original fitted width.

The reported crystal Hermans factor is Σₐ pₐ ∫ Kₐ(γ) P₂(cos γ) sin γ dγ over the hemisphere, with P₂(x) = (3x²−1)/2. The sum includes the resolved populations, and each population’s own axis defines γ. Each ring also reports its model-free f over the measured bins (§7.7). When the fit fails the quality criteria, its axes, widths and fractions are diagnostic estimates.

For the public P08 frame (λ = 0.688801 Å), the MAPbI₃ formula with four units per cell gives ρₑ = 1.0423 e Å⁻³, δ = 2.183 × 10⁻⁶, β = 1.269 × 10⁻⁷ and αc = 0.1197°. The benchmark CIF has refined positions for Pb and I only, and with automatic frame angles the roll stays at 0°. The fit to the film pole angles has joint reduced χ² = 6.63. The preferred candidate has a narrow population of 3.42 ± 0.52% with FWHM 11.43 ± 0.53°; the broader population is pooled with the isotropic material (fraction correlation −0.99985). The orientation is undetermined. [Public-frame benchmark](../tools/benchmarks/README.md#p08-crystal-orientation-at-the-measured-incidence-and-roll)

### 7.12 Unit-cell refinement from indexed peaks

For each visible phase, the observations are fitted radial centres qⱼ in Å⁻¹ with finite positive standard errors σⱼ from §7.4. Each measured peak contributes once. The assignment window is ±max(3σⱼ, FWHM/2). Tentative assignments and windows with distinct families of one or more phases are left out: texture can change the intensity ratio, so the centre of an unresolved multiplet has no measured weight for its families. After the first fit, all candidate positions are recalculated with the refined metrics and q offsets and the measured peaks are assigned again, until the assignments are stable. Reflections equivalent under the free metric count as one family.

Let A contain the direct lattice vectors as columns, M = (AᵀA)⁻¹ and hⱼ be the column of CIF indices. The independent entries of the reciprocal metric are fitted by minimizing

```text
q̂ⱼ = 2π √(hⱼᵀ M hⱼ) + Δq₀
χ² = Σⱼ [(qⱼ − q̂ⱼ)/σⱼ]²
```

The cubic cell has one independent length, tetragonal and hexagonal/trigonal cells have a and c, and orthorhombic cells have a, b and c. A monoclinic cell also has the unique-axis angle (normally β), and a triclinic cell has all three angles. For rhombohedral CIF indices, the reported conventional hexagonal lengths are |bᵣ−aᵣ| and |aᵣ+bᵣ+cᵣ| and the reported volume is 3Vᵣ. The CIF symmetry defines the constraints; for a typed cell with centring alone the reference metric sets them.

An additive q zero shift Δq₀ represents an empirical offset over the measured range. For p independent metric parameters, it is fitted only with at least p+2 usable peaks and an independent shift column in the weighted Jacobian. Otherwise it is fixed at zero. The report gives the fitted offset and its standard error. A wavelength error rescales q. A detector-distance error approximates a common scale only at small scattering angles. The metric design must have full rank. For example, tetragonal 202 and 404 constrain only 4/a² + 4/c², even with both centres measured precisely.

**Refraction.** With a supplied film critical angle, use transmitted-channel q for both the incoming and outgoing beam as in §7.10. For a propagating exit ray, q∥ is conserved and qz = k[√(sin²αf − sin²αc) + Re√(sin²αi − sin²αc)]. When the overlay selects a different αc, the frame is binned again on this map and peak positions and errors are fitted again. A full ring has no single exit angle. A curve already reduced with that αc supplies its fitted centres directly. For a supplied 2D spot, the measured direction gives its exit angle, and its radial error is propagated at fixed direction as σT = |∂qT/∂qmap| σmap. The model assumes the transmitted branch for these positions. [Jiang, J. Appl. Cryst. 48, 917–926 (2015)](https://journals.iucr.org/j/issues/2015/03/00/rg5085/).

**Errors.** For the weighted Jacobian J and ν = N − number of fitted parameters, C = (JᵀJ)⁻¹ max(1, χ²/ν). At ν = 0 the supplied absolute peak errors determine C and χ²red is undefined. The full covariance propagates to the direct cell and V = |det A|. For a length u, εᵤ = u/u₀ − 1 and σ²(εᵤ) = (σᵤ/u₀)² + (u σ₀/u₀²)² for independent measured and reference cells. The volume change follows the same formula, with reference-cell errors propagated through V. Dimensions tied by symmetry share their CIF uncertainty. Other reference parameters are treated as independent. A missing CIF uncertainty contributes no known variance. Leave-one-centre-out fits give the range of parameters for retained subsets and identify subsets of insufficient rank. Separate fits remeasure radial positions with refraction off and on at the supplied film critical angle or the angle from CIF optics. Residuals are observed minus fitted q in Å⁻¹.

**Directional strain.** For one resolved fibre pole, each reference reciprocal vector is split into P perpendicular to the director and Z along it. The directional model fits q̂ = √[(P/(1+ε∥))² + (Z/(1+εz))²] + Δq₀ against independent measured radii with the same weighting and rank rules. A measured spot direction selects its symmetry-equivalent pole. For sector I(q) the angular limits must select a single distinct reference pole angle. The panel uses the non-overlapping in-plane and out-of-plane sectors. The fitted pole indices are held fixed. For the directional strains and any fitted offset, the reported covariance adds G C₀ Gᵀ. G is the derivative of the refitted parameters with respect to each independent CIF cell coordinate, including the change in director and reciprocal-vector projections. C₀ contains the reported reference variances.

**Series.** The selected range and stride refer to individual physical frames. Each accepted ring is tracked in both directions, starting near the indexed frame. The Series centroid initializes a Gaussian on a linear background fitted to native radial intensities with their σ. The search half-width is the larger of 2.5 indexed FWHM and five radial sampling steps, limited by midpoints between neighbouring peaks in the indexed frame. Each frame uses its own fitted qⱼ, σⱼ and covariance. A provisional fit to the tracked orders sets the metric and offset for checking every visible phase against the measured peak widths. Only isolated families that retain their tracked identity enter the final fit. An unresolved or missing peak is left out; an undetermined frame leaves a gap in the plot.

## 8. GISAXS automatic analysis

For a GISAXS frame the automatic analysis places the horizontal cut at the Yoneda band, moves the beam-centre column
to the mirror axis of the cut, decides which halves of the cut to use, gives the length scale 2π/q* of a side maximum
or shoulder (section 8.4), and fits the prepared cut with Predict parameters (section 8.5). It runs from **Run Automatic Analysis** in
Analyze and from the command line. The technique and the frames are chosen as in section 7.1.

Units: q_y and q_z in Å⁻¹ in Analyze; the fit converts to nm⁻¹ (q_nm = 10 q_Å); lengths in nm.

Implementation: `src/gimap/features/assistant/application/gisaxs_procedure.py`

### 8.1 Yoneda band and the cuts

Section 6.7 defines the horizontal Yoneda-row band and the off-rod vertical band. Each gives one point per occupied detector line with the standard errors of section 6.4. Gaps between measured lines remain gaps.

The Yoneda search of section 6.7 is empirical: it takes the brightest diffuse row between `α_f = −0.2°` and `α_f = 1.5°`, whatever its origin. The report gives r_Y and its exit angle α_f(r_Y) from (2.14). When the window holds no valid pixels, the cut is placed 10 px above the horizon, with a warning and a *Needs attention* item (check α_i).

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
probability. A shift of more than 20 px from the calibrated centre is put under *Needs attention*. The automatic
analysis does not apply a shift of more than 100 px, since the calibration of a measurement is off by a few pixels at
most: it keeps the calibrated centre and reports that the calibration does not match the frame. An axis that cannot be
found is an open item too. The report's *asymmetry a → b* is E at the calibrated and at the refined axis.

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
the fitted D differs from 2π/q* when the disorder is large. On 14 BornAgain simulations with known parameters, 2π/q*
was within 10 % of the true distance in 1 of 10 cuts with interference, and a maximum or shoulder appeared on all 4
cuts without interference. D* is therefore a result of its own, and the fit of section 8.5 takes no start from it. The
records keep its value under the key `spacing.distance_nm`.

Implementation: `src/gimap/features/assistant/domain/gisaxs_cut.py`, `src/gimap/features/assistant/application/gisaxs_tools.py`

### 8.5 The model fit and how the solution is chosen

The prepared curve (section 8.3; q = 10 `abs(q_y)` in nm⁻¹) is fitted with **Predict parameters**, the network method
of Fitting (section 9.13): GIMaP's network proposes compositions and values, every solution is refined by bounded
least squares in Fitting's model (9.1), its posterior is sampled around it, and the solutions are weighted by their
posterior probability. The same function fits the cut in Batch Export's *Particle model of the horizontal cut*, in
Process with AI and the MCP tool `fit_horizontal_cut`, and on the command line (`tools/gimap_agent.py`). It takes about
25 s per cut on one core.

**Model and compositions.** When the geometry supplies α_i and λ and the cut is on the Yoneda band of exit angle α_f,
the automatic fit uses four-term DWBA if the session or frame metadata identifies the substrate (level 4).
Otherwise it uses the Born term at the cut's vacuum q_z = (2π/λ)(sin α_i + sin α_f), in nm⁻¹ for λ in nm
(level 2, section 9.13). When either angle is unknown, it uses the effective interface at q_z = 0 (level 1).
The report states the level under *Decisions*. The other model settings are Fitting's defaults: sizes Gaussian
by number, the paracrystal of section 9.5 in the LMA, the beam tail A / (1 + (q/w)^ν) and a constant background (9.1).

Without a given composition (as in Run Automatic Analysis) the network chooses among one or two populations of
spheres and upright cylinders, each with or without a paracrystal (the head's four most probable of these, and its
most probable of one and of two populations). A given composition (1 sphere, 2 random cylinder, 3 vertical cylinder,
each population with a paracrystal: Batch Export's **Model**, the `components` of the tool) is fitted whole. Random
cylinders and three populations are fitted only then; random cylinders take far longer. The network needs no start.
The fit needs at least 8 points; a cut of more than 1000 points is merged first (section 6.9).

**Scores.** For each solution, with r_i = (I_model(q_i) − I_i)/σ_i:

```text
χ²ᵣ = Σ_i r_i² / (N − k)                                                                      (8.6)
logRMSE = √( mean over I_i > 0 of ( ln(I_model,i / I_i) )² )                                      (8.7)
```

N counts the points Predict parameters fitted and k its free values (the scales included); σ is the curve's own.
Predict leaves out the points below the network's q window (q < q_max/300, section 9.13) and those without a usable
value, and every report says how many (*596 of 598 points; 2 below the network's q window*; `points` of a solution:
`fitted`, `given`, `below_window`, `unusable`). χ²ᵣ is Fitting's χ²_red (9.23) of the same solution on those points.
It is the one number shown for a solution on the Fit page, in the automatic analysis's table, report, CSV and JSON,
and in Batch Export's table.

The probability p of a solution is its composition's posterior probability times the solution's share of it
(section 9.13). The solutions are sorted by p, most probable first, and the five most probable are kept.
Probabilities are relative to the compositions tried; the single solution of a given composition shows none. A
solution's "weight" `a_j / Σ a` is a normalised amplitude, not a probability or a volume fraction.

**Values at a bound.** The least squares of Predict parameters keep every population inside the box of the network's
prior (section 9.13: R from 0.5/q_max to 5/q_min, σR/R, σh/h and σnn/D 0.02–0.5, D and h within the prior's D/2R and
h/2R ranges at the current radius, the tail's ν 1.8–2.2 and w at most 30 q_min). A free shape or tail value
within 10⁻⁴ of its range of a bound has *ended on the bound*, and the solution names it in `at_bounds` (parameter,
lower or upper bound, component). The keys `sigma_Res` and `nu_Res` are the tail's w and ν, which every text calls
Tail w and Tail ν, the names of Fitting's Model table; `shared/fit_names.py` holds the one name of every quantity
and the sentence of every flag. With Tail A = 0 the tail is absent, and its w and ν are neither listed nor flagged;
the same holds for the values of a population whose Scale is 0. The linear terms a_j, BG and A are solved by
non-negative least squares and are not in `at_bounds`. One at 0 is held by that limit (a background the curve would
want below zero, or an absent beam tail, so that w and ν no longer change the curve); the report names it for the
reported solution among the *Decisions* (*fit caveat*: *Background at 0, the non-negativity limit*).

**The solution the automatic analysis reports** is the most probable one, the first. The fit record keeps its index
(`chosen`), the Results tab selects it, and `fit_curve.csv` holds its curve. The report's *Decisions* give:

- *fit*: the model, the R of every population and the D of those with a paracrystal in the order of the radii
  (`R = 3.06 / 6.02 nm, D = 49.4 / – nm`), χ²ᵣ and the probability;
- *fit intervals*: the refined R, h and D of every population, each followed by its 68 % interval
  (`R (1) 3.17 [3.04, 3.26] nm`), from the posterior sampled around the solution, or the least-squares errors when the
  sampling did not converge (section 8.6, *uncertainties*);
- the notes of the fit: the flags of section 9.13 as sentences, one per kind, with those about the background, the
  beam tail and the resolution in one line after those about the particles;
- *fit caveat*: its values at a bound.

Further rules of the report (`gisaxs_choice.py`):

- If another composition has probability or BIC weight ≥ 0.2, *Needs attention* (**model**) lists the reported
  composition and each such alternative with its composition probability or BIC weight. Each composition is represented
  by its most probable solution, and the report marks representatives whose D is within 15 % of D* (section 8.4).
  For a model with several solutions, the fit decision gives both the model's total weight and the reported solution's weight.
- Each other solution of the reported composition with solution weight ≥ 0.2 is compared with the reported one on the
  refined R, h and D of each population. When the 68 % intervals of every applicable value overlap, the two starts
  ended in one mode, and the fit decision records this. When the intervals of a value do not overlap or an interval is
  missing, *Needs attention* (**values**) names the parameter and shows the estimates and the available intervals.
- A value of the reported solution at a bound: *Needs attention* (**fit**) says that the curve does not determine it.
- D ≥ 250 nm in the reported solution: there is no interparticle correlation in the q range, and the curve does not
  determine D.
- A reported solution with χ²ᵣ > 10 is a **poor fit** (*Needs attention*, **fit**): none of the compositions tried
  describes the cut, and its R and D are not measurements. Fitting uses the same limit for the poor fits of a series
  (User Manual, section 6.5).

The table and *Needs attention* use the same fit record. Where a 68 % interval is available, its half-width sets the decimal place for R, h and D. **Save Fit Table…** writes one line per
population of each solution: rank, model, χ²ᵣ (`chi2`; a `#` line gives the fitted points), probability, log RMSE, the parameters (nm; the spreads relative), the
68 % intervals of R, h and D, `at_bounds` (`name:upper` or `name:lower`) and `chosen`.

**GALAXI.** On the public GALAXI frame (α_i = 0.463°, `galaxi_bornagain.poni`; the worked example of User Manual
section 10.7) the leading model is a sphere with a paracrystal and an upright cylinder. Its probability rounds to 1.00,
shared by two solutions whose 68 % intervals overlap (R (1) 3.267–3.441 and 3.294–3.471 nm), so the report records one
mode and asks nothing about the values. The two populations describe the cut's broad log-normal sizes and its low-q
rise. The sample's particles are spheres (BornAgain's model, section 9.10); a fit with a sphere in Model and the
particles of Model as the composition describes them (section 9.13). The choice of model is the user's: the automatic
analysis describes the cut and gives Fitting a start.

Implementation: `create_curve_fit()` in `src/gimap/features/fitting/bootstrap.py` (Predict parameters as a function:
`domain/posterior/refine.py`, the rows of `infrastructure/adapters/posterior_job.py`), the choice and its texts in
`src/gimap/features/assistant/application/gisaxs_choice.py` and `gisaxs_procedure.py`, forward model `cut_model`
(`src/gimap/features/fitting/domain/cut_model/`)

### 8.6 What the GISAXS analysis does not model

- With the Born term (level 2, section 8.5) the fit uses a single vacuum q_z for the Yoneda band; reflected paths,
  refraction and the change of the Yoneda enhancement across the band are left out.
- The term `A / (1 + (q/w)^ν)` is an additive empirical tail of the direct and reflected beam; no resolution
  convolution is applied.
- The Yoneda row is found without αc, and the beam-centre row is not refined (sections 8.1, 8.2).

Like the GIWAXS report (section 7.9), the GISAXS report's *Decisions* always say what its numbers are:

- *q*: kinematic, no refraction correction. Near the critical angle refraction mostly shifts q_z, most of all on the
  Yoneda band (α_f ≈ α_c), by about k sin α_i; the line gives that number for the frame (`2π/λ · sin α_i`; 0.0379 Å⁻¹
  for the public GALAXI frame at α_i = 0.463°, λ = 1.34 Å). q_y and 2π/q* barely change. With the film's critical
  angle given (section 2.6) it reads *q_z corrected for refraction in the film (αc = …)*;
- *uncertainties*: statistical only, from the σ of the cut's points (which normalise χ²ᵣ) with the error model of the
  frame (Poisson counting statistics, or the pixel scatter of each bin). The geometry and refraction are not included.
  The 68 % intervals of R, h and D (*fit intervals*, section 8.5) are those of the posterior within the model given
  these σ; the choice of model is not in them.

## 9. Model fitting of 1D curves (Fitting)

Fitting fits one curve, or every frame of a series, with its own model. The Fit step has three methods: the
numerical *Refine the values in Model* (Refine) and *Search the ranges, then refine* (Search), described in this
chapter, and *Predict parameters*, GIMaP's network and the posterior of this model (section 9.13). Predict parameters
also fits the cut of the GISAXS automatic analysis (section 8.5); the 1D Predict window is chapter 10. *Find the
particle shape (no AI)* and *Predict parameters (fit1d)* were removed in 0.2.0 (sections 9.9 and 9.12).

**Units: Fitting works in nm⁻¹ internally.** A curve read in Å⁻¹ (an Analyze `_fit_input.dat`) is multiplied by 10.
Sizes and distances are in nm, w in nm⁻¹, the spreads σ_R/R, σ_h/h, σnn/D are relative.

**The cut's q_z.** With αi, αf and λ, Predict parameters evaluates spheres with two paths and a free effective r
when the substrate is unknown (section 9.10), or with four paths and fixed reflection coefficients when the substrate
is known (section 9.17). Both Predict panels accept the substrate under **Cut geometry and substrate…**.
The 1D Predict window also accepts a single **Cut qz (nm⁻¹)** for a Born fit (chapter 10). The effective interface
uses q_z = 0. Run Automatic Analysis takes αf from the Yoneda band and αi and λ from the geometry (section 8.5).

### 9.1 Points that are fitted

**Text-table input.** The import preview shows the header and the first ten rows of a `.dat` or `.txt` file.
The single-curve layout reads `q I [sigma]`. The shared-q layout reads `q I1 I2 …` as separate curves, each paired
with the first q column. Column names such as `q I1 I2` suggest the shared-q layout; an unlabelled three-column table
stays single-curve until you choose otherwise. Analyze's marked exports and its `q I sigma pixels` schema are always
one curve. Named uncertainty columns (such as `sigma1`, `sigma2`) also turn off shared-q splitting. An interleaved
`q I1 sigma1 I2 sigma2 …` table is read as one curve, from its first q/I/sigma triple. A unit stated in the file
overrides the setting for files without a unit; contradictory unit declarations are an error. A named intensity
column is never used as sigma: rearrange a table such as `q I1 I2 sigma` into a supported layout.

Each curve takes its q, I and optional sigma from the same row. Rows whose selected values are missing, unreadable or
not finite are dropped from that curve. When the table keeps its column positions (an empty comma-separated cell or
`nan`), a missing I1 leaves a valid I2 at that q. A whitespace-separated row with too few values is skipped, since the
missing column cannot be identified. Missing sigma stays missing; no counting errors are added. The legacy
single-curve reader also skips a row whose sigma is malformed.

Each imported curve records its source file, the selected column indexes, the original column names (if any), the
table width and whether sigma came from a column. Projects and fit records keep this mapping. When the header has
changed or a selected column is missing, the program reports it and asks you to import the table again.

A curve with signed q (a horizontal GISAXS cut) is fitted on one of: the mean of both halves (6.16)–(6.17), both halves on
`abs(q)`, or one half. Then the fitting range and points left out apply. Of these points only `abs(q) > 0` with finite
values are kept. With file weighting, only finite, positive intensity uncertainties are used.
If the selection has no usable σ, its positive intensities are fitted in ln I (9.7).

Implementation: `src/gimap/features/fitting/application/curve_sources.py`, `src/gimap/features/fitting/infrastructure/adapters/curve_tables.py`, `src/gimap/features/fitting/infrastructure/adapters/scattering_curve_files.py`, `src/gimap/features/fitting/application/single_fit.py`, `src/gimap/features/fitting/domain/fit_engine.py`

### 9.2 The intensity model

```text
I(q) = BG + k · [ Σ_i Int_i · T_i(q) + A / ( 1 + (q / w)^ν ) ],   q = abs(q) in nm⁻¹        (9.1)
T_i = P_i · S_i  (LMA)   or   P_i + C_i (S_i − 1)  (DA),   P_i(0) = 1
```

Int_i is the scale of component i. Every family is normalised to P(0) = 1 (section 9.4), so Int_i is the intensity
of component i at q = 0 (times S_i(0)). BG is a constant background (not multiplied by k), A, w and ν the amplitude,
width and exponent of the beam-tail term, and k an overall factor fixed at 1 by default. By default the model depends on
`abs(q)` only: spheres are evaluated at q_z = 0, the effective interface approximation. With the angles of the cut
entered, spheres can use the direct and the exit-reflected path of the DWBA instead (section 9.10). The beam-tail term is
additive, not a resolution convolution. With w ≤ 0 or ν ≤ 0 the term is 1.

**q resolution (σ_q, off by default).** With σ_q > 0 each particle term `Int_i·T_i` is convolved with a Gaussian
of standard deviation σ_q in q; the beam-tail term and BG are not. The terms are made on a uniform grid from
max(0, q_min − 5σ_q) to q_max + 5σ_q, at most σ_q/5 apart, folded at q = 0 (the cut is symmetric: I(−q) = I(q)),
convolved with the Gaussian cut at ±5σ_q and interpolated back to the points (within 10⁻³ of a direct convolution).
Set σ_q for well-ordered films: when σ_q is above about 0.3 of the width of the structure peak, a fit without it takes
up the resolution as disorder in σnn/D.

**Resolution from the file (Refine and Search).** For a measured Gaussian standard deviation dQᵢ, each particle term
T(qᵢ) in (9.1) becomes E[T(|qᵢ + dQᵢ Z|)], Z ∼ N(0,1); a zero width gives T(|qᵢ|).
The integral uses the normalized Gaussian over −7 ≤ Z ≤ 7, with adaptive Gauss–Legendre quadrature and explicit
subdivision at sharp 1D paracrystal peaks. Each scale column uses the same convolution. BG and the beam tail
keep their values at the measured qᵢ. A named dQ or sigma_q text column, or a Gaussian NXcanSAS Qdev,
supplies one finite, nonnegative standard deviation in inverse length per fitted point. With both halves
on |q| each measurement keeps its width. For their mean, the two Gaussian responses are approximated by
one Gaussian with the mean of their variances, interpolating variances as needed. Where only one side contributes
its width is used.

The plot's **Components** are the additive intensities `k·Int_i·T_i`, the scaled beam-tail term and `BG`; their sum
is the total model. Component visibility and axis limits change only the plot. A logarithmic axis leaves out zero and
negative values.

**One forward model.** Fitting computes (9.1) with `cut_model`, GIMaP's one implementation of a 1D cut, which the
proposal network's training and refinement are to use as well. On q = 0.02–3 nm⁻¹ its largest relative deviation over a
curve is 2.7 · 10⁻⁵ from an independent implementation (`tools/benchmarks/yoneda_truth.py`, no shared code) and
3.9 · 10⁻⁴ from BornAgain 24 for spheres and upright cylinders on Si in the four-term DWBA with σR/R = 0.2
(`tools/benchmarks/cut_model_checks.py`, `cut_model_bornagain.py`; [README](../tools/benchmarks/README.md)).
`cut_model` uses the reflection of the specified substrate with air around the particles, so BornAgain's *average
layer materials* is off in this comparison. BornAgain's default gives the particle layer its volume-averaged
refractive index, which changes the reflection the DWBA terms carry: for spheres R = 4 nm on a paracrystal of
D = 14 nm (about 17 % of the layer) the curve then differs by 32 % (median). The free effective r of the two-path
model takes up such a difference (section 9.10). One core computes 1000 curves of 500 points (R 1–15 nm, σR/R
0.05–0.4) in 1.4 s (spheres), 1.2 s (upright cylinders) and 11 s (random cylinders).

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

Each component's **Sizes** defines the fitted location and spread. New manual populations use a normalized
number distribution n(x) for radius x. The same law applies independently to a fitted cylinder height.

```text
Gaussian, by number (default)    n(x) ∝ exp( −½ ((x − R)/σ)² ),   max(0, R − 4σ) ≤ x ≤ R + 4σ                 (9.2)
Log-normal, by number (median)    n(x) = exp(−½[ln(x/M)/s]²)/(x s √(2π)),   x > 0                              (9.3)
Schulz (Gamma), by number         n(x) = x^(k−1) exp(−x/θ)/(Γ(k) θ^k),   k = 1/w², θ = R w², x > 0
Gaussian, by intensity            g(x) ∝ exp( −½ ((x − R)/σ)² ),   n(x) = g(x)/V(x)²,   same range as (9.2)
P(q) = Σ_j n_j V_j² |A_j(q)|² / Σ_j n_j V_j²,     C(q) = | Σ_j n_j V_j A_j(q) |² / Σ_j n_j V_j²                  (9.4)
```

A_j is the amplitude of a particle of size x_j normalised to A(0) = 1 (section 9.4) and V_j its volume, so P(0) = 1
whatever the law. The Gaussian by number is cut at ±4σ, as in `tools/benchmarks/yoneda_truth.py`. The cut is part of
the definition: weighted by R⁶, the upper cut removes about 1.5 % of a sphere's intensity at σR/R = 0.35, more above.
The **older mean-parameterized log-normal** takes R and σR/R as the number mean and relative standard deviation. It is
integrated from 4s below its median to 4s above the median of its V²-weighted form (a log-normal of the same s with
the median times exp(p s²), V² ∝ R^p), so its largest particles, which dominate the intensity of a broad
distribution, are included. The Gaussian by intensity is Fitting's earlier distribution for spheres and random
cylinders: every size scatters the same at q = 0.

For (9.2), R and σ = R·w are the Gaussian centre and standard deviation before truncation. The reported
number mean and relative width are ⟨x⟩ and √(⟨x²⟩ − ⟨x⟩²)/⟨x⟩ of the retained population.
For (9.3), M is the median and s = SD(ln x). The number mean is M exp(s²/2) and the relative width is
√(exp(s²) − 1). For Schulz they are R and w. Zero spread gives a single size. The contact condition in
section 9.6 uses the fitted radius parameter (M for the median log-normal).
The median and log-width convention follows [SasView's distribution definitions](https://www.sasview.org/docs/user/qtgui/Perspectives/Fitting/pd/polydispersity.html#lognormal-distribution).

Manual integration resolves the size oscillations for these three distributions. For a log-normal, the
integration in ln x extends from ln M − 6.5s to ln M + p s² + 6.5s. For Schulz the bounds are the 10⁻¹⁰
quantile at shape k and the upper-tail 10⁻¹⁰ quantile at shape k+p, both at scale θ. Here p accounts for
the forward intensity (6 for sphere radius, 4 for cylinder radius, 2 for cylinder height). The quadrature
weights are normalized before computing (9.4). Independent integrations of the sphere intensity check all
three laws up to relative width 0.5 within a relative tolerance of 10⁻⁴.

**Quadrature.** Composite Gauss–Legendre with 8 nodes per panel. A form factor squared oscillates in x with
period π/q, so the panels are the number of such periods across the distribution at the largest q of the curve,
rounded up to a power of 2, at least 2 (at most 4096).

**Large q·σ in Predict and earlier models.** Averaged over sizes that vary on a scale σ (σ = R·σR/R; for the log-normal the lower
end of its intensity-weighted part), the oscillating part of |F|² is damped by about e^{−2(qσ)²}. From
q_cut = 6/σ on it is negligible, and the average is that of the envelope of |F|², its average over one oscillation:
9(1 + x²)/(2x⁶) for a sphere (for two DWBA paths (9/(2ab)) Re[h₁(a)h₁(b)*] with the spherical Hankel function h₁),
2(J₁² + Y₁²)/x² for a disc and 2/x² for the height factor of a cylinder. Between q_cut and 2·q_cut the average goes over
smoothly (a smoothstep in q) from the resolved one to the envelope's, so the model stays a smooth function of R and σ,
and the rule only has to resolve 2·q_cut. The coherent term C fades to 0 with the same step (it is damped by
e^{−(qσ)²/2}). The envelope leaves out the ringing of the hard cut at R + 4σ of the Gaussian by number, an artefact
of the cut: compared with the fully resolved average the curve differs by at most 3 · 10⁻⁵ for σR/R ≤ 0.2, 10⁻³ at
0.35 and 10⁻² at 0.5. A Gaussian by intensity keeps much weight at small R; it is always fully resolved.

**Tables for Predict and earlier models.** For sizes by number (a Gaussian or a log-normal with 0.02 ≤ σR/R ≤ 1) and one scattering
path (the effective interface, or Born at the cut's q_z), the averages come from small tables (`cut_model/tables.py`,
3.4 MB, built by `tools/ml1d/build_tables.py` and checked against a fresh build). Each form factor splits exactly into
a smooth part and a smooth amplitude times an oscillation, f(x)² = env(x) + Re[G(x) e^{iωx}] (ω = 2 for the sphere and
the disc, 1 for the height factor). Averaged over sizes, ⟨env⟩/env(x̄) and ⟨G e^{iωx̄(t−1)}⟩/G(x̄) (t = R/R̄) are smooth
in ln x̄ (respectively u = q·σ) and ln(σR/R); they are interpolated with four-point Lagrange polynomials. The width grid
has a node at σR/R = 0.25, where the Gaussian starts to be cut at R = 0. The tables carry the definition above
(resolved below q_cut, the envelope beyond) and agree with the direct quadrature to 10⁻⁴ (typically 2 · 10⁻⁵). One
evaluation of a sphere or an upright cylinder then takes about 1 ms at 750 points, whatever R and σ.

Spheres with two paths (A = c₁ f(aR) e^{−iQ₁R} + c₂ f(bR) e^{−iQ₂R}, a = √(q² + Q₁²), b = √(q² + Q₂²)) are the two Born
terms from the tables and a cross term X = ⟨f(aR) f(bR) e^{−i(Q₁−Q₂)R}⟩, P̂ = |c₁|²⟨f(aR)²⟩ + |c₂|²⟨f(bR)²⟩ + 2 Re(c₁c₂* X).
With f(x) = Re[A(x) e^{ix}], A(x) = −3/x² − 3i/x³ exactly, X is a sum of four terms e^{iωR̄} times exact powers of aR̄
and bR̄ times the moments m_k(ν) = ⟨n t^{6−k} e^{iν(t−1)}⟩/⟨n t⁶⟩ (k = 4, 5, 6, ν = ωR̄, ω = ±a ± b − (Q₁ − Q₂)), which
depend on ν and σR/R only. They are tabulated on the nodes of the direct quadrature (3.6 MB more) and interpolated in
the same way. As in the direct quadrature, the terms the envelope of the pair leaves out (ω ≈ ±(a + b) − (Q₁ − Q₂)) fade
with the smoothstep of the in-plane q, and below bR̄ = 2 X is taken on the nodes. A row comes from these tables when the
direct quadrature would need 16 panels (128 nodes) or more; on fewer panels the direct quadrature is as fast and is
kept. On 800 random rows the result differs from the direct quadrature by at most 1.4 · 10⁻³ of the curve's largest
value, and two paths are about 23 times faster. Narrower distributions (and points below x̄ = qR̄ = 2), sizes by
intensity, the DWBA, two paths in the decoupling approximation and the coherent term at q_z ≠ 0 use the direct
quadrature.

Earlier versions used fixed rectangle rules, which alias at large q·R·σ: on q ≤ 3 nm⁻¹ a sphere of R = 20 nm and
σR/R = 0.4 was off by up to 67 % (`tools/benchmarks/cut_model_checks.json`). Fits of large or broad particles to high
q therefore change with this version.

**By number and by intensity.** After a fit the results give each particle's mean radius and relative spread both
ways (*Sizes by number | by intensity*; `derived` in the exported record: `R_number_mean_nm`, `sigma_R_number_rel`,
`R_intensity_mean_nm`, `sigma_R_intensity_rel`), weighted by V² ∝ R⁶ for spheres and R⁴ for cylinders. Microscopy
counts particles; the curve is dominated by the intensity. For the log-normal the moments are exact: by intensity it
is a log-normal with the same width and the mean R (1 + (σR/R)²)^p.

The number distribution of a Gaussian by intensity is G(R)/R⁶ for spheres. It has a maximum only for
σR/R < 1/√24 ≈ 0.20 (1/√16 = 0.25 for cylinders); above that it rises towards small R, and its mean would be set by
where the distribution is cut. GIMaP gives the number mean only when the largest number weight lies inside the range
and the smallest node holds less than 10 % of it; otherwise it writes *not defined*. A broad intensity-weighted sphere
distribution therefore does not say how many small particles there are.

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

For the manual size laws the two independent size averages are tabulated on a grid linear near q = 0 and
logarithmic at large q. For broad Gamma and log-normal cylinders the oscillatory factors are integrated
on a rotated size contour where they decay. The axial Gamma factor also has an analytic characteristic function.

⟨ ⟩ are averages by number (9.4). The orientation average is taken in α on [0, π/2] with weights sin α, by composite
Gauss–Legendre with one panel per period π/(q · max(R, h/2)). In α the arguments q sin α and q cos α are smooth; in
u = cos α the radius argument q√(1 − u²) oscillates ever faster towards u = 1. The radius average depends on q sin α
only and the height average on q cos α only. For legacy size laws each is made once on a uniform 1D grid with 64
points per period of the largest size and taken at every (α, q) by four-point interpolation; with the envelope of
section 9.3 the grids and the orientation rule end at 2·q_cut, and beyond it the envelope averages are taken directly.
For one size the result agrees with adaptive quadrature to 10⁻⁵. With sizes by number the radius and height averages
on the grids come from the tables of section 9.3, and the orientation rule follows the characteristic size of the
intensity-weighted distribution. A random cylinder takes 15–200 ms to evaluate once (a broad log-normal up to 0.8 s),
much more than a sphere. Earlier versions used a 24-point rectangle rule in α, off by up to 55 % at q ≤ 3 nm⁻¹
(R = 5 nm, h = 30 nm).

**Vertical cylinder** at q_z = 0 (cylinders standing on the substrate, q in the plane; V ∝ R² at a fixed height):

```text
P_vc(q) = ⟨R⁴ (2 J_1(q R)/(q R))²⟩ / ⟨R⁴⟩,   P_vc(0) = 1                                                (9.10)
```

For independent radii and heights, the horizontal scale absorbs the axial factor in P_vc. The DA also needs the
coherent height average (section 9.5). Earlier versions did not normalise the vertical cylinder
(`P_vc = 10⁻⁶ ⟨R⁴ (2 J_1(qR)/(qR))²⟩/4`, R in nm), so its Int had other units than a sphere's; a model saved then is
converted when read (section 9.2).

In the two-path or DWBA geometry the sphere's amplitude is the sum of its paths (9.29, 9.27). Random cylinders are
averaged at |q| = √(q² + q_z²): with q_z = 0 in the effective interface and the two-path model, with the cut's vacuum
q_z in the Born and DWBA geometries of `cut_model`.

Implementation: `src/gimap/features/fitting/domain/cut_model/shapes.py`

### 9.5 Structure factor: one-dimensional paracrystal

For a component whose structure factor is switched on, with `D > 0` and `σnn/D > 0` (Hosemann & Bagchi, 1962; for
GISAXS see Renaud et al., 2009):

```text
S(q) = ( 1 − p² ) / ( 1 − 2 p cos(q D) + p² ),   p = exp( −q² σ_nn² / 2 ),   σ_nn = (σnn/D) · D             (9.12)
```

σnn/D is the relative standard deviation of the distance between neighbours, a Gaussian of mean D.
S is evaluated as `−expm1(2 ln p) / (expm1(ln p)² + 4p sin²(qD/2))`, without cancellation at small q, and
`S(0) = (σnn/D)²`. Otherwise S = 1.

Earlier versions used σD/D with `p = exp(−π q² σ_D²)`, σ_D = (σD/D)·D. This is the same function for

```text
σ_nn = √(2π) · σ_D,   i.e.   σnn/D = √(2π) · σD/D ≈ 2.507 · σD/D                                       (9.13)
```

A model saved by an earlier version is converted when read, its limits too: σD/D = 0.2 becomes σnn/D = 0.50.

D is kept at or above 2R by the fits (section 9.6).

**Coupling: LMA or DA.** With **Coupling** of a card (shown when Distance D is on) the paracrystal enters in one of
two ways:

```text
LMA:  I = P(q) · S(q)                                                                              (9.14a)
DA:   I = P(q) + C(q) (S(q) − 1)                                                                   (9.14b)
```

- LMA (the default) takes every size among neighbours of its own size. DA takes sizes and positions as independent.
- C = |⟨VA⟩|²/⟨V²⟩ is the coherent term of (9.4), the amplitude averaged by number (random cylinders also over
  orientation).
- For upright cylinders with an explicit height distribution, let G(H) = H Σₚ cₚ sinc(QₚH/2) exp(−iQₚH/2)
  on the paths of the horizontal row. Its scale absorbs ⟨|G|²⟩. The coherent radial term is therefore multiplied
  by |⟨G⟩|²/⟨|G|²⟩, averaging the complex amplitudes by number. At q_z = 0 this factor is ⟨H⟩²/⟨H²⟩.
  A single height gives a factor of one wherever the row scatters. Level 4 of Predict parameters carries a given
  height distribution into the DA. With fixed height, or with the LMA, the radial model suffices.
- With one size, DA and LMA are the same curve for spheres and vertical cylinders; randomly oriented cylinders differ,
  their orientation being a disorder of its own. With a size spread, DA keeps the part `P − C` free of the spacing:
  more intensity below the structure peak, a weaker peak. At σR/R = 0.2 and σnn/D = 0.75, the low-q intensity of
  spheres is about 1.5 times that of the LMA with a Gaussian by intensity, 1.2 times with a Gaussian by number.
- This is the part that a same-size population without a structure factor takes up in a three-population fit
  (section 9.11). On simulated films the curve tells the two couplings apart (a film made in the DA: χ²ᵣ ≈ 2.5 in DA,
  8.3 in LMA), and D and σnn/D come out within 0.3 % and 0.005 of the true values with either coupling.

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
| t_shell | t (core–shell sphere, 9.14) | nm | 2 | 0–200 | yes |
| core_contrast | ρc/ρs (core–shell sphere) | – | 2 | 0–100 | yes |
| R_hs | R_HS (hard spheres, 9.14) | nm | 1.1 × the outer radius | 0.3–1000 | yes |
| phi | φ (hard spheres) | – | 0.2 | 0–0.55 | yes |

Each component has Int, R, σR/R (h and σh/h for random cylinders; t and ρc/ρs for core–shell spheres) and, with its
structure factor (**S(q)**), the values of its **Interference** function — D and σnn/D of the 1D paracrystal (the
default) or the 2D hexagonal paracrystal, R_HS and φ of hard spheres (9.14), its **Sizes** (9.3) and,
with S(q), its **Coupling**.

**D is at least 2R.** A fit keeps the distance of a component's paracrystal at or above twice its mean radius, so
particles of the mean radius do not overlap. With R and D both free, D is searched between max(D_min, 2R) and
D_max at every R; with R fixed, D's minimum is raised to 2R; with D fixed, R's maximum is lowered to D/2. A D that
ends at 2R is reported at its lower bound (section 9.8): the particles touch, and the curve does not hold D there.
Values that are not fitted are left as they are. For a random cylinder 2R is the smallest in-plane size, so the limit
is a lower bound of its contact distance. The contact is the switch **Particles do not overlap** of the Model step (on
by default; 9.15); for core–shell spheres and hard spheres it is the inequality of 9.14.

**The beam tail in Refine and Search.** With the generic ranges of the table (A ≥ 0 without an upper limit, w
0.001–0.1 nm⁻¹, ν 1–20), a tail of huge A and a width far below the first point acts as a power law over the whole
curve, with no defined error. The tail's ranges are therefore tied to the curve where a model meets one: when a curve
is opened with the starting model (or with the ranges the page set for the previous curve), when Refine or Search
starts from the generic ranges (said in the status line; a step of Undo), and for every frame of a series whose start
model has them:

```text
0 ≤ A ≤ 10 · max(I) / k,    0.3 q_min ≤ w ≤ min(30 q_min, q_max),    1 ≤ ν ≤ 6       (over the fitted points)
```

The w window is that of Predict's prior (9.13). A starting model also takes w = q_min and ν = −(slope of the first six
points in log–log) when that slope is below −1 (else 3; at most 6); A is then solved with the scales (9.17). Ranges typed
in Model are kept, and the fit itself never changes a range (9.8). After Refine or Search, a tail in use (A above the
lower edge of its range) is flagged when A, w or ν is at an edge of its range or has no finite error. The Results step
and the report then show a warning. In a series the frame counts as **flagged**, as does a frame with any value
without a finite error; the series record lists them under `flagged`. Predict parameters keeps its own prior's ranges
(w ≤ 30 q_min, ν 1.8–2.2), and its solutions are left unchanged.

Implementation: `src/gimap/features/fitting/domain/fit_model.py`, `domain/beam_tail.py`,
`application/single_fit.py` (`tail_for_curve`)

### 9.7 Objective and the scales

**Residuals.**

```text
r_i = ( I_i − m_i ) / σ_i                         with σ                                          (9.15)
r_i = ln I_i − ln max( m_i, 10⁻³⁰⁰ )               without σ (each point the same relative weight)    (9.16)
```

m_i is the model (9.1). The fit minimises Σ r_i².

For **Refine** and **Search**, **σ from the file** uses the measured intensity uncertainty in (9.15).
If no usable σ is available, it uses (9.16). **Relative** sets σᵢ = Iᵢ and **Poisson** sets σᵢ = √Iᵢ in (9.15).
**None** uses (9.16) with equal log weights. The last three choices use only Iᵢ > 0. With positive measured
σᵢ, zero and negative intensities enter (9.15) too. With relative weights or log residuals the reduced sum of squares
measures the misfit in that metric; it is a statistical reduced χ² only with a suitable noise model.
The fit record stores both the selected weight and the residual that was used.

**Scales by bounded least squares (variable projection).** The intensity is linear in Int_i, BG and A. Those of
them that are free are eliminated at every evaluation of the model (Golub & Pereyra, 1973). With the non-linear values
set, the columns `k P_i S_i`, `1` and `k / (1 + (q/w)^ν)` form the matrix X, the fixed terms the vector m_fixed, and

```text
c = argmin_{l ≤ c ≤ u} ‖ W ( X c − (I − m_fixed) ) ‖²,   W = diag(1/σ_i), or diag(1/I_i) without σ        (9.17)
```

Solved with bounded linear least squares (`scipy.optimize.lsq_linear`, BVLS, `tol=1e-10`, at most 100 times the number of free linear terms iterations). Columns are scaled to unit norm. l and u are the Model limits, default [0, ∞). Without σ, (9.17) minimises relative residuals while the outer objective is logarithmic (9.16), so there the elimination is approximate.

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

so that with the finite-difference Jacobian a start uses about B evaluations. Refine returns the best model evaluated
during the whole run, which need not be the last iterate. "Converged" means the solver ended on a convergence
criterion.

The starting curve competes too, by the reported objective. When it beats every curve the fit tried by more than
10⁻⁶ of its Σr², the start values are returned unchanged and the result says so (`start_kept`, *did not improve on the
start values*). Such a fit is not called converged and has no errors, since no minimum was found. Without σ a start
can stay ahead this way, because the scales are solved with relative weights (9.17) while the objective is in ln I.

A manual fit (Refine, Search, a frame of an In-situ series refined) then **goes on from its own result**: the same
least squares runs again from the values it returned, as long as each run lowers χ²ᵣ by more than 10⁻³ of itself, at
most 5 times (`REFINE_RESTARTS`, `RESTART_TOLERANCE`). The result says how often it improved (`restarts` in the record
and the report, a line in the Run Log). In a long, narrow valley of χ² (values correlated near ±1) a run ends on `ftol`
with a small trust region; the next run starts again with the first steps of `x_scale`. The least squares that gives a
solution of Predict parameters its errors and χ² (Predict each frame, the chain's restart) is not run again, and
Predict's own refinement (9.13) calls the engine directly.

Refine and Search refuse a curve with no more points than free values and one without an intensity above zero. With
σ in the file, points whose σ is zero, negative or not a number are left out and counted (`sigma_invalid` among the
curve's notes); σ is dropped only when no point has one.

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
by s² = χ²_red**, so they also shrink when χ²_red is below 1 (the peak fits of chapter 7 only inflate, (7.12)).
Pairs with `abs(ρ) ≥ 0.95` are listed as strongly correlated. A non-linear parameter within 10⁻³ of either end of its
(log-)unit range, and a scale at either of its limits, is listed as at a bound; its error is not meaningful.

Implementation: `src/gimap/features/fitting/domain/fit_engine.py`

**Values at the edge of their range.** Model's fixed values and limits are hard constraints: Refine and Search never
widen a range or fix a free parameter. The values still at an edge are named under **Fit details…**
(*… is still at the … edge of its range: the curve does not determine it beyond there*).

Implementation: `src/gimap/features/fitting/domain/fit_engine.py`, `src/gimap/features/fitting/domain/fit_edges.py`,
`src/gimap/features/fitting/application/single_fit.py` (`fit_resolving`)

### 9.9 Find the particle shape (removed in 0.2.0)

Until 0.2.0 the Fit step had a fourth method, *Find the particle shape (no AI)*, the quick numerical fit that the GISAXS
automatic analysis then used. *Predict parameters* (section 9.13) replaces it in Fitting and in the automatic analysis
(section 8.5), whose solutions **Show in Fitting** takes over unchanged; `native_solution.py` still converts the quick
fit's solutions in records made before 0.2.0.

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

**Upright cylinders at a fixed exit angle.** For a cylinder standing on the surface the amplitude separates:

```text
F(q_∥, Q) = πR²H · [2J₁(q_∥R)/(q_∥R)] · sinc(QH/2) e^{−iQH/2}                                    (9.28)
```

All four terms of (9.27) share the in-plane factor 2J₁(q_∥R)/(q_∥R). For a fixed H their axial sum is constant along
the row and changes only the scale. Independent height and radius distributions preserve the radial shape of P.
In the DA, a spread of heights changes the strength of the interference through the coherent average in section 9.5.
Heights correlated with radii would also change the radial weighting (section 9.18).

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

- For the cuts in this table the EIA radius is within about 5 % up to q₊R ≈ 1.7 for spheres and ≈ 1.2 for
  hemispheres. Section 9.13 gives the distribution of errors over simulated cuts.
- Beyond that the error depends on the unknown substrate, sign included, so it cannot be corrected afterwards.
  χ²ᵣ usually shows it, though not for the hemispheres on PS in the table. On bare Si hemispheres stay within 3 % up
  to q₊R ≈ 2.0.
- D and σ_nn/D stay within 1 % in every case: the spacing comes from the structure factor.
- With α_i and α_f entered, Fitting shows q₊R of every sphere under the result, as information. Predict parameters
  fitted at q_z = 0 flags a sphere above q₊R 1.7 when it knows the cut's angles (*qzR*, section 9.13).

**Two paths with a free reflection coefficient.** In *Spheres in grazing incidence* (Model step), α_i, α_f and λ
of the cut switch spheres to the two terms that dominate:

```text
A = f(√(q²+q₊²)R) e^{−iq₊R} + r f(√(q²+q₋²)R) e^{−iq₋R},   r = |r| e^{i arg r}                       (9.29)
|A|² = f₊² + |r|² f₋² + 2|r| f₊ f₋ cos(arg r + 2k_f R)
```

r is a free effective coefficient, so the model needs the measured angles but not the substrate. Free |r| and arg r
to fit them. On the simulated cuts above it brings the radius of spheres and hemispheres with q₊R ≥ 2 back to within a
few percent on both substrates, and also on a dense particle film whose own layer changes the reflection. Its limits:

- Below q₊R ≈ 1.5 the two paths differ too little: r and R trade against each other (R off by +15 % to +38 % at
  R = 2 nm). Use the EIA there.
- Two settings of r can describe a cut equally well with radii about 5 % apart. Refine from the EIA radius, and
  compare a second start with arg r shifted by π.
- r is an effective number: it also takes up r_i, the particle layer and the phase convention. It is not the
  reflectivity of the substrate (a dense film needed |r| > 1; the default range stops at 1).
- With the angles entered and |r| fixed at 0 the sphere is the Born term at the vacuum q_z. It does no better than
  q_z = 0: on Si it is worse from R = 4 nm, because the exit-reflected path pulls the effective q_z below q₊.
- Upright cylinders of one height retain the radial shape (9.28). Random cylinders stay at |q|.

When the substrate is known, Model's **Substrate** supplies the reflection amplitudes for (9.27).
Section 9.17 gives the Fresnel coefficient of a bare interface and the Parratt coefficient of a layered substrate.

Implementation: `src/gimap/features/fitting/domain/fit_model.py` (`grazing`, `qz_plus_r`),
`src/gimap/features/fitting/domain/scattering_model.py` (`sphere_two_path_pd`)

**A measured cut: GALAXI against BornAgain's model of it.** BornAgain's example of the public GALAXI frame (BornAgain
r23, `hardcode_galaxi.py`) has Ag spheres of radius 5.75 nm, buried 18.5 nm deep in HMDSO on PTFE on Si, on a radial
paracrystal (D = 53.6 nm, disorder 10.5 nm) with size–spacing coupling κ = 17.5, in the DWBA of the layer stack. With
the example's optional log-normal sizes (median 5.75 nm, σ of ln R = 0.4), BornAgain 24.1 describes the Yoneda cut of
Analyze with one scale and one background at χ²_red = 2.1 above q = 0.3 nm⁻¹; with one size, at 22.

- Fitting describes the measured cut with a sphere and a paracrystal at χ²_red ≈ 1.1 (EIA or two paths, LMA or
  DA): R = 9.4 to 11.5 nm, σ_R/R = 0.17 to 0.35. On BornAgain's model, simulated on the same cut with the counts of
  the measurement, it finds the same: R = 8.5 to 10.0 nm, σ_R/R = 0.18 to 0.35.
- These R lie between the number mean of the log-normal (6.2 nm) and its R⁶-weighted mean (16.3 nm), as a Gaussian on
  intensity weights fitted to a broad log-normal does; σ_R/R is close to the log-normal's relative spread, 0.42.
- The cut shows no correlation peak and does not determine D: the fitted D ends anywhere from about 200 to 3000 nm.
  With D and the disorder held at BornAgain's values, χ²_red rises to 6 to 22: the paracrystal of (9.12) has no
  size–spacing coupling, which damps BornAgain's S strongly at κ = 17.5.

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
(`utils/gisaxs1d_gui_kit`) with its own model, network and least-squares fit; *Predict parameters* (section 9.13)
replaces it. Its benchmark on the held-out curves of `tools/benchmarks/recovery_cases.py`, which section 9.13 compares
with, runs with a copy of the kit in `GISAXS1D_KIT_HOME` (`tools/benchmarks/fit1d_recovery.py`, stored summary
`tools/benchmarks/history/fit1d_recovery_summary.json`; [README](../tools/benchmarks/README.md)).

### 9.13 Predict parameters

*Predict parameters* compares particle compositions on a measured cut. A network trained on simulated cuts proposes
compositions and starting values. Bounded least squares refines them. Importance samples from all retained starts
of a composition give one estimate of its evidence. Each solution receives a share of its composition's weight.
When sampling fails, the comparison uses the lowest BIC found for each composition.

**Compositions and values.** A composition is one to three populations, listed by increasing mean radius. Each is a
sphere, an upright cylinder or a random cylinder, with or without the paracrystal of section 9.5: 258 compositions in
all. Sizes are Gaussian by number (section 9.3), and the coupling is the LMA. A population has R and σR/R, h/2R and
σh/h (random cylinders), and D/2R and σnn/D (with a paracrystal). The cut adds the beam tail's width w and exponent ν
(section 9.2), the Gaussian resolution σ_q of the particle terms and the amplitude τ of a smooth multiplicative model
error exp(Σ_k b_k B_k(x)), b_k ~ N(0, τ²), over 8 cubic B-splines in x = ln(q/q_min)/ln(q_max/q_min). The network
does not propose the scales (every population's Int, the tail's A, the background): they are solved for every set of
the other values.

**What the network sees.** Everything is dimensionless: q̃ = q/q_max, lengths × q_max. The model depends on q·R only
(given q_z/q_max and σ_q/q_max), so one network serves cuts at any distance and energy. The window is the measured q
range, with ρ = q_max/q_min at most 300 and at least 10; points below the window are left out and counted. The curve
is binned on 128 logarithmic points: a bin holds the σ-weighted mean of its measured points, and a bin without points
is masked. The network reads ln I relative to its median, ln(σ/I), the mask and the position. It is also told ln ρ,
q₊/q_max and q₋/q_max and whether both angles are known (q₊ = k_i + k_f and q₋ = k_i − k_f, vacuum, encoding 4). The
training curves with both angles have two sphere paths and a free r (section 9.10). On a known substrate the same
network proposes values, and the four-path likelihood below weights them.

**The prior.** The parameter coordinates are uniform over their ranges. R̃ is log-uniform from 0.5 to 5ρ, every
population above the previous one (by stick-breaking). The other ranges: σR/R 0.02–0.5; h/2R 0.1–10, with h̃ ≤ 200
and R̃ ≤ 50 for random cylinders; D/2R 1–20; σnn/D 0.02–0.5; w from 0.3 to 30 q_min (log); ν 1.8–2.2; σ_q/q_max
10⁻⁴–10⁻² (log); τ 0.003–0.15 (log). The scales are uniform from 0 to 10 times the curve's maximum. For a sphere at
level 3 the complex reflection coefficient r is uniform in area on |r| ≤ 1. An equal-area map takes the flow's two
square coordinates onto this disk. The prior density in (Re r, Im r) is 1/π there, or 2a for the magnitude a = |r|
with an independent uniform phase. The exponent of the beam tail stays near 2, the Lorentzian wings of a beam profile.
With a free exponent, a wide tail would take over the particles' own scattering at low q.

**The network** (`src/gimap/features/fitting/domain/posterior/flow.py`):

- a 1D CNN (four stages of two convolutions and a pooling, 32–128 channels) and two dense layers give a summary h of
  the curve and its conditions;
- a head gives the probabilities of the 258 compositions from h;
- for a composition, a conditional normalising flow — ten rational-quadratic spline couplings (Durkan et al. 2019),
  10 bins on [−6, 6], conditioners of two layers of 256 — maps the values (the logit of their position in the box)
  to a standard normal. Values a composition does not have are set to 0 before every conditioner and never moved, so
  the density of the others is exact; sampling runs the couplings backwards.

It was trained by maximum likelihood (neural posterior estimation), with the cross-entropy of the head, on 5 million
simulated cuts of `cut_model` (`tools/ml1d`, `prior_v2.json`). Of these, 40 % are at the effective interface, 18 % at
the Born term of a q₊/q_max from 0 to 1 and 42 % with two paths (q₋ = q₊ × U(−0.5, 0.9), real and imaginary r drawn
from [−1, 1]²). Every population's term is computed on 8 points per bin, averaged as a measured bin is, and convolved
with the cut's resolution. Every batch draws anew the populations' shares (2.5 decades), the tail, the background,
the model error, 100–1000 measured points (linear or logarithmic in q), Poisson counts (10²–10⁷ at the highest point)
and gaps. Training took 30 epochs (2 h) on one GPU. The weights (13 MB, float32) come with GIMaP with their SHA-256
and model card. GIMaP evaluates them with numpy, so neither PyTorch nor TensorFlow is needed.

**From the proposals to the solutions** (`posterior/infer.py`, `laplace.py`, `refine.py` `predict`). The Fit step
has one path. **Particles** chooses the compositions: *The particles in Model* (their families and whether each has
the paracrystal, in the order of their radii; with an empty Model it takes the next choice), *Spheres and upright
cylinders (1–2)* or *Any particles (1–3)*.

1. The compositions: that of Model, or the head's four most probable among the choice and its most probable of every
   number of populations among them (`infer.PER_COUNT`). The head often prefers one number of populations, and a
   composition that is not tried cannot be chosen.
2. For each composition, N = 1000 samples of the flow (300 for a composition with a random cylinder). For each
   sample, the terms of cut_model are computed on the measured points (with the sample's resolution) and the scales
   by non-negative least squares. On a cut with two paths, every sphere sample has its own r. The likelihood is
   Gaussian in σ, with the model error integrated out exactly (Woodbury identity) and the scales integrated out under
   their uniform priors (Laplace for the positive ones, a half Gaussian for those at 0). The weights are L·π/q. On a
   bright curve the likelihood is far narrower than any proposal amortised over all curves: the weights sit on a few
   samples (effective sample size of order 1), which only point at the modes. Without σ in the file, σ is a relative
   error from the scatter of ln I about its local trend (the robust second difference; at least 1 %).
3. The samples fall into modes: radii and distances 10 % apart, or spreads 0.05 apart, start a new one. Least
   squares (section 9.8; σ_q free) refines the up to three heaviest modes of each composition from their heaviest
   sample. It also refines the up to three modes of the weights tempered to 50 effective samples (w^β, the largest
   β ≤ 1 that gives them) whose heaviest sample is another one. On a bright curve the weights sit on one or two
   samples, and a mode the network proposed with a slightly lower likelihood would otherwise never be refined. Every
   population is held to the prior's box: R from 0.5/q_max to 5/q_min, the spreads 0.02–0.5, D/2R and h/2R in the
   prior's ranges at the current radius, with the random-cylinder size limits and the radius order of sampling.

   On a cut with two paths, r is free in least squares (|r| ≤ 1, its phase within ±π of the start's) and goes with
   the solution's model into Model. For sphere fits with two paths the sampled |r| and phase are summarized; phase
   quantiles use the 2π branch centred on the refined phase.

   Two more kinds of start are added. The moves (`posterior/moves.py`): a population's paracrystal peak fixes its D to
   a fraction of a per cent, while its form factor fixes R only loosely. Another R at the same D, or the paracrystals
   of two populations exchanged, often describe the curve as well, and the network seldom proposes them. For the two
   best compositions (by BIC) of two or three populations, each of their two best solutions gives two more starts.
   Every population's R × 0.5 and × 2 at its D, and every pair's (D, σnn/D) exchanged (where the radius order and
   D ≥ 2R still hold), are scored by χ² with the scales solved; the best two are refined. The search: when no solution
   of one of the two best compositions (by BIC) refines below χ²/N = 2, the network's proposals have missed the curve.
   One search of the ranges (section 9.8), started from its 16 heaviest samples, is then added.

   A composition with a random cylinder gets neither the tempered modes nor the moves: its least-squares fits take
   20–30 s each. The network weights and these first least-squares fits use the counting errors. The reported χ²ᵣ is
   evaluated with those errors.
4. From here on the σ of every point includes a model error of 5 % of its intensity, σ_m² = σ² + (0.05 I)²
   (`refine.MODEL_ERROR_FLOOR`; *The model error* below). Each candidate is refitted with σ_m and ranked by
   BIC = χ²_m + k ln N, where χ²_m is the minimized sum of squared residuals in σ_m, k the number of free values
   including the scales, and N the number of fitted data points. Solutions more than 20 above the best are dropped.
   Around each of the others (at most four) the posterior is sampled by population Monte Carlo, in rounds of 400
   samples. The first round is a Student t (5 degrees of freedom) over the values the curve constrains. Its scale
   comes from the Gauss–Newton curvature of the residuals plus the prior's, inflated by 1.4 (and by √(χ²/N) on a
   curve the model does not describe within σ). The other values (an absent tail, a resolution at its bound, τ) come
   from piecewise densities that a scan of the likelihood along each sets. Each later round is a Student t over all
   values together, the scanned ones and τ included, with the mean and the covariance (inflated by 1.4) of all samples
   so far, their weights tempered to two effective samples per value. It follows a posterior that lies away from the
   least-squares optimum, or along a direction the first round has no width in (a spread at its bound that trades
   against the radius). Up to eight rounds run (three for a composition with a random cylinder), until 200 samples
   are effective at that start.
5. Pool all the draws for each composition c. Index the sampling rounds over *every* retained start by j.
   Proposal qⱼ generated mⱼ draws, and M = Σⱼ mⱼ. In the sampler coordinates z, use the same mixture density
   at every draw (deterministic-mixture multiple importance sampling).

   ```text
   q_mix(z) = Σⱼ (mⱼ/M) qⱼ(z)
   wᵢ = L(data | zᵢ, c) π(zᵢ | c) / q_mix(zᵢ)
   Z_c = (1/M) Σᵢ wᵢ                  P(c | data) = Z_c / Σ_b Z_b
   ```

   The retained compositions have equal prior probabilities. Assign each draw to the start that generated it.
   The share of solution s within c is Σᵢ∈s wᵢ / Σᵢ wᵢ. Its reported weight is that share times P(c | data).
   When starts cover the same peak, their individual shares depend on the proposals and still sum to their
   combined share. Conditional quantiles use the draws assigned to s.

**BIC fallback.** If any retained composition has fewer than 10 effective pooled samples or a retained solution has
no local sampler, all compositions receive BIC weights. For each composition take B_c = minₛ BICₛ over its
solutions. Normalize exp(−B_c/2) across compositions. Within c, normalize exp[−(BICₛ − B_c)/2] across its
solutions to divide that weight. Finding the same optimum twice leaves B_c unchanged. On this branch the flag
is *bic* and the reported intervals use the least-squares errors with σ_m (section 9.8).

**What the reported numbers measure.** The composition weight includes every retained solution, also those below the
display cutoff. The record names the comparison and the interval source separately. Intermediate results pool the
starts sampled so far and normalize over the compositions available at that point. The evidence uses the original
weights, and the pooled ESS = (Σw)²/Σw² governs the composition comparison. A solution with a conditional ESS below
50 takes its sampled quantiles from tempered weights. Beside the least-squares value and its reported interval,
**posterior median** gives the weighted 50th percentile with the original 16th and 84th percentiles, before the
interval is extended to the optimum. These entries are empty where sampling gave no local posterior. Scales and
background are solved for each sample and have no sampled marginal.

The Monte Carlo range of a composition's probability comes from a bootstrap. In each replicate, resample mⱼ
corrected weights with replacement within each proposal round j, compute Z*_c as their pooled mean and normalize the
Z*_c across compositions. Repeat 200 times with seed 7529. The 16th and 84th percentiles of the resulting
probabilities give the Monte Carlo range; the sample standard deviation of log Z*_c estimates the Monte Carlo error.
Both hold for the proposals observed: they leave out missed modes and other sets of adaptive proposals. Where a local
sampler is missing, the record names the network estimate using counting errors. BIC weights have no evidence-based
probability range.

**Solutions and flags.** Every solution with a normalized weight of at least 1 % is returned, in descending order.
Each carries the refined values with their least-squares errors (9.8), the posterior median and intervals, χ²ᵣ, its
probability and flags. The most probable goes into Model, the others to **Other solutions**. The displayed fit
normally uses the counting errors. When it falls outside the sampled 95 % limits and a refit with σ_m puts more of
its population parameters within those limits, the refit is selected and flagged *model_error*. An interval always
holds the displayed value: it is extended to the displayed optimum. A value the curve does not see at all (the shape
of a population whose amplitude is 0) has no least-squares error. The flags:

- a value at a bound of its range;
- *unconstrained*: its 68 % interval covers more than 0.7 of the prior's range;
- strongly correlated pairs (beyond ±0.95);
- few effective samples (below 50): the intervals are then taken with the weights raised to the largest power β ≤ 1
  that gives 50 effective samples, which moves them towards the wider proposal;
- *bic*: the composition weights are BIC weights (above);
- *large* and *small* for a population whose size the measured q range does not show. *large* (q_min·R ≥ 0.7): its
  Guinier region lies at or below the lowest q, and the decay the curve shows is also what roughness or a rise at low
  q produce; on simulated curves with roughness, such an extra population sometimes takes up the roughness. *small*
  (q_max·R < 1): it acts as a smooth background;
- *model_error*: the displayed fit is the refit with σ_m;
- *qzR* and the other flags of the substrate levels (below).

Predict searches its prior range for each proposed parameter and keeps supplied level-4 upright-cylinder heights and
widths fixed. **Refine the values in Model** afterwards uses Model's fixed values and ranges.

Time per curve on one core (Intel Core i9-13900HX): about 6 s with the composition given and about 26 s choosing
among one or two populations (`tools/benchmarks/posterior_timing_summary.json`). Choosing among one to three spheres
and upright cylinders takes roughly 30–70 s, a few minutes when random cylinders are among the choice, and about 2.4
times as long with both angles (two paths).

Steps 2–4 use Model's coupling (DA) and legacy size law (the log-normal parameterized by its mean). For the other
number distributions of manual fitting, Predict uses its Gaussian, except for upright cylinders given a height
distribution at level 4. The network, trained on the LMA and Gaussian sizes, then only proposes, and the weights
correct it.

**Substrate levels and the flag *qzR*.** The information available about the cut determines its scattering model.

1. the effective interface, the form factor at q_z = 0 (the default; no angle needed);
2. the Born term at the cut's vacuum q₊ = k(sin α_i + sin α_f) (q_z alone: *Cut qz* of the 1D Predict window);
3. the two paths of spheres, the Born term and the exit-reflected path with a free r (both angles: α_i, α_f and λ
   under Model ▸ *Spheres in grazing incidence* on the Fit page), for an unknown substrate.
4. the four DWBA paths of equation (9.40) with α_i, α_f, λ and a known substrate. For each proposed sphere radius,
   the form factor uses P = ⟨|A|²⟩/⟨V²⟩ and, for DA, C = |⟨A⟩|²/⟨V²⟩, where A is the coherent sum of the four
   volume amplitudes at that radius. The fixed r_i and r_f come from (9.41) for a bare interface or (9.41a) for
   a stack. Refinement, amplitude fitting and posterior sampling use this geometry. The resulting intervals are
   conditional on the angles and the supplied optical constants and thicknesses.

At level 4 the proposal still receives q₊/q_max and q₋/q_max. Its two sampled coordinates for the free r enter
only the proposal correction. The likelihood is constant in these coordinates and their normalized prior integrates
to one. Local posterior sampling omits both coordinates and their prior factors.

For upright cylinders the horizontal scale absorbs the common axial intensity. At level 4 a given independent height
distribution multiplies the DA coherent term by |⟨G(H)⟩|²/⟨|G(H)|²⟩ (section 9.5). The ratio is one for fixed height.
For **The particles in Model**, the prediction record keeps each upright cylinder's supplied height and width together
with its size law and LMA or DA setting. For example, a lognormal specified by its median keeps the standard deviation
of ln(height) as its width. Refinement keeps these inputs fixed and preserves the order of populations by radius.
The posterior check reads the same inputs from the saved solution.
At all levels the likelihood and refinement apply the same Gaussian resolution to the particle terms.
Widths σ_q up to one quarter of the smallest positive spacing between fitted q values are treated as sharp.
For larger widths the two stages use the same convolution table over the fitted q range.
For randomly oriented cylinders, sampling and refinement both average the form factor at |q| = q at
levels 1 and 3, and at |q| = √(q² + q₊²) at levels 2 and 4 (q is the in-plane magnitude).
Level 4 includes only the direct Born path for these cylinders. It omits their reflected-wave interference.

For spheres at level 1, the median radius error stays within about 5 % up to q₊R ≈ 1.7; single fits can exceed it.
Above q₊R 1.7, on simulated four-term DWBA cuts on Si, the median error of R is 10–23 % at level 1, 4–34 % at level 2
and 4–8 % at level 3. DWBA programs such as BornAgain and IsGISAXS evaluate the terms with the substrate's refractive
index and need no such approximation, so GIMaP says when it matters:

- A solution fitted at level 1 whose sphere or random cylinder has q₊R above 1.7 (R its radius, the populations
  numbered as for *large*) gets the flag *qzR* (`refine.interface_flags`, `refine.QZR_LIMIT`): *R (n): q_z·R is
  above 1.7 at the cut's q_z, where the form factor at q_z = 0 (the default effective interface) biases the radius,
  usually too small; give both angles of the cut (αi, αf) so that spheres are fitted with the exit-reflected path
  (random cylinders stay at q_z = 0)*. This flag applies only at level 1.
- Automatic analysis uses level 4 when it knows the substrate, α_i, λ and the cut's Yoneda exit angle α_f. With the
  same geometry and an unknown substrate it uses level 2. For a cut away from the Yoneda band or without α_i, the fit
  uses level 1 and has no measured q₊ for these flags. The report records the level and the source of q₊ (section
  14.10).
- The Fit page and the 1D Predict window take the level from the angles they are given. A curve file does not carry
  the angles of its cut. At level 1 the measured q₊ can be entered separately for the flags on reflected-beam data;
  the fit still evaluates the effective interface at q_z = 0.
- For a sphere or random cylinder on a reflected-beam cut fitted at level 1 or 2, `qzR_cov:R_n` marks the intervals
  as not calibrated above q₊R = 1.2: on simulated DWBA cuts at q₊R 1.2–1.7, the 68 % intervals held the true radius
  in only 20–24 % of cases. Level 2 adds `qzR_fail:R_n` above 2, where its model error becomes strong: use level 3 or
  a four-term DWBA fit. Level 3 raises `qzR_cov:R_n` above q₊R = 1.7. These sphere thresholds also apply to random
  cylinders, as a precaution.
- At levels 2–4, an upright cylinder with fitted q₊R > 2 receives `cyl_sphere:R_n`: its radial profile can fit the
  scattering of spheres of a similar radius, also at level 4 with the known substrate (see the
  [paired benchmark results](../tools/benchmarks/README.md)). The horizontal cut alone leaves the cylinder height
  undetermined. At any level, a random or upright cylinder with a known height also receives this flag when
  q₊R > 1.2 and 0.8 ≤ h/(2R) ≤ 1.25.
- At level 3, a curve without a reflected wave can be described as well by a sphere of about half the radius with a
  strong r. At the Yoneda angle of the substrate |r_f| is near 1 and the reflected wave strong; on a cut well above
  the critical angle it is weak, and a solution with a large |r| and a small sphere there is suspect. Fix `refl_amp`
  to the substrate's |r_f|, or to 0, and refine. With broad sizes and α_f at the critical angle (the public GALAXI
  cut), the free r can also take up what the sizes and the paracrystal should describe. Check such solutions, which
  carry flags (*at bound*, *large*, *unconstrained*), against what is known of the sample.
- The Born-approximation flags apply to levels 1–3. Level 4 already includes the reflected sphere amplitudes.
- Predict assigns the flags after fitting. The fitted values, intervals and probabilities depend on the substrate
  level used.

**The model error.** A model error per point says how well the model can describe a measured cut. Without it, the
68 % intervals were too narrow on simulated curves that Fitting's model describes closely but not exactly: the values
the curve fixes sharply, above all D and σnn/D, moved by more than the counting errors allow. 5 % is the smallest
floor at which the 68 % intervals held the truth in at least 68 % of the cases on simulated development curves, both
in Fitting's own model and with a beam tail, roughness and resolution. On the benchmark curves with these effects the
intervals of R hold the truth less often (*Benchmark* below). Steps 1–3 keep the counting errors, because the floor there led some fits into
other modes. On curves that the model describes exactly, the 68 % intervals are therefore conservative.

The floor was calibrated on a **linear grid** of 397 points from 0.02 to 2 nm⁻¹ (Δq = 0.005 nm⁻¹; on average 198.5
points per decade). To compare the sampling at a feature, the report gives the number of points in the fitted window
and Δq at the measured intensity maximum, in nm⁻¹ and as a multiple of 0.005 nm⁻¹. This Δq is the mean of the gaps to
the nearest distinct |q| on each side, or the one gap at an endpoint. A log-binned cut can have several times the
validation spacing near its maximum. Below 80 fitted points, check that the features are resolved. Above 400,
importance sampling may fall back to a BIC comparison: rebin to 150–250 log-spaced points.

**Benchmark.** On 200 simulated cuts that the network was not trained on, half in Fitting's own model and half with
a resolution, a beam tail and roughness, the most probable solution has the true composition in 83 % and 65 % of the
cuts. With the composition given, the median error of R is 0.4 % and 1.9 %, and the 68 % intervals hold the true R in
79 % and 45 % of the cases. The head alone often ranks another composition first: many compositions describe a cut
about equally well, and the likelihood decides among the head's most probable. The runs, their summaries and the
scripts are in `tools/benchmarks/` ([README](../tools/benchmarks/README.md)).

Implementation: `src/gimap/features/fitting/domain/posterior/`, `application/single_predict.py`,
`infrastructure/adapters/posterior_job.py`; training and checks: `tools/ml1d/`; benchmarks:
`tools/benchmarks/posterior_recovery.py`, `posterior_pairs.py`, `summary_tables.py`, `v5_recovery.py`,
`galaxi_posterior.py`.

### 9.14 Models for manual fitting

Refine and Search also fit the models below. Predict parameters does not propose them: it knows only the families it
was trained on (sphere, random cylinder, vertical cylinder, each with or without the 1D paracrystal, 9.13). In the
Model step they are the family **Core–shell sphere** and, with **S(q)** on, the **Interference** choices
**2D hexagonal paracrystal** and **Hard spheres (Percus–Yevick)**. The LMA and the DA (9.5) apply to every
interference function. Each is unit-scale like the others: a form factor is 1 at q = 0, a structure factor → 1 at
large q.

**Core–shell sphere.** A core of radius R (its size distribution as chosen in **Sizes**, 9.3) inside a shell of fixed
thickness t; the core's scattering contrast is ρc/ρs times the shell's (0: a hollow shell; 1: a homogeneous sphere of
radius R + t):

```text
A(q; R) = R_s³ f(q R_s) + (ρc/ρs − 1) R³ f(q R),   R_s = R + t,   f(x) = 3 (sin x − x cos x)/x³          (9.32)
P̂(q) = ⟨A²⟩ / ⟨A(0)²⟩,    C(q) = ⟨A⟩² / ⟨A(0)²⟩      (number averages over R)                          (9.33)
```

This is SasView's `core_shell_sphere` with ρc/ρs = (ρ_core − ρ_solvent)/(ρ_shell − ρ_solvent) (Guinier & Fournet,
1955). It is evaluated at q_z = 0 (the effective interface; the angles of *Spheres in grazing incidence* do not apply).
The quadrature over R is that of 9.3 (Gauss–Legendre panels following q_max times the width of the distribution),
without the envelope at large q. With *Gaussian, by intensity* the intensity weight of a core radius is taken as R⁶
(the core's V²), an approximation: V² of a core–shell particle depends on t and ρc/ρs too.

**Hard spheres (Percus–Yevick).** The structure factor of hard spheres of radius R_HS at volume fraction φ in the
Percus–Yevick closure (Percus & Yevick, 1958; solved by Wertheim, 1963, and Thiele, 1963), in the closed form of
Kinning & Thomas (1984) that SasView's `hardsphere` uses:

```text
S(q) = 1 / [ 1 + 24 φ I(x) ],   x = 2 q R_HS                                                              (9.34)
I(x) = ∫₀¹ (α s² + β s³ + γ s⁵) sin(x s)/(x s) ds
     = α (sin x − x cos x)/x³ + β (2x sin x + (2 − x²) cos x − 2)/x⁴
       + γ (−x⁴ cos x + 4 [(3x² − 6) cos x + (x³ − 6x) sin x + 6])/x⁶                                     (9.35)
α = (1 + 2φ)²/(1 − φ)⁴,   β = −6φ (1 + φ/2)²/(1 − φ)⁴,   γ = φ α / 2,   S(0) = (1 − φ)⁴/(1 + 2φ)²
```

Below x = 1 the integral is summed as its Taylor series (14 terms), which avoids the cancellation of (9.35). A new
R_HS is 1.1 × the particle's outer radius; φ is limited to 0.55 (the Percus–Yevick closure is reliable up to about 0.5).
For particles on a surface, S of a three-dimensional hard-sphere liquid is an approximation of their arrangement in the
plane; hard disks in 2D are not in the library.

**2D hexagonal paracrystal.** A hexagonal lattice of nearest-neighbour distance D (lattice angle 120°) whose
neighbours are displaced by an isotropic Gaussian of standard deviation ω = σ·D in each in-plane direction (σ is the
model's σnn/D), with infinite domains, averaged over the in-plane orientation ξ of the domains (Hosemann & Bagchi,
1962; Renaud, Lazzari & Leroy, 2009, Surf. Sci. Rep. 64, 255; BornAgain's `Paracrystal2D` with integration over ξ,
Pospelov et al., 2020, J. Appl. Cryst. 53, 262):

```text
S(q) = (1/π) ∫₀^π s(q D cos ξ) · s(q D cos(ξ + 2π/3)) dξ                                                  (9.36)
s(θ) = Re[ (1 + φ)/(1 − φ) ] = (1 − p²)/(1 − 2p cos θ + p²),   φ = p e^{iθ},   p = exp(−q² ω²/2)              (9.37)
```

s is evaluated as in (9.12), without cancellation near p = 1. The integrand has period π in ξ and is integrated by the
midpoint rule with 3 points per half-width (1 − p)/(q D) of its narrowest peak (a power of 2 per q, 64 to 2¹⁸ points;
the error of the rule falls as e^(−2π·3)). As in BornAgain and IsGISAXS, this product form rises as 1/(q D) towards
q = 0; fit windows with q D ≳ 1.

**Contact.** With **Particles do not overlap** on (the default, 9.15) and both values fitted, the paracrystal's D is
at least 2(R + t) for a core–shell sphere, and R_HS at least the outer radius R (+ t). Both act as inequalities
(9.15). The other families keep D ≥ 2R (9.6).

**Checks** (`tests/test_fit_model_library.py`):

| Model | Against | Agreement (relative) |
|---|---|---|
| Hard spheres, φ = 0.05–0.5 | the 3D Fourier transform of Wertheim's direct correlation function, by adaptive quadrature | 10⁻⁸; S(0) = (1 − φ)⁴/(1 + 2φ)² |
| Hard spheres | SasView's closed form (Kinning & Thomas) | 10⁻⁹ |
| Core–shell, one size | SasView's amplitude with scattering-length densities | 10⁻⁹ |
| Core–shell, Gaussian of R (σR/R 0.1) | the same, averaged by a 4001-point rule | 2 · 10⁻⁵ (P̂ and C) |
| Core–shell, ρc/ρs = 1 and t = 0 | `cut_model`'s sphere of R + t and of R | 10⁻⁹ |
| 2D paracrystal, D 12–35 nm, σ 0.05–0.25 | BornAgain's formula, complex, by adaptive quadrature | 10⁻⁶ |
| 2D paracrystal, D 20 nm, σ 0.1 | a BornAgain 24.1 simulation (Born approximation, particles of 0.5 nm, divided by a dilute layout and by the lattice's density) | 2 · 10⁻³ (skipped without BornAgain) |

Against BornAgain simulations over D = 12–35 nm and σ = 0.04–0.25 the largest deviation was 10⁻⁵–10⁻² (median
10⁻⁶–10⁻³), the largest at the sharp peaks of the smallest σ, where BornAgain's adaptive integration is the coarser.

Implementation: `src/gimap/features/fitting/domain/model_library.py`

### 9.15 Links and constraints between parameters

**Names.** Every parameter has a name: a component's `<key>_<n>` (`R_1`, `sigma_R_2`, `D_1`, `t_shell_1`, `R_hs_2`;
n from 1), a global one its key (`background`, `res_width`). The tooltip of a parameter's name in Model gives it.

**Links.** A parameter can be linked (its **=** button) to another (**Same as**) or to an expression of others
(**Expression…**): `sigma_R_1` shares σR/R of population 1, `2*R_1` sets R₂ = 2R₁, `R_1 + t_shell_1`… Expressions are
numbers, names, + − * / **, parentheses, the functions sqrt, exp, log, log10, sin, cos, tan, abs, min and max, and the
constants pi and e. They are parsed with Python's `ast` and evaluated by this fixed set of operations, never by `eval`
(at most 200 characters; an exponent above 1000 is refused). A linked parameter is not fitted: at every evaluation
the free values are set and then the links in the order of their dependencies (a circle of links is refused). A
linked scale (Int, BG, A) must be linear in the fitted scales it reads (`8*Int_1`, `Int_1*(R_1/R_2)**3`): its term
joins their columns in (9.17), so the scales are still solved exactly; a size linked to a fitted scale is refused. The
error of a linked value comes from the covariance (9.25):

```text
σ_link² = gᵀ C g,    g_k = ∂(link)/∂x_k   (central differences, step 10⁻⁴ × the size of x_k)               (9.38)
```

A linked parameter's own range is not enforced.

**Inequalities.** A constraint `NAME >= expression` or `NAME <= expression` (**Add Constraint…** in the card *Links and
constraints*) keeps a fitted non-linear value on one side of an expression of the others, e.g. `D_2 >= 2*R_2 + 3`. It
acts by a change of variables like the contact D ≥ 2R (9.6): with the other values set, the value is searched between
max(l, bound) and u (≥), or between l and min(u, bound) (≤), on a log scale when its range is; constraints act in the
order listed, after the contact. A constraint on a fixed value or a scale does not act (it stays in the model). A
value that ends at its bound is reported at a bound (9.8). Clearing **Particles do not overlap** switches the contact
of 9.6 and 9.14 off, for models where touching or interpenetrating particles are meant.

Links, constraints and the contact switch are saved with the model (a parameter's `tie`, `constraints`, `contact`;
written only when used) and in the fit's record (`links`). Removing a component drops the links and constraints that
read it and renumbers the names of the later ones. Predict parameters does not use links: its proposals are complete
solutions of the families it knows.

Implementation: `src/gimap/features/fitting/domain/fit_links.py`, `linked_fit.py`, `fit_engine.py`

### 9.16 Uncertainty after Refine and Search

**Correlation matrix.** The covariance C of (9.25) is kept with the fit (its parameter order with it) and shown in
**Results ▸ Uncertainty and Report…** as the correlation matrix ρ_kl (9.26): a heat map with the values (blue −1, white
0, red +1; grey for a direction the curve does not see, whose error is not defined), beside every free and linked value
with its 1σ error and relative error. The pairs with |ρ| > 0.9 are listed; the Results step's own warning uses 0.95.
The record of **Save Data and Fit…** holds the same (`fit.correlation`, `fit.covariance`, `fit.correlated_above_0.9`).

**Values not determined.** A free value whose 1σ error is not finite, or — for sizes, spreads, distances and the
tail's shape — larger than 10 times the value, is *not determined by the curve*: ⚠ in the Results table and the report,
`fit.undetermined` in the record. Not counted: values at a bound (said as such), a scale at 0, and the values of a term
whose scale is 0.

**χ² profile.** For one parameter x: x fixed at each value of a grid (by default 15 values over ±3σ around the fitted
value, inside its range; its range when σ is not known, on a log grid when the range spans a decade or more), the
other free values fitted again by Refine, from the neighbouring grid point outwards from the best fit, and

```text
Δχ²(x) = [ Σr²(x) − Σr²_best ] / s²,    s² = Σr²_best / (N − p)                                            (9.39)
```

The 68 % interval is where Δχ² ≤ 1 (linear interpolation between grid points; *open* on a side where the grid ends
first). With χ²_red = 1 this is the usual Δχ² = 1 rule. In units of s², the covariance errors scaled by χ²_red (9.8)
are the linear approximation of this interval: the two agree for a well-determined value, and the profile shows where
they do not (an asymmetric or open interval). A grid point with Δχ² < 0 means the fit had not reached its minimum:
Refine again. The profile runs in the background; **Stop** keeps the points done.

**Fit report.** Markdown, HTML or plain text (**Copy**, **Save Report…**), made from the same record as **Save Data
and Fit…**: the data file with its SHA-256, the points, their q range, halves and weighting; the model (families,
sizes, interference function and coupling, links, constraints, contact) and its equation; the method and its outcome;
χ²_red over the points and free values, and log RMSE; every parameter with value, ± 1σ, status (fitted, fixed,
linked, at a bound, not determined) and range; the pairs with |ρ| > 0.9; and the warnings.

Implementation: `src/gimap/features/fitting/domain/fit_profile.py`, `application/fit_uncertainty.py`,
`application/fit_report.py`

### 9.17 The four-term DWBA with a known substrate

**What it is.** A fourth level beside the three of section 9.10 (q_z = 0; the Born term at the cut's q_z; two paths
with a free effective r): when the substrate is known, Model's **Substrate** (card *Spheres in grazing incidence*)
takes a material of a small table or δ and β typed, and with αi and αf of the cut the spheres are evaluated in the
four-term DWBA of a particle on that substrate (Renaud, Lazzari & Leroy, 2009; IsGISAXS, Lazzari, 2002):

```text
A(q_∥) = F(q_∥, k_i+k_f) + r_i F(q_∥, k_f−k_i) + r_f F(q_∥, k_i−k_f) + r_i r_f F(q_∥, −k_i−k_f)              (9.40)
r(α) = (k_z − k_z,s) / (k_z + k_z,s),   k_z = k sin α,   k_z,s = k √(sin²α − 2δ + 2iβ)                       (9.41)
```

The four terms in (9.40) use the reflection amplitudes at αi and αf. Equation (9.41) gives the bare-interface
amplitude and (9.41a) below gives it for a layered substrate. Both coefficients are fixed by the optical inputs.
Random cylinders are averaged at |q| = √(q² + (k_i + k_f)²) using only the direct Born path.
The likelihood and refinement both omit their reflected-wave interference. Upright cylinders with a height
distribution use these four paths for the horizontal DA (sections 9.5, 9.10). Core–shell spheres stay at q_z = 0.
The size average uses the quadrature of section 9.3. For batches of resolved sphere intensities, six-point
interpolation in radius lets the size distributions share one table. On a densely sampled cut, a single
distribution can be evaluated on a grid in q and interpolated with eight points. The largest lateral separation
in a sphere of radius R is 2R, so the q step is at most 0.32/(2R_max) for the largest quadrature radius R_max.
These interpolations precede multiplication by S(q) and resolution smearing. The line
under the substrate says which level is in use (*Level: Four-term DWBA, substrate known on Si (δ = …, β = … at λ =
… nm)*). With an incomplete pair of angles, the line explains that level 4 needs both αi and αf.

**The table.** δ and β of a material of density ρ and mass fractions w_j of elements of atomic weight A_j and atomic
scattering factors f₁ + i f₂:

```text
δ + iβ = (r_e λ² / 2π) Σ_j n_j (f₁,j + i f₂,j),    n_j = ρ N_A w_j / A_j                                      (9.42)
```

f₁ and f₂ are those of Henke, Gullikson & Davis (1993) as CXRO tabulates them, every fourth point from 4.9 to 30 keV
and every point between the L edges of gold; f₁ is interpolated linearly in ln E, f₂ linearly in ln f₂ against ln E
(within 0.15 % of f₁ and 0.35 % of f₂ of the full tables). Outside 4.9–30 keV the table refuses and δ, β are typed.

| Material | ρ (g/cm³) | Composition | δ, β at 8.048 keV |
|---|---|---|---|
| Si | 2.33 | Si | 7.58 · 10⁻⁶, 1.73 · 10⁻⁷ |
| SiO₂ | 2.20 | SiO₂ (thermal oxide, fused silica) | 7.12 · 10⁻⁶, 9.2 · 10⁻⁸ |
| Glass (plate) | 2.40 | NIST's “Glass, plate”: O 0.4598, Na 0.0964, Si 0.3366, Ca 0.1072 by mass | 7.74 · 10⁻⁶, 1.37 · 10⁻⁷ |
| Au | 19.32 | Au | 4.71 · 10⁻⁵, 4.85 · 10⁻⁶ |

The values of Si, SiO₂ and Au agree with CXRO's to 1 % (δ) and 2 % (β) (`tests/test_joint_dwba.py`).

**A film on the substrate.** Each finite layer has a thickness d in nm and its own δ and β at the cut's energy.
Index the media from air (0) through the layers (1 … m) to the semi-infinite substrate (m+1). With
p_j = k√(sin²α − 2δ_j + 2iβ_j) on the decaying branch and f_j = (p_j − p_{j+1})/(p_j + p_{j+1}), the
Parratt recursion gives the reflection amplitude at the upper surface.

```text
R_m = f_m
u_j = R_{j+1} exp(2i p_{j+1} d_{j+1})
R_j = (f_j + u_j)/(1 + f_j u_j),    j = m−1, …, 0                                              (9.41a)
```

Insert R₀(α_i) and R₀(α_f) in (9.40). For m = 0 this gives the bare Fresnel amplitude (9.41).
The implementation propagates the field and its normal derivative, an equivalent form that also handles a zero
normal wavevector. Independent simultaneous matching at all interfaces checks the complex amplitude below, near
and above the critical angles for one and two layers (`tests/test_layered_substrate.py`). The interfaces are
smooth and the particle layer has the optical constants of air. The fit takes the layer thicknesses and optical
constants as given: a horizontal cut does not determine them.

**Predict parameters.** The fourth level uses these same coefficients in the likelihood and bounded refinement
(section 9.13). The fitted model records the substrate and any layers for subsequent evaluation.
[Benchmark results](../tools/benchmarks/README.md) compare simulated cuts on Si and on a 100 nm polystyrene film on Si.

**Checked against BornAgain 24.1** (`tools/benchmarks/joint_dwba_bornagain.py`, average layer materials off as in
section 9.2; Si of the table at λ = 0.1045 nm, δ = 3.46 · 10⁻⁶, β = 3.74 · 10⁻⁸; αi = 0.40°, the horizontal cut at the
Yoneda row αf = 0.151°; Fitting's model with the substrate known; q = 0.05–2.5 nm⁻¹; after one scale):

| Case | Largest deviation | Median |
|---|---|---|
| sphere R = 4 nm, one size | 2.6 · 10⁻² (at the zero near q = 1.90 nm⁻¹) | 1.0 · 10⁻³ |
| spheres σR/R = 0.2 by number (BornAgain: the 24 radii and weights of the quadrature) | 3.9 · 10⁻⁴ | 1.5 · 10⁻⁴ |
| sphere, one size, radial paracrystal D = 14 nm, σnn = 2.8 nm | 2.6 · 10⁻² (at the zero) | 1.0 · 10⁻³ |
| upright cylinder R = 4 nm, H = 6 nm, one size | at the zeros only | 1.2 · 10⁻³ |

This is the level the checks of `cut_model` reached (section 9.2).

Implementation: `src/gimap/features/fitting/domain/substrates.py`, `domain/fit_model.py` (`grazing`),
`application/substrate_level.py`, `presentation/single/substrate_row.py`

### 9.18 Several curves fitted together; the vertical cut

**Fitting ▸ Fit Several Curves Together…** fits a list of curves with one χ² (SasView's simultaneous fit; IsGISAXS's
joint fit of a q_y and a q_z cut of the same frame). Every curve c has its own model: the components of Model with the
same values, freedom and ranges, its own background, beam tail, factor k and cut geometry. A free non-linear value is
**shared** (fitted once; by default every particle value: R, σR/R, h, σh/h, D, σnn/D …) or each curve's own (by default
the tail's width and exponent). The scales (Int of every population, background, tail A) are always each curve's own
and solved exactly for that curve at every step ((9.17), section 9.7). With several populations, their ratio is not
tied between the curves.

```text
χ² = Σ_c Σ_i r_c,i²(θ_shared, θ_c, scales_c),   r as (9.15) or (9.16) for each curve                            (9.43)
χ²_red = χ² / (N − p),   N = Σ_c N_c,   p = shared + per-curve non-linear values + every curve's free scales
```

Refine and Search are those of section 9.8 on the joint vector of the free non-linear values; the errors come from
the Jacobian of all residuals with respect to every free value (shared, per curve and the scales), (9.24)–(9.26). The
summary gives χ²_red and every curve's Σr²/N, and marks curves weighted differently (some by σ, some by ln I) as not
comparable. For a vertical cut with file or Poisson standard deviations, the report also gives its reduced
χ² = Σᵢr_c,i² / [N_c − tr(H_cc)]. Here H = JJ⁺ projects onto the columns of the weighted joint Jacobian J. The
diagonal block H_cc assigns the cut its share of the fitted parameters, including shared ones.

**Kinds of curve.** A *horizontal cut* (I(|q_y|) at one exit angle; both halves of a signed cut on |q|) has αi, αf and
λ: with both angles and a substrate chosen in the window it is in the four-term DWBA (section 9.17), without angles at
q_z = 0, otherwise at the level of Model. *Other curve*: q_z = 0. A *vertical cut* (I(q_z) at one in-plane q) has αi,
λ, its q_∥ and needs the substrate. In a vertical cut every point is an exit angle,

```text
sin α_f = q_z/k − sin α_i                                         (vacuum q_z)                                   (9.44)
sin² α_f = (q_z/k − Re √(sin²α_i − sin²α_c))² + sin²α_c            (q_z refracted in a film of critical angle α_c)
```

with its own k_f = k sin α_f and r_f = r(α_f) in the four terms (9.40). **Lowest exit angle of the vertical cut**
retains valid points with α_f ≥ m α_c,substrate, where α_c,substrate = √(2δ) radians at the cut's wavelength.
The default m = 2 excludes the region around the substrate's critical angle. Set m = 0 to keep all valid points
above the horizon. For a refracted q_z the film critical angle enters (9.44), then the substrate threshold is applied
to the recovered α_f. The record and report give the selected multiple and its angle in degrees.

If Model has no height for an upright-cylinder population and m > 0, Refine compares starting heights of 10, 5
and 20 nm on the retained points. For each trial it solves the free intensity scales and uses the joint residual
sum of squares to choose the start. Populations are checked in model order. A supplied height starts at its
value in Model. The record includes the starting model of each curve.

For particles standing on the substrate (base at z = 0):

```text
sphere            F(q_∥, Q) = V f(√(q_∥² + Q²) R) e^{−iQR}                                                       (9.45)
upright cylinder  F(q_∥, Q) = V [2 J₁(q_∥ R)/(q_∥ R)] sinc(QH/2) e^{−iQH/2}
P̂(α_f) = ⟨V²|A|²⟩ / ⟨V²⟩,   C(α_f) = |⟨V A⟩|² / ⟨V²⟩,   T = P̂ S(q_∥) (LMA) or P̂ + C (S(q_∥) − 1) (DA)
```

averaged over the sizes by number with R and H independent (composite Gauss–Legendre, panels following the largest
√(q_∥² + Q²) of the cut). The structure factor is that of the cut's in-plane q, the same at every point: in the LMA it
goes into the scale, and a vertical cut holds D and σnn/D only through the horizontal cut they are shared with. When
a vertical cut is among the curves, upright cylinders get their height h and σh/h as parameters in every curve
(shared). The vertical cut samples the height oscillations. In the horizontal DA, height enters through the single
coherence factor of section 9.5. Use in Model keeps h fixed at the joint value. A vertical cut
has no beam tail (Tail A fixed at 0) and no q resolution; a tail held at 0 has its width and exponent fixed. Random
cylinders and core–shell spheres are not available in a vertical cut.

**Analyze ▸ Send Both Cuts to Fitting** writes the horizontal cut as Send to Fitting does (`<stem>_fit_input.dat`)
and the vertical cut beside it as `<stem>_fit_input_qz.dat` (q_z in Å⁻¹; a series folder's `*_fit_input.dat` does not
match it), with its geometry in the `# observation:` line: `cut` = `vertical`, `alpha_i_deg`, `wavelength_nm`,
`q_parallel_nm` (the q_y of the column band's centre on the sample horizon), `critical_angle_deg` (Analyze's film
critical angle when q_z is refracted) and `horizontal_alpha_f_deg` (the exit angle of the horizontal band's centre row,
on the beam-centre column). The joint fit warns when the exported vertical band intersects q_y = 0 or |q_y| at its centre is below 0.02 nm⁻¹. A wide rod can also reach a more distant band. Inspect its position on the image.

**Checked against BornAgain 24.1** (`tools/benchmarks/joint_dwba_bornagain.py`: the vertical cut at φ_f = asin(0.3
nm⁻¹/k), α_f = 0.02°–1.2°, αi = 0.40°, Si of the table; each point at BornAgain's own q_∥ = √(q_x² + q_y²); after one
scale):

| Case | Largest deviation | Median |
|---|---|---|
| sphere R = 4 nm, one size | 9.8 · 10⁻⁴ | 1.9 · 10⁻⁴ |
| the same at the cut's one q_∥ = q_y (as a fit uses it: q_x left out) | 9.0 · 10⁻⁴ | 2.0 · 10⁻⁴ |
| spheres σR/R = 0.2 by number (24 radii) | 2.0 · 10⁻⁴ | 4.7 · 10⁻⁵ |
| sphere, one size, radial paracrystal D = 14 nm | 9.8 · 10⁻⁴ | 1.9 · 10⁻⁴ |
| upright cylinder R = 4 nm, H = 6 nm, one size | 1.1 · 10⁻³ | 2.6 · 10⁻⁴ |
| upright cylinders, radii and heights by number, σ = 0.15 each (64 particles) | 2.1 · 10⁻⁴ | 1.7 · 10⁻⁴ |

The tests compare the four paths with an independent sum, the cylinder averages with a direct double sum over radii and
heights (2 · 10⁻⁵), and fit a simulated horizontal and vertical cut of upright cylinders (R = 4 nm, h = 6 nm, Poisson
noise): the joint fit finds h within its error, which the horizontal cut alone does not determine.

**Warnings.** The joint fit assumes a dilute particle layer and independent heights and radii. **Needs attention**
shows two warnings:

- the vertical cut's reduced χ² is above 3 (with supplied or Poisson uncertainties). This statistic divides the
  vertical residual sum of squares by the vertical cut's effective degrees of freedom;
- the fitted height of an upright cylinder depends on the range. A second joint fit uses the full valid vertical
  range, with the horizontal data and parameter links kept (from 2α_c when the selected range is already the full
  cut). With heights h and h′ on the selected and comparison ranges, the warning shows when
  |h − h′| > 2√(σ_h² + σ_h′²) or |h − h′| > 0.02|h|. The quadrature sum is a diagnostic scale for overlapping fits;
  the relative threshold also works when a fit has no finite standard error.

**BornAgain height benchmark** (`tools/benchmarks/vertical_height_bias.py`, the shipped model, ten noise draws per
case). With BornAgain's average layer off, the noise-free fit recovers σ_R/R = 0.150005 and h = 5.9997 nm for inputs
of 0.15 and 6 nm. The average layer, which the model leaves out, lowers the height from the whole vertical cut by 3 %
for a Si-like layer of 10 % coverage, by 17 % at 40 % coverage and by 13–60 % for Au-like layers of 10–40 % coverage.
From 2α_c, the default, the Au-like heights are 6–20 % low and the dense Si-like height 5.5 % low. The two warnings
above flag every case with the average layer in every noise draw. Heights correlated with the radii (H = 1.5R) come
out about 14 % high in both ranges and are not flagged.

Implementation: `src/gimap/features/fitting/domain/joint_fit.py`, `domain/vertical_cut.py`,
`application/joint_fit.py`, `presentation/single/joint_fit_dialog.py`, `features/analyze/application/both_cuts.py`

### 9.19 Posterior check by MCMC

**Results ▸ Uncertainty and Report… ▸ Posterior check (MCMC)** compares intervals for the solution in Model.
For Predict parameters it samples the same composition likelihood and prior, starting from the selected solution.
The reference intervals use the draws assigned to that solution (§9.13), so chains that visit another peak can give
different intervals. Agreement measures the sampling accuracy for the assumed model. How often an interval holds the
true value is measured by the benchmarks of section 9.13.

**What is sampled.**

- A solution of Predict parameters in Model: the posterior Predict samples for it (section 9.13): the curve's σ with
  the model error floor, σ_eff² = σ² + (0.05 I)², the likelihood with the smooth multiplicative model error of
  amplitude τ and the amplitudes integrated out, and the prior uniform in the unit box of the network's values, in its
  coordinates z = logit(v):

  ```text
  ln p(z) = ln L(values(z) | cut) + ln π(z)                                                                     (9.46)
  ```

  over the composition parameters, including the beam tail, σ_q and τ. For level-3 spheres the same equal-area
  map gives |r| ≤ 1. The reference is
  Predict's sampled median and 68 % / 95 % intervals, with the same tempering as in section 9.13. If no sampled
  summary is available and the comparison used BIC, the reference is the reported least-squares interval.
- A manual fit (the last Refine or Search): its free non-linear values, uniform in the fit's coordinates t ∈ [0, 1]
  of (9.18)–(9.19) (the contact D ≥ 2R and the inequalities as in the fit), sampled in u = logit(t), the scales solved
  at every point as the fit solves them (a profile):

  ```text
  ln p(u) = −Σ_i r_i² / (2 s²) + Σ_k ln[t_k (1 − t_k)],   s² = χ²_red of the fit                                (9.47)
  ```

  the likelihood whose curvature the errors (9.25) linearise, so a curve the model describes only to χ²_red > 1 widens
  the intervals as it widens the errors. The reference is the fit's value ± 1σ (± 1.96σ for 95 %).

**The sampler** runs two independent affine-invariant ensembles of W walkers each (Foreman-Mackey et al., 2013).
W is even. The automatic setting uses at least 16 walkers and at least four per value. Each half of an ensemble
moves against its other half, at every step by the differential-evolution move of DE-MC (ter Braak,
2006; DREAM's proposal, Vrugt et al., 2009) with probability 0.8, y = x_k + γ (x_a − x_b), γ = 2.38/√(2d), or else by
the stretch move of Goodman & Weare (2010), y = x_j + Z (x_k − x_j), Z ~ 1/√z on [1/2, 2], accepted with
min(1, Z^(d−1) p(y)/p(x_k)). Predict supplies representative importance samples of the selected mode. The starting
points are divided along the first principal component of their standardised logit coordinates. Each ensemble draws
from a different half. Missing starts are drawn from the uniform prior box, also split this way. A manual fit starts
from the prior box. By default each ensemble runs for 3000 steps (**Steps**) and discards the first half.
**Seed** reproduces the starts and moves.

**Is it long enough?** For every value: the integrated autocorrelation time τ (the walkers' normalised autocorrelation
functions averaged, summed to Sokal's window M ≥ 5τ) and split R̂ across the independent ensembles (Gelman & Rubin,
1992). At each step the walkers' mean gives one trace value per ensemble. Splitting both kept traces in time gives
four series of length m. With V the mean within-series variance and B = m times the variance of the four series'
means, R̂ = √[((m−1)V + B)/(mV)]. Ensembles trapped in different modes can each look stationary while disagreeing in
R̂. The verdict is *converged* when the kept steps n per ensemble are at least 50 τ for every value and every R̂ is
below 1.05. The estimated effective sample count is 2Wn/τ. A ⚠ marks a value whose 68 % interval does not overlap
the reference. Agreement between ensembles can still miss a mode that neither visited.

**Checks.** With two ensembles of 4000 steps, the quantiles of a correlated four-dimensional Gaussian are within
0.062σ of the exact ones. On a synthetic sphere cut, both checks started from the prior did not converge (largest R̂
1984 for a manual fit and 60 for Predict): read the verdict before the intervals. [Results](../tools/benchmarks/predict_composition_checks.json)
On one core a sphere population takes about 1.5 ms per evaluation of a manual fit and 1 ms of Predict's likelihood per
row: 4000 steps of 16 walkers about 90 s, of 24 walkers on Predict's posterior about 130 s. A random cylinder takes
15–200 ms per evaluation, so a check of it takes hours.

Implementation: `src/gimap/features/fitting/domain/mcmc.py`, `domain/posterior_check.py`,
`application/posterior_check.py`, `presentation/single/posterior_check.py`

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
`q > 0`; q = 0 is dropped, and nothing is interpolated. q in Å⁻¹ is multiplied by 10; a file whose header states no
unit is read in the unit of the setting (Å⁻¹ by default, as on the Fit page). Without σ the curve gets
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
number of populations. **Cut geometry and substrate…** selects level 4 with a known substrate and both angles.
Otherwise **Cut qz (nm⁻¹)** selects the Born term when ticked and the effective interface when clear.
The eight most probable solutions of a half are its candidates, with
the stage *Network + posterior*; each carries its probability, its posterior intervals and flags (`solutions.json`:
`probability`, `posterior`, `warnings`, `ess`). The run writes `request.json`, `batch_summary.json`,
`top20_candidates.json` and `fitting_run_summary.json`, and per half `solutions.json`, `display.npz` and
`fitting_curves.csv`.

### 10.3 Workflow V5, removed in 0.2.0

Until 0.2.0 the window's default was *General V5*, a TensorFlow workflow, with *Single RC specialist* for one random
cylinder. *Predict parameters* (section 10.2) replaces both; it recovers the parameters better on fit1d's benchmark
(`tools/benchmarks/v5_recovery.py`).

## 11. Series

A series is an ordered list of frames of one measurement: files, a folder, or a multi-frame NeXus/HDF5 file. GIMaP reduces every frame (or every group of `N` summed frames) with the same geometry, masks, corrections and cut, and stacks one curve per frame into a map. This chapter states what is computed on such a stack: the map itself, a peak followed in a fixed window, the fit of every frame's ring peaks, the odd frames and stages of a series, and the comparison of several series.

Units follow the source of the curves. In Analyze the x of a map is q (Å⁻¹), q_y (Å⁻¹) or χ (°). Fitting ▸ In-situ series compares its curves on the q (nm⁻¹) of the first usable curve. Compare uses the x of its series, which must all carry the same x label.

### 11.1 The series map

Each row of the map is one frame, or one group of summed frames, in list order. Each column is one x value. The cut found on the chosen frame is held for every frame. The map keeps the measured values and resamples only when it must (case 3).

1. When all curves have the same x, the rows are stacked as they are.
2. When the curves were measured on the same points but some points are missing in some frames (valid detector columns that differ between frames), the union of all x is the grid, and every value goes to the column of its own x. A value that occurs twice at one x (both halves of a signed cut on |x|) gets two columns. This is used when the union holds at most 1.5 times as many points as the longest curve.
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

**Mean intensity.** For every row, the mean of the finite values in the columns with `|x − x_0| ≤ h`, where `x_0` is the window centre and `h` its half width (at least half the median grid step; one grid step by default). When no column lies inside, the nearest point is used, unless the window lies in a hole or beyond the ends of the map; then every row has no value.

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

The window is the same for every row and does not follow the peak. `FWHM_m` is a moment width; it equals the FWHM of a Gaussian that lies well inside the window. It grows with the window (tails, a sloping wing) and has no error; the batch fit (section 11.3) gives a fitted FWHM with its error. The CSV export says this in its header.

Implementation: `src/gimap/features/analyze/domain/series_map.py`, `src/gimap/features/analyze/application/series_map.py`

### 11.3 Batch fitting of the ring peaks of every frame

Batch Export ▸ Fit ▸ Peaks of the ring regions fits one peak per cut region in every GIWAXS frame. Each cut region with a q window `[q_lo, q_hi]`, a χ range (default 0–90°) and a both-sides flag is one target. GISAXS frames have no peak targets.

**Fit window and data.** The fit window is the region's window widened by its own width on each side:

```text
W = [ max(0, 2 q_lo − q_hi), 2 q_hi − q_lo ]                                                   (11.15)
```

A window of only ± one FWHM would leave too little background to tell it from the tails of a pseudo-Voigt. With the frame's q maps, I(q) is binned straight from the valid pixels above the sample horizon in the region's χ range, at the detector's own q step (at least 10 bins; the width/100 when the step is unknown). The uncertainty is Poisson when the frame holds counts (an integer frame without a background file and without mirror filling, section 6.4); otherwise it is the pixel scatter. With an active intensity correction the variance is scaled as in section 5.5. Without q maps, the region's own I(q) curve is used (2θ converted to q).

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

For GISAXS frames, the batch fit sends each frame's horizontal cut (the halves chosen) to Predict parameters, with the compositions of the automatic analysis (section 8.5); **Model** can give the composition (a sphere, a random cylinder or a vertical cylinder, with a paracrystal). Every frame is predicted from its own data, so the starting-value rule applies to the peaks only. The table has the most probable solution of each frame with its probability, χ²ᵣ (8.6) and log RMSE; lengths are in nm, q in nm⁻¹. A frame takes some seconds to a few minutes, depending on the number of points and the computer.

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

Fewer than 3 kept bins is an error ("widen the q range"). Non-positive values (gap codes, zero counts) are missing, not zero. A missing value filled with the median of its bin cannot make a frame look odd. With *shape only* an overall change of intensity (beam current, thickness, exposure) does not count as a change of structure; the level `ȳ_f` is exported separately.

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

`N_b` is the number of kept bins. At an end of the series `c = d` of the one side. A steady change lies between its neighbours and gives `c = 0`; a step matches one side. The threshold 8 is higher than 5 because these bins are the largest of the frame: in noise alone they reach about 3.5 times their usual step. The spike test is needed because a spike is diluted over the whole curve: a hot column 25 times too bright in 1 % of the bins is 1.4 in log10 there but only about 0.14 RMS.

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

and stop at the first `M` that fails, without look-ahead. The 5 % is of the *one-stage* residual `C_1`, not of the residual still left: real curves are not exactly piecewise straight, so a noise criterion alone would keep splitting. Invented example: with `C_1 = 100`, `C_2 = 17`, `C_3 = 1.0`, `C_4 = 0.45` and `P = 0.3`, the gains are 83, 16 and 0.55 against the bar 5, so 3 stages are suggested. The share `C_M / C_1` is recorded for every `M` (`unexplained_by_stages`), and the person may choose another number of stages.

**What changes between stages.** For each stage, the mean of `log10 I` over its kept rows on the compared bins (unfilled values). Between stage `s−1` and `s`:

```text
Δ_b = m_{s,b} − m_{s−1,b},    Δ_b ← Δ_b − mean_b Δ_b                                            (11.49)
change in percent = 100 · (10^{Δ_b} − 1)                                                        (11.50)
```

Up to 3 q of the largest positive `Δ` ("grows most") and of the most negative ("falls most") are listed, each more than 2 % of the compared q span from the others. The change is relative to the rest of the curve. Compare shows them in the tooltip of its folded *Odd frames and stages of every series* and writes them into its JSON record (`stage_details`: each `x` with its `percent` of (11.50)). The representative frame of a stage is its kept frame nearest (least squares) to the stage's mean `x`.

A stage is a description, not a phase: a smooth change is also cut where its pace bends. What grows or falls between stages says whether a boundary is structural.

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
- A long gap in some frames removes its q region from the comparison (the 95 % rule of section 11.4).
- The thresholds (5, 8, 0.8, 5 %, 2×, 2 %) are working choices stated in the code; no noise model of the user's detector gives them. Stage boundaries, odd-frame flags and groups carry no confidence values.
- A series measured on a much finer grid than the first series is compared only at the first series' points (section 11.10).

### 11.12 The timeline: how far each frame is from its neighbour and from a reference frame

The comparison of consecutive frames follows Xi-cam’s Timeline. GIMaP estimates the noise from the series as described below. [Xi-cam, section 3.4](https://journals.iucr.org/s/issues/2018/04/00/xe5032/)

Under the map (Analyze ▸ Series, the plot under the map: *χ² to the frame before* or *χ² to the reference frame*), every row `r` of the map is compared with the row before it and with a reference row `r_ref` (Options ▸ Reference frame, the first by default). The rows are mean intensities without their σ, so the noise of every x column is taken from the series itself, as in Eq. (11.5) (Eq. (11.6) below 4 rows), and no column counts as quieter than the median column:

```text
s(x)      = max( ν(x), median_x ν )                                                              (11.62)
χ²(a, b)  = 1/(2 n) · ∑_k [ (I_a(x_k) − I_b(x_k)) / s(x_k) ]²                                    (11.63)
χ²_prev(r) = χ²(r, r − 1),   χ²_ref(r) = χ²(r, r_ref)                                            (11.64)
```

The sum runs over the `n` columns where both rows have a value (the x range shown in the map when Options ▸ Stages compare says so). Two frames that differ by noise alone give χ² ≈ 1 (their difference has twice the variance of one frame). A **jump** is a row whose `χ²_prev` is a local maximum (not smaller than its neighbours) above both 3 and `median + 5 · 1.4826 · MAD` of the finite `χ²_prev` values; at most the 10 largest are marked. The plot also draws the stage boundaries (section 11.7, dashed) and the odd frames (section 11.5, ×). A click on the plot picks the row nearest on the frame axis and shows its frame. The table (Export ▾ ▸ Timeline as Table…) writes per frame `chi2_to_previous`, `chi2_to_reference`, `jump`, `stage`, `odd` (and `time_s`), with the method in its record.

`χ²` is a distance in units of the frame-to-frame noise, not a goodness of fit: σ is estimated from the series, so a series whose every frame changes (a steady drift) raises the noise estimate only through the curvature of the drift, and a column that is noisy in every frame weighs less. Implementation: `src/gimap/features/analyze/domain/series_timeline.py`, `src/gimap/features/analyze/application/series_results.py`

### 11.13 Chained fit of an In-situ series

For the fitted-parameter trend, elapsed time is the detector time of the frame minus the earliest
detector time among the fitted frames with times. The plot and the exported `elapsed_time_s` column
use the same origin. A missing acquisition time stays blank. Numeric frame times and calendar
timestamps are kept separately in the JSON. Refine error bars show the recorded local standard
errors (§9.8). For Predict, the trend shows the median and central 68% interval when available (§9.13).

When the network and the manual fit use different size laws, weights or resolution, the network supplies a
starting point for a local fit under the manual settings. The two candidate fits are ranked on the same objective.

Fitting ▸ In-situ series ▸ *Previous result, Predict at chain breaks* refines every frame from the converged result of the frame before (as *The previous frame's result*). The chain **breaks** if the fit fails or does not converge, a new stage begins, or χ²ᵣ satisfies Eq. (11.65).

```text
χ²ᵣ(i) > 2   and   [χ²ᵣ(i) > 3 · χ²ᵣ(p)   or   χ²ᵣ(i) > 3 · χ²ᵣ(s)]                      (11.65)
```

Here p is the preceding frame in the direction of the fit, and s is the most recent chain start (the first frame, a frame started from Single, or the frame of a break), using the χ²ᵣ of the fit kept there.

Bounds, undetermined values and poor fits (χ²ᵣ > 10) are flagged separately. At a break, Predict proposes a restart with the particles of the starting model. The chained fit and the restarted fit are compared on the same objective. A converged fit takes precedence, then a completed fit, then the lower χ²ᵣ. Ties retain the original fit. The optional backward pass repeats this comparison using the next frame as the starting point.

**Choosing a Predict solution.** Let pₜ(j) be Predict's probability for solution j from frame t alone. Rank 1 has the largest pₜ. For two solutions, match populations of the same family and paracrystal status in ascending order of radius within each group. For each matched population use the shared positive coordinates x = (ln R, ln D, ln(σ_R/R)), omitting D for a population without a structure factor. The distance is

```text
d²(i,j) = Σ_matched populations Σ_shared coordinates (xₜ(j) − xₜ₋₁(i))².                     (11.66)
```

A chained restart admits solutions with pₜ ≥ 0.1 p_max. *Predict each frame* with *Prefer the solution that continues the previous frame* admits those with pₜ ≥ 0.2 p_max. The closest eligible solution with matching composition is kept. Ties go to the more probable solution, as does the choice when no composition matches. With no reference frame, Predict takes rank 1. The cutoff and the pass direction can move a step in the fitted series. The closest rule leaves the single-frame probabilities unchanged. The coordinates for each candidate come from its Predict fit. The greedy rule compares them with the reference frame’s retained least-squares fit, while the HMM compares the Predict candidates on adjacent frames.

**Smooth across frames (hidden Markov model).** The states are all solutions returned by Predict for each frame. The probability of a path is proportional to the product of its single-frame evidences and its transition weights. For adjacent states,

```text
w(i,j) = exp[−d²(i,j)/(2s²)] × ε(s)^[composition changes]
ε(s) = 0.01^[(0.1/s)²].                                                                    (11.67)
```

A composition change means a change in the number of populations of any family or paracrystal status. The penalty is 0.01 at the default log step s = 0.1 (about 10% per selected frame). Larger s permits larger simultaneous changes. At s → ∞ every transition has unit weight, so the path reduces to independent choices. The pair weights are symmetric. Forward–backward sums the globally normalized path probabilities to give the marginal probability of every state, and Viterbi chooses the single most probable path. A state on that path can differ from the largest marginal at its frame. A frame without a prediction separates segments. Choosing a fresh start at each stage also splits the path at the stage boundary.

**Reading the results.** The table and exports distinguish the kept fit from each attempted Predict restart and give the selection rule, single-frame rank, chosen p, maximum p and, for the HMM, the chosen state's marginal. The record identifies the candidate populations and stores all the marginals in single-frame rank order. The trend shows the median and central 68% posterior interval of the selected mode. Its small open symbol, when enabled, marks the mode's least-squares value. These intervals describe the uncertainty within that mode; the choice of the mode is not in them.

**Time per frame on synthetic detectors.** On eight int32 TIFF frames of 1000 × 1000 and of 2000 × 2000 pixels (Intel Core i9-13900HX, one worker and one numerical-library thread, 369 fitted points per cut), reduction, cuts and chained fit together took 0.24–0.37 s per frame, and one Predict call about 2.5 s. [Benchmark settings and per-frame times](../tools/benchmarks/insitu_online.json)

### 11.14 The results database

Analyze records the curve of every row of a map (as reduced: both halves of a cut as measured, on the x of the frame, with its frame number, file, frame index, sum and time) in `gimap_series.sqlite` (SQLite, WAL), Fitting the fit of every frame (state, χ²ᵣ, values and errors in nm and nm⁻¹, flags, start), and a watch the files it skipped and why. Arrays are kept as little-endian float64; a frame written again replaces its row. Compare stacks one curve of one series on the x of its first frame by linear interpolation inside each curve's own range, never across a gap wider than 10 of its steps. Implementation: `src/gimap/shared/series_store.py`

## 12. Sputter

The Sputter workspace follows an in-situ sputter deposition series. Its **Analyze** runs a separate analysis package (`iampred`, bundled with GIMaP as `utils/sputter_client`) as a subprocess and shows what the package writes. The networks, the physical forward model and the refinement are the package's and are documented with it; this chapter states what GIMaP does with the package's input and output (12.1), with the deposition rate (12.2), and how it draws a solution in real space (12.3).

### 12.1 Analyze: what the main screen shows

**Inputs handed to the package.** The folder of CBF frames; the metal and the substrate; the deposition rate in nm/s, either as the rate at the sample or as the reading of the quartz balance (section 12.2); α_i; the sample–detector distance; the energy or the wavelength; the direct-beam position, entered in GIMaP's pixel coordinates and handed over minus 0.5 px as 0-based pixel centres (section 1.3), or, when it is empty, the request to estimate it from the frames; the sample roll (0° when empty); optionally a detector mask (without one the package uses the pixel mask it was built with), the time of the scan at which the deposition started, its duration, and the thickness of a polymer film. GIMaP changes none of these values. The package keeps an imported scan under a label; GIMaP derives the label from all these inputs, so a changed input is imported anew and never read from an earlier import.

**Network and photon energy.** The package picks its networks by metal and substrate. Each system has a network trained at one wavelength (12 keV; Cu 0.105 nm) and α_i 0.40°, and an any-energy network trained on images simulated at 8–15 keV and α_i 0.25°, 0.40° and 0.55°, with parts of the detector area cut away at random (as by another distance, detector size or beamstop); the any-energy network of the polymers was then fine-tuned on measured scans by rendering its candidates with the differentiable forward model and comparing them with the counts. The any-energy networks use the package's optical constants at E₀ = 12 keV (Cu: 0.105 nm), changed with the energy by the ratio of the tabulated values, δ(E) = δ(E₀) · δ_tab(E)/δ_tab(E₀) and the same for β, so at E₀ they are the earlier ones exactly; they read the image on a grid of k_z = k₀ sin α_f with k₀ = 2π/λ and get k₀ and k₀ sin α_i as inputs. Within 5 % of a one-wavelength network's wavelength (Au: 12 keV only) the package runs that network with its own optical constants, the earlier networks of the same kind and the any-energy network where it renders with the same optical constants (at that wavelength), and keeps the one with the lowest median fit deviation. Elsewhere within 8–15 keV it runs the any-energy network alone at the scan's energy: on simulated scans at 10 and 14 keV its first candidates were closer to the true values than those of the one-wavelength network in each quantity compared (size, spacing, coverage). For Au and Ag on TiO₂ the any-energy network is the primary network everywhere in 8–15 keV, and the earlier 12 keV network runs beside it at 12 keV. Outside 8–15 keV the package computes the optical constants at the scan's energy from its table (4–30 keV) and uses the network's candidates only as starting solutions. GIMaP hands over the energy unchanged and shows which network was used.

**A time between the windows.** For a time t that is not one of the analysed windows the package (`iampred.at_time`) interpolates the parameters of the joint fit between the two neighbouring windows t_a < t < t_b, linearly in u = ln(t − t_onset), the axis along which the joint fit is smooth: x(t) = (1 − w) x(t_a) + w x(t_b), w = (u − u_a)/(u_b − u_a) (linearly in t when a neighbour is not after the onset; at or beyond the first or last window the nearest window's solution). The window of the frames at t is then built as the analysis builds its windows and refined by the same least-squares refinement as a window of mode Refine (8 evaluations per parameter), starting from the better of the interpolated solution and the nearest window's joint solution. Both entries are drawn and described like any entry: the interpolation shows what the joint fit implies at t, the refinement what the frames at t ask for; the difference of their fit deviations says how much the frames at t add.

**Quantities.** For every time window the package gives several *entries*: up to three candidates of its network, ranked, and, when it refined, the refined solution and the joint fit of the whole scan (every window refined together over the scan, smooth in time; the package then marks it as lying on its smooth evolution). For Cu and Ag on a polymer film (every network family of these systems — `cu_p5`, `cu_v4`, `cu_u1`, `polyag_p3`, `polyag_v5`, `polyag_u1` — whichever the comparison of the networks chose or the energy of the scan routed to) the joint fit of modes Refine and Full has more starting points: in every window whose best candidate has at least half of its metal in the continuous layer, two copies of that candidate — one with the whole-film density fluctuation switched on (correlation length 6 nm, amplitude 1), one as a dense film (base share 0.9, grains of 3 nm radius on it) with the same fluctuation — are each refined on their own for 15 steps, and the time path the joint fit starts from may take them; the optimiser does not switch the fluctuation on or reach the dense film by itself. For these families the no-decrease rule of the joint fit holds only for the metal amount and the film thickness, not for the grain size, spacing and coverage: on a closed film the grains are relief on it, and keeping them from shrinking held the metal out of the film, which then was too thin for its thickness fringes. The package's development test on its polymer scans required that no scan's median fit deviation exceed that of the previous setting by more than 5 %, that the median over the scans not exceed it, and that on a Cu scan with a thick film the residual of the fringes fall to at most 0.8 of the plain joint fit's; the joint fit takes a few times longer than without the extra starts. An entry holds the grain radius R, the grain height H, the centre-to-centre distance D, the gap D − 2R, the width σ of the log-normal distribution of the radius, the coverage (0 to 1) and the metal amount (nm), each as the package defines it for the model family of the sample; a quantity a family does not have is absent and is not shown. Every entry of a grain family also holds the thickness t_b of the continuous metal layer under the grains (0 for separate islands; the grains stand on it, their caps H − t_b high, section 12.3); it is shown as a value only when the result lists it among its quantities (a newer package), otherwise it is used for the real-space picture alone. Lengths are in nm, the time t in s from the first frame; the nominal deposit of a window is the package's value. GIMaP shows the values with three significant digits (two for σ and the coverage) and computes none of them.

**Fit deviation.** The package's measure of how far the model image of an entry is from the measured image (its deviance per cell with a floor for the model error; lower is better). GIMaP shows the number and does not turn it into a grade, a probability or a threshold.

**Which entry is shown.** In this order: the entry the user chose for the window; the entry the package marks as lying on its smooth evolution over the whole scan; the entry it marks as having the lowest fit deviation; the entry with the lowest fit deviation; the first entry. The evolution plots show the entries so chosen, one per window, against t; a window without a solution leaves a gap in the lines.

**Images.** The measured count rate and the model of each entry are arrays over the exit angle α_f (°) and q_y (nm⁻¹) on the package's cells, which need not be evenly spaced. Both are drawn on those cells (cell edges halfway between neighbouring centres), with q_y on a logarithmic axis when every q_y is positive, on identical axes and with one logarithmic colour scale. Its limits are the 1 % and 99.7 % percentiles of the positive measured values (of the model's when nothing was measured), so they do not depend on the entry shown. Cells without data are left empty; measured values below the lower limit, zero counts included, are drawn in the lowest colour. The arrays are not changed.

**Components.** A newer package writes, for every entry, the model image split into the physical terms of its forward model, each alpha_f-smeared and scaled like the model so that the terms add up to the model image (the package checks this to about 10⁻¹⁵): grains and islands with their interference; large clusters; the film term (density fluctuation, connected network or roughness of a continuous film); voids under the continuous layer; buried clusters; metal-rich domains in a polymer; the rough top of the continuous layer; fluorescence (flat in q_y); the substrate scattering seen through the metal; and the beam-path background (the flat share of the pre-state and, where the family has it, the difference term of the beam path, which can be negative). On porous TiO₂ the terms are the clusters in the pores and on the cap layer, the metal network in the pores (the difference with the network strength set to zero) and the substrate seen through the metal. For each entry the package gives each term's share of the model counts on the fitted cells, Σ E c_k / Σ E μ (E the exposure, c_k the term, μ the model), and per cell the term's share of the model image (float16, about three significant digits). GIMaP draws a term's image as that share times the model image, on the colour scale of the measured image (12.1, Images); it draws terms with a share of at least 0.1 % and computes nothing else.

**Growth process.** From the entries shown by default (the joint fit when every window has one, else the smooth path) the package describes the growth against the metal amount h (nm; the equivalent thickness of the joint fit — mass factor × rate × time — so it does not depend on the X-ray wavelength): the amounts at which the coverage first reaches 0.5 and at which half of the metal is in the continuous layer (the film forms; log-linear interpolation between windows); the power laws D = D₁ h^p and R = R₁ h^q fitted by least squares in ln h over the windows before the film forms (at least three windows spanning at least a factor e^0.3 in h); the median D and R once the film has formed; for a scan whose deposition stopped, the amount then and the change of ln D per decade of time after it; the median fit deviation; and the number of steps back between neighbouring windows (R by more than 10 %, D by more than 5 %, the coverage by more than 0.03). The stage of a window is *islands* below coverage 0.5, *coalescence* above it, *continuous film* once half of the metal is in the continuous layer and *after the deposition* once the amount stays constant, taken on the running maximum of the coverage and of the film share, because sputtering only adds metal; on porous TiO₂ the stages are *clusters* and *metal film* (the film term above half of its largest value). The package lists the median and range of each descriptor over the scans of the same metal on the same substrate it was built from (development reference, without independent characterisation) and whether this scan lies within, below or above that range (only with at least two scans). GIMaP shows these values and draws the stages as bands behind the evolution plots (a boundary halfway between the windows of neighbouring stages); it computes none of them.

**Growth-law prior.** With the prior on (*Analyze options*), the joint fit of the whole scan gets, for every window before the deposition stopped and for each law X ∈ {D, R}, the residual r = √(w g(h)) (ln X − ln X₁ − p ln h) / σ(h), σ(h)² = s_a² + (ln h)² s_p², g(h) = 1 / (1 + (h / h_base)⁶). ln X₁ and p are the medians of the power laws above over the package's reference scans of the same metal on the same substrate, s_a = max(half their range of ln X₁, 0.15), s_p = max(half their range of p, 0.10), h_base the median amount at which the film forms, and w = 1, so a departure of one width costs as much as one unit of fit deviation in a window. A law of fewer than two reference scans is not used. The same terms also weigh the candidates when the start path through the scan is chosen. *Auto* switches the prior on only for the systems where it passed a leave-one-out test whose criteria were fixed before the runs (each reference scan fitted with the laws of the other scans of its system and compared with its joint fit without the prior: median and sum of the fit deviation at most 1.15 times, no more steps back, and on most scans no larger jumps Σ(|Δ ln D| + |Δ ln R|) between neighbouring windows before the film): at present Ag on PS. The laws are model values of the reference scans (development reference, without independent characterisation).

**Compare Scans.** **Compare…** places several results side by side and computes nothing. Its curves are against the nominal deposit of each window as the package gives it (rate × time since the onset; or against that time): two scans at different rates meet at the same nominal amount. They are the solution shown of each window and, as open squares, the report's model-free readings whose source is *reading* (the film thickness from the fringes, the grain height from its fringe, (α/α_c)² of the metal's Yoneda peak, 2π/q of the knee); the report's estimates are left out. Its growth table holds the descriptors above, whose metal amounts are those of the joint fit (mass factor × rate × time): where the package's mass factor of a scan is not 1, its descriptors sit at amounts that differ from the nominal axis of the curves by that factor.

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

On a polymer film a line under the views says that no metal inside the polymer is drawn: the package's model has none. For Cu on polymer films the line says the data did not resolve it (two forms were tested with the package: clusters buried below the film, which improved single-window fits too little and which the whole-scan fit, started without them, took up in few windows, and a Cu/polymer composite base layer with Cu domains, which the whole-scan fit did not take up); for the other metals it says the model has none. The reasons are in the measurement: at an incidence angle above the critical angles of the metal, the polymer and Si the incident field reaches the whole film, so only the narrow band of exit angles below the polymer's critical angle carries depth information, and it is shaped as much by the optics of the metal layer on top; clusters below about 1.5 nm and dissolved atoms scatter nearly evenly up to the end of the package's q_y grid (about 1.8 nm⁻¹), like fluorescence and background; the pre-deposition frames of these polymer films show no lateral correlation peak that metal in one phase would strengthen at a fixed q_y; and a metal amount below the quartz balance's is not separable from the balance's and the intensity scale's errors.

Two grains touch when their centres are no farther apart than the sum of their radii; touching grains form a cluster. The arrangement is *islands* when no two grains of the view touch, *network* when a cluster connects two opposite edges of the square (percolated), else *coalescing*. A solution with b > 0 is a closed film: the top view is the metal of the base layer over the whole square, with the grains as relief on it.

Implementation: `src/gimap/features/sputter/domain/real_space.py`, `src/gimap/features/sputter/presentation/real_space_view.py`

## 13. Tool windows and Labs

The tools of this chapter work beside the Analyze → Fitting pipeline and keep their own conventions. For each
tool this chapter states what is computed, in which units and pixel convention, what is assumed and what is not
corrected. Lengths on the detector are in pixels unless a unit is given.

### 13.1 XRR Series Extractor

**Purpose and assumptions.** The extractor turns an angle series of area-detector frames into a specular
reflectivity curve, one point per frame (Tools ▸ XRR Series Extractor). It assumes one geometry:

- the beam and the detector are fixed; the sample turns by θ about a horizontal axis through the beam;
- the detector is planar with the specified rot1, rot2 and rot3;
- the specular ray lies in the vertical scattering plane;
- θ is the sample angle as read or computed, without any offset.

**Wavelength and q_z.** λ follows from the photon energy E in keV by (1.1), and

```text
q_z = 4π sin θ / λ                                                               (13.1)
```

This is the specular wavevector transfer (Als-Nielsen & McMorrow, 2011), in Å⁻¹. It is the vacuum value, without
refraction in the film.

**Position of the specular beam.** A sample turned by θ sends the specular beam to a scattering angle 2θ from the
direct beam. With rotations, the ROI follows the intersection of the detector plane with the ray `(cos 2θ, 0, −s_dir sin 2θ)` in laboratory (along, right, up) coordinates. The pixel axes convert this intersection to fractional column and row. For the perpendicular detector in orientation 3 this gives

```text
x_s = x_b                                                                        (13.2)
y_s = y_b + s_dir · L · tan(2θ) / p_y                                                (13.3)
```

| Symbol | Meaning | Unit, default |
|---|---|---|
| (x_b, y_b) | direct-beam position at θ = 0 (column, row) | px |
| L | sample–detector distance | m (entered in mm; default 5000 mm) |
| p_y | vertical pixel size | m (entered in µm; default 172 µm) |
| s_dir | −1 for **Up in the laboratory frame** (default), +1 for **Down in the laboratory frame** | — |
| E | photon energy | keV (default 12 keV) |

In (13.2) the horizontal pixel size p_x does not enter: x_s does not change with θ.

**Integration window.** The intensity of a point is read in a circular region of interest (ROI) of radius r
around (x_s, y_s). Pixel (row i, column j) belongs to the ROI when

```text
r = 0:   j = ⌊x_s + ½⌋ and i = ⌊y_s + ½⌋   (the single nearest pixel)              (13.4)
r > 0:   (j − x_s)² + (i − y_s)² ≤ r²                                             (13.5)
```

Here x_s and y_s are 0-based pixel-centre indices (see below). Invalid pixels, defined below, are left out. The
point's intensity is then

```text
I_sum  = Σ_{valid ROI pixels} I_ij                                               (13.6)
I_mean = I_sum / N_valid                                                         (13.7)
```

`Sum` is the default; r has a default of 2 px (range 0–500 px). N_valid is written with each point. A ROI with no
valid pixel gives no intensity (an empty CSV cell). A circle of r = 2 centred on a pixel holds 13 pixels; one of
r = 1 holds 5. The ROI is clipped at the frame edges.

A pixel is invalid when it is not finite, when it is negative in an integer frame (a gap code), or when the
reader masks it (section 4.1); in a NeXus/HDF5 frame the reader masks the rejected `pixel_mask` bits, the maximum
of an unsigned type and the parts of a stitched canvas that no module covers. Negative values in floating-point
frames stay data. Analyze's bad-pixel search, drawn masks and thresholds are not applied.

**θ of each frame.** Either a linear sequence or a motor dataset in the NeXus file:

```text
θ_n = θ_start + n · Δθ,    n = 0, 1, … (position in the whole series)            (13.8)
θ_n = v_n · c_unit,         c_unit = 180/(1000π) for "mrad", 180/π for "rad", 1 otherwise   (13.9)
```

Defaults: θ_start = 0°, Δθ = 0.01°. In (13.9) the `units` attribute of the dataset decides:
`deg`/`degree`/`degrees`, `rad`, `mrad` and `urad` (also µrad/μrad) are read, and a dataset without units is in
degrees. Unknown units and non-finite angles stop the extraction.

**Pixel coordinates.** The extraction works internally in 0-based pixel-centre indices (the index convention of
section 1.3: the centre of the first pixel is (0, 0), x is the column, y the row, row 0 at the top). The window
shows and accepts canonical pixel coordinates, as Analyze and Geometry Calibration do, and converts with (1.5). The exported CSV (`roi_center_x_px`, `roi_center_y_px`) and the JSON record (`geometry.beam_center_x_px`,
`beam_center_y_px`) keep the index values, 0.5 px lower than the window shows; the record states this in
`center_convention`.

**Where the geometry comes from.** Each value is tagged with its source. A value typed by the user is never
replaced. Otherwise the metadata of the first frame (`file`) replaces the last applied Geometry Calibration
(`calibration`), which replaces the built-in default (`default`). A calibration made on frames of another shape
contributes only its energy. Without any beam centre, the frame centre is used (`image center`). A beam centre
from a NeXus header is taken in the displayed (transformed) coordinates, as in Analyze; a stitched module set has
no single header centre.

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
- a θ offset (sample misalignment).

No uncertainty is written; for photon counts and `Sum`, √I_sum is the Poisson estimate. No reflectivity model (for
example Parratt, 1954) is fitted.

A CBF frame without an energy in its header takes the energy of a NeXus file found in a sibling folder of the
scan; the record then marks the energy as `file` without naming that file.

Implementation: `src/gimap/features/xrr/domain/reflectometry.py` (equations), `src/gimap/features/xrr/application/use_cases.py`
(θ, one frame at a time), `src/gimap/features/xrr/infrastructure/adapters/detector_series.py` (series, angle units),
`src/gimap/features/xrr/presentation/pixel_coordinates.py` (coordinates).

### 13.2 SEM Particle Analysis

**Purpose.** The tool measures projected particle sizes in an electron micrograph, or reads sizes measured
elsewhere, and compares their radius distribution with one predicted by 2D Prediction (Tools ▸ SEM Particle
Analysis). Measurements are made on the pixel grid of the original image.

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

for large digitised discs. On synthetic discs of radius 5–100 px the median C is 0.90 at every radius (spread
0.84–0.95 at 5 px). A digitised square gives about 0.81 (ideal π/4 = 0.785). A minimum circularity above about
0.85 therefore rejects digitised discs too.

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

**The prediction handed over by 2D Prediction.** It is accepted when the module declares its row axis with the key
`r` and the unit `nm`. The radius distribution is the row marginal of the predicted matrix (equation (13.38)), on the
bin centres of the module's axis (equation (13.37)). The comparison holds a frozen copy with the module, model
path, preprocessing steps and settings, runtime and input files.

**What is and is not computed.** GIMaP computes the counts (detected, accepted, rejected, invalid pixels), the
per-particle values of (13.18)–(13.23) and the binned probabilities of (13.27) and (13.30). It computes no summary
statistic (mean, median, standard deviation, polydispersity index, distribution fit, confidence interval) and no
test of agreement between SEM and prediction; compute such numbers from `sem_particles.csv` and state the method.
The two distributions are different quantities: SEM counts particles in a small field of view, and the scattering
prediction describes the illuminated area as the module's model defines it.

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
negative and the stored type is not floating point. The reader's mask holds the rejected NeXus `pixel_mask` bits,
the maximum of an unsigned type and the parts of a stitched canvas that no module covers. Negative values in
integer data are the gap and bad-pixel codes of counting detectors (Pilatus −1, −2).

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
code is written in the smallest signed type that holds its values; in HDF5 it keeps its type, and the largest value
of the type marks the pixel (the NeXus convention). A floating-point frame, or a frame whose values are no longer
whole numbers, is written as float with NaN on every pixel without data. CBF holds integers only: float values are
truncated to int32 and a pixel without data is −1. A negative value (−1 or less) then reads back as a gap code under
rule (4.1), so CBF suits counts only. `pixel_values` counts those pixels (`negative_data_read_as_gap_codes`), the
values truncated (`values_truncated`) and every pixel that reads back without data
(`pixels_read_back_without_data`).

```text
I_u16 = round(65534 · clip((I − I_min) / (I_max − I_min), 0, 1)),   65535 on pixels without data   (13.31)
I_u16 = trunc(clip(I, 0, 65535)),                                     65535 on pixels without data   (13.32)
```

I_min and I_max are taken over the pixels with data; a constant frame gives zeros in (13.31). In (13.32) the
fractional part is cut off, not rounded, and a value at or above 65535 becomes 65535. The value 65535 marks a pixel
without data, as the largest value of an unsigned type does in NeXus.

An HDF5 output, and the one-file container while all frames have one size, also marks every pixel without data in
the NeXus `pixel_mask` (bit 0, "gap") at `/entry/instrument/detector/pixel_mask`.

**What reads back as it was.** A round trip of the repository's frame `TestSAXSdata/…_00005.cbf` (int32 with gap
codes) through every output format gives:

- default mode: int32 with the codes −1 and −2 in HDF5, TIFF, CBF and NumPy; GIMaP reads the same pixels as without
  data;
- 32-bit float: NaN on those pixels (CBF: int32 with −1); the same pixels read back as without data;
- the uint16 modes: the pixels without data hold 65535. HDF5 reads them back as without data (its `pixel_mask` and
  the NeXus convention); after (13.32) so are the pixels whose values were clipped to 65535. TIFF, CBF and NumPy
  files carry no mask, so there the pixels without data read back as valid pixels of 65535.

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

**Input.** Only CBF frames are read, directly with fabio (Knudsen et al., 2013), as float32 of the stored values.
The shared reader of Analyze is bypassed: no P03 transform and no mask are applied, and negative gap codes stay in
the image. A stack of N consecutive files (sorted by name; in a folder batch, each group of *Files per prediction*) is summed:

```text
I = Σ_{f=1…N} I_f                                                                (13.33)
```

**Module.** A module is a folder under `modules/` with a `module.yaml` and its own `preprocess.py`. GIMaP calls the
entry named in `preprocess.entry` with the image, the configured steps and parameters, and the module folder. The
steps are the module's own code. The shipped modules use these steps:

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

The marginals of (13.38) are not normalised. When a module declares no axes, GIMaP draws the rows with the key `r`,
unit `nm` and 0.05–15 nm, and the columns with the key `h` and 0.05–15 nm; these placeholder axes are not a
physical scale. For `Au_Silicon_15nm`, the YAML comment says the network was trained with rows = R and
columns = H/2, and the column axis shows the full height H (0.1–30 nm).

**Points to confirm with the model authors.**

- The Ag and Cu distribution modules declare no axes, so they are shown on the placeholder axes and are not handed
  to the SEM comparison.
- The four SF parameters have no units in the YAML; nm for `t_Cu`, `t_polymer`, `D` and `sigma` is likely.

**What is not done.** A prediction has no uncertainty. The input is not corrected (no flat field, background or
transform), and the preprocessing differs between modules. Use a module only for frames like those it was trained
on.

Implementation: `src/gimap/features/prediction/infrastructure/adapters/fabio_images.py` (input, stack sum),
`src/gimap/features/prediction/infrastructure/adapters/module_preprocessing.py` (module entry, shape),
`src/gimap/features/prediction/domain/array_contracts.py` (shape, (13.36)–(13.38)),
`modules/Au_Silicon_15nm/preprocess.py`, `modules/Cu_Polymer_Silicon_Sputter/preprocess.py`,
`modules/Cu_Polymer_Silicon_Sputter_SF/preprocess.py` (steps), `src/gimap/integrations/tensorflow/predictor.py` (model run).

### 13.5 Trainset Build (Labs)

This workspace has been removed from GIMaP. Models are trained outside GIMaP and imported into 2D Prediction
(section 13.4); existing research files are not changed.

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
| AI assistant | `assistant.provider` and `assistant.model`: the service and the model of the run, when the AI exported the curves |
| Geometry | `geometry` (pixel sizes, distance, beam centre, wavelength, α_i; `rot1_rad`, `rot2_rad`, `rot3_rad`: the detector's rotations of section 3.7; the `dropped_…` and `flat_detector_…` fields of 0.1 are `null`), `geometry_convention` (canonical pixel-corner frame, row 0 at the top, kinematic (vacuum) grazing-incidence q of a flat detector, tilted by the rotations, in Å⁻¹), `instrument_profile` (name, source, time of update; the source begins with `automatic analysis:` when Run Automatic Analysis or `tools/gimap_agent.py` chose the geometry and with `assistant:` when the AI did; records before 0.2.0 write `assistant:` for both) |
| Statistics and binning | `error_model` (`poisson`, `scaled_poisson`, `pixel_scatter` or `sigma_frame`: the model used for the frame, section 6.4; `null` without a reduction), `binning` (GIWAXS: `radial_bins`, `q_min`, `q_max`, `bin_width` and `unit` of the fixed-edge bins of every I(q), `bins_chosen`, `radial_points` (the points of I(q) after merging), `merge` (the rule of section 6.5), `chi_bin_deg` and `chi_merge`; GISAXS: one point per detector column or row, no merging) |
| Time | `created`: the UTC time the record was written (ISO 8601, for example `2026-11-01T05:52:31+00:00`) |
| Corrections | `corrections` (every setting; `flat_field`: file, path, SHA-256, normalisation mean, floor, pixels left out, shape and how it was applied, section 4.10, `null` without one), `intensity_corrections` (GIWAXS: how applied, formulas, parameters), `bad_pixels_left_out`, `drawn_mask_pixels`, `pixels_filled_from_mirror` |
| Results | `markers` (beam centre, horizon row, cut bands, Yoneda estimate, I(χ) window, peaks …), `curves` (key, title, x label, region of each curve), `messages` |
| Files | `source_files` (every file a frame was read from: module files, master and data files), `input_sha256` (every input file by name with its SHA-256: the source, its module or data files and the frames summed with it), `source_layout` for NeXus/HDF5 (kind, summary, dataset path, `transformations`), `files` (the files written) |

The numbers are those of the reduction; `geometry` is the geometry that was applied.

`reproduction` names the source revision, input files and numerical settings for replay. Its reference columns keep
the original array type and bytes in little-endian order (zlib, then Base64). Replay reads the detector files,
checks their SHA-256 digests and runs the reduction with the recorded geometry and corrections.

A manual fit record keeps the starting model, the points in fit order, their units and uncertainties, the per-point
q resolution when supplied, and the weighting used. It records how many local restarts were accepted and their
evaluation budget. A joint fit also stores the geometry of each vertical cut. Predict parameters uses the recorded
seed, options and network weights.

For a reference column `r` and a reproduced column `a`, the maximum relative difference is `max_i (|a_i − r_i| / |r_i|)`. Equal entries contribute zero, including matching infinities and paired NaNs. A nonzero value against a zero reference, or unmatched nonfinite values, give `"infinity"`. A missing column or unequal row count gives `null`. The separate `identical` test compares array types and bytes, so it distinguishes positive and negative zero. The record gives the Python, NumPy and SciPy versions used for the calculation.

An NXcanSAS export (`<stem>_nxcansas.h5`, NeXus definition NXcanSAS 1.1; Batch Export `nxcansas/`, the command
line `curves_nxcansas.h5`) carries the same record, with the key `curve` (key, title, x label and region of that
curve) added, as the JSON `data` of an NXnote `gimap_record` in each entry's SASprocess. Each SASentry holds one
curve against q: `Q` in 1/angstrom exactly as reduced (a horizontal cut keeps its signed q_y; the entry's `title`
says which q), `I` in `1/cm` for absolute intensity and `arbitrary` otherwise, `Idev` = statistical σ in the same
units, and `mask` (1 where Q or I is not finite). SASinstrument holds the wavelength (`incident_wavelength`,
angstrom), the sample–detector distance (`SDD`, m) and the pixel size; SASsample `pitch` (degree) is αi, the
rotation about the horizontal axis normal to the beam in NeXus axes. Curves against χ or 2θ are not written, since
Q of NXcanSAS is a momentum transfer. Fitting's fits (Save ▸ Data and Fit, an In-situ series' Fits as NXcanSAS)
write the points fitted and the model at their Q as two entries each; the SASentry attribute `gimap_role` says
`data` or `model` (NeXus allows such an extra attribute, and other readers ignore it). Opened again, such a file
offers its measured curves only (older files mark the two entries by the ends of their titles, ": the points
fitted" and ": the model").

Implementation: `src/gimap/features/analyze/application/export_record.py`, `src/gimap/features/analyze/infrastructure/adapters/csv_export.py`;
NXcanSAS: `src/gimap/shared/nxcansas.py`, `src/gimap/features/analyze/application/nxcansas_export.py`.

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
  GISAXS, `radial_bins` for GIWAXS), `file_format`, `intensity_unit` (`counts_per_pixel` by default, cm⁻¹ for absolute intensity), `counting_model_valid`
  (whether σ may be taken from Poisson statistics: true for photon counts in the sense of section 6.4, without
  GIWAXS intensity corrections or a reference-peak normalisation), `threshold_enabled` (a value range
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
number of pixels filled from the mirror image. A value `null` means that the step was off. The reproduction
recipe stores both the user mask (`mask`) and the final excluded pixels (`excluded_pixels`). Each bitmap gives its
shape and alternating run lengths in row-major order, starting with zeros. Lengths are unsigned 64-bit
little-endian integers, compressed with zlib and encoded in Base64. SHA-256 covers one byte per pixel, 0 or 1,
before compression. Replay restores the user mask and checks the final excluded pixels after preprocessing.

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

Refitting from an exported single-curve record adds `parent_fit_record` to the new fit JSON.
It names the original record and the SHA-256 of its bytes at opening. If replay uses a copy of a
missing input beside the record, `comparison.json` lists its actual path and verified digest in
`relocated_inputs`.

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

The 1D Predict request and summary also record the SHA-256 of the network's record, which names the SHA-256 of the
weights, checked when they are loaded. The request keeps the normalised options and states the seed policy:
Predict parameters uses a fixed seed (0), so a run repeats with the same curve, settings and network.

### 14.9 What the records do not yet contain

Keep the following information separately when you need it.

- Fitting model files and masks/cuts settings name no release. Sputter, AI assistant, command-line and 1D Predict
  run records name it (`software_version`); older records may lack it.
- The 2D Prediction record has no hash of the model weights, the YAML file or `preprocess.py`. It names the steps
  without their parameters (crop margins, thresholds), which stay in the module's YAML file and can change later;
  the copy handed to the SEM comparison carries them. Axes filled in by default (section 13.4) are not marked.
- The XRR record does not name the companion NeXus file from which a CBF series took its energy.
- The SEM record holds no summary statistics, because none are computed (section 13.2).
- The uint16 modes of the Format Converter write pixels without data as 65535; outside HDF5, which marks them in
  its `pixel_mask`, they read back as valid values (section 13.3).
- Older Analyze and Fitting records may lack the inputs and settings for replay. See User Manual §12.
- `<stem>_analysis.json` gives its time of creation in UTC (`created`), as the XRR and SEM records do; the 2D
  Prediction record gives local time without a time zone.

### 14.10 The record of a run without the window

`tools/gimap_agent.py auto` (and `run_standard_pipeline` of its MCP server, when it is given a folder) writes one
folder per frame and a summary of the batch. The record of a frame is `report.json`; it uses the keys of the
window's records:

| Group | Keys of `report.json` |
|---|---|
| Program and time | `software_version` (the release, section 14.5); `created` (UTC, ISO 8601, to the second; one time for every report of a batch and its `summary.json`) |
| Options | `options`: every option of the command line under its own name, as it was given (`frames`: the arguments before patterns and folders were expanded; `calibration`, `mask`, `flat_field`, `incidence_deg`, `technique`, `no_fit` …; `notes` and `notes_file`), and `notes_text`: the notes the analysis read. `null` when the files were written through MCP or `call` |
| Inputs | `inputs.frame`: every file the frame was read from (module files, a master with its data files) and every frame summed with it; `inputs.calibration`: the `.poni`, calibration `.json` or image of a standard the geometry was taken from (`null` for a saved instrument profile); `inputs.mask`, `inputs.background` and `inputs.flat_field`: the mask, the background and the flat-field file when one was used, else `null` (the flat field is also named among the Decisions, with the pixels it left out, section 4.10). Each file is `{"file": name, "path": the path as given, "bytes": size, "sha256": SHA-256}` |
| Mask | `mask`: `applied` (true or false), `file` (as above), `kind` (a mask image whose non-zero pixels are left out, or a GIMaP masks file with its number of drawn shapes), `masked_pixels`, and `bad_pixels_left_out` (section 14.4): the pixels found automatically are counted apart from the mask. `report.md` says the same under **Mask**, also when no mask was applied |
| Analyze's record | `analysis_record`: the record of section 14.2 for the frame as it was reduced (geometry, corrections, error model, binning, markers, `input_sha256`), without `files` |
| Structural replay | A completed run has a top-level `reproduction` recipe with the source commit, numerical environment, SHA-256 file references (including a supplied notes file), initial Analyze settings, pipeline options, phase definitions and CIF hashes, and measured and applied incidence and roll. A Predict fit records the hashes of its weights and network manifest, composition choices, sampling settings and seed. |
| Results | the decisions with their reasons (`decisions`), the open questions (`needs_attention`), the steps, the peaks and rings (GIWAXS) or `gisaxs` (cuts, symmetry axis, halves, length scale, fit), `tables` and `folder` (the frame's folder, relative to the summary); in `summary.json` each frame also has `outputs` (the files written) |
| Model fit | `gisaxs.fit.method`: `name` (`Predict parameters`), `network` (`name`, for example `flow_v1`, and `sha256`: the SHA-256 of the weights file, the value the network's record names and the loader checks) and `seed` (0, section 14.8). `gisaxs.fit.solutions` holds every solution with its model, parameters, probability, χ²ᵣ (`chi2`, with `points`: `fitted`, `given`, `below_window`, `unusable`), `flags`, `warnings` and `more_warnings` (the flags as sentences: of the particles and the solution, and of background, beam tail and resolution) and `intervals` (median, 68 % and 95 % limits; the texts print the refined value with the 68 % limits); `fitted_points` repeats the points of the fit; `curve`, `points` and `q_range_inv_angstrom` say what was fitted, `data` holds the fitted points; `substrate_level` and `sphere_model` record the approximation used. At level 2 (`Born term at the cut's q_z`), `cut_qz` holds the vacuum `q_z_inv_nm` and its inputs `alpha_i_deg`, `alpha_f_deg` and `wavelength_A`. Level 4 also records `substrate` (material or optical constants and any layers) and `grazing` (angles and wavelength in nm). When the file optics cannot be used, `substrate_metadata_warning` and the Decisions entry explain why. At level 1 (`effective interface (q_z = 0)`), `cut_qz` is `null`. Section 9.13 explains the q_z·R flags |

- `gimap reproduce report.json --out check` measures the original frames again and compares the reduction columns,
  the measured angles and the structural sections of the report: for GIWAXS the assignments, orientation verdicts,
  fitted parameters and unit cells; for GISAXS the fitted solutions with their evidence probabilities or BIC
  weights. Run times and relocated file paths are not compared. `comparison.json` gives a digest per section, and
  `reproduced_report.json` holds the values.
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
- `geometry` holds the values q was computed with, to 15 significant digits; `report.md` rounds them for reading. For GISAXS its `beam_center_px[0]` is
  the symmetry axis the run moved the centre to. `calibration` and `tables.geometry_used` hold the calibration as
  it was read and first applied, before that refinement (`gisaxs.symmetry` has both: `initial_x_px`, `x_px`, and
  `not_applied` when the axis was too far to be used). `geometry.critical_angle_deg` is `null` when no critical
  angle was given (q is kinematic). `tables` keeps the tools' answers as the
  AI saw them (four significant digits), except `geometry_used`. Every crystallite size names its unit:
  `tables.crystallite_sizes[].size_A`, `size_err_A`, `size_nm`, `size_err_nm`, beside `peaks[].size_nm` and `peaks[].size_err_nm`.
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

This chapter lists the limits of chapters 1 to 14 in one place. Each item appears once, under the first heading it
fits, and names the section that states it in full.

### 15.1 Scattering physics

- **Refraction and the DWBA.** Every q of Analyze, Fitting and 1D Predict is the kinematic vacuum value unless
  Analyze is given the film's critical angle (q_z of one film, section 2.6). The reflected-wave terms of the
  distorted-wave Born approximation are left out (sections 2.5, 8.6, 9.2), except in Fitting's optional two-path
  sphere with a free effective reflection coefficient (section 9.10), the four-term DWBA on a known bare or layered
  substrate for a horizontal cut (section 9.17) and the vertical cut of a fit of several curves (section 9.18);
  these models assume smooth interfaces. The Yoneda band is located empirically (sections 6.7, 8.1). GIWAXS
  intensity corrections and orientation fits can use refracted waves in a uniform film (sections 5.4, 7.11). The Sputter island height H(t) uses the refracted wavevector transfer for fringe minima (section 12.12). The GIWAXS
  and GISAXS reports say in their *Decisions* that q is kinematic and give the q_z shift of up to about k sin α_i
  that refraction causes near or below the film's critical angle; in GISAXS the Yoneda band shifts most (sections
  7.9, 8.6).
- **Absolute intensity.** A secondary standard sets the absolute scale for its own measurement conditions; in
  grazing incidence the footprint, illuminated volume and interface effects need a sample-specific treatment
  (section 4.12). Fitting's scale Int is an empirical amplitude, since the form factors use P(0) = 1 (section 9.4).
- **Lorentz weighting.** The optional detector Jacobian applies to integrated crystalline intensities (§5.6).
- **Texture.** The analysis assumes fibre symmetry. Its systematic range varies the angular cuts, background,
  attenuation and edge fills; a population concentrated beyond the measured angles can give f outside that range
  (§§7.7, 7.11).
- **Phases.** No phase is assigned. Lamellar and hexagonal series are reported as hints (section 7.4).
- **GISAXS models.** The 1D models of a horizontal cut depend on |q_y| only: q_z at the Yoneda row and the Yoneda
  enhancement are not modelled (sections 8.6, 9.2), except in the four-term DWBA with a known substrate (section
  9.17). A vertical cut is modelled only in a fit of several curves, for spheres and upright cylinders, at one
  in-plane q, without q resolution or the specular rod (section 9.18). The empirical beam tail is added after the
  horizontal particle terms are smeared (sections 8.6, 9.2, 9.13). Fitting and Predict parameters use sizes by
  number unless a component's Sizes says otherwise (sections 8.5, 9.3). Without a given composition the automatic
  analysis lets the network choose among one or two populations of spheres and upright cylinders; random cylinders
  and three populations are fitted only when given (section 8.5).
- **Cylinder height from a joint fit.** The bare-substrate DWBA leaves out the particle layer's effective
  refractive index and assumes independent radii and heights. Dense or high-Z layers can give heights that are too
  low, and radius–height correlation can bias the height even with a good fit (section 9.18).
- **Paracrystal spread.** Results of the quick fit and 1D Predict written before 0.2.0 give σ_D/D, √(2π) ≈ 2.51
  times smaller than the relative standard deviation of the neighbour distance σnn/D used now (section 9.5); the
  1D Predict window gives both.

### 15.2 Instrument and geometry

- **Detector tilt and rotation.** At SAXS distances rings constrain the tilt weakly (1σ of 0.15° for rot1 and
  0.18° for rot2 on the public AgBh frame) and leave rot3 undetermined; check rot3 against the observed GISAXS rod
  and horizon (section 3.7). The initial ring search uses circles. Automatic row bands warn when the horizon
  changes by more than half a row across the image.
- **Sample orientation.** A single image measures the sample roll relative to the calibrated detector axes, so an
  error in rot3 looks like sample roll. Ring symmetry gives the inclination of the fibre axis; reading it as sample
  roll needs an independently resolved surface feature or the stated assumption that the fibre axis is the surface
  normal (section 2.3).
- **Pixels.** A pixel is represented by its centre: there is no parallax, no sensor thickness and no pixel splitting (sections 2.5, 6.1).
- **Beam.** Divergence, energy bandwidth and footprint are not modelled (section 2.5).
- **Calibration.** Wavelength and pixel sizes are inputs. Calibrant constants are fixed without a temperature
  correction, and ring radii use a 1 px grid. A good fit to the calibrant image does not show that the calibration
  belongs to the sample's measurement. `.poni` orientations 1–4 are supported (section 3.6).
- **Instrumental broadening.** Run Automatic Analysis subtracts no broadening (instrument, footprint, divergence,
  distance, pixel size, binning, strain, paracrystalline disorder), so every automatic Scherrer size is a lower
  bound (section 7.8).

### 15.3 Statistics and uncertainties

Curve files give the statistical error of each point for a fixed absolute scale; the scale uncertainty from the
standard is in the calibration record. Data on the same scale share its error σ_A_abs of equation (4.19) (section
4.12).

- **Choice of the error model.** Poisson errors (scaled when a flat field or the GIWAXS intensity corrections are on) are used only for frames stored as integers, with no background subtracted and no mirror filling. The choice follows the stored type, not the physical meaning of the values (sections 4.1, 6.4).
- **Independence.** Pixels are taken as independent: charge sharing and correlated noise are not modelled, and no covariance between the points of a curve is computed (section 6.4).
- **Pixel scatter.** Real structure inside a bin inflates the pixel-scatter error; after mirror filling it is somewhat too small (section 6.4).
- **Maps.** q maps, the cake and the series maps carry no σ (sections 6.8, 11.1).
- **Fitted parameters.** All parameter errors are linearised (Wald) errors, not profile intervals. χ²_red only
  inflates the GIWAXS peak errors and scales the Fitting errors in both directions (sections 7.4, 9.8). σ_f of the
  Hermans factor propagates only the bin errors (Poisson counting errors or the pixel scatter), not the background,
  the window, the geometry or the bins next to the sample horizon (section 7.7).
- **Geometry in the uncertainties.** No reported ± of a peak position, a d-spacing, a size or f contains the uncertainty of the beam centre, the distance or α_i, or the q_z shift of refraction; the GIWAXS report says so in *Decisions* (section 7.9).
- **Quantities without an uncertainty.** The symmetry centre (section 8.2), the reference-peak position (section 6.10), the moment width of a peak in a window (section 11.2), stage boundaries, odd-frame flags and Compare groups (section 11.11), XRR points (section 13.1), 2D Prediction outputs (section 13.4).
- **Scores.** The scores of the automatic analysis and the 1D Predict window (χ²ᵣ, RMS/σ, logRMSE) are not
  probabilities, and mixture weights are not volume or mass fractions. The probabilities and intervals of Predict
  parameters hold within Fitting's model and its prior (section 9.13); its benchmarks are simulated curves of an
  independent model (BornAgain). The Sputter reliability numbers come from simulation within the hemisphere model and are exploratory (sections 12.5, 12.14).
- **SEM.** No summary statistic, distribution fit or test of agreement is computed (section 13.2).

### 15.4 Data preparation

- **Bad pixels.** Only single hot and dead pixels and scattered pixels of defective lines are found; clusters and the bad pixels of a background frame are not searched for (section 4.4).
- **Mirror filling** assumes a pattern symmetric in ±q_∥ (section 4.8).
- **Gaps.** Analyze never interpolates across gaps (sections 6.1, 11.1); Compare interpolates across a gap inside a frame (section 11.10).
- **Reference-peak corrections.** The peak position is limited by the bin width, and the distance scaling is a small-angle relation made exact by iteration (section 6.10).

### 15.5 Series and Sputter

- **Series.** Stages and odd frames are found in frame order, not in time; only positive intensities enter; a q bin with data in fewer than 95 % of the frames is not compared; *shape only* removes the overall level; the thresholds are working choices, not derived from a noise model of the detector (section 11.11).
- **Sputter.** The values are the analysis package's (section 12.1): GIMaP neither computes nor grades them; the rate and t_0 are not measured by GISAXS, and the nominal deposit scales with the rate given; the range of usual rates is a check of the unit, not a limit (section 12.2); the real-space picture places the grains at random (section 12.3). The package's model has no metal inside a polymer film (section 12.3), and its whole-scan fit can miss the thickness fringes of a continuous film on a polymer along the exit angle where the joint solution does not take the dense film of section 12.1 (and always without a joint fit, mode Quick); the report reads the film thickness from the fringes directly.

### 15.6 Tool windows and Labs

- **XRR Series Extractor.** No background subtraction, no footprint, monitor, polarisation, absorption, resolution
  or refraction correction, and no θ offset (section 13.1).
- **SEM Particle Analysis.** The scale comes only from the user. The default StarDist model is not validated for SEM. The circularity of a digitised disc is about 0.90, not 1. A SEM distribution and a scattering prediction are different quantities (section 13.2).
- **Format Converter.** The uint16 modes lose the marking of pixels without data outside HDF5 (section 13.3).
- **2D Prediction.** No geometry and no correction enter; the preprocessing is the module's own; axes that a module does not declare are filled in by default; the units of the SF parameters are not declared (section 13.4).

### 15.7 Records

What the records do not contain is listed in section 14.9.

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
- Cromer, D. T. & Mann, J. B. (1968). *Acta Cryst.* A**24**, 321–324. https://doi.org/10.1107/S0567739468000550
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
- Maier, W. & Saupe, A. (1958). *Z. Naturforsch. A*, **13**, 564–566. https://doi.org/10.1515/zna-1958-0716
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
- Prince, E. (Ed.) (2004). *International Tables for Crystallography*, Vol. C, *Mathematical, Physical and Chemical Tables*, 3rd ed. Dordrecht: Kluwer.
- Renaud, G., Lazzari, R. & Leroy, F. (2009). *Surf. Sci. Rep.* **64**, 255–380. https://doi.org/10.1016/j.surfrep.2009.07.002
- Robertson, T., Wright, F. T. & Dykstra, R. L. (1988). *Order Restricted Statistical Inference*. Chichester: Wiley.
- Ryan, C. G., Clayton, E., Griffin, W. L., Sie, S. H. & Cousens, D. R. (1988). *Nucl. Instrum. Methods Phys. Res. B*, **34**, 396–402. https://doi.org/10.1016/0168-583X(88)90063-8
- Savikhin, V., Steinrück, H.-G., Liang, R.-Z., Collins, B. A., Oosterhout, S. D., Beaujuge, P. M. & Toney, M. F. (2020). *J. Appl. Cryst.* **53**, 1108–1129. https://doi.org/10.1107/S1600576720005476
- Schrode, B., Pachmajer, S., Dohr, M., Röthel, C., Domke, J., Fritz, T., Resel, R. & Werzer, O. (2019). *J. Appl. Cryst.* **52**, 683–689. https://doi.org/10.1107/S1600576719004485
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
- Toney, M. F. & Brennan, S. (1989). *Phys. Rev. B*, **39**, 7963–7966. https://doi.org/10.1103/PhysRevB.39.7963
- Virtanen, P. et al. (2020). *Nat. Methods*, **17**, 261–272. https://doi.org/10.1038/s41592-019-0686-2
- Ward, J. H. (1963). *J. Am. Stat. Assoc.* **58**, 236–244. https://doi.org/10.1080/01621459.1963.10500845
- Wilkinson, M. D. et al. (2016). *Sci. Data*, **3**, 160018. https://doi.org/10.1038/sdata.2016.18
- Yoneda, Y. (1963). *Phys. Rev.* **131**, 2010–2013. https://doi.org/10.1103/PhysRev.131.2010
