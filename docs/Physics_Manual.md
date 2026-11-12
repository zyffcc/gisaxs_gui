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
- [3. Calibration](#3-calibration)
  - [3.1 Calibrants](#31-calibrants)
  - [3.2 Ring model](#32-ring-model)
  - [3.3 What the engine refines and how](#33-what-the-engine-refines-and-how)
  - [3.4 Quality measures](#34-quality-measures)
  - [3.5 From a calibration to an instrument profile](#35-from-a-calibration-to-an-instrument-profile)
  - [3.6 pyFAI `.poni` files](#36-pyfai-poni-files)
  - [3.7 What a `.poni` tilt left out costs in |q|](#37-what-a-poni-tilt-left-out-costs-in-q)
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
  - [8.5 The quick model fit and how the solution is chosen](#85-the-quick-model-fit-and-how-the-solution-is-chosen)
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
  - [9.9 Find the particle shape](#99-find-the-particle-shape)
- [10. 1D Predict (Workflow V5, conditional_v2)](#10-1d-predict-workflow-v5-conditional_v2)
  - [10.1 Input](#101-input)
  - [10.2 What the networks propose](#102-what-the-networks-propose)
  - [10.3 Forward model](#103-forward-model)
  - [10.4 Known composition and resolution: the conditional path](#104-known-composition-and-resolution-the-conditional-path)
  - [10.5 The damped Gauss–Newton refinement](#105-the-damped-gaussnewton-refinement)
  - [10.6 Unknown composition or resolution: the discovery path](#106-unknown-composition-or-resolution-the-discovery-path)
  - [10.7 Ranking and the solutions returned](#107-ranking-and-the-solutions-returned)
  - [10.8 Validation status](#108-validation-status)
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
  - [12.1 Inputs and the projection of the frames](#121-inputs-and-the-projection-of-the-frames)
  - [12.2 Time windows](#122-time-windows)
  - [12.3 Direct features of each window](#123-direct-features-of-each-window)
  - [12.4 The growth stage of a window: an observation rule](#124-the-growth-stage-of-a-window-an-observation-rule)
  - [12.5 The two networks](#125-the-two-networks)
  - [12.6 The physical path](#126-the-physical-path)
  - [12.7 Thickness substitutes](#127-thickness-substitutes)
  - [12.8 The intensity-modulation period](#128-the-intensity-modulation-period)
  - [12.9 Windows where the beam may be off the sample](#129-windows-where-the-beam-may-be-off-the-sample)
  - [12.10 The literature method (mass balance for hemispheres)](#1210-the-literature-method-mass-balance-for-hemispheres)
  - [12.11 Checks against published growth](#1211-checks-against-published-growth)
  - [12.12 Island height H(t) from moving q_z fringes](#1212-island-height-ht-from-moving-q_z-fringes)
  - [12.13 Grades](#1213-grades)
  - [12.14 Stated limits](#1214-stated-limits)
  - [12.15 Literature references of the method](#1215-literature-references-of-the-method)
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
| q* | Å⁻¹ (nm⁻¹ in Sputter) | position of the in-plane correlation peak of a GISAXS cut | sections 8.4, 12.3 |
| R | nm | radius of a particle or an island | sections 9.4, 12.5 |
| σ_R/R, σ_h/h, σ_D/D | – | relative spreads of R, h and D (the parameters labelled σ_R/R, σ_h/h and σ_D/D); σ_D is the absolute spread of D, and σ_D/D is GIMaP's paracrystal spread, defined in section 9.5 | sections 9.3, 9.5 |
| P(q), S(q) | – | form factor and structure factor | sections 9.4, 9.5 |
| φ | – | paracrystal factor exp(−π q² σ_D²) | (9.13) |
| δ, β | – | refractive-index decrement and absorption index, n = 1 − δ + iβ (written δ_X in chapter 12, where δ is the nominal thickness of the deposited metal) | sections 12.12, 13.5 |
| α_c | ° | critical angle of total external reflection | section 12.12 |
| t | s | time; t_exp is an exposure time | chapters 11, 12 |

### 0.2 Units

- **Analyze, Geometry Calibration and XRR Series Extractor** use q in Å⁻¹ and d in Å. The curves Analyze exports, `<stem>_fit_input.dat` included, are in Å⁻¹.
- **Fitting works in nm⁻¹ internally.** A curve read in Å⁻¹ is multiplied by 10 (1 Å⁻¹ = 10 nm⁻¹; 1 nm = 10 Å). Sizes and distances of the models are in nm. The quick physical fit converts in the same way (section 8.5).
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

GIMaP describes a measurement with seven numbers. The detector is flat and perpendicular to the direct beam. The sample surface is tilted by the grazing angle α_i.

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

1. The detector is flat and perpendicular to the direct beam. `L` is measured along the beam.
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

Examples from the public data: the GALAXI frame reaches `2θ_max` = 5.3° (GISAXS). The P08 GIWAXS frame (`giwaxs_p08_mapi`, PerkinElmer 2048 × 2048 px, `p` = 200 µm, `λ` = 0.6888 Å, `L` = 811.8 mm, direct beam (545.5, 1826.9) px from its `.poni`, `α_i` = 0.075°) reaches 30.2° (GIWAXS).

The choice decides the reduction (chapter 6). Mirror filling and the intensity corrections are applied only to GIWAXS frames.

Implementation: `src/gimap/features/analyze/domain/classification.py`

### 2.5 What the q map does not model

The following effects are not in the q map, nor anywhere else in Analyze:

- **Refraction and the distorted-wave Born approximation (DWBA).** `q_z` is the kinematic vacuum value. Neither α_i nor α_f is corrected for the critical angle. Near and below the critical angle the `q_z` inside the film differs from the value given here.
- **Detector tilt and rotation.** The model has no rotation parameters. A `.poni` tilt is folded into the beam centre and the distance and then left out (section 3.6); what that costs in |q| is computed for every frame and reported, and a tilt above 1° or an error above 0.5 % is a point to check of the automatic analysis (section 3.7). Omitted Rot3 is checked separately for direction, with its own review reminder above 1° in magnitude; the |q| error does not measure direction error. The calibration refines no tilt (section 3.3).
- **Sample azimuth and roll.** The sample is assumed to be tilted only about the horizontal axis perpendicular to the beam. A rotation about the surface normal or about the beam is not described.
- **Parallax and sensor thickness.** A pixel is its centre; there is no depth of interaction and no pixel splitting.
- **Flat field and efficiency.** No flat-field, detector-efficiency, air or window absorption correction is applied.
- **Absolute intensity.** Intensities are mean counts per pixel. They are not scaled by a monitor, the transmission, the exposure time or the sample volume. They are divided by a geometric factor only when the GIWAXS intensity corrections are on (chapter 5); a series can be normalised to a reference peak (section 6.10).
- **Beam properties.** Divergence, energy bandwidth and footprint are not modelled.

The flat, perpendicular detector is also assumed by the straight horizon (section 2.3), by mirror filling (section 4.8) and by the solid-angle and polarisation factors (chapter 5).

Implementation: `src/gimap/shared/geometry/q_mapping.py` (the only q mapping Analyze uses)

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
5. **Ranking.** A weighted score rewards matched rings, azimuthal coverage, valid area, centre support, an unbroken run of orders and prominence, and a Gaussian prior of 20 % width around an estimated distance. It penalises the residual, inconsistent per-ring distances, unexplained inner peaks and (for AgBH) a missing first order.
6. **Joint refinement.** For up to 6 groups of proposals with up to 3 distinct distances each, the beam centre and L are refined together. Ring points are the valid pixels within `h = clip(0.012 r, 3, 10)` px of each predicted ring whose preprocessed signal reaches the 70th percentile of the annulus, at most 800 per ring. The residual (3.8) is minimised with `scipy.optimize.least_squares` (soft-L1 loss, f_scale 2 px, at most 80 evaluations). The centre may move by 5 px (direct-beam proposal), by 40 to 140 px (arc voting) or by 12 to 60 px (other proposals); L may move by ±15 %. Only `x_c`, `y_c` and `L` are free.
7. **Re-matching.** At the refined centre the rings are detected again and matched at the refined L, and L is solved again with (3.7). **The distance and the residual that are reported come from this step**, not from the joint fit. On a synthetic AgBH frame (true L = 1600.0 mm, 4 rings) the joint fit gave 1599.96 mm and the reported value was 1599.85 mm; the centre was within 0.03 px of the true one.
8. **Result.** The best three candidates are kept, plus the best candidate of each other standard tried, up to five. The record holds the SHA-256, size and modification time of the image, the energy, λ, the pixel sizes, every candidate and the image shape. With fewer than two matched rings a warning says that the distance is ambiguous.

```text
L = Σ_m u_m r_m / Σ_m u_m²,     u_m = tan 2θ_m / p̄     (r_m: observed radii in px)     (3.7)
ρ_p = [ √( ((x_p − x_c) p_x)² + ((y_p − y_c) p_y)² ) − L tan 2θ(q_p) ] / p̄             (3.8)
```

In (3.8), `(x_p, y_p)` are the integer index coordinates of ring pixel p, `q_p` the line it was assigned to, and ρ_p is in pixels.

What is not refined: the wavelength, the pixel sizes, and any detector tilt or rotation. The field `detector_rotation_deg` of a candidate stays 0.

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
- The tilt itself and `rot_3` are left out of the geometry, but not forgotten: the instrument profile keeps the file's distance, PONI, rotations and pixel sizes (`dropped_tilt`), and section 3.7 separates the effects on |q| and on directional quantities. Above 0.5° the file's inspection also notes that the flat geometry makes q approximate far from the beam.
- A `.poni` is applied as it is: its pixel size and frame size are not checked against the data.

Examples from the public data: the P08 file gives (545.46, 1826.95) px, `L` = 811.77 mm, tilt 0.31°; the pyFAI AgBh file gives (179.98, 263.86) px, `L` = 1583.31 mm, tilt 0.57° (note); the Xeuss file gives (7.05, 589.49) px, `L` = 147.58 mm, tilt 6.06° (note).

Implementation: `src/gimap/features/assistant/domain/calibration_files.py`

### 3.7 What a `.poni` tilt left out costs in |q|

For every frame analysed with a profile that keeps a `.poni` tilt (section 3.6), Analyze compares |q| of the tilted geometry with |q| of the flat geometry made from the same `.poni` (its direct beam (3.10)–(3.11) and the distance L (3.12) to it), pixel by pixel. pyFAI's vector from the sample to the centre of pixel (row i, column j), with `p_1 = (i + ½) p_1 − poni_1`, `p_2 = (j + ½) p_2 − poni_2` (the pixel sizes `p_1`, `p_2` of the file), `c_k = cos rot_k` and `s_k = sin rot_k`, is

```text
t_1 = p_1 c_2 c_3 + p_2 (c_3 s_1 s_2 − c_1 s_3) − L_poni (c_1 c_3 s_2 + s_1 s_3)
t_2 = p_1 c_2 s_3 + p_2 (c_1 c_3 + s_1 s_2 s_3) − L_poni (−c_3 s_1 + c_1 s_2 s_3)            (3.14)
t_3 = p_1 s_2 − p_2 c_2 s_1 + L_poni c_1 c_2
```

with `t_3` along the beam. GIMaP's ray is `(L, X, Y)` with X and Y of (1.3) and (1.4). For a ray with the component `a` along the beam and the rest `b`, `2 sin²θ = b² / (R (R + a))` with `R = √(a² + b²)` (this form has no cancellation near the beam), and since `|q| = 4π sin θ / λ`:

```text
ε_p = sin θ_flat / sin θ_tilted − 1                                                         (3.15)
ε_max = max_p |ε_p|,      ε_rms = √( (1/N) Σ_p ε_p² )                                      (3.16)
```

- The sum runs over the frame's valid pixels (Analyze's mask, gaps and bad pixels left out) whose centre lies at least 10 px from the direct beam, where |q| tends to zero and a relative error says nothing.
- ε depends neither on λ nor on α_i. A rotation `rot_3` about the beam leaves `t_3` and `t_1² + t_2²` unchanged, so it does not change |q|; it turns χ by `rot_3`, which ε does not measure.
- The omitted beam-axis rotation is reported independently: nonzero Rot3 changes the directional assignment
  of qy/qz and in-plane/out-of-plane cuts even when ε is zero. Its signed angle is reduced modulo 360° to
  [−180°, 180°]; residues no larger than 10⁻¹⁰° are treated as zero solely to suppress floating-point roundoff
  for full turns. The three original angles remain recorded. Every nonzero reduced angle is explained in
  Analyze and the report; above **1° in magnitude** it is a separate *Needs attention* item (CLI exit code 2),
  otherwise a *Decision*. This review threshold is not an orientation-accuracy tolerance and does not block
  reduction. A pure Rot3 = 30° has ε ≈ 0 while its transverse ray direction turns by 30°; |q| error cannot bound
  the error of directional quantities. Use the full `.poni` geometry to check those quantities.
- The report's `geometry.flat_detector` stores `dropped_beam_axis_rotation_deg`, `q_needs_attention` and
  `direction_needs_attention`; its `needs_attention` is the logical OR of the two flags. The first flag retains
  the normal-tilt/|q| thresholds below, and the second uses only Rot3. `limits.beam_axis_rotation_deg` is 1.0 and
  `limits.rotation_zero_tolerance_deg` is 10⁻¹⁰. With no valid pixels the |q| errors are `null`, while the rotation
  and its direction flag remain known. No rotation correction is applied to the scientific arrays.
- The comparison is with the flat geometry of the `.poni`, not with the geometry in use, so ε is what the tilt alone costs: a beam centre moved afterwards (the GISAXS symmetry axis of the automatic analysis, section 8.2, or a centre set by hand) or another distance is no part of it, and the 10 px are counted from the `.poni`'s direct beam. On the public AgBh frame the symmetry axis moves the centre by −0.39 px; ε_max stays 0.08 %, where a comparison with the moved centre would read 4.1 %. The report's tilt line gives the estimate of the frame it reports (frames summed since the geometry was applied change the valid pixels).
- Above **1°** of tilt or **0.5 %** in ε_max, the frame gets a message (Analyze's status line and its Geometry step, marked with a warning; `messages` of the record), and the automatic analysis lists the tilt under *Needs attention* (`tools/gimap_agent.py` exits with code 2). Below both, its *Decisions* say how much ε_max is. The tilt, the three rotations, ε_max, ε_rms, the number of pixels compared and the limits are written into `<stem>_analysis.json` (`geometry`, section 14.2) and `report.json` (`geometry.flat_detector`).
- A test checks (3.15) on a hand-computed case: rot_1 = 60°, `L_poni` = 0.05 m, so the direct beam lies L = 0.1 m from the sample; a pixel 0.1 m from the beam along the row is at 2θ = 75° on the tilted detector and 45° on the flat one, ε = sin 22.5° / sin 37.5° − 1 = −0.3714.

Examples from the public data: the P08 file (tilt 0.31°) gives ε_max = 0.20 % (a decision), the Xeuss file (tilt 6.06°) ε_max = 6.3 % and ε_rms = 4.2 % on the 278 518 valid pixels compared (a point to check). The largest error lies at the corner farthest from the beam.

Implementation: `src/gimap/shared/geometry/detector_tilt.py`, `src/gimap/features/analyze/application/flat_detector.py`, `src/gimap/features/assistant/application/pipeline_geometry.py`

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
- They are applied after all other preparation steps (section 4.9) and need the geometry. X, Y and L_p are those of (1.3), (1.4) and (2.2), and `cos 2θ = L/L_p` (2.12).
- Comparable geometric corrections are part of other grazing-incidence packages, for example GIXSGUI (Jiang, 2015).

Implementation: `src/gimap/features/analyze/domain/intensity.py`, `src/gimap/features/analyze/application/frame_preprocessing.py`

### 5.2 Solid angle

A flat pixel perpendicular to the beam at distance L_p from the sample subtends

```text
Ω_p = p_x p_y L / L_p³ = (p_x p_y / L²) cos³ 2θ                                   (5.2)
F_Ω = cos³ 2θ                                                                   (5.3)
```

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

**GISAXS q map.** A 600 × 600 grid in `q_y`–`q_z` over the range of the usable pixels (section 6.8).

Implementation: `src/gimap/features/analyze/domain/gisaxs.py`, `src/gimap/features/analyze/domain/binning.py`, `src/gimap/features/analyze/domain/symmetry.py`, `src/gimap/features/analyze/domain/symmetry_coarse.py`

### 6.8 The q map and the cake

Both are means of pixel values on a regular grid:

```text
cell(p) = ( ⌊N_y (y_p − y_0)/(y_1 − y_0)⌋ , ⌊N_x (x_p − x_0)/(x_1 − x_0)⌋ ),    value = mean of I_p in the cell      (6.15)
```

| Map | x | y | Grid | Range |
|---|---|---|---|---|
| GISAXS q map | `q_y` | `q_z` | 600 × 600 | min to max of the usable pixels |
| GIWAXS q map | `q_∥` | `q_z` | 800 × 800 | min to max of the usable pixels |
| Cake | q | χ | 800 (q) × 360 (χ, 0.5° each) | q from 0 to the largest q; χ from −90° to +90° |

- Only usable pixels (valid and above the horizon) enter.
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

- One half gives the |q| points of the mean. The prepared cut of Run Automatic Analysis and of the quick physical fit, and the Series map, use the `q_y > 0` half. Fitting uses the half with more points (the `q_y > 0` half when both have as many). A half with fewer than two points never gives the grid.
- The tilde marks linear interpolation of the other half, and of its σ, at that |q|.
- Points are averaged only where the other half has measured points on both sides without a hole, or a point at the same |q| (within 1 % of a step). A hole is an interval wider than 2.5 times the local step, the median of the 9 surrounding intervals.
- Elsewhere the measured half stands alone with its own σ. Points of the other half that lie in a hole of this half, or beyond its ends, are added as measured. Nothing is interpolated across a hole.

**Automatic choice** (Run Automatic Analysis and the quick physical fit). Only points with finite values, `I > 0` and at least one pixel are used. For each half GIMaP takes:

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
| fit_failed | no usable profile fit | caveat |

A peak with a caveat (step_edges, spike, fit_failed, weak or broad, in this order of precedence) is listed in the
report with the reason, but it is not used for ring selection, sizes or series hints. Neither is a peak flagged
at_edge, whose shape, position and width the end of the searched range may cut off: the report lists it as "at the
end of the data: check". The peaks left, without a caveat and not at the edge, are the *reliable* peaks.

**Series hints.** From each of the three lowest-q reliable peaks, GIMaP looks for the orders of a lamellar
series (1 : 2 : 3 : 4 : 5) and of a hexagonal series (1 : √3 : 2 : √7 : 3), each order matched within 2 % of its
expected q. A series is reported when at least three orders (the first included) match. This is a hint; no phase is
assigned.

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
  frame at α_i = 0.075°, λ = 0.6888 Å), or says that it is unknown while α_i is 0° for want of a value;
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
or shoulder (a length the curve shows, not necessarily the particle spacing), and fits the prepared cut with a quick
numerical model. It runs from **Run Automatic Analysis** in
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
probability. A shift of more than 20 px from the calibrated centre is put under *Needs attention*.

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
(|qy| = … Å⁻¹)*, with the reason *a length scale, not necessarily the particle spacing; check it with a model fit,
which starts one of its distances D from it*. If there is none, the strongest broad or fit_failed feature with
SNR ≥ 5 is reported as a **shoulder** (a change of slope, not a resolved peak): *… at a shoulder of the cut (…), no
resolved peak*, with the same reason and *a form factor or the beam tail makes a shoulder too*. The 0.15 q_0 "broad" limit of GIWAXS is reused, so a wide correlation peak is often classed a shoulder.
Example (invented): q* = 0.020 Å⁻¹ gives D* = 314 Å = 31.4 nm.

**Why D\* is not called a spacing.** A side maximum of I(q_y) can come from the interparticle correlation, but the
knee of a form factor and the tail of the direct beam make a shoulder too, and with the paracrystal model of chapter 9
the fitted D differs from 2π/q* when the disorder is large. On BornAgain 24.1 simulations of BornAgain's documented
model of the GALAXI frame, placed on the measured q grid, the search found a shoulder at 2π/q* = 73.9 nm for a peak
distance of 53.6 nm, and a shoulder at 60.6 nm for the same particles without interference. On 14 simulated cuts with
known parameters, 2π/q* was within 10 % of the true distance in 1 of 10 cuts with interference, and a maximum or
shoulder was found on all 4 cuts without interference. D* is therefore only a start for the distance D of the fit
(section 8.5). The records keep its value under the earlier key (`spacing.distance_nm`).

Implementation: `src/gimap/features/assistant/domain/gisaxs_cut.py`, `src/gimap/features/assistant/application/gisaxs_tools.py`

### 8.5 The quick model fit and how the solution is chosen

The prepared curve is fitted by a numerical variable-projection fit (no neural network). The same fit is Fitting's
**Find the particle shape (no AI)** (section 9.9) and the *Physical fit (numerical)* method of 1D Predict.

**Model.** With q = 10 `abs(q_y)` in nm⁻¹:

```text
I(q) = BG + Σ_j a_j P_j(q) S_j(q) + A / ( 1 + (q / w)^ν )                                      (8.6)
```

P_j is a sphere, random-cylinder or vertical-cylinder form factor with a Gaussian size distribution, and S_j the
paracrystal structure factor; the formulas and the size sampling are those of chapter 9, with one difference: the
vertical cylinder is **normalised**, `P = Σ_j w_j (2 J_1(q R_j)/(q R_j))²` (26 nodes, ±3σ), so every family has P(0) = 1.
Every component carries a structure factor. The sphere series threshold differs slightly from Fitting's (x below 0.1,
up to x⁶).

**Families and starts.** Without a given composition (as in the automatic analysis) three single-component families
are screened: sphere, vertical cylinder, random cylinder. Each family is started from R = 1.5 nm and R = 4.0 nm, and,
when D* of section 8.4 is known, from R = 4.0 nm with D = D* (maximum or shoulder alike): 6 or 9 fits. A given composition is
fitted whole; component j starts at R (1 + 0.6 j). Other starts: σ_R/R = 0.2, σ_D/D = 0.12, h = 10 nm, σ_h/h = 0.2,
w = 0.02 nm⁻¹, ν = 2.5, D = 1.5 D_min when no D* is given.

**Parameters and bounds.**

```text
D = D_min · (500 nm / D_min)^η,   η ∈ [0, 1],   D_min = max(3 nm, 2.002 R)                       (8.7)
```

| Parameter | Range |
|---|---|
| R (fitted as ln R) | 1–100 nm |
| σ_R/R | 0.02–0.9 |
| D | from 2.002 R (and 3 nm) to 500 nm, through η |
| σ_D/D | 0.05–0.9 |
| h (random cylinder, fitted as ln h) | 2–500 nm |
| σ_h/h | 0.02–0.9 |
| w | 0.001–0.1 nm⁻¹ |
| ν | 1–20 |

(8.7) enforces the hard-core condition D ≥ 2.002 R.

**Fit.** At every evaluation the amplitudes a_j, BG and A are solved by non-negative least squares on the σ-weighted
design matrix, with its columns normalised. The shape parameters are fitted by bounded least squares
(`scipy.optimize.least_squares`, trust-region reflective) with a soft-L1 loss on

```text
r_i = ( I_model(q_i) − I_i ) / σ_i,   cost = ½ Σ_i 2 ( √(1 + r_i²) − 1 )                          (8.8)
```

with at most 150 function evaluations per start (plus those of the finite-difference Jacobian, relative step 10⁻⁴)
and tolerances 10⁻⁷.

**Scores.** For each solution:

```text
M_H = (1/N) Σ_i Huber_2( abs(r_i) ) = (1/N) Σ_i { ½ r_i² if abs(r_i) ≤ 2;  2 (abs(r_i) − 1) otherwise }   (8.9)
χ²/N = (1/N) Σ_i r_i²                                                                         (8.10)
logRMSE = √( mean over I_i > 0 of ( ln(I_model,i / I_i) )² )                                      (8.11)
```

M_H is the robust misfit of the program, the mean Huber loss (δ = 2) of the σ-normalised residuals (Huber, 1964). χ²/N has no degrees-of-freedom correction and is not Fitting's χ²_red (9.23).

**Ranking.** The solutions are sorted by **M_H, not by χ²**. A solution of the same composition whose
parameters, amplitudes and global values all lie within 1 % (relative) of a better one is a duplicate. At most 5
distinct solutions are kept (8 in the 1D Predict window). The fit needs at least 8 points.

**Values at a bound.** A fitted value closer to a bound of its range than 10⁻⁴ of the range (in the fitted variable:
ln R, σ_R/R, η, σ_D/D, ln h, σ_h/h, w, ν; the reports call w and ν σ_Res and ν_Res) has *ended on the bound*. The fit
names each such value in `at_bounds` (parameter, lower or upper bound, component), with the warning that values may
be unidentifiable. η = 1 is D = 500 nm,
η = 0 is D = D_min (contact). A value on a bound is not determined by the curve: the bound, not the data, holds it.
These are the non-linear values only. The linear terms a_j, BG and A are solved by non-negative least squares and are
not in `at_bounds`: one at 0 is held by that limit (a background the curve would want below zero, a beam tail that is
absent, so that w and ν no longer change the curve), and the report names it for the reported solution among the
*Decisions* (*fit caveat*: *background BG at 0, the non-negativity limit*) without counting it in the choice below.
On the GALAXI frame the reported sphere has BG = 0.

**The solution the automatic analysis reports.** Let χ²_1/N be that of the first-ranked solution. Among the solutions
with χ²/N ≤ 1.1 χ²_1/N (the first included), which the curve does not tell apart (the same 10 % as the model rule
below), the report takes the **best-ranked one that converged with no non-linear value at a bound**; when there is none, it
takes the first. This is the minimal rule that avoids reporting a bound without hiding the ranking: it never
reaches beyond the solutions the curve cannot distinguish, it never reorders them, and the table keeps every solution
with its rank. The fit record keeps the index of the reported solution (`chosen`); the table marks it (*chosen*),
names the values at a bound of every solution (*D at its upper bound*), the Results tab selects it, and `fit_curve.csv`
holds its curve. On the GALAXI frame the first-ranked solution, a random cylinder with χ²/N = 1.26, has D at its upper
bound (500 nm) and σ_h/h at its upper bound; the second (χ²/N = 1.28) has σ_h/h at its lower bound; the third, a sphere
with χ²/N = 1.36 ≤ 1.1 × 1.26, has none and is reported. The rule is the automatic analysis' own: Fitting's
*Find the particle shape (no AI)* (section 9.9) ranks by its own χ²_red. The report's *Decisions* state which solution
was reported and why. Further rules of the report:

- When the reported solution is the first and has a value at a bound (no close solution is free of one), *Needs
  attention* says so (**fit**), and the *Decisions* name the values at a bound.
- D ≥ 250 nm (half of the 500 nm bound) in the reported solution: it has no interparticle correlation in the q range,
  and D is not determined by the curve.
- Among the kept solutions, solutions of more than one family whose χ²/N is at most 1.1 times that of the first: the
  curve alone does not decide the model (*Needs attention*). The point says which solution is reported when it is not
  the first, and names those whose D is within 15 % of D* (two lengths that happen to agree, not evidence for the
  model, since D* is not necessarily a spacing, section 8.4).
- A reported solution that did not converge (budget reached) is put under *Needs attention*.
- A reported solution that converged with χ²/N > 10 is a **poor fit** (*Needs attention*, **fit**): it misses the
  curve by far more than the σ of its points, so none of the families describes the cut and its R and D are not
  measurements. The limit is that of Fitting's poor fits of a series (χ²_red > 10; User Manual, section 6.6), which
  differs from χ²/N only by the degrees-of-freedom correction. On the public AgBh SAXS frame (Bragg rings of a
  calibrant, no particles) the reported sphere has χ²/N = 472.

Every number of these texts is formatted once from the fit record (R, h, D to 4 decimals of nm), so a value reads the
same in the table and in *Needs attention* (48.3484 nm is 48.3 nm in both).

Each solution also carries warnings: a shape or resolution value closer to a bound than 10⁻⁴ of its range, the
budget reached, a component with an amplitude below 10⁻⁶ of the total. Its "weight" `a_j / Σ a` is a normalised amplitude,
not a probability or a volume fraction.

**A screening step.** The quick fit describes the cut and gives Fitting a start; it does not measure sizes. On 14
BornAgain simulations of cuts of Ag spheres with known parameters (the GALAXI model with R, scale, D and disorder
varied, Poisson noise at the data's count level), it ranked a random cylinder first in 12, a vertical cylinder in 1 and
a sphere in 1; 10 first-ranked solutions had a value at a bound; the best sphere solution's R was a median 11 % above
the R⁶-weighted mean radius of the simulation (−29 % to +26 %); D and the disorder were not recovered. The model
evaluates P at q_z = 0 (section 8.6), which is part of this bias. Confirm the model in Fitting (User Manual,
chapter 13).

Implementation: `src/gimap/features/fitting/infrastructure/adapters/experimental_fit.py`, entry
`create_quick_fit()` in `src/gimap/features/fitting/bootstrap.py`, forward model
`modules/Fitting_1D_Model/Workflow_v5/conditional_v2/numpy_forward64.py`

### 8.6 What the GISAXS analysis does not model

- The cut is treated as a function of `abs(q_y)` only. q_z at the Yoneda row, reflection and refraction (DWBA) terms
  and the Yoneda enhancement itself are not modelled.
- The term `A / (1 + (q/w)^ν)` is an additive empirical tail of the direct and reflected beam; no resolution
  convolution is applied.
- The Yoneda row is found without αc; the beam-centre row is not refined; the centre has no uncertainty.
- Only single-component families are screened automatically, so a mixture is never proposed without a given
  composition.

Like the GIWAXS report (section 7.9), the GISAXS report's *Decisions* always say what its numbers are:

- *q*: kinematic, no refraction correction. Near the critical angle refraction mostly shifts q_z, that of the Yoneda
  band (α_f ≈ α_c) most, by an amount of the order of k sin α_i, and the line gives that number for the frame
  (`2π/λ · sin α_i`; 0.0379 Å⁻¹ for the public GALAXI frame at α_i = 0.463°, λ = 1.34 Å); q_y and 2π/q* barely change;
- *uncertainties*: statistical only — the σ of the cuts' points, by which χ²/N is normalised, with the error model of
  the frame (Poisson counting statistics, or the pixel scatter of each bin); the geometry and refraction are not
  included, and the quick fit gives no ± for R, h or D.

## 9. Model fitting of 1D curves (Fitting)

Fitting fits one curve, or every frame of a series, with its own model. This chapter describes the model and the
numerical methods *Refine the values in Model* (Refine) and *Search the ranges, then refine* (Search) of the Fit step,
and *Find the particle shape (no AI)*. *Predict parameters (fit1d)* uses the fit1d kit that comes with GIMaP
(`utils/gisaxs1d_gui_kit`), whose internals are not described here; *AI proposal (1D Predict)* is chapter 10.

**Units: Fitting works in nm⁻¹ internally.** A curve read in Å⁻¹ (an Analyze `_fit_input.dat`) is multiplied by 10.
Sizes and distances are in nm, w in nm⁻¹, the spreads σ_R/R, σ_h/h, σ_D/D are relative.

**fit1d cut qz** is a user-supplied vacuum value in nm⁻¹, associated with the curve being fitted.
Opening a curve clears it; a project or session restores the value explicitly saved for its curve.
Analyze's finite-width horizontal band is not automatically replaced by a single inferred qz.
This changes the ownership and restoration of the input, not the model equations.

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
I(q) = BG + k · [ Σ_i Int_i · P_i(q) · S_i(q) + A / ( 1 + (q / w)^ν ) ],   q = abs(q) in nm⁻¹        (9.1)
```

Int_i is the scale of component i, BG a constant background (not multiplied by k), A, w and ν the amplitude, width
and exponent of the beam-tail term, and k an overall factor fixed at 1 by default. The model depends on `abs(q)` only;
q_z and DWBA terms are not modelled. The beam-tail term is additive, not a resolution convolution (a Gaussian smearing
function exists in the code but the model does not use it). With w ≤ 0 or ν ≤ 0 the term is 1.

Implementation: `src/gimap/features/fitting/domain/fit_model.py`

### 9.3 Size distributions

Every size average is a quadrature over a Gaussian of mean μ and absolute width σ = μ · (σμ/μ), truncated at 0:

```text
x_j = linspace( max(0, μ − n_σ σ), μ + n_σ σ, n ),   j = 1 … n                                   (9.2)
w_j = exp( −½ ((x_j − μ)/σ)² ) / Σ_l exp( −½ ((x_l − μ)/σ)² )                                    (9.3)
P(q) = Σ_j w_j P(q; x_j)                                                                         (9.4)
```

With σ = 0 a single node at μ is used. The weights sum to 1 over the nodes. The nodes are equally spaced, so (9.4)
is a rectangle rule over the truncated Gaussian.

| Family | Variable | Nodes n | Range n_σ |
|---|---|---|---|
| Sphere | R | 25 | ±4σ |
| Random cylinder | R and h, averaged separately | 13 each | ±4σ |
| Vertical cylinder | R | 26 | ±3σ |

Implementation: `src/gimap/features/fitting/domain/scattering_model.py`

### 9.4 Form factors as coded

**Sphere** (amplitude normalised to F(0) = 1; series `1 − x²/10 + x⁴/280` for x below 10⁻⁶):

```text
F_s(x) = 3 ( sin x − x cos x ) / x³,   x = q R                                                      (9.5)
P_sphere(q) = Σ_j w_j F_s(q R_j)²                                                                   (9.6)
```

**Random cylinder** (homogeneous cylinder of radius R and height h, isotropic orientation):

```text
F_c(q, R, h, α) = [ 2 J_1(q R sin α) / (q R sin α) ] · sinc( q h cos α / 2 ),   sinc(u) = sin u / u     (9.7)
α_a = (a − 1) · (π/2) / 23,   ω_a = sin α_a / Σ_b sin α_b,   a = 1 … 24                              (9.8)
P_cyl(q) = Σ_a ω_a · [ Σ_j w_j (2 J_1(q R_j sin α_a)/(q R_j sin α_a))² ] · [ Σ_l v_l sinc²(q h_l cos α_a / 2) ]   (9.9)
```

w_j and v_l are the radius and height weights (9.3). Because the square of (9.7) is separable, the radius and height
averages are taken separately for each α. The orientation average is a 24-point rectangle rule with sin α weights on
[0, π/2]; its first node (α = 0) has weight 0. This coarse quadrature is part of the numerical definition of the
model.

**Vertical cylinder** at q_z = 0 (cylinders standing on the substrate, q in the plane):

```text
P_vc(q) = 10⁻⁶ Σ_j w_j [ R_j J_1(q R_j) / q ]²  =  10⁻⁶ Σ_j w_j (R_j⁴ / 4) (2 J_1(q R_j)/(q R_j))²      (9.10)
P_vc(0) = 10⁻⁶ Σ_j w_j R_j⁴ / 4      (R in nm)                                                       (9.11)
```

**Normalisation differs between the families.** The sphere and the random cylinder have P(0) = 1, so every node of the
size distribution contributes the same forward intensity: the Gaussian describes an intensity-weighted (V²-weighted)
size distribution, and Int is not proportional to number density × V² × Δρ². The vertical cylinder is not normalised:
each radius is weighted by R⁴ (cross-section squared at a fixed height), scaled by 10⁻⁶. Example: R = 5 nm gives
P_vc(0) = 1.56 · 10⁻⁴ for σ_R/R = 0 and 1.94 · 10⁻⁴ for σ_R/R = 0.2. (The tip of the Scale parameter says so: the form
factor is 1 at q = 0 for spheres and random cylinders, and 10⁻⁶ ⟨R⁴⟩/4 for vertical cylinders.) The quick fit of section 8.5 and 1D Predict use a normalised vertical
cylinder; see section 9.9 for the conversion.

Implementation: `src/gimap/features/fitting/domain/scattering_model.py`, `src/gimap/features/fitting/domain/fit_model.py`

### 9.5 Structure factor: one-dimensional paracrystal

For a component whose structure factor is switched on, with `D > 0` and `σ_D/D > 0`:

```text
S(q) = ( 1 − φ² ) / ( 1 + φ² − 2 φ cos(q D) )                                                    (9.12)
φ = exp( −π q² σ_D² )                                                                            (9.13)
```

σ_D is the absolute spread in nm: the parameter σ_D/D times D. The denominator is limited to at least 10⁻¹⁵. Otherwise S = 1.

**Relation to the usual form.** The one-dimensional paracrystal with a Gaussian distribution of nearest-neighbour
distances of mean D and standard deviation σ_nn (Hosemann & Bagchi, 1962; for GISAXS see Renaud et al., 2009) has the
same expression with `φ = exp(−q² σ_nn² / 2)`. The two are identical when

```text
σ_nn = √(2π) · σ_D,   i.e.   (σ_D/D)_GIMaP = σ_nn / ( √(2π) D )                                     (9.14)
```

(numerically identical within 2 · 10⁻¹⁶). **GIMaP's σ_D/D is therefore not the relative standard deviation of the
neighbour distance**; that is √(2π) ≈ 2.507 times larger. The low-q limit is `S(0) = 2π (σ_D/D)² = (σ_nn/D)²`. Example
(invented): D = 20 nm and σ_D/D = 0.2 mean σ_D = 4 nm, σ_nn = 10.0 nm and S(0) = 0.251.

No hard-core condition D ≥ 2R is applied in Refine or Search; the quick fit and 1D Predict enforce D ≥ 2.002 R.

Implementation: `src/gimap/features/fitting/domain/scattering_model.py`

### 9.6 Parameters and defaults

| Key | Label | Unit | Default | Range | Free |
|---|---|---|---|---|---|
| Int | Scale | – | 1 | ≥ 0 (solved, section 9.7) | yes |
| R | R | nm | 5 | 0.3–500 | yes |
| sigma_R | σR/R | – | 0.2 | 0.01–1.0 | yes |
| h | h | nm | 10 | 0.5–1000 | yes |
| sigma_h | σh/h | – | 0.2 | 0.01–1.0 | yes |
| D | D | nm | 30 (a new component: max(3R, 1)) | 1–3000 | yes |
| sigma_D | σD/D | – | 0.3 | 0.02–1.0 | yes |
| background | Background (BG) | – | 0 | ≥ 0 (solved) | yes |
| res_amplitude | Peak A | – | 0 | ≥ 0 (solved) | yes |
| res_width | Peak w | nm⁻¹ | 0.02 | 0.001–0.1 | yes |
| res_exponent | Peak ν | – | 3 | 1–20 | yes |
| k | Factor k | – | 1 | 10⁻⁶–10⁶ | no |

Each component has Int, R, σ_R/R (h and σ_h/h for random cylinders) and, with its structure factor, D and σ_D/D.

Implementation: `src/gimap/features/fitting/domain/fit_model.py`

### 9.7 Objective and the scales

**Residuals.**

```text
r_i = ( I_i − m_i ) / σ_i                         with σ                                          (9.15)
r_i = ln I_i − ln max( m_i, 10⁻³⁰⁰ )               without σ (each point the same relative weight)    (9.16)
```

m_i is the model (9.1). The fit minimises Σ r_i².

**Scales by non-negative least squares (variable projection).** The intensity is linear in Int_i, BG and A. Those of
them that are free are eliminated at every evaluation of the model (Golub & Pereyra, 1973). With the non-linear values
set, the columns `k P_i S_i`, `1` and `k / (1 + (q/w)^ν)` form the matrix X, the fixed terms the vector m_fixed, and

```text
c = argmin_{c ≥ 0} ‖ W ( X c − (I − m_fixed) ) ‖²,   W = diag(1/σ_i), or diag(1/I_i) without σ        (9.17)
```

is solved by NNLS (Lawson & Hanson, 1974; `scipy.optimize.nnls`). The optimiser therefore moves only the sizes,
spreads, distances and the width and exponent of the beam-tail term. Without σ, (9.17) minimises relative residuals
while the outer objective is (9.16): an approximation of the exact profile. NNLS enforces only c ≥ 0; the scales have
no range in the interface.

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

**Reduced χ².** With N points and p free parameters, including the scales solved by NNLS:

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
(log-)unit range, and a scale at 0, is listed as at a bound; its error is not meaningful.

Implementation: `src/gimap/features/fitting/domain/fit_engine.py`

**Values at the edge of their range.** With *Resolve values at the edge of their range automatically* (on by
default), after Refine or Search a size, distance, or beam-tail width or exponent at an edge gets that side of its range widened
(× 2 upwards, ÷ 2 downwards, the latter only for a positive lower limit), and the model is refined again; the change
is kept when `χ²_red,new ≤ 1.02 χ²_red,old`, for at most 2 rounds. Spreads, scales and k are not widened. A value still at an
edge (at most three of them) is then walked inward with the others refitted: sizes, the width and the exponent by a
factor 1.25 per step, spreads by 0.05, at most 8 steps, while

```text
Σ r² (trial) − Σ r² (best) ≤ max( 1, χ²_red )                                                         (9.27)
```

It is kept at the least extreme value allowed, fixed in Model, and reported as a one-sided bound (≥ or ≤). The
tolerance is Δχ² = 1 for one parameter, scaled by χ²_red when that exceeds 1.

Implementation: `src/gimap/features/fitting/domain/fit_edges.py`

### 9.9 Find the particle shape

*Find the particle shape (no AI)* runs the quick fit of section 8.5 on the points of section 9.1 (q passed in Å⁻¹ and converted back
to nm⁻¹; without σ it uses `σ = √((0.1 I)² + (0.001 max abs(I))²)`). *Try every family* screens sphere, vertical cylinder
and random cylinder as single components; *The families in Model* fits the composition of Model as a whole. Each
family is started from several points: R = 1.5 nm and R = 4.0 nm, and R = 4.0 nm with the Model's D when the Model has
been edited and its first component has a distance. At most 5 distinct solutions are returned.

Each solution is converted into Fitting's model. Sizes are taken over, the spreads stay relative, and every Int is
solved again by NNLS against the solution's own fitted curve with relative residuals, in two trials (with the
solution's BG and A, or with all linear terms refitted); the closer trial is kept, and its largest relative deviation
`max abs(m_GIMaP / m_solution − 1)` is reported. For spheres and random cylinders the forms agree and the conversion is
exact up to rounding. For vertical cylinders the R⁴ weighting of (9.10) makes the curves differ when `σ_R/R > 0`, and the
deviation says by how much. The converted solutions are scored with Fitting's residuals (9.15–9.16) and
χ²_red (9.23) and **sorted by that χ²_red** (the automatic analysis of chapter 8 sorts the same fit by M_H). The best goes
into Model, and Refine is proposed next.

Implementation: `src/gimap/features/fitting/presentation/single/run.py`, `src/gimap/features/fitting/domain/native_solution.py`

## 10. 1D Predict (Workflow V5, conditional_v2)

1D Predict is an experimental neural workflow that proposes compositions and parameters for a 1D GISAXS curve and
corrects each proposal numerically against the measured points. It is used by **Tools ▸ 1D Predict — Fit Many
Curves…** (method *General V5 (experimental)*, the default) and by Fitting's *AI proposal (1D Predict)*. The research
code and the trained weights are in `modules/Fitting_1D_Model/Workflow_v5/conditional_v2/`; GIMaP runs them in a
separate, cancellable process (TensorFlow is imported only there). Its solutions are starts to check and refine, not
results.

Units: q and σ_res in nm⁻¹; R, h, D in nm; σ_R/R, σ_h/h, σ_D/D relative (absolute widths are also given).

Implementation: `src/gimap/features/fitting/infrastructure/adapters/workflow_v5.py`,
`modules/Fitting_1D_Model/Workflow_v5/conditional_v2/native_observation_adapter.py`

### 10.1 Input

Each side of a signed curve is fitted separately (no folding). A side needs 8–1000 points with strictly increasing
`q > 0`; q = 0 is dropped, nothing is interpolated or invented. q in Å⁻¹ is multiplied by 10. Without σ:

```text
σ_i = √( (0.1 I_i)² + (0.001 max abs(I))² )                                                          (10.1)
```

(at least 10⁻¹²). The network backend accepts only positive intensities in a fixed scale. GIMaP divides by the side's
largest intensity N_I and replaces non-positive values by a fraction of σ:

```text
ỹ_i = max( I_i, 0.1 σ_i ) / N_I,   σ̃_i = σ_i / N_I,   N_I = max_i I_i                                  (10.2)
```

The scores (10.11) use the original signed I_i and σ_i. The model card warns that inputs in other units or scales
invalidate its validation.

### 10.2 What the networks propose

A candidate is one of 34 compositions (1–4 components chosen from sphere, random cylinder and vertical cylinder,
repeats allowed: 3 + 6 + 10 + 15), with, per component, six normalised coordinates p ∈ [0, 1] for R, σ_R/R, h, σ_h/h,
D, σ_D/D, the mixture-weight logits, a gate per component for the structure factor and one for the beam-tail term, and
four global coordinates g. They decode to

| Quantity | Decoded value | Range |
|---|---|---|
| R | 100^p nm | 1–100 nm |
| σ_R/R, σ_h/h | 0.02 + 0.88 p | 0.02–0.90 |
| h | 2 · 250^p nm | 2–500 nm |
| D | exp( u + p (ln 500 − u) ) nm, u = max(ln 3, ln 2.002R) | max(3, 2.002R)–500 nm |
| σ_D/D | 0.05 + 0.85 p | 0.05–0.90 |
| BG level b | 10⁻⁶ · 10⁴^g | 10⁻⁶–10⁻² × median of the particle curve |
| σ_res | 0.007 · (13/7)^g nm⁻¹ | 0.007–0.013 nm⁻¹ |
| ν_res | 5 + 5 g | 5–10 |
| beam-tail amplitude a | 10 · 100^g | 10–1000 (relative, see section 10.3) |

Mixture weights are a softmax of the logits (the last logit is 0).

Implementation: `modules/Fitting_1D_Model/Workflow_v5/conditional_v2/numpy_forward64.py`,
`modules/Fitting_1D_Model/Workflow_v5/conditional_v2/flow_codec.py`

### 10.3 Forward model

The forward model of the workflow is its own (float64 NumPy, in the normalised intensity of (10.2)):

```text
p(q) = Σ_j w_j P_j(q) S_j(q)                                                                         (10.3)
R_t(q) = 1 / ( 1 + (q / σ_res)^ν_res )                                                                (10.4)
y(q) = p(q) + b · median_q p(q) + a · [ max_{first 5 q} p / max_{first 5 q} R_t ] · R_t(q)            (10.5)
```

with the R_t term only when its gate is on, and y at least 10⁻³⁰. The form factors, the size quadrature and the
paracrystal are those of section 8.5: the vertical cylinder is normalised, and

```text
S(q) = ( 1 − φ² ) / ( (1 − φ)² + 4 φ sin²(q D / 2) ),   φ = exp( −π q² (D · σ_D/D)² )                   (10.6)
```

which is algebraically identical to (9.12). BG and the beam-tail amplitude are relative to the particle curve, so they
are not Fitting's BG and A; the solutions are converted as in section 9.9. The beam-tail term is the additive term of the
training forward model, not an instrument convolution.

### 10.4 Known composition and resolution: the conditional path

When the complete composition and both σ_res and ν_res are given (σ_res in 0.007–0.013 nm⁻¹, ν_res in 5–10), one
curve gives 30 candidates:

1. **Proposal network.** An encoder of the curve (log q, log intensity and log σ, on 256 log-spaced points) and a
   decoder give 6 hypotheses for the given composition. A base corrector network is applied 20 times; after each pass
   every hypothesis keeps the old or the new values, whichever has the lower observed logRMSE.
2. **Conditional corrector.** A network that receives σ_res and ν_res, which stay fixed exactly, is applied in 4 passes
   (again keeping the better of old and new). This gives the 6 *anchor* candidates.
3. **Update network.** A bounded update network (tanh-limited steps, D updated in coordinates that keep D ≥ 2.002 R) is
   iterated 4 times. Between iterations, the forward curve of the current candidate and its log difference to the
   observation are fed back as input. Snapshots after iterations 1 and 4 add 12 candidates (18 neural in all).
4. **Numerical correction.** A fixed 4-step damped Gauss–Newton refinement (section 10.5) of the 6 candidates of
   iteration 4;
   snapshots after steps 1 and 4 add 12 candidates (30 in all).

The original candidates are always kept beside the corrected ones.

Implementation: `modules/Fitting_1D_Model/Workflow_v5/conditional_v2/conditional_fast_predict_v2.py`,
`modules/Fitting_1D_Model/Workflow_v5/conditional_v2/conditional_resolution.py`

### 10.5 The damped Gauss–Newton refinement

The state z of a candidate has 32 coordinates (logits). Free are R and σ_R/R of every active component, h and σ_h/h of
random cylinders, D and σ_D/D where the structure factor is on, the weight logits when there is more than one component,
and the BG level and beam-tail amplitude. The composition, the gates, σ_res and ν_res stay fixed. The objective is the
unweighted mean square of the log residuals, on the backend's observation (10.2):

```text
Φ(z) = (1/N) Σ_i ( ln y(q_i; z) − ln ỹ_i )²                                                            (10.7)
```

The Jacobian J of ln y with respect to the free coordinates comes from forward-mode automatic differentiation
(TensorFlow, float32 forward). With r = ln y − ln ỹ, each step solves, for four dampings,

```text
( JᵀJ/N + μ_d · diag'(JᵀJ/N) + 10⁻⁸ Id ) Δ = −Jᵀr / N,   μ_d ∈ {10⁻³, 10⁻², 10⁻¹, 1}                       (10.8)
diag' = max( diag(JᵀJ/N), 10⁻⁶ )                                                                       (10.9)
Δ ← Δ / max( 1, max_k abs(Δ_k) )                                                                       (10.10)
```

This is a Levenberg–Marquardt-type damping with Marquardt's diagonal scaling (Levenberg, 1944; Marquardt, 1963), but
with a fixed number of steps. Of the four trial states the one with the lowest Φ is taken, and only if it lowers Φ;
otherwise the candidate stays. The step limit (10.10) keeps every coordinate change at most 1 in logit units. There
are always exactly 4 steps; no convergence test is made.

Implementation: `modules/Fitting_1D_Model/Workflow_v5/conditional_v2/active_autodiff_gn.py`

### 10.6 Unknown composition or resolution: the discovery path

GIMaP's default leaves the composition and both resolution values free (*Auto*). Then the proposal network first
proposes candidates for the 12 compositions its classifier rates highest (6 hypotheses each, 72 candidates, with the
20 base-corrector passes). These are scored by (10.11). In order of score, the first 3 distinct compositions, each with
the σ_res and ν_res of its own best candidate (or the values given), are passed through the conditional path (section 10.4):
3 × 30 candidates. When neither σ_res nor ν_res is given, the 72 discovery candidates stay in the pool too: up to 162
raw candidates per side. With a given composition and free resolution there is one composition, so 6 + 30 = 36. When
only one of σ_res and ν_res is given, the discovery candidates are dropped, because they need not respect it. The
numbers 12 and 3 are options (1–34). This path is labelled experimental: the estimated conditions do not inherit the
validation of section 10.8.

Implementation: `src/gimap/features/fitting/infrastructure/adapters/workflow_v5.py`

### 10.7 Ranking and the solutions returned

Every raw candidate's curve is recomputed with the float64 forward (10.5), multiplied by N_I, and scored against the
original signed observation:

```text
M_H = (1/N) Σ_i Huber_2( abs( (N_I y_i − I_i) / σ_i ) ),   Huber_2 as in (8.9)                          (10.11)
RMS/σ = √( (1/N) Σ_i ((N_I y_i − I_i)/σ_i)² )                                                          (10.12)
logRMSE = √( mean over I_i > 0 of ( ln(N_I y_i / I_i) )² )                                              (10.13)
```

The score is computed **before** duplicates are removed. Then, in order of M_H, a candidate is a duplicate of a better
one with the same composition and gates when the RMS distance of their normalised states (37 dimensions, minimised over
permutations of equal components) is below 0.03; across different numbers of components only identical states count.
At most 8 solutions are kept. Each reports its stage (for example `discovery_neural`, `anchor`, `neural_4`,
`numerical_4`), M_H, RMS/σ and logRMSE (natural logarithm). The 500-point display curve is not used for scoring. The
scores are not probabilities and no pass/fail threshold is applied. In Fitting's *AI proposal* the 8 solutions are
converted and sorted by Fitting's χ²_red (section 9.9).

**Checksum gate.** Before the networks are loaded, the SHA-256 checksum of every file listed in
`conditional_fast_manifest_v2.json` (73 files: code, configuration and all network weights) is compared with the
manifest. Any missing or changed file stops *General V5*, and the message names the file. (*Single RC specialist*,
another method of the window, checks its own files and falls back to a numerical fit, giving *Fast model
unavailable* as the reason.)

Implementation: `modules/Fitting_1D_Model/Workflow_v5/conditional_v2/native_observation_adapter.py`,
`modules/Fitting_1D_Model/Workflow_v5/conditional_v2/conditional_fast_predict_v2.py`

### 10.8 Validation status

The following numbers are **stated by the model card and validation report of the package**
(`MODEL_CARD_zh.md`, `VALIDATION_zh.md`, `VALIDATION.json`). They were **not re-run for this document**.

- Test set: 128 independent **synthetic** curves, 32 for each number of components K = 1…4, simulated with the beam-tail
  term on. The complete true composition and the true σ_res and ν_res were given (the conditional path of
  section 10.4).
  Model and rules were frozen beforehand; candidates were selected by their error against the noisy observation and
  evaluated against the clean simulated curve. logRMSE uses the natural logarithm.

| Mode | clean logRMSE below 0.05 | below 0.03 | below 0.01 | median | P90 |
|---|---|---|---|---|---|
| neural only, 18 candidates | 84/128 | 55/128 | 12/128 | 0.03483 | 0.08229 |
| neural + fixed 4-step GN, 30 candidates | 114/128 | 96/128 | 41/128 | 0.01558 | 0.05216 |

- Paired: 30 curves gained the 0.05 level and none lost it; the bootstrap 95 % interval of the difference is 16.41 to
  30.47 percentage points.
- Timing (GPU, full call without reading the input from disk): first call 30.95 s, then median 1.81 s and P90 3.83 s
  over the other 127 curves. Verified local environment: Windows, Python 3.10.18, tensorflow-intel 2.15.1 (CPU),
  NumPy 1.26.4, SciPy 1.15.3.

What this does **not** validate, as the model card itself states: composition discovery (the default *Auto* path of
section 10.6), unknown resolution, real measured data, coverage of all equivalent solutions (an earlier development version
recovered 20 of 135 reference parameter modes on 32 development curves), or any posterior probability. The 114/128 is a curve-fit rate, not a rate of correct
parameters. Mixture weights are not probabilities, mass fractions or volume fractions. On real data only observed
residuals (M_H, RMS/σ, logRMSE) are reported, and a good fit proves neither that the parameters are true nor that they
are unique.

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

For GISAXS frames, the batch fit instead sends each frame's horizontal cut to the quick physical fit of Fitting (sphere, random cylinder or vertical cylinder with size dispersity and a spacing D; section 8.5). With *previous* or *first*, only the particle family and D are carried to the next frame. Its results are in nm⁻¹ and nm.

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

The Sputter workspace follows an in-situ sputter deposition series: PILATUS CBF frames of Ag or Au sputtered on a polymer film. Window by window it gives the direct features of the GISAXS pattern, a growth stage, the island radius R and spacing D (two methods), thickness substitutes, flags for windows where the beam may be off the sample, and the island height H(t) from moving q_z fringes.

GIMaP does not import this code. It runs the bundled toolkit (`utils/sputter_toolkit`, entry `scripts/sputter_struct_v1.py`) and the fringe tool (`utils/sputter_fringes`) as subprocesses and reads their files. Inside both, q is in nm⁻¹, lengths in nm, angles in degrees and time in s from the first frame. The two trained networks are part of the toolkit and bound by SHA-256 to its source files.

### 12.1 Inputs and the projection of the frames

**Inputs.** The folder of CBF frames of one scan; the X-ray energy (eV) or wavelength; the sample–detector distance (mm); the incidence angle α_i (°); the metal (Ag, Au, or Cu); the polymer type (used only to pick the published band, section 12.11); the sample roll (default 0°); optionally the direct-beam position, a detector mask (`.npy`), the window length, the deposition rate F (nm/s), the start t_0 and the end t_end of the deposition (s). The page takes the beam position in GIMaP's pixel coordinates and subtracts 0.5 px to give the toolkit's 0-based pixel centres (section 1.3). Without a beam position, the toolkit estimates it from the frames. The page proposes the energy from the `Wavelength` line of the first CBF header or from the `beam_energy` of a NeXus file of the same scan, and says which; the `E<eV>` in the name of a flat-field file is the detector's calibration energy and is never used. The pixel size is fixed at 0.172 mm. Cu is analysed with the Ag model, for reference only: the networks were trained for Ag and Au, and the errors are not validated for Cu.

**Frames.** Time stamps come from the CBF headers; `t = 0` at the first frame. They must increase strictly, else the run stops ("cannot invent a clock"). A truncated or undecodable frame is listed with its reason and left out; no pixel is invented. Valid pixels are finite, non-negative, below the header's count cut-off and inside the mask. Only a fixed detector region around the beam is used (939 rows above to 31 rows below the beam, about ±555 columns); pixels within ±0.13 nm⁻¹ of |q| = 2.1 nm⁻¹ are excluded (the code does not state why).

**Projection.** For each pixel, with `u` the unit vector from the sample to the pixel (flat detector perpendicular to the beam at the given distance), `x̂` the beam direction, `n_s` the sample normal tilted by α_i (and the roll), and `ŷ_s` the in-plane direction normal to the beam:

```text
α_f = arcsin( n_s · u )                                                                         (12.1)
q   = k (u − x̂),    k = 2π/λ,   λ = 1239.841984 / E[eV]  nm                                      (12.2)
q_y = ŷ_s · q,      side = sign(q_y)                                                            (12.3)
```

Counts `Y` and pixel exposure `U = (number of pixels) × exposure time` are summed per cell of a native grid: |q_y| from 0.05 to 1.80 nm⁻¹ in 70 bins of 0.025 nm⁻¹, α_f from 0.02° to 1.50° in cells of 0.01° (with a few extra band edges), and the two sides of the beam.

Implementation: `utils/sputter_toolkit/sputtergisaxs/polymer_dev_intake.py`, `utils/sputter_toolkit/sputtergisaxs/stage5_input.py`, `src/gimap/features/sputter/infrastructure/toolkit.py`

### 12.2 Time windows

Frames are summed into windows of 0.1 s during the first 5 s and of `T` afterwards (`T` = 0.5 s by default; the page accepts 0.05–60 s). With `Δt` the median frame spacing:

```text
t' = t + min(Δt/2, T/4)                          (absorbs acquisition jitter at the edges)      (12.4)
window = ⌊t'/0.1⌋                 for t' < 5 s                                                  (12.5)
window = 50 + ⌊(t' − 5)/T⌋         for t' ≥ 5 s                                                  (12.6)
Y_w = ∑_{i∈w} Y_i,   U_w = ∑_{i∈w} n_i t_exp,i,   t_w = ∑ t_exp,i t_i / ∑ t_exp,i                            (12.7)
```

`n_i` is the pixel count per cell and `t_exp,i` the header exposure of frame `i`. The sums are exact. The run checks that the total counts and the total exposure of the windows equal those of the frames ("photon/exposure closure"). A window with fewer than 1000 counts in total (shutter closed, beam off) gets no stage and no values.

Implementation: `utils/sputter_toolkit/sputtergisaxs/polymer_review_data.py`

### 12.3 Direct features of each window

**Correlation peak q\*.** In each of four α_f bands (0.08–0.30°, 0.30–0.60°, 0.60–1.00°, 1.00–1.50°) and on each side, the horizontal cut `Y(q)` with exposure `U(q)` is fitted by Poisson deviance on `[q_lo, 1.75] nm⁻¹`, `q_lo = max(0.05, lowest q with exposure)`:

```text
μ(q) = U(q) · s · [ b_0 + b_1 (q/0.5)^(−p) + A · exp( −(q − q*)² / (2σ_q²) ) ]                     (12.8)
dev  = 2 ∑ [ y ln(y/μ) − y + μ ]                                                                (12.9)
ΔAIC = dev_0 − dev − 6                       (dev_0: the same model without the Gaussian)       (12.10)
```

(12.10) is the difference of the Akaike information criterion (Akaike, 1974) for three added parameters.

```text
FWHM = 2 √(2 ln 2) · σ_q,     D* = 2π / q*                                                    (12.11)
```

`s` is the 80 % quantile of the rates (a scale only). Bounds: `b_0, b_1, A ∈ [0, 10³]`, `p ∈ [0, 5]`, `q* ∈ [q_lo + 0.01, 1.74]`, `σ_q ∈ [0.02, 0.6] nm⁻¹`. At least 12 points and 100 counts are needed. Several starts are tried (the largest excess over the null fit, 0.57 nm⁻¹, low-q excesses below 0.25 nm⁻¹, 0.0875 and 0.1375 nm⁻¹, with widths 0.03, 0.07 and 0.16 nm⁻¹); the lowest deviance wins. The covariance is `(JᵀJ)⁺ · max(1, dev/(N − 6))`.

The peak is *resolved* when the optimiser succeeded, `ΔAIC > 10`, `A > 2 SE(A)`, it is not at an edge (`q* ± FWHM/2` inside the supported window within 0.0125 nm⁻¹, `0.0205 < σ_q < 0.59 nm⁻¹`), and there is no gap wider than 0.051 nm⁻¹ in the data at or within ±σ_q of q*.

Per α_f band: *bilateral* when both sides are resolved and agree within 10 % (`(max − min)/mean ≤ 0.1`; q\* is their mean), *one side* when one side is resolved. The window's q\* is that of the first bilateral band (lowest α_f first), else of the first one-side band. The peak is total scattering: host and metal are not separated.

**Yoneda ridge.** The α_f profile (cells of 0.01° from 0.02° to 0.80°, both sides, |q_y| from 0.05 to 1.2 nm⁻¹) as a rate; its highest local maximum in 0.08–0.55°; the ridge position is the centroid of the rate above its 20 % quantile within ±0.07° of that maximum.

**High-q slope.** The slope of `ln(Y/U)` against `ln q` over 1.0–1.75 nm⁻¹ (both sides, all α_f), weighted by √Y, with at least 8 points. It is an apparent slope of the total scattering, not a metal-only size.

**Intensity rate.** `r = ∑Y / ∑U` (counts per pixel-second of the native grid), `SE(r) = √(∑Y) / ∑U`.

Implementation: `utils/sputter_toolkit/sputtergisaxs/polymer_review_features.py`, `utils/sputter_toolkit/sputtergisaxs/features.py`

### 12.4 The growth stage of a window: an observation rule

The stage is decided from the direct features by a fixed rule, not by a network. The code calls the stages *candidates* and states that no connectivity, zero-metal reference or cause of sample motion is inferred. The quantities, per window at time `t`:

- Reference Yoneda `Y_0`: the median ridge position of the windows in the first 2 s. Smoothed ridge `ỹ(t)`: the median over `[t − 5 s, t]` (past only). Shift `= ỹ − Y_0`; *ever moved* once `|shift| ≥ 0.02°`; *unmoved* when `|shift| ≤ 0.01°` and never moved.
- *Plateau*: over `[t − 10 s, t]` at least 5 smoothed values spanning at least 8 s, their 10–90 % spread ≤ 0.02° and the slope of a straight line ≤ 0.001 °/s in magnitude.
- *Peak moving to lower q*: the q\* history of the same α_f band over the last 10 s (at least 5 points over at least 2.5 s) has Spearman ρ ≤ −0.6, a negative slope, and a drop (median of the first third minus median of the last third) of at least `max(0.025 nm⁻¹, 0.05 q*_first)`. The history restarts when the band changes or q\* jumps by more than ×1.5 or less than ×0.5.
- *Lost*: no resolved peak now, and the last one at least 1 s ago.
- *Low count*: `r ≤ min( median_{0–2 s} r + max(3 · 1.4826 · MAD, 3 · median SE), Q_25(r) )`, where `Q_25` is the 25 % quantile of all windows of the scan.

Decision, in this order:

| Stage | Condition |
|---|---|
| coalescence or film candidate | peak lost, a lower-q motion seen before, Yoneda ever moved, and now on a plateau |
| coalescence candidate | one-side peak that moves to lower q now |
| separate islands candidate | a resolved peak |
| pre-nucleation candidate | low count, Yoneda unmoved, no lower-q motion seen |
| unresolved | everything else |

Implementation: `utils/sputter_toolkit/sputtergisaxs/polymer_review_stages.py`

### 12.5 The two networks

**Routing.** A window whose rule stage is coalescence or coalescence-or-film goes to the TS network (`TS_layer`, a late, film-like texture model); every other window to the polymer network (`polymer_embed`, islands on a polymer). Every window is also run through the other network, so that the physical path (section 12.6) can use the network of its own stage.

TS stands for Teubner–Strey: the TS network was trained on a Teubner–Strey-type effective two-phase scattering scenario (a positive lateral spectrum peaked at q_0 over a finite metal-network layer; Teubner & Strey, 1987, J. Chem. Phys. 87, 3195), which the code states is neither an island model nor a verified connected morphology (`utils/sputter_toolkit/sputtergisaxs/scenario_ts.py`).

**Input.** Eight horizontal cuts (four α_f bands × two sides × 70 |q_y| bins) and two vertical cuts I(α_f) (two sides), summed over the three |q_y| bins around q\* (over 0.15–1.2 nm⁻¹ when no peak is resolved). Each cut's rate is divided by its own median positive rate and passed through arcsinh, with a support mask. Added are the four direct features (q\*, FWHM, Yoneda, high-q slope / 5) with presence flags, the peak-known flag, `ln(E / 12 keV)`, α_i and the metal flag (Au or Ag; Cu enters as Ag).

**Output.** A Gaussian-mixture posterior per window. Up to three modes are kept, with their probabilities and component intervals (68 % and 95 % of the component Gaussian, mapped to physical units; these are not calibrated intervals). The tables use the most probable mode.

| Network | Quantities |
|---|---|
| polymer_embed | D (centre-to-centre spacing, nm); R (number-mean base radius of an equivalent flat-based hemisphere, h = R, nm); σ_ln (log-normal width of R); burial depth (nm); polymer thickness (nm) |
| TS_layer | Λ_TS = 2π/q_0 (`L` in the toolkit's output), a lateral texture length of the late film-like pattern (nm; not an island spacing); width; metal fraction f_m; film thickness; a depth correlation; lamellar surrogates `w = f_m Λ_TS`, `g = (1 − f_m) Λ_TS` |

The recorded domains are D in 2–120 nm (polymer) and Λ_TS in 3–200 nm (TS).

**Reliability from simulation.** The configuration `configs/sputter_struct_reliability_v1.json` holds numbers from a simulation scoring of 200 independent simulated `polymer_embed` families (100 Ag, 100 Au) at the counts of one 0.5 s window. Its own scope note: exploratory numbers computed after the simulation was scored, not a pre-registered test, valid within the hemisphere model. The typical relative scatter is

```text
e = exp( 1.4826 · MAD( ln(estimate/true) ) ) − 1                    (about a 68 % band)         (12.12)
```

| Quantity | Network (grades C, P) | Converged refinement (grades A, B) |
|---|---|---|
| R, Ag | e = 0.307, P90 = 0.634, order right 0.92 | e = 0.157, P90 = 0.67, order right 0.935 |
| R, Au | e = 0.256, P90 = 0.634, order right 0.924 | e = 0.101, P90 = 0.496, order right 0.963 |
| D, Ag | e = 0.26, P90 = 0.635, order right 0.902 | e = 0.119, P90 = 0.629, order right 0.918 |
| D, Au | e = 0.24, P90 = 0.627, order right 0.905 | e = 0.052, P90 = 0.627, order right 0.89 |

"Order right" is the share of pairs of windows that differ by at least ×1.2 whose order is predicted correctly. P90 is the 90 % quantile of the relative error: about 10–20 % of the values are more than 50 % off, because another structure fits the counts equally well. Burial depth is unreliable and the polymer thickness is not determined (rank correlation 0.08–0.13). The TS length Λ_TS has only a development calibration. The TS metal fraction and film thickness are not identifiable: in simulation their posterior equals the prior.

Each R or D value carries the band `[v/(1 + e), v(1 + e)]`. A network D is checked against the direct observation:

```text
D_direct_check = consistent  ⇔  0.587 ≤ D / (2π/q*) ≤ 1.75                                      (12.13)
```

The band holds 90 % of the true `D/(2π/q*)` in the simulation.

Implementation: `utils/sputter_toolkit/sputtergisaxs/polymer_review_infer.py`, `utils/sputter_toolkit/sputtergisaxs/neural_refine_input.py`, `utils/sputter_toolkit/sputtergisaxs/sputter_struct.py`

### 12.6 The physical path

The rule decides every window on its own, so stage, network and source of R and D can switch back and forth. The physical path reads the series once, with one model assumption stated in the code: during deposition the film does not turn back into islands.

**Stage path.** Stages are mapped to an ordinal scale: pre-nucleation 0, islands 1, coalescence 2, film 3 (the coalescence-or-film candidate counts as film). Windows with an unresolved stage, without data (section 12.2) or with the beam possibly off the sample (section 12.9) cast no vote. The path is the non-decreasing L1 fit of the votes by pool-adjacent violators (isotonic regression; Ayer et al., 1955; Robertson et al., 1988), the value of a pooled block being its lower median. A window without data gets no stage. Any other window without a vote takes the value of the previous voting window (the next one at the start) and is marked `stage_imputed`.

**Network of the path.** Path stages 2 and 3 use the TS network, stages 0 and 1 the polymer network. So coalescence windows are already film-type windows (`film = true`) and have no R or D. R (grade **P**) is given in every window whose path stage is islands; D in those of them with a resolved peak. The values come from the most probable mode of the path's network, with the network bands of section 12.5. Pre-nucleation and film windows have none.

**Film onset.** The onset is the first window on the TS network. For each possible step position `m` (and "no TS at all"), the number of resolved windows that disagree with a single islands→TS step at `m` is counted:

```text
N(m) = #{ resolved i < m with a late stage } + #{ resolved i ≥ m with an early stage }           (12.14)
```

The onset interval spans every `m` with `N(m) ≤ min N + 2`. The onset is *not established* when "no TS at all" also lies within `min N + 2`.

**Trends of R and D.** At each window `i`, a Theil–Sen fit (Theil, 1950; Sen, 1968) of `ln v` against `t` over the island windows (beam-off windows left out) within `±span/2`:

```text
span = max( modulation period (section 12.8), 2.5 s )                                                   (12.15)
β    = median over pairs (j < l) of (ln v_l − ln v_j) / (t_l − t_j)                              (12.16)
trend_i = exp( median_j [ ln v_j − β (t_j − t_i) ] )                                            (12.17)
```

With fewer than 4 points in the span, the trend is the local median. The trend is also evaluated at beam-off island windows.

**Back-and-forth counts.** Recorded before (rule) and after (path): stage excursions A→B→A, module changes TS→polymer, and zig-zags of R and D (a step `|Δ ln v| > ln 1.2` reversed to within 20 % of the value before it within the next 1–3 values). Beam-off windows are left out; the R and D sequences break at them.

Implementation: `utils/sputter_toolkit/sputtergisaxs/sputter_physical.py`

### 12.7 Thickness substitutes

GISAXS does not measure the metal thickness here. Two substitutes are given.

**Island-volume equivalent thickness** (island windows with R and D only). For flat-based hemispheres (h = R) on a hexagonal lattice of spacing D, with a log-normal size distribution of width σ_ln:

```text
t_eq = (2/3) π R³ exp(3σ_ln²) / ( (√3/2) D² ) = 2.4184 · R³ exp(3σ_ln²) / D²                            (12.18)
band: t_eq / f_band … t_eq · f_band,    f_band = exp( √( 9 ε_R² + 4 ε_D² ) ),   ε = ln(1 + e)                   (12.19)
```

`R` is the number-mean radius; `exp(3σ_ln²)` turns `R̄³` into the mean `⟨R³⟩` of a log-normal distribution. The band is first-order propagation of `R³/D²` in log space with the network scatters `e` of section 12.5: `f_band = 2.53` for Ag (and Cu), `2.24` for Au. Invented example: `R = 3 nm`, `D = 8 nm`, `σ_ln = 0.2` give `t_eq = 1.15 nm`, band 0.45–2.91 nm (Ag). It is the metal in the visible islands under this shape assumption, not the deposited amount.

**Nominal thickness** (when a rate is given):

```text
h_nom(t) = F · min( max(0, t − t_0), t_end − t_0 )                                                 (12.20)
```

`t_0 = 0` (the first frame) when not given, which the notes call an assumption; each second of error in t_0 shifts `h_nom` by F × 1 s. Without `t_end` the deposition lasts to the end of the scan. A rate that is not positive, or `t_end ≤ t_0`, is refused before any frame is read.

**Mass continuity.** The median `t_eq` and the median `h_nom` over the last 5 island windows before the film onset (beam-off windows left out), and their ratio.

Film windows get no thickness: the TS film thickness and metal fraction are not identifiable.

Implementation: `utils/sputter_toolkit/sputtergisaxs/sputter_physical.py`

### 12.8 The intensity-modulation period

During deposition the sample stage is translated back and forth so that the beam does not damage one spot. Near the ends of the travel the grazing-incidence footprint can reach the sample edge or leave the sample, and the intensity dips once or twice per cycle. The code states that the data alone do not prove this source; the page asks rather than asserts ("sample stage translation?").

1. Windows after 5 s with the modal nominal length `Δ` (0.5 s by default) and `r > 0`; at least 20 are needed.
2. `y_m = ln r` on the regular index grid `m = round((t − t_first)/Δ)`; `z = y − ` a centred local least-squares line over ±30 s; mean removed.
3. `ACF(κ) = mean(z_m z_{m+κ}) / var(z)` over valid pairs (at least 10 pairs).
4. Local maxima of the ACF of at least 0.1 at lags 2–60 s, refined by a parabola through three points.
5. The period is the shortest maximum whose double lag (within `max(1.5, 0.1 κ)` windows) is also a maximum, with `max(ACF(κ), ACF(2κ)) ≥ 0.3`, and with at least 3 full cycles in the series. When `2κ` is beyond 60 s, a maximum of at least 0.3 alone suffices. Failing that, the first maximum of at least 0.3 (with 3 cycles).
6. When `ACF(2κ) ≥ ACF(κ) + 0.05`, the dips have unequal depths and the pattern repeats exactly only after `2κ` (`full_repeat_s`): possibly one full back-and-forth cycle of the stage.

The strongest periodogram line in the same band is recorded as a cross-check. The summary keeps the period as `modulation_period_s` (and under its older key `rotation_period_s`).

Implementation: `utils/sputter_toolkit/sputtergisaxs/sputter_physical.py`

### 12.9 Windows where the beam may be off the sample

From the data only (the intensity rate of each window with data):

```text
S = max(P, 2 s) with a detected modulation period P;   S = 10 s without                          (12.21)
ℓ_w = min( Q_90{ ln r_j : t_w − S ≤ t_j < t_w },  Q_90{ ln r_j : t_w < t_j ≤ t_w + S } )          (12.22)
u_w = exp( ln r_w − ℓ_w )                                                                        (12.23)
core:  u_w < 0.5 (with a modulation),   u_w < 0.2 (without)                                      (12.24)
```

Each side needs at least 3 windows with data; the first and last windows are not judged. The lower side counts, so a steady rise or fall during growth is never a dip, nor is a step that stays. Below 50 % more than half of the scattering is missing, as when more than half of the footprint has left the sample. From each core the flag extends along the time-ordered windows with data, on both sides, while `u_w < 0.8`; an unjudged window stops it, windows without data are skipped. Without a modulation only an extreme dip has a core (no sample seen, or the beam interrupted), and its flanks are extended the same way. A flagged window without a Yoneda ridge, while at least 50 % of the other windows have one, says so in its reason; this corroborates but does not decide. The record also gives how many windows a core limit or release level 0.1 lower or higher would flag.

Flagged windows keep their values in the tables. They cast no stage vote and are left out of the R and D trends, the back-and-forth counts, the mass continuity, the literature method's trend of D and its checks of the network, and the fringe analysis. Known limits stated in the toolkit: a dip that never reaches the core limit is not flagged however long; a window where the footprint reaches the sample edge can show *more* intensity (edge scattering) and is not flagged; the stage position is not read.

Implementation: `utils/sputter_toolkit/sputtergisaxs/sputter_offsample.py`

### 12.10 The literature method (mass balance for hemispheres)

The in-situ GISAXS studies of sputter growth at the P03 beamline (Schwartzkopf et al., 2015; Gensch et al., 2019, 2021; Bulut et al., 2024) do not fit R. They take the nominal thickness δ from a deposition rate and the spacing D = 2π/q\* of the direct peak, and compute R by mass balance for flat-based hemispheres on a hexagonal lattice. With a deposition rate and at least 10 usable peak windows, this is the default explanation of the page (grade **Lit**); otherwise the physical path is.

```text
δ(t) = F · ( min(t, t_end) − t_0 )                                                               (12.25)
D_lit = D* = 2π / q*                                                                               (12.26)
δ = (2/3) π R³ / ( (√3/2) D² ) = 2.4184 · R³ / D²                                               (12.27)
R_lit = ( δ D² / 2.4184 )^{1/3}                                                                 (12.28)
Θ = (2π/√3) (R/D)² = 3.6276 (R/D)²                                                              (12.29)
2R/D = 2 ( δ / (2.4184 D) )^{1/3}                                                               (12.30)
```

Unlike (12.20), δ is not clipped at 0: windows before t_0 have `δ < 0` and are marked as before the deposition. Invented example: `F = 0.10 nm/s`, `t_0 = 0`, `q* = 1.047 nm⁻¹` at `t = 10 s` give `δ = 1.0 nm`, `D = 6.0 nm`, `R = 2.46 nm`, `2R/D = 0.82`, `Θ = 0.61`.

**Which peaks are used.** D is taken only from windows with data and a resolved peak (bilateral or one side). A peak is not used, in this order of tests, when:

1. the window lies before the deposition (`t < t_0`, or `δ ≤ 0`): host;
2. the beam may be off the sample (section 12.9);
3. `D > 2 ×` the largest published D(δ) of *all* reference curves of all metals at that δ (log–log interpolation, ends held): host;
4. it comes after the first gap longer than `max(20 s, 2 P)` between two usable peaks: late peak (every later peak too);
5. it is a jump: after at least 3 kept peaks, with the median `D_ref` and median time `t_ref` of the last 5 kept peaks (times are deposition times `min(t, t_end)`),

```text
ln( D / D_ref ) > ln 2 + ln max( 1, (t − t_0)/(t_ref − t_0) )                                      (12.31)
```

6. it is more than a factor 2 from the trend (12.32) (`|ln D − ln D_trend| > ln 2`), tested twice with a refit in between: outlier.

A jump or outlier within ×1.35 of the median host-peak D (at least 2 host peaks) is called a host peak too. With fewer than 10 used peaks there are no literature-method values.

**Trend of D.** The non-decreasing L1 fit of `ln D_lit` in time order: pool-adjacent violators with block medians. It is extended as a step function (the value of the last used window at or before each time) to every window with data between the first and the last used peak.

```text
D_trend(t) = exp( isotonic L1 fit of ln D_lit )                                                 (12.32)
```

**Percolation.** The first window after nucleation (`δ > 0.4 nm`) where `2R/D` computed from the trend reaches 1, linearly interpolated in δ between that window and the one before. At `2R = D`:

```text
Θ_c = π/(2√3) = 0.9069,     δ_c = 2.4184 · D(t_c) / 8 = 0.3023 · D(t_c) = 0.6046 · R             (12.33)
```

So the percolation thickness is fixed by D at the crossing time; the rate and t_0 decide only *when* the crossing happens. Invented example: if the trend is 10 nm at the crossing, `δ_c = 3.02 nm`, reached at `t = 30.2 s` with `F = 0.10 nm/s`.

**68 % interval.** It holds the scatter of the windows and the errors of F and t_0:

- Scatter: residuals of `ln D_lit` around a running median of `±max(2, ⌊b/2⌋)` neighbours; `b = max(3, round(max(P, 2.5 s) / median spacing))` windows. 200 moving-block bootstrap samples (Künsch, 1989; blocks of `b` residuals, random starts, seed 0) are added to the fitted trend and refitted isotonically.
- Lower end: the 16 % quantile of the bootstrap crossings with the rate `F(1 + ε_F)` and the start `t_0 − ε_t`. Upper end: the 84 % quantile with `F(1 − ε_F)` and `t_0 + ε_t`. A bootstrap trend that never crosses counts as infinite.
- Defaults: `ε_F = 0.10`; `ε_t = 2 s` when t_0 is assumed at the first frame, `0.5 s` (one window) when t_0 is given. The configuration calls them engineering choices (a QCM reading with an untuned tooling factor; a shutter opened by hand). The toolkit's `physical` command accepts `--rate-error` and `--t_0-error`; GIMaP always uses the defaults.
- Times: `t_low = (t_0 − ε_t) + δ_low / (F (1 + ε_F))`, `t_high = (t_0 + ε_t) + δ_high / (F (1 − ε_F))`.

A systematic error of 2π/q\* is not in the interval. The δ range where the trend's `|2R/D − 1| ≤ 0.05` is recorded; there a small change of F or t_0 moves the crossing far.

**States.**

| State | Condition | Meaning |
|---|---|---|
| reached | central crossing exists and both ends are finite | δ_c with its interval (the interval always contains δ_c) |
| first_window | 2R/D ≥ 1 already at the first counted window (both ends finite) | upper bound; lower end 0 |
| not determined | the lower end is finite, but the central trend does not cross or the upper end is infinite | 2R/D may reach 1 within the errors but need not; R is given throughout |
| not reached | the lower end is infinite (even the early setting does not cross in at least 84 % of the resamples) | lower bound: the peak phase ends first |

After a reached (or first-window) percolation, windows later than `t_c` count as film: D = 2π/q\* is kept, R is not given. This is the page's choice, after molecular dynamics finding that R loses its island meaning once a network forms (Abraham et al., 2016); the P03 studies go on reporting R and 2R/D after percolation.

**Network quantities on the same axis.** For island windows of the physical path: `2R/D_net` and `Θ_net` from the network R and D with (12.29), and `η = t_eq / δ` (for `δ > 0`), the share of the nominal metal in the islands the network sees.

Implementation: `utils/sputter_toolkit/sputtergisaxs/sputter_literature.py`, `src/gimap/features/sputter/domain/literature.py`

### 12.11 Checks against published growth

The Literature tab gives six checks, each *consistent*, *inconsistent* or *not testable*, with what was measured, what is published and the references. The literature-method R agrees with the literature by construction (it is the same mass balance); the independent tests are the shape of D(δ), the percolation δ against the published thresholds, and the network's 2R/D and mass balance.

**Published band.** The host class is chosen from the polymer name: oxides (Si, SiO₂, TiO₂, Al₂O₃, glass, …) give Au/Si, Ag/oxides or Cu/oxides; Au on PS gives Au/PS; Au on another polymer gives Au/other polymers; Ag and Cu give `<metal>/polymers`; an unknown polymer uses all polymers of the metal. A named polymer that no published value of its class covers (for example PNO, PSBMA or PEO) keeps the band of the class, and the note says that no published percolation data exist for it. Each published threshold counts as `δ_c ± its printed ±` (or `± 10 %` of itself when none is printed, so that a class with one value is not a band of zero width); a published range counts as it is. The band is the minimum to maximum of these. A rate-scaled band `δ_c (F/F_ref)^(−γ_F)`, `γ_F ∈ [0.07, 0.17]`, is printed for information; it does not enter the verdict, because chambers differ by more than the rate effect.

**Interval against band.** Overlap is consistent; an interval entirely below the band is inconsistent ("earlier"), entirely above inconsistent ("later"). A one-sided result (first window, not determined, not reached) that overlaps the band is not testable.

1. **D does not decrease with δ** (after nucleation, `δ > 0.4 nm`; in time without a rate). With the used peaks (at least 10, and at least 24 for a verdict): Spearman ρ of ln D against δ is reported; the peaks are split into `min(12, ⌊n/8⌋)` equal-count bins in time order; per bin the median of ln D and `SE = 1.2533 · 1.4826 · MAD / √n_bin`. Inconsistent when some earlier bin's median exceeds a later one's by more than `max(ln 1.1, 2 √(SE_i² + SE_j²))`. Late peaks are compared with the end of the trend in a note.
2. **Percolation** of the literature method against the band. For Cu the crossing is also computed with the published shape factor `f_H = H/R = 2.75` (2R/D ∝ f_H^{−1/3}); when the two verdicts differ, the result is not testable.
3. **TS onset** of the physical path, converted to δ, against the same band; open above when the onset is not established.
4. **Network 2R/D and coverage**: island windows with `δ ≥ 1 nm` (beam-off left out), at least 5. Consistent when the median `2R/D ∈ [0.55, 1.0]` or the median `Θ ∈ [0.5, 0.96]`; coverage follows from 2R/D, so they are one check.
5. **Network mass balance**: the same windows; inconsistent when `median η · f_band < 0.5` or `median η / f_band > 1.5`, with `f_band` the band factor of (12.19). The 0.5 is an engineering choice.
6. **TS film thickness** grows about linearly with δ: always not testable while the network's film thickness is not identifiable.

Without a deposition rate only check 1 (in time) is made.

Implementation: `utils/sputter_toolkit/sputtergisaxs/sputter_literature.py`, `utils/sputter_toolkit/configs/literature_reference_v1.json`

### 12.12 Island height H(t) from moving q_z fringes

While the metal grows, vertical cuts at low q_y show minima that move to lower α_f: the minima of the vertical form factor of the islands or the layer. GIMaP measures them in the background after a result is shown (Sputter ▸ Size Trend, panel "H, island height from q_z fringes"), with 1 s time groups, the metal of the result and the default q_y band below; the page passes no α_c and no other band.

**Optics.** The fringe tool takes k and λ as in (12.2), with the same constant.

```text
α_c = √(2δ_X),   δ_X = r_e λ² ρ_e / (2π)                                                         (12.34)
k_z(α) = k √max( sin²α − sin²α_c, 0 )                                                           (12.35)
Q(α_f) = k_z(α_i) + k_z(α_f)              (inside the metal)                                     (12.36)
q_z = k ( sin α_i + sin α_f )             (vacuum, for the baseline only)                        (12.37)
H_n = 2π n / Q(α_f)                                                                             (12.38)
```

`r_e = 2.8179403262 × 10⁻⁶ nm`. This is the refraction-corrected (distorted-wave) wavevector transfer of grazing-incidence scattering (Renaud et al., 2009); only the position of the minima is used, not a full DWBA intensity. Bulk electron densities: Cu 2.46, Ag 2.77, Au 4.66 e/Å³ (no anomalous dispersion). `H_n` is the height of a uniform layer (box profile) whose order-n minimum of the vertical form factor `sinc(Q H/2)` lies at α_f, with refraction inside the metal. It is the vertical size of the islands or the layer; it equals the film thickness only once a continuous film has formed. Invented example at 12.0 keV and α_i = 0.40°: `α_c(Ag) = 0.295°`; an order-1 minimum at α_f = 1.00° gives `Q = 1.301 nm⁻¹` and `H = 4.83 nm` (the vacuum q_z = 1.486 nm⁻¹ would give 4.23 nm).

**Data.** For each window, both sides, the counts and exposure of the |q_y| bins inside 0.05–0.15 nm⁻¹, merged into regular 0.01° α_f cells. Windows flagged as beam-off and windows with fewer than 1000 counts in the band are left out. Every chunk is checked against the SHA-256 recorded at binning. Windows are grouped in 1 s groups from the first usable window and summed.

**Clean α_f range.** A cell whose exposure (summed over the scan) is below half the median is unusable; a run of at least 0.02° of such cells is a gap (module gap, detector edge) and is widened by one cell. Nothing below `max(α_i + 0.12°, α_c + 0.20°)` is used (specular, the metal's Yoneda region, the steep k_z near α_c). Runs shorter than 0.12° are dropped.

**Baseline.** `ln(Y/U) = a + b ln q_z`, weighted by the counts, over the whole clean range of each cut; a local baseline would absorb the fringes. Static detector rows: a cell whose narrow residual (minus a running median of 7 cells), as a median over all groups, exceeds `max(0.006, 4 × robust σ)` is left out.

**Minima.** On the residual's running mean over 5 cells, a local minimum (the lowest within ±3 cells). At scales `w` of 0.03°, 0.06° and 0.12°: a count-weighted parabola within `min(w, 0.05°)` places the vertex (within 0.02° of the local minimum); the depth is the mean of the shoulders at `±w` minus the centre (each a summed rate over `max(0.01°, w/4)`):

```text
depth = ½ (ℓ_L + ℓ_R) − ℓ_C,     σ_depth = √( 1/Y_C + ¼ (1/Y_L + 1/Y_R) )                        (12.39)
accepted  ⇔  depth > 3 σ_depth   and   depth > 0.006   (at one scale at least)                   (12.40)
```

The position error is the standard deviation of 30 Poisson-bootstrap re-locations (at least 10 must fall within 0.05°; floor 0.002°). Minima of the two sides within `max(0.012°, 3 σ_combined)` are one minimum seen on *both* sides (inverse-variance mean); within 0.05° but beyond that tolerance they *disagree* and give no height; otherwise the minimum is one-sided.

**Phase-locked fit** of each side's cut, without counting orders:

```text
ln(Y/U) = c_0 + c_1 ζ + c_2 ζ² + A cos(H Q),   ζ = ln q_z,   A < 0                               (12.41)
```

H on a grid from 1.5 to 80 nm in steps of 0.05 nm; the cosine is projected orthogonal to the baseline terms; the best H gives the largest χ² gain; `σ_H` from the curvature of χ² near the best H (`Δχ² = χ²_red`, `χ²_red ≥ 1`). A group's phase-locked H is accepted when on both sides at least 1.5 fringe periods are in range (`H ΔQ / 2π`), the best H is better by `Δχ² ≥ 10` than any H more than 25 % away (no alias), the sides agree within `max(3 σ, 3 %)`, and the group lies within 10 % of a straight line through the accepted groups (iterated; only the longest run with at most 5 missing groups; at least 4 groups).

**Tracks and orders.** Minima that stay within 0.01° for 20 s or longer are static (detector rows, a film's own fringes) and dropped. Both-side minima are linked in time (at most 3 groups apart, moving between 0.08° down and 0.01° up); a track is accepted with at least 4 minima and a net descent of at least 0.05°. One-sided minima within 0.02° of an accepted track's line join it. Orders, in this sequence:

1. from the phase-locked fit: `n = median( H_phase Q / 2π )` over at least 3 minima, within 0.25 of an integer;
2. else the first accepted track is `n = 1` when it entered at the top of the clean range (its first minimum within 0.12° of the top, or a straight line of `1/Q` over its first 8 minima reaching the top at most 30 s earlier), not before the third group;
3. a track coexisting with one of known order: `n = n_ref · median(Q / Q_ref)`, within 0.25 of an integer;
4. a track without overlap: `n = H Q / 2π` with H from a straight line through known heights within ±30 s, the median within 0.3 of an integer and every estimate within 0.5.

When more than a third of the pairs of coexisting orders (and of minima against the phase-locked fit) differ by more than `max(8 %, 3 σ)`, all orders are reset and reported as ambiguous.

**Critical angle used.** In this order: a given value; a scan of the phase-locked fits over 0.6–1.25 times the bulk value in steps of 0.0025° (at least 8 accepted groups with at least 2 periods on both sides; error from the rise of the summed χ² by its reduced value, floor 0.01°; rejected at the scan edge); a fit that makes coexisting orders give the same H (at least 3 groups with two orders; `χ² = ∑ (H_a − H_b)² / (σ_a² + σ_b²)`; rejected at the edge, when `χ²/dof > 5`, or outside 0.6–1.25 times the bulk value); else the bulk value with an assumed error of ±0.02° ("islands may be less dense than the bulk"). Tracks and orders are built with the bulk (or given) value; only the phase-locked pass is redone with the final α_c when it differs by more than 0.001°. GIMaP gives no α_c, so the first source is never used from the page.

**H of a group and its 68 % interval.** The accepted phase-locked H when it holds at least 2 periods (or when no minimum of known order exists); else the inverse-variance mean over the minima of known order, its error multiplied by `√max(1, scatter)`.

```text
σ_H² = σ_position² + σ_αc² + (s_sys · H)²                                                       (12.42)
s_sys = 8 % (< 0.75 periods), 4 % (< 1.25), 2 % (< 2), 1.5 % (otherwise)                         (12.43)
```

`σ_position` comes from the bootstrap (or the χ² curvature), `σ_αc` from shifting α_c by its error, and `s_sys` is a systematic baseline error by the number of periods in the clean range (`H · span(Q)/2π`), estimated in the code from synthetic cuts with power-law backgrounds. The slab phase assumed (minima at `H Q = 2π n`) may leave a constant offset of about 1 nm in H; this is not in the interval.

**Status of a group.** measured (seen on both sides, or phase-locked); one side only; sides disagree; ambiguous order; below range (before the first order-1 minimum entered the range: `H < 2π / Q(top)` is given as a bound); not detected.

**Growth rate.** A weighted straight line `H = a + v t` through the measured groups (at least 4 over at least 3 s), errors scaled by `√max(1, χ²/dof)`; `t_0,H = −a/v` is the deposition start the fringes imply, for information only. The change of v for α_c ± 0.01° is recorded.

**Accuracy statements.** The tests on synthetic result folders (a growing layer's fringes with Poisson noise and invented detector gaps) require that at least 90 % of the 68 % intervals contain the true H, that every accepted track gets its true order (within 0.25), and that the fitted v is within `3 σ_v + 3 %` of the true value. These are statements about the synthetic cases, not about real islands.

**Freshness.** The outputs are measured again when the tool's source (SHA-256), the binned run record, the metal, or the set of beam-off windows change.

Implementation: `utils/sputter_fringes/optics.py`, `utils/sputter_fringes/minima.py`, `utils/sputter_fringes/detect.py`, `utils/sputter_fringes/phase.py`, `utils/sputter_fringes/tracks.py`, `utils/sputter_fringes/heights.py`, `utils/sputter_fringes/cli.py`, `src/gimap/features/sputter/infrastructure/fringes.py`

### 12.13 Grades

Every R and D value carries a grade letter (Sputter ▸ Windows and Explanation).

| Grade | Source | Meaning |
|---|---|---|
| A | converged physical refinement, unique solution | refinement needs a GPU and is not in the bundle; refined windows made elsewhere can be added |
| B | converged physical refinement, an equally good alternative exists | as A, with a second solution |
| C | network only, per window | rule routing of section 12.5; band from the network scatter |
| P | network on the physical path | one stage path through time; no switching of source between windows |
| Lit | literature method | mass balance (12.27) with the nominal thickness and D = 2π/q\* |
| S | time-consistent path, data-supported | exploratory view |
| T | time-consistent path, constraint-driven | exploratory view |
| L | learned selector | exploratory view |

The selector and the time-consistent path did not pass their registered simulation tests for R; they are offered as exploratory views only.

Implementation: `src/gimap/features/sputter/domain/structure.py`

### 12.14 Stated limits

- R is the base radius of an equivalent flat-based hemisphere (h = R). Real clusters on polymers are often truncated spheres with h/R between 1 and 2. In the literature method R scales as `f_H^{−1/3}` with the shape factor `f_H = H/R`; flatter Ag on oxides and `f_H = 2.75` for Cu on PS-b-PEO are reported. Trends are more reliable than absolute values.
- D = 2π/q\* can be off by more than 40 % for `D > 10 nm` once coalescence starts (Hodas et al., 2017). The peak is total scattering: host and metal are not separated.
- The deposition rate and t_0 are not measured by GISAXS. The sticking coefficient of these metals on these polymers is not known; all metal is assumed to stay on the surface.
- Published thresholds come from other chambers and rates; the rate dependence (`δ_c ∝ F^{−γ_F}`, γ_F ≈ 0.07–0.17) is weaker than the differences between chambers. A band is a comparison, not a prediction.
- The thickness of a continuous film and the TS metal fraction are not identifiable. A film with particles on it is not modelled.
- The reliability numbers come from simulation within the hemisphere model, at the counts of 0.5 s windows; they are exploratory, not a pre-registered test. Cu is read with the Ag model and its errors are not validated.
- Rule stages are candidates; a connected film is not proven. The path assumes no return from film to islands.
- The source of the intensity modulation is not proven. Beam-off flags miss shallow dips and edge scattering that raises the intensity.
- H is that of a uniform layer, measurable only while a minimum lies inside the clean q_z range and its order is known; a constant offset of about 1 nm is possible.
- Fixed in the code and not exposed in the page: the pixel size (0.172 mm), the detector region used, the rate and t_0 errors of the percolation interval, the q_y band (0.05–0.15 nm⁻¹) and the α_c source of the fringe analysis.

### 12.15 Literature references of the method

These references are part of the method: their published values (percolation thresholds by metal and host, 2R/D and coverage ranges, D(δ) curves, the rate dependence, notes on mass balance and on 2π/q\*) are in `utils/sputter_toolkit/configs/literature_reference_v1.json`, each with its key, provenance (T text or table, F read from a figure, C calculated, R from a review, S abstract only) and DOI. The citations and DOIs below are copied from that file. Each was matched to a publisher or Crossref record, or is a standard reference of the field.

| Key | Citation (as in the configuration) | DOI or URL | Used for |
|---|---|---|---|
| Abr16 | Abraham J.W., Strunskus T., Faupel F., Bonitz M., J. Appl. Phys. 119, 185301 (2016) | 10.1063/1.4948375 | R loses its island meaning after a network forms; coverage at 1 nm |
| Aste96 | Aste T., Phys. Rev. E 53, 2571 (1996) | 10.1103/PhysRevE.53.2571 | coverage at percolation |
| Bulut24 | Bulut Y. et al., Langmuir 40, 22591 (2024) | 10.1021/acs.langmuir.4c02344 | Au percolation on PS and other polymers; D(δ) Au/PS dcMS and HiPIMS; 2R/D; coverage at 4 nm |
| Colin20 | Colin J. et al., Nanomaterials 10, 2225 (2020) | 10.3390/nano10112225 | Ag and Cu percolation on SiO₂; coverage |
| Dend20 | Dendooven J. et al., Phys. Chem. Chem. Phys. 22, 24917 (2020) | 10.1039/d0cp03563g | 2π/q_max against XRF loading |
| Gen19 | Gensch M. et al., ACS Appl. Mater. Interfaces 11, 29416 (2019) | 10.1021/acsami.9b08594 | Ag percolation on PS, PMMA, PS-b-PMMA; D(δ) Ag/PS and Ag/PMMA |
| Gen21 | Gensch M. et al., ACS Appl. Nano Mater. 4, 4245 (2021) | 10.1021/acsanm.1c00829 | Ag percolation (DC); D(δ) Ag/PMMA DC |
| GenTh23 | Gensch M., PhD thesis, Technical University of Munich (2023) | https://mediatum.ub.tum.de/doc/1690709/1690709.pdf | geometry; Ag 2R/D range; D(δ) figures |
| Gra13 | Grachev S.Y. et al., J. Phys. D: Appl. Phys. 46, 375305 (2013) | 10.1088/0022-3727/46/37/375305 | Ag on Al₂O₃: early 2R/D and coverage, sticking, flattening |
| Hod17 | Hodas M. et al., Phys. Rev. B 95, 035424 (2017) | 10.1103/PhysRevB.95.035424 | D(δ) Cu/graphene; bias of 2π/q\*; 2R/D test band |
| Jam19 | Jamnig A. et al., Sci. Rep. 9, 6640 (2019) | 10.1038/s41598-019-43107-8 | rate dependence χ; Ag and Cu percolation on a-C |
| Jeffers | Jeffers G., Dubson M.A., Duxbury P.M., J. Appl. Phys. 75, 5016 (1994) | 10.1063/1.355742 | coverage at percolation |
| Kaune09 | Kaune G. et al., ACS Appl. Mater. Interfaces 1, 353 (2009) | 10.1021/am8000727 | metal embedded in the polymer (mass balance) |
| Liang22 | Liang S., Schwartzkopf M., Roth S.V., Mueller-Buschbaum P., Nanoscale Adv. 4, 2533 (2022) | 10.1039/d2na00127f | review through which some Au thresholds are taken |
| Nittler12 | Nittler L., Delcorte A., Bertrand P., Migeon H.-N., Surf. Interface Anal. 44, 1072 (2012) | 10.1002/sia.4866 | condensation coefficients on polymers |
| Quint00 | Quintanilla J., Torquato S., Ziff R.M., J. Phys. A 33, L399 (2000) | 10.1088/0305-4470/33/42/104 | coverage at percolation (penetrable discs) |
| Ruff11 | Ruffino F. et al., Appl. Phys. A 103, 939 (2011) | 10.1007/s00339-011-6413-1 | coverage at percolation, Au on PS and PMMA |
| Sch13 | Schwartzkopf M. et al., Nanoscale 5, 5053 (2013) | 10.1039/c3nr34216f | Au percolation on SiOₓ; coverage at 1 nm |
| Sch15 | Schwartzkopf M. et al., ACS Appl. Mater. Interfaces 7, 13547 (2015) | 10.1021/acsami.5b02901 | Au/PS percolation; D(δ) Au/PS RF; geometry; shape f_H = 1 |
| Sch17 | Schwartzkopf M. et al., ACS Appl. Mater. Interfaces 9, 5629 (2017) | 10.1021/acsami.6b15172 | Au/PS rate series; χ for Au |
| Sch21 | Schwartzkopf M. et al., Nanoscale Horiz. 6, 132 (2021) | 10.1039/d0nh00538j | Au/PS percolation at a high rate; nucleation |
| Schap21 | Schaper S.J. et al., Nanoscale 13, 10555 (2021) | 10.1039/d1nr01480c | Cu percolation on PS-b-PEO; shape f_H = 2.75 |
| Schu21 | Schuster C. et al., Nanomaterials 11, 589 (2021) | 10.3390/nano11030589 | electrical percolation of Au, Ag, Cu on resin films |
| Thran99 | Thran A., Kiene M., Zaporojtchenko V., Faupel F., Phys. Rev. Lett. 82, 1903 (1999) | 10.1103/PhysRevLett.82.1903 | condensation coefficient of Ag on polymers |
| Yu15 | Yu S. et al., J. Phys. Chem. C 119, 4406 (2015) | 10.1021/jp512675w | Ag percolation on Alq₃ |

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
| Geometry | `geometry` (pixel sizes, distance, beam centre, wavelength, α_i; and what the flat detector leaves out of a `.poni`: `dropped_tilt_deg`, `dropped_rotations_deg` (original rot1, rot2, rot3), `dropped_beam_axis_rotation_deg` (Rot3 modulo full turns), `flat_detector_q_needs_attention` and `flat_detector_direction_needs_attention` (independent normal-tilt/\|q\| and direction review flags), `flat_detector_q_error_max` and `flat_detector_q_error_rms` (ε_max and ε_rms of section 3.7, fractions), `flat_detector_pixels_compared`, `flat_detector_limits`; `null` when no tilt is known), `geometry_convention` (canonical pixel-corner frame, row 0 at the top, kinematic (vacuum) grazing-incidence q of a flat detector perpendicular to the beam, in Å⁻¹), `instrument_profile` (name, source, time of update; the source begins with `automatic analysis:` when Run Automatic Analysis or `tools/gimap_agent.py` chose the geometry and with `assistant:` when the AI did, as it did for both before 0.2.0) |
| Statistics and binning | `error_model` (`poisson`, `pixel_scatter` or `scaled_poisson`: the three error models of section 6.4, as `photon_counts` and the GIWAXS intensity corrections decided; `null` without a reduction), `binning` (GIWAXS: `radial_bins`, `q_min`, `q_max`, `bin_width` and `unit` of the fixed-edge bins of every I(q), `bins_chosen`, `radial_points` (the points of I(q) after merging), `merge` (the rule of section 6.5), `chi_bin_deg` and `chi_merge`; GISAXS: one point per detector column or row, no merging) |
| Time | `created`: the UTC time the record was written (ISO 8601, for example `2026-11-01T05:52:31+00:00`) |
| Corrections | `corrections` (every setting), `intensity_corrections` (GIWAXS: how applied, formulas, parameters), `bad_pixels_left_out`, `drawn_mask_pixels`, `pixels_filled_from_mirror` |
| Results | `markers` (beam centre, horizon row, cut bands, Yoneda estimate, I(χ) window, peaks …), `curves` (key, title, x label, region of each curve), `messages` |
| Files | `source_files` (every file a frame was read from: module files, master and data files), `source_layout` for NeXus/HDF5 (kind, summary, dataset path, `transformations`), `files` (the files written) |

The numbers are those of the reduction; `geometry` is the geometry that was applied.

Implementation: `src/gimap/features/analyze/application/export_record.py`, `src/gimap/features/analyze/infrastructure/adapters/csv_export.py`.

### 14.3 Header lines of the text tables

Text tables begin with `#` lines, so that `numpy.loadtxt`, Origin and spreadsheet programs skip them. They are
UTF-8 without a byte-order mark; GIMaP's own symbols are written in ASCII (`1/A`, `deg`).

- Curve CSV `<stem>_<curve>.csv`: a title line `# GIMaP Analyze export (q in 1/A, angles in deg)`, then
  `# source:` (the path as it is), `# frame:`, `# sum of …` when frames were summed, `# files:` when the frame was read
  from several files, `# curve:` and `# region:` (JSON).
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
`software_version` with the value `SOFTWARE` (the records that do not yet are listed in section 14.9):

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
- 2D Prediction: an optional `manifest.json` beside a `.keras` model may give a SHA-256, which is checked before
  the model is loaded. The digest is not written to the export record.

### 14.8 Seeds and deterministic runs

Where an algorithm draws random numbers, the seed is fixed, so the same input and settings give the same result:

| Where | Seed | Use |
|---|---|---|
| Fitting, global search | 1729 | Latin-hypercube start population and differential evolution |
| Fitting, manual refinement | `random_seed`, default 1729 | Latin-hypercube population and differential evolution, or Sobol candidates |
| Fitting, AI fitting profile | `random_seed`, default 123 (can be set in the AI settings) | passed to the research code in its subprocess |
| Geometry Calibration, centre estimate | 0 | random pairs of gradient pixels |
| Run Automatic Analysis, Hermans orientation factor | 0 (default) | Monte Carlo error of f |

StarDist inference and the 2D Prediction networks draw no random numbers in GIMaP. The seeds are part of the code
or of the saved settings. Research run records state their seed policy; not every runtime exposes a controllable seed.

The 1D Predict request and summary also record the SHA-256 of the selected model manifest. This is a fingerprint of the declared bundle, not a per-run check of every model file. Requests preserve normalized options and explicitly identify the runtime-default random-seed policy; no fixed run seed is claimed.

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
- Seeds are not written into the records (section 14.8).
- Bad pixels, drawn masks and mirror-filled pixels are recorded as counts, not as positions (section 14.4).
- `<stem>_analysis.json` gives its time of creation in UTC (`created`), as the XRR and SEM records do; the 2D
  Prediction record gives local time without a time zone.

## 15. What GIMaP does not model

This chapter collects the limits stated in chapters 1 to 14, so that they can be checked in one place. Each item names the section that states it in full. An item is listed once, under the first heading it fits.

### 15.1 Scattering physics

- **Refraction and the DWBA.** Every q of Analyze, Fitting and 1D Predict is the kinematic vacuum value. Neither α_i nor α_f is corrected for the critical angle, and no reflected-wave terms of the distorted-wave Born approximation are computed (sections 2.5, 8.6, 9.2). The Yoneda band is located empirically, without any critical angle (sections 6.7, 8.1). The film absorption correction and the q_z of the XRR extractor also ignore refraction (sections 5.4, 13.1). The only refraction-corrected quantity is the Sputter island height H(t), which uses the refracted wavevector transfer for the positions of fringe minima, not a DWBA intensity (section 12.12). The GIWAXS report says that q is kinematic in its *Decisions*, with the q_z shift of up to about k sin α_i that refraction can cause near or below the film's critical angle at the frame's α_i (section 7.9); the GISAXS report says so too, with the Yoneda band's q_z shifted most (section 8.6).
- **Absolute intensity.** Intensities are mean counts per pixel. No monitor, transmission, exposure-time, sample-volume, flat-field, detector-efficiency, air- or window-absorption correction is applied (section 2.5). Only the optional GIWAXS geometric factors divide the intensities (chapter 5), and a series can be normalised to a reference peak (section 6.10). The scales of the Fitting models are therefore relative. The sphere and random-cylinder form factors are normalised to P(0) = 1, so Int is not number density × V² × Δρ², and the vertical cylinder is weighted by R⁴ instead (section 9.4).
- **Lorentz and Ewald-sphere corrections.** I(χ) and the ring intensities are intensities on the detector. No Lorentz factor is applied, and the missing wedge near χ = 0 is not reconstructed (sections 2.2, 7.7).
- **Texture.** The Hermans factor assumes a fibre texture about the surface normal; the missing wedge enters only through the coverage C_w and the isotropic reference f_iso, and the background under a ring is one χ-independent constant (section 7.7). No exit-angle correction (film absorption, transmission function, shadowing) is applied to I(χ), and the sin χ weight makes the bins next to the sample horizon count most: f can depend on them by many σ_f, which the report states but does not correct (section 7.7). The in-plane and out-of-plane comparison uses two sectors of 20°; it is not a pole figure (section 7.6).
- **Phases.** No phase is assigned. Lamellar and hexagonal series are reported as hints (section 7.4).
- **GISAXS models.** The 1D models depend on the magnitude of q_y only; q_z at the Yoneda row and the Yoneda enhancement are not modelled (sections 8.6, 9.2). The beam-tail term is an additive empirical term; no resolution convolution or instrumental smearing is applied (sections 8.6, 9.2, 10.3). The size distributions are intensity-weighted for spheres and random cylinders (section 9.4). Refine and Search apply no hard-core condition D ≥ 2R (section 9.5). The automatic analysis screens single-component families only, so it never proposes a mixture by itself (section 8.6).
- **Paracrystal spread.** GIMaP's σ_D/D is √(2π) ≈ 2.51 times smaller than the relative standard deviation of the neighbour distance (sections 9.5, 13.5).

### 15.2 Instrument and geometry

- **Detector tilt and rotation.** The detector is taken as flat and perpendicular to the direct beam. The calibration refines only x_c, y_c and L; the tilt of a `.poni` file is folded into the beam centre and the distance, then left out; what that costs in |q| is computed and reported, a point to check above 1° or 0.5 % (sections 2.5, 3.3, 3.6, 3.7). Rot3 leaves |q| unchanged but affects χ, qy/qz and directional cuts; it is explained separately and flagged for review above 1° in magnitude, without being corrected. The same assumption underlies the straight horizon, mirror filling, the solid-angle and polarisation factors and the XRR geometry (sections 2.3, 4.8, 5.2, 5.3, 13.1).
- **Sample orientation.** The sample is tilted by α_i about the horizontal axis perpendicular to the beam and by nothing else; a rotation about the surface normal or about the beam is not described in Analyze (section 2.5). The Sputter toolkit alone accepts a sample roll (section 12.1).
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
- **Scores.** The scores of the quick fit and of 1D Predict (M_H, RMS/σ, logRMSE) are not probabilities, and mixture weights are not volume or mass fractions. The validation of 1D Predict covers synthetic curves with known composition and resolution only (sections 8.5, 10.7, 10.8). The Sputter reliability numbers come from simulation within the hemisphere model and are exploratory (sections 12.5, 12.14).
- **SEM.** No summary statistic, distribution fit or test of agreement is computed (section 13.2).

### 15.4 Data preparation

- **Bad pixels.** Only single hot and dead pixels and scattered pixels of defective lines are found; clusters and the bad pixels of a background frame are not searched for (section 4.4).
- **Mirror filling** assumes a pattern symmetric in ±q_∥ (section 4.8).
- **Gaps.** Analyze never interpolates across gaps (sections 6.1, 11.1). Compare does interpolate across a gap inside a frame (section 11.10).
- **Reference-peak corrections.** The peak position is limited by the bin width, and the distance scaling is a small-angle relation made exact by iteration (section 6.10).

### 15.5 Series and Sputter

- **Series.** Stages and odd frames are found in frame order, not in time; only positive intensities enter; a q bin with data in fewer than 95 % of the frames is not compared; *shape only* removes the overall level; the thresholds are working choices, not derived from a noise model of the detector (section 11.11).
- **Sputter.** Islands are flat-based hemispheres (h = R); D = 2π/q* can be biased once coalescence starts; the sticking coefficient is not known and all metal is assumed to stay on the surface; the rate and t_0 are not measured by GISAXS; published thresholds come from other chambers; the thickness and metal fraction of a continuous film are not identifiable; rule stages are candidates and a connected film is not proven; shallow beam-off dips and edge scattering are not flagged; H(t) is the height of a uniform layer, with a possible constant offset of about 1 nm; Cu is read with the Ag model (section 12.14).

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
