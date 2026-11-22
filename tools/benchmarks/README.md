# Benchmarks

Scripts behind the numbers that *Physics and Methods* and the GIMaP paper quote. They use public or synthetic data
only and write their summaries next to themselves.

The fit1d kit (*Predict parameters (fit1d)*) was removed from GIMaP in 0.2.0; it remains in the repository's history
(`utils/gisaxs1d_gui_kit`). The scripts that compare against it keep the record of that comparison and need a copy of
the kit: `fit1d_recovery.py` takes it from the environment variable `GISAXS1D_KIT_HOME`; `galaxi_fit1d.py` and the
fit1d part of `cut_model_checks.py` look for it in `utils/gisaxs1d_gui_kit` (the latter skips that part without it).
`cut_model_checks.py` writes the stored `cut_model_checks.json` by default; `--out PATH` writes elsewhere. When an
optional comparison (V5, fit1d) was skipped because its old code is not in the checkout, the run refuses to
overwrite the stored file, which would lose those rows: give `--out PATH`, or `--force` to overwrite it anyway. The
summary names what was skipped (`skipped`).
The held-out curves themselves (`recovery_cases.py`, `heldout_cases.py`) need neither the kit nor GIMaP.

## Tests

On the Windows release-check machine, `python -m pytest --collect-only -q` collected **4322 tests**.
Five tests skipped for missing optional Python packages. One reads an NXcanSAS export with SasView
(`sasdata`), and four compare detector orientations with `pyFAI`. A separate test skipped because
the `h5clear` executable was missing. Run `python -m pytest -q tests/test_predict_regression.py` for the
five-cut Predict regression (72 s on the Windows test machine). In `per_cut/posterior_recovery.json.gz`
it checks known-composition fits 184 and 167 and composition choice 176. It also checks automatic level-2
fit 89 in `per_cut/posterior_newseed_auto2.json.gz` and known-composition fit 208 in `per_cut/posterior_newseed.json.gz`.

The archive was produced on Windows with Python 3.10.18 and one BLAS thread per fit. On that platform
and Python version the test requires exact agreement for every archived numerical field except run time.
Elsewhere it checks the compositions, allows an absolute difference of 0.05 in each solution probability (Windows with Python 3.11 moved one of two competing solutions by 0.031),
and checks every fitted population value and posterior median at a relative tolerance of 0.01 and every 68% and
95% interval endpoint at 0.02 (Ubuntu CI differs by up to 0.16% in a fitted value); the least-squares standard
errors, which differ by up to 45% at a bound, are compared on the archive platform only. This includes the
populations in the reported alternative solutions.
The diagnostic comparison across platforms covers `bic`, `low_ess`, `model_error`, `qzR_*` and `cyl_sphere`,
including the population identified by each flag. The first archive predates the sphere warning at
q_z R > 2, which now applies to fits 184 and 167. Refinement near a bound can change which bound,
correlation and unconstrained-parameter flags appear.

The tolerances follow these differences. With the release wheels on Ubuntu and Python 3.11,
the upper 95% endpoint for the second population's radius spread in cut 167 was 0.30541181.
The archived endpoint is 0.30579296, a relative difference of 0.001246. In cut 184, the
standard error of the spacing spread differed by 5.69% between the two runs.

## The runs the paper quotes

The recovery numbers of *Predict parameters* in the GIMaP paper, in Physics and Methods 9.13 and in the network's
`MODEL_CARD.md` come from the runs below. All were made with one version of the fitting code: git tree
`7e68831a52cf841a56fe05f8cc4e36ee62b41cb7` of `src/gimap/features/fitting/domain` (commits 2eb0243 and 7179b85 have
this tree), the network `flow_v1` (its SHA-256 is in every summary), ten workers on an Intel Core i9-13900HX; the
timing run alone used one worker. Each summary written at 7179b85 names the commit and the tree (`code`);
`posterior_recovery` and `posterior_extended` were written at 2eb0243, before summaries had that field.

| Run (Table 3 of the paper) | Summary | Every cut | Options after `--network … --out NAME` |
|---|---|---|---|
| fit1d's 200 held-out curves | `posterior_recovery_summary.json` | `per_cut/posterior_recovery.json.gz` | `--n 200 --workers 10` |
| extended set, 600 curves | `posterior_extended_summary.json` | `per_cut/posterior_extended.json.gz` | `--set extended --n 600 --seed0 20261201 --workers 10` |
| third set, 300 curves, drawn with a new seed after the procedure was frozen, with the automatic analysis's own call (stage `auto`, first level as at commit 6bb1a18) | `posterior_newseed_summary.json` | `per_cut/posterior_newseed.json.gz` | `--n 300 --seed0 20261122 --auto --workers 8` |
| third set, the automatic analysis's call at the second level (since f73d9c8) | `posterior_newseed_auto2_summary.json` | `per_cut/posterior_newseed_auto2.json.gz` | `--n 300 --seed0 20261122 --stages auto --auto-level 2 --workers 8` |
| the automatic analysis's call (second level) on the 200 cuts simulated with the four-term DWBA on Si | `posterior_dwba_auto2_summary.json` | `per_cut/posterior_dwba_auto2.json.gz` | `--n 200 --scattering dwba --substrate si --cut born --stages auto --auto-level 2 --workers 8` |
| DWBA on Si, both angles given | `posterior_dwba_tp_summary.json` | `per_cut/posterior_dwba_tp.json.gz` | `--n 200 --workers 10 --scattering dwba --substrate si --cut two_path` |
| DWBA on Si, q_z given | `posterior_dwba_born_summary.json` | `per_cut/posterior_dwba_born.json.gz` | `--n 200 --workers 10 --scattering dwba --substrate si --cut born` |
| DWBA on Si, neither given | `posterior_dwba_eia_summary.json` | `per_cut/posterior_dwba_eia.json.gz` | `--n 200 --workers 10 --scattering dwba --substrate si --cut eia` |
| the DWBA runs paired with the first, by q₊R | `posterior_dwba_pairs.json` | – | `posterior_pairs.py`, below |
| Fitting's own model as the truth | `posterior_own_summary.json` | `per_cut/posterior_own.json.gz` | `--n 200 --workers 10 --truth own --seed0 20260601` |
| model error floor 3 % | `posterior_floor03_summary.json` | `per_cut/posterior_floor03.json.gz` | `--n 200 --workers 10 --model-error-floor 0.03` |
| model error floor 8 % | `posterior_floor08_summary.json` | `per_cut/posterior_floor08.json.gz` | `--n 200 --workers 10 --model-error-floor 0.08` |
| amplitude range 100 | `posterior_amplitude100_summary.json` | `per_cut/posterior_amplitude100.json.gz` | `--n 200 --workers 10 --constant infer.AMPLITUDE_RANGE=100` |
| time per curve, 24 cuts, one worker | `posterior_timing_summary.json` | `per_cut/posterior_timing.json.gz` | `--n 24 --workers 1` |

The commands as they were run (`OMP_NUM_THREADS=1`, `GIMAP_HOME` a temporary folder; the runs were written as
`NAME_new` and then stored here without `_new`, the per-cut files of `results/` gzip-compressed in `per_cut/`):

```bash
N=src/gimap/features/fitting/domain/posterior/data/flow_v1
python tools/benchmarks/posterior_recovery.py --network $N --out posterior_recovery_new --n 200 --workers 10
python tools/benchmarks/posterior_recovery.py --network $N --out posterior_extended_new --set extended --n 600 --seed0 20261201 --workers 10
python tools/benchmarks/posterior_recovery.py --network $N --out posterior_floor03_new --n 200 --workers 10 --model-error-floor 0.03
python tools/benchmarks/posterior_recovery.py --network $N --out posterior_floor08_new --n 200 --workers 10 --model-error-floor 0.08
python tools/benchmarks/posterior_recovery.py --network $N --out posterior_amplitude100_new --n 200 --workers 10 --constant infer.AMPLITUDE_RANGE=100
python tools/benchmarks/posterior_recovery.py --network $N --out posterior_timing_new --n 24 --workers 1
python tools/benchmarks/posterior_recovery.py --network $N --out posterior_dwba_tp_new --n 200 --workers 10 --scattering dwba --substrate si --cut two_path
python tools/benchmarks/posterior_recovery.py --network $N --out posterior_dwba_born_new --n 200 --workers 10 --scattering dwba --substrate si --cut born
python tools/benchmarks/posterior_recovery.py --network $N --out posterior_dwba_eia_new --n 200 --workers 10 --scattering dwba --substrate si --cut eia
python tools/benchmarks/posterior_recovery.py --network $N --out posterior_own_new --n 200 --workers 10 --truth own --seed0 20260601
R=tools/benchmarks/results
python tools/benchmarks/posterior_pairs.py tools/benchmarks/posterior_dwba_pairs.json born=$R/posterior_recovery_new.json si_tp_new=$R/posterior_dwba_tp_new.json si_born_new=$R/posterior_dwba_born_new.json si_eia_new=$R/posterior_dwba_eia_new.json
python tools/benchmarks/summary_tables.py            # the tables of the documents, from the summaries; --lang zh, --only NAME
```

`posterior_pairs.py` reads the files of `per_cut/` as well (`born=tools/benchmarks/per_cut/posterior_recovery.json.gz …`).
In `posterior_dwba_pairs.json` the runs keep the labels they were paired under: `born`, `si_tp_new`, `si_born_new`,
`si_eia_new`. In the summaries, *true composition most probable* (`composition.exact`) is judged on the most probable
solution: the first solution Predict lists has the true families; `true_in_equivalent_set` says that a listed
solution (probability at least 0.05) has them.

For spheres in the two-path DWBA comparison (`si_tp_new`), 68 % interval coverage was 60 %, 36 % and
48 % in the q₊R bins 1.2–1.7, 1.7–2.3 and 2.3–3.5. Predict therefore raises `qzR_cov` above 1.7 at
level 3 (`reporting.QZR_TWO_PATH_COVERAGE`). The frequent sphere/cylinder confusion in the DWBA comparison motivates a separate shape warning:
at levels 2 and 3 an upright cylinder above q₊R = 2 (`reporting.QZR_UPRIGHT_SPHERE`) gets `cyl_sphere`, including fits without a height.
The public GALAXI worked example gives q₊R = 3.13 for the 5.99 nm upright cylinder.

`history/` holds the summaries of earlier code and of removed methods; they are not the paper's numbers
(`history/README.md`). A new run writes `NAME_summary.json` here and `results/NAME.json` (not committed); files named
`*_new_summary.json` are left out of the repository.

## The scripts

| Script | What it measures | Summary |
|---|---|---|
| `recovery_cases.py` | the held-out curves of the 1D recovery benchmarks: fit1d's 200 (seed 20261116), their draw and their curve from the independent forward model `yoneda_truth.py`; shared by the scripts below | – |
| `fit1d_recovery.py` | how well *Predict parameters (fit1d)* (removed in 0.2.0; a copy of the kit in `GISAXS1D_KIT_HOME`) recovered known parameters on the 200 curves of `recovery_cases.py` (Physics and Methods, section 9.12); with `--set extended` on 1200 curves of `heldout_cases.py` | `history/fit1d_recovery_summary.json`, `history/fit1d_heldout_summary.json` (a new run writes beside the script) |
| `posterior_recovery.py` | how well *Predict parameters* (the proposal network, its posterior and refinement) recovers known parameters on fit1d's 200 held-out curves (`--set fit1d`) or on curves of `heldout_cases.py` (`--set extended`; 600 in the paper's run): errors, coverage of the 68 %/95 % intervals, how often the most probable solution is in the truth's mode, the composition chosen, time per curve on one core (Physics and Methods, section 9.13); `--truth own` draws the same cases in Predict's own model (the coverage when the model is exact), `--model-error-floor`, `--explored`, `--moves`, `--moved-solutions`, `--per-count` and `--widths` set those steps, `--constant module.NAME=value` any other constant of `refine` or `infer`; `--scattering dwba` draws the same curves with the four DWBA terms (`--substrate si` or `ps100`, 100 nm polystyrene on Si); `--cut` says what Predict is given of the cut: `born` its q₊ = k_i + k_f (the Born term at it; the default), `two_path` both angles (spheres with the exit-reflected path at q₋ = k_i − k_f and a free effective reflection coefficient), `eia` nothing (the effective interface) | `posterior_*_summary.json` and `per_cut/` (*The runs the paper quotes*, above); earlier runs in `history/` |
| `posterior_pairs.py` | pairs runs of `posterior_recovery.py` on the same cases that differ in the truth's physics or Predict's geometry: errors of R and D by q₊R for spheres and upright cylinders, composition and truth's mode by the largest sphere's q₊R, whether χ²/N marks the spheres that are off (Physics and Methods 9.13, *The waves the surface reflects*) | `posterior_dwba_pairs.json`; earlier ones in `history/` |
| `evidence_integration_check.py` | Predict's evidence, the scales integrated by Laplace at their least-squares values after the model error's spline coefficients (8 B-splines, a partition of unity, so a constant in them trades against every scale), against the scales and the coefficients integrated together (adaptive importance sampling, each value checked by a second run), on six one-population cuts of the first benchmark set (both halves) with four compositions each (the truth, without its paracrystal, the other family, two populations): the difference at the refined solution, at τ/2 and 2τ and at 40 draws of each solution's posterior, and what it changes in each solution's evidence and in the log Bayes factors against the truth (statistics referee, round 5, S7 ii). The joint integral is larger by 0.3–1.8 for the true composition (it grows with τ, spread 0.3–1.0 within a posterior); the log Bayes factors of the compositions that describe the curve change by −0.9 to +1.7 (+5.6 for one at χ²/N 1.5), those of compositions far off by +24 to +1350 in their favour, and no ranking changes | `evidence_integration_check.json` |
| `summary_tables.py` | prints the tables of Physics and Methods 9.13 and 10.3 and of the model card from the stored summaries (`--lang zh` for the Chinese headers, `--only recovery extended sensitivity own dwba pairs timing v5 large card`); computes nothing again | – |
| `search_baseline.py` | the baseline without a network: Fitting's *Search the ranges, then refine* from its default values on the same 200 held-out curves, the true composition given (median error of R 2.8 % / 6.7 %, 90th percentile 81 % / 136 %, about 10 s per curve; Predict parameters 0.5 % / 0.7 % and 10 % / 14 %, `posterior_recovery_summary.json`) | `search_baseline_summary.json` |
| `heldout_cases.py` | a second held-out set drawn as fit1d's, on a seed range never used during development (20261201): one to three populations, the LMA or the DA | – |
| `v5_recovery.py` | 1D Predict V5 (removed in 0.2.0; runs in a checkout of commit 06e7395) on fit1d's 200 held-out curves, the true composition given (Physics and Methods, section 10.3) | `history/v5_recovery_summary.json` |
| `galaxi_posterior.py` | *Predict parameters* on the measured GALAXI cut and on BornAgain's model of it (Physics and Methods, section 9.13) | `galaxi_posterior.json` |
| `analyze_speed.py` | how long Analyze takes to reduce the public frames of `tests/data/external` with their own `.poni` (first frame and every further one) | `analyze_speed.json` |
| `pyfai_comparison.py` | GIMaP against pyFAI on the public frames: |q|, 2θ, q∥, q_z, solid angle and polarisation of every pixel, and the azimuthal I(q) on the same pixels and bins (Physics and Methods 3.7, 6.1). Needs pyFAI, which GIMaP does not install | `pyfai_comparison.json` |
| `calibration_check.py` | Geometry Calibration checked (section below): GIMaP's calibration of the public pyFAI silver behenate frame against pyFAI's own calibration of the same frame, what GIMaP minimises and which 1σ it reports, and how often that 1σ covers the truth on simulated frames of known geometry (`calibration_coverage.py`). Part 1 needs pyFAI | `calibration_check.json`; every simulated frame in `per_cut/calibration_check_frames.json.gz` |
| `pygid_comparison.py` | GIMaP against pygid on its two example frames: q∥ and q_z of every pixel, and the rings of pygid's polar map; with `--published DIR`, also the maps published with the examples (Zenodo 10.5281/zenodo.17466183) (Physics and Methods 3.7). Needs pygid | `pygid_comparison.json` |
| `crystal_orientation.py` | Joint fits to Poisson draws of known MAPbI₃ and PbI₂ fibre textures and to the public P08 MAPbI₃ frame, using the supplied CIFs | [Results below](#crystal-orientation) |
| `make_external_reference.py` | writes `tests/data/external/reference_values.json`: pyFAI's and pygid's q at fixed pixels of the public frames, which `tests/test_external_reference.py` checks GIMaP against without either installed. Needs pyFAI and pygid | `tests/data/external/reference_values.json` |
| `galaxi_fit.py` | Fitting on the measured GALAXI cut, sphere and paracrystal (EIA or two paths, LMA or DA, with BornAgain's D held or free) (Physics and Methods 9.10) | `galaxi_fit.json` |
| `galaxi_fit1d.py` | fit1d (removed in 0.2.0; a copy of the kit in `utils/gisaxs1d_gui_kit`) on the measured GALAXI cut and on BornAgain's model of it: one sphere in Born or DWBA, and the composition chosen by fit1d (Physics and Methods 9.12). Needs BornAgain | `galaxi_fit1d.json` |
| `galaxi_bornagain_model.py` | BornAgain's model of the GALAXI frame on the same cut: how well it describes the measured cut, and what Fitting recovers from it (Physics and Methods 9.10). Needs BornAgain | `galaxi_bornagain_model.json` |
| `cut_model_checks.py` | `cut_model`, GIMaP's one 1D forward model, against Fitting's formulas before 0.3, 1D Predict V5's forward, the fit1d kit's tables (when a copy of the kit is in `utils/gisaxs1d_gui_kit`) and the independent `yoneda_truth.py` (Born at the cut's q_z and four-term DWBA, LMA and DA); time of 1000 curves on one core (Physics and Methods 9.2) | `cut_model_checks.json` |
| `cut_model_bornagain.py` | `cut_model` against BornAgain: spheres and upright cylinders on Si in the four-term DWBA, one size, a size distribution and a radial paracrystal (Physics and Methods 9.2). Needs BornAgain | `cut_model_bornagain.json` |
| `vertical_height_bias.py` | the height of upright cylinders from a joint fit of a horizontal and a vertical cut (`application.joint_fit`, the vertical cut in the four-term DWBA on Si) when BornAgain's cuts hold what that model leaves out: the average layer at coverage 10–40 % with Si and Au particles, a rough substrate (σ 0.5 and 1 nm), the detector resolution and beam divergence, heights correlated with radii, all at once; each noise-free and with 10 Poisson draws, the whole vertical cut and from 2α_c (referee round 5, M8; section below). Needs BornAgain | `vertical_height_bias.json` |
| `yoneda_truth.py` | the forward model of a horizontal GISAXS cut used by the fit1d benchmark: sphere or upright cylinder on a substrate, four-term DWBA, paracrystal, resolution, beam tail, roughness, counts. It shares no code with GIMaP or fit1d | – |

The other scripts (the commands of the paper's Predict runs are above):

```bash
GISAXS1D_KIT_HOME=<copy of the kit> python tools/benchmarks/fit1d_recovery.py --n 200 --workers 10
GISAXS1D_KIT_HOME=<copy of the kit> python tools/benchmarks/fit1d_recovery.py --set extended --n 1200 --seed0 20261201 --out fit1d_heldout
python tools/benchmarks/search_baseline.py --n 200 --workers 10
python tools/benchmarks/galaxi_posterior.py   # in an environment with BornAgain 24
python tools/benchmarks/analyze_speed.py --repeat 5 --json tools/benchmarks/analyze_speed.json
python tools/benchmarks/pyfai_comparison.py --json tools/benchmarks/pyfai_comparison.json  # in an environment with pyFAI
python tools/benchmarks/calibration_check.py --part all --cases 300 --workers 3  # part 1 with pyFAI; --part coverage needs none
python tools/benchmarks/pygid_comparison.py --json tools/benchmarks/pygid_comparison.json --published FOLDER  # with pygid
python tools/benchmarks/galaxi_fit.py
python tools/benchmarks/galaxi_bornagain_model.py
python tools/benchmarks/cut_model_checks.py            # on a clean checkout: add --out <file> (or --force)
python tools/benchmarks/cut_model_bornagain.py   # in an environment with BornAgain 24
python tools/benchmarks/vertical_height_bias.py  # with BornAgain 24; about 3 minutes with two workers
python tools/benchmarks/evidence_integration_check.py   # about 25 minutes on one core
```

Times depend on the computer; `analyze_speed.json` names the one it was measured on.

## The height from a vertical cut, against BornAgain with what the model leaves out

`vertical_height_bias.py` (referee round 5, M8). BornAgain 24 simulates the Yoneda row (α_f = α_c(Si)) and the
vertical cut at q_y ≈ 0.3 nm⁻¹ (α_f 0.02–1.2°) of upright cylinders on Si (R = 4 nm, H = 6 nm, σ_R/R = σ_H/H = 0.15,
independent; radial paracrystal, σnn/D = 0.2; λ 0.1045 nm, α_i 0.40°), scaled to 10⁴ counts at the maximum on 2
counts of background. Fitting fits both cuts together as a user would (*Fit Several Curves Together*, Refine, the
substrate Si known, R, σR/R, h, σh/h, D, σnn/D shared, the DA as BornAgain's Mixture), from R 4.3 nm, h 6.6 nm, D
+7 %. The coverage is BornAgain's density of a radial paracrystal, 2/(√3 D²), times π⟨R²⟩; the average layer is cut
into 24 slices. BornAgain's average layer was checked first against the four-term DWBA in a uniform layer of the
average index (one size, one slice): the shapes agree within 1.5·10⁻³. Biases below are of the noise-free fit; the
mean of 10 Poisson draws lies within 0.1 % of it, and their scatter (0.03–0.3 % of h) matches the 1σ the fit reports
where the model holds. The table records the original sweep, before the horizontal DA included height averaging.
χ²_r is the joint reduced χ² averaged over the draws. Per-cut Σr²/N is recorded separately in the JSON.

| Case | Coverage | h bias, whole cut | R bias | D bias | χ²_r | h bias, from 2α_c | χ²_r |
|---|---:|---:|---:|---:|---:|---:|---:|
| average layer off, smooth (D 14 nm) | 30 % | +0.0 % | +0.1 % | -0.2 % | 1.1 | -0.0 % | 1.1 |
| average layer off, smooth (D 24.4 nm) | 10 % | -0.0 % | -0.6 % | -0.1 % | 1.1 | +0.0 % | 1.1 |
| average layer off, smooth (D 12.2 nm) | 40 % | +0.1 % | +0.4 % | -0.2 % | 1.2 | -0.0 % | 1.2 |
| average layer, 10 % Si | 10 % | -2.7 % | -0.6 % | -0.1 % | 2.8 | -1.1 % | 1.1 |
| average layer, 10 % Au | 10 % | -13.0 % | -0.8 % | -0.1 % | 26.6 | -5.8 % | 1.5 |
| average layer, 30 % Au (D 14 nm) | 30 % | -51.4 % | +0.1 % | -0.4 % | 126.3 | -16.7 % | 4.9 |
| average layer, 40 % Si | 40 % | -16.6 % | +0.5 % | -0.3 % | 14.9 | -5.2 % | 1.4 |
| average layer, 40 % Au | 40 % | -60.5 % | +1.0 % | -0.5 % | 220.1 | -19.7 % | 8.8 |
| roughness 0.5 nm (r and diffuse) | 30 % | -0.1 % | -0.1 % | -0.2 % | 1.4 | -0.1 % | 1.4 |
| roughness 1.0 nm (r and diffuse) | 30 % | -0.8 % | -0.3 % | -0.2 % | 3.1 | -0.8 % | 3.3 |
| roughness 0.5 nm (r only) | 30 % | -0.1 % | +0.1 % | -0.2 % | 1.1 | -0.1 % | 1.1 |
| roughness 1.0 nm (r only) | 30 % | -0.6 % | +0.1 % | -0.2 % | 1.1 | -0.8 % | 1.1 |
| resolution σ 0.01° | 30 % | -0.7 % | +0.1 % | -0.2 % | 2.8 | -0.1 % | 1.2 |
| resolution σ 0.02°, divergence σ 0.01° | 30 % | -2.5 % | +0.0 % | -0.2 % | 9.7 | -0.2 % | 1.2 |
| H = 1.5 R (correlated) | 30 % | +14.4 % | +0.6 % | -0.2 % | 1.2 | +14.2 % | 1.2 |
| 40 % Au, average layer, roughness 1.0 nm, resolution 0.02° + divergence 0.01° | 40 % | -59.6 % | +1.0 % | -0.6 % | 242.4 | -19.5 % | 10.3 |

**Height consistency between fit ranges.** We paired the stored full-range and 2α_c fits in
`vertical_height_bias.json` by case and Poisson seed. With h from the selected 2α_c range and h′ from the
full cut, the warning fires if |h − h′| exceeds either 2√(σ_h² + σ_h′²) or 0.02|h|. The two fits
share data, so the quadrature sum serves as a diagnostic scale. The counts below refer to this range check
alone. The table above gives joint reduced χ². The separate χ² warning in the program uses the vertical cut.

| Cases in the original sweep | Noise-free range warning | Poisson draws flagged per case |
|---|---|---|
| All five Si/Au average-layer cases, 0.02° resolution with divergence, combined 40 % Au case | Yes | 10/10 |
| 0.01° resolution alone | Yes | 5/10 |
| Three smooth controls, four roughness-only cases, H = 1.5R | No | 0/10 |

For 10 % Au and 40 % Si, the noise-free heights changed by 7.6 % and 12.0 % of the selected height.
With H = 1.5R they changed by only 0.18 % despite a +14 % bias. Agreement between the two ranges
can therefore coexist with a bias from correlated heights and radii.

The roughness is BornAgain's (K-correlation, Hurst 0.7, lateral correlation 25 nm): *r only* subtracts the bare rough
substrate's cut, which leaves the reduced reflection coefficients; *r and diffuse* keeps the substrate's own diffuse
scattering, which raises χ²_r through the horizontal cut. The resolution is a Gaussian detector resolution in α_f and
φ_f, with a Gaussian beam divergence in α_i in the second case. The original horizontal DA used only the radial
coherent term. It fitted σR/R of 0.16–0.19 for distributed heights and 0.150 for one height.

**Baseline with the height average.** For each height H, the four axial paths give a complex amplitude G(H),
including the factor H from the cylinder volume. The horizontal DA now multiplies its radial coherent term
by |⟨G⟩|²/⟨|G|²⟩. In the D = 14 nm baseline, with the average layer off, the noise-free whole-cut fit recovered
σR/R = 0.150006 and h = 5.99987 nm. From 2α_c it recovered 0.150005 and 5.99967 nm. Inputs were 0.15 and 6 nm.
Both fits retained all 471 horizontal points and respectively 300 and 228 vertical points.
The mean joint χ²_r over three Poisson draws was 1.01 in each window.
Run `python tools/benchmarks/vertical_height_bias.py --cases a_off_D14 --seeds 3 --workers 1 --out baseline.json`.
The output is stored under `height_averaged_da_baseline` in `vertical_height_bias.json`.

## Online GISAXS series

`insitu_online.py --synthetic-mpx 1 4 --frames 8 --json tools/benchmarks/insitu_online.json` processes eight
synthetic, locally written TIFFs of each detector size. The sphere radius steps from 5 to 7 nm at frame 5, which
is supplied to the chain as a new stage. One forward pass fits the horizontal cut over 0.05–1.5 nm⁻¹,
averaging both halves (369 points). Predict uses the model's one sphere and paracrystal at the break.

| Detector | Reduction (s/frame) | Cuts (s/frame) | Chained fit (s/frame) | Predict (s/call) | Predict averaged over all frames (s/frame) |
|---|---:|---:|---:|---:|---:|
| 1000 × 1000 | 0.0718 | 0.00467 | 0.167 | 2.58 | 0.323 |
| 2000 × 2000 | 0.265 | 0.00583 | 0.103 | 2.47 | 0.309 |

Each number is an arithmetic mean on an Intel Core i9-13900HX with one worker and the OpenMP, MKL and OpenBLAS
thread limits set to one. Reduction includes reading the TIFF and finding bad pixels. Cuts include the Yoneda
search and the horizontal and vertical profiles. Chained fit includes refinement after a restart. Predict was
timed separately in the same process and ran once per detector. The times cover processing only. Frame
generation, watch polling, process startup, screen redraws and export are outside the stopwatch.
The JSON gives each frame's timings, settings and the network checksum. Running the script without
`--synthetic-mpx` times the watched folder on crops of the public GALAXI frame.
## Geometry Calibration against pyFAI, and the coverage of its 1σ

Run `calibration_check.py --part all --cases 300 --workers 3` with pyFAI installed.
The simulation alone (`--part coverage`) uses the GIMaP environment. `--part summarize` rebuilds the coverage
tables from `per_cut/calibration_check_frames.json.gz`. The JSON records the seed, settings and command.

**Geometry and uncertainty.** GIMaP fits the beam centre, sample-to-direct-beam distance L and two detector
rotations to the ring points. Each point is a centroid from a sector about 2 px of arc wide,
weighted by recorded intensity above the sector's lower quartile. The sector must contain valid
pixels on both flanks of the predicted ring. A detector edge or gap can otherwise truncate a peak,
pulling the centroid toward the visible side and biasing the tilt fit.
The residual in pixels is r_i = L (tan 2θ_i − tan 2θ_ring) / p̄, where p̄ is the mean pixel size and 2θ_ring
comes from the order of AgBh (d₀₀₁ = 58.38 Å). The fit minimises Σ f²(√(1 + r_i²/f²) − 1) with f = 2 px.
After the initial fit, the engine searches narrow radial bands along the predicted ellipses.
It tests each sector against the background on both sides of its peak. Once two orders are confirmed,
the refit uses the confirmed orders, including any missing from the initial assignment. Independently
resolved close reflections can also enter when their spacing leaves too little room to test the flanks.
Automatic tilt refinement needs usable centroids from at least three orders.
The final ranking counts confirmed rings, penalises missing observable
orders and unexplained strong radial peaks, and uses the ring-point RMS of the fitted geometry.
The benchmark records the points used in the final fit to check the uncertainties independently.

For n contributing rings, the leave-one-ring-out variance is v_J = (n − 1)/n Σ_k (θ_k − θ̄)².
The full-fit covariance is C = [Σ_i r_i²/(N − p)] (JᵀJ)⁻¹ for N points and p free parameters.
With at least three rings, GIMaP reports √v_J for each free parameter. With exactly three and fitted
tilts, all three two-ring refits must converge, constrain all five parameters and stay within 25% of
the full distance and ±30° tilt. Identifiability requires the smallest singular value of the unit-column
Jacobian to exceed √ε times the largest (ε is machine precision). If any subset fails these checks,
all three refits hold tilt at its full-fit values. The conditional fallback reports √max(v_J, C_jj)
for x, y and L, and √C_jj for the tilts. With two rings, the standard errors come from C.

**Public silver behenate frame.** GIMaP matched orders 1–5 on `saxs_agbh_pyfai/Pilatus1M.edf` at λ = 1.0 Å
and 172 µm pixels. pyFAI 2026.9 located points on six rings using `Recalibration.extract_cpt` (massif, one
point per degree, fixed seed) and repeatedly refined its geometry with `GeometryRefinement.curve_fit` and
rot3 = 0. The table uses pixel-corner coordinates and sample-to-direct-beam distance. The last column divides
the difference by the two standard errors added in quadrature.

| Parameter | GIMaP ± ring jackknife | pyFAI ± curve_fit [ring jackknife] | Difference / combined σ |
|---|---:|---:|---:|
| beam x (px) | 179.736 ± 0.076 | 179.923 ± 0.017 [0.073] | −2.41 |
| beam y (px) | 264.012 ± 0.154 | 263.968 ± 0.018 [0.094] | +0.29 |
| L (mm) | 1633.47 ± 0.35 | 1634.05 ± 0.07 [0.17] | −1.65 |
| rot1 (°) | −0.305 ± 0.148 | −0.297 ± 0.035 [0.089] | −0.05 |
| rot2 (°) | 0.670 ± 0.178 | 0.687 ± 0.035 [0.091] | −0.10 |

The two fits differed by 0.193 px in beam position and 0.580 mm in distance. The headless line
check returned the same five-order geometry to the precision shown. A half-distance candidate
places the same five image rings at orders 2, 4, 6, 8 and 10. Its smaller mean relative q error
on those five orders previously overrode the window's solution. The headless AgBh comparison
now measures competing candidates on the independently confirmed orders of the leading geometry.
On GIMaP's ring points the RMS residuals of the GIMaP and pyFAI geometries were 0.142 and 0.297 px.
On pyFAI's points they were 0.349 and 0.222 px. The shipped `Pilatus1M.poni`, a starting geometry for
pyFAI, gave 14.4 and 15.3 px on these two sets. `calibration_check.json` also gives both geometries in PONI
coordinates and the results of a second pyFAI fit started 3% farther away with zero tilt.

**Recovery and coverage.** The 300 simulated Pilatus 1M frames have 3–9 target rings, L between
1.5 and 3 m, and rotations up to ±3° about either axis. Each fit receives an estimate within ±15% of the
true distance. Recovery requires a fitted tilt with an uncertainty, a beam centre within 5 px of the
truth and a distance error below 5%. Counts and coverage use the calibration window's first-ranked
candidate. The archive records the headless line check separately. The ring-count classes below
are the simulation targets. Additional orders can reach a corner of the detector.

**Diagnosis on the original frames.** At the true centre of case 0, orders 3–5 peaked at
563, 751 and 939 px. Only 1.1%, 0.6% and 0% of the rays passed the old angular support test,
below the 1.5% cutoff, despite clear arcs in the image. A bright sector had to exceed the
median excess of the other sectors at the same radius. The revised test compares each
sector to its own radial background. In case 3, the polished centre was 0.4 px from truth.
Three strong peaks at 175, 350 and 525 px agreed on a distance of 1964 mm. A broad background
peak at 811 px was assigned to order 5 and pulled the initial least-squares fit to 1871 mm.
AgBh matching now requires agreement with the harmonic spacing before fitting the distance.
In case 208 the selected solution put the detector at half its true distance. It predicted
intermediate orders in regions where the image shows background. All three cases had a
centre proposal within 1 px of truth and pass with the existing centre search.

In the first guided run the rot1 errors still depended on where the rings reached the detector.
After comparing centroid methods on 40 frames with at least seven matched rings, we checked 32
other frames in the same class. Using recorded counts and requiring both flanks reduced RMS
error from 0.044 to 0.004 px for x, 0.032 to 0.003 px for y, 0.156 to 0.016 mm for L,
0.141° to 0.012° for rot1 and 0.098° to 0.017° for rot2. Both fits used the same assignments.

Replaying all 87 failures with the previous engine found a centre within 5 px in 73 frames
before polishing and in all 87 after polishing. The table identifies the earliest missing
step at centres within 5 px. This audit counts a true order when the simulated ring crosses
at least four 5° sectors on active pixels. A strong coarse peak has prominence ≥0.1 and lies
within max(2.5 px, 0.012 r) of the nominal radius. An accurate assignment meets the same
5 px and 5% tolerances as the benchmark and matches at least three true orders.

| Earliest missing step in the previous engine | Previous geometry found | Current geometry found |
|---|---:|---:|
| Fewer than three strong true orders detected at any accurate centre | 0/12 | 12/12 |
| Three detected, no accurate three-order assignment generated | 0/7 | 7/7 |
| Accurate assignment generated, lost during refinement or ranking | 0/68 | 68/68 |

The shipped engine found **300/300 geometries** on the same seeds for which commit `7fc06ff` found
213/300. It retained all 213 successful fits, recovered all 53 frames previously missing a tilt fit
and all 34 previously assigned a wrong geometry. Among those 34, the old fit missed both tolerances
on 25 frames, only the distance tolerance on six, and only the centre tolerance on three.

| Simulation group | Previous | Current |
|---|---:|---:|
| 3 target rings | 41/100 | 100/100 |
| 4–6 target rings | 74/100 | 100/100 |
| 7–9 target rings | 98/100 | 100/100 |
| Tilt <1° | 24/30 | 30/30 |
| Tilt 1–<2° | 63/87 | 87/87 |
| Tilt 2–<3° | 79/117 | 117/117 |
| Tilt ≥3° | 47/66 | 66/66 |

Tilt in this table is the combined angle arccos(cos rot1 · cos rot2). The separate set with only
orders 1–3 drawn improved from 114/150 geometries recovered to 150/150, and from 112 to 150
recoveries with exactly three matched orders. The headless workflow also recovered all 450 geometries.

Mean worker time for calibration and the headless line check rose from 13.371 to 15.502 s per frame
on the main set (median 13.090 to 15.115 s), and from 9.143 to 11.514 s on the separate set.
These times exclude simulation and the benchmark's independent uncertainty refits. The archived
baseline used four workers and this run used three.

Among recovered frames, each cell gives the measured fraction within 1σ, fraction within 2σ and median |error|/σ.
The first row uses the independent 150-frame set with only orders 1–3 drawn. The remaining rows use the
300-frame set, grouped by the number of matched rings.

| Matched rings | Frames | beam x | beam y | L | rot1 | rot2 |
|---|---:|---|---|---|---|---|
| 3 (orders 1–3 only) | 150 | 0.613 / 0.820 / 0.666 | 0.687 / 0.853 / 0.533 | 0.593 / 0.820 / 0.754 | 0.640 / 0.873 / 0.745 | 0.673 / 0.833 / 0.596 |
| 3 | 6 | 0.667 / 0.833 / 0.737 | 0.667 / 1.000 / 0.396 | 0.833 / 0.833 / 0.440 | 0.333 / 0.833 / 1.397 | 0.500 / 0.667 / 1.437 |
| 4–6 | 138 | 0.594 / 0.855 / 0.684 | 0.616 / 0.862 / 0.699 | 0.565 / 0.819 / 0.747 | 0.623 / 0.862 / 0.745 | 0.565 / 0.841 / 0.874 |
| ≥7 | 156 | 0.660 / 0.955 / 0.684 | 0.679 / 0.904 / 0.598 | 0.622 / 0.891 / 0.792 | 0.641 / 0.923 / 0.710 | 0.609 / 0.891 / 0.832 |

With few rings, estimated standard errors themselves vary. A Student t reference with n − 1 degrees
of freedom gives 57.7%, 61.9% and 65.4% within 1σ for the separate three-ring set, the 4–6 group and
the ≥7 group respectively. Coverage for L and rot2 was 56.5% in the 4–6 group. With ≥7 rings, taking
the larger of the covariance and ring-jackknife errors would give 76.9–84.6% across the five parameters.
The reported values use the ring jackknife. Repeating it on the final points agreed with the stored
standard errors to within 1.38 × 10⁻⁵ relative across all 450 frames.

## The third set

`posterior_newseed` was run at commit 6bb1a18 (fitting tree d139e6f5). The fitting code differs from the tree of the other runs (7e68831a) in its reporting. The five-cut regression above checks the archived numerical results from three runs. The seed (20261122) was chosen after the last change of the procedure, so no change was judged on it. The stage `auto` in this archive is the automatic analysis's call as it was then: the first level (q_z = 0), spheres and upright cylinders, one or two populations.
`posterior_newseed_auto2` repeats the stage `auto` on the same 300 cuts after the automatic analysis was changed to fit at the second level when it knows both angles (commit f73d9c8): the true composition is the most probable in 75 % of the cuts (27 % at the first level) and the median error of R is 0.8 % (13 %).

`posterior_dwba_auto2` (commit 9c90e6a): with DWBA truth the automatic analysis's call chooses the true composition in 31 % of the cuts; by the largest q_zR of a cut: 80 % (median |ΔR/R| 1 %) at 1.2–1.7, 29 % (10 %) at 1.7–2.3 and 23 % (9 %) above 2.3; cuts of spheres alone 9 %, of cylinders alone 78 %. The q_zR flags of Predict mark the solutions above these limits.

## Crystal orientation

Run `python tools/benchmarks/crystal_orientation.py --part all --trials 32 --json results.json`. The JSON contains each simulated fit and its 1σ errors, the public-frame report, the film-absorption refit, an independent-width refit on the original angular cut, and signed-sector measurements around the strongest peak.

### Synthetic recovery

Three rings with independent scales sample each simulated fibre distribution. Direct integration of the antipodal Gaussian uses 96 azimuthal and 16 within-bin Gauss nodes. Individual poles contribute their calculated |F|². Two independent Poisson draws at ±χ are folded with their counting errors after subtraction of the known background. The missing wedge below 5° and a detector gap at 42–45° are present in every draw. The two single-component sets use αf ≥ 0.30°. The mixed set includes the exit field for δ = 4.2 × 10⁻⁶, β = 3.1 × 10⁻⁷ at λ = 1 Å and αi = 0.15°, with all bin centres above the horizon accepted. Its true (110) populations have widths 8° and 36° and absolute fractions 0.186 and 0.434.

Seeds 41027–41058 give the following results. Width and population-fraction coverage counts are conditional on selecting the generating model. Isotropic coverage includes all 32 draws.

| Phase and true texture | Correct model | Correct and determined | Width within 1σ | Population fraction within 1σ | Isotropic fraction within 1σ | Mean joint χ²red |
|---|---:|---:|---:|---:|---:|---:|
| MAPbI₃ (110), 16°, φ = 0.38 | 29/32 | 29/32 | 19/29 | 18/29 | 18/32 | 1.00 |
| PbI₂ (001), 22°, φ = 0.29 | 30/32 | 30/32 | 20/30 | 17/30 | 17/32 | 1.01 |
| MAPbI₃ (110), 8° + 36°, φ = 0.38 | 32/32 | 32/32 | 16/32, 17/32 | 20/32, 21/32 | 22/32 | 1.00 |

Every selected model included the true axis. One single-component PbI₂ draw passed the determination checks with a spurious second (001) population of about 0.9%. The other four single-component draws with an extra population failed the determination checks. Interval coverage reflects counting noise, the estimated bin weights and the model selection step.

### Public grazing-incidence frames and sample roll

Run `python tools/benchmarks/sample_roll.py` for the frame estimates and the P08 signed-ring check. Each frame uses its published αi, the adjacent PONI and Analyze’s default bad-pixel rejection. Xeuss also uses its published mask. Positive ψ tips the surface normal toward laboratory right. The columns give degrees and estimated standard errors.

| Frame | Horizon or Yoneda band | Ring symmetry or GISAXS rod | Selected ψ | Automatically applied ψ |
|---|---:|---:|---:|---:|
| P08 MAPbI₃ | −1.50 ± 1.63 | −1.30 ± 0.25 | −1.30 ± 0.25, combined | −1.30459 |
| Xeuss PbTe | −0.90 ± 0.78 | Unresolved | −0.90 ± 0.78, horizon | 0 |
| GALAXI GISAXS | 0.00 ± 0.23 | Unresolved | 0.00 ± 0.23, band | 0 |

The P08 normal leans left. The horizon uncertainty reflects variation across detector strips. The ring carries most of the weight in the combined estimate. The Xeuss beam is only 7 px from the left edge, and the GALAXI frame gives no independently resolved specular-rod slope. Only P08 passes the three-standard-error rule.

The same command measures incidence independently of the supplied αi. The table gives degrees and 1σ errors at the calibrated direct beam. “Used” applies the automatic three-standard-error comparison with the published value.

Synthetic frames with a shadow edge and a specular spot at true αi = 0.310°, 0.520° and 0.450° gave 0.311 ± 0.021°, 0.519 ± 0.021° and 0.455 ± 0.021°, respectively, with 1.89° entered. These tests use tilted, non-square pixels and ψ = 2.2°. Both constituent estimates recover the generated angle within one reported standard error.

| Frame | Published αi | Shadow intercept | Specular spot | Yoneda band | Combined αi | Used |
|---|---:|---:|---:|---:|---:|---:|
| P08 MAPbI₃ | 0.075 | 0.65 ± 0.57 | 0.5012 ± 0.0119 | — | 0.5012 ± 0.0119 | 0.5012 |
| Xeuss PbTe | 0.235 | −0.067 ± 0.200 | — | — | Unresolved | 0.235 |
| GALAXI GISAXS | 0.463 | — | 0.4615 ± 0.0042 | 0.444 ± 0.026 | 0.4611 ± 0.0042 | 0.463 |

The P08 specular-like maximum lies at (543.84, 1755.95) px. The full shadow trace crosses about 46 px above the direct beam, but using only the nearer half of the strips gives 0.085°. This strip dependence limits what the edge alone establishes. The selected angle is set chiefly by the spot.

On Xeuss the beam position prevents an independent bilateral spot measurement, and the shadow intercept is consistent with zero. On GALAXI the band centre is at row 607.5 px. Its incidence estimate assumes the Si substrate critical angle 0.19483° from δ = 5.7816 × 10⁻⁶ in the BornAgain GALAXI model. The multilayer also contains PTFE and HMDSO. The specular spot by itself agrees with 0.463°. [Public geometries and source files](../../tests/data/external/README.md)

### P08 crystal orientation at the measured incidence and roll

On the 0.9973 Å⁻¹ ring, the vacuum horizon obeys `cos χ = (2π/λ) sin αi / q`. With λ = 0.688801 Å, its predicted |χ| is 85.41° at the measured incidence and 89.31° at the notebook angle. The observed loss around 84–86° agrees with the former. This check does not enter the incidence estimate.

`python tools/benchmarks/crystal_orientation.py --part public --json measured.json` runs the automatic analysis with the published αi as its reference. For this exposure the reduction uses αi = 0.50124° and ψ = −1.30459°. The JSON keys `crystal_orientation` and `benchmark_film_absorption` contain the exit-transmission fit and the refit including film absorption, respectively. Both use the same CIFs, measured pixels, uncertainties, integration windows and per-ring intensity scales.

The sample YAML gives the coating and annealing conditions but no measured film thickness. The refit allows one effective optical depth τ = μd per phase, common to its rings. With s = csc αi + csc αf and s₀ = 2 csc αi, the normalized F_A = [(1 − exp(−τs))/s] / [(1 − exp(−τs₀))/s₀] follows §5.4 of the Physics Manual. The model multiplies T_f by F_A on each measured ray before binning and folding. It uses geometrical path lengths, an approximation near the critical angle. The extra parameter enters BIC, the covariance and effective degrees of freedom. Both phases reach the thick limit F_A = s₀/s, which supplies no finite estimate of d. [P08 sample and notebook](https://zenodo.org/records/17466183)

| Phase and q (Å⁻¹) | χ²red, T_f | χ²red, T_f × F_A |
|---|---:|---:|
| **MAPbI₃ joint (746 bins)** | **18.32** | **4.24** |
| 0.9973 | 34.94 | 5.96 |
| 1.4091 | 7.75 | 2.71 |
| 1.7248 | 12.14 | 5.92 |
| 1.9903 | 26.37 | 3.97 |
| 2.2256 | 30.19 | 2.94 |
| 2.4382 | 6.61 | 3.88 |
| 2.8148 | 22.03 | 5.25 |
| 2.9854 | 21.47 | 3.78 |
| 3.4480 | 4.39 | 3.76 |
| 4.1040 | 3.71 | 3.66 |
| **PbI₂ 0.8989 (83 bins)** | **1.90** | **1.86** |

For MAPbI₃ the best pair changes from (102) + (001) with T_f to (221) + (110) with absorption. The latter gives conditional widths of 78.6 ± 6.1° and 11.25 ± 0.39° and fractions of 0.74 ± 0.39 and 0.046 ± 0.003. Three rings still exceed the unchanged χ²red limit of 5. The data do not establish the fibre planes or their populations. The largest residuals on the eight innermost rings are negative at |χ| = 84.5–88.5°. At 84.5° on the 0.9973 Å⁻¹ ring, the measured folded signal is 28.2 ± 3.3 and the model gives 84.4, an excess of 17.2σ. At 85.5° the signal is −0.4 ± 3.6 and the model gives 3.6. The film model describes much of the overall angular attenuation but misses the sharp loss just above the horizon. The outer two rings have their largest residuals at 69.5° and 26.5°.

PbI₂ retains two basal (001) components under both models. With absorption, the widths are 13.9 ± 1.7° and 53.4 ± 5.1° and their fractions are 0.055 ± 0.041 and 0.62 ± 0.47. The BIC difference between the two absorption treatments is only 0.78 on this single reflection. Its broad fraction and the optical depth are poorly constrained.

### P08 crystal orientation before and after roll

To reproduce the comparison at the published angle, run `python tools/benchmarks/crystal_orientation.py --part public --public-incidence-deg 0.075 --public-roll-deg 0 --json zero.json`, then omit `--public-roll-deg 0` to measure the roll. Both runs use the same two CIFs. Each ring has its own intensity scale. The central columns give χ²red for the original exit-angle cut and the CIF-based exit-transmission fit. The last column compares the two raw arms, using RMS of the difference between mean pixel intensities in matching 2° |χ| bins. Each bin must have at least 12 pixels on each arm in both runs.

| Phase and q (Å⁻¹) | Original cut, ψ = 0 → measured | Exit transmission, ψ = 0 → measured | Paired bins | ±χ RMS, ψ = 0 → measured |
|---|---:|---:|---:|---:|
| **MAPbI₃ joint** | **8.20 → 9.71** | **33.07 → 22.45** | | |
| 0.9972 | 16.86 → 19.61 | 40.15 → 37.73 | 44 | 47.43 → 15.96 |
| 1.4091 | 2.88 → 3.06 | 13.26 → 14.51 | 28 | 3.09 → 4.57 |
| 1.7247 | 8.31 → 7.84 | 14.53 → 13.49 | 20 | 2.17 → 2.69 |
| 1.9902 | 11.24 → 8.27 | 59.36 → 63.39 | 16 | 14.83 → 13.25 |
| 2.2255 | 11.11 → 20.87 | 123.25 → 26.69 | 13 | 4.12 → 3.58 |
| 2.4382 | 2.62 → 2.54 | 8.94 → 8.15 | 12 | 1.06 → 1.59 |
| 2.8148 | 8.93 → 10.73 | 20.88 → 22.11 | 9 | 2.34 → 3.43 |
| 2.9854 | 7.56 → 10.60 | 8.44 → 10.73 | 8 | 1.62 → 2.40 |
| 3.4480 | 4.85 → 4.00 | 5.69 → 4.86 | 6 | 0.91 → 0.65 |
| 4.1040 | 2.82 → 3.21 | 2.95 → 3.51 | 0 | — |
| **PbI₂ 0.8989** | **1.43 → 1.32** | **2.76 → 2.08** | 44 | 3.38 → 2.50 |

At |χ| = 84–86° on the strongest MAPbI₃ ring, the negative and positive arm means change from 82.4 and 10.5 to 26.7 and 33.8 counts/pixel. Over 87 common 1° bins their RMS difference falls from 45.6 to 20.2. This improvement is largest on the ring with both ends visible. Some outer rings have higher bilateral residuals after correction, and the outermost has no bin with enough pixels on both sides.

With exit transmission the best MAPbI₃ candidate remains (212) + (001). Their FWHM values change from 30.0° and 58.5° to 33.3 ± 3.4° and 53.6 ± 2.9°, with fitted fractions 0.64 ± 0.42 and 0.36 ± 0.21. The fit admits 742 bins after roll, up from 738. Its residuals still exceed the determination threshold, so the data do not establish these populations. On the original cut both runs prefer two (001) components. PbI₂ retains two basal (001) components on its single ring. With exit transmission their widths change from 16.6° and 65.4° to 16.4 ± 1.7° and 66.9 ± 6.9°, with fitted fractions of 8.9% and 91.1%.

### P08 film at fixed ψ = 0 (reference)

The benchmark uses the dark-subtracted frame, its PONI (λ = 0.688801 Å) and αi = 0.075° from the public example. The peak search assigns ten rings to tetragonal MAPbI₃ and one to PbI₂. The profiles use pixel-scatter errors, a locally fitted radial background constant in χ, polarization 0.99 and the original q windows. The MAPbI₃ CIF supplies four CH₆NPbI₃ units per cell for the index, and refined Pb and I sites for diffraction. These give ρₑ = 1.0423 e Å⁻³, δ = 2.18265 × 10⁻⁶, β = 1.26942 × 10⁻⁷ and αc = 0.1197°. The PbI₂ fixture now uses the symmetry of its declared space group P −3 m 1, which gives one Pb and two I atoms per cell. [Public dataset and notebook](https://zenodo.org/records/17466183)

| Phase and q (Å⁻¹) | Families | Shared width, original cut | Independent widths, original cut | Independent widths, exit transmission |
|---|---|---:|---:|---:|
| **MAPbI₃ joint** | **10 rings** | **10.94** | **8.20** | **33.07** |
| 0.9972 | 002, 110 | 25.39 | 16.86 | 40.15 |
| 1.4091 | 112, 200 | 4.07 | 2.88 | 13.26 |
| 1.7247 | 202 | 9.14 | 8.31 | 14.53 |
| 1.9902 | 004, 220 | 19.21 | 11.24 | 59.36 |
| 2.2255 | 114, 222, 310 | 14.05 | 11.11 | 123.25 |
| 2.4382 | 204, 312 | 3.64 | 2.62 | 8.94 |
| 2.8148 | 224, 400 | 9.59 | 8.93 | 20.88 |
| 2.9854 | 006, 314, 402, 330 | 8.98 | 7.56 | 8.44 |
| 3.4480 | 404 | 4.74 | 4.85 | 5.69 |
| 4.1040 | 118, 523, 406, 514, 442 | 2.75 | 2.82 | 2.95 |
| **PbI₂ 0.8989** | **001** | **2.02** | **1.43** | **2.76** |

Each entry is χ² divided by that fit’s effective degrees of freedom. The first column reproduces commit `bb45957`. The first two use 717 measured MAPbI₃ bins at αf ≥ 0.15° with the original shadow and weak-horizon exclusions. The last includes 738 measured bins and applies the pixel-averaged exit transmission. These fits use different observations, so the last column also measures disagreement with newly admitted bins.

**Which planes are parallel?** On the original cut, two (001) components give 11.63 ± 0.54° and 104.9 ± 18.1°, with 3.9 ± 0.6% and 89 ± 66% of the phase, respectively. On the full profile the best candidate changes to (212) at 30.00 ± 3.34° (56.8 ± 29.4%) and (001) at 58.46 ± 3.97° (43.2 ± 18.3%). The runner-up is (112) + (001), ΔBIC = 256. Nine individual rings exceed χ²red = 5. The large residuals and the change of preferred planes preclude a quantitative assignment of MAPbI₃ fibre planes, widths or oriented fraction.

For **PbI₂**, the only observed reflection is 001. Both selected populations have their basal planes parallel to the substrate. The narrow component is 16.64 ± 2.03° and 8.9 ± 2.5% of this phase. The broad component is 65.37 ± 7.07° and 91.1 ± 16.4%. The original cut gave 14.92° and 53.67°, with an isotropic fraction of 35.6%. The full-profile estimate of φ reaches zero with a local standard error of 17.6 percentage points. A single basal reflection supports this conditional shape comparison and supplies no independent check of the broad population.

**Why the strongest MAPbI₃ ring fails.** At χ = 3.5°, the folded signal is 52.97 ± 1.25 and the joint fit predicts 23.61. At 80.5° and 84.5° the Fresnel multipliers are 1.008 and 1.027, respectively, while the measured signal drops from 142.7 ± 8.7 to 0.6 ± 2.7. At 88.5° the measured net signal is negative after subtraction of the radial background. The optical maximum occurs much closer to the horizon than the onset of the loss.

The two detector sectors expose the asymmetry before polarization correction and background subtraction. At |χ| = 70–75°, mean intensities in the strongest ring window are 195 and 191 for the negative and positive sides. At 84–86° they are 82 and 10. Their respective radial maxima, 0.9948 and 0.9968 Å⁻¹, still lie inside the 0.9888–1.0057 Å⁻¹ window. Extending each edge by 0.015 Å⁻¹ and integrating above the measured flank background gives areas of 1.43 and 0.10 in intensity × Å⁻¹, compared with 3.44 and 3.31 at 70–75°. The asymmetry persists after the wider integration. The flank intensities themselves fall from about 18 near the missing wedge to 5 on the positive side at 84–86°, while the fitted constant background is 16.88. These observations identify a loss of measured scattering and a changing diffuse background. The frame alone does not identify the attenuating object.

At q = 2.2255 Å⁻¹, 60% of the residual χ² lies above 80°. At 3.4480 Å⁻¹ the detector ends before the horizon. Its worst bin is at 68.5°, where only 76% of the radial window is spanned. At 69.5° that falls to 22%.

The calibrated direct-beam height is 1826.95 px. In detector columns 543–547 (zero-based indices), the strongest on-axis excess between rows 1680 and 1839 occurs at row 1755, after subtracting the mean of the flanking strips at columns 510–527 and 560–577. If that feature were specular, its 71.45 px separation would imply αi ≈ ½ arctan(71.45 × 0.200 / 811.77) = 0.50°. The public notebook uses 0.075°. The related P08 chamber study reports 0.5° for its experiment. As a sensitivity test, `--part public --public-incidence-deg 0.5 --public-roll-deg 0` gives joint χ²red = 27.35 and strongest-ring χ²red = 37.82. This fit also fails. The measured-incidence section above compares this feature with an independent, less precise shadow-edge intercept. [P08 chamber study, §III](https://doi.org/10.1063/5.0149613)

The public example’s authors describe the perovskite as visually close to random and attribute additional low-q arcs to an intermediate phase. Those arcs are outside the ten fitted windows. [Public frame, Fig. 6](https://journals.iucr.org/j/issues/2026/01/00/yr5162/index.html)
