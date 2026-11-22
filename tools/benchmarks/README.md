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
five-cut exact Predict regression (72 s on the Windows test machine). In `per_cut/posterior_recovery.json.gz`
it checks known-composition fits 184 and 167 and composition choice 176. It also checks automatic level-2
fit 89 in `per_cut/posterior_newseed_auto2.json.gz` and known-composition fit 208 in `per_cut/posterior_newseed.json.gz`.

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
python tools/benchmarks/calibration_check.py --part all --cases 300 --three-ring-cases 150 --workers 4  # part 1 with pyFAI; --part coverage needs none
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

Run `calibration_check.py --part all --cases 300 --workers 4` with pyFAI installed.
The simulation alone (`--part coverage`) uses the GIMaP environment. `--part summarize` rebuilds the coverage
tables from `per_cut/calibration_check_frames.json.gz`. The JSON records the seed, settings and command.

**Geometry and uncertainty.** GIMaP fits the beam centre, sample-to-direct-beam distance L and two detector
rotations to the ring points. Each point is an intensity-weighted centroid from a sector about 2 px of arc wide.
The residual in pixels is r_i = L (tan 2θ_i − tan 2θ_ring) / p̄, where p̄ is the mean pixel size and 2θ_ring
comes from the order of AgBh (d₀₀₁ = 58.38 Å). The fit minimises Σ f²(√(1 + r_i²/f²) − 1) with f = 2 px.
The benchmark records the points used in the final fit to check the uncertainties independently.

For n contributing rings, the leave-one-ring-out variance is v_J = (n − 1)/n Σ_k (θ_k − θ̄)².
The full-fit covariance is C = [Σ_i r_i²/(N − p)] (JᵀJ)⁻¹ for N points and p free parameters.
With four or more rings, GIMaP reports √v_J for each free parameter. With exactly three rings and fitted
tilts, each two-ring refit holds both tilts at their full-fit values. GIMaP reports √max(v_J, C_jj) for x, y
and L, and √C_jj for the tilts. A calibration with fixed tilts uses the ring jackknife for x, y and L
with three or more rings. With two rings, the standard errors come from C.

**Public silver behenate frame.** GIMaP matched orders 1–5 on `saxs_agbh_pyfai/Pilatus1M.edf` at λ = 1.0 Å
and 172 µm pixels. pyFAI 2026.9 located points on six rings using `Recalibration.extract_cpt` (massif, one
point per degree, fixed seed) and repeatedly refined its geometry with `GeometryRefinement.curve_fit` and
rot3 = 0. The table uses pixel-corner coordinates and sample-to-direct-beam distance. The last column divides
the difference by the two standard errors added in quadrature.

| Parameter | GIMaP ± ring jackknife | pyFAI ± curve_fit [ring jackknife] | Difference / combined σ |
|---|---:|---:|---:|
| beam x (px) | 179.796 ± 0.186 | 179.923 ± 0.017 [0.073] | −0.68 |
| beam y (px) | 264.005 ± 0.174 | 263.968 ± 0.018 [0.094] | +0.22 |
| L (mm) | 1633.84 ± 0.66 | 1634.05 ± 0.07 [0.17] | −0.32 |
| rot1 (°) | −0.126 ± 0.154 | −0.297 ± 0.035 [0.089] | +1.08 |
| rot2 (°) | 0.717 ± 0.185 | 0.687 ± 0.035 [0.091] | +0.16 |

On GIMaP's ring points the RMS residuals of the GIMaP and pyFAI geometries were 0.584 and 0.623 px.
On pyFAI's points they were 0.300 and 0.222 px. The shipped `Pilatus1M.poni`, a starting geometry for
pyFAI, gave 14.6 and 15.3 px on these two sets. `calibration_check.json` also gives both geometries in PONI
coordinates and the results of a second pyFAI fit started 3% farther away with zero tilt.

**Recovery and coverage.** The 300 simulated Pilatus 1M frames have 3–9 target rings, L between
1.5 and 3 m, and rotations up to ±3° about either axis. Each fit receives an estimate within ±15% of the
true distance. Recovery requires a fitted tilt with an uncertainty, a beam centre within 5 px of the
truth and a distance error below 5%. Counts and coverage use the calibration window's first-ranked
candidate. The archive records the headless line check separately.

The shipped engine recovered **213/300** geometries. On the same seeds, the engine before
transmission polishing recovered 111/300 (archived in commit `7cd8999`). Coarse arc votes can place the
centre several pixels off, spreading a narrow ring over several radii in the profile. The engine now
tests refined copies of these votes alongside the original proposals. In the separate 150 frames with
only orders 1–3 drawn, it recovered 114/150 geometries. Of these, 112 matched exactly three rings.

Among recovered frames, each cell gives the fraction within 1σ, fraction within 2σ and median |error|/σ.
The three-ring row uses the separate set. The other rows use the 300-frame set.

| Matched rings | Frames | beam x | beam y | L | rot1 | rot2 |
|---|---:|---|---|---|---|---|
| 3 | 112 | 0.74 / 0.92 / 0.54 | 0.55 / 0.82 / 0.86 | 0.75 / 0.96 / 0.58 | 0.53 / 0.83 / 0.95 | 0.52 / 0.78 / 0.95 |
| 4–6 | 81 | 0.78 / 0.97 / 0.54 | 0.74 / 0.90 / 0.59 | 0.79 / 0.97 / 0.47 | 0.72 / 0.99 / 0.75 | 0.81 / 0.95 / 0.55 |
| ≥7 | 125 | 0.52 / 0.94 / 0.97 | 0.67 / 0.95 / 0.72 | 0.79 / 0.99 / 0.51 | 0.57 / 0.91 / 0.84 | 0.70 / 0.98 / 0.71 |

The main set also yielded 7 three-ring recoveries, reported separately in the JSON. With seven or more
rings, a Student t reference for ring scatter gives 65.1% within 1σ. The measured rates were 52.0% for beam x
and 56.8% for rot1. Taking the larger of the jackknife and covariance standard errors on the same
points gave 56.8% and 57.6%, respectively. The engine retains the ring jackknife for these fits.

## The third set

`posterior_newseed` was run at commit 6bb1a18 (fitting tree d139e6f5). The fitting code differs from the tree of the other runs (7e68831a) in its reporting. The five-cut regression above checks the archived numerical results from three runs. The seed (20261122) was chosen after the last change of the procedure, so no change was judged on it. The stage `auto` in this archive is the automatic analysis's call as it was then: the first level (q_z = 0), spheres and upright cylinders, one or two populations.
Among recovered frames, each cell gives the fraction within 1σ, fraction within 2σ and median |error|/σ.
The three-ring row uses the separate set. The other rows use the 300-frame set.

| Matched rings | Frames | beam x | beam y | L | rot1 | rot2 |
|---|---:|---|---|---|---|---|
| 3 | 112 | 0.74 / 0.92 / 0.54 | 0.55 / 0.82 / 0.86 | 0.75 / 0.96 / 0.58 | 0.53 / 0.83 / 0.95 | 0.52 / 0.78 / 0.95 |
| 4–6 | 81 | 0.78 / 0.97 / 0.54 | 0.74 / 0.90 / 0.59 | 0.79 / 0.97 / 0.47 | 0.72 / 0.99 / 0.75 | 0.81 / 0.95 / 0.55 |
| ≥7 | 125 | 0.52 / 0.94 / 0.97 | 0.67 / 0.95 / 0.72 | 0.79 / 0.99 / 0.51 | 0.57 / 0.91 / 0.84 | 0.70 / 0.98 / 0.71 |

The main set also yielded 7 three-ring recoveries, reported separately in the JSON. With seven or more
rings, a Student t reference for ring scatter gives 65.1% within 1σ. The measured rates were 52.0% for beam x
and 56.8% for rot1. Taking the larger of the jackknife and covariance standard errors on the same
points gave 56.8% and 57.6%, respectively. The engine retains the ring jackknife for these fits.

`posterior_newseed_auto2` repeats the stage `auto` on the same 300 cuts after the automatic analysis was changed to fit at the second level when it knows both angles (commit f73d9c8): the true composition is the most probable in 75 % of the cuts (27 % at the first level) and the median error of R is 0.8 % (13 %).
