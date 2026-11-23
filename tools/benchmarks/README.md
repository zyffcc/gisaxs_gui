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

## Running an independent validation

`posterior_recovery.py --set fresh` draws 300 cuts, 75 of each of the four first-set compositions.
The model-only and added-effects groups each contain 150 cuts. The SHA-256 of the UTF-8 set name supplies
eight little-endian words for the NumPy `SeedSequence` spawn key. The case index selects a child, whose
two children supply the parameter generator and a 128-bit seed for Poisson noise. A run stores this noise
seed in `case.seed`. The set name, root seed and index determine the same case in any worker order.
The named `fit1d` and `extended` sets also use separate streams. The extended fit receives the true LMA/DA coupling.

From the repository root, with `GIMAP_HOME` set to a temporary directory, a three-cut smoke run is:

```bash
B=tools/benchmarks
N=src/gimap/features/fitting/domain/posterior/data/flow_v1
python $B/check_case_overlap.py
python $B/posterior_recovery.py --network "$N" --set fresh --n 3 --workers 1 --auto --out smoke --output-dir "$GIMAP_HOME"
python $B/check_case_overlap.py --new-archive "$GIMAP_HOME/results/smoke.json"
python $B/summary_tables.py --per-cut "$GIMAP_HOME/results/smoke.json" --stage choose
python $B/posterior_pairs.py --by-cut "$GIMAP_HOME/paired.json" "fresh=$GIMAP_HOME/results/smoke.json"
```

Omit `--n` for the full named set. `--checkpoint DIR` resumes completed cuts with matching source,
network, generator and fit settings. An unsuccessful fit is recorded and gives a nonzero exit status.
`check_case_overlap.py` reads every shipped posterior gzip archive. For earlier posterior runs with
only a summary in `history/`, it reconstructs case descriptors from the recorded generator, seed and
count, including the 1,200-cut extended set. The audit labels these reconstructed draws.
It counts identical (composition, populations, noise seed) tuples, population records, first-population
numerical draws and noise seeds. It groups archives with identical indexed draws. Use `--indices` for
the matching cut pairs. Reuse involving the new set gives exit status 1. `--strict` also fails on
historical overlap. The archived third set shares 194 noise seeds with the first and 221 with the extended
set. Every one of its 300 noise seeds and first-population numerical draws occurs in at least one earlier set.

For replay, `--set legacy-first`, `legacy-extended` and `legacy-third` restore the original parameters,
noise and grid-dependent beam tail. The archived third set has 152 model-only and 148 added-effects cuts.
For a new run the central intensity after broadening is $A K(q)$, where $A$ is the integrated sharp
intensity and the discrete resolution density satisfies Σ K(qᵢ) Δq = 1.

**Composition and parameter errors.** The score uses the largest summed weight over retained modes of
each composition index, including population radius order and each population's paracrystal status.
Exact ties use the smaller index. Each successful fit, choose and auto record contains
`composition_weights`, `compared_by` (`evidence` or `bic`) and `retained_modes`. Its top-level
composition index, probability and fitted populations describe the leading individual solution.
`truth_composition_index` and the stage's `candidates` record whether the shortlist included the truth.
The summary gives attempted, failed, missing-weight, scored and correct cut counts. Its reliability
bins give correct and total cuts by branch, and each branch includes binary Brier and constant-forecast
scores. Historic runs lack the full weights. A parameter error in `fit.populations` uses the leading
solution's least-squares `values.best`. The separate `fit.posterior` fields hold posterior medians.

**Coverage bars for the validation figure.** Import `coverage_by_cut` from
`tools.benchmarks.coverage_bootstrap`. For example, call
`coverage_by_cut(run["results"], "model", "R")`. The point estimate counts covered populations
among eligible fits with matching population count. Each bootstrap sample draws whole contributing
cuts with replacement and recomputes the ratio of hits to eligible populations. The `bootstrap68`
and `bootstrap95` fields give 68% percentile bounds for the two nominal coverages. The default is
10,000 resamples with seed 0. Wilson intervals in composition reliability bins use one outcome per cut.

**Table 3, revision 14.** Run `python tools/benchmarks/table3_sources.py` to check the historical table.
The JSON report maps all 23 rows to their inputs, fields, producer information, reductions and counts.
Recovery rows read `per_cut/*.json.gz` directly. The verifier reproduces the composition definition
used for each printed historical row. For example, `--only first.high third.auto extended.composition`
checks three rows. Use `--bench DIR` for a different copy of the benchmark directory.
The all-pixel geometry, held-out calibration and measured Au rows require separate evidence. Supply
`--all-pixel-q FILE --heldout-calibration FILE --au-fit FILE --sem-reference FILE` and the
published `--table TABLE.tex --numbers MACROS.tex`. The SEM reference JSON supplies `citation`,
`diameter_mean_nm` and `diameter_sd_nm`. Each missing source or mismatched printed value gives a
nonzero exit status. Unstamped historical runs report an unknown producing commit.
`--strict-provenance` also fails when a computational run lacks that commit.

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

`python -m pytest -q tests/test_reproduce_automatic.py` runs the public P08 and GALAXI examples through automatic analysis, then replays their JSON exports from copies of the original inputs. The P08 reference uses the notebook incidence as input and applies the measured incidence with zero sample roll. It assigns 11 of 12 peaks. The MAPbI₃ orientation fails the residual check and the single-ring PbI₂ fit passes. The resolved MAPbI₃ reflections (202 and 404) give only one independent constraint on the tetragonal cell. The single PbI₂ peak also leaves its cell undetermined. A second P08 run checks replay with `--assume-fibre-normal` enabled.

The references in `tests/data/automatic_replay/` store the input hashes and the fields selected by `structural_results(report)` in the test. To refresh P08, use the runtime listed in `p08.json` and the test’s `tools.gimap_agent auto` command with one BLAS thread, CRLF copies of the calibration and CIFs, and the frame and phases in the spec. Take the input hashes from `report.json` under `reproduction.inputs` and the expected values from `structural_results` applied to that export. On the archive platform the regression checks the selected structural fields bit for bit. On other platforms it checks assignments and verdicts exactly, probabilities and BIC weights to 0.05 absolute, fitted parameters to 1% and interval endpoints to 2%. The replay command itself always reports exact agreement or differences.

## The runs the paper quotes

The recovery numbers of *Predict parameters* in the GIMaP paper, in Physics and Methods 9.13 and in the network's
`MODEL_CARD.md` come from the runs below. The original 200-cut and extended runs used git tree
`7e68831a52cf841a56fe05f8cc4e36ee62b41cb7` of `src/gimap/features/fitting/domain` (commits 2eb0243 and 7179b85 have
this tree), the network `flow_v1` (its SHA-256 is in every summary), ten workers on an Intel Core i9-13900HX; the
timing run alone used one worker. Each summary written at 7179b85 names the commit and the tree (`code`);
`posterior_recovery` and `posterior_extended` were written at 2eb0243, before summaries had that field.

The options in the table and the commands immediately below apply at the original producing revision.
To replay the inputs with the current scripts add `--set legacy-first` for first-set and own-model
runs, use `--set legacy-extended` for the extended runs, and `--set legacy-third` for the third set.
For its original automatic pass also select `--auto-level 1`.

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
| DWBA on known Si, specified and chosen compositions | `posterior_dwba4_si_summary.json` | `per_cut/posterior_dwba4_si.json.gz` | `--n 200 --workers 1 --scattering dwba --substrate si --cut dwba` |
| DWBA on known Si, automatic analysis | `posterior_dwba4_si_auto_summary.json` | `per_cut/posterior_dwba4_si_auto.json.gz` | `--n 200 --workers 1 --scattering dwba --substrate si --cut dwba --stages auto --auto-level 4` |
| DWBA on known 100 nm polystyrene on Si, specified and chosen compositions | `posterior_dwba4_ps100_summary.json` | `per_cut/posterior_dwba4_ps100.json.gz` | `--n 200 --workers 1 --scattering dwba --substrate ps100 --cut dwba` |
| DWBA on known 100 nm polystyrene on Si, automatic analysis | `posterior_dwba4_ps100_auto_summary.json` | `per_cut/posterior_dwba4_ps100_auto.json.gz` | `--n 200 --workers 1 --scattering dwba --substrate ps100 --cut dwba --stages auto --auto-level 4` |
| Same draws compared by largest q₊R and composition | `posterior_dwba4_pairs.json` | – | `posterior_pairs.py --by-cut`, below |
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
`si_eia_new`. In the historical summaries, *true composition most probable* (`composition.exact`) is judged on the most probable
solution: the first solution Predict lists has the true families; `true_in_equivalent_set` says that a listed
solution (probability at least 0.05) has them.

For spheres in the two-path DWBA comparison (`si_tp_new`), 68 % interval coverage was 60 %, 36 % and
48 % in the q₊R bins 1.2–1.7, 1.7–2.3 and 2.3–3.5. Predict therefore raises `qzR_cov` above 1.7 at
level 3 (`reporting.QZR_TWO_PATH_COVERAGE`). The frequent sphere/cylinder confusion in the DWBA comparison motivates a separate shape warning:
at levels 2 and 3 an upright cylinder above q₊R = 2 (`reporting.QZR_UPRIGHT_SPHERE`) gets `cyl_sphere`, including fits without a height.
The public GALAXI worked example gives q₊R = 3.10 for the 5.94 nm upright cylinder.

`history/` holds the summaries of earlier code and of removed methods; they are not the paper's numbers
(`history/README.md`). A new run writes `NAME_summary.json` here and `results/NAME.json` (not committed); files named
`*_new_summary.json` are left out of the repository.

## GALAXI automatic analysis and interpolation errors

From the source checkout, with one BLAS thread and a fresh `GIMAP_HOME`, run

```sh
python -m tools.gimap_agent auto tests/data/external/gisaxs_galaxi/galaxi_data.tif --calibration tests/data/external/gisaxs_galaxi/galaxi_bornagain.poni --incidence-deg 0.463 --out galaxi_auto
```

The comparison run used the same command at commit `7604ae3b` with a separate output folder.
The calibrated, symmetry-centred Yoneda cut contains 598 points on |qᵧ|. Across the 369 points
that average both arms, using squared interpolation weights reduces the marginal σ by a median
13.19% compared with interpolating σ linearly. The other 229 marginal errors and all q and I
values match the earlier run. The 367 adjacent pairs of averaged points have a median covariance
of 0.2044 in squared intensity units and a median correlation of 0.1635 (range 0.0959–0.2243).

`fit_curve.csv` gives the marginal `sigma` and the effective `fit_sigma` passed to Predict.
For independent input bins and folding matrix W, C = W diag(s²) Wᵀ. Using the row sums of
C for the diagonal fit variance increases σ by a median 15.19% over the corrected marginal
errors on the averaged points. The median ratio of fit σ to the old σ is 1.00010.
On an 80-point synthetic half-step grid with equal source variances, using marginal errors
alone understates the fitted constant's interval width by 13.3%. The row-sum approximation
agrees within 0.1% with the variance calculated using the full covariance.

| Level-2 fit (sphere with paracrystal + upright cylinder) | Linear interpolation of σ | Corrected fit weights |
|---|---:|---:|
| Best-fit R₁, R₂ (nm) | 3.2771, 5.9900 | 3.3392, 5.9413 |
| Best-fit spacing D₁ (nm) | 48.9128 | 49.6790 |
| Reduced χ² | 1.048 | 1.051 |
| Reported 68% interval on D₁ (nm) | [36.83, 59.97] | [27.05, 91.25] |

The earlier run assigned probability 0.9996 to this composition. Sampling in the corrected
run triggered the BIC fallback. The two reported modes of this composition have BIC weights
0.6540 and 0.3443 (composition weight 0.9983). The corrected run uses the least-squares
errors for the reported D₁ interval. Its diagnostic sampling summary gives a median of
48.21 nm and 16th and 84th percentiles of 36.62 and 60.92 nm.

The public replay regression also fits the positive arm alone. For this control the test masks
columns 0–499 of the same frame. The negative arm then reaches only 0.0436 Å⁻¹ and the positive
arm reaches 0.179 Å⁻¹. The procedure keeps the calibrated centre because the mask prevents a
symmetry refinement. Evidence sampling gives mode probabilities 0.4467, 0.4110 and 0.1423.
All three modes have the same composition, so its probability among the retained compositions is 1.

## Four-term DWBA on a known substrate

Each substrate uses the 200 original draws (seed 20261116), λ = 0.1045 nm, αᵢ = 0.40° and the Si Yoneda exit
angle. For ps100 the truth and the fit both place a 100 nm polystyrene film on Si. Changing the substrate changes
the simulated intensity and Poisson counts. The particle parameters, measurement settings and noise seeds remain
paired by cut index. The independent simulator uses the same optical constants as the fit. It includes surface
roughness, resolution and a beam tail on half of the cuts. The level-2 and level-3 DWBA comparisons both use Si.
The Born `choose` comparison uses these same particle draws. The Born `auto` reference uses the archived
300-cut third set, which overlaps the first and extended draws.

The `fit` stage gives Predict the true composition. The `choose` stage compares compositions with a paracrystal
in every population. The `auto` stage also considers populations without a paracrystal. All three use the same
proposal network and the substrate's fixed reflection coefficients throughout the likelihood and refinement.

In the historical tables below, composition recovery counts cuts whose leading solution has the correct
unordered multiset of particle families. The truth's mode requires those families and every fitted R
and D within 20 % of its input. A listed mode also counts an alternative of
probability at least 0.05. The parameter errors and coverages compare populations in order of increasing
radius whenever the fit returns the correct number of populations. Each q₊R bin uses the largest true R
of the cut, including upright cylinders. The shape-confusion counts use the *fitted* q₊R to evaluate the
warning a user would see. They compare a single input population with a single fitted population and
count errors in population number separately. Times are median elapsed seconds per cut for each stage.

With composition free, level 4 selected the correct families on 71.5 % of Si cuts and 75.0 % of ps100 cuts
(`choose`), compared with 33.0 % and 69.5 % for levels 2 and 3 on Si and 76.5 % for Born truth. Median radius
error fell from 7.99 % and 6.09 % to 1.26 % on Si, near the Born value of 0.89 %. On Si the 68/95 % radius
intervals covered 60/78 % of true radii, compared with 60/77 % for Born truth.

Automatic selection recovered the families on 69.5 % of Si cuts and 74.0 % of ps100 cuts. In the largest
true q₊R bins 1.7–2.3 and ≥2.3, recovery rose from 29/23 % at level 2 on Si to 75/68 % at level 4 on Si
and 71/74 % on ps100. For the 100 cuts containing only spheres (one or two populations), recovery rose from
9 % to 66 % on Si and reached 72 % on ps100. Of the 50 cuts with one upright cylinder, 78 % were correct at
level 2 on Si and 76 % at level 4 on each substrate. The network's candidate list included the true particle
families on all 200 cuts in each level-4 selection run. These runs used the existing network and proposal starts.

Among single-population cuts fitted as one upright cylinder at fitted q₊R > 2, spheres accounted for 2/27
(`choose`) and 2/25 (`auto`) on Si, and 0/24 and 0/23 on ps100. The Si results motivated retaining the
`cyl_sphere` advisory at level 4. The archives use the numerical code at `fda067c`. After adding the advisory,
`278df6af` reproduced all numerical fields of Si cut 100 exactly and flagged its upright-cylinder solution.
Median times for `fit`, `choose` and `auto` were 9.5, 57.7 and 38.0 s on Si and 8.9, 53.8 and 39.7 s on ps100.
These runs used one worker per substrate on the same machine. The historical runs used eight or ten concurrent
fits, so the timing differences also reflect CPU contention.

| Truth and fitted level | Stage | Cuts | Composition (%) | Truth mode first / listed (%) | Median error R / D (%) | R coverage 68 / 95 (%) | D coverage 68 / 95 (%) | Median s/cut |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Born truth, level 2 | fit | 200 | given | 89.0 / 93.0 | 0.61 / 0.07 | 67 / 85 | 85 / 95 | 15.2 |
| Si, level 2 | fit | 200 | given | 72.5 / 73.0 | 5.61 / 0.20 | 26 / 39 | 57 / 71 | 13.7 |
| Si, level 3 | fit | 200 | given | 87.0 / 96.5 | 2.96 / 0.06 | 57 / 73 | 88 / 99 | 28.6 |
| Si, level 4 | fit | 200 | given | 95.5 / 97.5 | 0.55 / 0.06 | 72 / 90 | 88 / 99 | 9.5 |
| ps100, level 4 | fit | 200 | given | 95.5 / 97.0 | 0.53 / 0.06 | 77 / 94 | 89 / 100 | 8.9 |
| Born truth, level 2 | choose | 200 | 76.5 | 72.0 / 85.5 | 0.89 / 0.06 | 60 / 77 | 87 / 96 | 69.4 |
| Si, level 2 | choose | 200 | 33.0 | 32.5 / 44.5 | 7.99 / 0.07 | 28 / 41 | 78 / 96 | 43.8 |
| Si, level 3 | choose | 200 | 69.5 | 59.5 / 88.0 | 6.09 / 0.06 | 56 / 67 | 89 / 100 | 93.6 |
| Si, level 4 | choose | 200 | 71.5 | 69.0 / 90.5 | 1.26 / 0.06 | 60 / 78 | 88 / 99 | 57.7 |
| ps100, level 4 | choose | 200 | 75.0 | 72.0 / 91.0 | 0.82 / 0.06 | 64 / 84 | 88 / 99 | 53.8 |
| Born truth, level 2 (300 cuts) | auto | 300 | 75.3 | 70.7 / 85.7 | 0.80 / 0.07 | 67 / 82 | 86 / 96 | 49.7 |
| Si, level 2 | auto | 200 | 30.5 | 30.0 / 44.0 | 8.18 / 0.07 | 28 / 40 | 78 / 96 | 55.2 |
| Si, level 4 | auto | 200 | 69.5 | 67.0 / 89.0 | 1.28 / 0.06 | 60 / 78 | 88 / 99 | 38.0 |
| ps100, level 4 | auto | 200 | 74.0 | 71.0 / 89.5 | 0.83 / 0.06 | 64 / 84 | 88 / 99 | 39.7 |

<details>
<summary>Composition, truth mode, errors, coverage and timing by cut class</summary>

**<1.2**

| Truth and fitted level | Stage | Cuts | Composition (%) | Truth mode first / listed (%) | Median error R / D (%) | R coverage 68 / 95 (%) | D coverage 68 / 95 (%) | Median s/cut |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Born truth, level 2 (300 cuts) | auto | 4 | 100.0 | 100.0 / 100.0 | 0.67 / 0.06 | 75 / 100 | 75 / 100 | 31.5 |

**1.2–1.7**

| Truth and fitted level | Stage | Cuts | Composition (%) | Truth mode first / listed (%) | Median error R / D (%) | R coverage 68 / 95 (%) | D coverage 68 / 95 (%) | Median s/cut |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Born truth, level 2 | fit | 25 | given | 100.0 / 100.0 | 0.41 / 0.04 | 68 / 84 | 92 / 100 | 9.8 |
| Si, level 2 | fit | 25 | given | 100.0 / 100.0 | 1.28 / 0.03 | 40 / 60 | 92 / 100 | 5.6 |
| Si, level 3 | fit | 25 | given | 92.0 / 96.0 | 1.05 / 0.03 | 52 / 72 | 92 / 100 | 6.9 |
| Si, level 4 | fit | 25 | given | 100.0 / 100.0 | 0.37 / 0.06 | 72 / 96 | 92 / 100 | 4.4 |
| ps100, level 4 | fit | 25 | given | 100.0 / 100.0 | 0.28 / 0.05 | 76 / 96 | 92 / 100 | 4.3 |
| Born truth, level 2 | choose | 25 | 76.0 | 76.0 / 84.0 | 0.48 / 0.04 | 59 / 82 | 86 / 100 | 40.1 |
| Si, level 2 | choose | 25 | 84.0 | 84.0 / 88.0 | 1.04 / 0.05 | 36 / 68 | 86 / 100 | 23.9 |
| Si, level 3 | choose | 25 | 80.0 | 76.0 / 84.0 | 1.05 / 0.03 | 65 / 78 | 87 / 100 | 49.3 |
| Si, level 4 | choose | 25 | 80.0 | 80.0 / 88.0 | 0.34 / 0.05 | 64 / 91 | 91 / 100 | 54.6 |
| ps100, level 4 | choose | 25 | 80.0 | 80.0 / 88.0 | 0.28 / 0.05 | 68 / 91 | 91 / 100 | 44.9 |
| Born truth, level 2 (300 cuts) | auto | 35 | 65.7 | 65.7 / 77.1 | 0.20 / 0.06 | 78 / 81 | 89 / 96 | 27.6 |
| Si, level 2 | auto | 25 | 80.0 | 80.0 / 88.0 | 1.04 / 0.07 | 38 / 71 | 86 / 100 | 28.3 |
| Si, level 4 | auto | 25 | 76.0 | 76.0 / 88.0 | 0.34 / 0.05 | 65 / 95 | 95 / 100 | 39.8 |
| ps100, level 4 | auto | 25 | 76.0 | 76.0 / 84.0 | 0.28 / 0.05 | 70 / 95 | 95 / 100 | 34.7 |

**1.7–2.3**

| Truth and fitted level | Stage | Cuts | Composition (%) | Truth mode first / listed (%) | Median error R / D (%) | R coverage 68 / 95 (%) | D coverage 68 / 95 (%) | Median s/cut |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Born truth, level 2 | fit | 24 | given | 91.7 / 91.7 | 0.26 / 0.05 | 77 / 96 | 92 / 100 | 6.9 |
| Si, level 2 | fit | 24 | given | 100.0 / 100.0 | 0.85 / 0.09 | 54 / 65 | 77 / 88 | 5.6 |
| Si, level 3 | fit | 24 | given | 87.5 / 100.0 | 2.63 / 0.05 | 54 / 62 | 88 / 100 | 16.5 |
| Si, level 4 | fit | 24 | given | 100.0 / 100.0 | 0.36 / 0.04 | 65 / 96 | 88 / 100 | 4.3 |
| ps100, level 4 | fit | 24 | given | 91.7 / 91.7 | 0.34 / 0.06 | 81 / 92 | 85 / 100 | 4.0 |
| Born truth, level 2 | choose | 24 | 87.5 | 87.5 / 91.7 | 0.30 / 0.03 | 68 / 88 | 84 / 100 | 33.9 |
| Si, level 2 | choose | 24 | 33.3 | 33.3 / 41.7 | 9.60 / 0.03 | 33 / 46 | 79 / 100 | 25.9 |
| Si, level 3 | choose | 24 | 66.7 | 58.3 / 95.8 | 8.50 / 0.04 | 52 / 52 | 80 / 100 | 46.4 |
| Si, level 4 | choose | 24 | 75.0 | 75.0 / 91.7 | 0.42 / 0.04 | 54 / 79 | 92 / 100 | 42.6 |
| ps100, level 4 | choose | 24 | 70.8 | 70.8 / 87.5 | 0.50 / 0.03 | 67 / 79 | 88 / 100 | 52.8 |
| Born truth, level 2 (300 cuts) | auto | 38 | 86.8 | 86.8 / 89.5 | 0.39 / 0.05 | 70 / 95 | 89 / 100 | 23.9 |
| Si, level 2 | auto | 24 | 29.2 | 29.2 / 37.5 | 9.64 / 0.03 | 30 / 43 | 83 / 100 | 35.0 |
| Si, level 4 | auto | 24 | 75.0 | 75.0 / 87.5 | 0.42 / 0.04 | 54 / 79 | 92 / 100 | 31.3 |
| ps100, level 4 | auto | 24 | 70.8 | 70.8 / 83.3 | 0.50 / 0.03 | 67 / 79 | 88 / 100 | 34.9 |

**≥2.3**

| Truth and fitted level | Stage | Cuts | Composition (%) | Truth mode first / listed (%) | Median error R / D (%) | R coverage 68 / 95 (%) | D coverage 68 / 95 (%) | Median s/cut |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Born truth, level 2 | fit | 151 | given | 86.8 / 92.1 | 0.76 / 0.07 | 66 / 84 | 83 / 94 | 23.7 |
| Si, level 2 | fit | 151 | given | 63.6 / 64.2 | 7.43 / 0.27 | 22 / 34 | 51 / 67 | 14.9 |
| Si, level 3 | fit | 151 | given | 86.1 / 96.0 | 3.31 / 0.06 | 58 / 74 | 88 / 99 | 34.6 |
| Si, level 4 | fit | 151 | given | 94.0 / 96.7 | 0.66 / 0.06 | 73 / 89 | 88 / 99 | 12.7 |
| ps100, level 4 | fit | 151 | given | 95.4 / 97.4 | 0.67 / 0.06 | 76 / 94 | 89 / 100 | 10.6 |
| Born truth, level 2 | choose | 151 | 74.8 | 68.9 / 84.8 | 1.00 / 0.07 | 60 / 76 | 87 / 95 | 78.1 |
| Si, level 2 | choose | 151 | 24.5 | 23.8 / 37.7 | 8.93 / 0.07 | 27 / 37 | 77 / 95 | 47.9 |
| Si, level 3 | choose | 151 | 68.2 | 57.0 / 87.4 | 6.30 / 0.06 | 55 / 67 | 90 / 100 | 109.8 |
| Si, level 4 | choose | 151 | 69.5 | 66.2 / 90.7 | 1.46 / 0.07 | 60 / 77 | 88 / 98 | 61.9 |
| ps100, level 4 | choose | 151 | 74.8 | 70.9 / 92.1 | 0.95 / 0.06 | 64 / 83 | 88 / 99 | 56.1 |
| Born truth, level 2 (300 cuts) | auto | 223 | 74.4 | 68.2 / 86.1 | 1.07 / 0.07 | 66 / 80 | 86 / 96 | 53.0 |
| Si, level 2 | auto | 151 | 22.5 | 21.9 / 37.7 | 9.23 / 0.07 | 27 / 37 | 77 / 95 | 58.7 |
| Si, level 4 | auto | 151 | 67.5 | 64.2 / 89.4 | 1.51 / 0.06 | 60 / 77 | 87 / 98 | 40.5 |
| ps100, level 4 | auto | 151 | 74.2 | 70.2 / 91.4 | 0.96 / 0.06 | 64 / 83 | 88 / 99 | 43.0 |

**Spheres only (one or two)**

| Truth and fitted level | Stage | Cuts | Composition (%) | Truth mode first / listed (%) | Median error R / D (%) | R coverage 68 / 95 (%) | D coverage 68 / 95 (%) | Median s/cut |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Born truth, level 2 | fit | 100 | given | 83.0 / 88.0 | 0.70 / 0.06 | 65 / 83 | 85 / 93 | 16.6 |
| Si, level 2 | fit | 100 | given | 63.0 / 64.0 | 7.58 / 0.33 | 13 / 20 | 48 / 64 | 14.3 |
| Si, level 3 | fit | 100 | given | 78.0 / 96.0 | 5.17 / 0.05 | 53 / 65 | 90 / 99 | 35.0 |
| Si, level 4 | fit | 100 | given | 92.0 / 95.0 | 0.69 / 0.06 | 67 / 87 | 90 / 99 | 7.6 |
| ps100, level 4 | fit | 100 | given | 92.0 / 94.0 | 0.68 / 0.06 | 77 / 92 | 89 / 100 | 8.6 |
| Born truth, level 2 | choose | 100 | 70.0 | 64.0 / 82.0 | 1.04 / 0.05 | 55 / 72 | 87 / 93 | 67.2 |
| Si, level 2 | choose | 100 | 10.0 | 10.0 / 19.0 | 10.81 / 0.05 | 12 / 21 | 81 / 96 | 42.7 |
| Si, level 3 | choose | 100 | 66.0 | 53.0 / 85.0 | 7.92 / 0.05 | 50 / 59 | 91 / 100 | 96.1 |
| Si, level 4 | choose | 100 | 67.0 | 62.0 / 87.0 | 1.72 / 0.06 | 59 / 73 | 91 / 99 | 55.1 |
| ps100, level 4 | choose | 100 | 72.0 | 67.0 / 88.0 | 0.88 / 0.06 | 62 / 79 | 90 / 99 | 52.8 |
| Born truth, level 2 (300 cuts) | auto | 150 | 71.3 | 66.0 / 83.3 | 1.03 / 0.07 | 65 / 82 | 86 / 97 | 51.7 |
| Si, level 2 | auto | 100 | 9.0 | 9.0 / 19.0 | 10.81 / 0.05 | 13 / 21 | 81 / 96 | 54.0 |
| Si, level 4 | auto | 100 | 66.0 | 61.0 / 86.0 | 1.72 / 0.06 | 59 / 74 | 91 / 99 | 39.2 |
| ps100, level 4 | auto | 100 | 72.0 | 67.0 / 87.0 | 0.87 / 0.06 | 63 / 80 | 90 / 99 | 39.2 |

**Upright cylinders only**

| Truth and fitted level | Stage | Cuts | Composition (%) | Truth mode first / listed (%) | Median error R / D (%) | R coverage 68 / 95 (%) | D coverage 68 / 95 (%) | Median s/cut |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Born truth, level 2 | fit | 50 | given | 100.0 / 100.0 | 0.26 / 0.06 | 74 / 94 | 84 / 100 | 8.5 |
| Si, level 2 | fit | 50 | given | 100.0 / 100.0 | 0.26 / 0.06 | 74 / 94 | 84 / 100 | 5.6 |
| Si, level 3 | fit | 50 | given | 100.0 / 100.0 | 0.26 / 0.06 | 74 / 94 | 84 / 100 | 5.6 |
| Si, level 4 | fit | 50 | given | 100.0 / 100.0 | 0.26 / 0.06 | 74 / 100 | 84 / 100 | 5.6 |
| ps100, level 4 | fit | 50 | given | 100.0 / 100.0 | 0.26 / 0.06 | 74 / 100 | 84 / 100 | 5.3 |
| Born truth, level 2 | choose | 50 | 86.0 | 86.0 / 86.0 | 0.23 / 0.06 | 70 / 95 | 77 / 100 | 43.8 |
| Si, level 2 | choose | 50 | 86.0 | 86.0 / 86.0 | 0.23 / 0.06 | 70 / 95 | 77 / 100 | 26.2 |
| Si, level 3 | choose | 50 | 80.0 | 80.0 / 86.0 | 0.23 / 0.06 | 79 / 95 | 76 / 100 | 45.2 |
| Si, level 4 | choose | 50 | 82.0 | 82.0 / 88.0 | 0.23 / 0.06 | 71 / 100 | 80 / 100 | 52.5 |
| ps100, level 4 | choose | 50 | 80.0 | 80.0 / 88.0 | 0.25 / 0.06 | 70 / 100 | 80 / 100 | 56.8 |
| Born truth, level 2 (300 cuts) | auto | 75 | 84.0 | 84.0 / 85.3 | 0.23 / 0.06 | 89 / 100 | 87 / 100 | 42.8 |
| Si, level 2 | auto | 50 | 78.0 | 78.0 / 84.0 | 0.23 / 0.06 | 72 / 95 | 77 / 100 | 39.4 |
| Si, level 4 | auto | 50 | 76.0 | 76.0 / 84.0 | 0.25 / 0.05 | 68 / 100 | 82 / 100 | 36.9 |
| ps100, level 4 | auto | 50 | 76.0 | 76.0 / 84.0 | 0.25 / 0.05 | 68 / 100 | 82 / 100 | 41.4 |

**Spheres and upright cylinders**

| Truth and fitted level | Stage | Cuts | Composition (%) | Truth mode first / listed (%) | Median error R / D (%) | R coverage 68 / 95 (%) | D coverage 68 / 95 (%) | Median s/cut |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Born truth, level 2 | fit | 50 | given | 90.0 / 96.0 | 1.03 / 0.08 | 68 / 83 | 84 / 94 | 27.9 |
| Si, level 2 | fit | 50 | given | 64.0 / 64.0 | 7.08 / 0.20 | 22 / 39 | 57 / 68 | 17.0 |
| Si, level 3 | fit | 50 | given | 92.0 / 94.0 | 3.27 / 0.08 | 55 / 74 | 88 / 99 | 37.9 |
| Si, level 4 | fit | 50 | given | 98.0 / 100.0 | 0.80 / 0.06 | 78 / 91 | 87 / 99 | 17.5 |
| ps100, level 4 | fit | 50 | given | 98.0 / 100.0 | 0.83 / 0.06 | 78 / 93 | 92 / 99 | 14.6 |
| Born truth, level 2 | choose | 50 | 80.0 | 74.0 / 92.0 | 1.30 / 0.08 | 63 / 77 | 91 / 98 | 88.7 |
| Si, level 2 | choose | 50 | 26.0 | 24.0 / 54.0 | 5.97 / 0.09 | 32 / 44 | 75 / 93 | 52.8 |
| Si, level 3 | choose | 50 | 66.0 | 52.0 / 96.0 | 6.67 / 0.08 | 55 / 67 | 91 / 99 | 121.7 |
| Si, level 4 | choose | 50 | 70.0 | 70.0 / 100.0 | 1.58 / 0.06 | 58 / 77 | 87 / 98 | 65.5 |
| ps100, level 4 | choose | 50 | 76.0 | 74.0 / 100.0 | 1.30 / 0.06 | 65 / 83 | 89 / 99 | 50.5 |
| Born truth, level 2 (300 cuts) | auto | 75 | 74.7 | 66.7 / 90.7 | 1.25 / 0.06 | 61 / 75 | 86 / 95 | 54.6 |
| Si, level 2 | auto | 50 | 26.0 | 24.0 / 54.0 | 5.97 / 0.09 | 32 / 44 | 75 / 93 | 65.7 |
| Si, level 4 | auto | 50 | 70.0 | 70.0 / 100.0 | 1.58 / 0.06 | 58 / 77 | 87 / 98 | 38.1 |
| ps100, level 4 | auto | 50 | 76.0 | 74.0 / 100.0 | 1.30 / 0.06 | 65 / 83 | 89 / 99 | 40.0 |

</details>

To repeat the runs, set `GIMAP_HOME` to a temporary directory and run these commands from the repository root.
Each substrate runs the plain and automatic commands in sequence on one worker, with up to two workers on the machine.
The paired output goes in `results/` beside the new per-cut files.

```bash
N=src/gimap/features/fitting/domain/posterior/data/flow_v1
B=tools/benchmarks
P=$B/per_cut
R=$B/results
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
for substrate in si ps100; do
  (
    name=posterior_dwba4_$substrate
    python $B/posterior_recovery.py --network $N --set legacy-first --out "${name}_repeat" --n 200 --workers 1 --scattering dwba --substrate "$substrate" --cut dwba &&
    python $B/posterior_recovery.py --network $N --set legacy-first --out "${name}_auto_repeat" --n 200 --workers 1 --scattering dwba --substrate "$substrate" --cut dwba --stages auto --auto-level 4
  ) &
done
wait
python $B/posterior_pairs.py --by-cut $R/posterior_dwba4_pairs.json \
  born=$P/posterior_recovery.json.gz si_l2=$P/posterior_dwba_born.json.gz si_l3=$P/posterior_dwba_tp.json.gz \
  si_auto2=$P/posterior_dwba_auto2.json.gz si_l4=$R/posterior_dwba4_si_repeat.json \
  ps100_l4=$R/posterior_dwba4_ps100_repeat.json si_auto4=$R/posterior_dwba4_si_auto_repeat.json \
  ps100_auto4=$R/posterior_dwba4_ps100_auto_repeat.json
python $B/posterior_pairs.py --by-cut $R/posterior_dwba4_born_auto_reference.json born_auto2=$P/posterior_newseed_auto2.json.gz
```

The pairing command above uses the fresh `results/<name>.json` files and recomputes timings for that machine.
To read the archived level-4 parameter and timing results, use the four bundled archives as inputs:
`si_l4=$P/posterior_dwba4_si.json.gz`, `ps100_l4=$P/posterior_dwba4_ps100.json.gz`,
`si_auto4=$P/posterior_dwba4_si_auto.json.gz` and `ps100_auto4=$P/posterior_dwba4_ps100_auto.json.gz`.

## The scripts

| Script | What it measures | Summary |
|---|---|---|
| `recovery_cases.py` | the held-out curves of the 1D recovery benchmarks: fit1d's 200 (seed 20261116), their draw and their curve from the independent forward model `yoneda_truth.py`; shared by the scripts below | – |
| `fit1d_recovery.py` | how well *Predict parameters (fit1d)* (removed in 0.2.0; a copy of the kit in `GISAXS1D_KIT_HOME`) recovered known parameters on the 200 curves of `recovery_cases.py` (Physics and Methods, section 9.12); with `--set extended` on 1200 curves of `heldout_cases.py` | `history/fit1d_recovery_summary.json`, `history/fit1d_heldout_summary.json` (a new run writes beside the script) |
| `posterior_recovery.py` | how well *Predict parameters* (the proposal network, its posterior and refinement) recovers known parameters on fit1d's 200 held-out curves (`--set legacy-first`) or on curves of `heldout_cases.py` (`--set legacy-extended`; 600 in the paper's run): errors, coverage of the 68 %/95 % intervals, how often the most probable solution is in the truth's mode, the composition chosen, time per curve on one core (Physics and Methods, section 9.13); `--truth own` draws the same cases in Predict's own model (the coverage when the model is exact), `--model-error-floor`, `--explored`, `--moves`, `--moved-solutions`, `--per-count` and `--widths` set those steps, `--constant module.NAME=value` any other constant of `refine` or `infer`; `--scattering dwba` draws the same curves with the four DWBA terms (`--substrate si` or `ps100`, 100 nm polystyrene on Si); `--cut` says what Predict is given of the cut: `born` its q₊ = k_i + k_f (the Born term at it; the default), `two_path` both angles (spheres with the exit-reflected path at q₋ = k_i − k_f and a free effective reflection coefficient), `dwba` both angles and the substrate optics (four fixed paths), `eia` nothing (the effective interface). `--stages auto --auto-level 4` runs automatic analysis with these known optics | `posterior_*_summary.json` and `per_cut/` (*The runs the paper quotes*, above); earlier runs in `history/` |
| `posterior_pairs.py` | pairs runs of `posterior_recovery.py` on the same cases that differ in the truth's physics or Predict's geometry: errors of R and D by q₊R for spheres and upright cylinders, composition and truth's mode by the largest sphere's q₊R, whether χ²/N marks the spheres that are off (Physics and Methods 9.13, *The waves the surface reflects*). `--by-cut` compares the largest true q₊R, true composition, parameters, intervals, timing and single-population shape substitutions, checking the draws match | `posterior_dwba4_pairs.json`, `posterior_dwba_pairs.json`; earlier ones in `history/` |
| `evidence_integration_check.py` | Predict's evidence, the scales integrated by Laplace at their least-squares values after the model error's spline coefficients (8 B-splines, a partition of unity, so a constant in them trades against every scale), against the scales and the coefficients integrated together (adaptive importance sampling, each value checked by a second run), on six one-population cuts of the first benchmark set (both halves) with four compositions each (the truth, without its paracrystal, the other family, two populations): the difference at the refined solution, at τ/2 and 2τ and at 40 draws of each solution's posterior, and what it changes in each solution's evidence and in the log Bayes factors against the truth (statistics referee, round 5, S7 ii). The joint integral is larger by 0.3–1.8 for the true composition (it grows with τ, spread 0.3–1.0 within a posterior); the log Bayes factors of the compositions that describe the curve change by −0.9 to +1.7 (+5.6 for one at χ²/N 1.5), those of compositions far off by +24 to +1350 in their favour, and no ranking changes | `evidence_integration_check.json` |
| `summary_tables.py` | prints the tables of Physics and Methods 9.13 and 10.3 and of the model card from the stored summaries (`--lang zh` for the Chinese headers, `--only recovery extended sensitivity own dwba pairs timing v5 large card`); computes nothing again | – |
| `search_baseline.py` | the baseline without a network: Fitting's *Search the ranges, then refine* from its default values on the same 200 held-out curves, the true composition given (median error of R 2.8 % / 6.7 %, 90th percentile 81 % / 136 %, about 10 s per curve; Predict parameters 0.5 % / 0.7 % and 10 % / 14 %, `posterior_recovery_summary.json`) | `search_baseline_summary.json` |
| `heldout_cases.py` | one to three populations, with LMA or DA. The historical extended set shared draws with the first and third sets. | – |
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
The simulations use the GIMaP environment. Run the development and held-out sets with the commands below.
Each summary gives the code revision, command and link to its compressed per-frame archive.
Use `--part summarize --json <summary>` to rebuild its tables from the archived frames.

```
python tools/benchmarks/calibration_check.py --part coverage --cases 300 --workers 3
python tools/benchmarks/calibration_check.py --part coverage --cases 120 --seed 20270000 --workers 3 --json tools/benchmarks/calibration_heldout.json
```

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
truth and a distance error below 5%. The benchmark scores the geometry that the independent
line-position check returns. Each frame records the selected rank, beam x and y, L, rot1, rot2,
their five standard errors and the orders used for that fit. Additional orders can reach a
corner of the detector, so the observed count can exceed the simulation target.

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

In the development comparison at `7604ae3b`, the engine found **300/300 geometries** on the same seeds for which commit `7fc06ff` found
213/300. It retained all 213 successful fits, recovered all 53 frames previously missing a tilt fit
and all 34 previously assigned a wrong geometry. Among those 34, the old fit missed both tolerances
on 25 frames, only the distance tolerance on six, and only the centre tolerance on three.

| Simulation group | Earlier engine | Engine at `7604ae3b` |
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
baseline used four workers and the development comparison used three.

**Returned geometry.** The line check selected another candidate in 102 of the 300 development frames.
All 300 returned geometries met the recovery criteria. The 95th percentile of
√[(rot1_fit − rot1_true)² + (rot2_fit − rot2_true)²] was **0.03594°**. The largest true combined
tilt was 4.109°. In 75 frames the returned fit included 10–12 orders.

Each cell gives the measured fraction within one standard error, the fraction within two standard
errors and median |error|/σ. The denominator includes every returned joint fit with five errors.
The first row uses the independent 150-frame control with only orders 1–3 drawn. The remaining
rows use the development set, grouped by the number of rings in the returned fit.

| Matched rings | Frames | beam x | beam y | L | rot1 | rot2 |
|---|---:|---|---|---|---|---|
| 3 (orders 1–3 only) | 150 | 0.620 / 0.820 / 0.722 | 0.687 / 0.853 / 0.590 | 0.620 / 0.860 / 0.733 | 0.633 / 0.867 / 0.742 | 0.680 / 0.847 / 0.587 |
| 3 | 6 | 0.500 / 0.833 / 1.008 | 0.667 / 1.000 / 0.350 | 0.833 / 0.833 / 0.475 | 0.333 / 0.833 / 1.405 | 0.500 / 0.500 / 1.496 |
| 4–6 | 138 | 0.616 / 0.870 / 0.672 | 0.616 / 0.855 / 0.705 | 0.565 / 0.819 / 0.745 | 0.630 / 0.877 / 0.731 | 0.551 / 0.826 / 0.904 |
| ≥7 | 156 | 0.660 / 0.955 / 0.681 | 0.686 / 0.904 / 0.614 | 0.615 / 0.891 / 0.793 | 0.635 / 0.929 / 0.710 | 0.609 / 0.885 / 0.822 |

With few rings, estimated standard errors themselves vary. A Student t reference with n − 1 degrees
of freedom gives 57.7%, 61.9% and 65.4% within 1σ for the separate three-ring set, the 4–6 group and
the ≥7 group respectively. In the 4–6 group the distance lay within one standard error in
78/138 cases (56.52%), and rot2 in 76/138 (55.07%). The selected fits used the ring jackknife.
Repeating it on their own points reproduced the stored standard errors to within 8.9 × 10⁻⁶
relative across the 450 development and control frames.

The held-out run recovered all 120 geometries. The line check switched 38 candidates and
the returned tilt-vector error had a 95th percentile of **0.02760°**. With 4–6 matched rings
(59 frames), the five empirical one-standard-error coverages ranged from 57.6% to 62.7%.
With ≥7 rings (58 frames) they ranged from 53.4% to 67.2%. Three held-out frames matched
three rings. A further 150 controls with only three orders drawn all returned three-ring
solutions. Every three-ring control in both runs admitted a full five-parameter jackknife.
The [held-out summary](calibration_heldout.json) and [per-frame archive](per_cut/calibration_heldout_frames.json.gz)
give the seed, fitted geometry, uncertainties and coverage counts.

## The third set

`posterior_newseed` was run at commit 6bb1a18 (fitting tree d139e6f5). The fitting code differs from the tree of the other runs (7e68831a) in its reporting. The five-cut regression above checks the archived numerical results from three runs. The seed (20261122) was chosen after the last change of the procedure, so no change was judged on it. The stage `auto` in this archive is the automatic analysis's call as it was then: the first level (q_z = 0), spheres and upright cylinders, one or two populations.
`posterior_newseed_auto2` repeats the stage `auto` on the same 300 cuts after automatic analysis began fitting at the second level when it knows both angles (commit f73d9c8). The leading solution recovers the unordered particle families in 75 % of the cuts (27 % at the first level). The median error of R is 0.8 % (13 %) among cuts with matching population count.

`posterior_dwba_auto2` (commit 9c90e6a): with DWBA truth the automatic analysis's call chooses the true composition in 31 % of the cuts; by the largest q_zR of a cut: 80 % (median |ΔR/R| 1 %) at 1.2–1.7, 29 % (10 %) at 1.7–2.3 and 23 % (9 %) above 2.3; cuts of spheres alone 9 %, of cylinders alone 78 %. The q_zR flags of Predict mark the solutions above these limits.

## GIWAXS unit-cell refinement

Run `python tools/benchmarks/cell_refinement.py --part all --trials 100 --json tools/benchmarks/cell_refinement.json`. The [archive](cell_refinement.json) contains synthetic truth, public assignments and residuals, the refraction and assignment sensitivity, and the time-dependent cells and uncertainties. The public runs use the supplied PONIs, the Xeuss mask and a temporary settings directory. To reproduce the [coverage archive](cell_coverage.json), run `python tools/benchmarks/cell_coverage.py --trials 100 --json tools/benchmarks/cell_coverage.json`. The numerical method is in [Physics and Methods §7.12](../../docs/Physics_Manual.md#712-unit-cell-refinement-from-indexed-peaks).

**Noisy peak lists.** Each trial draws independent Gaussian peak-centre errors between 0.0002 and 0.0008 Å⁻¹ about a known strained cell with an additive shift of 0.0015 Å⁻¹. The 100-draw runs cover all seven crystal systems, including trigonal CIFs in hexagonal and rhombohedral settings. Across the 38 reported parameters, volumes and offsets, the empirical 1σ coverage averaged 72.8%, with individual values from 62% to 81%. Covariance scaling above unit reduced χ² makes the intervals slightly conservative. The JSON lists the truth, mean recovered value, coverage and mean reported error for each parameter.

**Unlabelled overlapping positions.** A separate 100-draw tetragonal simulation generates 19 noisy peak positions, including nine unresolved centroids with random component weights. The fit receives positions, errors and widths. Across the 100 draws, all 900 multiplets were excluded and all 1,000 singleton centres received their generating indices. One-standard-error coverage was 64/100 for a, 76/100 for c and 73/100 for the offset.

**Public frames.** The P08 reduction used an incidence of 0.50125° and zero roll. Both kinematic q and the reduction at the CIF film critical angle of 0.11971° retained only 202 and 404 for MAPbI₃. These two harmonics give rank one for two metric parameters. The program reports that a, c and their separate strains are undetermined. The single PbI₂ 001 ring also leaves its cell undetermined.

| Sample and indexing | Cell estimate ± 1σ | Relative to CIF | q offset (Å⁻¹) | χ²red |
|---|---|---|---|---:|
| Xeuss PbTe, conditional 111/200/220 indexing | a = 6.832 ± 0.021 Å | +5.72 ± 0.33% against 6.462 Å | −0.0203 ± 0.0080 | 0.60 |

With the Xeuss geometry, the three measured centres (1.56733, 1.81998 and 2.58111 Å⁻¹) fall outside the bulk-CIF assignment tolerances. The benchmark records that refusal. It then tests the rock-salt sequence documented with the data, using the narrowest fractional-width order to initialize a trial cell and retaining bulk PbTe as the strain reference. The apparent expansion depends on that indexing and on the q calibration. [Public frames and geometry](../../tests/data/external/README.md)

**In situ series.** Eight synthetic Poisson detector frames contain six tetragonal rings with known trajectories a(t) = 10 + 0.009t and c(t) = 16 − 0.004t Å over 0–14 s. Checking the tracked centres against the complete catalogue at each frame's fitted metric gives RMS errors of 0.00347 Å for a, 0.00430 Å for c and 1.55 Å³ for V. The archive lists each frame's retained and excluded centres. A GUI regression also indexes the stack, inserts a missing frame and checks the gap in the exported table.

## Crystal orientation

Run `python tools/benchmarks/crystal_orientation.py --part all --trials 32 --json results.json`. The JSON contains each simulated fit and its 1σ errors, the public-frame report, the film-absorption refit, an independent-width refit on the original angular cut, and signed-sector measurements around the strongest peak.

### Synthetic recovery

The per-draw archive can be reproduced with `python tools/benchmarks/crystal_orientation.py --part synthetic --trials 32 --json tools/benchmarks/orientation_coverage.json`. Each seed has the preferred axes, pooled fractions, standard errors, BIC separation and verdict. [Orientation coverage archive](orientation_coverage.json)

The detector-image controls in `tests/test_film_pole_frames.py` project a near-horizon arc onto the detector from its known internal pole angle. The vacuum maximum and the recovered internal maximum differ by at least one 1° bin. Four further frames include finite-depth (700 nm) attenuation, Poisson noise and a fitted radial background. Two have random orientation and two have 0.1% basal texture. None yields a determined axis.

Three rings with independent scales sample each simulated fibre distribution. Direct integration of the antipodal Gaussian uses 96 azimuthal and 16 within-bin Gauss nodes. Individual poles contribute their calculated |F|². Two independent Poisson draws at ±χ are folded with their counting errors after subtraction of the known background. The missing wedge below 5° and a detector gap at 42–45° are present in every draw. The two single-component sets use αf ≥ 0.30°. The mixed set includes the exit field for δ = 4.2 × 10⁻⁶, β = 3.1 × 10⁻⁷ at λ = 1 Å and αi = 0.15°, with all bin centres above the horizon accepted. Its true (110) populations have widths 8° and 36° and absolute fractions 0.186 and 0.434.

Seeds 41027–41058 give the following results. The population comparison uses the reported axes after pooling near-random material. Width and population-fraction coverage counts are conditional on reporting the generating populations. Isotropic coverage includes all 32 draws.

| Phase and true texture | Correct populations | Correct and determined | Width within 1σ | Population fraction within 1σ | Isotropic fraction within 1σ | Mean joint χ²red |
|---|---:|---:|---:|---:|---:|---:|
| MAPbI₃ (110), 16°, φ = 0.38 | 32/32 | 29/32 | 21/32 | 20/32 | 20/32 | 1.00 |
| PbI₂ (001), 22°, φ = 0.29 | 31/32 | 30/32 | 21/31 | 17/31 | 17/32 | 1.01 |
| MAPbI₃ (110), 8° + 36°, φ = 0.38 | 32/32 | 32/32 | 16/32, 17/32 | 20/32, 21/32 | 22/32 | 1.00 |

Every fit included the true axis. One single-component PbI₂ draw passed the determination checks with a spurious second (001) population of about 0.9%. Four other single-component draws reported the generating populations, conditional on the preferred model because a competitor was within six BIC units. Interval coverage reflects counting noise, the estimated bin weights and model selection.

### Public grazing-incidence frames and sample roll

Run `python tools/benchmarks/sample_roll.py` for the surface-feature and ring estimates. Each frame uses its published αi, the adjacent PONI and Analyze’s default bad-pixel rejection. Xeuss also uses its published mask. Positive ψ tips the surface normal toward laboratory right. The columns give degrees and estimated standard errors.

| Frame | Horizon or Yoneda band | Ring symmetry or GISAXS rod | Selected ψ | Automatically applied ψ |
|---|---:|---:|---:|---:|
| P08 MAPbI₃ | −1.50 ± 1.63 | −1.30 ± 0.25 | Unresolved surface roll | 0 |
| Xeuss PbTe | −0.90 ± 0.78 | Unresolved | −0.90 ± 0.78, horizon | 0 |
| GALAXI GISAXS | 0.00 ± 0.23 | Unresolved | 0.00 ± 0.23, band | 0 |

The P08 fibre rings lean left. The edge tilt is within one standard error of zero, so this frame leaves the surface roll unresolved. The horizon uncertainty reflects variation across detector strips. The Xeuss beam is only 7 px from the left edge, and the GALAXI frame gives no independently resolved specular-rod slope.

The same command measures incidence independently of the supplied αi. The table gives degrees and 1σ errors at the calibrated direct beam. “Used” applies the automatic three-standard-error comparison with the published value.

Synthetic frames with a shadow edge and a specular spot at true αi = 0.310°, 0.520° and 0.450° gave 0.311 ± 0.021°, 0.519 ± 0.021° and 0.455 ± 0.021°, respectively, with 1.89° entered. These tests use tilted, non-square pixels and ψ = 2.2°. Both constituent estimates recover the generated angle within one reported standard error.

| Frame | Published αi | Shadow intercept | Specular spot | Yoneda band | Combined αi | Used |
|---|---:|---:|---:|---:|---:|---:|
| P08 MAPbI₃ | 0.075 | 0.65 ± 0.57 | 0.5012 ± 0.0125 | — | 0.5012 ± 0.0125 | 0.5012 |
| Xeuss PbTe | 0.235 | −0.067 ± 0.200 | — | — | Unresolved | 0.235 |
| GALAXI GISAXS | 0.463 | — | 0.4615 ± 0.0042 | 0.444 ± 0.026 | 0.4611 ± 0.0042 | 0.463 |

The P08 specular-like maximum lies at (545.46, 1755.93) px, above the direct beam at (545.46, 1826.95) px. The full shadow trace crosses about 46 px above the direct beam. Using only the nearer half of the strips gives 0.085°. The single specular-like spot provides 99.95% of the inverse-variance weight in the incidence estimate.

On Xeuss the beam position prevents an independent bilateral spot measurement, and the shadow intercept is consistent with zero. On GALAXI the band centre is at row 607.5 px. Its incidence estimate assumes the Si substrate critical angle 0.19483° from δ = 5.7816 × 10⁻⁶ in the BornAgain GALAXI model. The multilayer also contains PTFE and HMDSO. The specular spot by itself agrees with 0.463°. [Public geometries and source files](../../tests/data/external/README.md)

### P08 crystal orientation with internal film angles

`python tools/benchmarks/crystal_orientation.py --part public --json measured.json` uses the repository's two reference CIFs and applies the measured incidence with zero surface roll. The radial windows remain on the active q map. For each pixel the phase fit bins the real internal scattering-vector direction, then averages the optical factors over the same measured rays.

At αi = 0.50125° and ψ = 0°, the thick-film MAPbI₃ fit covers 732 measured bins and gives joint χ²red = 6.63. The 1.990 Å⁻¹ ring has χ²red = 18.17. The orientation verdict is **not determined**. Conditional on the preferred model, the narrow (001) population has a width of 11.43 ± 0.53° and occupies 3.42 ± 0.52% of the phase. The broad fraction correlates with the random fraction at −0.99985. Pooling them gives 96.58 ± 0.52% isotropic or near-random material. The PbI₂ joint χ²red is 1.76 at fixed thick-film depth. Freeing the thickness returns the thick limit. In that fit the broad PbI₂ fraction trades with random material and the narrow fraction falls below three standard errors, so its axis also remains conditional.

At the notebook incidence of 0.075° and the same zero roll (`--public-incidence-deg 0.075 --public-roll-deg 0`), MAPbI₃ gives joint χ²red = 14.93 and the strongest ring gives 45.47. The analogous values at the measured incidence are 6.63 and 5.09. Both choices leave the preferred axis conditional.

### Historical comparison at the ring-symmetry roll

The results in this subsection use commit `7604ae3b`, which applied the ring-symmetry roll and retained the active q-map pole angles in the phase fit.

On the 0.9973 Å⁻¹ ring, the vacuum horizon obeys `cos χ = (2π/λ) sin αi / q`. With λ = 0.688801 Å, its predicted |χ| is 85.41° at the measured incidence and 89.31° at the notebook angle. The observed loss around 84–86° agrees with the former. This check does not enter the incidence estimate.

At this revision the automatic analysis used αi = 0.50124° and ψ = −1.30459°. In its JSON, `crystal_orientation` used the thick-film limit and `benchmark_film_absorption` fitted one optical depth μd per phase. The public sample record gives the coating and annealing conditions but no measured thickness. [P08 sample and notebook](https://zenodo.org/records/17466183)

For the reference columns, run the same command at commit `27f8b22d` with `--json reference.json`. The refit there used the exit-field intensity T_f = |t_f|² multiplied by the geometrical film absorption F_A of Physics Manual §5.4. At `7604ae3b` add `--reference-json reference.json` to compare both refits. The benchmark compares the CIF hashes, geometry, q windows, χ bins, fit masks, signals and errors. Its `film_comparison` block records the per-ring χ²red and the measured and predicted intensities for fitted bins above 80°. Each model retains the per-ring intensity scales and evaluates its optical factor on each measured ray before folding. The fitted optical depth counts toward BIC and the effective degrees of freedom.

| Phase and q (Å⁻¹) | χ²red, T_f only | χ²red, T_f × geometrical F_A | χ²red, refracted waves |
|---|---:|---:|---:|
| **MAPbI₃ joint (746 bins)** | **18.32** | **4.24** | **3.37** |
| 0.9973 | 34.94 | 5.96 | 1.64 |
| 1.4091 | 7.75 | 2.71 | 2.46 |
| 1.7248 | 12.14 | 5.92 | 6.04 |
| 1.9903 | 26.37 | 3.97 | 2.89 |
| 2.2256 | 30.19 | 2.94 | 2.10 |
| 2.4382 | 6.61 | 3.88 | 3.72 |
| 2.8148 | 22.03 | 5.25 | 4.62 |
| 2.9854 | 21.47 | 3.78 | 3.13 |
| 3.4480 | 4.39 | 3.76 | 3.79 |
| 4.1040 | 3.71 | 3.66 | 3.64 |
| **PbI₂ 0.8989 (83 bins)** | **1.90** | **1.86** | **1.81** |

Both depth fits reach the thick limit for both phases. Fixing that limit in the wave model gives joint χ²red = 3.366 for MAPbI₃ and 1.786 for PbI₂. Fitting the depth adds one parameter per phase and gives 3.371 and 1.809. The frame provides no finite thickness estimate.

On the 0.9973 Å⁻¹ ring, the central ray at |χ| = 84.5° has αf ≈ 0.10° from sin αf = q cos χ / k − sin αi. Below the MAPbI₃ critical angle of 0.120° the exit wave samples less film. The 85.5° bin straddles the vacuum horizon and its measured rays lie above it. Residuals in the last two columns are (observed − fitted) / σ.

| Ring q (Å⁻¹) | &#124;χ&#124; (°) | Observed ± σ | Geometrical fit | Refracted fit | Geometrical residual | Refracted residual |
|---:|---:|---:|---:|---:|---:|---:|
| 0.9973 | 83.5 | 85.36 ± 5.96 | 89.70 | 85.18 | −0.73 | +0.03 |
| 0.9973 | 84.5 | 28.17 ± 3.27 | 84.37 | 24.13 | −17.19 | +1.24 |
| 0.9973 | 85.5 | −0.36 ± 3.56 | 3.56 | 0.45 | −1.10 | −0.23 |

Only the 1.7248 Å⁻¹ ring exceeds the determination limit of χ²red = 5. Its largest residual is at 85.5°, where the folded background-subtracted signal is −10.05 ± 2.05 and the wave model gives 8.70 (−9.16σ). At fixed thick-film depth, the MAPbI₃ fit contains a narrow (110) component, FWHM 11.08 ± 0.33°, with a phase fraction of 0.0483 ± 0.0022. A broad modulation at 78.68 ± 5.52° improves the profile BIC by 524.4 over a single (110) population plus isotropic material. Its fraction of 0.952 ± 0.428 trades strongly with the isotropic fraction. The report groups it as near-random and gives a pooled fraction of 0.9517 ± 0.0022. The failed ring prevents a quantitative determination of the overall crystal orientation.

The single PbI₂ ring passes the determination checks for two (001) populations, with FWHM values of 51.24 ± 5.27° and 13.47 ± 1.75° and phase fractions of 0.570 ± 0.336 and 0.049 ± 0.029. The isotropic fraction is 0.381 ± 0.362. This basal reflection supplies no independent check on the broad fraction.

### P08 crystal orientation before and after roll

To reproduce the historical exit-transmission comparison at the published angle, use commit `27f8b22d`. Run `python tools/benchmarks/crystal_orientation.py --part public --public-incidence-deg 0.075 --public-roll-deg 0 --json zero.json`, then omit `--public-roll-deg 0` to measure the roll. Both runs use the same two CIFs. Each ring has its own intensity scale. The central columns give χ²red for the original exit-angle cut and the CIF-based exit-transmission fit. The last column compares the two raw arms, using RMS of the difference between mean pixel intensities in matching 2° |χ| bins. Each bin must have at least 12 pixels on each arm in both runs.

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

The historical fits below use the dark-subtracted frame, its PONI (λ = 0.688801 Å) and αi = 0.075° from the public example. At these fixed angles the thick-film wave model at commit `7604ae3b` gave joint χ²red = 31.75 for MAPbI₃ and 2.76 for PbI₂ with q-map pole angles. The peak search assigns ten rings to tetragonal MAPbI₃ and one to PbI₂. The profiles use pixel-scatter errors, a locally fitted radial background constant in χ, polarization 0.99 and the original q windows. The MAPbI₃ CIF supplies four CH₆NPbI₃ units per cell for the index, and refined Pb and I sites for diffraction. These give ρₑ = 1.0423 e Å⁻³, δ = 2.18265 × 10⁻⁶, β = 1.26942 × 10⁻⁷ and αc = 0.1197°. The PbI₂ fixture uses the symmetry of its declared space group P −3 m 1, which gives one Pb and two I atoms per cell. [Public dataset and notebook](https://zenodo.org/records/17466183)

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
