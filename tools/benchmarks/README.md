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
the `h5clear` executable was missing. The default suite includes the exact Predict regression for known-composition
cuts 184 and 167 of the first benchmark. Run `python -m pytest -q tests/test_predict_regression.py`
to repeat both fits (24 s here).

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
| `summary_tables.py` | prints the tables of Physics and Methods 9.13 and 10.3 and of the model card from the stored summaries (`--lang zh` for the Chinese headers, `--only recovery extended sensitivity own dwba pairs timing v5 large card`); computes nothing again | – |
| `search_baseline.py` | the baseline without a network: Fitting's *Search the ranges, then refine* from its default values on the same 200 held-out curves, the true composition given (median error of R 2.8 % / 6.7 %, 90th percentile 81 % / 136 %, about 10 s per curve; Predict parameters 0.5 % / 0.7 % and 10 % / 14 %, `posterior_recovery_summary.json`) | `search_baseline_summary.json` |
| `heldout_cases.py` | a second held-out set drawn as fit1d's, on a seed range never used during development (20261201): one to three populations, the LMA or the DA | – |
| `v5_recovery.py` | 1D Predict V5 (removed in 0.2.0; runs in a checkout of commit 06e7395) on fit1d's 200 held-out curves, the true composition given (Physics and Methods, section 10.3) | `history/v5_recovery_summary.json` |
| `galaxi_posterior.py` | *Predict parameters* on the measured GALAXI cut and on BornAgain's model of it (Physics and Methods, section 9.13) | `galaxi_posterior.json` |
| `analyze_speed.py` | how long Analyze takes to reduce the public frames of `tests/data/external` with their own `.poni` (first frame and every further one) | `analyze_speed.json` |
| `pyfai_comparison.py` | GIMaP against pyFAI on the public frames: |q|, 2θ, q∥, q_z, solid angle and polarisation of every pixel, and the azimuthal I(q) on the same pixels and bins (Physics and Methods 3.7, 6.1). Needs pyFAI, which GIMaP does not install | `pyfai_comparison.json` |
| `pygid_comparison.py` | GIMaP against pygid on its two example frames: q∥ and q_z of every pixel, and the rings of pygid's polar map; with `--published DIR`, also the maps published with the examples (Zenodo 10.5281/zenodo.17466183) (Physics and Methods 3.7). Needs pygid | `pygid_comparison.json` |
| `make_external_reference.py` | writes `tests/data/external/reference_values.json`: pyFAI's and pygid's q at fixed pixels of the public frames, which `tests/test_external_reference.py` checks GIMaP against without either installed. Needs pyFAI and pygid | `tests/data/external/reference_values.json` |
| `galaxi_fit.py` | Fitting on the measured GALAXI cut, sphere and paracrystal (EIA or two paths, LMA or DA, with BornAgain's D held or free) (Physics and Methods 9.10) | `galaxi_fit.json` |
| `galaxi_fit1d.py` | fit1d (removed in 0.2.0; a copy of the kit in `utils/gisaxs1d_gui_kit`) on the measured GALAXI cut and on BornAgain's model of it: one sphere in Born or DWBA, and the composition chosen by fit1d (Physics and Methods 9.12). Needs BornAgain | `galaxi_fit1d.json` |
| `galaxi_bornagain_model.py` | BornAgain's model of the GALAXI frame on the same cut: how well it describes the measured cut, and what Fitting recovers from it (Physics and Methods 9.10). Needs BornAgain | `galaxi_bornagain_model.json` |
| `cut_model_checks.py` | `cut_model`, GIMaP's one 1D forward model, against Fitting's formulas before 0.3, 1D Predict V5's forward, the fit1d kit's tables (when a copy of the kit is in `utils/gisaxs1d_gui_kit`) and the independent `yoneda_truth.py` (Born at the cut's q_z and four-term DWBA, LMA and DA); time of 1000 curves on one core (Physics and Methods 9.2) | `cut_model_checks.json` |
| `cut_model_bornagain.py` | `cut_model` against BornAgain: spheres and upright cylinders on Si in the four-term DWBA, one size, a size distribution and a radial paracrystal (Physics and Methods 9.2). Needs BornAgain | `cut_model_bornagain.json` |
| `yoneda_truth.py` | the forward model of a horizontal GISAXS cut used by the fit1d benchmark: sphere or upright cylinder on a substrate, four-term DWBA, paracrystal, resolution, beam tail, roughness, counts. It shares no code with GIMaP or fit1d | – |

The other scripts (the commands of the paper's Predict runs are above):

```bash
GISAXS1D_KIT_HOME=<copy of the kit> python tools/benchmarks/fit1d_recovery.py --n 200 --workers 10
GISAXS1D_KIT_HOME=<copy of the kit> python tools/benchmarks/fit1d_recovery.py --set extended --n 1200 --seed0 20261201 --out fit1d_heldout
python tools/benchmarks/search_baseline.py --n 200 --workers 10
python tools/benchmarks/galaxi_posterior.py   # in an environment with BornAgain 24
python tools/benchmarks/analyze_speed.py --repeat 5 --json tools/benchmarks/analyze_speed.json
python tools/benchmarks/pyfai_comparison.py --json tools/benchmarks/pyfai_comparison.json  # in an environment with pyFAI
python tools/benchmarks/pygid_comparison.py --json tools/benchmarks/pygid_comparison.json --published FOLDER  # with pygid
python tools/benchmarks/galaxi_fit.py
python tools/benchmarks/galaxi_bornagain_model.py
python tools/benchmarks/cut_model_checks.py            # on a clean checkout: add --out <file> (or --force)
python tools/benchmarks/cut_model_bornagain.py   # in an environment with BornAgain 24
```

Times depend on the computer; `analyze_speed.json` names the one it was measured on.

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
