# Benchmarks

Scripts behind the numbers that *Physics and Methods* and the GIMaP paper quote. They use public or synthetic data
only and write their summaries next to themselves.

The fit1d kit (*Predict parameters (fit1d)*) was removed from GIMaP in 0.2.0; it remains in the repository's history
(`utils/gisaxs1d_gui_kit`). The scripts that compare against it keep the record of that comparison and need a copy of
the kit: `fit1d_recovery.py` takes it from the environment variable `GISAXS1D_KIT_HOME`; `galaxi_fit1d.py` and the
fit1d part of `cut_model_checks.py` look for it in `utils/gisaxs1d_gui_kit` (the latter skips that part without it).
The held-out curves themselves (`recovery_cases.py`, `heldout_cases.py`) need neither the kit nor GIMaP.

| Script | What it measures | Summary |
|---|---|---|
| `recovery_cases.py` | the held-out curves of the 1D recovery benchmarks: fit1d's 200 (seed 20261116), their draw and their curve from the independent forward model `yoneda_truth.py`; shared by the scripts below | – |
| `fit1d_recovery.py` | how well *Predict parameters (fit1d)* (removed in 0.2.0; a copy of the kit in `GISAXS1D_KIT_HOME`) recovered known parameters on the 200 curves of `recovery_cases.py` (Physics and Methods, section 9.12); with `--set extended` on `heldout_cases.py`'s 1200 curves | `fit1d_recovery_summary.json`, `fit1d_heldout_summary.json` |
| `posterior_recovery.py` | how well *Predict parameters* (the proposal network, its posterior and refinement) recovers known parameters on fit1d's 200 held-out curves (`--set fit1d`) or on 1200 curves of `heldout_cases.py` (`--set extended`): errors, coverage of the 68 %/95 % intervals, how often the most probable solution is in the truth's mode, the composition chosen, time per curve on one core (Physics and Methods, section 9.13); `--truth own` draws the same cases in Predict's own model (the coverage when the model is exact), `--model-error-floor`, `--explored`, `--moves` and `--widths` set those steps | `posterior_recovery_summary.json`, `posterior_extended_summary.json` (`*_first_summary.json`: the run before the rules of Physics and Methods 9.13 steps 3 and 5 were added), `posterior_own_summary.json`, `posterior_own_extended_summary.json` (`--truth own` on the development seeds 20260601) |
| `heldout_cases.py` | a second held-out set drawn as fit1d's, on a seed range never used during development (20261201): one to three populations, the LMA or the DA | – |
| `v5_recovery.py` | 1D Predict V5 (removed in 0.2.0; runs in a checkout of commit 06e7395) on fit1d's 200 held-out curves, the true composition given (Physics and Methods, section 10.3) | `v5_recovery_summary.json` |
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

```bash
GISAXS1D_KIT_HOME=<copy of the kit> python tools/benchmarks/fit1d_recovery.py --n 200 --workers 10
GISAXS1D_KIT_HOME=<copy of the kit> python tools/benchmarks/fit1d_recovery.py --set extended --n 1200 --seed0 20261201 --out fit1d_heldout
python tools/benchmarks/posterior_recovery.py --network src/gimap/features/fitting/domain/posterior/data/flow_v1 --n 200
python tools/benchmarks/posterior_recovery.py --network src/gimap/features/fitting/domain/posterior/data/flow_v1 --set extended --n 1200 --seed0 20261201 --out posterior_extended
python tools/benchmarks/galaxi_posterior.py   # in an environment with BornAgain 24
python tools/benchmarks/analyze_speed.py --repeat 5 --json tools/benchmarks/analyze_speed.json
python tools/benchmarks/pyfai_comparison.py --json tools/benchmarks/pyfai_comparison.json  # in an environment with pyFAI
python tools/benchmarks/pygid_comparison.py --json tools/benchmarks/pygid_comparison.json --published FOLDER  # with pygid
python tools/benchmarks/galaxi_fit.py
python tools/benchmarks/galaxi_bornagain_model.py
python tools/benchmarks/cut_model_checks.py
python tools/benchmarks/cut_model_bornagain.py   # in an environment with BornAgain 24
```

Times depend on the computer; `analyze_speed.json` names the one it was measured on.
