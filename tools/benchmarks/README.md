# Benchmarks

Scripts behind the numbers that *Physics and Methods* and the GIMaP paper quote. They use public or synthetic data
only and write their summaries next to themselves.

| Script | What it measures | Summary |
|---|---|---|
| `fit1d_recovery.py` | how well *Predict parameters (fit1d)* recovers known parameters on 200 curves it has never seen, made with the independent forward model `yoneda_truth.py` (Physics and Methods, section 9.12) | `fit1d_recovery_summary.json` |
| `analyze_speed.py` | how long Analyze takes to reduce the public frames of `tests/data/external` with their own `.poni` (first frame and every further one) | `analyze_speed.json` |
| `yoneda_truth.py` | the forward model of a horizontal GISAXS cut used by the fit1d benchmark: sphere or upright cylinder on a substrate, four-term DWBA, paracrystal, resolution, beam tail, roughness, counts. It shares no code with GIMaP or fit1d | – |

```bash
python tools/benchmarks/fit1d_recovery.py --n 200 --workers 10
python tools/benchmarks/analyze_speed.py --repeat 5 --json tools/benchmarks/analyze_speed.json
```

Times depend on the computer; `analyze_speed.json` names the one it was measured on.
