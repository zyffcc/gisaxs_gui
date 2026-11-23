# Earlier benchmark results

These records preserve results from earlier versions of the code and from methods removed from GIMaP.
The Predict benchmarks cited in the paper, Physics and Methods §9.13 and the model card are documented
in the [benchmark README](../README.md).

| File | What it is |
|---|---|
| `vertical_height_bias.json` | Source for the revision 14 joint-fit height comparison. It contains the original BornAgain sweep and the later `height_averaged_da_baseline` check. The [current benchmark](../README.md#joint-fit-height-bias-and-warnings-on-the-shipped-model) records the warnings produced by the shipped model. |
| `posterior_recovery_first_summary.json`, `posterior_extended_first_summary.json` | *Predict parameters* with the network flow_v3, before the rules of Physics and Methods 9.13 steps 3 and 5 (200 and 1200 curves) |
| `posterior_dwba_si_fit1d_summary.json`, `posterior_dwba_ps100_fit1d_summary.json`, `posterior_dwba_si_born_fit1d_summary.json`, `posterior_dwba_si_eia_fit1d_summary.json` | earlier code on the 200 curves drawn with the four DWBA terms, on Si and on 100 nm polystyrene on Si (both angles given), and on Si with q_z alone or nothing given |
| `posterior_dwba_si_extended_summary.json`, `posterior_dwba_ps100_extended_summary.json`, `posterior_dwba_si_born_extended_summary.json`, `posterior_dwba_si_eia_extended_summary.json` | the same on 1200 curves of the extended set |
| `posterior_dwba_pairs_fit1d.json`, `posterior_dwba_pairs_extended.json` | the pairs of those runs by q₊R (`posterior_pairs.py`) |
| `posterior_own_extended_summary.json` | earlier code on 300 curves of the extended kind drawn in Fitting's own model (development seeds) |
| `fit1d_recovery_summary.json`, `fit1d_heldout_summary.json` | the removed fit1d method on the 200 held-out curves and on 1200 curves of the extended set (Physics and Methods 9.12; the fit1d columns of 9.13) |
| `v5_recovery_summary.json` | the removed 1D Predict V5 on the 200 held-out curves (Physics and Methods 10.3) |

The earlier `posterior_recovery_summary.json`, `posterior_extended_summary.json` (1200 curves) and
`posterior_own_summary.json` were replaced by the final runs under the same names; they are in the repository's
history (commit 34574e9 and before).
