# Per-cut results of the paper's Predict parameters runs

Every cut of the runs that the GIMaP paper, Physics and Methods 9.13 and the network's model card quote
(`../README.md`, *The runs the paper quotes*). All curves are simulated; no measured data are in these files.

Each `posterior_NAME.json.gz` is the gzip-compressed file `results/NAME.json` that
`tools/benchmarks/posterior_recovery.py` wrote, unchanged, and belongs to `../posterior_NAME_summary.json`:

```python
import gzip, json
run = json.loads(gzip.decompress(open("tools/benchmarks/per_cut/posterior_recovery.json.gz", "rb").read()))
run["summary"]            # the same object as ../posterior_recovery_summary.json
run["results"][0]         # one cut: "case" (the truth: composition, level, populations, counts, seed),
                          # "network" (the network's medians), "fit" (the true composition given) and "choose"
                          # (the composition chosen): the most probable solution with its values, errors, 68 % and
                          # 95 % intervals, chi2, flags, probability, "in_set" and "alternatives" (the other solutions
                          # with a probability of at least 0.05), seconds
run["seconds"]            # wall time of this invocation (the final part for checkpointed runs)
```

`tools/benchmarks/posterior_pairs.py` reads these files directly
(`LABEL=tools/benchmarks/per_cut/posterior_NAME.json.gz`).

These historical archives omit modes below the display cutoff and the full ordered composition weights.
The current scorer reports how many cuts lack those weights. The historical Table 3 definitions and their
source fields are available through `../table3_sources.py`. For the independent rerun and its per-cut
fields, see [Running an independent validation](../README.md#running-an-independent-validation).

| File | Cuts | Written at commit | Fitting code |
|---|---|---|---|
| `posterior_recovery.json.gz` | 200 | 2eb0243 (the summary has no `code` field) | tree 7e68831a |
| `posterior_extended.json.gz` | 600 | 2eb0243 (the summary has no `code` field) | tree 7e68831a |
| `posterior_newseed.json.gz` | 300 | 6bb1a18 | see summary.code.fitting_tree |
| `posterior_newseed_auto2.json.gz` | 300 | f73d9c8 | see summary.code.fitting_tree |
| `posterior_dwba_auto2.json.gz` | 200 | 9c90e6a | see summary.code.fitting_tree |
| `posterior_floor03.json.gz`, `posterior_floor08.json.gz`, `posterior_amplitude100.json.gz` | 200 each | 7179b85 | tree 7e68831a |
| `posterior_timing.json.gz` | 24 | 7179b85 | tree 7e68831a |
| `posterior_dwba_tp.json.gz`, `posterior_dwba_born.json.gz`, `posterior_dwba_eia.json.gz` | 200 each | 7179b85 | tree 7e68831a |
| `posterior_own.json.gz` | 200 | 7179b85 | tree 7e68831a |
| `posterior_dwba4_si.json.gz`, `posterior_dwba4_si_auto.json.gz` | 200 each | fda067c | tree edf4d396 |
| `posterior_dwba4_ps100.json.gz`, `posterior_dwba4_ps100_auto.json.gz` | 200 each | fda067c | tree edf4d396 |
| `posterior_r17_first.json.gz` | 200 | 69d8d668 | tree d8b9dffd |
| `posterior_r17_fresh.json.gz` | 300 | 19578769 | tree d8b9dffd |
| `posterior_r17_extended.json.gz` | 600 | 19578769 | tree d8b9dffd |
| `posterior_r17_dwba_tp.json.gz`, `posterior_r17_dwba_born.json.gz`, `posterior_r17_dwba_eia.json.gz` | 200 each | 19578769 | tree d8b9dffd |
| `posterior_r17_dwba4_si.json.gz`, `posterior_r17_dwba4_ps100.json.gz` | 200 each | 19578769 | tree d8b9dffd |
| `posterior_r17_own.json.gz` | 200 | 19578769 | tree d8b9dffd |
| `posterior_r17_floor03.json.gz`, `posterior_r17_floor08.json.gz`, `posterior_r17_amplitude100.json.gz` | 200 each | 19578769 | tree d8b9dffd |
| `posterior_r17_timing.json.gz` | 24 | 19578769 | tree d8b9dffd |
| `search_baseline_r17.json.gz` (`search_baseline.py`, not a Predict run) | 200 | 19578769 | tree d8b9dffd |
| `posterior_r18_first.json.gz` | 200 | 84151378 | tree d8b9dffd |
| `posterior_r18_fresh.json.gz` | 300 | 84151378 | tree d8b9dffd |
| `posterior_r18_extended.json.gz` | 600 | 84151378 | tree d8b9dffd |
| `posterior_r18_dwba_tp.json.gz`, `posterior_r18_dwba_born.json.gz`, `posterior_r18_dwba_eia.json.gz` | 200 each | 84151378 | tree d8b9dffd |
| `posterior_r18_dwba4_si.json.gz`, `posterior_r18_dwba4_ps100.json.gz` | 200 each | 84151378 | tree d8b9dffd |
| `posterior_r18_own.json.gz` | 200 | 84151378 | tree d8b9dffd |
| `posterior_r18_floor03.json.gz`, `posterior_r18_floor08.json.gz`, `posterior_r18_amplitude100.json.gz` | 200 each | 84151378 | tree d8b9dffd |
| `posterior_r18_timing.json.gz` | 24 | 84151378 | tree d8b9dffd |
| `search_baseline_r18.json.gz` (`search_baseline.py`, not a Predict run) | 200 | 84151378 | tree d8b9dffd |

The fitting tree identifies `src/gimap/features/fitting/domain` at the recorded commit.
The two runs written at 2eb0243 predate the `code` field of the summaries, so their commit and tree
are stated here and not added to the result files. The network is `flow_v1` (its SHA-256 is in every summary);
the r17 and r18 runs use the `flow_v1` trained as `flow_v6` (encoding 6), the others the one before it.
The r18 runs use root seeds of their own (41261125 for the first and third sets, 41261201 for the extended
set), recorded as `seed0` in each summary.
The four level-4 runs used per-cut checkpoints. Each stage's `seconds` measures that fit, including for resumed draws.
The `code.source_sha256` in each level-4 summary identifies the benchmark scripts and fitting sources used for those draws.

`evidence_integration_check.json.gz` is not a Predict run: it is everything `../evidence_integration_check.py` computed
(every parameter point of the 24 solutions, with the scales at the least-squares and at the joint optimum, both runs
of the joint integral and the parts of the difference), written at 958aa9b with the fitting code of tree `d139e6f5`;
`../evidence_integration_check.json` holds the same results with the posterior draws as rows.
`calibration_check_frames.json.gz` holds every simulated frame of `../calibration_check.py` (`calibration_coverage.py`).
`main` and `three_ring_set` each contain `meta` (command, frame count, first seed and seconds) and `frames`.
Each frame records the simulated geometry and the result from the shipped engine, including whether it found the
geometry. The headless beam centre, distance and recovery flag record the independent line check's choice
under the same criterion. Successful fits include the reported 1σ, the covariance standard errors on the final points and the
largest relative difference when repeating the reported uncertainty estimate. Three-ring records include all
three two-ring refits. The recorded method indicates whether the tilts varied in those refits. Run `--part summarize` to rebuild the
coverage summaries in `../calibration_check.json`.
