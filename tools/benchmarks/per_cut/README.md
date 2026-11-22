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

| File | Cuts | Written at commit | Fitting code |
|---|---|---|---|
| `posterior_recovery.json.gz` | 200 | 2eb0243 (the summary has no `code` field) | tree 7e68831a |
| `posterior_extended.json.gz` | 600 | 2eb0243 (the summary has no `code` field) | tree 7e68831a |
| `posterior_floor03.json.gz`, `posterior_floor08.json.gz`, `posterior_amplitude100.json.gz` | 200 each | 7179b85 | tree 7e68831a |
| `posterior_timing.json.gz` | 24 | 7179b85 | tree 7e68831a |
| `posterior_dwba_tp.json.gz`, `posterior_dwba_born.json.gz`, `posterior_dwba_eia.json.gz` | 200 each | 7179b85 | tree 7e68831a |
| `posterior_own.json.gz` | 200 | 7179b85 | tree 7e68831a |
| `posterior_dwba4_si.json.gz`, `posterior_dwba4_si_auto.json.gz` | 200 each | fda067c | tree edf4d396 |
| `posterior_dwba4_ps100.json.gz`, `posterior_dwba4_ps100_auto.json.gz` | 200 each | fda067c | tree edf4d396 |

The fitting tree identifies `src/gimap/features/fitting/domain` at the recorded commit.
The two runs written at 2eb0243 predate the `code` field of the summaries, so their commit and tree
are stated here and not added to the result files. The network is `flow_v1` (its SHA-256 is in every summary).
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
