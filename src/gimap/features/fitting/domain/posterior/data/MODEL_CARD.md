# Model card: flow_v1 (GIMaP's proposal network of 1D cuts)

Trained as `flow_v4`, shipped as `flow_v1`.

**What it is.** An amortised posterior p(composition, values | curve) for horizontal GISAXS cuts: a 1D CNN summary, a head over the 258 compositions of one to three populations (sphere, upright or random cylinder, each with or without a paracrystal) and a conditional rational-quadratic spline flow over their values (Physics and Methods, section 9.13). GIMaP uses it only to propose: the values, intervals and probabilities of a solution come from the model and the curve (importance sampling and least squares).

**Files.** `flow_v1.npz` (float32 arrays, SHA-256 `7514c66e8cf384d9d4bc6ed7bdad1fbbca3d7d8e0c1b83c7c1aa9ea05a36ec7e`), `.json` (architecture, prior, data, training), this card. Encoding version 4; format `gimap_ml1d_flow_v2`.

**Architecture.** channels [32, 64, 96, 128], kernel 5, dense 512, summary 256, context 256, layers 10, hidden 256, blocks 2, bins 10, bound 6.0, head 256.

**Training data.** 5000000 cuts simulated with cut_model from `prior_v2.json` (seed 555, code a10d26d): One to three populations (sphere, upright or random cylinder, each with or without a paracrystal), Gaussian sizes by number, LMA; the cut at the effective interface, the Born term at q+ = k_i + k_f, or (both angles known) spheres with the exit-reflected path at q- = k_i - k_f and an effective reflection coefficient r; the bounds of the values are those of posterior/encoding.py (encoding 4). Every batch draws the populations' shares, the beam tail, the background, the model error, the measured points, Poisson counts and gaps anew.

**Training.** Neural posterior estimation (−log q of the values + cross-entropy of the head), AdamW, batch 2048, learning rate 0.0007 with warm-up and cosine decay, 30 epochs, early stopping on 20000 held-out cuts; best validation loss 22.625163134765625; torch 2.5.1+cu121. Last epoch: {"epoch": 30, "train_nll": 19.1828, "train_ce": 3.4817, "validation_nll": 19.1406, "validation_ce": 3.4872, "validation_loss": 22.6279, "validation_composition_top1": 0.2091, "seconds": 6441.4}.

## Calibration on simulated cuts not trained on

2000 cuts, 1000 samples each, the true composition given; composition head top-1 0.202; 168 ms per curve on one thread.

| value | n | 68 % coverage | 95 % coverage | rank χ² (19 dof) | median error |
|---|---|---|---|---|---|
| R | 4041 | 0.674 | 0.939 | 11.9 | 0.202 (relative) |
| sigma_R | 4041 | 0.673 | 0.943 | 12.7 | 0.088 (absolute) |
| h_over_2R | 974 | 0.696 | 0.947 | 14.3 | 0.616 (relative) |
| sigma_h | 974 | 0.655 | 0.923 | 29.2 | 0.115 (absolute) |
| D_over_2R | 2879 | 0.666 | 0.948 | 19.8 | 0.361 (relative) |
| sigma_nn | 2879 | 0.676 | 0.944 | 22.8 | 0.047 (absolute) |
| tail_width | 2000 | 0.692 | 0.947 | 21.4 | 0.410 (relative) |
| tail_exponent | 2000 | 0.679 | 0.946 | 23.2 | 0.089 (absolute) |
| q_resolution | 2000 | 0.688 | 0.946 | 20.0 | 0.706 (relative) |
| model_error | 2000 | 0.674 | 0.947 | 8.6 | 0.564 (relative) |
| reflection_re | 2000 | 0.671 | 0.944 | 15.1 | 0.478 (absolute) |
| reflection_im | 2000 | 0.666 | 0.947 | 19.3 | 0.478 (absolute) |

## Benchmarks

Predict parameters with this network on held-out curves of an independent forward model (`tools/benchmarks/posterior_recovery.py`; Physics and Methods, section 9.13). These are the runs the GIMaP paper quotes, all made with one version of the fitting code (git tree 7e68831a of `src/gimap/features/fitting/domain`, commits 2eb0243 and 7179b85); `tools/benchmarks/summary_tables.py --only card` printed this section from `tools/benchmarks/posterior_*_summary.json`, and every cut is in `tools/benchmarks/per_cut/`. Errors of R and D are relative, of the spreads absolute; *network* is the network's median alone, *fit* the most probable solution with the true composition given; *model* curves have the physics of the removed fit1d method, *measured* ones a resolution, a beam tail and roughness. "True composition most probable" is judged on the most probable solution; the truth's mode is every R and D within 20 % of the truth. Time per curve on one core (24 cuts, one worker, Intel Core i9-13900HX; `posterior_timing_summary.json`): network 0.35 s, composition given 6.2 s, chosen among one or two populations 25.6 s. Runs of earlier code are in `tools/benchmarks/history/`.

### Benchmark: posterior_recovery (200 held-out curves, one or two populations)

| stage | R (median / P90 · 68 % / 95 % coverage) | σR/R | D | σnn/D |
|---|---|---|---|---|
| network/all | 0.0824 / 0.395 | 0.0307 / 0.119 | 0.0069 / 0.090 | 0.0031 / 0.020 |
| fit/model | 0.0054 / 0.100 · 0.70 / 0.87 | 0.0054 / 0.074 · 0.76 / 0.95 | 0.0004 / 0.004 · 0.91 / 0.94 | 0.0005 / 0.005 · 0.89 / 0.95 |
| fit/measured | 0.0069 / 0.143 · 0.65 / 0.83 | 0.0077 / 0.090 · 0.69 / 0.89 | 0.0010 / 0.004 · 0.78 / 0.95 | 0.0016 / 0.006 · 0.58 / 0.91 |
| fit/all | 0.0061 / 0.138 · 0.67 / 0.85 | 0.0066 / 0.086 · 0.73 / 0.92 | 0.0007 / 0.004 · 0.85 / 0.95 | 0.0010 / 0.005 · 0.73 / 0.93 |

True composition most probable (the composition of the most probable solution is the true one), model / measured / all: 0.88 / 0.65 / 0.77; true composition listed: 0.96 / 0.84 / 0.90; truth's mode first: 0.89 / 0.89 / 0.89; listed: 0.94 / 0.92 / 0.93.

### Benchmark: posterior_extended (600 held-out curves of the extended set, one to three populations)

| stage | R (median / P90 · 68 % / 95 % coverage) | σR/R | D | σnn/D |
|---|---|---|---|---|
| network/all | 0.1395 / 0.450 | 0.0526 / 0.134 | 0.0265 / 0.753 | 0.0082 / 0.061 |
| fit/model | 0.0148 / 0.276 · 0.81 / 0.89 | 0.0114 / 0.148 · 0.84 / 0.93 | 0.0006 / 0.020 · 0.86 / 0.90 | 0.0008 / 0.016 · 0.86 / 0.92 |
| fit/measured | 0.0212 / 0.446 · 0.67 / 0.81 | 0.0155 / 0.180 · 0.71 / 0.87 | 0.0013 / 0.062 · 0.79 / 0.89 | 0.0018 / 0.013 · 0.77 / 0.91 |
| fit/all | 0.0170 / 0.396 · 0.74 / 0.85 | 0.0130 / 0.166 · 0.78 / 0.90 | 0.0009 / 0.034 · 0.82 / 0.89 | 0.0013 / 0.014 · 0.81 / 0.92 |

True composition most probable (the composition of the most probable solution is the true one), model / measured / all: 0.69 / 0.58 / 0.64; true composition listed: 0.81 / 0.75 / 0.78; truth's mode first: 0.80 / 0.75 / 0.78; listed: 0.90 / 0.83 / 0.86.

### Benchmark: posterior_dwba_tp (the 200 curves drawn with the four DWBA terms on Si, both angles given)

| stage | R (median / P90 · 68 % / 95 % coverage) | σR/R | D | σnn/D |
|---|---|---|---|---|
| network/all | 0.0489 / 0.192 | 0.0317 / 0.114 | 0.0051 / 0.025 | 0.0025 / 0.010 |
| fit/model | 0.0196 / 0.149 · 0.60 / 0.78 | 0.0130 / 0.090 · 0.72 / 0.91 | 0.0004 / 0.002 · 0.97 / 1.00 | 0.0005 / 0.003 · 0.97 / 0.99 |
| fit/measured | 0.0338 / 0.226 · 0.55 / 0.67 | 0.0190 / 0.123 · 0.65 / 0.81 | 0.0011 / 0.004 · 0.80 / 0.99 | 0.0016 / 0.005 · 0.68 / 0.95 |
| fit/all | 0.0296 / 0.198 · 0.57 / 0.73 | 0.0169 / 0.121 · 0.68 / 0.86 | 0.0006 / 0.003 · 0.88 / 0.99 | 0.0009 / 0.004 · 0.82 / 0.97 |

True composition most probable (the composition of the most probable solution is the true one), model / measured / all: 0.80 / 0.59 / 0.69; true composition listed: 1.00 / 0.86 / 0.93; truth's mode first: 0.94 / 0.80 / 0.87; listed: 1.00 / 0.93 / 0.96.

## Where it does not apply

- cuts spanning less than a factor 10 in q; points below q_max/300 are left out;
- more than three populations, other families, random cylinders larger than R q_max = 50 or h q_max = 200, beam tails wider than 30 q_min or with an exponent outside 1.8–2.2 (encoding 4);
- the network itself knows Gaussian sizes by number and the LMA; other laws and the DA are weighted from its proposals; the four-term DWBA is not in the model (with both angles of the cut, spheres have the direct and the exit-reflected path with a free effective reflection coefficient: two paths); on a cut without a reflected wave a smaller sphere with a large |r| can describe the curve as well;
- its own probabilities are those of the simulated prior; GIMaP reports the importance-weighted ones.
