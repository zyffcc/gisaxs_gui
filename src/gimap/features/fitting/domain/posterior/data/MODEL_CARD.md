# Model card: flow_v1 (GIMaP's proposal network of 1D cuts)

Trained as `flow_v3`, shipped as `flow_v1`.

**What it is.** An amortised posterior p(composition, values | curve) for horizontal GISAXS cuts: a 1D CNN summary, a head over the 258 compositions of one to three populations (sphere, upright or random cylinder, each with or without a paracrystal) and a conditional rational-quadratic spline flow over their values (Physics and Methods, section 9.13). GIMaP uses it only to propose: the values, intervals and probabilities of a solution come from the model and the curve (importance sampling and least squares).

**Files.** `flow_v1.npz` (float32 arrays, SHA-256 `bd991ca36cac7fd9a263fe1146a60a4cd00ec82448928893bfac4951509ee6cc`), `.json` (architecture, prior, data, training), this card. Encoding version 3; format `gimap_ml1d_flow_v2`.

**Architecture.** channels [32, 64, 96, 128], kernel 5, dense 512, summary 256, context 256, layers 10, hidden 256, blocks 2, bins 10, bound 6.0, head 256.

**Training data.** 5000000 cuts simulated with cut_model from `prior_v1.json` (seed 333, code e5a609e): One to three populations (sphere, upright or random cylinder, each with or without a paracrystal), Gaussian sizes by number, LMA; the bounds of the values are those of posterior/encoding.py. Every batch draws the populations' shares, the beam tail, the background, the model error, the measured points, Poisson counts and gaps anew.

**Training.** Neural posterior estimation (−log q of the values + cross-entropy of the head), AdamW, batch 2048, learning rate 0.0007 with warm-up and cosine decay, 30 epochs, early stopping on 20000 held-out cuts; best validation loss 18.771033056640626; torch 2.5.1+cu121. Last epoch: {"epoch": 30, "train_nll": 15.3123, "train_ce": 3.455, "validation_nll": 15.3122, "validation_ce": 3.4588, "validation_loss": 18.771, "validation_composition_top1": 0.2044, "seconds": 6862.5}.

## Calibration on simulated cuts not trained on

2000 cuts, 1000 samples each, the true composition given; composition head top-1 0.209; 279 ms per curve on one thread.

| value | n | 68 % coverage | 95 % coverage | rank χ² (19 dof) | median error |
|---|---|---|---|---|---|
| R | 3999 | 0.664 | 0.941 | 27.7 | 0.230 (relative) |
| sigma_R | 3999 | 0.675 | 0.946 | 10.4 | 0.091 (absolute) |
| h_over_2R | 1010 | 0.693 | 0.948 | 20.3 | 0.642 (relative) |
| sigma_h | 1010 | 0.705 | 0.952 | 18.9 | 0.110 (absolute) |
| D_over_2R | 2796 | 0.683 | 0.951 | 10.6 | 0.373 (relative) |
| sigma_nn | 2796 | 0.689 | 0.949 | 23.6 | 0.049 (absolute) |
| tail_width | 2000 | 0.674 | 0.941 | 15.2 | 0.434 (relative) |
| tail_exponent | 2000 | 0.681 | 0.943 | 16.4 | 0.086 (absolute) |
| q_resolution | 2000 | 0.669 | 0.956 | 18.4 | 0.713 (relative) |
| model_error | 2000 | 0.683 | 0.945 | 22.7 | 0.529 (relative) |

## Benchmark: posterior_recovery

| stage | R (median / P90 · 68 % coverage) | σR/R | D | σnn/D |
|---|---|---|---|---|
| network/all | 0.0540 / 0.368 | 0.0298 / 0.126 | 0.0056 / 0.070 | 0.0030 / 0.020 |
| fit/model | 0.0076 / 0.451 · 0.48 | 0.0061 / 0.118 · 0.57 | 0.0005 / 0.006 · 0.62 | 0.0005 / 0.006 · 0.57 |
| fit/measured | 0.0074 / 0.245 · 0.43 | 0.0084 / 0.143 · 0.47 | 0.0011 / 0.009 · 0.45 | 0.0016 / 0.006 · 0.45 |
| fit/all | 0.0075 / 0.377 · 0.46 | 0.0074 / 0.126 · 0.52 | 0.0007 / 0.007 · 0.53 | 0.0011 / 0.006 · 0.51 |

Composition chosen: {"model": {"n": 100, "exact": 0.84, "true_in_equivalent_set": 0.94}, "measured": {"n": 100, "exact": 0.65, "true_in_equivalent_set": 0.85}, "all": {"n": 200, "exact": 0.745, "true_in_equivalent_set": 0.895}}; median seconds: {"network": 0.7356773529900238, "fit": 7.229943868936971, "choose": 40.18591395649128}.

## Benchmark: posterior_extended

| stage | R (median / P90 · 68 % coverage) | σR/R | D | σnn/D |
|---|---|---|---|---|
| network/all | 0.1290 / 0.470 | 0.0515 / 0.139 | 0.0298 / 0.747 | 0.0085 / 0.060 |
| fit/model | 0.0271 / 0.585 · 0.35 | 0.0222 / 0.242 · 0.37 | 0.0010 / 0.324 · 0.42 | 0.0013 / 0.038 · 0.44 |
| fit/measured | 0.0349 / 0.668 · 0.30 | 0.0284 / 0.278 · 0.30 | 0.0018 / 0.673 · 0.37 | 0.0023 / 0.058 · 0.35 |
| fit/all | 0.0306 / 0.619 · 0.33 | 0.0248 / 0.260 · 0.34 | 0.0014 / 0.501 · 0.40 | 0.0018 / 0.049 · 0.40 |

Composition chosen: {"model": {"n": 600, "exact": 0.63, "true_in_equivalent_set": 0.6866666666666666}, "measured": {"n": 600, "exact": 0.40166666666666667, "true_in_equivalent_set": 0.4716666666666667}, "all": {"n": 1200, "exact": 0.5158333333333334, "true_in_equivalent_set": 0.5791666666666667}}; median seconds: {"network": 0.7637892864877358, "fit": 14.247374181053601, "choose": 85.72291259851772}.

## Where it does not apply

- cuts spanning less than a factor 10 in q; points below q_max/300 are left out;
- more than three populations, other families, random cylinders larger than R q_max = 50 or h q_max = 200, beam tails wider than 30 q_min or with an exponent outside 1.8–2.2 (encoding 3);
- the network itself knows Gaussian sizes by number and the LMA; other laws and the DA are weighted from its proposals; the four-term DWBA is not in the model;
- its own probabilities are those of the simulated prior; GIMaP reports the importance-weighted ones.
