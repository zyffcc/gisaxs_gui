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

## Benchmark: posterior_recovery

| stage | R (median / P90 · 68 % coverage) | σR/R | D | σnn/D |
|---|---|---|---|---|
| network/all | 0.0824 / 0.395 | 0.0307 / 0.119 | 0.0069 / 0.090 | 0.0031 / 0.020 |
| fit/model | 0.0055 / 0.168 · 0.73 | 0.0053 / 0.082 · 0.73 | 0.0004 / 0.004 · 0.88 | 0.0005 / 0.005 · 0.87 |
| fit/measured | 0.0069 / 0.229 · 0.62 | 0.0081 / 0.104 · 0.65 | 0.0011 / 0.005 · 0.78 | 0.0016 / 0.006 · 0.63 |
| fit/all | 0.0066 / 0.190 · 0.68 | 0.0070 / 0.093 · 0.69 | 0.0007 / 0.005 · 0.83 | 0.0011 / 0.005 · 0.75 |

Composition chosen: {"model": {"n": 100, "exact": 0.88, "true_in_equivalent_set": 0.97}, "measured": {"n": 100, "exact": 0.6, "true_in_equivalent_set": 0.82}, "all": {"n": 200, "exact": 0.74, "true_in_equivalent_set": 0.895}}; median seconds: {"network": 0.35584187088534236, "fit": 7.332877419190481, "choose": 29.746416351990774}.

## Benchmark: posterior_extended

| stage | R (median / P90 · 68 % coverage) | σR/R | D | σnn/D |
|---|---|---|---|---|
| network/all | 0.1349 / 0.451 | 0.0520 / 0.135 | 0.0277 / 0.731 | 0.0080 / 0.063 |
| fit/model | 0.0131 / 0.290 · 0.83 | 0.0130 / 0.138 · 0.87 | 0.0006 / 0.011 · 0.88 | 0.0008 / 0.010 · 0.88 |
| fit/measured | 0.0214 / 0.459 · 0.75 | 0.0167 / 0.173 · 0.78 | 0.0013 / 0.307 · 0.80 | 0.0017 / 0.019 · 0.77 |
| fit/all | 0.0171 / 0.380 · 0.79 | 0.0142 / 0.158 · 0.82 | 0.0009 / 0.034 · 0.84 | 0.0012 / 0.013 · 0.82 |

Composition chosen: {"model": {"n": 600, "exact": 0.6916666666666667, "true_in_equivalent_set": 0.8033333333333333}, "measured": {"n": 600, "exact": 0.5633333333333334, "true_in_equivalent_set": 0.7133333333333334}, "all": {"n": 1200, "exact": 0.6275, "true_in_equivalent_set": 0.7583333333333333}}; median seconds: {"network": 0.36106953804846853, "fit": 16.237736282404512, "choose": 83.72140427341219}.

## Benchmark: posterior_dwba_si_fit1d

| stage | R (median / P90 · 68 % coverage) | σR/R | D | σnn/D |
|---|---|---|---|---|
| network/all | 0.0489 / 0.192 | 0.0317 / 0.114 | 0.0051 / 0.025 | 0.0025 / 0.010 |
| fit/model | 0.0206 / 0.170 · 0.69 | 0.0176 / 0.105 · 0.71 | 0.0004 / 0.002 · 0.96 | 0.0005 / 0.003 · 0.96 |
| fit/measured | 0.0309 / 0.204 · 0.67 | 0.0186 / 0.116 · 0.74 | 0.0009 / 0.004 · 0.85 | 0.0016 / 0.004 · 0.70 |
| fit/all | 0.0282 / 0.178 · 0.68 | 0.0179 / 0.114 · 0.73 | 0.0005 / 0.003 · 0.90 | 0.0009 / 0.004 · 0.83 |

Composition chosen: {"model": {"n": 100, "exact": 0.76, "true_in_equivalent_set": 0.99}, "measured": {"n": 100, "exact": 0.57, "true_in_equivalent_set": 0.82}, "all": {"n": 200, "exact": 0.665, "true_in_equivalent_set": 0.905}}; median seconds: {"network": 0.3607358889421448, "fit": 20.635519499424845, "choose": 106.9544027770171}.

## Benchmark: posterior_dwba_si_extended

| stage | R (median / P90 · 68 % coverage) | σR/R | D | σnn/D |
|---|---|---|---|---|
| network/all | 0.1112 / 0.456 | 0.0515 / 0.144 | 0.0211 / 0.634 | 0.0063 / 0.052 |
| fit/model | 0.0354 / 0.331 · 0.79 | 0.0243 / 0.160 · 0.84 | 0.0006 / 0.010 · 0.88 | 0.0008 / 0.009 · 0.89 |
| fit/measured | 0.0480 / 0.421 · 0.73 | 0.0319 / 0.184 · 0.78 | 0.0012 / 0.052 · 0.83 | 0.0018 / 0.016 · 0.80 |
| fit/all | 0.0402 / 0.373 · 0.76 | 0.0273 / 0.173 · 0.81 | 0.0009 / 0.018 · 0.86 | 0.0012 / 0.012 · 0.85 |

Composition chosen: {"model": {"n": 600, "exact": 0.5966666666666667, "true_in_equivalent_set": 0.8083333333333333}, "measured": {"n": 600, "exact": 0.5066666666666667, "true_in_equivalent_set": 0.7116666666666667}, "all": {"n": 1200, "exact": 0.5516666666666666, "true_in_equivalent_set": 0.76}}; median seconds: {"network": 0.3528274989221245, "fit": 39.12034843652509, "choose": 239.33043969795108}.

## Where it does not apply

- cuts spanning less than a factor 10 in q; points below q_max/300 are left out;
- more than three populations, other families, random cylinders larger than R q_max = 50 or h q_max = 200, beam tails wider than 30 q_min or with an exponent outside 1.8–2.2 (encoding 4);
- the network itself knows Gaussian sizes by number and the LMA; other laws and the DA are weighted from its proposals; the four-term DWBA is not in the model (with both angles of the cut, spheres have the direct and the exit-reflected path with a free effective reflection coefficient: two paths); on a cut without a reflected wave a smaller sphere with a large |r| can describe the curve as well;
- its own probabilities are those of the simulated prior; GIMaP reports the importance-weighted ones.
