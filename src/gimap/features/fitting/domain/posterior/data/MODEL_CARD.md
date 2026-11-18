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
| network/all | 0.0552 / 0.366 | 0.0302 / 0.128 | 0.0055 / 0.099 | 0.0030 / 0.019 |
| fit/model | 0.0062 / 0.095 · 0.72 | 0.0056 / 0.074 · 0.76 | 0.0004 / 0.003 · 0.91 | 0.0005 / 0.004 · 0.91 |
| fit/measured | 0.0082 / 0.184 · 0.60 | 0.0086 / 0.100 · 0.64 | 0.0011 / 0.009 · 0.75 | 0.0016 / 0.007 · 0.61 |
| fit/all | 0.0071 / 0.142 · 0.66 | 0.0069 / 0.087 · 0.70 | 0.0006 / 0.005 · 0.83 | 0.0009 / 0.005 · 0.76 |

Composition chosen: {"model": {"n": 100, "exact": 0.87, "true_in_equivalent_set": 0.93}, "measured": {"n": 100, "exact": 0.68, "true_in_equivalent_set": 0.81}, "all": {"n": 200, "exact": 0.775, "true_in_equivalent_set": 0.87}}; median seconds: {"network": 0.16645509097725153, "fit": 2.994166672346182, "choose": 10.113789149560034}.

## Benchmark: posterior_extended

| stage | R (median / P90 · 68 % coverage) | σR/R | D | σnn/D |
|---|---|---|---|---|
| network/all | 0.1266 / 0.468 | 0.0515 / 0.139 | 0.0300 / 0.745 | 0.0084 / 0.060 |
| fit/model | 0.0171 / 0.435 · 0.78 | 0.0140 / 0.151 · 0.82 | 0.0007 / 0.024 · 0.85 | 0.0009 / 0.016 · 0.84 |
| fit/measured | 0.0259 / 0.551 · 0.70 | 0.0205 / 0.186 · 0.74 | 0.0014 / 0.529 · 0.74 | 0.0019 / 0.056 · 0.71 |
| fit/all | 0.0211 / 0.480 · 0.74 | 0.0171 / 0.172 · 0.78 | 0.0010 / 0.394 · 0.79 | 0.0013 / 0.034 · 0.78 |

Composition chosen: {"model": {"n": 600, "exact": 0.58, "true_in_equivalent_set": 0.6533333333333333}, "measured": {"n": 600, "exact": 0.445, "true_in_equivalent_set": 0.5516666666666666}, "all": {"n": 1200, "exact": 0.5125, "true_in_equivalent_set": 0.6025}}; median seconds: {"network": 0.1687047219602391, "fit": 6.839433523011394, "choose": 28.52268945099786}.

## Where it does not apply

- cuts spanning less than a factor 10 in q; points below q_max/300 are left out;
- more than three populations, other families, random cylinders larger than R q_max = 50 or h q_max = 200, beam tails wider than 30 q_min or with an exponent outside 1.8–2.2 (encoding 3);
- the network itself knows Gaussian sizes by number and the LMA; other laws and the DA are weighted from its proposals; the four-term DWBA is not in the model;
- its own probabilities are those of the simulated prior; GIMaP reports the importance-weighted ones.
