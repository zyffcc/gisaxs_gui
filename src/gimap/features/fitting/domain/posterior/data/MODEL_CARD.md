# Model card: flow_v1 (GIMaP's proposal network of 1D cuts)

Trained as `flow_v5`, shipped as `flow_v1`.

**What it is.** An amortised posterior p(composition, values | curve) for horizontal GISAXS cuts: a 1D CNN summary, a head over the 258 compositions of one to three populations (sphere, upright or random cylinder, each with or without a paracrystal) and a conditional rational-quadratic spline flow over their values (Physics and Methods, section 9.13). GIMaP uses it only to propose: the values, intervals and probabilities of a solution come from the model and the curve (importance sampling and least squares).

**Files.** `flow_v1.npz` (float32 arrays, SHA-256 `7e72ff728313f04125f95b15d1760ff1ca760416227612384079cd46aacbc6bc`), `.json` (architecture, prior, data, training), this card. Encoding version 5; format `gimap_ml1d_flow_v2`.

**Architecture.** channels [32, 64, 96, 128], kernel 5, dense 512, summary 256, context 256, layers 10, hidden 256, blocks 2, bins 10, bound 6.0, head 256.

**Training data.** 5000000 cuts simulated with cut_model from `prior_v3.json` (seed 777, code 860de8a7): One to three populations (sphere, upright or random cylinder, each with or without a paracrystal), Gaussian sizes by number, LMA; the cut at the effective interface, the Born term at q+ = k_i + k_f, or (both angles known) spheres with the exit-reflected path at q- = k_i - k_f and an effective reflection coefficient r; the bounds of the values are those of posterior/encoding.py (encoding 5: the beam tail's exponent in [1.8, 4]). Every batch draws the populations' shares, the beam tail, the background, the model error, the measured points, Poisson counts and gaps anew.

**Training.** Neural posterior estimation (−log q of the values + cross-entropy of the head), AdamW, batch 2048, learning rate 0.0007 with warm-up and cosine decay, 30 epochs, early stopping on 20000 held-out cuts; best validation loss 22.410965771484374; torch 2.5.1+cu121. Last epoch: {"epoch": 30, "train_nll": 18.9216, "train_ce": 3.4952, "validation_nll": 18.9364, "validation_ce": 3.4746, "validation_loss": 22.411, "validation_composition_top1": 0.2106, "seconds": 6886.7}.

## Calibration on simulated cuts not trained on

2000 cuts, 1000 samples each, the true composition given; composition head top-1 0.196; 257 ms per curve on one thread.

| value | n | 68 % coverage | 95 % coverage | rank χ² (19 dof) | median error |
|---|---|---|---|---|---|
| R | 3981 | 0.680 | 0.946 | 24.0 | 0.200 (relative) |
| sigma_R | 3981 | 0.680 | 0.945 | 32.9 | 0.085 (absolute) |
| h_over_2R | 1015 | 0.676 | 0.943 | 8.5 | 0.605 (relative) |
| sigma_h | 1015 | 0.658 | 0.940 | 16.1 | 0.113 (absolute) |
| D_over_2R | 2764 | 0.688 | 0.946 | 21.7 | 0.347 (relative) |
| sigma_nn | 2764 | 0.678 | 0.945 | 9.6 | 0.051 (absolute) |
| tail_width | 2000 | 0.677 | 0.952 | 12.3 | 0.408 (relative) |
| tail_exponent | 2000 | 0.683 | 0.942 | 23.3 | 0.330 (absolute) |
| q_resolution | 2000 | 0.674 | 0.938 | 15.3 | 0.734 (relative) |
| model_error | 2000 | 0.670 | 0.956 | 18.9 | 0.552 (relative) |
| reflection_re | 2000 | 0.688 | 0.953 | 19.3 | 0.474 (absolute) |
| reflection_im | 2000 | 0.667 | 0.946 | 12.5 | 0.481 (absolute) |

## Where it does not apply

- cuts spanning less than a factor 10 in q; points below q_max/300 are left out;
- more than three populations, other families, random cylinders larger than R q_max = 50 or h q_max = 200, beam tails wider than 30 q_min or with an exponent outside 1.8–4 (encoding 5);
- the network itself knows Gaussian sizes by number and the LMA; other laws and the DA are weighted from its proposals; the four-term DWBA is not in the model (with both angles of the cut, spheres have the direct and the exit-reflected path with a free effective reflection coefficient: two paths); on a cut without a reflected wave a smaller sphere with a large |r| can describe the curve as well;
- its own probabilities are those of the simulated prior; GIMaP reports the importance-weighted ones.
