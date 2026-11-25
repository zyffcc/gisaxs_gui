# Model card: flow_v1 (GIMaP's proposal network of 1D cuts)

Trained as `flow_v6`, shipped as `flow_v1`.

**What it is.** An amortised posterior p(composition, values | curve) for horizontal GISAXS cuts: a 1D CNN summary, a head over the 258 compositions of one to three populations (sphere, upright or random cylinder, each with or without a paracrystal) and a conditional rational-quadratic spline flow over their values (Physics and Methods, section 9.13). GIMaP uses it only to propose: the values, intervals and probabilities of a solution come from the model and the curve (importance sampling and least squares).

**Files.** `flow_v1.npz` (float32 arrays, SHA-256 `857e867db83a4e727ca4fe23f6133efe038b708f2de5763cd64e9779d486814a`), `.json` (architecture, prior, data, training), this card. Encoding version 6; format `gimap_ml1d_flow_v2`.

**Architecture.** channels [32, 64, 96, 128], kernel 5, dense 512, summary 256, context 256, layers 10, hidden 256, blocks 2, bins 10, bound 6.0, head 256.

**Training data.** 5000000 cuts simulated with cut_model from `prior_v4.json` (seed 777, code e2161c2c): As posterior_v3, with two terms at low q in place of one tail (encoding 6): the beam tail with the exponent 2, the Lorentzian wings 1/(1 + (q/w)^2) of the resolution function, and the diffuse term (1 + (q xi)^2)^-(1+H) of a rough interface; each with an amplitude of 0.001 to 10 times the particles at q_min; the bounds of the values are those of posterior/encoding.py. Every batch draws the populations' shares, the amplitudes of the beam tail and the diffuse term, the background, the model error, the measured points, Poisson counts and gaps anew.

**Training.** Neural posterior estimation (−log q of the values + cross-entropy of the head), AdamW, batch 2048, learning rate 0.0007 with warm-up and cosine decay, 30 epochs, early stopping on 20000 held-out cuts; best validation loss 25.18726025390625; torch 2.5.1+cu121. Last epoch: {"epoch": 30, "train_nll": 21.6796, "train_ce": 3.4407, "validation_nll": 21.7497, "validation_ce": 3.4376, "validation_loss": 25.1873, "validation_composition_top1": 0.2143, "seconds": 6738.4}.

## Calibration on simulated cuts not trained on

2000 cuts, 1000 samples each, the true composition given; composition head top-1 0.220; 173 ms per curve on one thread.

| value | n | 68 % coverage | 95 % coverage | rank χ² (19 dof) | median error |
|---|---|---|---|---|---|
| R | 3981 | 0.680 | 0.946 | 30.7 | 0.184 (relative) |
| sigma_R | 3981 | 0.657 | 0.944 | 28.2 | 0.087 (absolute) |
| h_over_2R | 1015 | 0.674 | 0.950 | 18.5 | 0.587 (relative) |
| sigma_h | 1015 | 0.677 | 0.936 | 17.5 | 0.112 (absolute) |
| D_over_2R | 2764 | 0.671 | 0.946 | 17.7 | 0.353 (relative) |
| sigma_nn | 2764 | 0.670 | 0.944 | 21.3 | 0.049 (absolute) |
| tail_width | 2000 | 0.693 | 0.947 | 20.0 | 0.660 (relative) |
| diffuse_width | 2000 | 0.655 | 0.943 | 20.1 | 0.753 (relative) |
| diffuse_hurst | 2000 | 0.674 | 0.939 | 25.0 | 0.217 (absolute) |
| q_resolution | 2000 | 0.676 | 0.939 | 22.9 | 0.697 (relative) |
| model_error | 2000 | 0.681 | 0.943 | 19.8 | 0.600 (relative) |
| reflection_re | 2000 | 0.678 | 0.947 | 13.7 | 0.469 (absolute) |
| reflection_im | 2000 | 0.692 | 0.952 | 26.9 | 0.453 (absolute) |

## Where it does not apply

- cuts spanning less than a factor 10 in q; points below q_max/300 are left out;
- more than three populations, other families, random cylinders larger than R q_max = 50 or h q_max = 200, beam tails wider than 1 q_min or falling other than as q⁻², diffuse terms with 1/ξ above 3 q_min (encoding 6);
- the network itself knows Gaussian sizes by number and the LMA; other laws and the DA are weighted from its proposals; the four-term DWBA is not in the model (with both angles of the cut, spheres have the direct and the exit-reflected path with a free effective reflection coefficient: two paths); on a cut without a reflected wave a smaller sphere with a large |r| can describe the curve as well;
- its own probabilities are those of the simulated prior; GIMaP reports the importance-weighted ones.
