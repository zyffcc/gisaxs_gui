# The 1D proposal network: simulation, training, checks

GIMaP proposes the parameters of a 1D GISAXS cut with an amortised posterior p(θ | curve, conditions): a small 1D CNN
summarises the curve, a conditional normalising flow of rational-quadratic spline couplings (Durkan et al. 2019)
gives the density of the parameters and samples from it (neural posterior estimation). The program evaluates the
exported network with numpy (`src/gimap/features/fitting/domain/posterior/flow.py`); only training needs PyTorch
(`pip install -r requirements-training.txt`).

Everything is dimensionless: q̃ = q/q_max and lengths × q_max, so one network serves cuts of any distance and energy.
The curve is binned on 128 log-spaced points from q_max/ρ to q_max (ρ = q_max/q_min in [10, 300]), and the network
sees ln I relative to its median, ln(σ/I), the mask of measured points and the position on the grid
(`posterior/encoding.py`). Its conditions are the particle family, whether there is a structure factor, ρ, the cut's
q_z/q_max (Born term; 0 for the effective interface) and σ_q/q_max of the resolution. The parameters of one population
and the beam tail are ln R̃, σR/R, ln(D/2R), σnn/D, ln w̃ and ν; the flow works on the logit of each one's position in
the bounds of the prior (`encoding.bounds`).

| File | What it does |
|---|---|
| `simulate.py` | draws parameters from a prior file and writes shards of noise-free particle terms T(q̃) and beam-tail shapes from `cut_model`, the forward model Fitting uses; `augment` adds scale, tail, background, Poisson counts and gaps anew for every batch |
| `prior_k1.json` | one population, sphere or upright cylinder, Gaussian sizes by number, LMA, a paracrystal in 3 of 4 cuts, Born q_z in half the spheres, a resolution in 30 % |
| `flow_torch.py` | the network in PyTorch, with the same names and layout as the numpy version |
| `train_flow.py` | maximum likelihood of the true parameters; the lowest validation loss is exported as `NAME.npz` and `NAME.json` (architecture, prior, data manifest with the SHA-256 of every shard, history, SHA-256 of the npz) |
| `evaluate_flow.py` | simulation-based calibration on curves not trained on: ranks of the truth among posterior samples, 68 % and 95 % coverage, errors of the posterior median, time per curve on one CPU thread |
| `make_flow_fixture.py` | the small random network and PyTorch's numbers with which `tests/test_ml1d_flow.py` checks the numpy flow without PyTorch |
| `build_tables.py` | the size-average tables of `cut_model` |

```bash
python tools/ml1d/simulate.py --prior tools/ml1d/prior_k1.json --count 200000 --seed 1 --out DATA/train_k1 --workers 2
python tools/ml1d/simulate.py --prior tools/ml1d/prior_k1.json --count 5000 --seed 2 --out DATA/test_k1
python tools/ml1d/train_flow.py --data DATA/train_k1 --prior tools/ml1d/prior_k1.json --out NET/k1_v1 --epochs 30
python tools/ml1d/evaluate_flow.py --network NET/k1_v1 --data DATA/test_k1 --prior tools/ml1d/prior_k1.json --out report.json --figure ranks.png
```

Inactive parameters (D and σnn/D without a structure factor) are trained as standard normal noise, so the flow's
density of the active ones is its density divided by that of the noise. The seeds fix every shard
(`numpy.random.SeedSequence(seed).spawn`); the shards themselves are not kept in the repository.

## First network: k1_v1 (development, not shipped)

200 000 cuts of `prior_k1.json` (seed 1), 30 epochs on 4 CPU threads (46 min), validation loss 4.85 nats (11.2 after
the first epoch, still falling at the last). On 2000 held-out cuts (seed 2) with 1000 samples each
(`evaluation_k1_v1.json`):

| parameter | curves | 68 % coverage | 95 % coverage | rank χ² (19 dof) | median abs. error of the posterior median |
|---|---|---|---|---|---|
| ln R̃ | 2000 | 0.693 | 0.946 | 12.8 | 0.155 (R 16 %: spheres 24 %, upright cylinders 10 %) |
| σR/R | 2000 | 0.670 | 0.947 | 28.5 | 0.068 |
| ln(D/2R) | 1510 | 0.665 | 0.944 | 12.0 | 0.236 (D itself 6.6 %) |
| σnn/D | 1510 | 0.697 | 0.948 | 16.7 | 0.019 |
| ln w̃ | 2000 | 0.662 | 0.940 | 21.9 | 0.54 |
| ν | 2000 | 0.678 | 0.951 | 22.4 | 0.46 |

The posterior is calibrated in the model (coverages within 0.04 of nominal, ranks flat). The errors are those of the
amortised proposal over the whole prior, before any refinement, and include curves whose particle term is hidden by
the beam tail or the background; they are not comparable with fit1d's recovery benchmark (section 9.12 of Physics and
Methods), which the next milestone measures. Context and 1000 samples take 58 ms per curve on one thread. The weights
are not kept in the repository (SHA-256 in `evaluation_k1_v1.json`; the seeds and settings above train an
equivalent network, not necessarily bit for bit on another computer). Known for the next prior:
the background is drawn relative to T at q_max and can swamp a whole curve, and the beam tail (width up to q_max/3,
amplitude up to 100 × T at q_min) hides the particle term in some curves.
