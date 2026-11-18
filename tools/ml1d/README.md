# The 1D proposal network: simulation, training, checks

GIMaP proposes the parameters of a 1D GISAXS cut with an amortised posterior p(composition, θ | curve): a 1D CNN
summarises the curve, a head gives the probabilities of the compositions, and a conditional normalising flow of
rational-quadratic spline couplings (Durkan et al. 2019) gives the density of the parameters of a composition and
samples from it (neural posterior estimation). The program evaluates the exported network with numpy
(`src/gimap/features/fitting/domain/posterior/`); only training needs PyTorch
(`pip install -r requirements-training.txt`).

## What the network sees and gives

Everything is dimensionless: q̃ = q/q_max and lengths × q_max, so one network serves cuts of any distance and energy.
The curve is binned on 128 log-spaced points from q_max/ρ to q_max (ρ = q_max/q_min in [10, 300]); the network sees
ln I relative to its median, ln(σ/I), the mask of bins with measured points and the position on the grid, and is told
ρ and the cut's q_z/q_max (the Born term at that q_z; 0 for the effective interface) (`posterior/encoding.py`).

A composition is one to three populations listed by increasing radius, each a sphere, an upright cylinder or a random
cylinder, with or without a paracrystal: 258 compositions. A population has R̃, σR/R, h/2R and σh/h (random cylinders),
D/2R and σnn/D (with a paracrystal); the cut has the beam tail's width and exponent, the Gaussian resolution σ_q/q_max
and the amplitude τ of a smooth multiplicative model error. The prior is uniform in a unit box of these values (ln for
lengths, σ_q and τ; the radii ordered by stick-breaking); the flow works on the logit of the box position; values a
composition does not have are never seen or moved by the flow, so the density of the others is exact.

## From samples to solutions (`posterior/infer.py`, `posterior/refine.py`)

For a measured cut and each composition considered (given, or the head's most probable ones), N samples of the
network are weighted by importance sampling against cut_model on the measured points: amplitudes of the populations,
the beam tail and a constant by non-negative least squares and integrated out in the Laplace approximation, the
model error integrated out exactly; weights L·π/q, effective sample size, and the evidence of the composition
(Dax et al. 2023). The weighted samples fall into modes; each mode is refined by Fitting's least squares
(`fit_engine`, method `lm`), or one search per composition starts from the samples (`global`), or the samples alone
give the solution (`posterior`). Every solution carries the refined values and errors, the posterior's median and
68 %/95 % intervals, χ²/N, its probability and flags (at a bound, unconstrained, correlated, few effective samples).

## Files

| File | What it does |
|---|---|
| `simulate.py` | compositions and values from a prior file; every population's term from `cut_model` on 8 points per bin (then averaged, as a measured bin), convolved with the cut's resolution (`resolution.tabulated`); shards of noise-free terms. `augment` draws shares, beam tail, background, model error, measured-point layout (100–1000 points, linear or logarithmic), Poisson counts and gaps anew for every batch |
| `prior_v1.json` | the prior of the first network: K = 1, 2, 3 with weights 0.3, 0.4, 0.3; spheres 0.4, upright cylinders 0.35, random cylinders 0.25; a paracrystal in 70 %; the Born term at a q_z/q_max up to 1 in 60 % |
| `flow_torch.py` | the network in PyTorch, with the names and layout of the numpy version |
| `train_flow.py` | −log q(θ | curve, composition) + cross-entropy of the head; GPU, noise drawn in worker processes; the state of the lowest validation loss is exported as `NAME.npz` and `NAME.json` (architecture, prior, data manifest with the SHA-256 of every shard, history, SHA-256 of the npz) |
| `evaluate_flow.py` | simulation-based calibration on curves not trained on: ranks of the truth, 68 %/95 % coverage per kind of value and K, the head's accuracy, time per curve on one thread |
| `make_flow_fixture.py` | the small random network and PyTorch's numbers with which `tests/test_ml1d_flow.py` checks the numpy flow without PyTorch |
| `build_tables.py` | the size-average tables of `cut_model` |
| `maxwell/` | Slurm scripts of the runs on DESY's Maxwell cluster (array simulation on CPU nodes, training on one GPU) |
| `../benchmarks/posterior_recovery.py` | the recovery benchmark of fit1d (same curves, same scoring) for the network, the refined solutions and the choice of composition |

```bash
python tools/ml1d/simulate.py --prior tools/ml1d/prior_v1.json --count 5000000 --seed 101 --out DATA/train_v1 --workers 64
python tools/ml1d/simulate.py --prior tools/ml1d/prior_v1.json --count 20000 --seed 202 --out DATA/test_v1
python tools/ml1d/train_flow.py --data DATA/train_v1 --prior tools/ml1d/prior_v1.json --out NET/flow_v1 --device cuda --loaders 24
python tools/ml1d/evaluate_flow.py --network NET/flow_v1 --data DATA/test_v1 --prior tools/ml1d/prior_v1.json --out report.json --figure ranks.png
python tools/benchmarks/posterior_recovery.py --network NET/flow_v1 --n 200 --workers 8
```

The seeds fix every shard (`numpy.random.SeedSequence(seed).spawn`); the shards are not kept in the repository.
