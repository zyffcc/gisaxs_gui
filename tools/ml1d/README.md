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
ρ, the cut's q₊/q_max = (k_i + k_f)/q_max, q₋/q_max = (k_i − k_f)/q_max and whether the cut has two paths
(`posterior/encoding.py`, encoding 6): with both angles known, spheres have the direct and the exit-reflected path with
an effective reflection coefficient r (`cut_model.geometry`, `two_path`), whose real and imaginary parts are two more
values of the cut; with q₊ alone the Born term at q₊; with neither the effective interface.

A composition is one to three populations listed by increasing radius, each a sphere, an upright cylinder or a random
cylinder, with or without a paracrystal: 258 compositions. A population has R̃, σR/R, h/2R and σh/h (random cylinders),
D/2R and σnn/D (with a paracrystal); the cut has the beam tail's width (its exponent is 2, the Lorentzian wings of the
resolution function), the width 1/ξ and the Hurst exponent H of the diffuse term of a rough interface, the Gaussian
resolution σ_q/q_max
and the amplitude τ of a smooth multiplicative model error. The prior is uniform in a unit box of these values (ln for
lengths, σ_q and τ; the radii ordered by stick-breaking); the flow works on the logit of the box position; values a
composition does not have are never seen or moved by the flow, so the density of the others is exact.

## From samples to solutions (`posterior/infer.py`, `posterior/refine.py`)

`refine.predict` is the one path that Predict parameters takes (Fitting, the 1D Predict window, Analyze's automatic
analysis, Batch Export and the command line all call it; there are no options for how the solutions are finished).
For a measured cut and each composition considered (given, or the head's four most probable among those allowed: `refine.SHORTLIST`),
1000 samples of the network are weighted by importance sampling against cut_model on the measured points: amplitudes
of the populations, the beam tail, the diffuse term, the Gaussian core of the beam (width σ_q) and a constant by
non-negative least squares and integrated out in the Laplace
approximation, the model error integrated out exactly; weights L·π/q, effective sample size, and the evidence of the
composition (Dax et al. 2023). The weighted samples fall into modes; the three heaviest modes of each composition are
refined by Fitting's bounded least squares (`fit_engine`, method `local`). When no solution of one of the two best
compositions refines below χ²/N = 2, one search of the ranges started from its 16 heaviest samples is added. Only the
solutions within a BIC of 20 of the best (at most four) get their posterior sampled locally (`posterior/laplace.py`,
400 samples per round); with fewer than 50 effective samples their intervals are taken with tempered weights, and when
any of them has fewer than 10 the solutions are compared by BIC instead of their evidence, with the least-squares
errors as intervals. Without σ, a relative σ is taken from the scatter of ln I about its local trend (at least 1 %).
Every solution carries the refined values and errors, the posterior's median and 68 %/95 % intervals, χ²/N, its
probability and flags (at a bound, unconstrained, correlated, few effective samples, compared by BIC).

## Files

| File | What it does |
|---|---|
| `simulate.py` | compositions and values from a prior file; every population's term from `cut_model` on 8 points per bin (then averaged, as a measured bin), convolved with the cut's resolution (`resolution.tabulated`); shards of noise-free terms. `augment` draws shares, the amplitudes of the beam tail and the diffuse term, background, model error, measured-point layout (100–1000 points, linear or logarithmic), Poisson counts and gaps anew for every batch |
| `prior_v4.json` | the prior of the network that comes with GIMaP (encoding 6): as `prior_v3.json`, with the beam tail (ν = 2) and the diffuse term (1/ξ from 0.03 to 3 q_min, H from 0 to 1) in place of one tail, each with an amplitude of 10⁻³–10 times the particles at q_min |
| `prior_v3.json` | the prior of `flow_v5` (encoding 5): as `prior_v2.json`, with the beam tail's exponent ν in [1.8, 4] |
| `prior_v2.json` | the prior of the network before it (encoding 4): as `prior_v1.json`, and 70 % of the cuts at a q₊ have two paths, q₋ = q₊ × U(−0.5, 0.9), r uniform in [−1, 1]²; ν in [1.8, 2.2] |
| `prior_v1.json` | the prior of the first network: K = 1, 2, 3 with weights 0.3, 0.4, 0.3; spheres 0.4, upright cylinders 0.35, random cylinders 0.25; a paracrystal in 70 %; the Born term at a q_z/q_max up to 1 in 60 % |
| `flow_torch.py` | the network in PyTorch, with the names and layout of the numpy version |
| `train_flow.py` | −log q(θ | curve, composition) + cross-entropy of the head; GPU, noise drawn in worker processes; the state of the lowest validation loss is exported as `NAME.npz` and `NAME.json` (architecture, prior, data manifest with the SHA-256 of every shard, history, SHA-256 of the npz) |
| `export_checkpoint.py` | the best state so far of a run (its `NAME.pt`) as a numpy network, while it still trains |
| `model_card.py` | `MODEL_CARD.md` of an exported network from its record, `evaluate_flow.py`'s report and benchmark summaries |
| `evaluate_flow.py` | simulation-based calibration on curves not trained on: ranks of the truth, 68 %/95 % coverage per kind of value and K, the head's accuracy, time per curve on one thread |
| `make_flow_fixture.py` | the small random network and PyTorch's numbers with which `tests/test_ml1d_flow.py` checks the numpy flow without PyTorch |
| `build_tables.py` | the size-average tables of `cut_model` |
| `maxwell/` | Slurm scripts of the runs on DESY's Maxwell cluster (array simulation on CPU nodes, training on one GPU) |
| `../benchmarks/posterior_recovery.py` | the recovery benchmark of fit1d (removed in 0.2.0; same curves of `../benchmarks/recovery_cases.py`, same scoring) for the network, the refined solutions and the choice of composition |

## The network that comes with GIMaP

`src/gimap/features/fitting/domain/posterior/data/flow_v1.npz` was trained as `flow_v6` (encoding 6: the beam tail
with the exponent 2 and the diffuse term of a rough interface) at commit e2161c2c, on one V100 in 1.9 h, with:

```bash
python tools/ml1d/simulate.py --prior tools/ml1d/prior_v4.json --count 5000000 --seed 777 --out DATA/train_v6 --shard 10000 --workers 64
python tools/ml1d/simulate.py --prior tools/ml1d/prior_v4.json --count 20000 --seed 888 --out DATA/test_v6 --shard 10000
python tools/ml1d/train_flow.py --data DATA/train_v6 --prior tools/ml1d/prior_v4.json --out NET/flow_v6 --device cuda --loaders 24 --epochs 30 --batch 2048 --lr 7e-4 --validation 20000 --patience 6
python tools/ml1d/evaluate_flow.py --network NET/flow_v6 --data DATA/test_v6 --prior tools/ml1d/prior_v4.json --curves 2000 --samples 1000 --out report.json --figure ranks.png
python tools/ml1d/model_card.py --network NET/flow_v6 --evaluation report.json --out MODEL_CARD.md
```

(On Maxwell `maxwell/simulate.sbatch` ran the simulation as an array of tasks with `--first-shard/--shards`, joined
with `--join`, and `maxwell/train.sbatch` the training.) Its record names the
SHA-256 of the shards; the shipped record leaves out the paths of the computer it was trained on. On 2000 simulated
cuts not trained on, its 68 %/95 % intervals hold the truth in 66–69 %/94–95 % for every value (the head ranks the
true composition first in 22 %); the benchmarks are in Physics and Methods, section 9.13, and `MODEL_CARD.md`, whose benchmark
sections `tools/benchmarks/summary_tables.py --only card` prints from the final runs of `tools/benchmarks/`. The
networks shipped before it were trained the same way: `flow_v5` (encoding 5: one tail with the exponent ν from 1.8
to 4) from `prior_v3.json` (seeds 777 and 888) at commit 860de8a7, `flow_v4` (encoding 4: the beam tail's ν from 1.8 to 2.2)
from `prior_v2.json` (seeds 555 and 666) at commit a10d26d, and `flow_v3` (encoding 3: the Born term at q_z, no
reflected path) from `prior_v1.json` (seeds 333 and 444) at commit e5a609e.

The seeds fix every shard (`numpy.random.SeedSequence(seed).spawn`); the shards are not kept in the repository.
