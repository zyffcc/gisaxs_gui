# Sputter structure toolkit (bundled research code)

From an in-situ sputter GISAXS scan (a folder of Pilatus CBF frames) on a polymer substrate, this toolkit
gives, per time window, the particle radius R and spacing D of the islands, the lateral texture length L of
late film-like windows, and how reliable each value is. GIMaP's **Sputter** workspace runs it as a
subprocess; GIMaP itself never imports it.

It is research code, bundled unchanged except for the local patches listed in `SOURCE.json` (where it came
from, the original checkpoint and policy SHA-256, and why each patch was made). `python verify.py` checks
every file against `MANIFEST.json`. The two network checkpoints refuse to load when one of the five source
files they are bound to changes (`neural_refine_posterior.py`, `neural_refine_training.py`,
`neural_refine_chart.py`, `v2_posterior.py`, `neural_refine_input.py`), so those files are never edited.

## Requirements

numpy, scipy, matplotlib, Pillow, fabio and PyTorch (CPU is enough), see `requirements.txt`. NumPy 1.24 to 2.x
and PyTorch 2.3 or later work. GIMaP adds PyTorch to its environment when it is missing.

## Command line

```
python scripts/sputter_struct_v1.py analyze --input SCAN_FOLDER --out OUT --energy-ev 12000 --sdd-mm 2500 \
    --alpha-i-deg 0.4 --metal Ag --polymer-expert experts/polymer --ts-expert experts/TS \
    [--beam-col C --beam-row R] [--sample-roll-deg 0] [--mask MASK.npy] [--workers 4] \
    [--rate-nm-s RATE --t0-s T0]
python scripts/sputter_struct_v1.py physical --folder OUT [--rate-nm-s RATE] [--t0-s T0]
python scripts/sputter_struct_v1.py table --binned OUT/binned --out OUT --metal Ag [--refined REFINED/windows]
python scripts/sputter_struct_v1.py select|sequence --binned OUT/binned --out OUT --metal Ag \
    --polymer-expert experts/polymer --ts-expert experts/TS
```

`analyze` reads the frames (raw data are only read), sums the exact counts into time windows (0.1 s for the
first 5 s, then 0.5 s), measures the direct features (peak position q*, width, Yoneda position, high-q slope),
applies the observation rule for the growth stage and runs both networks on every window. A rerun continues
from the steps already done. Each worker needs about 0.3 GB of memory.

Progress lines for the GUI: `SPUTTER_PROGRESS<TAB>step<TAB>fraction<TAB>message`, `INTAKE_PLAN <chunks> <frames>`,
`INTAKE_CHUNK <name> <index> <frames> <seconds>` and `INTAKE_EXCLUDED <frame> <reason>`.

A truncated or undecodable CBF frame is left out explicitly: it is listed with its reason in
`OUT/intake/excluded_frames.json` and `intake/run.json` (`analysed_frames = selected_frames - excluded`), and
the windows count only the frames read. No pixel is invented.

`--metal Cu` is accepted: Cu is approximated with the Ag model, for reference only (errors not validated).

## Outputs

| File | Content |
|---|---|
| `structure.csv` | one row per window: the per-window reading (rule stage, the network it routes to, R/D from converged refinement when merged, else the network) |
| `structure_physical.csv` | the physical path: every column of `structure.csv`, plus `stage_path`, `module_path`, `film`, `path_changed`, `R_trend_nm`, `D_trend_nm`, `t_eq_nm`, `t_eq_lo_nm`, `t_eq_hi_nm`, `nominal_thickness_nm`, `film_thickness_status`, `stage_imputed` |
| `structure_summary.json`, `structure_physical_summary.json` | counts, window-to-window scatter, reliability numbers; the physical one also the TS onset, modulation (rotation) period, reversal counts before/after and the thickness notes |
| `binned/frames.csv`, `binned/frames_alt.csv` | all network modes per window, from the routed network and from the other one |
| `structure_selected.csv`, `structure_sequence.csv` | learned selector and time-consistent Viterbi path (exploratory, see below) |

### The physical path (`structure_physical.csv`)

The per-window rule decides every window on its own, so stage, network and R/D source can switch back and forth.
The physical path reads the whole series once:

- **Stage**: the rule stages on the scale pre_nucleation < islands < coalescence < film, fitted non-decreasing
  (pool-adjacent-violators with block medians). Windows whose stage is unresolved cast no vote and take the
  value of the previous resolved window (`stage_imputed` = true; `stage_path_counts_imputed` counts them).
  Windows with fewer than 1000 counts (shutter closed, beam off) get no stage and no values. Path stage `film`
  is the rule's coalescence-or-film candidate: a connected film is not proven; `film` = true marks every window
  on the TS network (coalescence or film). `film_onset_s` is the first window on the path that the toolkit
  routes to the TS network; `film_onset_interval_s` spans every onset that disagrees with at most two more
  resolved windows, whether or not the path has an onset, and `film_onset_no_TS_within_slack` says when "no TS
  at all" is within that slack too (the onset is then not established). Model assumption: during deposition
  the film does not turn back into islands.
- **Values**: R and D are the network estimate (grade **P**) in every window whose path stage is islands, from
  the island network; pre-nucleation windows have none. Refined values stay in `refined_R_nm`/`refined_D_nm`.
  `R_trend_nm`/`D_trend_nm` are a local log-linear (Theil-Sen) fit of ln R / ln D against time over the island
  windows in a centred span of max(modulation period, 2.5 s), evaluated at each window: they follow growth and
  are not biased at the ends of the island phase. `reversals.after` counts zig-zags of the per-window values on
  the path, `reversals.after_trend` those of the trend lines.
- **Rotation**: a rotating sample modulates the intensity. `rotation_period_s` is the period of that modulation
  (the spacing of its dips): from the autocorrelation of the detrended log intensity (0.5 s windows after 5 s,
  60 s local linear trend), the shortest maximum (>= 0.1) at 2-60 s lag whose double lag is also a maximum,
  with >= 0.3 at one of the two and at least 3 full cycles. When the double lag is clearly stronger, the dips
  have unequal depths and the pattern repeats exactly only after `full_repeat_s`, which may be one turn of the
  sample. `rotation_detection` lists all maxima and the strongest periodogram line as a cross-check.
- **Thickness substitutes** (GISAXS does not measure the metal thickness here):
  - `t_eq_nm` = (2/3)·π·R³·exp(3σ²) / ((√3/2)·D²): island-volume equivalent thickness of the *visible*
    particles, conditional on flat-based hemispheres (h=R) on a hexagonal lattice; not the deposited amount.
    The band divides and multiplies by exp(√(9·eR² + 4·eD²)), eR and eD = ln(1 + typical relative error).
  - `nominal_thickness_nm` = rate·(t − t0) when a deposition rate is given (e.g. from a QCM); not measured by
    GISAXS. Without `--t0-s`, t0 = 0 s (the first frame) is an assumption and the summary says so.
    `mass_continuity` compares both just before the TS onset.
  - Film windows: the TS film thickness and metal fraction are not identifiable (in simulation the network's
    posterior equals its prior), so `film_thickness_status` says so and no number is given; `L_TS_nm` is the
    lateral texture length, not an island spacing.

## How reliable the values are

Simulation test of 200 new independent windows (100 Ag, 100 Au), at the counts of a 0.5 s window
(`configs/sputter_struct_reliability_v1.json`):

| | R typical error | D typical error | order correct when two windows differ by ≥1.2× |
|---|---|---|---|
| network (grades C, P) | Ag ±31 %, Au ±26 % | Ag ±26 %, Au ±24 % | 90–92 % |
| converged refinement (A, B) | Ag ±16 %, Au ±10 % | Ag ±12 %, Au ±5 % | 92–96 % |

- About 10–20 % of the values are more than 50 % off (P90 50–65 %): another structure fits the counts equally well.
  Trends are more reliable than absolute values.
- R is the base radius of an equivalent flat-based hemisphere (h=R); real clusters on polymer are often
  truncated spheres with h/R between 1 and 2.
- D is given only for windows with a resolved correlation peak; `D_corr_nm` = 2π/q* is the direct observation.
  A D outside the simulated 90 % range of D/(2π/q*) is flagged in `D_direct_check`.
- Burial depth and polymer thickness are not reliable (conditional values only); the TS metal fraction and film
  thickness are not identifiable.
- The selector (`select`) and the time-consistent path (`sequence`) did not pass their registered simulation
  tests for R; they are offered as exploratory views.

Physical refinement (grades A/B) needs a GPU and is not part of this bundle; `table --refined` merges refined
windows made elsewhere.

## Files

`scripts/sputter_struct_v1.py` (entry), `sputtergisaxs/` (the import closure of the five subcommands;
`sputter_struct.py` pipeline and table, `sputter_physical.py` physical path), `experts/{polymer,TS}`
(checkpoints and input policies), `configs/` (reliability numbers, selector model), `MANIFEST.json`,
`verify.py`, `SOURCE.json`, `requirements.txt`.
