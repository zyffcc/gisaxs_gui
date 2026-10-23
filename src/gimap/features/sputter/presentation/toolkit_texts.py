"""The bundled toolkit's own sentences as templates, so that the page says them in its language.

The toolkit (``utils/sputter_toolkit``, run as a subprocess) writes its notes, reasons and literature checks
in English from fixed templates (``sputter_physical.NOTE_TEMPLATES``, ``sputter_literature.TEMPLATES``) with
plain ``{field}`` values. ``texts.note_text`` matches a sentence of the toolkit against this copy and fills the
same values into the translated template. ``tests/test_sputter_feature.py`` checks that the copy follows the
bundled toolkit.
"""

TOOLKIT_TEMPLATES = (
    ('t_eq = (2/3)*pi*R^3*exp(3*sigma^2) / ((sqrt(3)/2)*D^2): island-volume equivalent thickness of the '
     'visible particles, conditional on flat-based hemispheres (h=R) on a hexagonal lattice; not the '
     'deposited amount. Band t_eq/f..t_eq*f with f = exp(sqrt(9*eR^2 + 4*eD^2)), eR and eD = ln(1 + '
     'typical relative error) of the network R and D.'),
    ('nominal deposited thickness = rate * (t - t0) from the given deposition rate (e.g. QCM); not '
     'measured by GISAXS.'),
    ('nominal deposited thickness = rate * (t - t0) from the given deposition rate (e.g. QCM); not '
     'measured by GISAXS. t0 not given: assumed 0 s = the first frame; the start of the deposition is '
     'not known from the data (each 1 s of error in t0 shifts the nominal thickness by {rate} nm).'),
    'no deposition rate given: nominal thickness not computed',
    ('not determined by GISAXS: the TS film thickness and metal fraction are not identifiable (in '
     "simulation the network's posterior equals its prior); see L_TS_nm for the lateral texture length"),
    ('first window of the TS-network part of the path (rule stage coalescence or coalescence-or-film '
     'candidate); a connected film is not proven'),
    ('the network of the path module was not run on this window in this folder (binned/frames_alt.csv '
     'missing; run analyze again on this folder to add it)'),
    'film-type (TS layer) window on the physical path: island R/D not defined (see L_TS_nm)',
    'pre-nucleation on the physical path: no island estimate',
    'no window of the series has a resolved stage: no island estimate',
    'fewer than 1000 counts in the window (shutter closed or beam off): no stage, no values',
    'no resolved peak: D not reported',
    'no estimate',
    'outside simulated 90% range of D/(2pi/q*)',
    ('Model assumption: during deposition the TS onset is monotone (no return from film to islands); '
     'the stage path is the non-decreasing L1 fit of the rule stages, and unresolved windows cast no '
     'vote.'),
    ("stage_path 'film' is the rule's coalescence-or-film candidate (a connected film is not proven); "
     'film=true marks every window on the TS network (coalescence or film).'),
    ('R_trend_nm / D_trend_nm: local log-linear (Theil-Sen) fit of ln R / ln D over the island windows '
     'in a centred span of {span} s (max of the modulation period and 2.5 s), evaluated at each window.'),
    ('R_nm / D_nm are network estimates in every window whose path stage is islands (grade P); refined '
     'values stay in refined_R_nm / refined_D_nm. Pre-nucleation windows have no island values.'),
    ('TS film windows: no thickness or metal fraction is reported (not identifiable); L_TS_nm is the '
     'lateral texture length, not an island spacing.'),
    'Cu approximated with the Ag model, for reference only (errors not validated).',
    ('{count} windows with an unresolved rule stage carry the stage of the neighbouring resolved window '
     '(stage_imputed=true; stage_path_counts_imputed).'),
    '{count} windows with fewer than {counts} counts have no stage and no values.',
    ('TS onset not established: no TS window at all disagrees with {no_ts} resolved windows, within '
     '{slack} of the {best} at the chosen onset ({onset} s); the series may stay in the island phase.'),
    ('No TS onset on the path, but onsets from {start} s disagree with at most {most} resolved windows: '
     'a late TS onset is not excluded.'),
    ('The path reaches coalescence but never the film stage: film_onset_s is the onset of TS-network '
     '(coalescence) windows, not of a continuous film.'),
    ('The intensity dips every {period} s with unequal depths; the pattern repeats exactly after '
     '{repeat} s, which may be one turn of the sample.'),
    ('The smoothing span ({span} s) covers the whole island phase ({start}-{end} s): R_trend_nm / '
     'D_trend_nm are one log-linear fit over it; the per-window R_nm / D_nm (grade P) show more.'),
    ('needs a deposition rate: the literature method computes R from the nominal thickness delta = '
     'rate*(t - t0); without it only the monotony of D is tested (in time)'),
    'fewer than {minimum} windows with a resolved peak of the metal: no literature-method values',
    ('literature method of the P03 in-situ GISAXS studies: D = 2*pi/q* of the direct peak, R = '
     '(delta*D^2/2.418)^(1/3) for hemispheres (h=R) on a hexagonal lattice, coverage = '
     '(2*pi/sqrt(3))*(R/D)^2; percolation where the trend of 2R/D reaches 1; after it this tool keeps D '
     '= 2*pi/q* but gives no R (Abr16; the P03 studies go on reporting R and 2R/D)'),
    'fewer than 1000 counts in the window: no value',
    'before the start of the deposition (delta = 0): no island value',
    'no resolved peak: D and R not given',
    'peak far above every literature D(delta) at this delta: probably the host, not used',
    'peak before the start of the deposition: the host, not used',
    'peak at the spacing of the host peak and far from the trend of D: probably the host, not used',
    'peak more than a factor 2 from the trend of D: another feature, not used',
    ('peak more than a factor 2 above the preceding peaks, beyond any growth since t0: a jump, not '
     'growth; not used'),
    'peak after a long stretch without a peak: another structure, not used',
    'after the percolation of the literature method: D = 2*pi/q* kept, R not given (film)',
    'no deposition rate: only D is given',
    'D (= 2*pi/q*) does not decrease with delta',
    'Percolation by the literature method against the published band',
    'TS onset of the physical path against the published band',
    'Network 2R/D and coverage against the literature',
    'Network mass balance eta = t_eq/delta',
    'TS film thickness grows with delta',
    'needs a deposition rate',
    'fewer than {minimum} windows with a resolved peak of the metal',
    ('Spearman rho = {rho} over {windows} windows; the binned median D drops by {drop}% between delta '
     '{low} and {high} nm, beyond the scatter'),
    ('Spearman rho = {rho} over {windows} windows; the binned median D drops by {drop}% between {low} '
     'and {high} s, beyond the scatter'),
    ('Spearman rho = {rho} over {windows} windows; no drop of the binned median D beyond the scatter '
     '(largest {drop}%)'),
    ('D grows or stays with delta after nucleation (delta > {start} nm); only during nucleation a '
     'slight decrease is reported'),
    ('D grows or stays with time during the deposition; only during nucleation a slight decrease is '
     'reported'),
    '{count} peak windows were not used (host peak, outlier, jump or late peak; see lit_status).',
    'Without a rate the nucleation phase is not known: early windows are included.',
    ('{count} peak windows after a stretch without a peak (t = {low}-{high} s) were not used: their '
     'median 2*pi/q* is {median} nm, {percent}% of the end of the trend ({end} nm).'),
    ('delta = {delta} nm at t = {time} s (68% interval {low}-{high} nm with the noise, the rate '
     '±{rate_error}% and t0 ±{t0_error} s)'),
    ('not determined: within the noise, the rate ±{rate_error}% and t0 ±{t0_error} s, 2R/D reaches 1 '
     'from delta = {low} nm or not before the peak phase ends (delta = {last} nm, t = {time} s)'),
    ('delta <= {delta} nm: 2R/D is at or above 1 from the first window with a peak after nucleation (t '
     '= {time} s)'),
    ('2R/D stays below 1 (at most {ratio}) up to delta = {delta} nm (t = {time} s), where the peak '
     'phase ends, also with a {rate_error}% higher rate and t0 {t0_error} s earlier'),
    ('percolation at delta = {low}-{high} nm for {host} (published values: {count}, {gisaxs} of them by '
     'GISAXS; each with its printed ± or ±{tol}%)'),
    ('percolation at delta = {low}-{high} nm for {host} (published values: {count}, {gisaxs} of them by '
     'GISAXS; each with its printed ± or ±{tol}%); scaled to {rate} nm/s with chi = 0.07-0.17: '
     '{scaled_low}-{scaled_high} nm'),
    'no published percolation threshold for {host}',
    ('68% interval: a moving-block bootstrap of the scatter of the windows around their local median, '
     'added to the trend of D and fitted again, evaluated with the rate {rate_error}% higher and t0 '
     '{t0_error} s earlier for the lower end and the other way round for the upper end (unless given, '
     'an engineering choice; the toolkit options --rate-error and --t0-error set them). A systematic '
     'error of 2*pi/q* is not in it.'),
    ('The trend of 2R/D stays within 5% of 1 from delta = {low} to {high} nm: there a small change of '
     'the rate or t0 moves the crossing far.'),
    'Earlier than every published value of this band.',
    'Later than every published value of this band.',
    'Only a bound is measured, and it overlaps the band: the comparison is not decided.',
    'Only one published value for this class: the comparison rests on it and its ±{tol}%.',
    ('With H/R = {f} (Cu on PS-b-PEO), R is {factor} times this R and 2R/D reaches 1 at delta = {delta} '
     'nm.'),
    ('With H/R = {f} (Cu on PS-b-PEO), R is {factor} times this R and 2R/D stays below 1 in the peak '
     'phase.'),
    ('With f = 1 and with f = {f} the percolation is outside the band on opposite sides (earlier with f '
     '= 1, later with the published f): the direction depends on the island shape.'),
    ('With f = 1 and with f = {f} the comparison gives different verdicts: it depends on the island '
     'shape and is not decided.'),
    'The same holds with f = {f}.',
    'TS onset at t = {time} s, delta = {delta} nm (onsets within the slack: {low}-{high} nm)',
    'no TS onset on the path; onsets from delta = {low} nm are within the slack',
    'no TS onset on the path up to delta = {delta} nm',
    ('The TS onset is the first window the path gives to the TS network (coalescence or film '
     'candidate); a connected film is not proven.'),
    'The onset is not established: no TS window at all is within the slack.',
    ('2R/D median {median} (10-90%: {low}-{high}), coverage median {c_median} (10-90%: '
     '{c_low}-{c_high}) over {windows} island windows with delta >= 1 nm'),
    'fewer than {minimum} island windows with network R and D at delta >= 1 nm',
    ('2R/D {low}-{high} after nucleation (mass-balance studies: Au 0.8-0.95, Ag 0.66-0.95; independent: '
     'Cu on graphene 0.55-0.7), 1 at percolation; coverage {c_low}-{c_high} at delta >= 1 nm (0.5-0.65 '
     'at 1 nm, 0.58 at 3 nm for Ag on SiO2, 0.76-0.96 at 4 nm for Au on polymers), 0.56-0.9 at '
     'percolation'),
    ('The network R comes from the shape of the scattering, not from the nominal thickness: an '
     'independent test.'),
    ('coverage = (2*pi/sqrt(3))*(R/D)^2 = 0.907*(2R/D)^2 for equal hemispheres on a hexagonal lattice: '
     'it follows from 2R/D, so the two are one test.'),
    ('Only one of the two published ranges holds the network value; the published ranges of 2R/D and '
     'coverage do not map onto each other exactly.'),
    ('median {median} (10-90%: {low}-{high}) over {windows} island windows with delta >= 1 nm; with the '
     't_eq band {band_low}-{band_high}'),
    ('1 by construction in the literature method; embedding in glassy polymers holds 0.6-6%; sticking '
     'below 1 is possible on polymers but not measured for sputtering; below {threshold} is flagged '
     '(engineering choice)'),
    ('Even the upper end of the t_eq band holds less than half of the nominal metal: most of the '
     'deposited metal is not in the islands the network sees, or the rate, t0 or the hemisphere '
     'condition is off.'),
    'The islands hold more metal than was deposited: check the rate and t0.',
    'eta uses the same network R and D as the 2R/D check: the two are not independent of each other.',
    ('network film thickness {t_low}-{t_high} nm (metal-equivalent {m_low}-{m_high} nm) while delta '
     'grows from {d_low} to {d_high} nm: slope {slope}'),
    'fewer than 5 TS windows over at least 2 nm of delta',
    'no value: the network film thickness is not identifiable',
    ('after percolation the metal-equivalent film thickness grows about linearly with delta (slope near '
     '1)'),
    ('The TS film thickness is not determined: in simulation the network posterior equals its prior, so '
     'a constant value says nothing about the sample.'),
    ('The literature-method R agrees with the literature by construction: it uses the same mass balance '
     'with the nominal thickness. The independent tests are the shape of D(delta), the percolation '
     'delta against the published band, and the 2R/D and mass balance of the network.'),
    ('R scales as f^(-1/3) with the shape factor f = H/R (here f = 1): the literature reports flatter '
     'Ag on oxides and f = 2.75 for Cu on PS-b-PEO.'),
    ('delta = rate*(t - t0) with the given rate ({rate} nm/s), e.g. from a QCM; GISAXS does not measure '
     'it. The percolation interval allows the rate ±{rate_error}% and t0 ±{t0_error} s.'),
    ('t0 not given: assumed 0 s (the first frame); delta is an upper bound if the deposition started at '
     'or after the first frame, and each 1 s of error in t0 shifts delta by {rate} nm.'),
    'D = 2*pi/q* can be off by more than 40% for D > 10 nm once coalescence starts (Hod17).',
    ('The condensation (sticking) coefficient of these metals on these polymers is not known; the '
     'literature method assumes that all deposited metal stays on the surface.'),
    ('Published thresholds come from other chambers and rates; the rate dependence (delta_c ~ F^-chi, '
     'chi 0.07-0.17) is weaker than differences between chambers.'),
    ('Cu: the literature method needs no network, but the TS onset and the network checks use the Ag '
     'model.'),
    '{count} peak windows probably belong to the host (median D {D} nm) and are not used.',
    'Polymer type not given: the band of all polymers is used for {metal}.',
    'No published percolation data for {polymer}: the band of {host} is used.',
    'The band of {host} includes published values for {polymer}.',
)

__all__ = ["TOOLKIT_TEMPLATES"]
