"""Recipe of a series1d run: validation, defaults, GUI-recipe compatibility.

Canonical form (sections named as in GIMaP's gimap_insitu_recipe_v1:
gisaxs_gui/src/gimap/features/fitting/domain/insitu_recipe.py):

    {"schema": "gisaxs1d_series_recipe_v1",
     "experiment_setup": {"geometry_file": "GEOMETRY.json"}          # or inline: wavelength_nm, sdd_mm,
                                                                    # pixel_mm, direct_beam_px, alpha_i_deg
     "preprocessing": {"saturation": 1382835, "rectangles": [], "edge_margin": 3},
     "cut": {"yoneda_row": 1173, "band_rows": 5},                    # or alpha_f_deg; neither = auto-detect
     "tracking": {"yoneda": "fixed", "center": "first"},
     "model": {"composition": "sphere,sphere", "mode": "gisaxs", "qz": 0.0},
     "fitting": {"track_steps": 6, "chi2_jump": 2.0, "chi2_drift": 1.5},
     "series": {"time_step_s": 0.1, "chunk_frames": 96, "overlap_frames": 5, "warmup_frames": 15}}

Flat aliases (the names used in the task description of M3) are accepted and mapped onto the sections:
geometry -> experiment_setup, mask -> preprocessing, yoneda {policy, row, alpha_f_deg, band_rows} ->
tracking.yoneda + cut, centre/center {refine} -> tracking.center, composition/mode/qz/sphere_model/lowq/
lowq_nu/structure_factor/constraints/sigma -> model, skip/time_step_s -> series, continuity -> fitting.
GUI spellings inside the sections are accepted too (distance_mm, grazing_angle_deg, beam_center_x_px,
beam_center_y_px [bottom origin], pixel_size_x_um/pixel_size_y_um, threshold_enabled/threshold_max,
center_vertical_px [bottom origin], cut_vertical_px, tracking 'detect_each_frame'/'previous_success').
Everything is validated before any frame is read; errors are RecipeError (ValueError) with the key path.
See README_zh.md for the meaning and default of every key.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np

SCHEMA = 'gisaxs1d_series_recipe_v1'
GUI_SCHEMA = 'gimap_insitu_recipe_v1'
MODES = ('live', 'review', 'batch')


class RecipeError(ValueError):
    """Invalid recipe (message names the key and the accepted values)."""


DEFAULTS = dict(
    experiment_setup=None,
    preprocessing=dict(saturation=None, rectangles=[], edge_margin=3),
    cut=dict(yoneda_row=None, alpha_f_deg=None, band_rows=5, alpha_range_deg=[0.02, 1.0], min_shift_rows=2),
    tracking=dict(yoneda='fixed', center='first'),
    model=dict(composition=None, mode='gisaxs', qz=None, sphere_model='born', lowq=None, lowq_nu=2.0,
               structure_factor=None, constraints={}, sigma='side', substrate='Si', substrate_delta=None,
               substrate_beta=None, q_unit='nm^-1', model_dir=None),
    fitting=dict(track_steps=6, track_max_steps=24, chi2_jump=2.0, chi2_drift=1.5, fallback_cooldown=10,
                 off_probe_every=3, refresh_every=0,
                 seed=0, first_policy='robust', fallback_policy='auto', chunk_start_policy='robust',
                 drift_policy='auto',
                 change_point_sigma=[3.0, 4.5],
                 continuity=None),
    series=dict(time_step_s=0.1, time_origin_frame=1, skip=dict(fraction=0.2, min_counts=1000.0),
                chunk_frames=96, overlap_frames=5, warmup_frames=15),
)
CONTINUITY_DEFAULTS = dict(alpha=0.3, feedback=False)
GEOMETRY_KEYS = ('wavelength_nm', 'sdd_mm', 'pixel_mm', 'direct_beam_px', 'alpha_i_deg')
META_KEYS = ('schema', 'note', 'version', 'created_at', 'source', 'parent_version')
# GUI keys that have no meaning here: accepted and listed in recipe['ignored']
GUI_IGNORED = dict(preprocessing=('threshold_min', 'mirror_fill_gaps', 'mirror_gap_margin_px', 'flip_ud',
                                  'threshold_enabled', 'threshold_max'),
                   cut=('center_parallel_px', 'cut_parallel_px', 'auto_horizontal_thickness_px',
                        'center_vertical_px', 'cut_vertical_px'),
                   fitting=('initialization', 'refinement', 'refine_every_n', 'failure'),
                   model=('schema', 'model_parameters'))
FLAT_MODEL = ('composition', 'mode', 'qz', 'sphere_model', 'lowq', 'lowq_nu', 'structure_factor', 'constraints',
              'sigma', 'substrate', 'substrate_delta', 'substrate_beta', 'q_unit', 'model_dir')


def _fail(path, message):
    raise RecipeError(f'recipe: {path}: {message}')


def _number(value, path, lo=None, hi=None, integer=False, optional=False):
    if value is None and optional:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        _fail(path, f'expected a number, got {value!r}')
    v = float(value)
    if not np.isfinite(v) or (lo is not None and v < lo) or (hi is not None and v > hi):
        rng = f'>= {lo}' if hi is None else (f'<= {hi}' if lo is None else f'within [{lo}, {hi}]')
        _fail(path, f'must be a finite number {rng}, got {value!r}')
    if integer:
        if v != int(v):
            _fail(path, f'must be an integer, got {value!r}')
        return int(v)
    return v


def _section(raw, name):
    value = raw.get(name)
    if value is None:
        return {}
    if not isinstance(value, dict):
        _fail(name, f'expected an object, got {type(value).__name__}')
    return dict(value)


def _flatten_aliases(raw):
    """Flat task-style keys moved into the sections (a key given both ways is an error)."""
    raw = dict(raw)
    out = {k: _section(raw, k) for k in DEFAULTS}

    def put(section, key, value, source):
        if key in out[section]:
            _fail(source, f'given twice (also as {section}.{key})')
        out[section][key] = value

    if 'geometry' in raw:
        g = raw.pop('geometry')
        if isinstance(g, str):
            put('experiment_setup', 'geometry_file', g, 'geometry')
        elif isinstance(g, dict):
            for k, v in g.items():
                put('experiment_setup', k, v, f'geometry.{k}')
        else:
            _fail('geometry', 'expected a path to GEOMETRY.json or an object with the geometry values')
    if 'mask' in raw:
        m = raw.pop('mask')
        if not isinstance(m, dict):
            _fail('mask', 'expected an object (saturation, rectangles, edge_margin)')
        for k, v in m.items():
            put('preprocessing', k, v, f'mask.{k}')
    if 'yoneda' in raw:
        y = raw.pop('yoneda')
        if isinstance(y, str):
            y = dict(policy=y)
        if not isinstance(y, dict):
            _fail('yoneda', "expected an object {policy: fixed|track, row | alpha_f_deg}")
        names = dict(policy=('tracking', 'yoneda'), row=('cut', 'yoneda_row'), alpha_f_deg=('cut', 'alpha_f_deg'),
                     band_rows=('cut', 'band_rows'), alpha_range_deg=('cut', 'alpha_range_deg'),
                     min_shift_rows=('cut', 'min_shift_rows'))
        for k, v in y.items():
            if k not in names:
                _fail(f'yoneda.{k}', f'unknown key (allowed: {", ".join(names)})')
            put(*names[k], v, f'yoneda.{k}')
    for name in ('centre', 'center'):
        if name in raw:
            c = raw.pop(name)
            if isinstance(c, dict):
                unknown = set(c)-{'refine'}
                if unknown:
                    _fail(f'{name}.{sorted(unknown)[0]}', 'unknown key (allowed: refine)')
                c = c.get('refine', 'first')
            put('tracking', 'center', c, name)
    for key in FLAT_MODEL:
        if key in raw:
            put('model', key, raw.pop(key), key)
    if 'skip' in raw:
        put('series', 'skip', raw.pop('skip'), 'skip')
    for key in ('time_step_s', 'time_step'):
        if key in raw:
            put('series', 'time_step_s', raw.pop(key), key)
    if 'continuity' in raw:
        put('fitting', 'continuity', raw.pop('continuity'), 'continuity')
    unknown = [k for k in raw if k not in DEFAULTS and k not in META_KEYS]
    if unknown:
        allowed = sorted(list(DEFAULTS)+['geometry', 'mask', 'yoneda', 'centre', 'skip', 'time_step_s', 'continuity',
                                         *FLAT_MODEL, *META_KEYS])
        _fail(unknown[0], f'unknown key (allowed top-level keys: {", ".join(allowed)})')
    return out, {k: raw[k] for k in META_KEYS if k in raw}


def _geometry(section, base_dir, notes):
    if not section:
        return None
    s = dict(section)
    out = {}
    if 'geometry_file' in s:
        path = Path(str(s.pop('geometry_file'))).expanduser()
        if not path.is_absolute():
            path = (base_dir/path).resolve()
        if not path.exists():
            _fail('experiment_setup.geometry_file', f'{path} does not exist')
        try:
            g = json.loads(path.read_text(encoding='utf-8'))
        except json.JSONDecodeError as exc:
            _fail('experiment_setup.geometry_file', f'{path} is not a JSON file ({exc})')
        out.update(wavelength_nm=g.get('wavelength_nm'), sdd_mm=g.get('sdd_mm'), pixel_mm=g.get('pixel_mm'),
                   alpha_i_deg=g.get('alpha_i_deg'), geometry_file=str(path),
                   geometry_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        if 'direct_beam_px' in g:
            out['direct_beam_px'] = g['direct_beam_px']
        elif 'direct_beam_row_top_origin' in g:
            # a series' GEOMETRY.json: the row is measured, the column is refined per series
            out['direct_beam_row'] = g['direct_beam_row_top_origin']
        for key in ('yoneda_row', 'yoneda_alpha_f_deg'):
            if g.get(key) is not None:
                out[f'file_{key}'] = g[key]
    # GUI spellings
    aliases = dict(distance_mm='sdd_mm', grazing_angle_deg='alpha_i_deg')
    for old, new in aliases.items():
        if old in s:
            if new in s:
                _fail(f'experiment_setup.{old}', f'given twice (also as {new})')
            s[new] = s.pop(old)
    if 'pixel_size_x_um' in s or 'pixel_size_y_um' in s:
        if 'pixel_mm' in s:
            _fail('experiment_setup.pixel_size_x_um', 'given twice (also as pixel_mm)')
        px, py = s.pop('pixel_size_x_um', None), s.pop('pixel_size_y_um', None)
        if px is None or py is None:
            _fail('experiment_setup.pixel_size_x_um', 'give both pixel_size_x_um and pixel_size_y_um')
        s['pixel_mm'] = [_number(px, 'experiment_setup.pixel_size_x_um', 1e-3)/1000.0,
                         _number(py, 'experiment_setup.pixel_size_y_um', 1e-3)/1000.0]
    if 'beam_center_x_px' in s or 'beam_center_y_px' in s:
        if 'direct_beam_px' in s:
            _fail('experiment_setup.beam_center_x_px', 'given twice (also as direct_beam_px)')
        x, y = s.pop('beam_center_x_px', None), s.pop('beam_center_y_px', None)
        if x is None or y is None:
            _fail('experiment_setup.beam_center_x_px', 'give both beam_center_x_px and beam_center_y_px')
        out['direct_beam_col'] = _number(x, 'experiment_setup.beam_center_x_px')
        out['direct_beam_row_from_bottom'] = _number(y, 'experiment_setup.beam_center_y_px')
        out.pop('direct_beam_px', None)
        out.pop('direct_beam_row', None)
        notes.append('experiment_setup.beam_center_y_px is counted from the bottom edge (GUI convention): '
                     'row from the top = image rows - value')
    if 'direct_beam_col' in s:
        out['direct_beam_col'] = _number(s.pop('direct_beam_col'), 'experiment_setup.direct_beam_col')
    for key in GEOMETRY_KEYS:
        if key in s:
            out[key] = s.pop(key)
            if key == 'direct_beam_px':
                out.pop('direct_beam_row', None)
    if s:
        _fail(f'experiment_setup.{sorted(s)[0]}', 'unknown key (allowed: geometry_file, '
              f'{", ".join(GEOMETRY_KEYS)}, direct_beam_col and the GUI names distance_mm, grazing_angle_deg, '
              'beam_center_x_px, beam_center_y_px, pixel_size_x_um, pixel_size_y_um)')
    for key in ('wavelength_nm', 'sdd_mm', 'alpha_i_deg'):
        if out.get(key) is None:
            _fail(f'experiment_setup.{key}', 'missing (give it inline or through geometry_file)')
        out[key] = _number(out[key], f'experiment_setup.{key}', 1e-9)
    pix = out.get('pixel_mm')
    if isinstance(pix, (int, float)) and not isinstance(pix, bool):
        pix = [pix, pix]
    if not isinstance(pix, (list, tuple)) or len(pix) != 2:
        _fail('experiment_setup.pixel_mm', f'expected [column pitch, row pitch] in mm, got {pix!r}')
    out['pixel_mm'] = [_number(v, 'experiment_setup.pixel_mm', 1e-6) for v in pix]
    if 'direct_beam_px' in out:
        beam = out.pop('direct_beam_px')
        if not isinstance(beam, (list, tuple)) or len(beam) != 2:
            _fail('experiment_setup.direct_beam_px', f'expected [column, row from the top], got {beam!r}')
        out['direct_beam_col'] = _number(beam[0], 'experiment_setup.direct_beam_px[0]')
        out['direct_beam_row'] = _number(beam[1], 'experiment_setup.direct_beam_px[1]')
    if out.get('direct_beam_row') is None and out.get('direct_beam_row_from_bottom') is None:
        _fail('experiment_setup.direct_beam_px', 'missing (direct beam [column, row from the top])')
    return out


def _constraints_json(value):
    """Constraint dict with tuples turned into lists (JSON) - the fit1d parser accepts both."""
    return json.loads(json.dumps(value))


def _check_constraint_values(constraints):
    """Values of the constraint dict (a number, [lo, hi], 'lo:hi' or {fixed | min, max}); an error names the
    key path. Unknown targets and parameters are reported by the fit1d parser (ModelSpec.from_recipe)."""
    from research.fit1d import constraints as cons
    for name, body in constraints.items():
        nested = isinstance(body, dict) and str(name).strip() not in cons.GLOBAL_PARAMS
        for key, value in (body.items() if nested else [(None, body)]):
            path = f'model.constraints.{name}'+(f'.{key}' if key is not None else '')
            if value is None or isinstance(value, bool):
                _fail(path, f'expected a number or a range [lo, hi], got {value!r}')
            try:
                cons.normalise_value(value, path)
            except cons.ConstraintError as exc:
                text = str(exc)
                _fail(path, text[len(path)+2:] if text.startswith(f'{path}: ') else text)
            except (TypeError, ValueError) as exc:
                _fail(path, f'expected a number or a range [lo, hi], got {value!r} ({exc})')


def normalise(raw, base_dir='.'):
    """Validated recipe with every default filled in (plain JSON types). Raises RecipeError."""
    if not isinstance(raw, dict):
        raise RecipeError('recipe: expected a JSON object')
    if raw.get('normalised') is True and raw.get('schema') == SCHEMA:
        return copy.deepcopy(raw)                            # the output of an earlier call
    schema = raw.get('schema', SCHEMA)
    if schema not in (SCHEMA, GUI_SCHEMA):
        _fail('schema', f'unsupported schema {schema!r} (accepted: {SCHEMA}, {GUI_SCHEMA})')
    sections, meta = _flatten_aliases(raw)
    notes, ignored = [], []
    out = dict(schema=SCHEMA, normalised=True, source_schema=schema, meta=meta)
    for name, keys in GUI_IGNORED.items():
        for key in keys:
            if key in sections[name] and key not in ('center_vertical_px', 'cut_vertical_px', 'threshold_enabled',
                                                     'threshold_max', 'flip_ud'):
                sections[name].pop(key)
                ignored.append(f'{name}.{key}')

    out['experiment_setup'] = _geometry(sections['experiment_setup'], Path(base_dir), notes)

    # ---- preprocessing (mask)
    pre = dict(sections['preprocessing'])
    if pre.pop('flip_ud', False):
        _fail('preprocessing.flip_ud', 'not supported (rows are counted from the top of the stored image)')
    threshold_on, threshold_max = pre.pop('threshold_enabled', None), pre.pop('threshold_max', None)
    if threshold_on and 'saturation' not in pre and threshold_max is not None:
        pre['saturation'] = threshold_max
        notes.append('preprocessing.threshold_max used as the saturation limit')
    p = copy.deepcopy(DEFAULTS['preprocessing'])
    unknown = set(pre)-set(p)
    if unknown:
        _fail(f'preprocessing.{sorted(unknown)[0]}', f'unknown key (allowed: {", ".join(p)})')
    p.update(pre)
    p['saturation'] = _number(p['saturation'], 'preprocessing.saturation', 0, optional=True)
    p['edge_margin'] = _number(p['edge_margin'], 'preprocessing.edge_margin', 0, 50, integer=True)
    rects = p['rectangles']
    if not isinstance(rects, (list, tuple)):
        _fail('preprocessing.rectangles', 'expected a list of [col_lo, col_hi, row_lo, row_hi]')
    clean = []
    for i, r in enumerate(rects):
        if not isinstance(r, (list, tuple)) or len(r) != 4:
            _fail(f'preprocessing.rectangles[{i}]', f'expected [col_lo, col_hi, row_lo, row_hi], got {r!r}')
        r = [_number(v, f'preprocessing.rectangles[{i}]', 0, integer=True) for v in r]
        if r[1] <= r[0] or r[3] <= r[2]:
            _fail(f'preprocessing.rectangles[{i}]', f'empty rectangle {r} (need col_lo < col_hi and row_lo < row_hi)')
        clean.append(r)
    p['rectangles'] = clean
    out['preprocessing'] = p

    # ---- cut
    cut_in = dict(sections['cut'])
    c = copy.deepcopy(DEFAULTS['cut'])
    if 'center_vertical_px' in cut_in:
        c['yoneda_row_from_bottom'] = _number(cut_in.pop('center_vertical_px'), 'cut.center_vertical_px', 0)
        notes.append('cut.center_vertical_px is counted from the bottom edge (GUI convention)')
    if 'cut_vertical_px' in cut_in:
        cut_in.setdefault('band_rows', cut_in.pop('cut_vertical_px'))
    cut_in.pop('cut_vertical_px', None)
    unknown = set(cut_in)-set(c)
    if unknown:
        _fail(f'cut.{sorted(unknown)[0]}', f'unknown key (allowed: {", ".join(DEFAULTS["cut"])})')
    c.update(cut_in)
    c['yoneda_row'] = _number(c['yoneda_row'], 'cut.yoneda_row', 0, optional=True)
    c['alpha_f_deg'] = _number(c['alpha_f_deg'], 'cut.alpha_f_deg', 1e-6, 10, optional=True)
    given = [k for k in ('yoneda_row', 'alpha_f_deg', 'yoneda_row_from_bottom') if c.get(k) is not None]
    if len(given) > 1:
        _fail('cut', f'give only one of yoneda_row / alpha_f_deg / center_vertical_px (got {", ".join(given)})')
    c['band_rows'] = _number(c['band_rows'], 'cut.band_rows', 1, 201, integer=True)
    c['min_shift_rows'] = _number(c['min_shift_rows'], 'cut.min_shift_rows', 1, 100, integer=True)
    rng = c['alpha_range_deg']
    if not isinstance(rng, (list, tuple)) or len(rng) != 2:
        _fail('cut.alpha_range_deg', f'expected [lo, hi] in degrees, got {rng!r}')
    c['alpha_range_deg'] = [_number(rng[0], 'cut.alpha_range_deg[0]'), _number(rng[1], 'cut.alpha_range_deg[1]')]
    if c['alpha_range_deg'][0] >= c['alpha_range_deg'][1]:
        _fail('cut.alpha_range_deg', f'need lo < hi, got {rng!r}')
    out['cut'] = c

    # ---- tracking
    tr = dict(DEFAULTS['tracking'])
    tin = dict(sections['tracking'])
    unknown = set(tin)-{'yoneda', 'center', 'centre'}
    if unknown:
        _fail(f'tracking.{sorted(unknown)[0]}', 'unknown key (allowed: yoneda, center)')
    if 'centre' in tin:
        if 'center' in tin:
            _fail('tracking.centre', 'given twice (also as center)')
        tin['center'] = tin.pop('centre')
    tr.update(tin)
    yoneda = {'detect_each_frame': 'track', 'previous_success': 'track'}.get(tr['yoneda'], tr['yoneda'])
    if yoneda not in ('fixed', 'track'):
        _fail('tracking.yoneda', f"must be 'fixed' or 'track' (GUI: 'detect_each_frame' = track), got {tr['yoneda']!r}")
    centre = tr['center']
    if isinstance(centre, (int, np.integer)) and not isinstance(centre, bool):
        centre = f'frame {int(centre)}'
    if isinstance(centre, str):
        text = centre.strip().lower()
        text = {'fixed': 'none', 'detect_each_frame': 'first', 'previous_success': 'first'}.get(text, text)
        parts = text.split()
        if text in ('first', 'none'):
            centre = text
        elif len(parts) == 2 and parts[0] == 'frame' and parts[1].lstrip('-').isdigit():
            centre = f'frame {int(parts[1])}'
        else:
            _fail('tracking.center', f"must be 'first', 'frame N' (N = frame number) or 'none', got {tr['center']!r}")
    else:
        _fail('tracking.center', f"must be 'first', 'frame N' or 'none', got {centre!r}")
    out['tracking'] = dict(yoneda=yoneda, center=centre)

    # ---- model
    m = copy.deepcopy(DEFAULTS['model'])
    min_ = dict(sections['model'])
    for key in GUI_IGNORED['model']:
        if key in min_:
            min_.pop(key)
            ignored.append(f'model.{key}')
    unknown = set(min_)-set(m)
    if unknown:
        _fail(f'model.{sorted(unknown)[0]}', f'unknown key (allowed: {", ".join(m)})')
    m.update(min_)
    if not isinstance(m['composition'], str) or not m['composition'].strip():
        _fail('model.composition', "required, e.g. 'sphere,sphere' or 'sphere,vc,vc' (shapes: sphere, rc, vc)")
    if m['mode'] not in ('gisaxs', 'saxs'):
        _fail('model.mode', f"must be 'gisaxs' or 'saxs', got {m['mode']!r}")
    if m['sphere_model'] not in ('born', 'dwba'):
        _fail('model.sphere_model', f"must be 'born' or 'dwba', got {m['sphere_model']!r}")
    if m['lowq'] not in (None, 'lorentz', 'powerlaw', 'none'):
        _fail('model.lowq', f"must be 'lorentz', 'powerlaw', 'none' or null, got {m['lowq']!r}")
    if m['sigma'] not in ('side', 'poisson'):
        _fail('model.sigma', f"must be 'side' (max(Poisson, left/right difference)) or 'poisson', got {m['sigma']!r}")
    if m['structure_factor'] not in (None, True, False):
        _fail('model.structure_factor', f'must be true, false or null, got {m["structure_factor"]!r}')
    if isinstance(m['qz'], str):
        if m['qz'].strip().lower() not in ('vacuum', 'auto'):
            _fail('model.qz', f"must be a number (nm^-1), 'vacuum' or null, got {m['qz']!r}")
        m['qz'] = None
    m['qz'] = _number(m['qz'], 'model.qz', 0, optional=True)
    if isinstance(m['lowq_nu'], str):
        if m['lowq_nu'].strip().lower() != 'free':
            _fail('model.lowq_nu', f"must be a number or 'free', got {m['lowq_nu']!r}")
        m['lowq_nu'] = None
    elif m['lowq_nu'] is not None:
        m['lowq_nu'] = _number(m['lowq_nu'], 'model.lowq_nu', 0.5, 12)
    if m['mode'] == 'saxs' and m['qz'] not in (None, 0.0):
        _fail('model.qz', 'SAXS mode has no qz (omit it or give 0)')
    if m['mode'] == 'saxs' and m['sphere_model'] == 'dwba':
        _fail('model.sphere_model', "'dwba' applies to mode 'gisaxs' only")
    if m['q_unit'] not in ('nm^-1', 'A^-1'):
        _fail('model.q_unit', f"must be 'nm^-1' or 'A^-1', got {m['q_unit']!r}")
    if not isinstance(m['constraints'], dict):
        _fail('model.constraints', 'expected an object in the fit1d syntax, e.g. {"pop1": {"R_nm": [2, 3]}}')
    _check_constraint_values(m['constraints'])
    m['constraints'] = _constraints_json(m['constraints'])
    m['composition'] = m['composition'].strip().lower()
    out['model'] = m

    # ---- fitting
    f = copy.deepcopy(DEFAULTS['fitting'])
    fin = dict(sections['fitting'])
    unknown = set(fin)-set(f)
    if unknown:
        _fail(f'fitting.{sorted(unknown)[0]}', f'unknown key (allowed: {", ".join(f)})')
    f.update(fin)
    f['track_steps'] = _number(f['track_steps'], 'fitting.track_steps', 1, 200, integer=True)
    f['track_max_steps'] = _number(f['track_max_steps'], 'fitting.track_max_steps', 1, 1000, integer=True)
    if f['track_max_steps'] < f['track_steps']:
        _fail('fitting.track_max_steps', f"must be >= track_steps ({f['track_steps']})")
    f['chi2_jump'] = _number(f['chi2_jump'], 'fitting.chi2_jump', 1.0)
    f['chi2_drift'] = _number(f['chi2_drift'], 'fitting.chi2_drift', 0.0)
    if 0 < f['chi2_drift'] <= 1:
        _fail('fitting.chi2_drift', f"must be 0 (off) or greater than 1, got {f['chi2_drift']!r}")
    f['fallback_cooldown'] = _number(f['fallback_cooldown'], 'fitting.fallback_cooldown', 0, integer=True)
    f['off_probe_every'] = _number(f['off_probe_every'], 'fitting.off_probe_every', 0, integer=True)
    f['refresh_every'] = _number(f['refresh_every'], 'fitting.refresh_every', 0, integer=True)
    f['seed'] = _number(f['seed'], 'fitting.seed', 0, integer=True)
    for key in ('first_policy', 'fallback_policy', 'chunk_start_policy', 'drift_policy'):
        if f[key] not in ('fast', 'auto', 'robust'):
            _fail(f'fitting.{key}', f"must be 'fast', 'auto' or 'robust', got {f[key]!r}")
    cps = f['change_point_sigma']
    if not isinstance(cps, (list, tuple)) or len(cps) != 2:
        _fail('fitting.change_point_sigma', f'expected [two-frame threshold, single-frame threshold], got {cps!r}')
    f['change_point_sigma'] = [_number(cps[0], 'fitting.change_point_sigma[0]', 1.0),
                               _number(cps[1], 'fitting.change_point_sigma[1]', 1.0)]
    cont = f['continuity']
    if cont in (None, False):
        f['continuity'] = None
    else:
        if cont is True:
            cont = {}
        if not isinstance(cont, dict):
            _fail('fitting.continuity', 'expected null/false, true or an object {alpha, feedback}')
        unknown = set(cont)-set(CONTINUITY_DEFAULTS)
        if unknown:
            _fail(f'fitting.continuity.{sorted(unknown)[0]}', 'unknown key (allowed: alpha, feedback)')
        cc = dict(CONTINUITY_DEFAULTS, **cont)
        cc['alpha'] = _number(cc['alpha'], 'fitting.continuity.alpha', 1e-3, 1.0)
        if not isinstance(cc['feedback'], bool):
            _fail('fitting.continuity.feedback', f'must be true or false, got {cc["feedback"]!r}')
        f['continuity'] = cc
    out['fitting'] = f

    # ---- series
    s = copy.deepcopy(DEFAULTS['series'])
    sin = dict(sections['series'])
    unknown = set(sin)-set(s)
    if unknown:
        _fail(f'series.{sorted(unknown)[0]}', f'unknown key (allowed: {", ".join(s)})')
    skip = sin.pop('skip', None)
    s.update(sin)
    if skip is False:
        s['skip'] = dict(fraction=0.0, min_counts=0.0)
    elif skip is not None:
        if isinstance(skip, (int, float)) and not isinstance(skip, bool):
            skip = dict(fraction=skip)
        if not isinstance(skip, dict):
            _fail('series.skip', 'expected an object {fraction, min_counts}, a number (fraction) or false')
        skip = dict(skip)
        if 'fraction_of_running_median' in skip:
            skip.setdefault('fraction', skip.pop('fraction_of_running_median'))
        unknown = set(skip)-{'fraction', 'min_counts'}
        if unknown:
            _fail(f'series.skip.{sorted(unknown)[0]}', 'unknown key (allowed: fraction, min_counts)')
        s['skip'].update(skip)
    s['skip'] = dict(fraction=_number(s['skip']['fraction'], 'series.skip.fraction', 0, 1),
                     min_counts=_number(s['skip']['min_counts'], 'series.skip.min_counts', 0))
    s['time_step_s'] = _number(s['time_step_s'], 'series.time_step_s', 1e-9)
    s['time_origin_frame'] = _number(s['time_origin_frame'], 'series.time_origin_frame', integer=True)
    s['chunk_frames'] = _number(s['chunk_frames'], 'series.chunk_frames', 8, integer=True)
    s['overlap_frames'] = _number(s['overlap_frames'], 'series.overlap_frames', 1, integer=True)
    if s['overlap_frames'] >= s['chunk_frames']:
        _fail('series.overlap_frames', f"must be smaller than chunk_frames ({s['chunk_frames']})")
    s['warmup_frames'] = _number(s['warmup_frames'], 'series.warmup_frames', 0, integer=True)
    out['series'] = s
    out['notes'] = notes
    out['ignored'] = ignored
    if ignored:
        notes.append('GUI keys without a meaning in series1d were ignored: '+', '.join(ignored))

    # ---- model checks that need fit1d (composition syntax, constraints against the composition)
    from research.series1d.model import ModelSpec
    ModelSpec.from_recipe(out, independent=True)
    return out


def load(path):
    """Read and validate a recipe file (relative geometry_file paths are relative to the recipe)."""
    path = Path(path).expanduser()
    if not path.exists():
        raise RecipeError(f'recipe: {path} does not exist')
    try:
        raw = json.loads(path.read_text(encoding='utf-8'))
    except json.JSONDecodeError as exc:
        raise RecipeError(f'recipe: {path} is not a JSON file ({exc})') from None
    return normalise(raw, base_dir=path.parent)


def sha256(recipe):
    """Hash of the normalised recipe (identifies the settings of a run: --resume refuses another hash)."""
    body = {k: v for k, v in recipe.items() if k not in ('meta', 'notes', 'ignored', 'source_schema', 'normalised')}
    return hashlib.sha256(json.dumps(body, sort_keys=True, allow_nan=False).encode('utf-8')).hexdigest()


def check_mode(recipe, mode, source_kind):
    """Mode-dependent rules (causality, input kind). Raises RecipeError."""
    if mode not in MODES:
        raise RecipeError(f'mode must be one of {MODES}, got {mode!r}')
    centre = recipe['tracking']['center']
    if mode == 'live' and centre.startswith('frame'):
        _fail('tracking.center', f"{centre!r} is not causal: live mode calibrates on frames <= t only "
              "(use 'first' or 'none'), review/batch accept a reference frame")
    if source_kind == 'precut' and recipe['tracking']['yoneda'] == 'track':
        _fail('tracking.yoneda', "'track' needs raw detector frames (a pre-cut series has one fixed cut)")
    if source_kind == 'raw' and recipe['experiment_setup'] is None:
        _fail('experiment_setup', 'raw detector frames need the geometry (geometry_file or inline values)')
    if source_kind != 'curves':
        from research.series1d.model import ModelSpec
        ModelSpec.from_recipe(recipe)                      # 'auto' compositions are refused for tracked series
    if source_kind == 'curves' and recipe['model']['sphere_model'] == 'dwba':
        _fail('model.sphere_model', "'dwba' needs the cut geometry (raw frames or a pre-cut series with a geometry)")
    if source_kind == 'curves' and recipe['model']['mode'] == 'gisaxs' and recipe['model']['qz'] is None:
        # curve files carry no qz of the cut: spheres / random cylinders would silently be fitted at qz = 0
        from research.fit1d.api import parse_compositions
        from research.multishape_v1.physics import COMBOS
        combos, _ = parse_compositions(recipe['model']['composition'])
        if any(bool(np.any(np.isin(COMBOS[int(c)], (1, 2)))) for c in combos):
            _fail('model.qz', 'not given, and curve files carry no qz of the cut: give model.qz in nm^-1 (the '
                  'vacuum qz at the Yoneda row; 0 only on purpose) for compositions with spheres or random '
                  "cylinders, or use mode 'saxs'")
