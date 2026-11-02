"""Input readers of fit1d: Yoneda-cut npz (+json), plain CSV/TXT columns, DPDAK text exports.

load_input(path) -> (q, intensity, sigma or None, meta)
  *.npz   research.real_data.yoneda_cut output: both sides merged after estimating the qy origin from
          left/right symmetry, q column q_par, sigma = max(Poisson, left/right difference)
          (research.conditional_fit.tool.load_curve(side='both', q_column='q_par', sigma_mode='side')).
          meta carries the cut metadata (qz_at_yoneda_mean = vacuum qz at the cut row, recentering, ...).
  text    numeric columns separated by commas, semicolons, tabs or spaces; lines starting with '#' are
          comments; non-numeric lines before the data are headers (e.g. DPDAK 'q[chi]_15<TAB>I_15<TAB>'
          or '<X> <Y>' of *_fit.txt).
          Columns: a header (the last '#' comment line or header line with one name per column) that names
          a q column (q_par... preferred, then q..., then qy...), an intensity column (intensity..., I,
          I_..., I[...], counts) and optionally sigma (sigma..., err..., dI, std...) decides which columns
          are read, e.g. the per-side CSV of a yoneda cut 'qy,q_par,qz,intensity,sigma,n_pixels,column'.
          Without such a header the file must have 2 or 3 columns, read as q, I[, sigma]; more columns
          without a naming header are an error (the reader does not guess).
          A sigma column that is not positive everywhere is ignored (sigma then assumed).
q is expected in nm^-1; q_unit='A^-1' converts from inverse Angstrom.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

_SPLIT = re.compile(r'[,;\s]+')
Q_UNITS = {'nm^-1': 1.0, '1/nm': 1.0, 'nm-1': 1.0, 'A^-1': 10.0, '1/A': 10.0, 'A-1': 10.0}


def _numbers(line):
    tokens = [t for t in _SPLIT.split(line.strip()) if t]
    if not tokens:
        return None
    try:
        return [float(t) for t in tokens]
    except ValueError:
        return None


def _names(line):
    return [tok.strip().strip('"\'').lower() for tok in _SPLIT.split(line.strip().lstrip('#').strip()) if tok.strip()]


def column_map(names):
    """(q index, intensity index, sigma index or None) from column names, or None when they do not say."""
    def q_rank(name):
        if name.startswith('q_par') or name.startswith('qpar'):
            return 3
        if name == 'q' or name.startswith(('q[', 'q(', 'q_', 'q/', 'q ')) and not name.startswith(('q_z', 'q_x')):
            return 2
        if name.startswith('qy') or name.startswith('q_y'):
            return 1
        return 0

    def is_intensity(name):
        return name.startswith(('intensity', 'counts')) or name in ('i', 'iq', 'i(q)') or name.startswith(('i_', 'i[', 'i('))

    def is_sigma(name):
        return name.startswith(('sigma', 'err', 'uncert', 'std')) or name in ('di', 'de') or name.startswith(('di_', 'di['))
    ranks = [q_rank(n) for n in names]
    best = max(ranks, default=0)
    q_cols = [i for i, r in enumerate(ranks) if r == best and best > 0]
    i_cols = [i for i, n in enumerate(names) if is_intensity(n)]
    s_cols = [i for i, n in enumerate(names) if is_sigma(n)]
    if len(q_cols) != 1 or len(i_cols) != 1 or len(s_cols) > 1:
        return None
    return q_cols[0], i_cols[0], (s_cols[0] if s_cols else None)


def read_text(path):
    """(table [n, 2 or 3] with columns q, I[, sigma], meta) of a column text file (see module docstring)."""
    path = Path(path)
    comments, headers, rows, widths = [], [], [], []
    started = False
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        for number, raw in enumerate(f, 1):
            line = raw.strip()
            if not line:
                continue
            if line.startswith('#'):
                comments.append(line)
                continue
            values = _numbers(line)
            if values is None:
                if started:
                    raise ValueError(f'{path}: line {number} is not numeric after the data started: {line[:60]!r}')
                headers.append(line)
                continue
            started = True
            rows.append(values)
            widths.append(len(values))
    if not rows:
        raise ValueError(f'{path}: no numeric data lines found')
    ncol = min(widths)
    if ncol < 2:
        raise ValueError(f'{path}: need at least 2 numeric columns (q, I), found {ncol}')
    table = np.array([r[:ncol] for r in rows], float)
    mapping, named_by = None, None
    for line in headers[::-1]+comments[::-1]:                  # the header closest to the data first
        names = _names(line)
        if len(names) == ncol and column_map(names) is not None:
            mapping, named_by = column_map(names), names
            break
    if mapping is None and ncol > 3:
        raise ValueError(f'{path}: {ncol} numeric columns and no header that names the q, intensity and sigma '
                         'columns; give 2-3 columns (q, I[, sigma]) or a header line such as "# q,intensity,sigma,..."')
    if mapping is None:
        mapping = (0, 1, 2 if ncol >= 3 else None)
    table = table[:, [c for c in mapping if c is not None]]
    text = ' '.join(comments[:3]+headers[:2])
    kind = 'dpdak' if 'DPDAK' in text or any(h.lower().startswith('q[') for h in headers) else \
        ('xy_fit' if any('<X>' in h for h in headers) else 'columns')
    meta = dict(reader=f'text:{kind}', columns=int(ncol), header=headers[:3], comments=comments[:5],
                n_rows=int(len(rows)), ragged=bool(len(set(widths)) > 1),
                columns_used=dict(q=int(mapping[0]), intensity=int(mapping[1]),
                                  sigma=None if mapping[2] is None else int(mapping[2])),
                columns_named_by_header=named_by is not None, column_names=named_by)
    return table, meta


def load_input(path, *, q_unit='nm^-1', side='both', sigma_mode='side'):
    """q (nm^-1), intensity, sigma (None when not given) and meta of an input file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'{path} does not exist')
    if q_unit not in Q_UNITS:
        raise ValueError(f'q_unit must be one of {tuple(Q_UNITS)}')
    factor = Q_UNITS[q_unit]
    if path.is_dir():
        raise ValueError(f'{path} is a directory; give one curve file (yoneda_cut .npz or a q, I[, sigma] text file)')
    if path.suffix.lower() == '.npz':
        from research.conditional_fit.tool import load_curve
        with np.load(path) as f:
            keys = list(f.files)
            shapes = {k: f[k].shape for k in ('left_intensity', 'right_intensity') if k in keys}
        needed = [f'{sd}_{k}' for sd in (('left', 'right') if side == 'both' else (side,))
                  for k in ('q_par', 'intensity', 'sigma')]
        if any(k not in keys for k in needed):
            raise ValueError(f'{path}: not a yoneda_cut npz of one cut (missing {[k for k in needed if k not in keys]}; '
                             f'keys present: {keys[:12]})')
        if any(len(shape) != 1 for shape in shapes.values()):
            raise ValueError(f'{path}: holds a series of cuts (intensity arrays of shape {list(shapes.values())}); '
                             'fit1d takes one curve, extract a single frame first')
        q, y, s, meta = load_curve(path, side=side, q_column='q_par', recenter=True, sigma_mode=sigma_mode)
        meta = dict(meta)
        meta.update(reader='yoneda_cut_npz', side=side, q_column='q_par', sigma_mode=sigma_mode)
        return np.asarray(q, float)*factor, np.asarray(y, float), np.asarray(s, float), meta
    table, meta = read_text(path)
    q, y = table[:, 0]*factor, table[:, 1]
    sigma = None
    if table.shape[1] >= 3:
        s = table[:, 2]
        if np.all(np.isfinite(s) & (s > 0)):
            sigma = s
        else:
            meta['note'] = 'third column not positive everywhere: ignored, sigma assumed'
    meta.update(q_unit=q_unit, file=str(path))
    return q, y, sigma, meta
