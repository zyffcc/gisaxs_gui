"""Frame sources of series1d and the cut calibration.

  precut   one npz in the REAL-PACK series layout (research.real_data.realpack): per side 2D arrays
           [frame, column] left_/right_ q_par, intensity, sigma, n_pixels (optional qz), NaN padded, plus
           frame_number
  raw      detector frames (cbf, tif, edf, npy, ...; research.real_data.yoneda_cut.read_image) cut with a
           CutPlan built from the recipe geometry and the reference frame
  curves   a folder / glob of 1D curve files (DPDAK exports, CSV/TXT; research.fit1d.io.load_input), one
           independent fit per file

Calibration (JSON dict, stored in the header line of frames.jsonl; review and batch: calibrate()):
  reference frame   tracking.center 'first' = the first frame with at least series.skip.min_counts counts in
                    the cut (causal); 'frame N' = frame number N (review/batch only); 'none' = no refinement.
                    It must show both sides of the cut (an InputError says so otherwise).
  raw frames        beam-centre column refined by left/right symmetry on the reference frame
                    (yoneda_cut(refine_center=True), as research.real_data.realpack), then the CutPlan at the
                    recipe's Yoneda row / alpha_f (or the row found on the reference frame)
  both              qy-origin shift `delta` and left/right ratio of the reference cut
                    (symmetry.estimate_center_shift); every frame is merged with these two numbers
Live mode (prepare_live / complete_live): before the first frame only the setup is built (for raw frames the
cut geometry with the recipe's beam centre and the detector layout of the first file); the calibration is
completed on the first usable frame when it arrives and stored in that row. Raw frames: the qy shift of the
symmetry is applied to every cut (shift_sides) instead of new cuts at a refined beam centre.
Per frame: merge_sides(delta, ratio) -> sigma = max(Poisson, left/right difference) (model.sigma 'side') ->
finite points with q > 0 and sigma > 0, sorted by q. Points whose intensity or sigma is not finite are no
points; a frame that cannot be read is reported by the tracker as a skipped row.
"""
from __future__ import annotations

import glob as globlib
import hashlib
import re
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from research.conditional_fit.pipeline import _clean_inputs
from research.real_data.symmetry import estimate_center_shift, merge_sides, side_difference_sigma
from research.real_data.yoneda_cut import Geometry

IMAGE_SUFFIXES = ('.cbf', '.tif', '.tiff', '.edf', '.npy', '.h5', '.hdf5', '.nxs', '.gz')
CURVE_SUFFIXES = ('.txt', '.dat', '.csv', '.asc')
SIDE_KEYS = ('q_par', 'intensity', 'sigma', 'n_pixels')
FORMATS = ('auto', 'npz', 'raw', 'dpdak', 'csv')
MIN_SIDE_POINTS = 10              # points with a positive intensity that each side of the reference cut needs


class InputError(ValueError):
    """Unusable --files / --input-format (message says what was found)."""


def frame_number_of(name, index):
    """Trailing integer of a file stem (sample_00001_00033.cbf -> 33); index + 1 without one."""
    m = re.search(r'(\d+)(?!.*\d)', Path(name).stem)
    return int(m.group(1)) if m else index+1


def _slice(n, frames):
    if frames is None:
        return list(range(n))
    if isinstance(frames, str):
        m = re.fullmatch(r'\s*(-?\d*)\s*:\s*(-?\d*)\s*', frames)
        if not m:
            raise InputError(f"--frames must look like A:B (index slice, e.g. 150:190), got {frames!r}")
        frames = (int(m.group(1)) if m.group(1) else None, int(m.group(2)) if m.group(2) else None)
    idx = list(range(n))[slice(frames[0], frames[1])]
    if not idx:
        raise InputError(f'--frames {frames} selects no frame out of {n}')
    return idx


# ---------------------------------------------------------------------------- sources
class PrecutSeries:
    kind = 'precut'

    def __init__(self, path, frames=None):
        self.path = Path(path)
        with np.load(self.path) as f:
            keys = set(f.files)
            needed = [f'{side}_{k}' for side in ('left', 'right') for k in ('q_par', 'intensity', 'sigma')]
            missing = [k for k in needed if k not in keys]
            if missing:
                raise InputError(f'{self.path}: not a pre-cut series npz (missing {missing}; keys: {sorted(keys)[:12]})')
            if f['left_intensity'].ndim != 2:
                raise InputError(f'{self.path}: holds one cut (1-D arrays), not a series [frame, column]; '
                                 'use python3 -m research.fit1d for a single cut')
            wanted = [f'{side}_{k}' for side in ('left', 'right') for k in SIDE_KEYS+('qz',)]
            self.arrays = {k: np.asarray(f[k], float) for k in wanted if k in keys}
            n = self.arrays['left_intensity'].shape[0]
            self.frame_numbers = np.asarray(f['frame_number'], int) if 'frame_number' in keys else np.arange(1, n+1)
        self.index = _slice(n, frames)
        self.frames = frames

    def __len__(self):
        return len(self.index)

    def info(self, i):
        return int(self.frame_numbers[self.index[i]]), f'{self.path.name}[{self.index[i]}]'

    def spec(self):
        return dict(kind='precut', path=str(self.path), frames=self.frames)

    def describe(self):
        st = self.path.stat()
        return dict(kind='precut', path=str(self.path), bytes=st.st_size, sha256=_sha256(self.path),
                    frames=len(self), first_index=self.index[0], last_index=self.index[-1])

    def numbers_all(self):
        return [int(v) for v in self.frame_numbers]

    def sides(self, i):
        return self.sides_abs(self.index[i])

    def sides_abs(self, k):
        """Sides of the frame at position k of the whole file (the reference frame may lie outside --frames)."""
        out = {}
        for side in ('left', 'right'):
            ok = np.isfinite(self.arrays[f'{side}_q_par'][k]) & np.isfinite(self.arrays[f'{side}_intensity'][k]) \
                & np.isfinite(self.arrays[f'{side}_sigma'][k])
            out[side] = {key: self.arrays[f'{side}_{key}'][k][ok] for key in SIDE_KEYS+('qz',)
                         if f'{side}_{key}' in self.arrays}
        return out


class RawFrames:
    kind = 'raw'

    def __init__(self, files, frames=None):
        self.all_files = [str(f) for f in files]
        self.index = _slice(len(self.all_files), frames)
        self.frames = frames

    def __len__(self):
        return len(self.index)

    def path(self, i):
        return self.all_files[self.index[i]]

    def info(self, i):
        name = Path(self.path(i)).name
        return frame_number_of(name, self.index[i]), name

    def spec(self):
        return dict(kind='raw', files=self.all_files, frames=self.frames)

    def describe(self):
        names = '\n'.join(Path(self.path(i)).name for i in range(len(self)))
        return dict(kind='raw', directory=str(Path(self.path(0)).parent), frames=len(self), first=Path(self.path(0)).name,
                    last=Path(self.path(len(self)-1)).name, names_sha256=hashlib.sha256(names.encode()).hexdigest())

    def numbers_all(self):
        return [frame_number_of(Path(f).name, k) for k, f in enumerate(self.all_files)]

    def image(self, i):
        return self.image_abs(self.index[i])

    def image_abs(self, k):
        from research.real_data.yoneda_cut import read_image
        return read_image(self.all_files[k])[0]


class CurveFolder:
    kind = 'curves'

    def __init__(self, files, frames=None, q_unit='nm^-1'):
        self.all_files = [str(f) for f in files]
        self.index = _slice(len(self.all_files), frames)
        self.frames, self.q_unit = frames, q_unit

    def __len__(self):
        return len(self.index)

    def path(self, i):
        return self.all_files[self.index[i]]

    def info(self, i):
        name = Path(self.path(i)).name
        return frame_number_of(name, self.index[i]), name

    def spec(self):
        return dict(kind='curves', files=self.all_files, frames=self.frames, q_unit=self.q_unit)

    def describe(self):
        names = '\n'.join(f'{Path(self.path(i)).name} {_sha256(self.path(i))}' for i in range(len(self)))
        return dict(kind='curves', directory=str(Path(self.path(0)).parent), frames=len(self),
                    files_sha256=hashlib.sha256(names.encode()).hexdigest(), q_unit=self.q_unit)

    def curve(self, i):
        from research.fit1d.io import load_input
        return load_input(self.path(i), q_unit=self.q_unit)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def reopen(spec):
    """Source from Source.spec() (used by pool workers)."""
    if spec['kind'] == 'precut':
        return PrecutSeries(spec['path'], spec['frames'])
    if spec['kind'] == 'raw':
        return RawFrames(spec['files'], spec['frames'])
    return CurveFolder(spec['files'], spec['frames'], spec['q_unit'])


def open_source(files, input_format='auto', frames=None, q_unit='nm^-1'):
    """Source from the --files argument (an npz, a directory or a glob) and --input-format."""
    if input_format not in FORMATS:
        raise InputError(f'--input-format must be one of {FORMATS}, got {input_format!r}')
    text = str(Path(str(files)).expanduser())
    path = Path(text)
    if path.is_file():
        paths = [path]
    elif path.is_dir():
        paths = sorted(p for p in path.iterdir() if p.is_file() and not p.name.startswith('.'))
    else:
        paths = sorted(Path(p) for p in globlib.glob(text) if Path(p).is_file())
    if not paths:
        raise InputError(f'--files {files!r}: no file found (give a series npz, a directory or a glob pattern)')
    suffixes = {p.suffix.lower() for p in paths}
    if input_format == 'auto':
        if len(paths) == 1 and paths[0].suffix.lower() == '.npz':
            input_format = 'npz'
        elif any(s in CURVE_SUFFIXES for s in suffixes) and not any(s in IMAGE_SUFFIXES for s in suffixes):
            input_format = 'dpdak' if any(p.name.endswith('_fit.txt') for p in paths) or _looks_dpdak(paths[0]) else 'csv'
        elif any(s in IMAGE_SUFFIXES for s in suffixes):
            input_format = 'raw'
        else:
            raise InputError(f'--files {files!r}: cannot tell the input format from the suffixes {sorted(suffixes)}; '
                             f'use --input-format {"|".join(FORMATS[1:])}')
    if input_format == 'npz':
        if len(paths) != 1:
            raise InputError(f'--input-format npz takes one series file, --files matched {len(paths)}')
        return PrecutSeries(paths[0], frames)
    if input_format == 'raw':
        keep = [p for p in paths if p.suffix.lower() in IMAGE_SUFFIXES]
        if not keep:
            raise InputError(f'--files {files!r}: no detector image among {len(paths)} files '
                             f'(suffixes {sorted(suffixes)})')
        return RawFrames(keep, frames)
    keep = [p for p in paths if p.suffix.lower() in CURVE_SUFFIXES]
    if input_format == 'dpdak':
        keep = [p for p in keep if not p.name.endswith('_fit.txt')]          # *_fit.txt = the user's manual fits
    if not keep:
        raise InputError(f'--files {files!r}: no curve file ({", ".join(CURVE_SUFFIXES)}) found')
    return CurveFolder(keep, frames, q_unit)


def _looks_dpdak(path):
    try:
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            return 'DPDAK' in f.read(200)
    except OSError:
        return False


# ---------------------------------------------------------------------------- cuts
def total_counts(sides):
    """Counts in the cut band: sum over the columns of mean intensity x pixels (both sides, finite values)."""
    total = 0.0
    for side in ('left', 'right'):
        c = sides[side]
        n = np.asarray(c['n_pixels'], float) if 'n_pixels' in c else 1.0
        values = np.asarray(c['intensity'], float)*n
        total += float(np.sum(values[np.isfinite(values)]))
    return total


def curve_from_sides(sides, delta, ratio, sigma_mode='side'):
    """(q, I, sigma) of one frame: both sides merged with the reference shift/ratio; sigma = max(Poisson,
    left/right difference) for sigma_mode 'side' (research.conditional_fit.batch.curve_from_cut)."""
    left, right = sides['left'], sides['right']
    q, y, s, _ = merge_sides(left['q_par'], left['intensity'], left['sigma'], right['q_par'], right['intensity'],
                             right['sigma'], delta=delta, ratio=ratio)
    if sigma_mode == 'side':
        try:
            sig, _ = side_difference_sigma(left['q_par'], left['intensity'], right['q_par'], right['intensity'],
                                           delta=delta, ratio=ratio)
            est = sig(q, y)
            s = np.where(np.isfinite(est), np.maximum(est, s), s)
        except (ValueError, IndexError):          # too few overlapping points for the estimate: Poisson only
            pass
    return _clean_inputs(q, y, s)


def alpha_f_of_row(geometry, row):
    """Exit angle (degrees) of an image row at the beam-centre column (the rule of realpack._plan_from)."""
    dx = (int(geometry.direct_beam_px[0])-geometry.direct_beam_px[0])*geometry.pixel_mm[0]
    dz = (row-geometry.direct_beam_px[1])*geometry.pixel_mm[1]
    if not geometry.row_up:
        dz = -dz
    return float(np.degrees(np.arctan2(dz, np.hypot(geometry.sdd_mm, dx))-np.radians(geometry.alpha_i_deg)))


@dataclass
class CachedGeometry(Geometry):
    """Geometry whose per-pixel q maps are computed once per image shape (the CutPlan constructor and the
    Yoneda finder ask for the same maps again; 0.09 s per call on a 2.5 Mpixel frame). The values are the
    ones of Geometry.q; release() drops the maps (about 80 MB for such a frame)."""

    def q(self, shape):
        cache = self.__dict__.setdefault('_q_maps', {})
        key = tuple(int(v) for v in shape)
        if key not in cache:
            cache.clear()
            cache[key] = Geometry.q(self, shape)
        return cache[key]

    def release(self):
        self.__dict__.pop('_q_maps', None)


def geometry_from(recipe, shape):
    g = recipe['experiment_setup']
    if g.get('direct_beam_col') is None:
        from research.series1d.recipe import RecipeError
        raise RecipeError('recipe: experiment_setup.direct_beam_col: missing (start value of the beam-centre '
                          'column in pixels; it is refined on the reference frame unless tracking.center is none)')
    row = g.get('direct_beam_row')
    if row is None:
        row = shape[0]-g['direct_beam_row_from_bottom']
    return CachedGeometry(wavelength_nm=g['wavelength_nm'], sdd_mm=g['sdd_mm'], pixel_mm=tuple(g['pixel_mm']),
                          direct_beam_px=(g['direct_beam_col'], row), alpha_i_deg=g['alpha_i_deg'])


class YonedaFinder:
    """Row of the Yoneda maximum of one frame (off-specular vertical profile, 3-row smoothing), with the
    per-pixel angles precomputed once (yoneda_cut.find_yoneda recomputes them for every call)."""

    def __init__(self, geometry, shape, mask, alpha_range_deg, qy_window=(0.05, 0.3)):
        _, qy, _, af = geometry.q(shape)
        use = ~mask & (np.abs(qy) >= qy_window[0]) & (np.abs(qy) <= qy_window[1])
        counts = use.sum(axis=1)
        af_row = np.where(counts > 0, np.where(use, af, 0).sum(axis=1)/np.maximum(counts, 1), np.nan)
        ok = np.isfinite(af_row) & (np.degrees(af_row) > alpha_range_deg[0]) & (np.degrees(af_row) < alpha_range_deg[1])
        rows = np.flatnonzero(ok)
        if not len(rows):
            raise InputError('no valid off-specular rows in cut.alpha_range_deg for the Yoneda search')
        self.r0, self.r1 = int(rows.min()), int(rows.max())+1
        self.use = use[self.r0:self.r1]
        self.ok = ok[self.r0:self.r1]

    def row(self, image):
        block = np.asarray(image[self.r0:self.r1], float)
        use = self.use & np.isfinite(block) & (block >= 0)
        counts = use.sum(axis=1)
        mean = np.where(counts > 0, np.where(use, block, 0).sum(axis=1)/np.maximum(counts, 1), 0.0)
        valid = self.ok & (counts > 0)
        smooth = np.convolve(np.where(valid, mean, 0.0), np.ones(3)/3, 'same')
        return self.r0+int(np.argmax(np.where(valid, smooth, -np.inf)))


def usable(total, recipe):
    """A frame can serve as the reference / first frame: at least series.skip.min_counts counts in the cut."""
    return bool(np.isfinite(total) and total >= max(recipe['series']['skip']['min_counts'], 1e-300))


def check_reference_sides(sides, label):
    """The left/right symmetry needs both sides of the reference cut (InputError says what to do)."""
    for side in ('left', 'right'):
        y = np.asarray(sides[side]['intensity'], float)
        good = int(np.sum(np.isfinite(y) & (y > 0)))
        if good < MIN_SIDE_POINTS:
            raise InputError(
                f'reference frame {label}: the {side} side of the cut has {good} points with a positive intensity '
                f'(at least {MIN_SIDE_POINTS} needed), so the centre cannot be refined from the left/right '
                "symmetry. Use tracking.center 'none' (no refinement), or 'frame N' with a frame that shows both "
                'sides (review/batch).')


def symmetry_of(sides, centre, label):
    """Calibration entries delta (qy-origin shift) and ratio (left/right) of a reference cut."""
    if centre == 'none':
        return dict(delta=0.0, ratio=1.0)
    check_reference_sides(sides, label)
    left, right = sides['left'], sides['right']
    shift = estimate_center_shift(left['q_par'], left['intensity'], right['q_par'], right['intensity'])
    ok = bool(np.isfinite(shift['delta']) and np.isfinite(shift['ratio_left_over_right'])
              and shift['ratio_left_over_right'] > 0)
    return dict(delta=float(shift['delta']) if ok else 0.0,
                ratio=float(shift['ratio_left_over_right']) if ok else 1.0,
                mismatch_before=float(shift['mismatch_before']), mismatch_after=float(shift['mismatch_after']),
                at_search_edge=bool(shift['at_search_edge']), symmetry_ok=ok)


def first_readable(source):
    """(image, position in the whole input) of the first detector frame of the run that can be read."""
    first_error = None
    for i in range(len(source)):
        try:
            return source.image(i), source.index[i]
        except Exception as exc:          # noqa: BLE001 - any reader error: try the next file
            first_error = first_error or f'{source.path(i)}: {type(exc).__name__}: {exc}'
    raise InputError(f'none of the {len(source)} detector frames can be read (first error: {first_error})')


def calibrate(source, recipe, with_cutter=False):
    """Calibration dict of a review / batch run (see module docstring). Reads frames only up to the
    reference frame. reference_index is the position of the reference frame in the whole input (it may lie
    outside --frames for tracking.center 'frame N'). with_cutter: also return the Cutter built for it."""
    cal, cutter = _calibrate(source, recipe)
    if with_cutter:
        return cal, (cutter if cutter is not None else Cutter(source, recipe, cal))
    return cal


def _calibrate(source, recipe):
    centre = recipe['tracking']['center']
    cal = dict(kind=source.kind, center=centre)
    if source.kind == 'curves':
        return cal, None
    cutter = None
    if source.kind == 'raw':
        image0, _ = first_readable(source)
        cal['shape'] = [int(v) for v in image0.shape]
        cutter = Cutter(source, recipe, dict(cal, provisional=True), reference_image=image0)

    def sides_of(k):
        """Sides of the frame at position k of the whole input; None when the frame cannot be read or cut."""
        if source.kind == 'precut':
            return source.sides_abs(k)
        try:
            image = source.image_abs(k)
        except Exception:                 # noqa: BLE001 - an unreadable frame is no reference frame
            return None
        if tuple(image.shape) != tuple(cutter.shape):
            return None
        return cutter.plan.apply(image)

    numbers = source.numbers_all()
    if centre.startswith('frame'):
        number = int(centre.split()[1])
        if number not in numbers:
            raise InputError(f'recipe: tracking.center: frame number {number} is not in the input '
                             f'(frame numbers {numbers[0]}..{numbers[-1]})')
        ref = numbers.index(number)
        sides = sides_of(ref)
        if sides is None:
            raise InputError(f'recipe: tracking.center: frame number {number} ({source.all_files[ref]}) cannot be '
                             'read or has another image shape than the first frame')
    else:
        ref, sides = None, None
        for i in range(len(source)):
            sides = sides_of(source.index[i])
            if sides is not None and usable(total_counts(sides), recipe):
                ref = source.index[i]
                break
        if ref is None:
            raise InputError(f'no frame has at least series.skip.min_counts = {recipe["series"]["skip"]["min_counts"]:g} '
                             'counts in the cut: nothing to calibrate or fit')
    label = f'{numbers[ref]} (input position {ref})'
    cal.update(reference_index=int(ref), reference_frame=int(numbers[ref]))
    if centre != 'none':
        check_reference_sides(sides, label)
    if source.kind == 'raw':
        try:
            cutter = Cutter(source, recipe, cal, reference_image=source.image_abs(ref), refine=centre != 'none')
        except ValueError as exc:
            if isinstance(exc, InputError):
                raise
            raise InputError(f'reference frame {label}: the centre refinement failed ({exc}); use '
                             "tracking.center 'none' or another reference frame") from None
        cal.update(cutter.calibration)
        sides = sides_of(ref)
    cal.update(symmetry_of(sides, centre, label))
    if cutter is not None:
        cutter.calibration = dict(cal)
    return cal, cutter


def prepare_live(source, recipe):
    """Live mode, before the first frame arrives: (calibration with 'pending', Cutter).

    Nothing that depends on the counts of a frame is done here. For detector frames the layout of the
    detector (image shape and flagged pixels: module gaps, negative values) is read from the first file and
    the cut geometry is built with the beam centre of the recipe; the calibration is completed by
    complete_live on the first usable frame at its arrival time (causal: frames <= t only)."""
    cal = dict(kind=source.kind, center=recipe['tracking']['center'], pending=True)
    if source.kind == 'curves':
        return dict(cal, pending=False), None
    if source.kind != 'raw':
        return cal, Cutter(source, recipe, cal)
    image, position = first_readable(source)
    cal.update(shape=[int(v) for v in image.shape], mask_index=int(position), mask_saturation=False)
    return cal, Cutter(source, recipe, cal, layout_image=image)


def shift_sides(sides, shift):
    """A CutPlan cut seen from a beam centre moved by `shift` in qy (signed convention of
    estimate_center_shift: the true qy = 0 lies at signed qy = shift): qy of every column is moved,
    q_par = sqrt(qx^2 + qy^2) recomputed with the column's own qx, and the columns are assigned to the
    sides again. This is what a new cut with the beam-centre column moved by shift / (dqy/dcolumn) gives
    (same pixels per column), without building a new plan."""
    left, right = sides['left'], sides['right']
    qy = np.r_[-np.asarray(left['qy'], float), np.asarray(right['qy'], float)]-shift
    qx2 = np.maximum(np.r_[np.asarray(left['q_par'], float)**2-np.asarray(left['qy'], float)**2,
                           np.asarray(right['q_par'], float)**2-np.asarray(right['qy'], float)**2], 0.0)
    both = {k: np.r_[left[k], right[k]] for k in left if k not in ('qy', 'q_par') and k in right}
    out = {k: v for k, v in sides.items() if k not in ('left', 'right')}
    for side, keep in (('left', qy < 0), ('right', qy > 0)):
        idx = np.flatnonzero(keep)
        idx = idx[np.argsort(np.abs(qy[idx]))]
        out[side] = dict({k: v[idx] for k, v in both.items()}, qy=np.abs(qy[idx]),
                         q_par=np.sqrt(qx2[idx]+qy[idx]**2))
    return out


def complete_live(cal, cutter, source, recipe, i, sides):
    """Live mode: the calibration completed on frame i of the run (the first usable one), from its own cut.
    Returns (calibration, the sides of this frame in that calibration).

    Detector frames: the beam-centre column is not refined by new cuts of the whole image (0.5 s). The shift
    of the qy origin from the left/right symmetry of the cut - the first iteration of that refinement - is
    applied to the qy of every column of every cut (shift_sides; calibration entry qy_shift, and
    center_col_equivalent = the beam-centre column it stands for). Review and batch iterate twice on the
    reference frame and build the plan at the refined column."""
    number, _ = source.info(i)
    centre = recipe['tracking']['center']
    label = f'{number} (input position {source.index[i]})'
    out = {k: v for k, v in cal.items() if k != 'pending'}
    out.update(reference_index=int(source.index[i]), reference_frame=int(number))
    if source.kind != 'raw':
        out.update(symmetry_of(sides, centre, label))
        return out, sides
    out.update(cutter.fix())
    if centre == 'none':
        out.update(delta=0.0, ratio=1.0)
        return out, sides
    check_reference_sides(sides, label)
    left, right = sides['left'], sides['right']
    shift = estimate_center_shift(left['qy'], left['intensity'], right['qy'], right['intensity'])
    ok = bool(np.isfinite(shift['delta']) and np.isfinite(shift['ratio_left_over_right'])
              and shift['ratio_left_over_right'] > 0)
    geo = cutter.geometry
    dq_dcol = 2*np.pi/geo.wavelength_nm*np.cos(np.radians(cutter.alpha_f))*geo.pixel_mm[0]/geo.sdd_mm
    qy_shift = float(shift['delta']) if ok else 0.0
    out.update(qy_shift=qy_shift, delta=0.0, ratio=float(shift['ratio_left_over_right']) if ok else 1.0,
               mismatch_before=float(shift['mismatch_before']), mismatch_after=float(shift['mismatch_after']),
               at_search_edge=bool(shift['at_search_edge']), symmetry_ok=ok,
               center_col_equivalent=float(geo.direct_beam_px[0]+qy_shift/dq_dcol))
    cutter.shift = qy_shift
    return out, (shift_sides(sides, qy_shift) if qy_shift else sides)


class Cutter:
    """Cuts of one run in one process: pre-cut sides, or a CutPlan applied to raw frames.

    The mask of the plan comes from one image: the reference frame (review / batch: flagged and saturated
    pixels) or, in live mode, the detector layout of the first file (flagged pixels only; calibration
    entries mask_index / mask_saturation). Saturated and flagged pixels of every frame are masked by the
    plan itself. state() / restore(state) carry the Yoneda row of tracking.yoneda 'track' (the plan follows
    the detected row when it moved by at least cut.min_shift_rows; causal: only the current frame is used).
    While a live calibration is pending and the recipe gives no Yoneda row, the row is searched on every
    frame (fix() ends that on the first usable frame)."""

    def __init__(self, source, recipe, calibration, reference_image=None, refine=False, layout_image=None):
        self.source, self.recipe = source, recipe
        self.calibration = dict(calibration)
        self.plan = None
        self.alpha_f = None
        self.qz = None
        self.row = None
        self.search = False
        self.shift = float(calibration.get('qy_shift') or 0.0)      # live: qy-origin shift applied to every cut
        if source.kind != 'raw':
            return
        from research.real_data.yoneda_cut import MaskSpec, yoneda_cut
        pre, cut = recipe['preprocessing'], recipe['cut']
        saturation = pre['saturation']
        if layout_image is not None:
            mask_image, saturation = layout_image, None
        elif calibration.get('mask_index') is not None:
            mask_image = source.image_abs(calibration['mask_index'])
            if not calibration.get('mask_saturation', True):
                saturation = None
        else:
            if reference_image is None:
                reference_image = source.image_abs(calibration['reference_index'])
            mask_image = reference_image
        self.shape = mask_image.shape
        geo = geometry_from(recipe, self.shape)
        self.mask = MaskSpec(saturation=saturation, rectangles=[tuple(r) for r in pre['rectangles']],
                             edge_margin=pre['edge_margin']).build(mask_image)
        if 'direct_beam_px' in calibration:                     # worker / resume: the stored calibration
            geo = replace(geo, direct_beam_px=tuple(calibration['direct_beam_px']))
        elif refine:
            ref = yoneda_cut(reference_image, geo, self.mask, refine_center=True, band_rows=cut['band_rows'],
                             alpha_range_deg=tuple(cut['alpha_range_deg']))
            info = ref['meta']['center_refinement']
            geo = replace(geo, direct_beam_px=tuple(info['refined_direct_beam_px']))
            self.calibration['center_refinement'] = dict(
                original_direct_beam_px=[float(v) for v in info['original_direct_beam_px']],
                column_shifts=[float(h['column_shift']) for h in info['iterations']])
        self.geometry = geo
        self.finder = None
        if 'yoneda_alpha_f_deg' in calibration:
            alpha = calibration['yoneda_alpha_f_deg']
        elif cut['alpha_f_deg'] is not None:
            alpha = cut['alpha_f_deg']
        else:
            row = cut['yoneda_row']
            if row is None and cut.get('yoneda_row_from_bottom') is not None:
                row = self.shape[0]-cut['yoneda_row_from_bottom']
            if row is None and reference_image is None:
                self.search = True                              # live: found on the frames as they arrive
                self._finder()
                return
            if row is None:
                row = self._finder().row(reference_image)
                self.calibration['yoneda_row_found'] = int(row)
            alpha = alpha_f_of_row(geo, row)
        self._set_alpha(alpha)
        self.calibration.update(self._entries())

    def _entries(self):
        return dict(direct_beam_px=[float(v) for v in self.geometry.direct_beam_px],
                    yoneda_alpha_f_deg=float(self.alpha_f), band_rows=int(self.recipe['cut']['band_rows']),
                    qz_vacuum=float(self.qz), masked_fraction=float(self.mask.mean()))

    def _release(self):
        """Drop the per-pixel q maps when the plan cannot change any more."""
        if not self.search and self.recipe['tracking']['yoneda'] != 'track' and not self.calibration.get('pending') \
                and not self.calibration.get('provisional'):
            self.geometry.release()

    def _finder(self):
        if self.finder is None:
            self.finder = YonedaFinder(self.geometry, self.shape, self.mask, self.recipe['cut']['alpha_range_deg'])
        return self.finder

    def _set_alpha(self, alpha):
        from research.real_data.yoneda_cut import CutPlan
        pre, cut = self.recipe['preprocessing'], self.recipe['cut']
        self.plan = CutPlan(self.geometry, self.shape, self.mask, float(alpha), band_rows=cut['band_rows'],
                            saturation=pre['saturation'], edge_margin=pre['edge_margin'])
        self.alpha_f = float(alpha)
        self.qz = float(self.plan.meta['qz_at_yoneda_mean'])
        af = self.geometry.q(self.shape)[3]
        centre = np.argmin(np.abs(af[:, int(self.geometry.direct_beam_px[0])]-np.radians(alpha)))
        self.row = int(centre)
        self._release()

    def fix(self):
        """Live: the cut of the current frame becomes the cut of the series; returns the calibration entries."""
        entries = self._entries()
        if self.search:
            entries['yoneda_row_found'] = int(self.row)
        self.search = False
        self.calibration.pop('pending', None)
        self.calibration.update(entries)
        self._release()
        return entries

    def calibrated(self, calibration):
        """A completed live calibration read back from frames.jsonl (resume)."""
        self.calibration = dict(calibration)
        if self.source.kind == 'raw':
            if self.search and calibration.get('yoneda_alpha_f_deg') is not None and self.plan is None:
                self._set_alpha(calibration['yoneda_alpha_f_deg'])
            self.search = False
            self.shift = float(calibration.get('qy_shift') or 0.0)

    def state(self):
        return None if self.plan is None else dict(alpha_f_deg=self.alpha_f, yoneda_row=self.row)

    def restore(self, state):
        if self.source.kind == 'raw' and state and state.get('alpha_f_deg') is not None \
                and state['alpha_f_deg'] != self.alpha_f:
            self._set_alpha(state['alpha_f_deg'])

    def sides(self, i):
        """(sides dict, cut info dict) of frame i."""
        if self.source.kind == 'precut':
            sides = self.source.sides(i)
            qz = None
            if 'qz' in sides['left'] and 'qz' in sides['right']:
                both = np.r_[sides['left']['qz'], sides['right']['qz']]
                qz = float(np.mean(both)) if len(both) else None
            return sides, dict(qz_vacuum=qz)
        image = self.source.image(i)
        if tuple(image.shape) != tuple(self.shape):
            raise InputError(f'{self.source.path(i)}: image shape {tuple(image.shape)} differs from the reference '
                             f'frame {tuple(self.shape)}')
        if self.search:
            row = self._finder().row(image)
            if self.plan is None or row != self.row:
                self._set_alpha(alpha_f_of_row(self.geometry, row))
        elif self.recipe['tracking']['yoneda'] == 'track':
            row = self._finder().row(image)
            if abs(row-self.row) >= self.recipe['cut']['min_shift_rows']:
                self._set_alpha(alpha_f_of_row(self.geometry, row))
        cut = self.plan.apply(image)
        if self.shift:
            cut = shift_sides(cut, self.shift)
        return cut, dict(qz_vacuum=self.qz, alpha_f_deg=self.alpha_f, yoneda_row=self.row)


class NullCutter:
    """Cutter of a tracker that only replays stored rows (batch stitching: state without cuts)."""
    calibration: dict = {}

    def state(self):
        return None

    def restore(self, state):
        return None

    def calibrated(self, calibration):
        return None

    def sides(self, i):
        raise RuntimeError('NullCutter cannot cut frames')
