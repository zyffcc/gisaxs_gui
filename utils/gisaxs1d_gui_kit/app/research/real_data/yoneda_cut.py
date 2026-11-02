"""Horizontal line cuts at the Yoneda peak from 2D GISAXS detector images.

Geometry (flat detector perpendicular to the direct beam):
  dx = (col - col0) * pixel_col,  dz = (row0 - row) * pixel_row   [mm, z up]
  2theta_f = atan2(dx, L),  alpha_f = atan2(dz, sqrt(L^2 + dx^2)) - alpha_i
  k = 2 pi / lambda
  qx = k (cos a_f cos 2t_f - cos a_i),  qy = k cos a_f sin 2t_f,  qz = k (sin a_f + sin a_i)
(col0, row0) is the direct-beam position (alpha_f = -alpha_i); if only the
specular spot is known, row0 = row_spec + L tan(2 alpha_i) / pixel_row.
`row_up=False` (default) means alpha_f grows toward smaller row index, the
usual display orientation; set row_up=True for images stored bottom-up.

The Yoneda peak alpha_Y is taken as the maximum of the off-specular vertical
profile above the horizon (or supplied). The cut averages valid pixels in a
band of rows around alpha_Y, per detector column, separately for qy<0 and
qy>0. Per column it reports qy, q_par = sqrt(qx^2+qy^2), qz, the mean pixel
intensity, a Poisson sigma when the image holds raw counts, and the number of
pixels. Masked pixels (negative detector flags, gaps, beamstop, saturation,
user rectangles) never enter a mean; nothing is interpolated.
"""
from __future__ import annotations

import hashlib
import json
import re
import zlib
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np


# ----------------------------------------------------------------------------- readers
def _read_edf(path):
    raw = Path(path).read_bytes()
    end = raw.index(b'}')+1
    header = raw[:end].decode('latin-1')
    keys = dict(re.findall(r'(\w+)\s*=\s*([^;]*?)\s*;', header))
    dtypes = {'SignedByte': 'i1', 'UnsignedByte': 'u1', 'SignedShort': 'i2', 'UnsignedShort': 'u2',
              'SignedInteger': 'i4', 'UnsignedInteger': 'u4', 'SignedLong': 'i4', 'UnsignedLong': 'u4',
              'Signed64': 'i8', 'Unsigned64': 'u8', 'FloatValue': 'f4', 'FLOATVALUE': 'f4', 'DoubleValue': 'f8'}
    order = '<' if keys.get('ByteOrder', 'LowByteFirst') == 'LowByteFirst' else '>'
    dtype = np.dtype(order+dtypes[keys['DataType']])
    d1, d2 = int(keys['Dim_1']), int(keys['Dim_2'])
    size = int(keys.get('Size', d1*d2*dtype.itemsize))
    start = len(raw)-size  # data block sits at the end of the (padded) header
    return np.frombuffer(raw[start:start+size], dtype).reshape(d2, d1).astype(float), keys


def _cbf_byte_offset(buf, n):
    """CBF byte-offset decompression: runs of 1-byte deltas are decoded in bulk with numpy;
    only the escape codes (0x80 -> int16, 0x8000 -> int32, 0x80000000 -> int64) are walked."""
    raw = np.frombuffer(buf, np.uint8)
    b = raw.astype(np.int8)
    candidates = np.flatnonzero(raw == 0x80)
    deltas = np.empty(n, np.int64)
    i, j, c = 0, 0, 0
    while j < n:
        c = np.searchsorted(candidates, i, side='left') if c < len(candidates) and candidates[c] < i else c
        nxt = candidates[c] if c < len(candidates) else len(b)
        run = min(nxt-i, n-j)
        if run > 0:
            deltas[j:j+run] = b[i:i+run]
            i += run
            j += run
            if j >= n:
                break
        # escape at position i
        d = int.from_bytes(buf[i+1:i+3], 'little', signed=True)
        i += 3
        if d == -32768:
            d = int.from_bytes(buf[i:i+4], 'little', signed=True)
            i += 4
            if d == -2**31:
                d = int.from_bytes(buf[i:i+8], 'little', signed=True)
                i += 8
        deltas[j] = d
        j += 1
        c += 1
    return np.cumsum(deltas)


def _read_cbf(path):
    try:
        import fabio
        img = fabio.open(str(path))
        return np.asarray(img.data, float), dict(img.header)
    except ImportError:
        pass
    raw = Path(path).read_bytes()
    marker = raw.index(b'\x0c\x1a\x04\xd5')
    header = raw[:marker].decode('latin-1')
    get = lambda key: int(re.search(key+r':\s*(\d+)', header).group(1))
    fast, slow = get('X-Binary-Size-Fastest-Dimension'), get('X-Binary-Size-Second-Dimension')
    if 'x-CBF_BYTE_OFFSET' not in header:
        raise ValueError('Only byte-offset CBF is supported without fabio')
    data = _cbf_byte_offset(raw[marker+4:marker+4+get('X-Binary-Size')], fast*slow)
    return data.reshape(slow, fast).astype(float), dict(raw_header=header[:4000])


def _read_tiff_raw(path):
    """Minimal reader for uncompressed TIFFs (e.g. Pilatus/ALBULA signed 32-bit) that PIL rejects."""
    import struct
    raw = Path(path).read_bytes()
    order = {b'II': '<', b'MM': '>'}[raw[:2]]
    offset = struct.unpack(order+'I', raw[4:8])[0]
    count = struct.unpack(order+'H', raw[offset:offset+2])[0]
    sizes = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2, 9: 4, 10: 8, 11: 4, 12: 8}
    codes = {1: 'B', 3: 'H', 4: 'I', 6: 'b', 8: 'h', 9: 'i'}
    tags = {}
    for k in range(count):
        entry = raw[offset+2+12*k:offset+14+12*k]
        tag, typ, n = struct.unpack(order+'HHI', entry[:8])
        size = sizes.get(typ, 1)*n
        data = entry[8:12] if size <= 4 else raw[struct.unpack(order+'I', entry[8:12])[0]:][:size]
        if typ in codes:
            tags[tag] = struct.unpack(order+codes[typ]*n, data[:sizes[typ]*n])
    width, height = tags[256][0], tags[257][0]
    bits = tags.get(258, (8,))[0]
    fmt = tags.get(339, (1,))[0]
    if tags.get(259, (1,))[0] != 1:
        raise ValueError('compressed TIFF not supported by the fallback reader')
    kind = {1: 'u', 2: 'i', 3: 'f'}[fmt]
    dtype = np.dtype(order+kind+str(bits//8))
    strips, counts = tags[273], tags[279]
    buffer = b''.join(raw[o:o+c] for o, c in zip(strips, counts))
    return np.frombuffer(buffer, dtype)[:width*height].reshape(height, width).astype(float), dict(tiff_tags=len(tags))


def read_image(path, dataset=None):
    """Return (image[rows, cols] float, header dict). Supports tif, edf, cbf, npy/npz, h5/nxs."""
    path = Path(path)
    name = path.name.lower()
    if name.endswith('.gz'):
        # The format readers take a path: decompress into a private temporary directory that is removed
        # (with the decompressed copy) before returning, so nothing is left behind.
        import gzip
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            inner = Path(tmp)/path.name[:-3]
            inner.write_bytes(gzip.decompress(path.read_bytes()))
            return read_image(inner, dataset)
    if name.endswith(('.tif', '.tiff')):
        try:
            from PIL import Image
            with Image.open(path) as im:
                return np.asarray(im, dtype=float), dict(im.info)
        except Exception:
            return _read_tiff_raw(path)
    if name.endswith('.edf'):
        return _read_edf(path)
    if name.endswith('.cbf'):
        return _read_cbf(path)
    if name.endswith('.npy'):
        return np.load(path).astype(float), {}
    if name.endswith('.npz'):
        with np.load(path) as f:
            return np.asarray(f[dataset or f.files[0]], float), {}
    if name.endswith(('.h5', '.hdf5', '.nxs')):
        import h5py
        with h5py.File(path, 'r') as f:
            return np.asarray(f[dataset], float), {}
    if name.endswith(('.txt', '.dat', '.asc')):
        return np.loadtxt(path), {}
    raise ValueError(f'Unsupported image format: {path.name}')


# ----------------------------------------------------------------------------- geometry
@dataclass
class Geometry:
    wavelength_nm: float
    sdd_mm: float
    pixel_mm: tuple          # (column pitch, row pitch)
    direct_beam_px: tuple    # (col0, row0), alpha_f = -alpha_i
    alpha_i_deg: float
    row_up: bool = False

    @staticmethod
    def from_specular(wavelength_nm, sdd_mm, pixel_mm, specular_px, alpha_i_deg, row_up=False):
        col, row = specular_px
        shift = sdd_mm*np.tan(2*np.radians(alpha_i_deg))/pixel_mm[1]
        row0 = row-shift if row_up else row+shift
        return Geometry(wavelength_nm, sdd_mm, tuple(pixel_mm), (col, row0), alpha_i_deg, row_up)

    def angles(self, shape):
        rows, cols = np.indices(shape, dtype=float)
        dx = (cols-self.direct_beam_px[0])*self.pixel_mm[0]
        dz = (rows-self.direct_beam_px[1])*self.pixel_mm[1]
        if not self.row_up:
            dz = -dz
        two_theta = np.arctan2(dx, self.sdd_mm)
        alpha_f = np.arctan2(dz, np.hypot(self.sdd_mm, dx))-np.radians(self.alpha_i_deg)
        return alpha_f, two_theta

    def q(self, shape):
        af, tt = self.angles(shape)
        ai = np.radians(self.alpha_i_deg)
        k = 2*np.pi/self.wavelength_nm
        qx = k*(np.cos(af)*np.cos(tt)-np.cos(ai))
        qy = k*np.cos(af)*np.sin(tt)
        qz = k*(np.sin(af)+np.sin(ai))
        return qx, qy, qz, af


# ----------------------------------------------------------------------------- masking
@dataclass
class MaskSpec:
    min_valid: float = 0.0                 # pixels below this are detector flags (Pilatus: -1/-2)
    saturation: float | None = None       # pixels >= this are excluded
    rectangles: list = field(default_factory=list)   # [(col_lo, col_hi, row_lo, row_hi)] inclusive-exclusive
    extra: np.ndarray | None = None       # bool array, True = masked
    edge_margin: int = 3                   # also mask pixels within this distance of detector-flag pixels
                                           # (module-gap edges respond abnormally; Pilatus practice ~3 px)

    def build(self, image):
        flags = ~np.isfinite(image) | (image < self.min_valid)
        m = flags.copy()
        if self.edge_margin > 0 and flags.any():
            from scipy.ndimage import binary_dilation
            m |= binary_dilation(flags, iterations=int(self.edge_margin))
        if self.saturation is not None:
            m |= image >= self.saturation
        for c0, c1, r0, r1 in self.rectangles:
            m[int(r0):int(r1), int(c0):int(c1)] = True
        if self.extra is not None:
            m |= self.extra
        return m


# ----------------------------------------------------------------------------- Yoneda + cut
def vertical_profile(image, mask, geometry, qy_window=(0.05, 0.3)):
    """Mean intensity per row over off-specular columns with qy_window[0] <= |qy| <= qy_window[1]."""
    _, qy, _, af = geometry.q(image.shape)
    use = ~mask & (np.abs(qy) >= qy_window[0]) & (np.abs(qy) <= qy_window[1])
    counts = use.sum(axis=1)
    mean = np.where(counts > 0, np.where(use, image, 0).sum(axis=1)/np.maximum(counts, 1), np.nan)
    af_row = np.where(counts > 0, np.where(use, af, 0).sum(axis=1)/np.maximum(counts, 1), np.nan)
    return af_row, mean, counts


def find_yoneda(image, mask, geometry, qy_window=(0.05, 0.3), alpha_range_deg=(0.02, 1.0), smooth_rows=3):
    af_row, mean, counts = vertical_profile(image, mask, geometry, qy_window)
    ok = np.isfinite(mean) & (np.degrees(af_row) > alpha_range_deg[0]) & (np.degrees(af_row) < alpha_range_deg[1])
    if not ok.any():
        raise ValueError('No valid off-specular rows in the Yoneda search range')
    values = np.where(ok, mean, -np.inf)
    if smooth_rows > 1:
        kernel = np.ones(smooth_rows)/smooth_rows
        values = np.where(ok, np.convolve(np.where(ok, mean, 0), kernel, 'same'), -np.inf)
    row = int(np.argmax(values))
    return dict(row=row, alpha_f_deg=float(np.degrees(af_row[row])), profile_alpha_f_deg=np.degrees(af_row),
                profile=mean, profile_pixels=counts)


def yoneda_cut(image, geometry, mask=None, *, refine_center=False, **kwargs):
    """Return dict(left=..., right=..., meta=...) of per-column cuts in a band around the Yoneda peak.

    refine_center=True moves the beam-centre column so that the two sides of the
    cut agree (left/right symmetry of an isotropic in-plane sample), recuts, and
    records the correction in meta['center_refinement'] (two iterations).
    """
    cut = _yoneda_cut(image, geometry, mask, **kwargs)
    if not refine_center:
        return cut
    from dataclasses import replace
    from research.real_data.symmetry import estimate_center_shift
    history = []
    geo = geometry
    for _ in range(2):
        shift = estimate_center_shift(cut['left']['qy'], cut['left']['intensity'], cut['right']['qy'], cut['right']['intensity'])
        k = 2*np.pi/geo.wavelength_nm
        dqy_dcol = k*np.cos(np.radians(cut['meta']['yoneda_alpha_f_deg']))*geo.pixel_mm[0]/geo.sdd_mm
        dcol = shift['delta']/dqy_dcol
        history.append(dict(delta_q=shift['delta'], column_shift=dcol, mismatch_before=shift['mismatch_before'],
                            mismatch_after=shift['mismatch_after']))
        geo = replace(geo, direct_beam_px=(geo.direct_beam_px[0]+dcol, geo.direct_beam_px[1]))
        cut = _yoneda_cut(image, geo, mask, **kwargs)
    cut['meta']['center_refinement'] = dict(iterations=history, original_direct_beam_px=list(geometry.direct_beam_px),
                                            refined_direct_beam_px=list(geo.direct_beam_px))
    return cut


def _yoneda_cut(image, geometry, mask=None, *, yoneda_alpha_f_deg=None, band_rows=5, raw_counts=True,
                qy_window=(0.05, 0.3), alpha_range_deg=(0.02, 1.0), min_pixels=1):
    image = np.asarray(image, float)
    # Default: mask detector flags (negative / non-finite pixels, e.g. Pilatus gaps = -1).
    mask = MaskSpec().build(image) if mask is None else np.asarray(mask, bool) | ~np.isfinite(image)
    qx, qy, qz, af = geometry.q(image.shape)
    if yoneda_alpha_f_deg is None:
        found = find_yoneda(image, mask, geometry, qy_window, alpha_range_deg)
        yoneda_alpha_f_deg = found['alpha_f_deg']
    else:
        found = None
    # central row of the band: per column, the row whose alpha_f is closest to alpha_Y
    target = np.radians(yoneda_alpha_f_deg)
    centre = np.argmin(np.abs(af-target), axis=0)
    half = band_rows//2
    rows = np.arange(image.shape[0])[:, None]
    band = (np.abs(rows-centre[None, :]) <= half) & ~mask
    n = band.sum(axis=0)
    total = np.where(band, image, 0).sum(axis=0)
    mean = lambda arr: np.where(band, arr, 0).sum(axis=0)/np.maximum(n, 1)
    col_qy, col_qx, col_qz, col_af = mean(qy), mean(qx), mean(qz), mean(af)
    intensity = total/np.maximum(n, 1)
    sigma = np.sqrt(np.maximum(total, 1.0))/np.maximum(n, 1) if raw_counts else np.full_like(intensity, np.nan)
    valid = (n >= min_pixels) & (np.abs(col_qy) > 0)
    out = {}
    for side, keep in (('left', valid & (col_qy < 0)), ('right', valid & (col_qy > 0))):
        idx = np.flatnonzero(keep)
        order = np.argsort(np.abs(col_qy[idx]))
        idx = idx[order]
        out[side] = dict(column=idx, qy=np.abs(col_qy[idx]), q_par=np.hypot(col_qx[idx], col_qy[idx]),
                         qz=col_qz[idx], alpha_f_deg=np.degrees(col_af[idx]), intensity=intensity[idx],
                         sigma=sigma[idx], n_pixels=n[idx])
    out['meta'] = dict(geometry=asdict(geometry), yoneda_alpha_f_deg=float(yoneda_alpha_f_deg),
                       yoneda_found_automatically=found is not None, band_rows=int(band_rows),
                       raw_counts=bool(raw_counts), qy_window_for_search=list(qy_window),
                       qz_at_yoneda_mean=float(np.mean(col_qz[valid])) if valid.any() else None,
                       masked_fraction=float(mask.mean()),
                       sigma_rule='sqrt(sum counts)/n_pixels (Poisson)' if raw_counts else 'unknown (not raw counts)')
    if found is not None:
        out['meta']['yoneda_search'] = dict(row=found['row'], alpha_range_deg=list(alpha_range_deg))
        out['profile'] = dict(alpha_f_deg=found['profile_alpha_f_deg'], intensity=found['profile'])
    return out


# ----------------------------------------------------------------------------- I/O
def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_cut(cut, out_dir, name, source=None, quicklook_image=None, geometry=None, mask=None):
    """Write <name>_{left,right}.csv, <name>.npz, <name>.json and an optional quicklook PNG."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for side in ('left', 'right'):
        c = cut[side]
        table = np.column_stack([c['qy'], c['q_par'], c['qz'], c['intensity'], c['sigma'], c['n_pixels'], c['column']])
        np.savetxt(out_dir/f'{name}_{side}.csv', table, delimiter=',', fmt='%.8g',
                   header='qy_nm-1,q_par_nm-1,qz_nm-1,intensity,sigma,n_pixels,detector_column')
        arrays.update({f'{side}_{k}': np.asarray(v) for k, v in c.items()})
    np.savez_compressed(out_dir/f'{name}.npz', **arrays)
    from research.real_data.symmetry import check_cut
    meta = dict(cut['meta'], name=name, source=source or {}, side_check=check_cut(cut) if cut['meta']['raw_counts'] else None,
                units=dict(q='nm^-1', intensity='detector counts per pixel (mean over band)'
                                                                      if cut['meta']['raw_counts'] else 'as stored in image'))
    (out_dir/f'{name}.json').write_text(json.dumps(meta, indent=1, ensure_ascii=False, default=float))
    if quicklook_image is not None:
        quicklook(quicklook_image, cut, geometry, out_dir/f'{name}.png', mask=mask, title=name)
    return meta


def quicklook(image, cut, geometry, path, mask=None, title=''):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    shown = np.where(mask, np.nan, image) if mask is not None else image
    axes[0].imshow(np.log10(np.clip(shown, 1e-1, None)), cmap='viridis', origin='upper' if not geometry.row_up else 'lower')
    for side, colour in (('left', 'r'), ('right', 'w')):
        c = cut[side]
        _, _, _, af = geometry.q(image.shape)
        rows = np.argmin(np.abs(af[:, c['column']]-np.radians(cut['meta']['yoneda_alpha_f_deg'])), axis=0)
        axes[0].plot(c['column'], rows, '.', ms=1, color=colour)
    axes[0].set_title(f'{title}: log10 image + cut band')
    if 'profile' in cut:
        axes[1].semilogy(cut['profile']['alpha_f_deg'], cut['profile']['intensity'])
        axes[1].axvline(cut['meta']['yoneda_alpha_f_deg'], color='r', ls='--')
        axes[1].set_xlim(-0.1, 1.2)
        axes[1].set_xlabel('alpha_f (deg)')
        axes[1].set_title('off-specular vertical profile')
    for side in ('left', 'right'):
        c = cut[side]
        pos = c['intensity'] > 0
        axes[2].loglog(c['qy'][pos], c['intensity'][pos], '.', ms=2, label=side)
    axes[2].set_xlabel('|qy| (nm^-1)')
    axes[2].legend()
    axes[2].set_title('Yoneda cut')
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ----------------------------------------------------------------------------- self test
def selftest(seed=0):
    """Synthetic detector image with a known 1D law along qy and a Yoneda peak; checks recovery."""
    rng = np.random.default_rng(seed)
    geo = Geometry(wavelength_nm=0.1045, sdd_mm=4300.0, pixel_mm=(0.172, 0.172), direct_beam_px=(740.0, 1500.0),
                   alpha_i_deg=0.4)
    shape = (1679, 1475)   # Pilatus 2M-like
    qx, qy, qz, af = geo.q(shape)
    alpha_c = 0.22
    law = lambda q: 1e4*(1/(1+(q*25)**2))**2*(1+0.8*np.exp(-((q-0.18)/0.02)**2))+5
    yoneda = np.exp(-((np.degrees(af)-alpha_c)/0.05)**2)*3+np.where(af > 0, 0.3, 0.0)
    clean = law(np.abs(qy))*yoneda
    image = rng.poisson(clean).astype(float)
    image[:, 487:494] = -1                       # module gap
    image[1440:1560, 700:780] = 0                # beamstop region, masked below
    mask = MaskSpec(rectangles=[(700, 780, 1440, 1560)]).build(image)
    cut = yoneda_cut(image, geo, mask, band_rows=5)
    found = cut['meta']['yoneda_alpha_f_deg']
    right = cut['right']
    # The band mean of the Yoneda factor is a constant along the cut, so intensity/law should be flat.
    ratio = right['intensity']/law(right['qy'])
    return dict(yoneda_found_deg=found, yoneda_true_deg=alpha_c, band_pixel_deg=float(np.degrees(0.172/4300)),
                right_points=len(right['qy']), left_points=len(cut['left']['qy']),
                gap_columns_excluded=bool(not np.isin(np.arange(487, 494), np.r_[right['column'], cut['left']['column']]).any()),
                ratio_cv=float(np.std(ratio[right['qy'] < 0.3])/np.mean(ratio[right['qy'] < 0.3])),
                qy_range=[float(right['qy'].min()), float(right['qy'].max())])


if __name__ == '__main__':
    print(json.dumps(selftest(), indent=1))


def _dilate(flags, steps):
    """scipy.ndimage.binary_dilation(flags, iterations=steps) for a 2D bool array (default cross-shaped
    element, outside = False), written as array shifts: the same mask, about 9x faster on a band block."""
    out = flags
    for _ in range(int(steps)):
        grown = out.copy()
        grown[1:] |= out[:-1]
        grown[:-1] |= out[1:]
        grown[:, 1:] |= out[:, :-1]
        grown[:, :-1] |= out[:, 1:]
        out = grown
    return out


class CutPlan:
    """Precomputed Yoneda cut for a series recorded with one geometry (real-time use).

    The band rows, per-column pixel lists and q values are computed once from a reference
    image (or given alpha_f); apply(image) then only averages pixel values per column, so a
    frame costs a fraction of a millisecond. Pixels flagged in the reference mask stay masked.

    apply(image) masks the frame exactly like the full cut with
    mask = reference_mask | MaskSpec(saturation=saturation, edge_margin=edge_margin).build(image):
    the per-frame detector flags (non-finite or negative pixels) are dilated by edge_margin, saturated
    pixels (>= saturation) are masked without dilation. The dilation sees edge_margin rows above and
    below the band block (clamped at the image border), so a flag just outside the block masks the
    same band pixels as in the full image. q per column is the mean over the pixels used: the
    precomputed reference values, recomputed for a frame whose own flags remove band pixels. Column,
    intensity, sigma, n_pixels, qy, q_par and qz are bit-identical to yoneda_cut with that mask and the
    plan's alpha_f; the plan cut has no per-column alpha_f_deg and its meta is the plan's (reference) meta.
    """

    def __init__(self, geometry, shape, mask, yoneda_alpha_f_deg, band_rows=5, saturation=None, edge_margin=3):
        self.geometry, self.shape = geometry, shape
        self.saturation, self.edge_margin = saturation, max(int(edge_margin or 0), 0)
        qx, qy, qz, af = geometry.q(shape)
        centre = np.argmin(np.abs(af-np.radians(yoneda_alpha_f_deg)), axis=0)
        half = band_rows//2
        rows = np.arange(shape[0])[:, None]
        band = (np.abs(rows-centre[None, :]) <= half) & ~np.asarray(mask, bool)
        self.rows_lo = np.clip(centre-half, 0, shape[0]-1)
        self.band = band[self.rows_lo.min():self.rows_lo.max()+band_rows]
        self.row0 = int(self.rows_lo.min())
        sub = slice(self.row0, self.row0+self.band.shape[0])
        # rows read per frame: the band block plus edge_margin rows on each side for the flag dilation
        self._ext = slice(max(self.row0-self.edge_margin, 0), min(sub.stop+self.edge_margin, shape[0]))
        self._inner = slice(self.row0-self._ext.start, self.row0-self._ext.start+self.band.shape[0])
        n = self.band.sum(axis=0)
        self._n_ref = n
        self._q_block = tuple(arr[sub].copy() for arr in (qy, qx, qz))      # copies: the full-image q arrays are released
        mean = lambda arr: np.where(self.band, arr, 0).sum(axis=0)/np.maximum(n, 1)
        self.col_qy, self.col_qx, self.col_qz = (mean(arr) for arr in self._q_block)
        self.valid = (n > 0) & (np.abs(self.col_qy) > 0)
        self.meta = dict(geometry=asdict(geometry), yoneda_alpha_f_deg=float(yoneda_alpha_f_deg), band_rows=int(band_rows),
                         qz_at_yoneda_mean=float(np.mean(self.col_qz[self.valid])))

    def apply(self, image):
        ext = np.asarray(np.asarray(image)[self._ext], float)
        block = ext[self._inner]
        # detector flags of the extended block, dilated by edge_margin (the rule of MaskSpec.build), cropped
        bad = _dilate(~np.isfinite(ext) | (ext < 0), self.edge_margin)[self._inner]
        if self.saturation is not None:
            bad = bad | (block >= self.saturation)      # saturated pixels are not dilated (MaskSpec.build)
        use = self.band & ~bad
        n = use.sum(axis=0)
        total = np.where(use, block, 0).sum(axis=0)
        intensity = total/np.maximum(n, 1)
        sigma = np.sqrt(np.maximum(total, 1.0))/np.maximum(n, 1)
        col_qy, col_qx, col_qz = self.col_qy, self.col_qx, self.col_qz
        if not np.array_equal(n, self._n_ref):
            # this frame's own flags removed band pixels: q is the mean over the pixels used (full-cut rule)
            col_qy, col_qx, col_qz = (np.where(use, arr, 0).sum(axis=0)/np.maximum(n, 1) for arr in self._q_block)
        valid = (n > 0) & (np.abs(col_qy) > 0)
        out = {}
        for side, keep in (('left', valid & (col_qy < 0)), ('right', valid & (col_qy > 0))):
            idx = np.flatnonzero(keep)
            idx = idx[np.argsort(np.abs(col_qy[idx]))]
            out[side] = dict(column=idx, qy=np.abs(col_qy[idx]), q_par=np.hypot(col_qx[idx], col_qy[idx]),
                             qz=col_qz[idx], intensity=intensity[idx], sigma=sigma[idx], n_pixels=n[idx])
        out['meta'] = dict(self.meta, raw_counts=True)
        return out
