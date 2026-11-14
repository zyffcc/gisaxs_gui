''

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import iam_paths as IAMP

MASK = ROOT / "assets" / "static_p03_2026_pixmask_v1.npy"
_W = {}


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def _init(geometry, zips, mask_path=None):
    sys.path.insert(0, str(IAMP.DIST))
    from sputtergisaxs.cbfio import _decode_byte_offset, _parse_header
    from sputtergisaxs.stage5_input import AtomicMap
    mask = np.load(mask_path or MASK).astype(bool)
    _W.update(dec=_decode_byte_offset, parse=_parse_header, mask=mask, zips=zips,
              amap=AtomicMap(geometry, mask.shape, geometry["alpha_i_deg"], crop=None, exclude_rings=((2.1, .13),)), zf={})


def _frame(task):
    zname, member = task
    zf = _W["zf"].get(zname) or _W["zf"].setdefault(zname, zipfile.ZipFile(Path(_W["zips"]) / zname))
    return frame_bytes(zf.read(member))


def _file(path):
    'A frame from a file on disk (iampred/static.py: the frames of a folder).'
    return frame_bytes(Path(path).read_bytes())


def frame_bytes(raw):
    'One CBF frame (file bytes) -> counts Y and pixel exposure E on the native grid, with the per-frame bookkeeping.'
    arr, hdr = _W["dec"](raw)
    h = _W["parse"](hdr)
    cut = float(re.search(r"Count_cutoff\s+([\d.eE+-]+)", hdr).group(1))
    img = arr.astype(np.float64)
    bad = (img < 0) | (img >= cut)
    sat = binary_dilation(img >= 0.5 * cut, structure=np.ones((5, 5), bool)) & ~bad
    img[bad | sat] = np.nan
    exp = float(h["exposure_s"])
    Y, E = _W["amap"].project(img, exp, _W["mask"])
    hstart = raw.find(b"\x0c\x1a\x04\xd5")
    return dict(Y=Y, E=E, exposure_s=exp, timestamp=h.get("timestamp"), count_cutoff=cut,
                cutoff_removed_pixels=int(((arr >= cut)).sum()), saturation_masked_pixels=int(sat.sum()),
                raw_sha256=sha_bytes(raw), header_sha256=sha_bytes(raw[:hstart]),
                selected_counts=float(Y.sum()), valid_pixel_count=float(E.sum() / exp))


def write_cache(data, lab, t, res, geometry, marker, source, alpha_i_note=None):
    'DATA/<lab>/binned/{chunk_00000.npz, chunk_00000.json, run.json, membership.json} and DATA/<lab>/pixmask.json from'
    out = data / lab / "binned"; out.mkdir(parents=True, exist_ok=True)
    Y = np.asarray([r["Y"] for r in res]).astype(np.uint64)
    E = np.asarray([r["E"] for r in res]).astype(np.float64)
    if not np.all(np.isclose(Y.astype(np.float64), np.asarray([r["Y"] for r in res]))):
        raise ValueError("non-integer counts")
    np.savez_compressed(out / "chunk_00000.npz", Y=Y, E=E)
    rows = []
    for i, (f, r) in enumerate(zip(t["frames"], res)):
        rows.append(dict(frame=f"{lab}_{i:03d}", frame_id=i, time_s=float(i), first_time_s=float(i), last_time_s=float(i),
                         time_note="frame index, not a clock", timestamp=r["timestamp"], exposure_s=r["exposure_s"],
                         raw_member=f["member"], raw_scan=f["raw_scan"], scan_folder=f["scan_folder"],
                         state=f["state"], reason=f["reason"], count_cutoff=r["count_cutoff"],
                         cutoff_removed_pixels=r["cutoff_removed_pixels"], saturation_masked_pixels=r["saturation_masked_pixels"],
                         raw_sha256=r["raw_sha256"], header_sha256=r["header_sha256"],
                         selected_counts=r["selected_counts"], valid_pixel_count=r["valid_pixel_count"]))
    (out / "chunk_00000.json").write_text(json.dumps(dict(rows=rows), indent=1) + "\n", encoding="utf-8", newline="\n")
    chunk = dict(file="chunk_00000.npz", sha256=hashlib.sha256((out / "chunk_00000.npz").read_bytes()).hexdigest(),
                 metadata_sha256=hashlib.sha256((out / "chunk_00000.json").read_bytes()).hexdigest())
    run = dict(status="completed", input_source=source, label=lab,
               sample=t["sample"], substrate=t["substrate"], mode=t["mode"], nominal_nm=t["nominal_nm"],
               reference=t.get("reference"), note=t.get("note"), frames=len(rows), chunks=[chunk],
               geometry=geometry, alpha_i_note=alpha_i_note or "alpha_i_deg 0.40 is the binning coordinate; the inferred incidence "
               "0.383 +- 0.01 deg enters the model as eps (DATA_REGISTER section 17)",
               metal="Au", model_metal="Au", pixmask_sha256=marker["sha256"],
               saturation_rule=">= 0.5 x Count_cutoff, 5x5 dilation, per frame",
               photon_closure=dict(total_counts=float(Y.sum()), total_pixel_exposure=float(E.sum())),
               source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), raw_readonly=True, raw_saved=False)
    (out / "run.json").write_text(json.dumps(run, indent=1) + "\n", encoding="utf-8", newline="\n")
    (out / "membership.json").write_text(json.dumps({r["frame"]: i for i, r in enumerate(rows)}) + "\n", encoding="utf-8")
    (data / lab / "pixmask.json").write_text(json.dumps(dict(marker, label=lab), indent=1) + "\n", encoding="utf-8", newline="\n")
    return rows, float(Y.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=None)
    ap.add_argument("--data", default=str(ROOT / "_scratch" / "iam_data_static_m2"))
    ap.add_argument("--zips", default=None)
    ap.add_argument("--sdd", type=float, default=None, help="SDD (mm) override for the sensitivity caches")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    table = json.loads((ROOT / "configs" / "static_frames.json").read_text(encoding="utf-8"))["labels"]
    geometry = json.loads((ROOT / "configs" / "geometry_static_p03_2026.json").read_text(encoding="utf-8"))
    geometry = {k: v for k, v in geometry.items() if not k.startswith("_") and k != "det_shape"}
    if a.sdd:
        geometry["sdd_mm"] = float(a.sdd)
    data = Path(a.data); data.mkdir(parents=True, exist_ok=True)
    msha = hashlib.sha256(MASK.read_bytes()).hexdigest()
    marker = dict(file=MASK.name, path=str(MASK), sha256=msha)
    (data / "pixmask.json").write_text(json.dumps(marker, indent=1) + "\n", encoding="utf-8", newline="\n")
    src_grid = ROOT / "_scratch" / "iam_data_m1" / "native_grid.npz"
    if src_grid.exists() and not (data / "native_grid.npz").exists():
        shutil.copy(src_grid, data / "native_grid.npz")
    labels = a.labels.split(",") if a.labels else sorted(table)
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(geometry, a.zips)) as pool:
        for lab in labels:
            t = table[lab]
            res = list(pool.map(_frame, [(t["zip"], f["member"]) for f in t["frames"]]))
            rows, total = write_cache(data, lab, t, res, geometry, marker, f"{a.zips}/{t['zip']} (members read in memory)")
            st = {}
            for r in rows:
                st[r["state"]] = st.get(r["state"], 0) + 1
            print(lab, len(rows), "frames", st, "counts %.3g" % total, "sat px max", max(r["saturation_masked_pixels"] for r in rows), flush=True)


if __name__ == "__main__":
    main()
