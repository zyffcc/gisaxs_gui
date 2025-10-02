# utils_xy.py
# -*- coding: utf-8 -*-
"""
Robust loader for 2~3 column scattering data:
(q, I [, err]) from .txt / .dat with unknown header & delimiter.

Usage:
    from utils_xy import load_xy_any, XYData, LoadOptions
    data = load_xy_any("path/to/file.txt", LoadOptions(assume_poisson_err=True))
"""

from __future__ import annotations
import os, re
from dataclasses import dataclass
from typing import Optional, List, Tuple
import numpy as np
from scipy.signal import savgol_filter


# ---------- Public data types ----------

@dataclass
class XYData:
    q: np.ndarray
    I: np.ndarray
    err: Optional[np.ndarray]  # None if not provided & not synthesized
    meta: dict                 # path, delimiter, start_line, bad_lines, has_error, etc.

@dataclass
class LoadOptions:
    assume_poisson_err: bool = False   # if only 2 cols: err = sqrt(max(I,0))
    allow_extra_cols: bool = True      # allow >3 cols, take first 3
    comment_prefixes: tuple = ("#", "%", ";", "//")
    # Accept \t, ',', ';', or any whitespace; auto-detect
    encodings_try: tuple = ("utf-8", "utf-8-sig", "latin-1")

# ---------- Internal helpers ----------

_NUM_LINE = re.compile(r"""
^[\s]*                                # leading spaces
[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?   # first number
[\s,;\t]+
[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?   # second number
""", re.VERBOSE)

def _read_text(path: str, encodings: tuple) -> str:
    last_err = None
    for enc in encodings:
        try:
            with open(path, "r", encoding=enc, errors="strict") as f:
                return f.read()
        except Exception as e:
            last_err = e
    # fallback with ignoring errors
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def _looks_like_data(line: str, comments: tuple) -> bool:
    s = line.strip()
    if not s:
        return False
    if any(s.startswith(p) for p in comments):
        return False
    return bool(_NUM_LINE.match(s))

def _detect_start_and_delim(lines: List[str], comments: tuple) -> Tuple[int, str]:
    """Return (first_data_line_index, delimiter_key)"""
    start = None
    for i, ln in enumerate(lines):
        if _looks_like_data(ln, comments):
            start = i
            break
    if start is None:
        raise ValueError("未找到数据行（至少需要两列数字）。")

    sample = lines[start].strip()
    if "\t" in sample:
        delim = "\t"
    elif sample.count(",") >= 1 and (" " not in sample or sample.count(",") >= sample.count(" ")):
        delim = ","
    elif ";" in sample:
        delim = ";"
    else:
        delim = "whitespace"
    return start, delim

def _split(line: str, delim: str) -> List[str]:
    s = line.strip()
    if not s:
        return []
    if delim == "whitespace":
        return s.split()
    return s.split(delim)

def _to_float(tok: str, treat_comma_decimal: bool) -> float:
    # If delimiter is not comma and token contains exactly one comma -> treat as decimal comma
    if treat_comma_decimal and tok.count(",") == 1 and "." not in tok:
        tok = tok.replace(",", ".")
    return float(tok)

# ---------- Public API ----------

def load_xy_any(path: str, options: LoadOptions = LoadOptions()) -> XYData:
    """
    Load 2 or 3 column numeric data (q, I [, err]) with unknown header & delimiter.
    - Skips header/comments/blank lines
    - Detects delimiter: tab, comma, semicolon, or any whitespace
    - Accepts scientific notation
    - If more than 3 cols and allow_extra_cols=True -> take first 3
    - Optionally synthesizes err = sqrt(I) for 2-column data
    """
    text = _read_text(path, options.encodings_try)
    lines = text.splitlines()

    start_idx, delim = _detect_start_and_delim(lines, options.comment_prefixes)
    treat_comma_decimal = (delim != ",")  # allow "1,23" as 1.23 when not CSV

    q_list: List[float] = []
    I_list: List[float] = []
    e_list: List[float] = []
    bad_lines = 0

    for ln in lines[start_idx:]:
        s = ln.strip()
        if not s or any(s.startswith(p) for p in options.comment_prefixes):
            continue
        parts = _split(s, delim)
        if len(parts) < 2:
            bad_lines += 1
            continue
        if len(parts) > 3 and not options.allow_extra_cols:
            bad_lines += 1
            continue
        try:
            q = _to_float(parts[0], treat_comma_decimal)
            I = _to_float(parts[1], treat_comma_decimal)
            q_list.append(q)
            I_list.append(I)
            if len(parts) >= 3:
                e_list.append(_to_float(parts[2], treat_comma_decimal))
        except Exception:
            bad_lines += 1
            continue

    if not q_list:
        raise ValueError("解析失败：没有成功读取到任何 (q, I[, err]) 数据。")

    q = np.asarray(q_list, dtype=float)
    I = np.asarray(I_list, dtype=float)

    err: Optional[np.ndarray]
    if e_list:
        err = np.asarray(e_list, dtype=float)
        n = min(len(q), len(I), len(err))
        q, I, err = q[:n], I[:n], err[:n]
    else:
        err = None
        if options.assume_poisson_err:
            err = np.sqrt(np.clip(I, 0.0, None))

    # Drop non-finite rows
    mask = np.isfinite(q) & np.isfinite(I)
    if err is not None:
        mask &= np.isfinite(err)
    q, I = q[mask], I[mask]
    if err is not None:
        err = err[mask]

    if q.size < 2:
        raise ValueError("有效数据点太少（< 2）。")

    meta = {
        "path": os.path.abspath(path),
        "rows_total": len(lines),
        "data_start_line": start_idx + 1,
        "delimiter": {"\t": "tab", ",": "comma", ";": "semicolon"}.get(delim, "whitespace"),
        "bad_lines": bad_lines,
        "has_error": err is not None,
    }
    return XYData(q=q, I=I, err=err, meta=meta)

# 误差估计函数
def estimate_sigma(q, I, lowq_frac=0.2, win_frac=0.06, poly=2,
                          n_bins=25, rN=0.0, floor_scale=1.0, monotone=True):
    q = np.asarray(q, float); I = np.asarray(I, float)
    m = np.isfinite(q) & np.isfinite(I) & (q > 0)
    q, I = q[m], I[m]
    order = np.argsort(q); q, I = q[order], I[order]
    n = len(q); logq = np.log(q)

    # 等距 log q 重采样 + SavGol 平滑作为趋势
    logq_u = np.linspace(logq[0], logq[-1], n)
    Iu = np.interp(logq_u, logq, I)
    win = max(5, int(n*win_frac) | 1)
    trend = savgol_filter(Iu, win, poly)
    resid = Iu - trend

    # 低-q 残差标定 s * sqrt(I)
    cut = np.quantile(logq_u, lowq_frac)
    low = logq_u <= cut
    mad = np.median(np.abs(resid[low] - np.median(resid[low])))
    sigma_ref = 1.4826 * mad
    msqrt = np.median(np.sqrt(np.clip(Iu[low], 0, None)) + 1e-16)
    s = 0.0 if msqrt == 0 else sigma_ref / msqrt

    # 局部加性地板：分箱 MAD -> std，并可强制随 q 非减
    edges = np.linspace(0, n, n_bins + 1, dtype=int)
    centers, floor = [], []
    for b in range(n_bins):
        seg = resid[edges[b]:edges[b+1]]
        med = np.median(seg); mad = np.median(np.abs(seg - med))
        floor.append(1.4826 * mad)
        centers.append(np.median(logq_u[edges[b]:edges[b+1]]))
    floor = np.array(floor) * floor_scale
    if monotone: floor = np.maximum.accumulate(floor)
    floor_u = np.interp(logq_u, np.array(centers), floor)

    # 合成误差并映射回原网格
    Ipos = np.clip(I, 0, None)
    sigma = np.sqrt((s * np.sqrt(Ipos))**2 + floor_u**2 + (rN * I)**2)
    inv = np.empty_like(order); inv[order] = np.arange(n)
    return sigma[inv], dict(scale=s, sigma_floor=floor_u, win=win)
