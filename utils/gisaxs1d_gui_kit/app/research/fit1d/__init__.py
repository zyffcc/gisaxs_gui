"""fit1d: unified fast-path fitting of 1D GISAXS Yoneda cuts and SAXS curves (PLAN_zh.md M2).

    from research.fit1d import fit1d, to_csv, plot, load_input
    python3 -m research.fit1d INPUT --composition sphere,vc,vc --geometry GEOMETRY.json --out DIR

See README_zh.md (Chinese usage guide), api.fit1d (arguments), controller (stages S0-S4 and state
signals), constraints (fixes/ranges in physical units), flags and schema (result format).
"""
from research.fit1d.api import fit1d, get_fitter
from research.fit1d.constraints import ConstrainedCodec, ConstraintError, parse_cli
from research.fit1d.io import load_input
from research.fit1d.schema import CSV_COLUMNS, SCHEMA, VERSION, plot, to_csv

__all__ = ['fit1d', 'get_fitter', 'ConstrainedCodec', 'ConstraintError', 'parse_cli', 'load_input', 'CSV_COLUMNS',
           'SCHEMA', 'VERSION', 'plot', 'to_csv']
