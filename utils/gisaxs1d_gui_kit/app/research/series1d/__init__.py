"""series1d: tracking fits of in-situ series of 1D GISAXS Yoneda cuts (PLAN_zh.md M3).

    python3 -m research.series1d --files SERIES.npz --recipe RECIPE.json --mode review --out DIR
    from research.series1d import run
    summary = run('series.npz', recipe_dict, mode='batch', workers=6, out='out')

Built on research.fit1d (first frame and fallbacks) and the cut tools of research.real_data. See
README_zh.md (Chinese guide), recipe (recipe keys), tracker (per-frame algorithm, on/off states, live
backfill and re-tracking), runner (modes, batch stitching, resume), annotate (chi2 baseline, change points,
continuity smoothing), outputs (CSV columns, summary).
"""
from research.series1d.inputs import InputError
from research.series1d.outputs import csv_columns, series_statistics
from research.series1d.recipe import RecipeError, load as load_recipe, normalise as normalise_recipe
from research.series1d.runner import VERSION, latest_rows, read_frames, run

__all__ = ['run', 'load_recipe', 'normalise_recipe', 'RecipeError', 'InputError', 'read_frames', 'latest_rows',
           'csv_columns', 'series_statistics', 'VERSION']
