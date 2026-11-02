"""Where the kit finds its code, networks and tables.

Packaged layout (what pack_gui_kit.py writes; KIT = the folder that contains gisaxs1d_kit/):
    KIT/app/research/...        the fitting code (vendored)
    KIT/models/<name>/          weights_numpy.npz + model_config.json, and KIT/models/models.json (catalogue)
    KIT/tables_packed/*.fft     form-factor tables (unpacked once into the table cache), KIT/tablepack.py
Development layout (this repository): work/gui_kit/gisaxs1d_kit, code in work/portable_project, networks in
work/models, tables in the user's cache.
The environment variable GISAXS1D_KIT_HOME overrides KIT, GISAXS1D_KIT_CODE the code directory; GISAXS1D_TABLES
overrides the table cache (default ~/.cache/gisaxs1d_tables, see research.fast_inference.tables).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

KIT_VERSION = 'gisaxs1d_kit 1.0.0 (2026-10-13)'
# development catalogue: name -> (directory under work/models, title, description)
DEV_MODELS = {
    'general': ('cond_v2_ft4', 'General (physics-trained)',
                'Proposal network trained on synthetic curves with the physics (chi2) loss; the default.'),
    'au_np_si': ('cond_v2_ftr1', 'Au nanoparticles on Si (adapted)',
                 'The general network adapted without labels to measured in-situ curves of Au nanoparticles on Si.'),
    'legacy': ('cond_v2_a', 'cond_v2_a (first network)', 'The first cond_v2 network (parameter loss only).'),
}


def kit_home() -> Path:
    env = os.environ.get('GISAXS1D_KIT_HOME')
    return Path(env).expanduser().resolve() if env else Path(__file__).resolve().parents[1]


def app_dir() -> Path:
    """Directory to put on sys.path for `import research` (GISAXS1D_KIT_CODE overrides it: packaging, tests)."""
    env = os.environ.get('GISAXS1D_KIT_CODE')
    if env:
        return Path(env).expanduser().resolve()
    home = kit_home()
    if (home/'app'/'research').is_dir():
        return home/'app'
    dev = home.parent/'portable_project'
    if (dev/'research').is_dir():
        return dev
    raise FileNotFoundError(f'the fitting code was not found: neither {home/"app"/"research"} nor {dev/"research"} exists')


def packed_tables() -> Path | None:
    path = kit_home()/'tables_packed'
    return path if path.is_dir() else None


def model_catalogue() -> dict:
    """{'default': name, 'models': {name: {'dir': absolute path, 'title': ..., 'description': ...}}}."""
    home = kit_home()
    listing = home/'models'/'models.json'
    if listing.is_file():
        raw = json.loads(listing.read_text(encoding='utf-8'))
        models = {name: dict(item, dir=str((home/'models'/item['dir']).resolve())) for name, item in raw['models'].items()}
        return dict(default=raw.get('default') or next(iter(models)), models=models)
    base = home.parent/'models'
    models = {name: dict(dir=str((base/folder).resolve()), title=title, description=text)
              for name, (folder, title, text) in DEV_MODELS.items() if (base/folder/'weights_numpy.npz').is_file()}
    if not models:
        raise FileNotFoundError(f'no network found under {home/"models"} or {base}')
    return dict(default='general' if 'general' in models else next(iter(models)), models=models)


def model_dir(name: str | None) -> tuple[str, Path]:
    """(catalogue name, directory) of a network given by catalogue name, or by a directory path."""
    catalogue = model_catalogue()
    if not name:
        name = catalogue['default']
    if name in catalogue['models']:
        return name, Path(catalogue['models'][name]['dir'])
    path = Path(name).expanduser()
    if (path/'weights_numpy.npz').is_file():
        return path.name, path.resolve()
    raise ValueError(f'unknown network {name!r}: use one of {sorted(catalogue["models"])} or a directory with weights_numpy.npz')
