# fit1d kit (bundled)

The method **fit1d** of Tools ▸ 1D Predict and of Fitting ▸ Fit: network proposals followed by a physical
fit of a 1D GISAXS or SAXS cut, with errors and the ranges of equivalent solutions. It comes with GIMaP;
nothing has to be installed.

- GIMaP finds it here (`utils/gisaxs1d_gui_kit`). The environment variable `GISAXS1D_KIT_HOME` points
  GIMaP to another copy instead.
- GIMaP imports `gisaxs1d_kit.core` and `gisaxs1d_kit.results` inside its job process only, never in
  the window's process
  (`src/gimap/features/fitting/infrastructure/adapters/fit1d_method.py`).
- It needs numpy, scipy and threadpoolctl from `requirements.txt`. It does not need TensorFlow: the
  two networks run in numpy.
- On first use the packed form-factor tables (`tables_packed/*.fft`) are unpacked once into
  `~/.cache/gisaxs1d_tables`, about 74 MB. The environment variable `GISAXS1D_TABLES` chooses another
  folder.

## What is here

| Path | What it is |
|---|---|
| `gisaxs1d_kit/` | The kit's interface: `core.fit_curve`, results tables, a worker client and a stand-alone Qt panel. |
| `app/research/` | The fitting code that `gisaxs1d_kit` imports (fit1d, series1d and what they import). |
| `models/general/` | General network, trained on synthetic curves with a physics (χ²) loss. This is the default. |
| `models/au_np_si/` | The general network adapted without labels to measured in-situ curves of Au nanoparticles on Si. |
| `tables_packed/` | Form-factor tables of the size distributions. They are packed losslessly (`tablepack.py`). |
| `FILES_SHA256.json` | The SHA-256 of every file in this folder. |
| `SOURCE.json` | Where the folder comes from, what was left out and what was changed. |

## Provenance

The source is gisaxs1d_gui_kit 1.0.0 (2026-10-13). Only the files GIMaP needs at run time are here.
Example measurements, self-tests with results on measured data, and research notes that cite measured
data were left out. The networks and the tables are byte-for-byte those of the source. `SOURCE.json`
lists their checksums and the few text patches.

The files are checked byte for byte, so git does no line-ending conversion in this folder
(`.gitattributes`). After any change, write the new checksums into `FILES_SHA256.json`.
`tests/test_fit1d_kit_bundle.py` checks the folder against that file.
