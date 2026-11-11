# Third-party notices

This file lists what in the GIMaP (Grazing-Incidence Mapping Package) repository and in its Windows and macOS
portable package is not GIMaP's own work, the licence it comes under, and where it comes from.
Packaging coverage updated: 2026-11-11. The dependency table below records its original audit date;
each package's `licenses/README.txt` identifies its actual dependencies and versions.

## 1. GIMaP itself

- GIMaP's own code and documentation are released under the **MIT License**: see [LICENSE](LICENSE)
  (Copyright (c) 2025–2026 Yufeng Zhai).
- The MIT License covers the files of this repository except those named in sections 2 and 5, and except
  the frame content in the manual screenshots described at the end of section 2. These keep the licences
  and terms given there.
- No third-party source code is known to be copied into `src/`. The libraries GIMaP uses are installed
  from PyPI and conda-forge (`requirements.txt`), not copied into the repository.

## 2. Data in the repository under other licences

The public test data in `tests/data/external/` are not GIMaP's data. Keep their licences when you
redistribute them. `tests/data/external/README.md` gives their sources and geometry.

### GALAXI GISAXS frame (GPL-3.0)

- **File**: `tests/data/external/gisaxs_galaxi/galaxi_data.tif` (GISAXS, GALAXI at Forschungszentrum
  Jülich, Pilatus 1M).
- **Source**: the test data of BornAgain, release r23:
  [galaxi_data.tif.gz](https://jugit.fz-juelich.de/mlz/bornagain/-/raw/r23/testdata/scatter2d/galaxi_data.tif.gz),
  from the [BornAgain project](https://www.bornagainproject.org).
- **Licence**: GNU General Public License version 3 (GPL-3.0), as BornAgain:
  [gnu.org/licenses/gpl-3.0](https://www.gnu.org/licenses/gpl-3.0.html).
- **Changes**: decompressed from `galaxi_data.tif.gz`. No other change is recorded.
- `galaxi_bornagain.poni` in the same folder is GIMaP's own file. GIMaP wrote it from the geometry that
  BornAgain's example `fit/scatter2d/experiment_at_galaxi` documents.

### Silver behenate calibration image (pyFAI test images)

- **Files**: `tests/data/external/saxs_agbh_pyfai/Pilatus1M.edf` and `Pilatus1M.poni` (ESRF BM26,
  Pilatus 1M).
- **Source**: the pyFAI test images,
  [silx.org/pub/pyFAI/testimages](http://www.silx.org/pub/pyFAI/testimages/).
- **Licence**: `tests/data/external/README.md` records "MIT (pyFAI)". pyFAI itself is released under
  the MIT License. Whether the files on the test-image server come under the same licence is not stated
  there, so this is still **to be confirmed**.
- **Changes**: none recorded.

### pygid usage examples (CC BY 4.0)

- **Files**: six of the 25 files of the record:
  - `tests/data/external/giwaxs_p08_mapi/`: `S121_MAI_A2_00841.tif`, `LaB6_2021_12_DESY_P08.poni` and
    `S121_MAI_A2_metadata.yaml`;
  - `tests/data/external/giwaxs_xeuss_pbte/`: `PbTe_3010_r1_PT_1_10_0_00028.edf`, `mask_PbTe_3010.edf`
    and `AgBe_250403_XENOCS.poni`.
- **Creators**: Ainur Abukaev, Constantin Völter, Mikhail Romodin, Sebastian Schwartzkopff,
  Florian Bertram, Oleg Konovalov, Alexander Hinderhofer, Dmitry Lapkin and Frank Schreiber.
- **Title**: "pygid: Fast Preprocessing of Grazing Incidence Diffraction Data (GID). Usage examples."
  Zenodo, published 2025-10-28.
- **DOI**: [10.5281/zenodo.17466183](https://doi.org/10.5281/zenodo.17466183).
- **Licence**: Creative Commons Attribution 4.0 International (CC BY 4.0):
  [creativecommons.org/licenses/by/4.0](https://creativecommons.org/licenses/by/4.0/).
- **Changes**: none. The six files are copied unchanged: their MD5 checksums are the ones the Zenodo
  record lists (checked 2026-11-01). Only the two folder names are GIMaP's.
- Cite them as: Abukaev, A., Völter, C., Romodin, M., Schwartzkopff, S., Bertram, F., Konovalov, O.,
  Hinderhofer, A., Lapkin, D. & Schreiber, F. (2025). *pygid: Fast Preprocessing of Grazing Incidence
  Diffraction Data (GID). Usage examples.* Zenodo. https://doi.org/10.5281/zenodo.17466183

### The authors' own example data

- `TestSAXSdata/`: Pilatus CBF frames, Lambda NeXus module files and 1D curves measured by the GIMaP authors.
  They are GIMaP's own data and are released with it under the MIT License (`LICENSE`).

### Other example data

The following files come with no licence or source of their own. Their provenance is **to be confirmed
by the authors**:

- `utils/ML_Fitting_1D_GISAXS/Training/Cut_Data_unmasked.txt`: a 1D curve used by the research code;
- `WAXS/n4_glass_0cl_air_0_vd_00001-00002.jpg`: an image the program does not use.

### Manual screenshots

The screenshots in `docs/manual_images/en/` and `docs/manual_images/zh/` (two copies with the same names,
also shown in `README.md`) show some of these example frames and what GIMaP computed from them. GIMaP's
interface in them is under the MIT License. The frames, and the images and curves made from them, keep the
licence and attribution of their source above:

- `08_gisaxs_results.png`, `11_fitting_single.png`: the GALAXI frame and its cut (GPL-3.0);
- `02_analyze_overview.png`, `04_geometry_dialog.png`, `05_mask_step.png`, `06_cuts_cake.png`,
  `07_giwaxs_results.png`, `23_analyze_menu.png`: `S121_MAI_A2_00841.tif` of the pygid usage examples
  (CC BY 4.0; cite as given above; adapted: rendered, masked and integrated by GIMaP);
- `03_calibration.png`: the pyFAI silver behenate image (licence to be confirmed);
- `15_format_converter.png`, `16_xrr.png`, `20_prediction.png`, `22_sputter.png`: frames of
  `TestSAXSdata/` (the authors' own data, MIT License).

The other screenshots show synthetic data or no data.

## 3. The Windows portable package

The portable package `GIMaP-<version>-windows-x64-portable.zip` holds GIMaP's files and, in `runtime\`,
a packed Conda environment. That environment has Python 3.10 from conda-forge and the packages of
`requirements.txt` from PyPI. `tools/build_portable_release.ps1` builds it.

**macOS packages** are built by `tools/build_macos_release.py` for Apple Silicon and Intel separately,
using `requirements-release.txt`. They contain the same application licence and third-party notices,
plus the dependency licences collected from their own environment. Their Python environment is in
`runtime.tar.gz`, extracted into the user's runtime cache at first launch; Python packages are under
`lib/python3.10/site-packages/`. The Windows paths and DLL examples below describe the Windows package.
Use the macOS package's `licenses/README.txt` and `pip-freeze.txt` for its components and versions.
The third-party licence terms are retained on both platforms; they are not replaced by GIMaP's MIT licence.

### The licences/ folder of the package

At build time, `tools/collect_licenses.py` writes a `licenses\` folder into the package. It contains:

- `README.txt`: every Python distribution and every Conda package in `runtime\`, with its version, the
  licence its metadata declares, and its licence files;
- the licence files of each of them (`python\<name>-<version>\`, `conda\<name>-<version>\`);
- a copy of this file and of GIMaP's `LICENSE`.

A wheel that declares the GPL-3.0, the LGPL-3.0 or the Apache-2.0 but carries no text of it gets that
text, copied from another package. PyQt5 5.15.11 is one: it gets the GPL-3.0 text. Other GPL and LGPL
versions are not checked for a full text: the builder lists them for a check by hand. `README.txt` names where each text was
copied from. The versions in `licenses\README.txt` are the ones that were packed. The table below names
the components, not their exact versions.

### GPL-3.0 and LGPL-3.0 components

- **PyQt5** is released under the GPL-3.0 (or a commercial licence from Riverbank Computing).
  **BornAgain** 24.1 is released under the GPL-3.0 or later. Because the package contains these two
  components, **the portable package is distributed under the terms of the GPL-3.0 for those parts.**
  GIMaP's own files stay under the MIT License, which is compatible with the GPL-3.0.
- **Qt** 5.15 (the `PyQt5-Qt5` wheel) is released under the LGPL-3.0. Its libraries are separate DLL
  files in `runtime\Lib\site-packages\PyQt5\Qt5\bin`, and can be replaced by another build of Qt 5.15.
- `opencv-python` contains FFmpeg (`cv2\opencv_videoio_ffmpeg*.dll`, LGPL-2.1), and `fabio` contains
  `ccp4_pack` (LGPL-3.0 or later). Both are separate libraries inside those packages.
- The full texts are in `licenses\` (GPL-3.0, for example, in the folders of BornAgain and PyQt5).

### Where the corresponding source is

| Component | Source |
|---|---|
| GIMaP | [github.com/zyffcc/gisaxs_gui](https://github.com/zyffcc/gisaxs_gui), at the tag `v<version>` of the release, and the Zenodo record of that release where one exists |
| PyQt5 | the source distribution `PyQt5-<version>.tar.gz` on [pypi.org/project/PyQt5](https://pypi.org/project/PyQt5/#files); [Riverbank Computing](https://www.riverbankcomputing.com/software/pyqt/) |
| PyQt5-sip | [pypi.org/project/PyQt5-sip](https://pypi.org/project/PyQt5-sip/#files) |
| Qt 5.15 | [download.qt.io/archive/qt/5.15](https://download.qt.io/archive/qt/5.15/); [code.qt.io](https://code.qt.io/) |
| BornAgain 24.1 | [jugit.fz-juelich.de/mlz/bornagain](https://jugit.fz-juelich.de/mlz/bornagain) |
| FFmpeg (in opencv-python) | [ffmpeg.org](https://ffmpeg.org/download.html); the build scripts in [github.com/opencv/opencv-python](https://github.com/opencv/opencv-python) |
| fabio | [github.com/silx-kit/fabio](https://github.com/silx-kit/fabio) |

Take the exact versions from `licenses\README.txt` of the package you received.

### What the package bundles

The licences below are the ones the packages declare in their metadata (Python distributions in
`*.dist-info`, Conda packages in `conda-meta`). They were read from the development environment on
2026-11-01.

| Component | Licence | Source |
|---|---|---|
| Python 3.10 (conda-forge) | PSF License (Python-2.0) | [python.org](https://www.python.org/downloads/source/); [conda-forge/python-feedstock](https://github.com/conda-forge/python-feedstock) |
| PyQt5 | GPL-3.0 (or Riverbank commercial) | see above |
| PyQt5-Qt5 (Qt 5.15) | LGPL-3.0 | see above |
| PyQt5-sip | BSD-2-Clause | see above |
| BornAgain 24.1 | GPL-3.0-or-later | see above |
| vtk (comes with BornAgain) | BSD-3-Clause | [gitlab.kitware.com/vtk/vtk](https://gitlab.kitware.com/vtk/vtk) |
| TensorFlow (`tensorflow-intel` on Windows), Keras, TensorBoard | Apache-2.0 | [github.com/tensorflow/tensorflow](https://github.com/tensorflow/tensorflow); [github.com/keras-team/keras](https://github.com/keras-team/keras); [github.com/tensorflow/tensorboard](https://github.com/tensorflow/tensorboard) |
| absl-py, grpcio, flatbuffers, ml-dtypes, libclang, opt-einsum, gast, google-pasta, astunparse, wrapt, termcolor (come with TensorFlow) | Apache-2.0, BSD or MIT | their PyPI pages |
| protobuf | BSD-3-Clause | [github.com/protocolbuffers/protobuf](https://github.com/protocolbuffers/protobuf) |
| NumPy, SciPy | BSD-3-Clause; their wheels also contain OpenBLAS with LAPACK (BSD-3-Clause), the GCC runtime library (GPL-3.0 with the GCC Runtime Library Exception) and, in NumPy, libquadmath (LGPL-2.1-or-later) | [github.com/numpy/numpy](https://github.com/numpy/numpy); [github.com/scipy/scipy](https://github.com/scipy/scipy) |
| Matplotlib | Matplotlib License (PSF-based) | [github.com/matplotlib/matplotlib](https://github.com/matplotlib/matplotlib) |
| contourpy, kiwisolver, cycler, fonttools, pyparsing, python-dateutil, packaging, six (come with Matplotlib) | BSD, MIT or Apache-2.0 | their PyPI pages |
| Pillow | MIT-CMU | [github.com/python-pillow/Pillow](https://github.com/python-pillow/Pillow) |
| pyqtgraph | MIT | [github.com/pyqtgraph/pyqtgraph](https://github.com/pyqtgraph/pyqtgraph) |
| opencv-python | Apache-2.0, with FFmpeg (LGPL-2.1) | [github.com/opencv/opencv-python](https://github.com/opencv/opencv-python) |
| h5py, with the HDF5 library | BSD-3-Clause; HDF5: BSD-style | [github.com/h5py/h5py](https://github.com/h5py/h5py); [github.com/HDFGroup/hdf5](https://github.com/HDFGroup/hdf5) |
| hdf5plugin | MIT; the compression filters it bundles keep their own licences | [github.com/silx-kit/hdf5plugin](https://github.com/silx-kit/hdf5plugin) |
| fabio | MIT, with `ccp4_pack` under LGPL-3.0 or later | [github.com/silx-kit/fabio](https://github.com/silx-kit/fabio) |
| PyYAML | MIT | [github.com/yaml/pyyaml](https://github.com/yaml/pyyaml) |
| psutil, threadpoolctl | BSD-3-Clause | [github.com/giampaolo/psutil](https://github.com/giampaolo/psutil); [github.com/joblib/threadpoolctl](https://github.com/joblib/threadpoolctl) |
| anthropic | MIT | [github.com/anthropics/anthropic-sdk-python](https://github.com/anthropics/anthropic-sdk-python) |
| openai | Apache-2.0 | [github.com/openai/openai-python](https://github.com/openai/openai-python) |
| httpx, httpcore, h11, anyio, sniffio, idna, distro, jiter, pydantic, pydantic-core, annotated-types, typing-extensions (come with anthropic and openai) | BSD, MIT, Apache-2.0 or PSF-2.0 | their PyPI pages |
| certifi | MPL-2.0 | [github.com/certifi/python-certifi](https://github.com/certifi/python-certifi) |
| tqdm (comes with openai) | MPL-2.0 and MIT | [github.com/tqdm/tqdm](https://github.com/tqdm/tqdm) |
| OpenSSL | Apache-2.0 | [github.com/openssl/openssl](https://github.com/openssl/openssl) |
| Tcl/Tk, SQLite, zlib, xz (liblzma), bzip2, libffi, libexpat, tzdata, ca-certificates (conda-forge, with Python) | TCL, public domain, Zlib, 0BSD, bzip2, MIT, MIT, public domain, ISC (as conda-forge declares them) | the feedstocks under [github.com/conda-forge](https://github.com/conda-forge) |
| Microsoft Visual C++ runtime and Universal C Runtime (`vc14_runtime`, `vcomp14`, `ucrt`) | Microsoft's redistribution terms; no source | the conda-forge packages `vc14_runtime` and `ucrt` |

- certifi and tqdm are under the MPL-2.0: their source files are in the package, unchanged, in
  `runtime\Lib\site-packages`.
- The package also holds what `requirements.txt` pulls in beyond this table, for example pip and
  setuptools. `licenses\README.txt` lists them all.

## 4. Parts users install themselves

None of these is in the repository or in the portable package.

- **PyTorch** (BSD-3-Clause, [github.com/pytorch/pytorch](https://github.com/pytorch/pytorch)) runs the
  networks of the Sputter workspace. The workspace installs it from PyPI on its first **Predict**
  (`requirements-sputter.txt`). In a source folder, `Start_GIMaP.bat` installs it.
- **StarDist** 0.9.2 and **CSBDeep** 0.8.2 (both BSD-3-Clause,
  [github.com/stardist/stardist](https://github.com/stardist/stardist),
  [github.com/CSBDeep/CSBDeep](https://github.com/CSBDeep/CSBDeep)) run with TensorFlow 2.15.1 in a
  separate Python environment, made from `requirements-sem-stardist.txt`. SEM Particle Analysis starts
  them in a subprocess. GIMaP neither ships nor downloads StarDist model weights. A model you use keeps
  the licence its publisher gives it.

## 5. Research code and trained networks in the repository

These parts are research code and trained networks bundled with GIMaP; the program runs the research
code in subprocesses. Their author, **Yufeng Zhai**, confirmed on 2026-11-10 that the research code and bundled networks
are released under the **MIT License**. Each bundle includes `LICENSE`; `SOURCE.json` records authorship
where present. This confirmation does not change the separate licences of third-party dependencies.

- **1D Predict Workflow V5** (`modules/Fitting_1D_Model/Workflow_v5/`): code and trained network weights
  (`*.h5`, `*.npz`) for Tools ▸ 1D Predict. They propose candidate fits of a 1D GISAXS curve with up to
  four sphere or cylinder components.
  - It was integrated on 2026-09-22 from a research source archive. `IMPORT_MANIFEST.json` records the
    SHA-256 of that archive and of every imported file.
  - Before 1D Predict runs, the program checks the weights against `conditional_v2/conditional_fast_manifest_v2.json`
    and `stable_blue_rc_v1/MANIFEST.json`.
  - The model card is `conditional_v2/MODEL_CARD_zh.md` (in Chinese). It states that the validation used
    simulated data.
- **Earlier 1D models** (`modules/Fitting_1D_Model/k1_k2_k3_k4_phys/`, `modules/Fitting_1D_Model/saved_model_K3K4/`)
  and the 1D fitting research code in `utils/ML_Fitting_1D_GISAXS/`. `saved_model_K3K4/` and the research
  code were added to the repository on 2026-06-08 with the first 1D Predict; `k1_k2_k3_k4_phys/` was added
  on 2026-08-11.
- **Sputter toolkit** (`utils/sputter_toolkit/`): the research code `sputter_struct_v1` with two PyTorch
  network checkpoints (`experts/polymer/model/best.pt`, `experts/TS/model/best.pt`). It is bundled from
  the source commit `19d775bc`, with the local patches that `SOURCE.json` lists.
  - `MANIFEST.json` and `verify.py` check every file.
  - `SOURCE.json` records Yufeng Zhai as the author and MIT as the licence.
  - The published values it compares with are cited, with their DOIs, in
    `configs/literature_reference_v1.json`.
- **fit1d kit** (`utils/gisaxs1d_gui_kit/`): the research code of the fit1d method (gisaxs1d_gui_kit
  1.0.0, 2026-10-13) with two numpy networks (`models/general`, `models/au_np_si`) and packed form-factor
  tables. It runs Fitting ▸ **Predict parameters (fit1d)** and the method *fit1d* of 1D Predict, imported in
  a job process only. It was added to the repository on 2026-11-02 at the authors' request.
  - Only its runtime files are bundled. `SOURCE.json` lists what was left out (example measurements, results
    on measured data, research notes) and the few text changes; the networks and the tables are unchanged.
  - `FILES_SHA256.json` holds the SHA-256 of every file; `tests/test_fit1d_kit_bundle.py` checks them.
  - The network `au_np_si` was adapted without labels to measured curves, which are not distributed.
- The **2D Prediction** modules (`modules/*/module.yaml`, `preprocess.py`) are configuration and
  preprocessing only. No 2D network weights are shipped.

## 6. Questions

Write to [yufeng.zhai@desy.de](mailto:yufeng.zhai@desy.de) about licences, sources or attribution, or
open an issue on [GitHub](https://github.com/zyffcc/gisaxs_gui/issues).
