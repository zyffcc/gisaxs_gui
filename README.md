# GIMaP

GIMaP (Grazing-Incidence Mapping Package) is a desktop application for GISAXS/GIWAXS data visualization, analysis, fitting, and machine-learning-assisted workflows. It is built with PyQt and is currently an early pre-release / beta scientific GUI.

## Overview

GIMaP provides a single desktop interface for working with grazing-incidence scattering data. The current codebase includes tools for detector image viewing, cut-and-fit workflows, trained-model prediction, AI-assisted 1D fitting, trainset generation, classification, and an embedded WAXS/GIWAXS in-situ processing page.

The application entry point is `main.py`.

## Key Features

- **Cut & Fitting**: load GISAXS detector images, inspect 2D data, define cuts, fit 1D curves, and run least-squares refinement.
- **AI-assisted fitting**: generate model candidates, rank predicted candidates, refine selected candidates, and export prediction/refinement results.
- **GIMaP Predict**: run configured trained-model prediction modules on single files or multi-file batches.
- **Model import**: import or select trained models through module configuration files.
- **Trainset Build**: generate synthetic or simulated training data through an experimental GUI workflow (Not implemented).
- **Classification**: import datasets, preview data, reduce dimensions, train classifiers, and save/load classification models.
- **WAXS/GIWAXS**: embedded in-situ processing page for `.nxs`, `.tif`, and `.tiff` detector data, including display, masking, geometry, cut, 1D integration, and batch export controls.

## Installation

### Windows Installer

* Visit the https://github.com/zyffcc/gisaxs_gui/releases page.
* Look for the latest version marked tag.
* Download the `00_Download_and_Install_GIMaP-*.bat`.
* Place it in an empty folder.
* Double-click the file. The installer will automatically download, verify, join, and extract all package parts.


### Running from Source

Recommended source environment: Windows with Python 3.10 or 3.11.

```powershell
cd gisaxs_gui

python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python main.py
```

If PowerShell blocks activation in the current terminal session:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Conda can also be used if preferred:

```powershell
conda create -n GUI python=3.11 -y
conda activate GUI
python -m pip install -r requirements.txt
python main.py
```

#### BornAgain on Windows and macOS

GIMaP requires BornAgain 24.1. On Windows (and Linux), BornAgain provides a PyPI
wheel for supported Python versions, so `requirements.txt` installs it directly.

BornAgain does not publish a prebuilt macOS wheel on PyPI. On macOS, first install
the official Homebrew formula:

```bash
brew tap mlz/homebrew https://jugit.fz-juelich.de/mlz/homebrew/
brew install mlz/homebrew/bornagain@24.1
bornagain_info
```

The Homebrew wheel must have the same CPython ABI tag as the GIMaP environment:
for example, a `cp314` wheel cannot be installed into Python 3.10. Do not force an
ABI-mismatched wheel; use a matching project wheel or build BornAgain 24.1 against
the same Python interpreter. See [the development guide](docs/development.md) for
the complete platform-specific setup and verification steps.


### GUI Layout Too Large for Small Screen

The GUI contains dense scientific controls. Use a larger display, maximize the window, or adjust system scaling if controls are clipped. For the best user experience, a 1080p or higher-resolution screen is recommended.

## Basic Workflow

1. Start the application with `python main.py`.
2. Choose a workspace from the left navigation panel.
3. Load a GISAXS/GIWAXS file or a folder of files.

## Cut & Fitting Workflow

1. Open the **Cut & Fitting** page.
2. Import a GISAXS detector image or supported data file.
3. Adjust display options such as scale, intensity range, and log display.
4. Configure detector and cut parameters.
5. Generate a 1D curve from the selected region.
6. Select a fitting model and parameter bounds.
7. Run manual fitting or Auto Refine. Auto Refine uses least-squares optimization.
8. Export plots, fitting curves, and fitting results.

The current fitting code includes sphere, cylinder, vertical cylinder components.

## AI Auto Fitting

The default AI fitting model is `modules/Fitting_1D_Model/k1_k2_k3_k4_phys`. It is a single K1-K4 slot model with an explicit optional-D head, relational D constraints, and a delayed physical-reconstruction loss. Its `manifest.json` records the checkpoint checksum, supported K values, required inputs/outputs, training state, and remote provenance. Model discovery validates this contract and loads the checkpoint lazily; the portable SavedModel is used automatically when the Python-version-specific `.keras` Lambda bytecode cannot be restored.

The Cut & Fitting page opens one AI Auto Fitting workspace with the following flow:

1. Prepare the current ROI/Yoneda 1D curve and its uncertainty.
2. Select the versioned K1-K4 model and a Fast, Balanced, or Exhaustive profile.
3. Build the same geometry-aware constraint payload used by prediction and refinement.
4. Run one neural proposal pass, sample posterior parameter modes at multiple normalized scales, and verify every candidate with the physical forward model.
5. Optionally refine the best modes while optimizing structural parameters and global background/resolution/scale parameters together. GUI runs rank with a hybrid log/relative score so narrow linear-intensity overshoots cannot win solely by touching few q points.
6. Re-rank the verified modes with the simpler-K prior, display constraint violations explicitly, and automatically load/plot the valid candidate selected in the results table.

Balanced is the default. Editing a profile parameter changes the workspace state to Custom; selecting a named profile again restores its defaults. Runs execute in a separate process and support progress, cancellation, a reproducible random seed, and an optional time budget. The in-situ workflow uses the same pipeline and profile definitions.

| Profile | Candidates | Sampling scales | Refine modes | Max evaluations | q stride | Full per-K comparison | Time budget |
|---|---:|---|---:|---:|---:|---|---:|
| Fast | 48 | 1 | 0 | 24 | 8 | No | 30 s |
| Balanced | 192 | 0.5, 1, 2 | 2 | 40 | 4 | No | 180 s |
| Exhaustive | 512 | 0.5, 1, 2, 4 | 6 | 120 | 1 | Yes | None |

Reference CPU benchmark on `TestSAXSdata/test_1d_data.dat` (Windows, Python 3.10, TensorFlow 2.15.1, seed 123, cold process/model load, default `max_diameter` constraints): Fast 15.76 s / best logRMSE 0.601; Balanced 39.47 s / 0.0699; Exhaustive 151.57 s / 0.0450. Each profile returned 20 valid parameter modes. Runtime depends strongly on CPU, curve length, geometry mix, and whether TensorFlow/model loading is already warm.

The random-cylinder forward model uses the exact cylindrical Bessel function `J1`. Its independent radius/height averages are factorized and vectorized, avoiding the former `n_R*n_h*n_orient` Python loop during candidate verification and numerical refinement.

Checkpoints trained before this forward-model revision do not contain the corresponding `forward_model.random_cylinder_radial_amplitude` metadata. They remain loadable and their candidates are re-scored/refined with the corrected NumPy physics, but the random-cylinder proposal head should be retrained on a regenerated dataset before its posterior probabilities are interpreted quantitatively.

The default physical constraints are:

- all parameters are finite and non-negative;
- sphere and random-cylinder distribution widths are absolute and satisfy `0 < sigma <= 0.9 * size`;
- vertical-cylinder `sigma_R` is fractional and satisfies `0 < sigma_R <= 0.9`;
- `D=0, sigma_D=0` is a valid no-D state;
- when D is present, sphere and vertical cylinder require `D > margin * 2R`;
- a randomly oriented cylinder conservatively requires `D > margin * sqrt((2R)^2 + h^2)`;
- multi-component max/mean spacing rules use the maximum or arithmetic mean of those geometry-specific exclusion sizes. The default margin is 1.001.

## GIMaP Predict Workflow

1. Open the **GIMaP Predict** page.
2. Choose **Single File** or **Multi Files** mode.
3. Select a GISAXS file or input folder.
4. Set stack, range, and step/every options when working with stacked or batch data.
5. Select a prediction module.
6. Confirm or edit the module configuration.
7. Import or load the trained model.
8. Run prediction.
9. Review outputs in the result tabs.
10. Export the current result or all multi-file results.

## Model Import

Prediction modules are configured through `module.yaml` files under `modules/`. Existing module configurations include fields such as:

- `id`
- `name`
- `framework`
- `version`
- `model.model_path`
- `preprocess.entry`
- `preprocess.steps`
- `preprocess.params`
- `io.input_type`
- `io.input_shape`
- `outputs`

AI fitting models are discovered from fitting-model folders under `modules/`, including `modules/Fitting_1D_Model`. **For normal users, only the model path should be changed when importing or replacing a trained model. Different models may require different preprocessing settings, so please make sure the selected model and preprocessing workflow match.** 


## System Requirements

- Windows is the primary development and usage target.
- Python 3.10 or 3.11 is recommended for source execution.
- A display large enough for scientific control panels is recommended.
- CPU execution is supported. GPU TensorFlow setups are not configured by this repository and must be installed separately if required.

## Development

Install the development dependencies and run the unified verification command:

```bash
python -m pip install -r requirements-dev.txt
python tools/check.py
```

This runs pytest, a real offscreen five-workspace startup/close smoke check, and
the deliberately narrow Ruff baseline without formatting unrelated code.
Configuration is in `pyproject.toml`; details are in
[docs/development.md](docs/development.md).

## Dependencies

Dependencies are listed in `requirements.txt`:

- numpy>=1.24,<2.0
- scipy>=1.10
- matplotlib>=3.7
- BornAgain==24.1 on Windows and Linux; installed separately on macOS
- PyQt5>=5.15
- opencv-python>=4.8
- h5py>=3.9
- fabio>=2023.4.0
- tqdm>=4.65
- umap-learn>=0.5.5
- scikit-learn>=1.3
- tensorflow-intel>=2.15,<=2.16; platform_system=="Windows"
- tensorflow>=2.15,<=2.16; platform_system!="Windows"
- PyYAML>=6.0


## Feedback and Contact

For feedback, bug reports, or collaboration questions, contact:

[yufeng.zhai@desy.de](mailto:yufeng.zhai@desy.de)


## License

This project is released under the MIT License. See [LICENSE](LICENSE) for the full license text.
