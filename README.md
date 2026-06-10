# GIMaP

GIMaP (Grazing-Incidence Mapping Package) is a desktop application for GISAXS/GIWAXS data visualization, analysis, fitting, and machine-learning-assisted workflows. It is built with PyQt and is currently an early pre-release / beta scientific GUI.

## Overview

GIMaP provides a single desktop interface for working with grazing-incidence scattering data. The current codebase includes tools for detector image viewing, cut-and-fit workflows, trained-model prediction, AI-assisted 1D fitting, trainset generation, classification, and an experimental WAXS/GIWAXS window.

The application entry point is `main.py`.

## Key Features

- **Cut & Fitting**: load GISAXS detector images, inspect 2D data, define cuts, fit 1D curves, and run least-squares refinement.
- **AI-assisted fitting**: generate model candidates, rank predicted candidates, refine selected candidates, and export prediction/refinement results.
- **GIMaP Predict**: run configured trained-model prediction modules on single files or multi-file batches.
- **Model import**: import or select trained models through module configuration files.
- **Trainset Build**: generate synthetic or simulated training data through an experimental GUI workflow (Not implemented).
- **Classification**: import datasets, preview data, reduce dimensions, train classifiers, and save/load classification models.
- **WAXS/GIWAXS**: standalone experimental window for WAXS/GIWAXS-related workflows.

## Installation

### Windows Installer

* Visit the [GitHub Releases](https://github.com) page.
* Look for the latest version marked with the **Latest** tag.
* Download the `*-setup.exe` (or the provided `.zip` archive).

### Windows SmartScreen Warning

Unsigned pre-release executables can trigger Windows SmartScreen. Only run files downloaded from a trusted release source.

If you do not want to run an unsigned executable, you can run GIMaP from source in a local Python environment instead. See the “Running from Source” section below for instructions.

### Running from Source

Recommended source environment: Windows with Python 3.10 or 3.11.

```powershell
cd gisaxs_gui

python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt

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
pip install -r requirements.txt
python main.py
```

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

## Multi-file Prediction

The GIMaP Predict page includes a multi-file workflow for processing folders or file collections. Results can be reviewed in an external multi-file results window and exported for later analysis.

## Exporting Results

Export support depends on the active page and workflow. The current codebase includes exports for fitting results, prediction outputs, multi-file prediction tables, AI fitting candidates, curves, residuals, and plots.

AI fitting output is written under `AI_Fitting_Output/current_prediction` and may include JSON, CSV, NPZ, and PNG files.

## System Requirements

- Windows is the primary development and usage target.
- Python 3.10 or 3.11 is recommended for source execution.
- A display large enough for scientific control panels is recommended.
- CPU execution is supported. GPU TensorFlow setups are not configured by this repository and must be installed separately if required.

## Dependencies

Dependencies are listed in `requirements.txt`:

- PyQt5
- NumPy
- SciPy
- Matplotlib
- OpenCV
- h5py
- fabio
- umap-learn
- scikit-learn
- TensorFlow 2.15.x (`tensorflow-intel` on Windows)

## Project Structure

```text
main.py                         Application entry point
controllers/                    Page controllers and workflow logic
ui/                             PyQt UI definitions and components
utils/                          Fitting, prediction, loading, and helper utilities
modules/                        Prediction and fitting model modules
config/                         Application parameters and configuration files
core/                           Shared settings and global parameter helpers
WAXS/                           Standalone experimental WAXS/GIWAXS window
AI_Fitting_Output/              AI fitting output directory
docs/                           User and developer documentation
requirements.txt                Python dependency list
```

## Current Limitations

- This is early pre-release / beta scientific software.
- Some pages and workflows are experimental or under active development.
- Some model configuration paths may need to be adjusted for each local installation.
- Small screens may require window resizing or scrolling because scientific controls are dense.


## Feedback and Contact

For feedback, bug reports, or collaboration questions, contact:

[yufeng.zhai@desy.de](mailto:yufeng.zhai@desy.de)


## License

This project is released under the MIT License. See [LICENSE](LICENSE) for the full license text.
