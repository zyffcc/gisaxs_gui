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


## System Requirements

- Windows is the primary development and usage target.
- Python 3.10 or 3.11 is recommended for source execution.
- A display large enough for scientific control panels is recommended.
- CPU execution is supported. GPU TensorFlow setups are not configured by this repository and must be installed separately if required.

## Dependencies

Dependencies are listed in `requirements.txt`:

- numpy>=1.24,<2.0
- scipy>=1.10
- matplotlib>=3.7
- BornAgain==24.1
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
