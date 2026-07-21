# GIMaP User Manual

## 1. Introduction

GIMaP is a PyQt-based desktop GUI for GISAXS/GIWAXS data visualization, analysis, fitting, and machine-learning-assisted workflows. It is intended for users who work with grazing-incidence scattering data and want a graphical workflow for inspecting detector images, extracting 1D curves, fitting physical models, running trained-model predictions, and organizing batch results.

The current software should be treated as early pre-release / beta scientific software. Some workflows are stable enough for daily testing, while others appear experimental or under development.

## 2. Installation

### Windows Installer

* Visit the [GitHub Releases](https://github.com) page.
* Look for the latest version marked with the **Latest** tag.
* Download the `*-setup.exe` (or the provided `.zip` archive).


### Running from Source

Install Python 3.10 or 3.11, then run:

```powershell
cd gisaxs_gui

python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt

python main.py
```

If using Conda:

```powershell
conda create -n GUI python=3.11 -y
conda activate GUI
pip install -r requirements.txt
python main.py
```

The source dependency file is `requirements.txt`. No `environment.yml` file was found.

## 3. Main Interface

The main window contains a left navigation panel and a central working area.

The main pages are:

- **Cut & Fitting**
- **GIMaP Predict**
- **Trainset Build**
- **Classification**
- **WAXS**

The application also exposes settings and parameter-related actions through the GUI menu system. These are used for reusable application options and global parameters.

## 4. Cut & Fitting Page

The Cut & Fitting page is used to inspect GISAXS data, create 1D cuts, and fit model curves.

### Import GISAXS Image

Use the file loading controls to import detector images or data files. The codebase includes support for scientific image loading through libraries such as `fabio`, `h5py`, OpenCV, and custom loaders. Existing documentation and code references indicate support for common image files and GISAXS detector formats such as `.cbf`, with actual behavior depending on the loader and file structure.

### Single Image / Stack Loading

The application includes logic for single-file and stacked data workflows. When a loaded file contains multiple frames or a selected workflow uses multiple files, use the available navigation controls to move through frames or files.

### Previous / Next Navigation

Previous and Next controls are used to move between files or frames when a multi-file or stack workflow is active.

### Display Options

Display controls are available for image preview and fitting plots. Depending on the active data view, these may include display scaling, color limits, linear/log intensity, and preview updates.

### Auto Scale

Auto scale adjusts the display intensity range for the current image or plot. It is intended for visualization and does not change the underlying data.

### Intensity Log Scale

Log display helps inspect weak scattering features and high-dynamic-range detector images. Use it for visualization; fitting behavior depends on the selected fitting/evaluation settings.

### Detector Preview

The detector preview shows the current 2D data and cut geometry. Use it to confirm that the loaded image, detector orientation, and selected region are reasonable before cutting or fitting.

### Cut Line Settings

Cut settings define how the 2D scattering image is converted into a 1D curve. The current GUI includes controls for selecting or configuring the cutting region and related detector/cut parameters.

### Detector Parameters

Detector parameters are used for coordinate conversion and interpretation of scattering data. Set these carefully before quantitative analysis. Incorrect detector parameters can produce incorrect q-space values and fitting results.

### Cutting Selected Region

After selecting the desired region or cut geometry, run the cut/integration action to generate a 1D curve. Inspect the generated curve before fitting.

### Manual Fitting

Manual fitting uses the currently selected model and parameter values. The current fitting code includes physical-model components such as sphere, cylinder, vertical cylinder, structure factor, global scale/background, and resolution-related terms.

### Auto Fitting / Auto Refine

Auto Refine runs least-squares optimization from the current parameter state. It is intended to refine selected parameters according to the GUI settings.

If Auto Refine produces a better result than an AI fitting run, check that the same parameters, bounds, fixed/free selections, scale/background options, and data weighting are used in both workflows.

### AI Auto Fitting

The AI fitting workspace can generate candidate parameters, rank candidates, refine selected candidates, and export results. This workflow is currently under active development.

Typical AI fitting outputs are written to:

```text
AI_Fitting_Output/current_prediction
```

The output may include:

- `top20_candidates.json`
- `top20_candidates.csv`
- `best_fit_curves.npz`
- `residuals_top5.npz`
- PNG plots for top candidates and residuals

### Resolution Function Parameters

Resolution-related controls are available in the fitting workflow. These parameters affect the model calculation and should be matched to the experiment when quantitative fitting is required.

### Global Scale / Background

The fitting model includes global scale and background terms. These are important for matching the measured intensity level and baseline.

### Export Plot / Export Fitting Result

Use export actions to save plots, fitting results, and generated data. Export availability depends on the active result and workflow state.

## 5. GIMaP Predict Page

The GIMaP Predict page runs trained prediction modules on GISAXS data.

### Choose Single File or Multi Files

Use **Single File** mode for one input file and **Multi Files** mode for folder or batch prediction workflows.

### Choose GISAXS File or Folder

Select a GISAXS detector file in single-file mode. In multi-file mode, select a folder or file collection according to the available controls.

### Set Stack / Range / Every

For stacked data or batch processing, configure the stack index, range, and every/step controls to decide which frames or files are processed.

### Select Module

Prediction modules are discovered from the `modules/` directory. Select a module that matches the input file type and intended prediction output.

### Edit Module Configuration

Module configuration is stored in `module.yaml`. The GUI includes controls for viewing or editing module configuration where connected. This feature appears to be experimental or under development in some workflows.

### Import Model

Import or select the trained model associated with the module. The module configuration contains model path information.

### Framework Selection

Module configurations include a `framework` field. Existing repository modules use TensorFlow/Keras-style model loading. Other frameworks should be considered unsupported unless a working module and loader are present.

### Model Loaded Status Indicator

The page shows whether a model is loaded or ready. If prediction is unavailable, first confirm that the model path is valid and the required framework dependency is installed.

### Run Prediction

After selecting input data and loading a model, run prediction from the page controls. Prediction output depends on the selected module.

### View Prediction Output

Outputs may include scalar values, parameter vectors, structured prediction results, or 2D prediction displays depending on the module configuration.

### GISAXS Preview Tab

The preview tab shows the selected GISAXS input data before or during prediction.

### Predict-2D Tab

The Predict-2D tab is used for 2D output display when the selected module provides compatible output.

### Export Current Result

Export the current prediction result from the page when a result is available.

### Multi-file Results External Window

Multi-file prediction results can be opened in an external results window for review.

### Export All Multi-file Results

Use the multi-file results window or export controls to save all batch prediction results.

## 6. Trainset Build Page

The Trainset Build page provides controls for generating training data. The controller includes beam parameters, detector parameters, sample parameters, preprocessing parameters, generation settings, output folder/name settings, run, and stop controls.

**This feature is under development.**

## 7. Classification Page

The Classification page supports dataset import, category management, 1D/2D preview, feature extraction, dimensionality reduction, classifier training, and model save/load workflows.

Detected functionality includes:

- Import category lists or folders
- Preview imported data
- Use dimensionality reduction methods such as PCA, t-SNE, or UMAP
- Train classifiers such as KNN or SVM
- Save and load classification models

**This feature is under development.**

## 8. WAXS Page

The WAXS / in-situ workflow is embedded directly in the main GUI. Use the **WAXS** item in the left navigation panel to open it.

The embedded page supports `.nxs`, `.tif`, and `.tiff` input through **Open File** or drag-and-drop. For `.nxs` files, the frame selector is enabled when multiple frames are detected. The page includes a large detector preview with zoom/pan controls, display and mask settings, geometry parameters, Q-range cut controls, 1D integration, and batch/in-situ export controls.

## 9. Geometry Calibration Tool

Open **Tools > Geometry Calibration...** (`Ctrl+Shift+G`) to calibrate a SAXS, GISAXS, or GIWAXS detector geometry without leaving the current page.

1. Open a calibration-standard `.nxs` or `.cbf` image, or paste its path into the image field and press Enter.
2. Confirm the detected energy and pixel size. For a CBF with matching scan NXS metadata, the energy is filled automatically. Enter the energy in keV if it is still missing.
3. Confirm the detector model. If it cannot be identified, choose a known detector from **Detector model**, or select **Custom pixel size** and enter the values in **Advanced Settings**.
4. Choose a standard, or leave **Auto Detect** selected. An approximate detector distance is optional but helps resolve harmonic alternatives.
5. Click **Auto Calibration**. The calculation runs in the background and can be cancelled.
6. Review the center, distance, residual, confidence, high-contrast overlay legend, and alternative candidates. **Clean image** temporarily hides all calibration overlays, **Reset view** restores the complete detector mosaic after zooming or panning, and **Focus image** hides the result panels to give tall WAXS mosaics more room. The horizontal divider can also be dragged.
7. After calibration, **Manual refine** opens automatically. Drag the center marker, edit the center/distance values, or pair a detected ring with a theoretical peak. Use **Finish manual** to collapse the panel when more image space is needed.
8. Click **Apply** to update the shared application geometry. Calibration results can also be exported to or imported from JSON.

Solid yellow overlays are matched theoretical rings, dashed orange overlays are unused theoretical rings, and dotted white overlays are detected experimental radii; the preview legend identifies each style. Partial WAXS arcs and centers outside the active detector area are supported. A low-confidence result or a one-ring result should be treated as ambiguous and reviewed manually.

## 10. Model Configuration

Prediction modules are configured with YAML files under `modules/`. Existing module files contain fields such as:

- `id`: internal module identifier
- `name`: user-visible module name
- `framework`: model framework, for example TensorFlow/Keras
- `version`: module version
- `model.model_path`: path to the trained model
- `preprocess.entry`: preprocessing entry point
- `preprocess.steps`: preprocessing steps
- `preprocess.params`: preprocessing parameters
- `io.input_type`: expected input type, for example `cbf`
- `io.stack_axis`: stack axis when applicable
- `io.input_shape`: expected model input shape
- `outputs`: output definition, parameter names, ranges, or output type

When adding a new module, make sure the model path, input type, preprocessing code, and output definition match the trained model.

AI fitting model discovery is handled separately and searches fitting-model folders under `modules/`, including `modules/Fitting_1D_Model`.

## 11. Troubleshooting

### File Cannot Be Loaded

Check that the file format is supported by the active loader. For detector formats such as `.cbf`, confirm that `fabio` is installed. For HDF5/Nexus-like files, confirm that `h5py` is installed and that the internal dataset path matches what the loader expects. If support for a custom file format is required, please contact yufeng.zhai@desy.de.

### Model Not Imported

Check the selected module configuration and model path. If the path points to a local machine-specific location, update it for the current computer.

### Predict Button Disabled or Prediction Fails

Confirm that input data is selected, a module is selected, the model is loaded, and required dependencies are installed.

### TensorFlow / PyTorch Not Available

`requirements.txt` lists TensorFlow 2.15.x. PyTorch is not listed as a project dependency. If a module requires another framework, install and configure it separately and confirm that loader code exists.

### Windows SmartScreen Warning

Unsigned pre-release executables can trigger SmartScreen. Only run files downloaded from a trusted release source. If you do not want to run an unsigned executable, you can run GIMaP from source in a local Python environment instead. See the “Running from Source” section in README file.

### Missing Dependency

Activate the correct Python environment and reinstall dependencies:

```powershell
pip install -r requirements.txt
```

### Empty Output

Check the input range, stack selection, model compatibility, and whether the selected frame contains valid data. For fitting workflows, also inspect masks, cut ranges, parameter bounds, and scale/background settings.

### Export Fails

Confirm that the output folder exists and is writable. Avoid exporting into protected system folders.

### GUI Layout Too Large for Small Screen

The GUI contains dense scientific controls. Use a larger display, maximize the window, or adjust system scaling if controls are clipped. For the best user experience, a 1080p or higher-resolution screen is recommended.

## 12. FAQ

### Is GIMaP production-ready?

No. The current repository should be treated as early pre-release / beta scientific software.

### Can I use my own trained model?

Yes, if it can be represented by a compatible module configuration and supported loader. Add or edit a `module.yaml` under `modules/` and make sure preprocessing and output definitions match the model.

### Where are AI fitting results saved?

AI fitting results are saved under `AI_Fitting_Output/current_prediction`.

## 13. Version and Contact

This documentation describes the current source repository state and may change as the GUI evolves.

Contact:

[yufeng.zhai@desy.de](mailto:yufeng.zhai@desy.de)
