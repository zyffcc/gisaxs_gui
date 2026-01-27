# GISAXS/GIWAXS Toolkit (GUI)

A Windows-friendly GUI for GISAXS/GIWAXS data processing: cut & fitting, CNN-based prediction, dataset classification, and GIWAXS 1D integration. Supports single images and batch workflows with clear outputs.

## Features
- **Cut & Fitting**:
  - File import and 2D preview (TIFF/PNG/JPG, text `.ctxti`, P03-style `.nxs`).
  - Interactive ROI for 1D integration (radial/azimuthal) with pixel or Q-space views.
  - Batch processing to export 1D curves and 2D images.
  - Traditional model-based fitting workflows with session save/restore.
- **GISAXS Prediction**:
  - CNN-based prediction using trained models for structures/parameters.
  - Single-file and multi-file prediction with result manager and export.
- **Classification**:
  - Batch import of experimental data, dimensionality reduction (UMAP/others),
    feature extraction/visualization, and classification workflows.
- **GIWAXS Module**:
  - GIWAXS data import, signal integration, and batch processing.

## Installation
Recommended: Anaconda on Windows (Python 3.10/3.11).

```bash
# clone your project repo then
cd gisaxs_gui

# optional: create a new environment
conda create -n GISAXS python=3.11 -y
conda activate GISAXS

# install dependencies
pip install -r requirements.txt
```

Notes:
- TensorFlow 2.15 is pinned for Windows/Python 3.11 compatibility.
- If you have a GPU build of TensorFlow/CUDA, install it separately as needed.

## Quick Start
```bash
# from the project root
python main.py
```
- Use the left file tree or "Select File" to open images.
- Top right: switch between "Original" (pixel space) and "Cut" (Q-space) views.
- Adjust color limits; Flip toggles display orientation only.

### ROI and 1D Integration
- Click "Select ROI" and pick four clicks:
  1) First radial line (angle start), 2) second radial line (angle end),
  3) inner radius point, 4) outer radius point.
- Angle convention (consistent across UI and Cut mode):
  - 0° along +Qr (right). Upper half-plane is −180..0°, lower half-plane is 0..180°.
  - Azimuth in Cut mode integrates in Q-space chi within your wedge.
- Press "Integrate" and choose Radial/Azimuthal with Log/Linear.
- Batch 1D output headers include: `# q/chi <file_or_frame_names>`.

### Batch Processing
- Panel: "Batch Process".
- Choose a folder and filename pattern (e.g. `*.tif`, `*.ctxti`, `*.nxs`).
- Options:
  - Export images (2D PNG/JPG), export 1D curves, background subtraction.
  - For `.nxs`, frames are processed in sequence and suffixed (e.g., `f0001`).
- Outputs:
  - 2D: `image/<name>[_fNNNN].jpg`.
  - 1D: `1D/output.txt` and (if enabled) `1D/output_subBk.txt`, both with header.

## Module Guide

### Cut & Fitting
- Import data → preview → select ROI → integrate 1D.
- Switch to Cut (Q-space) for azimuthal profiles in Q (chi) with q-range inferred from ROI.
- Run traditional fitting via the fitting panel; sessions are saved/restored automatically.

### GISAXS Prediction
- Load data, select a trained model, and run prediction.
- Supports single-file and batch prediction with a results manager UI and export dialogs.

### Classification
- Import batches, compute features, dimensionality reduction (e.g., UMAP), visualize clusters,
  and run classification workflows.

### GIWAXS Module
- Import GIWAXS images, perform 1D signal integration, and batch export curves.

## Supported Formats
- Images: `.tif`, `.tiff`, `.png`, `.jpg`, `.jpeg`, `.bmp`
- Text: `.ctxti` (2D numeric arrays)
- P03 NXS (module series): `.nxs` with automatic module stitching and frame selection

## Tips & Conventions
- Intensity thresholds mask hot/bad pixels; NaNs break bins for cleaner plots.
- Azimuth bins cover −180..180°. Wedges that cross the seam are handled automatically.
- In Cut mode, q-range for azimuth integration is derived from the ROI’s q distribution.

## Troubleshooting
- If TensorFlow install fails on Windows, try a fresh conda env and ensure Python 3.10/3.11.
- For `.nxs` series, ensure all module files are in the same folder.
- If batch 1D outputs look repeated, verify the pattern and that files match.

## License
Proprietary, all rights reserved by the project owner.
