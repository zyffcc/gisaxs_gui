# Display system

- **Status**: Current
- **Scope**: detector and curve views of Analyze and Fitting (pyqtgraph), and the Matplotlib views that remain
- **Owners**: `src/gimap/app/presentation/components/detector_view.py`,
  `curve_plot.py`; Fitting's `curve_rendering.py` (Matplotlib)
- **Last verified**: 2026-11-01 (Fitting section and `ScientificImageViewer` checked against the code; the rest
  2026-09-28)

## Analyze (pyqtgraph)

`DetectorView` shows a frame in the canonical detector frame (row 0 at the top, pixel `(i, j)`
covering `[j, j+1] × [i, i+1]`) or a reciprocal-space map with the y axis pointing up. Display
choices — log scale, colour map, colour levels, down-sampling — never change the scientific array;
the cursor readout reports the full-resolution value. NaN pixels (invalid or masked) are drawn
transparent; `NanSafeImageItem` resets pyqtgraph's NaN cache at each render so zooming never
reuses a mask of another down-sampling factor.

Overlays are owned by the view and driven by the page: horizontal and vertical cut bands, the
horizon line, the draggable beam-centre target (`beamCenterMoved`), a pick mode for the centre,
and the q-box rectangle on the q map (`boxChanged`). `display_state()` hands the displayed array
and its styling to the figure exporter.

`CurvePlot` draws reduced curves; its background, axes and legend follow the light/dark theme.
Both views must be disposed (`dispose()`) before their parent is deleted: pyqtgraph views torn
down with a parent that Python still references can abort the interpreter.

## Fitting

The pages shown — Single analysis (`single/page.py`, layout `views/fit_page_view.py`) and In-situ series
(`single/series_page.py`, layout `views/fit_series_view.py`) — draw their curves, residuals and trends with the
shared pyqtgraph `CurvePlot`, like Analyze.

Matplotlib remains where the older views still are: the 1D Predict window (`workflow_v5_dialog.py`), the classic
single-curve page and the former in-situ page, which are still built but never shown
(`fitting/presentation/workspace.py`; the in-situ heatmap with its colour bar is in
`bindings/insitu_monitoring.py`), and the independent curve windows. Their curve plots go through one
specification and renderer, `render_curve_plot` (`curve_rendering.py`), which reuses its artists across refreshes.

`ScientificImageViewer` (`components/scientific_image_viewer.py`) is exported by the components package and has
its own test (`tests/test_scientific_image_viewer.py`), but no feature uses it now.

## Figures

Publication figures (PNG/TIFF 600 dpi, SVG/PDF) are rendered by the Analyze infrastructure
adapter `MatplotlibFigureWriter` without pyplot state, from what the views show. Exported figures
use the light colours whatever the theme (`components/plot_export.export_plot` for pyqtgraph,
`theme.figures.exported_colors` for themed Matplotlib figures, `shared/figures.save_figure` for new
publication figures).
