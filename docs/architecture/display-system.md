# Display system

- **Status**: Current
- **Scope**: detector and curve views of Analyze (pyqtgraph) and of the classic Fitting page (Matplotlib)
- **Owners**: `src/gimap/app/presentation/components/detector_view.py`,
  `curve_plot.py`, `scientific_image_viewer.py`; Fitting's `curve_rendering.py`
- **Last verified**: 2026-09-28

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

## Fitting (Matplotlib)

The classic Fitting page keeps one canvas, image axes and colour bar alive across refreshes;
curve plots reuse their artists (`render_curve_plot`). `ScientificImageViewer` is the shared
interactive detector window used by Fitting.

## Figures

Publication figures (PNG/TIFF 600 dpi, SVG/PDF) are rendered by the Analyze infrastructure
adapter `MatplotlibFigureWriter` without pyplot state, from what the views show.
