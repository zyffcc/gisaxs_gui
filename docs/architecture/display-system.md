# Detector display lifecycle

- **Status**: Current; interactive pixel projection is opt-in
- **Scope**: WAXS detector, fitting curves, shared detector inspection
- **Related code**: `src/gimap/app/presentation/components/scientific_image_viewer.py`,
  `src/gimap/features/waxs/presentation/image_viewer.py`,
  `src/gimap/features/fitting/presentation/curve_rendering.py`
- **Related tests**: `tests/test_display_artist_reuse.py`, `tests/test_scientific_image_viewer.py`
- **Last verified**: 2026-09-14

## Matplotlib projection

WAXS keeps the canvas, image axes and colorbar alive across image refreshes. Pixel images use
`set_data`; q meshes use `set_array` while coordinates match. Changed coordinates, signed-branch
layout or pixel/q transitions replace only image artists and rebind the existing colorbar. Mesh
coordinates are compared by value, including in-place changes, rather than by object identity.
A 1D/2D layout transition may recreate axes on the same canvas. Image overlays are explicitly owned
and removed separately from selectors; cancelling a selector removes artists and event connections.
The image viewport persists across same-geometry refreshes; overlays cannot expand its limits.

Fitting curve views use the same existing `CurvePlotSpec`. Lines and scatter artists are reused by
series identity; removed layers and ROI guides are removed explicitly. The graphics scene owns one
persistent canvas/proxy. Explicit clear invalidates its references. Legacy cut/result entrypoints
share that canvas but still rebuild their local artists; the next specification render clears the
legacy content once. Normal refreshes use `draw_idle`, allowing Qt to coalesce pending draws.

## Interactive pixel projection

`ScientificImageViewer` belongs to app presentation and has real WAXS and fitting callers. It uses
pyqtgraph `ImageItem` (row-major), `ViewBox`, `RectROI`, `InfiniteLine`, and `HistogramLUTItem`.
It does not import feature logic, read files, run scientific transforms or own analysis revisions.
The feature supplies prepared display pixels, full-resolution unlogged intensity, pixel extent,
origin and color limits. The cursor indexes the full-resolution input, not the display LOD.
NaNs stay no-data. Flipping remains in the existing feature display pipeline.

ROI changes remain provisional until **Apply ROI**; each feature delegates this intent to its
existing selection handler. Log/linear and histogram levels commit through feature controls, keeping
main/independent projections synchronized. Zoom, pan, crosshair and temporary ROI are local viewport
state. A custom histogram gradient is explicitly an inspection-only LUT; exports retain the workspace
colormap. Center/region overlays follow workspace state; WAXS also projects circle/sector outlines.

WAXS playback requests one frame at a time through its existing loader. A returned frame acknowledges
the request before another is emitted; there is no new stack cache or worker model. Closing/hiding,
a load failure or switching to q/1D stops playback. GISAXS streaming playback is not connected yet.

## Compatibility boundary and remaining work

The default detector and publication/export paths remain Matplotlib, and existing Matplotlib public
attributes remain available. The interactive window is an additional opt-in projection, not a full
backend replacement. Source refreshes still update the Matplotlib view, so interactive panning speed
must not be advertised as end-to-end detector frame throughput.

Nonuniform q-space and signed-Qr discontinuities continue to use the existing Matplotlib mesh;
the interactive window hides stale pixel data and selection controls in q mode. Never map a curved
q grid to a rectangular `ImageItem` extent. A future q backend needs explicit cell geometry,
branch splitting and nearest-cell selection regression tests before it can replace this projection.

Next candidates are a validated q mesh backend, GISAXS frame-source integration, avoiding hidden
Matplotlib raster work during streaming, and a common independent-window artist lifecycle. The
existing scientific data-flow contract remains authoritative.

Reference: [pyqtgraph ImageItem 0.13.7](https://pyqtgraph.readthedocs.io/en/pyqtgraph-0.13.7/api_reference/graphicsItems/imageitem.html).
