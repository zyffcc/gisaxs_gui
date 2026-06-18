#!/usr/bin/env python3
"""Interactive tool for selecting and exporting masked q ranges on 1D curves.

Usage:
    python select_mask_ranges.py --input_csv path/to/Cut_Data.txt --output_json mask_ranges.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector
from PyQt5 import QtCore, QtGui, QtWidgets


@dataclass
class MaskRange:
    start: float
    end: float

    def normalized(self) -> "MaskRange":
        if self.start <= self.end:
            return self
        return MaskRange(self.end, self.start)


def load_curve_file(file_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load x/y data from txt or csv by parsing first two numeric columns."""
    xs: List[float] = []
    ys: List[float] = []

    with file_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            # Accept both CSV and whitespace-separated data.
            tokens = stripped.replace(",", " ").split()
            numeric_values: List[float] = []
            for token in tokens:
                try:
                    numeric_values.append(float(token))
                except ValueError:
                    continue

            if len(numeric_values) >= 2:
                xs.append(numeric_values[0])
                ys.append(numeric_values[1])

    if len(xs) == 0:
        raise ValueError(f"No usable numeric x/y rows found in file: {file_path}")

    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)

    # Sort by x for stable visualization and range operations.
    sort_idx = np.argsort(x)
    return x[sort_idx], y[sort_idx]


def merge_ranges(ranges: List[MaskRange], tol: float = 1e-12) -> List[MaskRange]:
    if not ranges:
        return []

    normalized = sorted((r.normalized() for r in ranges), key=lambda r: r.start)
    merged: List[MaskRange] = [normalized[0]]

    for cur in normalized[1:]:
        prev = merged[-1]
        if cur.start <= prev.end + tol:
            merged[-1] = MaskRange(prev.start, max(prev.end, cur.end))
        else:
            merged.append(cur)

    return merged


class MaskRangeSelector(QtWidgets.QMainWindow):
    def __init__(self, input_file: Optional[Path], output_json: Path):
        super().__init__()

        self.setWindowTitle("Mask Range Selector")
        self.resize(1200, 800)

        self.input_file = input_file
        self.output_json = output_json

        self.x: Optional[np.ndarray] = None
        self.y: Optional[np.ndarray] = None
        self.ranges: List[MaskRange] = []
        self.selected_row: int = -1
        self._is_updating_table = False
        self.use_signed_log_x = True
        self.use_log_y = True
        self._display_state = {
            "xlim": None,
            "ylim": None,
            "xscale": "linear",
            "yscale": "linear",
            "autoscale_x": True,
            "autoscale_y": True,
        }
        self._has_drawn_data_once = False
        self._x_pos_shift = 0.0
        self._x_neg_shift = 0.0

        self._build_ui()
        self._bind_events()

        if self.input_file is not None:
            self._load_data(self.input_file)

    @staticmethod
    def _safe_log10(values: np.ndarray) -> np.ndarray:
        return np.log10(np.clip(values, 1e-300, None))

    def _compute_x_shifts(self) -> None:
        """Compute shifts so +q stays on +half-axis and -q stays on -half-axis."""
        self._x_pos_shift = 0.0
        self._x_neg_shift = 0.0

        if self.x is None:
            return

        pos = self.x[self.x > 0]
        if pos.size > 0:
            min_log_pos = float(np.min(self._safe_log10(pos)))
            self._x_pos_shift = -min_log_pos if min_log_pos < 0 else 0.0

        neg_abs = np.abs(self.x[self.x < 0])
        if neg_abs.size > 0:
            min_log_neg = float(np.min(self._safe_log10(neg_abs)))
            self._x_neg_shift = -min_log_neg if min_log_neg < 0 else 0.0

    def _forward_signed_log_x(self, x_values: np.ndarray) -> np.ndarray:
        x_values = np.asarray(x_values, dtype=float)
        out = np.zeros_like(x_values, dtype=float)

        pos_mask = x_values > 0
        neg_mask = x_values < 0

        if np.any(pos_mask):
            out[pos_mask] = self._safe_log10(x_values[pos_mask]) + self._x_pos_shift
        if np.any(neg_mask):
            out[neg_mask] = -(self._safe_log10(np.abs(x_values[neg_mask])) + self._x_neg_shift)

        return out

    def _inverse_signed_log_x(self, x_display: np.ndarray) -> np.ndarray:
        x_display = np.asarray(x_display, dtype=float)
        out = np.zeros_like(x_display, dtype=float)

        pos_mask = x_display > 0
        neg_mask = x_display < 0

        if np.any(pos_mask):
            out[pos_mask] = np.power(10.0, x_display[pos_mask] - self._x_pos_shift)
        if np.any(neg_mask):
            out[neg_mask] = -np.power(10.0, (-x_display[neg_mask]) - self._x_neg_shift)

        return out

    def _x_to_display(self, x_values: np.ndarray) -> np.ndarray:
        if self.use_signed_log_x:
            return self._forward_signed_log_x(x_values)
        return x_values

    def _y_to_display(self, y_values: np.ndarray) -> np.ndarray:
        if self.use_log_y:
            return self._safe_log10(y_values)
        return y_values

    def _capture_display_state_from_axes(self) -> None:
        # Keep user edits from Figure Options (scale/limits) instead of resetting on redraw.
        if not self._has_drawn_data_once:
            return

        xscale = self.ax.get_xscale()
        yscale = self.ax.get_yscale()

        # Figure Options "x log" toggles signed-log mode for negative-q support.
        if xscale == "log":
            self.use_signed_log_x = True
            xscale = "linear"

        # Y axis uses normal log10 transform for display.
        if yscale == "log":
            self.use_log_y = True
            yscale = "linear"

        self._display_state = {
            "xlim": self.ax.get_xlim(),
            "ylim": self.ax.get_ylim(),
            "xscale": xscale,
            "yscale": yscale,
            "autoscale_x": bool(self.ax.get_autoscalex_on()),
            "autoscale_y": bool(self.ax.get_autoscaley_on()),
        }

    def _apply_display_state_to_axes(self) -> None:
        if not self._has_drawn_data_once:
            return

        self.ax.set_xscale(self._display_state.get("xscale", "linear"))
        self.ax.set_yscale(self._display_state.get("yscale", "linear"))
        self.ax.set_autoscalex_on(self._display_state.get("autoscale_x", True))
        self.ax.set_autoscaley_on(self._display_state.get("autoscale_y", True))

        if self._display_state.get("xlim") is not None:
            self.ax.set_xlim(self._display_state["xlim"])
        if self._display_state.get("ylim") is not None:
            self.ax.set_ylim(self._display_state["ylim"])

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)
        root_layout = QtWidgets.QVBoxLayout(central)

        # Plot area
        self.figure = Figure(constrained_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.toolbar = NavigationToolbar(self.canvas, self)
        root_layout.addWidget(self.toolbar)
        root_layout.addWidget(self.canvas, stretch=1)

        # Bottom controls and table
        bottom_widget = QtWidgets.QWidget(self)
        bottom_layout = QtWidgets.QVBoxLayout(bottom_widget)
        root_layout.addWidget(bottom_widget)

        controls_layout = QtWidgets.QHBoxLayout()
        bottom_layout.addLayout(controls_layout)

        self.btn_open = QtWidgets.QPushButton("Open txt/csv")
        self.btn_delete = QtWidgets.QPushButton("Delete Selected Range")
        self.btn_save_json = QtWidgets.QPushButton("Save Ranges JSON")
        self.btn_export_filtered = QtWidgets.QPushButton("Export Unmasked Data (txt/csv)")

        controls_layout.addWidget(self.btn_open)
        controls_layout.addWidget(self.btn_delete)
        controls_layout.addWidget(self.btn_save_json)
        controls_layout.addWidget(self.btn_export_filtered)
        controls_layout.addStretch(1)

        self.info_label = QtWidgets.QLabel(
            "Drag on the plot to add a q range. Right-click inside a range on the plot to delete it."
        )
        bottom_layout.addWidget(self.info_label)

        self.table = QtWidgets.QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["x_start", "x_end", "point_count"])
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        bottom_layout.addWidget(self.table)

        self.span_selector = SpanSelector(
            self.ax,
            self._on_span_select,
            "horizontal",
            useblit=True,
            props=dict(alpha=0.18, facecolor="tab:red"),
            interactive=False,
            drag_from_anywhere=False,
        )

    def _bind_events(self) -> None:
        self.btn_open.clicked.connect(self._open_file_dialog)
        self.btn_delete.clicked.connect(self._delete_selected_range)
        self.btn_save_json.clicked.connect(self._save_json_dialog)
        self.btn_export_filtered.clicked.connect(self._export_filtered_dialog)

        self.table.itemSelectionChanged.connect(self._on_table_selection_changed)
        self.table.itemChanged.connect(self._on_table_item_changed)

        self.canvas.mpl_connect("button_press_event", self._on_plot_click)

    def _load_data(self, file_path: Path) -> None:
        try:
            x, y = load_curve_file(file_path)
        except Exception as e:  # pragma: no cover - GUI error path
            QtWidgets.QMessageBox.critical(self, "Load Error", str(e))
            return

        self.input_file = file_path
        self.x = x
        self.y = y
        self._compute_x_shifts()
        self.ranges = []
        self.selected_row = -1
        self._has_drawn_data_once = False
        self.setWindowTitle(f"Mask Range Selector - {file_path}")
        self._refresh_all()

    def _open_file_dialog(self) -> None:
        start_dir = str(self.input_file.parent) if self.input_file else str(Path.cwd())
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open Data File",
            start_dir,
            "Data Files (*.txt *.csv);;All Files (*)",
        )
        if path:
            self._load_data(Path(path))

    def _on_span_select(self, xmin: float, xmax: float) -> None:
        if self.x is None or self.y is None:
            return
        if not np.isfinite(xmin) or not np.isfinite(xmax):
            return

        # Span selector works on displayed coordinates, convert back to original x domain.
        if self.use_signed_log_x:
            xmin = float(self._inverse_signed_log_x(np.array([xmin]))[0])
            xmax = float(self._inverse_signed_log_x(np.array([xmax]))[0])

        start, end = (xmin, xmax) if xmin <= xmax else (xmax, xmin)
        if np.isclose(start, end):
            return

        self.ranges.append(MaskRange(start, end))
        self.ranges = merge_ranges(self.ranges)
        self.selected_row = len(self.ranges) - 1
        self._refresh_all()

    def _on_plot_click(self, event) -> None:
        if event.inaxes != self.ax or event.xdata is None:
            return

        # Right-click opens range picker and cancels/deletes the selected range.
        if event.button == 3:
            clicked_x = event.xdata
            if self.use_signed_log_x:
                clicked_x = float(self._inverse_signed_log_x(np.array([clicked_x]))[0])

            self._show_delete_range_picker(clicked_x)

    def _show_delete_range_picker(self, clicked_x: float) -> None:
        if not self.ranges:
            return

        items: List[str] = []
        default_idx = 0
        for idx, r in enumerate(self.ranges):
            rn = r.normalized()
            count = int(np.count_nonzero(self._masked_indices_for_range(rn)))
            items.append(f"{idx + 1}: [{rn.start:.8g}, {rn.end:.8g}] ({count} points)")
            if rn.start <= clicked_x <= rn.end:
                default_idx = idx

        selected_text, ok = QtWidgets.QInputDialog.getItem(
            self,
            "Cancel/Delete Range",
            "Select one range to cancel/delete:",
            items,
            default_idx,
            False,
        )
        if not ok:
            return

        try:
            picked_idx = int(selected_text.split(":", 1)[0]) - 1
        except Exception:
            return

        if 0 <= picked_idx < len(self.ranges):
            del self.ranges[picked_idx]
            self.ranges = merge_ranges(self.ranges)
            self.selected_row = -1
            self._refresh_all()

    def _find_range_index_by_x(self, x_value: float) -> Optional[int]:
        for i, r in enumerate(self.ranges):
            rn = r.normalized()
            if rn.start <= x_value <= rn.end:
                return i
        return None

    def _delete_selected_range(self) -> None:
        if self.selected_row < 0 or self.selected_row >= len(self.ranges):
            return
        del self.ranges[self.selected_row]
        self.ranges = merge_ranges(self.ranges)
        self.selected_row = -1
        self._refresh_all()

    def _refresh_all(self) -> None:
        self._redraw_plot()
        self._refresh_table()
        self._refresh_info_label()

    def _masked_indices_for_range(self, r: MaskRange) -> np.ndarray:
        if self.x is None:
            return np.zeros(0, dtype=bool)
        rn = r.normalized()
        return (self.x >= rn.start) & (self.x <= rn.end)

    def _masked_indices_union(self) -> np.ndarray:
        if self.x is None:
            return np.zeros(0, dtype=bool)
        mask = np.zeros_like(self.x, dtype=bool)
        for r in self.ranges:
            mask |= self._masked_indices_for_range(r)
        return mask

    def _redraw_plot(self) -> None:
        self._capture_display_state_from_axes()
        self.ax.clear()
        self._apply_display_state_to_axes()

        if self.use_signed_log_x:
            self.ax.set_xlabel("q (signed log10 folded x)")
        else:
            self.ax.set_xlabel("q (x)")

        if self.use_log_y:
            self.ax.set_ylabel("Intensity (log10 y)")
        else:
            self.ax.set_ylabel("Intensity (y)")

        self.ax.grid(True, alpha=0.25)

        if self.x is None or self.y is None:
            self.ax.set_title("No data loaded")
            self.canvas.draw_idle()
            return

        x_disp = self._x_to_display(self.x)
        y_disp = self._y_to_display(self.y)

        self.ax.scatter(x_disp, y_disp, s=12, color="tab:blue", label="Curve", zorder=3)

        union_mask = self._masked_indices_union()
        if np.any(union_mask):
            self.ax.scatter(
                x_disp[union_mask],
                y_disp[union_mask],
                s=18,
                c="red",
                edgecolors="none",
                label="Masked points",
                zorder=4,
            )

        for r in self.ranges:
            rn = r.normalized()
            span_start = float(self._x_to_display(np.array([rn.start]))[0])
            span_end = float(self._x_to_display(np.array([rn.end]))[0])
            self.ax.axvspan(span_start, span_end, color="red", alpha=0.10, zorder=1)

        if 0 <= self.selected_row < len(self.ranges):
            sr = self.ranges[self.selected_row].normalized()
            selected_mask = self._masked_indices_for_range(sr)
            if np.any(selected_mask):
                self.ax.scatter(
                    x_disp[selected_mask],
                    y_disp[selected_mask],
                    s=42,
                    facecolors="none",
                    edgecolors="gold",
                    linewidths=1.1,
                    label="Selected range",
                    zorder=5,
                )
            span_start = float(self._x_to_display(np.array([sr.start]))[0])
            span_end = float(self._x_to_display(np.array([sr.end]))[0])
            self.ax.axvspan(span_start, span_end, color="gold", alpha=0.20, zorder=2)

        self.ax.legend(loc="best")
        self._has_drawn_data_once = True
        self.canvas.draw_idle()

    def _refresh_table(self) -> None:
        self._is_updating_table = True
        try:
            self.table.setRowCount(len(self.ranges))
            for row, r in enumerate(self.ranges):
                rn = r.normalized()
                count = int(np.count_nonzero(self._masked_indices_for_range(rn)))

                start_item = QtWidgets.QTableWidgetItem(f"{rn.start:.8g}")
                end_item = QtWidgets.QTableWidgetItem(f"{rn.end:.8g}")
                count_item = QtWidgets.QTableWidgetItem(str(count))
                count_item.setFlags(count_item.flags() & ~QtCore.Qt.ItemIsEditable)

                self.table.setItem(row, 0, start_item)
                self.table.setItem(row, 1, end_item)
                self.table.setItem(row, 2, count_item)

            if 0 <= self.selected_row < len(self.ranges):
                self.table.selectRow(self.selected_row)
            else:
                self.table.clearSelection()
        finally:
            self._is_updating_table = False

    def _on_table_selection_changed(self) -> None:
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            self.selected_row = -1
            self._redraw_plot()
            return

        self.selected_row = rows[0].row()
        self._redraw_plot()

    def _on_table_item_changed(self, item: QtWidgets.QTableWidgetItem) -> None:
        if self._is_updating_table:
            return
        row = item.row()
        if row < 0 or row >= len(self.ranges):
            return

        start_item = self.table.item(row, 0)
        end_item = self.table.item(row, 1)
        if start_item is None or end_item is None:
            return

        try:
            start = float(start_item.text())
            end = float(end_item.text())
        except ValueError:
            # Revert invalid edit.
            self._refresh_table()
            return

        self.ranges[row] = MaskRange(start, end).normalized()
        self.ranges = merge_ranges(self.ranges)

        # Try to keep the edited row selected after merge.
        if self.ranges:
            self.selected_row = min(row, len(self.ranges) - 1)
        else:
            self.selected_row = -1

        self._refresh_all()

    def _refresh_info_label(self) -> None:
        if self.x is None:
            self.info_label.setText("No data loaded. Click 'Open txt/csv' to load a curve.")
            return

        union_mask = self._masked_indices_union()
        masked_count = int(np.count_nonzero(union_mask))
        total_count = int(len(self.x))
        self.info_label.setText(
            f"Ranges: {len(self.ranges)} | Masked points: {masked_count}/{total_count} | "
            "Drag to add range, click table row to highlight, right-click plot to choose a range to delete."
        )

    def _build_json_payload(self) -> dict:
        payload_ranges = []
        for r in self.ranges:
            rn = r.normalized()
            count = int(np.count_nonzero(self._masked_indices_for_range(rn)))
            payload_ranges.append(
                {
                    "start": float(rn.start),
                    "end": float(rn.end),
                    "count": count,
                }
            )

        masked_count = int(np.count_nonzero(self._masked_indices_union())) if self.x is not None else 0
        total_count = int(len(self.x)) if self.x is not None else 0

        return {
            "input_file": str(self.input_file) if self.input_file else None,
            "ranges": payload_ranges,
            "masked_points": masked_count,
            "total_points": total_count,
        }

    def _save_json_dialog(self) -> None:
        default_path = str(self.output_json)
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Mask Ranges JSON",
            default_path,
            "JSON Files (*.json)",
        )
        if not path:
            return

        out = Path(path)
        payload = self._build_json_payload()
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self.output_json = out
        QtWidgets.QMessageBox.information(self, "Saved", f"Ranges saved to:\n{out}")

    def _export_filtered_dialog(self) -> None:
        if self.x is None or self.y is None:
            QtWidgets.QMessageBox.warning(self, "No Data", "Please load a data file first.")
            return

        default_name = "filtered_unmasked.csv"
        if self.input_file is not None:
            default_name = f"{self.input_file.stem}_unmasked.csv"

        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export Unmasked Data",
            str(Path.cwd() / default_name),
            "CSV Files (*.csv);;Text Files (*.txt)",
        )
        if not path:
            return

        out = Path(path)
        masked = self._masked_indices_union()
        keep = ~masked
        data = np.column_stack((self.x[keep], self.y[keep]))

        if out.suffix.lower() == ".csv":
            np.savetxt(out, data, delimiter=",", fmt="%.10g", header="x,y", comments="")
        else:
            np.savetxt(out, data, delimiter=" ", fmt="%.10g", header="x y", comments="")

        QtWidgets.QMessageBox.information(
            self,
            "Exported",
            f"Unmasked points exported: {data.shape[0]}\nSaved to:\n{out}",
        )


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive q-range mask selector for txt/csv curves.")
    parser.add_argument(
        "--input_csv",
        type=str,
        default=None,
        help="Input txt/csv curve file (first 2 numeric columns used as x,y).",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="mask_ranges.json",
        help="Default output JSON path for saved mask ranges.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    input_file = Path(args.input_csv).expanduser().resolve() if args.input_csv else None
    if input_file is not None and not input_file.exists():
        print(f"Input file does not exist: {input_file}", file=sys.stderr)
        return 1

    output_json = Path(args.output_json).expanduser().resolve()

    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)
    app = QtWidgets.QApplication(sys.argv)
    font = QtGui.QFont("Segoe UI", 9) if sys.platform.startswith("win") else QtGui.QFont(app.font())
    font.setPointSize(9)
    app.setFont(font)
    win = MaskRangeSelector(input_file=input_file, output_json=output_json)
    win.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
