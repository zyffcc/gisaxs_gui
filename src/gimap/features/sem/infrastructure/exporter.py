"""Export figures and scientific values with their complete provenance."""

from __future__ import annotations

import csv
from dataclasses import asdict, fields
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
from PIL import Image

from ..application import SemParticle
from .figures import export_distribution_figures, export_overlay, overlay_rgba


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot export {type(value).__name__} to JSON")


class LocalSemExporter:
    def export(self, measurements, comparison, out_dir: Path) -> tuple[Path, ...]:
        out_dir = Path(out_dir)
        reserved = ("sem_particles.csv", "sem_distribution.csv", "sem_display_arrays.npz",
                    "sem_distribution.png", "sem_distribution.pdf", "sem_distribution.svg",
                    "sem_segmentation.npz", "sem_instance_labels.tif", "sem_overlay.png", "sem_overlay.pdf", "sem_overlay.svg", "sem_analysis.json")
        if any((out_dir / name).exists() for name in reserved):
            raise FileExistsError("SEM output files already exist. Choose a new output directory.")
        out_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        record = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
                  "source": dict(measurements.source), "measurement_settings": dict(measurements.settings),
                  "measurement_summary": dict(measurements.summary),
                  "comparison": dict(comparison.metadata) if comparison is not None else None,
                  "particle_quantity": "radius_nm and diameter_nm", "plot": {
                      "figure_size_cm": [15, 10], "font_size_pt": 8, "tick_font_size_pt": 7,
                      "raster_dpi": 600, "pdf_fonttype": 42, "svg_fonttype": "none",
                      "colors": {"sem": "#79AFC8", "prediction": "#B74C41"},
                      "single_panel_alignment": "not_applicable", "background": "white",
                      "overlay_display_stretch": "1st/99th percentile of finite stored grayscale values; display only",
                      "overlay_colors": {"accepted": [0.05, 0.75, 0.9, 0.95], "rejected": [1.0, 0.55, 0.1, 0.95]},
                      "overlay_extent": "x=[0,width], y=[height,0]; pixel centers at column+0.5,row+0.5"}}
        header = "# " + json.dumps({"source": record["source"], "settings": record["measurement_settings"],
                                     "comparison": record["comparison"]}, default=_json_default, ensure_ascii=False)
        particle_path = out_dir / "sem_particles.csv"
        with particle_path.open("w", encoding="utf-8", newline="") as stream:
            stream.write(header + "\n")
            writer = csv.DictWriter(stream, fieldnames=[item.name for item in fields(SemParticle)])
            writer.writeheader()
            writer.writerows(asdict(particle) for particle in measurements.particles)
        paths.append(particle_path)
        if comparison is not None:
            histogram_path = out_dir / "sem_distribution.csv"
            edges = comparison.bin_edges_nm
            model = comparison.prediction_probability
            with histogram_path.open("w", encoding="utf-8", newline="") as stream:
                stream.write(header + "\n")
                writer = csv.writer(stream)
                writer.writerow(["radius_left_nm", "radius_right_nm", "sem_probability", "prediction_probability"])
                for index, probability in enumerate(comparison.sem_probability):
                    writer.writerow([edges[index], edges[index + 1], probability,
                                     "" if model is None else model[index]])
            paths.append(histogram_path)
            display_path = out_dir / "sem_display_arrays.npz"
            display = {"sem_edges_nm": comparison.display_sem_edges_nm, "sem_values": comparison.display_sem_values,
                       "radius_nm": measurements.radius_nm}
            if comparison.display_prediction_values is not None:
                display.update({"prediction_radius_nm": comparison.display_prediction_radius_nm,
                                "prediction_values": comparison.display_prediction_values})
            np.savez_compressed(display_path, **display)
            paths.append(display_path)
            paths.extend(export_distribution_figures(comparison, out_dir))
        if measurements.labels is not None:
            labels_path = out_dir / "sem_instance_labels.tif"
            Image.fromarray(np.asarray(measurements.labels, dtype=np.int32)).save(labels_path, compression="tiff_deflate")
            paths.append(labels_path)
            segmentation_path = out_dir / "sem_segmentation.npz"
            np.savez_compressed(segmentation_path, labels=measurements.labels, image=measurements.image,
                                overlay_rgba=overlay_rgba(measurements))
            paths.append(segmentation_path)
            paths.extend(export_overlay(measurements, out_dir))
        record["files"] = [path.name for path in paths]
        record_path = out_dir / "sem_analysis.json"
        record_path.write_text(json.dumps(record, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
        paths.append(record_path)
        return tuple(paths)
