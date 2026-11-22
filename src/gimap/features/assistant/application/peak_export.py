"""CSV of the measured GIWAXS peaks, their fit errors and candidate reflections."""

from __future__ import annotations

import csv
import io
import json


def peak_table_csv(report: dict) -> str:
    """Use the numbers in the finished report, including unassigned peaks and ambiguous alternatives."""
    output = io.StringIO(newline="")
    # Keep the reduction and the indexing inputs with the table when it leaves the project.
    for key in ("frame", "frames", "software_version", "created", "inputs", "analysis_record",
                "geometry", "mask", "peak_search", "error_model"):
        if key in report:
            output.write(f"# {key}: {json.dumps(report[key], ensure_ascii=False)}\n")
    assignment = {key: value for key, value in (report.get("phase_assignment") or {}).items()
                  if key in ("phases", "rule", "q_axis")}
    output.write(f"# phase_assignment: {json.dumps(assignment, ensure_ascii=False)}\n")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(("q (A^-1)", "sigma_q (A^-1)", "d (A)", "sigma_d (A)", "FWHM (A^-1)", "sigma_FWHM (A^-1)",
                     "area", "SNR", "orientation", "size (nm)", "size_is_lower_bound", "size_note", "flags", "caveat",
                     "phase", "hkl", "delta_q (A^-1)", "ambiguous", "tentative", "alternatives"))
    for peak in report.get("peaks") or ():
        fit = peak.get("fit") or {}
        writer.writerow((peak.get("q"), fit.get("q_err"), peak.get("d_A"), fit.get("d_err_A"),
                         peak.get("fwhm"), fit.get("fwhm_err"), fit.get("area"), peak.get("snr"),
                         peak.get("orientation"), peak.get("size_nm"), peak.get("size_is_lower_bound"),
                         peak.get("size_note"), ", ".join(peak.get("flags") or ()), peak.get("caveat"),
                         peak.get("phase"), peak.get("hkl"), peak.get("delta_q"), peak.get("ambiguous"),
                         peak.get("tentative"), json.dumps(peak.get("alternatives") or [], ensure_ascii=False)))
    return output.getvalue()
