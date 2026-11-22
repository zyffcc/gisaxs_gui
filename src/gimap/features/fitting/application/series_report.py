"""A readable series report made from the same record that accompanies the CSV."""

from __future__ import annotations


def _cell(value):
    if value is None or value == "":
        return "—"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value).replace("|", r"\|").replace("\r", " ").replace("\n", " ")


def _row(*values):
    return "| " + " | ".join(_cell(value) for value in values) + " |"


def _choice_cells(choice):
    if not choice:
        return [None] * 5
    rule = choice["rule"]
    if rule == "closest":
        rule = f"closest to frame {choice['reference_frame']}"
    return [rule, choice["rank"], choice["probability"], choice["top_probability"],
            choice.get("smoothed_probability")]


def series_report(record: dict) -> str:
    """Include discarded Predict restarts, including their pass and the reason for the call."""
    policy = record["series_policy"]
    breaks, choice = policy["chain_breaks"], policy["selection"]
    settings, counts = record["settings"], record["counts"]
    lines = ["# GIMaP series fit", "", f"{record['software_version']} · {record['saved']}", "",
             f"Source folder: {record['folder']}  ", f"File pattern: {record['pattern']}  ",
             f"Method: {settings['method']} · start: {settings['start']} · cut: {settings['side']}  ",
             f"Frames: {counts['frames']} ({counts['ok']} converged, {counts['not_converged']} unconverged, "
             f"{counts['failed']} failed, {counts['stopped']} stopped).", "", "## Selection settings", ""]
    if breaks:
        lines.append("A failed fit, an unconverged fit or a new stage breaks the chain. A rise in χ²ᵣ also "
                     f"breaks it when χ²ᵣ exceeds both {breaks['chi2_ratio_above']:g} times the preceding fit's "
                     f"value and {breaks['chi2_reduced_above']:g}.")
    else:
        lines.append("Chain restarts were disabled.")
    if choice["rule"] == "closest":
        lines.append(f"Predict takes the closest solution with p ≥ {choice['min_relative_probability']:g} p_max "
                     "in ln R, ln D and ln(σ_R/R). Without a reference it takes the most probable solution.")
    elif choice["rule"] == "HMM":
        smoothed = sum(f.get("selected_solution", {}).get("rule") == "HMM" for f in record["per_frame"])
        lines.append(f"The requested prior has s = {choice['scale']:g} and composition-change factor "
                     f"ε = {choice['composition_penalty']:.6g}. {smoothed} frames have a Viterbi selection. "
                     "Smoothed p is the forward–backward probability of the selected state.")
    elif choice["rule"] == "unrecorded":
        lines.append("This project predates the record of selection distances and single-frame probabilities.")
    elif choice["rule"] == "most probable":
        lines.append("Predict takes the most probable solution in each frame.")
    else:
        lines.append("This fit uses no Predict solution selection.")
    if settings["backward_pass"]:
        lines.append("The settings requested a backward pass.")
    lines.extend(["", "## Frames", "",
                  "Rank starts at 1. The single-frame probabilities are the evidence weights from Predict. "
                  "The 68% intervals describe parameters within the selected solution. "
                  "They exclude uncertainty about which solution to choose.", "",
                  _row("Frame", "File", "Fit state", "Pass", "Fit or restart", "Break or error", "Rule",
                       "Rank", "Kept or proposed p", "Maximum frame p", "Smoothed p"),
                  "|---|---|---|---|---|---|---|---:|---:|---:|---:|"])
    for frame in record["per_frame"]:
        attempts = frame.get("predict_attempts", [])
        if not any(a["kept"] for a in attempts):
            cells = _choice_cells(frame.get("selected_solution"))
            if cells[0] is None:
                cells[0] = frame.get("predict_choice")
            lines.append(_row(frame["frame"], frame["file"], frame["state"], frame.get("fit_pass"),
                              frame.get("start_from"), frame.get("chain_break") or frame.get("reason"),
                              *cells))
        for attempt in attempts:
            notes = " · ".join(filter(None, (attempt["chain_break"], attempt.get("error"))))
            lines.append(_row(frame["frame"], frame["file"], frame["state"], attempt["fit_pass"],
                              "Predict restart kept" if attempt["kept"] else "Predict restart discarded",
                              notes, *_choice_cells(attempt.get("choice"))))
    return "\n".join(lines) + "\n"
