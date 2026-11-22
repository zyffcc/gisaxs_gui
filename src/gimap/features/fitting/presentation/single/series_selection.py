"""Predict solution choices and the policy of the completed run, in the interface language."""

from src.gimap.app.presentation.i18n import tr, trf

from ...application.series_policy import series_policy

CHOSEN_BY = "Chosen by"
RANK = "Rank"
RAW_P = "p chosen"
MAX_P = "p max"
SMOOTH_P = "p smoothed"
CHOICE_HEADERS = (CHOSEN_BY, RANK, RAW_P, MAX_P)
CHOICE_TIPS = {
    CHOSEN_BY: "How the Predict solution was selected. A discarded retry did not supply the fitted values.",
    RANK: "Rank of the chosen solution by its probability in this frame (1 is most probable).",
    RAW_P: "Probability of the chosen solution using this frame alone.",
    MAX_P: "Largest solution probability using this frame alone.",
    SMOOTH_P: "Probability of the chosen solution given the series, from forward–backward.",
}
INTERVAL_NOTE = "The 68% interval is conditional on the chosen solution. It excludes uncertainty about which solution to use."


def choice_headers(settings):
    if settings.method != "predict" and settings.start != "chain":
        return ()
    return CHOICE_HEADERS + ((SMOOTH_P,) if settings.method == "predict" and settings.smooth else ())


def choice_cells(frame, headers):
    """One line per Predict attempt, in the same order across columns when both passes retried."""
    if not headers:
        return []
    attempts = frame.predict_attempts
    records = [(attempt.choice, attempt) for attempt in attempts] if attempts else [(frame.selection, None)]
    values = [[] for _ in headers]
    for choice, attempt in records:
        if choice is None:
            line = [tr("Predict failed") if attempt else "", "", "", "", ""]
        else:
            label = trf("closest to frame {n}", n=choice.reference_frame) if choice.rule == "closest" else tr(choice.rule)
            if attempt is not None and not attempt.kept:
                label = trf("{rule} (discarded retry)", rule=label)
            line = [label, str(choice.rank), f"{choice.probability:.4g}", f"{choice.top_probability:.4g}",
                    "" if choice.smoothed_probability is None else f"{choice.smoothed_probability:.4g}"]
        if len(records) > 1:
            line[0] = f"{tr(attempt.fit_pass)} · {line[0]}"
        for column, value in zip(values, line):
            column.append(value)
    return ["\n".join(column) for column in values]


def selection_summary(settings):
    policy = series_policy(settings)
    selection = policy["selection"]
    rule = selection["rule"]
    if rule == "unused":
        return ""
    if rule == "unrecorded":
        return tr("This project did not record Predict solution probabilities or its selection distance.")
    chain = policy["chain_breaks"]
    reason = (trf("fit failure, non-convergence, new stage, or χ²ᵣ above {floor} and {ratio}× the previous χ²ᵣ",
                  floor=f"{chain['chi2_reduced_above']:g}", ratio=f"{chain['chi2_ratio_above']:g}")
              if chain else tr("not chained"))
    if rule == "closest":
        choice = trf("closest eligible solution (p ≥ {fraction} p max)", fraction=f"{selection['min_relative_probability']:g}")
    elif rule == "HMM":
        choice = trf("HMM (s = {scale}, ε = {penalty})", scale=f"{selection['scale']:g}",
                     penalty=f"{selection['composition_penalty']:g}")
    else:
        choice = tr("most probable")
    return trf("Chain breaks at: {reason}; Predict solution chosen by: {choice}.", reason=reason, choice=choice)
