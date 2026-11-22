"""Choose the sample frame before reducing cuts or fitting crystal orientations."""

from src.gimap.shared.numbers import applied_angle, format_number

from .pipeline_geometry import INCIDENCE_ITEM
from .pipeline_progress import PipelineStopped

MANUAL_ROLL = "Sample roll set in Analyze."
PROFILE_ROLL = "Sample roll from the instrument geometry."
GIVEN_ROLL = "Sample roll supplied for this run."
MEASURED_ROLL = "The frame estimate reaches three standard errors."
UNRESOLVED_ROLL = "The frame does not resolve a significant sample roll."
DISAGREE_ROLL = "The frame estimates disagree by more than two combined standard errors."


def choose_frame_angles(pipeline):
    bench = pipeline._workbench_with("apply_frame_angles")
    if bench is None:
        return
    if pipeline.stop is not None and pipeline.stop.is_set():
        raise PipelineStopped("frame angles")
    status = pipeline.catalog.results.status or {}
    geometry = status.get("geometry") or {}
    supplied = pipeline.options.sample_roll_deg
    previous = status.get("sample_roll_measurement") or {}
    if supplied is not None:
        if not -180 <= supplied <= 180:
            raise ValueError("Sample roll must be between −180° and 180°.")
        policy = "set"
        why = GIVEN_ROLL
    elif status.get("sample_roll_override") is not None and previous.get("origin") != "automatic":
        policy, why = "keep", MANUAL_ROLL
    elif geometry.get("sample_roll_deg") and status.get("sample_roll_override") is None:
        policy, why = "keep", PROFILE_ROLL
    else:
        policy, why = "measure", UNRESOLVED_ROLL
    status = bench.apply_frame_angles(policy, supplied, pipeline._incidence, pipeline._incidence_source,
                                      pipeline.options.keep_incidence or status.get("incidence_locked", False),
                                      pipeline.options.yoneda_critical_deg, pipeline.options.yoneda_material)
    if policy == "measure":
        measurement = status.get("sample_roll_measurement") or {}
        if measurement.get("significant"):
            why = MEASURED_ROLL
        elif measurement.get("method") == "disagreement":
            why = DISAGREE_ROLL
    pipeline.catalog.results.status = status
    used = status["geometry"]["sample_roll_deg"]
    pipeline._decide("Sample roll ψ", f"{applied_angle(used, status.get('sample_roll_measurement'))}°", why)
    incidence = status.get("incidence_measurement")
    if incidence:
        final = status["geometry"]["incidence_deg"]
        pipeline._decide("Incidence angle from frame", f"αi = {applied_angle(final, incidence)}°", incidence_reason(incidence))
        if incidence.get("applied_deg") is not None or (incidence.get("reference_deg") or 0) > 0:
            pipeline._incidence = final
            pipeline._run.attention[:] = [entry for entry in pipeline._run.attention if entry["item"] != INCIDENCE_ITEM]


def incidence_reason(record):
    reference = record.get("override_deg", record["reference_deg"])
    source = record.get("override_source", record["reference_source"])
    value, error = record.get("deg"), record.get("error_deg")
    nominal = (f"{source} αi = {reference:g}°" if reference is not None
               else "no nominal αi")
    if value is None:
        return f"{nominal}. The frame estimate is unresolved ({record['method']})."
    measured = f"frame αi = {format_number(value, error)}° ({record['method']})"
    if reference is not None and reference > 0:
        apart = "differ by more than 3σ" if abs(value - reference) > 3 * error else "agree within 3σ"
        comparison = f"{nominal} and {measured} {apart}."
    else:
        comparison = f"{nominal}. {measured}."
    reasons = {"disagreement": "The reduction uses the measured angle. Select Keep entered αi or pass --keep-incidence to override.",
               "measured": "The measured angle supplies the missing incidence.",
               "agreement": "The reduction retains the nominal angle.",
               "user_override": "The user selected the nominal angle."}
    return comparison + " " + reasons[record["decision"]]


def incidence_lines(report):
    record = report.get("incidence_measurement")
    if not record:
        return []
    lines = ["- incidence measurement: " + incidence_reason(record)]
    methods = ("horizon", "specular", "yoneda") if report.get("procedure") == "gisaxs" else ("horizon", "specular")
    for method in methods:
        estimate = record.get(method)
        value = format_number(estimate["deg"], estimate["error_deg"]) + "°" if estimate else "unresolved"
        if method == "yoneda" and not record.get("critical_angle_deg"):
            value = "requires the critical angle of the material giving the band"
        if method == "yoneda" and record.get("yoneda_material"):
            material = record["yoneda_material"]
            wavelength = record["material_wavelength_angstrom"]
            optics = record.get("material_optics_error") or f"αc = {record['critical_angle_deg']:.5g}°"
            value += f" ({material}, λ = {wavelength:g} Å, {optics})"
        lines.append(f"  - {method}: {value}")
    return lines
