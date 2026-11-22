"""Choose the sample frame before reducing cuts or fitting crystal orientations."""

from .pipeline_progress import PipelineStopped

MANUAL_ROLL = "Sample roll set in Analyze."
PROFILE_ROLL = "Sample roll from the instrument geometry."
GIVEN_ROLL = "Sample roll supplied for this run."
MEASURED_ROLL = "The frame estimate reaches three standard errors."
UNRESOLVED_ROLL = "The frame does not resolve a significant sample roll."
DISAGREE_ROLL = "The frame estimates disagree by more than two combined standard errors."


def choose_sample_roll(pipeline):
    bench = pipeline._workbench_with("apply_sample_roll")
    if bench is None:
        return
    if pipeline.stop is not None and pipeline.stop.is_set():
        raise PipelineStopped("sample roll")
    status = pipeline.catalog.results.status or {}
    geometry = status.get("geometry") or {}
    supplied = pipeline.options.sample_roll_deg
    previous = status.get("sample_roll_measurement") or {}
    if supplied is not None:
        if not -180 <= supplied <= 180:
            raise ValueError("Sample roll must be between −180° and 180°.")
        status = bench.apply_sample_roll(supplied)
        why = GIVEN_ROLL
    elif status.get("sample_roll_override") is not None and previous.get("origin") != "automatic":
        why = MANUAL_ROLL
    elif geometry.get("sample_roll_deg") and status.get("sample_roll_override") is None:
        why = PROFILE_ROLL
    else:
        status = bench.apply_sample_roll(None)
        measurement = status.get("sample_roll_measurement") or {}
        if measurement.get("significant"):
            why = MEASURED_ROLL
        elif measurement.get("method") == "disagreement":
            why = DISAGREE_ROLL
        else:
            why = UNRESOLVED_ROLL
    pipeline.catalog.results.status = status
    used = status["geometry"]["sample_roll_deg"]
    pipeline._decide("Sample roll ψ", f"{used:.4g}°", why)
