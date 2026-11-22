"""Resolve detector settings after the frame's instrument profile has been identified."""

from dataclasses import asdict, replace

from src.gimap.shared.geometry.detector_response import DetectorResponse


def requested_response(corrections):
    return DetectorResponse(corrections.counts_per_adu, corrections.offset_adu,
                            corrections.error_model, corrections.sigma_path)


def effective_response(corrections, profile):
    requested = requested_response(corrections)
    if corrections.use_profile_response and requested == DetectorResponse() and profile is not None:
        return getattr(profile, "detector_response", requested)
    return requested


def with_profile_response(corrections, profile):
    response = effective_response(corrections, profile)
    return replace(corrections, **asdict(response))
