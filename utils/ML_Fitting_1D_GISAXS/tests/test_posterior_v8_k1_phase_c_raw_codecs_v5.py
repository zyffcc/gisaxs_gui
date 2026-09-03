from __future__ import annotations

import pytest

from utils.ML_Fitting_1D_GISAXS.PosteriorV8.k1_phase_c_raw_codecs_v5 import (
    RAW_REPRESENTATIVE_PAYLOAD_SCHEMA,
    V5_K1_PHASE_C_RAW_ARTIFACT_VERSION,
    V5K1PhaseCRawFile,
    _parse_representative,
)
from utils.ML_Fitting_1D_GISAXS.tests.test_posterior_v8_k1_phase_c_filesystem_replay_v5 import (
    _parameter_payload,
)
from utils.ML_Fitting_1D_GISAXS.tests.test_posterior_v8_k1_phase_c_replay_v5 import (
    _candidate_payload,
    _sha,
)


@pytest.mark.parametrize(
    ("exact_values", "message"),
    (
        ([True], "JSON numeric scalars"),
        (["1.0"], "JSON numeric scalars"),
        ([0.0], "finite and strictly positive"),
        ([-1.0], "finite and strictly positive"),
        ([float("nan")], "finite and strictly positive"),
        ([float("inf")], "finite and strictly positive"),
        ([float("-inf")], "finite and strictly positive"),
    ),
    ids=("bool", "string", "zero", "negative", "nan", "positive-inf", "negative-inf"),
)
def test_candidate_exact_intensity_rejects_non_json_or_nonpositive_scalars(
    exact_values: list[object],
    message: str,
) -> None:
    """Exercise decoded payloads too; canonical file JSON rejects NaN/Infinity earlier."""

    source = _candidate_payload("candidate-1", _sha("query"))
    parameter = _parameter_payload(source)
    parameter["exact_intensity"] = exact_values
    parameter["exact_intensity_sha256"] = _sha("unreachable-invalid-intensity")
    raw = V5K1PhaseCRawFile(
        file_id="candidate-payload",
        role="representative_payload",
        relative_path="candidate-payload.json",
        file_sha256=_sha("candidate-payload-file"),
        payload={
            "schema": RAW_REPRESENTATIVE_PAYLOAD_SCHEMA,
            "version": V5_K1_PHASE_C_RAW_ARTIFACT_VERSION,
            "representative_id": source.representative_id,
            "role": source.role,
            "global_branch_key": source.global_branch_key,
            "query_context_sha256": source.query_context_sha256,
            "parameter": parameter,
        },
    )

    with pytest.raises(ValueError, match=message):
        _parse_representative(raw)
