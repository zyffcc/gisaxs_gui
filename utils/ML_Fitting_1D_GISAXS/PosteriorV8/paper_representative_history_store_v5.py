"""Sealed representative snapshots bound to an independently replayed trace."""

from __future__ import annotations

import json
from pathlib import Path

from .grouped_artifact_v5 import canonical_json
from .k1_staging_files_v5 import lexical_no_symlinks, read_only_bytes_identity
from .paper_budget_evaluator_v5 import V5MethodExactCallTrace
from .paper_representative_history_v5 import V5RepresentativeHistory
from .read_only_json_publication_v5 import publish_read_only_canonical_json


def read_v5_representative_history(
    path: Path,
    *,
    trace: V5MethodExactCallTrace,
    expected_file_sha256: str,
    expected_history_sha256: str,
) -> V5RepresentativeHistory:
    """Require external file/logical identities, not just a self-consistent JSON."""

    raw, identity = read_only_bytes_identity(
        path, "representative history", maximum_bytes=64 * 1024 * 1024,
    )
    if identity["mode_octal"] != "0400":
        raise ValueError("representative history must have mode 0400")
    if identity["sha256"] != expected_file_sha256:
        raise ValueError("representative history file SHA-256 differs")
    payload = json.loads(raw)
    history = V5RepresentativeHistory.from_payload(payload, trace=trace)
    if history.sha256 != expected_history_sha256:
        raise ValueError("representative history logical SHA-256 differs")
    # Canonical bytes also reject duplicate JSON keys and alternate encodings;
    # parsing alone would silently keep the last duplicate value.
    if canonical_json(history.to_payload()).encode("utf-8") != raw:
        raise ValueError("representative history is not canonical JSON")
    return history


def publish_v5_representative_history(
    path: Path, history: V5RepresentativeHistory,
) -> str:
    """Publish and replay snapshots after the physical trace has been sealed.

    The caller must retain and independently verify the referenced physical
    trace artifact. This file carries no replacement parameter payloads and is
    not itself a completion marker or a scientific acceptance authorization.
    """

    if type(history) is not V5RepresentativeHistory:
        raise TypeError("history must be an exact V5RepresentativeHistory")
    if history.trace.trace_artifact_sha256 == "0" * 64:
        raise ValueError("cannot publish history for a provisional trace")
    target = lexical_no_symlinks(path, "representative history")
    file_sha = publish_read_only_canonical_json(target, history.to_payload())
    restored = read_v5_representative_history(
        target, trace=history.trace, expected_file_sha256=file_sha,
        expected_history_sha256=history.sha256,
    )
    if restored.to_payload() != history.to_payload():
        raise RuntimeError("representative history failed its publication round trip")
    return file_sha
