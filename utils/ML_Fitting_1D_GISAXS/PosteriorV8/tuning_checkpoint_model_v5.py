"""Load a sealed tuning checkpoint and verify the actual model, not hash echoes."""

from hashlib import sha256
from pathlib import Path

from .grouped_artifact_v5 import canonical_json
from .k1_phase_b_contract_v5 import model_weights_sha256
from .k1_staging_files_v5 import lexical_no_symlinks, read_only_identity, read_only_json
from .tuning_checkpoint_runtime_v5 import V5RetainedFullCheckpoint


def _load_graph(path):
    import tensorflow as tf
    # Import registers the versioned custom layers before Keras deserialization.
    from .model_v5 import validate_model_v5_graph_contract

    model = tf.keras.models.load_model(path, compile=False)
    validate_model_v5_graph_contract(model)
    return model


def load_v5_verified_tuning_checkpoint(checkpoint: V5RetainedFullCheckpoint):
    """Return a graph-checked model only if file and loaded weights remain bound.

    Production callers run on workers. Loading does not authorize gradients or
    make this checkpoint a scientifically accepted model.
    """
    if type(checkpoint) is not V5RetainedFullCheckpoint:
        raise TypeError("expected a typed retained full checkpoint")
    path = checkpoint.checkpoint_path
    before = read_only_identity(path, "tuning model checkpoint")
    if before["mode_octal"] != "0400" or before["sha256"] != checkpoint.checkpoint_artifact_sha256:
        raise ValueError("tuning checkpoint is not the expected sealed artifact")
    model = _load_graph(path)
    if model_weights_sha256(model) != checkpoint.checkpoint_weights_sha256:
        raise ValueError("loaded tuning model weights differ from the frozen checkpoint")
    after = read_only_identity(path, "tuning model checkpoint")
    if after != before:
        raise RuntimeError("tuning checkpoint changed during model loading")
    return model


def read_v5_retained_checkpoints_from_training_result(
    result_path, *, expected_result_file_sha256, expected_result_sha256,
    expected_full_epochs, expected_warmup_epochs,
):
    """Bind every full epoch to an actual sealed training result and file.

    The caller pins the result from the seed handoff and supplies the frozen
    epoch schedule. This is not model selection: the tuning runner subsequently
    verifies loaded graph/weight identities and evaluates every returned epoch.
    """
    from .grouped_training_v5 import (
        V5_GROUPED_TRAINER_SCHEMA, V5_GROUPED_TRAINER_VERSION,
        FULL_CHECKPOINT_SELECTION_STATUS, _full_checkpoint_relative_path,
    )

    if (type(expected_full_epochs) is not int or expected_full_epochs < 1
            or type(expected_warmup_epochs) is not int or expected_warmup_epochs < 0):
        raise ValueError("tuning requires a positive full-epoch and nonnegative warmup schedule")
    path = lexical_no_symlinks(Path(result_path), "training result").resolve(strict=True)
    manifest, before = read_only_json(path, "training result")
    if before["mode_octal"] != "0400" or before["sha256"] != expected_result_file_sha256:
        raise ValueError("training result differs from its sealed file binding")
    core = dict(manifest)
    logical_sha = core.pop("result_sha256", None)
    if (logical_sha != expected_result_sha256
            or sha256(canonical_json(core).encode()).hexdigest() != logical_sha):
        raise ValueError("training result logical identity does not reproduce")
    if (manifest.get("schema") != V5_GROUPED_TRAINER_SCHEMA
            or manifest.get("version") != V5_GROUPED_TRAINER_VERSION
            or manifest.get("status") != "complete"):
        raise ValueError("tuning requires a completed supported training result")
    selection = manifest.get("checkpoint_selection", {})
    status = manifest.get("paper_model_status", {})
    if not isinstance(selection, dict) or not isinstance(status, dict):
        raise ValueError("training result selection/status must be objects")
    if (selection.get("status") != FULL_CHECKPOINT_SELECTION_STATUS
            or selection.get("paper_selected_checkpoint") is not None
            or selection.get("validation_objective_alone_selects_paper_checkpoint") is not False
            or status.get("full_training_completed") is not True
            or status.get("paper_checkpoint_candidates_eligible_for_external_selection") is not True
            or type(status.get("full_checkpoint_count")) is not int
            or type(status.get("expected_full_checkpoint_count")) is not int
            or status.get("full_checkpoint_count") != expected_full_epochs
            or status.get("expected_full_checkpoint_count") != expected_full_epochs
            or status.get("paper_model_eligible") is not False):
        raise ValueError("training result has no complete pending full-checkpoint candidate inventory")
    rows = manifest.get("full_checkpoints")
    if not isinstance(rows, list) or len(rows) != expected_full_epochs:
        raise ValueError("training result dropped a retained full checkpoint")
    checkpoints, identities = [], []
    for full_epoch, row in enumerate(rows, 1):
        if not isinstance(row, dict) or set(row) != {
            "epoch", "phase_epoch", "relative_path", "file_sha256", "weights_sha256",
        }:
            raise ValueError("retained full checkpoint record fields differ")
        epoch = expected_warmup_epochs + full_epoch
        relative = _full_checkpoint_relative_path(epoch=epoch, phase_epoch=full_epoch)
        if (type(row["epoch"]) is not int or row["epoch"] != epoch
                or type(row["phase_epoch"]) is not int or row["phase_epoch"] != full_epoch
                or row["relative_path"] != relative.as_posix()):
            raise ValueError("retained checkpoint escaped the complete frozen epoch schedule")
        checkpoint_path = lexical_no_symlinks(path.parent / relative, "retained checkpoint").resolve(strict=True)
        identity = read_only_identity(checkpoint_path, "retained checkpoint")
        if identity["mode_octal"] != "0400" or identity["sha256"] != row["file_sha256"]:
            raise ValueError("retained checkpoint differs from its sealed file binding")
        checkpoints.append(V5RetainedFullCheckpoint(
            full_epoch=full_epoch, checkpoint_path=checkpoint_path,
            checkpoint_artifact_sha256=row["file_sha256"],
            checkpoint_weights_sha256=row["weights_sha256"], training_result_sha256=logical_sha,
        ))
        identities.append(identity)
    if len({item.checkpoint_artifact_sha256 for item in checkpoints}) != expected_full_epochs:
        raise ValueError("retained full checkpoint artifact identities must be unique")
    for checkpoint, identity in zip(checkpoints, identities, strict=True):
        if read_only_identity(checkpoint.checkpoint_path, "retained checkpoint") != identity:
            raise RuntimeError("retained checkpoint changed during inventory loading")
    if read_only_identity(path, "training result") != before:
        raise RuntimeError("training result changed during checkpoint inventory loading")
    return tuple(checkpoints)
