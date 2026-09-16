"""Filesystem-only tests; these do not claim scientific training acceptance."""

import os

import pytest

from utils.ML_Fitting_1D_GISAXS.PosteriorV8 import k1_training_chain_runtime_v5 as runtime


def test_seed_seals_models_before_exclusive_completion(tmp_path):
    output = tmp_path / "fresh-seed"
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(parents=True)
    model = checkpoints / "epoch.keras"
    model.write_bytes(b"test-only-model")
    result = output / "result_manifest.json"
    result.write_text("{}")
    runtime._seal_seed_artifacts(output)
    assert model.stat().st_mode & 0o777 == 0o400
    assert result.stat().st_mode & 0o777 == 0o400
    receipt = output / runtime.K1_SEED_BINDING_FILENAME
    assert not receipt.exists()
    runtime._write_json_exclusive(receipt, {"test_only": True})
    assert receipt.stat().st_mode & 0o777 == 0o400
    assert receipt.stat().st_nlink == 1
    assert receipt.stat().st_mtime_ns >= result.stat().st_mtime_ns
    with pytest.raises(FileExistsError):
        runtime._write_json_exclusive(receipt, {"changed": True})


@pytest.mark.parametrize("kind", ["hardlink", "symlink", "fifo"])
def test_unsafe_tree_rejected_before_any_permission_change(tmp_path, kind):
    output = tmp_path / "fresh-seed"
    output.mkdir()
    regular = output / "first-model.keras"
    regular.write_bytes(b"fresh")
    regular.chmod(0o600)
    older = tmp_path / "older-model.keras"
    older.write_bytes(b"keep")
    older.chmod(0o600)
    unsafe = output / "unsafe"
    if kind == "hardlink":
        os.link(older, unsafe)
    elif kind == "symlink":
        unsafe.symlink_to(older)
    else:
        os.mkfifo(unsafe)
    with pytest.raises(ValueError, match="unsafe"):
        runtime._seal_seed_artifacts(output)
    assert regular.stat().st_mode & 0o777 == 0o600
    assert older.stat().st_mode & 0o777 == 0o600
    assert older.read_bytes() == b"keep"
    assert not (output / runtime.K1_SEED_BINDING_FILENAME).exists()


def test_seed_reader_requires_exact_private_read_only_mode(tmp_path):
    model = tmp_path / "model.keras"
    model.write_bytes(b"test")
    model.chmod(0o444)
    with pytest.raises(ValueError, match="0400"):
        runtime._sealed_sha(model)
    model.chmod(0o400)
    assert len(runtime._sealed_sha(model)) == 64
