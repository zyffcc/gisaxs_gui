"""Real archive/tree checks for consumption after the training inventory grows."""

from hashlib import sha256
from pathlib import Path

import pytest

from PosteriorV8 import k1_balanced_full_search_replay_v5 as replay
from PosteriorV8 import k1_training_chain_plan_v5 as training
from PosteriorV8.k1_balanced_dataset_launch_plan_v5 import (
    build_v5_k1_balanced_dataset_launch_plan, write_v5_k1_balanced_dataset_launch_plan,
)
from test_posterior_v8_k1_balanced_dataset_launch_plan_v5 import launch_fixture as launch_fixture


@pytest.fixture
def historical(launch_fixture, monkeypatch):
    dust, config, _ = launch_fixture
    producer = build_v5_k1_balanced_dataset_launch_plan(config, allowed_root=dust)
    path = Path(producer["layout"]["plan"])
    path.parent.mkdir(parents=True)
    write_v5_k1_balanced_dataset_launch_plan(path, producer)
    plan = {
        "balanced_dataset": {"launch_plan_sha256": producer["plan_sha256"],
                             "launch_plan_file_sha256": sha256(path.read_bytes()).hexdigest()},
        "source": {"archive_sha256": producer["source"]["archive_sha256"],
                   "bundle_sha256": producer["source"]["bundle_sha256"]},
        "parents": [{"array_task_id": index} for index in range(60)],
    }
    args = dict(plan=plan, plan_path=dust / "search-plan.json",
                source_root=config.source_root, source_archive=config.source_archive,
                schedule_path=dust / "schedule", calibration_path=dust / "calibration",
                balanced_dataset_launch_plan_path=path)
    monkeypatch.setattr(replay.collector, "MAXWELL_DUST_ROOT", dust)
    calls = []
    def task(**kwargs):
        calls.append(kwargs["selected"]["array_task_id"])
        return {"task": calls[-1], "source_bundle_sha256": kwargs["source_bundle_sha256"]}
    monkeypatch.setattr(replay, "_task_input_identity_after_source_replay", task)
    # These 60 task file checks are tested separately; the source replay is real.
    monkeypatch.setattr(training, "K1_TRAINING_REQUIRED_SOURCE_FILES",
                        (*training.K1_TRAINING_REQUIRED_SOURCE_FILES, Path("future_module.py")))
    return args, config, calls


def test_historical_archive_survives_consumer_inventory_growth(historical):
    args, config, calls = historical
    with pytest.raises(FileNotFoundError, match="future_module"):
        training.fingerprint_v5_k1_training_source(config.source_root)
    result = replay._source_input_identity(**args)
    assert calls == list(range(60))
    assert all(row["source_bundle_sha256"] == args["plan"]["source"]["bundle_sha256"] for row in result)


@pytest.mark.parametrize("drift", ["file_hash", "self_hash", "root", "archive_path", "archive_hash",
                                  "bundle", "plan_mode", "symlink", "tree_file", "unlisted_tree_file"])
def test_historical_source_binding_remains_fail_closed(historical, drift):
    args, config, calls = historical
    if drift == "file_hash":
        args["plan"]["balanced_dataset"]["launch_plan_file_sha256"] = "f" * 64
    elif drift == "self_hash":
        args["plan"]["balanced_dataset"]["launch_plan_sha256"] = "f" * 64
    elif drift == "root":
        args["source_root"] = config.source_root.parent
    elif drift == "archive_path":
        args["source_archive"] = config.source_archive.with_suffix(".other")
    elif drift == "archive_hash":
        args["plan"]["source"]["archive_sha256"] = "f" * 64
    elif drift == "bundle":
        args["plan"]["source"]["bundle_sha256"] = "f" * 64
    elif drift == "plan_mode":
        args["balanced_dataset_launch_plan_path"].chmod(0o600)
    elif drift == "symlink":
        link = config.source_archive.parent / "linked-plan.json"
        link.symlink_to(args["balanced_dataset_launch_plan_path"])
        args["balanced_dataset_launch_plan_path"] = link
    else:
        relative = (training.K1_TRAINING_REQUIRED_SOURCE_FILES[0] if drift == "tree_file"
                    else Path("docs/research/placeholder.md"))
        target = config.source_root / relative
        target.chmod(0o600)
        target.write_text("changed historical source")
        target.chmod(0o444)
    with pytest.raises((ValueError, RuntimeError)):
        replay._source_input_identity(**args)
    assert calls == []


def test_historical_plan_replacement_during_task_replay_rejected(historical, monkeypatch):
    args, _, _ = historical
    original = replay._task_input_identity_after_source_replay
    def replace(**kwargs):
        path = args["balanced_dataset_launch_plan_path"]
        if kwargs["selected"]["array_task_id"] == 59:
            replacement = path.with_suffix(".replacement")
            replacement.write_bytes(path.read_bytes())
            replacement.chmod(0o400)
            replacement.replace(path)
        return original(**kwargs)
    monkeypatch.setattr(replay, "_task_input_identity_after_source_replay", replace)
    with pytest.raises(RuntimeError, match="changed during historical"):
        replay._source_input_identity(**args)
