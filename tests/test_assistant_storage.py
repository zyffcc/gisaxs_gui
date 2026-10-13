"""Where the assistant keeps the API key, run records, tables and tool requests."""

from __future__ import annotations

import json
from pathlib import Path

from src.gimap.features.assistant.infrastructure import (
    FEATURE_REQUESTS_FILE,
    KEY_FILE,
    RUNS_FOLDER,
    ApiKeyStore,
    JsonResultStore,
)


def test_a_saved_key_is_used_before_the_environment(tmp_path: Path) -> None:
    keys = ApiKeyStore(tmp_path)
    assert keys.load() is None
    assert keys.source({}) in ("", "ant CLI login profile")
    assert keys.source({"ANTHROPIC_API_KEY": "x"}) == "environment variable ANTHROPIC_API_KEY"
    keys.save("  sk-ant-test  ")
    assert keys.load() == "sk-ant-test"
    assert (tmp_path / KEY_FILE).read_text("utf-8") == "sk-ant-test\n"
    assert keys.source({"ANTHROPIC_API_KEY": "x"}) == "API key saved in GIMaP"
    assert keys.delete() and keys.load() is None and not keys.delete()


def test_an_in_memory_session_has_nowhere_to_save_a_key() -> None:
    keys = ApiKeyStore(None)
    assert keys.load() is None and not keys.delete()
    try:
        keys.save("sk")
    except OSError:
        pass
    else:
        raise AssertionError("saving without a data folder must fail")


def test_results_runs_and_tool_requests_are_written_as_json(tmp_path: Path) -> None:
    store = JsonResultStore(tmp_path / "data")
    table = store.write_tables(str(tmp_path / "frames" / "gimap_analysis"), "frame_001", {"peaks": [1.0]})
    assert json.loads(Path(table).read_text("utf-8")) == {"peaks": [1.0]}
    assert Path(table).name == "frame_001_assistant.json"

    run = store.save_run({"frame": "C:/data/frame_001.tif", "state": "completed", "steps": []})
    assert Path(run).parent == tmp_path / "data" / RUNS_FOLDER
    assert Path(run).name.endswith("_frame_001.json")
    assert json.loads(Path(run).read_text("utf-8"))["state"] == "completed"

    store.append_feature_request({"capability": "pole figure", "reason": "r"})
    store.append_feature_request({"capability": "indexing", "reason": "r"})
    lines = (tmp_path / "data" / FEATURE_REQUESTS_FILE).read_text("utf-8").splitlines()
    assert [json.loads(line)["capability"] for line in lines] == ["pole figure", "indexing"]

    assert JsonResultStore(None).save_run({"frame": None}) is None
