"""Where the assistant keeps the API key, run records, tables and tool requests."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from src.gimap.shared.version import SOFTWARE

from src.gimap.features.assistant.infrastructure import (
    FEATURE_REQUESTS_FILE,
    KEY_FILE,
    RUNS_FOLDER,
    ApiKeyStore,
    JsonResultStore,
)


def test_a_saved_key_is_used_before_the_environment(tmp_path: Path, monkeypatch) -> None:
    from src.gimap.features.assistant.infrastructure import storage
    monkeypatch.setattr(storage, "system_credentials", lambda folder: None)
    keys = ApiKeyStore(tmp_path)
    assert keys.load() is None
    assert keys.source({}) in ("", "ant CLI login profile")
    assert keys.source({"ANTHROPIC_API_KEY": "x"}) == "environment variable ANTHROPIC_API_KEY"
    keys.save("  sk-ant-test  ")
    assert keys.load() == "sk-ant-test"
    assert (tmp_path / KEY_FILE).read_text("utf-8") == "sk-ant-test\n"
    assert keys.source({"ANTHROPIC_API_KEY": "x"}) == "API key saved in GIMaP"
    assert keys.delete() and keys.load() is None and not keys.delete()


def test_keys_move_from_plaintext_on_save_and_can_be_removed(tmp_path, monkeypatch):
    from src.gimap.features.assistant.infrastructure import storage

    class Credentials(dict):
        read = dict.get
        write = dict.__setitem__

        def delete(self, account):
            return self.pop(account, None) is not None

    backend = Credentials()
    monkeypatch.setattr(storage, "system_credentials", lambda folder: backend)
    claude = ApiKeyStore(tmp_path)
    others = storage.ProviderKeyStore(tmp_path, environ={})
    claude.path.write_text("old-claude", encoding="utf-8")
    others.path.write_text('{"qwen":"old-qwen","openai":"old-openai"}', encoding="utf-8")
    assert claude.load() == "old-claude" and others.load("qwen") == "old-qwen"
    claude.save("new-claude")
    others.save("qwen", "new-qwen")
    assert not claude.path.exists() and json.loads(others.path.read_text("utf-8")) == {"openai": "old-openai"}
    assert backend == {"anthropic": "new-claude", "qwen": "new-qwen"}
    assert others.load("qwen") == "new-qwen" and claude.load() == "new-claude"
    assert claude.delete() and others.delete("qwen") and others.delete("openai") and not backend

    def locked(*args):
        raise OSError("locked")

    monkeypatch.setattr(backend, "write", locked)
    claude.path.write_text("legacy", encoding="utf-8")
    with pytest.raises(OSError, match="locked"):
        claude.save("replacement")
    assert claude.load() == "legacy"


def test_windows_credential_targets_belong_to_one_user_data_folder(tmp_path, monkeypatch):
    from src.gimap.features.assistant.infrastructure import system_keys

    class MissingCredential(Exception):
        winerror = 1168

    entries = {}

    def read(target, kind):
        if target not in entries:
            raise MissingCredential()
        return {"CredentialBlob": entries[target].encode("utf-16-le")}

    def write(item):
        entries[item["TargetName"]] = item["CredentialBlob"]

    def delete(target, kind):
        if target not in entries:
            raise MissingCredential()
        del entries[target]

    # Exercise the Windows backend even on machines without Windows Credential Manager.
    api = SimpleNamespace(CredRead=read, CredWrite=write, CredDelete=delete, CRED_TYPE_GENERIC=1,
                          CRED_PERSIST_LOCAL_MACHINE=2)
    wrapper = ModuleType("win32ctypes.pywin32")
    wrapper.win32cred = api
    wrapper.pywintypes = SimpleNamespace(error=MissingCredential)
    monkeypatch.setitem(sys.modules, "win32ctypes.pywin32", wrapper)
    monkeypatch.setattr(sys, "platform", "win32")
    one = system_keys.system_credentials(tmp_path / "one")
    two = system_keys.system_credentials(tmp_path / "two")
    assert one.read("anthropic") is None
    one.write("anthropic", "sample-key")
    assert one.read("anthropic") == "sample-key" and two.read("anthropic") is None
    assert one.delete("anthropic") and not one.delete("anthropic") and not entries


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
    assert json.loads(Path(table).read_text("utf-8")) == {"peaks": [1.0], "software_version": SOFTWARE}
    assert Path(table).name == "frame_001_assistant.json"

    run = store.save_run({"frame": "C:/data/frame_001.tif", "state": "completed", "steps": []})
    assert Path(run).parent == tmp_path / "data" / RUNS_FOLDER
    assert Path(run).name.endswith("_frame_001.json")
    record = json.loads(Path(run).read_text("utf-8"))
    assert record["state"] == "completed"
    assert record["software_version"] == SOFTWARE

    store.append_feature_request({"capability": "pole figure", "reason": "r"})
    store.append_feature_request({"capability": "indexing", "reason": "r"})
    lines = (tmp_path / "data" / FEATURE_REQUESTS_FILE).read_text("utf-8").splitlines()
    assert [json.loads(line)["capability"] for line in lines] == ["pole figure", "indexing"]

    assert JsonResultStore(None).save_run({"frame": None}) is None
