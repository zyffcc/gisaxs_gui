"""Local files of the assistant: the optional saved API key, result tables, reports and tool requests."""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Mapping, Optional

KEY_FILE = "claude_api_key.txt"
PROVIDER_KEYS_FILE = "assistant_provider_keys.json"
FEATURE_REQUESTS_FILE = "assistant_feature_requests.jsonl"
RUNS_FOLDER = "assistant_runs"


def _atomic_write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    return path


class ApiKeyStore:
    """An API key saved in the user data folder (plain text, readable by this user only).

    The environment variable ANTHROPIC_API_KEY or an ``ant auth login`` profile
    work without it; a key saved here is used in preference to them.
    """

    def __init__(self, data_dir: Optional[str | Path]):
        self.path = Path(data_dir) / KEY_FILE if data_dir is not None else None

    def load(self) -> Optional[str]:
        if self.path is None or not self.path.is_file():
            return None
        key = self.path.read_text(encoding="utf-8").strip()
        return key or None

    def save(self, key: str) -> None:
        if self.path is None:
            raise OSError("There is no user data folder to save the key in.")
        _atomic_write(self.path, key.strip() + "\n")
        try:
            os.chmod(self.path, 0o600)
        except OSError:
            pass

    def delete(self) -> bool:
        if self.path is None or not self.path.is_file():
            return False
        self.path.unlink()
        return True

    def source(self, environ: Mapping[str, str] = os.environ) -> str:
        """Where the credentials of the next run come from ("" when none are found)."""
        if self.load():
            return "API key saved in GIMaP"
        if environ.get("ANTHROPIC_API_KEY"):
            return "environment variable ANTHROPIC_API_KEY"
        if environ.get("ANTHROPIC_AUTH_TOKEN"):
            return "environment variable ANTHROPIC_AUTH_TOKEN"
        if environ.get("ANTHROPIC_PROFILE") or (Path.home() / ".config" / "anthropic").is_dir():
            return "ant CLI login profile"
        return ""


class ProviderKeyStore:
    """API keys of the other AI providers, one per provider, in the user data folder (this user only).

    A provider's usual environment variable (``DEEPSEEK_API_KEY`` …) works without a saved key;
    a saved key takes precedence.
    """

    def __init__(self, data_dir: Optional[str | Path], environ: Optional[Mapping[str, str]] = None):
        self.path = Path(data_dir) / PROVIDER_KEYS_FILE if data_dir is not None else None
        self._environ = os.environ if environ is None else environ

    def _all(self) -> dict:
        if self.path is None or not self.path.is_file():
            return {}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    def saved(self, provider: str) -> Optional[str]:
        key = str(self._all().get(provider) or "").strip()
        return key or None

    def load(self, provider: str, env_key: str = "") -> Optional[str]:
        """The key to use: the saved one, else the provider's environment variable."""
        return self.saved(provider) or (self._environ.get(env_key, "").strip() if env_key else "") or None

    def source(self, provider: str, env_key: str = "") -> str:
        if self.saved(provider):
            return "API key saved in GIMaP"
        if env_key and self._environ.get(env_key, "").strip():
            return f"environment variable {env_key}"
        return ""

    def save(self, provider: str, key: str) -> None:
        if self.path is None:
            raise OSError("There is no user data folder to save the key in.")
        data = self._all()
        data[provider] = key.strip()
        _atomic_write(self.path, json.dumps(data, indent=1) + "\n")
        try:
            os.chmod(self.path, 0o600)
        except OSError:
            pass

    def delete(self, provider: str) -> bool:
        data = self._all()
        if provider not in data or self.path is None:
            return False
        del data[provider]
        _atomic_write(self.path, json.dumps(data, indent=1) + "\n")
        return True


class JsonResultStore:
    """Tables next to the data, reports where the user chooses, tool requests in the data folder."""

    def __init__(self, data_dir: Optional[str | Path]):
        self.data_dir = Path(data_dir) if data_dir is not None else None

    def write_tables(self, folder: str, stem: str, payload: dict) -> str:
        path = Path(folder) / f"{stem}_assistant.json"
        return str(_atomic_write(path, json.dumps(payload, indent=2, ensure_ascii=False) + "\n"))

    def save_text(self, path: str | Path, text: str) -> str:
        return str(_atomic_write(Path(path), text))

    def save_run(self, record: dict) -> Optional[str]:
        """Keep the full record of a run (steps, results, transcript) for later review."""
        if self.data_dir is None:
            return None
        stamp = time.strftime("%Y%m%d-%H%M%S")
        stem = Path(str(record.get("frame") or "frame")).stem or "frame"
        path = self.data_dir / RUNS_FOLDER / f"{stamp}_{stem}.json"
        text = json.dumps(record, indent=1, ensure_ascii=False, default=str)
        return str(_atomic_write(path, text + "\n"))

    def append_feature_request(self, entry: dict) -> None:
        if self.data_dir is None:
            return
        self.data_dir.mkdir(parents=True, exist_ok=True)
        with (self.data_dir / FEATURE_REQUESTS_FILE).open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(entry, ensure_ascii=False) + "\n")


__all__ = [
    "ApiKeyStore",
    "FEATURE_REQUESTS_FILE",
    "JsonResultStore",
    "KEY_FILE",
    "PROVIDER_KEYS_FILE",
    "ProviderKeyStore",
    "RUNS_FOLDER",
]
