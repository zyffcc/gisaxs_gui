"""JSON storage for instrument profiles (named detector geometries)."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Optional

from src.gimap.shared.geometry.instrument_profile import (
    SCHEMA_VERSION,
    InstrumentProfile,
    best_profile,
)


class InMemoryInstrumentProfileRepository:
    """Profiles kept in memory, for tests and sessions that must not touch disk."""

    def __init__(self, profiles: list[InstrumentProfile] | None = None):
        self._profiles = {profile.name: profile for profile in profiles or ()}

    def load_all(self) -> list[InstrumentProfile]:
        return sorted(self._profiles.values(), key=lambda item: item.name)

    def save(self, profile: InstrumentProfile) -> None:
        self._profiles[profile.name] = profile

    def delete(self, name: str) -> bool:
        return self._profiles.pop(name, None) is not None

    def find(self, name: str) -> Optional[InstrumentProfile]:
        return self._profiles.get(name)

    def match(
        self, *, detector_name: Optional[str], shape: Optional[tuple[int, int]]
    ) -> Optional[InstrumentProfile]:
        return best_profile(self.load_all(), detector_name=detector_name, shape=shape)


class JsonInstrumentProfileRepository:
    """All profiles in one human-readable JSON file, written atomically."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def load_all(self) -> list[InstrumentProfile]:
        if not self.path.is_file():
            return []
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        version = int(payload.get("schema_version", 0))
        if version > SCHEMA_VERSION:
            raise ValueError(
                f"{self.path} was written by a newer GIMaP (schema {version}); "
                f"this version reads schema {SCHEMA_VERSION}."
            )
        return [InstrumentProfile.from_dict(item) for item in payload.get("profiles", [])]

    def save(self, profile: InstrumentProfile) -> None:
        """Insert or replace the profile with the same name."""
        profiles = [item for item in self.load_all() if item.name != profile.name]
        profiles.append(profile)
        self._write(profiles)

    def delete(self, name: str) -> bool:
        profiles = self.load_all()
        kept = [item for item in profiles if item.name != name]
        if len(kept) == len(profiles):
            return False
        self._write(kept)
        return True

    def find(self, name: str) -> Optional[InstrumentProfile]:
        return next((item for item in self.load_all() if item.name == name), None)

    def match(
        self, *, detector_name: Optional[str], shape: Optional[tuple[int, int]]
    ) -> Optional[InstrumentProfile]:
        return best_profile(self.load_all(), detector_name=detector_name, shape=shape)

    def _write(self, profiles: list[InstrumentProfile]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": SCHEMA_VERSION,
            "profiles": [item.to_dict() for item in sorted(profiles, key=lambda item: item.name)],
        }
        handle, temporary = tempfile.mkstemp(
            prefix=".instrument_profiles.", suffix=".json", dir=str(self.path.parent)
        )
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, indent=2, ensure_ascii=False)
                stream.write("\n")
            os.replace(temporary, self.path)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise
