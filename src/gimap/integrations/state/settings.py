"""SettingsRepository 的 JSON 与 legacy global_params 实现。"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any


def _get_nested(values: dict[str, Any], key: str, default: Any) -> Any:
    current: Any = values
    for segment in key.split("."):
        if not isinstance(current, dict) or segment not in current:
            return default
        current = current[segment]
    return deepcopy(current)


def _set_nested(values: dict[str, Any], key: str, value: Any) -> None:
    segments = key.split(".")
    current = values
    for segment in segments[:-1]:
        child = current.get(segment)
        if not isinstance(child, dict):
            child = {}
            current[segment] = child
        current = child
    current[segments[-1]] = deepcopy(value)


class InMemorySettingsRepository:
    """测试、CLI 和无真实配置文件场景使用的 settings repository。"""

    def __init__(self, initial: dict[str, dict[str, Any]] | None = None):
        self._values = deepcopy(initial or {})

    def get(self, section: str, key: str, default: Any = None) -> Any:
        return _get_nested(self._values.get(section, {}), key, default)

    def set(self, section: str, key: str, value: Any) -> None:
        _set_nested(self._values.setdefault(section, {}), key, value)

    def get_section(self, section: str) -> dict[str, Any]:
        return deepcopy(self._values.get(section, {}))

    def update_section(self, section: str, values: dict[str, Any]) -> None:
        self._values.setdefault(section, {}).update(deepcopy(values))

    def snapshot(self) -> dict[str, dict[str, Any]]:
        return deepcopy(self._values)

    def reload(self) -> None:
        return None

    def save(self) -> None:
        return None


class JsonSettingsRepository(InMemorySettingsRepository):
    """保持 legacy user_parameters.json 顶层 module mapping 的实现。"""

    def __init__(
        self,
        path: str | Path,
        initial: dict[str, dict[str, Any]] | None = None,
    ):
        self.path = Path(path)
        super().__init__(initial)
        self.reload()

    def reload(self) -> None:
        if not self.path.is_file():
            return
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Settings JSON must contain a top-level object.")
        for section, values in payload.items():
            if isinstance(values, dict):
                self._values.setdefault(str(section), {}).update(deepcopy(values))

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self._values, indent=4, ensure_ascii=False),
            encoding="utf-8",
        )


class GlobalParamsSettingsRepository:
    """把 legacy GlobalParameterManager 暴露为新的 SettingsRepository。"""

    def __init__(self, manager):
        self._manager = manager

    def get(self, section: str, key: str, default: Any = None) -> Any:
        return self._manager.get_parameter(section, key, default)

    def set(self, section: str, key: str, value: Any) -> None:
        self._manager.set_parameter(section, key, value)

    def get_section(self, section: str) -> dict[str, Any]:
        return self._manager.get_module_parameters(section)

    def update_section(self, section: str, values: dict[str, Any]) -> None:
        self._manager.set_module_parameters(section, values)

    def snapshot(self) -> dict[str, dict[str, Any]]:
        return self._manager.get_all_parameters()

    def reload(self) -> None:
        user_path = getattr(self._manager, "user_params_file", None)
        if user_path:
            self._manager.load_parameters(user_path)

    def save(self) -> None:
        self._manager.save_user_parameters()
