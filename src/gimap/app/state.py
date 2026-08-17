"""显式 project / feature session state。"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, TypeVar, runtime_checkable


@runtime_checkable
class FeatureState(Protocol):
    """可被 ProjectState 保存与恢复的 feature state 最小协议。"""

    def snapshot(self) -> dict[str, Any]: ...

    def restore(self, state: dict[str, Any]) -> None: ...


FeatureStateT = TypeVar("FeatureStateT", bound=FeatureState)


@dataclass
class ProjectState:
    """当前项目身份、dirty 状态及已注册 feature state 的聚合根。"""

    project_path: str | None = None
    dirty: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    _features: dict[str, FeatureState] = field(default_factory=dict, init=False, repr=False)
    _pending_feature_state: dict[str, dict[str, Any]] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    def feature_state(
        self,
        name: str,
        factory: Callable[[], FeatureStateT],
    ) -> FeatureStateT:
        existing = self._features.get(name)
        if existing is not None:
            return existing  # type: ignore[return-value]
        state = factory()
        pending = self._pending_feature_state.pop(name, None)
        if pending is not None:
            state.restore(deepcopy(pending))
        self._features[name] = state
        return state

    def mark_dirty(self, dirty: bool = True) -> None:
        self.dirty = bool(dirty)

    def snapshot(self) -> dict[str, Any]:
        features = {
            name: deepcopy(state.snapshot())
            for name, state in self._features.items()
        }
        features.update(deepcopy(self._pending_feature_state))
        return {
            "project_path": self.project_path,
            "dirty": self.dirty,
            "metadata": deepcopy(self.metadata),
            "features": features,
        }

    def restore(self, state: dict[str, Any]) -> None:
        self.project_path = state.get("project_path")
        self.dirty = bool(state.get("dirty", False))
        metadata = state.get("metadata", {})
        self.metadata = deepcopy(metadata) if isinstance(metadata, dict) else {}
        features = state.get("features", {})
        feature_payloads = features if isinstance(features, dict) else {}
        self._pending_feature_state = {
            str(name): deepcopy(payload)
            for name, payload in feature_payloads.items()
            if isinstance(payload, dict)
        }
        for name, feature_state in self._features.items():
            payload = self._pending_feature_state.pop(name, None)
            if payload is not None:
                feature_state.restore(payload)
