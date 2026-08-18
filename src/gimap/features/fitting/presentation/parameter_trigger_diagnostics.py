"""Read-only diagnostics and explicit commit commands for parameter triggers."""

from __future__ import annotations

from typing import Dict, Optional


class ParameterTriggerDiagnosticsMixin:
    def debug_dump_meta(self, verbose: bool = False) -> Dict[str, dict]:
        snapshot = {}
        for widget_id, info in self._meta_registry.items():
            widget = info["widget"]
            try:
                current_value = widget.value() if hasattr(widget, "value") else None
            except Exception:
                current_value = None
            meta = info["meta"]
            snapshot[widget_id] = {
                "current": current_value,
                "last": info.get("last_value"),
                "pending": info.get("pending_value"),
                "debounce_ms": meta.get("debounce_ms"),
                "persist": meta.get("persist"),
                "trigger_fit": meta.get("trigger_fit"),
                "param": meta.get("param"),
                "particle_id": meta.get("particle_id"),
                "shape": meta.get("shape"),
                "key_path": meta.get("key_path"),
            }
            if verbose:
                print(f"[META] {widget_id}: {snapshot[widget_id]}")
        return snapshot

    def get_meta_entry(self, widget_id: str) -> Optional[dict]:
        info = self._meta_registry.get(widget_id)
        if not info:
            return None
        return {
            "widget_id": widget_id,
            "last_value": info.get("last_value"),
            "pending_value": info.get("pending_value"),
            "meta": dict(info["meta"]),
        }

    def force_commit_meta(self, widget_id: str):
        info = self._meta_registry.get(widget_id)
        if not info:
            return False
        if info["timer"].isActive():
            info["timer"].stop()
        if info.get("pending_value") is None:
            return False
        self._commit_meta_widget(widget_id)
        return True


__all__ = ["ParameterTriggerDiagnosticsMixin"]
