"""Navigation and preference callbacks for the fitting workflow header."""

from __future__ import annotations


def save_guided_mode(preferences, guided: bool) -> None:
    preferences.set("fitting.guided_workflow", bool(guided))
    preferences.save()


def navigate_workflow_step(workspace, key: str) -> None:
    workspace.show_workflow_step(key)


__all__ = ["navigate_workflow_step", "save_guided_mode"]
