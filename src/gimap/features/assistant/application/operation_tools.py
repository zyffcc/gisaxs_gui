"""Tool handlers and actions for previewable operations, mixed into ``ToolCatalog``."""

from __future__ import annotations

from typing import Optional

from .models import ToolInputError, ToolOutcome
from .operations import (
    APPLIED,
    DISMISSED,
    FAILED,
    FROM_PROPOSAL,
    FROM_RUN,
    OPERATION_TOOLS,
    PROPOSED,
    SUPERSEDED,
    UNDONE,
    Operation,
    describe,
    inverse_arguments,
    setting_state,
)

MAX_PROPOSALS = 8
THUMBNAIL_PX = 360


def _state(tool: str, status: dict) -> object:
    """The effective state, rounded like the tool results so before and after compare."""
    from .tools import clean  # the catalog module imports this mixin

    return setting_state(tool, clean(status))


class OperationToolsMixin:
    """Needs ``workbench``, ``goals``, ``results``, ``_specs`` and ``_ok`` from ``ToolCatalog``."""

    def _inverse(self, tool: str) -> Optional[dict]:
        return self._before(tool)[0]

    def _before(self, tool: str) -> tuple[Optional[dict], object]:
        """(arguments that restore the setting, its effective state) now."""
        try:
            status = self.workbench.status()
        except Exception:  # a status that cannot be read means the change cannot be undone
            return None, None
        return inverse_arguments(tool, status), _state(tool, status)

    def _untracked(self, tool: str, arguments: dict) -> ToolOutcome:
        """Run a settings tool without recording it (previews, apply, undo)."""
        try:
            return getattr(self, f"_tool_{tool}")(**arguments)
        except ToolInputError as exc:
            return ToolOutcome(str(exc), str(exc), is_error=True)
        except Exception as exc:  # reported, never raised into the GUI
            message = str(exc) or type(exc).__name__
            return ToolOutcome(message, f"failed: {message}", is_error=True)

    def _thumbnail(self) -> Optional[bytes]:
        try:
            return self.workbench.preview_png(THUMBNAIL_PX)
        except Exception:
            return None

    def _tracked(self, tool: str, arguments: dict) -> ToolOutcome:
        """A settings change the model makes: recorded with the arguments that undo it."""
        before, state = self._before(tool)
        outcome = getattr(self, f"_tool_{tool}")(**arguments)
        if not outcome.is_error:
            operation = Operation(
                tool, dict(arguments), describe(tool, arguments, self.goals.language), state=APPLIED,
                source=FROM_RUN, inverse=before, effect=outcome.summary,
                before_state=state, after_state=_state(tool, outcome.data or {}),
            )
            if state is not None and operation.after_state == state:
                operation.no_effect = True  # changed nothing that shows: undone like the rest, never a card
            elif self._previewing():
                operation.preview_png = self._thumbnail()
            self.results.operations.append(operation)
        return outcome

    def _previewing(self) -> bool:
        from .models import PERMISSION_PREVIEW

        return self.goals.permission == PERMISSION_PREVIEW

    # -- the model's suggestion tool ------------------------------------------------------

    def _tool_propose_operations(self, operations: list) -> ToolOutcome:
        if not operations:
            raise ToolInputError("Propose at least one operation.")
        made, restore = [], []
        try:
            for item in operations[:MAX_PROPOSALS]:
                tool = item["tool"]
                if tool not in OPERATION_TOOLS:
                    made.append({"tool": tool, "error": f"not a previewable change (use one of {', '.join(OPERATION_TOOLS)})"})
                    continue
                try:
                    from .tools import validate

                    arguments = validate(self._specs[tool].schema, dict(item.get("arguments") or {}))
                except ToolInputError as exc:
                    made.append({"tool": tool, "error": str(exc)})
                    continue
                operation = Operation(
                    tool, arguments, str(item.get("title") or describe(tool, arguments, self.goals.language)),
                    why=str(item.get("why") or ""), state=PROPOSED, source=FROM_PROPOSAL,
                )
                before = self._inverse(tool)
                if before is None:
                    operation.effect = "no preview: the current state of this setting cannot be restored here"
                else:
                    outcome = self._untracked(tool, arguments)
                    if outcome.is_error:
                        operation.state, operation.effect = FAILED, outcome.summary
                    else:
                        restore.append((tool, before))
                        operation.effect = outcome.summary
                        operation.preview_png = self._thumbnail()
                self.results.operations.append(operation)
                made.append(operation.record())
        finally:
            # Previews are cumulative in the order given; afterwards everything is as before.
            for tool, before in reversed(restore):
                self._untracked(tool, before)
        payload = {
            "proposed": made,
            "note": "Previewed and restored: nothing is changed until the user applies a card in the panel.",
        }
        good = sum(1 for item in made if item.get("state") == PROPOSED)
        return self._ok(payload, f"{good} change(s) proposed to the user")

    # -- actions of the person (the panel) ------------------------------------------------

    def operation(self, identifier: str) -> Operation:
        found = next((item for item in self.results.operations if item.id == identifier), None)
        if found is None:
            raise KeyError(identifier)
        return found

    def apply_operation(self, identifier: str) -> ToolOutcome:
        operation = self.operation(identifier)
        before = self._inverse(operation.tool)
        outcome = self._untracked(operation.tool, operation.arguments)
        if not outcome.is_error:
            operation.state, operation.inverse, operation.effect = APPLIED, before, outcome.summary
        return outcome

    def undo_operation(self, identifier: str) -> ToolOutcome:
        operation = self.operation(identifier)
        if operation.inverse is None:
            return ToolOutcome("This change cannot be undone here.", "cannot be undone", is_error=True)
        outcome = self._untracked(operation.tool, operation.inverse)
        if not outcome.is_error:
            operation.state = UNDONE if operation.source == FROM_RUN else PROPOSED
        return outcome

    def dismiss_operation(self, identifier: str) -> None:
        operation = self.operation(identifier)
        if operation.state == PROPOSED:
            operation.state = DISMISSED

    def undo_all_operations(self) -> int:
        """Undo every applied change, newest first; returns how many were undone."""
        count = 0
        for operation in reversed(list(self.results.operations)):
            if operation.state == APPLIED and not self.undo_operation(operation.id).is_error:
                count += 1
        return count

    def restore_run_changes(self) -> int:
        """Preview first: restore what the model changed during the run and offer the net changes.

        Every change is undone (newest first).  What remains as a card is the last change of each
        setting — the state the model ended in, not each exploration step — unless it changed
        nothing, or the model proposed a value for that setting itself (with a title and a reason):
        its proposal is what it recommends, the state it ended in may just be the last thing it tried.
        """
        run = [operation for operation in self.results.operations if operation.source == FROM_RUN]
        first_state = {}
        for operation in run:  # the state each setting had before the model first touched it
            first_state.setdefault(operation.tool, operation.before_state)
        restored = []
        for operation in reversed(run):
            if operation.state == APPLIED and operation.inverse is not None:
                if not self._untracked(operation.tool, operation.inverse).is_error:
                    operation.state = PROPOSED
                    restored.append(operation)
        suggested = {  # a proposal that could not be previewed does not replace anything
            operation.tool for operation in self.results.operations
            if operation.source == FROM_PROPOSAL and operation.state != FAILED
        }
        seen: set[str] = set()
        for operation in restored:  # newest first: the first seen per setting is its final state
            unchanged = operation.after_state is not None and operation.after_state == first_state.get(operation.tool)
            if operation.tool in seen or operation.tool in suggested or unchanged or operation.no_effect:
                operation.state = SUPERSEDED
            seen.add(operation.tool)
        return len(restored)


__all__ = ["MAX_PROPOSALS", "OperationToolsMixin"]
