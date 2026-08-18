"""Qt-free in-situ workflow state commands."""

from __future__ import annotations


class FittingInSituViewModel:
    def __init__(self, workflow, on_state_changed):
        self._workflow = workflow
        self._on_state_changed = on_state_changed

    @property
    def state(self):
        return self._workflow.state

    def start_insitu_workflow(self, paths, *, continue_on_error=True) -> None:
        self._workflow.start(paths, continue_on_error=continue_on_error)
        self._notify("In-situ workflow started")

    def enqueue_insitu_files(self, paths) -> None:
        self._workflow.enqueue(paths)
        self._notify("In-situ files queued")

    def begin_next_insitu_file(self, batch_size=1):
        record = self._workflow.begin_next(batch_size)
        self._notify("Processing in-situ file")
        return record

    def complete_insitu_file(self, values=None):
        record = self._workflow.complete_current(values)
        self._notify("In-situ file completed")
        return record

    def fail_insitu_file(self, error_message, values=None):
        record = self._workflow.fail_current(error_message, values)
        self._notify("In-situ file failed")
        return record

    def pause_insitu_workflow(self) -> None:
        self._workflow.pause()
        self._notify("In-situ workflow paused")

    def resume_insitu_workflow(self) -> None:
        self._workflow.resume()
        self._notify("In-situ workflow resumed")

    def cancel_insitu_workflow(self) -> None:
        self._workflow.cancel()
        self._notify("In-situ workflow cancelled")

    def snapshot_insitu_workflow(self):
        return self._workflow.snapshot()

    def restore_insitu_workflow(self, snapshot) -> None:
        self._workflow.restore(snapshot)
        self._notify("In-situ workflow restored")

    def _notify(self, message: str) -> None:
        self._on_state_changed(self._workflow.state, message)
