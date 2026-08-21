"""Qt-free in-situ workflow state commands."""

from __future__ import annotations


class FittingInSituViewModel:
    def __init__(
        self,
        workflow,
        create_recipe,
        revise_recipe,
        on_state_changed,
        on_recipe_changed,
    ):
        self._workflow = workflow
        self._create_recipe = create_recipe
        self._revise_recipe = revise_recipe
        self._on_state_changed = on_state_changed
        self._on_recipe_changed = on_recipe_changed
        self._recipe = None
        self._recipe_scope = "future"

    @property
    def state(self):
        return self._workflow.state

    @property
    def recipe(self):
        return self._recipe

    @property
    def recipe_scope(self) -> str:
        return self._recipe_scope

    def create_recipe_from_single(self, snapshot):
        self._recipe = self._create_recipe.execute(snapshot, self._recipe)
        self._recipe_scope = "future"
        self._on_recipe_changed(
            self._recipe,
            self._recipe_scope,
            f"In-situ Recipe v{self._recipe.version} created from Single analysis",
        )
        return self._recipe

    def revise_recipe(self, request):
        revision = self._revise_recipe.execute(request)
        self._recipe = revision.recipe
        self._recipe_scope = revision.scope
        self._on_recipe_changed(
            self._recipe,
            self._recipe_scope,
            f"In-situ Recipe v{self._recipe.version} applies to {revision.scope}",
        )
        return revision

    def snapshot_recipe(self):
        return None if self._recipe is None else self._recipe.to_dict()

    def restore_recipe(self, snapshot) -> None:
        from ..application import InSituProcessingRecipe

        self._recipe = InSituProcessingRecipe.from_dict(snapshot)
        self._recipe_scope = "future"
        self._on_recipe_changed(
            self._recipe,
            self._recipe_scope,
            f"In-situ Recipe v{self._recipe.version} restored",
        )

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
