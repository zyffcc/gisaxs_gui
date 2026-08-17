import json

from src.gimap.features.fitting.application import (
    InSituFileFitResult,
    InSituWorkflowCoordinator,
    InSituWorkflowRequest,
    RunInSituWorkflow,
)


class _ThreeFileFit:
    def __init__(self):
        self.paths = []

    def execute(self, request):
        path = request.paths[0]
        self.paths.append(path)
        if path.endswith("002.cbf"):
            raise ValueError("damaged frame")
        return InSituFileFitResult({"chi_square": len(self.paths) / 10})


def test_three_file_workflow_continues_after_one_file_error():
    fit = _ThreeFileFit()
    progress = []
    workflow = RunInSituWorkflow(fit)

    state = workflow.execute(
        InSituWorkflowRequest(
            ("sample_001.cbf", "sample_002.cbf", "sample_003.cbf")
        ),
        on_progress=progress.append,
    )

    assert fit.paths == ["sample_001.cbf", "sample_002.cbf", "sample_003.cbf"]
    assert state.status == "completed"
    assert [record.status for record in state.records] == [
        "succeeded",
        "failed",
        "succeeded",
    ]
    assert state.records[1].error_message == "damaged frame"
    assert state.processed_count == 3
    assert state.failed_count == 1
    assert [item.processed for item in progress] == [1, 2, 3]


def test_workflow_can_be_cancelled_between_files():
    workflow = None

    class CancellingFit:
        def execute(self, request):
            workflow.cancel()
            return InSituFileFitResult({"path": request.paths[0]})

    workflow = RunInSituWorkflow(CancellingFit())

    state = workflow.execute(InSituWorkflowRequest(("one.cbf", "two.cbf")))

    assert state.status == "cancelled"
    assert state.processed_count == 1
    assert state.pending_paths == ("two.cbf",)


def test_workflow_state_is_json_serializable_and_restorable():
    coordinator = InSituWorkflowCoordinator()
    coordinator.start(("one.cbf", "two.cbf"))
    coordinator.begin_next()
    coordinator.complete_current({"chi_square": 0.25, "parameters": [1.0, 2.0]})

    encoded = json.dumps(coordinator.snapshot(), ensure_ascii=False)
    restored = InSituWorkflowCoordinator()
    restored.restore(json.loads(encoded))

    assert restored.state == coordinator.state
    assert restored.state.pending_paths == ("two.cbf",)
    assert restored.state.records[0].values["parameters"] == [1.0, 2.0]


def test_non_serializable_worker_output_is_rejected():
    coordinator = InSituWorkflowCoordinator()
    coordinator.start(("one.cbf",))
    coordinator.begin_next()

    try:
        coordinator.complete_current({"bad": object()})
    except ValueError as exc:
        assert "JSON serializable" in str(exc)
    else:
        raise AssertionError("non-serializable worker output was accepted")
