"""The assistant's model loop: tools, permissions, reminders, limits, retries and stopping."""

from __future__ import annotations

import json

import pytest

from src.gimap.features.assistant.application import (
    GIWAXS_GOALS,
    GOALS,
    PERMISSION_AUTO,
    PERMISSION_CONFIRM,
    RUN_CANCELLED,
    RUN_COMPLETED,
    RUN_FAILED,
    RUN_INCOMPLETE,
    SYSTEM_PROMPT,
    AnalysisGoals,
    LlmError,
    RunAssistantTask,
    ToolCatalog,
    RunResults,
    tool_specs,
)
from src.gimap.features.assistant.application.tools import ToolInputError, validate
from tests.assistant_fakes import FakeWorkbench, ScriptedLlm, call, report, tool_results, turn

ALL = AnalysisGoals(goals=GIWAXS_GOALS, permission=PERMISSION_AUTO)


class Confirmer:
    def __init__(self, answer: bool):
        self.answer = answer
        self.questions: list[str] = []

    def confirm(self, title: str, text: str) -> bool:
        self.questions.append(text)
        return self.answer


class Store:
    def __init__(self):
        self.tables: list[tuple] = []
        self.requests: list[dict] = []

    def write_tables(self, folder, stem, payload):
        self.tables.append((folder, stem, payload))
        return f"{folder}/{stem}_assistant.json"

    def append_feature_request(self, entry):
        self.requests.append(entry)


class Events:
    def __init__(self):
        self.log: list[tuple] = []

    def step_started(self, step):
        self.log.append(("started", step.tool))

    def step_finished(self, step):
        self.log.append(("finished", step.tool, step.ok))

    def model_text(self, text):
        self.log.append(("text", text))

    def model_progress(self, kind, text):
        self.log.append(("progress", kind))

    def usage(self, turn_usage, total_usage):
        self.log.append(("usage", total_usage.input_tokens))


def _task(llm, workbench=None, **options):
    return RunAssistantTask(llm, workbench or FakeWorkbench(), retry_delays=(0.0, 0.0), **options)


def test_a_full_run_computes_every_result_and_the_report_quotes_them() -> None:
    llm = ScriptedLlm([
        turn(call("find_peaks", curve="radial"), text="Finding the peaks first."),
        turn(call("compare_sectors")),
        turn(call("ring_orientation", q_center=0.40)),
        turn(call("crystallite_size", q_center=1.20)),
        turn(report(("peaks", "done"), ("orientation", "done"), ("ring_orientation", "done"), ("crystallite_size", "done"))),
    ])
    workbench = FakeWorkbench()
    events = Events()
    outcome = _task(llm, workbench, events=events)(ALL)

    assert outcome.state == RUN_COMPLETED, outcome.message
    results = outcome.results
    peaks = [peak.q for peak in results.peak_searches["radial"].peaks]
    for expected in (0.40, 0.80, 1.20, 1.65):
        assert min(abs(q - expected) for q in peaks) < 0.02, peaks
    def preference(q):
        return min(results.sector_rows, key=lambda row: abs(row.q - q)).preference

    assert preference(0.40) == "mainly out-of-plane"
    assert preference(1.65) == "mainly in-plane"
    assert preference(1.20) == "both sectors (no strong preference)"
    ring = results.rings[0]
    assert ring.herman is not None and ring.herman > 0.4
    assert "surface normal" in ring.texture
    assert ("set_chi_window", pytest.approx(ring.q_window[0]), pytest.approx(ring.q_window[1])) in workbench.calls
    assert ("show", None, "azimuthal") in workbench.calls
    size = results.sizes[0]
    assert size.lower_bound and size.size == pytest.approx(2 * 3.14159 * 0.9 / 0.04, rel=0.2)
    assert [item.item for item in results.report.items] == list(GIWAXS_GOALS)
    # get_status ran first, then one step per tool call.
    assert [step.tool for step in outcome.steps] == [
        "get_status", "find_peaks", "compare_sectors", "ring_orientation", "crystallite_size", "submit_report",
    ]
    assert all(step.ok for step in outcome.steps)
    assert ("text", "Finding the peaks first.") in events.log
    assert ("progress", "thinking") in events.log
    assert outcome.usage.input_tokens == 500


def test_every_request_repeats_the_same_system_prompt_and_tools_and_answers_each_call() -> None:
    first, second = call("get_status"), call("find_peaks", curve="radial")
    llm = ScriptedLlm([turn(first, second), turn(report(("peaks", "done")))])
    _task(llm)(AnalysisGoals(goals=("peaks",)))

    assert all(request["system"] == SYSTEM_PROMPT for request in llm.requests)
    assert llm.requests[0]["tools"] == llm.requests[1]["tools"]
    task_message = llm.requests[0]["messages"][0]["content"]
    assert "peaks" in task_message and '"measurement": "giwaxs"' in task_message
    history = llm.requests[1]["messages"]
    assert history[1]["role"] == "assistant"
    assert [block["tool_use_id"] for block in tool_results(history)] == [first.id, second.id]


def test_write_actions_ask_first_and_a_refusal_is_reported_not_executed() -> None:
    confirmer = Confirmer(False)
    workbench = FakeWorkbench()
    llm = ScriptedLlm([
        turn(call("export_results", include_tables=True)),
        turn(report(("peaks", "partial"))),
    ])
    goals = AnalysisGoals(goals=("peaks",), permission=PERMISSION_CONFIRM)
    outcome = _task(llm, workbench, confirmer=confirmer, store=Store())(goals)

    assert outcome.state == RUN_COMPLETED
    assert confirmer.questions and "gimap_analysis" in confirmer.questions[0]
    assert ("export_curves",) not in workbench.calls
    result = json.loads(tool_results(llm.requests[1]["messages"])[0]["content"])
    assert result["declined"] is True
    assert outcome.steps[1].summary == "declined by the user"


def test_the_automatic_mode_writes_without_asking() -> None:
    confirmer, store = Confirmer(False), Store()
    workbench = FakeWorkbench()
    llm = ScriptedLlm([
        turn(call("find_peaks", curve="radial")),
        turn(call("export_results", include_tables=True)),
        turn(report(("peaks", "done"))),
    ])
    goals = AnalysisGoals(goals=("peaks",), permission=PERMISSION_AUTO)
    outcome = _task(llm, workbench, confirmer=confirmer, store=store)(goals)

    assert not confirmer.questions
    assert ("export_curves",) in workbench.calls
    folder, stem, payload = store.tables[0]
    assert stem == "synthetic" and folder.endswith("gimap_analysis")
    assert payload["peaks"]["radial"]["peaks"]
    assert outcome.results.exports[-1].endswith("synthetic_assistant.json")


def test_invalid_and_unknown_calls_come_back_as_errors_and_the_run_continues() -> None:
    llm = ScriptedLlm([
        turn(call("find_peaks", curve="nonsense"), call("find_peaks"), call("make_coffee")),
        turn(report(("peaks", "not_available"))),
    ])
    outcome = _task(llm)(AnalysisGoals(goals=("peaks",)))

    assert outcome.state == RUN_COMPLETED
    results = tool_results(llm.requests[1]["messages"])
    assert [block.get("is_error") for block in results] == [True, True, True]
    assert "must be one of" in results[0]["content"]
    assert "Missing argument(s): curve" in results[1]["content"]
    assert "Unknown tool" in results[2]["content"]


def test_validation_reaches_into_the_report_items() -> None:
    schema = next(spec.schema for spec in tool_specs(allow_images=False) if spec.name == "submit_report")
    good = {"summary": "s", "items": [], "caveats": [], "suggestions": []}
    assert validate(schema, good) == good
    bad_status = dict(good, items=[{"item": "peaks", "status": "maybe", "findings": "", "evidence": "", "reason": ""}])
    with pytest.raises(ToolInputError, match=r"items\[0\]\.status"):
        validate(schema, bad_status)
    truncated = dict(good, items=[{"item": "peaks", "status": "done"}])
    with pytest.raises(ToolInputError, match=r"items\[0\]\.findings"):
        validate(schema, truncated)


def test_a_model_that_stops_early_is_reminded_once() -> None:
    llm = ScriptedLlm([turn(text="The peaks are at 0.4 and 1.2."), turn(report(("peaks", "done")))])
    outcome = _task(llm)(AnalysisGoals(goals=("peaks",)))
    assert outcome.state == RUN_COMPLETED
    assert "submit_report" in llm.requests[1]["messages"][-1]["content"]

    llm = ScriptedLlm([turn(text="Done."), turn(text="Still done.")])
    outcome = _task(llm)(AnalysisGoals(goals=("peaks",)))
    assert outcome.state == RUN_INCOMPLETE
    assert outcome.results.report is None


def test_the_turn_limit_ends_a_run_without_a_report() -> None:
    llm = ScriptedLlm([turn(call("get_status")) for _ in range(3)])
    outcome = _task(llm, max_turns=3)(AnalysisGoals(goals=("peaks",)))
    assert outcome.state == RUN_INCOMPLETE and "3 model turns" in outcome.message


def test_refusals_and_cut_off_tool_calls_fail_the_run() -> None:
    refused = turn(stop="refusal")
    refused = type(refused)(**{**refused.__dict__, "refusal_category": "cyber"})
    outcome = _task(ScriptedLlm([refused]))(AnalysisGoals(goals=("peaks",)))
    assert outcome.state == RUN_FAILED and "cyber" in outcome.message

    cut = turn(call("find_peaks", curve="radial"), stop="max_tokens")
    workbench = FakeWorkbench()
    outcome = _task(ScriptedLlm([cut]), workbench)(AnalysisGoals(goals=("peaks",)))
    assert outcome.state == RUN_FAILED and "cut off" in outcome.message
    assert [step.tool for step in outcome.steps] == ["get_status"]


def test_transient_api_errors_are_retried_and_permanent_ones_fail() -> None:
    llm = ScriptedLlm([
        LlmError("Overloaded.", retryable=True),
        turn(report(("peaks", "not_available"))),
    ])
    events = Events()
    outcome = _task(llm, events=events)(AnalysisGoals(goals=("peaks",)))
    assert outcome.state == RUN_COMPLETED
    assert ("progress", "retry") in events.log

    llm = ScriptedLlm([LlmError("Bad key.")])
    outcome = _task(llm)(AnalysisGoals(goals=("peaks",)))
    assert outcome.state == RUN_FAILED and outcome.message == "Bad key."

    llm = ScriptedLlm([LlmError("Overloaded.", retryable=True)] * 3)
    outcome = _task(llm)(AnalysisGoals(goals=("peaks",)))
    assert outcome.state == RUN_FAILED and len(llm.requests) == 3


def test_stopping_ends_the_run_between_steps() -> None:
    stop = {"now": False}

    def second_turn(_messages):
        stop["now"] = True
        return turn(call("find_peaks", curve="radial"), call("compare_sectors"))

    llm = ScriptedLlm([turn(call("get_status")), second_turn])
    outcome = _task(llm)(AnalysisGoals(goals=("peaks",)), cancelled=lambda: stop["now"])
    assert outcome.state == RUN_CANCELLED
    assert [step.tool for step in outcome.steps] == ["get_status", "get_status"]

    # A turn abandoned by the model client while stopping counts as stopped, not failed.
    stopped = {"now": False}

    def abandon(_messages):
        stopped["now"] = True
        raise LlmError("Stopped by the user.")

    class Abandoning(ScriptedLlm):
        def respond(self, **kwargs):
            abandon(None)

    outcome = _task(Abandoning([]))(AnalysisGoals(goals=("peaks",)), cancelled=lambda: stopped["now"])
    assert outcome.state == RUN_CANCELLED


def test_missing_capabilities_are_recorded_for_later() -> None:
    store = Store()
    llm = ScriptedLlm([
        turn(call("note_missing_capability", capability="pole figure", reason="needs several incidence angles")),
        turn(report(("ring_orientation", "not_available"))),
    ])
    outcome = _task(llm, store=store)(AnalysisGoals(goals=("ring_orientation",)))
    assert store.requests[0]["capability"] == "pole figure"
    assert store.requests[0]["file"] == "C:/data/synthetic.tif"
    assert outcome.results.feature_requests == store.requests


def test_tools_explain_what_is_missing_instead_of_guessing() -> None:
    results = RunResults()
    catalog = ToolCatalog(FakeWorkbench(), AnalysisGoals(goals=("crystallite_size",)), results)
    size = catalog.execute(call("crystallite_size", q_center=1.2))
    assert size.is_error and "find_peaks" in size.content
    catalog.execute(call("find_peaks", curve="radial"))
    far = catalog.execute(call("crystallite_size", q_center=2.1))
    assert far.is_error and "No fitted peak near q = 2.1" in far.content
    preview = catalog.execute(call("view_preview"))
    assert preview.is_error and "Unknown tool" in preview.content  # images were not allowed
    images = ToolCatalog(FakeWorkbench(), AnalysisGoals(goals=("peaks",), allow_images=True), RunResults())
    shown = images.execute(call("view_preview"))
    assert shown.content[0]["type"] == "image" and shown.content[0]["source"]["media_type"] == "image/png"
