"""One assistant run: ask the model, run the tools it calls, until it submits its report.

The loop is ours rather than an SDK runner: every tool call must reach the GUI
thread through the workbench, the user can stop between steps, and the whole
loop runs against a scripted model in tests.
"""

from __future__ import annotations

import time
from typing import Callable, Optional, Sequence

from .models import (
    PERMISSION_PREVIEW,
    RUN_CANCELLED,
    RUN_COMPLETED,
    RUN_FAILED,
    RUN_INCOMPLETE,
    AnalysisGoals,
    LlmError,
    LlmUsage,
    RunOutcome,
    RunResults,
    StepRecord,
    ToolCall,
    ToolOutcome,
)
from .ports import AnalysisWorkbench, AssistantLlm, Chooser, Confirmer, FileExplorer, CurveFitter, GeometryCalibrator, ResultStore, RunEvents
from .prompts import REMINDER, SYSTEM_PROMPT, task_message
from .tools import ToolCatalog, clean

DEFAULT_MAX_TURNS = 30
RETRY_DELAYS_S = (5.0, 20.0)


class Silent:
    def step_started(self, step) -> None:
        pass

    def step_finished(self, step) -> None:
        pass

    def model_text(self, text) -> None:
        pass

    def model_progress(self, kind, text) -> None:
        pass

    def notice(self, text) -> None:
        pass

    def usage(self, turn_usage, total_usage) -> None:
        pass


def run_step(catalog: ToolCatalog, call: ToolCall, steps: list[StepRecord], events) -> ToolOutcome:
    """Run one tool call, record it as the next step and report it to ``events``."""
    step = StepRecord(index=len(steps) + 1, tool=call.name, arguments=clean(call.input or {}))
    steps.append(step)
    events.step_started(step)
    started = time.perf_counter()
    outcome = catalog.execute(call)
    step.seconds = time.perf_counter() - started
    step.ok = not outcome.is_error
    step.summary = outcome.summary
    step.result = outcome.data if outcome.data is not None else outcome.content
    events.step_finished(step)
    return outcome


class RunAssistantTask:
    def __init__(
        self,
        llm: AssistantLlm,
        workbench: AnalysisWorkbench,
        *,
        confirmer: Optional[Confirmer] = None,
        store: Optional[ResultStore] = None,
        events: Optional[RunEvents] = None,
        explorer: Optional[FileExplorer] = None,
        calibrator: Optional[GeometryCalibrator] = None,
        chooser: Optional[Chooser] = None,
        fitter: Optional[CurveFitter] = None,
        max_turns: int = DEFAULT_MAX_TURNS,
        retry_delays: Sequence[float] = RETRY_DELAYS_S,
    ):
        self.llm = llm
        self.retry_delays = tuple(float(delay) for delay in retry_delays)
        self.workbench = workbench
        self.confirmer = confirmer
        self.store = store
        self.explorer = explorer
        self.calibrator = calibrator
        self.fitter = fitter
        self.chooser = chooser
        self.events = events or Silent()
        self.max_turns = max(1, int(max_turns))

    def __call__(self, goals: AnalysisGoals, *, cancelled: Callable[[], bool] = lambda: False) -> RunOutcome:
        results = RunResults()
        catalog = ToolCatalog(
            self.workbench, goals, results, confirmer=self.confirmer, store=self.store,
            explorer=self.explorer, calibrator=self.calibrator, chooser=self.chooser, fitter=self.fitter,
            cancelled=cancelled,
        )
        tools = catalog.definitions()
        steps: list[StepRecord] = []
        usage = LlmUsage()
        status = self._run_tool(catalog, ToolCall("status", "get_status", {}), steps)
        messages: list[dict] = [{"role": "user", "content": task_message(goals, status.data or {}, catalog.access_notes())}]
        state, message = RUN_INCOMPLETE, f"Stopped after {self.max_turns} model turns without a report."
        reminded = False
        for _turn in range(self.max_turns):
            if cancelled():
                state, message = RUN_CANCELLED, "Stopped by the user."
                break
            try:
                turn = self._respond(tools, messages, cancelled)
            except LlmError as exc:
                state, message = (RUN_CANCELLED, "Stopped by the user.") if cancelled() else (RUN_FAILED, exc.message)
                break
            usage = usage + turn.usage
            self.events.usage(turn.usage, usage)
            if turn.text.strip():
                self.events.model_text(turn.text.strip())
            if turn.stop_reason == "refusal":
                category = f" ({turn.refusal_category})" if turn.refusal_category else ""
                state, message = RUN_FAILED, f"Claude declined this request{category}."
                break
            if turn.stop_reason == "max_tokens" and turn.tool_calls:
                state, message = RUN_FAILED, "The model's reply was cut off at the output limit."
                break
            messages.append({"role": "assistant", "content": list(turn.content)})
            if not turn.tool_calls:
                if results.report is not None:
                    state, message = RUN_COMPLETED, "Report submitted."
                    break
                if reminded:
                    state, message = RUN_INCOMPLETE, "The model stopped without submitting its report."
                    break
                reminded = True
                messages.append({"role": "user", "content": REMINDER})
                continue
            tool_results, finished = [], False
            for call in turn.tool_calls:
                if cancelled():
                    break
                outcome = self._run_tool(catalog, call, steps)
                block = {"type": "tool_result", "tool_use_id": call.id, "content": outcome.content}
                if outcome.is_error:
                    block["is_error"] = True
                tool_results.append(block)
                finished = finished or outcome.final
            if cancelled():
                state, message = RUN_CANCELLED, "Stopped by the user."
                break
            messages.append({"role": "user", "content": tool_results})
            if finished:
                state, message = RUN_COMPLETED, "Report submitted."
                break
        if goals.permission == PERMISSION_PREVIEW:
            catalog.restore_run_changes()  # the person applies what they want from the cards
        return RunOutcome(state, message, steps, results, usage, model=getattr(self.llm, "model", ""), transcript=messages)

    def _respond(self, tools: list[dict], messages: list[dict], cancelled: Callable[[], bool]):
        """One model turn; overload, rate-limit and network errors are retried after a pause."""
        for delay in (*self.retry_delays, None):
            try:
                return self.llm.respond(
                    system=SYSTEM_PROMPT,
                    tools=tools,
                    messages=messages,
                    cancelled=cancelled,
                    progress=self.events.model_progress,
                )
            except LlmError as exc:
                if not exc.retryable or delay is None or cancelled():
                    raise
                self.events.model_progress("retry", f"{exc.message} Trying again in {delay:g} s.")
                deadline = time.monotonic() + delay
                while time.monotonic() < deadline:
                    if cancelled():
                        raise
                    time.sleep(min(0.2, max(0.0, deadline - time.monotonic())))
        raise AssertionError("unreachable")

    def _run_tool(self, catalog: ToolCatalog, call: ToolCall, steps: list[StepRecord]) -> ToolOutcome:
        return run_step(catalog, call, steps, self.events)


__all__ = ["DEFAULT_MAX_TURNS", "RETRY_DELAYS_S", "RunAssistantTask", "Silent", "run_step"]
