"""One run with a brain that owns its tool loop (Claude Code on the user's subscription).

GIMaP keeps everything that matters for the science: the same tool catalog,
the same validation and write confirmations, the same step record and the
same computed results as ``RunAssistantTask``.  Claude Code only decides which
tool to call next; tool calls are answered one at a time.
"""

from __future__ import annotations

import threading
from typing import Callable, Optional

from .models import (
    PERMISSION_PREVIEW,
    RUN_CANCELLED,
    RUN_COMPLETED,
    RUN_FAILED,
    RUN_INCOMPLETE,
    AnalysisGoals,
    RunOutcome,
    RunResults,
    StepRecord,
    ToolCall,
    ToolOutcome,
)
from .ports import AgentRuntime, AnalysisWorkbench, Chooser, Confirmer, FileExplorer, CurveFitter, GeometryCalibrator, ResultStore, RunEvents
from .prompts import AGENT_TOOLS_NOTE, REMINDER, SYSTEM_PROMPT, task_message
from .run_task import DEFAULT_MAX_TURNS, Silent, run_step
from .tools import ToolCatalog

STOPPED = ToolOutcome("The user stopped the run; do not call more tools.", "stopped", is_error=True)


class RunAgentTask:
    def __init__(
        self,
        agent: AgentRuntime,
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
    ):
        self.agent = agent
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
        steps: list[StepRecord] = []
        status = run_step(catalog, ToolCall("status", "get_status", {}), steps, self.events)
        one_at_a_time = threading.Lock()
        reminded = False

        def call_tool(name: str, arguments: dict) -> ToolOutcome:
            with one_at_a_time:
                if cancelled():
                    return STOPPED
                call = ToolCall(f"agent_{len(steps) + 1}", name, arguments if isinstance(arguments, dict) else {})
                return run_step(catalog, call, steps, self.events)

        def follow_up() -> Optional[str]:
            nonlocal reminded
            if results.report is not None or reminded or cancelled():
                return None
            reminded = True
            return REMINDER

        result = self.agent.run(
            system=SYSTEM_PROMPT + AGENT_TOOLS_NOTE,
            tools=catalog.definitions(),
            prompt=task_message(goals, status.data or {}, catalog.access_notes()),
            call_tool=call_tool,
            follow_up=follow_up,
            cancelled=cancelled,
            events=self.events,
            max_turns=self.max_turns,
        )
        if goals.permission == PERMISSION_PREVIEW:
            catalog.restore_run_changes()  # the person applies what they want from the cards
        if cancelled():
            state, message = RUN_CANCELLED, "Stopped by the user."
        elif results.report is not None:
            state, message = RUN_COMPLETED, "Report submitted."
        elif not result.ok:
            state, message = RUN_FAILED, result.message
        else:
            state, message = RUN_INCOMPLETE, result.message or "Claude stopped without submitting its report."
        return RunOutcome(
            state,
            message,
            steps,
            results,
            result.usage,
            model=result.model or getattr(self.agent, "model", ""),
            transcript=result.transcript,
            cost_usd=result.cost_usd,
            billing=result.billing,
        )


__all__ = ["RunAgentTask"]
