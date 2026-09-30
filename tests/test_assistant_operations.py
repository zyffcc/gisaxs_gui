"""The assistant's changes as operations: recorded with their undo, proposed with a preview, applied by the person."""

from __future__ import annotations

import json

import pytest

from src.gimap.features.assistant.application import (
    APPLIED,
    DISMISSED,
    FROM_PROPOSAL,
    FROM_RUN,
    GOALS,
    PERMISSION_AUTO,
    PERMISSION_PREVIEW,
    PROPOSED,
    RUN_COMPLETED,
    SUPERSEDED,
    UNDONE,
    AnalysisGoals,
    RunAssistantTask,
    RunResults,
    ToolCatalog,
    describe,
    inverse_arguments,
    results_payload,
)
from tests.assistant_fakes import FakeWorkbench, ScriptedLlm, call, report, turn


class SettingsWorkbench(FakeWorkbench):
    """A workbench whose status follows every settings change, and whose picture shows the settings."""

    def __init__(self):
        super().__init__()
        self.widths = (10.0, 10.0)
        self.sector = None
        self.box = None
        self.frame, self.summed, self.frames = 400, 1, 403
        self.mode, self.override, self.bins = "giwaxs", None, "auto"
        self.limits = [None, None]

    def status(self) -> dict:
        status = super().status()
        status.update(mode=self.mode, incidence_override=self.override, frame=self.frame, summed_frames=self.summed,
                      frames=self.frames, corrections={"valid_range": list(self.limits)})
        status["giwaxs"].update(
            in_plane_half_width_deg=self.widths[0], out_of_plane_half_width_deg=self.widths[1], radial_bins=self.bins,
            custom_sector=None if self.sector is None else {"chi_deg": list(self.sector[0]), "q": list(self.sector[1])},
            q_box=None if self.box is None else {"q_parallel": list(self.box[0]), "qz": list(self.box[1])},
        )
        return status

    def settings(self) -> tuple:
        return (self.widths, self.sector, self.box, self.frame, self.summed, self.mode, self.override, self.bins, tuple(self.limits))

    def set_sector_widths(self, in_plane_deg, out_of_plane_deg):
        self.widths = (in_plane_deg, out_of_plane_deg)
        return super().set_sector_widths(in_plane_deg, out_of_plane_deg)

    def set_custom_sector(self, chi, q_range):
        self.sector = None if chi is None else (tuple(chi), tuple(q_range))
        return super().set_custom_sector(chi, q_range)

    def set_q_box(self, q_parallel, qz):
        self.box = None if q_parallel is None else (tuple(q_parallel), tuple(qz))
        return super().set_q_box(q_parallel, qz)

    def set_frame(self, frame_number, sum_count):
        self.frame, self.summed = frame_number, sum_count
        return self._record("set_frame", frame_number, sum_count)

    def set_incidence(self, degrees):
        self.override = degrees
        return super().set_incidence(degrees)

    def set_valid_range(self, minimum, maximum):
        self.limits = [minimum, maximum]
        return super().set_valid_range(minimum, maximum)

    def preview_png(self, max_size=900):
        return json.dumps({"sector": self.sector, "frame": self.frame}).encode()


def catalog(workbench, permission=PERMISSION_AUTO, confirmer=None) -> ToolCatalog:
    return ToolCatalog(workbench, AnalysisGoals(goals=tuple(GOALS), permission=permission, language="中文"), RunResults(), confirmer=confirmer)


def test_every_settings_change_is_recorded_with_its_undo() -> None:
    workbench = SettingsWorkbench()
    tools = catalog(workbench)
    before = workbench.settings()
    assert not tools.execute(call("set_sector_widths", in_plane_half_width_deg=5, out_of_plane_half_width_deg=20)).is_error
    assert not tools.execute(call("set_custom_sector", enabled=True, chi_min_deg=15, chi_max_deg=55)).is_error
    first, second = tools.results.operations
    assert (first.state, first.source, first.inverse) == (APPLIED, FROM_RUN, {"in_plane_half_width_deg": 10.0, "out_of_plane_half_width_deg": 10.0})
    assert first.title == "扇区半宽：面内 ±5°，面外 ±20°" and second.title == "添加自定义扇区 χ 15…55°"
    assert tools.undo_all_operations() == 2 and workbench.settings() == before
    assert first.state == second.state == UNDONE
    assert not tools.execute(call("find_peaks", curve="radial")).is_error and len(tools.results.operations) == 2  # reads are not operations


def test_proposals_are_previewed_then_restored_and_applied_by_the_person() -> None:
    workbench = SettingsWorkbench()
    tools = catalog(workbench)
    before = workbench.settings()
    outcome = tools.execute(call("propose_operations", operations=[
        {"tool": "set_custom_sector", "arguments": {"enabled": True, "chi_min_deg": 15, "chi_max_deg": 55},
         "title": "只用亮区", "why": "面内扇区在阴影里"},
        {"tool": "set_frame", "arguments": {"frame": 1, "sum": 10}},
        {"tool": "set_frame", "arguments": {"sum": 10}},
    ]))
    payload = json.loads(outcome.content)
    assert not outcome.is_error and outcome.summary == "2 change(s) proposed to the user"
    assert workbench.settings() == before  # nothing stays changed
    sector, frames = tools.results.operations
    assert (sector.state, sector.source, sector.title, sector.why) == (PROPOSED, FROM_PROPOSAL, "只用亮区", "面内扇区在阴影里")
    # Previews are cumulative: the frame preview still shows the sector proposed before it.
    assert json.loads(frames.preview_png) == {"sector": [[15.0, 55.0], [None, None]], "frame": 1}
    assert frames.title == "第 1 帧起求和 10 帧"
    assert "Missing argument(s): frame" in payload["proposed"][2]["error"]

    assert not tools.apply_operation(sector.id).is_error
    assert sector.state == APPLIED and workbench.sector == ((15.0, 55.0), (None, None))
    assert not tools.undo_operation(sector.id).is_error and sector.state == PROPOSED and workbench.sector is None
    tools.dismiss_operation(frames.id)
    assert frames.state == DISMISSED
    assert tools.execute(call("propose_operations", operations=[{"tool": "use_geometry", "arguments": {}}])).is_error


def test_a_change_that_cannot_be_restored_is_not_previewed() -> None:
    workbench = SettingsWorkbench()
    tools = catalog(workbench)
    workbench.status = lambda: {"mode": "odd"}  # nothing known about the state
    tools.execute(call("propose_operations", operations=[{"tool": "set_measurement_mode", "arguments": {"mode": "gisaxs"}}]))
    (operation,) = tools.results.operations
    assert operation.state == PROPOSED and "no preview" in operation.effect and operation.preview_png is None
    assert workbench.measurement == "giwaxs"  # not changed


def test_preview_first_restores_what_the_model_changed_and_offers_it() -> None:
    workbench = SettingsWorkbench()
    asked = []

    class Confirmer:
        def confirm(self, title, text):
            asked.append(text)
            return True

    before = workbench.settings()
    llm = ScriptedLlm([
        turn(call("set_sector_widths", in_plane_half_width_deg=4, out_of_plane_half_width_deg=4)),
        turn(call("set_valid_intensity_range", minimum=None, maximum=5000.0)),
        turn(report(("peaks", "done"))),
    ])
    outcome = RunAssistantTask(llm, workbench, confirmer=Confirmer())(
        AnalysisGoals(goals=tuple(GOALS), permission=PERMISSION_PREVIEW)
    )
    assert outcome.state == RUN_COMPLETED and asked == []  # settings changes do not ask in this mode
    assert workbench.settings() == before
    widths, limits = outcome.results.operations
    assert widths.state == limits.state == PROPOSED and widths.source == FROM_RUN
    assert widths.preview_png is not None  # a picture of what the change looked like
    assert "preview first" in llm.requests[0]["messages"][0]["content"]
    assert [item["state"] for item in results_payload(outcome.results)["operations"]] == [PROPOSED, PROPOSED]


def test_a_preview_first_run_offers_one_net_change_per_setting() -> None:
    # The real P03 run: exploration steps and a suggestion repeating the final state gave 8 noisy cards.
    workbench = SettingsWorkbench()
    before = workbench.settings()
    llm = ScriptedLlm([
        turn(call("set_sector_widths", in_plane_half_width_deg=4, out_of_plane_half_width_deg=4)),
        turn(call("set_sector_widths", in_plane_half_width_deg=6, out_of_plane_half_width_deg=6)),
        turn(call("set_custom_sector", enabled=True, chi_min_deg=15, chi_max_deg=30)),
        turn(call("set_custom_sector", enabled=True, chi_min_deg=35, chi_max_deg=50)),
        turn(call("propose_operations", operations=[{
            "tool": "set_custom_sector", "arguments": {"enabled": True, "chi_min_deg": 35, "chi_max_deg": 50},
            "title": "只看亮区 χ 35–50°", "why": "(200)/(111) 面积比在这里最大",
        }])),
        turn(call("set_frame", frame=-1, sum=10)),
        turn(report(("peaks", "done"))),
    ])
    outcome = RunAssistantTask(llm, workbench)(AnalysisGoals(goals=tuple(GOALS), permission=PERMISSION_PREVIEW, language="中文"))
    assert workbench.settings() == before
    shown = [(operation.title, operation.source) for operation in outcome.results.operations if operation.state == PROPOSED]
    assert shown == [
        ("扇区半宽：面内 ±6°，面外 ±6°", FROM_RUN),
        ("只看亮区 χ 35–50°", FROM_PROPOSAL),
        ("最后 10 帧求和", FROM_RUN),
    ]
    assert sum(1 for operation in outcome.results.operations if operation.state == SUPERSEDED) == 3


@pytest.mark.parametrize(("tool", "arguments", "english"), [
    ("set_q_box", {"enabled": True, "q_parallel_min": 0.1, "q_parallel_max": 0.5, "qz_min": 1.0, "qz_max": 1.5},
     "q box q∥ 0.1…0.5, qz 1…1.5 Å⁻¹"),
    ("set_incidence_angle", {"degrees": 0.4}, "Incidence angle αi = 0.4°"),
    ("set_radial_bins", {"bins": 0}, "Radial bins: auto"),
    ("set_measurement_mode", {"mode": "giwaxs"}, "Measurement mode: GIWAXS"),
    ("set_valid_intensity_range", {"minimum": None, "maximum": 5000}, "Valid intensity range auto…5000"),
])
def test_titles_and_inverses(tool, arguments, english) -> None:
    assert describe(tool, arguments) == english
    status = SettingsWorkbench().status()
    assert inverse_arguments(tool, status) is not None
    assert inverse_arguments("set_sector_widths", {"mode": "gisaxs"}) is None  # no sectors outside GIWAXS


def test_changes_without_a_visible_effect_or_without_a_net_change_are_not_cards() -> None:
    # The second real run: αi and the last ten frames were already set by the guided analysis.
    class Effective(SettingsWorkbench):
        def status(self):
            status = super().status()
            status["geometry"] = {**status["geometry"], "incidence_deg": 0.4 if self.override is None else self.override}
            return status

    workbench = Effective()
    workbench.frame, workbench.summed = 394, 10
    llm = ScriptedLlm([
        turn(call("set_incidence_angle", degrees=0.4)),  # the profile already gives 0.4°
        turn(call("set_frame", frame=1, sum=10)),         # a look at the start of the series …
        turn(call("set_frame", frame=-1, sum=10)),        # … and back to where it was
        turn(call("set_sector_widths", in_plane_half_width_deg=5, out_of_plane_half_width_deg=5)),
        turn(report(("peaks", "done"))),
    ])
    outcome = RunAssistantTask(llm, workbench)(AnalysisGoals(goals=tuple(GOALS), permission=PERMISSION_PREVIEW))
    cards = [operation.tool for operation in outcome.results.operations if operation.state == PROPOSED]
    assert cards == ["set_sector_widths"]
    assert (workbench.frame, workbench.summed, workbench.override) == (394, 10, None)


def test_a_proposal_for_a_setting_replaces_where_the_run_ended() -> None:
    # The third real run: it tried χ 14–30°, 36–55°, 55–70° and proposed 14–55°; 55–70° was only its last try.
    workbench = SettingsWorkbench()
    llm = ScriptedLlm([
        turn(call("set_custom_sector", enabled=True, chi_min_deg=14, chi_max_deg=30)),
        turn(call("set_custom_sector", enabled=True, chi_min_deg=55, chi_max_deg=70)),
        turn(call("propose_operations", operations=[{
            "tool": "set_custom_sector", "arguments": {"enabled": True, "chi_min_deg": 14, "chi_max_deg": 55},
            "title": "Measured region χ 14–55°", "why": "Everything else is shadowed or in the missing wedge.",
        }])),
        turn(report(("peaks", "done"))),
    ])
    outcome = RunAssistantTask(llm, workbench)(AnalysisGoals(goals=tuple(GOALS), permission=PERMISSION_PREVIEW))
    cards = [operation.title for operation in outcome.results.operations if operation.state == PROPOSED]
    assert cards == ["Measured region χ 14–55°"]
