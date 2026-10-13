"""Release 0.1.0, the automatic analysis and the AI settings in the window.

No frame open: “Open a frame first.”, never “Results are for ;”. A calibration the search found is named
with its path under the geometry. The status line of a run says its steps in the interface language. “Claude
Code was not found …” and the provider names are translated at display (English in the code and the record).
The buttons of the settings rows do not touch their fields.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from PyQt5.QtWidgets import QApplication, QHBoxLayout, QLabel, QMainWindow, QWidget

from src.gimap.app.presentation import i18n
from src.gimap.app.presentation.i18n import DEFAULT_LANGUAGE, apply_language
from src.gimap.app.presentation.task_runner import TaskRunner
from src.gimap.features.assistant.application import PROVIDERS
from src.gimap.features.assistant.application.pipeline_geometry import INCIDENCE_MISSING, SEARCHED
from src.gimap.features.assistant.infrastructure.claude_code_agent import INSTALL_HINT
from src.gimap.features.assistant.presentation import (
    AssistantController,
    AssistantSettingsPage,
    AssistantStartDialog,
    GuidedAnalysis,
    GuidedResultsPanel,
)
from src.gimap.features.assistant.presentation.code_section import code_error_text, describe_code_status
from src.gimap.features.assistant.presentation.guided_text import IDLE_TEXT, NO_FRAME_TEXT, live_summary
from src.gimap.features.assistant.presentation.guided_words import words
from src.gimap.features.assistant.presentation.provider_section import describe_provider
from src.gimap.integrations.state import InMemorySettingsRepository
from tests.test_assistant_gui import _app, _services
from tests.test_assistant_providers_gui import FakeProviders, _with

NOT_FOUND_AT = "Claude Code was not found at {cli}."
ZH_NEW = {
    # What the Chinese table gets for this release (the integrator merges these entries).
    INSTALL_HINT: "找不到 Claude Code。请先安装（claude.com/claude-code），或在 设置 ▸ AI 助手 中点“浏览…”选择 claude.exe"
                  "（Claude 桌面版自带一份，位于 %LOCALAPPDATA%\\Packages\\Claude_…\\LocalCache\\Roaming\\Claude\\claude-code）。",
    NOT_FOUND_AT: "在 {cli} 找不到 Claude Code。",
    "DeepSeek": "DeepSeek（深度求索）",
    "Qwen (Alibaba Cloud Model Studio)": "通义千问 Qwen（阿里云百炼）",
    "Kimi (Moonshot AI)": "Kimi（月之暗面）",
    "Zhipu GLM": "智谱 GLM",
    "Google Gemini (OpenAI-compatible endpoint)": "Google Gemini（OpenAI 兼容端点）",
    "OpenRouter (many models)": "OpenRouter（多家模型）",
    "SiliconFlow": "硅基流动 SiliconFlow",
    "Ollama (this computer)": "Ollama（本机）",
    "LM Studio (this computer)": "LM Studio（本机）",
    "Custom OpenAI-compatible service": "自定义 OpenAI 兼容服务",
    "Found by the automatic search, not named by you: {path} — {where}": "由自动搜索找到，不是你指定的：{path} — {where}",
    "found by the search: {name}": "由自动搜索找到：{name}",
    "The geometry is from a calibration the automatic search found ({name}): see Geometry below.":
        "几何来自自动搜索找到的标定（{name}）：见下面的“几何”。",
    "{place}, seen from the frame's folder": "{place}（从这一帧所在的文件夹看）",
    "{place}, subfolder {path}, seen from the frame's folder": "{place}，子文件夹 {path}（从这一帧所在的文件夹看）",
    "same folder": "同一个文件夹",
    "parent folder": "上一层文件夹",
    "sibling folder {name}": "同级文件夹 {name}",
    "two levels up": "往上两层的文件夹",
    "folder {name} two levels up": "往上两层里的文件夹 {name}",
    "three levels up": "往上三层的文件夹",
    "folder {name} three levels up": "往上三层里的文件夹 {name}",
    "near the frame": "这一帧附近",
    INCIDENCE_MISSING["gisaxs"]: "选项、笔记和仪器配置都没有给出 αi，因此使用 0°。qy 和面内间距几乎不变，但样品地平线会被放在直射束处："
                                 "每一行的出射角 αf（包括 Yoneda 带标出的 αf）都偏大 αi，直射束和真实地平线之间的那些行也被算作样品上方。",
    "fitting spheres and cylinders to I(qy) (about half a minute)…": "正在用球和圆柱模型拟合 I(qy)（大约半分钟）…",
    "no side maximum or shoulder away from qy = 0": "在 qy = 0 之外没有侧峰或肩峰",
}


@pytest.fixture()
def zh(monkeypatch):
    for english, chinese in ZH_NEW.items():
        chinese = chinese.replace("▸", "›")  # as the table is loaded (the Chinese fonts have no ▸)
        monkeypatch.setitem(i18n.ZH, english, chinese)
        monkeypatch.setitem(i18n._TO_ENGLISH, chinese, english)
    apply_language("zh")
    yield
    apply_language(DEFAULT_LANGUAGE)


class _Analyze:
    """What ``AnalyzeAutomation.status`` says: with no frame open the status names no path."""

    def __init__(self, path=None):
        self.path = path

    def status(self) -> dict:
        return {"file": Path(self.path).name if self.path else None, "path": self.path, "files_listed": 0,
                "mode": "auto", "measurement": None, "message": "No frame is open."}


def test_the_entries_of_this_release_are_in_the_table_whole_or_not_at_all() -> None:
    """Until the entries are merged the tests inject them; a merge that leaves some out, or changes the fields
    of one, would show English (or a broken sentence) in the middle of a Chinese window."""
    present = {english: english in i18n.ZH for english in ZH_NEW}
    assert all(present.values()) or not any(present.values()), [english for english, there in present.items() if not there]
    for english, chinese in ZH_NEW.items():
        merged = i18n.ZH.get(english, chinese)
        assert sorted(re.findall(r"\{(\w+)\}", merged)) == sorted(re.findall(r"\{(\w+)\}", english)), english
    # Every key is a text the code shows: the adapter's hint, the provider names, the sentence forms, the literals.
    from src.gimap.features.assistant.presentation.guided_results import SEARCHED_ABOVE, SEARCHED_NOTE
    from src.gimap.features.assistant.presentation.guided_words import SENTENCES

    shown = {INSTALL_HINT, NOT_FOUND_AT, SEARCHED_NOTE, SEARCHED_ABOVE, INCIDENCE_MISSING["gisaxs"], *SENTENCES,
             *(preset.name for preset in PROVIDERS)}
    literals = {"found by the search: {name}", "same folder", "parent folder", "two levels up", "three levels up",
                "near the frame", "fitting spheres and cylinders to I(qy) (about half a minute)…",
                "no side maximum or shoulder away from qy = 0"}
    package = Path(__file__).parents[1] / "src" / "gimap" / "features" / "assistant"
    source = "\n".join(path.read_text(encoding="utf-8") for path in sorted(package.rglob("*.py")))
    assert all(text in source for text in literals), [text for text in literals if text not in source]
    assert set(ZH_NEW) <= shown | literals, set(ZH_NEW) - shown - literals


# -- no frame open (R6) -----------------------------------------------------------------------------------


def test_run_and_find_geometry_with_no_frame_open_say_so_and_start_nothing() -> None:
    _app()
    analyze = _Analyze()
    guided = GuidedAnalysis(lambda: analyze)
    started = []
    guided.started.connect(started.append)
    for start in (guided.run, guided.find_geometry, guided.run_button.click):
        start()
        assert not guided.running() and started == [] and guided.report is None
        assert guided.status_label.text() == NO_FRAME_TEXT == "Open a frame first."  # Analyze's sentence
        assert guided.run_button.isEnabled() and guided.results.isHidden() and guided.progress_panel.isHidden()
    apply_language("zh")
    try:
        guided.refresh_language()
        assert guided.status_label.text() == "请先打开一帧。"
    finally:
        apply_language(DEFAULT_LANGUAGE)
        guided.refresh_language()
    # The first frame opened: the line is the one from before, not a notice about a frame that is there now.
    analyze.path = "C:/data/film_a.tif"
    guided.frame_shown(analyze.path)
    assert guided.status_label.text() == IDLE_TEXT
    assert not guided._no_frame()  # a frame is open: a run may start
    guided.results.deleteLater()


def test_a_script_without_an_analyze_page_is_not_held_back() -> None:
    _app()
    guided = GuidedAnalysis(lambda: None)  # no page to ask: the run itself says what is missing
    assert guided._status() == {} and not guided._no_frame() and guided.status_label.text() == IDLE_TEXT
    guided.results.deleteLater()


def test_the_notice_gives_way_to_the_results_line_it_covered() -> None:
    _app()
    analyze = _Analyze("C:/data/film_a.tif")
    guided = GuidedAnalysis(lambda: analyze)
    report = {"ok": True, "procedure": "giwaxs", "frame": analyze.path, "peaks": [], "rings": [],
              "needs_attention": [], "decisions": [], "steps": []}
    guided.frame_shown(analyze.path)
    guided._finished(report)
    done = guided.status_label.text()
    assert done.startswith("Done")
    analyze.path = None  # nothing selected in Analyze
    guided.run()
    assert guided.status_label.text() == NO_FRAME_TEXT and guided.report is report
    analyze.path = "C:/data/film_a.tif"
    guided.frame_shown(analyze.path)
    assert guided.status_label.text() == done
    guided.results.deleteLater()


def test_results_of_no_frame_are_never_named_as_another_files() -> None:
    """A run that found no frame open reports no file; the first frame shown afterwards must not read
    “Results are for ; run again for this frame”."""
    _app()
    guided = GuidedAnalysis(lambda: None)
    nothing = {"ok": False, "procedure": "giwaxs", "frame": None, "peaks": [], "rings": [], "decisions": [], "steps": [],
               "needs_attention": [{"item": "frame", "why": "No detector image is open.", "option": None, "hint": ""}]}
    guided._finished(nothing)
    assert guided.report is nothing and not guided.results.isHidden()
    guided.frame_shown("C:/data/film_a.tif")
    assert guided.status_label.text() == IDLE_TEXT and "Results are for" not in guided.status_label.text()
    assert guided.report is None and guided.results.isHidden() and guided.questions.isHidden()
    # A report of a real file is still named when another file is shown.
    guided._finished({**nothing, "ok": True, "frame": "C:/data/film_a.tif", "needs_attention": []})
    guided.frame_shown("C:/data/film_b.tif")
    assert guided.status_label.text() == "Results are for film_a.tif; run again for this frame"
    guided.results.deleteLater()


# -- a calibration the search found, in the Results tab (R3, R78) -------------------------------------------

LONG_PATH = r"C:\Users\someone\Documents\beamtime_2031\raw\calibration_files\saxs_agbh_pyfai\Pilatus1M_long_name_00001.poni"


def _geometry_report(*decisions) -> dict:
    return {
        "ok": True, "procedure": "geometry", "frame": "C:/data/lonely/galaxi_data.tif", "needs_attention": [], "steps": [],
        "geometry": {"source": "Pilatus1M_long_name_00001.poni", "distance_mm": 1583.0, "beam_center_px": [180.0, 263.9],
                     "wavelength_A": 1.0, "incidence_deg": 0.2},
        "calibration_quality": {"assessment": "a saved calibration file (not re-checked against a standard image)", "warnings": []},
        "decisions": [{"what": "geometry", "decision": "from Pilatus1M_long_name_00001.poni",
                       "why": "a saved calibration (not re-checked against a standard image)"}, *decisions],
    }


def _searched_decision(where: str = "folder data three levels up, subfolder data/saxs_agbh_pyfai") -> dict:
    return {"what": SEARCHED, "decision": LONG_PATH, "why": f"{where}, seen from the frame's folder"}


def test_the_results_name_a_searched_calibration_with_its_path_and_where_it_lies() -> None:
    _app()
    panel = GuidedResultsPanel(lambda: None)
    panel.show_report(_geometry_report(_searched_decision()))
    note = panel.findChild(QLabel, "guidedSearchedCalibration")
    assert note is not None and note.property("gimapRole") == "info" and not note.isHidden()
    assert note.text() == (
        "Found by the automatic search, not named by you: " + LONG_PATH.replace("\\", "/")
        + " — folder data three levels up, subfolder data/saxs_agbh_pyfai, seen from the frame's folder")
    # The path wraps (written with “/”): the narrow Results tab is not pushed wider than its view.
    assert panel.minimumSizeHint().width() <= 360, panel.minimumSizeHint()
    assert LONG_PATH in panel.report_view.toPlainText()  # the full text keeps the path as the system writes it
    assert panel.findChild(QLabel, "guidedSearchedAbove") is None  # Find Geometry: its outcome line says it
    # A calibration the person named, or a kept profile: no such note.
    panel.show_report(_geometry_report())
    QApplication.processEvents()
    assert panel.findChild(QLabel, "guidedSearchedCalibration") is None
    panel.deleteLater()


def test_an_analysis_says_under_its_outcome_that_the_geometry_was_searched(zh) -> None:
    """The Geometry section is the last of a long Results tab (below the peaks, the checks, the fit): one line
    under the outcome names the file and points there. Information only: no badge, no point to check."""
    _app()
    apply_language(DEFAULT_LANGUAGE)
    analysed = {**_geometry_report(_searched_decision()), "procedure": "giwaxs", "peaks": [], "rings": []}
    panel = GuidedResultsPanel(lambda: None)
    panel.show_report(analysed)
    above = panel.findChild(QLabel, "guidedSearchedAbove")
    assert above is not None and above.property("gimapRole") == "muted"
    assert above.text() == ("The geometry is from a calibration the automatic search found "
                            "(Pilatus1M_long_name_00001.poni): see Geometry below.")
    labels = panel.findChildren(QLabel)
    assert labels.index(above) == 1  # right under the outcome
    assert panel.findChild(QLabel, "guidedSearchedCalibration") is not None  # the path and the place: under Geometry
    apply_language("zh")
    panel.refresh_language()
    above = panel.findChild(QLabel, "guidedSearchedAbove")
    assert above.text() == "几何来自自动搜索找到的标定（Pilatus1M_long_name_00001.poni）：见下面的“几何”。"
    apply_language(DEFAULT_LANGUAGE)
    panel.show_report({**analysed, "decisions": analysed["decisions"][:1]})  # named, or a kept profile
    QApplication.processEvents()
    assert panel.findChild(QLabel, "guidedSearchedAbove") is None
    panel.deleteLater()


def test_the_searched_note_is_said_in_chinese_with_the_path_as_it_is(zh) -> None:
    _app()
    assert words("sibling folder calib, seen from the frame's folder") == "同级文件夹 calib（从这一帧所在的文件夹看）"
    assert words("same folder, seen from the frame's folder") == "同一个文件夹（从这一帧所在的文件夹看）"
    deep = "folder data three levels up, subfolder data/saxs_agbh_pyfai, seen from the frame's folder"
    assert words(deep) == "往上三层里的文件夹 data，子文件夹 data/saxs_agbh_pyfai（从这一帧所在的文件夹看）"
    for place, shown in (("parent folder", "上一层文件夹"), ("two levels up", "往上两层的文件夹"), ("three levels up", "往上三层的文件夹"),
                         ("folder calib_2019 two levels up", "往上两层里的文件夹 calib_2019"), ("near the frame", "这一帧附近")):
        assert words(f"{place}, seen from the frame's folder") == f"{shown}（从这一帧所在的文件夹看）"  # every place the search names
    assert words(INCIDENCE_MISSING["gisaxs"]).startswith("选项、笔记和仪器配置都没有给出 αi，因此使用 0°。qy 和面内间距几乎不变")
    panel = GuidedResultsPanel(lambda: None)
    panel.show_report(_geometry_report(_searched_decision("sibling folder calib")))
    note = panel.findChild(QLabel, "guidedSearchedCalibration")
    assert note.text() == ("由自动搜索找到，不是你指定的：" + LONG_PATH.replace("\\", "/")
                           + " — 同级文件夹 calib（从这一帧所在的文件夹看）")
    panel.deleteLater()


def test_every_place_the_file_search_writes_is_said_in_chinese(zh, tmp_path: Path) -> None:
    """The places are the file search's own words (``LocalFileExplorer``): if it words one differently, the
    note under the geometry would stay English unseen. Here the real search names them on a real folder tree."""
    from src.gimap.features.assistant.application.pipeline_geometry import SEEN_FROM_FRAME
    from src.gimap.features.assistant.infrastructure import LocalFileExplorer

    root = tmp_path / "top" / "beamtime" / "raw"
    frame = root / "film" / "film_00001.tif"
    spots = ("top/beamtime/raw/film", "top/beamtime/raw/film/sub", "top/beamtime/raw", "top/beamtime/raw/calib",
             "top/beamtime/raw/calib/deep", "top/beamtime", "top/beamtime/processed", "top/beamtime/processed/deep",
             "top", "top/data", "top/data/deep")
    for index, spot in enumerate(spots):
        folder = tmp_path / spot
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"geometry_{index}.poni").write_text("Distance: 0.3\n", encoding="utf-8")
    frame.write_bytes(b"x")
    relations = {entry["relation"] for entry in LocalFileExplorer().related_files(str(frame))["entries"]}
    assert {"same folder", "parent folder", "sibling folder calib", "two levels up", "folder processed two levels up",
            "three levels up", "folder data three levels up"} <= relations, relations
    assert any(", subfolder " in relation for relation in relations)
    for relation in relations:
        said = words(SEEN_FROM_FRAME.format(place=relation))
        assert said.endswith("（从这一帧所在的文件夹看）"), (relation, said)
        assert not re.search(r"\b(same|parent|sibling|folder|subfolder|levels|seen|from)\b", said), (relation, said)


def test_find_geometry_says_at_once_that_the_calibration_was_searched(zh) -> None:
    """The line on the step and in the status bar is what a person reads right after Find Calibration
    Automatically (the Results tab, with the path, is not brought forward for a geometry-only run)."""
    from src.gimap.features.assistant.presentation import automatic_outcome

    name = Path(LONG_PATH).name
    apply_language(DEFAULT_LANGUAGE)
    state, line = automatic_outcome(_geometry_report(_searched_decision()))
    assert state == "ok"
    assert line == f"Geometry — a saved calibration file (not re-checked against a standard image); found by the search: {name}."
    assert "found by the search" not in automatic_outcome(_geometry_report())[1]  # named, or a kept profile
    asked = {"item": "incidence angle αi", "why": "No αi.", "option": "incidence_deg", "hint": ""}
    state, line = automatic_outcome({**_geometry_report(_searched_decision()), "needs_attention": [asked]})
    assert state == "warn" and line.endswith(f"; found by the search: {name}; 1 question(s).")
    apply_language("zh")
    assert f"；由自动搜索找到：{name}。" in automatic_outcome(_geometry_report(_searched_decision()))[1]


# -- the status line while a run lasts (R37) ----------------------------------------------------------------

HALVES = ("mean: Both halves are usable (points in 100% and 100% of their |qy| ranges) and agree within 8 %: "
          "they are averaged where both exist, which halves the noise.")


def test_english_keeps_every_summary_of_a_running_step() -> None:
    assert live_summary("find_peaks", {"curve": "radial"}, "12 peak(s) on radial: 0.452, 0.899") == "12 peak(s) on radial: 0.452, 0.899"
    assert live_summary("set_halves", {"side": "mean"}, "halves: mean") == "Mean of both halves"
    assert live_summary("choose_halves", {}, HALVES).startswith("Mean of both halves: Both halves are usable")


def test_another_language_says_only_what_it_can_say_of_a_running_step(zh, monkeypatch) -> None:
    for english, chinese in {
        "Both halves are usable (points in {a} and {b} of their |qy| ranges) and {agree}: they are averaged where both "
        "exist, which halves the noise{extend}.": "两半都可用（{a}、{b}），并且{agree}：取平均{extend}。",
        "agree within {percent} %": "相差在 {percent} % 以内",
    }.items():
        monkeypatch.setitem(i18n.ZH, english, chinese)
    # A summary nobody translated would make the line half English: the step is said alone.
    assert live_summary("find_peaks", {"curve": "radial"}, "12 peak(s) on radial: 0.452, 0.899") == ""
    assert live_summary("use_geometry", {}, "geometry from Pilatus1M.poni → GIWAXS") == ""
    # What the table or the sentence forms know is said: the slow fit, the halves and their reason.
    fitting = "fitting spheres and cylinders to I(qy) (about half a minute)…"
    assert live_summary("fit_horizontal_cut", None, fitting) == "正在用球和圆柱模型拟合 I(qy)（大约半分钟）…"
    halves = live_summary("choose_halves", {}, HALVES)
    assert halves.endswith("两半都可用（100%、100%），并且相差在 8 % 以内：取平均。") and "Both halves" not in halves
    # An error is never swallowed, whatever its language.
    assert live_summary("find_peaks", {}, "Curve 'radial' is not available", failed=True) == "Curve 'radial' is not available"

    _app()
    guided = GuidedAnalysis(lambda: None)
    guided._stepped({"state": "start", "tool": "find_peaks", "arguments": {"curve": "radial"}})
    guided._stepped({"state": "done", "tool": "find_peaks", "arguments": {"curve": "radial"}, "error": False})
    line = guided._readable("find_peaks: 12 peak(s) on radial: 0.452, 0.899")
    assert " — " not in line and not re.search(r"[A-Za-z]{3,}", line), line  # the step alone, in Chinese
    line = guided._readable("fit_horizontal_cut: " + fitting)
    assert line.endswith(" — 正在用球和圆柱模型拟合 I(qy)（大约半分钟）…") and "fitting" not in line
    guided._stepped({"state": "start", "tool": "find_peaks", "arguments": {}})
    guided._stepped({"state": "done", "tool": "find_peaks", "arguments": {}, "error": True})
    assert guided._readable("find_peaks: Curve 'radial' is not available").endswith(" — Curve 'radial' is not available")
    guided.results.deleteLater()


# -- “Claude Code was not found …” (R24, R34) ---------------------------------------------------------------


def test_the_not_found_sentences_are_table_entries_together() -> None:
    """The install hint is keyed by the adapter's exact sentence (its “…”, “▸” and backslashes): whoever has
    the template of the short sentence must have the hint too, so a reworded hint cannot stay English unseen."""
    assert INSTALL_HINT.startswith("Claude Code was not found. Install it")
    assert (INSTALL_HINT in i18n.ZH) == (NOT_FOUND_AT in i18n.ZH)


def test_english_shows_the_adapters_sentences_as_they_are() -> None:
    assert describe_code_status({"error": INSTALL_HINT}) == (False, INSTALL_HINT)
    assert code_error_text("Claude Code was not found at D:/tools/claude.exe.") == "Claude Code was not found at D:/tools/claude.exe."


def test_claude_code_not_found_is_said_in_chinese_in_the_start_dialog_and_the_settings(zh, tmp_path: Path) -> None:
    _app()
    ready, text = describe_code_status({"error": INSTALL_HINT})
    assert not ready and text.startswith("找不到 Claude Code。") and "Install it" not in text
    assert "claude.com/claude-code" in text and "%LOCALAPPDATA%\\Packages\\Claude_…" in text  # names and paths stay
    assert describe_code_status({"error": "Claude Code was not found at D:/tools/claude.exe."})[1] == "在 D:/tools/claude.exe 找不到 Claude Code。"
    assert code_error_text("exit code 3: something else") == "exit code 3: something else"  # not ours: as given

    settings = InMemorySettingsRepository({})
    dialog = AssistantStartDialog(settings, status={"file": "film.tif", "measurement": "giwaxs"})
    dialog.set_status(*describe_code_status({"error": INSTALL_HINT}))
    assert dialog.credentials_label.text().startswith("找不到 Claude Code。") and not dialog.start_button.isEnabled()
    assert "was not found" not in dialog.credentials_label.text()

    tasks = TaskRunner()
    page = AssistantSettingsPage(settings, _services(tmp_path, None, code_info={"error": INSTALL_HINT}), tasks)
    page.code_section.check_button.click()
    assert tasks.wait(10)
    QApplication.processEvents()
    assert page.code_section.account_label.text().startswith("找不到 Claude Code。")
    tasks.shutdown()
    dialog.deleteLater()
    page.deleteLater()


def test_the_set_up_dialog_shows_the_whole_answer_and_fits_the_screen(tmp_path: Path) -> None:
    """The answer about Claude Code stood beside the buttons, where the form gave it one line: the hint that
    says where claude.exe is was cut off in the middle (and the dialog was taller than a laptop screen)."""
    _app()
    window = QMainWindow()
    controller = AssistantController(window, _services(tmp_path, None, code_info={"error": INSTALL_HINT}),
                                     settings=InMemorySettingsRepository({}), automation=lambda: None)
    dialog, page = controller.setup_dialog()
    dialog.show()
    QApplication.processEvents()
    page.code_section.check()
    assert controller.tasks.wait(10)
    for _ in range(3):
        QApplication.processEvents()
    answer = page.code_section.account_label
    assert answer.text() == INSTALL_HINT
    needed = answer.heightForWidth(answer.width())
    assert needed >= 2 * answer.fontMetrics().height()  # the hint wraps: more than the one line it used to get
    assert answer.height() >= needed  # every line of it
    screen = dialog.screen().availableGeometry()
    assert dialog.height() <= screen.height() and dialog.width() >= 640
    dialog.close()
    window.close()


# -- the provider names (R44) ---------------------------------------------------------------------------------


def _combo_texts(section) -> list[str]:
    return [section.provider_combo.itemText(index) for index in range(section.provider_combo.count())]


def test_the_provider_names_are_english_in_the_code_and_in_the_english_interface(tmp_path: Path) -> None:
    for preset in PROVIDERS:
        assert preset.name.isascii() and re.search(r"[A-Za-z]{3,}", preset.name), preset.name
    _app()
    settings = InMemorySettingsRepository({})
    tasks = TaskRunner()
    page = AssistantSettingsPage(settings, _with(_services(tmp_path, None), FakeProviders()), tasks)
    assert _combo_texts(page.provider_section) == [preset.name for preset in PROVIDERS]
    assert "Custom OpenAI-compatible service" in _combo_texts(page.provider_section)
    tasks.shutdown()
    page.deleteLater()


def test_chinese_shows_the_names_chinese_users_know_and_the_record_keeps_english(zh, tmp_path: Path) -> None:
    _app()
    settings = InMemorySettingsRepository({})
    tasks = TaskRunner()
    providers = FakeProviders()
    page = AssistantSettingsPage(settings, _with(_services(tmp_path, None), providers), tasks)
    section = page.provider_section
    shown = _combo_texts(section)
    assert shown[:3] == ["OpenAI", "DeepSeek（深度求索）", "通义千问 Qwen（阿里云百炼）"] and shown[-1] == "自定义 OpenAI 兼容服务"
    assert section.key() == "deepseek"  # the setting is the key, whatever the language
    ready, text, label = describe_provider(settings, providers.services())
    assert not ready and "DeepSeek（深度求索）" in text  # what the person reads
    assert label == "DeepSeek · deepseek-chat"  # what the run record is named by: English
    # Switching back to English restores the names in a combo that was built in Chinese.
    apply_language(DEFAULT_LANGUAGE, [page])
    assert _combo_texts(section) == [preset.name for preset in PROVIDERS]
    tasks.shutdown()
    page.deleteLater()


# -- the rows of the settings (R21) -----------------------------------------------------------------------------


def test_no_button_of_the_assistant_settings_touches_the_field_beside_it(tmp_path: Path) -> None:
    _app()
    tasks = TaskRunner()
    host = QWidget()
    page = AssistantSettingsPage(InMemorySettingsRepository({}), _with(_services(tmp_path, None), FakeProviders()), tasks, host)
    host.resize(760, 1100)
    page.setGeometry(0, 0, 760, 1100)
    host.show()
    QApplication.processEvents()
    rows = [layout for layout in page.findChildren(QHBoxLayout) if layout.count() >= 2]
    assert len(rows) >= 7  # program, account, two key rows, the model list, two test rows
    checked = 0
    for row in rows:
        widgets = [row.itemAt(index).widget() for index in range(row.count())]
        widgets = [widget for widget in widgets if widget is not None and widget.isVisibleTo(page)]
        for left, right in zip(widgets, widgets[1:]):
            gap = right.geometry().left() - left.geometry().right() - 1
            assert gap >= 4, (left.objectName() or left.text(), right.objectName() or right.text(), gap)
            checked += 1
    assert checked >= 9
    tasks.shutdown()
    host.close()
    host.deleteLater()
