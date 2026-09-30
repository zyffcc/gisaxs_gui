"""Robustness and memory between sessions: unexpected errors, unreadable files, recent data, last set-up."""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QMainWindow

from src.gimap.app.error_guard import LOG_NAME, ErrorGuard
from src.gimap.features.analyze.infrastructure.adapters.frames import (
    DetectorIoFrameSource,
    FrameReadError,
    readable_read_error,
)
from tests.test_analyze_workspace import _app


def _process(seconds: float = 0.3) -> None:
    import time

    app = _app()
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        app.processEvents()
        time.sleep(0.01)


# -- unexpected errors ----------------------------------------------------------------------


@pytest.fixture
def guard(tmp_path: Path):
    _app()
    seen = []
    previous = sys.excepthook
    guard = ErrorGuard(tmp_path / "logs", version="test", notify=lambda summary, details: seen.append((summary, details)))
    guard.install()
    guard.seen = seen
    yield guard
    guard.uninstall()
    assert sys.excepthook is previous


def test_an_error_in_a_slot_is_logged_and_shown_and_the_application_goes_on(guard) -> None:
    ran_after = []

    def broken() -> None:
        return 1 / 0

    QTimer.singleShot(0, broken)
    QTimer.singleShot(20, lambda: ran_after.append(True))
    _process()

    assert ran_after == [True]  # the event loop kept running
    assert len(guard.seen) == 1
    summary, details = guard.seen[0]
    assert summary.startswith("ZeroDivisionError") and "broken" in details
    log = (guard.log_dir / LOG_NAME).read_text(encoding="utf-8")
    assert "GIMaP test" in log and "ZeroDivisionError" in log


def test_an_error_in_a_worker_thread_reaches_the_window(guard) -> None:
    def work() -> None:
        raise RuntimeError("worker failed")

    thread = threading.Thread(target=work, name="reader")
    thread.start()
    thread.join()
    _process()

    assert [summary for summary, _details in guard.seen] == ["RuntimeError: worker failed"]
    assert "thread reader" in (guard.log_dir / LOG_NAME).read_text(encoding="utf-8")


def test_interrupts_go_to_the_default_handler(tmp_path: Path, monkeypatch) -> None:
    _app()
    passed = []
    monkeypatch.setattr(sys, "excepthook", lambda *args: passed.append(args[0]))
    shown = []
    guard = ErrorGuard(tmp_path, notify=lambda *args: shown.append(args)).install()
    try:
        sys.excepthook(KeyboardInterrupt, KeyboardInterrupt(), None)
        _process(0.1)
    finally:
        guard.uninstall()
    assert passed == [KeyboardInterrupt] and not shown
    assert not (tmp_path / LOG_NAME).exists()


def test_the_message_box_counts_a_burst_of_errors(tmp_path: Path) -> None:
    _app()
    guard = ErrorGuard(tmp_path)
    guard._show("ValueError: one", "details one")
    box = guard._box
    assert box is not None and box.isVisible() and not box.isModal()
    assert "ValueError: one" in box.message.text() and str(tmp_path) in box.message.text()
    assert box.details.isHidden()
    box.details_button.click()
    assert not box.details.isHidden()
    guard._show("ValueError: two", "details two")
    assert guard._box is box and box.headline.text().startswith("2 unexpected errors")
    assert box.details.toPlainText() == "details two"
    box.close()
    guard._show("ValueError: three", "details three")  # after closing: a new window, counting anew
    assert guard._box is not box and guard._box.headline.text().startswith("An unexpected error")
    guard._box.close()


# -- files that cannot be read ------------------------------------------------------------


def test_unreadable_frames_say_why_in_words(tmp_path: Path) -> None:
    source = DetectorIoFrameSource()
    text = tmp_path / "notes.tif"
    text.write_text("not an image", encoding="utf-8")
    empty = tmp_path / "frame_00001.cbf"
    empty.write_bytes(b"")
    cut = tmp_path / "scan_00001.nxs"
    cut.write_bytes(b"\x89HDF\r\n\x1a\n" + b"\x00" * 64)

    messages = {}
    for path in (text, empty, cut, tmp_path / "gone.cbf"):
        with pytest.raises(FrameReadError) as caught:
            source.load(path)
        messages[path.name] = str(caught.value)

    assert "not a detector image GIMaP can read" in messages["notes.tif"]
    assert "empty (0 bytes)" in messages["frame_00001.cbf"]
    assert "damaged or incomplete" in messages["scan_00001.nxs"]
    assert "not there any more" in messages["gone.cbf"]
    assert all("Traceback" not in message for message in messages.values())


def test_the_last_resort_message_keeps_the_library_text(tmp_path: Path) -> None:
    frame = tmp_path / "odd.edf"
    frame.write_bytes(b"x" * 10)
    assert readable_read_error(frame, RuntimeError("strange")) == "odd.edf could not be read (RuntimeError: strange)."
    assert "no permission" in readable_read_error(frame, PermissionError(13, "denied"))


# -- recent data and the last set-up --------------------------------------------------------


def _page(settings=None, profiles=None):
    from src.gimap.app import AppContext
    from src.gimap.features.analyze.bootstrap import create_analyze_view_model
    from src.gimap.features.analyze.presentation.page import AnalyzePage
    from src.gimap.integrations.state import (
        InMemorySessionRepository,
        InMemorySettingsRepository,
        InMemoryUserPreferencesRepository,
    )

    _app()
    context = AppContext(
        settings=settings if settings is not None else InMemorySettingsRepository({}),
        session=InMemorySessionRepository(),
        preferences=InMemoryUserPreferencesRepository(),
        instrument_profiles=profiles,
    )
    return AnalyzePage(create_analyze_view_model(context))


def test_recent_data_is_remembered_newest_first_without_repeats(tmp_path: Path) -> None:
    from src.gimap.features.analyze.presentation.bindings.session_memory import MAX_RECENT
    from src.gimap.integrations.state import InMemorySettingsRepository

    settings = InMemorySettingsRepository({})
    page = _page(settings)
    folders = [tmp_path / f"run_{index}" for index in range(MAX_RECENT + 2)]
    for folder in folders:
        folder.mkdir()
        page.remember_recent([folder])
    page.remember_recent([str(folders[3]).upper() if os.name == "nt" else folders[3]])

    recent = page.recent_paths()
    assert len(recent) == MAX_RECENT
    assert str(recent[0]).casefold() == str(folders[3]).casefold()  # opened again: first, and only once
    assert [str(path).casefold() for path in recent].count(str(folders[3]).casefold()) == 1
    assert recent[1] == folders[-1]

    folders[-1].rmdir()  # gone since: not offered
    assert folders[-1] not in _page(settings).recent_paths()

    page.clear_recent()
    assert page.recent_paths() == []
    page.close()


def test_open_recent_lists_the_paths_and_opens_the_chosen_one(tmp_path: Path) -> None:
    from src.gimap.app.presentation.menu_bar import MainMenuBar, MenuCommands

    _app()
    window = QMainWindow()
    frame = tmp_path / "frame_001.tif"
    frame.write_bytes(b"x")
    paths = [frame, tmp_path]
    opened, cleared = [], []
    commands = MenuCommands(recent_paths=lambda: list(paths), open_recent=opened.append,
                            clear_recent=lambda: cleared.append(True))
    menus = MainMenuBar(window, commands)
    menus._fill_recent()
    texts = [action.text() for action in menus.recent_menu.actions()]
    assert texts[0] == "&1  frame_001.tif" and texts[1].endswith("(folder)") and texts[-1] == "Clear the List"
    menus.recent_menu.actions()[0].trigger()
    menus.recent_menu.actions()[-1].trigger()
    assert opened == [str(frame)] and cleared == [True]

    paths.clear()
    menus._fill_recent()
    empty = menus.recent_menu.actions()
    assert [action.text() for action in empty] == ["No recent data"] and not empty[0].isEnabled()
    window.close()


def test_the_start_page_lists_recent_data(tmp_path: Path) -> None:
    from src.gimap.app.presentation.home_page import HomePage

    _app()
    page = HomePage()
    page.set_recent_provider(lambda: [tmp_path / "a.cbf", tmp_path])
    assert not page.recent_box.isHidden()
    assert page.recent_rows.count() == 2
    requested = []
    page.recentRequested.connect(requested.append)
    page.recent_rows.itemAt(0).widget().click()
    assert requested == [str(tmp_path / "a.cbf")]
    page.set_recent_provider(lambda: [])
    assert page.recent_box.isHidden()
    page.close()


def test_the_last_set_up_is_kept_and_offered_once_in_the_next_session(tmp_path: Path, monkeypatch) -> None:
    from src.gimap.features.analyze.domain import CutRegion
    from src.gimap.features.analyze.presentation.bindings import session_memory
    from src.gimap.integrations.state import InMemoryInstrumentProfileRepository
    from src.gimap.shared.geometry import InstrumentProfile
    from tests.test_assistant_calibration import SHAPE, giwaxs_frame, save_tiff
    from tests.test_giwaxs_workspace import GEOMETRY, _settle

    frame = save_tiff(tmp_path / "data" / "sample_001.tif", giwaxs_frame(seed=1))
    setup = tmp_path / "user" / "last_analyze_setup.json"
    profiles = InMemoryInstrumentProfileRepository([InstrumentProfile("synthetic", GEOMETRY, None, SHAPE)])
    offers = []
    monkeypatch.setattr(session_memory, "show_toast",
                        lambda _parent, text, **kwargs: offers.append((text, kwargs.get("action"))))

    first = _page(profiles=profiles)
    first.view_model.last_setup_path = setup
    assert first.save_last_setup() is None  # nothing analysed: nothing kept
    first.set_mode_choice("giwaxs")
    first.add_paths([str(frame)])
    _settle(first, lambda: first.view_model.state.analysis is not None and first.view_model.state.analysis.reduction is not None)
    first.view_model.add_region(CutRegion("Ring q 1.100", (1.07, 1.13)))
    first.dispose()  # closing Analyze keeps the set-up
    first.close()
    assert setup.is_file()

    second = _page(profiles=profiles)
    second.view_model.last_setup_path = setup
    second.add_paths([str(frame)])
    _settle(second, lambda: second.view_model.state.analysis is not None and second.view_model.state.analysis.reduction is not None)
    _process(0.2)
    assert len(offers) == 1
    text, (label, use) = offers[0]
    assert "last session" in text and "1 cut regions" in text and label == "Use It"
    assert not second.view_model.current_settings().giwaxs.regions  # never applied without asking
    use()
    _settle(second, lambda: bool(second.view_model.current_settings().giwaxs.regions))
    assert second.view_model.current_settings().mode == "giwaxs"

    second.run_analysis()  # later frames of the session do not ask again
    _settle(second, lambda: second.view_model.state.analysis.reduction is not None)
    _process(0.2)
    assert len(offers) == 1
    second.tasks.wait(30)
    second.close()


def test_the_fitting_curve_card_shows_q_in_both_units() -> None:
    from src.gimap.features.fitting.presentation.curve_card import CurveSourceCard

    class Ui:
        pass

    from PyQt5.QtWidgets import QLineEdit, QPushButton

    _app()
    ui = Ui()
    ui.fitImport1dFileButton = QPushButton()
    ui.fitImport1dFileValue = QLineEdit()
    card = CurveSourceCard(ui)
    card.show_curve("cut.dat", np.array([0.01, 0.2]))
    assert "q 0.1 … 2 nm⁻¹ (0.01 … 0.2 Å⁻¹)" in card.detail_label.text()
    card.show_curve("cut_nm.dat", np.array([0.1, 2.0]), unit="nm")
    assert "q 0.1 … 2 nm⁻¹ (0.01 … 0.2 Å⁻¹)" in card.detail_label.text()
