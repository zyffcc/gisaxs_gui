"""A file taken off Analyze's list takes the automatic analysis's reports of it along."""

from __future__ import annotations

from types import SimpleNamespace


class _Hidden:
    def __init__(self):
        self.hidden = 0

    def hide(self) -> None:
        self.hidden += 1

    def dismiss(self) -> None:
        self.hidden += 1

    def set_actions_enabled(self, _enabled) -> None:
        pass


def _guided(busy: bool = False):
    from src.gimap.features.assistant.presentation.guided_frames import GuidedFramesMixin

    class Guided(GuidedFramesMixin):
        def __init__(self):
            self._reports, self.report, self.start_report = {}, None, None
            self._shown_path = self._pending_path = self._frames_given = None
            self._frames_elsewhere = False
            self.results, self.questions, self.progress_panel = _Hidden(), _Hidden(), _Hidden()
            self.progress_panel.save_button = SimpleNamespace(setEnabled=lambda _on: None)
            self.said = []

        def _busy(self):
            return busy

        def _say(self, text):
            self.said.append(text())

        def _status(self):
            return {}

    return Guided()


def test_the_reports_of_a_removed_file_are_forgotten(tmp_path) -> None:
    from src.gimap.features.assistant.presentation.guided_frames import frame_key

    guided = _guided()
    shown, other = tmp_path / "a.tif", tmp_path / "b.tif"
    guided.report = {"frame": str(shown)}
    guided._shown_path = frame_key(shown)
    guided._reports[frame_key(other)] = ({"frame": str(other)}, None)
    guided.file_removed(str(other))
    assert frame_key(other) not in guided._reports and guided.report is not None  # the shown results stay
    guided.file_removed(str(shown))
    assert guided.report is None and guided._shown_path is None and guided.results.hidden == 1


def test_nothing_is_forgotten_during_a_run(tmp_path) -> None:
    from src.gimap.features.assistant.presentation.guided_frames import frame_key

    guided = _guided(busy=True)
    guided._reports[frame_key(tmp_path / "b.tif")] = ({}, None)
    guided.file_removed(str(tmp_path / "b.tif"))
    assert guided._reports
