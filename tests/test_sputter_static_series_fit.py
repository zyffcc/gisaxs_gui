"""After Deposition: the samples of a series fitted together after the queue (``StaticService.fit_series``)."""

from pathlib import Path
from types import SimpleNamespace

from src.gimap.features.sputter.application.client_static import StaticService


class _Reader:
    def __init__(self, status="ok"):
        self.status, self.read_from = status, None

    def analysis(self, out):
        return {"status": self.status}

    def failure_details(self, out, analysis):
        return ""

    def read(self, out):
        self.read_from = out
        return SimpleNamespace(static=True, folder=out)


class _Runner:
    def __init__(self, supports=True):
        self.supports, self.calls = supports, []

    def supports_series(self):
        return self.supports

    def fit_series(self, folder, on_line, should_stop, workers=0):
        self.calls.append((Path(folder), workers))
        on_line("[series] ...")
        return SimpleNamespace(code=0, stopped=False, tail=[])


def _series(tmp_path, samples):
    tmp_path.mkdir(parents=True, exist_ok=True)
    for name in samples:
        (tmp_path / name).mkdir()
        (tmp_path / name / "frames.csv").write_text("label,frame\n", encoding="utf-8")
    return tmp_path


def test_a_series_of_samples_is_fitted_together_and_read_again(tmp_path):
    folder = _series(tmp_path, ["4_nm", "6_nm"])
    runner, reader, lines = _Runner(), _Reader(), []
    result = StaticService(runner, reader).fit_series(folder, lines.append, workers=3)
    assert runner.calls == [(folder, 3)] and lines == ["[series] ..."]
    assert result.static and reader.read_from == folder


def test_one_sample_or_an_older_package_skips_it(tmp_path):
    one = _series(tmp_path / "one", ["4_nm"])
    runner = _Runner()
    assert StaticService(runner, _Reader()).fit_series(one, lambda line: None) is None and not runner.calls
    two = _series(tmp_path / "two", ["4_nm", "6_nm"])
    old = _Runner(supports=False)
    assert StaticService(old, _Reader()).fit_series(two, lambda line: None) is None and not old.calls


def test_a_failed_series_fit_says_so(tmp_path):
    import pytest

    from src.gimap.features.sputter.application import ClientFailed

    folder = _series(tmp_path, ["4_nm", "6_nm"])
    with pytest.raises(ClientFailed) as caught:
        StaticService(_Runner(), _Reader(status="failed at series")).fit_series(folder, lambda line: None)
    assert "together" in str(caught.value)
