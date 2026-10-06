"""Check of the Analyze cross-area items: what the review pass fixed on top of them.

A set-up the person loads makes its mode theirs again (a switch of the automatic run is no longer kept);
Clear forgets the frame the status line named; a language switch keeps only the axes the person zoomed and
the region list fitted to its rows; the αi tooltip is composed again in the new language.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from src.gimap.app.presentation import i18n
from src.gimap.app.presentation.i18n import apply_language
from tests.test_analyze_workspace import P08, GALAXI, _context, _done, _page, _public_page, requires_public


@pytest.fixture
def english():
    yield
    apply_language("en")


def test_a_loaded_setup_makes_its_mode_the_persons_again(tmp_path: Path) -> None:
    page = _page(_context())

    def saved():
        return page.view_model.settings.get("analyze", "mode")

    page.choose_mode("gisaxs")
    written = page.save_settings(tmp_path / "setup.json")  # a set-up in GISAXS
    page.choose_mode("auto")
    page.automation().set_mode("giwaxs", lambda ok, message: None, remember=False)  # the run's switch
    assert saved() == "auto" and page._kept_mode == "auto"
    assert page.load_settings(written)
    assert page.view_model.state.mode == "gisaxs" and saved() == "gisaxs"  # the loaded mode, not the kept one
    assert page._kept_mode is None
    page.dispose()
    page.close()


@requires_public
def test_clear_forgets_the_frame_the_status_named() -> None:
    page = _public_page()
    page.add_paths([str(GALAXI)])
    _done(page)
    assert page._shown_status is not None and page._shown_status[0] is page.view_model.state.analysis
    page.clear_files()
    assert page._shown_status is None
    page.refresh_language()  # nothing shown: no error, the status stays as Clear left it
    assert page.status_text() == "Ready"
    page.dispose()
    page.close()


@requires_public
def test_a_language_switch_keeps_only_the_axes_the_person_set(english) -> None:
    page = _public_page()
    page.add_paths([str(GALAXI)])
    _done(page)
    box = page.top_plot.plot.getViewBox()
    assert box.autoRangeEnabled()[0]  # nothing zoomed: qy follows the data (log I has its own robust range)
    apply_language("zh", [page])
    page.refresh_language()
    assert box.autoRangeEnabled()[0]  # still follows the data, not frozen at the range it had
    box.setXRange(0.01, 0.05, padding=0)  # the person zoomed along qy
    x_range = box.viewRange()[0]
    apply_language("en", [page])
    page.refresh_language()
    assert np.allclose(box.viewRange()[0], x_range)  # their qy range stays
    page.dispose()
    page.close()


@requires_public
def test_the_region_list_and_the_incidence_tip_follow_the_language(monkeypatch, english) -> None:
    from src.gimap.features.analyze.presentation.bindings.incidence import OVERRIDE_TIP

    page = _public_page()
    page.add_paths([str(P08)])
    _done(page)
    page.incidence_spin.setValue(0.2)  # αi set by hand: the tooltip names both values
    _done(page)
    monkeypatch.setitem(i18n.ZH, OVERRIDE_TIP, "你设置的 αi：{value}° · 配置：{profile}°")
    apply_language("zh", [page])
    page.refresh_language()
    assert page.incidence_spin.toolTip().startswith("你设置的 αi：0.2°")
    listing = page.region_list
    rows = sum(listing.sizeHintForRow(row) for row in range(listing.count()))
    assert listing.height() == min(300, 2 * listing.frameWidth() + rows + 2)
    page.dispose()
    page.close()
