"""The component separation and the growth-process analysis of an in-situ result: read from the package's files
(``domain/process.py``), the Components picture of the main screen, the stage bands of the evolution and Details ▸
Growth Process. Offline, with invented results (``tests/sputter_client_fixtures.py``); every number is made up."""

from __future__ import annotations

import json
import os
import re

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tests.sputter_client_fixtures import entry, window, write_result
from tests.test_sputter_main_screen import _close, _page

KEYS = ("grains", "film", "substrate", "background")
PROCESS = {"source": "joint fit of the whole scan",
           "descriptors": {"h_cov50": 0.8, "h_base": 3.1, "D1": 6.0, "p_D": 0.5, "D_film": 14.0, "drops": 0,
                           "h_max": 4.0},
           "stages": [{"stage": "islands", "t_start": 2.0, "t_end": 2.0, "h_start": 0.4, "h_end": 0.4},
                      {"stage": "coalescence", "t_start": 4.0, "t_end": 6.0, "h_start": 0.8, "h_end": 1.2},
                      {"stage": "film", "t_start": 8.0, "t_end": 10.0, "h_start": 3.2, "h_end": 4.0}],
           "reference": {"system": "Ag/Zz", "n": 3, "D1": {"median": 7.0, "min": 6.5, "max": 7.5, "n": 3},
                         "p_D": {"median": 0.45, "min": 0.4, "max": 0.6, "n": 3}},
           "compare": {"D1": "below", "p_D": "within"}}


def _shares(t: float) -> dict:
    shares = {"grains": 0.8 - 0.05 * t, "film": 0.02 * t, "substrate": 0.15, "background": 0.05 + 0.03 * t}
    return {key: round(value, 5) for key, value in shares.items()}


def _write(folder, *, with_terms: bool = True):
    times = (2.0, 4.0, 6.0, 8.0, 10.0)
    stage = {2.0: "islands", 4.0: "coalescence", 6.0: "coalescence", 8.0: "film", 10.0: "film"}
    windows = []
    for t in times:
        item = window(t, [entry("joint", 2.0, 1.0 + 0.3 * t, kind="joint", components=_shares(t)),
                          entry("candidate 1", 3.0, 1.2 + 0.3 * t, rank=1, components=_shares(t + 1))],
                      smooth="joint", pick="joint")
        item["stage"] = stage[t]
        windows.append(item)
    out = write_result(folder, windows=windows)
    if not with_terms:
        return out
    main = json.loads((out / "main" / "main.json").read_text(encoding="utf-8"))
    main["components"] = [{"key": key, "label": key} for key in KEYS]
    main["process"] = PROCESS
    (out / "main" / "main.json").write_text(json.dumps(main), encoding="utf-8")
    for item in windows:
        path = out / "main" / item["arrays"]
        arrays = dict(np.load(path))
        shares = [[_shares(item["t"])[key] for key in KEYS], [_shares(item["t"] + 1)[key] for key in KEYS]]
        fractions = np.broadcast_to(np.asarray(shares)[:, :, None, None], (2, len(KEYS)) + arrays["measured"].shape)
        np.savez(path, **arrays, component_keys=np.array(KEYS), component_fractions=fractions.astype(np.float16))
    return out


def test_the_terms_and_the_process_are_read_from_the_result(tmp_path):
    from src.gimap.features.sputter.infrastructure import ClientFolderReader

    reader = ClientFolderReader()
    result = reader.read(_write(tmp_path / "out"))
    assert [key for key, _name in result.components] == list(KEYS)
    assert dict(result.components)["film"] == "Film fluctuation and network"
    first = result.windows[0]
    assert first.stage == "islands" and first.chosen().components["grains"] == 0.7
    process = result.process
    assert process.source == "joint fit of the whole scan" and process.system == "Ag/Zz" and process.system_scans == 3
    assert [stage.stage for stage in process.stages] == ["islands", "coalescence", "film"]
    assert process.ranges["D1"].low == 6.5 and process.positions == {"D1": "below", "p_D": "within"}
    assert [key for key, *_rest in process.shown()][:3] == ["h_cov50", "h_base", "D1"]
    patterns = reader.patterns(result, first)
    terms = patterns.components("joint")
    assert [key for key, _image in terms] == list(KEYS)
    model = patterns.model("joint")
    total = sum(image for _key, image in terms)
    good = np.isfinite(model)
    assert np.allclose(total[good], model[good], rtol=2e-3)  # the terms add up to the model (float16 shares)
    positions, shares = result.component_series()
    assert positions == [2.0, 4.0, 6.0, 8.0, 10.0] and shares["film"] == [0.04, 0.08, 0.12, 0.16, 0.2]


def test_an_older_result_has_no_terms_and_no_process(tmp_path):
    from src.gimap.features.sputter.infrastructure import ClientFolderReader

    reader = ClientFolderReader()
    result = reader.read(_write(tmp_path / "old", with_terms=False))
    assert result.components == () and result.process is None
    assert reader.patterns(result, result.windows[0]).components("joint") == []


def test_the_components_picture_follows_the_window_and_the_solution(tmp_path):
    page = _page(tmp_path)
    page.show()
    assert page.open_client_result(_write(tmp_path / "out"))
    keys = [page.image_switch.itemData(index) for index in range(page.image_switch.count())]
    assert keys == ["patterns", "space", "components"]
    page.image_switch.button(2).click()
    assert page.pattern_host.currentIndex() == 2
    assert page.components.titles[0].replace(" ", " ") == "Grains and islands · 70.0 %"
    chart = page.components.figure.axes[-1]
    assert [line.get_label() for line in chart.lines if not line.get_label().startswith("_")][:2] == [
        "Grains and islands", "Film fluctuation and network"]
    page.select_window(3)
    assert page.components.titles[0].replace(" ", " ") == "Grains and islands · 40.0 %"
    page.entry_control.button(1).click()  # another solution: its own terms
    assert page.components.titles[0].replace(" ", " ") == "Grains and islands · 35.0 %"
    # the stages behind the evolution, named where wide enough
    assert [text.get_text() for text in page.evolution._stage_texts] == ["Separate islands", "Coalescence",
                                                                         "Continuous film"]
    _close(page)


def test_details_show_the_growth_process(tmp_path):
    from src.gimap.features.sputter.presentation.process_tab import process_html

    page = _page(tmp_path)
    page.show()
    assert page.open_client_result(_write(tmp_path / "out"))
    assert page.tabs.tabText(1) == "Growth Process" and page.tab_process.currentIndex() == 1
    shown = page.process_view.toPlainText()
    for words in ("Stages of the growth", "Continuous film", "Growth laws", "Ag/Zz, 3 scans", "below the range",
                  "6.5", "Physical terms of the model", "Film fluctuation and network", "20.0 %"):
        assert words in shown, words
    page.entry_control.button(1).click()  # the table of terms follows the choice of the window shown
    assert "6.0 %" in page.process_view.toPlainText()
    assert page.open_client_result(_write(tmp_path / "old", with_terms=False))
    assert "no analysis of the growth process" in page.process_view.toPlainText()
    result = page.client_result
    assert "no analysis of the growth process" in process_html(result, {})
    _close(page)


def test_every_new_text_is_in_the_chinese_table():
    from src.gimap.app.presentation.i18n.zh import ZH
    from src.gimap.features.sputter.domain.process import COMPONENTS, DESCRIPTORS, STAGES
    from src.gimap.features.sputter.presentation import components_view, process_tab
    from src.gimap.features.sputter.presentation.views import sputter_page_view as details

    held = [*components_view.TEMPLATES, *process_tab.TEMPLATES, details.TAB_PROCESS,
            *(name for _key, name in COMPONENTS), *(name for _key, name in STAGES),
            *(name for _key, name, _unit in DESCRIPTORS)]
    assert [template for template in held if template not in ZH] == []
    for template in held:
        assert sorted(re.findall(r"\{(\w+)\}", template)) == sorted(re.findall(r"\{(\w+)\}", ZH[template])), template
