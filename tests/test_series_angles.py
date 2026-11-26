"""The rows of a live Series map split by incidence angle as they arrive (``application/series_angles.py``,
``AngleSplit``): the split kept from redraw to redraw is the split of all the rows, row for row — frames arriving at
several angles, a short sum completed, another angle chosen, a frame listed before the others, a frame not listed
yet. Synthetic rows with invented names."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.gimap.features.analyze.application.live_rows import ListedRows, numbers_agree
from src.gimap.features.analyze.application.series_angles import ALL, AngleSplit
from src.gimap.features.analyze.application.series_growing import GrowingSeriesMap
from src.gimap.features.analyze.application.series_map import build_series_map
from src.gimap.features.analyze.presentation.frame_sequences import FrameNumbers

PATTERN = (0.4, 0.4, 0.4, 0.1, 0.4, 0.4, 0.25, 0.400002, 0.4, 0.6, None, 0.4)
"""The reference angle, short looks at others between pulses, a motor reading off by a little, a frame without one."""


def _row(file: int, *, angle=None, summed: int = 1) -> tuple:
    angle = PATTERN[file % len(PATTERN)] if angle is None else angle
    return (None, None, "qy (Å⁻¹)", SimpleNamespace(incidence_deg=angle, summed=summed), (f"C:/Spray/p_{file:04d}.edf", 0))


def _numbers(files) -> FrameNumbers:
    return FrameNumbers((f"C:/spray/p_{file:04d}.edf", 1) for file in files)


def _same(split, whole) -> bool:
    return (len(split.kept) == len(whole.kept) and all(a is b for a, b in zip(split.kept, whole.kept))
            and len(split.aside) == len(whole.aside) and all(a is b for a, b in zip(split.aside, whole.aside))
            and (split.angle, split.groups, split.rows, split.line) == (whole.angle, whole.groups, whole.rows, whole.line)
            and dict(split.places) == dict(whole.places))


def test_a_live_split_is_the_split_of_all_the_rows() -> None:
    splitter, files, rows, aside, choice, grown = AngleSplit(), [], [], [], None, 0
    for step, more in enumerate((1, 3, 8, 2, 12, 5, 0, 7, 4, 9, 3, 6)):
        new = list(range(len(files), len(files) + more))
        files.extend(new[:-1] if step == 4 else new)  # (the last frame of step 4 is listed one step later)
        if step == 5:
            files.append(new[0] - 1)
        rows = rows + [_row(file) for file in new]
        if step == 6:  # (a short sum completed: the same frame, summing more, in its place)
            rows[2] = _row(int(rows[2][4][0][-8:-4]), summed=3)
        if step == 8:
            choice = 0.1
        if step == 9:
            files.insert(3, 999)  # (a frame listed before the others: every number after it moves)
        if step == 10:
            choice = ALL
        numbers = _numbers(files)
        last = splitter._last
        split = splitter.split(rows, aside, numbers, choice)
        grown += last is not None and splitter._last is last
        assert _same(split, AngleSplit().split(rows, aside, numbers, choice)), step
        rows, aside = split.kept, split.aside
    assert grown >= 5  # (most redraws only added the rows that arrived)


def test_the_angle_of_most_rows_can_change_while_rows_arrive() -> None:
    splitter, rows, aside = AngleSplit(), [], []
    files = list(range(12))
    for step, angle in enumerate((0.4,) * 3 + (0.1,) * 6):
        rows = rows + [_row(step, angle=angle)]
        split = splitter.split(rows, aside, _numbers(files), None)
        assert _same(split, AngleSplit().split(rows, aside, _numbers(files), None))
        rows, aside = split.kept, split.aside
    assert split.angle == 0.1 and len(split.kept) == 6 and len(split.aside) == 3


def _listed_whole(rows, numbers) -> list:
    """The order the watch made from all its rows: every listed frame once (the row summing more), by number."""
    unique = {}
    for row in rows:
        identity = (str(row[4][0]).casefold(), int(row[4][1]))
        if identity in numbers and (identity not in unique or row[3].summed > unique[identity][3].summed):
            unique[identity] = row
    return [unique[key] for key in sorted(unique, key=numbers.get)]


def test_the_rows_of_a_watch_keep_the_order_of_the_frame_list() -> None:
    listed, files, rows, grown = ListedRows(), [], [], 0
    for step, more in enumerate((2, 4, 3, 6, 1, 5, 4, 2, 3)):
        new = list(range(len(files), len(files) + more))
        files.extend(new[:-1] if step == 2 else new)  # (a frame not listed yet: left out, as the watch does)
        rows = rows + [_row(file) for file in new]
        if step == 4:
            rows[1] = _row(1, summed=2)  # (a short sum completed)
        if step == 6:
            files.insert(2, 999)  # (a frame listed before the others)
        numbers = _numbers(files)
        last = listed._last
        ordered, places = listed.order(rows, numbers)
        grown += last is not None and listed._last is last
        want = _listed_whole(rows, numbers)
        assert len(ordered) == len(want) and all(a is b for a, b in zip(ordered, want)), step
        assert all(places[(str(row[4][0]).casefold(), 0)] == place for place, row in enumerate(ordered))
        rows = ordered
    assert grown >= 4


def test_frame_numbers_agree_while_the_list_grows_at_its_end() -> None:
    before = _numbers(range(10))
    assert numbers_agree(before, _numbers(range(14))) and numbers_agree(before, before)
    assert not numbers_agree(before, _numbers([0, 1, 99, *range(2, 10)]))  # (one listed before the others)
    assert not numbers_agree(_numbers(range(14)), before)
    assert numbers_agree({("a", 0): 1}, {("a", 0): 1, ("b", 0): 2})
    assert not numbers_agree({("a", 0): 1}, {("a", 0): 2}) and not numbers_agree(before, {("a", 0): 1})
    assert numbers_agree(None, None) and not numbers_agree(None, before)


@pytest.mark.parametrize("late", [False, True])
def test_a_growing_map_takes_frame_numbers_that_agree_as_they_are(late) -> None:
    """Rows numbered as before are not looked up again; a frame without a number (listed late) is numbered by its
    row until it is listed, and the map then counts it as listed."""
    import numpy as np

    from src.gimap.features.analyze.application.series_map import RowLabel

    x = np.linspace(-0.1, 0.1, 11)

    def row(file):
        name = f"p_{file:04d}.edf"
        label = RowLabel(name, name=name, source=f"C:/Spray/{name}", frame_index=0)
        return (x, np.full(11, float(file)), "qy (Å⁻¹)", label, (f"C:/Spray/{name}", 0))

    growing, rows, files = GrowingSeriesMap(), [], []
    for step in range(6):
        new = list(range(len(rows), len(rows) + 3))
        rows.extend(row(file) for file in new)
        files.extend(new[1:] if late and step == 2 else new)
        if late and step == 3:
            files.append(new[0] - 3)  # (the frame of step 2 listed after the frames of step 3)
        numbers = _numbers(files)
        grown = growing.build(list(rows), "horizontal", numbers=numbers)
        whole = build_series_map(list(rows), "horizontal", numbers=numbers)
        assert [fact.number for fact in grown.series.facts] == [fact.number for fact in whole.facts]
        if step and not (late and step == 3):
            assert grown.same == len(rows) - 3
