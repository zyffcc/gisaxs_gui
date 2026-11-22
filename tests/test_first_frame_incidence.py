"""Follow the incidence-angle prompt on the public GISAXS example."""

from pathlib import Path

import pytest
from PyQt5.QtCore import Qt
from PyQt5.QtTest import QTest

from src.gimap.app.presentation.i18n import apply_language
from tests.test_analyze_workspace import _context, _done, _page


@pytest.mark.parametrize(("language", "hint"), [("en", "αi field above the image"), ("zh", "图像上方的 αi 框")])
def test_the_zero_angle_message_points_to_the_editable_field(language, hint):
    frame = Path(__file__).parent / "data/external/gisaxs_galaxi/galaxi_data.tif"
    page = _page(_context())
    try:
        apply_language(language, [page])
        page.show()
        page.add_paths([str(frame)])
        _done(page)
        assert hint in page.status_text()
        editor = page.incidence_spin.lineEdit()
        QTest.mouseClick(editor, Qt.LeftButton)
        page.incidence_spin.selectAll()
        QTest.keyClicks(editor, "0.463")
        QTest.keyClick(editor, Qt.Key_Return)
        _done(page)
        assert page.view_model.state.analysis.geometry.incidence_deg == pytest.approx(0.463)
        assert hint not in page.status_text()
    finally:
        page.dispose()
        page.close()
        apply_language("en")
