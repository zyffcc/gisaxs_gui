"""The release Help menu: its documents, its windows and its commands.

* The documents (User Manual, Physics and Methods, Third-party Notices) are rendered whole: a bare
  ``<placeholder>`` once blanked everything after it, so every heading of every document Help opens is
  checked here with its text, and every heading has the anchor its Contents link points at.
* Every command does what its entry says, without a modal dialog left open (all monkeypatched) and without
  opening a browser (``QDesktopServices.openUrl`` is monkeypatched).
* Keyboard Shortcuts lists every ``QShortcut`` of the window: the static table of page keys is checked
  against the real window.
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
import yaml
from PyQt5.QtWidgets import QApplication, QDialog, QMainWindow, QShortcut, QWidget

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
HELP_DOCUMENTS = [path for path in (DOCS / "User_Manual.md", DOCS / "User_Manual_zh.md", DOCS / "Physics_Manual.md",
                                    DOCS / "Physics_Manual_zh.md", ROOT / "THIRD_PARTY_NOTICES.md") if path.is_file()]
_MARKUP = re.compile(r"[`*_\\]")


def _app():
    return QApplication.instance() or QApplication([])


def _plain(text: str) -> str:
    return " ".join(_MARKUP.sub("", text).split())


def _markdown_headings(text: str) -> list[str]:
    """The ATX headings outside fenced code, with the targets of their links dropped."""
    headings, fence = [], False
    for line in text.splitlines():
        if re.match(r"^[ \t]*(```|~~~)", line):
            fence = not fence
            continue
        found = None if fence else re.match(r"^#{1,6}\s+(.+?)\s*#*\s*$", line)
        if found:
            headings.append(re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", found.group(1)))
    return headings


def _slug(heading: str) -> str:
    """GitHub's anchor of a Markdown heading (as tests/test_release_docs_contents.py), of the text as it reads:
    code marks and bold go, the underscore of a name in code stays."""
    shown = re.sub(r"[`*\\]", "", heading)
    return re.sub(r"[^\w\- ]", "", shown.strip().lower()).replace(" ", "-")


@pytest.fixture()
def opened(monkeypatch):
    """The URLs handed to the browser (none is opened)."""
    from PyQt5.QtGui import QDesktopServices

    urls = []
    monkeypatch.setattr(QDesktopServices, "openUrl", lambda url: urls.append(url) or True)
    return urls


# -- the renderer ----------------------------------------------------------------------------------------


@pytest.mark.parametrize("path", HELP_DOCUMENTS, ids=lambda path: path.name)
def test_every_heading_of_a_help_document_is_rendered_with_its_text_and_anchor(path) -> None:
    from src.gimap.app.presentation.help_documents import heading_texts, render_markdown

    _app()
    text = path.read_text(encoding="utf-8")
    page = render_markdown(text, path.stem, path.parent)
    wanted = _markdown_headings(text)
    assert [_plain(heading) for heading in heading_texts(page)] == [_plain(heading) for heading in wanted]
    anchors = set(re.findall(r'<h[1-6] id="([^"]+)"', page))
    for heading in wanted:
        assert _slug(heading) in anchors, heading
    for target in re.findall(r"\]\(#([^)\s]+)\)", text):  # the Contents links jump somewhere
        assert target in anchors, target
    words = len(re.sub(r"<[^>]+>", " ", page).split())
    assert words > 0.8 * len(re.sub(r"[#*`|>-]", " ", text).split())  # nothing after a heading is lost either


def test_a_bare_placeholder_never_swallows_what_follows() -> None:
    from src.gimap.app.presentation.help_documents import escape_raw_html, heading_texts, render_markdown

    _app()
    markdown = ("# Title\n\nFiles go to a subfolder named “<name>” and *<sample or batch> · <curve>*.\n\n"
                "## 2. After the placeholder\n\nStill here. Code `<kept>` and <https://example.org/a?b=1> stay.\n\n"
                "```text\nraw <tag> in a block\n```\n\nAn escaped \\<one\\> stays escaped once.\n")
    escaped = escape_raw_html(markdown)
    assert "“\\<name>”" in escaped and "*\\<sample or batch> · \\<curve>*" in escaped
    assert "`<kept>`" in escaped and "<https://example.org/a?b=1>" in escaped and "raw <tag> in a block" in escaped
    assert "\\<one\\>" in escaped and "\\\\<" not in escaped
    page = render_markdown(markdown, "Test", ROOT)
    assert heading_texts(page) == ["Title", "2. After the placeholder"]
    assert "Still here." in page and "&lt;name&gt;" in page and "&lt;curve&gt;" in page
    assert 'href="https://example.org/a?b=1"' in page
    # Without the escaping Qt drops the rest (the bug this guards against).
    from PyQt5.QtGui import QTextDocument

    raw = QTextDocument()
    raw.setMarkdown(markdown)
    assert "Still here." not in raw.toHtml()


def test_headings_get_github_anchors_and_links_point_at_the_files_beside_the_document(tmp_path) -> None:
    from src.gimap.app.presentation.help_documents import render_markdown

    _app()
    (tmp_path / "images").mkdir()
    markdown = ("# Manual\n\n[Second](#2-second-part) · [Licence](LICENSE) · [Web](https://example.org) · "
                "![shot](images/one.png)\n\n## 2. Second part\n\n## 2. Second part\n\n### 3.1 Ein Abschnitt — 中文\n")
    page = render_markdown(markdown, "Manual", tmp_path)
    assert re.findall(r'<h[1-6] id="([^"]+)"', page) == ["manual", "2-second-part", "2-second-part-1",
                                                         "31-ein-abschnitt--中文"]
    assert 'href="#2-second-part"' in page and 'href="https://example.org"' in page
    assert 'src="file:///' in page and "images/one.png" in page and "<base" not in page  # a <base> breaks #links
    assert re.search(r'href="file:///[^"]*/LICENSE"', page)


@pytest.fixture()
def temporary(monkeypatch, tmp_path):
    """The temporary folder the pages are written to (not the real one)."""
    import tempfile

    folder = tmp_path / "temporary"
    folder.mkdir()
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(folder))
    return folder


def _document_links(page: Path) -> list:
    """``(page linked, anchor)`` of every link of ``page`` to another page in the temporary folder."""
    from PyQt5.QtCore import QUrl

    found = []
    for href in re.findall(r'href="([^"]+)"', page.read_text(encoding="utf-8")):
        url = QUrl(href.replace("&amp;", "&"))
        if url.isLocalFile() and url.toLocalFile().endswith(".html"):
            found.append((Path(url.toLocalFile()), url.fragment()))
    return found


def test_a_link_to_another_document_opens_its_page_at_the_section(tmp_path, temporary) -> None:
    from src.gimap.app.presentation.help_documents import page_path, write_page

    _app()
    docs = tmp_path / "docs"
    docs.mkdir()
    manual, physics = docs / "User_Manual.md", docs / "Physics_Manual.md"
    changelog, notices = tmp_path / "CHANGELOG.md", tmp_path / "THIRD_PARTY_NOTICES.md"
    manual.write_text("# Manual\n\nSee [the physics](Physics_Manual.md#2-second-part), [the changes](../CHANGELOG.md), "
                      "[the licence](../LICENSE) and [a lost note](Gone.md).\n", encoding="utf-8")
    physics.write_text("# Physics\n\n[Back](User_Manual.md#manual)\n\n## 2. Second part\n", encoding="utf-8")
    changelog.write_text("# Changelog\n\n[Notices](THIRD_PARTY_NOTICES.md)\n", encoding="utf-8")
    notices.write_text("# Notices\n\nPyQt5\n", encoding="utf-8")
    (tmp_path / "LICENSE").write_text("MIT\n", encoding="utf-8")
    page = write_page(manual, "User Manual")
    assert page == page_path(manual) and page.parent == temporary
    assert _document_links(page) == [(page_path(physics), "2-second-part"), (page_path(changelog), "")]
    text = page.read_text(encoding="utf-8")
    assert re.search(r'href="file:///[^"]*/LICENSE"', text) and re.search(r'href="file:///[^"]*/docs/Gone\.md"', text)
    raw = [href for href in re.findall(r'href="([^"]+)"', text) if ".md" in href]
    assert len(raw) == 1 and raw[0].endswith("/docs/Gone.md")  # only the missing one is left as it was
    shown = page_path(physics).read_text(encoding="utf-8")  # the linked documents are rendered, theirs too
    assert '<h2 id="2-second-part"' in shown and "<title>Physics</title>" in shown
    assert _document_links(page_path(physics)) == [(page, "manual")]
    assert _document_links(page_path(changelog)) == [(page_path(notices), "")] and page_path(notices).is_file()
    # A linked page is kept while its .md is unchanged, and rendered again once the .md is newer.
    with page_path(physics).open("a", encoding="utf-8") as kept:
        kept.write("<!-- kept -->")
    write_page(manual, "User Manual")
    assert page_path(physics).read_text(encoding="utf-8").endswith("<!-- kept -->")
    physics.write_text("# Physics\n\n## 3. Third part\n", encoding="utf-8")
    later = page_path(physics).stat().st_mtime + 5
    os.utime(physics, (later, later))
    write_page(manual, "User Manual")
    shown = page_path(physics).read_text(encoding="utf-8")
    assert '<h2 id="3-third-part"' in shown and "kept" not in shown
    # Two documents of one name (two README.md) never share a page.
    (tmp_path / "README.md").write_text("# One\n", encoding="utf-8")
    (docs / "README.md").write_text("# Two\n", encoding="utf-8")
    assert page_path(tmp_path / "README.md") != page_path(docs / "README.md")


@pytest.mark.parametrize("name", ["User_Manual.md", "User_Manual_zh.md"])
def test_the_links_of_the_manual_to_physics_and_methods_open_its_page_at_the_section(name, temporary) -> None:
    from src.gimap.app.presentation.help_documents import write_page

    _app()
    page = write_page(DOCS / name, "User Manual")
    links = _document_links(page)
    assert links and {target.name.split("_")[1] for target, _anchor in links} >= {"Physics"}
    for target, anchor in links:
        assert target.is_file(), target
        assert not anchor or f'id="{anchor}"' in target.read_text(encoding="utf-8"), (target.name, anchor)


def test_a_document_that_is_missing_is_said_and_nothing_opens(tmp_path, monkeypatch, opened) -> None:
    from src.gimap.app.presentation import help_documents

    _app()
    said = []
    monkeypatch.setattr(help_documents, "warn", lambda _parent, title, text: said.append((title, text)))
    monkeypatch.setattr(help_documents, "PHYSICS_MANUAL", tmp_path / "Physics_Manual.md")
    monkeypatch.setattr(help_documents, "PHYSICS_MANUAL_ZH", tmp_path / "Physics_Manual_zh.md")
    assert help_documents.open_physics_manual(None) is None
    assert said == [("Physics and Methods", f"The physics and methods manual was not found:\n{tmp_path / 'Physics_Manual.md'}")]
    assert opened == []
    # Once it is there it opens as a web page; the Chinese interface takes the Chinese edition when it exists.
    (tmp_path / "Physics_Manual.md").write_text("# Physics\n\nq = 4π sin θ / λ\n", encoding="utf-8")
    page = help_documents.open_physics_manual(None)
    assert page is not None and "q = 4π sin θ / λ" in page.read_text(encoding="utf-8")
    assert opened and opened[-1].toLocalFile() == str(page).replace("\\", "/")
    assert help_documents.physics_manual_for_language("zh") == tmp_path / "Physics_Manual.md"
    (tmp_path / "Physics_Manual_zh.md").write_text("# 物理\n", encoding="utf-8")
    assert help_documents.physics_manual_for_language("zh") == tmp_path / "Physics_Manual_zh.md"


def test_the_third_party_notices_open_through_the_same_renderer(tmp_path, monkeypatch, opened) -> None:
    from src.gimap.app.presentation import help_documents

    _app()
    said = []
    monkeypatch.setattr(help_documents, "warn", lambda _parent, title, text: said.append(title))
    monkeypatch.setattr(help_documents, "THIRD_PARTY_NOTICES", tmp_path / "THIRD_PARTY_NOTICES.md")
    monkeypatch.setattr(help_documents, "PACKAGED_NOTICES", tmp_path / "licenses" / "THIRD_PARTY_NOTICES.md")
    assert help_documents.open_third_party_notices(None) is None and said == ["Third-party Notices"]
    (tmp_path / "licenses").mkdir()
    (tmp_path / "licenses" / "THIRD_PARTY_NOTICES.md").write_text("# Notices\n\nPyQt5 <version>\n", encoding="utf-8")
    page = help_documents.open_third_party_notices(None)  # the package's copy beside the licence texts
    assert page is not None and "PyQt5 &lt;version&gt;" in page.read_text(encoding="utf-8") and len(opened) == 1


# -- the Help menu ----------------------------------------------------------------------------------------


@pytest.fixture()
def menu_bar():
    from src.gimap.app.help_commands import HelpCommands
    from src.gimap.app.presentation.menu_bar import MainMenuBar, MenuCommands
    from src.gimap.app.presentation.navigation import NAVIGATION_ITEMS

    _app()
    window = QMainWindow()

    def nothing(*_arguments):
        return None

    commands = MenuCommands(open_files=nothing, settings=nothing, show_workspace=nothing, open_data_folder=nothing,
                            change_font_size=nothing, reset_font_size=nothing,
                            examples=HelpCommands(None).examples(), open_example=nothing, report_problem=nothing,
                            open_error_log=nothing, citation=HelpCommands.citation)
    bar = MainMenuBar(window, commands, NAVIGATION_ITEMS)
    yield bar
    for dialog in window.findChildren(QDialog):
        dialog.close()
    window.close()
    window.deleteLater()


HELP_ENTRIES = ["&User Manual", "&Physics and Methods", "&Keyboard Shortcuts…", "E&xample Data", "",
                "&Report a Problem…", "Open Error &Log Folder", "Open User &Data Folder", "",
                "&GitHub Repository", "Check for Upda&tes…", "How to &Cite GIMaP…", "Third-party &Notices", "",
                "&About GIMaP"]


def _mnemonics(actions) -> list[str]:
    letters = []
    for action in actions:
        found = re.search(r"&([^&])", action.text().replace("&&", ""))
        if not action.isSeparator():
            assert found, action.text()
            letters.append(found.group(1).upper())
    return letters


def test_the_help_menu_has_its_entries_with_a_key_of_their_own_in_both_languages(menu_bar) -> None:
    from src.gimap.app.presentation.i18n import apply_language

    help_menu, examples = menu_bar.help_menu, menu_bar.examples_menu
    assert [action.text() for action in help_menu.actions()] == HELP_ENTRIES
    assert [action.text() for action in examples.actions()] == [
        "GISAXS — Ag &Nanoparticles (GALAXI)…", "SAXS — Ag&Bh Calibrant (pyFAI)…",
        "GIWAXS — &MAPbI₃ Film (DESY P08)…", "GIWAXS — &PbTe, Tilted Detector (Xeuss)…"]
    for action in examples.actions():  # source and licence in the tooltip
        assert "Source:" in action.toolTip() and "Licence:" in action.toolTip(), action.text()
    entries = [action for action in (*help_menu.actions(), *examples.actions()) if not action.isSeparator()]
    english = [(action.text(), action.toolTip()) for action in entries]
    try:
        apply_language("zh", [menu_bar.window])
        help_menu.aboutToShow.emit()
        for menu in (help_menu, examples):
            letters = _mnemonics(menu.actions())
            assert len(letters) == len(set(letters)), [action.text() for action in menu.actions()]
        for action, (text, tip) in zip(entries, english):  # every entry and its tooltip in Chinese
            assert action.text() != text and action.toolTip() != tip, text
        assert menu_bar.examples_menu.menuAction().toolTip().startswith("GIMaP 自带")
    finally:
        apply_language("en", [menu_bar.window])
    for menu in (help_menu, examples):
        letters = _mnemonics(menu.actions())
        assert len(letters) == len(set(letters))


def test_check_for_updates_names_the_running_version_and_only_opens_the_browser(menu_bar, opened) -> None:
    from src.gimap.app.presentation.i18n import apply_language
    from src.gimap.app.presentation.menu_bar import GITHUB_URL
    from src.gimap.shared.version import __version__

    action = menu_bar.actions["check_updates"]
    assert f"GIMaP {__version__}" in action.toolTip() and "no network request" in action.toolTip()
    action.trigger()
    assert [url.toString() for url in opened] == [GITHUB_URL + "/releases"]  # /releases/latest skips pre-releases
    try:
        apply_language("zh", [menu_bar.window])
        menu_bar.sync()
        assert action.toolTip().startswith(f"你正在运行 GIMaP {__version__}")
    finally:
        apply_language("en", [menu_bar.window])
        menu_bar.sync()
    assert action.toolTip().startswith("You are running")


def test_how_to_cite_shows_the_reference_with_copy_and_copy_bibtex(menu_bar) -> None:
    from src.gimap.shared.version import __version__

    menu_bar.actions["cite"].trigger()
    dialog = menu_bar.window.findChild(QDialog, "citationDialog")
    assert dialog is not None and dialog.isVisible() and not dialog.isModal()
    text = dialog.findChild(QWidget, "citationText").toPlainText()
    bibtex = dialog.findChild(QWidget, "citationBibtex").toPlainText()
    assert __version__ in text and f"version = {{{__version__}}}" in bibtex
    dialog.findChild(QWidget, "citationCopy").click()
    assert QApplication.clipboard().text() == text
    dialog.findChild(QWidget, "citationCopyBibtex").click()
    assert QApplication.clipboard().text() == bibtex
    assert dialog.findChild(QWidget, "citationStatus").text()  # the click is seen to work


def test_about_names_the_licence_and_the_versions_and_copies_them(menu_bar) -> None:
    from src.gimap.app.presentation.help_dialogs import runtime_versions
    from src.gimap.shared.version import __version__

    menu_bar.actions["about"].trigger()
    dialog = menu_bar.window.findChild(QDialog, "aboutDialog")
    assert dialog is not None and not dialog.isModal()
    shown = dialog.findChild(QWidget, "aboutText").text()
    assert "MIT licence" in shown and "Third-party Notices" in shown and f"v{__version__}" in shown
    assert all(f"{name} {version}" in shown for name, version in runtime_versions().items())
    dialog.findChild(QWidget, "aboutCopy").click()
    copied = QApplication.clipboard().text()
    assert __version__ in copied and "numpy" in copied and "<" not in copied
    menu_bar.actions["about"].trigger()  # one window at a time
    assert len(menu_bar.window.findChildren(QDialog, "aboutDialog")) == 1
    from src.gimap.app.presentation.i18n import apply_language

    try:  # the composed texts follow a switch of the language while the window is open
        apply_language("zh", [menu_bar.window])
        assert "MIT 许可证" in dialog.findChild(QWidget, "aboutText").text() and dialog.windowTitle() == "关于 GIMaP"
    finally:
        apply_language("en", [menu_bar.window])
    assert "MIT licence" in dialog.findChild(QWidget, "aboutText").text()
    dialog.close()
    QApplication.processEvents()
    apply_language("en", [menu_bar.window])  # a deleted window is no longer told


def test_keyboard_shortcuts_lists_the_menu_keys_live_and_the_page_keys(menu_bar) -> None:
    from PyQt5.QtGui import QKeySequence

    from src.gimap.app.presentation.help_dialogs import PAGE_KEYS, page_key_rows
    from src.gimap.app.presentation.i18n import apply_language

    menu_bar.actions["keyboard_shortcuts"].trigger()
    dialog = menu_bar.window.findChild(QDialog, "keyboardShortcutsDialog")
    assert dialog is not None and not dialog.isModal()
    table = dialog.table
    rows = [tuple(table.item(row, column).text() for column in range(3)) for row in range(table.rowCount())]
    native = QKeySequence("Ctrl+O").toString(QKeySequence.NativeText)
    assert ("File menu", native, "Open Data…") in rows
    assert ("Help menu", QKeySequence("F1").toString(QKeySequence.NativeText), "User Manual") in rows
    larger = ", ".join(key.toString(QKeySequence.NativeText) for key in menu_bar.actions["font_larger"].shortcuts())
    assert ("View menu", larger, "Font Size ▸ Larger") in rows
    assert rows[-len(PAGE_KEYS):] == page_key_rows() and len(PAGE_KEYS) > 10
    assert not [row for row in rows if row[2] in ("Undo", "Repeat Analysis")]  # page keys: in the page rows only
    assert table.property("gimapTableCopy")  # Ctrl+C copies rows
    try:
        apply_language("zh", [menu_bar.window, dialog])
        rows = [tuple(table.item(row, column).text() for column in range(3)) for row in range(table.rowCount())]
        assert ("文件菜单", native, "打开数据…") in rows and ("分析", "F5", "重新分析这一帧（菜单：分析 › 重新分析）") in rows
        assert table.horizontalHeaderItem(0).text() == "在哪里"
    finally:
        apply_language("en", [menu_bar.window, dialog])


# -- the commands -----------------------------------------------------------------------------------------


class _Page:
    def __init__(self, accept=True):
        self.imported, self.accept = [], accept

    def import_paths(self, paths):
        self.imported.append(list(paths))
        return self.accept


class _Window(QMainWindow):
    def __init__(self, data_dir=None, page=None):
        super().__init__()
        self.components = type("Components", (), {"analyze_page": page or _Page()})()
        self.app_context = type("Context", (), {"data_dir": data_dir})()


class _Menus:
    def __init__(self, window):
        self.window, self.shown = window, []

    def show_workspace(self, key):
        self.shown.append(key)


@pytest.fixture()
def commands(monkeypatch, tmp_path):
    from src.gimap.app import help_commands

    _app()
    said = []
    monkeypatch.setattr(help_commands, "inform", lambda _parent, title, text: said.append(("inform", title, text)))
    monkeypatch.setattr(help_commands, "warn", lambda _parent, title, text: said.append(("warn", title, text)))
    window = _Window(tmp_path / "home")
    menus = _Menus(window)
    help = help_commands.HelpCommands(menus)
    help.said = said
    yield help
    window.close()
    window.deleteLater()


def test_open_error_log_folder_says_when_nothing_was_logged(commands, opened) -> None:
    from src.gimap.app.error_guard import LOG_NAME

    logs = commands.window.app_context.data_dir / "logs"
    assert commands.open_error_log() is False
    assert commands.said == [("inform", "Error Log", "No error has been logged.")] and opened == []
    assert not logs.exists()  # nothing is created to show an empty folder
    logs.mkdir(parents=True)
    (logs / LOG_NAME).write_text("Traceback …\n", encoding="utf-8")
    assert commands.open_error_log() is True
    assert len(opened) == 1 and Path(opened[0].toLocalFile()) == logs.resolve()


def test_report_a_problem_copies_versions_only_then_opens_a_new_issue(commands, opened, monkeypatch) -> None:
    from src.gimap.app import help_commands
    from src.gimap.app.presentation.menu_bar import GITHUB_URL
    from src.gimap.shared.version import __version__

    toasts = []
    monkeypatch.setattr(help_commands, "show_toast", lambda _parent, text, **_options: toasts.append(text))
    summary = commands.report_problem()
    assert QApplication.clipboard().text() == summary
    assert f"GIMaP {__version__}" in summary and "numpy" in summary and "PyQt" in summary
    assert "/" not in summary and "\\" not in summary  # no path of any kind
    assert toasts and "clipboard" in toasts[0]
    assert [url.toString() for url in opened] == [GITHUB_URL + "/issues/new"]  # nothing is put in the URL


def test_every_example_of_the_menu_is_in_the_program_folder() -> None:
    import json

    from src.gimap.app.help_commands import EXAMPLES, EXAMPLES_FOLDER, example_frame

    manifest = json.loads((EXAMPLES_FOLDER / "manifest.json").read_text(encoding="utf-8"))
    assert sorted(example.key for example in EXAMPLES) == sorted(manifest)
    readme = (EXAMPLES_FOLDER / "README.md").read_text(encoding="utf-8")
    for example in EXAMPLES:
        assert example_frame(example.key).is_file()
        licence = example.tip.split("Licence: ")[1]
        assert licence.split(" ")[0].rstrip(",") in readme, example.key  # GPL-3.0, MIT, CC BY …


def test_an_example_is_copied_to_the_folder_chosen_and_opened_in_analyze(commands, monkeypatch, tmp_path) -> None:
    from src.gimap.app import help_commands

    toasts = []
    monkeypatch.setattr(help_commands, "show_toast", lambda _parent, text, **options: toasts.append((text, options)))
    asked = []
    target = tmp_path / "my examples" / "pbte"
    monkeypatch.setattr(help_commands, "ask_example_folder",
                        lambda _parent, name, default: asked.append((name, default)) or str(target))
    frame = commands.open_example("giwaxs_xeuss_pbte")
    assert asked == [("GIWAXS — PbTe, Tilted Detector (Xeuss)",
                      str(commands.default_folder("giwaxs_xeuss_pbte")))]
    assert asked[0][1].endswith(str(Path("GIMaP examples") / "giwaxs_xeuss_pbte"))
    assert frame == target / "PbTe_3010_r1_PT_1_10_0_00028.edf"
    assert sorted(path.name for path in target.iterdir()) == sorted(
        ["PbTe_3010_r1_PT_1_10_0_00028.edf", "mask_PbTe_3010.edf", "AgBe_250403_XENOCS.poni", "SOURCE_AND_LICENCE.md"])
    source = help_commands.EXAMPLES_FOLDER / "giwaxs_xeuss_pbte" / "AgBe_250403_XENOCS.poni"
    assert (target / source.name).read_bytes() == source.read_bytes()
    assert commands.window.components.analyze_page.imported == [[str(frame)]]
    assert commands.menus.shown == ["analyze"]
    assert toasts[-1][1]["level"] == "ok" and str(target) in toasts[-1][0] and toasts[-1][1]["action"]
    # Again into the same folder: a file changed there is kept, never overwritten.
    (target / source.name).write_text("edited", encoding="utf-8")
    commands.open_example("giwaxs_xeuss_pbte")
    assert (target / source.name).read_text(encoding="utf-8") == "edited"


def test_an_example_copy_that_stops_part_way_leaves_nothing_under_the_frame_name(monkeypatch, tmp_path) -> None:
    import errno
    import shutil

    from src.gimap.app import help_commands

    frame = help_commands.example_frame("giwaxs_xeuss_pbte")
    real_copy = shutil.copy2

    def disk_full(source, destination):
        if Path(source).name == frame.name:  # half of the frame is written, then the disk is full
            Path(destination).write_bytes(Path(source).read_bytes()[: frame.stat().st_size // 2])
            raise OSError(errno.ENOSPC, "No space left on device")
        return real_copy(source, destination)

    target = tmp_path / "examples"
    monkeypatch.setattr(help_commands.shutil, "copy2", disk_full)
    with pytest.raises(OSError):
        help_commands.copy_example("giwaxs_xeuss_pbte", target)
    assert not (target / frame.name).exists()
    assert not [path for path in target.iterdir() if path.name.endswith(".part")]
    monkeypatch.setattr(help_commands.shutil, "copy2", real_copy)  # space freed: the next copy makes it afresh
    copied = help_commands.copy_example("giwaxs_xeuss_pbte", target)
    assert copied.read_bytes() == frame.read_bytes()


def test_an_example_is_never_copied_when_the_folder_is_not_chosen(commands, monkeypatch, tmp_path) -> None:
    from src.gimap.app import help_commands

    monkeypatch.setattr(help_commands, "ask_example_folder", lambda *_args: "")
    assert commands.open_example("gisaxs_galaxi") is None
    assert commands.window.components.analyze_page.imported == [] and commands.menus.shown == []
    # The GALAXI folder holds an export folder of the tests: only the files are copied, never a folder.
    monkeypatch.setattr(help_commands, "ask_example_folder", lambda *_args: str(tmp_path / "galaxi"))
    monkeypatch.setattr(help_commands, "show_toast", lambda *_args, **_options: None)
    commands.window.components.analyze_page.accept = False  # Analyze busy: copied all the same, said so
    assert commands.open_example("gisaxs_galaxi") == tmp_path / "galaxi" / "galaxi_data.tif"
    assert all(path.is_file() for path in (tmp_path / "galaxi").iterdir())
    assert (tmp_path / "galaxi" / "galaxi_bornagain.poni").is_file() and commands.menus.shown == []


def test_an_example_missing_from_the_installation_is_said(commands, monkeypatch, tmp_path) -> None:
    from src.gimap.app import help_commands

    monkeypatch.setattr(help_commands, "EXAMPLES_FOLDER", tmp_path / "nothing")
    monkeypatch.setattr(help_commands, "ask_example_folder", lambda *_args: pytest.fail("asked although missing"))
    assert commands.open_example("saxs_agbh_pyfai") is None
    assert commands.said and commands.said[0][0] == "warn" and "not in this installation" in commands.said[0][2]


def test_the_folder_dialog_proposes_the_default_folder(monkeypatch) -> None:
    from src.gimap.app.presentation.help_dialogs import ask_example_folder

    _app()
    monkeypatch.setattr(QDialog, "exec_", lambda self: QDialog.Accepted)
    assert ask_example_folder(None, "GISAXS", str(Path.home() / "GIMaP examples" / "x")) == str(
        Path.home() / "GIMaP examples" / "x")
    monkeypatch.setattr(QDialog, "exec_", lambda self: QDialog.Rejected)
    assert ask_example_folder(None, "GISAXS", "anything") == ""


def test_the_citation_comes_from_citation_cff_with_the_running_version_and_no_made_up_doi(tmp_path) -> None:
    from src.gimap.app.help_commands import CITATION_FILE, read_citation
    from src.gimap.app.presentation.menu_bar import GITHUB_URL
    from src.gimap.shared.version import __version__

    citation = read_citation()
    assert __version__ in citation.text and __version__ in citation.bibtex
    cff = yaml.safe_load(CITATION_FILE.read_text(encoding="utf-8")) if CITATION_FILE.is_file() else {}
    if "doi" not in str(cff).lower():
        assert "doi" not in citation.text.lower() and "doi" not in citation.bibtex.lower() and citation.doi == ""
    if cff:
        assert cff["title"] in citation.text and cff["authors"][0]["family-names"] in citation.bibtex
    fallback = read_citation(tmp_path / "missing.cff", "9.8.7")
    assert "9.8.7" in fallback.text and GITHUB_URL in fallback.text and "doi" not in fallback.bibtex.lower()
    (tmp_path / "CITATION.cff").write_text(
        "cff-version: 1.2.0\ntitle: Invented Tool\ndoi: 10.0000/invented.1\ndate-released: 2030-01-02\n"
        "authors:\n  - family-names: Doe\n    given-names: Jane Ann\n  - name: Some Lab\n"
        "preferred-citation:\n  type: article\n  title: An invented article\n  journal: J. Invented\n"
        "  volume: 9\n  start: 1\n  end: 7\n  year: 2031\n  authors:\n    - family-names: Doe\n      given-names: Jane\n",
        encoding="utf-8")
    given = read_citation(tmp_path / "CITATION.cff", "9.8.7", GITHUB_URL)
    assert given.doi == "10.0000/invented.1" and "https://doi.org/10.0000/invented.1" in given.text
    assert "Doe, J. A., & Some Lab (2030). Invented Tool (version 9.8.7)" in given.text
    assert given.text.startswith("Doe, J. (2031). An invented article. J. Invented, 9, 1–7.")
    assert "@article{" in given.bibtex and "pages = {1--7}" in given.bibtex and "doi = {10.0000/invented.1}" in given.bibtex


# -- the keys of the real window -----------------------------------------------------------------------


def _settle(seconds: float = 0.2) -> None:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        QApplication.processEvents()
        time.sleep(0.01)


def test_the_page_keys_of_keyboard_shortcuts_are_every_shortcut_of_the_window() -> None:
    from main import MainWindow
    from src.gimap.app import AppContext
    from src.gimap.app.presentation.help_dialogs import PAGE_KEYS
    from src.gimap.app.presentation.navigation import NAVIGATION_ITEMS
    from src.gimap.integrations.jobs import LocalProcessJobRunner
    from src.gimap.integrations.state import (
        InMemorySessionRepository,
        InMemorySettingsRepository,
        InMemoryUserPreferencesRepository,
    )

    _app()
    context = AppContext(settings=InMemorySettingsRepository(), session=InMemorySessionRepository(),
                         preferences=InMemoryUserPreferencesRepository(), jobs=LocalProcessJobRunner())
    window = MainWindow(context)
    try:
        window.show()
        end = time.monotonic() + 40
        while not window._initialization_completed and time.monotonic() < end:
            _settle(0.02)
        help_menu = window.menus.menu_bar.help_menu  # the composition root provides every Help command
        assert [action.text() for action in help_menu.actions()] == HELP_ENTRIES
        assert len(window.menus.menu_bar.examples_menu.actions()) == 4
        for item in NAVIGATION_ITEMS:  # pages built when first shown add their keys too
            window.runtime.navigate(item.key) if getattr(window, "runtime", None) else window.components.show_page(item.key)
            _settle(0.1)
        found = {}
        for shortcut in window.findChildren(QShortcut):
            owners, widget = set(), shortcut.parent()
            while widget is not None:
                owners.add(widget.objectName())
                widget = widget.parent()
            key = shortcut.key().toString()
            rows = [row for row in PAGE_KEYS if row.owner in owners and key in row.keys]
            assert rows, f"{key} of {shortcut.parent().objectName()} is not in Help ▸ Keyboard Shortcuts"
            for row in rows:
                found.setdefault((row.owner, row.where, row.what), set()).add(key)
        for row in PAGE_KEYS:  # a row of a page names keys the page has (none left over)
            if row.owner:
                assert found.get((row.owner, row.where, row.what)) == set(row.keys), row
    finally:
        window.close()
        _settle(0.05)
