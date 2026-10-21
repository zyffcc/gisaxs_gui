"""The documents a user reads say what the program does: version, menus, files, where things are.

README.md, CHANGELOG.md and docs/User_Manual.md (what Help ▸ User Manual opens) fell behind the program
once: they named a Tools entry that was gone, an output folder nothing wrote any more, packages nobody
imported and curve files Fitting refuses. Each statement that can be read from the code is checked
against the code here, so the next change of the program fails a test until the text follows.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
CHANGELOG = ROOT / "CHANGELOG.md"
MANUAL = ROOT / "docs" / "User_Manual.md"


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _section(text: str, heading: str) -> str:
    """The text of one ``## heading`` (up to the next heading of the same level)."""
    start = text.index(heading)
    end = text.find("\n## ", start + len(heading))
    return text[start:] if end < 0 else text[start:end]


# -- the version ------------------------------------------------------------------------------------


def test_the_documents_name_the_version_of_the_program() -> None:
    from src.gimap.shared.version import __version__

    assert re.search(rf"^## {re.escape(__version__)} ", _text(CHANGELOG), flags=re.M), \
        "CHANGELOG.md needs an entry for this version"
    assert f"version is {__version__}" in _text(README)
    manual = _text(MANUAL)
    last = _section(manual, "Version and Contact")
    assert f"GIMaP {__version__}" in last and "CHANGELOG.md" in last
    assert f"version {__version__}" in _section(manual, "# GIMaP User Manual")
    assert "CHANGELOG.md" in _text(README)


# -- the manual's structure -------------------------------------------------------------------------


def test_the_manual_numbers_its_sections_once_and_refers_to_sections_that_exist() -> None:
    manual = _text(MANUAL)
    headings = re.findall(r"^## (\d+)\. (.+)$", manual, flags=re.M)
    numbers = [int(number) for number, _title in headings]
    assert numbers == list(range(1, len(numbers) + 1)), numbers
    for first, last in re.findall(r"[Ss]ections? (\d+)(?:[–-](\d+))?", manual):
        assert 1 <= int(first) <= int(last or first) <= len(numbers), (first, last)
    # Subsections are numbered N.1, N.2 … under their chapter.
    subsections = re.findall(r"^### (\d+)\.(\d+) (.+)$", manual, flags=re.M)
    for chapter in set(numbers):
        inside = [int(sub) for number, sub, _title in subsections if int(number) == chapter]
        assert inside == list(range(1, len(inside) + 1)), (chapter, inside)
    # A tool named with its section points at the section of that name (a chapter or a subsection).
    titles = {title.strip(): f"{number}.{sub}" for number, sub, title in subsections}
    titles.update({title.strip(): number for number, title in headings})
    for name in ("Geometry Calibration", "Format Converter", "XRR Series Extractor"):
        section = next(number for title, number in titles.items() if name in title)
        for found in re.findall(rf"{name}\s*\(section (\d+(?:\.\d+)?)\)", manual):
            assert section == found or section.startswith(found + "."), (name, found)
    assert any(title.startswith("Format Converter") for title in titles)


# -- the menus --------------------------------------------------------------------------------------


@pytest.fixture()
def menus():
    from PyQt5.QtWidgets import QApplication, QMainWindow

    from src.gimap.app.presentation.menu_bar import MainMenuBar, MenuCommands
    from src.gimap.app.presentation.navigation import NAVIGATION_ITEMS

    QApplication.instance() or QApplication([])

    def nothing(*_arguments):
        return None

    window = QMainWindow()
    commands = MenuCommands(
        open_files=nothing, open_folder=nothing, open_recent=nothing, clear_recent=nothing,
        save_parameters=nothing, load_parameters=nothing, open_project=nothing, save_project=nothing,
        save_project_as=nothing, quit=nothing, show_workspace=nothing, set_theme=nothing, set_language=nothing,
        change_font_size=nothing, reset_font_size=nothing, set_sidebar_collapsed=nothing,
        toggle_full_screen=nothing, geometry_calibration=nothing, format_converter=nothing,
        convert_current_file=nothing, xrr_extractor=nothing, ai_fitting_workspace=nothing,
        claude_assistant=nothing, settings=nothing, open_data_folder=nothing,
    )
    yield MainMenuBar(window, commands, NAVIGATION_ITEMS)
    window.close()
    window.deleteLater()


def _row(manual: str, menu: str) -> str:
    """The row of the manual's Menus table for one menu."""
    return next(line for line in manual.splitlines() if line.startswith(f"| {menu} |"))


def test_the_manual_lists_the_entries_of_the_real_menus(menus) -> None:
    from src.gimap.app.presentation.navigation import NAVIGATION_ITEMS

    manual = _text(MANUAL)
    workspaces = {item.title for item in NAVIGATION_ITEMS}
    for name, menu in (("File", menus.file_menu), ("View", menus.view_menu), ("Tools", menus.tools_menu),
                       ("Help", menus.help_menu)):
        row = _row(manual, name)
        for action in menu.actions():
            title = action.text().replace("&", "")
            if action.isSeparator() or title in workspaces:
                continue
            assert title in row, f"{name} ▸ {title} is not in the manual's Menus table"
    assert f"`Ctrl+{len(NAVIGATION_ITEMS)}`" in _row(manual, "View")  # one shortcut per page
    table = "\n".join(_row(manual, name) for name in ("File", "View", "Tools", "Help"))
    for name in ("open_folder", "open_project", "save_project_as", "collapse_sidebar", "font_reset",
                 "geometry_calibration", "format_converter", "xrr_extractor", "claude_assistant", "settings"):
        shortcut = menus.actions[name].shortcut().toString()
        assert shortcut and f"`{shortcut}`" in table, (name, shortcut)


def test_the_documents_name_the_tools_entry_of_1d_predict(menus) -> None:
    entry = menus.actions["ai_fitting_workspace"].text().replace("&", "")
    for path in (README, MANUAL, ROOT / "docs" / "ui" / "workspaces" / "fitting.md"):
        assert f"Tools ▸ {entry}" in _text(path), path.name


# -- statements read from the code ------------------------------------------------------------------


def test_the_documents_say_which_curve_files_fitting_reads() -> None:
    from src.gimap.features.fitting.application import CURVE_SUFFIXES

    refused = "Fitting does not read `.csv`"
    said = [refused in _text(MANUAL), refused in _text(CHANGELOG)]
    if ".csv" in CURVE_SUFFIXES:
        assert not any(said), "Fitting reads .csv now: take the sentence out of the manual and the changelog"
    else:
        assert all(said), "Fitting refuses .csv: the manual and the changelog's known limitations must say so"
        assert "`.csv` curves that Analyze exports are not read by Fitting" in _text(README)
    for suffix in CURVE_SUFFIXES:
        assert f"`{suffix}`" in _text(MANUAL)


def test_the_readme_lists_the_requirements() -> None:
    pattern = r"^(?:- )?([A-Za-z0-9][A-Za-z0-9_.\-]*)\s*(?:>=|==)"
    required = {
        match.group(1).lower()
        for line in _text(ROOT / "requirements.txt").splitlines()
        if (match := re.match(pattern, line.strip()))
    }
    listed = set()
    for line in _section(_text(README), "## Dependencies").splitlines():
        if line.startswith("- "):  # the names before any environment marker or remark
            names = re.split(r"[;(]", line[2:], maxsplit=1)[0]
            listed.update(name.lower() for name in re.findall(r"([A-Za-z0-9][A-Za-z0-9_.\-]*)\s*(?:>=|==)", names))
    assert listed == required, (sorted(listed - required), sorted(required - listed))


def test_the_readme_counts_the_pages_of_the_smoke_check() -> None:
    count = int(re.search(r"count\(\) != (\d+)", _text(ROOT / "tools" / "offscreen_smoke.py")).group(1))
    word = {5: "five", 6: "six", 7: "seven", 8: "eight"}[count]
    assert f"{word} pages" in _section(_text(README), "## Development")


def test_where_1d_predict_writes_is_what_the_documents_say() -> None:
    """Every run goes to ``AI_Fitting_Output/v5_<time>`` under the working folder (two call sites)."""
    fitting = ROOT / "src" / "gimap" / "features" / "fitting" / "presentation"
    for source in (fitting / "workflow_v5_dialog.py", fitting / "single" / "run.py"):
        code = _text(source)
        assert '"AI_Fitting_Output"' in code and '"v5_"' in code and "Path.cwd()" in code, source.name
    for path in (README, MANUAL, CHANGELOG):
        assert "AI_Fitting_Output/v5_<date>_<time>/" in _text(path), path.name


def test_the_files_of_a_1d_predict_run_are_the_ones_the_documents_list() -> None:
    """Three methods, three writers: what every one writes for a curve, and what only General V5 writes."""
    adapters = ROOT / "src" / "gimap" / "features" / "fitting" / "infrastructure" / "adapters"
    writers = {name: _text(adapters / f"{name}.py") for name in ("workflow_v5", "experimental_fit", "stable_blue")}
    for_the_run = ("request.json", "batch_summary.json", "top20_candidates.json", "fitting_run_summary.json")
    for_every_curve = ("solutions.json", "display.npz", "fitting_curves.csv")
    for name in for_the_run:
        assert f'"{name}"' in writers["workflow_v5"], name
    for name in for_every_curve:
        assert all(f'"{name}"' in code for code in writers.values()), name
    assert '"candidates.npz"' in writers["workflow_v5"]
    only_general = not any('"candidates.npz"' in writers[name] for name in ("experimental_fit", "stable_blue"))
    for path in (README, MANUAL):
        text = _text(path)
        for name in for_the_run + for_every_curve + ("candidates.npz",):
            assert f"`{name}`" in text, (path.name, name)
        said = re.search(r"General V5\*\s+also\s+writes\s+`candidates\.npz`", text) is not None
        assert said == only_general, f"{path.name}: which method writes candidates.npz changed"


def test_the_documents_say_where_the_manual_opens() -> None:
    """Help ▸ User Manual writes the manual as a web page and hands it to the default browser."""
    code = _text(ROOT / "src" / "gimap" / "app" / "presentation" / "menu_bar.py")
    opener = code[code.index("def open_user_manual"):code.index("def show_about")]
    in_browser = "QDesktopServices.openUrl" in opener
    for path in (README, CHANGELOG):
        text = _text(path)
        assert ("manual in your browser" in text) == in_browser, path.name
        assert "opens inside the program" not in text, path.name


def test_the_links_of_the_release_documents_resolve() -> None:
    """Relative links (and the screenshots of both manuals) point at files that exist; ``#section`` links at a
    heading of the same document."""
    for path in (README, CHANGELOG, MANUAL, MANUAL.with_name("User_Manual_zh.md")):
        text = _text(path)
        anchors = {
            re.sub(r"[^\w\- ]", "", heading.strip().lower()).replace(" ", "-")
            for heading in re.findall(r"^#+\s+(.+)$", text, flags=re.M)
        }
        for target in re.findall(r"\[[^\]]*\]\(([^)\s]+)\)", text):
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            if target.startswith("#"):
                assert target[1:] in anchors, (path.name, target)
            else:
                assert (path.parent / target.split("#")[0]).exists(), (path.name, target)


def test_statements_about_removed_things_are_gone() -> None:
    stale = ("Fit Settings & Batch", "Fit settings & batch", "AI guess only", "Global Search / Local Refine",
             "current_prediction", "setup.exe", "environment.yml", "five-workspace", "umap-learn", "scikit-learn",
             "GIMaP Predict Page", "Predict-2D Tab", "Analyze ▸ Options")
    for path in (README, MANUAL):
        text = _text(path)
        assert not [phrase for phrase in stale if phrase in text], path.name
    manual = _text(MANUAL)
    assert "### 8.1 Format Converter" in manual and "Remove Source from List" in manual and "Copy Table" in manual


V5 = "modules/Fitting_1D_Model/Workflow_v5"
V5_MANIFESTS = (f"{V5}/conditional_v2/conditional_fast_manifest_v2.json", f"{V5}/stable_blue_rc_v1/MANIFEST.json")
WEIGHTS_SAID_MISSING = (
    "not in the Git repository", "not of the Git repository", "not tracked by Git", "weights are missing",
    "until the weight files", "until they are placed", "weight files are added",
)
"""How the documents said, until the weights were committed, that a checkout cannot run 1D Predict."""


def _model_weights() -> list[str]:
    """The weight files (``*.h5``, ``*.npz``) the manifests of the bundled 1D Predict model name."""
    weights = []
    for manifest in V5_MANIFESTS:
        folder = manifest.rsplit("/", 1)[0]
        files = json.loads(_text(ROOT / manifest))["files"]
        weights.extend(f"{folder}/{name}" for name in files if name.endswith((".h5", ".npz")))
    return weights


def test_the_documents_do_not_say_the_model_weights_are_missing() -> None:
    """The weights of 1D Predict are in the repository: no document may send a user looking for them."""
    documents = (README, CHANGELOG, MANUAL, ROOT / "docs" / "development.md")
    for path in documents:
        text = " ".join(_text(path).split())
        assert not [phrase for phrase in WEIGHTS_SAID_MISSING if phrase in text], path.name
        # One sentence that names the weights and says Git or the repository does not have them.
        for sentence in re.split(r"(?<=[.:;])\s", text):
            if re.search(r"\bweights?\b", sentence):
                assert not re.search(r"\bnot\b[^.]*\b(?:in|of|part of|tracked by)\b[^.]*\b(?:Git|repository)\b",
                                     sentence), (path.name, sentence)
    for path in (README, MANUAL):
        assert "weights of 1D Predict are in the Git repository" in " ".join(_text(path).split()), path.name
    assert "1D Predict also runs from a source checkout" in " ".join(_text(CHANGELOG).split())
    assert "### Known limitations" in _text(CHANGELOG)
    limitations = _text(CHANGELOG).split("### Known limitations", 1)[1].split("\n### ", 1)[0]
    assert "weights" not in limitations, "the weights are no known limitation any more"


def test_git_tracks_the_model_weights_the_documents_promise() -> None:
    """README, the manual and the changelog say a source checkout runs 1D Predict: true while git tracks
    every weight file the model's manifests name."""
    weights = _model_weights()
    assert len(weights) >= 8, weights
    if not shutil.which("git") or not (ROOT / ".git").exists():
        pytest.skip("not a git checkout (or git is not installed)")
    tracked = set(subprocess.run(
        ["git", "ls-files", "-z", "--", V5], cwd=ROOT, capture_output=True, check=True, timeout=120,
    ).stdout.decode("utf-8").split("\0"))
    missing = [name for name in weights if name not in tracked]
    assert not missing, (
        "git does not track these weight files, so a clone cannot run 1D Predict although the documents say "
        "it can (.gitignore ignores *.h5 and *.npz: add them with `git add -f`): " + ", ".join(missing)
    )
