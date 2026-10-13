"""The documents follow the decisions taken for 0.1.0 on 2026-10-13.

- A source checkout is updated with ``tools/update_source.ps1``; the history of ``main`` was rewritten.
- Geometry Calibration, the XRR Series Extractor and Analyze show the same beam centre; the files the two
  tools export keep pixel-centre indices and say so.
- Auto Detect of the calibration standard can put the wrong standard first, and a wrong estimated distance
  misleads the search: a known limitation, documented now and corrected later. The test that reproduces it
  fails when it is corrected, so the texts follow. The window no longer fills the estimate with a built-in
  distance: the rule it follows is checked here on the window itself.
- A case built on private data is not in the repository, and no document names one.
- Fitting starts a new user with a neutral model and stores one block of values per shape; 0.1.0 is still a
  pre-release.

As in ``test_release_docs_contents.py``, a statement that can be read from the code is checked against it.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
CHANGELOG = ROOT / "CHANGELOG.md"
MANUAL = ROOT / "docs" / "User_Manual.md"
AGENTS = ROOT / "AGENTS.md"
DEVELOPMENT = ROOT / "docs" / "development.md"
PLAYBOOK = ROOT / "docs" / "agents" / "giwaxs-playbook.md"
CALIBRATION_PAGE = ROOT / "docs" / "ui" / "workspaces" / "calibration.md"
XRR_PAGE = ROOT / "docs" / "ui" / "workspaces" / "xrr.md"
FITTING_PAGE = ROOT / "docs" / "ui" / "workspaces" / "fitting.md"
UPDATE_SCRIPT = ROOT / "tools" / "update_source.ps1"
PUBLIC_AGBH = ROOT / "tests" / "data" / "external" / "saxs_agbh_pyfai" / "Pilatus1M.edf"


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _flat(path: Path) -> str:
    """The text with every run of white space as one blank: sentences wrap differently in each document."""
    return " ".join(_text(path).split())


def _part(text: str, heading: str, level: str = "## ") -> str:
    """One section: from ``heading`` to the next heading of ``level``."""
    start = text.index(heading)
    end = text.find(f"\n{level}", start + len(heading))
    return text[start:] if end < 0 else text[start:end]


# -- updating a source checkout -----------------------------------------------------------------------


def test_the_documents_say_how_to_update_a_source_checkout() -> None:
    command = r"powershell -ExecutionPolicy Bypass -File tools\update_source.ps1"
    for path in (README, MANUAL, CHANGELOG):
        text = _flat(path)
        assert command in text, path.name
        for plain in ("git fetch origin", "git reset --hard origin/main"):
            assert plain in text, (path.name, plain)
        # The warning stands with the command that loses work.
        after_reset = text[text.index("git reset --hard origin/main"):][:1300]
        assert "discards your edits of tracked files" in after_reset, path.name
        assert "git stash" in after_reset and "git branch" in after_reset, path.name
        # ... and so does what the two commands do not do, and the one kind of untracked file they replace.
        assert re.search(r"[Rr]un (?:the script|it) once", after_reset), path.name
        assert "line endings" in after_reset and "does not start until the script has written them again" in after_reset
        assert re.search(r"where the new version has (?:a file|one)", after_reset), path.name
        assert re.search(r"[Dd]o not merge or push (?:`my-work`|that branch)", after_reset), path.name
        assert re.search(r"history of `main`[^.]*rewritten on 2026-10-13", text), path.name
        assert "Do not merge" in text, path.name
    for path in (README, MANUAL):
        text = _flat(path)
        # The first update that keeps everything: the script alone is taken out of origin, then it runs.
        assert "`git checkout origin/main -- tools/update_source.ps1`" in text, path.name
        # What a checkout of 0.0.2 holds as an edit: 0.0.2 saved the model of Fitting into this tracked file.
        assert "`config/model_parameters.json`, where 0.0.2 kept the model of Fitting" in text, path.name
        assert "`git pull` works again after the two commands" in text, path.name
    assert "To update a source checkout: `git pull`" not in _text(README)
    assert command in _flat(DEVELOPMENT)
    rules = _text(AGENTS)  # an agent on an old clone must not pull, merge or push either
    assert "tools/update_source.ps1" in rules and "2026-10-13" in rules and "不要 `git pull`" in rules
    updating = _part(_text(CHANGELOG), "### Updating from 0.0.2", "### ")
    assert "update_source.ps1" in updating and "git reset --hard origin/main" in updating


def test_the_update_script_does_what_the_documents_say() -> None:
    assert UPDATE_SCRIPT.is_file(), "README.md, the manual and CHANGELOG.md name tools/update_source.ps1"
    script = UPDATE_SCRIPT.read_text(encoding="utf-8", errors="replace")
    # What README.md and the manual say of it, each read from the script.
    said = {
        "updates from origin": "fetch origin" in script,
        "moves forward when origin only added commits": "--ff-only" in script,
        "keeps the state before as a backup branch": '"backup/$Stamp' in script,
        "moves the branch to the new history": '"reset", "--hard"' in script,
        "puts modifications aside and back": '"stash", "push"' in script and '"stash", "apply"' in script,
        "stops before it overwrites a file of yours": "different files of your own" in script,
        "reports only with -DryRun": "[switch]$DryRun" in script,
        "says when the packages changed": "python -m pip install -r requirements.txt" in script,
        "prints how to go back": "To go back to what you had" in script,
        "prints how to delete the backup": "Once things look right, delete what was kept" in script,
        # ``git ls-files --eol`` finds the files a checkout holds converted although .gitattributes says -text.
        "writes converted model files again": '"--eol"' in script and "attr/-text" in script,
        "also when there is nothing to update": "only line endings were corrected" in script,
        "needs Git 2.16 or newer and says so": "it needs 2.16 or newer" in script and "-lt 216" in script,
        "puts the version you had back when a file is held open": (
            "Nothing is left half way: the files git had already replaced were written again as they were." in script
            and "Your modifications are in the files, as before." in script
            and "Close the programs that use the folder and run this script again:" in script),
        # An edit that does not fit: the new version whole, the edit in the stash, exit code 2 ...
        "ends with exit code 2 when an edit does not fit": (
            "your modifications do NOT fit the new version: they were left in the stash" in script
            and "if ($NeedsYou.Count -gt 0) { exit 2 }" in script),
        # ... and the command the documents warn of for config/model_parameters.json.
        "names git stash pop for edits somebody wants back": bool(
            re.search(r"git stash pop\s+tries again and marks the conflicts in the files", script)),
    }
    assert all(said.values()), [what for what, true in said.items() if not true]
    for path in (README, MANUAL):
        text = _flat(path)
        for words in ("`backup/<date>-<time>-<commit>`", "`git stash`", "`-DryRun`", "stops before it changes anything",
                      "how to go back to what you had", "how to delete the backup once things look right",
                      "converted line endings are written again", "also when there is nothing new to download",
                      "needs Git 2.16 or newer and says so", "another program has it open",
                      "the version you had is put back whole", "ends with exit code 2",
                      "delete that copy with `git stash drop`", "it would write conflict markers into the file",
                      "`git checkout HEAD -- config/model_parameters.json`"):
            assert words in text, (path.name, words)
    # The model folder is the one .gitattributes hands out byte for byte.
    assert "modules/Fitting_1D_Model/Workflow_v5/** -text" in _text(ROOT / ".gitattributes")
    development = _flat(DEVELOPMENT)
    assert "1: not updated, or not completely" in development and "After the two plain commands, run the script once" in development
    for words in ("needs Git 2.16 or newer", "another program holds it open", "`git stash drop` when the stash holds nothing else",
                  "writes conflict markers into the template"):
        assert words in development, words


class _Repositories:
    """Git on repositories made in a temporary folder, with a configuration of its own (never this checkout)."""

    def __init__(self, folder: Path) -> None:
        config = folder / "gitconfig"
        config.write_text("[init]\n\tdefaultBranch = main\n[core]\n\tautocrlf = false\n[user]\n\tname = Somebody\n"
                          "\temail = somebody@example.org\n", "ascii")
        self.env = {key: value for key, value in os.environ.items() if not key.upper().startswith("GIT_")}
        self.env.update(GIT_CONFIG_GLOBAL=str(config), GIT_CONFIG_NOSYSTEM="1", GIT_TERMINAL_PROMPT="0")

    def run(self, folder: Path, *arguments: str):
        import subprocess

        return subprocess.run(["git", *arguments], cwd=folder, env=self.env, capture_output=True, timeout=120,
                              encoding="utf-8", errors="replace")

    def git(self, folder: Path, *arguments: str) -> str:
        done = self.run(folder, *arguments)
        assert done.returncode == 0, f"git {' '.join(arguments)}\n{done.stdout}\n{done.stderr}"
        return done.stdout.strip()

    @staticmethod
    def write(folder: Path, files: dict[str, str]) -> None:
        for name, text in files.items():
            (folder / name).parent.mkdir(parents=True, exist_ok=True)
            (folder / name).write_bytes(text.encode("utf-8"))

    def commit(self, folder: Path, message: str, files: dict[str, str]) -> None:
        self.write(folder, files)
        self.git(folder, "add", "--force", "--", *files)
        self.git(folder, "commit", "--quiet", "--message", message)


def test_the_two_plain_commands_do_what_the_documents_say(tmp_path: Path) -> None:
    """What README.md, the manual and CHANGELOG.md say of ``git fetch origin`` and
    ``git reset --hard origin/main`` on a clone of the old history, on repositories made here.

    Edits of tracked files and own commits leave the branch (a stash and a branch keep them, ``git stash
    pop`` and ``git cherry-pick`` bring them back); files git does not track stay, except one at a place
    where the new version has a file: it is replaced without a question; afterwards ``git pull`` works again.
    """
    if not shutil.which("git"):
        pytest.skip("git is not installed")
    repositories = _Repositories(tmp_path)
    git, write, commit = repositories.git, repositories.write, repositories.commit

    origin, author, other = tmp_path / "origin.git", tmp_path / "author", tmp_path / "other"
    git(tmp_path, "init", "--quiet", "--bare", str(origin))
    git(tmp_path, "clone", "--quiet", str(origin), str(author))
    commit(author, "old", {".gitignore": "*.h5\n", "src/app.py": "one\n", "src/model.py": "a\nb\nc\n"})
    git(author, "push", "--quiet", "origin", "HEAD:main")
    git(tmp_path, "clone", "--quiet", str(origin), str(other))
    # The history is rewritten: a new first commit, which also brings the weights.
    git(author, "checkout", "--quiet", "--orphan", "rewritten")
    commit(author, "new", {"src/app.py": "two\n", "models/w.h5": "the weights of the repository\n"})
    git(author, "push", "--quiet", "--force", "origin", "HEAD:main")

    write(other, {"src/model.py": "mine\nb\nc\n"})  # an edit of a tracked file
    commit(other, "my own commit", {"src/mine.py": "my own\n"})
    write(other, {"notes.txt": "my notes\n", "models/w.h5": "weights copied in by hand\n"})
    git(other, "stash")  # "run `git stash` (edits) or `git branch my-work` (commits) first"
    git(other, "branch", "my-work")

    git(other, "fetch", "--quiet", "origin")
    git(other, "reset", "--quiet", "--hard", "origin/main")

    assert (other / "src" / "app.py").read_text() == "two\n" and (other / "src" / "model.py").read_text() == "a\nb\nc\n"
    assert not (other / "src" / "mine.py").exists()  # the own commit is no longer on the branch
    assert (other / "notes.txt").read_text() == "my notes\n"  # "Files Git does not track stay"
    # "... with one exception: where the new version has a file at the same place".
    assert (other / "models" / "w.h5").read_text() == "the weights of the repository\n"
    git(other, "stash", "pop")  # "afterwards `git stash pop` puts the edits back"
    assert (other / "src" / "model.py").read_text() == "mine\nb\nc\n"
    git(other, "cherry-pick", "my-work")  # "`git cherry-pick <commit>` brings a commit over"
    assert (other / "src" / "mine.py").read_text() == "my own\n"

    # "`git pull` works again after the two commands": a later, ordinary update of a checkout without commits.
    third = tmp_path / "third"
    git(tmp_path, "clone", "--quiet", str(origin), str(third))
    commit(author, "newer", {"src/app.py": "three\n"})
    git(author, "push", "--quiet", "origin", "HEAD:main")
    git(third, "pull", "--quiet")
    assert (third / "src" / "app.py").read_text() == "three\n"


def test_the_model_file_a_checkout_of_0_0_2_edited_is_dropped_as_the_documents_say(tmp_path: Path) -> None:
    """ "delete that copy with ``git stash drop``. Do not run ``git stash pop`` for it … it would write conflict
    markers into the file": ``config/model_parameters.json`` as an older version tracked it, with the numbers
    a fit left in it (all invented), against the template this version has, on repositories made here.

    The script leaves such a checkout as the plain commands below do: the new version whole, the edit in the
    stash (``tests/test_release2_updater_scenarios.py`` runs the script itself on an edit that does not fit).
    """
    if not shutil.which("git"):
        pytest.skip("git is not installed")
    name = "config/model_parameters.json"
    template = _text(ROOT / name)  # the starting model of a new user

    def other_numbers(value, factor: float):
        if isinstance(value, dict):
            return {key: other_numbers(item, factor) for key, item in value.items()}
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return round(value * factor + factor, 4)
        return value

    tracked_before = json.dumps(other_numbers(json.loads(template), 1.7), indent=4)
    after_a_fit = json.dumps(other_numbers(json.loads(template), 2.3), indent=4)  # as 0.0.2 saved it: indent 4
    repositories = _Repositories(tmp_path)
    git, write, commit = repositories.git, repositories.write, repositories.commit
    origin, author, other = tmp_path / "origin.git", tmp_path / "author", tmp_path / "other"
    git(tmp_path, "init", "--quiet", "--bare", str(origin))
    git(tmp_path, "clone", "--quiet", str(origin), str(author))
    commit(author, "old", {name: tracked_before, "src/app.py": "a\nb\nc\n"})
    git(author, "push", "--quiet", "origin", "HEAD:main")
    git(tmp_path, "clone", "--quiet", str(origin), str(other))
    git(author, "checkout", "--quiet", "--orphan", "rewritten")
    commit(author, "new", {name: template, "src/app.py": "a\nb\nc\nd\n"})
    git(author, "push", "--quiet", "--force", "origin", "HEAD:main")

    def stashes() -> int:
        return len(git(other, "stash", "list").splitlines())

    write(other, {name: after_a_fit, "src/app.py": "mine\nb\nc\n"})  # the fits of 0.0.2, and an edit of one's own
    git(other, "stash")
    git(other, "fetch", "--quiet", "origin")
    git(other, "reset", "--quiet", "--hard", "origin/main")
    # "an edit that does not fit the new version": the script tries this, then leaves the new version whole.
    assert repositories.run(other, "stash", "apply").returncode != 0
    git(other, "reset", "--quiet", "--hard")
    # "the new version is complete, and what you had changed waits in the stash": nothing is left to repair, so
    # with this file as the only edit "delete that copy with `git stash drop`" is all there is to do.
    assert (other / name).read_text("utf-8") == template and stashes() == 1
    assert git(other, "status", "--porcelain") == ""

    # "it would write conflict markers into the file, which is the model a new user of Fitting starts with".
    assert repositories.run(other, "stash", "pop").returncode != 0
    marked = (other / name).read_text("utf-8")
    assert "<<<<<<<" in marked and ">>>>>>>" in marked
    with pytest.raises(ValueError):
        json.loads(marked)
    assert stashes() == 1  # a pop that ends in a conflict keeps the stash
    # "With other edits in the same stash: `git stash pop`, then `git checkout HEAD -- config/model_parameters.json`
    # to put the file of the new version back, then `git stash drop`."
    git(other, "checkout", "HEAD", "--", name)
    git(other, "stash", "drop")
    assert (other / name).read_text("utf-8") == template and stashes() == 0
    assert (other / "src" / "app.py").read_text("utf-8") == "mine\nb\nc\nd\n"  # the own edit, on the new version
    assert git(other, "status", "--porcelain").split() == ["M", "src/app.py"]
    # The file is the template the store copies for a new user (what the conflict markers would have broken).
    from src.gimap.features.fitting.infrastructure.adapters.model_parameters import TEMPLATE_PATH

    assert Path(TEMPLATE_PATH) == ROOT / name


# -- the beam centre of Geometry Calibration ----------------------------------------------------------


def test_the_documents_say_calibration_and_analyze_show_the_same_beam_centre() -> None:
    from src.gimap.shared.geometry.conventions import canonical_from_index_center

    assert canonical_from_index_center(0.0, 0.0) == (0.5, 0.5)  # "the centre of the first pixel is (0.5, 0.5)"

    changelog = _text(CHANGELOG)
    limitations = " ".join(_part(changelog, "### Known limitations", "### ").split())
    for stale in ("Geometry Calibration shows the beam centre half a pixel lower", "add 0.5 px when you copy"):
        assert stale not in limitations, stale
    tools = " ".join(_part(changelog, "### Tools", "### ").split())
    assert "same pixel coordinates as Analyze" in tools and "first pixel is at 0.5" in tools

    section = " ".join(_part(_text(MANUAL), "## 9. Geometry Calibration Tool").split())
    for stale in ("in this window counts pixel centres from 0", "needs 0.5 px added", "Analyze counts from 0.5"):
        assert stale not in section, stale
    assert "same pixel coordinates as in Analyze" in section and "(0.5, 0.5)" in section
    assert "`metadata.center_convention`" in section  # the exported file keeps its own numbers, and says so

    page = _text(CALIBRATION_PAGE)
    assert "shared/geometry/conventions.py" in page and "和 Analyze 相同" in page


def test_the_calibration_window_shows_what_the_documents_say() -> None:
    """The centre on screen is the canonical one; the fit and the exported file keep pixel-centre indices."""
    from types import SimpleNamespace

    from src.gimap.features.calibration.domain.models import CENTER_CONVENTION_KEY, INDEX_CENTER_CONVENTION
    from src.gimap.features.calibration.presentation.pixel_coordinates import fitted_center, shown_center

    candidate = SimpleNamespace(center_x_px=180.0, center_y_px=263.5)
    assert shown_center(candidate) == (180.5, 264.0)
    assert fitted_center(180.5, 264.0) == (180.0, 263.5)
    assert INDEX_CENTER_CONVENTION["name"] == "index"  # what calibration.md says of the exported .json
    assert CENTER_CONVENTION_KEY == "center_convention"
    assert "metadata.center_convention" in _text(CALIBRATION_PAGE)


def test_the_pages_of_the_changed_windows_name_tests_that_exist() -> None:
    """The developer pages of Geometry Calibration, the XRR window and Fitting point at their regression tests."""
    expected = {
        CALIBRATION_PAGE: ("tests/test_release2_calibration_window.py", "tests/test_release2_calibration_edges.py",
                           "tests/test_release2_calibration_files.py", "tests/test_release2_calibration_estimate.py"),
        XRR_PAGE: ("tests/test_release2_xrr_window.py",),
        FITTING_PAGE: ("tests/test_release2_defaults_template.py", "tests/test_release2_fitting_cylinder_key.py"),
    }
    for page, names in expected.items():
        text = _text(page)
        for name in names:
            assert name in text, (page.name, name)
        for name in sorted(set(re.findall(r"tests/test_\w+\.py", text))):
            assert (ROOT / name).is_file(), (page.name, name)


def test_the_documents_say_the_xrr_window_shows_the_pixels_of_analyze() -> None:
    changelog = _text(CHANGELOG)
    limitations = " ".join(_part(changelog, "### Known limitations", "### ").split())
    for stale in ("still counts the direct-beam centre from pixel centres at 0", "subtract 0.5 px"):
        assert stale not in limitations, "the XRR window's pixels are no known limitation any more"
    tools = " ".join(_part(changelog, "### Tools", "### ").split())
    assert "pixel coordinates of Analyze and Geometry Calibration" in tools and "(`center_convention`)" in tools
    fixed = " ".join(_part(changelog, "### Fixed before the release", "### ").split())
    for said in ("XRR Series Extractor showed and took the direct-beam centre half a pixel lower",
                 "is typed 0.5 px larger in x and y now", "is now the point clicked",
                 "keep their numbers (pixel centres counted from 0)"):
        assert said in fixed, said

    section = " ".join(_part(_text(MANUAL), "## 11. XRR Series Extractor").split())
    for stale in ("still counts pixel centres from 0", "subtract 0.5 px", "half a pixel to the right of and below the point"):
        assert stale not in section, stale
    for said in ("same pixel coordinates as in Analyze and Geometry Calibration", "(0.5, 0.5)",
                 "A centre can be copied between the three windows as it is",
                 "the pixel you click is the pixel that is read", "(columns / 2, rows / 2)",
                 "`roi_center_x_px` / `roi_center_y_px`", "`geometry.beam_center_x_px` / `beam_center_y_px`",
                 "0.5 px lower than the window shows", "(`center_convention`)", "add 0.5 px to x and y"):
        assert said in section, said

    page = _text(XRR_PAGE)
    assert "已知不一致" not in page and "presentation/pixel_coordinates.py" in page
    assert "与 Analyze 和 Geometry Calibration 相同" in page and "`center_convention`" in page
    rules = _text(AGENTS)
    assert "还没有换" not in rules and "`xrr/presentation/pixel_coordinates.py`" in rules
    assert "Analyze、Geometry Calibration 和 XRR Series Extractor 显示和输入的光束中心都用这套坐标" in rules
    contract = _flat(ROOT / "docs" / "architecture" / "xrr-series-workflow.md")
    assert "are pixel-centre indices" in contract and "`center_convention`" in contract


def test_the_xrr_window_shows_what_the_documents_say() -> None:
    """The window is in the pixels of Analyze; the extraction and the exported files keep pixel-centre indices.

    A one-pixel ROI at index 2.6 reads column 3 (2.6 is nearest to the centre of pixel 3): that is the
    extraction, unchanged. In the window 2.6 lies inside pixel 2, and that is the pixel a centre typed or
    picked there reads.
    """
    from types import SimpleNamespace

    import numpy as np

    from src.gimap.features.xrr.application.export_record import (
        CENTER_CONVENTION_KEY,
        INDEX_CENTER_CONVENTION,
        xrr_export_record,
    )
    from src.gimap.features.xrr.domain.reflectometry import extract_circular_roi
    from src.gimap.features.xrr.presentation import pixel_coordinates
    from src.gimap.shared.geometry.conventions import canonical_from_index_center

    image = np.arange(6, dtype=float)[np.newaxis, :].repeat(3, axis=0)  # the value of a pixel is its column
    assert extract_circular_roi(image, None, 2.6, 1.0, 0, "sum").intensity == 3.0  # the extraction: indices

    # "the pixel you click is the pixel that is read": wherever in pixel 2 the window's position lies.
    for shown in (2.0, 2.2, 2.5, 2.6, 2.99):
        index = pixel_coordinates.extraction_coordinate(shown, 2)
        assert index == pytest.approx(shown - 0.5)
        assert extract_circular_roi(image, None, index, 1.0, 0, "sum").intensity == 2.0, shown
    # What the window shows of the extraction's numbers: 0.5 px more, by the shared conventions.
    assert pixel_coordinates.shown_coordinate(2.0) == 2.5 == canonical_from_index_center(2.0, 2.0)[0]
    point = SimpleNamespace(roi_center_x_px=182.47, roi_center_y_px=264.34)
    assert pixel_coordinates.shown_roi_center(point) == pytest.approx((182.97, 264.84))  # "0.5 px lower than the window shows"
    # "the middle of the frame, (columns / 2, rows / 2)", for a frame of 1043 rows and 981 columns.
    assert pixel_coordinates.shown_image_center((1043, 981)) == (490.5, 521.5)

    # Every way from the screen to the extraction and back goes through that module.
    presentation = ROOT / "src" / "gimap" / "features" / "xrr" / "presentation"
    dialog = _text(presentation / "dialog.py")
    assert "beam_center_x_px=extraction_coordinate(center_x.value(), center_x.decimals())" in dialog
    assert "self.center_x_spin.setValue(float(event.xdata))" in dialog  # Pick: the axes are the pixels of the fields
    assert "extent=image_extent(values.shape, full_shape)" in _text(presentation / "plotting.py")
    assert pixel_coordinates.image_extent((3, 6), (3, 6)) == (0.0, 6.0, 3.0, 0.0)  # pixel j is drawn over j…j+1
    # "hover over the centre fields or the ROI x / ROI y headers to read this in the window".
    assert "(the centre of the first pixel is 0.5)" in pixel_coordinates.PIXEL_NOTE
    assert "its roi_center values are 0.5 px lower" in pixel_coordinates.ROI_PIXEL_NOTE
    assert "spin.setToolTip(PIXEL_NOTE)" in _text(presentation / "views" / "xrr_series_dialog_view.py")
    assert "item.setToolTip(tr(ROI_PIXEL_NOTE))" in dialog

    # "the record says so (`center_convention`)": the record of an export, and the columns of its CSV.
    request = SimpleNamespace(result=SimpleNamespace(points=()), settings=None, geometry_sources={}, calibration=None)
    record = xrr_export_record(request, "series_xrr.csv", created="2026-10-13T00:00:00+00:00")
    assert CENTER_CONVENTION_KEY == "center_convention" and record["center_convention"] == INDEX_CENTER_CONVENTION
    assert INDEX_CENTER_CONVENTION["name"] == "index"  # the name an exported calibration uses for the same convention
    for column in ("roi_center_x_px", "roi_center_y_px"):
        assert "pixel centres counted from 0" in record["columns"][column]


def test_the_nexus_header_limitation_of_the_xrr_window_is_still_true(tmp_path: Path) -> None:
    """A made-up NeXus file whose header points at one marked pixel, read by the loader and by the window.

    The loader turns the stored image (transpose and flip) and turns the header centre with it for Analyze:
    that centre lies on the marked pixel. The XRR window takes the two header numbers as column and row of
    the turned frame, marks them "from file" and lets them replace the centre of the last calibration: a
    one-pixel ROI there reads another pixel. When the window turns the header centre too, this test fails:
    remove the limitation from CHANGELOG.md, docs/User_Manual.md (section 11) and docs/ui/workspaces/xrr.md.
    """
    from types import SimpleNamespace

    import h5py
    import numpy as np
    from PyQt5.QtCore import QCoreApplication, QEvent
    from PyQt5.QtWidgets import QApplication

    from src.gimap.app import AppContext
    from src.gimap.features.xrr.application import InspectXrrSeries, XrrSeriesSpec
    from src.gimap.features.xrr.domain.reflectometry import extract_circular_roi
    from src.gimap.features.xrr.infrastructure.adapters.detector_series import LocalXrrSeriesRepository
    from src.gimap.features.xrr.presentation.dialog import XrrSeriesDialog
    from src.gimap.integrations.state import (
        InMemoryInstrumentProfileRepository,
        InMemorySessionRepository,
        InMemorySettingsRepository,
        InMemoryUserPreferencesRepository,
    )
    from src.gimap.shared.detector_io import load_detector_image

    limitations = " ".join(_part(_text(CHANGELOG), "### Known limitations", "### ").split())
    section = " ".join(_part(_text(MANUAL), "## 11. XRR Series Extractor").split())
    for text in (limitations, section):
        assert "without the turn (transpose and flip) the frame gets when it is loaded" in text
        assert "*from file*" in text and "replaces the centre of the last calibration" in text
        assert "check on the preview that the cross lies on the direct beam" in text.lower()
        assert "**Pick direct-beam center**" in text and "CBF series are not affected" in text
    assert "已知限制：NeXus 头里的光束中心" in _text(XRR_PAGE)

    marked = 1000.0
    stored = np.zeros((40, 60), dtype=np.float32)  # the fast axis of a NeXus image is its last one
    along_fast, along_slow = 12.0, 31.0
    stored[int(along_slow), int(along_fast)] = marked
    path = tmp_path / "made_up_scan.nxs"
    with h5py.File(path, "w") as handle:
        detector = handle.create_group("entry/instrument/detector")
        detector.create_dataset("data", data=stored[np.newaxis])
        detector.create_dataset("beam_center_x", data=along_fast)
        detector.create_dataset("beam_center_y", data=along_slow)

    # Analyze: the header centre, turned as the frame is turned, lies on the marked pixel (at its corner).
    loaded = load_detector_image(path)
    frame = np.asarray(loaded.data)
    row, column = (int(index) for index in np.unravel_index(int(np.nanargmax(frame)), frame.shape))
    assert frame.shape == (60, 40) and frame[row, column] == marked
    header_x, header_y = loaded.metadata["header_beam_center"]
    assert abs(header_x - (column + 0.5)) <= 0.5 and abs(header_y - (row + 0.5)) <= 0.5

    # The XRR window, opened after a calibration of frames of this size was applied (made-up values).
    applied = {
        "detector": {"distance": 1500.0, "pixel_size_x": 172.0, "pixel_size_y": 172.0,
                     "beam_center_x": 20.0, "beam_center_y": 25.0},
        "beam": {"energy_kev": 12.0},
        "system": {"geometry_calibration": {"source_image": "standard.cbf", "timestamp": "2026-10-13T08:00:00+00:00",
                                            "image_shape": [60, 40], "detector": "made up"}},
    }
    QApplication.instance() or QApplication([])
    context = AppContext(
        settings=InMemorySettingsRepository(applied), session=InMemorySessionRepository(),
        preferences=InMemoryUserPreferencesRepository(), instrument_profiles=InMemoryInstrumentProfileRepository(),
        jobs=SimpleNamespace(run=None, cancel=lambda _job: False))
    window = XrrSeriesDialog(app_context=context)
    try:
        assert (window.center_x_spin.value(), window.center_y_spin.value()) == (20.5, 25.5)  # the calibration, as shown
        assert window.geometry_source_labels["center_x"].text() == "(from last calibration)"
        window._on_inspected(InspectXrrSeries(LocalXrrSeriesRepository()).execute(XrrSeriesSpec(source_path=path)))
        # "marked *from file*, and it replaces the centre of the last calibration"
        assert window.geometry_source_labels["center_x"].text() == "from file"
        assert window.geometry_source_labels["center_y"].text() == "from file"
        # "the two numbers of the header as column and row of the frame it shows" (shown 0.5 px higher)
        shown = (window.center_x_spin.value(), window.center_y_spin.value())
        assert shown == (along_fast + 0.5, along_slow + 0.5), (
            "the XRR window no longer takes a NeXus header centre as column and row of the frame it shows: "
            "remove the limitation from CHANGELOG.md, docs/User_Manual.md (section 11) and "
            "docs/ui/workspaces/xrr.md"
        )
        # "so the centre can lie away from the direct beam": a one-pixel ROI there reads another pixel, far away.
        geometry = window._geometry()
        read = extract_circular_roi(frame, None, geometry.beam_center_x_px, geometry.beam_center_y_px, 0, "sum")
        assert read.valid_pixels == 1 and read.intensity != marked
        assert abs(geometry.beam_center_x_px - column) > 10 and abs(geometry.beam_center_y_px - row) > 10
    finally:
        for name in ("_inspect_thread", "_extract_thread"):
            thread = getattr(window, name, None)
            if thread is not None:
                thread.wait(10000)
        window.close()
        window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)


# -- Auto Detect of the standard: a known limitation --------------------------------------------------


def test_the_auto_detect_limitation_is_in_the_documents() -> None:
    limitations = " ".join(_part(_text(CHANGELOG), "### Known limitations", "### ").split())
    section = " ".join(_part(_text(MANUAL), "## 9. Geometry Calibration Tool").split())
    for text in (limitations, section):
        assert "Auto Detect" in text
        assert "Choose the standard yourself when you know it" in text
        assert "check that the distance is the one of your set-up" in text
        assert "silver behenate" in text and "LaB₆ solution at a much shorter distance came first" in text
        assert "corrected in a later version" in text
        # The second way to a wrong solution: a wrong estimate, which somebody typed or the window took over.
        assert "**Estimated distance**" in text and "2000 mm typed in" in text
        assert "a solution near 1830 mm came first" in text and "about 1630 mm" in text
        assert "approximate distance of your set-up" in text and "*Optional*" in text
        assert "the distance the image file states" in text and "the distance of the calibration applied last" in text
        # No longer true: the window does not put the built-in 2000 mm into the field.
        for stale in ("or 2000 mm on a new installation", "2000 mm left in the field", "with 2000 mm in it"):
            assert stale not in text, stale
    assert "set a number you did not type back to 0" in limitations
    assert "set the field to 0 (it then shows *Optional*)" in section
    assert "with 1500 to 1800 mm typed in, the silver behenate solution came first" in section
    page = _text(CALIBRATION_PAGE)
    assert "已知限制：Auto Detect" in page and "STANDARD_SOURCE_ALIASES" in page
    for stale in ("Estimated distance 不是空的", "新安装时是内置默认值", "这一版同样只写进了文档"):
        assert stale not in page, stale
    for said in ("估计距离错了会带偏结果", "`_prefill_estimated_distance`", "`system.geometry_calibration`",
                 "default_settings.py", "tests/test_release2_calibration_estimate.py", "内置的 2000 mm 不再出现在框里"):
        assert said in page, said
    # The names the documents use are the ones on the window.
    presentation = ROOT / "src" / "gimap" / "features" / "calibration" / "presentation"
    assert 'tr("Auto Detect"), "auto"' in _text(presentation / "bindings" / "form_setup.py")
    texts = _text(presentation / "views" / "geometry_calibration_translations.py")
    assert '"Estimated distance:"' in texts and '"Optional"' in texts  # the field at 0 shows "Optional"


def test_the_names_the_manual_gives_select_a_standard() -> None:
    """Step 4 of the manual: one standard named in the path is selected, several give Auto Detect."""
    from src.gimap.features.calibration.domain import detect_standard_keys

    section = _part(_text(MANUAL), "## 9. Geometry Calibration Tool")
    for name, key in (("AgBh", "agbh"), ("LaB6", "lab6"), ("CeO2", "ceo2")):
        assert f"`{name}`" in section
        assert detect_standard_keys(f"calibration/{name}_00001.tif") == (key,)
        assert detect_standard_keys(f"{name}/image_00001.tif") == (key,)  # the folder counts too
        assert detect_standard_keys(f"{name}_2024/week 3/run/image_00001.tif") == (key,)  # "a folder above it"
    assert detect_standard_keys("calibration/image_00001.tif") == ()  # nothing named: the field is left as it was
    # "when it names more than one, the field is set to Auto Detect": the window does so for more than one key.
    assert "when it names more than one, the field is set to **Auto Detect**" in " ".join(section.split())
    assert len(detect_standard_keys("LaB6/AgBh_00001.tif")) > 1 and len(detect_standard_keys("lab6_ceo2_00001.tif")) > 1
    loading = _text(ROOT / "src" / "gimap" / "features" / "calibration" / "presentation" / "bindings" / "image_loading.py")
    assert "elif len(detected_standards) > 1:" in loading and 'self.standard_combo.findData("auto")' in loading


def test_the_estimated_distance_is_filled_in_as_the_documents_say(tmp_path: Path) -> None:
    """Section 9 of the manual, on the window of a new installation, with made-up frames and a made-up result.

    "It is never filled with a built-in value: on a new installation it stays on *Optional*. But each time an
    image is opened it takes the distance the image file states, or else the distance of the calibration
    applied last, also when you had typed another value or cleared the field; with neither, a value you typed
    stays." When this fails the rule changed: change section 9 of docs/User_Manual.md, the limitation and the
    fix in CHANGELOG.md and docs/ui/workspaces/calibration.md (the rule itself is tested, case by case, in
    tests/test_release2_calibration_estimate.py).
    """
    from datetime import datetime, timezone

    import numpy as np
    from PyQt5.QtCore import QCoreApplication, QEvent
    from PyQt5.QtWidgets import QApplication

    from src.gimap.app import AppContext
    from src.gimap.features.calibration.domain import CalibrationCandidate, CalibrationResult, DetectorImage
    from src.gimap.features.calibration.infrastructure.adapters.local import SettingsGeometryAdapter
    from src.gimap.features.calibration.presentation.dialog import GeometryCalibrationDialog
    from src.gimap.integrations.state import (
        InMemoryInstrumentProfileRepository,
        InMemorySessionRepository,
        StorePreferencesRepository,
        StoreSettingsRepository,
        UserStore,
    )

    section = " ".join(_part(_text(MANUAL), "## 9. Geometry Calibration Tool").split())
    for said in ("It is never filled with a built-in value: on a new installation it stays on *Optional*",
                 "also when you had typed another value or cleared the field", "with neither, a value you typed stays",
                 "a NeXus file can; the distance in a CBF or EDF header is not used"):
        assert said in section, said
    fixed = " ".join(_part(_text(CHANGELOG), "### Fixed before the release", "### ").split())
    for said in ("no longer fills **Estimated distance** with a distance nobody measured", "the built-in 2000 mm",
                 "with neither it shows *Optional* and the search gets no estimate",
                 "(more than 100 m) is no longer turned into 100000.00 mm"):
        assert said in fixed, said

    # The store of a new installation: no file yet, so the built-in settings. They still hold a distance ...
    store = UserStore(tmp_path / "home" / "settings.json")
    settings = StoreSettingsRepository(store)
    assert settings.get("detector", "distance") == 2000 and settings.get("system", "geometry_calibration") is None
    # ... which is no geometry in use: only an Apply makes one (its record is what tells the two apart).
    unknown = {"distance": 0.0, "beam_center_x": 0.0, "beam_center_y": 0.0}
    assert SettingsGeometryAdapter(settings).current_geometry(unknown) == unknown

    def frame(name: str, distance_m: float | None = None) -> DetectorImage:
        data = np.ones((96, 120), dtype=np.float32)
        return DetectorImage(data, np.zeros(data.shape, dtype=bool), Path(name), distance_m=distance_m)

    QApplication.instance() or QApplication([])
    context = AppContext(settings=settings, preferences=StorePreferencesRepository(store),
                         session=InMemorySessionRepository(), instrument_profiles=InMemoryInstrumentProfileRepository())
    window = GeometryCalibrationDialog(app_context=context)
    field = window.estimated_distance_spin
    try:
        assert field.text() == "Optional"
        window._image_loaded(frame("first.cbf"))  # what the loader hands over for a file that states no distance
        assert (field.value(), field.text()) == (0.0, "Optional")  # "on a new installation it stays on *Optional*"
        run = _text(ROOT / "src" / "gimap" / "features" / "calibration" / "presentation" / "bindings" / "calibration_run.py")
        assert '"estimated_distance_mm": self.estimated_distance_spin.value() or None' in run  # 0: no estimate

        field.setValue(1600.0)  # "with neither, a value you typed stays"
        window._image_loaded(frame("second.cbf"))
        assert field.text() == "1600.00 mm"
        window._image_loaded(frame("third.nxs", distance_m=0.85))  # "the distance the image file states"
        assert field.text() == "850.00 mm"
        window._image_loaded(frame("fourth.cbf"))  # what was filled in for another file does not stay
        assert field.text() == "Optional"
        window._image_loaded(frame("fifth.nxs", distance_m=100.01))  # more than the field can show (100 m)
        assert field.maximum() == 100000.0 and field.text() == "Optional"

        # "or else the distance of the calibration applied last, also when you had typed another value or
        # cleared the field".
        candidate = CalibrationCandidate("agbh", 123.5, 234.5, 1456.7, matched_ring_count=3, rms_residual_px=1.25)
        window.view_model.result = CalibrationResult(
            str(tmp_path / "fourth.cbf"), 10, 20, "abc", 12.0, 12.398419843320026 / 12.0, "Pilatus 1M", 172e-6, 172e-6,
            candidate, [candidate], datetime.now(timezone.utc).isoformat(), metadata={"image_shape": [96, 120]})
        window.view_model.apply_result()
        assert isinstance(settings.get("system", "geometry_calibration"), dict)
        for typed in (900.0, 0.0):
            field.setValue(typed)
            window._image_loaded(frame("sixth.cbf"))
            assert field.text() == "1456.70 mm", typed
        window._image_loaded(frame("seventh.nxs", distance_m=0.85))  # the file's own distance comes first
        assert field.text() == "850.00 mm"
    finally:
        for name in ("_load_thread", "_cal_thread"):
            thread = getattr(window, name, None)
            if thread is not None:
                thread.wait(10000)
        window.close()
        window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)


def test_the_auto_detect_limitation_is_still_true(tmp_path: Path) -> None:
    """A public silver behenate image under a name that names no standard.

    Auto Detect without an estimated distance (what a new installation has) ranks a LaB₆ solution at a much
    shorter distance first; with the standard chosen and no estimated distance the distance is right; with the
    standard chosen and a wrong estimate typed in (2000 mm) a solution near 1830 mm comes first. When one of
    these is corrected this test fails: change the limitation in CHANGELOG.md, docs/User_Manual.md
    (section 9) and docs/ui/workspaces/calibration.md, and turn that part of this test into the regression
    test. What helps is checked too: an approximate distance puts the right standard first for Auto Detect.
    """
    if not PUBLIC_AGBH.is_file():
        pytest.skip("the public silver behenate image is not in this checkout")
    from src.gimap.features.calibration.application import CalibrationRequest, LoadCalibrationImage, RunCalibration
    from src.gimap.features.calibration.domain import detect_standard_keys
    from src.gimap.features.calibration.infrastructure.adapters import (
        LegacyCalibrationRunnerAdapter,
        LocalCalibrationImageAdapter,
    )

    manifest = json.loads(_text(PUBLIC_AGBH.parents[1] / "manifest.json"))["saxs_agbh_pyfai"]
    energy_kev = 12.398419843 / float(manifest["wavelength_angstrom"])
    expected_mm = float(manifest["ring_distance_mm_at_stated_wavelength"])
    image_path = tmp_path / "standard_image.edf"
    shutil.copyfile(PUBLIC_AGBH, image_path)
    assert detect_standard_keys(image_path) == ()
    image = LoadCalibrationImage(LocalCalibrationImageAdapter())(image_path, None)
    run = RunCalibration(LegacyCalibrationRunnerAdapter())

    chosen = run(CalibrationRequest(image=image, energy_kev=energy_kev, standard_key="agbh"), None, None)
    best = chosen.selected_candidate
    assert best.standard_key == "agbh" and abs(best.distance_mm - expected_mm) <= 0.03 * expected_mm

    automatic = run(CalibrationRequest(image=image, energy_kev=energy_kev, standard_key="auto"), None, None)
    first = automatic.selected_candidate
    assert not (first.standard_key == "agbh" and abs(first.distance_mm - expected_mm) <= 0.03 * expected_mm), (
        "Auto Detect now ranks the right standard first on this image: remove the known limitation from "
        "CHANGELOG.md, docs/User_Manual.md (section 9) and docs/ui/workspaces/calibration.md"
    )
    # What the documents say was seen: another standard, at a much shorter distance; silver behenate last.
    assert first.standard_key == "lab6" and first.distance_mm < 0.25 * expected_mm
    assert automatic.candidates[-1].standard_key == "agbh"

    # "With silver behenate chosen and 2000 mm typed in, a solution near 1830 mm came first."
    misled = run(CalibrationRequest(image=image, energy_kev=energy_kev, standard_key="agbh",
                                    estimated_distance_mm=2000.0), None, None).selected_candidate
    assert not abs(misled.distance_mm - expected_mm) <= 0.03 * expected_mm, (
        "an estimate of 2000 mm no longer misleads the search on this image: change the second "
        "case of the limitation in CHANGELOG.md, docs/User_Manual.md (section 9) and "
        "docs/ui/workspaces/calibration.md"
    )
    assert misled.standard_key == "agbh" and round(misled.distance_mm, -1) == 1830.0
    assert round(best.distance_mm, -1) == 1630.0  # "the real distance is about 1630 mm"

    # "An approximate distance helps Auto Detect too: with 1500 to 1800 mm typed in, the silver behenate
    # solution came first on this image." The solution is the one found with the standard chosen.
    for estimate_mm in (1500.0, 1800.0):
        helped = run(CalibrationRequest(image=image, energy_kev=energy_kev, standard_key="auto",
                                        estimated_distance_mm=estimate_mm), None, None).selected_candidate
        assert helped.standard_key == "agbh" and abs(helped.distance_mm - best.distance_mm) < 0.5, estimate_mm


# -- what 1D Predict does with a changed model file ----------------------------------------------------


def test_the_documents_say_what_each_network_does_with_a_changed_model_file(tmp_path: Path) -> None:
    """README, section 4 and section 13 of the manual: *General V5* does not start and names the file,
    *Single RC specialist* goes on with its numerical fit and gives "Fast model unavailable" as the reason."""
    for path in (README, MANUAL):
        text = _flat(path)
        assert "*General V5* does not start" in text, path.name
        assert "goes on with its numerical fit" in text, path.name
        assert "Fast model unavailable" in text, path.name
        assert "with a changed or missing file it does not start" not in text, path.name  # true of one network only
    assert "the message names the file" in _flat(README) and "the message names the file" in _flat(MANUAL)

    # General V5: every file of its manifest is compared before anything is loaded; the assertion carries the name.
    v5 = ROOT / "modules" / "Fitting_1D_Model" / "Workflow_v5"
    general = _text(v5 / "conditional_v2" / "conditional_fast_predict_v2.py")
    assert "assert hashlib.sha256((R/name).read_bytes()).hexdigest()==h,name" in general

    # Single RC specialist: the predictor refuses a changed file ...
    from src.gimap.features.fitting.infrastructure.adapters.stable_blue import BluePredictor

    bundle = tmp_path / "stable_blue_rc_v1"
    shutil.copytree(v5 / "stable_blue_rc_v1", bundle)
    BluePredictor(bundle)  # as it is in the repository: accepted
    changed = bundle / "protocol.json"
    changed.write_bytes(changed.read_bytes().replace(b"\n", b"\r\n") + b" ")  # what a converting checkout does
    with pytest.raises(ValueError, match="checksum mismatch: protocol.json"):
        BluePredictor(bundle)
    # ... and the engine that uses it catches that, records the reason and goes on with the numerical fit.
    engine = _text(ROOT / "src" / "gimap" / "features" / "fitting" / "infrastructure" / "adapters" / "stable_blue.py")
    assert 'reason = f"Fast model unavailable: {type(exc).__name__}: {exc}"' in engine
    assert 'best_source="stable_numerical_fallback"' in engine and "fallback_reason=reason" in engine
    # The reason reaches the table of the 1D Predict window.
    dialog = _text(ROOT / "src" / "gimap" / "features" / "fitting" / "presentation" / "workflow_v5_dialog.py")
    assert 'trf("Reason: {reason}", reason=r["fallback_reason"])' in dialog

    section = " ".join(_part(_part(_text(MANUAL), "## 13. Troubleshooting"), "### 1D Predict Does Not Start", "### ").split())
    for said in ("tools\\update_source.ps1", "converted line endings", "`git status` does not show it",
                 "git checkout -- modules/Fitting_1D_Model/Workflow_v5", "also when there is nothing new to download"):
        assert said in section, said


# -- the private evaluation case ----------------------------------------------------------------------


def test_the_agent_rules_keep_private_cases_out_of_the_repository() -> None:
    rules = _text(AGENTS)
    assert "唯一的例外" not in rules, "AGENTS.md: no exception for a tracked case any more"
    assert "没有例外" in rules and rules.count("docs/agents/eval/local/") >= 2
    assert "`docs/agents/eval/*.json`" not in rules

    evaluation = _part(_text(PLAYBOOK), "## 评估")
    for said in ("docs/agents/eval/local/", "--case", "退出码 2", "被 git 忽略", "eval/README.md"):
        assert said in evaluation, said
    assert "现有" not in evaluation, "the playbook describes how a case is found, not one case"
    assistant = _text(ROOT / "src" / "gimap" / "features" / "assistant" / "README.md")
    assert "docs/agents/eval/local/" in assistant
    assert "docs/agents/eval/local/" in _text(MANUAL) and "docs/agents/eval/local/" in _text(DEVELOPMENT)


def test_the_evaluation_tool_looks_where_the_documents_say() -> None:
    from tools import eval_giwaxs_agent as tool

    assert Path(tool.LOCAL_CASES) == ROOT / "docs" / "agents" / "eval" / "local"
    assert (ROOT / "docs" / "agents" / "eval" / "README.md").is_file(), "the documents point at this file"


def test_the_evaluation_tool_says_when_there_is_no_case(tmp_path: Path, capsys) -> None:
    """"没有案例……工具说明原因，以退出码 2 结束": a folder without a case, given as the case."""
    from tools import eval_giwaxs_agent as tool

    with pytest.raises(tool.NothingToEvaluate):
        tool.find_cases(None, folder=tmp_path)
    assert tool.main(["baseline", "--case", str(tmp_path / "no_such_case.json")]) == 2
    assert "No such case" in capsys.readouterr().err


def test_the_evaluation_tool_reads_reports_as_the_changelog_says(tmp_path: Path, capsys) -> None:
    """ "Reports saved by Windows tools are read (UTF-8 with a byte order mark, UTF-16). When there is no case,
    or a case or a report cannot be used — another encoding, an empty report, a malformed case — the tool says
    why in one sentence and ends with exit code `2`." An invented case and invented reports."""
    from tools import eval_giwaxs_agent as tool

    said = " ".join(_part(_text(CHANGELOG), "### Command line and agents", "### ").split())
    for words in ("`tools/eval_giwaxs_agent.py`", "`docs/agents/eval/local/`, a folder Git ignores",
                  "the repository ships none", "(UTF-8 with a byte order mark, UTF-16)",
                  "another encoding, an empty report, a malformed case", "ends with exit code `2`"):
        assert words in said, words

    case = tmp_path / "invented-film.json"
    rule = {"id": "made-up-rule", "any": ["cubic"], "why": "an invented rule"}
    case.write_text(json.dumps({"report": {"mistakes": [rule]}}), encoding="utf-8")
    sentence = "The film is cubic, with a spacing of 4.2 Å.\n"
    as_windows_saves_it = {"plain.md": sentence.encode("utf-8"), "marked.md": b"\xef\xbb\xbf" + sentence.encode("utf-8"),
                           "powershell.md": sentence.encode("utf-16")}
    for name, data in as_windows_saves_it.items():
        (tmp_path / name).write_bytes(data)
        assert tool.main(["report", str(tmp_path / name), "--case", str(case)]) == 1, name  # read: the rule finds its word
        assert "made-up-rule" in capsys.readouterr().out, name

    for name, data in {"another-encoding.md": sentence.encode("latin-1"), "empty.md": b" \r\n"}.items():
        (tmp_path / name).write_bytes(data)
        assert tool.main(["report", str(tmp_path / name), "--case", str(case)]) == 2, name
        printed = capsys.readouterr()
        assert printed.err.strip() and "made-up-rule" not in printed.out, name  # a sentence, and nothing scored
    for name, text in {"cut-off.json": '{"report": {"mistakes": [', "a-list.json": "[]", "no-report.json": "{}"}.items():
        (tmp_path / name).write_text(text, encoding="utf-8")
        assert tool.main(["report", str(tmp_path / "plain.md"), "--case", str(tmp_path / name)]) == 2, name
        assert capsys.readouterr().err.strip(), name


def test_no_document_points_at_one_case_file() -> None:
    """A committed text may name the folder of the cases and its README, never one case in it."""
    one_case = re.compile(r"docs/agents/eval/(?:local/)?(?!README)[A-Za-z0-9][\w.\-]*\.json")
    documents = [README, CHANGELOG, MANUAL, AGENTS, DEVELOPMENT, PLAYBOOK,
                 ROOT / "docs" / "agents" / "gisaxs-playbook.md", ROOT / "docs" / "agents" / "eval" / "README.md",
                 ROOT / "src" / "gimap" / "features" / "assistant" / "README.md"]
    found = [path.relative_to(ROOT).as_posix() for path in documents if path.is_file() and one_case.search(_text(path))]
    assert not found, found


# -- Fitting's first model, and the kind of release ----------------------------------------------------


def _numbers(value):
    if isinstance(value, dict):
        for item in value.values():
            yield from _numbers(item)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        yield float(value)


def test_the_first_fitting_model_is_neutral_as_the_manual_says() -> None:
    assert "A new installation starts with a neutral model" in _flat(MANUAL)
    assert "Fitting starts a new user with a neutral model" in _flat(CHANGELOG)
    template = json.loads(_text(ROOT / "config" / "model_parameters.json"))
    values = list(_numbers(template.get("fitting", {})))
    assert values
    # A value left by a fit has many digits; a default somebody chose has few.
    fitted = [value for value in values if float(f"{value:.4g}") != value]
    assert not fitted, f"config/model_parameters.json holds values of a fit: {fitted[:5]}"


def test_a_vertical_cylinder_is_stored_as_the_changelog_says(tmp_path: Path) -> None:
    """ "kept the radius and distance that were there and gained a second block named `vertical cylinder`" is
    what an earlier copy did. Now every spelling of a shape's name means the one block, and a file with the
    extra block still loads (made-up numbers; the cases are in tests/test_release2_fitting_cylinder_key.py)."""
    from src.gimap.features.fitting.application import ManageFittingModelParameters
    from src.gimap.features.fitting.infrastructure.adapters import FittingModelParametersAdapter
    from src.gimap.features.fitting.infrastructure.adapters.model_parameters_store import shape_block_name

    fixed = " ".join(_part(_text(CHANGELOG), "### Fixed before the release", "### ").split())
    for said in ("A vertical cylinder taken over from a solution selected in the 1D Predict window is saved with its own values",
                 "**File ▸ Labs Parameters ▸ Save As…**", "gained a second block named `vertical cylinder`",
                 "`vertical_cylinder` is the block that is used",
                 "the extra one leaves the file the next time a value of a vertical cylinder is saved"):
        assert said in fixed, said
    menu = _text(ROOT / "src" / "gimap" / "app" / "presentation" / "menu_bar.py")
    assert 'addMenu("&Labs Parameters")' in menu and '"&Save As…"' in menu
    page = _text(FITTING_PAGE)
    assert "`model_parameters_store.shape_block_name`" in page and "tests/test_release2_fitting_cylinder_key.py" in page
    # The same page says what the template is tied to (a changed default fails that test until the JSON follows).
    assert "`config/model_parameters.json` 是这个存储的出厂模板" in page and "tests/test_release2_defaults_template.py" in page

    path = tmp_path / "model_parameters.json"

    def store() -> ManageFittingModelParameters:
        return ManageFittingModelParameters(FittingModelParametersAdapter(path))

    def blocks() -> dict:
        return json.loads(path.read_text(encoding="utf-8"))["fitting"]["particles"]["particle_1"]["parameters"]

    # "is saved with its own values": the name as the page shows it goes to the block the page reads.
    first = store()
    assert shape_block_name("Vertical Cylinder") == "vertical_cylinder"
    assert first.set_particle_parameter("fitting", "particle_1", "Vertical Cylinder", "radius", 7.25)
    assert first.save_parameters()
    assert "vertical cylinder" not in blocks() and blocks()["vertical_cylinder"]["radius"] == 7.25

    # A file as an earlier copy left it: the extra block beside the one the page reads.
    document = json.loads(path.read_text(encoding="utf-8"))
    document["fitting"]["particles"]["particle_1"]["parameters"]["vertical cylinder"] = {"radius": 6.25, "diameter": 41.0}
    path.write_text(json.dumps(document, indent=4), encoding="utf-8")
    before = path.read_bytes()
    again = store()
    assert again.get_particle_parameter("fitting", "particle_1", "Vertical Cylinder", "radius") == 7.25  # "the block that is used"
    assert path.read_bytes() == before  # "still loads": reading rewrites nothing
    # "the extra one leaves the file the next time a value of a vertical cylinder is saved".
    assert again.set_particle_parameter("fitting", "particle_1", "Vertical Cylinder", "radius", 8.5)
    assert again.save_parameters()
    assert "vertical cylinder" not in blocks() and blocks()["vertical_cylinder"]["radius"] == 8.5


def test_version_0_1_0_is_still_called_a_pre_release() -> None:
    from src.gimap.shared.version import __version__

    if __version__ != "0.1.0":
        pytest.skip("another version: its documents say what kind of release it is")
    assert "version is 0.1.0, a pre-release" in _text(README)
    assert re.search(r"^## 0\.1\.0 — \d{4}-\d{2}-\d{2} \(pre-release\)$", _text(CHANGELOG), flags=re.M)
    manual = _text(MANUAL)
    assert "version 0.1.0, a pre-release" in _part(manual, "## 1. Introduction")
    assert "**GIMaP 0.1.0** (pre-release)" in _part(manual, "## 15. Version and Contact")
    assert "Version 0.1.0 is a pre-release" in _part(manual, "## 14. FAQ")
    about = _text(ROOT / "src" / "gimap" / "app" / "presentation" / "menu_bar.py")
    assert "(Pre-release)" in about
    assert "0.1.0 is a pre-release" in _text(DEVELOPMENT)
