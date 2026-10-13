# Development setup

- **Last verified**: 2026-10-06 (Windows 11, Python 3.10.18, pytest 9.1, ruff 0.16)

GIMaP currently supports Python 3.10 and 3.11. Python 3.10 is the safest common
choice for TensorFlow 2.15 and BornAgain 24.1 compatibility.

## Install the development dependencies

Create and activate an environment first, then install the development dependency
set. It includes the runtime requirements plus pytest and Ruff.

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

On macOS, install a compatible BornAgain wheel separately as described below.
`requirements.txt` intentionally does not request BornAgain from PyPI on macOS
because the project does not publish a macOS wheel there.

## BornAgain 24.1

### Windows and Linux

BornAgain publishes Python wheels for Windows and Linux. With Python 3.10 or 3.11,
the pinned entry in `requirements.txt` installs it normally:

```bash
python -m pip install -r requirements-dev.txt
python -c "import bornagain; print(bornagain.__file__)"
```

### macOS

BornAgain recommends its Homebrew tap because it does not publish prebuilt macOS
packages. Install version 24.1 and inspect the generated wheel:

```bash
brew tap mlz/homebrew https://jugit.fz-juelich.de/mlz/homebrew/
brew install mlz/homebrew/bornagain@24.1
bornagain_info
```

The wheel is CPython-ABI-specific. For example, a filename containing `cp314`
cannot be installed into GIMaP's Python 3.10 environment. Install the Homebrew
wheel only when its `cp3xx` tag matches `python --version`:

```bash
python -m pip install /path/from/bornagain_info/bornagain-24.1-cp310-*.whl
python -c "import bornagain; print(bornagain.__file__)"
```

If the tags do not match, do not force the installation. Build BornAgain 24.1
against the same interpreter used by GIMaP, or use a matching wheel produced by
the project team. The upstream build documentation is at
<https://bornagainproject.org/24/deploy/build>.

## Checks

Run every check with one command (any platform):

```bash
python tools/check.py
```

It runs these steps in order. Each step runs even when an earlier one failed, and
a summary with one line per step (result, time, pytest's counts) comes at the end.

| Step | Command | Role |
| --- | --- | --- |
| Main test suite | `python -m pytest tests` | gate |
| Offscreen smoke | `python tools/offscreen_smoke.py` | gate |
| Ruff | `python -m ruff check .` | gate |
| Research tests | `python -m pytest utils/ML_Fitting_1D_GISAXS/tests --continue-on-collection-errors` | reported only; skipped on Windows by default |

- The **offscreen smoke** starts and closes the real main window with in-memory
  settings, session and preferences. It requires 6 pages (Start, Analyze, Fitting,
  Compare, 2D Prediction, Trainset Build), the Fitting, Prediction and Trainset
  bindings, and the Compare page, and prints `Offscreen startup OK: pages=6, …`.
- **Ruff** selects only `E9`, `F63`, `F7` and `F82`: syntax errors, invalid
  comparisons and control flow, and undefined names. A pass says nothing about style.
  Do not run a broad formatter; it would rewrite many files.
- The **research tests** belong to the fitting research code in
  `utils/ML_Fitting_1D_GISAXS`, which the GUI only runs as a subprocess. One of
  them imports the POSIX-only `resource` module, which stops collection on Windows,
  and many others fail there. Their result never changes the exit code.
- Exit code: `0` when every gate step that ran passed, `1` when one failed or the
  run was interrupted (`2` for a wrong command line).
- Qt runs offscreen (`QT_QPA_PLATFORM=offscreen` unless set), and `GIMAP_HOME`
  points to a temporary folder unless set, so the check never touches the real
  user data folder.

Options:

```bash
python tools/check.py --skip tests              # smoke and ruff only (seconds)
python tools/check.py --research run            # also the research tests (on Windows too)
python tools/check.py -- -x -k analyze          # arguments after -- go to the main suite's pytest
python tools/check.py -- tests/test_ui_source_of_truth.py   # test paths replace tests/
```

An option's value (`-k tests`, `--deselect tests/test_x.py::t`) is not a test path,
so `tests/` still runs.

Times on the development machine (2026-10-06): main suite 14–15 minutes
(about 1,380 tests), smoke about 7 s, ruff under 1 s, research tests about 5 minutes.

When the output goes to a file or a pipe (a log, an agent), `check.py` writes UTF-8
so that units such as `Å⁻¹` survive; set `PYTHONIOENCODING` to choose another encoding.

### Windows

A bare `python -m pytest` runs only `tests/` (`testpaths` in `pyproject.toml`); the research
tests run only as the separate `check.py` step, which Windows skips (one of them imports the
POSIX-only `resource` module and would stop collection). To run the main suite by hand:

```powershell
$env:QT_QPA_PLATFORM = "offscreen"
python -m pytest tests -q
python -m pytest tests/test_architecture_dependencies.py tests/test_ui_source_of_truth.py tests/test_ui_design_system.py
```

`tests/conftest.py` already sets `QT_QPA_PLATFORM=offscreen` (unless set), gives every
run its own `GIMAP_HOME`, loads Arial, Segoe UI and Microsoft YaHei (offscreen Qt on
Windows has no font database), and applies the light theme at 9 pt.

### Screenshots

Tests are already themed and have fonts. A standalone screenshot script has to do
what `main()` and `conftest.py` do: load fonts including a CJK one
(`QFontDatabase.addApplicationFont` on `C:/Windows/Fonts/msyh.ttc`, or
`QT_QPA_FONTDIR=C:/Windows/Fonts`; without it Chinese shows as empty boxes),
`app.setFont(QFont("Segoe UI", 9))`, and `apply_theme(mode, 9.0)`. Check both the
light and the dark theme, and both languages. Set `GIMAP_HOME` to a temporary
folder: a script that calls `create_app_context()` reads and writes the real user
data folder, and even with in-memory repositories (as in `tools/offscreen_smoke.py`)
opening Fitting writes `model_parameters.json` there when it is missing.

## Python Views

GIMaP does not use Qt Designer forms or pyuic output. Static layouts are
hand-maintained `presentation/views/*_view.py` files: widgets, layouts, object
names, tab order and visual defaults only, importing no application, domain or
infrastructure module. Behaviour lives in the page, its bindings or mixins.
Update `EXPECTED_VIEWS_BY_OWNER` in `tests/test_ui_source_of_truth.py` whenever a
View is added, removed, or renamed. See `docs/architecture/dependency-rules.md` for
every rule the tests enforce (600 lines per module, 300 per `view_model.py`, imports).

## pyqtgraph

`requirements.txt` pins `pyqtgraph>=0.13.7,<0.15`. It draws the detector images
and curves of Analyze, Fitting, Compare and the automatic analysis
(`DetectorView`, `CurvePlot`), so it is required; modules import it lazily, inside
the functions that use it.

Focused display verification:

```bash
QT_QPA_PLATFORM=offscreen python -m pytest tests/test_display_artist_reuse.py tests/test_scientific_image_viewer.py tests/test_fitting_curve_rendering.py
```

## Version and release

- The version is written once, in `src/gimap/shared/version.py` (`__version__`). The About box and the
  export records that name the software read it from there.
- Every version gets an entry in `CHANGELOG.md`: what a user gets, and the known limitations. The first
  paragraph of `README.md` and sections 1 and 15 of `docs/User_Manual.md` name the version too.
- The portable Windows version is built with `tools/build_portable_release.ps1`: it packs a Conda
  environment with `conda-pack` together with the application files. When the ZIP is larger than one
  GitHub asset may be, the script also writes the part files and
  `00_Download_and_Install_GIMaP-<version>.bat`, the file users download from the GitHub Release tagged
  `v<version>`.
- `-ListOnly` prints the application files the package would hold, without packing anything: check
  the list before a release.
- The trained weights of 1D Predict (eight `*.h5` / `*.npz` files under
  `modules/Fitting_1D_Model/Workflow_v5/`, about 56 MB) are tracked by Git, so a clone runs 1D Predict and
  its tests. `.gitignore` still ignores `*.h5`, `*.npz` and every `results/` folder: a new or renamed
  weight file is not picked up by `git add` on its own (`git add -f <file>`), and the model manifests
  (`conditional_v2/conditional_fast_manifest_v2.json`, `stable_blue_rc_v1/MANIFEST.json`) must list its
  checksum. `.gitattributes` switches line-ending conversion off for that folder, because the checksums
  were taken from the bytes on disk. A checkout made on Windows before that rule (2026-10-13) holds part
  of these files with converted line endings, and Git does not notice it: `tools/update_source.ps1`
  writes them again.
- 0.1.0 is a pre-release: the About box, `README.md`, the manual and `CHANGELOG.md` say so. Change them
  together when a version is released as final.

## Updating a source checkout

`tools/update_source.ps1` updates the checked-out branch of a checkout from `origin`
(`powershell -ExecutionPolicy Bypass -File tools\update_source.ps1`, `-DryRun` to report only): a
fast-forward when origin only added commits; when the histories diverged, a branch
`backup/<date>-<time>-<commit>` of the state before, then `git reset --hard origin/<branch>`. Modifications
of tracked files go through `git stash` and back; untracked and ignored files are never changed (the script
stops when the new version would overwrite one). Tracked files that `.gitattributes` marks `-text` (the
model files of 1D Predict) and that the checkout holds with converted line endings are written again, also
when there is nothing to update. Exit code 0: updated or nothing to do; 1: not updated, or not completely;
2: updated, but the modifications wait in the stash. `README.md` describes it for users, with the two plain
commands for a checkout that does not have the script yet.

- The script needs Git 2.16 or newer and stops with that sentence when it finds an older one.
- When Git cannot replace a file because another program holds it open, the script writes the files Git had
  already replaced again as they were and puts the modifications back where they were (exit code 1,
  "Nothing is left half way"); the next run, with the program closed, finishes the update
  (`tests/test_release2_updater_recovery.py`).
- A checkout that ran 0.0.2 ends with exit code 2 because of one file: `config/model_parameters.json`, which
  0.0.2 wrote its fits into and which is now the tracked template of a new user's model. That modification
  is meant to be dropped (`git stash drop` when the stash holds nothing else). `git stash pop`, which the
  script prints for modifications somebody wants back, writes conflict markers into the template; with other
  modifications in the stash, `git checkout HEAD -- config/model_parameters.json` after the pop puts the
  template back.

The history of `main` was rewritten on 2026-10-13. A clone last updated before that day can hold commits
that are gone from GitHub: do not update it with `git pull`, and merge or push nothing from it, since that
would bring the old history back to GitHub.
Move such a clone to the new `main` first: with the script (a clone that does not have it yet gets it with
`git fetch origin` and `git checkout origin/main -- tools/update_source.ps1`), or with `git fetch origin`
and then `git reset --hard origin/main` (edits of tracked files and local commits are lost unless they were
stashed or kept on a branch, and an untracked or ignored file at a place where the new version has a file
is overwritten). After the two plain commands, run the script once: `git reset --hard` does not write
again the model files whose content did not change, so on Windows they keep their converted line endings
and 1D Predict fails its checksums. Carry own commits over with `git cherry-pick` onto the new `main`, not
by merging the old branch, and delete the `backup/…` branch once things look right: until then it keeps
the old history on that computer.

What still holds the old history in such a clone afterwards, when it must be gone from that computer too:

- the `backup/…` branch and the stash the script left (`git branch -D backup/…`, `git stash drop`), and any
  branch of your own that was made before the rewrite;
- a tag that was moved or deleted on GitHub: `git fetch` leaves an existing local tag as it is
  (`git fetch origin --tags --force` takes the moved ones over, `git tag -d <name>` deletes one);
- Git's own record of where the branches were (the reflog) and the objects nothing points at any more:
  they stay readable by their commit id until Git prunes them, after weeks.
  `git reflog expire --expire=now --all` and then `git gc --prune=now` remove them at once.

A fresh clone has none of this.

## Evaluation cases

`tools/eval_giwaxs_agent.py` reads its cases from `docs/agents/eval/local/`, which Git ignores: a case names
real files and what was measured on them, so a case built on private or unpublished data stays on the
computer that holds the data. A fresh clone has no case; the tool then says so and ends with exit code 2,
and the tests that need a real case are skipped. `docs/agents/eval/README.md` describes the format.
