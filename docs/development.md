# Development setup

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

The individual commands are:

```bash
python -m pytest
python -m ruff check .
```

Run the complete repository verification with one command:

```bash
python tools/check.py
```

The unified command sets Qt to the offscreen platform for tests, runs the full
suite, and then runs the repository lint baseline. Ruff initially
checks syntax, invalid control flow, and undefined names across the repository.
Four explicitly listed legacy files are temporarily exempted from their known
`F821` or `F823` findings in `pyproject.toml`; no broad formatting pass is enabled.

## Python Views

GIMaP no longer uses Qt Designer forms or pyuic output. Each application or feature
presentation owns hand-maintained layout files under `presentation/views/`. The
View defines only widgets, layouts, object names, tab order, and visual defaults;
`page.py` or `dialog.py` injects ViewModels and connects behavior. Update the
explicit inventory in `tests/test_ui_source_of_truth.py` whenever a View is added,
removed, or renamed.
