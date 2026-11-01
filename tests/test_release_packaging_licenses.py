"""The portable package carries the licence texts of everything it bundles (``licenses/`` in the ZIP).

conda-pack copies ``site-packages``, where most wheels keep their licence files, but not every wheel has
one: PyQt5 5.15.11 is GPL-3.0 and its wheel carries no text of it. The licences of the conda packages
(Python, OpenSSL, Tcl/Tk ...) sit in the package cache, outside the environment. ``tools/collect_licenses.py``
gathers all of them, with ``THIRD_PARTY_NOTICES.md`` and GIMaP's ``LICENSE``, and the release builder puts
the folder into the ZIP. A GPL-3.0 or LGPL-3.0 package without its text stops the build, and so does a conda
package that is not from conda-forge; another GPL or LGPL version is listed for a check by hand.

The environments here are made in a temporary folder; their licence texts are stand-ins (only the
first lines of the real ones, which is what the helper recognises a licence by).
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

from tools.collect_licenses import collect

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "tools" / "collect_licenses.py"
GPL = "                    GNU GENERAL PUBLIC LICENSE\n                       Version 3, 29 June 2007\n\n stand-in\n"
LGPL = "                   GNU LESSER GENERAL PUBLIC LICENSE\n                       Version 3, 29 June 2007\n"


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _distribution(site: Path, name: str, version: str, headers: str, files: dict[str, str] | None = None,
                  record: list[str] = ()) -> None:
    folder = site / f"{name}-{version}.dist-info"
    _write(folder / "METADATA", f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n{headers}\n")
    for relative, text in (files or {}).items():
        _write(folder / relative, text)
    if record:
        _write(folder / "RECORD", "".join(f"{entry},sha256=x,1\n" for entry in record))


def make_environment(prefix: Path) -> Path:
    """A small Conda environment: wheels with and without licence files, one conda package."""
    site = prefix / "Lib" / "site-packages"
    _distribution(site, "alpha", "1.0", "License-Expression: MIT\nLicense-File: LICENSE.txt",
                  {"licenses/LICENSE.txt": "MIT License\n"},
                  record=["alpha/__init__.py", "alpha/_vendor/LICENSE", "alpha/license.py",
                          "alpha-1.0.dist-info/METADATA"])
    _write(site / "alpha" / "_vendor" / "LICENSE", "BSD License of a vendored library\n")
    _write(site / "alpha" / "license.py", "print('code, not a licence text')\n")
    _distribution(site, "BornAgain", "24.1", "License: GNU General Public License",
                  {"licenses/COPYING": GPL})
    _distribution(site, "PyQt5", "5.15.11", "License: GPL v3")  # the real wheel has no licence file
    _distribution(site, "PyQt5-Qt5", "5.15.2", "License: LGPL v3", {"LICENSE": LGPL})
    _distribution(site, "quiet", "2.0", "License: MIT")
    _distribution(site, "~ld", "1.0", "License: MIT", {"LICENSE": "pip's leftover\n"})
    pkgs = prefix.parent / "pkgs" / "python-3.10.0-h0"
    _write(pkgs / "info" / "licenses" / "LICENSE", "PSF License\n")
    _write(prefix / "LICENSE_PYTHON.txt", "PSF License\n")
    _write(prefix / "conda-meta" / "python-3.10.0-h0.json", json.dumps({
        "name": "python", "version": "3.10.0", "license": "Python-2.0",
        "channel": "https://conda.anaconda.org/conda-forge/win-64",
        "extracted_package_dir": str(pkgs), "files": ["LICENSE_PYTHON.txt", "python.exe"]}))
    return prefix


def make_project(root: Path) -> Path:
    _write(root / "THIRD_PARTY_NOTICES.md", "# Third-party notices\n")
    _write(root / "LICENSE", "MIT License\n")
    return root


def test_every_package_gets_its_texts_and_a_gpl_wheel_without_one_gets_the_gpl(tmp_path):
    prefix = make_environment(tmp_path / "env" / "gimap-release")
    out = tmp_path / "licenses"
    summary = collect(prefix, out, make_project(tmp_path / "gimap"))
    assert summary["ok"], summary["problems"]
    assert (summary["python"], summary["conda"]) == (5, 1)  # the "~" leftover is not packed
    written = set(summary["written"])
    assert {
        "README.txt", "licenses.json", "THIRD_PARTY_NOTICES.md", "GIMaP/LICENSE",
        "python/alpha-1.0/licenses/LICENSE.txt", "python/alpha-1.0/alpha/_vendor/LICENSE",
        "python/BornAgain-24.1/licenses/COPYING", "python/PyQt5-5.15.11/GPL-3.0.txt",
        "python/PyQt5-Qt5-5.15.2/LICENSE", "python/PyQt5-Qt5-5.15.2/GPL-3.0.txt",
        "conda/python-3.10.0/LICENSE", "conda/python-3.10.0/runtime/LICENSE_PYTHON.txt",
    } <= written
    assert written == {path.relative_to(out).as_posix() for path in out.rglob("*") if path.is_file()}
    assert not [path for path in written if path.endswith(".py") or "~ld" in path]
    # The GPL text PyQt5 lacks is BornAgain's, byte for byte; the LGPL travels with the GPL.
    assert (out / "python/PyQt5-5.15.11/GPL-3.0.txt").read_text(encoding="utf-8") == GPL
    assert summary["without_text"] == ["quiet 2.0 (MIT)"]
    assert "PyQt5 5.15.11: GPL-3.0 text from BornAgain 24.1" in summary["filled"]
    readme = (out / "README.txt").read_text(encoding="utf-8")
    assert "PyQt5 5.15.11 -- GPL v3" in readme and "copied from BornAgain 24.1" in readme
    assert "quiet 2.0 -- MIT" in readme and "(no licence file in the installed package)" in readme
    record = json.loads((out / "licenses.json").read_text(encoding="utf-8"))
    assert {component["name"] for component in record["components"]} == {
        "alpha", "BornAgain", "PyQt5", "PyQt5-Qt5", "quiet", "python"}
    # Nothing of the computer that built the package goes into it.
    for text in (readme, (out / "licenses.json").read_text(encoding="utf-8")):
        assert str(tmp_path) not in text and tmp_path.as_posix() not in text


def test_a_gpl_package_without_any_text_stops_the_build(tmp_path):
    site = tmp_path / "env" / "Lib" / "site-packages"
    _distribution(site, "PyQt5", "5.15.11",
                  "Classifier: License :: OSI Approved :: GNU General Public License v3 (GPLv3)")
    _distribution(site, "unversioned", "1.0", "License: GNU General Public License")
    summary = collect(tmp_path / "env", tmp_path / "out")
    assert not summary["ok"]
    assert any(problem.startswith("PyQt5 5.15.11 declares") and "GPL-3.0" in problem
               for problem in summary["problems"]), summary["problems"]
    assert any(problem.startswith("unversioned 1.0 declares") for problem in summary["problems"])


def test_another_gpl_or_lgpl_version_is_listed_for_a_check_by_hand_but_does_not_stop_the_build(tmp_path):
    prefix = make_environment(tmp_path / "env" / "gimap-release")
    site = prefix / "Lib" / "site-packages"
    _distribution(site, "older", "1.0", "License: LGPL-2.1-or-later", {"AUTHORS": "Someone\n"})
    _distribution(site, "twofold", "2.0", "License: GPL-2.0-only", {"NOTICE": "Notice\n"})
    _distribution(site, "carried", "3.0", "License: LGPL-2.1",
                  {"COPYING": "GNU LESSER GENERAL PUBLIC LICENSE\nVersion 2.1, February 1999\n"})
    summary = collect(prefix, tmp_path / "out", make_project(tmp_path / "gimap"))
    assert summary["ok"], summary["problems"]
    assert summary["unchecked_copyleft"] == ["older 1.0 (LGPL-2.1-or-later)", "twofold 2.0 (GPL-2.0-only)"]


def test_a_conda_package_that_is_not_from_conda_forge_stops_the_build(tmp_path):
    prefix = make_environment(tmp_path / "env" / "gimap-release")
    _write(prefix / "conda-meta" / "openssl-3.0.0-h1.json", json.dumps({
        "name": "openssl", "version": "3.0.0", "license": "Apache-2.0",
        "channel": "https://repo.anaconda.com/pkgs/main/win-64", "files": []}))
    _write(prefix / "conda-meta" / "zlib-1.3-h2.json", json.dumps({
        "name": "zlib", "version": "1.3", "license": "Zlib", "files": []}))  # no channel recorded
    summary = collect(prefix, tmp_path / "out", make_project(tmp_path / "gimap"))
    assert not summary["ok"]
    assert summary["other_channels"] == ["openssl 3.0.0 from https://repo.anaconda.com/pkgs/main/win-64",
                                         "zlib 1.3 from an unknown channel"]
    assert any("not from conda-forge" in problem and "--override-channels" in problem
               for problem in summary["problems"]), summary["problems"]


def test_what_is_missing_is_a_problem(tmp_path):
    assert "No site-packages folder" in collect(tmp_path / "empty", tmp_path / "out1")["problems"][0]
    prefix = make_environment(tmp_path / "env" / "gimap-release")
    summary = collect(prefix, tmp_path / "out2", tmp_path / "no-project")
    assert not summary["ok"]
    assert [problem.split(" is missing")[0] for problem in summary["problems"]] == [
        "THIRD_PARTY_NOTICES.md", "LICENSE"]
    full = tmp_path / "full"
    _write(full / "old.txt", "from an earlier build\n")
    summary = collect(prefix, full)
    assert not summary["ok"] and "not empty" in summary["problems"][0]


def test_the_command_line_prints_one_json_line_and_says_so_by_its_exit_code(tmp_path):
    prefix = make_environment(tmp_path / "env" / "gimap-release")
    done = subprocess.run(
        [sys.executable, str(HELPER), "--root", str(make_project(tmp_path / "gimap")), "--prefix", str(prefix),
         "--out", str(tmp_path / "out")],
        capture_output=True, timeout=120)
    assert done.returncode == 0, done.stderr
    assert done.stdout.isascii() and len(done.stdout.decode().strip().splitlines()) == 1
    assert json.loads(done.stdout)["ok"] is True
    again = subprocess.run(
        [sys.executable, str(HELPER), "--prefix", str(prefix), "--out", str(tmp_path / "out")],
        capture_output=True, timeout=120)
    assert again.returncode == 1 and json.loads(again.stdout)["ok"] is False
    wrong = subprocess.run([sys.executable, str(HELPER), "--out", str(tmp_path / "x")], capture_output=True,
                           timeout=120)
    assert wrong.returncode == 2


def test_the_notices_name_the_licences_the_package_depends_on():
    """THIRD_PARTY_NOTICES.md is plain Markdown that Qt can show, and says what the builder relies on."""
    text = (ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    for said in ("MIT License", "PyQt5", "BornAgain", "GPL-3.0", "LGPL-3.0", "terms of the GPL-3.0",
                 "10.5281/zenodo.17466183", "creativecommons.org/licenses/by/4.0", "tools/collect_licenses.py"):
        assert said in text, said
    outside_code = re.sub(r"`[^`]*`", "", text)
    assert not re.search(r"<[A-Za-z/!][^>]*>", outside_code), "raw HTML (or an unquoted <placeholder>)"
    assert "$" not in outside_code  # no math
    # The pygid creators are the same in both places that name them.
    data_readme = (ROOT / "tests" / "data" / "external" / "README.md").read_text(encoding="utf-8")
    creators = re.search(r"\*\*Creators\*\*: (.+?)\.\n", data_readme, flags=re.S).group(1)
    names = [name.strip() for name in re.split(r",\s*", re.sub(r"\s+", " ", creators))]
    assert len(names) == 9
    flat = re.sub(r"\s+", " ", text)
    assert all(name in flat for name in names)
    assert re.search(r"Copyright \(c\) 2025–2026 ", (ROOT / "LICENSE").read_text(encoding="utf-8"))
