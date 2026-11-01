"""What the Help menu does with the program's files and pages (``presentation/menu_bar.py`` builds it).

* **Example Data**: the public frames in ``tests/data/external/`` (with their ``.poni`` and any mask) are
  copied to a folder the user chooses, then opened in Analyze. Never opened in place: Export writes
  ``gimap_analysis/`` next to the data, and *Find Calibration Automatically* would find the other
  examples' ``.poni`` in the folders beside the frame.
* **Report a Problem**: the versions only (no path, no data) go to the clipboard; the browser opens a new
  issue. The program sends nothing.
* **Open Error Log Folder**: ``<user data>/logs``, where ``ErrorGuard`` writes ``errors.log``.
* **How to Cite**: the reference read from ``CITATION.cff`` with the version that runs; a DOI only when the
  file gives one.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import yaml
from PyQt5.QtCore import QStandardPaths, QUrl
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtWidgets import QApplication

from .error_guard import LOG_NAME
from .presentation.app_dialogs import inform, warn
from .presentation.components import show_toast
from .presentation.help_dialogs import ask_example_folder, plain_text, system_summary
from .presentation.i18n import tr, trf
from .presentation.menu_bar import GITHUB_URL, ISSUES_URL
from src.gimap.shared.version import APP_NAME, __version__

ROOT = Path(__file__).resolve().parents[3]
EXAMPLES_FOLDER = ROOT / "tests" / "data" / "external"
"""The public examples (``manifest.json`` names each one's frame; ``README.md`` their sources and licences)."""
CITATION_FILE = ROOT / "CITATION.cff"
EXAMPLE_NOTE = "SOURCE_AND_LICENCE.md"
"""The sources and licences of the examples (``tests/data/external/README.md``), copied beside each example."""


@dataclass(frozen=True)
class Example:
    key: str
    """The example's folder in ``tests/data/external`` and its entry in ``manifest.json``."""
    text: str
    tip: str


EXAMPLES = (
    Example("gisaxs_galaxi", "GISAXS — Ag &Nanoparticles (GALAXI)…",
            "GISAXS of Ag nanoparticles, GALAXI (Jülich), Pilatus 1M, with its calibration (.poni). "
            "Source: BornAgain test data. Licence: GPL-3.0"),
    Example("saxs_agbh_pyfai", "SAXS — Ag&Bh Calibrant (pyFAI)…",
            "Silver behenate calibration image, ESRF BM26, Pilatus 1M, with its .poni. "
            "Source: pyFAI test images. Licence: MIT (pyFAI), to be confirmed for the image files"),
    Example("giwaxs_p08_mapi", "GIWAXS — &MAPbI₃ Film (DESY P08)…",
            "GIWAXS of a MAPbI₃ perovskite film, DESY P08, PerkinElmer 2048 × 2048, with its LaB₆ calibration (.poni). "
            "Source: pygid usage examples, Zenodo 10.5281/zenodo.17466183. Licence: CC BY 4.0"),
    Example("giwaxs_xeuss_pbte", "GIWAXS — &PbTe, Tilted Detector (Xeuss)…",
            "GIWAXS of PbTe nanoplatelets, Xeuss 2.0 lab source, Pilatus 300K, 6° detector tilt, with its .poni "
            "and mask. Source: pygid usage examples, Zenodo 10.5281/zenodo.17466183. Licence: CC BY 4.0"),
)


def example_frame(key: str, root: Path | None = None) -> Path:
    """The frame of the example ``key`` (``manifest.json`` of ``root``, by default ``EXAMPLES_FOLDER``);
    ``FileNotFoundError`` when it is not there."""
    root = EXAMPLES_FOLDER if root is None else root
    try:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        frame = root / manifest[key]["frame"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise FileNotFoundError(str(root / key)) from exc
    if not frame.is_file():
        raise FileNotFoundError(str(frame))
    return frame


def copy_example(key: str, target: Path, root: Path | None = None) -> Path:
    """Copy the files of the example ``key`` (its frame, ``.poni``, mask and metadata; not the folders beside
    them, such as ``gimap_analysis/``) and the note on their sources and licences into ``target``, and return
    the copied frame. A file already in ``target`` is kept, never overwritten. Each file is copied under a
    hidden temporary name and moved into place when complete, so a copy that stops part-way (a full disk)
    leaves nothing under the file's name, and the next copy makes the file afresh."""
    root = EXAMPLES_FOLDER if root is None else root
    frame = example_frame(key, root)
    target.mkdir(parents=True, exist_ok=True)
    files = [path for path in sorted(frame.parent.iterdir()) if path.is_file() and not path.name.startswith(".")]
    copies = [(path, target / path.name) for path in files]
    if (root / "README.md").is_file():
        copies.append((root / "README.md", target / EXAMPLE_NOTE))
    for source, destination in copies:
        if destination.exists():
            continue  # kept: the user may have changed it (an edited .poni)
        partial = destination.with_name(f".{destination.name}.part")
        try:
            shutil.copy2(source, partial)
            os.replace(partial, destination)
        except OSError:
            partial.unlink(missing_ok=True)
            raise
    return target / frame.name


# -- How to Cite -------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Citation:
    text: str
    bibtex: str
    doi: str = ""
    """The DOI ``CITATION.cff`` gives (``""``: none; none is ever made up)."""


def _doi(entry: dict) -> str:
    doi = entry.get("doi")
    if not doi:
        for identifier in entry.get("identifiers") or ():
            if isinstance(identifier, dict) and identifier.get("type") == "doi" and identifier.get("value"):
                doi = identifier["value"]
                break
    return str(doi or "").strip()


def _people(entry: dict) -> list[dict]:
    return [person for person in entry.get("authors") or () if isinstance(person, dict)]


def _short_name(person: dict) -> str:
    """``Zhai, Y.`` (an organisation by its name)."""
    family, given = person.get("family-names", ""), person.get("given-names", "")
    if not family:
        return str(person.get("name", "")).strip()
    initials = " ".join(f"{part[0]}." for part in str(given).replace("-", " ").split() if part)
    return f"{family}, {initials}" if initials else str(family)


def _full_name(person: dict) -> str:
    family, given = person.get("family-names", ""), person.get("given-names", "")
    if not family:
        return "{" + str(person.get("name", "")).strip() + "}"
    return f"{family}, {given}".strip(", ")


def _year(entry: dict) -> str:
    """The year of ``date-released`` (or ``year``); ``""`` when the file gives none."""
    released = entry.get("date-released") or entry.get("year") or ""
    if isinstance(released, date):
        return str(released.year)
    year = str(released)[:4]
    return year if year.isdigit() else ""


def _reference(entry: dict, title: str, tail: str) -> str:
    """``Zhai, Y., & Other, A. (2026). <title><tail>`` — without a year when the file gives none."""
    names = [name for name in (_short_name(person) for person in _people(entry)) if name]
    authors = names[0] if len(names) == 1 else ", ".join(names[:-1]) + ", & " + names[-1] if names else ""
    year = _year(entry)
    head = " ".join(part for part in (authors, f"({year})" if year else "") if part)
    if not head:
        return f"{title}{tail}"
    return f"{head}{' ' if head.endswith('.') else '. '}{title}{tail}"


def _bibtex(kind: str, key: str, fields: list[tuple[str, str]]) -> str:
    lines = [f"@{kind}{{{key},"] + [f"  {name} = {{{value}}}," for name, value in fields if value]
    return "\n".join(lines) + "\n}"


def read_citation(path: Path = CITATION_FILE, version: str = __version__, url: str = GITHUB_URL) -> Citation:
    """The reference to GIMaP ``version`` from ``CITATION.cff`` (and its preferred article, when it names one);
    without a readable file, one built from the name, the version and the repository."""
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError):
        data = None
    if not isinstance(data, dict):
        text = f"{APP_NAME} (version {version}) [Computer software]. {url}"
        return Citation(text, _bibtex("software", "gimap", [("title", "{" + APP_NAME + "}"), ("version", version),
                                                            ("url", url)]))
    title = str(data.get("title") or APP_NAME).strip()
    url = str(data.get("repository-code") or data.get("url") or url).strip()
    doi = _doi(data)
    link = f" https://doi.org/{doi}" if doi else f" {url}"
    text = _reference(data, title, f" (version {version}) [Computer software].{link}")
    bibtex = _bibtex("software", "gimap", [
        ("author", " and ".join(_full_name(person) for person in _people(data))), ("title", "{" + title + "}"),
        ("version", version), ("year", _year(data)), ("url", url), ("doi", doi), ("license", str(data.get("license") or ""))])
    article = data.get("preferred-citation")
    if isinstance(article, dict) and article.get("title"):
        journal, volume = str(article.get("journal") or ""), str(article.get("volume") or "")
        pages = "–".join(str(article[name]) for name in ("start", "end") if article.get(name))
        where = ", ".join(part for part in (journal, volume, pages) if part)
        article_doi = _doi(article)
        tail = (f". {where}." if where else ".") + (f" https://doi.org/{article_doi}" if article_doi else "")
        text = _reference(article, str(article["title"]).strip(), tail) + "\n\n" + text
        bibtex = _bibtex("article", "gimap_article", [
            ("author", " and ".join(_full_name(person) for person in _people(article))),
            ("title", "{" + str(article["title"]).strip() + "}"), ("journal", journal), ("volume", volume),
            ("pages", pages.replace("–", "--")), ("year", _year(article)), ("doi", article_doi)]) + "\n\n" + bibtex
        doi = doi or article_doi
    return Citation(text, bibtex, doi)


# -- the commands --------------------------------------------------------------------------------------


def _documents() -> Path:
    documents = QStandardPaths.writableLocation(QStandardPaths.DocumentsLocation)
    return Path(documents) if documents else Path.home() / "Documents"


class HelpCommands:
    """The Help entries that need the window: its Analyze page, its user data folder, its toasts."""

    def __init__(self, menus):
        self.menus = menus  # ApplicationMenus: window, show_workspace

    @property
    def window(self):
        return self.menus.window

    def examples(self) -> tuple:
        return tuple((example.key, example.text, example.tip) for example in EXAMPLES)

    @staticmethod
    def default_folder(key: str) -> Path:
        """``Documents/GIMaP examples/<folder of the example>``."""
        return _documents() / "GIMaP examples" / key

    def open_example(self, key: str):
        """Ask where, copy the example there, open its frame in Analyze; the copied frame, or ``None``."""
        example = next(example for example in EXAMPLES if example.key == key)
        name = plain_text(tr(example.text)).rstrip("…")
        try:
            example_frame(key)
        except FileNotFoundError:
            warn(self.window, tr("Example Data"),
                 trf("The example data are not in this installation of GIMaP (looked in {folder}).",
                     folder=str(EXAMPLES_FOLDER)))
            return None
        chosen = ask_example_folder(self.window, name, str(self.default_folder(key)))
        if not chosen:
            return None
        target = Path(chosen).expanduser()
        if not target.is_absolute():  # a bare name: under Documents/GIMaP examples, never the working folder
            target = self.default_folder(key).parent / target
        try:
            frame = copy_example(key, target)
        except OSError as exc:
            warn(self.window, tr("Example Data"),
                 trf("The example could not be copied: {error}", error=exc.strerror or str(exc)))
            return None
        folder = (tr("Open Folder"), lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(target))))
        if self.window.components.analyze_page.import_paths([str(frame)]):
            self.menus.show_workspace("analyze")
            show_toast(self.window, trf("Copied the example to {folder} and opened it in Analyze.", folder=str(target)),
                       level="ok", action=folder)
        else:  # Analyze is busy (a batch, an import): the page says so; the copy is there
            show_toast(self.window, trf("Copied the example to {folder}. Analyze is busy: open {name} from there "
                                        "when it is free.", folder=str(target), name=frame.name),
                       level="warning", action=folder)
        return frame

    def log_folder(self) -> Path:
        """Where ``ErrorGuard`` writes ``errors.log`` (``main.py``: ``<user data>/logs``)."""
        return (Path(self.window.app_context.data_dir or ".") / "logs").resolve()

    def open_error_log(self) -> bool:
        folder = self.log_folder()
        if not (folder / LOG_NAME).is_file():
            inform(self.window, tr("Error Log"), tr("No error has been logged."))
            return False
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))
        return True

    def report_problem(self) -> str:
        """The versions to the clipboard (a toast says so), then a new issue on GitHub in the browser."""
        summary = system_summary()
        QApplication.clipboard().setText(summary)
        show_toast(self.window, tr("The versions of GIMaP and of your system (no paths, no data) are on the clipboard: "
                                   "paste them into the report."), level="ok")
        QDesktopServices.openUrl(QUrl(ISSUES_URL))
        return summary

    @staticmethod
    def citation() -> Citation:
        return read_citation()


__all__ = ["CITATION_FILE", "Citation", "EXAMPLES", "EXAMPLES_FOLDER", "Example", "HelpCommands", "copy_example",
           "example_frame", "read_citation"]
