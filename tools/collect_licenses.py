"""Collect the licence texts of a Conda environment for the portable release (standard library only).

``tools/build_portable_release.ps1`` packs the Conda environment ``gimap-release`` into ``runtime\\`` of
the portable package. The wheels in it keep their licence files in their ``*.dist-info`` folders and
next to code they bundle; the Conda packages keep theirs in the package cache
(``pkgs/<package>/info/licenses``), outside the environment. Some wheels carry no licence text at all:
PyQt5 5.15.11 (GPL-3.0) is one. This helper copies all of them into one folder, which the builder puts
into the ZIP as ``licenses/``:

    README.txt                    every component: version, declared licence, the files copied here
    licenses.json                 the same, for programs
    THIRD_PARTY_NOTICES.md        GIMaP's notices (with --root)
    GIMaP/LICENSE                 GIMaP's own licence, MIT (with --root)
    python/<name>-<version>/...   the licence files of each Python distribution in site-packages
    conda/<name>-<version>/...    the licence files of each Conda package

A component that declares the GPL-3.0, the LGPL-3.0 or the Apache-2.0 but carries no text of it gets
the text copied from another component that has it (README.txt names which). The same is done, as far as
a text is found, for a GPL-3.0 or LGPL-3.0 that a component's own licence files name. When a component
declares the GPL-3.0 or the LGPL-3.0 and still has no text of it, or declares any GPL or LGPL and carries
no licence file at all, shipping it would break that licence: the helper reports a problem, and the
builder stops before it packs anything. Other GPL and LGPL versions (GPL-2.0, LGPL-2.1, ...) are not
checked for a full text: a component under one of them that carries no GNU licence text is listed in
``unchecked_copyleft`` (the builder warns; check it by hand against THIRD_PARTY_NOTICES.md).

THIRD_PARTY_NOTICES.md says the runtime's conda packages come from conda-forge: a conda package from
another channel (Anaconda's "defaults" comes under Anaconda's terms), or from an unknown one, is a problem.

    python tools/collect_licenses.py --prefix <environment> --out <new folder> [--root <GIMaP folder>]

The standard output is one JSON object (ASCII): ``ok``, ``environment``, ``python``, ``conda`` (counts),
``written`` (the files, relative to ``--out``), ``without_text``, ``filled``, ``unchecked_copyleft``,
``other_channels`` and ``problems``.
Exit code 0: collected; 1: a problem (listed in ``problems``); 2: a wrong command line.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import re
import shutil
import sys
from email.parser import HeaderParser
from pathlib import Path, PurePosixPath

LICENCE_NAME = re.compile(
    r"^(licen[cs]e|copying|copyright|notice|authors|third[-_ ]?party|patents)", re.IGNORECASE)
NOT_TEXT = {
    ".py", ".pyc", ".pyo", ".pyi", ".pyd", ".dll", ".so", ".dylib", ".exe", ".lib", ".a", ".h",
    ".hpp", ".c", ".cc", ".cpp", ".js", ".mjs", ".ts", ".css", ".html", ".htm", ".json", ".xml",
    ".png", ".jpg", ".gif", ".svg", ".pb", ".whl", ".zip",
}
TEXTS = {  # licence: the words its text starts with (white space joined, upper case)
    "GPL-3.0": "GNU GENERAL PUBLIC LICENSE VERSION 3, 29 JUNE 2007",
    "LGPL-3.0": "GNU LESSER GENERAL PUBLIC LICENSE VERSION 3, 29 JUNE 2007",
    "Apache-2.0": "APACHE LICENSE VERSION 2.0, JANUARY 2004",
}
NEEDS = {"GPL-3.0": ("GPL-3.0",), "LGPL-3.0": ("LGPL-3.0", "GPL-3.0"), "Apache-2.0": ("Apache-2.0",)}
"""The texts a licence must travel with: the LGPL-3.0 is a set of additions to the GPL-3.0."""
DECLARED = (
    ("LGPL-3.0", re.compile(r"\bLGPL\s*-?\s*v?3|LESSER GENERAL PUBLIC LICENSE\s*\(?v(ersion)?\s*3", re.I)),
    ("GPL-3.0", re.compile(
        r"(?<![LA])GPL\s*-?\s*v?3|(?<!LESSER )(?<!AFFERO )GENERAL PUBLIC LICENSE\s*\(?v(ersion)?\s*3", re.I)),
    ("Apache-2.0", re.compile(r"APACHE[^;]*?\b2(\.0)?\b", re.I)),
)
COPYLEFT = re.compile(r"GPL|GENERAL PUBLIC LICENSE", re.I)
GNU_TEXT = ("GNU GENERAL PUBLIC LICENSE", "GNU LESSER GENERAL PUBLIC LICENSE", "GNU LIBRARY GENERAL PUBLIC LICENSE")
"""How a GNU licence text starts (any version): a component carrying one is not listed as unchecked."""
CONDA_FORGE = re.compile(r"(^|/)conda-forge(/|$)")
MENTIONED = (
    ("LGPL-3.0", re.compile(r"\bLGPL-?v?3(\.0)?\b", re.I)),
    ("GPL-3.0", re.compile(r"\bGPL-?v?3(\.0)?\b", re.I)),
)
"""SPDX-like names of a licence inside a licence file, e.g. fabio's ``License: LGPL-3.0+``."""
READ_LIMIT = 4 * 1024 * 1024


def _licence_name(name: str) -> bool:
    return bool(LICENCE_NAME.match(name)) and PurePosixPath(name).suffix.lower() not in NOT_TEXT


def _safe(part: str) -> str:
    return re.sub(r"[^A-Za-z0-9._+-]", "_", part) or "_"


def _relative(text: str) -> PurePosixPath | None:
    path = PurePosixPath(text.replace("\\", "/"))
    if not path.parts or path.is_absolute() or ":" in text or any(p in {"", ".", ".."} for p in path.parts):
        return None
    return path


def _normalized(path: Path) -> str:
    try:
        with path.open("rb") as handle:
            data = handle.read(READ_LIMIT)
    except OSError:
        return ""
    return re.sub(r"\s+", " ", data.decode("utf-8", errors="replace")).upper()


def _families(declared: str) -> set[str]:
    return {name for name, pattern in DECLARED if pattern.search(declared)}


def _site_packages(prefix: Path) -> list[Path]:
    found = [prefix / "Lib" / "site-packages", *sorted(prefix.glob("lib/python3*/site-packages"))]
    unique = {}
    for folder in found:
        if folder.is_dir():
            unique.setdefault(str(folder.resolve()).lower(), folder)
    return list(unique.values())


def _declared(meta) -> str:
    if meta is None:
        return ""
    parts = []
    for key in ("License-Expression", "License"):
        text = (meta.get(key) or "").strip()
        if text and text.upper() != "UNKNOWN" and "\n" not in text and len(text) <= 100:
            parts.append(text)
    for classifier in meta.get_all("Classifier") or []:
        if classifier.startswith("License ::"):
            parts.append(classifier.split("::")[-1].strip())
    return "; ".join(dict.fromkeys(parts))


class _Collector:
    def __init__(self, prefix: Path, out: Path):
        self.prefix, self.out = prefix, out
        self.prefix_resolved = prefix.resolve()
        self.written: list[str] = []
        self.components: list[dict] = []

    def copy(self, source: Path, destination: str) -> str:
        target = self.out / destination
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        self.written.append(destination)
        return destination

    def add(self, kind: str, name: str, version: str, declared: str, sources: dict[str, Path], note: str = ""):
        folder = f"{kind}/{_safe(name)}-{_safe(version)}"
        files = [self.copy(source, f"{folder}/{relative}") for relative, source in sources.items()]
        self.components.append({
            "kind": kind, "name": name, "version": version, "declared": declared, "folder": folder,
            "files": files, "copied_texts": [], "note": note})

    def python(self, site: Path, folder: Path) -> None:
        meta = None
        for metadata in ("METADATA", "PKG-INFO"):
            if (folder / metadata).is_file():
                text = (folder / metadata).read_text(encoding="utf-8", errors="replace")
                meta = HeaderParser().parsestr(text)
                break
        stem = folder.name[: -len(folder.suffix)]
        guessed_name, _, guessed_version = stem.partition("-")
        name = (meta.get("Name") if meta else None) or guessed_name
        version = (meta.get("Version") if meta else None) or guessed_version.split("-")[0] or "unknown"
        sources: dict[str, Path] = {}
        seen: set[Path] = set()

        def take(relative: str, path: Path) -> None:
            key = path.resolve()
            if key not in seen and _relative(relative) is not None:
                seen.add(key)
                sources[relative] = path

        for value in (meta.get_all("License-File") or []) if meta else []:
            for base in (folder, folder / "licenses"):
                candidate = base / value
                if _relative(value) is not None and candidate.is_file():
                    take(candidate.relative_to(folder).as_posix(), candidate)
                    break
        for path in sorted(folder.rglob("*")):
            relative = path.relative_to(folder)
            if path.is_file() and (relative.parts[0] == "licenses"
                                   or (len(relative.parts) == 1 and _licence_name(path.name))):
                take(relative.as_posix(), path)
        record = folder / "RECORD"
        rows = []
        if record.is_file():
            rows = list(csv.reader(record.read_text(encoding="utf-8", errors="replace").splitlines()))
        for row in rows:
            entry = row[0] if row else ""
            if not entry or entry.startswith(folder.name + "/") or not _licence_name(entry.rsplit("/", 1)[-1]):
                continue
            if "__pycache__" in entry or ":" in entry or entry.startswith(("/", "\\")):
                continue
            target = (site / entry).resolve()
            try:
                inside = target.relative_to(self.prefix_resolved).as_posix()
            except ValueError:
                continue
            if target.is_file():
                take(entry if ".." not in PurePosixPath(entry).parts else f"prefix/{inside}", target)
        note = "" if meta is not None else "no metadata: an incomplete installation"
        self.add("python", name, version, _declared(meta), sources, note)

    def conda(self, record: dict) -> None:
        name, version = str(record.get("name") or "unknown"), str(record.get("version") or "unknown")
        sources: dict[str, Path] = {}
        folders = [record.get("extracted_package_dir"), (record.get("link") or {}).get("source")]
        tarball = record.get("package_tarball_full_path")
        if tarball:
            folders.append(re.sub(r"(\.conda|\.tar\.bz2)$", "", str(tarball)))
        for folder in folders:
            licences = Path(folder) / "info" / "licenses" if folder else None
            if licences is not None and licences.is_dir():
                for path in sorted(licences.rglob("*")):
                    if path.is_file():
                        sources[path.relative_to(licences).as_posix()] = path
                break
        for entry in record.get("files") or []:
            relative = _relative(str(entry))
            if relative is not None and _licence_name(relative.name) and "__pycache__" not in relative.parts:
                path = self.prefix / relative
                if path.is_file():
                    sources[f"runtime/{relative.as_posix()}"] = path
        self.add("conda", name, version, str(record.get("license") or ""), sources)

    def fill(self, problems: list[str], filled: list[str], unchecked: list[str]) -> None:
        """Copy the GPL, LGPL or Apache text to a component that declares it but does not carry it; list a
        component under another GPL or LGPL version without any GNU text in ``unchecked``."""
        texts = {relative: _normalized(self.out / relative) for relative in self.written}
        donors: dict[str, tuple[str, str]] = {}
        for component in self.components:
            for relative in component["files"]:
                head = texts[relative][:1000]
                for licence, words in TEXTS.items():
                    best = donors.get(licence)
                    if words in head and (best is None or len(texts[relative]) < len(texts[best[0]])):
                        donors[licence] = (relative, f"{component['name']} {component['version']}")
        for component in self.components:
            label = f"{component['name']} {component['version']}"
            present = {licence for licence, words in TEXTS.items()
                       if any(words in texts[relative] for relative in component["files"])}
            declared = _families(component["declared"])
            mentioned = {licence for licence, pattern in MENTIONED
                         for relative in component["files"] if pattern.search(texts[relative])}
            wanted = [text for licence in sorted(declared | mentioned) for text in NEEDS[licence]]
            reported = False
            for licence in dict.fromkeys(wanted):
                if licence in present:
                    continue
                donor = donors.get(licence)
                if donor is None:
                    if any(licence in NEEDS[family] for family in declared) and licence != "Apache-2.0":
                        problems.append(f"{label} declares {component['declared']!r} but carries no "
                                        f"{licence} text, and no other package has one")
                        reported = True
                    continue
                destination = self.copy(self.out / donor[0], f"{component['folder']}/{licence}.txt")
                texts[destination] = texts[donor[0]]
                component["files"].append(destination)
                component["copied_texts"].append(
                    {"file": destination, "licence": licence, "from": f"{donor[1]}: {donor[0]}"})
                present.add(licence)
                filled.append(f"{label}: {licence} text from {donor[1]}")
            if not component["files"] and not reported and COPYLEFT.search(component["declared"]):
                problems.append(f"{label} declares {component['declared']!r} but carries no licence text")
            elif COPYLEFT.search(component["declared"]) and not declared and not any(
                    words in texts[relative][:1000] for relative in component["files"] for words in GNU_TEXT):
                unchecked.append(f"{label} ({component['declared']})")


def _readme(collector: _Collector, environment: str, with_gimap: bool) -> str:
    lines = ["Licence texts of the components of this package", "=" * 47, ""]
    if with_gimap:
        lines += [
            "GIMaP itself is released under the MIT License: GIMaP\\LICENSE. THIRD_PARTY_NOTICES.md names",
            "the parts that come under other licences (among them the data in tests\\data\\external) and",
            "explains why this Windows package is distributed under the terms of the GPL-3.0 for its",
            "GPL-3.0 components (PyQt5, BornAgain), and where their source is.", ""]
    lines += [
        f"runtime\\ is a packed Conda environment ({environment}). Below is every Python distribution",
        "(licence files in python\\<name>-<version>\\) and every Conda package (conda\\<name>-<version>\\)",
        "in it, with the licence its metadata declares. The files in runtime\\ are not changed.",
        f"Written by tools/collect_licenses.py on {datetime.date.today().isoformat()}.", ""]
    for kind, title in (("python", "Python distributions (site-packages)"), ("conda", "Conda packages")):
        components = [c for c in collector.components if c["kind"] == kind]
        lines += [f"{title}: {len(components)}", "-" * (len(title) + 2 + len(str(len(components)))), ""]
        for component in sorted(components, key=lambda c: c["name"].lower()):
            declared = component["declared"] or "no licence declared in the metadata"
            lines.append(f"{component['name']} {component['version']} -- {declared}")
            copied = {text["file"]: text for text in component["copied_texts"]}
            for relative in component["files"]:
                shown = relative.replace("/", "\\")
                if relative in copied:
                    shown += (f"  (the package carries no {copied[relative]['licence']} text; copied from "
                              f"{copied[relative]['from'].replace('/', chr(92))})")
                lines.append(f"    {shown}")
            if not component["files"]:
                lines.append("    (no licence file in the installed package)")
            if component["note"]:
                lines.append(f"    ({component['note']})")
        lines.append("")
    return "\r\n".join(lines) + "\r\n"


def collect(prefix: Path, out: Path, root: Path | None = None) -> dict:
    """Write the licence folder ``out`` for the environment ``prefix``; return the summary."""
    problems: list[str] = []
    filled: list[str] = []
    unchecked: list[str] = []
    other_channels: list[str] = []
    summary = {"ok": False, "environment": prefix.name, "python": 0, "conda": 0, "written": [],
               "without_text": [], "filled": filled, "unchecked_copyleft": unchecked,
               "other_channels": other_channels, "problems": problems}
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        problems.append(f"The output folder is not empty: {out}")
        return summary
    sites = _site_packages(prefix)
    if not sites:
        problems.append(f"No site-packages folder in {prefix}: is it a Conda environment?")
        return summary
    out.mkdir(parents=True, exist_ok=True)
    collector = _Collector(prefix, out)
    if root is not None:
        for name, destination in (("THIRD_PARTY_NOTICES.md", "THIRD_PARTY_NOTICES.md"), ("LICENSE", "GIMaP/LICENSE")):
            if (root / name).is_file():
                collector.copy(root / name, destination)
            else:
                problems.append(f"{name} is missing in {root}: the package must carry it")
    for site in sites:
        for folder in sorted(site.iterdir(), key=lambda path: path.name.lower()):
            if folder.is_dir() and not folder.name.startswith("~") and folder.suffix in {".dist-info", ".egg-info"}:
                collector.python(site, folder)
    for path in sorted((prefix / "conda-meta").glob("*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            problems.append(f"Cannot read conda-meta/{path.name}: {error}")
            continue
        if isinstance(record, dict):
            collector.conda(record)
            channel = str(record.get("schannel") or record.get("channel") or "")
            if not CONDA_FORGE.search(channel):
                other_channels.append(f"{record.get('name')} {record.get('version')} from "
                                      f"{channel or 'an unknown channel'}")
    if other_channels:
        shown = ", ".join(other_channels[:5]) + (", ..." if len(other_channels) > 5 else "")
        problems.append(
            f"{len(other_channels)} conda package(s) are not from conda-forge ({shown}): THIRD_PARTY_NOTICES.md "
            "says the runtime's conda packages come from conda-forge; create the environment again with "
            "conda create -n gimap-release -c conda-forge --override-channels python=3.10 -y")
    collector.fill(problems, filled, unchecked)
    for component in collector.components:
        if not component["files"]:
            declared = component["declared"] or "no licence declared"
            summary["without_text"].append(f"{component['name']} {component['version']} ({declared})")
    (out / "README.txt").write_bytes(_readme(collector, prefix.name, root is not None).encode("utf-8"))
    (out / "licenses.json").write_text(json.dumps({
        "generated_by": "tools/collect_licenses.py", "date": datetime.date.today().isoformat(),
        "environment": prefix.name, "components": collector.components}, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8")
    collector.written += ["README.txt", "licenses.json"]
    summary.update({
        "ok": not problems, "written": sorted(collector.written),
        "python": sum(c["kind"] == "python" for c in collector.components),
        "conda": sum(c["kind"] == "conda" for c in collector.components)})
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Collect the licence texts of a Conda environment.")
    parser.add_argument("--prefix", required=True, type=Path, help="the environment to be packed")
    parser.add_argument("--out", required=True, type=Path, help="a new (or empty) folder for the texts")
    parser.add_argument("--root", type=Path, help="GIMaP's folder: adds THIRD_PARTY_NOTICES.md and LICENSE")
    arguments = parser.parse_args(argv)
    summary = collect(arguments.prefix, arguments.out, arguments.root)
    sys.stdout.write(json.dumps(summary, ensure_ascii=True) + "\n")
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
