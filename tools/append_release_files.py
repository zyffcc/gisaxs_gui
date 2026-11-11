"""Append the checked application/licence list without rewriting the packed runtime's ZIP entries.

Windows PowerShell's .NET ZIP update can change the local header of an existing empty deflated
file while retaining its compressed bytes. Standard Windows extraction then writes those two
bytes into an empty Python __init__.py. Python's append mode leaves existing entry bytes intact.
Standard library only; the builder supplies the same allow list it has already checked.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
from zipfile import ZIP_DEFLATED, ZipFile


def _relative(name: str) -> str:
    path = PurePosixPath(name)
    if (not name or path.is_absolute() or path.as_posix() != name
            or any(part in {'.', '..'} for part in path.parts) or '\\' in name or ':' in name):
        raise ValueError(f'Invalid release path: {name!r}')
    return name


def append_files(archive: Path, root: Path, package: str, licences: Path, selection: dict) -> None:
    package = _relative(package)
    files = [_relative(name) for name in selection['files']]
    licence_files = [_relative(name) for name in selection['licences']]
    generated = selection.get('generated', {})
    if not set(generated).issubset(files):
        raise ValueError('Generated files must belong to the checked application list')
    names = [f'{package}/{name}' for name in files] + [f'{package}/licenses/{name}' for name in licence_files]
    with ZipFile(archive, 'a', compression=ZIP_DEFLATED, allowZip64=True, strict_timestamps=False) as packed:
        known = {name.casefold() for name in packed.namelist()}
        for name in names:
            if name.casefold() in known:
                raise ValueError(f'Duplicate release entry: {name}')
            known.add(name.casefold())
        for name in files:
            target = f'{package}/{name}'
            if name in generated:
                packed.writestr(target, generated[name].encode('utf-8'))
            else:
                packed.write(root / name, target)
        for name in licence_files:
            packed.write(licences / name, f'{package}/licenses/{name}')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--archive', required=True, type=Path)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--package', required=True)
    parser.add_argument('--licences', required=True, type=Path)
    parser.add_argument('--selection', required=True, type=Path)
    args = parser.parse_args()
    selection = json.loads(args.selection.read_text(encoding='utf-8-sig'))
    append_files(args.archive, args.root, args.package, args.licences, selection)


if __name__ == '__main__':
    main()
