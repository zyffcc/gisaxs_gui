"""First start of the portable release: write this folder into the packed environment, in little memory.

conda-pack's own conda-unpack reads every listed file whole before it looks for the old path. In the
0.0.2 release that meant one 698 MB TensorFlow library at once (three copies of it), and on a computer
with little free memory or no paging file the first start stopped with "Not enough memory resources are
available to process this command". This script uses the same list and conda-unpack's own
``update_prefix``, but looks for the old path through a memory map first: a file without it (nearly all
large binaries) is never read into memory, and the few files that have it (scripts, small launchers,
configuration such as qt.conf) are rewritten exactly as conda-unpack does.

Run with the packed Python, from the GIMaP folder (Start_GIMaP.bat does this once):

    runtime\\python.exe tools\\portable_unpack.py            prepare runtime\\ for this folder
    runtime\\python.exe tools\\portable_unpack.py --check    only count what would be rewritten

Exit code: 0 done, 1 failed (the reason is printed), 2 not enough memory.
Standard library only: it runs before anything else of the environment is usable.
"""

from __future__ import annotations

import argparse
import ast
import mmap
import os
import sys
from pathlib import Path

RECORDS_START = "_prefix_records = ["
SCRIPT_NAME = "conda-unpack-script.py"


def read_unpack_script(script: Path):
    """conda-unpack's functions (the code before the list) and an iterator over the list's entries.

    The list (64 000 entries in 0.0.2) is read line by line, never held whole in memory.
    """
    header: list[str] = []
    with script.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith(RECORDS_START):
                break
            header.append(line)
        else:
            raise ValueError(f"{script.name} has no '{RECORDS_START}' line: not a conda-unpack script.")
    namespace: dict = {"__name__": "conda_unpack", "__file__": str(script)}
    exec(compile("".join(header), str(script), "exec"), namespace)

    def records():
        with script.open("r", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith(RECORDS_START):
                    break
            for line in fh:
                text = line.strip()
                if text.startswith("]"):
                    return
                if not text:
                    continue
                path, placeholder, mode = ast.literal_eval(text.rstrip(","))
                yield path, placeholder, mode

    return namespace, records


def count_records(script: Path) -> int:
    total = 0
    with script.open("r", encoding="utf-8") as fh:
        inside = False
        for line in fh:
            if not inside:
                inside = line.startswith(RECORDS_START)
                continue
            if line.lstrip().startswith("]"):
                break
            total += line.lstrip().startswith("(")
    return total


def needles(placeholder: str, mode: str, on_win: bool) -> tuple[bytes, ...]:
    """The bytes whose presence makes conda-unpack change a file (see its text/binary replace)."""
    exact = placeholder.encode("utf-8")
    if mode == "binary" and on_win:
        return (exact, placeholder.lower().encode("utf-8"))
    return (exact,)


def contains_any(path: str, wanted: tuple[bytes, ...]) -> bool:
    """True when the file holds one of the byte strings, read through a memory map (no copy)."""
    with open(path, "rb") as fh:
        if os.fstat(fh.fileno()).st_size == 0:
            return False
        with mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as view:
            return any(view.find(item) != -1 for item in wanted)


def unpack(runtime: Path, check_only: bool = False, out=None) -> dict:
    out = sys.stdout if out is None else out
    scripts = runtime / "Scripts"
    script = scripts / SCRIPT_NAME
    if not script.is_file():
        raise FileNotFoundError(f"{script} was not found: not a portable GIMaP environment.")
    sys.path.insert(0, str(scripts))  # conda-unpack imports its progress bar from there
    namespace, records = read_unpack_script(script)
    update_prefix = namespace.get("update_prefix")
    if not callable(update_prefix):
        raise ValueError(f"{script.name} defines no update_prefix: not a conda-unpack script of conda-pack.")
    on_win = bool(namespace.get("on_win", os.name == "nt"))
    new_prefix = os.path.abspath(str(runtime))
    total = count_records(script) or 1
    stats = {"listed": 0, "missing": 0, "changed": 0, "largest_changed": 0}
    step = max(1, total // 20)
    for index, (path, placeholder, mode) in enumerate(records(), start=1):
        stats["listed"] += 1
        target = os.path.join(new_prefix, path)
        if not os.path.isfile(target):
            stats["missing"] += 1
        elif contains_any(target, needles(placeholder, mode, on_win)):
            stats["changed"] += 1
            stats["largest_changed"] = max(stats["largest_changed"], os.path.getsize(target))
            if not check_only:
                update_prefix(target.replace("\\", "/") if on_win else target, new_prefix, placeholder, mode=mode)
        if index % step == 0 or index == total:
            out.write(f"\r[GIMaP] Preparing the environment: {min(100, 100 * index // total)} %")
            out.flush()
    out.write("\n")
    return stats


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runtime", type=Path, default=Path(__file__).resolve().parents[1] / "runtime",
                        help="the packed environment (default: runtime beside the tools folder)")
    parser.add_argument("--check", action="store_true", help="count the files to rewrite, change nothing")
    args = parser.parse_args(argv)
    try:
        stats = unpack(args.runtime, check_only=args.check)
    except MemoryError:
        print("\n[GIMaP] Not enough memory to prepare the environment. Close other programs, check that "
              "virtual memory (the paging file) is on, and start again.")
        return 2
    except Exception as exc:  # the reason is shown; Start_GIMaP.bat then does not mark the folder ready
        print(f"\n[GIMaP] The environment could not be prepared: {exc}")
        return 1
    verb = "would rewrite" if args.check else "rewrote"
    print(f"[GIMaP] {stats['listed']} files listed, {verb} {stats['changed']} "
          f"(largest {stats['largest_changed'] / 2**20:.1f} MB), {stats['missing']} not in this package.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
