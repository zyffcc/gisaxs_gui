"""Compare archived recovery draws with a new set, without fitting a cut."""
from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
from itertools import combinations
import json
from pathlib import Path

if __package__:
    from . import heldout_cases
    from .case_streams import SETS
    from .recovery_cases import draw
else:
    import heldout_cases
    from case_streams import SETS
    from recovery_cases import draw

HERE = Path(__file__).resolve().parent
FIELDS = ("full_tuple", "populations", "first_population_draw", "noise_seed")
DIGITS = 9
"""Significant digits of the drawn numbers that are compared. The same draw on another computer can differ in the
last bits (R and D come from exp: up to 3 units in the last place between Windows and Linux); two different draws
do not agree to 9 digits."""


def _digits(value):
    return float(f"{value:.{DIGITS}g}") if isinstance(value, float) else value


def fingerprints(case):
    """The identifying tuple, plus its parts to detect partially repeated draws (numbers to ``DIGITS`` digits)."""
    populations = tuple(tuple(sorted((key, _digits(value)) for key, value in population.items()))
                        for population in case["populations"])
    first = case["populations"][0]
    first_draw = tuple(_digits(first[key]) for key in ("R", "sR", "D", "s_nn"))
    noise = case["seed"]
    return ((case["composition"], populations, noise), populations, first_draw, noise)


def load_cases(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return [row["case"] for row in json.load(stream)["results"]]


def archived_sets(directory):
    """Read per-cut records and reconstruct earlier posterior sets stored only as summaries."""
    archives = {p.name: load_cases(p) for p in sorted(directory.glob("posterior_*.json.gz"))}
    history = directory.parent / "history"
    for path in sorted(history.glob("posterior_*_summary.json")):
        run = json.loads(path.read_text(encoding="utf-8"))
        generate, name = {"fit1d": (draw, "legacy-first"),
                          "extended": (heldout_cases.draw, "legacy-extended")}[run["set"]]
        archives[f"history/{path.name} (reconstructed)"] = [
            generate(i, run["seed0"], set_name=name) for i in range(run["n"])]
    return archives


def _indexed(cases):
    if not cases or len({case["index"] for case in cases}) != len(cases):
        raise ValueError("Expected a nonempty set with distinct case indices")
    indices = [defaultdict(list) for _ in FIELDS]
    for case in cases:
        for lookup, key in zip(indices, fingerprints(case)):
            lookup[key].append(case["index"])
    return indices


def compare(archives, new_cases):
    """Group archives with identical draws; report matching pairs of cut indices for each field."""
    groups = defaultdict(list)
    for label, cases in archives.items():
        signature = tuple(sorted((c["index"], fingerprints(c)) for c in cases))
        groups[signature].append(label)
    sets = {", ".join(labels): archives[labels[0]] for labels in groups.values()}
    sets["new"] = new_cases
    indexed = {name: _indexed(cases) for name, cases in sets.items()}
    pairs = {}
    for left, right in [*combinations(indexed, 2), ("new", "new")]:
        fields = {}
        for field, a, b in zip(FIELDS, indexed[left], indexed[right]):
            fields[field] = sorted((i, j) for key in a.keys() & b.keys()
                                   for i in a[key] for j in b[key] if left != right or i < j)
        pairs[left, right] = fields
    return sets, pairs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-dir", type=Path, default=HERE / "per_cut")
    parser.add_argument("--new-archive", type=Path, help="use actual new per-cut records in JSON or gzip")
    parser.add_argument("--n", type=int, help="generate this many fresh cases (default: 300)")
    parser.add_argument("--indices", action="store_true", help="print each matching pair of case indices")
    parser.add_argument("--strict", action="store_true", help="fail for historical reuse as well")
    args = parser.parse_args(argv)
    if args.new_archive and args.n is not None:
        parser.error("--n applies only when generating cases")
    archives = archived_sets(args.archive_dir)
    if args.new_archive and args.new_archive.resolve().parent == args.archive_dir.resolve():
        archives.pop(args.new_archive.name, None)
    if not archives:
        parser.error("no archived posterior runs found")
    n, seed = SETS["fresh"]
    if args.n is not None and args.n <= 0:
        parser.error("--n must be positive")
    new_cases = (load_cases(args.new_archive) if args.new_archive else
                 [draw(i, seed, set_name="fresh") for i in range(args.n or n)])
    sets, pairs = compare(archives, new_cases)
    labels = {group: str(i + 1) for i, group in enumerate(sets)}
    for name, cases in sets.items():
        print(f"[{labels[name]}] {len(cases)} cuts: {name}")
        if ", " in name:
            print("    All identifying tuples and indices coincide within this archive group.")
    print("set A / set B   " + "   ".join(FIELDS))
    failed = args.strict and len(sets) - 1 < len(archives)
    for (left, right), fields in pairs.items():
        print(f"{labels[left]} / {labels[right]}   " + "   ".join(str(len(v)) for v in fields.values()))
        if args.indices:
            for field, matches in fields.items():
                if matches:
                    print(f"    {field}: {matches}")
        if any(fields.values()) and (right == "new" or args.strict):
            failed = True
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
