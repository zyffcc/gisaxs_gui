"""Recompute recorded results through their feature's numerical pipeline and compare each column."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import sys
import tempfile
import zlib
from concurrent.futures import CancelledError
from pathlib import Path

import numpy as np

from src.gimap.shared.reproduction import compare_column, numerical_environment, unpack_column, verified_path, verify_run
from src.gimap.shared.revision import source_commit
from src.gimap.shared.version import SOFTWARE


def _check_cancel(should_cancel):
    if should_cancel and should_cancel():
        raise CancelledError()


def _write_table(path, columns, record_path, table, provenance):
    with path.open("x", encoding="utf-8", newline="") as stream:
        stream.write("# source_record: " + json.dumps(str(record_path), ensure_ascii=False) + "\n")
        stream.write("# table: " + json.dumps(table, ensure_ascii=False) + "\n")
        if provenance:
            stream.write("# source: " + json.dumps(provenance, ensure_ascii=False) + "\n")
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(columns)
        for row in zip(*columns.values(), strict=True):
            writer.writerow([format(float(value), ".17g") if isinstance(value, (float, np.floating))
                             else int(value) for value in row])


def _replay_batch(record, record_path, folder, should_cancel):
    recipe = record["reproduction"]
    items = recipe["records"]
    if not items:
        raise ValueError("The batch record contains no results to reproduce.")
    # Only x and intensity from the requested frames participate in a common-grid table.
    retained = [{} for _ in items]
    for name, indices in recipe.get("table_frames", {}).items():
        for index in indices:
            retained[index][name] = None
    results = []
    for i, item in enumerate(items):
        child_dir = f"result_{i + 1:04d}"
        child_record = record_path
        if "record_file" in item:
            child_record = verified_path(item["record_file"], record_path.parent)
            item = json.loads(child_record.read_text(encoding="utf-8-sig"))
        result = _replay(item, child_record, folder / child_dir, should_cancel, retain=retained[i])
        results.append({**result, "folder": child_dir})
    comparison = {"kind": "batch", "results": results, "status": "identical"}
    if recipe.get("reference"):
        from src.gimap.features.analyze.application.batch_tables import reproduced_batch_tables
        tables = reproduced_batch_tables(record, retained)
        comparison.update(_compare_tables(record, tables, record_path, folder))
    if any(r["status"] != "identical" for r in results):
        comparison["status"] = "different"
    return comparison


def _replay(record, record_path, folder, should_cancel, *, retain=None):
    _check_cancel(should_cancel)
    recipe = record.get("reproduction") if isinstance(record, dict) else None
    if not isinstance(recipe, dict):
        raise ValueError("This record has no reproduction recipe. Export the result again with this version of GIMaP.")
    if recipe.get("schema") != 1:
        raise ValueError(f"Unsupported reproduction schema: {recipe.get('schema')}")
    folder.mkdir(exist_ok=True)
    kind = recipe["kind"]
    if kind == "batch":
        return _replay_batch(record, record_path, folder, should_cancel)
    if kind == "automatic":
        from .reproduce_automatic import replay_automatic
        return replay_automatic(record, record_path, folder, should_cancel)
    if kind == "reduction":
        from src.gimap.features.analyze.infrastructure.reproduce import replay_reduction
        tables = replay_reduction(record, record_path.parent)
    elif kind == "series":
        from src.gimap.features.analyze.infrastructure.reproduce_series import replay_series
        tables = replay_series(record, record_path.parent, should_cancel=should_cancel)
    elif kind == "fit":
        from src.gimap.features.fitting.infrastructure.reproduce_fit import replay_fit
        tables = replay_fit(record, record_path.parent, should_cancel=should_cancel)
    else:
        raise ValueError(f"Unknown reproduction kind: {kind}")
    _check_cancel(should_cancel)
    for name in retain or ():
        retain[name] = {key: tables[name][key] for key in ("x", "intensity")}
    return {"kind": kind, **_compare_tables(record, tables, record_path, folder)}


def _compare_tables(record, tables, record_path, folder):
    recipe = record["reproduction"]
    reference = recipe["reference"]
    if not reference:
        raise ValueError("The record has no reference columns for comparison.")
    report = {}
    identical = True
    for i, name in enumerate(dict.fromkeys([*reference, *tables]), 1):
        columns = tables.get(name, {})
        expected = reference.get(name, {})
        if not columns or not expected:
            identical = False
        file = None
        if columns:
            # Names originate in the input record. Only the descriptive part of our filename comes from them.
            label = re.sub(r"[^\w.-]+", "_", name)[:70]
            file = f"reproduced_{i:02d}_{label}.csv"
            provenance = {key: record[key] for key in ("source_file", "frame_index", "output_intensity_unit")
                          if key in record}
            for curve in record.get("curves") or ():
                if isinstance(curve, dict) and curve.get("key") == name:
                    provenance.update({key: curve[key] for key in ("title", "x_label", "region") if key in curve})
                    break
            _write_table(folder / file, columns, record_path, name, provenance)
        differences = {}
        for key in dict.fromkeys([*expected, *columns]):
            if key in expected and key in columns:
                differences[key] = compare_column(unpack_column(expected[key]), columns[key])
            else:
                reason = "missing reproduced column" if key in expected else "extra reproduced column"
                differences[key] = {"status": "different", "reason": reason, "max_relative_difference": None}
            if differences[key]["status"] != "identical":
                identical = False
        report[name] = {"file": file, "columns": differences}
    return {"status": "identical" if identical else "different", "tables": report,
            "recorded_version": recipe.get("software_version"), "recorded_commit": recipe.get("commit")}


def _publish(staged, out):
    """Add complete results to the destination. Existing exports and inputs are never replaced."""
    created = []
    try:
        # Publish the comparison last. Its presence means all the referenced CSVs are in place.
        for path in sorted(staged.rglob("*"), key=lambda p: (p == staged / "comparison.json", p)):
            dest = out / path.relative_to(staged)
            if path.is_dir():
                dest.mkdir()
            else:
                with dest.open("xb") as stream:
                    created.append(dest)
                    with path.open("rb") as source:
                        shutil.copyfileobj(source, stream)
                continue
            created.append(dest)
    except BaseException:
        for dest in reversed(created):
            if dest.is_dir():
                dest.rmdir()
            else:
                dest.unlink()
        raise


def reproduce_record(record_path, out=None, *, should_cancel=None) -> dict:
    path = Path(record_path).resolve()
    original = path.read_bytes()
    record_digest = hashlib.sha256(original).hexdigest()
    record = json.loads(original)
    del original
    destination = Path(out).resolve() if out is not None else path.with_name(path.stem + "_reproduced")
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".gimap-reproduce-", dir=destination) as temporary:
        staged = Path(temporary)
        try:
            with verify_run(should_cancel) as relocated:
                comparison = _replay(record, path, staged, should_cancel)
        except (KeyError, TypeError, IndexError, AttributeError, OverflowError, zlib.error) as exc:
            raise ValueError(f"Invalid reproduction record {path.name}: {exc}") from exc
        if relocated:
            comparison["relocated_inputs"] = relocated
        comparison.update(source_record=str(path), source_record_sha256=record_digest,
                          output_dir=str(destination), software_version=SOFTWARE,
                          commit=source_commit(), runtime=numerical_environment())
        (staged / "comparison.json").write_text(
            json.dumps(comparison, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        _check_cancel(should_cancel)
        _publish(staged, destination)
    return comparison


def reproduce_cli(argv):
    parser = argparse.ArgumentParser(prog="gimap reproduce",
                                     description="Repeat a recorded reduction, fit or automatic analysis and compare its results.")
    parser.add_argument("record", type=Path, metavar="RECORD.json")
    parser.add_argument("--out", type=Path, metavar="FOLDER", help="output folder (default: <record stem>_reproduced)")
    args = parser.parse_args(argv)
    try:
        report = reproduce_record(args.record, args.out)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"gimap reproduce: {exc}", file=sys.stderr)
        return 1
    print(f"{report['status']}  {Path(report['output_dir']) / 'comparison.json'}")
    return 0 if report["status"] == "identical" else 2
