"""GIMaP for command-line agents (Codex, Claude Code, scripts): GIWAXS and GISAXS analysis without the window.

Read docs/agents/giwaxs-playbook.md (GIWAXS) or docs/agents/gisaxs-playbook.md (GISAXS) first.
The usual call is one command:

    python tools/gimap_agent.py auto FRAME [FRAME ...] [--incidence-deg 0.4] [--notes "..."]

It runs the procedure of the technique GIMaP's Auto detection gives each frame once its
geometry is applied (GIWAXS: peaks, orientation, sizes; GISAXS: Yoneda cut, halves,
spacing, fit); --technique giwaxs|gisaxs forces one.  The notes are the person's own
words: αi, the energy (keV) and the pixel size (µm) are read when the notes give exactly
one value, a calibration file they name (.poni, GIMaP .json, an image of a standard) is
tried before the search, and the calibrant (AgBH, LaB6, CeO2, LaB6+CeO2) is used when
they name exactly one.  Options beat the notes and the image header.

Other commands:

    status FRAME              what GIMaP sees: detector, frames, header, geometry or not
    find-calibration FRAME    calibration files, standard images and logs around the frame, ranked
    tools                     the tool definitions (for `call`)
    call FRAME STEPS          run a JSON list of {"tool": ..., "args": {...}} on one frame
    mcp                       the same tools as an MCP server on stdio (for Codex / Claude Code)

Nothing is written next to the data: results go to --out (default: the GIMaP data
folder, assistant_runs/cli/<time>), and export_results writes to gimap_analysis/ there.

Exit codes (auto):
  0  every frame was analysed and nothing is open (Status OK).
  2  at least one frame has an open item in needs_attention (report.md "Needs attention"):
     a missing value with the flag that supplies it (e.g. --incidence-deg, --energy-kev,
     --technique), a judgement no flag settles (e.g. the GISAXS model choice), or nothing
     analysed because no geometry was found (--calibration).  Answer what you can and run
     again.
  1  at least one frame failed (a missing file, a file GIMaP cannot read, an error); the
     other frames still have their reports.  1 wins over 2.
  Other commands: 1 when the frame file does not exist; status and call also when GIMaP
  cannot read it, find-calibration when the search fails, call when a step returned an
  error; else 0.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

EXIT_DONE, EXIT_FAILED, EXIT_QUESTIONS = 0, 1, 2


def _utf8_console() -> None:
    # A Chinese Windows console is GBK: Å⁻¹, χ and αi would crash print().
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def _notes(arguments) -> str:
    parts = [arguments.notes or ""]
    if arguments.notes_file:
        parts.append(Path(arguments.notes_file).read_text(encoding="utf-8", errors="replace"))
    return "\n".join(part for part in parts if part)


def _print_json(data) -> None:
    print(json.dumps(data, indent=1, ensure_ascii=False, default=str))


def _outcome_data(outcome) -> dict:
    try:
        return json.loads(outcome.content) if isinstance(outcome.content, str) else {"content": outcome.content}
    except ValueError:
        return {"text": outcome.content}


def command_auto(arguments) -> int:
    from src.gimap.app.headless_assistant import analyse_frames, default_output_folder
    from src.gimap.features.assistant.application import PipelineOptions, batch_markdown

    options = PipelineOptions(
        calibration=arguments.calibration, standard=arguments.standard, energy_kev=arguments.energy_kev,
        incidence_deg=arguments.incidence_deg, pixel_size_um=arguments.pixel_size_um, frame=arguments.frame,
        sum_frames=arguments.sum,
        rings=arguments.rings, recalibrate=arguments.recalibrate, notes=_notes(arguments),
        technique=arguments.technique, follow_detection=arguments.technique is None, fit=not arguments.no_fit,
    )
    out = Path(arguments.out) if arguments.out else default_output_folder()
    progress = (lambda _text: None) if arguments.quiet else (lambda text: print(text, file=sys.stderr, flush=True))
    reports = analyse_frames(arguments.frames, options, out, progress=progress, saved_profiles=not arguments.no_saved_profiles)
    print(batch_markdown(reports))
    for report in reports:
        folder = Path(report["outputs"][0]).parent if report.get("outputs") else None
        if folder is not None:
            print(f"- {Path(report.get('frame') or '').name}: {folder / 'report.md'}")
    print(f"\nAll outputs: {out}")
    return exit_code(reports)


def exit_code(reports: list[dict]) -> int:
    """1 when a frame failed (an error, a file GIMaP could not read), else 2 when anything is open, else 0."""
    if any(_failed(report) for report in reports):
        return EXIT_FAILED
    return EXIT_QUESTIONS if any(report.get("needs_attention") for report in reports) else EXIT_DONE


def _failed(report: dict) -> bool:
    return bool(report.get("error") or report.get("failed") or (not report.get("ok") and not report.get("needs_attention")))


def _session(arguments):
    from src.gimap.app.headless_assistant import open_headless_session

    return open_headless_session(
        arguments.frame, notes=_notes(arguments), saved_profiles=not arguments.no_saved_profiles,
        out_dir=getattr(arguments, "out", None),
    )


def command_status(arguments) -> int:
    session = _session(arguments)
    try:
        _print_json(_outcome_data(session.call("get_status")))
        unreadable = session.unreadable()
    finally:
        session.close()
    return EXIT_FAILED if unreadable else EXIT_DONE


def command_find_calibration(arguments) -> int:
    session = _session(arguments)
    try:
        outcome = session.call("find_calibration_files", {"max_results": arguments.max_results})
        _print_json(_outcome_data(outcome))
    finally:
        session.close()
    return EXIT_FAILED if outcome.is_error else EXIT_DONE


def command_tools(_arguments) -> int:
    from src.gimap.app.headless_assistant import mcp_tools

    _print_json(mcp_tools())
    return EXIT_DONE


def command_call(arguments) -> int:
    text = sys.stdin.read() if arguments.steps == "-" else Path(arguments.steps).read_text(encoding="utf-8")
    steps = json.loads(text)
    session = _session(arguments)
    failed = False
    try:
        for number, step in enumerate(steps, 1):
            outcome = session.call(step["tool"], step.get("args") or step.get("arguments") or {})
            failed = failed or outcome.is_error
            print(json.dumps({
                "step": number, "tool": step["tool"], "summary": outcome.summary, "error": outcome.is_error,
                "result": _outcome_data(outcome),
            }, ensure_ascii=False, default=str), flush=True)
        if arguments.out:
            from src.gimap.app.headless_assistant import write_outputs
            from src.gimap.features.assistant.application import results_payload

            report = {"ok": not failed, "frame": arguments.frame, "steps": steps, "tables": results_payload(session.results)}
            write_outputs(session, report, Path(arguments.out))
        failed = failed or bool(session.unreadable())
    finally:
        session.close()
    return EXIT_FAILED if failed else EXIT_DONE


def command_mcp(arguments) -> int:
    from src.gimap.app.headless_assistant import serve_mcp_stdio

    serve_mcp_stdio(saved_profiles=not arguments.no_saved_profiles)
    return EXIT_DONE


def parser() -> argparse.ArgumentParser:
    main = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    notes_help = (
        "the person's notes, verbatim: αi, energy (keV) and pixel size (µm) when they give exactly one value; "
        "a calibration file named there (.poni, GIMaP .json, standard image) is tried first; the calibrant "
        "(AgBH, LaB6, CeO2, LaB6+CeO2) when exactly one is named; every path named becomes readable"
    )
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--no-saved-profiles", action="store_true", help="ignore instrument profiles saved in GIMaP")
    commands = main.add_subparsers(dest="command", required=True)

    auto = commands.add_parser(
        "auto", parents=[common], help="the standard analysis (GIWAXS or GISAXS, as Auto detects) of one or more frames",
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    auto.add_argument("frames", nargs="+", help="detector images (for a NeXus series: any one module file, e.g. *_m01.nxs)")
    auto.add_argument(
        "--calibration",
        help="a .poni, GIMaP calibration .json or image of a standard: tried first, then files the notes name, "
             "the instrument profile and the search (a calibration given but not used is flagged)",
    )
    auto.add_argument(
        "--standard", choices=("agbh", "lab6", "ceo2", "lab6_ceo2"),
        help="the standard in calibration images (default: the one calibrant the notes name, else the file name, "
             "else compared)",
    )
    auto.add_argument("--energy-kev", type=float, help="X-ray energy in keV (beats the header and the notes)")
    auto.add_argument("--incidence-deg", type=float, help="incidence angle αi in degrees (beats the notes and the profile)")
    auto.add_argument(
        "--pixel-size-um", type=float,
        help="detector pixel size in µm: used even when the image header gives another (Decisions names both)",
    )
    auto.add_argument("--frame", type=int, help="frame of a series (1-based, -1 = last; default: the last frames)")
    auto.add_argument("--sum", type=int, help="frames summed (default 10 for a series)")
    auto.add_argument("--rings", type=int, default=3, help="strongest reliable peaks analysed for orientation and size")
    auto.add_argument("--recalibrate", action="store_true", help="calibrate even when a saved profile exists")
    auto.add_argument(
        "--technique", choices=("giwaxs", "gisaxs"),
        help="force a procedure (default: what GIMaP's Auto detection gives once the geometry is applied; "
             "GIWAXS, flagged, when it cannot tell)",
    )
    auto.add_argument("--no-fit", action="store_true", help="GISAXS: prepare the cut but do not fit it")
    auto.add_argument("--notes", help=notes_help)
    auto.add_argument("--notes-file", help="a text file with the notes (added to --notes)")
    auto.add_argument("--out", help="output folder (default: <GIMaP data folder>/assistant_runs/cli/<time>)")
    auto.add_argument("--quiet", action="store_true", help="no progress on stderr")
    auto.set_defaults(run=command_auto)

    for name, function, text in (
        ("status", command_status, "what GIMaP sees for a frame"),
        ("find-calibration", command_find_calibration, "calibration material around a frame, ranked"),
    ):
        sub = commands.add_parser(name, parents=[common], help=text)
        sub.add_argument("frame")
        sub.add_argument("--notes", help=notes_help)
        sub.add_argument("--notes-file", help="a text file with the notes (added to --notes)")
        if name == "find-calibration":
            sub.add_argument("--max-results", type=int, default=12)
        sub.set_defaults(run=function)

    tools = commands.add_parser("tools", parents=[common], help="tool definitions (JSON)")
    tools.set_defaults(run=command_tools)

    call = commands.add_parser("call", parents=[common], help="run a JSON list of tool calls on one frame")
    call.add_argument("frame")
    call.add_argument("steps", help='file with [{"tool": ..., "args": {...}}, ...] or - for stdin')
    call.add_argument("--notes", help=notes_help)
    call.add_argument("--notes-file", help="a text file with the notes (added to --notes)")
    call.add_argument(
        "--out",
        help="also write the curves, q-map and tables here; export_results writes to gimap_analysis/ in it "
             "(default for exports: <GIMaP data folder>/assistant_runs/cli/<time>/<frame>)",
    )
    call.set_defaults(run=command_call)

    mcp = commands.add_parser(
        "mcp", parents=[common],
        help="serve the tools over MCP stdio (files under out_dir or <GIMaP data folder>/assistant_runs/mcp)",
    )
    mcp.set_defaults(run=command_mcp)
    return main


def main(argv=None) -> int:
    _utf8_console()
    arguments = parser().parse_args(argv)
    try:
        return arguments.run(arguments)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_FAILED


if __name__ == "__main__":
    sys.exit(main())
