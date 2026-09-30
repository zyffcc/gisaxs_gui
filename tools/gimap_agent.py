"""GIMaP for command-line agents (Codex, Claude Code, scripts): GIWAXS analysis without the window.

Read docs/agents/giwaxs-playbook.md first.  The usual call is one command:

    python tools/gimap_agent.py auto FRAME [FRAME ...] [--incidence-deg 0.4] [--notes "..."]

Other commands:

    status FRAME              what GIMaP sees: detector, frames, header, geometry or not
    find-calibration FRAME    calibration files, standard images and logs around the frame, ranked
    tools                     the tool definitions (for `call`)
    call FRAME STEPS          run a JSON list of {"tool": ..., "args": {...}} on one frame
    mcp                       the same tools as an MCP server on stdio (for Codex / Claude Code)

Nothing is written next to the data: results go to --out (default: the GIMaP data
folder, assistant_runs/cli/<time>).  Exit codes: 0 done, 2 done but questions
remain (needs_attention: answer them and run again), 1 failed.
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
        technique=arguments.technique, fit=not arguments.no_fit,
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
    if any(report.get("error") or not report.get("ok") and not report.get("needs_attention") for report in reports):
        return EXIT_FAILED
    return EXIT_QUESTIONS if any(report.get("needs_attention") for report in reports) else EXIT_DONE


def _session(arguments):
    from src.gimap.app.headless_assistant import open_headless_session

    return open_headless_session(arguments.frame, notes=_notes(arguments), saved_profiles=not arguments.no_saved_profiles)


def command_status(arguments) -> int:
    session = _session(arguments)
    try:
        _print_json(_outcome_data(session.call("get_status")))
    finally:
        session.close()
    return EXIT_DONE


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
    finally:
        session.close()
    return EXIT_FAILED if failed else EXIT_DONE


def command_mcp(arguments) -> int:
    from src.gimap.app.headless_assistant import serve_mcp_stdio

    serve_mcp_stdio(saved_profiles=not arguments.no_saved_profiles)
    return EXIT_DONE


def parser() -> argparse.ArgumentParser:
    main = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--no-saved-profiles", action="store_true", help="ignore instrument profiles saved in GIMaP")
    commands = main.add_subparsers(dest="command", required=True)

    auto = commands.add_parser("auto", parents=[common], help="the standard GIWAXS analysis of one or more frames")
    auto.add_argument("frames", nargs="+", help="detector images (for a NeXus series: any module file, e.g. *_m01.nxs)")
    auto.add_argument("--calibration", help="calibration image of a standard, .poni or GIMaP calibration file")
    auto.add_argument("--standard", choices=("agbh", "lab6", "ceo2", "lab6_ceo2"), help="standard in --calibration")
    auto.add_argument("--energy-kev", type=float, help="X-ray energy when the header has none")
    auto.add_argument("--incidence-deg", type=float, help="incidence angle αi in degrees")
    auto.add_argument("--pixel-size-um", type=float, help="detector pixel size when the header has none (TIFF)")
    auto.add_argument("--frame", type=int, help="frame of a series (1-based, -1 = last; default: the last frames)")
    auto.add_argument("--sum", type=int, help="frames summed (default 10 for a series)")
    auto.add_argument("--rings", type=int, default=3, help="strongest reliable peaks analysed for orientation and size")
    auto.add_argument("--recalibrate", action="store_true", help="calibrate even when a saved profile exists")
    auto.add_argument("--technique", choices=("giwaxs", "gisaxs"), help="procedure (default: GIWAXS)")
    auto.add_argument("--no-fit", action="store_true", help="GISAXS: prepare the cut but do not fit it")
    auto.add_argument("--notes", help="beamtime notes: αi, energy, calibrant, paths")
    auto.add_argument("--notes-file", help="a text file with the notes")
    auto.add_argument("--out", help="output folder")
    auto.add_argument("--quiet", action="store_true", help="no progress on stderr")
    auto.set_defaults(run=command_auto)

    for name, function, text in (
        ("status", command_status, "what GIMaP sees for a frame"),
        ("find-calibration", command_find_calibration, "calibration material around a frame, ranked"),
    ):
        sub = commands.add_parser(name, parents=[common], help=text)
        sub.add_argument("frame")
        sub.add_argument("--notes")
        sub.add_argument("--notes-file")
        if name == "find-calibration":
            sub.add_argument("--max-results", type=int, default=12)
        sub.set_defaults(run=function)

    tools = commands.add_parser("tools", parents=[common], help="tool definitions (JSON)")
    tools.set_defaults(run=command_tools)

    call = commands.add_parser("call", parents=[common], help="run a JSON list of tool calls on one frame")
    call.add_argument("frame")
    call.add_argument("steps", help='file with [{"tool": ..., "args": {...}}, ...] or - for stdin')
    call.add_argument("--notes")
    call.add_argument("--notes-file")
    call.add_argument("--out", help="also write the curves, q-map and tables here")
    call.set_defaults(run=command_call)

    mcp = commands.add_parser("mcp", parents=[common], help="serve the tools over MCP stdio")
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
