"""The portable release's first start: the folder is written into the packed environment in little memory."""

import importlib.util
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PLACEHOLDER = r"D:\conda\envs\GUI"

spec = importlib.util.spec_from_file_location("portable_unpack", REPO / "tools" / "portable_unpack.py")
portable_unpack = importlib.util.module_from_spec(spec)
spec.loader.exec_module(portable_unpack)

# The shape of conda-pack's conda-unpack-script.py: functions, the list, then the command line.
HEADER = '''
import os
on_win = os.name == "nt"
CALLS = []

def update_prefix(path, new_prefix, placeholder, mode="text"):
    CALLS.append(path)
    with open(path, "rb+") as fh:
        data = fh.read()
        fh.seek(0)
        if mode == "text":
            data = data.replace(placeholder.encode(), new_prefix.encode())
        fh.write(data)
        fh.truncate()
'''
FOOTER = "\nif __name__ == '__main__':\n    raise SystemExit('the command line is not used')\n"


def _runtime(tmp_path, files):
    runtime = tmp_path / "runtime"
    (runtime / "Scripts").mkdir(parents=True)
    lines = []
    for rel, (data, mode) in files.items():
        lines.append(f"    ({rel!r}, {PLACEHOLDER!r}, {mode!r}),\n")
        if data is not None:
            (runtime / rel).parent.mkdir(parents=True, exist_ok=True)
            (runtime / rel).write_bytes(data)
    script = HEADER + "_prefix_records = [\n" + "".join(lines) + "]\n" + FOOTER
    (runtime / "Scripts" / "conda-unpack-script.py").write_text(script, encoding="utf-8")
    return runtime


def test_only_files_with_the_old_path_are_rewritten(tmp_path, capsys):
    big = b"\0" * (2 * 2**20)
    runtime = _runtime(tmp_path, {
        "qt.conf": (f"Prefix = {PLACEHOLDER}/Library\n".encode(), "text"),
        "Lib/plain.txt": (b"no path\n", "text"),
        "Lib/empty.txt": (b"", "text"),
        "Lib/big.pyd": (big, "binary"),
        "Lib/gone.txt": (None, "text"),
    })
    before = (runtime / "Lib/big.pyd").stat().st_mtime_ns

    stats = portable_unpack.unpack(runtime)

    prefix = os.path.abspath(runtime)
    assert (runtime / "qt.conf").read_bytes() == f"Prefix = {prefix}/Library\n".encode()
    assert (runtime / "Lib/plain.txt").read_bytes() == b"no path\n"
    assert (runtime / "Lib/big.pyd").stat().st_mtime_ns == before  # never opened for writing
    assert stats == {"listed": 5, "missing": 1, "changed": 1, "largest_changed": stats["largest_changed"]}
    assert "100 %" in capsys.readouterr().out


def test_check_changes_nothing(tmp_path):
    data = f"x {PLACEHOLDER} y".encode()
    runtime = _runtime(tmp_path, {"a.pth": (data, "text")})

    stats = portable_unpack.unpack(runtime, check_only=True)

    assert stats["changed"] == 1
    assert (runtime / "a.pth").read_bytes() == data


def test_binary_files_are_matched_in_lower_case_on_windows():
    assert portable_unpack.needles(PLACEHOLDER, "binary", on_win=True) == (
        PLACEHOLDER.encode(), PLACEHOLDER.lower().encode())
    assert portable_unpack.needles(PLACEHOLDER, "text", on_win=True) == (PLACEHOLDER.encode(),)


def test_records_are_read_without_running_the_command_line(tmp_path):
    runtime = _runtime(tmp_path, {"a.txt": (b"a", "text"), "b.txt": (b"b", "binary")})
    script = runtime / "Scripts" / "conda-unpack-script.py"

    namespace, records = portable_unpack.read_unpack_script(script)

    assert callable(namespace["update_prefix"])
    assert list(records()) == [("a.txt", PLACEHOLDER, "text"), ("b.txt", PLACEHOLDER, "binary")]
    assert portable_unpack.count_records(script) == 2


def test_a_folder_without_the_script_fails_with_its_reason(tmp_path, capsys):
    assert portable_unpack.main(["--runtime", str(tmp_path)]) == 1
    assert "not a portable GIMaP environment" in capsys.readouterr().out


def test_launchers_stay_ascii():
    """cmd.exe and Windows PowerShell 5.1 read these in the system code page."""
    for name in ("Start_GIMaP.bat", "tools/setup_source_env.ps1", "tools/portable_unpack.py"):
        data = (REPO / name).read_bytes()
        assert all(byte < 128 for byte in data), name
