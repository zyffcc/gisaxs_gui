"""Portable launcher extraction, relocation and application source selection."""
import hashlib
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tarfile

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_release_sources_exclude_retired_training_workspace_and_private_files():
    from tools.build_macos_release import source_files
    names, manifest = source_files(ROOT)
    assert 'src/gimap/app/main_window.py' in names
    assert not any(name.startswith('src/gimap/features/trainset/') for name in names)
    assert not any('/_handoff/' in name or name.startswith('docs/agents/eval/local/') for name in names)
    assert 'utils/gisaxs1d_gui_kit/LICENSE' in names
    assert 'LICENSE' in manifest['files']


@pytest.mark.skipif(sys.platform != 'darwin', reason='Exercises the macOS shell launcher')
@pytest.mark.parametrize('corrupt', [False, True])
def test_launcher_handles_unicode_spaces_moving_packages_and_bad_downloads(tmp_path, corrupt):
    package = tmp_path / '包 with spaces'
    package.mkdir()
    runtime = tmp_path / 'fake runtime'
    (runtime / 'bin').mkdir(parents=True)
    python = runtime / 'bin/python'
    python.write_text('#!/bin/sh\nprintf "%s\\n" "$@" >> "$GIMAP_HOME/calls.txt"\n')
    python.chmod(0o755)
    (runtime / 'bin/conda-unpack').touch()
    archive = package / 'runtime.tar.gz'
    with tarfile.open(archive, 'w:gz') as stream:
        stream.add(runtime / 'bin', arcname='bin')
    checksum = hashlib.sha256(archive.read_bytes()).hexdigest()
    launcher = package / 'Start_GIMaP.command'
    launcher.write_text((ROOT / 'tools/macos_launcher.command').read_text()
                        .replace('@ARCH@', platform.machine()).replace('@RUNTIME_SHA@', checksum))
    launcher.chmod(0o755)
    home, data = tmp_path / 'user home', tmp_path / 'data'
    home.mkdir()
    data.mkdir()
    env = dict(os.environ, HOME=str(home), GIMAP_HOME=str(data))
    if corrupt:
        archive.write_bytes(b'incomplete')
        result = subprocess.run([str(launcher)], env=env, capture_output=True, text=True)
        assert result.returncode != 0 and 'damaged' in result.stdout
        assert not (data / 'calls.txt').exists()
        assert not list(home.rglob('*.lock'))
        return
    subprocess.run([str(launcher), '--probe'], env=env, check=True, capture_output=True)
    cache = home / 'Library/Application Support/GIMaP/runtimes' / checksum
    assert (cache / '.ready').is_file()
    calls = (data / 'calls.txt').read_text().splitlines()
    assert calls == [str(cache / 'bin/conda-unpack'), str(package / 'app/main.py'), '--probe']
    moved = tmp_path / 'moved app'
    shutil.move(str(package), moved)
    subprocess.run([str(moved / launcher.name)], env=env, check=True, capture_output=True)
    assert (data / 'calls.txt').read_text().splitlines() == calls + [str(moved / 'app/main.py')]


def test_application_starts_without_the_retired_training_package(tmp_path):
    """A complete source tree must not conceal an import that breaks the smaller portable package."""
    code = '''
import sys, importlib.abc, runpy
from pathlib import Path
class NoTrainset(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(('src.gimap.features.trainset', 'src.gimap.integrations.bornagain')):
            raise ModuleNotFoundError('Retired workspace is not shipped: ' + fullname)
sys.meta_path.insert(0, NoTrainset())
runpy.run_path(str(Path(sys.argv[1]) / 'tools/offscreen_smoke.py'), run_name='__main__')
'''
    env = dict(os.environ, GIMAP_HOME=str(tmp_path), QT_QPA_PLATFORM='offscreen', GIMAP_SPUTTER_NO_INSTALL='1')
    result = subprocess.run([sys.executable, '-c', code, str(ROOT)], env=env, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'Offscreen startup OK' in result.stdout
