"""Appending application files must not rewrite already packed runtime headers or compressed bytes."""
import json
import struct
import zipfile

import pytest

from tools.append_release_files import append_files


@pytest.mark.parametrize('zip64', [False, True])
def test_runtime_entries_survive_appending_including_empty_deflated_files(tmp_path, monkeypatch, zip64):
    if zip64:
        monkeypatch.setattr(zipfile, 'ZIP64_LIMIT', 128)
    archive = tmp_path / 'package.zip'
    root, licences = tmp_path / 'app', tmp_path / 'licences'
    root.mkdir(); licences.mkdir()
    (root / 'main.py').write_bytes(b"print('example')\r\n")
    (root / 'manifest.json').write_text('{"private_example": "not shipped"}')
    (licences / 'MIT.txt').write_text('MIT License\n')
    runtime = {'GIMaP/runtime/Lib/urllib/__init__.py': b'',
               'GIMaP/runtime/Lib/reference.py': b'# reference\n' * 100}
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as packed:
        for name, data in runtime.items():
            packed.writestr(name, data)
    with zipfile.ZipFile(archive) as packed:
        original = archive.read_bytes()[:packed.start_dir]
    subset = json.dumps({'files': {'main.py': 'synthetic'}})
    append_files(archive, root, 'GIMaP', licences,
                 {'files': ['main.py', 'manifest.json'], 'licences': ['MIT.txt'],
                  'generated': {'manifest.json': subset}})
    # This checks headers as well as data: rewriting an empty file's method caused Windows to emit 03 00.
    assert archive.read_bytes()[:len(original)] == original
    with zipfile.ZipFile(archive) as packed:
        assert packed.testzip() is None
        for name, data in runtime.items():
            assert packed.read(name) == data
            entry = packed.getinfo(name)
            with archive.open('rb') as stream:
                stream.seek(entry.header_offset + 8)
                assert struct.unpack('<H', stream.read(2))[0] == entry.compress_type
        assert packed.read('GIMaP/main.py') == (root / 'main.py').read_bytes()
        assert packed.read('GIMaP/manifest.json').decode() == subset
        assert packed.read('GIMaP/licenses/MIT.txt') == (licences / 'MIT.txt').read_bytes()
    assert 'private_example' in (root / 'manifest.json').read_text()


def test_appending_rejects_replacing_a_runtime_entry(tmp_path):
    archive = tmp_path / 'package.zip'
    with zipfile.ZipFile(archive, 'w') as packed:
        packed.writestr('GIMaP/runtime/python.exe', b'original')
    with pytest.raises(ValueError, match='Duplicate release entry'):
        append_files(archive, tmp_path, 'GIMaP', tmp_path,
                     {'files': ['runtime/python.exe'], 'licences': []})
    with zipfile.ZipFile(archive) as packed:
        assert packed.read('GIMaP/runtime/python.exe') == b'original'
