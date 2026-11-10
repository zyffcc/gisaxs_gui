"""Build an architecture-specific macOS portable ZIP from tracked application files and a Conda environment.

Run on the target architecture, with a clean release environment, never a developer environment.
The double-click launcher verifies and relocates the runtime into a writable user cache on first launch.
No publishing, signing, or Git mutations are performed here.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.collect_licenses import collect
from tools.list_fit1d_release_files import KIT, MANIFEST, release_files

EXCLUDE = re.compile(
    r'^(\.github/|docs/research/|docs/agents/eval/local/|work/|src/gimap/features/trainset/)'
    r'|(^|/)(_handoff|research|validation|development|__pycache__|\.pytest_cache|\.gimap_cache)/'
    r'|^config/(user_[^/]*|instrument_profiles)\.json$|\.(ipynb|py[co])$'
    r'|(^|/)(gimap_analysis|AI_Fitting_Output|trainset_jobs|test_output|test_results)/'
    r'|^(runtime|release|licenses)/|^GISAXS_HANDOFF_[^/]*\.md$'
)


def source_files(root: Path) -> tuple[list[str], dict | None]:
    names = subprocess.check_output(['git', '-C', str(root), 'ls-files', '-z']).decode().split('\0')
    kit = release_files(root)
    selected = {name for name in names if name and not name.startswith(KIT + '/') and not EXCLUDE.search(name)}
    selected.update(kit['files'])
    for name in selected:
        path = root / name
        if not path.is_file() or path.is_symlink():
            raise ValueError(f'Not a regular release file: {name}')
    return sorted(selected), kit['manifest']


def build(prefix: Path, output: Path) -> Path:
    if sys.platform != 'darwin':
        raise ValueError('Build the macOS package on macOS, on the processor architecture you will distribute.')
    arch = platform.machine()
    if arch not in {'arm64', 'x86_64'}:
        raise ValueError(f'Unsupported macOS architecture: {arch}')
    if not (prefix / 'conda-meta').is_dir():
        raise ValueError('Use a dedicated Conda release environment.')
    subprocess.run(['git', '-C', str(ROOT), 'diff', '--exit-code', 'HEAD', '--'], check=True)
    actual_arch = subprocess.check_output([str(prefix / 'bin/python'), '-c',
                                           'import platform; print(platform.machine())'], text=True).strip()
    if actual_arch != arch:
        raise ValueError(f'Release environment architecture {actual_arch} does not match builder {arch}')
    # In a fresh process: catch incompatible native wheels before producing a package.
    subprocess.run([str(prefix / 'bin/python'), '-c',
                    'import numpy, scipy.signal, scipy.sparse.linalg, h5py, cv2, matplotlib; '
                    'from PyQt5 import QtCore; import tensorflow; print("Release runtime imports passed")'], check=True)

    from src.gimap.shared.version import __version__
    names, kit_manifest = source_files(ROOT)
    output.mkdir(parents=True, exist_ok=True)
    package_name = f'GIMaP-{__version__}-macos-{arch}-portable'
    with tempfile.TemporaryDirectory(prefix='gimap-package-') as temporary:
        package = Path(temporary) / package_name
        application = package / 'app'
        for name in names:
            target = application / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, target)
        if kit_manifest:
            (application / KIT / MANIFEST).write_text(json.dumps(kit_manifest, indent=1) + '\n')
        licences = collect(prefix, package / 'licenses', ROOT)
        if not licences['ok']:
            raise ValueError('Incomplete dependency licences: ' + '; '.join(licences['problems']))
        import conda_pack
        archive = package / 'runtime.tar.gz'
        conda_pack.pack(prefix=str(prefix), output=str(archive), format='tar.gz')
        digest = _sha(archive)
        launcher = (ROOT / 'tools/macos_launcher.command').read_text().replace('@ARCH@', arch).replace('@RUNTIME_SHA@', digest)
        start = package / 'Start_GIMaP.command'
        start.write_text(launcher)
        start.chmod(0o755)
        with (package / 'pip-freeze.txt').open('w') as stream:
            subprocess.run([str(prefix / 'bin/python'), '-m', 'pip', 'freeze'], stdout=stream, check=True)
        (package / 'BUILD.json').write_text(json.dumps({
            'software_version': __version__, 'architecture': arch, 'macos_build': platform.mac_ver()[0],
            'commit': subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD']).decode().strip(),
            'runtime_sha256': digest, 'files': len(names),
        }, indent=2) + '\n')
        # ditto preserves the executable bit of the double-click launcher and Unicode file names.
        result = output / (package_name + '.zip')
        subprocess.run(['ditto', '-c', '-k', '--keepParent', str(package), str(result)], check=True)
    (output / (package_name + '.sha256')).write_text(f'{_sha(result)}  {result.name}\n')
    return result


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prefix', type=Path, default=Path(sys.prefix))
    parser.add_argument('--out', type=Path, default=ROOT / 'release')
    parser.add_argument('--list-only', action='store_true')
    args = parser.parse_args()
    if args.list_only:
        print('\n'.join(source_files(ROOT)[0]))
    else:
        print(build(args.prefix.resolve(), args.out.resolve()))
