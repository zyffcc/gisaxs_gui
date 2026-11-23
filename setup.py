"""Ship application resources and stamp the source revision in distributions."""

import json
import runpy
import shutil
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py
from setuptools.command.sdist import sdist


def stamp_revision(destination):
    module = Path("src/gimap/shared/revision.py")
    commit = runpy.run_path(str(module))["source_commit"]()
    if commit:
        path = Path(destination) / module.with_name("revision.json")
        # sdist may have hard-linked this file from the working tree.
        path.unlink(missing_ok=True)
        path.write_text(json.dumps({"commit": commit}) + "\n", encoding="utf-8")


class BuildPy(build_py):
    @staticmethod
    def resource_files():
        # Keep the source layout for the manuals' relative links. Only the named catalogue is public.
        files = [Path(name) for name in (
            "config/detectors.json", "docs/User_Manual.md", "docs/User_Manual_zh.md",
            "docs/Physics_Manual.md", "docs/Physics_Manual_zh.md", "LICENSE",
            "THIRD_PARTY_NOTICES.md", "CHANGELOG.md", "CITATION.cff",
            "tools/benchmarks/README.md", "tools/benchmarks/insitu_online.json",
        )]
        for pattern in ("docs/manual_images/en/*.png", "docs/manual_images/zh/*.png",
                        "assets/icons/*.svg", "assets/icons/*.png"):
            files.extend(sorted(Path().glob(pattern)))
        return files

    def get_source_files(self):
        # Also include the originals in an sdist, so building a wheel from it gives the same resources.
        return super().get_source_files() + [str(path) for path in self.resource_files()]

    def run(self):
        super().run()
        if not self.editable_mode:
            stamp_revision(self.build_lib)
            root = Path(self.build_lib) / "src" / "gimap" / "_resources"
            for source in self.resource_files():
                destination = root / source
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)


class SDist(sdist):
    def make_release_tree(self, base_dir, files):
        super().make_release_tree(base_dir, files)
        stamp_revision(base_dir)


setup(cmdclass={"build_py": BuildPy, "sdist": SDist})
