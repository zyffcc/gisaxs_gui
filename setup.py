"""Stamp the source revision in installed wheels and source distributions."""

import json
import runpy
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
    def run(self):
        super().run()
        if not self.editable_mode:
            stamp_revision(self.build_lib)


class SDist(sdist):
    def make_release_tree(self, base_dir, files):
        super().make_release_tree(base_dir, files)
        stamp_revision(base_dir)


setup(cmdclass={"build_py": BuildPy, "sdist": SDist})
