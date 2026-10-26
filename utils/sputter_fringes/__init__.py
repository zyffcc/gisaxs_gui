"""Vertical size H(t) of the metal islands or layer from the moving q_z fringes of an in-situ sputter GISAXS series.

Reads a GIMaP Sputter result folder (``binned/`` of the bundled toolkit) read only and writes
``fringe_heights.csv`` and ``fringe_summary.json`` elsewhere. numpy and scipy only; GIMaP runs it as a subprocess
(``fringes.py``), like the toolkit. See ``cli.py`` for the outputs and ``minima.py``, ``tracks.py``, ``heights.py``
for the method.
"""

VERSION = "1"
