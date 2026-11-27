"""The ``__main__`` of a spawned job or batch worker (``jobs/light_main``).

It caps the threads of the worker's OpenBLAS libraries before anything imports NumPy (``shared/blas_threads``) and
does nothing else. It lies outside the ``jobs`` package, whose ``__init__`` imports NumPy, and it keeps ``main.py``,
the GUI and PyQt5 out of the worker (PyQt5's msvcp140.dll crashes BornAgain).
"""

from src.gimap.shared.blas_threads import cap_environment

cap_environment()
