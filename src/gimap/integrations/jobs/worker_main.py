"""The ``__main__`` of a spawned worker process: nothing (see ``light_main``).

A spawned worker runs its parent's main module first; pointing that at this empty module keeps
``main.py``, the GUI and PyQt5 out of job and batch workers (PyQt5's msvcp140.dll crashes BornAgain).
"""
