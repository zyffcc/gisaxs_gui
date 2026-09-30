"""The ``__main__`` of a batch worker process: nothing (see ``frame_workers._light_main``).

A spawned worker runs its parent's main module first; pointing that at this empty module keeps the
GUI, PyQt5 and BornAgain out of the workers.
"""
