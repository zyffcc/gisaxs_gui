"""The version of GIMaP: the one place it is written.

The About box, the error log and every export record that names the software read it from here, so a
file written by one release never carries the number of another.
"""

from __future__ import annotations

APP_NAME = "GIMaP"
__version__ = "0.1.0"
SOFTWARE = f"{APP_NAME} v{__version__}"
"""How an export record names the program that wrote it (``software_version``)."""

__all__ = ["APP_NAME", "SOFTWARE", "__version__"]
