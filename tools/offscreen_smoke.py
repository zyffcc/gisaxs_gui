"""Start and close the real GIMaP window using in-memory state repositories."""

from __future__ import annotations

import gc
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from PyQt5.QtWidgets import QApplication  # noqa: E402

from main import MainWindow  # noqa: E402
from src.gimap.app import AppContext  # noqa: E402
from src.gimap.integrations.jobs import LocalProcessJobRunner  # noqa: E402
from src.gimap.integrations.state import (  # noqa: E402
    InMemorySessionRepository,
    InMemorySettingsRepository,
    InMemoryUserPreferencesRepository,
)


def _wait_until(app: QApplication, predicate, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return
        time.sleep(0.01)
    raise TimeoutError("GIMaP offscreen startup did not finish in time")


def main() -> int:
    app = QApplication.instance() or QApplication([])
    context = AppContext(
        settings=InMemorySettingsRepository(),
        session=InMemorySessionRepository(),
        preferences=InMemoryUserPreferencesRepository(),
        jobs=LocalProcessJobRunner(),
    )
    window = MainWindow(context)
    window.show()
    _wait_until(
        app,
        lambda: bool(
            window._initialization_completed
            and hasattr(window, "runtime")
            and window.runtime.prediction._initialized
        ),
        timeout=15.0,
    )
    bindings = tuple(
        name
        for name in ("fitting", "prediction", "trainset", "classification")
        if getattr(window.runtime, name, None) is not None
    )
    if window.mainWindowWidget.count() != 5 or len(bindings) != 4:
        raise RuntimeError(
            f"Unexpected workspace composition: pages={window.mainWindowWidget.count()}, "
            f"bindings={bindings}"
        )
    print(
        "Offscreen startup OK: "
        f"pages={window.mainWindowWidget.count()}, "
        f"current={window.mainWindowWidget.currentIndex()}, "
        f"bindings={','.join(bindings)}"
    )
    window.close()
    app.processEvents()
    gc.collect()
    app.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
