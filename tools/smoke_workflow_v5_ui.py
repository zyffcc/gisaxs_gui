"""Offscreen real-window/button/worker smoke and visual review artifacts."""

import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_FONT_DPI", "96")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import numpy as np
    from PyQt5.QtWidgets import QApplication
    from main import MainWindow
    from src.gimap.app import AppContext
    from src.gimap.integrations.jobs import LocalProcessJobRunner
    from src.gimap.integrations.state import (
        InMemorySessionRepository,
        InMemorySettingsRepository,
        InMemoryUserPreferencesRepository,
    )
    from src.gimap.features.fitting.infrastructure.adapters.workflow_v5 import (
        read_curve,
        bundled_workflow,
    )

    app = QApplication.instance() or QApplication([])
    from PyQt5.QtGui import QFontDatabase, QFont

    if not QFontDatabase().families():
        for name in ("arial.ttf", "segoeui.ttf", "msyh.ttc"):
            QFontDatabase.addApplicationFont(
                str(Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / name)
            )
        app.setFont(QFont("Segoe UI", 9))
    context = AppContext(
        settings=InMemorySettingsRepository(),
        session=InMemorySessionRepository(),
        preferences=InMemoryUserPreferencesRepository(),
        jobs=LocalProcessJobRunner(),
    )
    window = MainWindow(context)
    window.resize(1440, 900)
    window.show()

    def until(check, timeout=90):
        end = time.monotonic() + timeout
        while not check():
            app.processEvents()
            time.sleep(0.02)
            if time.monotonic() > end:
                raise TimeoutError("UI smoke timeout")

    until(
        lambda: (
            window._initialization_completed
            and hasattr(window, "runtime")
            and window.runtime.fitting._initialized
        )
    )
    fitting = window.runtime.fitting
    q, y, _ = read_curve(bundled_workflow().parent / "development/fixtures/Cut_Data.txt")
    keep = q > 0
    fitting.current_1d_data = dict(
        q=q[keep], I=y[keep], err=np.hypot(0.1 * y[keep], 50), q_source_unit="nm"
    )
    window.fitCurrentDataCheckBox.setChecked(False)
    fitting._roi_min = fitting._roi_max = None
    fitting._save_workflow_options(
        dict(
            method="model",
            components=[2, 2],
            sigma_res=0.012953743397082464,
            nu_res=5.000743985235129,
            numerical=False,
        )
    )
    workspace = window.components.fitting_workspace
    workspace.show_workflow_step("fit")
    window.fittingModeTabs.setCurrentIndex(3)
    from PyQt5.QtWidgets import QMessageBox

    QMessageBox.warning = lambda _p, title, message, *a: (_ for _ in ()).throw(
        RuntimeError(f"{title}: {message}")
    )
    assert fitting._current_ai_curve_arrays() is not None
    window.aiFittingFastPredictButton.click()
    until(
        lambda: (
            getattr(fitting, "_workflow_v5_dialog", None) is not None
            and fitting._ai_job_thread is None
        ),
        180,
    )
    dialog = fitting._workflow_v5_dialog
    assert dialog.rows and abs(dialog.rows[0]["best_log_rmse"] - 0.17658074125669987) < 1e-5
    out = ROOT / "validation/workflow_v5"
    out.mkdir(parents=True, exist_ok=True)
    for _ in range(4):
        app.processEvents()
        time.sleep(0.1)
    dialog.grab().save(str(out / "prediction.png"))
    dialog.settings_button.setChecked(True)
    app.processEvents()
    dialog.grab().save(str(out / "prediction_settings.png"))
    dialog.close()
    workspace.show_context("insitu")
    app.processEvents()
    page = window.fittingInsituSeriesPage
    page.setParent(None)
    page.resize(1280, 850)
    page.show()
    app.processEvents()
    page.ui.workflowControls.show_step("fit")
    page.grab().save(str(out / "insitu.png"))
    (out / "UI_VERIFIED.json").write_text(
        json.dumps(
            dict(
                passed=True,
                actual_button="aiFittingFastPredictButton",
                actual_worker="spawned LocalProcessJobRunner",
                candidate_count=len(dialog.rows),
                logrmse=dialog.rows[0]["best_log_rmse"],
            ),
            indent=2,
        )
    )
    page.close()
    window.close()
    app.processEvents()


if __name__ == "__main__":
    main()
