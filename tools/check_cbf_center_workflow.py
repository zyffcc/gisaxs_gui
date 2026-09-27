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
    saved = json.loads((ROOT / "config/user_parameters.json").read_text(encoding="utf-8"))
    context = AppContext(
        settings=InMemorySettingsRepository(
            {"beam": saved["beam"], "fitting": {"detector": saved["fitting"]["detector"]}}
        ),
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
    wait_until = time.monotonic() + 1.5
    until(lambda: time.monotonic() >= wait_until)
    window.resize(1600, 1050)
    app.processEvents()
    fitting = window.runtime.fitting
    from PyQt5.QtWidgets import QMessageBox
    from src.gimap.features.fitting.presentation.detector_data_access import analysis_image_for
    from matplotlib.figure import Figure

    out = ROOT / "validation/center_symmetry"
    geometry_only = "--geometry-only" in sys.argv
    if geometry_only:
        out = out / "geometry_controls"
    if "--insitu" in sys.argv:
        out = out / "insitu"
    if "--quick" in sys.argv:
        out = ROOT / "validation/usability_20260921/gui_quick" / ("insitu" if "--insitu" in sys.argv else "single")
    if "--native-model" in sys.argv:
        out = ROOT / "validation/usability_20260921/gui_native_model"
    if "--preprocess-mask" in sys.argv:
        out = ROOT / "validation/preprocess_mask_20260921" / ("geometry" if geometry_only else "insitu" if "--insitu" in sys.argv else "single")
    out.mkdir(parents=True, exist_ok=True)
    messages = []
    fitting.status_updated.connect(messages.append)

    def fail(_p, title, message, *args):
        raise RuntimeError(f"{title}: {message}")

    QMessageBox.warning = fail
    QMessageBox.critical = fail
    # User confirmed the last saved detector/beam geometry is the correct calibration.
    fitting.ui.fittingDetectorSetupPanel._load()
    fitting.ui.fitDataPointsNumValue.setValue(500)
    path = ROOT / "TestSAXSdata/jg_gisaxs_4nm_old_3ml_insitu_ds03_00001_00033.cbf"
    window.gisaxsInputImportButtonValue.setText(str(path))
    window.gisaxsInputImportButtonValue.returnPressed.emit()
    until(
        lambda: (
            analysis_image_for(fitting) is not None
            and any("Auto center found" in m for m in messages)
        )
    )
    before_y = window.gisaxsInputCenterVerticalValue.value()
    before_x = window.gisaxsInputCenterParallelValue.value()
    window.gisaxsInputCutButton.click()
    assert fitting._has_existing_cut_result()
    original_image = analysis_image_for(fitting).copy()
    old_height = window.gisaxsInputCutLineVerticalValue.value()
    window.gisaxsOptimizeCenterXButton.click()
    settle = time.monotonic() + 0.5
    until(lambda: time.monotonic() > settle)
    assert window.gisaxsInputCutLineVerticalValue.value() == old_height
    assert hasattr(fitting, "_last_center_symmetry"), messages[-10:]
    result = fitting._last_center_symmetry
    assert abs(window.gisaxsInputCenterParallelValue.value() - result["center_x"]) < 0.01
    assert result["score_after"] < result["score_before"]
    assert window.gisaxsInputCenterVerticalValue.value() == before_y
    assert (
        abs(context.settings.get("fitting", "detector.beam_center_x") - result["center_x"]) < 1e-8
    )
    np.testing.assert_array_equal(analysis_image_for(fitting), original_image)
    q, y, sigma = fitting._current_ai_curve_arrays()
    assert np.any(q < 0) and np.any(q > 0), (q.min(), q.max())
    if "--preprocess-mask" in sys.argv:
        state = fitting.current_detector_image
        assert state.preprocessing.mask_negative_pixels
        assert not np.any(state.analysis_image[np.isfinite(state.analysis_image)] < 0)
        assert state.masked_pixels > 0
        np.testing.assert_allclose(q, fitting.current_cut_data["x_coords"])
        np.testing.assert_allclose(y, fitting.current_cut_data["y_intensity"])
        if geometry_only:
            # A changed preprocessing revision cannot be fitted with an old cut.
            old_margin = fitting._invalid_margin_px
            fitting._invalid_margin_px = 0
            fitting._reapply_input_image_options(refresh=False)
            try:
                fitting._current_ai_curve_arrays()
            except ValueError as exc:
                assert "Extract / Update" in str(exc)
            else:
                raise AssertionError("Stale cut accepted after changing the mask")
            fitting._invalid_margin_px = old_margin
            fitting._reapply_input_image_options(refresh=False)
            window.gisaxsInputCutButton.click()
    np.savetxt(
        out / "optimized_cut.csv",
        np.column_stack((q, y, sigma)),
        delimiter=",",
        header="q_nm^-1,I,sigma",
    )
    workspace = window.components.fitting_workspace
    workspace.show_workflow_step("cut")
    app.processEvents()
    window.grab().save(str(out / "yoneda_cut_ui.png"))
    from PyQt5.QtGui import QPixmap, QColor

    pixmap = QPixmap(workspace.cut_line_card.size())
    pixmap.fill(QColor("#f5f7fb"))
    workspace.cut_line_card.render(pixmap)
    pixmap.save(str(out / "yoneda_controls.png"))
    r0, r1, x0, x1 = result["pixel_region"]
    band = np.where(original_image[r0 : r1 + 1] >= 0, original_image[r0 : r1 + 1], np.nan)
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        profile = np.nanmedian(band, axis=0)
    fig = Figure(figsize=(11, 4), tight_layout=True)
    for i, (center, title) in enumerate(
        ((result["initial_x"], "Before"), (result["center_x"], "After")), 1
    ):
        ax = fig.add_subplot(1, 2, i)
        offset = np.arange(len(profile)) - center
        ax.plot(offset[offset > 0], profile[offset > 0], label="Right")
        ax.plot(-offset[offset < 0], profile[offset < 0], label="Left", alpha=0.75)
        ax.set_yscale("symlog", linthresh=1)
        ax.set_xlabel("Distance from Center X (pixels)")
        ax.set_ylabel("Measured intensity")
        ax.set_title(f"{title}: X={center:.3f} px")
        ax.legend()
        ax.grid(alpha=0.2)
    fig.savefig(out / "symmetry_before_after.png", dpi=140)
    if geometry_only:
        # Manual Yoneda override remains effective; q-mode changes preserve rows.
        window.gisaxsInputCenterVerticalValue.setValue(before_y + 2)
        settle = time.monotonic() + 0.5
        until(lambda: time.monotonic() > settle)
        window.gisaxsInputCutButton.click()
        manual_y = window.gisaxsInputCenterVerticalValue.value()
        window.gisaxsOptimizeCenterXButton.click()
        assert window.gisaxsInputCenterVerticalValue.value() == manual_y
        panel = window.fittingDetectorSetupPanel
        panel.show_q_axis_checkbox.setChecked(True)
        panel.apply_button.click()
        app.processEvents()
        region_before = fitting._current_selection_pixel_region(q_mode=True, horizontal_axis="qy")
        window.gisaxsOptimizeCenterXButton.click()
        region_after = fitting._current_selection_pixel_region(q_mode=True, horizontal_axis="qy")
        assert region_after[:2] == region_before[:2], (region_before, region_after)
        assert fitting._should_show_q_axis()
        q_native, y_native, sigma_native = fitting._current_ai_curve_arrays()
        assert len(q_native) > 500 and np.all(sigma_native > 0)
        assert (
            abs(
                context.settings.get("fitting", "detector.beam_center_y")
                - saved["fitting"]["detector"]["beam_center_y"]
            )
            < 1e-8
        )
        (out / "VERIFIED.json").write_text(
            json.dumps(
                dict(
                    passed=True,
                    manual_y_preserved=True,
                    q_mode_rows_preserved=True,
                    beam_y_preserved=True,
                    rows=region_after[:2],
                ),
                indent=2,
            )
        )
        window.close()
        app.processEvents()
        return
    fitting._save_workflow_options(dict(components=[], numerical=True, method="experimental" if "--quick" in sys.argv else "model"))
    if "--insitu" in sys.argv:
        from src.gimap.features.fitting.application.insitu_records import ManageInSituRecords
        from src.gimap.features.fitting.infrastructure.adapters.local_insitu_records import (
            LocalInSituRecordRepository,
        )

        class TestRecords(LocalInSituRecordRepository):
            def cache_directory(self):
                return out / "cache"

        fitting.fitting_view_model.storage._insitu_records = ManageInSituRecords(TestRecords())
        workspace.show_context("insitu")
        page = window.fittingInsituSeriesPage
        page.ui.captureRecipeButton.click()
        assert fitting.fitting_view_model.insitu.recipe is not None
        widgets = page.workflow_widgets()
        widgets["run_mode"].setCurrentText("Process Existing Sequence")
        widgets["sequence_folder"].setText(str(path.parent))
        widgets["sequence_pattern"].setText(path.name)
        widgets["recursive"].setChecked(False)
        tick = time.perf_counter()
        widgets["process"].click()
        until(
            lambda: (
                len(fitting._insitu_workflow_results) > 0
                and not fitting._insitu_workflow_busy
                and fitting._insitu_workflow_state == "Idle"
            ),
            600,
        )
        records = fitting._insitu_workflow_results
        assert len(records) == 1, records
        assert records[0]["fit_status"] == "ok", records[0]
        assert len(records[0]["v5_side_candidates"]) == 2
        (out / "VERIFIED.json").write_text(
            json.dumps(
                dict(passed=True, seconds=time.perf_counter() - tick, records=records),
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        page.grab().save(str(out / "insitu_complete.png"))
        (out / "messages.txt").write_text("\n".join(messages), encoding="utf-8")
        window.close()
        app.processEvents()
        return
    workspace.show_workflow_step("fit")
    window.fittingModeTabs.setCurrentIndex(3)
    tick = time.perf_counter()
    if "--quick" in sys.argv:
        window.aiFittingExperimentalButton.click()
    else:
        window.aiFittingFullAutoFitButton.click()
    until(
        lambda: (
            getattr(fitting, "_workflow_v5_dialog", None) is not None
            and fitting._ai_job_thread is None
        ),
        600,
    )
    dialog = fitting._workflow_v5_dialog
    assert dialog.rows, messages[-20:]
    assert {r["side"] for r in dialog.rows} == {"positive", "negative"}
    for _ in range(3):
        app.processEvents()
        time.sleep(0.1)
    dialog.grab().save(str(out / "cbf_fitting.png"))
    (out / "VERIFIED.json").write_text(
        json.dumps(
            dict(
                passed=True,
                source=str(path),
                initial_cut_x=before_x,
                yoneda_y=before_y,
                symmetry=result,
                fit_wall_seconds=time.perf_counter() - tick,
                fit_output=str(dialog.output_dir),
                candidate_count=len(dialog.rows),
                observations=getattr(fitting, "_workflow_observation_metadata", {}),
                best_by_side=[
                    dict(
                        side=r["side"],
                        logrmse=r["best_log_rmse"],
                        rms_sigma=r["signed_weighted_rms"],
                        combination=r["combination"],
                    )
                    for r in dialog.rows
                    if r["rank"] == 1
                ],
                geometry=fitting.ui.fittingDetectorSetupPanel.current_settings().__dict__,
                calibration_note="User confirmed geometry from config/user_parameters.json; inference uses that calibration. Model parameters still have no experimental ground truth.",
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    (out / "messages.txt").write_text("\n".join(messages), encoding="utf-8")
    dialog.close()
    window.close()
    app.processEvents()


if __name__ == "__main__":
    main()
