"""Scientific input, isolated-job routing and simplified UI regression checks."""

from pathlib import Path
from types import SimpleNamespace
import json
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import numpy as np
import pytest

from src.gimap.features.fitting.application.workflow_v5 import validate_options, bundled_workflow
from src.gimap.features.fitting.infrastructure.adapters.workflow_v5 import prepare_sides, read_curve
from src.gimap.features.fitting.infrastructure.adapters.ai_pipeline import AiPipelinePredictor
from src.gimap.features.fitting.application.ai_models import CandidateGenerationRequest


def test_signed_native_nodes_units_and_uncertainties():
    q = np.r_[-np.arange(8, 0, -1), np.arange(1, 9)] * 0.01
    y = np.arange(16.0) - 2
    sigma = np.full(16, 0.5)
    sides = prepare_sides(q, y, sigma, validate_options(dict(q_unit="A^-1")))
    assert len(sides) == 2
    negative = sides[1]
    np.testing.assert_allclose(negative["q"], np.arange(1, 9) * 0.1)
    assert negative["observed"][-1] == -2
    np.testing.assert_array_equal(negative["sigma"], np.full(8, 0.5))
    assert not negative["sigma_estimated"]
    assert len(negative["q"]) == 8  # rendering resolution must never enter this input


@pytest.mark.parametrize(
    "change",
    [
        dict(method="model", sigma_res=0.1),
        dict(method="model", nu_res=4),
        dict(components=[4]),
        dict(normalizer=0),
        dict(search_combinations=35),
        dict(q_unit="pixel"),
    ],
)
def test_reject_invalid_conditions(change):
    with pytest.raises(ValueError):
        validate_options(change)


def test_do_not_silently_average_duplicates_or_drop_bad_sigma():
    q = np.arange(1, 10, dtype=float)
    q[-1] = q[-2]
    with pytest.raises(ValueError, match="duplicate"):
        prepare_sides(q, np.ones(9), np.ones(9), validate_options({}))
    with pytest.raises(ValueError, match="sigma"):
        prepare_sides(np.arange(1, 10), np.ones(9), np.zeros(9), validate_options({}))


def test_text_reader_header_and_invalid_rows(tmp_path):
    p = tmp_path / "input.txt"
    p.write_text("X\tY\n.1\t-2\n.2\t3\n")
    q, y, sigma = read_curve(p)
    assert y.tolist() == [-2, 3] and sigma is None
    p.write_text("q,I\n.1,2\nbad,row\n")
    with pytest.raises(ValueError, match="Non-numeric"):
        read_curve(p)


def test_new_bundle_routes_to_new_worker_preserving_existing_outputs(tmp_path):
    sentinel = tmp_path / "previous.json"
    sentinel.write_text("previous")
    request = CandidateGenerationRequest(
        bundled_workflow(),
        tmp_path,
        np.arange(1, 9),
        np.arange(8.0) - 2,
        np.ones(8),
        {},
        constraints={"workflow_v5": dict(numerical=False)},
        clear_output_dir=True,
    )
    job = AiPipelinePredictor().create_job_request(request)
    assert job.handler.endswith("workflow_v5:run_workflow_job")
    assert job.payload["intensity"][0] == -2
    assert not job.payload["options"]["numerical"]
    assert sentinel.read_text() == "previous"


def test_imported_assets_remain_identical():
    import hashlib

    root = bundled_workflow()
    manifest = json.loads((root / "conditional_fast_manifest_v2.json").read_text())
    for name, sha in manifest["files"].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == sha, name


def test_single_candidate_selection_cannot_replace_active_insitu_frame():
    from src.gimap.features.fitting.presentation.bindings.workflow_v5_binding import (
        WorkflowV5BindingMixin,
    )

    binding = WorkflowV5BindingMixin()
    binding._insitu_workflow_state = "Processing"
    binding.fitting = {"frame": "active"}
    assert binding._apply_workflow_candidate({}) is False
    assert binding.fitting == {"frame": "active"}


def test_completed_job_reports_observed_residual_without_mandatory_cutoff(tmp_path):
    from PyQt5.QtWidgets import QApplication
    from src.gimap.features.fitting.presentation.workflow_v5_dialog import WorkflowV5Dialog

    app = QApplication.instance() or QApplication([])
    dialog = WorkflowV5Dialog()
    row = dict(file="test", side="positive", rank=1, combination="sphere", best_log_rmse=.05, signed_weighted_rms=1, best_source="experimental_physical", native_q=[1,2], observed=[1,1], sigma=[.1,.1], display_q=[1,2], display_fit=[1,1])
    result = SimpleNamespace(succeeded=True, value=dict(candidates=[row], output_dir=str(tmp_path), records=[dict(status="complete")], summary=dict(runtime_seconds=1)))
    dialog._completed(result)
    assert "Review curve shape and residuals" in dialog.status.text()
    assert "no mandatory cutoff" in dialog.table.item(0, 3).toolTip()
    assert "measurement noise" in dialog.table.item(0, 3).toolTip()
    dialog.close()
    app.processEvents()


def test_the_1d_predict_figure_follows_the_light_and_dark_theme(tmp_path):
    from PyQt5.QtCore import QCoreApplication, QEvent
    from PyQt5.QtWidgets import QApplication
    from src.gimap.app.presentation.theme import apply_theme, theme_manager
    from src.gimap.features.fitting.presentation.workflow_v5_dialog import WorkflowV5Dialog

    app = QApplication.instance() or QApplication([])

    def corner(dialog):
        dialog.canvas.draw()
        return tuple(int(value) for value in np.asarray(dialog.canvas.buffer_rgba())[2, 2, :3])

    def token(name):
        colour = theme_manager().color(name)
        return colour.red(), colour.green(), colour.blue()

    try:
        apply_theme("dark", 9.0)
        dialog = WorkflowV5Dialog()
        assert corner(dialog) == token("plot_bg") != (255, 255, 255)
        assert dialog.figure.texts[0].get_color() == theme_manager().color("plot_fg").name()
        q = np.linspace(0.1, 2.0, 40)
        observed = 100 * np.exp(-(q / 0.6) ** 2) + 1
        row = dict(file="test", side="positive", rank=1, combination="sphere", best_log_rmse=.05, signed_weighted_rms=1,
                   best_source="experimental_physical", native_q=q.tolist(), observed=observed.tolist(),
                   sigma=(0.05 * observed).tolist(), display_q=q.tolist(), display_fit=(observed * 1.01).tolist())
        dialog.set_results([row], str(tmp_path))
        axes = dialog.figure.axes[0]
        assert corner(dialog) == token("plot_bg") and axes.get_facecolor()[:3] == pytest.approx(
            tuple(value / 255 for value in token("plot_bg")))
        np.testing.assert_allclose(axes.lines[-1].get_ydata(), observed * 1.01)  # the fit as it is
        apply_theme("light", 9.0)
        assert corner(dialog) == token("plot_bg") == (255, 255, 255)
        apply_theme("dark", 9.0)
        assert corner(dialog) == token("plot_bg")
        dialog.close()
        dialog.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        app.processEvents()
    finally:
        apply_theme("light", 9.0)  # the suite's theme (and a deleted dialog no longer follows it)


def test_simplified_insitu_controls_and_dialog_settings():
    from PyQt5.QtWidgets import QApplication, QWidget
    from src.gimap.features.fitting.presentation.views.insitu_series_page_view import (
        InSituSeriesPageView,
    )
    from src.gimap.features.fitting.presentation.workflow_v5_dialog import WorkflowV5Dialog

    app = QApplication.instance() or QApplication([])
    page = QWidget()
    view = InSituSeriesPageView()
    view.setupUi(page)
    page.show()
    assert tuple(view.workflowButtons) == ("source", "fit", "results")
    controls = view.workflowControls
    controls.show_step("fit")
    assert controls.stack.currentWidget() is controls.pages["fit"]
    controls.workflowModeCombo.setCurrentIndex(2)
    assert controls.fit_mode() == 2 and controls.workflowModeCombo.currentText() == "Plot curves only"
    controls.show_step("geometry")  # detector steps are gone: unknown keys are ignored
    assert controls.stack.currentWidget() is controls.pages["fit"]
    dialog = WorkflowV5Dialog(
        options=dict(components=[2, 2], sigma_res=0.01, nu_res=7, numerical=False)
    )
    assert dialog.options()["components"] == [2, 2]
    assert dialog.options()["sigma_res"] == 0.01
    assert not dialog.options()["numerical"]
    dialog.close()
    page.close()
    app.processEvents()


def test_insitu_keeps_both_native_forward_sides_without_legacy_recalculation(tmp_path):
    from src.gimap.features.fitting.presentation.bindings.insitu_refinement_lifecycle import (
        InsituRefinementLifecycleMixin,
    )
    from src.gimap.features.fitting.presentation.bindings.workflow_v5_binding import (
        WorkflowV5BindingMixin,
    )

    class Binding(WorkflowV5BindingMixin, InsituRefinementLifecycleMixin):
        def _perform_manual_fitting(self):
            raise AssertionError("V5 output must never be recomputed by the legacy forward")

        def _complete_insitu_workflow_fit(self, record, status):
            self.completed = (record, status)

    binding = Binding()
    binding._ai_output_dir = tmp_path
    rows = []
    for side, sign in (("positive", 1), ("negative", -1)):
        for rank in (1, 2):
            rows.append(
                dict(
                    workflow="native_v5",
                    side=side,
                    rank=rank,
                    native_q=(sign * np.arange(1, 9)).tolist(),
                    native_fit=(rank * np.arange(1, 9)).tolist(),
                    observed=np.arange(1, 9).tolist(),
                    sigma=np.ones(8).tolist(),
                    components=[dict(params=dict(R=4.0, h=None))],
                    global_params={},
                    signed_weighted_rms=0.0,
                )
            )
            rows[-1]["components"][0]["weight"] = 1.0
    binding._on_insitu_ai_full_fit_finished(
        {}, 0, SimpleNamespace(candidates=rows, runtime_seconds=1.2)
    )
    assert binding.completed[1] == "ok"
    assert len(binding.fitting["q"]) == 16
    np.testing.assert_array_equal(binding.fitting["I"], np.abs(binding.fitting["q"]))
    assert binding._calculate_current_chi_square() == 0
    assert len(binding.completed[0]["v5_side_candidates"]) == 2
