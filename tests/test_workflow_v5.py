"""The 1D Predict window and its job: scientific input, isolated-job routing, options (V5's network, the Single RC
specialist, the physical fit and fit1d were retired in 0.2.0: their saved options open as Predict parameters), UI
regression checks."""

from pathlib import Path
from types import SimpleNamespace
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import numpy as np
import pytest

from src.gimap.features.fitting.application.workflow_v5 import RETIRED, default_options, validate_options
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
        dict(components=[4]),
        dict(components=[1, 1, 1, 1]),
        dict(cut_qz=-0.1),
        dict(cut_qz=float("nan")),
        dict(method="network"),
        dict(max_solutions=0),
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
        tmp_path / "unused",
        tmp_path,
        np.arange(1, 9),
        np.arange(8.0) - 2,
        np.ones(8),
        {},
        constraints={"workflow_v5": dict(method="experimental", components=[2])},
        clear_output_dir=True,
    )
    job = AiPipelinePredictor().create_job_request(request)
    assert job.handler.endswith("workflow_v5:run_workflow_job")
    assert job.payload["intensity"][0] == -2
    assert job.payload["options"]["method"] == "posterior"  # a retired method's options: Predict parameters
    assert job.payload["options"]["components"] == [2]
    assert sentinel.read_text() == "previous"


def test_options_saved_with_the_retired_methods_open_as_predict_parameters():
    saved = {**default_options(), "method": "model", "numerical": False, "search_combinations": 12, "normalizer": None,
             "amplitude_calibration": True, "condition_combinations": 3}
    options = validate_options(saved)
    assert options["method"] == "posterior" and not set(RETIRED) & set(options)
    assert validate_options({"method": "stable"})["method"] == "posterior"
    for retired in ("experimental", "fit1d"):  # the physical fit and fit1d, with their own options
        options = validate_options({"method": retired, "sigma_res": 0.5, "nu_res": 40, "fit1d_mode": "auto",
                                    "fit1d_qz": 0.3})
        assert options["method"] == "posterior" and not set(RETIRED) & set(options)
        assert options["cut_qz"] == 0.3  # fit1d's q_z of the cut is kept under its new name
    assert default_options()["method"] == "posterior"


def test_completed_job_reports_observed_residual_without_mandatory_cutoff(tmp_path):
    from PyQt5.QtWidgets import QApplication
    from src.gimap.features.fitting.presentation.workflow_v5_dialog import WorkflowV5Dialog

    app = QApplication.instance() or QApplication([])
    dialog = WorkflowV5Dialog()
    row = dict(file="test", side="positive", rank=1, combination="sphere", best_log_rmse=.05, signed_weighted_rms=1, best_source="posterior", native_q=[1,2], observed=[1,1], sigma=[.1,.1], display_q=[1,2], display_fit=[1,1])
    result = SimpleNamespace(succeeded=True, value=dict(candidates=[row], output_dir=str(tmp_path), records=[dict(status="complete")], summary=dict(runtime_seconds=1)))
    dialog._completed(result)
    # The status line says what finished; what the candidates are worth is the warning under the title, how to
    # judge one the tooltip of its score.
    assert dialog.status.text() == "Finished in 1.00 s · 0 failed files · 1 candidates saved."
    from PyQt5.QtWidgets import QLabel

    warning = dialog.findChild(QLabel, "workflowV5Experimental")
    assert warning.isVisibleTo(dialog) and "probability" in warning.toolTip()
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
                   best_source="posterior", native_q=q.tolist(), observed=observed.tolist(),
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


def test_dialog_settings_keep_the_components_and_resolution():
    from PyQt5.QtWidgets import QApplication
    from src.gimap.features.fitting.presentation.workflow_v5_dialog import WorkflowV5Dialog

    app = QApplication.instance() or QApplication([])
    dialog = WorkflowV5Dialog(
        options=dict(method="experimental", components=[2, 2], sigma_res=0.01, nu_res=7, cut_qz=0.42)
    )
    options = dialog.options()
    assert options["components"] == [2, 2] and options["cut_qz"] == 0.42 and dialog.use_qz.isChecked()
    assert options["method"] == "posterior" and "sigma_res" not in options and "nu_res" not in options
    assert not hasattr(dialog, "method")  # one method: nothing to choose
    dialog.use_qz.setChecked(False)
    assert dialog.options()["cut_qz"] is None
    dialog.close()
    app.processEvents()
