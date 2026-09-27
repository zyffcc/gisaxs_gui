"""Scientific candidate exports must not use stale manual control values."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np

from src.gimap.features.fitting.presentation.bindings.fit_result_export import FitResultExportMixin
from src.gimap.features.fitting.presentation.bindings.insitu_refinement_lifecycle import (
    InsituRefinementLifecycleMixin,
)
from src.gimap.features.fitting.presentation.bindings.workflow_v5_binding import (
    WorkflowV5BindingMixin,
)


def candidate(side="positive"):
    sign = 1 if side == "positive" else -1
    return dict(
        workflow="native_v5",
        side=side,
        rank=1,
        best_source="stable_amplitude_calibrated",
        model_id="frozen_model",
        forward_version="blue_rc_gauss48_96_v1",
        components=[
            dict(
                type="random_cylinder",
                type_id=2,
                weight=1.0,
                amplitude=1234.0,
                structure_factor=True,
                params=dict(R=1.7, h=12.0, sigma_R=0.2),
            )
        ],
        global_params=dict(
            background=1.2, resolution_amplitude=250000.0, sigma_Res=0.02, nu_Res=3.0
        ),
        unit_contract=dict(
            q="nm^-1", R_h_D="nm", sigma_Res="nm^-1", amplitudes="input intensity units"
        ),
        native_q=(sign * np.arange(1.0, 9.0)).tolist(),
        native_fit=np.arange(1.0, 9.0).tolist(),
        observed=np.arange(1.0, 9.0).tolist(),
        sigma=np.ones(8).tolist(),
        signed_weighted_rms=0.0,
    )


class ExportBinding(FitResultExportMixin):
    def _collect_active_particles(self):
        raise AssertionError("Do not export stale manual controls for native_v5")


def test_native_export_describes_actual_candidate_not_manual_controls():
    binding = ExportBinding()
    binding.fitting = dict(meta=dict(source="native_v5", candidate=candidate(), params={"R1": 999}))
    text = "\n".join(binding._get_fitting_parameter_comment_lines())
    assert "native_v5_candidate_snapshot" in text
    assert "Forward Version: blue_rc_gauss48_96_v1" in text
    assert "Candidate Source: stable_amplitude_calibrated" in text
    assert '"q": "nm^-1"' in text
    assert "component_1_amplitude = 1234" in text
    assert "resolution_amplitude = 250000" in text
    assert "sigma_Res = 0.02" in text
    assert "R1 = 999" not in text and "export error" not in text


def test_native_export_retains_both_side_snapshots_and_legacy_references():
    first, second = candidate(), candidate("negative")
    second["components"][0]["amplitude"] = 4321.0
    second.update(normalizer=500.0, forward_reference={"background": 0.001})
    binding = ExportBinding()
    binding.fitting = dict(
        meta=dict(source="native_v5", candidate=first, side_candidates=[first, second])
    )
    text = "\n".join(binding._get_fitting_parameter_comment_lines())
    assert "Side: positive" in text and "Side: negative" in text
    assert "component_1_amplitude = 1234" in text and "component_1_amplitude = 4321" in text
    assert 'forward_reference: {"background": 0.001}' in text
    assert "normalizer: 500.0" in text


def test_missing_native_snapshot_never_falls_back_to_manual_parameters():
    binding = ExportBinding()
    binding.fitting = dict(meta=dict(source="native_v5"))
    text = "\n".join(binding._get_fitting_parameter_comment_lines())
    assert "No native workflow candidate snapshot available" in text
    assert "export error" not in text


def test_legacy_manual_export_keeps_original_parameter_schema(monkeypatch):
    from src.gimap.features.fitting.presentation.bindings import fit_result_export

    binding = FitResultExportMixin()
    binding.fitting = dict(meta=dict(shapes=["sphere"], params=dict(R1=4.0, BG=0.5)))
    monkeypatch.setattr(
        fit_result_export,
        "_scientific_commands",
        lambda _: SimpleNamespace(model=SimpleNamespace(parameter_names=lambda _: ["R1", "BG"])),
    )
    text = "\n".join(binding._get_fitting_parameter_comment_lines())
    assert "Parameter Source: last_fitting_result" in text
    assert "R1 = 4" in text and "BG = 0.5" in text
    assert "native_v5" not in text and "export error" not in text


def test_insitu_parameter_time_series_contains_side_amplitudes_and_resolution(tmp_path):
    class Binding(WorkflowV5BindingMixin, InsituRefinementLifecycleMixin):
        def _complete_insitu_workflow_fit(self, record, status):
            self.completed = (record, status)

    binding = Binding()
    binding._ai_output_dir = tmp_path
    rows = [candidate(), candidate("negative")]
    rows[1]["global_params"]["sigma_Res"] = 0.03
    rows[1]["components"][0]["amplitude"] = 4321.0
    original = deepcopy(rows)
    binding._on_insitu_ai_full_fit_finished(
        {}, 0, SimpleNamespace(candidates=rows, runtime_seconds=0.5)
    )
    assert binding.completed[1] == "ok"
    params = binding._current_fitting_parameter_dict()
    assert params["positive_component_1_amplitude"] == 1234.0
    assert params["negative_component_1_amplitude"] == 4321.0
    assert params["positive_sigma_Res"] == 0.02 and params["negative_sigma_Res"] == 0.03
    assert params["positive_background"] == 1.2
    assert params["negative_resolution_amplitude"] == 250000.0
    assert params["positive_component_1_type_id"] == 2.0
    record = binding.completed[0]
    assert record["v5_forward_versions"]["negative"] == "blue_rc_gauss48_96_v1"
    assert record["v5_candidate_sources"]["positive"] == "stable_amplitude_calibrated"
    assert record["v5_unit_contracts"]["positive"]["q"] == "nm^-1"
    assert rows == original
