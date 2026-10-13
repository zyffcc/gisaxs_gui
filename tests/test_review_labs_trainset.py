"""Review fixes, Labs / Trainset: form and preview never overlap, loaded parameters are not
validated, and a design edit after a preview resets every design check."""

from __future__ import annotations

import copy

import numpy as np
from PyQt5.QtCore import Qt

from src.gimap.features.trainset.presentation.sections.responsive_style import _pane_min_width
from tests.test_labs_trainset import _settle, _trainset


def _gates(page):
    table = page.preview_gate_table
    return [table.item(row, 1).text() for row in range(table.rowCount())]


def _previewed(page):
    """The page after a successful local preview (what _preview_finished leaves)."""
    for row in range(3):
        page.preview_gate_table.item(row, 1).setText("Ready")
    page.set_validation_state("Preview ready", "ok")
    page.set_step_state(1, "Preview ready")
    page.set_step_state(2, "Contract ready")
    page.set_step_state(3, "Package ready")


def test_form_and_preview_are_side_by_side_only_when_both_fit(tmp_path):
    window, binding, page = _trainset(tmp_path)
    window.show()
    window.runtime.navigate("trainset")
    page.step_list.setCurrentRow(0)
    _settle(0.3)
    splitter = page.dataset_splitter
    form, preview = splitter.widget(0), splitter.widget(1)
    need = page._dataset_side_by_side_width()
    overhead = window.width() - page.stack.width()  # sidebar, step list, margins, handles
    edge = need + overhead  # the narrowest window that fits both panes
    widths = list(range(edge - 90, edge + 91, 15))
    seen = set()
    for width in widths + widths[::-1]:  # wider and narrower again
        window.resize(width, 800)
        _settle(0.12)
        horizontal = splitter.orientation() == Qt.Horizontal
        seen.add(horizontal)
        if horizontal:
            sizes = splitter.sizes()
            assert sizes[0] >= _pane_min_width(form), (width, page.stack.width(), sizes)
            assert sizes[1] >= _pane_min_width(preview), (width, page.stack.width(), sizes)
        else:
            assert page.stack.width() < need, (width, page.stack.width(), need)
    assert seen == {True, False}  # the sweep crossed the switch
    window.close()


def _other_design(binding, tmp_path, *, new_reference: bool):
    other = copy.deepcopy(binding.get_parameters())
    other["project"]["name"] = "another_project"
    other["roi"]["width"] = max(4, int(other["roi"].get("width", 64)) // 2)
    if new_reference:
        path = tmp_path / "other.npy"
        np.save(path, (np.random.default_rng(0).random((80, 100)) * 50 + 1).astype(np.float32))
        other["project"]["reference_file"] = str(path)
    else:
        other["project"]["reference_file"] = ""
    return other


def test_parameters_loaded_through_the_shell_are_not_validated(tmp_path):
    window, binding, page = _trainset(tmp_path)
    for new_reference in (False, True):
        _previewed(page)
        # File ▸ Load parameters: WorkspaceParameterCoordinator.load → trainset.set_parameters
        window.runtime.workspace_parameters.trainset.set_parameters(
            _other_design(binding, tmp_path, new_reference=new_reference)
        )
        _settle(0.1)

        assert page.validation_badge.text() == "Not validated", new_reference
        assert page.validation_state() == "pending", new_reference
        assert _gates(page)[:3] == ["Pending"] * 3, new_reference
        assert page.step_states()[1:4] == ["Not started"] * 3, new_reference
        assert {"Configuration valid", "Local samples generated", "Tensor shapes compatible"} <= set(
            binding._missing_submission_gates()
        )
    window.close()


def test_a_design_edit_after_a_preview_resets_every_design_check(tmp_path):
    window, binding, page = _trainset(tmp_path)
    _previewed(page)

    page.fields["roi.width"].setValue(page.fields["roi.width"].value() - 4)  # a new tensor shape

    assert page.validation_state() == "warn"
    assert page.validation_badge.text() == "Changed since validation"
    assert _gates(page)[:3] == ["Pending"] * 3
    assert page.step_states()[1:3] == ["Not started"] * 2
    # Validating again is not enough to submit: the preview and the shape check are pending.
    page.preview_gate_table.item(0, 1).setText("Ready")
    page.set_validation_state("Configuration valid", "ok")
    missing = binding._missing_submission_gates()
    assert "Local samples generated" in missing and "Tensor shapes compatible" in missing

    for name in ("particle_parameter_table", "layer_table", "model_layer_table", "mask_shape_table"):
        table = getattr(page, name)
        if table.rowCount() == 0 or table.item(0, 1) is None:
            continue
        _previewed(page)
        table.item(0, 1).setText(table.item(0, 1).text() + " ")
        assert _gates(page)[:3] == ["Pending"] * 3, name
    window.close()


def test_job_settings_and_the_project_name_keep_the_design_checks(tmp_path):
    window, binding, page = _trainset(tmp_path)
    _previewed(page)
    steps = page.step_states()

    page.fields["hpc.user"].setText("someone_else")
    page.fields["training.epochs"].setValue(page.fields["training.epochs"].value() + 1)
    page.fields["runtime.dataset_output_dir"].setText(str(tmp_path / "dataset"))
    page.project_name.setText("renamed_project")

    assert page.validation_state() == "ok"
    assert page.validation_badge.text() == "Preview ready"
    assert _gates(page)[:3] == ["Ready"] * 3
    assert page.step_states() == steps
    window.close()
