"""The manual parameter store of Fitting keeps one block of values per shape, whatever the name is spelled like.

The store (``model_parameters.json``) keeps the values of each component in a block named after its shape:
``sphere``, ``cylinder``, ``vertical_cylinder``. The former Fitting page — built but not shown any more; the
pages people see keep their model themselves — takes a model from a candidate chosen in Fit Settings &
Batch (1D Predict), from an earlier candidate table and from Single analysis when its series set-up is
captured, and names a shape as it shows it: "Vertical Cylinder". The store only lower-cased that name, so
a vertical cylinder put into the page was saved under ``vertical cylinder`` while the page read
``vertical_cylinder``: its fields kept the numbers they had, and so did the model drawn from them, the
series set-up and every parameter file copied from the store. Sphere and Cylinder have no space in their
names and were right.

Now every spelling means the one block (``shape_block_name``). A file an earlier version left with the
other spelling keeps loading and is not rewritten by reading it:

* only the other spelling there: its values are the shape's values;
* both there: the proper block wins — it is what the page showed and where an edit or a refinement wrote —
  and the other one only completes it;
* the next time a value of that shape is saved, the values go into the proper block and the other spelling
  leaves the file, so the file and every series record copied from it hold one set of numbers per shape.

The names of a family are also read one way where a candidate is reviewed and loaded: "random_cylinder"
(the quick fit, 1D Predict, Analyze) is the page's Cylinder, as "random cylinder" already was.

And the page reads the shapes in use by the selected entry of each shape box, not by the text the box
shows: the interface language renames that text, so in Chinese no shape was one the model knew, the
renamed "None" counted as a component and the model put into the page was not drawn.

Every number in this file is made up.
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PyQt5.QtWidgets import QMessageBox

from src.gimap.features.fitting.application import ManageFittingModelParameters
from src.gimap.features.fitting.application.single_fit import FitModel, evaluate, new_component, to_manual
from src.gimap.features.fitting.domain.candidates import candidate_parameter_mapping
from src.gimap.features.fitting.domain.physical_constraints import ConstraintSet, normalize_geometry
from src.gimap.features.fitting.domain.scattering_model import _shape_key as model_shape_key
from src.gimap.features.fitting.infrastructure.adapters import FittingModelParametersAdapter
from src.gimap.features.fitting.infrastructure.adapters.model_parameters_store import (
    ModelParametersManager,
    shape_block_name,
)
from src.gimap.features.fitting.presentation.bindings.particle_registry import ParticleRegistryMixin
from src.gimap.features.fitting.presentation.scientific_commands import (
    COMPONENT_ORDER,
    COMPONENT_PARAMETER_SCHEMAS,
)

Q_NM = np.linspace(0.08, 2.0, 240)
SPELLINGS = ("Vertical Cylinder", "vertical cylinder", "vertical_cylinder", "Vertical-Cylinder", " VERTICAL CYLINDER ")
BLOCKS = {"sphere", "cylinder", "vertical_cylinder"}
"""The blocks the store builds and the shipped template has."""
OTHER = {"intensity": 3.0, "radius": 6.25, "sigma_radius": 0.25, "diameter": 41.0, "sigma_diameter": 8.2}
"""What an earlier version saved under "vertical cylinder" when a model was put into the page."""
OWN = {"intensity": 1.5, "radius": 9.5, "sigma_radius": 0.3, "diameter": 55.0, "sigma_diameter": 11.0}
"""What the page showed meanwhile, from ``vertical_cylinder``."""


def _store(path: Path) -> ManageFittingModelParameters:
    return ManageFittingModelParameters(FittingModelParametersAdapter(path))


def _saved(path: Path, particle: str = "particle_1") -> dict:
    return json.loads(path.read_text(encoding="utf-8"))["fitting"]["particles"][particle]["parameters"]


def _earlier_file(path: Path, *, own: dict | None) -> bytes:
    """A user's file as an earlier version left it: a vertical cylinder whose values were saved under
    "vertical cylinder" — beside the block the page reads (``own``), or without it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    _store(path)  # no file yet: the store writes its defaults
    document = json.loads(path.read_text(encoding="utf-8"))
    particle = document["fitting"]["particles"]["particle_1"]
    particle.update(shape="Vertical Cylinder", enabled=True)
    particle["parameters"].pop("vertical_cylinder")
    if own is not None:
        particle["parameters"]["vertical_cylinder"] = dict(own)
    particle["parameters"]["vertical cylinder"] = dict(OTHER)
    path.write_text(json.dumps(document, indent=4), encoding="utf-8")
    return path.read_bytes()


@contextmanager
def _window(home: Path, monkeypatch):
    """The main window on Fitting with ``home`` as the user folder: (binding, window, the model file)."""
    from tests.test_fitting_presentation import _fitting_window

    monkeypatch.setenv("GIMAP_HOME", str(home))
    app, window = _fitting_window()
    try:
        yield window.runtime.fitting, window, home / "model_parameters.json"
    finally:
        window.close()
        app.processEvents()


def _shown(binding, widget: int) -> dict:
    """What the fields of one component of the former page show, by the store's names."""
    shape = binding.get_particle_shape(widget)
    return {row[0]: getattr(binding.ui, f"fitParticle{shape.replace(' ', '')}{row[1]}Value_{widget}").value()
            for row in COMPONENT_PARAMETER_SCHEMAS[shape]}


def _near(values: dict):
    """The same values as far as a field of the page holds them (12 decimals)."""
    return pytest.approx(values, rel=1e-9, abs=1e-11)


def _two_sided_curve(folder: Path, model: FitModel) -> Path:
    """A horizontal cut through the beam (both halves), as Analyze writes it: q in Å⁻¹."""
    exact = evaluate(model.with_values({(0, "Int"): 2000.0, ("globals", "background"): 3.0}), Q_NM)
    q = np.concatenate([-Q_NM[::-1], Q_NM]) / 10.0
    intensity = np.concatenate([exact[::-1], exact])
    path = folder / "made_up_fit_input.dat"
    np.savetxt(path, np.column_stack([q, intensity, np.sqrt(intensity) + 0.5]))
    return path


# -- one rule for the name ------------------------------------------------------------------------


def test_the_store_the_page_and_the_model_name_a_block_the_same_way(tmp_path) -> None:
    page_key = ParticleRegistryMixin._shape_key
    built = ModelParametersManager(str(tmp_path / "model_parameters.json"))._build_default_shape_parameters()
    assert set(built) == BLOCKS
    shapes = [shape for shape in COMPONENT_ORDER if shape != "None"]
    assert shapes == list(COMPONENT_PARAMETER_SCHEMAS)
    for shape in shapes:  # every shape the page offers has its block, under the name the page reads
        assert shape_block_name(shape) in built, shape
        for spelling in (shape, shape.lower(), shape.upper(), shape.replace(" ", "_"), shape.replace(" ", "-"), f" {shape} "):
            assert shape_block_name(spelling) == page_key(None, spelling) == model_shape_key(spelling) == shape_block_name(shape)
    assert {shape_block_name(spelling) for spelling in SPELLINGS} == {"vertical_cylinder"}


@pytest.mark.parametrize("spelling", SPELLINGS)
def test_a_vertical_cylinder_is_saved_in_the_block_every_reader_uses(spelling: str, tmp_path) -> None:
    path = tmp_path / "model_parameters.json"
    store = _store(path)

    assert store.set_particle_parameter("fitting", "particle_1", spelling, "radius", 7.25)
    assert store.save_parameters()

    blocks = _saved(path)
    assert set(blocks) == BLOCKS  # no second block beside the three the page has
    assert blocks["vertical_cylinder"]["radius"] == 7.25
    again = _store(path)
    for name in SPELLINGS:
        assert again.get_particle_parameter("fitting", "particle_1", name, "radius") == 7.25, name
        assert again.get_particle_parameter("fitting", "particle_1", name) == blocks["vertical_cylinder"], name


@pytest.mark.parametrize("spelling, block", [("Sphere", "sphere"), ("SPHERE", "sphere"), ("Cylinder", "cylinder"),
                                             ("cylinder", "cylinder")])
def test_sphere_and_cylinder_keep_their_blocks(spelling: str, block: str, tmp_path) -> None:
    path = tmp_path / "model_parameters.json"
    store = _store(path)
    assert store.set_particle_parameter("fitting", "particle_2", spelling, "radius", 3.75)
    assert store.save_parameters()
    assert set(_saved(path, "particle_2")) == BLOCKS and _saved(path, "particle_2")[block]["radius"] == 3.75
    assert _store(path).get_particle_parameter("fitting", "particle_2", block, "radius") == 3.75


def test_a_value_without_a_shape_is_refused_as_before(tmp_path) -> None:
    path = tmp_path / "model_parameters.json"
    store = _store(path)
    assert store.set_particle_parameter("fitting", "particle_1", None, "radius", 3.75) is False
    assert store.save_parameters() and set(_saved(path)) == BLOCKS
    assert store.get_particle_parameter("fitting", "particle_1")["shape"] == "Sphere"  # no shape asked: the component


# -- a file an earlier version left ----------------------------------------------------------------


def test_a_file_with_only_the_other_spelling_is_read_and_becomes_the_block_at_the_next_save(tmp_path) -> None:
    path = tmp_path / "model_parameters.json"
    before = _earlier_file(path, own=None)
    store = _store(path)

    for name in SPELLINGS:  # its values are the shape's values: the only ones the file has
        assert store.get_particle_parameter("fitting", "particle_1", name) == OTHER, name
        for key, value in OTHER.items():
            assert store.get_particle_parameter("fitting", "particle_1", name, key) == value
    assert store.get_particle_shape("fitting", "particle_1") == "Vertical Cylinder"
    assert path.read_bytes() == before  # reading rewrites nothing

    assert store.set_particle_parameter("fitting", "particle_1", "Vertical Cylinder", "radius", 6.75)
    assert store.save_parameters()
    blocks = _saved(path)
    assert set(blocks) == BLOCKS  # renamed, not copied: one block
    assert blocks["vertical_cylinder"] == {**OTHER, "radius": 6.75}  # every other value kept
    assert _store(path).get_particle_parameter("fitting", "particle_1", "vertical_cylinder") == blocks["vertical_cylinder"]


def test_with_both_spellings_the_block_the_page_showed_wins_and_the_other_leaves_at_the_next_save(tmp_path) -> None:
    path = tmp_path / "model_parameters.json"
    before = _earlier_file(path, own=OWN)
    store = _store(path)

    for name in SPELLINGS:
        assert store.get_particle_parameter("fitting", "particle_1", name) == OWN, name
        assert store.get_particle_parameter("fitting", "particle_1", name, "radius") == OWN["radius"]
    assert path.read_bytes() == before

    # Other values of the document are saved: that shape was not, and its two blocks stay as they are.
    assert store.set_particle_parameter("fitting", "particle_1", "Sphere", "radius", 4.5)
    assert store.set_global_parameter("fitting", "background", 0.75)
    assert store.save_parameters()
    blocks = _saved(path)
    assert blocks["vertical cylinder"] == OTHER and blocks["vertical_cylinder"] == OWN

    assert store.set_particle_parameter("fitting", "particle_1", "vertical_cylinder", "sigma_radius", 0.35)
    assert store.save_parameters()
    blocks = _saved(path)
    assert set(blocks) == BLOCKS
    assert blocks["vertical_cylinder"] == {**OWN, "sigma_radius": 0.35}  # what was read before the write


def test_a_value_only_the_other_spelling_has_completes_the_block(tmp_path) -> None:
    path = tmp_path / "model_parameters.json"
    partial = {key: value for key, value in OWN.items() if key != "sigma_diameter"}
    before = _earlier_file(path, own=partial)
    store = _store(path)

    expected = {**OTHER, **partial}
    assert store.get_particle_parameter("fitting", "particle_1", "Vertical Cylinder") == expected
    assert store.get_particle_parameter("fitting", "particle_1", "Vertical Cylinder", "sigma_diameter") == 8.2
    assert path.read_bytes() == before

    assert store.set_particle_parameter("fitting", "particle_1", "Vertical Cylinder", "intensity", 2.5)
    assert store.save_parameters()
    assert _saved(path) == {**_saved(path), "vertical_cylinder": {**expected, "intensity": 2.5}}
    assert set(_saved(path)) == BLOCKS


def test_a_model_section_of_an_earlier_version_put_into_the_store_reads_the_same_way(tmp_path) -> None:
    """Load Fitting Parameters… and a project put a whole saved section into the store (``replace_section``)."""
    source = tmp_path / "earlier" / "model_parameters.json"
    _earlier_file(source, own=None)
    section = json.loads(source.read_text(encoding="utf-8"))["fitting"]
    path = tmp_path / "mine" / "model_parameters.json"
    store = _store(path)

    store.replace_section("fitting", section)

    assert store.get_particle_parameter("fitting", "particle_1", "vertical_cylinder") == OTHER
    assert store.save_parameters()
    assert _saved(path) == section["particles"]["particle_1"]["parameters"]  # saved as it came: nothing was written to it
    assert _store(path).get_particle_parameter("fitting", "particle_1", "Vertical Cylinder", "radius") == OTHER["radius"]


def test_the_former_page_shows_an_earlier_file_and_an_edit_makes_it_one_block(tmp_path, monkeypatch) -> None:
    home = tmp_path / "home"
    before = _earlier_file(home / "model_parameters.json", own=None)
    with _window(home, monkeypatch) as (binding, window, path):
        widget = binding._iter_particle_widget_ids()[0]
        assert binding.get_particle_shape(widget) == "Vertical Cylinder"
        assert _shown(binding, widget) == OTHER  # not the fields' own defaults
        assert path.read_bytes() == before  # opening the page rewrites nothing

        field = window.fitParticleVerticalCylinderRValue_1
        field.setValue(6.75)
        field.editingFinished.emit()  # the page's own way to save an edited value

        assert set(_saved(path)) == BLOCKS
        assert _saved(path)["vertical_cylinder"] == {**OTHER, "radius": 6.75}
        assert _shown(binding, widget) == {**OTHER, "radius": 6.75}
    assert _saved(path)["vertical_cylinder"] == {**OTHER, "radius": 6.75}  # also after the window closed


def test_a_labs_parameter_file_of_an_earlier_version_loads_and_the_next_one_saved_has_one_block(tmp_path, monkeypatch) -> None:
    """File ▸ Save / Load Labs Parameters… carries the stored model: a file saved by an earlier version after a
    vertical cylinder was sent to the page holds both blocks."""
    from src.gimap.app.workspace_parameters import WorkspaceParameterCoordinator
    from src.gimap.integrations.state.project_parameters import JsonProjectParametersRepository

    with _window(tmp_path / "home", monkeypatch) as (binding, window, path):
        runtime = window.runtime  # the test window has no file store for these commands: the application's one
        labs = WorkspaceParameterCoordinator(
            repository=JsonProjectParametersRepository(), trainset=runtime.trainset, fitting=runtime.fitting,
            prediction=runtime.prediction, status=lambda message: None)
        saved = tmp_path / "made_up_labs_parameters.json"
        assert labs.save(str(saved))
        document = json.loads(saved.read_text(encoding="utf-8"))
        particle = document["fitting_model_parameters"]["fitting"]["particles"]["particle_1"]
        particle.update(shape="Vertical Cylinder", enabled=True)
        particle["parameters"].update({"vertical_cylinder": dict(OWN), "vertical cylinder": dict(OTHER)})
        saved.write_text(json.dumps(document), encoding="utf-8")

        assert labs.load(str(saved))

        widget = binding._iter_particle_widget_ids()[0]
        assert binding.get_particle_shape(widget) == "Vertical Cylinder"
        assert _shown(binding, widget) == OWN  # what the earlier version showed with this file
        assert _saved(path)["vertical cylinder"] == OTHER and _saved(path)["vertical_cylinder"] == OWN  # as it came

        sent = to_manual(FitModel((new_component("vertical_cylinder", radius=7.0, distance=44.0),)))
        binding._load_parameter_mapping(sent)  # the next model sent to the page

        assert _shown(binding, widget) == _near(sent.components[0].parameters)
        assert set(_saved(path)) == BLOCKS and _saved(path)["vertical_cylinder"] == _near(sent.components[0].parameters)
        again = tmp_path / "made_up_labs_parameters_now.json"
        assert labs.save(str(again))
        blocks = json.loads(again.read_text(encoding="utf-8"))["fitting_model_parameters"]["fitting"]["particles"]["particle_1"]["parameters"]
        assert set(blocks) == BLOCKS and blocks["vertical_cylinder"] == _near(sent.components[0].parameters)


# -- the former page: every way a model reaches it --------------------------------------------------


def test_a_model_put_into_the_former_page_shows_its_own_values(tmp_path, monkeypatch) -> None:
    """``_load_parameter_mapping``: where every sender ends. Sphere and Cylinder were right before."""
    with _window(tmp_path / "home", monkeypatch) as (binding, _window_, path):
        first, second = binding._iter_particle_widget_ids()[:2]
        for family, radius, distance in (("sphere", 6.5, 39.0), ("vertical_cylinder", 7.0, 44.0),
                                         ("cylinder", 4.0, 31.0), ("vertical_cylinder", 8.0, 47.0)):
            sent = to_manual(FitModel((new_component(family, radius=radius, distance=distance),)))
            binding._load_parameter_mapping(sent)
            component = sent.components[0]
            assert binding.get_particle_shape(first) == component.shape, family
            assert _shown(binding, first) == _near(component.parameters), family
            assert set(_saved(path)) == BLOCKS and _saved(path)[family] == _near(component.parameters), family

        both = to_manual(FitModel((new_component("sphere", radius=5.5, distance=33.0),
                                   new_component("vertical_cylinder", radius=9.0, distance=58.0))))
        binding._load_parameter_mapping(both)
        assert binding.get_particles_status()[first] == "Sphere"
        assert binding.get_particles_status()[second] == "Vertical Cylinder"
        assert _shown(binding, second) == _near(both.components[1].parameters)
        assert _saved(path, f"particle_{second}")["vertical_cylinder"] == _near(both.components[1].parameters)
        assert set(_saved(path, f"particle_{second}")) == BLOCKS

        # A value edited in a field afterwards goes to the same block.
        field = getattr(binding.ui, f"fitParticleVerticalCylinderDValue_{second}")
        field.setValue(61.0)
        field.editingFinished.emit()
        assert _saved(path, f"particle_{second}")["vertical_cylinder"]["diameter"] == 61.0
        assert set(_saved(path, f"particle_{second}")) == BLOCKS
        assert _shown(binding, second) == _near({**both.components[1].parameters, "diameter": 61.0})


def test_in_situ_takes_a_vertical_cylinder_of_single_analysis_and_records_it_once(tmp_path, monkeypatch) -> None:
    model = FitModel((new_component("vertical_cylinder", radius=7.0, distance=44.0),))
    sent = to_manual(model).components[0]
    with _window(tmp_path / "home", monkeypatch) as (binding, window, path):
        components = window.components
        components._send_curve_to_fitting(str(_two_sided_curve(tmp_path, model)), "mean")
        page = components.fitting_workspace.fit_page
        page.set_model(model)

        components.fitting_workspace.insitu_series_page.capture_recipe_requested.emit()  # Use current setup

        widget = binding._iter_particle_widget_ids()[0]
        assert binding.get_particle_shape(widget) == "Vertical Cylinder"
        assert _shown(binding, widget) == _near(sent.parameters)
        assert (_shown(binding, widget)["radius"], _shown(binding, widget)["diameter"]) == (7.0, 44.0)
        recipe = binding.fitting_view_model.insitu.recipe
        recorded = recipe.to_dict()["model"]["model_parameters"]["fitting"]["particles"][f"particle_{widget}"]
        assert recorded["shape"] == "Vertical Cylinder" and recorded["enabled"] is True
        assert set(recorded["parameters"]) == BLOCKS  # one set of numbers per shape in the series record
        assert recorded["parameters"]["vertical_cylinder"] == _near(sent.parameters)
        assert _saved(path)["vertical_cylinder"] == _near(sent.parameters)


def test_a_1d_predict_candidate_with_a_vertical_cylinder_is_shown_and_drawn_with_its_values(tmp_path, monkeypatch) -> None:
    from src.gimap.features.fitting.infrastructure.adapters.experimental_fit import forward

    params = {"R": 8.5, "sigma_R": 0.25, "D": 52.0, "sigma_D": 0.3}
    globals_ = {"background": 1.3, "resolution_amplitude": 800.0, "sigma_Res": 0.02, "nu_Res": 4.0}
    components = [{"type": "vertical_cylinder", "type_id": 3, "weight": 1.0, "amplitude": 1200.0, "params": params}]
    fit = forward(Q_NM, components, globals_)
    row = {"workflow": "native_v5", "combination": "vertical_cylinder", "components": components,
           "global_params": dict(globals_), "native_q": (-Q_NM).tolist(), "native_fit": fit.tolist()}
    curve = tmp_path / "made_up_fit_input.dat"
    np.savetxt(curve, np.column_stack([-Q_NM / 10.0, fit, np.sqrt(fit)]))
    with _window(tmp_path / "home", monkeypatch) as (binding, window, path):
        binding.import_1d_file(curve, q_view="fold")
        window.components.fitting_workspace.show_fit_curve()

        assert binding.show_workflow_candidate(row) is True  # what choosing a row in Fit Settings & Batch calls

        widget = binding._iter_particle_widget_ids()[0]
        shown = _shown(binding, widget)
        assert binding.get_particle_shape(widget) == "Vertical Cylinder"
        assert (shown["radius"], shown["sigma_radius"], shown["diameter"]) == (8.5, 0.25, 52.0)  # σR/R for this shape
        assert shown["sigma_diameter"] == pytest.approx(52.0 * 0.3)  # σD in nm on the page
        assert shown["intensity"] > 1.0  # found for Fitting's model of it: not the field's default
        drawn = binding.fitting["meta"]["params"]  # the model the page drew on the data
        assert (drawn["R1"], drawn["sigma_R1"], drawn["D1"]) == (8.5, 0.25, 52.0)
        assert drawn["Int1"] == pytest.approx(shown["intensity"])
        assert set(_saved(path)) == BLOCKS and _saved(path)["vertical_cylinder"] == _near(shown)
        assert "vertical cylinder is in Components and Global" in window.fittingInlineFeedback.text()


def test_in_chinese_the_former_page_reads_its_shapes_by_their_own_names(tmp_path, monkeypatch) -> None:
    """The interface language renames what the shape boxes show ("竖直圆柱", "无"); the page read its
    active shapes from that text, so in Chinese no shape was known, "无" (None) counted as a component
    and the model of a chosen candidate was not drawn, while the note said it was."""
    from src.gimap.app.presentation.i18n import DEFAULT_LANGUAGE, apply_language
    from src.gimap.features.fitting.infrastructure.adapters.experimental_fit import forward

    params = {"R": 8.5, "sigma_R": 0.25, "D": 52.0, "sigma_D": 0.3}
    globals_ = {"background": 1.3, "resolution_amplitude": 800.0, "sigma_Res": 0.02, "nu_Res": 4.0}
    components = [{"type": "vertical_cylinder", "type_id": 3, "weight": 1.0, "amplitude": 1200.0, "params": params}]
    fit = forward(Q_NM, components, globals_)
    row = {"workflow": "native_v5", "combination": "vertical_cylinder", "components": components,
           "global_params": dict(globals_), "native_q": (-Q_NM).tolist(), "native_fit": fit.tolist()}
    curve = tmp_path / "made_up_fit_input.dat"
    np.savetxt(curve, np.column_stack([-Q_NM / 10.0, fit, np.sqrt(fit)]))
    with _window(tmp_path / "home", monkeypatch) as (binding, window, path):
        binding.import_1d_file(curve, q_view="fold")
        window.components.fitting_workspace.show_fit_curve()
        assert binding.show_workflow_candidate(row) is True
        english = (binding._collect_active_particles(), dict(binding.fitting["meta"]["params"]))
        assert english[0] == (["vertical cylinder"], [binding._iter_particle_widget_ids()[0]])
        try:
            apply_language("zh", [window])  # as View ▸ Language does, on every open window
            widget = binding._iter_particle_widget_ids()[0]
            box = getattr(binding.ui, f"fitParticleShapeCombox_{widget}")
            assert box.currentText() != "Vertical Cylinder"  # the box shows the Chinese name
            binding.fitting = None

            assert binding.show_workflow_candidate(row) is True

            assert binding._collect_active_particles() == english[0]  # the shape's own name; None is no component
            assert binding.get_particle_shape(widget) == "Vertical Cylinder"
            shown = _shown(binding, widget)
            assert (shown["radius"], shown["sigma_radius"], shown["diameter"]) == (8.5, 0.25, 52.0)
            assert binding.fitting is not None, "the model of the candidate was not drawn"
            assert binding.fitting["meta"]["shapes"] == ["vertical cylinder"]
            assert binding.fitting["meta"]["params"] == pytest.approx(english[1])  # the same model in both languages
            assert set(_saved(path)) == BLOCKS and _saved(path)["vertical_cylinder"] == _near(shown)
        finally:
            apply_language(DEFAULT_LANGUAGE, [window])
        assert box.currentText() == "Vertical Cylinder"


def test_a_candidate_of_the_earlier_pipeline_is_loaded_as_its_own_shape(tmp_path, monkeypatch) -> None:
    """Rows without ``workflow`` (``top20_candidates.json`` of the research pipeline) go through
    ``candidate_parameter_mapping``; a random cylinder named with an underscore became a Sphere there."""
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *args, **kwargs: warnings.append(args[1:])))
    rows = {
        "Vertical Cylinder": {"type": "vertical_cylinder", "weight": 0.75,
                              "params": {"R": 6.0, "sigma_R": 0.2, "D": 40.0, "sigma_D": 8.0}},
        "Cylinder": {"type": "random_cylinder", "weight": 0.5,
                     "params": {"R": 4.0, "sigma_R": 0.4, "h": 12.0, "sigma_h": 1.2, "D": 30.0, "sigma_D": 6.0}},
    }
    names = {"R": "radius", "sigma_R": "sigma_radius", "h": "height", "sigma_h": "sigma_height",
             "D": "diameter", "sigma_D": "sigma_diameter"}
    with _window(tmp_path / "home", monkeypatch) as (binding, _window_, path):
        widget = binding._iter_particle_widget_ids()[0]
        for shape, component in rows.items():
            row = {"rank": 1, "combination": component["type"], "components": [component], "global_params": {"BG": 0.5}}
            assert binding._load_ai_candidate_params(row, refresh_plot=False) is True, shape
            assert binding.get_particle_shape(widget) == shape
            expected = {"intensity": component["weight"], **{names[key]: value for key, value in component["params"].items()}}
            assert _shown(binding, widget) == expected, shape
            assert set(_saved(path)) == BLOCKS and _saved(path)[shape_block_name(shape)] == expected
    assert warnings == []


# -- the names of a family ---------------------------------------------------------------------------

FAMILY_SPELLINGS = [
    ("sphere", "Sphere", "sphere"), ("Sphere", "Sphere", "sphere"), (" SPHERE ", "Sphere", "sphere"),
    ("cylinder", "Cylinder", "cylinder"), ("Cylinder", "Cylinder", "cylinder"),
    ("random_cylinder", "Cylinder", "cylinder"), ("random cylinder", "Cylinder", "cylinder"),
    ("Random cylinder", "Cylinder", "cylinder"), ("random-cylinder", "Cylinder", "cylinder"),
    ("isotropic cylinder", "Cylinder", "cylinder"), ("isotropic_cylinder", "Cylinder", "cylinder"),
    ("vertical_cylinder", "Vertical Cylinder", "vertical_cylinder"),
    ("vertical cylinder", "Vertical Cylinder", "vertical_cylinder"),
    ("Vertical cylinder", "Vertical Cylinder", "vertical_cylinder"),
    ("Vertical-Cylinder", "Vertical Cylinder", "vertical_cylinder"),
    ("VERTICAL CYLINDER", "Vertical Cylinder", "vertical_cylinder"),
    ("verticalcylinder", "Vertical Cylinder", "vertical_cylinder"),  # a spelling the candidate mapping knew
]


@pytest.mark.parametrize("name, shape, geometry", FAMILY_SPELLINGS)
def test_a_family_is_reviewed_and_loaded_as_the_same_shape_however_it_is_named(name, shape, geometry) -> None:
    params = {"R": 10.0, "sigma_R": 0.5, "h": 20.0, "sigma_h": 2.0, "D": 5.0, "sigma_D": 1.0}  # D inside the particle
    mapping = candidate_parameter_mapping({"components": [{"type": name, "params": params}]})

    assert mapping.components[0].shape == shape
    assert shape in COMPONENT_PARAMETER_SCHEMAS  # a shape the page has: not turned into a Sphere there
    assert normalize_geometry(name) == geometry == shape_block_name(shape)  # family, and the block of its values
    violated = {item.constraint_id for item in ConstraintSet.defaults().validate_components([{"type": name, "params": params}])}
    assert "hard_core_spacing" in violated  # the constraints of its family apply to it


def test_an_unknown_family_keeps_its_name() -> None:
    assert normalize_geometry("core_shell") == normalize_geometry("Core-Shell") == "core_shell"
    assert candidate_parameter_mapping({"components": [{"type": "core_shell", "params": {"R": 3.0}}]}).components[0].shape == "core_shell"
    assert candidate_parameter_mapping({"components": [{"params": {"R": 3.0}}]}).components[0].shape == "Sphere"  # no type given
