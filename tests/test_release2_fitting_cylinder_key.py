"""The manual parameter store of Fitting keeps one block of values per shape, whatever the name is spelled like.

The store (``model_parameters.json``) keeps the values of each component in a block named after its shape:
``sphere``, ``cylinder``, ``vertical_cylinder``. Fitting's former page named a shape as it showed it,
"Vertical Cylinder", and the store only lower-cased that name, so a vertical cylinder was saved under
``vertical cylinder`` beside ``vertical_cylinder``. Sphere and Cylinder have no space in their names and
were right.

Now every spelling means the one block (``shape_block_name``). A file an earlier version left with the
other spelling keeps loading and is not rewritten by reading it:

* only the other spelling there: its values are the shape's values;
* both there: the proper block wins — it is what the page showed and where an edit or a refinement wrote —
  and the other one only completes it;
* the next time a value of that shape is saved, the values go into the proper block and the other spelling
  leaves the file, so the file and every series record copied from it hold one set of numbers per shape.

The names of a family are also read one way where a candidate is reviewed and loaded: "random_cylinder"
(the quick fit, 1D Predict, Analyze) is the store's Cylinder, as "random cylinder" already was.

Every number in this file is made up.
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from src.gimap.features.fitting.application import ManageFittingModelParameters
from src.gimap.features.fitting.domain.candidates import candidate_parameter_mapping
from src.gimap.features.fitting.domain.physical_constraints import ConstraintSet, normalize_geometry
from src.gimap.features.fitting.domain.scattering_model import _shape_key as model_shape_key
from src.gimap.features.fitting.infrastructure.adapters import FittingModelParametersAdapter
from src.gimap.features.fitting.infrastructure.adapters.model_parameters_store import (
    ModelParametersManager,
    shape_block_name,
)

SPELLINGS = ("Vertical Cylinder", "vertical cylinder", "vertical_cylinder", "Vertical-Cylinder", " VERTICAL CYLINDER ")
SHAPES = ("Sphere", "Cylinder", "Vertical Cylinder")
"""The shapes of the stored model's components."""
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






# -- one rule for the name ------------------------------------------------------------------------


def test_the_store_and_the_model_name_a_block_the_same_way(tmp_path) -> None:
    built = ModelParametersManager(str(tmp_path / "model_parameters.json"))._build_default_shape_parameters()
    assert set(built) == BLOCKS
    for shape in SHAPES:  # every shape has its block
        assert shape_block_name(shape) in built, shape
        for spelling in (shape, shape.lower(), shape.upper(), shape.replace(" ", "_"), shape.replace(" ", "-"), f" {shape} "):
            assert shape_block_name(spelling) == model_shape_key(spelling) == shape_block_name(shape)
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




def test_a_labs_parameter_file_of_an_earlier_version_loads_and_is_saved_again(tmp_path, monkeypatch) -> None:
    """File ▸ Save / Load Labs Parameters… carries the stored model: a file saved by an earlier version with a
    vertical cylinder holds both blocks; it loads, and the stored model is saved again as it came."""
    from src.gimap.app.workspace_parameters import WorkspaceParameterCoordinator
    from src.gimap.integrations.state.project_parameters import JsonProjectParametersRepository

    with _window(tmp_path / "home", monkeypatch) as (binding, window, path):
        runtime = window.runtime  # the test window has no file store for these commands: the application's one
        labs = WorkspaceParameterCoordinator(
            repository=JsonProjectParametersRepository(), fitting=runtime.fitting,
            prediction=runtime.prediction, status=lambda message: None)
        saved = tmp_path / "made_up_labs_parameters.json"
        assert labs.save(str(saved))
        document = json.loads(saved.read_text(encoding="utf-8"))
        particle = document["fitting_model_parameters"]["fitting"]["particles"]["particle_1"]
        particle.update(shape="Vertical Cylinder", enabled=True)
        particle["parameters"].update({"vertical_cylinder": dict(OWN), "vertical cylinder": dict(OTHER)})
        saved.write_text(json.dumps(document), encoding="utf-8")

        assert labs.load(str(saved))

        assert _saved(path)["vertical cylinder"] == OTHER and _saved(path)["vertical_cylinder"] == OWN  # as it came
        again = tmp_path / "made_up_labs_parameters_now.json"
        assert labs.save(str(again))
        blocks = json.loads(again.read_text(encoding="utf-8"))["fitting_model_parameters"]["fitting"]["particles"]["particle_1"]["parameters"]
        assert blocks["vertical_cylinder"] == OWN












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
    assert shape in SHAPES  # a shape of the store: not turned into a Sphere
    assert normalize_geometry(name) == geometry == shape_block_name(shape)  # family, and the block of its values
    violated = {item.constraint_id for item in ConstraintSet.defaults().validate_components([{"type": name, "params": params}])}
    assert "hard_core_spacing" in violated  # the constraints of its family apply to it


def test_an_unknown_family_keeps_its_name() -> None:
    assert normalize_geometry("core_shell") == normalize_geometry("Core-Shell") == "core_shell"
    assert candidate_parameter_mapping({"components": [{"type": "core_shell", "params": {"R": 3.0}}]}).components[0].shape == "core_shell"
    assert candidate_parameter_mapping({"components": [{"params": {"R": 3.0}}]}).components[0].shape == "Sphere"  # no type given
