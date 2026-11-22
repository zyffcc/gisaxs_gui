"""``config/model_parameters.json``: a new user's first model of Fitting holds neutral defaults.

The file is the template of the manual parameter store of Fitting: it is copied into the user data
folder when that folder has no ``model_parameters.json`` (``default_model_parameters_path`` and the
migration of the user store) and never afterwards. It holds the starting model of Fitting
(``starting_model()``: the defaults of the domain's ``new_component`` and ``FitModel``) written in
the units and conventions of the store, those of earlier Fitting — sizes in nm, the spreads of spheres and
random cylinders and σD = D·(σnn/D)/√(2π) in nm, the spread of a vertical cylinder as σR/R and its scale in
10⁻⁶⟨R⁴⟩/4, the resolution width in nm⁻¹ — so the page people see and the store behind it start from the same
model, inside the ranges of the domain and of the physical constraints. The store has no size law: read back,
its sizes are those of earlier Fitting (``LEGACY_LAWS``). No value in it comes from a fit.

A file the user already has is never rewritten by a start — whatever it holds: an earlier version's
components, half a document, nothing — and the copy a new user gets can be written to, also when the
program folder it comes from is read-only.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

import src.gimap.features.fitting.infrastructure.adapters.model_parameters as template_adapter
from src.gimap.features.fitting.application import ManageFittingModelParameters
from dataclasses import replace

from src.gimap.features.fitting.application.single_fit import (
    FitModel,
    evaluate,
    from_manual,
    new_component,
    to_manual,
)
from src.gimap.features.fitting.domain.fit_model import LEGACY_LAWS
from src.gimap.features.fitting.domain.candidates import (
    CandidateComponentParameters,
    CandidateParameterMapping,
)
from src.gimap.features.fitting.domain.physical_constraints import ConstraintSet
from src.gimap.features.fitting.domain.scattering_model import make_mixed_model, params_template
from src.gimap.features.fitting.infrastructure.adapters import FittingModelParametersAdapter
from src.gimap.features.fitting.infrastructure.adapters.model_parameters import (
    TEMPLATE_PATH,
    default_model_parameters_path,
)
from src.gimap.features.fitting.infrastructure.adapters.model_parameters_store import (
    ModelParametersManager,
)
from src.gimap.features.fitting.presentation.single.session import starting_model
from src.gimap.integrations.state import migrate_legacy_files

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "config" / "model_parameters.json"
Q_NM = np.linspace(0.02, 3.0, 400)
STORE_NAMES = {
    "Int": "intensity", "R": "radius", "sigma_R": "sigma_radius", "h": "height", "sigma_h": "sigma_height",
    "D": "diameter", "sigma_D": "sigma_diameter",
    "BG": "background", "sigma_Res": "sigma_res", "nu_Res": "nu_res", "int_Res": "int_res", "k": "k_value",
}
"""The names of the manual model's parameters (``params_template``) in the store."""


STORE_KEYS = {
    "Sphere": ["intensity", "radius", "sigma_radius", "diameter", "sigma_diameter"],
    "Cylinder": ["intensity", "radius", "sigma_radius", "height", "sigma_height", "diameter", "sigma_diameter"],
    "Vertical Cylinder": ["intensity", "radius", "sigma_radius", "diameter", "sigma_diameter"],
}
"""The values of a component's block in ``model_parameters.json``, in the order ``make_mixed_model`` takes them."""

def _template() -> dict:
    return json.loads(TEMPLATE.read_text(encoding="utf-8"))


def _shape_key(shape: str) -> str:
    """How the store names the parameters of a shape ("Vertical Cylinder" → ``vertical_cylinder``)."""
    return shape.strip().lower().replace(" ", "_")


def _active(fitting: dict) -> list[dict]:
    return [entry for entry in fitting["particles"].values() if entry["enabled"] and entry["shape"] != "None"]


def _numbers(node):
    if isinstance(node, dict):
        for value in node.values():
            yield from _numbers(value)
    elif isinstance(node, (int, float)) and not isinstance(node, bool):
        yield float(node)


def _program_folder(folder: Path) -> Path:
    """A program folder with the shipped template and nobody's own files; returns its template."""
    (folder / "config").mkdir(parents=True)
    return Path(shutil.copyfile(TEMPLATE, folder / "config" / "model_parameters.json"))


def test_the_adapter_copies_this_file() -> None:
    assert TEMPLATE_PATH == TEMPLATE and TEMPLATE.is_file()


def test_the_template_is_the_starting_model_of_fitting_in_the_units_of_the_store() -> None:
    fitting = _template()["fitting"]
    active = _active(fitting)
    expected = to_manual(starting_model())

    assert [entry["shape"] for entry in active] == [component.shape for component in expected.components]
    for entry, component in zip(active, expected.components):
        assert entry["parameters"][_shape_key(entry["shape"])] == component.parameters
    assert fitting["global_parameters"] == expected.global_parameters

    # Read back it is that model — values, free or fixed, and ranges: no range had to be widened.
    components = tuple(
        CandidateComponentParameters(entry["shape"], 1.0, entry["parameters"][_shape_key(entry["shape"])])
        for entry in active
    )
    _same_model(from_manual(CandidateParameterMapping(components, fitting["global_parameters"])), starting_model())


def _same_model(read: FitModel, model: FitModel) -> None:
    """``read`` is ``model`` with the store's sizes (``LEGACY_LAWS``): values (to rounding), free or fixed, ranges."""
    assert [(c.family, c.law, c.structure, c.coupling) for c in read.components] == [
        (c.family, LEGACY_LAWS[c.family], c.structure, c.coupling) for c in model.components]
    for (path, got), (_same, expected) in zip(read.parameters(), model.parameters()):
        assert got.value == pytest.approx(expected.value, rel=1e-12), path
        assert (got.free, got.lower, got.upper) == (expected.free, expected.lower, expected.upper), path


@pytest.mark.parametrize("family", sorted(LEGACY_LAWS))
def test_every_component_holds_the_defaults_of_a_new_component_of_each_family(family: str) -> None:
    """Each family representable in the legacy store has the same defaults as a new component."""
    fitting = _template()["fitting"]
    fresh = FitModel((new_component(family),))
    expected = to_manual(fresh).components[0]
    for name, entry in fitting["particles"].items():
        assert entry["parameters"][family] == expected.parameters, name

    block = fitting["particles"]["particle_1"]["parameters"][family]
    mapping = CandidateParameterMapping(
        (CandidateComponentParameters(expected.shape, 1.0, block),), fitting["global_parameters"]
    )
    _same_model(from_manual(mapping), fresh)  # inside the ranges of the domain
    assert ConstraintSet.defaults().validate_components([{"type": family, "params": block}]) == []


@pytest.mark.parametrize("family", sorted(LEGACY_LAWS))
def test_the_store_and_the_domain_draw_the_same_finite_curve(family: str) -> None:
    """The store's manual model on fine rules of its own formulas is the domain's model with the store's sizes.
    (On its default rules it is not: 25 radii and 24 orientations alias at large q, Physics and Methods 9.3.)"""
    fitting = _template()["fitting"]
    block = fitting["particles"]["particle_1"]["parameters"][family]
    values = []
    for name in params_template([family]):
        key = STORE_NAMES[name.rstrip("1")]
        values.append(block[key] if key in block else fitting["global_parameters"][key])
    q = Q_NM[::4]
    fine = make_mixed_model([family], pd_opts={"n_samples": 4001, "nsig": 4.0},
                            cyl_opts={"n_R": 101, "n_h": 101, "n_orient": 2000, "nsig": 4.0})
    curve = np.asarray(fine(q, *values), dtype=float)
    assert np.all(np.isfinite(curve)) and np.all(curve > 0)
    legacy = FitModel((replace(new_component(family), law=LEGACY_LAWS[family]),))
    # the manual model cuts a vertical cylinder's sizes at ±3σ (the domain at ±4σ); its orientations converge as 1/n
    tolerance = {"sphere": 1e-3, "vertical_cylinder": 1e-2, "cylinder": 5e-3}[family]
    np.testing.assert_allclose(curve, evaluate(legacy, q), rtol=tolerance)


def test_every_number_is_a_round_default_and_not_the_result_of_a_fit() -> None:
    """Round, except what the store's conventions make of a round default: σD = D·(σnn/D)/√(2π) and a vertical
    cylinder's scale in units of 10⁻⁶⟨R⁴⟩/4."""
    fitting = _template()["fitting"]
    converted = {"sigma_diameter"}
    values = list(_numbers(fitting["global_parameters"]))
    for entry in fitting["particles"].values():
        for family, block in entry["parameters"].items():
            values += [value for key, value in block.items()
                       if key not in converted and not (family == "vertical_cylinder" and key == "intensity")]
    assert values
    for value in values:
        assert value == float(f"{value:.2g}"), value


def test_the_template_has_the_structure_the_store_builds_and_the_former_page_reads(tmp_path) -> None:
    template = _template()
    built = ModelParametersManager(str(tmp_path / "model_parameters.json"))._parameters  # no file: its own defaults

    assert set(template) == set(built)
    for section in ("gisaxs_predict", "classification", "metadata"):
        assert template[section] == built[section]
    assert set(built["fitting"]["global_parameters"]) <= set(template["fitting"]["global_parameters"])

    particles, built_particles = template["fitting"]["particles"], built["fitting"]["particles"]
    assert list(particles) == list(built_particles)  # the components the page has: nothing left from a session
    schemas = {_shape_key(shape): keys for shape, keys in STORE_KEYS.items()}
    for name, entry in particles.items():
        assert set(entry) == {"shape", "enabled", "parameters"}
        assert (entry["shape"], entry["enabled"]) == (built_particles[name]["shape"], built_particles[name]["enabled"])
        assert set(entry["parameters"]) == set(built_particles[name]["parameters"]) == set(schemas)
        for shape, block in entry["parameters"].items():
            assert list(block) == schemas[shape], (name, shape)  # the values in the order of the model


def test_a_new_user_folder_gets_the_template_and_an_existing_file_is_never_touched(tmp_path, monkeypatch) -> None:
    program = _program_folder(tmp_path / "program").parents[1]
    home = tmp_path / "home"

    imported = migrate_legacy_files(home, project_root=program)

    target = home / "model_parameters.json"
    assert [Path(item).name for item in imported] == ["model_parameters.json"]
    assert target.read_bytes() == TEMPLATE.read_bytes()

    mine = _template()
    mine["fitting"]["particles"]["particle_1"]["parameters"]["sphere"]["radius"] = 7.5
    mine["fitting"]["global_parameters"]["background"] = 0.25
    target.write_text(json.dumps(mine, indent=4), encoding="utf-8")
    before = target.read_bytes()

    assert migrate_legacy_files(home, project_root=program) == []  # a later start imports nothing
    monkeypatch.setenv("GIMAP_HOME", str(home))
    assert default_model_parameters_path() == target
    store = ManageFittingModelParameters(FittingModelParametersAdapter())
    assert store.get_particle_parameter("fitting", "particle_1", "sphere", "radius") == 7.5
    assert store.get_global_parameter("fitting", "background") == 0.25
    assert target.read_bytes() == before

    fresh = tmp_path / "fresh"  # the other way a new folder gets it: Fitting's own adapter
    monkeypatch.setenv("GIMAP_HOME", str(fresh))
    assert default_model_parameters_path().read_bytes() == TEMPLATE.read_bytes()


def test_a_read_only_program_folder_gives_a_new_user_a_file_fitting_can_save(tmp_path, monkeypatch) -> None:
    """Both copies take the content only. With the read-only flag of the program's file on it, every later
    save of Fitting failed without a word, and the model sent to the series set-up was read back unchanged."""
    template = _program_folder(tmp_path / "program")
    template.chmod(stat.S_IREAD)
    try:
        migrated = tmp_path / "migrated" / "model_parameters.json"
        assert migrate_legacy_files(migrated.parent, project_root=template.parents[1])
        monkeypatch.setattr(template_adapter, "TEMPLATE_PATH", template)
        monkeypatch.setenv("GIMAP_HOME", str(tmp_path / "adapter"))
        for target in (migrated, default_model_parameters_path()):
            assert target.read_bytes() == TEMPLATE.read_bytes()
            store = ManageFittingModelParameters(FittingModelParametersAdapter(target))
            assert store.set_particle_parameter("fitting", "particle_1", "sphere", "radius", 7.5)
            assert store.save_parameters(), target
            saved = json.loads(target.read_text(encoding="utf-8"))["fitting"]["particles"]["particle_1"]
            assert saved["parameters"]["sphere"]["radius"] == 7.5
    finally:
        template.chmod(stat.S_IREAD | stat.S_IWRITE)


OLDER_USER_FILE = {
    "fitting": {
        "global_parameters": {"sigma_res": 0.1, "k_value": 2.5, "background": 0.5},  # before nu_res and int_res
        "particles": {
            "particle_7": {"shape": "Sphere", "enabled": True, "parameters": {"sphere": {
                "intensity": 1.0, "radius": 10.0, "sigma_radius": 0.1, "diameter": 20.0, "sigma_diameter": 0.1,
                "background": 0.0}}},
            "particle_1": {"shape": "Cylinder", "enabled": True, "parameters": {
                "cylinder": {"intensity": 2.0, "radius": 8.0, "sigma_radius": 0.8, "height": 12.0,
                             "sigma_height": 1.2, "diameter": 40.0, "sigma_diameter": 8.0},
                "vertical cylinder": {"intensity": 1.0, "radius": 6.0, "sigma_radius": 0.25, "diameter": 25.0,
                                      "sigma_diameter": 5.0}}},
        },
    },
}
"""What a user of an earlier version can have (made-up numbers): fewer global values, a component the page
no longer has, a shape named with a space, a value the page never read."""


def _existing_user_files() -> dict[str, bytes]:
    older = json.dumps(OLDER_USER_FILE, indent=4).encode("utf-8")
    return {
        "an earlier version's": older,
        "with a byte-order mark": b"\xef\xbb\xbf" + older,
        "half a document": older[: len(older) // 2],
        "empty": b"",
        "an empty document": b"{}",
    }


@pytest.mark.parametrize("kind", sorted(_existing_user_files()))
def test_no_existing_user_file_is_rewritten_by_reading_it(kind: str, tmp_path, monkeypatch) -> None:
    """Also one the store cannot read: it then works from its own defaults in memory, and the file stays."""
    content = _existing_user_files()[kind]
    target = tmp_path / "home" / "model_parameters.json"
    target.parent.mkdir()
    target.write_bytes(content)
    template = _program_folder(tmp_path / "program")

    assert migrate_legacy_files(target.parent, project_root=template.parents[1]) == []
    monkeypatch.setattr(template_adapter, "TEMPLATE_PATH", template)
    monkeypatch.setenv("GIMAP_HOME", str(target.parent))
    assert default_model_parameters_path() == target
    store = ManageFittingModelParameters(FittingModelParametersAdapter())
    store.get_particle_shape("fitting", "particle_1")
    store.get_all_global_parameters("fitting")

    assert target.read_bytes() == content
    assert sorted(path.name for path in target.parent.iterdir()) == ["model_parameters.json"]


def test_a_start_and_a_close_leave_the_model_file_of_an_earlier_version_as_it_is(tmp_path, monkeypatch) -> None:
    from tests.test_fitting_presentation import _fitting_window

    target = tmp_path / "home" / "model_parameters.json"
    target.parent.mkdir()
    target.write_bytes(_existing_user_files()["an earlier version's"])
    monkeypatch.setenv("GIMAP_HOME", str(target.parent))
    app, window = _fitting_window()
    try:
        # the page people see starts from the code's model whatever the file holds
        assert window.components.fitting_workspace.fit_page.session.model == starting_model()
    finally:
        window.close()
        app.processEvents()
    assert target.read_bytes() == _existing_user_files()["an earlier version's"]


def test_fitting_opens_with_the_neutral_model_in_an_empty_user_folder(tmp_path, monkeypatch) -> None:
    from tests.test_fitting_presentation import _fitting_window

    home = tmp_path / "home"  # not there yet: a new user
    monkeypatch.setenv("GIMAP_HOME", str(home))
    app, window = _fitting_window()
    try:
        assert window.components.current_page_key() == "fitting"
        assert (home / "model_parameters.json").read_bytes() == TEMPLATE.read_bytes()

        page = window.components.fitting_workspace.fit_page
        assert page.session.model == starting_model() and not page.session.edited

        curve = evaluate(page.session.model, Q_NM)
        assert np.all(np.isfinite(curve)) and np.all(curve > 0)
        assert (home / "model_parameters.json").read_bytes() == TEMPLATE.read_bytes()  # opening changes nothing
    finally:
        window.close()
        app.processEvents()
