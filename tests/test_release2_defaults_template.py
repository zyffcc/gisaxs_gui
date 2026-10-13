"""``config/model_parameters.json``: a new user's first model of Fitting holds neutral defaults.

The file is the template of the manual parameter store of Fitting: it is copied into the user data
folder when that folder has no ``model_parameters.json`` (``default_model_parameters_path`` and the
migration of the user store) and never afterwards. It holds the starting model of Fitting
(``starting_model()``: the defaults of the domain's ``new_component`` and ``FitModel``) written in
the units of the store — sizes in nm, the spreads of spheres and random cylinders and σD in nm, the
spread of a vertical cylinder as σR/R, the resolution width in nm⁻¹ — so the page people see and the
store behind it start from the same model, inside the ranges of the domain and of the physical
constraints. No value in it comes from a fit.

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
from src.gimap.features.fitting.application.single_fit import (
    FAMILIES,
    FitModel,
    evaluate,
    from_manual,
    new_component,
    to_manual,
)
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
from src.gimap.features.fitting.presentation.scientific_commands import COMPONENT_PARAMETER_SCHEMAS
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
    assert from_manual(CandidateParameterMapping(components, fitting["global_parameters"])) == starting_model()


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_every_component_holds_the_defaults_of_a_new_component_of_each_family(family: str) -> None:
    """Choosing another type on a component shows the domain's defaults of that family, valid as they are."""
    fitting = _template()["fitting"]
    fresh = FitModel((new_component(family),))
    expected = to_manual(fresh).components[0]
    for name, entry in fitting["particles"].items():
        assert entry["parameters"][family] == expected.parameters, name

    block = fitting["particles"]["particle_1"]["parameters"][family]
    mapping = CandidateParameterMapping(
        (CandidateComponentParameters(expected.shape, 1.0, block),), fitting["global_parameters"]
    )
    assert from_manual(mapping) == fresh  # inside the ranges of the domain
    assert ConstraintSet.defaults().validate_components([{"type": family, "params": block}]) == []


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_the_store_and_the_domain_draw_the_same_finite_curve(family: str) -> None:
    fitting = _template()["fitting"]
    block = fitting["particles"]["particle_1"]["parameters"][family]
    values = []
    for name in params_template([family]):
        key = STORE_NAMES[name.rstrip("1")]
        values.append(block[key] if key in block else fitting["global_parameters"][key])

    curve = np.asarray(make_mixed_model([family])(Q_NM, *values), dtype=float)

    assert np.all(np.isfinite(curve)) and np.all(curve > 0)
    np.testing.assert_allclose(curve, evaluate(FitModel((new_component(family),)), Q_NM), rtol=1e-9)


def test_every_number_is_a_round_default_and_not_the_result_of_a_fit() -> None:
    values = list(_numbers(_template()["fitting"]))
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
    schemas = {_shape_key(shape): [row[0] for row in schema] for shape, schema in COMPONENT_PARAMETER_SCHEMAS.items()}
    for name, entry in particles.items():
        assert set(entry) == {"shape", "enabled", "parameters"}
        assert (entry["shape"], entry["enabled"]) == (built_particles[name]["shape"], built_particles[name]["enabled"])
        assert set(entry["parameters"]) == set(built_particles[name]["parameters"]) == set(schemas)
        for shape, block in entry["parameters"].items():
            assert list(block) == schemas[shape], (name, shape)  # the values the page's fields read


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
        binding = window.runtime.fitting  # the former page shows the user's model, not the template's
        assert binding.get_particles_status() == {1: "Cylinder", 2: "None", 3: "None"}
        shapes, widget_ids = binding._collect_active_particles()
        values = binding._get_current_manual_param_values(shapes, widget_ids)
        own = OLDER_USER_FILE["fitting"]["particles"]["particle_1"]["parameters"]["cylinder"]
        assert values[:7] == [own[row[0]] for row in COMPONENT_PARAMETER_SCHEMAS["Cylinder"]]
        assert (values[7], values[11]) == (0.5, 2.5)  # its background and factor k
        assert np.all(np.isfinite(make_mixed_model(shapes)(Q_NM, *values)))
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

        binding = window.runtime.fitting  # the former page, fed by the file: the same model
        assert binding.get_particles_status() == {1: "Sphere", 2: "None", 3: "None"}
        shapes, widget_ids = binding._collect_active_particles()
        values = binding._get_current_manual_param_values(shapes, widget_ids)
        expected = to_manual(page.session.model)
        component = expected.components[0].parameters
        assert shapes == ["sphere"]
        assert values == [
            *(component[row[0]] for row in COMPONENT_PARAMETER_SCHEMAS["Sphere"]),
            *(expected.global_parameters[STORE_NAMES[name]] for name in params_template(shapes)[-5:]),
        ]

        curve = np.asarray(make_mixed_model(shapes)(Q_NM, *values), dtype=float)
        assert np.all(np.isfinite(curve)) and np.all(curve > 0)
        np.testing.assert_allclose(curve, evaluate(page.session.model, Q_NM), rtol=1e-9)
        assert (home / "model_parameters.json").read_bytes() == TEMPLATE.read_bytes()  # opening changes nothing
    finally:
        window.close()
        app.processEvents()
