"""Known Yoneda material through the calibrated measurement and command line."""

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.gimap.features.analyze.application.sample_roll import MATERIALS, critical_angle_deg, measure_frame_angles
from src.gimap.features.fitting.domain.substrates import Substrate
from tests.test_frame_incidence_measurement import _frame


@pytest.mark.parametrize("material", MATERIALS)
def test_shared_optics_give_the_fitting_critical_angle_at_each_wavelength(material):
    for wavelength_nm in (.075, .134, .200):
        assert critical_angle_deg(material, wavelength_nm) == Substrate(material).critical_angle_deg(wavelength_nm)
    with pytest.raises(ValueError, match="covers"):
        critical_angle_deg(material, .50)


def test_unsupported_wavelength_leaves_other_estimates_available():
    data, valid, g = _frame(spot=.41)
    g = replace(g, wavelength_angstrom=5., critical_angle_deg=.14)
    frame = SimpleNamespace(data=data, valid=valid, geometry=g, corrections=None, kind="gisaxs")
    _, result = measure_frame_angles(frame, None, roll_deg=2.2, yoneda_material="Si")
    assert result.method == "specular" and result.yoneda is None and result.critical_angle_deg is None
    assert result.material_wavelength_angstrom == 5. and result.yoneda_material == "Si"
    assert result.material_optics_error and "outside" in result.material_optics_error


def test_cli_named_material_in_public_frame_report(tmp_path):
    from tools.gimap_agent import main

    folder = Path(__file__).parent / "data" / "external" / "gisaxs_galaxi"
    frame = folder / "galaxi_data.tif"
    if not frame.is_file():
        pytest.skip("Public GALAXI data unavailable.")
    result = main(["auto", str(frame), "--calibration", str(folder / "galaxi_bornagain.poni"),
                   "--technique", "gisaxs", "--incidence-deg", ".463", "--yoneda-material", "Si",
                   "--no-saved-profiles", "--no-fit", "--out", str(tmp_path)])
    assert result in (0, 2)
    report = json.loads((tmp_path / frame.stem / "report.json").read_text(encoding="utf-8"))
    record = report["incidence_measurement"]
    assert record["yoneda_material"] == "Si" and record["material_wavelength_angstrom"] == pytest.approx(1.34)
    assert record["critical_angle_source"] == "run option" and "Henke" in record["material_optics_source"]
    assert record["critical_angle_deg"] == pytest.approx(critical_angle_deg("Si", .134), rel=.001)
    assert record["yoneda"] and record["decision"] == "agreement"


def test_cli_cannot_simultaneously_assign_a_material_and_numeric_angle():
    from tools.gimap_agent import parser

    with pytest.raises(SystemExit):
        parser().parse_args(["auto", "sample.tif", "--yoneda-critical-deg", ".25", "--yoneda-material", "glass"])
