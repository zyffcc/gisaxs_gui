import numpy as np
import pytest

from tools.benchmarks import yoneda_truth as yt


def test_broadened_central_peak_conserves_area_and_tail_on_refinement():
    # With zero transfer and no interference, the particle peak is exactly at q=0.
    case = yt.Case().with_(f_scattering="born_qy", f_D=0, i_sigma_g=0.005,
                           i_eta=0.03, i_gamma=0.002, i_central=0.7, i_peak_counts=0)
    q = np.arange(161) * 0.001
    reference = yt.particle_intensity(np.array([0.0]), case)[0]
    profiles = []
    for refinement in (4, 8, 16):
        parts = yt.measured_curve(q, case, oversample=refinement)["parts"]
        density = parts["central"] / reference
        # The symmetric peak has half its area at nonnegative q.
        area = 2 * np.sum((density[1:] + density[:-1]) * np.diff(q) / 2)
        assert area == pytest.approx(case.instrument.central, rel=2e-5)
        profiles.append(density[:40])
    np.testing.assert_allclose(profiles[0], profiles[2], rtol=2e-5)
    np.testing.assert_allclose(profiles[1], profiles[2], rtol=2e-5)


def test_archived_beam_tail_replays_the_old_quadrature_factor():
    q = np.linspace(0.02, 0.3, 57)
    case = yt.Case().with_(i_sigma_g=0.004, i_central=0.7, i_peak_counts=0)
    actual = yt.measured_curve(q, case, oversample=8)["parts"]["central"]
    archived = yt.measured_curve(q, case, oversample=8, legacy_beam_tail=True)["parts"]["central"]
    np.testing.assert_allclose(archived, actual * (q[1] - q[0]) / 8, rtol=1e-14)
