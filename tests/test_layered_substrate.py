"""Independent checks of the complex surface amplitude and of the batched four-path sphere integral."""

import math

import numpy as np
import pytest

from src.gimap.features.fitting.domain.cut_model import size_averages
from src.gimap.features.fitting.domain.cut_model import shapes
from src.gimap.features.fitting.domain.substrates import Layer, Substrate


def _interface_matching(geometry, angle):
    """Solve all interface boundary conditions at once for up/down waves in local layer coordinates."""
    layers = geometry.layers
    media = [(0., 0.), *((s.delta, s.beta) for s in layers), (geometry.delta, geometry.beta)]
    wave = geometry.k * np.sqrt(np.sin(np.deg2rad(angle))**2
                                + np.array([-2*d + 2j*b for d, b in media]))
    system = np.zeros((2*len(media), 2*len(media)), complex)
    rhs = np.zeros(system.shape[0], complex)
    system[0, 0] = rhs[0] = 1.  # incident amplitude in air
    system[-1, -1] = 1.  # no upgoing wave in the backing
    for j, thickness in enumerate([0., *(layer.thickness_nm for layer in layers)]):
        down, up = np.exp(1j * wave[j] * thickness * np.array([1., -1.]))
        columns = slice(2*j, 2*j + 4)
        system[2*j + 1, columns] = down, up, -1., -1.
        system[2*j + 2, columns] = wave[j]*down, -wave[j]*up, -wave[j+1], wave[j+1]
    return np.linalg.solve(system, rhs)[1]


LAYERS = (Layer(100., 1.67e-6, 2e-9), Layer(7., 2.9e-6, 5e-8))


@pytest.mark.parametrize("angle", [0.045, 0.105, 0.15, 0.21, 0.4, 1.0])
@pytest.mark.parametrize("layers", [(), LAYERS[:1], LAYERS])
def test_complex_reflection_matches_continuous_field_and_derivative(angle, layers):
    substrate = Substrate("Si", layers=layers)
    assert Substrate.from_dict(substrate.to_dict()) == substrate
    geometry = substrate.geometry(0.1045, 0.4, angle)
    expected = _interface_matching(geometry, angle)
    assert geometry.reflection(angle) == pytest.approx(expected, abs=5e-14)
    if not layers:
        assert geometry.reflection(angle) == geometry.fresnel(angle)
    incident, outgoing = _interface_matching(geometry, 0.4), expected
    assert [c for c, _ in geometry.sphere_terms()] == pytest.approx([1, incident, outgoing, incident*outgoing])


def test_layer_at_zero_normal_wavevector_and_a_thick_absorbing_film():
    angle = 0.2
    delta = 0.5 * math.sin(math.radians(angle))**2
    def reflected(d, b, thickness):
        return Substrate("Au", layers=(Layer(thickness, d, b),)).geometry(0.1, 0.3, angle).reflection(angle)

    assert reflected(delta, 0., 100.) == pytest.approx(reflected(delta, 1e-16, 100.), abs=1e-8)
    expected = Substrate(delta=delta, beta=2e-7).geometry(0.1, 0.3, angle).fresnel(angle)
    assert reflected(delta, 2e-7, 1e7) == pytest.approx(expected, abs=1e-14)


def test_four_path_batched_quadrature_matches_direct_scalar_evaluations():
    q = np.linspace(0.03, 2.0, 100)
    radius = np.linspace(3., 6., 48)
    spread = np.linspace(0.1, 0.24, len(radius))
    geometry = Substrate("Si", layers=LAYERS[:1]).geometry(0.1045, 0.4, 0.15)
    form = size_averages("sphere", q, radius=radius, radius_spread=spread, geometry=geometry, panels=16)[0]
    direct = np.stack([size_averages("sphere", q, radius=r, radius_spread=s, geometry=geometry, panels=16)[0][0]
                       for r, s in zip(radius, spread)])
    np.testing.assert_allclose(form, direct, rtol=2e-6, atol=1e-14)


@pytest.mark.parametrize("incidence,exit_angle", [(0.4, 0.15), (0.12, 0.12)])
def test_resolved_transition_and_envelope_agree_with_explicit_four_path_sum(monkeypatch, incidence, exit_angle):
    geometry = Substrate("Si", layers=LAYERS[:1]).geometry(0.1045, incidence, exit_angle)
    q = np.geomspace(0.005, 3., 401)
    def explicit(q, radii, weights, terms, coherent, envelope):
        if envelope:
            p = shapes._sphere_envelope(q, radii, weights, terms)  # all ten Hankel pairs separately
            return p, np.zeros_like(p) if coherent else None
        r = radii[:, :, None]
        field = sum(c * shapes.sphere_amplitude(r * np.hypot(q, z)) * np.exp(-1j*z*r) for c,z in terms)
        v2 = weights * radii**6
        p = np.einsum("bn,bnq->bq", v2, abs(field)**2) / v2.sum(axis=1, keepdims=True)
        mean = np.einsum("bn,bnq->bq", weights*radii**3, field)
        return p, abs(mean)**2 / v2.sum(axis=1, keepdims=True) if coherent else None

    kwargs = dict(radius=[4., 27., 66.], radius_spread=[0.02, 0.22, 0.43], geometry=geometry)
    fast_p = size_averages("sphere", q, **kwargs)[0]
    full_p, full_c = size_averages("sphere", q, coherent=True, **kwargs)
    monkeypatch.setattr(shapes, "sphere", explicit)
    ref_p, ref_c = size_averages("sphere", q, coherent=True, **kwargs)
    np.testing.assert_allclose(fast_p, ref_p, rtol=2e-6, atol=1e-16)
    np.testing.assert_allclose(full_p, ref_p, rtol=2e-12, atol=1e-16)
    np.testing.assert_allclose(full_c, ref_c, rtol=2e-12, atol=1e-16)
