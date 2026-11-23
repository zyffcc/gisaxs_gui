"""Film pole angles and negative controls on independently simulated detector images."""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.ndimage import gaussian_filter

from src.gimap.features.analyze.application.crystal_orientation import _profile
from src.gimap.features.analyze.domain.corrections import Corrections
from src.gimap.features.analyze.domain.crystal.texture import TextureRing, fit_crystal_orientation
from src.gimap.features.analyze.domain.giwaxs import giwaxs_maps
from src.gimap.features.analyze.domain.orientation_reduction import BACKGROUND_PEAK
from src.gimap.shared.geometry import DetectorGeometry
from tools.benchmarks.crystal_orientation import expected_rings


def _project_arc(geometry, optics, q, chi):
    """Forward projection from the real film q vector, conserving tangential wavevectors."""
    k = 2*np.pi / geometry.wavelength_angstrom
    ai = np.radians(geometry.incidence_deg)
    medium = -2*optics["delta"] + 2j*optics["beta"]
    qpar_k = q*np.sin(np.radians(chi)) / k
    normal = q*np.cos(np.radians(chi)) / k - np.sqrt(np.sin(ai)**2 + medium).real
    # For kz/k = u+iv, uv=beta. Recover the external sine from its square.
    sine = np.sqrt(normal**2 - (optics["beta"]/normal)**2 + 2*optics["delta"])
    along = (1 - sine**2 + np.cos(ai)**2 - qpar_k**2) / (2*np.cos(ai))
    across = np.sqrt(1 - sine**2 - along**2)
    forward = along*np.cos(ai) - sine*np.sin(ai)
    up = along*np.sin(ai) + sine*np.cos(ai)
    dx = geometry.distance_m/geometry.pixel_size_x_m * across/forward
    dy = geometry.distance_m/geometry.pixel_size_y_m * up/forward
    return geometry.beam_center_x_px + dx, geometry.beam_center_y_px - dy


def test_projected_horizon_arc_moves_into_the_correct_internal_pole_bin():
    geometry = DetectorGeometry(2.5e-5, 2.5e-5, .12, 15., 160., 1., .16)
    optics = {"available": True, "delta": 4.e-6, "beta": 1.e-7, "model": "frame corrected with refracted waves"}
    q, pole = .91, 88.50
    chi = np.linspace(82., 89.2, 901)
    x, y = _project_arc(geometry, optics, q, chi)
    shape = (164, 745)
    amplitude = 1. + 100.*np.exp(-.5*((chi-pole)/.13)**2)
    yy, xx = np.arange(shape[0]+1), np.arange(shape[1]+1)
    image = 10. + gaussian_filter(np.histogram2d(y, x, (yy, xx), weights=amplitude)[0], .55)
    maps = giwaxs_maps(shape, geometry)
    analysis = SimpleNamespace(geometry=geometry, corrections=Corrections(orientation_polarization=False,
                                                                          orientation_min_exit_deg=0.))
    pixels = image, np.ones(shape, bool), True
    window = (.9, .922)
    inside, settings, _, _ = _profile(maps, pixels, analysis, window, q, optics)
    outside, *_ = _profile(maps, pixels, analysis, window, q, {"available": False})
    def peak(profile):
        density = profile.value / np.sin(np.radians(profile.chi))
        return profile.chi[np.nanargmax(density)]

    assert settings["pole_angle_coordinate"] == "inside film"
    assert peak(inside) == pole
    assert pole - peak(outside) >= 1.


@pytest.mark.parametrize("fraction, seed", [(0., 42071), (0., 42072), (.001, 42071), (.001, 42072)])
def test_noisy_weak_and_isotropic_frames_do_not_establish_an_axis(fraction, seed):
    phase, rings = expected_rings("PbI2")
    q, low, high = rings[0][:3]
    shape = (185, 375)
    geometry = DetectorGeometry(.0001, .0001, .12, 187.5, 182.5, 1., .32)
    delta, beta, thickness_nm = 3.e-6, 2.e-7, 700.
    optics = {"available": True, "delta": delta, "beta": beta, "thickness_nm": thickness_nm}
    ai = np.radians(geometry.incidence_deg)
    row, col = np.indices(shape)
    right = (col + .5 - geometry.beam_center_x_px) * geometry.pixel_size_x_m / geometry.distance_m
    up = (geometry.beam_center_y_px - row - .5) * geometry.pixel_size_y_m / geometry.distance_m
    length = np.sqrt(1. + right**2 + up**2)
    s = (up*np.cos(ai) - np.sin(ai)) / length
    along = (np.cos(ai) + up*np.sin(ai)) / length
    k = 2*np.pi
    parallel = k * np.hypot(along - np.cos(ai), right/length)
    kz_i = k*np.sqrt(np.sin(ai)**2 - 2*delta + 2j*beta)
    kz_f = k*np.sqrt(s**2 - 2*delta + 2j*beta)
    qz = (kz_i + kz_f).real
    internal_q = np.hypot(parallel, qz)
    chi = np.arctan2(parallel, qz)
    # Normalise a weak basal Gaussian on the hemisphere; incident and exit path lengths
    # enter the independent depth integral in angstroms.
    spread = np.radians(24.) / np.sqrt(8.*np.log(2.))
    norm = quad(lambda t: np.sin(t)*np.exp(-.5*(t/spread)**2), 0., np.pi)[0]
    texture = 1. - fraction + fraction*np.exp(-.5*(chi/spread)**2)/norm
    s = np.maximum(s, 0.)
    t_f = 2*k*s/(k*s + kz_f)
    t_spec = 2*k*np.sin(ai)/(k*np.sin(ai) + kz_i)
    rate = 2*(kz_i + kz_f).imag
    depth = -np.expm1(-rate * (10*thickness_nm))/rate
    specular = -np.expm1(-4*kz_i.imag * (10*thickness_nm))/(4*kz_i.imag)
    wave = abs(t_f/t_spec)**2 * depth/specular
    background = 32. + 3.*(internal_q-q)
    signal = 38. * texture * wave * np.exp(-.5*((internal_q - q)/.006)**2)
    image = np.random.default_rng(seed).poisson(background + signal)
    # Use the same radial coordinate as the simulated ring, retaining the complex film index
    # in the independently calculated truth. The map's real critical angle excludes subcritical exits.
    geometry = geometry.with_critical_angle(np.degrees(np.arcsin(np.sqrt(2*delta))))
    maps = giwaxs_maps(shape, geometry)
    analysis = SimpleNamespace(geometry=geometry, corrections=Corrections(orientation_polarization=False))
    profile, region, factor, _ = _profile(maps, (image, np.ones(shape, bool), True),
                                           analysis, (low, high), q, optics)
    assert region["background_source"] == BACKGROUND_PEAK
    assert region["background"] == pytest.approx(32., abs=1.)
    assert np.ptp(factor[profile.measured]) > .4
    fit = fit_crystal_orientation(phase, [TextureRing(0, q, low, high, profile, transmission=factor)])
    assert fit["ok"] and not fit["determined"]
    assert fit["parameter_status"] == "conditional on the preferred candidate"
