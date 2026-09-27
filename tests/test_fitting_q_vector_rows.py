"""Physical row alignment across all detector q components."""
import numpy as np

from src.gimap.features.fitting.domain.q_space_geometry import create_detector_from_image_and_params


def _detector():
    return create_detector_from_image_and_params(
        image_shape=(17, 23), pixel_size_x=172., pixel_size_y=172.,
        beam_center_x=7.3, beam_center_y=4.1, distance=1730.,
        theta_in_deg=.463, wavelength=.134,
    )


def test_all_components_at_one_pixel_describe_an_elastic_outgoing_ray():
    detector = _detector()
    qx, qy, qz, _ = detector.calculate_q_vectors()
    k, alpha = detector.k0, detector.theta_in
    outgoing = np.sqrt((qx+k*np.cos(alpha))**2 + qy**2 + (qz-k*np.sin(alpha))**2)
    np.testing.assert_allclose(outgoing, k, rtol=0, atol=2e-13)


def test_qy_and_qr_follow_top_to_bottom_analysis_rows():
    detector = _detector()
    qx, qy, qz, qr = detector.calculate_q_vectors()
    # Independent pixel locations under the retained endpoint-grid convention.
    for row, col in ((0, 1), (3, 21), (16, 9)):
        x = col*(23*.172)/22 - 7.3*.172
        y = (16-row)*(17*.172)/16 - 4.1*.172 - 1730*np.tan(detector.theta_in)
        theta, psi = np.arctan2(y,1730.), np.arctan2(x,1730.)
        expected_x = detector.k0*(np.cos(theta)*np.cos(psi)-np.cos(detector.theta_in))
        expected_y = detector.k0*np.cos(theta)*np.sin(psi)
        expected_z = detector.k0*(np.sin(theta)+np.sin(detector.theta_in))
        np.testing.assert_allclose([qx[row,col], qy[row,col], qz[row,col], qr[row,col]],
                                   [expected_x,expected_y,expected_z,
                                    np.copysign(np.hypot(expected_x,expected_y),expected_y)],
                                   rtol=0,atol=1e-13)


def test_cached_grids_and_all_getters_preserve_row_alignment():
    detector = _detector()
    first = detector.calculate_q_vectors()
    second = detector.calculate_q_vectors()
    assert all(a is b for a,b in zip(first,second))
    for actual,expected in zip((detector.get_qx(),detector.get_qy(),detector.get_qz(),detector.get_qr()),first):
        np.testing.assert_array_equal(actual,expected)
    for actual,expected in zip(detector.get_q_coordinate_meshgrids(),first[1:]):
        np.testing.assert_array_equal(actual,expected)
    np.testing.assert_allclose(detector.get_q(),np.sqrt(sum(a*a for a in first[:3])),rtol=0,atol=0)
    fresh = detector.calculate_q_vectors(force_recalculate=True)
    assert all(a is not b for a,b in zip(first,fresh))
    for a,b in zip(first,fresh):
        np.testing.assert_array_equal(a,b)
