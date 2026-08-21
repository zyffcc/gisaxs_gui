import numpy as np
import pytest

from src.gimap.features.fitting.domain import prepare_signed_q_curve


def test_separate_signed_q_preserves_negative_coordinates():
    prepared = prepare_signed_q_curve(
        [-3.0, -1.0, 1.0, 3.0],
        [30.0, 10.0, 12.0, 32.0],
        branch="both",
        combination="separate",
    )

    np.testing.assert_allclose(prepared.q, [-3.0, -1.0, 1.0, 3.0])
    np.testing.assert_allclose(prepared.intensity, [30.0, 10.0, 12.0, 32.0])


def test_fold_maps_both_branches_without_silently_averaging_them():
    prepared = prepare_signed_q_curve(
        [-2.0, -1.0, 1.0, 2.0],
        [20.0, 10.0, 12.0, 22.0],
        combination="fold",
    )

    np.testing.assert_allclose(prepared.q, [1.0, 1.0, 2.0, 2.0])
    np.testing.assert_allclose(prepared.intensity, [10.0, 12.0, 20.0, 22.0])
    np.testing.assert_array_equal(prepared.source_sign, [-1, 1, -1, 1])


def test_average_uses_only_the_shared_absolute_q_domain():
    prepared = prepare_signed_q_curve(
        [-3.0, -2.0, -1.0, 1.0, 2.0, 4.0],
        [30.0, 20.0, 10.0, 14.0, 24.0, 44.0],
        combination="average",
    )

    np.testing.assert_allclose(prepared.q, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(prepared.intensity, [12.0, 22.0, 32.0])
    np.testing.assert_array_equal(prepared.source_sign, [0, 0, 0])


def test_average_rejects_single_branch_selection():
    with pytest.raises(ValueError, match="Both branches"):
        prepare_signed_q_curve(
            [-2.0, -1.0, 1.0, 2.0],
            [20.0, 10.0, 12.0, 22.0],
            branch="positive",
            combination="average",
        )
