import sys
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from TrainSetBuild.contribution_weights import balance_component_weights


def test_compensates_power_without_mutating_components():
    components = [dict(weight=.25, scale=1e-10), dict(weight=.75, scale=1e10)]
    def forward(q, comps, globals_):
        assert comps[0]['weight'] == 1.
        return np.array([1.,2.,3.]) * comps[0]['scale']
    with patch('TrainSetBuild.contribution_weights.evaluate_clean', forward):
        weights = balance_component_weights(np.arange(3), components)
    contributions = weights * np.array([1e-10, 1e10])
    np.testing.assert_allclose(contributions / contributions.sum(), [.25,.75], rtol=1e-6)
    assert components[0]['weight'] == .25
    assert weights.sum() == 1.


@pytest.mark.parametrize('bad', [np.nan, np.inf, 0., -1.])
def test_rejects_invalid_target(bad):
    with patch('TrainSetBuild.contribution_weights.evaluate_clean', return_value=np.ones(3)):
        with pytest.raises(ValueError):
            balance_component_weights(np.arange(3), [dict(weight=bad)])
