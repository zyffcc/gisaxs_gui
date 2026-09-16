"""Optional curriculum weights balancing actual q-window component power."""
import numpy as np

from TrainSetBuild.physics_adapter import evaluate_clean

VERSION = "q-window-rms-v1"


def balance_component_weights(q, components):
    """Return normalized Int weights; do not alter geometry or caller records.

    Preserve the sampled relative contribution targets while compensating for
    the different absolute form-factor scales. This does not prove uniqueness.
    """
    if not components:
        raise ValueError("At least one component is required")
    global_unit = dict(BG=0., sigma_Res=.01, nu_Res=2., int_Res=0., k=1.)
    log_weights = []
    for component in components:
        curve = evaluate_clean(q, [dict(component, weight=1.)], global_unit)
        peak = float(np.max(curve))
        target = float(component['weight'])
        if not np.isfinite(curve).all() or peak <= 0 or not np.isfinite(target) or target <= 0:
            raise ValueError("Component power and target must be finite and positive")
        log_rms = np.log(peak) + .5 * np.log(np.mean((curve / peak)**2))
        log_weights.append(np.log(target) - log_rms)
    weights = np.exp(np.asarray(log_weights) - np.max(log_weights))
    weights = (weights / weights.sum()).astype(np.float32)
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("Balanced weights cannot be represented in float32")
    return weights
