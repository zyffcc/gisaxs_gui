"""现有 `utils.fitting` scientific model 的兼容 adapter。"""

from __future__ import annotations

import numpy as np


class LegacyMixedModelAdapter:
    def parameter_names(self, shapes: tuple[str, ...]) -> tuple[str, ...]:
        from utils.fitting import params_template

        return tuple(params_template(list(shapes)))

    def evaluate(self, shapes, q_model, parameters):
        from utils.fitting import make_mixed_model

        model = make_mixed_model(list(shapes))
        return np.asarray(model(np.asarray(q_model, dtype=float), *parameters), dtype=float)
