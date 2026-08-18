"""Adapter exposing the feature-owned scattering model through application ports."""

from __future__ import annotations

import numpy as np

from ...domain.scattering_model import (
    make_mixed_model,
    mixed_model_components,
    params_template,
)


class MixedScatteringModelAdapter:
    def parameter_names(self, shapes: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(params_template(list(shapes)))

    def evaluate(self, shapes, q_model, parameters):
        model = make_mixed_model(list(shapes))
        return np.asarray(
            model(np.asarray(q_model, dtype=float), *parameters), dtype=float
        )

    def components(self, shapes, q_model, parameters):
        return mixed_model_components(
            list(shapes),
            np.asarray(q_model, dtype=float),
            list(parameters),
        )

    def build_function(self, shapes):
        return make_mixed_model(list(shapes))
