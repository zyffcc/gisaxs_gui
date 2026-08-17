"""Worker-local compatibility shims for trusted legacy Keras artifacts。"""

from __future__ import annotations


def install_legacy_keras_load_shims() -> None:
    """Install old Keras module aliases inside the disposable worker only."""
    try:
        import sys
        import types

        import keras
        from keras.src.layers.core.lambda_layer import Lambda
        from keras.src.layers.normalization.batch_normalization import BatchNormalization
        from keras.src.models.functional import Functional
        from keras.src.utils import python_utils
    except Exception:
        return

    engine_package = types.ModuleType("keras.src.engine")
    functional_module = types.ModuleType("keras.src.engine.functional")
    functional_module.Functional = Functional
    sys.modules.setdefault("keras.src.engine", engine_package)
    sys.modules.setdefault("keras.src.engine.functional", functional_module)

    if not getattr(Lambda, "_gimap_legacy_from_config", False):
        original_from_config = Lambda.from_config

        @classmethod
        def legacy_lambda_from_config(cls, config, custom_objects=None, safe_mode=None):
            if isinstance(config, dict):
                config = dict(config)
                for key in ("function_type", "module", "output_shape_type", "output_shape_module"):
                    config.pop(key, None)
                function = config.get("function")
                if isinstance(function, (list, tuple)) and function:
                    try:
                        defaults = function[1] if len(function) > 1 else None
                        closure = function[2] if len(function) > 2 else None
                        config["function"] = python_utils.func_load(
                            function[0], defaults=defaults, closure=closure
                        )
                    except Exception:
                        pass
                if callable(config.get("function")):
                    return cls(**config)
            try:
                return original_from_config(
                    config, custom_objects=custom_objects, safe_mode=safe_mode
                )
            except TypeError:
                return original_from_config(config)

        Lambda.from_config = legacy_lambda_from_config
        Lambda._gimap_legacy_from_config = True

    if not getattr(BatchNormalization, "_gimap_legacy_from_config", False):
        original_batch_normalization_from_config = BatchNormalization.from_config

        @classmethod
        def legacy_batch_normalization_from_config(cls, config):
            if isinstance(config, dict):
                config = dict(config)
                axis = config.get("axis")
                if isinstance(axis, list) and len(axis) == 1:
                    config["axis"] = axis[0]
            return original_batch_normalization_from_config(config)

        BatchNormalization.from_config = legacy_batch_normalization_from_config
        BatchNormalization._gimap_legacy_from_config = True

    if "keras.src.layers.core.tf_op_layer" not in sys.modules:
        operation_module = types.ModuleType("keras.src.layers.core.tf_op_layer")

        @keras.saving.register_keras_serializable(package="keras.src.layers.core.tf_op_layer")
        class SlicingOpLambda(keras.layers.Layer):
            def __init__(self, function=None, **kwargs):
                super().__init__(**kwargs)
                self.function = function

            def call(self, inputs, slice_spec=None, **kwargs):
                del kwargs
                if slice_spec is None:
                    return inputs
                slices = []
                for spec in slice_spec:
                    if isinstance(spec, dict):
                        slices.append(slice(spec.get("start"), spec.get("stop"), spec.get("step")))
                    else:
                        slices.append(spec)
                return inputs[tuple(slices)]

            def get_config(self):
                config = super().get_config()
                config["function"] = self.function
                return config

        operation_module.SlicingOpLambda = SlicingOpLambda
        sys.modules["keras.src.layers.core.tf_op_layer"] = operation_module
