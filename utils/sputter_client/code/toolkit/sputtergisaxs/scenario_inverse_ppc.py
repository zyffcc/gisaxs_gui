'Explicit native-PPC adapters for the independent inverse experiments.'
from .scenario_mainline_ppc import MainlinePPCRuntime
from .scenario_mainline_posterior import (
    ARCHITECTURE as CAPACITY_ARCHITECTURE,
    TRAINING_SCHEMA as CAPACITY_SCHEMA)
from .scenario_mainline_interaction_posterior import (
    InteractionRuntime, ARCHITECTURE as INTERACTION_ARCHITECTURE,
    TRAINING_SCHEMA as INTERACTION_SCHEMA)

TRAIN_SEED = 682120261013
VARIANTS = ('capacity', 'interaction')
ARCHITECTURES = dict(capacity=CAPACITY_ARCHITECTURE, interaction=INTERACTION_ARCHITECTURE)
SCHEMAS = dict(capacity=CAPACITY_SCHEMA, interaction=INTERACTION_SCHEMA)


class InteractionPPCRuntime(InteractionRuntime, MainlinePPCRuntime):
    'New strict loader and the original native posterior-PPC implementation.'
    pass


def load_runtime(variant, checkpoint, *, device='cpu', package_root=None):
    if variant not in VARIANTS:
        raise ValueError('Explicit capacity or interaction experiment required')
    factory = MainlinePPCRuntime if variant == 'capacity' else InteractionPPCRuntime
    runtime = factory(checkpoint, device=device, package_root=package_root)
    declaration = runtime.declaration
    if (runtime.mode != 'qimage' or declaration.get('variant') != variant
            or declaration.get('architecture') != ARCHITECTURES[variant]
            or declaration.get('schema') != SCHEMAS[variant]
            or declaration.get('seed') != TRAIN_SEED
            or declaration.get('hidden') != 256 or declaration.get('components') != 8
            or runtime.receipt['training_terminal_status'] not in
                ('completed_CAL_early_stop', 'completed_epoch_limit')):
        raise ValueError('Completed independently declared qimage256/8 experiment weights required')
    return runtime
