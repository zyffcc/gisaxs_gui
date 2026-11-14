'Independent q-axis-retention candidate for the four-scenario joint model.'
from torch import nn

from .scenario_posterior import ScenarioCandidate


BASELINE_ARCHITECTURE = 'baseline_v1'
Q_PRESERVE_ARCHITECTURE = 'q_preserve70_v1'
GAIN_ANCHOR_ARCHITECTURE = 'gain_anchor_v1'
ARCHITECTURES = (BASELINE_ARCHITECTURE, Q_PRESERVE_ARCHITECTURE, GAIN_ANCHOR_ARCHITECTURE)


def architecture_spec(architecture):
    'Plain declaration metadata; unknown architecture names are rejected.'
    if architecture not in ARCHITECTURES:
        raise ValueError('Unknown four-scenario encoder architecture: '+str(architecture))
    if architecture == GAIN_ANCHOR_ARCHITECTURE:
        from .scenario_gain_anchor import architecture_spec as gain_spec
        return gain_spec(architecture)
    preserved = architecture == Q_PRESERVE_ARCHITECTURE
    return dict(name=architecture, input_channels=4, input_AF_q_shape=[32, 70],
        first_conv_stride=[2, 1], second_conv_stride=[2, 1 if preserved else 2],
        pooled_AF_q_shape=[4, 70 if preserved else 10],
        image_embedding_coordinates=48*4*(70 if preserved else 10),
        final_embedding='same declared hidden dimension as the original joint model',
        joint_likelihood_and_sampling='inherited without modification',
        parameter_intervals_calibrated=False, structural_release=False)


class QPreserveScenarioCandidate(ScenarioCandidate):
    'Keep q70 in the image encoder; all joint operators remain inherited.'
    def __init__(self, mode, continuous_dim=None, coarse_classes=None,
                 template_classes=1, *, components=4, hidden=128, catalog=None,
                 bank_scenario_prior=None):
        if mode not in ('qimage', 'hybrid', 'rowimage'):
            raise ValueError('q_preserve70_v1 requires an existing image input mode')
        super().__init__(mode, continuous_dim, coarse_classes, template_classes,
            components=components, hidden=hidden, catalog=catalog,
            bank_scenario_prior=bank_scenario_prior)


        self.conv[2].stride = (2, 1)
        self.conv[4] = nn.AdaptiveAvgPool2d((4, 70))
        original = self.encoder[0]
        metadata_size = original.in_features - 48*4*10
        self.encoder[0] = nn.Linear(metadata_size + 48*4*70,
                                    original.out_features, bias=original.bias is not None)
        self.architecture = Q_PRESERVE_ARCHITECTURE


def make_candidate(mode, continuous_dim=None, coarse_classes=None,
                   template_classes=1, *, architecture=BASELINE_ARCHITECTURE,
                   components=4, hidden=128, catalog=None,
                   bank_scenario_prior=None):
    'Strict shared factory for a trainer, warm loader and runtime.'
    architecture_spec(architecture)
    if architecture == GAIN_ANCHOR_ARCHITECTURE:
        from .scenario_gain_anchor import GainAnchoredCandidate
        candidate = GainAnchoredCandidate
    else:
        candidate = (ScenarioCandidate if architecture == BASELINE_ARCHITECTURE
                     else QPreserveScenarioCandidate)
    model = candidate(mode, continuous_dim, coarse_classes, template_classes,
        components=components, hidden=hidden, catalog=catalog,
        bank_scenario_prior=bank_scenario_prior)
    model.architecture = architecture
    return model
