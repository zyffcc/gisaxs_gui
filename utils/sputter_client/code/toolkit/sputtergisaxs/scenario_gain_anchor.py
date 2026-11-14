'Observed-gain anchor inside the complete four-scenario joint posterior.'
import torch

from .scenario_chart import COORDINATES, DIM
from .scenario_posterior import ScenarioCandidate


GAIN_ANCHOR_ARCHITECTURE = 'gain_anchor_v1'
_VECTOR_DIM = dict(features=54, hybrid=54, qimage=14, rowimage=14)
_ANCHOR_INDEX = dict(features=43, hybrid=43, qimage=3, rowimage=3)
_BASE_DIAGONAL_FLOOR = .02
_GAIN_DIAGONAL_FLOOR = .001


def architecture_spec(architecture=GAIN_ANCHOR_ARCHITECTURE):
    'Declaration metadata for this explicit architecture; no release claim.'
    if architecture != GAIN_ANCHOR_ARCHITECTURE:
        raise ValueError('Unknown gain-anchor architecture: '+str(architecture))
    return dict(name=architecture, base_architecture='baseline_v1',
        input_preparation='unchanged v2_input.prepare',
        anchor_input='observed mandatory[3] log q80 rate, not latent truth',
        vector_dimensions=dict(_VECTOR_DIM), anchor_vector_indices=dict(_ANCHOR_INDEX),
        joint_coordinate_dimension=DIM, shifted_coordinate=COORDINATES[-1],
        encoder='unchanged baseline; append the observed anchor to its hidden embedding',
        discrete_heads='unchanged, consume only the original hidden embedding',
        conditional_mean='original residual mean plus observed anchor only in log_target_scale',
        conditional_covariance='complete joint Cholesky; only last diagonal floor changes',
        original_diagonal_floor=_BASE_DIAGONAL_FLOOR,
        gain_diagonal_floor=_GAIN_DIAGONAL_FLOOR,
        additional_trainable_parameters=0, translation_jacobian=1,
        posterior_sampling='inherited complete original joint sampling',
        physical_chart_and_targets='unchanged absolute coordinates',
        parameter_intervals_calibrated=False, structural_release=False)


class GainAnchoredCandidate(ScenarioCandidate):
    'Learn joint gain residuals without losing the known input gain signal.'
    def __init__(self, mode, continuous_dim=None, coarse_classes=None,
                 template_classes=1, *, components=4, hidden=128, catalog=None,
                 bank_scenario_prior=None):
        dim = DIM if continuous_dim is None else continuous_dim
        if dim != DIM or len(COORDINATES) != DIM or COORDINATES[-1] != 'log_target_scale':
            raise ValueError('gain_anchor_v1 requires the complete chart with log_target_scale last')
        if mode not in _VECTOR_DIM:
            raise ValueError('gain_anchor_v1 requires an existing four-scenario input mode')
        super().__init__(mode, dim, coarse_classes, template_classes,
            components=components, hidden=hidden, catalog=catalog,
            bank_scenario_prior=bank_scenario_prior)
        self._gain_anchor_hidden = self.encoder[2].out_features
        self.architecture = GAIN_ANCHOR_ARCHITECTURE
        self.validated = False
        self.structural_release = False

    def embedding(self, encoded):
        vector = encoded['vector']
        if not isinstance(vector, torch.Tensor) or vector.ndim != 2 or vector.shape[1] != _VECTOR_DIM[self.mode]:
            raise ValueError('Gain anchor requires the original prepared vector dimensions')
        index = _ANCHOR_INDEX[self.mode]
        anchor = vector[:, index:index+1]
        if not torch.isfinite(anchor).all():
            raise ValueError('Observed log-gain anchor must be finite')
        base = super().embedding(encoded)
        if base.shape != (len(vector), self._gain_anchor_hidden):
            raise ValueError('Baseline encoder hidden dimensions differ')
        return torch.cat([base, anchor], dim=-1)

    def _split_embedding(self, emb):
        if not isinstance(emb, torch.Tensor) or emb.ndim != 2 or emb.shape[1] != self._gain_anchor_hidden+1:
            raise ValueError('Gain-anchored head requires the complete H+1 embedding')
        return emb[:, :self._gain_anchor_hidden], emb[:, -1:]

    def discrete(self, emb, legal_templates=None):
        base, _ = self._split_embedding(emb)
        return super().discrete(base, legal_templates)

    def conditional(self, emb, coarse, template):
        base, anchor = self._split_embedding(emb)
        logits, means, factors = super().conditional(base, coarse, template)


        means = torch.cat([means[..., :-1], means[..., -1:]+anchor[:, None, :]], dim=-1)


        correction = factors.new_zeros((self.dim, self.dim))
        correction[-1, -1] = _GAIN_DIAGONAL_FLOOR-_BASE_DIAGONAL_FLOOR
        factors = factors+correction
        return logits, means, factors
