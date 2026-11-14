'One observed q*/FWHM pair and mandatory geometry; full joint S1--S4 MDN.'
from numbers import Integral

import numpy as np
import torch
from torch import nn

from .scenario_chart import CATALOG, COORDINATES, DIM, SCHEMA as CHART_SCHEMA
from .scenario_posterior import ScenarioCandidate

SCHEMA = 'four_scenario_minimal_qwidth_joint_1'
INPUT_NAMES = ('log_q_star_nm_inv', 'log_FWHM_nm_inv', 'normalized_energy',
               'normalized_incidence', 'metal_is_Au', 'q_star_known', 'FWHM_known')
INPUT_DIM = len(INPUT_NAMES)
DECLARATION = dict(schema=SCHEMA, architecture='minimal_qwidth_joint',
    input_names=list(INPUT_NAMES), input_dim=INPUT_DIM,
    physical_inputs=['q_star_nm_inv','FWHM_nm_inv','energy_ev','alpha_i_deg','metal'],
    geometry_transform=dict(energy_center_ev=14000.,energy_scale_ev=6000.,
                            incidence_center_deg=.425,incidence_scale_deg=.275),
    AF_selection='caller-bound original noisy public observation band; no latent/CAL selection',
    missing_features='zero encoded log value plus explicit derived missing flag',
    excluded_inputs=['mandatory_log_intensity','mandatory_log_pixel_support',
                     'other_AF_band_pairs','highq','Yoneda','pre','QCM','host_truth','layer_optional'],
    chart_schema=CHART_SCHEMA, coordinates=list(COORDINATES), continuous_dim=DIM,
    nuisance_strategy='complete hierarchical joint learned under the same original scenario',
    nuisance_prior_resampled=False, scenarios_pruned=False, structural_release=False)


def _flags(values, supplied, np_shape):
    if supplied is None:
        return np.isfinite(values) & (values > 0)
    mask = np.broadcast_to(np.asarray(supplied), np_shape)
    if not np.isin(mask,[0,1]).all():
        raise ValueError('Feature-known flags must be Boolean or0/1')
    mask = mask.astype(bool)
    if np.any(mask & (~np.isfinite(values) | (values <= 0))):
        raise ValueError('Measured q*/FWHM must be finite and positive')
    return mask


def encode_pair(q_star_nm_inv, FWHM_nm_inv, energy_ev, alpha_i_deg, metal, *,
                q_star_known=None, FWHM_known=None):
    'Encode scalar/batched physical inputs, preserving feature absence.'
    q,w,E,ai,m = np.broadcast_arrays(np.atleast_1d(np.asarray(q_star_nm_inv,float)),
        np.atleast_1d(np.asarray(FWHM_nm_inv,float)),np.atleast_1d(np.asarray(energy_ev,float)),
        np.atleast_1d(np.asarray(alpha_i_deg,float)),np.atleast_1d(np.asarray(metal,object)))
    if q.ndim != 1 or not len(q):
        raise ValueError('Scalar or one-dimensional batch inputs required')
    if not np.isfinite(E).all() or not np.isfinite(ai).all() or np.any(E<=0) or np.any(ai<=0):
        raise ValueError('Positive finite mandatory energy/incidence required')
    if not np.isin(m,['Au','Ag']).all():
        raise ValueError('Mandatory metal must be Au or Ag')
    qk = _flags(q,q_star_known,q.shape); wk = _flags(w,FWHM_known,w.shape)
    vector = np.zeros((len(q),INPUT_DIM),np.float32)
    vector[qk,0] = np.log(q[qk]); vector[wk,1] = np.log(w[wk])
    vector[:,2] = (E-14000.)/6000.; vector[:,3] = (ai-.425)/.275
    vector[:,4] = (m=='Au'); vector[:,5] = qk; vector[:,6] = wk
    if not np.isfinite(vector).all():
        raise ValueError('Minimal input is not finite in the declared float32 encoder')
    return dict(vector=vector,image=None,feature_known=np.stack([qk,wk],axis=-1),
                complete_feature_pair=qk & wk)


def prepare(batch, *, AF_band):
    'Read exactly one measured pair from bank features40; no band search.'
    features = np.asarray(batch['features'])
    mandatory = np.asarray(batch['mandatory'])
    if features.ndim != 2 or features.shape[1] != 40:
        raise ValueError('Original features40 batch required')
    if mandatory.ndim != 2 or len(mandatory) != len(features) or mandatory.shape[1] < 3:
        raise ValueError('Original mandatory geometry/metal batch required')
    geometry = np.asarray(mandatory[:,:3],float)
    if not np.isfinite(geometry).all() or not np.isin(geometry[:,2],[0,1]).all():
        raise ValueError('Finite mandatory[:3] with Au/Ag encoding required')
    bands = np.broadcast_to(np.asarray(AF_band,dtype=object),(len(features),))
    vector = np.zeros((len(features),INPUT_DIM),np.float32); vector[:,2:5] = geometry
    selected = []
    for i,b in enumerate(bands):
        if b is None or (isinstance(b,Integral) and not isinstance(b,(bool,np.bool_)) and int(b)==-1):
            selected.append(None); continue
        if not isinstance(b,Integral) or isinstance(b,(bool,np.bool_)) or not 0<=int(b)<4:
            raise ValueError('Original observed AF band must be0..3 or missing')
        b = int(b); selected.append(b)
        qk,wk = features[i,20+2*b:22+2*b]
        if not np.isin([qk,wk],[0,1]).all():
            raise ValueError('Selected original feature flags must be0/1')
        if qk:
            logq = float(features[i,2*b])
            if not np.isfinite(logq):raise ValueError('Known original q* log feature is nonfinite')
            vector[i,0] = logq; vector[i,5] = 1.
            if wk:
                logwidth = logq+float(features[i,2*b+1])
                if not np.isfinite(logwidth):raise ValueError('Known original FWHM log feature is nonfinite')
                vector[i,1] = logwidth; vector[i,6] = 1.
    if not np.isfinite(vector).all():
        raise ValueError('Minimal encoder inputs are not finite float32 values')
    known = vector[:,5:7].astype(bool)
    return dict(vector=vector,image=None,feature_known=known,
        complete_feature_pair=known.all(axis=1),AF_band=selected,
        feature_selection=DECLARATION['AF_selection'])


class MinimalScenarioCandidate(ScenarioCandidate):
    'Seven-input encoder with original complete hierarchical conditional MDN.'
    def __init__(self, continuous_dim=DIM, coarse_classes=None, template_classes=1, *,
                 components=4, hidden=128, catalog=None, bank_scenario_prior=None):
        chosen = CATALOG if catalog is None else catalog
        super().__init__('features',continuous_dim,coarse_classes,template_classes,
            components=components,hidden=hidden,catalog=chosen,bank_scenario_prior=bank_scenario_prior)
        if self.dim != DIM:
            raise ValueError('Minimal model retains the complete original44D chart')
        self.encoder = nn.Sequential(nn.Linear(INPUT_DIM,hidden),nn.SiLU(),
                                     nn.Linear(hidden,hidden),nn.SiLU())
        self.architecture = DECLARATION['architecture']
        self.input_schema = SCHEMA
        self.validated = False

    def embedding(self, encoded):
        vector = encoded['vector']
        if (not torch.is_tensor(vector) or vector.ndim != 2 or vector.shape[1] != INPUT_DIM
                or not torch.isfinite(vector).all()):
            raise ValueError('Minimal finite [batch,7] tensor required')
        flags = vector[:,5:7]
        if ((flags!=0)&(flags!=1)).any() or ((vector[:,4]!=0)&(vector[:,4]!=1)).any():
            raise ValueError('Minimal derived flags and Au/Ag metal encoding required')
        if (vector[:,:2][flags==0]!=0).any():
            raise ValueError('Missing feature values must not carry hidden signal')
        return self.encoder(vector)
