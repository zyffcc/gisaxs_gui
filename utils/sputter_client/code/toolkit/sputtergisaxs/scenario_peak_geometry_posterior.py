'Single-cut joint posterior with its measured exit-angle geometry retained.'
from numbers import Integral

import numpy as np
import torch
from torch import nn

from .features import AF_BANDS
from .scenario_chart import DIM
from .scenario_minimal_posterior import (
    DECLARATION as MINIMAL_SPEC, encode_pair as minimal_encode, prepare as minimal_prepare)
from .scenario_peak_highq_posterior import (
    DECLARATION as HIGHQ_SPEC, PeakHighqScenarioCandidate,
    encode_pair as highq_encode, prepare as highq_prepare)

SCHEMA='four_scenario_peak_geometry_joint_1'
FEATURE_MODES=('qwidth','qwidth_highq')
GEOMETRY_NAMES=('alpha_f_lo_over_1_5deg','alpha_f_hi_over_1_5deg','alpha_f_geometry_known')


def input_spec(feature_mode):
    if feature_mode not in FEATURE_MODES:raise ValueError('Declared qwidth/qwidth_highq mode required')
    original=MINIMAL_SPEC if feature_mode=='qwidth' else HIGHQ_SPEC
    return dict(original,schema=SCHEMA,architecture='peak_geometry_joint',feature_mode=feature_mode,
        input_names=list(original['input_names'])+list(GEOMETRY_NAMES),input_dim=original['input_dim']+3,
        physical_inputs=list(original['physical_inputs'])+['measured_alpha_f_bounds_deg'],
        alpha_f_geometry=dict(bounds_scale_deg=1.5,original_bands_deg=[list(v) for v in AF_BANDS],
            centre='mean of observed bounds; no extra fitted feature',
            unknown='zero encoded bounds and geometry-known false; no default band'),
        D_primary=HIGHQ_SPEC['D_primary'],
        geometry_repair_hypothesis='omitting measured cut alpha_f can mix distinct conditional observations; no guaranteed performance')


INPUT_SPECS={mode:input_spec(mode) for mode in FEATURE_MODES}
DECLARATION=dict(schema=SCHEMA,architecture='peak_geometry_joint',feature_modes=list(FEATURE_MODES),
                 input_specs=INPUT_SPECS,structural_release=False)


def _geometry(n,AF_band,alpha_f_bounds_deg=None):
    'Measured geometry only; reconcile band/bounds rather than invent a cut.'
    bands=np.broadcast_to(np.asarray(AF_band,dtype=object),(n,))
    bounds=np.zeros((n,2),float);known=np.zeros(n,bool);selected=[]
    for i,b in enumerate(bands):
        if b is None or (isinstance(b,Integral) and not isinstance(b,(bool,np.bool_)) and int(b)==-1):
            selected.append(None);continue
        if not isinstance(b,Integral) or isinstance(b,(bool,np.bool_)) or not 0<=int(b)<len(AF_BANDS):
            raise ValueError('Actual measured AF band must be0..3 or unknown')
        selected.append(int(b));bounds[i]=AF_BANDS[int(b)];known[i]=True
    if alpha_f_bounds_deg is not None:
        supplied=np.broadcast_to(np.asarray(alpha_f_bounds_deg,float),(n,2))
        for i,(lo,hi) in enumerate(supplied):
            if not np.isfinite([lo,hi]).all():
                if np.isfinite([lo,hi]).any():raise ValueError('Both measured alpha_f bounds or neither are required')
                continue
            if lo<0 or hi<=lo:raise ValueError('Finite increasing nonnegative measured alpha_f bounds required')
            if known[i] and not np.allclose(bounds[i],[lo,hi],rtol=0,atol=1e-10):
                raise ValueError('Caller measured band/bounds disagree')
            bounds[i]=[lo,hi];known[i]=True
    geometry=np.column_stack([bounds/1.5,known]).astype(np.float32)
    if not np.isfinite(geometry).all():raise ValueError('Finite float32 measured-cut geometry required')
    return geometry,selected,[b.tolist() if k else None for b,k in zip(bounds,known)]


def encode_pair(q_star_nm_inv,FWHM_nm_inv,energy_ev,alpha_i_deg,metal,*,AF_band=None,
                alpha_f_bounds_deg=None,highq_ratio=None,feature_mode='qwidth_highq',
                q_star_known=None,FWHM_known=None,highq_known=None):
    'Physical q/width, optional same-cut high-q, mandatory geometry/metal.'
    input_spec(feature_mode)
    kwargs=dict(q_star_known=q_star_known,FWHM_known=FWHM_known)
    if feature_mode=='qwidth':base=minimal_encode(q_star_nm_inv,FWHM_nm_inv,energy_ev,alpha_i_deg,metal,**kwargs)
    else:base=highq_encode(q_star_nm_inv,FWHM_nm_inv,energy_ev,alpha_i_deg,metal,
                         highq_ratio=highq_ratio,highq_known=highq_known,**kwargs)
    geo,selected,bounds=_geometry(len(base['vector']),AF_band,alpha_f_bounds_deg)
    return dict(base,vector=np.concatenate([base['vector'],geo],axis=1),feature_mode=feature_mode,
        AF_band=selected,AF_bounds_deg=bounds,AF_center_deg=[None if b is None else float(np.mean(b)) for b in bounds],
        AF_geometry_known=geo[:,-1].astype(bool))


def prepare(batch,*,AF_band,feature_mode='qwidth_highq'):
    'Same original scalar cut; append its geometry, never choose a new band.'
    input_spec(feature_mode)
    base=(minimal_prepare(batch,AF_band=AF_band) if feature_mode=='qwidth' else
          highq_prepare(batch,AF_band=AF_band))
    geo,selected,bounds=_geometry(len(base['vector']),base['AF_band'])
    return dict(base,vector=np.concatenate([base['vector'],geo],axis=1),feature_mode=feature_mode,
        AF_band=selected,AF_bounds_deg=bounds,AF_center_deg=[None if b is None else float(np.mean(b)) for b in bounds],
        AF_geometry_known=geo[:,-1].astype(bool))


class PeakGeometryScenarioCandidate(PeakHighqScenarioCandidate):
    'Original joint MDN and proper exact D score; only measured-cut encoder differs.'
    def __init__(self,continuous_dim=DIM,coarse_classes=None,template_classes=1,*,feature_mode='qwidth_highq',
                 components=4,hidden=128,catalog=None,bank_scenario_prior=None):
        spec=input_spec(feature_mode)
        super().__init__(continuous_dim,coarse_classes,template_classes,components=components,hidden=hidden,
                         catalog=catalog,bank_scenario_prior=bank_scenario_prior)
        self.feature_mode=feature_mode;self.input_dim=spec['input_dim']
        self.encoder=nn.Sequential(nn.Linear(self.input_dim,hidden),nn.SiLU(),nn.Linear(hidden,hidden),nn.SiLU())
        self.architecture=spec['architecture'];self.input_schema=SCHEMA;self.validated=False

    def embedding(self,encoded):
        vector=encoded['vector']
        if (not torch.is_tensor(vector) or vector.ndim!=2 or vector.shape[1]!=self.input_dim
                or not torch.isfinite(vector).all()):raise ValueError('Finite declared scalar-cut geometry input required')
        flags=vector[:,[5,6]] if self.feature_mode=='qwidth' else vector[:,[5,6,8]]
        values=vector[:,:2] if self.feature_mode=='qwidth' else vector[:,[0,1,7]]
        geometry_known=vector[:,-1]
        if (((flags!=0)&(flags!=1)).any() or ((vector[:,4]!=0)&(vector[:,4]!=1)).any()
                or (values[flags==0]!=0).any() or ((geometry_known!=0)&(geometry_known!=1)).any()
                or (vector[:,-3:-1][geometry_known==0]!=0).any()):
            raise ValueError('Explicit known flags and zero missing feature/geometry values required')
        known=geometry_known==1
        if ((vector[known,-3]<0).any() or (vector[known,-2]<=vector[known,-3]).any()):
            raise ValueError('Known measured alpha_f bounds must remain increasing')
        return self.encoder(vector)
