"""All conditional mixture components for the formal refinement interface.

Does not alter old1--3 representative selection or its frozen output. No
component is dropped, renormalized after selection, clipped or redrawn.
Gaussian chart centres are starts, not recovered structures or physical means.
"""
from __future__ import annotations
import copy
import numpy as np
from .neural_refine_chart import NAMES,decode
from .neural_refine_infer import physical_values
from .neural_refine_calibration import extract_mixture,calibrated_summary,safe_json


def component_starts(expert,mix,active,*,energy_ev,alpha_i_deg,metal):
    """Convert one complete mixture into all branch/component start records."""
    if len(mix['mu'])!=1:raise ValueError('One observed input per candidate collection')
    active=np.asarray(active,int);mask=np.zeros(len(NAMES),bool);mask[active]=True
    if not np.all(expert.active==mask):raise ValueError('Same active chart required by bound full-mixture calibration')
    weights=np.exp(mix['logw'][0]);centres=mix['mu'][0]
    if (centres.shape[:2]!=weights.shape or weights.shape[0]!=len(expert.prototypes)
            or not np.isfinite(weights).all() or not np.isfinite(centres).all()
            or not np.isclose(weights.sum(),1.,atol=1e-6)):raise ValueError('Complete finite normalized mixture required')
    bank=expert.declaration['bank'];mean=np.asarray(bank['normalization_mean']);scale=np.asarray(bank['normalization_scale'])
    out=[]
    for b in range(len(expert.prototypes)):
        prototype=copy.deepcopy(expert.prototypes[b]);prototype.update(energy_ev=float(energy_ev),alpha_i_deg=float(alpha_i_deg),metal=metal)
        for k in range(weights.shape[1]):
            z=np.zeros(len(NAMES));z[active]=centres[b,k]*scale[active]+mean[active]
            decoded=decode(prototype,z,mask)
            value=physical_values(z,mask)
            reasons=list(decoded['reasons']) if not decoded['valid'] else []
            status='component_centre_ready_for_physical_domain_check' if decoded['valid'] else 'component_centre_decoder_invalid'
            # A valid decoder centre may still lie outside a generator bound;
            # the unchanged refiner checks coupled physical bounds separately.
            if 'D_nm' in value and not 2<=value['D_nm']<=120:
                reasons.append('component_centre_D_outside_physical_2_120nm');status='component_centre_outside_D_model_domain'
            out.append(dict(mode_id=f'{b}:{k}',branch=bank['branches'][b],probability=float(weights[b,k]),
                original_component_probability=float(weights[b,k]),prototype=prototype,chart_z=z.tolist(),active=mask.tolist(),
                values={name:float(v) for name,v in value.items()},status=status,
                reason='; '.join(reasons) if reasons else 'Conditional chart centre; remaining coupled bounds checked by physical refinement',
                representative_source='Gaussian component centre in standardized joint chart, not a mean over physical branches',
                intervals=dict(scope='Per-component intervals not empirically calibrated; use separately returned full-mixture CAL intervals'),
                probability_scope='Whole original component mass, not a measured morphology probability or estimate of its physically valid mass',
                parameter_conditions={name:'No free parameter in this conditional forward; use model-specific conditions and distinguishing measurements'
                                      for j,name in enumerate(NAMES) if not mask[j]}))
    if len(out)!=int(np.prod(weights.shape)):raise ValueError('A mixture component disappeared')
    return out


def all_mode_prediction(expert,encoded,metal,calibration,*,energy_ev,alpha_i_deg):
    """Prediction-only all-mode entry. No truth, stage or Dcorr gate argument."""
    x=np.asarray(encoded,np.float32)
    if x.ndim==1:x=x[None]
    if len(x)!=1:raise ValueError('One transformed measured input required')
    # This also enforces checkpoint, policy and unchanged calibration arithmetic.
    intervals=calibrated_summary(expert,x,metal,calibration)
    mix,active=extract_mixture(expert,x)
    candidates=component_starts(expert,mix,active,energy_ev=energy_ev,alpha_i_deg=alpha_i_deg,metal=metal)
    return safe_json(dict(module=expert.declaration['bank']['module'],representatives=candidates,
        original_component_count=len(candidates),displayed_probability_not_renormalized=sum(r['probability'] for r in candidates),
        omitted_component_probability=0.,Dcorr_truncation=False,physical_D_domain_nm=[2.,120.],
        calibrated_full_mixture=intervals,
        stage=dict(status='conditional_module_only',reason='Common stage/scenario router not supplied to this component-level entry'),
        checkpoint_sha256=expert.checkpoint_sha256,policy_sha256=expert.declaration['bank']['input_policy_sha256'],
        refinement=dict(status='not_run_by_prediction_entry',reason='Pass every retained start to unchanged H0-H3 refine_bounded, recording invalid/out-of-domain starts'),
        validation_scope='All conditional network starts and CAL-adjusted full mixture; no post-refinement uncertainty or independent structural validation'))
