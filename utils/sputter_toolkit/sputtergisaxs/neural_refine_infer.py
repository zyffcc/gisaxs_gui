"""Load one new NR conditional expert and retain its mixture representatives.

Frozen v2 weights are not compatible and are never loaded here. This Python
entry is a component of the new workflow, not a claim of validated weights.
The caller can feed the returned representatives to refine_modes and retains
the sampled invalid/outside-window mass for end-to-end TEST accounting.
"""
from __future__ import annotations
from pathlib import Path
import copy,json
import numpy as np
from .neural_refine_chart import NAMES,INDEX,LOG1P,LOGIT,decode
from .neural_refine_input import extract,encode
from .neural_refine_training import apply_policy,policy_transform_sha,sha

# The shipped policy.json files keep only the keys that change the network input
# (policy_transform_sha) plus D_window_ratio=null. Each checkpoint records the
# SHA-256 of its original, longer policy file; this maps that record to the
# transform hash the shipped file must reproduce.
POLICY_TRANSFORM_SHA256={
    '938069fefc48b58940692065c67f1a5d69c5374f2b034e74e1d50b2f12372ce9':'150069e26b81c3185d1b81cb18b2402669395811bbe8dc2066a7ad95734c7d9e',
    '362f09a6c763b23a274dfeee96971f57b4aa43e7bacad9c053197fc71eca0ff9':'ffa7caa64cd9c40e4943a964c2b5b8c4c39be599f13db74c2bee4af6ca612fa6'}


def check_policy(policy,recorded_sha256):
    """The input transform of ``policy`` must be the one the checkpoint was trained with."""
    expected=POLICY_TRANSFORM_SHA256.get(recorded_sha256)
    if expected is None:raise ValueError('Checkpoint input policy is not one of the known trained policies')
    if policy_transform_sha(policy)!=expected:raise ValueError('The selected trained input policy changed')
    if policy.get('D_window_ratio') is not None:raise ValueError('Trained input policy has no D window (D_window_ratio must be null)')


def physical_values(z,active):
    z=np.asarray(z,float);out={}
    with np.errstate(over='ignore',invalid='ignore'):
        for j,name in enumerate(NAMES):
            if active[j]:out[name]=np.expm1(z[...,j]) if name in LOG1P else 1/(1+np.exp(-z[...,j])) if name in LOGIT else np.exp(z[...,j])
    return out


class NRExpert:
    def __init__(self,checkpoint,input_policy,*,device='cpu'):
        import torch
        from .neural_refine_posterior import NRPosterior
        self.device=torch.device(device);self.checkpoint=str(checkpoint);self.checkpoint_sha256=sha(checkpoint)
        saved=torch.load(checkpoint,map_location=self.device,weights_only=False);d=saved['declaration'];bank=d['bank']
        source=Path(__file__).resolve().parent
        for field,file in [('network_sha256','neural_refine_posterior.py'),('loader_sha256','neural_refine_training.py'),
                           ('chart_sha256','neural_refine_chart.py'),('mixture_sha256','v2_posterior.py'),('input_sha256','neural_refine_input.py')]:
            if sha(source/file)!=d[field]:raise ValueError('Inference arithmetic/input/chart differs from selected training source: '+file)
        self.policy=json.loads(Path(input_policy).read_text());self.declaration=d
        check_policy(self.policy,bank['input_policy_sha256'])
        self.net=NRPosterior(bank['input_dim'],bank['branches'],hidden=d['hidden'],components=d['components'],
            chart_mean=bank['normalization_mean'],chart_scale=bank['normalization_scale']).to(self.device)
        self.net.load_state_dict(saved['model']);self.net.eval()
        if not np.array_equal(self.net.chart_mean.detach().cpu().numpy(),np.array(bank['normalization_mean'],np.float32)) or not np.array_equal(self.net.chart_scale.detach().cpu().numpy(),np.array(bank['normalization_scale'],np.float32)):
            raise ValueError('Checkpoint chart normalization and TRAIN declaration differ')
        self.prototypes=bank['branch_prototypes'];self.active=np.asarray(bank['branch_active_masks'],bool)
        self.validation_scope='conditional NR expert; calibration and physical refinement validation reported separately'

    def predict_native(self,Y,E,*,energy_ev,alpha_i_deg,metal,draws_per_branch=512,max_representatives=3,
                       seed=202610208321,D_window_ratio=None,metal_peak_condition=False):
        """Return up to3 actual posterior-component representatives.

        D_window_ratio is a frozen DEV rule only under a declared metal-peak
        condition. A stationary host peak is not automatically such a peak.
        Retained ranges are conditional component quantiles, not post-refinement
        uncertainty. Their displayed union must be evaluated on the new TEST.
        """
        import torch
        if draws_per_branch<16 or not 1<=max_representatives<=3:raise ValueError('>=16 original draws and1--3 representatives required')
        item=extract(Y,E,energy_ev,alpha_i_deg,metal=metal);vector=apply_policy(encode(item),self.policy)
        x=dict(vector=torch.as_tensor(vector[None],device=self.device));rng=torch.Generator(device=self.device).manual_seed(seed)
        candidates=[];accounting=[];Dcorr=item['observations'].get('D_corr_nm')
        totals=dict(invalid_decoder_domain_probability=0.,valid_outside_D_probability=0.,sparse_valid_probability=0.,unsampled_component_probability=0.)
        # 2026-10-20: q*/Dcorr is input information, never an implicit hard
        # truncation. A caller must explicitly request any legacy window.
        if D_window_ratio is not None:
            if len(D_window_ratio)!=2 or not 0<D_window_ratio[0]<D_window_ratio[1]:raise ValueError('Declared frozen DEV D ratio rule required')
        with torch.no_grad():
            embedding=self.net.embedding(x)
            for label,branch in enumerate(self.net.branches):
                prototype=copy.deepcopy(self.prototypes[label]);prototype.update(energy_ev=float(energy_ev),alpha_i_deg=float(alpha_i_deg),metal=metal)
                active=self.active[label];draw=self.net.sample_branch(x,label,draws_per_branch,generator=rng)
                zz=draw['z'][0].cpu().numpy();component=draw['component'][0].cpu().numpy();branch_mass=float(draw['branch_probability'][0])
                c=torch.tensor([label],device=self.device);logits,_,_=self.net.conditional(embedding,c,torch.zeros_like(c))
                mixture_mass=torch.softmax(logits[0],-1).cpu().numpy()
                def valid_proposal(z):
                    result=decode(prototype,z,active)
                    return result['valid'] and 0<=result['parameters']['critical_angle_deg']<1
                valid=np.array([valid_proposal(z) for z in zz]);within=np.ones(len(zz),bool)
                anchor=bool(metal_peak_condition and Dcorr is not None and active[INDEX['D_nm']] and D_window_ratio is not None)
                if anchor:
                    ratios=np.exp(zz[:,INDEX['D_nm']])/Dcorr;within=(ratios>=D_window_ratio[0])&(ratios<=D_window_ratio[1])
                audit=dict(branch=list(branch),branch_probability=branch_mass,original_draws=len(zz),
                    draw_invalid_fraction=float((~valid).mean()),draw_valid_outside_D_fraction=float((valid&~within).mean()),
                    D_window_applied=anchor,components=[]);accounting.append(audit)
                for k,mass in enumerate(mixture_mass):
                    original=component==k;count=int(original.sum());original_mass=float(branch_mass*mass)
                    component_audit=dict(component=k,original_probability=original_mass,original_draws=count)
                    audit['components'].append(component_audit)
                    if count==0:
                        totals['unsampled_component_probability']+=original_mass;component_audit['status']='unsampled_component';continue
                    ids=np.flatnonzero(original&valid&within)
                    invalid_mass=original_mass*float((original&~valid).sum()/count)
                    outside_mass=original_mass*float((original&valid&~within).sum()/count)
                    retained_mass=original_mass*len(ids)/count
                    totals['invalid_decoder_domain_probability']+=invalid_mass;totals['valid_outside_D_probability']+=outside_mass
                    component_audit.update(invalid_decoder_probability=invalid_mass,valid_outside_D_probability=outside_mass,retained_probability=retained_mass)
                    if len(ids)<5:
                        totals['sparse_valid_probability']+=retained_mass;component_audit['status']='sparse_valid_component';continue
                    component_audit['status']='representative_available'
                    z=zz[ids];center=np.median(z[:,active],axis=0);norm=self.net.chart_scale.cpu().numpy()[active]
                    medoid=ids[np.argmin(np.sum(((z[:,active]-center)/norm)**2,axis=1))]
                    value=physical_values(zz[medoid],active);samples=physical_values(z,active)
                    intervals={name:dict(interval68=np.quantile(v,[.16,.84]).tolist(),interval95=np.quantile(v,[.025,.975]).tolist()) for name,v in samples.items()}
                    candidates.append(dict(mode_id=f'{label}:{k}',branch=list(branch),probability=retained_mass,original_component_probability=original_mass,
                        prototype=prototype,chart_z=zz[medoid].tolist(),active=active.tolist(),values={key:float(v) for key,v in value.items()},intervals=intervals,
                        original_component_draws=count,valid_conditioned_draws=len(ids),
                        representative_source='actual joint sample nearest the component coordinate-wise median in TRAIN-scaled chart; no averaging across physical branches',
                        interval_scope='component quantiles conditional on valid decoder/optical domain and any applied D window, before refinement; new TEST displayed-union coverage pending'))
        candidates.sort(key=lambda v:v['probability'],reverse=True);displayed=candidates[:max_representatives]
        undisplayed=float(sum(v['probability'] for v in candidates[max_representatives:]));shown=float(sum(v['probability'] for v in displayed))
        return dict(module=self.declaration['bank']['module'],observations=item['observations'],representatives=displayed,
            status='conditional_component_representatives' if displayed else 'no_supported_representative_from_original_draws',
            representative_reason='valid component draws available' if displayed else 'all components invalid, outside declared D window, or fewer than5 original valid draws; no redraw or fabricated structure',
            branch_accounting=accounting,undisplayed_valid_probability=undisplayed,displayed_probability_not_renormalized=shown,
            probability_accounting=dict(**totals,displayed_valid_probability=shown,undisplayed_valid_probability=undisplayed,
                total=shown+undisplayed+sum(totals.values()),scope='theoretical component mass times original-draw fractions; exclusive categories, finite Monte Carlo error'),
            stage_probability=None,stage_reason='single-module conditional expert is not a cross-module stage classifier',
            checkpoint_sha256=self.checkpoint_sha256,input_policy=self.policy,validation_scope=self.validation_scope,
            training_scope=self.declaration['bank'].get('training_scope','complete_production_bank'),
            D_anchor_policy='input_only_no_implicit_hard_window')
