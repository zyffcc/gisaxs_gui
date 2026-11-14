'Actual51D single-pair minimal encoder, full-joint model and strict runtime.'
from numbers import Integral
from pathlib import Path
import json

import numpy as np
import torch
from torch import nn

from .scenario_mainline_chart import DIM,SCHEMA,COORDINATES,CATALOG,SCENARIO_NAMES,decode
from .scenario_posterior import ScenarioCandidate
from .scenario_mainline_posterior import legal_from_metadata
from .scenario_mainline_training import generating_source_sha,BANK_SCHEMAS
from .scenario_mainline_ppc import MainlinePPCRuntime
from .scenario_candidates import observe_candidate_frame
from .scenario_peak_support_ppc import comparison_observed,enhance_ppc
from .scenario_runtime import (_sha,_source_sha,_state_digest,_array_sha,
    _frame_identity,_labels,_conditional_summaries,_public_features)
from .v2_input import FEATURE_SCALE

ARCHITECTURE='mainline51_minimal_qwidth_joint_v1'
TRAINING_SCHEMA='mainline51_minimal_qwidth_training_1'
MODE='features'
ROOT_SEED=683120261014
INPUT_SPEC=dict(schema='mainline51_selected_single_pair_input_1',architecture=ARCHITECTURE,
    encoder_dim=7,encoder_names=['log_q_star_nm_inv','log_FWHM_nm_inv',
        'normalized_energy','normalized_incidence','metal_is_Au','q_star_known','FWHM_known'],
    envelope_dim=54,envelope_pair_indices=[0,1],envelope_known_indices=[20,21],
    envelope_geometry_indices=[40,41,42],
    envelope_pair_definition='0=log(q*), 1=log(FWHM/q*); zero value with missing flag when unknown',
    encoder_width_definition='log(FWHM)=envelope0+envelope1 only when both flags known',
    geometry_transform=dict(energy_center_ev=14000.,energy_scale_ev=6000.,
        incidence_center_deg=.425,incidence_scale_deg=.275),
    selected_pair='observe_candidate_frame plus comparison_observed on original public counts; selected measured AF retained for PPC',
    single_side='q retained; FWHM unknown; no synthetic bilateral width',
    host_input='legal scenario/template mask only, never encoder',
    excluded_inputs=['other_AF_pairs','image','intensity','support_magnitude','highq','valley',
        'Yoneda','signed_delta','reference','QCM','layer_thickness','host_truth'],
    chart_schema=SCHEMA,continuous_dim=DIM,coordinates=list(COORDINATES),
    posterior='hierarchical coarse/template plus4component complete51D Gaussian mixture',
    gain_anchor=False,delta_anchor=False,nuisance_prior_resampled=False,structural_release=False)


def _positive(value):
    return bool(isinstance(value,(int,float,np.integer,np.floating))
        and not isinstance(value,(bool,np.bool_)) and np.isfinite(value) and value>0)


def pair_features(observed):
    'Selected observed pair to the b0-compatible40D cache; not AF band0.'
    if not isinstance(observed,dict) or not isinstance(observed.get('values'),dict):
        raise ValueError('Actual selected-pair observed descriptor required')
    q=observed['values'].get('q_star');width=observed['values'].get('FWHM')
    qknown=_positive(q)
    wknown=qknown and _positive(width) and observed.get('support')!='one_side'
    result=np.zeros(40,np.float32)
    if qknown:result[0]=np.log(float(q));result[20]=1.
    if wknown:result[1]=np.log(float(width)/float(q));result[21]=1.
    if not np.isfinite(result).all():raise ValueError('Selected pair does not fit finite float32 input')
    return result


def _envelope(features,mandatory):
    f=np.asarray(features);m=np.asarray(mandatory)
    if f.ndim==1:f=f[None]
    if m.ndim==1:m=m[None]
    if f.ndim!=2 or f.shape[1]!=40 or m.ndim!=2 or len(m)!=len(f) or m.shape[1]<3:
        raise ValueError('Selected-pair40D and original mandatory[:3] batch required')
    flags=np.asarray(f[:,[20,21]],float);geometry=np.asarray(m[:,:3],float)
    if (not np.isin(flags,[0,1]).all() or np.any((flags[:,1]!=0)&(flags[:,0]==0))
            or not np.isfinite(geometry).all() or not np.isin(geometry[:,2],[0,1]).all()):
        raise ValueError('Derived pair flags and required normalized geometry/metal differ')
    pair=np.where(flags!=0,np.asarray(f[:,:2],float),0.)
    if not np.isfinite(pair).all():raise ValueError('Known selected pair must be finite')
    vector=np.zeros((len(f),54),np.float32)
    vector[:,:2]=pair;vector[:,20:22]=flags;vector[:,40:43]=geometry
    if not np.isfinite(vector).all():raise ValueError('Required input is not finite float32')
    return dict(vector=vector,image=None,feature_known=flags.astype(bool),
        complete_feature_pair=flags.astype(bool).all(axis=1),input_mode=MODE,input_spec=INPUT_SPEC)


def encode_pair(q_or_features,width_or_mandatory,energy_ev=None,alpha_i_deg=None,metal=None,*,
        q_star_known=None,FWHM_known=None):
    'Return the compatible54D envelope with exactly seven used coordinates.'
    if energy_ev is None and alpha_i_deg is None and metal is None:
        if q_star_known is not None or FWHM_known is not None:
            raise ValueError('Cache form already carries its original two known flags')
        return _envelope(q_or_features,width_or_mandatory)
    if energy_ev is None or alpha_i_deg is None or metal is None:
        raise ValueError('Physical input form requires E, incidence and Au/Ag')
    q,w,E,ai,m=np.broadcast_arrays(np.atleast_1d(np.asarray(q_or_features,float)),
        np.atleast_1d(np.asarray(width_or_mandatory,float)),np.atleast_1d(np.asarray(energy_ev,float)),
        np.atleast_1d(np.asarray(alpha_i_deg,float)),np.atleast_1d(np.asarray(metal,object)))
    if (q.ndim!=1 or not len(q) or not np.isfinite(E).all() or not np.isfinite(ai).all()
            or np.any(E<=0) or np.any(ai<=0) or not np.isin(m,['Au','Ag']).all()):
        raise ValueError('Scalar/batch positive energy/incidence and Au/Ag required')
    def known(values,supplied):
        valid=np.isfinite(values)&(values>0)
        if supplied is None:return valid
        flag=np.broadcast_to(np.asarray(supplied),values.shape)
        if not np.isin(flag,[0,1]).all() or np.any((flag!=0)&~valid):
            raise ValueError('Known feature flags require actual finite positive measurements')
        return flag.astype(bool)
    qk=known(q,q_star_known);wk=known(w,FWHM_known)&qk
    f=np.zeros((len(q),40),np.float32)
    f[qk,0]=np.log(q[qk]);f[wk,1]=np.log(w[wk]/q[wk]);f[:,20]=qk;f[:,21]=wk
    mandatory=np.column_stack([(E-14000.)/6000.,(ai-.425)/.275,m=='Au'])
    return _envelope(f,mandatory)


class Minimal51Candidate(ScenarioCandidate):
    'Proper full51D mixture; only its encoder is the new seven-input one.'
    def __init__(self,mode=MODE,continuous_dim=DIM,coarse_classes=len(CATALOG),template_classes=1,*,
            components=4,hidden=128,catalog=None,bank_scenario_prior=None):
        chosen=CATALOG if catalog is None else tuple(catalog)
        if (mode!=MODE or continuous_dim!=DIM or coarse_classes!=len(CATALOG)
                or tuple(chosen)!=CATALOG or components!=4 or hidden!=128):
            raise ValueError('Explicit actual51D/catalog and fixed features128/four-component factory required')
        if not np.array_equal(FEATURE_SCALE[:2],np.ones(2)):
            raise ValueError('Existing pair-envelope log feature scales changed')
        super().__init__(MODE,continuous_dim,coarse_classes,template_classes,
            components=components,hidden=hidden,catalog=chosen,bank_scenario_prior=bank_scenario_prior)
        self.encoder=nn.Sequential(nn.Linear(7,hidden),nn.SiLU(),nn.Linear(hidden,hidden),nn.SiLU())
        self.architecture=ARCHITECTURE;self.input_spec=INPUT_SPEC
        self.validated=False;self.structural_release=False

    def embedding(self,encoded):
        vector=encoded['vector']
        if not torch.is_tensor(vector) or vector.ndim!=2 or vector.shape[1]!=54:
            raise ValueError('Compatible54D selected-pair envelope tensor required')


        flags=vector[:,20:22];geometry=vector[:,40:43]
        if (not torch.all((flags==0)|(flags==1)) or torch.any((flags[:,1]!=0)&(flags[:,0]==0))
                or not torch.isfinite(geometry).all() or not torch.all((geometry[:,2]==0)|(geometry[:,2]==1))):
            raise ValueError('Minimal flags and required geometry/metal must remain valid')
        logq=torch.where(flags[:,:1]!=0,vector[:,:1],torch.zeros_like(vector[:,:1]))
        logwidth=torch.where(flags[:,1:2]!=0,vector[:,:1]+vector[:,1:2],torch.zeros_like(vector[:,1:2]))
        used=torch.cat([logq,logwidth,geometry,flags],-1)
        if not torch.isfinite(used).all():raise ValueError('The seven actual used inputs must be finite')
        return self.encoder(used)


def selected_pair_input(full,public,energy_ev,alpha_i_deg,metal):
    'Actual public runtime/preflight operator; return envelope and observed.'
    identity=_frame_identity(full,public)
    if (np.any(public.P>full.P) or np.any(public.Y>full.Y)
            or np.any((full.P==public.P)&(full.Y!=public.Y))):
        raise ValueError('Original public photons/support must remain a subset of full observation')
    if metal not in ('Au','Ag') or not np.isfinite([energy_ev,alpha_i_deg]).all() or min(energy_ev,alpha_i_deg)<=0:
        raise ValueError('Positive mandatory geometry and Au/Ag required')
    observed=comparison_observed(public,observe_candidate_frame(public,
        estimate_stat_se=identity['photon_statistics_known']))
    mandatory=np.array([(float(energy_ev)-14000.)/6000.,(float(alpha_i_deg)-.425)/.275,metal=='Au'],float)
    return encode_pair(pair_features(observed),mandatory),observed


class Minimal51Runtime(MainlinePPCRuntime):
    'Strict independent weights; native actual51D PPC is inherited unchanged.'
    def __init__(self,checkpoint,*,device='cpu',package_root=None):
        self.path=Path(checkpoint).resolve()
        package=Path(package_root).resolve() if package_root is not None else Path(__file__).resolve().parents[1]
        side=self.path.with_name(self.path.name+'.sha256');digest=side.read_text(encoding='ascii').split()[0]
        if _sha(self.path)!=digest:raise ValueError('Actual selected minimal51 checkpoint SHA differs')
        saved=torch.load(self.path,map_location='cpu',weights_only=True)
        if _sha(self.path)!=digest:raise ValueError('Selected minimal51 checkpoint changed during read')
        d=saved['declaration'];self.declaration=d
        if (d.get('schema')!=TRAINING_SCHEMA or d.get('architecture')!=ARCHITECTURE
                or d.get('input_spec')!=INPUT_SPEC or d.get('mode')!=MODE
                or d.get('seed')!=ROOT_SEED or d.get('hidden')!=128 or d.get('components')!=4
                or d.get('fixture') is not False or d.get('TEST_inspected') is not False
                or d.get('physics_schema')!=SCHEMA or d['dim']!=DIM or d['continuous_dim']!=DIM
                or d['coordinates']!=list(COORDINATES) or d['catalog']!=list(CATALOG)
                or d['bank'].get('schema') not in BANK_SCHEMAS or d['bank'].get('original_families')!=30000
                or d['bank'].get('TEST_scientific_rows_decoded') is not False
                or d['bank'].get('optional_signed_delta_bank') is not None):
            raise ValueError('Only new actual51D selected-pair minimal training is accepted')
        cache=d.get('pair_cache',{})
        if (cache.get('schema')!='mainline51_minimal_selected_pair_cache_1' or cache.get('complete') is not True
                or cache.get('original_families')!=30000 or cache.get('TEST_read') is not False
                or cache.get('raw_read') is not False or cache.get('processing_failed_family_ids')
                or cache.get('bank_plan_sha256')!=d['bank']['plan_sha256']
                or cache.get('bank_run_sha256')!=d['bank']['bank_run_sha256']
                or cache.get('split_rows')!=d['bank']['split_rows']
                or cache.get('successful_split_rows')!=d['bank']['successful_split_rows']
                or cache.get('generation_failed_family_ids')!=d['bank']['generation_failed_family_ids']
                or cache.get('generation_source_pins')!=d['bank']['source_pins']):
            raise ValueError('Complete original selected-pair cache must bind the exact generating bank')
        self.templates=tuple(d['template_metadata']);self.keys=tuple(d['template_keys'])
        if (self.keys!=tuple(t['key'] for t in self.templates) or len(set(self.keys))!=len(self.keys)
                or d['coarse_classes']!=len(CATALOG) or d['template_classes']!=len(self.keys)+1
                or any(t.get('split')!='TRAIN' for t in self.templates)):
            raise ValueError('Unique bound original TRAIN template/catalog dimensions required')
        self.pins=dict(d['source_pins']);current=Path(__file__).resolve().parents[1]
        required={**d['bank']['source_pins'],**cache['source_pins']}
        if any(self.pins.get(k)!=v for k,v in required.items()):
            raise ValueError('Original generating and scalar-cache source pins must remain bound')
        for relative,value in self.pins.items():
            fn=generating_source_sha if relative in d['bank']['source_pins'] else _source_sha
            if fn(package/relative)!=value or fn(current/relative)!=value:
                raise ValueError('Original trained/executing minimal51 source differs: '+relative)
        if _state_digest(saved['model'])!=saved['state_sha256']:
            raise ValueError('Complete selected minimal51 state digest differs')
        rp=self.path.parent/'run.json'
        if _sha(rp)!=rp.with_name(rp.name+'.sha256').read_text(encoding='ascii').split()[0]:
            raise ValueError('Original selected run receipt SHA differs')
        run=json.loads(rp.read_text(encoding='utf8'))
        if (run.get('declaration')!=d or run.get('best') is None
                or any(saved[k]!=run['best'][k] for k in ('epoch','state_sha256','CAL_NLL'))):
            raise ValueError('Selected actual complete epoch differs from its training run')
        self.device=torch.device(device);self.mode=MODE;self.package=package
        self.net=Minimal51Candidate(MODE,DIM,len(CATALOG),len(self.keys)+1,
            components=d['components'],hidden=d['hidden'],catalog=CATALOG,
            bank_scenario_prior=d['bank_scenario_prior']).to(self.device)
        self.net.load_state_dict(saved['model'],strict=True);self.net.eval()
        self.receipt=dict(schema='actual_mainline51_minimal_runtime_1',weights_sha256=digest,
            state_sha256=saved['state_sha256'],selected_epoch=saved['epoch'],CAL_NLL=saved['CAL_NLL'],
            architecture=ARCHITECTURE,input_spec=INPUT_SPEC,bank=d['bank'],pair_cache=cache,
            mode=MODE,source_pins=self.pins,training_terminal_status=run.get('status'),trained=True,
            calibrated=False,production_selected=False,structural_release=False)

    def sample(self,full,public,energy_ev,alpha_i_deg,metal,*,host_category=None,
            pre=None,thickness_nm=None,qcm_rate=None,delta=None,n=256,seed=ROOT_SEED):
        if (isinstance(n,bool) or not isinstance(n,Integral) or n<2
                or isinstance(seed,bool) or not isinstance(seed,Integral) or seed<0):
            raise ValueError('Original draw count>=2 and nonnegative integer seed required')
        identity=_frame_identity(full,public)
        prepared,observed=selected_pair_input(full,public,energy_ev,alpha_i_deg,metal)
        encoded=dict(vector=torch.as_tensor(prepared['vector'],device=self.device),image=None)
        legal=legal_from_metadata(energy_ev,alpha_i_deg,self.templates,host_categories=[host_category],device=self.device)
        generator=torch.Generator(device=self.device).manual_seed(int(seed))
        sampled=self.net.sample_all_applicable(encoded,int(n),legal_templates=legal,generator=generator,host_categories=[host_category])
        scenes={}
        for scene in SCENARIO_NAMES:
            part=sampled['samples'][scene];applicable=bool(part['applicable'][0]);records=[]
            original={k:part[k][0].detach().cpu().numpy() for k in ('z','coarse','template','component')}
            if applicable:
                for i in range(int(n)):
                    try:
                        physical=decode(original['z'][i],int(original['coarse'][i]),int(original['template'][i]),
                            metal=metal,E=float(energy_ev),ai=float(alpha_i_deg),template_keys=self.keys,template_metadata=self.templates)
                        if physical['scenario']!=scene:raise ValueError('Original fixed-S joint changed scenario')
                        labels=_labels(physical)
                        if scene=='S4':labels['domain_nm']=float(physical['domain_nm'])
                        records.append(dict(index=i,decode_failed=False,physical=physical,conditional_labels=labels))
                    except (ValueError,TypeError,KeyError,IndexError,OverflowError,FloatingPointError,RuntimeError) as error:
                        records.append(dict(index=i,decode_failed=True,physical=None,conditional_labels={},reason=str(error)))
            scenes[scene]=dict(applicable=applicable,not_applicable_reason=part['not_applicable_reason'][0],
                original_draws=len(records),original_joint=original,draws=records,
                conditional_quantities=_conditional_summaries(scene,records),


                public_quantities=dict(D_nm=None,diameter_nm=None,height_nm=None,gap_nm=None,coverage_fraction=None),
                missing_reason='Internal research metadata; public representative structures are assembled by the candidate wrapper',
                calibrated=False,structural_release=False)
        probabilities=sampled['scenario_comparison']['probabilities'][0].detach().cpu().numpy()
        ignored=[key for key,value in (('reference',pre),('thickness_nm',thickness_nm),('QCM',qcm_rate),('signed_delta',delta)) if value is not None]
        return dict(receipt=self.receipt,geometry=dict(energy_ev=float(energy_ev),alpha_i_deg=float(alpha_i_deg),metal=metal),
            frame_identity=identity,observed=observed,features=_public_features(observed),scenarios=scenes,
            input_AF_band=observed['AF_band'],input_vector_sha256=_array_sha(prepared['vector']),
            input_pair_available=bool(prepared['complete_feature_pair'][0]),host_category_input=host_category,
            input_spec=INPUT_SPEC,optional_inputs_not_consumed_by_minimal_encoder=ignored,
            reference_scenario_probabilities=dict(zip(SCENARIO_NAMES,probabilities.tolist())),
            scenario_probabilities_are_reference_only=True,scenarios_pruned=False,
            original_seed=int(seed),original_draws_per_applicable_scenario=int(n),
            conditioned_outside_D=False,nuisance_prior_resampled=False,calibrated=False,structural_release=False)

    def posterior_predictive(self,sampled,full,public,*,bank_plan,feature_certificate=None,
            backend='cpu',seed=ROOT_SEED,profile_intensity=True):
        'Native51D original joint PPC; q/width gates, other features warnings.'
        result=super().posterior_predictive(sampled,full,public,bank_plan=bank_plan,
            feature_certificate=feature_certificate,backend=backend,seed=seed,profile_intensity=profile_intensity)
        for scene,part in result['scenarios'].items():
            if part.get('draws') is None:
                part.update(fastline_qwidth_pass=False,fastline_feature_gates=['q_star','FWHM'])
                continue
            stats=enhance_ppc(sampled['observed'],part['draws'],sampled['scenarios'][scene]['original_draws'])
            measured=[stats['features'][key] for key in ('q_star','FWHM') if stats['features'][key]['applicable']]
            part.update(PPC=stats,fastline_qwidth_pass=bool(measured and all(p['passed'] for p in measured)),
                fastline_feature_gates=['q_star','FWHM'],fastline_measured_gate_count=len(measured),
                warning_features={key:dict(stats['features'][key],fastline_gate=False)
                    for key in ('highq_ratio','valley_ratio','Yoneda')})
        result.update(input_spec=INPUT_SPEC,fastline_feature_gates=['q_star','FWHM'],
            peak_support_and_other_features_are_diagnostics=True,structural_release=False)
        return result

