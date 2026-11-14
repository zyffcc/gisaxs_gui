'Explicit mainline51D full-joint mixture and uncalibrated sampling adapter.'
from numbers import Integral
from pathlib import Path

import numpy as np
import torch

from .scenario_mainline_chart import (SCHEMA,DIM,COORDINATES,CATALOG,
    SCENARIO_NAMES,template_compatibility,decode)
from .scenario_posterior import ScenarioCandidate
from .scenario_runtime import (_sha,_source_sha,_array_sha,_frame_identity,
    _labels,_conditional_summaries,_public_features,_state_digest)
from .v2_input import MODES,INPUT_KEYS,prepare
from .v2_observation import frame_input,ppc_features
from .scenario_mainline_delta import VECTOR_DIM as DELTA_DIM,INPUT_SPEC as DELTA_SPEC,prepare as prepare_delta
from .scenario_mainline_training import generating_source_sha,BANK_SCHEMAS

ARCHITECTURE='mainline_gain_anchor_optional_signed_delta_v2'
TRAINING_SCHEMA='exploratory_mainline_joint_training_1'
VECTOR_DIM=dict(features=54,hybrid=54,qimage=14,rowimage=14)
ANCHOR_INDEX=dict(features=43,hybrid=43,qimage=3,rowimage=3)


def architecture_spec():
    return dict(name=ARCHITECTURE,continuous_dim=DIM,catalog_classes=len(CATALOG),
        chart_schema=SCHEMA,coordinate_names=list(COORDINATES),
        input_preparation='unchanged four-mode v2_input.prepare',
        vector_dimensions=VECTOR_DIM,anchor_vector_indices=ANCHOR_INDEX,
        gain='observed mandatory3 plus learned joint residual; translation Jacobian1',
        covariance='complete joint Cholesky; gain floor.001, all other floors.02',
        optional_signed_delta=dict(DELTA_SPEC,projection='1124 to64 to hidden; contribution zero when no delta points known; gain anchor unchanged'),
        sampler='original fixed-S categorical/template/component/complete continuous draw',
        exploratory=True,parameter_intervals_calibrated=False,structural_release=False)


class MainlineCandidate(ScenarioCandidate):
    def __init__(self,mode,continuous_dim=DIM,coarse_classes=len(CATALOG),template_classes=1,
                 *,components=4,hidden=128,catalog=None,bank_scenario_prior=None):
        selected=CATALOG if catalog is None else tuple(catalog)
        if continuous_dim!=DIM or coarse_classes!=len(CATALOG) or tuple(selected)!=CATALOG or COORDINATES[-1]!='log_target_scale':
            raise ValueError('Explicit actual mainline51D/catalog factory required')
        super().__init__(mode,continuous_dim,coarse_classes,template_classes,
            components=components,hidden=hidden,catalog=selected,bank_scenario_prior=bank_scenario_prior)
        self.delta_encoder=torch.nn.Sequential(torch.nn.Linear(DELTA_DIM,64),torch.nn.SiLU(),torch.nn.Linear(64,hidden))
        torch.nn.init.zeros_(self.delta_encoder[-1].weight);torch.nn.init.zeros_(self.delta_encoder[-1].bias)
        self._anchor_hidden=hidden;self.architecture=ARCHITECTURE;self.validated=False;self.structural_release=False

    def embedding(self,encoded):
        vector=encoded['vector']
        if vector.ndim!=2 or vector.shape[1]!=VECTOR_DIM[self.mode]:raise ValueError('Original four-mode input dimension differs')
        anchor=vector[:,ANCHOR_INDEX[self.mode]:ANCHOR_INDEX[self.mode]+1]
        if not torch.isfinite(anchor).all():raise ValueError('Finite observed log-gain anchor required')
        base=super().embedding(encoded);delta=encoded.get('delta')
        if delta is not None:
            if delta.shape!=(len(base),DELTA_DIM) or not torch.isfinite(delta).all():
                raise ValueError('Finite separate signed-delta input vector required')
            cells=(DELTA_DIM-4)//2;known=delta[:,cells:2*cells]
            if not torch.all((known==0)|(known==1)):raise ValueError('Explicit binary signed-delta presence flags required')
            enabled=known.any(-1).to(base.dtype)[:,None]
            base=base+self.delta_encoder(delta)*enabled
        return torch.cat([base,anchor],-1)

    def _split(self,embedding):
        if embedding.ndim!=2 or embedding.shape[1]!=self._anchor_hidden+1:raise ValueError('Complete mainline anchor embedding required')
        return embedding[:,:-1],embedding[:,-1:]

    def discrete(self,embedding,legal_templates=None):
        base,_=self._split(embedding)
        return super().discrete(base,legal_templates)

    def conditional(self,embedding,coarse,template):
        base,anchor=self._split(embedding);logits,means,factors=super().conditional(base,coarse,template)
        means=torch.cat([means[...,:-1],means[...,-1:]+anchor[:,None,:]],-1)
        correction=factors.new_zeros((self.dim,self.dim));correction[-1,-1]=-.019
        return logits,means,factors+correction


def legal_from_metadata(E,ai,templates,*,host_categories=None,device=None):
    E,ai=np.broadcast_arrays(np.atleast_1d(np.asarray(E,float)),np.atleast_1d(np.asarray(ai,float)))
    if E.ndim!=1:raise ValueError('One-dimensional mandatory geometry required')
    categories=[None]*len(E) if host_categories is None else [host_categories]*len(E) if isinstance(host_categories,str) else list(host_categories)
    if len(categories)!=len(E):raise ValueError('One optional host category per frame required')
    return torch.as_tensor(np.stack([template_compatibility(e,a,templates,host_category=h)
        for e,a,h in zip(E,ai,categories)]),dtype=torch.bool,device=device)


class MainlineRuntime:
    'Strict actual mainline best.pt loader; no release/calibration receipt implied.'
    def __init__(self,checkpoint,*,device='cpu',package_root=None):
        import json
        self.path=Path(checkpoint).resolve();package=Path(package_root).resolve() if package_root is not None else Path(__file__).resolve().parents[1]
        side=self.path.with_name(self.path.name+'.sha256');digest=side.read_text(encoding='ascii').split()[0]
        if _sha(self.path)!=digest:raise ValueError('Selected actual mainline checkpoint SHA differs')
        saved=torch.load(self.path,map_location='cpu',weights_only=True)
        if _sha(self.path)!=digest:raise ValueError('Selected checkpoint changed during read')
        d=saved['declaration'];self.declaration=d
        if (d.get('schema')!=TRAINING_SCHEMA or d.get('architecture')!=ARCHITECTURE
                or d.get('architecture_spec')!=architecture_spec() or d.get('TEST_inspected') is not False
                or d.get('physics_schema')!=SCHEMA or d['dim']!=DIM or d['continuous_dim']!=DIM
                or d['coordinates']!=list(COORDINATES) or d['catalog']!=list(CATALOG)
                or d['bank'].get('schema') not in BANK_SCHEMAS
                or d['bank'].get('TEST_scientific_rows_decoded') is not False):
            raise ValueError('Actual explicit mainline chart/model declaration required')
        self.templates=tuple(d['template_metadata']);self.keys=tuple(d['template_keys'])
        if (self.keys!=tuple(t['key'] for t in self.templates) or len(set(self.keys))!=len(self.keys)
                or d['coarse_classes']!=len(CATALOG) or d['template_classes']!=len(self.keys)+1
                or any(t.get('split')!='TRAIN' for t in self.templates)):
            raise ValueError('Actual unique TRAIN host templates/catalog dimensions differ')
        self.pins=dict(d['source_pins']);current=Path(__file__).resolve().parents[1]
        if (not set(d['bank']['source_pins']).issubset(self.pins)
                or any(self.pins[k]!=v for k,v in d['bank']['source_pins'].items())):
            raise ValueError('Original generating bank sources must remain bound in the runtime')
        for relative,value in self.pins.items():
            digest_fn=generating_source_sha if relative in d['bank']['source_pins'] else _source_sha
            if digest_fn(package/relative)!=value or digest_fn(current/relative)!=value:
                raise ValueError('Actual training/executing mainline source differs: '+relative)
        if _state_digest(saved['model'])!=saved['state_sha256']:raise ValueError('Complete mainline state digest differs')
        run_path=self.path.parent/'run.json'
        if _sha(run_path)!=run_path.with_name(run_path.name+'.sha256').read_text(encoding='ascii').split()[0]:
            raise ValueError('Original mainline training run SHA differs')
        run=json.loads(run_path.read_text(encoding='utf8'))
        if (run.get('declaration')!=d or run.get('best') is None
                or any(saved[k]!=run['best'][k] for k in ('epoch','state_sha256','CAL_NLL'))):
            raise ValueError('Selected complete mainline epoch differs from its original run')
        self.device=torch.device(device);self.mode=d['mode']
        self.net=MainlineCandidate(self.mode,DIM,len(CATALOG),len(self.keys)+1,
            components=d['components'],hidden=d['hidden'],catalog=CATALOG,
            bank_scenario_prior=d['bank_scenario_prior']).to(self.device)
        self.net.load_state_dict(saved['model'],strict=True);self.net.eval()
        self.receipt=dict(schema='actual_exploratory_mainline_runtime_1',weights_sha256=digest,
            state_sha256=saved['state_sha256'],selected_epoch=saved['epoch'],CAL_NLL=saved['CAL_NLL'],
            architecture=ARCHITECTURE,bank=d['bank'],mode=self.mode,source_pins=self.pins,
            training_terminal_status=run.get('status'),trained=True,calibrated=False,
            exploratory=True,production_selected=False,structural_release=False)

    def sample(self,full,public,energy_ev,alpha_i_deg,metal,*,pre=None,host_category=None,
               thickness_nm=None,qcm_rate=None,delta=None,n=256,seed=682120261013):
        if (isinstance(n,bool) or not isinstance(n,Integral) or n<2
                or isinstance(seed,bool) or not isinstance(seed,Integral) or seed<0):
            raise ValueError('Original draw count>=2 and nonnegative integer seed required')
        frame=frame_input(full,public,energy_ev,alpha_i_deg,metal,pre=pre,host_category=host_category,
            thickness_nm=thickness_nm,qcm_rate=qcm_rate)
        identity=_frame_identity(full,public);observed=ppc_features(public,estimate_stat_se=identity['photon_statistics_known'])
        x=prepare({k:np.stack([frame[k]]) for k in INPUT_KEYS},self.mode)
        encoded={k:torch.as_tensor(x[k],device=self.device) for k in ('vector','image') if x[k] is not None}
        delta_identity=None
        if delta is not None:
            branch=prepare_delta({k:np.asarray(delta[k])[None] for k in ('delta','delta_known','platform','platform_known')})
            encoded['delta']=torch.as_tensor(branch['vector'],device=self.device)
            delta_identity=dict(schema=DELTA_SPEC['schema'],
                input_arrays_sha256={k:_array_sha(np.asarray(delta[k])) for k in ('delta','delta_known','platform','platform_known')},
                prepared_vector_sha256=_array_sha(branch['vector']),
                known_points=int(branch['known'].sum()),reference_provenance=delta.get('summary'),
                common_reference_mask_not_assumed_equal_to_full_current=True)
        legal=legal_from_metadata(energy_ev,alpha_i_deg,self.templates,host_categories=[host_category],device=self.device)
        generator=torch.Generator(device=self.device).manual_seed(int(seed))
        sampled=self.net.sample_all_applicable(encoded,int(n),legal_templates=legal,generator=generator,host_categories=[host_category])
        results={}
        for scene in SCENARIO_NAMES:
            part=sampled['samples'][scene];applicable=bool(part['applicable'][0]);records=[]
            original={k:part[k][0].detach().cpu().numpy() for k in ('z','coarse','template','component')}
            if applicable:
                for i in range(int(n)):
                    try:
                        physical=decode(original['z'][i],int(original['coarse'][i]),int(original['template'][i]),
                            metal=metal,E=energy_ev,ai=alpha_i_deg,template_keys=self.keys,template_metadata=self.templates)
                        if physical['scenario']!=scene:raise ValueError('Original fixed-S sample changed scenario')
                        records.append(dict(index=i,decode_failed=False,physical=physical,conditional_labels=_labels(physical)))
                    except (ValueError,TypeError,KeyError,OverflowError,FloatingPointError) as error:
                        records.append(dict(index=i,decode_failed=True,physical=None,conditional_labels={},reason=str(error)))
            results[scene]=dict(applicable=applicable,not_applicable_reason=part['not_applicable_reason'][0],
                original_draws=len(records),original_joint=original,draws=records,
                conditional_quantities=_conditional_summaries(scene,records),
                public_quantities=dict(D_nm=None,diameter_nm=None,height_nm=None,gap_nm=None,coverage_fraction=None),
                missing_reason='Exploratory pilot; no independent calibration/OOD/PPC release receipt',
                calibrated=False,structural_release=False)
        probabilities=sampled['scenario_comparison']['probabilities'][0].detach().cpu().numpy()
        return dict(receipt=self.receipt,geometry=frame['geometry'],frame_identity=identity,
            observed=observed,features=_public_features(observed),scenarios=results,
            reference_scenario_probabilities=dict(zip(SCENARIO_NAMES,probabilities.tolist())),
            scenario_probabilities_are_reference_only=True,scenarios_pruned=False,
            original_seed=int(seed),original_draws_per_applicable_scenario=int(n),
            signed_delta_input_supplied=delta is not None,
            signed_delta_input_identity=delta_identity,
            conditioned_outside_D=False,nuisance_prior_resampled=False,
            exploratory=True,calibrated=False,structural_release=False)
