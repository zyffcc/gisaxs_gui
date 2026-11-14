'Actual51D exploratory TRAIN-library coverage and individual-draw PPC.'
from collections import OrderedDict
from contextlib import ExitStack
from numbers import Integral
from pathlib import Path
import time
import zipfile

import numpy as np

from .scenario_mainline_chart import (CATALOG,SCENARIO_NAMES,decode,
    template_compatibility)
from .scenario_mainline_forward import mean_forward
from .scenario_mainline_posterior import MainlineRuntime
from .scenario_mainline_training import MainlineBank,read,bound,generating_source_sha
from .scenario_forward import QUICK,harddisk_table
from .scenario_candidates import (_matching,build_candidates,
    observe_candidate_frame,_peak_kind)
from .scenario_fixed_af_observation import observe_candidate_frame_fixed_af
from .scenario_library_candidates import _gate,_match_score,_receipts,TrainLibraryCandidates
from .scenario_runtime import (_array_sha,_frame_identity,_labels,_sha,_source_sha)
from .stage5_partial import selected_array
from .v2_dataset import read_split
from .v2_observation import CountFrame
from .v2_ppc import GATES,evaluate

SCHEMA='exploratory_mainline51_TRAIN_predictive_1'
POLICY=dict(real_split='TRAIN',shortlist_per_scenario=32,
    geometry='same metal; energy within1pct; incidence within.01degree',
    support='actual public P exactly once',fluence='global scalar in library coverage; PPC caller may profile one total-photon scalar per original draw, without changing physical joint draw',
    feature_gates='existing noise-aware five-feature gates; known no-peak is not unknown',
    coverage='finite shortlist actual-support clean-mean matching, not full PPC',
    PPC='original51D fixed-S joint draws individually; no images averaged/no nuisance redraw',
    numerical='expanded-domain exploratory QUICK; numerical equivalence not yet certified',
    original_TRAIN_CAL_bank_receipts_inspected_for_binding=True,
    CAL_family_rows_used_for_retrieval=False,TEST_read=False,calibrated=False,structural_release=False)
SOURCE_FILES=('sputtergisaxs/scenario_mainline_ppc.py',
    'sputtergisaxs/scenario_mainline_training.py','sputtergisaxs/scenario_mainline_posterior.py',
    'sputtergisaxs/scenario_mainline_chart.py','sputtergisaxs/scenario_mainline_forward.py',
    'sputtergisaxs/scenario_candidates.py','sputtergisaxs/scenario_fixed_af_observation.py',
    'sputtergisaxs/scenario_library_candidates.py','sputtergisaxs/scenario_runtime.py',
    'sputtergisaxs/scenario_forward.py','sputtergisaxs/v2_observation.py',
    'sputtergisaxs/v2_ppc.py','sputtergisaxs/v2_dataset.py',
    'sputtergisaxs/stage5_partial.py','sputtergisaxs/prediction.py',
    'sputtergisaxs/features.py','sputtergisaxs/stage5_measurement.py')
INDEX_FIELDS=('features','mandatory','geometry','z','coarse','template','scenario','active','generation_failed')
ERRORS=(ValueError,TypeError,KeyError,IndexError,OverflowError,FloatingPointError,
        RuntimeError,OSError,MemoryError)


def source_pins():
    package=Path(__file__).resolve().parents[1]
    return {k:_source_sha(package/k) for k in SOURCE_FILES}


def _failure(index,band,reason):
    return dict(index=int(index),AF_band=band,forward_failed=True,outside_range=None,
        values={k:None for k in GATES},feature_missing_reasons={k:'original_draw_failure' for k in GATES},
        failure_reason=reason,structural_release=False)


def _ppc_feature(frame,band):

    out=observe_candidate_frame_fixed_af(frame,AF_band=0 if band is None else band,estimate_stat_se=False)
    if band is None:
        out.update(AF_band=None,support='unresolved')
        for k in ('q_star','FWHM','highq_ratio','valley_ratio'):
            out['values'][k]=None;out['feature_missing_reasons'][k]='no_observed_peak_band'
    elif out['support']!='bilateral':
        for k in ('FWHM','valley_ratio'):
            out['values'][k]=None;out['feature_missing_reasons'][k]='unresolved_or_missing_support'
        if out['support']!='one_side':
            out['values']['q_star']=None;out['feature_missing_reasons']['q_star']='unresolved_or_missing_support'
    return out


class MainlineCoverage:
    'Original51D TRAIN-only thin columns, lazy sealed means, fixed32/S budget.'
    def __init__(self,bank,*,producer_package=None):
        self.bank=bank if isinstance(bank,MainlineBank) else MainlineBank(bank,producer_package=producer_package)
        self.templates=self.bank.templates;self.keys=self.bank.keys;self.shortlist=32
        arrays={k:[] for k in ('family_id',)+INDEX_FIELDS};self.metadata={};self.locations={}
        for path,digest in self.bank.files:
            values,header=read_split(path,digest,'TRAIN',{k:self.bank.shapes[k] for k in INDEX_FIELDS})
            receipts=_receipts(path.parent/'run.json',set(map(int,values['family_id'])))
            good=~values['generation_failed'].astype(bool)
            for k in arrays:arrays[k].append(values[k][good])
            for j in np.flatnonzero(good):
                fid=int(values['family_id'][j]);raw=receipts[fid]['physical']
                if fid%10>=8 or raw is None:raise ValueError('Successful original TRAIN source required')
                E=float(raw['E']);ai=float(raw['ai'])
                if not np.allclose(values['geometry'][j],[E,ai],rtol=0,atol=2e-6):
                    raise ValueError('Original encoded/receipt geometry differs')
                self.metadata[fid]=dict(E=E,ai=ai,category=raw['category'],
                    donor_groups=list(raw['donor_groups']),noise_seed=raw['noise_seed'],
                    combination=raw['combination'])
                self.locations[fid]=(path,digest,int(header['selected_row_indices'][j]))
        self.arrays={k:np.concatenate(v) for k,v in arrays.items()}
        if len(self.arrays['family_id'])!=self.bank.successful_split_rows['TRAIN']:
            raise ValueError('Complete successful original TRAIN library required')
        self._means=OrderedDict();self._verified=set()
        self.receipt=dict(schema=SCHEMA+'_finite_library',bank=self.bank.receipt,
            successful_original_TRAIN_rows=len(self.arrays['family_id']),
            failed_original_TRAIN_rows=self.bank.split_rows['TRAIN']-len(self.arrays['family_id']),
            policy=POLICY,execution_source_pins=source_pins(),calibrated=False,structural_release=False)

    def _mean(self,fid):
        if fid in self._means:
            value=self._means.pop(fid);self._means[fid]=value;return value
        path,digest,index=self.locations[fid]
        if path not in self._verified:
            if _sha(path)!=digest:raise ValueError('Original mainline TRAIN mean shard changed')
            self._verified.add(path)
        with zipfile.ZipFile(path) as archive:
            ids=selected_array(archive,'family_id',[index]);rate=selected_array(archive,'mean_rate',[index])[0]
        if int(ids[0])!=fid or not np.isfinite(rate).all() or np.any(rate<0):
            raise ValueError('Original mainline TRAIN mean/identity invalid')
        value=(rate.astype(float),dict(family_id=fid,shard_sha256=digest,
            original_row_index=index,mean_rate_sha256=_array_sha(rate),saved_precision='float32'))
        self._means[fid]=value
        while len(self._means)>32:self._means.popitem(last=False)
        return value

    def query(self,full,public,E,ai,metal,*,host_category=None,exclude_groups=(),observed=None,
              max_per_scenario=3):
        begin=time.perf_counter();identity=_frame_identity(full,public)
        if isinstance(max_per_scenario,bool) or not isinstance(max_per_scenario,Integral) or not 1<=max_per_scenario<=3:
            raise ValueError('One to three original medoid representatives per scenario required')
        if metal not in ('Au','Ag') or not np.isfinite([E,ai]).all() or min(E,ai)<=0:
            raise ValueError('Actual positive mandatory geometry and Au/Ag required')
        observed=observe_candidate_frame(public,estimate_stat_se=identity['photon_statistics_known']) if observed is None else observed
        for key in GATES:_gate(observed,key)
        a=self.arrays;exclude=set(exclude_groups);geometry=a['geometry'].astype(float)
        legal=template_compatibility(E,ai,self.templates,host_category=host_category)
        allowed=((a['mandatory'][:,2]==(1 if metal=='Au' else 0))
            &(abs(geometry[:,0]/E-1)<=.01)&(abs(geometry[:,1]-ai)<=.01))
        allowed &=legal[a['coarse'].astype(int),a['template'].astype(int)]
        allowed &=np.array([not exclude.intersection(self.metadata[int(fid)]['donor_groups']) for fid in a['family_id']])
        sampled=dict(receipt=self.receipt,candidate_source='新51D原TRAIN库有限条件集合；未校准',
            frame_identity=identity,scenarios={},structural_release=False)
        predictive=dict(scenarios={},structural_release=False);reports={}
        for s,scene in enumerate(SCENARIO_NAMES):
            if host_category=='Si' and scene in ('S2','S3'):
                sampled['scenarios'][scene]=dict(applicable=False,not_applicable_reason='known Si excludes S2/S3',
                    original_draws=0,draws=[])
                reports[scene]=dict(applicable=False,original_shortlist=0,accepted_particles=0,ESS=0,
                    fallback=False,checks=[],structural_release=False)
                continue
            indices=np.flatnonzero(allowed&(a['scenario']==s))


            selected,ranking,stored=TrainLibraryCandidates._rank(self,indices,observed)
            draws=[];descriptors=[];checks=[]
            for ordinal,(i,stored_score) in enumerate(zip(selected,stored)):
                fid=int(a['family_id'][i]);meta=self.metadata[fid]
                draw=dict(index=ordinal,family_id=fid,decode_failed=False)
                try:
                    physical=decode(a['z'][i],int(a['coarse'][i]),int(a['template'][i]),metal=metal,
                        E=meta['E'],ai=meta['ai'],template_keys=self.keys,template_metadata=self.templates)
                    if physical['scenario']!=scene:raise ValueError('Original51D scenario changed')
                    draw.update(physical=physical,conditional_labels=_labels(physical),
                        candidate_source='原新51D TRAIN族'+str(fid)+'完整联合记录；主粒群结构含声明的额外群／背景条件')
                except ERRORS as error:
                    draw.update(decode_failed=True,physical=None,reason=type(error).__name__+': '+str(error))
                try:
                    if draw['decode_failed']:raise ValueError(draw['reason'])
                    rate,mean_source=self._mean(fid);raw_mu=rate*public.P
                    total=float(raw_mu.sum());photons=float(public.Y.sum())
                    if not np.isfinite(raw_mu).all() or total<=0 or photons<=0:raise ValueError('No supported positive mean/observation')
                    gain=photons/total;mu=gain*raw_mu
                    descriptor=observe_candidate_frame_fixed_af(CountFrame(mu,public.P,False),
                        AF_band=observed['AF_band'],estimate_stat_se=False)
                    descriptor.update(index=ordinal,family_id=fid,mean_source=mean_source,
                        fitted_scalar_gain=gain,count_support_applications=1,
                        gain_kind='Poisson_global_scalar_MLE' if identity['photon_statistics_known'] else 'shape_moment_scalar; not PoissonMLE',
                        raw_expected_photons_sha256=_array_sha(raw_mu),profiled_mean_sha256=_array_sha(mu),
                        original_geometry=[meta['E'],meta['ai']],query_geometry=[float(E),float(ai)],
                        geometry_tolerance_matched_not_recomputed=bool(meta['E']!=E or meta['ai']!=ai),
                        posterior_predictive_replica=False,original_joint_modified=False)
                    match=_matching(observed,descriptor);score=_match_score(match)
                except ERRORS as error:
                    descriptor=_failure(ordinal,observed.get('AF_band'),type(error).__name__+': '+str(error))
                    descriptor['family_id']=fid;match=_matching(observed,descriptor);score=None
                accepted=bool(not draw['decode_failed'] and match['matches_measured_features'] and score is not None and score<=1+1e-12)
                checks.append(dict(index=ordinal,family_id=fid,accepted=accepted,
                    decode_failed=draw['decode_failed'],actual_support_checked=not descriptor['forward_failed'],
                    maximum_normalized_error=score,stored_shortlist_error=float(stored_score),
                    modifiers=CATALOG[int(a['coarse'][i])]['modifiers'],matching=match,
                    failure_reason=descriptor.get('failure_reason',draw.get('reason'))))
                draws.append(draw);descriptors.append(descriptor)
            accepted=[c['index'] for c in checks if c['accepted']];fallback=not accepted
            keep=list(range(len(draws))) if fallback else accepted
            joint={k:np.array(a[k][selected[keep]],copy=True) for k in ('z','coarse','template')}
            ensemble=[dict(draws[j],index=n,original_shortlist_index=j) for n,j in enumerate(keep)]
            predictions=[dict(descriptors[j],index=n,original_shortlist_index=j) for n,j in enumerate(keep)]
            sampled['scenarios'][scene]=dict(applicable=True,original_draws=len(ensemble),draws=ensemble,original_joint=joint,
                candidate_source='有限接受集合等权近似；未校准' if accepted else '最近完整集合回退；无接受粒子',
                library_fallback=fallback)
            predictive['scenarios'][scene]=dict(draws=predictions,original_joint_sha256={k:_array_sha(v) for k,v in joint.items()})
            reports[scene]=dict(applicable=True,ranking=ranking,original_shortlist=len(draws),
                accepted_particles=len(accepted),ESS=len(accepted),fallback=fallback,
                accepted_family_ids=[checks[j]['family_id'] for j in accepted],
                ensemble_family_ids=[draws[j]['family_id'] for j in keep],checks=checks,
                selected_original_joint_sha256={k:_array_sha(v) for k,v in joint.items()},
                interval_calibration_status='finite conditional cluster intervals; not CAL/TEST validation',
                structural_release=False)
        candidates=build_candidates(sampled,observed,predictive,max_per_scenario=int(max_per_scenario))
        for scene,part in candidates['scenarios'].items():
            byindex={r['index']:r for r in sampled['scenarios'][scene]['draws']}
            for candidate in part['candidates']:
                candidate['representative_family_id']=byindex[candidate['representative_index']]['family_id']
        return dict(schema=SCHEMA+'_finite_coverage',candidates=candidates,acceptance=reports,
            sampled=sampled,predictive=predictive,
            observed=observed,frame_identity=identity,elapsed_s=time.perf_counter()-begin,
            binding=dict(receipt=self.receipt,geometry=dict(E=float(E),ai=float(ai),metal=metal,host_category=host_category),
                excluded_donor_groups=sorted(exclude),original_AF_band=observed.get('AF_band'),
                unknown_peak_is_not_failed_physics=True,mean_support_applications=1,
                full_PPC_performed=False,calibrated=False,structural_release=False))


class MainlinePPCRuntime(MainlineRuntime):
    'Actual51D runtime plus bound new51D per-original-draw QUICK forward.'
    def posterior_predictive(self,sampled,full,public,*,bank_plan,feature_certificate=None,
                             backend='cpu',seed=682220261013,profile_intensity=False):
        if feature_certificate is not None:raise ValueError('No old44D feature certificate applies to new51D physics')
        if sampled['receipt']!=self.receipt or _frame_identity(full,public)!=sampled['frame_identity']:
            raise ValueError('Original actual51D posterior/observation identity changed')
        if (isinstance(seed,bool) or not isinstance(seed,Integral) or seed<0 or backend not in ('cpu','cuda')
                or not isinstance(profile_intensity,bool)):
            raise ValueError('Original integer replica seed and CPU/CUDA backend required')
        if not sampled['frame_identity']['photon_statistics_known']:
            return dict(status='not_ready_unknown_photon_statistics',scenarios={s:dict(status='unverified_unknown_photon_statistics')
                for s in SCENARIO_NAMES},calibrated=False,structural_release=False)
        path=Path(bank_plan).resolve();plan=read(path);bank=self.declaration['bank']
        if (_sha(path)!=bank['plan_sha256'] or plan['catalog']!=self.declaration['catalog']
                or plan['coordinates']!=self.declaration['coordinates'] or plan['source_pins']!=bank['source_pins']):
            raise ValueError('Original actual51D generating plan differs')
        current=Path(__file__).resolve().parents[1]
        for relative,value in bank['source_pins'].items():
            if generating_source_sha(current/relative)!=value:raise ValueError('Executing original51D forward source changed: '+relative)
        hp=bound(path.parent,plan['bindings']['host_shapes.npz']);hm=bound(path.parent,plan['bindings']['host_shapes.json'])
        dp=bound(path.parent,plan['bindings']['harddisk.npz'])
        if tuple(read(hm)['rows'])!=self.templates:raise ValueError('Bound original TRAIN template metadata changed')
        observed=sampled['observed'];band=observed['AF_band'];output={}
        with np.load(hp,allow_pickle=False) as hosts,np.load(dp,allow_pickle=False) as disks:
            disk=harddisk_table(disks)
            for s,scene in enumerate(SCENARIO_NAMES):
                part=sampled['scenarios'][scene]
                if not part['applicable']:
                    output[scene]=dict(status='not_applicable',reason=part['not_applicable_reason']);continue
                original=part['original_joint'];records=part['draws'];replicas=[]
                if len(records)!=part['original_draws'] or [r['index'] for r in records]!=list(range(len(records))):
                    raise ValueError('Original joint draw denominator/order changed')
                for i,record in enumerate(records):
                    try:
                        if record['decode_failed']:raise ValueError(record['reason'])
                        row=decode(original['z'][i],int(original['coarse'][i]),int(original['template'][i]),
                            metal=sampled['geometry']['metal'],E=sampled['geometry']['energy_ev'],
                            ai=sampled['geometry']['alpha_i_deg'],template_keys=self.keys,template_metadata=self.templates)
                        if row!=record['physical'] or row['scenario']!=scene:raise ValueError('Original51D physical draw changed')
                        rate,_,forward=mean_forward(row,full.P,hosts,recipe=QUICK,backend=backend,disk=disk)
                        mu=np.asarray(rate,float)*public.P
                        if (not np.isfinite(mu).all() or np.any(mu<0) or np.any(mu[public.P==0]!=0)):
                            raise ValueError('Original51D unsupported expected mean')
                        raw_mu_sha=_array_sha(mu);gain=1.
                        if profile_intensity:
                            total=float(mu.sum())
                            if not np.isfinite(total) or total<=0:raise ValueError('Cannot profile a zero/nonfinite original supported mean')
                            gain=float(public.Y.sum())/total
                            mu=mu*gain
                        rng=np.random.default_rng(np.random.SeedSequence([int(seed),s,i]));Y=rng.poisson(mu)
                        desc=_ppc_feature(CountFrame(Y,public.P,True),band)
                        desc.update(index=i,outside_range=None,expected_photons_sha256=_array_sha(mu),
                            replica_photons_sha256=_array_sha(Y),mean_support_applications=1,
                            mainline_coarse=int(original['coarse'][i]),modifiers=row['modifiers'],
                            forward_receipt=forward,original_physical_sha256=None,structural_release=False,
                            profiled_intensity=profile_intensity,fitted_scalar_gain=gain,
                            unprofiled_expected_photons_sha256=raw_mu_sha,
                            joint_physical_record_modified=False)
                    except ERRORS as error:
                        desc=_failure(i,band,type(error).__name__+': '+str(error))
                    replicas.append(desc)
                output[scene]=dict(status='research_individual_draw_PPC_complete',draws=replicas,
                    original_joint_sha256={k:_array_sha(v) for k,v in original.items()},
                    PPC=evaluate(observed,replicas,expected_draws=part['original_draws']),
                    calibrated=False,OOD_qualified=False,structural_release=False)
        if _frame_identity(full,public)!=sampled['frame_identity']:raise ValueError('Original observation changed during PPC')
        return dict(schema=SCHEMA+'_original_joint_PPC',status='four_scenario_research_PPC_complete',scenarios=output,
            host_shapes_sha256=_sha(hp),harddisk_sha256=_sha(dp),original_replica_seed=int(seed),backend=backend,
            actual_joint_dim=51,old44D_decoder_used=False,execution_source_pins=source_pins(),
            posterior_images_averaged=False,count_support_applications=1,nuisance_resampled=False,
            profiled_intensity=profile_intensity,
            intensity_rule='one Poisson-MLE supported total-count scalar per original draw' if profile_intensity else 'original joint gain',
            numerical_certificate=False,expanded_numerical_equivalence='not yet certified',
            calibrated=False,OOD_qualified=False,structural_release=False)
