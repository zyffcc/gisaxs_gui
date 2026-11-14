'Research runtime for the source-bound single-pair minimal posterior.'
from numbers import Integral
from pathlib import Path

import numpy as np

from .scenario_chart import CATALOG, COORDINATES, DIM, SCENARIO_NAMES, decode
from .scenario_minimal_posterior import MinimalScenarioCandidate, DECLARATION as INPUT_SPEC, encode_pair
from .scenario_posterior import legal_from_metadata
from .scenario_runtime import (_sha, _source_sha, _json, _inside, _state_digest,
    _array_sha, _frame_identity, _labels, _conditional_summaries, _public_features)
from .v2_observation import CountFrame, ppc_features
from .v2_ppc import GATES, evaluate

TRAINING_COMMIT = '83d02cb55afcb692d6ab362e6e04594a13e47f7d'
SAMPLE_SEED = 681420261013
REPLICA_SEED = 672820261018


def _domain_summary(records):
    values = [r.get('conditional_labels',{}).get('domain_nm') for r in records]
    resolved = [float(v) for v in values if v is not None and np.isfinite(v) and v>0]
    item = dict(median=None,interval68=None,interval95=None,original_draws=len(records),
        resolved_original_draws=len(resolved),calibrated=False,structural_release=False,
        definition='intrinsic Teubner-Strey domain length; not metal centre D or island size')
    if len(records) and len(resolved)==len(records):
        q = np.quantile(resolved,[.025,.16,.5,.84,.975])
        item.update(median=float(q[2]),interval68=q[[1,3]].tolist(),interval95=q[[0,4]].tolist(),
                    reason='uncalibrated conditional research interval')
    else:item['reason'] = 'original draws missing; no retained-subset interval'
    return item


class MinimalScenarioRuntime:
    'Actual83d-selected minimal weights; four hypotheses remain research-only.'
    def __init__(self,checkpoint,*,training_package,producer_package,device='cpu'):
        import torch
        self.path = Path(checkpoint).resolve(); self.package = Path(__file__).resolve().parent.parent
        self.training_package = Path(training_package).resolve()
        self.producer_package = Path(producer_package).resolve()
        side = self.path.with_name(self.path.name+'.sha256')
        digest = side.read_text(encoding='ascii').split()[0]
        if _sha(self.path)!=digest:raise ValueError('Selected minimal checkpoint SHA differs')
        try:saved = torch.load(self.path,map_location='cpu',weights_only=True)
        except TypeError as error:raise RuntimeError('Safe weights_only checkpoint loading required') from error
        if _sha(self.path)!=digest:raise ValueError('Selected checkpoint changed during load')
        d = saved['declaration']; bank = d['bank']
        if (d.get('schema')!='four_scenario_minimal_training_1' or d.get('architecture')!='minimal_qwidth_joint'
                or d.get('fixture') is not False or d.get('TEST_inspected') is not False
                or d.get('input_spec')!=INPUT_SPEC or bank.get('fixture') is not False
                or bank.get('TEST_scientific_rows_decoded') is not False
                or d['dim']!=DIM or d['continuous_dim']!=DIM
                or d['catalog']!=list(CATALOG) or d['coordinates']!=list(COORDINATES)
                or d['coarse_classes']!=len(CATALOG)):
            raise ValueError('Actual source-bound44D minimal training declaration required')
        self.templates = tuple(d['template_metadata']); self.keys = tuple(d['template_keys'])
        if (self.keys!=tuple(t['key'] for t in self.templates) or len(set(self.keys))!=len(self.keys)
                or d['template_classes']!=len(self.keys)+1
                or any(t.get('split')!='TRAIN' for t in self.templates)):
            raise ValueError('Unique bound TRAIN template catalog required')
        build_path = self.training_package/'BUILD_INFO.json'; build = _json(build_path)
        if build.get('git_commit')!=TRAINING_COMMIT or build.get('git_dirty') is not False:
            raise ValueError('Exact clean original83d training source package required')
        for relative,value in d['source_pins'].items():
            if (_source_sha(_inside(self.training_package,relative))!=value
                    or _source_sha(_inside(self.package,relative))!=value):
                raise ValueError('Executing/original selected minimal source differs: '+relative)
        for relative,value in bank['source_pins'].items():
            if (_source_sha(_inside(self.producer_package,relative))!=value
                    or _source_sha(_inside(self.package,relative))!=value):
                raise ValueError('Executing/original generating source differs: '+relative)
        if (_state_digest(saved['model'])!=saved['state_sha256']
                or not isinstance(saved['epoch'],int) or isinstance(saved['epoch'],bool)
                or saved['epoch']<0 or not np.isfinite(saved['CAL_NLL'])):
            raise ValueError('Actual selected complete-epoch tensor state required')
        self.declaration = d; self.device = torch.device(device)
        if self.device.type not in ('cpu','cuda'):raise ValueError('CPU or CUDA device required')
        self.net = MinimalScenarioCandidate(DIM,len(CATALOG),len(self.keys)+1,
            components=d['components'],hidden=d['hidden'],catalog=d['catalog'],
            bank_scenario_prior=d['bank_scenario_prior']).to(self.device)
        self.net.load_state_dict(saved['model'],strict=True); self.net.eval()
        self.receipt = dict(schema='minimal_single_pair_research_runtime_1',weights_sha256=digest,
            state_sha256=saved['state_sha256'],selected_epoch=saved['epoch'],CAL_NLL=saved['CAL_NLL'],
            training_commit=TRAINING_COMMIT,training_BUILD_INFO_sha256=_sha(build_path),
            plan_sha256=bank['plan_sha256'],input_mode='minimal_single_pair',
            execution_source_pins={name:_source_sha(Path(__file__).parent/name) for name in (
                'scenario_minimal_runtime.py','scenario_minimal_posterior.py','scenario_posterior.py',
                'scenario_runtime.py','v2_ppc.py','v2_observation.py')},
            calibrated=False,OOD_qualified=False,structural_release=False)

    def sample(self,full,public,energy_ev,alpha_i_deg,metal,*,host_category=None,n=256,seed=SAMPLE_SEED):
        'Actual one-pair input; complete fixed-S joint draws without rejection.'
        import torch
        if (not isinstance(n,Integral) or isinstance(n,(bool,np.bool_)) or n<2
                or not isinstance(seed,Integral) or isinstance(seed,(bool,np.bool_)) or seed<0):
            raise ValueError('At least two original draws and nonnegative integer seed required')
        identity = _frame_identity(full,public)
        if (np.any(public.P>full.P) or np.any(public.Y>full.Y)
                or np.any((public.P==full.P)&(public.Y!=full.Y))):
            raise ValueError('Public observations must remain a subset of original full counts')
        observed = ppc_features(public,estimate_stat_se=identity['photon_statistics_known'])
        q = observed['values']['q_star']; w = observed['values']['FWHM']
        x = encode_pair(q,w,energy_ev,alpha_i_deg,metal)
        encoded = dict(vector=torch.as_tensor(x['vector'],device=self.device),image=None)
        legal = legal_from_metadata(energy_ev,alpha_i_deg,self.templates,
                                    host_categories=[host_category],device=self.device)
        generator = torch.Generator(device=self.device).manual_seed(int(seed))
        sampled = self.net.sample_all_applicable(encoded,int(n),legal_templates=legal,
            generator=generator,host_categories=[host_category])
        scenes = {}
        for scene in SCENARIO_NAMES:
            part = sampled['samples'][scene]; applicable = bool(part['applicable'][0])
            original = {k:part[k][0].detach().cpu().numpy() for k in ('z','coarse','template','component')}
            records = []
            if applicable:
                for i in range(int(n)):
                    try:
                        physical = decode(original['z'][i],int(original['coarse'][i]),int(original['template'][i]),
                            metal=metal,E=float(energy_ev),ai=float(alpha_i_deg),template_keys=self.keys,
                            template_metadata=self.templates)
                        if physical['scenario']!=scene:raise ValueError('Original fixed-S draw changed scenario')
                        labels = _labels(physical)
                        if scene=='S4':labels['domain_nm'] = float(physical['domain_nm'])
                        records.append(dict(index=i,decode_failed=False,physical=physical,conditional_labels=labels))
                    except (ValueError,OverflowError,FloatingPointError) as error:
                        records.append(dict(index=i,decode_failed=True,reason=str(error),physical=None,conditional_labels={}))
            summaries = _conditional_summaries(scene,records)
            if scene=='S4':summaries['domain_nm'] = _domain_summary(records)
            scenes[scene] = dict(applicable=applicable,not_applicable_reason=part['not_applicable_reason'][0],
                original_draws=len(records),original_joint=original,draws=records,
                conditional_quantities=summaries,
                public_quantities=dict(D_nm=None,diameter_nm=None,height_nm=None,gap_nm=None,coverage_fraction=None),
                missing_reason='No independent calibration/TEST and frame release receipt',
                calibrated=False,structural_release=False)
        probabilities = sampled['scenario_comparison']['probabilities'][0].detach().cpu().numpy()
        return dict(receipt=dict(self.receipt),frame_identity=identity,
            geometry=dict(energy_ev=float(energy_ev),alpha_i_deg=float(alpha_i_deg),metal=metal),
            host_category_input=host_category,input_vector_sha256=_array_sha(x['vector']),
            input_AF_band=observed['AF_band'],input_pair_available=bool(x['complete_feature_pair'][0]),
            observed=observed,features=_public_features(observed),scenarios=scenes,
            reference_scenario_probabilities=dict(zip(SCENARIO_NAMES,probabilities.tolist())),
            scenario_probabilities_are_reference_only=True,scenarios_pruned=False,
            original_seed=int(seed),original_draws_per_applicable_scenario=int(n),
            conditioned_outside_D=False,nuisance_prior_resampled=False,calibrated=False,structural_release=False)

    def posterior_predictive(self,sampled,full,public,*,bank_plan,feature_certificate,
                             host_shapes=None,host_metadata=None,harddisk=None,
                             backend='cpu',seed=REPLICA_SEED):
        'Original per-draw forward, profiled global fluence, one count replica.'
        from .scenario_forward import mean_forward,harddisk_table,QUICK
        if sampled['receipt']!=self.receipt or _frame_identity(full,public)!=sampled['frame_identity']:
            raise ValueError('Original minimal weights or full/public frame changed')
        if not isinstance(seed,Integral) or isinstance(seed,(bool,np.bool_)) or seed<0:
            raise ValueError('Nonnegative original replica seed required')
        if backend not in ('cpu','cuda'):raise ValueError('Explicit CPU/CUDA forward required')
        if not sampled['frame_identity']['photon_statistics_known']:
            return dict(status='not_ready_unknown_photon_statistics',features=sampled['features'],
                scenarios={s:dict(status='not_verified_unknown_photon_statistics',fastline_qwidth_pass=False,
                                   structural_release=False) for s in SCENARIO_NAMES},
                missing_reason='Photon-count units unestablished; no Poisson gain-fit PPC',
                calibrated=False,structural_release=False)
        plan_path = Path(bank_plan).resolve(); plan = _json(plan_path); bank = self.declaration['bank']
        pins = bank['source_pins']
        for relative,value in pins.items():
            if (_source_sha(_inside(self.producer_package,relative))!=value
                    or _source_sha(_inside(self.package,relative))!=value):
                raise ValueError('Original predictive generating source changed: '+relative)
        if (_sha(plan_path)!=bank['plan_sha256'] or plan['source_pins']!=pins
                or plan['catalog']!=self.declaration['catalog'] or plan['coordinates']!=self.declaration['coordinates']):
            raise ValueError('Original bank plan/chart/source binding differs')
        cert_path = Path(feature_certificate).resolve(); cert = _json(cert_path)
        if (_sha(cert_path)!=bank['certificate_sha256'] or cert.get('schema')!='simple_scenario_feature_certificate_1'
                or cert.get('passed') is not True or cert.get('TEST_decoded') is not False
                or cert.get('quick')!=QUICK or cert['quick']!=plan['quick_recipe']
                or any(cert.get('source_pins',{}).get(k)!=v for k,v in pins.items())):
            raise ValueError('Actual original source-bound QUICK certificate required')
        def bound(name,supplied):
            item = plan['bindings'][name]
            path = _inside(plan_path.parent,item['path']) if supplied is None else Path(supplied).resolve()
            if _sha(path)!=item['sha256']:raise ValueError('Original TRAIN predictive asset changed: '+name)
            return path
        hp = bound('host_shapes.npz',host_shapes); hm = bound('host_shapes.json',host_metadata)
        dp = bound('harddisk.npz',harddisk)
        if tuple(_json(hm)['rows'])!=self.templates:raise ValueError('Original normalized TRAIN template metadata differs')
        observed = sampled['observed']; band = observed['AF_band']; output = {}
        observed_total = float(public.Y.sum())
        with np.load(hp,allow_pickle=False) as hostarr,np.load(dp,allow_pickle=False) as diskarr:
            disk = harddisk_table(diskarr)
            for si,scene in enumerate(SCENARIO_NAMES):
                part = sampled['scenarios'][scene]
                if not part['applicable']:
                    output[scene] = dict(status='not_applicable',reason=part['not_applicable_reason'],
                        fastline_qwidth_pass=False,structural_release=False); continue
                replicas = []; joint = part['original_joint']
                for i,record in enumerate(part['draws']):
                    try:
                        if record['decode_failed']:raise ValueError('Original decoder failure: '+record['reason'])
                        physical = decode(joint['z'][i],int(joint['coarse'][i]),int(joint['template'][i]),
                            metal=sampled['geometry']['metal'],E=sampled['geometry']['energy_ev'],
                            ai=sampled['geometry']['alpha_i_deg'],template_keys=self.keys,template_metadata=self.templates)
                        if physical!=record['physical'] or physical['scenario']!=scene:
                            raise ValueError('Original full joint physical draw changed')
                        rate,_,receipt = mean_forward(physical,full.P,hostarr,recipe=QUICK,backend=backend,disk=disk)
                        raw_mu = np.asarray(rate,float)*public.P
                        if (not np.isfinite(raw_mu).all() or np.any(raw_mu<0)
                                or np.any(raw_mu[public.P==0]!=0) or float(raw_mu.sum())<=0):
                            raise ValueError('Original expected photon mean unsupported')
                        gain = observed_total/float(raw_mu.sum()); mu = raw_mu*gain
                        if not np.isfinite(gain) or gain<0 or not np.isfinite(mu).all():
                            raise ValueError('Poisson scalarMLE gain not finite nonnegative')
                        rng = np.random.default_rng(np.random.SeedSequence([int(seed),si,i]))
                        photons = rng.poisson(mu)
                        extracted = ppc_features(CountFrame(photons,public.P,True),AF_band=band,estimate_stat_se=False)
                        if band is None:
                            extracted['AF_band'] = None; extracted['support'] = 'unresolved'
                            for key in ('q_star','FWHM','highq_ratio','valley_ratio'):
                                extracted['values'][key] = None
                                extracted['feature_missing_reasons'][key] = 'no_observed_peak_band'
                        extracted.update(index=i,outside_range=None,fitted_scalar_gain=gain,
                            raw_expected_photons_sha256=_array_sha(raw_mu),expected_photons_sha256=_array_sha(mu),
                            replica_photons_sha256=_array_sha(photons),forward_receipt=receipt,
                            original_joint_modified=False,count_support_applications=1)
                    except (ValueError,OverflowError,FloatingPointError,RuntimeError,MemoryError,OSError) as error:
                        extracted = dict(index=i,AF_band=band,forward_failed=True,outside_range=None,
                            values={k:None for k in GATES},feature_missing_reasons={k:'original_draw_failure' for k in GATES},
                            failure_reason=type(error).__name__+': '+str(error),structural_release=False)
                    replicas.append(extracted)
                if len(replicas)!=part['original_draws']:raise ValueError('Original conditional draw denominator changed')
                stats = evaluate(observed,replicas,expected_draws=part['original_draws'])
                passed = bool(stats['features']['q_star']['passed'] and stats['features']['FWHM']['passed'])
                output[scene] = dict(status='research_profiled_gain_individual_draw_PPC_complete',draws=replicas,PPC=stats,
                    fastline_qwidth_pass=passed,fastline_feature_gates=['q_star','FWHM'],
                    warning_features={name:dict(stats['features'][name],fastline_gate=False)
                                      for name in ('highq_ratio','valley_ratio','Yoneda')},
                    calibrated=False,OOD_qualified=False,structural_release=False)
        if _frame_identity(full,public)!=sampled['frame_identity']:
            raise ValueError('Original observation changed during predictive calculation')
        return dict(status='minimal_four_scenario_profiled_gain_PPC_complete',scenarios=output,
            feature_certificate_sha256=_sha(cert_path),host_shapes_sha256=_sha(hp),harddisk_sha256=_sha(dp),
            backend=backend,original_replica_seed=int(seed),posterior_images_averaged=False,count_support_applications=1,
            gain_profile='sum(public.Y)/sum(original rate*public.P) per joint draw; plug-in Poisson scalarMLE',
            morphology_or_relative_background_changed=False,original_joint_modified=False,
            brightness_posterior_calibrated=False,fastline_feature_gates=['q_star','FWHM'],
            calibrated=False,OOD_qualified=False,structural_release=False)
