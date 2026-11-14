'Portable real four-scenario weights, predictive assets and quantity evidence.'
import argparse
from datetime import datetime, timezone
import copy
import hashlib
import json
from pathlib import Path
import shutil
import time

from .scenario_runtime import ScenarioRuntime, _frame_identity, _array_sha, _sha, _source_sha
from .scenario_qpreserve_runtime import load_runtime
from .scenario_forward import QUICK
from .scenario_credibility import ScenarioCredibility
from .scenario_release import (SCHEMA as RELEASE_SCHEMA, NOT_BEFORE,
    POLICY, joint_identity, pending_qualification, qualify_evidence, release_frame)

SCHEMA='four_scenario_portable_bundle_1'
CODE=Path(__file__).resolve().parent
EXECUTION_FILES=('scenario_bundle.py','scenario_release.py','scenario_credibility.py',
                 'scenario_runtime.py','scenario_posterior.py','v2_ppc.py',
                 'scenario_qpreserve_runtime.py','scenario_qpreserve.py','scenario_acceptance.py',
                 'scenario_optional_credibility.py','scenario_cached_cpu.py','scenario_cpu_certificate.py',
                 'scenario_cached_ppc.py')
TEST_ASSETS=('selection_manifest','new_TEST_plan','simulation_TEST','real_TEST')


def _json(path):return json.loads(Path(path).read_text(encoding='utf8'))


def _optional_identity(*,pre=None,host_category=None,thickness_nm=None,qcm_rate=None):
    'Record the actual first sampler inputs, separately from replay checks.'
    values=dict(host_category=host_category,
        thickness_nm=None if thickness_nm is None else float(thickness_nm),
        qcm_rate=None if qcm_rate is None else float(qcm_rate),
        pre=None if pre is None else dict(Y=_array_sha(pre.Y),P=_array_sha(pre.P),
            photon_counts_known=bool(pre.photon_counts_known),
            exposure_s=None if pre.exposure_s is None else float(pre.exposure_s)))
    digest=hashlib.sha256(json.dumps(values,sort_keys=True,separators=(',',':'),
        allow_nan=False).encode('utf8')).hexdigest()
    return dict(values=values,sha256=digest)


def _inside(root, relative):
    p=Path(relative)
    if p.is_absolute():raise ValueError('Bundle assets must use portable relative paths')
    root=Path(root).resolve();p=(root/p).resolve()
    if root not in p.parents:raise ValueError('Asset path escapes the portable bundle')
    return p


def _write(path, value):
    path=Path(path);temporary=path.with_name(path.name+'.partial')
    temporary.write_bytes((json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode('utf8'))
    temporary.replace(path)
    path.with_name(path.name+'.sha256').write_bytes((_sha(path)+'  '+path.name+'\n').encode('ascii'))


def _asset(root, source, relative):
    source=Path(source).resolve();target=_inside(root,relative);target.parent.mkdir(parents=True,exist_ok=True)
    digest=_sha(source);shutil.copyfile(source,target)
    if _sha(source)!=digest or _sha(target)!=digest:raise ValueError('Actual bundle source changed while copying')
    return dict(path=target.relative_to(root).as_posix(),sha256=digest,bytes=target.stat().st_size)


def build_bundle(out, checkpoint, *, ood_run=None, optional_ood_run=None, bank_plan=None, feature_certificate=None,
                 selection_manifest=None, new_test_plan=None, simulated_test=None, real_test=None,
                 package_root=None, cpu_cache_certificate=None):
    'Copy latest actual assets, preserving checkpoint and scientific identities.'
    test_paths=(selection_manifest,new_test_plan,simulated_test,real_test)
    if any(p is not None for p in test_paths):
        if datetime.now(timezone.utc)<NOT_BEFORE:raise RuntimeError('TEST evidence packaging is locked until 2026-10-18 Europe/Berlin')
        if any(p is None for p in test_paths):raise ValueError('All original TEST evidence bindings required; no partial qualification')
    out=Path(out).resolve()
    if out.exists():raise FileExistsError('Build a new actual bundle directory; do not overwrite unknown assets')
    runtime=load_runtime(checkpoint,package_root=package_root)
    out.mkdir(parents=True);assets={}
    assets['checkpoint']=_asset(out,runtime.path,'model/best.pt')
    side=runtime.path.with_name(runtime.path.name+'.sha256')
    shutil.copyfile(side,out/'model/best.pt.sha256')


    for relative,digest in runtime.pins.items():
        source=_inside(runtime.package,relative)
        if _source_sha(source)!=digest:raise ValueError('Actual original scientific source changed: '+relative)
        _asset(out,source,'scientific_source/'+relative)
    for name,digest in runtime.receipt['execution_source_pins'].items():
        source=CODE/name
        if _source_sha(source)!=digest:raise ValueError('Actual runtime execution source changed: '+name)
        _asset(out,source,'scientific_source/sputtergisaxs/'+name)
    if ood_run is not None:
        ood_path=Path(ood_run).resolve();ood_path=ood_path/'run.json' if ood_path.is_dir() else ood_path
        ood=_json(ood_path)


        for relative,digest in ood['identity']['source_pins'].items():
            choices=(_inside(runtime.package,relative),_inside(CODE.parent,relative))
            source=next((p for p in choices if p.is_file() and _source_sha(p)==digest),None)
            if source is None:raise ValueError('Original OOD diagnostic source unavailable: '+relative)
            _asset(out,source,'scientific_source/'+relative)


        runtime=load_runtime(checkpoint,package_root=out/'scientific_source')
        checker=ScenarioCredibility(runtime,ood_path)
        diagnostic=_inside(ood_path.parent,ood['diagnostic_file'])
        assets['OOD_run']=_asset(out,ood_path,'ood/run.json')
        assets['OOD_arrays']=_asset(out,diagnostic,'ood/'+ood['diagnostic_file'])
        if assets['OOD_arrays']['sha256']!=checker.ood_binding['diagnostic_sha256']:raise ValueError('Actual OOD artifact differs')
    if optional_ood_run is not None:
        from .scenario_optional_credibility import load_optional_ood
        optional=load_optional_ood(runtime,optional_ood_run);path=optional['path'];value=optional['value']
        for relative,digest in value['identity']['execution_source_pins'].items():
            choices=(_inside(runtime.package,relative),_inside(CODE.parent,relative))
            if relative.startswith('scripts/') and relative.endswith('.py'):
                choices=(_inside(path.parent,'scientific_source/'+relative),)+choices
            source=next((p for p in choices if p.is_file() and _source_sha(p)==digest),None)
            if source is None:raise ValueError('Actual optional OOD diagnostic source unavailable: '+relative)


            destination=('ood/optional/scientific_source/' if relative.startswith('scripts/')
                         else 'scientific_source/')+relative
            target=_inside(out,destination)
            if target.is_file() and _source_sha(target)!=digest:
                raise ValueError('Optional OOD source conflicts with original model: '+relative)
            _asset(out,source,destination)
        arrays=_inside(path.parent,value['diagnostic_file'])
        assets['optional_OOD_run']=_asset(out,path,'ood/optional/run.json')
        assets['optional_OOD_arrays']=_asset(out,arrays,'ood/optional/'+value['diagnostic_file'])
        if assets['optional_OOD_arrays']['sha256']!=optional['binding']['diagnostic_sha256']:
            raise ValueError('Actual optional OOD arrays differ')
    if bank_plan is not None:
        path=Path(bank_plan).resolve();plan=_json(path)
        if _sha(path)!=runtime.receipt['plan_sha256'] or plan['source_pins']!=runtime.declaration['bank']['source_pins']:
            raise ValueError('Original predictive generating plan required')
        assets['bank_plan']=_asset(out,path,'forward/plan.json')
        for name in ('host_shapes.npz','host_shapes.json','harddisk.npz'):
            binding=plan['bindings'][name];src=_inside(path.parent,binding['path'])
            if _sha(src)!=binding['sha256']:raise ValueError('Original TRAIN forward asset changed: '+name)
            assets[name]=_asset(out,src,'forward/'+binding['path'])
        if tuple(_json(_inside(out,assets['host_shapes.json']['path']))['rows'])!=runtime.templates:
            raise ValueError('Original TRAIN template catalog differs')
    if feature_certificate is not None:
        cert=Path(feature_certificate).resolve();value=_json(cert);bank=runtime.declaration['bank']
        if (_sha(cert)!=bank['certificate_sha256'] or value.get('passed') is not True
                or value.get('schema')!='simple_scenario_feature_certificate_1'
                or value.get('quick')!=QUICK
                or any(value.get('source_pins',{}).get(k)!=v for k,v in bank['source_pins'].items())):
            raise ValueError('Actual original numerical evidence required')
        assets['feature_certificate']=_asset(out,cert,'forward/certificate.json')
    if (bank_plan is None)!=(feature_certificate is None):raise ValueError('Predictive plan and numerical certificate must be supplied together')
    if cpu_cache_certificate is not None:
        if bank_plan is None:raise ValueError('CPU acceleration requires the original bound forward assets')
        from .scenario_cpu_certificate import cpu_certificate
        cpu_certificate(cpu_cache_certificate)
        assets['CPU_cache_certificate']=_asset(out,Path(cpu_cache_certificate).resolve(),'forward/cpu_cache_certificate.json')
    original=None
    if all(p is not None for p in test_paths):
        if bank_plan is None:raise ValueError('Actual TEST bundle requires its original predictive plan/certificate')
        for name,path in zip(TEST_ASSETS,test_paths):assets[name]=_asset(out,path,'evidence/'+name+'.json')
        selection=_json(_inside(out,assets['selection_manifest']['path']))
        plan=_json(_inside(out,assets['new_TEST_plan']['path']))
        simulated=_json(_inside(out,assets['simulation_TEST']['path']))
        real=_json(_inside(out,assets['real_TEST']['path']))
        original=_evidence(runtime,assets,selection,plan,simulated,real)
    qualification=original or pending_qualification()
    manifest=dict(schema=SCHEMA,status='actual_evidence_bound' if original else 'pending_independent_TEST',
        created_utc=datetime.now(timezone.utc).isoformat(),runtime_receipt=dict(runtime.receipt),
        model=dict(weights_sha256=runtime.receipt['weights_sha256'],state_sha256=runtime.receipt['state_sha256'],
                   input_mode=runtime.mode,selected_epoch=runtime.receipt['selected_epoch'],
                   bank_plan_sha256=runtime.receipt['plan_sha256']),assets=assets,
        original_scientific_source_pins=dict(runtime.pins),
        execution_source_pins={name:_source_sha(CODE/name) for name in EXECUTION_FILES},
        release_schema=RELEASE_SCHEMA,release_policy=copy.deepcopy(POLICY),
        qualification=qualification,models_selected_using_TEST=False,
        raw_or_bank_scientific_arrays_packaged=False,original_weights_relabelled=False,
        pending_does_not_confer_structure_release=True)
    _write(out/'bundle.json',manifest)
    return manifest


def _evidence(runtime,assets,selection,test_plan,simulated,real):
    'Actual SHA/seed/source links; no flag-only acceptance receipt.'
    models=[m for m in selection['models'] if m['sha256']==runtime.receipt['weights_sha256']
        and m['mode']==runtime.receipt['input_mode']]
    if len(models)!=1 or assets.get('OOD_run',{}).get('sha256')!=models[0].get('OOD_run',{}).get('sha256'):
        raise ValueError('Qualified bundle must retain the exact pre-TEST selected original OOD artifact')
    expected=assets['selection_manifest']['sha256']
    if (test_plan['selection_sha256']!=expected or simulated['identity']['selection_sha256']!=expected
            or real['selection_sha256']!=expected
            or test_plan['full_bank_plan_sha256']!=assets['bank_plan']['sha256']
            or simulated['identity']['TEST_plan_sha256']!=assets['new_TEST_plan']['sha256']
            or selection['full_bank_plan_sha256']!=assets['bank_plan']['sha256']
            or selection['feature_certificate']['sha256']!=assets['feature_certificate']['sha256']):
        raise ValueError('Original selection/TEST/plan/numerical identities differ')
    if (test_plan['source_pins'].get('scripts/v2_scenario_new_test.py')!=simulated['identity']['source_sha256']
            or real['identity']['certificate_sha256']!=assets['feature_certificate']['sha256']):
        raise ValueError('Actual original TEST execution/numerical bindings differ')
    for relative,digest in selection['source_pins'].items():
        if _source_sha(_inside(CODE.parent,relative))!=digest:raise ValueError('Original frozen TEST executable changed: '+relative)
    qualification=qualify_evidence(runtime.receipt,selection,test_plan,simulated,real)
    qualification['evidence_binding'].update({name:assets[name]['sha256'] for name in TEST_ASSETS})
    return qualification


class ScenarioBundle:
    'One actual model instance and bound metadata, reused over all frames.'
    def __init__(self,bundle,*,device='cpu',package_root=None):
        self.root=Path(bundle).resolve();path=self.root/'bundle.json';digest=_sha(path)
        if path.with_name(path.name+'.sha256').read_text(encoding='ascii').split()[0]!=digest:
            raise ValueError('Portable bundle manifest SHA differs')
        value=_json(path)
        if (value.get('schema')!=SCHEMA or value.get('release_schema')!=RELEASE_SCHEMA
                or value.get('release_policy')!=POLICY or value.get('models_selected_using_TEST') is not False
                or value.get('original_weights_relabelled') is not False):
            raise ValueError('Actual source-bound four-scenario bundle required')
        if any(name in value['assets'] for name in TEST_ASSETS) and datetime.now(timezone.utc)<NOT_BEFORE:
            raise RuntimeError('TEST evidence bundle is locked until 2026-10-18 Europe/Berlin')
        self.assets={}
        for name,item in value['assets'].items():
            asset=_inside(self.root,item['path'])
            if _sha(asset)!=item['sha256'] or asset.stat().st_size!=item['bytes']:
                raise ValueError('Portable actual bundle asset changed: '+name)
            self.assets[name]=asset
        for name,digest_value in value['execution_source_pins'].items():
            if name not in EXECUTION_FILES or _source_sha(CODE/name)!=digest_value:
                raise ValueError('Actual bundle execution source changed: '+name)
        if set(value['execution_source_pins'])!=set(EXECUTION_FILES):raise ValueError('Complete deployment source pins required')
        source_package=self.root/'scientific_source' if package_root is None else Path(package_root)
        self.runtime=load_runtime(self.assets['checkpoint'],device=device,package_root=source_package)
        if self.runtime.receipt!=value['runtime_receipt'] or self.runtime.pins!=value['original_scientific_source_pins']:
            raise ValueError('Actual bundle checkpoint/source/runtime differs')
        self.credibility=ScenarioCredibility(self.runtime,self.assets.get('OOD_run'),
            optional_ood_run=self.assets.get('optional_OOD_run'))
        tests=[self.assets.get(name) for name in TEST_ASSETS]
        if all(p is None for p in tests):qualification=pending_qualification()
        elif any(p is None for p in tests):raise ValueError('Incomplete original TEST evidence binding')
        else:
            dictionaries=[_json(p) for p in tests]
            qualification=_evidence(self.runtime,value['assets'],*dictionaries)
        if qualification!=value['qualification']:raise ValueError('Actual quantity evidence differs; no copied pass flags')
        self.qualification=qualification;self.manifest=value;self.bank_plan=self.assets.get('bank_plan')
        self.feature_certificate=self.assets.get('feature_certificate')
        self.cpu_cache_certificate=self.assets.get('CPU_cache_certificate')
        if self.cpu_cache_certificate is not None:
            from .scenario_cpu_certificate import cpu_certificate
            self.cpu_acceleration=cpu_certificate(self.cpu_cache_certificate)
        else:self.cpu_acceleration=None
        self.receipt=dict(schema=SCHEMA,bundle_sha256=_sha(path),runtime_receipt=dict(self.runtime.receipt),
            asset_sha256={k:v['sha256'] for k,v in value['assets'].items()},
            execution_source_pins=dict(value['execution_source_pins']),qualification_status=qualification['status'])

    def sample(self,*args,**kwargs):
        context=_optional_identity(**{k:kwargs.get(k) for k in
            ('pre','host_category','thickness_nm','qcm_rate')})
        sampled=self.runtime.sample(*args,**kwargs)
        sampled['bundle_original_optional_identity']=dict(context,
            runtime_receipt=copy.deepcopy(sampled['receipt']),
            joint_sha256=joint_identity(sampled),original_seed=sampled['original_seed'])
        return sampled

    def posterior_predictive(self,sampled,full,public,*,backend='cpu',seed=673220261018):
        if self.bank_plan is None or self.feature_certificate is None:
            return dict(status='not_ready_bound_forward_assets_missing',scenarios={},structural_release=False)


        from .scenario_cached_ppc import posterior_predictive_cached
        result=posterior_predictive_cached(self.runtime,sampled,full,public,bank_plan=self.bank_plan,
            feature_certificate=self.feature_certificate,backend=backend,seed=seed,
            cpu_cache_certificate=self.cpu_cache_certificate)
        result['bundle_PPC_binding']=dict(runtime_receipt=sampled['receipt'],frame_identity=sampled['frame_identity'],
            joint_sha256=joint_identity(sampled),original_seed=sampled['original_seed'],
            original_draws_per_applicable_scenario=sampled['original_draws_per_applicable_scenario'])
        return result

    def assess(self,full,public,sampled,ppc=None,*,pre=None,host_category=None,thickness_nm=None,qcm_rate=None):
        'No draw replacement; caller retains original sampled research pool.'
        begin=time.perf_counter();cpu=time.process_time()
        original_context=sampled.get('bundle_original_optional_identity')
        if original_context is not None:
            expected=dict(_optional_identity(pre=pre,host_category=host_category,
                thickness_nm=thickness_nm,qcm_rate=qcm_rate),
                runtime_receipt=sampled['receipt'],joint_sha256=joint_identity(sampled),
                original_seed=sampled['original_seed'])
            if original_context!=expected:raise ValueError('Original bundle optional inputs or joint pool changed')
        credibility=self.credibility.assess(full,public,sampled,pre=pre,host_category=host_category,
                                            thickness_nm=thickness_nm,qcm_rate=qcm_rate)
        credibility['original_optional_input_identity_independently_bound']=original_context is not None
        if original_context is not None:
            credibility['original_optional_input_identity']=copy.deepcopy(original_context)
            credibility['context_verification_scope']='first sampler input identity plus original-pool replay'
        released=release_frame(sampled,ppc,credibility,self.qualification,host_category=host_category)
        return dict(bundle_receipt=copy.deepcopy(self.receipt),credibility=credibility,release=released,
            qualification=copy.deepcopy(self.qualification),cpu_seconds=time.process_time()-cpu,
            wall_seconds=time.perf_counter()-begin,original_scientific_samples_replaced=False)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--ood-run',type=Path);p.add_argument('--optional-ood-run',type=Path)
    p.add_argument('--bank-plan',type=Path);p.add_argument('--feature-certificate',type=Path)
    p.add_argument('--cpu-cache-certificate',type=Path)
    p.add_argument('--selection-manifest',type=Path);p.add_argument('--new-test-plan',type=Path)
    p.add_argument('--simulated-test',type=Path);p.add_argument('--real-test',type=Path)
    p.add_argument('--package-root',type=Path)
    a=p.parse_args();m=build_bundle(a.out,a.checkpoint,ood_run=a.ood_run,optional_ood_run=a.optional_ood_run,bank_plan=a.bank_plan,
        cpu_cache_certificate=a.cpu_cache_certificate,
        feature_certificate=a.feature_certificate,selection_manifest=a.selection_manifest,
        new_test_plan=a.new_test_plan,simulated_test=a.simulated_test,real_test=a.real_test,package_root=a.package_root)
    print('ACTUAL_BUNDLE_CREATED',m['status'],m['model']['input_mode'],flush=True)


if __name__=='__main__':main()
