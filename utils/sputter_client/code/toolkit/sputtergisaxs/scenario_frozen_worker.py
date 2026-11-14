'Isolated execution of an unchanged manifest-bound training runtime.'
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback

SCHEMA='manifest_bound_original_runtime_process_bridge_1'
if __name__!='__main__':


    from .scenario_minimal51 import Minimal51Runtime as _RuntimeBase
else:_RuntimeBase=object


def _require(condition,message):
    if not condition:raise ValueError(message)


def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _bytes(value):
    return (json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n').encode('utf8')


def _encode(value):
    import numpy as np
    if isinstance(value,np.ndarray):
        _require(not value.dtype.hasobject,'Object arrays are forbidden in runtime transport')
        a=np.ascontiguousarray(value)
        return {'__sgx_array__':True,'dtype':a.dtype.str,'shape':list(a.shape),
            'bytes':base64.b64encode(a.tobytes()).decode('ascii')}
    if isinstance(value,np.generic):return _encode(value.item())
    if isinstance(value,complex):return {'__sgx_complex__':[value.real,value.imag]}
    if isinstance(value,tuple):return {'__sgx_tuple__':[_encode(v) for v in value]}
    if isinstance(value,list):return [_encode(v) for v in value]
    if isinstance(value,dict):return {str(k):_encode(v) for k,v in value.items()}
    return value


def _decode(value):
    import numpy as np
    if isinstance(value,list):return [_decode(v) for v in value]
    if not isinstance(value,dict):return value
    if value.get('__sgx_array__') is True:
        dtype=np.dtype(value['dtype']);shape=tuple(value['shape'])
        _require(not dtype.hasobject and all(type(n) is int and n>=0 for n in shape),
            'Numeric typed arrays required')
        raw=base64.b64decode(value['bytes'],validate=True)
        _require(len(raw)==int(np.prod(shape,dtype=np.int64))*dtype.itemsize,'Transport array size differs')
        return np.frombuffer(raw,dtype=dtype).reshape(shape).copy()
    if set(value)=={'__sgx_complex__'}:return complex(*value['__sgx_complex__'])
    if set(value)=={'__sgx_tuple__'}:return tuple(_decode(v) for v in value['__sgx_tuple__'])
    return {k:_decode(v) for k,v in value.items()}


def _digest(value):return hashlib.sha256(_bytes(_encode(value))).hexdigest()


def _inside(root,relative):
    root=Path(root).resolve();path=(root/relative).resolve()
    _require(root in path.parents,'Bound runtime member escapes its package: '+relative)
    return path


def _manifest(path):
    path=Path(path).resolve();raw=path.read_bytes();value=json.loads(raw)
    _require(value.get('schema')=='conditional_mainline_preview_model_1',
        'An actual selected portable preview manifest is required')
    _require(value.get('architecture')=='mainline51_minimal_qwidth_joint_v1'
        and value.get('mode')=='features','This bridge executes the original minimal51 architecture only')
    for relative,item in value['files'].items():
        p=_inside(path.parent,relative)
        _require(p.is_file() and p.stat().st_size==item['bytes'] and _sha(p)==item['sha256'],
            'Original manifest asset differs: '+relative)
    bindings={k:_inside(path.parent,value[k]) for k in ('checkpoint','training_package','bank_plan')}
    _require(value.get('feature_certificate') is None,'No old44D certificate applies to this runtime')
    return value,bindings,hashlib.sha256(raw).hexdigest()


def _execution_tree(package,manifest_path,work_dir):
    'Materialize original pinned bytes, without editing either source tree.'
    package=Path(package).resolve();manifest_path=Path(manifest_path).resolve()
    manifest,bindings,manifest_sha=_manifest(manifest_path)
    original=bindings['training_package'];members={}
    for folder in ('sputtergisaxs','scripts'):
        for path in sorted((package/folder).rglob('*.py')):
            if '__pycache__' not in path.parts:
                relative=path.relative_to(package).as_posix()
                members[relative]=(path,'unchanged_current_dependency')
    for path in sorted((package/'assets').rglob('*')):
        if path.is_file() and '__pycache__' not in path.parts:
            members[path.relative_to(package).as_posix()]=(path,'unchanged_current_asset')
    for relative in manifest['files']:
        prefix=manifest['training_package'].rstrip('/')+'/'
        if relative.startswith(prefix):
            name=relative[len(prefix):]
            members[name]=(_inside(original,name),'original_manifest_training_source')
    _require('sputtergisaxs/__init__.py' in members,'Complete caller runtime package required')
    records={name:dict(sha256=_sha(path),bytes=path.stat().st_size,origin=origin)
        for name,(path,origin) in sorted(members.items())}
    identity=dict(schema=SCHEMA+'_execution',manifest_sha256=manifest_sha,
        bridge_source_sha256=_sha(__file__),files=records,
        policy='Original manifest source bytes take precedence; all source guards execute unchanged',
        model_or_prior_modified=False)
    identity_sha=hashlib.sha256(_bytes(identity)).hexdigest()
    root=Path(work_dir).resolve()/('original_runtime_'+identity_sha[:20])
    _require(root!=package and not any(root==p or p in root.parents for p in (
        package/'sputtergisaxs',package/'scripts',package/'assets',manifest_path.parent)),
        'Place derived runtime work outside immutable source/asset/release subtrees')
    root.mkdir(parents=True,exist_ok=True)
    for name,(source,_) in sorted(members.items()):
        target=_inside(root,name);record=records[name]
        if target.exists():_require(_sha(target)==record['sha256'],'Derived runtime source changed: '+name)
        else:
            target.parent.mkdir(parents=True,exist_ok=True);raw=source.read_bytes()
            _require(hashlib.sha256(raw).hexdigest()==record['sha256'],'Runtime dependency changed while copying')
            temporary=target.with_name(target.name+'.partial.'+str(os.getpid()))
            temporary.write_bytes(raw);temporary.replace(target)
    target=root/'execution.json';blob=_bytes(identity)
    if target.exists():_require(target.read_bytes()==blob,'Derived runtime identity changed')
    else:
        temporary=target.with_name(target.name+'.partial.'+str(os.getpid()))
        temporary.write_bytes(blob);temporary.replace(target)
    side=root/'execution.json.sha256'
    if side.exists():_require(side.read_text().split()[0]==identity_sha,'Derived runtime seal changed')
    else:
        temporary=side.with_name(side.name+'.partial.'+str(os.getpid()))
        temporary.write_bytes((identity_sha+'\n').encode('ascii'));temporary.replace(side)
    return root,identity_sha,manifest_sha


class FrozenManifestRuntime(_RuntimeBase):
    'Persistent CPU child with unchanged strict loading, sample and preview.'
    def __init__(self,manifest,*,package_root,work_dir,device='cpu',preview_on_sample=True):
        _require(device=='cpu','The frozen runtime process bridge is CPU only')
        _require(type(preview_on_sample) is bool,'preview_on_sample requires an explicit boolean')
        self.preview_on_sample=preview_on_sample
        self.manifest_path=Path(manifest).resolve()
        root,execution_sha,manifest_sha=_execution_tree(package_root,self.manifest_path,work_dir)
        self._lock=threading.Lock();self._closed=False;self._last=None
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',
            OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        self._process=subprocess.Popen([sys.executable,'-B',str(Path(__file__).resolve()),
            '--worker','--execution-root',str(root),'--execution-sha256',execution_sha,
            '--manifest',str(self.manifest_path),'--manifest-sha256',manifest_sha],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=None,env=env,
            text=True,encoding='utf8',bufsize=1)
        try:initial=self._receive()
        except Exception:
            self.close();raise
        self.receipt=initial['runtime_receipt'];self.declaration=initial['declaration']
        self.mode=initial['mode'];self.keys=tuple(initial['keys']);self.templates=tuple(initial['templates'])
        self.bridge_receipt=initial['bridge_receipt'];self.bridge_receipt['worker_pid']=self._process.pid
        self.bridge_receipt['preview_on_sample']=self.preview_on_sample
        self.bridge_receipt['preview_execution']='lazy explicit preview call; no forward during sample'
        self.child_timings={}
        _,bindings,_=_manifest(self.manifest_path)
        self.path=bindings['checkpoint'];self.package=bindings['training_package']
        self.pins=dict(self.declaration['source_pins'])
        import torch
        self.device=torch.device('cpu')

    def _receive(self):
        line=self._process.stdout.readline()
        _require(bool(line),'Original runtime worker exited before returning a result')
        result=json.loads(line)
        if result.get('ok') is not True:
            raise RuntimeError('Original runtime worker: '+str(result.get('error')))
        return _decode(result['value'])

    def _request(self,value):
        with self._lock:
            _require(not self._closed and self._process.poll() is None,'Original runtime worker is closed')
            self._process.stdin.write(_bytes(_encode(value)).decode('utf8'));self._process.stdin.flush()
            return self._receive()

    def sample(self,full,public,energy_ev,alpha_i_deg,metal,*,host_category=None,
            pre=None,thickness_nm=None,qcm_rate=None,delta=None,n=128,seed=673120261018):
        def frame(x):return dict(Y=x.Y,P=x.P,photon_counts_known=x.photon_counts_known,exposure_s=x.exposure_s)
        self._last=None
        result=self._request(dict(command='sample',full=frame(full),public=frame(public),
            energy_ev=energy_ev,alpha_i_deg=alpha_i_deg,metal=metal,host_category=host_category,n=n,seed=seed,
            pre=frame(pre) if pre is not None else None,thickness_nm=thickness_nm,qcm_rate=qcm_rate,delta=delta,
            preview_on_sample=self.preview_on_sample))
        self._last=result
        self.child_timings=dict(result['child_timings'])
        self.bridge_receipt['last_child_call_timings']=dict(self.child_timings)
        return result['sampled']

    def preview_medoid_features(self,sampled,candidates,full,public,*,bank_plan,
            feature_certificate=None,backend='cpu',observed=None,**kwargs):
        from .scenario_runtime import _frame_identity
        _require(not kwargs and backend=='cpu' and feature_certificate is None,
            'Only the existing original CPU preview bindings are supported')
        _require(self.preview_on_sample,
            'Representative preview was disabled by the caller; no preview forward was executed')
        _require(self._last is not None and _digest(sampled)==self._last['sampled_sha256'],
            'Preview must use the same unmodified original sampled pool')
        _require(_frame_identity(full,public)==self._last['frame_identity']
            and _sha(bank_plan)==self._last['bank_plan_sha256'],'Preview observation or bank differs')
        _require(observed is None or _digest(observed)==self._last['observed_sha256'],
            'Preview must use the original measured comparison descriptor')
        indices={s:[c['original_draw_index'] for c in group['candidates']]
            for s,group in candidates['scenarios'].items()}
        _require(indices==self._last['original_medoid_indices'],
            'Current geometry clustering changed original medoid indices')
        result=self._request(dict(command='preview_last',sampled_sha256=self._last['sampled_sha256'],
            frame_identity=self._last['frame_identity'],bank_plan_sha256=self._last['bank_plan_sha256'],
            observed_sha256=self._last['observed_sha256'],original_medoid_indices=indices))
        _require(result['sampled_sha256']==self._last['sampled_sha256']
            and result['frame_identity']==self._last['frame_identity'],
            'Original lazy preview returned another sample or observation')
        self.child_timings.update(result['child_timings'])
        self.bridge_receipt['last_child_call_timings']=dict(self.child_timings)
        self.bridge_receipt['last_preview_cache_reused']=result['preview_cache_reused']
        return result['preview']

    def posterior_predictive(self,sampled,full,public,*,bank_plan,feature_certificate=None,
            backend='cpu',seed=673220261018,**kwargs):
        'Explicit public PPC only: original method, arguments and seed in child.'
        _require(backend=='cpu' and feature_certificate is None,
            'Original CPU51D PPC accepts no old44D certificate')
        def frame(x):return dict(Y=x.Y,P=x.P,photon_counts_known=x.photon_counts_known,exposure_s=x.exposure_s)
        result=self._request(dict(command='posterior_predictive',sampled=sampled,
            full=frame(full),public=frame(public),bank_plan=str(Path(bank_plan).resolve()),
            feature_certificate=None,backend=backend,seed=seed,kwargs=kwargs))
        self.bridge_receipt['explicit_public_PPC_calls']=self.bridge_receipt.get('explicit_public_PPC_calls',0)+1
        return result

    def close(self):
        if getattr(self,'_closed',True):return
        self._closed=True;process=getattr(self,'_process',None)
        if process is not None:
            try:
                if process.stdin:process.stdin.close()
                process.wait(timeout=10)
            except (OSError,subprocess.TimeoutExpired):
                process.terminate()
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:process.kill();process.wait()

    def __enter__(self):return self
    def __exit__(self,*args):self.close()


def _worker(args):
    root=args.execution_root.resolve();identity=json.loads((root/'execution.json').read_bytes())
    _require(_sha(root/'execution.json')==args.execution_sha256
        and (root/'execution.json.sha256').read_text().split()[0]==args.execution_sha256
        and identity['bridge_source_sha256']==_sha(__file__),'Original runtime execution seal differs')
    for relative,item in identity['files'].items():
        path=_inside(root,relative)
        _require(path.stat().st_size==item['bytes'] and _sha(path)==item['sha256'],
            'Original runtime execution source differs: '+relative)
    manifest,bindings,manifest_sha=_manifest(args.manifest)
    _require(manifest_sha==args.manifest_sha256==identity['manifest_sha256'],'Original manifest changed')
    _require(not any(k=='sputtergisaxs' or k.startswith('sputtergisaxs.') for k in sys.modules),
        'Original runtime requires a fresh isolated process')
    sys.path[:0]=[str(root),str(root/'scripts')]
    import torch
    torch.set_num_threads(1)
    from sputtergisaxs.scenario_minimal51 import Minimal51Runtime
    from sputtergisaxs.scenario_candidates import build_candidates,preview_medoid_features
    from sputtergisaxs.scenario_runtime import _frame_identity
    from sputtergisaxs.v2_observation import CountFrame
    runtime=Minimal51Runtime(bindings['checkpoint'],device='cpu',package_root=bindings['training_package'])
    _require(runtime.receipt['weights_sha256']==manifest['files'][manifest['checkpoint']]['sha256'],
        'An unrequested model was loaded')
    bridge=dict(schema=SCHEMA,execution_sha256=args.execution_sha256,manifest_sha256=manifest_sha,
        bridge_source_sha256=_sha(__file__),original_source_guards_executed=True,
        original_runtime_receipt_sha256=_digest(runtime.receipt),separate_process=True,
        original_pinned_source_overrides=[k for k,v in identity['files'].items()
            if v['origin']=='original_manifest_training_source'],
        current_frontend_not_executed_as_old_source=True,raw_opened=False,TEST_read=False,
        posterior_predictive_calls_at_construction=0)
    def respond(value):
        sys.stdout.write(_bytes(dict(ok=True,value=_encode(value))).decode('utf8'));sys.stdout.flush()
    respond(dict(runtime_receipt=runtime.receipt,declaration=runtime.declaration,mode=runtime.mode,
        keys=runtime.keys,templates=runtime.templates,bridge_receipt=bridge))
    child_last=None
    def timed(function,*positional,**keywords):
        begin=time.perf_counter();cpu=time.process_time();value=function(*positional,**keywords)
        return value,dict(wall_seconds=time.perf_counter()-begin,CPU_seconds=time.process_time()-cpu)
    for line in sys.stdin:
        try:
            request=_decode(json.loads(line))
            if request.get('command')=='preview_last':
                _require(child_last is not None and child_last['preview_enabled'],
                    'No original sampled pool with a permitted lazy preview exists')
                for key in ('sampled_sha256','frame_identity','bank_plan_sha256',
                        'observed_sha256','original_medoid_indices'):
                    _require(request.get(key)==child_last[key],'Original lazy preview identity differs: '+key)
                _require(_digest(child_last['sampled'])==child_last['sampled_sha256']
                    and _frame_identity(child_last['full'],child_last['public'])==child_last['frame_identity']
                    and _sha(bindings['bank_plan'])==child_last['bank_plan_sha256'],
                    'Stored original pool, frame or bank changed before lazy preview')
                reused=child_last['preview'] is not None
                if not reused:
                    child_last['preview'],child_last['preview_timing']=timed(preview_medoid_features,
                        child_last['sampled'],child_last['initial'],child_last['full'],child_last['public'],
                        runtime=runtime,bank_plan=bindings['bank_plan'],feature_certificate=None,
                        backend='cpu',observed=child_last['observed'])
                _require(_digest(child_last['sampled'])==child_last['sampled_sha256'],
                    'Original lazy preview modified the joint sample')
                preview=child_last['preview']
                total=sum(len(group['draws']) for group in preview['scenarios'].values())
                failed=sum(d.get('forward_failed',False) for group in preview['scenarios'].values()
                    for d in group['draws'])
                respond(dict(preview=preview,sampled_sha256=child_last['sampled_sha256'],
                    frame_identity=child_last['frame_identity'],preview_cache_reused=reused,
                    child_timings=dict(original_representative_preview=child_last['preview_timing'],
                        this_preview_compute=dict(wall_seconds=0.,CPU_seconds=0.)
                            if reused else child_last['preview_timing']),
                    representative_preview_total=total,representative_preview_failed=failed,
                    all_representative_previews_failed=bool(total and total==failed)))
                continue
            def frame(x):return CountFrame(x['Y'],x['P'],x['photon_counts_known'],x['exposure_s'])
            full,public=frame(request['full']),frame(request['public'])
            if request.get('command')=='posterior_predictive':
                _require(_sha(request['bank_plan'])==_sha(bindings['bank_plan'])
                    and request['feature_certificate'] is None and request['backend']=='cpu',
                    'Original requested PPC bindings differ')
                respond(runtime.posterior_predictive(request['sampled'],full,public,
                    bank_plan=bindings['bank_plan'],feature_certificate=None,backend='cpu',
                    seed=request['seed'],**request['kwargs']))
                continue
            _require(request.get('command')=='sample','Unknown original runtime command')
            _require(type(request.get('preview_on_sample')) is bool,
                'Each original runtime request must declare preview permission')
            child_last=None
            sampled,sample_timing=timed(runtime.sample,full,public,
                request['energy_ev'],request['alpha_i_deg'],request['metal'],
                host_category=request['host_category'],n=request['n'],seed=request['seed'],
                pre=frame(request['pre']) if request.get('pre') is not None else None,
                thickness_nm=request.get('thickness_nm'),qcm_rate=request.get('qcm_rate'),delta=request.get('delta'))
            sampled_sha=_digest(sampled);observed=sampled['observed']
            initial,medoid_timing=timed(build_candidates,sampled,observed,max_per_scenario=3)
            indices={s:[c['original_draw_index'] for c in group['candidates']]
                for s,group in initial['scenarios'].items()}
            _require(_digest(sampled)==sampled_sha,'Original medoid extraction modified the joint sample')
            child_last=dict(sampled=sampled,sampled_sha256=sampled_sha,full=full,public=public,
                initial=initial,observed=observed,preview=None,preview_timing=None,
                preview_enabled=request['preview_on_sample'],frame_identity=_frame_identity(full,public),
                observed_sha256=_digest(observed),original_medoid_indices=indices,
                bank_plan_sha256=_sha(bindings['bank_plan']))
            respond(dict(sampled=sampled,sampled_sha256=sampled_sha,preview=None,
                frame_identity=child_last['frame_identity'],observed_sha256=child_last['observed_sha256'],
                original_medoid_indices=indices,bank_plan_sha256=child_last['bank_plan_sha256'],
                preview_on_sample=request['preview_on_sample'],forward_calls_during_sample=0,
                child_timings=dict(original_sample=sample_timing,original_geometry_medoids=medoid_timing)))
        except Exception as error:
            sys.stdout.write(_bytes(dict(ok=False,error=type(error).__name__+': '+str(error))).decode('utf8'))
            sys.stdout.flush()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--execution-root',type=Path,required=True)
    parser.add_argument('--execution-sha256',required=True)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--manifest-sha256',required=True)
    args=parser.parse_args()
    _require(args.worker,'Use FrozenManifestRuntime from the public manifest adapter')
    try:_worker(args)
    except Exception as error:
        sys.stdout.write(_bytes(dict(ok=False,error=type(error).__name__+': '+str(error))).decode('utf8'));sys.stdout.flush()
        traceback.print_exc(file=sys.stderr)
        return 1
    return 0


if __name__=='__main__':raise SystemExit(main())
