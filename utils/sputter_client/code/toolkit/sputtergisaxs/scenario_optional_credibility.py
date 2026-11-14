'Consume actual TRAIN/CAL OOD fits for exact optional-input presence.'
import json
from pathlib import Path
import numpy as np

from .scenario_runtime import _inside, _sha, _source_sha, _array_sha
from .stage5_posterior import ood_score

SCHEMA='four_scenario_optional_embedding_OOD_1'
FLAG_ORDER=('host','thickness','pre','QCM')
MASKS=np.array([[bool(i & (1 << j)) for j in range(4)] for i in range(16)],bool)
REGIMES=tuple('mask_'+''.join('1' if v else '0' for v in m) for m in MASKS)
GROUPS=('Si','oxide_particles','polymer','unknown')


def load_optional_ood(runtime,path):
    'Validate actual source, original rows, applicability and CAL thresholds.'
    path=Path(path).resolve();path=path/'run.json' if path.is_dir() else path
    digest=_sha(path);value=json.loads(path.read_text(encoding='utf8'));identity=value['identity']
    if (value.get('status')!='completed_optional_OOD_diagnostic'
        or value.get('schema')!=SCHEMA or identity.get('schema')!=SCHEMA
        or identity.get('runtime')!=runtime.receipt or identity.get('bank')!=runtime.declaration['bank']
        or identity.get('original_producer_source_pins')!=runtime.pins
        or identity.get('regimes')!=list(REGIMES) or identity.get('requested_known')!=MASKS.tolist()
        or identity.get('flag_order')!=list(FLAG_ORDER) or identity.get('groups')!=list(GROUPS)
        or identity.get('training_optional_policy')!=runtime.declaration['optional_policy']
        or identity.get('TEST_scientific_rows_decoded') is not False
        or value.get('TEST_scientific_rows_decoded') is not False
        or identity.get('p_x_estimated') is not False
        or identity.get('independent_OOD_validation_performed') is not False):
        raise ValueError('Actual original-model optional TRAIN/CAL artifact required')
    pins=identity['source_pins'];execution=identity['execution_source_pins']
    if pins!=execution:raise ValueError('Original producer and diagnostic execution source bindings differ')
    if identity.get('source_sha256')!=execution.get('scripts/v2_scenario_optional_ood.py'):
        raise ValueError('Actual optional OOD producer must be pinned')
    for relative,d in runtime.pins.items():
        if _source_sha(_inside(runtime.package,relative))!=d:
            raise ValueError('Original generating/training source changed: '+relative)
    for relative,d in execution.items():
        current=_inside(Path(__file__).parents[1],relative)


        choices=(_inside(runtime.package,relative),current)
        if relative.startswith('scripts/') and relative.endswith('.py'):
            choices=(_inside(path.parent,'scientific_source/'+relative),)+choices
        if not any(p.is_file() and _source_sha(p)==d for p in choices):
            raise ValueError('Actual optional diagnostic source unavailable: '+relative)
        if relative.startswith('sputtergisaxs/') and (not current.is_file() or _source_sha(current)!=d):
            raise ValueError('Executing optional scoring/input library differs: '+relative)
        if relative in runtime.pins and runtime.pins[relative]!=d:
            raise ValueError('Optional diagnostics changed original model/input science')
    records=value['records'];ids=identity['original_family_ids']
    if set(ids)!=set(('TRAIN','CAL')) or len(records)!=sum(len(v) for v in ids.values()):
        raise ValueError('Complete original optional OOD family denominators required')
    expected=ids['TRAIN']+ids['CAL']
    if len(set(expected))!=len(expected):raise ValueError('Original family IDs duplicated')
    for split in ('TRAIN','CAL'):
        if (any(isinstance(fid,bool) or not isinstance(fid,int) or fid<0
            or not (fid%10<8 if split=='TRAIN' else fid%10==8) for fid in ids[split])
            or sorted(rec['family_id'] for rec in records if rec['split']==split)!=sorted(ids[split])):
            raise ValueError('Original optional OOD split differs or contains TEST')
    if any(rec['index']!=rec['family_id'] or rec['split']!=('TRAIN' if i<len(ids['TRAIN']) else 'CAL')
        or rec['host_category'] not in GROUPS or rec['scenario'] not in range(4)
        for i,rec in enumerate(records)):
        raise ValueError('Original optional OOD row identity/order differs')
    artifact=_inside(path.parent,value['diagnostic_file'])
    if _sha(artifact)!=value['diagnostic_sha256']:raise ValueError('Optional OOD diagnostic changed')
    fits={}
    with np.load(artifact,allow_pickle=False) as z:
        n=len(records);family=z['family_id'];splits=z['split_index'];known=z['original_known']
        if (family.dtype.kind not in 'iu' or family.tolist()!=[r['family_id'] for r in records]
            or splits.tolist()!=[0 if r['split']=='TRAIN' else 1 for r in records]
            or known.dtype.kind!='b' or known.shape!=(n,4)
            or known.tolist()!=[r['original_known'] for r in records]
            or z['host_index'].tolist()!=[GROUPS.index(r['host_category']) for r in records]
            or z['requested_known'].tolist()!=MASKS.tolist()):
            raise ValueError('Original optional OOD persisted identities differ')
        applicable=np.all(known[:,None,:] | ~MASKS[None],axis=2)
        scores=z['score'];valid=z['valid_score']
        if (scores.shape!=(n,16) or valid.shape!=scores.shape
            or not np.array_equal(valid,np.isfinite(scores))
            or not np.array_equal(z['applicable'],applicable)
            or np.isfinite(scores[~applicable]).any()):
            raise ValueError('Exact presence applicability/score completeness differs')
        for i,regime in enumerate(REGIMES):
            report=value['reports'][regime];train=(splits==0)&applicable[:,i];cal=(splits==1)&applicable[:,i]
            if (report.get('requested_known')!=MASKS[i].tolist() or report.get('exact_presence_mask') is not True
                or report.get('applicable_original_TRAIN_rows')!=int(train.sum())
                or report.get('applicable_original_CAL_rows')!=int(cal.sum())):
                raise ValueError('Optional OOD original applicable denominators differ')
            groups=np.array([r['host_category'] for r in records]) if MASKS[i,0] else np.full(n,'unknown')
            scopes=[(cal,report['overall'])]+[(cal&(groups==g),report['host_groups'][g]) for g in GROUPS]
            for sel,scope in scopes:
                count=int(sel.sum());cut=scope['own_CAL_threshold']
                expected_cut=float(np.quantile(scores[sel,i],.99)) if count>=30 and np.isfinite(scores[sel,i]).all() else None
                if (cut.get('minimum_CAL')!=30 or cut.get('original_CAL_rows')!=count
                    or cut.get('valid_CAL_scores')!=int(np.isfinite(scores[sel,i]).sum())
                    or cut.get('value')!=expected_cut):
                    raise ValueError('Original exact presence CAL99 threshold differs')
            cut=report['overall']['own_CAL_threshold']['value'];scope=np.full(n,-1,np.int8)
            if cut is not None:scope[valid[:,i]]=(scores[valid[:,i],i]<=cut).astype(np.int8)
            if not np.array_equal(z['in_scope'][:,i],scope):raise ValueError('Actual optional OOD scope scores differ')
            if report['TRAIN_fit_available']:
                dim=runtime.declaration.get('embedding_dim',runtime.declaration['hidden']);mean=z[regime+'__mean'].copy();precision=z[regime+'__precision'].copy();ridge=float(z[regime+'__ridge'])
                if (train.sum()<2 or not valid[train,i].all() or mean.shape!=(dim,) or precision.shape!=(dim,dim)
                    or not np.isfinite(mean).all() or not np.isfinite(precision).all() or not np.isfinite(ridge) or ridge<=0):
                    raise ValueError('Optional OOD TRAIN fit incomplete or nonfinite')
                fits[regime]=dict(mean=mean,precision=precision,ridge=ridge)
    if _sha(path)!=digest or _sha(artifact)!=value['diagnostic_sha256']:
        raise ValueError('Optional OOD source changed during loading')
    return dict(path=path,value=value,fits=fits,binding=dict(run_sha256=digest,
        diagnostic_sha256=value['diagnostic_sha256'],source_pins=pins,
        original_TRAIN_rows=len(ids['TRAIN']),original_CAL_rows=len(ids['CAL']),
        TEST_scientific_rows_decoded=False,independent_OOD_validation_performed=False))


def score_optional_ood(runtime,frame,encoded,loaded,result):
    'Score the original verified encoder context; never redraw a posterior.'
    import torch
    flags=np.asarray(frame['known'],bool);regime='mask_'+''.join('1' if b else '0' for b in flags)
    result.update(input_regime=regime,exact_presence_mask=flags.tolist(),
        parameter_calibration_for_this_presence_required=True)
    if loaded is None:
        result['missing_reason']='Completed exact optional-presence TRAIN/CAL OOD artifact not supplied'
        return result
    path=loaded['path'];value=loaded['value'];binding=loaded['binding']
    if _sha(path)!=binding['run_sha256'] or _sha(_inside(path.parent,value['diagnostic_file']))!=binding['diagnostic_sha256']:
        raise ValueError('Bound optional OOD artifact changed before scoring')
    report=value['reports'][regime];cut=report['overall']['own_CAL_threshold']
    local=report['host_groups'][result['host_category']]['own_CAL_threshold']
    result.update(threshold=cut['value'],threshold_status=cut['status'],
        threshold_available=cut['value'] is not None,input_regime_matched=True,
        host_threshold=local['value'],host_threshold_status=local['status'],optional_OOD_binding=binding)
    if regime not in loaded['fits']:
        result['missing_reason']=report['fit_missing_reason'];return result
    with torch.no_grad():embedding=runtime.net.embedding(encoded).detach().cpu().numpy()
    if not np.isfinite(embedding).all():
        result['missing_reason']='Actual inference embedding is nonfinite';return result
    score=float(ood_score(embedding.astype(float),loaded['fits'][regime])[0])
    if not np.isfinite(score):
        result['missing_reason']='Actual optional OOD score is nonfinite';return result
    result.update(score=score,OOD=bool(score>cut['value']) if cut['value'] is not None else None,
        host_OOD=bool(score>local['value']) if local['value'] is not None else None,
        embedding_sha256=_array_sha(embedding),score_bound=True,
        missing_reason=None if cut['value'] is not None else 'Complete original exact-presence CAL99 threshold unavailable')
    return result
