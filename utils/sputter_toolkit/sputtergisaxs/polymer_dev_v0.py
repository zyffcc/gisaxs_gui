"""Reusable development-only polymer scan workflow. No TEST or training access.

Python: run(input_path, out, checkpoint=..., input_policy=..., energy_ev=...,
            sdd_mm=2500, alpha_i_deg=.4, metal='Ag', device='auto').
Phases also permit a read-only intake on one computer and computation on another.
"""
from __future__ import annotations
from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
from functools import lru_cache
import copy,csv,gzip,json,time,os
import numpy as np
from .neural_refine_training import sha,apply_policy
from .polymer_dev_intake import prepare,read,write
from .neural_refine_input import extract,encode
from .neural_refine_calibration import extract_mixture,physical,safe_json
from .neural_refine_chart import NAMES
from .neural_refine_fullmodes import component_starts

VERSION='聚合物开发版v0'
FIELDS=('D_nm','R_number_mean_nm','radius_sigma_log','burial_base_depth_nm','height_nm','polymer_thickness_nm')


def public(value,reason='未提供或未测得；见对应状态与条件'):
    if value is None:return reason
    if isinstance(value,dict):return {k:public(v,reason) for k,v in value.items()}
    if isinstance(value,(list,tuple,np.ndarray)):return [public(v,reason) for v in value]
    return safe_json(value)


def feature_chunk(task):
    intake,chunk_name,out,comparison=task;intake=Path(intake);out=Path(out)
    dest=out/(Path(chunk_name).stem+('_12keV' if comparison else '')+'.json.gz')
    meta=read(intake/Path(chunk_name).with_suffix('.json'));run=read(intake/'run.json')
    identity=dict(intake_sha256=sha(intake/chunk_name),geometry=run['geometry'],comparison_12keV=comparison,
                  input_source_sha256=sha(Path(__file__).with_name('neural_refine_input.py')))
    if dest.exists():
        with gzip.open(dest,'rt',encoding='utf-8') as f:saved=json.load(f)
        if saved['identity']!=identity:raise ValueError('Feature source changed')
        return str(dest)
    z=np.load(intake/chunk_name);suffix='_12keV' if comparison else '';Y=z['Y'+suffix];P=z['P'+suffix]
    rows=[];vectors=[];energy=12000. if comparison else run['geometry']['energy_ev']
    for i,metadata in enumerate(meta['rows']):
        item=extract(Y[i],P[i]*metadata['exposure_s'],energy,run['geometry']['alpha_i_deg'],metal=run['model_metal'])
        rows.append(dict(metadata=metadata,observations=item['observations'],chunk=chunk_name,chunk_row=i))
        vectors.append(encode(item))
    out.mkdir(parents=True,exist_ok=True)
    array=dest.with_suffix('.npz');np.savez_compressed(array,encoded=np.asarray(vectors,np.float32))
    with gzip.open(dest,'wt',encoding='utf-8') as f:json.dump(dict(identity=identity,rows=rows,vector_file=array.name,vector_sha256=sha(array)),f,ensure_ascii=False,allow_nan=False)
    print('FEATURE_CHUNK',intake.name,chunk_name,len(rows),flush=True);return str(dest)


def features(intake,out,*,workers=16,comparison=False):
    intake=Path(intake);out=Path(out);out.mkdir(parents=True,exist_ok=True);run=read(intake/'run.json')
    tasks=[(str(intake),c['file'],str(out),comparison) for c in run['chunks']]
    with ProcessPoolExecutor(max_workers=workers) as pool:files=list(pool.map(feature_chunk,tasks))
    write(out/'run.json',dict(status='completed',intake=str(intake),intake_run_sha256=sha(intake/'run.json'),
        comparison_12keV=comparison,files=[dict(file=Path(p).name,sha256=sha(p)) for p in files],frames=run.get('analysed_frames',run['selected_frames'])))


def infer(intake,feature_dir,out,*,checkpoint,input_policy,device='auto'):
    import torch
    from .neural_refine_infer import NRExpert
    from .staged_v3 import classify_sequence
    intake=Path(intake);feature_dir=Path(feature_dir);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    manifest=read(intake/'run.json');frun=read(feature_dir/'run.json')
    if frun['intake_run_sha256']!=sha(intake/'run.json'):raise ValueError('Intake manifest changed')
    device=('cuda' if torch.cuda.is_available() else 'cpu') if device=='auto' else device
    expert=NRExpert(checkpoint,input_policy,device=device)
    if expert.declaration['bank']['module']!='polymer_embed':raise ValueError('Polymer expert required')
    start=time.monotonic();allrows=[];energy=12000. if frun['comparison_12keV'] else manifest['geometry']['energy_ev']
    def seconds(stamp):
        try:return datetime.fromisoformat(stamp).timestamp()
        except (TypeError,ValueError):return None
    t0=seconds(manifest['first_timestamp']);mean=expert.net.chart_mean.cpu().numpy();scale=expert.net.chart_scale.cpu().numpy()
    predfile=out/'network.jsonl.gz'
    with gzip.open(predfile,'wt',encoding='utf-8') as dest:
        for record in frun['files']:
            p=feature_dir/record['file']
            if sha(p)!=record['sha256']:raise ValueError('Feature receipt changed')
            with gzip.open(p,'rt',encoding='utf-8') as f:doc=json.load(f)
            array=feature_dir/doc['vector_file']
            if sha(array)!=doc['vector_sha256']:raise ValueError('Encoded feature array changed')
            x=apply_policy(np.load(array)['encoded'],expert.policy);mix,active=extract_mixture(expert,x)
            names=[NAMES[j] for j in active];sd=np.sqrt((mix['chol']**2).sum(-1))
            for i,row in enumerate(doc['rows']):
                one={k:v[i:i+1] for k,v in mix.items()};modes=component_starts(expert,one,active,energy_ev=energy,
                    alpha_i_deg=manifest['geometry']['alpha_i_deg'],metal=manifest['model_metal'])
                modes.sort(key=lambda m:m['probability'],reverse=True);selected=modes[:3]
                for m in selected:
                    b,k=map(int,m['mode_id'].split(':'));u=mix['mu'][i,b,k];s=sd[i,b,k]
                    quant=physical(np.stack([u-1.95996398454*s,u-.99445788321*s,u+.99445788321*s,u+1.95996398454*s]),names,mean[active],scale[active])
                    m['intervals']={name:dict(interval68=quant[1:3,j].tolist(),interval95=quant[[0,3],j].tolist()) for j,name in enumerate(names)}
                    m['values']['height_nm']=m['values']['R_number_mean_nm'];m['intervals']['height_nm']=copy.deepcopy(m['intervals']['R_number_mean_nm'])
                    m['height_condition']='hemisphere h=R; exposed mean-mark height=max(R-burial,0), not independent height'
                    m['interval_scope']='Raw Gaussian component marginal conditional ranges; not empirical CAL or refined confidence intervals'
                ti=seconds(row['metadata']['timestamp']);elapsed=ti-t0 if ti is not None and t0 is not None else None
                data=dict(**row,key=row['metadata']['frame'],time_s=elapsed,modes=selected,
                    omitted_component_probability=float(1-sum(m['probability'] for m in selected)),energy_ev=energy,
                    checkpoint_sha256=expert.checkpoint_sha256,input_policy_sha256=sha(input_policy),version=VERSION)
                dest.write(json.dumps(public(data),ensure_ascii=False,allow_nan=False)+'\n')
                allrows.append(dict(key=data['key'],metadata=dict(time_s=elapsed),observations=row['observations']))
    # Some acquisitions repeat rounded header timestamps. Use order, never
    # invent sub-second timing when the source clock is not strictly ordered.
    times=[r['metadata']['time_s'] for r in allrows]
    monotonic=all(t is not None for t in times) and all(b>a for a,b in zip(times[:-1],times[1:]))
    if not monotonic:
        for r in allrows:r['metadata']['time_s']=None
    route=classify_sequence(allrows);write(out/'stages.json',public(route))
    write(out/'inference.json',dict(status='completed',version=VERSION,frames=len(allrows),intake=str(intake),
        feature_dir=str(feature_dir),checkpoint=str(checkpoint),checkpoint_sha256=expert.checkpoint_sha256,input_policy=str(input_policy),
        policy_sha256=sha(input_policy),network_sha256=sha(predfile),stages_sha256=sha(out/'stages.json'),device=str(device),
        wall_seconds=time.monotonic()-start,comparison_12keV=frun['comparison_12keV'],energy_ev=energy,
        stage_clock='header_time' if monotonic else 'input_order_due_to_missing_or_repeated_header_time',
        scope='Development exception; sparse polymer domain not accepted; no new TEST',Dcorr_truncation=False,
        probability_scope='Original joint branch/component masses; top3 not renormalized; not empirical morphology probabilities',
        source_sha256=sha(__file__)))
    print('INFERENCE_COMPLETE',intake.name,len(allrows),round(time.monotonic()-start,1),flush=True)


@lru_cache(maxsize=2)
def chunk_arrays(path):
    with np.load(path) as z:return {k:z[k] for k in z.files}


def refinement_task(task):
    intake,row,out,deadline,seconds,comparison,reference=task;out=Path(out)/f"{row['metadata']['frame_id']:06d}.json"
    if out.exists():
        old=read(out)
        if old['checkpoint_sha256']!=row['checkpoint_sha256']:raise ValueError('Refinement belongs to different weight')
        return old['status']
    if time.time()>=deadline:return 'network_only_total_budget'
    from .polymer_dev_quick import quick_refine
    run=read(Path(intake)/'run.json');a=chunk_arrays(str(Path(intake)/row['chunk']));i=row['chunk_row'];suffix='_12keV' if comparison else ''
    obs={k:None if isinstance(v,str) and v.startswith('未提供') else v for k,v in row['observations'].items()}
    try:
        result=quick_refine(a['Y'+suffix][i],a['P'+suffix][i]*row['metadata']['exposure_s'],row['modes'][0],obs,reference,
            energy_ev=row['energy_ev'],alpha_i_deg=run['geometry']['alpha_i_deg'],metal=run['model_metal'],actual_metal=run['metal'],
            seconds=min(seconds,max(.01,deadline-time.time())))
    except Exception as exc:result=dict(status='network_only',reason=type(exc).__name__+': '+str(exc),wall_seconds=0.)
    result.update(frame=row['key'],checkpoint_sha256=row['checkpoint_sha256'],mode_id=row['modes'][0]['mode_id'])
    write(out,public(result));return result['status']


def refine(intake,out,*,workers=32,budget_seconds=7200.,frame_seconds=45.):
    from collections import Counter
    intake=Path(intake);out=Path(out);meta=read(out/'inference.json');run=read(intake/'run.json')
    with gzip.open(out/'network.jsonl.gz','rt',encoding='utf-8') as f:rows=[json.loads(line) for line in f]
    d=out/'refinement';d.mkdir(exist_ok=True);comparison=meta['comparison_12keV'];suffix='_12keV' if comparison else ''
    first=rows[0];array=chunk_arrays(str(intake/first['chunk']));Y=array['Y'+suffix][0];E=array['P'+suffix][0]*first['metadata']['exposure_s']
    reference=dict(Y=Y,E=E,observations=extract(Y,E,meta['energy_ev'],run['geometry']['alpha_i_deg'],metal=run['model_metal'])['observations'],
                   source=first['key']+' earliest observed; not certified zero metal')
    start=time.time();deadline=start+min(budget_seconds,7200.)
    write(out/'refinement_run.json',dict(status='running',started_UTC=datetime.now(timezone.utc).isoformat(),deadline_unix=deadline,frames=len(rows),
        per_frame_seconds=frame_seconds,workers=workers,maximum_budget_seconds=min(budget_seconds,7200.),source_sha256=sha(Path(__file__).with_name('polymer_dev_quick.py'))))
    # Spread limited attempts over the entire evolution deterministically,
    # while every frame already has all three network modes and descriptors.
    order=np.argsort([int.from_bytes(__import__('hashlib').sha256(r['key'].encode()).digest()[:8],'big') for r in rows])
    # Chunk order spans the scan; each worker can reuse a chunk's compressed
    # arrays instead of random-reading a large NPZ for every single frame.
    chunks=list(dict.fromkeys(r['chunk'] for r in rows));chunks.sort(key=lambda k:__import__('hashlib').sha256(k.encode()).hexdigest())
    order=[i for c in chunks for i,r in enumerate(rows) if r['chunk']==c]
    tasks=iter((str(intake),rows[i],str(d),deadline,frame_seconds,comparison,reference) for i in order)
    counts=Counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending=set();submitted=0;completed=0
        while True:
            while len(pending)<2*workers and time.time()<deadline:
                task=next(tasks,None)
                if task is None:break
                pending.add(pool.submit(refinement_task,task));submitted+=1
            if not pending:break
            done,pending=wait(pending,return_when=FIRST_COMPLETED)
            for future in done:counts[future.result()]+=1;completed+=1
            if completed%100<len(done):print('REFINEMENT_PROGRESS',intake.name,completed,len(rows),dict(counts),flush=True)
        counts['network_only_total_budget']+=len(rows)-submitted
    doc=read(out/'refinement_run.json');doc.update(status='completed_budgeted_attempts',wall_seconds=time.time()-start,counts=dict(counts));write(out/'refinement_run.json',doc)


def export(intake,out,*,qcm_rate_nm_s=None,deposition_start_s=None):
    intake=Path(intake);out=Path(out);manifest=read(intake/'run.json');info=read(out/'inference.json')
    route=read(out/'stages.json')['rows'];stages={r['key']:r for r in route};rows=[];first_yoneda=None;fit_count=0;compatible=0
    with gzip.open(out/'network.jsonl.gz','rt',encoding='utf-8') as f:
        for line in f:
            d=json.loads(line);o=d['observations'];has_peak=o['feature_status'] in ('bilateral','one_side')
            y=o.get('Yoneda_deg');first_yoneda=y if first_yoneda is None and isinstance(y,(int,float)) else first_yoneda
            row=dict(frame=d['key'],frame_id=d['metadata']['frame_id'],time_s=d['time_s'],header_timestamp=d['metadata']['timestamp'],
                exposure_s=d['metadata']['exposure_s'],selected_counts=d['metadata']['selected_counts'],valid_pixels=d['metadata']['valid_pixel_count'],
                stage=stages[d['key']]['stage_label'],stage_reason=stages[d['key']]['stage_reason'],stage_scope='观测条件下的阶段候选，非形貌真值',
                D_corr_nm=o['D_corr_nm'],q_star_nm_inv=o['q_star_nm_inv'],FWHM_nm_inv=o['FWHM_nm_inv'],
                feature_status=o['feature_status'],feature_AF_band=o['feature_AF_band'],Yoneda_deg=y,
                Yoneda_shift_deg=y-first_yoneda if isinstance(y,(int,float)) and first_yoneda is not None else '未测得Yoneda位置',
                highq_log_slope=o['highq_log_slope'],highq_reason=o['highq_reason'],direct_grade='直接测量；总散射特征，未唯一分离宿主/金属',
                D_grade='D网络-有峰；开发条件候选' if has_peak else '无可分辨峰，不输出D',R_grade='R网络＋范围；域门未验收',
                burial_height_grade='埋深与高度仅条件；h=R半球条件',version=VERSION,system_note=manifest['system_note'],
                metal=manifest['metal'],model_metal=manifest['model_metal'],Cu_scope=manifest['Cu_scope'],
                energy_ev=d['energy_ev'],wavelength_nm=1239.841984/d['energy_ev'],sdd_mm=manifest['geometry']['sdd_mm'],
                beam_col=manifest['geometry']['db_col'],beam_row=manifest['geometry']['db_row'],alpha_i_deg=manifest['geometry']['alpha_i_deg'],
                geometry_scope=manifest['centre']['geometry_status'],checkpoint_sha256=d['checkpoint_sha256'],policy_sha256=d['input_policy_sha256'],
                raw_sha256=d['metadata']['raw_sha256'],omitted_mode_probability=d['omitted_component_probability'],
                interval_scope='原始分量条件68/95%范围；未经经验校准，不是精修后区间',
                distinguish='SEM/AFM区分D与R；XRR/椭偏约束聚合物厚度；截面TEM或多角GISAXS区分埋深和高度')
            for rank,m in enumerate(d['modes'],1):
                prefix=f'mode{rank}_';row[prefix+'probability']=m['probability'];row[prefix+'branch']=str(m['branch']);row[prefix+'status']=m['status']
                for name in FIELDS:
                    row[prefix+name]=m['values'].get(name,'条件参数未进入此正向自由度')
                    for level in ('68','95'):
                        interval=m['intervals'].get(name,{}).get('interval'+level,['条件范围未定义']*2)
                        row[prefix+name+'_lo'+level],row[prefix+name+'_hi'+level]=interval
                    if name=='D_nm' and not has_peak:
                        for k in list(row):
                            if k.startswith(prefix+'D_nm'):row[k]='无可分辨峰，不输出D'
            fp=out/'refinement'/f"{d['metadata']['frame_id']:06d}.json";fit=read(fp) if fp.exists() else dict(status='network_only_total_budget',reason='总精修预算内未完成；输出网络候选')
            row['refinement_status']=fit['status'];row['refinement_reason']=fit.get('reason','完成固定预算精修');row['refinement_seconds']=fit.get('wall_seconds','未运行物理精修')
            # A timed-out trial (possibly only a finite-difference probe) is
            # not a completed refinement. Preserve its curve/score in the
            # audit but keep the public structure at the network solution.
            converged=fit['status']=='refined'
            vals=fit.get('values',{}) if converged else {}
            row['refinement_converged']=converged
            for name in FIELDS:row['refined_'+name]=vals.get(name,'精修未完成，见网络条件候选') if name!='D_nm' or has_peak else '无可分辨峰，不输出D'
            score=next((s for s in fit.get('model_scores',[]) if abs(s.get('relative_model_error',s.get('eps',-1))-.05)<1e-10),None)
            if score is None:
                # compare_floors uses model_error_fraction in current source.
                score=next((s for s in fit.get('model_scores',[]) if abs(s.get('model_error_fraction',-1)-.05)<1e-10),None)
            row['compatibility_5pct']=score['compatible_at_5pct_asymptotic'] if score else '未计算；精修未完成或无可用曲线'
            row['compatibility_p_5pct']=score['asymptotic_tail_probability'] if score else '未计算'
            row['compatibility_solution']='收敛精修曲线' if score and converged else '预算内已算的候选曲线；精修未收敛，结构仍取网络候选' if score else '尚无可用物理曲线'
            row['compatibility_scope']='仅保留计数区的5%模型误差相容度；不含峰特征惩罚，不认证结构正确'
            if score:fit_count+=1;compatible+=int(row['compatibility_5pct'])
            row['QCM_nominal_thickness_nm']=max(d['time_s']-deposition_start_s,0.)*qcm_rate_nm_s if qcm_rate_nm_s is not None and deposition_start_s is not None and isinstance(d['time_s'],(int,float)) else '未给QCM速率与沉积起点；不推绝对质量'
            rows.append(public(row))
    path=out/'frames.csv'
    with path.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    if len(rows)!=manifest.get('analysed_frames',manifest['selected_frames']):raise ValueError('Full selected frame denominator lost')
    write(out/'delivery.json',dict(version=VERSION,status='completed',frames=len(rows),received_frames=manifest['received_frames'],
        geometry=dict(manifest['geometry'],energy_ev=info['energy_ev']),centre=manifest['centre'],system_note=manifest['system_note'],metal=manifest['metal'],Cu_scope=manifest['Cu_scope'],
        fit_score_frames=fit_count,compatible_5pct_frames=compatible,all_frame_compatibility_rate=compatible/len(rows),
        fit_evaluable_compatibility_rate=compatible/fit_count if fit_count else '未取得可计算相容度的精修曲线',
        converged_refinement_frames=sum(r['refinement_converged'] for r in rows),
        csv_sha256=sha(path),checkpoint_sha256=info['checkpoint_sha256'],policy_sha256=info['policy_sha256'],
        no_new_TEST=True,formal_validation=False,Dcorr_truncation=False,physical_D_domain_nm=[2,120]))
    return rows


def run(input_path,out,*,checkpoint,input_policy,energy_ev,sdd_mm,alpha_i_deg,metal,device='auto',workers=8,
        qcm_rate_nm_s=None,deposition_start_s=None,budget_seconds=7200,frame_seconds=45,**intake_options):
    """Single CBF or scan. CPU fallback; measured pixels are never overwritten."""
    out=Path(out);intake=out/'intake';feature_dir=out/'features'
    prepare(input_path,intake,energy_ev=energy_ev,sdd_mm=sdd_mm,alpha_i_deg=alpha_i_deg,metal=metal,workers=workers,**intake_options)
    features(intake,feature_dir,workers=workers);infer(intake,feature_dir,out,checkpoint=checkpoint,input_policy=input_policy,device=device)
    refine(intake,out,workers=workers,budget_seconds=budget_seconds,frame_seconds=frame_seconds)
    rows=export(intake,out,qcm_rate_nm_s=qcm_rate_nm_s,deposition_start_s=deposition_start_s)
    from .polymer_dev_plot import plot
    plot(rows,out)
    if intake_options.get('compare_12keV'):
        alt=out/'comparison_12keV';features(intake,alt/'features',workers=workers,comparison=True)
        infer(intake,alt/'features',alt,checkpoint=checkpoint,input_policy=input_policy,device=device)
        altrows=export(intake,alt);plot(altrows,alt);compare_geometry(out,alt)
    return read(out/'delivery.json')


def compare_geometry(primary,alternative):
    primary,alternative=Path(primary),Path(alternative)
    a=read(primary/'delivery.json');b=read(alternative/'delivery.json')
    if a['checkpoint_sha256']!=b['checkpoint_sha256']:raise ValueError('Geometry comparison requires same weight')
    def load(p):
        with (p/'frames.csv').open(encoding='utf-8') as f:return {r['frame']:r for r in csv.DictReader(f)}
    aa,bb=load(primary),load(alternative)
    if set(aa)!=set(bb):raise ValueError('Geometry pair denominator differs')
    rows=[];summary={}
    for frame,left in aa.items():
        row=dict(frame=frame)
        for key in ('D_corr_nm','mode1_D_nm','mode1_R_number_mean_nm'):
            row[key+'_primary']=left[key];row[key+'_12keV']=bb[frame][key]
            try:row[key+'_relative_difference_pct']=100*(float(bb[frame][key])/float(left[key])-1)
            except (ValueError,ZeroDivisionError):row[key+'_relative_difference_pct']='无峰或条件数值不适用'
        rows.append(row)
    with (primary/'geometry_comparison.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    for key in ('D_corr_nm','mode1_D_nm','mode1_R_number_mean_nm'):
        values=[r[key+'_relative_difference_pct'] for r in rows if isinstance(r[key+'_relative_difference_pct'],(float,int))]
        summary[key]=dict(paired_frames=len(values),median_signed_pct=float(np.median(values)) if values else '无适用配对',
            P90_absolute_pct=float(np.quantile(np.abs(values),.9)) if values else '无适用配对')
    write(primary/'geometry_comparison.json',dict(frames=len(rows),summary=summary,scope='Same actual images reprojected then rerun through same weight; not analytic rescaling of network R',
        primary_energy_ev=read(primary/'inference.json')['energy_ev'],alternative_energy_ev=12000.,checkpoint_sha256=a['checkpoint_sha256']))
