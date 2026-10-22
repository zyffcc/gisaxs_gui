"""Exact photon aggregation and paired low-q review on existing raw-count intake.

Original CBF files remain read-only; only SHA-bound derived count chunks
are consumed. No model, new TEST, or raw image copies.
"""
from __future__ import annotations
from pathlib import Path
from datetime import datetime
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import csv,gzip,json,copy,hashlib
import numpy as np
from .neural_refine_report import read,write as _write,sha
from .neural_refine_input import extract,encode
from .linecut_v3 import direct_features
from .polymer_review_features import features_review
from .polymer_review_stages import stages,periodicity,number


def jsonable(obj):
    if isinstance(obj,dict):return {str(k):jsonable(v) for k,v in obj.items()}
    if isinstance(obj,(list,tuple,np.ndarray)):return [jsonable(v) for v in obj]
    if isinstance(obj,np.generic):return jsonable(obj.item())
    if isinstance(obj,float) and not np.isfinite(obj):return None
    return obj


def write(path,obj):return _write(path,jsonable(obj))


def dump_gz(path,obj):
    with gzip.open(path,'wt',encoding='utf-8') as f:json.dump(jsonable(obj),f,ensure_ascii=False,allow_nan=False)


def load_gz(path):
    with gzip.open(path,'rt',encoding='utf-8') as f:return json.load(f)


def clock(rows):
    values=np.array([datetime.fromisoformat(r['timestamp']).timestamp() for r in rows]);t=values-values[0]
    if np.any(np.diff(t)<=0):raise ValueError('Header timestamps missing/repeated/nonmonotonic; cannot invent0.5s clock')
    return t


def bin_ids(times,*,seconds=.5,early_seconds=.1,early_until=5.):
    t=np.asarray(times,float);dt=float(np.median(np.diff(t))) if len(t)>1 else 0.
    if len(t)>1 and np.any(np.diff(t)<=0):raise ValueError('Strictly ordered header times required')
    # Round acquisition jitter at nominal bin edges; record actual first/last
    # source times and exposures, so 9/11 or partial bins are not called10.
    t=t+min(dt/2,seconds/4)
    if early_until<=0:return np.floor(t/seconds).astype(int)
    cut=int(round(early_until/early_seconds))
    return np.where(t<early_until,np.floor(t/early_seconds),cut+np.floor((t-early_until)/seconds)).astype(int)


def intake_rows(intake):
    intake=Path(intake);run=read(intake/'run.json');rows=[]
    for c in run['chunks']:
        p=intake/c['file'];meta=p.with_suffix('.json')
        if sha(p)!=c['sha256'] or sha(meta)!=c['metadata_sha256']:raise ValueError('Original count/metadata SHA changed')
        doc=read(meta)
        for j,r in enumerate(doc['rows']):rows.append(dict(metadata=r,chunk=c['file'],chunk_row=j))
    # analysed_frames = selected - explicitly excluded damaged frames (older
    # intakes without the key had no exclusions, or already stored readable frames).
    if len(rows)!=run.get('analysed_frames',run['selected_frames']):raise ValueError('Intake denominator differs')
    t=clock([r['metadata'] for r in rows])
    for r,ti in zip(rows,t):r['time_s']=float(ti)
    return run,rows


def raw_worker(task):
    intake,name,dest,geometry=task;intake=Path(intake);dest=Path(dest)
    p=intake/name;meta=read(p.with_suffix('.json'));z=np.load(p);allY=z['Y'];allP=z['P'];result=[]
    for i,m in enumerate(meta['rows']):
        Y=allY[i];E=allP[i]*m['exposure_s'];obs=features_review(Y,E)
        result.append(dict(metadata=m,observations=obs,chunk=name,chunk_row=i))
    dest.parent.mkdir(parents=True,exist_ok=True);dump_gz(dest,result)
    print('REVIEW_RAW',name,len(result),flush=True);return dict(file=dest.name,sha256=sha(dest))


def aggregate(intake,out,run,rows,ids,*,nominal_seconds):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);lookup={};source_chunk=None;z=None
    sumsY=sumsE=None;current=None;group=[];saved=[];arraysY=[];arraysE=[];metadata=[];serial=0
    raw_total=0;merged_total=0;raw_exposure=0.;merged_exposure=0.
    cadence=float(np.median(np.diff([r['time_s'] for r in rows]))) if len(rows)>1 else rows[0]['metadata']['exposure_s']
    def save_chunk():
        nonlocal arraysY,arraysE,metadata
        if not metadata:return
        path=out/f'chunk_{len(saved):05d}.npz';np.savez_compressed(path,Y=np.asarray(arraysY,dtype=np.uint64),E=np.asarray(arraysE))
        write(path.with_suffix('.json'),dict(rows=metadata));saved.append(dict(file=path.name,sha256=sha(path),metadata_sha256=sha(path.with_suffix('.json'))))
        arraysY=[];arraysE=[];metadata=[]
    def close():
        nonlocal serial,merged_total,merged_exposure
        if not group:return
        exposure=sum(r['metadata']['exposure_s'] for r in group);members=[r['metadata']['frame'] for r in group]
        m=dict(frame=f'bin_{serial:06d}',frame_id=serial,time_s=float(np.average([r['time_s'] for r in group],weights=[r['metadata']['exposure_s'] for r in group])),
            timestamp=group[0]['metadata']['timestamp'],exposure_s=exposure,
            first_time_s=group[0]['time_s'],last_time_s=group[-1]['time_s'],first_raw_frame=members[0],last_raw_frame=members[-1],
            raw_frames=len(group),raw_members=members,raw_sha256=hashlib.sha256(('\n'.join(r['metadata']['raw_sha256'] for r in group)).encode()).hexdigest(),
            selected_counts=int(sumsY.sum()),valid_pixel_count=float(sumsE.sum()/exposure),
            nominal_bin_seconds=nominal_seconds,actual_time_span_s=group[-1]['time_s']-group[0]['time_s'])
        effective_nominal=.1 if nominal_seconds==.5 and group[0]['time_s']<5 else nominal_seconds
        m['nominal_bin_seconds']=effective_nominal
        m['covered_duration_s']=m['actual_time_span_s']+cadence
        m['complete_nominal_bin']=bool(m['covered_duration_s']>=.9*effective_nominal)
        for frame in members:lookup[frame]=serial
        arraysY.append(sumsY.copy());arraysE.append(sumsE.copy());metadata.append(m);serial+=1
        merged_total+=int(sumsY.sum());merged_exposure+=float(sumsE.sum())
        if len(metadata)>=32:save_chunk()
    for r,bid in zip(rows,ids):
        if source_chunk!=r['chunk']:
            if z is not None:z.close()
            source_chunk=r['chunk'];z=np.load(Path(intake)/source_chunk)
            allY=z['Y'];allP=z['P']
        Y=allY[r['chunk_row']].astype(np.uint64);E=allP[r['chunk_row']].astype(float)*r['metadata']['exposure_s']
        raw_total+=int(Y.sum());raw_exposure+=float(E.sum())
        if current!=bid:
            close();current=bid;group=[];sumsY=np.zeros_like(Y);sumsE=np.zeros_like(E)
        group.append(r);sumsY+=Y;sumsE+=E
    close();save_chunk()
    if z is not None:z.close()
    if raw_total!=merged_total or not np.isclose(raw_exposure,merged_exposure,rtol=1e-12):raise ValueError('Photon/exposure closure failed')
    receipt=dict(status='completed',source_intake=str(intake),source_intake_sha256=sha(Path(intake)/'run.json'),
        raw_frames=len(rows),bins=serial,excluded_raw_frames=run.get('excluded_frames',[]),chunks=saved,nominal_seconds=nominal_seconds,
        geometry=run['geometry'],model_metal=run['model_metal'],metal=run['metal'],system_note=run['system_note'],
        total_raw_counts=raw_total,total_merged_counts=merged_total,total_pixel_exposure=merged_exposure,
        photon_exposure_closure=True,exposure_normalization='Exact Y sum, exact E=sum(P*header exposure); no parameter averaging',
        incomplete_edges='Retained with actual membership and timestamps; no invented frames',source_sha256=sha(__file__))
    write(out/'run.json',receipt);write(out/'membership.json',lookup);return receipt


def merged_worker(task):
    folder,c,run=task;folder=Path(folder);z=np.load(folder/c['file']);allY=z['Y'];allE=z['E'];meta=read((folder/c['file']).with_suffix('.json'));rows=[];encoded=[]
    for i,m in enumerate(meta['rows']):
        Y,E=allY[i],allE[i];obs=features_review(Y,E);old=direct_features(Y,E)
        item=extract(Y,E,run['geometry']['energy_ev'],run['geometry']['alpha_i_deg'],metal=run['model_metal'],observations=obs)
        encoded.append(encode(item));rows.append(dict(metadata=m,time_s=m['time_s'],key=m['frame'],observations=obs,old_observations_same_aggregate=old,chunk=c['file'],chunk_row=i))
    dest=folder/(Path(c['file']).stem+'_features.json.gz');dump_gz(dest,rows)
    np.savez_compressed(dest.with_suffix('.npz'),encoded=np.asarray(encoded,np.float32))
    print('REVIEW_MERGED',c['file'],len(rows),flush=True)
    return dict(file=dest.name,sha256=sha(dest),vector_file=dest.with_suffix('.npz').name,vector_sha256=sha(dest.with_suffix('.npz')))


def measure_groups(folder,workers):
    folder=Path(folder);run=read(folder/'run.json')
    with ProcessPoolExecutor(max_workers=workers) as pool:files=list(pool.map(merged_worker,[(str(folder),c,run) for c in run['chunks']]))
    rows=[r for f in files for r in load_gz(folder/f['file'])];route,summary=stages(rows)
    for r,s in zip(rows,route):r['stage_review']=s
    dump_gz(folder/'observations.json.gz',rows);write(folder/'features.json',dict(files=files,rows=len(rows),stage_summary=summary))
    return rows


def write_csv(path,rows):
    from .polymer_dev_v0 import public
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows([{k:public(r.get(k),'该项不适用或证据不足；见对应状态') for k in keys} for r in rows])


def refresh_stages(out):
    """Recompute routing only from immutable measured features, never refit.

    Explicit AF-switch guard correction; preserves prior summary/checksums.
    A receipt prevents this postprocessing from being repeated silently.
    """
    out=Path(out);receipt=out/'stage_revision_record.json';stage_sha=sha(Path(__file__).with_name('polymer_review_stages.py'))
    if receipt.exists():
        if read(receipt)['stage_source_sha256']!=stage_sha:raise ValueError('Review routing source changed again')
        return
    summary=read(out/'feature_summary.json');old_summary=copy.deepcopy(summary)
    for f in summary['raw_feature_files']:
        if sha(out/'raw_features'/f['file'])!=f['sha256']:raise ValueError('Original review features changed')
    raw=[r for f in summary['raw_feature_files'] for r in load_gz(out/'raw_features'/f['file'])]
    times=clock([r['metadata'] for r in raw])
    for r,ti in zip(raw,times):r['time_s']=float(ti)
    rawroute,_=stages(raw)
    merged=load_gz(out/'binned/observations.json.gz');route,_=stages(merged)
    old_observation_sha=sha(out/'binned/observations.json.gz')
    for r,s in zip(merged,route):r['stage_review']=s
    dump_gz(out/'binned/observations.json.gz',merged);membership=read(out/'binned/membership.json')
    with (out/'original_frame_changes.csv').open(encoding='utf-8') as f:paired=list(csv.DictReader(f))
    if len(paired)!=len(raw) or any(p['frame']!=r['metadata']['frame'] for p,r in zip(paired,raw)):raise ValueError('Routing comparison row identity changed')
    for row,s in zip(paired,rawroute):
        row['new_stage_raw']=s['stage'];row['binned_stage']=merged[membership[row['frame']]]['stage_review']['stage']
    write_csv(out/'original_frame_changes.csv',paired)
    if (out/'period_average/observations.json.gz').exists():
        coarse=load_gz(out/'period_average/observations.json.gz')
        for c in coarse:
            votes=Counter(merged[membership[frame]]['stage_review']['stage'] for frame in c['metadata']['raw_members']);label,n=votes.most_common(1)[0]
            c['stage_review']['stage']=label if n/len(c['metadata']['raw_members'])>=.5 else 'stage_unresolved'
            c['stage_review']['stage_reason']='Majority of0.5s routes after same-AF branch reset correction'
            c['stage_review']['stage_votes']=dict(votes)
        dump_gz(out/'period_average/observations.json.gz',coarse)
    summary.update(new_raw_stage_counts=dict(Counter(r['new_stage_raw'] for r in paired)),binned_stage_counts=dict(Counter(r['stage'] for r in route)),
        binned_stage_original_frame_counts=dict(Counter(r['binned_stage'] for r in paired)),stage_source_sha256=stage_sha)
    write(out/'feature_summary.json',summary)
    write(receipt,dict(status='complete',reason='Same-AF feature history reset; no numerical feature, model, period or threshold retuning',
        stage_source_sha256=stage_sha,old_summary=old_summary,old_binned_observation_sha256=old_observation_sha,
        raw_frames=len(raw),merged_bins=len(merged),feature_rerun=False,TEST_read=False))


def conditional_period_average(out,workers=16):
    """Optional detected-period sensitivity output; never changes20s gate.

    Uses an already tested stable high-q period at the original p<=.01 gate.
    The common-Yoneda/high-q confirmation stays false when it failed. This
    output is explicitly conditional on high-q modulation, not sample motion.
    """
    out=Path(out);period=read(out/'periodicity.json')
    if period.get('confirmed'):return
    high=period.get('features',{}).get('highq_log_slope',{})
    records=list(period.get('features',{}).values())
    significant=lambda r:r.get('stable') and r.get('max_frequency_corrected_AR1_p',1)<=.01 and 10<=r.get('period_s',0)<=40
    shared=(len(records)==2 and all(significant(r) for r in records)
            and max(r['period_s'] for r in records)/min(r['period_s'] for r in records)<=1.1)
    if not shared and not significant(high):return
    # The original20s-target gate is retained. All-frequency p-values already
    # cover10-40s; a different stable shared period is reported separately.
    dest=out/('period_average_detected' if shared else 'period_average_highq_conditional')
    if dest.exists():return
    T=float(np.mean([r['period_s'] for r in records])) if shared else high['period_s']
    intake=read(out/'binned/run.json')['source_intake'];run,rows=intake_rows(intake)
    aggregate(intake,dest,run,rows,bin_ids([r['time_s'] for r in rows],seconds=T,early_until=0),nominal_seconds=T)
    coarse=measure_groups(dest,workers);fine=load_gz(out/'binned/observations.json.gz');membership=read(out/'binned/membership.json')
    for c in coarse:
        votes=Counter(fine[membership[f]]['stage_review']['stage'] for f in c['metadata']['raw_members']);label,n=votes.most_common(1)[0]
        c['stage_review']['stage']=label if n/len(c['metadata']['raw_members'])>=.5 else 'stage_unresolved'
        c['stage_review']['stage_reason']='Majority of0.5s routes; average at detected scattering period, not sample-motion correction'
    dump_gz(dest/'observations.json.gz',coarse)
    write(dest/'period_condition.json',dict(period_s=T,source='Previously tested10-40s frequency search, corrected p<=.01 and split stability unchanged',
        joint_period_confirmed=bool(shared),target20s_joint_gate_passed=False,sample_motion_confirmed=False,
        reason='Exploratory different-period average; original20s-target failure retained' if shared else 'Single-highq sensitivity average; original common-feature gate failure retained',
        source_periodicity_sha256=sha(out/'periodicity.json')))


def prepare_revision(intake,old_csv,out,workers=32):
    out=Path(out)
    if out.exists():raise ValueError('Existing revision must be inspected, not overwritten or repeated')
    out.mkdir(parents=True);run,rows=intake_rows(intake)
    write(out/'started.json',dict(intake_sha256=sha(Path(intake)/'run.json'),old_csv_sha256=sha(old_csv),status='running',TEST_read=False))
    with Path(old_csv).open(encoding='utf-8') as f:old={r['frame']:r for r in csv.DictReader(f)}
    if not set(old)<={r['metadata']['frame'] for r in rows}:raise ValueError('Old paired frames absent from current intake')
    tasks=[(str(intake),c['file'],str(out/'raw_features'/(Path(c['file']).stem+'.json.gz')),run['geometry']) for c in run['chunks']]
    with ProcessPoolExecutor(max_workers=workers) as pool:rawfiles=list(pool.map(raw_worker,tasks))
    raw=[r for f in rawfiles for r in load_gz(out/'raw_features'/f['file'])]
    for r,source in zip(raw,rows):r['time_s']=source['time_s']
    rawstage,rawsummary=stages(raw)
    groups=bin_ids([r['time_s'] for r in rows]);aggregate(intake,out/'binned',run,rows,groups,nominal_seconds=.5)
    merged=measure_groups(out/'binned',workers);membership=read(out/'binned/membership.json')
    paired=[]
    for r,s in zip(raw,rawstage):
        frame=r['metadata']['frame'];o=r['observations'];paired_before=frame in old
        before=old.get(frame,dict(q_star_nm_inv=None,feature_status='未纳入旧版抽样',stage='未纳入旧版抽样'))
        after=number(o['q_star_nm_inv']);prev=number(before['q_star_nm_inv'])
        group=merged[membership[frame]]
        paired.append(dict(frame=frame,time_s=r['time_s'],baseline_available=paired_before,old_q_star=prev,new_q_star=after,
            old_feature_status=before['feature_status'],new_feature_status=o['feature_status'],
            newly_resolved=paired_before and not np.isfinite(prev) and np.isfinite(after),lost_resolved=paired_before and np.isfinite(prev) and not np.isfinite(after),
            newly_resolved_below015=paired_before and not np.isfinite(prev) and np.isfinite(after) and after<.15,
            q_support_lower_nm_inv=o['q_support_lower_nm_inv'],old_stage=before['stage'],new_stage_raw=s['stage'],
            binned_stage=group['stage_review']['stage'],bin_id=membership[frame],raw_sha256=r['metadata']['raw_sha256']))
    write_csv(out/'original_frame_changes.csv',paired)
    period=periodicity(merged);write(out/'periodicity.json',period)
    if period['confirmed']:
        ids=bin_ids([r['time_s'] for r in rows],seconds=period['period_s'],early_until=0)
        aggregate(intake,out/'period_average',run,rows,ids,nominal_seconds=period['period_s']);coarse=measure_groups(out/'period_average',workers)
        for c in coarse:
            votes=Counter(merged[membership[frame]]['stage_review']['stage'] for frame in c['metadata']['raw_members'])
            label,n=votes.most_common(1)[0]
            c['stage_review']['stage']=label if n/len(c['metadata']['raw_members'])>=.5 else 'stage_unresolved'
            c['stage_review']['stage_reason']='Majority of original-frame memberships routed at0.5s; period average does not recreate fine history'
            c['stage_review']['stage_votes']=dict(votes)
        dump_gz(out/'period_average/observations.json.gz',coarse)
    summary=dict(status='features_complete',original_frames=len(rows),merged_bins=len(merged),
        baseline_comparable_original_frames=len(old),
        old_resolved=sum(np.isfinite(r['old_q_star']) for r in paired),new_resolved=sum(np.isfinite(r['new_q_star']) for r in paired),
        new_resolved_on_old_selected_frames=sum(r['baseline_available'] and np.isfinite(r['new_q_star']) for r in paired),
        newly_resolved=sum(r['newly_resolved'] for r in paired),lost_resolved=sum(r['lost_resolved'] for r in paired),
        newly_resolved_below015=sum(r['newly_resolved_below015'] for r in paired),
        old_stage_counts=dict(Counter(r['old_stage'] for r in paired)),new_raw_stage_counts=dict(Counter(r['new_stage_raw'] for r in paired)),
        binned_stage_counts=dict(Counter(r['stage_review']['stage'] for r in merged)),
        binned_stage_original_frame_counts=dict(Counter(r['binned_stage'] for r in paired)),
        same_aggregate_old_resolved=sum(r['old_observations_same_aggregate']['q_star_nm_inv'] is not None for r in merged),
        same_aggregate_new_resolved=sum(r['observations']['q_star_nm_inv'] is not None for r in merged),
        period_confirmed=period['confirmed'],period_average_written=period['confirmed'],
        raw_feature_files=rawfiles,source_sha256=sha(__file__),feature_source_sha256=sha(Path(__file__).with_name('polymer_review_features.py')),
        note='Original fit already started0.05; lowq initialization/support review and aggregation effects separated; boundary-only features excluded from resolved counts')
    write(out/'feature_summary.json',summary);print('REVIEW_FEATURES_COMPLETE',out.name,json.dumps(jsonable({k:v for k,v in summary.items() if k.endswith('resolved') or k.endswith('bins')})),flush=True)
    return summary
