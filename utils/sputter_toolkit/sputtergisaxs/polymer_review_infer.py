"""Conditional polymer/TS development routing, export and correlated ranges.

The TS spectral L is never relabelled as island D. w=fL and g=(1-f)L are
declared lamellar surrogates only, not direct TS truth or recovered topology.
"""
from __future__ import annotations
from pathlib import Path
import copy,hashlib,json,time
import numpy as np
from .neural_refine_infer import NRExpert
from .neural_refine_training import apply_policy,sha
from .neural_refine_calibration import extract_mixture,physical
from .neural_refine_chart import NAMES
from .neural_refine_fullmodes import component_starts
from .neural_refine_report import read,write
from .polymer_review_data import load_gz,dump_gz,write_csv
from .polymer_review_stages import number
from .polymer_dev_v0 import public

POLY=('D_nm','R_number_mean_nm','radius_sigma_log','burial_base_depth_nm','height_nm','polymer_thickness_nm')
TS=('spectral_domain_nm','width_rel','metal_fraction','film_thickness_nm','depth_corr_nm','metal_band_lamellar_nm','gap_lamellar_nm')
LATE={'coalescence_candidate','coalescence_or_film_candidate','continuous_film_candidate'}


def candidates(expert,x,geometry,metal,keys):
    mix,active=extract_mixture(expert,apply_policy(x,expert.policy));names=[NAMES[j] for j in active]
    mean=expert.net.chart_mean.cpu().numpy()[active];scale=expert.net.chart_scale.cpu().numpy()[active]
    sd=np.sqrt((mix['chol']**2).sum(-1));out=[]
    for i,key in enumerate(keys):
        one={k:v[i:i+1] for k,v in mix.items()};modes=component_starts(expert,one,active,
            energy_ev=geometry['energy_ev'],alpha_i_deg=geometry['alpha_i_deg'],metal=metal)
        modes.sort(key=lambda m:m['probability'],reverse=True);modes=modes[:3]
        for m in modes:
            b,k=map(int,m['mode_id'].split(':'));mu=mix['mu'][i,b,k];s=sd[i,b,k]
            qq=physical(np.stack([mu-1.95996398454*s,mu-.99445788321*s,mu+.99445788321*s,mu+1.95996398454*s]),names,mean,scale)
            m['intervals']={name:dict(interval68=qq[1:3,j].tolist(),interval95=qq[[0,3],j].tolist()) for j,name in enumerate(names)}
            if expert.declaration['bank']['module']=='polymer_embed':
                m['values']['height_nm']=m['values']['R_number_mean_nm'];m['intervals']['height_nm']=copy.deepcopy(m['intervals']['R_number_mean_nm'])
            else:
                L=m['values']['spectral_domain_nm'];f=m['values']['metal_fraction']
                m['values'].update(metal_band_lamellar_nm=f*L,gap_lamellar_nm=(1-f)*L)
                seed=int(hashlib.sha256((str(key)+m['mode_id']).encode()).hexdigest()[:8],16)
                rng=np.random.default_rng(seed);z=mu+rng.normal(size=(2048,len(active)))@mix['chol'][i,b,k].T
                v=physical(z,names,mean,scale);length=v[:,names.index('spectral_domain_nm')];fraction=v[:,names.index('metal_fraction')]
                for name,values in [('metal_band_lamellar_nm',length*fraction),('gap_lamellar_nm',length*(1-fraction))]:
                    m['intervals'][name]=dict(interval68=np.quantile(values,[.16,.84]).tolist(),interval95=np.quantile(values,[.025,.975]).tolist())
                m['width_gap_condition']='Lamellar surrogate w=fL,g=(1-f)L; TS spectrum does not uniquely define physical metal band/gap'
                m['width_gap_interval']='2048 correlated original component draws; no rejection or truncation; not empirically calibrated'
            m['interval_scope']='Raw component marginal conditional68/95%; not CAL-certified new-input or refined intervals'
        out.append(modes)
    return out


def export_row(r,p,run):
    """One frames.csv row for window r and its network prediction p (unchanged columns)."""
    m=r['metadata'];o=r['observations'];s=r['stage_review'];late=p['module']=='TS_layer';peak=o['feature_status'] in ('bilateral','one_side')
    row=dict(frame=r['key'],time_s=r['time_s'],first_time_s=m['first_time_s'],last_time_s=m['last_time_s'],
        raw_frames=m['raw_frames'],first_raw_frame=m['first_raw_frame'],last_raw_frame=m['last_raw_frame'],
        exposure_s=m['exposure_s'],actual_time_span_s=m['actual_time_span_s'],selected_counts=m['selected_counts'],
        nominal_bin_seconds=m['nominal_bin_seconds'],covered_duration_s=m['covered_duration_s'],complete_nominal_bin=m['complete_nominal_bin'],
        intensity_rate=o['intensity_rate'],intensity_rate_SE=o['intensity_rate_SE'],
        stage=s['stage'],stage_reason=s['stage_reason'],stage_scope=s['stage_scope'],module=p['module'],
        q_star_nm_inv=o['q_star_nm_inv'],D_corr_nm=o['D_corr_nm'],FWHM_nm_inv=o['FWHM_nm_inv'],
        q_star_fit_se_nm_inv=o['q_star_fit_se_nm_inv'],q_support_lower_nm_inv=o['q_support_lower_nm_inv'],
        lowq_boundary_candidates=len(o['lowq_boundary_candidates']),feature_status=o['feature_status'],feature_AF_band=o['feature_AF_band'],
        Yoneda_deg=o['Yoneda_deg'],Yoneda_shift_deg=s['Yoneda_shift_deg'],Yoneda_old_q015_deg=o['Yoneda_old_q015_deg'],
        Yoneda_smoothed_shift_deg=s['Yoneda_smoothed_shift_deg'],ridge_platform=s['ridge_platform'],
        peak_lowq_motion_history=s['peak_lowq_motion_history'],highq_log_slope=o['highq_log_slope'],
        highq_reason=o['highq_reason'],system_note=run['system_note'],metal=run['metal'],model_metal=run['model_metal'],
        Cu_scope='Cu以Ag模型近似；电子密度0.89假设，光学差异未认证' if run['metal']=='Cu' else 'Native Ag/Au model label',
        energy_ev=run['geometry']['energy_ev'],wavelength_nm=1239.841984/run['geometry']['energy_ev'],
        sdd_mm=run['geometry']['sdd_mm'],beam_col=run['geometry']['db_col'],beam_row=run['geometry']['db_row'],alpha_i_deg=run['geometry']['alpha_i_deg'],
        checkpoint_sha256=p['checkpoint_sha256'],policy_sha256=p['policy_sha256'],raw_members_sha256=m['raw_sha256'],
        version='聚合物开发版v0—复查修订r1',new_content='低q初值/支持复查；0.5s计数合并（0–5s为0.1s）；时序阶段；后期TS',
        interval_scope='分量原始条件68/95%范围；输入特征已修订，未由原CAL覆盖认证',
        omitted_mode_probability=1-sum(a['probability'] for a in p['modes']),
        direct_grade='直接观测；总散射，未唯一分离宿主/金属',
        D_grade='后期TS定义不同，不输出岛D' if late else 'D网络-有峰；开发条件' if peak else '无可分辨峰，不输出D',
        R_grade='TS不定义岛R' if late else 'R网络＋范围；开发条件',burial_height_grade='TS不定义粒子埋深/高度' if late else '埋深与高度仅条件；h=R',
        TS_definition='L=2π/q0为TS谱特征长度；不是岛间距D',
        width_gap_definition='层状几何替代量w=fL,g=(1-f)L；不是TS正向真值或已测金属宽度/缝隙',
        refinement_status='本轮未重做物理精修；旧逐帧未收敛评分不适用于合并帧',
        compatibility_5pct='未计算本轮合并帧曲线；不能移用旧逐帧相容度',
        distinguish='SEM/AFM或截面TEM检验连接性/带宽/缝隙；多角GISAXS与XRR约束层厚/埋深；位置日志核对周期来源')
    for rank,mode in enumerate(p['modes'],1):
        prefix=f'mode{rank}_';row[prefix+'probability']=mode['probability'];row[prefix+'status']=mode['status'];row[prefix+'reason']=mode['reason']
        for name in POLY+TS:
            reason='TS后期不适用岛参数' if late and name in POLY else '当前为聚合物岛候选，TS参数不适用' if not late and name in TS else '无可分辨峰，不输出D' if name=='D_nm' and not peak else None
            row[prefix+name]=reason or mode['values'].get(name,'此正向未包含此自由参数')
            for level in ('68','95'):
                interval=mode['intervals'].get(name,{}).get('interval'+level,['条件范围未定义']*2)
                row[prefix+name+'_lo'+level]=reason or interval[0];row[prefix+name+'_hi'+level]=reason or interval[1]
    return row


def _experts(polymer,ts,device):
    poly=NRExpert(Path(polymer)/'model/best.pt',Path(polymer)/'policy.json',device=device)
    film=NRExpert(Path(ts)/'model/best.pt',Path(ts)/'policy.json',device=device)
    if poly.declaration['bank']['module']!='polymer_embed' or film.declaration['bank']['module']!='TS_layer':raise ValueError('Bound polymer/TS modules required')
    return poly,film


def _vectors(folder,frun):
    arrays=[]
    for f in frun['files']:
        if sha(folder/f['file'])!=f['sha256'] or sha(folder/f['vector_file'])!=f['vector_sha256']:raise ValueError('Feature binding differs')
        arrays.append(np.load(folder/f['vector_file'])['encoded'])
    return np.concatenate(arrays)


def predict(rows,x,run,poly,film,*,alternate=False):
    """Network modes per window: the routed module, or with alternate=True the other module.

    Routing (and batching) of the routed module is unchanged; the alternate
    run only lets a later time-consistent path use the other network.
    """
    pred=[None]*len(rows)
    for late,expert in [(False,poly),(True,film)]:
        ids=[i for i,r in enumerate(rows) if ((r['stage_review']['stage'] in LATE)!=alternate)==late]
        for start_i in range(0,len(ids),128):
            batch=ids[start_i:start_i+128];modes=candidates(expert,x[batch],run['geometry'],run['model_metal'],[rows[i]['key'] for i in batch])
            for i,m in zip(batch,modes):pred[i]=dict(module=expert.declaration['bank']['module'],modes=m,
                checkpoint_sha256=expert.checkpoint_sha256,policy_sha256=expert.declaration['bank']['input_policy_sha256'])
    return pred


def infer_alternate(folder,polymer,ts,*,device='cpu',experts=None):
    """Write frames_alt.csv: every window through the network its stage rule did NOT route it to."""
    import torch
    torch.set_num_threads(1)
    folder=Path(folder);run=read(folder/'run.json');frun=read(folder/'features.json');rows=load_gz(folder/'observations.json.gz')
    poly,film=experts or _experts(polymer,ts,device)
    alt=predict(rows,_vectors(folder,frun),run,poly,film,alternate=True)
    write_csv(folder/'frames_alt.csv',[export_row(r,p,run) for r,p in zip(rows,alt)])
    return folder/'frames_alt.csv'


def infer_groups(folder,polymer,ts,*,device='cuda'):
    import torch
    torch.set_num_threads(1)
    folder=Path(folder);run=read(folder/'run.json');frun=read(folder/'features.json');rows=load_gz(folder/'observations.json.gz')
    if (folder/'inference.json').exists():raise ValueError('Existing inference must be checked, not overwritten')
    poly,film=_experts(polymer,ts,device)
    x=_vectors(folder,frun);start=time.monotonic()
    pred=predict(rows,x,run,poly,film)
    exported=[];posterior=[]
    for r,p in zip(rows,pred):
        row=export_row(r,p,run)
        exported.append(row);posterior.append(dict(key=r['key'],**p))
    write_csv(folder/'frames.csv',exported);dump_gz(folder/'network.json.gz',public(posterior))
    alt=predict(rows,x,run,poly,film,alternate=True)
    write_csv(folder/'frames_alt.csv',[export_row(r,p,run) for r,p in zip(rows,alt)])
    write(folder/'inference.json',dict(status='complete',rows=len(rows),device=device,elapsed_s=time.monotonic()-start,
        TS_rows=sum(r['module']=='TS_layer' for r in exported),polymer_rows=sum(r['module']=='polymer_embed' for r in exported),
        polymer_checkpoint_sha256=poly.checkpoint_sha256,TS_checkpoint_sha256=film.checkpoint_sha256,
        csv_sha256=sha(folder/'frames.csv'),alternate_csv_sha256=sha(folder/'frames_alt.csv'),
        alternate_scope='frames_alt.csv: every window through the other network (for the physical path); not a routing decision',TEST_read=False,Dcorr_truncation=False,
        physical_D_domain_nm=[2,120],TS_spectral_domain_nm=[3,200],
        source_sha256=sha(__file__),input_revision_validation=False,
        stage='Observation rules, no cross-module neural stage calibration',refinement_run=False,
        metadata=run))
    return exported
