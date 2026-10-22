"""Load sealed NR shards and one frozen TRAIN input policy, never TEST.

Per-module experts are a first trained component, not a completed four-stage
classifier. Module-specific input masks must not leak the true class into a
cross-module classifier; that classifier requires a common input policy.
"""
from __future__ import annotations
from pathlib import Path
import json,hashlib,copy
import numpy as np
from .neural_refine_input import self_normalize_cuts,NATIVE_AF,Q
from .neural_refine_chart import NAMES,branch


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def policy_transform_sha(policy):
    keys=('module','horizontal','vertical','direct_features','normalization','highq_only','common_signal_indices')
    value={k:policy[k] for k in keys}
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def verify_policy(policy,run,manifest,*,development_snapshot=False):
    score=policy['domain_balanced_accuracy']
    if not np.isfinite(score) or (score>.70 and not development_snapshot) or not policy['frozen_on_TRAIN'] or policy['module']!=run['identity']['module']:
        raise ValueError('Finite passing frozen actual-input TRAIN policy required')
    if development_snapshot and (policy.get('formal_eligible',True) or policy.get('training_scope')!='sealed_subset_development_only'):
        raise ValueError('Domain-failed trial must be explicitly development-only; formal gate unchanged')
    evidence=policy['domain_evidence'];path=Path(evidence['path']);real_path=Path(evidence['real_cache_path'])
    if sha(path)!=evidence['sha256'] or sha(real_path)!=evidence['real_cache_sha256']:
        raise ValueError('Actual domain/real input evidence changed')
    report=json.loads(path.read_text());real_sha=run['identity']['real_cache_sha256']
    if report['policy_transform_sha256']!=policy_transform_sha(policy):raise ValueError('Actual checked input transform differs from training policy')
    if report['status']!='complete' or report['real_sha256']!=real_sha or evidence['real_cache_sha256']!=real_sha:
        raise ValueError('Actual domain evidence refers to different original TRAIN inputs')
    if report['normalization']!=policy['normalization'] or report['bank_sha256'][policy['module']]!=evidence['bank_sha256'] or evidence['bank_sha256']!=sha(manifest):
        raise ValueError('Domain normalization/library source differs')
    folds=[r for r in report['fold_records'] if r['key']==evidence['case_key'] and r.get('cut')==evidence['cut'] and r['status']=='complete']
    if len(folds)<2 or abs(float(np.mean([r['balanced_accuracy'] for r in folds]))-score)>1e-8:
        raise ValueError('Passing score is not the actual multi-fold TRAIN result')
    common=set(policy['common_signal_indices'])
    for row in folds:
        if row['module']!=policy['module'] or row['kept_H']!=policy['horizontal'] or row['kept_V']!=policy.get('vertical',False) or row['kept_direct_features']!=policy['direct_features']:
            raise ValueError('Policy and actual checked input selection differ')
        if not common<=set(row['common_signal_indices']):raise ValueError('Policy reintroduced an unsupported discarded coordinate')
    if not common:raise ValueError('Policy has no shared supported signal coordinates')


def apply_policy(encoded,policy):
    x=np.asarray(encoded,dtype=np.float32).copy();one=x.ndim==1
    if one:x=x[None]
    n=8*len(Q)+2*len(NATIVE_AF)
    if x.shape[1]!=2*n+14:raise ValueError('Original NR input dimensions differ')
    relative=np.sinh(x[:,:n].astype(float));support=x[:,n:2*n]>0;keep=np.zeros(n,bool)
    for b in policy['horizontal']:keep[b*2*len(Q):(b+1)*2*len(Q)]=True
    if policy.get('vertical'):keep[8*len(Q):]=True
    if policy.get('highq_only'):keep[:8*len(Q)]&=np.tile(Q>=.8,8)
    common=np.zeros(n,bool);common[policy['common_signal_indices']]=True;keep&=common
    support&=keep;relative[:,~keep]=0.
    if policy['normalization']=='per_cut':
        x[:,:n]=np.stack([self_normalize_cuts(v,m) for v,m in zip(relative,support)])
    elif policy['normalization']=='global':
        for i in range(len(x)):
            good=support[i]&(relative[i]>0);scale=max(float(np.median(relative[i,good])) if good.any() else 1.,1e-12)
            x[i,:n]=np.arcsinh(relative[i]/scale)
    else:raise ValueError('Frozen NR normalization required')
    x[:,n:2*n]=support
    for j in range(4):
        if j not in policy['direct_features']:x[:,2*n+j]=0.;x[:,2*n+4+j]=0.
    x[:,-3]=0.
    return x[0] if one else x


class NRBank:
    def __init__(self,manifest,policy_file,*,development_snapshot=False):
        manifest=Path(manifest);root=manifest.parent;run=json.loads(manifest.read_text());policy=json.loads(Path(policy_file).read_text())
        if development_snapshot:
            if run['status']!='development_snapshot' or run.get('formal_eligible',True) or not 4000<=run['development_families']<=5000:
                raise ValueError('Explicit sealed 4000--5000-family development snapshot required')
            expected=run['development_families']
        elif run['status']!='complete' or not run['production_minimum_met'] or run['identity']['families']<20000:
            raise ValueError('Complete>=20000-family module library required; no pilot training shortcut')
        else:expected=run['identity']['families']
        for name,digest in run['identity']['code_sha256'].items():
            if sha(Path(__file__).resolve().parent/name)!=digest:raise ValueError('Library physical/input/target code differs from current expert source: '+name)
        verify_policy(policy,run,manifest,development_snapshot=development_snapshot)
        xs=[];zs=[];acts=[];labels=[];rows=[];indices=[];catalog={}
        for shard in run['shards']:
            receipt=root/shard['receipt'];arr=root/shard['array_file']
            if sha(receipt)!=shard['receipt_sha256'] or sha(arr)!=shard['array_sha256']:raise ValueError('Completed library shard changed')
            r=json.loads(receipt.read_text());rr=r['rows']
            if r['identity']!=run['identity'] or any(v['split'] not in ('TRAIN','CAL') for v in rr):raise ValueError('Original TRAIN/CAL binding differs; TEST never opened')
            with np.load(arr,allow_pickle=False) as z:
                if len(z['encoded'])!=len(rr):raise ValueError('Original array/receipt mapping differs')
                xs.append(apply_policy(z['encoded'],policy));zs.append(z['target'].astype(np.float32));acts.append(z['target_active'].astype(bool))
            for row in rr:
                key=tuple(row['branch'])
                if key not in catalog:catalog[key]=len(catalog)
                if tuple(branch(row['parameters']))!=key:raise ValueError('Physical branch label differs')
                labels.append(catalog[key]);rows.append(row);indices.append(row['split'])
        self.X=np.concatenate(xs);self.Z=np.concatenate(zs);self.active=np.concatenate(acts);self.coarse=np.array(labels,dtype=np.int64);self.split=np.array(indices)
        self.catalog=tuple(catalog);self.rows=rows;train=self.split=='TRAIN'
        prototypes=[];prototype_active=[]
        for label in range(len(self.catalog)):
            i=int(np.flatnonzero(train&(self.coarse==label))[0]);prototype=copy.deepcopy(rows[i]['parameters'])
            for key in ('truth','family_id','family_index','seed','D_anchor_source','donor_metal'):prototype.pop(key,None)
            prototypes.append(prototype);prototype_active.append(self.active[i].tolist())
        mean=np.zeros(len(NAMES));scale=np.ones(len(NAMES))
        for j in range(len(NAMES)):
            use=train&self.active[:,j]
            if use.any():mean[j]=self.Z[use,j].mean();scale[j]=max(float(self.Z[use,j].std()),.05)
        self.mean=mean.astype(np.float32);self.scale=scale.astype(np.float32);self.Z=(self.Z-self.mean)/self.scale;self.Z[~self.active]=0.
        if len(self.X)!=expected or len(set(r['family_id'] for r in rows))!=len(rows):raise ValueError('Original family denominator differs')
        self.receipt=dict(manifest_sha256=sha(manifest),input_policy_sha256=sha(policy_file),module=run['identity']['module'],
            training_scope='sealed_subset_development_only' if development_snapshot else 'complete_production_bank',formal_eligible=not development_snapshot,
            actual_domain_balanced_accuracy=policy['domain_balanced_accuracy'],actual_domain_gate70_passed=bool(policy['domain_balanced_accuracy']<=.70),
            TRAIN_families=int(train.sum()),CAL_families=int((~train).sum()),branches=[list(b) for b in self.catalog],input_dim=self.X.shape[1],targets=list(NAMES),
            normalization_mean=self.mean.tolist(),normalization_scale=self.scale.tolist(),TEST_read=False,
            branch_prototypes=prototypes,branch_active_masks=prototype_active,
            prototype_scope='branch/nuisance starts only; original truth and family identity removed; measured geometry/metal replaced at inference',
            stage_classifier='Not trained by a single-module expert; no class-dependent policy leak permitted')

    def batches(self,split,epoch,batch,seed):
        ids=np.flatnonzero(self.split==split)
        if split=='TRAIN':np.random.default_rng(seed+epoch).shuffle(ids)
        for start in range(0,len(ids),batch):yield ids[start:start+batch]
