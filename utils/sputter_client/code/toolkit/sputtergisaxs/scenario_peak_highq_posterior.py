'Same-band q*/FWHM/high-q input and complete four-scenario joint posterior.'
import math
from numbers import Integral

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .scenario_chart import CATALOG, COORDINATES, DIM, INDEX, SCHEMA as CHART_SCHEMA
from .scenario_minimal_posterior import (
    DECLARATION as MINIMAL_SPEC, _flags, encode_pair as _minimal_encode,
    prepare as _minimal_prepare)
from .scenario_posterior import ScenarioCandidate

SCHEMA = 'four_scenario_peak_width_highq_joint_1'
INPUT_NAMES = tuple(MINIMAL_SPEC['input_names']) + ('log_highq_ratio','highq_known')
INPUT_DIM = len(INPUT_NAMES)
DECLARATION = dict(schema=SCHEMA,architecture='peak_width_highq_joint',
    input_names=list(INPUT_NAMES),input_dim=INPUT_DIM,
    physical_inputs=['q_star_nm_inv','FWHM_nm_inv','energy_ev','alpha_i_deg','metal','highq_ratio'],
    geometry_transform=dict(MINIMAL_SPEC['geometry_transform']),
    AF_selection=MINIMAL_SPEC['AF_selection'],
    highq_definition='existing same-AF highq_ratio, features[9+b] and known flag[29+b]',
    missing_features='zero encoded log value plus explicit feature-known flag; no family removed',
    excluded_inputs=[v for v in MINIMAL_SPEC['excluded_inputs'] if v!='highq'],
    chart_schema=CHART_SCHEMA,coordinates=list(COORDINATES),continuous_dim=DIM,
    nuisance_strategy='complete hierarchical joint learned under the same original scenario',
    D_primary='exact p(z_D|x,S), every legal C/T/component marginalized; S4 zero',
    nuisance_prior_resampled=False,scenarios_pruned=False,structural_release=False)


def encode_pair(q_star_nm_inv,FWHM_nm_inv,energy_ev,alpha_i_deg,metal,*,
                highq_ratio=None,q_star_known=None,FWHM_known=None,highq_known=None):
    'Encode measured physical features; high-q may be absent independently.'
    base = _minimal_encode(q_star_nm_inv,FWHM_nm_inv,energy_ev,alpha_i_deg,metal,
                           q_star_known=q_star_known,FWHM_known=FWHM_known)
    vector = np.zeros((len(base['vector']),INPUT_DIM),np.float32); vector[:,:7] = base['vector']
    values = np.broadcast_to(np.asarray(highq_ratio,float),(len(vector),))
    known = _flags(values,highq_known,values.shape)
    vector[known,7] = np.log(values[known]); vector[:,8] = known
    if not np.isfinite(vector).all():raise ValueError('Finite same-band high-q encoder required')
    return dict(base,vector=vector,highq_known=known,
        complete_feature_triplet=base['complete_feature_pair'] & known)


def prepare(batch,*,AF_band):
    'Extend the original caller-bound single-band encoding, without band search.'
    base = _minimal_prepare(batch,AF_band=AF_band)
    features = np.asarray(batch['features']); vector = np.zeros((len(features),INPUT_DIM),np.float32)
    vector[:,:7] = base['vector']
    for i,b in enumerate(base['AF_band']):
        if b is None:continue
        flag = features[i,29+b]
        if flag not in (0,1):raise ValueError('Same-AF high-q known flag must remain0/1')
        if flag:
            value = float(features[i,9+b])
            if not np.isfinite(value):raise ValueError('Known same-band high-q log ratio is nonfinite')
            vector[i,7] = value; vector[i,8] = 1.
    if not np.isfinite(vector).all():raise ValueError('Finite9-coordinate feature input required')
    known = vector[:,8].astype(bool)
    return dict(base,vector=vector,highq_known=known,
        complete_feature_triplet=base['complete_feature_pair'] & known,
        highq_selection=DECLARATION['highq_definition'])


class PeakHighqScenarioCandidate(ScenarioCandidate):
    'Nine-input original joint MDN, with an exact sparse-head S-marginal D score.'
    def __init__(self,continuous_dim=DIM,coarse_classes=None,template_classes=1,*,
                 components=4,hidden=128,catalog=None,bank_scenario_prior=None):
        chosen = CATALOG if catalog is None else catalog
        super().__init__('features',continuous_dim,coarse_classes,template_classes,
            components=components,hidden=hidden,catalog=chosen,bank_scenario_prior=bank_scenario_prior)
        if self.dim!=DIM or INDEX['D_nm']!=0:
            raise ValueError('Sparse D marginal requires the original44D chart with D first')
        self.encoder = nn.Sequential(nn.Linear(INPUT_DIM,hidden),nn.SiLU(),nn.Linear(hidden,hidden),nn.SiLU())
        self.architecture = DECLARATION['architecture']; self.input_schema = SCHEMA
        self.validated = False

    def embedding(self,encoded):
        vector = encoded['vector']
        if (not torch.is_tensor(vector) or vector.ndim!=2 or vector.shape[1]!=INPUT_DIM
                or not torch.isfinite(vector).all()):raise ValueError('Finite [batch,9] feature vector required')
        flags = vector[:,[5,6,8]]
        if (((flags!=0)&(flags!=1)).any() or ((vector[:,4]!=0)&(vector[:,4]!=1)).any()
                or (vector[:,[0,1,7]][flags==0]!=0).any()):
            raise ValueError('Explicit missing features/metal flags and zero unknown logs required')
        return self.encoder(vector)

    def conditional_D(self,embedding,coarse,template):
        'Exact logits, D means and L00 from the same full-joint learned output.'
        hidden = self.continuous[1](self.continuous[0](torch.cat([
            embedding,self.coarse_embedding(coarse),self.template_embedding(template)],-1)))
        block = 1+self.dim+self.ntri
        slots = (torch.arange(self.K,device=hidden.device)[:,None]*block
                 +torch.tensor([0,1,1+self.dim],device=hidden.device)[None]).reshape(-1)
        last = self.continuous[2]
        selected = F.linear(hidden,last.weight.index_select(0,slots),last.bias.index_select(0,slots))
        selected = selected.reshape(-1,self.K,3)
        return selected[:,:,0],selected[:,:,1],F.softplus(selected[:,:,2])+.02

    def scenario_D_nll(self,encoded,target,legal_templates=None,*,chunk_size=4096):
        'Proper exact conditional D score, with all original rows and S4 zero.'
        if (not isinstance(chunk_size,Integral) or isinstance(chunk_size,bool) or chunk_size<1):
            raise ValueError('Positive sparse-head chunk size required')
        emb = self.embedding(encoded); pc,pt,legal = self.discrete(emb,legal_templates)
        scenario = target['scenario'].long(); active = target['active'][:,0]
        if (scenario.shape!=(len(emb),) or torch.any((scenario<0)|(scenario>3))
                or not torch.equal(active,scenario!=3)
                or not torch.isfinite(target['z'][active,0]).all()):
            raise ValueError('Original scenario and active first-D chart labels required')
        answer = torch.zeros(len(emb),device=emb.device,dtype=torch.float64)
        for s in range(3):
            rows = torch.nonzero(scenario==s).ravel()
            if not len(rows):continue
            classes = torch.tensor([c['id'] for c in self._scenario_catalog if c['scenario_index']==s],
                                   device=emb.device,dtype=torch.long)
            permitted = legal[rows][:,classes,:]
            local_row,local_c,t = torch.nonzero(permitted,as_tuple=True)
            if not permitted.any(dim=(1,2)).all():raise ValueError('Original scenario has no legal C/T branch')
            norm = torch.logsumexp(pc[rows][:,classes].double(),-1)
            if not torch.isfinite(norm).all():raise ValueError('Conditional scenario probability is nonfinite')
            coarse = classes[local_c]; batch_rows = rows[local_row]
            terms = []
            for start in range(0,len(batch_rows),int(chunk_size)):
                br,cc,tt = (v[start:start+chunk_size] for v in (batch_rows,coarse,t))
                logits,means,sd = self.conditional_D(emb[br],cc,tt)
                logits,means,sd = logits.double(),means.double(),sd.double()
                difference = (target['z'][br,0].double()[:,None]-means)/sd
                normal = -.5*difference.square()-torch.log(sd)-.5*math.log(2*math.pi)
                density = torch.logsumexp(torch.log_softmax(logits,-1)+normal,-1)
                terms.append(pc[br,cc].double()+pt[br,cc,tt].double()
                             -norm[local_row[start:start+chunk_size]]+density)
            dense = torch.full((len(rows),len(classes)*self.T),-torch.inf,device=emb.device,dtype=torch.float64)
            dense[local_row,local_c*self.T+t] = torch.cat(terms)
            answer[rows] = -torch.logsumexp(dense,-1)
        return answer
