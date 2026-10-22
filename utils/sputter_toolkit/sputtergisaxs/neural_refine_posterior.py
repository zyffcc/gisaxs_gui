"""NR cut-input stage classifier and branch-conditional joint MDN.

Reuses the existing, independently tested full-covariance mixture arithmetic;
no frozen network weights or old physics decoder are changed. This definition
is not a trained or calibrated model. Branch labels describe the generator,
not a microscopy proof of a real frame's stage or connectivity.
"""
from __future__ import annotations
import torch
from torch import nn
from .v2_posterior import HierarchicalCandidate,mixture_marginal
from .neural_refine_chart import NAMES


class NRPosterior(HierarchicalCandidate):
    def __init__(self,input_dim,branches,*,hidden=192,components=4,chart_mean=None,chart_scale=None):
        super().__init__('features',len(NAMES),len(branches),1,hidden=hidden,components=components)
        self.input_dim=int(input_dim);self.branches=tuple(tuple(b) for b in branches)
        self.encoder=nn.Sequential(nn.Linear(input_dim,hidden),nn.SiLU(),nn.Linear(hidden,hidden),nn.SiLU())
        self.stage_logits=nn.Linear(hidden,4)
        mean=torch.zeros(len(NAMES)) if chart_mean is None else torch.as_tensor(chart_mean,dtype=torch.float32)
        scale=torch.ones(len(NAMES)) if chart_scale is None else torch.as_tensor(chart_scale,dtype=torch.float32)
        if mean.shape!=(len(NAMES),) or scale.shape!=(len(NAMES),) or not torch.isfinite(mean).all() or not torch.isfinite(scale).all() or (scale<=0).any():
            raise ValueError('Finite TRAIN chart normalization with positive scales required')
        self.register_buffer('chart_mean',mean.clone());self.register_buffer('chart_scale',scale.clone())
        self.stage_head_trained=False
        self.validation_scope='untrained_NR_definition_not_structural_validation'

    def objective(self,encoded,z,active,coarse,stage,*,legal=None,stage_known=None):
        emb=self.embedding(encoded);pc,_,allowed=self.discrete(emb,legal)
        ix=torch.arange(len(emb),device=emb.device);template=torch.zeros_like(coarse)
        if not allowed[ix,coarse,template].all():raise ValueError('Original module/branch incompatible with declared input condition')
        mix=self.conditional(emb,coarse,template)
        continuous=mixture_marginal(*(v.double() for v in mix),z.double(),active)
        joint=-pc[ix,coarse].double()-continuous
        known=(stage>=0) if stage_known is None else stage_known.bool()&(stage>=0)
        stages=torch.zeros_like(joint)
        if known.any():stages[known]=torch.nn.functional.cross_entropy(self.stage_logits(emb[known]),stage[known],reduction='none').double()
        return dict(joint=joint,stage=stages,total=joint+stages)

    @torch.no_grad()
    def sample_branch(self,encoded,coarse,n=256,*,generator=None,legal=None):
        """All requested branches remain available, not just maximum mass.

        Returns original untruncated chart samples. The caller records invalid
        physical mass and calibrates actual displayed candidate interval unions.
        """
        if n<1 or not 0<=coarse<self.C:raise ValueError('A legal branch and positive draw count required')
        emb=self.embedding(encoded);batch=len(emb);pc,_,allowed=self.discrete(emb,legal)
        if not allowed[:,coarse,0].all():raise ValueError('Requested branch incompatible with the supplied material condition')
        expanded=emb[:,None].expand(-1,n,-1).reshape(-1,emb.shape[-1])
        branch=torch.full((batch*n,),coarse,device=emb.device,dtype=torch.long)
        logits,mu,L=self.conditional(expanded,branch,torch.zeros_like(branch))
        k=torch.multinomial(torch.softmax(logits,-1),1,generator=generator).ravel();i=torch.arange(len(k),device=k.device)
        eps=torch.randn((len(k),self.dim),device=emb.device,dtype=mu.dtype,generator=generator)
        z_standardized=mu[i,k]+(L[i,k]@eps[...,None]).squeeze(-1)
        z=z_standardized*self.chart_scale+self.chart_mean
        return dict(z=z.reshape(batch,n,self.dim),z_standardized=z_standardized.reshape(batch,n,self.dim),
            chart_normalization_applied=True,component=k.reshape(batch,n),branch=coarse,
            branch_probability=pc[:,coarse].exp(),stage_probability=torch.softmax(self.stage_logits(emb),-1) if self.stage_head_trained else None,
            stage_probability_reason='conditional_module_only_stage_classifier_not_trained' if not self.stage_head_trained else 'model-defined_stage_labels',
            conditioned_outside_D=False,structural_validation=False)
