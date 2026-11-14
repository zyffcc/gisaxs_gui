'Direct S1--S4 conditional sampling of the complete hierarchical joint draw.'
from numbers import Integral

import numpy as np
import torch

from .scenario_chart import CATALOG, DIM, SCENARIO_NAMES, template_compatibility
from .v2_posterior import HierarchicalCandidate


def _scenario_name(scenario):
    if isinstance(scenario, Integral) and not isinstance(scenario, (bool, np.bool_)):
        if 0 <= int(scenario) < len(SCENARIO_NAMES):
            return SCENARIO_NAMES[int(scenario)]
    elif isinstance(scenario, str) and scenario in SCENARIO_NAMES:
        return scenario
    raise ValueError('Physical scenario must be S1--S4 or its zero-based index')


def _catalog(model, catalog=None):
    chosen = getattr(model, '_scenario_catalog', CATALOG) if catalog is None else catalog
    chosen = tuple(dict(c) for c in chosen)
    if len(chosen) != model.C or any(c.get('id') != i for i, c in enumerate(chosen)):
        raise ValueError('Checkpoint coarse dimensions/IDs differ from the bound scenario catalog')
    if any(c.get('scenario') not in SCENARIO_NAMES for c in chosen):
        raise ValueError('Catalog must explicitly identify physical S1--S4 scenarios')
    if any(not any(c['scenario'] == s for c in chosen) for s in SCENARIO_NAMES):
        raise ValueError('Bound catalog must retain every physical scenario')
    return chosen


def _state(model, encoded, legal_templates=None, *, catalog=None, host_categories=None):
    emb = model.embedding(encoded)
    if legal_templates is not None:
        legal_templates = torch.as_tensor(legal_templates, device=emb.device)
    if host_categories is not None:
        categories = _categories(host_categories, len(emb))
        si_rows = torch.tensor([h == 'Si' for h in categories], device=emb.device)
        if si_rows.any():
            if legal_templates is None:
                legal_templates = torch.ones((len(emb), model.C, model.T),
                                             dtype=torch.bool, device=emb.device)
            elif legal_templates.shape == (len(emb), model.T):
                legal_templates = legal_templates[:, None].expand(-1, model.C, -1)
            legal_templates = legal_templates.clone()


            if legal_templates.shape != (len(emb), model.C, model.T):
                raise ValueError('Template compatibility dimensions differ')
            forbidden = torch.tensor([c['scenario'] in ('S2', 'S3') or c['layer']
                                      for c in catalog], device=emb.device)
            legal_templates &= ~(si_rows[:, None, None] & forbidden[None, :, None])
    pc, pt, legal = model.discrete(emb, legal_templates)
    return emb, pc, pt, legal


def _categories(host_categories, batch):
    if host_categories is None:
        return [None]*batch
    if isinstance(host_categories, str):
        host_categories = [host_categories]*batch
    if len(host_categories) != batch:
        raise ValueError('One optional host category per input frame required')
    return list(host_categories)


def legal_from_metadata(energy_ev, alpha_i_deg, templates, *, host_categories=None,
                        device=None):
    "Build the new chart's [batch,coarse,template] physical applicability mask."
    def array(x):
        if torch.is_tensor(x):
            x = x.detach().cpu().numpy()
        return np.atleast_1d(np.asarray(x, float))
    E, ai = np.broadcast_arrays(array(energy_ev), array(alpha_i_deg))
    if E.ndim != 1 or not len(E):
        raise ValueError('Scalar or one-dimensional batch geometry required')
    categories = _categories(host_categories, len(E))
    rows = [template_compatibility(float(e), float(a), templates, host_category=h)
            for e, a, h in zip(E, ai, categories)]
    return torch.as_tensor(np.stack(rows), dtype=torch.bool, device=device)


def _conditional(model, state, scenario, n, catalog, generator, host_categories):
    if not isinstance(n, Integral) or isinstance(n, (bool, np.bool_)) or n < 1:
        raise ValueError('Positive integer original conditional draw count required')
    n = int(n)
    name = _scenario_name(scenario); scene_index = SCENARIO_NAMES.index(name)
    emb, pc, pt, legal = state
    batch = len(emb); categories = _categories(host_categories, batch)
    classes = [c['id'] for c in catalog if c['scenario'] == name]
    ix_classes = torch.tensor(classes, device=emb.device, dtype=torch.long)
    available = legal[:, ix_classes].any(dim=(1, 2))


    z = torch.full((batch, n, model.dim), torch.nan, dtype=emb.dtype, device=emb.device)
    coarse = torch.full((batch, n), -1, dtype=torch.long, device=emb.device)
    template = torch.full_like(coarse, -1); component = torch.full_like(coarse, -1)
    scenario_index = torch.full_like(coarse, -1)
    reasons = [None if bool(ok) else
               ('known Si: S2/S3 are not applicable' if h == 'Si' and name in ('S2', 'S3')
                else 'no legal coarse variant/template for this scenario')
               for ok, h in zip(available.detach().cpu().tolist(), categories)]
    if available.any():
        rows = torch.nonzero(available).ravel()


        logp = torch.log_softmax(pc[rows][:, ix_classes], dim=-1)
        local = torch.multinomial(logp.exp(), n, replacement=True, generator=generator)
        cc = ix_classes[local]
        bb = rows[:, None].expand(-1, n)
        tt = torch.multinomial(pt[bb, cc].exp().reshape(-1, model.T), 1,
                               generator=generator).reshape(cc.shape)
        repeated = emb[rows, None].expand(-1, n, -1).reshape(-1, emb.shape[-1])
        mix_logits, means, factors = model.conditional(repeated, cc.ravel(), tt.ravel())
        kk = torch.multinomial(torch.softmax(mix_logits, -1), 1,
                               generator=generator).ravel()
        draw_ix = torch.arange(len(repeated), device=emb.device)
        mean = means[draw_ix, kk]; L = factors[draw_ix, kk]
        eps = torch.randn(mean.shape, device=mean.device, dtype=mean.dtype, generator=generator)
        zz = mean+(L@eps[..., None]).squeeze(-1)
        z[rows] = zz.reshape(len(rows), n, model.dim)
        coarse[rows] = cc; template[rows] = tt
        component[rows] = kk.reshape(cc.shape); scenario_index[rows] = scene_index
    return dict(z=z, coarse=coarse, template=template, component=component,
        embedding=emb, scenario=name, scenario_index=scenario_index,
        applicable=available, not_applicable_reason=reasons,
        requested_draws=n, original_draw_count=available.to(torch.long)*n,
        direct_scenario_conditioning=True, unconditional_rejection_used=False,
        conditioned_outside_D=False, nuisance_prior_resampled=False,
        validated=bool(getattr(model, 'validated', False)), structural_release=False)


def _probabilities(model, state, catalog, bank_scenario_prior=None):
    emb, pc, _, legal = state
    prior = getattr(model, '_bank_scenario_prior', None) if bank_scenario_prior is None else bank_scenario_prior
    if prior is None:
        prior = [.25]*len(SCENARIO_NAMES)
    if isinstance(prior, dict):
        if set(prior) != set(SCENARIO_NAMES):
            raise ValueError('One bank prior weight for every S1--S4 required')
        prior = [prior[s] for s in SCENARIO_NAMES]
    prior = torch.as_tensor(prior, device=emb.device, dtype=emb.dtype)
    if prior.shape != (len(SCENARIO_NAMES),) or not torch.isfinite(prior).all() or (prior <= 0).any():
        raise ValueError('Finite strictly positive bank scenario prior required')
    prior = prior/prior.sum()
    scene_lp = []; available = []
    for name in SCENARIO_NAMES:
        classes = [c['id'] for c in catalog if c['scenario'] == name]
        scene_lp.append(torch.logsumexp(pc[:, classes], dim=-1))
        available.append(legal[:, classes].any(dim=(1, 2)))
    scene_lp = torch.stack(scene_lp, dim=-1)
    applicable = torch.stack(available, dim=-1)
    equal_prior_lp = torch.log_softmax(scene_lp-torch.log(prior)[None], dim=-1)
    return dict(probabilities=equal_prior_lp.exp(),
        trained_bank_probabilities=scene_lp.exp(), applicable=applicable,
        scenario_names=SCENARIO_NAMES, bank_scenario_prior=prior,
        requested_scenario_prior='equal across applicable physical scenarios',
        probabilities_are_reference_only=True, scenarios_pruned=False,
        validated=bool(getattr(model, 'validated', False)), structural_release=False)


@torch.no_grad()
def sample_conditional(model, encoded, scenario, n=256, *, legal_templates=None,
                       generator=None, catalog=None, host_categories=None):
    'Draw every nuisance jointly under one directly fixed physical scenario.'
    chosen = _catalog(model, catalog)
    return _conditional(model, _state(model, encoded, legal_templates,
                        catalog=chosen, host_categories=host_categories), scenario,
                        n, chosen, generator, host_categories)


@torch.no_grad()
def scenario_probabilities(model, encoded, *, legal_templates=None, catalog=None,
                           bank_scenario_prior=None, host_categories=None):
    'Equal-S-prior reference comparison, with no winner-only output policy.'
    chosen = _catalog(model, catalog)
    return _probabilities(model, _state(model, encoded, legal_templates,
                          catalog=chosen, host_categories=host_categories), chosen,
                          bank_scenario_prior)


@torch.no_grad()
def sample_all_applicable(model, encoded, n=256, *, legal_templates=None,
                          generator=None, catalog=None, host_categories=None,
                          bank_scenario_prior=None):
    'Retain four result entries and sample every applicable S independently.'
    chosen = _catalog(model, catalog)
    state = _state(model, encoded, legal_templates, catalog=chosen,
                   host_categories=host_categories)
    results = {s: _conditional(model, state, s, n, chosen, generator, host_categories)
               for s in SCENARIO_NAMES}
    return dict(samples=results,
        scenario_comparison=_probabilities(model, state, chosen, bank_scenario_prior),
        all_physical_scenarios_retained=True, scenarios_pruned=False,
        validated=bool(getattr(model, 'validated', False)), structural_release=False)


class ScenarioCandidate(HierarchicalCandidate):
    'Weight-compatible subclass adding fixed-S methods to the candidate MDN.'
    def __init__(self, mode, continuous_dim=None, coarse_classes=None, template_classes=1,
                 *, components=4, hidden=128, catalog=None, bank_scenario_prior=None):
        chosen = tuple(dict(c) for c in (CATALOG if catalog is None else catalog))
        dim = DIM if continuous_dim is None else continuous_dim
        count = len(chosen) if coarse_classes is None else coarse_classes
        super().__init__(mode, dim, count, template_classes, components=components, hidden=hidden)
        self._scenario_catalog = _catalog(self, chosen)

        self._bank_scenario_prior = bank_scenario_prior

    def sample_conditional(self, encoded, scenario, n=256, *, legal_templates=None,
                           generator=None, host_categories=None):
        return sample_conditional(self, encoded, scenario, n, legal_templates=legal_templates,
                                  generator=generator, host_categories=host_categories)

    def scenario_probabilities(self, encoded, *, legal_templates=None, bank_scenario_prior=None,
                               host_categories=None):
        return scenario_probabilities(self, encoded, legal_templates=legal_templates,
                                      bank_scenario_prior=bank_scenario_prior,
                                      host_categories=host_categories)

    def sample_all_applicable(self, encoded, n=256, *, legal_templates=None,
                             generator=None, host_categories=None, bank_scenario_prior=None):
        return sample_all_applicable(self, encoded, n, legal_templates=legal_templates,
            generator=generator, host_categories=host_categories,
            bank_scenario_prior=bank_scenario_prior)
