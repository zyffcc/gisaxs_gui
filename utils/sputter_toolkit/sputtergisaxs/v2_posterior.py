"""Private hierarchical joint mixture for the effective-host v2 candidate.

This statistical module has no physics decoder, trained weight or release
receipt. The final physics schema must declare continuous coordinates and
discrete scene classes. Coordinates used for a public marginal must have the
same meaning in every class (e.g. log D/diameter/height, not true-shape radius).
Shape/scene and empirical template are drawn first, then the complete joint
continuous vector conditioned on both. No nuisance is stitched from a prior;
no outside-D mass is removed before predictive sampling.
"""
import math
import torch
from torch import nn
from .v2_input import MODES


def mixture_marginal(logits, means, factors, target, active):
    """Exact selected-coordinate Gaussian mixture, including empty subsets."""
    n, components, dim = means.shape
    if factors.shape != (n, components, dim, dim) or target.shape != (n, dim) or active.shape != target.shape:
        raise ValueError('Joint mixture dimensions differ')
    if active.dtype != torch.bool or not torch.isfinite(target[active]).all():
        raise ValueError('Active coordinates require finite values and a Boolean mask')
    lp = torch.log_softmax(logits, -1); result = torch.empty(n, device=target.device, dtype=target.dtype)
    patterns, inverse = torch.unique(active, dim=0, return_inverse=True)
    for j, pattern in enumerate(patterns):
        rows = inverse == j; ix = torch.nonzero(pattern).ravel(); d = len(ix)
        if d == 0:
            result[rows] = torch.logsumexp(lp[rows], -1)
            continue
        if d == dim:
            chol = factors[rows]
        else:
            fac = factors[rows][:, :, ix, :]
            chol = torch.linalg.cholesky(fac @ fac.transpose(-1, -2))
        delta = target[rows][:, None, ix] - means[rows][:, :, ix]
        white = torch.linalg.solve_triangular(chol, delta[..., None], upper=False).squeeze(-1)
        logn = -.5*(white*white).sum(-1) - torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)).sum(-1)
        logn -= .5*d*math.log(2*math.pi)
        result[rows] = torch.logsumexp(lp[rows] + logn, -1)
    return result


class HierarchicalCandidate(nn.Module):
    """Dimension-configurable candidate; final catalog/compatibility are external.

    Template0 is reserved for a parametric host by the eventual schema. Legal
    masks may be [batch,templates] or [batch,coarse,templates]. A coarse class
    with no legal template gets zero probability. All unavailable is an input
    contract error, not silently reassigned to a known material.
    """
    def __init__(self, mode, continuous_dim, coarse_classes, template_classes, *, components=4, hidden=128):
        super().__init__()
        if mode not in MODES or min(continuous_dim, coarse_classes, template_classes, components) < 1:
            raise ValueError('Valid candidate dimensions/input mode required')
        self.mode = mode; self.dim = continuous_dim; self.C = coarse_classes; self.T = template_classes; self.K = components
        self.validated = False
        if mode != 'features':
            self.conv = nn.Sequential(nn.Conv2d(4, 24, 3, stride=(2, 1), padding=1), nn.SiLU(),
                nn.Conv2d(24, 48, 3, stride=(2, 2), padding=1), nn.SiLU(),
                nn.AdaptiveAvgPool2d((4, 10)), nn.Flatten())
        size = 54 if mode in ('features', 'hybrid') else 14
        if mode != 'features': size += 48*4*10
        self.encoder = nn.Sequential(nn.Linear(size, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU())
        self.coarse_logits = nn.Linear(hidden, self.C)
        self.coarse_embedding = nn.Embedding(self.C, 12); self.template_embedding = nn.Embedding(self.T, 16)
        self.template_logits = nn.Linear(hidden+12, self.T)
        self.ntri = self.dim*(self.dim+1)//2
        self.continuous = nn.Sequential(nn.Linear(hidden+28, hidden), nn.SiLU(),
                                        nn.Linear(hidden, self.K*(1+self.dim+self.ntri)))
        row, col = torch.tril_indices(self.dim, self.dim); self.register_buffer('tril_row', row); self.register_buffer('tril_col', col)
        nn.init.normal_(self.continuous[-1].weight, std=.001); nn.init.zeros_(self.continuous[-1].bias)
        with torch.no_grad():
            bias = self.continuous[-1].bias.reshape(self.K, -1); diag = torch.nonzero(row==col).ravel()
            bias[:, 1+self.dim+diag] = math.log(math.expm1(.7-.02))

    def embedding(self, encoded):
        vector = encoded['vector']
        values = [vector]
        if self.mode != 'features': values.append(self.conv(encoded['image']))
        return self.encoder(torch.cat(values, 1))

    def discrete(self, emb, legal_templates=None):
        n = len(emb); c = torch.arange(self.C, device=emb.device)
        ce = self.coarse_embedding(c)[None].expand(n, -1, -1)
        logits = self.template_logits(torch.cat([emb[:, None].expand(-1, self.C, -1), ce], -1))
        legal = torch.ones_like(logits, dtype=torch.bool) if legal_templates is None else legal_templates
        if legal.dtype != torch.bool:
            raise ValueError('Boolean scene/template compatibility required')
        if legal.shape == (n, self.T): legal = legal[:, None].expand(-1, self.C, -1)
        if legal.shape != logits.shape:
            raise ValueError('Template compatibility dimensions differ')
        valid = legal.any(-1)
        if not valid.any(-1).all():
            raise ValueError('No compatible scene/template; schema must retain parametric fallback')
        # Unreachable coarse classes need finite conditional probabilities to
        # avoid (-inf + NaN) when their zero mass is marginalized.
        safe = legal.clone(); safe[:, :, 0] |= ~valid
        tpl_lp = torch.log_softmax(logits.masked_fill(~safe, -torch.inf), -1)
        coarse_lp = torch.log_softmax(self.coarse_logits(emb).masked_fill(~valid, -torch.inf), -1)
        return coarse_lp, tpl_lp, legal

    def conditional(self, emb, coarse, template):
        raw = self.continuous(torch.cat([emb, self.coarse_embedding(coarse), self.template_embedding(template)], -1))
        raw = raw.reshape(-1, self.K, 1+self.dim+self.ntri); tri = raw[:, :, 1+self.dim:]
        L = torch.zeros(len(raw), self.K, self.dim, self.dim, device=raw.device, dtype=raw.dtype)
        L[:, :, self.tril_row, self.tril_col] = tri
        diag = self.tril_row == self.tril_col; ix = self.tril_row[diag]
        L[:, :, ix, ix] = torch.nn.functional.softplus(tri[:, :, diag])+.02
        return raw[:, :, 0], raw[:, :, 1:1+self.dim], L

    def private_log_prob(self, encoded, target, coarse, template, active, *, legal_templates=None):
        emb = self.embedding(encoded); pc, pt, legal = self.discrete(emb, legal_templates)
        b = torch.arange(len(emb), device=emb.device)
        if not legal[b, coarse, template].all():
            raise ValueError('Training label violates registered scene compatibility')
        mix = self.conditional(emb, coarse, template)
        return pc[b, coarse] + pt[b, coarse, template] + mixture_marginal(*mix, target, active)

    def marginal_log_prob(self, encoded, target, active, *, legal_templates=None):
        """Marginalize every discrete branch and inactive continuous coordinate.

        This is a score in the final declared target coordinates, not by itself
        a density in physical units. Decoder Jacobians, if needed, are external.
        Branches are streamed to avoid materializing batch*C*T covariance heads.
        """
        emb = self.embedding(encoded); pc, pt, legal = self.discrete(emb, legal_templates)
        answer = torch.full((len(emb),), -torch.inf, device=emb.device, dtype=emb.dtype)
        for c in range(self.C):
            for t in range(self.T):
                rows = legal[:, c, t]
                if not rows.any(): continue
                coarse = torch.full((int(rows.sum()),), c, device=emb.device, dtype=torch.long)
                template = torch.full_like(coarse, t)
                mix = self.conditional(emb[rows], coarse, template)
                value = pc[rows, c] + pt[rows, c, t] + mixture_marginal(*mix, target[rows], active[rows])
                # Do not evaluate logaddexp(-inf,-inf) for other rows: its
                # undefined derivative can contaminate later reachable terms.
                updated = answer.clone()
                updated[rows] = torch.logaddexp(answer[rows], value)
                answer = updated
        return answer

    @torch.no_grad()
    def sample(self, encoded, n=256, *, legal_templates=None, generator=None):
        if n < 1: raise ValueError('Positive original posterior draw count required')
        emb = self.embedding(encoded); pc, pt, _ = self.discrete(emb, legal_templates)
        coarse = torch.multinomial(pc.exp(), n, replacement=True, generator=generator)
        b = torch.arange(len(emb), device=emb.device)[:, None]
        template = torch.multinomial(pt[b, coarse].exp().reshape(-1, self.T), 1, generator=generator).reshape(coarse.shape)
        e = emb[:, None].expand(-1, n, -1).reshape(-1, emb.shape[-1])
        logits, means, factors = self.conditional(e, coarse.ravel(), template.ravel())
        component = torch.multinomial(torch.softmax(logits, -1), 1, generator=generator).ravel()
        ix = torch.arange(len(e), device=emb.device)
        mean = means[ix, component]; L = factors[ix, component]
        eps = torch.randn(mean.shape, device=mean.device, dtype=mean.dtype, generator=generator)
        z = mean+(L@eps[..., None]).squeeze(-1)
        return dict(z=z.reshape(len(emb), n, self.dim), coarse=coarse, template=template,
                    component=component.reshape(coarse.shape), embedding=emb,
                    conditioned_outside_D=False, structural_release=False)
