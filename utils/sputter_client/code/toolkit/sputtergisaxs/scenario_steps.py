'S3 effective columns on a two/three-step metal depth profile.'
from dataclasses import dataclass, replace
import numpy as np

from .effective_host import HostOptics, chi, field_arrays, field_at
from .formfactor import _gauss_legendre, jinc
from .forward_v06 import ssca_intensity
from .optics import nd, wavelength_nm


@dataclass(frozen=True)
class StepProfile:
    'Upward z edges (nm), metal volume fractions, and footprint occupation.'
    edges_nm: tuple = (-8., 0., 2.)
    metal_fractions: tuple = (.2, .35)
    coverage: float = .65

    def arrays(self, thickness_nm):
        z = np.asarray(self.edges_nm, float)
        f = np.asarray(self.metal_fractions, float)
        if (z.ndim != 1 or len(f) not in (2, 3) or len(z) != len(f)+1
                or not np.isfinite(z).all() or not np.isfinite(f).all()
                or np.any(np.diff(z) <= 0) or np.any(f < 0) or np.any(f > 1)
                or z[0] < -float(thickness_nm) or z[-1] <= 0
                or not np.any(z == 0.) or z[0] >= 0):
            raise ValueError('Two/three positive-width steps across the host surface required')
        if not np.isfinite(self.coverage) or not 0 < self.coverage <= 1:
            raise ValueError('Effective footprint occupation must be in (0,1]')
        return z, f


def build_step_context(host, particle, profile=StepProfile(), metal='Au',
                       energy_ev=12000., alpha_i_deg=.4, *, nbins=24,
                       column_density_nm2=None):
    'Build a common finite reference and a consistent marked-column model.'
    if isinstance(profile, dict):
        profile = StepProfile(**profile)
    if not isinstance(profile, StepProfile) or not isinstance(host, HostOptics):
        raise TypeError('HostOptics and StepProfile required')
    host.validate()
    if (metal not in ('Au', 'Ag') or not np.isfinite([energy_ev, alpha_i_deg]).all()
            or energy_ev <= 0 or alpha_i_deg <= 0 or int(nbins) < 4):
        raise ValueError('Finite Au/Ag geometry and positive mark order required')
    ze, fractions = profile.arrays(host.thickness_nm)
    state = particle.state()
    if state.film_thickness or state.infil_mass:
        raise ValueError('Use explicit step fractions, not additive continuous-film parameters')
    radii, _, weights = state.size_bins(int(nbins))
    radius_scale = ((1+state.aspect**2)/(2*state.aspect)
                    if state.shape == 'cap' and state.aspect > 1 else 1.)
    radii = radii*radius_scale
    mean_area = float(weights @ (np.pi*radii*radii))
    density = profile.coverage/mean_area if column_density_nm2 is None else float(column_density_nm2)
    occupation = density*mean_area
    if (not np.isfinite(density) or density <= 0 or occupation > 1+1e-12
            or np.any(fractions > occupation+1e-12)):
        raise ValueError('Column density must give max(phi) <= mean footprint occupation <= 1')

    sharp = replace(host, transition_nm=0.,
                    roughness_nm=max(host.roughness_nm, host.transition_nm),
                    buried_roughness_nm=max(host.buried_roughness_nm, host.transition_nm))
    edges = np.unique(np.r_[ze, 0., -host.thickness_nm,
                           -host.thickness_nm-host.oxide_nm])
    widths = np.diff(edges)
    middle = (edges[:-1]+edges[1:])/2
    ix = np.searchsorted(ze, middle, side='right')-1
    present = (ix >= 0) & (ix < len(fractions))
    phi = np.zeros(len(middle))
    phi[present] = fractions[ix[present]]
    base = np.zeros(len(middle), complex)
    base[(middle < 0) & (middle > -host.thickness_nm)] = host.decrement
    base[middle < -host.thickness_nm] = nd('SiO2', energy_ev=energy_ev)
    metal_nd = nd(metal, energy_ev=energy_ev)
    post = base+(metal_nd-base)*phi
    filling = phi/occupation
    sigma = np.zeros(len(middle)+2)
    for boundary, rough in ((0., sharp.roughness_nm),
                            (-host.thickness_nm, sharp.buried_roughness_nm)):
        where = np.flatnonzero(edges[::-1] == boundary)
        if len(where):
            sigma[int(where[0])+1] = max(sigma[int(where[0])+1], rough)
    active = fractions > 0
    occupied_edges = ze[np.r_[active, False] | np.r_[False, active]]
    depth_interval = [float(occupied_edges[0]), float(occupied_edges[-1])] if len(occupied_edges) else None
    mass = float(phi @ widths)

    mean_error = float(np.max(abs(density*mean_area*filling-phi)))
    return dict(schema='S3_effective_step_columns_1', scenario='S3',
        host=sharp, original_host=host, particle=replace(particle), profile=profile,
        metal=metal, E=float(energy_ev), ai=float(alpha_i_deg),
        R=radii, H=np.full(len(radii), ze[-1]-ze[0]), w=weights, n=density,
        shape='effective_step_column', input_shape=state.shape,
        edges=edges, phi=phi, filling=filling, base=base,
        pre_nds=[0j]+base[::-1].tolist()+[nd('Si', energy_ev=energy_ev)],
        post_nds=[0j]+post[::-1].tolist()+[nd('Si', energy_ev=energy_ev)],
        thick=widths[::-1].tolist(), sigma=sigma.tolist(),
        mass_nm=mass, primitive_mass_nm=mass, mean_scale=1.,
        footprint_occupation=occupation, mean_footprint_area_nm2=mean_area,
        max_phi=float(phi.max()), mean_profile_error=mean_error,
        labels=dict(D_nm=float(particle.D_nm),
            D_definition='conditional lateral correlation step; not SEM nearest neighbour',
            lateral_diameter_nm=float(2*particle.base_radius_nm*radius_scale),
            diameter_definition='median effective column maximum footprint diameter',
            depth_interval_nm=depth_interval,
            depth_span_nm=None if depth_interval is None else depth_interval[1]-depth_interval[0],
            exposed_span_nm=0. if depth_interval is None else max(0., depth_interval[1]),
            burial_span_nm=0. if depth_interval is None else max(0., -depth_interval[0]),
            island_height_nm=None,
            height_missing_reason='step-column vertical support is not an island height',
            coverage=float(occupation),
            coverage_definition='effective mean footprint occupation; geometric union not validated',
            gap_nm=None, gap_missing_reason='no validated centre configuration',
            conditional_geometry=True, structural_release=False),
        approximation=dict(vertical='two/three sharp metal-fraction steps',
            lateral='common marked column footprints; SSCA correlation model',
            host_transition='single NC roughness instead of a graded profile',
            original_transition_nm=float(host.transition_nm),
            cap_hemi='input shape only sets footprint scale; not recovered vertical shape'))


def _decay_exprel(z):
    'expm1(z)/z on Re(z)<=0, including the exact critical limit.'
    z = np.asarray(z, complex)
    out = np.empty(z.shape, complex)
    small = abs(z) < 1e-5
    zz = z[small]
    out[small] = 1+zz/2+zz*zz/6+zz**3/24+zz**4/120
    out[~small] = np.expm1(z[~small])/z[~small]
    return out


def _wave_at_bottom(coefficient, exponent):
    'Stable coefficient*exp(exponent), avoiding intermediate growing waves.'
    coefficient, exponent = np.broadcast_arrays(np.asarray(coefficient, complex),
                                                 np.asarray(exponent, complex))
    out = np.zeros(coefficient.shape, complex)
    active = abs(coefficient) > 0
    if active.any():
        logarithm = np.log(abs(coefficient[active]))+exponent[active].real
        if np.any(logarithm > 700):
            raise ValueError('Nonfinite effective-stack wave amplitude')
        phase = np.angle(coefficient[active])+exponent[active].imag
        out[active] = np.exp(logarithm)*np.exp(1j*phase)
    return out


def step_vertical_amplitude(context, af_deg, *, order=4, method='analytic'):
    'Complex depth factor from the same reference and metal profile.'
    af = np.atleast_1d(af_deg).astype(float)
    if not np.isfinite(af).all() or np.any(af <= 0):
        raise ValueError('Finite positive exit angles required')
    if method not in ('analytic', 'quadrature') or int(order) < 2:
        raise ValueError('Analytic or positive Gauss cell integration required')
    e = context['edges']; cells = len(e)-1
    inc, rec = field_arrays(context, af)
    result = np.zeros(len(af), complex)
    for upward in np.flatnonzero(context['filling'] > 0):
        lo, hi = e[upward:upward+2]; width = hi-lo
        contrast = chi(nd(context['metal'], energy_ev=context['E'])-context['base'][upward])
        filling = context['filling'][upward]
        if method == 'quadrature':
            t, w = _gauss_legendre(int(order))
            integral = field_at(context, af, lo+width*t) @ (width*w)
        else:
            layer = cells-upward
            ki = inc['kz'][layer, 0]
            ko = rec['kz'][layer]
            integral = np.zeros(len(af), complex)
            for si, keyi in ((1, 'T'), (-1, 'R')):
                ci = inc[keyi][layer, 0]
                for so, keyo in ((1, 'T'), (-1, 'R')):
                    co = rec[keyo][layer]
                    phase = 1j*(si*ki+so*ko)*width
                    bottom = phase.real > 0
                    term = np.empty(len(af), complex)


                    term[~bottom] = ci*co[~bottom]*width*_decay_exprel(phase[~bottom])
                    if bottom.any():
                        bi = _wave_at_bottom(ci, 1j*si*ki*width)
                        bo = _wave_at_bottom(co[bottom], 1j*so*ko[bottom]*width)
                        term[bottom] = bi*bo*width*_decay_exprel(-phase[bottom])
                    integral += term
        result += filling*contrast*integral
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite effective-step depth amplitude')
    return result


def step_intensity(context, af_deg, qpar_nm_inv, *, order=4, backend='cpu',
                   af_block=64, vertical_method='analytic'):
    'S3 intensity on q[q] or exact ray qpar[AF,q]; optional local CUDA import.'
    af = np.atleast_1d(af_deg).astype(float)
    q = np.asarray(qpar_nm_inv, float)
    if q.ndim == 1:
        q = np.broadcast_to(q, (len(af), len(q)))
    if (q.ndim != 2 or q.shape[0] != len(af) or not np.isfinite(q).all()
            or np.any(q < 0) or backend not in ('cpu', 'cuda') or int(af_block) < 1):
        raise ValueError('Finite nonnegative ray q grid and cpu/cuda backend required')
    p = context['particle']; R = context['R']; weights = context['w']
    pref = (2*np.pi/wavelength_nm(context['E']))**4/(16*np.pi*np.pi)*context['n']
    out = np.zeros(q.shape)
    if backend == 'cuda':
        import torch
        from .bessel_cuda import j1
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA required for the CUDA step backend')
        capability = torch.cuda.get_device_capability()
        if capability[0] < 7:
            raise RuntimeError('Step CUDA backend requires compute capability >=7.0')

        arch = 'sm_%d%d' % capability
        if capability[0] >= 12 and arch not in torch.cuda.get_arch_list():
            raise RuntimeError('Current torch does not support %s; exclude this node' % arch)
        device = torch.device('cuda', torch.cuda.current_device())
        rr = torch.as_tensor(R, dtype=torch.float64, device=device)
        ww = torch.as_tensor(weights, dtype=torch.float64, device=device)
        dr = rr-torch.sum(rr*ww)
    for start in range(0, len(af), int(af_block)):
        stop = min(start+int(af_block), len(af))
        vertical = step_vertical_amplitude(context, af[start:stop], order=order,
                                            method=vertical_method)
        qq = q[start:stop]
        if backend == 'cpu':
            lateral = np.pi*R[:, None, None]**2*jinc(R[:, None, None]*qq[None, :, :])
            F = lateral*vertical[None, :, None]
            power = ssca_intensity(F.reshape(len(R), -1), R, weights, qq.ravel(),
                                   p.D_nm, p.omega_rel*p.D_nm, p.kappa).reshape(qq.shape)
        else:
            with torch.no_grad():
                qt = torch.as_tensor(np.array(qq), dtype=torch.float64, device=device)
                x = rr[:, None, None]*qt[None, :, :]
                safe = torch.where(abs(x) < 1e-7, torch.ones_like(x), x)
                J = torch.where(abs(x) < 1e-7, 1-x*x/8, 2*j1(x)/safe)
                L = np.pi*rr[:, None, None]**2*J
                phase = torch.exp(1j*p.kappa*dr[:, None, None]*qt[None, :, :])
                M = torch.sum(ww[:, None, None]*L*L, dim=0)
                fp = torch.sum(ww[:, None, None]*L*phase, dim=0)
                fm = torch.sum(ww[:, None, None]*L/phase, dim=0)
                p2 = torch.sum(ww[:, None, None]*phase*phase, dim=0)
                p0 = torch.exp(1j*qt*p.D_nm-.5*(qt*p.omega_rel*p.D_nm)**2-p.D_nm/1e4)
                raw = M+2*torch.real(p0*fp*torch.conj(fm)/(1-p0*p2))
                V = torch.as_tensor(abs(vertical)**2, dtype=torch.float64, device=device)
                power = (raw*V[:, None]).cpu().numpy()
        if (not np.isfinite(power).all()
                or power.min() < -1e-7*max(float(abs(power).max()), 1e-30)):
            raise ValueError('Effective-step SSCA positivity failure')
        out[start:stop] = pref*np.maximum(power, 0.)


        context.get('_field_cache', {}).clear()
    return out
