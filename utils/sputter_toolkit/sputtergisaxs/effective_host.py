"""v2 research forward: one metal island law on an effective optical host.

The same size marks enter exact depth means and complex DWBA amplitudes.
Buried metal replaces the first host layer; no penetration of a second
interface is supported. SSCA D is a conditional radial step, not a certified
2D nearest-neighbour distance. Empirical host transport/cross are effective
nuisances, not a recovered microscopic host geometry. No inverse is certified.
"""
from dataclasses import dataclass, replace
import numpy as np
from .optics import nd, wavelength_nm
from .surface_graded import mean_stack, OccupancyError
from .formfactor import cap_profile, hemiellipsoid_profile, _gauss_legendre, jinc
from .multilayer import amplitudes
from .forward_v06 import ssca_intensity


@dataclass(frozen=True)
class HostOptics:
    """Optical decrement delta+i*beta and lengths in nm; not inferred outputs."""
    decrement: complex = 0j
    thickness_nm: float = 0.
    oxide_nm: float = 1.5
    roughness_nm: float = 0.
    buried_roughness_nm: float = 0.
    transition_nm: float = 0.

    def validate(self):
        values = [self.decrement.real, self.decrement.imag, self.thickness_nm,
                  self.oxide_nm, self.roughness_nm, self.buried_roughness_nm,self.transition_nm]
        if not np.isfinite(values).all() or min(values) < 0:
            raise ValueError('Passive finite host constants and nonnegative lengths required')
        if self.transition_nm and (self.roughness_nm or self.buried_roughness_nm):
            raise ValueError('Use graded reference or legacy NC, never both')


def host_fractions(host,z):
    """Positive phase fractions: one common Gaussian convolution of the stack.

    Width describes an effective optical reference, not recovered physical RMS.
    In the sharp limit, exact interfaces use the downward-side value; GL nodes
    are strictly within their cells so this measure-zero convention is inert.
    """
    z=np.asarray(z,float)
    if host.transition_nm:
        from scipy.special import ndtr
        p0=ndtr(-z/host.transition_nm)
        p1=ndtr((-z-host.thickness_nm)/host.transition_nm)
        p2=ndtr((-z-host.thickness_nm-host.oxide_nm)/host.transition_nm)
    else:
        p0=(z<=0).astype(float);p1=(z<=-host.thickness_nm).astype(float)
        p2=(z<=-host.thickness_nm-host.oxide_nm).astype(float)
    return np.array([1-p0,p0-p1,p1-p2,p2])


def host_base(host,E,z):
    f=host_fractions(host,z)
    return host.decrement*f[1]+nd('SiO2',energy_ev=E)*f[2]+nd('Si',energy_ev=E)*f[3]


def chi(decrement):
    return -2 * np.real(decrement) + 2j * np.imag(decrement)


def build_context(host, particle, metal='Au', energy_ev=12000., alpha_i_deg=.4,
                  *, burial_nm=0., nbins=28, ndepth=64, mean_scale=1.,transition_order=8,tail_sigma=8.,transition_steps=16,reference_mesh='full'):
    """Top-down reference cells, with absolute physical z=0 at host surface.

    mean_scale=0 is a directed numerical bridge only. Production uses 1.
    Occupancy is checked BEFORE this bridge; impossible geometries are rejected.
    """
    host.validate()
    if metal not in ('Au', 'Ag') or not np.isfinite([energy_ev, alpha_i_deg, burial_nm, mean_scale]).all():
        raise ValueError('Finite Au/Ag geometry required')
    if energy_ev <= 0 or alpha_i_deg <= 0 or burial_nm < 0 or not 0 <= mean_scale <= 1:
        raise ValueError('Invalid geometry or mean bridge')
    if burial_nm > host.thickness_nm:
        raise ValueError('Burial beyond the first finite effective host is unsupported')
    st = mean_stack(particle, nbins, ndepth)
    R, H, w = st['R'], st['H'], st['w']
    if reference_mesh not in ('full','compact'):raise ValueError('Unknown optical mesh')
    primitive_edges=st['edges_up_nm']
    if reference_mesh=='compact':
        # Numerical candidate only: integrate each clipped mark EXACTLY into
        # regular/tail cells, rather than adding every mark tip as an interface.
        # The full original mark law and exact point-occupancy guard remain.
        top=float(H.max());dense=min(top,4*particle.height_nm)
        primitive_edges=np.unique(np.r_[np.linspace(0,dense,ndepth+1),
            np.geomspace(dense,top,max(2,ndepth//4+1))])
    # Include every primitive boundary and each physical host interface.
    edges = np.unique(np.r_[primitive_edges - burial_nm, 0.,
                            -host.thickness_nm, -host.thickness_nm-host.oxide_nm])
    if host.transition_nm:
        if transition_order<2 or tail_sigma<8 or transition_steps<8:raise ValueError('Underresolved graded-reference declaration')
        # Explicit negligible-tail support and positive transition mesh. Never
        # truncate/renormalize fractions at the original island/film endpoints.
        width=host.transition_nm
        interfaces=(0.,-host.thickness_nm,-host.thickness_nm-host.oxide_nm)
        grids=[b+np.linspace(-tail_sigma*width,tail_sigma*width,int(2*tail_sigma*transition_steps)+1) for b in interfaces]
        edges=np.unique(np.concatenate([edges]+grids))
    widths = np.diff(edges)
    mid = (edges[:-1] + edges[1:]) / 2
    phi = np.zeros(len(widths))
    t, wt = _gauss_legendre(2)
    for a, h, ww in zip(R, H, w):
        lo = np.clip(edges[:-1] + burial_nm, 0., h)
        hi = np.clip(edges[1:] + burial_nm, 0., h)
        dz = hi-lo
        z = lo[:, None] + dz[:, None]*t
        tip = (h-hi[:, None]) + dz[:, None]*(1-t)
        area = tip*(a*a/h+z) if st['shape']=='cap' else a*a*tip*(h+z)/(h*h)
        phi += ww*np.pi*dz*(area@wt)*st['n']/widths
    if np.any(phi < 0) or np.max(phi) > 1+1e-12 or not np.isfinite(phi).all():
        raise OccupancyError(float(np.max(phi)))
    base = np.zeros(len(mid), complex)
    if host.transition_nm:
        tt,wwt=_gauss_legendre(transition_order)
        zz=edges[:-1,None]+widths[:,None]*tt
        base=host_base(host,energy_ev,zz)@wwt
        overlap=np.zeros(len(mid),complex)
        for a,h,ww in zip(R,H,w):
            lo=np.clip(edges[:-1]+burial_nm,0.,h);hi=np.clip(edges[1:]+burial_nm,0.,h);dz=hi-lo
            zz=lo[:,None]+dz[:,None]*tt
            tip=(h-hi[:,None])+dz[:,None]*(1-tt)
            area=tip*(a*a/h+zz) if st['shape']=='cap' else a*a*tip*(h+zz)/(h*h)
            overlap+=ww*np.pi*dz*((host_base(host,energy_ev,zz-burial_nm)*area)@wwt)*st['n']/widths
    else:
        base[(mid < 0) & (mid > -host.thickness_nm)] = host.decrement
        base[mid < -host.thickness_nm] = nd('SiO2', energy_ev=energy_ev)
        overlap=base*phi
    m = nd(metal, energy_ev=energy_ev)
    post = base + mean_scale*(m*phi-overlap)
    # Internal mean interfaces are not assigned physical roughness.
    sigma = np.zeros(len(mid)+2)
    topdown = edges[::-1]
    for z, rough in [(0.,host.roughness_nm),(-host.thickness_nm,host.buried_roughness_nm)]:
        # Literal physical boundaries were inserted above; a tolerance could
        # select an ulp-adjacent primitive boundary and roughen the wrong cell.
        ix = np.flatnonzero(topdown==z)
        if len(ix): sigma[int(ix[0])+1] = max(sigma[int(ix[0])+1], rough)
    return dict(host=host, particle=replace(particle), metal=metal, E=energy_ev, ai=alpha_i_deg,
                burial=burial_nm, R=R, H=H, w=w, n=st['n'], shape=st['shape'],
                edges=edges, phi=phi, base=base,
                pre_nds=[0j]+base[::-1].tolist()+[nd('Si',energy_ev=energy_ev)],
                post_nds=[0j]+post[::-1].tolist()+[nd('Si',energy_ev=energy_ev)],
                thick=widths[::-1].tolist(), sigma=sigma.tolist(), max_phi=st['max_fraction'],
                mass_nm=float(phi@widths), primitive_mass_nm=st['primitive_mass_nm'], mean_scale=mean_scale,
                transition_order=transition_order,tail_sigma=tail_sigma,transition_steps=transition_steps,
                reference_mesh=reference_mesh,
                field_particle_parameters={key:getattr(particle,key) for key in
                    ('D_nm','base_radius_nm','height_nm','sigma_log','shape','film_thickness_nm','average_material')})


def _cell_field(kz,down,up,distance):
    """Critical-limit cos/sinc, switching for deep evanescent cancellation.

    cosh/sinh individually grow in a thick cell although its downward wave
    decays. Above |Im(kz*d)|=4 evaluate the two physical waves separately.
    """
    kz,down,up,distance=np.broadcast_arrays(kz,down,up,distance)
    phase=kz*distance;deep=abs(phase.imag)>4
    out=np.empty(phase.shape,complex)
    shallow=~deep
    out[shallow]=(down[shallow]+up[shallow])*np.cos(phase[shallow])+1j*kz[shallow]*(down[shallow]-up[shallow])*distance[shallow]*np.sinc(phase[shallow]/np.pi)
    out[deep]=down[deep]*np.exp(1j*phase[deep])+up[deep]*np.exp(-1j*phase[deep])
    return out


def field_arrays(context, af_deg, *, post=True):
    """Shared CPU Parratt layer tables for CPU and optional CUDA evaluation."""
    af = np.atleast_1d(af_deg).astype(float)
    if not np.isfinite(af).all() or np.any(af <= 0):
        raise ValueError('Finite positive AF required')
    nds = context['post_nds'] if post else context['pre_nds']
    k = 2*np.pi/wavelength_nm(context['E'])
    signature=(context['E'],context['ai'],bool(post),af.shape,af.tobytes(),tuple(nds),tuple(context['thick']),tuple(context['sigma']))
    cache=context.setdefault('_field_cache',{})
    if signature not in cache:
        inc=amplitudes(k,np.radians([context['ai']]),nds,context['thick'],context['sigma'])
        rec=amplitudes(k,np.radians(af),nds,context['thick'],context['sigma'])
        cache[signature]=tuple({key:np.asarray(amp[key]) for key in ('kz','T','R')} for amp in (inc,rec))
    return cache[signature]


def field_at(context, af_deg, z_nm, *, post=True):
    """Incident times reciprocal-exit field at physical z, AF x depth."""
    z = np.atleast_1d(z_nm).astype(float)
    if not np.isfinite(z).all():raise ValueError('Finite physical heights required')
    e = context['edges']; cells = len(e)-1
    if np.any(z < e[0]-1e-12) or np.any(z > e[-1]+1e-12):
        raise ValueError('Field query outside represented effective reference')
    upward = np.clip(np.searchsorted(e, z, side='right')-1, 0, cells-1)
    layer = cells-upward
    distance = e[upward+1]-z
    inc,rec=field_arrays(context,af_deg,post=post)
    ki=inc['kz'][layer,0];Di=inc['T'][layer,0];Ui=inc['R'][layer,0]
    ko=rec['kz'][layer].T;Do=rec['T'][layer].T;Uo=rec['R'][layer].T
    Ei = _cell_field(ki,Di,Ui,distance)
    Ef = _cell_field(ko,Do,Uo,distance)
    out = Ef*Ei[None,:]
    if not np.isfinite(out).all(): raise ValueError('Nonfinite stratified field')
    return out


def metal_quadrature(context,order):
    """Cache exact mark nodes only for their complete physical context."""
    if any(getattr(context['particle'],key)!=value for key,value in context['field_particle_parameters'].items()):
        raise ValueError('Changed particle mean-field family; rebuild context (D is not free)')
    key=(order,context['E'],context['metal'],context['shape'],context['burial'],
         context['edges'].tobytes(),context['R'].tobytes(),context['H'].tobytes(),context['base'].tobytes())
    cache=context.setdefault('_mark_cache',{})
    if key not in cache:
        e=context['edges'];t,wt=_gauss_legendre(order);rows=[]
        profile=cap_profile if context['shape']=='cap' else hemiellipsoid_profile
        m=nd(context['metal'],energy_ev=context['E'])
        for a,h in zip(context['R'],context['H']):
            lo=np.maximum(e[:-1],-context['burial']);hi=np.minimum(e[1:],h-context['burial']);use=hi>lo;lo,hi=lo[use],hi[use]
            z=lo[:,None]+(hi-lo)[:,None]*t;rad=profile(a,h,(z+context['burial'])/h).ravel()
            local_base=host_base(context['host'],context['E'],z.ravel()) if context['host'].transition_nm else np.repeat(context['base'][use],order)
            weight=(wt[None,:]*(hi-lo)[:,None]*np.pi).ravel()*rad*rad*chi(m-local_base)
            rows.append((z.ravel(),rad,weight))
        cache[key]=rows
    return cache[key]


def metal_amplitudes(context, af_deg, qpar_nm_inv, *, order=4, af_block=8):
    """Complex contrast-weighted F[mark,AF,q]; exact ray qpar or separable q.

    q[AF,q] uses each ray's true qpar. q[q] is a proposed fast approximation
    only when used in place of |q_y|; its 200-context feature gate is separate.
    """
    af = np.atleast_1d(af_deg).astype(float); q = np.asarray(qpar_nm_inv,float)
    if q.ndim==1: qshape=(len(af),len(q)); separable=True
    elif q.ndim==2 and q.shape[0]==len(af): qshape=q.shape; separable=False
    else: raise ValueError('Expected q[q] or exact qpar[AF,q]')
    if not np.isfinite(q).all() or np.any(q<0): raise ValueError('Invalid q')
    F = np.zeros((len(context['R']),)+qshape,complex)
    for mark,(z,rad,weight) in enumerate(metal_quadrature(context,order)):
        fields = field_at(context,af,z)
        if separable:
            B = weight[:,None]*jinc(rad[:,None]*q[None,:])
            F[mark] = fields@B
        else:
            for i in range(0,len(af),af_block):
                B = weight[None,:,None]*jinc(rad[None,:,None]*q[i:i+af_block,None,:])
                F[mark,i:i+af_block] = np.einsum('az,azq->aq',fields[i:i+af_block],B,optimize=True)
    if not np.isfinite(F).all(): raise ValueError('Nonfinite contrast amplitudes')
    return F


def metal_intensity(context, af_deg, qpar_nm_inv, *, order=4):
    F = metal_amplitudes(context,af_deg,qpar_nm_inv,order=order)
    q = np.asarray(qpar_nm_inv)
    if q.ndim==1: q=np.broadcast_to(q,F.shape[1:])
    p = context['particle']
    power = ssca_intensity(F.reshape(len(context['R']),-1),context['R'],context['w'],
                           q.ravel(),p.D_nm,p.omega_rel*p.D_nm,p.kappa).reshape(q.shape)
    if not np.isfinite(power).all() or power.min() < -1e-7*max(float(abs(power).max()),1e-30):
        raise ValueError('SSCA positivity failure')
    k = 2*np.pi/wavelength_nm(context['E'])
    return np.maximum(power,0)*(k*k/(4*np.pi))**2*context['n']


def host_transport(context, af_deg, *, depth_nm=5., order=24):
    """Effective exponential-depth weighted post/pre field-power ratio.

    This is a conditional bin-transport nuisance, not deconvolution of an
    empirical template. Zero denominators return unsupported, never epsilon.
    """
    depth=min(context['host'].thickness_nm+context['host'].oxide_nm,5*depth_nm)
    if depth_nm<=0: raise ValueError('Positive host scatter depth required')
    if depth==0: z=np.array([0.]); w=np.ones(1)
    else:
        # Resolve actual optical cell boundaries and cumulative film phase.
        cuts=np.unique(np.r_[-depth,0.,context['edges'][(context['edges']>-depth)&(context['edges']<0)]])
        t,wt=_gauss_legendre(order)
        z=(cuts[:-1,None]+np.diff(cuts)[:,None]*t).ravel()
        w=(np.diff(cuts)[:,None]*wt).ravel()*np.exp(z/depth_nm)
        w/=w.sum()
    pre=abs(field_at(context,af_deg,z,post=False))**2@w
    post=abs(field_at(context,af_deg,z,post=True))**2@w
    valid=(pre>0)&np.isfinite(pre)&np.isfinite(post)
    ratio=np.divide(post,pre,out=np.zeros_like(post),where=valid)
    return ratio,valid


def combine_host(metal_rate, host_rate, *, gamma=0., transmission=None,
                 transmitted_fraction=1.):
    """Same measured-rate units; bounded PSD cross, no positivity clamp.

    Untransported fraction accommodates instrumental/uncertain template
    components. Templates already contain PSF; do not blur them a second time.
    """
    m,h=np.broadcast_arrays(np.asarray(metal_rate,float),np.asarray(host_rate,float))
    if not np.isfinite(m).all() or not np.isfinite(h).all() or np.any(m<0) or np.any(h<0):
        raise ValueError('Finite nonnegative rates in common units required')
    if not np.isfinite([np.real(gamma),np.imag(gamma),transmitted_fraction]).all() or abs(gamma)>1 or not 0<=transmitted_fraction<=1:
        raise ValueError('Cross covariance must be PSD')
    if transmission is not None:
        T=np.asarray(transmission,float)
        if not np.isfinite(T).all() or np.any(T<0): raise ValueError('Unsupported transport requires parametric fallback')
        h=h*((1-transmitted_fraction)+transmitted_fraction*T)
    # This algebraic form retains the negative cross and avoids cancellation.
    g=float(np.real(gamma))
    sm,sh=np.sqrt(m),np.sqrt(h)
    return (sm-sh)**2+2*(1+g)*(sm*sh)
