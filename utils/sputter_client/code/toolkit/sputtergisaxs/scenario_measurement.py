'Positive native-bin measurement for simplified S1/S3/S4 forwards.'
import numpy as np

from .cell_response import rule, response
from .effective_host import host_transport
from .stage5_measurement import NATIVE_EDGES, NATIVE_AF, Q
from .tool_qinput import AF_EDGES, Q_EDGES, angles
from .optics import wavelength_nm


def _peak_qy(qpar, af_deg, energy_ev, alpha_i_deg, qscale):
    'Exact in-plane Ewald conversion of a target qpar to detector |qy|.'
    af = np.radians(np.asarray(af_deg, float))
    ai = np.radians(alpha_i_deg)
    k = 2*np.pi/wavelength_nm(energy_ev)
    a, b = np.cos(af), np.cos(ai)
    cosine = (a*a+b*b-(qpar/k)**2)/(2*a*b)
    good = (cosine >= -1) & (cosine <= 1)
    return k*a[good]*np.sqrt(np.maximum(1-cosine[good]**2, 0.))/qscale


class NativeScenarioOperator:
    'Fixed-context quick/reference projection with explicit numerical identity.'
    def __init__(self, context, scenario='S1', reference=False, qscale=1.,
                 af_shift=0., psf_af_bins=.2, psf_q_bins=.2, backend='cpu',
                 intensity_fn=None):
        if scenario not in ('S1', 'S3', 'S4'):
            raise ValueError('S1/S3/S4 latent forwards required; S2 is composed afterwards')
        if (backend not in ('cpu', 'cuda')
                or not np.isfinite([qscale, af_shift, psf_af_bins, psf_q_bins]).all()
                or qscale <= 0 or not .03 <= psf_af_bins <= 1
                or not .03 <= psf_q_bins <= 1):
            raise ValueError('Finite geometry and declared PSF range required')
        self.c = context; self.scenario = scenario; self.reference = bool(reference)
        self.qscale = float(qscale); self.shift = float(af_shift)
        self.psf_a = float(psf_af_bins); self.psf_q = float(psf_q_bins)
        self.backend = backend; self.intensity_fn = intensity_fn
        self.order = 4 if self.reference else 2
        self.F_order = 16 if scenario == 'S4' and self.reference else 8 if self.reference else 4
        self.transport_order = 16 if self.reference else 8
        if backend == 'cuda' and scenario != 'S4':
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError('Requested CUDA backend has no CUDA device')
            capability = torch.cuda.get_device_capability()
            if capability[0] < 7:
                raise RuntimeError('Current forward requires compute capability >=7.0')
            arch = 'sm_%d%d' % capability
            if capability[0] >= 12 and arch not in torch.cuda.get_arch_list():
                raise RuntimeError('Unsupported %s allocation; exclude this Maxwell node' % arch)
        da = float(AF_EDGES[1]-AF_EDGES[0])
        dq = float(Q_EDGES[1]-Q_EDGES[0])
        sigma_a, sigma_q = self.psf_a*da, self.psf_q*dq


        nds = np.unique(np.asarray(context['post_nds'], complex))
        minimum_a = .03*da
        extra_a = [-self.shift]
        extra_a.extend((NATIVE_EDGES[:, None]+np.array([-2., 2.])[None, :]*minimum_a).ravel())
        critical_angles = []
        for decrement in nds:
            if decrement.real <= 0:
                continue
            critical = np.sqrt(2*decrement.real)
            width = max(abs(decrement.imag)/critical, 1e-6)
            critical_angles.append(float(np.degrees(critical)))
            extra_a.extend(np.degrees(critical+np.array([-4., -1., 0., 1., 4.])*width)-self.shift)
        thickness = max(float(np.sum(context['thick'])), .5)


        divisor = 8. if self.reference else 4.
        maximum_a = min(da, np.degrees(wavelength_nm(context['E'])/(divisor*thickness)))
        self.ac, self.aw = rule(NATIVE_EDGES, self.order, 4*sigma_a, extra_a, maximum_a)
        amin = max(float(self.ac.min())+self.shift, 0.)
        amax = max(float(self.ac.max())+self.shift, amin)
        angle_samples = np.unique([amin, amax, np.clip(context['ai'], amin, amax)])
        extra_q = list((Q_EDGES[:, None]+np.array([-2., 2.])[None, :]*(.03*dq)).ravel())
        if scenario == 'S4':
            hint = context['measurement_hint']
            peak = float(hint['q_peak_nm_inv']); width = float(hint['width_nm_inv'])
            locations = [(peak, width/2.)]
        else:
            p = context['particle']


            D, omega = float(p.D_nm), float(p.omega_rel)
            highest = Q_EDGES[-1]+4*sigma_q
            harmonics = max(1, int(np.ceil(highest*self.qscale*D/(2*np.pi))))
            locations = [(2*np.pi*m/D,
                          2*np.pi*np.pi*m*m*omega*omega/D+1e-4)
                         for m in range(1, harmonics+1)]
        for peak, width in locations:
            gamma = width/self.qscale
            if gamma >= dq/2:
                continue
            peaks = _peak_qy(peak, angle_samples, context['E'], context['ai'], self.qscale)
            if not len(peaks):
                continue
            lo, hi = float(peaks.min()-6*gamma), float(peaks.max()+6*gamma)
            extra_q.extend(np.linspace(lo, hi, max(2, int(np.ceil((hi-lo)/gamma))+1)))
        self.qc, self.qw = rule(Q_EDGES, self.order, 4*sigma_q, extra_q)
        self.Wa, closure_a = response(NATIVE_EDGES, self.ac, self.aw, sigma_a)
        self.Wq, closure_q = response(Q_EDGES, self.qc, self.qw, sigma_q)
        self.meta = dict(schema='simplified_scenario_measurement_1',
            scenario=scenario, reference=self.reference, backend=backend,
            effective_scatter_backend='cpu analytic TS' if scenario == 'S4' else backend,
            custom_intensity_fn=intensity_fn is not None,
            AF_nodes=len(self.ac), q_nodes=len(self.qc), AF_order=self.order,
            q_order=self.order, F_order=self.F_order, transport_order=self.transport_order,
            native_AF_bins=len(NATIVE_AF), native_q_bins=len(Q),
            AF_max_segment_deg=float(maximum_a), complete_stack_phase_nm=thickness,
            unique_optical_constants=len(nds), critical_angles_deg=critical_angles,
            raw_constant_AF=closure_a, raw_constant_q=closure_q,
            exact_ray_qpar=bool(self.reference or scenario!='S1'),
            quick_qy_approximation=bool(not self.reference and scenario=='S1'),
            positive_quadrature=True, psf_applied_once=True,
            context_marks_unchanged=True, original_native_certificate=False,
            quick_feature_error_gate_required=True,
            S3_vertical_integral='analytic identical step profile' if scenario == 'S3' else None)

    def _intensity(self, af, q):
        if self.intensity_fn is not None:
            return self.intensity_fn(self.c, af, q, order=self.F_order,
                                     reference=self.reference)
        if self.scenario == 'S1':
            if self.backend == 'cuda':
                from .effective_cuda import metal_intensity_cuda
                return metal_intensity_cuda(self.c, af, q, order=self.F_order)
            from .effective_host import metal_intensity
            return metal_intensity(self.c, af, q, order=self.F_order)
        if self.scenario == 'S3':
            from .scenario_steps import step_intensity
            return step_intensity(self.c, af, q, order=self.F_order,
                                  backend=self.backend)
        from .scenario_ts import metal_intensity
        return metal_intensity(self.c, af, q, order=self.F_order,
                               reference=self.reference)

    def evaluate(self, db_xi_nm=4., power_exponent=2., scatter_depth_nm=5.):
        'Return unscaled latent components compatible with effective compose.'
        if (not np.isfinite([db_xi_nm, power_exponent, scatter_depth_nm]).all()
                or min(db_xi_nm, power_exponent, scatter_depth_nm) <= 0):
            raise ValueError('Positive finite background/transport scales required')
        shape = (len(NATIVE_AF), len(Q))
        metal = np.zeros(shape); db = np.zeros(shape)
        power = np.zeros(shape); ring = np.zeros(shape)
        transported = np.zeros(len(NATIVE_AF))
        positive_weights = np.zeros(len(NATIVE_AF))
        block = 16 if self.scenario == 'S1' else 64
        for start in range(0, len(self.ac), block):
            stop = min(start+block, len(self.ac))
            physical_af = self.ac[start:stop]+self.shift
            good = physical_af > 0
            if not good.any():
                continue
            aa = physical_af[good]
            ang = angles(self.c['E'], self.c['ai'], self.qscale, 0.,
                         q_centres=self.qc, af_centres=aa)


            scatter_q = abs(self.qc)*self.qscale if not self.reference and self.scenario=='S1' else ang['qpar']
            I = np.asarray(self._intensity(aa, scatter_q), float)
            if I.shape != ang['qpar'].shape or not np.isfinite(I).all() or np.any(I < 0):
                raise ValueError('Scenario forward returned unsupported intensity')
            wa = self.Wa[:, start:stop][:, good]
            metal += wa @ (I @ self.Wq.T)
            T, supported = host_transport(self.c, aa, depth_nm=scatter_depth_nm,
                                            order=self.transport_order)
            if not supported.all():
                raise ValueError('Effective host transport has a zero/nonfinite denominator')
            transported += wa @ T
            positive_weights += wa @ np.ones(len(aa))
            qp, qt = ang['qpar'], ang['q_total']
            db += wa @ ((1/(1+(qp*db_xi_nm)**2)**2) @ self.Wq.T)
            power += wa @ (((1+qp/.05)**(-power_exponent)) @ self.Wq.T)
            ring += wa @ (np.exp(-.5*((qt-2.1)/.1)**2) @ self.Wq.T)


            self.c.get('_field_cache', {}).clear()
        if np.any(positive_weights <= 0):
            raise ValueError('No positive-angle support in a native output bin')
        transported /= positive_weights
        return dict(metal=metal, transport=transported[:, None], db=db, power=power,
                    ring=ring, flat=np.ones(shape), meta=self.meta,
                    transport_definition='positive-bin-averaged conditional field-power ratio; empirical response not inverted')
