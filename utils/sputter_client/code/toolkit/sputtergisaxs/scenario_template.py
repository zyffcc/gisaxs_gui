'S2: an effective, measured-rate host-template decoration scenario.'
from dataclasses import dataclass

import numpy as np

from .effective_host import combine_host


@dataclass(frozen=True)
class TemplateGrowth:
    'Nonnegative host gain that grows monotonically with metal loading.'
    base_gain: float = 1.
    growth_gain: float = 1.
    loading_scale: float = 1.
    power: float = 1.

    def gain(self, loading):
        values = [self.base_gain, self.growth_gain, self.loading_scale,
                  self.power, loading]
        if not np.isfinite(values).all() or min(values[:2]) < 0:
            raise ValueError('Finite nonnegative host growth gains required')
        if self.loading_scale <= 0 or self.power <= 0 or loading < 0:
            raise ValueError('Positive loading scale/power and nonnegative loading required')
        if loading == 0:
            fraction = 0.
        else:


            log_scaled = self.power * (np.log(float(loading)) - np.log(self.loading_scale))
            fraction = 1. if log_scaled > np.log(745.) else float(-np.expm1(-np.exp(log_scaled)))
        return float(self.base_gain + self.growth_gain * fraction)


def compose_template(metal_rate, host_rate=None, *, loading=1., growth=None,
                     metal_gain=1., cross_coefficient=0., transmission=None,
                     transmitted_fraction=1., fallback_host_rate=None,
                     instrumental_rate=None, loading_unit='nuisance_proxy',
                     host_source='unspecified_conditional_template'):
    'Compose S2 in common measured-rate units, returning rate and provenance.'
    growth = TemplateGrowth() if growth is None else growth
    if not isinstance(growth, TemplateGrowth):
        raise TypeError('growth must be TemplateGrowth')
    c = float(cross_coefficient)
    gain = float(metal_gain)
    if not np.isfinite([c, gain, transmitted_fraction]).all() or abs(c) > 1 or gain < 0:
        raise ValueError('Finite metal gain>=0 and real cross coefficient in [-1,1] required')
    if not 0 <= transmitted_fraction <= 1:
        raise ValueError('transmitted_fraction must be in [0,1]')
    host_gain = growth.gain(float(loading))
    source_kind = 'supplied_template_or_completed_template'
    if host_rate is None:
        host_rate = 0. if fallback_host_rate is None else fallback_host_rate
        source_kind = 'no_host_component' if fallback_host_rate is None else 'caller_parametric_fallback'
    m, h = np.broadcast_arrays(np.asarray(metal_rate, float) * gain,
                               np.asarray(host_rate, float) * host_gain)


    rate = combine_host(m, h, gamma=c, transmission=transmission,
                        transmitted_fraction=transmitted_fraction)
    if transmission is not None:
        h = h * ((1. - transmitted_fraction) + transmitted_fraction * np.asarray(transmission, float))
        m, h = np.broadcast_arrays(m, h)
    background = np.zeros_like(rate) if instrumental_rate is None else np.broadcast_to(np.asarray(instrumental_rate, float), rate.shape)
    if not np.isfinite(background).all() or np.any(background < 0):
        raise ValueError('Instrument background must be finite and nonnegative')
    rate = rate + background
    if not np.isfinite(rate).all():
        raise ValueError('S2 common-unit composition overflowed')
    return dict(
        rate=rate,
        components=dict(metal=m, host=h, cross=2. * c * np.sqrt(m) * np.sqrt(h),
                        background=background),
        metadata=dict(
            scenario='S2', model='effective_host_template_intensity_cross_v1',
            host_source=str(host_source), host_source_kind=source_kind,
            host_gain=host_gain, metal_gain=gain, loading=float(loading),
            loading_unit=str(loading_unit),
            growth_parameters=dict(base_gain=float(growth.base_gain),
                                   growth_gain=float(growth.growth_gain),
                                   loading_scale=float(growth.loading_scale),
                                   power=float(growth.power)),
            cross_coefficient=c, cross_interpretation='effective intensity approximation; no inferred complex phase or shared microscopic realization',
            transmission_applied=transmission is not None,
            transmitted_fraction=float(transmitted_fraction),
            instrument_response_already_included=True,
            PSF_applications_here=0, count_support_applications_here=0,
            exposure_applications_here=0,
            metal_D_label='upstream conditional metal-model radial correlation step, never the empirical host peak',
            host_peak_not_metal_D=True,
            metal_size_label='upstream conditional metal primitive; may remain prior dominated under host-template scattering',
            geometry_certified=False, structural_release=False,
            empirical_host_purity='not assumed; chronological candidates need not be zero metal',
            limitations=['Approximate intensity cross term does not identify a host/metal joint morphology',
                         'Same-class TRAIN donor and common q geometry must be enforced by caller',
                         'D_corr can follow the host template and is not an S2 metal centre-distance truth',
                         'No gap, coverage, or three-dimensional arrangement is inferred by this module']))
